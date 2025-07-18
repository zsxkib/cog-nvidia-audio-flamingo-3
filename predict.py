# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import sys
import copy
import torch
import time
import subprocess
from typing import Optional
from cog import BasePredictor, Input, Path
from peft import PeftModel

# Set up model cache
MODEL_CACHE = "model_cache"

# Critical: Set CUDA_HOME for DeepSpeed (equivalent to CUDA_HOME=$CONDA_PREFIX from our working setup)
os.environ["CUDA_HOME"] = "/usr/local/cuda"

# Add local llava module to Python path (equivalent to pip install -e llava)
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import llava

BASE_URL = "https://weights.replicate.delivery/default/audio-flamingo-3/model_cache/"

def download_weights(url: str, dest: str) -> None:
    start = time.time()
    print("[!] Initiating download from URL: ", url)
    print("[~] Destination path: ", dest)
    if ".tar" in dest:
        dest = os.path.dirname(dest)
    command = ["pget", "-vf" + ("x" if ".tar" in url else ""), url, dest]
    try:
        print(f"[~] Running command: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to download weights. Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        )
        raise
    print("[+] Download completed in: ", time.time() - start, "seconds")


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Create model cache directory if it doesn't exist
        if not os.path.exists(MODEL_CACHE):
            os.makedirs(MODEL_CACHE)
            
        # Set environment variables for model caching
        os.environ["HF_HOME"] = MODEL_CACHE
        os.environ["TORCH_HOME"] = MODEL_CACHE
        os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
        os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
        os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE

        model_files = [
            ".locks.tar",
            "models--Qwen--Qwen2-Audio-7B.tar",
            "models--nvidia--audio-flamingo-3-chat.tar",
            "models--nvidia--audio-flamingo-3.tar",
            "version.txt",
        ]

        for model_file in model_files:
            url = BASE_URL + model_file
            filename = url.split("/")[-1]
            dest_path = os.path.join(MODEL_CACHE, filename)
            if not os.path.exists(dest_path.replace(".tar", "")):
                download_weights(url, dest_path)

        # Set up model paths - same as before, just using cached models
        self.MODEL_BASE_SINGLE = os.path.join(MODEL_CACHE, "models--nvidia--audio-flamingo-3", "snapshots", "504150e751238e1471971f8bef3303e32b5fd23d")
        self.MODEL_BASE_THINK = os.path.join(self.MODEL_BASE_SINGLE, 'stage35')
        print(f"[+] Model paths set: {self.MODEL_BASE_SINGLE}")
        
        print("[+] Loading single-turn model...")
        self.model_single = llava.load(self.MODEL_BASE_SINGLE, model_base=None)
        self.model_single = self.model_single.to("cuda")
        self.model_single_copy = copy.deepcopy(self.model_single)  # Thread-safe copy
        self.generation_config_single = self.model_single.default_generation_config
        print("[+] Single-turn model loaded successfully")
        
        print("[+] Loading think model...")
        self.model_think = PeftModel.from_pretrained(
            self.model_single,
            self.MODEL_BASE_THINK,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        print("[+] Think model loaded successfully")
        print("[+] Model setup complete!")

    def predict(
        self,
        audio: Path = Input(
            description="Audio file to analyze. Supports speech, music, and sound effects. Maximum duration: 10 minutes."
        ),
        prompt: str = Input(
            description="Question or instruction about the audio",
            default="Please describe this audio in detail."
        ),
        system_prompt: str = Input(
            description="System instructions to customize the model's behavior, output format, or analysis style. Leave empty for default behavior.",
            default=""
        ),
        enable_thinking: bool = Input(
            description="Enable detailed chain-of-thought reasoning for complex analysis. False for faster responses, True for deeper insights.",
            default=False
        ),
        temperature: float = Input(
            description="Controls response creativity and randomness. Use 0.0 for deterministic (default), 0.1-0.3 for factual analysis, 0.7-0.9 for creative interpretation.",
            default=0.0,
            ge=0.0,
            le=1.0
        ),
        max_length: int = Input(
            description="Maximum length of the response in tokens. Use 0 for model default, or specify 50-2048 for custom length.",
            default=0,
            ge=0,
            le=2048
        ),
        start_time: Optional[float] = Input(
            description="Start time in seconds for audio segment analysis (optional). Useful for long audio files.",
            default=None
        ),
        end_time: Optional[float] = Input(
            description="End time in seconds for audio segment analysis (optional). Must be greater than start_time.",
            default=None
        ),
    ) -> str:
        """Analyze audio using Audio Flamingo 3 - supports speech, music, and sound analysis up to 10 minutes"""
        
        # Validate audio segment timing
        if start_time is not None and start_time < 0:
            raise ValueError("start_time must be non-negative")
        if end_time is not None and end_time < 0:
            raise ValueError("end_time must be non-negative") 
        if start_time is not None and end_time is not None:
            if end_time <= start_time:
                raise ValueError("end_time must be greater than start_time")
        
        # Create sound object with optional segment timing
        if start_time is not None or end_time is not None:
            # Load audio and apply segment timing
            import librosa
            y, sr = librosa.load(str(audio))
            
            start_sample = int(start_time * sr) if start_time is not None else 0
            end_sample = int(end_time * sr) if end_time is not None else len(y)
            
            # Create temporary file for segmented audio
            import tempfile
            import soundfile as sf
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                sf.write(tmp_file.name, y[start_sample:end_sample], sr)
                sound = llava.Sound(tmp_file.name)
        else:
            sound = llava.Sound(str(audio))
        
        # Prepare generation config - start with model defaults to match app.py behavior
        generation_config = copy.deepcopy(self.generation_config_single)
        
        # Only modify generation config if user explicitly changed defaults
        if max_length > 0:
            generation_config.max_new_tokens = max_length
        
        # Handle temperature and sampling - match app.py default behavior
        if temperature > 0:
            generation_config.temperature = temperature
            generation_config.do_sample = True
        # If temperature is 0, use model defaults (likely greedy/deterministic)
        # Don't explicitly set do_sample=False as this might override model defaults
        
        # Construct prompt - match app.py format by default
        if system_prompt.strip():
            # Only add system prompt format when explicitly provided
            full_prompt = f"<sound>\n{system_prompt.strip()}\n\n{prompt}"
        else:
            # Use exact same format as app.py for default behavior
            full_prompt = f"<sound>\n{prompt}"
        
        if enable_thinking:
            # Use thinking model for detailed analysis
            response = self.model_think.generate_content(
                [sound, full_prompt], 
                generation_config=generation_config
            )
        else:
            # Use standard model for faster responses - match app.py
            response = self.model_single_copy.generate_content(
                [sound, full_prompt], 
                generation_config=generation_config
            )
        
        return response