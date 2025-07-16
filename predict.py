# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import sys
import copy
import torch
from typing import Optional
from cog import BasePredictor, Input, Path
from huggingface_hub import snapshot_download
from peft import PeftModel

# Set up model cache and environment variables
MODEL_CACHE = "model_cache"
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE

# Critical: Set CUDA_HOME for DeepSpeed (equivalent to CUDA_HOME=$CONDA_PREFIX from our working setup)
os.environ["CUDA_HOME"] = "/usr/local/cuda"

# Add local llava module to Python path (equivalent to pip install -e llava)
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import llava


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        print("[+] Starting model setup...")
        
        # Create model cache directory if it doesn't exist
        if not os.path.exists(MODEL_CACHE):
            os.makedirs(MODEL_CACHE)
            print(f"[+] Created model cache directory: {MODEL_CACHE}")

        # Download and load models - exact copy from updated app.py
        print("[+] Downloading single-turn model...")
        self.MODEL_BASE_SINGLE = snapshot_download(repo_id="nvidia/audio-flamingo-3")
        self.MODEL_BASE_THINK = os.path.join(self.MODEL_BASE_SINGLE, 'stage35')
        print(f"[+] Single-turn model downloaded to: {self.MODEL_BASE_SINGLE}")
        
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
        mode: str = Input(
            description="Analysis mode: 'standard' for quick responses, 'reasoning' for detailed chain-of-thought analysis",
            choices=["standard", "reasoning"],
            default="standard"
        ),
    ) -> str:
        """Analyze audio using Audio Flamingo 3 - supports speech, music, and sound analysis up to 10 minutes"""
        
        # Create sound object
        sound = llava.Sound(str(audio))
        
        if mode == "reasoning":
            # Use thinking model for detailed analysis
            full_prompt = f"<sound>\n{prompt}"
            response = self.model_think.generate_content(
                [sound, full_prompt], 
                generation_config=self.generation_config_single
            )
        else:
            # Use standard model for faster responses
            full_prompt = f"<sound>\n{prompt}"
            response = self.model_single_copy.generate_content(
                [sound, full_prompt], 
                generation_config=self.generation_config_single
            )
        
        return response