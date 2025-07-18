# Audio Flamingo 3

You know how most audio AI is either "transcribe this speech" or "this is probably music"? Audio Flamingo 3 actually understands audio. 

Ask it what's happening in a 10-minute podcast and it'll tell you. Play it a song and it can analyze the emotional progression. Give it a complex soundscape with multiple people talking and background noise, and it figures it out.

## Try it right now

Got a GPU and Docker? Three commands and you're analyzing audio:

```bash
git clone https://github.com/zsxkib/cog-nvidia-audio-flamingo-3.git
cd cog-nvidia-audio-flamingo-3
sudo cog predict -i audio=@static/audio/audio2.wav -i prompt="What type of audio is this?" -i enable_thinking=true
```

That's it. No setup, no config files, no hunting for model weights. It downloads everything automatically and starts working.

## What makes this different

Most audio models do one thing. Audio Flamingo 3 understands audio like a human would.

Turn on "thinking mode" and watch it reason through what it's hearing step by step. It breaks down complex audio, makes connections, and builds up to insights. 

It handles up to 10 minutes of audio in one go. Perfect for podcasts, interviews, music analysis, or any audio where context matters.

## Some things you can try

```bash
# Describe any audio file
sudo cog predict -i audio=@my_song.wav -i prompt="What's happening in this audio?"

# Transcribe speech
sudo cog predict -i audio=@interview.wav -i prompt="Transcribe this"

# Analyze music with step-by-step reasoning
sudo cog predict -i audio=@track.wav -i prompt="Analyze the musical structure" -i enable_thinking=true

# Just analyze part of a long file
sudo cog predict -i audio=@podcast.wav -i prompt="What are they discussing?" -i start_time=120 -i end_time=300
```

## All the parameters

- `audio` - Your audio file
- `prompt` - What you want to know about it
- `enable_thinking` - Set to `true` to see step-by-step reasoning
- `system_prompt` - Custom instructions for output format
- `temperature` - How creative the response is (0.0-1.0)
- `max_length` - Response length (50-2048 tokens)
- `start_time` / `end_time` - Analyze specific segments

## What you need

- NVIDIA GPU (tested on A100, but others work)
- Docker
- Cog (`pip install cog`)

## Use cases

Content creators analyzing podcasts and interviews. Musicians getting feedback on compositions. Researchers working with audio data. Anyone who needs to understand what's in audio files beyond basic transcription.

## How it works

Audio Flamingo 3 uses a unified encoder that treats speech, music, and sound effects the same way. It connects to a 7 billion parameter language model that can reason about what it hears.

The thinking mode runs chain-of-thought reasoning—like having an audio expert walk through their analysis out loud.

## Performance

It beats other models on audio understanding benchmarks. But benchmarks are just numbers. Try it on your audio and see if it's useful.

## If you want to run the web demo

```bash
# Set up conda environment
conda create -n audio-flamingo python=3.10 -y
conda activate audio-flamingo

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install everything else
pip install gradio peft huggingface_hub hydra-core loguru soundfile librosa openai-whisper ftfy jiwer einops wandb kaldiio matplotlib opencv-python-headless pydub pytorchvideo==0.1.5

# Install CUDA toolkit and DeepSpeed
conda install nvidia/label/cuda-12.4.0::cuda-toolkit -y
export CUDA_HOME=$CONDA_PREFIX
pip install deepspeed==0.15.4 transformers==4.46.0

# Run it
CUDA_HOME=$CONDA_PREFIX python app.py
```

## Code

- `predict.py` - Main Cog interface
- `app.py` - Gradio web demo  
- `llava/` - Model code
- `static/` - Example audio files

## License

Code is MIT. Model weights are non-commercial use only. Built on Qwen2.5-7B.

## Citation

```bibtex
@article{ghosh2025audio,
  title={Audio Flamingo 2: An Audio-Language Model with Long-Audio Understanding and Expert Reasoning Abilities},
  author={Ghosh, Sreyan and Kong, Zhifeng and Kumar, Sonal and Sakshi, S and Kim, Jaehyeon and Ping, Wei and Valle, Rafael and Manocha, Dinesh and Catanzaro, Bryan},
  journal={arXiv preprint arXiv:2503.03983},
  year={2025}
}
```

---

Made by NVIDIA. Packaged for Replicate by [@zsxkib](https://twitter.com/zsxkib).

[![Replicate](https://replicate.com/zsxkib/audio-flamingo-3/badge)](https://replicate.com/zsxkib/audio-flamingo-3)