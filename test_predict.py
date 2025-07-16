#!/usr/bin/env python3

import sys
import os
from pathlib import Path
sys.path.insert(0, '.')

from predict import Predictor

def test_predict():
    """Test the predict functionality matching app.py patterns"""
    print("=== Testing Audio Flamingo 3 Predict ===")
    
    # Initialize predictor
    predictor = Predictor()
    
    # Run setup
    print("\n1. Running setup...")
    predictor.setup()
    
    # Find available audio files
    audio_files = ["./static/voice/voice_0.mp3", "./static/voice/voice_1.mp3", "./static/voice/voice_2.mp3"]
    test_audio = None
    for audio in audio_files:
        if os.path.exists(audio):
            test_audio = audio
            break
    
    if not test_audio:
        print("No audio files found for testing")
        return
    
    print(f"Using test audio: {test_audio}")
    
    # Test single-turn inference (matches app.py single_turn_infer)
    print("\n2. Testing single-turn inference...")
    prompt = "What can you hear in this audio?"
    
    result = predictor.predict(
        audio=Path(test_audio),
        prompt=prompt,
        enable_thinking=False
    )
    print(f"Single-turn result: {result}")
    
    # Test thinking mode (matches app.py think_infer)
    print("\n3. Testing thinking mode...")
    result_think = predictor.predict(
        audio=Path(test_audio),
        prompt=prompt,
        enable_thinking=True
    )
    print(f"Think mode result: {result_think}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_predict()