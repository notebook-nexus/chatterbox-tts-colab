# 🎙️ Chatterbox TTS Colab - Easy Voice Cloning & Text-to-Speech

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1o_PnrXpxvAYozOYtnid74eqbHyOD9A45?usp=sharing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![GitHub stars](https://img.shields.io/github/stars/UKR-PROJECTS/chatterbox-tts-colab.svg?style=social&label=Star)](https://github.com/UKR-PROJECTS/chatterbox-tts-colab)

> 🚀 **One-click voice cloning and text-to-speech in Google Colab with Chatterbox TTS**

Transform any text into natural-sounding speech, clone voices from audio samples, and create professional voiceovers - all running free in Google Colab!

## 🚀 Quick Start
1. Click the "Open in Colab" button above
2. Run all cell in the notebook
3. Upload your voice sample (optional)
4. Enter your text and generate speech!

## ✨ Features
- 🎯 **Zero Setup**: Run immediately in Google Colab
- 🗣️ **Voice Cloning**: Clone any voice from a short audio sample
- 🌍 **Multilingual**: Support for multiple languages
- 🎛️ **Advanced Controls**: Fine-tune voice characteristics
- 💾 **Google Drive Integration**: Automatic saving to your drive
- 🔧 **Robust Error Handling**: Graceful fallbacks and clear error messages

## 🔊 Demo: Text & Audio Samples

Here’s a quick demo so you can see—and hear—how Chatterbox-TTS-Colab performs.

---

### 📝 Sample Text
> “This is a test of the Chatterbox TTS system. I hope this works properly now with the improved error handling and correct repository. The model should now load from ResembleAI/chatterbox instead of the old fluffyox repository.”  

---

### 🎤 Original Voice Clip (for cloning)

https://github.com/user-attachments/assets/b34c7eb1-8fda-46c9-a62f-d94318d9f12a

---

### 🤖 AI-Generated TTS Output

https://github.com/user-attachments/assets/7ff42492-8928-41af-8d9a-d5e952566cbe

---

## 📦 Installation

The Colab notebook handles all installations automatically. If you want to run locally:

```bash
# Install required packages
pip install chatterbox-tts
pip install torch torchaudio
pip install gradio
pip install librosa soundfile

# For Google Drive integration
pip install google-colab-tools
```

## 🎯 Usage

### Basic Text-to-Speech

```python
from chatterbox.tts import ChatterboxTTS
import torchaudio as ta

# Initialize the model
model = ChatterboxTTS.from_pretrained(device="cuda")

# Generate speech from text
text = "Hello world! This is Chatterbox TTS in action."
wav = model.generate(text)

# Save the audio
ta.save("output.wav", wav, model.sr)
```

### Voice Cloning

```python
# Clone a voice using reference audio
AUDIO_PROMPT_PATH = "path/to/your/reference_audio.wav"
text = "This text will be spoken in the cloned voice."

wav = model.generate(
    text, 
    audio_prompt_path=AUDIO_PROMPT_PATH,
    exaggeration=0.5,  # Emotion intensity (0.0-1.0)
    cfg=0.5           # Classifier-free guidance (0.0-1.0)
)

ta.save("cloned_voice_output.wav", wav, model.sr)
```

### Batch Processing

```python
# Process multiple texts
texts = [
    "First sentence to synthesize.",
    "Second sentence with different content.",
    "Third sentence for batch processing."
]

for i, text in enumerate(texts):
    wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH)
    ta.save(f"batch_output_{i}.wav", wav, model.sr)
```

## 🎛️ Advanced Controls

### Emotion and Intensity Control

Chatterbox TTS offers unique emotion exaggeration control:

```python
# Subtle, natural speech
wav = model.generate(text, exaggeration=0.3, cfg=0.5)

# More dramatic, expressive speech
wav = model.generate(text, exaggeration=0.8, cfg=0.3)

# Highly exaggerated, theatrical speech
wav = model.generate(text, exaggeration=1.0, cfg=0.2)
```

### Parameter Guide

| Parameter | Range | Description | Recommended Use |
|-----------|-------|-------------|-----------------|
| `exaggeration` | 0.0-1.0 | Controls emotional intensity and expressiveness | 0.5 for natural speech, 0.7+ for dramatic |
| `cfg` | 0.0-1.0 | Classifier-free guidance for speech pacing | 0.5 for normal, 0.3 for slower pacing |
| `temperature` | 0.1-2.0 | Controls randomness in generation | 0.7 for balanced, 1.0+ for more variation |
| `top_p` | 0.1-1.0 | Nucleus sampling parameter | 0.9 for most cases |

### Audio Quality Settings

```python
# High quality (slower generation)
wav = model.generate(
    text,
    audio_prompt_path=AUDIO_PROMPT_PATH,
    exaggeration=0.5,
    cfg=0.5,
    temperature=0.7,
    top_p=0.9,
    steps=30 
)

# Fast generation (lower quality)
wav = model.generate(
    text,
    audio_prompt_path=AUDIO_PROMPT_PATH,
    steps=15  # Fewer steps = faster generation
)
```

## 🎤 Voice Cloning Guide

### Preparing Reference Audio

For best voice cloning results:

1. **Audio Quality**: Use clear, high-quality audio (WAV or MP3)
2. **Duration**: 3-30 seconds of speech is optimal
3. **Content**: Choose audio with clear pronunciation
4. **Background**: Minimal background noise
5. **Format**: Supported formats: WAV, MP3, FLAC, M4A

### Voice Cloning Tips

```python
# For different speaker types:

# Fast-speaking reference
wav = model.generate(text, audio_prompt_path=path, cfg=0.3, exaggeration=0.5)

# Slow, deliberate speaker
wav = model.generate(text, audio_prompt_path=path, cfg=0.7, exaggeration=0.4)

# Emotional, expressive speaker
wav = model.generate(text, audio_prompt_path=path, cfg=0.3, exaggeration=0.8)

# Professional, neutral speaker
wav = model.generate(text, audio_prompt_path=path, cfg=0.5, exaggeration=0.3)
```

### Audio Preprocessing

```python
import librosa
import soundfile as sf

def preprocess_audio(input_path, output_path):
    """Preprocess audio for better voice cloning"""
    # Load audio
    audio, sr = librosa.load(input_path, sr=22050)
    
    # Normalize volume
    audio = librosa.util.normalize(audio)
    
    # Remove silence
    audio, _ = librosa.effects.trim(audio, top_db=20)
    
    # Save preprocessed audio
    sf.write(output_path, audio, sr)
    return output_path

# Use preprocessed audio for cloning
processed_audio = preprocess_audio("raw_audio.wav", "processed_audio.wav")
wav = model.generate(text, audio_prompt_path=processed_audio)
```

## 💾 Google Drive Integration

### Automatic Saving

```python
from google.colab import drive
import os

# Mount Google Drive
drive.mount('/content/drive')

# Set up directories
output_dir = '/content/drive/MyDrive/ChatterboxTTS_Outputs'
os.makedirs(output_dir, exist_ok=True)

# Save with timestamp
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f"{output_dir}/tts_output_{timestamp}.wav"

wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH)
ta.save(output_path, wav, model.sr)
print(f"Audio saved to: {output_path}")
```

### Batch Processing with Drive

```python
# Process multiple files from Drive
input_dir = '/content/drive/MyDrive/ChatterboxTTS_Inputs'
output_dir = '/content/drive/MyDrive/ChatterboxTTS_Outputs'

# Read text files
for filename in os.listdir(input_dir):
    if filename.endswith('.txt'):
        with open(os.path.join(input_dir, filename), 'r') as f:
            text = f.read()
        
        wav = model.generate(text)
        output_path = os.path.join(output_dir, f"{filename[:-4]}.wav")
        ta.save(output_path, wav, model.sr)
```

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. CUDA Out of Memory Error

```python
# Solution: Clear cache and reduce batch size
import torch
torch.cuda.empty_cache()

# Use smaller text chunks
def split_text(text, max_length=200):
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk + sentence) < max_length:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

# Process in chunks
text_chunks = split_text(long_text)
audio_chunks = []

for chunk in text_chunks:
    wav = model.generate(chunk, audio_prompt_path=AUDIO_PROMPT_PATH)
    audio_chunks.append(wav)

# Concatenate chunks
final_audio = torch.cat(audio_chunks, dim=-1)
ta.save("long_text_output.wav", final_audio, model.sr)
```

#### 2. Audio Quality Issues

```python
# Solution: Adjust generation parameters
wav = model.generate(
    text,
    audio_prompt_path=AUDIO_PROMPT_PATH,
    exaggeration=0.4,  # Lower for more natural speech
    cfg=0.6,          # Higher for more controlled output
    temperature=0.6,   # Lower for more consistent quality
    steps=25         
)
```

#### 3. Voice Cloning Not Working

```python
# Check audio file format and quality
import librosa
import numpy as np

def check_audio_quality(audio_path):
    try:
        audio, sr = librosa.load(audio_path)
        duration = len(audio) / sr
        
        print(f"Audio duration: {duration:.2f} seconds")
        print(f"Sample rate: {sr} Hz")
        print(f"Audio shape: {audio.shape}")
        
        # Check for silence
        silence_threshold = 0.01
        non_silent_ratio = np.mean(np.abs(audio) > silence_threshold)
        print(f"Non-silent ratio: {non_silent_ratio:.2f}")
        
        if duration < 3:
            print("⚠️  Audio might be too short for good cloning")
        if non_silent_ratio < 0.5:
            print("⚠️  Audio might have too much silence")
        
        return True
    except Exception as e:
        print(f"❌ Error loading audio: {e}")
        return False

# Check your reference audio
check_audio_quality("your_reference_audio.wav")
```

#### 4. Slow Generation Speed

```python
# Optimization tips
import gc

def optimize_generation():
    # Clear memory
    torch.cuda.empty_cache()
    gc.collect()
    
    # Use mixed precision
    with torch.cuda.amp.autocast():
        wav = model.generate(
            text,
            audio_prompt_path=AUDIO_PROMPT_PATH,
            steps=15,  # Reduce steps for speed
            cfg=0.5
        )
    
    return wav
```

#### 5. Google Drive Mount Issues

```python
# Force remount Drive
from google.colab import drive
drive.flush_and_unmount()
drive.mount('/content/drive', force_remount=True)

# Check permissions
import os
test_path = '/content/drive/MyDrive/test_file.txt'
try:
    with open(test_path, 'w') as f:
        f.write('test')
    os.remove(test_path)
    print("✅ Drive access working")
except Exception as e:
    print(f"❌ Drive access issue: {e}")
```

### Error Messages and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `RuntimeError: CUDA out of memory` | GPU memory exhausted | Clear cache, reduce text length, restart runtime |
| `FileNotFoundError` | Audio file path incorrect | Check file path, ensure file exists |
| `ValueError: Invalid audio format` | Unsupported audio format | Convert to WAV/MP3, check file integrity |
| `ModuleNotFoundError` | Missing dependencies | Run installation cell again |
| `ConnectionError` | Network issues | Check internet connection, restart runtime |

```
## 🎨 Custom Voice Effects

### Emotion Presets

```python
# Define emotion presets
EMOTION_PRESETS = {
    'neutral': {'exaggeration': 0.3, 'cfg': 0.5, 'temperature': 0.7},
    'happy': {'exaggeration': 0.8, 'cfg': 0.4, 'temperature': 0.8},
    'sad': {'exaggeration': 0.6, 'cfg': 0.6, 'temperature': 0.6},
    'angry': {'exaggeration': 0.9, 'cfg': 0.3, 'temperature': 0.9},
    'calm': {'exaggeration': 0.2, 'cfg': 0.7, 'temperature': 0.5},
    'excited': {'exaggeration': 1.0, 'cfg': 0.3, 'temperature': 1.0},
    'whisper': {'exaggeration': 0.1, 'cfg': 0.8, 'temperature': 0.4}
}

def generate_with_emotion(text, voice_file, emotion='neutral'):
    """Generate speech with specific emotion"""
    params = EMOTION_PRESETS.get(emotion, EMOTION_PRESETS['neutral'])
    
    wav = model.generate(
        text,
        audio_prompt_path=voice_file,
        **params
    )
    
    return wav

# Usage
text = "I can't believe this is happening!"
emotions = ['happy', 'sad', 'angry', 'excited']

for emotion in emotions:
    wav = generate_with_emotion(text, voice_file, emotion)
    ta.save(f"emotion_{emotion}.wav", wav, model.sr)
```

## 🔒 Security and Privacy

### Data Protection

```python
import tempfile
import os

def secure_audio_processing(audio_data, output_path):
    """Process audio with temporary files for security"""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_path = temp_file.name
        
        try:
            # Save to temporary file
            ta.save(temp_path, audio_data, model.sr)
            
            # Process and move to final location
            shutil.move(temp_path, output_path)
            
        finally:
            # Clean up temporary file if it still exists
            if os.path.exists(temp_path):
                os.remove(temp_path)
```

### Watermark Detection

```python
def detect_watermark(audio_path):
    """Check if audio contains Chatterbox watermark"""
    try:
        # This is a placeholder - actual watermark detection
        # would require Resemble AI's Perth watermark detector
        print("⚠️  All Chatterbox-generated audio contains watermarks")
        print("   Use responsibly and follow ethical guidelines")
        return True
    except Exception as e:
        print(f"Error checking watermark: {e}")
        return False
```

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Report Bugs**: Use the GitHub Issues tab
2. **Feature Requests**: Suggest new features via Issues
3. **Code Contributions**: Fork the repo and submit PRs
4. **Documentation**: Help improve this README and docs
5. **Examples**: Share your creative use cases

## 🙏 Acknowledgments

- **Resemble AI** for creating the incredible Chatterbox TTS model
- **Google Colab** for providing free GPU access
- **Hugging Face** for model hosting and distribution
- **PyTorch** and **Torchaudio** for the underlying framework
- **The Open Source Community** for continuous support and contributions

### Special Thanks

- Original Chatterbox TTS: [resemble-ai/chatterbox](https://github.com/resemble-ai/chatterbox)
- Resemble AI Team for open-sourcing this state-of-the-art model
- Contributors who help maintain and improve this Colab implementation

## 📞 Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/UKR-PROJECTS/chatterbox-tts-colab/issues)
- **Discussions**: [Community discussions and Q&A](https://github.com/UKR-PROJECTS/chatterbox-tts-colab/discussions)
- **Email**: ujjwalkrai@gmail.com

---

<div align="center">

**Made with ❤️ by the Ujjwal Nova**

[⭐ Star this repo](https://github.com/UKR-PROJECTS/chatterbox-tts-colab) | [🐛 Report Bug](https://github.com/UKR-PROJECTS/chatterbox-tts-colab/issues) | [💡 Request Feature](https://github.com/UKR-PROJECTS/chatterbox-tts-colab/issues)

</div>

