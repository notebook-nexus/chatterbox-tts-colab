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

