# 🎙️ Chatterbox TTS Colab - Easy Voice Cloning & Text-to-Speech

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1o_PnrXpxvAYozOYtnid74eqbHyOD9A45?usp=sharing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Active](https://img.shields.io/badge/Status-Active-brightgreen.svg)](STATUS.md)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![GitHub stars](https://img.shields.io/github/stars/notebook-nexus/chatterbox-tts-colab.svg?style=social&label=Star)](https://github.com/notebook-nexus/chatterbox-tts-colab)

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

For more detailed documentation, see our [USAGE.md](USAGE.md)

---

## 🤝 Contributing

Please see our [Contributing Guide](CONTRIBUTING.md) for details.

---


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

- 📧 Email: ujjwalkrai@gmail.com
- 🐛 Issues: [Repo Issues](https://github.com/notebook-nexus/chatterbox-tts-colab/issues)
- 🔓 Security: [Repo Security](https://github.com/notebook-nexus/chatterbox-tts-colab/security)
- ⛏ Pull Request: [Repo Pull Request](https://github.com/notebook-nexus/chatterbox-tts-colab/pulls)
- 📖 Docs: [Repo Documentation](https://github.com/notebook-nexus/chatterbox-tts-colab/tree/main/docs)
---

## 🔗 Connect

#### 📝 Writing & Blogging
[![Hashnode](https://img.shields.io/badge/Hashnode-2962FF?style=for-the-badge&logo=hashnode&logoColor=white)](https://ukr-projects.hashnode.dev/)
[![Medium](https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/@ukrpurojekuto)

#### 💼 Professional
[![Website](https://img.shields.io/badge/Website-000000?style=for-the-badge&logo=About.me&logoColor=white)](https://ukr-projects.github.io/ukr-projects/)
[![ukr-projects](https://img.shields.io/badge/main-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ukr-projects)
[![cyberx-projects](https://img.shields.io/badge/cybersecurity-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/cyberx-projects)
[![contro-projects](https://img.shields.io/badge/frontend-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/contro-projects)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/u-k-r/ )
[![Main Channel](https://img.shields.io/badge/main-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@ujjwal-krai)

#### 🌐 Social
[![Twitter](https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://x.com/ukr_projects)
[![Instagram](https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://www.instagram.com/ukr_projects)
[![Tech Channel](https://img.shields.io/badge/tech-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@ukr-projects)
[![Telegram](https://img.shields.io/badge/Telegram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white)](https://t.me/ukr_projects)
[![Reddit](https://img.shields.io/badge/Reddit-FF4500?style=for-the-badge&logo=reddit&logoColor=white)](https://www.reddit.com/user/mrujjwalkr)

---

<div align="center">
  Made with ❤️ by <a href="https://github.com/ukr-projects">ukr</a>
</div>

---

