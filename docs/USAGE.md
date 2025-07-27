# Chatterbox TTS Google Colab - Usage Guide

A comprehensive text-to-speech generation and voice cloning solution using Chatterbox TTS in Google Colab.

## 🚀 Quick Start

### 1. Open the Notebook
Click here to open in Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1o_PnrXpxvAYozOYtnid74eqbHyOD9A45?usp=sharing)

### 2. Set Up Runtime
- Go to **Runtime** → **Change runtime type**
- Select **GPU** as Hardware accelerator (recommended)
- Click **Save**

### 3. Run the Cells in Order
Execute each cell sequentially by clicking the play button or pressing `Shift + Enter`.

## 📋 Step-by-Step Instructions

### Cell 1: Installation and Dependencies
- **Run this cell first**
- Installs PyTorch, Chatterbox TTS, and required dependencies
- **Wait for kernel restart** before proceeding

### Cell 2: Verify Installation
- Run after kernel restart
- Confirms all packages are installed correctly
- Shows PyTorch and CUDA information

### Cell 3: Google Drive Setup
- Mounts Google Drive for file storage
- Creates a `Chatterbox` folder in your Drive
- **Click "Connect to Google Drive" when prompted**

### Cell 4: Model Loading
- Loads the Chatterbox TTS model
- Automatically selects GPU or CPU based on availability
- May take 1-2 minutes on first run

### Cell 5: Configure Settings
Choose your voice generation settings:

```python
# Quick presets (uncomment one):
# config.__dict__.update(config.get_preset("neutral"))      # Balanced
# config.__dict__.update(config.get_preset("expressive"))   # Emotional
# config.__dict__.update(config.get_preset("storytelling")) # Narrative
# config.__dict__.update(config.get_preset("audiobook"))    # Clear

# Or customize manually:
config.exaggeration = 0.6  # 0.0-2.0 (emotional intensity)
config.cfg_weight = 0.4     # 0.1-1.0 (adherence to style)
```

### Cell 6: Voice Cloning (Optional)
For voice cloning:
1. Upload a voice sample (10+ seconds, WAV format)
2. Place it in `Google Drive/Chatterbox/voice_sample.wav`
3. Or use the file upload option in the cell

### Cell 7: Text Processing
- Handles text chunking and analysis
- Estimates processing time
- No action required

### Cell 8: Generate Speech
1. **Edit your text** in the `your_text` variable:
```python
your_text = """
Your text here. This can be multiple paragraphs
and will be automatically processed in chunks
for optimal quality.
"""
```
2. Run the cell to generate audio
3. Files are saved to Google Drive and locally

### Cell 9: Play Audio
- Plays the generated audio
- Shows waveform analysis
- Audio controls appear automatically

### Cell 10: Experiment (Optional)
- Tests different parameter combinations
- Generates short samples with various settings
- Useful for finding your preferred voice style

## 🎛️ Parameter Guide

### Exaggeration (0.0 - 2.0)
- **0.3-0.4**: Subtle, calm speech
- **0.5**: Neutral (default)
- **0.6-0.8**: More expressive
- **1.0+**: Very dramatic

### CFG Weight (0.1 - 1.0)
- **0.3**: Faster pacing, less strict
- **0.5**: Balanced (default)
- **0.7+**: Slower, more faithful to reference

## 🎤 Voice Cloning Tips

For best results, your voice sample should:
- Be **10-30 seconds long**
- Use **WAV format**
- Have **clear audio quality**
- Contain **natural speech** (avoid monotone)
- Be recorded in a **quiet environment**
- Match the **speaking style** you want

## 📁 File Locations

- **Google Drive**: `MyDrive/Chatterbox/`
- **Generated audio**: `generated_speech.wav`
- **Voice samples**: `voice_sample.wav`
- **Local copies**: `/content/` (temporary)

## 🔧 Troubleshooting

### Installation Issues
- If installation fails, restart runtime and try again
- For persistent issues, use CPU-only mode

### Memory Errors
- Reduce `max_chunk_words` to 30-50
- Use shorter text segments
- Restart runtime to clear memory

### Audio Quality Issues
- Try different parameter combinations
- Ensure voice sample is high quality
- Use presets like "audiobook" for clarity

### Voice Cloning Not Working
- Check file path and format (WAV recommended)
- Ensure sample is at least 10 seconds
- Verify file is uploaded correctly

## 💡 Usage Tips

1. **Start with presets** before customizing parameters
2. **Keep text chunks under 100 words** for best quality  
3. **Use punctuation** to control pacing and pauses
4. **Experiment with short samples** before long generation
5. **Save your settings** by noting successful parameter combinations

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/notebook-nexus/chatterbox-tts-colab/issues)
- **Repository**: [GitHub Repo](https://github.com/notebook-nexus/chatterbox-tts-colab)
- **Colab Notebook**: [Open in Colab](https://colab.research.google.com/drive/1o_PnrXpxvAYozOYtnid74eqbHyOD9A45?usp=sharing)

## 📄 License

MIT License - See repository for details.