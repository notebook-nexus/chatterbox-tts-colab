"""
Chatterbox TTS Google Colab Script
==================================

A comprehensive script for text-to-speech generation and voice cloning using
Chatterbox TTS in Google Colab environment.

- Author: Ujjwal Nova
- License: MIT
- Repository: https://github.com/UKR-PROJECTS/chatterbox-tts-colab

Features:
- Automatic dependency installation with fallbacks
- Voice cloning from audio samples
- Long text processing with chunking
- Google Drive integration
- Robust error handling
- GPU/CPU automatic detection
"""

import subprocess
import sys
import os
import logging
from pathlib import Path
from typing import Optional, List, Generator
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


class ChatterboxInstaller:
    """Handles the installation of Chatterbox TTS and dependencies."""
    
    def __init__(self):
        self.installation_success = {
            'pytorch': False,
            'chatterbox': False,
            'git_lfs': False
        }
    
    def run_command(self, command: str, description: str = "") -> bool:
        """
        Run a shell command with error handling.
        
        Args:
            command: Shell command to execute
            description: Human-readable description of the command
            
        Returns:
            bool: True if command succeeded, False otherwise
        """
        logger.info(f"Running: {description or command}")
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                logger.warning(f"{description} failed with return code {result.returncode}")
                logger.debug(f"stderr: {result.stderr}")
                logger.debug(f"stdout: {result.stdout}")
                return False
            else:
                logger.info(f"✓ {description} completed successfully")
                return True
                
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out: {description}")
            return False
        except Exception as e:
            logger.error(f"Error running command: {e}")
            return False
    
    def install_pytorch(self) -> bool:
        """Install PyTorch with CUDA support, fallback to CPU version."""
        logger.info("Installing PyTorch...")
        
        # Try CUDA version first
        success = self.run_command(
            "pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir",
            "Installing PyTorch with CUDA support"
        )
        
        if not success:
            logger.info("CUDA installation failed, trying CPU version...")
            success = self.run_command(
                "pip install torch torchaudio --no-cache-dir",
                "Installing PyTorch (CPU fallback)"
            )
        
        self.installation_success['pytorch'] = success
        return success
    
    def install_git_lfs(self) -> bool:
        """Install git-lfs for handling large model files."""
        success = self.run_command(
            "apt update && apt install -y git-lfs",
            "Installing git-lfs"
        )
        self.installation_success['git_lfs'] = success
        return success
    
    def install_chatterbox(self) -> bool:
        """Install Chatterbox TTS with fallback options."""
        logger.info("Installing Chatterbox TTS...")
        
        # Try PyPI first
        success = self.run_command(
            "pip install chatterbox-tts --no-cache-dir",
            "Installing Chatterbox TTS from PyPI"
        )
        
        if not success:
            logger.info("PyPI installation failed, trying GitHub...")
            success = (
                self.run_command(
                    "git clone https://github.com/resemble-ai/chatterbox.git /tmp/chatterbox",
                    "Cloning Chatterbox repository"
                ) and 
                self.run_command(
                    "cd /tmp/chatterbox && pip install -e . --no-cache-dir",
                    "Installing Chatterbox from source"
                )
            )
        
        self.installation_success['chatterbox'] = success
        return success
    
    def install_all(self) -> bool:
        """Install all dependencies."""
        logger.info(f"Python version: {sys.version}")
        
        # Update pip first
        self.run_command("pip install --upgrade pip", "Upgrading pip")
        
        # Install components
        pytorch_ok = self.install_pytorch()
        git_lfs_ok = self.install_git_lfs()
        chatterbox_ok = self.install_chatterbox()
        
        return all([pytorch_ok, chatterbox_ok])  # git-lfs is optional
    
    def verify_installation(self) -> bool:
        """Verify that all components are properly installed."""
        logger.info("Verifying installations...")
        
        try:
            import torch
            logger.info(f"✓ PyTorch version: {torch.__version__}")
            logger.info(f"✓ CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.info(f"✓ CUDA version: {torch.version.cuda}")
                logger.info(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        except ImportError as e:
            logger.error(f"✗ PyTorch import failed: {e}")
            return False
        
        try:
            import torchaudio
            logger.info(f"✓ TorchAudio version: {torchaudio.__version__}")
        except ImportError as e:
            logger.error(f"✗ TorchAudio import failed: {e}")
            return False
        
        try:
            from chatterbox.tts import ChatterboxTTS
            logger.info("✓ Chatterbox TTS imported successfully")
            return True
        except ImportError as e:
            logger.error(f"✗ Chatterbox TTS import failed: {e}")
            return False


class GoogleDriveManager:
    """Manages Google Drive integration for Colab."""
    
    def __init__(self):
        self.drive_path: Optional[str] = None
        self.mounted = False
    
    def setup_drive(self) -> Optional[str]:
        """Setup Google Drive mount and create necessary directories."""
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            
            self.drive_path = '/content/drive/MyDrive/Chatterbox'
            os.makedirs(self.drive_path, exist_ok=True)
            self.mounted = True
            
            logger.info(f"✓ Drive setup complete: {self.drive_path}")
            return self.drive_path
            
        except ImportError:
            logger.warning("Not running in Google Colab, Drive integration disabled")
            return None
        except Exception as e:
            logger.error(f"✗ Drive setup failed: {e}")
            return None


class ChatterboxTTSModel:
    """Wrapper for Chatterbox TTS model with enhanced functionality."""
    
    def __init__(self):
        self.model = None
        self.device = None
        self.sr = None
    
    def load_model(self, max_retries: int = 3) -> bool:
        """
        Load the Chatterbox model with retry logic.
        
        Args:
            max_retries: Maximum number of retry attempts
            
        Returns:
            bool: True if model loaded successfully
        """
        import torch
        
        for attempt in range(max_retries):
            try:
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                logger.info(f"Loading model on device: {self.device} (attempt {attempt + 1})")
                
                from chatterbox.tts import ChatterboxTTS
                
                self.model = ChatterboxTTS.from_pretrained(device=self.device)
                self.sr = self.model.sr
                
                logger.info("✓ Model loaded successfully")
                return True
                
            except Exception as e:
                logger.warning(f"Model loading attempt {attempt + 1} failed: {e}")
                
                if attempt == max_retries - 1:
                    logger.error("All loading attempts failed")
                    return False
                    
                # Try CPU fallback on last attempt
                if attempt == max_retries - 2 and self.device == 'cuda':
                    logger.info("Trying CPU fallback...")
                    self.device = 'cpu'
        
        return False
    
    def generate_speech(
        self, 
        text: str, 
        voice_sample_path: Optional[str] = None,
        exaggeration: float = 0.6,
        cfg_weight: float = 0.5
    ):
        """
        Generate speech from text.
        
        Args:
            text: Input text to synthesize
            voice_sample_path: Path to voice sample for cloning
            exaggeration: Voice exaggeration parameter
            cfg_weight: CFG weight parameter
            
        Returns:
            torch.Tensor: Generated audio waveform
        """
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            if voice_sample_path and os.path.exists(voice_sample_path):
                logger.info("Using voice cloning...")
                return self.model.generate(
                    text=text,
                    audio_prompt_path=voice_sample_path,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight
                )
            else:
                if voice_sample_path:
                    logger.warning(f"Voice sample not found at {voice_sample_path}, using default voice")
                return self.model.generate(text)
                
        except Exception as e:
            logger.error(f"Error generating speech: {e}")
            raise


class TextProcessor:
    """Handles text processing and chunking for long texts."""
    
    @staticmethod
    def split_into_chunks(text: str, max_words: int = 100) -> Generator[str, None, None]:
        """
        Split text into manageable chunks for TTS processing.
        
        Args:
            text: Input text to split
            max_words: Maximum words per chunk
            
        Yields:
            str: Text chunks
        """
        # Clean and normalize text
        text = text.strip()
        if not text:
            return
        
        words = text.split()
        for i in range(0, len(words), max_words):
            chunk = ' '.join(words[i:i + max_words])
            yield chunk
    
    @staticmethod
    def preprocess_text(text: str) -> str:
        """
        Preprocess text for better TTS output.
        
        Args:
            text: Raw input text
            
        Returns:
            str: Preprocessed text
        """
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Add pauses for better speech flow
        text = text.replace('.', '. ')
        text = text.replace(',', ', ')
        text = text.replace(';', '; ')
        text = text.replace(':', ': ')
        
        # Remove excessive spaces created by replacements
        text = ' '.join(text.split())
        
        return text


def main():
    """Main execution function with comprehensive workflow."""
    
    # Configuration
    SAMPLE_TEXT = """
    Welcome to Chatterbox TTS! This is a demonstration of high-quality text-to-speech synthesis 
    with voice cloning capabilities. The system can process long texts by breaking them into 
    manageable chunks, and optionally clone voices from audio samples for personalized speech generation.
    """
    
    CHUNK_SIZE = 50  # words per chunk
    
    logger.info("🎙️ Starting Chatterbox TTS Pipeline")
    
    # Step 1: Install dependencies
    installer = ChatterboxInstaller()
    if not installer.install_all():
        logger.error("❌ Installation failed. Please check the logs and try again.")
        return
    
    if not installer.verify_installation():
        logger.error("❌ Installation verification failed.")
        return
    
    # Step 2: Setup Google Drive
    drive_manager = GoogleDriveManager()
    drive_path = drive_manager.setup_drive()
    
    # Step 3: Load TTS model
    tts_model = ChatterboxTTSModel()
    if not tts_model.load_model():
        logger.error("❌ Failed to load TTS model.")
        return
    
    # Step 4: Process text and generate speech
    try:
        import torch
        import torchaudio
        
        # Preprocess text
        processed_text = TextProcessor.preprocess_text(SAMPLE_TEXT)
        chunks = list(TextProcessor.split_into_chunks(processed_text, CHUNK_SIZE))
        
        logger.info(f"Processing {len(chunks)} text chunks...")
        
        wav_tensors = []
        voice_sample_path = None
        
        if drive_path:
            voice_sample_path = os.path.join(drive_path, "my_voice_sample.wav")
        
        # Generate speech for each chunk
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}: '{chunk[:50]}...'")
            
            try:
                wav = tts_model.generate_speech(
                    text=chunk,
                    voice_sample_path=voice_sample_path,
                    exaggeration=0.6,
                    cfg_weight=0.5
                )
                wav_tensors.append(wav)
                
                # Clear GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {e}")
                continue
        
        # Save final audio
        if wav_tensors:
            full_audio = torch.cat(wav_tensors, dim=1)
            
            if drive_path:
                output_file = os.path.join(drive_path, "generated_speech.wav")
                torchaudio.save(output_file, full_audio, tts_model.sr)
                logger.info(f"✅ Audio saved to: {output_file}")
            else:
                logger.info("✅ Audio generated successfully (no save path available)")
                
            logger.info(f"📊 Generated {full_audio.shape[1] / tts_model.sr:.2f} seconds of audio")
        else:
            logger.warning("⚠️ No audio was generated")
            
    except Exception as e:
        logger.error(f"❌ Error during speech generation: {e}")
        raise


if __name__ == "__main__":
    main()