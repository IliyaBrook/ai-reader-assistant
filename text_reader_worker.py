#!/usr/bin/env python3
"""
Text Reader Worker - Heavy subprocess that loads models and processes text.

This process:
1. Loads translation models (MarianMT) on first request
2. Loads TTS models (Piper, Silero, MMS) on first request
3. Listens for requests via Unix socket
4. Translates and speaks text
5. Exits after timeout, fully freeing all RAM
"""

import os
import sys
import json
import socket
import signal
import subprocess
import tempfile
import threading
import time
import logging
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

WORKER_TIMEOUT = 60  # Exit after this many seconds of inactivity
SOCKET_PATH = "/tmp/text_reader_worker.sock"

# Translation Models (MarianMT)
TRANSLATION_MODELS = {
    ("en", "ru"): "Helsinki-NLP/opus-mt-en-ru",
    ("ru", "en"): "Helsinki-NLP/opus-mt-ru-en",
    ("en", "he"): "Helsinki-NLP/opus-mt-en-he",
    ("he", "en"): "Helsinki-NLP/opus-mt-tc-big-he-en",
}

TRANSLATION_ENABLED = True

# TTS Settings
PIPER_MODELS_DIR = Path(__file__).parent / "models" / "piper"
PIPER_VOICE_EN = "en_US-amy-medium"
SILERO_MODEL_ID = "v4_ru"
SILERO_SPEAKER = "xenia"
SILERO_SAMPLE_RATE = 48000
MMS_HEBREW_MODEL = "facebook/mms-tts-heb"

# Audio
AUDIO_PLAYER = "ffplay"
AUDIO_PLAYER_ARGS = ["-nodisp", "-autoexit", "-loglevel", "quiet"]

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ============================================================================
# WORKER IMPLEMENTATION
# ============================================================================

class TextReaderWorker:
    """Worker that loads models and handles text processing requests."""
    
    def __init__(self):
        logger.info(f"🔧 Worker starting (timeout: {WORKER_TIMEOUT}s)...")
        
        # TTS models
        self.piper_voice_en = None
        self.silero_model = None
        self.silero_device = None
        self.mms_model = None
        self.mms_tokenizer = None
        
        # Translation models
        self.translation_models = {}
        
        self.models_loaded = False
        self.last_activity = time.time()
        self.running = True
        self.server_socket = None
        self.audio_process = None
        
        # Start timeout monitor
        self.timeout_thread = threading.Thread(target=self._monitor_timeout, daemon=True)
        self.timeout_thread.start()
    
    def _load_all_models(self):
        """Load all models."""
        if self.models_loaded:
            return
        
        logger.info("🚀 Loading all models...")
        
        self._load_piper_english()
        self._load_silero_russian()
        self._load_mms_hebrew()
        self._load_translation_models()
        
        self.models_loaded = True
        logger.info("✅ All models loaded!")
    
    def _load_piper_english(self):
        """Load Piper voice model for English."""
        try:
            # noinspection PyUnresolvedReferences
            from piper import PiperVoice
            
            model_path = PIPER_MODELS_DIR / f"{PIPER_VOICE_EN}.onnx"
            config_path = PIPER_MODELS_DIR / f"{PIPER_VOICE_EN}.onnx.json"
            
            if model_path.exists() and config_path.exists():
                self.piper_voice_en = PiperVoice.load(str(model_path), str(config_path))
                logger.info(f"   Loaded Piper voice: {PIPER_VOICE_EN}")
            else:
                logger.warning(f"   Piper model not found: {model_path}")
        except Exception as e:
            logger.error(f"   Failed to load Piper: {e}")
    
    def _load_silero_russian(self):
        """Load Silero TTS for Russian."""
        try:
            import torch
            
            self.silero_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-models',
                model='silero_tts',
                language='ru',
                speaker=SILERO_MODEL_ID
            )
            model.to(self.silero_device)
            self.silero_model = model
            
            logger.info(f"   Loaded Silero Russian on {self.silero_device}")
        except Exception as e:
            logger.error(f"   Failed to load Silero: {e}")
    
    def _load_mms_hebrew(self):
        """Load Facebook MMS-TTS for Hebrew."""
        try:
            import torch
            # noinspection PyUnresolvedReferences
            from transformers import VitsModel, AutoTokenizer
            
            self.mms_model = VitsModel.from_pretrained(MMS_HEBREW_MODEL)
            self.mms_tokenizer = AutoTokenizer.from_pretrained(MMS_HEBREW_MODEL)
            
            if torch.cuda.is_available():
                self.mms_model = self.mms_model.to("cuda")
                logger.info("   Loaded MMS Hebrew on GPU")
            else:
                logger.info("   Loaded MMS Hebrew on CPU")
        except Exception as e:
            logger.error(f"   Failed to load MMS Hebrew: {e}")
    
    def _load_translation_models(self):
        """Load MarianMT translation models."""
        if not TRANSLATION_ENABLED:
            return
        
        try:
            # noinspection PyUnresolvedReferences
            from transformers import MarianMTModel, MarianTokenizer
            
            logger.info("   Loading translation models...")
            
            for (src, tgt), model_name in TRANSLATION_MODELS.items():
                try:
                    tokenizer = MarianTokenizer.from_pretrained(model_name)
                    model = MarianMTModel.from_pretrained(model_name)
                    self.translation_models[(src, tgt)] = (model, tokenizer)
                    logger.info(f"     Loaded {src}->{tgt}")
                except Exception as e:
                    logger.error(f"     Failed {src}->{tgt}: {e}")
            
            logger.info(f"   Translation models: {len(self.translation_models)} pairs")
        except Exception as e:
            logger.error(f"   Failed to load translation models: {e}")
    
    def _translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text using MarianMT."""
        if source_lang == target_lang:
            return text
        
        if not TRANSLATION_ENABLED:
            return text
        
        import torch
        import re
        
        # Direct translation
        if (source_lang, target_lang) in self.translation_models:
            model, tokenizer = self.translation_models[(source_lang, target_lang)]
            
            # Split text into sentences to avoid truncation issues
            # This regex splits on sentence-ending punctuation while keeping the punctuation
            sentence_pattern = r'(?<=[.!?])\s+'
            sentences = re.split(sentence_pattern, text)
            
            # If only one "sentence" (no splits), just translate it directly
            if len(sentences) == 1:
                sentences = [text]
            
            translated_parts = []
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                try:
                    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
                    with torch.no_grad():
                        translated = model.generate(**inputs, max_length=512)
                    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
                    translated_parts.append(translated_text)
                except Exception as e:
                    logger.warning(f"Translation error for chunk: {e}")
                    translated_parts.append(sentence)  # Keep original if translation fails
            
            return " ".join(translated_parts)
        
        # Via English for ru<->he
        if source_lang == "ru" and target_lang == "he":
            english = self._translate(text, "ru", "en")
            return self._translate(english, "en", "he")
        elif source_lang == "he" and target_lang == "ru":
            english = self._translate(text, "he", "en")
            return self._translate(english, "en", "ru")
        
        logger.warning(f"No translation model for {source_lang}->{target_lang}")
        return text
    
    def _speak(self, text: str, language: str):
        """Convert text to speech."""
        if language == "ru":
            self._speak_silero_russian(text)
        elif language == "he":
            self._speak_mms_hebrew(text)
        else:
            self._speak_piper_english(text)
    
    def _speak_piper_english(self, text: str):
        """Use Piper TTS for English."""
        if not self.piper_voice_en:
            logger.error("Piper English not loaded")
            return
        
        try:
            import wave
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
            
            with wave.open(temp_path, "wb") as wav_file:
                self.piper_voice_en.synthesize_wav(text, wav_file)
            
            self._play_audio(temp_path)
        except Exception as e:
            logger.error(f"Piper TTS error: {e}")
    
    def _speak_silero_russian(self, text: str):
        """Use Silero TTS for Russian."""
        if self.silero_model is None:
            logger.error("Silero Russian not loaded")
            return
        
        try:
            import torch
            import scipy.io.wavfile
            import numpy as np
            import re
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
            
            # Split into sentences
            sentences = re.split(r'(?<=[.!?])\s+', text)
            audio_chunks = []
            
            for sentence in sentences:
                if not sentence.strip():
                    continue
                audio = self.silero_model.apply_tts(
                    text=sentence,
                    speaker=SILERO_SPEAKER,
                    sample_rate=SILERO_SAMPLE_RATE
                )
                audio_np = audio.cpu().numpy() if torch.is_tensor(audio) else audio
                audio_chunks.append(audio_np)
            
            if audio_chunks:
                full_audio = np.concatenate(audio_chunks)
                scipy.io.wavfile.write(temp_path, SILERO_SAMPLE_RATE, full_audio)
                self._play_audio(temp_path)
        except Exception as e:
            logger.error(f"Silero TTS error: {e}")
    
    def _speak_mms_hebrew(self, text: str):
        """Use MMS-TTS for Hebrew."""
        if not self.mms_model or not self.mms_tokenizer:
            logger.error("MMS Hebrew not loaded")
            return
        
        try:
            import torch
            import scipy.io.wavfile
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
            
            inputs = self.mms_tokenizer(text, return_tensors="pt")
            
            if next(self.mms_model.parameters()).is_cuda:
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            with torch.no_grad():
                output = self.mms_model(**inputs).waveform
            
            waveform = output.squeeze().cpu().numpy()
            sample_rate = self.mms_model.config.sampling_rate
            
            scipy.io.wavfile.write(temp_path, sample_rate, waveform)
            self._play_audio(temp_path)
        except Exception as e:
            logger.error(f"MMS Hebrew TTS error: {e}")
    
    def _play_audio(self, filepath: str):
        """Play audio file."""
        try:
            cmd = [AUDIO_PLAYER] + AUDIO_PLAYER_ARGS + [filepath]
            self.audio_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.audio_process.wait()
            self.audio_process = None
            
            # Cleanup temp file
            try:
                os.unlink(filepath)
            except:
                pass
        except Exception as e:
            logger.error(f"Audio playback error: {e}")
    
    def _stop_audio(self):
        """Stop current audio playback."""
        if self.audio_process:
            try:
                self.audio_process.terminate()
                self.audio_process = None
            except:
                pass
    
    def _handle_request(self, request: dict) -> dict:
        """Handle a request."""
        command = request.get("command", "")
        
        if command == "stop":
            self._stop_audio()
            return {"success": True}
        
        elif command == "speak":
            text = request.get("text", "")
            source_lang = request.get("source_lang", "unknown")
            target_lang = request.get("target_lang", "en")
            
            if not text:
                return {"success": False, "error": "No text provided"}
            
            try:
                # Ensure models are loaded
                self._load_all_models()
                
                # Update activity
                self.last_activity = time.time()
                
                # Translate if needed
                final_text = text
                if source_lang != "unknown" and source_lang != target_lang:
                    logger.info(f"Translating {source_lang}->{target_lang}...")
                    final_text = self._translate(text, source_lang, target_lang)
                    logger.info(f"Translation: {final_text[:100]}...")
                
                # Speak
                logger.info(f"Speaking in {target_lang}...")
                self._speak(final_text, target_lang)
                
                return {"success": True}
            except Exception as e:
                logger.error(f"Processing error: {e}")
                return {"success": False, "error": str(e)}
        
        return {"success": False, "error": f"Unknown command: {command}"}
    
    def _handle_client(self, conn):
        """Handle a client connection."""
        try:
            # Read all data until newline (no size limit)
            chunks = []
            while True:
                chunk = conn.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)
                if b'\n' in chunk:
                    break
            
            data = b''.join(chunks).decode()
            if not data:
                return
            
            logger.debug(f"Received {len(data)} bytes from client")
            
            request = json.loads(data.strip())
            result = self._handle_request(request)
            conn.sendall(json.dumps(result).encode())
            
        except Exception as e:
            logger.error(f"Client handler error: {e}")
            try:
                conn.sendall(json.dumps({"success": False, "error": str(e)}).encode())
            except:
                pass
        finally:
            conn.close()
    
    def _monitor_timeout(self):
        """Monitor for inactivity and exit when timeout reached."""
        while self.running:
            time.sleep(5)
            
            if self.models_loaded:
                elapsed = time.time() - self.last_activity
                if elapsed >= WORKER_TIMEOUT:
                    logger.info(f"💤 Worker timeout ({int(elapsed)}s idle) - exiting to free RAM...")
                    self.running = False
                    
                    if self.server_socket:
                        try:
                            self.server_socket.close()
                        except:
                            pass
                    
                    os._exit(0)
    
    def run(self):
        """Run the worker server."""
        socket_path = Path(SOCKET_PATH)
        socket_path.unlink(missing_ok=True)
        
        self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_socket.bind(SOCKET_PATH)
        self.server_socket.listen(5)
        self.server_socket.settimeout(1.0)
        
        logger.info(f"✅ Worker ready, listening on {SOCKET_PATH}")
        
        while self.running:
            try:
                conn, _ = self.server_socket.accept()
                threading.Thread(target=self._handle_client, args=(conn,), daemon=True).start()
            except socket.timeout:
                continue
            except OSError:
                break
            except Exception as e:
                logger.error(f"Server error: {e}")
                break
        
        try:
            self.server_socket.close()
        except:
            pass
        socket_path.unlink(missing_ok=True)
        logger.info("👋 Worker exiting")


def cleanup(signum=None, frame=None):
    """Cleanup on exit."""
    Path(SOCKET_PATH).unlink(missing_ok=True)
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)
    
    try:
        worker = TextReaderWorker()
        worker.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Worker fatal error: {e}")
    finally:
        cleanup()
