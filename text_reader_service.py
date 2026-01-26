#!/usr/bin/env python3
"""
AI Text Reader Service (Fully Local)
Reads selected text aloud with automatic translation based on keyboard layout.

Uses:
- Helsinki-NLP/MarianMT for fast translation (local transformer models)
- Piper TTS for English/Russian (local ONNX models)
- Facebook MMS-TTS for Hebrew (local transformer model)

Press Home (or configured hotkey) to read selected text.
Press Ctrl+Home to stop playback.
If text alphabet differs from keyboard layout language, it will be translated first.
"""

import os
import sys
import signal
import subprocess
import tempfile
import threading
import logging
import time
import gc
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Hotkey configuration
# Format: "modifier+key" or just "key"
# Examples: "XF86Calculator", "ctrl+XF86Calculator", "ctrl+shift+r"
HOTKEY = "Insert"
STOP_HOTKEY = "ctrl+Insert"

# Model Unload Timeout (seconds)
# Models will be unloaded from GPU/memory after this period of inactivity
# Set to 0 to disable auto-unloading (always keep models in memory)
MODEL_UNLOAD_TIMEOUT = 60  # 1 minute

# ============================================================================
# TRANSLATION SETTINGS (Helsinki-NLP MarianMT - Local)
# ============================================================================

# MarianMT models for translation (fast, local)
# Format: (source_lang, target_lang) -> model_name
TRANSLATION_MODELS = {
    ("en", "ru"): "Helsinki-NLP/opus-mt-en-ru",
    ("ru", "en"): "Helsinki-NLP/opus-mt-ru-en",
    ("en", "he"): "Helsinki-NLP/opus-mt-en-he",
    ("he", "en"): "Helsinki-NLP/opus-mt-tc-big-he-en",
    # For ru<->he, translate through English
}

# Disable translation (just read in original language)
TRANSLATION_ENABLED = True

# ============================================================================
# TTS SETTINGS
# ============================================================================

# Piper TTS for English (high quality)
PIPER_MODELS_DIR = Path(__file__).parent / "models" / "piper"
PIPER_VOICE_EN = "en_US-amy-medium"

# Silero TTS V5 for Russian (high quality, proper stress/intonation)
SILERO_MODEL_ID = "v4_ru"  # v4_ru is stable, supports SSML
SILERO_SPEAKER = "xenia"   # Options: aidar, baya, kseniya, xenia, eugene, random
SILERO_SAMPLE_RATE = 48000

# Hebrew uses Facebook MMS-TTS (transformer model)
MMS_HEBREW_MODEL = "facebook/mms-tts-heb"

# ============================================================================
# KEYBOARD LAYOUT MAPPING
# ============================================================================

LAYOUT_LANGUAGE_MAP = {
    0: "en",  # English
    1: "ru",  # Russian
    2: "he",  # Hebrew
}

LAYOUT_NAMES = {
    0: "English",
    1: "Русский",
    2: "עברית",
}

# ============================================================================
# AUDIO SETTINGS
# ============================================================================

AUDIO_PLAYER = "ffplay"
AUDIO_PLAYER_ARGS = ["-nodisp", "-autoexit", "-loglevel", "quiet"]

# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL = logging.INFO

# ============================================================================
# IMPORTS
# ============================================================================

try:
    from Xlib import X, XK, display
    from Xlib.ext import record
    from Xlib.protocol import rq
    XK.load_keysym_group('xf86')
except ImportError:
    print("ERROR: python-xlib is required")
    sys.exit(1)

try:
    from langdetect import detect as detect_language
    from langdetect.lang_detect_exception import LangDetectException
except ImportError:
    print("ERROR: langdetect is required")
    sys.exit(1)

try:
    from piper import PiperVoice
    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False
    print("WARNING: piper-tts not available")

try:
    from transformers import VitsModel, AutoTokenizer, MarianMTModel, MarianTokenizer
    import torch
    import scipy.io.wavfile
    MMS_AVAILABLE = True
    MARIAN_AVAILABLE = True
except ImportError:
    MMS_AVAILABLE = False
    MARIAN_AVAILABLE = False
    print("WARNING: transformers/torch not available")

import wave

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ============================================================================
# TEXT READER SERVICE
# ============================================================================

class TextReaderService:
    """Service that reads selected text aloud with optional translation (fully local)."""

    def __init__(self):
        self.running = True
        self.processing_lock = threading.Lock()
        self.is_processing = False
        self.audio_process = None

        # Parse hotkeys
        self.modifier_mask, self.trigger_key = self._parse_hotkey(HOTKEY)
        self.stop_modifier_mask, self.stop_trigger_key = self._parse_hotkey(STOP_HOTKEY)

        # X11 setup
        self.display = display.Display()
        self.root = self.display.screen().root

        # Get keycode for trigger key
        self.trigger_keycode = self._get_keycode(self.trigger_key)
        if self.trigger_keycode is None:
            logger.error(f"Could not find keycode for key: {self.trigger_key}")
            sys.exit(1)

        # Get keycode for stop key
        self.stop_keycode = self._get_keycode(self.stop_trigger_key)
        if self.stop_keycode is None:
            logger.error(f"Could not find keycode for stop key: {self.stop_trigger_key}")
            sys.exit(1)

        logger.info(f"Hotkey configured: {HOTKEY}")
        logger.info(f"Stop hotkey configured: {STOP_HOTKEY}")

        # Models are loaded lazily on first use
        self.piper_voice_en = None
        self.silero_model = None
        self.silero_device = None
        self.mms_model = None
        self.mms_tokenizer = None
        self.translation_models = {}

        # Model management
        self.models_lock = threading.Lock()
        self.models_loaded = False
        self.last_used_time: float = 0
        self.unload_timer: threading.Timer | None = None

        logger.info(f"TextReaderService initialized (lazy model loading)")
        logger.info(f"Model unload timeout: {MODEL_UNLOAD_TIMEOUT}s")
        logger.info(f"Translation: {'Enabled (MarianMT)' if TRANSLATION_ENABLED else 'Disabled'}")
        logger.info("Models will load on first use")

    def _load_all_models(self):
        """Load all models into memory (called on demand)."""
        with self.models_lock:
            if self.models_loaded:
                # Models already loaded, just update last used time
                self.last_used_time = time.time()
                self._schedule_unload()
                return

            logger.info("Loading all models...")

            # Load TTS models
            self._load_piper_english()
            self._load_silero_russian()
            self._load_mms_hebrew()

            # Load translation models (MarianMT)
            self._load_translation_models()

            self.models_loaded = True
            self.last_used_time = time.time()
            self._schedule_unload()
            logger.info("All models loaded successfully!")

    def _unload_all_models(self):
        """Unload all models from memory to free resources."""
        with self.models_lock:
            if not self.models_loaded:
                return  # Already unloaded

            # Check if enough time has passed since last use
            elapsed = time.time() - self.last_used_time
            if elapsed < MODEL_UNLOAD_TIMEOUT:
                # Not enough time passed, reschedule
                self._schedule_unload()
                return

            logger.info(f"Unloading models (idle for {int(elapsed)}s)...")

            # Unload Piper English
            if self.piper_voice_en is not None:
                del self.piper_voice_en
                self.piper_voice_en = None

            # Unload Silero Russian
            if self.silero_model is not None:
                del self.silero_model
                self.silero_model = None

            # Unload MMS Hebrew
            if self.mms_model is not None:
                del self.mms_model
                del self.mms_tokenizer
                self.mms_model = None
                self.mms_tokenizer = None

            # Unload translation models
            if self.translation_models:
                for key in list(self.translation_models.keys()):
                    del self.translation_models[key]
                self.translation_models = {}

            self.models_loaded = False

            # Force garbage collection
            gc.collect()

            # Clear CUDA cache if torch is available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            logger.info("Models unloaded, memory freed!")

    def _schedule_unload(self):
        """Schedule model unloading after timeout."""
        if MODEL_UNLOAD_TIMEOUT <= 0:
            return  # Auto-unload disabled

        # Cancel existing timer if any
        if self.unload_timer is not None:
            self.unload_timer.cancel()

        # Schedule new unload
        self.unload_timer = threading.Timer(MODEL_UNLOAD_TIMEOUT, self._unload_all_models)
        self.unload_timer.daemon = True
        self.unload_timer.start()

    def _load_piper_english(self):
        """Load Piper voice model for English."""
        if not PIPER_AVAILABLE:
            logger.warning("Piper TTS not available")
            return

        if not PIPER_MODELS_DIR.exists():
            logger.warning(f"Piper models directory not found: {PIPER_MODELS_DIR}")
            return

        model_path = PIPER_MODELS_DIR / f"{PIPER_VOICE_EN}.onnx"
        config_path = PIPER_MODELS_DIR / f"{PIPER_VOICE_EN}.onnx.json"

        if model_path.exists() and config_path.exists():
            try:
                self.piper_voice_en = PiperVoice.load(str(model_path), str(config_path))
                logger.info(f"Loaded Piper voice for English: {PIPER_VOICE_EN}")
            except Exception as e:
                logger.error(f"Failed to load Piper English voice: {e}")

    def _load_silero_russian(self):
        """Load Silero TTS V4 for Russian (high quality with proper stress)."""
        try:
            logger.info(f"Loading Silero Russian model: {SILERO_MODEL_ID}, speaker: {SILERO_SPEAKER}...")

            # Determine device
            self.silero_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-models',
                model='silero_tts',
                language='ru',
                speaker=SILERO_MODEL_ID
            )
            model.to(self.silero_device)
            self.silero_model = model

            logger.info(f"Silero Russian model loaded on {self.silero_device}")
            logger.info(f"Silero model type: {type(self.silero_model)}")
        except Exception as e:
            logger.error(f"Failed to load Silero Russian model: {e}")
            import traceback
            traceback.print_exc()
            self.silero_model = None

    def _load_mms_hebrew(self):
        """Load Facebook MMS-TTS model for Hebrew."""
        if not MMS_AVAILABLE:
            logger.warning("MMS-TTS not available for Hebrew")
            return

        try:
            logger.info(f"Loading MMS Hebrew model: {MMS_HEBREW_MODEL}...")
            self.mms_model = VitsModel.from_pretrained(MMS_HEBREW_MODEL)
            self.mms_tokenizer = AutoTokenizer.from_pretrained(MMS_HEBREW_MODEL)

            # Move to GPU if available
            if torch.cuda.is_available():
                self.mms_model = self.mms_model.to("cuda")
                logger.info("MMS Hebrew model loaded on GPU")
            else:
                logger.info("MMS Hebrew model loaded on CPU")

        except Exception as e:
            logger.error(f"Failed to load MMS Hebrew model: {e}")
            self.mms_model = None
            self.mms_tokenizer = None

    def _load_translation_models(self):
        """Load MarianMT translation models."""
        if not TRANSLATION_ENABLED:
            return

        if not MARIAN_AVAILABLE:
            logger.warning("MarianMT not available - translation disabled")
            return

        logger.info("Loading translation models (MarianMT)...")

        for (src, tgt), model_name in TRANSLATION_MODELS.items():
            try:
                logger.info(f"  Loading {src}->{tgt}: {model_name}...")
                tokenizer = MarianTokenizer.from_pretrained(model_name)
                model = MarianMTModel.from_pretrained(model_name)
                # Keep on CPU - translation models are small and fast
                self.translation_models[(src, tgt)] = (model, tokenizer)
                logger.info(f"  Loaded {src}->{tgt}")
            except Exception as e:
                logger.error(f"  Failed to load {model_name}: {e}")

        logger.info(f"Translation models loaded: {len(self.translation_models)} pairs")

    def _parse_hotkey(self, hotkey_str: str) -> tuple:
        """Parse hotkey string into modifier mask and key name."""
        parts = hotkey_str.lower().split('+')
        modifier_mask = 0

        for part in parts[:-1]:
            if part in ('ctrl', 'control'):
                modifier_mask |= X.ControlMask
            elif part == 'shift':
                modifier_mask |= X.ShiftMask
            elif part in ('alt', 'mod1'):
                modifier_mask |= X.Mod1Mask
            elif part in ('super', 'mod4', 'win'):
                modifier_mask |= X.Mod4Mask

        # Restore original case for key name
        original_parts = hotkey_str.split('+')
        key_name = original_parts[-1]

        return modifier_mask, key_name

    def _get_keycode(self, key_name: str) -> int:
        """Convert key name to X11 keycode."""
        keysym = XK.string_to_keysym(key_name)

        if keysym == 0 and key_name.startswith("XF86"):
            alt_name = key_name[:4] + "_" + key_name[4:]
            keysym = XK.string_to_keysym(alt_name)

        if keysym == 0:
            logger.error(f"Unknown key: {key_name}")
            return None

        keycode = self.display.keysym_to_keycode(keysym)
        if keycode == 0:
            logger.error(f"Could not get keycode for keysym: {keysym}")
            return None

        return keycode

    def get_selected_text(self) -> str:
        """Get currently selected text using xclip."""
        try:
            result = subprocess.run(
                ["xclip", "-selection", "primary", "-o"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except subprocess.TimeoutExpired:
            logger.warning("xclip timeout")
        except FileNotFoundError:
            logger.error("xclip not found")
        except Exception as e:
            logger.error(f"Error getting selected text: {e}")
        return ""

    def get_current_layout(self) -> tuple:
        """Get current keyboard layout index and language code."""
        try:
            result = subprocess.run(
                ["qdbus6", "org.kde.keyboard", "/Layouts", "getLayout"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                layout_id = int(result.stdout.strip())
                lang = LAYOUT_LANGUAGE_MAP.get(layout_id, "en")
                name = LAYOUT_NAMES.get(layout_id, "Unknown")
                logger.debug(f"Current layout: {name} ({lang})")
                return layout_id, lang
        except Exception as e:
            logger.debug(f"KDE layout detection failed: {e}")

        return 0, "en"

    def detect_text_script(self, text: str) -> str:
        """Detect language by analyzing character scripts (alphabet)."""
        if not text or len(text.strip()) < 1:
            return "unknown"

        # Count characters by script
        latin_count = 0
        cyrillic_count = 0
        hebrew_count = 0

        for char in text:
            # Latin alphabet (English)
            if ('a' <= char <= 'z') or ('A' <= char <= 'Z'):
                latin_count += 1
            # Cyrillic alphabet (Russian)
            elif ('\u0400' <= char <= '\u04FF'):
                cyrillic_count += 1
            # Hebrew alphabet
            elif ('\u0590' <= char <= '\u05FF'):
                hebrew_count += 1

        total = latin_count + cyrillic_count + hebrew_count
        if total == 0:
            return "unknown"

        # Determine dominant script
        if latin_count >= cyrillic_count and latin_count >= hebrew_count:
            return "en"
        elif cyrillic_count >= latin_count and cyrillic_count >= hebrew_count:
            return "ru"
        else:
            return "he"

    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text using MarianMT (fast, local)."""
        if source_lang == target_lang:
            return text

        if not TRANSLATION_ENABLED:
            return text

        # Direct translation if model exists
        if (source_lang, target_lang) in self.translation_models:
            return self._translate_marian(text, source_lang, target_lang)

        # For ru<->he, translate through English
        if source_lang == "ru" and target_lang == "he":
            english = self._translate_marian(text, "ru", "en")
            return self._translate_marian(english, "en", "he")
        elif source_lang == "he" and target_lang == "ru":
            english = self._translate_marian(text, "he", "en")
            return self._translate_marian(english, "en", "ru")

        logger.warning(f"No translation model for {source_lang}->{target_lang}")
        return text

    def _translate_marian(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate using MarianMT model."""
        if (source_lang, target_lang) not in self.translation_models:
            logger.error(f"Translation model not loaded: {source_lang}->{target_lang}")
            return text

        model, tokenizer = self.translation_models[(source_lang, target_lang)]

        try:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

            with torch.no_grad():
                translated = model.generate(**inputs)

            result = tokenizer.decode(translated[0], skip_special_tokens=True)
            logger.info(f"Translated {source_lang}->{target_lang}")
            return result

        except Exception as e:
            logger.error(f"MarianMT translation error: {e}")
            return text

    def speak_text(self, text: str, language: str):
        """Convert text to speech using appropriate TTS model."""
        if language == "ru":
            self._speak_silero_russian(text)
        elif language == "he":
            self._speak_mms_hebrew(text)
        elif language == "en":
            self._speak_piper_english(text)
        else:
            # Fallback to English
            logger.warning(f"No TTS for {language}, falling back to English")
            self._speak_piper_english(text)

    def _speak_piper_english(self, text: str):
        """Use Piper TTS for English."""
        if not self.piper_voice_en:
            logger.error("Piper English voice not loaded")
            return

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name

            with wave.open(temp_path, "wb") as wav_file:
                self.piper_voice_en.synthesize_wav(text, wav_file)

            self._play_audio(temp_path)

        except Exception as e:
            logger.error(f"Piper TTS error: {e}")
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass

    def _speak_silero_russian(self, text: str):
        """Use Silero TTS V4 for Russian (high quality)."""
        if self.silero_model is None:
            logger.error("Silero Russian model not loaded (is None)")
            return

        try:
            logger.info(f"Silero input text ({len(text)} chars): {text[:200]}...")

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name

            # Split long text into sentences for better processing
            # Silero works better with shorter chunks
            import re
            sentences = re.split(r'(?<=[.!?])\s+', text)
            audio_chunks = []

            for sentence in sentences:
                if not sentence.strip():
                    continue
                logger.debug(f"Processing sentence: {sentence[:50]}...")
                audio = self.silero_model.apply_tts(
                    text=sentence,
                    speaker=SILERO_SPEAKER,
                    sample_rate=SILERO_SAMPLE_RATE
                )
                audio_np = audio.cpu().numpy() if torch.is_tensor(audio) else audio
                audio_chunks.append(audio_np)

            if not audio_chunks:
                logger.warning("No audio chunks generated")
                return

            # Concatenate all audio chunks
            import numpy as np
            full_audio = np.concatenate(audio_chunks)

            # Save to WAV file
            scipy.io.wavfile.write(temp_path, SILERO_SAMPLE_RATE, full_audio)
            logger.info(f"Generated audio: {len(full_audio)} samples")

            self._play_audio(temp_path)

        except Exception as e:
            logger.error(f"Silero TTS error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass

    def _speak_mms_hebrew(self, text: str):
        """Use Facebook MMS-TTS for Hebrew."""
        if not self.mms_model or not self.mms_tokenizer:
            logger.error("MMS Hebrew model not loaded")
            return

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name

            inputs = self.mms_tokenizer(text, return_tensors="pt")

            # Move to GPU if model is on GPU
            if next(self.mms_model.parameters()).is_cuda:
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            with torch.no_grad():
                output = self.mms_model(**inputs).waveform

            # Convert to numpy and save
            waveform = output.squeeze().cpu().numpy()
            sample_rate = self.mms_model.config.sampling_rate

            scipy.io.wavfile.write(temp_path, sample_rate, waveform)

            self._play_audio(temp_path)

        except Exception as e:
            logger.error(f"MMS Hebrew TTS error: {e}")
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass

    def _play_audio(self, filepath: str):
        """Play audio file."""
        try:
            cmd = [AUDIO_PLAYER] + AUDIO_PLAYER_ARGS + [filepath]
            self.audio_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.audio_process.wait()
            self.audio_process = None
        except FileNotFoundError:
            logger.error(f"Audio player not found: {AUDIO_PLAYER}")
        except Exception as e:
            logger.error(f"Audio playback error: {e}")

    def stop_playback(self):
        """Stop current audio playback."""
        if self.audio_process:
            try:
                self.audio_process.terminate()
                self.audio_process = None
                logger.info("Playback stopped")
            except Exception as e:
                logger.error(f"Error stopping playback: {e}")

    def process_selection(self):
        """Main processing: get selected text, detect/translate, and speak."""
        # Stop any current playback first
        self.stop_playback()

        with self.processing_lock:
            if self.is_processing:
                logger.info("Stopping previous reading, starting new...")
                time.sleep(0.1)
            self.is_processing = True

        try:
            text = self.get_selected_text()
            if not text:
                logger.info("No text selected")
                return

            logger.info(f"Selected text: {text[:100]}{'...' if len(text) > 100 else ''}")

            # Ensure models are loaded
            self._load_all_models()

            _, target_lang = self.get_current_layout()
            logger.info(f"Target language (keyboard layout): {target_lang}")

            # Detect text script by character alphabet
            text_script = self.detect_text_script(text)
            logger.info(f"Detected text script: {text_script}")

            final_text = text
            # If text alphabet doesn't match keyboard layout - translate first
            if text_script != "unknown" and text_script != target_lang:
                logger.info(f"Text alphabet ({text_script}) differs from layout ({target_lang}) - translating...")
                final_text = self.translate_text(text, text_script, target_lang)
                logger.info(f"Translation result: {final_text[:100]}{'...' if len(final_text) > 100 else ''}")

            logger.info(f"Speaking in {target_lang}...")
            self.speak_text(final_text, target_lang)
            logger.info("Done speaking")

        except Exception as e:
            logger.error(f"Processing error: {e}")
        finally:
            with self.processing_lock:
                self.is_processing = False

    def _keyboard_callback(self, reply):
        """Callback for keyboard events."""
        if reply.category != record.FromServer:
            return
        if reply.client_swapped:
            return

        data = reply.data
        while len(data):
            event, data = rq.EventField(None).parse_binary_value(
                data, self.record_display.display, None, None
            )

            if event.type == X.KeyPress:
                keycode = event.detail
                state = event.state
                clean_state = state & (X.ControlMask | X.ShiftMask | X.Mod1Mask | X.Mod4Mask)

                # Check for stop hotkey
                if keycode == self.stop_keycode and clean_state == self.stop_modifier_mask:
                    logger.debug("Stop hotkey pressed!")
                    self.stop_playback()
                # Check for main hotkey
                elif keycode == self.trigger_keycode and clean_state == self.modifier_mask:
                    logger.debug("Hotkey pressed!")
                    threading.Thread(target=self.process_selection, daemon=True).start()

    def start(self):
        """Start the keyboard listener."""
        logger.info("Starting keyboard listener...")

        self.record_display = display.Display()

        ctx = self.record_display.record_create_context(
            0,
            [record.AllClients],
            [{
                'core_requests': (0, 0),
                'core_replies': (0, 0),
                'ext_requests': (0, 0, 0, 0),
                'ext_replies': (0, 0, 0, 0),
                'delivered_events': (0, 0),
                'device_events': (X.KeyPress, X.KeyRelease),
                'errors': (0, 0),
                'client_started': False,
                'client_died': False,
            }]
        )

        logger.info("Listening for hotkeys...")
        logger.info(f"Press {HOTKEY} to read selected text")
        logger.info(f"Press {STOP_HOTKEY} to stop playback")

        self.record_display.record_enable_context(ctx, self._keyboard_callback)
        self.record_display.record_free_context(ctx)

    def stop(self):
        """Stop the service."""
        self.running = False

        # Cancel unload timer
        if self.unload_timer is not None:
            self.unload_timer.cancel()
            self.unload_timer = None

        # Unload models immediately
        with self.models_lock:
            if self.models_loaded:
                if self.piper_voice_en is not None:
                    del self.piper_voice_en
                    self.piper_voice_en = None
                if self.silero_model is not None:
                    del self.silero_model
                    self.silero_model = None
                if self.mms_model is not None:
                    del self.mms_model
                    del self.mms_tokenizer
                    self.mms_model = None
                    self.mms_tokenizer = None
                self.translation_models = {}
                self.models_loaded = False
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        logger.info("Service stopped")


# ============================================================================
# MAIN
# ============================================================================

def cleanup():
    """Cleanup on exit."""
    pid_file = Path("/tmp/text_reader_service.pid")
    if pid_file.exists():
        pid_file.unlink()
    logger.info("Cleanup complete")


def main():
    pid_file = Path("/tmp/text_reader_service.pid")
    if pid_file.exists():
        try:
            old_pid = int(pid_file.read_text().strip())
            os.kill(old_pid, 0)
            logger.error(f"Service already running with PID {old_pid}")
            sys.exit(1)
        except (ProcessLookupError, ValueError):
            pid_file.unlink()

    pid_file.write_text(str(os.getpid()))

    signal.signal(signal.SIGTERM, lambda sig, frame: cleanup() or sys.exit(0))
    signal.signal(signal.SIGINT, lambda sig, frame: cleanup() or sys.exit(0))

    try:
        service = TextReaderService()
        service.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        cleanup()


if __name__ == "__main__":
    main()
