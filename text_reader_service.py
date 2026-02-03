#!/usr/bin/env python3
"""
AI Text Reader Service - Lightweight Listener + Heavy Worker Architecture

This service uses a two-process architecture to minimize memory usage:
1. Listener (this process): Lightweight, always running (~50-100 MB RAM)
   - Listens for hotkey press
   - Gets selected text
   - Spawns worker subprocess for translation/TTS
   
2. Worker (subprocess): Heavy, spawned on demand
   - Loads translation models (MarianMT)
   - Loads TTS models (Piper, Silero, MMS)
   - Processes text and plays audio
   - Dies after timeout, fully freeing RAM
"""

import os
import sys
import signal
import subprocess
import json
import socket
import threading
import time
import logging
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Hotkey configuration
HOTKEY = "Insert"
STOP_HOTKEY = "ctrl+Insert"

# Worker Timeout (seconds)
# Worker subprocess will exit after this period of inactivity
WORKER_TIMEOUT = 60  # 1 minute

# KDE Keyboard Layout Mapping
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

# IPC Settings
SOCKET_PATH = "/tmp/text_reader_worker.sock"
PID_FILE = "/tmp/text_reader_service.pid"

# Logging
LOG_LEVEL = logging.DEBUG

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
# TEXT READER LISTENER
# ============================================================================

class TextReaderListener:
    """Lightweight listener that handles hotkey and text selection."""
    
    def __init__(self):
        logger.info("🚀 Starting Text Reader Listener (lightweight mode)...")
        
        self.running = True
        self.worker_process = None
        self.processing_lock = threading.Lock()
        self.is_processing = False
        
        # Parse hotkeys
        self.modifier_mask, self.trigger_key = self._parse_hotkey(HOTKEY)
        self.stop_modifier_mask, self.stop_trigger_key = self._parse_hotkey(STOP_HOTKEY)
        
        # X11 setup
        self.display = display.Display()
        self.root = self.display.screen().root
        
        # Get keycodes
        self.trigger_keycode = self._get_keycode(self.trigger_key)
        if self.trigger_keycode is None:
            logger.error(f"Could not find keycode for key: {self.trigger_key}")
            sys.exit(1)
        
        self.stop_keycode = self._get_keycode(self.stop_trigger_key)
        if self.stop_keycode is None:
            logger.error(f"Could not find keycode for stop key: {self.stop_trigger_key}")
            sys.exit(1)
        
        logger.info(f"Hotkey configured: {HOTKEY}")
        logger.info(f"Stop hotkey configured: {STOP_HOTKEY}")
        logger.info(f"Worker timeout: {WORKER_TIMEOUT}s (full RAM release)")
    
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
    
    def _get_selected_text(self) -> str:
        """Get currently selected text using Xlib directly."""
        from Xlib import X
        
        try:
            # Create a temporary window to receive selection
            screen = self.display.screen()
            window = screen.root.create_window(
                0, 0, 1, 1, 0, screen.root_depth,
                event_mask=X.PropertyChangeMask
            )
            
            # Atoms
            PRIMARY = self.display.intern_atom("PRIMARY")
            UTF8_STRING = self.display.intern_atom("UTF8_STRING")
            XSEL_DATA = self.display.intern_atom("XSEL_DATA")
            
            try:
                # Request selection conversion
                window.convert_selection(PRIMARY, UTF8_STRING, XSEL_DATA, X.CurrentTime)
                self.display.flush()
                
                # Wait for SelectionNotify event with timeout
                start_time = time.time()
                timeout = 5.0
                
                while time.time() - start_time < timeout:
                    if self.display.pending_events():
                        event = self.display.next_event()
                        if event.type == X.SelectionNotify:
                            if event.property == X.NONE:
                                logger.warning("No selection available")
                                return ""
                            
                            # Read the selection data
                            result = window.get_full_property(XSEL_DATA, UTF8_STRING)
                            if result:
                                data = result.value
                                if isinstance(data, bytes):
                                    text = data.decode('utf-8', errors='replace').strip()
                                else:
                                    text = str(data).strip()
                                logger.debug(f"Got selection: {len(text)} chars")
                                return text
                            return ""
                    else:
                        time.sleep(0.01)
                
                logger.warning("Selection timeout")
                return ""
            finally:
                window.destroy()
                
        except Exception as e:
            logger.error(f"Error getting selected text: {e}")
            return ""
    
    def _get_current_layout(self) -> str:
        """Get current keyboard layout language code."""
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
                return lang
        except Exception as e:
            logger.debug(f"KDE layout detection failed: {e}")
        return "en"
    
    def _detect_text_script(self, text: str) -> str:
        """Detect language by analyzing character scripts."""
        if not text or len(text.strip()) < 1:
            return "unknown"
        
        latin_count = 0
        cyrillic_count = 0
        hebrew_count = 0
        
        for char in text:
            if ('a' <= char <= 'z') or ('A' <= char <= 'Z'):
                latin_count += 1
            elif '\u0400' <= char <= '\u04FF':
                cyrillic_count += 1
            elif '\u0590' <= char <= '\u05FF':
                hebrew_count += 1
        
        total = latin_count + cyrillic_count + hebrew_count
        if total == 0:
            return "unknown"
        
        if latin_count >= cyrillic_count and latin_count >= hebrew_count:
            return "en"
        elif cyrillic_count >= latin_count and cyrillic_count >= hebrew_count:
            return "ru"
        else:
            return "he"
    
    def _ensure_worker_running(self):
        """Ensure worker subprocess is running."""
        if self.worker_process and self.worker_process.poll() is None:
            return True
        
        logger.info("🚀 Starting worker subprocess...")
        
        # Clean up old socket
        socket_path = Path(SOCKET_PATH)
        socket_path.unlink(missing_ok=True)
        
        # Start worker process
        worker_script = Path(__file__).parent / "text_reader_worker.py"
        self.worker_process = subprocess.Popen(
            [sys.executable, str(worker_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # Print worker output
        def print_worker_output():
            for line in self.worker_process.stdout:
                print(f"[Worker] {line.rstrip()}")
        
        threading.Thread(target=print_worker_output, daemon=True).start()
        
        # Wait for socket
        for _ in range(100):  # 10 seconds max
            if socket_path.exists():
                return True
            time.sleep(0.1)
        
        logger.error("Worker failed to start")
        return False
    
    def _send_to_worker(self, command: str, text: str = "", source_lang: str = "", target_lang: str = ""):
        """Send command to worker."""
        if not self._ensure_worker_running():
            logger.error("Worker not available")
            return
        
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(SOCKET_PATH)
            
            request = {
                "command": command,
                "text": text,
                "source_lang": source_lang,
                "target_lang": target_lang
            }
            sock.sendall(json.dumps(request).encode() + b'\n')
            
            response = sock.recv(4096).decode()
            result = json.loads(response)
            
            sock.close()
            
            if not result.get("success"):
                logger.error(f"Worker error: {result.get('error')}")
                
        except Exception as e:
            logger.error(f"Worker communication error: {e}")
    
    def stop_playback(self):
        """Stop current audio playback."""
        if self.worker_process and self.worker_process.poll() is None:
            self._send_to_worker("stop")
            logger.info("Playback stopped")
    
    def process_selection(self):
        """Main processing: get selected text and send to worker."""
        logger.debug(">>> process_selection() called")
        
        # Stop any current playback first
        self.stop_playback()
        
        with self.processing_lock:
            if self.is_processing:
                logger.info("Stopping previous reading, starting new...")
                time.sleep(0.1)
            self.is_processing = True
        
        try:
            logger.debug(">>> Getting selected text...")
            text = self._get_selected_text()
            
            logger.info(f">>> Got text: {len(text)} chars, {len(text.split())} words")
            
            if not text:
                logger.info("No text selected")
                return
            
            logger.info(f"Selected text: {text[:100]}{'...' if len(text) > 100 else ''}")
            
            target_lang = self._get_current_layout()
            logger.info(f"Target language (keyboard layout): {target_lang}")
            
            text_script = self._detect_text_script(text)
            logger.info(f"Detected text script: {text_script}")
            
            logger.debug(">>> Sending to worker...")
            # Send to worker for translation and TTS
            self._send_to_worker(
                command="speak",
                text=text,
                source_lang=text_script,
                target_lang=target_lang
            )
            logger.debug(">>> Sent to worker")
            
        except Exception as e:
            logger.error(f"Processing error: {e}")
            import traceback
            logger.error(traceback.format_exc())
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
                
                if keycode == self.stop_keycode and clean_state == self.stop_modifier_mask:
                    logger.debug("Stop hotkey pressed!")
                    self.stop_playback()
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
        
        # Terminate worker
        if self.worker_process and self.worker_process.poll() is None:
            logger.info("Stopping worker...")
            self.worker_process.terminate()
            self.worker_process.wait(timeout=5)
        
        logger.info("Service stopped")


# ============================================================================
# MAIN
# ============================================================================

def cleanup():
    """Cleanup on exit."""
    pid_file = Path(PID_FILE)
    if pid_file.exists():
        pid_file.unlink()
    logger.info("Cleanup complete")


def main():
    pid_file = Path(PID_FILE)
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
        listener = TextReaderListener()
        listener.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        cleanup()


if __name__ == "__main__":
    main()
