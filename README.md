# AI Reader Assistant

Reads selected text aloud with automatic translation based on keyboard layout.
**Fully local** - no internet required.

## How it works

1. Select text in any application
2. Press `Ctrl + XF86Calculator`
3. Service:
   - Detects text language
   - Checks keyboard layout
   - Translates if needed (Ollama)
   - Reads aloud (Piper/MMS TTS)

## Supported languages

| Layout | Language | TTS model |
|--------|----------|-----------|
| 0 | English | Piper |
| 1 | Russian | Piper |
| 2 | Hebrew | Facebook MMS |

## Service Installation

```bash
# 1. Navigate to project directory
cd /mnt/DiskE_Crucial/codding/My_Projects/ai-reader-assistant

# 2. Make script executable
chmod +x text_reader_service.py

# 3. Create systemd user directory
mkdir -p ~/.config/systemd/user

# 4. Copy service file
cp text-reader.service ~/.config/systemd/user/

# 5. Reload systemd, enable and start service
systemctl --user daemon-reload
systemctl --user enable text-reader.service
systemctl --user start text-reader.service

# 6. Check status
systemctl --user status text-reader.service

# 7. View logs (live)
journalctl --user -u text-reader.service -f
```

## Service Management

```bash
# Start
systemctl --user start text-reader.service

# Stop
systemctl --user stop text-reader.service

# Restart (after config changes)
systemctl --user restart text-reader.service

# Status
systemctl --user status text-reader.service

# Enable auto-start
systemctl --user enable text-reader.service

# Disable auto-start
systemctl --user disable text-reader.service

# Recent logs
journalctl --user -u text-reader.service -n 50

# Live logs
journalctl --user -u text-reader.service -f
```

## Configuration

Settings in `text_reader_service.py`:

```python
# Hotkey
HOTKEY = "ctrl+XF86Calculator"

# Ollama model for translation
OLLAMA_MODEL = "qwen3:8b"

# Enable/disable translation
TRANSLATION_ENABLED = True

# Keyboard layouts
LAYOUT_LANGUAGE_MAP = {
    0: "en",  # English
    1: "ru",  # Russian
    2: "he",  # Hebrew
}
```

Find key name:
```bash
xev | grep keysym
# Press desired key
```

## Dependencies

System:
```bash
sudo apt install xclip ffmpeg
```

Ollama:
```bash
ollama pull qwen3:8b
```

## Manual run (for testing)

```bash
cd /mnt/DiskE_Crucial/codding/My_Projects/ai-reader-assistant
uv run python text_reader_service.py
```

## Troubleshooting

If service fails to start:
1. Check logs: `journalctl --user -u text-reader.service -n 50`
2. Verify paths in `~/.config/systemd/user/text-reader.service`
3. Test manually: `uv run python text_reader_service.py`
