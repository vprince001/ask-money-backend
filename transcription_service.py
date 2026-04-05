"""
Transcription Service - Singleton Faster Whisper model loader
Modular service for speech-to-text, ready for future GPU/queue scaling
"""
import os
import logging
import tempfile
import uuid
from typing import Optional

logger = logging.getLogger(__name__)

# Singleton model instance
_model = None
_model_loading = False
MAX_FILE_SIZE_MB = 25  # 25MB limit


def get_model():
    """Get or initialize the Faster Whisper model (singleton pattern)"""
    global _model, _model_loading

    if _model is not None:
        return _model

    if _model_loading:
        raise RuntimeError("Model is still loading, please retry in a moment")

    _model_loading = True

    try:
        from faster_whisper import WhisperModel

        model_size = os.getenv("WHISPER_MODEL_SIZE", "medium")
        compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
        device = os.getenv("WHISPER_DEVICE", "auto")
        cpu_threads = int(os.getenv("WHISPER_CPU_THREADS", "6"))

        logger.info("🚀 Starting Whisper model load...")
        logger.info(f"Model: {model_size}, Device: {device}, Threads: {cpu_threads}")

        _model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads
        )

        logger.info("✅ Whisper model FULLY loaded and ready")

        return _model  # ✅ INSIDE function

    except Exception as e:
        logger.error(f"❌ Failed to load Faster Whisper model: {e}")
        _model = None
        raise RuntimeError(f"Failed to load transcription model: {e}")

    finally:
        _model_loading = False  # ✅ ALWAYS reset

def transcribe_audio_file(file_path: str, language: Optional[str] = None) -> dict:
    """
    Transcribe an audio file using Faster Whisper.

    Args:
        file_path: Path to the audio file
        language: Language code (e.g., 'en', 'hi'). Strongly recommended for accuracy.

    Returns:
        dict with 'text', 'language', 'segments' keys
    """
    model = get_model()

    kwargs = {
        "task": "transcribe",
        "beam_size": 2,
        "vad_filter": True,
        "vad_parameters": {"min_silence_duration_ms": 500},
        "condition_on_previous_text": False,
    }

    # Explicitly set language - critical for Hindi and non-English accuracy
    if language:
        kwargs["language"] = language

    segments, info = model.transcribe(file_path, **kwargs)

    # Collect all segment texts
    text_parts = []
    segment_details = []
    for segment in segments:
        text_parts.append(segment.text.strip())
        segment_details.append({
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": segment.text.strip(),
        })

    full_text = " ".join(text_parts)

    return {
        "text": full_text,
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "duration": round(info.duration, 2),
        "segments": segment_details,
    }


def validate_file_size(file_size_bytes: int) -> bool:
    """Check if file size is within limits"""
    return file_size_bytes <= MAX_FILE_SIZE_MB * 1024 * 1024


def save_temp_audio(audio_bytes: bytes, extension: str = ".m4a") -> str:
    """Save audio bytes to a temporary file"""
    temp_path = os.path.join(tempfile.gettempdir(), f"audio_{uuid.uuid4().hex[:12]}{extension}")
    with open(temp_path, "wb") as f:
        f.write(audio_bytes)
    return temp_path


def cleanup_temp_file(file_path: str):
    """Safely delete a temporary file"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        logger.warning(f"Failed to cleanup temp file {file_path}: {e}")


def preload_model():
    """Preload the model at startup (call during app init)"""
    try:
        get_model()
        logger.info("Transcription model preloaded")
    except Exception as e:
        logger.warning(f"Model preload failed (will retry on first request): {e}")
