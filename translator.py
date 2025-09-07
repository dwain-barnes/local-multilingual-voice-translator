import json
from pathlib import Path
import gradio as gr
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastrtc import (
    AdditionalOutputs,
    ReplyOnPause,
    Stream,
    audio_to_bytes,
    get_twilio_turn_credentials,
)
from gradio.utils import get_space
from pydantic import BaseModel
import librosa
import io
import os
from openai import AsyncOpenAI
import wave
import tempfile
import asyncio
import base64
from typing import Optional
import torchaudio as ta
import torch

# Import Chatterbox TTS
try:
    from chatterbox.tts import ChatterboxTTS
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    CHATTERBOX_AVAILABLE = True
    print("Chatterbox TTS successfully imported")
except ImportError:
    CHATTERBOX_AVAILABLE = False
    print("Chatterbox TTS not available. Install with: pip install chatterbox-tts")

# Import Distil-Whisper FastRTC
try:
    from distil_whisper_fastrtc import get_stt_model
    DISTIL_WHISPER_AVAILABLE = True
    print("Distil-Whisper FastRTC successfully imported")
except ImportError:
    DISTIL_WHISPER_AVAILABLE = False
    print("Distil-Whisper FastRTC not available. Install with: pip install distil-whisper-fastrtc")

cur_dir = Path(__file__).parent
load_dotenv()

# Initialize LM Studio client (OpenAI-compatible API)
LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
LM_STUDIO_API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")

lm_studio_client = AsyncOpenAI(
    base_url=LM_STUDIO_BASE_URL,
    api_key=LM_STUDIO_API_KEY
)

# Initialize models
chatterbox_english = None
chatterbox_multilingual = None
whisper_model = None

def initialize_chatterbox():
    """Initialize Chatterbox TTS models"""
    global chatterbox_english, chatterbox_multilingual
    
    if not CHATTERBOX_AVAILABLE:
        print("Chatterbox TTS not available")
        return False
        
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing Chatterbox TTS on device: {device}")
        
        # Initialize English-only model
        chatterbox_english = ChatterboxTTS.from_pretrained(device=device)
        print("English Chatterbox TTS model loaded")
        
        # Initialize multilingual model
        chatterbox_multilingual = ChatterboxMultilingualTTS.from_pretrained(device=device)
        print("Multilingual Chatterbox TTS model loaded")
        
        return True
    except Exception as e:
        print(f"Error initializing Chatterbox TTS: {e}")
        return False

def initialize_whisper():
    """Initialize Distil-Whisper STT model"""
    global whisper_model
    
    if not DISTIL_WHISPER_AVAILABLE:
        print("Distil-Whisper FastRTC not available")
        return False
        
    try:
        print("Initializing Distil-Whisper STT model...")
        
        # Choose model based on available resources
        # For multilingual support, use distil-large-v3
        # For English-only and faster inference, use distil-medium.en
        model_name = os.getenv("WHISPER_MODEL", "distil-whisper/distil-large-v3")
        
        whisper_model = get_stt_model(model_name)
        print(f"Distil-Whisper model '{model_name}' loaded successfully")
        
        return True
    except Exception as e:
        print(f"Error initializing Distil-Whisper: {e}")
        return False

# Supported languages with their codes and names (Chatterbox TTS supported only)
SUPPORTED_LANGUAGES = {
    "ar": "Arabic",
    "da": "Danish", 
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ms": "Malay",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sv": "Swedish",
    "sw": "Swahili",
    "tr": "Turkish",
    "zh": "Chinese"
}

# Audio buffer to accumulate short chunks
audio_buffer = []
buffer_duration = 0.0
MIN_AUDIO_DURATION = 1.0

# Global variables
current_tts_enabled = False
current_webrtc_id = None
current_target_language = "es"  # Default to Spanish
current_source_language = "en"  # Default source is English

# TTS audio queue for streaming
tts_audio_queue = {}

async def transcribe_with_distil_whisper(audio_data, sample_rate):
    """Transcribe audio using Distil-Whisper FastRTC"""
    if not DISTIL_WHISPER_AVAILABLE or not whisper_model:
        return "Distil-Whisper not available"
        
    try:
        print(f"Transcribing with Distil-Whisper - Sample rate: {sample_rate}, Length: {len(audio_data)/sample_rate:.2f}s")
        
        # Ensure audio is float32 and properly normalized
        if audio_data.dtype != np.float32:
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            else:
                audio_data = audio_data.astype(np.float32)
        
        # Ensure audio is 1D
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Distil-Whisper expects (sample_rate, audio_array) tuple
        audio_tuple = (sample_rate, audio_data)
        
        # Transcribe using Distil-Whisper
        transcribed_text = whisper_model.stt(audio_tuple)
        
        print(f"Distil-Whisper STT result: '{transcribed_text}'")
        return transcribed_text
        
    except Exception as e:
        print(f"Distil-Whisper STT error: {e}")
        return "Transcription failed"

async def translate_text(text, target_language):
    """Translate text to target language using LM Studio"""
    try:
        # Get language names
        source_lang_name = SUPPORTED_LANGUAGES.get(current_source_language, current_source_language)
        target_lang_name = SUPPORTED_LANGUAGES.get(target_language, target_language)
        
        system_prompt = f"""You are a professional translator that converts {source_lang_name} text into natural, conversational {target_lang_name}.

Guidelines:
- Translate idiomatically, not word-for-word
- Keep the natural flow and tone of the original
- Use contemporary, everyday language that native speakers would actually use
- Do not provide explanations - only return the translated text
- Maintain the same level of formality as the original"""
        
        # Get available models from LM Studio
        try:
            models = await lm_studio_client.models.list()
            model_name = models.data[0].id if models.data else "local-model"
            print(f"Using LM Studio model: {model_name}")
        except Exception:
            model_name = "local-model"
        
        response = await lm_studio_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Translate this {source_lang_name} text to {target_lang_name}: {text}"}
            ],
            max_tokens=1000,
            temperature=0.3,
            stream=False
        )
        
        translated_text = response.choices[0].message.content.strip()
        print(f"LM Studio translation ({source_lang_name}→{target_lang_name}): '{text}' → '{translated_text}'")
        return translated_text
        
    except Exception as e:
        print(f"LM Studio translation error: {e}")
        print(f"Make sure LM Studio is running at {LM_STUDIO_BASE_URL}")
        return text

async def synthesize_speech_chatterbox(text: str, language_code: str) -> bytes:
    """Generate speech using Chatterbox TTS"""
    try:
        if not CHATTERBOX_AVAILABLE or (not chatterbox_english and not chatterbox_multilingual):
            print("Chatterbox TTS not available")
            return None
            
        print(f"Generating TTS with Chatterbox for text: '{text}' (language: {language_code})")
        
        # Use English model for English, multilingual for others
        if language_code == "en" and chatterbox_english:
            model = chatterbox_english
            wav = model.generate(text)
        elif chatterbox_multilingual and language_code in SUPPORTED_LANGUAGES:
            model = chatterbox_multilingual
            wav = model.generate(text, language_id=language_code)
        else:
            # Fallback: try English model even for other languages
            if chatterbox_english:
                model = chatterbox_english
                wav = model.generate(text)
            else:
                print(f"No suitable model for language: {language_code}")
                return None
        
        # Convert tensor to numpy if needed
        if hasattr(wav, 'cpu'):
            wav = wav.cpu().numpy()
        elif torch.is_tensor(wav):
            wav = wav.detach().numpy()
        
        # Ensure correct shape (should be 1D)
        if wav.ndim > 1:
            wav = wav.squeeze()
        
        # Convert to bytes (WAV format)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            ta.save(tmp_file.name, torch.from_numpy(wav).unsqueeze(0), model.sr)
            tmp_file.seek(0)
            audio_bytes = tmp_file.read()
        
        # Clean up temp file
        os.unlink(tmp_file.name)
        
        print(f"Chatterbox TTS generated successfully, audio size: {len(audio_bytes)} bytes")
        return audio_bytes
        
    except Exception as e:
        print(f"Chatterbox TTS generation error: {e}")
        return None

async def queue_tts_audio(webrtc_id: str, audio_data: bytes):
    """Queue TTS audio for streaming to client"""
    if webrtc_id not in tts_audio_queue:
        tts_audio_queue[webrtc_id] = asyncio.Queue()
    
    audio_b64 = base64.b64encode(audio_data).decode('utf-8')
    await tts_audio_queue[webrtc_id].put({
        "type": "tts_audio",
        "audio": audio_b64,
        "format": "wav"
    })

async def transcribe(audio: tuple[int, np.ndarray], transcript: str = ""):
    global audio_buffer, buffer_duration, current_tts_enabled, current_webrtc_id, current_target_language
    
    try:
        sample_rate, audio_data = audio
        
        print(f"Received audio - Sample rate: {sample_rate}, Shape: {audio_data.shape}")
        print(f"TTS enabled: {current_tts_enabled}, Target language: {current_target_language}")
        
        if audio_data.size == 0:
            print("Warning: Empty audio data received")
            return
            
        if audio_data.ndim == 0:
            print("Warning: 0-dimensional audio data received")
            return
            
        # Convert to float32 and normalize if needed
        if audio_data.dtype != np.float32:
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            else:
                audio_data = audio_data.astype(np.float32)
        
        # Handle multi-channel audio - convert to mono
        if audio_data.ndim > 1:
            print(f"Multi-dimensional audio detected: {audio_data.shape}")
            if audio_data.shape[0] < audio_data.shape[1]:
                if audio_data.shape[0] == 2:
                    audio_data = np.mean(audio_data, axis=0)
                else:
                    audio_data = audio_data[0, :]
            else:
                if audio_data.shape[1] == 2:
                    audio_data = np.mean(audio_data, axis=1)
                else:
                    audio_data = audio_data[:, 0]
        
        audio_data = np.squeeze(audio_data)
        
        if audio_data.ndim == 0 or audio_data.size == 0:
            print("ERROR: Audio became invalid after processing!")
            return
        
        # Add to buffer
        audio_buffer.append(audio_data)
        buffer_duration += len(audio_data) / sample_rate
        
        print(f"Buffer duration: {buffer_duration:.2f}s, chunks: {len(audio_buffer)}")
        
        # Only process when we have enough audio
        if buffer_duration < MIN_AUDIO_DURATION:
            print(f"Buffering audio... need {MIN_AUDIO_DURATION}s, have {buffer_duration:.2f}s")
            return
        
        # Concatenate buffered audio
        combined_audio = np.concatenate(audio_buffer)
        audio_buffer = []
        buffer_duration = 0.0
        
        print(f"Processing combined audio - Length: {len(combined_audio)/sample_rate:.2f}s")
        
        # Resample if necessary (Whisper works best with 16kHz)
        if sample_rate != 16000:
            combined_audio = librosa.resample(combined_audio, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        
        # Run transcription using Distil-Whisper
        transcribed_text = await transcribe_with_distil_whisper(combined_audio, sample_rate)
        transcribed_text = transcribed_text.strip()
        
        print(f"Transcription result: '{transcribed_text}'")
        
        # Translate to target language
        final_text = transcribed_text
        if transcribed_text and current_target_language != current_source_language:
            print(f"Translating to {SUPPORTED_LANGUAGES.get(current_target_language, current_target_language)}...")
            final_text = await translate_text(transcribed_text, current_target_language)
            
        # Generate TTS if enabled
        if current_tts_enabled and final_text and final_text.lower() not in ["", " ", "...", ".", ","] and current_webrtc_id:
            print(f"Generating TTS for: '{final_text}'")
            asyncio.create_task(generate_and_queue_tts(final_text, current_webrtc_id, current_target_language))
        
        # Only yield if we have actual text
        if final_text and final_text.lower() not in ["", " ", "...", ".", ","]:
            if transcript is None:
                transcript = ""
            yield AdditionalOutputs(transcript + "\n" + final_text)
        else:
            print("No meaningful transcription result")
        
    except Exception as e:
        print(f"Error in transcribe function: {e}")
        audio_buffer = []
        buffer_duration = 0.0

async def generate_and_queue_tts(text: str, webrtc_id: str, language_code: str):
    """Generate TTS and queue it for streaming"""
    try:
        audio_bytes = await synthesize_speech_chatterbox(text, language_code)
        if audio_bytes and webrtc_id:
            await queue_tts_audio(webrtc_id, audio_bytes)
            print(f"TTS audio queued for webrtc_id: {webrtc_id}")
    except Exception as e:
        print(f"Error in generate_and_queue_tts: {e}")

# Initialize models on startup
initialize_chatterbox()
initialize_whisper()

# Gradio interface components
transcript = gr.Textbox(label="Transcript")

stream = Stream(
    ReplyOnPause(transcribe),
    modality="audio",
    mode="send",
    additional_inputs=[transcript],
    additional_outputs=[transcript],
    additional_outputs_handler=lambda a, b: b,
    rtc_configuration=get_twilio_turn_credentials() if get_space() else None,
    concurrency_limit=5 if get_space() else None,
    time_limit=90 if get_space() else None,
)

app = FastAPI()
app.mount("/static", StaticFiles(directory=cur_dir), name="static")
stream.mount(app)

class SendInput(BaseModel):
    webrtc_id: str
    transcript: str
    tts_enabled: bool = False
    target_language: str = "es"  # Default to Spanish
    source_language: str = "en"  # Default source

@app.post("/send_input")
def send_input(body: SendInput):
    global current_tts_enabled, current_webrtc_id, current_target_language, current_source_language
    current_tts_enabled = body.tts_enabled
    current_webrtc_id = body.webrtc_id
    current_target_language = body.target_language
    current_source_language = body.source_language
    
    print(f"Updated settings - TTS: {current_tts_enabled}, Target: {current_target_language}")
    stream.set_input(body.webrtc_id, body.transcript)

@app.get("/transcript")
def get_transcript(webrtc_id: str):
    async def output_stream():
        async for output in stream.output_stream(webrtc_id):
            transcript = output.args[0].split("\n")[-1]
            yield f"event: transcript\ndata: {transcript}\n\n"
    
    return StreamingResponse(output_stream(), media_type="text/event-stream")

@app.get("/tts_audio")
def get_tts_audio(webrtc_id: str):
    """Stream TTS audio to client"""
    async def audio_stream():
        if webrtc_id not in tts_audio_queue:
            tts_audio_queue[webrtc_id] = asyncio.Queue()
        
        queue = tts_audio_queue[webrtc_id]
        
        try:
            while True:
                audio_data = await queue.get()
                yield f"event: audio\ndata: {json.dumps(audio_data)}\n\n"
                queue.task_done()
        except asyncio.CancelledError:
            print(f"TTS audio stream cancelled for {webrtc_id}")
        except Exception as e:
            print(f"Error in TTS audio stream: {e}")
    
    return StreamingResponse(audio_stream(), media_type="text/event-stream")

@app.get("/supported_languages")
def get_supported_languages():
    """Return list of supported languages"""
    return {"languages": SUPPORTED_LANGUAGES}

@app.get("/")
def index():
    rtc_config = get_twilio_turn_credentials() if get_space() else None
    html_content = (cur_dir / "index.html").read_text(encoding='utf-8')
    html_content = html_content.replace("__RTC_CONFIGURATION__", json.dumps(rtc_config))
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import os
    if (mode := os.getenv("MODE")) == "UI":
        stream.ui.launch(server_port=7860)
    elif mode == "PHONE":
        stream.fastphone(host="0.0.0.0", port=7860)
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=7860)