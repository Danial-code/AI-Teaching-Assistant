from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
import threading

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

OUTPUT_DIR = Path("outputs/lecture_summaries")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

recording = False
recording_thread = None
SAMPLE_RATE = 16000
buffer = []

def record_audio():
    global recording, buffer
    buffer = []
    print("🎙️ Recording started...")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16') as stream:
        while recording:
            chunk, _ = stream.read(1024)
            buffer.append(chunk)

@app.post("/start-recording")
def start_recording():
    global recording, recording_thread
    recording = True
    recording_thread = threading.Thread(target=record_audio)
    recording_thread.start()
    return {"message": "Recording started"}

@app.post("/stop-recording")
def stop_recording():
    global recording
    recording = False
    recording_thread.join()

    audio_data = np.concatenate(buffer, axis=0)
    filename = f"lecture_live_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    write(filename, SAMPLE_RATE, audio_data)
    print(f"✅ Audio saved to: {filename}")

    transcript_text = transcribe_audio(filename)
    summary_text = summarize_text(transcript_text)

    transcript_path = OUTPUT_DIR / "latest_transcript.txt"
    summary_path = OUTPUT_DIR / "latest_summary.txt"
    transcript_path.write_text(transcript_text)
    summary_path.write_text(summary_text)

    return {"message": "Recording stopped", "transcript": transcript_text, "summary": summary_text}

def transcribe_audio(path):
    model = whisper.load_model("base")
    print("🧠 Transcribing...")
    result = model.transcribe(path)
    return result["text"]

def summarize_text(text):
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)
    summarizer = pipeline("text-generation", model=model, tokenizer=tokenizer)
    prompt = f"Summarize this lecture in 5 concise bullet points:\n\n{text}"
    summary = summarizer(prompt, max_new_tokens=400)[0]["generated_text"]
    return summary
