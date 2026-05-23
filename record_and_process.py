import argparse
import queue
import sys
import threading
import time
from pathlib import Path
import sounddevice as sd
import soundfile as sf
import subprocess
import shutil

ROOT = Path(__file__).resolve().parents[0]

def record_to_file(filename: str, samplerate=44100, channels=1, subtype="PCM_16"):

    q = queue.Queue()

    def callback(indata, frames, time_info, status):

        if status:
            print(f"Recording status: {status}", file=sys.stderr)
        q.put(indata.copy())


    with sf.SoundFile(filename, mode='w', samplerate=samplerate,
                      channels=channels, subtype=subtype) as f:
        with sd.InputStream(samplerate=samplerate, channels=channels, callback=callback):
            print("Recording... Press ENTER to stop.")

            stop = False

            def writer_loop():
                while True:
                    try:
                        data = q.get()
                        if data is None:
                            break
                        f.write(data)
                    except Exception as e:
                        print("Writer error:", e)
                        break

            writer = threading.Thread(target=writer_loop, daemon=True)
            writer.start()

            try:
                input()
            except KeyboardInterrupt:
                print("\nInterrupted by user (Ctrl+C). Stopping recording.")

            q.put(None)

            time.sleep(0.2)

    print(f"Saved recording to: {filename}")

def find_python_executable():

    return sys.executable

def call_transcribe_script(audio_path: str, model: str, language: str, device: str, delete_audio: bool, out: str | None):

    py = find_python_executable()
    script = ROOT / "summarizer" / "transcribe_and_summarize.py"
    if not script.exists():
        raise FileNotFoundError(f"{script} not found. Make sure summarizer/transcribe_and_summarize.py exists.")

    cmd = [py, str(script), str(audio_path), "--model", model, "--language", language, "--device", device]
    if delete_audio:
        cmd.append("--delete-audio")
    if out:
        cmd += ["--out", out]

    print("Running transcription & summarization...")
    print(" ".join(cmd))
    proc = subprocess.run(cmd)
    return proc.returncode

def default_out_path(audio_path: Path):

    stem = audio_path.stem
    notes = audio_path.with_name(f"{stem}_notes.md")
    return str(notes)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="Output notes path (markdown). Default: <audio>_notes.md")
    ap.add_argument("--audio-dir", default=str(ROOT / "recordings"), help="Where to save audio files")
    ap.add_argument("--filename", default=None, help="If provided, use this filename (e.g. lecture.wav). Otherwise a timestamped name is used.")
    ap.add_argument("--model", default="base", help="faster-whisper model for transcription (tiny/base/small/medium/large-v3)")
    ap.add_argument("--language", default="en", help="Language code (en)")
    ap.add_argument("--device", default=None, help="Device for Whisper: cpu (recommended on Mac). If not set, script chooses cpu when MPS exists.")
    ap.add_argument("--delete-audio", action="store_true", help="Delete audio after processing")
    ap.add_argument("--samplerate", type=int, default=44100, help="Recording sample rate")
    ap.add_argument("--channels", type=int, default=1, help="Number of channels (1 mono, 2 stereo)")
    args = ap.parse_args()

    outdir = Path(args.audio_dir)
    outdir.mkdir(parents=True, exist_ok=True)


    if args.filename:
        audio_path = outdir / args.filename
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        audio_path = outdir / f"lecture_{ts}.wav"

    if args.device:
        device = args.device
    else:

        try:
            import torch
            device = "cpu" if torch.backends.mps.is_available() else "cpu"
        except Exception:
            device = "cpu"


    try:
        record_to_file(str(audio_path), samplerate=args.samplerate, channels=args.channels)
    except Exception as e:
        print("Error recording audio:", e)
        return 2


    out_notes = args.out if args.out else default_out_path(audio_path)


    try:
        rc = call_transcribe_script(str(audio_path), args.model, args.language, device, args.delete_audio, out_notes)
    except Exception as e:
        print("Error during transcription/summarization:", e)
        return 3

    if rc != 0:
        print(f"Transcription script exited with code {rc}. Check logs.")
        return rc

    print("All done.")
    if args.delete_audio:
        print("Audio file was deleted by the transcription script (if supported).")
    else:
        print(f"Audio preserved at: {audio_path}")
        print(f"Notes saved to: {out_notes}")

    return 0

if __name__ == "__main__":
    sys.exit(main())

