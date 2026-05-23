from flask import Flask, render_template_string, request, jsonify, Response, send_file, url_for
import os
import uuid
import base64
import re
import threading
import queue
import sys

from werkzeug.utils import secure_filename

# Audio / STT
import sounddevice as sd
import soundfile as sf  # pip install soundfile
import whisper

# Your local LLaMA-powered helpers
from main_ollama import (
    extract_text,
    generate_quiz_stream,
    regenerate_quiz_stream,
    summarize_lecture,
)

# Unified quiz grading engine
from grading import grade_quiz


app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
RECORDINGS_FOLDER = os.path.join(BASE_DIR, "outputs", "recordings")
TRANSCRIPTS_FOLDER = os.path.join(BASE_DIR, "outputs", "transcripts")
SUMMARIES_FOLDER = os.path.join(BASE_DIR, "outputs", "summaries")
STATIC_FOLDER = os.path.join(BASE_DIR, "static")

for d in [UPLOAD_FOLDER, RECORDINGS_FOLDER, TRANSCRIPTS_FOLDER, SUMMARIES_FOLDER, STATIC_FOLDER]:
    os.makedirs(d, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024
ALLOWED_EXTS = {".pdf", ".pptx"}
ALLOWED_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}

SESSIONS = {}

# Global lecture notes text for recording/summarization
CURRENT_LECTURE_NOTES_TEXT = ""


UKH_LOGO_STATIC_FILENAME = "Logo-Transparent.png"
UKH_LOGO_LOCAL_PATH = "/Users/danial/Desktop/Ai_teaching_assistant/Ai-teaching-assistant/Logo-Transparent.png"


def _logo_url() -> str:

    if UKH_LOGO_LOCAL_PATH and os.path.exists(UKH_LOGO_LOCAL_PATH):
        return url_for("ukh_logo")

    static_path = os.path.join(STATIC_FOLDER, UKH_LOGO_STATIC_FILENAME)
    if os.path.exists(static_path):
        return url_for("static", filename=UKH_LOGO_STATIC_FILENAME)

    return url_for("favicon")


@app.route("/ukh-logo")
def ukh_logo():

    if not UKH_LOGO_LOCAL_PATH:
        return Response("UKH_LOGO_LOCAL_PATH is not set", status=404)
    if not os.path.exists(UKH_LOGO_LOCAL_PATH):
        return Response("UKH logo not found at UKH_LOGO_LOCAL_PATH", status=404)

    ext = os.path.splitext(UKH_LOGO_LOCAL_PATH)[1].lower()
    mimetype = "image/png" if ext == ".png" else ("image/svg+xml" if ext == ".svg" else "application/octet-stream")
    return send_file(UKH_LOGO_LOCAL_PATH, mimetype=mimetype)



# Whisper model (local, via openai-whisper)

WHISPER_DEVICE = "cpu"
WHISPER_MODEL = None


def get_whisper():
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        WHISPER_MODEL = whisper.load_model("base", device=WHISPER_DEVICE)
    return WHISPER_MODEL


# Recording manager (manual start/stop, no time limit)

class RecordingManager:
    def __init__(self):
        self.is_recording = False
        self.filepath = None
        self.thread = None
        self.q = None
        self.samplerate = 16000
        self.channels = 1
        self.stream = None

    def _callback(self, indata, frames, time_, status):
        if status:
            print("Sounddevice status:", status, file=sys.stderr)
        if self.is_recording and self.q is not None:
            self.q.put(indata.copy())

    def _writer_thread(self, filepath):
        with sf.SoundFile(
            filepath,
            mode="w",
            samplerate=self.samplerate,
            channels=self.channels,
            subtype="PCM_16",
        ) as f:
            while self.is_recording or (not self.q.empty()):
                try:
                    data = self.q.get(timeout=0.2)
                except queue.Empty:
                    continue
                f.write(data)

    def start(self):
        if self.is_recording:
            return self.filepath

        session_id = uuid.uuid4().hex
        self.filepath = os.path.join(RECORDINGS_FOLDER, f"{session_id}.wav")
        self.q = queue.Queue()
        self.is_recording = True

        self.thread = threading.Thread(
            target=self._writer_thread, args=(self.filepath,), daemon=True
        )
        self.thread.start()

        self.stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            callback=self._callback,
        )
        self.stream.start()
        return self.filepath

    def stop(self):
        if not self.is_recording:
            return None

        self.is_recording = False
        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
        except Exception:
            pass

        if self.thread is not None:
            self.thread.join(timeout=3.0)

        fp = self.filepath
        self.filepath = None
        self.q = None
        self.stream = None
        return fp


RECORDER = RecordingManager()


# Small helpers for quiz parsing / grading

_STOPWORDS = {
    "the", "a", "an", "is", "are", "to", "of", "and", "in", "on", "for", "with",
    "that", "this", "it", "as", "be", "by", "or", "at", "from", "your", "you",
    "i", "we", "they", "them", "their", "our",
}


def _clean(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _tokens(s: str) -> list:
    return [t for t in re.findall(r"[a-z0-9_]+", (s or "").lower()) if t and t not in _STOPWORDS]


def _jaccard(a: list, b: list) -> float:
    A, B = set(a), set(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def _numbers(s: str) -> set:
    return set(re.findall(r"-?\d+(?:\.\d+)?", s or ""))


def _strip_choice_prefix(s: str) -> str:
    return re.sub(r"^[A-Da-d]\)\s*", "", s or "").strip()


def _mcq_letter(s: str):
    if not s:
        return None
    s = s.strip()
    m = re.match(r"^[A-Da-d]\b", s)
    if m:
        return m.group(0).upper()[0]
    m = re.search(r"\b([A-Da-d])\)", s)
    if m:
        return m.group(1).upper()
    return None


def _looks_blank_or_idk(s: str) -> bool:
    t = _clean(s)
    return (not t) or t in {
        "idk", "i don t know", "dont know", "do not know", "no idea", "not sure", "unknown", "i dont know"
    }


def _parse_quiz_text_to_struct(text: str):
    """
    Recognizes:
    Question N (Type): ...
    A) ...
    Answer: ...
    """
    if not text:
        return {"questions": [], "answers": [], "types": []}

    lines = [ln.rstrip() for ln in text.splitlines()]
    q_re = re.compile(r"^\s*Question\s+(\d+)\s*\(([^)]+)\):\s*(.*)$", re.IGNORECASE)
    opt_re = re.compile(r"^\s*([A-Da-d])\)\s*(.*)$")
    ans_re = re.compile(r"^\s*Answer:\s*(.*)$", re.IGNORECASE)
    exp_re = re.compile(r"^\s*Expected\s*Answer:\s*(.*)$", re.IGNORECASE)

    questions, answers, types = [], [], []
    cur_qtext, cur_type, cur_opts = None, None, []

    def flush():
        nonlocal cur_qtext, cur_type, cur_opts
        if cur_qtext is not None:
            questions.append({
                "type": cur_type or "short_answer",
                "question": cur_qtext.strip(),
                "options": cur_opts[:] if cur_type == "mcq" else [],
            })
            types.append(cur_type or "short_answer")
            if len(answers) < len(questions):
                answers.append("")
            cur_qtext, cur_type, cur_opts = None, None, []

    for ln in lines:
        if not ln.strip():
            continue

        m_q = q_re.match(ln)
        if m_q:
            flush()
            cur_type = (m_q.group(2) or "").strip().lower().replace(" ", "_")
            if cur_type not in {"mcq", "short_answer", "coding", "math"}:
                cur_type = "short_answer"
            cur_qtext = (m_q.group(3) or "").strip()
            cur_opts = []
            continue

        if cur_qtext is None:
            continue

        m_opt = opt_re.match(ln)
        if m_opt:
            if cur_type != "mcq":
                cur_type = "mcq"
                cur_opts = []
            cur_opts.append(f"{m_opt.group(1).upper()}) {(m_opt.group(2) or '').strip()}")
            continue

        m_ans = ans_re.match(ln)
        if m_ans:
            answers.append((m_ans.group(1) or "").strip())
            continue

        m_exp = exp_re.match(ln)
        if m_exp:
            answers.append((m_exp.group(1) or "").strip())
            continue

        cur_qtext = (cur_qtext + " " + ln.strip()) if cur_qtext else ln.strip()

    flush()
    if len(answers) < len(questions):
        answers.extend([""] * (len(questions) - len(answers)))

    return {"questions": questions, "answers": answers, "types": types}



BASE_CSS = """
<style>
  :root{
    --bg: #f2f5fb;
    --bg2:#eef3fb;
    --card: rgba(255,255,255,0.78);
    --border: rgba(15,23,42,0.10);
    --text: #0b1220;
    --muted: rgba(11,18,32,0.62);
    --nav: #071a33;
    --nav2:#0a2b52;
    --accent: #1d4ed8;
    --accent2:#0ea5e9;
    --accent3:#7c3aed;
    --shadow: 0 18px 50px rgba(15,23,42,0.10);
    --shadow2: 0 10px 24px rgba(15,23,42,0.08);
    --radius: 18px;
  }
  *{ box-sizing: border-box; }

  body{
    background:
      radial-gradient(1300px 750px at 10% -10%, rgba(29,78,216,0.16), transparent 60%),
      radial-gradient(1100px 700px at 95% 0%, rgba(14,165,233,0.12), transparent 60%),
      radial-gradient(1000px 700px at 55% 110%, rgba(124,58,237,0.08), transparent 55%),
      linear-gradient(180deg, var(--bg2) 0%, var(--bg) 55%, var(--bg) 100%);
    color: var(--text);
    font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
    min-height: 100vh;
    letter-spacing: 0.1px;
  }

  .ukh-nav{
    position: sticky; top: 0; z-index: 1000;
    background: linear-gradient(90deg, var(--nav), var(--nav2));
    border-bottom: 1px solid rgba(255,255,255,0.16);
  }
  .ukh-container{
    max-width: 1260px; margin: 0 auto;
    padding: 14px 18px;
    display:flex; align-items:center; justify-content:space-between; gap: 16px;
  }
  .ukh-brand{ display:flex; align-items:center; gap: 12px; min-width: 320px; }
  .ukh-logo{
    height: 70px;
    width: auto;
    object-fit: contain;
    padding: 6px 10px;
    border-radius: 14px;
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.18);
    box-shadow: 0 16px 30px rgba(0,0,0,0.18);
  }
  .ukh-title{ line-height: 1.05; }
  .ukh-uni{ color:#fff; font-weight: 900; font-size: 14px; letter-spacing: 0.25px; }
  .ukh-app{ color: rgba(255,255,255,0.80); font-weight: 650; font-size: 12.5px; margin-top: 3px; }

  .ukh-links{ display:flex; gap: 10px; flex-wrap: wrap; justify-content:flex-end; align-items:center; }
  .ukh-link{
    text-decoration:none; color:#fff;
    padding: 9px 13px;
    border-radius: 999px;
    font-weight: 800; font-size: 13px;
    border: 1px solid rgba(255,255,255,0.20);
    background: rgba(255,255,255,0.08);
    transition: transform .15s ease, background .15s ease, box-shadow .15s ease;
    user-select: none;
  }
  .ukh-link:hover{
    background: rgba(255,255,255,0.14);
    transform: translateY(-1px);
    box-shadow: 0 10px 24px rgba(0,0,0,0.18);
  }

  .page{ max-width: 1260px; margin: 0 auto; padding: 22px 18px 30px 18px; }

  .hero{
    border-radius: 22px;
    padding: 22px 22px;
    background:
      radial-gradient(900px 420px at 0% 0%, rgba(29,78,216,0.18), transparent 60%),
      radial-gradient(700px 420px at 110% 0%, rgba(14,165,233,0.14), transparent 60%),
      linear-gradient(180deg, rgba(255,255,255,0.78), rgba(255,255,255,0.64));
    border: 1px solid var(--border);
    box-shadow: var(--shadow);
    display:flex; align-items:flex-end; justify-content:space-between; gap: 16px;
    overflow: hidden;
    position: relative;
  }
  .hero:before{
    content:"";
    position:absolute; inset:-2px;
    background: linear-gradient(120deg, rgba(29,78,216,0.18), rgba(14,165,233,0.14), rgba(124,58,237,0.10));
    filter: blur(28px);
    opacity: 0.55;
    z-index: 0;
  }
  .hero > *{ position: relative; z-index: 1; }
  .hero h1{ margin:0; font-size: 26px; font-weight: 950; letter-spacing: -0.2px; }
  .hero p{ margin: 6px 0 0 0; color: var(--muted); font-size: 14px; max-width: 720px; }

  .pill{
    display:inline-flex; align-items:center; gap: 8px;
    padding: 7px 11px;
    border-radius: 999px;
    background: rgba(29,78,216,0.10);
    border: 1px solid rgba(29,78,216,0.18);
    color: rgba(11,31,58,0.90);
    font-weight: 900; font-size: 12px;
    white-space: nowrap;
  }
  .pill .dot{ width:8px; height:8px; border-radius:999px; background: linear-gradient(90deg, var(--accent), var(--accent2)); }

  .grid-2{ display:grid; grid-template-columns: 1fr 1.15fr; gap: 16px; margin-top: 16px; }
  @media (max-width: 992px){ .grid-2{ grid-template-columns: 1fr; } }

  .u-card{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    box-shadow: var(--shadow2);
    backdrop-filter: blur(10px);
    overflow: hidden;
  }
  .u-card-header{
    padding: 14px 16px;
    border-bottom: 1px solid rgba(15,23,42,0.08);
    display:flex; align-items:center; justify-content:space-between; gap: 10px;
  }
  .u-card-title{ margin:0; font-size: 14px; font-weight: 950; color: #071a33; }
  .u-card-body{ padding: 16px; }
  .help{ color: var(--muted); font-size: 13px; margin-top: 8px; }

  .form-control, textarea, input, select{
    background: rgba(255,255,255,0.92) !important;
    color: var(--text) !important;
    border: 1px solid rgba(15,23,42,0.18) !important;
    border-radius: 12px !important;
  }
  .form-control::placeholder, textarea::placeholder{ color: rgba(15,23,42,0.45) !important; }

  .btn{ border-radius: 12px !important; font-weight: 850 !important; }
  .btn-primary{
    background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
    border: none !important;
    box-shadow: 0 16px 32px rgba(29,78,216,0.18);
  }
  .btn-outline-light{
    border-color: rgba(15,23,42,0.22) !important;
    color: rgba(11,31,58,0.92) !important;
    background: rgba(255,255,255,0.58);
  }

  .dropzone{
    border: 2px dashed rgba(15,23,42,0.22);
    border-radius: 16px;
    padding: 18px;
    background: rgba(255,255,255,0.62);
    cursor: pointer;
    transition: transform .15s ease, background .15s ease, border-color .15s ease;
  }
  .dropzone:hover{ transform: translateY(-1px); }
  .dropzone.active{ border-color: rgba(29,78,216,0.60); background: rgba(29,78,216,0.06); }

  .file-chip{
    display:inline-flex; align-items:center; gap: 8px;
    padding: 8px 10px;
    border-radius: 999px;
    border: 1px solid rgba(15,23,42,0.14);
    background: rgba(255,255,255,0.72);
    color: rgba(11,18,32,0.80);
    font-size: 12.5px;
    font-weight: 800;
  }

  .editor{
    background: #ffffff;
    color: #0f172a;
    min-height: 290px;
    border-radius: 14px;
    padding: 16px;
    white-space: pre-wrap;
    border: 1px solid rgba(15,23,42,0.18);
  }

  .top-progress{
    position: fixed; left: 0; top: 0;
    width: 100%; height: 3px;
    z-index: 2000;
    opacity: 0;
    transition: opacity .2s ease;
  }
  .top-progress.show{ opacity: 1; }
  .top-progress .bar{
    height: 100%;
    width: 30%;
    background: linear-gradient(90deg, var(--accent), var(--accent2), var(--accent3));
    animation: move 1.1s linear infinite;
  }
  @keyframes move{ 0%{ transform: translateX(-40%);} 100%{ transform: translateX(340%);} }

  .toast-wrap{
    position: fixed; right: 18px; bottom: 18px;
    z-index: 2100;
    display:flex; flex-direction:column; gap: 10px;
    pointer-events:none;
  }
  .toastx{
    pointer-events:auto;
    min-width: 260px; max-width: 340px;
    background: rgba(255,255,255,0.92);
    border: 1px solid rgba(15,23,42,0.14);
    border-radius: 14px;
    box-shadow: var(--shadow);
    padding: 10px 12px;
    display:flex; gap: 10px; align-items:flex-start;
    animation: toastIn .22s ease-out;
  }
  @keyframes toastIn{ from{ transform: translateY(8px); opacity:0;} to{ transform: translateY(0); opacity:1;} }
  .toastx .badge{
    width: 10px; height: 10px; border-radius: 999px; margin-top: 6px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
  }
  .toastx .t-title{ font-weight: 950; font-size: 13px; margin:0; }
  .toastx .t-msg{ color: var(--muted); font-size: 12.5px; margin:2px 0 0 0; }

  footer{ max-width: 1260px; margin: 26px auto 18px auto; padding: 0 18px; color: rgba(15,23,42,0.55); font-size: 12px; }
  footer .f{ border-top: 1px solid rgba(15,23,42,0.12); padding-top: 12px; display:flex; justify-content:space-between; gap: 10px; flex-wrap: wrap; }
</style>
"""

BASE_NAV = """
<nav class="ukh-nav">
  <div class="ukh-container">
    <div class="ukh-brand">
      <img src="{{ logo_url }}" alt="UKH logo" class="ukh-logo" />
      <div class="ukh-title">
        <div class="ukh-uni">University of Kurdistan Hewlêr (UKH)</div>
        <div class="ukh-app">AI Teaching Assistant • Academic Proposal Prototype</div>
      </div>
    </div>
    <div class="ukh-links">
      <a href="/" class="ukh-link">Generate</a>
      <a href="/grade" class="ukh-link">Grade</a>
      <a href="/record" class="ukh-link">Record</a>
    </div>
  </div>
</nav>
"""

# -------------------------------------------------
# GENERATE PAGE (WOW) — includes auto-load from Record
# -------------------------------------------------
GEN_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>AI Teaching Assistant — Generate</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet" />
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css" rel="stylesheet">
  <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/jspdf@2.5.1/dist/jspdf.umd.min.js"></script>
  """ + BASE_CSS + """
</head>
<body>
  {{nav|safe}}
  <div class="page">

    <div class="top-progress" id="topProgress"><div class="bar"></div></div>
    <div class="toast-wrap" id="toastWrap"></div>

    <div class="hero">
      <div>
        <div class="pill"><span class="dot"></span>UKH • Academic Prototype</div>
        <h1>AI Teaching Assistant</h1>
        <p>Generate assessment-ready quizzes from lecture material, refine them interactively, and export to PDF/TXT — designed for institutional use.</p>
      </div>
      <div class="d-flex gap-2 flex-wrap justify-content-end">
        <div class="pill">Local STT + Local LLM</div>
        <div class="pill">Quiz • Grading • Summaries</div>
      </div>
    </div>

    <div class="grid-2">
      <div>
        <div class="u-card">
          <div class="u-card-header">
            <h3 class="u-card-title"><i class="bi bi-file-earmark-arrow-up me-2"></i>Inputs</h3>
            <div class="pill"><span class="dot"></span>PDF / PPTX</div>
          </div>
          <div class="u-card-body">
            <form id="quizForm" enctype="multipart/form-data">
              <div id="dropzone" class="dropzone mb-3">
                <div class="fw-bold">Drop or Click to Upload Lecture Notes</div>
                <div class="help mb-0">Supported: PDF, PPTX • Max 25MB</div>
                <input id="fileInput" type="file" name="lecture_file" accept=".pdf,.pptx" class="d-none" />
              </div>

              <div id="fileMeta" style="display:none;" class="mb-3">
                <span class="file-chip" id="fileChip"></span>
              </div>

              <div class="mb-3">
                <label class="form-label fw-semibold">Instruction (optional)</label>
                <textarea name="instruction" class="form-control" rows="4"
                  placeholder="Example: Create 10 questions (6 MCQ + 4 short-answer). Include Answer: lines."></textarea>
                <div class="help">If no file is uploaded, this text becomes the source content.</div>
              </div>

              <button type="submit" class="btn btn-primary w-100">
                <i class="bi bi-magic me-1"></i> Generate Quiz
              </button>
            </form>
          </div>
        </div>
      </div>

      <div>
        <div class="u-card" id="resultCard" style="display:none;">
          <div class="u-card-header">
            <h3 class="u-card-title"><i class="bi bi-journal-text me-2"></i>Generated Quiz</h3>
            <div class="pill"><span class="dot"></span>Editable</div>
          </div>
          <div class="u-card-body">
            <div id="typingBox" class="editor" contenteditable="true"></div>

            <div class="mt-3 d-flex gap-2 flex-wrap justify-content-end">
              <input type="text" id="followupInstruction" class="form-control" style="max-width:280px"
                     placeholder="Refine: harder, more MCQ, add explanations..." />
              <button class="btn btn-outline-light" onclick="regenerateQuiz()">
                <i class="bi bi-arrow-repeat me-1"></i> Regenerate
              </button>
              <button class="btn btn-outline-light" onclick="copyToClipboard()">
                <i class="bi bi-clipboard me-1"></i> Copy
              </button>
              <button class="btn btn-outline-light" onclick="downloadTXT()">
                <i class="bi bi-filetype-txt me-1"></i> TXT
              </button>
              <button class="btn btn-outline-light" onclick="downloadPDF()">
                <i class="bi bi-filetype-pdf me-1"></i> PDF
              </button>
            </div>

            <div class="help mt-3">
              Keep the “Answer:” lines for grading. Removing them disables automatic scoring on the Grade page.
            </div>
          </div>
        </div>
      </div>
    </div>

  </div>

  <footer>
    <div class="f">
      <span>© UKH — AI Teaching Assistant (Proposal Prototype)</span>
      <span>Quiz generation • Grading • Lecture summarization</span>
    </div>
  </footer>

<script>
let currentThreadId = "";

function showProgress(){ document.getElementById("topProgress").classList.add("show"); }
function hideProgress(){ document.getElementById("topProgress").classList.remove("show"); }

function toast(title, msg){
  const wrap = document.getElementById("toastWrap");
  const el = document.createElement("div");
  el.className = "toastx";
  el.innerHTML = `<div class="badge"></div><div><p class="t-title">${title}</p><p class="t-msg">${msg||""}</p></div>`;
  wrap.appendChild(el);
  setTimeout(()=>{ el.style.opacity = "0"; el.style.transform = "translateY(6px)"; }, 2600);
  setTimeout(()=>{ try{ wrap.removeChild(el);}catch(e){} }, 3100);
}

/* ✅ AUTO-LOAD quiz passed from Record page */
(function(){
  try {
    const q = localStorage.getItem("ukh_last_quiz_text") || "";
    const tid = localStorage.getItem("ukh_last_thread_id") || "";
    if (q.trim()) {
      showResultCard();
      $("#typingBox").text(q);
      currentThreadId = tid || "";
      toast("Loaded", "Quiz imported from Record page.");
      localStorage.removeItem("ukh_last_quiz_text");
      localStorage.removeItem("ukh_last_thread_id");
    }
  } catch(e) {}
})();

function showResultCard() {
  const el = document.getElementById("resultCard");
  if (el.style.display === "none") el.style.display = "block";
}

const dz = document.getElementById("dropzone");
const fi = document.getElementById("fileInput");
const fm = document.getElementById("fileMeta");

dz.addEventListener("click", ()=>fi.click());
dz.addEventListener("dragover",(e)=>{e.preventDefault();dz.classList.add("active");});
dz.addEventListener("dragleave",(e)=>{e.preventDefault();dz.classList.remove("active");});
dz.addEventListener("drop",(e)=>{
  e.preventDefault();dz.classList.remove("active");
  if (e.dataTransfer.files && e.dataTransfer.files.length) {
    fi.files = e.dataTransfer.files;
    fm.style.display = "block";
    document.getElementById("fileChip").innerText = "Selected: " + e.dataTransfer.files[0].name;
    toast("File selected", e.dataTransfer.files[0].name);
  }
});
fi.addEventListener("change",()=>{
  if (fi.files.length) {
    fm.style.display = "block";
    document.getElementById("fileChip").innerText = "Selected: " + fi.files[0].name;
    toast("File selected", fi.files[0].name);
  }
});

$("#quizForm").on("submit", function(e){
  e.preventDefault();
  const fd = new FormData(this);
  showResultCard();
  $("#typingBox").text("Generating quiz...");
  showProgress();

  $.ajax({
    url: "/generate_quiz",
    type: "POST",
    data: fd,
    processData: false,
    contentType: false,
    success: function(data){
      hideProgress();
      $("#typingBox").text(data.quiz_text || "No content.");
      currentThreadId = data.thread_id || "";
      toast("Quiz ready", "Edit, regenerate, copy, or export.");
    },
    error: function(xhr){
      hideProgress();
      $("#typingBox").text("Error: " + (xhr.responseText || "unknown"));
      toast("Error", "Please check server logs and try again.");
    }
  });
});

function regenerateQuiz(){
  const follow = $("#followupInstruction").val();
  const edited = $("#typingBox").text();
  if (!currentThreadId) {
    $("#typingBox").text("No active session. Generate first.");
    toast("No session", "Generate a quiz first.");
    return;
  }
  $("#typingBox").text("Regenerating...");
  showProgress();
  $.ajax({
    url: "/regenerate_quiz",
    type: "POST",
    contentType: "application/json",
    data: JSON.stringify({edited_quiz: edited, followup: follow, thread_id: currentThreadId}),
    success: function(data){
      hideProgress();
      $("#typingBox").text(data.quiz_text || "Empty result.");
      toast("Updated", "Quiz regenerated successfully.");
    },
    error: function(xhr){
      hideProgress();
      $("#typingBox").text("Error: " + (xhr.responseText || ""));
      toast("Error", "Regeneration failed.");
    }
  });
}

function copyToClipboard(){
  const t = $("#typingBox").text();
  navigator.clipboard.writeText(t);
  toast("Copied", "Quiz copied to clipboard.");
}

function downloadTXT(){
  const t = $("#typingBox").text();
  const blob = new Blob([t], {type: "text/plain"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "quiz.txt";
  a.click();
  toast("Download started", "quiz.txt");
}

async function downloadPDF(){
  const { jsPDF } = window.jspdf;
  const doc = new jsPDF();
  const text = $("#typingBox").text();
  const lines = doc.splitTextToSize(text, 180);
  let y = 10;
  lines.forEach(line=>{
    if (y > 280) { doc.addPage(); y = 10; }
    doc.text(line, 10, y);
    y += 8;
  });
  toast("Download started", "quiz.pdf");
  doc.save("quiz.pdf");
}
</script>
</body>
</html>
"""


RECORD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>AI Teaching Assistant — Record</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet" />
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css" rel="stylesheet">
  <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
  """ + BASE_CSS + """
  <style>
    .rec-dot {
      width: 10px; height: 10px; border-radius: 9999px; background: #ef4444;
      animation: pulse 1.2s infinite;
      display:inline-block;
      margin-right:6px;
    }
    @keyframes pulse {
      0% { transform: scale(1); opacity: 1; }
      50% { transform: scale(1.35); opacity: .5; }
      100% { transform: scale(1); opacity: 1; }
    }
  </style>
</head>
<body>
  {{nav|safe}}
  <div class="page">

    <div class="hero">
      <div>
        <div class="pill"><span class="dot"></span>Lecture Processing</div>
        <h1>Record & Summarize</h1>
        <p>Record a lecture (or upload audio), generate transcript, and produce a structured summary aligned with your notes.</p>
      </div>
      <div class="d-flex gap-2 flex-wrap justify-content-end">
        <div class="pill">Whisper STT</div>
        <div class="pill">LLM Summary</div>
      </div>
    </div>

    <div class="grid-2">
      <div>
        <div class="u-card mb-3">
          <div class="u-card-header">
            <h3 class="u-card-title"><i class="bi bi-file-earmark-text me-2"></i>Lecture Notes</h3>
          </div>
          <div class="u-card-body">
            <div class="help mb-2">Upload PDF/PPTX slides. Summary will use notes + transcript.</div>
            <form id="notesForm" enctype="multipart/form-data">
              <input type="file" name="notes_file" id="notesFile" accept=".pdf,.pptx" class="form-control mb-2">
              <button class="btn btn-outline-light w-100" type="submit"><i class="bi bi-upload me-1"></i>Upload Notes</button>
            </form>
            <div id="notesMsg" class="mt-2 small"></div>
          </div>
        </div>

        <div class="u-card mb-3">
          <div class="u-card-header">
            <h3 class="u-card-title"><i class="bi bi-mic-fill me-2"></i>Live Recording</h3>
          </div>
          <div class="u-card-body">
            <div class="help mb-3">Start when lecture begins. Stop to transcribe & summarize.</div>
            <div id="recStatus" class="small mb-3">Idle</div>
            <div class="d-flex gap-2">
              <button id="btnStart" class="btn btn-primary w-50"><i class="bi bi-record-fill me-1"></i>Start</button>
              <button id="btnStop" class="btn btn-outline-light w-50" disabled><i class="bi bi-stop-circle me-1"></i>Stop</button>
            </div>
            <div id="recMeta" class="mt-3 small"></div>
          </div>
        </div>

        <div class="u-card">
          <div class="u-card-header">
            <h3 class="u-card-title"><i class="bi bi-music-note-beamed me-2"></i>Upload Audio</h3>
          </div>
          <div class="u-card-body">
            <div class="help mb-2">Upload an existing recording to transcribe & summarize.</div>
            <form id="audioForm" enctype="multipart/form-data">
              <input type="file" name="audio_file" id="audioFile" accept="audio/*" class="form-control mb-2">
              <button class="btn btn-outline-light w-100" type="submit"><i class="bi bi-soundwave me-1"></i>Process Audio</button>
            </form>
            <div id="audioMsg" class="mt-2 small"></div>
          </div>
        </div>
      </div>

      <div>
        <div class="u-card mb-3">
          <div class="u-card-header">
            <h3 class="u-card-title">Transcript</h3>
            <div class="pill"><span class="dot"></span>Editable</div>
          </div>
          <div class="u-card-body">
            <textarea id="transcriptBox" class="form-control" rows="10" placeholder="Transcript will appear here..."></textarea>
            <div class="mt-3 d-flex gap-2 justify-content-end flex-wrap">
              <button class="btn btn-outline-light btn-sm" onclick="copyTranscript()">Copy</button>
              <button class="btn btn-primary btn-sm" onclick="sendToQuiz('transcript')">Generate Quiz</button>
            </div>
          </div>
        </div>

        <div class="u-card">
          <div class="u-card-header">
            <h3 class="u-card-title">Summary</h3>
            <div class="pill"><span class="dot"></span>Structured</div>
          </div>
          <div class="u-card-body">
            <textarea id="summaryBox" class="form-control" rows="10" placeholder="Summary will appear here..."></textarea>
            <div class="mt-3 d-flex gap-2 justify-content-end flex-wrap">
              <button class="btn btn-outline-light btn-sm" onclick="copySummary()">Copy</button>
              <button class="btn btn-primary btn-sm" onclick="sendToQuiz('summary')">Generate Quiz</button>
            </div>
            <div id="transcriptMsg" class="mt-2 small"></div>
          </div>
        </div>
      </div>
    </div>

  </div>

  <footer>
    <div class="f">
      <span>© UKH — AI Teaching Assistant (Proposal Prototype)</span>
      <span>Recording • Transcription • Summarization</span>
    </div>
  </footer>

<script>
$("#notesForm").on("submit", function(e){
  e.preventDefault();
  const fd = new FormData(this);
  $("#notesMsg").text("Uploading & extracting text...");
  $.ajax({
    url: "/record/upload_notes",
    type: "POST",
    data: fd,
    processData: false,
    contentType: false,
    success: function(data){ $("#notesMsg").text(data.message || "Notes uploaded."); },
    error: function(xhr){ $("#notesMsg").text("Error: " + (xhr.responseJSON?.message || xhr.responseText || "unknown")); }
  });
});

$("#btnStart").on("click", function(){
  $("#recStatus").text("Starting...");
  $.post("/record/start", {}, function(data){
    $("#recStatus").html('<span class="rec-dot"></span> Recording');
    $("#btnStart").prop("disabled", true);
    $("#btnStop").prop("disabled", false);
    $("#recMeta").text("Recording to: " + (data.filepath || "unknown"));
  }).fail(function(xhr){
    $("#recStatus").text("Error: " + (xhr.responseJSON?.message || xhr.responseText || "unknown"));
  });
});

$("#btnStop").on("click", function(){
  $("#recStatus").text("Transcribing & summarizing...");
  $("#btnStop").prop("disabled", true);
  $.post("/record/stop", {}, function(data){
    $("#recStatus").text("Completed");
    $("#btnStart").prop("disabled", false);
    $("#transcriptBox").val(data.transcript || "");
    $("#summaryBox").val(data.summary || "");
    $("#transcriptMsg").text("Transcript: " + (data.transcript_path || "") + " | Summary: " + (data.summary_path || ""));
  }).fail(function(xhr){
    $("#recStatus").text("Error: " + (xhr.responseJSON?.message || xhr.responseText || "unknown"));
    $("#btnStart").prop("disabled", false);
  });
});

$("#audioForm").on("submit", function(e){
  e.preventDefault();
  const fd = new FormData(this);
  $("#audioMsg").text("Uploading & processing audio...");
  $.ajax({
    url: "/record/upload_audio",
    type: "POST",
    data: fd,
    processData: false,
    contentType: false,
    success: function(data){
      $("#audioMsg").text("Completed.");
      $("#transcriptBox").val(data.transcript || "");
      $("#summaryBox").val(data.summary || "");
      $("#transcriptMsg").text("Transcript: " + (data.transcript_path || "") + " | Summary: " + (data.summary_path || ""));
    },
    error: function(xhr){
      $("#audioMsg").text("Error: " + (xhr.responseJSON?.message || xhr.responseText || "unknown"));
    }
  });
});

function copyTranscript(){ navigator.clipboard.writeText($("#transcriptBox").val() || ""); }
function copySummary(){ navigator.clipboard.writeText($("#summaryBox").val() || ""); }

function sendToQuiz(source){
  let text = (source === "summary") ? ($("#summaryBox").val() || "") : ($("#transcriptBox").val() || "");
  if (!text.trim()) { $("#transcriptMsg").text("No text available."); return; }
  $("#transcriptMsg").text("Generating quiz...");
  $.ajax({
    url: "/record/to_quiz",
    type: "POST",
    contentType: "application/json",
    data: JSON.stringify({ text: text }),
    success: function(data){
      try {
        localStorage.setItem("ukh_last_quiz_text", data.quiz_text || "");
        localStorage.setItem("ukh_last_thread_id", data.thread_id || "");
      } catch(e) {}
      $("#transcriptMsg").text("Quiz generated. Opening Generate page...");
      window.location.href = "/";
    },
    error: function(xhr){
      $("#transcriptMsg").text("Error: " + (xhr.responseJSON?.message || xhr.responseText || "unknown"));
    }
  });
}
</script>
</body>
</html>
"""


GRADE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>AI Teaching Assistant — Grade</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet" />
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css" rel="stylesheet">
  <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
  """ + BASE_CSS + """
</head>
<body>
  {{nav|safe}}
  <div class="page">

    <div class="hero">
      <div>
        <div class="pill"><span class="dot"></span>Assessment</div>
        <h1>Grading</h1>
        <p>Paste a quiz that includes the Answer key and automatically score responses (MCQ + short answers).</p>
      </div>
      <div class="d-flex gap-2 flex-wrap justify-content-end">
        <div class="pill">Auto Scoring</div>
        <div class="pill">Feedback Table</div>
      </div>
    </div>

    <div class="grid-2">
      <div>
        <div class="u-card">
          <div class="u-card-header">
            <h3 class="u-card-title"><i class="bi bi-upload me-2"></i>Step 1 — Load Quiz</h3>
          </div>
          <div class="u-card-body">
            <form id="gradePrepareForm" enctype="multipart/form-data">
              <div class="mb-3">
                <label class="form-label fw-semibold">Lecture note (optional)</label>
                <input id="gFile" type="file" name="lecture_file" accept=".pdf,.pptx" class="form-control" />
              </div>
              <div class="mb-3">
                <label class="form-label fw-semibold">Quiz text</label>
                <textarea id="quizText" name="quiz_text" rows="12" class="form-control"
                          placeholder="Paste the generated quiz here (must include Answer: lines)..."></textarea>
              </div>
              <button class="btn btn-primary w-100" type="submit">
                <i class="bi bi-arrow-right-circle me-1"></i> Load Questions
              </button>
            </form>
            <div id="gUploadMsg" class="mt-2 small"></div>
          </div>
        </div>
      </div>

      <div>
        <div class="u-card">
          <div class="u-card-header">
            <h3 class="u-card-title"><i class="bi bi-pencil-square me-2"></i>Step 2 — Answer</h3>
          </div>
          <div class="u-card-body">
            <div id="takeArea"></div>
            <button id="gradeBtn" class="btn btn-primary mt-3" style="display:none;">
              <i class="bi bi-check2-circle me-1"></i> Grade
            </button>
            <div id="gradeSummary" class="mt-3"></div>
          </div>
        </div>
      </div>
    </div>

  </div>

  <footer>
    <div class="f">
      <span>© UKH — AI Teaching Assistant (Proposal Prototype)</span>
      <span>Grading • Feedback</span>
    </div>
  </footer>

<script>
let __sid = null;

$("#gradePrepareForm").on("submit", function(e){
  e.preventDefault();
  const fd = new FormData(this);
  const q = ($("#quizText").val() || "").trim();
  if (!q) { $("#gUploadMsg").text("Paste quiz first."); return; }
  $("#gUploadMsg").text("Parsing quiz...");
  $.ajax({
    url: "/grade/prepare",
    type: "POST",
    data: fd,
    processData: false,
    contentType: false,
    success: function(data){
      __sid = data.sid;
      renderQuestions(data.questions || []);
      $("#gUploadMsg").text("Questions loaded.");
    },
    error: function(xhr){
      $("#gUploadMsg").text(xhr.responseJSON?.message || "Error");
    }
  });
});

function renderQuestions(qs){
  const c = $("#takeArea");
  c.empty();
  if (!qs.length) { c.text("No questions detected."); return; }

  qs.forEach((q, idx)=>{
    const wrap = $('<div class="mb-3 p-3" style="border:1px solid rgba(15,23,42,0.14); border-radius:14px; background: rgba(255,255,255,0.72);"></div>');
    wrap.append(`<div class="mb-2"><strong>Q${idx+1} (${q.type})</strong><div class="mt-1">${q.question}</div></div>`);
    if ((q.type||"").toLowerCase()==="mcq") {
      (q.options||[]).forEach(opt=>{
        wrap.append(`<label class="d-flex gap-2 mb-1"><input type="radio" name="qq${idx}" value="${opt}"> <span>${opt}</span></label>`);
      });
    } else {
      wrap.append(`<textarea data-i="${idx}" class="form-control" rows="2" placeholder="Your answer..."></textarea>`);
    }
    c.append(wrap);
  });

  $("#gradeBtn").show().off("click").on("click", submitForGrading);
}

function collectAnswers(){
  const res = {};
  $("#takeArea input[type='radio']:checked").each(function(){
    const name = $(this).attr("name");
    const idx = name.replace("qq","");
    res[idx] = $(this).val();
  });
  $("#takeArea textarea[data-i]").each(function(){
    const idx = $(this).attr("data-i");
    res[idx] = $(this).val();
  });
  return res;
}

function submitForGrading(){
  if (!__sid) { alert("No session"); return; }
  const ans = collectAnswers();
  $.ajax({
    url: "/grade/submit",
    type: "POST",
    contentType: "application/json",
    data: JSON.stringify({sid: __sid, answers: ans}),
    success: function(data){
      let html = `<div class="alert" style="background: rgba(255,255,255,0.86); border:1px solid rgba(15,23,42,0.14); color: #0b1220;">
                    <strong>Score:</strong> ${data.correct} / ${data.total} (${data.percent}%)
                  </div>`;
      html += `<div class="table-responsive">
               <table class="table table-sm align-middle">
               <thead><tr><th>#</th><th>Type</th><th>Your</th><th>Expected</th><th>OK</th></tr></thead><tbody>`;
      (data.details||[]).forEach(r=>{
        html += `<tr><td>${r.i}</td><td>${r.type}</td><td>${escapeHtml(r.your||"")}</td><td>${escapeHtml(r.expected||"")}</td><td>${r.ok?"✅":"❌"}</td></tr>`;
      });
      html += `</tbody></table></div>`;
      $("#gradeSummary").html(html);
    },
    error: function(xhr){
      $("#gradeSummary").text(xhr.responseJSON?.message || "Error");
    }
  });
}

function escapeHtml(text){
  return (text||"").replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;");
}
</script>
</body>
</html>
"""


@app.route("/")
def home():
    nav_html = render_template_string(BASE_NAV, logo_url=_logo_url())
    return render_template_string(GEN_HTML, nav=nav_html)


@app.route("/grade")
def grade_page():
    nav_html = render_template_string(BASE_NAV, logo_url=_logo_url())
    return render_template_string(GRADE_HTML, nav=nav_html)


@app.route("/record")
def record_page():
    nav_html = render_template_string(BASE_NAV, logo_url=_logo_url())
    return render_template_string(RECORD_HTML, nav=nav_html)



@app.route("/generate_quiz", methods=["POST"])
def generate_quiz():
    try:
        file = request.files.get("lecture_file")
        instruction = (request.form.get("instruction") or "").strip()
        lecture_text = ""
        filepath = ""

        if file and file.filename:
            ext = os.path.splitext(file.filename)[1].lower()
            if ext not in ALLOWED_EXTS:
                return jsonify({"quiz_text": "⚠️ Unsupported file type.", "thread_id": ""})
            base = os.path.splitext(secure_filename(file.filename))[0]
            unique_name = f"{base}_{uuid.uuid4().hex[:6]}{ext}"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
            file.save(filepath)
            lecture_text = extract_text(filepath)

        if not lecture_text:
            lecture_text = instruction

        if not lecture_text:
            return jsonify({"quiz_text": "⚠️ Please upload a lecture note or enter an instruction.", "thread_id": ""})

        quiz_text, thread_id = generate_quiz_stream(
            lecture_text,
            instruction or "Create a quiz from this lecture.",
            filepath,
        )
        return jsonify({"quiz_text": quiz_text, "thread_id": thread_id})
    except Exception as e:
        return jsonify({"quiz_text": f"⚠️ Server error: {e}", "thread_id": ""})


@app.route("/regenerate_quiz", methods=["POST"])
def regenerate_quiz():
    try:
        data = request.get_json(force=True)
        edited_quiz = data.get("edited_quiz", "")
        followup = data.get("followup", "")
        thread_id = data.get("thread_id", "")
        quiz_text = regenerate_quiz_stream(edited_quiz, followup, thread_id)
        return jsonify({"quiz_text": quiz_text})
    except Exception as e:
        return jsonify({"quiz_text": f"⚠️ Server error: {e}"}), 500



@app.route("/grade/prepare", methods=["POST"])
def grade_prepare():
    try:
        file = request.files.get("lecture_file")
        if file and file.filename:
            ext = os.path.splitext(file.filename)[1].lower()
            if ext in ALLOWED_EXTS:
                base = os.path.splitext(secure_filename(file.filename))[0]
                unique = f"{base}_{uuid.uuid4().hex[:6]}{ext}"
                fpath = os.path.join(app.config["UPLOAD_FOLDER"], unique)
                file.save(fpath)
                _ = extract_text(fpath)

        quiz_text = (request.form.get("quiz_text") or "").strip()
        if not quiz_text:
            return jsonify({"message": "Paste quiz text first"}), 400

        struct = _parse_quiz_text_to_struct(quiz_text)
        if not struct["questions"]:
            return jsonify({"message": "Could not detect questions"}), 400

        sid = uuid.uuid4().hex
        SESSIONS[sid] = struct
        return jsonify({"sid": sid, "questions": struct["questions"]})
    except Exception as e:
        return jsonify({"message": f"Server error: {e}"}), 500


@app.route("/grade/submit", methods=["POST"])
def grade_submit():

    try:
        data = request.get_json(force=True)
        sid = data.get("sid")
        ans = data.get("answers", {}) or {}

        if not sid or sid not in SESSIONS:
            return jsonify({"message": "Invalid sid"}), 400

        sess = SESSIONS[sid]
        result = grade_quiz(sess, ans)

        return jsonify(result)
    except Exception as e:
        return jsonify({"message": f"Server error: {e}"}), 500


# -------------------------------------------------
# ROUTES: RECORDING / TRANSCRIBE / SUMMARIZE
# -------------------------------------------------
@app.route("/record/upload_notes", methods=["POST"])
def record_upload_notes():
    global CURRENT_LECTURE_NOTES_TEXT
    try:
        file = request.files.get("notes_file")
        if not file or not file.filename:
            return jsonify({"message": "No file provided."}), 400
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXTS:
            return jsonify({"message": "Unsupported file type."}), 400

        base = os.path.splitext(secure_filename(file.filename))[0]
        unique = f"{base}_{uuid.uuid4().hex[:6]}{ext}"
        path = os.path.join(UPLOAD_FOLDER, unique)
        file.save(path)
        text = extract_text(path)
        CURRENT_LECTURE_NOTES_TEXT = text or ""
        return jsonify({"message": "Lecture notes uploaded & extracted."})
    except Exception as e:
        return jsonify({"message": f"Failed to upload notes: {e}"}), 500


@app.route("/record/start", methods=["POST"])
def record_start():
    try:
        fp = RECORDER.start()
        return jsonify({"status": "recording", "filepath": fp})
    except Exception as e:
        return jsonify({"message": f"Failed to start recording: {e}"}), 500


@app.route("/record/stop", methods=["POST"])
def record_stop():
    try:
        audio_path = RECORDER.stop()
        if not audio_path:
            return jsonify({"message": "Not recording."}), 400

        model = get_whisper()
        result = model.transcribe(audio_path, language="en", fp16=False)
        text = (result.get("text") or "").strip()

        base = os.path.splitext(os.path.basename(audio_path))[0]
        transcript_path = os.path.join(TRANSCRIPTS_FOLDER, f"{base}.txt")
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(text)

        lecture_text = CURRENT_LECTURE_NOTES_TEXT or ""
        summary = summarize_lecture(text, lecture_text)

        summary_path = os.path.join(SUMMARIES_FOLDER, f"{base}_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)

        return jsonify({
            "status": "stopped",
            "audio_path": audio_path,
            "transcript": text,
            "transcript_path": transcript_path,
            "summary": summary,
            "summary_path": summary_path,
        })
    except Exception as e:
        return jsonify({"message": f"Failed to stop/transcribe: {e}"}), 500


@app.route("/record/upload_audio", methods=["POST"])
def record_upload_audio():
    try:
        file = request.files.get("audio_file")
        if not file or not file.filename:
            return jsonify({"message": "No audio file provided."}), 400

        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_AUDIO_EXTS:
            return jsonify({"message": "Unsupported audio type."}), 400

        base_name = os.path.splitext(secure_filename(file.filename))[0]
        unique = f"{base_name}_{uuid.uuid4().hex[:6]}{ext}"
        audio_path = os.path.join(RECORDINGS_FOLDER, unique)
        file.save(audio_path)

        model = get_whisper()
        result = model.transcribe(audio_path, language="en", fp16=False)
        text = (result.get("text") or "").strip()

        base = os.path.splitext(os.path.basename(audio_path))[0]
        transcript_path = os.path.join(TRANSCRIPTS_FOLDER, f"{base}.txt")
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(text)

        lecture_text = CURRENT_LECTURE_NOTES_TEXT or ""
        summary = summarize_lecture(text, lecture_text)

        summary_path = os.path.join(SUMMARIES_FOLDER, f"{base}_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)

        return jsonify({
            "audio_path": audio_path,
            "transcript": text,
            "transcript_path": transcript_path,
            "summary": summary,
            "summary_path": summary_path,
        })
    except Exception as e:
        return jsonify({"message": f"Failed to process audio: {e}"}), 500


@app.route("/record/to_quiz", methods=["POST"])
def record_to_quiz():
    try:
        data = request.get_json(force=True, silent=True) or {}
        text = (data.get("text") or "").strip()
        if not text:
            return jsonify({"message": "No text provided."}), 400

        quiz_text, thread_id = generate_quiz_stream(
            text,
            "Create a 10-question quiz (MCQ + short answers) from this lecture. Include Answer: lines.",
            None,
        )
        return jsonify({"quiz_text": quiz_text, "thread_id": thread_id})
    except Exception as e:
        return jsonify({"message": f"Failed to generate from text: {e}"}), 500


_PNG_1PX = base64.b64decode(
    b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Yb9d7kAAAAASUVORK5CYII="
)
_SVG_FAVICON = """<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'>
  <defs><linearGradient id='g' x1='0' x2='1' y1='0' y2='1'>
    <stop stop-color='#1d4ed8'/><stop offset='1' stop-color='#0ea5e9'/></linearGradient></defs>
  <rect width='64' height='64' rx='12' fill='#0f172a'/>
  <path d='M32 10l6.6 13.4 14.8 2.1-10.7 10.4 2.5 14.7L32 43.8 18.8 50.6l2.5-14.7L10.6 25.5l14.8-2.1L32 10z' fill='url(#g)'/>
</svg>"""


@app.route("/favicon.ico")
def favicon():
    return Response(_SVG_FAVICON, mimetype="image/svg+xml")


@app.route("/apple-touch-icon.png")
def apple_touch_icon():
    return Response(_PNG_1PX, mimetype="image/png")


if __name__ == "__main__":
    port = 7860
    print(f"🚀 App running at: http://127.0.0.1:{port}")
    print("✅ Logo check: http://127.0.0.1:7860/ukh-logo")
    app.run(debug=False, port=port, host="0.0.0.0", use_reloader=False)