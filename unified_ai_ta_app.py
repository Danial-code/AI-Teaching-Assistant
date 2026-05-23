import os, uuid, re, traceback, base64
from flask import Flask, render_template_string, request, jsonify, Response, send_file, url_for
from werkzeug.utils import secure_filename
from main_ollama import extract_text, generate_quiz_stream, regenerate_quiz_stream, summarize_lecture
from grading import grade_quiz
import whisper
from langchain_ollama import OllamaLLM

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
ALLOWED_DOC_EXTS = {".pdf", ".pptx"}
ALLOWED_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".webm"}

LLM_MODEL_NAME = os.getenv("LECTURE_MODEL", "llama3.1")
CURRENT_LECTURE = {"text": "", "filename": ""}

print("🔁 Loading Whisper model (base)...")
whisper_model = whisper.load_model("base")
print("✅ Whisper ready.")
print(f"🔁 Connecting to Ollama model: {LLM_MODEL_NAME} ...")
llm = OllamaLLM(model=LLM_MODEL_NAME)
print("✅ Llama (via Ollama) ready.")

def summarize_transcript_with_context(transcript: str, lecture_text: str = "") -> str:
    lecture_context = (lecture_text or "").strip()
    transcript = (transcript or "").strip()
    prompt = f"""
You are an AI Teaching Assistant for MSc student Danial.

You will receive:
1) A lecture transcript (spoken audio converted to text, may be noisy)
2) Optional lecture notes content (text extracted from slides / PDF)

Your job is to produce clean, well-structured student notes.

LECTURE NOTES:
{lecture_context}

TRANSCRIPT:
{transcript}
"""
    try:
        return llm.invoke(prompt)
    except Exception as e:
        return f"⚠️ Failed to summarize: {type(e).__name__}: {e}"

HTML_DASHBOARD = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>UKH AI Teaching Assistant</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<style>
:root{--navy:#0b1c2d;--panel:#0f243a;--border:rgba(255,255,255,.12);--text:#f5f7fb;--muted:#a8b2c3;--accent:#2563eb}
*{font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto}
body{background:var(--navy);color:var(--text)}
.header-wrap{max-width:1100px;margin:40px auto 30px;background:rgba(255,255,255,.06);border:1px solid var(--border);border-radius:22px;padding:40px 30px;box-shadow:0 25px 60px rgba(0,0,0,.28);text-align:center}
.ukh-logo{max-width:320px;margin-bottom:18px}
.header-title{font-weight:900;font-size:30px;letter-spacing:.3px}
.header-sub{color:var(--muted);margin-top:6px;font-weight:600}
.navbar-custom{display:flex;justify-content:center;gap:44px;margin-top:28px;font-weight:800;font-size:16px}
.navbar-custom span{cursor:pointer;padding-bottom:8px;border-bottom:4px solid transparent;color:rgba(245,247,251,.9)}
.navbar-custom span.active{border-color:var(--accent);color:#fff}
.page{display:none;max-width:1100px;margin:0 auto 60px}
.page.active{display:block}
.card-box{background:rgba(255,255,255,.06);border:1px solid var(--border);border-radius:20px;padding:30px;box-shadow:0 18px 45px rgba(0,0,0,.20);margin-bottom:30px}
.output{white-space:pre-wrap;background:rgba(255,255,255,.05);border-radius:14px;padding:18px;min-height:220px;font-size:14px;border:1px solid var(--border);color:#eef2ff}
.form-control{background:rgba(255,255,255,.06)!important;border:1px solid var(--border)!important;color:#fff!important}
.form-control::placeholder{color:rgba(168,178,195,.9)}
.btn-main{background:var(--accent);border:none;font-weight:900}
.btn-main:hover{background:#1d4ed8}
.smallmuted{color:var(--muted);font-size:12px}
h5{color:#fff}
.table{background:transparent}
.table td,.table th{color:#0f172a}
</style>
</head>
<body>

<div class="header-wrap">
  <img src="/ukh-logo" class="ukh-logo" alt="UKH" onerror="this.style.display='none'">
  <div class="header-title">AI Teaching Assistant</div>
  <div class="header-sub">University of Kurdistan Hewlêr</div>
  <div class="navbar-custom">
    <span class="tab active" data-target="quiz">Quiz Generator</span>
    <span class="tab" data-target="grade">Quiz Grader</span>
    <span class="tab" data-target="live">Live Class</span>
    <span class="tab" data-target="audio">Audio Summary</span>
  </div>
</div>

<div id="quiz" class="page active">
  <div class="card-box">
    <div class="row g-3">
      <div class="col-lg-5">
        <h5 class="mb-3">Quiz Generator</h5>
        <form id="quizForm" enctype="multipart/form-data">
          <input id="quizFile" type="file" name="lecture_file" class="form-control mb-3" accept=".pdf,.pptx" required>
          <label class="smallmuted mb-1">Quiz instruction (optional)</label>
          <textarea name="instruction" class="form-control mb-3" rows="3" placeholder="Optional: specify focus areas (e.g., definitions and examples)."></textarea>
          <button class="btn btn-main w-100 text-white" type="submit">Generate Quiz</button>
        </form>
        <div id="quizStatus" class="smallmuted mt-2"></div>
        <hr class="my-4" style="border-color:rgba(255,255,255,.12)">
        <label class="smallmuted mb-1">Edit instruction (optional)</label>
        <input id="followupInstruction" class="form-control mb-2" placeholder="Optional: describe what you want to adjust.">
        <button class="btn btn-outline-light w-100" type="button" onclick="regen()">Regenerate</button>
      </div>
      <div class="col-lg-7">
        <div class="d-flex justify-content-between align-items-center">
          <h5 class="mb-0">Quiz Output</h5>
          <div class="d-flex gap-2">
            <button class="btn btn-sm btn-outline-light" type="button" onclick="copyQuiz()">Copy</button>
            <button class="btn btn-sm btn-outline-light" type="button" onclick="downloadQuiz()">TXT</button>
          </div>
        </div>
        <div id="quizOut" class="output mt-3"></div>
      </div>
    </div>
  </div>
</div>

<div id="grade" class="page">
  <div class="card-box">
    <div class="row g-3">
      <div class="col-lg-5">
        <h5 class="mb-3">Quiz Grader</h5>
        <form id="gradePrepareForm" enctype="multipart/form-data">
          <input id="gFile" type="file" name="lecture_file" class="form-control mb-3" accept=".pdf,.pptx">
          <textarea id="quizText" name="quiz_text" class="form-control mb-3" rows="10" placeholder="Paste quiz here (must include Answer:)"></textarea>
          <button class="btn btn-main w-100 text-white" type="submit">Load Questions</button>
        </form>
        <div id="gUploadMsg" class="smallmuted mt-2"></div>
      </div>
      <div class="col-lg-7">
        <div class="d-flex justify-content-between align-items-center">
          <h5 class="mb-0">Answer Sheet</h5>
          <button id="gradeBtn" class="btn btn-main text-white btn-sm" style="display:none;">Grade</button>
        </div>
        <div id="takeArea" class="mt-3"></div>
        <div id="gradeSummaryBox" class="mt-3"></div>
      </div>
    </div>
  </div>
</div>

<div id="live" class="page">
  <div class="card-box">
    <div class="row g-3">
      <div class="col-lg-5">
        <h5 class="mb-3">Live Class</h5>
        <form id="liveLectureForm" enctype="multipart/form-data">
          <input id="liveLectureFile" type="file" name="lecture_file" class="form-control mb-3" accept=".pdf,.pptx">
          <button class="btn btn-outline-light w-100" type="submit">Load Notes</button>
        </form>
        <div id="liveLectureStatus" class="smallmuted mt-2"></div>
        <hr class="my-4" style="border-color:rgba(255,255,255,.12)">
        <div class="d-flex gap-2">
          <button id="liveStartBtn" class="btn btn-main w-50 text-white" type="button">Start</button>
          <button id="liveStopBtn" class="btn btn-outline-light w-50" type="button" disabled>Stop</button>
        </div>
        <div id="liveStatus" class="smallmuted mt-2"></div>
      </div>
      <div class="col-lg-7">
        <h5 class="mb-2">Transcript</h5>
        <div id="liveTranscript" class="output"></div>
        <h5 class="mt-4 mb-2">Summary</h5>
        <div id="liveSummary" class="output"></div>
      </div>
    </div>
  </div>
</div>

<div id="audio" class="page">
  <div class="card-box">
    <div class="row g-3">
      <div class="col-lg-5">
        <h5 class="mb-3">Audio Summary</h5>
        <form id="audioForm" enctype="multipart/form-data">
          <input id="audioLectureFile" type="file" name="lecture_file" class="form-control mb-3" accept=".pdf,.pptx">
          <input id="audioFile" type="file" name="audio_file" class="form-control mb-3" accept="audio/*,.webm">
          <button class="btn btn-main w-100 text-white" type="submit">Transcribe & Summarize</button>
        </form>
        <div id="audioStatus" class="smallmuted mt-2"></div>
      </div>
      <div class="col-lg-7">
        <h5 class="mb-2">Transcript</h5>
        <div id="audioTranscript" class="output"></div>
        <h5 class="mt-4 mb-2">Summary</h5>
        <div id="audioSummary" class="output"></div>
      </div>
    </div>
  </div>
</div>

<script>
let threadId=""
$(".tab").on("click",function(){
  $(".tab").removeClass("active");$(".page").removeClass("active");
  $(this).addClass("active");$("#"+$(this).data("target")).addClass("active");
  if($(this).data("target")==="grade"){
    const curQuiz = ($("#quizOut").text()||"").trim();
    const box = $("#quizText");
    if(curQuiz && box.length && !box.val().trim()){
      box.val(curQuiz);
      $("#gUploadMsg").text("Quiz pasted from generator. Click Load Questions.");
    }
  }
})
try{const t=localStorage.getItem("ukh_tid")||"";if(t.trim()) threadId=t}catch(e){}
$("#quizForm").on("submit",function(e){
  e.preventDefault()
  threadId=""
  try{localStorage.removeItem("ukh_quiz");localStorage.removeItem("ukh_tid")}catch(e){}
  $("#quizOut").text("")
  $("#quizStatus").text("Generating...")
  let fd=new FormData(this)
  $.ajax({url:"/generate_quiz",type:"POST",data:fd,processData:false,contentType:false,
    success:d=>{
      threadId=d.thread_id||""
      $("#quizOut").text(d.quiz_text||"")
      $("#quizStatus").text("Done")
      try{localStorage.setItem("ukh_tid",threadId);localStorage.setItem("ukh_quiz",$("#quizOut").text()||"")}catch(e){}
    },
    error:x=>{
      $("#quizOut").text(x.responseJSON?.quiz_text||x.responseText||"Error")
      $("#quizStatus").text("Failed")
    }
  })
})
$("#quizFile").on("change",function(){
  threadId=""
  try{localStorage.removeItem("ukh_quiz");localStorage.removeItem("ukh_tid")}catch(e){}
  $("#quizOut").text("")
  $("#quizStatus").text("")
})
function regen(){
  const f=$("#followupInstruction").val()||""
  const q=$("#quizOut").text()||""
  if(!f.trim()){$("#quizStatus").text("Enter an edit instruction to refine the quiz.");return}
  if(!q.trim()){$("#quizStatus").text("Generate a quiz first.");return}
  if(!threadId){try{threadId=localStorage.getItem("ukh_tid")||""}catch(e){threadId=""}}
  if(!threadId){$("#quizStatus").text("No session. Generate again.");return}
  $("#quizStatus").text("Updating...")
  $.ajax({url:"/regenerate_quiz",type:"POST",contentType:"application/json",
    data:JSON.stringify({edited_quiz:q,followup:f,thread_id:threadId}),
    success:d=>{
      $("#quizOut").text(d.quiz_text||"")
      $("#quizStatus").text("Updated")
      try{localStorage.setItem("ukh_quiz",$("#quizOut").text()||"");localStorage.setItem("ukh_tid",threadId)}catch(e){}
    },
    error:x=>{
      $("#quizOut").text(x.responseJSON?.quiz_text||x.responseText||"Error")
      $("#quizStatus").text("Failed")
    }
  })
}
function copyQuiz(){navigator.clipboard.writeText($("#quizOut").text()||"")}
function downloadQuiz(){const t=$("#quizOut").text()||"";const blob=new Blob([t],{type:"text/plain"});const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="quiz.txt";a.click()}

let __sid=null
$("#gradePrepareForm").on("submit",function(e){
  e.preventDefault()
  const fd=new FormData(this)
  if(!($("#quizText").val()||"").trim()){ $("#gUploadMsg").text("Paste quiz text first."); return }
  $("#gUploadMsg").text("Parsing...")
  $.ajax({url:"/grade/prepare",type:"POST",data:fd,processData:false,contentType:false,
    success:d=>{__sid=d.sid;renderQuestions(d.questions||[]);$("#gUploadMsg").text("Loaded")},
    error:x=>{$("#gUploadMsg").text(x.responseJSON?.message||"Error")}
  })
})
function renderQuestions(qs){
  const c=$("#takeArea"); c.empty()
  if(!qs.length){c.text("No questions detected.");return}
  qs.forEach((q,i)=>{
    const w=$("<div class='mb-3 p-3 border rounded' style='border-color:rgba(255,255,255,.12)!important;background:rgba(255,255,255,.03)'></div>")
    w.append(`<div class='mb-2'><b>Q${i+1} (${q.type})</b> ${q.question}</div>`)
    if((q.type||"").toLowerCase()==="mcq"){
      (q.options||[]).forEach(opt=>{w.append(`<label class='d-flex gap-2 mb-1'><input type='radio' name='qq${i}' value='${opt}'> <span>${opt}</span></label>`)})
    }else{
      w.append(`<textarea class='form-control' data-i='${i}' rows='2' placeholder='Your answer...'></textarea>`)
    }
    c.append(w)
  })
  $("#gradeBtn").show().off("click").on("click",submitForGrading)
}
function collectAnswers(){
  const r={}
  $("#takeArea input[type='radio']:checked").each(function(){r[$(this).attr("name").replace("qq","")]=$(this).val()})
  $("#takeArea textarea[data-i]").each(function(){r[$(this).attr("data-i")]=$(this).val()})
  return r
}
function submitForGrading(){
  if(!__sid)return
  const ans=collectAnswers()
  $("#gradeSummaryBox").html("<div class='text-white-50'>Grading...</div>")
  $.ajax({
    url:"/grade/submit",
    type:"POST",
    dataType:"json",
    contentType:"application/json",
    data:JSON.stringify({sid:__sid,answers:ans}),
    success:d=>{
      try{
        let h=`<div class='alert alert-light border'><b>Score:</b> ${d.correct}/${d.total} (${d.percent}%)</div>`
        h+=`<div class='table-responsive'><table class='table table-sm'><thead><tr><th>#</th><th>Type</th><th>Your Answer</th><th>Expected</th><th>Feedback</th><th>OK</th></tr></thead><tbody>`
        ;(d.details||[]).forEach((r,idx)=>{
          const your = (r.answer ?? r.your ?? "").toString()
          const exp  = (r.expected ?? "").toString()
          const note = (r.note ?? r.feedback ?? r.rationale ?? "").toString()
          const okVal = (r.correct ?? r.ok ?? false)
          const fb = note ? note : (okVal ? "Correct" : "Review the expected answer")
          h+=`<tr><td>${idx+1}</td><td>${r.type||""}</td><td>${your}</td><td>${exp}</td><td>${fb}</td><td>${okVal?"✅":"❌"}</td></tr>`
        })
        h+=`</tbody></table></div>`
        $("#gradeSummaryBox").html(h)
      }catch(err){
        $("#gradeSummaryBox").html("<div class='text-danger'>Render error: "+err+"</div>")
      }
    },
    error:x=>{
      const msg = (x.responseJSON && x.responseJSON.message) ? x.responseJSON.message : (x.responseText||"Error")
      $("#gradeSummaryBox").html("<div class='text-danger'>"+msg+"</div>")
    }
  })
}

let mediaRecorder=null,recordedChunks=[]
$("#liveLectureForm").on("submit",function(e){
  e.preventDefault()
  const fd=new FormData(this)
  $("#liveLectureStatus").text("Uploading...")
  $.ajax({url:"/live/upload_lecture",type:"POST",data:fd,processData:false,contentType:false,
    success:d=>$("#liveLectureStatus").text(d.message||"Loaded"),
    error:x=>$("#liveLectureStatus").text(x.responseJSON?.message||"Error")
  })
})
$("#liveStartBtn").on("click",async function(){
  try{
    const stream=await navigator.mediaDevices.getUserMedia({audio:true})
    recordedChunks=[]
    mediaRecorder=new MediaRecorder(stream)
    mediaRecorder.ondataavailable=e=>{if(e.data.size>0)recordedChunks.push(e.data)}
    mediaRecorder.start()
    $("#liveStatus").text("Recording...")
    $("#liveStartBtn").prop("disabled",true)
    $("#liveStopBtn").prop("disabled",false)
  }catch(err){$("#liveStatus").text("Microphone error: "+err)}
})
$("#liveStopBtn").on("click",function(){
  if(!mediaRecorder)return
  mediaRecorder.onstop=()=>{
    const blob=new Blob(recordedChunks,{type:"audio/webm"})
    const fd=new FormData(); fd.append("audio_file",blob,"live_recording.webm")
    $("#liveStatus").text("Uploading and summarizing...")
    $("#liveTranscript").text(""); $("#liveSummary").text("Summarizing...")
    $.ajax({url:"/live/stop_and_summarize",type:"POST",data:fd,processData:false,contentType:false,
      success:d=>{$("#liveTranscript").text(d.transcript||"");$("#liveSummary").text(d.summary||"");$("#liveStatus").text("Done")},
      error:x=>{$("#liveSummary").text(x.responseJSON?.message||"Error");$("#liveStatus").text("Failed")}
    })
  }
  mediaRecorder.stop()
  $("#liveStartBtn").prop("disabled",false)
  $("#liveStopBtn").prop("disabled",true)
})
$("#audioForm").on("submit",function(e){
  e.preventDefault()
  const fd=new FormData(this)
  if(!$("#audioFile")[0].files.length){$("#audioStatus").text("Choose an audio file.");return}
  $("#audioStatus").text("Processing...")
  $("#audioTranscript").text(""); $("#audioSummary").text("Summarizing...")
  $.ajax({url:"/audio/summarize",type:"POST",data:fd,processData:false,contentType:false,
    success:d=>{$("#audioTranscript").text(d.transcript||"");$("#audioSummary").text(d.summary||"");$("#audioStatus").text("Done")},
    error:x=>{$("#audioSummary").text(x.responseJSON?.message||"Error");$("#audioStatus").text("Failed")}
  })
})
</script>
</body>
</html>
"""

_PNG_1PX = base64.b64decode(b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Yb9d7kAAAAASUVORK5CYII=")
_SVG_FAVICON = "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'><rect width='64' height='64' rx='12' fill='#0b1c2d'/><path d='M18 44h28v4H18z' fill='#fff'/></svg>"

@app.route("/favicon.ico")
def favicon():
    return Response(_SVG_FAVICON, mimetype="image/svg+xml")

@app.route("/apple-touch-icon.png")
def apple_touch_icon():
    return Response(_PNG_1PX, mimetype="image/png")

@app.route("/")
def dashboard():
    return render_template_string(HTML_DASHBOARD, llm_model=LLM_MODEL_NAME)

@app.route("/generate_quiz", methods=["POST"])
def generate_quiz():
    try:
        file = request.files.get("lecture_file")
        instruction = (request.form.get("instruction") or "").strip()
        if not file or not file.filename:
            return jsonify({"quiz_text": "⚠️ No lecture file provided. Upload PDF or PPTX.", "thread_id": ""}), 400
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_DOC_EXTS:
            return jsonify({"quiz_text": "⚠️ Unsupported file type. Use .pdf or .pptx", "thread_id": ""}), 400
        base = os.path.splitext(secure_filename(file.filename))[0]
        unique_name = f"{base}_{uuid.uuid4().hex[:6]}{ext}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
        file.save(filepath)
        lecture_text = extract_text(filepath)
        if not lecture_text or lecture_text.startswith("⚠️"):
            lecture_text = instruction
        if not lecture_text:
            return jsonify({"quiz_text": "⚠️ Could not read lecture. Try different file or add instruction.", "thread_id": ""}), 400
        quiz_text, thread_id = generate_quiz_stream(
            lecture_text,
            instruction or "Create a quiz that covers the main concepts of this lecture.",
            filepath
        )
        return jsonify({"quiz_text": quiz_text, "thread_id": thread_id})
    except Exception as e:
        return jsonify({"quiz_text": f"⚠️ Server Error: {type(e).__name__}: {e}", "thread_id": ""}), 500

@app.route("/regenerate_quiz", methods=["POST"])
def regenerate_quiz():
    try:
        data = request.get_json(force=True, silent=True) or {}
        edited_quiz = data.get("edited_quiz", "")
        followup = data.get("followup", "")
        thread_id = data.get("thread_id", "")
        quiz_text = regenerate_quiz_stream(edited_quiz, followup, thread_id)
        return jsonify({"quiz_text": quiz_text})
    except Exception as e:
        return jsonify({"quiz_text": f"⚠️ Server Error: {e}"}), 500

SESSIONS = {}

def _parse_quiz_text_to_struct(text: str):
    if not text:
        return {"questions": [], "answers": [], "types": []}

    def norm_type(raw: str) -> str:
        t = (raw or "").strip().lower().replace("-", " ").replace("/", " ")
        t = re.sub(r"\s+", " ", t).replace(" ", "_")
        if t in {"mcq", "multiple_choice", "multiplechoice"}: return "mcq"
        if t in {"short_answer", "shortanswer", "short"}: return "short_answer"
        if t in {"true_false", "truefalse", "true_or_false", "trueorfalse"}: return "true_false"
        if t in {"coding", "code"}: return "coding"
        if t in {"math"}: return "math"
        return "short_answer"

    lines = [ln.rstrip() for ln in text.splitlines()]
    q_a = re.compile(r"^\s*Question\s+(\d+)\s*\(([^)]+)\)\s*[:\-]\s*(.*)$", re.IGNORECASE)
    q_b = re.compile(r"^\s*(\d+)[\.)]\s*(?:Question\s+\d+\s*\(([^)]+)\)|([A-Za-z_\-/ ]+))\s*[:\-]\s*(.*)$", re.IGNORECASE)
    q_c = re.compile(r"^\s*Q\s*(\d+)\s*\(([^)]+)\)\s*[:\-]\s*(.*)$", re.IGNORECASE)
    opt = re.compile(r"^\s*([A-Da-d])\)\s*(.*)$")
    ans = re.compile(r"^\s*Answer:\s*(.*)$", re.IGNORECASE)
    exp = re.compile(r"^\s*Expected\s*Answer:\s*(.*)$", re.IGNORECASE)

    qs, gold, types = [], [], []
    cur_q, cur_t, cur_o = None, None, []

    def flush():
        nonlocal cur_q, cur_t, cur_o
        if cur_q is None:
            return
        qs.append({"type": cur_t or "short_answer", "question": cur_q.strip(), "options": cur_o[:] if cur_t == "mcq" else []})
        types.append(cur_t or "short_answer")
        if len(gold) < len(qs):
            gold.append("")
        cur_q, cur_t, cur_o = None, None, []

    for ln in lines:
        if not ln.strip():
            continue

        m_prefix = re.match(r"^\s*\d+[\.)]\s*(Question\s+\d+\s*\([^)]*\)\s*[:\-].*)$", ln, flags=re.IGNORECASE)
        if m_prefix:
            ln = m_prefix.group(1)

        m = q_a.match(ln) or q_b.match(ln) or q_c.match(ln)
        if m:
            flush()
            if q_a.match(ln):
                cur_t = norm_type((m.group(2) or "").strip())
                cur_q = (m.group(3) or "").strip()
            elif q_c.match(ln):
                cur_t = norm_type((m.group(2) or "").strip())
                cur_q = (m.group(3) or "").strip()
            else:
                cur_t = norm_type((m.group(2) or m.group(3) or "").strip())
                cur_q = (m.group(4) or "").strip()
            cur_o = []
            continue

        if cur_q is None:
            continue

        mo = opt.match(ln)
        if mo:
            if cur_t != "mcq":
                cur_t = "mcq"
                cur_o = []
            cur_o.append(f"{mo.group(1).upper()}) {(mo.group(2) or '').strip()}")
            continue

        ma = ans.match(ln)
        if ma:
            gold.append((ma.group(1) or "").strip())
            continue

        me = exp.match(ln)
        if me:
            gold.append((me.group(1) or "").strip())
            continue

        cur_q = (cur_q + " " + ln.strip()) if cur_q else ln.strip()

    flush()

    if len(gold) < len(qs):
        gold.extend([""] * (len(qs) - len(gold)))

    return {"questions": qs, "answers": gold, "types": types}

@app.route("/grade/prepare", methods=["POST"])
def grade_prepare():
    try:
        file = request.files.get("lecture_file")
        if file and file.filename:
            ext = os.path.splitext(file.filename)[1].lower()
            if ext in ALLOWED_DOC_EXTS:
                base = os.path.splitext(secure_filename(file.filename))[0]
                unique_name = f"{base}_{uuid.uuid4().hex[:6]}{ext}"
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
                file.save(filepath)
                _ = extract_text(filepath)

        quiz_text = (request.form.get("quiz_text") or "").strip()
        if not quiz_text:
            return jsonify({"message": "Please paste the quiz text."}), 400

        struct = _parse_quiz_text_to_struct(quiz_text)
        if not struct["questions"]:
            return jsonify({"message": "Could not detect any questions. Check the format."}), 400

        qs = struct.get("questions", [])
        ans_list = struct.get("answers", [])
        for i in range(min(len(qs), len(ans_list))):
            qs[i]["answer"] = ans_list[i]
        struct["questions"] = qs

        sid = uuid.uuid4().hex
        SESSIONS[sid] = struct
        return jsonify({"sid": sid, "questions": qs})
    except Exception as e:
        return jsonify({"message": f"Server Error: {type(e).__name__}: {e}"}), 500

@app.route("/grade/submit", methods=["POST"])
def grade_submit():
    try:
        data = request.get_json(force=True, silent=True) or {}
        sid = data.get("sid", "")
        user_answers = data.get("answers", {}) or {}
        if not sid or sid not in SESSIONS:
            return jsonify({"message": "Invalid session id."}), 400
        result = grade_quiz(SESSIONS[sid], user_answers)
        return jsonify(result)
    except Exception as e:
        return jsonify({"message": f"Server Error: {type(e).__name__}: {e}\n{traceback.format_exc()}"}), 500

@app.route("/live/upload_lecture", methods=["POST"])
def live_upload_lecture():
    try:
        file = request.files.get("lecture_file")
        if not file or not file.filename:
            return jsonify({"message": "No lecture file provided."}), 400
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_DOC_EXTS:
            return jsonify({"message": "Unsupported file type. Use .pdf or .pptx"}), 400
        base = os.path.splitext(secure_filename(file.filename))[0]
        unique_name = f"{base}_{uuid.uuid4().hex[:6]}{ext}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
        file.save(filepath)
        lecture_text = extract_text(filepath)
        if not lecture_text or lecture_text.startswith("⚠️"):
            return jsonify({"message": "Failed to extract text from lecture file."}), 500
        CURRENT_LECTURE["text"] = lecture_text
        CURRENT_LECTURE["filename"] = file.filename
        return jsonify({"message": f"Lecture '{file.filename}' loaded for summarization."})
    except Exception as e:
        return jsonify({"message": f"Server Error: {type(e).__name__}: {e}"}), 500

def _save_uploaded_audio(file_storage, prefix="audio") -> str:
    ext = os.path.splitext(file_storage.filename)[1].lower()
    if ext not in ALLOWED_AUDIO_EXTS and not ext.startswith("."):
        ext = ".webm"
    unique_name = f"{prefix}_{uuid.uuid4().hex[:6]}{ext}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file_storage.save(filepath)
    return filepath

def _transcribe_audio(path: str) -> str:
    try:
        result = whisper_model.transcribe(path)
        return result.get("text", "").strip()
    except Exception as e:
        return f"⚠️ Failed to transcribe: {type(e).__name__}: {e}"

@app.route("/live/stop_and_summarize", methods=["POST"])
def live_stop_and_summarize():
    try:
        file = request.files.get("audio_file")
        if not file:
            return jsonify({"message": "No audio file received."}), 400
        path = _save_uploaded_audio(file, prefix="live")
        transcript = _transcribe_audio(path)
        summary = summarize_lecture(transcript, CURRENT_LECTURE.get("text", ""))
        return jsonify({"transcript": transcript, "summary": summary})
    except Exception as e:
        return jsonify({"message": f"Failed to stop/transcribe: {type(e).__name__}: {e}"}), 500

@app.route("/audio/summarize", methods=["POST"])
def audio_summarize():
    try:
        lecture_file = request.files.get("lecture_file")
        if lecture_file and lecture_file.filename:
            ext = os.path.splitext(lecture_file.filename)[1].lower()
            if ext in ALLOWED_DOC_EXTS:
                base = os.path.splitext(secure_filename(lecture_file.filename))[0]
                unique_name = f"{base}_{uuid.uuid4().hex[:6]}{ext}"
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
                lecture_file.save(filepath)
                lecture_text = extract_text(filepath)
                if lecture_text and not lecture_text.startswith("⚠️"):
                    CURRENT_LECTURE["text"] = lecture_text
                    CURRENT_LECTURE["filename"] = lecture_file.filename

        audio_file = request.files.get("audio_file")
        if not audio_file or not audio_file.filename:
            return jsonify({"message": "No audio file provided."}), 400

        path = _save_uploaded_audio(audio_file, prefix="upload")
        transcript = _transcribe_audio(path)
        summary = summarize_lecture(transcript, CURRENT_LECTURE.get("text", ""))
        return jsonify({"transcript": transcript, "summary": summary})
    except Exception as e:
        return jsonify({"message": f"Server Error: {type(e).__name__}: {e}"}), 500

UKH_LOGO_LOCAL_PATH = "/Users/danial/Desktop/Ai_teaching_assistant/Ai-teaching-assistant/Logo-Transparent.png"

@app.route("/ukh-logo")
def ukh_logo():
    if not UKH_LOGO_LOCAL_PATH or not os.path.exists(UKH_LOGO_LOCAL_PATH):
        return Response("", status=404)
    return send_file(UKH_LOGO_LOCAL_PATH, mimetype="image/png")

if __name__ == "__main__":
    port = 7860
    print(f"🚀 Unified AI-TA running at: http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)