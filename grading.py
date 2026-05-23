from __future__ import annotations

import difflib
import re
import string
from typing import Any, Dict, List, Optional, Tuple

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)

# Tuning (safe defaults)
MCQ_TEXT_SIM_THRESHOLD = 0.88
SHORT_SIM_THRESHOLD = 0.82
CODING_SIM_THRESHOLD = 0.70


def _normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.translate(_PUNCT_TABLE)
    s = re.sub(r"\s+", " ", s)
    return s


def _sim(a: str, b: str) -> float:
    a_n = _normalize_text(a)
    b_n = _normalize_text(b)
    if not a_n and not b_n:
        return 1.0
    if not a_n or not b_n:
        return 0.0
    return difflib.SequenceMatcher(None, a_n, b_n).ratio()


def _extract_mcq_letter(ans: str) -> Optional[str]:

    if not ans:
        return None
    a = ans.strip()
    m = re.match(r"^\s*([A-Da-d])\b", a)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-Da-d])\)", a)
    if m:
        return m.group(1).upper()
    return None


def _strip_choice_prefix(s: str) -> str:
    return re.sub(r"^\s*[A-Da-d]\)\s*", "", s or "").strip()


def _numbers(s: str) -> set[str]:
    return set(re.findall(r"-?\d+(?:\.\d+)?", s or ""))


def _tokens(s: str) -> List[str]:
    return re.findall(r"[a-z0-9_]+", _normalize_text(s))


def _jaccard(a: List[str], b: List[str]) -> float:
    A, B = set(a), set(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def _resolve_expected_answer(quiz_json: Dict[str, Any], i: int, q: Dict[str, Any]) -> str:

    if isinstance(q.get("answer"), str) and q.get("answer"):
        return str(q.get("answer"))
    answers = quiz_json.get("answers")
    if isinstance(answers, list) and i < len(answers):
        return str(answers[i] or "")
    return ""


def _resolve_type(quiz_json: Dict[str, Any], i: int, q: Dict[str, Any]) -> str:
    t = (q.get("type") or "").strip().lower()
    types = quiz_json.get("types")
    if (not t) and isinstance(types, list) and i < len(types):
        t = str(types[i] or "").strip().lower()
    # normalize
    t = t.replace(" ", "_")
    if t in {"multiple_choice", "multiple-choice"}:
        t = "mcq"
    if t in {"short", "shortanswer"}:
        t = "short_answer"
    return t or "short_answer"


def grade_quiz(quiz_json: Dict[str, Any], user_answers: Dict[str, Any]) -> Dict[str, Any]:
    """Grade a quiz.

    user_answers example: {"0":"A", "1":"text...", ...} (keys are indices as strings)

    Returns:
      {
        "total": int,
        "correct": int,
        "percent": float,
        "details": [ {"i":1,"type":"mcq","question":"...","your":"...","expected":"...","ok":true,"note":"..."}, ... ]
      }
    """

    questions = quiz_json.get("questions") or []
    if not isinstance(questions, list):
        questions = []

    details: List[Dict[str, Any]] = []
    correct = 0

    for i, q in enumerate(questions):
        if not isinstance(q, dict):
            continue

        qtype = _resolve_type(quiz_json, i, q)
        expected = _resolve_expected_answer(quiz_json, i, q)
        guess = str(user_answers.get(str(i), "") or "")

        ok = False
        note = ""

        if qtype == "mcq":
            gold_letter = _extract_mcq_letter(expected)
            user_letter = _extract_mcq_letter(guess)

            if gold_letter and user_letter:
                ok = (gold_letter == user_letter)
                note = "letter_match" if ok else f"expected={gold_letter}, got={user_letter}"
            else:
                # compare option text
                g_txt = _strip_choice_prefix(expected)
                u_txt = _strip_choice_prefix(guess)
                ratio = _sim(g_txt, u_txt)
                ok = ratio >= MCQ_TEXT_SIM_THRESHOLD
                note = f"text_similarity={ratio:.2f}"

        elif qtype == "math":
            if not guess.strip():
                ok = False
                note = "blank"
            else:
                g_nums = _numbers(expected)
                u_nums = _numbers(guess)
                if g_nums:
                    ok = (g_nums == u_nums)
                    note = f"numbers={len(u_nums)}/{len(g_nums)}"
                else:
                    ratio = _sim(expected, guess)
                    ok = ratio >= SHORT_SIM_THRESHOLD
                    note = f"text_similarity={ratio:.2f}"

        elif qtype == "coding":
            if not guess.strip():
                ok = False
                note = "blank"
            else:
                # Prefer keyword overlap if expected contains terms
                terms = [t.strip() for t in re.split(r"[,\n;`]+", expected or "") if t.strip()]
                if terms:
                    exp_norm = _normalize_text(expected)
                    g_norm = _normalize_text(guess)
                    hits = sum(1 for t in terms if _normalize_text(t) in g_norm)
                    ok = hits >= max(1, len(terms) // 2)
                    note = f"matched_keywords={hits}/{len(terms)}"
                else:

                    jac = _jaccard(_tokens(expected), _tokens(guess))
                    ratio = _sim(expected, guess)
                    ok = (jac >= 0.55) or (ratio >= CODING_SIM_THRESHOLD)
                    note = f"jaccard={jac:.2f}, similarity={ratio:.2f}"

        else:
            if not guess.strip():
                ok = False
                note = "blank"
            else:
                ratio = _sim(expected, guess)

                g_n = _normalize_text(expected)
                u_n = _normalize_text(guess)
                ok = (g_n == u_n) or (g_n and g_n in u_n) or (u_n and u_n in g_n) or (ratio >= SHORT_SIM_THRESHOLD)
                note = f"text_similarity={ratio:.2f}"

        if ok:
            correct += 1

        details.append({
            "i": i + 1,
            "type": qtype,
            "question": q.get("question", ""),
            "your": guess,
            "expected": expected,
            "ok": bool(ok),
            "note": note,
        })

    total = len(questions)
    percent = round((100.0 * correct / total), 2) if total else 0.0

    return {
        "total": total,
        "correct": correct,
        "percent": percent,
        "details": details,
    }
