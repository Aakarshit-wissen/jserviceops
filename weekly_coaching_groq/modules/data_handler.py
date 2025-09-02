# modules/data_handler.py
import pandas as pd
import numpy as np
import re
import math
import datetime as dt
from collections import Counter, defaultdict
from typing import Dict, List, Optional


def _norm(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    if isinstance(s, (int, float)) and not pd.isna(s):
        return str(s).strip()
    if isinstance(s, str):
        return s.strip()
    return None


def pick_first_available_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    if df is None or df.empty:
        return None

    def col_to_text(col) -> str:
        if isinstance(col, (list, tuple)):
            col = " ".join(str(x) for x in col)
        return str(col)

    def norm(s: str) -> str:
        return s.lower().strip()

    def norm_simple(s: str) -> str:
        return re.sub(r"[^a-z]", "", norm(s))

    lower_map, simple_map = {}, {}
    for original in df.columns:
        t = col_to_text(original)
        lower_map.setdefault(norm(t), original)
        simple_map.setdefault(norm_simple(t), original)
    for cand in candidates:
        key = norm(str(cand))
        if key in lower_map:
            return lower_map[key]
    for cand in candidates:
        key = norm_simple(str(cand))
        if key in simple_map:
            return simple_map[key]
    return None


def week_of_month(date: dt.date) -> int:
    first_day = date.replace(day=1)
    adjusted_dom = date.day + (first_day.weekday())
    return int(math.ceil(adjusted_dom / 7.0))


def to_date(s) -> Optional[dt.date]:
    if pd.isna(s):
        return None
    if isinstance(s, dt.datetime):
        return s.date()
    if isinstance(s, dt.date):
        return s
    try:
        return pd.to_datetime(s, errors="coerce").date()
    except Exception:
        return None


def try_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def letter_grade_to_score(val: str) -> Optional[float]:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    m = {
        "A+": 5.0, "A": 5.0, "A-": 4.7,
        "B+": 4.3, "B": 4.0, "B-": 3.7,
        "C+": 3.3, "C": 3.0, "C-": 2.7,
        "D+": 2.3, "D": 2.0, "D-": 1.7,
        "E": 1.0, "F": 0.0,
    }
    s = str(val).strip().upper()
    return m.get(s)


def average_of_grades(series: pd.Series) -> Optional[float]:
    s_num = pd.to_numeric(series, errors="coerce")
    if s_num.notna().any():
        vals = s_num.dropna().tolist()
        return round(float(np.mean(vals)), 2) if vals else None
    mapped = [letter_grade_to_score(v) for v in series]
    mapped = [v for v in mapped if v is not None]
    return round(float(np.mean(mapped)), 2) if mapped else None


# Text cleaning functions
YESNO_PATTERNS = [
    r":\s*-?\s*Yes\b", r":\s*-?\s*No\b", r":\s*-?\s*Yes/No\b",
    r":\s*-?\s*Resolved\b", r":\s*-?\s*Escalated\b", r":\s*-?\s*Closed\b",
    r":\s*Accurate\b",
]
YESNO_SUFFIX = re.compile("|".join(YESNO_PATTERNS), re.IGNORECASE)
BULLET_LINE = re.compile(r"^\s*(?:\d+[\.\)]|[-*•·])\s+")
CHECKLIST_KEYS = [
    "checked for previous tickets", "preference contact method", "preferred contact method", "contact type",
    "ms teams chat integration", "ms teams integration", "short description", "configuration item", "category",
    "computer name documented", "error screenshot attached", "ticket state", "work note documentation",
    "additional assistance", "call opening followed", "user validation procedure", "display empathy & assurance",
    "call holding procedure", "professional and positive tone", "sufficient probing", "proper 3 reminder process",
    "bomgar chat messages attached", "computer name documented/ bomgar chat messages attached",
]


def _is_checklist_line(line: str) -> bool:
    L = line.strip().strip('"').lower()
    if not L:
        return True
    if L.startswith("observations") or "observation/areas for improvement" in L:
        return True
    if any(k in L for k in CHECKLIST_KEYS):
        return True
    if YESNO_SUFFIX.search(line):
        return True
    if BULLET_LINE.match(line):
        return True
    return False


def clean_comment(raw: str) -> str:
    if not isinstance(raw, str):
        return ""
    raw = raw.replace("\r", "\n")
    lines = [ln.strip() for ln in raw.split("\n")]
    kept: List[str] = []
    for ln in lines:
        if _is_checklist_line(ln):
            continue
        if len(ln.strip(" -•*·\t\"'")) < 3:
            continue
        kept.append(ln.strip('"').strip())
    text = " ".join(kept)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_output(text: str) -> str:
    if not text:
        return ""
    prefixes = [
        "Here is a concise summary of Areas to Improve:",
        "Here is a summary of Areas to Improve:",
        "Based on the raw comments,",
        "Here are the areas to improve:",
        "Here are the Areas to Improve:",
        "Here is the summary of Areas to Improve:"
    ]
    for p in prefixes:
        text = text.replace(p, "")
    return text.strip()