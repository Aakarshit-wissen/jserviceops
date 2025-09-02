# modules/tracker.py
import pandas as pd
import numpy as np
import datetime as dt
from collections import Counter, defaultdict
from typing import Dict, List, Optional

from modules.data_handler import (
    _norm, pick_first_available_column, week_of_month, to_date,
    average_of_grades, build_employee_key, clean_comment
)
from modules.groq_client import groq_chat_completion

# Column mapping constants
COLUMN_CANDIDATES = {
    "employee_name": ["Employee Name", "Analyst Name", "Trainee", "Analyst", "Name"],
    "trainee_name": ["Trainee", "Trainee Name", "Analyst Name", "Employee Name", "Analyst"],
    "analyst_name": ["Analyst Name", "Employee Name", "Analyst"],
    "team_lead": ["Team Lead", "Team Leader", "TL", "Manager", "Reporting Manager"],
    "manager": ["Manager", "Reporting Manager", "Supervisor"],
    "quality_assessor": ["Assessor", "Quality Assessor", "QA", "Quality Analyst", "Analyst Name", "Coach"],
    "week": ["Week", "Week Number", "Week_of_Month", "WOM"],
    "date": ["Date", "Assessed On", "Created", "Assessment Date"],
    "sn_rating": ["Trainee rating", "Trainee Rating", "Rating"],
    "manual_grade": ["Grade", "Score", "Marks"],
    "comments": ["Comments", "Notes", "QA Comments", "Reviewer Comments"],
}


def extract_people_fields(df: pd.DataFrame) -> Dict[str, str]:
    out = {}
    out["name_col"] = pick_first_available_column(df, COLUMN_CANDIDATES["employee_name"]) or \
                      pick_first_available_column(df, COLUMN_CANDIDATES["trainee_name"]) or \
                      pick_first_available_column(df, COLUMN_CANDIDATES["analyst_name"])
    out["tl_col"] = pick_first_available_column(df, COLUMN_CANDIDATES["team_lead"]) or \
                    pick_first_available_column(df, COLUMN_CANDIDATES["manager"])
    out["qa_col"] = pick_first_available_column(df, COLUMN_CANDIDATES["quality_assessor"])
    out["week_col"] = pick_first_available_column(df, COLUMN_CANDIDATES["week"])
    out["date_col"] = pick_first_available_column(df, COLUMN_CANDIDATES["date"])
    out["sn_rating_col"] = pick_first_available_column(df, COLUMN_CANDIDATES["sn_rating"])
    out["manual_grade_col"] = pick_first_available_column(df, COLUMN_CANDIDATES["manual_grade"])
    out["comments_col"] = pick_first_available_column(df, COLUMN_CANDIDATES["comments"])
    return out


def get_week_filter_mask(df: pd.DataFrame, week_col: Optional[str], date_col: Optional[str], month_n: int,
                         week_n: int) -> pd.Series:
    if df is None or df.empty:
        return pd.Series([False] * 0)
    mask = pd.Series([True] * len(df))
    if week_col and week_col in df.columns:
        wvals = df[week_col].astype(str).str.extract(r"(\d+)")[0].astype(float)
        mask &= (wvals == week_n)
    elif date_col and date_col in df.columns:
        dates = df[date_col].apply(to_date)
        wom = dates.apply(lambda d: week_of_month(d) if d else np.nan)
        months = dates.apply(lambda d: d.month if d else np.nan)
        mask &= (months == month_n) & (wom == week_n)
    else:
        mask &= True
    return mask.fillna(False)


def synthesize_areas_to_improve(all_comments: List[str], employee_name: str, seed_examples: List[str]) -> str:
    try:
        cleaned_bits = [clean_comment(c).strip() for c in (all_comments or []) if _norm(c)]
    except NameError:
        cleaned_bits = [clean_comment(c).strip() for c in (all_comments or []) if isinstance(c, str) and c.strip()]
    if not cleaned_bits:
        return ""
    combined = " ".join(cleaned_bits)
    user_prompt = (
            f"Raw observations (cleaned): {combined}\n"
            + "Write only 2–3 sentences focusing on specific improvement actions and missed steps. "
              "Avoid strengths, disclaimers, or generic best practices. Do not include personally identifiable customer data."
    )
    return groq_chat_completion(user_prompt)


def generate_tracker(manual_df: pd.DataFrame, sn_df: pd.DataFrame, month_n: int, week_n: int,
                     seed_examples: List[str]) -> pd.DataFrame:
    m_cols = extract_people_fields(manual_df)
    s_cols = extract_people_fields(sn_df)
    m_mask = get_week_filter_mask(manual_df, m_cols["week_col"], m_cols["date_col"], month_n, week_n)
    s_mask = get_week_filter_mask(sn_df, s_cols["week_col"], s_cols["date_col"], month_n, week_n)
    mdf = manual_df[m_mask].copy() if len(manual_df) else manual_df.copy()
    sdf = sn_df[s_mask].copy() if len(sn_df) else sn_df.copy()

    manual_names = mdf[m_cols["name_col"]].dropna().astype(str).map(build_employee_key) if m_cols[
        "name_col"] else pd.Series([], dtype=str)
    sn_names = sdf[s_cols["name_col"]].dropna().astype(str).map(build_employee_key) if s_cols[
        "name_col"] else pd.Series([], dtype=str)
    all_names = sorted(set(manual_names.dropna()).union(set(sn_names.dropna())))

    def per_employee(df: pd.DataFrame, cols: Dict[str, str]) -> Dict[str, Dict[str, list]]:
        d = defaultdict(
            lambda: {"team_leads": [], "assessors": [], "comments": [], "sn_ratings": [], "manual_grades": []})
        if df is None or df.empty or not cols.get("name_col"):
            return d
        for _, row in df.iterrows():
            name = build_employee_key(row.get(cols["name_col"]))
            if not name:
                continue
            tl = row.get(cols["tl_col"]) if cols.get("tl_col") else None
            qa = row.get(cols["qa_col"]) if cols.get("qa_col") else None
            cm = row.get(cols["comments_col"]) if cols.get("comments_col") else None
            sr = row.get(cols["sn_rating_col"]) if cols.get("sn_rating_col") else None
            mg = row.get(cols["manual_grade_col"]) if cols.get("manual_grade_col") else None
            if tl: d[name]["team_leads"].append(str(tl))
            if qa: d[name]["assessors"].append(str(qa))
            if cm: d[name]["comments"].append(str(cm))
            if sr is not None: d[name]["sn_ratings"].append(sr)
            if mg is not None: d[name]["manual_grades"].append(mg)
        return d

    agg_manual = per_employee(mdf, m_cols)
    agg_sn = per_employee(sdf, s_cols)
    records = []

    for name in all_names:
        team_leads = agg_manual[name]["team_leads"] + agg_sn[name]["team_leads"]
        assessors = agg_manual[name]["assessors"] + agg_sn[name]["assessors"]
        comments = agg_manual[name]["comments"] + agg_sn[name]["comments"]
        sn_ratings = pd.to_numeric(pd.Series(agg_sn[name]["sn_ratings"]), errors="coerce").dropna().tolist()
        manual_grades_series = pd.Series(agg_manual[name]["manual_grades"])

        team_lead_final = None
        if team_leads:
            c = Counter([_norm(tl) for tl in team_leads if _norm(tl)])
            team_lead_final = c.most_common(1)[0][0] if c else None

        assessors_final = ", ".join(sorted(set([_norm(a) for a in assessors if _norm(a)]))) or None
        service_now_count = len(sn_ratings)
        service_now_avg = round(float(np.mean(sn_ratings)), 2) if sn_ratings else None
        manual_count = len(agg_manual[name]["manual_grades"]) if agg_manual[name]["manual_grades"] else 0
        manual_avg = average_of_grades(manual_grades_series) if manual_count else None
        areas_text = synthesize_areas_to_improve(comments, name, seed_examples)

        rec = {
            "Employee Name": name,
            "Emp ID": "",
            "Organization": "",
            "Team Lead": team_lead_final or "",
            "Status": "",
            "Quality Assessors": assessors_final or "",
            "Week": f"{dt.date(1900, month_n, 1).strftime('%b')} - Week {week_n}",
            "Service Now Assessments Count": service_now_count if service_now_count else "",
            "Service Now Assessments Average Rating": service_now_avg if service_now_avg is not None else "",
            "Manual Assessments Count": manual_count if manual_count else "",
            "Manual Assessments Average Rating": manual_avg if manual_avg is not None else "",
            "Areas to Improve": areas_text,
        }
        records.append(rec)

    tracker = pd.DataFrame.from_records(records)
    col_order = [
        "Employee Name", "Emp ID", "Organization", "Team Lead", "Status", "Quality Assessors", "Week",
        "Service Now Assessments Count", "Service Now Assessments Average Rating",
        "Manual Assessments Count", "Manual Assessments Average Rating", "Areas to Improve",
    ]
    tracker = tracker.reindex(columns=col_order)
    return tracker