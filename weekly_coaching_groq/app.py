import os
import io
import datetime as dt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Import modules
from modules.data_handler import clean_comment
from modules.tracker import generate_tracker
from modules.views import ensure_session_defaults, render_treemap, show_quick_views
from modules.groq_client import get_api_keys_from_env

# Load environment variables
load_dotenv()

# ========================= UI CONSTANTS ========================= #
APP_TITLE = "Weekly Coaching Assessment Tracker"
AREAS_TO_IMPROVE_EXAMPLES = [
    "Probe for full issue context before troubleshooting.",
    "Summarize resolution and confirm next steps with the customer.",
    "Reduce hold time by using knowledge base bookmarks.",
    "Follow escalation matrix within SLA when blockers occur.",
    "Ensure ticket notes are complete: symptoms, root cause, and fix.",
]


def main():
    ensure_session_defaults()
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    # Show API key status
    api_keys = get_api_keys_from_env()
    if not api_keys:
        st.warning("⚠️ No GROQ API keys found. Please set GROQ_API_KEY_1, GROQ_API_KEY_2, etc. in your .env file.")
    else:
        st.sidebar.success(f"✅ Found {len(api_keys)} API key(s)")

    

    # The rest of your main function code goes here...
    # You would copy the main() function from your original code
    # but replace the function calls with the modular versions


if __name__ == "__main__":
    main()