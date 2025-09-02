# modules/groq_client.py
import os
from dotenv import load_dotenv
from collections import defaultdict
from typing import List, Dict
import streamlit as st

try:
    from groq import Groq
except Exception:
    Groq = None

# Load environment variables
load_dotenv()

# Constants
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
RATE_LIMIT_PATTERNS = [
    "rate limit", "quota", "limit exceeded",
    "too many requests", "429", "insufficient_quota"
]


def get_api_keys_from_env():
    """Extract all GROQ API keys from environment variables"""
    api_keys = []
    for i in range(1, 21):  # Check for keys 1-20
        key_name = f"GROQ_API_KEY_{i}"
        key_value = os.getenv(key_name, "").strip()
        if key_value:
            api_keys.append(key_value)

    # Remove duplicates while preserving order
    seen = set()
    unique_keys = []
    for key in api_keys:
        if key not in seen:
            seen.add(key)
            unique_keys.append(key)

    return unique_keys


def is_rate_limit_error(error_msg):
    """Check if error message indicates a rate limit"""
    if not error_msg:
        return False
    error_lower = error_msg.lower()
    return any(pattern in error_lower for pattern in RATE_LIMIT_PATTERNS)


def get_next_groq_client():
    """Get the next available Groq client with fallback keys"""
    api_keys = get_api_keys_from_env()

    if not api_keys:
        st.error("No API keys available. Please check your .env file.")
        return None

    # Initialize session state if not exists
    if "groq_key_index" not in st.session_state:
        st.session_state.groq_key_index = 0
    if "groq_key_limits" not in st.session_state:
        st.session_state.groq_key_limits = defaultdict(bool)

    # Try current key first if it hasn't hit limit
    current_index = st.session_state.groq_key_index
    current_key = api_keys[current_index]

    if current_key and not st.session_state.groq_key_limits[current_index]:
        try:
            client = Groq(api_key=current_key)
            # Quick test to see if key is valid
            client.models.list(timeout=5)
            return client
        except Exception as e:
            if is_rate_limit_error(str(e)):
                st.session_state.groq_key_limits[current_index] = True
                st.warning(f"API key {current_index + 1} hit rate limit. Switching to next key.")
            else:
                print(f"API key {current_index + 1} test failed: {e}")

    # If current key hit limit or failed, try other keys
    for i in range(1, len(api_keys)):
        next_index = (current_index + i) % len(api_keys)
        next_key = api_keys[next_index]

        if next_key and not st.session_state.groq_key_limits[next_index]:
            try:
                client = Groq(api_key=next_key)
                client.models.list(timeout=5)  # Quick test
                st.session_state.groq_key_index = next_index
                return client
            except Exception as e:
                if is_rate_limit_error(str(e)):
                    st.session_state.groq_key_limits[next_index] = True
                    print(f"API key {next_index + 1} hit rate limit during test")
                else:
                    print(f"API key {next_index + 1} test failed: {e}")
                continue

    # If all keys hit limits, try to reset and start from beginning
    if all(st.session_state.groq_key_limits.get(i, False) for i in range(len(api_keys))):
        st.warning("All API keys hit limits. Resetting and trying keys again.")
        for i in range(len(api_keys)):
            st.session_state.groq_key_limits[i] = False  # Reset all limits
        st.session_state.groq_key_index = 0
        return get_next_groq_client()  # Recursive call to try again

    st.error("All available API keys have hit rate limits or are invalid.")
    return None


def groq_complete(messages: List[Dict[str, str]], max_tokens: int = 220) -> str:
    from modules.data_handler import clean_output

    api_keys = get_api_keys_from_env()
    if not api_keys:
        return ""

    client = get_next_groq_client()
    if not client:
        return ""

    try:
        resp = client.chat.completions.create(
            model=DEFAULT_GROQ_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=max_tokens,
            timeout=30
        )
        out = (resp.choices[0].message.content or "").strip()
        return clean_output(out)
    except Exception as e:
        current_index = st.session_state.groq_key_index
        error_msg = str(e)

        if is_rate_limit_error(error_msg):
            st.session_state.groq_key_limits[current_index] = True
            print(f"(Groq) rate limit with key {current_index + 1}: {error_msg}")
        else:
            print(f"(Groq) completion error with key {current_index + 1}: {error_msg}")

        return ""


def groq_chat_completion(prompt: str, system: str = None) -> str:
    from modules.data_handler import clean_output

    client = get_next_groq_client()
    if not client:
        return ""

    try:
        sys_msg = system or (
            "You are a QA coaching assistant.\n"
            "From the raw comments, output only improvement issues.\n"
            "- Each issue: a unique, very short phrase or one short sentence.\n"
            "- Merge duplicates; avoid filler and repetition.\n"
            "- No strengths, no generic advice, no checklists.\n"
            '- If no improvements, respond exactly: "Analyst performed well. No improvement as of now."'
        )

        response = client.chat.completions.create(
            model=DEFAULT_GROQ_MODEL,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": prompt},
            ],
            timeout=30
        )

        if response and response.choices:
            raw = response.choices[0].message.content or ""
            return clean_output(raw).strip()
        return ""
    except Exception as e:
        current_index = st.session_state.groq_key_index
        error_msg = str(e)

        if is_rate_limit_error(error_msg):
            st.session_state.groq_key_limits[current_index] = True
            print(f"Groq API rate limit with key {current_index + 1}: {error_msg}")
        else:
            print(f"Groq API call failed with key {current_index + 1}: {error_msg}")

        return ""