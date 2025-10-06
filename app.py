import streamlit as st
import datetime
import time
import random
import io
import os
import pandas as pd
import requests
import matplotlib.pyplot as plt
from langdetect import detect, LangDetectException
from gtts import gTTS
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ---------------- Optional Libraries ----------------
# Groq LLM (optional)
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except Exception:
    GROQ_AVAILABLE = False

# Lottie animations (optional)
try:
    from streamlit_lottie import st_lottie
    LOTTIE_AVAILABLE = True
except Exception:
    LOTTIE_AVAILABLE = False

# Mic recorder for voice input (optional)
try:
    from streamlit_mic_recorder import mic_recorder
    MIC_AVAILABLE = True
except Exception:
    MIC_AVAILABLE = False

# Google Translate (optional, graceful fallback)
try:
    from googletrans import Translator
    TRANSLATOR_AVAILABLE = True
    translator = Translator()
except Exception:
    TRANSLATOR_AVAILABLE = False


# -------------------- App Config --------------------
st.set_page_config(
    page_title="AI Doctor Pocket 🩺",
    page_icon="🩺",
    layout="wide"
)

APP_TITLE = "AI Doctor Pocket 🩺"
st.markdown(f"<h1 style='text-align:center; font-weight:700;'>{APP_TITLE}</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; font-size:14px; color:gray;'>🌐 Multilingual • 🎙️ Voice reply • 📊 Analytics • 📄 PDF Export • 🚨 Emergency-Aware</p>",
    unsafe_allow_html=True
)

with st.expander("💡 Quick Tip (Click to Expand)", expanded=False):
    st.markdown("""
    <div style='
        background: linear-gradient(135deg, #1f1c2c, #928DAB);
        color: #fff;
        padding: 1.2rem 1.5rem;
        border-radius: 16px;
        box-shadow: 0px 4px 25px rgba(0,0,0,0.4);
        font-family: "Poppins", sans-serif;
        transition: all 0.3s ease;
    '>
        <h4 style='margin-bottom:0.7rem;'>⚙️ Pro Tip</h4>
        <p style='font-size: 15px; line-height:1.6;'>
            👉 Type or speak your symptoms naturally.  
            The AI Doctor will analyze your condition and provide smart medical insights.  
            <br><br>
            💬 Example: <em>"I have body pain and a mild fever since last night."</em>  
            <br>
            🎧 Voice input supported — perfect for multilingual users.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ---------------- CSS ----------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;500;700&display=swap');
    html, body { font-family: 'Poppins', sans-serif; background: #f5f7fa; color: #0b1220; }

    /* Topbar */
    .topbar {
        display:flex; justify-content:space-between; align-items:center;
        padding:12px 18px; background:linear-gradient(90deg,#3b82f6,#6366f1);
        color:white; border-radius:10px; box-shadow:0 4px 12px rgba(0,0,0,0.1);
        margin-bottom:12px;
    }

    /* Panels */
    .panel {
        background: rgba(255,255,255,0.05);
        padding:14px; border-radius:12px; margin-bottom:12px;
        box-shadow:0 4px 12px rgba(0,0,0,0.04);
    }

    /* Chat bubbles */
    .bubble-user {
        background: linear-gradient(90deg,#00b4db,#0083b0);
        color:white; padding:12px 16px; border-radius:18px;
        margin:6px 0; float:right; clear:both; max-width:78%;
        box-shadow:0 3px 8px rgba(0,0,0,0.12); word-wrap: break-word;
    }

    .bubble-ai {
        background: #f3f4f6; color:#0b1220; padding:12px 16px;
        border-radius:18px; margin:6px 0; float:left; clear:both;
        max-width:78%; box-shadow:0 3px 8px rgba(0,0,0,0.08);
        word-wrap: break-word;
    }

    /* Emergency */
    .emergency {
        background:#ff4c4c; color:white; padding:14px; border-radius:12px;
        font-weight:700; text-align:center; box-shadow:0 4px 12px rgba(0,0,0,0.12);
    }

    /* Summary */
    .summary {
        background:white; padding:14px; border-radius:12px;
        box-shadow:0 6px 18px rgba(0,0,0,0.08); color:#0b1220;
    }

    .muted { color:gray; font-size:13px; }
    footer { text-align:center; color:gray; margin-top:16px; font-size:14px; }
    .typing { color:gray; font-style:italic; }

    /* Badges */
    .badge-low {
        background:#d1fae5; color:#065f46; padding:6px 12px;
        border-radius:999px; font-weight:600;
    }

    .badge-med {
        background:#fff7ed; color:#92400e; padding:6px 12px;
        border-radius:999px; font-weight:600;
    }

    .badge-high {
        background:#ffe4e6; color:#7f1d1d; padding:6px 12px;
        border-radius:999px; font-weight:700;
    }

    /* Confidence bar */
    .confbar { height:14px; border-radius:10px; background:#e6eef8;
        overflow:hidden; margin-top:4px; margin-bottom:8px; }
    .conffill { height:100%; border-radius:10px;
        background:linear-gradient(90deg,#3b82f6,#6366f1); }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------- Sidebar & Next-Gen Theme ----------------
st.sidebar.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1f2937 0%, #111827 100%);
        color: #f3f4f6; padding: 1rem; border-radius: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Sidebar ----------------
st.sidebar.title("⚙️ Settings & Tools")

# ---------------- API Key ----------------
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
st.session_state.api_key = st.sidebar.text_input(
    "🔑 Groq API Key (optional)", 
    type="password", 
    value=st.session_state.api_key,
    help="Used for AI calls. Leave empty for demo mode."
)

# ---------------- Doctor Modes ----------------
DOCTOR_MODES = {
    "👨‍⚕️ General Physician": "Causes, precautions, next steps.",
    "🍎 Dietician": "Meal tips & nutrition guidance.",
    "🧘 Mental Health": "Empathetic mental health support.",
    "🦷 Dentist": "Dental care tips & red flags.",
    "👁️ Eye Specialist": "Eye care & vision advice."
}
st.sidebar.markdown("### 🩺 Doctor Mode")
mode = st.sidebar.radio(
    "",
    options=list(DOCTOR_MODES.keys()),
    format_func=lambda x: f"{x} — {DOCTOR_MODES[x]}"
)

# ---------------- Theme ----------------
st.sidebar.markdown("### 🎨 Theme")
theme_choice = st.sidebar.radio("Choose theme", ["Auto", "Light", "Dark"])
if theme_choice == "Dark":
    st.markdown("<style>body{background:#0b1220;color:#e6eef8;}</style>", unsafe_allow_html=True)
elif theme_choice == "Light":
    st.markdown("<style>body{background:#f5f8ff;color:#0b1220;}</style>", unsafe_allow_html=True)

# ---------------- Extra Toggles ----------------
st.sidebar.markdown("### ⚡ Features")
voice_reply = st.sidebar.checkbox("🔊 Voice reply (gTTS)", value=False)
auto_translate = st.sidebar.checkbox(
    "🌐 Auto-translate for AI calls", value=True
) if TRANSLATOR_AVAILABLE else False
save_local = st.sidebar.checkbox("💾 Save logs locally (CSV)", value=True)

# Optional: subtle separator
st.sidebar.markdown("---")
st.sidebar.caption("Your settings are saved for this session.")




import random
import streamlit as st

# --- Sidebar Quick Tip ---
st.sidebar.markdown("<hr>", unsafe_allow_html=True)

# 🌟 Stylish Header
st.sidebar.markdown("""
<div style='text-align:center; font-size:20px; font-weight:600;
            background: linear-gradient(90deg, #00c6ff, #0072ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-family:"Poppins", sans-serif;'>
    💡 Quick Health Tip
</div>
""", unsafe_allow_html=True)

# 🌿 Tips List
tips = [
    "💧 Stay hydrated – drink at least 2L of water daily.",
    "🚶 Take a 20-minute walk to boost circulation.",
    "😴 Get 7–8 hours of sleep to help your body recover.",
    "🍎 Eat more fruits and vegetables for a stronger immune system.",
    "💻 Take short breaks from your screen every hour.",
    "🌞 Get some sunlight – Vitamin D helps your mood and energy.",
    "🧘‍♂️ Practice deep breathing to reduce stress.",
    "🥗 Avoid junk food – fuel your body with nutrients.",
    "🚿 Take a cold shower for energy and mental clarity.",
]

# 🧠 Random Tip Display Box (with glassmorphism effect)
tip = random.choice(tips)
st.sidebar.markdown(f"""
<div style='
    background: rgba(255, 255, 255, 0.12);
    backdrop-filter: blur(10px);
    border-radius: 15px;
    padding: 1rem 1.2rem;
    margin-top: 10px;
    font-size:15px;
    line-height:1.5;
    color: #fff;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.3);
    font-family:"Poppins", sans-serif;'>
    {tip}
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("<hr>", unsafe_allow_html=True)



# ---------------- Clear Chat UI ----------------
with st.sidebar.expander("🧹 Clear Chat & Logs", expanded=False):
    st.markdown("""
    <div style='font-size:15px; color:#cfcfcf; line-height:1.4;'>
        This will permanently delete all chat history and logs.<br>
        Use this only if you want a <b>fresh start</b>.
    </div>
    """, unsafe_allow_html=True)

    confirm_clear = st.checkbox("✅ I understand and want to clear data", key="confirm_clear")

    if confirm_clear:
        if st.button("🚨 Clear Now", use_container_width=True):
            # reset session history
            st.session_state.chat_history = []
            st.session_state.logs_df = pd.DataFrame(columns=[
                "timestamp", "user", "ai", "language", "mode", "confidence", "risk", "emergency"
            ])

            # delete local CSV if exists
            try:
                if os.path.exists("ai_doctor_logs.csv"):
                    os.remove("ai_doctor_logs.csv")
            except Exception:
                pass

            st.success("✅ All chat history and logs have been cleared successfully.")
            st.balloons()
            st.rerun()
    else:
        st.markdown(
            "<div style='color:gray; font-size:13px;'>Check the box above to enable clearing.</div>",
            unsafe_allow_html=True
        )


# ---------------- Session State Initialization ----------------
LOG_COLUMNS = ["timestamp", "user", "ai", "language", "mode", "confidence", "risk", "emergency"]

def init_session_key(key: str, default_value):
    if key not in st.session_state:
        st.session_state[key] = default_value

init_session_key("chat_history", [])
init_session_key("logs_df", pd.DataFrame(columns=LOG_COLUMNS))
init_session_key("user_text", "")
init_session_key("voice_enabled", False)
init_session_key("dark_mode", True)
init_session_key("last_tip", None)

st.caption("⚙️ Session initialized and ready — data auto-saves during conversation.")

# ---------------- Helper Functions ----------------
def safe_detect_language(text: str) -> str:
    if not text:
        return "English"
    try:
        lang = detect(text)
        if lang and lang.startswith("ur"):
            return "Urdu"
        if lang and lang.startswith("hi"):
            return "Hindi"
        return "English"
    except LangDetectException:
        return "English"
    except Exception:
        return "English"

def emergency_check(text: str) -> bool:
    if not text:
        return False
    red_flags = [
        "chest pain", "difficulty breathing", "severe bleeding", "unconscious",
        "vomit blood", "stroke", "sudden weakness", "sudden numbness", "slurred speech"
    ]
    t = text.lower()
    return any(flag in t for flag in red_flags)

def fallback_response(prompt: str) -> str:
    if not prompt:
        return "Please describe your symptoms (onset, duration, severity)."
    p = prompt.lower()
    advice = []
    # support common English + Urdu keywords
    if any(k in p for k in ["fever", "temperature", "bukhar"]):
        advice += ["Possible causes: viral or bacterial infection.", "Precautions: rest, fluids, paracetamol (take per label)."]
    if any(k in p for k in ["headache", "sar dard", "sir dard"]):
        advice += ["Possible causes: tension, migraine, dehydration.", "Precautions: hydrate, rest, avoid bright light/noise."]
    if any(k in p for k in ["cough", "khansi"]):
        advice += ["Cough: may be viral; avoid smoke, rest, fluids.", "See doctor if cough persists >2 weeks or worsens."]
    if not advice:
        advice = ["Please provide more details: when it started, severity, any other symptoms (cough, rash, vomiting)."]
    advice.append("This is not a medical diagnosis. See a doctor if concerned or symptoms worsen.")
    return "\n".join(advice)

def estimate_confidence(text: str) -> int:
    if not isinstance(text, str) or not text.strip():
        return random.randint(0, 30)
    if text.startswith("⚠️ (AI API error"):
        return random.randint(30, 60)
    uncertain = ["maybe", "possible", "could be", "consider"]
    if any(w in text.lower() for w in uncertain):
        return random.randint(60, 80)
    return random.randint(82, 96)

def synthesize_speech(text: str, lang: str = "English"):
    try:
        code = 'en' if lang == "English" else 'ur'
        tts = gTTS(text=text, lang=code)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception:
        return None

def translate_text(text: str, dest: str):
    if not TRANSLATOR_AVAILABLE or translator is None:
        return text
    try:
        res = translator.translate(text, dest=dest)
        return res.text
    except Exception:
        return text

def compute_risk(text: str) -> tuple:
    if not text:
        return ("Low", 10)
    t = text.lower()
    score = 10
    if any(k in t for k in ["chest pain", "difficulty breathing", "shortness of breath", "vomit blood", "loss of consciousness", "stroke"]):
        return ("High", 95)
    if "fever" in t and any(k in t for k in ["rash", "breath", "chest"]):
        score += 60
    if "fever" in t:
        score += 20
    if "cough" in t:
        score += 10
    if "headache" in t:
        score += 5
    if any(k in t for k in ["vomit", "diarrhea", "nausea"]):
        score += 10
    score = min(95, max(10, score))
    if score >= 70:
        return ("High", score)
    if score >= 40:
        return ("Medium", score)
    return ("Low", score)

def ai_call(prompt: str, api_key: str, mode_name: str, language: str, context=None) -> str:
    """Single, safe ai_call. If Groq available & api_key provided, uses it; otherwise falls back."""
    persona = DOCTOR_MODES.get(mode_name, "")
    system_prompt = (
        f"You are an experienced medical AI assistant. {persona}\n"
        f"User language: {language}. Reply in same language.\n"
        "Provide possible causes, precautions, red flags, and when to consult a doctor.\n"
        "Always include: 'This is not a medical diagnosis.'"
    )
    try:
        # If Groq is available and api_key provided, attempt call
        if api_key and GROQ_AVAILABLE:
            try:
                client = Groq(api_key=api_key)
                messages = [{"role": "system", "content": system_prompt}]
                if context:
                    messages.extend(context[-3:])
                messages.append({"role": "user", "content": prompt})
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    max_tokens=700,
                    temperature=0.2
                )
                text = response.choices[0].message.content
                if not isinstance(text, str) or not text.strip():
                    return fallback_response(prompt)
                return text
            except Exception as e:
                print(f"Groq error: {e}")
                return fallback_response(prompt)
        # fallback: local rule-based reply
        return fallback_response(prompt)
    except Exception as e:
        print(f"ai_call unexpected error: {e}")
        return fallback_response(prompt)

def save_logs_to_local(df: pd.DataFrame, filename: str = "ai_doctor_logs.csv") -> bool:
    try:
        df.to_csv(filename, index=False)
        return True
    except Exception:
        return False

def export_pdf(df: pd.DataFrame):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    textobj = c.beginText(40, 800)
    textobj.setFont("Helvetica", 11)
    textobj.textLine("AI Doctor Pocket — Consultation Report")
    textobj.textLine("")
    for idx, row in df.iterrows():
        textobj.textLine(f"[{row['timestamp']}] You: {row['user']}")
        ai_text = str(row['ai']).replace("<br>", "\n")
        for line in ai_text.splitlines():
            # wrap lines at ~100 chars (simple)
            while len(line) > 100:
                textobj.textLine(f"AI: {line[:100]}")
                line = line[100:]
            textobj.textLine(f"AI: {line}")
        textobj.textLine(f"Lang: {row['language']}  Mode: {row['mode']}  Conf: {row['confidence']}  Risk: {row.get('risk','')}")
        textobj.textLine("-" * 80)
    c.drawText(textobj)
    c.save()
    buf.seek(0)
    return buf

# ---------------- UI Layout ----------------
st.markdown(f"<div class='topbar'><div><b>{APP_TITLE}</b></div><div>Built by AhmadXCode</div></div>", unsafe_allow_html=True)
st.markdown("")

if LOTTIE_AVAILABLE:
    try:
        r = requests.get("https://assets9.lottiefiles.com/packages/lf20_yg2caxn7.json", timeout=3)
        if r.status_code == 200:
            st_lottie(r.json(), height=120)
    except Exception:
        pass

# Two-column layout


# ------------------ Chat Input Section ------------------
col_main, col_side = st.columns([3, 1], gap="large")

with col_main:
    st.subheader("💬 Describe your symptoms or ask a health question")

    # Text area for input
    if "temp_input" not in st.session_state:
        st.session_state.temp_input = ""

    user_text = st.text_area(
        "💬 Type here (English / Urdu / Hindi)",
        height=140,
        key="temp_input",
    )

    # Quick action buttons
    send_col, clear_col, em_col = st.columns([1, 1, 1])

    with send_col:
        send = st.button("Send ▶️", use_container_width=True)
        

    with em_col:
        if st.button("🚨 Emergency?"):
            st.warning(
                "If you have chest pain, severe bleeding, difficulty breathing or other emergency — call local emergency services immediately."
            )

    # --- Process the message if Send pressed ---
    if send:
        if not user_text.strip():
            st.warning("⚠️ Please enter your symptoms first.")
        else:
            original_text = user_text.strip()

            # Language detection
            lang = safe_detect_language(original_text)
            text_for_ai = original_text
            if auto_translate and TRANSLATOR_AVAILABLE and lang != "English":
                try:
                    text_for_ai = translate_text(original_text, dest='en')
                except Exception:
                    text_for_ai = original_text

            # ----------------- HYBRID SMART MODE CHECK (NEW) -----------------
            # If user selected Mental Health but input contains physical/emergency keywords,
            # warn and offer to switch to General Physician for this consultation.
            mode_for_call = mode  # default: use selected mode
            if mode == "🧘 Mental Health":
                # physical keywords to detect mismatch
                physical_keywords = [
                    "fever", "temperature", "chest pain", "difficulty breathing",
                    "shortness of breath", "vomit", "vomiting", "bleeding", "blood",
                    "cough", "diarrhea", "nausea", "stomach", "abdominal", "severe",
                    "unconscious", "dizziness", "faint", "rash"
                ]
                text_lower = original_text.lower()
                found_physical = any(pk in text_lower for pk in physical_keywords)
                if found_physical:
                    st.warning(
                        "⚠️ You selected Mental Health, but your symptoms look like a physical condition. "
                        "Would you like to switch to General Physician for this consultation?"
                    )
                    switch = st.radio("Choose:", ["Keep Mental Health", "Switch to General Physician"], key=f"hybrid_switch_{int(time.time())}")
                    if switch == "Switch to General Physician":
                        mode_for_call = "👨‍⚕️ General Physician"
                        st.info("Mode switched to General Physician for this consultation.")
                    else:
                        st.info("Continuing with Mental Health mode as requested.")
            # ----------------------------------------------------------------

            # Emergency check
            is_emergency = emergency_check(original_text)
            if is_emergency:
                ai_text = "🚨 Emergency signs detected — please seek immediate medical attention or call emergency services!"
                conf = 100
                risk_label, risk_score = ("High", 95)
                st.markdown(f"<div class='emergency'>{ai_text}</div>", unsafe_allow_html=True)
            else:
                # Context: last few messages
                context_msgs = []
                for m in st.session_state.chat_history[-6:]:
                    role = "assistant" if m["role"] == "ai" else "user"
                    context_msgs.append({"role": role, "content": m["text"]})

                placeholder = st.empty()
                placeholder.markdown("<div class='typing'>AI Doctor is typing...</div>", unsafe_allow_html=True)
                # Use mode_for_call (which may be switched by Hybrid Smart check)
                ai_response = ai_call(text_for_ai, st.session_state.api_key, mode_for_call, lang, context_msgs if auto_translate else None)
                if auto_translate and TRANSLATOR_AVAILABLE and lang != "English":
                    try:
                        dest = 'ur' if lang == "Urdu" else 'hi' if lang == "Hindi" else 'en'
                        ai_text = translate_text(ai_response, dest=dest)
                    except Exception:
                        ai_text = ai_response
                else:
                    ai_text = ai_response
                placeholder.empty()

                conf = estimate_confidence(ai_text)
                risk_label, risk_score = compute_risk(original_text)

            # Store chat and logs
            st.session_state.chat_history.append({"role": "user", "text": original_text, "time": datetime.datetime.now()})
            st.session_state.chat_history.append({"role": "ai", "text": ai_text, "time": datetime.datetime.now()})

            new_row = {
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "user": original_text,
                "ai": ai_text,
                "language": lang,
                "mode": mode_for_call,
                "confidence": conf,
                "risk": f"{risk_label} ({risk_score}%)",
                "emergency": is_emergency
            }
            st.session_state.logs_df = pd.concat([st.session_state.logs_df, pd.DataFrame([new_row])], ignore_index=True)

            if save_local:
                try:
                    st.session_state.logs_df.to_csv("ai_doctor_logs.csv", index=False)
                    st.success("Saved logs to ai_doctor_logs.csv")
                except Exception:
                    st.error("Could not save logs locally.")

            if voice_reply:
                audio_bytes = synthesize_speech(ai_text, lang=lang)
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3")
                else:
                    st.info("Voice output unavailable.")


    # Show conversation (top-down so reply appears right under user)
    st.markdown("---")
    st.subheader("💬 Conversation")
    for msg in st.session_state.chat_history[-30:]:
        t = msg.get("time", "")
        try:
            t = msg["time"].strftime("%Y-%m-%d %H:%M")
        except Exception:
            t = ""
        content = msg['text'] if isinstance(msg['text'], str) else str(msg['text'])
        # support newlines in messages
        content_html = content.replace("\n", "<br>")
        if msg["role"] == "user":
            st.markdown(f"<div class='bubble-user'><b>You</b> <span class='muted' style='font-size:12px;'> {t}</span><br>{content_html}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bubble-ai'><b>AI</b> <span class='muted' style='font-size:12px;'> {t}</span><br>{content_html}</div>", unsafe_allow_html=True)

    # Health summary of last interaction
    if not st.session_state.logs_df.empty:
        last = st.session_state.logs_df.iloc[-1]
        st.markdown("<div class='summary'>", unsafe_allow_html=True)
        st.markdown(f"**🩺 Health Summary — {last['timestamp']}**")
        st.markdown(f"**Issue:** {last['user']}")
        st.markdown(f"**Advice:** {last['ai']}")
        st.markdown(f"**Confidence:** {last['confidence']}%")
        rlabel = str(last['risk'])
        if "High" in rlabel:
            st.markdown(f"<div class='badge-high'>⚠️ Risk: {rlabel}</div>", unsafe_allow_html=True)
        elif "Medium" in rlabel:
            st.markdown(f"<div class='badge-med'>⚠️ Risk: {rlabel}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='badge-low'>✅ Risk: {rlabel}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

with col_side:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.subheader("📊 Quick Dashboard")
    df = st.session_state.logs_df
    st.metric("Total Consultations", len(df))
    if not df.empty:
        avg_conf = int(df['confidence'].astype(int).mean())
        st.metric("Avg Confidence", f"{avg_conf:.1f}%")
        all_text = " ".join(df['user'].astype(str).tolist()).lower()
        tokens = [w.strip(".,!?") for w in all_text.split() if len(w) > 3]
        freq = {}
        stop = {"since","have","been","with","and","from","that","this","your"}
        for tkn in tokens:
            if tkn in stop: continue
            freq[tkn] = freq.get(tkn, 0) + 1
        top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:6]
        if top:
            st.markdown("**Top words in symptoms:**")
            st.write(", ".join([f"{w} ({c})" for w, c in top]))
        modes_count = df['mode'].value_counts().to_dict()
        st.markdown("**Conversations by Mode:**")
        st.write(modes_count)
        fig, ax = plt.subplots(figsize=(3, 2.2))
        ax.plot(df['confidence'].astype(int).tolist(), marker='o')
        ax.set_ylim(0, 100)
        ax.set_title("Confidence trend")
        ax.set_xlabel("Consultation #")
        ax.set_ylabel("Confidence %")
        st.pyplot(fig, clear_figure=True)
    else:
        st.info("No consultations yet — results show here after first chat.")
    st.markdown("</div>", unsafe_allow_html=True)

    # Export & Tools panel (still in right column)
    st.markdown("<div class='panel' style='margin-top:12px;'>", unsafe_allow_html=True)
    st.subheader("📥 Export & Tools")
    if not df.empty:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV", csv, file_name=f"ai_doctor_logs_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        txt = "\n\n".join([f"{r['timestamp']}\nYou: {r['user']}\nAI: {r['ai']}" for _, r in df.iterrows()])
        st.download_button("📄 Download TXT", txt, file_name="ai_doctor_chat.txt")
        pdf_buf = export_pdf(df)
        st.download_button("📕 Download PDF Report", pdf_buf, file_name="ai_doctor_report.pdf")
        if save_local:
            ok = save_logs_to_local(df)
            if ok: st.success("Saved logs to ai_doctor_logs.csv")
            else: st.error("Could not save logs locally.")
    else:
        st.info("No data to export yet.")
    st.markdown("### Emergency")
    st.markdown("If signs are serious, search nearby hospitals:")
    st.markdown("[🔎 Find nearby hospitals on Google Maps](https://www.google.com/maps/search/hospital+near+me)")
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<footer>Built with ❤️ by AhmadXCode • Demo app — for educational purposes only.</footer>", unsafe_allow_html=True)
