import streamlit as st
import faiss
import numpy as np
import pickle
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import logic 

# ============================================================
# ⚠️ SECURE CONFIGURATION (Gemini 2.5 Flash)
# ============================================================
# Load API Key from .env file for security
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("❌ API Key not found. Please ensure your .env file is set up correctly.")
    st.stop()

genai.configure(api_key=API_KEY)

@st.cache_resource
def load_assets():
    """Load and cache heavy assets like models and vector indices."""
    model = SentenceTransformer('BAAI/bge-m3')
    index = faiss.read_index("health_index.faiss")
    with open("doc_map.pkl", "rb") as f:
        doc_map = pickle.load(f)
    return model, index, doc_map

# Initialize the Engine
embed_model, faiss_index, doc_map = load_assets()

# ============================================================
# 🎨 UI & BRANDING (Clinical Aesthetics)
# ============================================================
st.set_page_config(page_title="AgeWell", layout="wide", page_icon="🧬")

st.markdown("""
    <style>
    .main { background-color: #ffffff; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3.5em; background-color: #1a73e8; color: white; font-weight: bold; border: none; }
    .report-box { padding: 30px; border-radius: 15px; border: 1px solid #dce1e6; background-color: #fdfdfd; box-shadow: 2px 2px 15px rgba(0,0,0,0.02); line-height: 1.6; }
    .hook-container { background-color: #e8f0fe; padding: 20px; border-radius: 12px; border-left: 5px solid #1a73e8; margin-top: 10px; margin-bottom: 25px; }
    .welcome-card { padding: 40px; background-color: #f8f9fa; border-radius: 15px; border: 1px solid #eee; }
    .disclaimer { margin-top: 40px; padding: 15px; background-color: #fff3cd; border-left: 4px solid #ffc107; border-radius: 5px; font-size: 0.85em; color: #856404; }
    </style>
    """, unsafe_allow_html=True)

# --- Header Section ---
st.title("🧬 AgeWell")
st.markdown("##### *Age better with evidence.*")
st.markdown("###### *Advanced Clinical Reasoning Engine — Powered by 1100+ Semantic Chunks*")

# --- The Empathy Bridge ---
st.markdown("""
    <div class="hook-container">
        <p style="margin: 0; color: #1a73e8; font-weight: bold; font-size: 1.1em;">🔬 Confused by your latest health checkup?</p>
        <p style="margin: 5px 0 0 0; color: #444; font-size: 0.95em; line-height: 1.5;">
            Just finished a medical exam but don't know where to start? 
            <b>AgeWell</b> translates complex lab results into a direct, evidence-based roadmap.
        </p>
    </div>
""", unsafe_allow_html=True)

# ============================================================
# 📋 SIDEBAR: COMPREHENSIVE INTAKE
# ============================================================
with st.sidebar:
    st.header("📋 Health Intake")
    st.info("💡 **Evidence-Based:** Data is cross-referenced with 1,100+ medical chunks.")
    
    with st.form("agewell_intake"):
        name = st.text_input("Nickname*", value="Guest")
        
        st.subheader("Core Biometrics")
        age = st.number_input("Age*", 18, 110, 35)
        gender = st.selectbox("Biological Sex*", ["Male", "Female"])
        height = st.number_input("Height (cm)*", 100, 250, 170)
        weight = st.number_input("Weight (kg)*", 30, 250, 70)
        
        activity = st.select_slider(
            "Current Activity Level*", 
            options=["Sedentary", "Light", "Moderate", "Active", "Athlete"],
            value="Moderate"
        )
        
        with st.expander("📊 Detailed Lab Results (Optional)"):
            st.caption("Missing values will be assumed as stable/normal.")
            systolic_bp = st.number_input("Systolic BP (mmHg)", 0, 250, 0)
            apob = st.number_input("ApoB (mg/dL)", 0, 300, 0)
            hba1c = st.number_input("HbA1c (%)", 0.0, 15.0, 0.0)
            glucose = st.number_input("Fasting Glucose (mg/dL)", 0, 400, 0)
            vo2max = st.number_input("VO2 Max", 0.0, 90.0, 0.0)
            vit_d = st.number_input("Vitamin D (ng/mL)", 0, 200, 0)
            myopia = st.number_input("Myopia Diopters", -25.0, 0.0, 0.0)

        st.write("---")
        goal = st.selectbox("Primary Longevity Focus", ["Lifespan Extension", "Metabolic Optimization", "Athletic Performance", "Cognitive Health"])
        
        submitted = st.form_submit_button("🚀 GENERATE MY STRATEGY")

# ============================================================
# 🚀 RAG & DIAGNOSTIC ENGINE
# ============================================================
if submitted:
    # 1. Calculation Logic (BMI)
    bmi = round(weight / ((height/100)**2), 1)
    
    # 2. Package Data for logic.py
    profile = {
        "name": name, "age": age, "gender": gender, "bmi": bmi, "height": height, "weight": weight,
        "activity": activity,
        "systolic_bp": systolic_bp if systolic_bp > 0 else None,
        "apob": apob if apob > 0 else None,
        "hba1c": hba1c if hba1c > 0 else None,
        "glucose": glucose if glucose > 0 else None,
        "vo2max": vo2max if vo2max > 0 else None,
        "vit_d": vit_d if vit_d > 0 else None,
        "myopia": myopia if myopia != 0 else None,
        "goal": goal
    }

    with st.status("🧬 Consulting AgeWell Research Engine...", expanded=True) as status:
        # Step A: Vector Search (RAG)
        search_query = f"{age}yo {gender} BMI {bmi} {activity} activity {goal} " + (f"ApoB {apob}" if apob else "")
        q_emb = embed_model.encode([search_query], normalize_embeddings=True)
        _, indices = faiss_index.search(np.array(q_emb).astype('float32'), 3)
        context = "\n\n".join([doc_map[idx] for idx in indices[0]])
        
        # Step B: LLM Synthesis with Gemini 2.5 Flash
        final_prompt = logic.get_clinical_prompt(profile, context)
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(final_prompt)
        
        status.update(label="✅ Strategy Synthesized!", state="complete")

    # --- 3. DISPLAY RESULTS (Clinical Grade Sanitization) ---
    st.markdown(f"### 📑 Comprehensive Longevity Strategy for {name}")
    st.info(f"**Calculated BMI:** {bmi} | **Clinical Goal:** {goal}")
    
    # 3.1. Aggressive Markdown Sanitization (Removes hashes and stars)
    clean_text = response.text.replace("**", "").replace("*", "").replace("###", "").replace("##", "").replace("#", "")
    
    # 3.2. Inject Structured HTML Styling
    styled_report = clean_text.replace("EXECUTIVE SUMMARY:", "<h4 style='color:#1a73e8; border-bottom:1px solid #eee;'>🩺 EXECUTIVE SUMMARY</h4>") \
                              .replace("CLINICAL MECHANISM:", "<h4 style='color:#1a73e8; border-bottom:1px solid #eee;'>🧠 CLINICAL MECHANISM</h4>") \
                              .replace("SHIFT:", "<br><strong style='color:#d93025; font-size:1.2em;'>🔴 SHIFT:</strong>") \
                              .replace("FIX:", "<br><strong style='color:#f9ab00; font-size:1.2em;'>🟡 FIX:</strong>") \
                              .replace("ENHANCE:", "<br><strong style='color:#1e8e3e; font-size:1.2em;'>🟢 ENHANCE:</strong>") \
                              .replace("EXPECTED OUTCOMES:", "<h4 style='color:#1a73e8; border-bottom:1px solid #eee;'>📈 90-DAY EXPECTATIONS</h4>") \
                              .replace("SCIENTIFIC FOOTNOTE:", "<div style='background-color:#f0f4f8; padding:15px; border-radius:8px; margin-top:20px; border-left:4px solid #1a73e8;'>🧪 <i>Scientific Footnote:</i>")
    
    if "Scientific Footnote:" in styled_report:
        styled_report += "</div>"

    # 3.3. Convert remaining newlines to HTML breaks
    formatted_report = styled_report.replace('\n', '<br>')
    
    # 3.4. Final Render in custom CSS Box
    st.markdown(f"""
        <div class="report-box">
            {formatted_report}
        </div>
    """, unsafe_allow_html=True)

else:
    # --- Welcome Interface ---
    st.markdown("""
        <div class="welcome-card">
            <h2 style="color: #1a73e8;">Welcome to AgeWell</h2>
            <p>Our RAG engine integrates your physical metrics with over 1,100 clinical evidence chunks to create a roadmap that matches your biology.</p>
            <div style="display: flex; gap: 20px; margin-top: 20px;">
                <div style="flex: 1; padding: 20px; background: white; border-radius: 10px; border: 1px solid #eee;">
                    <h4 style="margin: 0; color: #1a73e8;">1. Input</h4>
                    <p style="font-size: 0.9em;">Enter your biometrics and lab results in the sidebar.</p>
                </div>
                <div style="flex: 1; padding: 20px; background: white; border-radius: 10px; border: 1px solid #eee;">
                    <h4 style="margin: 0; color: #1a73e8;">2. Reason</h4>
                    <p style="font-size: 0.9em;">AI analyzes your profile against peer-reviewed science.</p>
                </div>
                <div style="flex: 1; padding: 20px; background: white; border-radius: 10px; border: 1px solid #eee;">
                    <h4 style="margin: 0; color: #1a73e8;">3. Act</h4>
                    <p style="font-size: 0.9em;">Receive your prioritized Shift/Fix/Enhance roadmap.</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# ============================================================
# ⚖️ MEDICAL DISCLAIMER
# ============================================================
st.markdown("""
    <div class="disclaimer">
        <strong>⚠️ Medical Disclaimer:</strong> This report is generated by an AI research engine for informational purposes only. 
        It is not a substitute for professional medical advice, diagnosis, or treatment. 
        Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
    </div>
""", unsafe_allow_html=True)