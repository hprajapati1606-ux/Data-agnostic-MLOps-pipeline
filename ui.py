import streamlit as st
import pandas as pd
import os
import joblib
import requests
from streamlit_lottie import st_lottie

# Import Backend Modules
from src.ingestion.ingest_data import load_data
from src.preprocessing.preprocess import preprocess_data
from src.training.train import train_model

# Page Config
st.set_page_config(page_title="Universal MLOps Pipeline", page_icon="🧠", layout="wide")

def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200: return None
        return r.json()
    except: return None

# UI Header
col1, col2 = st.columns([1, 4])
with col1:
    lottie_ai = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_qp1q7mct.json")
    if lottie_ai: st_lottie(lottie_ai, height=150, key="ai")
with col2:
    st.title("🧠 Universal MLOps Pipeline")
    st.write("### Data Agnostic Modeling System")
    st.write("Automated Ingestion • Preprocessing • Training • Inference")

st.markdown("---")

tab1, tab2 = st.tabs(["🚂 **Train Model**", "🔮 **Prediction Interface**"])

# --- TAB 1: TRAIN ---
with tab1:
    st.subheader("Step 1: Ingest & Train")
    uploaded_file = st.file_uploader("Upload Dataset (CSV, XLSX, JSON)", type=["csv", "xlsx", "json"])

    if uploaded_file:
        save_dir = "data/raw"
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, uploaded_file.name)
        with open(file_path, "wb") as f: f.write(uploaded_file.getbuffer())
        
        # Load & Preview
        df = load_data(file_path)
        st.write("📊 Data Preview:")
        st.dataframe(df.head(3))
        
        # Select Target
        all_cols = df.columns.tolist()
        target_col = st.selectbox("🎯 Select Target Column to Predict:", all_cols, index=len(all_cols)-1)
        
        if st.button("🚀 Initialize Training Pipeline", type="primary"):
            with st.spinner("Processing Pipeline..."):
                try:
                    # Preprocessing
                    df_clean = preprocess_data(df)
                    
                    # Save Processed Data
                    processed_path = "data/processed/clean_data.csv"
                    os.makedirs("data/processed", exist_ok=True)
                    df_clean.to_csv(processed_path, index=False)
                    
                    # Training
                    model_path = "models/model.pkl"
                    train_model(processed_path, model_path, target_col)
                    
                    st.success(f"✅ Training Complete! Model optimized for target: '{target_col}'")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"❌ Pipeline Error: {e}")

# --- TAB 2: PREDICT ---
# =========================================
# TAB 2: PREDICTION DASHBOARD (Visual Version)
# =========================================
# =========================================
# TAB 2: PREDICTION INTERFACE (Clean & Centered)
# =========================================
with tab2:
    st.header("🔮 Prediction Interface")
    
    model_path = "models/model.pkl"
    meta_path = "models/model_meta.pkl"
    
    # Check if model exists
    if os.path.exists(model_path) and os.path.exists(meta_path):
        try:
            # 1. Load Model & Metadata
            model = joblib.load(model_path)
            meta = joblib.load(meta_path)
            target_col = meta.get('target_col', 'Unknown')
            problem_type = meta.get('problem_type', 'Unknown')

            # 2. Input Form (Ab ye Sidebar mein nahi, Main Page pe dikhega)
            st.info(f"🧠 Model is Ready! Predicting Target: **{target_col}**")
            
            # Generate Inputs Automatically based on columns
            processed_path = "data/processed/clean_data.csv"
            if os.path.exists(processed_path):
                df_schema = pd.read_csv(processed_path)
                feature_cols = [c for c in df_schema.columns if c != target_col]
                
                # Form Container
                with st.form("inference_form"):
                    st.write("### 🎛️ Enter Input Parameters:")
                    
                    # Columns create karte hain taaki inputs sundar dikhein
                    cols = st.columns(3) # 3 columns grid layout
                    input_data = {}
                    
                    for i, col in enumerate(feature_cols):
                        with cols[i % 3]: # Grid logic
                            input_data[col] = st.number_input(f"{col}", value=0.0)
                    
                    st.markdown("---")
                    submit = st.form_submit_button("🚀 Run Prediction", type="primary")

            # 3. PREDICTION LOGIC & DASHBOARD
            if submit:
                # Prepare Data
                input_df = pd.DataFrame([input_data])
                prediction = model.predict(input_df)[0]
                
                # --- VISUALIZATION ---
                st.divider()
                
                # Classification Logic (Categories)
                if problem_type == "Classification":
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        st.subheader("🎯 Result")
                        st.metric(label="Predicted Category", value=f"Class {int(prediction)}")
                        st.caption("Note: Class labels depend on your data (e.g., 0=No, 1=Yes).")

                    with c2:
                        st.subheader("🤖 Confidence Score")
                        if hasattr(model, "predict_proba"):
                            proba = model.predict_proba(input_df)[0]
                            chart_data = pd.DataFrame({
                                "Category": list(range(len(proba))),
                                "Confidence": proba
                            })
                            st.bar_chart(chart_data.set_index("Category"))
                        else:
                            st.warning("⚠️ Probability score not available for this model.")

                # Regression Logic (Numbers)
                else:
                    st.subheader("📈 Estimation Result")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric(label=f"Predicted {target_col}", value=f"{prediction:.2f}")
                    with c2:
                        st.success(f"Estimated value for **{target_col}** is **{prediction:.2f}**")

        except Exception as e:
            st.error(f"⚠️ Error loading model: {e}")
            st.write("Tip: Try retraining the model in the 'Train' tab.")
    else:
        # Agar Model nahi hai to ye dikhega
        st.warning("🚧 No Model Found! Please train a model in the 'Train Model' tab first.")
        st.image("https://assets5.lottiefiles.com/packages/lf20_qp1q7mct.json", width=200) # Robot waiting