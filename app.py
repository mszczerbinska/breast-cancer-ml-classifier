import streamlit as st
import pandas as pd
import joblib
from sklearn.datasets import load_breast_cancer

# 1. Page configuration
st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon="🩺",
    layout="wide"
)

# 2. Load the trained model and scaler
@st.cache_resource
def load_ml_components():
    model = joblib.load('models/breast_cancer_log_reg_model.pkl')
    scaler = joblib.load('models/breast_cancer_scaler.pkl')
    return model, scaler

try:
    model, scaler = load_ml_components()
except FileNotFoundError:
    st.error("Error: Model files not found. Please ensure .pkl files are in the 'models/' directory.")
    st.stop()

# 3. Load dataset info to automatically generate sliders with correct ranges
data = load_breast_cancer()
feature_names = data.feature_names
df_baseline = pd.DataFrame(data.data, columns=feature_names)

# 4. UI Layout - Main Page
st.title("🩺 Breast Cancer Diagnostic Assistant")
st.markdown("""
This application uses a Machine Learning model (Logistic Regression) to predict whether a breast mass is **Malignant** or **Benign** based on cell nuclei measurements from a Fine Needle Aspirate (FNA).
""")
st.divider()

# 5. UI Layout - Sidebar for inputs
st.sidebar.header("Input Clinical Parameters")
st.sidebar.info(
    "💡 **Where do these values come from?**\n"
    "These features are computed by specialized medical software from a "
    "digitized image of a Fine Needle Aspirate (FNA) biopsy. In reality, "
    "a pathologist or lab system would provide this data."
)
st.sidebar.write("Adjust the sliders to set the cell properties.")

input_data = {}

# Dynamically generate sliders for all 30 features in the sidebar
for feature in feature_names:
    min_val = float(df_baseline[feature].min())
    max_val = float(df_baseline[feature].max())
    mean_val = float(df_baseline[feature].mean())
    
    input_data[feature] = st.sidebar.slider(
        label=feature,
        min_value=min_val,
        max_value=max_val,
        value=mean_val
    )

# 6. Prediction Logic
if st.button("Run Diagnostic Prediction", type="primary"):
    
    # Convert input dict to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Scale the input data using the loaded scaler
    scaled_input = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(scaled_input)
    prediction_proba = model.predict_proba(scaled_input)
    
    # In sklearn dataset: 0 = Malignant, 1 = Benign
    st.subheader("Diagnostic Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction[0] == 0:
            st.error("### 🚨 Diagnosis: Malignant")
            confidence = prediction_proba[0][0] * 100
        else:
            st.success("### ✅ Diagnosis: Benign")
            confidence = prediction_proba[0][1] * 100
            
    with col2:
        st.metric(label="Model Confidence", value=f"{confidence:.2f}%")
        
    st.info("Disclaimer: This tool is a Proof of Concept and should not be used as a substitute for professional medical advice, diagnosis, or treatment.")
