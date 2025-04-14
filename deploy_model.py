import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- APP CONFIG ---
st.set_page_config(
    page_title="🧪 Leaf Diagnosis AI",
    page_icon="🍃",
    layout="wide"
)

# --- LOAD MODEL ONCE ---
@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model("trained_model.keras")

model = load_trained_model()

# --- CUSTOM STYLES ---
st.markdown("""
<style>
h1, h2, h3 {
    color: #2E8B57;
}
.sidebar .sidebar-content {
    background-color: #F0FFF0;
}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- PREDICTION FUNCTION ---
def classify_leaf(image_file):
    image = Image.open(image_file).convert("RGB").resize((128, 128))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_batch = np.expand_dims(image_array, axis=0)
    predictions = model.predict(image_batch)
    return np.argmax(predictions)

# --- CLASS LABELS ---
CLASSES = [
    'Apple - Apple Scab', 'Apple - Black Rot', 'Apple - Cedar Apple Rust', 'Apple - Healthy',
    'Blueberry - Healthy', 'Cherry - Powdery Mildew', 'Cherry - Healthy',
    'Corn - Gray Leaf Spot', 'Corn - Common Rust', 'Corn - Northern Leaf Blight', 'Corn - Healthy',
    'Grape - Black Rot', 'Grape - Esca', 'Grape - Leaf Blight', 'Grape - Healthy',
    'Orange - Citrus Greening', 'Peach - Bacterial Spot', 'Peach - Healthy',
    'Bell Pepper - Bacterial Spot', 'Bell Pepper - Healthy', 'Potato - Early Blight',
    'Potato - Late Blight', 'Potato - Healthy', 'Raspberry - Healthy', 'Soybean - Healthy',
    'Squash - Powdery Mildew', 'Strawberry - Leaf Scorch', 'Strawberry - Healthy',
    'Tomato - Bacterial Spot', 'Tomato - Early Blight', 'Tomato - Late Blight',
    'Tomato - Leaf Mold', 'Tomato - Septoria Leaf Spot', 'Tomato - Spider Mites',
    'Tomato - Target Spot', 'Tomato - Yellow Leaf Curl Virus', 'Tomato - Mosaic Virus', 'Tomato - Healthy'
]

# --- SIDEBAR ---
st.sidebar.title("🍃 Leaf Diagnosis")
section = st.sidebar.radio("Navigate to", ["🌍 Dashboard", "📚 Research Lab", "🔎 Try Diagnosis"])

# --- DASHBOARD PAGE ---
if section == "🌍 Dashboard":
    st.title("🧬 Leaf Diagnosis: Deep Learning Edition")
    st.subheader("Early Detection. Smart Agriculture. Happy Plants.")
    image_path = "leaf.png"
    st.image(image_path, use_column_width=True)

    st.markdown("""
Welcome to **Leaf Diagnosis**, a deep learning powered web tool for detecting plant diseases through images.  
Use this tool to understand what might be affecting your crops — powered by AI, trained on thousands of real samples! 🌾🔬

#### 🚀 Why This Project?
- Reduce crop loss with early detection
- Empower gardeners & farmers
- Explore cutting-edge AI in agriculture

#### 🛠 Features:
- Fast ⚡️ image analysis
- Over 38 plant disease categories
- Open-source friendly

👉 Ready to test it out? Head to **Try Diagnosis** from the sidebar!
    """)

# --- ABOUT / LAB PAGE ---
elif section == "📚 Research Lab":
    st.title("🧪 The Research Behind the Roots")
    st.markdown("""
**Dataset Origin:**  
Recreated from the Plant Village Dataset with image augmentations for improved generalization.

**Model Architecture:**  
Convolutional Neural Network (CNN) trained on 87K+ RGB images across 38 leaf disease categories.

**Train/Valid Split:**  
- Training: 70,295 images  
- Validation: 17,572 images  
- Test Set: 33 images (for blind prediction)

**Technologies Used:**
- 📦 TensorFlow / Keras
- 📊 Numpy, Pandas
- 🖥️ Streamlit

Find the full codebase on GitHub:  
[🌿 leaf-diagnosis-deep-learning](https://github.com/yzdpirate28/leaf-diagnosis-deep-learning)
""")

# --- PREDICTION PAGE ---
elif section == "🔎 Try Diagnosis":
    st.title("🔎 Diagnose Your Leaf")
    st.markdown("Upload an image of a leaf, and let our AI take care of the rest.")

    uploaded_file = st.file_uploader("📂 Choose a leaf image (JPG, PNG)", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        st.image(uploaded_file, caption="🖼️ Uploaded Leaf", use_column_width=True)

        if st.button("🧠 Run Diagnosis"):
            with st.spinner("Analyzing with Deep Neural Roots..."):
                idx = classify_leaf(uploaded_file)
                result = CLASSES[idx]
                st.success(f"🧾 Diagnosis Result: **{result}**")
