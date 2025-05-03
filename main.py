import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Set wide layout for single-screen display
st.set_page_config(page_title="Pomegranate Disease Classification", layout="wide")

# Class labels
class_names = ["Alternaria", "Anthracnose", "Bacterial Blight", "Cercospora", "Healthy"]

# Load the model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("DenseNet161.keras")

model = load_model()

# Model Info
best_model_name = "Final Fine-Tuned Model"
best_model_accuracy = 92.5  # Example accuracy

# Main Title
st.title("🍎 Pomegranate Disease Classification using Deep Learning")

# File uploader
uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert image to RGB and resize
    image = Image.open(uploaded_file).convert("RGB")
    img_resized = image.resize((224, 224))
    
    # Preprocess the image
    img_array = np.array(img_resized) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Ensure 3 channels
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    # Prediction
    predictions = model.predict(img_array)[0]
    probabilities = predictions * 100
    predicted_class_index = np.argmax(predictions)
    predicted_class_name = class_names[predicted_class_index]
    confidence = probabilities[predicted_class_index]

    # Display in a single window using columns
    col1, col2 = st.columns([1.2, 1])  # Adjust column width ratios

    # Left Column - Image
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=False)

    # Right Column - Prediction Details
    with col2:
        st.metric(label="🧠 Best Model", value=best_model_name)
        st.metric(label="🎯 Predicted Disease", value=predicted_class_name)
        st.metric(label="🔬 Confidence", value=f"{confidence:.2f}%")
        
        # Display all probabilities compactly
        st.write("### 📊 Class Probabilities:")
        df = {class_names[i]: [f"{probabilities[i]:.2f}%"] for i in range(len(class_names))}
        # st.dataframe(df, use_container_width=True, height=150)  # Compact table
        st.table(df)

     