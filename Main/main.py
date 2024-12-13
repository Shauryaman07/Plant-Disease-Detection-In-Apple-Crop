import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the TensorFlow model using a relative path
MODEL_PATH = "C:/Users/shaur/OneDrive/Desktop/PlantDiseaseDetection/models/applesNew.keras"
try:
    MODEL = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()  # Stop the app if the model can't be loaded

CLASS_NAMES = ["Healthy", "Rust", "Scab"]  # Updated class order

# Function for model prediction
def model_prediction(image):
    image = image.resize((256, 256))  # Resize to match model input
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)  # Add batch dimension
    predictions = MODEL.predict(input_arr)
    return CLASS_NAMES[np.argmax(predictions)], np.max(predictions)

# Function to validate uploaded image
def validate_image(image):
    # Example validation logic: check if the image size is reasonable
    # You can implement more sophisticated checks based on your requirements
    if image.size[0] < 100 or image.size[1] < 100:  # Minimum size check
        return False
    return True

# Custom CSS for styling
st.markdown("""
    <style>
    body {
        background-color: #d3f9d8; 
        font-family: 'Arial', sans-serif; 
    }
    .title {
        color: #3e8e41; 
        font-size: 2.5em;
    }
    .header {
        color: #4b6f9e; 
    }
    .stButton>button {
        background-color: #4CAF50; 
        color: white;
        font-size: 16px;
        border: None;
        border-radius: 5px;
        padding: 10px;
    }
    .stButton>button:hover {
        background-color: #45a049; 
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Plant Disease Detection")
app_mode = st.sidebar.selectbox("Select Page", ["About", "Disease Recognition"])

# About Page
if app_mode == "About":
    st.markdown('<p class="title">🌿 PLANT DISEASE RECOGNITION SYSTEM 🌿</p>', unsafe_allow_html=True)

    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    This application utilizes deep learning to identify diseases in apple plants from uploaded images.
    Navigate to the "Disease Recognition" page to upload a leaf image,
    and our model will predict its health status.
    """)

    st.markdown('<p class="header">🔍 About</p>', unsafe_allow_html=True)
    st.markdown("""
    #### About the Dataset
    This application is designed to recognize diseases in apple plants. 
    The dataset comprises images of apple leaves categorized into three classes:
    - **Healthy**
    - **Rust**
    - **Scab**
    
    The model has been trained using TensorFlow and Keras to provide accurate predictions.
    """)

    st.markdown("""
    ### Leaf Conditions:
    - **Healthy Leaves**: These leaves are vibrant green and firm, indicating that the plant is receiving proper nutrients and water. They are free from spots, discoloration, or wilting.
    
    - **Rust Leaves**: Leaves affected by rust disease may show orange or reddish-brown spots on the upper side, with a powdery substance on the underside. This fungal infection can weaken the plant and reduce yield if not managed promptly.
    
    - **Scab Leaves**: Apple scab typically presents as dark, olive-green spots on the upper leaf surface. As the disease progresses, these spots can turn brown and cause the leaves to curl and drop prematurely, affecting the overall health of the tree.
    """)

    sample_images = {
        "Healthy": "C:/Users/shaur/OneDrive/Desktop/PlantDiseaseDetection/images/healthy.jpg",
        "Rust": "C:/Users/shaur/OneDrive/Desktop/PlantDiseaseDetection/images/rust.jpg",
        "Scab": "C:/Users/shaur/OneDrive/Desktop/PlantDiseaseDetection/images/scab.jpg",
    }

    st.subheader("Sample Images")
    cols = st.columns(3)
    for idx, (label, path) in enumerate(sample_images.items()):
        with cols[idx]:
            st.image(path, caption=label, use_column_width=True)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.markdown('<p class="header">🌱 Disease Recognition</p>', unsafe_allow_html=True)
    
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])
    
    if test_image is not None:
        image = Image.open(test_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Validate the uploaded image
        if not validate_image(image):
            st.error("Please upload a valid image of an apple leaf.")
        else:
            if st.button("Predict"):
                with st.spinner("Predicting... Please wait."):
                    class_name, confidence = model_prediction(image)
                    st.success(f"Model predicts: **{class_name}** with confidence **{confidence:.2f}**")

                    predictions = MODEL.predict(np.expand_dims(image.resize((256, 256)), axis=0))
                    
                    plt.clf()
                    plt.figure(figsize=(10, 5))
                    plt.bar(CLASS_NAMES, predictions[0], color=['lightgreen', 'lightblue', 'lightcoral'])
                    plt.xlabel('Classes', fontsize=14)
                    plt.ylabel('Confidence', fontsize=14)
                    plt.title('Prediction Confidence', fontsize=16)
                    st.pyplot(plt)

                    recommendations = {
                        "Scab": """
                        ### Recommendation: 
                        - Remove infected leaves and apply fungicide as needed.
                        - Improve air circulation around plants to reduce humidity.
                        - Water plants at the base to keep leaves dry.
                        """,
                        "Rust": """
                        ### Recommendation: 
                        - Prune affected areas to prevent spread.
                        - Ensure proper sanitation by cleaning up fallen leaves.
                        - Apply fungicides specifically for rust control.
                        """,
                        "Healthy": """
                        ### Good News! Your plant is healthy! 
                        Keep up the good care! Regularly monitor your plants for any signs of diseases.
                        """
                    }
                    st.markdown(recommendations[class_name])
