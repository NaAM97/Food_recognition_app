import streamlit as st
import requests
from PIL import Image
import io
import base64
from time import sleep

# Define your classes for predictions
# classes = ['fish_and_chips', 'french_toast', 'fried_calamari', 'garlic_bread', 'grilled_salmon', 'hamburger', 'ice_cream', 'lasagna', 'macaroni_and_cheese', 'macarons']
def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(
            md,
            unsafe_allow_html=True,
        )

def get_prediction(image_bytes):
    # Define the correct API endpoint URL
    url = "https://nfg-repo-rmhuz6i5bq-ew.a.run.app/predict"
    # Create an in-memory file-like object
    files = {'file': ('image.jpg', image_bytes, 'image/jpeg')}

    try:
        response = requests.post(url, files=files)
        if response.status_code == 200:
            # Assuming the server responds with JSON
            return response.json()
        else:
            return f"Error: Received status code {response.status_code}"
    except requests.RequestException as e:
        return f"Request failed: {e}"

# Streamlit application setup
st.set_page_config(page_title="Food Recognition", page_icon=":shallow_pan_of_food:", layout="wide")
st.title("Food Image Recognition")
st.write("Upload an image of a plate, and we'll recognize the food placed in it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "jfif"])

if uploaded_file:
    # Display the uploaded image
    image = Image.open(io.BytesIO(uploaded_file.getvalue()))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert the uploaded file to bytes for the request
    img_bytes = uploaded_file.getvalue()
    prediction = get_prediction(img_bytes)

    # Display the prediction result
    st.write(f"Prediction: {prediction['class']}")
    # st.audio(f"./audios/{prediction['class']}.mp3",autoplay=True)


    st.write("# Auto-playing Audio!")
    sleep(1)
    autoplay_audio(f"./audios/{prediction['class']}.mp3")
