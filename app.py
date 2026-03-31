import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

# 🔥 Page Config
st.set_page_config(
    page_title="Traffic Sign Classifier",
    page_icon="🚦",
    layout="centered"
)

# 🔥 Custom Dark Theme Styling
st.markdown("""
    <style>
    body {
        background-color: #0e1117;
        color: white;
    }
    .stApp {
        background-color: #0e1117;
    }
    </style>
""", unsafe_allow_html=True)

# 🔥 Title
st.markdown("<h1 style='text-align: center;'>🚦 Traffic Sign Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload or drag & drop an image to classify</p>", unsafe_allow_html=True)

# 🔥 Build Model
def build_model():
    model = Sequential()

    model.add(Conv2D(32, (5,5), activation='relu', input_shape=(30,30,3)))
    model.add(Conv2D(32, (5,5), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(43, activation='softmax'))

    return model

# 🔥 Load Model (cached)
@st.cache_resource
def load_model_cached():
    model = build_model()
    model.load_weights("model/weights.weights.h5")
    return model

model = load_model_cached()

# 🔥 Class Labels
classes = {
    0:'Speed limit (20km/h)', 1:'Speed limit (30km/h)', 2:'Speed limit (50km/h)',
    3:'Speed limit (60km/h)', 4:'Speed limit (70km/h)', 5:'Speed limit (80km/h)',
    6:'End of speed limit (80km/h)', 7:'Speed limit (100km/h)', 8:'Speed limit (120km/h)',
    9:'No passing', 10:'No passing veh over 3.5 tons', 11:'Right-of-way at intersection',
    12:'Priority road', 13:'Yield', 14:'Stop', 15:'No vehicles',
    16:'Veh > 3.5 tons prohibited', 17:'No entry', 18:'General caution',
    19:'Dangerous curve left', 20:'Dangerous curve right', 21:'Double curve',
    22:'Bumpy road', 23:'Slippery road', 24:'Road narrows on right',
    25:'Road work', 26:'Traffic signals', 27:'Pedestrians',
    28:'Children crossing', 29:'Bicycles crossing', 30:'Beware of ice/snow',
    31:'Wild animals crossing', 32:'End speed + passing limits',
    33:'Turn right ahead', 34:'Turn left ahead', 35:'Ahead only',
    36:'Go straight or right', 37:'Go straight or left', 38:'Keep right',
    39:'Keep left', 40:'Roundabout mandatory', 41:'End of no passing',
    42:'End no passing veh > 3.5 tons'
}

# 🔥 Upload (Drag & Drop supported automatically)
uploaded_file = st.file_uploader(
    "📤 Drag & Drop or Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Display Image
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    # 🔥 Preprocess
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (30, 30))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    # 🔥 Predict
    predictions = model.predict(image)[0]

    # 🔥 Top-3 Predictions
    top_indices = predictions.argsort()[-3:][::-1]

    st.markdown("## 🎯 Top Predictions")

    for i, idx in enumerate(top_indices):
        confidence = predictions[idx]
        st.write(f"**{i+1}. {classes[idx]} — {confidence*100:.2f}%**")

    # 🔥 Confidence warning
    if np.max(predictions) < 0.5:
        st.warning("⚠️ Low confidence prediction")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>🚀 Built with Streamlit</p>", unsafe_allow_html=True)