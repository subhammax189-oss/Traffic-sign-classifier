import numpy as np
from preprocess import preprocess_image

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

# 🔥 Build same model architecture used in training
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


# 🔥 Load model + weights
model = build_model()
model.load_weights("../model/weights.weights.h5")

print("✅ Model loaded successfully!")


# 🔥 Class labels
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


# 🔥 Prediction function
def predict(image_path):
    image = preprocess_image(image_path)

    predictions = model.predict(image)

    class_index = np.argmax(predictions)
    confidence = np.max(predictions)

    return classes[class_index], confidence


# 🔥 Main execution
if __name__ == "__main__":
    img_path = "../static/test.jpeg"   # make sure filename matches

    print("🔍 Running prediction...")

    label, conf = predict(img_path)

    print("\n✅ Prediction Result:")
    print(f"Class: {label}")
    print(f"Confidence: {conf*100:.2f}%")