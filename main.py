import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
import joblib

model = joblib.load('emotion_model.pkl')

cap = cv2.VideoCapture(0)

emotion_dict = {
    'angry': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3,
    'neutral': 4,
    'sad': 5,
    'surprise': 6
}

reverse_emotion_dict = {v: k for k, v in emotion_dict.items()}

# ----------------------------------------------------------------------------------------------
# training the model 
def update_training_data(new_data, new_label):
    X_train = np.load('X_train.npy')
    y_train = np.load('y_train_encoded.npy')
    X_train = np.vstack([X_train, new_data])
    y_train = np.append(y_train, new_label)
    np.save('X_train.npy', X_train)
    np.save('y_train_encoded.npy', y_train)

def train_model():
    X_train = np.load('X_train.npy')
    y_train = np.load('y_train_encoded.npy')
    global model
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    joblib.dump(model, 'emotion_model.pkl')
    print("Model retrained successfully.")

def setup_training(window):
    # Hide the main window to free up the camera and system resources
    window.withdraw()

    # Create a new top-level window for training
    training_window = tk.Toplevel()
    training_window.title("Training Interface")

    # Create a label in the training window for displaying the video
    video_label_training = tk.Label(training_window)
    video_label_training.pack()

    # Define a function to update video in the training window
    def update_video():
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label_training.imgtk = imgtk
        video_label_training.configure(image=imgtk)
        video_label_training.after(50, update_video)

    # Start updating the video
    update_video()

    # Add buttons for training data capture
    for emotion, id in emotion_dict.items():
        btn = tk.Button(training_window, text=f"Train {emotion}", command=lambda id=id: capture_training_data(id))
        btn.pack()

    # Define a function to close the training window and show the main window
    def close_training():
        training_window.destroy()
        window.deiconify()

    # Set the protocol for the window close button
    training_window.protocol("WM_DELETE_WINDOW", close_training)

def capture_training_data(emotion_id):
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) > 0:  # Check if any face is detected
        x, y, w, h = faces[0]  # Use the first detected face
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48)).flatten() / 255.0
        update_training_data(roi_gray, emotion_id)
        train_model()
        print(f"Trained on new {reverse_emotion_dict[emotion_id]} data")
    else:
        print("No face detected.")

# ----------------------------------------------------------------------------------------------
# main window

def detect_and_classify(video_label):
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48)).flatten() / 255.0
        prediction = model.predict([roi_gray])
        emotion = reverse_emotion_dict.get(prediction[0], "Unknown")  # Safely get the label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    cv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img_tk = ImageTk.PhotoImage(image=Image.fromarray(cv_img))
    video_label.imgtk = img_tk
    video_label.configure(image=img_tk)
    video_label.after(10, lambda: detect_and_classify(video_label))  # Continue detection

# ----------------------------------------------------------------------------------------------
def main():
    window = tk.Tk()
    window.title("Real-Time Facial Emotion Recognition")

    video_label = tk.Label(window)
    video_label.pack()

    detect_and_classify(video_label)

    train_button = tk.Button(window, text="Help Train This Model!", command=lambda: setup_training(window))
    train_button.pack()

    window.mainloop()

if __name__ == "__main__":
    main()
