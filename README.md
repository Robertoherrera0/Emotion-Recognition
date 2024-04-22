# Real-Time Facial Emotion Recognition

This application uses real-time video capture to detect and recognize facial emotions, allowing users to train the model with new data through a simple graphical user interface (GUI). The system utilizes a pre-trained K-Nearest Neighbors (KNN) model and allows dynamic training to improve recognition accuracy based on user input.

## Features

- **Real-Time Emotion Detection**: Detects and displays emotions on faces in real-time using a webcam.
- **Interactive Training Interface**: Provides a training mode where users can help improve the model by capturing new training instances for various emotions.
- **Dynamic Model Retraining**: The system can retrain the underlying KNN model on-the-fly with new data captured during the training sessions.

## Technology Stack

- **Python**: Primary programming language.
- **OpenCV**: Used for all the image processing operations including face detection.
- **Tkinter**: For creating the GUI.
- **NumPy**: For handling numerical operations on arrays.
- **Joblib**: For saving and loading the trained KNN model.

## Setup and Installation
# run to install dependencies
pip install -r requirements.txt


1. **Clone the repository:**
   ```bash
   git clone https://example.com/your-repository.git

2. **Install dependencies:**
    ```bash 
    pip install -r requirements.txt

3. **run the application:**
   ```bash 
    python main.py

