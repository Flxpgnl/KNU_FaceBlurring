# Live Face Detection and Blurring with OpenCV and Dlib

Author: Felix Pignal

## Overview

This project demonstrates a live face detection and blurring application using OpenCV and Dlib. The program detects faces in a live video stream, identifies facial landmarks, and applies a Gaussian blur to the detected face regions to obscure them. The main goal is to provide a foundation for privacy-preserving applications like anonymizing faces in real-time video.

## Key Objectives

1. **Face Detection**: Use a pre-trained deep learning model (Caffe-based) to detect faces in a live video stream.
2. **Facial Landmark Detection**: Employ Dlib's 68-point shape predictor to identify key facial features.
3. **Blurring Faces**: Apply Gaussian blur to the detected face regions while leaving the rest of the frame unaltered.
4. **Real-Time Processing**: Ensure smooth, real-time performance for video streams.

## Results and Insights

- **Accurate Face Detection**: The application uses a confidence threshold of 0.5, balancing accuracy and false positives.
- **Dynamic Blurring**: Facial landmarks enable precise blurring of not only the face but also the forehead region for additional coverage.
- **Real-Time Performance**: Achieves real-time performance on most modern systems, suitable for live applications.

## Source Code

The source code consists of a primary Python file for real-time video blurring using a webcam, as well as configuration and model files necessary for the application.

I have opted to add my initial file that i created to get started. This file is for still images only and works in the same fashion as the video code. This code can be used for computers with less computing power.

### File Structure

```
.
├── Video.py                                  # Main script for live video processing
├── Image.py                                  # Main script for Image processing
├── deploy.prototxt                           # Caffe model configuration file
├── res10_300x300_ssd_iter_140000.caffemodel  # Pre-trained Caffe model
├── shape_predictor_68_face_landmarks.dat     # Dlib facial landmark model
```

### Key Files

- `Video.py`: The main script containing the logic for face detection and blurring using the computer webcam to take a live video feed.
- `Image.py`: The main script containing the logic for face detection and blurring using a downloaded image from the internet.
- `deploy.prototxt`: Configuration file for the Caffe face detection model.
- `res10_300x300_ssd_iter_140000.caffemodel`: Pre-trained Caffe model weights for face detection.
- `shape_predictor_68_face_landmarks.dat`: Dlib’s pre-trained model for 68-point facial landmark detection.

## Performance Metrics

- **Face Detection Accuracy**: The model has a detection accuracy of approximately 90% for 	frontal faces under normal lighting conditions, which is the best i could do with current open source generative models. 

- **Processing Speed**: ~30 FPS on modern CPUs.
- **Memory Usage**: ~200 MB RAM runtime usage.

## Installation and Usage

### Prerequisites

- Python 3.6 or later
- OpenCV (cv2) library
- Dlib library
- NumPy

### Installation

1. Clone the Repository:
    ```bash
    git clone https://github.com/Flxpgnl/KNU_FaceBlurring.git
    cd face-blurring
    ```

2. Install Dependencies:
    ```bash
    pip install opencv-python dlib numpy
    ```

3. Download Required Models:
   Ensure the following files are in the project directory:
   - `deploy.prototxt`
   - `res10_300x300_ssd_iter_140000.caffemodel`
   - `shape_predictor_68_face_landmarks.dat`

### Running the Application

1. Execute the script:
    ```bash
    python detect_and_blur_faces_live.py
    ```

2. The live video feed will appear with blurred faces.
3. Press `q` to exit the application.

## Example Usage

This project can be extended for:
- **Anonymization**: Blur faces in live streams to protect privacy.
- **Content Moderation**: Obscure sensitive faces in video feeds.
- **Real-Time Effects**: Add filters or other effects to detected face regions.

## Issues and Contributions

### Known Issues

- The model may not detect and blur hair accurately, as it is not trained for that purpose.

### Contribution Opportunities

- Train a new model capable of more accurately detecting hair along with faces.
- Please feel free to fork the repository and create a branch for contributions.
- Any feedback would be greatly appreciated

## Future Work

- Enhance performance with GPU acceleration.
- Add support for detecting and blurring multiple faces simultaneously in high-density scenes (at the moment my M2 Macbook Air can handle 4 faces).

## References and Documentation

