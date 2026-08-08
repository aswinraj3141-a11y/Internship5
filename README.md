# 🚗 Car Color Detection & Traffic Signal People Counter

## 📌 Overview

The Car Color Detection & Traffic Signal People Counter is a computer vision application developed using Python and OpenCV. It detects cars in images, videos, or live webcam streams, identifies whether a car is blue or another color, counts the total number of cars, and detects pedestrians in the scene.

The application includes a graphical user interface (GUI) built with Tkinter, making it easy for users to load images, videos, or use a webcam for real-time detection.

---

## ✨ Features

- Detects cars using Haar Cascade Classifier
- Classifies blue-colored cars
- Draws **red bounding boxes** around blue cars
- Draws **blue bounding boxes** around other cars
- Detects people using OpenCV HOG Person Detector
- Counts total cars detected
- Counts blue cars detected
- Counts people detected
- Supports:
  - Image files
  - Video files
  - Live webcam
- Interactive GUI built using Tkinter
- Automatic fallback detector when the car cascade file is unavailable

---

## 🛠 Technologies Used

- Python 3.x
- OpenCV
- NumPy
- Pillow (PIL)
- Tkinter

---

## 📂 Project Structure

```
Project Folder
│
├── car_color_detector.py
├── cars.xml
├── README.md
└── requirements.txt
```

---

## 📦 Requirements

Install the required libraries:

```bash
pip install opencv-python
pip install opencv-contrib-python
pip install pillow
pip install numpy
```

Or install everything at once:

```bash
pip install opencv-python opencv-contrib-python pillow numpy
```

---

## ▶️ How to Run

1. Download or clone the project.
2. Place the **cars.xml** Haar Cascade file in the project folder.
3. Install all required libraries.
4. Run:

```bash
python car_color_detector.py
```

---

## 🖥 Application Functions

### Load Image
- Select an image from your computer.
- Detects cars and people.
- Displays annotated output with counts.

### Load Video
- Select a video file.
- Processes every frame.
- Displays live detection.

### Start Webcam
- Uses your webcam for real-time detection.
- Detects cars and pedestrians continuously.

### Choose Cascade
- Allows selecting a different Haar Cascade XML file.

---

## 📊 Detection Results

The application displays:

- Total Cars Detected
- Total Blue Cars
- Total People Detected

Bounding Box Colors:

| Color | Meaning |
|--------|---------|
| 🔴 Red | Blue Car |
| 🔵 Blue | Other Car |
| 🟢 Green | Person |

---

## ⚙️ Detection Method

### Car Detection
- Haar Cascade Classifier
- Fallback contour-based detector when cascade is unavailable

### Person Detection
- OpenCV HOG Descriptor
- Pre-trained SVM Person Detector

### Color Detection
- Converts car region to HSV color space
- Filters low-saturation and shadow pixels
- Computes dominant hue
- Classifies blue vehicles based on HSV hue range

---

## 🎯 Applications

- Smart Traffic Management
- Intelligent Transportation Systems
- Vehicle Monitoring
- Parking Surveillance
- Traffic Signal Automation
- Smart City Projects

---

## 🚀 Future Enhancements

- Detect additional vehicle colors
- Vehicle tracking across frames
- Number plate recognition
- Deep Learning (YOLOv8) vehicle detection
- Vehicle speed estimation
- Traffic density analysis
- Cloud database integration

-

.
