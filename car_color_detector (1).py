"""
Car Color Detection & Traffic Signal People Counter
=====================================================

Features
--------
1. GUI (Tkinter) with a preview of the input image/video frame and the
   processed output side by side.
2. Detects cars in the frame (Haar cascade) and classifies each car's
   dominant color.
   - BLUE cars  -> drawn with a RED rectangle
   - Other cars -> drawn with a BLUE rectangle
3. Detects people in the frame (HOG + built-in SVM person detector, no
   external file needed) and shows the count.
4. Shows total car count and total people count.
5. Works with:
   - A single image file
   - A video file
   - The live webcam

Setup
-----
pip install opencv-python opencv-contrib-python pillow numpy

You also need a car Haar cascade XML file named `cars.xml` placed in the
same folder as this script (or choose a different path from the GUI).
A commonly used one ("cars.xml") is freely available online, e.g. from
public OpenCV-cascade repositories on GitHub (search "haarcascade cars.xml
opencv"). Download it and place it next to this script.

If no cascade file is found, the app will still run and will fall back to
a simple color/contour based "blob" detector so you can still test the
color-classification and people-counting parts of the pipeline.

Run
---
python car_color_detector.py
"""

import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk


# ----------------------------------------------------------------------
# Detection / classification helpers
# ----------------------------------------------------------------------

class CarColorDetector:
    """Wraps car detection, person detection and color classification."""

    def __init__(self, cascade_path="cars.xml"):
        self.cascade_path = cascade_path
        self.car_cascade = None
        if os.path.isfile(cascade_path):
            self.car_cascade = cv2.CascadeClassifier(cascade_path)
            if self.car_cascade.empty():
                self.car_cascade = None

        # Built-in pedestrian detector - no external file required.
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # ---------------- Car detection ----------------

    def detect_cars(self, frame_bgr):
        """Return a list of (x, y, w, h) bounding boxes for cars."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if self.car_cascade is not None:
            boxes = self.car_cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=5, minSize=(40, 40)
            )
            return [tuple(map(int, b)) for b in boxes]

        # ---- Fallback: simple blob/contour based detector ----
        # Useful only as a placeholder when cars.xml isn't available,
        # so the rest of the pipeline (color classification, counting,
        # GUI) can still be demonstrated end to end.
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 60, 160)
        edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        boxes = []
        h_img, w_img = gray.shape[:2]
        min_area = (w_img * h_img) * 0.01  # ignore tiny specks
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            aspect = w / float(h) if h > 0 else 0
            if area > min_area and 0.6 < aspect < 3.5:
                boxes.append((x, y, w, h))
        return boxes

    # ---------------- Person detection ----------------

    def detect_people(self, frame_bgr):
        """Return a list of (x, y, w, h) bounding boxes for people."""
        # Downscale for speed if the frame is large.
        h_img, w_img = frame_bgr.shape[:2]
        scale = 640.0 / w_img if w_img > 640 else 1.0
        small = cv2.resize(frame_bgr, (int(w_img * scale), int(h_img * scale)))

        rects, _ = self.hog.detectMultiScale(
            small, winStride=(8, 8), padding=(8, 8), scale=1.05
        )

        boxes = []
        for (x, y, w, h) in rects:
            boxes.append(
                (int(x / scale), int(y / scale), int(w / scale), int(h / scale))
            )
        return boxes

    # ---------------- Color classification ----------------

    @staticmethod
    def is_blue_car(frame_bgr, box):
        """Classify the dominant color inside a car's bounding box.

        Returns True if the car is judged to be BLUE, False otherwise.
        """
        x, y, w, h = box
        x, y = max(x, 0), max(y, 0)
        roi = frame_bgr[y:y + h, x:x + w]
        if roi.size == 0:
            return False

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Ignore near-black / near-white / low-saturation (shadow, glass,
        # road reflections) pixels so the color estimate reflects the
        # car's paint rather than background noise.
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]
        mask = (s_channel > 40) & (v_channel > 40) & (v_channel < 250)

        if np.count_nonzero(mask) < 20:
            # Not enough colorful pixels to judge -> assume non-blue.
            return False

        hue_values = hsv[:, :, 0][mask]
        mean_hue = float(np.mean(hue_values))

        # In OpenCV's HSV, Hue range is 0-179.
        # Blue paint typically falls roughly between 95 and 135.
        return 95 <= mean_hue <= 135

    # ---------------- Full pipeline on one frame ----------------

    def process_frame(self, frame_bgr):
        """Run detection + classification + draw annotations.

        Returns (annotated_frame, car_count, blue_car_count, people_count)
        """
        annotated = frame_bgr.copy()

        car_boxes = self.detect_cars(frame_bgr)
        people_boxes = self.detect_people(frame_bgr)

        blue_count = 0
        for box in car_boxes:
            x, y, w, h = box
            blue = self.is_blue_car(frame_bgr, box)
            if blue:
                blue_count += 1
                color = (0, 0, 255)   # RED rectangle (BGR) for blue cars
                label = "Blue car"
            else:
                color = (255, 0, 0)   # BLUE rectangle (BGR) for other cars
                label = "Car"
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                annotated, label, (x, max(y - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

        for (x, y, w, h) in people_boxes:
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                annotated, "Person", (x, max(y - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

        # Summary banner
        summary = (
            f"Cars: {len(car_boxes)}  |  Blue cars: {blue_count}  |  "
            f"People: {len(people_boxes)}"
        )
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(
            annotated, summary, (8, 21),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )

        return annotated, len(car_boxes), blue_count, len(people_boxes)


# ----------------------------------------------------------------------
# GUI
# ----------------------------------------------------------------------

class App(tk.Tk):
    PREVIEW_W, PREVIEW_H = 480, 360

    def __init__(self):
        super().__init__()
        self.title("Car Color Detection & Traffic Signal Counter")
        self.geometry("1040x620")
        self.resizable(False, False)

        self.detector = CarColorDetector(cascade_path="cars.xml")

        self.video_source = None       # cv2.VideoCapture object (video/webcam)
        self.current_frame_bgr = None  # for single-image mode
        self.running_stream = False    # True while video/webcam loop is active

        self._build_widgets()
        self._refresh_cascade_status()

    # ---------------- UI construction ----------------

    def _build_widgets(self):
        top_bar = ttk.Frame(self)
        top_bar.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        ttk.Button(top_bar, text="Load Image", command=self.load_image).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(top_bar, text="Load Video", command=self.load_video).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(top_bar, text="Start Webcam", command=self.start_webcam).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(top_bar, text="Stop", command=self.stop_stream).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(
            top_bar, text="Choose Cascade (cars.xml)", command=self.choose_cascade
        ).pack(side=tk.LEFT, padx=4)

        self.cascade_status_var = tk.StringVar(value="")
        ttk.Label(top_bar, textvariable=self.cascade_status_var).pack(
            side=tk.LEFT, padx=10
        )

        # --- Preview panels ---
        panels = ttk.Frame(self)
        panels.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10)

        left = ttk.LabelFrame(panels, text="Input Preview")
        left.grid(row=0, column=0, padx=8, pady=8)
        self.input_canvas = tk.Canvas(
            left, width=self.PREVIEW_W, height=self.PREVIEW_H, bg="#222"
        )
        self.input_canvas.pack()

        right = ttk.LabelFrame(panels, text="Detection Output")
        right.grid(row=0, column=1, padx=8, pady=8)
        self.output_canvas = tk.Canvas(
            right, width=self.PREVIEW_W, height=self.PREVIEW_H, bg="#222"
        )
        self.output_canvas.pack()

        # --- Stats bar ---
        stats = ttk.LabelFrame(self, text="Results")
        stats.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        self.cars_var = tk.StringVar(value="Cars detected: -")
        self.blue_var = tk.StringVar(value="Blue cars: -")
        self.people_var = tk.StringVar(value="People detected: -")

        ttk.Label(stats, textvariable=self.cars_var, font=("Segoe UI", 12)).pack(
            side=tk.LEFT, padx=20, pady=8
        )
        ttk.Label(stats, textvariable=self.blue_var, font=("Segoe UI", 12)).pack(
            side=tk.LEFT, padx=20, pady=8
        )
        ttk.Label(stats, textvariable=self.people_var, font=("Segoe UI", 12)).pack(
            side=tk.LEFT, padx=20, pady=8
        )

        # Legend
        legend = ttk.Frame(self)
        legend.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(0, 8))
        ttk.Label(legend, text="Legend:  ").pack(side=tk.LEFT)
        ttk.Label(legend, text="Red box = Blue car   ", foreground="red").pack(
            side=tk.LEFT
        )
        ttk.Label(legend, text="Blue box = Other car   ", foreground="blue").pack(
            side=tk.LEFT
        )
        ttk.Label(legend, text="Green box = Person", foreground="green").pack(
            side=tk.LEFT
        )

    def _refresh_cascade_status(self):
        if self.detector.car_cascade is not None:
            self.cascade_status_var.set(f"Cascade loaded: {self.detector.cascade_path}")
        else:
            self.cascade_status_var.set(
                "No cars.xml found - using fallback contour-based car detector"
            )

    # ---------------- Actions ----------------

    def choose_cascade(self):
        path = filedialog.askopenfilename(
            title="Select Haar cascade XML for cars",
            filetypes=[("XML files", "*.xml")],
        )
        if path:
            self.detector = CarColorDetector(cascade_path=path)
            self._refresh_cascade_status()

    def load_image(self):
        self.stop_stream()
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")],
        )
        if not path:
            return
        frame = cv2.imread(path)
        if frame is None:
            messagebox.showerror("Error", "Could not read the selected image.")
            return
        self.current_frame_bgr = frame
        self._show_on_canvas(self.input_canvas, frame)
        self._process_and_show(frame)

    def load_video(self):
        self.stop_stream()
        path = filedialog.askopenfilename(
            title="Select a video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")],
        )
        if not path:
            return
        self.video_source = cv2.VideoCapture(path)
        if not self.video_source.isOpened():
            messagebox.showerror("Error", "Could not open the selected video.")
            return
        self.running_stream = True
        threading.Thread(target=self._stream_loop, daemon=True).start()

    def start_webcam(self):
        self.stop_stream()
        self.video_source = cv2.VideoCapture(0)
        if not self.video_source.isOpened():
            messagebox.showerror("Error", "Could not access the webcam.")
            return
        self.running_stream = True
        threading.Thread(target=self._stream_loop, daemon=True).start()

    def stop_stream(self):
        self.running_stream = False
        if self.video_source is not None:
            self.video_source.release()
            self.video_source = None

    # ---------------- Core processing ----------------

    def _stream_loop(self):
        while self.running_stream and self.video_source is not None:
            ok, frame = self.video_source.read()
            if not ok:
                break
            self._show_on_canvas(self.input_canvas, frame)
            self._process_and_show(frame)
        self.running_stream = False

    def _process_and_show(self, frame_bgr):
        annotated, n_cars, n_blue, n_people = self.detector.process_frame(frame_bgr)
        self._show_on_canvas(self.output_canvas, annotated)

        # Update labels (thread-safe enough for this simple demo via after())
        self.after(0, lambda: self.cars_var.set(f"Cars detected: {n_cars}"))
        self.after(0, lambda: self.blue_var.set(f"Blue cars: {n_blue}"))
        self.after(0, lambda: self.people_var.set(f"People detected: {n_people}"))

    def _show_on_canvas(self, canvas, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img.thumbnail((self.PREVIEW_W, self.PREVIEW_H))
        photo = ImageTk.PhotoImage(img)
        canvas.delete("all")
        canvas.create_image(
            self.PREVIEW_W // 2, self.PREVIEW_H // 2, image=photo
        )
        # keep a reference so it isn't garbage collected
        canvas.image = photo

    def on_close(self):
        self.stop_stream()
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
