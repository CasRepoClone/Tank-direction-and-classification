from ultralytics import YOLO
import cv2
import numpy as np
import os
from glob import glob
import mss
import pygetwindow as gw
import tkinter as tk
from PIL import Image, ImageTk

# Overlay size (change as needed)
OVERLAY_WIDTH = 400 * 2  # since you concatenate two images
OVERLAY_HEIGHT = 400

# Load your trained OBB model
model = YOLO('runs/obb/train4/weights/best.pt')  # use your trained weights

def get_histogram_from_img(img):
    img = cv2.resize(img, (128, 128))
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# Precompute histograms for all relative images
relative_dir = 'relativeImages'
relative_paths = glob(os.path.join(relative_dir, '*.png'))
relative_hists = []
for path in relative_paths:
    img = cv2.imread(path)
    if img is not None:
        relative_hists.append((path, get_histogram_from_img(img)))

# Set capture ratios (easy to modify)
CAPTURE_WIDTH_RATIO = 0.7
CAPTURE_HEIGHT_RATIO = 0.7

with mss.mss() as sct:
    monitor = sct.monitors[1]  # 1 = primary monitor
    screen_width = monitor["width"]
    screen_height = monitor["height"]

    # Calculate perfectly centered inner region of the screen
    inner_left = int((screen_width - screen_width * CAPTURE_WIDTH_RATIO) / 2)
    inner_top = int((screen_height - screen_height * CAPTURE_HEIGHT_RATIO) / 2)
    inner_width = int(screen_width * CAPTURE_WIDTH_RATIO)
    inner_height = int(screen_height * CAPTURE_HEIGHT_RATIO)

    region = {
        "top": inner_top,
        "left": inner_left,
        "width": inner_width,
        "height": inner_height
    }

    # --- Tkinter overlay setup ---
    root = tk.Tk()
    root.title("Overlay")
    root.geometry(f"{OVERLAY_WIDTH}x{OVERLAY_HEIGHT}+0+0")
    root.attributes("-topmost", True)
    root.overrideredirect(True)
    panel = tk.Label(root)
    panel.pack(fill="both", expand=True)

    def show_overlay(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Example: crop 10% from each border (customize as needed)
        h, w = img.shape[:2]
        crop_x = int(w * 0.1) # percentage of width to crop
        crop_y = int(h * 0.1) # percentage of height to crop
        cropped = img[crop_y:h-crop_y, crop_x:w-crop_x]
        img = cv2.resize(cropped, (OVERLAY_WIDTH, OVERLAY_HEIGHT))
        im_pil = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=im_pil)
        panel.imgtk = imgtk
        panel.config(image=imgtk)
        root.update_idletasks()
        root.update()

    while True:
        sct_img = sct.grab(region)
        frame = np.array(sct_img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Run YOLO detection on the frame
        results = model.predict(source=frame, save=False, conf=0.25)
        img = results[0].plot()
        h, w = img.shape[:2]
        new_w = 900
        new_h = int(h * (new_w / w))
        img_resized = cv2.resize(img, (new_w, new_h))

        # Find closest relative image
        result_hist = get_histogram_from_img(img)
        distances = [cv2.compareHist(result_hist, rhist, cv2.HISTCMP_BHATTACHARYYA) for _, rhist in relative_hists]
        if distances:
            closest_idx = np.argmin(distances)
            closest_path = relative_paths[closest_idx]
            closest_img = cv2.imread(closest_path)
            closest_img = cv2.resize(closest_img, (new_w, new_h))
            closest_name = os.path.basename(closest_path)
        else:
            closest_img = np.zeros_like(img_resized)
            closest_name = "N/A"

        # Only show the closest relevant image in the overlay
        display_img = closest_img.copy()
        cv2.putText(display_img, f"Closest: {closest_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(display_img, "Press ESC to exit", (10, display_img.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        show_overlay(display_img)

        if cv2.waitKey(1) == 27:
            break

    root.destroy()