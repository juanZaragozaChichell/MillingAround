#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2

# Base directory where the frame_* folders live
BASE_DIR = "/Users/jzaragoza/PhD/Codes/MillingAround/video_stuff/frames"

# Output video path
OUTPUT_VIDEO = "/Users/jzaragoza/PhD/Codes/MillingAround/video_stuff/milling_animation.mp4"

FPS = 15
N_FRAMES = 450  # frame_0 ... frame_449

def main():
    frames = []

    # Load frames in order
    for i in range(N_FRAMES):
        frame_name = "frame_{}".format(i)
        frame_dir = os.path.join(BASE_DIR, frame_name)
        img_path = os.path.join(frame_dir, "{}.png".format(frame_name))

        if not os.path.exists(img_path):
            print("[WARN] Missing image:", img_path)
            continue

        img = cv2.imread(img_path)
        if img is None:
            print("[WARN] Could not read image:", img_path)
            continue

        frames.append(img)

    if not frames:
        print("[ERROR] No frames loaded. Check paths.")
        return

    # Use the size of the first valid frame
    height, width, channels = frames[0].shape
    print("Video size: {}x{}".format(width, height))

    # Create video writer (MP4, H.264-friendly)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (width, height))

    # Write frames (resize if any is slightly different in size)
    for idx, img in enumerate(frames):
        if img.shape[0] != height or img.shape[1] != width:
            img = cv2.resize(img, (width, height))
            print("[INFO] Resized frame index {} to {}x{}".format(idx, width, height))
        writer.write(img)

    writer.release()
    print("Done. Video saved to:")
    print("  {}".format(OUTPUT_VIDEO))

if __name__ == "__main__":
    main()