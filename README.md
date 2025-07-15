# 🎯 Player Re-Identification in Sports Videos

A computer vision pipeline for **player detection, tracking, and re-identification** using **YOLOv8**, **DeepSORT**, and **TorchReID**. This system assigns consistent IDs to players across frames in sports footage, enabling performance analytics, automated stats, and tactical breakdowns.

---

## 📌 Overview

This project automates **player identification** across video frames using a combination of object detection (YOLOv8), object tracking (DeepSORT), and appearance-based re-identification (TorchReID). It aims to maintain **consistent player IDs** over time, even through occlusions, camera motion, and re-entry into the scene.

---

## 🎥 Demo

| Frame 1 | Frame N |
|--------|--------|
| ![Demo]() | ![Demo]() |

_(Replace above with your screenshots or a GIF of tracked_output.mp4)_

---

## 🗂️ Project Structure
- best.pt # Custom-trained YOLOv8 weights (player detection)
- main.py # Main script (run this to execute tracking)
- tracked_output.mp4 # Final output video with re-identified players
- requirements.txt # List of dependencies
- README.md # You're reading it
- deep-person-reid/ # TorchReID repo (after cloning)
