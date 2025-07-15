# ⚽ Player Tracking and Re-Identification

This project focuses on tracking football players from broadcast videos using object detection, DeepSORT tracking, and appearance-based Re-ID (Re-Identification). It enables consistent identification of players across frames, even with occlusions or temporary exits from the scene.

---

## 📌 Features

- 🎯 Custom-trained YOLOv8 model for detecting players
- 🎥 Multi-object tracking with DeepSORT
- 🔍 Appearance-based Re-Identification using OSNet
- 📦 Efficient cosine similarity matching with memory dictionary
- 🧠 Embedding normalization and ID consistency
- 📤 Export of final annotated video with consistent player tags

---

## 🧠 Architecture and Model Choices

- **YOLOv8 (Ultralytics) for Player Detection**
  - Used for high-accuracy player, ball, and referee detection.
  - Trained using Roboflow for domain-specific performance.
  - Real-time capable with balanced speed and precision.

- **DeepSORT for Object Tracking**
  - Tracks multiple objects (players) across frames.
  - Combines Kalman filtering and IoU matching.
  - Handles occlusions and missed detections robustly.

- **TorchReID (OSNet_x1_0) for Appearance Features**
  - Extracts 2048-D embeddings for cropped player images.
  - Helps maintain identity over long sequences.
  - OSNet excels in person Re-ID tasks and works well for sports footage.

- **Cosine Similarity for Re-ID**
  - Compares current player embeddings with stored embeddings.
  - Ensures continuity in identity across frames using threshold-based ID assignment.
  - Normalized embeddings improve similarity accuracy.

- **Memory Management**
  - Dictionary-based ID memory for embeddings.
  - Ensures fast lookup and embedding updates for each player.

---

## 📁 Project Structure

```bash
.
├── best.pt                      # Custom-trained YOLOv8 weights
├── tracked_output.mp4          # Final output video
├── broadcast.mp4               # Input football match clip
├── reid_tracking.py            # Main processing script
├── README.md                   # This file
```

---

## ✅ Prerequisites

Before running the project, ensure you have the following:

### 🔧 System Requirements
- **Python** ≥ 3.8
- **pip** ≥ 21.0
- **CUDA-compatible GPU** (for accelerated inference with PyTorch & YOLOv8)
- **RAM**: Minimum 8GB (16GB+ recommended for smoother processing)

### 📦 Required Python Libraries

- Install the essential libraries via pip:
### Libraries Used:
- torch – Core deep learning framework

- torchvision – For image transformations and pretrained CNNs

- opencv-python-headless – Video I/O and image processing

- ultralytics – YOLOv11 object detection models

- deep_sort_realtime – Multi-object tracking with DeepSORT

- scipy – Cosine distance computation

- torchreid – Deep Re-ID features with OSNet architecture

### 🔁 Optional Tools
-  makesense.ai – To annotate ground truth data for evaluation
---

## ⚙️ Installation

### Step 1: Clone and Install TorchReID
```bash
git clone https://github.com/KaiyangZhou/deep-person-reid.git
cd deep-person-reid
pip install -r requirements.txt
python setup.py install
cd ..
```
### Step 2: Install other dependencies
```bash
pip install ultralytics deep_sort_realtime opencv-python-headless torch torchvision scipy
```

---

## How To Run

### Prepare video
- Place your input video in the root directory and update the cap_path in main.py.

### Place YOLO weights
- Place best.pt in the root directory.

---

## Methodology

### Ultrlytics Yolov11
- Used for detecting players in each frame. Trained on custom sports dataset using Roboflow.

### DeepSORT
- Handles multi-object tracking using motion + appearance cues.

### TorchReID (osnet X1)
- Extracts 2048-D feature vectors from each player's crop. Used for comparing and re-identifying players using cosine similarity.

### Cosine Similarity + ID Memory
- We store embeddings of tracked IDs. If cosine similarity between a new player and a known one is below 0.4, we assign the old ID (re-identification).

---

## 



---

## Limitations
- Player confusion if players look similar (same jersey, posture, etc.)

- Struggles with occlusion and long re-entries.

- YOLO class ID must be correct and robust.

- Video resolution directly affects ReID accuracy.


---

## 🚀 Future Improvements
- Use YOLOv8-seg for better mask-based tracking.

- Incorporate jersey number recognition via OCR.

- Fine-tune TorchReID on player crops for better embeddings.

- Integrate temporal consistency filters (e.g., Kalman + ReID hybrid revalidation).


## 🧑‍💻 Contributors
- Meet Bahl – ML Engineer, Developer

  
