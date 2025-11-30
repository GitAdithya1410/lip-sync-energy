# lip-sync-energy
A Python application that performs automatic cartoon lip-syncing using audio analysis, dynamic mouth-shape switching, and alpha-blended character rendering.
This system takes:
	• A cartoon character
	• A set of phoneme-based mouth PNGs
	• Any speech audio
	• An optional background image
	
This outputs a lip-synced animation video.

Built with OpenCV, Librosa, MoviePy, and ONNX-based background removal.


Features
- Energy-based lip movement engine (no transcript needed)
- Auto background removal using ONNX Runtime
- Supports transparent PNG characters
- Smooth alpha blending for mouth overlays
- Full video rendering with audio
- Customizable phoneme sets (A, E, O, U, M, L, TH, W/Q, Neutral)

Project Structure
lip-sync/
│
├── assets/
│   ├── mouth_shapes/
│   ├── character.png
│   ├── background.png
│   
│
├── main.py
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md

How It Works
1. Loads transparent character
2. Removes background (if needed)
3. Reads audio → splits into tiny frames
4. Calculates energy → assigns appropriate mouth shape
5. Alpha-overlays mouth on character
6. Combines frames into a video
7. Adds audio back to final output

Installation
1. Clone the repo
2. Create venv
3. Install requirements

#Usage
python main.py

#requirements.txt
opencv-python
numpy
moviepy
librosa
onnxruntime
rembg
soundfile

#.gitignore
venv/
__pycache__/
*.mp4
*.wav
*.m4a
*.mov
*.onnx
.DS_Store
Thumbs.db

## This project is released under the CC BY-NC License: you may use and modify the code for personal and non-commercial purposes with proper attribution.
