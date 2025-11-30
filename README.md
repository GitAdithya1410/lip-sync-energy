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

	Energy-based lip movement engine (no transcript needed)
	
	Auto background removal using ONNX Runtime
	
	Supports transparent PNG characters
	
	Smooth alpha blending for mouth overlays
	
	Full video rendering with audio
	
	Customizable phoneme sets (A, E, O, U, M, L, TH, W/Q, Neutral)
