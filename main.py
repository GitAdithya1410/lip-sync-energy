import os
import random

import cv2
import numpy as np
import librosa
from moviepy import ImageSequenceClip, AudioFileClip
from rembg import remove  

# ------------- CONFIG -------------

ASSETS_DIR = "assets"
BACKGROUND_PATH = os.path.join(ASSETS_DIR, "background.png")   # or .jpg
CHARACTER_PATH = os.path.join(ASSETS_DIR, "character_old.png")
MOUTH_DIR = os.path.join(ASSETS_DIR, "mouth_shapes")
AUDIO_PATH = "audio3.wav"                                       #  audio 

FPS = 30
MOUTH_POSITION = (1065, 1050)      # top-left of mouth on character (x, y)
MOUTH_TARGET_WIDTH = 340           # resize all mouths to this width in pixels

OUTPUT_VIDEO_NO_AUDIO = "output_no_audio.mp4"
FINAL_OUTPUT_VIDEO = "final_output.mp4"

# phoneme to file mapping 
PHONEME_TO_FILE = {
    "A": "A.png",
    "E": "E.png",
    "O": "O.png",
    "U": "U.png",
    "M": "M.png",
    "F_V": "F_V.png",
    "T_H": "T_H.png",
    "L": "L.png",
    "W_Q": "W_Q.png",
    "C_D_G_K_N_R_S_T": "C_D_G_K_N_R_S_T.png",
}


# ------------- HELPERS -------------

def overlay_alpha(background, overlay, pos):
    """
    Overlay RGBA or RGB 'overlay' onto BGR 'background' at position pos=(x, y).
    If no alpha channel, it just pastes the overlay block.
    """
    x, y = pos
    bh, bw = background.shape[:2]
    oh, ow = overlay.shape[:2]

    # bounds check
    if x < 0 or y < 0 or x + ow > bw or y + oh > bh:
        return background

    # RGB only
    if overlay.shape[2] == 3:
        background[y:y + oh, x:x + ow] = overlay
        return background

    # RGBA blend
    alpha_overlay = overlay[:, :, 3] / 255.0
    alpha_background = 1.0 - alpha_overlay

    for c in range(0, 3):
        background[y:y + oh, x:x + ow, c] = (
            alpha_overlay * overlay[:, :, c] +
            alpha_background * background[y:y + oh, x + 0: x + ow, c]
        )

    return background


def load_image(path, allow_fail=False):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None and not allow_fail:
        raise FileNotFoundError(f"Could not load image at {path}")
    return img

def make_character_rgba_from_bgcolor(char_bgr, tolerance=15):
    """
    Take a 3-channel character image on flat background
    and convert it to 4-channel by making the background transparent.
    Background color is taken from the top-left pixel.
    """
    if char_bgr.shape[2] != 3:
        return char_bgr  # already RGBA or bad shape, just return

    h, w, _ = char_bgr.shape
    # get background color from top-left
    bg_color = char_bgr[0, 0].astype(np.int16)

    # compute distance from background color
    diff = char_bgr.astype(np.int16) - bg_color[None, None, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    # alpha: 0 for near background, 255 otherwise
    alpha = np.where(dist <= tolerance, 0, 255).astype(np.uint8)

    char_rgba = np.dstack([char_bgr, alpha])
    return char_rgba

def load_mouth_images():
    mouths = {}
    for ph, filename in PHONEME_TO_FILE.items():
        path = os.path.join(MOUTH_DIR, filename)
        img = load_image(path, allow_fail=True)
        if img is None:
            print(f"[WARN] Missing mouth image for {ph}: {path}")
            continue

        # resize to consistent width
        scale = MOUTH_TARGET_WIDTH / img.shape[1]
        resized = cv2.resize(
            img,
            (int(img.shape[1] * scale), int(img.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
        mouths[ph] = resized
    return mouths


def get_audio_frame_count(audio_path, fps):
    if not os.path.exists(audio_path):
        print("[WARN] audio.wav not found, generating 3 seconds of fake animation.")
        duration = 3.0
    else:
        y, sr = librosa.load(audio_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
    return int(duration * fps)

def build_mouth_sequence_from_energy(audio_path, fps):
    """
    Uses audio loudness per frame to drive mouth shapes.
    No transcript, no phonemes, just energy bands.
    """
    if not os.path.exists(audio_path):
        print("[WARN] Audio file not found, using 3 seconds dummy animation.")
        duration = 3.0
        frame_count = int(duration * fps)
        return ["M"] * frame_count, frame_count

    # Load audio
    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    frame_count = int(duration * fps)

    # Compute energy per frame
    frame_len = int(sr / fps)  # samples per video frame
    energies = []
    for i in range(frame_count):
        start = i * frame_len
        end = start + frame_len
        if start >= len(y):
            energies.append(0.0)
            continue
        chunk = y[start:end]
        # simple energy: mean absolute value
        e = float(np.mean(np.abs(chunk)))
        energies.append(e)

    energies = np.array(energies)

    # Avoid degenerate case
    if np.all(energies == 0):
        return ["M"] * frame_count, frame_count

    # Compute thresholds (quartiles)
    t1 = np.quantile(energies, 0.25)
    t2 = np.quantile(energies, 0.50)
    t3 = np.quantile(energies, 0.75)

    mouth_sequence = []

    for e in energies:
        if e < t1:
            shape = "M"         # closed / almost silent
        elif e < t2:
            shape = "E"         # small open, side stretch
        elif e < t3:
            shape = "A"         # mid open
        else:
            # big open / rounded sounds
            if "O" in PHONEME_TO_FILE:
                shape = "O"
            elif "W_Q" in PHONEME_TO_FILE:
                shape = "W_Q"
            else:
                shape = "A"

        mouth_sequence.append(shape)

    return mouth_sequence, frame_count

# ------------- MAIN PIPELINE -------------

def main():
    print("[INFO] Loading assets...")
    background = load_image(BACKGROUND_PATH, allow_fail=True)
    # Load character and immediately run AI background removal
    character_raw = load_image(CHARACTER_PATH)
    character = remove(character_raw)

    print("Character shape:", character.shape if character is not None else None)
    print("Background shape:", background.shape if background is not None else None)

    if background is None:
        # if you have no background, use character size and plain white bg
        h, w = character.shape[:2]
        background = np.ones((h, w, 3), dtype=np.uint8) * 255
    else:
        # ensure background is BGR 3-channel
        if background.shape[2] == 4:
            # drop alpha for background, we will treat it as solid
            background = background[:, :, :3]
    # ---- resize BACKGROUND to match CHARACTER size (safe for mouth coords) ----
    ch, cw = character.shape[:2]
    bh, bw = background.shape[:2]

    if (bh, bw) != (ch, cw):
        background = cv2.resize(background, (cw, ch), interpolation=cv2.INTER_AREA)


    # preload mouths
    mouths = load_mouth_images()
    if not mouths:
        print("[ERROR] No mouth images loaded. Check assets/mouth_shapes.")
        return

    # list of phonemes you actually have images for
    available_phonemes = list(mouths.keys())
    print("[INFO] Available phonemes:", available_phonemes)

    # how many frames based on audio length
    print("[INFO] Building mouth sequence from audio energy...")
    phoneme_sequence, frame_count = build_mouth_sequence_from_energy(
        AUDIO_PATH, FPS
    )
    print(f"[INFO] Will render {frame_count} frames at {FPS} FPS.")
    print("[INFO] First 20 shapes:", phoneme_sequence[:20])

    frames = []
    x, y = MOUTH_POSITION

    print("[INFO] Generating frames...")
    for idx, ph in enumerate(phoneme_sequence):
        # base frame: background plus character
        frame = background.copy()

        # make sure character is 3 channel BGR with optional alpha handled
        if character.shape[2] == 4:
            # separate char alpha and blend onto frame
            char = character.copy()
            alpha = char[:, :, 3] / 255.0
            alpha_bg = 1.0 - alpha
            for c in range(3):
                frame[:, :, c] = (
                    alpha * char[:, :, c] +
                    alpha_bg * frame[:, :, c]
                )
        else:
            frame[0:character.shape[0], 0:character.shape[1]] = character

        # apply mouth
        mouth_img = mouths.get(ph)
        if mouth_img is not None:
            frame = overlay_alpha(frame, mouth_img, (x, y))

        # convert to RGB for MoviePy
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

        if (idx + 1) % 50 == 0:
            print(f"[INFO] Generated {idx + 1}/{frame_count} frames")

    print("[INFO] Writing video without audio...")
    clip = ImageSequenceClip(frames, fps=FPS)
    clip.write_videofile(OUTPUT_VIDEO_NO_AUDIO, codec="libx264", audio=False)

    if os.path.exists(AUDIO_PATH):
        print("[INFO] Adding audio...")
        audio = AudioFileClip(AUDIO_PATH)

        # Attach audio directly to the clip
        clip.audio = audio

        clip.write_videofile(FINAL_OUTPUT_VIDEO, codec="libx264", audio_codec="aac")
        print(f"[DONE] Saved {FINAL_OUTPUT_VIDEO}")
    else:
        print(f"[DONE] Saved {OUTPUT_VIDEO_NO_AUDIO} (no audio file found)")


if __name__ == "__main__":
    main()

