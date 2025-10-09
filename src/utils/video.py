# src/utils/video.py
import numpy as np
from PIL import Image

try:
    from decord import VideoReader, cpu
    _BACKEND = "decord"
except Exception:
    import cv2
    _BACKEND = "opencv"

def sample_frame_indices(num_frames: int, total_frames: int, strategy: str = "uniform"):
    if total_frames <= num_frames:
        return np.linspace(0, max(total_frames - 1, 0), num_frames).astype(int).tolist()
    if strategy == "uniform":
        return np.linspace(0, total_frames - 1, num_frames).astype(int).tolist()
    return np.sort(np.random.choice(total_frames, num_frames, replace=False)).astype(int).tolist()

def _load_frames_decord(video_path: str, num_frames: int, strategy: str):
    vr = VideoReader(video_path, ctx=cpu(0))
    idx = sample_frame_indices(num_frames, len(vr), strategy)
    frames = vr.get_batch(idx).asnumpy()
    return [Image.fromarray(f) for f in frames]

def _load_frames_opencv(video_path: str, num_frames: int, strategy: str):
    import cv2
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or num_frames
    idx = sample_frame_indices(num_frames, total, strategy)
    frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok: continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
    cap.release()
    return frames

def load_frames_as_pil(video_path: str, num_frames: int, strategy: str = "uniform"):
    if _BACKEND == "decord":
        try:
            return _load_frames_decord(video_path, num_frames, strategy)
        except Exception:
            return _load_frames_opencv(video_path, num_frames, strategy)
    else:
        return _load_frames_opencv(video_path, num_frames, strategy)
