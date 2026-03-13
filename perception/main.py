from ultralytics import YOLO
import cv2
import math
import os
import time
import numpy as np
import threading
import queue
import subprocess
from collections import deque

def _sapi_speak(text: str) -> None:
    # Uses Windows built-in System.Speech (offline) via PowerShell.
    # Escape single quotes for PowerShell single-quoted string.
    safe = (text or "").replace("'", "''")
    cmd = [
        "powershell",
        "-NoProfile",
        "-Command",
        "Add-Type -AssemblyName System.Speech; "
        "$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$speak.Speak('{safe}');",
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

# ==============================
# PATHS / CAMERA
# ==============================

MODEL_PATH = "yolov8x.pt"
CAMERA_INDEX = 0  # default webcam

# ==============================
# RESOLUTION / PERFORMANCE
# ==============================

RESIZE_WIDTH = 512  # resize width for YOLO input (lower = faster)
SHOW_FPS_OVERLAY = True
PROCESS_EVERY_N_FRAMES = 1  # process every frame
TERMINAL_SUMMARY_INTERVAL = 7.0  # seconds between short terminal summaries

# Distance display thresholds (meters)
CLOSE_DISTANCE_MAX = 4.0
FAR_DISPLAY_MAX = 12.0  # ignore objects farther than this to avoid clutter

# GPU / speed settings
# Prefer GPU if available, gracefully fall back to CPU.
USE_GPU = True
USE_HALF_PRECISION = True  # used only if GPU is successfully enabled
YOLO_CONF = 0.4

# Camera motion estimation (optical flow) is expensive.
# Disable for much higher FPS, or keep enabled with downscale.
ENABLE_CAMERA_MOTION = False
CAM_MOTION_DOWNSCALE = 0.5

# ==============================
# MOTION CONFIG
# ==============================

HISTORY_FRAMES = 15
GROWTH_THRESHOLD = 0.15
SHRINK_THRESHOLD = -0.15
PASSING_THRESHOLD = 0.06

# ==============================
# TEMPORAL FILTER
# ==============================

MOTION_HISTORY = 8
CONFIRM_THRESHOLD = 4

# ==============================
# MOTION IDS
# ==============================

MOTION_MAP = {
    "stationary": 0,
    "approaching": 1,
    "moving away": 2,
    "passing": 3,
    "tracking": -1
}

REVERSE_MOTION_MAP = {v: k for k, v in MOTION_MAP.items()}

# ==============================
# RISK CONFIG
# ==============================

HIGH_RISK_THRESHOLD = 11

# ==============================
# HELPER FUNCTIONS
# ==============================

class _TTSWorker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        # Latest-only queue to avoid speech backlog/lag.
        self._q = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self.last_error = {"message": "", "ts": 0.0}
        self._enabled = True

    def submit(self, text):
        if not self._enabled:
            return False
        try:
            self._q.put_nowait(text)
            return True
        except queue.Full:
            try:
                _ = self._q.get_nowait()
                self._q.task_done()
            except Exception:
                pass
            try:
                self._q.put_nowait(text)
                return True
            except Exception:
                return False

    def get_last_error(self):
        with self._lock:
            return dict(self.last_error)

    def run(self):
        if not self._enabled:
            return
        while True:
            try:
                text = self._q.get()
                if text:
                    _sapi_speak(text)
            except Exception as e:
                with self._lock:
                    self.last_error = {"message": str(e), "ts": time.time()}
            finally:
                self._q.task_done()


_tts_worker = _TTSWorker()
_tts_worker.start()
print("TTS:", "ENABLED")


def estimate_distance(area, frame_width):
    return max(1, (frame_width * 0.7) / (math.sqrt(area) + 1e-6))


def calculate_risk_score(label, motion, distance):

    weights = {
        "car": 5,
        "bus": 6,
        "truck": 6,
        "person": 3,
        "bicycle": 4,
        "dog": 3
    }

    base = weights.get(label, 2)

    proximity = 5 if distance < 4 else 2 if distance < 8 else 0
    motion_val = 4 if motion == "approaching" else 0

    return base + proximity + motion_val


def get_risk_category(score):

    if score >= HIGH_RISK_THRESHOLD:
        return "high"
    elif score >= 6:
        return "medium"
    else:
        return "low"


def get_direction(center_x, frame_width):

    if center_x < frame_width / 3:
        return "left"
    elif center_x > (2 * frame_width) / 3:
        return "right"
    else:
        return "center"


def _is_dangerous_and_close(d):
    if not d:
        return False
    dist = float(d.get("distance", 999.0))
    risk = d.get("risk", "low")
    motion = d.get("motion", "stationary")
    if dist > CLOSE_DISTANCE_MAX:
        return False
    if risk == "high":
        return True
    if risk == "medium" and motion in ("approaching", "passing"):
        return True
    return False


def build_natural_summary(detections):
    """
    7-second surroundings summary (natural language).
    Ignore far, stationary, and low-risk objects; prioritize higher risk first.
    """
    if not detections:
        return "", ()

    def matters(d):
        dist = float(d.get("distance", 999.0))
        risk = d.get("risk", "low")
        motion = d.get("motion", "stationary")
        if dist > 8.0:
            return False
        if risk == "low":
            return False
        if motion == "stationary":
            return False
        return True

    relevant = [d for d in detections if matters(d)]
    if not relevant:
        return "Mostly clear nearby.", ("clear",)

    risk_rank = {"high": 0, "medium": 1, "low": 2}
    motion_rank = {"approaching": 0, "passing": 1, "moving away": 2, "stationary": 3, "tracking": 4}
    relevant.sort(
        key=lambda d: (
            risk_rank.get(d.get("risk", "low"), 2),
            motion_rank.get(d.get("motion", "tracking"), 4),
            float(d.get("distance", 999.0)),
        )
    )

    phrases = []
    key_items = []
    for d in relevant[:3]:
        obj = d.get("object", "")
        direction = d.get("direction", "center")
        dist_m = int(round(float(d.get("distance", 999.0))))
        motion = d.get("motion", "tracking")
        risk = d.get("risk", "low")

        dir_phrase = "ahead" if direction == "center" else direction
        motion_phrase = " approaching" if motion == "approaching" else " passing" if motion == "passing" else ""
        meter_word = "meter" if dist_m == 1 else "meters"
        phrases.append(f"{obj}{motion_phrase} {dir_phrase} ({dist_m} {meter_word})")
        key_items.append((obj, direction, dist_m, motion, risk))

    text = "Around you: " + "; ".join(phrases) + "."
    return text, tuple(key_items)


# ==============================
# LOAD MODEL
# ==============================

model = YOLO(MODEL_PATH)
try:
    model.fuse()  # speed-up if supported
except Exception:
    pass

# Decide device once (GPU if possible, else CPU).
_DEVICE_ARG = "cpu"
_HALF_PRECISION_FLAG = False
if USE_GPU:
    try:
        # Attempt to move model to CUDA/GPU; if it fails, we stay on CPU.
        model.to("cuda")
        _DEVICE_ARG = 0  # first GPU
        _HALF_PRECISION_FLAG = USE_HALF_PRECISION
        print("Using GPU (CUDA) for inference.")
    except Exception:
        _DEVICE_ARG = "cpu"
        _HALF_PRECISION_FLAG = False
        print("GPU not available, falling back to CPU.")
else:
    print("Configured to use CPU only.")

cap = cv2.VideoCapture(CAMERA_INDEX)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Detection started...")

# ==============================
# MEMORY
# ==============================

object_memory = {}
motion_memory = {}

prev_gray = None

# For FPS measurement and periodic summaries.
prev_time = time.time()
fps_value = 0.0
fps_alpha = 0.9  # smoothing factor
last_summary_time = 0.0
frame_index = 0

_last_alert_submit_ts = 0.0
_last_alert_key_submitted = None
_last_summary_submit_ts = 0.0
_last_summary_key_printed = None

# Alert (TTS) de-dupe
ALERT_COOLDOWN = 1.5
_last_alert_tts_key = None
_last_alert_tts_ts = 0.0

# Summary de-dupe
_last_summary_key = ()

# ==============================
# MAIN LOOP
# ==============================

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    frame_index += 1

    # ==============================
    # FPS MEASUREMENT
    # ==============================
    now_ts = time.time()
    dt = now_ts - prev_time
    prev_time = now_ts
    if dt > 0:
        instant_fps = 1.0 / dt
        fps_value = instant_fps if fps_value == 0.0 else fps_alpha * fps_value + (1.0 - fps_alpha) * instant_fps

    # ==============================
    # RESIZE FRAME FOR YOLO
    # ==============================

    frame_small = cv2.resize(
        frame,
        (RESIZE_WIDTH, int(height * (RESIZE_WIDTH / width)))
    )

    scale_x = width / frame_small.shape[1]
    scale_y = height / frame_small.shape[0]

    # ==============================
    # CAMERA MOTION
    # ==============================

    cam_motion_x = 0.0
    if ENABLE_CAMERA_MOTION:
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        if CAM_MOTION_DOWNSCALE != 1.0:
            gray = cv2.resize(
                gray,
                (0, 0),
                fx=CAM_MOTION_DOWNSCALE,
                fy=CAM_MOTION_DOWNSCALE,
                interpolation=cv2.INTER_AREA,
            )

        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                gray,
                None,
                0.5,
                2,
                12,
                2,
                5,
                1.1,
                0,
            )
            cam_motion_x = float(np.mean(flow[..., 0]))

        prev_gray = gray

    frame_output_data = []
    closest_close_obj = None  # (label, distance, direction)

    # ==============================
    # YOLO DETECTION
    # ==============================

    results = model.track(
        frame_small,
        persist=True,
        conf=YOLO_CONF,
        verbose=False,
        device=_DEVICE_ARG,
        half=_HALF_PRECISION_FLAG,
    )

    if results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.int().cpu().numpy()
        cls = results[0].boxes.cls.int().cpu().numpy()

        active_ids = set()

        for box, track_id, class_idx in zip(boxes, ids, cls):

            active_ids.add(track_id)

            x1, y1, x2, y2 = box

            # SCALE BACK TO ORIGINAL FRAME
            x1 = int(x1 * scale_x)
            x2 = int(x2 * scale_x)
            y1 = int(y1 * scale_y)
            y2 = int(y2 * scale_y)

            label = results[0].names[class_idx]

            center_x = (x1 + x2) / 2
            area = (x2 - x1) * (y2 - y1)

            distance = estimate_distance(area, width)

            # Skip very far objects entirely from drawing/summary.
            if distance > FAR_DISPLAY_MAX:
                continue

            if track_id not in object_memory:
                object_memory[track_id] = deque(maxlen=HISTORY_FRAMES)

            object_memory[track_id].append({
                "area": area,
                "cx": center_x
            })

            motion_guess = "tracking"

            if len(object_memory[track_id]) >= 5:

                history = object_memory[track_id]

                old = history[0]

                area_change = (area - old["area"]) / (old["area"] + 1e-6)

                delta_x = (center_x - old["cx"]) - cam_motion_x

                if area_change > GROWTH_THRESHOLD:
                    motion_guess = "approaching"

                elif area_change < SHRINK_THRESHOLD:
                    motion_guess = "moving away"

                elif abs(delta_x) > width * PASSING_THRESHOLD:
                    motion_guess = "passing"

                else:
                    motion_guess = "stationary"

            if track_id not in motion_memory:
                motion_memory[track_id] = deque(maxlen=MOTION_HISTORY)

            motion_memory[track_id].append(MOTION_MAP[motion_guess])

            counts = {}

            for m in motion_memory[track_id]:
                counts[m] = counts.get(m, 0) + 1

            motion = "tracking"

            for mid, count in counts.items():
                if count >= CONFIRM_THRESHOLD:
                    motion = REVERSE_MOTION_MAP[mid]

            direction = get_direction(center_x, width)

            risk_score = calculate_risk_score(label, motion, distance)

            risk_category = get_risk_category(risk_score)

            frame_output_data.append({
                "object": label,
                "motion": motion,
                "direction": direction,
                "risk": risk_category,
                "distance": distance,
            })

            # Box color by risk: yellow for medium, red for high, white for low.
            if risk_category == "high":
                box_color = (0, 0, 255)      # red
            elif risk_category == "medium":
                box_color = (0, 255, 255)    # yellow
            else:
                box_color = (255, 255, 255)  # white
            text_color = (0, 0, 0)           # black text for all

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            cv2.putText(
                frame,
                f"{label} | {motion} | {risk_category}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                text_color,
                2
            )

            # Track closest object within CLOSE_DISTANCE_MAX meters for on-screen alert.
            if distance < CLOSE_DISTANCE_MAX:
                candidate = {
                    "object": label,
                    "distance": distance,
                    "direction": direction,
                    "motion": motion,
                    "risk": risk_category,
                    "risk_score": risk_score,
                }
                if _is_dangerous_and_close(candidate):
                    if closest_close_obj is None or distance < closest_close_obj["distance"]:
                        closest_close_obj = candidate

        for obj_id in list(object_memory.keys()):

            if obj_id not in active_ids:

                del object_memory[obj_id]

                if obj_id in motion_memory:
                    del motion_memory[obj_id]

    # On-screen alert banner if something is dangerous and close.
    if closest_close_obj is not None:
        label = closest_close_obj["object"]
        dist = closest_close_obj["distance"]
        direction = closest_close_obj["direction"]
        motion = closest_close_obj.get("motion", "tracking")
        if direction == "left":
            dir_phrase = "on your left"
        elif direction == "right":
            dir_phrase = "on your right"
        else:
            dir_phrase = "ahead"

        motion_phrase = "approaching" if motion == "approaching" else "passing" if motion == "passing" else ""
        if motion_phrase:
            alert_text = f"ALERT: {label} {motion_phrase} {dir_phrase} ({dist:.1f} meters)"
        else:
            alert_text = f"ALERT: {label} {dir_phrase} ({dist:.1f} meters)"
        banner_color = (255, 255, 255)  # white banner
        text_color = (0, 0, 0)          # black text
        cv2.rectangle(frame, (0, 0), (width, 40), banner_color, -1)
        cv2.putText(
            frame,
            alert_text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            text_color,
            2,
        )
        # TTS alert: only when dangerous+close, and only when changed.
        now_for_alert = time.time()
        dist_key = int(round(dist))
        alert_key = (label, direction, motion, closest_close_obj.get("risk", "low"), dist_key)
        if (now_for_alert - _last_alert_tts_ts) >= ALERT_COOLDOWN and alert_key != _last_alert_tts_key:
            _tts_worker.submit(alert_text.replace("ALERT:", "").strip())
            _last_alert_tts_key = alert_key
            _last_alert_tts_ts = now_for_alert

    # Periodic short summary in the terminal based on current frame data.
    now_for_summary = time.time()
    if frame_output_data and (now_for_summary - last_summary_time) > TERMINAL_SUMMARY_INTERVAL:
        summary_text, summary_key = build_natural_summary(frame_output_data)
        if summary_text and summary_key != _last_summary_key:
            print("Summary:", summary_text)
            _tts_worker.submit(summary_text)
            _last_summary_key = summary_key

        last_summary_time = now_for_summary

    # (Removed combined realtime narration in favor of focused alert + 7s summary)

    # Optional FPS overlay (bottom-left), white text.
    if SHOW_FPS_OVERLAY and fps_value > 0.0:
        cv2.putText(
            frame,
            f"FPS: {fps_value:.1f}",
            (10, height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    cv2.imshow("Vision Assistant", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

print("Processing finished.")
