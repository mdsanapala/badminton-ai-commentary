"""
===============================================================
  AI-Based Automatic Commentary System for Badminton
  Stack   : Python · OpenCV · YOLOv8 · pyttsx3
  Fix     : Removed cv2.imshow / waitKey (crashes headless)
            Added progress logging
===============================================================
"""

import cv2
import numpy as np
import time
from pathlib import Path
from collections import deque
from itertools import combinations
from ultralytics import YOLO

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
CONFIG = {
    "input_video":         "input.mp4",
    "output_video":        ".dist/output.mp4",
    "yolo_model":          "yolov8n.pt",

    # Motion thresholds (tune these if needed)
    "hit_threshold":        8_000,
    "smash_threshold":     22_000,
    "rally_threshold":      4_000,
    "diff_thresh_val":         25,

    # Commentary cooldown in seconds
    "commentary_cooldown":    1.8,

    # Rolling motion average window
    "motion_history":           5,

    # How many frames to show commentary text
    "text_hold_frames":        45,

    # End-card duration (seconds)
    "end_card_duration":        5,

    # Process every Nth frame with YOLO (speeds things up)
    "yolo_every_n_frames":      2,

    # Optional: polygon around playable court area.
    # Example: [[220, 120], [1180, 170], [1020, 690], [160, 640]]
    # Leave empty [] to disable.
    "court_polygon":            [[220, 120], [1180, 170], [1020, 690], [160, 640]],  # Example coordinates - adjust to your video's court

    # Optional: exclude regions where referees typically stand (e.g., center sidelines)
    # Format: list of polygons, each polygon is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    "referee_exclusion_zones":   [[[500, 100], [700, 100], [700, 150], [500, 150]]],  # Example rectangle - adjust as needed

    # Optional: axis for consistent Player A/B labeling in diagonal views.
    # Player A is closer to first point, Player B to second point.
    # Example: [[180, 660], [1140, 140]]
    "player_axis_points":       [],
}

# ──────────────────────────────────────────────
# COMMENTARY BANK
# ──────────────────────────────────────────────
EVENTS = {
    "A_smash": ["Player A smashes the shuttle!",
                "Powerful smash from Player A!",
                "Player A goes for the kill!"],
    "A_hit":   ["Player A hits the shuttle.",
                "Player A plays a shot.",
                "Shot from Player A!"],
    "B_smash": ["Player B smashes it back!",
                "Thunderous smash by Player B!",
                "Player B attacks!"],
    "B_hit":   ["Player B returns the shot.",
                "Player B plays the shuttle.",
                "Nice return by Player B!"],
    "rally":   ["Great rally continuing!",
                "Both players trading shots!",
                "Intense rally underway!"],
    "idle":    [""],
}
_counters = {k: 0 for k in EVENTS}


def pick_commentary(event_key: str) -> str:
    phrases = EVENTS[event_key]
    idx = _counters[event_key] % len(phrases)
    _counters[event_key] += 1
    return phrases[idx]


# ──────────────────────────────────────────────
# MOTION DETECTION
# ──────────────────────────────────────────────
def compute_motion_mask(prev_gray, curr_gray, thresh_val: int):
    diff = cv2.absdiff(prev_gray, curr_gray)
    _, mask = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def region_motion_score(mask, bbox) -> float:
    if mask is None or bbox is None:
        return 0.0
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1);  y1 = max(0, y1)
    x2 = min(mask.shape[1], x2);  y2 = min(mask.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float(np.sum(mask[y1:y2, x1:x2]))


def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_foot_point(bbox):
    x1, _, x2, y2 = bbox
    return (int((x1 + x2) / 2), int(y2))


def center_distance(b1, b2) -> float:
    c1 = bbox_center(b1)
    c2 = bbox_center(b2)
    return float(np.hypot(c1[0] - c2[0], c1[1] - c2[1]))


def point_in_polygon(point, polygon) -> bool:
    if not polygon:
        return True
    poly = np.array(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(poly, point, False) >= 0


# ──────────────────────────────────────────────
# PLAYER DETECTION
# ──────────────────────────────────────────────
def detect_players(model, frame, frame_h: int, cfg: dict, prev_players=None):
    results = model(frame, verbose=False)
    persons = []
    court_polygon = cfg.get("court_polygon", [])
    exclusion_zones = cfg.get("referee_exclusion_zones", [])

    for r in results:
        for box in r.boxes:
            if model.names[int(box.cls[0])] != "person":
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bbox = (x1, y1, x2, y2)
            foot_point = bbox_foot_point(bbox)
            if court_polygon and not point_in_polygon(foot_point, court_polygon):
                continue
            # Check if person is in referee exclusion zones
            in_exclusion = False
            for zone in exclusion_zones:
                if point_in_polygon(foot_point, zone):
                    in_exclusion = True
                    break
            if in_exclusion:
                continue
            area = (x2 - x1) * (y2 - y1)
            conf = float(box.conf[0])
            persons.append((area * conf, bbox))

    persons.sort(reverse=True)
    candidates = [bbox for (_, bbox) in persons[:6]]
    if len(candidates) < 2:
        return None, None

    top2 = None
    if prev_players and prev_players[0] and prev_players[1]:
        prev_a, prev_b = prev_players
        best_score = float("-inf")
        for b1, b2 in combinations(candidates, 2):
            consistency_1 = center_distance(b1, prev_a) + center_distance(b2, prev_b)
            consistency_2 = center_distance(b2, prev_a) + center_distance(b1, prev_b)
            consistency = min(consistency_1, consistency_2)
            size_score = (b1[2] - b1[0]) * (b1[3] - b1[1]) + (b2[2] - b2[0]) * (b2[3] - b2[1])
            score = size_score - 4.0 * consistency
            if score > best_score:
                best_score = score
                top2 = [b1, b2]

    if top2 is None:
        top2 = candidates[:2]

    playerA, playerB = None, None
    axis_points = cfg.get("player_axis_points", [])
    if len(axis_points) == 2:
        p0 = np.array(axis_points[0], dtype=np.float32)
        p1 = np.array(axis_points[1], dtype=np.float32)
        axis = p1 - p0
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 1e-6:
            axis = axis / axis_norm
            projs = []
            for bbox in top2:
                c = np.array(bbox_center(bbox), dtype=np.float32)
                projs.append((float(np.dot(c - p0, axis)), bbox))
            projs.sort(key=lambda x: x[0])
            playerA, playerB = projs[0][1], projs[1][1]

    if playerA is None or playerB is None:
        for bbox in top2:
            x1, y1, x2, y2 = bbox
            y_center = (y1 + y2) / 2
            if y_center < frame_h / 2:
                playerA = bbox
            else:
                playerB = bbox

        if len(top2) == 2 and (playerA is None or playerB is None):
            top2_sorted_x = sorted(top2, key=lambda b: (b[0] + b[2]) / 2)
            playerA, playerB = top2_sorted_x[0], top2_sorted_x[1]

    return playerA, playerB


# ──────────────────────────────────────────────
# EVENT CLASSIFICATION
# ──────────────────────────────────────────────
def classify_event(score_a: float, score_b: float, cfg: dict) -> str:
    if score_a >= cfg["smash_threshold"]:
        return "A_smash"
    if score_b >= cfg["smash_threshold"]:
        return "B_smash"
    if score_a >= cfg["hit_threshold"]:
        return "A_hit"
    if score_b >= cfg["hit_threshold"]:
        return "B_hit"
    if (score_a + score_b) >= cfg["rally_threshold"]:
        return "rally"
    return "idle"


# ──────────────────────────────────────────────
# DRAWING
# ──────────────────────────────────────────────
def draw_bbox(frame, bbox, label: str, color):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)


def draw_commentary(frame, text: str):
    if not text:
        return
    h, w = frame.shape[:2]
    banner_h = 70
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    font       = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.9
    thickness  = 2

    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    tx = (w - tw) // 2
    ty = (banner_h + th) // 2

    # Drop shadow
    cv2.putText(frame, text, (tx + 2, ty + 2), font, font_scale, (0, 0, 0), thickness)
    # White text
    cv2.putText(frame, text, (tx, ty),         font, font_scale, (255, 255, 255), thickness)


# ──────────────────────────────────────────────
# END-CARD
# ──────────────────────────────────────────────
def create_end_card(w: int, h: int) -> np.ndarray:
    card = np.zeros((h, w, 3), dtype=np.uint8)
    # Gradient background
    for row in range(h):
        t = row / h
        card[row, :] = (int(20 + t*30), int(10 + t*60), int(30 + t*50))

    items = [
        ("AI Badminton Commentary System", 1.0,  (255, 220, 60),  2, 0.18),
        ("Tech Stack",                     0.75, (160, 255, 200), 2, 0.30),
        ("Python 3.x",                     0.65, (220, 220, 220), 1, 0.40),
        ("OpenCV  —  Video Processing",    0.65, (220, 220, 220), 1, 0.48),
        ("YOLOv8 (Ultralytics) — Detection", 0.65, (220, 220, 220), 1, 0.56),
        ("pyttsx3  —  Text-to-Speech",     0.65, (220, 220, 220), 1, 0.64),
        ("NumPy  —  Motion Analysis",      0.65, (220, 220, 220), 1, 0.72),
        ("Rule-Based Event Classification",0.55, (150, 190, 255), 1, 0.82),
    ]
    for text, scale, color, thick, yf in items:
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, thick)
        tx = (w - tw) // 2
        ty = int(h * yf)
        cv2.putText(card, text, (tx + 2, ty + 2),
                    cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), thick)
        cv2.putText(card, text, (tx, ty),
                    cv2.FONT_HERSHEY_DUPLEX, scale, color, thick)
    return card


# ──────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────
def run_pipeline(cfg: dict):
    print("[INFO] Initialising …")

    base_dir = Path(__file__).resolve().parent
    input_path = Path(cfg["input_video"])
    if not input_path.is_absolute():
        input_path = base_dir / input_path

    output_path = Path(cfg["output_video"])
    if not output_path.is_absolute():
        output_path = base_dir / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {input_path}")

    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0
    TOTAL = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Video: {W}x{H} @ {FPS:.1f} fps, {TOTAL} frames")

    out = cv2.VideoWriter(str(output_path),
                          cv2.VideoWriter_fourcc(*"mp4v"),
                          FPS, (W, H))
    if not out.isOpened():
        raise RuntimeError("VideoWriter failed to open. Check codec support.")

    model = YOLO(cfg["yolo_model"])

    # State
    prev_gray     = None
    motion_buf_a  = deque(maxlen=cfg["motion_history"])
    motion_buf_b  = deque(maxlen=cfg["motion_history"])
    current_text  = ""
    text_frames   = 0
    last_event    = "idle"
    last_spoken_t = 0.0
    playerA_cache = None
    playerB_cache = None
    frame_idx     = 0
    t_start       = time.time()

    print("[INFO] Processing … (no preview window — headless mode)")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Motion mask
        mask = compute_motion_mask(prev_gray, gray, cfg["diff_thresh_val"]) \
               if prev_gray is not None else None
        prev_gray = gray

        # YOLO detection (every N frames to keep speed up)
        if frame_idx % cfg["yolo_every_n_frames"] == 0:
            playerA_cache, playerB_cache = detect_players(
                model,
                frame,
                H,
                cfg,
                prev_players=(playerA_cache, playerB_cache),
            )

        playerA = playerA_cache
        playerB = playerB_cache

        # Motion scores (smoothed)
        motion_buf_a.append(region_motion_score(mask, playerA))
        motion_buf_b.append(region_motion_score(mask, playerB))
        smooth_a = float(np.mean(motion_buf_a)) if motion_buf_a else 0.0
        smooth_b = float(np.mean(motion_buf_b)) if motion_buf_b else 0.0

        # Event + commentary
        event = classify_event(smooth_a, smooth_b, cfg)
        now   = time.time()
        cooldown_ok = (now - last_spoken_t) >= cfg["commentary_cooldown"]

        if event != "idle" and cooldown_ok and event != last_event:
            current_text  = pick_commentary(event)
            text_frames   = cfg["text_hold_frames"]
            last_event    = event
            last_spoken_t = now
            # NOTE: pyttsx3 voice is intentionally omitted here.
            # It blocks the loop and makes output video ~10x slower to render.
            # Add it back if running interactively (not for batch output).
            # engine.say(current_text); engine.runAndWait()

        # Decay text hold counter
        if text_frames > 0:
            text_frames -= 1
        else:
            current_text = ""

        # Draw
        COLORS = {"A": (255, 100, 50), "B": (50, 220, 100)}
        if playerA:
            draw_bbox(frame, playerA, "Player A", COLORS["A"])
        if playerB:
            draw_bbox(frame, playerB, "Player B", COLORS["B"])
        draw_commentary(frame, current_text)

        # Progress counter (bottom-right corner)
        pct = f"{frame_idx}/{TOTAL}"
        cv2.putText(frame, pct, (W - 150, H - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        out.write(frame)

        # Log progress every 300 frames
        if frame_idx % 300 == 0:
            elapsed = time.time() - t_start
            fps_proc = frame_idx / elapsed
            eta = (TOTAL - frame_idx) / fps_proc if fps_proc > 0 else 0
            print(f"  [{frame_idx}/{TOTAL}]  {fps_proc:.1f} fps  ETA {eta:.0f}s")

    cap.release()

    # Append end-card
    print("[INFO] Appending tech-stack end-card …")
    end_card   = create_end_card(W, H)
    end_frames = int(FPS * cfg["end_card_duration"])
    for _ in range(end_frames):
        out.write(end_card)

    out.release()
    elapsed = time.time() - t_start
    print(f"[INFO] Done in {elapsed:.1f}s  →  {output_path}")


if __name__ == "__main__":
    run_pipeline(CONFIG)