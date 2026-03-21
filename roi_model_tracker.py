import argparse
import io
import json
import os
import re
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import urlparse

import cv2
from PIL import Image
import torch

PROJECT_DIR = os.path.dirname(os.path.realpath(__file__))
HF_HOME_DIR = os.path.join(PROJECT_DIR, ".hf")
HF_HUB_CACHE_DIR = os.path.join(HF_HOME_DIR, "hub")
HF_MODULES_CACHE_DIR = os.path.join(HF_HOME_DIR, "modules")
WEB_DIR = os.path.join(PROJECT_DIR, "web")
HTML_PAGE_PATH = os.path.join(WEB_DIR, "roi_model_tracker.html")

os.environ.setdefault("HF_HOME", HF_HOME_DIR)
os.environ.setdefault("HF_HUB_CACHE", HF_HUB_CACHE_DIR)
os.environ.setdefault("HF_MODULES_CACHE", HF_MODULES_CACHE_DIR)
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from model_registry import MODEL_INDEX_PATH, choose_best_model, ensure_models_dir, load_model_index, model_index_needs_refresh
from rtsp_person_tracker import CAMERA_IP, CAMERA_NAME, CAMERA_PORT, RTSP_URL, resolve_default_model
from siyi_sdk.siyi_sdk import SIYISDK
from siyi_sdk.stream import SIYIRTSP
from ultralytics import YOLO


DEFAULT_MOONDREAM_PROMPT = "Name the main object in this cropped image using one or two words only."
MOONDREAM_MODEL_DIR = os.path.join(PROJECT_DIR, "moondream_model")

VIDEO_WIDTH = 640
VIDEO_HEIGHT = 360
DETECTION_WIDTH = 320
TRACK_INTERVAL_MS = 40
ATTITUDE_INTERVAL_MS = 300
DISPLAY_INTERVAL_MS = 40
MAX_YAW_SPEED = 25
MAX_PITCH_SPEED = 20
X_DEADBAND = 0.08
Y_DEADBAND = 0.10
YAW_GAIN = 70
PITCH_GAIN = 55
CONF_THRESHOLD = 0.35
TARGET_MATCH_DISTANCE = 140
INFERENCE_IDLE_MS = 0.01
YAW_SIGN = 1
PITCH_SIGN = 1
MJPEG_BOUNDARY = b"--frame"
AUTO_START_TRACKING = True


class MoondreamClassifier:
    _shared_model = None

    def __init__(self) -> None:
        self._setup()

    def _setup(self) -> None:
        if not os.path.isdir(MOONDREAM_MODEL_DIR):
            raise RuntimeError(f"Moondream model folder was not found at {MOONDREAM_MODEL_DIR}.")

        os.makedirs(HF_HOME_DIR, exist_ok=True)
        os.makedirs(HF_HUB_CACHE_DIR, exist_ok=True)
        os.makedirs(HF_MODULES_CACHE_DIR, exist_ok=True)

        if MoondreamClassifier._shared_model is None:
            from moondream_model.hf_moondream import HfConfig, HfMoondream

            config = HfConfig.from_pretrained(MOONDREAM_MODEL_DIR, local_files_only=True)
            load_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            model = HfMoondream.from_pretrained(
                MOONDREAM_MODEL_DIR,
                config=config,
                local_files_only=True,
                torch_dtype=load_dtype,
            )
            model = model.to("cuda" if torch.cuda.is_available() else "cpu")
            MoondreamClassifier._shared_model = model
            MoondreamClassifier._shared_model.eval()

        self.client = MoondreamClassifier._shared_model

    def classify(self, image_bgr, prompt: str = DEFAULT_MOONDREAM_PROMPT) -> str:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image_rgb)
        result = self.client.query(image, prompt)

        if isinstance(result, dict):
            text = result.get("answer") or result.get("text") or result.get("caption") or ""
        else:
            text = str(result)

        cleaned = re.sub(r"[\r\n]+", " ", text).strip().strip(".")
        if not cleaned:
            raise RuntimeError("Moondream returned an empty response.")
        return cleaned


def _choose_class_from_answer(answer: str, classes: list[str]) -> Optional[str]:
    normalized_answer = re.sub(r"[^a-z0-9]+", " ", answer.strip().lower())
    normalized_answer = re.sub(r"\s+", " ", normalized_answer).strip()

    for class_name in classes:
        normalized_class = re.sub(r"[^a-z0-9]+", " ", class_name.strip().lower())
        normalized_class = re.sub(r"\s+", " ", normalized_class).strip()
        if normalized_answer == normalized_class:
            return class_name

    for class_name in classes:
        normalized_class = re.sub(r"[^a-z0-9]+", " ", class_name.strip().lower())
        normalized_class = re.sub(r"\s+", " ", normalized_class).strip()
        if normalized_class and normalized_class in normalized_answer:
            return class_name

    return None


def classify_with_model_classes(classifier: MoondreamClassifier, crop, model_entry: dict) -> Optional[str]:
    classes = [str(item) for item in model_entry.get("classes", []) if str(item).strip()]
    if not classes:
        return None
    if len(classes) == 1:
        return classes[0]

    class_list = ", ".join(classes)
    prompt = (
        "Classify the main object in this cropped image. "
        f"Reply with exactly one label from this list: {class_list}. "
        "Do not add any extra words."
    )
    raw_answer = classifier.classify(crop, prompt=prompt)
    print(f'[Moondream] Constrained answer: {raw_answer}', flush=True)
    matched = _choose_class_from_answer(raw_answer, classes)
    if matched is not None:
        return matched

    lowered = {item.lower(): item for item in classes}
    if "civilian" in lowered:
        return lowered["civilian"]
    return classes[0]


def resolve_model_from_crop(crop, refresh_index: bool) -> tuple[str, str, str]:
    classifier = MoondreamClassifier()
    detected_object = classifier.classify(crop)
    print(f'[Moondream] Raw answer: {detected_object}', flush=True)
    best_model, best_class, _ranked = choose_best_model(detected_object, refresh=refresh_index)

    if best_model is None or best_class is None:
        if best_model is None:
            raise RuntimeError(
                f'Moondream detected "{detected_object}", but no matching model was found in {MODEL_INDEX_PATH}.'
            )
        best_class = classify_with_model_classes(classifier, crop, best_model)
        if best_class is None:
            raise RuntimeError(
                f'Moondream detected "{detected_object}", but no matching class was found in {MODEL_INDEX_PATH}.'
            )
    elif best_class not in [str(item) for item in best_model.get("classes", [])]:
        best_class = classify_with_model_classes(classifier, crop, best_model)

    return detected_object, str(best_model["path"]), best_class


class RoiTrackerService:
    _model_cache: dict[str, YOLO] = {}

    def __init__(self, rtsp_url: str, refresh_index: bool) -> None:
        self.rtsp_url = rtsp_url
        self.refresh_index = refresh_index
        self.cam: Optional[SIYISDK] = None
        self.stream: Optional[SIYIRTSP] = None
        self.model = None
        self.model_name: Optional[str] = None
        self.model_path: Optional[str] = None
        self.use_cuda = torch.cuda.is_available()

        self.tracking_enabled = False
        self.current_yaw = 0.0
        self.current_pitch = 0.0
        self.last_target = None
        self.target_center = None
        self.last_detection_time = 0.0
        self.target_class: Optional[str] = None

        self.selected_model_path: Optional[str] = None
        self.selected_class: Optional[str] = None
        self.detected_object: Optional[str] = None
        self.selected_roi: Optional[tuple[int, int, int, int]] = None

        self.status_text = "Connecting..."
        self.result_text = "No object selected."

        self.frame_lock = threading.Lock()
        self.target_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.latest_frame = None
        self.latest_jpeg: Optional[bytes] = None
        self.frame_shape = {"width": VIDEO_WIDTH, "height": VIDEO_HEIGHT}

        default_model_path, _ = resolve_default_model()
        self._load_tracking_model(default_model_path)
        self._connect()

        self.inference_thread = threading.Thread(target=self.inference_loop, daemon=True)
        self.render_thread = threading.Thread(target=self.render_loop, daemon=True)
        self.track_thread = threading.Thread(target=self.track_loop, daemon=True)
        self.attitude_thread = threading.Thread(target=self.attitude_loop, daemon=True)

        self.inference_thread.start()
        self.render_thread.start()
        self.track_thread.start()
        self.attitude_thread.start()

    def _connect(self) -> None:
        try:
            self.cam = SIYISDK(server_ip=CAMERA_IP, port=CAMERA_PORT)
            if not self.cam.connect():
                self.status_text = "Failed to connect to gimbal control."
                return

            self.cam.requestFollowMode()
            self.cam.requestHardwareID()
            self.cam.requestGimbalAttitude()
            self.stream = SIYIRTSP(rtsp_url=self.rtsp_url, cam_name=CAMERA_NAME, debug=False)
            device_name = "CUDA" if self.use_cuda else "CPU"
            self.status_text = f"Connected to {CAMERA_IP}. Model loaded: {self.model_name} on {device_name}"
        except Exception as exc:
            self.status_text = f"Connection error: {exc}"

    def _load_tracking_model(self, model_path: str) -> None:
        self.model_path = model_path
        if model_path not in self._model_cache:
            self._model_cache[model_path] = YOLO(model_path)
        self.model = self._model_cache[model_path]
        self.model_name = os.path.basename(model_path)

    def _prime_initial_target(self, bbox: tuple[int, int, int, int]) -> None:
        x, y, w, h = bbox
        cx = x + (w / 2.0)
        cy = y + (h / 2.0)
        seeded_target = {
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "confidence": 1.0,
            "cls_id": -1,
            "class_name": self.target_class or "selected",
            "cx": cx,
            "cy": cy,
        }
        with self.target_lock:
            self.last_target = seeded_target
            self.last_detection_time = time.time()
        self.target_center = (cx, cy)

    def rotate_camera(self, yaw_speed: int, pitch_speed: int) -> None:
        if self.cam is None:
            self.status_text = "Gimbal control is not connected."
            return
        self.cam.requestGimbalSpeed(yaw_speed, pitch_speed)
        self.status_text = f"Manual camera move: yaw={yaw_speed}, pitch={pitch_speed}"

    def stop_motion(self) -> None:
        if self.cam is not None:
            self.cam.requestGimbalSpeed(0, 0)

    def center_gimbal(self) -> None:
        if self.cam is None:
            self.status_text = "Gimbal control is not connected."
            return
        self.cam.requestCenterGimbal()
        self.current_yaw = 0.0
        self.current_pitch = 0.0
        with self.target_lock:
            self.last_target = None
        self.target_center = None
        self.status_text = "Center command sent."

    def toggle_tracking(self) -> None:
        self.tracking_enabled = not self.tracking_enabled
        if self.tracking_enabled:
            self.status_text = "Tracking enabled."
        else:
            self.stop_motion()
            self.target_center = None
            with self.target_lock:
                self.last_target = None
            self.status_text = "Tracking stopped."

    def resize_for_detection(self, frame):
        height, width = frame.shape[:2]
        if width <= DETECTION_WIDTH:
            return frame
        scale = DETECTION_WIDTH / float(width)
        return cv2.resize(frame, (DETECTION_WIDTH, int(height * scale)))

    def class_name(self, cls_id: int) -> str:
        names = self.model.names
        if isinstance(names, dict):
            return str(names.get(cls_id, cls_id))
        if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
            return str(names[cls_id])
        return str(cls_id)

    def detect_person(self, frame):
        resized = self.resize_for_detection(frame)
        results = self.model.predict(
            source=resized,
            conf=CONF_THRESHOLD,
            imgsz=DETECTION_WIDTH,
            device=0 if self.use_cuda else "cpu",
            half=self.use_cuda,
            max_det=10,
            verbose=False,
        )
        if not results:
            return None

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None

        scale_x = frame.shape[1] / resized.shape[1]
        scale_y = frame.shape[0] / resized.shape[0]
        candidates = []

        xyxy_list = boxes.xyxy.cpu().tolist()
        conf_list = boxes.conf.cpu().tolist()
        cls_list = boxes.cls.cpu().tolist() if boxes.cls is not None else [0.0] * len(xyxy_list)

        for xyxy, confidence, cls_id in zip(xyxy_list, conf_list, cls_list):
            class_name = self.class_name(int(cls_id))
            if self.target_class and class_name.lower() != self.target_class.lower():
                continue

            x1, y1, x2, y2 = xyxy
            x = int(x1 * scale_x)
            y = int(y1 * scale_y)
            w = int((x2 - x1) * scale_x)
            h = int((y2 - y1) * scale_y)
            cx = x + (w / 2.0)
            cy = y + (h / 2.0)
            candidates.append(
                {
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "confidence": float(confidence),
                    "cls_id": int(cls_id),
                    "class_name": class_name,
                    "cx": cx,
                    "cy": cy,
                }
            )

        if not candidates:
            return None

        selected = self.select_target(candidates)
        self.target_center = (selected["cx"], selected["cy"])
        return selected

    def select_target(self, candidates):
        if self.target_center is None:
            return max(candidates, key=lambda item: item["confidence"] * item["w"] * item["h"])

        px, py = self.target_center
        best = None
        best_score = None

        for item in candidates:
            distance = ((item["cx"] - px) ** 2 + (item["cy"] - py) ** 2) ** 0.5
            score = distance - (item["confidence"] * 40.0)
            if distance > TARGET_MATCH_DISTANCE and item["confidence"] < 0.6:
                continue
            if best is None or score < best_score:
                best = item
                best_score = score

        if best is None:
            best = max(candidates, key=lambda item: item["confidence"] * item["w"] * item["h"])
        return best

    def compute_speed(self, normalized_error: float, deadband: float, gain: float, max_speed: int) -> int:
        if abs(normalized_error) < deadband:
            return 0
        speed = int(gain * normalized_error)
        if speed > max_speed:
            return max_speed
        if speed < -max_speed:
            return -max_speed
        return speed

    def draw_overlay(self, frame):
        output = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
        h, w = output.shape[:2]

        cv2.line(output, (w // 2, 0), (w // 2, h), (0, 255, 255), 1)
        cv2.line(output, (0, h // 2), (w, h // 2), (0, 255, 255), 1)

        target = None
        with self.target_lock:
            if self.last_target is not None:
                target = dict(self.last_target)

        if target is not None:
            x = target["x"]
            y = target["y"]
            bw = target["w"]
            bh = target["h"]
            confidence = target["confidence"]
            cls_id = target["cls_id"]
            class_name = target.get("class_name", self.class_name(cls_id))
            scale_x = VIDEO_WIDTH / frame.shape[1]
            scale_y = VIDEO_HEIGHT / frame.shape[0]

            x1 = int(x * scale_x)
            y1 = int(y * scale_y)
            x2 = int((x + bw) * scale_x)
            y2 = int((y + bh) * scale_y)

            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                output,
                f"{class_name} {confidence:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        if self.tracking_enabled:
            cv2.putText(
                output,
                "TRACKING ON",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        return output

    def inference_loop(self) -> None:
        while not self.stop_event.is_set():
            if not self.tracking_enabled or self.stream is None:
                time.sleep(INFERENCE_IDLE_MS)
                continue

            frame = self.stream.getFrame()
            if frame is None:
                time.sleep(INFERENCE_IDLE_MS)
                continue

            target = self.detect_person(frame)
            with self.target_lock:
                self.last_target = target
                self.last_detection_time = time.time()

            if target is None:
                self.target_center = None

    def render_loop(self) -> None:
        while not self.stop_event.is_set():
            frame = self.stream.getFrame() if self.stream is not None else None
            if frame is not None:
                with self.frame_lock:
                    self.latest_frame = frame.copy()
                    self.frame_shape = {"width": int(frame.shape[1]), "height": int(frame.shape[0])}
                display_frame = self.draw_overlay(frame)
                ok, encoded = cv2.imencode(".jpg", display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok:
                    with self.frame_lock:
                        self.latest_jpeg = encoded.tobytes()
            time.sleep(DISPLAY_INTERVAL_MS / 1000.0)

    def track_loop(self) -> None:
        while not self.stop_event.is_set():
            frame = self.stream.getFrame() if self.stream is not None else None
            target = None

            with self.target_lock:
                if self.last_target is not None:
                    target = dict(self.last_target)
                detection_age = time.time() - self.last_detection_time if self.last_detection_time else None

            if self.tracking_enabled and self.cam is not None and frame is not None:
                if target is None or (detection_age is not None and detection_age > 0.5):
                    self.stop_motion()
                    self.target_center = None
                    self.status_text = "No model detection."
                else:
                    center_x = target["cx"]
                    center_y = target["cy"]
                    class_name = target["class_name"]
                    frame_center_x = frame.shape[1] / 2.0
                    frame_center_y = frame.shape[0] / 2.0

                    x_error = (center_x - frame_center_x) / frame.shape[1]
                    y_error = (center_y - frame_center_y) / frame.shape[0]

                    yaw_speed = self.compute_speed(x_error, X_DEADBAND, YAW_GAIN, MAX_YAW_SPEED) * YAW_SIGN
                    pitch_speed = self.compute_speed(y_error, Y_DEADBAND, PITCH_GAIN, MAX_PITCH_SPEED) * PITCH_SIGN

                    self.cam.requestGimbalSpeed(yaw_speed, pitch_speed)
                    self.status_text = f"Tracking target. yaw_speed={yaw_speed}, pitch_speed={pitch_speed}"

            time.sleep(TRACK_INTERVAL_MS / 1000.0)

    def attitude_loop(self) -> None:
        while not self.stop_event.is_set():
            if self.cam is not None:
                try:
                    self.cam.requestGimbalAttitude()
                    yaw, pitch, _ = self.cam.getAttitude()
                    self.current_yaw = yaw
                    self.current_pitch = pitch
                except Exception:
                    pass
            time.sleep(ATTITUDE_INTERVAL_MS / 1000.0)

    def select_object(self, x: int, y: int, w: int, h: int) -> None:
        with self.frame_lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()

        if frame is None:
            raise RuntimeError("No live frame available yet.")
        if w <= 0 or h <= 0:
            raise RuntimeError("Selection canceled.")

        crop = frame[y : y + h, x : x + w].copy()
        self.status_text = "Running Moondream and matching a tracking model..."
        detected_object, model_path, model_class = resolve_model_from_crop(crop, refresh_index=self.refresh_index)

        was_tracking = self.tracking_enabled
        if was_tracking:
            self.toggle_tracking()

        self.detected_object = detected_object
        self.selected_model_path = model_path
        self.selected_class = model_class
        self.selected_roi = (x, y, w, h)
        self.target_class = None
        self._load_tracking_model(model_path)
        self._prime_initial_target(self.selected_roi)
        self.result_text = f'Detected "{detected_object}". Selected model "{os.path.basename(model_path)}".'

        if AUTO_START_TRACKING and not self.tracking_enabled:
            self.toggle_tracking()
            self.status_text = "Model selected. Tracking started."
        else:
            self.status_text = "Model selected. Ready to track."

    def status_payload(self) -> dict:
        target = None
        with self.target_lock:
            if self.last_target is not None:
                target = dict(self.last_target)

        return {
            "status": self.status_text,
            "result": self.result_text,
            "tracking_enabled": self.tracking_enabled,
            "detected_object": self.detected_object,
            "selected_model": os.path.basename(self.selected_model_path) if self.selected_model_path else None,
            "selected_class": self.selected_class,
            "yaw": round(self.current_yaw, 1),
            "pitch": round(self.current_pitch, 1),
            "frame": self.frame_shape,
            "target": target,
        }

    def latest_mjpeg_frame(self) -> Optional[bytes]:
        with self.frame_lock:
            return self.latest_jpeg

    def shutdown(self) -> None:
        self.tracking_enabled = False
        self.stop_event.set()

        try:
            self.stop_motion()
        except Exception:
            pass

        try:
            if self.stream is not None:
                self.stream.close()
        except Exception:
            pass

        try:
            if self.cam is not None:
                self.cam.disconnect()
        except Exception:
            pass


class RoiTrackerRequestHandler(BaseHTTPRequestHandler):
    server_version = "RoiTrackerHTTP/1.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._serve_html()
            return
        if parsed.path == "/api/status":
            self._serve_json(self.server.app.status_payload())
            return
        if parsed.path == "/stream.mjpg":
            self._serve_mjpeg()
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        payload = self._read_json_body()
        try:
            if parsed.path == "/api/select-roi":
                self.server.app.select_object(
                    int(payload["x"]),
                    int(payload["y"]),
                    int(payload["w"]),
                    int(payload["h"]),
                )
                self._serve_json({"ok": True, "status": self.server.app.status_payload()})
                return
            if parsed.path == "/api/toggle-tracking":
                self.server.app.toggle_tracking()
                self._serve_json({"ok": True, "status": self.server.app.status_payload()})
                return
            if parsed.path == "/api/center":
                self.server.app.center_gimbal()
                self._serve_json({"ok": True, "status": self.server.app.status_payload()})
                return
            if parsed.path == "/api/move":
                self.server.app.rotate_camera(int(payload.get("yaw", 0)), int(payload.get("pitch", 0)))
                self._serve_json({"ok": True, "status": self.server.app.status_payload()})
                return
            if parsed.path == "/api/stop-motion":
                self.server.app.stop_motion()
                self._serve_json({"ok": True, "status": self.server.app.status_payload()})
                return
        except Exception as exc:
            self._serve_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        self.send_error(HTTPStatus.NOT_FOUND)

    def _serve_html(self) -> None:
        with open(HTML_PAGE_PATH, "rb") as handle:
            body = handle.read()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_mjpeg(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Pragma", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        try:
            while not self.server.app.stop_event.is_set():
                frame = self.server.app.latest_mjpeg_frame()
                if frame is None:
                    time.sleep(0.05)
                    continue

                self.wfile.write(MJPEG_BOUNDARY + b"\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                time.sleep(DISPLAY_INTERVAL_MS / 1000.0)
        except (BrokenPipeError, ConnectionResetError):
            return

    def _read_json_body(self) -> dict:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            return {}
        raw = self.rfile.read(content_length)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def log_message(self, format: str, *args) -> None:
        return


class RoiTrackerHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address, app: RoiTrackerService):
        super().__init__(server_address, RoiTrackerRequestHandler)
        self.app = app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Web ROI object selection with Moondream-based model choice.")
    parser.add_argument("--rtsp-url", default=RTSP_URL, help="RTSP stream URL.")
    parser.add_argument("--refresh-index", action="store_true", help="Rebuild the models index before matching.")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP host to bind.")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port to bind.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_models_dir()
    refresh_index = args.refresh_index or model_index_needs_refresh()
    models = load_model_index(refresh=refresh_index)
    if refresh_index:
        print(f"[Model Index] Indexed {len(models)} model(s).", flush=True)
    else:
        print(f"[Model Index] No new model found. Using existing index for {len(models)} model(s).", flush=True)

    app = RoiTrackerService(rtsp_url=args.rtsp_url, refresh_index=refresh_index)
    server = RoiTrackerHTTPServer((args.host, args.port), app)
    print(f"[Web UI] Open http://{args.host}:{args.port}", flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        app.shutdown()


if __name__ == "__main__":
    main()
