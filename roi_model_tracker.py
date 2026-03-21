import argparse
import os
import re
import sys
import tkinter as tk
import time
from tkinter import ttk
from typing import Optional

import cv2
from PIL import Image
import torch

PROJECT_DIR = os.path.dirname(os.path.realpath(__file__))
HF_HOME_DIR = os.path.join(PROJECT_DIR, ".hf")
HF_HUB_CACHE_DIR = os.path.join(HF_HOME_DIR, "hub")
HF_MODULES_CACHE_DIR = os.path.join(HF_HOME_DIR, "modules")

os.environ.setdefault("HF_HOME", HF_HOME_DIR)
os.environ.setdefault("HF_HUB_CACHE", HF_HUB_CACHE_DIR)
os.environ.setdefault("HF_MODULES_CACHE", HF_MODULES_CACHE_DIR)
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from model_registry import MODEL_INDEX_PATH, choose_best_model, ensure_models_dir, load_model_index, model_index_needs_refresh
from rtsp_person_tracker import CAMERA_IP, CAMERA_NAME, CAMERA_PORT, RTSP_URL, HumanTrackerApp, resolve_default_model
from siyi_sdk.siyi_sdk import SIYISDK
from siyi_sdk.stream import SIYIRTSP
from ultralytics import YOLO


DEFAULT_MOONDREAM_PROMPT = "Name the main object in this cropped image using one or two words only."
MANUAL_YAW_SPEED = 18
MANUAL_PITCH_SPEED = 18
MOONDREAM_MODEL_DIR = os.path.join(PROJECT_DIR, "moondream_model")
AUTO_START_TRACKING = True


class MoondreamClassifier:
    _shared_model = None

    def __init__(self) -> None:
        self._setup()

    def _setup(self) -> None:
        if not os.path.isdir(MOONDREAM_MODEL_DIR):
            raise RuntimeError(
                f"Moondream model folder was not found at {MOONDREAM_MODEL_DIR}."
            )

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
            if torch.cuda.is_available():
                model = model.to("cuda")
            else:
                model = model.to("cpu")
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

    # Safety fallback for human categories: default to the least specific label.
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


class LiveObjectSelectorApp(HumanTrackerApp):
    _model_cache: dict[str, YOLO] = {}

    def __init__(self, root: tk.Tk, rtsp_url: str, refresh_index: bool) -> None:
        self.rtsp_url = rtsp_url
        self.refresh_index = refresh_index
        self.selected_model_path: Optional[str] = None
        self.selected_class: Optional[str] = None
        self.detected_object: Optional[str] = None
        self.selected_roi: Optional[tuple[int, int, int, int]] = None
        self.result_var = tk.StringVar(value="No object selected.")
        default_model_path, _ = resolve_default_model()
        super().__init__(root, model_path=default_model_path, target_class=None, auto_start=False)
        self.track_button.configure(state="disabled")

    def _bind_hold_button(self, button: ttk.Button, yaw_speed: int, pitch_speed: int) -> None:
        button.bind("<ButtonPress-1>", lambda _event: self.rotate_camera(yaw_speed, pitch_speed))
        button.bind("<ButtonRelease-1>", lambda _event: self.stop_motion())

    def rotate_camera(self, yaw_speed: int, pitch_speed: int) -> None:
        if self.cam is None:
            self.status_var.set("Gimbal control is not connected.")
            return
        self.cam.requestGimbalSpeed(yaw_speed, pitch_speed)
        self.status_var.set(f"Manual camera move: yaw={yaw_speed}, pitch={pitch_speed}")

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill="both", expand=True)

        self.video_label = ttk.Label(main, text="Waiting for RTSP video...", anchor="center")
        self.video_label.pack(fill="x")

        info_frame = ttk.Frame(main, padding=(0, 10, 0, 10))
        info_frame.pack(fill="x")

        ttk.Label(info_frame, textvariable=self.status_var).pack(anchor="w")
        ttk.Label(info_frame, textvariable=self.result_var).pack(anchor="w")
        ttk.Label(info_frame, textvariable=self.angle_var).pack(anchor="w")
        ttk.Label(info_frame, textvariable=self.target_var).pack(anchor="w")

        controls = ttk.Frame(main)
        controls.pack(fill="x")
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(2, weight=1)

        self.select_button = ttk.Button(controls, text="Select Object", command=self.select_object)
        self.select_button.grid(row=0, column=0, padx=6, pady=6, sticky="ew")

        self.track_button = ttk.Button(controls, text="Start Tracking", command=self.toggle_tracking, state="disabled")
        self.track_button.grid(row=0, column=1, padx=6, pady=6, sticky="ew")

        ttk.Button(controls, text="Close", command=self.on_close).grid(row=0, column=2, padx=6, pady=6, sticky="ew")

        gimbal = ttk.LabelFrame(main, text="Camera Rotation", padding=12)
        gimbal.pack(fill="x", pady=(8, 0))
        for column in range(3):
            gimbal.columnconfigure(column, weight=1)

        up_button = ttk.Button(gimbal, text="Up")
        up_button.grid(row=0, column=1, padx=6, pady=6, sticky="ew")
        self._bind_hold_button(up_button, yaw_speed=0, pitch_speed=-MANUAL_PITCH_SPEED)

        left_button = ttk.Button(gimbal, text="Left")
        left_button.grid(row=1, column=0, padx=6, pady=6, sticky="ew")
        self._bind_hold_button(left_button, yaw_speed=-MANUAL_YAW_SPEED, pitch_speed=0)

        ttk.Button(gimbal, text="Stop", command=self.stop_motion).grid(row=1, column=1, padx=6, pady=6, sticky="ew")

        right_button = ttk.Button(gimbal, text="Right")
        right_button.grid(row=1, column=2, padx=6, pady=6, sticky="ew")
        self._bind_hold_button(right_button, yaw_speed=MANUAL_YAW_SPEED, pitch_speed=0)

        down_button = ttk.Button(gimbal, text="Down")
        down_button.grid(row=2, column=1, padx=6, pady=6, sticky="ew")
        self._bind_hold_button(down_button, yaw_speed=0, pitch_speed=MANUAL_PITCH_SPEED)

        ttk.Button(gimbal, text="Center", command=self.center_gimbal).grid(row=3, column=1, padx=6, pady=6, sticky="ew")

        note = (
            "The preview stays live. Press Select Object when the target is visible. "
            "A rectangle selector will open on the current frame only at that moment. "
            "Tracking uses the same logic as rtsp_person_tracker.py."
        )
        ttk.Label(main, text=note, wraplength=820, justify="left").pack(anchor="w", pady=(12, 0))

    def _connect(self) -> None:
        try:
            self.cam = SIYISDK(server_ip=CAMERA_IP, port=CAMERA_PORT)
            if not self.cam.connect():
                self.status_var.set("Failed to connect to gimbal control.")
                return

            self.cam.requestFollowMode()
            self.cam.requestHardwareID()
            self.cam.requestGimbalAttitude()
            self.stream = SIYIRTSP(rtsp_url=self.rtsp_url, cam_name=CAMERA_NAME, debug=False)
            device_name = "CUDA" if self.use_cuda else "CPU"
            self.status_var.set(f"Connected to {CAMERA_IP}. Model loaded: {self.model_name} on {device_name}")
        except Exception as exc:
            self.status_var.set(f"Connection error: {exc}")

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
        self.target_var.set(
            f"Target: {seeded_target['class_name']} x={int(cx)} y={int(cy)} w={w} h={h} conf=1.00"
        )

    def select_object(self) -> None:
        frame = self.stream.getFrame() if self.stream is not None else None
        if frame is None:
            self.status_var.set("No live frame available yet.")
            return

        self.select_button.configure(state="disabled")
        self.track_button.configure(state="disabled")
        self.status_var.set("Select a rectangle on the current frame.")

        frame = frame.copy()
        roi = cv2.selectROI("Select target object", frame, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow("Select target object")
        x, y, w, h = [int(value) for value in roi]

        if w <= 0 or h <= 0:
            self.status_var.set("Selection canceled.")
            self.select_button.configure(state="normal")
            return

        crop = frame[y : y + h, x : x + w].copy()
        self.status_var.set("Running Moondream and matching a tracking model...")
        self.root.update_idletasks()

        try:
            detected_object, model_path, model_class = resolve_model_from_crop(crop, refresh_index=self.refresh_index)
        except Exception as exc:
            self.result_var.set(f"Selection failed: {exc}")
            self.status_var.set("Could not resolve a model from the selected object.")
            self.select_button.configure(state="normal")
            return

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
        self.result_var.set(
            f'Detected "{detected_object}". Selected model "{os.path.basename(model_path)}".'
        )
        self.select_button.configure(state="normal")
        self.track_button.configure(state="normal", text="Start Tracking")

        if AUTO_START_TRACKING:
            self.status_var.set("Model selected. Starting tracking with the selected ROI...")
            self.root.update_idletasks()
            self.toggle_tracking()
            return

        self.status_var.set("Model selected. Press Start Tracking to begin.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live RTSP object selection with Moondream-based model choice.")
    parser.add_argument("--rtsp-url", default=RTSP_URL, help="RTSP stream URL.")
    parser.add_argument("--refresh-index", action="store_true", help="Rebuild the models index before matching.")
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

    root = tk.Tk()
    app = LiveObjectSelectorApp(root, rtsp_url=args.rtsp_url, refresh_index=refresh_index)
    root.mainloop()


if __name__ == "__main__":
    main()
