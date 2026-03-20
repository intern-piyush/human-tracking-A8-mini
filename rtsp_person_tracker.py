import os
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk

import cv2
import torch


project_dir = os.path.dirname(os.path.realpath(__file__))
ultralytics_dir = os.path.join(project_dir, "Ultralytics")
sdk_dir = os.path.join(project_dir, "siyi_sdk")
os.makedirs(ultralytics_dir, exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = project_dir
sys.path.append(sdk_dir)

from siyi_sdk.siyi_sdk import SIYISDK
from siyi_sdk.stream import SIYIRTSP
from ultralytics import YOLO


RTSP_URL = "rtsp://192.168.144.25:8554/main.264"
CAMERA_IP = "192.168.144.25"
CAMERA_PORT = 37260
CAMERA_NAME = "A8 Mini"
MODEL_PATH = os.path.join(project_dir, "person.pt")

VIDEO_WIDTH = 640
VIDEO_HEIGHT = 360
DETECTION_WIDTH = 320

TRACK_INTERVAL_MS = 40
ATTITUDE_INTERVAL_MS = 300
MAX_YAW_SPEED = 25
MAX_PITCH_SPEED = 20
X_DEADBAND = 0.08
Y_DEADBAND = 0.10
YAW_GAIN = 70
PITCH_GAIN = 55
CONF_THRESHOLD = 0.35
TARGET_MATCH_DISTANCE = 140
DISPLAY_INTERVAL_MS = 40
INFERENCE_IDLE_MS = 0.01

YAW_SIGN = 1
PITCH_SIGN = 1


class HumanTrackerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("SIYI Human Tracker")
        self.root.geometry("760x680")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.cam = None
        self.stream = None
        self.video_image = None
        self.tracking_enabled = False
        self.current_yaw = 0.0
        self.current_pitch = 0.0
        self.last_target = None
        self.target_center = None
        self.last_detection_time = 0.0
        self.target_lock = threading.Lock()
        self.stop_event = threading.Event()

        self.status_var = tk.StringVar(value="Connecting...")
        self.angle_var = tk.StringVar(value="Yaw: 0.0 deg | Pitch: 0.0 deg")
        self.target_var = tk.StringVar(value="Target: none")

        self.model = YOLO(MODEL_PATH)
        self.model_name = os.path.basename(MODEL_PATH)
        self.use_cuda = torch.cuda.is_available()

        self._build_ui()
        self._connect()

        self.inference_thread = threading.Thread(target=self.inference_loop, daemon=True)
        self.inference_thread.start()

        self.root.after(DISPLAY_INTERVAL_MS, self.update_video)
        self.root.after(ATTITUDE_INTERVAL_MS, self.refresh_attitude)
        self.root.after(TRACK_INTERVAL_MS, self.track_loop)

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill="both", expand=True)

        self.video_label = ttk.Label(main, text="Waiting for RTSP video...", anchor="center")
        self.video_label.pack(fill="x")

        info_frame = ttk.Frame(main, padding=(0, 10, 0, 10))
        info_frame.pack(fill="x")

        ttk.Label(info_frame, textvariable=self.status_var).pack(anchor="w")
        ttk.Label(info_frame, textvariable=self.angle_var).pack(anchor="w")
        ttk.Label(info_frame, textvariable=self.target_var).pack(anchor="w")

        controls = ttk.LabelFrame(main, text="Tracking Controls", padding=12)
        controls.pack(fill="x")

        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(2, weight=1)

        self.track_button = ttk.Button(controls, text="Start Tracking", command=self.toggle_tracking)
        self.track_button.grid(row=0, column=0, padx=6, pady=6, sticky="ew")

        ttk.Button(controls, text="Center Gimbal", command=self.center_gimbal).grid(row=0, column=1, padx=6, pady=6, sticky="ew")
        ttk.Button(controls, text="Stop Motion", command=self.stop_motion).grid(row=0, column=2, padx=6, pady=6, sticky="ew")

        note = (
            f"Tracking uses {self.model_name} at {DETECTION_WIDTH}px inference width. "
            "If gimbal motion is reversed, change YAW_SIGN or PITCH_SIGN at the top of this file."
        )
        ttk.Label(main, text=note, wraplength=700, justify="left").pack(anchor="w", pady=(12, 0))

    def _connect(self) -> None:
        try:
            self.cam = SIYISDK(server_ip=CAMERA_IP, port=CAMERA_PORT)
            if not self.cam.connect():
                self.status_var.set("Failed to connect to gimbal control.")
                return

            self.cam.requestFollowMode()
            self.cam.requestHardwareID()
            self.cam.requestGimbalAttitude()
            self.stream = SIYIRTSP(rtsp_url=RTSP_URL, cam_name=CAMERA_NAME, debug=False)
            device_name = "CUDA" if self.use_cuda else "CPU"
            self.status_var.set(f"Connected to {CAMERA_IP}. Model loaded: {self.model_name} on {device_name}")
        except Exception as exc:
            self.status_var.set(f"Connection error: {exc}")

    def toggle_tracking(self) -> None:
        self.tracking_enabled = not self.tracking_enabled
        self.track_button.configure(text="Stop Tracking" if self.tracking_enabled else "Start Tracking")

        if self.tracking_enabled:
            self.status_var.set("Tracking enabled.")
        else:
            self.stop_motion()
            self.target_center = None
            self.last_target = None
            self.target_var.set("Target: none")
            self.status_var.set("Tracking stopped.")

    def center_gimbal(self) -> None:
        if self.cam is None:
            self.status_var.set("Gimbal control is not connected.")
            return

        self.cam.requestCenterGimbal()
        self.current_yaw = 0.0
        self.current_pitch = 0.0
        self.last_target = None
        self.target_center = None
        self.target_var.set("Target: none")
        self._update_angle_label()
        self.status_var.set("Center command sent.")

    def stop_motion(self) -> None:
        if self.cam is not None:
            self.cam.requestGimbalSpeed(0, 0)

    def refresh_attitude(self) -> None:
        if self.cam is not None:
            try:
                self.cam.requestGimbalAttitude()
                yaw, pitch, _ = self.cam.getAttitude()
                self.current_yaw = yaw
                self.current_pitch = pitch
                self._update_angle_label()
            except Exception:
                pass

        self.root.after(ATTITUDE_INTERVAL_MS, self.refresh_attitude)

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
                    "cx": cx,
                    "cy": cy,
                }
            )

        if not candidates:
            return None

        selected = self.select_target(candidates)
        self.target_center = (selected["cx"], selected["cy"])
        return selected

    def resize_for_detection(self, frame):
        height, width = frame.shape[:2]
        if width <= DETECTION_WIDTH:
            return frame

        scale = DETECTION_WIDTH / float(width)
        return cv2.resize(frame, (DETECTION_WIDTH, int(height * scale)))

    def compute_speed(self, normalized_error: float, deadband: float, gain: float, max_speed: int) -> int:
        if abs(normalized_error) < deadband:
            return 0

        speed = int(gain * normalized_error)
        if speed > max_speed:
            return max_speed
        if speed < -max_speed:
            return -max_speed
        return speed

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

    def class_name(self, cls_id: int) -> str:
        names = self.model.names
        if isinstance(names, dict):
            return names.get(cls_id, str(cls_id))
        if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
            return str(names[cls_id])
        return str(cls_id)

    def track_loop(self) -> None:
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
                self.target_var.set("Target: none")
                self.status_var.set("No model detection.")
            else:
                w = target["w"]
                h = target["h"]
                confidence = target["confidence"]
                center_x = target["cx"]
                center_y = target["cy"]
                frame_center_x = frame.shape[1] / 2.0
                frame_center_y = frame.shape[0] / 2.0

                x_error = (center_x - frame_center_x) / frame.shape[1]
                y_error = (center_y - frame_center_y) / frame.shape[0]

                yaw_speed = self.compute_speed(x_error, X_DEADBAND, YAW_GAIN, MAX_YAW_SPEED) * YAW_SIGN
                pitch_speed = self.compute_speed(y_error, Y_DEADBAND, PITCH_GAIN, MAX_PITCH_SPEED) * PITCH_SIGN

                self.cam.requestGimbalSpeed(yaw_speed, pitch_speed)
                self.target_var.set(
                    f"Target: x={int(center_x)} y={int(center_y)} w={w} h={h} conf={confidence:.2f}"
                )
                self.status_var.set(f"Tracking target. yaw_speed={yaw_speed}, pitch_speed={pitch_speed}")

        self.root.after(TRACK_INTERVAL_MS, self.track_loop)

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
            scale_x = VIDEO_WIDTH / frame.shape[1]
            scale_y = VIDEO_HEIGHT / frame.shape[0]

            x1 = int(x * scale_x)
            y1 = int(y * scale_y)
            x2 = int((x + bw) * scale_x)
            y2 = int((y + bh) * scale_y)

            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                output,
                f"{self.class_name(cls_id)} {confidence:.2f}",
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

    def update_video(self) -> None:
        frame = self.stream.getFrame() if self.stream is not None else None
        if frame is not None:
            display_frame = self.draw_overlay(frame)
            rgb_frame = display_frame
            ok, encoded = cv2.imencode(".ppm", rgb_frame)
            if ok:
                self.video_image = tk.PhotoImage(data=encoded.tobytes())
                self.video_label.configure(image=self.video_image, text="")

        self.root.after(DISPLAY_INTERVAL_MS, self.update_video)

    def _update_angle_label(self) -> None:
        self.angle_var.set(f"Yaw: {self.current_yaw:.1f} deg | Pitch: {self.current_pitch:.1f} deg")

    def on_close(self) -> None:
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

        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    HumanTrackerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
