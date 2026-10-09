"""Checks required by the UAV vision specification."""

from __future__ import annotations

import json
import socket
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from capture.rtsp_source import RtspSource
from capture.video_source import VideoFileSource
from detection.class_catalog import classify_weights
from detection.device_select import resolve_device
from detection.model_manager import ModelError
from detection.yolo_v4_detector import Detection
from detection.yolo_v8_detector import YoloV8Detector
from integration.command_center import CommandCenterClient, build_event, validate_event
from ptz.virtual_ptz import VirtualPtz
from tracking.engine import PipelineConfig, VisionEngine
from tracking.multi_tracker import BotSortTracker, ByteTracker
from training.dataset import import_images, load_measured_metrics, prepare_split
from training.video_eval import evaluate_video


class FakeBoxes:
    def __init__(self, xyxy, conf, cls) -> None:
        self.xyxy = np.array(xyxy, dtype=float)
        self.conf = np.array(conf, dtype=float)
        self.cls = np.array(cls, dtype=float)


class FakeResult:
    def __init__(self, boxes) -> None:
        self.boxes = boxes


class FakeModel:
    def __init__(self, names, fail_cuda=False, classes=(0, 1, 2)) -> None:
        self.names = names
        self.fail_cuda = fail_cuda
        self.devices = []
        self.classes = classes

    def predict(self, frame, **kwargs):
        device = kwargs.get("device")
        self.devices.append(device)
        if self.fail_cuda and device != "cpu":
            raise RuntimeError("CUDA device is not available")
        return [FakeResult(FakeBoxes([[8, 12, 40, 50], [70, 80, 120, 140], [200, 30, 260, 80]], [0.91, 0.88, 0.8], list(self.classes)))]


def _moving_box(index: int, x0: int = 30) -> Detection:
    x = x0 + index * 3
    return Detection(0, "drone", 0.95, (x, 80, 36, 36))


class ModelAndDeviceTests(unittest.TestCase):
    def test_missing_weights_are_rejected(self) -> None:
        detector = YoloV8Detector(loader=lambda path: FakeModel({0: "drone"}))
        with self.assertRaises(ModelError):
            detector.load(str(ROOT / "model" / "missing-uav.pt"))

    def test_custom_weights_keep_only_known_classes(self) -> None:
        path = ROOT / "model" / "fixture-drone.pt"
        path.write_bytes(b"fixture")
        try:
            detector = YoloV8Detector(loader=lambda _path: FakeModel({0: "drone", 1: "person", 2: "bird"}))
            detector.load(str(path), "CPU")
            self.assertEqual(detector.kind, "uav_custom")
            self.assertEqual(detector.warning_code, "")
            frame = np.zeros((120, 320, 3), dtype=np.uint8)
            found = detector.detect(frame, 0.25, 0.5, 640)
            names = [item.class_name for item in found]
            self.assertEqual(names, ["drone", "bird"])
        finally:
            path.unlink(missing_ok=True)

    def test_coco_weights_are_not_a_uav_model(self) -> None:
        names = {
            0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
            5: "bus", 6: "train", 7: "truck", 8: "boat", 14: "bird", 15: "cat", 16: "dog",
        }
        kind, warning = classify_weights(names, "yolov8n.pt")
        self.assertEqual(kind, "coco_general")
        self.assertEqual(warning, "coco_not_uav")
        path = ROOT / "model" / "yolov8n.pt"
        path.write_bytes(b"not-a-real-checkpoint")
        try:
            detector = YoloV8Detector(loader=lambda _path: FakeModel(names, classes=(4, 0, 14)))
            detector.load(str(path), "CPU")
            self.assertIn("не модель БПЛА", detector.display_name)
            found = detector.detect(np.zeros((90, 300, 3), np.uint8), 0.2, 0.5, 640)
            self.assertEqual([item.class_name for item in found], ["aircraft", "bird"])
        finally:
            path.unlink(missing_ok=True)

    def test_cuda_request_falls_back_to_cpu(self) -> None:
        import detection.device_select as device_select

        original = device_select.cuda_available
        device_select.cuda_available = lambda: False
        try:
            device, label = resolve_device("CUDA")
            self.assertEqual(device, "cpu")
            self.assertEqual(label, "CPU")
        finally:
            device_select.cuda_available = original
        path = ROOT / "model" / "fixture-cuda.pt"
        path.write_bytes(b"fixture")
        try:
            model = FakeModel({0: "drone"}, fail_cuda=True)
            detector = YoloV8Detector(loader=lambda _path: model)
            detector.load(str(path), "CPU")
            detector.device = 0
            detector.backend_label = "CUDA"
            found = detector.detect(np.zeros((40, 40, 3), np.uint8), 0.2, 0.5, 640)
            self.assertEqual(detector.device, "cpu")
            self.assertEqual(detector.backend_label, "CPU (fallback)")
            self.assertTrue(found)
            self.assertEqual(model.devices[-1], "cpu")
        finally:
            path.unlink(missing_ok=True)


class SourceTests(unittest.TestCase):
    def test_video_file_opens(self) -> None:
        from simulation.scenarios import write_scene

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "demo_uav_tracking.mp4"
            write_scene(path, "uav_tracking", seconds=0.4, fps=10, size=(320, 180))
            source = VideoFileSource(str(path))
            self.assertTrue(source.cap.isOpened())
            ok, frame = source.read()
            source.release()
            self.assertTrue(ok)
            self.assertIsNotNone(frame)
            self.assertEqual(source.content_index, 0)

    def test_empty_rtsp_url_is_rejected(self) -> None:
        with self.assertRaises(ConnectionError):
            RtspSource("   ")

    def test_rtsp_open_failure_is_reported(self) -> None:
        class Closed:
            def set(self, *_args) -> None:
                return None

            def isOpened(self) -> bool:
                return False

            def release(self) -> None:
                return None

        original = cv2.VideoCapture
        cv2.VideoCapture = lambda *_args, **_kwargs: Closed()
        try:
            with self.assertRaises(ConnectionError):
                RtspSource("rtsp://127.0.0.1:9/absent", timeout_ms=200)
        finally:
            cv2.VideoCapture = original


class TrackTests(unittest.TestCase):
    def test_track_id_survives_a_short_gap(self) -> None:
        tracker = ByteTracker()
        now = 0.0
        last_id = None
        for index in range(6):
            tracks = tracker.update([_moving_box(index)], now, 10, 1.5, 0.0, 0.4, 0.1, 0.3)
            now += 0.1
            last_id = tracks[0].track_id
            if index == 5:
                self.assertEqual(tracks[0].status, "TRACKING")
        for _ in range(3):
            tracks = tracker.update([], now, 10, 1.5, 0.0, 0.4, 0.1, 0.3)
            now += 0.1
        self.assertEqual(tracks[0].status, "OCCLUDED")
        self.assertEqual(tracks[0].track_id, last_id)
        tracks = tracker.update([_moving_box(9)], now, 10, 1.5, 0.0, 0.4, 0.1, 0.3)
        self.assertEqual(tracks[0].track_id, last_id)
        self.assertEqual(tracks[0].status, "REACQUIRED")

    def test_botsort_keeps_identity_and_hides_kmh_without_calibration(self) -> None:
        tracker = BotSortTracker()
        now = 0.0
        for index in range(4):
            tracks = tracker.update([_moving_box(index)], now, 10, 1.5, 0.0, 0.4, 0.1, 0.3)
            now += 0.1
        self.assertIsNone(tracks[0].speed_kmh)
        self.assertGreater(tracks[0].speed_px_s, 0.0)
        tracks = tracker.update([_moving_box(4)], now, 10, 1.5, 0.05, 0.4, 0.1, 0.3)
        self.assertIsNotNone(tracks[0].speed_kmh)
        self.assertGreater(tracks[0].speed_kmh, 0.0)

    def test_demo_pipeline_is_marked_simulated(self) -> None:
        from simulation.scenarios import scene_by_key

        scene = scene_by_key("multiple_drones")
        engine = VisionEngine()
        config = PipelineConfig(
            mission="uav",
            demo=True,
            demo_scene="multiple_drones",
            algorithm="BYTETRACK",
            confidence=0.4,
            lost_timeout=1.5,
            source_fps=25,
            overlays={"boxes": True, "ids": True, "status": True, "speed": True, "fps": True},
        )
        view = None
        info = {}
        for index in range(6):
            frame = scene.frame(index, 640, 360, 25)
            config.frame_index = index
            view, info = engine.step(frame, config)
        self.assertTrue(info["demo"])
        self.assertGreaterEqual(info["detected_count"], 2)
        self.assertEqual(len({track["id"] for track in info["tracks"]}), len(info["tracks"]))
        self.assertTrue(any(track["status"] == "TRACKING" for track in info["tracks"]))
        banner = view[0:40, :, 2]
        self.assertGreater(int(banner.max()), 140)
        self.assertEqual(info["banner"], "ДЕМО / СИНТЕТИЧЕСКИЕ ДАННЫЕ")
        self.assertEqual(info["unit_kmh"], "км/ч")
        self.assertIsNone(info["speed_kmh"])
        config.language = "en"
        config.frame_index = 6
        _view, info = engine.step(scene.frame(6, 640, 360, 25), config)
        self.assertEqual(info["banner"], "DEMO / SIMULATED DATA")
        self.assertEqual(info["unit_kmh"], "km/h")


class EventTests(unittest.TestCase):
    def test_json_event_has_no_gps(self) -> None:
        event = build_event(
            sensor_id="uav-vision-01",
            track_id=7,
            class_name="drone",
            confidence=0.82,
            box=(10, 20, 30, 40),
            frame_w=100,
            frame_h=200,
            source="demo_uav_tracking.mp4",
            status="TRACKING",
            speed_px_s=12.5,
        )
        validate_event(event)
        self.assertTrue(event["cv"])
        self.assertEqual(event["id"], "uav-vision-01-7")
        self.assertEqual(event["bbox_norm"]["w"], 0.3)
        self.assertNotIn("lat", event)
        self.assertNotIn("lon", event)
        with self.assertRaises(ValueError):
            validate_event({**event, "lat": 47.0})

    def test_udp_and_jsonl_export(self) -> None:
        event = build_event(
            sensor_id="sensor-a",
            track_id=1,
            class_name="bird",
            confidence=0.5,
            box=(0, 0, 10, 10),
            frame_w=100,
            frame_h=100,
            source="file",
            status="DETECTED",
        )
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", 0))
        host, port = sock.getsockname()
        sock.settimeout(2)
        with tempfile.TemporaryDirectory() as folder:
            log_path = Path(folder) / "events.jsonl"
            count = CommandCenterClient().send_events([event], host, port, log_path, min_interval=0)
            payload, _addr = sock.recvfrom(65535)
            written = log_path.read_text(encoding="utf-8")
        sock.close()
        self.assertEqual(count, 1)
        received = json.loads(payload.decode("utf-8"))
        validate_event(received)
        self.assertEqual(received["class"], "bird")
        self.assertIn('"track_id": 1', written)


class TrainingAndPtzTests(unittest.TestCase):
    def test_split_and_metrics_are_not_invented(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            incoming = root / "incoming"
            image = incoming / "images"
            image.mkdir(parents=True)
            for name in ("a.jpg", "b.jpg", "c.jpg", "d.jpg"):
                cv2.imwrite(str(image / name), np.zeros((16, 16, 3), np.uint8))
            (incoming / "labels").mkdir()
            (incoming / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            (incoming / "classes.txt").write_text("drone\n", encoding="utf-8")
            summary = prepare_split(incoming, root / "dataset", 0.25, seed=2)
            self.assertEqual(summary["images"], 4)
            self.assertEqual(summary["classes"], ["drone"])
            self.assertTrue((root / "dataset" / "data.yaml").is_file())
            self.assertIsNone(load_measured_metrics(root / "missing-run"))
            run = root / "run"
            run.mkdir()
            (run / "results.csv").write_text(
                "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n"
                "1,0.25,0.50,0.40,0.20\n",
                encoding="utf-8",
            )
            metrics = load_measured_metrics(run)
            self.assertEqual(metrics["precision"], 0.25)
            self.assertEqual(metrics["recall"], 0.50)
            self.assertEqual(metrics["map50"], 0.40)
            self.assertEqual(metrics["map50_95"], 0.20)
            self.assertIsNone(metrics["confusion_matrix"])

    def test_video_eval_reports_counts_without_false_accuracy(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "clip.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5, (64, 48))
            for _ in range(3):
                writer.write(np.zeros((48, 64, 3), np.uint8))
            writer.release()
            report = evaluate_video(str(path), lambda _frame: [Detection(0, "drone", 0.9, (1, 1, 8, 8))])
            self.assertEqual(report["detections"], 3)
            self.assertNotIn("measured", report)
            self.assertNotIn("map50", report)

    def test_virtual_ptz_stops_inside_limits(self) -> None:
        ptz = VirtualPtz()
        ptz.set_limits(-20, 20, -10, 10)
        ptz.calibrate(0, 0, 40)
        moved = ptz.slew_toward(1.0, 1.0, 0.5)
        self.assertLessEqual(moved.pan_deg, 20)
        self.assertLessEqual(moved.tilt_deg, 10)
        stopped = ptz.stop()
        self.assertTrue(stopped.stopped)
        self.assertEqual(stopped.rate_pan_dps, 0.0)
        held = ptz.slew_toward(-1.0, -1.0, 0.5)
        self.assertEqual(held.pan_deg, stopped.pan_deg)
        self.assertEqual(held.tilt_deg, stopped.tilt_deg)


class WeightLibraryTests(unittest.TestCase):
    def test_catalog_checks_links_and_stores_files(self) -> None:
        import hashlib
        import json

        from library.weights import download_all

        payload = b"weight-bytes-not-html"
        digest = hashlib.sha256(payload).hexdigest()
        calls = []

        class _Body:
            def __init__(self) -> None:
                self.status = 200
                self.headers = {"Content-Type": "application/octet-stream", "Content-Length": str(len(payload))}
                self._pos = 0

            def geturl(self):
                return "https://example.com/best.pt"

            def read(self, size=-1):
                if self._pos >= len(payload):
                    return b""
                take = len(payload) - self._pos if size is None or size < 0 else size
                chunk = payload[self._pos:self._pos + take]
                self._pos += len(chunk)
                return chunk

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        def opener(request, timeout=120):
            calls.append(request.full_url)
            return _Body()

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            catalog = root / "catalog.json"
            catalog.write_text(
                json.dumps(
                    [
                        {
                            "id": "uav-test",
                            "title": "Test UAV",
                            "role": "uav",
                            "url": "https://example.com/best.pt",
                            "filename": "best.pt",
                            "sha256": digest,
                            "license": "test",
                            "note": "",
                        },
                        {
                            "id": "vehicle-missing",
                            "title": "Missing vehicle",
                            "role": "vehicle",
                            "url": "",
                            "filename": "custom.weights",
                            "sha256": "",
                            "license": "",
                            "note": "NOT_PUBLIC",
                        },
                    ]
                ),
                encoding="utf-8",
            )
            rows = download_all(
                catalog,
                root / "weights.sqlite",
                root / "files",
                opener=opener,
                read_names=lambda _path: ["Drone", "AirPlane", "Helicopter"],
            )
            self.assertEqual(rows[0]["status"], "downloaded")
            self.assertEqual(rows[0]["usable"], 1)
            self.assertEqual(json.loads(rows[0]["kept_classes"]), ["aircraft", "drone"])
            self.assertTrue(Path(rows[0]["local_path"]).is_file())
            self.assertEqual(rows[1]["status"], "not_public")
            self.assertEqual(len(calls), 2)
            again = download_all(
                catalog,
                root / "weights.sqlite",
                root / "files",
                opener=opener,
                read_names=lambda _path: ["Drone", "AirPlane"],
            )
            self.assertEqual(again[0]["status"], "already")
            self.assertEqual(len(calls), 3)

    def test_html_page_and_bad_hash_are_rejected(self) -> None:
        import json

        from library.weights import download_all

        def opener_for(data, url):
            class _Body:
                def __init__(self) -> None:
                    self.status = 200
                    self.headers = {"Content-Type": "text/html" if data.startswith(b"<") else "application/octet-stream"}
                    self._data = data
                    self._pos = 0
                    self._url = url

                def geturl(self):
                    return self._url

                def read(self, size=-1):
                    if self._pos >= len(self._data):
                        return b""
                    take = len(self._data) - self._pos if size is None or size < 0 else size
                    chunk = self._data[self._pos:self._pos + take]
                    self._pos += len(chunk)
                    return chunk

                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    return False

            def opener(request, timeout=120):
                return _Body()

            return opener

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            html_catalog = root / "html.json"
            html_catalog.write_text(
                json.dumps(
                    [{
                        "id": "page",
                        "title": "Page",
                        "role": "uav",
                        "url": "https://example.com/page",
                        "filename": "best.pt",
                        "sha256": "",
                    }]
                ),
                encoding="utf-8",
            )
            rows = download_all(html_catalog, root / "a.sqlite", root / "a", opener=opener_for(b"<!DOCTYPE html>", "https://example.com/page"), read_names=lambda _path: ["drone"])
            self.assertEqual(rows[0]["status"], "failed")
            self.assertEqual(list((root / "a").rglob("*.pt")), [])
            bad_catalog = root / "bad.json"
            bad_catalog.write_text(
                json.dumps(
                    [{
                        "id": "bad-hash",
                        "title": "Bad",
                        "role": "uav",
                        "url": "https://example.com/file.pt",
                        "filename": "best.pt",
                        "sha256": "0" * 64,
                    }]
                ),
                encoding="utf-8",
            )
            rows = download_all(
                bad_catalog,
                root / "b.sqlite",
                root / "b",
                opener=opener_for(b"weight-bytes-not-html", "https://example.com/file.pt"),
                read_names=lambda _path: ["drone"],
            )
            self.assertEqual(rows[0]["status"], "failed")
            self.assertIn("sha256", rows[0]["detail"])
            self.assertEqual(list((root / "b").rglob("*.pt")), [])


class ImportTests(unittest.TestCase):
    def test_image_import(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "src"
            source.mkdir()
            cv2.imwrite(str(source / "one.png"), np.zeros((8, 8, 3), np.uint8))
            count = import_images([str(source / "one.png")], root / "incoming")
            self.assertEqual(count, 1)
            self.assertTrue((root / "incoming" / "images" / "one.png").is_file())


if __name__ == "__main__":
    unittest.main()
