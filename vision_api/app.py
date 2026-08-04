"""PP-OCRv6 Tiny HTTP API for the SpaceMIT K3 / Milk-V Jupiter 2."""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile


ROOT = Path(os.environ.get("VISION_MODEL_DIR", "/root/vision-api/models/ppocrv6-tiny"))
DET_DIR = ROOT / "PP-OCRv6_tiny_det_onnx"
REC_DIR = ROOT / "PP-OCRv6_tiny_rec_onnx"
MAX_BYTES = 16 * 1024 * 1024
MAX_PIXELS = 40_000_000

# Which recognizer runs. The detector is always PP-OCRv6 Tiny on ORT CPU.
#   native-ppocrv6 - PP-OCRv6 Tiny recognizer as a hand-written RVV FP32 engine
#                    on the A100 AI harts (default). Bit-equivalent to the ONNX
#                    path and 2-3x faster; see BARE_IME_OCR_PROGRESS.md.
#   onnx-ppocrv6   - the same model through ONNX Runtime CPU. Rollback target.
#   native-rosetta - native ResNet34+CTC. Correct, but a 36-character
#                    lowercase-only dictionary makes it unusable for documents.
RECOGNIZER = os.environ.get("VISION_RECOGNIZER", "native-ppocrv6").lower()

# A decoded character whose runner-up holds at least this much probability mass
# is reported alongside its alternative. Measured on the FinScan comma/decimal
# repro: digit positions put 1e-5 or less on their runner-up, correctly-read
# thousands separators 0.9-2.5%, and actually-misread separators 2-49%. 1% sits
# below every observed ambiguous separator and far above ordinary characters.
ALT_THRESHOLD = float(os.environ.get("VISION_ALT_THRESHOLD", "0.01"))
# Per-character runner-ups are opt-in per request (?alternatives=true). On dense
# prose this model is genuinely uncertain nearly everywhere -- 1641 alternatives
# across 97 lines of a paper page -- which is honest but doubles the payload for
# callers who only want text. `min_char_score` is always present: it is one float
# and it is the whole point, since a mean cannot show a single bad character.
ALT_DEFAULT = os.environ.get("VISION_ALTERNATIVES", "0").lower() in ("1", "true", "yes")


def _session(path: Path) -> tuple[ort.InferenceSession, str]:
    # SpaceMIT EP 2.0.5 currently segfaults during session creation for both
    # PP-OCRv6 graphs. Keep CPU as the reliable baseline until the vendor EP is
    # fixed; opt-in explicitly so a service restart cannot silently select it.
    provider = os.environ.get("VISION_PROVIDER", "cpu").lower()
    if provider == "spacemit":
        import spacemit_ort  # noqa: F401 - registers the plugin with ORT

        providers = ["SpaceMITExecutionProvider", "CPUExecutionProvider"]
    elif provider == "cpu":
        providers = ["CPUExecutionProvider"]
    else:
        raise RuntimeError(f"unsupported VISION_PROVIDER={provider!r}")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = int(os.environ.get("VISION_ORT_THREADS", "8"))
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=opts, providers=providers)
    return session, provider


def _order_box(points: np.ndarray) -> np.ndarray:
    points = points.astype(np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    return np.array(
        [points[np.argmin(sums)], points[np.argmin(diffs)],
         points[np.argmax(sums)], points[np.argmax(diffs)]],
        dtype=np.float32,
    )


class OCRBackend:
    def __init__(self) -> None:
        self.det, self.provider = _session(DET_DIR / "inference.onnx")
        self.recognizer = RECOGNIZER
        self.rec = None
        self.native = None
        if RECOGNIZER.startswith("native-"):
            from ocr_native import OCRNative, default_paths

            lib_path, blob_path, label = default_paths(RECOGNIZER)
            self.native = OCRNative(
                lib_path, blob_path, harts=int(os.environ.get("OCR_HARTS", "8"))
            )
            if RECOGNIZER == "native-rosetta":
                # Rosetta's own dictionary: blank + 36 ic15 characters, and
                # deliberately NOT PP-OCRv6's trailing-space convention.
                self.characters = [""] + list("0123456789abcdefghijklmnopqrstuvwxyz")
            else:
                with open(REC_DIR / "inference.yml", encoding="utf-8") as handle:
                    config = yaml.safe_load(handle)
                chars = list(config["PostProcess"]["character_dict"])
                self.characters = [""] + chars + [" "]
            if self.native.n_classes != len(self.characters):
                raise RuntimeError(
                    f"recognizer has {self.native.n_classes} classes but dictionary "
                    f"has {len(self.characters)} entries"
                )
            self.rec_model = label
            self.rec_provider = f"spacemit-ai-harts x{self.native.harts}"
        elif RECOGNIZER == "onnx-ppocrv6":
            self.rec, rec_provider = _session(REC_DIR / "inference.onnx")
            if rec_provider != self.provider:
                raise RuntimeError("detector and recognizer providers differ")
            with open(REC_DIR / "inference.yml", encoding="utf-8") as handle:
                config = yaml.safe_load(handle)
            # Paddle CTCLabelDecode reserves index zero for blank and appends space.
            chars = list(config["PostProcess"]["character_dict"])
            self.characters = [""] + chars + [" "]
            output_classes = self.rec.get_outputs()[0].shape[-1]
            if isinstance(output_classes, int) and output_classes != len(self.characters):
                raise RuntimeError(
                    f"recognizer has {output_classes} classes but dictionary has "
                    f"{len(self.characters)} entries"
                )
            self.rec_model = "PP-OCRv6_tiny"
            self.rec_provider = self.provider
        else:
            raise RuntimeError(f"unsupported VISION_RECOGNIZER={RECOGNIZER!r}")

    @staticmethod
    def _det_input(image: np.ndarray) -> tuple[np.ndarray, float, float]:
        height, width = image.shape[:2]
        scale = min(1.0, 736.0 / max(height, width))
        resized_h = max(32, int(round(height * scale / 32.0) * 32))
        resized_w = max(32, int(round(width * scale / 32.0) * 32))
        resized = cv2.resize(image, (resized_w, resized_h))
        x = resized.astype(np.float32) / 255.0
        x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
            [0.229, 0.224, 0.225], np.float32
        )
        x = np.transpose(x, (2, 0, 1))[None, ...]
        return np.ascontiguousarray(x), height / resized_h, width / resized_w

    @staticmethod
    def _boxes(probability: np.ndarray, ratio_h: float, ratio_w: float) -> list[np.ndarray]:
        bitmap = (probability > 0.2).astype(np.uint8) * 255
        contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        found: list[tuple[float, np.ndarray]] = []
        for contour in contours[:3000]:
            if cv2.contourArea(contour) < 4:
                continue
            mask = np.zeros_like(probability, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 1, -1)
            score = float(cv2.mean(probability, mask=mask)[0])
            if score < 0.4:
                continue
            rect = cv2.minAreaRect(contour)
            if min(rect[1]) < 3:
                continue
            # Lightweight DB-style expansion. Exact polygon unclipping can be
            # added later; this approximation is deterministic and dependency-free.
            expanded = (rect[0], (rect[1][0] * 1.4, rect[1][1] * 1.4), rect[2])
            box = _order_box(cv2.boxPoints(expanded))
            box[:, 0] *= ratio_w
            box[:, 1] *= ratio_h
            found.append((float(box[:, 1].min()), box))
        found.sort(key=lambda item: (round(item[0] / 10), float(item[1][:, 0].min())))
        return [box for _, box in found]

    @staticmethod
    def _crop(image: np.ndarray, box: np.ndarray) -> np.ndarray:
        box = _order_box(box)
        width = max(1, int(max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[2] - box[3]))))
        height = max(1, int(max(np.linalg.norm(box[0] - box[3]), np.linalg.norm(box[1] - box[2]))))
        dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float32)
        matrix = cv2.getPerspectiveTransform(box, dst)
        crop = cv2.warpPerspective(image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
        if height / max(width, 1) >= 1.5:
            crop = np.rot90(crop).copy()
        return crop

    @staticmethod
    def _rec_batch(crops: list[np.ndarray], target_h: int = 48) -> np.ndarray:
        widths = [max(8, min(3200, int(np.ceil(target_h * c.shape[1] / c.shape[0])))) for c in crops]
        target_w = max(16, int(np.ceil(max(widths) / 8.0) * 8))
        batch = np.zeros((len(crops), 3, target_h, target_w), np.float32)
        for i, (crop, width) in enumerate(zip(crops, widths)):
            resized = cv2.resize(crop, (width, target_h)).astype(np.float32)
            resized = resized / 127.5 - 1.0
            batch[i, :, :, :width] = np.transpose(resized, (2, 0, 1))
        return np.ascontiguousarray(batch)

    def _decode(self, output: np.ndarray) -> list[tuple[str, float, float, list[dict]]]:
        """CTC greedy decode, plus the per-character uncertainty the model already
        computed and the line score throws away.

        `score` stays exactly what it always was -- the mean top-1 probability
        over kept timesteps -- so existing consumers are unaffected. But a mean
        is the wrong statistic for spotting a single bad character: a five-digit
        figure whose separator the model genuinely split 0.51/0.49 still averages
        to 0.90+, which is why a misread can look like a confident read. So also
        report the minimum per-character probability and, for any character whose
        runner-up holds a non-trivial share of the mass, that runner-up.
        """
        decoded = []
        for sequence in output:
            ids = np.argmax(sequence, axis=1)
            scores = np.max(sequence, axis=1)
            text: list[str] = []
            kept: list[float] = []
            alternatives: list[dict] = []
            previous = -1
            for step, (index, score) in enumerate(zip(ids.tolist(), scores.tolist())):
                if index != 0 and index != previous and index < len(self.characters):
                    position = len(text)
                    text.append(self.characters[index])
                    kept.append(float(score))
                    # Partial sort of one row only -- kept timesteps are a small
                    # fraction of the sequence, so this is far cheaper than a
                    # top-2 over the whole (steps, classes) array.
                    row = sequence[step]
                    runner = int(np.argpartition(row, -2)[-2:][0])
                    if runner == index:
                        runner = int(np.argpartition(row, -2)[-2:][1])
                    p_alt = float(row[runner])
                    if p_alt >= ALT_THRESHOLD and runner < len(self.characters):
                        alternatives.append({
                            "index": position,
                            "char": self.characters[index],
                            "alt": self.characters[runner],
                            "p": round(float(score), 6),
                            "p_alt": round(p_alt, 6),
                        })
                previous = index
            decoded.append((
                "".join(text),
                float(np.mean(kept)) if kept else 0.0,
                float(np.min(kept)) if kept else 0.0,
                alternatives,
            ))
        return decoded

    def infer(self, image: np.ndarray, alternatives: bool = False) -> dict:
        started = time.perf_counter()
        det_input, ratio_h, ratio_w = self._det_input(image)
        t0 = time.perf_counter()
        probability = self.det.run(None, {self.det.get_inputs()[0].name: det_input})[0][0, 0]
        t1 = time.perf_counter()
        boxes = self._boxes(probability, ratio_h, ratio_w)
        crops = [self._crop(image, box) for box in boxes]
        t2 = time.perf_counter()
        # Group similarly shaped crops so one unusually wide line does not
        # force every other item in its batch to use the same padded width.
        order = sorted(
            range(len(crops)),
            key=lambda index: crops[index].shape[1] / max(crops[index].shape[0], 1),
        )
        results: list[tuple[str, float, float, list]] = [("", 0.0, 0.0, [])] * len(crops)
        for offset in range(0, len(order), 16):
            indices = order[offset:offset + 16]
            batch = [crops[index] for index in indices]
            if self.native is not None:
                # Deliberately reuse _rec_batch unchanged, zero padding and all:
                # feeding the native engine byte-identical input to what ONNX
                # Runtime would see makes this a drop-in replacement whose only
                # difference is speed. The batch dimension is independent, so
                # running the rows one at a time is the same arithmetic.
                rec_input = self._rec_batch(batch, self.native.height)
                for index, probs in zip(indices, self.native.infer_rows(rec_input)):
                    results[index] = self._decode(probs[None, ...])[0]
            else:
                rec_input = self._rec_batch(batch)
                output = self.rec.run(None, {self.rec.get_inputs()[0].name: rec_input})[0]
                for index, result in zip(indices, self._decode(output)):
                    results[index] = result
        t3 = time.perf_counter()
        lines = []
        want_alt = alternatives or ALT_DEFAULT
        for box, (text, score, min_score, alts) in zip(boxes, results):
            if text and score >= 0.2:
                line = {
                    "text": text,
                    "score": round(score, 6),
                    "box": np.rint(box).astype(int).tolist(),
                    # Additive: the mean `score` above cannot expose a single
                    # contested character. These two can.
                    "min_char_score": round(min_score, 6),
                }
                if want_alt and alts:
                    line["alternatives"] = alts
                lines.append(line)
        return {
            "model": self.rec_model,
            "detector": "PP-OCRv6_tiny_det",
            "recognizer": self.recognizer,
            "provider": self.provider,
            "rec_provider": self.rec_provider,
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "lines": lines,
            "text": "\n".join(line["text"] for line in lines),
            "timing_ms": {
                "detect": round((t1 - t0) * 1000, 3),
                "postprocess": round((t2 - t1) * 1000, 3),
                "recognize": round((t3 - t2) * 1000, 3),
                "total": round((t3 - started) * 1000, 3),
            },
        }


backend: OCRBackend | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global backend
    backend = OCRBackend()
    yield
    backend = None


app = FastAPI(title="Jupiter 2 Vision API", version="0.1.0", lifespan=lifespan)


@app.get("/healthz")
def health() -> dict:
    if backend is None:
        raise HTTPException(503, "OCR backend is not initialized")
    return {
        "status": "ok",
        "model": backend.rec_model,
        "detector": "PP-OCRv6_tiny_det",
        "recognizer": backend.recognizer,
        "provider": backend.provider,
        "rec_provider": backend.rec_provider,
        "accelerated_provider_status": "known-crash" if backend.provider == "cpu" else "enabled",
    }


@app.post("/v1/ocr")
async def ocr(image: UploadFile = File(...), alternatives: bool = False) -> dict:
    if backend is None:
        raise HTTPException(503, "OCR backend is not initialized")
    data = await image.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "image exceeds 16 MiB")
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise HTTPException(400, "invalid JPEG or PNG image")
    if decoded.shape[0] * decoded.shape[1] > MAX_PIXELS:
        raise HTTPException(413, "decoded image exceeds 40 megapixels")
    try:
        return backend.infer(decoded, alternatives=alternatives)
    except Exception as exc:
        raise HTTPException(500, f"inference failed: {type(exc).__name__}") from exc
