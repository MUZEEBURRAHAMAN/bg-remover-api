import gc
import io
import time
from contextlib import asynccontextmanager

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image

MODEL_PATH = "/app/isnet-general-use.onnx"
INPUT_SIZE = 1024
MAX_DIMENSION = 1280

session: ort.InferenceSession | None = None
input_name: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global session, input_name
    print("Loading isnet-general-use ONNX model...")
    start = time.time()
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(MODEL_PATH, sess_options=so, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    print(f"Model loaded in {time.time() - start:.1f}s")
    yield


app = FastAPI(
    title="Background Remover API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess(img: Image.Image) -> np.ndarray:
    resized = img.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 1.0
    arr = arr.transpose(2, 0, 1)[None, ...].astype(np.float32)
    return arr


def mask_from_output(output: np.ndarray, size: tuple[int, int]) -> Image.Image:
    mask = output[0, 0]
    mask_min, mask_max = float(mask.min()), float(mask.max())
    mask = (mask - mask_min) / (mask_max - mask_min + 1e-8)
    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    return mask_img.resize(size, Image.BILINEAR)


@app.post("/remove")
async def remove_background(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    input_bytes = await file.read()
    if len(input_bytes) > 20 * 1024 * 1024:
        raise HTTPException(413, "Image too large (max 20MB)")

    try:
        img = Image.open(io.BytesIO(input_bytes)).convert("RGB")
        img.load()
    except Exception as e:
        raise HTTPException(400, f"Could not decode image: {e}")
    finally:
        del input_bytes

    if max(img.size) > MAX_DIMENSION:
        scale = MAX_DIMENSION / max(img.size)
        new_size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
        img = img.resize(new_size, Image.LANCZOS)

    work_size = img.size

    start = time.time()
    try:
        tensor = preprocess(img)
        outputs = session.run(None, {input_name: tensor})
        mask = mask_from_output(outputs[0], work_size)

        rgba = img.convert("RGBA")
        rgba.putalpha(mask)

        buf = io.BytesIO()
        rgba.save(buf, format="PNG")
        output_bytes = buf.getvalue()
    finally:
        del img
        gc.collect()
    elapsed = time.time() - start

    return Response(
        content=output_bytes,
        media_type="image/png",
        headers={
            "X-Processing-Time": f"{elapsed:.3f}",
            "Content-Disposition": "inline; filename=result.png",
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model": "isnet-general-use", "ready": session is not None}
