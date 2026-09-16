import gc
import io
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image
from rembg import remove, new_session

session = None
MAX_DIMENSION = 1280


@asynccontextmanager
async def lifespan(app: FastAPI):
    global session
    print("Loading isnet-general-use model...")
    start = time.time()
    session = new_session("isnet-general-use")
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


def downscale_if_needed(input_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(input_bytes))
    img.load()
    width, height = img.size

    if max(width, height) <= MAX_DIMENSION:
        img.close()
        return input_bytes

    scale = MAX_DIMENSION / max(width, height)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = img.resize(new_size, Image.LANCZOS)
    img.close()

    buf = io.BytesIO()
    resized.save(buf, format="PNG")
    resized.close()
    return buf.getvalue()


@app.post("/remove")
async def remove_background(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    input_bytes = await file.read()
    if len(input_bytes) > 20 * 1024 * 1024:
        raise HTTPException(413, "Image too large (max 20MB)")

    try:
        processed_bytes = downscale_if_needed(input_bytes)
    except Exception as e:
        raise HTTPException(400, f"Could not decode image: {e}")

    del input_bytes

    start = time.time()
    try:
        output_bytes = remove(
            processed_bytes,
            session=session,
            post_process_mask=False,
        )
    finally:
        del processed_bytes
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
