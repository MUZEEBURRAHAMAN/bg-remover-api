import io
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from rembg import remove, new_session

session = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global session
    print("Loading BiRefNet model...")
    start = time.time()
    session = new_session("birefnet-general")
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
    allow_methods=["POST"],
    allow_headers=["*"],
)


@app.post("/remove")
async def remove_background(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    input_bytes = await file.read()
    if len(input_bytes) > 20 * 1024 * 1024:
        raise HTTPException(413, "Image too large (max 20MB)")

    start = time.time()
    output_bytes = remove(
        input_bytes,
        session=session,
        post_process_mask=True,
    )
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
    return {"status": "ok", "model": "birefnet-general", "ready": session is not None}
