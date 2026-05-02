"""
Self-hosted background removal microservice.
Uses briaai/RMBG-1.4 (open-source, runs fully locally).

Deploy on: Railway / Render / Fly.io / any VPS with Python

Install:
  pip install fastapi uvicorn torch torchvision pillow transformers huggingface_hub python-multipart

Run:
  uvicorn main:app --host 0.0.0.0 --port 8080
"""
import io
import os
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Background Removal Service")

# ── Load RMBG-1.4 model once at startup ───────────────────────────────────────
model = None
transform_image = None

@app.on_event("startup")
async def load_model():
    global model, transform_image
    logger.info("Loading RMBG-1.4 model...")

    from transformers import AutoModelForImageSegmentation
    model = AutoModelForImageSegmentation.from_pretrained(
        "briaai/RMBG-1.4",
        trust_remote_code=True
    )
    model.eval()

    # Use GPU if available, otherwise CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    logger.info(f"Model loaded on {device}")

    transform_image = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def remove_background(image_bytes: bytes) -> bytes:
    """Process image and return transparent PNG bytes."""
    device = next(model.parameters()).device

    # Open + convert to RGB
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_size = img.size

    # Preprocess
    input_tensor = transform_image(img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        preds = model(input_tensor)

    # Extract mask (first output, first item)
    pred = preds[0][0].squeeze()
    mask = torch.sigmoid(pred).cpu().numpy()

    # Resize mask back to original image size
    mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize(orig_size, Image.LANCZOS)

    # Apply mask as alpha channel
    orig_rgba = img.convert("RGBA")
    r, g, b, _ = orig_rgba.split()
    result = Image.merge("RGBA", (r, g, b, mask_img))

    # Output PNG bytes
    out_buf = io.BytesIO()
    result.save(out_buf, format="PNG", optimize=True)
    return out_buf.getvalue()

# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "briaai/RMBG-1.4",
        "device": str(next(model.parameters()).device) if model else "not loaded"
    }

# ── Main endpoint ──────────────────────────────────────────────────────────────
@app.post("/remove-bg")
async def remove_bg(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    image_bytes = await file.read()
    if len(image_bytes) < 100:
        raise HTTPException(400, "Image file is empty")

    logger.info(f"Processing image: {round(len(image_bytes)/1024)}KB")

    png_bytes = remove_background(image_bytes)

    logger.info(f"Done: {round(len(png_bytes)/1024)}KB PNG")
    return Response(content=png_bytes, media_type="image/png")
