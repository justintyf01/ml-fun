"""
HDB Resale Price Prediction API
================================
FastAPI application serving XGBoost predictions.
"""
import os
from contextlib import asynccontextmanager

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.model_loader import ModelArtifacts
from app.schemas import FeedbackRequest, OptionsResponse, PredictionRequest, PredictionResponse
from app.inference import predict

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")

artifacts = ModelArtifacts()
limiter   = Limiter(key_func=get_remote_address, default_limits=["200/hour"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    artifacts.load()
    yield


app = FastAPI(
    title="HDB Resale Price Predictor",
    description="Predict Singapore HDB resale flat prices using XGBoost (R²=0.981)",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    if not artifacts.models:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {"status": "ok", "models_loaded": len(artifacts.models)}


@app.get("/api/options", response_model=OptionsResponse)
async def options():
    return OptionsResponse(**artifacts.reference_data)


@app.post("/api/predict", response_model=PredictionResponse)
@limiter.limit("10/minute")
async def predict_price(request: Request, req: PredictionRequest):
    return predict(req, artifacts)


@app.get("/api/block-info")
@limiter.limit("60/minute")
async def block_info(
    request: Request,
    block: str = Query(..., description="HDB block number e.g. 406"),
    street: str = Query(..., description="Street name e.g. ANG MO KIO AVE 10"),
):
    """
    Return auto-fill data for a given block + street: town, lease commencement year,
    typical floor areas and flat models per flat type.

    Falls back to street-level town inference if the exact block is unknown
    (e.g. recently completed blocks not yet in training data).
    """
    key = f"{block.upper().strip()} {street.upper().strip()}"
    if key in artifacts.block_lookup:
        return artifacts.block_lookup[key]

    town = artifacts.street_to_town.get(street.upper().strip())
    if town:
        return {"town": town, "lease_commence_date": None, "floor_areas": {}, "flat_models": {}}

    raise HTTPException(status_code=404, detail="Block not found")


@app.get("/api/nearest-town")
@limiter.limit("60/minute")
async def nearest_town(
    request: Request,
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
):
    """Return the town of the nearest known address to the given coordinates."""
    if not artifacts.address_town_points:
        raise HTTPException(status_code=503, detail="Town data not loaded")
    nearest = min(artifacts.address_town_points, key=lambda p: (lat - p[0]) ** 2 + (lon - p[1]) ** 2)
    return {"town": nearest[2]}


@app.post("/api/feedback")
@limiter.limit("5/hour")
async def submit_feedback(request: Request, body: FeedbackRequest):
    """Forward anonymous feedback to Telegram."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise HTTPException(status_code=503, detail="Feedback not configured")

    stars_display = "★" * body.stars + "☆" * (5 - body.stars)
    text = f"🏠 *HDB Estimator Feedback*\n⭐ {body.stars}/5  {stars_display}"
    if body.message.strip():
        text += f"\n\n{body.message.strip()}"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"},
        )
    if not r.is_success:
        raise HTTPException(status_code=502, detail="Failed to deliver feedback")
    return {"ok": True}


@app.get("/")
async def serve_frontend():
    return FileResponse("app/static/index.html")
