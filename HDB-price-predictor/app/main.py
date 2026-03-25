"""
HDB Resale Price Prediction API
================================
FastAPI application serving XGBoost predictions.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.model_loader import ModelArtifacts
from app.schemas import PredictionRequest, PredictionResponse, OptionsResponse
from app.inference import predict

artifacts = ModelArtifacts()


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    return {"status": "ok", "models_loaded": len(artifacts.models)}


@app.get("/api/options", response_model=OptionsResponse)
async def options():
    return OptionsResponse(**artifacts.reference_data)


@app.post("/api/predict", response_model=PredictionResponse)
async def predict_price(req: PredictionRequest):
    return predict(req, artifacts)
