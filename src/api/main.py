"""FastAPI application entrypoint for IDR-GE."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router, get_data_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up spatial data caches on application startup."""
    # Pre-load dataset and spatial weights
    dm = get_data_manager()
    dm.load()
    yield


app = FastAPI(
    title="IDR-GE Spatial Analytics API",
    description="Spatial econometrics and disaster readiness analytics backend for 514 Indonesian regencies.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for local Streamlit / frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register route handlers
app.include_router(router)
