"""Thin FastAPI wrapper around the shared pipeline inference implementation."""
from contextlib import asynccontextmanager
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import DEFAULT_TRAINING
from .inference import generate, load_generator


class InferenceRequest(BaseModel):
    question: str = Field(min_length=1)
    max_new_tokens: int = Field(default=50, ge=1, le=4096)
    do_sample: bool = False


def create_app(model_path=None):
    selected_model_path = Path(
        model_path or os.environ.get("MODEL_PATH", DEFAULT_TRAINING.output_dir)
    )
    state = {"generator": None}

    @asynccontextmanager
    async def lifespan(app):
        try:
            state["generator"] = load_generator(selected_model_path)
            yield
        except Exception:
            state["generator"] = None
            raise
        finally:
            state["generator"] = None

    app = FastAPI(title="Math Modal Inference", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": state["generator"] is not None}

    @app.post("/infer")
    def infer(request: InferenceRequest):
        if state["generator"] is None:
            raise HTTPException(status_code=503, detail="Model is not initialized.")
        try:
            result = generate(state["generator"], request.question,
                              request.max_new_tokens, request.do_sample)
            return {"generated_text": result}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return app


app = create_app()
