"""
services/case_service/main.py
----------------------------------------------------------------
case-service :8002 — non-ML component.

REST API for the analyst console (list/resolve cases) PLUS a background
thread consuming the payment.decided stream to open new cases (FR5, FR6).
----------------------------------------------------------------
"""
import threading
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from services.case_service.consumer import run as run_consumer
from services.case_service.store import get_connection, list_cases, resolve_case

conn = get_connection()


@asynccontextmanager
async def lifespan(app: FastAPI):
    thread = threading.Thread(target=run_consumer, daemon=True)
    thread.start()
    yield


app = FastAPI(title="Case Service", version="1.0.0", lifespan=lifespan)


@app.get("/health", tags=["ops"])
def health():
    return {"status": "healthy"}


@app.get("/cases", tags=["cases"])
def get_cases(status: Literal["OPEN", "RESOLVED"] | None = None):
    return list_cases(get_connection(), status)


class ResolveRequest(BaseModel):
    verdict: Literal["CONFIRMED_FRAUD", "GENUINE"]


@app.post("/cases/{case_id}/resolve", tags=["cases"])
def resolve(case_id: str, body: ResolveRequest):
    result = resolve_case(get_connection(), case_id, body.verdict)
    if result is None:
        raise HTTPException(status_code=404, detail="Case not found")
    return result
