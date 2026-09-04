from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ============================================================================
# PATH SETUP
# ============================================================================

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# CANONICAL PROJECT IMPORTS
# ============================================================================

from app.phase18_bridge import result_to_dict, run_live_evaluation


# ============================================================================
# FASTAPI
# ============================================================================

app = FastAPI(
    title="Adaptive Risk Governor API",
    description="Backend API for the Adaptive Risk Governor intelligence console.",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# MODELS
# ============================================================================


class EvaluationRequest(BaseModel):
    seed: int = 42


# ============================================================================
# HEALTH
# ============================================================================


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "adaptive-risk-governor",
        "backend": "fastapi",
        "canonical_source": "src/",
    }


# ============================================================================
# PROJECT OVERVIEW
# ============================================================================


@app.get("/api/dashboard")
def dashboard() -> dict[str, Any]:
    """
    Product-level overview.

    We deliberately do not manufacture fake transaction metrics here.
    The page describes the live architecture and exposes real Phase 18
    evaluation data through the evaluation endpoint.
    """

    return {
        "product": {
            "name": "Adaptive Risk Governor",
            "subtitle": "Closed-loop AI risk intelligence for adaptive abuse",
            "status": "OPERATIONAL",
        },
        "architecture": [
            {
                "id": "case",
                "label": "Incoming Claim",
                "description": "Refund or chargeback request enters the system.",
                "status": "ACTIVE",
            },
            {
                "id": "evidence",
                "label": "Evidence Agent",
                "description": "Collects and cross-checks evidence.",
                "status": "ACTIVE",
            },
            {
                "id": "intelligence",
                "label": "Risk Intelligence",
                "description": "Combines behavioral, semantic and network signals.",
                "status": "ACTIVE",
            },
            {
                "id": "governor",
                "label": "Risk Governor",
                "description": "Controls how the support agent may proceed.",
                "status": "ACTIVE",
            },
            {
                "id": "outcome",
                "label": "Business Outcome",
                "description": "Decision produces an observable economic outcome.",
                "status": "ACTIVE",
            },
            {
                "id": "adaptation",
                "label": "Adaptive Feedback",
                "description": "Adaptive customers change behavior in response.",
                "status": "ACTIVE",
            },
        ],
        "governor_actions": [
            "ALLOW_AGENT_A_DECISION",
            "REQUEST_ADDITIONAL_EVIDENCE",
            "ESCALATE_TO_HUMAN_REVIEW",
        ],
        "signals": [
            "Behavioral risk",
            "Strategic behavior",
            "Network risk",
            "GNN risk",
            "Evidence consistency",
            "Temporal abnormality",
            "Semantic claim switching",
            "Request velocity",
            "Amount acceleration",
        ],
    }


# ============================================================================
# PHASE 18
# ============================================================================


@app.post("/api/evaluation/phase18")
def phase18(request: EvaluationRequest) -> dict[str, Any]:
    try:
        result = run_live_evaluation(seed=request.seed)

        return {
            "success": True,
            "result": result_to_dict(result),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Phase 18 evaluation failed: {exc}",
        ) from exc


# ============================================================================
# ROOT
# ============================================================================


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Adaptive Risk Governor API",
        "docs": "/docs",
    }