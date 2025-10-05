from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .agents import run_scout_pipeline
import os
from typing import Dict, Any

app = FastAPI(
    title="Project AURA API",
    description="Configurable API for the AURA Exoplanet Detection Pipeline",
    version="4.0.0"
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/analyze-star-id/", response_model=Dict[str, Any])
async def analyze_star_id(
    star_id: str = Form(...),
    download_all: bool = Form(False),
    flatten_method: str = Form('biweight'),
    window_length: float = Form(0.5),
    use_threads: int = Form(1),
    period_max: float = Form(100.0),
    oversampling_factor: int = Form(5),
    duration_grid_step: float = Form(1.1),
    n_transits_min: int = Form(2),
    transit_depth_min: float = Form(100.0),
    timeout_seconds: int = Form(600) 
):
    """
    Accepts a star ID and analysis parameters, runs the full AURA pipeline,
    and returns the final analysis including the light curve and any found signals.
    """
    if not star_id:
        raise HTTPException(status_code=400, detail="Star ID cannot be empty.")

    try:
        # Ensure use_threads is at least 1
        if use_threads <= 0:
            use_threads = os.cpu_count() or 1

        analysis_results = run_scout_pipeline(
            star_id=star_id, 
            download_all=download_all, 
            flatten_method=flatten_method,
            window_length=window_length, 
            use_threads=use_threads, 
            period_max=period_max,
            oversampling_factor=oversampling_factor, 
            duration_grid_step=duration_grid_step,
            n_transits_min=n_transits_min, 
            transit_depth_min=transit_depth_min
        )
        
        return {
            "status": "ANALYSIS_COMPLETE",
            "star_id": star_id,
            "light_curve": analysis_results.get("light_curve", {}),
            "scout_results": analysis_results.get("signals", [])
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred in the pipeline: {e}")

