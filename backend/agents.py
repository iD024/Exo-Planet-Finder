import numpy as np
import pandas as pd
import joblib
from transitleastsquares import transitleastsquares, transit_mask, catalog_info
import lightkurve as lk
from lightkurve.correctors import PLDCorrector
from wotan import flatten
import os
from astropy import constants as const
from astropy import units as u
import re
from typing import List, Dict, Any, Tuple

# --- CONFIGURATION ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'coordinator_model.joblib')
SDE_THRESHOLD = 6.0
MINIMUM_DATA_POINTS = 100
PLOT_DATA_MAX_POINTS = 10000

# --- LOAD ML MODEL ONCE AT STARTUP ---
try:
    COORDINATOR_MODEL = joblib.load(MODEL_PATH)
except FileNotFoundError:
    print(f"Warning: Coordinator model not found at {MODEL_PATH}. AI predictions will be disabled.")
    COORDINATOR_MODEL = None

def _validate_range(value: float, min_val: float, max_val: float) -> Tuple[float, float]:
    is_valid = lambda x: x is not None and not np.isnan(x)
    if not is_valid(min_val) or min_val > value: min_val = value * 0.9
    if not is_valid(max_val) or max_val < value: max_val = value * 1.1
    return min_val, max_val


# --- HELPER FUNCTIONS FOR MODULARITY ---
def get_stellar_parameters(star_id: str, lc: lk.LightCurve) -> Dict[str, Any]:
    print("Attempting to retrieve stellar parameters...")
    params = {
        "limb_dark": None, "stellar_radius": None, "stellar_radius_min": None, "stellar_radius_max": None,
        "stellar_mass": None, "stellar_mass_min": None, "stellar_mass_max": None,
        "stellar_teff": None, "stellar_logg": None
    }
    
    try:
        is_tic = "tic" in star_id.lower() or (lc.meta.get('MISSION') == 'TESS')
        is_kic = "kic" in star_id.lower()
        
        clean_id = None
        if is_tic or is_kic:
            try:
                clean_id = int(star_id[3:])
            except (ValueError, TypeError):
                clean_id = lc.meta.get('TICID') or lc.meta.get('KEPLERID')

        if clean_id:
            if is_tic:
                print(f"Querying TESS Input Catalog (TIC) for ID: {clean_id}...")
                ab, M_star, M_star_min, M_star_max, R_star, R_star_min, R_star_max = catalog_info(TIC_ID=clean_id)
            else: # is_kic or default
                print(f"Querying Kepler Input Catalog (KIC) for ID: {clean_id}...")
                ab, M_star, M_star_min, M_star_max, R_star, R_star_min, R_star_max = catalog_info(KIC_ID=clean_id)
            
            # --- Convert all numpy types to native Python floats ---
            params.update({
                "limb_dark": tuple(map(float, ab)) if ab is not None else None,
                "stellar_radius": float(R_star) if R_star is not None else None,
                "stellar_radius_min": float(R_star_min) if R_star_min is not None else None,
                "stellar_radius_max": float(R_star_max) if R_star_max is not None else None,
                "stellar_mass": float(M_star) if M_star is not None else None,
                "stellar_mass_min": float(M_star_min) if M_star_min is not None else None,
                "stellar_mass_max": float(M_star_max) if M_star_max is not None else None
            })
            print("Successfully retrieved parameters from professional catalog.")
        else:
            print("Could not determine a valid catalog ID.")

    except Exception as e:
        print(f"Catalog query failed: {e}. Using light curve metadata as fallback.")

    is_valid = lambda x: x is not None and not np.isnan(x)
    if not is_valid(params["stellar_radius"]): params["stellar_radius"] = float(lc.meta.get('RADIUS', 1.0))
    if not is_valid(params["stellar_mass"]): params["stellar_mass"] = float(lc.meta.get('MASS', 1.0))
    
    params["stellar_radius_min"], params["stellar_radius_max"] = _validate_range(params["stellar_radius"], params["stellar_radius_min"], params["stellar_radius_max"])
    params["stellar_mass_min"], params["stellar_mass_max"] = _validate_range(params["stellar_mass"], params["stellar_mass_min"], params["stellar_mass_max"])

    params["stellar_teff"] = float(lc.meta.get('TEFF', 5000))
    params["stellar_logg"] = float(lc.meta.get('LOGG', 4.5))
    
    print(f"Using Stellar Parameters: R={params['stellar_radius']:.2f}, M={params['stellar_mass']:.2f}, Teff={params['stellar_teff']:.0f}")
    return params

def prepare_light_curve(lc: lk.LightCurve, flatten_method: str, window_length: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Takes a light curve (which may have been corrected) and flattens it to remove stellar variability.
    """
    time, flux = lc.time.value, lc.flux.value
    
    normalized_flux = flux / np.nanmedian(flux)
    if len(time) < MINIMUM_DATA_POINTS: 
        raise ValueError(f"Light curve too short after cleaning ({len(time)} points).")
    
    print(f"Flattening stellar variability using method='{flatten_method}'...")
    flattened_flux, _ = flatten(time, normalized_flux, method=flatten_method, window_length=window_length, return_trend=True)
    return time, flattened_flux


def calculate_planetary_parameters(tls_results, stellar_params: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculates physically accurate planetary parameters. The semi-major axis is derived
    using Kepler's Third Law for robust results.
    """
    params = {}
    try:
        # 1. Planet Radius 
        r_star_in_r_earth = (stellar_params["stellar_radius"] * u.R_sun).to(u.R_earth).value
        params['planetary_radius_earth'] = float(r_star_in_r_earth * np.sqrt(tls_results.depth))

        # --- FIX: Calculate Semi-Major Axis using Kepler's Third Law ---
        # a^3 = (G * M * P^2) / (4 * pi^2)
        M_star_kg = (stellar_params["stellar_mass"] * u.M_sun).to(u.kg)
        period_sec = (tls_results.period * u.day).to(u.s)
        
        # Calculate semi-major axis in meters
        semi_major_axis_m = ((const.G * M_star_kg * period_sec**2) / (4 * np.pi**2))**(1/3)
        
        # Convert to AU for reporting
        params['semi_major_axis_au'] = semi_major_axis_m.to(u.au).value

        # 3. Equilibrium Temperature (in K)
        # T_eq = T_star * sqrt(R_star / (2 * a))
        T_star_k = stellar_params["stellar_teff"]
        R_star_m = (stellar_params["stellar_radius"] * u.R_sun).to(u.m)

        # Use an adaptive albedo based on a preliminary temperature calculation
        prelim_eq_temp = T_star_k * np.sqrt(R_star_m / (2 * semi_major_axis_m))
        
        bond_albedo = 0.3 if prelim_eq_temp.value < 1000 else 0.1 # Cooler planets are more reflective
        temp_factor = (1 - bond_albedo)**0.25
        eq_temp = T_star_k * temp_factor * np.sqrt(R_star_m / (2 * semi_major_axis_m))
        params['equilibrium_temp_k'] = float(eq_temp.value)

    except Exception as e:
        print(f"Error calculating planetary parameters: {e}")
        params.setdefault('planetary_radius_earth', float('nan'))
        params.setdefault('semi_major_axis_au', float('nan'))
        params.setdefault('equilibrium_temp_k', float('nan'))
    return params

def run_coordinator(scout_results: dict) -> dict:
    if not COORDINATOR_MODEL:
        return {"prediction": "AI UNAVAILABLE", "confidence": 0, "explanation": "Coordinator model is not loaded."}
    
    features_data = {
        'period': scout_results.get('period', 0),
        'duration': scout_results.get('duration', 0) * 24,
        'depth': scout_results.get('depth', 0) * 1e6,
        'stellar_teff': scout_results.get('stellar_teff', 5000),
        'stellar_logg': scout_results.get('stellar_logg', 4.5),
        'impact': scout_results.get('impact', 0.5) # Use a neutral default if missing
    }
    features = pd.DataFrame([features_data])
    probabilities = COORDINATOR_MODEL.predict_proba(features)[0]
    prediction = np.argmax(probabilities)
    confidence = probabilities[prediction]
    final_label = "PLANET CANDIDATE" if prediction == 1 else "FALSE POSITIVE"

    explanation = "The AI model analyzed the signal's characteristics."
    if final_label == "PLANET CANDIDATE" and scout_results.get('snr', 0) < 10:
        explanation += " However, its Signal-to-Noise Ratio is low, and follow-up observations are recommended."
    elif final_label == "FALSE POSITIVE" and scout_results.get('odd_even_mismatch', 0) > 3:
        explanation += f" The significant odd-even depth mismatch ({scout_results['odd_even_mismatch']:.2f}σ) suggests a background eclipsing binary."
            
    return {"prediction": final_label, "confidence": float(confidence), "explanation": explanation}


# --- MAIN PIPELINE ---

def run_scout_pipeline(
    star_id: str, 
    download_all: bool, 
    flatten_method: str, 
    window_length: float, 
    use_threads: int,
    period_max: float, 
    oversampling_factor: int, 
    duration_grid_step: float, 
    n_transits_min: int,
    transit_depth_min: float
) -> List[Dict[str, Any]]:
    print(f"--- Starting AURA Pipeline for {star_id} ---")
    
    # 1. Data Acquisition - Now focused on Kepler and TESS
    search_result = lk.search_lightcurve(star_id, author=("Kepler", "TESS", "TESS-SPOC"))
    if not search_result: 
        raise ValueError(f"No light curve found for '{star_id}'.")


    if "TIC" in star_id:
        print("Detected mission: TIC. Applying mission-specific data acquisition strategy.")
        # For TESS, we need the raw pixel data for the best systematics correction (PLD)
        print("TESS mission detected. Searching for Target Pixel Files...")
        search_result = lk.search_targetpixelfile(star_id, author="TESS-SPOC")
        if not search_result: 
            raise ValueError(f"No TESS Target Pixel Files found for '{star_id}'.")
        
        # Download all TPFs regardless of 'download_all' for TESS, as stitching sectors is crucial
        tpf_collection = search_result.download_all(quality_bitmask='default', cache=False)
        if not tpf_collection:
            raise ValueError(f"Could not download TESS Target Pixel Files for '{star_id}'.")
        
        # --- Apply PLD correction sector-by-sector and then stitch ---
        print("Applying TESS PLD systematics correction sector-by-sector...")
        corrected_lc_list = []
        for tpf in tpf_collection:
            try:
                # --- FIXED: Initialize the corrector with the TPF object, not the light curve ---
                corrector = PLDCorrector(tpf)
                # Perform the correction, which returns a LightCurve object
                corrected_lc = corrector.correct()
                corrected_lc_list.append(corrected_lc)
            except Exception as e:
                print(f"Warning: Could not process TESS sector {tpf.sector}. Error: {e}")
        
        if not corrected_lc_list:
            raise ValueError("Failed to process any TESS sectors after PLD correction.")
            
        # Stitch the list of corrected light curves into a single object
        lc_collection_corrected = lk.LightCurveCollection(corrected_lc_list)
        lc = lc_collection_corrected.stitch()

    else: # For Kepler, we can download the light curves directly
        print("Detected mission: KIC. Applying mission-specific data acquisition strategy.")
        if download_all:
            print("Downloading ALL available Kepler light curve data...")
            lc_collection = search_result.download_all(flux_column="pdcsap_flux", cache=False)
            if not lc_collection: 
                raise ValueError(f"Could not download any data for '{star_id}'.")
            lc = lc_collection.stitch()
        else:
            print("Downloading the first available Kepler light curve segment...")
            lc = search_result[0].download(flux_column="pdcsap_flux", cache=False)
    
    lc = lc.remove_nans()
    
    # 2. Get Stellar Parameters
    stellar_params = get_stellar_parameters(star_id, lc)

    # 3. Prepare Light Curve (Systematics Correction & Flattening)
    time, flattened_flux = prepare_light_curve(lc, flatten_method, window_length)

    # 4. Dynamic Search Period Adjustment
    time_span = time[-1] - time[0]
    sensible_period_max = time_span / 2
    if period_max > sensible_period_max:
        print(f"Warning: User-defined max period ({period_max}d) is too long for the observation span ({time_span:.1f}d).")
        period_max = sensible_period_max
        print(f"Adjusting max search period to {period_max:.1f} days.")
    
    # 5. Iterative Signal Search (Two-Pass)
    all_found_signals = []
    current_time, current_flux = time, flattened_flux
    for i in range(2):
        print(f"\n--- Starting Search Pass {i+1} ---")
        if len(current_time) < MINIMUM_DATA_POINTS: 
            print("Not enough data points remaining for another search pass.")
            break

        model = transitleastsquares(current_time, current_flux)
        # --- Passing parameters to the model ---
        results = model.power(
            u=stellar_params["limb_dark"],
            limb_dark="quadratic", 
            R_star=stellar_params["stellar_radius"], 
            R_star_min=stellar_params["stellar_radius_min"], 
            R_star_max=stellar_params["stellar_radius_max"],
            M_star=stellar_params["stellar_mass"], 
            M_star_min=stellar_params["stellar_mass_min"], 
            M_star_max=stellar_params["stellar_mass_max"],
            n_transits_min=n_transits_min, 
            period_max=period_max, 
            oversampling_factor=oversampling_factor,
            duration_grid_step=duration_grid_step, 
            use_threads=use_threads,
            show_progress_bar=True, 
            transit_depth_min=(transit_depth_min / 1e6)
        )
        

        if results.SDE < SDE_THRESHOLD:
            print("No significant signal found in this pass (SDE < threshold).")
            break

        print(f"Pass {i+1}: Found signal with SDE={results.SDE:.2f} at Period={results.period:.3f} days.")
        
        # 6. Manual Photometry for Robust Depth & Duration
        folded_time = (current_time - results.T0 + 0.5 * results.period) % results.period - 0.5 * results.period
        in_transit_mask = np.abs(folded_time) < (results.duration / 24 / 2) # Duration is in hours, convert to days
        
        if np.any(in_transit_mask) and np.any(~in_transit_mask):
            # Robust Depth
            robust_depth = np.median(current_flux[~in_transit_mask]) - np.median(current_flux[in_transit_mask])
            print(f"TLS model depth: {results.depth:.6f}. Measured depth: {robust_depth:.6f}")
            results.depth = robust_depth
            
            in_transit_times = folded_time[in_transit_mask]
            # Duration in days, then convert to hours for reporting
            robust_duration_days = in_transit_times.max() - in_transit_times.min()
            robust_duration_hours = robust_duration_days * 24
            print(f"TLS model duration: {results.duration:.2f}h. Measured duration: {robust_duration_hours:.2f}h")
            results.duration = robust_duration_hours # Overwrite with the reliable, measured value

        else:
            print("Warning: Could not perform robust photometry. Falling back to model depth.")

        # 7. Package, Calculate, & Predict
        scout_results = {
            "sde": results.SDE, 
            "snr": results.snr, 
            "T0": results.T0,
            "odd_even_mismatch": results.odd_even_mismatch,
            "false_alarm_probability": results.FAP, 
            "period": results.period, 
            "period_uncertainty": results.period_uncertainty,
            "duration": results.duration, 
            "depth": results.depth, 
            "transit_count": results.transit_count,
            "distinct_transit_count": results.distinct_transit_count, 
            "empty_transit_count": results.empty_transit_count,
            **stellar_params
        }
        scout_results.update(calculate_planetary_parameters(results, stellar_params))
        scout_results["coordinator_prediction"] = run_coordinator(scout_results)
        all_found_signals.append(scout_results)

        # 8. Mask Signal for Next Pass
        print(f"Masking transit times for period {results.period:.3f} and re-searching.")
        in_transit_to_mask = transit_mask(t=current_time, period=results.period, duration=results.duration, T0=results.T0)
        current_time, current_flux = current_time[~in_transit_to_mask], current_flux[~in_transit_to_mask]

    # 9. Serialize and Return Results
    if not all_found_signals:
        return {
            "light_curve": {"time": [], "flux": []},
            "signals": []
        }

    def _to_serializable(obj):
        if isinstance(obj, (np.generic, np.number)): 
            return obj.item()
        if isinstance(obj, dict):
             return {k: _to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, np.ndarray)):
             return [_to_serializable(x) for x in obj]
        return obj

    # Downsample light curve 
    plot_time, plot_flux = time, flattened_flux
    if len(plot_time) > PLOT_DATA_MAX_POINTS:
        step = len(plot_time) // PLOT_DATA_MAX_POINTS
        plot_time = plot_time[::step]
        plot_flux = plot_flux[::step]

    return {
        "light_curve": _to_serializable({"time": plot_time, "flux": plot_flux}),
        "signals": _to_serializable(sorted(all_found_signals, key=lambda x: x['sde'], reverse=True))
    }
