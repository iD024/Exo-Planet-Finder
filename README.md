# Project AURA 🌌

A configurable, end-to-end exoplanet detection pipeline with a user-friendly web interface and an AI-powered vetting agent.

Project AURA (Automated Universal Reality-Analyzer) is a full-stack application designed to discover and analyze exoplanet candidates from raw astronomical light curve data. It provides a powerful, configurable backend pipeline driven by a simple and intuitive web-based frontend.
Core Features

  **1)** End-to-End Analysis: From star ID to planetary parameters, the entire analysis is handled in one seamless pipeline.

  **2)** Mission-Aware Data Processing: Automatically detects whether data is from the Kepler or TESS missions and applies the appropriate, state-of-the-art systematics corrections (e.g., Pixel Level Decorrelation for TESS).

  **3)** Highly Configurable Search: The web UI allows for fine-grained control over the entire analysis, including light curve flattening methods, CPU usage, and all major transitleastsquares search parameters.

  **4)** AI-Powered Vetting: A custom-trained XGBoost model (the "Coordinator Agent") analyzes each detected signal, classifying it as a "Planet Candidate" or "False Positive" and providing a detailed explanation.

  **5)** Robust Multi-Planet Detection: Implements a two-pass search algorithm that finds the strongest signal, masks it out, and re-searches the data to find additional planets in multi-planet systems.

  **6)** Scientifically Robust Measurements: The pipeline doesn't just trust the model fit. It performs its own manual photometry on the light curve to derive robust measurements for transit depth and duration. It also queries professional astronomical catalogs (KIC/TIC) for the most    accurate stellar parameters.

  **7)** Interactive Visualization: Displays the full, flattened light curve and a detailed, organized breakdown of all vetting statistics, stellar parameters, and calculated planetary parameters for each signal found.

# How It Works

The AURA pipeline follows a workflow designed to mirror the process used by professional astrophysicists:

   Input: The user enters a star ID (e.g., "Kepler-227" or "TIC 88863718") and configures analysis parameters in the Streamlit UI.
   
   API Request: The frontend sends a request to the FastAPI backend.

   Data Acquisition: The backend uses lightkurve to search for and download the appropriate data from the MAST archive (raw Target Pixel Files for TESS, pre-processed light curves for Kepler).

   Systematics Correction: For TESS data, a Pixel Level Decorrelation (PLD) correction is applied to each observing sector to remove instrumental noise. The corrected light curves are then stitched together.

   Stellar Variability Flattening: The pipeline uses wotan to flatten the light curve, removing noise from the star's own variability.

   Signal Search: transitleastsquares performs an exhaustive search for transit-like signals, powered by high-quality stellar parameters (radius, mass, limb darkening) fetched from online catalogs.

   Robust Measurement: The pipeline performs its own photometry on the flattened light curve to get a reliable measurement of the transit's depth and duration.

   AI Vetting: The features of the detected signal (period, depth, duration, impact parameter) are passed to the trained XGBoost model, which provides a classification and confidence score.

   Iterative Search: The first signal is masked, and the entire search process is run a second time to find any weaker, secondary signals.

   Visualization: The final results, including the light curve plot and detailed parameters for all found signals, are sent back to the Streamlit frontend for display.

# Technology Stack

   Backend API: FastAPI

   Frontend UI: Streamlit

   Scientific Pipeline:

   Data Acquisition & Correction: lightkurve

   Transit Search: transitleastsquares

   Light Curve Flattening: wotan

   AI Model: xgboost, scikit-learn

   Data Handling: pandas, numpy

   Astrophysical Calculations: astropy

# Installation

    Clone the repository:

    git clone [https://github.com/your-username/project-aura.git](https://github.com/your-username/project-aura.git)
    cd project-aura

    Create and activate a virtual environment:

    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

    Install the required packages:
    (It is recommended to create a requirements.txt file for this project)

    pip install fastapi uvicorn streamlit requests lightkurve transitleastsquares wotan scikit-learn xgboost pandas astropy plotly

# Usage

The application consists of three main parts: the AI training script, the backend API, and the frontend UI.
1. (Optional) Train the AI Model

If you have new training data or want to retrain the AI, run the training script. This will generate the coordinator_model.joblib file in the models/ directory.

```python train_coordinator.py```

2. Run the Backend Server

The backend must be running for the frontend to communicate with it. From the project's root directory, run:

```uvicorn main:app --reload```

The API will be available at `http://127.0.0.1:8000.`
3. Run the Frontend Application

In a new terminal window, navigate to the project's root directory and run the Streamlit app:

```streamlit run app.py```

This will open the Project AURA web interface in your browser, ready for you to start discovering planets!

This project was built as a collaborative effort. Happy planet hunting!
