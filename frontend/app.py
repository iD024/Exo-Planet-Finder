# app.py

import streamlit as st
import requests
import os
import pandas as pd
import plotly.graph_objects as go

API_URL = "http://127.0.0.1:8000/analyze-star-id/"
st.set_page_config(page_title="Project AURA", layout="wide")
st.title("Project AURA 🌌")
st.subheader("Configurable Exoplanet Detection Pipeline")

# --- Plotting Functions ---
def plot_full_lightcurve(lc_data, star_id):
    """Generates an interactive plot of the full light curve."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=lc_data['time'], 
        y=lc_data['flux'], 
        mode='markers', 
        marker=dict(size=2, color='blue', opacity=0.6),
        name='Flux'
    ))
    fig.update_layout(
        title=f"Full Flattened Light Curve for {star_id}",
        xaxis_title="Time (BJD)",
        yaxis_title="Normalized Flux",
        height=400,
        template="plotly_dark"
    )
    return fig

def plot_folded_lightcurve(lc_data, signal):
    """Generates an interactive plot of the folded light curve for a signal."""
    if 'T0' not in signal:
        return None # Cannot plot without T0
        
    # Phase-fold the light curve
    period = signal['period']
    t0 = signal['T0']
    df = pd.DataFrame(lc_data)
    df['phased_time_days'] = ((df['time'] - t0 + 0.5 * period) % period) - 0.5 * period
    
    # Bin the data to see the average transit shape
    bins = np.linspace(-0.5 * period, 0.5 * period, 100)
    df['binned'] = pd.cut(df['phased_time_days'], bins=bins)
    binned_data = df.groupby('binned')['flux'].median().reset_index()
    binned_data['mid_time'] = binned_data['binned'].apply(lambda x: x.mid)
    
    fig = go.Figure()
    
    # Plot raw phase-folded points
    fig.add_trace(go.Scatter(
        x=df['phased_time_days'] * 24, # Convert to hours
        y=df['flux'],
        mode='markers',
        marker=dict(size=2, color='grey', opacity=0.5),
        name='Folded Data'
    ))

    # Plot binned median points
    fig.add_trace(go.Scatter(
        x=binned_data['mid_time'] * 24, # Convert to hours
        y=binned_data['flux'],
        mode='markers',
        marker=dict(size=6, color='red'),
        name='Binned Median'
    ))

    # Set plot range to +/- 3 transit durations for context
    plot_width_hours = signal.get('duration', 1) * 3
    fig.update_xaxes(range=[-plot_width_hours, plot_width_hours])

    fig.update_layout(
        title=f"Folded Light Curve (Period: {period:.5f} days)",
        xaxis_title="Time from Transit Center (Hours)",
        yaxis_title="Normalized Flux",
        height=400,
        template="plotly_dark"
    )
    return fig


# --- UI Layout ---
with st.sidebar:
    st.header("Target Input")
    star_id = st.text_input("Enter a Star Name or ID", placeholder="e.g., Kepler-227")
    
    st.header("Analysis Configuration")
    download_all = st.checkbox("Download All Available Data (Slower)", value=True)
    
    timeout_seconds = st.number_input("Request Timeout (seconds)", min_value=30, value=600, step=30, help="Maximum time to wait for the analysis to complete.")
    with st.expander("Light Curve Flattening Settings"):
        flatten_method = st.selectbox("Flattening Method", ['biweight', 'cosine', 'savgol', 'median'], index=0)
        window_length = st.number_input("Window Length (days)", min_value=0.01, value=0.5, step=0.1)

    with st.expander("Advanced Transit Search Settings (TLS)"):
        use_threads = st.slider("CPU Cores to Use", min_value=1, max_value=os.cpu_count(), value=os.cpu_count() - 1 if os.cpu_count() > 1 else 1)
        period_max = st.number_input("Max Period to Search (days)", min_value=1.0, value=20.0, step=1.0)
        oversampling_factor = st.number_input("Oversampling Factor", min_value=1, max_value=15, value=5)
        duration_grid_step = st.number_input("Duration Grid Step", min_value=1.01, value=1.1, format="%.2f")
        n_transits_min = st.number_input("Minimum Transits to Detect", min_value=2, value=2, step=1)
        transit_depth_min = st.number_input("Minimum Transit Depth (ppm)", min_value=10.0, value=100.0, step=10.0, help="Set lower to find smaller planets, higher for noisy data.")


    analyze_button = st.button("Analyze for Planets", type="primary", use_container_width=True)

st.markdown("Enter a star's ID, configure the analysis, and click 'Analyze' to begin.")

# --- Analysis Logic ---
if analyze_button and star_id:
    with st.spinner(f'Analyzing {star_id}... This may take several minutes.'):
        payload = {
            'star_id': star_id, 
            'download_all': download_all, 
            'flatten_method': flatten_method,
            'window_length': window_length,
            'use_threads': use_threads,
            'period_max': period_max,
            'oversampling_factor': oversampling_factor,
            'duration_grid_step': duration_grid_step,
            'n_transits_min': n_transits_min,
            'transit_depth_min': transit_depth_min
        }

        try:
            response = requests.post(API_URL, data=payload, timeout=timeout_seconds)

            if response.status_code == 200:
                results_data = response.json()
                st.header("Analysis Results")
                
                # --- NEW: Display Full Light Curve Plot ---
                light_curve = results_data.get('light_curve')
                if light_curve and light_curve.get('time'):
                    full_lc_fig = plot_full_lightcurve(light_curve, star_id)
                    st.plotly_chart(full_lc_fig, use_container_width=True)
                else:
                    st.info("Light curve data was not returned from the API.")

                if results_data.get('status') == "ANALYSIS_COMPLETE" and results_data.get('scout_results'):
                    
                    for i, scout_res in enumerate(results_data['scout_results']):
                        st.subheader(f"Signal {i+1} (SDE: {scout_res.get('sde', 0):.2f})")
                        
                        # --- NEW: Display Folded Light Curve Plot ---
                        folded_fig = plot_folded_lightcurve(light_curve, scout_res)
                        if folded_fig:
                            st.plotly_chart(folded_fig, use_container_width=True)
                        else:
                            st.warning("Could not generate folded light curve plot (missing T0 parameter from backend).")

                        analysis = scout_res.get('coordinator_prediction', {})
                        col1, col2 = st.columns(2)
                        with col1: 
                            st.metric(label="AI Prediction", value=analysis.get('prediction', 'N/A'))
                        with col2: 
                            st.metric(label="Confidence", value=f"{analysis.get('confidence', 0) * 100:.2f}%")
                        st.info(f"**Explanation:** {analysis.get('explanation', 'N/A')}")

                        st.markdown("---")
                        st.subheader("Vetting Statistics")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Signal Detection Efficiency (SDE)", f"{scout_res.get('sde', 0):.2f}")
                        col2.metric("Signal-to-Noise Ratio (SNR)", f"{scout_res.get('snr', 0):.2f}")
                        col3.metric("Odd-Even Mismatch (σ)", f"{scout_res.get('odd_even_mismatch', 0):.2f} σ")
                        col4.metric("False Alarm Probability", f"{scout_res.get('false_alarm_probability', 0):.1e}")

                        st.subheader("Transit Properties")
                        col1, col2, col3, col4 = st.columns(4)
                        period = scout_res.get('period', 0)
                        period_unc = scout_res.get('period_uncertainty', 0)
                        col1.metric("Orbital Period (days)", f"{period:.5f} ± {period_unc:.5f}")
                        col2.metric("Transit Duration (hours)", f"{scout_res.get('duration', 0)*24:.2f}")
                        col3.metric("Transit Depth (ppm)", f"{scout_res.get('depth', 0) * 1e6:.0f}")
                        transit_count_str = f"{scout_res.get('distinct_transit_count', 0)} / {scout_res.get('transit_count', 0)}"
                        col4.metric("Transits Observed", transit_count_str, help="Number of transits with data / Total expected transits")
                        
                        st.subheader("Calculated Planet Parameters")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Planet Radius (Earth Radii)", f"{scout_res.get('planetary_radius_earth', 0):.2f}")
                        col2.metric("Est. Temperature (Kelvin)", f"{scout_res.get('equilibrium_temp_k', 0):.0f} K")
                        col3.metric("Orbital Distance (AU)", f"{scout_res.get('semi_major_axis_au', 0):.3f} AU")
                        st.markdown("---")

                else:
                    st.warning(f"Analysis complete, but no significant signals were found for {star_id}.")
                
            else:
                st.error(f"Error from API ({response.status_code}): {response.json().get('detail', 'N/A')}")
        except requests.exceptions.Timeout:
            st.error(f"The request timed out after {timeout_seconds} seconds. Try increasing the timeout or using a faster analysis configuration.")
        except requests.exceptions.RequestException as e:
            st.error(f"Could not connect to the backend API: {e}")