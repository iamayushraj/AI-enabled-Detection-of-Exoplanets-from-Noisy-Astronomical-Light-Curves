# ExoplanetAI: AI-Driven Detection of Exoplanets from Noisy Light Curves
**Project Report**

## 1. Introduction & Objectives
The ExoplanetAI project is an end-to-end artificial intelligence pipeline designed to robustly identify and classify exoplanet transit signals from noisy stellar light curves. Given the vast amount of time-series photometric data from missions like TESS (Transiting Exoplanet Survey Satellite), manual identification is computationally infeasible. Our system automates this process by combining classical astrophysical signal processing (Box Least Squares) with a modern hybrid Machine Learning architecture (XGBoost + 1D Convolutional Neural Networks) to detect, classify, and parameterize astrophysical signals.

## 2. Methodology & Approach
Our pipeline follows a structured, multi-stage methodology to transform raw, noisy flux data into high-confidence transit classifications:

### 2.1. Data Preprocessing
Raw light curves suffer from instrumental noise, cosmic ray hits, and long-term stellar variability. 
* **Outlier Removal:** We utilize iterative $\sigma$-clipping (e.g., $3\sigma$ threshold) to filter out extreme flux spikes and anomalies.
* **Detrending:** We apply localized polynomial detrending (and median filtering) to flatten the light curve, effectively removing long-term astrophysical trends (like stellar rotation) while preserving the high-frequency transit dips.

### 2.2. Transit Detection
* **Box Least Squares (BLS):** We use the Astropy BLS algorithm to scan the flattened light curve for periodic, box-shaped dips. The BLS periodogram identifies the most dominant period, duration, and transit epoch ($T_0$).
* **Phase Folding:** Using the best-fit period and $T_0$, the time-series is phase-folded, aligning all potential transits into a single characteristic dip to enhance the Signal-to-Noise Ratio (SNR).

### 2.3. Machine Learning Classification
We implement a **Hybrid ML Architecture** to mitigate false positives (like Eclipsing Binaries or stellar variability):
* **XGBoost Classifier:** Trained on 14+ engineered tabular features extracted from the light curve and BLS periodogram (e.g., Signal Detection Efficiency (SDE), SNR, even/odd transit depth ratios, and transit shape symmetry). XGBoost handles non-linear tabular relationships exceptionally well.
* **1D Convolutional Neural Network (CNN):** A deep learning model built with PyTorch that ingests the raw phase-folded flux array. The CNN autonomously learns complex spatial features and transit morphologies that are difficult to manually engineer.
* **Ensemble Voting:** The predictions of the XGBoost and CNN models are weighted (e.g., 60% / 40%) to yield a final, highly-confident classification probability.

## 3. Parameter Estimation & Uncertainties
Once a signal is classified as a valid transit, physical parameters are estimated by fitting the folded data:
* **Orbital Period ($P$):** Derived directly from the peak of the BLS power spectrum.
* **Transit Depth ($\delta$):** Calculated as the fractional flux difference between the out-of-transit baseline and the in-transit minimum.
* **Transit Duration ($T_{14}$):** Extracted from the width of the BLS model fit.

### How Uncertainties are Estimated
Uncertainties are rigorously calculated to provide scientific confidence:
* **Period Uncertainty:** Estimated analytically based on the time baseline of the observations ($T_{obs}$) and the SNR, typically modeled as $\sigma_P \approx P^2 / (T_{obs} \cdot \text{SNR})$.
* **Depth & Radius Ratio Uncertainties:** Estimated using standard error propagation. The flux variance outside the transit is calculated, and the depth uncertainty is scaled by $1/\sqrt{N_{in}}$, where $N_{in}$ is the number of data points inside the transit window.
* **Classification Confidence:** The final ensemble ML model outputs a softmax probability distribution across classes, providing a direct confidence percentage (e.g., 94% Transit, 4% Eclipsing Binary, 2% Noise). We also utilize **SHAP (SHapley Additive exPlanations)** to break down exactly *why* the XGBoost model assigned its confidence, ensuring full interpretability of the AI's decision-making process.

## 4. Assumptions Made
* **Strict Periodicity:** The pipeline assumes transits occur at strictly regular intervals, meaning it is less sensitive to systems with high Transit Timing Variations (TTVs) caused by multi-planet gravitational interactions.
* **Spherically Symmetric Stars:** The transit depth calculations assume the host star is a uniform sphere, utilizing simplified limb-darkening approximations during parameter estimation.
* **Polynomial Variability:** We assume that stellar variability occurs on time scales significantly longer than the transit duration, allowing polynomial detrending to separate the two signals without distorting the transit depth.

## 5. Tools and Libraries Used
The solution was built entirely in Python using industry-standard libraries:
* **Astrophysics & Data Processing:** `lightkurve` (TESS data retrieval), `astropy` (BLS & time-series analysis), `numpy`, `pandas`, `scipy` (signal processing).
* **Machine Learning:** `xgboost` (tabular classification), `torch` (PyTorch for 1D CNNs), `scikit-learn` (metrics and data splitting).
* **Interpretability:** `shap` (game-theoretic feature importance).
* **Dashboard & Backend:** `streamlit` (interactive web UI), `fastapi` (REST API), `plotly` (interactive data visualizations).
