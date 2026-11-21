# PCA-GMM Streamlit project

## Run locally
1. Install Python 3.10+.
2. Open a terminal in this folder.
3. Create a virtual environment:
   - Windows: `py -m venv .venv` then `.venv\\Scripts\\activate`
   - macOS/Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
4. Install packages: `pip install -r requirements.txt`
5. Start dashboard: `streamlit run app.py`
6. Open the local URL printed by Streamlit (normally http://localhost:8501).

## Demo
Select Normal, then run through Temperature bias, Pressure drift, Flow drop and Concentration spike. Change fault strength and compare sensor trends, T²/Q alarm charts, diagnosis probabilities and PCA score plot.

## Implementation notes
Four synthetic correlated sensors are generated. PCA reduces them to three components. A two-component Gaussian mixture is fitted to normal PCA scores. T² and Q limits use the 99th percentile of the normal training statistics. Fault diagnosis uses LDA followed by class-conditional one-component Gaussian mixtures and Bayesian class priors.

## Limitations
This is simulated data and an educational DA-GMM-inspired implementation, not a validated reproduction of every equation in the source paper. It does not read real sensors or control equipment and must not be used for safety decisions.
