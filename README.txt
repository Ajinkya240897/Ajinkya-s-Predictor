
Ajinkya's Predictor - Final Release (RF + Ridge, Risk Controls)
Files:
- ajinkya_predictor_final_release.py : Streamlit app
- requirements.txt : minimal requirements (no LightGBM)
- README.txt : this file

Notes:
- Adds VaR and CVaR risk controls (historical and parametric) computed on daily returns.
- UI shows required fields only, with appealing styling. Title is 'Ajinkya's Predictor'.
- Designed to deploy on Streamlit Cloud (avoid heavy native-compiled deps).
