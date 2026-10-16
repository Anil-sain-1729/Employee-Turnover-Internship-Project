@echo off
python -m pip install -r requirements.txt
python train_model.py
streamlit run app.py
pause
