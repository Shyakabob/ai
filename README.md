## AI Learning Project (Python)

This is a small, beginner-friendly AI/ML project using the classic Iris dataset. It includes:

- A simple training and prediction script (`src/ml_demo.py`)
- A Jupyter Notebook walkthrough (`notebooks/iris_walkthrough.ipynb`)
 - A Streamlit playground app (`app/streamlit_app.py`)

### Prerequisites
- Windows with PowerShell
- Python 3.9+ available as `py`

### Setup (Windows PowerShell)
```powershell
cd C:\xampp\htdocs\ai
py -3 -m venv .venv
./.venv/Scripts/Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks scripts, run this once as Administrator and then restart your shell:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Run the simple ML demo
```powershell
python src/ml_demo.py
```
This will:
- Train a Logistic Regression model on Iris
- Print accuracy and a classification report
- Save a model file to `models/iris_logreg.joblib`

### Open the learning notebook
```powershell
jupyter notebook notebooks/iris_walkthrough.ipynb
```

### Run the Streamlit app
```powershell
streamlit run app/streamlit_app.py
```
This opens a browser UI with sliders to input features and compare models.

### Project Structure
```
ai/
  README.md
  requirements.txt
  .gitignore
  src/
    ml_demo.py
  app/
    streamlit_app.py
  notebooks/
    iris_walkthrough.ipynb
  models/  (created at runtime)
```

### Notes
- You can modify `src/ml_demo.py` to try different models (e.g., `SVC`, `RandomForestClassifier`).
- Use the notebook to iterate step-by-step and visualize results.

