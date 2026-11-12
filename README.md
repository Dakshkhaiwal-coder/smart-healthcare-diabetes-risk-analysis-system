# Smart Healthcare Risk Analysis System
## Diabetes Risk Prediction Using Machine Learning

**Students:** Daksh (2554670), Deepanshu (2554675)  
**Guide:** Manpreet Ma'am  
**Institution:** Chandigarh Group of Colleges, College of Engineering, Landran

### Project technology
Python, pandas, scikit-learn, Logistic Regression, joblib, Streamlit, GitHub.

### Run in VS Code (Windows)
1. Install Python 3.11 and VS Code.
2. Extract this ZIP and open the folder in VS Code.
3. Open Terminal > New Terminal.
4. Create a virtual environment:
   `python -m venv .venv`
5. Activate it in PowerShell:
   `.venv\Scripts\Activate.ps1`
   (If blocked, use Command Prompt: `.venv\Scripts\activate.bat`)
6. Install libraries:
   `pip install -r requirements.txt`
7. Train the model and download the dataset automatically:
   `python train_model.py`
8. Start the Streamlit website:
   `streamlit run app.py`
9. Open the local URL shown in terminal, usually http://localhost:8501.

### GitHub upload
Create a GitHub repository named `smart-healthcare-risk-analysis`. Then run these commands in the project folder, replacing YOUR-USERNAME:

```bash
git init
git add .
git commit -m "Smart Healthcare Diabetes Risk Analysis"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/smart-healthcare-risk-analysis.git
git push -u origin main
```

Do not upload `.venv`, credentials, or private patient information. The public sample dataset is downloaded by the training script; review its source and licence before redistributing it.

### Streamlit deployment
1. Sign in at https://share.streamlit.io/ using GitHub.
2. Choose Create app.
3. Select your repository, branch `main`, and main file `app.py`.
4. Click Deploy.
5. Ensure `diabetes_model.pkl` is committed to GitHub (the training script creates it).
6. Test the deployed page.

### Methodology
The project uses the Pima Indians Diabetes dataset. It uses eight features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction and Age. Outcome is the target column. Zero-coded values in selected clinical measurements are treated as missing, median imputation is applied, features are standardized, and Logistic Regression is trained using a stratified 80:20 train-test split. Actual test metrics are printed when the model is trained.

### Important limitation
This is a classroom demonstration using a limited historical dataset, not a medical device. It must not be used to diagnose, rule out diabetes, or make treatment decisions.
