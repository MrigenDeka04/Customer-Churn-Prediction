import os
import io
import traceback
import base64
import numpy as np
import pandas as pd
import pickle
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Rectangle
import shap
from flask import Flask, request, render_template

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

# Load the models
with open(os.path.join(BASE_DIR, 'model2.pkl'), 'rb') as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, 'survivemodel2.pkl'), 'rb') as f:
    survmodel = pickle.load(f)

def get_int(form, name, default=0):
    v = form.get(name)
    if v is None or v == '': return default
    try: return int(v)
    except Exception: return default

def get_float(form, name, default=0.0):
    v = form.get(name)
    if v is None or v == '': return default
    try: return float(v)
    except Exception: return default

def get_checkbox(form, name):
    return 1 if form.get(name) is not None else 0

@app.route('/')
def home():
    # Pass an empty form_data dictionary on initial load
    return render_template('index.html', form_data={})

@app.route('/predict', methods=['POST'])
def predict():
    # --- Parse all inputs from the form ---
    SeniorCitizen = get_checkbox(request.form, "SeniorCitizen")
    Partner = get_checkbox(request.form, "Partner")
    Dependents = get_checkbox(request.form, "Dependents")
    PaperlessBilling = get_checkbox(request.form, "PaperlessBilling")
    PhoneService = get_checkbox(request.form, "PhoneService")
    MultipleLines = get_checkbox(request.form, "MultipleLines")
    OnlineSecurity = get_checkbox(request.form, "OnlineSecurity")
    OnlineBackup = get_checkbox(request.form, "OnlineBackup")
    DeviceProtection = get_checkbox(request.form, "DeviceProtection")
    TechSupport = get_checkbox(request.form, "TechSupport")
    StreamingTV = get_checkbox(request.form, "StreamingTV")
    StreamingMovies = get_checkbox(request.form, "StreamingMovies")
    gender = get_int(request.form, "gender", 0)
    internet_service = get_int(request.form, "InternetService", 0)
    contract_val = get_int(request.form, "Contract", 0)
    payment = get_int(request.form, "PaymentMethod", 0)
    MonthlyCharges = get_float(request.form, "MonthlyCharges", 0.0)
    Tenure = get_int(request.form, "Tenure", 0)
    
    # --- Create one-hot encoded features ---
    TotalCharges = MonthlyCharges * Tenure
    InternetService_Fiberoptic = 1 if internet_service == 2 else 0
    InternetService_No = 1 if internet_service == 0 else 0
    Contract_Oneyear = 1 if contract_val == 1 else 0
    Contract_Twoyear = 1 if contract_val == 2 else 0
    PaymentMethod_CreditCard = 1 if payment == 1 else 0
    PaymentMethod_ElectronicCheck = 1 if payment == 2 else 0
    PaymentMethod_MailedCheck = 1 if payment == 3 else 0
    if PhoneService == 0: MultipleLines = 0

    # --- Assemble feature lists ---
    # List for the CHURN PREDICTION model (23 features)
    churn_model_features = [
        gender, SeniorCitizen, Partner, Dependents, Tenure, PhoneService,
        MultipleLines, OnlineSecurity, OnlineBackup, DeviceProtection,
        TechSupport, StreamingTV, StreamingMovies, PaperlessBilling,
        MonthlyCharges, TotalCharges, InternetService_Fiberoptic, InternetService_No,
        Contract_Oneyear, Contract_Twoyear, PaymentMethod_CreditCard,
        PaymentMethod_ElectronicCheck, PaymentMethod_MailedCheck
    ]
    
    # List for the SURVIVAL model (22 features - EXCLUDING TotalCharges)
    survival_model_features = [
        gender, SeniorCitizen, Partner, Dependents, Tenure, PhoneService,
        MultipleLines, OnlineSecurity, OnlineBackup, DeviceProtection,
        TechSupport, StreamingTV, StreamingMovies, PaperlessBilling,
        MonthlyCharges, InternetService_Fiberoptic, InternetService_No,
        Contract_Oneyear, Contract_Twoyear, PaymentMethod_CreditCard,
        PaymentMethod_ElectronicCheck, PaymentMethod_MailedCheck
    ]

    # Convert to NumPy arrays with the correct shape
    churn_features_np = np.array(churn_model_features).reshape(1, -1)
    surv_feats = np.array(survival_model_features).reshape(1, -1)

    # -------- model prediction --------
    try:
        prediction_proba = model.predict_proba(churn_features_np)
        output = float(prediction_proba[0, 1])
    except Exception:
        traceback.print_exc()
        output = 0.0

    # -------- SHAP force plot --------
    try:
        feature_names = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
            'StreamingMovies', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges', 'InternetService_Fiber optic', 
            'InternetService_No', 'Contract_One year', 'Contract_Two year', 'PaymentMethod_Credit card (automatic)',
            'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
        ]
        explainer = shap.Explainer(model, feature_names=feature_names)
        shap_values_obj = explainer(churn_features_np)

        shap_img = io.BytesIO()
        
        base_value = shap_values_obj.base_values[0, 1]
        shap_values_for_plot = shap_values_obj.values[0, :, 1]
        
        shap.plots.force(base_value, shap_values_for_plot, matplotlib=True, show=False, feature_names=feature_names).savefig(
            shap_img, bbox_inches="tight", format='png'
        )
        plt.close('all')
        shap_img.seek(0)
        shap_url = base64.b64encode(shap_img.getvalue()).decode()

    except Exception:
        print("SHAP generation failed:")
        traceback.print_exc()
        # Fallback image generation
        buf = io.BytesIO()
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "SHAP unavailable", ha='center', va='center')
        ax.axis('off')
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        shap_url = base64.b64encode(buf.getvalue()).decode()

    # -------- Survival & hazard plots --------
    try:
        hazard_img = io.BytesIO()
        fig, ax = plt.subplots()
        survmodel.predict_cumulative_hazard(surv_feats).plot(ax=ax, color='red')
        ax.axvline(x=Tenure, color='blue', linestyle='--')
        ax.set_xlabel('Tenure')
        ax.set_ylabel('Cumulative Hazard')
        ax.set_title('Cumulative Hazard Over Time')
        plt.tight_layout()
        plt.savefig(hazard_img, format='png')
        plt.close(fig)
        hazard_img.seek(0)
        hazard_url = base64.b64encode(hazard_img.getvalue()).decode()

        surv_img = io.BytesIO()
        fig, ax = plt.subplots()
        survmodel.predict_survival_function(surv_feats).plot(ax=ax, color='red')
        ax.axvline(x=Tenure, color='blue', linestyle='--')
        ax.set_xlabel('Tenure')
        ax.set_ylabel('Survival Probability')
        ax.set_title('Survival Probability Over Time')
        plt.tight_layout()
        plt.savefig(surv_img, format='png')
        plt.close(fig)
        surv_img.seek(0)
        surv_url = base64.b64encode(surv_img.getvalue()).decode()

        life = survmodel.predict_survival_function(surv_feats).reset_index()
        life.columns = ['Tenure', 'Probability']
        max_life = life.Tenure[life.Probability > 0.1].max()
        if np.isnan(max_life): max_life = Tenure
        CLTV = round(float(max_life) * MonthlyCharges, 2)
    except Exception:
        print("Survival model failed:")
        traceback.print_exc()
        hazard_url = ""
        surv_url = ""
        CLTV = round(MonthlyCharges * Tenure, 2)

    # --- Gauge image for probability (No changes needed here) ---
    def gauge(Probability=1.0):
        lbls, colors = ['LOW', 'MEDIUM', 'HIGH', 'EXTREME'], ['#007A00', '#0063BF', '#FFCC00', '#ED1C24']
        gauge_img = io.BytesIO()
        fig, ax = plt.subplots()
        start = np.linspace(0, 180, len(lbls) + 1)
        end = start[1:]
        start = start[:-1]
        mid_points = start + (end - start) / 2.
        for ang_range, c in zip(np.c_[start, end], colors[::-1]):
            ax.add_patch(Wedge((0., 0.), .4, ang_range[0], ang_range[1], facecolor='w', lw=2))
            ax.add_patch(Wedge((0., 0.), .4, ang_range[0], ang_range[1], width=0.10, facecolor=c, lw=2, alpha=0.5))
        for mid, lab in zip(mid_points, lbls[::-1]):
            ax.text(0.35 * np.cos(np.radians(mid)), 0.35 * np.sin(np.radians(mid)), lab, ha='center', va='center', fontsize=8)
        ax.add_patch(Rectangle((-0.4, -0.1), 0.8, 0.1, facecolor='w', lw=2))
        ax.text(0, -0.05, f'Churn Probability {round(Probability,2)}', ha='center', va='center', fontsize=10, fontweight='bold')
        pos = (1 - Probability) * 180
        ax.arrow(0, 0, 0.225 * np.cos(np.radians(pos)), 0.225 * np.sin(np.radians(pos)), width=0.03, head_width=0.06, head_length=0.07, fc='k', ec='k')
        ax.add_patch(Circle((0, 0), radius=0.02, facecolor='k'))
        ax.set_frame_on(False)
        ax.axes.set_xticks([])
        ax.axes.set_yticks([])
        ax.axis('equal')
        plt.tight_layout()
        plt.savefig(gauge_img, format='png')
        plt.close(fig)
        gauge_img.seek(0)
        return base64.b64encode(gauge_img.getvalue()).decode()

    gauge_url = gauge(Probability=output)
    prediction_text = f'Churn probability is {round(output, 4)} and Expected Life Time Value is ${CLTV}'

    # *** NEW: Create a dictionary of the form inputs to pass back to the template ***
    form_data = {
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "PaperlessBilling": PaperlessBilling,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "gender": gender,
        "InternetService": internet_service,
        "Contract": contract_val,
        "PaymentMethod": payment,
        "MonthlyCharges": MonthlyCharges,
        "Tenure": Tenure
    }

    # *** MODIFIED: Pass the form_data dictionary to the template ***
    return render_template('index.html',
                           prediction_text=prediction_text,
                           url_1=gauge_url,
                           url_2=shap_url,
                           url_3=hazard_url,
                           url_4=surv_url,
                           form_data=form_data) # Pass the collected form data

if __name__ == "__main__":
    app.run(debug=True)