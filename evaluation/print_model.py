import os
import numpy as np
from joblib import load
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

def print_polynomial_coefficients(model, model_name):
    """Print polynomial coefficients in a readable format"""
    # Get coefficients from loaded model
    poly_features = model.named_steps['polynomialfeatures']
    linear_reg = model.named_steps['linearregression']
    coefficients = linear_reg.coef_
    intercept = linear_reg.intercept_
    print(f"\n{model_name} Model:")
    print(f"Intercept: {intercept:.6f}")
    
    # Get feature names
    feature_names = poly_features.get_feature_names_out(['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'])
    
    # Print each coefficient with its corresponding term
    for i, name in enumerate(feature_names):
        coef = coefficients[i]
        if abs(coef) > 10:  # Only print significant coefficients
            print(f"{name}: {coef:.6f}")
    
def main():
    # Load models from optimizer/trained_model_from_simu_data directory
    model_dir = os.path.join('..', 'optimizer', 'trained_model_from_simu_data')

    model_down = load(os.path.join(model_dir, 'model_down.joblib'))
    model_up = load(os.path.join(model_dir, 'model_up.joblib'))
   
    print_polynomial_coefficients(model_down, "model_down")
    print_polynomial_coefficients(model_up, "model_up")

if __name__ == "__main__":
    main()
