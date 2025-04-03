import numpy as np
import os
import pandas as pd
from joblib import load

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Define geometric parameters for each model (copied from tuning.py)
model_params = {
    'model1': {
        'beamA': 30, 'beamC': 30, 'theta': 30,
        'tendonThickness': 1, 'tendonWidth': 1.6, 'hingeLength': 2,
        'beamB': 25,  # from DEFAULT_GEOPARAMS
        'hingeThickness': 0.8,  # from DEFAULT_GEOPARAMS
        'hingeWidth': 2.4  # from DEFAULT_GEOPARAMS
    },
    'model2': {
        'beamA': 20, 'beamC': 20, 'theta': 30,
        'tendonThickness': 1, 'tendonWidth': 1.6, 'hingeLength': 2,
        'beamB': 25,  # from DEFAULT_GEOPARAMS
        'hingeThickness': 0.8,  # from DEFAULT_GEOPARAMS
        'hingeWidth': 2.4  # from DEFAULT_GEOPARAMS
    },
    'model3': {
        'beamA': 30, 'beamC': 30, 'theta': 40,
        'tendonThickness': 1, 'tendonWidth': 1.6, 'hingeLength': 2,
        'beamB': 25,  # from DEFAULT_GEOPARAMS
        'hingeThickness': 0.8,  # from DEFAULT_GEOPARAMS
        'hingeWidth': 2.4  # from DEFAULT_GEOPARAMS
    },
    'model4': {
        'beamA': 30, 'beamC': 30, 'theta': 30,
        'tendonThickness': 0.5, 'tendonWidth': 1.6, 'hingeLength': 2,
        'beamB': 25,  # from DEFAULT_GEOPARAMS
        'hingeThickness': 0.8,  # from DEFAULT_GEOPARAMS
        'hingeWidth': 2.4  # from DEFAULT_GEOPARAMS
    },
    'model5': {
        'beamA': 30, 'beamC': 30, 'theta': 30,
        'tendonThickness': 1, 'tendonWidth': 1.6, 'hingeLength': 3,
        'beamB': 25,  # from DEFAULT_GEOPARAMS
        'hingeThickness': 0.8,  # from DEFAULT_GEOPARAMS
        'hingeWidth': 2.4  # from DEFAULT_GEOPARAMS
    }
}

def load_trained_models(model_dir=None):
    if model_dir is None:
        model_dir = os.path.join(parent_dir, "optimizer", "trained_model_from_simu_data")
    model_down = load(os.path.join(model_dir, 'model_down.joblib'))
    model_up = load(os.path.join(model_dir, 'model_up.joblib'))
    return model_down, model_up

def find_critical_points(x, y):
    # Check if array has enough points
    if len(y) < 3:  # Need at least 3 points to calculate slope
        return (None, None), (None, None)
    
    peak = (None, None)
    valley = (None, None)
    
    # Find valley (global minimum)
    min_idx = np.argmin(y)
    valley = (x[min_idx], y[min_idx].item())
    
    # Find peak (maximum point before the valley)
    if min_idx > 0:  # Only look for peak if valley isn't at start or end
        peak_idx = np.argmax(y[:min_idx])
        peak = (x[peak_idx], y[peak_idx].item())
    
    return peak, valley

def process_real_data(real_data_path, model_dir=None):
    # Load trained models
    model_down, model_up = load_trained_models(model_dir)
    
    # Load real data from CSV
    real_data = pd.read_csv(real_data_path)
    angles = real_data.iloc[:, 0].values  # First column for angles
    torques = real_data.iloc[:, 4].values  # Fourth column for torques
    
    # Find critical points from real data
    (tdown_angle, tdown_torque), (tup_angle, tup_torque) = find_critical_points(angles, torques)
    
    if tdown_angle is None or tup_angle is None:
        print("Warning: Could not find critical points in real data")
        return None
    
    # Get model number from filename (e.g., "id1.csv" -> 1)
    model_num = int(os.path.basename(real_data_path).replace('id', '').replace('.csv', ''))
    
    # Get geometric parameters for this model
    geo_params = model_params[f'model{model_num}']
    
    # Prepare input for model prediction (using only the 7 parameters from training)
    x = np.array([
        geo_params['beamA'],      # beamALength
        geo_params['beamC'],      # beamCLength
        geo_params['theta'],      # theta
        geo_params['tendonThickness'],  # tendonThickness
        geo_params['tendonWidth'],      # tendonWidth
        geo_params['hingeLength'],      # hingeLength
        geo_params['hingeThickness']    # hingeThickness
    ])
    x_reshaped = x.reshape(1, -1)
    
    # Get model predictions
    tdown_pred = float(model_down.predict(x_reshaped)[0])
    tup_pred = float(model_up.predict(x_reshaped)[0])
    
    # Calculate errors
    tdown_error = abs(tdown_pred - tdown_torque)
    tup_error = abs(tup_pred - tup_torque)
    tdown_error_percent = (tdown_error / abs(tdown_torque)) * 100
    tup_error_percent = (tup_error / abs(tup_torque)) * 100
    
    # Calculate MSE and RMSE
    tdown_mse = (tdown_pred - tdown_torque) ** 2
    tup_mse = (tup_pred - tup_torque) ** 2
    tdown_rmse = np.sqrt(tdown_mse)
    tup_rmse = np.sqrt(tup_mse)
    
    return {
        'real_data': {
            'tdown': (tdown_angle, tdown_torque),
            'tup': (tup_angle, tup_torque)
        },
        'model_predictions': {
            'tdown': tdown_pred,
            'tup': tup_pred
        },
        'errors': {
            'tdown_abs': tdown_error,
            'tup_abs': tup_error,
            'tdown_percent': tdown_error_percent,
            'tup_percent': tup_error_percent,
            'tdown_rmse': tdown_rmse,
            'tup_rmse': tup_rmse
        }
    }

def main():
    # Process all models
    total_tdown_error = 0
    total_tup_error = 0
    total_tdown_error_percent = 0
    total_tup_error_percent = 0
    total_tdown_rmse = 0
    total_tup_rmse = 0
    num_models = 0
    
    for model_num in range(1, 6):
        print(f"\nProcessing model {model_num}")
        real_data_path = os.path.join(parent_dir, "hyperparameter_tuning", "processed_real_data", f"id{model_num}.csv")
        results = process_real_data(real_data_path)
        
        if results:
            print("Real data critical points:")
            print(f"tdown: {results['real_data']['tdown']}")
            print(f"tup: {results['real_data']['tup']}")
            print("\nModel predictions:")
            print(f"tdown: {results['model_predictions']['tdown']}")
            print(f"tup: {results['model_predictions']['tup']}")
            print("\nError metrics:")
            print(f"tdown absolute error: {results['errors']['tdown_abs']:.4f}")
            print(f"tup absolute error: {results['errors']['tup_abs']:.4f}")
            print(f"tdown percentage error: {results['errors']['tdown_percent']:.2f}%")
            print(f"tup percentage error: {results['errors']['tup_percent']:.2f}%")
            print(f"tdown RMSE: {results['errors']['tdown_rmse']:.4f}")
            print(f"tup RMSE: {results['errors']['tup_rmse']:.4f}")
            
            # Accumulate errors for average calculation
            total_tdown_error += results['errors']['tdown_abs']
            total_tup_error += results['errors']['tup_abs']
            total_tdown_error_percent += results['errors']['tdown_percent']
            total_tup_error_percent += results['errors']['tup_percent']
            total_tdown_rmse += results['errors']['tdown_rmse']
            total_tup_rmse += results['errors']['tup_rmse']
            num_models += 1
    
    # Print average errors
    if num_models > 0:
        print("\nAverage Error Metrics Across All Models:")
        print(f"Average tdown absolute error: {total_tdown_error/num_models:.4f}")
        print(f"Average tup absolute error: {total_tup_error/num_models:.4f}")
        print(f"Average tdown percentage error: {total_tdown_error_percent/num_models:.2f}%")
        print(f"Average tup percentage error: {total_tup_error_percent/num_models:.2f}%")
        print(f"Average tdown RMSE: {total_tdown_rmse/num_models:.4f}")
        print(f"Average tup RMSE: {total_tup_rmse/num_models:.4f}")

if __name__ == "__main__":
    main()



# For tdown (torque down):
# Average absolute error: 4.56 N·m
# Average percentage error: 18.72%
# Average RMSE: 4.56 N·m
# For tup (torque up):
# Average absolute error: 1.91 N·m
# Average percentage error: 36.09%
# Average RMSE: 1.91 N·m