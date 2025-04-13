import numpy as np
from joblib import load
from scipy.optimize import differential_evolution
import sys
import os
from datetime import datetime
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import FunctionTransformer

# Add simulator to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(parent_dir, "simulator"))

import usingSimulator
import mjcSimulator as simulation
import customizeMjcModel
import json


def load_hyperparameters(hyperparams_path=None):
    if hyperparams_path is None:
        hyperparams_path = os.path.join(parent_dir, "hyperparameter_tuning", 
                                      "optimized_params.json")
    with open(hyperparams_path, 'r') as f:
        return json.load(f)

def optimized_through_trained_model(x0, target_angle, bounds, target_torqueDown, 
                                  target_torqueUp, model_dir=None):
    print("\nStarting optimization through trained model...")
    print("target_torqueDown: ", target_torqueDown)
    print("target_torqueUp: ", target_torqueUp)

    # load models
    if model_dir is None:
        model_dir = os.path.join(current_dir, "trained_model_from_simu_data")
    print(f"Loading models from: {model_dir}")
    
    model_down = load(os.path.join(model_dir, 'model_down_2.joblib'))
    model_up = load(os.path.join(model_dir, 'model_up_2.joblib'))
    poly = load(os.path.join(model_dir, 'poly_transformer_2.joblib'))

    # Initialize x with natural angle
    x = x0.copy()
    x[2] = target_angle
    
    def objective(x):
        # Enforce beamALength = beamCLength
        x[1] = x[0]  # Set beamCLength equal to beamALength
        x_reshaped = np.array(x).reshape(1, -1)
        x_poly = poly.transform(x_reshaped)
        td = float(model_down.predict(x_poly)[0])
        tu = float(model_up.predict(x_poly)[0])
        return (td - target_torqueDown)**2 + (tu - target_torqueUp)**2
    
    # Modify bounds to use the same range for both beamALength and beamCLength
    bounds = bounds.copy()
    bounds[1] = bounds[0]  # Set beamCLength bounds equal to beamALength bounds
    
    result = differential_evolution(objective, bounds, popsize=100, 
                                  maxiter=1000, tol=1e-3)
    print(f"Optimization completed. Success: {result.success}, Message: {result.message}")
    optimal_x = result.x
    optimal_x[1] = optimal_x[0]  # Ensure final solution also has beamALength = beamCLength

    # Verify results with reshaped input
    optimal_x_reshaped = np.array(optimal_x).reshape(1, -1)
    optimal_x_poly = poly.transform(optimal_x_reshaped)
    td = float(model_down.predict(optimal_x_poly)[0])
    tu = float(model_up.predict(optimal_x_poly)[0])
    
    return optimal_x, td, tu

def customize_target(target_angle, target_torqueDown, target_torqueUp, 
                    model_dir=None, output_dir=None):
    bounds = [
        (25.0, 50.0),  # beamALength
        (25.0, 50.0),  # beamCLength
        (10.0, 40.0),  # theta
        (0.4, 1.0),   # tendonThickness
        (0.4, 2.0),   # tendonWidth
        (1.2, 2.0),  # hingeLength
        (0.4, 1.0),   # hingeThickness
    ]
    name_of_bounds = ["beamALength", "beamCLength", "thetaDeg", "tendonThickness", 
                     "tendonWidth", "hingeLength", "hingeThickness"]


    bounds_constrained = bounds.copy()
    bounds_constrained[2] = (target_angle, target_angle)
    # Initial guess
    x0 = [25.0, 30.0, target_angle, 1.0, 1.6, 2.0, 1.0]
    
    x, td, tu = optimized_through_trained_model(x0, target_angle, bounds_constrained,
                                              target_torqueDown, target_torqueUp, 
                                              model_dir)

    print(f"Optimized X: {x}")
    print(f"Model_prediction: Torque Down = {td}, Torque Up = {tu}")
    if output_dir is None:
        output_dir = os.path.join(current_dir, "results")
    os.makedirs(output_dir, exist_ok=True)

    hyperparams = load_hyperparameters()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"model_{timestamp}.xml"
    mjc_model_file, scaled_params = usingSimulator.update_model_parameters(
        x, hyperparams=hyperparams, 
        saved_file=os.path.join(output_dir, model_filename))
    
    _, _, torque_curve = simulation.simulate(mjc_model_file)
    torqueDown, torqueUp = usingSimulator.extract_torques(torque_curve)
    print(f"Simulation: Torque Down = {torqueDown}, Torque Up = {torqueUp}")
    
    def update_parameters(x):
        """
        Create geometric parameters dictionary from optimization results
        """
        geoparams = {
            'beamA': x[0],
            'beamC': x[1],
            'theta': x[2],
            'tendonThickness': x[3],
            'tendonWidth': x[4],
            'hingeLength': x[5],
            'hingeThickness': x[6],
            'beamB': customizeMjcModel.DEFAULT_GEOPARAMS['beamB'],  # Use default for beamB
            'hingeWidth': customizeMjcModel.DEFAULT_GEOPARAMS['hingeWidth']  # Use default for hingeWidth
        }
        return geoparams
    
    geometry_vals = update_parameters(x)

    # Save optimization results
    results = {
        'timestamp': timestamp,
        'geometry_vals': geometry_vals,
        'torqueDown': float(torqueDown),
        'torqueUp': float(torqueUp),
        'model_file': model_filename
    }

    results_filename = f"results_{timestamp}.json"
    with open(os.path.join(output_dir, results_filename), 'w') as f:
        json.dump(results, f, indent=4)

    return x, float(torqueDown), float(torqueUp), mjc_model_file, geometry_vals, torque_curve

def customize(input_params, strength="default", model_dir=None, output_dir=None):
    if (input_params['ptorqueExtend'] > input_params['atorqueBend'] or 
        input_params['naturalAngle'] < 0 or input_params['naturalAngle'] > 90):
        raise ValueError("Invalid input parameters: ptorqueExtend must be less than "
                       "atorqueBend, and naturalAngle must be between 0 and 90 degree.")

    target_torqueDown = (input_params['ptorqueExtend'] + input_params['atorqueBend']) / 2.0
    target_torqueUp = -input_params['atorqueExtend'] / 2.0
    if strength == "soft":
        target_torqueDown = (input_params['ptorqueExtend'] + 
                           (input_params['atorqueBend'] - input_params['ptorqueExtend']) / 3.0)
        target_torqueUp = -input_params['atorqueExtend'] / 3.0
    elif strength == "stiff":
        target_torqueDown = (input_params['ptorqueExtend'] + 
                           (input_params['atorqueBend'] - input_params['ptorqueExtend']) * 2.0 / 3.0)
        target_torqueUp = -input_params['atorqueExtend'] * 2.0 / 3.0
    elif strength == "easyUp":
        target_torqueDown = (input_params['ptorqueExtend'] + 
                           (input_params['atorqueBend'] - input_params['ptorqueExtend']) * 2.0 / 3.0)
        target_torqueUp = -input_params['atorqueExtend'] / 3.0
    elif strength == "easyDown":
        target_torqueDown = (input_params['ptorqueExtend'] + 
                           (input_params['atorqueBend'] - input_params['ptorqueExtend']) / 3.0)
        target_torqueUp = -input_params['atorqueExtend'] * 2.0 / 3.0
          
    X = customize_target(input_params['naturalAngle'], target_torqueDown, 
                        target_torqueUp, model_dir=model_dir, output_dir=output_dir)

    return X


if __name__ == "__main__":
    customize_target(target_angle=30, target_torqueDown=50, target_torqueUp=-10)

