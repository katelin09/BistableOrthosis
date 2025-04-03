import numpy as np
from scipy.signal import argrelextrema
import customizeMjcModel as customize
import mjcSimulator as simulation
import os

def update_model_parameters(x, hyperparams=None, saved_file=None):
    """
    Update model parameters and generate new model
    Args:
        x: list of geometric parameters [beamA, beamC, theta, tendonThickness, tendonWidth, hingeLength, hingeThickness]
        hyperparams: dictionary of hyperparameters (optional)
        saved_file: path to save the updated model (optional)
    """
    # Create geometric parameters dictionary
    geoparams = {
        'beamA': x[0],
        'beamC': x[1],
        'theta': x[2],
        'tendonThickness': x[3],
        'tendonWidth': x[4],
        'hingeLength': x[5],
        'hingeThickness': x[6],
        'beamB': customize.DEFAULT_GEOPARAMS['beamB'],  # Use default for beamB
        'hingeWidth': customize.DEFAULT_GEOPARAMS['hingeWidth']  # Use default for hingeWidth
    }

    # Use default hyperparameters if none provided
    if hyperparams is None:
        hyperparams = customize.DEFAULT_HYPERPARAMS

    # If no saved_file specified, use default path
    if saved_file is None:
        saved_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "updated_model.xml")
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(saved_file), exist_ok=True)

    return customize.generate_model(geoparams, hyperparams, saved_file=saved_file)


def extract_torques(torque_curve):
    # Use scipy.signal.argrelextrema to identify local extrema
    max_indices = argrelextrema(torque_curve, np.greater)[0]
    min_indices = argrelextrema(torque_curve, np.less)[0]

    if len(max_indices) == 0 or len(min_indices) == 0:
       return None, None
    
    torqueDown = torque_curve[max_indices[0]]
    torqueUp = torque_curve[min_indices[0]]
    
    torqueDown = float(torqueDown[0])
    torqueUp = float(torqueUp[0])
    
    return torqueDown, torqueUp


def get_torques(x, hyperparams, output_dir=None):
    """
    Get torques for given parameters
    Args:
        x: list of geometric parameters
        hyperparams: dictionary of hyperparameters (optional)
        output_dir: directory to save intermediate files (optional)
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
    os.makedirs(output_dir, exist_ok=True)
    
    mjc_model_file, scaled_params = update_model_parameters(x, hyperparams, saved_file=os.path.join(output_dir, "temp_model.xml"))
    _, _, torque_curve = simulation.simulate(mjc_model_file, nonLinear=scaled_params["nonLinearStiffness"], scaleFactor=scaled_params["scaleFactor"])
    return extract_torques(torque_curve)

def main():
    # Test parameters
    x = [30, 30, 30, 1.0, 1.6, 2.0, 1.0]  # Example parameters
    torqueDown, torqueUp = get_torques(x)
    print(f"Torque Down: {torqueDown:.2f} Nmm")
    print(f"Torque Up: {torqueUp:.2f} Nmm")

if __name__ == "__main__":
    main()