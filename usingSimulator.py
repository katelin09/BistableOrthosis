import numpy as np
from scipy.signal import argrelextrema
import customizeMjcModel as customize
import mjcSimulator as simulation
import matplotlib.pyplot as plt

def update_model_parameters(x):
    customize.vals['beamA'] = x[0]
    customize.vals['beamC'] = x[1]
    customize.vals['theta'] = x[2]
    customize.vals['tendonThickness'] = x[3]
    customize.vals['tendonWidth'] = x[4]
    customize.vals['hingeLength'] = x[5]
    customize.vals['hingeThickness'] = x[6]

    return customize.generate_model(customize.vals, saved_file="intermediate_files/updated_model.xml")


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


def get_torques(x):
    file = update_model_parameters(x)
    _, _, torque_curve = simulation.simulate(file)
    torqueDown, torqueUp = extract_torques(torque_curve)

    if torqueDown is None or torqueUp is None:
       return None, None
    
    return torqueDown, torqueUp


def generate_and_simulate_variations():
    # Base parameters (all in mm)
    bounds = [
        (12.0, 40.0),  # beamALength
        (10.0, 40.0),  # beamCLength
        (10.0, 40.0),  # theta
        (0.4, 2.0),   # tendonThickness
        (0.4, 5.0),   # tendonWidth
        (1.0, 3.0),  # hingeLength
        (0.4, 1.7),   # hingeThickness
    ]
    base_x = [15.0, 15.0, 25.0, 0.5, 2.0, 2.0, 0.5]  # values within bounds
    variations = {
        'length_a': [(x, 15.0, 25.0, 0.5, 2.0, 2.0, 0.5) for x in [12.0, 25.0, 40.0]],
        'length_c': [(15.0, x, 25.0, 0.5, 2.0, 2.0, 0.5) for x in [10.0, 25.0, 40.0]],
        'theta': [(15.0, 15.0, x, 0.5, 2.0, 2.0, 0.5) for x in [10.0, 25.0, 40.0]],
        'tendon_thickness': [(15.0, 15.0, 25.0, x, 2.0, 2.0, 0.5) for x in [0.3, 0.5, 0.7]],
        'hinge': [(15.0, 15.0, 25.0, 0.5, 2.0, x, 0.5) for x in [1.0, 2.0, 3.0]]
    }

    param_to_bound_index = {
        'length_a': 0,
        'length_c': 1,
        'theta': 2,
        'tendon_thickness': 3,
        'hinge': 5
    }

    for param_name, param_variations in variations.items():
        plt.figure(figsize=(10, 6))
        
        for params in param_variations:
            file = update_model_parameters(params)
            varied_value = params[param_to_bound_index[param_name]]
            angles, _, torques = simulation.simulate(file, byPos=True, 
                               plot=False, animate=True, 
                               animatefile=f"results/{param_name}={varied_value}.gif")
            # Create a list of values for current parameter variation
            all_values = [p[param_to_bound_index[param_name]] for p in param_variations]
            # Get rank of current value (0 = largest, 1 = middle, 2 = smallest)
            rank = sorted(all_values, reverse=True).index(varied_value)
            # Map darkness based on rank
            shades = {0: '0', 1: '0.3', 2: '0.6'}
            darkness = shades[rank]
            plt.plot(angles, torques, color=f'{darkness}', label=f'{param_name}={varied_value}')
        plt.title(f'Torque vs Angle for Different {param_name} Values')
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Torque')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'results/{param_name}_comparison.png')
        plt.close()


generate_and_simulate_variations()