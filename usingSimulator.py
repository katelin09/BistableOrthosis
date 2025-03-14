import numpy as np
from scipy.signal import argrelextrema
import customizeMjcModel as customize
import mjcSimulator as simulation

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