import numpy as np
from joblib import load
from scipy.optimize import differential_evolution
import usingSimulator
import mjcSimulator as simulation
import customizeMjcModel


def optimized_through_trained_model(x0, input_params, bounds, target_torqueDown, target_torqueUp):
    # load models
    model_down = load('trained_model_from_simu_data/model_down.joblib')
    model_up = load('trained_model_from_simu_data/model_up.joblib')

    # Initialize x with natural angle
    x = x0.copy()
    x[2] = input_params['naturalAngle']
    

    def objective(x):
        x_reshaped = np.array(x).reshape(1, -1)
        td = float(model_down.predict(x_reshaped)[0])
        tu = float(model_up.predict(x_reshaped)[0])
        return (td - target_torqueDown)**2 + (tu - target_torqueUp)**2
    

    def objective_exp(x):
        x_reshaped = np.array(x).reshape(1, -1)
        td = float(model_down.predict(x_reshaped)[0])
        tu = float(model_up.predict(x_reshaped)[0])
        # return (td - target_torqueDown)**2 + (tu - target_torqueUp)**2
        penalty = 0
        td_lower = input_params['ptorqueExtend']
        td_upper = input_params['atorqueBend']
        penalty = np.exp(td_lower - td) + np.exp(td - td_upper)
        
        # Continuous penalty for torque up (tu)
        tu_lower = -input_params['atorqueExtend']
        tu_upper = 0
        penalty += np.exp(tu_lower - tu) + np.exp(tu - tu_upper)
        return penalty


    #TODO: try different optimization methods
    result = differential_evolution(objective_exp, bounds, popsize=10, maxiter=10)
    optimal_x = result.x

    # Verify results with reshaped input
    optimal_x_reshaped = np.array(optimal_x).reshape(1, -1)
    td = float(model_down.predict(optimal_x_reshaped)[0])
    tu = float(model_up.predict(optimal_x_reshaped)[0])
    simu_td, simu_tu = usingSimulator.get_torques(np.array(optimal_x))
    # print(f"Predicted torque down: {td}")
    # print(f"Predicted torque up: {tu}")
    # print(f"Simulated torque down: {simu_td}")
    # print(f"Simulated torque up: {simu_tu}")
    # print(f"Optimal parameters: {optimal_x}")
    
    return optimal_x


def just_optimize(input_params, bounds, target_torqueDown, target_torqueUp):
    def objective(x):
        td, tu = usingSimulator.get_torques(x)
        if td is None or tu is None:
            return np.inf
        if input_params['atorqueBend'] > td > input_params['ptorqueExtend'] and input_params['atorqueExtend'] < tu < 0:
            return 0
        penalty = (td - target_torqueDown)**2 + (tu - target_torqueUp)**2
        print(f"td: {td}, tu: {tu}, penalty: {penalty}")
        return penalty
    

    def objective_exp(x):
        td, tu = usingSimulator.get_torques(x)
        if td is None or tu is None:
            return np.inf
        if input_params['atorqueBend'] > td > input_params['ptorqueExtend'] and input_params['atorqueExtend'] < tu < 0:
            return 0
        # Calculate distance from desired ranges using exp
        penalty = 0
        td_lower = input_params['ptorqueExtend']
        td_upper = input_params['atorqueBend']
        penalty = np.exp(td_lower - td) + np.exp(td - td_upper)
        tu_lower = -input_params['atorqueExtend']
        tu_upper = 0
        penalty += np.exp(tu_lower - tu) + np.exp(tu - tu_upper)

        print(f"td: {td}, tu: {tu}, penalty: {penalty}")
        return penalty
    

    def objective_heuristic(x):
        td, tu = usingSimulator.get_torques(x)
        if td is None or tu is None:
            return np.inf
        
        penalty = 0
        w1, w2, w3 = 1, 1, 10
        if not input_params['atorqueBend'] > td > input_params['ptorqueExtend']:
            penalty += w1 * (td - target_torqueDown)**2

        if not input_params['atorqueExtend'] < tu < 0:
            penalty += w2 * (tu - target_torqueUp)**2

        if penalty == 0:
            return 0
        
        heuristic = 0
        if td < target_torqueDown: # tendon is too soft
            heuristic += x[4] / (x[1] + x[3] + 1e-6) # penalize small beamA, beamC, tendonThickness and large tendonWidth
        else:
            heuristic += (x[1] + x[3]) / (x[4] + 1e-6) # penalize small tendonWidth and large beamA, beamC, tendonThickness

        if tu > target_torqueUp: # joint is too stiff
            heuristic += x[6] / (x[5] + 1e-6) # penalize small hingeLength and large hingeThickness
        else:   
            heuristic += x[5] / (x[6] + 1e-6) # penalize small hingeThickness and large hingeLength
        penalty += w3 * heuristic
        print(f"td: {td}, tu: {tu}, penalty: {penalty}")
        return penalty


    result = differential_evolution(objective_exp, bounds, popsize=10, maxiter=10)
    optimal_x = result.x

    td, tu = usingSimulator.get_torques(optimal_x)
    # print(f"Predicted torque down: {td}")
    # print(f"Predicted torque up: {tu}")
    # print(f"Optimal parameters: {optimal_x}")

    return optimal_x



def customize(input_params, strength = "default", optimizer = "fast"):
    if input_params['ptorqueExtend'] > input_params['atorqueBend'] or input_params['naturalAngle'] < 0 or input_params['naturalAngle'] > 90:
        raise ValueError("Invalid input parameters: ptorqueExtend must be less than atorqueBend, and naturalAngle must be between 0 and 90 degree.")

    target_torqueDown = (input_params['ptorqueExtend'] + input_params['atorqueBend']) / 2.0
    target_torqueUp = -input_params['atorqueExtend'] / 2.0
    if strength == "soft":
        target_torqueDown = input_params['ptorqueExtend'] + (input_params['atorqueBend'] - input_params['ptorqueExtend']) / 3.0
        target_torqueUp = -input_params['atorqueExtend'] / 3.0
    elif strength == "stiff":
        target_torqueDown = input_params['ptorqueExtend'] + (input_params['atorqueBend'] - input_params['ptorqueExtend']) * 2.0 / 3.0
        target_torqueUp = -input_params['atorqueExtend'] * 2.0 / 3.0
    elif strength == "easyUp":
        target_torqueDown = input_params['ptorqueExtend'] + (input_params['atorqueBend'] - input_params['ptorqueExtend']) * 2.0 / 3.0
        target_torqueUp = -input_params['atorqueExtend'] / 3.0
    elif strength == "easyDown":
        target_torqueDown = input_params['ptorqueExtend'] + (input_params['atorqueBend'] - input_params['ptorqueExtend']) / 3.0
        target_torqueUp = -input_params['atorqueExtend'] * 2.0 / 3.0
        
    bounds = [
        (12.0, 40.0),  # beamALength
        (10.0, 40.0),  # beamCLength
        (10.0, 40.0),  # theta
        (0.4, 2.0),   # tendonThickness
        (0.4, 5.0),   # tendonWidth
        (1.0, 3.0),  # hingeLength
        (0.4, 1.7),   # hingeThickness
    ]
    name_of_bounds = ["beamALength", "beamCLength", "thetaDeg", "tendonThickness", "tendonWidth", "hingeLength", "hingeThickness"]

    bounds_constrained = bounds.copy()
    bounds_constrained[2] = (input_params['naturalAngle'], input_params['naturalAngle'])
    # Initial guess
    x0 = [25.0, 30.0, input_params['naturalAngle'], 1.0, 1.6, 2.0, 1.0]
    
    if optimizer == "fast":
        x = optimized_through_trained_model(x0, input_params, bounds_constrained, target_torqueDown, target_torqueUp)
    elif optimizer == "slow":
        x = just_optimize(input_params, bounds_constrained, target_torqueDown, target_torqueUp)
    
    mjc_model_file = usingSimulator.update_model_parameters(x)
    _, _, torque_curve = simulation.simulate(mjc_model_file)
    torqueDown, torqueUp = usingSimulator. extract_torques(torque_curve)
    
    def update_parameters(x):
        customizeMjcModel.vals['beamA'] = x[0]
        customizeMjcModel.vals['beamC'] = x[1]
        customizeMjcModel.vals['theta'] = x[2]
        customizeMjcModel.vals['tendonThickness'] = x[3]
        customizeMjcModel.vals['tendonWidth'] = x[4]
        customizeMjcModel.vals['hingeLength'] = x[5]
        customizeMjcModel.vals['hingeThickness'] = x[6]
        return customizeMjcModel.vals
    
    geometry_vals = update_parameters(x)
    return x, torqueDown, torqueUp, mjc_model_file, geometry_vals, torque_curve


"""
Customizes brace geometry parameters based on input requirements.

Args:
    input_params (dict): Dictionary containing:
        - naturalAngle (float): Rest angle of the joint in degrees [0-90]
        - ptorqueExtend (float): Force required the pull the finger to straighten. That is, minimum required torque for brace downward motion
        - atorqueBend (float): Force user can actively / intentionally exert. That is, maximum allowable torque for brace downward motion
        - atorqueExtend (float): Maximum allowable torque (the absoluate value) for upward motion (must be negative)
    strength (str, optional): Optimization strategy:
        - "default": Balanced torques
        - "soft": Lower torques
        - "stiff": Higher torques
        - "easyUp": Lower upward torque
        - "easyDown": Lower downward torque
    optimizer (str, optional): Optimization method:
        - "fast": Uses ML models for quick optimization
        - "slow": Run simulation-based optimization without any pre-trained model

Returns:
    tuple: (
        optimized_parameters (list): [beamALength, beamCLength, theta, tendonThickness, tendonWidth, hingeLength, hingeThickness],
        torque_down (float): Achieved downward torque,
        torque_up (float): Achieved upward torque,
        mjc_model_file (str): Path to generated MuJoCo model file,
        geometry_vals (dict): Dictionary of optimized geometry values
        torque_curve (np.ndarray): Simulated torque curve
    )

Raises:
    ValueError: If input parameters violate constraints
"""
# Define input parameters for the customization
input_params = {
    'naturalAngle': 22.0,   # Example value for naturalAngle (in degrees or radians as appropriate)
    'ptorqueExtend': 30.0,    # Lower bound for torqueDown
    'atorqueBend': 50.0,     # Upper bound for torqueDown
    'atorqueExtend': 20.0    # Lower bound for atorqueExtend relative to torqueUp (torqueUp must be less than this)
}

x, torque_down, torque_up, mjc_model_file, geometry_vals, torque_curve = customize(input_params, strength="default", optimizer="fast")
print(f"Optimized parameters: {x}")
print(f"Torque Down: {torque_down}")
print(f"Torque Up: {torque_up}")
print(f"Generated MuJoCo model file: {mjc_model_file}")
print(f"Geometry values: {geometry_vals}")
print(f"Torque curve: {torque_curve}")