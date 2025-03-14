import numpy as np
from scipy.optimize import differential_evolution
from scipy.signal import argrelextrema
import customize
import simulation
from joblib import load


def update_model_parameters(x):
    customize.vals['beamA'] = x[0]
    customize.vals['beamC'] = x[1]
    customize.vals['theta'] = x[2]
    customize.vals['tendonThickness'] = x[3]
    customize.vals['tendonWidth'] = x[4]
    customize.vals['hingeLength'] = x[5]
    customize.vals['hingeThickness'] = x[6]
    customize.generate_model(customize.vals, "2DModel.xml", "optimizing.xml")


def run_simulation(x, plot=False):
    update_model_parameters(x)
    return simulation.simulate("optimizing.xml", plot=plot)


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
    update_model_parameters(x)
    _, _, torque_curve = simulation.simulate("optimizing.xml")
    torqueDown, torqueUp = extract_torques(torque_curve)

    if torqueDown is None or torqueUp is None:
       return None, None
    
    return torqueDown, torqueUp


def optimized_through_trained_model(x0, input_params, bounds):
    # load models
    model_down = load('model_down.joblib')
    model_up = load('model_up.joblib')

    # Initialize x with natural angle
    x = x0.copy()
    x[2] = input_params['naturalAngle']

    target_torqueDown = (input_params['ptorqueExtend'] + input_params['atorqueBend']) / 2.0
    target_torqueUp = -input_params['atorqueExtend'] / 2.0
    
    # Modify bounds to enforce naturalAngle constraint
    bounds_constrained = bounds.copy()
    bounds_constrained[2] = (input_params['naturalAngle'], input_params['naturalAngle'])

    # Define optimization objective
    def objective(x):
        # Reshape input for prediction
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

    # Run optimization with constrained bounds
    result = differential_evolution(objective, bounds_constrained, popsize=10, maxiter=10)
    optimal_x = result.x

    # Verify results with reshaped input
    optimal_x_reshaped = np.array(optimal_x).reshape(1, -1)
    td = float(model_down.predict(optimal_x_reshaped)[0])
    tu = float(model_up.predict(optimal_x_reshaped)[0])
    simu_td, simu_tu = get_torques(np.array(optimal_x))
    print(f"Predicted torque down: {td}")
    print(f"Predicted torque up: {tu}")
    print(f"Simulated torque down: {simu_td}")
    print(f"Simulated torque up: {simu_tu}")
    print(f"Optimal parameters: {optimal_x}")
    
    return optimal_x


def just_optimize(input_params, bounds):
    target_torqueDown = (input_params['ptorqueExtend'] + input_params['atorqueBend']) / 2.0
    target_torqueUp = -input_params['atorqueExtend'] / 2.0

    def objective(x):
        td, tu = get_torques(x)
        if td is None or tu is None:
            return np.inf
        if input_params['atorqueBend'] > td > input_params['ptorqueExtend'] and input_params['atorqueExtend'] < tu < 0:
            return 0
        print(f"td: {td}, tu: {tu}")
        return (td - target_torqueDown)**2 + (tu - target_torqueUp)**2
    

    def objective_exp(x):
        td, tu = get_torques(x)
        if td is None or tu is None:
            return np.inf
        if input_params['atorqueBend'] > td > input_params['ptorqueExtend'] and input_params['atorqueExtend'] < tu < 0:
            return 0
        # Calculate distance from desired ranges using log
        penalty = 0
        td_lower = input_params['ptorqueExtend']
        td_upper = input_params['atorqueBend']
        penalty = np.exp(td_lower - td) + np.exp(td - td_upper)
        
        # Continuous penalty for torque up (tu)
        tu_lower = -input_params['atorqueExtend']
        tu_upper = 0
        penalty += np.exp(tu_lower - tu) + np.exp(tu - tu_upper)

        print(f"td: {td}, tu: {tu}, penalty: {penalty}")
        return penalty
    

    def objective_heuristic(x):
        td, tu = get_torques(x)
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


    x = [25.0, 30.0, input_params['naturalAngle'], 1.0, 1.6, 2.0, 1.0]
    bounds_constrained = bounds.copy()
    bounds_constrained[2] = (input_params['naturalAngle'], input_params['naturalAngle'])

    # Try different optimization methods
    result = differential_evolution(objective_exp, bounds_constrained, popsize=10, maxiter=10)
    optimal_x = result.x

    # Verify results
    td, tu = get_torques(optimal_x)
    print(f"Predicted torque down: {td}")
    print(f"Predicted torque up: {tu}")
    print(f"Optimal parameters: {optimal_x}")



if __name__ == '__main__':
    # Define input parameters for the simulation
    input_params = {
        'naturalAngle': 22.0,   # Example value for naturalAngle (in degrees or radians as appropriate)
        'ptorqueExtend': 30.0,    # Lower bound for torqueDown
        'atorqueBend': 50.0,     # Upper bound for torqueDown
        'atorqueExtend': 20.0    # Lower bound for atorqueExtend relative to torqueUp (torqueUp must be less than this)
    }

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
    
    # Initial guess
    x0 = [25.0, 30.0, input_params['naturalAngle'], 1.0, 1.6, 2.0, 1.0]

    # just_optimize(input_params, bounds)

    x = optimized_through_trained_model(x0, input_params, bounds)


