import numpy as np
from scipy.optimize import differential_evolution
from scipy.signal import argrelextrema
from autograd import grad
import customize
import simulation
import matplotlib.pyplot as plt
import json



def update_model_parameters(x, theta):
    customize.vals['beamA'] = x[0]
    customize.vals['beamC'] = x[1]
    customize.vals['theta'] = theta
    customize.vals['tendonThickness'] = x[3]
    customize.vals['tendonWidth'] = x[4]
    customize.vals['hingeLength'] = x[5]
    customize.vals['hingeThickness'] = x[6]

    customize.generate_model(customize.vals, "2DModel.xml", "optimizing.xml")


def run_simulation(x, input_params, plot=False):
    update_model_parameters(x, input_params['naturalAngle'])
    return simulation.simulate("optimizing.xml", plot=plot)


def extract_torques(torque_curve):
    # Use scipy.signal.argrelextrema to identify local extrema
    max_indices = argrelextrema(torque_curve, np.greater)[0]
    min_indices = argrelextrema(torque_curve, np.less)[0]
    
    torqueDown = torque_curve[max_indices[0]]
    torqueUp = torque_curve[min_indices[0]]

    torqueDown = float(torqueDown)
    torqueUp = float(torqueUp)
    
    return torqueDown, -torqueUp


def extract_torque_model_from_json(name_of_bounds, model_file='combined_torque_model.json'):
    """
    Load the torque model from JSON and return callable functions for torque prediction
    and their gradients.
    Returns four functions: 
    - predict_torque_down(x)
    - predict_torque_up(x)
    - grad_torque_down(x)
    - grad_torque_up(x)
    """
    with open(model_file, 'r') as f:
        model_data = json.load(f)

    def create_predictor(model_params):
        def predictor(x):
            result = model_params['intercept']
            for i, param_name in enumerate(name_of_bounds):
                param_coeffs = model_params['parameters'][param_name]
                for power, coeff in param_coeffs.items():
                    result += coeff * (x[i] ** float(power))
            return np.array(result)
        return predictor

    predict_torque_down = create_predictor(model_data['torque_down'])
    predict_torque_up = create_predictor(model_data['torque_up'])
    
    # # Create gradient functions using autograd
    # grad_torque_down = grad(predict_torque_down, 0)
    # grad_torque_up = grad(predict_torque_up, 0)
    # # algorithms to efficiently navigate the parameter space and find the optimal
    # # set of parameters that minimize or maximize the objective function.
    # grad_torque_down = grad(predict_torque_down)
    # grad_torque_up = grad(predict_torque_up)

    return predict_torque_down, predict_torque_up


# def objective(x, input_params, use_heuristic=True):
#     """
#     Three main conditions:
#     - 1. Bistability: The curve should cross the zero-torque line three times (i.e., have three intersections with y = 0).
#     - 2-1. TorqueDown Bounds: ptorqueExtend < torqueDown < atorqueBend
#     - 2-2. TorqueUp Bound: torqueUp < atorqueExtend

#     Heuristics:

#     """
#     np_angles, np_forces, np_torques = run_simulation(x, input_params)
#     bistable, torqueDown, torqueUp = extract_torques(np_torques)
    
#     # 1. Check bistability: the curve should have three intersections with y=0.
#     penalty_bistability = 0
#     if bistable == False:
#         print("Design is not bistable.")
#         plt.plot(np_angles, np_torques)
#         plt.show()
#         penalty_bistability = 1
#     else:
#         print("Design is bistable.")

#     if torqueDown is None or torqueUp is None:
#         print("No valid torque values.")
#         plt.plot(np_angles, np_torques)
#         plt.show()
#         penalty_bistability = 1e9

#     # 2. Check torque bounds
#     ptorqueExtend = input_params['ptorqueExtend']
#     atorqueBend = input_params['atorqueBend']
#     atorqueExtend = input_params['atorqueExtend']
#     # print(f"torqueDown: {torqueDown:.2f}, torqueUp: {torqueUp:.2f}")

#     # 2-1. Constraint: ptorqueExtend < torqueDown < atorqueBend
#     penalty_torqueDown = 0
#     if torqueDown < ptorqueExtend:
#         penalty_torqueDown = ptorqueExtend - torqueDown
#         print(f"torqueDown: {torqueDown:.2f} is less than ptorqueExtend: {ptorqueExtend:.2f}")
#     elif torqueDown > atorqueBend:
#         penalty_torqueDown = torqueDown - atorqueBend
#         print(f"torqueDown: {torqueDown:.2f} is greater than atorqueBend: {atorqueBend:.2f}")
    
#     # 2-2. Constraint: atorqueExtend > torqueUp
#     penalty_torqueUp = 0
#     if torqueUp > atorqueExtend:
#         penalty_torqueUp = torqueUp - atorqueExtend
#         print(f"torqueUp: {torqueUp:.2f} is greater than atorqueExtend: {atorqueExtend:.2f}")
    
#     # Weights for the penalties (tune these)
#     W1, W2, W3 = 1e6, 10.0, 10.0
    
#     # Total objective function:
#     f = W1 * penalty_bistability + W2 * penalty_torqueDown + W3 * penalty_torqueUp

#     # 3. Additional heuristic-based adjustments (if enabled)
#     # Heuristic terms (initially zero)
#     heuristic_increaseTdown = 0
#     heuristic_decreaseTdown = 0
#     heuristic_increaseTup = 0
#     heuristic_decreaseTup   = 0
#     # Weights for the heuristic terms.
#     H_increaseTdown    = 10.0
#     H_decreaseTdown    = 10.0
#     H_increaseTup      = 10.0
#     H_decreaseTup      = 10.0
#     if use_heuristic:
#         if torqueDown < ptorqueExtend:
#             heuristic_increaseTdown = 1.0 / (x[0] + x[1] + x[3] + 1e-6)
#         elif torqueDown > atorqueBend:
#             heuristic_decreaseTdown = (x[0] + x[1] + x[3])

#         if penalty_bistability > 0:
#             heuristic_increaseTup = x[6] / (x[5] + 1e-6)
#         elif torqueUp > atorqueExtend:
#             heuristic_decreaseTup = x[5] / (x[6] + 1e-6)

#         f += H_increaseTdown * heuristic_increaseTdown + H_decreaseTdown * heuristic_decreaseTdown + H_increaseTup * heuristic_increaseTup + H_decreaseTup * heuristic_decreaseTup

#     print(x)

#     return f


def greedy_search(x0, input_params, bounds, step_size=0.2, max_iter=100):
    x = x0.copy()
    target_torqueDown = (input_params['ptorqueExtend'] + input_params['atorqueBend']) / 2.0
    target_torqueUp = input_params['atorqueExtend'] / 2.0

    span_down = input_params['atorqueBend'] - input_params['ptorqueExtend']
    span_up = input_params['atorqueExtend'] - 0

    for i in range(max_iter):
        _, _, np_torques = run_simulation(x, input_params)
        torqueDown, torqueUp = extract_torques(np_torques) #flip torqueUp sign
        
        if torqueDown < target_torqueDown:
            if x[1] < bounds[1][1]: # beamC
                x[1] += step_size * (bounds[1][1] - bounds[1][0]) * abs(target_torqueDown - torqueDown) / span_down
            elif x[4] < bounds[4][1]: # tendonWidth
                x[4] += step_size * (bounds[4][1] - bounds[4][0]) * abs(target_torqueDown - torqueDown) / span_down
            elif x[3] < bounds[3][1]: # tendonThickness
                x[3] += step_size * (bounds[3][1] - bounds[3][0]) * abs(target_torqueDown - torqueDown) / span_down
            elif x[0] < bounds[0][1]: # beamA
                x[0] += step_size * (bounds[0][1] - bounds[0][0]) * abs(target_torqueDown - torqueDown) / span_down
        else:
            if x[3] > bounds[3][0]: # tendonThickness
                x[3] -= step_size * (bounds[3][1] - bounds[3][0]) * abs(target_torqueDown - torqueDown) / span_down
            elif x[4] > bounds[4][0]: # tendonWidth
                x[4] -= step_size * (bounds[4][1] - bounds[4][0]) * abs(target_torqueDown - torqueDown) / span_down
            elif x[1] > bounds[1][0]: # beamC
                x[1] -= step_size * (bounds[1][1] - bounds[1][0]) * abs(target_torqueDown - torqueDown) / span_down
            elif x[0] > bounds[0][0]: # beamA
                x[0] -= step_size * (bounds[0][1] - bounds[0][0]) * abs(target_torqueDown - torqueDown) / span_down
        
        if torqueUp > target_torqueUp:
            if x[6] < bounds[6][1]: # hingeThickness
                x[6] += step_size * (bounds[6][1] - bounds[6][0]) * abs(torqueUp - target_torqueUp) / span_up
            if x[5] > bounds[5][0]: # hingeLength
                x[5] -= step_size * (bounds[5][1] - bounds[5][0]) * abs(torqueUp - target_torqueUp) / span_up
        else:
            if x[6] > bounds[6][0]: # hingeThickness
                x[6] -= step_size * (bounds[6][1] - bounds[6][0]) * abs(torqueUp - target_torqueUp) / span_up
            if x[5] < bounds[5][1]: # hingeLength
                x[5] += step_size * (bounds[5][1] - bounds[5][0]) * abs(torqueUp - target_torqueUp) / span_up
        
        if input_params['atorqueBend'] > torqueDown > input_params['ptorqueExtend'] and 0 < torqueUp < input_params['atorqueExtend']:
            break
        
        print(f"Iter {i+1}: torqueDown = {torqueDown}, torqueUp = {torqueUp}")
        print(f"Current parameters: {x}")

        
    print(f"Updated parameters: {x}")
    return x


if __name__ == '__main__':
    # Define input parameters for the simulation
    input_params = {
        'naturalAngle': 22.0,   # Example value for naturalAngle (in degrees or radians as appropriate)
        'ptorqueExtend': 50.0,    # Lower bound for torqueDown
        'atorqueBend': 250.0,     # Upper bound for torqueDown
        'atorqueExtend': 70.0    # Lower bound for atorqueExtend relative to torqueUp (torqueUp must be less than this)
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

    # Initial guess
    x0 = [25.0, 30.0, input_params['naturalAngle'], 1.0, 1.6, 2.0, 1.0]
    
    # Test helpers
    # np_angles, np_forces, np_torques = run_simulation(x0, input_params)
    # plt.plot(np_torques)
    # bistable, torqueDown, torqueUp = extract_torques(np_torques)
    # print(f"bistable?: {bistable}, torqueDown: {torqueDown}, torqueUp: {torqueUp}")

    # Run the optimization
    # result = differential_evolution(
    #     func=lambda x0: objective(x0, input_params, use_heuristic=True),
    #     bounds=bounds,
    #     maxiter=100,
    #     popsize=15,
    #     polish=True,
    #     workers=1  # <-- Add this line to force single-threaded execution
    # )
    # print("Optimization Result:")
    # print(result)

    # x = greedy_search(x0, input_params, bounds)
    # run_simulation(x, input_params, plot=True)

    name_of_bounds = ["beamALength", "beamCLength", "thetaDeg", "tendonThickness", "tendonWidth", "hingeLength", "hingeThickness"]
    
    predict_torque_down, predict_torque_up = extract_torque_model_from_json(name_of_bounds=name_of_bounds)
    # Initialize x with natural angle
    x = x0.copy()
    x[2] = input_params['naturalAngle']
    
    # Modify bounds to enforce naturalAngle constraint
    bounds_constrained = bounds.copy()
    bounds_constrained[2] = (input_params['naturalAngle'], input_params['naturalAngle'])

    # Define optimization objective
    def objective(x):
        td = predict_torque_down(x)
        tu = predict_torque_up(x)
        penalty = 0
        
        if td < input_params['ptorqueExtend']:
            penalty += (input_params['ptorqueExtend'] - td) ** 2
        if td > input_params['atorqueBend']:
            penalty += (td - input_params['atorqueBend']) ** 2
        if tu > 0:
            penalty += tu ** 2
        if tu < -input_params['atorqueExtend']:
            penalty += (tu + input_params['atorqueExtend']) ** 2
        
        return penalty

    # Run optimization with constrained bounds
    result = differential_evolution(objective, bounds_constrained, maxiter=100)
    optimal_x = result.x

    # Verify results
    td = predict_torque_down(optimal_x)
    tu = predict_torque_up(optimal_x)
    print(f"Predicted torque down: {td}")
    print(f"Predicted torque up: {tu}")
    print(f"Optimal parameters: {optimal_x}")
