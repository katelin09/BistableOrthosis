import numpy as np
from scipy.optimize import differential_evolution
from scipy.signal import argrelextrema
from autograd import grad
import customize
import simulation
import matplotlib.pyplot as plt
import json
from scipy.optimize import minimize



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

    torqueDown = float(torqueDown[0])
    torqueUp = float(torqueUp[0])
    
    return torqueDown, torqueUp


def extract_torque_model_from_json(name_of_bounds, model_file='combined_torque_model.json'):
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
    
    def create_gradient_predictor(model_params):
        def gradient_predictor(x):
            gradients = []
            for i, param_name in enumerate(name_of_bounds):
                grad = 0
                if param_name in model_params['gradients']:
                    coeffs = model_params['gradients'][param_name]['coefficients']
                    for j, coeff in enumerate(coeffs, 1):
                        grad += coeff * j * (x[i] ** (j-1))
                gradients.append(grad)
            return np.array(gradients)
        return gradient_predictor

    predict_torque_down = create_predictor(model_data['torque_down'])
    predict_torque_up = create_predictor(model_data['torque_up'])
    grad_model_down = create_gradient_predictor(model_data['torque_down'])
    grad_model_up = create_gradient_predictor(model_data['torque_up'])

    return predict_torque_down, predict_torque_up, grad_model_down, grad_model_up


# def greedy_search(x0, input_params, bounds, step_size=1, max_iter=100):
#     x = x0.copy()
#     target_torqueDown = (input_params['ptorqueExtend'] + input_params['atorqueBend']) / 2.0
#     target_torqueUp = -input_params['atorqueExtend'] / 2.0
    
#     # Get the torque prediction functions
#     predict_torque_down, predict_torque_up, _, _ = extract_torque_model_from_json(name_of_bounds)

#     for i in range(max_iter):
#         # Get current predictions
#         torqueDown = predict_torque_down(x)
#         torqueUp = predict_torque_up(x)
        
#         # Calculate gradients for each parameter
#         grad_down = np.zeros_like(x)
#         grad_up = np.zeros_like(x)
        
#         h = 1e-6
#         for j in range(len(x)):
#             x_plus = x.copy()
#             x_plus[j] += h
#             x_minus = x.copy()
#             x_minus[j] -= h
#             grad_down[j] = (predict_torque_down(x_plus) - predict_torque_down(x_minus)) / (2*h)
#             grad_up[j] = (predict_torque_up(x_plus) - predict_torque_up(x_minus)) / (2*h)
        
#         # Update parameters based on gradients and bounds
#         if torqueDown < target_torqueDown:
#             if x[1] < bounds[1][1]:  # beamC
#                 x[1] += step_size * grad_down[1]
#             elif x[4] < bounds[4][1]:  # tendonWidth
#                 x[4] += step_size * grad_down[4]
#             elif x[3] < bounds[3][1]:  # tendonThickness
#                 x[3] += step_size * grad_down
#             elif x[0] < bounds[0][1]:  # beamA
#                 x[0] += step_size * grad_down[0]
#         else:
#             if x[3] > bounds[3][0]:  # tendonThickness
#                 x[3] -= step_size * grad_down[3]
#             elif x[4] > bounds[4][0]:  # tendonWidth
#                 x[4] -= step_size * grad_down[4]
#             elif x[1] > bounds[1][0]:  # beamC
#                 x[1] -= step_size * grad_down[1]
#             elif x[0] > bounds[0][0]:  # beamA
#                 x[0] -= step_size * grad_down[0]

#         if torqueUp > target_torqueUp:
#             if x[6] < bounds[6][1]:  # hingeThickness
#                 x[6] += step_size * grad_up[6]
#             if x[5] > bounds[5][0]:  # hingeLength
#                 x[5] -= step_size * grad_up[5]
#         else:
#             if x[6] > bounds[6][0]:  # hingeThickness
#                 x[6] -= step_size * grad_up[6]
#             if x[5] < bounds[5][1]:  # hingeLength
#                 x[5] += step_size * grad_up[5]

#         # Clip to bounds  
#         for j in range(len(x)):        
#             x[j] = np.clip(x[j], bounds[j][0], bounds[j][1])
        
#         # Check stop criteria
#         if (input_params['atorqueBend'] > torqueDown > input_params['ptorqueExtend'] and 
#             0 > torqueUp > input_params['atorqueExtend']):
#             break
        
#         torqueDown = predict_torque_down(x)
#         torqueUp = predict_torque_up(x)
#         print(f"Iter {i+1}: torqueDown = {torqueDown}, torqueUp = {torqueUp}")
#         print(f"Current parameters: {x}")

#     print(f"Updated parameters: {x}")
#     return x


# def greedy_search(x0, input_params, bounds, step_size=0.2, max_iter=100):
#     x = x0.copy()
#     target_torqueDown = (input_params['ptorqueExtend'] + input_params['atorqueBend']) / 2.0
#     target_torqueUp = input_params['atorqueExtend'] / 2.0

#     span_down = input_params['atorqueBend'] - input_params['ptorqueExtend']
#     span_up = input_params['atorqueExtend'] - 0

#     for i in range(max_iter):
#         _, _, np_torques = run_simulation(x, input_params)
#         torqueDown, torqueUp = extract_torques(np_torques) #flip torqueUp sign
        
#         if torqueDown < target_torqueDown:
#             if x[1] < bounds[1][1]: # beamC
#                 x[1] += step_size * (bounds[1][1] - bounds[1][0]) * abs(target_torqueDown - torqueDown) / span_down
#             elif x[4] < bounds[4][1]: # tendonWidth
#                 x[4] += step_size * (bounds[4][1] - bounds[4][0]) * abs(target_torqueDown - torqueDown) / span_down
#             elif x[3] < bounds[3][1]: # tendonThickness
#                 x[3] += step_size * (bounds[3][1] - bounds[3][0]) * abs(target_torqueDown - torqueDown) / span_down
#             elif x[0] < bounds[0][1]: # beamA
#                 x[0] += step_size * (bounds[0][1] - bounds[0][0]) * abs(target_torqueDown - torqueDown) / span_down
#         else:
#             if x[3] > bounds[3][0]: # tendonThickness
#                 x[3] -= step_size * (bounds[3][1] - bounds[3][0]) * abs(target_torqueDown - torqueDown) / span_down
#             elif x[4] > bounds[4][0]: # tendonWidth
#                 x[4] -= step_size * (bounds[4][1] - bounds[4][0]) * abs(target_torqueDown - torqueDown) / span_down
#             elif x[1] > bounds[1][0]: # beamC
#                 x[1] -= step_size * (bounds[1][1] - bounds[1][0]) * abs(target_torqueDown - torqueDown) / span_down
#             elif x[0] > bounds[0][0]: # beamA
#                 x[0] -= step_size * (bounds[0][1] - bounds[0][0]) * abs(target_torqueDown - torqueDown) / span_down
        
#         if torqueUp > target_torqueUp:
#             if x[6] < bounds[6][1]: # hingeThickness
#                 x[6] += step_size * (bounds[6][1] - bounds[6][0]) * abs(torqueUp - target_torqueUp) / span_up
#             if x[5] > bounds[5][0]: # hingeLength
#                 x[5] -= step_size * (bounds[5][1] - bounds[5][0]) * abs(torqueUp - target_torqueUp) / span_up
#         else:
#             if x[6] > bounds[6][0]: # hingeThickness
#                 x[6] -= step_size * (bounds[6][1] - bounds[6][0]) * abs(torqueUp - target_torqueUp) / span_up
#             if x[5] < bounds[5][1]: # hingeLength
#                 x[5] += step_size * (bounds[5][1] - bounds[5][0]) * abs(torqueUp - target_torqueUp) / span_up
        
#         if input_params['atorqueBend'] > torqueDown > input_params['ptorqueExtend'] and 0 < torqueUp < input_params['atorqueExtend']:
#             break
        
#         print(f"Iter {i+1}: torqueDown = {torqueDown}, torqueUp = {torqueUp}")
#         print(f"Current parameters: {x}")

        
#     print(f"Updated parameters: {x}")
#     return x


def differential_evolution_search(x0, input_params, bounds, name_of_bounds):
    predict_torque_down, predict_torque_up, _, _ = extract_torque_model_from_json(name_of_bounds=name_of_bounds)
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
    
    return optimal_x



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

    name_of_bounds = ["beamALength", "beamCLength", "thetaDeg", "tendonThickness", "tendonWidth", "hingeLength", "hingeThickness"]
    
    # Initial guess
    x0 = [25.0, 30.0, input_params['naturalAngle'], 1.0, 1.6, 2.0, 1.0]

    # x = greedy_search(x0, input_params, bounds, step_size=1, max_iter=100)
    # run_simulation(x, input_params, plot=True)

    # x = differential_evolution_search(x0, input_params, bounds, name_of_bounds)

    predict_torque_down, predict_torque_up, _, _ = extract_torque_model_from_json(name_of_bounds)
    target_torqueDown = (input_params['ptorqueExtend'] + input_params['atorqueBend']) / 2.0
    target_torqueUp = -input_params['atorqueExtend'] / 2.0

    def objective(x):
        td = predict_torque_down(x)
        tu = predict_torque_up(x)
        return (td - target_torqueDown)**2 + (tu - target_torqueUp)**2

    # Try different optimization methods

    methods = ['Nelder-Mead', 'L-BFGS-B', 'Powell']
    best_result = None
    best_score = float('inf')

    bounds[2] = (input_params['naturalAngle'], input_params['naturalAngle'])

    for method in methods:
        result = minimize(objective, x0, method=method, bounds=bounds)
        print(f"Method: {method}, Result: {result.message}, Fun: {result.fun}")
        print(f"Parameters: {result.x}")
        _, _, np_torques = run_simulation(result.x, input_params)
        torqueDown, torqueUp = extract_torques(np_torques)
        print(f"Torque Down: {torqueDown}, Torque Up: {torqueUp}")
    #     if result.fun < best_score:
    #         best_score = result.fun
    #         best_result = result

    # x = best_result.x
    # td = predict_torque_down(x)
    # tu = predict_torque_up(x)

    # print(f"Best optimization method: {best_result.message}")
    # print(f"Found solution:")
    # print(f"torqueDown = {td:.2f} (target: {target_torqueDown:.2f})")
    # print(f"torqueUp = {tu:.2f} (target: {target_torqueUp:.2f})")
    # print(f"Parameters: {x}")
