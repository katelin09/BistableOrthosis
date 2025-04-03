import numpy as np
import sys
sys.path.append("../simulator")
from customizeMjcModel import DEFAULT_GEOPARAMS, DEFAULT_HYPERPARAMS, generate_model
from mjcSimulator import simulate
from usingSimulator import extract_torques, update_model_parameters, get_torques
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
import json
import os
import time

def plot_comparison(x, exp_data, model_params):
    hyperparams = {
        'jointStiffness': x[0],
        'jointDamping': x[1], 
        'tendonExtendStiffness': x[2],
        'tendonExtendDamping': x[3],
        'tendonBendStiffness': x[4],
        'tendonBendDamping': x[5],
        'nonLinearStiffness': x[6],
        'scaleFactor': x[7]
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()
    
    # total_curve_error = 0
    # total_peak_error = 0
    # total_valley_error = 0
    
    for model_num in range(len(model_params)):
        model_name = f"model{model_num+1}"
        geo_params = model_params[model_name]
    
        model_file, scaled_params = generate_model(geo_params, hyperparams, xml_file=f"../simulator/orthosis_model.xml", saved_file=f"../simulator/temp/{model_name}.xml")
        sim_angles, _, sim_torques = simulate(
            model_file,     
            nonLinear=scaled_params['nonLinearStiffness'], 
            scaleFactor=scaled_params['scaleFactor']
        )
        
        print("sim_torques: ", sim_torques)
        print("model_file: ", model_file)

        exp_angles, exp_torques = exp_data[model_num]
        (exp_tdown_angle, exp_tdown_torque), (exp_tup_angle, exp_tup_torque) = find_critical_points(exp_angles, exp_torques)
        (sim_tdown_angle, sim_tdown_torque), (sim_tup_angle, sim_tup_torque) = find_critical_points(sim_angles, sim_torques)
        
        print("exp_torque_down: ", exp_tdown_torque)
        print("exp_torque_up: ", exp_tup_torque)
        print("sim_torque_down: ", sim_tdown_torque)
        print("sim_torque_up: ", sim_tup_torque)
        
        # Plot comparison
        axes[model_num].plot(sim_angles, sim_torques, 'b-', label='Simulation')
        axes[model_num].plot(exp_angles, exp_torques, 'r--', label='Experiment')
        
        # Plot critical points
        if sim_tdown_torque is not None:
            axes[model_num].plot(sim_tdown_angle, sim_tdown_torque, 'go', label='Sim Max')
        if sim_tup_torque is not None:
            axes[model_num].plot(sim_tup_angle, sim_tup_torque, 'mo', label='Sim Valley')
        
        if exp_tdown_torque is not None:
            axes[model_num].plot(exp_tdown_angle, exp_tdown_torque, 'go', label='Exp Max')
        if exp_tup_torque is not None:
            axes[model_num].plot(exp_tup_angle, exp_tup_torque, 'mo', label='Exp Valley')

        # # Calculate peak and valley errors
        # peak_error = 0
        # valley_error = 0
        # if sim_torque_down is not None and exp_torque_down is not None:
        #     peak_error = abs(sim_torque_down - exp_torque_down)
        #     total_peak_error += peak_error
        # if sim_torque_up is not None and exp_torque_up is not None:
        #     valley_error = abs(sim_torque_up - exp_torque_up)
        #     total_valley_error += valley_error
            
        axes[model_num].set_xlabel('Angle (degrees)')
        axes[model_num].set_ylabel('Torque (Nmm)')
        axes[model_num].set_ylim(-20, 50)
        axes[model_num].grid(True)
        axes[model_num].legend()
        axes[model_num].set_title(f'Model {model_num+1}')
    
    # avg_peak_error = float(total_peak_error / len(model_params))
    # avg_valley_error = float(total_valley_error / len(model_params))
    # print(f"Average Peak Error: {avg_peak_error:.3f} Nmm")
    # print(f"Average Valley Error: {avg_valley_error:.3f} Nmm")
    
    plt.tight_layout()
    plt.show()

def objective_function(x, exp_data, model_params):
    """
    Objective function for optimization. Takes parameter vector x and returns total error.
    """
    # Create hyperparameters dictionary from current values
    hyperparams = {
        'jointStiffness': x[0],
        'jointDamping': x[1], 
        'tendonExtendStiffness': x[2],
        'tendonExtendDamping': x[3],
        'tendonBendStiffness': x[4],
        'tendonBendDamping': x[5],
        'nonLinearStiffness': x[6],
        'scaleFactor': 1
    }

    total_max_error = 0
    total_min_error = 0
    
    for model_num in range(len(model_params)):
        model_name = f"model{model_num+1}"
        geo_params = model_params[model_name]
    
        model_file, scaled_params = generate_model(geo_params, hyperparams, xml_file=f"../simulator/orthosis_model.xml", saved_file=f"../simulator/temp/{model_name}.xml")
        sim_angles, _, sim_torques = simulate(
            model_file,     
            nonLinear=scaled_params['nonLinearStiffness'], 
            scaleFactor=scaled_params['scaleFactor']
        )
        
        # print("sim_torques: ", sim_torques)
        # print("model_file: ", model_file)

        exp_angles, exp_torques = exp_data[model_num]
        (exp_tdown_angle, exp_tdown_torque), (exp_tup_angle, exp_tup_torque) = find_critical_points(exp_angles, exp_torques)
        (sim_tdown_angle, sim_tdown_torque), (sim_tup_angle, sim_tup_torque) = find_critical_points(sim_angles, sim_torques)
        
        # print("--------------------------------")
        # print("model_num: ", model_num)
        # print("exp_torque_down: ", exp_tdown_torque)
        # print("exp_torque_up: ", exp_tup_torque)
        # print("sim_torque_down: ", sim_tdown_torque)
        # print("sim_torque_up: ", sim_tup_torque)
        # print("geo_params: ", geo_params)
        # print("hyperparams: ", hyperparams)
        # print("--------------------------------")
        
        # Calculate normalized errors
        if sim_tdown_torque is not None:
            max_error = (sim_tdown_torque - exp_tdown_torque)**2
            total_max_error += max_error
        else:
            max_error = 1e6
            total_max_error += max_error
        
        if sim_tup_torque is not None:
            min_error = (sim_tup_torque - exp_tup_torque)**2
            total_min_error += min_error
        else:
            min_error = 1e6
            total_min_error += min_error
        
    avg_max_error = total_max_error / len(model_params)
    avg_min_error = total_min_error / len(model_params)
    
    # Add penalty of 2 times standard deviation
    total_error = avg_max_error + avg_min_error + (avg_max_error - avg_min_error) **2

    

    print("--------------------------------")
    print("x: ", x)
    print(f"Current error: {total_error} (max: {avg_max_error:.3f}, min: {avg_min_error:.3f})")
    print("--------------------------------")
    return total_error

def optimize_parameters(x0, exp_data, model_params, range = 10):
    bounds = [
        (x0[0] / range, x0[0] * range),   # jointStiffness
        (x0[1] / range, x0[1] * range),   # jointDamping 
        (x0[2] / range, x0[2] * range),   # tendonExtendStiffness
        (x0[3] / range, x0[3] * range),   # tendonExtendDamping
        (x0[4] / range, x0[4] * range),   # tendonBendStiffness
        (x0[5] / range, x0[5] * range),   # tendonBendDamping
        (x0[6] / range, x0[6] * range),   # nonLinearStiffness
        # (x0[3] * 0.5, x0[3] * 2.0)    # scaleFactor
    ]
    
    # result = minimize(
    #     lambda x: objective_function(x, exp_data, model_params),
    #     x0,
    #     method='L-BFGS-B',
    #     bounds=bounds,
    #     options={'maxiter': 100, 'eps': 1e3}
    # )
    result = differential_evolution(
        lambda x: objective_function(x, exp_data, model_params),
        bounds=bounds,
        x0=x0,
        maxiter=100
    )
    # bounds_pso = (
    #     np.array([bound[0] for bound in bounds]),  # Min bounds
    #     np.array([bound[1] for bound in bounds])   # Max bounds
    # )
    # print("bounds_pso: ", bounds_pso)
    # options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
    # optimizer = ps.single.GlobalBestPSO(n_particles=30, dimensions=len(bounds), options=options, bounds=bounds_pso)

    # cost, result = optimizer.optimize(
    #     lambda x: objective_function(x, exp_data, model_params),
    #     iters=100
    # )
    optimized_params = {
        'jointStiffness': result.x[0],
        'jointDamping': result.x[1],
        'tendonExtendStiffness': result.x[2], 
        'tendonExtendDamping': result.x[3],
        'tendonBendStiffness': result.x[4],
        'tendonBendDamping': result.x[5],
        'nonLinearStiffness': result.x[6],
        'scaleFactor': result.x[7]
    }
    # optimized_params = {
    #     'Stiffness': result.x[0],
    #     'Damping': result.x[1],
    #     'nonLinearStiffness': result.x[2], 
    # }

    print("\nOptimization complete!")
    print(f"Final error: {result.fun}")
    print("\nOptimized parameters:")
    for param, value in optimized_params.items():
        print(f"{param}: {value:.2e}")

    return optimized_params

def find_zero_crossings(angles, torques):
    zero_crossings = []
    zero_indices = []
    
    # Find where torque changes sign
    for i in range(len(torques) - 1):
        if (torques[i] * torques[i + 1]) <= 0:  # Sign change detected
            # Linear interpolation to find more precise zero crossing
            if torques[i] != torques[i + 1]:  # Avoid division by zero
                t = -torques[i] / (torques[i + 1] - torques[i])
                zero_angle = angles[i] + t * (angles[i + 1] - angles[i])
                zero_crossings.append(zero_angle)
                zero_indices.append(i)
    
    # If we don't find enough zero crossings, return None values
    if len(zero_crossings) < 3:
        return None, None, None, None
    
    # Sort zero crossings and corresponding indices together
    zero_crossings, zero_indices = zip(*sorted(zip(zero_crossings, zero_indices)))
    
    # Convert to numpy arrays and extract single elements
    zero_crossings = np.array(zero_crossings)
    zero_indices = np.array(zero_indices)
    
    return float(zero_crossings[1].item()), float(zero_crossings[2].item()), int(zero_indices[1].item()), int(zero_indices[2].item())

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

def create_dataset(model_params, x0, range, step_size = 10):
    bounds = [
        (x0[0] / range, x0[0] * range),   # jointStiffness
        (x0[1] / range, x0[1] * range),   # jointDamping 
        (x0[2] / range, x0[2] * range),   # tendonExtendStiffness
        (x0[3] / range, x0[3] * range),   # tendonExtendDamping
        (x0[4] / range, x0[4] * range),   # tendonBendStiffness
        (x0[5] / range, x0[5] * range),   # tendonBendDamping
        (x0[6] / range, x0[6] * range),   # nonLinearStiffness
    ]

    for model_num in range(len(model_params)):
        model_name = f"model{model_num+1}"
        geo_params = model_params[model_name]

        # Create grid of hyperparameter values within bounds
        hyperparams_list = []
        
        for js in np.arange(bounds[0][0], bounds[0][1], (bounds[0][1]-bounds[0][0]) / step_size):
            for jd in np.arange(bounds[1][0], bounds[1][1], (bounds[1][1]-bounds[1][0]) / step_size):
                for tes in np.arange(bounds[2][0], bounds[2][1], (bounds[2][1]-bounds[2][0]) / step_size):
                    for ted in np.arange(bounds[3][0], bounds[3][1], (bounds[3][1]-bounds[3][0]) / step_size):
                        for tbs in np.arange(bounds[4][0], bounds[4][1], (bounds[4][1]-bounds[4][0]) / step_size):
                            for tbd in np.arange(bounds[5][0], bounds[5][1], (bounds[5][1]-bounds[5][0]) / step_size):
                                for nls in np.arange(bounds[6][0], bounds[6][1], (bounds[6][1]-bounds[6][0]) / step_size):
                                    hyperparams = {
                                        'jointStiffness': js,
                                        'jointDamping': jd,
                                        'tendonExtendStiffness': tes, 
                                        'tendonExtendDamping': ted,
                                        'tendonBendStiffness': tbs,
                                        'tendonBendDamping': tbd,
                                        'nonLinearStiffness': nls,
                                        'scaleFactor': 1
                                    }
                                    hyperparams_list.append(hyperparams)

                                    model_file, scaled_params = generate_model(geo_params, hyperparams, xml_file=f"../simulator/orthosis_model.xml", saved_file=f"../simulator/temp/{model_name}.xml")
                                    sim_angles, _, sim_torques = simulate(
                                        model_file,     
                                        nonLinear=scaled_params['nonLinearStiffness'], 
                                        scaleFactor=scaled_params['scaleFactor']
                                    )

                                    # Save simulation data to dataset folder
                                    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
                                    os.makedirs(dataset_dir, exist_ok=True)
                                    
                                    dataset = {
                                        'sim_angles': sim_angles.tolist(),
                                        'sim_torques': sim_torques.tolist(),
                                        'geo_params': geo_params,
                                        'hyper_params': hyperparams
                                    }
        
                                    # Convert hyperparams to string representation for filename
                                    param_str = f"{int(np.log10(js))}-{int(np.log10(jd))}-{int(np.log10(tes))}-{int(np.log10(ted))}-{int(np.log10(tbs))}-{int(np.log10(tbd))}-{int(np.log10(nls))}"
                                    dataset_file = os.path.join(dataset_dir, f"{model_name}_{param_str}_data.json")
                                    with open(dataset_file, 'w') as f:
                                        json.dump(dataset, f, indent=4)


def main():
    # Define geometric parameters for each model
    model_params = {
        'model1': {
            'beamA': 30, 'beamC': 30, 'theta': 30,
            'tendonThickness': 1, 'tendonWidth': 1.6, 'hingeLength': 2,
            'beamB': DEFAULT_GEOPARAMS['beamB'],
            'hingeThickness': DEFAULT_GEOPARAMS['hingeThickness'],
            'hingeWidth': DEFAULT_GEOPARAMS['hingeWidth']
        },
        'model2': {
            'beamA': 20, 'beamC': 20, 'theta': 30,
            'tendonThickness': 1, 'tendonWidth': 1.6, 'hingeLength': 2,
            'beamB': DEFAULT_GEOPARAMS['beamB'],
            'hingeThickness': DEFAULT_GEOPARAMS['hingeThickness'],
            'hingeWidth': DEFAULT_GEOPARAMS['hingeWidth']
        },
        'model3': {
            'beamA': 30, 'beamC': 30, 'theta': 40,
            'tendonThickness': 1, 'tendonWidth': 1.6, 'hingeLength': 2,
            'beamB': DEFAULT_GEOPARAMS['beamB'],
            'hingeThickness': DEFAULT_GEOPARAMS['hingeThickness'],
            'hingeWidth': DEFAULT_GEOPARAMS['hingeWidth']
        },
        'model4': {
            'beamA': 30, 'beamC': 30, 'theta': 30,
            'tendonThickness': 0.5, 'tendonWidth': 1.6, 'hingeLength': 2,
            'beamB': DEFAULT_GEOPARAMS['beamB'],
            'hingeThickness': DEFAULT_GEOPARAMS['hingeThickness'],
            'hingeWidth': DEFAULT_GEOPARAMS['hingeWidth']
        },
        'model5': {
            'beamA': 30, 'beamC': 30, 'theta': 30,
            'tendonThickness': 1, 'tendonWidth': 1.6, 'hingeLength': 3,
            'beamB': DEFAULT_GEOPARAMS['beamB'],
            'hingeThickness': DEFAULT_GEOPARAMS['hingeThickness'],
            'hingeWidth': DEFAULT_GEOPARAMS['hingeWidth']
        }
    }
    
    # Extract model numbers from model_params
    model_num = len(model_params)
    
    # Initialize dictionaries to store experimental torques for each model
    exp_torque_down = np.zeros(model_num)
    exp_torque_up = np.zeros(model_num)
    exp_data = []

    # Load experimental data for each model
    for i in range(model_num):
        data = np.loadtxt(
            f'processed_real_data/id{i+1}.csv',
            delimiter=',',
            skiprows=1
        )
        exp_angles = data[:, 0]
        exp_torques = data[:, 4]
        exp_data.append((exp_angles, exp_torques))
        exp_peak, exp_valley = find_critical_points(exp_angles, exp_torques)
        # print(exp_peak, exp_valley)
        
        exp_torque_down[i] = exp_peak[1]
        exp_torque_up[i] = exp_valley[1]              
    

    # print("exp_torque_down: ", exp_torque_down)
    # print("exp_torque_up: ", exp_torque_up)
    
    # STEP 1: Initial parameter guess, based on PCTPE document
    x0 = np.array([
        7e6,    # jointStiffness
        1e4,    # jointDamping
        7e6,    # tendonExtendStiffness
        1e4,    # tendonExtendDamping 
        7e6,    # tendonBendStiffness
        1e4,    # tendonBendDamping
        2e8,    # nonLinearStiffness
    ])

    x0 = np.array([
        7e6,    # k1
        1e4,    # d
        2e8,    # k2
    ])

    # ERROR 169

    # STEP 2: use differential evolution to find the best stiffness and damping
    x0 = np.array([5.24208019e+06, 9.94505384e+04, 1.98451964e+08])
    # Current error: 53.671888525545995 (max: 23.400, min: 30.272)

    # ERROR 54

    # STEP 3: use the best stiffness and damping to tune each of the parameters
    x0 = np.array([
        x0[0], # jointStiffness
        x0[1], # jointDamping
        x0[0], # tendonExtendStiffness
        x0[1], # tendonExtendDamping
        x0[0], # tendonBendStiffness
        x0[1], # tendonBendDamping
        x0[2], # nonLinearStiffness
        1 # scaleFactor
    ])
    x0 = np.array([
        2.08075867e+07, 
        9.54346100e+05, 
        3.92490027e+06, 
        4.96370007e+05, 
        3.50530613e+07, 
        1.19566343e+05, 
        1.99619680e+07, 
        1
    ])
    
    # ERROR Current error: 29.184188598475714 (max: 23.218, min: 5.966)

    # STEP 4: create dataset
    # create_dataset(model_params, x0, range=10)
    # optimized_params = optimize_parameters(x0, exp_data, model_params, range=10)
    # optimized_params = list(optimized_params.values())
    optimized_params = x0
    # Plot comparison using optimized parameters
    plot_comparison(optimized_params, exp_data, model_params)

    # save_optimized_params
    params = {
        'jointStiffness': optimized_params[0],
        'jointDamping': optimized_params[1],
        'tendonExtendStiffness': optimized_params[2],
        'tendonExtendDamping': optimized_params[3],
        'tendonBendStiffness': optimized_params[4],
        'tendonBendDamping': optimized_params[5],
        'nonLinearStiffness': optimized_params[6],
        'scaleFactor': 1
    }
    with open('optimized_params.json', 'w') as f:
        json.dump(params, f)

if __name__ == "__main__":
    main()
