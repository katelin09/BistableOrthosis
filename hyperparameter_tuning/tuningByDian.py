import numpy as np
import sys
sys.path.append("../simulator")
from customizeMjcModel import vals, generate_model
from mjcSimulator import simulate
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def tune_stiffness_parameters(exp_ids, initial_params, method='grid'):
    """
    Tune parameters using either grid search or optimization
    Args:
        exp_ids: list of experimental data IDs
        initial_params: dictionary of initial parameters
        method: 'grid' for grid search or 'optimize' for optimization
    Returns optimized parameters and errors
    """
    if method == 'grid':
        # Define parameter ranges with with 6 spaced points
        param_ranges = {            
                'jointStiffness': [1e5,3e5],
                'jointStiffnessDampingRatio': [1e2,3e2],
                'tendonExtendStiffness': [1e7,2e8],
                'tendonExtendStiffnessDampingRatio': [20,40],
                'tendonBendStiffness': [14e10,18e10],
                'tendonBendStiffnessDampingRatio': [1e5,3e5],
                'nonLinearStiffness': [5,15],
                'scaleFactor': [0.9,1.1]
        }
        
        # Initialize results storage
        best_params = None
        best_total_error = float('inf')
        results = []

        # Calculate total combinations
        total_combinations = np.prod([len(vals) for vals in param_ranges.values()])
        print(f"Total parameter combinations to test: {total_combinations}")

        # Counter for progress tracking
        counter = 0

        # Grid search through parameter combinations
        for js in param_ranges['jointStiffness']:
            for jsd in param_ranges['jointStiffnessDampingRatio']:
                for tes in param_ranges['tendonExtendStiffness']:
                    for tesd in param_ranges['tendonExtendStiffnessDampingRatio']:
                        for tbs in param_ranges['tendonBendStiffness']:
                            for tbsd in param_ranges['tendonBendStiffnessDampingRatio']:
                                for nls in param_ranges['nonLinearStiffness']:
                                    for sf in param_ranges['scaleFactor']:
                                        counter += 1
                                        if counter % 4 == 0:
                                            print(f"Progress: {counter}/{total_combinations} combinations tested")

                                        test_params = {
                                            'jointStiffness': js,
                                            'jointStiffnessDampingRatio': jsd,
                                            'tendonExtendStiffness': tes,
                                            'tendonExtendStiffnessDampingRatio': tesd,
                                            'tendonBendStiffness': tbs,
                                            'tendonBendStiffnessDampingRatio': tbsd,
                                            'nonLinearStiffness': nls,
                                            'scaleFactor': sf
                                        }

                                        # Calculate errors for all models
                                        total_max_error = 0
                                        total_min_error = 0
                                        total_error = 0

                                        modify_parameters(test_params)
                                        
                                        for model_num in range(1, 6): #exclude model 6
                                            model_name = f"model{model_num}"
                                            model_file = f"generatedModel/{model_name}.xml"
                                            # Update geometric parameters for this specific model
                                            model_params = {
                                                'model1': {'beamA': 30, 'beamC': 30, 'theta': 30, 'tendonThickness': 1, 'tendonWidth': 1.6, 'hingeLength': 2},
                                                'model2': {'beamA': 20, 'beamC': 20, 'theta': 30, 'tendonThickness': 1, 'tendonWidth': 1.6, 'hingeLength': 2},
                                                'model3': {'beamA': 30, 'beamC': 30, 'theta': 40, 'tendonThickness': 1, 'tendonWidth': 1.6, 'hingeLength': 2},
                                                'model4': {'beamA': 30, 'beamC': 30, 'theta': 30, 'tendonThickness': 0.5, 'tendonWidth': 1.6, 'hingeLength': 2},
                                                'model5': {'beamA': 30, 'beamC': 30, 'theta': 30, 'tendonThickness': 1, 'tendonWidth': 1.6, 'hingeLength': 3}
                                            }
                                            modify_parameters(model_params[model_name])
                                            generate_model(vals, "2DModel.xml", model_file)
                                            
                                            # Pass nonlinear stiffness and scale factor to simulate function
                                            sim_angles, _, sim_torques = simulate(
                                                model_file,
                                                nonLinear=test_params['nonLinearStiffness'],
                                                scaleFactor=test_params['scaleFactor']
                                            )
                                            
                                            exp_id = exp_ids[model_num-1]
                                            data = np.loadtxt(f'processed_real_data/{exp_id}.csv', delimiter=',', skiprows=1)
                                            exp_angles = data[:, 0]
                                            exp_torques = data[:, 4]
                                            
                                            # Get peaks and valleys
                                            sim_peak, sim_valley = find_critical_points(sim_angles, sim_torques)
                                            exp_peak, exp_valley = find_critical_points(exp_angles, exp_torques)
                                            
                                            if sim_peak and exp_peak:
                                                max_error = abs(sim_peak[1] - exp_peak[1]) / abs(exp_peak[1])
                                                total_max_error += max_error
                                            
                                            if sim_valley and exp_valley:
                                                min_error = abs(sim_valley[1] - exp_valley[1]) / abs(exp_valley[1])
                                                total_min_error += min_error

                                        avg_max_error = total_max_error / 5
                                        avg_min_error = total_min_error / 5
                                        total_error = avg_max_error + avg_min_error

                                        print("average errors:", avg_max_error, avg_min_error, total_error)

                                        results.append({
                                            'params': test_params,
                                            'max_error': avg_max_error,
                                            'min_error': avg_min_error,
                                            'total_error': total_error
                                        })

                                        if total_error < best_total_error:
                                            best_total_error = total_error
                                            best_params = test_params.copy()
                                            print(f"\nNew best parameters found (error: {total_error}):")
                                            for param, value in best_params.items():
                                                print(f"{param}: {value}")

        # print counter to verify
        print(f"\nTotal combinations tested: {counter}")
        
        # Sort results by total error
        results.sort(key=lambda x: x['total_error'])

        # save results to a file
        with open('tuning_results.txt', 'w') as f:
            for result in results:
                f.write(f"Params: {result['params']}, Max Error: {result['max_error']}, Min Error: {result['min_error']}, Total Error: {result['total_error']}\n")

        # Print top 10 results
        print("\nTop 10 parameter combinations:")
        for i, result in enumerate(results[:10]):
            print(f"\nRank {i+1}:")
            print(f"Max Error: {float(result['max_error']):.4f}")
            print(f"Min Error: {float(result['min_error']):.4f}")
            print(f"Total Error: {float(result['total_error']):.4f}")
            for param, value in result['params'].items():
                print(f"{param}: {float(value)}")

        return best_params

    else:  # method == 'optimize'
        # Define parameter bounds based on initial values
        bounds = [
            (initial_params['jointStiffness']*0.01, initial_params['jointStiffness']*100),   # jointStiffness
            (initial_params['jointStiffnessDampingRatio']*0.01, initial_params['jointStiffnessDampingRatio']*100),   # jointStiffnessDampingRatio 
            (initial_params['tendonExtendStiffness']*0.01, initial_params['tendonExtendStiffness']*100),   # tendonExtendStiffness
            (initial_params['tendonExtendStiffnessDampingRatio']*0.01, initial_params['tendonExtendStiffnessDampingRatio']*100),   # tendonExtendStiffnessDampingRatio
            (initial_params['tendonBendStiffness']*0.01, initial_params['tendonBendStiffness']*100),   # tendonBendStiffness
            (initial_params['tendonBendStiffnessDampingRatio']*0.01, initial_params['tendonBendStiffnessDampingRatio']*100),   # tendonBendStiffnessDampingRatio
            (initial_params.get('nonLinearStiffness', 10)*0.01, initial_params.get('nonLinearStiffness', 10)*100),   # nonLinearStiffness
            (initial_params.get('scaleFactor', 1)*0.9, initial_params.get('scaleFactor', 1)*1.1)    # scaleFactor
        ]
        
        # Convert initial parameters to array format
        x0 = np.array([
            initial_params['jointStiffness'],
            initial_params['jointStiffnessDampingRatio'],
            initial_params['tendonExtendStiffness'],
            initial_params['tendonExtendStiffnessDampingRatio'],
            initial_params['tendonBendStiffness'],
            initial_params['tendonBendStiffnessDampingRatio'],
            initial_params.get('nonLinearStiffness', 10),
            initial_params.get('scaleFactor', 1)
        ])

        def objective(x):
            # Create hyperparameters dictionary from current values
            test_params = {
                'jointStiffness': x[0],
                'jointStiffnessDampingRatio': x[1],
                'tendonExtendStiffness': x[2],
                'tendonExtendStiffnessDampingRatio': x[3],
                'tendonBendStiffness': x[4],
                'tendonBendStiffnessDampingRatio': x[5],
                'nonLinearStiffness': x[6],
                'scaleFactor': x[7]
            }

            total_max_error = 0
            total_min_error = 0

            modify_parameters(test_params)
            
            for model_num in range(1, 6): #exclude model 6
                model_name = f"model{model_num}"
                model_file = f"generatedModel/{model_name}.xml"
                # Update geometric parameters for this specific model
                model_params = {
                    'model1': {'beamA': 30, 'beamC': 30, 'theta': 30, 'tendonThickness': 1, 'tendonWidth': 1.6, 'hingeLength': 2},
                    'model2': {'beamA': 20, 'beamC': 20, 'theta': 30, 'tendonThickness': 1, 'tendonWidth': 1.6, 'hingeLength': 2},
                    'model3': {'beamA': 30, 'beamC': 30, 'theta': 40, 'tendonThickness': 1, 'tendonWidth': 1.6, 'hingeLength': 2},
                    'model4': {'beamA': 30, 'beamC': 30, 'theta': 30, 'tendonThickness': 0.5, 'tendonWidth': 1.6, 'hingeLength': 2},
                    'model5': {'beamA': 30, 'beamC': 30, 'theta': 30, 'tendonThickness': 1, 'tendonWidth': 1.6, 'hingeLength': 3}
                }
                modify_parameters(model_params[model_name])
                generate_model(vals, "2DModel.xml", model_file)
                
                # Pass nonlinear stiffness and scale factor to simulate function
                sim_angles, _, sim_torques = simulate(
                    model_file,
                    nonLinear=test_params['nonLinearStiffness'],
                    scaleFactor=test_params['scaleFactor']
                )
                
                exp_id = exp_ids[model_num-1]
                data = np.loadtxt(f'processed_real_data/{exp_id}.csv', delimiter=',', skiprows=1)
                exp_angles = data[:, 0]
                exp_torques = data[:, 4]
                
                # Get peaks and valleys
                sim_peak, sim_valley = find_critical_points(sim_angles, sim_torques)
                exp_peak, exp_valley = find_critical_points(exp_angles, exp_torques)
                
                if sim_peak and exp_peak:
                    max_error = abs(sim_peak[1] - exp_peak[1]) / abs(exp_peak[1])
                    total_max_error += max_error
                
                if sim_valley and exp_valley:
                    min_error = abs(sim_valley[1] - exp_valley[1]) / abs(exp_valley[1])
                    total_min_error += min_error

            avg_max_error = total_max_error / 5
            avg_min_error = total_min_error / 5
            total_error = avg_max_error + avg_min_error

            print(f"Current error: {total_error} (max: {avg_max_error:.3f}, min: {avg_min_error:.3f})")
            return total_error

        # Run optimization
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100}
        )

        # Convert optimized parameters back to dictionary format
        optimized_params = {
            'jointStiffness': result.x[0],
            'jointStiffnessDampingRatio': result.x[1],
            'tendonExtendStiffness': result.x[2],
            'tendonExtendStiffnessDampingRatio': result.x[3],
            'tendonBendStiffness': result.x[4],
            'tendonBendStiffnessDampingRatio': result.x[5],
            'nonLinearStiffness': result.x[6],
            'scaleFactor': result.x[7]
        }

        print("\nOptimization complete!")
        print(f"Final error: {result.fun}")
        print("\nOptimized parameters:")
        for param, value in optimized_params.items():
            print(f"{param}: {value:.2e}")

        return optimized_params

def analyze_results(results_file):
    """
    Analyze tuning results and create visualization of parameter impacts
    Args:
        results_file: Path to the results text file
    """
    # Parse results from file
    results = []
    with open(results_file, 'r') as f:
        for line in f:
            # Extract parameters and errors using string manipulation
            params_str = line[line.find('{'): line.find('}')+1]
            errors_str = line[line.find('}')+1:]
            
            # Convert string to dictionary using eval
            params = eval(params_str)
            
            # Extract errors using string manipulation
            max_error = float(errors_str.split('[')[1].split(']')[0])
            min_error = float(errors_str.split('[')[2].split(']')[0])
            total_error = float(errors_str.split('[')[3].split(']')[0])
            
            results.append({
                'params': params,
                'max_error': max_error,
                'min_error': min_error,
                'total_error': total_error
            })

    # Convert to numpy arrays for analysis
    params = {}
    for param in results[0]['params'].keys():
        params[param] = np.array([r['params'][param] for r in results])
    
    max_errors = np.array([r['max_error'] for r in results])
    min_errors = np.array([r['min_error'] for r in results])
    
    # Create subplots for each parameter's impact
    fig, axs = plt.subplots(3, 2, figsize=(10, 10))
    axs = axs.flatten()
    
    # Analyze each parameter's correlation with errors
    for idx, (param_name, param_values) in enumerate(params.items()):
        # Plot parameter vs errors
        axs[idx].scatter(param_values, max_errors, alpha=0.5, label='Max Error')
        axs[idx].scatter(param_values, min_errors, alpha=0.5, label='Min Error')
        
        # Calculate correlations
        max_corr = np.corrcoef(param_values, max_errors)[0,1]
        min_corr = np.corrcoef(param_values, min_errors)[0,1]
        
        axs[idx].set_xlabel(param_name)
        axs[idx].set_ylabel('Error')
        axs[idx].set_title(f'{param_name}\nMax Corr: {max_corr:.3f}, Min Corr: {min_corr:.3f}')
        axs[idx].legend()
        
        # Scientific notation for large numbers
        if param_values.max() > 1000:
            axs[idx].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    plt.tight_layout()
    plt.show()
    
    # Print statistical summary
    print("\nParameter Impact Analysis:")
    for param_name, param_values in params.items():
        print(f"\n{param_name}:")
        print(f"  Range: {param_values.min():.2e} to {param_values.max():.2e}")
        print(f"  Best value for max error: {param_values[np.argmin(max_errors)]:.2e}")
        print(f"  Best value for min error: {param_values[np.argmin(min_errors)]:.2e}")



def modify_parameters(param_dict):
    """
    Modify parameters in vals dictionary
    param_dict: dictionary of parameter names and values
    """
    for param, value in param_dict.items():
        vals[param] = value
    #print("After update:", vals)

def find_critical_points(x, y):
    # Check if array has enough points
    if len(y) < 3:  # Need at least 3 points to calculate slope
        return None, None
    
    peak = None
    valley = None
    
    # Find valley (global minimum)
    min_idx = np.argmin(y)
    valley = (x[min_idx], y[min_idx])
    
    # Find peak (maximum point before the valley)
    peak = None
    if min_idx > 0:  # Only look for peak if valley isn't at start
        peak_idx = np.argmax(y[:min_idx])
        peak = (x[peak_idx], y[peak_idx])
    
    return peak, valley

# def find_critical_points(x, y):
#     # Check if array has enough points
#     if len(y) < 2:
#         return None, None
        
#     peak = None
#     valley = None
    
#     # Find first peak (maximum in first positive region)
#     positive_region_start = None
#     for i in range(len(y)-1):
#         if y[i] > 0 and positive_region_start is None:
#             positive_region_start = i
#         if positive_region_start is not None and y[i] < 0:
#             # Found end of first positive region
#             peak_idx = positive_region_start + np.argmax(y[positive_region_start:i+1])
#             peak = (x[peak_idx], y[peak_idx])
#             break
    
#     # Find valley (minimum in second negative region)
#     negative_region_start = None
#     found_first_positive = False
#     for i in range(len(y)-1):
#         if y[i] > 0 and not found_first_positive:
#             found_first_positive = True
#         elif found_first_positive and y[i] < 0 and negative_region_start is None:
#             negative_region_start = i
#         elif negative_region_start is not None and y[i] > 0:
#             # Found end of second negative region
#             valley_idx = negative_region_start + np.argmin(y[negative_region_start:i+1])
#             valley = (x[valley_idx], y[valley_idx])
#             break
    
#     return peak, valley

def calculate_error(sim_data, exp_data):
    """
    Calculate error between simulation and experimental critical points
    Returns (peak_error, valley_error)
    """
    sim_peak, sim_valley = find_critical_points(sim_data[0], sim_data[1])
    exp_peak, exp_valley = find_critical_points(exp_data[0], exp_data[1])
    
    peak_error = 0
    valley_error = 0
    
    # Calculate peak error if both peaks exist
    if sim_peak and exp_peak:
        peak_error = (sim_peak[1] - exp_peak[1])**2
    
    # Calculate valley error if both valleys exist
    if sim_valley and exp_valley:
        valley_error = (sim_valley[1] - exp_valley[1])**2
    
    return np.sqrt(peak_error), np.sqrt(valley_error)

def plot_comparison(sim_data, exp_data, title):
    """
    Plot simulation vs experimental data
    """
    plt.figure()
    plt.plot(sim_data[0], sim_data[1], 'b-', label='Simulation')
    plt.plot(exp_data[0], exp_data[1], 'r--', label='Experimental')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Force (N)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Define parameters for each model
    parameter_sets = {
        'model1': {'beamA': 30, 'beamC': 30, 'theta': 30, 'tendonThickness': 1, 'tendonWidth': 1.6, 'hingeLength': 2},
        'model2': {'beamA': 20, 'beamC': 20, 'theta': 30, 'tendonThickness': 1, 'tendonWidth': 1.6, 'hingeLength': 2},
        'model3': {'beamA': 30, 'beamC': 30, 'theta': 40, 'tendonThickness': 1, 'tendonWidth': 1.6, 'hingeLength': 2},
        'model4': {'beamA': 30, 'beamC': 30, 'theta': 30, 'tendonThickness': 0.5, 'tendonWidth': 1.6, 'hingeLength': 2},
        'model5': {'beamA': 30, 'beamC': 30, 'theta': 30, 'tendonThickness': 1, 'tendonWidth': 1.6, 'hingeLength': 3},
        'model6': {'beamA': 25, 'beamC': 30, 'theta': 30, 'tendonThickness': 1, 'tendonWidth': 1.6, 'hingeLength': 2}
    }

    # Create subplots for force and torque
    fig_force, axs_force = plt.subplots(2, 3, figsize=(15, 10))
    fig_torque, axs_torque = plt.subplots(2, 3, figsize=(15, 10))

    # Flatten axes arrays for easier indexing
    axs_force = axs_force.flatten()
    axs_torque = axs_torque.flatten()

    exp_ids = ['id1', 'id2', 'id3', 'id4', 'id5', 'id6ver1']

    # Variables to track global min/max values
    force_min, force_max = float('inf'), float('-inf')
    torque_min, torque_max = float('inf'), float('-inf')

    # First pass to collect all data and find global min/max
    all_data = []
    for model_name, params in parameter_sets.items():
        # [Same parameter setup and simulation code as before...]
        base_params = {
            'jointStiffness': 2e5,
            'jointStiffnessDampingRatio': 2e2,
            'tendonExtendStiffness': 110000000.00000001,
            'tendonExtendStiffnessDampingRatio': 20,
            'tendonBendStiffness': 14e10,
            'tendonBendStiffnessDampingRatio': 1e5
        }
        base_params = {
            'jointStiffness': 300000.0,
            'jointStiffnessDampingRatio': 100.0,
            'tendonExtendStiffness': 100000000.0,
            'tendonExtendStiffnessDampingRatio': 12.0,
            'tendonBendStiffness': 140000000000.0,
            'tendonBendStiffnessDampingRatio': 100000.0
        }
        base_params.update(params)
        modify_parameters(base_params)
        model_file = f"generatedModel/{model_name}.xml"
        generate_model(vals, "2DModel.xml", model_file)
        sim_angles, sim_forces, sim_torques = simulate(model_file)
        
        exp_id = 'id6ver1' if model_name == 'model6' else f'id{model_name[-1]}'
        data = np.loadtxt(f'processed_real_data/{exp_id}.csv', delimiter=',', skiprows=1)
        exp_angles = data[:, 0]
        exp_forces = data[:, 1]
        exp_torques = data[:, 4]

        # Update global min/max
        force_min = min(force_min, np.min(sim_forces), np.min(exp_forces))
        force_max = max(force_max, np.max(sim_forces), np.max(exp_forces))
        torque_min = min(torque_min, np.min(sim_torques), np.min(exp_torques))
        torque_max = max(torque_max, np.max(sim_torques), np.max(exp_torques))

        all_data.append((sim_angles, sim_forces, sim_torques, exp_angles, exp_forces, exp_torques))

    # Add some padding to the limits (5% of range)
    force_padding = (force_max - force_min) * 0.05
    torque_padding = (torque_max - torque_min) * 0.05
    
    # Second pass to plot with consistent scales
    for idx, ((model_name, params), (sim_angles, sim_forces, sim_torques, exp_angles, exp_forces, exp_torques)) in enumerate(zip(parameter_sets.items(), all_data)):
        # Plot force comparison
        axs_force[idx].plot(sim_angles, sim_forces, 'b-', label='Simulation')
        axs_force[idx].plot(exp_angles, exp_forces, 'r--', label='Experimental')
        
        # Plot critical points for force
        sim_force_peak, sim_force_valley = find_critical_points(sim_angles, sim_forces)
        exp_force_peak, exp_force_valley = find_critical_points(exp_angles, exp_forces)
        
        if sim_force_peak:
            axs_force[idx].plot(sim_force_peak[0], sim_force_peak[1], 'go', label='Sim Peak')
        if sim_force_valley:
            axs_force[idx].plot(sim_force_valley[0], sim_force_valley[1], 'mo', label='Sim Valley')
        if exp_force_peak:
            axs_force[idx].plot(exp_force_peak[0], exp_force_peak[1], 'ko', label='Exp Peak')
        if exp_force_valley:
            axs_force[idx].plot(exp_force_valley[0], exp_force_valley[1], 'yo', label='Exp Valley')

        # Set consistent y-axis limits for force plots
        axs_force[idx].set_ylim(force_min - force_padding, force_max + force_padding)
        axs_force[idx].set_title(f'Force Comparison - {model_name}')
        axs_force[idx].set_xlabel('Angle (degrees)')
        axs_force[idx].set_ylabel('Force (N)')
        axs_force[idx].grid(True)
        axs_force[idx].legend()

        # Plot torque comparison
        axs_torque[idx].plot(sim_angles, sim_torques, 'b-', label='Simulation')
        axs_torque[idx].plot(exp_angles, exp_torques, 'r--', label='Experimental')
        
        # Plot critical points for torque
        sim_torque_peak, sim_torque_valley = find_critical_points(sim_angles, sim_torques)
        exp_torque_peak, exp_torque_valley = find_critical_points(exp_angles, exp_torques)
        
        if sim_torque_peak:
            axs_torque[idx].plot(sim_torque_peak[0], sim_torque_peak[1], 'go', label='Sim Peak')
        if sim_torque_valley:
            axs_torque[idx].plot(sim_torque_valley[0], sim_torque_valley[1], 'mo', label='Sim Valley')
        if exp_torque_peak:
            axs_torque[idx].plot(exp_torque_peak[0], exp_torque_peak[1], 'ko', label='Exp Peak')
        if exp_torque_valley:
            axs_torque[idx].plot(exp_torque_valley[0], exp_torque_valley[1], 'yo', label='Exp Valley')

        # Set consistent y-axis limits for torque plots
        axs_torque[idx].set_ylim(torque_min - torque_padding, torque_max + torque_padding)
        axs_torque[idx].set_title(f'Torque Comparison - {model_name}')
        axs_torque[idx].set_xlabel('Angle (degrees)')
        axs_torque[idx].set_ylabel('Torque (N·m)')
        axs_torque[idx].grid(True)
        axs_torque[idx].legend()

        # Calculate and print errors
        torque_peak_rmse, torque_valley_rmse = calculate_error(
            (sim_angles, sim_torques),
            (exp_angles, exp_torques)
        )
        print(f"\nModel {model_name}:")
        print(f"Torque Peak RMSE: {torque_peak_rmse}")
        print(f"Torque Valley RMSE: {torque_valley_rmse}")

    # Adjust layout and display plots
    fig_force.tight_layout()
    fig_torque.tight_layout()
    plt.show()

    # Get optimized parameters
    initial_stiff_params = {
            'jointStiffness': 2e5,
            'jointStiffnessDampingRatio': 2e2,
            'tendonExtendStiffness': 1e8,
            'tendonExtendStiffnessDampingRatio': 30,
            'tendonBendStiffness': 16e10,
            'tendonBendStiffnessDampingRatio': 2e5,
            'nonLinearStiffness': 10,
            'scaleFactor': 1.0
        }
    
    # UNCOMMENT this part of code to run the optimization model
    #exp_data = (exp_angles, exp_forces)
    # optimized_params = tune_stiffness_parameters(exp_ids, initial_stiff_params)
    # print("\nOptimized parameters:")
    # for param, value in optimized_params.items():
    #     print(f"{param}: {value}")

    analyze_results("tuning_results.txt")
    
if __name__ == "__main__":
    main()
