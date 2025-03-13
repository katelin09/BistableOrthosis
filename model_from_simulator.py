import numpy as np
from scipy.signal import argrelextrema
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
import csv
import matplotlib.pyplot as plt
import json
import customize
import simulation
from sklearn.linear_model import RidgeCV
import random




def update_model_parameters(x):
    customize.vals['beamA'] = x[0]
    customize.vals['beamC'] = x[1]
    customize.vals['theta'] = x[2]
    customize.vals['tendonThickness'] = x[3]
    customize.vals['tendonWidth'] = x[4]
    customize.vals['hingeLength'] = x[5]
    customize.vals['hingeThickness'] = x[6]

    customize.generate_model(customize.vals, "2DModel.xml", "testing.xml")


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


def sample_data(x0, bounds, name_of_bounds, num_of_samples=10):
    # Create directory if it doesn't exist
    output_dir = "plot_for_model"
    filename = f"{output_dir}/torque_data.csv"
    
    # Open CSV file for writing
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['para_name', 'para_value', 'torque_down', 'torque_up', 'x[0]', 'x[1]', 'x[2]', 'x[3]', 'x[4]', 'x[5]', 'x[6]'])
        
        # For each parameter
        for i in range(len(bounds)):
            for j in range(num_of_samples):
                # Create parameter vector with varied value for current parameter
                x = x0.copy()
                param_value = bounds[i][0] + (bounds[i][1] - bounds[i][0]) / (num_of_samples - 1) * j
                x[i] = param_value
                
                # Run simulation and extract torque values
                update_model_parameters(x)
                _, _, np_torques = simulation.simulate("testing.xml", plot=False)
                torque_down, torque_up = extract_torques(np_torques)

                # Write to CSV
                writer.writerow([name_of_bounds[i], param_value, torque_down, torque_up, x[0], x[1], x[2], x[3], x[4], x[5], x[6]])
    
    print(f"Torque data saved to: {filename}")
    return filename


def extract_torques_from_csv(csv_filename):
    data = {}
    training_data = {'x': [], 'torque_down': [], 'torque_up': []}
    with open(csv_filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            param_name = row['para_name']
            if param_name not in data:
                data[param_name] = {'param_value': [], 'torque_down': [], 'torque_up': [], 'x': []}
            data[param_name]['param_value'].append(float(row['para_value']))
            data[param_name]['torque_down'].append(float(row['torque_down']))
            data[param_name]['torque_up'].append(float(row['torque_up']))
            # Extract x[0]-x[6] to numpy arrays "x"
            x = np.array([float(row['x[0]']), float(row['x[1]']), float(row['x[2]']), 
                         float(row['x[3]']), float(row['x[4]']), float(row['x[5]']), 
                         float(row['x[6]'])])
            data[param_name]['x'].append(x)
            training_data['x'].append(x)
            training_data['torque_down'].append(float(row['torque_down']))
            training_data['torque_up'].append(float(row['torque_up']))
    
    return data, training_data


def find_model_for_each_para_from_csv(data, name_of_bounds):
    degrees = [1, 2, 3, 4]
    
    for param_name in name_of_bounds:
        # Sort data by parameter value
        sorted_indices = np.argsort(data[param_name]['param_value'])
        x_vals = np.array([data[param_name]['param_value'][i] for i in sorted_indices]).reshape(-1, 1)
        y_vals_down = np.array([data[param_name]['torque_down'][i] for i in sorted_indices])
        y_vals_up = np.array([data[param_name]['torque_up'][i] for i in sorted_indices])
        
        # Torque down models and plots
        plt.figure(figsize=(12, 14))
        plt.scatter(x_vals, y_vals_down, color='black', label='Samples')
        
        for degree in degrees:
            poly = PolynomialFeatures(degree=degree)
            x_poly = poly.fit_transform(x_vals)
            
            model_down = LinearRegression().fit(x_poly, y_vals_down)
            y_pred_down = model_down.predict(x_poly)
            error_down = np.mean((y_vals_down - y_pred_down) ** 2)
            expression = f"{model_down.intercept_:.2f}"
            for i in range(1, degree + 1):
                expression += f" + {model_down.coef_[i]:.2f}* x^{i}"
            
            plt.plot(x_vals, y_pred_down, label=f'Degree: {degree}, {expression} (Error: {error_down:.2f})')

            x_ploy_i = poly.fit_transform((1 / x_vals))
            model_down = LinearRegression().fit(x_ploy_i, y_vals_down)
            y_pred_down = model_down.predict(x_ploy_i)
            error_down = np.mean((y_vals_down - y_pred_down) ** 2)
            expression = f"{model_down.intercept_:.2f}"
            for i in range(1, degree + 1):
                expression += f" + {model_down.coef_[i]:.2f}* (1/x)^{i}"

            plt.plot(x_vals, y_pred_down, label=f'Degree: {-degree}, {expression} (Error: {error_down:.2f})')
         
        plt.title(f'Model for {param_name} vs Torque Down')
        plt.xlabel(param_name)
        plt.ylabel('Torque Down')
        # Move legend below the plot
        plt.legend(bbox_to_anchor=(0.5, -0.05), loc='upper center', ncol=1, fontsize=11)
        plt.subplots_adjust(bottom=0.2)
        plt.grid(True)
        plt.savefig(f'plot_for_model/{param_name}_torque_down_model.png')
        
        # Torque up models and plots
        plt.figure(figsize=(12, 14))
        plt.scatter(x_vals, y_vals_up, color='black', label='Samples')
        
        for degree in degrees:
            poly = PolynomialFeatures(degree=degree)
            x_poly = poly.fit_transform(x_vals)
            
            model_up = LinearRegression().fit(x_poly, y_vals_up)
            y_pred_up = model_up.predict(x_poly)
            error_up = np.mean((y_vals_up - y_pred_up) ** 2)
            expression = f"{model_up.intercept_:.2f}"
            for i in range(1, degree + 1):
                expression += f" + {model_up.coef_[i]:.2f}* x^{i}"
            
            plt.plot(x_vals, y_pred_up, label=f'Degree: {degree}, {expression} (Error: {error_up:.2f})')
            
            x_ploy_i = poly.fit_transform((1 / x_vals))
            model_up = LinearRegression().fit(x_ploy_i, y_vals_up)
            y_pred_up = model_up.predict(x_ploy_i)
            error_up = np.mean((y_vals_up - y_pred_up) ** 2)
            expression = f"{model_up.intercept_:.2f}"
            for i in range(1, degree + 1):
                expression += f" + {model_up.coef_[i]:.2f}* (1/x)^{i}"
            plt.plot(x_vals, y_pred_up, label=f'Degree: {-degree}, {expression} (Error: {error_up:.2f})')
          
        plt.title(f'Model for {param_name} vs Torque Up')
        plt.xlabel(param_name)
        plt.ylabel('Torque Up')
        # Move legend below the plot
        plt.legend(bbox_to_anchor=(0.5, -0.05), loc='upper center', ncol=1, fontsize=11)
        plt.subplots_adjust(bottom=0.2)
        plt.grid(True)
        plt.savefig(f'plot_for_model/{param_name}_torque_up_model.png')
        
    
def build_torque_models(data, name_of_bounds, degree_down, degree_up):
    models_down = []
    models_up = []
    
    # Set up the plot grid
    n_params = len(name_of_bounds)
    fig_down, axes_down = plt.subplots(n_params, 1, figsize=(10, 5*n_params))
    fig_up, axes_up = plt.subplots(n_params, 1, figsize=(10, 5*n_params))
    
    # Process each parameter independently
    for i, param_name in enumerate(name_of_bounds):
        # Get data for current parameter
        x_vals = np.array(data[param_name]['param_value']).reshape(-1, 1)
        y_vals_down = np.array(data[param_name]['torque_down'])
        y_vals_up = np.array(data[param_name]['torque_up'])
        
        # Create fine x values for smooth plotting
        x_plot = np.linspace(min(x_vals), max(x_vals), 100).reshape(-1, 1)
        
        # Create models based on specified degrees
        if degree_down[i] > 0:
            model_down = make_pipeline(
                PolynomialFeatures(degree=degree_down[i]),
                LinearRegression()
            )
        else:
            model_down = make_pipeline(
                FunctionTransformer(lambda x: 1/x),
                PolynomialFeatures(degree=abs(degree_down[i])),
                LinearRegression()
            )
        
        if degree_up[i] > 0:
            model_up = make_pipeline(
                PolynomialFeatures(degree=degree_up[i]),
                LinearRegression()
            )
        else:
            model_up = make_pipeline(
                FunctionTransformer(lambda x: 1/x),
                PolynomialFeatures(degree=abs(degree_up[i])),
                LinearRegression()
            )
        
        # Fit models
        model_down.fit(x_vals, y_vals_down)
        model_up.fit(x_vals, y_vals_up)
        
        # Plot predictions
        axes_down[i].scatter(x_vals, y_vals_down, color='blue', label='Training data')
        axes_down[i].plot(x_plot, model_down.predict(x_plot), color='red', label='Model prediction')
        axes_down[i].set_title(f'{param_name} vs Torque Down (degree={degree_down[i]})')
        axes_down[i].set_xlabel(param_name)
        axes_down[i].set_ylabel('Torque Down')
        axes_down[i].legend()
        axes_down[i].grid(True)
        
        axes_up[i].scatter(x_vals, y_vals_up, color='blue', label='Training data')
        axes_up[i].plot(x_plot, model_up.predict(x_plot), color='red', label='Model prediction')
        axes_up[i].set_title(f'{param_name} vs Torque Up (degree={degree_up[i]})')
        axes_up[i].set_xlabel(param_name)
        axes_up[i].set_ylabel('Torque Up')
        axes_up[i].legend()
        axes_up[i].grid(True)
        
        # Store models with their parameter index
        models_down.append((model_down, i))
        models_up.append((model_up, i))
    
    # Adjust layout and save plots
    fig_down.tight_layout()
    fig_up.tight_layout()
    plt.savefig('plot_for_model/model_predictions.png')
    
    return models_down, models_up


def build_a_torque_model(training_data, degree_down, degree_up):
    # Convert training data to numpy arrays 
    X_raw = np.array(training_data['x'])
    y_down = np.array(training_data['torque_down'])
    y_up = np.array(training_data['torque_up'])

    # Define powers for each parameter based on degree_down and degree_up
    powers_down = []
    powers_up = []
    
    for deg in degree_down:
        if deg > 0:
            powers_down.append(list(range(1, deg + 1)))  # Positive degrees
        else:
            powers_down.append(list(range(-1, deg - 1, -1)))  # Negative degrees
            
    for deg in degree_up:
        if deg > 0:
            powers_up.append(list(range(1, deg + 1)))  # Positive degrees
        else:
            powers_up.append(list(range(-1, deg - 1, -1)))  # Negative degrees

    # Transform features according to powers
    def transform_features(X, powers_list):
        features = []
        for i, powers in enumerate(powers_list):
            for power in powers:
                features.append(X[:, i] ** power)
        return np.column_stack(features)

    X_down = transform_features(X_raw, powers_down)
    X_up = transform_features(X_raw, powers_up)

    # Create and fit Ridge regression models with cross-validation
    model_down = RidgeCV(alphas=[0.1, 1.0, 10.0])
    model_up = RidgeCV(alphas=[0.1, 1.0, 10.0])
    
    # Fit the models
    model_down.fit(X_down, y_down)
    model_up.fit(X_up, y_up)
    
    # Calculate and print R² scores
    r2_down = model_down.score(X_down, y_down)
    r2_up = model_up.score(X_up, y_up)
    print(f"R² score for torque down model: {r2_down:.4f}")
    print(f"R² score for torque up model: {r2_up:.4f}")
    
    # Calculate and organize gradients based on degrees
    # Store the gradient models for each parameter
    grad_model_down = {}
    grad_model_up = {}
    
    coef_idx = 0
    for i, param in enumerate(name_of_bounds):
        coeffs = []
        if degree_down[i] > 0:
            # For positive degrees: regular polynomial derivatives
            for power in range(1, degree_down[i] + 1):
                coeffs.append(model_down.coef_[coef_idx] * power)
                coef_idx += 1
            grad_model_down[param] = RidgeCV().fit(X_down, y_down)
            grad_model_down[param].coef_ = np.array(coeffs)
        else:
            # For negative degrees: inverse function derivatives
            for power in range(-1, degree_down[i] - 1, -1):
                coeffs.append(-model_down.coef_[coef_idx] * power)
                coef_idx += 1
            grad_model_down[param] = RidgeCV().fit(X_down, y_down)
            grad_model_down[param].coef_ = np.array(coeffs)
    
    coef_idx = 0
    for i, param in enumerate(name_of_bounds):
        coeffs = []
        if degree_up[i] > 0:
            # For positive degrees: regular polynomial derivatives
            for power in range(1, degree_up[i] + 1):
                coeffs.append(model_up.coef_[coef_idx] * power)
                coef_idx += 1
            grad_model_up[param] = RidgeCV().fit(X_up, y_up)
            grad_model_up[param].coef_ = np.array(coeffs)
        else:
            # For negative degrees: inverse function derivatives
            for power in range(-1, degree_up[i] - 1, -1):
                coeffs.append(-model_up.coef_[coef_idx] * power)
                coef_idx += 1
            grad_model_up[param] = RidgeCV().fit(X_up, y_up)
            grad_model_up[param].coef_ = np.array(coeffs)
    
    return model_down, model_up, grad_model_down, grad_model_up


# def print_model(model_list, title, name_of_bounds, degree_list):
#     print(f"\n{title}:")
#     for model, index in model_list:
#         param_name = name_of_bounds[index]
#         degree = degree_list[index]
        
#         print(f"\n  Parameter: {param_name}")
        
#         if degree > 0:  # Polynomial model
#             linear_reg = model.named_steps['linearregression']
#             intercept = linear_reg.intercept_
#             coeffs = linear_reg.coef_
            
#             print(f"    Type: Polynomial (degree {degree})")
#             print(f"    Intercept: {intercept:.6f}")
#             print(f"    Coefficients: {', '.join([f'{c:.6f}' for c in coeffs])}")
            
#             # Create simplified formula representation
#             if degree == 1:
#                 formula = f"{intercept:.4f} + {coeffs[1]:.4f}*{param_name}"
#             elif degree == 2:
#                 formula = f"{intercept:.4f} + {coeffs[1]:.4f}*{param_name} + {coeffs[2]:.4f}*{param_name}²"
#             elif degree == 3:
#                 formula = f"{intercept:.4f} + {coeffs[1]:.4f}*{param_name} + {coeffs[2]:.4f}*{param_name}² + {coeffs[3]:.4f}*{param_name}³"
#             else:
#                 formula = f"{intercept:.4f} + [polynomial terms of degree {degree}]"
                
#             print(f"    Formula: {formula}")
                
#         else:  # Negative power model
#             linear_reg = model.named_steps['linearregression']
#             intercept = linear_reg.intercept_
#             coeffs = linear_reg.coef_
            
#             print(f"    Type: Negative power (x^{degree})")
#             print(f"    Intercept: {intercept:.6f}")
#             print(f"    Coefficients: {', '.join([f'{c:.6f}' for c in coeffs])}")
            
#             # Create formula for negative power model
#             formula = f"{intercept:.4f} + {coeffs[0]:.4f}*{param_name} + {coeffs[1]:.4f}/(x^{abs(degree)})"
#             print(f"    Formula: {formula}")


# def save_models_to_file(models_down, models_up, name_of_bounds, degree_down, degree_up):
#     models = {
#         'models_down': [],
#         'models_up': []
#     }
    
#     for model, index in models_down:
#         param_name = name_of_bounds[index]
#         degree = degree_down[index]
        
#         if degree > 0:
#             coeffs = model.named_steps['linearregression'].coef_
#             intercept = model.named_steps['linearregression'].intercept_
#             models['models_down'].append({
#                 'param_name': param_name,
#                 'type': 'polynomial',
#                 'degree': degree,
#                 'coefficients': coeffs.tolist(),
#                 'intercept': intercept
#             })
#         else:
#             coeffs = model.named_steps['linearregression'].coef_
#             intercept = model.named_steps['linearregression'].intercept_
#             models['models_down'].append({
#                 'param_name': param_name,
#                 'type': 'negative_power',
#                 'degree': degree,
#                 'coefficients': coeffs.tolist(),
#                 'intercept': intercept
#             })
    
#     for model, index in models_up:
#         param_name = name_of_bounds[index]
#         degree = degree_up[index]
        
#         if degree > 0:
#             coeffs = model.named_steps['linearregression'].coef_
#             intercept = model.named_steps['linearregression'].intercept_
#             models['models_up'].append({
#                 'param_name': param_name,
#                 'type': 'polynomial',
#                 'degree': degree,
#                 'coefficients': coeffs.tolist(),
#                 'intercept': intercept
#             })
#         else:
#             coeffs = model.named_steps['linearregression'].coef_
#             intercept = model.named_steps['linearregression'].intercept_
#             models['models_up'].append({
#                 'param_name': param_name,
#                 'type': 'negative_power',
#                 'degree': degree,
#                 'coefficients': coeffs.tolist(),
#                 'intercept': intercept
#             })
    
#     with open('torque_models.json', 'w') as f:
#         json.dump(models, f, indent=4)
    
#     print("Models saved to torque_models.json")


def print_and_save_model(model_down, model_up, grad_model_down, grad_model_up, name_of_bounds, degree_down, degree_up):
    # Print the combined model and save to a dictionary
    model_data = {
        'torque_down': {
            'intercept': model_down.intercept_,
            'alpha': model_down.alpha_,
            'parameters': {},
            'gradients': {}
        },
        'torque_up': {
            'intercept': model_up.intercept_,
            'alpha': model_up.alpha_,
            'parameters': {},
            'gradients': {}
        }
    }
    
    print("\nCombined Torque Model:")
    print("\nTorque Down Model:")
    print(f"Intercept: {model_down.intercept_:.4f}")
    print("Coefficients:")
    feature_idx = 0
    for i, param in enumerate(name_of_bounds):
        if degree_down[i] > 0:
            powers = range(1, degree_down[i] + 1)
        else:
            powers = range(-1, degree_down[i] - 1, -1)
        
        print(f"\n  {param}:")
        model_data['torque_down']['parameters'][param] = {}
        for power in powers:
            coef = model_down.coef_[feature_idx]
            print(f"    x^{power}: {coef:.4f}")
            model_data['torque_down']['parameters'][param][power] = coef
            feature_idx += 1
            
        # Add gradient information
        model_data['torque_down']['gradients'][param] = {
            'coefficients': grad_model_down[param].coef_.tolist(),
            'alpha': grad_model_down[param].alpha_
        }
        print(f"  Gradient coefficients: {grad_model_down[param].coef_}")
    
    print(f"Alpha selected: {model_down.alpha_:.4f}")

    print("\nTorque Up Model:")
    print(f"Intercept: {model_up.intercept_:.4f}")
    print("Coefficients:")
    feature_idx = 0
    for i, param in enumerate(name_of_bounds):
        if degree_up[i] > 0:
            powers = range(1, degree_up[i] + 1)
        else:
            powers = range(-1, degree_up[i] - 1, -1)
        
        print(f"\n  {param}:")
        model_data['torque_up']['parameters'][param] = {}
        for power in powers:
            coef = model_up.coef_[feature_idx]
            print(f"    x^{power}: {coef:.4f}")
            model_data['torque_up']['parameters'][param][power] = coef
            feature_idx += 1
            
        # Add gradient information
        model_data['torque_up']['gradients'][param] = {
            'coefficients': grad_model_up[param].coef_.tolist(),
            'alpha': grad_model_up[param].alpha_
        }
        print(f"  Gradient coefficients: {grad_model_up[param].coef_}")
    
    print(f"Alpha selected: {model_up.alpha_:.4f}")

    # Save model data to JSON file
    with open('combined_torque_model.json', 'w') as f:
        json.dump(model_data, f, indent=4)
    print("\nModel saved to combined_torque_model.json")



bounds = [
    (12.0, 40.0),  # beamA
    (10.0, 40.0),  # beamC
    (10.0, 40.0),  # theta
    (0.4, 2.0),   # tendonThickness
    (0.4, 5.0),   # tendonWidth
    (1.1, 3.0),  # hingeLength
    (0.4, 1.7),   # hingeThickness
]

name_of_bounds = ["beamALength", "beamCLength", "thetaDeg", "tendonThickness", "tendonWidth", "hingeLength", "hingeThickness"]

x0 = [25, 25, 30, 1.0, 1.6, 2.0, 1.0, 2.0]

# Step 1: sample data by varying each para based on x0 and save it to speed up the other functions
# sample_data(x0, bounds, name_of_bounds, num_of_samples=10)

# Step 2: extract the torque data from the sampled csv file
data, training_data = extract_torques_from_csv("plot_for_model/torque_data.csv")
# for key in data:
#     print(f"{key}:")
#     for i in range(len(data[key]['param_value'])):
#         print(f"{data[key]['x']}")

# Step 3: find the degree of each parameter
# find_model_for_each_para_from_csv(data, name_of_bounds) # Analyze parameter relationships and find degree
degree_down = [1,2,2,2,2,-2,2]
degree_up = [2,1,2,3,3,-3,2]

# Step 4 v1: Build torque models
# models_down, models_up = build_torque_models(data, name_of_bounds, degree_down, degree_up)

# Step 4 v2: directly build a torque model based on the data
model_down, model_up, grad_model_down, grad_model_up = build_a_torque_model(training_data, degree_down, degree_up)
# Step 4.5: Test the model with random parameters

def random_params_within_bounds(bounds):
    return [random.uniform(low, high) for low, high in bounds]

def predict_torques(x, model_down, model_up, degree_down, degree_up):
    # Transform features according to degrees
    features_down = []
    features_up = []
    
    for i, (param_val, deg_down, deg_up) in enumerate(zip(x, degree_down, degree_up)):
        if deg_down > 0:
            features_down.extend([param_val ** p for p in range(1, deg_down + 1)])
        else:
            features_down.extend([(1/param_val) ** p for p in range(1, abs(deg_down) + 1)])
            
        if deg_up > 0:
            features_up.extend([param_val ** p for p in range(1, deg_up + 1)])
        else:
            features_up.extend([(1/param_val) ** p for p in range(1, abs(deg_up) + 1)])
    
    # Make predictions
    torque_down = model_down.predict([features_down])[0]
    torque_up = model_up.predict([features_up])[0]
    
    return torque_down, torque_up

# Test with 10 random parameter sets
print("\nTesting model with random parameters:")
print("Parameters | Model Prediction | Simulator Result | Difference")
print("-" * 70)

total_error_down = 0
total_error_up = 0
n_tests = 10

for _ in range(n_tests):
    # Generate random parameters
    test_params = random_params_within_bounds(bounds)
    
    # Get model predictions
    pred_down, pred_up = predict_torques(test_params, model_down, model_up, degree_down, degree_up)
    
    # Get simulator results
    update_model_parameters(test_params)
    _, _, np_torques = simulation.simulate("testing.xml", plot=False)
    sim_down, sim_up = extract_torques(np_torques)

    if sim_down is None or sim_up is None:
        continue
    
    # Calculate errors
    error_down = abs(pred_down - sim_down)
    error_up = abs(pred_up - sim_up)
    total_error_down += error_down
    total_error_up += error_up
    
    print(f"Params: {[f'{p:.2f}' for p in test_params]}")
    print(f"Torque Down: {pred_down:.2f} | {sim_down:.2f} | {error_down:.2f}")
    print(f"Torque Up: {pred_up:.2f} | {sim_up:.2f} | {error_up:.2f}")
    print("-" * 70)

# Print average errors
avg_error_down = total_error_down / n_tests
avg_error_up = total_error_up / n_tests
print(f"\nAverage Error - Torque Down: {avg_error_down:.2f}")
print(f"Average Error - Torque Up: {avg_error_up:.2f}")

# Step 5: print and save the model
print_and_save_model(model_down, model_up, grad_model_down, grad_model_up, name_of_bounds, degree_down, degree_up)