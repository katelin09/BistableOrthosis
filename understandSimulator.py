import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
import csv
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
import usingSimulator



def sample_data(x0, bounds, name_of_bounds, num_of_samples=10, filename = "plot_ea_para/torque_data.csv"):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['para_name', 'para_value', 'torque_down', 'torque_up', 'x[0]', 'x[1]', 'x[2]', 'x[3]', 'x[4]', 'x[5]', 'x[6]'])
        
        for i in range(len(bounds)):
            for j in range(num_of_samples):
                x = x0.copy()
                param_value = bounds[i][0] + (bounds[i][1] - bounds[i][0]) / (num_of_samples - 1) * j
                x[i] = param_value
                
                # Run simulation and extract torque values
                torque_down, torque_up = usingSimulator.get_torques(x)

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
        plt.savefig(f'plot_ea_para/{param_name}_torque_down_model.png')
        
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
        plt.savefig(f'plot_ea_para/{param_name}_torque_up_model.png')
        
    
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
    plt.savefig('intermediate_files/model_predictions.png')
    
    return models_down, models_up


def build_a_torque_model(training_data, degree_down, degree_up):
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


def plot_parameter_influence(models_down, models_up, bounds, name_of_bounds):
    plt.figure(figsize=(15, 10))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(name_of_bounds)))
    
    ax1.set_title('Parameter Influence on Torque Down')
    ax1.set_xlabel('Normalized Parameter Value')
    ax1.set_ylabel('Torque Down')

    ax2.set_title('Parameter Influence on Torque Up')
    ax2.set_xlabel('Normalized Parameter Value')
    ax2.set_ylabel('Torque Up')
    
    for (model_down, i), (model_up, _), color, param_name, bound in zip(models_down, models_up, colors, name_of_bounds, bounds):
        # Create normalized x values
        x_norm = np.linspace(0, 1, 100)
        # Scale x values to actual parameter range
        x_actual = x_norm * (bound[1] - bound[0]) + bound[0]
        x_reshaped = x_actual.reshape(-1, 1)
        
        # Predict torques
        y_down = model_down.predict(x_reshaped)
        y_up = model_up.predict(x_reshaped)
        
        # Normalize predictions for comparison
        # y_down_norm = (y_down - np.min(y_down)) / (np.max(y_down) - np.min(y_down))
        # y_up_norm = (y_up - np.min(y_up)) / (np.max(y_up) - np.min(y_up))
        
        # Plot normalized curves
        ax1.plot(x_norm, y_down, color=color, label=param_name)
        ax2.plot(x_norm, y_up, color=color, label=param_name)
    
    ax1.grid(True)
    ax2.grid(True)
    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    plt.savefig('plot_ea_para/parameter_influence.png')
    plt.close()


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

filename = "plot_ea_para/torque_data.csv"
# Step 1: sample data by varying each para based on x0 and save it to speed up the other functions
# sample_data(x0, bounds, name_of_bounds, num_of_samples=10, filename = filename)

# Step 2: extract the torque data from the sampled csv file
data, training_data = extract_torques_from_csv(filename)
# for key in data:
#     print(f"{key}:")
#     for i in range(len(data[key]['param_value'])):
#         print(f"{data[key]['x']}")

# Step 3: find the degree of each parameter
find_model_for_each_para_from_csv(data, name_of_bounds) # Analyze parameter relationships and find degree
degree_down = [1,2,2,2,2,-2,2]
degree_up = [2,1,2,3,3,-3,2]

# Step 4: build the torque models
# models_down, models_up = build_torque_models(data, name_of_bounds, degree_down, degree_up)
# plot_parameter_influence(models_down, models_up, bounds, name_of_bounds)

# Step 5: build a torque model
model_down, model_up, grad_model_down, grad_model_up = build_a_torque_model(training_data, degree_down, degree_up)