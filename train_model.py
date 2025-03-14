
import numpy as np
import customize
import simulation
from scipy.signal import argrelextrema
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from joblib import dump

def update_model_parameters(x):
    customize.vals['beamA'] = x[0]
    customize.vals['beamC'] = x[1]
    customize.vals['theta'] = x[2]
    customize.vals['tendonThickness'] = x[3]
    customize.vals['tendonWidth'] = x[4]
    customize.vals['hingeLength'] = x[5]
    customize.vals['hingeThickness'] = x[6]

    customize.generate_model(customize.vals, "2DModel.xml", "training.xml")


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
    _, _, torque_curve = simulation.simulate("training.xml")
    torqueDown, torqueUp = extract_torques(torque_curve)

    if torqueDown is None or torqueUp is None:
       return None, None
    
    return torqueDown, torqueUp


def generate_data(num_samples=1000):
    # Sample data points
    X = []
    y_down = []
    y_up = []

    for _ in range(num_samples):
        # Random sampling within bounds
        x_sample = [np.random.uniform(low, high) for low, high in x_bounds]
        torqueDown, torqueUp = get_torques(x_sample)
        
        if torqueDown is not None and torqueUp is not None:
            X.append(x_sample)
            y_down.append(torqueDown)
            y_up.append(torqueUp)

    X = np.array(X)
    y_down = np.array(y_down)
    y_up = np.array(y_up)
    
    return X, y_down, y_up


def test_models(model_down, model_up, X_test, y_test_down, y_test_up):
    # Make predictions on test data
    pred_down = model_down.predict(X_test)
    pred_up = model_up.predict(X_test)

    # Calculate errors
    errors_down = np.abs(pred_down - y_test_down)
    errors_up = np.abs(pred_up - y_test_up)

    # Calculate mean absolute error
    mean_error_down = np.mean(errors_down)
    mean_error_up = np.mean(errors_up)

    # Print results
    print(f"\nMean Absolute Error - Torque Down: {mean_error_down:.2f}")
    print(f"Mean Absolute Error - Torque Up: {mean_error_up:.2f}")

    # # Optional: Print detailed predictions vs actual values
    # for i in range(len(X_test)):
    #     print(f"\nTest case {i+1}:")
    #     print(f"Parameters: {[f'{p:.2f}' for p in X_test[i]]}")
    #     print(f"Torque Down - Predicted: {pred_down[i]:.2f}, Actual: {y_test_down[i]:.2f}, Error: {errors_down[i]:.2f}")
    #     print(f"Torque Up - Predicted: {pred_up[i]:.2f}, Actual: {y_test_up[i]:.2f}, Error: {errors_up[i]:.2f}")



x = [25.0, 30.0, 30.0, 1.0, 1.6, 2.0, 1.0]
x_bounds = [
    (12.0, 40.0),  # beamALength
    (10.0, 40.0),  # beamCLength
    (10.0, 40.0),  # theta
    (0.4, 2.0),   # tendonThickness
    (0.4, 5.0),   # tendonWidth
    (1.0, 3.0),  # hingeLength
    (0.4, 1.7),   # hingeThickness
]

# Generate data
# print("Generating training data...")
# X_train, y_train_down, y_train_up = generate_data(num_samples=1000)
# np.save('X_train.npy', X_train)
# np.save('y_train_down.npy', y_train_down)
# np.save('y_train_up.npy', y_train_up)
# print("Training data generated and saved.")

# print("Generating testing data...")
# X_test, y_test_down, y_test_up = generate_data(num_samples=10)
# np.save('X_test.npy', X_test)
# np.save('y_test_down.npy', y_test_down)
# np.save('y_test_up.npy', y_test_up)
# print("Testing data generated and saved.")

# Load training data
print("Loading training data...")
X_train = np.load('X_train.npy')
y_train_down = np.load('y_train_down.npy')
y_train_up = np.load('y_train_up.npy')

# Train models using random forest
print("Training models...")
# Create polynomial features and linear regression pipeline
model_down = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
model_up = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())

model_down.fit(X_train, y_train_down)
model_up.fit(X_train, y_train_up)

# Load testing data
X_test = np.load('X_test.npy')
y_test_down = np.load('y_test_down.npy')
y_test_up = np.load('y_test_up.npy')
print("Testing models...")
test_models(model_down, model_up, X_test, y_test_down, y_test_up)

# Save models
dump(model_down, 'model_down.joblib')
dump(model_up, 'model_up.joblib')