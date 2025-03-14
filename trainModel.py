
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from joblib import dump
import usingSimulator


def generate_data(num_samples=1000):
    X = []
    y_down = []
    y_up = []

    for _ in range(num_samples):
        x_sample = [np.random.uniform(low, high) for low, high in x_bounds]
        torqueDown, torqueUp = usingSimulator.get_torques(x_sample)
        
        if torqueDown is not None and torqueUp is not None:
            X.append(x_sample)
            y_down.append(torqueDown)
            y_up.append(torqueUp)

    X = np.array(X)
    y_down = np.array(y_down)
    y_up = np.array(y_up)
    
    return X, y_down, y_up


def test_models(model_down, model_up, X_test, y_test_down, y_test_up):
    pred_down = model_down.predict(X_test)
    pred_up = model_up.predict(X_test)

    errors_down = np.abs(pred_down - y_test_down)
    errors_up = np.abs(pred_up - y_test_up)

    # mean absolute error
    mean_error_down = np.mean(errors_down)
    mean_error_up = np.mean(errors_up)

    print(f"\nMean Absolute Error - Torque Down: {mean_error_down:.2f}")
    print(f"Mean Absolute Error - Torque Up: {mean_error_up:.2f}")

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

dir = 'trained_model_from_simu_data/'

# TODO To improve the model
# 1. When generate data, we could sample the data in a "grid"
# 2. We could try other models

# Step1: Generate data
# print("Generating training data...")
# X_train, y_train_down, y_train_up = generate_data(num_samples=1000)
# np.save(dir + 'X_train.npy', X_train)
# np.save(dir + 'y_train_down.npy', y_train_down)
# np.save(dir + 'y_train_up.npy', y_train_up)
# print("Training data generated and saved.")

# print("Generating testing data...")
# X_test, y_test_down, y_test_up = generate_data(num_samples=10)
# np.save(dir + 'X_test.npy', X_test)
# np.save(dir + 'y_test_down.npy', y_test_down)
# np.save(dir + 'y_test_up.npy', y_test_up)
# print("Testing data generated and saved.")

# Step2: Train model (don't need to collect data again)
# Load data
print("Loading data...")
X_train = np.load(dir + 'X_train.npy')
y_train_down = np.load(dir + 'y_train_down.npy')
y_train_up = np.load(dir + 'y_train_up.npy')

X_test = np.load(dir + 'X_test.npy')
y_test_down = np.load(dir + 'y_test_down.npy')
y_test_up = np.load(dir + 'y_test_up.npy')

print("Training models...")
# Create polynomial features and linear regression pipeline
model_down = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
model_up = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())

model_down.fit(X_train, y_train_down)
model_up.fit(X_train, y_train_up)

# Step 3: Test model
print("Testing models...")
test_models(model_down, model_up, X_test, y_test_down, y_test_up)

# Step 4: Save models
dump(model_down, dir + 'model_down.joblib')
dump(model_up, dir + 'model_up.joblib')