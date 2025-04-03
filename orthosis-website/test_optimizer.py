import sys
import os

# Add the optimizer directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(parent_dir, "optimizer"))

from optimization import customize, create_trig_features

# Run optimization with custom output directory
input_params = {
    'naturalAngle': 22.0,
    'ptorqueExtend': 30.0,
    'atorqueBend': 50.0,
    'atorqueExtend': 20.0
}
output_dir = "my_results"
x, torqueDown, torqueUp, mjc_model_file, geometry_vals, torque_curve = customize(input_params, strength="default", model_dir=None, output_dir=None)
print(f"Results saved to {output_dir}")
print(f"Model file: {mjc_model_file}")
print(f"Torque down: {torqueDown:.2f}")
print(f"Torque up: {torqueUp:.2f}")