# Model Simulation Project

This project provides a framework for simulating and optimizing mechanical models, particularly focused on orthosis and tendon-based mechanisms. It includes tools for parameter optimization, torque modeling, and simulation visualization.

## Project Structure

```
.
├── simulator/                    # Core simulation components
│   ├── mjcSimulator.py          # Core MuJoCo simulation engine
│   ├── customizeMjcModel.py     # MuJoCo model customization
│   ├── usingSimulator.py        # Simulation interface
│   ├── orthosis_model.xml       # Base MuJoCo model definition
│   ├── temp/                    # Temporary simulation files
│   └── results/                 # Simulation results
│
├── hyperparameter_tuning/       # Model tuning and validation
│   ├── tuning.py               # Model tuning and parameter adjustment
│   ├── optimized_params.json   # Optimized hyperparameters
│   ├── processed_real_data/    # Processed experimental data
│   ├── temp/                   # Temporary tuning files
│   └── results/                # Tuning results
│
├── optimizer/                   # Parameter optimization
│   ├── optimization.py         # Parameter optimization algorithms
│   ├── trainModel.py          # Model training script
│   ├── temp/                   # Temporary optimization files
│   ├── results/                # Optimization results
│   └── trained_model_from_simu_data/ # Pre-trained ML models
│       ├── model_down.joblib   # Model for downward torque prediction
│       └── model_up.joblib     # Model for upward torque prediction
│
└── test_optimizer.py          # Test script for optimization
```

## Features

- **Simulator**: Core MuJoCo-based physics simulation engine for mechanical systems
- **Hyperparameter Tuning**: Tools for tuning model parameters using real experimental data
- **Optimizer**: Framework for optimizing mechanical parameters to achieve desired torque characteristics
- **ML Models**: Pre-trained models for fast optimization of mechanical parameters
- **Visualization**: Tools for plotting and analyzing simulation results

## Dependencies

- Python 3.x
- NumPy
- MuJoCo
- Matplotlib
- scikit-learn
- joblib

## Usage

### 1. Basic Simulation

```python
from simulator.mjcSimulator import simulate

# Run simulation with default parameters
angles, forces, torques = simulate("simulator/orthosis_model.xml", plot=True)
```

### 2. Hyperparameter Tuning

```python
from hyperparameter_tuning.tuning import objective_function

# Define model parameters
model_params = {
    'model1': {
        'beamA': 30,
        'beamC': 30,
        'theta': 30,
        'tendonThickness': 1,
        'tendonWidth': 1.6,
        'hingeLength': 2
    }
}

# Run tuning using real data
result = objective_function(x0, exp_ids, model_params, model_nums)
```

### 3. Parameter Optimization

The optimization process supports two modes:
- **Fast**: Uses pre-trained ML models for quick optimization
- **Slow**: Uses direct simulation for more accurate but slower optimization

```python
from optimizer.optimization import customize

# Define input parameters
input_params = {
    'naturalAngle': 22.0,    # Rest angle in degrees [0-90]
    'ptorqueExtend': 30.0,   # Minimum required torque for downward motion
    'atorqueBend': 50.0,     # Maximum allowable torque for downward motion
    'atorqueExtend': 20.0    # Maximum allowable torque for upward motion (must be negative)
}

# Run optimization with different strength configurations
x, torqueDown, torqueUp, mjc_model_file, geometry_vals, torque_curve = customize(
    input_params,
    strength="default",  # Options: "default", "soft", "stiff", "easyUp", "easyDown"
    optimizer="fast"     # Options: "fast", "slow"
)
```

### 4. Testing Optimization

```python
python3 test_optimizer.py
```

This will run optimization tests with different strength configurations:
- Default strength (balanced torques)
- Strong strength (higher torques)
- Soft strength (lower torques)

## Parameter Descriptions

### Geometric Parameters
- `beamALength`: Length of beam A (12.0-40.0 mm)
- `beamCLength`: Length of beam C (10.0-40.0 mm)
- `thetaDeg`: Angle parameter (10.0-40.0 degrees)
- `tendonThickness`: Thickness of tendon (0.4-2.0 mm)
- `tendonWidth`: Width of tendon (0.4-5.0 mm)
- `hingeLength`: Length of hinge (1.0-3.0 mm)
- `hingeThickness`: Thickness of hinge (0.4-1.7 mm)

### Torque Parameters
- `naturalAngle`: Rest angle of the joint (0-90 degrees)
- `ptorqueExtend`: Minimum required torque for downward motion
- `atorqueBend`: Maximum allowable torque for downward motion
- `atorqueExtend`: Maximum allowable torque for upward motion (must be negative)

### Optimization Strength Options
- `default`: Balanced torques between upper and lower bounds
- `soft`: Lower torques (1/3 of the range)
- `stiff`: Higher torques (2/3 of the range)
- `easyUp`: Lower upward torque, higher downward torque
- `easyDown`: Higher upward torque, lower downward torque

## Output Files

- Simulation results are saved in `simulator/results/`
- Tuning results are saved in `hyperparameter_tuning/results/`
- Optimization results are saved in `optimizer/results/`
- Pre-trained models are stored in `optimizer/trained_model_from_simu_data/`
- Temporary files are stored in respective `temp/` directories
- Optimized hyperparameters are stored in `hyperparameter_tuning/optimized_params.json`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Add your license information here] 