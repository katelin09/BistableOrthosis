import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import json
import mujoco
import shutil

# Add simulator to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(parent_dir, "simulator"))

import usingSimulator
import mjcSimulator as simulation
import customizeMjcModel

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

def load_hyperparameters():
    hyperparams_path = os.path.join(parent_dir, "hyperparameter_tuning", "optimized_params.json")
    with open(hyperparams_path, 'r') as f:
        return json.load(f)

def create_results_folder():
    results_dir = os.path.join(current_dir, "results2")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def generate_model_and_simulate(lengthA, lengthC, theta, output_dir, hyperparams):
    # Create geo_params with updated values
    geo_params = customizeMjcModel.DEFAULT_GEOPARAMS.copy()
    geo_params.update({
        'beamA': lengthA,
        'beamC': lengthC,
        'theta': theta
    })
    
    # Generate model using optimized hyperparameters
    model_path = os.path.join(output_dir, f"model_A{lengthA:.1f}_C{lengthC:.1f}_theta{theta:.1f}.xml")
    customizeMjcModel.generate_model(
        geo_params,
        hyperparams,
        "../simulator/orthosis_model.xml",
        model_path
    )
    
    # Create temporary directory for animation
    temp_dir = os.path.join(parent_dir, "temp_anim")
    os.makedirs(temp_dir, exist_ok=True)
    temp_anim_path = os.path.join(temp_dir, "simulation.gif")
    
    # Run simulation with parameters from hyperparams
    angles, forces, torques = simulation.simulate(
        model_path,
        nonLinear=hyperparams['nonLinearStiffness'],
        scaleFactor=hyperparams['scaleFactor'],
        byPos=True,
        plot=False,
        animate=True,
        animatefile=temp_anim_path
    )
    
    # Save torque-displacement curve with improved style
    plt.figure(figsize=(4, 4))
    plt.plot(angles, torques, linewidth=2, color='black')
    plt.xlabel('Angle (degrees)', fontsize=14, fontweight='bold')
    plt.ylabel('Torque (N·mm)', fontsize=14, fontweight='bold')
    plt.title(f'lengthA={lengthA:.1f}, lengthC={lengthC:.1f}, theta={theta:.1f}', 
              fontsize=16, pad=15, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(-30, 60)
    plt.gca().set_aspect('auto')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"torque_A{lengthA:.1f}_C{lengthC:.1f}_theta{theta:.1f}.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Move animation to results directory
    final_anim_path = os.path.join(output_dir, f"anim_A{lengthA:.1f}_C{lengthC:.1f}_theta{theta:.1f}.gif")
    shutil.move(temp_anim_path, final_anim_path)
    
    # Clean up temporary directory
    shutil.rmtree(temp_dir)

def main():
    # Load optimized hyperparameters
    hyperparams = load_hyperparameters()
    print("Loaded hyperparameters:", hyperparams)
    
    # Create results folder
    output_dir = create_results_folder()
    
    num_samples = 5
    # Define parameter ranges
    lengthA_values = np.linspace(20.0, 45.0, num_samples)  # 5 values
    lengthC_values = np.linspace(20.0, 45.0, num_samples)  # 5 values
    theta_values = np.linspace(20.0, 45.0, num_samples)    # 5 values
    
    # Generate models for different combinations
    for lengthA in lengthA_values:
        for lengthC in lengthC_values:
            for theta in theta_values:
                print(f"Generating model with lengthA={lengthA:.2f}, lengthC={lengthC:.2f}, theta={theta:.2f}")
                generate_model_and_simulate(lengthA, lengthC, theta, output_dir, hyperparams)
    
    print(f"Evaluation complete. Results saved in: {output_dir}")

if __name__ == "__main__":
    main() 