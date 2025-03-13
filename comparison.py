import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# plot actual/real data
def plot_real_data(real_angles, real_data, ylabel):
    plt.figure(figsize=(10, 6))
    plt.plot(real_angles, real_data)
    plt.xlabel('Angle (degrees)')
    plt.ylabel(ylabel)
    plt.title("Real Data")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_real_simu_data(real_angles, real_data, simu_angles, simu_data, ylabel):
    plt.figure(figsize=(10, 6))
    plt.plot(real_angles, real_data, label="measured data")
    plt.plot(simu_angles, simu_data, label="simulation data")
    plt.xlabel('Angle (degrees)')
    plt.ylabel(ylabel)
    plt.title("Simu vs. Real Data")
    plt.grid(True)
    plt.legend()
    plt.show()


# load processed_data/m1-s1.csv
file_path = "processed_data/m1-s1.csv"
real_data = pd.read_csv(file_path, header=0)
real_angles = real_data['Angle']
real_forces = real_data['Force_smooth']
real_torques = real_data['Torque_smooth']
real_angles = real_angles.to_numpy()
real_forces = real_forces.to_numpy()
real_torques = real_torques.to_numpy()

# plot_real_data(real_angles, real_forces, 'Forces (N)')

# read simulation data
simu_angles = np.loadtxt("simu_angles.txt")
simu_forces = np.loadtxt("simu_forces.txt")
simu_torques = np.loadtxt("simu_torques.txt")

# dimension scale factor = 100; F=ma; TODO: check the scale of mass and acceleration
# simu_forces = simu_forces / 1200
# simu_torques = simu_torques / 8

# plot_real_simu_data(real_angles, real_forces, simu_angles, simu_forces, 'Forces (N)')
plot_real_simu_data(real_angles, real_torques, simu_angles, simu_torques, 'Torques (N.mm)')