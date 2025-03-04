import csv
import numpy as np
import matplotlib.pyplot as plt

def readData(filename):
    f = open(filename,"r")
    reader = csv.reader(f,delimiter="\t")
    data = np.array(list(reader),dtype=object)
    for i in range(1, data.shape[0]):  
        for j in range(data.shape[1]):  
            data[i, j] = float(data[i, j]) 
    return data

#real time data from 1-down
forceDownData = readData("1-down.tsv") 

Travel = forceDownData[:,2][1:] #unit: mm
Travel = Travel*0.001 #unit: m
load = forceDownData[:,1][1:] #unit: N

# convert travel (length) of actual data to angles
T = np.array(Travel, dtype=np.float64)
force_angles_rad = np.arctan(T/0.03) #T/lenA, unit in radians
force_angles = np.rad2deg(force_angles_rad) #unit: degrees
#force_angles = force_angles*360 - 1000

# convert load of actual data to relative perpentidular force
converted_load = np.array(load)*np.sin(force_angles_rad) #unit: N


# plot actual/real data
def plot_real_data(force_angles, converted_load):
    plt.plot(force_angles, converted_load, label="actual data")
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Force (N)')
    plt.grid(True)
    plt.legend()
    plt.show()

plot_real_data(force_angles, converted_load)

# read simulation data
angles = np.loadtxt("angles.txt")
total_forces = np.loadtxt("total_force.txt")

# Compare the total force vs. angle for simulation and actual data
plt.figure(figsize=(10, 6))
plt.plot(angles, total_forces,"-o",label="simulation data")
plt.plot(force_angles, converted_load,label="actual data")
plt.xlabel("Angle (degrees)")
plt.ylabel("Total Force (N)")
plt.title("Total Force vs. Angle for Orthosis Structure")
plt.grid()
plt.show()