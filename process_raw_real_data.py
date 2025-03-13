import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Define the path to the raw data folder
folder_path = 'raw_data/'

# List of files to read
files = ['m1-s1-p1.csv', 'm1-s1-p2.csv', 'm1-s1-p3.csv']
moment_arm_down = 40

# Initialize a figure
plt.figure(figsize=(10, 6))

# Before running the code, clean up the data by trimming the first and last few rows that the load is zero

# The first file: from straight to bend, the positive direction is down
file_path = os.path.join(folder_path, files[0])
data1 = pd.read_csv(file_path, skiprows=5, header=0, sep='\t')
data1['Angle'] = np.arctan(data1['Travel'] / moment_arm_down)
data1['Force'] = data1['Load'] * np.cos(data1['Angle'])
data1['Torque'] = data1['Load'] * moment_arm_down
data1['Angle'] = np.rad2deg(data1['Angle'])
# Shift data to start from (0,0)
data1['Angle'] = data1['Angle'] - data1['Angle'].iloc[0]
data1['Travel'] = data1['Travel'] - data1['Travel'].iloc[0]
# plt.plot(data1['Angle'], data1['Force'])

# The second file: from bend to straight, the positive direction is up, but flip the whole data in x axis for the bending process
file_path = os.path.join(folder_path, files[1])
data2 = pd.read_csv(file_path, skiprows=5, header=0, sep='\t')
# shift the data to start from the last point of 'p1' data
data2 = data2.iloc[::-1]
min_travel = data2['Travel'].min()
data2['Travel'] = data2['Travel'] - min_travel + data1['Travel'].iloc[-1]
data2['Angle'] = np.arctan(data2['Travel'] / (moment_arm_down + 2)) # 2 is the approximate offset
data2['Force'] = data2['Load'] * np.cos(data2['Angle'])
data2['Torque'] = data2['Load'] * moment_arm_down
data2['Angle'] = np.rad2deg(data2['Angle'])
# plt.plot(data2['Angle'], data2['Force'])

# The third file: from bend to bend more, after the second state, the positive direction is up
file_path = os.path.join(folder_path, files[2])
data3 = pd.read_csv(file_path, skiprows=5, header=0, sep='\t')
data3['Travel'] = data3['Travel'] - data3['Travel'].iloc[0] # shift the data travel from 0,0
data3['Angle'] = np.arcsin(data3['Travel'] / moment_arm_down)
data3['Force'] = data3['Load'] * np.cos(data3['Angle'])
data3['Angle'] = np.rad2deg(data3['Angle'])
data3['Torque'] = data3['Load'] * moment_arm_down
# Shift data to start from the last point of 'p2' data
data3['Angle'] = data3['Angle'] + data2['Angle'].iloc[-1]
# plt.plot(data3['Angle'], data3['Force'])



data = pd.concat([data1[['Angle', 'Force', 'Torque']], 
                  data2[['Angle', 'Force', 'Torque']], 
                  data3[['Angle', 'Force', 'Torque']]], ignore_index=True)

# Smooth the data
data['Force_smooth'] = data['Force'].ewm(span=5, adjust=False).mean()
data['Torque_smooth'] = data['Torque'].ewm(span=5, adjust=False).mean()

plt.plot(data['Angle'], data['Force_smooth'])

plt.xlabel('Angle (degrees)')
plt.ylabel('Load')
plt.title('Relationship between Force and Travel')
plt.legend()
plt.grid(True)
plt.show()


# Save the processed data to a new CSV file
folder_path = 'processed_data/'
output_file = 'm1-s1.csv'
output_path = os.path.join(folder_path, output_file)
data.to_csv(output_path, index=False)
print(f"Processed data saved to {output_path}")