import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Define the path to the raw data folder
folder_path = '/'

# List of files to read
files = ['1-1.csv', '2-2.csv', '1-2.csv']
moment_arm_down = 40
moment_arm_up = 10

# Initialize a figure
plt.figure(figsize=(10, 6))

file_path = os.path.join(folder_path, files[0])
data1 = pd.read_csv(file_path, skiprows=5, header=0, sep='\t')
data1 = data1[data1['Load'] > 0]
# a triangle abc, length a = 40*tan(alpha)-t, angle A = 30 deg; angle B = 90-alpha deg; length c = 40/cos(alpha) -40, I want to present alpha by t (travel)
# this is approximate
data1['Angle'] = np.arctan(- data1['Travel'] / 40)
# data1['Angle'] = np.arctan( data1['Travel'] / 39) 
data1['Force'] = data1['Load'] * np.sin(data1['Angle'])
# data1['torque'] = data1['Load'] * moment_arm_down
data1['Angle'] = np.rad2deg(data1['Angle'])
# Shift data to start from (0,0)
# shift the load data to start from 0
data1['Angle'] = data1['Angle'] - data1['Angle'].iloc[0]
plt.plot(data1['Angle'], data1['Force'], marker='o', linestyle='-')

file_path = os.path.join(folder_path, files[1])
data2 = pd.read_csv(file_path, skiprows=5, header=0, sep='\t')
# flip the whole data in x axis
data2 = data2.iloc[::-1]
data2['Angle'] = data2['Travel'] / 68
data2['Force'] = - data2['Load'] * np.sin(data2['Angle'])
# data1['torque'] = data1['Load'] * moment_arm_down
data2['Angle'] = np.rad2deg(data2['Angle'])
# Shift data to start from the last point of 'up' data
data2['Angle'] = data2['Angle'] - data2['Angle'].iloc[0] + data1['Angle'].iloc[-1]
plt.plot(data2['Angle'], data2['Force'], marker='o', linestyle='-')

file_path = os.path.join(folder_path, files[2])
data3 = pd.read_csv(file_path, skiprows=5, header=0, sep='\t')
data3['Angle'] = data3['Travel'] / 40
data3['Force'] = data3['Load'] * np.cos(data2['Angle'].iloc[-1] - data3['Angle'])
data3['Angle'] = np.rad2deg(data3['Angle'])
# # Shift data to start from the last point of 'down' data
data3['Angle'] = data3['Angle'] - data3['Angle'].iloc[0] + data2['Angle'].iloc[-1]
plt.plot(data3['Angle'], data3['Force'], marker='o', linestyle='-')

# # Add labels and title
plt.xlabel('Angle (degrees)')
plt.ylabel('Load')
plt.title('Relationship between Load and Travel')
plt.legend()
plt.grid(True)
plt.show()

data = pd.concat([data1[['Angle', 'Force']], data2[['Angle', 'Force']], data3[['Angle', 'Force']]], ignore_index=True)

# plt.plot(data['Angle'], data['Force'], marker='o', linestyle='-')
# plt.xlabel('Angle (degrees)')
# plt.ylabel('Load')
# plt.title('Relationship between Load and Travel')
# plt.legend()
# plt.grid(True)
# plt.show()

# smooth the data
# data['Force_smooth'] = data['Force'].rolling(window=5, center=True).mean()
data['Force_smooth'] = data['Force'].ewm(span=5, adjust=False).mean()
# plt.plot(data['Angle'], data['Force_smooth'], marker='o', linestyle='-')
# plt.xlabel('Angle (degrees)')
# plt.ylabel('Load')
# plt.title('Relationship between Load and Travel')
# plt.legend()
# plt.grid(True)
# plt.show()

# Save the processed data to a new CSV file
folder_path = 'processed_data/'
output_file = 'processed_data1.csv'
output_path = os.path.join(folder_path, output_file)
data.to_csv(output_path, index=False)
print(f"Processed data saved to {output_path}")