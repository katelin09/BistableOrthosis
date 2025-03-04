import mujoco
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time

#import model
filename = "modified_model.xml" #the xml file name   #modified_model
model = mujoco.MjModel.from_xml_path(filename) 
data = mujoco.MjData(model)

#get id
pip_joint = model.joint("PIP")
pip_dof_adr = pip_joint.dofadr
pip_id = pip_joint.id
extension_tendon_id = model.tendon("extensionTendon").id
# bend_tendon_id = model.tendon("bendingTendon").id
hingeBA_id = model.joint("hingeBA").id
force_sensor_id = model.sensor("finger_force").id
torque_sensor_id = model.sensor("finger_torque").id
tendon_sensor_id = model.sensor("tendon_force").id
beamA_id = model.body("Bnext").id

#  get acturator control range
actuator_id = model.actuator("finger_bend").id
actuator_range = model.actuator("finger_bend").ctrlrange
# print(actuator_range)

#initialize angles and forces for plotting
angles = []
forces = []
torques = []
tendon_lengths = []
tendon_torques = []
potential_energy = []

target_angle = np.deg2rad(150) #unit: radians
angle_step = model.opt.timestep

# Renderer setup
renderer = mujoco.Renderer(model, height=480, width=640)

def run_simulation_by_pos(model, data, target_angle, angle_step):
    #initialize
    angles = []
    forces = []
    torques = []
    tendon_lengths = []
    potential_energy = []
    frames = []
    mujoco.mj_resetData(model, data)
    for current_angle in np.arange(0, target_angle, angle_step): 
        # Set the angle of the PIP joint
        data.qpos[pip_id] = current_angle  
        
        mujoco.mj_step(model, data)

        force = data.sensordata[1] #get y-axis force from sensor
        torque = data.sensordata[5] #get torque from sensor
        potential_energy.append(data.energy[1]) #get potential energy
        
        # TEST: 
        # data.qfrc_actuator, always [0,0,0,0] (applied generalized force)
        # data.xfrc_applied, always [0,0,0,0,0,0]x6 (applied Cartesian force/torque on body)
        # data.sensordata[0],[1],[5] changing ([1] and [5] sync); always 0: data.sensordata[3],[4]
        # changing: mjData.qfrc_passive[], data.efc_force[hingeBA_id], data.efc_force[pip_id]
        # putting sensor on PIP joint
        # test_ = data.sensordata
        # potential_energy.append(test_)
        # print(test_)

        angles.append(data.qpos[hingeBA_id]) # current_angle or data.qpos[hingeBA_id]
        forces.append(force)
        torques.append(torque)
        tendon_lengths.append(data.ten_length[extension_tendon_id]) # == data.ten_length[bend_tendon_id]

        renderer.update_scene(data, camera = model.camera("cam").id)
        frame = renderer.render()
        frames.append(frame)

    np_angles = np.rad2deg(np.array(angles)) #change angle unit to degrees
    np_forces = np.array(forces)
    np_torques = np.array(torques)
    np_tendon_lengths = np.array(tendon_lengths)
    np_potential_energy = np.array(potential_energy)
    return np_angles, np_forces, np_torques, np_tendon_lengths, np_potential_energy, frames


def run_simulation_by_actuator(model, data, target_angle, control_step = 0.001):
    #initialize
    angles = []
    forces = []
    torques = []
    tendon_lengths = []
    potential_energy = []
    frames = []
    mujoco.mj_resetData(model, data)
    current_angle = 0
    data.ctrl = 0  
    while current_angle < target_angle and data.ctrl < actuator_range[1]: 
        # Set the angle of the PIP joint
        mujoco.mj_step(model, data)

        # mj_step computes the quantities 
        # mjData.cacc: Body acceleration
        # mjData.cfrc_int: Interaction force with the parent body.
        # mjData.crfc_ext: External force acting on the body. 
        # are used to compute the output of certain sensors (force, acceleration etc.)
        # The computed force arrays cfrc_int and cfrc_ext currently suffer from a know bug, 
        # they do not take into account the effect of spatial tendons
        # force = data.sensordata[1] #get y-axis force from sensor
        force = data.sensordata[1]
        torque = data.sensordata[5] #get torque from sensor
        potential_energy.append(data.energy[1]) #get potential energy
        
        angles.append(data.qpos[hingeBA_id]) # current_angle or data.qpos[hingeBA_id]
        forces.append(force)
        torques.append(torque)
        tendon_lengths.append(data.ten_length[extension_tendon_id]) # == data.ten_length[bend_tendon_id]

        current_angle = data.qpos[hingeBA_id] # update current angle
        renderer.update_scene(data, camera = model.camera("cam").id)
        frame = renderer.render()
        frames.append(frame)
        data.ctrl += control_step

    np_angles = np.rad2deg(np.array(angles)) #change angle unit to degrees
    np_forces = np.array(forces)
    np_torques = np.array(torques)
    np_tendon_lengths = np.array(tendon_lengths)
    np_potential_energy = np.array(potential_energy)
    
    return np_angles, np_forces, np_torques, np_tendon_lengths, np_potential_energy, frames


# Plot function
def plot_relationship(x_data, y_data, x_label, y_label, title, color='orange'):
    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data, color=color)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid()
    plt.show()


def save_animation(frames, filename="simulation.gif", duration=50):
    # Convert each frame (a NumPy array) to a PIL Image.
    pil_frames = [Image.fromarray(frame) for frame in frames]

    # Save as an animated GIF. 
    # duration is in milliseconds per frame, loop=0 means infinite loop.
    pil_frames[0].save(filename,
                       save_all=True,
                       append_images=pil_frames[1:],
                       duration=duration,  # adjust as needed (50 ms/frame ~20 FPS)
                       loop=0)


# np_angles, np_forces, np_torques, np_tendon_lengths, np_potential_energy, frames = run_simulation_by_pos(model, data, target_angle, angle_step)
np_angles, np_forces, np_torques, np_tendon_lengths, np_potential_energy, frames = run_simulation_by_actuator(model, data, target_angle, angle_step)

plot_relationship(np_angles, np_tendon_lengths, "Angle (degrees)", "Tendon Length (m)", "Tendon Length vs. Angle for Orthosis Structure")
plot_relationship(np_angles, np_potential_energy, "Angle (degrees)", "Potential Energy (J)", "Potential Energy vs. Angle for Orthosis Structure")
plot_relationship(np_angles, np_forces, "Angle (degrees)", "Total Force (N)", "Total Force vs. Angle for Orthosis Structure")
# plot_relationship(np_angles, np_torques, "Angle (degrees)", "Torque (N*mm)", "Torque vs. Angle for Orthosis Structure")


# save_animation(frames, filename="simulation.gif", duration=50)

# # #  save angles, total_force
# np.savetxt("angles.txt", np_angles)
# np.savetxt("total_force.txt", np_forces)
# np.savetxt("torques.txt", np_torques)