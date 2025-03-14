import mujoco
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def run_simulation_by_pos(model, data, target_angle, angle_step, pip_id, hingeBA_id, extension_tendon_id):
    # Renderer setup
    renderer = mujoco.Renderer(model, height=480, width=640)

    #initialize
    angles = []
    forces = []
    torques = []
    tendon_lengths = []
    potential_energy = []
    frames = []
    mujoco.mj_resetData(model, data)
    for current_angle in np.arange(0, target_angle, angle_step): 
        # Set the angle of the PIP joint, here data.ctrl = 0
        for _ in range(5):
            data.qpos[pip_id] = current_angle
            data.qvel[pip_id] = 0
            mujoco.mj_kinematics(model, data) #update the kinematics
        mujoco.mj_step(model, data)

        force = data.sensordata[1] #get y-axis force from sensor
        torque = -data.joint("PIP").qfrc_constraint + data.joint("PIP").qfrc_smooth
        potential_energy.append(data.energy[1]) #get potential energy: position-dependent energy
        
        angles.append(data.qpos[hingeBA_id]) 
        forces.append(force)
        torques.append(torque)
        tendon_lengths.append(data.ten_length[extension_tendon_id]) # == data.ten_length[bend_tendon_id]

        renderer.update_scene(data, camera = model.camera("cam").id)
        frame = renderer.render()
        frames.append(frame)

    np_angles = np.rad2deg(np.array(angles))
    np_forces = np.array(forces)
    np_torques = np.array(torques)
    np_tendon_lengths = np.array(tendon_lengths)
    np_potential_energy = np.array(potential_energy)

    return np_angles, np_forces, np_torques, np_tendon_lengths, np_potential_energy, frames


def run_simulation_by_actuator(model, data, target_angle, pip_id, hingeBA_id, extension_tendon_id, control_step = 0.001):
    # Renderer setup
    renderer = mujoco.Renderer(model, height=480, width=640)

    #initialize
    angles = []
    forces = []
    torques = []
    tendon_lengths = []
    potential_energy = []
    frames = []
    mujoco.mj_resetData(model, data)
    actuator_id = model.actuator("finger_bend").id
    actuator_range = model.actuator("finger_bend").ctrlrange
    current_angle = 0
    while current_angle < target_angle and data.ctrl < actuator_range[1]: 
        # actuator control
        data.ctrl[actuator_id] = data.ctrl[actuator_id] + control_step
        for _ in range(5):
            data.qvel[pip_id] = 0
            mujoco.mj_kinematics(model, data)
        mujoco.mj_step(model, data)

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


    np_angles = np.rad2deg(np.array(angles)) #change angle unit to degrees
    np_forces = np.array(forces)
    np_torques = np.array(torques)
    np_tendon_lengths = np.array(tendon_lengths)
    np_potential_energy = np.array(potential_energy)
    
    return np_angles, np_forces, np_torques, np_tendon_lengths, np_potential_energy, frames


def plot_relationship(x_data, y_data, x_label, y_label, title, color='orange'):
    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data, color=color)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid()
    plt.show()


def save_animation(frames, filename="intermediate_files/simulation.gif", duration=50):
    # Convert each frame (a NumPy array) to a PIL Image.
    pil_frames = [Image.fromarray(frame) for frame in frames]

    # Save as an animated GIF. 
    # duration is in milliseconds per frame, loop=0 means infinite loop.
    pil_frames[0].save(filename,
                       save_all=True,
                       append_images=pil_frames[1:],
                       duration=duration,  # adjust as needed (50 ms/frame ~20 FPS)
                       loop=0)


def simulate(model, byPos = True, plot=False, animate=False):
    #import model
    filename = model
    model = mujoco.MjModel.from_xml_path(filename) 
    data = mujoco.MjData(model)

    #get id
    pip_joint = model.joint("PIP")
    pip_id = pip_joint.id
    extension_tendon_id = model.tendon("extensionTendon").id
    hingeBA_id = model.joint("hingeBA").id

    target_angle = np.deg2rad(100) #unit: radians
    angle_step = model.opt.timestep

    if byPos:
        np_angles, np_forces, np_torques, np_tendon_lengths, np_potential_energy, frames = run_simulation_by_pos(model, data, target_angle, angle_step, pip_id, hingeBA_id, extension_tendon_id)
    else:
        np_angles, np_forces, np_torques, np_tendon_lengths, np_potential_energy, frames = run_simulation_by_actuator(model, data, target_angle, pip_id, hingeBA_id, extension_tendon_id, angle_step)

    # scale factor
    np_forces = np_forces / 1200
    np_torques = np_torques / 8
    
    if plot:
        # debug steps: 1. check tenden length; 2. check potential energy; 3. check force and torque
        plot_relationship(np_angles, np_tendon_lengths, "Angle (degrees)", "Tendon Length (m)", "Tendon Length vs. Angle for Orthosis Structure")
        plot_relationship(np_angles, np_potential_energy, "Angle (degrees)", "Potential Energy (J)", "Potential Energy vs. Angle for Orthosis Structure")
        plot_relationship(np_angles, np_forces, "Angle (degrees)", "Total Force (N)", "Total Force vs. Angle for Orthosis Structure")
        plot_relationship(np_angles, np_torques, "Angle (degrees)", "Torque (N*mm)", "Torque vs. Angle for Orthosis Structure")

    if animate:
        save_animation(frames, filename="intermediate_files/simulation.gif", duration=50)

    # save angles, forces, torques
    np.savetxt("simu_data/simu_angles.txt", np_angles)
    np.savetxt("simu_data/simu_forces.txt", np_forces)
    np.savetxt("simu_data/simu_torques.txt", np_torques)

    return np_angles, np_forces, np_torques


if __name__ == "__main__":
    simulate("intermediate_files/updated_model.xml", byPos = True, plot=False)