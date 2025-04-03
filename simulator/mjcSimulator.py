import mujoco
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

def run_simulation_by_pos(model, data, target_angle, angle_step, nonLinear):
    # Renderer setup with higher resolution
    renderer = mujoco.Renderer(model, height=1080, width=1920)  # Full HD resolution

    #get id
    pip_joint = model.joint("PIP")
    pip_id = pip_joint.id
    extension_tendon_id = model.tendon("extensionTendon").id
    hingeBA_id = model.joint("hingeBA").id
    hingeAL_id = model.joint("hingeAL").id
    hingeDL_id = model.joint("hingeDL").id

    #initialize
    angles = []
    forces = []
    torques = []
    tendon_lengths = []
    potential_energy = []
    frames = []
    mujoco.mj_resetData(model, data)
    
    # Initialize all joint velocities to zero
    data.qvel[:] = 0
    
    for current_angle in np.arange(0, target_angle, angle_step):
        # Set the angle of the PIP joint
        data.qpos[pip_id] = current_angle
        data.qvel[pip_id] = 0

        # Step simulation multiple times for stability
        for _ in range(5):
            data.qvel[pip_id] = 0
            mujoco.mj_kinematics(model, data)
        
        # Get joint states
        theta_BA  = data.qpos[hingeBA_id]  # Joint angle in radians
        theta_dot_BA = data.qvel[hingeBA_id]  # Joint angular velocity
        theta_AL = data.qpos[hingeAL_id]
        theta_DL = data.qpos[hingeDL_id]
        theta_dot_AL = data.qvel[hingeAL_id]
        theta_dot_DL = data.qvel[hingeDL_id]

        # Compute nonlinear torque components
        # Base stiffness from joint properties
        K_theta = model.joint("hingeBA").stiffness
        # Nonlinear cubic term
        alpha = nonLinear
        # Damping coefficient
        C_theta = model.joint("hingeBA").damping
        
        # Compute total torque with nonlinear effects - negative since we want to oppose motion
        torque_BA = K_theta * theta_BA + alpha * theta_BA**3 + C_theta * theta_dot_BA
        torque_DL = K_theta * theta_DL + alpha * theta_DL**3 + C_theta * theta_dot_DL
        
        # Bistable flexure AL - negative quadratic + quartic, with proper damping
        torque_AL = -K_theta * theta_AL + alpha * theta_AL**3 - C_theta * theta_dot_AL

        # Apply computed torque
        data.ctrl[model.actuator("flexure_motor_BA").id] = -float(torque_BA)
        data.ctrl[model.actuator("flexure_motor_AL").id] = float(torque_AL)
        data.ctrl[model.actuator("flexure_motor_DL").id] = -float(torque_DL)
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


def run_simulation_by_actuator(model, data, target_angle, nonLinear, control_step = 0.001):
    # Renderer setup
    renderer = mujoco.Renderer(model, height=480, width=640)

    #get id
    pip_joint = model.joint("PIP")
    pip_id = pip_joint.id
    extension_tendon_id = model.tendon("extensionTendon").id
    hingeBA_id = model.joint("hingeBA").id

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
        # for _ in range(5):
        #     data.qvel[pip_id] = 0
        #     mujoco.mj_kinematics(model, data)
        
         # Get joint states
        theta = data.qpos[pip_id]  # Joint angle in radians
        theta_dot = data.qvel[pip_id]  # Joint angular velocity
        
        # Compute nonlinear torque components
        # Base stiffness from joint properties
        K_theta = model.joint("hingeBA").stiffness
        # Nonlinear cubic term
        alpha = nonLinear
        # Damping coefficient
        C_theta = model.joint("hingeBA").damping
        
        # Compute total torque with nonlinear effects
        torque = K_theta * theta + alpha * theta**3 + C_theta * theta_dot
        
        # Apply computed torque
        data.ctrl[model.actuator("flexure_motor").id] = torque
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
    plt.figure(figsize=(8, 8))  # Square figure
    plt.plot(x_data, y_data, color=color, linewidth=2)
    plt.xlabel(x_label, fontsize=14, fontweight='bold')
    plt.ylabel(y_label, fontsize=14, fontweight='bold')
    plt.title(title, fontsize=16, pad=15, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # Make plot square
    plt.gca().set_aspect('auto')
    # Add more padding
    plt.tight_layout()
    plt.show()


def save_animation(frames, filename="../results/simulation.gif", duration=50):
    # Convert each frame (a NumPy array) to a PIL Image.
    pil_frames = [Image.fromarray(frame) for frame in frames]

    # Save as an animated GIF. 
    # duration is in milliseconds per frame, loop=0 means infinite loop.
    pil_frames[0].save(filename,
                       save_all=True,
                       append_images=pil_frames[1:],
                       duration=duration,  # adjust as needed (50 ms/frame ~20 FPS)
                       loop=0)


def simulate(model, nonLinear = 3.0, scaleFactor = 1.0, byPos = True, plot=False, animate=False, animatefile="../results/simulation.gif"):
    filename = model
    model = mujoco.MjModel.from_xml_path(filename) 
    data = mujoco.MjData(model)

    target_angle = np.deg2rad(100) #unit: radians
    angle_step = model.opt.timestep

    if byPos:
        np_angles, np_forces, np_torques, np_tendon_lengths, np_potential_energy, frames = run_simulation_by_pos(model, data, target_angle, angle_step, nonLinear)
    else:
        np_angles, np_forces, np_torques, np_tendon_lengths, np_potential_energy, frames = run_simulation_by_actuator(model, data, target_angle, angle_step, nonLinear)

    # scale factor
    # np_forces = np_forces / 1200
    np_torques = np_torques * scaleFactor
    
    if plot:
        # debug steps: 1. check tenden length; 2. check potential energy; 3. check force and torque
        plot_relationship(np_angles, np_tendon_lengths, "Angle (degrees)", "Tendon Length (m)", "Tendon Length vs. Angle for Orthosis Structure")
        plot_relationship(np_angles, np_potential_energy, "Angle (degrees)", "Potential Energy (J)", "Potential Energy vs. Angle for Orthosis Structure")
        plot_relationship(np_angles, np_forces, "Angle (degrees)", "Total Force (N)", "Total Force vs. Angle for Orthosis Structure")
        plot_relationship(np_angles, np_torques, "Angle (degrees)", "Torque (N*mm)", "Torque vs. Angle for Orthosis Structure")

    if animate:
        # Create directory for animation if it doesn't exist
        os.makedirs(os.path.dirname(animatefile), exist_ok=True)
        save_animation(frames, filename=animatefile, duration=50)

    return np_angles, np_forces, np_torques


#simulate(model_file, nonLinear = 6.0, scaleFactor = 1.0, byPos = True, plot = False)