import xml.etree.ElementTree as ET
import numpy as np
import math
import os

# Default hyperparameters for material properties
DEFAULT_HYPERPARAMS = {
    "jointStiffness": 7e6,
    "jointDamping": 1e5,
    "tendonExtendStiffness": 7e6,
    "tendonExtendDamping": 1e5,
    "tendonBendStiffness": 7e6,
    "tendonBendDamping": 1e5,
    "nonLinearStiffness": 1.0,
    "scaleFactor": 1.0
}

# Default geometric parameters
DEFAULT_GEOPARAMS = {
    "beamB": 25,
    "beamA": 30,
    "beamC": 30,
    "theta": 30,
    "tendonThickness": 0.8,
    "tendonWidth": 1.6,
    "hingeLength": 2.0,
    "hingeThickness": 0.8,
    "hingeWidth": 2.0
}

# larger deformation
def scaleBend(coeff, width, thickness, length, power = 3):
    # k = E * I / L^3
    # I = b * h^3 / 12
    # k = E * b * h^3 / 12 / L^3 = E * b * h / L^3
    # b: width, h: thickness, L: length
    return coeff * width * (thickness **power) / length / 12 * 2
    # return coeff * width * (thickness / length) **power / 4 * 2

# tensile test
def scaleExtend(coeff, width, thickness, length, power = 1):
    # k = E * A / L
    # A = b * h
    # k = E * b * h / L
    # b: width, h: thickness, L: length
    return coeff * width * (thickness / length)**power * 2

def calculate_derived_parameters(geoparams):
    """
    Calculate derived geometric parameters based on input parameters
    """
    b = geoparams["beamB"]
    c = geoparams["beamC"]
    a = geoparams["beamA"]
    theta = geoparams["theta"]
    theta_radians = np.deg2rad(theta)
    
    d = np.sqrt(b**2 + c**2 - 2 * b * c * np.cos(theta_radians))
    cos_beta = (d**2 + b**2 - c**2) / (2 * b * d)
    beta_radians = np.arccos(cos_beta)
    
    _1 = b + a
    _2 = d
    L = np.sqrt(_1**2 + _2**2 - 2 * _1 * _2 * np.cos(beta_radians))
    
    derived_params = {
        "beta": np.rad2deg(beta_radians),
        "beamD": d,
        "tendonL": L
    }
    
    return derived_params

def scale_parameters_to_model_size(geoparams, hyperparams, scale_factor = 100.0):
    """
    Scale parameters to model size and combine geometric and hyperparameters
    """
    # Calculate derived geometric parameters
    derived_params = calculate_derived_parameters(geoparams)

    # Add derived parameters
    scaled_geoparams = {}
    scaled_geoparams.update(geoparams)  # Add original parameters first
    scaled_geoparams.update(derived_params)  # Then add derived parameters
    
    for key, value in scaled_geoparams.items():
        if key not in ["theta", "beta"]:  # Don't scale angles
            scaled_geoparams[key] = value / scale_factor
        else:
            scaled_geoparams[key] = value
    
    # Get hyperparameters (use defaults if not provided)
    scaled_hyperparams = {**DEFAULT_HYPERPARAMS, **hyperparams}
    
    # Combine all parameters
    scaled_params = {**scaled_geoparams, **scaled_hyperparams}
    
    # Scale stiffness parameters based on geometry
    scaled_params["jointStiffness"] = scaleBend(
        scaled_params["jointStiffness"],
        scaled_params["hingeWidth"],
        scaled_params["hingeThickness"],
        scaled_params["hingeLength"]
    )
    
    scaled_params["tendonBendStiffness"] = scaleBend(
        scaled_params["tendonBendStiffness"],
        scaled_params["tendonWidth"],
        scaled_params["tendonThickness"],
        scaled_params["tendonL"]
    )
    
    scaled_params["tendonExtendStiffness"] = scaleExtend(
        scaled_params["tendonExtendStiffness"],
        scaled_params["tendonWidth"],
        scaled_params["tendonThickness"],
        scaled_params["tendonL"]
    )

    # Scale damping parameters based on geometry
    scaled_params["jointDamping"] = scaleBend(
        scaled_params["jointDamping"],
        scaled_params["hingeWidth"], 
        scaled_params["hingeThickness"],
        scaled_params["hingeLength"]
    )
    
    scaled_params["tendonBendDamping"] = scaleBend(
        scaled_params["tendonBendDamping"],
        scaled_params["tendonWidth"],
        scaled_params["tendonThickness"], 
        scaled_params["tendonL"]
    )
    
    scaled_params["tendonExtendDamping"] = scaleExtend(
        scaled_params["tendonExtendDamping"],
        scaled_params["tendonWidth"],
        scaled_params["tendonThickness"],
        scaled_params["tendonL"]
    )
    
    # Scale nonlinear stiffness
    scaled_params["tendonBendNonLinear"] = scaleBend(
        scaled_params["nonLinearStiffness"],
        scaled_params["tendonWidth"],
        scaled_params["tendonThickness"],
        scaled_params["tendonL"]
    )

    scaled_params["nonLinearStiffness"] = scaleBend(
        scaled_params["nonLinearStiffness"],
        scaled_params["hingeWidth"],
        scaled_params["hingeThickness"],
        scaled_params["hingeLength"]
    )
    
    return scaled_params

def update_geom(geom, parameters, angle = 0, parent_end = None):
    name = geom.get("name")
    fromto_values = list(map(float, geom.get("fromto").split()))

    # Compute the direction vector of the current capsule
    start = np.array(fromto_values[:3])
    end = np.array(fromto_values[3:])
    
    # Compute the direction vector
    direction = np.array([np.cos(angle), -np.sin(angle), 0])
    norm = np.linalg.norm(direction)
    if norm == 0:
        return  # Avoid division by zero
    unit_direction = direction / norm

    # Compute the new start point
    if parent_end is not None:
        new_start = parent_end
    else:
        new_start = start

    # Compute the new endpoint with updated length
    new_length = parameters[name]
    new_end = new_start + unit_direction * new_length

    # Update the fromto attribute
    new_fromto = f"{new_start[0]} {new_start[1]} {new_start[2]} {new_end[0]} {new_end[1]} {new_end[2]}"
    geom.set("fromto", new_fromto)
    return new_end

def modify_model(xml_file, saved_file, parameters):
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Find and update the fromto attribute for each beam
    new_positions = {}
    for geom in root.findall(".//geom"):
        name = geom.get("name")
        if name == "beamB": 
            new_positions["beamB"] = update_geom(geom, parameters, angle = np.deg2rad(180))
        if name == "beamA": 
            new_positions["beamA"] = update_geom(geom, parameters, angle = 0)
        if name == "beamD": 
            new_positions["beamD"] = update_geom(geom, parameters, angle = np.deg2rad(parameters["beta"]), parent_end = new_positions["beamB"])
            
    
    # modify the body after the beam to start from new beam-lengths
    for body in (root.findall(".//body")):
        name = body.get("name")
        if name == "Anext":
            new_pos = new_positions["beamA"]
            body.set("pos", f"{new_pos[0]} {new_pos[1]} {new_pos[2]}")
        if name == "Dnext": 
            new_pos = new_positions["beamD"]
            body.set("pos", f"{new_pos[0]} {new_pos[1]} {new_pos[2]}")

    # Update the joint properties
    for joint in root.findall(".//joint"):
        if joint.get("name") != "PIP": 
            joint.set("stiffness", str(parameters["jointStiffness"]))
            joint.set("damping", str(parameters["jointDamping"]))

    # Set the tendon properties
    for tendon in root.findall(".//spatial"):    
        if tendon.get("name") == "extensionTendon": 
            new_solreflimit = f"-{parameters['tendonExtendStiffness']} -{parameters['tendonExtendDamping']}"
            tendon.set("solreflimit", new_solreflimit)
            new_range = f"0 {parameters['tendonL']}"
            tendon.set("range", new_range)
        if tendon.get("name") == "bendingTendon": 
            new_solreflimit = f"-{parameters['tendonBendStiffness']} -{parameters['tendonBendDamping']}"
            tendon.set("solreflimit", new_solreflimit)
            new_range = f"{parameters['tendonL']*0.9} {parameters['tendonL']*2}"
            tendon.set("range", new_range)

    tree.write(saved_file)

def generate_model(geoparams, hyperparams, xml_file=None, saved_file=None):
    """
    Generate a new model with the given parameters
    """
    if xml_file is None:
        xml_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "orthosis_model.xml")
    
    if saved_file is None:
        saved_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "updated_model.xml")
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(saved_file), exist_ok=True)
    
    # Scale parameters
    scaled_params = scale_parameters_to_model_size(geoparams, hyperparams)
    
    # Modify model
    modify_model(xml_file, saved_file, scaled_params)
    
    return saved_file, scaled_params

# Define the default values for the changing parameters
vals = {
    **DEFAULT_GEOPARAMS,
    **DEFAULT_HYPERPARAMS
}

# Generate the model with the current parameters
if __name__ == "__main__":
    generate_model(DEFAULT_GEOPARAMS, DEFAULT_HYPERPARAMS)