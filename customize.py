import xml.etree.ElementTree as ET
import numpy as np
import math

vals = {
    "beamB": 25, 
    "beamA": 25, 
    "beamC": 30,
    "theta": 30,
    "tendonThickness": 0.8,
    "hingeLength": 0,
    "hingeThickness": 0.8,
    "jointStiffness" : 20,
    "jointDamping" : 10,
    "tendonExtendStiffness" : 10000,
    "tendonExtendDamping" : 100,
    "tendonBendStiffness" : 100,
    "tendonBendDamping" : 100,
}

# some manual tuning tips:
# jointStiffness -> 50, no bistable
# tendonExtendStiffness -> 1000, no bistable
# "jointStiffness" : 20,
# "jointDamping" : 10,
# "tendonExtendStiffness" : 10000,
# "tendonExtendDamping" : 100,
# "tendonBendStiffness" : 100,
# "tendonBendDamping" : 100,

def vals_to_parameters(vals):
    # Initialize parameters using provided vals (or defaults)
    parameters = {}
    
    # Base parameters from vals (with defaults if not provided)
    parameters["beamB"] = vals.get("beamB", 25)
    parameters["beamA"] = vals.get("beamA", 25)
    parameters["beamC"] = vals.get("beamC", 25)
    parameters["theta"] = vals.get("theta", 30)
    parameters["tendonThickness"] = vals.get("tendonThickness", 0.8)
    parameters["h1Length"] = vals.get("hingeLength", 2)
    parameters["h1Thickness"] = vals.get("hingeThickness", 0.8)
    parameters["h2Length"] = vals.get("hingeLength", 2)
    parameters["h2Thickness"] = vals.get("hingeThickness", 0.8)
    parameters["h3Length"] = vals.get("hingeLength", 2)
    parameters["h3Thickness"] = vals.get("hingeThickness", 0.8)
    
    # Additional parameters
    b = parameters["beamB"]
    c = parameters["beamC"]
    a = parameters["beamA"]
    h1 = parameters["h1Length"]
    h2 = parameters["h2Length"]
    h3 = parameters["h3Length"]
    
    # Use theta from parameters; convert to radians
    theta = parameters["theta"]
    theta_radians = np.deg2rad(theta)
    
    # Compute d using the cosine law:
    d = np.sqrt(b**2 + c**2 - 2 * b * c * np.cos(theta_radians))
    
    # Compute beta (angle between side b and d) using cosine law:
    cos_beta = (d**2 + b**2 - c**2) / (2 * b * d)
    beta_radians = np.arccos(cos_beta)
    
    # Compute tendon length using the cosine law:
    _1 = b + h1 + a + h2
    _2 = d + h3
    L = np.sqrt(_1**2 + _2**2 - 2 * _1 * _2 * np.cos(beta_radians))
    
    # Save computed values into parameters dictionary
    parameters["beta"] = np.rad2deg(beta_radians)
    parameters["beamD"] = d
    parameters["tendonL"] = L
    
    return parameters

def scale_parameters_to_model_size(scale_factor = 100.0):
    parameters= vals_to_parameters(vals)
    scaled_parameters = {}
    # Scale the parameters
    for key in parameters:
        scaled_parameters[key] = parameters[key] * 1.0 / scale_factor
    
    # Keep angle the same
    scaled_parameters["theta"] = parameters["theta"]
    scaled_parameters["beta"] = parameters["beta"]

    # Add material properties
    scaled_parameters["jointStiffness"] = vals.get("jointStiffness", 100)
    scaled_parameters["jointDamping"] = vals.get("jointDamping", 0.1)
    scaled_parameters["tendonExtendStiffness"] = vals.get("tendonExtendStiffness", 100)
    scaled_parameters["tendonBendStiffness"] = vals.get("tendonBendStiffness", 10)
    scaled_parameters["tendonExtendDamping"] = vals.get("tendonExtendDamping", 10)
    scaled_parameters["tendonBendDamping"] = vals.get("tendonBendDamping", 10)

    return scaled_parameters


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
    print(f"Updated {name} fromto: {new_fromto} length: {new_length}")
    return new_end


def modify_model(xml_file, parameters):
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
            # tendon.set("springlength", new_range)
        if tendon.get("name") == "bendingTendon": 
            new_solreflimit = f"-{parameters['tendonBendStiffness']} -{parameters['tendonBendDamping']}"
            tendon.set("solreflimit", new_solreflimit)
            new_range = f"{parameters['tendonL']} {parameters['tendonL']*2}"
            tendon.set("range", new_range)
            # tendon.set("springlength", new_range)

    # Save the modified XML file
    modified_xml_file = "modified_model.xml"
    tree.write(modified_xml_file)
    return modified_xml_file



xml_file = "2DModel.xml"  # Path to your MuJoCo XML file
parameters = scale_parameters_to_model_size()
# print("Parameters used after scaling:")
# for key, value in parameters.items():
#     print(f"{key}: {value}")
new_xml_file = modify_model(xml_file, parameters)
print(f"Modified XML saved as: {new_xml_file}")

# test: calculate the angle between beamB and beamC
# beamB = parameters["beamB"]
# beamC = parameters["beamC"]
# beamD = parameters["beamD"]
# print(f"BeamB: {beamB}, BeamC: {beamC}, BeamD: {beamD}")
# angle = math.acos((beamB**2 + beamC**2 - beamD**2) / (2 * beamB * beamC))
# angle = np.rad2deg(angle)
# print(f"Angle between beamB and beamC: {angle:.2f} degrees")



# parameters = {
#     "beamA": 25,
#     "beamB": 25,
#     "beamC": 25,
#     "beamD": 12.94,  # e.g. computed from your geometry
#     "theta": 30,
#     "tendonThickness": 0.8,
#     "hingeLength": 2,
#     "hingeThickness": 0.8,
#     "tendonL": 50,   # could be used elsewhere
#     "h1Length": 2,
#     "h1Thickness": 0.8,
#     "h2Length": 2,
#     "h2Thickness": 0.8,
#     "h3Length": 2,
#     "h3Thickness": 0.8,
# }
