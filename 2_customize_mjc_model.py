import xml.etree.ElementTree as ET
import numpy as np
import math

# Define the values for the changing parameters
vals = {
    "beamB": 25, 
    "beamA": 30, 
    "beamC": 30,
    "theta": 30,
    "tendonThickness": 1,
    "tendonWidth": 1.6,
    "hingeLength": 2,
    "hingeThickness": 1,
    "hingeWidth": 2.4,
    "jointStiffness" : 2e5,
    "jointStiffnessDampingRatio" : 2e2,
    "tendonExtendStiffness" : 1e8,
    "tendonExtendStiffnessDampingRatio" : 30,
    "tendonBendStiffness" : 16e10,
    "tendonBendStiffnessDampingRatio" : 2e5,
}

# some manual tuning tips:
# damping ≈ 0.1~1.0 * sqrt(stiffness*mass)?
# jointStiffness -> 50, no bistable
# tendonExtendStiffness -> 1000, no bistable
# "jointStiffness" : 20,
# "jointDamping" : 10,
# "tendonExtendStiffness" : 10000,
# "tendonExtendDamping" : 100,
# "tendonBendStiffness" : 100,
# "tendonBendDamping" : 100,


# TODO check more like: Euler–Bernoulli beam theory; stress, elongation
def scaleBendingStiffness(stiffness, width, thickness, length, power = 3):
    # k = E * I / L^3
    # I = b * h^3 / 12
    # k = E * b * h^3 / 12 / L^3 = E * b * h / L^3
    # b: width, h: thickness, L: length
    return stiffness * width * (thickness / length)**power / 12

def scaleExtendStiffness(stiffness, width, thickness, length, power = 1):
    # k = E * A / L
    # A = b * h
    # k = E * b * h / L
    # b: width, h: thickness, L: length
    return stiffness * width * (thickness / length)**power

def vals_to_parameters(vals):
    parameters = {}
    
    # Base parameters from vals (with defaults if not provided)
    parameters["beamB"] = vals.get("beamB", 25)
    parameters["beamA"] = vals.get("beamA", 25)
    parameters["beamC"] = vals.get("beamC", 25)
    parameters["theta"] = vals.get("theta", 30)
    parameters["tendonThickness"] = vals.get("tendonThickness", 0.8)
    parameters["tendonWidth"] = vals.get("tendonWidth", 1.6)
    parameters["hingeLength"] = vals.get("hingeLength", 2)
    parameters["hingeThickness"] = vals.get("hingeThickness", 0.8)
    parameters["hingeWidth"] = vals.get("hingeWidth", 2)
    # In this version, let's assume all hinges have the same dimensions
    # parameters["h1Length"] = vals.get("hingeLength", 2)
    # parameters["h1Thickness"] = vals.get("hingeThickness", 0.8)
    # parameters["h2Length"] = vals.get("hingeLength", 2)
    # parameters["h2Thickness"] = vals.get("hingeThickness", 0.8)
    # parameters["h3Length"] = vals.get("hingeLength", 2)
    # parameters["h3Thickness"] = vals.get("hingeThickness", 0.8)
    
    # Compute the length of beam D, the angle beta, and the tendon length
    b = parameters["beamB"]
    c = parameters["beamC"]
    a = parameters["beamA"]
    # In this version, we didn't model the length of the hinges
    # h1 = parameters["h1Length"]
    # h2 = parameters["h2Length"]
    # h3 = parameters["h3Length"]
    
    theta = parameters["theta"]
    theta_radians = np.deg2rad(theta)
    
    d = np.sqrt(b**2 + c**2 - 2 * b * c * np.cos(theta_radians))
    cos_beta = (d**2 + b**2 - c**2) / (2 * b * d)
    beta_radians = np.arccos(cos_beta)
    
    _1 = b + a
    _2 = d
    L = np.sqrt(_1**2 + _2**2 - 2 * _1 * _2 * np.cos(beta_radians))
    
    parameters["beta"] = np.rad2deg(beta_radians)
    parameters["beamD"] = d
    parameters["tendonL"] = L
    
    return parameters

def scale_parameters_to_model_size(vals, scale_factor = 100.0):
    parameters= vals_to_parameters(vals)
    scaled_parameters = {}
    # Scale the dimension parameters by the scale factor
    for key in parameters:
        scaled_parameters[key] = parameters[key] * 1.0 / scale_factor
    
    # Keep angle the same
    scaled_parameters["theta"] = parameters["theta"]
    scaled_parameters["beta"] = parameters["beta"]

    # Add material properties
    scaled_parameters["jointStiffness"] = vals.get("jointStiffness", 100)
    scaled_parameters["tendonExtendStiffness"] = vals.get("tendonExtendStiffness", 100)
    scaled_parameters["tendonBendStiffness"] = vals.get("tendonBendStiffness", 10)

    # Scale the stiffness with the beam dimensions
    jointStiffness = scaleBendingStiffness(scaled_parameters["jointStiffness"], scaled_parameters["hingeWidth"], scaled_parameters["hingeThickness"], scaled_parameters["hingeLength"])
    scaled_parameters["jointStiffness"] = jointStiffness
    tendonBendStiffness = scaleBendingStiffness(scaled_parameters["tendonBendStiffness"], scaled_parameters["tendonWidth"], scaled_parameters["tendonThickness"], scaled_parameters["tendonL"])
    scaled_parameters["tendonBendStiffness"] = tendonBendStiffness
    tendonExtendStiffness = scaleExtendStiffness(scaled_parameters["tendonExtendStiffness"], scaled_parameters["tendonWidth"], scaled_parameters["tendonThickness"], scaled_parameters["tendonL"])
    scaled_parameters["tendonExtendStiffness"] = tendonExtendStiffness
    # print(f"Joint stiffness: {jointStiffness}, tendonExtendStiffness: {tendonExtendStiffness}, tendonBendStiffness: {tendonBendStiffness}")

    # Set the damping values based on the stiffness values and damping ratios
    scaled_parameters["jointDamping"] = scaled_parameters["jointStiffness"] / vals.get("jointStiffnessDampingRatio", 2)
    scaled_parameters["tendonExtendDamping"] = scaled_parameters["tendonExtendStiffness"] / vals.get("tendonExtendStiffnessDampingRatio", 10)
    scaled_parameters["tendonBendDamping"] = scaled_parameters["tendonBendStiffness"] / vals.get("tendonBendStiffnessDampingRatio", 10)
    # print(f"Joint damping: {scaled_parameters['jointDamping']}, tendonExtendDamping: {scaled_parameters['tendonExtendDamping']}, tendonBendDamping: {scaled_parameters['tendonBendDamping']}")

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
    # print(f"Updated {name} fromto: {new_fromto} length: {new_length}")
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
            # tendon.set("springlength", new_range)
        if tendon.get("name") == "bendingTendon": 
            new_solreflimit = f"-{parameters['tendonBendStiffness']} -{parameters['tendonBendDamping']}"
            tendon.set("solreflimit", new_solreflimit)
            new_range = f"{parameters['tendonL']} {parameters['tendonL']*2}"
            tendon.set("range", new_range)
            # tendon.set("springlength", new_range)

    tree.write(saved_file)


def generate_model(vals, xml_file, saved_file, scale_factor = 100.0):
    # Generate the model with default parameters
    parameters = scale_parameters_to_model_size(vals, scale_factor)
    modify_model(xml_file, saved_file, parameters)
    # print(f"Generated model saved as: {saved_file}")


generate_model(vals, "2DModel.xml", "modified_model.xml")




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
