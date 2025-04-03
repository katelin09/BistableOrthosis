import bpy
import bmesh
import mathutils
import math
import random
import time
import os


################################################################
# helper functions BEGIN
################################################################


def purge_orphans():
    """
    Remove all orphan data blocks

    see this from more info:
    https://youtu.be/3rNqVPtbhzc?t=149
    """
    if bpy.app.version >= (3, 0, 0):
        # run this only for Blender versions 3.0 and higher
        bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
    else:
        # run this only for Blender versions lower than 3.0
        # call purge_orphans() recursively until there are no more orphan data blocks to purge
        result = bpy.ops.outliner.orphans_purge()
        if result.pop() != "CANCELLED":
            purge_orphans()


def clean_scene():
    """
    Removing all of the objects, collection, materials, particles,
    textures, images, curves, meshes, actions, nodes, and worlds from the scene

    Checkout this video explanation with example

    "How to clean the scene with Python in Blender (with examples)"
    https://youtu.be/3rNqVPtbhzc
    """
    # make sure the active object is not in Edit Mode
    if bpy.context.active_object and bpy.context.active_object.mode == "EDIT":
        bpy.ops.object.editmode_toggle()

    # make sure non of the objects are hidden from the viewport, selection, or disabled
    for obj in bpy.data.objects:
        obj.hide_set(False)
        obj.hide_select = False
        obj.hide_viewport = False

    # select all the object and delete them (just like pressing A + X + D in the viewport)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    # find all the collections and remove them
    collection_names = [col.name for col in bpy.data.collections]
    for name in collection_names:
        bpy.data.collections.remove(bpy.data.collections[name])

    # in the case when you modify the world shader
    # delete and recreate the world object
    world_names = [world.name for world in bpy.data.worlds]
    for name in world_names:
        bpy.data.worlds.remove(bpy.data.worlds[name])
    # create a new world data block
    bpy.ops.world.new()
    bpy.context.scene.world = bpy.data.worlds["World"]

    purge_orphans()


def active_object():
    """
    returns the currently active object
    """
    return bpy.context.active_object


def time_seed():
    """
    Sets the random seed based on the time
    and copies the seed into the clipboard
    """
    seed = time.time()
    print(f"seed: {seed}")
    random.seed(seed)

    # add the seed value to your clipboard
    bpy.context.window_manager.clipboard = str(seed)

    return seed


def set_scene_props(fps, frame_count):
    """
    Set scene properties
    """
    scene = bpy.context.scene
    scene.frame_end = frame_count

    # set the world background to black
    world = bpy.data.worlds["World"]
    if "Background" in world.node_tree.nodes:
        world.node_tree.nodes["Background"].inputs[0].default_value = (0, 0, 0, 1)

    scene.render.fps = fps

    scene.frame_current = 1
    scene.frame_start = 1


def scene_setup():
    fps = 30
    loop_seconds = 12
    frame_count = fps * loop_seconds

    seed = 0
    if seed:
        random.seed(seed)
    else:
        time_seed()

    clean_scene()

    set_scene_props(fps, frame_count)


################################################################
# helper functions END
################################################################

params = {
    "DIP_radius": 8.0,
    "PIP_radius": 9.0,
    "MCP_radius": 8.5,
    "TIP_radius": 5.0,

    "D2P_length": 20.0,
    "P2M_length": 18.0,
    "TIP_length": 18.0,

    "A_length": 25.0,
    "C_length": 25.0,
    "theta_angle": 30.0,
    "initial_angle": 2.0,

    "D_length": 0.0,
    "beta_angle": 0.0,
    "B_length": 15.0,
    "L_length": 0.0,
    "gamma_angle": 0.0,

    "H1_length": 2.0,
    "H2_length": 2.0,
    "H3_length": 2.0,

    "H1_thickness": 0.6,
    "H2_thickness": 0.6,
    "H3_thickness": 0.6,

    "L_thickness": 1.2,
    "L_width": 1.6,
    "L_offset": 0.8,

    "Rigid_thickness": 2.0,
    "Attach_thickness": 1.0,

    "resolution": 64
}


def updateParams():
    params["H2_length"] = params["H1_length"]
    params["H3_length"] = params["H1_length"]
    params["H2_thickness"] = params["H1_thickness"]
    params["H3_thickness"] = params["H1_thickness"]

    params["B_length"] = params["P2M_length"]

    params["D_length"] = math.sqrt(params["B_length"]**2 + params["C_length"]**2 - 
                                2*params["B_length"]*params["C_length"]*
                                math.cos(math.radians(params["theta_angle"])))
    params["beta_angle"] = math.degrees(math.acos(
                                (params["B_length"]**2 + params["D_length"]**2 - params["C_length"]**2) / 
                                (2 * params["B_length"] * params["D_length"])))
    params["L_length"] = math.sqrt((params["B_length"] + params["A_length"] + params["H1_length"] + params["H2_length"])**2 + 
                                 (params["D_length"] + params["H3_length"])**2 - 
                                 2*(params["B_length"] + params["A_length"] + params["H1_length"] + params["H2_length"])*
                                 (params["D_length"] + params["H3_length"])*math.cos(math.radians(params["beta_angle"])))
    # print("parameters updated:", params)


def create_hemisphere(radius, segments, position, bmesh_obj, orientation=(0,0,0)):
    verts = []
    # Create center vertex
    center = bmesh_obj.verts.new((position[0], position[1], position[2]))
    verts.append(center)
    
    # Create rotation matrix from orientation angles (in radians)
    rot_matrix = (mathutils.Matrix.Rotation(orientation[0], 4, 'X') @ 
                 mathutils.Matrix.Rotation(orientation[1], 4, 'Y') @ 
                 mathutils.Matrix.Rotation(orientation[2], 4, 'Z'))
    
    # Create concentric rings of vertices
    for phi in range(segments//2):
        phi_angle = math.pi/2 * (phi + 1)/(segments/2)
        y = radius * math.cos(phi_angle)
        current_radius = radius * math.sin(phi_angle)
        
        for theta in range(segments):
            theta_angle = 2 * math.pi * theta/segments
            x = current_radius * math.cos(theta_angle)
            z = current_radius * math.sin(theta_angle)
            
            # Apply rotation and translation
            vert_pos = rot_matrix @ mathutils.Vector((x, y, z))
            vert_pos += mathutils.Vector(position)
            
            vert = bmesh_obj.verts.new(vert_pos)
            verts.append(vert)
    
    # Create faces
    for i in range(1, segments + 1):
        if i == segments:
            bmesh_obj.faces.new((verts[0], verts[i], verts[1]))
        else:
            bmesh_obj.faces.new((verts[0], verts[i], verts[i + 1]))
    
    for ring in range(1, segments//2):
        for i in range(segments):
            current = ring * segments + i + 1
            next_in_ring = ring * segments + ((i + 1) % segments) + 1
            prev_ring = (ring - 1) * segments + i + 1
            prev_ring_next = (ring - 1) * segments + ((i + 1) % segments) + 1
            
            bmesh_obj.faces.new((verts[prev_ring], verts[current], verts[next_in_ring], verts[prev_ring_next]))
    
    return {'verts': verts}


def create_finger():
    # Create a new mesh and bmesh
    mesh = bpy.data.meshes.new("finger_mesh")
    bmFinger = bmesh.new()

    # Create rotation matrix (90 degrees around X axis)
    rot_matrix = mathutils.Matrix.Rotation(math.radians(90), 4, 'X')

    # Create circles in bmesh
    TIPcircle = bmesh.ops.create_circle(bmFinger, cap_ends=False, matrix = rot_matrix, segments=32, radius=params["TIP_radius"])
    for v in TIPcircle['verts']:
        v.co.y = -(params["TIP_length"] + params["D2P_length"])
    DIPcircle = bmesh.ops.create_circle(bmFinger, cap_ends=False, matrix = rot_matrix, segments=32, radius=params["DIP_radius"])
    for v in DIPcircle['verts']:
        v.co.y = -params["D2P_length"]
    PIPcircle = bmesh.ops.create_circle(bmFinger, cap_ends=False, matrix = rot_matrix, segments=32, radius=params["PIP_radius"])
    MCPcircle = bmesh.ops.create_circle(bmFinger, cap_ends=False, matrix = rot_matrix, segments=32, radius=params["MCP_radius"])
    for v in MCPcircle['verts']:
        v.co.y = params["P2M_length"]

    # Bridge between circles
    bmesh.ops.bridge_loops(bmFinger, edges=[e for e in bmFinger.edges])

    # Create hemispheres at tip
    tip_pos = (0, -(params["TIP_length"] + params["D2P_length"]), 0)
    create_hemisphere(params["TIP_radius"], 32, tip_pos, bmFinger, orientation=(math.pi, 0, 0))
    
    # Fill any remaining holes
    bmesh.ops.holes_fill(bmFinger, edges=[e for e in bmFinger.edges if len(e.link_faces) < 2])
    # Add smoothing
    # for face in bmFinger.faces: face.smooth = True

    # Create mesh from bmesh
    bmFinger.to_mesh(mesh)
    bmFinger.free()

    # Create object from mesh
    obj = bpy.data.objects.new("finger", mesh)
    bpy.context.collection.objects.link(obj)

    return mesh, bmFinger, obj


def create_beamA(radius_at_negH1, radius_at_lengthA):
    meshA = bpy.data.meshes.new("beamA_mesh")
    beamA_mesh = bmesh.new()

    rot_matrix = mathutils.Matrix.Rotation(math.radians(90), 4, 'X')

    bmesh.ops.create_cone(beamA_mesh, 
                         cap_ends=False, 
                         cap_tris=False, 
                         segments=params["resolution"],
                         radius1=radius_at_negH1,  
                         radius2=radius_at_lengthA,  
                         depth=params["A_length"] - params["H1_length"]/2,  
                         matrix=mathutils.Matrix.Translation((0, -(params["A_length"] + params["H1_length"]/2)/2, 0)) @ rot_matrix)
    
    # Cut the cones, only keep z>=0
    geom = bmesh.ops.bisect_plane(beamA_mesh,
                                 geom=beamA_mesh.verts[:] + beamA_mesh.edges[:] + beamA_mesh.faces[:],
                                 plane_co=(0, 0, 0),
                                 plane_no=(0, 0, 1))
    
    verts_below = [v for v in beamA_mesh.verts if v.co.z < 0]
    bmesh.ops.delete(beamA_mesh, geom=verts_below, context='VERTS')

    # Create thickness
    all_faces = [f for f in beamA_mesh.faces]
    result = bmesh.ops.extrude_face_region(beamA_mesh,
                                         geom=all_faces)
    
    extruded_verts = [v for v in result['geom'] if isinstance(v, bmesh.types.BMVert)]
    
    for v in extruded_verts:
        v.co += v.normal * params["Rigid_thickness"]
    
    

    beamA_mesh.to_mesh(meshA)
    beamA_mesh.free()
    objA = bpy.data.objects.new("beamA", meshA)
    bpy.context.collection.objects.link(objA)

    return objA


def create_beamB(radius_at_posH1, radius_at_lengthB):
    # Create the mesh for each beam
    meshB = bpy.data.meshes.new("beamB_mesh")
    beamB_mesh = bmesh.new()

    rot_matrix = mathutils.Matrix.Rotation(math.radians(90), 4, 'X')

    # Create circles at y = H1_length/2
    circle1_matrix = mathutils.Matrix.Translation((0, params["H1_length"]/2, 0)) @ rot_matrix
    circle2_matrix = circle1_matrix
    # Create circle and rectangle at y = B_length, rotated by beta angle
    rot_beta = mathutils.Matrix.Rotation(-math.radians(params["beta_angle"] - 90), 4, 'X')
    final_matrix = mathutils.Matrix.Translation((0, params["B_length"], 0)) @ rot_beta @ rot_matrix

    circle1 = bmesh.ops.create_circle(beamB_mesh, cap_ends=False, segments=params["resolution"], radius=radius_at_posH1, matrix=circle1_matrix)
    circle2 = bmesh.ops.create_circle(beamB_mesh, cap_ends=False, segments=params["resolution"], radius=radius_at_posH1 + params["Rigid_thickness"], matrix=circle2_matrix)

    # Create circle and rectangle at y = B_length
    circle3 = bmesh.ops.create_circle(beamB_mesh, cap_ends=False, segments=params["resolution"], radius=radius_at_lengthB, matrix=final_matrix)
    rect_width = 2 * (radius_at_lengthB + params["Rigid_thickness"])
    rect_height = params["D_length"]
    rect_verts = []
    rect_verts.append(beamB_mesh.verts.new(final_matrix @ mathutils.Vector((-rect_width/2, -rect_height, 0))))
    rect_verts.append(beamB_mesh.verts.new(final_matrix @ mathutils.Vector((rect_width/2, -rect_height, 0))))
    rect_verts.append(beamB_mesh.verts.new(final_matrix @ mathutils.Vector((rect_width/2, rect_height, 0))))
    rect_verts.append(beamB_mesh.verts.new(final_matrix @ mathutils.Vector((-rect_width/2, rect_height, 0))))
    # Add edges to the rectangle
    beamB_mesh.edges.new((rect_verts[0], rect_verts[1]))
    beamB_mesh.edges.new((rect_verts[1], rect_verts[2]))
    beamB_mesh.edges.new((rect_verts[2], rect_verts[3]))
    beamB_mesh.edges.new((rect_verts[3], rect_verts[0]))
    
    # Get edges from each circle operation
    circle1_edges = [e for e in beamB_mesh.edges if any(v in circle1['verts'] for v in e.verts)]
    circle2_edges = [e for e in beamB_mesh.edges if any(v in circle2['verts'] for v in e.verts)]
    circle3_edges = [e for e in beamB_mesh.edges if any(v in circle3['verts'] for v in e.verts)]
    rect_edges = [e for e in beamB_mesh.edges if all(v in rect_verts for v in e.verts)]

    # Bridge
    bmesh.ops.bridge_loops(beamB_mesh, edges=circle1_edges + circle2_edges)
    bmesh.ops.bridge_loops(beamB_mesh, edges=circle1_edges + circle3_edges)
    bmesh.ops.bridge_loops(beamB_mesh, edges=circle2_edges + rect_edges)
    # bmesh.ops.bridge_loops(beamB_mesh, edges=circle3_edges + rect_edges)
    
    # Create duplicates of circle3 and rectangle, moved by H3_thickness
    verts_to_duplicate = []
    for e in circle3_edges:
        verts_to_duplicate.extend([v for v in e.verts])
    for e in rect_edges:
        verts_to_duplicate.extend([v for v in e.verts])
    verts_to_duplicate = list(set(verts_to_duplicate))  # Remove duplicates

    dup_verts = {}
    for v in verts_to_duplicate:
        new_co = v.co + mathutils.Vector((0, params["H3_thickness"], 0))
        new_v = beamB_mesh.verts.new(new_co)
        dup_verts[v] = new_v

    # Create edges for duplicated geometry
    dup_circle3_edges = []
    for e in circle3_edges:
        new_e = beamB_mesh.edges.new((dup_verts[e.verts[0]], dup_verts[e.verts[1]]))
        dup_circle3_edges.append(new_e)

    dup_rect_edges = []
    for e in rect_edges:
        new_e = beamB_mesh.edges.new((dup_verts[e.verts[0]], dup_verts[e.verts[1]]))
        dup_rect_edges.append(new_e)

    # Bridge between original and duplicated edges
    for e1, e2 in zip(circle3_edges + rect_edges, dup_circle3_edges + dup_rect_edges):
        bmesh.ops.bridge_loops(beamB_mesh, edges=[e1, e2])

    # Cut the cones, only keep z>=0
    geom = bmesh.ops.bisect_plane(beamB_mesh,
                                 geom=beamB_mesh.verts[:] + beamB_mesh.edges[:] + beamB_mesh.faces[:],
                                 plane_co=(0, 0, 0),
                                 plane_no=(0, 0, 1))
    
    verts_below = [v for v in beamB_mesh.verts if v.co.z < 0]
    bmesh.ops.delete(beamB_mesh, geom=verts_below, context='VERTS')

    # Fill any remaining holes
    bmesh.ops.holes_fill(beamB_mesh, edges=[e for e in beamB_mesh.edges if len(e.link_faces) < 2])

    # # Apply initial angle rotation to all vertices
    # rot_initial = mathutils.Matrix.Rotation(math.radians(params["initial_angle"]), 4, 'X')
    # for v in beamB_mesh.verts:
    #     v.co = rot_initial @ v.co

    beamB_mesh.to_mesh(meshB)
    beamB_mesh.free()
    objB = bpy.data.objects.new("beamB", meshB)
    bpy.context.collection.objects.link(objB)
    
    return objB


def create_hinge1():
    mesh = bpy.data.meshes.new("hinge1_mesh")
    bm = bmesh.new()
    
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        v.co.x *= params["Rigid_thickness"]
        v.co.y *= params["H1_length"]
        v.co.z *= params["H1_thickness"]
        v.co.x += params["PIP_radius"] + params["Rigid_thickness"]/2
        v.co.z += params["H1_thickness"]/2
        
    bm.to_mesh(mesh)
    bm.free()
    
    obj = bpy.data.objects.new("hinge1", mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def create_hinge2():
    mesh = bpy.data.meshes.new("hinge2_mesh")
    bm = bmesh.new()
    
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        v.co.x *= params["Rigid_thickness"]
        v.co.y *= params["H2_length"]
        v.co.z *= params["H2_thickness"]
        v.co.x += params["DIP_radius"] + params["Rigid_thickness"]/2
        v.co.y += -params["A_length"] - params["H2_length"]/2
        v.co.z += params["H2_thickness"]/2
        
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new("hinge2", mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def create_hinge3():
    mesh = bpy.data.meshes.new("hinge3_mesh")
    bm = bmesh.new()
    
    bmesh.ops.create_cube(bm, size=1.0) 
        
    for v in bm.verts:
        v.co.x *= params["Rigid_thickness"]
        v.co.y *= params["H3_thickness"]
        v.co.z *= params["H3_length"]
    
    for v in bm.verts:
        # Create shear matrix in the YZ plane
        shear_angle = math.radians(params["beta_angle"] - 90)
        shear_matrix = mathutils.Matrix.Identity(4)
        shear_matrix[1][2] = math.tan(shear_angle)  # Shear Y based on Z position
        v.co = shear_matrix @ v.co

    for v in bm.verts:
        v.co.x += params["MCP_radius"] + params["Rigid_thickness"]/2
        v.co.y += params["C_length"] * math.cos(math.radians(params["theta_angle"])) + params["H3_thickness"]
        v.co.z += params["C_length"] * math.sin(math.radians(params["theta_angle"])) + params["H3_length"]/2
        
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new("hinge3", mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def create_blockA():
    mesh = bpy.data.meshes.new("blockA_mesh")
    bm = bmesh.new()
    
    x_length = max(params["DIP_radius"], params["PIP_radius"], params["MCP_radius"]) + params["Rigid_thickness"] + params["L_offset"] + params["L_width"] - params["DIP_radius"]
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        v.co.x *= x_length
        v.co.y *= params["Rigid_thickness"]
        v.co.z *= params["Rigid_thickness"] * 2
        v.co.x += params["DIP_radius"] + x_length/2
        v.co.y += -params["A_length"] - params["H2_length"] - params["Rigid_thickness"] / 2
        v.co.z += params["Rigid_thickness"]
        
    bm.to_mesh(mesh)
    bm.free()
    
    obj = bpy.data.objects.new("blockA", mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def create_blockB():
    mesh = bpy.data.meshes.new("blockB_mesh")
    bm = bmesh.new()
    
    x_length = max(params["DIP_radius"], params["PIP_radius"], params["MCP_radius"]) + params["Rigid_thickness"] + params["L_offset"] + params["L_width"] - params["MCP_radius"]
    y = (params["D_length"] + params["H3_length"]) * math.cos(math.pi - math.radians(params["beta_angle"])) + params["B_length"] + params["H3_thickness"]
    z = (params["D_length"] + params["H3_length"]) * math.sin(math.pi - math.radians(params["beta_angle"]))
    bmesh.ops.create_cube(bm, size=1.0) 
    for v in bm.verts:
        v.co.x *= x_length
        v.co.y *= params["Rigid_thickness"] * 2
        v.co.z *= params["Rigid_thickness"]
        v.co.x += params["MCP_radius"] + x_length/2
        v.co.y += y + params["Rigid_thickness"] - params["H3_thickness"]
        v.co.z += z + params["Rigid_thickness"]/2
        
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new("blockB", mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def create_beamL():
    mesh = bpy.data.meshes.new("beamL_mesh")
    bm = bmesh.new()
    
    bmesh.ops.create_cube(bm, size=1.0)
    x = max(params["DIP_radius"], params["PIP_radius"], params["MCP_radius"]) + params["Rigid_thickness"] + params["L_offset"]
    for v in bm.verts:
        v.co.x *= params["L_width"]
        v.co.y *= params["L_length"]
        v.co.z *= params["L_thickness"]
        v.co.x += x + params["L_width"]/2
        v.co.z += params["L_thickness"]/2
    
    cos_gamma = ((params["A_length"] + params["B_length"] + params["H1_length"] + params["H2_length"])**2 + params["L_length"]**2 - (params["D_length"] + params["H3_length"])**2) / (
                2 * (params["A_length"] + params["B_length"] + params["H1_length"] + params["H2_length"]) * params["L_length"])
    cos_gamma = max(min(cos_gamma, 1), -1)  # Clamp value between -1 and 1
    gamma_radians = math.acos(cos_gamma)

    # Create rotation matrix around X axis
    rot_matrix = mathutils.Matrix.Rotation(gamma_radians, 4, 'X')

    # Apply rotation to all vertices
    for v in bm.verts:
        v.co = rot_matrix @ v.co
        v.co.y += - params["A_length"] - params["H2_length"] - params["H1_length"]/2 + params["L_length"]/2
        v.co.z += params["L_length"] * math.sin(gamma_radians)/2
    
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new("beamL", mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def create_brace():
    # Calculate radius at specific y positions
    def interpolate_radius(y_pos, y1, y2, r1, r2):
        """Linear interpolation of radius based on y position"""
        if y1 == y2:
            return r1
        t = (y_pos - y1) / (y2 - y1)
        return r1 + t * (r2 - r1)

    points = [
        (-params["D2P_length"], params["DIP_radius"]),
        (0, params["PIP_radius"]),
        (params["P2M_length"], params["MCP_radius"])
    ]

    radius_at_lengthA = interpolate_radius(-params["A_length"], points[0][0], points[1][0], points[0][1], points[1][1])
    radius_at_negH1 = interpolate_radius(-params["H1_length"]/2, points[0][0], points[1][0], points[0][1], points[1][1])
    radius_at_posH1 = interpolate_radius(params["H1_length"]/2, points[1][0], points[2][0], points[1][1], points[2][1])
    radius_at_lengthB = interpolate_radius(params["B_length"], points[1][0], points[2][0], points[1][1], points[2][1])


    objA = create_beamA(radius_at_negH1, radius_at_lengthA)
    objB = create_beamB(radius_at_posH1, radius_at_lengthB)
    objH1 = create_hinge1()
    objH2 = create_hinge2()
    objH3 = create_hinge3()
    objBlockA = create_blockA()
    objBlockB = create_blockB()
    objL = create_beamL()

    mirror_objects = objH1, objH2, objH3, objBlockA, objBlockB, objL
    for obj in mirror_objects:
        obj_copy = obj.copy()
        obj_copy.data = obj.data.copy()
        bpy.context.collection.objects.link(obj_copy)
        for v in obj_copy.data.vertices:
            v.co.x = -v.co.x
        obj_copy.name = obj.name + "_2"

    # Create cutter cylinder
    cut_mesh = bpy.data.meshes.new("cut_mesh")
    cut_bm = bmesh.new()

    cylinder_mat = mathutils.Matrix.Translation((0, 0, 0))
    bmesh.ops.create_cone(cut_bm, 
                         cap_ends=False, 
                         cap_tris=False, 
                         segments=params["resolution"],
                         radius1=params["PIP_radius"],  
                         radius2=params["PIP_radius"],  
                         depth=100,  
                         matrix=cylinder_mat)
    

    # Convert to mesh and create object
    cut_bm.to_mesh(cut_mesh)
    cut_bm.free()
    cut_obj = bpy.data.objects.new("cut_cylinder", cut_mesh)
    bpy.context.collection.objects.link(cut_obj)

    # Boolean difference
    bool_mod_A = objA.modifiers.new(name="bool_cut", type='BOOLEAN')
    bool_mod_A.object = cut_obj
    bool_mod_A.operation = 'DIFFERENCE'
    
    # TODO after boolean operation, the faces are broken, need to fix them
    bool_mod_B = objB.modifiers.new(name="bool_cut", type='BOOLEAN')
    bool_mod_B.object = cut_obj
    bool_mod_B.operation = 'DIFFERENCE'
    
    cut_obj.hide_viewport = True

    # Select all objects to join
    objects_list = [objA, objB, objH1, objH2, objH3, objBlockA, objBlockB, objL]
    mirror_objects = [obj for obj in bpy.data.objects if obj.name.endswith("_2")]
    objects_list.extend(mirror_objects)

    return objects_list


# TODO modify the following function to create the animation
def create_animation():
    # Create keyframe animation
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = 120  # 4 seconds at 30fps
    
    # Set keyframes for beamA rotation
    beamA = bpy.data.objects["beamA"]
    beamA.rotation_euler = (0, 0, 0)
    beamA.keyframe_insert(data_path="rotation_euler", frame=1)
    
    beamA.rotation_euler = (math.radians(params["initial_angle"]), 0, 0)
    beamA.keyframe_insert(data_path="rotation_euler", frame=60)
    
    beamA.rotation_euler = (0, 0, 0)
    beamA.keyframe_insert(data_path="rotation_euler", frame=120)
    
    
    # Make the animation loop
    for obj in [beamA]:
        for fcurve in obj.animation_data.action.fcurves:
            for kf in fcurve.keyframe_points:
                kf.interpolation = 'LINEAR'


def main():
    updateParams()
    scene_setup()
    create_finger()
    objects_list = create_brace()

    # Deselect all objects first
    bpy.ops.object.select_all(action='DESELECT')

    # Select all objects and make objA active
    for obj in objects_list:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects_list[0]

    # Join the objects
    bpy.ops.object.join()

    # Rename the resulting object
    objects_list[0].name = "brace"

    # Export the brace to STL format using absolute path
    export_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "brace.stl")
    bpy.ops.wm.stl_export(filepath=export_path)


if __name__ == "__main__":
    main()