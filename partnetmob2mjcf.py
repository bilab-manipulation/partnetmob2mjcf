import argparse
from xml.etree.ElementTree import Element, SubElement, ElementTree
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation as R
import glob
import os
from collections import defaultdict
from typing import Tuple


def indent(elem, level=0): # https://goo.gl/J8VoDK
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def get_obj_xml_files(meshdir):
    '''
    make sure you already run obj2mjcf
    $ obj2mjcf --obj-dir textured_objs --save-mjcf
    also solidify the mesh by blender python api, so that you add the thickness to the object meshes
    otherwise, it will cause an error because of the small volume
    '''
    obj_mesh_xml_files = glob.glob(os.path.join(meshdir, '**/*.xml'), recursive=True)
    return obj_mesh_xml_files

def create_asset(obj_mesh_files):

    mesh_name2mat_name = {}
    meta_asset = Element('asset')
    for obj_mesh_file in obj_mesh_files:
        children = [] # list of child for later change texture asset of the material
        prefix = obj_mesh_file.split('/')[-1].split('.')[0]
        folder_name = obj_mesh_file.split('/')[-2]
        tree = ET.parse(obj_mesh_file)
        root = tree.getroot()
        geoms = root.find('worldbody').find('body').findall('geom')
        material2mesh = {geom.get('material'): geom.get('mesh')  for geom in geoms if geom.get('class') == "visual"}
        asset = root.find('asset')
        texture2material = {a.get('texture'): a.get('name') for a in asset}
        # {texture_name 원래: 새로운 texture_name}
        old2new_texture_name = {}
        for child in asset:
            if child.tag == 'material':
                mat_name = child.get('name')
                if mat_name not in material2mesh.keys():
                    continue
                mesh_full_name = material2mesh[mat_name]
                mat_name = mesh_full_name + '_' + mat_name
                child.set('name', mat_name)
                mesh_name2mat_name[mesh_full_name] = mat_name
            elif child.tag == 'mesh':
                mesh_name = child.get('file')
                mesh_name = os.path.join(folder_name, mesh_name)
                child.set('file', mesh_name)
                child.set('name', mesh_name.split('/')[-1].split('.')[0])
            elif child.tag == 'texture':
                texture_name = child.get('name')
                mesh_full_name = material2mesh[texture2material[texture_name]]
                texture_new_name = mesh_full_name + '_' + texture_name
                old2new_texture_name[texture_name] = texture_new_name
                texture_name = texture_new_name
                filename = child.get('file')
                filename = os.path.join('textured_objs_thickness', folder_name, filename)
                child.set('name', texture_name)
                child.set('file', filename)
            else:
                raise NotImplementedError
            children.append(child)
        
        for child in children:
            # if material, change name by old2new_texture_name
            if child.tag == 'material':
                if child.get('texture') is not None:
                    child.set('texture', old2new_texture_name[child.get('texture')])
            
        for child in children:
            meta_asset.append(child)
    

    return meta_asset

def get_pos_rpy(cur_joint, scale) -> Tuple[np.ndarray, np.ndarray]:
    origin = cur_joint.find('origin')

    rpy = origin.get('rpy')
    quat = None
    if rpy is not None:
        rpy = rpy.split(' ')
        rpy = np.asarray([float(r) for r in rpy])
        rot = R.from_euler('xyz', rpy, degrees=False)
        quat = rot.as_quat()
        quat = quat[[3, 0, 1, 2]] # w, x, y, z
        
    
    xyz = origin.get('xyz')
    if xyz is not None:
        xyz = xyz.split(' ')
        xyz = np.asarray([float(x) * scale for x in xyz])
    
    return xyz, quat

def create_joint_element(elem, cur_joint, scale):
    joint_type = cur_joint.get('type')
    joint_name = cur_joint.get('name')

    if joint_type == 'fixed':
        return
    
    axis = cur_joint.find('axis').get('xyz')
    joint_elem = SubElement(elem, 'joint', attrib={'name': joint_name, 'pos': "0 0 0", 'axis': axis})

    if joint_type == 'continuous':
        # hinge joint with unlimited range
        joint_elem.set("range", "-inf inf")
        joint_elem.set("type", "hinge")
        return
            
    limit = cur_joint.find('limit')
    lower_limit = float(limit.get('lower'))
    upper_limit = float(limit.get('upper'))

    if joint_type == 'revolute':
        joint_elem.set("range", f"{str(lower_limit)} {str(upper_limit)}")
        joint_elem.set("type", "hinge")
    elif joint_type == 'prismatic':
        joint_elem.set("range", f"{str(lower_limit * scale)} {str(upper_limit * scale)}")
        joint_elem.set("type", "slide")
    else:
        raise NotImplementedError

def create_geom(elem, mesh_dict, material_dict, cur_link, scale):
    # cur_link == child_link of the joint_link
    
    is_helper = True
    for child in cur_link:
        is_helper = False
        # parse pos, quat
        xyz, quat = get_pos_rpy(child, scale)
        if xyz is None:
            xyz = [0, 0, 0]
        if quat is None:
            quat = [1, 0, 0, 0]
        

        class_name = child.tag
        assert class_name == 'visual' or class_name == 'collision'
        # parse obj mesh filename and obj_name which should be the  value of the mesh attribute
        filename = child.find('geometry').find('mesh').get('filename')
        filename = filename.split('/')[-1]
        # this is a prefix
        obj_name = filename.split('.')[0]    
        # get all of the names of the mesh files whose prefix is obj_name
        mesh_names = mesh_dict[obj_name]
        for mesh_name in mesh_names:
            material_name = material_dict[mesh_name]
            if class_name == 'visual':
                is_helper = False
                # here mesh_name equals to mesh_full_name
                #create visual geom
                SubElement(elem, "geom", attrib={'pos': f"{xyz[0]} {xyz[1]} {xyz[2]}", 'quat': f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}", 'type': "mesh", \
                'mesh': f"{mesh_name}", 'material': material_name,'density': "0",'class': "visual"})
            elif class_name == 'collision':
                is_helper = False
                # create collision geom
                SubElement(elem, "geom", attrib={'pos': f"{xyz[0]} {xyz[1]} {xyz[2]}", 'quat': f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}", 'type': "mesh", \
                'mesh': f"{mesh_name}", 'material': material_name, 'class': "collision"})
        
    if is_helper:
        # if the body contains no joint, then add inertia to prevent error. 
        SubElement(elem, 'inertial', attrib={"pos": "0 0 0", "mass": "0.001", "diaginertia": "0.001 0.001 0.001"})

def create_sub_body(elem, cur_link_name, cur_joint_name, relation_graph, urdf_links_dict, urdf_joints_dict, scale, mesh_dict, material_dict):
    cur_link = urdf_links_dict[cur_link_name]
    cur_joint = urdf_joints_dict[cur_joint_name]
    xyz, quat = get_pos_rpy(cur_joint, scale)
    body = SubElement(elem, 'body', attrib={'name': cur_link_name})
    if xyz is not None:
        body.set('pos', f"{str(xyz[0])} {str(xyz[1])} {str(xyz[2])}")
    if quat is not None:
        body.set('quat', f"{str(quat[0])} {str(quat[1])} {str(quat[2])} {str(quat[3])}")

    create_joint_element(body, cur_joint, scale)
    create_geom(body, mesh_dict, material_dict, cur_link, scale)

    if cur_link_name not in relation_graph.keys():
        return

    elem = body            
    for cur_link_name, cur_joint_name in relation_graph[cur_link_name]:
        create_sub_body(elem, cur_link_name, cur_joint_name, relation_graph, urdf_links_dict, urdf_joints_dict, scale, mesh_dict, material_dict)
    
def create_worldbody(elem, asset_elem, dataset_dir, obj_idx, scale):
    urdf_file = os.path.join(dataset_dir, str(obj_idx), 'mobility.urdf')
    urdf_tree = ET.parse(urdf_file)
    urdf_links = urdf_tree.findall('link')
    urdf_joints = urdf_tree.findall('joint')
    
    urdf_links_dict = {v.get('name'):v for v in urdf_links}
    urdf_joints_dict = {v.get('name'):v for v in urdf_joints}
    
    # relation_graph = {parent: [(child1, joint_name), (child2, joint_name), ...]}
    relation_graph = dict()
    for joint_name, urdf_joint in urdf_joints_dict.items():
        child = urdf_joint.find('child').get('link')
        parent = urdf_joint.find('parent').get('link')
        if parent not in relation_graph.keys():
            relation_graph[parent] = [(child, joint_name)]
        else:
            relation_graph[parent].append((child, joint_name))
    
    # elem should be a body element.
    # asset_elem tells the name of the mesh you need to take care of
    # TODO: asset_elem대신 mesh_dict와 material_dict를 전달해야할듯
    meshes = asset_elem.findall('mesh')
    
    mesh_dict = defaultdict(lambda: []) # {mesh_prefix: [mesh_filenames]}
    for mesh in meshes:
        mesh_name = mesh.get('name')
        assert len(mesh_name.split('_')) == 1 or len(mesh_name.split('_')) == 2 , mesh_name
        prefix = mesh_name.split('_')[0]
        mesh_dict[prefix].append(mesh_name)
    
    
    materials = asset_elem.findall('material')
    material_dict = {} # {mesh_full_name: [material_filenames]}
    for material in materials:
        material_name = material.get('name')
        assert material_name[material_name.find('material')-1] == '_', material_name
        mesh_full_name = material_name[:material_name.find('material')-1]
        assert mesh_full_name not in material_dict.keys()
        material_dict[mesh_full_name] = material_name

    for cur_link_name, cur_joint_name in relation_graph['base']:
        create_sub_body(elem, cur_link_name, cur_joint_name, relation_graph, urdf_links_dict, urdf_joints_dict, scale, mesh_dict, material_dict)
    

    return urdf_joints_dict

    

def func(args):
    '''
    mostly derived by urdf2mjcf api
    '''
    _obj_idx = args.obj_idx
    scale = args.scale
    dataset_dir = args.dataset_dir

    if _obj_idx == -1:
        obj_indices = glob.glob(os.path.join(dataset_dir, '*'))
        obj_indices = [int(obj) for obj in obj_indices] # assert folders contain only the indices of the instances, as the p-mobility format does. 
    else:
        obj_indices = [_obj_idx]
    
    for obj_idx in obj_indices:
        root = Element('mujoco', attrib={'model':f"partnet_{obj_idx}"})
        SubElement(root, 'option', attrib={'iterations': "50", 'timestep': "0.001", 'solver': "PGS", 'gravity': "0 0 -9.81"})
        SubElement(root, 'compiler', attrib={'angle': "radian", 'meshdir': "textured_objs_thickness", "eulerseq": "zyx", "autolimits": "true"})
        
        meshdir =  root.find('compiler').get('meshdir')
        meshdir = os.path.join(dataset_dir, str(obj_idx), meshdir)
        obj_mesh_xml_files = get_obj_xml_files(meshdir)

        default = SubElement(root, 'default')
        SubElement(default, 'mesh', attrib={'scale': f"{scale} {scale} {scale}"})
        SubElement(default, 'joint', attrib={'limited': "true", 'damping': "0.01", "armature": "0.01", "frictionloss": "0.01"})
        SubElement(default, 'motor', attrib={'ctrllimited': "true"})
        SubElement(default, 'equality', attrib={'solref': "0.001 2"}) 
        vis_default = SubElement(default, 'default', attrib={"class": "visual"})
        SubElement(vis_default, 'geom', attrib={'condim': '1', 'contype': "0", "conaffinity": "0", "group": "2"})
        col_default = SubElement(default, 'default', attrib={"class": "collision"})
        SubElement(col_default, 'geom', attrib={'condim': '4', 'contype': "1", "conaffinity": "14", "friction": "0.9 0.2 0.2", 'solref': "0.001 2", "group": "3"})
        asset = create_asset(obj_mesh_xml_files)
        root.append(asset)

        worldbody = SubElement(root, 'worldbody')
        SubElement(worldbody, 'light', attrib={'directional': "true", 'pos': f"0 0 {str(0.5*scale)}", 'dir': f"0 0 -1", 'castshadow': "false"})
        urdf_joints_dict = create_worldbody(worldbody, asset, dataset_dir, obj_idx, scale) 


        #actuator for test
        actuator = SubElement(root, 'actuator')
        for joint_name in urdf_joints_dict.keys():
            if urdf_joints_dict[joint_name].get('type') == 'fixed':
                continue
            SubElement(actuator, 'motor', attrib={'name': joint_name, 'joint': joint_name, 'ctrllimited': "true", 'ctrlrange': "-1 1", 'gear': "1"})

        indent(root)


        tree = ElementTree(root)
        tree.write(os.path.join(dataset_dir, str(obj_idx), 'mujoco.mjcf'))
        print("SAVE", os.path.join(dataset_dir, str(obj_idx), 'mujoco.mjcf'))




def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset_dir',
        type=str,
        required=True,
        help='directory where partnet mobility dataset located.'
    )

    # --obj_idx 인자 추가 (필수)
    parser.add_argument(
        '--obj_idx',
        type=int,
        required=True,
        help='integer instance index of the partnet mobility object, -1 for all objects'
    )

    parser.add_argument(
        '--scale',
        type=float,
        default=0.1,
        help='object scale, default=0.1.'
    )

    # 인자 파싱
    args = parser.parse_args()

    # 입력된 인자 출력 (예시)
    print(f"Dataset Directory: {args.dataset_dir}")
    print(f"Object Index: {args.obj_idx}")
    
    func(args)



if __name__ == "__main__":
    main()
