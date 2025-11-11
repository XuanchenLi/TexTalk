''' 
Generate 2d uv maps representing different attributes(colors, depth, image position, etc)
: render attributes to uv space.
'''
import argparse
import os, sys
import cv2
import numpy as np
import scipy.io as sio
from skimage import io
import skimage.transform
from time import time
import matplotlib.pyplot as plt
import pywavefront
from PIL import Image
from tqdm import tqdm
import copy
from glob import glob

import face3d
from face3d import mesh

def triangulate_faces(faces):
    """
    手动三角化网格。
    对于每个四边形面，使用第一个顶点到第三个顶点的对角线进行分割。
    """
    triangulated_faces = []
    for face in faces:
        if len(face) == 4: 
            triangulated_faces.append([face[0], face[1], face[2]])
            triangulated_faces.append([face[0], face[2], face[3]])
        elif len(face) == 3: 
            triangulated_faces.append(face)
    return triangulated_faces


def load_faces_vertices(file_path):
    faces = []
    uv_faces = []
    normal_faces = []
    uv2vex = {}
    vex2uv = {}
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('f '):
                parts = line.strip().split(' ')
                parts = [part for part in parts if not len(part) == 0]
                face = [int(part.split('/')[0]) - 1 for part in parts[1:]]  # OBJ索引从1开始，Python列表从0开始
                uv_face = [int(part.split('/')[1]) - 1 for part in parts[1:]]
                for temp in range(len(uv_face)):
                    uv2vex[uv_face[temp]] = face[temp]
                for temp in range(len(face)):
                    if vex2uv.get(face[temp]) is None:
                        vex2uv[face[temp]] = uv_face[temp]
                    else:
                        pass
                normal_faces = [int(part.split('/')[2]) - 1 for part in parts[1:] if len(part.split('/')) == 3]
                faces.append(face)
                uv_faces.append(uv_face)
    scene = pywavefront.Wavefront(file_path, collect_faces=True, create_materials=True)


    vertices = np.array(scene.vertices)
    #print(vertices)
    uv = np.array(scene.parser.tex_coords)
    return triangulate_faces(faces), triangulate_faces(uv_faces), vertices, normal_faces, uv


def process_uv(uv_coords, uv_h = 256, uv_w = 256):
    uv_coords[:,0] = uv_coords[:,0]*(uv_w - 1)
    uv_coords[:,1] = uv_coords[:,1]*(uv_h - 1)
    uv_coords[:,1] = uv_h - uv_coords[:,1] - 1
    uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0], 1)))) # add z
    return uv_coords

def duplicate_texture_vertex_color_2(uvs, uvs_texture, colors):
    color_render = []
    uv_dict = {}
    for idx, uvs_ in enumerate(uvs_texture):
        for uv in uvs_:
            uv_dict[tuple(uv)] = idx
    #print(uvs_texture[8279])
    color_render = [colors[uv_dict[tuple(uv)]] for uv in uvs]
    return np.array(color_render)


def write_texture(path, uv_coords, colors, faces, res=512, stat_dict=None, save_img=False):
    uv_h = uv_w = res
    uv_texture_map = mesh.render.render_colors(uv_coords, faces, colors, uv_h, uv_w, c=3)
    if save_img:
        image = Image.fromarray((uv_texture_map * 255).astype(np.uint8))
        image.save(os.path.join(os.path.dirname(path), os.path.basename(path).split('.')[0] + ".png"))

    np.save(path, uv_texture_map)
    if stat_dict is not None:
        np.save(os.path.join(os.path.dirname(path), "stat_id.npy"), stat_dict)

def get_vertex_uvs(scene):
    _, mesh = list(scene.meshes.items())[0]
    vertices = scene.vertices
    uvs = scene.parser.tex_coords

    vertex_uvs = {}
    for fid, face in enumerate(mesh.faces):
        for vid, vertex_index in enumerate(face):
            pos = fid*24+vid*8  # T2F_N3F_V3F
            uv = mesh.materials[0].vertices[pos:pos+2]
            if vertex_index in vertex_uvs.keys():
                #print("1231231231241241 2414124124")
                vertex_uvs[vertex_index].append(uv)
                #print(vertex_uvs[vertex_index])
                #print(len(vertex_uvs[vertex_index]))
            else:
                vertex_uvs[vertex_index] = [uv]  

    vertex_uvs = [list(set(tuple(item) for item in value)) for key, value in sorted(vertex_uvs.items())]
    return uvs, vertex_uvs


def render_diff_depth_global(save_folder, obj_name, stat_dict, face_mask, uv_textures_ori, first_vertices, res=512, save_img=False, scale=1.0):
    #gmax_0, gmin_0, gmax_1, gmin_1, gmax_2, gmin_2 = 756., 283., 846., 215., 212.0, -224.0
    obj_file = os.path.join(save_folder, obj_name)
    _, uv_faces_ori, vertices, _, uv_ori =  load_faces_vertices(obj_file)
    uv_faces_ori = np.array(uv_faces_ori)
    
    vertices = vertices - first_vertices
    vertices *= scale
    uv_h = uv_w = res
    image_h = image_w = res
    # --modify vertices(transformation. change position of obj)
    #s = image_h / (np.max(vertices[:,1]) - np.min(vertices[:,1]))
    s = 1.0
    R = mesh.transform.angle2matrix([0, 0, 0]) 
    t = [0, 0, 0]
    transformed_vertices = mesh.transform.similarity_transform(vertices, s, R, t)
    # --load uv coords
    uv_coords = uv_ori.copy()

    uv_coords = process_uv(uv_coords, uv_h, uv_w)
    projected_vertices = transformed_vertices.copy() # use standard camera & orth projection here
    #image_vertices= mesh.transform.to_image(projected_vertices, image_h, image_w)
    image_vertices = projected_vertices
    # image_vertices = np.zeros_like(projected_vertices)
    # image_vertices[face_mask] = projected_vertices[face_mask]
    position_global = copy.deepcopy(image_vertices)
    gmax_0, gmin_0= stat_dict["gmax_0"], stat_dict["gmin_0"]
    gmax_1, gmin_1= stat_dict["gmax_1"], stat_dict["gmin_1"]
    gmax_2, gmin_2= stat_dict["gmax_2"], stat_dict["gmin_2"]
    #stat_dict["scale"] = image_h / (np.max(vertices[:,1]) - np.min(vertices[:,1]))
    position_global[:,0] = (position_global[:,0] - gmin_0) / (gmax_0 - gmin_0)
    position_global[:,1] = (position_global[:,1] - gmin_1) / (gmax_1 - gmin_1)
    position_global[:,2] = (position_global[:,2] - gmin_2) / (gmax_2 - gmin_2) # translate z 
    # position_global_new = np.zeros_like(position_global)
    # position_global_new[face_mask] = position_global[face_mask]
    attribute = duplicate_texture_vertex_color_2(uv_ori, uv_textures_ori, position_global)
    # print(attribute.max(), attribute.min())
    # attribute = np.ones_like(attribute)
    write_texture('{}/{}.npy'.format(save_folder, "diff"), uv_coords, attribute, uv_faces_ori, res=uv_h, save_img=save_img)
    

def get_ids(rt):
    subdirs = os.listdir(rt)
    subdirs = [os.path.join(rt,sd) for sd in subdirs]
    subdirs = sorted([sd for sd in subdirs if os.path.isdir(sd) and os.path.exists(os.path.join(sd, "Models"))])
    return subdirs


def get_frames(rt):
    subdirs = os.listdir(rt)
    subdirs = [os.path.join(rt,sd) for sd in subdirs]
    subdirs = sorted([sd for sd in subdirs if os.path.isdir(sd) and os.path.basename(sd).startswith("0")])
    return subdirs

def obtain_curr_stat_new(vertices, scale, res=512):
    image_h = image_w = res
    # --modify vertices(transformation. change position of obj)
    s = scale
    R = mesh.transform.angle2matrix([0, 0, 0]) 
    t = [0, 0, 0]
    transformed_vertices = mesh.transform.similarity_transform(vertices, s, R, t)

    projected_vertices = transformed_vertices.copy() # use standard camera & orth projection here
    #image_vertices = mesh.transform.to_image(projected_vertices, image_h, image_w)

    position = projected_vertices

    max_0 = np.max(position[:,0])
    min_0 = np.min(position[:,0])

    max_1 = np.max(position[:,1])
    min_1 = np.min(position[:,1])

    max_2 = np.max(position[:,2])
    min_2 = np.min(position[:,2])

    return max_0, min_0, max_1, min_1, max_2, min_2




def get_uv_info(obj_path):
    scene = pywavefront.Wavefront(obj_path, collect_faces=True)
    return get_vertex_uvs(scene)



def compute_seq_diff_stat(rt, face_mask, res=512):
    frames = get_frames(os.path.join(rt, "Models"))
    stat_dict = {}
    stat_dict["gmin_0"] = 999999
    stat_dict["gmax_0"] = -999999
    stat_dict["gmin_1"] = 999999
    stat_dict["gmax_1"] = -999999
    stat_dict["gmin_2"] = 999999
    stat_dict["gmax_2"] = -999999
    stat_dict["scale"] = 999999
    first_obj_path = os.path.join(frames[0], "face_aligned.obj")
    _, _, fobj_v, _, _ =  load_faces_vertices(first_obj_path)
    diff_cache = []
    for idx in tqdm(range(1, len(frames)), desc="Compute Scale"):
        _, _, nobj_v, _, _ =  load_faces_vertices(os.path.join(frames[idx], "face_aligned.obj"))
        v_diff = nobj_v - fobj_v
        #v_diff = v_diff[face_mask]
        diff_cache.append(v_diff)
        stat_dict["scale"] = 1.0
        #stat_dict["scale"] = min(stat_dict["scale"], res / (np.max(v_diff[:,1]) - np.min(v_diff[:,1])))
    for idx in tqdm(range(1, len(frames)), desc="Compute Max Min"):
        v_diff = diff_cache[idx - 1]
        max_0, min_0, max_1, min_1, max_2, min_2 = obtain_curr_stat_new(v_diff, stat_dict["scale"])
        stat_dict["gmin_0"] = min(stat_dict["gmin_0"], min_0)
        stat_dict["gmin_1"] = min(stat_dict["gmin_1"], min_1)
        stat_dict["gmin_2"] = min(stat_dict["gmin_2"], min_2)
        stat_dict["gmax_0"] = max(stat_dict["gmax_0"], max_0)
        stat_dict["gmax_1"] = max(stat_dict["gmax_1"], max_1)
        stat_dict["gmax_2"] = max(stat_dict["gmax_2"], max_2)
    print(stat_dict)
    np.save(os.path.join(rt, "stat_seq_diff.npy"), stat_dict)
     

def compute_seq_diff_global_stat(dataset_dir):
    if os.path.exists("assets/stat_seq_diff_global.npy"):
        return
    ids = get_ids(dataset_dir)
    face_mask = np.load("assets/face_only_diff.npy")
    stat_dict_g = {}
    stat_dict_g["gmin_0"] = 999999
    stat_dict_g["gmax_0"] = -999999
    stat_dict_g["gmin_1"] = 999999
    stat_dict_g["gmax_1"] = -999999
    stat_dict_g["gmin_2"] = 999999
    stat_dict_g["gmax_2"] = -999999
    stat_dict_g["scale"] = 1.0
    
    for id_dir in ids:
        print(id_dir)
        #frames = get_frames(os.path.join(id_dir, "Models"))
        if not os.path.exists(os.path.join(id_dir, "stat_seq_diff.npy")):
            compute_seq_diff_stat(id_dir, face_mask)
        #compute_seq_diff_stat_new(id_dir, face_mask)
        stat_dict = np.load(os.path.join(id_dir, "stat_seq_diff.npy"), allow_pickle=True).item()
        for k in stat_dict_g.keys():
            if "min" in k or "scale" in k:
                stat_dict_g[k] = min(stat_dict_g[k], stat_dict[k])
            elif "max" in k:
                stat_dict_g[k] = max(stat_dict_g[k], stat_dict[k])
    np.save(os.path.join(dataset_dir, "stat_seq_diff_global.npy"), stat_dict_g)
    print(stat_dict_g)

                
def normalize_seq_diff_global(dataset_dir, use_local=False):
    ids = get_ids(dataset_dir)
    uv_coords, uv_textures_ori = get_uv_info("assets/template.obj")
    face_mask = np.load("assets/face_only_diff.npy")
    for id_dir in ids:
        print(id_dir)
        frames = get_frames(os.path.join(id_dir, "Models"))
        if use_local:
            if not os.path.exists(os.path.join(id_dir, "stat_seq_diff.npy")):
                compute_seq_diff_stat(id_dir, face_mask)
            stat_dict = np.load(os.path.join(id_dir, "stat_seq_diff.npy"), allow_pickle=True).item()
        else:
            stat_dict = np.load("assets/stat_seq_diff_global.npy", allow_pickle=True).item()
        # print(stat_dict)
        first_obj_path = os.path.join(frames[0], "face_aligned.obj")
        _, _, fobj_v, _, _ =  load_faces_vertices(first_obj_path)
        for idx in tqdm(range(1, len(frames)), desc=f"Render Diff Map"):
            render_diff_depth_global(frames[idx], "face_aligned.obj", stat_dict, face_mask, uv_textures_ori, fobj_v, save_img=True)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='/data/your_usrname/TexTalkTest/')
    args = parser.parse_args()
    compute_seq_diff_global_stat(args.input_dir)
    normalize_seq_diff_global(args.input_dir, use_local=False)
