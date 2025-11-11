import argparse
import math
import os
import tempfile
import warnings
from pathlib import Path
from glob import glob
import cv2
import librosa
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils.misc import get_device
from basicsr.utils.registry import ARCH_REGISTRY
import pywavefront
import time

warnings.filterwarnings('ignore', message='PySoundFile failed. Trying audioread instead.')

    

def get_frames(rt):
    subdirs = os.listdir(rt)
    subdirs = [os.path.join(rt,sd) for sd in subdirs]
    subdirs = sorted([sd for sd in subdirs if os.path.isdir(sd) and os.path.basename(sd).startswith("0")])
    return subdirs


@torch.no_grad()
def inference(args, cfg_cond=["audio"], cfg_scale=[1.0],
              pad_mode="zero", n_repetitions=1, 
              dynamic_threshold=None, out_dir=""):

    
    model = ARCH_REGISTRY.get('TexTalker')(feature_dim=1024, n_motions=90, n_prev_motions=15, d_style=72).to(args.device)
    model.load_state_dict(torch.load(args.model_path)['params'], strict=True)
    model.eval()
    model = model.to(args.device)
    # Step 1: Preprocessing
    # Preprocess audio
    audio, _ = torchaudio.load(args.audio_path)
    audio = audio.squeeze().to(args.device)
    seqlen=0

    assert audio.ndim == 1, 'Audio must be 1D tensor.'
    
    # divide into synthesize units and do synthesize
    clip_len = int(len(audio) / 16000 * args.fps)
    stride = model.n_motions
    if clip_len <= model.n_motions:
        n_subdivision = 1
    else:
        n_subdivision = math.ceil(clip_len / stride)
        
    # padding audio
    audio_unit = 16000. / args.fps
    n_audio_samples = int(np.floor(audio_unit * 2 * model.n_motions)) // 2
    n_padding_audio_samples = n_audio_samples * n_subdivision - len(audio)
    n_padding_frames = math.ceil(n_padding_audio_samples / audio_unit)
    if n_padding_audio_samples > 0:
        if pad_mode == 'zero':
            padding_value = 0
        elif pad_mode == 'replicate':
            padding_value = audio[-1]
        else:
            raise ValueError(f'Unknown pad mode: {pad_mode}')
        audio = F.pad(audio, (0, n_padding_audio_samples), value=padding_value)
    audio_feat = model.extract_audio_feature(audio.unsqueeze(0), model.n_motions * n_subdivision)
    style_code = torch.tensor([0]).to(args.device)

    #prepare map conditions
    shape_map_path = os.path.join(args.input, "latent_motion_ave.pth")
    tex_map_path = os.path.join(args.input, "latent_tex_ave.pth")
    shape_map = torch.from_numpy(torch.load(shape_map_path)).flatten().unsqueeze(0).to(args.device)
    tex_map = torch.from_numpy(torch.load(tex_map_path)).flatten().unsqueeze(0).to(args.device)

    # Step 2: Inferencing
    os.makedirs(out_dir, exist_ok=True)
    for i in tqdm(range(0, n_subdivision)):
        start_idx = i * stride
        end_idx = start_idx + model.n_motions
        start_frame = start_idx + seqlen
        indicator = torch.ones((n_repetitions, model.n_motions)).to(args.device)
        if indicator is not None and i == n_subdivision - 1 and n_padding_frames > 0:
            indicator[:, -n_padding_frames:] = 0
            
        # prepare audio input
        audio_in = audio_feat[:, start_idx:end_idx].expand(n_repetitions, -1, -1).to(args.device)

        # generate latents
        if i == 0:
            feat, noise, prev_audio_feat = model.sample(audio_in, shape_map, tex_map, style_code,
                                                                    indicator=indicator, 
                                                                    cfg_cond=cfg_cond, cfg_scale=cfg_scale,
                                                                    dynamic_threshold=dynamic_threshold)
        else:
            feat, noise, prev_audio_feat = model.sample(audio_in, shape_map, tex_map, style_code,
                                                                    prev_motion_feat, prev_tex_feat, prev_audio_feat,
                                                                    noise, indicator=indicator, 
                                                                    cfg_cond=cfg_cond, cfg_scale=cfg_scale,
                                                                    dynamic_threshold=dynamic_threshold)
        prev_feat = feat[:, -model.n_prev_motions:].clone()
        prev_motion_feat, prev_tex_feat = torch.split(prev_feat, prev_feat.shape[2]//2, dim=2)
        prev_audio_feat = prev_audio_feat[:, -model.n_prev_motions:]

        latents = feat
        if i == n_subdivision - 1 and n_padding_frames > 0:
            latents = latents[:, :-n_padding_frames]  # delete padded frames
        
        # save latents
        for idx in range(latents.shape[1]):
            frame_num = start_frame + idx + 1
            latents_out = latents[0, idx].unsqueeze(0)
            assert len(latents_out.shape) == 2
            motion_out, tex_out = torch.split(latents_out, latents_out.shape[1]//2, dim=1)
            motion_out = motion_out + shape_map
            tex_out = tex_out + tex_map
            outpath = os.path.join(out_dir, "%06d" % frame_num)
            os.makedirs(outpath, exist_ok=True)
            torch.save(motion_out.cpu().numpy(), os.path.join(outpath, "latent_motion.pth"))
            torch.save(tex_out.cpu().numpy(), os.path.join(outpath, "latent_tex.pth"))


def recover_map(img_0, w_map):
    w_map = w_map / 2 + 0.5
    res = torch.log((w_map + 1e-8)/(1-w_map + 1e-8)) * img_0
    assert not torch.any(torch.isnan(res))
    return res


def get_subdirs(rt):
    subdirs = os.listdir(rt)
    subdirs = [os.path.join(rt,sd) for sd in subdirs]
    subdirs = sorted([sd for sd in subdirs if os.path.isdir(sd)])
    return subdirs


@torch.no_grad()
def decode_map_from_latents(args, input_dir):
    model = ARCH_REGISTRY.get('VQAutoEncoder')(img_size=512, nf=64, codebook_size=1024, quantizer='nearest', ch_mult=[1, 2, 2, 4, 4, 8], emb_dim=16).to(args.device)
    model.load_state_dict(torch.load(args.tex_decoder_path)['params_ema'], strict=True)
    model.eval()
    model = model.to(args.device)
    tex_paths = sorted(glob(os.path.join(input_dir, "*", "latent_tex.pth")))
    img_0 = cv2.imread(os.path.join(args.input, "face.png"), cv2.IMREAD_COLOR).astype(np.float32) / 255.
    img_0 = img2tensor(img_0).to(args.device)


    for tex_path in tqdm(tex_paths):
        latent = torch.from_numpy(torch.load(tex_path)).reshape(1, 16, 16, 16).to(args.device)

        with torch.no_grad():
            quant, codebook_loss, quant_stats = model.quantize(latent)
            output = model.generator(quant)
            rmap = recover_map(img_0, output[0])
            rmap = tensor2img(rmap)
            output = tensor2img(output[0])
            out_dir = os.path.dirname(tex_path)
            cv2.imwrite(os.path.join(out_dir, 'wrinkle.png'), output)
            cv2.imwrite(os.path.join(out_dir, 'face.png'), rmap)

            
    model = ARCH_REGISTRY.get('VQAutoEncoder')(img_size=512, nf=64, codebook_size=1024, quantizer='nearest', ch_mult=[1, 2, 2, 4, 4, 8], emb_dim=16).to(args.device)
    model.load_state_dict(torch.load(args.motion_decoder_path)['params_ema'], strict=True)
    model.eval()
    model = model.to(args.device)
    tex_paths = sorted(glob(os.path.join(input_dir, "*", "latent_motion.pth")))
    for tex_path in tqdm(tex_paths):
        latent = torch.from_numpy(torch.load(tex_path)).reshape(1, 16, 16, 16).to(args.device)
        with torch.no_grad():
            quant, codebook_loss, quant_stats = model.quantize(latent)
            output = model.generator(quant)
            #print(output.shape)
            out_dir = os.path.dirname(tex_path)
            np.save(os.path.join(out_dir, 'diff.npy'), output[0].clone().permute(1, 2, 0).cpu().numpy())
            output = tensor2img(output[0])
            cv2.imwrite(os.path.join(out_dir, 'diff.png'), output)
    


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
                vertex_uvs[vertex_index].append(uv)
            else:
                vertex_uvs[vertex_index] = [uv]  

    vertex_uvs = [list(set(tuple(item) for item in value)) for key, value in sorted(vertex_uvs.items())]
    return uvs, vertex_uvs

def load_faces_vertices(file_path):
    scene = pywavefront.Wavefront(file_path, collect_faces=True, create_materials=True)
    vertices = np.array(scene.vertices)
    return vertices

def get_uv_info(obj_path):
    scene = pywavefront.Wavefront(obj_path, collect_faces=True)
    return get_vertex_uvs(scene)


def get_color_from_texture2(image, uv_coord):
    width, height, _ = image.shape
    #print(image.shape)
    u, v = uv_coord
    u = u % 1
    v = v % 1

    y = u * width
    x = (1 - v) * height

    x1, y1 = int(x), int(y)
    x2, y2 = min(x1 + 1, width - 1), min(y1 + 1, height - 1)

    Q11 = image[x1,y1]     #image.getpixel((x1, y1))
    Q21 = image[x2, y1]#.getpixel((x2, y1))
    Q12 = image[x1, y2]#.getpixel((x1, y2))
    Q22 = image[x2, y2]#.getpixel((x2, y2))

    r = 1
    m = 1
    while all([x < 1e-2 for x in Q11]) or all([x < 1e-2 for x in Q21]) or all([x < 1e-2 for x in Q12]) or all([x < 1e-2 for x in Q22]):

        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                #print(dx, dy)
                newx1 = x1 + dx
                newx2 = x2 + dx
                newy1 = y1 + dy
                newy2 = y2+dy
                newx = x + dx
                newy = y + dy
                if newx1 < 0 or newx1 > 511:
                    continue
                if newx2 < 0 or newx2 > 511:
                    continue
                if newy1 < 0 or newy1 > 511:
                    continue
                if newy2 < 0 or newy2 > 511:
                    continue
                Q11 = image[newx1, newy1]#image.getpixel((x1, y1))
                Q21 = image[newx2, newy1]#.getpixel((x2, y1))
                Q12 = image[newx1, newy2]#.getpixel((x1, y2))
                Q22 = image[newx2, newy2]#.getpixel((x2, y2))
                if not (all([x < 1e-2 for x in Q11]) or all([x < 1e-2 for x in Q21]) or all([x < 1e-2 for x in Q12]) or all([x < 1e-2 for x in Q22])):
                    R1 = ((newx2 - newx) * Q11[0] + (newx - newx1) * Q21[0],
                        (newx2 - newx) * Q11[1] + (newx - newx1) * Q21[1],
                        (newx2 - newx) * Q11[2] + (newx - newx1) * Q21[2])

                    R2 = ((newx2 - newx) * Q12[0] + (newx - newx1) * Q22[0],
                        (newx2 - newx) * Q12[1] + (newx - newx1) * Q22[1],
                        (newx2 - newx) * Q12[2] + (newx - newx1) * Q22[2])

                    P = ((newy2 - newy) * R1[0] + (newy - newy1) * R2[0],
                        (newy2 - newy) * R1[1] + (newy - newy1) * R2[1],
                        (newy2 - newy) * R1[2] + (newy - newy1) * R2[2])
                    #print( P)
                    return tuple([val for val in P])
        r += 1

    R1 = ((x2 - x) * Q11[0] + (x - x1) * Q21[0],
        (x2 - x) * Q11[1] + (x - x1) * Q21[1],
        (x2 - x) * Q11[2] + (x - x1) * Q21[2])

    R2 = ((x2 - x) * Q12[0] + (x - x1) * Q22[0],
        (x2 - x) * Q12[1] + (x - x1) * Q22[1],
        (x2 - x) * Q12[2] + (x - x1) * Q22[2])

    P = ((y2 - y) * R1[0] + (y - y1) * R2[0],
        (y2 - y) * R1[1] + (y - y1) * R2[1],
        (y2 - y) * R1[2] + (y - y1) * R2[2])
    #print( P)
    return tuple([val for val in P])


def recover_mesh_from_diff(fovj_v, map_path, stat_path, uv_textures_ori):
    #uv_coords, uv_textures_ori = get_uv_info(uv_tmp_path)
    texture = np.load(map_path)
    texture = texture.astype(np.float64)
    #colors = np.array([get_color_from_texture(texture, uv_coord) for uv_coord in uv_coords[vex2uv_np]])  
    colors = []
    for vid in range(len(uv_textures_ori)):
        clrs = [get_color_from_texture2(texture, uv_coord) for uv_coord in uv_textures_ori[vid]]
        
        if len(clrs) > 1:
            #print(clrs)
            clrs = [t for t in clrs if not all(x == 0 for x in t)]
            clrs = [(0, 0, 0)] if len(clrs) == 0 else clrs
            clrs = np.array(clrs)
            clrs = np.mean(clrs, axis=0)
            #print(clrs.shape)
            clrs = [tuple(clrs.tolist())]
            # clrs = clrs[0]
            # clrs = [clrs]
            pass
        else:
            pass    
        if all([x == 0 for x in clrs[0]]):
            print(clrs)
        colors.append(clrs)
    #print(cnt)
    #print(colors)
    colors = np.array(colors).squeeze(1)
    #print(colors.shape)
    stat_dict = np.load(stat_path, allow_pickle=True).item()
    gmax_0, gmin_0= stat_dict["gmax_0"], stat_dict["gmin_0"]
    gmax_1, gmin_1= stat_dict["gmax_1"], stat_dict["gmin_1"]
    gmax_2, gmin_2= stat_dict["gmax_2"], stat_dict["gmin_2"]
    colors[:,0] = colors[:,0] * (gmax_0 - gmin_0) + gmin_0
    colors[:,1] = colors[:,1] * (gmax_1 - gmin_1) + gmin_1
    colors[:,2] = colors[:,2] * (gmax_2 - gmin_2) + gmin_2
    
    res = texture.shape[0]

    colors *= 1/stat_dict["scale"]
    colors += fovj_v
    with open('{}/{}.obj'.format(os.path.dirname(map_path), "face"), 'w') as f:
        for color in colors:
            f.write('v ' + str(color[0]) + ' ' + str(color[1]) + ' ' + str(color[2]) + '\n')
        with open('assets/uv_diff.txt', 'r') as f2:
            lines = f2.readlines()
        for line in lines:
            f.write(line)
    return np.array(colors)

def recover_seq_from_diff_map(args, id_dir):
    uv_coords, uv_textures_ori = get_uv_info('assets/template.obj')
    obj_path = os.path.join(args.input, "face.obj")
    fobj_v=  load_faces_vertices(obj_path)
    frames = get_frames(id_dir)
    for idx in tqdm(range(0, len(frames)), desc=f"Render Diff Map"):
        vertices = recover_mesh_from_diff(fobj_v, os.path.join(frames[idx], "diff.npy"),
                                'assets/stat_seq_diff_global.npy', uv_textures_ori)



@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--tex_decoder_path',
        type=str,
        default= 'checkpoints/tex_vae.pth'
    )
    parser.add_argument(
        '--motion_decoder_path',
        type=str,
        default= 'checkpoints/motion_vae.pth'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default= 'checkpoints/textalker.pth'
    )

    # set arguments
    parser.add_argument('--input', type=str, default=f'example/test_id')
    parser.add_argument('--output', type=str, default='results', help='output folder')
    parser.add_argument('--audio_path', type=str, default=f'example/Records/enhanced_vocal.wav')
    
    args = parser.parse_args()
    
    exp_name = f"TexTalker"

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.fps = 30

    id_name = os.path.dirname(args.input)

    #args.output = os.path.join(args.output, exp_name, iter, id_name)
    out_dir = os.path.join(args.output, exp_name, id_name)

    # set up model
    cfg_cond = ["audio"]
    cfg_scale = [1.0]  #scale_audio, scale_style
    
    dynamic_threshold_ratio = 0.
    dynamic_threshold_min = 1.
    dynamic_threshold_max = 4.
    
    if dynamic_threshold_ratio > 0:
        dynamic_threshold = (dynamic_threshold_ratio, dynamic_threshold_min,
                            dynamic_threshold_max)
    else:
        dynamic_threshold = None

    out_path = out_dir

    inference(args, cfg_cond=cfg_cond, cfg_scale=cfg_scale,
            n_repetitions=1, 
            dynamic_threshold=dynamic_threshold, out_dir=out_path)
    
    decode_map_from_latents(args, out_path)
    recover_seq_from_diff_map(args, out_path)


if __name__ == '__main__':
    main()