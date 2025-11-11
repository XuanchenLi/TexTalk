import io
import math
import random
import numpy as np
from torch.utils import data as data
import torch as th
import torchaudio
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor, tensor2img
from basicsr.utils.registry import DATASET_REGISTRY
from glob import glob
import os
import cv2


@DATASET_REGISTRY.register()
class TexTalkerV5OffOffDataset(data.Dataset):

    def __init__(self, opt):
        self.opt = opt
        self.rank = th.distributed.get_rank()
        # file client (io backend)
        self.motion_set = opt['motion_type']
        self.wrinkle_set = opt['wrinkle_type']
        self.io_backend_opt = opt['io_backend']
        self.crop_strategy = opt['crop_strategy']
        self.crop_length = opt['n_motions'] * 2
        self.gt_folder = opt['dataroot_gt']
        self.fps = 30
        self.audio_unit = 16000. / self.fps  # num of samples per frame
        #self.n_audio_samples = int(np.floor(self.audio_unit * opt['n_motions']))
        self.n_motions = opt['n_motions']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'disk':
            id_list = glob(os.path.join(self.gt_folder, "*", "Records"))
            self.id_list = sorted([os.path.basename(os.path.dirname(p)) for p in id_list])
            self.tex_paths = {}
            self.motion_paths = {}
            for id in self.id_list:
                self.tex_paths[id] = self.get_frames(os.path.join(self.gt_folder, id, "Textures_512"))
                self.motion_paths[id] = self.get_frames(os.path.join(self.gt_folder, id, "Models"))
                

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):
        # Read audio and coef
        id_name = self.id_list[index]
        #frame_idx = random.randint(1, len(self.tex_paths[id_name]) - 1)
        seq_len = len(self.tex_paths[id_name])
        if index >= len(self.id_list) - 10:
            seq_len = np.ceil(len(self.tex_paths[id_name]) / 100) // 2 * 100
        audio_path = os.path.join(self.gt_folder, id_name, "Records", "enhanced_vocal.wav")
        audio_clip, sr = torchaudio.load(audio_path)
        #print(id_name, audio_clip.shape)
        #assert sr == 16000, f'Invalid sampling rate: {sr}'
        audio_clip = audio_clip.squeeze()
        audio_len = audio_clip.shape[0]
        seq_len = int(min(seq_len, audio_len // 16000 * 30))
        # Crop the audio and coef
        if self.crop_strategy == 'random':
            start_frame_idx = np.random.randint(1, seq_len - self.crop_length)
        elif self.crop_strategy == 'begin':
            start_frame_idx = 1
        elif self.crop_strategy == 'end':
            start_frame_idx = seq_len - self.crop_length - 1
        else:
            raise ValueError(f'Unknown crop strategy: {self.crop_strategy}')
        motion_latents = []
        tex_latents = []
        for idx in range(start_frame_idx, start_frame_idx + self.crop_length):
            motion_feat_path = self.motion_paths[id_name][idx]
            tex_feat_path = self.tex_paths[id_name][idx]
            motion_latents.append(th.load(os.path.join(motion_feat_path, "latent_gt.pth"), map_location=f"cuda:{self.rank}").flatten().unsqueeze(0))
            tex_latents.append(th.from_numpy(th.load(os.path.join(tex_feat_path, "latent_gt.pth"), map_location=f"cuda:{self.rank}")).flatten().unsqueeze(0))
        motion_latents = th.cat(motion_latents, dim=0)
        tex_latents = th.cat(tex_latents, dim=0)
        #print(motion_latents.shape, tex_latents.shape)
        
        
        audio_start = 16000 * (start_frame_idx // self.fps) + int(np.floor((start_frame_idx % self.fps) * self.audio_unit))
        audio_end = audio_start + int(np.floor(self.crop_length * self.audio_unit))
        #audio_end =  int(np.floor((start_frame_idx + self.crop_length) * self.audio_unit))
        audio_clip = audio_clip[audio_start:audio_end]
        #print(audio_start,audio_end)
        #print(audio_start, audio_end, start_frame_idx, self.audio_unit)
        assert audio_end <= audio_len, f'audio {id_name} out of range: {audio_end} exceeds {audio_len}, start frame idx: {start_frame_idx} seq_len: {seq_len}'
        assert audio_clip.shape[0] == audio_end - audio_start, f'Invalid audio length: {audio_clip.shape[0]} should be {audio_end - audio_start}, id: {id_name}'
        # audio_mean = audio_clip.mean()
        # audio_std = audio_clip.std()
        audio_mean = 0
        audio_std = 1
        #audio_clip = (audio_clip - audio_mean) / (audio_std + 1e-5)
        
        shape_map_path = os.path.join(self.gt_folder, id_name, "latent_motion_ave.pth")
        # tex_map_path = os.path.join(self.gt_folder, id_name, "latent_tex_ave.pth")
        tex_map_path = os.path.join(self.gt_folder, id_name, "latent_tex_ave.pth")
        shape_map = th.from_numpy(th.load(shape_map_path, map_location=f"cuda:{self.rank}")).flatten().unsqueeze(0)
        tex_map = th.from_numpy(th.load(tex_map_path, map_location=f"cuda:{self.rank}")).flatten().unsqueeze(0)
        if self.motion_set == "offset":
            motion_latents = motion_latents - shape_map.expand((motion_latents.shape[0], -1)).to(motion_latents)
        if self.wrinkle_set == "offset":
            tex_latents = tex_latents - tex_map.expand((tex_latents.shape[0], -1)).to(tex_latents)
        

        # Extract two consecutive audio/coef clips
        n_audio_samples = audio_clip.shape[0] // 2
        audio_pair = [audio_clip[:n_audio_samples], audio_clip[-n_audio_samples:]]
        motion_feat_pair = [motion_latents[:self.n_motions],
                     motion_latents[-self.n_motions:]]
        tex_feat_pair = [tex_latents[:self.n_motions],
                     tex_latents[-self.n_motions:]]
        

        style_code = th.tensor([index])
        # style_code[..., index] = 1.0
        #print(audio_pair[0].shape, audio_pair[1].shape)
        #return audio_pair, motion_feat_pair, tex_feat_pair, (audio_mean, audio_std), shape_map, tex_map
        return {"audio_pair": audio_pair, 
                "motion_latent_pair": motion_feat_pair, 
                "tex_latent_pair": tex_feat_pair,
                "audio_stat": (audio_mean, audio_std),
                "shape_map": shape_map,
                "tex_map": tex_map,
                "style_code": style_code
                }
    
    
    @staticmethod
    def get_frames(rt):
        subdirs = os.listdir(rt)
        subdirs = [os.path.join(rt,sd) for sd in subdirs]
        subdirs = sorted([sd for sd in subdirs if os.path.isdir(sd) and os.path.basename(sd).startswith("0")])
        return subdirs
