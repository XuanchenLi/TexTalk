import math
import random
import numpy as np
from torch.utils import data as data
import torch as th
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor, tensor2img
from basicsr.utils.registry import DATASET_REGISTRY
from glob import glob
import os
import cv2


@DATASET_REGISTRY.register()
class TexTalkerMixDataset(data.Dataset):

    def __init__(self, opt):
        super(TexTalkerMixDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        # self.mean = opt['mean'] if 'mean' in opt else None
        # self.std = opt['std'] if 'std' in opt else None
        self.gt_folder = opt['dataroot_gt']
        self.type = opt["asset_type"]
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'disk':
            if self.type == "texture":
                id_list = glob(os.path.join(self.gt_folder, "*", "Textures_512"))
                self.id_list = sorted([os.path.basename(os.path.dirname(p)) for p in id_list])
                self.paths = {}
                for id in self.id_list:
                    self.paths[id] = self.get_frames(os.path.join(self.gt_folder, id, "Textures_512"))
            elif self.type == "mesh":
                id_list = glob(os.path.join(self.gt_folder, "*", "Models"))
                self.id_list = sorted([os.path.basename(os.path.dirname(p)) for p in id_list])
                self.paths = {}
                for id in self.id_list:
                    self.paths[id] = self.get_frames(os.path.join(self.gt_folder, id, "Models"))


    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        id_name = self.id_list[index]
        if index >= len(self.id_list) - 10:
            frame_idx = random.randint(1, len(self.paths[id_name])//2 - 1)
        else:
            frame_idx = random.randint(1, len(self.paths[id_name]) - 1)

        if self.type == "texture":
            f0_path = os.path.join(self.gt_folder, id_name, "Textures_512", "000001", "face.png")
            #f0_path = os.path.join(self.paths[id_name][frame_idx_0], "face.png")
            img_bytes = self.file_client.get(f0_path, 'gt')
            img_f0 = imfrombytes(img_bytes, float32=True)
            fn_path = os.path.join(self.paths[id_name][frame_idx], "face.png")
            frame_number = os.path.basename(self.paths[id_name][frame_idx])
            img_bytes = self.file_client.get(fn_path, 'gt')
            img_fn = imfrombytes(img_bytes, float32=True)
            if self.opt['phase'] == 'train':
                img_f0, img_fn = augment([img_f0, img_fn], self.opt['use_hflip'], rotation=False)
            img_f0, img_fn = img2tensor([img_f0, img_fn], bgr2rgb=True, float32=True)
            img_gt = th.sigmoid(img_fn / (img_f0+1e-8))
            img_gt = (img_gt - 0.5) * 2
            
        elif self.type == "mesh":
            frame_number = os.path.basename(self.paths[id_name][frame_idx])
            diff_path = os.path.join(self.gt_folder, id_name, "Models", frame_number, "diff.npy")
            diff_map = np.load(diff_path)
            diff_map = tensor2img(th.from_numpy(diff_map).squeeze(0).permute(2, 0, 1))
        
            if self.opt['phase'] == 'train':
                diff_map = augment([diff_map], self.opt['use_hflip'], rotation=False)

            # BGR to RGB, HWC to CHW, numpy to tensor
            diff_map = img2tensor([diff_map], bgr2rgb=True, float32=True)
            img_gt = diff_map[0] / 255
            img_f0 = 0
        
        return {'gt': img_gt, 'img_f0': img_f0, 'id': id_name, 'frame': frame_number}

    def __len__(self):
        return len(self.id_list)
    
    @staticmethod
    def get_frames(rt):
        subdirs = os.listdir(rt)
        subdirs = [os.path.join(rt,sd) for sd in subdirs]
        subdirs = sorted([sd for sd in subdirs if os.path.isdir(sd) and os.path.basename(sd).startswith("0")])
        return subdirs
