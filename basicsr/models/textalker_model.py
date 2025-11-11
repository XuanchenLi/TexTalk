import numpy as np
import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
import cv2

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
import torch.nn.functional as F
from .base_model import BaseModel



def recover_map(img_0, w_map):
    w_map = w_map / 2 + 0.5
    res = torch.log((w_map + 1e-8)/(1-w_map + 1e-8)) * img_0
    #res = -torch.log((1 / (w_map + 1e-8)) - 1) * img_0
    #res = torch.logit(w_map, eps=1e-6) * img_0
    assert not torch.any(torch.isnan(res))
    return res


def truncate_latent_and_audio(audio, motion_latent, tex_latent, n_motions, audio_unit=16000/30, pad_mode='zero'):
    def _truncate_audio(audio, end_idx, pad_mode='zero'):
        #return audio, audio.sum()
        batch_size = audio.shape[0]
        audio_trunc = audio.clone()
        if pad_mode == 'replicate':
            for i in range(batch_size):
                # unused_sum += audio.sum()
                # unused_sum += audio_trunc[i, end_idx[i]:].sum()
                audio_trunc[i, end_idx[i]:] = audio_trunc[i, end_idx[i] - 1]
        elif pad_mode == 'zero':

            for i in range(batch_size):
                # unused_sum += audio.sum()
                # unused_sum += audio_trunc[i, end_idx[i]:].sum()
                audio_trunc[i, end_idx[i]:] = 0
        else:
            raise ValueError(f'Unknown pad mode {pad_mode}!')

        return audio_trunc
    
    def _truncate_latent(motion_latent, tex_latent, end_idx, pad_mode='zero'):
        batch_size = motion_latent.shape[0]
        #coef_dict_trunc = {k: v.clone() for k, v in coef_dict.items()}
        motion_latent_trunc = motion_latent.clone()
        tex_latent_trunc = tex_latent.clone()
        if pad_mode == 'replicate':
        
            for i in range(batch_size):
                # unused_sum += motion_latent[i, end_idx[i]:].sum() + tex_latent[i, end_idx[i]:].sum()
                motion_latent_trunc[i, end_idx[i]:] = motion_latent_trunc[i, end_idx[i] - 1]
                tex_latent_trunc[i, end_idx[i]:] = tex_latent_trunc[i, end_idx[i] - 1]
                
        elif pad_mode == 'zero':
           
            for i in range(batch_size):
                #unused_sum += motion_latent[i, end_idx[i]:].sum() + tex_latent[i, end_idx[i]:].sum()
                motion_latent_trunc[i, end_idx[i]:] = 0
                tex_latent_trunc[i, end_idx[i]:] = 0
        else:
            raise ValueError(f'Unknown pad mode: {pad_mode}!')

        return motion_latent_trunc, tex_latent_trunc
    
    batch_size = audio.shape[0]
    end_idx = torch.randint(1, n_motions, (batch_size,), device=audio.device)
    audio_end_idx = torch.round(end_idx * audio_unit).long()
    # mask = torch.arange(n_motions, device=audio.device).expand(batch_size, -1) < end_idx.unsqueeze(1)

    # truncate audio
    audio_trunc= _truncate_audio(audio, audio_end_idx, pad_mode=pad_mode)

    # prepare coef dict and stats
    #coef_dict = {'exp': motion_latent[..., :50], 'pose_any': motion_coef[..., 50:]}

    # truncate coef dict
    motion_latent_trunc, tex_latent_trunc = _truncate_latent(motion_latent, tex_latent, end_idx, pad_mode=pad_mode)
    #motion_coef_trunc = torch.cat([coef_dict_trunc['exp'], coef_dict_trunc['pose_any']], dim=-1)

    return audio_trunc, motion_latent_trunc, tex_latent_trunc, end_idx


@MODEL_REGISTRY.register()
class TexTalkerModel(BaseModel):
    
    def __init__(self, opt):
        super(TexTalkerModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)
        self.n_motions = opt['network_g']["n_motions"]
        self.n_prev_motions = opt['network_g']["n_prev_motions"]
        self.use_indicator = opt['network_g']["use_indicator"]
        self.trunc_prob1 = opt["trunc_prob1"]
        self.trunc_prob2 = opt["trunc_prob2"]
        self.gradient_accumulation_steps = opt["gradient_accumulation_steps"]
        self.use_context_audio_feat = True
        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()
        
    
    
    def feed_data(self, data):
        self.audio_pair = [f.to(self.device) for f in data["audio_pair"]]
        self.motion_latent_pair = [f.to(self.device) for f in data["motion_latent_pair"]]
        self.tex_latent_pair = [f.to(self.device) for f in data["tex_latent_pair"]]
        self.shape_map = data["shape_map"].to(self.device)
        self.tex_map = data["tex_map"].to(self.device)
        self.audio_stat = data["audio_stat"]
        self.style_code = data["style_code"].to(self.device)
        #self.img_f0 = data['img_f0'].to(self.device)
        #self.img_fn = data['img_fn'].to(self.device)
        self.b = self.motion_latent_pair[0].shape[0]


    def init_training_settings(self):
        logger = get_root_logger()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()
        #self.net_g.enable_gradient_checkpointing(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.net_g.train()

        # define losses
        if train_opt.get('sample_opt'):
            self.cri_sample = build_loss(train_opt['sample_opt']).to(self.device)
        else:
            self.cri_sample = None   

        self.net_g_start_iter = train_opt.get('net_g_start_iter', 0)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()


    def setup_optimizers(self):
        train_opt = self.opt['train']
        # optimizer g
        optim_params_g = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params_g.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params_g, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)


    def optimize_parameters(self, current_iter):
        logger = get_root_logger()
        loss_dict = OrderedDict()

        # optimize net_g
        #self.optimizer_g.zero_grad()
        if self.use_context_audio_feat:
            audio_feat = self.net_g.module.extract_audio_feature(torch.cat(self.audio_pair, dim=1), self.n_motions * 2)  # (N, 2L, :)
        l_g_sample = 0
        #unused_sum = 0
        unused_sum2 = 0
        for i in range(2):
            unused_sum2i = 0
            audio = self.audio_pair[i]  # (N, L_a)
            motion_latent = self.motion_latent_pair[i]  # (N, L, 50+x)
            #print(motion_latent.shape)
            tex_latent = self.tex_latent_pair[i]
            style = self.style_code
            #batch_size = audio.shape[0]

            # truncate input audio and motion according to trunc_prob
            if (i == 0 and np.random.rand() < self.trunc_prob1) or (i != 0 and np.random.rand() < self.trunc_prob2):
                audio_in, motion_latent_in, tex_latent_in, end_idx = truncate_latent_and_audio(
                    audio, motion_latent, tex_latent, self.n_motions, 16000/30, "zero")
                if i != 0 and self.use_context_audio_feat:
                    # use contextualized audio feature for the second clip
                    audio_in = self.net_g.module.extract_audio_feature(torch.cat([self.audio_pair[i - 1], audio_in], dim=1),
                                                           self.n_motions * 2)[:, -self.n_motions:]

            else:
                if self.use_context_audio_feat:
                    audio_in = audio_feat[:, i * self.n_motions : (i + 1) * self.n_motions]
                else:
                    audio_in = audio
                motion_latent_in, tex_latent_in, end_idx = motion_latent, tex_latent, None

            if self.use_indicator:
                if end_idx is not None:
                    indicator = torch.arange(self.n_motions, device=self.device).expand(self.b, -1) < end_idx.unsqueeze(
                        1)
                else:
                    indicator = torch.ones(self.b, self.n_motions, device=self.device)
            else:
                indicator = None
        
            if i == 0:
                noise, target, prev_motion_latent, prev_tex_latent, prev_audio_feat, unused_sum = self.net_g(
                    motion_latent_in, tex_latent_in, audio_in, self.shape_map, self.tex_map, style, indicator=indicator)
                if end_idx is not None:  # was truncated, needs to use the complete feature
                    prev_motion_latent = motion_latent[:, -self.n_prev_motions:]
                    prev_tex_latent = tex_latent[:, -self.n_prev_motions:]
                    #prev_audio_feat = audio_feat[:, self.n_motions - self.n_prev_motions:self.n_motions].detach()
                    if self.use_context_audio_feat:
                        prev_audio_feat = audio_feat[:, self.n_motions - self.n_prev_motions:self.n_motions].detach()
                    else:
                        with torch.no_grad():
                            prev_audio_feat = self.net_g.module.extract_audio_feature(audio)[:, -self.n_prev_motions:]
                else:
                    prev_motion_latent = prev_motion_latent[:, -self.n_prev_motions:]
                    prev_tex_latent = prev_tex_latent[:, -self.n_prev_motions:]
                    prev_audio_feat = prev_audio_feat[:, -self.n_prev_motions:]
            else:
                noise, target, _, _, _, unused_sum = self.net_g(motion_latent_in, tex_latent_in, audio_in, 
                                                    self.shape_map, self.tex_map, style,
                                                    prev_motion_latent, prev_tex_latent, prev_audio_feat, 
                                                    indicator=indicator)
            if self.cri_sample:
                if end_idx is None:
                    mask = torch.ones((target.shape[0], self.n_motions), dtype=torch.bool, device=target.device)
                else:
                    mask = torch.arange(self.n_motions, device=target.device).expand(target.shape[0], -1) < end_idx.unsqueeze(1)
                if i != 0:
                    mask = torch.cat([torch.ones_like(mask[:, :self.n_prev_motions]), mask], dim=1)
                if i == 0:
                    target = target[:, self.n_prev_motions:]
                    latent_gt = torch.cat([motion_latent_in, tex_latent_in], dim=2)
                else:
                    motion_latent_gt = torch.cat([prev_motion_latent, motion_latent_in], dim=1)
                    tex_latent_gt = torch.cat([prev_tex_latent, tex_latent_in], dim=1)
                    latent_gt = torch.cat([motion_latent_gt, tex_latent_gt], dim=2)
                    

                loss_sample = self.cri_sample(latent_gt, target, reduction='none')
                l_g_sample += loss_sample[mask].mean()
                #unused_sum2 += unused_sum2i
            
        
        l_g_total = 0
        if self.cri_sample:
            #l_g_sample = self.cri_pix(self.output, self.gt)
            l_g_total += l_g_sample
            loss_dict['l_g_sample'] = l_g_sample
            
            #if not torch.any(torch.isnan(l_g_total)):
        l_g_total += unused_sum * 0.0
        l_g_total.backward()
        if current_iter % self.gradient_accumulation_steps == 0:
            self.optimizer_g.step()
            self.optimizer_g.zero_grad()
        #self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)


    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)


    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        pass


    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        pass


    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['gt'] = self.gt.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        #self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)
