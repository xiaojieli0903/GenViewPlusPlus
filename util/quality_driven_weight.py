import math
import torch
import open_clip
import numpy as np
import torch.nn.functional as F

from PIL import Image
from typing import Dict, List, Optional, Union, Tuple


class quality_driven_module():
    def __init__(self, 
                 clip_model_name: str = 'convnext_base_w',
                 distance_type: str = 'cosine',
                 vis_input: bool = False,
                 standard_array_path: str = 'data/pca_results/convnext_base_w-laion2b-s13b-b82k-augreg/eigenvecters/pca_vectors.npy'
                 ):
        self.clip_model_name = clip_model_name
        self.distance_type = distance_type
        self.vis_input = vis_input
        self.mask_name = 'clipmask_' + self.clip_model_name
        pretrained = 'laion2B-s13B-b82K-augreg'
        self.patch_size = 32
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.clip_model_name,
            pretrained=pretrained,
            device='cuda')
        self.clip_model = model
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.standard_array = torch.Tensor(np.load(
                standard_array_path).reshape(
                -1)).cuda()
        print(f'Load standard_array done, shape = {self.standard_array.shape}')
        self.patch_number = 224 // self.patch_size
        
        
    def make_foreground_softmask(self,
                                tokens,
                                grid_size: Tuple[int, int] = (16, 16),
                                output_size: Tuple[int, int] = (16, 16),
                                view_index: Optional[int] = None):
        """Generate foreground mask.

        Args:
            tokens (torch.Tensor): Input tensor of shape (bs, 16*16, 768).
            grid_size (Tuple[int, int], optional): Grid size of the input tensor. Default is (16, 16).
            output_size (Tuple[int, int], optional): Size of the output mask. Default is (16, 16).
            view_index (int, optional): Optional view index for visualization. Default is None.

        Returns:
            torch.Tensor: Output foreground mask of shape (bs, 1, output_size[0], output_size[1]).
        """
        # Reshape tokens to (bs * 16*16, 768)
        projection = (tokens.reshape(-1, tokens.shape[-1]
                                        ) @ self.standard_array.type_as(
            tokens)).reshape(-1, 1, *grid_size)
         
        map_fg = projection
        bs, channel = map_fg.shape[0], map_fg.shape[1]

        # Calculate min and max values for both map_fg and map_bg
        max_fg = map_fg.view(bs, channel, -1).max(dim=-1,
                                                    keepdim=True).values.view(
            bs, channel, 1, 1)
        min_fg = map_fg.view(bs, channel, -1).min(dim=-1,
                                                    keepdim=True).values.view(
            bs, channel, 1, 1)

        # Normalize map_fg and map_bg
        map_fg = (map_fg - min_fg) / (max_fg - min_fg + 1e-7)
        map_bg = 1 - map_fg

        if view_index is not None:
            for idx in range(map_fg.shape[0]):
                mask_resized = map_fg[idx].reshape(*output_size)
                mask_array = mask_resized.cpu().numpy()
                # Save visualized foreground masks
                mask_img = Image.fromarray(
                    (mask_array * 255).astype(np.uint8))
                img_name = f'examples/{self.mask_name}/{idx}_view{view_index}_fg_maps_{output_size[0]}x{output_size[1]}.jpeg'
                if not os.path.exists(os.path.dirname(img_name)):
                    print(f'Make dirs: {os.path.dirname(img_name)}')
                    os.makedirs(os.path.dirname(img_name))
                mask_img.save(img_name)
                mask_resized = map_bg[idx].reshape(*output_size)
                mask_array = mask_resized.cpu().numpy()
                # Save visualized foreground masks
                mask_img = Image.fromarray(
                    (mask_array * 255).astype(np.uint8))
                img_name = f'examples/{self.mask_name}/{idx}_view{view_index}_bg_maps_{output_size[0]}x{output_size[1]}.jpeg'
                mask_img.save(img_name)

        map_fg = map_fg / (
                torch.sum(map_fg, dim=[2, 3]).view(bs, channel, 1,
                                                    1) + 1e-7)
        map_bg = map_bg / (
                torch.sum(map_bg, dim=[2, 3]).view(bs, channel, 1,
                                                    1) + 1e-7)

        return [map_fg, map_bg]


    def calculate_weight(self, sim_foreground, sim_background, gamma=1.0, mask=None):
        score = sim_foreground - sim_background
        score = torch.exp(gamma * score)
        if mask is not None:
            score = score * mask
        nonzero_mean = score[score!=0].mean()
        loss_weight = score / nonzero_mean
        return score.detach(), loss_weight.detach()


    def aggregate_feature(self, masks, tokens):
        features = torch.einsum('bchw, bkhw->bk', masks,
                                tokens.permute(0, 2, 1).reshape(
                                    tokens.shape[0],
                                    tokens.shape[-1],
                                    self.patch_number,
                                    self.patch_number))
        return features


    def calculate_sim(self, feature_1, feature_2, pairwise=False):
        if self.distance_type == 'cosine':
            feature_1 = F.normalize(feature_1, dim=-1)
            feature_2 = F.normalize(feature_2, dim=-1)
        if pairwise:
            sim_features = torch.matmul(feature_1, feature_2.transpose(0, 1))   # len(sim_features.shape)=2
        else:
            sim_features = torch.sum(feature_1 * feature_2, dim=-1) # len(sim_features.shape)=1
        return sim_features
    
    
    def compute_pairwise_weights(self, QD_img_img, QD_img_txt, views_list, labels, **kwargs):   
        if (not QD_img_img) and (not QD_img_txt):
            return {"loss_weight":None, "loss_weight_img_txt":None}
        
        n_views = len(views_list)
        bsz = views_list[0].shape[0]

        image = kwargs['image']     # [bsz, channels, W, H]
        text = kwargs['text']       # [bsz, dim]
        t_labels = kwargs['text_labels']
        gamma = kwargs['gamma']
                
        with torch.no_grad(), torch.cuda.amp.autocast():
            if QD_img_txt:
                outputs_list = [self.clip_model.encode_image(view, output_features_results=True) for view in views_list]
                views_tokens = [output[0][1] for output in outputs_list]     # n_views * [bsz, grid_size=49, dim=1024]
                views_embs = [output[1] for output in outputs_list]    # n_views * [bsz, dim=640]
                img_token = self.clip_model.encode_image(image, output_features=True)[1]
                txt_emb = self.clip_model.encode_text(text)
            else:
                views_tokens = [self.clip_model.encode_image(view, output_features=True)[1] for view in views_list]
        
        masks_list = [self.make_foreground_softmask(
            tokens,
            grid_size=(self.patch_number, self.patch_number),
            output_size=(self.patch_number, self.patch_number),
            view_index=i+1 if self.vis_input else None
        ) for i, tokens in enumerate(views_tokens)]
        
        z_bg_list = [self.aggregate_feature(m_bg, tokens) for (_, m_bg), tokens in zip(masks_list, views_tokens)]
        z_bg_tensor = torch.cat(z_bg_list, dim=0)   # [n_views * bsz,dim]
        
        # quality driven loss weight for image-image pairs
        loss_weight = None
        if QD_img_img:
            z_fg_list = [self.aggregate_feature(m_fg, tokens) for (m_fg, _), tokens in zip(masks_list, views_tokens)]
            z_fg_tensor = torch.cat(z_fg_list, dim=0)
            
            mask = torch.eq(labels.view(-1, 1), labels.contiguous().view(1, -1)).float()
            mask = mask.fill_diagonal_(0)
            
            sim_fg = self.calculate_sim(z_fg_tensor, z_fg_tensor, pairwise=True)    # [n_views * bsz, n_views * bsz]
            sim_bg = self.calculate_sim(z_bg_tensor, z_bg_tensor, pairwise=True)
            score, loss_weight = self.calculate_weight(sim_fg, sim_bg, gamma['gamma_ii'], mask)

        # quality driven loss weight for image-text pairs
        loss_weight_img_txt = None
        if QD_img_txt:
            # similarity between image and text
            views_embs_tensor = torch.cat(views_embs, dim=0)
            sim_img_txt = self.calculate_sim(views_embs_tensor, txt_emb, pairwise=True)
            
            # similarity of background between views and real image
            img_mask = self.make_foreground_softmask(
                img_token,
                grid_size=(self.patch_number, self.patch_number),
                output_size=(self.patch_number, self.patch_number),
                view_index=1 if self.vis_input else None
            )
            img_bg = self.aggregate_feature(img_mask[1], img_token)
            sim_bg_img_views = self.calculate_sim(z_bg_tensor, img_bg, pairwise=True)
            
            # loss weight matrix
            v_label_matrix = torch.eq(labels.view(-1, 1), t_labels.contiguous().view(1, -1)).float()
            score_img_txt, loss_weight_img_txt = self.calculate_weight(sim_img_txt, sim_bg_img_views, gamma['gamma_it'], v_label_matrix)
        
        return {"loss_weight":loss_weight, "loss_weight_img_txt":loss_weight_img_txt}
