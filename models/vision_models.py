import torch
import torch.nn as nn
import timm

from collections import OrderedDict


class VisionEncoder(nn.Module):
    def __init__(self,
                 # vision
                 vision_width: int,
                 vision_model: nn.Module,
                 # ssl
                 ssl_mlp_dim: int,
                 ssl_emb_dim: int,
                 **kwargs,
                 ):
        super().__init__()

        self.vision_width = vision_width
        self.visual = vision_model

        self.image_mlp = nn.Sequential(OrderedDict([
            ("layer1", nn.Linear(vision_width, ssl_mlp_dim)),
            ("bn1", nn.SyncBatchNorm(ssl_mlp_dim)),
            ("relu1", nn.ReLU(inplace=True)),
            ("layer2", nn.Linear(ssl_mlp_dim, ssl_mlp_dim)),
            ("bn2", nn.SyncBatchNorm(ssl_mlp_dim)),
            ("relu2", nn.ReLU(inplace=True)),
            ("layer3", nn.Linear(ssl_mlp_dim, ssl_emb_dim)),
        ]))

    def forward(self, img_cat, label, *args, **kwargs):
        # img_cat:[bsz, n_views*3,H,W]
        n_caption = img_cat.size(0)
        augs = torch.split(img_cat, 3, dim=1)   # tuple: n_views * tensor[bsz,3,H,W]
        n_views = len(augs)

        # do separate BN
        images = torch.cat(augs, dim=0) # [n_views * bsz,3,H,W]
        feats = self.visual(images)     # [n_views * bsz,D1]
        feats = torch.split(feats, n_caption, dim=0)    # tuple: n_views * tensor[bsz,D1]
        res = [self.image_mlp(feat) for feat in feats]  # list: n_views * [bsz,D2]
        h = torch.cat(res, dim=0)                       # [n_views * bsz,D2]

        # # can do a global BN instead of separate BN
        # images = torch.cat(augs, dim=0)
        # h = self.visual(images)
        # h = self.image_mlp(h)

        label = label.view(-1, 1)   # [bsz] -> [bsz,1]
        label_expand = label.repeat(n_views, 1).squeeze()   # [n_views * bsz], e.g. [0,1,2,0,1,2] when bsz=3 and n_views=2

        return {'feats': h,                 # [n_views * bsz,D2]    h[:bsz, :]是所有样本的第一个视图
                'labels': label_expand}     # [n_views * bsz]


def model_vision_small_patch16(ssl_mlp_dim, ssl_emb_dim, **kwargs):
    vision_model = timm.create_model('vit_small_patch16_224', num_classes=0)
    model = VisionEncoder(vision_width=384, vision_model=vision_model,
                          ssl_mlp_dim=ssl_mlp_dim, ssl_emb_dim=ssl_emb_dim, **kwargs)

    return model


def model_base_patch16(ssl_mlp_dim, ssl_emb_dim, **kwargs):
    vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
    model = VisionEncoder(vision_width=768, vision_model=vision_model,
                          ssl_mlp_dim=ssl_mlp_dim, ssl_emb_dim=ssl_emb_dim, **kwargs)

    return model


def model_vision_base_patch32(ssl_mlp_dim, ssl_emb_dim, **kwargs):
    vision_model = timm.create_model('vit_base_patch32_224', num_classes=0)
    model = VisionEncoder(vision_width=768, vision_model=vision_model,
                          ssl_mlp_dim=ssl_mlp_dim, ssl_emb_dim=ssl_emb_dim, **kwargs)

    return model


def model_vision_large_patch16(ssl_mlp_dim, ssl_emb_dim, **kwargs):
    vision_model = timm.create_model('vit_large_patch16_224', num_classes=0)
    model = VisionEncoder(vision_width=1024, vision_model=vision_model,
                          ssl_mlp_dim=ssl_mlp_dim, ssl_emb_dim=ssl_emb_dim, **kwargs)

    return model


model_dict = {
    'small': model_vision_small_patch16,
    'base': model_vision_base_patch16,
    'base_p32': model_vision_base_patch32,
    'large': model_vision_large_patch16,
}
