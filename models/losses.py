import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.nn

from util import misc


def compute_cross_entropy(p, q, loss_weight=None):
    q = F.log_softmax(q, dim=-1)
    loss = p * q * loss_weight if loss_weight is not None else p * q
    loss = torch.sum(loss, dim=-1)
    return - loss.mean()


def stablize_logits(logits):
    logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
    logits = logits - logits_max.detach()
    return logits


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def get_global_padded_weight(local_weight, local_size_dim0, local_size_dim1, global_size_dim1, rank, device):
    padded_weight = torch.zeros((local_size_dim0, global_size_dim1), device=device)
    start = rank * local_size_dim1
    end = start + local_size_dim1
    padded_weight[:, start:end] = local_weight
    return padded_weight


class MultiPosConLoss(nn.Module):
    """
    Multi-Positive Contrastive Loss: https://arxiv.org/pdf/2306.00984.pdf
    """

    def __init__(self, temperature=0.1):
        super(MultiPosConLoss, self).__init__()
        self.temperature = temperature
        self.logits_mask = None
        self.mask = None
        self.last_local_batch_size = None
                        
    def set_temperature(self, temp=0.1):
        self.temperature = temp

    def forward(self, outputs, coefs, **kwargs):
        feats = outputs['feats']    # feats shape: [n_views * bsz, D]
        labels = outputs['labels']    # labels shape: [n_views * bsz]
              
        device = (torch.device('cuda')
                  if feats.is_cuda
                  else torch.device('cpu'))

        feats = F.normalize(feats, dim=-1, p=2)
        local_batch_size = feats.size(0)    # n_views * bsz

        all_feats = torch.cat(torch.distributed.nn.all_gather(feats), dim=0)  # no gradient gather [n_gpus * n_views * bsz, D]
        all_labels = concat_all_gather(labels)  # no gradient gather [n_gpus * n_views * bsz]

        # compute the mask based on labels
        if local_batch_size != self.last_local_batch_size:
            mask = torch.eq(labels.view(-1, 1),
                            all_labels.contiguous().view(1, -1)).float().to(device)
            self.logits_mask = torch.scatter(
                torch.ones_like(mask),
                1,
                torch.arange(mask.shape[0]).view(-1, 1).to(device) +
                local_batch_size * misc.get_rank(),
                0
            )

            self.last_local_batch_size = local_batch_size
            self.mask = mask * self.logits_mask
            
        mask = self.mask

        # compute logits
        logits = torch.matmul(feats, all_feats.T) / self.temperature    # [n_views * bsz, n_gpus * n_views * bsz]
        logits = logits - (1 - self.logits_mask) * 1e9      # Avoid self-comparasion

        # optional: minus the largest logit to stablize logits
        logits = stablize_logits(logits)
        
        p = mask / mask.sum(1, keepdim=True).clamp(min=1.0)
        
        padded_weight = None
        if coefs[1]:
            loss_weight = kwargs.get('loss_weight', None)
            assert loss_weight is not None
            global_batch_size = all_feats.size(0)
            rank = misc.get_rank()
            padded_weight = get_global_padded_weight(
                local_weight=loss_weight, 
                local_size_dim0=local_batch_size, 
                local_size_dim1=local_batch_size, 
                global_size_dim1=global_batch_size, 
                rank=rank, 
                device=device)

        original_image_loss = compute_cross_entropy(p, logits, None) if coefs[0] else 0
        QD_image_loss = compute_cross_entropy(p, logits, padded_weight) if coefs[1] else 0
        loss = coefs[0] * original_image_loss + coefs[1] * QD_image_loss
        
        return {'loss': loss, 'original_image_loss': original_image_loss, 'QD_image_loss':QD_image_loss}


class MultiPosConLossMM(nn.Module):
    """Multi-positive contrastive loss, when multiple images corresponds to the same texts"""
    def __init__(self, temperature=0.1):
        super(MultiPosConLossMM, self).__init__()
        self.temperature = temperature
        self.last_local_batch_size = None
        self.v_label_matrix = None
        self.t_label_matrix = None
        self.mask = None
        self.logits_mask = None

    def forward(self, outputs, coefs, **kwargs):
        feats = outputs['image_feats']
        
        v_feats = outputs['image_emb']
        t_feats = outputs['text_emb']
        
        v_labels = outputs['image_labels']
        t_labels = outputs['text_labels']
        
        logit_scale = outputs['logit_scale']
        
        device = (torch.device('cuda')
                  if v_feats.is_cuda
                  else torch.device('cpu'))

        v_feats = F.normalize(v_feats, dim=-1, p=2)
        t_feats = F.normalize(t_feats, dim=-1, p=2)

        v_local_batch_size = v_feats.size(0)
        t_local_batch_size = t_feats.size(0)

        all_v_feats = torch.cat(torch.distributed.nn.all_gather(v_feats), dim=0)
        all_t_feats = torch.cat(torch.distributed.nn.all_gather(t_feats), dim=0)

        # compute the logits for image-text contrasting
        logits_v = logit_scale * torch.matmul(v_feats, all_t_feats.T)   # [n_views*bsz, bsz]
        logits_t = logit_scale * torch.matmul(t_feats, all_v_feats.T)   # [bsz, n_views*bsz]

        # compute the logits for image-only contrasting
        feats = F.normalize(feats, dim=-1, p=2)
        all_feats = torch.cat(torch.distributed.nn.all_gather(feats), dim=0)
        logits = torch.matmul(feats, all_feats.T) / self.temperature

        # Create label matrix, since in our specific case the
        # label matrix in side each batch is the same, so
        # we can just create it once and reuse it. For other
        # cases, user need to compute it for each batch
        if v_local_batch_size != self.last_local_batch_size:
            all_v_labels = concat_all_gather(v_labels)
            all_t_labels = concat_all_gather(t_labels)
            all_v_labels = all_v_labels.contiguous().view(1, -1)
            all_t_labels = all_t_labels.contiguous().view(1, -1)

            # mask matrix for image-text contrastive loss
            self.v_label_matrix = torch.eq(v_labels.view(-1, 1),
                                           all_t_labels).float().to(device)
            self.t_label_matrix = torch.eq(t_labels.view(-1, 1),
                                           all_v_labels).float().to(device)

            # mask matrix for image supervised contrastive loss
            self.mask = torch.eq(v_labels.view(-1, 1), all_v_labels).float().to(device)
            self.logits_mask = torch.scatter(
                torch.ones_like(self.mask),
                1,
                torch.arange(self.mask.shape[0]).view(-1, 1).to(device) +
                v_local_batch_size * misc.get_rank(),
                0
            )
            self.mask = self.mask * self.logits_mask

            self.last_local_batch_size = v_local_batch_size

        # image only loss
        mask = self.mask
        p = mask / mask.sum(1, keepdim=True).clamp(min=1.0)
            
        logits = logits - (1 - self.logits_mask) * 1e9
        logits = stablize_logits(logits)
        
        padded_weight = None
        if coefs[1]:
            loss_weight = kwargs.get('loss_weight', None)
            assert loss_weight is not None
            global_batch_size = all_feats.size(0)
            rank = misc.get_rank()
            padded_weight = get_global_padded_weight(
                local_weight=loss_weight, 
                local_size_dim0=v_local_batch_size, 
                local_size_dim1=v_local_batch_size, 
                global_size_dim1=global_batch_size, 
                rank=rank, 
                device=device)
            
        original_img_loss = compute_cross_entropy(p, logits, None) if coefs[0] else 0
        QD_img_loss = compute_cross_entropy(p, logits, padded_weight) if coefs[1] else 0
        
        # image text loss
        v_mask = self.v_label_matrix    # [n_views*bsz, bsz]
        p_v = v_mask / v_mask.sum(1, keepdim=True).clamp(min=1.0)
        t_mask = self.t_label_matrix    # [bsz, n_views*bsz]
        p_t = t_mask / t_mask.sum(1, keepdim=True).clamp(min=1.0)
        
        v_padded_weigh, t_padded_weight = None, None
        if coefs[3]:
            loss_weight_img_txt = kwargs.get('loss_weight_img_txt', None)
            assert loss_weight_img_txt is not None
            v_padded_weight = get_global_padded_weight(
                local_weight=loss_weight_img_txt, 
                local_size_dim0=v_local_batch_size, 
                local_size_dim1=t_local_batch_size, 
                global_size_dim1=all_t_feats.size(0), 
                rank=rank, 
                device=device)
            t_padded_weight = get_global_padded_weight(
                local_weight=loss_weight_img_txt.transpose(0, 1), 
                local_size_dim0=t_local_batch_size, 
                local_size_dim1=v_local_batch_size, 
                global_size_dim1=all_v_feats.size(0), 
                rank=rank, 
                device=device)
            
        original_img_txt_loss = (compute_cross_entropy(p_v, logits_v, None) + \
                        compute_cross_entropy(p_t, logits_t, None)) / 2 if coefs[2] else 0
        QD_img_txt_loss = (compute_cross_entropy(p_v, logits_v, v_padded_weight) + \
                        compute_cross_entropy(p_t, logits_t, t_padded_weight)) / 2 if coefs[3] else 0
        
        # total loss
        loss = coefs[0] * original_img_loss + coefs[1] * QD_img_loss + coefs[2] * original_img_txt_loss + coefs[3] * QD_img_txt_loss
        
        return {'loss': loss,
                'original_img_loss': original_img_loss,
                'QD_img_loss': QD_img_loss,
                'original_img_txt_loss': original_img_txt_loss,
                'QD_img_txt_loss': QD_img_txt_loss}


class ImgTxtConLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(ImgTxtConLoss, self).__init__()
        self.temperature = temperature
        self.last_local_batch_size = None
        self.v_label_matrix = None
        self.t_label_matrix = None
        self.mask = None
        self.logits_mask = None

    def forward(self, outputs, coefs, **kwargs):
        v_feats = outputs['image_emb']
        t_feats = outputs['text_emb']
        v_labels = outputs['image_labels']
        t_labels = outputs['text_labels']
        logit_scale = outputs['logit_scale']
        device = (torch.device('cuda')
                  if v_feats.is_cuda
                  else torch.device('cpu'))

        v_feats = F.normalize(v_feats, dim=-1, p=2)
        t_feats = F.normalize(t_feats, dim=-1, p=2)

        v_local_batch_size = v_feats.size(0)
        t_local_batch_size = t_feats.size(0)

        all_v_feats = torch.cat(torch.distributed.nn.all_gather(v_feats), dim=0)
        all_t_feats = torch.cat(torch.distributed.nn.all_gather(t_feats), dim=0)

        # compute the logits for image-text contrasting
        logits_v = logit_scale * torch.matmul(v_feats, all_t_feats.T)   # [n_views*bsz, bsz]
        logits_t = logit_scale * torch.matmul(t_feats, all_v_feats.T)   # [bsz, n_views*bsz]

        # Create label matrix, since in our specific case the
        # label matrix in side each batch is the same, so
        # we can just create it once and reuse it. For other
        # cases, user need to compute it for each batch
        if v_local_batch_size != self.last_local_batch_size:
            all_v_labels = concat_all_gather(v_labels)
            all_t_labels = concat_all_gather(t_labels)
            all_v_labels = all_v_labels.contiguous().view(1, -1)
            all_t_labels = all_t_labels.contiguous().view(1, -1)

            # mask matrix for image-text contrastive loss
            self.v_label_matrix = torch.eq(v_labels.view(-1, 1),
                                           all_t_labels).float().to(device)
            self.t_label_matrix = torch.eq(t_labels.view(-1, 1),
                                           all_v_labels).float().to(device)

            self.last_local_batch_size = v_local_batch_size

        # image text loss
        v_mask = self.v_label_matrix    # [n_views*bsz, bsz]
        p_v = v_mask / v_mask.sum(1, keepdim=True).clamp(min=1.0)
        t_mask = self.t_label_matrix    # [bsz, n_views*bsz]
        p_t = t_mask / t_mask.sum(1, keepdim=True).clamp(min=1.0)
        
        v_padded_weigh, t_padded_weight = None, None
        if coefs[3]:
            loss_weight_img_txt = kwargs.get('loss_weight_img_txt', None)
            assert loss_weight_img_txt is not None
            rank = misc.get_rank()
            v_padded_weight = get_global_padded_weight(
                local_weight=loss_weight_img_txt, 
                local_size_dim0=v_local_batch_size, 
                local_size_dim1=t_local_batch_size, 
                global_size_dim1=all_t_feats.size(0), 
                rank=rank, 
                device=device)
            t_padded_weight = get_global_padded_weight(
                local_weight=loss_weight_img_txt.transpose(0, 1), 
                local_size_dim0=t_local_batch_size, 
                local_size_dim1=v_local_batch_size, 
                global_size_dim1=all_v_feats.size(0), 
                rank=rank, 
                device=device)
            
        original_img_txt_loss = (compute_cross_entropy(p_v, logits_v, None) + \
                        compute_cross_entropy(p_t, logits_t, None)) / 2 if coefs[2] else 0
        QD_img_txt_loss = (compute_cross_entropy(p_v, logits_v, v_padded_weight) + \
                        compute_cross_entropy(p_t, logits_t, t_padded_weight)) / 2 if coefs[3] else 0
        
        # total loss
        loss = coefs[2] * original_img_txt_loss + coefs[3] * QD_img_txt_loss
        
        return {'loss': loss,
                'original_img_txt_loss': original_img_txt_loss,
                'QD_img_txt_loss': QD_img_txt_loss}

