import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class MultiCrossEntropyLoss(nn.Module):
    def __init__(self, focal=False, weight=None, reduce=True):
        super(MultiCrossEntropyLoss, self).__init__()
        self.focal = focal
        self.weight= weight
        self.reduce = reduce

    def forward(self, input, target):
        #IN: input: unregularized logits [B, C] target: multi-hot representaiton [B, C]
        target_sum = torch.sum(target, dim=1)
        target_div = torch.where(target_sum != 0, target_sum, torch.ones_like(target_sum)).unsqueeze(1)
        target = target/target_div
        logsoftmax = nn.LogSoftmax(dim=1).to(input.device)
        if not self.focal:
            if self.weight is None:
                output = torch.sum(-target * logsoftmax(input), 1)
            else:
                output = torch.sum(-target * logsoftmax(input) /self.weight, 1)
        else:
            softmax = nn.Softmax(dim=1).to(input.device)
            p = softmax(input)
            output = torch.sum(-target * (1 - p)**2 * logsoftmax(input), 1)
            
        if self.reduce:
            return torch.mean(output)
        else:
            return output
    

def cls_loss_func(y,output, use_focal=False, weight=None, reduce=True):
    input_size=y.size()
    y = y.float().cuda()
    if weight is not None:
        weight = weight.cuda()
    loss_func = MultiCrossEntropyLoss(focal=use_focal, weight=weight, reduce=reduce)
    
    y=y.reshape(-1,y.size(-1))
    output=output.reshape(-1,output.size(-1))
    loss = loss_func(output,y)
    
    if not reduce:
        loss = loss.reshape(input_size[:-1])
    
    return loss


def regress_loss_func(y,output):
    y = y.float().cuda()
    
    #y=y.unsqueeze(-1)
    y=y.reshape(-1,y.size(-1))
    output=output.reshape(-1,output.size(-1))
    
    bgmask= y[:,1] < -1e2
    
    fg_logits = output[~bgmask]
    bg_logits = output[bgmask]
    
    fg_target = y[~bgmask]
    bg_target = y[bgmask]
    
    loss = nn.functional.l1_loss(fg_logits,fg_target)
    #loss = nn.functional.smooth_l1_loss(fg_logits, fg_target, beta=0.5)
    
        
    if(loss.isnan()):
        return torch.tensor([0.0], requires_grad=True).cuda()
    return loss


def suppress_loss_func(y,output):
    y = y.float().cuda()

    #y=y.unsqueeze(-1)
    y=y.reshape(-1,y.size(-1))
    output=output.reshape(-1,output.size(-1))

    loss = nn.functional.binary_cross_entropy(output,y)

    return loss


def compute_cb_weights(action_frame_count, beta=0.999):
    """
    Class-balanced effective-number weights (Cui et al., CVPR 2019).
    action_frame_count: 1D tensor of per-class positive-frame counts (length = num_of_class,
    last entry is the background class).
    Returns a tensor of shape [num_of_class] suitable for multiplying per-class loss terms.
    Background slot is set to 1.0 (no reweight); only foreground classes are rebalanced.
    """
    counts = action_frame_count.float().clone()
    # avoid div-by-zero for any unseen class
    counts = torch.clamp(counts, min=1.0)
    eff_num = 1.0 - torch.pow(beta, counts)
    weights = (1.0 - beta) / eff_num
    # normalize foreground weights to mean 1 so the overall loss scale stays comparable
    fg = weights[:-1]
    fg = fg / fg.mean()
    weights[:-1] = fg
    weights[-1] = 1.0
    return weights


class ClassBalancedFocalLoss(nn.Module):
    """
    Multi-label focal loss with per-class balancing weights.
    Operates on logits with shape [N, C] and multi-hot targets [N, C].
    """
    def __init__(self, weights=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        if weights is not None:
            self.register_buffer('weights', weights)
        else:
            self.weights = None

    def forward(self, logits, target):
        # normalize multi-hot targets so each row sums to 1 (matches existing convention)
        target_sum = torch.sum(target, dim=1, keepdim=True)
        target_sum = torch.where(target_sum != 0, target_sum, torch.ones_like(target_sum))
        target = target / target_sum

        log_p = F.log_softmax(logits, dim=1)
        p = log_p.exp()
        focal = (1 - p) ** self.gamma
        per_class = -target * focal * log_p  # [N, C]

        if self.weights is not None:
            w = self.weights.to(logits.device).unsqueeze(0)  # [1, C]
            per_class = per_class * w

        return per_class.sum(dim=1).mean()


def cb_cls_loss_func(y, output, weights, gamma=2.0):
    """Wrapper mirroring cls_loss_func signature for the anchor classifier."""
    y = y.float().cuda().reshape(-1, y.size(-1))
    output = output.reshape(-1, output.size(-1))
    loss_fn = ClassBalancedFocalLoss(weights=weights, gamma=gamma)
    return loss_fn(output, y)


def diou_regress_loss_func(reg_target, reg_pred, anchors, l1_weight=0.2):
    """
    DIoU-based regression loss for temporal anchors.

    reg_target / reg_pred: [B, A, 2] where last dim is (offset_norm, log_length_ratio)
        following the convention in dataset.py:
            offset_norm   = (target_end - anchor_end) / anchor_length
            log_length_ratio = log(target_length / anchor_length)
    anchors: list/tensor of anchor scales (length A).

    Foreground mask: target[:, 1] >= -1e2 (background rows have v2[1] = -1e3).

    Returns a scalar loss = (1 - mean DIoU on foreground anchors)
                           + l1_weight * L1 on the same rows
    The small L1 term keeps gradients alive when DIoU saturates near 1 and stabilizes
    early training before predictions overlap any GT.
    """
    if reg_target.dim() != 3:
        return F.l1_loss(reg_pred, reg_target)
    A = reg_target.shape[1]

    device = reg_pred.device
    anchors_t = torch.as_tensor(anchors, dtype=torch.float32, device=device).view(1, A, 1)

    # Foreground mask per (batch, anchor) row
    fg_mask = reg_target[..., 1:2] > -1e2  # [B, A, 1]
    fg_flat = fg_mask.squeeze(-1)          # [B, A]
    if fg_flat.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Decode (offset, log_length) -> (start, end) in a frame-relative system.
    # The absolute anchor end position is the same for predicted and target (anchor_end),
    # so we can set anchor_end = 0 and only track relative positions.
    # length_pred = anchor * exp(log_r);  end_pred = 0 + anchor * offset
    # start_pred = end_pred - length_pred
    # Same for target.
    offset_p = reg_pred[..., 0:1]       # [B, A, 1]
    logr_p   = reg_pred[..., 1:2]
    offset_t = reg_target[..., 0:1]
    logr_t   = reg_target[..., 1:2]

    # Clamp logr to avoid exp overflow
    logr_p = torch.clamp(logr_p, min=-3.0, max=3.0)
    logr_t = torch.clamp(logr_t, min=-3.0, max=3.0)

    end_p   = anchors_t * offset_p
    len_p   = anchors_t * torch.exp(logr_p)
    start_p = end_p - len_p

    end_t   = anchors_t * offset_t
    len_t   = anchors_t * torch.exp(logr_t)
    start_t = end_t - len_t

    # IoU
    inter_start = torch.max(start_p, start_t)
    inter_end   = torch.min(end_p,   end_t)
    inter = torch.clamp(inter_end - inter_start, min=0.0)
    union = (len_p + len_t - inter).clamp(min=1e-6)
    iou = inter / union

    # Smallest enclosing segment
    enc_start = torch.min(start_p, start_t)
    enc_end   = torch.max(end_p,   end_t)
    enc_len   = (enc_end - enc_start).clamp(min=1e-6)

    # Center distance (squared, normalized by enclosing length squared)
    center_p = (start_p + end_p) / 2.0
    center_t = (start_t + end_t) / 2.0
    center_dist_sq = (center_p - center_t) ** 2
    diou = iou - center_dist_sq / (enc_len ** 2)

    diou_loss_per_row = (1.0 - diou).squeeze(-1)  # [B, A]
    diou_loss = diou_loss_per_row[fg_flat].mean()

    if torch.isnan(diou_loss):
        return torch.tensor(0.0, device=device, requires_grad=True)

    if l1_weight > 0:
        l1 = F.l1_loss(reg_pred[fg_flat], reg_target[fg_flat])
        return diou_loss + l1_weight * l1
    return diou_loss


def snip_loss_func(snip_label, snip_logits, weights=None):
    """
    Multi-label binary cross-entropy on the auxiliary snippet head.
    snip_label: [B, num_of_class] multi-hot (last dim = background).
    snip_logits: [B, num_of_class] raw logits.
    """
    snip_label = snip_label.float().cuda()
    # drop background slot - snippet head predicts foreground occurrence
    target = snip_label[:, :-1]
    logits = snip_logits[:, :-1]

    if weights is not None:
        w = weights[:-1].to(logits.device)
        pos_weight = w  # used as per-class positive weight in BCE
        loss = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=pos_weight
        )
    else:
        loss = F.binary_cross_entropy_with_logits(logits, target)
    return loss
