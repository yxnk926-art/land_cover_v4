# -*- coding: utf-8 -*-
"""
engine.py — 损失函数、评估指标、学习率调度等公共函数
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.numerical_diagnostics import NumericalValidationError


# ---------------------------------------------------------------- AMP 新旧 API 兼容
def autocast_ctx(device_type="cuda", enabled=True):
    """torch>=2.4 推荐 torch.amp.autocast(device_type)，旧版只有 torch.cuda.amp.autocast。"""
    try:
        return torch.amp.autocast(device_type, enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=enabled)


def make_grad_scaler(enabled=True):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


# ---------------------------------------------------------------- 损失
class DiceLoss(nn.Module):
    """多类 soft Dice（对应原仓库 BCEDiceLoss 中的 Dice 项），忽略 ignore_index 像素，
    并跳过背景通道（背景不参与监督）。"""

    def __init__(self, ignore_index=0, eps=1e-5):
        super().__init__()
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, logits, target):
        # logits: (B, C, H, W)  target: (B, H, W)
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)

        valid = (target != self.ignore_index).unsqueeze(1).float()  # (B,1,H,W)
        target_clamped = target.clamp(min=0, max=num_classes - 1)
        one_hot = F.one_hot(target_clamped, num_classes).permute(0, 3, 1, 2).float()

        probs = probs * valid
        one_hot = one_hot * valid

        dims = (0, 2, 3)
        inter = (probs * one_hot).sum(dims)
        cardinality = probs.sum(dims) + one_hot.sum(dims)

        # 只对真值中出现过的前景类别求平均，避免空类拉高分数
        keep = torch.ones(num_classes, dtype=torch.bool, device=logits.device)
        if 0 <= self.ignore_index < num_classes:
            keep[self.ignore_index] = False
        present = keep & (one_hot.sum(dims) > 0)
        if present.sum() == 0:
            return logits.sum() * 0.0
        dice = (2 * inter[present] + self.eps) / (cardinality[present] + self.eps)
        return 1.0 - dice.mean()


class SetCrossEntropyLoss(nn.Module):
    """Cross-entropy against a set of allowed classes: ``-log sum_{u in S(y)} p_u``.

    ``membership`` is a 0/1 matrix [L, C]: label y (one of L label values, e.g.
    the global native labels of the unified protocol) allows the model classes
    marked in row y.  With an identity matrix this is the ordinary
    cross-entropy.  Mean over non-ignored pixels, computed in FP32.
    """

    def __init__(self, membership, ignore_index=255):
        super().__init__()
        allowed = torch.as_tensor(membership).bool()
        if allowed.ndim != 2 or not allowed.any(1).all():
            raise ValueError("membership 必须是 [标签数, 类别数] 的 0/1 矩阵，且每行至少一个类别")
        self.register_buffer("allowed", allowed, persistent=False)
        self.ignore_index = ignore_index
        self.weight = None  # read like nn.CrossEntropyLoss.weight

    def forward(self, logits, target):
        if logits.shape[1] != self.allowed.shape[1]:
            raise ValueError(f"模型输出 {logits.shape[1]} 个类别，membership 有 "
                             f"{self.allowed.shape[1]} 个")
        valid = target != self.ignore_index
        if not valid.any():
            return logits.sum() * 0.0
        labels = target[valid]
        if labels.min() < 0 or labels.max() >= self.allowed.shape[0]:
            raise ValueError(f"标签超出 membership 的范围 [0, {self.allowed.shape[0]})")
        logits = logits.float().movedim(1, -1)[valid]          # [N, C]
        in_set = self.allowed[labels]                          # [N, C]
        log_set = torch.logsumexp(logits.masked_fill(~in_set, float("-inf")), dim=1)
        return (torch.logsumexp(logits, dim=1) - log_set).mean()


class SegLoss(nn.Module):
    """CE + Dice（对应原仓库 BCE+Dice），支持 4 尺度深监督。

    ``set_membership`` 换成集合交叉熵（统一细类协议），此时不支持 Dice 与类别权重。
    """

    def __init__(self, ignore_index=0, dice_weight=1.0,
                 deep_sup_weights=(1.0, 1.0, 1.0, 1.0), ce_class_weights=None,
                 set_membership=None):
        super().__init__()
        ce_weight = None
        if ce_class_weights is not None:
            ce_weight = torch.as_tensor(ce_class_weights, dtype=torch.float32)
            if ce_weight.ndim != 1:
                raise ValueError("CE_CLASS_WEIGHTS 必须是一维序列")
        if set_membership is None:
            self.ce = nn.CrossEntropyLoss(weight=ce_weight, ignore_index=ignore_index)
        elif dice_weight or ce_weight is not None:
            raise ValueError("集合交叉熵不支持 DICE_WEIGHT 与 CE_CLASS_WEIGHTS")
        else:
            self.ce = SetCrossEntropyLoss(set_membership, ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=ignore_index)
        self.ignore_index = ignore_index
        self.dice_weight = dice_weight
        self.deep_sup_weights = deep_sup_weights

    def forward(self, logits_list, target):
        if torch.is_tensor(logits_list):
            logits_list = [logits_list]
        elif not isinstance(logits_list, (list, tuple)):
            raise TypeError(
                "SegLoss expects a Tensor or a list/tuple of semantic logits; "
                "use PIDNetLoss for structured outputs"
            )
        if len(logits_list) != len(self.deep_sup_weights):
            raise ValueError(
                f"DEEP_SUP_WEIGHTS has {len(self.deep_sup_weights)} items, "
                f"but the model returned {len(logits_list)} semantic outputs"
            )
        if self.ce.weight is not None and self.ce.weight.numel() != logits_list[0].shape[1]:
            raise ValueError(
                f"CE_CLASS_WEIGHTS 有 {self.ce.weight.numel()} 项，"
                f"但模型输出有 {logits_list[0].shape[1]} 个类别"
            )
        total = 0.0
        for w, logits in zip(self.deep_sup_weights, logits_list):
            if logits.shape[-2:] != target.shape[-2:]:
                logits = F.interpolate(
                    logits, size=target.shape[-2:], mode="bilinear",
                    align_corners=False
                )
            total = total + w * self.single_loss(logits, target)
        return total

    def single_loss(self, logits, target):
        # PyTorch's mean CE is NaN when every pixel is ignored.  Empty or
        # heavily padded remote-sensing tiles should contribute a clean,
        # differentiable zero instead of poisoning the optimizer step.
        if not (target != self.ignore_index).any():
            return logits.sum() * 0.0
        if isinstance(self.ce, SetCrossEntropyLoss):
            return self.ce(logits, target)  # labels index sets, not classes: no Dice
        return self.ce(logits, target) + self.dice_weight * self.dice(logits, target)


def primary_logits(outputs):
    """Return the validation logits from every supported model output schema."""
    if torch.is_tensor(outputs):
        return outputs
    if isinstance(outputs, (list, tuple)):
        if not outputs:
            raise ValueError("model returned an empty output sequence")
        return outputs[0]
    if isinstance(outputs, dict):
        if "main" not in outputs:
            raise KeyError("structured model output is missing the 'main' tensor")
        return outputs["main"]
    raise TypeError(f"unsupported model output type: {type(outputs).__name__}")


def make_boundary_target(target, ignore_index=0, dilation=3):
    """Derive a binary semantic boundary map without an OpenCV dependency.

    A pixel is a boundary when it or a valid four-neighbour belongs to a
    different class.  Pairs touching ``ignore_index`` are excluded, so padded
    validation borders are never taught as object boundaries.  Max pooling
    supplies the dilation step used by PIDNet's boundary supervision.
    """
    if target.ndim != 3:
        raise ValueError("boundary target expects labels shaped [B,H,W]")
    valid = target != ignore_index
    edge = torch.zeros_like(valid)

    vertical = (
        (target[:, 1:, :] != target[:, :-1, :])
        & valid[:, 1:, :] & valid[:, :-1, :]
    )
    edge[:, 1:, :] |= vertical
    edge[:, :-1, :] |= vertical
    horizontal = (
        (target[:, :, 1:] != target[:, :, :-1])
        & valid[:, :, 1:] & valid[:, :, :-1]
    )
    edge[:, :, 1:] |= horizontal
    edge[:, :, :-1] |= horizontal

    boundary = edge.unsqueeze(1).float()
    if dilation > 1:
        if dilation % 2 == 0:
            raise ValueError("BOUNDARY_DILATION must be an odd integer")
        boundary = F.max_pool2d(
            boundary, kernel_size=dilation, stride=1, padding=dilation // 2
        )
    boundary = boundary * valid.unsqueeze(1).float()
    return boundary, valid.unsqueeze(1)


class WeightedBoundaryLoss(nn.Module):
    """Class-balanced BCE used for PIDNet's D-branch supervision."""

    def forward(self, logits, target, valid_mask):
        if logits.shape[-2:] != target.shape[-2:]:
            logits = F.interpolate(
                logits, size=target.shape[-2:], mode="bilinear",
                align_corners=False
            )
        valid = valid_mask.bool()
        positive = (target > 0.5) & valid
        negative = (~positive) & valid
        # Keep counts and BCE in FP32 even under autocast.  A normal
        # 4x512x512 batch already exceeds fp16's largest finite integer;
        # casting the counts to logits.dtype would turn them into ``inf`` and
        # make the balancing ratios NaN.
        work_logits = logits.float()
        work_target = target.float()
        pos_count = positive.sum(dtype=torch.float32)
        neg_count = negative.sum(dtype=torch.float32)
        total = pos_count + neg_count
        if total.item() == 0:
            return work_logits.sum() * 0.0

        weights = torch.zeros_like(work_logits)
        if pos_count.item() == 0 or neg_count.item() == 0:
            weights[valid] = 1.0
        else:
            weights[positive] = neg_count / total
            weights[negative] = pos_count / total
        raw = F.binary_cross_entropy_with_logits(
            work_logits, work_target, reduction="none"
        )
        return (raw * weights).sum() / valid.sum().clamp_min(1).to(raw.dtype)


class PIDNetLoss(nn.Module):
    """PIDNet l0/l1/l2/l3 objective with zero-safe boundary-aware CE."""

    def __init__(self, ignore_index=0, semantic_weights=(1.0, 0.4),
                 dice_weight=0.0, ce_class_weights=None,
                 boundary_weight=20.0, boundary_aware_weight=1.0,
                 boundary_threshold=0.8, boundary_dilation=3, set_membership=None):
        super().__init__()
        self.semantic = SegLoss(
            ignore_index=ignore_index,
            dice_weight=dice_weight,
            deep_sup_weights=tuple(semantic_weights),
            ce_class_weights=ce_class_weights,
            set_membership=set_membership,
        )
        self.ignore_index = ignore_index
        self.semantic_weights = tuple(semantic_weights)
        self.boundary_loss = WeightedBoundaryLoss()
        self.boundary_weight = float(boundary_weight)
        self.boundary_aware_weight = float(boundary_aware_weight)
        self.boundary_threshold = float(boundary_threshold)
        self.boundary_dilation = int(boundary_dilation)
        self.last_components = {}

    def _boundary_aware_ce(self, main, boundary, target):
        if main.shape[-2:] != target.shape[-2:]:
            main = F.interpolate(
                main, size=target.shape[-2:], mode="bilinear",
                align_corners=False
            )
        if boundary.shape[-2:] != target.shape[-2:]:
            boundary = F.interpolate(
                boundary, size=target.shape[-2:], mode="bilinear",
                align_corners=False
            )
        selected = (
            torch.sigmoid(boundary.detach()).squeeze(1)
            > self.boundary_threshold
        ) & (target != self.ignore_index)
        if not selected.any():
            return main.sum() * 0.0
        boundary_target = torch.full_like(target, self.ignore_index)
        boundary_target[selected] = target[selected]
        if isinstance(self.semantic.ce, SetCrossEntropyLoss):
            return self.semantic.ce(main, boundary_target)
        return F.cross_entropy(
            main,
            boundary_target,
            weight=self.semantic.ce.weight,
            ignore_index=self.ignore_index,
        )

    def forward(self, outputs, target):
        if not isinstance(outputs, dict):
            raise TypeError(
                "PIDNetLoss expects {'main', 'aux', 'boundary'} model output"
            )
        main = outputs.get("main")
        boundary = outputs.get("boundary")
        auxiliary = outputs.get("aux", [])
        if torch.is_tensor(auxiliary):
            auxiliary = [auxiliary]
        if not torch.is_tensor(main) or not torch.is_tensor(boundary):
            raise ValueError("PIDNet output must contain Tensor 'main' and 'boundary'")
        semantic_outputs = [main, *list(auxiliary)]
        semantic_loss = self.semantic(semantic_outputs, target)
        boundary_target, valid = make_boundary_target(
            target, self.ignore_index, self.boundary_dilation
        )
        raw_boundary = self.boundary_loss(boundary, boundary_target, valid)
        raw_boundary_aware = self._boundary_aware_ce(main, boundary, target)
        total = (
            semantic_loss
            + self.boundary_weight * raw_boundary
            + self.boundary_aware_weight * raw_boundary_aware
        )
        self.last_components = {
            "semantic": semantic_loss.detach(),
            "boundary": raw_boundary.detach(),
            "boundary_aware": raw_boundary_aware.detach(),
            "total": total.detach(),
        }
        return total


def build_criterion(cfg):
    """Build a loss from config while keeping old configs fully compatible."""
    common = dict(
        ignore_index=cfg.IGNORE_INDEX,
        dice_weight=cfg.DICE_WEIGHT,
        ce_class_weights=getattr(cfg, "CE_CLASS_WEIGHTS", None),
    )
    # Unified fine-class protocol: labels are global native labels and each
    # one allows a set of the model's fine classes.
    if getattr(cfg, "SET_MEMBERSHIP", None) is not None:
        common["set_membership"] = cfg.SET_MEMBERSHIP
    loss_name = getattr(cfg, "LOSS_NAME", "seg").lower()
    if loss_name == "pidnet":
        if not getattr(cfg, "PIDNET_AUXILIARY", True):
            raise ValueError(
                "LOSS_NAME='pidnet' requires PIDNET_AUXILIARY=True; for "
                "main-head-only training use LOSS_NAME='seg' and "
                "DEEP_SUP_WEIGHTS=(1.0,)"
            )
        return PIDNetLoss(
            semantic_weights=cfg.DEEP_SUP_WEIGHTS,
            boundary_weight=getattr(cfg, "BOUNDARY_WEIGHT", 20.0),
            boundary_aware_weight=getattr(cfg, "BOUNDARY_AWARE_WEIGHT", 1.0),
            boundary_threshold=getattr(cfg, "BOUNDARY_THRESHOLD", 0.8),
            boundary_dilation=getattr(cfg, "BOUNDARY_DILATION", 3),
            **common,
        )
    if loss_name != "seg":
        raise ValueError("LOSS_NAME must be 'seg' or 'pidnet'")
    return SegLoss(deep_sup_weights=cfg.DEEP_SUP_WEIGHTS, **common)


# ---------------------------------------------------------------- 优化器
def build_optimizer(model, cfg):
    """AdamW + 可选参数分组，由 config 两个开关控制（不写则与旧版完全一致）：

    - BACKBONE_LR_MULT（默认 1.0）：backbone 参数用 LR * 该系数。
      预训练骨干 + 随机初始化解码头共用大学习率容易在早期破坏预训练特征，常用 0.1。
    - NO_DECAY_BN_BIAS（默认 False）：BN 的 weight/bias 及所有层的 bias 不做 weight decay
      （标准做法：decay 只该作用于卷积/线性层的权重矩阵）。

    两个开关都取默认值时只产生一个参数组，与旧版 AdamW(model.parameters()) 的
    optimizer state_dict 结构相同，旧实验 checkpoint 可正常续训。
    每个参数组写入 initial_lr，供 adjust_learning_rate 按同一衰减系数等比缩放。

    模型可实现 ``no_weight_decay_keywords()``（如 Swin 的相对位置偏置表）；
    NO_DECAY_BN_BIAS=True 时名称含这些关键字的参数也不做 weight decay。
    """
    lr_mult = getattr(cfg, "BACKBONE_LR_MULT", 1.0)
    no_decay_bn_bias = getattr(cfg, "NO_DECAY_BN_BIAS", False)
    no_decay_keywords = ()
    if no_decay_bn_bias and hasattr(model, "no_weight_decay_keywords"):
        no_decay_keywords = tuple(model.no_weight_decay_keywords())

    groups = {"bb_decay": [], "bb_no_decay": [], "decay": [], "no_decay": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_bb = lr_mult != 1.0 and name.startswith("backbone")
        is_nd = no_decay_bn_bias and (
            p.ndim == 1 or name.endswith(".bias")
            or any(keyword in name for keyword in no_decay_keywords)
        )
        key = ("bb_" if is_bb else "") + ("no_decay" if is_nd else "decay")
        groups[key].append(p)

    param_groups = [g for g in [
        {"params": groups["bb_decay"], "lr": cfg.LR * lr_mult},
        {"params": groups["bb_no_decay"], "lr": cfg.LR * lr_mult, "weight_decay": 0.0},
        {"params": groups["decay"]},
        {"params": groups["no_decay"], "weight_decay": 0.0},
    ] if g["params"]]  # 空组剔除，保证默认配置下只有一个组

    optimizer = torch.optim.AdamW(param_groups, lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    for pg in optimizer.param_groups:
        pg["initial_lr"] = pg["lr"]
    return optimizer


# ---------------------------------------------------------------- 学习率
def adjust_learning_rate(optimizer, base_lr, cur_iter, max_iter, warmup_iters=200, power=0.9):
    """poly 衰减 + 线性预热（曲线与原仓库一致）。

    按各参数组的 initial_lr 乘同一衰减系数，兼容 build_optimizer 的分组差异化学习率；
    没有 initial_lr 的组（外部自建的 optimizer）按 base_lr 处理，行为同旧版。
    返回值是 base_lr 组的当前学习率（用于日志）。
    """
    lr = base_lr * (1 - cur_iter / max_iter) ** power
    if cur_iter < warmup_iters:
        lr = base_lr * 0.9 * (cur_iter + 1) / warmup_iters + 0.1 * base_lr
    scale = lr / base_lr  # 必须在 lr 算完（含预热）之后再求系数
    for pg in optimizer.param_groups:
        pg["lr"] = pg.setdefault("initial_lr", base_lr) * scale
    return lr


# ---------------------------------------------------------------- 指标
class ConfusionMatrix:
    """num_classes x num_classes 混淆矩阵，行=真值，列=预测。"""

    def __init__(self, num_classes, ignore_index=0):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred, target):
        """pred/target: 任意形状的整型数组（torch tensor 或 ndarray）。"""
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()
        pred, target = np.asarray(pred), np.asarray(target)
        if pred.shape != target.shape:
            raise ValueError(f"pred/target shapes differ: {pred.shape} != {target.shape}")
        if (not np.issubdtype(pred.dtype, np.integer)
                or not np.issubdtype(target.dtype, np.integer)):
            raise ValueError("pred/target must contain integer class IDs")
        pred = pred.reshape(-1)
        target = target.reshape(-1)
        target_valid = ((target >= 0) & (target < self.num_classes))
        if not (target_valid | (target == self.ignore_index)).all():
            bad = np.unique(target[~target_valid & (target != self.ignore_index)])
            raise ValueError(f"target contains out-of-taxonomy class IDs: {bad[:16].tolist()}")
        if not ((pred >= 0) & (pred < self.num_classes)).all():
            raise ValueError("pred contains out-of-taxonomy class IDs")
        mask = target_valid & (target != self.ignore_index)
        hist = np.bincount(
            self.num_classes * target[mask].astype(np.int64) + pred[mask],
            minlength=self.num_classes ** 2,
        ).reshape(self.num_classes, self.num_classes)
        self.mat += hist

    def scores(self):
        """返回 dict：per_class_iou / per_class_f1 / per_class_support /
        miou / miou_fixed / mf1 / mf1_fixed / oa（若 ignore 是输出内 ID，
        该行不计入均值与 OA；v3 的 ignore=255 位于输出范围外）。

        ``miou``（原有语义，**未改**）
            只对「真值中出现过」的可比较类取平均。域内训练/选模型沿用这个口径。
        ``miou_fixed``（Phase 0 新增）
            对**全部可比较模型类**取平均，真值中缺失的类按 IoU=0 计入分母。
            跨数据集评估必须用这个口径：目标域某类完全缺失时，``miou`` 会因为
            分母变小而**虚高**，只有固定分母才能反映「该类完全没预测对」。
        ``per_class_support`` 给出每类的真值像素数，便于解释某类为何为 0。
        ``confusion_matrix`` 原始像素计数（int64，行=真值，列=预测），ignore 像素不计入，
            供事后另算其他口径。
        """
        mat = self.mat.astype(np.float64)
        keep = np.ones(self.num_classes, dtype=bool)
        if 0 <= self.ignore_index < self.num_classes:
            keep[self.ignore_index] = False
            # 输出内 ignore 的真值整行剔除（预测成什么都不算对错）。
            mat[self.ignore_index, :] = 0

        tp = np.diag(mat)
        fp = mat.sum(0) - tp
        fn = mat.sum(1) - tp

        iou = tp / np.maximum(tp + fp + fn, 1)
        f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
        support = mat.sum(1)                 # 每类真值像素数
        valid = keep & (support > 0)         # 只统计真值中出现过的可比较类

        # miou/mf1/oa 转成纯 Python float：torch>=2.6 的 torch.load(weights_only=True)
        # 不接受 numpy 标量，保证 checkpoint 新旧版本都能直接加载
        return {
            "per_class_iou": iou,
            "per_class_f1": f1,
            "per_class_support": support,
            "miou": float(iou[valid].mean()) if valid.any() else 0.0,
            "mf1": float(f1[valid].mean()) if valid.any() else 0.0,
            # 固定分母：全部可比较类都进平均，缺失类以 IoU=0 计入
            "miou_fixed": float(iou[keep].mean()) if keep.any() else 0.0,
            "mf1_fixed": float(f1[keep].mean()) if keep.any() else 0.0,
            "oa": float(tp[keep].sum() / max(mat[keep, :].sum(), 1)),
            "valid_classes": valid,
            # 真值里一个像素都没有的可比较类
            "absent_classes": keep & ~valid,
            "n_keep_classes": int(keep.sum()),
            "confusion_matrix": self.mat.copy(),
        }


def _window_starts(size, window, stride):
    if size <= window:
        return [0]
    starts = list(range(0, size - window + 1, stride))
    if starts[-1] != size - window:
        starts.append(size - window)
    return starts


def sliding_window_logits(model, img, window, stride):
    """Average the logits of overlapping ``window``² crops (mmseg "slide" mode).

    Crops are cut from the already padded image, so their sides stay multiples
    of the pad divisor when ``window`` is; an image smaller than the window is
    a single crop.
    """
    if window < 1 or not 0 < stride <= window:
        raise ValueError(f"滑窗评估需要 window >= 1 且 0 < stride <= window，收到 {window}/{stride}")
    height, width = img.shape[-2:]
    total = None
    count = torch.zeros((1, 1, height, width), device=img.device, dtype=torch.float32)
    for top in _window_starts(height, window, stride):
        for left in _window_starts(width, window, stride):
            crop = img[..., top:top + window, left:left + window]
            logits = primary_logits(model(crop)).float()
            if logits.shape[-2:] != crop.shape[-2:]:
                logits = F.interpolate(logits, size=crop.shape[-2:], mode="bilinear",
                                       align_corners=False)
            if total is None:
                total = torch.zeros((img.shape[0], logits.shape[1], height, width),
                                    device=img.device, dtype=torch.float32)
            total[..., top:top + crop.shape[-2], left:left + crop.shape[-1]] += logits
            count[..., top:top + crop.shape[-2], left:left + crop.shape[-1]] += 1
    return total / count


@torch.no_grad()
def evaluate(model, loader, device, num_classes, ignore_index=0, amp=True,
             window=None, stride=None, projection=None):
    """在验证集上推理，返回 ConfusionMatrix.scores() 的结果。

    默认整图推理；给出 ``window``/``stride`` 时改用重叠滑窗（大幅影像用）。

    ``projection``（统一细类协议）：[num_classes, F] 的 0/1 矩阵，模型输出 F 个细类，
    每个原生类的得分是其细类集合的概率之和，取最大者为预测。结果另含
    ``fine_confusion``：[num_classes, F] 像素计数，行 = 原生真值，列 = 细类 argmax。
    """
    model.eval()
    cm = ConfusionMatrix(num_classes, ignore_index)
    use_amp = amp and device.type == "cuda"
    output_classes = num_classes
    if projection is not None:
        projection = torch.as_tensor(projection, dtype=torch.float32, device=device)
        if projection.ndim != 2 or projection.shape[0] != num_classes:
            raise ValueError(f"projection 必须是 [{num_classes}, 细类数] 矩阵")
        output_classes = projection.shape[1]
        fine_confusion = np.zeros((num_classes, output_classes), dtype=np.int64)
    for batch_index, batch in enumerate(loader):
        img, lab = batch[0].to(device, non_blocking=True), batch[1]
        names = batch[2] if len(batch) > 2 else [f"batch_{batch_index}_sample_{i}" for i in range(img.shape[0])]
        if isinstance(names, str):
            names = [names]

        def numerical_failure(message, stage):
            return NumericalValidationError(
                message, stage=stage, batch_index=batch_index, sample_names=names,
                images=img, labels=lab, amp=use_amp,
            )

        if not torch.isfinite(img).all():
            raise numerical_failure("evaluation inputs contain non-finite values", "input")
        with autocast_ctx(device.type, enabled=use_amp):
            if window is None:
                logits = primary_logits(model(img))
            else:
                logits = sliding_window_logits(model, img, int(window), int(stride or window))
            if logits.ndim != 4 or logits.shape[1] != output_classes:
                raise ValueError(
                    f"evaluation logits require [B,{output_classes},H,W], got {tuple(logits.shape)}"
                )
            if not torch.isfinite(logits).all():
                raise numerical_failure("evaluation logits contain non-finite values", "logits")
            if logits.shape[-2:] != lab.shape[-2:]:
                logits = F.interpolate(
                    logits, size=lab.shape[-2:], mode="bilinear",
                    align_corners=False
                )
            if not torch.isfinite(logits).all():
                raise numerical_failure("evaluation logits contain non-finite values after resize", "resized_logits")
        if projection is None:
            pred = logits.argmax(1)
        else:
            probs = torch.softmax(logits.float(), dim=1)
            pred = torch.einsum("cf,bfhw->bchw", projection, probs).argmax(1)
        cm.update(pred, lab)  # also rejects out-of-taxonomy labels
        if projection is not None:
            fine = logits.argmax(1).cpu().numpy().reshape(-1)
            target = np.asarray(lab).reshape(-1)
            keep = target != ignore_index
            fine_confusion += np.bincount(
                target[keep].astype(np.int64) * output_classes + fine[keep],
                minlength=num_classes * output_classes,
            ).reshape(num_classes, output_classes)
    scores = cm.scores()
    if projection is not None:
        scores["fine_confusion"] = fine_confusion
    return scores


def format_scores(scores, class_names):
    """把评估结果排版成表格字符串。同时给出两种 mIoU 口径。"""
    lines = [f"{'class':<18}{'IoU':>8}{'F1':>8}{'support':>12}"]
    absent = scores.get("absent_classes")
    for i, name in enumerate(class_names):
        tag = ""
        if not scores["valid_classes"][i]:
            tag = "  (absent in GT)" if absent is None or absent[i] else "  (ignored)"
        sup = scores.get("per_class_support")
        sup_s = f"{int(sup[i]):>12d}" if sup is not None else f"{'-':>12}"
        lines.append(f"{name:<18}{scores['per_class_iou'][i]:>8.4f}"
                     f"{scores['per_class_f1'][i]:>8.4f}{sup_s}{tag}")
    lines.append("-" * 46)
    lines.append(f"mIoU {scores['miou']:.4f} | mF1 {scores['mf1']:.4f} | OA {scores['oa']:.4f}")
    if "miou_fixed" in scores:
        n_abs = int(np.sum(scores["absent_classes"])) if absent is not None else 0
        note = f"（{n_abs} 个前景类在真值中缺失，已按 IoU=0 计入）" if n_abs else ""
        lines.append(f"mIoU_fixed {scores['miou_fixed']:.4f} | "
                     f"mF1_fixed {scores['mf1_fixed']:.4f}  "
                     f"← 固定分母，跨数据集对比用这个{note}")
    return "\n".join(lines)


# ---------------------------------------------------------------- 可视化
def colorize_mask(mask, palette):
    """(H, W) 类别图 → (H, W, 3) 彩色图。"""
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for idx, color in enumerate(palette):
        out[mask == idx] = color
    return out


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.n = 0

    def update(self, v, k=1):
        self.sum += float(v) * k
        self.n += k

    @property
    def avg(self):
        return self.sum / max(self.n, 1)
