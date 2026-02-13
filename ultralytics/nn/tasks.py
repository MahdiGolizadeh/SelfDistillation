# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import contextlib
import pickle
import re
import types
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.autobackend import check_class_names
from ultralytics.nn.modules import (
    AIFI,
    C1,
    C2,
    C2PSA,
    C3,
    C3TR,
    ELAN1,
    OBB,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    A2C2f,
    AConv,
    ADown,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    C3k2,
    C3x,
    CBFuse,
    CBLinear,
    Classify,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    Detect,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostBottleneck,
    GhostConv,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    Index,
    LRPCHead,
    Pose,
    RepC3,
    RepConv,
    RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    RTDETRDecoder,
    SCDown,
    Segment,
    TorchVision,
    WorldDetect,
    YOLOEDetect,
    YOLOESegment,
    v10Detect,
)
from ultralytics.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, YAML, colorstr, emojis
from ultralytics.utils.checks import check_requirements, check_suffix, check_yaml
from ultralytics.utils.loss import (
    E2EDetectLoss,
    generate_masks_from_teacher_tal,
    v8ClassificationLoss,
    v8DetectionLoss,
    v8OBBLoss,
    v8PoseLoss,
    v8SegmentationLoss,
)
from ultralytics.utils.ops import make_divisible
from ultralytics.utils.patches import torch_load
from ultralytics.utils.plotting import feature_visualization
from ultralytics.utils.tal import make_anchors
from ultralytics.utils.torch_utils import (
    fuse_conv_and_bn,
    fuse_deconv_and_bn,
    initialize_weights,
    intersect_dicts,
    model_info,
    scale_img,
    smart_inference_mode,
    time_sync,
)

class DualBatchNorm2d(nn.Module):
    """BatchNorm2d wrapper with independent running stats for full-net and subnet passes."""

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.bn_full = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.bn_subnet = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.mode = "full"

    @classmethod
    def from_batchnorm(cls, bn):
        """Create dual BN from an existing BatchNorm2d, cloning weights/stats to both branches."""
        dual = cls(
            bn.num_features,
            eps=bn.eps,
            momentum=bn.momentum,
            affine=bn.affine,
            track_running_stats=bn.track_running_stats,
        )
        dual.bn_full.load_state_dict(bn.state_dict())
        dual.bn_subnet.load_state_dict(bn.state_dict())
        return dual

    def set_mode(self, mode="full"):
        self.mode = "subnet" if mode == "subnet" else "full"

    @property
    def active_bn(self):
        return self.bn_subnet if self.mode == "subnet" else self.bn_full

    def forward(self, x):
        return self.active_bn(x)

    # Compatibility attrs used by fuse helpers that expect a BatchNorm-like object.
    @property
    def weight(self):
        return self.bn_full.weight

    @property
    def bias(self):
        return self.bn_full.bias

    @property
    def running_mean(self):
        return self.bn_full.running_mean

    @property
    def running_var(self):
        return self.bn_full.running_var

    @property
    def eps(self):
        return self.bn_full.eps

class BaseModel(torch.nn.Module):
    """
    Base class for all YOLO models in the Ultralytics family.

    This class provides common functionality for YOLO models including forward pass handling, model fusion,
    information display, and weight loading capabilities.

    Attributes:
        model (torch.nn.Module): The neural network model.
        save (list): List of layer indices to save outputs from.
        stride (torch.Tensor): Model stride values.

    Methods:
        forward: Perform forward pass for training or inference.
        predict: Perform inference on input tensor.
        fuse: Fuse Conv2d and BatchNorm2d layers for optimization.
        info: Print model information.
        load: Load weights into the model.
        loss: Compute loss for training.

    Examples:
        Create a BaseModel instance
        >>> model = BaseModel()
        >>> model.info()  # Display model information
    """

    def forward(self, x, *args, **kwargs):
        """
        Perform forward pass of the model for either training or inference.

        If x is a dict, calculates and returns the loss for training. Otherwise, returns predictions for inference.

        Args:
            x (torch.Tensor | dict): Input tensor for inference, or dict with image tensor and labels for training.
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            (torch.Tensor): Loss if x is a dict (training), or network predictions (inference).
        """
        if isinstance(x, dict):  # for cases of training and validating while training.
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool): Print the computation time of each layer if True.
            visualize (bool): Save the feature maps of the model if True.
            augment (bool): Augment image during prediction.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        if augment:
            return self._predict_augment(x)
        return self._predict_once(x, profile, visualize, embed)

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool): Print the computation time of each layer if True.
            visualize (bool): Save the feature maps of the model if True.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        y, dt, embeddings = [], [], []  # outputs
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            x = self._apply_active_channel_mask(x)
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def _apply_active_channel_mask(self, x):
        """Mask channels for slimmable-style subnet training using ratio or divisor."""
        spec = getattr(self, "_active_channel_ratio", 3.0)
        if not isinstance(x, torch.Tensor) or x.ndim < 2:
            return x

        channels = x.shape[1]
        keep = channels
        if isinstance(spec, (int, float)) and float(spec).is_integer() and float(spec) > 1:
            # Integer mode: `spec=3` means keep `channels // 3` channels.
            keep = max(1, channels // int(spec))
        else:
            ratio = float(spec)
            if ratio >= 1.0:
                return x
            keep = max(1, min(channels, int(round(channels * ratio))))

        if keep >= channels:
            return x

        masked = x.clone()
        masked[:, keep:] = 0
        return masked

    @contextlib.contextmanager
    def active_channel_ratio(self, ratio=1.0):
        """Temporarily set active output-channel ratio/divisor used during forward passes."""
        prev = getattr(self, "_active_channel_ratio", 3.0)
        self._active_channel_ratio = ratio
        try:
            yield
        finally:
            self._active_channel_ratio = prev

    @contextlib.contextmanager
    def active_bn_mode(self, mode="full"):
        """Temporarily switch dual-batchnorm layers between full-net and subnet statistics."""
        prev = getattr(self, "_active_bn_mode", "full")
        self._active_bn_mode = mode
        self._set_dual_bn_mode(mode)
        try:
            yield
        finally:
            self._active_bn_mode = prev
            self._set_dual_bn_mode(prev)

    def _set_dual_bn_mode(self, mode="full"):
        """Propagate active BN mode to all dual-batchnorm layers in the model."""
        for module in self.modules():
            if isinstance(module, DualBatchNorm2d):
                module.set_mode(mode)

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference."""
        LOGGER.warning(
            f"{self.__class__.__name__} does not support 'augment=True' prediction. "
            f"Reverting to single-scale prediction."
        )
        return self._predict_once(x)

    def _profile_one_layer(self, m, x, dt):
        """
        Profile the computation time and FLOPs of a single layer of the model on a given input.

        Args:
            m (torch.nn.Module): The layer to be profiled.
            x (torch.Tensor): The input data to the layer.
            dt (list): A list to store the computation time of the layer.
        """
        try:
            import thop
        except ImportError:
            thop = None  # conda support without 'ultralytics-thop' installed

        c = m == self.model[-1] and isinstance(x, list)  # is final layer list, copy input as inplace fix
        flops = thop.profile(m, inputs=[x.copy() if c else x], verbose=False)[0] / 1e9 * 2 if thop else 0  # GFLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f"{dt[-1]:10.2f} {flops:10.2f} {m.np:10.0f}  {m.type}")
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self, verbose=True):
        """
        Fuse the `Conv2d()` and `BatchNorm2d()` layers of the model into a single layer for improved computation
        efficiency.

        Returns:
            (torch.nn.Module): The fused model is returned.
        """
        if not self.is_fused():
            for m in self.model.modules():
                if isinstance(m, (Conv, Conv2, DWConv)) and hasattr(m, "bn"):
                    if isinstance(m, Conv2):
                        m.fuse_convs()
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                    delattr(m, "bn")  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, ConvTranspose) and hasattr(m, "bn"):
                    m.conv_transpose = fuse_deconv_and_bn(m.conv_transpose, m.bn)
                    delattr(m, "bn")  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, RepConv):
                    m.fuse_convs()
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, RepVGGDW):
                    m.fuse()
                    m.forward = m.forward_fuse
                if isinstance(m, v10Detect):
                    m.fuse()  # remove one2many head
            self.info(verbose=verbose)

        return self

    def is_fused(self, thresh=10):
        """
        Check if the model has less than a certain threshold of BatchNorm layers.

        Args:
            thresh (int, optional): The threshold number of BatchNorm layers.

        Returns:
            (bool): True if the number of BatchNorm layers in the model is less than the threshold, False otherwise.
        """
        bn = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        return sum(isinstance(v, bn) for v in self.modules()) < thresh  # True if < 'thresh' BatchNorm layers in model

    def info(self, detailed=False, verbose=True, imgsz=640):
        """
        Print model information.

        Args:
            detailed (bool): If True, prints out detailed information about the model.
            verbose (bool): If True, prints out the model information.
            imgsz (int): The size of the image that the model will be trained on.
        """
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)

    def _apply(self, fn):
        """
        Apply a function to all tensors in the model that are not parameters or registered buffers.

        Args:
            fn (function): The function to apply to the model.

        Returns:
            (BaseModel): An updated BaseModel object.
        """
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(
            m, Detect
        ):  # includes all Detect subclasses like Segment, Pose, OBB, WorldDetect, YOLOEDetect, YOLOESegment
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)
        return self

    def load(self, weights, verbose=True):
        """
        Load weights into the model.

        Args:
            weights (dict | torch.nn.Module): The pre-trained weights to be loaded.
            verbose (bool, optional): Whether to log the transfer progress.
        """
        model = weights["model"] if isinstance(weights, dict) else weights  # torchvision models are not dicts
        csd = model.float().state_dict()  # checkpoint state_dict as FP32
        updated_csd = intersect_dicts(csd, self.state_dict())  # intersect
        self.load_state_dict(updated_csd, strict=False)  # load
        len_updated_csd = len(updated_csd)
        first_conv = "model.0.conv.weight"  # hard-coded to yolo models for now
        # mostly used to boost multi-channel training
        state_dict = self.state_dict()
        if first_conv not in updated_csd and first_conv in state_dict:
            c1, c2, h, w = state_dict[first_conv].shape
            cc1, cc2, ch, cw = csd[first_conv].shape
            if ch == h and cw == w:
                c1, c2 = min(c1, cc1), min(c2, cc2)
                state_dict[first_conv][:c1, :c2] = csd[first_conv][:c1, :c2]
                len_updated_csd += 1
        if verbose:
            LOGGER.info(f"Transferred {len_updated_csd}/{len(self.model.state_dict())} items from pretrained weights")

    def loss(self, batch, preds=None):
        """
        Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | List[torch.Tensor], optional): Predictions.
        """
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        preds = self.forward(batch["img"]) if preds is None else preds
        return self.criterion(preds, batch)

    def init_criterion(self):
        """Initialize the loss criterion for the BaseModel."""
        raise NotImplementedError("compute_loss() needs to be implemented by task heads")


class DetectionModel(BaseModel):
    """
    YOLO detection model.

    This class implements the YOLO detection architecture, handling model initialization, forward pass,
    augmented inference, and loss computation for object detection tasks.

    Attributes:
        yaml (dict): Model configuration dictionary.
        model (torch.nn.Sequential): The neural network model.
        save (list): List of layer indices to save outputs from.
        names (dict): Class names dictionary.
        inplace (bool): Whether to use inplace operations.
        end2end (bool): Whether the model uses end-to-end detection.
        stride (torch.Tensor): Model stride values.

    Methods:
        __init__: Initialize the YOLO detection model.
        _predict_augment: Perform augmented inference.
        _descale_pred: De-scale predictions following augmented inference.
        _clip_augmented: Clip YOLO augmented inference tails.
        init_criterion: Initialize the loss criterion.

    Examples:
        Initialize a detection model
        >>> model = DetectionModel("yolo11n.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor)
    """
    FORCED_WEIGHTS = "yolo11n.pt"
    def __init__(
        self,
        cfg="yolo11n.yaml",
        ch=3,
        nc=None,
        verbose=True,
        pretrained=True,
        dual_channel_ratio=3,
        dual_channel_loss_weight=1.0,
        channel_scale=1.0,
        use_dual_bn=True,
    ):

        """
        Initialize the YOLO detection model with the given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
            pretrained (bool): If True, load forced pretrained YOLO11n weights after building.
            dual_channel_ratio (float, optional): If set in (0, 1), also train with a masked channel subnet.
            dual_channel_loss_weight (float): Weight for the masked subnet loss contribution.
            channel_scale (float): Global channel scaling factor for building a compact network.
            use_dual_bn (bool): If True, replace BatchNorm with dual BN (full/subnet stats).
        """
        super().__init__()
        cfg_ok = str(cfg) in {"yolo11n.yaml", self.FORCED_WEIGHTS}
        if isinstance(cfg, dict):
            cfg_ok = cfg.get("yaml_file") in {"hardcoded_yolo11n", "yolo11n.yaml"}
        if not cfg_ok:
            LOGGER.warning(
                f"Ignoring requested detection config '{cfg}'. This build always uses hardcoded YOLO11n architecture."
            )
        if ch != 3:
            LOGGER.warning(f"Ignoring requested input channels ch={ch}. This build always uses ch=3.")
        model_nc = nc if nc is not None else 80

        self.yaml = {"nc": model_nc, "channels": 3, "yaml_file": "hardcoded_yolo11n"}
        self.model, self.save = self._build_hardcoded_yolo11n(model_nc, width_mult=channel_scale)
        self.names = {i: f"{i}" for i in range(model_nc)}
        self.dual_channel_ratio = dual_channel_ratio
        self.dual_channel_loss_weight = dual_channel_loss_weight

        self.inplace = True
        self.end2end = getattr(self.model[-1], "end2end", False)

        m = self.model[-1]
        if isinstance(m, Detect):
            s = 256

            m.inplace = self.inplace

            def _forward(x):
                if self.end2end:
                    return self.forward(x)["one2many"]
                return self.forward(x)[0] if isinstance(m, (Segment, YOLOESegment, Pose, OBB)) else self.forward(x)

            self.model.eval()
            m.training = True
            m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, 3, s, s))])
            self.stride = m.stride
            self.model.train()
            m.bias_init()
        else:
            self.stride = torch.Tensor([32])

        initialize_weights(self)

        if pretrained:
            pretrained_model, _ = attempt_load_one_weight(self.FORCED_WEIGHTS)
            self.load(pretrained_model, verbose=verbose)
            if hasattr(pretrained_model, "stride"):
                self.stride = pretrained_model.stride.clone()
        elif verbose:
            LOGGER.info("Using hardcoded YOLO11n architecture with random initialization (pretrained=False).")

        if use_dual_bn:
            self._replace_batchnorm_with_dual()
            self._set_dual_bn_mode("full")

        if verbose:
            self.info()
            LOGGER.info("")
            
    def _replace_batchnorm_with_dual(self):
        """Replace every BatchNorm2d with dual BN to keep separate full/subnet running stats."""
        # Snapshot modules first so replacements don't recursively descend into newly created DualBatchNorm2d branches.
        for module in list(self.modules()):
            if isinstance(module, DualBatchNorm2d):
                continue
            for name, child in list(module.named_children()):
                if isinstance(child, nn.BatchNorm2d):
                    setattr(module, name, DualBatchNorm2d.from_batchnorm(child))

    def get_dual_bn_parameters(self):
        """Return BatchNorm parameters for full and subnet branches keyed by module path."""
        bn_params = {"full": {}, "subnet": {}}
        for name, module in self.named_modules():
            if isinstance(module, DualBatchNorm2d):
                bn_params["full"][name] = {
                    k: v.detach().cpu().clone() for k, v in module.bn_full.state_dict().items()
                }
                bn_params["subnet"][name] = {
                    k: v.detach().cpu().clone() for k, v in module.bn_subnet.state_dict().items()
                }
        return bn_params

    @staticmethod
    def _build_hardcoded_yolo11n(nc=80, width_mult=1.0):
        """Build YOLO11n detection architecture directly as PyTorch modules (no YAML parsing)."""
        layers = []

        def c(ch):
            return max(1, int(round(ch * float(width_mult))))

        def add(module, f):
            i = len(layers)
            module.i = i
            module.f = f
            module.type = str(module.__class__.__name__)
            module.np = sum(x.numel() for x in module.parameters())
            layers.append(module)

        add(Conv(3, c(48), 3, 2), -1)
        add(Conv(c(48), c(96), 3, 2), -1)
        add(C3k2(c(96), c(192), 1, False, 0.25), -1)
        add(Conv(c(192), c(192), 3, 2), -1)
        add(C3k2(c(192), c(384), 1, False, 0.25), -1)
        add(Conv(c(384), c(384), 3, 2), -1)
        add(C3k2(c(384), c(384), 1, True), -1)
        add(Conv(c(384), c(768), 3, 2), -1)
        add(C3k2(c(768), c(768), 1, True), -1)
        add(SPPF(c(768), c(768), 5), -1)
        add(C2PSA(c(768), c(768), 1), -1)
        add(nn.Upsample(None, 2, "nearest"), -1)
        add(Concat(1), [-1, 6])
        add(C3k2(c(1152), c(384), 1, False), -1)
        add(nn.Upsample(None, 2, "nearest"), -1)
        add(Concat(1), [-1, 4])
        add(C3k2(c(768), c(192), 1, False), -1)
        add(Conv(c(192), c(192), 3, 2), -1)
        add(Concat(1), [-1, 13])
        add(C3k2(c(576), c(384), 1, False), -1)
        add(Conv(c(384), c(384), 3, 2), -1)
        add(Concat(1), [-1, 10])
        add(C3k2(c(1152), c(768), 1, True), -1)
        add(Detect(nc, (c(192), c(384), c(768))), [16, 19, 22])

        return nn.Sequential(*layers), [4, 6, 10, 13, 16, 19, 22]

    def _predict_augment(self, x):
        """
        Perform augmentations on input image x and return augmented inference and train outputs.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            (torch.Tensor): Augmented inference output.
        """
        if getattr(self, "end2end", False) or self.__class__.__name__ != "DetectionModel":
            LOGGER.warning("Model does not support 'augment=True', reverting to single-scale prediction.")
            return self._predict_once(x)
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = super().predict(xi)[0]  # forward
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, -1), None  # augmented inference, train

    @staticmethod
    def _parse_dual_channel_ratio(ratio):
        """Parse subnet spec as ratio (<1) or divisor integer (>1)."""
        if ratio is None:
            return None
        if isinstance(ratio, int):
            return ratio
        if isinstance(ratio, float):
            return int(ratio) if ratio.is_integer() else ratio
        text = str(ratio).strip()
        if "/" in text:
            num, den = text.split("/", 1)
            return float(num.strip()) / float(den.strip())
        value = float(text)
        return int(value) if value.is_integer() else value

    def loss(self, batch, preds=None, return_bn_params=False):
        """Compute standard or dual-channel (subnet + full-net) detection loss."""
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        ratio = self._parse_dual_channel_ratio(self.dual_channel_ratio)
        ratio_is_fraction = isinstance(ratio, float) and 0.0 < ratio < 1.0
        ratio_is_divisor = ratio > 1
        use_dual = (
            preds is None
            and self.training
            and ratio is not None
            and (ratio_is_fraction or ratio_is_divisor)
            and float(self.dual_channel_loss_weight) > 0.0
        )
        if not use_dual:
            preds = self.forward(batch["img"]) if preds is None else preds
            loss_output = self.criterion(preds, batch)
            if return_bn_params:
                return (*loss_output, self.get_dual_bn_parameters())
            return loss_output

        # 1) Subnet pass (e.g. ratio=1/3 keeps first 16 of 48 channels in stem conv).
        with self.active_bn_mode("subnet"):
            with self.active_channel_ratio(ratio):
                preds_subnet = self.forward(batch["img"])
        loss_subnet, items_subnet = self.criterion(preds_subnet, batch)

        # 2) Full-net pass. Shared channels receive gradients from both passes.
        with self.active_bn_mode("full"):
            with self.active_channel_ratio(1.0):
                preds_full = self.forward(batch["img"])
        loss_full, items_full = self.criterion(preds_full, batch)

        alpha = float(self.dual_channel_loss_weight)
        cls_dist_alpha = float(getattr(self.args, "cls_alpha", 0.0))
        cls_dist_temp = float(getattr(self.args, "cls_dist_t", 1.0))
        m2d2_alpha = float(getattr(self.args, "m2d2_alpha", 0.0))
        m2d2_temp = float(getattr(self.args, "m2d2_t", 1.0))
        l2_alpha = float(getattr(self.args, "l2_alpha", 0.0))
        cls_dist_enabled = bool(getattr(self.args, "cls_dist", cls_dist_alpha > 0.0))
        m2d2_enabled = bool(getattr(self.args, "m2d2_dist", m2d2_alpha > 0.0))
        l2_enabled = bool(getattr(self.args, "l2_dist", l2_alpha > 0.0))
        cls_dist_alpha = cls_dist_alpha if cls_dist_enabled else 0.0
        m2d2_alpha = m2d2_alpha if m2d2_enabled else 0.0
        l2_alpha = l2_alpha if l2_enabled else 0.0
        use_cls_fg_mask = bool(getattr(self.args, "cls_fg_mask", False))
        use_dfl_fg_mask = bool(getattr(self.args, "dfl_fg_mask", False))
        use_l2_fg_mask = bool(getattr(self.args, "l2_fg_mask", False))
        mask_type = str(getattr(self.args, "mask_type", "original"))

        need_tal = (
            (cls_dist_alpha > 0.0 and use_cls_fg_mask)
            or (m2d2_alpha > 0.0 and use_dfl_fg_mask)
            or (l2_alpha > 0.0 and use_l2_fg_mask)
        )
        tal_data = None
        if need_tal:
            tal_data = self._compute_teacher_tal_data(self._extract_train_feats(preds_full), batch, self.criterion, mask_type=mask_type)

        if cls_dist_alpha > 0.0:
            distill_cls_loss = self._compute_cls_kl_distillation_loss(
                teacher_preds=preds_full,
                student_preds=preds_subnet,
                temperature=cls_dist_temp,
                level_weights=getattr(self.args, "level_weights", None),
                masks=tal_data["masks"] if use_cls_fg_mask and tal_data is not None else None,
            )
        else:
            distill_cls_loss = loss_full.new_tensor(0.0)

        if m2d2_alpha > 0.0:
            m2d2_loss = self._compute_m2d2_distillation_loss(
                teacher_preds=preds_full,
                student_preds=preds_subnet,
                temperature=m2d2_temp,
                target_labels=tal_data["target_labels"] if tal_data is not None else None,
                masks=tal_data["masks"] if use_dfl_fg_mask and tal_data is not None else None,
                level_weights=getattr(self.args, "level_weights", None),
            )
        else:
            m2d2_loss = loss_full.new_tensor(0.0)

        if l2_alpha > 0.0:
            l2_loss = self._compute_l2_bbox_distillation_loss(
                teacher_preds=preds_full,
                student_preds=preds_subnet,
                masks=tal_data["masks"] if use_l2_fg_mask and tal_data is not None else None,
                level_weights=getattr(self.args, "level_weights", None),
            )
        else:
            l2_loss = loss_full.new_tensor(0.0)

        if bool(getattr(self.args, "adaptive_distill_alpha", True)):
            # Use the mean of supervised components (box/cls/dfl) as the common reference scale.
            ref_items = (items_full.detach() + alpha * items_subnet.detach()).float()
            ref_loss = ref_items.mean()
            cls_dist_alpha = self._compute_adaptive_distill_alpha("cls", cls_dist_alpha, distill_cls_loss, ref_loss)
            m2d2_alpha = self._compute_adaptive_distill_alpha("m2d2", m2d2_alpha, m2d2_loss, ref_loss)
            l2_alpha = self._compute_adaptive_distill_alpha("l2", l2_alpha, l2_loss, ref_loss)

        train_items = torch.cat((items_full, items_subnet))
        loss_output = (
            loss_full + alpha * loss_subnet + cls_dist_alpha * distill_cls_loss + m2d2_alpha * m2d2_loss + l2_alpha * l2_loss,
            train_items,
        )
        logits = {"full": preds_full, "subnet": preds_subnet}
        if return_bn_params:
            return (*loss_output, self.get_dual_bn_parameters(), logits)
        return (*loss_output, logits)
    def _compute_adaptive_distill_alpha(self, name, base_alpha, distill_loss, ref_loss):
        """Adapt distillation alpha from EMA of supervised and distillation losses."""
        if base_alpha <= 0.0:
            return 0.0

        if not hasattr(self, "_adaptive_alpha_state"):
            self._adaptive_alpha_state = {}

        momentum = float(getattr(self.args, "adaptive_alpha_momentum", 0.9))
        momentum = min(max(momentum, 0.0), 0.9999)
        min_scale = float(getattr(self.args, "adaptive_alpha_min_scale", 0.25))
        max_scale = float(getattr(self.args, "adaptive_alpha_max_scale", 4.0))
        eps = 1e-6

        ref_tensor = ref_loss.detach()
        distill_tensor = distill_loss.detach()

        # Reference loss may come from vector components (e.g. box/cls/dfl).
        if ref_tensor.numel() > 1:
            ref_tensor = ref_tensor.mean()

        # Keep distillation tensors on the same scale basis as the supervised reference.
        # If a branch returns component-wise vectors, average them before adaptive scaling.
        if distill_tensor.numel() > 1:
            distill_tensor = distill_tensor.mean()

        ref_value = max(float(ref_tensor.item()), eps)
        distill_value = max(float(distill_tensor.item()), eps)
        state = self._adaptive_alpha_state.get(name)

        if state is None:
            state = {"ref_ema": ref_value, "distill_ema": distill_value}
        else:
            state["ref_ema"] = momentum * state["ref_ema"] + (1.0 - momentum) * ref_value
            state["distill_ema"] = momentum * state["distill_ema"] + (1.0 - momentum) * distill_value

        self._adaptive_alpha_state[name] = state
        scale = state["ref_ema"] / (state["distill_ema"] + eps)
        scale = min(max(scale, min_scale), max_scale)
        return base_alpha * scale

    @staticmethod
    def _extract_train_feats(preds):
        """Extract training feature maps from raw forward outputs."""
        return preds[1] if isinstance(preds, tuple) else preds

    def _compute_teacher_tal_data(self, teacher_feats, batch, criterion, mask_type="original"):
        """Build teacher TAL data including foreground masks and assigned labels."""
        with torch.no_grad():
            pred_distri, pred_scores = torch.cat(
                [xi.view(teacher_feats[0].shape[0], criterion.no, -1) for xi in teacher_feats], 2
            ).split((criterion.reg_max * 4, criterion.nc), 1)
            pred_scores = pred_scores.permute(0, 2, 1).contiguous()
            pred_distri = pred_distri.permute(0, 2, 1).contiguous()

            dtype = pred_scores.dtype
            batch_size = pred_scores.shape[0]
            imgsz = torch.tensor(teacher_feats[0].shape[2:], device=criterion.device, dtype=dtype) * criterion.stride[0]
            anchor_points, stride_tensor = make_anchors(teacher_feats, criterion.stride, 0.5)

            targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = criterion.preprocess(targets.to(criterion.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
            pred_bboxes = criterion.bbox_decode(anchor_points, pred_distri)
            target_labels, _, _, fg_mask, _ = criterion.assigner(
                pred_scores.detach().sigmoid(),
                (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                anchor_points * stride_tensor,
                gt_labels,
                gt_bboxes,
                mask_gt,
            )

        masks = generate_masks_from_teacher_tal(fg_mask, teacher_feats, mask_type=mask_type)
        return {"masks": masks, "target_labels": target_labels}

    def _compute_cls_kl_distillation_loss(self, teacher_preds, student_preds, temperature=1.0, level_weights=None, masks=None):
        """Compute KL class distillation between full-net (teacher) and subnet (student) prediction heads."""
        teacher_feats = self._extract_train_feats(teacher_preds)
        student_feats = self._extract_train_feats(student_preds)
        if len(teacher_feats) != len(student_feats):
            raise ValueError("Teacher and student feature levels must match for class distillation")

        num_levels = len(teacher_feats)
        if level_weights is None:
            level_weights = [1.0] * num_levels
        if len(level_weights) != num_levels:
            raise ValueError(f"level_weights length ({len(level_weights)}) must match number of levels ({num_levels})")
        if masks is not None and len(masks) != num_levels:
            raise ValueError(f"masks length ({len(masks)}) must match number of levels ({num_levels})")

        distill_cls_loss = teacher_feats[0].new_tensor(0.0)
        class_channels = int(self.model[-1].nc)
        for i, (s_pred, t_pred) in enumerate(zip(student_feats, teacher_feats)):
            s_logits = s_pred[:, -class_channels:, :, :]
            t_logits = t_pred[:, -class_channels:, :, :]

            with torch.no_grad():
                t_probs = F.softmax(t_logits / temperature, dim=1)
            s_log_probs = F.log_softmax(s_logits / temperature, dim=1)

            kl = F.kl_div(s_log_probs, t_probs, reduction="none").sum(dim=1)
            kl = kl * (temperature * temperature)

            if masks is not None:
                fg_mask = masks[i].squeeze(1)
                loss_level = (kl * fg_mask).sum() / (fg_mask.sum() + 1e-6)
            else:
                loss_level = kl.mean()

            distill_cls_loss = distill_cls_loss + float(level_weights[i]) * loss_level
        return distill_cls_loss

    @staticmethod
    def _connected_components_2d(mask):
        """Return connected component labels for a binary 2D numpy mask (4-connectivity)."""
        h, w = mask.shape
        labels = np.zeros((h, w), dtype=np.int32)
        comp_id = 0
        for y in range(h):
            for x in range(w):
                if mask[y, x] == 0 or labels[y, x] != 0:
                    continue
                comp_id += 1
                stack = [(y, x)]
                labels[y, x] = comp_id
                while stack:
                    cy, cx = stack.pop()
                    if cy > 0 and mask[cy - 1, cx] and labels[cy - 1, cx] == 0:
                        labels[cy - 1, cx] = comp_id
                        stack.append((cy - 1, cx))
                    if cy + 1 < h and mask[cy + 1, cx] and labels[cy + 1, cx] == 0:
                        labels[cy + 1, cx] = comp_id
                        stack.append((cy + 1, cx))
                    if cx > 0 and mask[cy, cx - 1] and labels[cy, cx - 1] == 0:
                        labels[cy, cx - 1] = comp_id
                        stack.append((cy, cx - 1))
                    if cx + 1 < w and mask[cy, cx + 1] and labels[cy, cx + 1] == 0:
                        labels[cy, cx + 1] = comp_id
                        stack.append((cy, cx + 1))
        return labels, comp_id

    def _build_m2d2_teacher_preds(self, teacher_feats, target_labels, masks):
        """Apply M2D2 component-wise/class-wise DFL averaging on teacher prediction maps."""
        class_channels = int(self.model[-1].nc)
        updated_preds = []
        offset = 0
        for lvl, feat in enumerate(teacher_feats):
            b, c, h, w = feat.shape
            hw = h * w
            lvl_labels = target_labels[:, offset : offset + hw].detach().to(torch.long).cpu().numpy()
            offset += hw

            feat_flat = feat.view(b, c, hw).clone()
            cls_part = feat_flat[:, -class_channels:, :]
            dfl_part = feat_flat[:, : c - class_channels, :]
            mask_np = masks[lvl].squeeze(1).detach().bool().cpu().numpy()

            for bi in range(b):
                mask_2d = mask_np[bi].reshape(h, w).astype(np.uint8)
                if mask_2d.sum() == 0:
                    continue
                labeled_array, num_features = self._connected_components_2d(mask_2d)
                if num_features == 0:
                    continue

                labels_flat = lvl_labels[bi]
                labeled_flat = labeled_array.reshape(-1)
                for comp_id in range(1, num_features + 1):
                    comp_idx = np.nonzero(labeled_flat == comp_id)[0]
                    if comp_idx.size == 0:
                        continue
                    for cls in np.unique(labels_flat[comp_idx]):
                        cls_idx = comp_idx[labels_flat[comp_idx] == cls]
                        if cls_idx.size == 0:
                            continue
                        pos_idx = torch.from_numpy(cls_idx).long().to(feat.device)
                        vals = dfl_part[bi][:, pos_idx]
                        dfl_part[bi][:, pos_idx] = vals.mean(dim=1, keepdim=True)

            updated_preds.append(torch.cat([dfl_part, cls_part], dim=1).view(b, c, h, w))
        return updated_preds

    def _compute_m2d2_distillation_loss(
        self,
        teacher_preds,
        student_preds,
        temperature=1.0,
        target_labels=None,
        masks=None,
        level_weights=None,
    ):
        """Compute M2D2 KL distillation on DFL channels after teacher component-wise smoothing."""
        teacher_feats = self._extract_train_feats(teacher_preds)
        student_feats = self._extract_train_feats(student_preds)
        if len(teacher_feats) != len(student_feats):
            raise ValueError("Teacher and student feature levels must match for M2D2 distillation")
        if target_labels is None:
            raise ValueError("target_labels from teacher TAL assignment are required for M2D2 distillation")

        num_levels = len(teacher_feats)
        if level_weights is None:
            level_weights = [1.0] * num_levels
        if len(level_weights) != num_levels:
            raise ValueError(f"level_weights length ({len(level_weights)}) must match number of levels ({num_levels})")

        if masks is None:
            masks = [torch.ones(f.shape[0], 1, f.shape[2], f.shape[3], device=f.device, dtype=f.dtype) for f in teacher_feats]
        if len(masks) != num_levels:
            raise ValueError(f"masks length ({len(masks)}) must match number of levels ({num_levels})")

        updated_teacher = self._build_m2d2_teacher_preds(teacher_feats, target_labels, masks)
        reg_max = int(self.model[-1].reg_max)
        dfl_channels = 4 * reg_max
        distill_loss = teacher_feats[0].new_tensor(0.0)
        total_weight = 0.0

        for i, (sp, tp) in enumerate(zip(student_feats, updated_teacher)):
            b, _, h, w = sp.shape
            sp_dfl = sp[:, :dfl_channels, :, :].view(b, 4, reg_max, h, w)
            tp_dfl = tp[:, :dfl_channels, :, :].view(b, 4, reg_max, h, w)

            with torch.no_grad():
                tp_prob = F.softmax(tp_dfl / temperature, dim=2)
            sp_log_prob = F.log_softmax(sp_dfl / temperature, dim=2)

            kl_spatial = F.kl_div(sp_log_prob, tp_prob, reduction="none").sum(dim=2).mean(dim=1)
            kl_spatial = kl_spatial * (temperature * temperature)

            if masks is not None:
                fg = masks[i].squeeze(1)
                loss_level = (kl_spatial * fg).sum() / (fg.sum() + 1e-6)
            else:
                loss_level = kl_spatial.mean()

            w = float(level_weights[i])
            distill_loss = distill_loss + w * loss_level
            total_weight += w

        return distill_loss / max(total_weight, 1e-6)

    def _compute_l2_bbox_distillation_loss(self, teacher_preds, student_preds, masks=None, level_weights=None):
        """Compute L2 distillation over expected DFL box offsets with optional foreground masking."""
        teacher_feats = self._extract_train_feats(teacher_preds)
        student_feats = self._extract_train_feats(student_preds)
        if len(teacher_feats) != len(student_feats):
            raise ValueError("Teacher and student feature levels must match for L2 bbox distillation")

        num_levels = len(teacher_feats)
        if level_weights is None:
            level_weights = [1.0] * num_levels
        if len(level_weights) != num_levels:
            raise ValueError(f"level_weights length ({len(level_weights)}) must match number of levels ({num_levels})")
        if masks is not None and len(masks) != num_levels:
            raise ValueError(f"masks length ({len(masks)}) must match number of levels ({num_levels})")

        reg_max = int(self.model[-1].reg_max)
        dfl_channels = 4 * reg_max
        bins = torch.arange(reg_max, device=teacher_feats[0].device, dtype=teacher_feats[0].dtype).view(1, 1, reg_max, 1, 1)

        loss = teacher_feats[0].new_tensor(0.0)
        total_weight = 0.0
        for i, (sp, tp) in enumerate(zip(student_feats, teacher_feats)):
            b, _, h, w = sp.shape
            sp_dfl = sp[:, :dfl_channels, :, :].view(b, 4, reg_max, h, w)
            tp_dfl = tp[:, :dfl_channels, :, :].view(b, 4, reg_max, h, w)

            sp_prob = F.softmax(sp_dfl, dim=2)
            tp_prob = F.softmax(tp_dfl, dim=2)

            sp_val = (sp_prob * bins).sum(dim=2) / reg_max
            tp_val = (tp_prob * bins).sum(dim=2) / reg_max
            l2_spatial = (sp_val - tp_val).pow(2).mean(dim=1)

            if masks is not None:
                fg = masks[i].squeeze(1)
                loss_level = (l2_spatial * fg).sum() / (fg.sum() + 1e-6)
            else:
                loss_level = l2_spatial.mean()

            w = float(level_weights[i])
            loss = loss + w * loss_level
            total_weight += w

        return loss / max(total_weight, 1e-6)

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        """
        De-scale predictions following augmented inference (inverse operation).

        Args:
            p (torch.Tensor): Predictions tensor.
            flips (int): Flip type (0=none, 2=ud, 3=lr).
            scale (float): Scale factor.
            img_size (tuple): Original image size (height, width).
            dim (int): Dimension to split at.

        Returns:
            (torch.Tensor): De-scaled predictions.
        """
        p[:, :4] /= scale  # de-scale
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y  # de-flip ud
        elif flips == 3:
            x = img_size[1] - x  # de-flip lr
        return torch.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, y):
        """
        Clip YOLO augmented inference tails.

        Args:
            y (List[torch.Tensor]): List of detection tensors.

        Returns:
            (List[torch.Tensor]): Clipped detection tensors.
        """
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4**x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[-1] // g) * sum(4**x for x in range(e))  # indices
        y[0] = y[0][..., :-i]  # large
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][..., i:]  # small
        return y

    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel."""
        return E2EDetectLoss(self) if getattr(self, "end2end", False) else v8DetectionLoss(self)


class Ensemble(torch.nn.ModuleList):
    """
    Ensemble of models.

    This class allows combining multiple YOLO models into an ensemble for improved performance through
    model averaging or other ensemble techniques.

    Methods:
        __init__: Initialize an ensemble of models.
        forward: Generate predictions from all models in the ensemble.

    Examples:
        Create an ensemble of models
        >>> ensemble = Ensemble()
        >>> ensemble.append(model1)
        >>> ensemble.append(model2)
        >>> results = ensemble(image_tensor)
    """

    def __init__(self):
        """Initialize an ensemble of models."""
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        """
        Generate the YOLO network's final layer.

        Args:
            x (torch.Tensor): Input tensor.
            augment (bool): Whether to augment the input.
            profile (bool): Whether to profile the model.
            visualize (bool): Whether to visualize the features.

        Returns:
            y (torch.Tensor): Concatenated predictions from all models.
            train_out (None): Always None for ensemble inference.
        """
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 2)  # nms ensemble, y shape(B, HW, C)
        return y, None  # inference, train output


# Functions ------------------------------------------------------------------------------------------------------------


@contextlib.contextmanager
def temporary_modules(modules=None, attributes=None):
    """
    Context manager for temporarily adding or modifying modules in Python's module cache (`sys.modules`).

    This function can be used to change the module paths during runtime. It's useful when refactoring code,
    where you've moved a module from one location to another, but you still want to support the old import
    paths for backwards compatibility.

    Args:
        modules (dict, optional): A dictionary mapping old module paths to new module paths.
        attributes (dict, optional): A dictionary mapping old module attributes to new module attributes.

    Examples:
        >>> with temporary_modules({"old.module": "new.module"}, {"old.module.attribute": "new.module.attribute"}):
        >>> import old.module  # this will now import new.module
        >>> from old.module import attribute  # this will now import new.module.attribute

    Note:
        The changes are only in effect inside the context manager and are undone once the context manager exits.
        Be aware that directly manipulating `sys.modules` can lead to unpredictable results, especially in larger
        applications or libraries. Use this function with caution.
    """
    if modules is None:
        modules = {}
    if attributes is None:
        attributes = {}
    import sys
    from importlib import import_module

    try:
        # Set attributes in sys.modules under their old name
        for old, new in attributes.items():
            old_module, old_attr = old.rsplit(".", 1)
            new_module, new_attr = new.rsplit(".", 1)
            setattr(import_module(old_module), old_attr, getattr(import_module(new_module), new_attr))

        # Set modules in sys.modules under their old name
        for old, new in modules.items():
            sys.modules[old] = import_module(new)

        yield
    finally:
        # Remove the temporary module paths
        for old in modules:
            if old in sys.modules:
                del sys.modules[old]


class SafeClass:
    """A placeholder class to replace unknown classes during unpickling."""

    def __init__(self, *args, **kwargs):
        """Initialize SafeClass instance, ignoring all arguments."""
        pass

    def __call__(self, *args, **kwargs):
        """Run SafeClass instance, ignoring all arguments."""
        pass


class SafeUnpickler(pickle.Unpickler):
    """Custom Unpickler that replaces unknown classes with SafeClass."""

    def find_class(self, module, name):
        """
        Attempt to find a class, returning SafeClass if not among safe modules.

        Args:
            module (str): Module name.
            name (str): Class name.

        Returns:
            (type): Found class or SafeClass.
        """
        safe_modules = (
            "torch",
            "collections",
            "collections.abc",
            "builtins",
            "math",
            "numpy",
            # Add other modules considered safe
        )
        if module in safe_modules:
            return super().find_class(module, name)
        else:
            return SafeClass


def torch_safe_load(weight, safe_only=False):
    """
    Attempt to load a PyTorch model with the torch.load() function. If a ModuleNotFoundError is raised, it catches the
    error, logs a warning message, and attempts to install the missing module via the check_requirements() function.
    After installation, the function again attempts to load the model using torch.load().

    Args:
        weight (str): The file path of the PyTorch model.
        safe_only (bool): If True, replace unknown classes with SafeClass during loading.

    Returns:
        ckpt (dict): The loaded model checkpoint.
        file (str): The loaded filename.

    Examples:
        >>> from ultralytics.nn.tasks import torch_safe_load
        >>> ckpt, file = torch_safe_load("path/to/best.pt", safe_only=True)
    """
    from ultralytics.utils.downloads import attempt_download_asset

    check_suffix(file=weight, suffix=".pt")
    file = attempt_download_asset(weight)  # search online if missing locally
    try:
        with temporary_modules(
            modules={
                "ultralytics.yolo.utils": "ultralytics.utils",
                "ultralytics.yolo.v8": "ultralytics.models.yolo",
                "ultralytics.yolo.data": "ultralytics.data",
            },
            attributes={
                "ultralytics.nn.modules.block.Silence": "torch.nn.Identity",  # YOLOv9e
                "ultralytics.nn.tasks.YOLOv10DetectionModel": "ultralytics.nn.tasks.DetectionModel",  # YOLOv10
                "ultralytics.utils.loss.v10DetectLoss": "ultralytics.utils.loss.E2EDetectLoss",  # YOLOv10
            },
        ):
            if safe_only:
                # Load via custom pickle module
                safe_pickle = types.ModuleType("safe_pickle")
                safe_pickle.Unpickler = SafeUnpickler
                safe_pickle.load = lambda file_obj: SafeUnpickler(file_obj).load()
                with open(file, "rb") as f:
                    ckpt = torch_load(f, pickle_module=safe_pickle)
            else:
                ckpt = torch_load(file, map_location="cpu")

    except ModuleNotFoundError as e:  # e.name is missing module name
        if e.name == "models":
            raise TypeError(
                emojis(
                    f"ERROR ❌️ {weight} appears to be an Ultralytics YOLOv5 model originally trained "
                    f"with https://github.com/ultralytics/yolov5.\nThis model is NOT forwards compatible with "
                    f"YOLOv8 at https://github.com/ultralytics/ultralytics."
                    f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
                    f"run a command with an official Ultralytics model, i.e. 'yolo predict model=yolo11n.pt'"
                )
            ) from e
        elif e.name == "numpy._core":
            raise ModuleNotFoundError(
                emojis(
                    f"ERROR ❌️ {weight} requires numpy>=1.26.1, however numpy=={__import__('numpy').__version__} is installed."
                )
            ) from e
        LOGGER.warning(
            f"{weight} appears to require '{e.name}', which is not in Ultralytics requirements."
            f"\nAutoInstall will run now for '{e.name}' but this feature will be removed in the future."
            f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
            f"run a command with an official Ultralytics model, i.e. 'yolo predict model=yolo11n.pt'"
        )
        check_requirements(e.name)  # install missing module
        ckpt = torch_load(file, map_location="cpu")

    if not isinstance(ckpt, dict):
        # File is likely a YOLO instance saved with i.e. torch.save(model, "saved_model.pt")
        LOGGER.warning(
            f"The file '{weight}' appears to be improperly saved or formatted. "
            f"For optimal results, use model.save('filename.pt') to correctly save YOLO models."
        )
        ckpt = {"model": ckpt.model}

    return ckpt, file


def attempt_load_weights(weights, device=None, inplace=True, fuse=False):
    """
    Load an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a.

    Args:
        weights (str | List[str]): Model weights path(s).
        device (torch.device, optional): Device to load model to.
        inplace (bool): Whether to do inplace operations.
        fuse (bool): Whether to fuse model.

    Returns:
        (torch.nn.Module): Loaded model.
    """
    ensemble = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt, w = torch_safe_load(w)  # load ckpt
        args = {**DEFAULT_CFG_DICT, **ckpt["train_args"]} if "train_args" in ckpt else None  # combined args
        model = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 model

        # Model compatibility updates
        model.args = args  # attach args to model
        model.pt_path = w  # attach *.pt file path to model
        model.task = getattr(model, "task", guess_model_task(model))
        if not hasattr(model, "stride"):
            model.stride = torch.tensor([32.0])

        # Append
        ensemble.append(model.fuse().eval() if fuse and hasattr(model, "fuse") else model.eval())  # model in eval mode

    # Module updates
    for m in ensemble.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, torch.nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model
    if len(ensemble) == 1:
        return ensemble[-1]

    # Return ensemble
    LOGGER.info(f"Ensemble created with {weights}\n")
    for k in "names", "nc", "yaml":
        setattr(ensemble, k, getattr(ensemble[0], k))
    ensemble.stride = ensemble[int(torch.argmax(torch.tensor([m.stride.max() for m in ensemble])))].stride
    assert all(ensemble[0].nc == m.nc for m in ensemble), f"Models differ in class counts {[m.nc for m in ensemble]}"
    return ensemble


def attempt_load_one_weight(weight, device=None, inplace=True, fuse=False):
    """
    Load a single model weights.

    Args:
        weight (str): Model weight path.
        device (torch.device, optional): Device to load model to.
        inplace (bool): Whether to do inplace operations.
        fuse (bool): Whether to fuse model.

    Returns:
        model (torch.nn.Module): Loaded model.
        ckpt (dict): Model checkpoint dictionary.
    """
    ckpt, weight = torch_safe_load(weight)  # load ckpt
    args = {**DEFAULT_CFG_DICT, **(ckpt.get("train_args", {}))}  # combine model and default args, preferring model args
    model = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 model

    # Model compatibility updates
    model.args = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # attach args to model
    model.pt_path = weight  # attach *.pt file path to model
    model.task = getattr(model, "task", guess_model_task(model))
    if not hasattr(model, "stride"):
        model.stride = torch.tensor([32.0])

    model = model.fuse().eval() if fuse and hasattr(model, "fuse") else model.eval()  # model in eval mode

    # Module updates
    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, torch.nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model and ckpt
    return model, ckpt


def parse_model(d, ch, verbose=True):
    """
    Parse a YOLO model.yaml dictionary into a PyTorch model.

    Args:
        d (dict): Model dictionary.
        ch (int): Input channels.
        verbose (bool): Whether to print model details.

    Returns:
        model (torch.nn.Sequential): PyTorch model.
        save (list): Sorted list of output layers.
    """
    import ast

    # Args
    legacy = True  # backward compatibility for v3/v5/v8/v9 models
    max_channels = float("inf")
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    if scales:
        scale = d.get("scale")
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = torch.nn.SiLU()
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # print

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    base_modules = frozenset(
        {
            Classify,
            Conv,
            ConvTranspose,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            C2fPSA,
            C2PSA,
            DWConv,
            Focus,
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C3k2,
            RepNCSPELAN4,
            ELAN1,
            ADown,
            AConv,
            SPPELAN,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            torch.nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
            RepC3,
            PSA,
            SCDown,
            C2fCIB,
            A2C2f,
        }
    )
    repeat_modules = frozenset(  # modules with 'repeat' arguments
        {
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C3k2,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            C3x,
            RepC3,
            C2fPSA,
            C2fCIB,
            C2PSA,
            A2C2f,
        }
    )
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        m = (
            getattr(torch.nn, m[3:])
            if "nn." in m
            else getattr(__import__("torchvision").ops, m[16:])
            if "torchvision.ops." in m
            else globals()[m]
        )  # get module
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        if m in base_modules:
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            if m is C2fAttn:  # set 1) embed channels and 2) num heads
                args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)
                args[2] = int(max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2])

            args = [c1, c2, *args[1:]]
            if m in repeat_modules:
                args.insert(2, n)  # number of repeats
                n = 1
            if m is C3k2:  # for M/L/X sizes
                legacy = False
                if scale in "mlx":
                    args[3] = True
            if m is A2C2f:
                legacy = False
                if scale in "lx":  # for L/X sizes
                    args.extend((True, 1.2))
            if m is C2fCIB:
                legacy = False
        elif m is AIFI:
            args = [ch[f], *args]
        elif m in frozenset({HGStem, HGBlock}):
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)  # number of repeats
                n = 1
        elif m is ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4
        elif m is torch.nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in frozenset(
            {Detect, WorldDetect, YOLOEDetect, Segment, YOLOESegment, Pose, OBB, ImagePoolingAttn, v10Detect}
        ):
            args.append([ch[x] for x in f])
            if m is Segment or m is YOLOESegment:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
            if m in {Detect, YOLOEDetect, Segment, YOLOESegment, Pose, OBB}:
                m.legacy = legacy
        elif m is RTDETRDecoder:  # special case, channels arg must be passed in index 1
            args.insert(1, [ch[x] for x in f])
        elif m is CBLinear:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        elif m is CBFuse:
            c2 = ch[f[-1]]
        elif m in frozenset({TorchVision, Index}):
            c2 = args[0]
            c1 = ch[f]
            args = [*args[1:]]
        else:
            c2 = ch[f]

        m_ = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        m_.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        if verbose:
            LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}{m_.np:10.0f}  {t:<45}{str(args):<30}")  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return torch.nn.Sequential(*layers), sorted(save)


def yaml_model_load(path):
    """
    Load a YOLOv8 model from a YAML file.

    Args:
        path (str | Path): Path to the YAML file.

    Returns:
        (dict): Model dictionary.
    """
    path = Path(path)
    if path.stem in (f"yolov{d}{x}6" for x in "nsmlx" for d in (5, 8)):
        new_stem = re.sub(r"(\d+)([nslmx])6(.+)?$", r"\1\2-p6\3", path.stem)
        LOGGER.warning(f"Ultralytics YOLO P6 models now use -p6 suffix. Renaming {path.stem} to {new_stem}.")
        path = path.with_name(new_stem + path.suffix)

    unified_path = re.sub(r"(\d+)([nslmx])(.+)?$", r"\1\3", str(path))  # i.e. yolov8x.yaml -> yolov8.yaml
    yaml_file = check_yaml(unified_path, hard=False) or check_yaml(path)
    d = YAML.load(yaml_file)  # model dict
    d["scale"] = guess_model_scale(path)
    d["yaml_file"] = str(path)
    return d


def guess_model_scale(model_path):
    """
    Extract the size character n, s, m, l, or x of the model's scale from the model path.

    Args:
        model_path (str | Path): The path to the YOLO model's YAML file.

    Returns:
        (str): The size character of the model's scale (n, s, m, l, or x).
    """
    try:
        return re.search(r"yolo(e-)?[v]?\d+([nslmx])", Path(model_path).stem).group(2)  # noqa
    except AttributeError:
        return ""


def guess_model_task(model):
    """
    Guess the task of a PyTorch model from its architecture or configuration.

    Args:
        model (torch.nn.Module | dict): PyTorch model or model configuration in YAML format.

    Returns:
        (str): Task of the model ('detect', 'segment', 'classify', 'pose', 'obb').
    """

    def cfg2task(cfg):
        """Guess from YAML dictionary."""
        m = cfg["head"][-1][-2].lower()  # output module name
        if m in {"classify", "classifier", "cls", "fc"}:
            return "classify"
        if "detect" in m:
            return "detect"
        if "segment" in m:
            return "segment"
        if m == "pose":
            return "pose"
        if m == "obb":
            return "obb"

    # Guess from model cfg
    if isinstance(model, dict):
        with contextlib.suppress(Exception):
            return cfg2task(model)
    # Guess from PyTorch model
    if isinstance(model, torch.nn.Module):  # PyTorch model
        for x in "model.args", "model.model.args", "model.model.model.args":
            with contextlib.suppress(Exception):
                return eval(x)["task"]
        for x in "model.yaml", "model.model.yaml", "model.model.model.yaml":
            with contextlib.suppress(Exception):
                return cfg2task(eval(x))
        for m in model.modules():
            if isinstance(m, (Segment, YOLOESegment)):
                return "segment"
            elif isinstance(m, Classify):
                return "classify"
            elif isinstance(m, Pose):
                return "pose"
            elif isinstance(m, OBB):
                return "obb"
            elif isinstance(m, (Detect, WorldDetect, YOLOEDetect, v10Detect)):
                return "detect"

    # Guess from model filename
    if isinstance(model, (str, Path)):
        model = Path(model)
        if "-seg" in model.stem or "segment" in model.parts:
            return "segment"
        elif "-cls" in model.stem or "classify" in model.parts:
            return "classify"
        elif "-pose" in model.stem or "pose" in model.parts:
            return "pose"
        elif "-obb" in model.stem or "obb" in model.parts:
            return "obb"
        elif "detect" in model.parts:
            return "detect"

    # Unable to determine task from model
    LOGGER.warning(
        "Unable to automatically guess model task, assuming 'task=detect'. "
        "Explicitly define task for your model, i.e. 'task=detect', 'segment', 'classify','pose' or 'obb'."
    )
    return "detect"  # assume detect
