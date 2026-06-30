import segmentation_models_pytorch as smp
from torch import nn

from utils.config import Config

ARCHS = {
    "deeplabv3plus": smp.DeepLabV3Plus,
    "deeplabv3": smp.DeepLabV3,
    "unet": smp.Unet,
    "unetplusplus": smp.UnetPlusPlus,
    "fpn": smp.FPN,
}


def build_model(cfg: Config) -> nn.Module:
    arch = cfg.model.arch
    if arch not in ARCHS:
        raise ValueError(arch)
    weights = "imagenet" if cfg.model.pretrained else None
    return ARCHS[arch](
        encoder_name=cfg.model.backbone,
        encoder_weights=weights,
        in_channels=cfg.model.in_channels,
        classes=cfg.model.out_classes,
    )
