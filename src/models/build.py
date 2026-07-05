import segmentation_models_pytorch as smp
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from utils.config import Config

ARCHS = {
    "deeplabv3plus": smp.DeepLabV3Plus,
    "deeplabv3": smp.DeepLabV3,
    "unet": smp.Unet,
    "unetplusplus": smp.UnetPlusPlus,
    "fpn": smp.FPN,
}

SEGFORMER_CHECKPOINT = "nvidia/mit-b2"


class SegformerB2(nn.Module):
    def __init__(self, in_channels: int, out_classes: int, pretrained: bool):
        super().__init__()
        if pretrained:
            self.net = SegformerForSemanticSegmentation.from_pretrained(
                SEGFORMER_CHECKPOINT,
                num_channels=in_channels,
                num_labels=out_classes,
                ignore_mismatched_sizes=True,
            )
        else:
            config = SegformerConfig.from_pretrained(
                SEGFORMER_CHECKPOINT,
                num_channels=in_channels,
                num_labels=out_classes,
            )
            self.net = SegformerForSemanticSegmentation(config)

    def forward(self, x: Tensor) -> Tensor:
        logits = self.net(pixel_values=x).logits
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


def build_model(cfg: Config) -> nn.Module:
    arch = cfg.model.arch
    if arch == "segformer_b2":
        return SegformerB2(
            in_channels=cfg.model.in_channels,
            out_classes=cfg.model.out_classes,
            pretrained=cfg.model.pretrained,
        )
    if arch not in ARCHS:
        raise ValueError(arch)
    weights = "imagenet" if cfg.model.pretrained else None
    return ARCHS[arch](
        encoder_name=cfg.model.backbone,
        encoder_weights=weights,
        in_channels=cfg.model.in_channels,
        classes=cfg.model.out_classes,
    )
