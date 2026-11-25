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

SEGFORMER_CHECKPOINTS = {
    "segformer_b2": "nvidia/mit-b2",
    "segformer_b4": "nvidia/mit-b4",
}


class Segformer(nn.Module):
    def __init__(
        self, checkpoint: str, in_channels: int, out_classes: int, pretrained: bool,
        *, checkpoint_restore: bool = False,
    ):
        super().__init__()
        if checkpoint_restore:
            # NVIDIA MiT-B2/B4 share channel sizes and differ in stage depths.
            # A complete Lightning checkpoint supplies every tensor, so startup
            # can build the architecture locally without loading ImageNet weights.
            depths = [3, 4, 6, 3] if checkpoint == "nvidia/mit-b2" else [3, 8, 27, 3]
            config = SegformerConfig(
                num_channels=in_channels,
                num_labels=out_classes,
                depths=depths,
                hidden_sizes=[64, 128, 320, 512],
                decoder_hidden_size=768,
            )
            self.net = SegformerForSemanticSegmentation(config)
        elif pretrained:
            self.net = SegformerForSemanticSegmentation.from_pretrained(
                checkpoint,
                num_channels=in_channels,
                num_labels=out_classes,
                ignore_mismatched_sizes=True,
            )
        else:
            config = SegformerConfig.from_pretrained(
                checkpoint,
                num_channels=in_channels,
                num_labels=out_classes,
            )
            self.net = SegformerForSemanticSegmentation(config)

    def forward(self, x: Tensor) -> Tensor:
        logits = self.net(pixel_values=x).logits
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


def build_model(cfg: Config, *, checkpoint_restore: bool = False) -> nn.Module:
    arch = cfg.model.arch
    if arch in SEGFORMER_CHECKPOINTS:
        return Segformer(
            checkpoint=SEGFORMER_CHECKPOINTS[arch],
            in_channels=cfg.model.in_channels,
            out_classes=cfg.model.out_classes,
            pretrained=cfg.model.pretrained,
            checkpoint_restore=checkpoint_restore,
        )
    if arch not in ARCHS:
        raise ValueError(arch)
    weights = "imagenet" if cfg.model.pretrained and not checkpoint_restore else None
    return ARCHS[arch](
        encoder_name=cfg.model.backbone,
        encoder_weights=weights,
        in_channels=cfg.model.in_channels,
        classes=cfg.model.out_classes,
    )
