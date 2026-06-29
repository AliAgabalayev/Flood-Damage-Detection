from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np


ChannelBuilder = Callable[[np.ndarray], np.ndarray]

ClipRange = Tuple[float, float]

def build_vv(image: np.ndarray) -> np.ndarray:
    return image[0].astype(np.float32)


def build_vh(image: np.ndarray) -> np.ndarray:
    return image[1].astype(np.float32)


def build_ratio(image: np.ndarray) -> np.ndarray:
    vv = image[0].astype(np.float32)
    vh = image[1].astype(np.float32)
    epsilon: float = 1e-8
    return vv / (vh + epsilon)



class Preprocessor:

    def __init__(self, channel_specs: List[Tuple[ChannelBuilder, ClipRange]]) -> None:
        if not channel_specs:
            raise ValueError(
                "Preprocessor requires at least one channel spec, but got an empty list."
            )
        self._specs: List[Tuple[ChannelBuilder, ClipRange]] = channel_specs



    @property
    def num_channels(self) -> int:
        # Number of output channels produced by this preprocessor
        return len(self._specs)

    def __call__(self, raw_image: np.ndarray) -> np.ndarray:
        planes: List[np.ndarray] = []
        for builder, clip_range in self._specs:
            plane = builder(raw_image)                          # (H, W)
            plane = self._clip_and_scale(plane, clip_range)    # (H, W) in [0, 1]
            planes.append(plane)

        return np.stack(planes, axis=0).astype(np.float32)     # (N, H, W)

   
    @staticmethod
    def _clip_and_scale(plane: np.ndarray, clip_range: ClipRange) -> np.ndarray:
        lo, hi = clip_range
        clipped = np.clip(plane, lo, hi)
        span = hi - lo if (hi - lo) != 0.0 else 1.0
        return ((clipped - lo) / span).astype(np.float32)




def default_preprocessor(config: object) -> Preprocessor:
    #Build the canonical ``[VV, VH, Ratio]`` preprocessor from a config object.
    data_cfg = config.data 
    specs: List[Tuple[ChannelBuilder, ClipRange]] = [
        (build_vv,    tuple(data_cfg.vv_clip)),     # channel 0
        (build_vh,    tuple(data_cfg.vh_clip)),     # channel 1
        (build_ratio, tuple(data_cfg.ratio_clip)),  # channel 2
    ]
    return Preprocessor(specs) 
