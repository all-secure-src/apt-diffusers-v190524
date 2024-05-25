import numpy as np
import torch

from ...utils import is_invisible_watermark_available


if is_invisible_watermark_available():
    from imwatermark import WatermarkEncoder

WATERMARK_MESSAGE = 0b110001001111110100010010001110101100110100110100
# bin(x)[2:] gives bits of x as str, use int to convert them to 0/1
WATERMARK_BITS = [int(bit) for bit in bin(WATERMARK_MESSAGE)[2:]]


class OmegaSpectraV2Watermarker:
    def __init__(self):
        self.watermark = WATERMARK_BITS
        self.encoder = WatermarkEncoder()

        self.encoder.set_watermark("bits", self.watermark)

    def apply_watermark(self, images: torch.Tensor):
        # can't encode images that are smaller than 256
        if images.shape[-1] < 256:
            return images

        images = (255 * (images / 2 + 0.5)).cpu().permute(0, 2, 3, 1).float().numpy()

        # Convert RGB to BGR, which is the channel order expected by the watermark encoder.
        images = images[:, :, :, ::-1]

        # Add watermark and convert BGR back to RGB
        images = [self.encoder.encode(image, "dwtDct")[:, :, ::-1] for image in images]

        images = np.array(images)

        images = torch.from_numpy(images).permute(0, 3, 1, 2)

        images = torch.clamp(2 * (images / 255 - 0.5), min=-1.0, max=1.0)
        return images
