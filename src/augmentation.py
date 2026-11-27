"""Fixed online augmentation for artwork classification."""
import math

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F, InterpolationMode


def uniform(low, high):
    return torch.empty(()).uniform_(low, high).item()


def proportional_crop(img):
    w, h = img.size
    fraction = math.sqrt(uniform(0.8, 1.0))
    cw, ch = max(1, round(w * fraction)), max(1, round(h * fraction))
    left = torch.randint(0, w - cw + 1, ()).item()
    top = torch.randint(0, h - ch + 1, ()).item()
    return F.crop(img, top, left, ch, cw)


def affine_no_border(img):
    # Input is square, because Resize runs before this function.
    size = img.width
    angle = uniform(-8, 8) if torch.rand(()) < 0.4 else 0.0
    if torch.rand(()) < 0.3:
        dx = round(uniform(-0.04, 0.04) * size)
        dy = round(uniform(-0.04, 0.04) * size)
    else:
        dx = dy = 0
    if angle == 0 and dx == 0 and dy == 0:
        return img

    # Inverse-map the output corners into the source image.
    # A one-pixel margin avoids sampling outside the image with bilinear interpolation.
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    half = size / 2
    extent = half * (abs(c) + abs(s)) + max(
        abs(c * dx + s * dy), abs(-s * dx + c * dy)
    )
    zoom = max(1.0, extent / (half - 1))
    return F.affine(img, angle, [dx, dy], zoom, [0.0, 0.0],
                    interpolation=InterpolationMode.BILINEAR)


def build_augmentation(img_size):
    if img_size <= 2:
        raise ValueError("Augmentation requires img_size > 2")
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([proportional_crop], p=0.4),
        T.Resize((img_size, img_size)),
        affine_no_border,
        T.RandomApply([
            T.ColorJitter(brightness=0.25, saturation=0.20, contrast=0.25)
        ], p=0.7),
    ])
