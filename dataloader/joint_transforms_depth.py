import math
import numbers
import random
import warnings
from collections.abc import Sequence
from typing import Tuple, List, Optional

import torch
from torch import Tensor

try:
    import accimage
except ImportError:
    accimage = None

import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode, _interpolation_modes_from_int


__all__ = [
    "Compose",
    "RandomApply",
    "RandomChoice",
    "RandomOrder",
    "RandomCrop",
    "RandomHorizontalFlip",
    "RandomVerticalFlip",
    "RandomSizedCrop",
    "RandomRotation",
    "RandomAffine",
    "RandomPerspective",
    "RandomErasing",
    "InterpolationMode",
]


class Compose:
    """Composes several transforms together. This transform does not support torchscript.
    Please, see the note below.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.PILToTensor(),
        >>>     transforms.ConvertImageDtype(torch.float),
        >>> ])

    .. note::
        In order to script the transformations, please use ``torch.nn.Sequential`` as below.

        >>> transforms = torch.nn.Sequential(
        >>>     transforms.CenterCrop(10),
        >>>     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>> )
        >>> scripted_transforms = torch.jit.script(transforms)

        Make sure to use only scriptable transformations, i.e. that work with ``torch.Tensor``, does not require
        `lambda` functions or ``PIL.Image``.

    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, lbl, disp,  top, t_):
        for t in self.transforms:
            img, lbl, disp, top, t_ = t(img, lbl, disp, top, t_)
        return img, lbl, disp, top, t_

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class RandomDistance(torch.nn.Module):

    @staticmethod
    def get_params(img, center, scale, min_size, max_size, img_size, top):
    
        w, h = F.get_image_size(img)
        
        scale_max = min(scale[1], max_size[0]/h, max_size[1]/w)
        
        #center = top+h//2
        if (top+h-1)<center:
            scale_max = min(center/(center-(top+h-1)),scale_max)
        if top>center:
            scale_max = min((img_size[0]-center-1)/(top-center),scale_max)
            
        scale_min = max(scale[0], max(min_size/h, min_size/w))
        
        
        if scale_max<=scale_min:
            return (h, w), 1
        else:
            scale_factor = 1/torch.empty(1).uniform_(1/scale_max, 1/scale_min).item()

        w *=scale_factor
        h *=scale_factor
        
        h = int(round(h))
        w = int(round(w))

        return (h, w), scale_factor

    def __init__(self,  center, scale, min_size, max_size, img_size, p=0.5, interpolation = InterpolationMode.BILINEAR, antialias = True):
        super().__init__()
        
        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")

        if (scale[0] > scale[1]):
            warnings.warn("Scale should be of kind (min, max)")
        
        self.scale = scale
        self.interpolation = interpolation
        self.antialias = antialias
        self.max_size = max_size
        self.min_size = min_size
        self.img_size = img_size
        #self.K = K
        self.center = center
        self.p = p

    def forward(self, img, lbl, disp, top, t_):
        
        if torch.rand(1) < self.p:
            size, scale = self.get_params(img, self.center, self.scale, self.min_size, self.max_size, self.img_size, top)
            top = int((top-self.center)*scale+self.center)
            disp = disp*scale
            t_dis = torch.Tensor([[scale,0,0],[0,scale,0],[0,0,1]])
            return F.resize(img, size, self.interpolation, antialias = self.antialias), F.resize(lbl, size, InterpolationMode.NEAREST), F.resize(disp, size, self.interpolation), top, torch.mm(t_dis,t_)
        else:
            return img, lbl, disp, top, t_
    def __repr__(self):
        return self.__class__.__name__ + f"(size={self.scale}, interpolation={self.interpolation})"




class RandomHorizontalFlip(torch.nn.Module):
    """Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img, lbl, disp, top, t_):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if torch.rand(1) < self.p:
            t_hflip=torch.Tensor([[-1,0,img.shape[2]],[0,1,0],[0,0,1]])
            return F.hflip(img), F.hflip(lbl), F.hflip(disp), top, torch.mm(t_hflip,t_)
        return img, lbl, disp, top, t_

    def __repr__(self):
        return self.__class__.__name__ + f"(p={self.p})"


class RandomPerspective(torch.nn.Module):
    """Performs a random perspective transformation of the given image with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        distortion_scale (float): argument to control the degree of distortion and ranges from 0 to 1.
            Default is 0.5.
        p (float): probability of the image being transformed. Default is 0.5.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
            For backward compatibility integer values (e.g. ``PIL.Image.NEAREST``) are still acceptable.
        fill (sequence or number): Pixel fill value for the area outside the transformed
            image. Default is ``0``. If given a number, the value is used for all bands respectively.
    """

    def __init__(self, min_size, distortion_scale=0.5, p=0.5, interpolation=InterpolationMode.BILINEAR, fill=0):
        super().__init__()
        self.p = p

        # Backward compatibility with integer value
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument interpolation should be of type InterpolationMode instead of int. "
                "Please, use InterpolationMode enum."
            )
            interpolation = _interpolation_modes_from_int(interpolation)

        self.interpolation = interpolation
        self.distortion_scale = distortion_scale

        if fill is None:
            fill = 0
        elif not isinstance(fill, (Sequence, numbers.Number)):
            raise TypeError("Fill should be either a sequence or a number.")

        self.fill = fill
        self.min_size = min_size

    def forward(self, img, lbl, disp, top):
        """
        Args:
            img (PIL Image or Tensor): Image to be Perspectively transformed.

        Returns:
            PIL Image or Tensor: Randomly transformed image.
        """

        fill = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F.get_image_num_channels(img)
            else:
                fill = [float(f) for f in fill]
                

        width, height = F.get_image_size(img)
        if width > self.min_size and height > self.min_size:
            if torch.rand(1) < self.p:
                startpoints, endpoints = self.get_params(width, height, self.distortion_scale)
                try:
                    return F.perspective(img, startpoints, endpoints, self.interpolation, fill), F.perspective(lbl, startpoints, endpoints, self.interpolation, 0), F.perspective(disp, startpoints, endpoints, self.interpolation, 0), top
                except:
                    return img, lbl, disp, top
        return img, lbl, disp, top

    @staticmethod
    def get_params(width: int, height: int, distortion_scale: float) -> Tuple[List[List[int]], List[List[int]]]:
        """Get parameters for ``perspective`` for a random perspective transform.

        Args:
            width (int): width of the image.
            height (int): height of the image.
            distortion_scale (float): argument to control the degree of distortion and ranges from 0 to 1.

        Returns:
            List containing [top-left, top-right, bottom-right, bottom-left] of the original image,
            List containing [top-left, top-right, bottom-right, bottom-left] of the transformed image.
        """
        half_height = height // 2
        half_width = width // 2
        topleft = [
            int(torch.randint(0, int(distortion_scale * half_width) + 1, size=(1,)).item()),
            int(torch.randint(0, int(distortion_scale * half_height) + 1, size=(1,)).item()),
        ]
        topright = [
            int(torch.randint(width - int(distortion_scale * half_width) - 1, width, size=(1,)).item()),
            int(torch.randint(0, int(distortion_scale * half_height) + 1, size=(1,)).item()),
        ]
        botright = [
            int(torch.randint(width - int(distortion_scale * half_width) - 1, width, size=(1,)).item()),
            int(torch.randint(height - int(distortion_scale * half_height) - 1, height, size=(1,)).item()),
        ]
        botleft = [
            int(torch.randint(0, int(distortion_scale * half_width) + 1, size=(1,)).item()),
            int(torch.randint(height - int(distortion_scale * half_height) - 1, height, size=(1,)).item()),
        ]
        startpoints = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        endpoints = [topleft, topright, botright, botleft]
        return startpoints, endpoints

    def __repr__(self):
        return self.__class__.__name__ + f"(p={self.p})"



def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


def _check_sequence_input(x, name, req_sizes):
    msg = req_sizes[0] if len(req_sizes) < 2 else " or ".join([str(s) for s in req_sizes])
    if not isinstance(x, Sequence):
        raise TypeError(f"{name} should be a sequence of length {msg}.")
    if len(x) not in req_sizes:
        raise ValueError(f"{name} should be sequence of length {msg}.")


def _setup_angle(x, name, req_sizes=(2,)):
    if isinstance(x, numbers.Number):
        if x < 0:
            raise ValueError(f"If {name} is a single number, it must be positive.")
        x = [-x, x]
    else:
        _check_sequence_input(x, name, req_sizes)

    return [float(d) for d in x]
