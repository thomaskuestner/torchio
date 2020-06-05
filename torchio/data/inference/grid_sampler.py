import copy
from typing import Tuple

import torch
import numpy as np
from torch.utils.data import Dataset

from ...utils import to_tuple
from ...torchio import LOCATION, TypeTuple, DATA, TypeTripletInt
from ..subject import Subject


class GridSampler(Dataset):
    r"""Extract patches across a whole volume.

    Grid samplers are useful to perform inference using all patches from a
    volume. It is often used with a :py:class:`~torchio.data.GridAggregator`.

    Args:
        sample: Instance of :py:class:`~torchio.data.subject.Subject`
            from which patches will be extracted.
        patch_size: Tuple of integers :math:`(d, h, w)` to generate patches
            of size :math:`d \times h \times w`.
            If a single number :math:`n` is provided,
            :math:`d = h = w = n`.
        patch_overlap: Tuple of integers :math:`(d_o, h_o, w_o)` specifying the
            overlap between patches for dense inference. If a single number
            :math:`n` is provided, :math:`d_o = h_o = w_o = n`.

    .. note:: Adapted from NiftyNet. See `this NiftyNet tutorial
        <https://niftynet.readthedocs.io/en/dev/window_sizes.html>`_ for more
        information.
    """
    def __init__(
            self,
            sample: Subject,
            patch_size: TypeTuple,
            patch_overlap: TypeTuple,
            ):
        self.sample = sample
        patch_size = to_tuple(patch_size, length=3)
        patch_overlap = to_tuple(patch_overlap, length=3)
        sizes = self.sample.spatial_shape, patch_size, patch_overlap
        self.parse_sizes(*sizes)
        self.locations = self.get_patches_locations(*sizes)

    def __len__(self):
        return len(self.locations)

    def __getitem__(self, index):
        # Assume 3D
        location = self.locations[index]
        index_ini = location[:3]
        index_fin = location[3:]
        cropped_sample = self.extract_patch(self.sample, index_ini, index_fin)
        cropped_sample[LOCATION] = location
        return cropped_sample

    @staticmethod
    def parse_sizes(
            image_size: TypeTripletInt,
            patch_size: TypeTripletInt,
            patch_overlap: TypeTripletInt,
            ) -> None:
        image_size = np.array(image_size)
        patch_size = np.array(patch_size)
        patch_overlap = np.array(patch_overlap)
        if np.any(patch_size > image_size):
            message = (
                f'Patch size {tuple(patch_size)} cannot be'
                f' larger than image size {tuple(image_size)}'
            )
            raise ValueError(message)
        if np.any(patch_overlap >= patch_size):
            message = (
                f'Patch overlap {tuple(patch_overlap)} must be smaller'
                f' larger than patch size {tuple(image_size)}'
            )
            raise ValueError(message)

    def extract_patch(
            self,
            sample: Subject,
            index_ini: TypeTripletInt,
            index_fin: TypeTripletInt,
            ) -> Subject:
        cropped_sample = self.copy_and_crop(
            sample,
            index_ini,
            index_fin,
        )
        return cropped_sample

    @staticmethod
    def copy_and_crop(
            sample: Subject,
            index_ini: np.ndarray,
            index_fin: np.ndarray,
            ) -> dict:
        cropped_sample = {}
        iterable = sample.get_images_dict(intensity_only=False).items()
        for image_name, image in iterable:
            cropped_sample[image_name] = copy.deepcopy(image)
            sample_image_dict = image
            cropped_image_dict = cropped_sample[image_name]
            cropped_image_dict[DATA] = crop(
                sample_image_dict[DATA], index_ini, index_fin)
        # torch doesn't like uint16
        cropped_sample['index_ini'] = index_ini.astype(int)
        return cropped_sample

    @staticmethod
    def get_patches_locations(
            image_size: TypeTripletInt,
            patch_size: TypeTripletInt,
            patch_overlap: TypeTripletInt,
            ) -> np.ndarray:
        indices = []
        zipped = zip(image_size, patch_size, patch_overlap)
        for im_size_dim, patch_size_dim, patch_overlap_dim in zipped:
            end = im_size_dim + 1 - patch_size_dim
            step = patch_size_dim - patch_overlap_dim
            indices_dim = list(range(0, end, step))
            if im_size_dim % step:
                indices_dim.append(im_size_dim - patch_size_dim)
            indices.append(indices_dim)
        indices_ini = np.array(np.meshgrid(*indices)).reshape(3, -1).T
        indices_ini = np.unique(indices_ini, axis=0)
        indices_fin = indices_ini + np.array(patch_size)
        locations = np.hstack((indices_ini, indices_fin))
        return np.array(sorted(locations.tolist()))


def crop(
        image: torch.Tensor,
        index_ini: np.ndarray,
        index_fin: np.ndarray,
        ) -> torch.Tensor:
    i_ini, j_ini, k_ini = index_ini
    i_fin, j_fin, k_fin = index_fin
    return image[..., i_ini:i_fin, j_ini:j_fin, k_ini:k_fin]
