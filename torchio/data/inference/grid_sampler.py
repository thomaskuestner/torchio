import numpy as np
from torch.utils.data import Dataset

from ..sampler.sampler import PatchSampler
from ...utils import to_tuple
from ...torchio import LOCATION, TypeTuple, TypeTripletInt
from ..subject import Subject


class GridSampler(PatchSampler, Dataset):
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
        PatchSampler.__init__(self, patch_size)
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
                f' than patch size {tuple(image_size)}'
            )
            raise ValueError(message)

    def extract_patch(
            self,
            sample: Subject,
            index_ini: TypeTripletInt,
            index_fin: TypeTripletInt,
            ) -> Subject:
        crop = self.get_crop_transform(
            sample.spatial_shape,
            index_ini,
            index_fin - index_ini,
        )
        cropped_sample = crop(sample)
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
