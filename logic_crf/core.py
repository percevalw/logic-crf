from typing import Sequence, Mapping, Tuple

import numpy as np
import torch


class Factor(torch.nn.Module):
    def __init__(self, shape, names, mapping=None):
        """
        Parameters
        ----------
        names
        """
        super().__init__()
        self.names = names
        # self.apparent_names = [var for (subvars, _) in mapping.values() for var in subvars]
        self.mapping = mapping  # type: Mapping[str, Tuple[Sequence[str], torch.Tensor]]
        self.shape = shape

    def __repr__(self):
        return "{}({})".format(
            self.__class__.__name__,
            ", ".join(map(str, self.names)),
        )

    def clone(self):
        raise NotImplementedError()

    def permute(self, names):
        raise NotImplementedError()

    def align_to(self, names):
        return self.permute(names)

    def forward(self, arg, **kwargs):
        """

        Example
        -------

        Parameters
        ----------
        arg: torch.Tensor
        kwargs: Any

        Returns
        -------
        Factor or torch.Tensor
        """
        pass

    def factorize(self):
        return [self]


def as_numpy_array(array):
    if isinstance(array, np.ndarray):
        return array
    elif hasattr(array, 'toarray'):
        return array.toarray()
    elif torch.is_tensor(array):
        return array.cpu().numpy()
    else:
        return np.asarray(array)
