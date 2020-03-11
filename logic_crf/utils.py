import numpy as np
import pandas as pd
import torch


def flatten_array(array, mask=None):
    """
    Flatten array to get the list of active entries
    If not mask is provided, it's just array.view(-1)
    If a mask is given, then it is array[mask]

    Parameters
    ----------
    array: np.ndarray or torch.Tensor
    mask: np.ndarray or torch.Tensor

    Returns
    -------
    np.ndarray or torch.Tensor
    """
    if isinstance(array, (list, tuple)):
        if mask is None:
            return array
        array = np.asarray(array)
    if isinstance(array, np.ndarray):
        if mask is not None:
            if not isinstance(array, np.ndarray):
                raise Exception(f"Mask type {repr(type(mask))} should be the same as array type {repr(type(array))}")
            return array[mask]
        else:
            return array.reshape(-1)
    elif torch.is_tensor(array):
        if mask is not None:
            if not torch.is_tensor(mask):
                raise Exception(f"Mask type {repr(type(mask))} should be the same as array type {repr(type(array))}")
            return array[mask]
        else:
            return array.reshape(-1)
    else:
        raise Exception(f"Unrecognized array type {repr(type(array))} during array flattening (mask type is {repr(type(mask))}')")


def factorize(values, mask=None, reference_values=None, freeze_reference=True, keep_old_values=False):
    """
    Express values in "col" as row numbers in a reference list of values
    The reference values list is the deduplicated concatenation of preferred_unique_values (if not None) and col

    Ex:
    >>> factorize(["A", "B", "C", "D"], None, ["D", "B", "C", "A", "E"])
    ... [3, 2, 1, 0], None, ["D", "B", "C", "A", "E"]
    >>> factorize(["A", "B", "C", "D"], None, None)
    ... [0, 1, 2, 3], None, ["A", "B", "C", "D"]

    Parameters
    ----------
    values: np.ndarray or torch.Tensor or list of (np.ndarray or torch.Tensor) or list of any
        values to factorize
    mask: np.ndarray or torch.Tensor or list of (np.ndarray or torch.Tensor) or None
        optional mask on col, useful for multiple dimension values arrays
    freeze_reference: bool
        Should we throw out values out of reference values (if given).
        Then we need a mask to mark those rows as disabled
        TODO: handle cases when a mask is not given
    reference_values: np.ndarray or torch.Tensor or list or None
        If given, any value in col that is not in prefered_unique_values will be thrown out
        and the mask will be updated to be False for this value

    Returns
    -------
    col, updated mask, reference values
    """
    if isinstance(values, list) and not hasattr(values[0], '__len__'):
        values = np.asarray(values)
    return_as_list = isinstance(values, list)
    all_values = values if isinstance(values, list) else [values]
    del values
    all_masks = mask if isinstance(mask, list) else [None for _ in all_values] if mask is None else [mask]
    del mask

    assert len(all_values) == len(all_masks), "Mask and values lists must have the same length"

    if reference_values is None:
        freeze_reference = False

    all_flat_values = []
    for values, mask in zip(all_values, all_masks):
        assert (
              (isinstance(mask, np.ndarray) and isinstance(values, np.ndarray)) or
              (torch.is_tensor(mask) and torch.is_tensor(values)) or
              (mask is None and (isinstance(values, (list, tuple, np.ndarray)) or torch.is_tensor(values)))), (
            f"values and (optional mask) should be of same type torch.tensor, numpy.ndarray. Given types are values: {repr(type(values))} and mask: {repr(type(mask))}")
        all_flat_values.append(flatten_array(values, mask))
        # return all_values[0], all_masks[0], all_values[0].tocsr().data if hasattr(all_values[0], 'tocsr') else all_values#col.tocsr().data if hasattr(col, 'tocsr')
    if torch.is_tensor(all_flat_values[0]):
        if reference_values is None:
            unique_values, relative_values = torch.unique(torch.cat(all_flat_values), return_inverse=True)
        else:
            relative_values, unique_values = torch.unique(torch.cat((reference_values, *all_flat_values)), return_inverse=True)[1], reference_values
    else:
        if reference_values is None:
            relative_values, unique_values = pd.factorize(np.concatenate(all_flat_values))
        else:
            relative_values, unique_values = pd.factorize(np.concatenate((reference_values, *all_flat_values)))[0], reference_values
    if freeze_reference:
        all_unk_masks = relative_values < len(reference_values)
    else:
        all_unk_masks = None

    offset = len(reference_values) if reference_values is not None else 0
    new_flat_values = []
    new_flat_values = []
    unk_masks = []
    for flat_values in all_flat_values:
        indexer = slice(offset, offset + len(flat_values))
        new_flat_values.append(relative_values[indexer])
        unk_masks.append(all_unk_masks[indexer] if all_unk_masks is not None else None)
        offset = indexer.stop
    all_flat_values = new_flat_values
    del new_flat_values

    if freeze_reference:
        unique_values = unique_values[:len(reference_values)]
    new_values = []
    new_masks = []
    for values, mask, flat_relative_values, unk_mask in zip(all_values, all_masks, all_flat_values, unk_masks):
        if isinstance(values, (list, tuple)):
            mask = unk_mask
            if mask is not None:
                values = [v for v, valid in zip(flat_relative_values, mask) if valid]
                new_values.append(values)
                new_masks.append(None)
            else:
                values = list(flat_relative_values)
                new_values.append(values)
                new_masks.append(None)
        elif isinstance(values, np.ndarray):
            new_mask = mask
            if freeze_reference:
                if mask is None:
                    new_mask = unk_mask.reshape(values.shape)
                else:
                    new_mask = mask & unk_mask.reshape(mask.shape)
            if mask is not None:
                values = np.zeros(values.shape, dtype=int)
                values[mask] = flat_relative_values
                new_values.append(values)
                new_masks.append(new_mask)
            else:
                values = flat_relative_values.reshape(values.shape)
                new_values.append(values)
                new_masks.append(new_mask)
        else:  # torch
            new_mask = mask
            if freeze_reference:
                if mask is None:
                    new_mask = unk_mask.view(*values.shape)
                else:
                    new_mask = mask & unk_mask.view(*mask.shape)
            if mask is not None:
                values = torch.zeros(values.shape, dtype=torch.long)
                values[mask] = flat_relative_values
                new_values.append(values)
                new_masks.append(mask)
            else:
                values = flat_relative_values.view(*values.shape)
                new_values.append(values)
                new_masks.append(new_mask)
    if return_as_list:
        return new_values, new_masks, unique_values
    return new_values[0], new_masks[0], unique_values


class Clique:
    __slots__ = ['items', 'hash']

    def __init__(self, items):
        self.items = set(items)
        self.hash = hash(tuple(sorted(items)))

    def __eq__(self, other):
        return self.items <= other.items or other.items <= self.items

    def __hash__(self):
        return 0

    def __repr__(self):
        return repr(self.items)
