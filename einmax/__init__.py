import torch


def einsum(subscripts, *operands):
    # Checks have already been handled
    einsum_str = subscripts
    input_str, results_index = einsum_str.split('->')
    input_left, input_right = input_str.split(',')
    idx_rm = "".join(sorted((set(input_left) | set(input_right)) - set(results_index)))
    joint_result_index = results_index + idx_rm
    joint_result = torch.einsum(input_str + "->" + joint_result_index, *(o[0] for o in operands))
    removed_shape = joint_result.shape[len(results_index):]
    joint_result = joint_result.reshape((*joint_result.shape[:len(results_index)], -1))
    scores, best_indices = joint_result.max(-1)
    if len(results_index):
        scores = scores.view(*joint_result.shape[:len(results_index)])
        best_indices = best_indices.view(*joint_result.shape[:len(results_index)])

    split_best_indices = []
    for size in reversed(removed_shape):
        split_best_indices.insert(0, best_indices % size)
        best_indices = best_indices // size

    # backtrack = {k: v for o in operands if o[1] is not None for k, v in o[1].items()}
    # for letter, best_indices in zip(idx_rm, split_best_indices):
    #     backtrack[letter] = (best_indices, results_index)
    return scores, [(results_index, tuple(zip(idx_rm, split_best_indices))), *(sub for o in operands if o[1] is not None for sub in o[1])]


def transpose(a, axes):
    return a[0].permute(*axes), a[1].permute(*axes)
