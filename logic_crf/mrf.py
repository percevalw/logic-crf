from itertools import chain

import numpy as np
import opt_einsum as oe
import pandas as pd
import torch

from logic_crf.utils import Clique, factorize


def logsumexp(expr, operands, no_trick=False):
    # a bit radical logsumexp trick to avoid overflow, we take that global max of each tensor
    # we could instead take the max of each sample in the [n_batch, ...] * [n_batch, ...] ... case
    if isinstance(expr, str):
        expr_str = expr
        expr = lambda *args, **kwargs: oe.contract(expr_str, *args, **kwargs)

    if no_trick:
        return expr(*(o.exp() for o in operands)).log()

    operands_max = [op.max() for op in operands]
    return expr(*((op - op_max).exp() for op, op_max in zip(operands, operands_max))).log() + sum(operands_max)


def logmaxexp(expr, operands, no_trick=False):
    # a bit radical logsumexp trick to avoid overflow, we take that global max of each tensor
    # we could instead take the max of each sample in the [n_batch, ...] * [n_batch, ...] ... case
    if isinstance(expr, str):
        expr_str = expr
        expr = lambda *args, **kwargs: oe.contract(expr_str, *args, **kwargs)

    if no_trick:
        scores, argmax = expr(*((o.exp(), None) for o in operands), backend="einmax")
        return scores.log(), argmax

    operands_max = [op.max() for op in operands]
    scores, argmax = expr(*(((op - op_max).exp(), None) for op, op_max in zip(operands, operands_max)), backend="einmax")
    return scores.log() + sum(operands_max), argmax


class MRF:
    def __init__(self, tensors, tensors_scheme, dims_scheme, contract=True, crf=None):
        if contract:
            expressions, tensors_scheme = self.make_cliques_contraction(
                tensors_scheme,
                tensors,
            )
            tensors = [logsumexp(expr, [tensors[i] for i in expr_inputs]) for expr, expr_inputs, _ in expressions]
        self.tensors = tensors
        self.dims_scheme = dims_scheme  # [(states, [3, 4, 5])]
        self.tensors_scheme = tensors_scheme  # [([0, 1, 3, 5])]
        self.crf = crf

    def make_cliques_contraction(self, new_tensors_scheme, tensors, old_tensors_scheme=None):
        if old_tensors_scheme is None:
            old_tensors_scheme = new_tensors_scheme
        inverse, new_tensors_scheme = zip(*sorted(enumerate(new_tensors_scheme), key=lambda x: -len(x[1])))
        tensor_clique_idx, cliques = pd.factorize(list(
            map(Clique, new_tensors_scheme),
        ))

        tensor_clique_idx = tensor_clique_idx[np.argsort(inverse)]
        expressions = []
        new_tensors_scheme = []
        for i, clique in enumerate(cliques):
            clique = tuple(sorted(clique.items))
            tensor_indices = np.flatnonzero(tensor_clique_idx == i)
            # print([old_tensors_scheme[i] for i in tensor_indices], "=>", clique)
            eq = ",".join("".join(oe.get_symbol(d) for d in old_tensors_scheme[i]) for i in tensor_indices)
            eq += "->" + "".join(oe.get_symbol(d) for d in clique)
            # print(eq, [tuple(tensors[i].shape) for i in tensor_indices])
            expressions.append((oe.contract_expression(eq, *(tensors[i].shape for i in tensor_indices), optimize="greedy", use_blas=False), tensor_indices, clique))
            new_tensors_scheme.append(clique)
        return expressions, new_tensors_scheme

    def _pre_run_op(self, dim=None, except_dim=None):

        assert dim is None or except_dim is None
        if dim is not None:
            except_dim = [i for i in range(len(self.dims_scheme)) if i not in dim]
        elif isinstance(except_dim, (int, str)):
            except_dim = [except_dim]
        if except_dim is None:
            except_dim = []

        hashed = hash((tuple(self.tensors_scheme), tuple(self.dims_scheme), tuple(except_dim)))
        cached = self.crf.cached_executions.get(hashed, None)
        if cached is not None:
            return cached

        remaining_old_dims = []
        remaining_states = []
        remaining_indices_in_input = []
        for dim, (states, indices_in_input) in enumerate(self.dims_scheme):
            if all(i in except_dim for i in indices_in_input):
                remaining_indices_in_input.append(indices_in_input)
                remaining_states.append(states)
                remaining_old_dims.append(dim)
        if len(remaining_indices_in_input) > 0:
            new_dims_scheme = list(zip(remaining_states, factorize(remaining_indices_in_input)[0]))
        else:
            new_dims_scheme = []
        new_tensors_scheme, _, old_dims = factorize(self.tensors_scheme, reference_values=remaining_old_dims, freeze_reference=True)
        old_dims = np.asarray(old_dims)

        expressions, new_tensors_scheme = self.make_cliques_contraction(
            [old_dims[t] for t in new_tensors_scheme],
            self.tensors,
            self.tensors_scheme,
        )
        self.crf.cached_executions[hashed] = expressions, new_tensors_scheme, new_dims_scheme
        return expressions, new_tensors_scheme, new_dims_scheme

    def map_observations(self, observations, packed=False):
        # 1) Compute mapped observations from variables observations
        if packed:
            batch_dims, n_vars = (observations.shape[0],), observations.shape[1]
        else:
            batch_dims, n_vars = observations[0].shape, len(observations)

        new_dims_scheme = []
        remaining_indices_in_input = []
        remaining_states = []
        observation_indexer = []
        remaining_old_dims = []
        for dim, (states, indices_in_input) in enumerate(self.dims_scheme):
            if all(i >= n_vars for i in indices_in_input):
                remaining_indices_in_input.append(np.asarray(indices_in_input))
                remaining_states.append(states)
                remaining_old_dims.append(dim)
                observation_indexer.append((dim, slice(None)))
                continue
            if states is not None:
                observation_indexer.append((dim, indices_in_input))
            else:
                observation_indexer.append((dim, indices_in_input[0]))
        batch_dims = tuple(range(len(batch_dims)))
        if len(remaining_indices_in_input) > 0:
            new_dims_scheme = [(None, [i]) for i in batch_dims] + [
                (states, list(indices_in_input + len(batch_dims))) for states, indices_in_input in zip(remaining_states, factorize(remaining_indices_in_input)[0])
            ]
        else:
            new_dims_scheme = [(None, [i]) for i in batch_dims]
        # remaining_old_dims = remaining_old_dims
        new_tensors_scheme = [batch_dims + tuple(d + len(batch_dims) for d in dims)
                              for dims in factorize(self.tensors_scheme, reference_values=remaining_old_dims, freeze_reference=True)[0]]
        return observation_indexer, new_tensors_scheme, new_dims_scheme

    @property
    def shape(self):
        shape = torch.zeros(sum(map(lambda x: len(x[1]), self.dims_scheme)), dtype=torch.long)
        for tensor_dims, tensor in zip(self.tensors_scheme, self.tensors):
            for i, dim in enumerate(tensor_dims):
                states = self.dims_scheme[dim][0]
                for subdim in self.dims_scheme[dim][1]:
                    shape[subdim] = tensor.shape[i] if states is None else 2
        return torch.Size(shape)

    def __getitem__(self, observations):
        packed = not isinstance(observations, tuple)

        hashed = (hash((tuple(self.dims_scheme), tuple(self.tensors_scheme), len(observations), packed)))
        cached = self.crf.cached_executions.get(hashed, None)
        observations_indexer, new_tensors_scheme, new_dims_scheme = self.map_observations(observations, packed=packed) if cached is None else cached[:3]

        new_observations = []
        n_batch_dims = 0
        if packed:
            n_batch_dims = len(observations.shape[:-1])
            for i, n in enumerate(observations.shape[:-1]):
                new_observations.append(torch.arange(n, device=observations.device).long().view(*((1 if i != j else -1 for j in range(n_batch_dims)))))

        for dim, indexer in observations_indexer[n_batch_dims:]:
            states = self.dims_scheme[dim][0]
            if isinstance(indexer, slice):
                new_observations.append(slice(None))
            elif states is None:
                if packed:
                    new_observations.append(observations[..., indexer - n_batch_dims])
                else:
                    new_observations.append(observations[indexer])
            else:
                if packed:
                    mapped_observations = observations[..., tuple(i - n_batch_dims for i in indexer)]
                else:
                    mapped_observations = torch.stack([observations[i] for i in indexer], 1)
                new_observations.append(torch.unique(torch.cat([states, mapped_observations]).long(), return_inverse=True, dim=0)[1][-len(mapped_observations):])
        del observations

        new_tensors = []
        for tensor, tensor_dim_indices in zip(self.tensors, self.tensors_scheme):
            new_tensors.append(tensor[tuple(new_observations[i] for i in tensor_dim_indices)])

        if cached is None:
            expressions, new_tensors_scheme = self.make_cliques_contraction(new_tensors_scheme, tensors=new_tensors)
        else:
            expressions, new_tensors_scheme = cached[3:5]

        if cached is None:
            self.crf.cached_executions[hashed] = observations_indexer, new_tensors_scheme, new_dims_scheme, expressions, new_tensors_scheme

        new_tensors = [logsumexp(expr, [new_tensors[i] for i in expr_inputs]) for expr, expr_inputs, _ in expressions]
        if len(new_tensors_scheme) == 1 and all(new_dims_scheme[i][0] is None for i in new_tensors_scheme[0]):
            return new_tensors[0].permute(tuple(np.argsort(new_tensors_scheme[0])))
        return self.__class__(new_tensors, new_tensors_scheme, new_dims_scheme, crf=self.crf)

    def logsumexp(self, dim=None, except_dim=None):
        expressions, new_tensors_scheme, new_dims_scheme = self._pre_run_op(dim=dim, except_dim=except_dim)

        tensors = [logsumexp(expr, [self.tensors[i] for i in expr_inputs])
                   for expr, expr_inputs, _ in expressions]

        if len(new_tensors_scheme) == 1 and all(new_dims_scheme[i][0] is None for i in new_tensors_scheme[0]):
            return tensors[0].permute(tuple(np.argsort(new_tensors_scheme[0])))
        return self.__class__(tensors, new_tensors_scheme, new_dims_scheme, crf=self.crf)

    def sum(self, dim=None, except_dim=None):
        expressions, new_tensors_scheme, new_dims_scheme = self._pre_run_op(dim=dim, except_dim=except_dim)
        tensors = [expr(*(self.tensors[i] for i in expr_inputs)) for expr, expr_inputs, _ in expressions]
        if len(new_tensors_scheme) == 1 and all(new_dims_scheme[i][0] is None for i in new_tensors_scheme[0]):
            return tensors[0].permute(tuple(np.argsort(new_tensors_scheme[0])))
        return self.__class__(tensors, new_tensors_scheme, new_dims_scheme, crf=self.crf)

    def mean(self, dim=None, except_dim=None):
        assert dim is None or except_dim is None
        if dim is not None:
            except_dim = [i for i in range(len(self.dims_scheme)) if i not in dim]
        elif isinstance(except_dim, (int, str)):
            except_dim = [except_dim]
        if except_dim is None:
            except_dim = []

        shape = self.shape
        result = self.sum()
        result.tensors[0] /= np.prod([size for i, size in enumerate(shape) if i not in except_dim])
        return result

    def max(self, dim=None, except_dim=None):
        device = self.tensors[0].device

        assert dim is None or except_dim is None
        if dim is not None:
            except_dim = [i for i in range(len(self.dims_scheme)) if i not in dim]
        elif isinstance(except_dim, (int, str)):
            except_dim = [except_dim]
        if except_dim is None:
            except_dim = []

        available_keep_dims = [dim for states, subdims in self.dims_scheme if states is None for dim in subdims]
        assert all(dim in available_keep_dims for dim in except_dim)

        expr, expr_inputs, expr_output = self._pre_run_op(except_dim=except_dim)[0][0]
        eq_inputs, eq_output = expr.contraction.split('->')
        marginalized_chars = sorted(set("".join(eq_inputs.split(","))) - set(eq_output))
        batch_chars = sorted(set(eq_output) - set(marginalized_chars))
        n_batch = len(batch_chars)
        batch_slices = [slice(None)] * n_batch
        letter_to_dim = dict(zip(marginalized_chars, range(len(marginalized_chars))))

        # scores, backtrack = expr(*[(self.tensors[i].exp(), None) for i in expr_inputs], backend='einmax')
        scores, backtrack = logmaxexp(expr, [self.tensors[i] for i in expr_inputs])
        argmax = torch.zeros((len(marginalized_chars), *scores.shape), dtype=torch.long, device=device)
        for requires, backpointers in backtrack:
            permutation = [requires.find(s) for s in batch_chars] + [requires.find(s) for s in sorted(set(marginalized_chars) & set(requires))]
            indices = tuple(argmax[letter_to_dim[letter]] for letter in sorted(set(requires) - set(batch_chars)))
            for dest_letter, backpointer in backpointers:
                backpointer = backpointer.permute(*permutation)
                flat_inds = tuple(inds.view(-1) for inds in indices)
                flat_backpointers = backpointer.reshape(-1, *backpointer.shape[len(batch_chars):])
                argmax[letter_to_dim[dest_letter]] = flat_backpointers[(torch.arange(len(flat_backpointers), device=device), *flat_inds)].view(argmax.shape[1:])

        letter_to_original_dim = dict(zip("".join(letters for letters in eq_inputs.split(",")), chain.from_iterable(self.tensors_scheme[i] for i in expr_inputs)))
        column_to_original_dim = [letter_to_original_dim[letter] for letter in marginalized_chars]

        all_dest_cols = [self.dims_scheme[i][1] for i in column_to_original_dim]
        all_dest_cols = factorize(all_dest_cols, reference_values=sorted(chain.from_iterable(all_dest_cols)))[0]

        result = torch.zeros(*argmax.shape[1:], len(set(chain.from_iterable(all_dest_cols))), device=device, dtype=torch.bool)
        for col_idx, original_dim, dest_cols in zip(range(argmax.shape[0]), column_to_original_dim, all_dest_cols):
            states = self.dims_scheme[original_dim][0]
            if states is None:
                result[..., dest_cols] = argmax[[col_idx]]
            else:
                result[..., dest_cols] = states[argmax[col_idx]]

        return scores, result  # [np.argsort([letter_to_dim[letter] for letter in marginalized_chars])]
