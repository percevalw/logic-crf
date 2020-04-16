from itertools import product

import numpy as np
import pandas as pd
import torch

from logic_crf.core import Factor


def can_merge(fac1, fac2):
    if type(fac1) == type(fac2):
        can_merge_fn = fac1.fn.indexers[:-1] == fac2.fn.indexers[:-1]
        can_merge_mask = not ((fac1.mask != 0) & (fac2.mask != 0)).any()
        return can_merge_mask and can_merge_fn
    return False


class MetaIndexer(type):
    def __getitem__(self, item):
        if isinstance(item, tuple):
            return self(*item)
        return self(item)


class Indexer(torch.nn.Module, metaclass=MetaIndexer):
    def __init__(self, *items):
        super().__init__()
        single_until = next((i for i, item in enumerate(items) if not isinstance(item, (str, int))), len(items) + 1)
        self.single_indexers = tuple(items[:single_until])
        self.multi_indexers = tuple(items[single_until:]) if single_until is not None else None
        assert all(isinstance(item, (int, slice, list)) for item in self.multi_indexers)

    @property
    def indexers(self):
        return self.single_indexers + self.multi_indexers

    def forward(self, _arg=None, **kwargs):
        obj = _arg if _arg is not None else kwargs
        for item in self.single_indexers:
            obj = obj[item]
        if self.multi_indexers is not None:
            obj = obj[self.multi_indexers]
        return obj

    @property
    def shape(self):
        return tuple(None if isinstance(item, slice) else len(item) for item in self.multi_indexers if not isinstance(item, int))

    def __repr__(self):
        return "Indexer[{}]".format(", ".join(repr(s) if s != slice(None) else ":" for s in (*self.single_indexers, *self.multi_indexers)))


class HintFactor(Factor):
    def __init__(self, expr, names=None, shape=None, mask=None, fn=None):
        assert not (expr is None and names is None and mask is None) and not (mask is not None and names is None)
        assert (names is None) == (shape is None)
        if names is None:
            names = [lit for lit in expr.support]
            shape = torch.as_tensor([2 for lit in expr.support])
        super().__init__(shape, names)
        self.expr = expr
        self.register_buffer('mask', mask)
        self.fn = fn

    def clone(self):
        return HintFactor(self.expr, self.names, self.shape, self.mask, self.fn)

    def permute(self, names):
        raise NotImplementedError()

    def factorize(self):
        # returns a list of factors that may be rearranged in cliques
        new_factors = []
        for clause in self.expr.to_dnf().children:
            new_factors.append(HintFactor(clause, fn=self.fn))
        return new_factors

    def forward(self, *args, **kwargs):
        potential = self.fn(*args, **kwargs)
        # mask = self.mask
        # return mask.float().view(*(-1 for _ in potential.shape), *mask.shape) * potential.view(*potential.shape, *(-1 for _ in mask.shape))
        return potential[..., self.mask].masked_fill(self.mask == -1, 0)

    def get_states(self, names=None):
        if names is None:
            if self.mask is not None:
                return self.mask.nonzero()
            names = self.names
        expr = self.expr
        valid_assignements = []
        F = expr.any_factory()
        for sat in expr.satisfy_all(lib='pyeda'):
            valid_assignements.append(torch.as_tensor([sat[name] for name in names], dtype=torch.int))
        return torch.unique(torch.stack(valid_assignements), dim=0)

    def change_variables(self, mapping):
        # Assum new_vars are either SuperVariables or Variables and all those scopes are exclusives
        states = self.get_states().int()
        valid_states_per_new_var = []
        new_names = []
        new_shape = []
        changed_vars = set()
        for new_name, subvars, mapping_states, _ in mapping:
            if mapping_states is None:
                continue
            shared_vars = sorted(set(subvars) & set(self.names))
            if shared_vars:
                indices_in_self = pd.factorize(np.asarray([*self.names, *shared_vars]))[0][len(self.names):]
                indices_in_new = pd.factorize(np.asarray([*subvars, *shared_vars]))[0][len(subvars):]
                valid_states_per_new_var.append([
                    vec.nonzero(as_tuple=True)[0].tolist() for vec in tuple(
                        (mapping_states.int()[:, indices_in_new].unsqueeze(1) == states[:, indices_in_self].unsqueeze(0)).all(-1).t())
                ])  # .any(-1)#.nonzero(as_tuple=True)[0]
                new_names.append(new_name)
                new_shape.append(len(mapping_states))
                changed_vars |= set(shared_vars)
        for name in sorted(set(self.names) - changed_vars):
            indice = np.flatnonzero(self.names == name)[0]
            valid_states_per_new_var.append(states[:, indice].nonzero(as_tuple=True)[0].tolist())
            new_names.append(name)
            new_shape.append(self.shape[indice])
        new_states = [
            state
            for comb in zip(*valid_states_per_new_var)
            for state in product(*comb)
        ]
        mask = torch.zeros(*new_shape, dtype=torch.bool)
        mask[tuple(torch.as_tensor(new_states).t())] = True
        return HintFactor(expr=None, names=new_names, shape=new_shape, mask=mask, fn=self.fn)


class ObservationFactor(Factor):
    def __init__(self, expr, names=None, shape=None, mask=None, fn=None):
        assert not (expr is None and names is None and mask is None) and not (mask is not None and names is None)
        assert (names is None) == (shape is None)
        if names is None:
            names = [lit for lit in expr.support]
            shape = torch.as_tensor([2 for lit in expr.support])
        super().__init__(shape, names)
        self.expr = expr
        self.register_buffer('mask', mask)
        self.fn = fn

    def clone(self):
        return ObservationFactor(self.expr, self.names, self.shape, self.mask, self.fn)

    def permute(self, names):
        raise NotImplementedError()

    def factorize(self):
        # returns a list of factors that may be rearranged in cliques
        new_factors = []
        for clause in self.expr.to_cnf().children:
            new_factors.append(ObservationFactor(clause, fn=self.fn))
        return new_factors

    def forward(self, *args, **kwargs):
        potential = self.fn(*args, **kwargs)
        return (potential[..., self.mask].masked_fill(self.mask == -1, 0).float()).log()

    def get_states(self, names=None):
        if names is None:
            if self.mask is not None:
                return self.mask.nonzero()
            names = self.names
        expr = self.expr
        valid_assignements = []
        F = expr.any_factory()
        for sat in expr.satisfy_all(lib='pyeda'):
            valid_assignements.append(torch.as_tensor([sat[name] for name in names], dtype=torch.int))
        return torch.unique(torch.stack(valid_assignements), dim=0)

    def change_variables(self, mapping):
        # Assum new_vars are either SuperVariables or Variables and all those scopes are exclusives
        states = self.get_states().int()
        valid_states_per_new_var = []
        new_names = []
        new_shape = []
        changed_vars = set()
        for new_name, subvars, mapping_states, _ in mapping:
            if mapping_states is None:
                continue
            shared_vars = sorted(set(subvars) & set(self.names))
            if shared_vars:
                indices_in_self = pd.factorize(np.asarray([*self.names, *shared_vars]))[0][len(self.names):]
                indices_in_new = pd.factorize(np.asarray([*subvars, *shared_vars]))[0][len(subvars):]
                valid_states_per_new_var.append([
                    vec.nonzero(as_tuple=True)[0].tolist() for vec in tuple(
                        (mapping_states.int()[:, indices_in_new].unsqueeze(1) == states[:, indices_in_self].unsqueeze(0)).all(-1).t())
                ])  # .any(-1)#.nonzero(as_tuple=True)[0]
                new_names.append(new_name)
                new_shape.append(len(mapping_states))
                changed_vars |= set(shared_vars)
        for name in sorted(set(self.names) - changed_vars):
            indice = np.flatnonzero(self.names == name)[0]
            valid_states_per_new_var.append(states[:, indice].nonzero(as_tuple=True)[0].tolist())
            new_names.append(name)
            new_shape.append(self.shape[indice])
        new_states = [
            state
            for comb in zip(*valid_states_per_new_var)
            for state in product(*comb)
        ]
        mask = torch.zeros(*new_shape, dtype=torch.bool)
        mask[tuple(torch.as_tensor(new_states).t())] = True
        return ObservationFactor(expr=None, names=new_names, shape=new_shape, mask=mask, fn=self.fn)


class ConstraintFactor(Factor):
    def __init__(self, expr, names=None, shape=None, mask=None):
        assert not (expr is None and names is None and mask is None) and not (mask is not None and names is None)
        assert (names is None) == (shape is None)
        if names is None:
            names = [lit for lit in expr.support]
            shape = torch.as_tensor([2 for lit in expr.support])
        super().__init__(shape, names)
        self.expr = expr
        self.register_buffer('mask', mask)

    def clone(self):
        return ConstraintFactor(self.expr, self.names, self.shape, self.mask)

    def permute(self, names):
        raise NotImplementedError()

    def factorize(self):
        # returns a list of factors that may be rearranged in cliques
        new_factors = []
        for clause in self.expr.to_cnf().children:
            new_factors.append(ConstraintFactor(clause))
        return new_factors

    def forward(self, *args, **kwargs):
        return (self.mask.float()).log()

    def get_states(self, names=None):
        if names is None:
            if self.mask is not None:
                return self.mask.nonzero()
            names = self.names
        expr = self.expr
        valid_assignements = []
        F = expr.any_factory()
        for sat in expr.satisfy_all(lib='pyeda'):
            valid_assignements.append(torch.as_tensor([sat[name] for name in names], dtype=torch.int))
        return torch.unique(torch.stack(valid_assignements), dim=0).bool()

    def change_variables(self, mapping):
        # Assum new_vars are either SuperVariables or Variables and all those scopes are exclusives
        states = self.get_states().int()
        valid_states_per_new_var = []
        new_names = []
        new_shape = []
        changed_vars = set()
        for new_name, subvars, mapping_states, _ in mapping:
            if mapping_states is None:
                continue
            shared_vars = sorted(set(subvars) & set(self.names))
            if shared_vars:
                indices_in_self = pd.factorize(np.asarray([*self.names, *shared_vars]))[0][len(self.names):]
                indices_in_new = pd.factorize(np.asarray([*subvars, *shared_vars]))[0][len(subvars):]
                valid_states_per_new_var.append([
                    vec.nonzero(as_tuple=True)[0].tolist() for vec in tuple(
                        (mapping_states.int()[:, indices_in_new].unsqueeze(1) == states[:, indices_in_self].unsqueeze(0)).all(-1).t())
                ])  # .any(-1)#.nonzero(as_tuple=True)[0]
                new_names.append(new_name)
                new_shape.append(len(mapping_states))
                changed_vars |= set(shared_vars)
        for name in sorted(set(self.names) - changed_vars):
            indice = self.names.index(name)
            valid_states_per_new_var.append(states[:, indice].nonzero(as_tuple=True)[0].tolist())
            new_names.append(name)
            new_shape.append(self.shape[indice])
        new_states = [
            state
            for comb in zip(*valid_states_per_new_var)
            for state in product(*comb)
        ]
        mask = torch.zeros(*new_shape, dtype=torch.bool)
        mask[tuple(torch.as_tensor(new_states).t())] = True
        return ConstraintFactor(expr=None, names=new_names, shape=new_shape, mask=mask)
