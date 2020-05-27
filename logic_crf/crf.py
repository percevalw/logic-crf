from typing import List

import torch

from logic_crf.core import Factor
import pandas as pd
import numpy as np
import networkx as nx

from logic_crf.logic import And
from logic_crf.logic_factors import HintFactor, can_merge, Indexer, ConstraintFactor, ObservationFactor
from logic_crf.mrf import MRF
from logic_crf.utils import factorize


class CRF(Factor):
    def __init__(self, factors, mapping=None, names=None, shape=None, factors_input_indices=None):
        """

        Parameters
        ----------
        factors: list of Factor
        """
        # Either all not None or all None
        assert len({mapping is None, names is None, shape is None}) == 1
        if mapping is None:
            names, shape = zip(*sorted(set((v, int(dim)) for fac in factors for v, dim in zip(fac.names, fac.shape))))
        super().__init__(shape, names, mapping)
        self.factors = torch.nn.ModuleList(factors)  # type: List[Factor]
        self.factor_input_indices = factors_input_indices
        self.cached_executions = {}
        self._curried_args = []
        self._curried_kwargs = {}

    def clone(self):
        new_self = CRF(
            list(self.factors),
            list(self.mapping),
            list(self.names),
            list(self.shape),
            list(self.factor_input_indices)
        )
        new_self._curried_args = self._curried_args
        new_self._curried_kwargs = self._curried_kwargs
        return new_self

    def optimize(self):  # observation_set = ["weight", "nnet_outputs"]
        factors = self.factors

        all_variables = sorted(set(v for factor in factors for v in factor.names))
        factors = [new_factor for factor in factors for new_factor in factor.factorize()]
        adjacency_matrix = np.eye(len(all_variables), dtype=bool)
        indices_to_factor = []
        for factor in factors:
            for v1 in (v for v in factor.names):
                for v2 in (v for v in factor.names):
                    adjacency_matrix[all_variables.index(v1), all_variables.index(v2)] = 1
            indices_to_factor.append({"indices": [all_variables.index(var) for var in factor.names],
                                      "factor": factor,
                                      "assigned": False})
        g = nx.from_numpy_matrix(adjacency_matrix)
        G = nx.make_clique_bipartite(g)
        cliques = [v for v in G.nodes() if G.nodes[v]['bipartite'] == 0]
        clique_graph = nx.project(G, cliques)

        # sort by decreasing number of neighbor cliques
        mapping = []
        new_cliques = []
        new_factors = []
        not_yet_mapped_names = set(self.names)
        changed_variables = set()
        ###################################
        # Merge ConstraintFactor together #
        # and compute new variables       #
        ###################################
        for clique_idx in sorted(clique_graph, key=lambda node_idx: len(clique_graph[node_idx]), reverse=True):
            # merge clique factors together
            clique_var_indices = list(G[clique_idx].keys())
            clique_factors = []
            for ind_fac in indices_to_factor:
                if set(ind_fac["indices"]) <= set(clique_var_indices) and not ind_fac["assigned"]:
                    ind_fac["assigned"] = True
                    clique_factors.append(ind_fac["factor"])

            constraint_factors = [fac for fac in clique_factors if isinstance(fac, ConstraintFactor)]
            non_constraint_factors = [fac for fac in clique_factors if fac not in constraint_factors]
            clique_factors = (
                [ConstraintFactor(And(*(fac.expr for fac in constraint_factors))), *non_constraint_factors]
                if len(constraint_factors) else non_constraint_factors)

            new_cliques.append(list(range(len(new_factors), len(new_factors) + len(clique_factors))))
            new_factors.extend(clique_factors)
            for factor in clique_factors:
                if isinstance(factor, ConstraintFactor):
                    variables_to_group = [v for v in factor.names if v not in changed_variables]
                    valid_assignements = torch.unique(factor.get_states(variables_to_group).long(), dim=0).bool()
                    super_variable_name = "/".join(map(str, variables_to_group))
                    indices_in_input = pd.factorize([*self.names, *variables_to_group])[0][len(self.names):]
                    mapping.append((super_variable_name, variables_to_group, valid_assignements, indices_in_input))
                    not_yet_mapped_names -= set(variables_to_group)
                    changed_variables |= set(variables_to_group)
        for name in sorted(not_yet_mapped_names):
            indice_in_input = self.names.index(name)
            mapping.append((name, [name], None, [indice_in_input]))
        # new_variables.extend(set(all_variables) - changed_variables)
        factors = [factor.change_variables(mapping) for factor in new_factors]
        cliques = new_cliques

        new_cliques = []
        new_factors = []
        cluster_hints = []

        ##############################
        # Merge HintFactors together #
        ##############################
        for clique in cliques:
            clique_factors = [factors[i] for i in clique]
            clique_hint_factors = [fac for fac in clique_factors if isinstance(fac, (HintFactor, ObservationFactor)) and isinstance(fac.fn, Indexer)]
            cluster_hints = []

            for fac in clique_hint_factors:
                matching_cluster_hint = next((cluster_hint for cluster_hint in cluster_hints if can_merge(cluster_hint, fac)), None)
                if matching_cluster_hint is None:
                    matching_cluster_hint = fac.clone()
                    matching_cluster_hint.mask = matching_cluster_hint.mask.long()
                    cluster_hints.append(matching_cluster_hint)
                else:
                    last_indexers_1 = matching_cluster_hint.fn.indexers[-1]
                    last_indexers_1 = list(last_indexers_1) if isinstance(last_indexers_1, (list, tuple)) else [last_indexers_1]
                    last_indexers_2 = fac.fn.indexers[-1]
                    last_indexers_2 = list(last_indexers_2) if isinstance(last_indexers_2, (list, tuple)) else [last_indexers_2]
                    new_last_indexer = last_indexers_1 + last_indexers_2
                    matching_cluster_hint.fn = Indexer[(*fac.fn.indexers[:-1], new_last_indexer)]
                    offseted_mask = fac.mask.long() + len(last_indexers_1)
                    offseted_mask[~fac.mask] = 0
                    matching_cluster_hint.mask = matching_cluster_hint.mask + offseted_mask
            for cluster_hint in cluster_hints:
                cluster_hint.mask -= 1

            new_factors.extend((fac for fac in clique_factors if fac not in clique_hint_factors))
            new_factors.extend(cluster_hints)

        factors_input_indices = factorize(
            values=[np.asarray(factor.names) for factor in new_factors],
            reference_values=[entry[0] for entry in mapping], freeze_reference=True
        )[0]

        return CRF(new_factors, mapping, names=self.names, shape=self.shape, factors_input_indices=factors_input_indices)

    def forward(self, *args, contract=False, curry=False, **kwargs):
        if curry:
            new_self = self.clone()
            new_self.curried_args.extend(args)
            new_self.curried_kwargs.update(kwargs)
            return new_self

        args = (*args, *self._curried_args)
        kwargs.update(self._curried_kwargs)
        tensors = []
        tensors_names = []
        variable_dims = []
        batch_dims = []
        for factor, factor_input_indices in zip(self.factors, self.factor_input_indices):
            tensor = factor(*args, **kwargs)
            tensors.append(tensor)
            batch_dims.append(tuple(range(len(tensor.shape) - len(factor_input_indices))))
            variable_dims.append(factor_input_indices)
        n_batch_dim = max(map(len, batch_dims))
        tensors_scheme = [tensor_batch_dims + tuple(d + n_batch_dim for d in tensor_variable_dims)
                          for tensor_batch_dims, tensor_variable_dims in zip(batch_dims, variable_dims)]

        return MRF(tensors, tensors_scheme, (
              [(None, (i,)) for i in range(n_batch_dim)] +
              [(states, tuple(i + n_batch_dim for i in indices)) for (name, _, states, indices) in self.mapping]
        ), crf=self, contract=contract)

    def permute(self, names):
        unkown_variables_names = set(names) - set(self.names)
        assert len(unkown_variables_names) == 0, f"Unkown variable names during permutation: {repr(unkown_variables_names)}"
        names = list(names) + [name for name in self.names if name not in names]

        new_mapping = []
        supernames = []
        remaining_names = set(names)
        # foreach supervariable, find if its subvariables are observed
        for supername, subnames, states, _ in self.mapping:
            indices_in_input = pd.factorize([*names, *subnames])[0][len(names):]
            new_mapping.append((supername, subnames, states, indices_in_input))
            remaining_names -= set(subnames)
        for name in sorted(remaining_names):
            indice_in_input = names.index(name)
            new_mapping.append((name, [name], [indice_in_input], None))
        factors_input_indices = factorize(
            values=[np.asarray(factor.names) for factor in self.factors],
            reference_values=[entry[0] for entry in new_mapping], freeze_reference=True
        )[0]
        new_self = self.clone()
        new_self.mapping = new_mapping
        new_self.factors_input_indices = factors_input_indices
        new_self.names = names
        return new_self
