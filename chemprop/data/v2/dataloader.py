from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from chemprop.data.v2.data import MolGraphDataset
from chemprop.data.v2.sampler import ClassBalanceSampler, MoleculeSampler, SeededSampler
from chemprop.featurizers.molgraph import MolGraph


def collate_graphs(mgs: Sequence[MolGraph]) -> tuple:
    n_atoms = 1
    n_bonds = 1
    a_scope = []
    b_scope = []

    # All start with zero padding so that indexing with zero padding returns zeros
    X_vs = [np.zeros(mgs[0].X_v.shape[0])]
    X_es = [np.zeros(mgs[0].X_e.shape[0])]
    a2b = [[]]
    b2a = [0]
    b2revb = [0]

    for mg in mgs:
        X_vs.append(mg.X_v)
        X_es.append(mg.X_e)

        # for a in range(mg.n_atoms):
        #     a2b.append([b + n_bonds for b in mg.a2b[a]])

        # for b in range(mg.n_bonds):
        #     b2a.append(n_atoms + mg.b2a[b])
        #     b2revb.append(n_bonds + mg.b2revb[b])

        a2b.extend([[b + n_bonds for b in mg.a2b[a]] for a in range(mg.n_atoms)])
        b2a.extend([n_atoms + mg.b2a[b] for b in range(mg.n_bonds)])
        b2revb.extend([n_bonds + mg.b2revb[b] for b in range(mg.n_bonds)])

        a_scope.append((n_atoms, mg.n_atoms))
        b_scope.append((n_bonds, mg.n_bonds))

        n_atoms += mg.n_atoms
        n_bonds += mg.n_bonds

    X_v = torch.from_numpy(np.concatenate(X_vs).astype(np.float32))
    X_e = torch.from_numpy(np.concatenate(X_es).astype(np.float32))

    # max with 1 to fix a crash in rare case of all single-heavy-atom mols
    max_num_bonds = max(1, max(len(in_bonds) for in_bonds in a2b))
    a2b = torch.tensor(
        [a2b[a] + [0] * (max_num_bonds - len(a2b[a])) for a in range(n_atoms)], dtype=torch.long
    )

    b2a = torch.tensor(b2a, dtype=torch.long)
    b2revb = torch.tensor(b2revb, dtype=torch.long)
    a2a =  b2a[a2b]

    return X_v, X_e, a2b, b2a, b2revb, a_scope, b_scope, a2a


class MoleculeDataLoader(DataLoader):
    """A :class:`MoleculeDataLoader` is a PyTorch :class:`DataLoader` for loading a :class:`MoleculeDataset`.
    
    dataset : MoleculeDataset
        The `MoleculeDataset` containing the molecules to load.
    batch_size : int, default=50
        the batch size to load
    num_workers : int, default=0
        the number of workers used to build batches.
    class_balance : bool, default=False
        Whether to perform class balancing (i.e., use an equal number of positive and negative 
        molecules). Class balance is only available for single task classification datasets. Set 
        shuffle to True in order to get a random subset of the larger class.
    seed : int, default=None
        the random seed to use for shuffling (only used when `shuffle` is `True`)
    shuffle : bool, default=False
        whether to shuffle the data during sampling
    """
    def __init__(
        self,
        dataset: MolGraphDataset,
        batch_size: int = 50,
        num_workers: int = 0,
        class_balance: bool = False,
        seed: Optional[int] = None,
        shuffle: bool = False,
    ):
        
        self.dataset = dataset
        self.class_balance = class_balance
        self.shuffle = shuffle

        if self.class_balance:
            self.sampler = ClassBalanceSampler(self.dataset, seed, self.shuffle)
        elif self.shuffle and seed is not None:
            self.sampler = SeededSampler(self.dataset, seed, self.shuffle)
        else:
            self.sampler = None

        super().__init__(
            self.dataset,
            batch_size,
            self.sampler is None and self.shuffle,
            self.sampler,
            num_workers=num_workers,
            collate_fn=collate_graphs,
        )

    @property
    def targets(self) -> np.ndarray:
        """
        Returns the targets associated with each molecule.

        :return: A list of lists of floats (or None) containing the targets.
        """
        if self.class_balance or self.shuffle:
            raise ValueError(
                "Cannot safely extract targets when class balance or shuffle are enabled."
            )

        return np.array([self.dataset[i].targets for i in self.sampler])

    @property
    def gt_targets(self) -> list[list[Optional[bool]]]:
        """
        Returns booleans for whether each target is an inequality rather than a value target, associated with each molecule.

        :return: A list of lists of booleans (or None) containing the targets.
        """
        if self.class_balance or self.shuffle:
            raise ValueError(
                "Cannot safely extract targets when class balance or shuffle are enabled."
            )

        if not hasattr(self.dataset[0], "gt_targets"):
            return None

        return [self.dataset[i].gt_targets for i in self.sampler]

    @property
    def lt_targets(self) -> list[list[Optional[bool]]]:
        """
        Returns booleans for whether each target is an inequality rather than a value target, associated with each molecule.

        :return: A list of lists of booleans (or None) containing the targets.
        """
        if self.class_balance or self.shuffle:
            raise ValueError(
                "Cannot safely extract targets when class balance or shuffle are enabled."
            )

        if not hasattr(self.dataset[0], "lt_targets"):
            return None

        return [self.dataset[i].lt_targets for i in self.sampler]

    @property
    def iter_size(self) -> int:
        """Returns the number of data points included in each full iteration through the :class:`MoleculeDataLoader`."""
        return len(self.sampler)
