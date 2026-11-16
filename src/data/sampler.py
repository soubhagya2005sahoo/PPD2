"""Random Identity Sampler for Triplet Loss Batch Construction.

In Person Re-Identification, standard random mini-batches often contain
only one image per person, preventing triplet loss from finding positive pairs.

RandomIdentitySampler guarantees that every batch contains:
- P distinct person identities.
- K images per identity.
- Total batch size = P * K.
"""

import copy
import random
from typing import Iterator, List
from torch.utils.data.sampler import Sampler
from .dataset import AGReIDDataset


class RandomIdentitySampler(Sampler):
    """Samples P identities, each with K instances, to form mini-batches."""

    def __init__(self, data_source: AGReIDDataset, num_pids_per_batch: int = 4, num_instances_per_pid: int = 4):
        """Initialize the RandomIdentitySampler.

        Args:
            data_source: An AGReIDDataset instance with pid_to_indices mapped.
            num_pids_per_batch: Number of distinct persons per mini-batch (P).
            num_instances_per_pid: Number of image instances per person (K).
        """
        self.data_source = data_source
        self.num_pids_per_batch = num_pids_per_batch
        self.num_instances_per_pid = num_instances_per_pid
        self.batch_size = num_pids_per_batch * num_instances_per_pid

        # Copy the mapping of pid -> list of dataset indices
        self.pid_to_indices = copy.deepcopy(data_source.pid_to_indices)
        self.pids = list(self.pid_to_indices.keys())
        self.num_identities = len(self.pids)

        # Estimate length of one epoch (total batches * batch_size)
        total_samples = len(data_source)
        self.length = (total_samples // self.batch_size) * self.batch_size

    def __iter__(self) -> Iterator[int]:
        batch_idx_dict = copy.deepcopy(self.pid_to_indices)

        # Shuffle instances for each identity
        for pid in self.pids:
            random.shuffle(batch_idx_dict[pid])

        # Track available identities that still have >= K samples
        avai_pids = copy.deepcopy(self.pids)
        final_indices: List[int] = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idx_dict[pid]
                if len(batch_idxs) < self.num_instances_per_pid:
                    # If not enough images remain, sample with replacement
                    batch_idxs = random.choices(self.pid_to_indices[pid], k=self.num_instances_per_pid)
                else:
                    batch_idxs = [batch_idxs.pop(0) for _ in range(self.num_instances_per_pid)]

                final_indices.extend(batch_idxs)

                if len(batch_idx_dict[pid]) < self.num_instances_per_pid:
                    avai_pids.remove(pid)

        return iter(final_indices)

    def __len__(self) -> int:
        return self.length

