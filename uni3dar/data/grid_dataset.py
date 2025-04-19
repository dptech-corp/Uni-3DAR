# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import time
from functools import lru_cache
from unicore.data import BaseWrapperDataset
from . import data_utils, grid_utils
from .base_grid_dataset import BaseGridDataset
from .crystal_grid_dataset import CrystalGridDataset
from .protein_grid_dataset import ProteinGridDataset


class GridDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        seed,
        dictionary,
        atom_key,
        pos_key,
        is_train,
        args,
    ):
        self.dataset = dataset
        self.seed = seed
        self.set_epoch(None)
        self.args = args
        dataset_args = (dataset, seed, dictionary, atom_key, pos_key, is_train, args)
        self.base_grid_dataset = BaseGridDataset(*dataset_args)
        self.crystal_grid_dataset = CrystalGridDataset(*dataset_args)
        self.protein_grid_dataset = ProteinGridDataset(*dataset_args)

        self.datasets = {
            "molecule": self.base_grid_dataset,
            "crystal": self.crystal_grid_dataset,
            "protein": self.protein_grid_dataset,
        }

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        with data_utils.numpy_seed(self.seed, epoch, index):
            start_time = time.time()
            data = self.dataset[index]
            data_type = data.get("data_type", "molecule")
            cur_handler = self.datasets.get(data_type)
            atom_feat = cur_handler.get_basic_feat(data)
            atom_feat.atom_pos, rot = grid_utils.normalize_pos(
                atom_feat.atom_pos, self.args.random_rotation_prob
            )
            if self.args.atom_target_key is not None and self.args.atom_target_rotate:
                atom_feat.apply_rot_to_atom_target("target", rot)
            feat = cur_handler.get_grid_feat(data, atom_feat)

            feat["decoder_input_len"] = torch.tensor(
                [feat["decoder_phy_pos"].shape[0]]
            ).int()

            feat["time"] = torch.tensor(time.time() - start_time).float()
            return feat

    def collater(self, items):
        def lens_to_offset(lens):
            offsets = []
            cur_pos = 0
            offsets.append(cur_pos)
            for l in lens:
                cur_pos += int(l[0])
                offsets.append(cur_pos)
            return torch.tensor(offsets).int()

        pad_fns = {
            "decoder_input_len": lens_to_offset,
        }

        non_pad_keys = set(
            [
                "time",
                "cutoff_atom_count",
                "raw_atom_count",
            ]
        )

        batched_data = {}
        for key in items[0].keys():
            samples = [item[key] for item in items]
            if key in pad_fns:
                batched_data[key] = pad_fns[key](samples)
            elif key not in non_pad_keys:
                batched_data[key] = data_utils.pad_flatten(samples)
            if key == "decoder_input_len":
                batched_data["decoder_input_max_len"] = torch.tensor(
                    [x[0].data for x in samples]
                ).max()
                batched_data["decoder_index"] = torch.cat(
                    [torch.zeros(x[0].data).int() + id for id, x in enumerate(samples)]
                )
            if key == "time":
                batched_data["time"] = torch.tensor([x.data for x in samples])
        batched_data["cutoff_atom_count"] = [
            item["cutoff_atom_count"] for item in items
        ]
        batched_data["raw_atom_count"] = [item["raw_atom_count"] for item in items]

        return batched_data

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)
