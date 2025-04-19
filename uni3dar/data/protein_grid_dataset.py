# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from .base_grid_dataset import BaseGridDataset
from . import crystal_data_utils
from .atom_dictionary import RawAtomFeature, AtomGridFeature


class ProteinGridDataset(BaseGridDataset):
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
        super().__init__(
            dataset,
            seed,
            dictionary,
            atom_key,
            pos_key,
            is_train,
            args,
        )

    def get_basic_feat(self, data):
        atom_pos = data[self.pos_key]
        atom_type = data[self.atom_key]
        alpha_c_index = atom_type == "CA"
        atom_type = atom_type[alpha_c_index]
        atom_pos = atom_pos[alpha_c_index]
        res_type = data[self.args.res_type_key][alpha_c_index]
        atom_type = [f"{res}_{atom}" for res, atom in zip(res_type, atom_type)]
        feat = RawAtomFeature.init_from_mol(atom_pos, atom_type, key="protein")
        atom_target = (
            data[self.args.atom_target_key][alpha_c_index]
            if self.args.atom_target_key
            else None
        )
        feat.add_atom_and_mol_targets(None, atom_target, data["uid"])
        return feat
