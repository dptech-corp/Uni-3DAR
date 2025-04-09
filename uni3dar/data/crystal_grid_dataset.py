# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from .base_grid_dataset import BaseGridDataset
from . import crystal_data_utils
from .atom_dictionary import RawAtomFeature, AtomGridFeature


class CrystalGridDataset(BaseGridDataset):
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
        return RawAtomFeature.init_from_crystal(
            data[self.pos_key],
            data[self.atom_key],
            data[self.args.lattice_matrix_key],
            self.is_train,
            self.args.crystal_random_shift_prob,
        )

    def get_grid_feat(self, data, atom_feat):
        num_lattice = 8
        assert atom_feat.data_type == "crystal"
        center = atom_feat.atom_pos[:num_lattice].mean(axis=0)
        atom_grid_pos, atom_grid_xyz, atom_feat, cutoff_atom_count = self.make_grid(
            atom_feat,
            center=center,
        )
        raw_atom_count = atom_grid_pos.shape[0]

        lattice_grid_feat = AtomGridFeature(
            atom_feat.atom_type[:num_lattice],
            atom_grid_pos[:num_lattice],
            atom_grid_xyz[:num_lattice],
            self.dictionary,
        )

        atom_grid_feat = AtomGridFeature(
            atom_feat.atom_type[num_lattice:],
            atom_grid_pos[num_lattice:],
            atom_grid_xyz[num_lattice:],
            self.dictionary,
        )

        decoder_lattice = self.construct_tree_for_decoder(lattice_grid_feat)
        decoder_atom = self.construct_tree_for_decoder(atom_grid_feat)
        decoder_cond = crystal_data_utils.get_crystal_cond(
            self.args,
            data,
            atom_feat.atom_type[num_lattice:],
            self.is_train,
            self.dictionary,
            self.xyz_null_id,
        )

        final_decoder_feature = self.concat_tree_feats(
            [decoder_cond, decoder_lattice, decoder_atom],
            trees=[0, 0, 1],
            spaces=[0, 0, 0],
            tree_losses=[0, 1, 1],
            keys=decoder_atom.keys(),
        )
        final_decoder_feature["cutoff_atom_count"] = cutoff_atom_count
        final_decoder_feature["raw_atom_count"] = raw_atom_count

        final_decoder_feature = self.wrap_decoder_features(final_decoder_feature)

        if self.args.crystal_pxrd:
            final_decoder_feature["pxrd"] = torch.from_numpy(
                decoder_cond["pxrd"]
            ).float()
        if self.args.crystal_component:
            final_decoder_feature["components"] = torch.from_numpy(
                decoder_cond["components"]
            ).float()

        return final_decoder_feature
