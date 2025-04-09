import torch
import numpy as np
from unicore.data import BaseWrapperDataset
from . import grid_utils
from .atom_dictionary import RawAtomFeature, AtomGridFeature


class BaseGridDataset(BaseWrapperDataset):

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
        self.args = args
        self.dictionary = dictionary
        self.atom_cls_id = self.dictionary["[CLS]"]
        self.atom_mask_id = self.dictionary["[MASK]"]
        self.atom_H_id = self.dictionary["H"]

        # merge tokens
        self.merge_level = self.args.merge_level

        self.atom_key = atom_key
        self.pos_key = pos_key
        self.grid_len = self.args.grid_len
        self.xyz_resolution = self.args.xyz_resolution

        self.is_train = is_train
        self.num_xyz = round(self.grid_len / self.xyz_resolution)
        assert self.num_xyz % 2 == 0
        self.xyz_null_id = self.num_xyz // 2

    def data_aug_trans(self, atom_pos, epsilon):
        return atom_pos + np.random.normal(size=3) * epsilon

    def phy_pos_to_grid_pos(self, atom_pos):
        atom_grid_pos = np.floor(atom_pos / self.grid_len).astype(np.int32)
        atom_grid_xyz = np.round(
            (atom_pos % self.grid_len) / self.xyz_resolution
        ).astype(np.int32)
        max_xyz = self.num_xyz
        overflow_mask = atom_grid_xyz >= max_xyz
        atom_grid_pos += overflow_mask.astype(np.int32)
        atom_grid_xyz[overflow_mask] = 0
        return atom_grid_pos, atom_grid_xyz

    def grid_pos_to_phy_pos(self, atom_grid_pos, atom_grid_xyz=None):
        if atom_grid_xyz is None:
            incell_pos = 0.5 * self.grid_len
        else:
            incell_pos = atom_grid_xyz.astype(np.float64) * self.xyz_resolution
        return atom_grid_pos.astype(np.float64) * self.grid_len + incell_pos

    def cutoff_by_atom(self, atom_feat, grid_size):

        select_atom = np.random.choice(atom_feat.atom_pos.shape[0], 1, replace=False)[0]
        select_c_pos = atom_feat.atom_pos[select_atom]
        axis_min_t = select_c_pos - grid_size * self.grid_len // 2
        axis_max_t = select_c_pos + (grid_size - 1) * self.grid_len // 2
        pos_flag = np.all(
            (atom_feat.atom_pos > axis_min_t) & (atom_feat.atom_pos < axis_max_t),
            axis=1,
        )
        return atom_feat.subset(pos_flag), (pos_flag == False).sum()

    def H_prob_strategy(self, all_atom_feat_withH):
        use_H = np.random.rand() < self.args.H_prob
        if use_H:
            all_atom_feat = all_atom_feat_withH
        else:
            atom_is_H = all_atom_feat_withH.atom_type == self.atom_H_id
            all_atom_feat = all_atom_feat_withH.subset(~atom_is_H)
        return all_atom_feat

    def get_tree_feat(self, atom_grid_pos):
        tree_pos, tree_type, tree_count = grid_utils.expand_tree_topdown(
            self.merge_level,
            atom_grid_pos,
        )

        tree_node_offset = self.dictionary["[TREE_0]"]
        for i in range(self.merge_level + 1):
            tree_type[i] += tree_node_offset

        tree_phy_pos = []
        for i in range(self.merge_level + 1):
            cur_cell_length = 2 ** (self.merge_level - i)
            tree_phy_pos.append(
                (tree_pos[i].astype(np.float64) + 0.5 * cur_cell_length) * self.grid_len
            )

        return (
            tree_pos,
            tree_type,
            tree_phy_pos,
            tree_count,
        )

    def get_atom_feat(
        self,
        all_atom_grid_feat,
    ):
        atom_type = all_atom_grid_feat.atom_type
        atom_pos = all_atom_grid_feat.atom_pos
        atom_xyz = all_atom_grid_feat.atom_xyz
        atom_phy_pos_coarse = self.grid_pos_to_phy_pos(atom_pos)
        atom_phy_pos_fine = self.grid_pos_to_phy_pos(atom_pos, atom_xyz)
        repeats = np.ones(atom_type.shape[0], dtype=np.int32) * 2
        atom_phy_pos = np.repeat(atom_phy_pos_fine, repeats, axis=0)

        atom_grid_type = np.repeat(atom_type, repeats, axis=0)
        atom_grid_xyz = np.repeat(atom_xyz, repeats, axis=0)
        is_second_atom = np.full(atom_type.shape[0] * 2, False, dtype=np.bool_)
        is_second_atom[1::2] = True
        atom_phy_pos[::2] = atom_phy_pos_coarse
        atom_grid_type[::2] = self.atom_mask_id
        atom_grid_xyz[::2] = self.xyz_null_id
        return (
            atom_grid_type,
            atom_grid_xyz,
            atom_phy_pos,
            is_second_atom,
        )

    def construct_tree_for_decoder(self, all_atom_grid_feat):

        max_grid_size = 2**self.merge_level
        grid_size = np.array([max_grid_size] * 3, dtype=np.int32)
        atom_grid_pos = all_atom_grid_feat.atom_pos
        # by default, each pos should have only one atom
        unique_atom_indices = grid_utils.sample_from_duplicate(atom_grid_pos)
        all_atom_grid_feat = all_atom_grid_feat.subset(unique_atom_indices)
        atom_grid_pos = all_atom_grid_feat.atom_pos
        (
            tree_pos,
            tree_type,
            tree_phy_pos,
            tree_count,
        ) = self.get_tree_feat(atom_grid_pos)

        all_atom_pos = all_atom_grid_feat.atom_pos
        atom_layer_atom_pos = tree_pos[-1]

        assert atom_layer_atom_pos.shape[0] == all_atom_pos.shape[0]
        atom_reorder_indices = grid_utils.reorder_atom_pos(
            all_atom_pos, atom_layer_atom_pos
        )
        assert np.unique(atom_reorder_indices).shape[0] == all_atom_pos.shape[0]
        # reorder based on tree's order
        new_atom_grid_feat = all_atom_grid_feat.subset(atom_reorder_indices)
        assert (new_atom_grid_feat.atom_pos == atom_layer_atom_pos).all()

        # get atom's fine feat
        (
            atom_cell_type,
            atom_cell_xyz,
            atom_phy_pos,
            is_second_atom,
        ) = self.get_atom_feat(
            new_atom_grid_feat,
        )

        # get target feats from atom grid feat
        target_feat = new_atom_grid_feat.extract_target_feat()

        total_atoms = tree_count[0][0]
        tree_xyz = []
        tree_is_second_atom = []
        tree_remaining_atoms = []
        tree_remaining_tokens = []
        for i in range(len(tree_pos)):
            cur_n = tree_pos[i].shape[0]
            tree_is_second_atom.append(np.full(cur_n, False, dtype=np.bool_))
            tree_xyz.append(np.full((cur_n, 3), self.xyz_null_id, dtype=np.int32))
            # accumulate the tree counts in the current level
            cur_count_acc = tree_count[i].cumsum()
            remaining_atoms = total_atoms - cur_count_acc + tree_count[i]
            assert (remaining_atoms > 0).all()
            tree_remaining_atoms.append(remaining_atoms)
            remaining_tokens = np.arange(cur_n, dtype=np.int32)[::-1] + 1
            assert (remaining_tokens > 0).all()
            tree_remaining_tokens.append(remaining_tokens)

        # the remaining atoms is unknown for root, and root need to predict for it
        tree_remaining_atoms[0][:] = 0

        # directly replace the last tree layer
        tree_type[-1] = atom_cell_type
        tree_xyz[-1] = atom_cell_xyz
        tree_phy_pos[-1] = atom_phy_pos
        tree_is_second_atom[-1] = is_second_atom

        def repeat_as_two_token(feat):
            repeats = np.ones(feat.shape[0], dtype=np.int32) * 2
            repeat_feat = np.repeat(feat, repeats, axis=0)
            return repeat_feat

        def is_random_keep(cur_level, len):
            ratio = np.random.rand() * self.args.tree_delete_ratio
            if (
                (not self.is_train)
                or self.args.finetune
                or cur_level > self.args.tree_delete_start_layer
            ):
                ratio = 0.0
            return np.random.rand(len) >= ratio

        for i in range(len(tree_type) - 1):
            cur_level = self.merge_level - i
            is_keep = is_random_keep(cur_level, tree_type[i].shape[0])
            tree_type[i] = repeat_as_two_token(tree_type[i][is_keep])
            tree_xyz[i] = repeat_as_two_token(tree_xyz[i][is_keep])
            tree_phy_pos[i] = repeat_as_two_token(tree_phy_pos[i][is_keep])
            tree_is_second_atom[i] = repeat_as_two_token(
                tree_is_second_atom[i][is_keep]
            )
            tree_remaining_atoms[i] = repeat_as_two_token(
                tree_remaining_atoms[i][is_keep]
            )
            tree_remaining_tokens[i] = repeat_as_two_token(
                tree_remaining_tokens[i][is_keep]
            )
            tree_count[i] = repeat_as_two_token(tree_count[i][is_keep])
            tree_is_second_atom[i][1::2] = True
            tree_type[i][::2] = self.atom_mask_id
            # unknown count for mask token
            tree_count[i][::2] = -1

        # should be all ones
        tree_count[-1] = repeat_as_two_token(tree_count[-1])
        # -1 for mask token
        tree_count[-1][::2] = -1
        tree_remaining_atoms[-1] = repeat_as_two_token(tree_remaining_atoms[-1])
        tree_remaining_tokens[-1] = repeat_as_two_token(tree_remaining_tokens[-1])

        def append_last(feat_list, feat):
            feat_list.append(feat)
            return feat_list

        # mask and cls pair at the end, but didn't predict it during training
        cls_type = np.full(2, self.atom_cls_id, dtype=np.int32)
        cls_type[0] = self.atom_mask_id
        cls_xyz = np.full((2, 3), self.xyz_null_id, dtype=np.int32)
        cls_phy_pos = grid_size.reshape(1, 3).astype(np.float64) * 0.5 * self.grid_len
        cls_phy_pos = np.repeat(cls_phy_pos, 2, axis=0)
        cls_second_atom = np.full(2, False)
        cls_remaining_atoms = np.array([0, 0], dtype=np.int32)
        cls_remaining_tokens = np.array([0, 0], dtype=np.int32)
        cls_count = np.array([0, 0], dtype=np.int32)

        tree_type = append_last(tree_type, cls_type)
        tree_xyz = append_last(tree_xyz, cls_xyz)
        tree_phy_pos = append_last(tree_phy_pos, cls_phy_pos)
        tree_is_second_atom = append_last(tree_is_second_atom, cls_second_atom)
        tree_remaining_atoms = append_last(tree_remaining_atoms, cls_remaining_atoms)
        tree_remaining_tokens = append_last(tree_remaining_tokens, cls_remaining_tokens)
        tree_count = append_last(tree_count, cls_count)

        tree_level = []
        for i in range(len(tree_type)):
            cur_len = tree_type[i].shape[0]
            tree_level.append(np.full(cur_len, self.merge_level - i, dtype=np.int32))

        decoder_type = np.concatenate(tree_type, axis=0)
        decoder_xyz = np.concatenate(tree_xyz, axis=0)
        decoder_phy_pos = np.concatenate(tree_phy_pos, axis=0)
        decoder_level = np.concatenate(tree_level, axis=0)
        decoder_is_second_atom = np.concatenate(tree_is_second_atom, axis=0)
        decoder_remaining_atoms = np.concatenate(tree_remaining_atoms, axis=0)
        decoder_remaining_tokens = np.concatenate(tree_remaining_tokens, axis=0)
        decoder_count = np.concatenate(tree_count, axis=0)

        assert decoder_phy_pos.shape[0] == decoder_type.shape[0]
        assert decoder_phy_pos.shape[0] == decoder_xyz.shape[0]
        assert decoder_phy_pos.shape[0] == decoder_is_second_atom.shape[0]
        assert decoder_phy_pos.shape[0] == decoder_level.shape[0]
        assert decoder_phy_pos.shape[0] == decoder_remaining_atoms.shape[0]
        assert decoder_phy_pos.shape[0] == decoder_remaining_tokens.shape[0]
        assert decoder_phy_pos.shape[0] == decoder_count.shape[0]

        feat = {
            "decoder_phy_pos": decoder_phy_pos,
            "decoder_type": decoder_type,
            "decoder_xyz": decoder_xyz,
            "decoder_level": decoder_level,
            "decoder_is_second_atom": decoder_is_second_atom,
            "decoder_remaining_atoms": decoder_remaining_atoms,
            "decoder_remaining_tokens": decoder_remaining_tokens,
            "decoder_count": decoder_count,
        }
        feat.update(target_feat)
        return feat

    def wrap_decoder_features(self, final_decoder_feature):
        bool_keys = ["decoder_is_second_atom"]
        int_keys = [
            "decoder_type",
            "decoder_xyz",
            "decoder_level",
            "decoder_remaining_atoms",
            "decoder_remaining_tokens",
            "decoder_count",
            "tree_index",
            "space_index",
            "tree_loss_flag",
            "mol_index",
            "atom_mol_index",
            "atom_index",
        ]
        ignore_keys = [
            "cutoff_atom_count",
            "raw_atom_count",
        ]
        feat = {}
        for key in final_decoder_feature.keys():
            if key in ignore_keys:
                feat[key] = final_decoder_feature[key]
                continue
            feat[key] = torch.from_numpy(final_decoder_feature[key])
            if key in bool_keys:
                feat[key] = feat[key].bool()
            elif key in int_keys:
                feat[key] = feat[key].long()
            else:
                feat[key] = feat[key].float()
        return feat

    def make_grid(self, atom_feat, center=None):
        max_grid_size = 2**self.merge_level

        def process(atom_feat, center=None):
            # normalize pos, the center will in the center of grid
            if center is None:
                center = atom_feat.atom_pos.mean(axis=0)
            atom_feat.atom_pos = atom_feat.atom_pos - center + self.grid_len / 2.0

            if self.is_train:
                # apply random translation during training
                atom_feat.atom_pos = self.data_aug_trans(
                    atom_feat.atom_pos,
                    epsilon=self.args.grid_len * 0.5,
                )
            # get the grid cell's position and in-cell xyz position
            atom_grid_pos, atom_grid_xyz = self.phy_pos_to_grid_pos(atom_feat.atom_pos)
            atom_grid_pos = atom_grid_pos - np.min(atom_grid_pos, axis=0, keepdims=True)

            assert (atom_grid_pos >= 0).all(), atom_grid_pos
            assert (atom_grid_xyz >= 0).all() and (
                atom_grid_xyz < self.num_xyz
            ).all(), atom_grid_xyz

            grid_size = np.array([max_grid_size] * 3, dtype=np.int32)
            mean_atom_grid_pos = (
                atom_grid_pos.astype(np.float32).mean(axis=0).astype(np.int32)
            )
            # place to the center
            atom_grid_pos = atom_grid_pos - mean_atom_grid_pos + grid_size // 2
            # random shift the center
            if self.is_train:
                atom_grid_pos = atom_grid_pos + np.random.randint(-1, 2, 3).reshape(
                    1, 3
                )
            if (atom_grid_pos.min(axis=0) < 0).any():
                atom_grid_pos = atom_grid_pos - atom_grid_pos.min(axis=0, keepdims=True)
            if (atom_grid_pos.max(axis=0) >= grid_size[0]).any():
                atom_grid_pos = atom_grid_pos - (
                    atom_grid_pos.max(axis=0, keepdims=True) - grid_size[0] + 1
                )
            return atom_grid_pos, atom_grid_xyz

        atom_grid_pos, atom_grid_xyz = process(atom_feat, center=center)
        cutoff_atom_count = 0
        is_exceed = atom_grid_pos.min() < 0 or atom_grid_pos.max() >= max_grid_size
        enable_cutoff = self.args.enable_cutoff
        if is_exceed and enable_cutoff:
            # perform cutoff, by random placing a grid_size * grid_size * grid_size grid into the current grid
            new_atom_feat, cutoff_atom_count = self.cutoff_by_atom(
                atom_feat, max_grid_size
            )
            assert new_atom_feat.atom_pos.shape[0] > 0
            atom_grid_pos, atom_grid_xyz = process(new_atom_feat, center=center)
            atom_feat = new_atom_feat

        assert (
            atom_grid_pos.min() >= 0 and atom_grid_pos.max() < max_grid_size
        ), "grid pos exceed max size, please use larger merge_level, or enable cutoff"
        return atom_grid_pos, atom_grid_xyz, atom_feat, cutoff_atom_count

    def get_grid_feat(self, data, atom_feat):
        assert atom_feat.atom_pos.shape[0] > 0
        atom_grid_pos, atom_grid_xyz, atom_feat, cutoff_atom_count = self.make_grid(
            atom_feat,
        )
        all_atom_grid_feat_withH = AtomGridFeature(
            atom_feat.atom_type,
            atom_grid_pos,
            atom_grid_xyz,
            self.dictionary,
            atom_feats=atom_feat.atom_feats,
            mol_feats=atom_feat.mol_feats,
        )
        # avoid reuse
        del atom_grid_pos, atom_grid_xyz
        all_atom_grid_feat = self.H_prob_strategy(all_atom_grid_feat_withH)
        raw_atom_count = all_atom_grid_feat.atom_pos.shape[0]

        # construct tree
        decoder_results = self.construct_tree_for_decoder(all_atom_grid_feat)

        decoder_phy_pos = decoder_results["decoder_phy_pos"]

        decoder_results["tree_index"] = np.full(
            decoder_phy_pos.shape[0], 0, dtype=np.int32
        )
        decoder_results["space_index"] = np.full(
            decoder_phy_pos.shape[0], 0, dtype=np.int32
        )
        decoder_results["tree_loss_flag"] = np.full(
            decoder_phy_pos.shape[0], 1, dtype=np.int32
        )
        decoder_results["cutoff_atom_count"] = cutoff_atom_count
        decoder_results["raw_atom_count"] = raw_atom_count

        feat = self.wrap_decoder_features(decoder_results)

        return feat

    def concat_tree_feats(self, feats, trees, spaces, tree_losses, keys):
        # Store final results in one dict
        final_feat = {}
        for key in keys:
            cur_feat = [feat[key] for feat in feats]
            final_feat[key] = np.concatenate(cur_feat, axis=0)

        def get_tree_level_feat(feats, vals):
            assert len(feats) == len(vals)
            feat_list = [
                np.full(feats[i]["decoder_phy_pos"].shape[0], vals[i])
                for i in range(len(feats))
            ]
            return np.concatenate(feat_list, axis=0)

        final_feat["tree_index"] = get_tree_level_feat(feats, trees)
        final_feat["space_index"] = get_tree_level_feat(feats, spaces)
        final_feat["tree_loss_flag"] = get_tree_level_feat(feats, tree_losses)
        return final_feat

    def add_atom_and_mol_targets(self, data, atom_feat):
        atom_target_key = self.args.atom_target_key
        mol_target_key = self.args.mol_target_key
        atom_target = data[atom_target_key] if atom_target_key is not None else None
        mol_target = data[mol_target_key] if mol_target_key is not None else None
        if atom_target is not None and self.args.atom_target_idx is not None:
            atom_target = atom_target[:, self.args.atom_target_idx]
        if mol_target is not None and self.args.mol_target_idx is not None:
            mol_target = mol_target[self.args.mol_target_idx]
        mol_id = data["uid"]
        atom_feat.add_atom_and_mol_targets(mol_target, atom_target, mol_id)
        return atom_feat

    def get_basic_feat(self, data):
        feat = RawAtomFeature.init_from_mol(
            data[self.pos_key],
            data[self.atom_key],
        )
        self.add_atom_and_mol_targets(data, feat)
        return feat
