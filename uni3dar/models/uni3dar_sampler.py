from unicore.models import (
    register_model,
    register_model_architecture,
)
from functools import partial
import torch
from torch.nn import functional as F
from .uni3dar import Uni3DAR, DecoderInputFeat, base_architecture
from uni3dar.data.atom_data import atom_list
from uni3dar.data.grid_utils import subcell_orders
from uni3dar.data.crystal_data_utils import (
    fit_parallelepiped_symmetric_right_hand,
    get_crystal_cond,
)
import numpy as np
import time


NUM_FEAT = 8
IDX_TYPE = 3
IDX_LEVEL = 4
IDX_TREE = 5
IDX_SPACE = 6
IDX_CNT = 7


def softmax_sampling(
    logits,
    top_p=1.0,
    temperature=1.0,
    top_k=None,
    prob_proc_func=None,
):
    assert 0 <= top_p <= 1, "top_p should be in the range [0, 1]."
    if top_k is not None:
        assert top_k > 0 and isinstance(
            top_k, int
        ), "top_k should be a positive integer."

    # Save the original logits for computing scores
    original_logits = logits.float().clone()

    # Adjust logits by temperature
    logits = logits.float() / temperature

    # Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(probs, dim=-1)

    # Apply top_p filter
    sorted_indices_to_remove_p = cumulative_probs > top_p
    sorted_indices_to_remove_p[:, 1:] = sorted_indices_to_remove_p[:, :-1].clone()
    # Ensure at least one token is kept
    sorted_indices_to_remove_p[:, 0] = False

    # Apply top_k filter (if specified)
    if top_k is not None:
        sorted_indices_to_remove_k = torch.ones_like(
            sorted_indices_to_remove_p, dtype=torch.bool
        )
        sorted_indices_to_remove_k[:, :top_k] = False  # Keep only the top_k logits
        sorted_indices_to_remove = (
            sorted_indices_to_remove_p | sorted_indices_to_remove_k
        )
    else:
        sorted_indices_to_remove = sorted_indices_to_remove_p

    # Mask logits that do not satisfy top_p and top_k
    sorted_logits = sorted_logits.masked_fill(sorted_indices_to_remove, float("-inf"))
    filtered_logits = torch.zeros_like(logits).scatter_(
        -1, sorted_indices, sorted_logits
    )

    # Compute probabilities from filtered logits
    probs = torch.softmax(filtered_logits, dim=-1)
    log_probs = torch.log_softmax(original_logits, dim=-1)
    if prob_proc_func is not None:
        probs = prob_proc_func(probs)

    # Sample from the distribution
    predictions = torch.multinomial(probs, num_samples=1).squeeze(-1)

    # Calculate log softmax scores using the original logits
    scores = log_probs.gather(-1, predictions.unsqueeze(-1)).squeeze(-1)

    return predictions.int(), scores


def results_from_predictions(
    data_type, minimum_level, finished_results, rank_ratio, **kwargs
):
    atom_results = []
    for result in finished_results:
        decoder_feat, decoder_phy_pos, score = result
        level = decoder_feat[:, 4]

        score_list = []
        tree_score = score[level > minimum_level]
        atom_score = score[level == minimum_level]
        tree_score_ppl = torch.exp(-tree_score.mean()).item()
        atom_score_ppl = torch.exp(-atom_score.mean()).item()
        score_list = [
            tree_score_ppl,
            atom_score_ppl,
            atom_score_ppl + tree_score_ppl,
        ]
        score_to_sort = score_list[1]
        score_list = [str(i) for i in score_list]
        score_list = " ".join(score_list)
        cur_res = []
        mask = (level == minimum_level) & (decoder_feat[:, 3] < len(atom_list))
        atom_id = decoder_feat[mask, 3]
        atom_id = [atom_list[i] for i in atom_id]
        pos = decoder_phy_pos[mask, :3]
        if data_type == "molecule":
            for cur_atom_name, cur_atom_pos in zip(atom_id, pos):
                cur_res.append(
                    f"{cur_atom_name}\t{cur_atom_pos[0]}\t{cur_atom_pos[1]}\t{cur_atom_pos[2]}"
                )
            if len(cur_res) > 0:
                xyz_res = [f"{len(cur_res)}"]
                xyz_res.append(f"score: {score_list}")
                xyz_res.extend(cur_res)
                xyz_res = "\n".join(xyz_res)
                atom_results.append((xyz_res, score_to_sort, decoder_feat.shape[0]))
        elif data_type == "crystal" and len(atom_id) > 0:
            try:
                lattice_mask = (level == minimum_level) & (
                    decoder_feat[:, 3] == kwargs["atom_lattice_id"]
                )
                assert (
                    lattice_mask.sum() == 8
                ), "should be 8 lattice points, but got {}".format(
                    lattice_mask.sum().item()
                )
                lattice_0, lattice_vectors = fit_parallelepiped_symmetric_right_hand(
                    decoder_phy_pos[lattice_mask, :3].detach().cpu().numpy()
                )
                lattice_vectors = lattice_vectors - lattice_0
                pos_arr = pos.float().detach().cpu().numpy() - lattice_0
                try:
                    from ase import Atoms

                    structure = Atoms(
                        symbols=atom_id,
                        positions=pos_arr,
                        cell=lattice_vectors,
                        pbc=True,
                    )
                except:
                    print("error in construct atoms", atom_id, pos_arr, lattice_vectors)
                    continue
                atom_results.append((structure, score_to_sort, decoder_feat.shape[0]))
            except Exception as e:
                # print(e)
                continue
    atom_results = sorted(atom_results, key=lambda x: x[1], reverse=False)
    num_selected = int(len(atom_results) * rank_ratio)
    selected_results = atom_results[:num_selected]
    atom_results = [selected_results[i][0] for i in range(num_selected)]
    atom_scores = [selected_results[i][1] for i in range(num_selected)]
    seq_len = [selected_results[i][2] for i in range(num_selected)]
    return atom_results, atom_scores, seq_len


@register_model("uni3dar_sampler")
class Uni3DARSampler(Uni3DAR):

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)

    def get_cls_phy_pos(self, merge_level):
        grid_size = 2**merge_level
        return grid_size * self.args.grid_len * 0.5

    def calc_root_level(
        self, merge_level, batch_size, device, tree_index=0, space_index=0
    ):
        cls_phy_pos = self.get_cls_phy_pos(merge_level)
        root_feat_first = torch.zeros((batch_size, 1, NUM_FEAT), device=device).long()
        root_feat_first[:, :, :3] = self.xyz_null_id
        root_feat_first[:, :, IDX_TYPE] = self.atom_mask_id
        root_feat_first[:, :, IDX_LEVEL] = merge_level
        root_feat_first[:, :, IDX_TREE] = tree_index
        root_feat_first[:, :, IDX_SPACE] = space_index
        root_feat_first[:, :, IDX_CNT] = -1
        root_phy_pos = torch.full(
            (batch_size, 1, 3), cls_phy_pos, device=device
        ).float()
        root_feat = torch.zeros((1, NUM_FEAT), dtype=torch.int32, device=device)
        # pos, type, level
        root_feat[0, IDX_TYPE] = -1
        root_feat[0, IDX_LEVEL] = merge_level
        root_feat[0, IDX_TREE] = tree_index
        root_feat[0, IDX_SPACE] = space_index
        root_feat[0, IDX_CNT] = -1
        raw_feat = [root_feat.clone() for i in range(batch_size)]
        return (
            root_feat_first,
            root_phy_pos,
            raw_feat,
        )

    def pop_next_feat(
        self,
        raw_feat,
    ):
        next_feat = []
        new_raw_feat = []
        for i, feat in enumerate(raw_feat):
            assert feat.shape[0] >= 1
            next_feat.append(feat[0:1])
            new_raw_feat.append(feat[1:])
        next_feat = torch.concat(next_feat, dim=0)
        return next_feat, new_raw_feat

    def pop_and_delete(
        self,
        raw_feat,
        decoder_feat,
        decoder_phy_pos,
        cond_vars,
        count_vars,
        kv_cache,
        scores,
        finished_results,
    ):
        is_delete = torch.zeros(
            decoder_feat.shape[0], device=decoder_feat.device
        ).bool()
        next_feat = []
        new_raw_feat = []
        empty = True
        for i, feat in enumerate(raw_feat):
            if feat.shape[0] > 0:
                next_feat.append(feat[0:1])
                new_raw_feat.append(feat[1:])
                empty = False
            else:
                is_delete[i] = True
                finished_results.append(
                    [
                        decoder_feat[i],
                        decoder_phy_pos[i],
                        scores[i],
                    ]
                )
        if empty:
            return (
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                finished_results,
            )
        next_feat = torch.concat(next_feat, dim=0)
        decoder_feat = decoder_feat[~is_delete]
        decoder_phy_pos = decoder_phy_pos[~is_delete]
        for key in cond_vars:
            if cond_vars[key] is not None:
                cond_vars[key] = cond_vars[key][~is_delete]
        for key in count_vars:
            if count_vars[key] is not None:
                count_vars[key] = count_vars[key][~is_delete]
        scores = scores[~is_delete]
        for k in range(len(kv_cache)):
            if kv_cache[k] is None:
                continue
            for i in range(len(kv_cache[k])):
                if kv_cache[k][i]["step"] == 1:
                    continue
                kv_cache[k][i]["k"] = kv_cache[k][i]["k"][~is_delete]
                kv_cache[k][i]["v"] = kv_cache[k][i]["v"][~is_delete]
        return (
            next_feat,
            new_raw_feat,
            decoder_feat,
            decoder_phy_pos,
            cond_vars,
            count_vars,
            kv_cache,
            scores,
            finished_results,
        )

    def predict_and_sample(
        self,
        is_pred_atom,
        next_feat,
        is_first_cnt,
        count_vars,
        dec_y,
        scores,
        tree_predict_funcs,
        atom_predict_funcs,
    ):
        next_type = next_feat[:, IDX_TYPE]
        next_level = next_feat[:, IDX_LEVEL]
        next_tree_index = next_feat[:, IDX_TREE]
        need_pred = next_type < 0
        is_pred_tree = (next_level > self.atom_level_index) & need_pred
        pred_type = next_type.clone()
        pred_count = torch.zeros(
            (dec_y.shape[0],), dtype=torch.int32, device=dec_y.device
        )
        pred_xyz = torch.full(
            (dec_y.shape[0], 3),
            self.xyz_null_id,
            dtype=torch.int32,
            device=dec_y.device,
        )
        score_temp = torch.zeros((dec_y.shape[0],), device=dec_y.device)
        if is_pred_atom.any():
            for i in range(self.num_tree):
                is_pred_atom_i = is_pred_atom & (next_tree_index == i)
                if is_pred_atom_i.any():
                    tmp, scores_ = atom_predict_funcs[i](
                        dec_y[is_pred_atom_i], is_pred_atom_i
                    )
                    pred_type[is_pred_atom_i] = tmp[:, 0]
                    pred_xyz[is_pred_atom_i] = tmp[:, 1:]
                    score_temp[is_pred_atom_i] += scores_
            # set to one's for the atom type
            pred_count[is_pred_atom] = 1
        if is_pred_tree.any():
            for i in range(self.num_tree):
                is_pred_tree_i = is_pred_tree & (next_tree_index == i)
                is_first_cnt_i = is_first_cnt & (next_tree_index == i)
                if is_pred_tree_i.any():
                    type, scores_, count = tree_predict_funcs[i](
                        dec_y[is_pred_tree_i], is_pred_tree_i
                    )
                    pred_type[is_pred_tree_i] = type + self.tree0_split_id + 1
                    score_temp[is_pred_tree_i] += scores_
                    if count is not None:
                        pred_count[is_pred_tree_i] = count
                        has_provide_cnt = (
                            (count_vars["known_atoms"][:, i] > 0)
                            & is_first_cnt_i
                            & is_pred_tree_i
                        )
                        if (has_provide_cnt).any():
                            # if the number of atoms is known, direct assign the count
                            pred_count[has_provide_cnt] = count_vars["known_atoms"][
                                has_provide_cnt, i
                            ]

        scores = torch.cat([scores, score_temp.unsqueeze(1)], dim=1)
        assert (pred_type >= 0).all()
        return pred_type, pred_count, pred_xyz, scores

    def tree_expand(
        self,
        raw_feat,
        cur_type,
        pred_type,
        next_feat,
        count_vars,
    ):
        next_pos = next_feat[:, :3]
        next_level = next_feat[:, IDX_LEVEL]  # the level of the predict token
        next_tree_index = next_feat[:, IDX_TREE]
        next_space_index = next_feat[:, IDX_SPACE]
        shifts = torch.tensor(
            subcell_orders,
            dtype=torch.int32,
            device=next_level.device,
        )

        def get_expand_pos(pos, level):
            next_cell_len = 2 ** (level - 1)
            expanded_pos = pos.unsqueeze(1) + shifts * next_cell_len.view(-1, 1, 1)
            return expanded_pos.view(pos.size(0), -1, 3)

        new_raw_feat = []
        is_split = (cur_type == self.atom_mask_id) & (
            next_level > self.atom_level_index
        )
        if is_split.any():
            bit = 1 << torch.arange(8, device=next_level.device)
            pred_type_binary = (
                (pred_type.unsqueeze(-1) - self.tree0_split_id) & bit
            ) > 0
            expand_cur_pos = get_expand_pos(next_pos, next_level)
            expand_cur_pos = expand_cur_pos[pred_type_binary]
            expand_cur_pos = torch.split(
                expand_cur_pos, pred_type_binary.sum(dim=-1).tolist()
            )

        for i, feat in enumerate(raw_feat):
            if is_split[i]:
                cur_expand_pos = expand_cur_pos[i]
                # append two tokens
                new_split_feat = torch.zeros(
                    (2 * cur_expand_pos.shape[0], NUM_FEAT),
                    dtype=torch.int32,
                    device=feat.device,
                )
                new_split_feat[::2, :3] = cur_expand_pos
                new_split_feat[1::2, :3] = cur_expand_pos
                new_split_feat[::2, IDX_TYPE] = self.atom_mask_id
                new_split_feat[1::2, IDX_TYPE] = -1
                new_split_feat[:, IDX_LEVEL] = next_level[i] - 1
                new_split_feat[:, IDX_TREE] = next_tree_index[i]
                new_split_feat[:, IDX_SPACE] = next_space_index[i]
                count_vars["acc_tokens"][i] += cur_expand_pos.shape[0]
                cur_feat = torch.concat([feat, new_split_feat], dim=0)
            else:
                cur_feat = feat
            new_raw_feat.append(cur_feat)
        return new_raw_feat, count_vars

    def add_new_tree(self, raw_feat, next_feat, is_new_level):
        next_level = next_feat[:, IDX_LEVEL]
        next_tree_index = next_feat[:, IDX_TREE]
        next_space_index = next_feat[:, IDX_SPACE]
        is_new_tree = (
            (next_level == self.atom_level_index)
            & is_new_level
            & (next_tree_index + 1 < self.num_tree)
        )
        new_raw_feat = []
        for i, feat in enumerate(raw_feat):
            if is_new_tree[i]:
                new_split_feat = torch.zeros(
                    (4, NUM_FEAT),
                    dtype=torch.int32,
                    device=feat.device,
                )
                # mask token and cls at the end of tree
                new_split_feat[0, IDX_TYPE] = self.atom_mask_id
                new_split_feat[1, IDX_TYPE] = self.atom_cls_id
                new_split_feat[:2, IDX_LEVEL] = self.atom_level_index - 1
                new_split_feat[:2, IDX_TREE] = next_tree_index[i]

                # new root tokens
                new_split_feat[2, IDX_TYPE] = self.atom_mask_id
                new_split_feat[3, IDX_TYPE] = -1
                new_split_feat[2:, IDX_LEVEL] = self.args.merge_level
                new_split_feat[2:, IDX_TREE] = next_tree_index[i] + 1

                new_split_feat[:, IDX_SPACE] = next_space_index[i]
                new_split_feat[:, IDX_CNT] = -1
                cur_feat = torch.concat([feat, new_split_feat], dim=0)
            else:
                cur_feat = feat
            new_raw_feat.append(cur_feat)
        return new_raw_feat

    def get_feat_by_pred(
        self,
        next_feat,
        decoder_feat,
        decoder_phy_pos,
        pred_type,
        pred_count=None,
        pred_xyz=None,
    ):
        next_pos = next_feat[:, :IDX_TYPE]
        next_level = next_feat[:, IDX_LEVEL]
        next_tree_index = next_feat[:, IDX_TREE]
        next_space_index = next_feat[:, IDX_SPACE]
        edge_len = (2**next_level).view(-1, 1)
        edge_len[edge_len < 1] = 1

        cur_decoder_feat = torch.zeros(
            next_pos.shape[0], NUM_FEAT, dtype=torch.int32, device=next_pos.device
        )
        cur_decoder_feat[:, :3] = self.xyz_null_id
        # set type
        cur_decoder_feat[:, IDX_TYPE] = pred_type
        cur_decoder_feat[:, IDX_LEVEL] = next_level
        cur_decoder_feat[:, IDX_TREE] = next_tree_index
        cur_decoder_feat[:, IDX_SPACE] = next_space_index
        if pred_count is not None:
            cur_decoder_feat[:, IDX_CNT] = pred_count
        else:
            cur_decoder_feat[:, IDX_CNT] = -1

        next_pos = next_pos.float()
        is_atom_level = next_level == self.atom_level_index
        is_end_of_tree_level = next_level == (self.atom_level_index - 1)

        cur_phy_pos = (next_pos + edge_len.float() * 0.5) * self.args.grid_len

        if pred_xyz is not None:
            cur_phy_pos_atom = (
                next_pos * self.args.grid_len + pred_xyz * self.args.xyz_resolution
            )
            cur_phy_pos[is_atom_level] = cur_phy_pos_atom[is_atom_level]
            cur_decoder_feat[is_atom_level, :3] = pred_xyz[is_atom_level]

        # fix pos for additional level
        cur_phy_pos[is_end_of_tree_level] = self.get_cls_phy_pos(self.args.merge_level)

        decoder_feat = torch.concat(
            [decoder_feat, cur_decoder_feat.unsqueeze(1)], dim=1
        )
        decoder_phy_pos = torch.concat(
            [decoder_phy_pos, cur_phy_pos.unsqueeze(1)], dim=1
        )
        return decoder_feat, decoder_phy_pos

    def get_tree_predict_funcs(self):

        def tree_sampling_func(logits, tree_i):
            tree_type, score = softmax_sampling(
                logits,
                1.0,
                self.tree_temperature[tree_i],
            )
            return tree_type, score

        def count_sampling_func(logits, tree_i):
            count_type, score = softmax_sampling(
                logits,
                1.0,
                self.count_temperature[tree_i],
            )
            return count_type, score

        def tree_predict_func(y, batch_indices, tree_i):
            tree_logits = self.tree_heads[tree_i](y)
            tree_pred, tree_score = tree_sampling_func(tree_logits, tree_i)
            if self.count_head is not None:
                y = y + self.tree_emb_for_count[tree_i](tree_pred)
                count_logits = self.count_head[tree_i](y)
                count_pred, count_score = count_sampling_func(count_logits, tree_i)
            else:
                count_pred = None
                count_score = 0.0
            return tree_pred, tree_score + count_score, count_pred

        all_predict_funcs = [
            partial(tree_predict_func, tree_i=i) for i in range(self.num_tree)
        ]
        return all_predict_funcs

    def get_atom_predict_funcs(
        self,
        batch_size,
        device,
        atom_constraint,
        cond_vars,
    ):
        atom_type_mask = torch.ones(
            self.num_atom,
            device=device,
            dtype=torch.bool,
        )

        allow_atoms = self.args.allow_atoms.split(",")
        rank_by = self.args.rank_by
        if "all" not in allow_atoms:
            for atom in allow_atoms:
                atom_type_mask[self.dictionary[atom]] = False
        else:
            atom_type_mask[: len(atom_list)] = False

        atom_type_mask[self.atom_null_id] = False

        if self.args.data_type == "crystal":
            atom_type_mask[self.atom_lattice_id] = False

        if atom_constraint is not None:
            cond_vars["atom_constraint"] = torch.zeros(
                batch_size,
                self.num_atom,
                self.num_tree,
                dtype=torch.long,
                device=device,
            )

            for atom_id in atom_constraint:
                cond_vars["atom_constraint"][:, atom_id, :] += 1
            if self.args.data_type == "crystal":
                # only 8 lattice points
                cond_vars["atom_constraint"][:, :, 0] = 0
                cond_vars["atom_constraint"][:, self.atom_lattice_id, 0] = 8

        def atom_prob_proc(probs):
            probs[:, atom_type_mask] = 0
            return probs

        def atom_constraint_proc(probs, tree_i, batch_indices):
            cur_atom_mask = cond_vars["atom_constraint"][:, :, tree_i] <= 0
            bsz = probs.shape[0]
            assert cur_atom_mask.shape[0] == batch_indices.shape[0]
            cur_atom_mask = cur_atom_mask[batch_indices]
            assert cur_atom_mask.shape[0] == bsz, (bsz, cur_atom_mask.shape)
            probs[cur_atom_mask] = 0
            zero_rows = probs.sum(axis=1) <= 1e-3
            probs[zero_rows, self.atom_null_id] = 1.0
            return probs

        def atom_sampling_func(logits, batch_indices, tree_i):
            atom_type, score = softmax_sampling(
                logits,
                1.0,
                self.atom_temperature[tree_i],
                prob_proc_func=atom_prob_proc,
            )
            return atom_type, score

        def atom_constraint_sampling_func(logits, batch_indices, tree_i):
            atom_type, score = softmax_sampling(
                logits,
                1.0,
                self.atom_temperature[tree_i],
                prob_proc_func=partial(
                    atom_constraint_proc,
                    tree_i=tree_i,
                    batch_indices=batch_indices,
                ),
            )
            cond_vars["atom_constraint"][batch_indices, atom_type, tree_i] -= 1
            return atom_type, score

        def xyz_sampling_func(logits, tree_i):
            return softmax_sampling(logits, 1.0, self.xyz_temperature[tree_i])

        score_ratio = [1.0]
        score_ratio += [0.0] * 3 if rank_by == "atom" else [1.0] * 3

        def atom_predict_func(y, batch_indices, tree_i):
            atom_func = (
                atom_sampling_func
                if "atom_constraint" not in cond_vars
                else atom_constraint_sampling_func
            )
            return self.atom_heads[tree_i].inference(
                y,
                [partial(atom_func, batch_indices=batch_indices, tree_i=tree_i)]
                + [partial(xyz_sampling_func, tree_i=tree_i)] * 3,
                score_ratio=score_ratio,
                xyz_temperature=self.xyz_temperature[tree_i],
            )

        all_predict_funcs = [
            partial(atom_predict_func, tree_i=i) for i in range(self.num_tree)
        ]
        return all_predict_funcs, cond_vars

    def update_running_stats(self, count_vars, is_new_level, last_level):
        cur_remaining_atoms = (
            count_vars["last_remaining_atoms"] - count_vars["pred_count"]
        )
        cur_remaining_tokens = count_vars["last_remaining_tokens"] - 1
        if is_new_level.any():
            cur_remaining_atoms[is_new_level] = count_vars["total_atoms"][is_new_level]
            cur_remaining_tokens[is_new_level] = count_vars["acc_tokens"][is_new_level]
            count_vars["acc_tokens"][is_new_level] = 0
        is_new_tree = is_new_level & (last_level == self.atom_level_index - 1)
        if is_new_tree.any():
            count_vars["total_atoms"][is_new_tree] = 0
            count_vars["acc_tokens"][is_new_tree] = 0
            cur_remaining_atoms[is_new_tree] = 0
            cur_remaining_tokens[is_new_tree] = 1

        pair_remaining_atoms = torch.stack(
            [count_vars["last_remaining_atoms"], cur_remaining_atoms], dim=1
        )
        pair_remaining_tokens = torch.stack(
            [count_vars["last_remaining_tokens"], cur_remaining_tokens], dim=1
        )
        return count_vars, pair_remaining_atoms, pair_remaining_tokens

    def generate(
        self,
        data=None,
        atom_constraint=None,
    ):
        batch_size = self.args.batch_size
        self.eval()
        torch.set_grad_enabled(False)
        torch.cuda.empty_cache()
        start_time = time.time()
        device = next(self.parameters()).device
        kv_cache = [
            [{"step": 1} for _ in range(self.args.layer)]
            for _ in range(self.args.recycle)
        ]
        target_tree_index, target_space_index = 0, 0
        cond_vars = {"cond": None}
        # using for running stats
        count_vars = {
            "known_atoms": torch.full(
                (batch_size, self.num_tree), -1, dtype=torch.int32, device=device
            ),
            "total_atoms": torch.zeros(batch_size, dtype=torch.int32, device=device),
            "acc_tokens": torch.zeros(batch_size, dtype=torch.int32, device=device),
        }
        if self.args.data_type == "crystal":
            count_vars["known_atoms"][:, 0] = 8

        if data is not None:
            if self.args.data_type == "crystal":
                (
                    kv_cache,
                    cond,
                    num_atoms,
                ) = self.crystal_cond_prefilling_with_data(
                    data,
                    batch_size,
                    device,
                    kv_cache,
                )
                cond_vars["cond"] = cond.unsqueeze(1)
                # set the total atoms to the known atoms
                count_vars["known_atoms"][:, 1] = num_atoms

        (decoder_feat, decoder_phy_pos, raw_feat) = self.calc_root_level(
            self.args.merge_level,
            batch_size,
            device,
            target_tree_index,
            target_space_index,
        )

        scores = torch.zeros((batch_size, 1), device=device)
        finished_results = []

        # construct atom constraints
        atom_predict_funcs, cond_vars = self.get_atom_predict_funcs(
            batch_size,
            device,
            atom_constraint,
            cond_vars,
        )
        tree_predict_funcs = self.get_tree_predict_funcs()

        with torch.no_grad():
            cnt = 0
            # prepare the first input, it only has one token
            cur_feat = decoder_feat[:, -1, :]
            cur_phy_pos = decoder_phy_pos[:, -1, :]
            cur_input = DecoderInputFeat(
                cur_feat[:, IDX_TYPE],
                cur_feat[:, :3],
                cur_feat[:, IDX_LEVEL],
                cur_phy_pos,
                cur_feat[:, IDX_SPACE],
                cur_feat[:, IDX_TREE],
                torch.zeros_like(cur_feat[:, IDX_TYPE]).int(),
                torch.ones_like(cur_feat[:, IDX_TYPE]).int(),
                cur_feat[:, IDX_CNT],
            ).add_seq_dim()
            while len(raw_feat) > 0:
                dec_y, _ = self.forward_model(
                    cur_input,
                    c=cond_vars["cond"],
                    kv_cache_list=kv_cache,
                )
                dec_y = dec_y[:, -1, :]

                last_level = cur_input.level[:, -1]
                last_type = cur_input.type[:, -1]
                last_remaining_atoms = cur_input.remaining_atoms[:, -1]
                last_remaining_tokens = cur_input.remaining_tokens[:, -1]

                # get the information of current predict result
                next_feat, raw_feat = self.pop_next_feat(raw_feat)
                next_type = next_feat[:, IDX_TYPE]
                # predict and sample
                is_atom_layer = last_level == self.atom_level_index
                is_pred_atom = (
                    is_atom_layer & (last_type == self.atom_mask_id) & (next_type < 0)
                )
                is_first_cnt = (last_level == self.args.merge_level) & (
                    last_type == self.atom_mask_id
                )
                pred_type, pred_count, pred_xyz, scores = self.predict_and_sample(
                    is_pred_atom,
                    next_feat,
                    is_first_cnt,
                    count_vars,
                    dec_y,
                    scores,
                    tree_predict_funcs,
                    atom_predict_funcs,
                )
                if is_first_cnt.any():
                    count_vars["total_atoms"][is_first_cnt] = pred_count[is_first_cnt]
                raw_feat, count_vars = self.tree_expand(
                    raw_feat,
                    last_type,
                    pred_type,
                    next_feat,
                    count_vars,
                )
                # get the new feat based on the prediction
                decoder_feat, decoder_phy_pos = self.get_feat_by_pred(
                    next_feat,
                    decoder_feat,
                    decoder_phy_pos,
                    pred_type,
                    pred_count,
                    pred_xyz,
                )
                cnt += 1
                count_vars["last_remaining_atoms"] = last_remaining_atoms
                count_vars["last_remaining_tokens"] = last_remaining_tokens
                count_vars["pred_count"] = pred_count
                # pop the next feat, and it should be mask
                (
                    next_feat,
                    raw_feat,
                    decoder_feat,
                    decoder_phy_pos,
                    cond_vars,
                    count_vars,
                    kv_cache,
                    scores,
                    finished_results,
                ) = self.pop_and_delete(
                    raw_feat,
                    decoder_feat,
                    decoder_phy_pos,
                    cond_vars,
                    count_vars,
                    kv_cache,
                    scores,
                    finished_results,
                )
                # no more nodes
                if next_feat is None:
                    break
                # the change of level happens before mask token
                last_level = decoder_feat[:, -1, IDX_LEVEL]
                is_new_level = last_level != next_feat[:, IDX_LEVEL]
                raw_feat = self.add_new_tree(raw_feat, next_feat, is_new_level)
                # direct append a mask token, don't need to predict
                decoder_feat, decoder_phy_pos = self.get_feat_by_pred(
                    next_feat,
                    decoder_feat,
                    decoder_phy_pos,
                    next_feat[:, IDX_TYPE],
                    pred_count=None,
                    pred_xyz=None,
                )
                scores = torch.cat(
                    [
                        scores,
                        torch.zeros((next_feat.shape[0], 1), device=cur_feat.device),
                    ],
                    dim=1,
                )
                count_vars, pair_remaining_atoms, pair_remaining_tokens = (
                    self.update_running_stats(count_vars, is_new_level, last_level)
                )

                # get the last two tokens
                cur_feat = decoder_feat[:, -2:, :]
                cur_phy_pos = decoder_phy_pos[:, -2:, :]
                cur_input = DecoderInputFeat(
                    cur_feat[:, :, IDX_TYPE],
                    cur_feat[:, :, :3],
                    cur_feat[:, :, IDX_LEVEL],
                    cur_phy_pos,
                    cur_feat[:, :, IDX_SPACE],
                    cur_feat[:, :, IDX_TREE],
                    pair_remaining_atoms,
                    pair_remaining_tokens,
                    cur_feat[:, :, IDX_CNT],
                )
                # count for mask token is -1
                assert (cur_input.count[:, -1] == -1).all()
                cur_is_cls_level = cur_input.level == (self.atom_level_index - 1)
                cur_input.remaining_atoms[cur_is_cls_level] = 0
                cur_input.remaining_tokens[cur_is_cls_level] = 0
                cur_input.count[cur_is_cls_level] = 0

                # avoid overflow
                cur_input.remaining_atoms[cur_input.remaining_atoms < 0] = 0
                cur_input.remaining_tokens[cur_input.remaining_tokens < 0] = 0
                cur_input.remaining_tokens[
                    cur_input.remaining_tokens >= self.args.max_num_atom
                ] = self.args.max_num_atom
                cnt += 1

        cost_time = time.time() - start_time
        xyz, score, seq_len = results_from_predictions(
            self.args.data_type,
            self.atom_level_index,
            finished_results,
            self.args.rank_ratio,
            atom_lattice_id=self.atom_lattice_id,
        )
        num_samples = len(xyz)
        print(
            "num_samples",
            num_samples,
            "max_seq_len",
            cnt,
            "avg_seq_len",
            np.mean(seq_len),
            "cost_time",
            cost_time,
            "score mean/std",
            np.mean(score),
            np.std(score),
        )
        # return the processed results
        return xyz, score

    def crystal_cond_prefilling_with_data(
        self,
        data,
        batch_size,
        device,
        kv_cache,
    ):

        atom_type = data[self.args.atom_type_key]
        data = get_crystal_cond(
            self.args,
            data,
            atom_type,
            False,
            self.dictionary,
            self.xyz_null_id,
            bsz=batch_size,
        )
        for key in data:
            data[key] = torch.from_numpy(data[key]).to(device)

        dummy_input = DecoderInputFeat(
            data["decoder_type"],
            data["decoder_xyz"],
            data["decoder_level"],
            data["decoder_phy_pos"],
            torch.zeros_like(data["decoder_type"]),
            torch.zeros_like(data["decoder_type"]),
            data["decoder_remaining_atoms"],
            data["decoder_remaining_tokens"],
            data["decoder_count"],
        )

        with torch.no_grad():
            _, c = self.forward_model(
                dummy_input,
                kv_cache_list=kv_cache,
                need_norm=False,
                pxrd=(data["pxrd"] if self.args.crystal_pxrd > 0 else None),
                components=(
                    data["components"] if self.args.crystal_component > 0 else None
                ),
            )

        return kv_cache, c, len(atom_type)


@register_model_architecture("uni3dar_sampler", "uni3dar_sampler")
def sampler_base_architecture(args):
    args = base_architecture(args)
