import logging
from functools import partial
import torch
from torch.nn import functional as F
from unicore.models import (
    BaseUnicoreModel,
    register_model,
    register_model_architecture,
)
from .autoregressive_prediction_head import AutoRegressivePredictionHead
from .layers import (
    Embedding,
    Linear,
    TaskHead,
    ClassificationHead,
    AdaNorm,
)
from unicore.modules import RMSNorm
from .diffusion_prediction_head import DiffusionPredictionHead
from .transformer_encoder import Transformer
from .rope import RoPEnD
import torch.nn as nn


logger = logging.getLogger(__name__)


class DecoderInputFeat:
    def __init__(
        self,
        type,
        xyz,
        level,
        phy_pos,
        space_index,
        tree_index,
        remaining_atoms,
        remaining_tokens,
        count,
    ):
        self.type = type
        self.xyz = xyz
        self.level = level
        self.phy_pos = phy_pos
        self.space_index = space_index
        self.tree_index = tree_index
        self.remaining_atoms = remaining_atoms
        self.remaining_tokens = remaining_tokens
        self.count = count

    def add_batch_dim(self):
        return DecoderInputFeat(
            self.type.unsqueeze(0),
            self.xyz.unsqueeze(0),
            self.level.unsqueeze(0),
            self.phy_pos.unsqueeze(0),
            self.space_index.unsqueeze(0),
            self.tree_index.unsqueeze(0),
            self.remaining_atoms.unsqueeze(0),
            self.remaining_tokens.unsqueeze(0),
            self.count.unsqueeze(0),
        )

    def add_seq_dim(self):
        return DecoderInputFeat(
            self.type.unsqueeze(1),
            self.xyz.unsqueeze(1),
            self.level.unsqueeze(1),
            self.phy_pos.unsqueeze(1),
            self.space_index.unsqueeze(1),
            self.tree_index.unsqueeze(1),
            self.remaining_atoms.unsqueeze(1),
            self.remaining_tokens.unsqueeze(1),
            self.count.unsqueeze(1),
        )


class DecoderGenTargetFeat:
    def __init__(
        self,
        type,
        xyz,
        count,
        level,
        tree_index,
        is_gen_tree,
        is_gen_atom,
    ):
        self.type = type
        self.xyz = xyz
        self.count = count
        self.level = level
        self.tree_index = tree_index
        self.is_gen_tree = is_gen_tree
        self.is_gen_atom = is_gen_atom

    def to_list(self, indices):
        return [
            self.type[indices],
            self.xyz[indices, 0],
            self.xyz[indices, 1],
            self.xyz[indices, 2],
        ]


class DecoderPredTargetFeat:
    def __init__(
        self,
        tree_index,
        is_input_atom,
        is_input_cls,
        atom_target,
        mol_target,
        atom_index,
        atom_mol_index,
        mol_index,
        rot=None,
    ):
        self.tree_index = tree_index
        self.is_input_atom = is_input_atom
        self.is_input_cls = is_input_cls
        self.atom_target = atom_target
        self.mol_target = mol_target
        self.atom_index = atom_index
        self.atom_mol_index = atom_mol_index
        self.mol_index = mol_index
        self.rot = rot


@register_model("uni3dar")
class Uni3DAR(BaseUnicoreModel):
    """
    Class for training a Masked Language Model. It also supports an
    additional sentence level prediction if the sent-loss argument is set.
    """

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # the model releated parameters
        parser.add_argument(
            "--emb-dim",
            type=int,
        )
        parser.add_argument(
            "--num-head",
            type=int,
        )
        parser.add_argument(
            "--residual-dropout",
            type=float,
        )
        parser.add_argument(
            "--ffn-multiple",
            type=int,
        )
        parser.add_argument(
            "--attn-dropout",
            type=float,
        )
        parser.add_argument(
            "--emb-dropout",
            type=float,
        )
        parser.add_argument(
            "--head-dropout",
            type=float,
            default=0.0,
        )
        parser.add_argument(
            "--pooler-dropout",
            type=float,
        )
        parser.add_argument(
            "--pooler-activation-fn",
            type=str,
        )
        parser.add_argument(
            "--layer",
            type=int,
        )
        parser.add_argument(
            "--recycle",
            type=int,
        )
        parser.add_argument(
            "--rope-theta",
            type=int,
        )
        parser.add_argument(
            "--loss-ratio-tree",
            type=str,
            default="1.0",
        )
        parser.add_argument(
            "--loss-ratio-atom",
            type=str,
            default="1.0",
        )
        parser.add_argument(
            "--loss-ratio-xyz",
            type=str,
            default="1.0",
        )
        parser.add_argument(
            "--loss-ratio-count",
            type=str,
            default="1.0",
        )
        parser.add_argument(
            "--loss-ratio-mol-target",
            type=str,
            default="1.0",
        )
        parser.add_argument(
            "--loss-ratio-atom-target",
            type=str,
            default="1.0",
        )
        parser.add_argument(
            "--checkpoint-activation-threshold",
            default=100000,
            type=int,
            help="threshold to enable checkpointing during training",
        )
        parser.add_argument(
            "--finetune",
            action="store_true",
            default=False,
            help="finetune",
        )
        parser.add_argument(
            "--atom-head-type",
            default="ar",
            type=str,
            choices=["diffusion", "ar"],
        )

    def __init__(self, args, dictionary):
        super().__init__()
        self.dictionary = dictionary
        base_architecture(args)
        self.args = args
        self.num_atom = len(self.dictionary)
        self.tree0_split_id = self.dictionary["[TREE_0]"]
        self.atom_mask_id = self.dictionary["[MASK]"]
        self.atom_cls_id = self.dictionary["[CLS]"]
        self.atom_null_id = self.dictionary["[NULL]"]

        self.num_xyz = round(self.args.grid_len / self.args.xyz_resolution)
        self.xyz_null_id = self.num_xyz // 2
        self.atom_level_index = 0

        emb_dim = self.args.emb_dim
        num_head = self.args.num_head

        self.head_dim = emb_dim // num_head
        self.atom_emb = Embedding(self.num_atom, emb_dim)
        self.xyz_emb = nn.ModuleList(
            [Embedding(self.num_xyz, emb_dim) for i in range(3)]
        )
        total_levels = self.args.merge_level + (
            3 if self.args.data_type == "crystal" else 2
        )
        self.cur_level = Embedding(total_levels, emb_dim)

        self.init_tree_and_space_embeding(self.args.data_type, self.args.task_name)

        if self.args.max_num_atom > 0:
            self.remaining_atom_emb = Embedding(self.args.max_num_atom + 1, emb_dim)
            self.remaining_token_emb = Embedding(self.args.max_num_atom + 1, emb_dim)
            self.count_emb = Embedding(self.args.max_num_atom + 2, emb_dim)
        else:
            self.remaining_atom_emb = None
            self.remaining_token_emb = None
            self.count_emb = None

        self.rope = RoPEnD(
            self.head_dim,
            self.args.rope_theta,
            n=3,
        )
        self.decoder = Transformer(
            emb_dim,
            self.args.layer,
            num_head,
            mlp_dim=emb_dim * self.args.ffn_multiple,
            residual_dropout=self.args.residual_dropout,
            attn_dropout=self.args.attn_dropout,
            rope=self.rope,
            checkpoint_activation_threshold=self.args.checkpoint_activation_threshold,
            deterministic=self.args.finetune,
            causal=True,
        )
        self.is_conditional = (
            self.args.crystal_pxrd > 0 or self.args.crystal_component > 0
        )
        self.norm = AdaNorm(emb_dim) if self.is_conditional else RMSNorm(emb_dim)
        self.tree_heads = nn.ModuleList(
            [
                TaskHead(
                    emb_dim,
                    255,
                    dropout=self.args.head_dropout,
                )
                for _ in range(self.num_tree)
            ]
        )
        AtomHead = (
            DiffusionPredictionHead
            if self.args.atom_head_type == "diffusion"
            else AutoRegressivePredictionHead
        )
        self.atom_heads = nn.ModuleList(
            [
                AtomHead(
                    [self.num_atom] + [self.num_xyz] * 3,
                    emb_dim,
                    emb_dropout=self.args.emb_dropout,
                    head_dropout=self.args.head_dropout,
                )
                for _ in range(self.num_tree)
            ]
        )
        if self.args.max_num_atom > 0:
            self.count_head = nn.ModuleList(
                [
                    TaskHead(
                        emb_dim,
                        self.args.max_num_atom + 1,
                        dropout=self.args.head_dropout,
                    )
                    for _ in range(self.num_tree)
                ]
            )
            self.tree_emb_for_count = nn.ModuleList(
                [Embedding(255, emb_dim) for _ in range(self.num_tree)]
            )
        else:
            self.count_head = None
            self.tree_emb_for_count = None

        self.main_tree = self.num_tree - 1

        def str_2_float_list(loss_ratio):
            t = [float(i) for i in loss_ratio.split(",")]
            if len(t) == 1:
                t = t * self.num_tree
            return t

        self.loss_ratio_tree = str_2_float_list(self.args.loss_ratio_tree)
        self.loss_ratio_atom = str_2_float_list(self.args.loss_ratio_atom)
        self.loss_ratio_xyz = str_2_float_list(self.args.loss_ratio_xyz)
        self.loss_ratio_count = str_2_float_list(self.args.loss_ratio_count)
        self.loss_ratio_atom_target = str_2_float_list(self.args.loss_ratio_atom_target)
        self.loss_ratio_mol_target = str_2_float_list(self.args.loss_ratio_mol_target)

        self.tree_temperature = str_2_float_list(self.args.tree_temperature)
        self.atom_temperature = str_2_float_list(self.args.atom_temperature)
        self.xyz_temperature = str_2_float_list(self.args.xyz_temperature)
        self.count_temperature = str_2_float_list(self.args.count_temperature)

        self.init_crystal_cond_embeding()

        if self.args.mol_target_key is not None:
            self.mol_pred_head = ClassificationHead(
                input_dim=self.args.emb_dim,
                inner_dim=self.args.emb_dim,
                num_classes=self.args.mol_num_classes,
                activation_fn=self.args.pooler_activation_fn,  # "tanh",
                pooler_dropout=self.args.pooler_dropout,
            )
        else:
            self.mol_pred_head = None
        if self.args.atom_target_key is not None:
            self.atom_pred_head = ClassificationHead(
                input_dim=self.args.emb_dim,
                inner_dim=self.args.emb_dim,
                num_classes=self.args.atom_num_classes,
                activation_fn=self.args.pooler_activation_fn,  # "tanh",
                pooler_dropout=self.args.pooler_dropout,
            )
        else:
            self.atom_pred_head = None
        assert self.head_dim % 2 == 0
        self._num_updates = 0
        self.dtype = torch.float32

    def init_tree_and_space_embeding(self, data_type, task_name):
        num_tree_mapper = {"crystal": 2}
        self.num_tree = num_tree_mapper.get(data_type, 1)
        self.num_space = 1

        if self.num_tree > 1:
            self.tree_emb = Embedding(self.num_tree, self.args.emb_dim)
        else:
            self.tree_emb = None

        if self.num_space > 1:
            self.space_emb = Embedding(self.num_space, self.args.emb_dim)
        else:
            self.space_emb = None

    def init_crystal_cond_embeding(self):
        self.atom_lattice_id = None
        if self.args.data_type == "crystal":
            self.atom_components_id = self.dictionary["[COMPONENTS]"]
            self.atom_lattice_id = self.dictionary["[LATTICE_O]"]
            if self.args.crystal_pxrd:
                self.atom_pxrd_id_min = self.dictionary["[PXRD_0]"]
                self.atom_pxrd_id_max = self.dictionary[
                    f"[PXRD_{self.args.crystal_pxrd - 1}]"
                ]
            if self.args.crystal_component > 0:
                self.atom_cnt_id_min = self.dictionary["[CNT_0]"]
                self.atom_cnt_id_max = self.dictionary[
                    f"[CNT_{self.dictionary.max_num_atom}]"
                ]
        if self.args.crystal_pxrd:
            self.num_pxrd_per_token = int(
                120 / self.args.crystal_pxrd_step / self.args.crystal_pxrd
            )
            self.pxrd_embed = nn.Sequential(
                Linear(self.num_pxrd_per_token, self.args.emb_dim, init="bert"),
                RMSNorm(self.args.emb_dim),
                nn.GELU(),
                Linear(self.args.emb_dim, self.args.emb_dim, bias=False, init="final"),
            )
            self.pxrd_gate = Linear(
                self.args.emb_dim, self.args.emb_dim, bias=False, init="bert"
            )
        if self.args.crystal_component:
            self.components_embed_tokens = nn.Sequential(
                Linear(128, self.args.emb_dim, init="bert"),
                RMSNorm(self.args.emb_dim),
                nn.GELU(),
                Linear(self.args.emb_dim, self.args.emb_dim, bias=False, init="final"),
            )

    def half(self):
        super().half()
        if self.mol_pred_head is not None:
            self.mol_pred_head = self.mol_pred_head.float()
        if self.atom_pred_head is not None:
            self.atom_pred_head = self.atom_pred_head.float()
        if self.args.atom_head_type == "diffusion":
            for i in range(self.num_tree):
                self.atom_heads[i].diffusion_loss.net.final_layer.linear = (
                    self.atom_heads[i].diffusion_loss.net.final_layer.linear.float()
                )
        self.dtype = torch.half
        return self

    def bfloat16(self):
        super().bfloat16()
        if self.mol_pred_head is not None:
            self.mol_pred_head = self.mol_pred_head.float()
        if self.atom_pred_head is not None:
            self.atom_pred_head = self.atom_pred_head.float()
        if self.args.atom_head_type == "diffusion":
            for i in range(self.num_tree):
                self.atom_heads[i].diffusion_loss.net.final_layer.linear = (
                    self.atom_heads[i].diffusion_loss.net.final_layer.linear.float()
                )
        self.dtype = torch.bfloat16
        return self

    def float(self):
        super().float()
        self.dtype = torch.float32
        return self

    def get_feat_from_dataloader(self, batched_data):
        decoder_type = batched_data["decoder_type"]
        decoder_xyz = batched_data["decoder_xyz"]
        decoder_phy_pos = batched_data["decoder_phy_pos"]
        decoder_level = batched_data["decoder_level"]
        decoder_is_second_atom = batched_data["decoder_is_second_atom"]
        decoder_space_index = batched_data["space_index"]
        decoder_tree_index = batched_data["tree_index"]
        decoder_remaining_atoms = batched_data["decoder_remaining_atoms"]
        decoder_remaining_tokens = batched_data["decoder_remaining_tokens"]
        decoder_count = batched_data["decoder_count"]
        tree_loss_flag = batched_data["tree_loss_flag"]

        is_task_layer = decoder_level >= self.atom_level_index
        is_tree_layer = decoder_level > self.atom_level_index
        is_atom_layer = decoder_level == self.atom_level_index

        has_task = (decoder_type == self.atom_mask_id) & is_task_layer
        decoder_target_type = decoder_type.clone()
        decoder_target_xyz = decoder_xyz.clone()
        decoder_target_count = decoder_count.clone()
        decoder_target_type[has_task] = decoder_type[decoder_is_second_atom]
        decoder_target_xyz[has_task] = decoder_xyz[decoder_is_second_atom]
        decoder_target_count[has_task] = decoder_count[decoder_is_second_atom]

        decoder_target_is_tree = is_tree_layer & has_task
        decoder_target_is_atom = is_atom_layer & has_task

        decoder_target_is_tree &= tree_loss_flag > 0
        decoder_target_is_atom &= tree_loss_flag > 0

        input_is_atom = (
            is_atom_layer
            & (decoder_type != self.atom_mask_id)
            & (decoder_type != self.atom_null_id)
        )
        input_is_cls = decoder_type == self.atom_cls_id
        atom_target = None
        mol_target = None
        atom_index = None
        atom_mol_index = None
        mol_index = None
        rot = None

        # extract understanding targets
        if self.args.atom_target_key is not None:
            atom_target = batched_data["atom_target"]
            atom_index = batched_data["atom_index"]
            atom_mol_index = batched_data["atom_mol_index"]
            if "atom_rot_T" in batched_data:
                rot = batched_data["mol_rot_T"]

        if self.args.mol_target_key is not None:
            mol_target = batched_data["mol_target"]
            mol_index = batched_data["mol_index"]

        decoder_input = DecoderInputFeat(
            decoder_type,
            decoder_xyz,
            decoder_level,
            decoder_phy_pos,
            decoder_space_index,
            decoder_tree_index,
            decoder_remaining_atoms,
            decoder_remaining_tokens,
            decoder_count,
        )
        decoder_gen_target = DecoderGenTargetFeat(
            decoder_target_type,
            decoder_target_xyz,
            decoder_target_count,
            decoder_level,
            decoder_tree_index,
            decoder_target_is_tree,
            decoder_target_is_atom,
        )
        docoder_pred_target = DecoderPredTargetFeat(
            decoder_tree_index,
            input_is_atom,
            input_is_cls,
            atom_target,
            mol_target,
            atom_index=atom_index,
            atom_mol_index=atom_mol_index,
            mol_index=mol_index,
            rot=rot,
        )
        return (
            decoder_input,
            decoder_gen_target,
            docoder_pred_target,
        )

    def input_embedding(
        self,
        decoder_input,
    ):
        emb_x = self.atom_emb(decoder_input.type)
        for i in range(3):
            emb_x = emb_x + self.xyz_emb[i](decoder_input.xyz[..., i])
        emb_x = emb_x + self.cur_level(decoder_input.level + 1)
        if self.tree_emb is not None:
            emb_x += self.tree_emb(decoder_input.tree_index)
        if self.space_emb is not None:
            emb_x += self.space_emb(decoder_input.space_index)

        if self.count_emb is not None:
            emb_x += self.remaining_atom_emb(decoder_input.remaining_atoms)
            emb_x += self.remaining_token_emb(decoder_input.remaining_tokens)
            emb_x += self.count_emb(decoder_input.count + 1)

        emb_x = F.dropout(emb_x, self.args.emb_dropout, self.training)
        return emb_x

    def forward_crystal_cond(self, bsz, dec_y, decoder_input, pxrd, components):
        c = 0
        if self.args.crystal_pxrd:
            pxrd_mask = (
                (decoder_input.level == self.args.merge_level + 1)
                & (decoder_input.type >= self.atom_pxrd_id_min)
                & (decoder_input.type <= self.atom_pxrd_id_max)
            )
            pxrd_emb = self.pxrd_embed(pxrd.to(self.dtype)).view(-1, self.args.emb_dim)
            pxrd_emb = F.dropout(pxrd_emb, self.args.emb_dropout, self.training)
            dec_y[pxrd_mask] += pxrd_emb
            cond = dec_y[pxrd_mask]
            gate_cond = F.silu(self.pxrd_gate(cond))
            n_cond = self.args.crystal_pxrd
            c += (gate_cond * cond).view(-1, n_cond, self.args.emb_dim).sum(dim=1)
        components_mask = decoder_input.type == self.atom_components_id
        if components_mask.any():
            comp_emb = self.components_embed_tokens(components.type_as(dec_y)).view(
                -1, self.args.emb_dim
            )
            comp_emb = F.dropout(comp_emb, self.args.emb_dropout, self.training)
            dec_y[components_mask] += comp_emb
            c += comp_emb
            is_cnt = (decoder_input.type >= self.atom_cnt_id_min) & (
                decoder_input.type <= self.atom_cnt_id_max
            )
            c += dec_y[is_cnt]
        return dec_y, c

    def forward_model(
        self,
        decoder_input,
        c=None,
        decoder_input_len=None,
        decoder_input_max_len=None,
        kv_cache_list=None,
        need_norm=True,
        **kwargs,
    ):

        dec_y = self.input_embedding(
            decoder_input,
        )
        if ("pxrd" in kwargs and kwargs["pxrd"] is not None) or (
            "components" in kwargs and kwargs["components"] is not None
        ):
            bsz = (
                dec_y.size(0)
                if decoder_input_len is None
                else decoder_input_len.shape[0] - 1
            )
            dec_y, c = self.forward_crystal_cond(
                bsz, dec_y, decoder_input, kwargs["pxrd"], kwargs["components"]
            )
            if decoder_input_len is not None:
                repeats = decoder_input_len[1:] - decoder_input_len[:-1]
                c = c.repeat_interleave(repeats, dim=0)

        self.rope.calc_and_cache(decoder_input.phy_pos / self.args.xyz_resolution)
        for i in range(self.args.recycle):
            dec_y = self.decoder(
                dec_y,
                decoder_input_len,
                decoder_input_max_len,
                kv_cache=kv_cache_list[i] if kv_cache_list is not None else None,
            )

        if need_norm:
            if self.is_conditional:
                dec_y = self.norm(dec_y, c)
            else:
                dec_y = self.norm(dec_y)
        return dec_y, c

    def forward_gen_heads(
        self,
        dec_y,
        decoder_target,
    ):
        level = decoder_target.level
        is_gen_tree = decoder_target.is_gen_tree
        is_gen_atom = decoder_target.is_gen_atom
        preds = []
        gts = []
        names = []
        counts = []
        loss_ratios = []
        sample_size = 0

        def one_tree(tree_i, sample_size):
            name_post_fix = "" if tree_i == self.main_tree else f"_tree{tree_i}"
            is_current_tree = decoder_target.tree_index == tree_i
            cur_is_gen_tree = is_gen_tree & is_current_tree
            cur_is_gen_atom = is_gen_atom & is_current_tree
            num_tree_nodes = cur_is_gen_tree.long().sum()
            num_atoms = (
                ((level == self.atom_level_index) & (cur_is_gen_atom)).long().sum()
            )
            if num_tree_nodes > 0:
                y_type = dec_y[cur_is_gen_tree]
                preds_type = self.tree_heads[tree_i](y_type)
                gts_type = (
                    decoder_target.type[cur_is_gen_tree] - self.tree0_split_id - 1
                )
                preds.append(preds_type)
                gts.append(gts_type)
                names.append("type_father" + name_post_fix)
                counts.append(num_tree_nodes)
                loss_ratios.append(self.loss_ratio_tree[tree_i])
                if self.count_head is not None:
                    y_type = y_type + F.dropout(
                        self.tree_emb_for_count[tree_i](gts_type),
                        self.args.emb_dropout,
                        self.training,
                    )
                    preds_count = self.count_head[tree_i](y_type)
                    gts_count = decoder_target.count[cur_is_gen_tree]
                    preds.append(preds_count)
                    gts.append(gts_count)
                    names.append("count_father" + name_post_fix)
                    counts.append(num_tree_nodes)
                    loss_ratios.append(self.loss_ratio_count[tree_i])

                sample_size += num_tree_nodes

            if num_atoms > 0:
                gts_xyz = decoder_target.to_list(cur_is_gen_atom)
                preds_xyz = self.atom_heads[tree_i](dec_y[cur_is_gen_atom], gts_xyz)
                gts_atom = gts_xyz[0]
                preds_atom = preds_xyz[0]
                preds_xyz = [preds_xyz[i] for i in range(1, 4)]
                gts_xyz = [gts_xyz[i] for i in range(1, 4)]
                preds.extend([preds_xyz, preds_atom])
                gts.extend([gts_xyz, gts_atom])
                names.extend(
                    [
                        "pos_atom" + name_post_fix,
                        "type_atom" + name_post_fix,
                    ]
                )
                loss_ratios.extend(
                    [
                        self.loss_ratio_xyz[tree_i],
                        self.loss_ratio_atom[tree_i],
                    ]
                )
                counts.extend([num_atoms, num_atoms])
                if self.count_head is not None:
                    # to predict the atom count
                    preds_atom_count = self.count_head[tree_i](dec_y[cur_is_gen_atom])
                    gts_atom_count = decoder_target.count[cur_is_gen_atom]
                    preds.append(preds_atom_count)
                    gts.append(gts_atom_count)
                    names.append("count_atom" + name_post_fix)
                    counts.append(num_atoms)
                    loss_ratios.append(self.loss_ratio_count[tree_i])

                sample_size += num_atoms
            return sample_size

        for i in range(self.num_tree):
            sample_size = one_tree(i, sample_size)

        return (
            preds,
            gts,
            names,
            counts,
            loss_ratios,
            sample_size,
        )

    def forward_pred_heads(self, dec_y, decoder_pred_target):
        preds = []
        gts = []
        names = []
        counts = []
        loss_ratios = []
        keys = []
        tree_index = decoder_pred_target.tree_index
        is_input_atom = decoder_pred_target.is_input_atom
        is_input_cls = decoder_pred_target.is_input_cls
        atom_index = decoder_pred_target.atom_index
        atom_mol_index = decoder_pred_target.atom_mol_index
        mol_index = decoder_pred_target.mol_index
        is_atom_tree_index = tree_index[is_input_atom]
        is_cls_tree_index = tree_index[is_input_cls]

        def one_tree(tree_i, sample_size):
            name_post_fix = "" if tree_i == self.main_tree else f"_tree{tree_i}"
            cur_is_input_atom = is_input_atom & (tree_index == tree_i)
            cur_is_input_cls = is_input_cls & (tree_index == tree_i)
            num_atoms = cur_is_input_atom.long().sum()
            num_cls = cur_is_input_cls.long().sum()
            if self.atom_pred_head is not None and num_atoms > 0:
                y = dec_y[cur_is_input_atom, :].float()
                pred_atom = self.atom_pred_head(y)
                gt_atom = decoder_pred_target.atom_target
                gt_atom = gt_atom[is_atom_tree_index == tree_i]
                cur_atom_index = atom_index[is_atom_tree_index == tree_i]
                cur_atom_mol_index = atom_mol_index[is_atom_tree_index == tree_i]
                if decoder_pred_target.rot is not None:
                    pred_atom = pred_atom.view(-1, 1, 3) @ decoder_pred_target.rot
                    pred_atom = pred_atom.view(-1, 3)
                preds.append(pred_atom)
                gts.append(gt_atom)
                names.append("atom_target" + name_post_fix)
                counts.append(num_atoms)
                loss_ratios.append(self.loss_ratio_atom_target[tree_i])
                keys.append([cur_atom_index, cur_atom_mol_index])
                sample_size += num_atoms

            if self.mol_pred_head is not None and num_cls > 0:
                y = dec_y[cur_is_input_cls, :].float()
                pred_mol = self.mol_pred_head(y)
                gt_mol = decoder_pred_target.mol_target
                gt_mol = gt_mol[is_cls_tree_index == tree_i]
                cur_mol_index = mol_index[is_cls_tree_index == tree_i]
                preds.append(pred_mol)
                gts.append(gt_mol)
                names.append("mol_target" + name_post_fix)
                counts.append(num_cls)
                loss_ratios.append(self.loss_ratio_mol_target[tree_i])
                keys.append(cur_mol_index)
                sample_size += num_cls
            return sample_size

        sample_size = 0
        for i in range(self.num_tree):
            sample_size = one_tree(i, sample_size)

        return (
            preds,
            gts,
            names,
            counts,
            loss_ratios,
            keys,
            sample_size,
        )

    def forward(self, batched_data):
        try:
            (
                decoder_input,
                decoder_gen_target,
                decoder_pred_target,
            ) = self.get_feat_from_dataloader(batched_data)

            dec_y, _ = self.forward_model(
                decoder_input,
                decoder_input_len=batched_data["decoder_input_len"],
                decoder_input_max_len=int(batched_data["decoder_input_max_len"]),
                pxrd=batched_data["pxrd"] if self.args.crystal_pxrd else None,
                components=(
                    batched_data["components"] if self.args.crystal_component else None
                ),
            )
            gen_outputs = self.forward_gen_heads(
                dec_y,
                decoder_gen_target,
            )
            pred_outputs = self.forward_pred_heads(dec_y, decoder_pred_target)

            return (
                gen_outputs,
                pred_outputs,
                batched_data["decoder_input_len"],
                batched_data["time"],
            )

        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.warning(
                    f'OOM, decoder input shape {batched_data["decoder_phy_pos"].shape}'
                )
            raise e

    @classmethod
    def build_model(cls, args, task):
        return cls(args, task.dictionary)

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        self._num_updates = num_updates


@register_model_architecture("uni3dar", "uni3dar")
def base_architecture(args):
    args.emb_dim = getattr(args, "emb_dim", 768)
    args.layer = getattr(args, "layer", 12)
    args.recycle = getattr(args, "recycle", 1)
    args.num_head = getattr(args, "num_head", 12)
    args.rope_theta = getattr(args, "rope_theta", 10000)
    args.residual_dropout = getattr(args, "residual_dropout", 0.1)
    args.ffn_multiple = getattr(args, "ffn_multiple", 4)
    args.emb_dropout = getattr(args, "emb_dropout", 0.1)
    args.attn_dropout = getattr(args, "attn_dropout", 0.1)
    args.head_dropout = getattr(args, "head_dropout", 0.1)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.1)
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.finetune = getattr(args, "finetune", False)
