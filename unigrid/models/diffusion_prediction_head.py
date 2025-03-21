import math
import torch
from torch import nn
from .layers import TaskHead
from torch.utils.checkpoint import checkpoint
from unicore.modules import LayerNorm
from .diffusion import create_diffusion


class DiffusionPredictionHead(nn.Module):
    """Head for masked autoregressive with diffusion loss."""

    def __init__(
        self,
        feat_size_list,
        emb_dim,
        emb_dropout=0.1,
        head_dropout=0.0,
        diff_mul=4,
    ):
        super().__init__()

        self.emb_dim = emb_dim
        self.dim = emb_dim

        self.num_feat = len(feat_size_list)

        self.dropout = nn.Dropout(emb_dropout)

        # suppose atom type is the first feature
        self.type_pred_head = TaskHead(emb_dim, feat_size_list[0], dropout=head_dropout)
        self.num_xyz = float(feat_size_list[1])
        self.diff_mul = diff_mul

        self.diffusion_loss = DiffLoss(
            target_channels=3,
            z_channels=emb_dim,
            depth=3,
            width=emb_dim,
            num_sampling_steps="100",
            grad_checkpointing=False,
        )
        self.xyz_normalize_for_diffusion = lambda x: x / self.num_xyz
        self.xyz_denormalize_for_diffusion = lambda x: x * self.num_xyz

        # this is for embedding the atom type. pred_head is used to predict the atom type
        self.atom_type_embedding = nn.Embedding(
            num_embeddings=feat_size_list[0], embedding_dim=emb_dim
        )

    def forward(self, x, decoder_target):
        # 1. Embed the atom type (used for xyz prediction)
        atom_type = decoder_target[0]
        z_atom = self.dropout(self.atom_type_embedding(atom_type))

        # Use advanced indexing (this will run on the GPU)
        # Here, decoder_target[keep_indices, :] has shape [num_excluded, ...],
        # so we transpose it to match the expected shape.
        target_wo_type = torch.stack(decoder_target[1:]).transpose(0, 1).float()
        target_wo_type = self.xyz_normalize_for_diffusion(target_wo_type)

        # 3. Merge the atom type embedding with the hidden state x
        z = z_atom + x

        # 4. Compute the diffusion loss for predicting xyz.
        target = target_wo_type.repeat(self.diff_mul, 1)
        z = z.repeat(self.diff_mul, 1)
        diffusion_loss = self.diffusion_loss(
            target=target,
            z=z,
        )

        n_atoms = target_wo_type.size(0)
        diffusion_loss = diffusion_loss * n_atoms

        # 5. Predict the atom type from the hidden state.
        atom_type_logits = self.type_pred_head(x)

        # 6. Build the output list.
        pred_logits = [atom_type_logits] + [
            diffusion_loss for _ in range(self.num_feat - 1)
        ]

        return pred_logits

    def inference(self, x, sampling_func_list, **kwargs):
        """Inference autogressive prediction, with temperature scaling."""
        bsz = x.size(0)
        feat = x.view(bsz, self.emb_dim)
        bsz = feat.size(0)
        pred_xyz = torch.zeros(
            bsz, self.num_feat, device=feat.device, dtype=torch.int32
        )
        score = torch.zeros(bsz, device=feat.device, dtype=torch.float32)

        # first predict the atom type
        atom_type_logits = self.type_pred_head(feat)
        pred_xyz[:, 0], cur_score = sampling_func_list[0](atom_type_logits)
        score += cur_score

        # then predict the xyz
        z = self.atom_type_embedding(pred_xyz[:, 0])
        z = z + feat

        # sample the xyz with diffusion loss
        xyz_ = self.diffusion_loss.sample(z, kwargs["xyz_temperature"])
        xyz_ = self.xyz_denormalize_for_diffusion(xyz_)

        # to ensure pred_xyz is valid, constrain each value into (0, num_xyz-1)
        xyz_ = torch.clamp(xyz_, 0, self.num_xyz - 1)

        pred_xyz[:, 1:] = torch.round(xyz_)

        return pred_xyz, score


class DiffLoss(nn.Module):
    """Diffusion Loss"""

    def __init__(
        self,
        target_channels,
        z_channels,
        depth,
        width,
        num_sampling_steps,
        grad_checkpointing=False,
    ):
        super(DiffLoss, self).__init__()
        self.in_channels = target_channels
        self.net = SimpleMLPAdaLN(
            in_channels=target_channels,
            model_channels=width,
            out_channels=target_channels * 2,  # for vlb loss
            z_channels=z_channels,
            num_res_blocks=depth,
            grad_checkpointing=grad_checkpointing,
        )

        self.train_diffusion = create_diffusion(
            timestep_respacing="", noise_schedule="cosine"
        )
        self.gen_diffusion = create_diffusion(
            timestep_respacing=num_sampling_steps, noise_schedule="cosine"
        )

    def forward(self, target, z, mask=None):
        t = torch.randint(
            0,
            self.train_diffusion.num_timesteps,
            (target.shape[0],),
            device=target.device,
        )
        model_kwargs = dict(c=z)
        loss_dict = self.train_diffusion.training_losses(
            self.net, target, t, model_kwargs
        )
        loss = loss_dict["loss"]
        if mask is not None:
            loss = (loss * mask).sum() / mask.sum()
        return loss.mean()

    def sample(self, z, temperature=1.0, cfg=1.0):
        # diffusion loss sampling
        if not cfg == 1.0:
            noise = torch.randn(z.shape[0] // 2, self.in_channels).cuda()
            noise = torch.cat([noise, noise], dim=0)
            model_kwargs = dict(c=z, cfg_scale=cfg)
            sample_fn = self.net.forward_with_cfg
        else:
            noise = torch.randn(z.shape[0], self.in_channels).cuda()
            model_kwargs = dict(c=z)
            sample_fn = self.net.forward

        sampled_token_latent = self.gen_diffusion.p_sample_loop(
            sample_fn,
            noise.shape,
            noise,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
            temperature=temperature,
        )

        return sampled_token_latent


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq.type_as(self.mlp[0].weight))
        return t_emb


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        self.in_ln = LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """

    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            model_channels, elementwise_affine=False, eps=1e-6
        )
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x.type_as(self.linear.weight))
        return x


class SimpleMLPAdaLN(nn.Module):
    """
    The MLP for Diffusion Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param z_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
        grad_checkpointing=False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)

        self.input_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(
                ResBlock(
                    model_channels,
                )
            )

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, c):
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C] Tensor of outputs.
        """
        x = self.input_proj(x.type_as(self.input_proj.weight))
        t = self.time_embed(t)
        c = self.cond_embed(c)

        y = t + c

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.res_blocks:
                x = checkpoint(block, x, y)
        else:
            for block in self.res_blocks:
                x = block(x, y)

        return self.final_layer(x, y)

    def forward_with_cfg(self, x, t, c, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, c)
        eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
