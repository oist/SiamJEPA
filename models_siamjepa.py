# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026 Makoto Yamada and contributors.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------


# Note on changes made after the paper: the paper's results were produced with
# the implementation at commit 8ad5771. Later changes marked "[post-paper]"
# below (speed-ups, an encoder-only EMA teacher, removal of never-used modules)
# do not change what is computed in the default fp32 setting, except that the
# random masks are drawn from a different random stream (same distribution);
# see bench/check_speedup_equivalence.py. The view2 predictor placement fix is
# opt-in (fix_view2_restore, default off = the paper setting).

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block
from timm.models.vision_transformer import CrossAttention, Attention, DropPath, Mlp

from util.pos_embed import get_2d_sincos_pos_embed

import copy
import torch.distributions as td
import torch.utils.checkpoint
import torch.nn.functional as F

def ema_model(modelA, modelB, m):
    with torch.no_grad():
        for paramA, paramB in zip(modelA.parameters(), modelB.parameters()):
            paramA.data = m * paramA.data + (1 - m) * paramB.data
    return modelA

class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=768, out_dim=768):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
        self.num_layers = 3
    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x 

class CSABlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm_kv1 = norm_layer(dim)
        self.cattn = CrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        mlp_hidden_dim = int(dim * mlp_ratio)

        '''self.mlp1 = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )'''
        self.norm3 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm4 = norm_layer(dim)
        self.mlp2 = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        # Run this block in fp32 even under bf16 autocast (the historical,
        # conservative setting). Set False to let the predictor run in bf16.
        self.force_fp32 = True

    def forward(self, x, kvx, src_mask=None):
        if not self.force_fp32:
            x = x + self.drop_path(self.cattn(self.norm1(x), self.norm_kv1(kvx), src_mask=src_mask))
            x = x + self.drop_path(self.attn(self.norm3(x)))
            x = x + self.drop_path(self.mlp2(self.norm4(x)))
            return x

        with torch.cuda.amp.autocast(enabled=False):
            x_n  = self.norm1(x.float())
            kv_n = self.norm_kv1(kvx.float())
            ca = self.cattn(x_n, kv_n, src_mask=src_mask)
        x = x + self.drop_path(ca.to(x.dtype))

        #x = x + self.mlp1(self.norm2(x))
        with torch.cuda.amp.autocast(enabled=False):
            sa = self.attn(self.norm3(x).float())
        x = x + self.drop_path(sa.to(x.dtype))
        #x = x + self.drop_path(self.attn(self.norm3(x)))

        with torch.cuda.amp.autocast(enabled=False):
            m = self.mlp2(self.norm4(x).float())
        x = x + self.drop_path(m.to(x.dtype))

        #x = x + self.drop_path(self.mlp2(self.norm4(x)))
        #x = x + self.mlp2(self.norm4(x))
        return x

        #x = x + self.drop_path(
        #    self.cattn(self.norm1(x), self.norm_kv1(kvx), src_mask=src_mask)
        #)
        #x = x + self.mlp1(self.norm2(x))
        #x = x + self.drop_path(self.attn(self.norm3(x)))
        #x = x + self.mlp2(self.norm4(x))
        #return x

def chk(name, t):
    ok = torch.isfinite(t).all()
    if not ok:
        print(name, "NOT finite",
              "min", torch.nanmin(t).item(),
              "max", torch.nanmax(t).item())
    return ok

class SiamJEPA(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    # submodules / parameters kept in the (encoder-only) EMA teacher.
    # [post-paper] the paper's runs kept a full copy of the model as the EMA
    # teacher; only its encoder is ever used, so the teacher output is identical.
    EMA_ENCODER_KEYS = ('patch_embed', 'cls_token', 'pos_embed', 'blocks', 'norm')
    # never-used modules removed from the model; older checkpoints still carry them.
    # [post-paper] they existed (without being used in forward()) in the paper's runs.
    REMOVED_MODULES = ('decoder_embed', 'decoder_pred_latent', 'decoder_pred', 'decoder_embed_mae')

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,stoch=32,
        discrete=32,kl_scale=0.01,
        kl_balance=0.2,kl_freebit=0.1,beta=0.996,mask_ratio=0.9,
        shuffle_teacher=False, fix_view2_restore=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # SiamJEPA encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # SiamJEPA decoder (predictor) specifics
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList(
            [
                CSABlock(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)

        stoch_size = stoch * discrete if discrete != 0 else stoch * 2
        self.decoder_embed_deter = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_embed_stoch = nn.Linear(stoch_size, decoder_embed_dim, bias=True)

        # Posterior takes both src_h and tgt_h
        # Thus it has embed_dim * 2 as an input dimension
        self.to_posterior = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, stoch_size),
        )

        # Prior only takes src_h
        # Thus it has embed_dim as an input dimension
        self.to_prior = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, stoch_size),
        )

        # width follows the encoder (its output feeds ca3 = Linear(embed_dim, embed_dim));
        # identical to the old fixed 768/768 for ViT-B
        self.projector = projection_MLP(embed_dim, hidden_dim=embed_dim, out_dim=embed_dim)
        self.ca3 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
        )

        self.norm_pix_loss = norm_pix_loss

        # Random Shuffle Teacher (RST): see forward_encoder() below. Set before
        # the EMA deep-copy so both branches carry the same setting.
        self.shuffle_teacher = shuffle_teacher

        # Opt-in fix (default off = the behaviour used for the paper's results):
        # by default the view2 predictor reuses view1's ids_restore, which puts
        # view2's context tokens in view1's slots (so they get view1's
        # decoder_pos_embed). With fix_view2_restore=True, view2 gets its own
        # ids_restore. Target slots / the loss are the same either way.
        self.fix_view2_restore = fix_view2_restore

        self.initialize_weights()

        self.beta=beta
        self.ema_model = copy.deepcopy(self)
        # The teacher is only ever used through forward_encoder(), so keep
        # just the encoder in the EMA copy (drops the predictor, posterior/
        # prior heads, projector, ... -- less memory and a cheaper EMA update).
        for name in list(self.ema_model._modules):
            if name not in self.EMA_ENCODER_KEYS:
                delattr(self.ema_model, name)
        for name in list(self.ema_model._parameters):
            if name not in self.EMA_ENCODER_KEYS:
                delattr(self.ema_model, name)
        self.ema_model.eval()   # ← train() ではなく eval()
        for p in self.ema_model.parameters():
            p.requires_grad = False

        self.stoch = stoch
        self.discrete = discrete
        self.kl_balance = kl_balance
        self.kl_scale = kl_scale
        self.kl_freebit=kl_freebit

        self.mask_ratio=mask_ratio

        # recompute the student encoder blocks in backward instead of storing
        # their activations (less memory, ~30% more compute); set from the
        # training script, not saved in checkpoints
        self.grad_checkpointing = False


    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking_dual(self, x, mask_ratio):
        """
        x: [N, L, D]
        ids_keep1 と ids_keep2 は overlap なし。
        ランダム正方形ブロックは両viewから除外。
        残りから非重複に ids_keep1, ids_keep2 を選ぶ。
        """
        N, L, D = x.shape
        device = x.device

        H = W = int(L ** 0.5)
        assert H * W == L

        len_keep = int(L * (1 - mask_ratio))
        assert 2 * len_keep <= L

        block_area = L - 2 * len_keep
        block_size = int(block_area ** 0.5)
        block_size = max(1, block_size)

        # [post-paper] batched over samples; the paper's runs used a per-sample
        # Python loop with the same distribution but a different random stream.
        # Batched over samples (formerly a per-sample Python loop): one random
        # block_size x block_size square per sample is excluded from both
        # views; the patches outside it are put in uniformly random order
        # (argsort of iid noise), and the first 2*len_keep become the keep
        # sets of view1/view2. Block patches get noise >= 1, so they always
        # sort after every outside patch.
        top = torch.randint(0, H - block_size + 1, (N, 1, 1), device=device)
        left = torch.randint(0, W - block_size + 1, (N, 1, 1), device=device)
        rows = torch.arange(H, device=device).view(1, H, 1)
        cols = torch.arange(W, device=device).view(1, 1, W)
        is_block = ((rows >= top) & (rows < top + block_size)
                    & (cols >= left) & (cols < left + block_size)).reshape(N, L)

        noise = torch.rand(N, L, device=device) + is_block.float()

        # 先頭 2*len_keep だけが view1/view2 に使われる
        # それ以降は両方から mask される
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep1 = ids_shuffle[:, :len_keep]
        ids_keep2 = ids_shuffle[:, len_keep:2 * len_keep]

        x_masked1 = torch.gather(
            x, dim=1,
            index=ids_keep1.unsqueeze(-1).repeat(1, 1, D)
        )

        x_masked2 = torch.gather(
            x, dim=1,
            index=ids_keep2.unsqueeze(-1).repeat(1, 1, D)
        )

        mask1 = torch.ones([N, L], device=device)
        mask2 = torch.ones([N, L], device=device)

        mask1[:, :len_keep] = 0
        mask2[:, len_keep:2 * len_keep] = 0

        mask1 = torch.gather(mask1, dim=1, index=ids_restore)
        mask2 = torch.gather(mask2, dim=1, index=ids_restore)

        return x_masked1, x_masked2, mask1, mask2, ids_restore

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder_dual(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x1, x2, mask1,mask2, ids_restore = self.random_masking_dual(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x1 = torch.cat((cls_tokens, x1), dim=1)
        x2 = torch.cat((cls_tokens, x2), dim=1)

        # [post-paper] (the paper's runs applied the blocks to x1 and x2 separately)
        # apply Transformer blocks to both views as one 2N batch (both views
        # have the same length, and every op is per-sample, so this is
        # identical to running them separately -- just fewer, larger kernels)
        x12 = torch.cat((x1, x2), dim=0)
        for blk in self.blocks:
            if self.grad_checkpointing and self.training:
                x12 = torch.utils.checkpoint.checkpoint(blk, x12, use_reentrant=False)
            else:
                x12 = blk(x12)
        x12 = self.norm(x12)
        x1, x2 = x12.chunk(2, dim=0)

        return x1, x2, mask1, mask2, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # forward_encoder is only ever called with mask_ratio=0 (full pass over
        # the teacher/EMA branch).
        assert mask_ratio == 0, "forward_encoder no longer supports masking; call with mask_ratio=0"

        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        if self.shuffle_teacher:
            # Random Shuffle Teacher (RST): randomly permute the patch tokens
            # (independently per sample) before the transformer blocks see
            # them. The positional embedding just added above still reflects
            # each token's true grid position, so this breaks the
            # correspondence between a token's position and its content --
            # the teacher's output at slot i is now some other patch's
            # representation. Predicting against this shuffled target (Sim-2)
            # removes the shortcut of relying on spatial position and pushes
            # the encoder toward semantic, position-invariant patch
            # representations. Intended to be combined with a curriculum:
            # train with --shuffle_teacher first, then continue without it
            # (via --init_checkpoint) to restore spatial correspondence.
            x, _, _ = self.random_masking(x, mask_ratio=0.0)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        N, L = x.shape[0], x.shape[1] - 1
        mask = torch.zeros([N, L], device=x.device)
        ids_restore = torch.arange(L, device=x.device).unsqueeze(0).expand(N, -1)

        return x, mask, ids_restore


    def forward_predictor(self, x, ids_restore,z):
        # embed tokens
        x = self.decoder_embed_deter(x)

        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        x = x + self.decoder_pos_embed
        
        if self.discrete != 0:
            z = z.reshape(*z.shape[:-2], 1, self.stoch * self.discrete)
        z = self.decoder_embed_stoch(z)
        kvx_h = torch.cat([z, x], dim=1)

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x, kvx=kvx_h)
        
        
        x_latent = self.decoder_norm(x)
        x_latent = x_latent[:,1:,:]

        return x_latent
    
    @torch.no_grad()
    def update_ema_model(self):
        if isinstance(self, torch.nn.parallel.DistributedDataParallel):
            model = self.module 
        else:
            model = self

        # pair by name (the EMA copy is encoder-only, so positional zip over
        # model.parameters() would no longer line up), then one fused update
        student = dict(model.named_parameters())
        ema_params = []
        student_params = []
        for name, ema_param in model.ema_model.named_parameters():
            ema_params.append(ema_param.data)
            student_params.append(student[name].data)
        torch._foreach_mul_(ema_params, model.beta)
        torch._foreach_add_(ema_params, student_params, alpha=1 - model.beta)

    @classmethod
    def _is_legacy_key(cls, key, own_keys):
        # keys that older checkpoints carry but this model no longer has: the
        # non-encoder part of the EMA copy, and REMOVED_MODULES (student or EMA)
        if key in own_keys:
            return False
        name = key[len('ema_model.'):] if key.startswith('ema_model.') else key
        return key.startswith('ema_model.') or name.split('.')[0] in cls.REMOVED_MODULES

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        # Older checkpoints also carry ema_model.{decoder_*, projector, ...}
        # and the removed unused modules; drop those so they still load with
        # strict=True (--resume/--init_checkpoint).
        own_keys = set(self.state_dict().keys())
        stale = [k for k in state_dict if self._is_legacy_key(k, own_keys)]
        if stale:
            state_dict = {k: v for k, v in state_dict.items() if k not in stale}
            print(f"load_state_dict: dropped {len(stale)} keys this model no longer has "
                  f"(non-encoder ema_model.*, removed unused modules)")
        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    def convert_legacy_optimizer_state(self, optim_state, model_state):
        """Make an optimizer state saved together with an older checkpoint
        (which still had REMOVED_MODULES) loadable into an optimizer built by
        optim_factory.add_weight_decay over this model: drop the removed
        parameters' entries and renumber the rest."""
        own_keys = set(self.state_dict().keys())
        if not any(self._is_legacy_key(k, own_keys) and not k.startswith('ema_model.')
                   for k in model_state):
            return optim_state
        trainable = [n for n, p in self.named_parameters() if p.requires_grad]
        trainable_set = set(trainable)
        # the old model's trainable parameters, in named_parameters() order
        # (state_dict order, minus buffers / frozen pos_embeds / the EMA copy)
        old_names = [k for k in model_state
                     if k in trainable_set or (not k.startswith('ema_model.')
                                               and self._is_legacy_key(k, own_keys))]

        def groups(names):  # same split and order as optim_factory.add_weight_decay
            no_decay = [n for n in names if model_state[n].ndim == 1 or n.endswith('.bias')]
            decay = [n for n in names if n not in set(no_decay)]
            return [no_decay, decay]

        old_groups = groups(old_names)
        new_names = [n for n in old_names if n in trainable_set]
        assert new_names == trainable, "parameter order differs from the checkpoint"
        new_groups = groups(new_names)
        assert [len(g) for g in old_groups] == [len(g['params']) for g in optim_state['param_groups']], \
            "optimizer state does not match the checkpoint's parameters"

        old_index = {n: i for i, n in enumerate(old_groups[0] + old_groups[1])}
        new_index = {n: i for i, n in enumerate(new_groups[0] + new_groups[1])}
        state = {new_index[n]: optim_state['state'][old_index[n]]
                 for n in new_index if old_index[n] in optim_state['state']}
        param_groups = []
        for g, names in zip(optim_state['param_groups'], new_groups):
            param_groups.append({**g, 'params': [new_index[n] for n in names]})
        print(f"convert_legacy_optimizer_state: dropped "
              f"{len(old_index) - len(new_index)} removed parameters from the optimizer state")
        return {'state': state, 'param_groups': param_groups}

    def get_feat(self, h, z,ids):
        h = self.decoder_embed_deter(h) + self.decoder_pos_embed
        if self.discrete != 0:
            z = z.reshape(*z.shape[:-2], 1, self.stoch * self.discrete)
        z = self.decoder_embed_stoch(z)
        feat = torch.cat([z, h], dim=1)
        return feat

    def make_dist(self, logits):
        if self.discrete != 0:
            logits = logits.reshape([-1, self.stoch, self.discrete])
            dist = td.Independent(td.OneHotCategoricalStraightThrough(logits=logits), 1)
        else:
            mean, std = torch.split(logits, 2, -1)
            std = F.softplus(std) + 1e-4
            dist = td.Normal(mean, std)
        return dist

    def kl_loss(self, post_logits, prior_logits):
        balance = self.kl_balance
        freebit = self.kl_freebit
        post_to_prior_kl = td.kl_divergence(
            self.make_dist(post_logits), self.make_dist(prior_logits.detach())
        )
        prior_to_post_kl = td.kl_divergence(
            self.make_dist(post_logits.detach()), self.make_dist(prior_logits)
        )
        kl_value = (
            post_to_prior_kl * balance + prior_to_post_kl * (1.0 - balance)
        ).mean()
        kl_loss = torch.maximum(kl_value, torch.ones_like(kl_value) * freebit)
        return kl_loss, kl_value


    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, src_imgs):
        #Encoders
        
        src_h, src_p, mask_src, mask_tgt, ids_restore = self.forward_encoder_dual(src_imgs, mask_ratio=self.mask_ratio)

        #f_long encoders
        with torch.no_grad():
            src_z, _, _ = self.ema_model.forward_encoder(src_imgs, mask_ratio=0)

        #CA3    
        src_h_ca3_cls = self.ca3(self.projector(src_h[:, 0]))
        src_h_ca3 = self.ca3(src_h)

        # Posterior distribution from both images
        post_h1 = torch.cat([src_h_ca3_cls, self.projector(src_p[:, 0])], -1)
        post_logits1 = self.to_posterior(post_h1).float()
        post_logits1 = post_logits1.clamp(-20, 20)
            
        post_dist1 = self.make_dist(post_logits1)
        post_z1 = post_dist1.rsample()

        # Prior distribution only from current images
        prior_h1 = src_h_ca3_cls

        prior_logits1 = self.to_prior(prior_h1.detach()).float()
        prior_logits1 = prior_logits1.clamp(-20, 20)

        src_p_ca3_cls = self.ca3(self.projector(src_p[:, 0]))
        src_p_ca3 = self.ca3(src_p)

        # Posterior distribution from both images
        post_p1 = torch.cat([src_p_ca3_cls, self.projector(src_h[:, 0])], -1)
        post_logits2 = self.to_posterior(post_p1).float()
        post_logits2 = post_logits2.clamp(-20, 20)
            
        post_dist2 = self.make_dist(post_logits2)
        post_z2 = post_dist2.rsample()

        # Prior distribution only from current images
        prior_p1 = src_p_ca3_cls

        prior_logits2 = self.to_prior(prior_p1.detach()).float()
        prior_logits2 = prior_logits2.clamp(-20, 20)

        #Predictor g
        if self.fix_view2_restore:
            # ids_shuffle = [view1 keep | view2 keep | masked in both];
            # swap the first two segments so view2's tokens land in their own slots
            len_keep = src_p.shape[1] - 1
            ids_shuffle = torch.argsort(ids_restore, dim=1)
            ids_restore2 = torch.argsort(torch.cat(
                [ids_shuffle[:, len_keep:2 * len_keep], ids_shuffle[:, :len_keep],
                 ids_shuffle[:, 2 * len_keep:]], dim=1), dim=1)
        else:
            ids_restore2 = ids_restore

        src_pred = self.forward_predictor(src_h_ca3, ids_restore, post_z1)
        tgt_pred = self.forward_predictor(src_p_ca3, ids_restore2, post_z2)

        # KL in fp32 regardless of the autocast mode (the logits are already fp32)
        with torch.cuda.amp.autocast(enabled=False):
           post_logits1_f = post_logits1.float()
           prior_logits1_f = prior_logits1.float()
           kl_loss1, kl_value1 = self.kl_loss(post_logits1_f, prior_logits1_f)

           post_logits2_f = post_logits2.float()
           prior_logits2_f = prior_logits2.float()

           kl_loss2, kl_value2 = self.kl_loss(post_logits2_f, prior_logits2_f)
        loss_sim1 = kl_loss1/2 + kl_loss2/2

        # cosine loss in fp32 (no-op casts when not running under bf16 autocast)
        src_pred_norm = F.normalize(src_pred.float(), dim=-1, eps=1e-6)
        tgt_pred_norm = F.normalize(tgt_pred.float(), dim=-1, eps=1e-6)
        src_z_norm    = F.normalize(src_z[:, 1:, :].detach().float(), dim=-1, eps=1e-6)

        

        target_mask = mask_src * mask_tgt
        den = target_mask.sum().clamp_min(1.0)

        
        loss_sim2_1 = (2 - 2*(src_pred_norm * src_z_norm).sum(dim=-1).clamp(-1.0, 1.0))  # [N,L]
        loss_sim2_2 = (2 - 2*(tgt_pred_norm * src_z_norm).sum(dim=-1).clamp(-1.0, 1.0))
        
        loss_sim2 = ((loss_sim2_1 * target_mask).sum() / den + (loss_sim2_2 * target_mask).sum() / den)/2
        
        loss = loss_sim2 + self.kl_scale*loss_sim1

        return loss,src_pred,loss_sim1,loss_sim2


def siamjepa_vit_base_patch16_dec512d8b(**kwargs):
    model = SiamJEPA(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=768, decoder_depth=1, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def siamjepa_vit_large_patch16_dec1024d1b(**kwargs):
    # true ViT-L/16 encoder (24 blocks); predictor mirrors ViT-B's
    # (one CSABlock at the encoder width)
    model = SiamJEPA(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=1024, decoder_depth=1, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# ViT-Huge support is still under construction.


#def siamjepa_vit_huge_patch14_dec512d8b(**kwargs):
#    model = PhiNets(
#        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
#        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
#        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#    return model


# set recommended archs
siamjepa_vit_base_patch16 = siamjepa_vit_base_patch16_dec512d8b  # decoder: 768 dim, 1 block (name kept for checkpoints/scripts)
siamjepa_vit_large_patch16 = siamjepa_vit_large_patch16_dec1024d1b  # decoder: 1024 dim, 1 block
#siamjepa_vit_huge_patch14 = siamjepa_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
