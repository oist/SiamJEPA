# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026 Makoto Yamada and contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file is based on the DeiT implementation and has been
# substantially modified for the SiamJEPA project.
#
# References:
# timm: https://github.com/rwightman/pytorch-image-models
# DeiT: https://github.com/facebookresearch/deit


from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block
from timm.models.vision_transformer import CrossAttention, Attention, DropPath, Mlp

from util.pos_embed import get_2d_sincos_pos_embed

import copy
import torch.distributions as td
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
            nn.BatchNorm1d(hidden_dim)
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

    def forward(self, x, kvx, src_mask=None):
        
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

def patch_whitening_loss(x_patch, eps=1e-4):
    # x_patch: [B, N, D]
    z = x_patch.reshape(-1, x_patch.shape[-1])  # [B*N, D]

    # normalize each dimension
    z = z - z.mean(dim=0, keepdim=True)
    z = z / (z.std(dim=0, keepdim=True) + eps)

    # covariance/correlation matrix
    C = (z.T @ z) / (z.shape[0] - 1)  # [D, D]
    D = C.shape[0]
    
    # off-diagonal penalty
    off_diag = C - torch.diag(torch.diag(C))
    #I = torch.eye(C.shape[0], device=C.device, dtype=C.dtype)
    loss = (off_diag ** 2).sum() / D 

    return loss

class KoLeoLoss(nn.Module):
    """Kozachenko-Leonenko entropic loss regularizer from Sablayrolles et al. - 2018 - Spreading vectors for similarity search"""

    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def pairwise_NNs_inner(self, x):
        """
        Pairwise nearest neighbors for L2-normalized vectors.
        Uses Torch rather than Faiss to remain on GPU.
        """
        # parwise dot products (= inverse distance)
        dots = torch.mm(x, x.t())
        n = x.shape[0]
        dots.view(-1)[:: (n + 1)].fill_(-1)  # Trick to fill diagonal with -1
        # max inner prod -> min distance
        _, I = torch.max(dots, dim=1)  # noqa: E741
        return I

    def forward(self, student_output, eps=1e-8):
        """
        Args:
            student_output (BxD): backbone output of student
        """
        with torch.cuda.amp.autocast(enabled=False):
            student_output = F.normalize(student_output, eps=eps, p=2, dim=-1)
            I = self.pairwise_NNs_inner(student_output)  # noqa: E741
            distances = self.pdist(student_output, student_output[I])  # BxD, BxD -> B
            loss = -torch.log(distances + eps).mean()
        return loss

class SiamJEPA(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,stoch=32,
        discrete=32,kl_scale=0.01,
        kl_balance=0.2,kl_freebit=0.1,beta=0.996,mask_ratio=0.9):
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
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

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
        self.decoder_pred_latent = nn.Linear(decoder_embed_dim, embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch

        stoch_size = stoch * discrete if discrete != 0 else stoch * 2
        self.decoder_embed_mae = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_embed_deter = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_embed_stoch = nn.Linear(stoch_size, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

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

        self.projector = projection_MLP(embed_dim)
        self.ca3 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
        )

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

        self.beta=beta
        self.ema_model = copy.deepcopy(self)
        self.ema_model.eval()   # ← train() ではなく eval()
        for p in self.ema_model.parameters():
            p.requires_grad = False

        self.stoch = stoch
        self.discrete = discrete
        self.kl_balance = kl_balance
        self.kl_scale = kl_scale
        self.kl_freebit=kl_freebit

        self.mask_ratio=mask_ratio


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

        all_ids = torch.arange(L, device=device)
        ids_shuffle_list = []

        for n in range(N):
            top = torch.randint(0, H - block_size + 1, (1,), device=device)
            left = torch.randint(0, W - block_size + 1, (1,), device=device)

            yy, xx = torch.meshgrid(
                torch.arange(block_size, device=device),
                torch.arange(block_size, device=device),
                indexing="ij"
            )

            block_ids = ((top + yy) * W + (left + xx)).reshape(-1)

            is_block = torch.zeros(L, dtype=torch.bool, device=device)
            is_block[block_ids] = True

            outside_ids = all_ids[~is_block]
            outside_ids = outside_ids[torch.randperm(outside_ids.numel(), device=device)]

            # keep に使うパッチ
            keep_ids = outside_ids[:2 * len_keep]

            # keep に使わない余りパッチ
            rest_ids = outside_ids[2 * len_keep:]

            block_ids = block_ids[torch.randperm(block_ids.numel(), device=device)]
            rest_ids = rest_ids[torch.randperm(rest_ids.numel(), device=device)]

            # 先頭 2*len_keep だけが view1/view2 に使われる
            # それ以降は両方から mask される
            ids_shuffle = torch.cat([keep_ids, rest_ids, block_ids], dim=0)
            ids_shuffle_list.append(ids_shuffle)

        ids_shuffle = torch.stack(ids_shuffle_list, dim=0)
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

        # apply Transformer blocks
        for blk in self.blocks:
            x1 = blk(x1)
            x2 = blk(x2)
        x1 = self.norm(x1)
        x2 = self.norm(x2)

        return x1, x2, mask1, mask2, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore


    def forward_predictor(self, x, ids_restore,z):
        # embed tokens
        #x = self.decoder_embed(x)
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

        for ema_param, param in zip(model.ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(model.beta).add_(param.data, alpha=1 - model.beta)
    
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
        post_logits1 = self.to_posterior(post_h1)
        post_logits1 = post_logits1.clamp(-20, 20)
            
        post_dist1 = self.make_dist(post_logits1)
        post_z1 = post_dist1.rsample()

        # Prior distribution only from current images
        prior_h1 = src_h_ca3_cls

        prior_logits1 = self.to_prior(prior_h1.detach())
        prior_logits1 = prior_logits1.clamp(-20, 20)

        src_p_ca3_cls = self.ca3(self.projector(src_p[:, 0]))
        src_p_ca3 = self.ca3(src_p)

        # Posterior distribution from both images
        post_p1 = torch.cat([src_p_ca3_cls, self.projector(src_h[:, 0])], -1)
        post_logits2 = self.to_posterior(post_p1)
        post_logits2 = post_logits2.clamp(-20, 20)
            
        post_dist2 = self.make_dist(post_logits2)
        post_z2 = post_dist2.rsample()

        # Prior distribution only from current images
        prior_p1 = src_p_ca3_cls

        prior_logits2 = self.to_prior(prior_p1.detach())
        prior_logits2 = prior_logits2.clamp(-20, 20)

        #Predictor g
        src_pred = self.forward_predictor(src_h_ca3, ids_restore, post_z1)
        tgt_pred = self.forward_predictor(src_p_ca3, ids_restore, post_z2)

        with torch.cuda.amp.autocast(enabled=True):
           post_logits1_f = post_logits1.float()
           prior_logits1_f = prior_logits1.float()
           kl_loss1, kl_value1 = self.kl_loss(post_logits1_f, prior_logits1_f)

           post_logits2_f = post_logits2.float()
           prior_logits2_f = prior_logits2.float()

           kl_loss2, kl_value2 = self.kl_loss(post_logits2_f, prior_logits2_f)
        loss_sim1 = kl_loss1/2 + kl_loss2/2

        src_pred_norm = F.normalize(src_pred, dim=-1, eps=1e-6)
        tgt_pred_norm = F.normalize(tgt_pred, dim=-1, eps=1e-6)
        src_z_norm    = F.normalize(src_z[:, 1:, :].detach(), dim=-1, eps=1e-6)

        

        target_mask = mask_src * mask_tgt
        den = target_mask.sum().clamp_min(1.0)

        
        loss_sim2_1 = (2 - 2*(src_pred_norm * src_z_norm).sum(dim=-1).clamp(-1.0, 1.0))  # [N,L]
        loss_sim2_2 = (2 - 2*(tgt_pred_norm * src_z_norm).sum(dim=-1).clamp(-1.0, 1.0))
        
        loss_sim2 = ((loss_sim2_1 * target_mask).sum() / den + (loss_sim2_2 * target_mask).sum() / den)/2
        
        loss = loss_sim2 + self.kl_scale*loss_sim1

        chk("loss_sim2",loss_sim2)
        chk("loss_sim1",loss_sim1)
        return loss,src_pred,loss_sim1,loss_sim2


def siamjepa_vit_base_patch16_dec512d8b(**kwargs):
    model = SiamJEPA(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=768, decoder_depth=1, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# ViT-Large and ViT-Huge support is still under construction.
#def siamjepa_vit_large_patch16_dec512d8b(**kwargs):
#    model = PhiNets(
#        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
#        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
#        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#    return model


#def siamjepa_vit_huge_patch14_dec512d8b(**kwargs):
#    model = PhiNets(
#        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
#        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
#        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#    return model


# set recommended archs
siamjepa_vit_base_patch16 = siamjepa_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
#siamjepa_vit_large_patch16 = siamjepa_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
#siamjepa_vit_huge_patch14 = siamjepa_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
