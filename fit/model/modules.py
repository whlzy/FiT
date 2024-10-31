import torch
import math
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from torch.jit import Final
from timm.layers.mlp import SwiGLU, Mlp  
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from fit.model.rope import rotate_half
from fit.model.utils import modulate
from fit.model.norms import create_norm
from functools import partial
from einops import rearrange, repeat

#################################################################################
#           Embedding Layers for Patches, Timesteps and Class Labels            #
#################################################################################

class PatchEmbedder(nn.Module):
    """
    Embeds latent features into vector representations
    """
    def __init__(self, 
        input_dim, 
        embed_dim, 
        bias: bool = True,
        norm_layer: Optional[Callable] = None,
    ):
        super().__init__()
        
        self.proj = nn.Linear(input_dim, embed_dim, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)    # (B, L, patch_size ** 2 * C) -> (B, L, D)
        x = self.norm(x)
        return x  

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
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1]).to(device=t.device)], dim=-1)
        return embedding.to(dtype=t.dtype)

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings




#################################################################################
#                                  Attention                                    #
#################################################################################

# modified from timm and eva-02
# https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
# https://github.com/baaivision/EVA/blob/master/EVA-02/asuka/modeling_finetune.py


class Attention(nn.Module):

    def __init__(self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        q_norm: Optional[str] = None,
        k_norm: Optional[str] = None,
        qk_norm_weight: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        rel_pos_embed: Optional[str] = None,
        add_rel_pe_to_v: bool = False, 
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if q_norm == 'layernorm' and qk_norm_weight == True:
            q_norm = 'w_layernorm'
        if k_norm == 'layernorm' and qk_norm_weight == True:
            k_norm = 'w_layernorm'
        
        self.q_norm = create_norm(q_norm, self.head_dim)
        self.k_norm = create_norm(k_norm, self.head_dim)
        

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.rel_pos_embed = None if rel_pos_embed==None else rel_pos_embed.lower() 
        self.add_rel_pe_to_v = add_rel_pe_to_v
        
        

    def forward(self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None, 
        freqs_cos: Optional[torch.Tensor] = None, 
        freqs_sin: Optional[torch.Tensor] = None, 
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0) # (B, n_h, N, D_h)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rel_pos_embed in ['rope', 'xpos']:  # multiplicative rel_pos_embed
            if self.add_rel_pe_to_v:
                v = v * freqs_cos + rotate_half(v) * freqs_sin
            q = q * freqs_cos + rotate_half(q) * freqs_sin
            k = k * freqs_cos + rotate_half(k) * freqs_sin
        
        attn_mask = mask[:, None, None, :]  # (B, N) -> (B, 1, 1, N)
        attn_mask = (attn_mask == attn_mask.transpose(-2, -1))  # (B, 1, 1, N) x (B, 1, N, 1) -> (B, 1, N, N)
        mask = torch.not_equal(mask, torch.zeros_like(mask)).to(mask)   # (B, N) -> (B, N)
        
        
        if x.device.type == "cpu":
            x = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            with torch.backends.cuda.sdp_kernel(enable_flash=True):
                '''
                F.scaled_dot_product_attention is the efficient implementation equivalent to the following:
                    attn_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0) if is_causal else attn_mask
                    attn_mask = attn_mask.masked_fill(not attn_mask, -float('inf')) if attn_mask.dtype==torch.bool else attn_mask
                    attn_weight = torch.softmax((Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))) + attn_mask, dim=-1)
                    attn_weight = torch.dropout(attn_weight, dropout_p)
                    return attn_weight @ V
                In conclusion:
                    boolean attn_mask will mask the attention matrix where attn_mask is False
                    non-boolean attn_mask will be directly added to Q@K.T
                '''
                x = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask,
                    dropout_p=self.attn_drop.p if self.training else 0.,
                )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = x * mask[..., None] # mask: (B, N) -> (B, N, 1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

#################################################################################
#                               Basic FiT Module                                #
#################################################################################

class FiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, 
        hidden_size, 
        num_heads, 
        mlp_ratio=4.0, 
        swiglu=True,
        swiglu_large=False,
        rel_pos_embed=None,
        add_rel_pe_to_v=False,
        norm_layer: str = 'layernorm',
        q_norm: Optional[str] = None,
        k_norm: Optional[str] = None,
        qk_norm_weight: bool = False,
        qkv_bias=True,
        ffn_bias=True,
        adaln_bias=True,
        adaln_type='normal',
        adaln_lora_dim: int = None,
        **block_kwargs
    ):
        super().__init__()
        self.norm1 = create_norm(norm_layer, hidden_size)
        self.norm2 = create_norm(norm_layer, hidden_size)
        
        self.attn = Attention(
            hidden_size, num_heads=num_heads, rel_pos_embed=rel_pos_embed, 
            q_norm=q_norm, k_norm=k_norm, qk_norm_weight=qk_norm_weight,
            qkv_bias=qkv_bias, add_rel_pe_to_v=add_rel_pe_to_v, 
            **block_kwargs
        )
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if swiglu:
            if swiglu_large:
                self.mlp = SwiGLU(in_features=hidden_size, hidden_features=mlp_hidden_dim, bias=ffn_bias)
            else:
                self.mlp = SwiGLU(in_features=hidden_size, hidden_features=(mlp_hidden_dim*2)//3, bias=ffn_bias)
        else:
            self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=lambda: nn.GELU(approximate="tanh"), bias=ffn_bias)
        if adaln_type == 'normal':
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=adaln_bias)
            )
        elif adaln_type == 'lora':
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, adaln_lora_dim, bias=adaln_bias),
                nn.Linear(adaln_lora_dim, 6 * hidden_size, bias=adaln_bias)
            )
        elif adaln_type == 'swiglu':
            self.adaLN_modulation = SwiGLU(
                in_features=hidden_size, hidden_features=(hidden_size//4)*3, out_features=6*hidden_size, bias=adaln_bias
            )

    def forward(self, x, c, mask, freqs_cos, freqs_sin, global_adaln=0.0):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.adaLN_modulation(c) + global_adaln).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), mask, freqs_cos, freqs_sin)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels, norm_layer: str = 'layernorm', adaln_bias=True, adaln_type='normal'):
        super().__init__()
        self.norm_final = create_norm(norm_type=norm_layer, dim=hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        if adaln_type == 'swiglu':
            self.adaLN_modulation = SwiGLU(in_features=hidden_size, hidden_features=hidden_size//2, out_features=2*hidden_size, bias=adaln_bias)
        else:   # adaln_type in ['normal', 'lora']
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 2 * hidden_size, bias=adaln_bias)
            )
        
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
