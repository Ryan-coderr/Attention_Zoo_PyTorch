import torch
import math
from einops import rearrange

def Attention_Vanilla(q, k, v):
    score = torch.softmax(torch.einsum("bhic,bhjc->bhij", q, k)/math.sqrt(k.shape[-1]), dim=-1)
    r = torch.einsum("bhij,bhjc->bhic", score, v)
    return r

class LinearAttention_Galerkin_and_Fourier(torch.nn.Module):
    def __init__(self, attn_type, n_dim, n_head, use_ln = False):
        super().__init__()
        self.attn_type = attn_type
        self.n_dim = n_dim
        self.n_head = n_head
        self.dim_head = self.n_dim // self.n_head
        self.use_ln = use_ln
        self.to_qkv = torch.nn.Linear(n_dim, n_dim*3, bias = False)
        self.project_out = (not self.n_head == 1)

        if attn_type == 'galerkin':
            if not self.use_ln:
                self.k_norm = torch.nn.InstanceNorm1d(self.dim_head)
                self.v_norm = torch.nn.InstanceNorm1d(self.dim_head)
            else:
                self.k_norm = torch.nn.LayerNorm(self.dim_head)
                self.v_norm = torch.nn.LayerNorm(self.dim_head)

        elif attn_type == 'fourier':
            if not self.use_ln:
                self.q_norm = torch.nn.InstanceNorm1d(self.dim_head)
                self.k_norm = torch.nn.InstanceNorm1d(self.dim_head)
            else:
                self.q_norm = torch.nn.LayerNorm(self.dim_head)
                self.k_norm = torch.nn.LayerNorm(self.dim_head)

        else:
            raise Exception(f'Unknown attention type {attn_type}')

        self.to_out = torch.nn.Sequential(
            torch.nn.Linear(self.n_dim, self.n_dim),
            torch.nn.Dropout(0.0)
        ) if self.project_out else torch.nn.Identity()

    def norm_wrt_domain(self, x, norm_fn):
        b = x.shape[0]
        return rearrange(
            norm_fn(rearrange(x, 'b h n d -> (b h) n d')),
            '(b h) n d -> b h n d', b=b)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.n_head), qkv)

        if self.attn_type == 'galerkin':
            k = self.norm_wrt_domain(k, self.k_norm)
            v = self.norm_wrt_domain(v, self.v_norm)
        elif self.attn_type == 'fourier':
            q = self.norm_wrt_domain(q, self.q_norm)
            k = self.norm_wrt_domain(k, self.k_norm)
        else:
            raise NotImplementedError("Invalid Attention Type!")

        dots = torch.matmul(k.transpose(-1, -2), v)
        out = torch.matmul(q, dots) * (1./q.shape[2])
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


