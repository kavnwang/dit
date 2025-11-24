import torch
import torch.nn as nn
import math
from typing import Tuple

class Attention(nn.Module):
    def __init__(
        self,
        heads: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.heads = heads
        self.hidden_dim = hidden_dim
        self.pre_norm = nn.LayerNorm(self.hidden_dim, elementwise_affine=False)
        self.q = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.k = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.o = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, x: torch.Tensor, alpha_1: torch.Tensor, rho_1: torch.Tensor, beta_1: torch.Tensor) -> torch.Tensor:
        hidden_states = x
        hidden_states = self.pre_norm(hidden_states) * rho_1.unsqueeze(1)+ beta_1.unsqueeze(1)
        # q_proj = self.q(hidden_states).reshape((*x.shape[:2], self.heads, self.hidden_dim // self.heads)).transpose(1,2) #b h s d
        q_proj = self.q(hidden_states).reshape((x.shape[0], -1, self.heads, self.hidden_dim // self.heads)).transpose(1,2) #b h s d
        k_proj = self.k(hidden_states).reshape((x.shape[0], -1, self.heads, self.hidden_dim // self.heads)).transpose(1,2) #b h t d
        v_proj = self.v(hidden_states).reshape((x.shape[0], -1, self.heads, self.hidden_dim // self.heads)).transpose(1,2)
        attention_scores = (q_proj @ k_proj.transpose(-1,-2)) / (self.hidden_dim // self.heads) ** 0.5
        attention_scores = torch.softmax(attention_scores, -1)
        values = attention_scores @ v_proj
        values = values.transpose(1,2).reshape(x.shape)
        hidden_states = self.o(values)
        hidden_states = hidden_states * alpha_1.unsqueeze(1)
        return hidden_states + x
    
class MLP(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        intermediate_ratio: float,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = int(hidden_dim * intermediate_ratio)
        self.pre_norm = nn.LayerNorm(self.hidden_dim, elementwise_affine=False)
        self.up_proj = nn.Linear(self.hidden_dim, self.intermediate_dim)
        self.silu = nn.SiLU()
        self.down_proj = nn.Linear(self.intermediate_dim, self.hidden_dim)
    
    def forward(self, x: torch.Tensor, alpha_2: torch.Tensor, rho_2: torch.Tensor, beta_2: torch.Tensor) -> torch.Tensor:
        hidden_states = self.pre_norm(x) * rho_2.unsqueeze(1) + beta_2.unsqueeze(1)
        hidden_states = self.up_proj(hidden_states)
        hidden_states = self.silu(hidden_states)
        hidden_states = self.down_proj(hidden_states)
        hidden_states = hidden_states * alpha_2.unsqueeze(1)
        return hidden_states + x

class Conditioner(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        condition_p: float,
        intermediate_ratio: float,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.condition_p = condition_p
        self.class_embedding = nn.Embedding(num_classes + 1, embedding_dim)
        self.time_intermediate_dim = int(intermediate_ratio * embedding_dim)
        self.time_up_proj = nn.Linear(self.embedding_dim, self.time_intermediate_dim)
        self.silu = nn.SiLU()
        self.time_down_proj = nn.Linear(self.time_intermediate_dim, self.embedding_dim)
        self.time_mlp = nn.Sequential(self.time_up_proj, self.silu, self.time_down_proj)

    def sinusodial_embedding(self, timestep: torch.Tensor) -> torch.Tensor:
        half_dim = self.embedding_dim // 2
        freq = torch.exp(-math.log(10000) * torch.arange(half_dim, device=timestep.device) / half_dim)
        args = timestep[:, None] * freq[None, :]
        embedding = torch.cat((torch.sin(args), torch.cos(args)), dim=-1)
        return embedding

    def forward(self, label: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        if self.training:
            label = torch.where(torch.rand(label.shape, device=label.device) < self.condition_p, torch.tensor(self.num_classes, dtype=label.dtype, device=label.device), label)
        class_embedding = self.class_embedding(label)
        time_embedding = self.sinusodial_embedding(timestep)
        return class_embedding + self.time_mlp(time_embedding)
    

class Block(nn.Module):
    def __init__(
        self,
        hidden_dim: int, 
        heads: int,
        intermediate_ratio: float,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.attention = Attention(heads, hidden_dim)
        self.mlp = MLP(hidden_dim, intermediate_ratio)
        

    def forward(self, x: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        alpha_1 = scales[:,:self.hidden_dim]
        rho_1 = scales[:,self.hidden_dim:2*self.hidden_dim]
        beta_1 = scales[:,2*self.hidden_dim:3*self.hidden_dim]
        alpha_2 = scales[:,3*self.hidden_dim:4*self.hidden_dim]
        rho_2 = scales[:,4*self.hidden_dim:5*self.hidden_dim]
        beta_2 = scales[:,5*self.hidden_dim:6*self.hidden_dim]
        residual = x
        residual = self.attention(residual, alpha_1, rho_1, beta_1)
        residual = self.mlp(residual, alpha_2, rho_2, beta_2)
        return residual

class AdaLN(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        intermediate_ratio: float,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.intermediate_dim = int(self.embedding_dim * intermediate_ratio)
        self.hidden_dim = hidden_dim
        self.up_proj = nn.Linear(self.embedding_dim, self.intermediate_dim)
        self.silu = nn.SiLU()
        self.down_proj = nn.Linear(self.intermediate_dim, 6 * self.hidden_dim)
        self.layer = nn.Sequential(self.up_proj, self.silu, self.down_proj)
        nn.init.zeros_(self.down_proj.weight)
        bias_init = torch.zeros(6 * self.hidden_dim)
        bias_init[self.hidden_dim:2*self.hidden_dim] = 1.0  # rho_1
        bias_init[4*self.hidden_dim:5*self.hidden_dim] = 1.0  # rho_2
        self.down_proj.bias.data.copy_(bias_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)

class DiT(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        hidden_dim: int, 
        heads: int,
        img_size: int,
        patches: int,
        T: int,
        eps: float,
        num_classes: int,
        embedding_dim: int,
        condition_p: float,
        intermediate_ratio: float,
        adaln_intermediate_ratio: float,
    ):
        super().__init__()
        self.T = T
        self.eps = eps
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.T = T
        self.eps = eps
        self.img_size = img_size
        self.patches = patches
        
        self.conv = nn.Conv2d(3, 4, 3, 2, 1)
        self.patch_conv = nn.Conv2d(3, self.hidden_dim, patches, patches)
        self.conditioner = Conditioner(num_classes, embedding_dim, condition_p, intermediate_ratio)
        self.blocks = nn.ModuleList()
        self.condition_blocks = nn.ModuleList()
        for _ in range(self.num_blocks):
            self.blocks.append(Block(hidden_dim, heads, intermediate_ratio))
            self.condition_blocks.append(AdaLN(embedding_dim, self.hidden_dim, adaln_intermediate_ratio))
        self.num_patches = self.img_size // patches
        patch_len = self.num_patches * self.num_patches
        self.pos_embeddings = nn.Embedding(patch_len, self.hidden_dim)
        self.post_norm = nn.LayerNorm(self.hidden_dim)
        self.post_linear = nn.Linear(self.hidden_dim, patches * patches * 3)
        self.upsample = nn.ConvTranspose2d(6, 3, 4, 2, 1)

    
    def forward(self, z: torch.Tensor, label: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        emb = self.conditioner(label, t)
        hidden_states = self.patch_conv(z).reshape(z.shape[0],-1,self.hidden_dim)
        pos_embedding = self.pos_embeddings(torch.arange(self.num_patches * self.num_patches, device=z.device)[None,:].expand(hidden_states.shape[0],-1))
        hidden_states = hidden_states + pos_embedding
        for layer, embedding in zip(self.blocks, self.condition_blocks):
            scales = embedding(emb)
            hidden_states = layer(hidden_states, scales)
        hidden_states = self.post_norm(hidden_states) 
        hidden_states = self.post_linear(hidden_states) #B, 64, 1024 -> B, 64, 96
        hidden_states = torch.reshape(hidden_states, (z.shape[0], self.num_patches, self.num_patches, self.patches, self.patches, 3))
        hidden_states = hidden_states.permute(0, 1, 3, 2, 4, 5)
        hidden_states  = hidden_states.reshape((-1, self.img_size, self.img_size, 3)).transpose(1,-1)
        #noise = self.upsample(hidden_states)
        return hidden_states