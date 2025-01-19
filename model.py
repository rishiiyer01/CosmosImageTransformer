import torch
import torch.nn as nn
import torch.nn.functional as F
from math import inf
import math
import sys
sys.path.append('cosmos') #git cloned repo rename
from cosmos_tokenizer.image_lib import ImageTokenizer

class FSQConverter:
    def __init__(self):
        self.levels = torch.tensor([8, 8, 8, 5, 5, 5]).to('cuda')
        self.dim = len(self.levels)
        self.basis = torch.cumprod(torch.tensor([1] + [self.levels[i].item() for i in range(len(self.levels)-1)]), dim=0).to('cuda')

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """Converts indices to codes using NVIDIA's method, matches original paper implementation"""
        indices = indices.unsqueeze(-1).float()
        codes_non_centered = (indices // self.basis) % self.levels
        
        # Scale and shift inverse (normalize to [-1, 1])
        half_width = self.levels // 2
        codes = (codes_non_centered - half_width) / half_width
        
        return codes

    def indices_to_latents(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert indices to latent values"""
        return self.indices_to_codes(indices)

    def latents_to_indices(self, latents: torch.Tensor) -> torch.Tensor:
        """Convert latent values back to indices using NVIDIA's method
        Args:
            latents: Tensor of shape (..., dim) containing values in [-1, 1]
        Returns:
            indices: Tensor containing integer indices
        """
        # First denormalize from [-1, 1] back to [0, levels-1]
        latents=latents.to(torch.float32)
        half_width = self.levels // 2
        latents=latents.permute(0,2,3,1)
        codes = (latents * half_width) + half_width
        
        
        # Calculate indices using the basis (matching their codes_to_indices method)
        indices = (codes * self.basis).sum(dim=-1).to(torch.int32)
        
        return indices

class cosmos_vae(nn.Module):
    def __init__(self):
        super().__init__()
        model_name = "Cosmos-Tokenizer-DI16x16"
        self.encoder = ImageTokenizer(checkpoint_enc=f'pretrained_ckpts/{model_name}/encoder.jit')
        self.decoder=ImageTokenizer(checkpoint_dec=f'pretrained_ckpts/{model_name}/decoder.jit')








    
#RoPE for transformer
class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096):
        """
        Initialize Rotary Position Embedding
        
        Args:
            dim: Dimension of the embedding (must be divisible by 2)
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        
        if dim % 2 != 0:
            raise ValueError("Dimension must be divisible by 2")
            
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Create position indices tensor
        position = torch.arange(max_seq_len).unsqueeze(1)  # [max_seq_len, 1]
        
        # Create dimension indices tensor for half the dimension
        # Since we'll rotate half the dimensions, we only need dim/2
        div_term = torch.exp(
            torch.arange(0, dim//2) * -(math.log(10000.0) / (dim//2))
        )
        
        # Compute sin and cos tables for half dimensions
        emb = position * div_term
        self.register_buffer("sin_table", emb.sin().unsqueeze(0))  # [1, max_seq_len, dim//2]
        self.register_buffer("cos_table", emb.cos().unsqueeze(0))  # [1, max_seq_len, dim//2]
        
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input."""
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary position embedding to input tensor
        
        Args:
            x: Input tensor of shape [batch_size, num_heads, seq_len, head_dim]
            
        Returns:
            Tensor with positional information encoded
        """
        batch_size, num_heads, seq_len, dim = x.shape
        
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum length {self.max_seq_len}")
            
        # Get sin and cos values for current sequence length
        sin = self.sin_table[:, :seq_len, :]  # [1, seq_len, dim//2]
        cos = self.cos_table[:, :seq_len, :]  # [1, seq_len, dim//2]
        
        # Duplicate the sin/cos for the full dimension
        sin = torch.cat([sin, sin], dim=-1)  # [1, seq_len, dim]
        cos = torch.cat([cos, cos], dim=-1)  # [1, seq_len, dim]
        
        # Reshape sin and cos for broadcasting
        sin = sin.unsqueeze(1)  # [1, 1, seq_len, dim]
        cos = cos.unsqueeze(1)  # [1, 1, seq_len, dim]
        
        # Expand to match input shape
        sin = sin.expand(batch_size, num_heads, -1, -1)
        cos = cos.expand(batch_size, num_heads, -1, -1)
        
        # Apply rotation using complex number multiplication:
        # (a + ib)(cos θ + i sin θ) = (a cos θ - b sin θ) + i(a sin θ + b cos θ)
        return (x * cos) + (self._rotate_half(x) * sin)
    



#very small transformer for testing cosmos tokenizer
class LatentModel(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_blocks,vocab=64000):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        
        # Initial processing
        self.initial_proj = nn.Linear(in_channels, hidden_dim)  
        
        # Main processing blocks
        self.blocks = nn.ModuleList([
            Block(hidden_dim) for _ in range(num_blocks)
        ])
        vae=cosmos_vae()
        #self.encoder=vae.encoder.to('cuda') #deprecated, previously handled cosmos tokenizer within the model class
        
        self.out_proj = nn.Linear(hidden_dim, vocab)  
        
        self.start_proj=nn.Linear(1,hidden_dim,bias=True)
        
     
    def forward(self, x,condition):
        #encode image into patches
        #x=x.to(torch.bfloat16)
        #with torch.no_grad():
            
            #indices,x=self.encoder.encode(x) #b,c,x,y  
            
        #x=x.to(torch.float)
        b,c,h,w=x.shape
           
        x=x.permute(0,2,3,1) #b,x,y,c
        x=x.reshape(b,h*w,c) #should be b,256,6 for imagenet + cosmos
       
        
        
        
        
        x = self.initial_proj(x)
        
        #the condition is the start token
        condition=condition.unsqueeze(-1).unsqueeze(-1)
        condition=self.start_proj(condition) #b,1,hidden_dim
        
        x=torch.cat((condition,x[:,:-1,:]),dim=1)
        
        for block in self.blocks:
            x = block(x)

        logits=self.out_proj(x)
        #x=x.reshape(b,h*w,-1)
        
        
        
        return logits #b,h*w,vocab




class Block(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention =Attention(hidden_dim,num_heads=12)
        self.ff = FeedForward(hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x



class FeedForward(nn.Module):
    def __init__(self, hidden_dim, expansion_factor=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * expansion_factor),
            nn.GELU(),
            nn.Linear(hidden_dim * expansion_factor, hidden_dim)
        )
        
    def forward(self, x):
        return self.net(x)




class Attention(nn.Module):
    def __init__(self, hidden_dim, num_heads=12):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.rope=RotaryPositionEmbedding(self.head_dim)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x):
        B, N,dim = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        #print(q.shape)
        q=q+self.rope(q)
        k=k+self.rope(k)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        causal_mask = torch.triu(torch.ones(N, N), diagonal=1).bool()
        attn = attn.masked_fill(causal_mask.to(attn.device), float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        
        x = self.proj(x)
        return x



