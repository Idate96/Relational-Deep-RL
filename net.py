import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

from math import sqrt
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange



def conv2d_size_out(size, kernel_size, stride):
  return (size - (kernel_size - 1) - 1) // stride + 1


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class MLP(nn.Module):
  def __init__(self, input_dim, layer_dim, depth):
      super().__init__()
      self.layers = nn.ModuleList([nn.Linear(input_dim, layer_dim), nn.GELU()])
      for _ in range(depth - 1):
        self.layers.append(nn.Linear(layer_dim, layer_dim))
        self.layers.append(nn.GELU())

  def forward(self, x):
    for layer in self.layers:
      x = layer(x)
    return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class MultiHeadAttention(nn.Module):

    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        # in the original paper there are two layers (196, 64 * 2) -> (196, 26)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )


    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        
        # b = batch size, h = number of parallel heads, n = input dim (sequence), d = embedding dim (or query size)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        
        # i and j are the same as n, dots have dimension n^2 -> quadratic 
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
      

    def attention_weights(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        
        # b = batch size, h = number of parallel heads, n = input dim (sequence), d = embedding dim (or query size)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        
        # i and j are the same as n, dots have dimension n^2 -> quadratic 
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        return attn # b, h, n, n 


class Attention(nn.Module):
    def __init__(self, dim_in, dim_out, dim_inner, causal = False):
        super().__init__()
        self.scale = dim_inner ** -0.5
        self.causal = causal

        self.to_qkv = nn.Linear(dim_in, dim_inner * 3, bias = False)
        self.to_out = nn.Linear(dim_inner, dim_out)

    def forward(self, x):
        device = x.device
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if self.causal:
            mask = torch.ones(sim.shape[-2:], device = device).triu(1).bool()
            sim.masked_fill_(mask[None, ...], -torch.finfo(q.dtype).max)

        attn = sim.softmax(dim = -1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, MultiHeadAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class PositionalEncodings(nn.Module):
  pass


class RelationalNet(nn.Module):
  """Relational net used in the paper Relational Deep RL

  The class provides two different implementatios: baseline and attention

  The Attention net has the following structure:
    - 2 x conv (12 and 24 kernels of size 2 and stride 1) + position encoding 
    - 2 to 4 attention blocks with embedding size 64
    - feature wide max pooling (n x n x k -> k)
    - MPL with 4 layers 
    - projection to policy (4 logits) and value function (1 scalar)

  The Resnet has the following architecture:
    - 2 x conv (12 and 24 kernels of size 2 and stride 1) + position encoding 
    - 2 to 4 attention blocks with embedding size 64
    - 3 to 6 residual blocks, each block with a conv layer with 26 channels and 3x3 kernels
  """
  def __init__(self, mlp_depth=4, depth_transformer=2, heads=2, baseline=False, recurrent_transformer=False):
      super().__init__()
      # convolulotional layers 
      # padding is used such that the output of the second layer is still 14x14
      self.conv = nn.Sequential(
        nn.Conv2d(3, 12, kernel_size=(2, 2), stride=1, padding=1),
        nn.GELU(),
        nn.Conv2d(12, 24, kernel_size=(2, 2), stride=1),
        nn.GELU()
      )
    
      # positional encodings of size (2, 14, 14) that contain the x and y pos of each tile 
      # linearly spaces between -1 and 1
      x = torch.linspace(-1, 1, steps=14)
      y = torch.linspace(-1, 1, steps=14)
      xx, yy = torch.meshgrid(x, y)
      xx = xx.flatten()
      yy = yy.flatten()
      # the stacking dimension 1 means that expends in that dimesion (196, 2)
      # the unqueeze just adds the batch dim, necessary when concat later on
      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      self.embeddings = torch.unsqueeze(torch.stack((xx, yy), dim=1), dim = 0).to(device)

      # attention layer
      self.recurrent_transformer = recurrent_transformer
      self.depth_transformer = depth_transformer
      if recurrent_transformer: 
        self.transformer = Transformer(dim=26, depth=depth_transformer, heads=heads, dim_head=64, mlp_dim=256)
      else:
        self.transformer = Transformer(dim=26, depth=1, heads=heads, dim_head=64, mlp_dim=256)
      
      self.transformer_project = nn.Linear(14 * 14 * heads, 14 * 14)
      self.mlp = MLP(input_dim=26, layer_dim=256, depth=mlp_depth)
      # logits for policy and value 
      self.baseline = baseline
      # device
      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  def forward(self, x):
    # initial feature extractors
    x = self.conv(x)

    # add positional encodings

    if not self.baseline:
      # project to from (n, c, i, j) -> (n, i * j, c)
      x = rearrange(x, 'b c i j -> b (i j) c')  
      # repeat along the batch dim
      batch_size = x.size()[0]
      embeddings = self.embeddings.repeat(batch_size, 1, 1)
      x = torch.cat([x, embeddings], dim=2) # stack along the dimension of the filters -> (n, 196, 26)

      if self.recurrent_transformer:
        for _ in range(self.depth_transformer):
          # depth 1 transormer applied multiple time for weight sharing 
          x = self.transformer(x)
      else:
        x = self.transformer(x)
    else:
      pass 
    # out of the 14^2 features per filter (26) pick one
    # feels kind of an information bottlececk though 
    x = torch.squeeze(F.max_pool1d(rearrange(x, 'n i j -> n j i'), kernel_size=x.shape[1]), dim=2)

    # x = self.transformer_project(x)
    x = self.mlp(x)
    return x



if __name__ == '__main__':
  input = np.zeros((2, 3, 14, 14))
  net = RelationalNet()
  output = net(torch.from_numpy(input).float())  
  print(output.shape)
  # pos_enc_ = pos_enc(output)
  # print(pos_enc_[0, 0, 0])
  # print(pos_enc_[1, 0, 0])
  # attend = nn.Softmax(dim = -1)

  # # the linear embeddings 
  # linear_embedding = torch.zeros((2, 3, 6)) # b n (h * d)
  # h = 2
  # qvk = linear_embedding.chunk(3, dim=-1)
  # # b = batch size, h = number of parallel heads, n = input dim (sequence), d = embedding dim (or query size)
  # q, v, k = map(lambda t: rearrange(t, 'b n (h d) ->b h n d', h=h), qvk) # split heads explicitly 

  # # i and j are the same as n, dots have dimension n^2 -> quadratic 
  # dots = einsum('b h i d, b h j d -> b h i j', q, k)

  # out = einsum('b h i j, b h j d -> b h i d', attend(dots), v) # input dim gets contracted 
  # out = rearrange(out, 'b h n d -> b n (h d)')
  # print(out.shape) 