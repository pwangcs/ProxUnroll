import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import einops
from einops import rearrange 
from einops.layers.torch import Rearrange, Reduce
from timm.models.layers import trunc_normal_, DropPath
import scipy.io as scio
import torch.utils.checkpoint as cp



def RelativePosition(window_size):
    coords_d = torch.arange(window_size[0])
    coords_h = torch.arange(window_size[1])
    coords_w = torch.arange(window_size[2])
    coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))  # 3, Wd, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
    relative_coords1 = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
    relative_coords = relative_coords1.permute(1, 2, 0).contiguous()  # 
    relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
    relative_coords[:, :, 1] += window_size[1] - 1
    relative_coords[:, :, 2] += window_size[2] - 1

    relative_coords[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
    relative_coords[:, :, 1] *= (2 * window_size[2] - 1)
    relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
    return relative_position_index

class WMSA(nn.Module):
    """ 
    Window Multi-head Self-Attention (WMSA) module in Swin Transformer
    """
    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim 
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim//head_dim
        self.window_size = window_size
        self.type=type
        self.qkv = nn.Linear(self.input_dim, 3*self.input_dim)
        self.linear = nn.Linear(self.input_dim, self.output_dim)
        # relative position encoding
        self.relative_position_params = nn.Parameter(torch.zeros((2 * window_size - 1) * (2 * window_size -1), self.n_heads))
        trunc_normal_(self.relative_position_params, std=.02)
        relative_position_index = RelativePosition(window_size=[1, self.window_size, self.window_size])
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)

    def generate_mask(self, h, w, p, shift):
        """ generating the mask of SW-MSA
        Args:
            shift: shift parameters in CyclicShift.
        Returns:
            attn_mask: should be (1 1 w p p),
        """
        # supporting sqaure.
        attn_mask = torch.zeros(h, w, p, p, p, p, dtype=torch.bool)
        if self.type == 'W':
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
        return attn_mask

    def forward(self, x):
        """ Forward pass of Window Multi-head Self-attention module.
        Args:
            x: input tensor with shape of [b h w c];
            attn_mask: attention mask, fill -inf where the value is True; 
        Returns:
            output: tensor shape [b h w c]
        """
        h, w = x.shape[1:-1]
        assert h%self.window_size==0 and w%self.window_size==0, 'Input cannot be divided into spatial windows.'
        if self.type!='W': x = torch.roll(x, shifts=(-(self.window_size//2), -(self.window_size//2)), dims=(1,2))

        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, 'b (w1 p1) (w2 p2) (threeh c) -> threeh b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size, c=self.head_dim).chunk(3, dim=0)
        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        # Adding learnable relative opsition bias
        sim = sim + rearrange(self.relative_position_params[self.relative_position_index.reshape(-1)], '(p q) h -> h 1 1 p q', p=self.window_size**2, q=self.window_size**2)
        # Using Attn Mask to distinguish different subwindows.
        if self.type != 'W':
            attn_mask = self.generate_mask(int(h/self.window_size), int(w/self.window_size), self.window_size, shift=self.window_size//2).to(sim.device)
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        atten = F.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', atten, v)
        output = rearrange(output, 'h b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) (h c)', w1=int(h/self.window_size), p1=self.window_size)
        output = self.linear(output)

        if self.type!='W': output = torch.roll(output, shifts=(self.window_size//2, self.window_size//2), dims=(1,2))
        return output

    
class SwinBlock(nn.Module):
    def __init__(self, dim, head_dim, window_size, ffn_ratio=2, type='W'):
        """ Our Shiwin Transformer Block
        """
        super().__init__()
        assert type in ['W', 'SW']

        self.msa = WMSA(dim, dim, head_dim, window_size, type)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, ffn_ratio *dim, 1, 1, 0, bias=True),
            nn.Conv2d(ffn_ratio * dim, ffn_ratio * dim, 3, 1, 1, groups=ffn_ratio * dim, bias=True),
            nn.GELU(),
            nn.Conv2d(ffn_ratio *dim, dim, 1, 1, 0, bias=True),
        )
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = Rearrange('b c h w -> b h w c')(x)
        x = x + self.msa(self.ln1(x))
        # x = x + cp.checkpoint(self.msa, self.ln1(x), preserve_rng_state=False, use_reentrant=False)
        x = x + self.ffn(self.ln2(x).permute(0,3,1,2).contiguous()).permute(0,2,3,1).contiguous()
        x = Rearrange('b h w c -> b c h w')(x)
        return x


class AdaConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, bases=16, bias=False):
        super(AdaConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.groups = groups
        self.bases = bases
        self.weight = nn.Parameter(torch.empty((1, groups, (out_channels//groups) * (in_channels//groups) * kernel_size**2, bases)))  
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.empty((1, groups, out_channels//groups, bases)))
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)
        self.predictor = nn.Sequential(
                       nn.Conv2d(in_channels, bases, 1, 1, 0),
                       nn.ReLU(inplace=True),
                       nn.Conv2d(bases, bases, 3, 1, 1))

    # def forward(self, x):
    #     bs, c, h, w = x.shape
    #     assert c == self.in_channels, 'Except dim is {}, but input dim is {}'.format(self.in_channels,c)
    #     para = self.predictor(x).reshape(bs, 1, self.bases, h*w)   # bs 1 b hw
    #     weight = torch.matmul(self.weight,para).reshape(bs, self.groups, self.out_channels//self.groups, (c//self.groups)*self.kernel_size**2, h*w)
    #     bias =  torch.matmul(self.bias,para) if self.bias is not None else None  # bs g c2 hw
    #     x = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding).reshape(bs, self.groups, 1, (c//self.groups)*self.kernel_size**2, h*w)
    #     x = torch.sum(weight * x, dim=-2) + bias if bias is not None else torch.sum(weight * x, dim=-2)
    #     return x.reshape(bs, c, h, w)

    def dynamic_conv(self,inp):
        x, para = inp
        bs, c, h, w = x.shape
        weight = torch.matmul(self.weight,para).reshape(bs, self.groups, self.out_channels//self.groups, (c//self.groups)*self.kernel_size**2, h*w)
        bias = torch.matmul(self.bias,para) if self.bias is not None else None  # bs g c2 hw
        x = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding).reshape(bs, self.groups, 1, (c//self.groups)*self.kernel_size**2, h*w)
        x = torch.sum(weight * x, dim=-2) + bias if bias is not None else torch.sum(weight * x, dim=-2)
        return x

    def forward(self, x):
        bs, c, h, w = x.shape
        assert c == self.in_channels, f'Except dim is {self.in_channels}, but input dim is {c}.'
        para = self.predictor(x).reshape(bs, 1, self.bases, h*w)   # bs 1 b hw
        x = cp.checkpoint(self.dynamic_conv, (x, para), preserve_rng_state=False, use_reentrant=False)
        return x.reshape(bs, c, h, w)

class ConvBlock(nn.Module):
    def __init__(self, dim, conv_ratio=2, kernel_size=3, groups=8, bases=16):
        super().__init__()
        mid_dim = int(conv_ratio*dim)
        self.ln = nn.LayerNorm(dim)
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, mid_dim, 1, 1, 0, bias=False),
            nn.Conv2d(mid_dim, mid_dim, 3, 1, 1, groups=mid_dim, bias=False)
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim, mid_dim, 1, 1, 0, bias=False),
            AdaConv(mid_dim, mid_dim, kernel_size=kernel_size, groups=int(conv_ratio*groups), bases=bases, bias=False),
            nn.GELU()
            )
        self.conv3 = nn.Conv2d(mid_dim, dim, 1, 1, 0, bias=False)
    def forward(self, x):
        res = self.ln(x.permute(0,2,3,1).contiguous()).permute(0,3,1,2).contiguous()
        res = self.conv1(res) * self.conv2(res)
        res = self.conv3(res)
        x = x + res
        return x
    
class ChanelCrossAttention(nn.Module):
    def __init__(self, dim, head_dim, bias=False):
        super().__init__()
        self.num_head = dim//head_dim
        self.temperature = nn.Parameter(torch.ones(self.num_head, 1, 1))
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        # x -> q, y -> kv
        assert x.shape == y.shape, 'The shape of feature maps from image and features are not equal!'
        h, w = x.shape[-2::]
        q = self.q_dwconv(self.q(x))
        k, v = self.kv_dwconv(self.kv(y)).chunk(2, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_head)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_head)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_head)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = q @ k.transpose(-2, -1) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_head, h=h, w=w)
        out = self.project_out(out)
        return out

class Block(nn.Module):
    def __init__(self, conv_dim, trans_dim, head_dim, groups, window_size, type='W', ffn_ratio=2):
        """ SwinTransformer and Conv Block
        """
        super().__init__()
        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        assert type in ['W', 'SW']
        self.trans_block = SwinBlock(trans_dim, head_dim, window_size, ffn_ratio, type)
        self.conv_block = ConvBlock(conv_dim, groups=groups)
        self.conv = nn.Conv2d(conv_dim+trans_dim, conv_dim+trans_dim, 1, 1, 0, bias=True)

    def forward(self, x):
        conv_x, trans_x = torch.split(x, (self.conv_dim, self.trans_dim), dim=1)
        conv_x = self.conv_block(conv_x)
        trans_x = self.trans_block(trans_x)
        res = self.conv(torch.cat((conv_x, trans_x), dim=1))
        x = x + res
        return x


class SwinConvNet(nn.Module):
    def __init__(self, color_channel=1, dim=64, head_dim=32, groups=4, window_size=8, ffn_ratio=2, mid_blocks=2, enc_blocks=[2,2,2], dec_blocks=[2,2,2]):
        super(SwinConvNet, self).__init__()

        self.pad_size =  window_size * (2 ** len(enc_blocks))
        self.embedding = nn.Conv2d(color_channel, dim, 3, 1, 1)
        self.mapping = nn.Conv2d(dim, color_channel, 3, 1, 1)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.memorizers = nn.ModuleList()

        type = ['W', 'SW']
        for num in enc_blocks:
            self.memorizers.append(ChanelCrossAttention(dim, head_dim))
            self.encoders.append(nn.Sequential(*[Block(dim//2, dim//2, head_dim, groups, window_size, type[i%2], ffn_ratio) for i in range(num)]))
            self.downs.append(nn.Conv2d(dim, 2 * dim, 2, 2, bias=False))
            dim = dim * 2
            groups = groups * 2

        self.memorizers.append(ChanelCrossAttention(dim, head_dim))
        self.bottleneck = nn.Sequential(*[Block(dim//2, dim//2, head_dim, groups, window_size, type[i%2], ffn_ratio) for i in range(mid_blocks)])

        for num in dec_blocks:
            self.ups.append(nn.ConvTranspose2d(dim, dim//2, 2, 2, bias=False))
            dim = dim // 2
            groups = groups // 2
            self.decoders.append(nn.Sequential(*[Block(dim//2, dim//2, head_dim, groups, window_size, type[i%2], ffn_ratio) for i in range(num)]))

    def forward(self, inp, memory= None):
        x = inp
        _, _, H, W = x.shape
        paddingBottom = int(np.ceil(H/self.pad_size)*self.pad_size-H)
        paddingRight = int(np.ceil(W/self.pad_size)*self.pad_size-W)
        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)
        x = self.embedding(x)
        encoder_list = []
        memory_list = []
        if memory is not None:
            for encoder, memorizer, skip_x, down in zip(self.encoders, self.memorizers, memory[::-1], self.downs):
                x = x + memorizer(x, skip_x)
                x = encoder(x)
                encoder_list.append(x)
                x = down(x)
        else:
            for encoder, down in zip(self.encoders, self.downs):
                x = encoder(x)
                encoder_list.append(x)
                x = down(x)

        x = x + self.memorizers[-1](x, memory[0]) if memory is not None else x
        x = self.bottleneck(x)
        memory_list.append(x)

        for decoder, up, skip_x in zip(self.decoders, self.ups,  encoder_list[::-1]):
            x = up(x)
            x = x + skip_x
            x = decoder(x)
            memory_list.append(x)

        x = self.mapping(x)
        x = x[:, :, :H, :W] + inp
        return x, memory_list
    

class ProxUnroll(nn.Module):
    def __init__(self,
                stage = 6,
                color_channel = 1,
                dim = 48,
                head_dim = 24,
                groups = 24,
                window_size = 8,
                ffn_ratio = 2,
                mid_blocks = 2,
                enc_blocks = [2,2,2],
                dec_blocks = [2,2,2]):
        super().__init__()
        self.stage = stage

        Phi_256_256 = scio.loadmat('measurement_matrix/blind_learned_256_256_matrices.mat')
        H_256_256 = torch.from_numpy(Phi_256_256['H']).float()
        W_256_256 = torch.from_numpy(Phi_256_256['W']).float()
        self.H_256_256 = nn.Parameter(H_256_256, requires_grad=True)
        self.W_256_256 = nn.Parameter(W_256_256, requires_grad=True)

        Phi_321_481 = scio.loadmat('measurement_matrix/blind_learned_321_481_matrices.mat')
        H_321_481 = torch.from_numpy(Phi_321_481['H']).float()
        W_321_481 = torch.from_numpy(Phi_321_481['W']).float()
        self.H_321_481 = nn.Parameter(H_321_481, requires_grad=True)
        self.W_321_481 = nn.Parameter(W_321_481, requires_grad=True)

        Phi_512_512 = scio.loadmat('measurement_matrix/blind_learned_512_512_matrices.mat')
        H_512_512 = torch.from_numpy(Phi_512_512['H']).float()
        W_512_512 = torch.from_numpy(Phi_512_512['W']).float()
        self.H_512_512 = nn.Parameter(H_512_512, requires_grad=True)
        self.W_512_512 = nn.Parameter(W_512_512, requires_grad=True)

        self.rho = nn.Parameter(torch.Tensor([1.0]).repeat(6))
        self.beta = [1,1,1,1,1,0]
        self.restorer = SwinConvNet(color_channel, dim, head_dim, groups, window_size, ffn_ratio, mid_blocks, enc_blocks, dec_blocks)

    def prox_f(self, X, Y, H, W, HT, WT, rho):
        Delta_Y = Y - H @ X @ WT
        Delta_X = HT @ Delta_Y @ W
        Z = X + rho * Delta_X
        return Z

    def prox_g(self, GT, Z, a, b=1):
        X = b*GT + a*Z
        X = X / (a+b)
        return X
    
    def forward(self, GT, cr):
        """
        input X: [b,h,w]
        """
        b, h, w = GT.shape

        outputs = []
        prox_outputs = []
        GT = GT.unsqueeze(1)

        height = int(np.ceil(h*np.sqrt(cr)))
        width = int(np.ceil(w*np.sqrt(cr)))
        if h==256 and w==256:
            H = self.H_256_256[:height,:].unsqueeze(0).unsqueeze(1)
            W = self.W_256_256[:width,:].unsqueeze(0).unsqueeze(1)
        elif h==512 and w==512:
            H = self.H_512_512[:height,:].unsqueeze(0).unsqueeze(1)
            W = self.W_512_512[:width,:].unsqueeze(0).unsqueeze(1)
        elif h==321 and w==481:
            H = self.H_321_481[:height,:].unsqueeze(0).unsqueeze(1)
            W = self.W_321_481[:width,:].unsqueeze(0).unsqueeze(1)

        HT = H.transpose(-2, -1)
        WT = W.transpose(-2, -1)

        Y = H @ GT @ WT
        X = HT @ Y @ W
        outputs.append(X)
        U = torch.zeros_like(X)
        memory = None

        for i in range(self.stage):
            Z = self.prox_f(X-U,Y,H,W,HT,WT,self.rho[i])
            X, memory = self.restorer(Z+U, memory)
            prox_X = self.prox_g(GT,Z,self.beta[i])
            outputs.append(X)
            prox_outputs.append(prox_X)
            U = U + (Z - X)

        outputs = torch.stack(outputs, dim=0)
        prox_outputs = torch.stack(prox_outputs, dim=0)

        return outputs.squeeze(2), prox_outputs.squeeze(2)