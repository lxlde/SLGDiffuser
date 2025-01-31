import torch
import math
import torch.nn as nn
import string
import numpy as np


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(
            x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.temb_proj = torch.nn.Linear(temb_channels,
                                         out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb = None):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class AttnBlock(nn.Module):
    def __init__(self, in_features, in_channels, cross=False, text_ch=180):
        super().__init__()
        self.in_channels = in_channels
        self.in_features = in_features
        self.text_ch = text_ch
        self.cross = cross

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        # self.k = torch.nn.Conv2d(in_channels,
        #                          in_channels,
        #                          kernel_size=1,
        #                          stride=1,
        #                          padding=0)
        # self.v = torch.nn.Conv2d(in_channels,
        #                          in_channels,
        #                          kernel_size=1,
        #                          stride=1,
        #                          padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

        # self.q = nn.Linear(in_channels, in_channels)
        self.k = nn.Linear(text_ch if cross else in_channels, in_features if cross else in_channels)
        self.v = nn.Linear(text_ch if cross else in_channels, in_features if cross else in_channels)

    def forward(self, x, condition = None):
        h_ = self.norm(x)
        B,C,H,W = x.shape
        assert (H*W == self.in_features)
        if self.cross == False:
            q = self.q(h_)
            k = self.k(h_)
            v = self.v(h_)
        else:
            assert (condition is not None)
            condition = condition.reshape(B, 1, -1)
            condition = condition.repeat(1, C, 1)  #B*C*264
            q = self.q(h_)
            k = self.k(condition)#B*C*(H*W)
            v = self.v(condition)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   # b,hw,c
        #k = k.reshape(b, c, h*w)  # b,c,hw
        w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        #v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        # if self.cross:
        #     return h_
        return x+h_


class Tokenizer:
    def __init__(self, max_len=15):
        self.tokens = {}
        self.chars = {}
        self.text = '_' + string.ascii_letters + string.digits + '.?!,\'\"- '
        self.numbers = np.arange(2, len(self.text) + 2)
        self.create_dict()
        self.vocab_size = len(self.text) + 2
        self.max_len = max_len

    def create_dict(self):
        for char, token, in zip(self.text, self.numbers):
            self.tokens[char] = token
            self.chars[token] = char
        self.chars[0], self.chars[1] = ' ', '<end>'  # only for decoding

    def encode(self, text):
        if isinstance(text, list) or isinstance(text, tuple):
            batch_size = len(text)
        else:
            text = [text]
            batch_size = 1

        all_tokenized = []
        for item in text:
            tokenized = []
            for char in item:
                if char in self.text:
                    tokenized.append(self.tokens[char])
                else:
                    tokenized.append(2)  # unknown character is '_', which has index 2
            while(len(tokenized) < self.max_len - 1):
                tokenized.append(2)

            tokenized.append(1)  # 1 is the end of sentence character
            all_tokenized.append(tokenized)
        return torch.tensor(all_tokenized)

    def decode(self, tokens):
        if isinstance(tokens, torch.Tensor): tokens = tokens.numpy()
        text = [self.chars[token] for token in tokens]
        return ''.join(text)
