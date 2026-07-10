from torch import nn
import torch
import torch.nn.functional as F
from timm.layers import trunc_normal_


class Adaptive_Spectral_Block(nn.Module):
    """
    Adaptive Spectral Block (ASB) for frequency domain feature enhancement.
    Uses FFT to adaptively filter high and low frequency components.
    """
    def __init__(self, dim):
        super().__init__()
        self.complex_weight_high = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        trunc_normal_(self.complex_weight_high, std=.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.threshold_param = nn.Parameter(torch.rand(1))
        self.norm = nn.LayerNorm(dim)

    def create_adaptive_high_freq_mask(self, x_fft):
        """Create adaptive high frequency mask based on energy distribution."""
        B, _, _ = x_fft.shape
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)
        flat_energy = energy.view(B, -1)
        median_energy = flat_energy.median(dim=1, keepdim=True)[0]
        median_energy = median_energy.view(B, 1)
        epsilon = 1e-6
        normalized_energy = energy / (median_energy + epsilon)
        adaptive_mask = ((normalized_energy > self.threshold_param).float() - self.threshold_param).detach() + self.threshold_param
        adaptive_mask = adaptive_mask.unsqueeze(-1)
        return adaptive_mask

    def forward(self, x_in):
        x_in = self.norm(x_in.transpose(1, 2))
        B, N, C = x_in.shape
        dtype = x_in.dtype
        x = x_in.to(torch.float32)
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x_weighted = x_fft * weight
        freq_mask = self.create_adaptive_high_freq_mask(x_fft)
        x_masked = x_fft * freq_mask.to(x.device)
        weight_high = torch.view_as_complex(self.complex_weight_high)
        x_weighted2 = x_masked * weight_high
        x_weighted += x_weighted2
        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')
        x = x.to(dtype)
        x = x.view(B, N, C)
        x = x.transpose(1, 2)
        return x


class SeparableConv1d(nn.Module):
    """Depthwise separable 1D convolution."""
    def __init__(self, ni, nf, ks):
        super(SeparableConv1d, self).__init__()
        self.depthwise_conv = nn.Conv1d(ni, ni, ks, stride=1, padding=int(ks/2), dilation=1, groups=ni, bias=False)
        self.pointwise_conv = nn.Conv1d(ni, nf, 1, stride=1, padding=0, dilation=1, groups=1, bias=False)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class Conv1x1(nn.Module):
    """1x1 convolution block with BatchNorm and optional activation."""
    def __init__(self, ni, nf, act=None):
        super(Conv1x1, self).__init__()
        layers = [
            nn.Conv1d(ni, nf, 1, bias=False),
            nn.BatchNorm1d(nf)
        ]
        if act is not None:
            layers.append(act)
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GAP1d(nn.Module):
    """Global Average Pooling 1D."""
    def __init__(self, output_size=1):
        super(GAP1d, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(output_size)

    def forward(self, x):
        return torch.flatten(self.gap(x), 1)


class XceptionModule(nn.Module):
    """Xception module with multi-scale separable convolutions."""
    def __init__(self, ni, nf, max_ks=40):
        super(XceptionModule, self).__init__()
        ks = [max_ks // (2 ** i) | 1 for i in range(3)]
        self.bottleneck = nn.Conv1d(ni, nf, 1, bias=False)
        self.convs = nn.ModuleList([SeparableConv1d(nf, nf, k) for k in ks])
        self.maxconvpool = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(ni, nf, 1, bias=False)
        )

    def forward(self, x):
        input_tensor = x
        x = self.bottleneck(x)
        convs_out = [conv(x) for conv in self.convs]
        x = torch.cat([l for l in convs_out] + [self.maxconvpool(input_tensor)], dim=1)
        return x


class XceptionBlock(nn.Module):
    """Xception block with stacked XceptionModules and residual connections."""
    def __init__(self, ni, nf, residual=True, **kwargs):
        super(XceptionBlock, self).__init__()
        self.residual = residual
        self.xception, self.shortcut = nn.ModuleList(), nn.ModuleList()
        
        # Pre-compute input/output channels for each layer
        n_in_list = []
        n_out_list = []
        for i in range(4):
            n_out = nf * 2 ** i
            n_in = ni if i == 0 else n_out_list[i-1] * 4
            n_in_list.append(n_in)
            n_out_list.append(n_out)
        
        # Build network layers
        for i in range(4):
            if self.residual and (i-1) % 2 == 0 and i > 0:
                shortcut_in = n_in_list[i-1]
                shortcut_out = n_out_list[i] * 4
                self.shortcut.append(Conv1x1(shortcut_in, shortcut_out))
            self.xception.append(XceptionModule(n_in_list[i], n_out_list[i], **kwargs))
        
        self.act = nn.ReLU()
        
    def forward(self, x):
        res = x
        for i in range(4):
            x = self.xception[i](x)
            if self.residual and (i + 1) % 2 == 0: 
                res = x = self.act(x + self.shortcut[i//2](res))
        return x    


class SpectralX(nn.Module):
    """
    SpectralX for Multi-Task Learning: Activity Recognition + Indoor Localization.
    
    A 1D CNN combining Xception architecture with Adaptive Spectral Block,
    featuring dual classification heads for multi-task prediction.
    
    Args:
        in_channels: Number of input channels (subcarriers)
        num_classes_act: Number of activity classes
        num_classes_loc: Number of location classes
        nf: Base number of feature channels (S=4, M=8, L=16)
        adaptive_size: Output size of adaptive pooling
        use_asb: Whether to use Adaptive Spectral Block
    """
    def __init__(self, in_channels, num_classes_act, num_classes_loc, 
                 nf=16, adaptive_size=50, use_asb=True, **kwargs):
        super(SpectralX, self).__init__()
        self.nf = nf
        self.use_asb = use_asb
        self.cls_nf = self.nf * 32
        
        # Shared feature extractor
        self.emb = XceptionBlock(in_channels, nf, **kwargs)
        
        # Adaptive Spectral Block
        self.asb = Adaptive_Spectral_Block(nf * 32)
        
        # Activity Recognition Head
        self.act_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(adaptive_size),
            Conv1x1(self.cls_nf, self.cls_nf // 2, nn.ReLU()), 
            Conv1x1(self.cls_nf // 2, self.cls_nf // 4, nn.ReLU()),
            Conv1x1(self.cls_nf // 4, num_classes_act),
            GAP1d(1)
        )
        
        # Indoor Localization Head
        self.loc_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(adaptive_size),
            Conv1x1(self.cls_nf, self.cls_nf // 2, nn.ReLU()), 
            Conv1x1(self.cls_nf // 2, self.cls_nf // 4, nn.ReLU()),
            Conv1x1(self.cls_nf // 4, num_classes_loc),
            GAP1d(1)
        )

    def forward(self, x):
        embedding = self.emb(x)
        if self.use_asb:
            embedding = self.asb(embedding)
        embedding_norm = F.normalize(embedding)
        act_output = self.act_head(embedding_norm)
        loc_output = self.loc_head(embedding_norm)   
        return act_output, loc_output, embedding_norm
