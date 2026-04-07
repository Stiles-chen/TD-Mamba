"""
Dual-Branch GCN + Mamba model for skeleton-based action recognition.

Architecture:
  - GCN branch  : unit_gcn (from tdgcn) models local joint-to-joint
                  relationships using the skeleton graph adjacency matrix.
  - Mamba branch: SpatialMamba models global spatial dependencies by
                  scanning over the joint sequence with a bidirectional
                  Mamba SSM (backed by the official mamba-ssm CUDA kernel).
  - Gate fusion : a learnable sigmoid gate adaptively blends the two
                  complementary feature streams.
  - TCN         : MultiScale_TemporalConv captures multi-scale temporal
                  patterns after spatial fusion.

Input/output contract is identical to model.tdgcn.Model so that the
existing training pipeline (main.py) and SHREC17 feeder require no changes.

Requires: pip install mamba-ssm  (CUDA ≥ 11.6, PyTorch ≥ 1.12)

Input  : (N, C, T, V, M)  –  batch, channels, time, joints, persons
Output : (N, num_class)
"""
#加入了手动控制哪层开始使用mamba
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from mamba_ssm import Mamba

# ──────────────────────────────────────────────────────────────────────────────
# Utility helpers (identical to tdgcn.py)
# ──────────────────────────────────────────────────────────────────────────────

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


# ──────────────────────────────────────────────────────────────────────────────
# Temporal convolution blocks (identical to tdgcn.py)
# ──────────────────────────────────────────────────────────────────────────────

class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MultiScale_TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 dilations=[1, 2, 3, 4], residual=True, residual_kernel_size=1):
        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, \
            '# out channels should be multiples of # branches'

        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size] * len(dilations)

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(branch_channels, branch_channels,
                             kernel_size=ks, stride=stride, dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(branch_channels)
        ))
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0,
                      stride=(stride, 1)),
            nn.BatchNorm2d(branch_channels)
        ))

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels,
                                         kernel_size=residual_kernel_size, stride=stride)
        self.apply(weights_init)

    def forward(self, x):
        res = self.residual(x)
        branch_outs = [branch(x) for branch in self.branches]
        out = torch.cat(branch_outs, dim=1)
        out += res
        return out


# ──────────────────────────────────────────────────────────────────────────────
# GCN components (identical to tdgcn.py)
# ──────────────────────────────────────────────────────────────────────────────

class TDGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(TDGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == 3 or in_channels == 9:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, A=None, alpha=1, beta=1, gamma=0.1):
        x1, x3 = self.conv1(x).mean(-2), self.conv3(x)
        x1 = self.tanh(x1.unsqueeze(-1) - x1.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
        x4 = self.tanh(x3.mean(-3).unsqueeze(-1) - x3.mean(-3).unsqueeze(-2))
        x3 = x3.permute(0, 2, 1, 3)
        x5 = torch.einsum('btmn,btcn->bctm', x4, x3)
        x1 = x1 * beta + x5 * gamma
        return x1


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4,
                 adaptive=True, residual=True):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList([TDGC(in_channels, out_channels)
                                    for _ in range(self.num_subset)])

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0

        if self.adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)),
                              requires_grad=False)

        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.gamma = nn.Parameter(torch.tensor(0.1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
        y = None
        A = self.PA if self.adaptive else self.A.cuda(x.get_device())
        for i in range(self.num_subset):
            z = self.convs[i](x, A[i], self.alpha, self.beta, self.gamma)
            y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)
        return y


# ──────────────────────────────────────────────────────────────────────────────
# Mamba SSM – backed by the official mamba-ssm CUDA kernel
# (SelectiveSSM has been replaced by mamba_ssm.Mamba)
# ──────────────────────────────────────────────────────────────────────────────


class SpatialMamba(nn.Module):
    """
    Applies a bidirectional selective SSM over the joint (spatial) dimension.

    The joint sequence order follows the SHREC17 skeleton topology: starting
    from the wrist (root), traversing each finger chain in order.  This
    topology-aware ordering gives the SSM a natural inductive bias.

    For each frame independently:
        (N*M, C, T, V) → reshape to (N*M*T, V, C)
                       → forward SSM + backward SSM
                       → fuse and project
                       → reshape back to (N*M, C, T, V)

    Args:
        channels : number of feature channels C
        d_state  : SSM state size (default 16)
        d_conv   : depthwise-conv kernel (default 4)
        expand   : inner-dim expansion (default 2)
    """

    def __init__(self, channels, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.channels = channels
        # Official mamba-ssm.Mamba: includes input projection, conv1d, SSM scan,
        # output projection and residual – all fused in a single CUDA kernel.
        self.ssm_fwd = Mamba(d_model=channels, d_state=d_state,
                             d_conv=d_conv, expand=expand)
        self.ssm_bwd = Mamba(d_model=channels, d_state=d_state,
                             d_conv=d_conv, expand=expand)
        # Fuse forward and backward outputs
        self.fuse_proj = nn.Sequential(
            nn.Linear(channels * 2, channels, bias=False),
            nn.LayerNorm(channels),
        )

    def forward(self, x):
        """
        x : (N*M, C, T, V)
        Returns same shape (N*M, C, T, V).
        """
        NM, C, T, V = x.shape

        # Reshape to sequence-first layout expected by SSM
        # (NM, C, T, V) → (NM*T, V, C)
        x_seq = x.permute(0, 2, 3, 1).contiguous().view(NM * T, V, C)

        fwd = self.ssm_fwd(x_seq)                     # (NM*T, V, C)
        bwd = self.ssm_bwd(x_seq.flip(1)).flip(1)     # reverse and un-reverse

        # Fuse bidirectional outputs
        y = self.fuse_proj(torch.cat([fwd, bwd], dim=-1))  # (NM*T, V, C)

        # Restore original tensor layout
        y = y.view(NM, T, V, C).permute(0, 3, 1, 2).contiguous()  # (NM, C, T, V)
        return y


# ──────────────────────────────────────────────────────────────────────────────
# Gate fusion
# ──────────────────────────────────────────────────────────────────────────────

class GateFusion(nn.Module):
    """
    Learnable sigmoid gate that adaptively blends GCN and Mamba features.

        F = σ(W [F_gcn ; F_mamba]) ⊙ F_gcn  +  (1 − σ(…)) ⊙ F_mamba

    The gate is channel-wise so each feature channel can independently
    decide how much to trust local vs. global information.

    Args:
        channels : C (number of feature channels)
    """

    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )
        bn_init(self.gate[1], 1)

    def forward(self, f_gcn, f_mamba):
        """
        f_gcn, f_mamba : (N*M, C, T, V)
        Returns        : (N*M, C, T, V)
        """
        alpha = self.gate(torch.cat([f_gcn, f_mamba], dim=1))
        return alpha * f_gcn + (1.0 - alpha) * f_mamba


# ──────────────────────────────────────────────────────────────────────────────
# Dual-branch block
# ──────────────────────────────────────────────────────────────────────────────

class DualBranchBlock(nn.Module):
    """
    One processing block combining:
      1. GCN branch (unit_gcn)  – local joint-to-joint relationships
      2. Mamba branch (SpatialMamba)  – global spatial dependencies
      3. Gate fusion (GateFusion)
      4. TCN (MultiScale_TemporalConv)  – temporal pattern extraction

    Input / output shape: (N*M, C_in, T, V) → (N*M, C_out, T', V)
    where T' = T // stride.

    Args:
        in_channels  : input channel dimension
        out_channels : output channel dimension
        A            : adjacency matrix (num_subset, V, V)
        stride       : temporal downsampling factor (default 1)
        residual     : whether to use a residual skip connection
        adaptive     : whether GCN uses learnable adjacency
        kernel_size  : TCN kernel size
        dilations    : TCN dilation list
        d_state      : Mamba SSM state size
        mamba_expand : Mamba inner-dim expansion factor
    """

    def __init__(self, in_channels, out_channels, A,
                 stride=1, residual=True, adaptive=True,
                 kernel_size=5, dilations=None,
                 d_state=16, mamba_expand=2):
        super().__init__()
        if dilations is None:
            dilations = [1, 2]

        # GCN branch
        self.gcn = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)

        # Mamba branch – receives the *input* x, not the GCN output,
        # so both branches process the same representation independently.
        self.mamba = SpatialMamba(in_channels, d_state=d_state,
                                  expand=mamba_expand)

        # Project Mamba output to out_channels if they differ
        if in_channels != out_channels:
            self.mamba_proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.mamba_proj = nn.Identity()

        # Gate fusion
        self.fusion = GateFusion(out_channels)

        # TCN after fusion
        self.tcn = MultiScale_TemporalConv(
            out_channels, out_channels,
            kernel_size=kernel_size, stride=stride,
            dilations=dilations, residual=False)

        # Block-level residual
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels,
                                     kernel_size=1, stride=stride)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # (N*M, C_in, T, V)
        res = self.residual(x)

        # GCN branch
        f_gcn = self.gcn(x)                  # (N*M, C_out, T, V)

        # Mamba branch
        f_mamba = self.mamba(x)              # (N*M, C_in, T, V)
        f_mamba = self.mamba_proj(f_mamba)   # (N*M, C_out, T, V)

        # Gate fusion
        fused = self.fusion(f_gcn, f_mamba)  # (N*M, C_out, T, V)

        # TCN
        out = self.tcn(fused)                # (N*M, C_out, T', V)
        out = self.relu(out + res)
        return out


class GCNOnlyBlock(nn.Module):
    """GCN+TCN block used when Mamba is disabled for a given layer."""

    def __init__(self, in_channels, out_channels, A,
                 stride=1, residual=True, adaptive=True,
                 kernel_size=5, dilations=None):
        super().__init__()
        if dilations is None:
            dilations = [1, 2]

        self.gcn = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
        self.tcn = MultiScale_TemporalConv(
            out_channels, out_channels,
            kernel_size=kernel_size, stride=stride,
            dilations=dilations, residual=False)

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels,
                                     kernel_size=1, stride=stride)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.residual(x)
        out = self.tcn(self.gcn(x))
        out = self.relu(out + res)
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Full model
# ──────────────────────────────────────────────────────────────────────────────

class Model(nn.Module):
    """
    Dual-Branch GCN + Mamba model for skeleton-based action recognition.

    Drop-in replacement for model.tdgcn.Model; all constructor arguments and
    the forward signature are the same so that main.py and the SHREC17 config
    need no modifications.

    Architecture mirrors the 10-layer channel progression of TD-GCN:
        [64]*4 → [128]*3 → [256]*3
    but each TCN_GCN_unit is replaced by a DualBranchBlock.

    Args:
        num_class   : number of action classes
        num_point   : number of skeleton joints (22 for SHREC17)
        num_person  : number of persons per sample (1 for SHREC17)
        graph       : dotted-path string to the graph class
        graph_args  : kwargs forwarded to the graph constructor
        in_channels : input feature channels (3 for xyz coordinates)
        drop_out    : dropout probability (0 to disable)
        adaptive    : use learnable adjacency in GCN
        d_state     : Mamba SSM state dimension
        mamba_expand: Mamba inner-dim expansion factor
        mamba_start_layer: enable Mamba from this layer index (1..10), inclusive
        mamba_layers: explicit list of layer indices (1..10) that use Mamba;
                      if provided, it overrides mamba_start_layer
    """

    TOTAL_LAYERS = 10

    @staticmethod
    def _resolve_mamba_layers(mamba_start_layer, mamba_layers):
        """Resolve enabled Mamba layers with 1-based indexing (l1..l10)."""
        if mamba_layers is not None:
            if not isinstance(mamba_layers, (list, tuple, set)):
                raise TypeError("mamba_layers must be a list/tuple/set of layer indices")
            layers = {int(i) for i in mamba_layers}
        elif mamba_start_layer is not None:
            start = int(mamba_start_layer)
            if start < 1 or start > Model.TOTAL_LAYERS:
                raise ValueError("mamba_start_layer must be in [1, 10]")
            layers = set(range(start, Model.TOTAL_LAYERS + 1))
        else:
            layers = set(range(1, Model.TOTAL_LAYERS + 1))

        invalid = sorted(i for i in layers if i < 1 or i > Model.TOTAL_LAYERS)
        if invalid:
            raise ValueError("mamba layer indices must be in [1, 10], got {}".format(invalid))
        return layers

    def __init__(self, num_class=60, num_point=25, num_person=2,
                 graph=None, graph_args=None, in_channels=3,
                 drop_out=0, adaptive=True,
                 d_state=16, mamba_expand=2,
                 mamba_start_layer=None, mamba_layers=None):
        super(Model, self).__init__()

        if graph_args is None:
            graph_args = {}
        if graph is None:
            raise ValueError("graph must be provided")

        Graph = import_class(graph)
        self.graph = Graph(**graph_args)
        A = self.graph.A                     # (num_subset, V, V)

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        kw = dict(adaptive=adaptive, d_state=d_state, mamba_expand=mamba_expand)
        self.mamba_layers = sorted(self._resolve_mamba_layers(mamba_start_layer, mamba_layers))

        layer_specs = [
            (in_channels, base_channel, 1, False),
            (base_channel, base_channel, 1, True),
            (base_channel, base_channel, 1, True),
            (base_channel, base_channel, 1, True),
            (base_channel, base_channel * 2, 2, True),
            (base_channel * 2, base_channel * 2, 1, True),
            (base_channel * 2, base_channel * 2, 1, True),
            (base_channel * 2, base_channel * 4, 2, True),
            (base_channel * 4, base_channel * 4, 1, True),
            (base_channel * 4, base_channel * 4, 1, True),
        ]

        self.layer_names = []
        for idx, (in_c, out_c, stride, residual) in enumerate(layer_specs, start=1):
            if idx in self.mamba_layers:
                block = DualBranchBlock(
                    in_c, out_c, A,
                    stride=stride, residual=residual,
                    **kw)
            else:
                block = GCNOnlyBlock(
                    in_c, out_c, A,
                    stride=stride, residual=residual,
                    adaptive=adaptive)
            name = "l{}".format(idx)
            setattr(self, name, block)
            self.layer_names.append(name)

        self.fc = nn.Linear(base_channel * 4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

        self.drop_out = nn.Dropout(drop_out) if drop_out else lambda x: x

    def forward(self, x):
        # Support both (N, T, V*C) flat input and canonical (N, C, T, V, M) input
        if len(x.shape) == 3:
            N, T, VC = x.shape
            # Reshape: (N, T, V*C) → (N, C, T, V, M=1)
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()

        # BatchNorm over flattened (person, joint, channel) for each time step
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        for name in self.layer_names:
            x = getattr(self, name)(x)

        # Global average pooling over time and joints, then classify
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)
        return self.fc(x)