import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.cluster.hierarchy import weighted

class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(30 * x)
class AIGPE(nn.Module):
    

    def __init__(self, in_channels, embed_dim=64, scale_factor=4):
        super().__init__()
        self.s = scale_factor
        self.embed_dim = embed_dim

        #  (a, b, c, d)
        #  M = [[a, b], [c, d]]，
        self.aniso_predictor = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, 4, kernel_size=1)
        )

        # (INR) 
        self.inr_net = nn.Sequential(
            nn.Linear(2, embed_dim // 2),
            Sine() if hasattr(torch, 'sin') else nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )

        #  (s, s, 2)
        self.register_buffer("base_grid", self.__make_unit_grid(scale_factor))

    def __make_unit_grid(self, s):
     
        coords_h = torch.linspace(-1, 1, s)
        coords_w = torch.linspace(-1, 1, s)
        grid = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'), dim=-1)  # (s, s, 2)
        return grid

    def forward(self, f_low):
        B, C, H1, W1 = f_low.shape
        s = self.s

        #  (B, 4, H1, W1)
        m_low = self.aniso_predictor(f_low)

        #  (B, H1, W1, 2, 2)
        m_matrix = m_low.permute(0, 2, 3, 1).reshape(B, H1, W1, 2, 2)

        #  (s, s, 2)
    
        grid = self.base_grid  # (s, s, 2)

  
        # b: Batch, h: H1, w: W1, i: s, j: s, k: 2 (原坐标), l: 2 (变换后坐标)

        transformed_grid = torch.einsum('b h w k l, i j k -> b h w i j l', m_matrix, grid)


        #  (B*H1*W1*s*s, 2)
        flat_grid = transformed_grid.reshape(-1, 2)
        pe_flat = self.inr_net(flat_grid)

        #  (B, C_embed, H1*s, W1*s)
        pe_high = pe_flat.view(B, H1, W1, s, s, self.embed_dim)
        # (B, embed, H1, s, W1, s)
        pe_high = pe_high.permute(0, 5, 1, 3, 2, 4).contiguous()
        pe_high = pe_high.view(B, self.embed_dim, H1 * s, W1 * s)

        return m_low, pe_high

def build_dct_matrix(N: int, device=None, dtype=torch.float32):

    mat = torch.zeros(N, N, dtype=dtype)
    for k in range(N):
        for n in range(N):
            if k == 0:
                mat[k, n] = math.sqrt(1 / N)
            else:
                mat[k, n] = math.sqrt(2 / N) * math.cos(
                    math.pi * (n + 0.5) * k / N
                )
    return mat.to(device)
class DCTStructureGate(nn.Module):

    def __init__(self, block: int = 8, hidden: int = 32):
        super().__init__()
        self.block = block

        self.mlp = nn.Sequential(
            nn.Linear(block * block, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, block * block),
            nn.Sigmoid()  
        )

    def forward(self, dct_block: torch.Tensor) -> torch.Tensor:
        """
        dct_block: (B, C, Hb, Wb, b, b)
        return:    same shape
        """
        B, C, Hb, Wb, b, _ = dct_block.shape
        energy = dct_block.abs().mean(dim=1)      # (B, Hb, Wb, b, b)

        energy = energy.view(B * Hb * Wb, -1)     # (B*Hb*Wb, b*b)

        gate = self.mlp(energy)                   # (B*Hb*Wb, b*b)
        gate = gate.view(B, Hb, Wb, b, b)

        return dct_block * gate.unsqueeze(1)

class BlockDCTStructure(nn.Module):

    def __init__(self, block: int = 8):
        super().__init__()
        self.block = block

        self.register_buffer(
            "dct_mat",
            build_dct_matrix(block)
        )

        self.structure_gate = DCTStructureGate(block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        b = self.block

        assert H % b == 0 and W % b == 0, \
            f"Input spatial size must be divisible by block={b}"

        x = x.view(B, C, H // b, b, W // b, b)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
        # (B, C, Hb, Wb, b, b)


        dct = self.dct_mat.to(x.device)

        x = torch.einsum("ij,bcxyjk->bcxyik", dct, x)
        x = torch.einsum("bcxyik,kj->bcxyij", x, dct.t())

        x = self.structure_gate(x)

        x = torch.einsum("ij,bcxyjk->bcxyik", dct.t(), x)
        x = torch.einsum("bcxyik,kj->bcxyij", x, dct)

        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
        x = x.view(B, C, H, W)
        out = torch.sigmoid(x)
        return out


class SpatialGatherModule(nn.Module):
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale

    def forward(self, features: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        batch_size, num_classes, h, w = probs.size()
        probs_flat = probs.view(batch_size, num_classes, -1)
        probs_flat = F.softmax(self.scale * probs_flat, dim=2)
        feats_flat = features.view(batch_size, features.size(1), -1).permute(0, 2, 1)
        context = torch.matmul(probs_flat, feats_flat)
        return context

class RSUS(nn.Module):
    def __init__(self,
                 in_ch_high,
                 out_ch,
                 cm=None,
                 scale=2,
                 k_enc=3,
                 k_reassembly=5,
                 num_context=6):
        super().__init__()
        self.c1 = in_ch_high
        self.out = out_ch
        self.s = scale
        self.s2 = scale*scale
        self.k_enc = k_enc
        self.k_rea = k_reassembly
        self.pad = k_reassembly // 2
        self.k2 = k_reassembly*k_reassembly
        if cm is None:
            cm=64
        self.cm = cm
        self.num_context = num_context

        self.high_proj1 = nn.Sequential(
            nn.Conv2d(self.c1, self.cm, 1, 1, 0),
            nn.BatchNorm2d(self.cm),
            nn.GELU()
        )

        self.PE = AIGPE(in_channels=self.c1,embed_dim=self.cm,scale_factor=self.s)

        self.DCT_gate = BlockDCTStructure(block=8)
        self.DCT_proj =  nn.Sequential(
            nn.Conv2d(self.cm, self.cm*self.s2, 1, 1, 0),
            nn.BatchNorm2d(self.cm*self.s2),
            nn.GELU()
        )


        self.context_prob_conv = nn.Conv2d(self.cm, self.num_context, kernel_size=1)
        self.sgm = SpatialGatherModule(scale=1.0)
        self.context_fusion = nn.Sequential(
            nn.Linear(self.num_context * self.cm, self.num_context*self.cm),
            nn.LayerNorm(self.num_context*self.cm),
            nn.ReLU(inplace=True),
            nn.Linear(self.num_context*self.cm, self.num_context*self.cm)
        )
        self.low_num_gate = nn.Sequential(
            nn.Conv2d(self.cm,self.num_context,kernel_size=1),
            nn.Conv2d(self.num_context,self.num_context,kernel_size=1),
            nn.BatchNorm2d(self.num_context),
            nn.GELU()
        )

        self.content_encoder = nn.Conv2d(self.cm+4, self.k2*self.s2,
                                              kernel_size=self.k_enc,
                                              stride=1,
                                              padding=self.k_enc // 2)

        self.pe_refiner = nn.Sequential(
            nn.Conv2d(self.cm + self.cm, self.cm, 1),  
            nn.GELU(),
            nn.Conv2d(self.cm, self.cm, 3, padding=1)
        )

        self.out_proj = nn.Sequential(
            nn.Conv2d(self.cm, self.out, 1, 1, 0),
            nn.BatchNorm2d(self.out),
            nn.GELU()
        )
    def pixel_shuffle(self,x,up):
        up = int(up)
        N, C_up2, H, W = x.shape
        C = C_up2 // (up * up)
        # assert C * up * up == C_up2  
        x = x.view(N, C, up, up, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(N, C, H * up, W * up)
        return x

    def pixel_unshuffle(self,x,down):
        down = int(down)
        N,C,H,W = x.shape
        h = H // down
        w = W // down
        x = x.view(N,C,down,h,down,w)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
        x = x.view(N, C*down*down, h, w)
        return x

    def carafe(self,x,mask):
        N, C, H, W = x.shape
        x_pad = F.pad(x,(self.pad,self.pad,self.pad,self.pad),mode='reflect')
        x_unfold = F.unfold(x_pad,kernel_size=self.k_rea,stride=1)
        x_unfold = x_unfold.view(N, C, self.k2, H, W)

        mask = mask.view(N, -1, self.k2, H, W)
        _,S2,_,_,_ = mask.shape
        s = math.sqrt(S2)
        mask = F.softmax(mask,dim=2)
        weighted = torch.einsum('nskhw,nckhw->nschw',mask,x_unfold)
        weighted = weighted.permute(0,2,1,3,4).contiguous()
        weighted = weighted.view(N, C * S2, H, W)

        out = self.pixel_shuffle(weighted,s)
        return out

    def compute_mod_vector(self,x:torch.Tensor) -> torch.Tensor:
        probs = self.context_prob_conv(x)
        context = self.sgm(x, probs)
        mod_vec = self.context_fusion(context.view(x.size(0),-1))
        return mod_vec.view(x.size(0),x.size(1),-1)

    def forward(self, high_feat):
    
    
        m_low, pe_high = self.PE(high_feat)


        high_feat = self.high_proj1(high_feat)

        #  (DCT)
        low_DCT_gate = self.DCT_gate(high_feat)
        low_DCT_gate = self.pixel_shuffle(self.DCT_proj(low_DCT_gate), self.s)

       
        low_num_gated = self.low_num_gate(high_feat)
        vector = self.compute_mod_vector(high_feat).permute(0, 2, 1)
        low_vector = torch.einsum('bkc,bkhw -> bchw', vector, low_num_gated)

        #  (融入 m_low)

        feat_for_mask = torch.cat([high_feat * (1 + low_vector), m_low], dim=1)
        low_gate_kernels = self.content_encoder(feat_for_mask)

        # 6. 动态重组 (CARAFE)
        high_up = self.carafe(high_feat, low_gate_kernels)  # 结果为 (B, cm, H2, W2)

        # (融入 pe_high)

        combined_pe = torch.cat([high_up, pe_high], dim=1)
        high_up = high_up + self.pe_refiner(combined_pe)


        out = self.out_proj(high_up * low_DCT_gate)
        return out

if __name__ == "__main__":
    print("RSUS testing...")
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")




    x = torch.randn(2, 64, 16, 16).to(device)


    up = RSUS(
       in_ch_high=64,
        out_ch=64,
        scale=4
    ).to(device)
    up.eval()


    print('2倍上采样测试')
    out = up(x)
    print(f"输入: {x.shape}→ 输出: {out.shape}]")
