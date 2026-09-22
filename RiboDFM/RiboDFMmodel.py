import math
import contextlib
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

def gather_nodes(nodes, neighbor_idx):
    neighbors_flat = neighbor_idx.reshape((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.reshape(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features

def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    return torch.cat([h_neighbors, h_nodes], -1)

def _get_final_step_mean_probability(probs, S):
    picked_probs = torch.gather(probs, -1, S.clamp(0, 3).unsqueeze(-1)).squeeze(-1)
    mean_probability = picked_probs.mean(dim=-1)
    return mean_probability

def _parse_scheduler_list(x):
    if isinstance(x, (list, tuple)):
        return [str(v).strip() for v in x if str(v).strip()]
    return [v.strip() for v in str(x).split(',') if v.strip()]

class GatedMLP(nn.Module):

    def __init__(self, d_in, d_out, d_hidden=None, dropout=0.1):
        super().__init__()
        if d_hidden is None:
            d_hidden = d_out
        self.W_in = nn.Linear(d_in, d_hidden * 2, bias=True)
        self.W_out = nn.Linear(d_hidden, d_out, bias=True)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        raw = self.W_in(x)
        val, gate = raw.chunk(2, dim=-1)
        h = val * self.act(gate)
        h = self.dropout(h)
        return self.W_out(h)

class DropPath(nn.Module):

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class PositionalEncodings(nn.Module):

    def __init__(self, num_embeddings, max_relative_feature=32):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2 * max_relative_feature + 1 + 1, num_embeddings)

    def forward(self, offset):
        d = torch.clip(offset + self.max_relative_feature, 0, 2 * self.max_relative_feature)
        d_onehot = F.one_hot(d, 2 * self.max_relative_feature + 1 + 1)
        return self.linear(d_onehot.float())

class TimestepEmbedder(nn.Module):

    def __init__(self, hidden_dim, fourier_dim=64):
        super().__init__()
        self.fourier_dim = int(fourier_dim)
        self.mlp = nn.Sequential(nn.Linear(self.fourier_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, t):
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=torch.float32)
        if t.ndim == 0:
            t = t[None]
        t = t.float()
        device = t.device
        half = self.fourier_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=device, dtype=torch.float32) / max(half - 1, 1))
        args = t[:, None] * freqs[None, :] * 2.0 * math.pi
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.size(-1) < self.fourier_dim:
            emb = F.pad(emb, (0, self.fourier_dim - emb.size(-1)))
        return self.mlp(emb)

class ImprovedEncLayer(nn.Module):

    def __init__(self, num_hidden, num_in=None, dropout=0.1, drop_path=0.1):
        super().__init__()
        msg_in_dim = 3 * num_hidden
        self.message_mlp1 = GatedMLP(d_in=msg_in_dim, d_out=num_hidden, d_hidden=num_hidden * 2, dropout=dropout)
        self.dense = GatedMLP(d_in=num_hidden, d_out=num_hidden, d_hidden=num_hidden * 4, dropout=dropout)
        self.message_mlp2 = GatedMLP(d_in=msg_in_dim, d_out=num_hidden, d_hidden=num_hidden * 2, dropout=dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm_edge = nn.LayerNorm(num_hidden)
        self.dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        residue = h_V
        h_V = self.norm1(h_V)
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.message_mlp1(h_EV)
        if mask_attend is not None:
            denom = mask_attend.sum(dim=-1, keepdim=True).clamp_min(1.0)
            dh = (h_message * mask_attend.unsqueeze(-1)).sum(dim=-2) / denom
        else:
            dh = h_message.mean(dim=-2)
        h_V = residue + self.drop_path(self.dropout(dh))
        dh = self.dense(self.norm2(h_V))
        h_V = h_V + self.drop_path(self.dropout(dh))
        if mask_V is not None:
            h_V = h_V * mask_V.unsqueeze(-1)
        residue = h_E
        h_E = self.norm_edge(h_E)
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.message_mlp2(h_EV)
        if mask_attend is not None:
            h_message = h_message * mask_attend.unsqueeze(-1)
        h_E = residue + self.drop_path(self.dropout(h_message))
        return (h_V, h_E)

class ImprovedDecLayer(nn.Module):

    def __init__(self, num_hidden, num_in=None, dropout=0.1, drop_path=0.1, ss_message_mode='gate', ss_scale_init=0.0, ss_max_scale=2.0, ss_exp_clip=2.0, use_ss=True):
        super().__init__()
        msg_in_dim = 4 * num_hidden
        self.message_mlp = GatedMLP(d_in=msg_in_dim, d_out=num_hidden, d_hidden=num_hidden * 2, dropout=dropout)
        self.dense = GatedMLP(d_in=num_hidden, d_out=num_hidden, d_hidden=num_hidden * 4, dropout=dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.attn_proj = nn.Linear(num_hidden, 1, bias=False)
        self.ss_message_mode = str(ss_message_mode).lower()
        self.ss_max_scale = float(ss_max_scale)
        self.ss_exp_clip = float(ss_exp_clip)
        self.use_ss = bool(use_ss)
        if self.use_ss:
            self.ss_scale = nn.Parameter(torch.tensor(float(ss_scale_init)))

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None, ss_weights=None):
        residue = h_V
        h_Vn = self.norm1(h_V)
        h_V_expand = h_Vn.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_E], dim=-1)
        h_message = self.message_mlp(h_EV)
        if self.use_ss and ss_weights is not None:
            ss = ss_weights.clamp(0.0, 1.0)
            if self.ss_message_mode == 'exp':
                raw = torch.clamp(self.ss_scale * ss, -self.ss_exp_clip, self.ss_exp_clip)
                multiplier = torch.exp(raw)
            elif self.ss_message_mode == 'none':
                multiplier = 1.0
            else:
                scale = self.ss_max_scale * torch.sigmoid(self.ss_scale)
                multiplier = 1.0 + scale * ss
            h_message = h_message * multiplier
        if mask_attend is not None:
            mask_attend_ = mask_attend if mask_attend.ndim == 4 else mask_attend.unsqueeze(-1)
            attn_logits = self.attn_proj(h_message)
            attn = F.softmax(attn_logits + torch.log(mask_attend_.clamp_min(1e-08)), dim=-2)
            dh = (h_message * attn).sum(dim=-2)
        else:
            dh = h_message.mean(dim=-2)
        h_V = residue + self.drop_path(self.dropout(dh))
        dh = self.dense(self.norm2(h_V))
        h_V = h_V + self.drop_path(self.dropout(dh))
        if mask_V is not None:
            h_V = h_V * mask_V.unsqueeze(-1)
        return h_V

class RNAFeatures(nn.Module):

    def __init__(self, edge_features, node_features, num_positional_embeddings=16, num_rbf=16, top_k=30, augment_eps=0.0):
        super().__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.n_all_atoms = 13
        self.n_core_atoms = 3
        rbf_in = num_rbf * self.n_core_atoms * self.n_core_atoms
        edge_in = num_positional_embeddings + rbf_in
        self.embeddings = PositionalEncodings(num_positional_embeddings)
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features)

    def _dist(self, X_repr):
        B, L, _ = X_repr.shape
        dX = torch.unsqueeze(X_repr, 1) - torch.unsqueeze(X_repr, 2)
        D = torch.sqrt(torch.sum(dX ** 2, 3) + 1e-06)
        D = D + torch.eye(L, device=D.device).unsqueeze(0) * 1000000.0
        k = min(self.top_k, X_repr.shape[1])
        _, E_idx = torch.topk(D, k, dim=-1, largest=False)
        return E_idx

    def _rbf(self, D):
        device = D.device
        D_min, D_max, D_count = (0.0, 20.0, self.num_rbf)
        D_mu = torch.linspace(D_min, D_max, D_count, device=device).view(1, 1, 1, -1)
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        return torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(torch.sum((A.unsqueeze(2) - B.unsqueeze(1)) ** 2, -1) + 1e-06)
        D_neighbor = torch.gather(D_A_B, 2, E_idx)
        return self._rbf(D_neighbor)

    def get_Cb(self, N, Ca, C, w_a, w_b, w_c):
        b = Ca - N
        c = C - Ca
        a = torch.cross(b, c, dim=-1)
        return w_a * a + w_b * b + w_c * c + Ca

    def forward(self, X, residue_idx):
        if self.augment_eps > 0.0 and self.training:
            X = X + self.augment_eps * torch.randn_like(X)
        atoms_list = [X[:, :, i, :] for i in range(12)]
        C4_p = atoms_list[5]
        C2_p = atoms_list[9]
        C1_p = atoms_list[11]
        N_na = self.get_Cb(C4_p, C1_p, C2_p, w_a=-0.56967352, w_b=0.51055973, w_c=-0.53122153)
        core_atoms = [C4_p, C1_p, N_na]
        E_idx = self._dist(C4_p)
        RBF_all = []
        for i in range(len(core_atoms)):
            for j in range(len(core_atoms)):
                RBF_all.append(self._get_rbf(core_atoms[i], core_atoms[j], E_idx))
        RBF_all = torch.cat(RBF_all, dim=-1)
        offset = residue_idx.unsqueeze(-1) - residue_idx.unsqueeze(1)
        offset = torch.gather(offset, 2, E_idx)
        E_positional = self.embeddings(offset.long())
        E = torch.cat((RBF_all, E_positional), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)
        return (E, E_idx)

class RNAMPNN_DFM(nn.Module):

    def __init__(self, args, node_features=256, edge_features=256, hidden_dim=256, augment_eps=0.0, dropout=0.2, use_ss=True):
        super().__init__()
        num_encoder_layers = args.num_encoder_layers
        num_decoder_layers = args.num_decoder_layers
        k_neighbors = args.k_neighbors
        self.augment_eps = augment_eps
        self.hidden_dim = int(hidden_dim)
        self.use_ss = bool(use_ss)
        self.num_letters = 4
        self.mask_token = 4
        self.vocab = 5
        self.dfm_num_steps = int(getattr(args, 'dfm_num_steps', 32))
        self.dfm_scheduler = str(getattr(args, 'dfm_scheduler', 'cosine'))
        self.dfm_train_min_t = float(getattr(args, 'dfm_train_min_t', 0.0001))
        self.dfm_train_max_t = float(getattr(args, 'dfm_train_max_t', 1.0 - 0.0001))
        self.dfm_random_train_scheduler = bool(getattr(args, 'dfm_random_train_scheduler', False))
        self.dfm_train_schedulers = _parse_scheduler_list(getattr(args, 'dfm_train_schedulers', self.dfm_scheduler))
        if len(self.dfm_train_schedulers) == 0:
            self.dfm_train_schedulers = [self.dfm_scheduler]
        self.dfm_time_condition = str(getattr(args, 'dfm_time_condition', 'kappa')).lower()
        self.features = RNAFeatures(edge_features=edge_features, node_features=node_features, top_k=k_neighbors, augment_eps=augment_eps)
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_t = TimestepEmbedder(hidden_dim, fourier_dim=int(getattr(args, 'dfm_time_fourier_dim', 64)))
        self.seq_time_norm = nn.LayerNorm(hidden_dim)
        self.node_time_norm = nn.LayerNorm(hidden_dim)
        self.encoder_layers = nn.ModuleList([ImprovedEncLayer(hidden_dim, hidden_dim * 2, dropout=dropout) for _ in range(num_encoder_layers)])
        if self.use_ss:
            self.ss_edge_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.use_rnafm = bool(getattr(args, 'use_rnafm', True))
        if not self.use_rnafm:
            raise ValueError('This RNA-FM version fully replaces W_s, so --use_rnafm must be enabled.')
        self.rnafm_layer = int(getattr(args, 'rnafm_layer', 12))
        self.rnafm_dim = int(getattr(args, 'rnafm_dim', 640))
        self.freeze_rnafm = bool(getattr(args, 'freeze_rnafm', True))
        self.rnafm_use_amp = bool(getattr(args, 'rnafm_use_amp', True))
        self.rnafm_unknown_mode = str(getattr(args, 'rnafm_unknown_mode', 'n'))
        self.rnafm_use_residue_idx_gaps = bool(getattr(args, 'rnafm_use_residue_idx_gaps', True))
        self.rnafm_max_full_len = int(getattr(args, 'rnafm_max_full_len', 1024))
        self.rnafm_gap_fallback_no_gaps = bool(getattr(args, 'rnafm_gap_fallback_no_gaps', False))
        import fm
        rnafm_checkpoint = getattr(args, 'rnafm_checkpoint', None)
        rnafm_model, rnafm_alphabet = fm.pretrained.rna_fm_t12(model_location=rnafm_checkpoint)
        self.rnafm_model = rnafm_model
        self.rnafm_alphabet = rnafm_alphabet
        self.rnafm_model.eval()
        if self.freeze_rnafm:
            for p in self.rnafm_model.parameters():
                p.requires_grad_(False)
        self._init_rnafm_token_indices()
        self.rnafm_proj = nn.Sequential(nn.LayerNorm(self.rnafm_dim), nn.Linear(self.rnafm_dim, hidden_dim))
        ss_message_mode = str(getattr(args, 'ss_message_mode', 'gate'))
        ss_scale_init = float(getattr(args, 'ss_scale_init', 0.0))
        ss_max_scale = float(getattr(args, 'ss_max_scale', 2.0))
        ss_exp_clip = float(getattr(args, 'ss_exp_clip', 2.0))
        self.decoder_layers = nn.ModuleList([ImprovedDecLayer(hidden_dim, hidden_dim * 3, dropout=dropout, ss_message_mode=ss_message_mode, ss_scale_init=ss_scale_init, ss_max_scale=ss_max_scale, ss_exp_clip=ss_exp_clip, use_ss=self.use_ss) for _ in range(num_decoder_layers)])
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.W_out = nn.Linear(hidden_dim, self.num_letters, bias=True)

    def train(self, mode: bool=True):
        super().train(mode)
        if getattr(self, 'freeze_rnafm', False) and hasattr(self, 'rnafm_model'):
            self.rnafm_model.eval()
        return self

    def _alphabet_get_idx(self, name, default=None):
        if hasattr(self.rnafm_alphabet, 'get_idx'):
            try:
                idx = self.rnafm_alphabet.get_idx(name)
                if idx is not None:
                    return int(idx)
            except Exception:
                pass
        return default

    def _init_rnafm_token_indices(self):
        alphabet = self.rnafm_alphabet
        self.rnafm_pad_idx = int(getattr(alphabet, 'padding_idx', self._alphabet_get_idx('<pad>', 0)))
        self.rnafm_cls_idx = int(getattr(alphabet, 'cls_idx', self._alphabet_get_idx('<cls>', 0)))
        self.rnafm_eos_idx = int(getattr(alphabet, 'eos_idx', self._alphabet_get_idx('<eos>', 2)))
        self.rnafm_A_idx = self._alphabet_get_idx('A')
        self.rnafm_U_idx = self._alphabet_get_idx('U')
        self.rnafm_C_idx = self._alphabet_get_idx('C')
        self.rnafm_G_idx = self._alphabet_get_idx('G')
        if any((v is None for v in [self.rnafm_A_idx, self.rnafm_U_idx, self.rnafm_C_idx, self.rnafm_G_idx])):
            raise RuntimeError('Failed to locate A/U/C/G tokens in RNA-FM alphabet.')
        self.rnafm_N_idx = self._alphabet_get_idx('N', self.rnafm_A_idx)
        self.rnafm_gap_idx = self._alphabet_get_idx('-', self.rnafm_N_idx)
        mode = self.rnafm_unknown_mode.lower()
        if mode in ['n', 'unk', 'unknown']:
            self.rnafm_condition_idx = int(self.rnafm_N_idx)
        elif mode in ['gap', '-']:
            self.rnafm_condition_idx = int(self.rnafm_gap_idx)
        else:
            self.rnafm_condition_idx = int(self.rnafm_N_idx)
        self.rnafm_prepend_bos = bool(getattr(alphabet, 'prepend_bos', True))
        self.rnafm_append_eos = bool(getattr(alphabet, 'append_eos', True))

    def _seq_idx_to_rnafm_idx(self, S):
        device = S.device
        mapper = torch.tensor([self.rnafm_A_idx, self.rnafm_U_idx, self.rnafm_C_idx, self.rnafm_G_idx, self.rnafm_condition_idx], dtype=torch.long, device=device)
        return mapper[S.long().clamp(0, 4)]

    def _add_special_tokens_varlen(self, seq_tokens, lengths):
        B, Lmax = seq_tokens.shape
        device = seq_tokens.device
        extra = int(self.rnafm_prepend_bos) + int(self.rnafm_append_eos)
        tokens = torch.full((B, Lmax + extra), self.rnafm_pad_idx, dtype=torch.long, device=device)
        seq_start = 1 if self.rnafm_prepend_bos else 0
        if self.rnafm_prepend_bos:
            tokens[:, 0] = self.rnafm_cls_idx
        tokens[:, seq_start:seq_start + Lmax] = seq_tokens
        if self.rnafm_append_eos:
            bidx = torch.arange(B, device=device)
            eos_pos = seq_start + lengths.long().clamp(0, Lmax)
            tokens[bidx, eos_pos] = self.rnafm_eos_idx
        return (tokens, seq_start)

    def _make_rnafm_tokens_and_gather_pos(self, S_t, residue_idx=None):
        B, L = S_t.shape
        device = S_t.device
        base_tokens = self._seq_idx_to_rnafm_idx(S_t)
        if residue_idx is None or not self.rnafm_use_residue_idx_gaps:
            lengths = torch.full((B,), L, dtype=torch.long, device=device)
            gather_pos = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
            tokens, seq_start = self._add_special_tokens_varlen(base_tokens, lengths)
            return (tokens, seq_start, gather_pos, L)
        res = residue_idx.long().to(device)
        start = res.min(dim=1).values
        rel_pos = res - start[:, None]
        lengths = rel_pos.max(dim=1).values + 1
        Lfull = int(lengths.max().item())
        if Lfull > self.rnafm_max_full_len:
            if self.rnafm_gap_fallback_no_gaps:
                lengths = torch.full((B,), L, dtype=torch.long, device=device)
                gather_pos = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
                tokens, seq_start = self._add_special_tokens_varlen(base_tokens, lengths)
                return (tokens, seq_start, gather_pos, L)
            raise RuntimeError(f'RNA-FM full length after filling gaps is {Lfull}, larger than rnafm_max_full_len={self.rnafm_max_full_len}. Increase the limit, or set --rnafm_gap_fallback_no_gaps.')
        full_tokens = torch.full((B, Lfull), self.rnafm_pad_idx, dtype=torch.long, device=device)
        for b in range(B):
            lb = int(lengths[b].item())
            full_tokens[b, :lb] = self.rnafm_gap_idx
            full_tokens[b, rel_pos[b]] = base_tokens[b]
        tokens, seq_start = self._add_special_tokens_varlen(full_tokens, lengths)
        return (tokens, seq_start, rel_pos, Lfull)

    def _run_rnafm_project(self, tokens, seq_start, Lfull):
        if self.freeze_rnafm:
            self.rnafm_model.eval()
        use_amp = self.rnafm_use_amp and tokens.is_cuda
        grad_context = torch.no_grad() if self.freeze_rnafm else contextlib.nullcontext()
        with grad_context:
            with torch.cuda.amp.autocast(enabled=use_amp):
                results = self.rnafm_model(tokens, repr_layers=[self.rnafm_layer])
                rep = results['representations'][self.rnafm_layer]
                rep = rep[:, seq_start:seq_start + Lfull, :]
        rep = rep.float()
        return self.rnafm_proj(rep)

    def _compute_rnafm_node_features(self, S_t, residue_idx=None):
        tokens, seq_start, gather_pos, Lfull = self._make_rnafm_tokens_and_gather_pos(S_t, residue_idx)
        h_full = self._run_rnafm_project(tokens, seq_start, Lfull)
        gather_idx = gather_pos.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        return torch.gather(h_full, 1, gather_idx)

    def kappa(self, t, scheduler=None):
        if scheduler is None:
            scheduler = self.dfm_scheduler
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=torch.float32)
        t = t.float().clamp(0.0, 1.0)
        name = str(scheduler).lower()
        if name == 'linear':
            k = t
        elif name == 'cosine':
            k = 1.0 - torch.cos(0.5 * math.pi * t)
        elif name == 'cubic':
            k = 1.0 - (1.0 - t) ** 3
        elif name in ('cubic_slow', 'slow_cubic', 't3'):
            k = t ** 3
        elif name == 'sqrt':
            k = torch.sqrt(t.clamp_min(0.0))
        else:
            raise ValueError(f'Unknown dfm_scheduler: {scheduler}')
        return k.clamp(0.0, 1.0)

    def _select_train_scheduler(self):
        if not self.dfm_random_train_scheduler:
            return self.dfm_scheduler
        idx = int(torch.randint(0, len(self.dfm_train_schedulers), (1,)).item())
        return self.dfm_train_schedulers[idx]

    def _sample_train_t(self, B, device):
        lo, hi = (self.dfm_train_min_t, self.dfm_train_max_t)
        return lo + (hi - lo) * torch.rand(B, device=device)

    def _time_condition_scalar(self, t, scheduler=None):
        mode = self.dfm_time_condition
        if mode in ['kappa', 'k']:
            return self.kappa(t, scheduler=scheduler)
        if mode in ['one_minus_kappa', '1-kappa']:
            return 1.0 - self.kappa(t, scheduler=scheduler)
        return t

    def _make_noisy_tokens(self, S, t, chain_M=None, scheduler=None):
        device = S.device
        B, L = S.shape
        if t.ndim == 0:
            t = t.expand(B)
        k = self.kappa(t, scheduler=scheduler).view(B, 1)
        valid = (S >= 0) & (S < self.num_letters)
        if chain_M is None:
            chain_M = torch.ones((B, L), device=device, dtype=torch.float32)
        else:
            chain_M = chain_M.float().to(device)
        design_mask = (chain_M > 0.5) & valid
        fixed_mask = (chain_M <= 0.5) & valid
        reveal = (torch.rand((B, L), device=device) < k) & design_mask
        S_t = torch.full_like(S, self.mask_token)
        S_t = torch.where(reveal, S, S_t)
        S_t = torch.where(fixed_mask, S, S_t)
        return (S_t.long(), reveal, design_mask)

    def _encode_structure(self, X, residue_idx):
        E, E_idx = self.features(X, residue_idx)
        h_E = self.W_e(E)
        h_V = torch.zeros_like(h_E[:, :, 0, :])
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx)
        return (h_V, h_E, E_idx)

    def _ss_weights(self, h_E, E_idx, real_ss=None, training_teacher=True):
        if not self.use_ss:
            return (None, None, None)
        B, L = (h_E.shape[0], h_E.shape[1])
        edge_ss_logits = self.ss_edge_head(h_E).squeeze(-1)
        if real_ss is not None and real_ss.ndim == 3 and (real_ss.size(1) == L) and (real_ss.size(2) == L):
            edge_ss_label = torch.gather(real_ss, 2, E_idx).float()
        else:
            edge_ss_label = None
        pred_ss_prob = torch.sigmoid(edge_ss_logits.detach())
        if self.training and training_teacher and (edge_ss_label is not None):
            if torch.rand(1, device=h_E.device).item() > 0.5:
                valid_mask = (edge_ss_label >= 0).float()
                ss_weights = edge_ss_label.clamp(0, 1) * valid_mask + pred_ss_prob * (1.0 - valid_mask)
            else:
                ss_weights = pred_ss_prob
        else:
            ss_weights = pred_ss_prob
        return (edge_ss_logits, edge_ss_label, ss_weights.unsqueeze(-1))

    def _decode_logits_from_encoded(self, h_V_enc, h_E, E_idx, S_t, t, ss_weights=None, chain_M=None, residue_idx=None, scheduler=None):
        B, L = S_t.shape
        if t.ndim == 0:
            t = t.expand(B)
        t_for_emb = self._time_condition_scalar(t.to(S_t.device), scheduler=scheduler)
        t_emb = self.W_t(t_for_emb)
        h_V = h_V_enc + self.node_time_norm(t_emb).unsqueeze(1)
        h_S = self._compute_rnafm_node_features(S_t.clamp(0, self.mask_token), residue_idx=residue_idx)
        h_S = h_S + self.seq_time_norm(t_emb).unsqueeze(1)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)
        for layer in self.decoder_layers:
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_V = layer(h_V, h_ESV, ss_weights=ss_weights)
        h_V = self.norm_out(h_V)
        return self.W_out(h_V)

    def forward(self, X, S, real_ss=None, residue_idx=None, chain_M=None, randn=None, t=None, S_t=None, return_noisy=False, scheduler=None):
        device = X.device
        B, L = S.shape
        if residue_idx is None:
            raise ValueError('residue_idx must be provided')
        if chain_M is None:
            chain_M = torch.ones((B, L), device=device, dtype=torch.float32)
        else:
            chain_M = chain_M.float().to(device)
        S = S.long().to(device)
        residue_idx = residue_idx.long().to(device)
        train_scheduler = scheduler if scheduler is not None else self._select_train_scheduler()
        if t is None:
            t = self._sample_train_t(B, device)
        else:
            t = t.to(device).float()
            if t.ndim == 0:
                t = t.expand(B)
        if S_t is None:
            S_t, reveal_mask, design_mask = self._make_noisy_tokens(S, t, chain_M=chain_M, scheduler=train_scheduler)
        else:
            S_t = S_t.long().to(device)
            reveal_mask = (S_t == S) & (S < self.num_letters) & (chain_M > 0.5)
            design_mask = (S < self.num_letters) & (chain_M > 0.5)
        h_V, h_E, E_idx = self._encode_structure(X, residue_idx)
        edge_ss_logits, edge_ss_label, ss_weights = self._ss_weights(h_E, E_idx, real_ss=real_ss, training_teacher=True)
        logits = self._decode_logits_from_encoded(h_V, h_E, E_idx, S_t, t, ss_weights=ss_weights, chain_M=chain_M, residue_idx=residue_idx, scheduler=train_scheduler)
        if return_noisy:
            aux = {'S_t': S_t, 't': t, 'reveal_mask': reveal_mask, 'design_mask': design_mask, 'kappa_t': self.kappa(t, scheduler=train_scheduler), 'dfm_scheduler': train_scheduler}
            return (logits, edge_ss_logits, edge_ss_label, aux)
        return (logits, edge_ss_logits, edge_ss_label)

    @torch.no_grad()
    def sample(self, X, residue_idx, temperature=0.1, decoding_order=None, randn=None, S_init=None, chain_M=None, num_steps=None, scheduler=None, greedy=False, keep_fixed=True, return_trace=False, corrector_remask=False, remask_rate=0.05, remask_schedule='middle', remask_low_conf=True, remask_min_step=1, remask_max_step=None, temperature_anneal=False, temperature_min=0.05, temperature_max=0.5):
        device = X.device
        B, L = (X.shape[0], X.shape[1])
        residue_idx = residue_idx.long().to(device)
        if num_steps is None:
            num_steps = self.dfm_num_steps
        num_steps = int(num_steps)
        if scheduler is None:
            scheduler = self.dfm_scheduler
        if remask_max_step is None:
            remask_max_step = num_steps - 1
        if chain_M is None:
            chain_M = torch.ones((B, L), device=device, dtype=torch.float32)
        else:
            chain_M = chain_M.float().to(device)
        design_mask = chain_M > 0.5
        if S_init is None:
            S = torch.full((B, L), self.mask_token, dtype=torch.long, device=device)
        else:
            S_init = S_init.long().to(device)
            S = torch.where(design_mask, torch.full_like(S_init, self.mask_token), S_init.clamp(0, self.mask_token))
        h_V, h_E, E_idx = self._encode_structure(X, residue_idx)
        edge_ss_logits, _, ss_weights = self._ss_weights(h_E, E_idx, real_ss=None, training_teacher=False)
        pred_ss_prob = torch.sigmoid(edge_ss_logits) if edge_ss_logits is not None else None
        final_probs = None
        final_probs_noT = None
        trace = []
        eps = 1e-08

        def _remask_strength(k0):
            k0 = k0.clamp(0.0, 1.0)
            name = str(remask_schedule).lower()
            if name == 'constant':
                sched = torch.ones_like(k0)
            elif name == 'early':
                sched = 1.0 - k0
            elif name == 'middle':
                sched = torch.sin(math.pi * k0).clamp_min(0.0)
            elif name == 'late':
                sched = k0
            else:
                raise ValueError(f'Unknown remask_schedule: {remask_schedule}')
            return (float(remask_rate) * sched).clamp(0.0, 0.95)
        for step in range(num_steps):
            t0_scalar = step / float(num_steps)
            t1_scalar = (step + 1) / float(num_steps)
            t = torch.full((B,), t0_scalar, device=device, dtype=torch.float32)
            k0 = self.kappa(t, scheduler=scheduler)
            k1 = self.kappa(torch.full_like(t, t1_scalar), scheduler=scheduler)
            r = ((k1 - k0) / (1.0 - k0).clamp_min(eps)).clamp(0.0, 1.0)
            r_view = r.view(B, 1, 1)
            logits = self._decode_logits_from_encoded(h_V, h_E, E_idx, S, t, ss_weights=ss_weights, chain_M=chain_M, residue_idx=residue_idx, scheduler=scheduler)
            probs_noT = F.softmax(logits.float(), dim=-1)
            if temperature_anneal:
                temp_t = float(temperature_min) + (float(temperature_max) - float(temperature_min)) * (1.0 - k0)
                temp_view = temp_t.view(B, 1, 1).clamp_min(1e-06)
                probs = F.softmax(logits.float() / temp_view, dim=-1)
            else:
                probs = F.softmax(logits.float() / max(float(temperature), 1e-06), dim=-1)
            p1_ext = torch.zeros((B, L, self.vocab), dtype=probs.dtype, device=device)
            p1_ext[..., :self.num_letters] = probs
            current = F.one_hot(S.clamp(0, self.mask_token), num_classes=self.vocab).float()
            pmf_next = (1.0 - r_view) * current + r_view * p1_ext
            do_remask = bool(corrector_remask) and step >= int(remask_min_step) and (step < int(remask_max_step))
            if do_remask and float(remask_rate) > 0.0:
                b = _remask_strength(k0)
                mask_pmf = torch.zeros_like(pmf_next)
                mask_pmf[..., self.mask_token] = 1.0
                if remask_low_conf:
                    conf = probs_noT.max(dim=-1).values.detach()
                    uncertainty = ((1.0 - conf) / 0.75).clamp(0.0, 1.0)
                    b_pos = b.view(B, 1) * uncertainty
                else:
                    b_pos = b.view(B, 1).expand(B, L)
                b_pos = torch.where(design_mask, b_pos, torch.zeros_like(b_pos))
                pmf_next = (1.0 - b_pos.unsqueeze(-1)) * pmf_next + b_pos.unsqueeze(-1) * mask_pmf
            if keep_fixed and S_init is not None:
                fixed_mask = ~design_mask
                fixed_onehot = F.one_hot(S_init.clamp(0, self.mask_token), num_classes=self.vocab).float()
                pmf_next = torch.where(fixed_mask.unsqueeze(-1), fixed_onehot, pmf_next)
            pmf_next = pmf_next.clamp_min(0.0)
            pmf_next = pmf_next / pmf_next.sum(dim=-1, keepdim=True).clamp_min(eps)
            if greedy:
                S_new = pmf_next.argmax(dim=-1)
            else:
                S_new = torch.multinomial(pmf_next.reshape(-1, self.vocab), num_samples=1).reshape(B, L)
            if step == num_steps - 1:
                base_probs = probs
                if greedy:
                    base_sample = base_probs.argmax(dim=-1)
                else:
                    base_sample = torch.multinomial(base_probs.reshape(-1, self.num_letters), num_samples=1).reshape(B, L)
                need_fill = design_mask & (S_new == self.mask_token)
                S_new = torch.where(need_fill, base_sample, S_new)
                if keep_fixed and S_init is not None:
                    fixed_mask = ~design_mask
                    S_new = torch.where(fixed_mask, S_init.clamp(0, self.mask_token), S_new)
            S = S_new.long()
            final_probs = probs
            final_probs_noT = probs_noT
            if return_trace:
                trace.append(S.detach().cpu())
        if final_probs is None:
            t = torch.zeros((B,), device=device, dtype=torch.float32)
            logits = self._decode_logits_from_encoded(h_V, h_E, E_idx, S, t, ss_weights=ss_weights, chain_M=chain_M, residue_idx=residue_idx, scheduler=scheduler)
            final_probs_noT = F.softmax(logits.float(), dim=-1)
            final_probs = F.softmax(logits.float() / max(float(temperature), 1e-06), dim=-1)
            S = final_probs.argmax(dim=-1)
        final_step_mean_probability = _get_final_step_mean_probability(final_probs_noT, S.clamp(0, 3))
        output = {'S': S, 'probs': final_probs, 'probs_withoutT': final_probs_noT, 'final_step_mean_probability': final_step_mean_probability, 'pred_SS_prob': pred_ss_prob, 'dfm_num_steps': num_steps, 'dfm_scheduler': scheduler, 'corrector_remask': bool(corrector_remask), 'remask_rate': float(remask_rate), 'remask_schedule': str(remask_schedule), 'remask_low_conf': bool(remask_low_conf), 'temperature_anneal': bool(temperature_anneal), 'E_idx': E_idx}
        if return_trace:
            output['trace'] = trace
        return (output, S)

    @torch.no_grad()
    def sample_random_order_onebyone(self, X, residue_idx, temperature=0.1, decoding_order=None, randn=None, S_init=None, chain_M=None, scheduler=None, greedy=False, keep_fixed=True, temperature_anneal=False, temperature_min=0.05, temperature_max=0.5, return_trace=False):
        device = X.device
        B, L = (X.shape[0], X.shape[1])
        residue_idx = residue_idx.long().to(device)
        if scheduler is None:
            scheduler = self.dfm_scheduler
        if chain_M is None:
            chain_M = torch.ones((B, L), device=device, dtype=torch.float32)
        else:
            chain_M = chain_M.float().to(device)
        design_mask = chain_M > 0.5
        if S_init is None:
            S = torch.full((B, L), self.mask_token, dtype=torch.long, device=device)
        else:
            S_init = S_init.long().to(device)
            S = torch.where(design_mask, torch.full_like(S_init, self.mask_token), S_init.clamp(0, self.mask_token))
        if keep_fixed and S_init is not None:
            fixed_mask = ~design_mask
            S = torch.where(fixed_mask, S_init.clamp(0, self.mask_token), S)
        if decoding_order is None:
            if randn is None:
                randn = torch.randn((B, L), device=device)
            order_score = randn.masked_fill(~design_mask, float('inf'))
            decoding_order = torch.argsort(order_score, dim=1)
        else:
            decoding_order = decoding_order.long().to(device)
        num_design = design_mask.sum(dim=1).long()
        max_steps = int(num_design.max().item())
        h_V, h_E, E_idx = self._encode_structure(X, residue_idx)
        edge_ss_logits, _, ss_weights = self._ss_weights(h_E, E_idx, real_ss=None, training_teacher=False)
        pred_ss_prob = torch.sigmoid(edge_ss_logits) if edge_ss_logits is not None else None
        trace = []
        final_probs = None
        final_probs_noT = None
        eps = 1e-08
        for step in range(max_steps):
            frac = step / max(float(max_steps), 1.0)
            t = torch.full((B,), frac, device=device, dtype=torch.float32)
            logits = self._decode_logits_from_encoded(h_V, h_E, E_idx, S, t, ss_weights=ss_weights, chain_M=chain_M, residue_idx=residue_idx, scheduler=scheduler)
            probs_noT = torch.softmax(logits.float(), dim=-1)
            if temperature_anneal:
                k0 = self.kappa(t, scheduler=scheduler)
                temp_t = float(temperature_min) + (float(temperature_max) - float(temperature_min)) * (1.0 - k0)
                temp_view = temp_t.view(B, 1, 1).clamp_min(eps)
                probs = torch.softmax(logits.float() / temp_view, dim=-1)
            else:
                probs = torch.softmax(logits.float() / max(float(temperature), eps), dim=-1)
            for b in range(B):
                if step >= int(num_design[b].item()):
                    continue
                pos = decoding_order[b, step]
                if not bool(design_mask[b, pos].item()):
                    continue
                pos_probs = probs[b, pos, :]
                if greedy:
                    sampled_base = torch.argmax(pos_probs, dim=-1)
                else:
                    sampled_base = torch.multinomial(pos_probs, num_samples=1).squeeze(0)
                S[b, pos] = sampled_base.long()
            if keep_fixed and S_init is not None:
                fixed_mask = ~design_mask
                S = torch.where(fixed_mask, S_init.clamp(0, self.mask_token), S)
            final_probs = probs
            final_probs_noT = probs_noT
            if return_trace:
                trace.append(S.detach().cpu())
        need_fill = design_mask & (S == self.mask_token)
        if need_fill.any():
            t = torch.ones((B,), device=device, dtype=torch.float32)
            logits = self._decode_logits_from_encoded(h_V, h_E, E_idx, S, t, ss_weights=ss_weights, chain_M=chain_M, residue_idx=residue_idx, scheduler=scheduler)
            probs_noT = torch.softmax(logits.float(), dim=-1)
            probs = torch.softmax(logits.float() / max(float(temperature), eps), dim=-1)
            if greedy:
                fill_base = probs.argmax(dim=-1)
            else:
                fill_base = torch.multinomial(probs.reshape(-1, self.num_letters), num_samples=1).reshape(B, L)
            S = torch.where(need_fill, fill_base.long(), S)
            final_probs = probs
            final_probs_noT = probs_noT
        final_step_mean_probability = _get_final_step_mean_probability(final_probs_noT, S.clamp(0, 3))
        output = {'S': S, 'probs': final_probs, 'probs_withoutT': final_probs_noT, 'final_step_mean_probability': final_step_mean_probability, 'pred_SS_prob': pred_ss_prob, 'dfm_scheduler': scheduler, 'random_order_onebyone': True, 'decoding_order': decoding_order.detach().cpu(), 'temperature_anneal': bool(temperature_anneal), 'E_idx': E_idx}
        if return_trace:
            output['trace'] = trace
        return (output, S)
class RNAMPNN_NoSS(RNAMPNN_DFM):

    def __init__(self, args, node_features=256, edge_features=256, hidden_dim=256, augment_eps=0.0, dropout=0.2):
        super().__init__(args, node_features=node_features, edge_features=edge_features, hidden_dim=hidden_dim, augment_eps=augment_eps, dropout=dropout, use_ss=False)


RNAMPNN = RNAMPNN_DFM
