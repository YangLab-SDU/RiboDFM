import math, inspect
import torch
from torch.nn import functional as F

def trajectory_confidence(prob_trace, sequence, valid_mask, tail_steps=10):
    if len(prob_trace) < tail_steps:
        return torch.full((sequence.shape[0],), float('nan'))
    probabilities = torch.stack(prob_trace[-tail_steps:], dim=0)
    sequence = sequence.detach().cpu().long()
    valid_mask = valid_mask.detach().cpu().bool()
    selected = torch.gather(probabilities, -1, sequence.unsqueeze(0).unsqueeze(-1).expand(tail_steps, -1, -1, 1)).squeeze(-1)
    scores = []
    for batch_index in range(sequence.shape[0]):
        values = selected[:, batch_index, valid_mask[batch_index]].clamp_min(1e-08)
        scores.append(torch.exp(torch.log(values).mean()))
    confidence = torch.stack(scores)
    return confidence

class RiboDFMSampler:

    def __init__(self, model_a, model_b, name_a='DFM_RNAFM', name_b='DFM_RNAFM_woSS', weight_a=0.5, weight_b=0.5, ensemble_mode='logit'):
        self.model_a = model_a
        self.model_b = model_b
        self.name_a = name_a
        self.name_b = name_b
        ws = float(weight_a) + float(weight_b)
        self.weight_a = float(weight_a) / ws
        self.weight_b = float(weight_b) / ws
        self.ensemble_mode = str(ensemble_mode).lower()
        self.model_a.eval()
        self.model_b.eval()
        self.num_letters = int(getattr(model_a, 'num_letters', 4))
        self.mask_token = int(getattr(model_a, 'mask_token', 4))
        self.vocab = int(getattr(model_a, 'vocab', 5))

    def eval(self):
        self.model_a.eval()
        self.model_b.eval()
        return self

    def _kappa(self, t, scheduler='cosine'):
        if hasattr(self.model_a, 'kappa'):
            return self.model_a.kappa(t, scheduler=scheduler)
        name = str(scheduler).lower()
        if name == 'linear':
            return t
        if name == 'cosine':
            return 1.0 - torch.cos(0.5 * math.pi * t)
        if name == 'cubic':
            return 1.0 - (1.0 - t).pow(3)
        if name in ('cubic_slow', 'slow_cubic', 't3'):
            return t.pow(3)
        if name == 'sqrt':
            return torch.sqrt(t.clamp_min(0.0))
        raise ValueError(f'Unknown scheduler: {scheduler}')

    def _inv_kappa(self, k, scheduler='cosine'):
        k = k.float().clamp(0.0, 1.0)
        name = str(scheduler).lower()
        if name == 'linear':
            t = k
        elif name == 'cosine':
            t = 2.0 / math.pi * torch.acos((1.0 - k).clamp(-1.0, 1.0))
        elif name == 'cubic':
            t = 1.0 - (1.0 - k).clamp_min(0.0).pow(1.0 / 3.0)
        elif name in ('cubic_slow', 'slow_cubic', 't3'):
            t = k.clamp_min(0.0).pow(1.0 / 3.0)
        elif name == 'sqrt':
            t = k.pow(2)
        else:
            raise ValueError(f'Unknown scheduler: {scheduler}')
        return t.clamp(0.0, 1.0)

    def _encode_model(self, model, X, residue_idx):
        h_V, h_E, E_idx = model._encode_structure(X, residue_idx)
        ss_weights = None
        pred_ss_prob = None
        edge_ss_logits = None
        if hasattr(model, '_ss_weights'):
            try:
                ss_out = model._ss_weights(h_E, E_idx, real_ss=None, training_teacher=False)
            except TypeError:
                ss_out = model._ss_weights(h_E, E_idx)
            if isinstance(ss_out, (tuple, list)):
                edge_ss_logits = ss_out[0]
                if len(ss_out) >= 3:
                    ss_weights = ss_out[2]
            else:
                edge_ss_logits = ss_out
            if edge_ss_logits is not None:
                pred_ss_prob = torch.sigmoid(edge_ss_logits)
        return {'h_V': h_V, 'h_E': h_E, 'E_idx': E_idx, 'ss_weights': ss_weights, 'pred_ss_prob': pred_ss_prob}

    def _decode_logits(self, model, cache, S, t, chain_M, residue_idx, scheduler):
        fn = model._decode_logits_from_encoded
        params = inspect.signature(fn).parameters
        kwargs = {}
        if 'ss_weights' in params and cache.get('ss_weights', None) is not None:
            kwargs['ss_weights'] = cache['ss_weights']
        if 'chain_M' in params:
            kwargs['chain_M'] = chain_M
        if 'residue_idx' in params:
            kwargs['residue_idx'] = residue_idx
        if 'scheduler' in params:
            kwargs['scheduler'] = scheduler
        logits = fn(cache['h_V'], cache['h_E'], cache['E_idx'], S, t, **kwargs)
        return logits

    def _temperature_softmax(self, logits, temperature, temperature_anneal, temperature_min, temperature_max, k0, eps=1e-08):
        probs_noT = F.softmax(logits.float(), dim=-1)
        if temperature_anneal:
            temp_t = float(temperature_min) + (float(temperature_max) - float(temperature_min)) * (1.0 - k0)
            temp_view = temp_t.view(-1, 1, 1).clamp_min(eps)
            probs = F.softmax(logits.float() / temp_view, dim=-1)
        else:
            probs = F.softmax(logits.float() / max(float(temperature), eps), dim=-1)
        return (probs_noT, probs)

    def _fuse(self, logits_a, logits_b, temperature, temperature_anneal, temperature_min, temperature_max, k0):
        eps = 1e-08
        wa, wb = (self.weight_a, self.weight_b)
        mode = self.ensemble_mode
        if mode == 'logit':
            logits_fused = wa * logits_a.float() + wb * logits_b.float()
            probs_noT, probs = self._temperature_softmax(logits_fused, temperature=temperature, temperature_anneal=temperature_anneal, temperature_min=temperature_min, temperature_max=temperature_max, k0=k0, eps=eps)
            return (probs_noT, probs)
        probs_noT_a, probs_a = self._temperature_softmax(logits_a, temperature=temperature, temperature_anneal=temperature_anneal, temperature_min=temperature_min, temperature_max=temperature_max, k0=k0, eps=eps)
        probs_noT_b, probs_b = self._temperature_softmax(logits_b, temperature=temperature, temperature_anneal=temperature_anneal, temperature_min=temperature_min, temperature_max=temperature_max, k0=k0, eps=eps)
        if mode == 'prob':
            probs_noT = wa * probs_noT_a + wb * probs_noT_b
            probs = wa * probs_a + wb * probs_b
        elif mode == 'entropy':
            ent_a = -(probs_noT_a * torch.log(probs_noT_a + eps)).sum(dim=-1, keepdim=True)
            ent_b = -(probs_noT_b * torch.log(probs_noT_b + eps)).sum(dim=-1, keepdim=True)
            dyn_a = wa / (ent_a + 0.0001)
            dyn_b = wb / (ent_b + 0.0001)
            dyn_sum = dyn_a + dyn_b
            alpha_a = dyn_a / dyn_sum
            alpha_b = dyn_b / dyn_sum
            probs_noT = alpha_a * probs_noT_a + alpha_b * probs_noT_b
            probs = alpha_a * probs_a + alpha_b * probs_b
        elif mode == 'poe':
            logp_noT = wa * torch.log(probs_noT_a + eps) + wb * torch.log(probs_noT_b + eps)
            logp = wa * torch.log(probs_a + eps) + wb * torch.log(probs_b + eps)
            probs_noT = F.softmax(logp_noT, dim=-1)
            probs = F.softmax(logp, dim=-1)
        else:
            raise ValueError(f'Unknown ensemble_mode: {mode}')
        probs_noT = probs_noT / probs_noT.sum(dim=-1, keepdim=True).clamp_min(eps)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)
        return (probs_noT, probs)

    def _merge_pred_ss_prob(self, pred_a, pred_b):
        if pred_a is not None and pred_b is not None:
            if pred_a.shape == pred_b.shape:
                return 0.5 * pred_a + 0.5 * pred_b
            return pred_a
        if pred_a is not None:
            return pred_a
        if pred_b is not None:
            return pred_b
        return None

    @torch.no_grad()
    def sample(self, X, residue_idx, temperature=0.1, decoding_order=None, randn=None, S_init=None, chain_M=None, num_steps=None, scheduler=None, greedy=False, keep_fixed=True, return_trace=False, temperature_anneal=False, temperature_min=0.05, temperature_max=0.5):
        self.eval()
        device = X.device
        B, L = (X.shape[0], X.shape[1])
        residue_idx = residue_idx.long().to(device)
        if num_steps is None:
            num_steps = int(getattr(self.model_a, 'dfm_num_steps', 50))
        num_steps = int(num_steps)
        if num_steps <= 0:
            raise ValueError(f'num_steps must be positive, got {num_steps}')
        if scheduler is None:
            scheduler = getattr(self.model_a, 'dfm_scheduler', 'cosine')
        if chain_M is None:
            chain_M = torch.ones((B, L), device=device, dtype=torch.float32)
        else:
            chain_M = chain_M.float().to(device)
        design_mask = chain_M > 0.5
        if S_init is None:
            S_init_full = torch.full((B, L), self.mask_token, dtype=torch.long, device=device)
        else:
            S_init_full = S_init.long().to(device).clamp(0, self.mask_token)
        S = torch.where(design_mask, torch.full_like(S_init_full, self.mask_token), S_init_full)
        if keep_fixed:
            S = torch.where(~design_mask, S_init_full, S)
        cache_a = self._encode_model(self.model_a, X, residue_idx)
        cache_b = self._encode_model(self.model_b, X, residue_idx)
        pred_ss_prob_a = cache_a['pred_ss_prob']
        pred_ss_prob_b = cache_b['pred_ss_prob']
        pred_ss_prob_ens = self._merge_pred_ss_prob(pred_ss_prob_a, pred_ss_prob_b)
        trace = []
        final_probs = None
        final_probs_noT = None
        prob_trace = []
        eps = 1e-08
        for step in range(num_steps):
            t0_scalar = step / float(num_steps)
            t1_scalar = (step + 1) / float(num_steps)
            t0 = torch.full((B,), t0_scalar, device=device, dtype=torch.float32)
            t1 = torch.full((B,), t1_scalar, device=device, dtype=torch.float32)
            k0 = self._kappa(t0, scheduler=scheduler).float().clamp(0.0, 1.0)
            k1 = self._kappa(t1, scheduler=scheduler).float().clamp(0.0, 1.0)
            r = ((k1 - k0) / (1.0 - k0).clamp_min(eps)).clamp(0.0, 1.0)
            r_view = r.view(B, 1, 1)
            logits_a = self._decode_logits(self.model_a, cache_a, S, t0, chain_M=chain_M, residue_idx=residue_idx, scheduler=scheduler)
            logits_b = self._decode_logits(self.model_b, cache_b, S, t0, chain_M=chain_M, residue_idx=residue_idx, scheduler=scheduler)
            probs_noT, probs = self._fuse(logits_a, logits_b, temperature=temperature, temperature_anneal=temperature_anneal, temperature_min=temperature_min, temperature_max=temperature_max, k0=k0)
            p1_ext = torch.zeros((B, L, self.vocab), dtype=probs.dtype, device=device)
            p1_ext[..., :self.num_letters] = probs
            current = F.one_hot(S.clamp(0, self.mask_token), num_classes=self.vocab).float()
            pmf_next = (1.0 - r_view) * current + r_view * p1_ext
            if keep_fixed and S_init is not None:
                fixed_mask = ~design_mask
                fixed_onehot = F.one_hot(S_init_full.clamp(0, self.mask_token), num_classes=self.vocab).float()
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
                S_new = torch.where(need_fill, base_sample.long(), S_new.long())
                if keep_fixed and S_init is not None:
                    S_new = torch.where(~design_mask, S_init_full.clamp(0, self.mask_token), S_new)
            S = S_new.long()
            final_probs = probs
            final_probs_noT = probs_noT
            prob_trace.append(probs_noT.detach().cpu())
            if return_trace:
                trace.append(S.detach().cpu())
        confidence = trajectory_confidence(prob_trace, S.clamp(0, 3), design_mask, tail_steps=10)
        output = {'S': S, 'probs': final_probs, 'probs_withoutT': final_probs_noT, 'confidence': confidence.to(device), 'confidence_tail_steps': 10, 'pred_SS_prob': pred_ss_prob_ens, 'pred_SS_prob_a': pred_ss_prob_a, 'pred_SS_prob_b': pred_ss_prob_b, 'dfm_num_steps': int(num_steps), 'dfm_scheduler': scheduler, 'normal_sample': True, 'random_order_onebyone': False, 'corrector_remask': False, 'ensemble': True, 'ensemble_mode': self.ensemble_mode, 'weight_a': self.weight_a, 'weight_b': self.weight_b, 'model_a': self.name_a, 'model_b': self.name_b, 'temperature_anneal': bool(temperature_anneal)}
        if return_trace:
            output['trace'] = trace
        return (output, S)

    @torch.no_grad()
    def teacher_force_qa(self, X, residue_idx, native_sequence, scheduler='cosine', decoding_order=None, randn=None):
        self.eval()
        device = X.device
        B, L = native_sequence.shape
        native_sequence = native_sequence.long().to(device)
        residue_idx = residue_idx.long().to(device)
        valid_mask = native_sequence < self.num_letters
        sequence = torch.where(valid_mask, torch.full_like(native_sequence, self.mask_token), native_sequence)
        if decoding_order is None:
            if randn is None:
                randn = torch.rand((B, L), device=device)
            order_score = randn.masked_fill(~valid_mask, float('inf'))
            decoding_order = torch.argsort(order_score, dim=1)
        else:
            decoding_order = decoding_order.long().to(device)
        num_valid = valid_mask.sum(dim=1).long()
        max_steps = int(num_valid.max().item())
        cache_a = self._encode_model(self.model_a, X, residue_idx)
        cache_b = self._encode_model(self.model_b, X, residue_idx)
        teacher_probs = torch.full((B, L, self.num_letters), float('nan'), device=device)
        teacher_steps = torch.full((B, L), -1, dtype=torch.long, device=device)
        chain_mask = valid_mask.float()
        for step in range(max_steps):
            revealed = torch.zeros((B,), dtype=torch.float32, device=device)
            active = num_valid > 0
            revealed[active] = float(step) / num_valid[active].float()
            time = self._inv_kappa(revealed, scheduler=scheduler)
            logits_a = self._decode_logits(self.model_a, cache_a, sequence, time, chain_mask, residue_idx, scheduler)
            logits_b = self._decode_logits(self.model_b, cache_b, sequence, time, chain_mask, residue_idx, scheduler)
            probs_noT, probs = self._fuse(logits_a, logits_b, temperature=1.0, temperature_anneal=False, temperature_min=1.0, temperature_max=1.0, k0=revealed)
            batch_indices = torch.arange(B, device=device)[step < num_valid]
            positions = decoding_order[batch_indices, step]
            keep = valid_mask[batch_indices, positions]
            batch_indices = batch_indices[keep]
            positions = positions[keep]
            teacher_probs[batch_indices, positions] = probs_noT[batch_indices, positions]
            teacher_steps[batch_indices, positions] = step
            sequence[batch_indices, positions] = native_sequence[batch_indices, positions]
        return {'teacher_forced_probs_withoutT': teacher_probs, 'teacher_forced_decoding_order': decoding_order.detach().cpu(), 'teacher_forced_step': teacher_steps}
