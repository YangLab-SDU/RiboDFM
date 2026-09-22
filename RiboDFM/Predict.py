import argparse
import random
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from RiboDFMmodel import RNAMPNN as RiboDFM
from RiboDFMmodel import RNAMPNN_NoSS as RiboDFMNoSS
from RiboDFMsampler import RiboDFMSampler
from pdb_utils import parse_pdb, write_fasta


ALPHABET = "AUCG"
SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", required=True)
    parser.add_argument("--checkpoint", default=str("./params/RiboDFM_model.pt"))
    parser.add_argument("--checkpoint-no-ss", default=str("./params/RiboDFM_model_woSShead.pt"))
    parser.add_argument("--rnafm-checkpoint", default=None)
    parser.add_argument("--output", default="./output")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--num-steps", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--scheduler", choices=["linear", "cosine", "cubic", "cubic_slow", "slow_cubic", "t3", "sqrt"], default="cosine")
    parser.add_argument("--chain", default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def checkpoint_state(path):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    return {key.removeprefix("module."): value for key, value in state.items()}


def layer_count(state, prefix):
    indices = []
    pattern = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.")
    for key in state:
        match = pattern.match(key)
        if match:
            indices.append(int(match.group(1)))
    if not indices:
        raise ValueError(f"Cannot infer {prefix} count from checkpoint")
    return max(indices) + 1


def model_config(state, num_steps, scheduler, rnafm_checkpoint=None):
    return SimpleNamespace(
        num_encoder_layers=layer_count(state, "encoder_layers"),
        num_decoder_layers=layer_count(state, "decoder_layers"),
        k_neighbors=128,
        dfm_num_steps=num_steps,
        dfm_scheduler=scheduler,
        dfm_train_min_t=1e-4,
        dfm_train_max_t=1.0 - 1e-4,
        dfm_random_train_scheduler=False,
        dfm_train_schedulers=scheduler,
        dfm_time_condition="kappa",
        dfm_time_fourier_dim=64,
        use_rnafm=True,
        rnafm_layer=12,
        rnafm_dim=640,
        freeze_rnafm=True,
        rnafm_use_amp=True,
        rnafm_unknown_mode="n",
        rnafm_use_residue_idx_gaps=True,
        rnafm_max_full_len=1024,
        rnafm_gap_fallback_no_gaps=False,
        rnafm_checkpoint=rnafm_checkpoint,
        ss_message_mode="gate",
        ss_scale_init=0.0,
        ss_max_scale=2.0,
        ss_exp_clip=2.0,
    )


def load_models(checkpoint, checkpoint_no_ss, device, num_steps, scheduler, rnafm_checkpoint=None):
    if rnafm_checkpoint is not None:
        rnafm_checkpoint = str(Path(rnafm_checkpoint).expanduser().resolve())
        if not Path(rnafm_checkpoint).is_file():
            raise FileNotFoundError(f"RNA-FM checkpoint not found: {rnafm_checkpoint}")
    state = checkpoint_state(checkpoint)
    state_no_ss = checkpoint_state(checkpoint_no_ss)
    model = RiboDFM(model_config(state, num_steps, scheduler, rnafm_checkpoint)).to(device)
    model_no_ss = RiboDFMNoSS(model_config(state_no_ss, num_steps, scheduler, rnafm_checkpoint)).to(device)
    model.load_state_dict(state, strict=True)
    model_no_ss.load_state_dict(state_no_ss, strict=True)
    model.eval()
    model_no_ss.eval()
    return model, model_no_ss


def main():
    args = parse_args()
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")
    if args.num_steps <= 0:
        raise ValueError("num-steps must be positive")
    if args.num_samples <= 0:
        raise ValueError("num-samples must be positive")
    device = torch.device(args.device)
    selected_chains = args.chain.split(",") if args.chain else None
    structure = parse_pdb(args.pdb, selected_chains=selected_chains)
    coords = torch.from_numpy(structure.coords).unsqueeze(0).to(device)
    residue_idx = torch.from_numpy(structure.residue_idx).unsqueeze(0).to(device)
    model, model_no_ss = load_models(args.checkpoint, args.checkpoint_no_ss, device, args.num_steps, args.scheduler, args.rnafm_checkpoint)
    sampler = RiboDFMSampler(model, model_no_ss, name_a="RiboDFM", name_b="RiboDFM_noSS", weight_a=0.5, weight_b=0.5, ensemble_mode="logit")
    predictions = []
    confidences = []
    for sample_index in range(args.num_samples):
        seed_all(args.seed + sample_index)
        output, sequence = sampler.sample(coords, residue_idx, temperature=args.temperature, num_steps=args.num_steps, scheduler=args.scheduler)
        predictions.append("".join(ALPHABET[index] for index in sequence[0].detach().cpu().tolist()))
        confidences.append(float(output["confidence"][0].detach().cpu().item()))
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / Path(args.pdb).with_suffix(".fasta").name
    write_fasta(output_path, structure, predictions, confidences=confidences)
    print(output_path)


if __name__ == "__main__":
    main()
