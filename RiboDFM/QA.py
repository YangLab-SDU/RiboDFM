import argparse
from pathlib import Path

import numpy as np
import torch

from Predict import load_models, seed_all
from RiboDFMsampler import RiboDFMSampler
from pdb_utils import ATOM_INDEX, parse_qa_pdb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", required=True)
    parser.add_argument("--checkpoint", default=str("./params/RiboDFM_model.pt"))
    parser.add_argument("--checkpoint-no-ss", default=str("./params/RiboDFM_model_woSShead.pt"))
    parser.add_argument("--rnafm-checkpoint", default=None)
    parser.add_argument("--output", default="./output")
    parser.add_argument("--scheduler", choices=["linear", "cosine", "cubic", "cubic_slow", "slow_cubic", "t3", "sqrt"], default="cosine")
    parser.add_argument("--chain", default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def calculate_qa(teacher_probabilities, sequence, coords):
    indices = np.arange(len(sequence))
    raw_scores = teacher_probabilities[indices, sequence]
    c4 = coords[:, ATOM_INDEX["C4'"]]
    distances = np.linalg.norm(c4[:, None, :] - c4[None, :, :], axis=-1)
    neighbors = np.argsort(distances, axis=1)[:, :5]
    local_scores = 100.0 * raw_scores[neighbors].mean(axis=1)
    return raw_scores, local_scores, float(local_scores.mean())


def format_score(value):
    return f"{float(value):.3f}".rstrip("0").rstrip(".")


def write_scores(path, labels, local_scores, global_score):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write(format_score(global_score) + "\n")
        for label, local_score in zip(labels, local_scores):
            residue_number = label[1]
            handle.write(f"{residue_number}\t{format_score(local_score)}\n")
    return path


def main():
    args = parse_args()
    seed_all(args.seed)
    selected_chains = args.chain.split(",") if args.chain else None
    coords, sequence, labels = parse_qa_pdb(args.pdb, selected_chains=selected_chains)
    device = torch.device(args.device)
    model, model_no_ss = load_models(args.checkpoint, args.checkpoint_no_ss, device, 32, args.scheduler, args.rnafm_checkpoint)
    sampler = RiboDFMSampler(model, model_no_ss, name_a="RiboDFM", name_b="RiboDFM_noSS", weight_a=0.5, weight_b=0.5, ensemble_mode="logit")
    coordinate_tensor = torch.from_numpy(coords).unsqueeze(0).to(device)
    sequence_tensor = torch.from_numpy(sequence).unsqueeze(0).to(device)
    residue_idx = torch.arange(1, len(sequence) + 1, dtype=torch.long, device=device).unsqueeze(0)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    randn = torch.rand((1, len(sequence)), generator=generator, device=device)
    output = sampler.teacher_force_qa(coordinate_tensor, residue_idx, sequence_tensor, scheduler=args.scheduler, randn=randn)
    probabilities = output["teacher_forced_probs_withoutT"][0].detach().cpu().numpy()
    raw_scores, local_scores, global_score = calculate_qa(probabilities, sequence, coords)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{Path(args.pdb).stem}_qa.txt"
    write_scores(output_path, labels, local_scores, global_score)
    print(f"Global QA: {global_score:.6f}")
    print(output_path)


if __name__ == "__main__":
    main()
