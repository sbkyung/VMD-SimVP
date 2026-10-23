import numpy as np
from scipy.io import savemat
import torch
from tqdm import tqdm

if __package__:
    from .data_loader import load_data, make_loader
    from .utils import parse_args, setup_device, read_checkpoint, build_model
else:
    from data_loader import load_data, make_loader
    from utils import parse_args, setup_device, read_checkpoint, build_model


def main():
    args = parse_args(training=False)
    device = setup_device(args)
    checkpoint = read_checkpoint(args)
    model = build_model(args, device, checkpoint)
    del checkpoint
    loader = make_loader(load_data(args), args, "test")
    destination = args.result_dir / "test"
    destination.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Test"):
            prediction = model(batch["input"].to(device), batch["ecmwf"].to(device)).cpu().numpy()
            for row, sample_id in enumerate(batch["sample_id"].tolist()):
                values = {
                    "input": batch["input"][row, :, 0].numpy(),
                    "target": batch["label"][row, :, 0].numpy(),
                    "output": prediction[row, :, 0],
                    "sample_id": sample_id,
                    "target_start_index_0based": int(batch["target_start_index"][row]),
                }
                filename = destination / f"PRED_TE_{sample_id}.{args.save_format}"
                if args.save_format == "npz":
                    np.savez_compressed(str(filename), **values)
                else:
                    savemat(str(filename), values)
    print(f"Saved {len(loader.dataset)} samples to {destination}")


if __name__ == "__main__":
    main()
