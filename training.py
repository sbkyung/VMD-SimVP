import json

import numpy as np
import torch
from tqdm import tqdm

if __package__:
    from .data_loader import load_data, make_loader
    from .utils import (parse_args, setup_device, read_checkpoint, build_model, build_scheduler,
                        restore_training, save_checkpoint, compute_loss, evaluate, public_config)
else:
    from data_loader import load_data, make_loader
    from utils import (parse_args, setup_device, read_checkpoint, build_model, build_scheduler,
                       restore_training, save_checkpoint, compute_loss, evaluate, public_config)


def main():
    args = parse_args(training=True)
    device = setup_device(args)
    checkpoint = read_checkpoint(args) if args.train_continue == "on" else None
    if args.checkpoint is not None and checkpoint is None:
        raise ValueError("Use --train_continue on to load [model] checkpoint_file")
    model = build_model(args, device, checkpoint)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.001)
    scheduler = build_scheduler(optimizer, args)
    start_epoch = restore_training(checkpoint, optimizer, scheduler, args) if checkpoint else 0
    del checkpoint

    data = load_data(args)
    if not np.isfinite(data.data_range) or data.data_range <= 0:
        raise ValueError("Training target must have a positive finite range for MS-SSIM")
    loader = make_loader(data, args, "train")
    test_loader = make_loader(data, args, "test")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    config = public_config(args)
    (args.run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    for epoch in range(start_epoch + 1, args.num_epoch + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        progress = tqdm(loader, desc=f"Train {epoch}/{args.num_epoch}")
        for batch_index, batch in enumerate(progress, 1):
            target = batch["label"].to(device)
            prediction = model(batch["input"].to(device), batch["ecmwf"].to(device))
            loss, _, _, _ = compute_loss(prediction, target, data.data_range)
            window_start = ((batch_index - 1) // args.accum_steps) * args.accum_steps
            samples = min(args.accum_steps * args.batch_size,
                          len(loader.dataset) - window_start * args.batch_size)
            (loss * (target.shape[0] / samples)).backward()
            if batch_index % args.accum_steps == 0 or batch_index == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(loss.item())
            progress.set_postfix(loss=float(np.mean(losses)))
        if scheduler is not None:
            scheduler.step()
        record = dict(epoch=epoch, train_loss=float(np.mean(losses)),
                      lr=optimizer.param_groups[0]["lr"])
        if epoch % args.eval_every == 0:
            record["test_mse"] = evaluate(model, test_loader, device)
        if epoch % args.save_every == 0 or epoch == args.num_epoch:
            save_checkpoint(model, optimizer, scheduler, epoch, args)
        print(record)
        with (args.run_dir / "training.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
