import argparse
import configparser
import math
import random
import re
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler

if __package__:
    from .model import SimVP
else:
    from model import SimVP



class CosineAnnealingWarmupRestarts(_LRScheduler):

    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma

        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = last_epoch

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)

        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr)\
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps)\
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


def parse_args(training=True, argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--paths", type=Path, required=True, help="User-defined path configuration TXT")
    p.add_argument("--folder_name")
    p.add_argument("--epoch", type=int, default=-1)
    p.add_argument("--vmd_mode", type=int, choices=range(5))
    p.add_argument("--channel", type=int)
    p.add_argument("--in_shape", type=int, nargs=4)
    for name in ("hid_S", "hid_T", "N_S", "N_T", "groups"):
        p.add_argument("--" + name, type=int)
    p.add_argument("--ny", type=int, default=224)
    p.add_argument("--nx", type=int, default=464)
    p.add_argument("--n_steps", type=int, default=1200)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=8 if training else 1)
    p.add_argument("--num_worker", type=int, default=10)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=7777)
    p.add_argument("--save_format", choices=("npz", "mat"), default="npz")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max_lr", type=float, default=5e-2)
    p.add_argument("--min_lr", type=float, default=5e-4)
    p.add_argument("--sched", choices=("on", "off"), default="off")
    p.add_argument("--accum_steps", type=int, default=1)
    p.add_argument("--num_epoch", type=int, default=100)
    p.add_argument("--train_continue", choices=("on", "off"), default="off")
    p.add_argument("--save_every", type=int, default=25)
    p.add_argument("--eval_every", type=int, default=25)
    args = p.parse_args(argv)
    for name in ("batch_size", "accum_steps", "stride", "num_epoch", "save_every", "eval_every"):
        if getattr(args, name) < 1:
            p.error(f"--{name} must be positive")
    if args.num_worker < 0 or args.n_steps < 1003:
        p.error("num_worker must be nonnegative; n_steps must be at least 1003")
    try:
        configure_paths(args)
    except (ValueError, OSError, configparser.Error) as error:
        p.error(str(error))
    return args


def configure_paths(args):
    args.paths = args.paths.expanduser().resolve()
    config = configparser.ConfigParser(interpolation=None)
    with args.paths.open(encoding="utf-8-sig") as stream:
        config.read_file(stream)
    if args.vmd_mode is None:
        args.vmd_mode = config.getint("model", "vmd_mode", fallback=4)
    if args.vmd_mode not in range(5):
        raise ValueError("[model] vmd_mode must be 0..4")

    def path_value(section, key, required=False, mode=False):
        value = config.get(section, key, fallback="").strip()
        if not value:
            if required:
                raise ValueError(f"Define [{section}] {key} in your path TXT")
            return None
        if mode:
            try:
                value = value.format(mode=args.vmd_mode)
            except (KeyError, ValueError, IndexError) as error:
                raise ValueError(f"Invalid [{section}] {key} template") from error
        path = Path(value).expanduser()
        return (path if path.is_absolute() else args.paths.parent / path).resolve()

    args.usdm_file = path_value("data", "usdm_file", required=True)
    imf = path_value("data", "imf_template", required=args.vmd_mode != 0)
    args.imf_template = str(imf) if imf is not None else None
    args.nldas_files = tuple(path_value("data", name, required=True)
                            for name in ("precipitation_file", "temperature_file", "soil_moisture_file"))
    args.eddi_template = str(path_value("data", "eddi_template", required=True))
    args.ecmwf_template = str(path_value("data", "ecmwf_template", required=True))
    args.output_root = path_value("output", "root", required=True)
    args.folder_name = args.folder_name or config.get("output", "name", fallback="simvp").strip()
    if not args.folder_name or args.folder_name in (".", "..") or any(c in args.folder_name for c in "/\\"):
        raise ValueError("Output name must be a single nonempty directory name")
    args.model_ver = f"{args.folder_name}_mode{args.vmd_mode}"
    args.run_dir = args.output_root / args.model_ver
    args.ckpt_dir = path_value("output", "checkpoint_dir", mode=True) or args.run_dir / "checkpoints"
    args.result_dir = path_value("output", "result_dir", mode=True) or args.run_dir / "results"
    args.load_ckpt_dir = path_value("model", "checkpoint_dir", mode=True) or args.ckpt_dir
    args.checkpoint = path_value("model", "checkpoint_file", mode=True)
    for name, fields in (("imf_template", {"mode": 1}),
                         ("eddi_template", {"weeks": 2}),
                         ("ecmwf_template", {"lead": 1, "variable": "vmax"})):
        template = getattr(args, name)
        if template is None:
            continue
        try:
            template.format(**fields)
        except (KeyError, ValueError, IndexError) as error:
            raise ValueError(f"Invalid [data] {name}") from error
        if any("{" + field not in template for field in fields):
            raise ValueError(f"[data] {name} must contain placeholders: {', '.join(fields)}")


def public_config(args):
    keys = ("vmd_mode", "channel", "in_shape", "hid_S", "hid_T", "N_S", "N_T", "groups",
            "ny", "nx", "n_steps", "stride", "batch_size", "num_worker", "seed", "save_format",
            "lr", "max_lr", "min_lr", "sched", "accum_steps", "num_epoch", "save_every", "eval_every")
    return {key: getattr(args, key) for key in keys}


def setup_device(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    name = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    return torch.device(name)


def read_checkpoint(args):
    path = args.checkpoint
    if path is None:
        if args.epoch >= 0:
            path = args.load_ckpt_dir / f"model_epoch{args.epoch}.pth"
        else:
            files = [(int(m.group(1)), p) for p in args.load_ckpt_dir.glob("model_epoch*.pth")
                     for m in [re.fullmatch(r"model_epoch(\d+)\.pth", p.name)] if m]
            if not files:
                raise FileNotFoundError(f"No model_epoch*.pth in {args.load_ckpt_dir}")
            _, path = max(files)
    # Load only checkpoints from a trusted training run.
    try:
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(path), map_location="cpu")
    if "net" not in checkpoint:
        checkpoint = {"net": checkpoint}
    state = {}
    for key, value in checkpoint["net"].items():
        name = key[7:] if key.startswith("module.") else key
        if name in state:
            raise ValueError(f"Duplicate checkpoint key: {name}")
        state[name] = value
    checkpoint["net"] = state
    match = re.fullmatch(r"model_epoch(\d+)\.pth", path.name)
    checkpoint.setdefault("epoch", int(match.group(1)) if match else 0)
    print(f"Checkpoint: {path} (epoch {checkpoint['epoch']})")
    return checkpoint


def model_settings(state):
    def depth(prefix):
        return 1 + max(int(k[len(prefix):].split(".")[0]) for k in state if k.startswith(prefix))
    hidden = state["hid.enc.0.layers.0.conv.weight"].shape[0]
    middle = state["hid.enc.0.conv1.weight"].shape[0]
    grouped_in = state["hid.enc.0.layers.0.conv.weight"].shape[1]
    channel = 1 + state["enc_cond.enc.0.conv.conv.weight"].shape[1] if "enc_cond.enc.0.conv.conv.weight" in state else 1
    return dict(channel=channel, hid_S=state["enc.enc.0.conv.conv.weight"].shape[0],
                hid_T=hidden, N_S=depth("enc.enc."), N_T=depth("hid.enc."),
                groups=middle // grouped_in)


def build_model(args, device, checkpoint=None):
    inferred = model_settings(checkpoint["net"]) if checkpoint else {}
    defaults = dict(hid_S=32, hid_T=64, N_S=2, N_T=2, groups=8)
    for key, default in defaults.items():
        requested = getattr(args, key)
        saved = inferred.get(key, default)
        if checkpoint and requested is not None and requested != saved:
            raise ValueError(f"--{key}={requested} differs from checkpoint ({saved})")
        setattr(args, key, saved if requested is None else requested)
    channels = 8 if args.vmd_mode == 0 else 12
    if args.channel not in (None, channels) or inferred.get("channel", channels) != channels:
        raise ValueError(f"vmd_mode={args.vmd_mode} requires {channels} channels; check the checkpoint/mode")
    args.channel = channels
    if args.in_shape is not None:
        if args.in_shape[:2] != [12, channels]:
            raise ValueError(f"--in_shape must be 12 {channels} H W")
        args.ny, args.nx = args.in_shape[2:]
    args.in_shape = [12, channels, args.ny, args.nx]
    if args.N_S < 2 or args.N_T < 2 or args.hid_S < 2 or args.hid_S % 2 or args.hid_T < 2:
        raise ValueError("N_S/N_T must be >=2; hid_S must be positive and even; hid_T must be >=2")
    if checkpoint and "args" in checkpoint and checkpoint["args"].get("vmd_mode", args.vmd_mode) != args.vmd_mode:
        raise ValueError("Checkpoint VMD mode does not match --vmd_mode")
    model = SimVP(tuple(args.in_shape), args.hid_S, args.hid_T, args.N_S, args.N_T, 12,
                  groups=args.groups).to(device)
    if checkpoint:
        model.load_state_dict(checkpoint["net"], strict=True)
    if device.type == "cuda" and device.index is None and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    print(f"Model: channels={channels}, hid_S={args.hid_S}, hid_T={args.hid_T}, groups={args.groups}")
    return model


def build_scheduler(optimizer, args):
    if args.sched == "off":
        return None
    return CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=50, cycle_mult=1.0,
                                        max_lr=args.max_lr, min_lr=args.min_lr,
                                        warmup_steps=10, gamma=0.9)


def restore_training(checkpoint, optimizer, scheduler, args):
    optimizer.load_state_dict(checkpoint["optim"])
    if scheduler is not None:
        if checkpoint.get("sched") is None:
            raise ValueError("Cannot resume scheduled training: checkpoint has no scheduler")
        scheduler.load_state_dict(checkpoint["sched"])
    else:
        for group in optimizer.param_groups:
            group["lr"] = args.lr
    return checkpoint["epoch"]


def save_checkpoint(model, optimizer, scheduler, epoch, args):
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    bare = model.module if isinstance(model, nn.DataParallel) else model
    saved_args = public_config(args)
    torch.save(dict(net=bare.state_dict(), optim=optimizer.state_dict(),
                    sched=scheduler.state_dict() if scheduler is not None else None,
                    epoch=epoch, args=saved_args), str(args.ckpt_dir / f"model_epoch{epoch}.pth"))


def compute_loss(prediction, target, data_range):
    from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
    metric = MultiScaleStructuralSimilarityIndexMeasure(data_range=data_range).to(prediction.device)
    similarity = torch.stack([metric(prediction[:, t], target[:, t])
                              for t in range(target.shape[1])]).mean()
    weights = torch.ones_like(target)
    positive = target > 0
    weights[positive] += (torch.ceil(target[positive]) + 1).square()
    error = (prediction - target).square()
    weighted = (error * weights).mean()
    return 1 - similarity + weighted, error.mean(), similarity, weighted


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    losses = []
    for batch in loader:
        prediction = model(batch["input"].to(device), batch["ecmwf"].to(device))
        losses.append((prediction - batch["label"].to(device)).square().mean().item())
    return float(np.mean(losses))
