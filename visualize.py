"""Visualise an image flowing through a CNN, from the input through each conv stage to the output.

For one image the figure shows, left to right:
  input image + Grad-CAM  ->  feature maps of every conv stage  ->  output class probabilities

Each stage column shows the most strongly activated channels at that depth. Early stages
respond to edges and colour; deeper stages show coarser, more abstract lesion structure.
Spatial resolution shrinks and the channel count grows as the image goes deeper.

Used by train.py during training, and also usable on its own with a saved checkpoint:
    python visualize.py models/resnet18_best.pt path/to/image.jpg [--out flow.png]
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch.nn import functional as F

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
plt = None


def init_backend(live=False):
    """Choose the matplotlib backend. Must be called before plotting: Agg writes files only, live opens a window."""
    global plt
    if not live:
        matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    plt = pyplot
    if live:
        plt.ion()


def stage_modules(model, arch):
    """(name, module) pairs whose outputs mark each step of the network from input to output."""
    if arch.startswith("resnet"):
        return [("conv1+bn+relu", model.relu), ("maxpool", model.maxpool), ("layer1", model.layer1),
                ("layer2", model.layer2), ("layer3", model.layer3), ("layer4", model.layer4)]
    return [(f"features[{i}]", block) for i, block in enumerate(model.features)]


class FlowVisualizer:
    def __init__(self, model, arch, classes, channels=6, live=False):
        if plt is None:
            init_backend(live)
        self.model, self.classes, self.channels, self.live = model, classes, channels, live
        self.stages = stage_modules(model, arch)
        self.live_fig = None

    @torch.no_grad()
    def _denormalise(self, x):
        return (x.cpu() * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1).permute(1, 2, 0).numpy()

    def capture(self, x):
        """Run one normalised image (C,H,W) through the model. Returns activations, Grad-CAM and probabilities."""
        was_training = self.model.training
        self.model.eval()  # keeps BatchNorm/Dropout deterministic and stops this pass changing BN statistics
        acts, grads = {}, {}

        def hook(name):
            def fn(_module, _inp, out):
                acts[name] = out
                if out.requires_grad:
                    out.register_hook(lambda g: grads.__setitem__(name, g))
            return fn

        handles = [m.register_forward_hook(hook(n)) for n, m in self.stages]
        try:
            with torch.enable_grad():
                # requires_grad on the input lets Grad-CAM work even while the backbone is frozen
                inp = x.unsqueeze(0).detach().requires_grad_(True)
                logits = self.model(inp)
                probs = torch.softmax(logits, 1)[0]
                pred = int(probs.argmax())
                self.model.zero_grad(set_to_none=True)
                logits[0, pred].backward()
        finally:
            for h in handles:
                h.remove()
            self.model.zero_grad(set_to_none=True)
            self.model.train(was_training)

        last = self.stages[-1][0]
        cam = None
        if last in grads:
            weights = grads[last].mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * acts[last]).sum(1, keepdim=True))
            cam = F.interpolate(cam, size=x.shape[1:], mode="bilinear", align_corners=False)[0, 0]
            cam = (cam / (cam.max() + 1e-8)).detach().cpu().numpy()
        acts = {n: a[0].detach().cpu() for n, a in acts.items()}
        return acts, cam, probs.detach().cpu().numpy(), pred

    def _panels(self, x, acts, cam, probs, pred, label, title):
        """Turn one captured pass into the data shown in each panel (shared by build and update)."""
        cols = []
        for name, _ in self.stages:
            fmap = acts[name]
            strongest = fmap.mean(dim=(1, 2)).argsort(descending=True)[:self.channels].tolist()
            maps = []
            for ch in strongest:
                m = fmap[ch].numpy()
                maps.append((ch, (m - m.min()) / (m.max() - m.min() + 1e-8)))
            cols.append((name, tuple(fmap.shape), maps))
        true_txt = f"\ntrue: {self.classes[label]}" if label is not None else ""
        return {
            "image": self._denormalise(x),
            "cam": cam if cam is not None else np.zeros(x.shape[1:]),
            "input_title": f"INPUT\n3×{x.shape[1]}×{x.shape[2]}{true_txt}",
            "cols": cols,
            "probs": probs,
            "pred": pred,
            "title": title or "Image flow through the network",
        }

    def _build(self, fig, p):
        """Lay out the figure once and keep handles to every artist, so live updates only swap data."""
        rows, n_cols = self.channels, len(self.stages) + 2
        grid = fig.add_gridspec(rows, n_cols, wspace=0.08, hspace=0.08)
        h = {"cells": []}

        # input column: image on top, Grad-CAM (where the prediction comes from) below
        half = rows // 2
        ax = fig.add_subplot(grid[:half, 0])
        h["image"] = ax.imshow(p["image"])
        h["input_title"] = ax.set_title(p["input_title"], fontsize=9)
        ax.axis("off")
        ax = fig.add_subplot(grid[half:, 0])
        h["cam_base"] = ax.imshow(p["image"])
        h["cam"] = ax.imshow(p["cam"], cmap="jet", alpha=0.45, vmin=0, vmax=1)
        ax.set_title("Grad-CAM", fontsize=9)
        ax.axis("off")

        # one column per conv stage: its most active channels
        for col, (name, (c, hh, ww), maps) in enumerate(p["cols"], start=1):
            column = []
            for row, (ch, m) in enumerate(maps):
                ax = fig.add_subplot(grid[row, col])
                im = ax.imshow(m, cmap="viridis", interpolation="nearest", vmin=0, vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])
                label = ax.set_ylabel(f"ch {ch}", fontsize=6, labelpad=1)
                if row == 0:
                    ax.set_title(f"→ {name}\n{c}×{hh}×{ww}", fontsize=9)
                column.append((im, label))
            h["cells"].append(column)

        # output column: class probabilities
        ax = fig.add_subplot(grid[1:rows - 1, -1])
        h["bars"] = ax.barh(self.classes, p["probs"])
        ax.set_xlim(0, 1)
        h["bar_text"] = [ax.text(0, i, "", va="center", fontsize=8) for i in range(len(self.classes))]
        ax.set_title("→ OUTPUT\nsoftmax", fontsize=9)
        ax.tick_params(labelsize=8)
        h["suptitle"] = fig.suptitle(p["title"], fontsize=11)
        self._update(h, p)
        return h

    def _update(self, h, p):
        h["image"].set_data(p["image"])
        h["cam_base"].set_data(p["image"])
        h["cam"].set_data(p["cam"])
        h["input_title"].set_text(p["input_title"])
        for column, (_, _, maps) in zip(h["cells"], p["cols"]):
            for (im, label), (ch, m) in zip(column, maps):
                im.set_data(m)
                label.set_text(f"ch {ch}")
        for i, (bar, prob) in enumerate(zip(h["bars"], p["probs"])):
            bar.set_width(prob)
            bar.set_color("tab:green" if i == p["pred"] else "tab:gray")
            h["bar_text"][i].set_position((min(prob + 0.02, 0.8), i))
            h["bar_text"][i].set_text(f"{prob:.2f}")
        h["suptitle"].set_text(p["title"])

    def render(self, x, label=None, title="", out_path=None):
        """Draw the input -> stages -> output flow for one normalised image tensor.

        In live mode the window is laid out on the first call. Later calls only swap the
        image data, which is fast enough to update on every training step.
        """
        acts, cam, probs, pred = self.capture(x)
        p = self._panels(x, acts, cam, probs, pred, label, title)
        figsize = (2.0 * (len(self.stages) + 2) + 1, 1.7 * self.channels + 1.2)

        if self.live:
            if self.live_fig is None or not plt.fignum_exists(self.live_fig.number):
                self.live_fig = plt.figure(figsize=figsize)
                self._live_handles = self._build(self.live_fig, p)
            else:
                self._update(self._live_handles, p)
            fig = self.live_fig
        else:
            fig = plt.figure(figsize=figsize)
            self._build(fig, p)

        if out_path is not None:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=90, bbox_inches="tight")
        if self.live:
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
        else:
            plt.close(fig)
        return probs


def make_gif(frame_paths, out_path, ms_per_frame=700):
    """Stitch per-epoch PNGs into a GIF that shows the feature maps changing as training goes on."""
    from PIL import Image

    frames = [Image.open(p).convert("RGB") for p in frame_paths]
    if not frames:
        return
    size = frames[0].size
    frames = [f.resize(size) for f in frames]
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=ms_per_frame, loop=0)


def main():
    from PIL import Image

    from train import ARCHS, build_model, build_transforms

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="save the figure here (default: show a window)")
    parser.add_argument("--channels", type=int, default=6, help="feature maps shown per stage")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    assert ckpt["arch"] in ARCHS, f"unknown arch {ckpt['arch']}"
    model = build_model(ckpt["arch"])
    model.load_state_dict(ckpt["state_dict"])
    _, eval_tf = build_transforms(ckpt["img_size"])
    x = eval_tf(Image.open(args.image).convert("RGB"))

    init_backend(live=args.out is None)
    viz = FlowVisualizer(model, ckpt["arch"], ckpt["classes"], channels=args.channels)
    viz.render(x, title=f"{ckpt['arch']} — {args.image.name}", out_path=args.out)
    if args.out is None:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
