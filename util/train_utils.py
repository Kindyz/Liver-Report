# util/train_utils.py
"""Utilities for training configuration, logging, and visualization."""
import os, csv, yaml, shutil, datetime, logging
from os.path import join
from collections import OrderedDict
from easydict import EasyDict
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import torch


FIRST_PRINT = True


def load_config(config_path):
    """Load config."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return EasyDict(config)


def save_config(config_path, save_dir, savename='config'):
    """Save config."""

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    filename = f"{savename}_{current_time}.yaml"

    save_path = os.path.join(save_dir, filename)


    os.makedirs(save_dir, exist_ok=True)

    shutil.copy(config_path, save_path)

    return save_path


class EpochCSVLogger:
    """Write one CSV record per epoch and support resumed runs."""
    def __init__(self, save_path, filename, resume=False):
        save_dir = join(save_path, 'results_csv')
        os.makedirs(save_dir, exist_ok=True)
        self.filepath = join(save_dir, filename)
        self.resume = resume and os.path.exists(self.filepath)


        self.data = OrderedDict()


        if self.resume:
            self._load()

    def _load(self):
        """ load."""
        with open(self.filepath, 'r', newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                epoch = int(row['epoch'])

                row.pop('epoch')
                self.data[epoch] = row

    def add_epoch(self, epoch, **kwargs):
        """Add epoch."""

        row_data = {k: str(v) for k, v in kwargs.items()}

        if epoch in self.data:

            self.data[epoch].update(row_data)
        else:

            self.data[epoch] = row_data


        self._flush()

    def _flush(self):
        """ flush."""
        if not self.data:
            return


        all_keys = set()
        for row in self.data.values():
            all_keys.update(row.keys())
        fieldnames = ['epoch'] + sorted(all_keys)


        with open(self.filepath, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for epoch, row in self.data.items():
                row_with_epoch = {'epoch': epoch}
                row_with_epoch.update(row)
                writer.writerow(row_with_epoch)


def add_column_titles(grid_np, titles, fontsize=None, font_color=(0, 255, 255)):
    """Add text labels to the visualization grid."""
    pil = Image.fromarray(grid_np)
    draw = ImageDraw.Draw(pil)
    font = fontsize if fontsize is None else ImageFont.load_default()

    ncol = len(titles)
    w, h = pil.size          # (W, H)
    col_w = w // ncol
    y = 5
    for idx, title in enumerate(titles):
        x = idx * col_w
        draw.text((x, y), title, font=font, fill=font_color)

    return pil


def plot_intensity_histogram(cond_HU, fake_HU, real_HU, vis_dir, epoch,
                              hu_min=-1000, hu_max=1000, num_bins=500,
                              smooth_window=5):
    """Plot intensity histogram."""


    SMOOTHING = True


    COLORS = {
        'CTA': '#A5A5A5',
        'Real_NCCT': '#2ecc71',
        'Fake_NCCT': '#7030A0'
    }


    sl = cond_HU.size(-1) // 2


    cond_slice = cond_HU[:, 0, :, :, sl].numpy().flatten()  # (B*H*W,)
    fake_slice = fake_HU[:, 0, :, :, sl].numpy().flatten()
    real_slice = real_HU[:, 0, :, :, sl].numpy().flatten()


    cond_vol = cond_HU[:, 0].numpy().flatten()
    fake_vol = fake_HU[:, 0].numpy().flatten()
    real_vol = real_HU[:, 0].numpy().flatten()


    def _compute_histogram(data, hu_min, hu_max, num_bins):
        """ compute histogram."""

        mask = (data >= hu_min) & (data <= hu_max)
        filtered = data[mask]

        if len(filtered) == 0:
            return None, None, 0


        bins = np.linspace(hu_min, hu_max, num_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_width = bins[1] - bins[0]

        counts, _ = np.histogram(filtered, bins=bins)
        density = counts / (counts.sum() * bin_width) if counts.sum() > 0 else counts

        return bin_centers, density, len(filtered)

    def _smooth_curve(x, y, window_size):
        """ smooth curve."""
        if window_size < 3 or window_size % 2 == 0:
            window_size = max(3, window_size + 1)
        half_window = window_size // 2
        y_smooth = np.copy(y).astype(float)
        for i in range(half_window, len(y) - half_window):
            y_smooth[i] = np.mean(y[i - half_window:i + half_window + 1])
        return x, y_smooth


    datasets = [
        ('Slice', cond_slice, fake_slice, real_slice, f'epoch{epoch:04d}_Histogram_slice.png'),
        ('Volume', cond_vol, fake_vol, real_vol, f'epoch{epoch:04d}_Histogram_volume.png')
    ]

    for data_type, cond_data, fake_data, real_data, filename in datasets:
        fig, ax = plt.subplots(figsize=(12, 7))

        stats_info = []


        for name, data, color in [
            ('CTA (Input)', cond_data, COLORS['CTA']),
            ('Real NCCT (Target)', real_data, COLORS['Real_NCCT']),
            ('Generated NCCT', fake_data, COLORS['Fake_NCCT'])
        ]:
            bin_centers, density, pixel_count = _compute_histogram(
                data, hu_min, hu_max, num_bins
            )

            if bin_centers is None:
                continue


            if SMOOTHING:
                bin_centers, density = _smooth_curve(bin_centers, density, smooth_window)


            ax.plot(bin_centers, density,
                    label=name,
                    color=color,
                    linewidth=2.5,
                    linestyle='-',
                    alpha=0.9)


            stats_info.append({
                'name': name,
                'mean': np.mean(data),
                'std': np.std(data),
                'count': pixel_count
            })


        ax.axvline(x=0, color='gray', linestyle=':', linewidth=1.5, alpha=0.6, label='Water (0 HU)')
        ax.axvline(x=50, color='lightcoral', linestyle=':', linewidth=2, alpha=0.8, label='Vessel (~50 HU)')


        ax.set_xlabel('HU Value', fontsize=12, fontweight='bold')
        ax.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
        ax.set_title(f'CT Intensity Distribution - {data_type} Level\n'
                     f'Epoch {epoch} (HU: [{hu_min}, {hu_max}])',
                     fontsize=13, fontweight='bold', pad=15)

        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim([hu_min, hu_max])
        ax.set_ylim(bottom=0)


        stats_text = 'Statistics:\n'
        for s in stats_info:
            stats_text += f'{s["name"].split()[0]}: mean={s["mean"]:.1f}, std={s["std"]:.1f}\n'

        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()


        save_path = join(vis_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        # print(f'[Histogram] Saved {data_type} distribution plot: {save_path}')


def setup_seed(seed: int):
    """Setup seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log_setup(savepath, log, name=None):
    if name is None:
        name = 'hetero'
    ldir = savepath
    if not os.path.exists(ldir):
        os.makedirs(ldir)

    lfile = name + '_' + log + '.txt' if log else name + '.txt'
    lfile = os.path.join(ldir, lfile)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        filename=lfile,
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logging.getLogger('').addHandler(console)
