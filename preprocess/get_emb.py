# preprocess/get_emb.py
"""Extract Merlin image embeddings from liver CT studies."""
import os, setproctitle
from pathlib import Path

setproctitle.setproctitle("liver-report-preprocessing")
import copy
import warnings
from typing import List, Tuple
import numpy as np
import pandas as pd
import nibabel as nib
import h5py
from tqdm import tqdm

import torch
import torch.nn as nn


import torchvision
from merlin.models import i3res

warnings.filterwarnings("ignore")


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Config:
    """Paths and preprocessing parameters for the liver CT dataset."""


    DATA_XLSX = PROJECT_ROOT / 'data' / 'data.xlsx'


    MERLIN_CHECKPOINT = PROJECT_ROOT / 'models' / 'merlin' / 'weights' / 'i3_resnet_clinical_longformer_best_clip_04-02-2024_23-21-36_epoch_99.pt'


    OUTPUT_DIR = PROJECT_ROOT / 'embedding'
    OUTPUT_H5 = OUTPUT_DIR / 'liver_embeddings.h5'


    HU_MIN = -1000.0
    HU_MAX = 1000.0
    TARGET_IN_PLANE = 224
    TARGET_Z = 160
    IN_PLANE_SPACING = 1.5
    Z_SPACING = 3.0


    DEVICE = 'cuda:0'


    BATCH_SIZE = 4


class MerlinImageEncoder(nn.Module):
    """Load the Merlin image encoder and produce 512-dimensional embeddings."""

    def __init__(self):
        super().__init__()


        resnet = torchvision.models.resnet152(pretrained=True)
        self.i3_resnet = i3res.I3ResNet(
            copy.deepcopy(resnet),
            class_nb=1692,
            conv_class=True,
            ImageEmbedding=False,
            PhenotypeCls=False,
            FiveYearPred=False,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Forward."""

        if image.dim() == 4:
            image = image.unsqueeze(1)  # [B, 1, D, H, W]


        contrastive_features, _ = self.i3_resnet(image)

        # contrastive_features: [B, 512]
        return contrastive_features

    @torch.no_grad()
    def extract_embedding(self, image: torch.Tensor) -> np.ndarray:
        """Extract embedding."""
        self.eval()
        image = image.to(Config.DEVICE)
        feat = self.forward(image)
        return feat.cpu().numpy()


class CTPreprocessor:
    """Normalize and resize a CT volume for the Merlin encoder."""

    def __init__(self):
        self.hu_min = Config.HU_MIN
        self.hu_max = Config.HU_MAX
        self.target_in_plane = Config.TARGET_IN_PLANE
        self.target_z = Config.TARGET_Z

    def __call__(self, nii_path: str) -> np.ndarray:
        """  call  ."""
        from scipy.ndimage import zoom


        nii = nib.load(nii_path)
        img = nii.get_fdata().astype(np.float32)


        in_h, in_w, depth = img.shape


        img = np.clip(img, self.hu_min, self.hu_max)
        img = (img - self.hu_min) / (self.hu_max - self.hu_min)


        zoom_h = self.target_in_plane / in_h  # 224/512
        zoom_w = self.target_in_plane / in_w  # 224/512

        img_resized = np.zeros((self.target_in_plane, self.target_in_plane, depth), dtype=np.float32)
        for i in range(depth):
            img_resized[:, :, i] = zoom(img[:, :, i], (zoom_h, zoom_w), order=1)
        img = img_resized  # [224, 224, depth]


        if depth < self.target_z:

            pad_before = (self.target_z - depth) // 2
            pad_after = self.target_z - depth - pad_before
            img = np.pad(img, ((0, 0), (0, 0), (pad_before, pad_after)),
                         mode='constant', constant_values=0)
        elif depth > self.target_z:

            zoom_d = self.target_z / depth
            img = zoom(img, (1, 1, zoom_d), order=1)


        img = img[np.newaxis, ...]

        assert img.shape == (1, self.target_in_plane, self.target_in_plane, self.target_z), \
            f"Shape mismatch: {img.shape}, expected (1, {self.target_in_plane}, {self.target_in_plane}, {self.target_z})"

        return img


def load_merlin_encoder(checkpoint_path: str) -> MerlinImageEncoder:
    """Load merlin encoder."""
    print(f"Loading Merlin checkpoint from: {checkpoint_path}")

    encoder = MerlinImageEncoder()
    encoder.to(Config.DEVICE)


    checkpoint = torch.load(checkpoint_path, map_location='cpu')


    state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint.state_dict()


    encoder_dict = encoder.state_dict()
    filtered_dict = {}

    for k, v in state_dict.items():

        if k.startswith('i3_resnet') or k.startswith('encode_image.i3_resnet'):

            if k.startswith('encode_image.i3_resnet'):
                new_k = k.replace('encode_image.', '')
            else:
                new_k = k
            if new_k in encoder_dict:
                filtered_dict[new_k] = v
            else:
                print(f"  Warning: key {new_k} not found in encoder")
        else:

            pass


    missing, unexpected = encoder.load_state_dict(filtered_dict, strict=False)
    print(f"Loaded weights: {len(filtered_dict)} keys")
    print(f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")


    for param in encoder.parameters():
        param.requires_grad = False

    encoder.eval()
    print("Merlin image encoder loaded and frozen.")

    return encoder


def extract_image_embeddings(
        encoder: MerlinImageEncoder,
        df: pd.DataFrame,
        batch_size: int = 4
) -> Tuple[np.ndarray, List[str], List[int]]:
    """Extract image embeddings."""
    preprocessor = CTPreprocessor()
    embeddings = []
    pids = []
    success_flags = []

    total = len(df)
    num_batches = (total + batch_size - 1) // batch_size

    pbar = tqdm(total=total, desc="Extracting image embeddings")

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total)
        batch_df = df.iloc[start_idx:end_idx]

        batch_tensors = []
        batch_pids = []

        for _, row in batch_df.iterrows():
            pid = str(row['id'])
            img_path = PROJECT_ROOT / row['Venous_img']

            try:

                img = preprocessor(img_path)  # [1, 160, 224, 224]
                batch_tensors.append(img)
                batch_pids.append(pid)
            except Exception as e:
                print(f"\n[ERROR] PID {pid}: preprocessing failed: {e}")

                pids.append(pid)
                embeddings.append(np.zeros(512, dtype=np.float32))
                success_flags.append(0)
                pbar.update(1)
                continue

        if len(batch_tensors) == 0:
            continue


        batch_array = np.stack(batch_tensors, axis=0)  # [B, 1, 160, 224, 224]
        batch_tensor = torch.from_numpy(batch_array).to(Config.DEVICE)


        with torch.no_grad():
            feats = encoder(batch_tensor)  # [B, 512]

        feats_np = feats.cpu().numpy()

        for i, pid in enumerate(batch_pids):
            pids.append(pid)
            embeddings.append(feats_np[i])
            success_flags.append(1)
            pbar.update(1)

    pbar.close()

    embeddings = np.stack(embeddings, axis=0)  # [N, 512]
    return embeddings, pids, success_flags


def save_to_hdf5(
        output_path: str,
        pids: List[str],
        image_embeds: np.ndarray,
        report_descs: List[str],
        success_flags: List[int]
):
    """Save to hdf5."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)


    pids_array = np.array(pids, dtype=h5py.string_dtype())
    reports_array = np.array(report_descs, dtype=h5py.string_dtype())

    with h5py.File(output_path, 'w') as f:
        f.create_dataset('pids', data=pids_array)
        f.create_dataset('image_embeds', data=image_embeds.astype(np.float32))
        f.create_dataset('report_desc', data=reports_array)
        f.create_dataset('success_flags', data=np.array(success_flags, dtype=np.int8))


        f.attrs['description'] = 'Liver CT embeddings: Merlin 512-dim image + report text'
        f.attrs['num_samples'] = len(pids)
        f.attrs['image_embed_dim'] = 512

    print(f"\nSaved to: {output_path}")
    print(f"  Samples: {len(pids)}")
    print(f"  Image embeds shape: {image_embeds.shape}")
    print(
        f"  Success rate: {sum(success_flags)}/{len(success_flags)} ({sum(success_flags) / len(success_flags) * 100:.1f}%)")


def main():
    print("=" * 60)
    print("Liver CT Preprocessing - Merlin + Qwen")
    print("=" * 60)
    print(f"Device: {Config.DEVICE}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. A CUDA-enabled PyTorch installation and GPU are required.")
    torch.cuda.set_device(0)


    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)


    print(f"\nLoading data from: {Config.DATA_XLSX}")
    df = pd.read_excel(Config.DATA_XLSX)
    print(f"Total samples: {len(df)}")


    df = df[df['Venous_img'].notna() & df['report_desc'].notna()].copy()
    print(f"Valid samples (with image + report): {len(df)}")


    encoder = load_merlin_encoder(Config.MERLIN_CHECKPOINT)


    print(f"\nExtracting image embeddings (batch_size={Config.BATCH_SIZE})...")
    image_embeds, pids, success_flags = extract_image_embeddings(
        encoder, df, batch_size=Config.BATCH_SIZE
    )


    print("\nExtracting report texts...")
    report_descs = []
    for pid in pids:
        row = df[df['id'].astype(str) == pid].iloc[0]
        report_descs.append(str(row['report_desc']))


    save_to_hdf5(
        Config.OUTPUT_H5,
        pids,
        image_embeds,
        report_descs,
        success_flags
    )


    print("\nVerifying HDF5 file...")
    with h5py.File(Config.OUTPUT_H5, 'r') as f:
        print(f"  Keys: {list(f.keys())}")
        print(f"  pids shape: {f['pids'].shape}")
        print(f"  image_embeds shape: {f['image_embeds'].shape}")
        print(f"  report_desc shape: {f['report_desc'].shape}")
        print(f"  Sample PID: {f['pids'][0]}")
        print(f"  Sample report: {f['report_desc'][0][:50]}...")

    print("\n" + "=" * 60)
    print("Preprocessing completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
