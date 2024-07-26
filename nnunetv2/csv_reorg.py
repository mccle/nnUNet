import pandas as pd
import argparse

from pathlib import Path
from typing import Sequence
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

def reorganize(csv: Path | str, outdir: Path | str, dataset: str = "DATASET", fold: int = 0, im_keys: Sequence[str] = ["T1Post"], train_all: bool = False):
    df = pd.read_csv(csv, dtype=str)
    df = df.sort_values(["Anon_PatientID", "Anon_StudyID"])
    df["nnUnetID"] = [f"{i:03d}" for i in range(df.shape[0])]

    if train_all:
        df[f"fold_{fold}"] = ["train"] * df.shape[0]

    for row in df.to_dict("records"):
        partition: str = row[f"fold_{fold}"]
        subdir: str = {"train": "imagesTr", "test": "imagesTs"}.get(partition, "imagesTr")
        for i, key in enumerate(im_keys):
            im_path = Path(outdir) / subdir / f"{dataset}_{row['nnUnetID']}_{i:04d}.nii.gz"
            im_path.parent.mkdir(parents=True, exist_ok=True)
            im_path.symlink_to(row[key])

        subdir: str = {"train": "labelsTr", "test": "labelsTs"}.get(partition, "labelsTr")
        seg_path = Path(outdir) / subdir / f"{dataset}_{row['nnUnetID']}.nii.gz"
        seg_path.parent.mkdir(parents=True, exist_ok=True)
        seg_path.symlink_to(row["seg"])

    generate_dataset_json(
        output_folder=str(outdir),
        channel_names= {"0": "T1Post"},
        labels={"background": 0},
        num_training_cases=df[df[f"fold_{fold}"].isin(["train", "val"])].shape[0],
        file_ending=".nii.gz"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("csv", type=Path)
    parser.add_argument("outdir", type=Path)
    parser.add_argument("--dataset", type=str, default="DATASET")
    parser.add_argument("--im-keys", type=str, default="T1Post")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--train-all", action="store_true")

    args = parser.parse_args()

    kwargs = {
        "csv": args.csv,
        "outdir": args.outdir,
        "dataset": args.dataset,
        "fold": args.fold,
        "im_keys": args.im_keys.split(","),
        "train_all": args.train_all
    }

    reorganize(**kwargs)
