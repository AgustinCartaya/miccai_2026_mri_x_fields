from pathlib import Path
import shutil

src_root = Path("/home/agustin/phd/miccai/miccai_2026/mri_x_fields/evaluation/test5_segmentation_prior/results/basic/chk_200000/task1_zclipped")

dst_root = Path("/home/agustin/phd/miccai/miccai_2026/mri_x_fields/evaluation/test5_segmentation_prior/results/basic/chk_200000/task1_zclipped_reformatted")

for modality_dir in src_root.iterdir():
    if not modality_dir.is_dir():
        continue

    modality = modality_dir.name

    for task_dir in modality_dir.iterdir():
        if not task_dir.is_dir():
            continue

        pred_src = task_dir / "pred"
        seg_src = task_dir / "seg"

        if pred_src.exists():
            pred_dst = dst_root / "pred" / modality / task_dir.name
            pred_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(pred_src, pred_dst)

        if seg_src.exists():
            seg_dst = dst_root / "seg" / modality / task_dir.name
            seg_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(seg_src, seg_dst)