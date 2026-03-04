import json
import random
import shutil
from pathlib import Path

random.seed(42)

# 경로 설정
ann_dir = Path("/workspace/output_extracted/annotations")
dataset_root = Path("/workspace/faap_dataset")

# 성별별 설정
gender_config = {
    "men": {
        "json": ann_dir / "gender_men.json",
        "img_src": Path("/workspace/output_extracted/men_images"),
        "split_dir": dataset_root / "men_split",
    },
    "women": {
        "json": ann_dir / "gender_women.json",
        "img_src": Path("/workspace/output_extracted/women_images"),
        "split_dir": dataset_root / "women_split",
    },
}

# split 비율 (8:1:1)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
# TEST_RATIO = 0.1 (나머지)

for gender, cfg in gender_config.items():
    print(f"\n=== {gender} ===")

    with open(cfg["json"], "r", encoding="utf-8") as f:
        data = json.load(f)

    images = data["images"]
    annotations = data["annotations"]

    # image_id → annotation 매핑
    ann_by_image = {}
    for ann in annotations:
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    # 이미지 셔플 후 split
    random.shuffle(images)
    n = len(images)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:],
    }

    for split_name, split_images in splits.items():
        # 디렉토리 생성
        img_dst_dir = cfg["split_dir"] / split_name
        img_dst_dir.mkdir(parents=True, exist_ok=True)

        # 해당 split의 image_id 집합
        split_image_ids = {img["id"] for img in split_images}

        # annotation 필터링
        split_anns = []
        for img in split_images:
            split_anns.extend(ann_by_image.get(img["id"], []))

        # COCO JSON 생성
        coco_json = {
            "info": {"description": f"FACET {gender} {split_name}"},
            "licenses": [],
            "images": split_images,
            "annotations": split_anns,
            "categories": [{"id": 1, "name": "person", "supercategory": "person"}],
        }

        json_path = cfg["split_dir"] / f"gender_{gender}_{split_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(coco_json, f, ensure_ascii=False)

        # 이미지 복사
        copied = 0
        for img in split_images:
            src = cfg["img_src"] / img["file_name"]
            dst = img_dst_dir / img["file_name"]
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
                copied += 1

        print(f"  {split_name}: images={len(split_images)}, anns={len(split_anns)}, copied={copied}")
        print(f"    JSON: {json_path}")
        print(f"    imgs: {img_dst_dir}")

print("\nDone!")
print(f"dataset_root: {dataset_root}")
