import json
import shutil
from pathlib import Path

# 경로 설정
base_dir = Path("/workspace/output_extracted/annotations")
json_path = base_dir / "instances_all.json"

# 이미지 원본 폴더들
img_src_dirs = [
    Path("/workspace/raw image/imgs_1"),
    Path("/workspace/raw image/imgs_2"),
    Path("/workspace/raw image/imgs_3"),
]

# 이미지 파일명 → 실제 경로 매핑
img_path_map = {}
for d in img_src_dirs:
    for f in d.iterdir():
        if f.is_file():
            img_path_map[f.name] = f

# JSON 파일 로드
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

annotations = data["annotations"]

# 성별 그룹 정의
gender_groups = {
    "men":   "gender_presentation_masc",
    "women": "gender_presentation_fem",
}

for group_name, attr_key in gender_groups.items():
    group_annotations = []
    group_image_ids = set()

    for ann in annotations:
        if ann["attributes"].get(attr_key, 0) >= 1:
            group_annotations.append(ann)
            group_image_ids.add(ann["image_id"])

    group_images = [img for img in data["images"] if img["id"] in group_image_ids]

    # --- JSON 저장 ---
    coco_group = {
        "info": {"description": f"FACET Gender Group - {group_name}"},
        "licenses": [],
        "images": group_images,
        "annotations": group_annotations,
        "categories": [{"id": 1, "name": "person", "supercategory": "person"}],
    }

    out_path = base_dir / f"gender_{group_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(coco_group, f, ensure_ascii=False)

    # --- 이미지 복사 ---
    img_out_dir = base_dir.parent / f"{group_name}_images"
    img_out_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing = 0
    for img in group_images:
        fname = img["file_name"]
        src = img_path_map.get(fname)
        dst = img_out_dir / fname
        if src and not dst.exists():
            shutil.copy2(src, dst)
            copied += 1
        elif not src:
            missing += 1

    print(f"{group_name} → JSON: {out_path} (images={len(group_images)}, anns={len(group_annotations)})")
    print(f"         images: {img_out_dir} (copied={copied}, missing={missing})")
