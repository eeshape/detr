import json
import os
from pathlib import Path

# JSON 파일 경로
json_path = "/home/dohyeong/Desktop/COCO/annotations/instances_all.json"
out_dir = Path("/home/dohyeong/Desktop/COCO/annotations/")

# 출력 디렉토리 생성
out_dir.mkdir(parents=True, exist_ok=True)

# JSON 파일 로드
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 피부톤 데이터 추출 (각 어노테이션의 skin_tone_1 ~ skin_tone_10 값 가져오기)
annotations = data["annotations"]
skin_tone_data = []

# 각 어노테이션에서 피부톤 정보를 추출
for ann in annotations:
    skin_tone = []
    for i in range(1, 11):  # skin_tone_1부터 skin_tone_10까지
        skin_tone_value = ann["attributes"].get(f"skin_tone_{i}", 0)  # skin_tone 값 확인
        if skin_tone_value >= 1:  # 피부톤 값이 1 이상인 경우 해당 그룹에 추가
            skin_tone.append(i)
    
    # skin_tone_na가 1인 경우 (피부톤 정보가 없으면 제외)
    #if ann["attributes"].get("skin_tone_na") >= 1:
    #    continue
    
    skin_tone_data.append(skin_tone)

# 피부톤 그룹별로 COCO JSON 파일 생성
# 각 MST 값에 대해 성능을 평가할 그룹을 생성
for i in range(1, 11):  # MST 1 ~ MST 10
    group_annotations = []
    group_image_ids = set()

    for ann, skin_tones in zip(annotations, skin_tone_data):
        # 현재 MST 값에 해당하는 인스턴스만 그룹에 포함
        if i in skin_tones:
            group_annotations.append(ann)
            group_image_ids.add(ann["image_id"])

    # 이미지 메타 데이터 필터링
    group_images = [img for img in data["images"] if img["id"] in group_image_ids]
    
    # COCO JSON 생성
    coco_group = {
        "info": {"description": f"FACET Skin Tone Group MST {i}"},
        "licenses": [],
        "images": group_images,
        "annotations": group_annotations,
        "categories": [{"id": 1, "name": "person", "supercategory": "person"}],
    }

    # 그룹별 JSON 파일 저장
    out_path = out_dir / f"skin{i}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(coco_group, f, ensure_ascii=False)
    
    print(f"Group MST {i} → {out_path} (images={len(group_images)}, anns={len(group_annotations)})")
