import csv
import json
import ast
from pathlib import Path

# CSV 파일 경로
csv_path = "/home/dohyeong/Desktop/COCO/annotations/annotations.csv"
json_path = "/home/dohyeong/Desktop/COCO/annotations/instances_all.json"

# COCO 형식 JSON 구조 초기화
coco_data = {
    "info": {
        "description": "FACET Dataset Annotations",
        "version": "1.0",
        "year": 2025
    },
    "licenses": [],
    "images": [],
    "annotations": [],
    "categories": [{"id": 1, "name": "person", "supercategory": "person"}]
}

# 이미지 ID 매핑 (filename -> image_id)
image_dict = {}
image_id_counter = 1
annotation_id_counter = 1

# CSV 파일 읽기
with open(csv_path, 'r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    
    for row in reader:
        filename = row['filename']
        
        # 이미지 정보 추가 (중복 방지)
        if filename not in image_dict:
            image_dict[filename] = image_id_counter
            coco_data["images"].append({
                "id": image_id_counter,
                "file_name": filename,
                "width": 2250,  # 기본값 (필요시 수정)
                "height": 1500  # 기본값 (필요시 수정)
            })
            image_id_counter += 1
        
        # bounding box 파싱
        bbox_str = row['bounding_box']
        bbox_dict = ast.literal_eval(bbox_str)
        x = bbox_dict['x']
        y = bbox_dict['y']
        width = bbox_dict['width']
        height = bbox_dict['height']
        
        # 어노테이션 속성 생성
        attributes = {
            "gender_presentation_masc": int(row['gender_presentation_masc']),
            "gender_presentation_fem": int(row['gender_presentation_fem']),
            "gender_presentation_non_binary": int(row['gender_presentation_non_binary']),
            "gender_presentation_na": int(row['gender_presentation_na']),
            "skin_tone_1": int(row['skin_tone_1']),
            "skin_tone_2": int(row['skin_tone_2']),
            "skin_tone_3": int(row['skin_tone_3']),
            "skin_tone_4": int(row['skin_tone_4']),
            "skin_tone_5": int(row['skin_tone_5']),
            "skin_tone_6": int(row['skin_tone_6']),
            "skin_tone_7": int(row['skin_tone_7']),
            "skin_tone_8": int(row['skin_tone_8']),
            "skin_tone_9": int(row['skin_tone_9']),
            "skin_tone_10": int(row['skin_tone_10']),
            "skin_tone_na": int(row['skin_tone_na']),
            "age_presentation_young": int(row['age_presentation_young']),
            "age_presentation_older": int(row['age_presentation_older']),
            "age_presentation_middle": int(row['age_presentation_middle']),
            "age_presentation_na": int(row['age_presentation_na']),
            "hair_color_brown": int(row['hair_color_brown']),
            "hair_color_blonde": int(row['hair_color_blonde']),
            "hair_color_grey": int(row['hair_color_grey']),
            "hair_color_na": int(row['hair_color_na']),
            "hair_color_black": int(row['hair_color_black']),
            "hair_color_colored": int(row['hair_color_colored']),
            "hair_color_red": int(row['hair_color_red']),
            "hairtype_coily": int(row['hairtype_coily']),
            "hairtype_dreadlocks": int(row['hairtype_dreadlocks']),
            "hairtype_bald": int(row['hairtype_bald']),
            "hairtype_straight": int(row['hairtype_straight']),
            "hairtype_curly": int(row['hairtype_curly']),
            "hairtype_wavy": int(row['hairtype_wavy']),
            "hairtype_na": int(row['hairtype_na']),
            "has_facial_hair": int(row['has_facial_hair']),
            "has_tattoo": int(row['has_tattoo']),
            "has_cap": int(row['has_cap']),
            "has_mask": int(row['has_mask']),
            "has_headscarf": int(row['has_headscarf']),
            "has_eyeware": int(row['has_eyeware']),
            "visible_torso": int(row['visible_torso']),
            "visible_face": int(row['visible_face']),
            "visible_minimal": int(row['visible_minimal']),
            "lighting_underexposed": int(row['lighting_underexposed']),
            "lighting_dimly_lit": int(row['lighting_dimly_lit']),
            "lighting_well_lit": int(row['lighting_well_lit']),
            "lighting_na": int(row['lighting_na']),
            "lighting_overexposed": int(row['lighting_overexposed'])
        }
        
        # 어노테이션 추가
        annotation = {
            "id": annotation_id_counter,
            "image_id": image_dict[filename],
            "category_id": 1,  # person
            "bbox": [x, y, width, height],
            "area": width * height,
            "iscrowd": 0,
            "attributes": attributes,
            "class1": row['class1'],
            "class2": row['class2']
        }
        coco_data["annotations"].append(annotation)
        annotation_id_counter += 1

# JSON 파일로 저장
with open(json_path, 'w', encoding='utf-8') as jsonfile:
    json.dump(coco_data, jsonfile, ensure_ascii=False, indent=2)

print(f"✅ 변환 완료!")
print(f"   - 이미지 수: {len(coco_data['images'])}")
print(f"   - 어노테이션 수: {len(coco_data['annotations'])}")
print(f"   - 저장 경로: {json_path}")
