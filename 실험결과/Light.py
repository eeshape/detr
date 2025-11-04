import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# 실험 데이터
skin_types = np.arange(1, 11)

# Well lighting condition mAR values
well_values = [0.847, 0.843, 0.840, 0.840, 0.840, 0.839, 0.838, 0.837, 0.830, 0.821]

# Dimly lighting condition mAR values
dimly_values = [0.818, 0.812, 0.806, 0.805, 0.802, 0.804, 0.804, 0.811, 0.814, 0.814]

# 살색에서 검은색으로 그라데이션 색상 생성
colors = []
for i in range(10):
    # 연한 살색 (RGB: 255, 220, 177)에서 검은색 (RGB: 0, 0, 0)으로 변화
    ratio = i / 9  # 0부터 1까지
    r = 1.0 - (ratio * 1.0)  # 1.0 -> 0.0
    g = (220/255) - (ratio * (220/255))  # 0.86 -> 0.0
    b = (177/255) - (ratio * (177/255))  # 0.69 -> 0.0
    colors.append((r, g, b))

# 그래프 생성
plt.figure(figsize=(10, 6))

# 각 skin type별로 dimly와 well을 연결하는 선 그리기
for i in range(len(skin_types)):
    line = plt.plot(['dimly', 'well'], [dimly_values[i], well_values[i]], 
                    marker='o', color=colors[i], linewidth=2, markersize=6)
    
    # well 쪽 점 위에 숫자 표시
    plt.text(1, well_values[i], str(i+1), fontsize=9, ha='left', va='bottom',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.8))

# 그래프 설정
plt.xlabel('Lighting Condition', fontsize=12)
plt.ylabel('mAR', fontsize=12)
plt.title('mAR by Lighting Condition and Skin Type', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 그래프 저장 및 표시
plt.savefig('/home/dohyeong/Desktop/lighting_condition_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("그래프가 생성되었습니다: lighting_condition_comparison.png")

# 데이터 요약 출력
print("\n=== Well Lighting Condition ===")
for i, val in enumerate(well_values, 1):
    print(f"Skin Type {i}: {val}")

print("\n=== Dimly Lighting Condition ===")
for i, val in enumerate(dimly_values, 1):
    print(f"Skin Type {i}: {val}")

print(f"\nWell 평균: {np.mean(well_values):.3f}")
print(f"Dimly 평균: {np.mean(dimly_values):.3f}")
print(f"차이: {np.mean(well_values) - np.mean(dimly_values):.3f}")
