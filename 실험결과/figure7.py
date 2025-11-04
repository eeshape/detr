import matplotlib.pyplot as plt
import numpy as np

# 실험 데이터
skin_types = np.arange(1, 11)

# Darkness Value 0.1 (가장 어두움)
darkness_01_values = [0.811, 0.804, 0.800, 0.796, 0.796, 0.793, 0.791, 0.787, 0.781, 0.782]

# Darkness Value 0.5 (중간)
darkness_05_values = [0.847, 0.841, 0.838, 0.837, 0.835, 0.834, 0.832, 0.834, 0.830, 0.824]

# Darkness Value 1.0 (가장 밝음)
darkness_10_values = [0.855, 0.847, 0.844, 0.841, 0.840, 0.840, 0.837, 0.839, 0.835, 0.828]

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

# 각 skin type별로 darkness value에 따른 변화를 선으로 그리기
x_positions = [0.1, 0.5, 1.0]
for i in range(len(skin_types)):
    y_values = [darkness_01_values[i], darkness_05_values[i], darkness_10_values[i]]
    line = plt.plot(x_positions, y_values, 
                    marker='o', color=colors[i], linewidth=2, markersize=6)
    
    # darkness 1.0 (오른쪽) 점 위에 숫자 표시
    plt.text(1.0, darkness_10_values[i], str(i+1), fontsize=9, ha='left', va='bottom',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.8))

# 그래프 설정
plt.xlabel('Darkness Value', fontsize=12)
plt.ylabel('mAR', fontsize=12)
plt.title('mAR by Darkness Value and Skin Type', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xlim(0.05, 1.05)
plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
plt.tight_layout()

# 그래프 저장 및 표시
plt.savefig('/home/dohyeong/Desktop/figure7_darkness.png', dpi=300, bbox_inches='tight')
plt.show()

print("그래프가 생성되었습니다: figure7_darkness.png")

# 데이터 요약 출력
print("\n=== Darkness Value 1.0 (가장 밝음) ===")
for i, val in enumerate(darkness_10_values, 1):
    print(f"Skin Type {i}: {val}")

print("\n=== Darkness Value 0.5 (중간) ===")
for i, val in enumerate(darkness_05_values, 1):
    print(f"Skin Type {i}: {val}")

print("\n=== Darkness Value 0.1 (가장 어두움) ===")
for i, val in enumerate(darkness_01_values, 1):
    print(f"Skin Type {i}: {val}")

print(f"\nDarkness 1.0 평균: {np.mean(darkness_10_values):.3f}")
print(f"Darkness 0.5 평균: {np.mean(darkness_05_values):.3f}")
print(f"Darkness 0.1 평균: {np.mean(darkness_01_values):.3f}")
