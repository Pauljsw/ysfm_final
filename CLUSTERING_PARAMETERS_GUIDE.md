# Clustering Parameters Guide - Enhanced Version

## 개요

개선된 geometric clustering은 **6가지 기준**으로 균열을 판정합니다:

1. **방향 (Direction)** - 교차 균열 방지
2. **거리 (Distance)** - 중심점 거리 (동적 조정)
3. **연결성 (Connectivity)** - Gap 및 주축 정렬
4. **BBox 겹침** - 공간적 근접성
5. **Point 근접성** - 세밀한 검증
6. **최종 Score** - 가중 조합

---

## 파라미터 전체 목록

```bash
python -m src.cluster_masks_3d_geometric \
  --masks-3d outputs/masks_3d.json \
  --output outputs/mask_clusters.json \

  # 1. Direction (방향)
  --min-angle-similarity 0.707 \        # cos(45°)

  # 2. Distance (거리)
  --max-centroid-distance 0.5 \         # meters
  --use-dynamic-threshold \             # BBox 크기 기반 동적 조정

  # 3. Connectivity (연결성)
  --gap-threshold 0.1 \                 # meters

  # 4. Point proximity
  --proximity-threshold 0.05 \          # meters

  # 5. Final score
  --overlap-threshold 0.3               # 0~1
```

---

## 1. Direction (방향 파라미터)

### `--min-angle-similarity` (기본값: 0.707)

**의미:** 두 균열의 주축 방향이 얼마나 비슷해야 같은 균열로 판정할지

**값 범위:** 0.0 ~ 1.0 (cosine similarity)
- `1.0` = 완전히 평행
- `0.707` = 45도 이내 (**권장**)
- `0.5` = 60도 이내
- `0.0` = 수직

**계산 방법:**
```python
# Principal axis (PCA로 계산)
mask_axis = [0.9, 0.1, 0.0]      # 거의 수평
cluster_axis = [0.1, 0.9, 0.0]   # 거의 수직

# Angle similarity = |cos(θ)|
similarity = |dot(mask_axis, cluster_axis)| = 0.18

# 45도 기준: 0.707
if similarity < 0.707:
    return 0.0  # 다른 방향 → 다른 균열!
```

**사용 사례:**

```bash
# Case 1: 엄격 (30도 이내만 허용)
--min-angle-similarity 0.866  # cos(30°)
# 효과: 교차 균열 완벽 분리
# 단점: 약간 구부러진 균열도 분리될 수 있음

# Case 2: 권장 (45도 이내)
--min-angle-similarity 0.707  # cos(45°)
# 효과: 교차 균열 방지, 구부러진 균열 허용
# 권장!

# Case 3: 느슨 (60도 이내)
--min-angle-similarity 0.5  # cos(60°)
# 효과: 구부러진 균열도 병합
# 단점: 교차 균열이 병합될 수 있음
```

**시각화:**
```
균열 A: ══════ (수평, axis = [1, 0, 0])
균열 B:   ║    (수직, axis = [0, 1, 0])
          ║

similarity = |1*0 + 0*1 + 0*0| = 0.0
→ 0.0 < 0.707 → Rejected! (교차 균열)
```

---

## 2. Distance (거리 파라미터)

### `--max-centroid-distance` (기본값: 0.5m)

**의미:** 두 균열의 중심점이 최대 얼마나 떨어질 수 있는가 (정적 threshold)

**값 범위:** 0.1 ~ 2.0 meters
- `0.3m` = 보수적 (작은 균열용)
- `0.5m` = 권장 (일반적)
- `1.0m` = 느슨 (큰 균열용)

**Dynamic threshold와의 관계:**
```python
if use_dynamic_threshold:
    # BBox 크기에 비례
    dynamic = max(mask_bbox_size, cluster_bbox_size) * 0.7

    # max_centroid_distance는 최대 제한으로 사용 가능
    if max_centroid_distance_limit is not None:
        dynamic = min(dynamic, max_centroid_distance_limit)
else:
    # 정적 threshold
    dynamic = max_centroid_distance
```

---

### `--use-dynamic-threshold` (기본값: True)

**의미:** BBox 크기에 따라 centroid threshold를 자동 조정

**효과:**
```
작은 균열 (BBox 20cm):
  dynamic_threshold = 0.2 * 0.7 = 0.14m

중간 균열 (BBox 50cm):
  dynamic_threshold = 0.5 * 0.7 = 0.35m

긴 균열 (BBox 2m):
  dynamic_threshold = 2.0 * 0.7 = 1.4m

→ 균열 크기에 맞춰 자동 조정!
```

**사용 사례:**

```bash
# Case 1: Dynamic ON (권장)
--use-dynamic-threshold
# 효과: 긴 균열도 자동으로 병합
# 작은 균열은 엄격, 긴 균열은 느슨

# Case 2: Dynamic OFF (고정 threshold)
--max-centroid-distance 0.5
# 효과: 모든 균열에 동일한 기준
# 긴 균열(>1m)은 분할될 수 있음
```

---

## 3. Connectivity (연결성 파라미터)

### `--gap-threshold` (기본값: 0.1m)

**의미:** 두 균열 사이 최소 거리가 이 값보다 작으면 "연결됨"으로 판정

**값 범위:** 0.03 ~ 0.2 meters
- `0.05m` = 엄격 (5cm)
- `0.1m` = 권장 (10cm)
- `0.2m` = 느슨 (20cm)

**작동 방식:**
```python
# Centroid가 멀어도 gap이 작으면 연결됨
if centroid_dist > dynamic_threshold:
    min_gap = compute_min_gap(mask, cluster)

    if min_gap < gap_threshold:  # 10cm 이내
        # 연결됨! 계속 진행
        pass
    else:
        # Gap도 크면 reject
        return 0.0
```

**시각화:**
```
Mask A: ●●●●●●    (gap: 8cm)    ●●●●●● Mask B

min_gap = 0.08m < 0.1m
→ 연결됨! (같은 균열)
```

**사용 사례:**

```bash
# Case 1: 엄격 (연속 균열만)
--gap-threshold 0.05
# 효과: 5cm 이상 떨어지면 다른 균열
# 단점: 약간 떨어진 균열도 분리

# Case 2: 권장 (일반적)
--gap-threshold 0.1
# 효과: 10cm 이내면 연결됨
# 권장!

# Case 3: 느슨 (분할 균열 병합)
--gap-threshold 0.2
# 효과: 20cm gap도 연결
# 단점: 과병합 위험
```

---

### `axis_alignment_threshold` (기본값: cos(30°) = 0.866)

**의미:** Mask가 Cluster의 주축 방향으로 정렬되어 있는지 체크

**코드 내부 파라미터** (CLI에서 수정 불가, 코드에서 조정)

**작동 방식:**
```python
# Connection vector (cluster → mask)
connection_vec = mask_centroid - cluster_centroid

# Cluster의 주축
cluster_axis = [1, 0, 0]  # 수평 균열

# Alignment check
alignment = |dot(connection_vec, cluster_axis)|

if alignment > 0.866:  # 30도 이내
    # 주축 방향으로 정렬됨
    # 긴 균열의 연장으로 판정
    return True
```

**시각화:**
```
Cluster: ══════ (axis = [1, 0, 0])
              ↗ (connection: 25도)
            Mask

alignment = cos(25°) = 0.906 > 0.866
→ Aligned! (같은 균열의 연장)
```

---

## 4. Point Proximity (점 근접성)

### `--proximity-threshold` (기본값: 0.05m)

**의미:** Mask의 점들 중 몇 %가 Cluster의 점들 가까이에 있는가

**값 범위:** 0.03 ~ 0.1 meters
- `0.03m` = 엄격 (3cm)
- `0.05m` = 권장 (5cm)
- `0.08m` = 느슨 (8cm)

**계산 방식:**
```python
# Mask의 각 점에서 Cluster까지 최소 거리
min_distances = [...]  # 100 points

# proximity_threshold 이내 비율
close_ratio = mean(min_distances < 0.05)

# 예: 80개 점이 5cm 이내
close_ratio = 0.8
```

**Final score에 사용:**
```python
score = 0.2*angle + 0.2*bbox + 0.6*close_ratio
                                    ↑
                            가장 큰 가중치!
```

**사용 사례:**

```bash
# Case 1: 엄격 (정밀 균열용)
--proximity-threshold 0.03
# 효과: 3cm 이내만 "가까움"
# 단점: 약간 떨어진 균열 분리

# Case 2: 권장 (일반적)
--proximity-threshold 0.05
# 효과: 5cm 이내면 가까움
# 권장!

# Case 3: 느슨 (큰 균열용)
--proximity-threshold 0.08
# 효과: 8cm 이내도 OK
# 단점: 과병합 위험
```

---

## 5. Final Score (최종 점수)

### `--overlap-threshold` (기본값: 0.3)

**의미:** 최종 overlap score가 이 값보다 높으면 같은 균열로 병합

**값 범위:** 0.2 ~ 0.5
- `0.25` = 적극적 병합 (중복 최소화)
- `0.3` = 권장 (균형)
- `0.4` = 보수적 (분리 우선)

**Score 계산:**
```python
score = (
    0.2 * angle_similarity +
    0.2 * bbox_iou +
    0.6 * close_ratio
)

if score > overlap_threshold:
    merge()
else:
    create_new_cluster()
```

**예시:**
```python
# 예 1: 같은 방향, 가까움
angle_similarity = 0.95  # 거의 평행
bbox_iou = 0.3           # 약간 겹침
close_ratio = 0.85       # 85% 점이 가까움

score = 0.2*0.95 + 0.2*0.3 + 0.6*0.85 = 0.76
→ 0.76 > 0.3 → Merge!

# 예 2: 다른 방향
angle_similarity = 0.2   # 거의 수직
bbox_iou = 0.5           # 많이 겹침
close_ratio = 0.9        # 90% 가까움

score = 0.2*0.2 + 0.2*0.5 + 0.6*0.9 = 0.68
→ 하지만 angle < 0.707 → Early reject!
```

---

### `score_weights` (코드 내부)

**의미:** Score 계산시 각 요소의 가중치

**기본값:**
```python
score_weights = {
    'angle': 0.2,   # 20% - 방향 유사도
    'bbox': 0.2,    # 20% - BBox IoU
    'point': 0.6    # 60% - Point 근접성 (가장 중요!)
}
```

**조정 방법** (코드 수정):
```python
# Direction을 더 중요하게
config = {
    'score_weights': {
        'angle': 0.4,  # 40%
        'bbox': 0.2,   # 20%
        'point': 0.4   # 40%
    }
}
```

---

## 권장 설정 (Use Cases)

### Case 1: 일반적인 균열 (권장)

```bash
python -m src.cluster_masks_3d_geometric \
  --masks-3d outputs/masks_3d.json \
  --output outputs/mask_clusters.json \
  --min-angle-similarity 0.707 \      # 45도
  --max-centroid-distance 0.5 \       # 50cm
  --use-dynamic-threshold \           # ON
  --gap-threshold 0.1 \               # 10cm
  --proximity-threshold 0.05 \        # 5cm
  --overlap-threshold 0.3             # 균형
```

**효과:**
- ✅ 교차 균열 분리 (45도)
- ✅ 긴 균열 병합 (dynamic)
- ✅ 분할 균열 연결 (gap 10cm)
- ✅ 중복 최소화

---

### Case 2: 중복 최소화 (적극적 병합)

```bash
python -m src.cluster_masks_3d_geometric \
  --masks-3d outputs/masks_3d.json \
  --output outputs/mask_clusters.json \
  --min-angle-similarity 0.5 \        # 60도 (느슨)
  --gap-threshold 0.15 \              # 15cm (느슨)
  --proximity-threshold 0.08 \        # 8cm (느슨)
  --overlap-threshold 0.25            # 낮음 (적극적)
```

**효과:**
- ✅ 중복 균열 적극 병합
- ⚠️ 과병합 위험 (서로 다른 균열도 병합 가능)

---

### Case 3: 보수적 분리 (정밀도 우선)

```bash
python -m src.cluster_masks_3d_geometric \
  --masks-3d outputs/masks_3d.json \
  --output outputs/mask_clusters.json \
  --min-angle-similarity 0.866 \      # 30도 (엄격)
  --max-centroid-distance 0.3 \       # 30cm (작음)
  --gap-threshold 0.05 \              # 5cm (엄격)
  --proximity-threshold 0.03 \        # 3cm (엄격)
  --overlap-threshold 0.4             # 높음 (보수적)
```

**효과:**
- ✅ 다른 균열 확실히 분리
- ⚠️ 같은 균열도 분할될 수 있음

---

## Troubleshooting

### 문제 1: "Too many clusters (중복 많음)"

**증상:**
```
Output clusters: 180
Reduction: 10%
Single-view clusters: 150 (83%)
```

**해결:**
```bash
# 파라미터 느슨하게
--min-angle-similarity 0.5 \        # 0.707 → 0.5
--gap-threshold 0.15 \              # 0.1 → 0.15
--overlap-threshold 0.25            # 0.3 → 0.25
```

---

### 문제 2: "Too few clusters (과병합)"

**증상:**
```
Output clusters: 15
Reduction: 92%
Max masks per cluster: 35
```

**해결:**
```bash
# 파라미터 엄격하게
--min-angle-similarity 0.866 \      # 0.707 → 0.866
--max-centroid-distance 0.3 \       # 0.5 → 0.3
--gap-threshold 0.05 \              # 0.1 → 0.05
--overlap-threshold 0.4             # 0.3 → 0.4
```

---

### 문제 3: "Cross-cracks merged (교차 균열 병합)"

**증상:**
- Visualization에서 수직/수평 균열이 같은 색

**해결:**
```bash
# Angle threshold 엄격하게
--min-angle-similarity 0.866        # 0.707 → 0.866 (30도)
```

---

### 문제 4: "Long crack split (긴 균열 분할)"

**증상:**
- 하나의 긴 균열이 여러 cluster로 분할됨

**해결:**
```bash
# Dynamic threshold 사용
--use-dynamic-threshold \

# Gap threshold 증가
--gap-threshold 0.15                # 0.1 → 0.15

# 또는 static threshold 증가
--max-centroid-distance 0.8         # 0.5 → 0.8
```

---

## 파라미터 우선순위

**조정 순서 (중요도):**

1. **`min_angle_similarity`** ⭐⭐⭐
   - 교차 균열 방지의 핵심
   - 먼저 조정!

2. **`overlap_threshold`** ⭐⭐⭐
   - 최종 병합 결정
   - 중복 vs 분리 조절

3. **`gap_threshold`** ⭐⭐
   - 분할 균열 연결
   - 긴 균열 처리

4. **`proximity_threshold`** ⭐
   - Fine-tuning용
   - Score에 큰 영향

5. **`max_centroid_distance`** ⭐
   - Dynamic ON이면 자동 조정
   - 수동 조정 필요 적음

---

## 요약

### 기본 설정 (대부분 OK)

```bash
--min-angle-similarity 0.707 \      # 45도
--max-centroid-distance 0.5 \       # 50cm (dynamic으로 자동 조정)
--use-dynamic-threshold \           # ON
--gap-threshold 0.1 \               # 10cm
--proximity-threshold 0.05 \        # 5cm
--overlap-threshold 0.3             # 균형
```

### 조정 필요시

1. **중복 많음** → `overlap_threshold` 낮추기 (0.25)
2. **과병합** → `min_angle_similarity` 높이기 (0.866)
3. **긴 균열 분할** → `gap_threshold` 높이기 (0.15)
4. **교차 균열 병합** → `min_angle_similarity` 높이기 (0.866)

---

**디버깅 팁:**
```bash
# DEBUG 모드로 상세 로그 확인
--log-level DEBUG
```

각 mask마다:
- angle_similarity
- centroid_dist
- min_gap
- bbox_iou
- close_ratio
- final_score

확인 가능!
