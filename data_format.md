# Data Format Spec (모델팀 ↔ 그래프팀)

이 문서는 모델 코드(`train.py`, `dataset.py`)가 기대하는 파일 형식을 정의합니다.
모든 파일은 `data_dir/` 한 폴더 아래 위치.

## Node ID 규칙 (Critical)

코드는 두 개의 카운트를 분리해서 사용합니다:

- **`num_total_nodes`** = 8298. 그래프 모든 노드. Embedding table 크기.
- **`num_ingredients`** = 6653. Substitution 후보로 쓰일 수 있는 ingredient 노드만.

ID 할당 권장:

| 범위 | 의미 | 개수 |
|---|---|---|
| `0 ~ 6652` | Ingredient nodes | 6,653 |
| `6653 ~ 8213` | Flavor compound nodes (F) | 1,561 |
| `8214 ~ 8297` | Drug compound nodes (D) | 84 |

→ recipes / pairs / usda_mapping에 등장하는 id는 **반드시 0~6652 범위**.
→ flavorgraph_edges는 0~8297 범위 모두 가능.

## 1. `flavorgraph_edges.csv` (필수)

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `src_id` | int | source 노드 id (0~8297) |
| `dst_id` | int | destination 노드 id (0~8297) |
| `weight` | float | 엣지 weight (I-I는 NPMI ∈ [0.25, 1], I-F/I-D는 1.0) |
| `edge_type` | str (optional) | "I-I", "I-F", "I-D" — 디버깅용 |

**무방향 엣지 1줄당 1줄 작성**. 코드가 자동으로 `(src,dst)`와 `(dst,src)` 둘 다 추가.

## 2. `recipes.json` (필수)

```json
{
  "12345": [101, 203, 415, ...],
  "12346": [...],
  ...
}
```

- key: recipe_id (str로 저장 OK, 코드에서 int 변환)
- value: ingredient_id 리스트 (**0~6652 범위만**)
- **source ingredient도 포함되어야 함** (substitution 전 원본)

## 3. `pairs_train.csv`, `pairs_val.csv`, `pairs_test.csv` (필수)

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `source_id` | int | 대체될 ingredient id (**0~6652**) |
| `target_id` | int | 대체할 ingredient id (**0~6652**) |
| `recipe_id` | int | substitution이 valid한 recipe id |

기대 크기: train 49044 / val 10729 / test 10747 (GISMo 기준).

## 4. `usda_mapping.json` (MVP에만 필수)

```json
{
  "0": {"sugar_g": 0.5, "sodium_mg": 1.2, "calorie_kcal": 80, "protein_g": 0.2},
  "1": {"sugar_g": 12.3, "sodium_mg": 380, "calorie_kcal": 250, "protein_g": 5.0},
  ...
}
```

- key: ingredient_id (str OK), **0~6652만**
- value: 100g 기준 영양값 (raw, 정규화 전)
- **단위 고정**: `sugar_g` = grams, `sodium_mg` = milligrams
- **누락 ingredient는 entry 자체를 빼주세요** (drop 정책)
- 매핑 안 된 ingredient는 학습에서 g=[0,0]으로 처리

`calorie_kcal`, `protein_g` 등 추가 영양소도 있으면 좋음 — 미래 확장용. MVP는 sugar + sodium만.

## 5. (선택) `ingredients.csv`

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `id` | int | 노드 id (0~6652) |
| `name` | str | "garlic", "olive_oil" 등 |

학습/평가에는 미사용. Case study에서 id를 이름으로 변환할 때.

---

## Sanity Check 스크립트 (산출물 받을 때 한 번 돌리기)

```python
import pandas as pd, json

# 1. Edge counts (대략 147K 기대)
edges = pd.read_csv('flavorgraph_edges.csv')
print(f'Total edges: {len(edges)}')
if 'edge_type' in edges.columns:
    print(edges['edge_type'].value_counts())
print(f'Max src id: {edges["src_id"].max()}')
print(f'Max dst id: {edges["dst_id"].max()}  (should be < 8298)')

# 2. Recipe coverage
with open('recipes.json') as f:
    recipes = json.load(f)
print(f'Recipes: {len(recipes)}')
all_recipe_ings = set()
for r in recipes.values():
    all_recipe_ings.update(r)
print(f'Max ingredient id in recipes: {max(all_recipe_ings)}  (should be < 6653)')

# 3. Pairs
for split in ['train', 'val', 'test']:
    p = pd.read_csv(f'pairs_{split}.csv')
    print(f'{split}: {len(p)} pairs')
    assert p['source_id'].max() < 6653
    assert p['target_id'].max() < 6653

# 4. USDA coverage
with open('usda_mapping.json') as f:
    usda = json.load(f)
print(f'USDA mapped: {len(usda)} / 6653 ingredients ({100*len(usda)/6653:.1f}%)')
# 70% 이상 권장. 30% 이상 누락이면 매핑 알고리즘 재검토.

# 5. USDA value sanity
import numpy as np
sugars = [v['sugar_g'] for v in usda.values()]
sodiums = [v['sodium_mg'] for v in usda.values()]
print(f'sugar:  min={min(sugars):.1f}  max={max(sugars):.1f}  median={np.median(sugars):.1f}')
print(f'sodium: min={min(sodiums):.1f}  max={max(sodiums):.1f}  median={np.median(sodiums):.1f}')
# 음수, NaN, 비현실적인 값 (e.g., 1000g sugar) 없는지 확인
```
