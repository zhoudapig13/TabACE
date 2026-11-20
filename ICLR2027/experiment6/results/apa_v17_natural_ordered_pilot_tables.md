# apa_v17_natural_ordered_pilot

Ordered split on original public data. No artificial label-ratio intervention is applied.

## 方法汇总

| 方法 | 组数 | Acc | BAcc | Macro-F1 | Acc变化 | BAcc变化 | Macro-F1变化 | ECE变化 | BAcc正/负组 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `apa_cf_directional_router` | 15 | 0.8625 | 0.7799 | 0.7806 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0 / 0 |
| `apa_cf_natural_guard_router` | 15 | 0.8625 | 0.7799 | 0.7806 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0 / 0 |
| `tabicl_baseline` | 15 | 0.8625 | 0.7799 | 0.7806 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0 / 0 |

## 数据集 x 方法

| 数据集 | 方法 | 组数 | Acc变化 | BAcc变化 | Macro-F1变化 | ECE变化 | BAcc正/负组 |
|---|---|---:|---:|---:|---:|---:|---:|
| `openml_151_electricity` | `apa_cf_directional_router` | 3 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0 / 0 |
| `openml_151_electricity` | `apa_cf_natural_guard_router` | 3 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0 / 0 |
| `openml_179_adult` | `apa_cf_directional_router` | 3 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0 / 0 |
| `openml_179_adult` | `apa_cf_natural_guard_router` | 3 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0 / 0 |
| `openml_37_diabetes` | `apa_cf_directional_router` | 3 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0 / 0 |
| `openml_37_diabetes` | `apa_cf_natural_guard_router` | 3 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0 / 0 |
| `openml_44_spambase` | `apa_cf_directional_router` | 3 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0 / 0 |
| `openml_44_spambase` | `apa_cf_natural_guard_router` | 3 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0 / 0 |
| `openml_46910_bank_marketing` | `apa_cf_directional_router` | 3 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0 / 0 |
| `openml_46910_bank_marketing` | `apa_cf_natural_guard_router` | 3 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0 / 0 |
