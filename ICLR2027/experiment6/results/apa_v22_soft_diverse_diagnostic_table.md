# v22 soft-diverse diagnostic table

| 数据集 | seed | 方法 | Acc变化 | BAcc变化 | Macro-F1变化 | eligible | changed |
|---|---:|---|---:|---:|---:|---:|---:|
| `openml_31_credit_g` | 1 | `apa_cf_no_shift_diverse_router` | +0.0000 | +0.0000 | +0.0000 | 0.0000 | 0.0000 |
| `openml_31_credit_g` | 1 | `apa_cf_no_shift_soft_diverse_router` | +0.0000 | +0.0000 | +0.0000 | 0.0000 | 0.0000 |
| `openml_31_credit_g` | 1 | `apa_cf_dual_mode_router_v17` | +0.0000 | +0.0000 | +0.0000 | 0.0000 | 0.0000 |
| `openml_31_credit_g` | 2 | `apa_cf_no_shift_diverse_router` | +0.0000 | +0.0000 | +0.0000 | 0.0000 | 0.0000 |
| `openml_31_credit_g` | 2 | `apa_cf_no_shift_soft_diverse_router` | +0.0000 | +0.0000 | +0.0000 | 0.0312 | 0.0000 |
| `openml_31_credit_g` | 2 | `apa_cf_dual_mode_router_v17` | +0.0000 | +0.0000 | +0.0000 | 0.0000 | 0.0000 |
| `openml_881_mv` | 1 | `apa_cf_no_shift_diverse_router` | +0.0000 | +0.0000 | +0.0000 | 0.0000 | 0.0000 |
| `openml_881_mv` | 1 | `apa_cf_no_shift_soft_diverse_router` | +0.0000 | +0.0000 | +0.0000 | 0.0000 | 0.0000 |
| `openml_881_mv` | 1 | `apa_cf_dual_mode_router_v17` | +0.0000 | +0.0000 | +0.0000 | 0.0000 | 0.0000 |
| `openml_881_mv` | 2 | `apa_cf_no_shift_diverse_router` | +0.0000 | +0.0000 | +0.0000 | 0.0000 | 0.0000 |
| `openml_881_mv` | 2 | `apa_cf_no_shift_soft_diverse_router` | +0.0000 | +0.0000 | +0.0000 | 0.0000 | 0.0000 |
| `openml_881_mv` | 2 | `apa_cf_dual_mode_router_v17` | +0.0000 | +0.0000 | +0.0000 | 0.0000 | 0.0000 |
| `openml_37_diabetes` | 1 | `apa_cf_no_shift_diverse_router` | +0.0156 | +0.0185 | +0.0193 | 0.0156 | 0.0156 |
| `openml_37_diabetes` | 1 | `apa_cf_no_shift_soft_diverse_router` | +0.0000 | +0.0000 | +0.0000 | 0.0156 | 0.0000 |
| `openml_37_diabetes` | 1 | `apa_cf_dual_mode_router_v17` | +0.0156 | +0.0185 | +0.0193 | 0.0156 | 0.0156 |
| `openml_37_diabetes` | 2 | `apa_cf_no_shift_diverse_router` | +0.0000 | +0.0000 | +0.0000 | 0.0312 | 0.0312 |
| `openml_37_diabetes` | 2 | `apa_cf_no_shift_soft_diverse_router` | +0.0312 | +0.0339 | +0.0359 | 0.0312 | 0.0312 |
| `openml_37_diabetes` | 2 | `apa_cf_dual_mode_router_v17` | +0.0000 | +0.0000 | +0.0000 | 0.0312 | 0.0312 |
| `openml_151_electricity` | 1 | `apa_cf_no_shift_diverse_router` | +0.0000 | +0.0000 | +0.0000 | 0.0000 | 0.0000 |
| `openml_151_electricity` | 1 | `apa_cf_no_shift_soft_diverse_router` | +0.0000 | +0.0000 | +0.0000 | 0.0000 | 0.0000 |
| `openml_151_electricity` | 1 | `apa_cf_dual_mode_router_v17` | +0.0000 | +0.0000 | +0.0000 | 0.0000 | 0.0000 |
| `openml_151_electricity` | 2 | `apa_cf_no_shift_diverse_router` | +0.0000 | +0.0000 | +0.0000 | 0.0000 | 0.0000 |
| `openml_151_electricity` | 2 | `apa_cf_no_shift_soft_diverse_router` | +0.0000 | +0.0000 | +0.0000 | 0.0000 | 0.0000 |
| `openml_151_electricity` | 2 | `apa_cf_dual_mode_router_v17` | +0.0000 | +0.0000 | +0.0000 | 0.0000 | 0.0000 |

## summary

| 方法 | 组数 | 平均Acc变化 | Acc正/负/零 | 平均BAcc变化 | 平均Macro-F1变化 |
|---|---:|---:|---:|---:|---:|
| `apa_cf_dual_mode_router_v17` | 8 | +0.0020 | 1/0/7 | +0.0023 | +0.0024 |
| `apa_cf_no_shift_diverse_router` | 8 | +0.0020 | 1/0/7 | +0.0023 | +0.0024 |
| `apa_cf_no_shift_soft_diverse_router` | 8 | +0.0039 | 1/0/7 | +0.0042 | +0.0045 |
