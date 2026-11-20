# v21 feature-evidence no-shift diagnostic table

| 数据集 | seed | 方法 | Acc变化 | BAcc变化 | Macro-F1变化 | eligible | changed | vote | disagreement | local support |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `openml_31_credit_g` | 1 | `apa_cf_no_shift_diverse_router` | +0.0000 | +0.0000 | +0.0000 | 0.0000 | 0.0000 | 0.8250 | 0.0687 | -0.0564 |
| `openml_31_credit_g` | 1 | `apa_cf_no_shift_clean_router` | +0.0000 | +0.0000 | +0.0000 | 0.0000 | 0.0000 | 0.8333 | 0.0781 | 0.0102 |
| `openml_31_credit_g` | 1 | `apa_cf_no_shift_boundary_router` | +0.0156 | +0.0116 | +0.0137 | 0.0156 | 0.0156 | 0.8490 | 0.0532 | 0.0190 |
| `openml_31_credit_g` | 1 | `apa_cf_dual_mode_router_v17` | +0.0000 | +0.0000 | +0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `openml_881_mv` | 1 | `apa_cf_no_shift_diverse_router` | +0.0000 | +0.0000 | +0.0000 | 0.0000 | 0.0000 | 0.8875 | 0.0573 | -0.0041 |
| `openml_881_mv` | 1 | `apa_cf_no_shift_clean_router` | +0.0000 | +0.0000 | +0.0000 | 0.0000 | 0.0000 | 0.8906 | 0.0543 | -0.0252 |
| `openml_881_mv` | 1 | `apa_cf_no_shift_boundary_router` | +0.0000 | +0.0000 | +0.0000 | 0.0000 | 0.0000 | 0.9375 | 0.0320 | -0.0094 |
| `openml_881_mv` | 1 | `apa_cf_dual_mode_router_v17` | +0.0000 | +0.0000 | +0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `openml_37_diabetes` | 1 | `apa_cf_no_shift_diverse_router` | +0.0156 | +0.0185 | +0.0193 | 0.0156 | 0.0156 | 0.8000 | 0.0628 | -0.0480 |
| `openml_37_diabetes` | 1 | `apa_cf_no_shift_clean_router` | +0.0000 | +0.0000 | +0.0000 | 0.0000 | 0.0000 | 0.8177 | 0.0907 | -0.0229 |
| `openml_37_diabetes` | 1 | `apa_cf_no_shift_boundary_router` | +0.0000 | +0.0000 | +0.0000 | 0.0000 | 0.0000 | 0.8385 | 0.0607 | -0.0475 |
| `openml_37_diabetes` | 1 | `apa_cf_dual_mode_router_v17` | +0.0156 | +0.0185 | +0.0193 | 0.0156 | 0.0156 | 0.0000 | 0.0000 | 0.0000 |
| `openml_37_diabetes` | 2 | `apa_cf_no_shift_diverse_router` | +0.0000 | +0.0000 | +0.0000 | 0.0312 | 0.0312 | 0.8687 | 0.0567 | -0.0147 |
| `openml_37_diabetes` | 2 | `apa_cf_no_shift_boundary_router` | +0.0000 | +0.0000 | +0.0000 | 0.0156 | 0.0000 | 0.8646 | 0.0548 | 0.0027 |
| `openml_37_diabetes` | 2 | `apa_cf_dual_mode_router_v17` | +0.0000 | +0.0000 | +0.0000 | 0.0312 | 0.0312 | 0.0000 | 0.0000 | 0.0000 |
| `openml_151_electricity` | 2 | `apa_cf_no_shift_diverse_router` | +0.0000 | +0.0000 | +0.0000 | 0.0000 | 0.0000 | 0.9594 | 0.0371 | 0.0029 |
| `openml_151_electricity` | 2 | `apa_cf_no_shift_boundary_router` | +0.0000 | +0.0000 | +0.0000 | 0.0000 | 0.0000 | 0.9948 | 0.0090 | -0.0030 |
| `openml_151_electricity` | 2 | `apa_cf_dual_mode_router_v17` | +0.0000 | +0.0000 | +0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `openml_151_electricity` | 1 | `apa_cf_no_shift_diverse_router` | +0.0000 | +0.0000 | +0.0000 | 0.0000 | 0.0000 | 0.9469 | 0.0230 | -0.0047 |
| `openml_151_electricity` | 1 | `apa_cf_no_shift_boundary_router` | +0.0000 | +0.0000 | +0.0000 | 0.0000 | 0.0000 | 0.9635 | 0.0149 | -0.0021 |
| `openml_151_electricity` | 1 | `apa_cf_dual_mode_router_v17` | +0.0000 | +0.0000 | +0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## summary

| 方法 | 组数 | 平均Acc变化 | Acc正/负/零 | 平均BAcc变化 | 平均Macro-F1变化 |
|---|---:|---:|---:|---:|---:|
| `apa_cf_dual_mode_router_v17` | 6 | +0.0026 | 1/0/5 | +0.0031 | +0.0032 |
| `apa_cf_no_shift_boundary_router` | 6 | +0.0026 | 1/0/5 | +0.0019 | +0.0023 |
| `apa_cf_no_shift_clean_router` | 3 | +0.0000 | 0/0/3 | +0.0000 | +0.0000 |
| `apa_cf_no_shift_diverse_router` | 6 | +0.0026 | 1/0/5 | +0.0031 | +0.0032 |
