# WHI Cox replication (mini-EWAS moderate CpGs)

Per-CpG Cox PH models in FHS and WHI:

    Surv(time, event) ~ methylation + age + sex

Input: `discovered_cpg_markers_moderate.csv` (1,262 CpGs)

Replication criterion (nominal): FHS p<0.05 AND WHI p<0.05 AND concordant Cox direction.

Results:
- CpGs tested: 1,262
- FHS fitted: 1,262 / 1,262
- WHI fitted: 1,260 / 1,262
- Nominal replications: **8 / 1,262**

Replicated CpGs:
| CpG | Gene | Class | FHS p | WHI p |
|-----|------|-------|-------|-------|
| cg05703009 | OR52N2;TRIM5 | accelerated | 0.034 | 0.024 |
| cg10965178 | TIE1 | decelerated | 1.1e-05 | 0.028 |
| cg03791579 | ACOX3 | decelerated | 0.004 | 0.029 |
| cg07581395 | CTD-2655K5.1 | decelerated | 8.3e-06 | 0.035 |
| cg25890838 | PDCD1 | accelerated | 0.003 | 0.040 |
| cg24363955 | NPR3 | accelerated | 0.023 | 0.042 |
| cg05873267 | POMT2 | decelerated | 0.0004 | 0.044 |
| cg02867514 | CCL5 | accelerated | 0.015 | 0.046 |

Files:
- `fhs_cpg_cox_results.csv`, `whi_cpg_cox_results.csv`
- `cpg_cox_replication.csv`, `cpg_cox_replication_summary.json`
- `manuscript_table_whi_replication_moderate_cpgs.csv/.md`
- `manuscript_table_whi_replicated_cpgs_only.csv`

Scripts:
- `gwas/run_fhs_cpg_cox.py`
- `run_cpg_cox_replication.py` (orchestrator)
- `make_manuscript_table.py`

Compare to SNP moderate tier: 2 / 97 nominal replications.
