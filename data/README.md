# Data

## Included

| File | Content |
| --- | --- |
| `solvent_db_1_clean.csv` | In-house MD dataset after cleaning: 268 compositions (31 solvents x LiFSI/LiPF6/LiTFSI x salt:solvent 1:2, 1:3, 1:5), ionic conductivity at 363 K and molecular descriptors |
| `solvent_db_changelog.csv` | Every change made to the original table during cleaning |
| `folds_inhouse.csv` | Fold assignments shared by every model: random (three shuffles), leave-solvent-out, leave-class-out, leave-ethers-out |

Ionic conductivity (`Ion_Con`) is stored in S/m; the code converts it to mS/cm.

MD details: GROMACS 2020.6; OPLS-AA for solvents and CL&P for Li+ and anions, ionic charges
scaled by 0.8; ~5 x 5 x 5 nm boxes, NPT equilibration, NVT production at 363 K, final 10 ns
analysed; conductivity from the Einstein relation including cation-anion cross terms.

## Downloaded separately

These datasets are published by others; download them into this folder.

| File | Source | Used for |
| --- | --- | --- |
| `oedb-electrolytes-v2026-05-11_1.csv` | OEDB, https://oedb.jp (Nakamura et al., *npj Comput. Mater.* 2026, 12, 215) | OEDB MD data |
| `oedb-experimental-v2026-05-11.csv` | OEDB experimental set (Raman spectra) | Raman analysis |
| `Ref_LCEdata.xlsx` | Supporting information of Sharma et al., *J. Chem. Inf. Model.* 2023, 63, 6998 (file `ci3c01030_si_004.xlsx`); data of Kim et al., *PNAS* 2023, 120, e2214357120 | Coulombic-efficiency task |
| `HOMO_LUMO_data.xlsx` | Supporting information of Sharma et al. 2023 (file `ci3c01030_si_002.xlsx`) | Molecular pretraining |

## Generated

```bash
python scripts/compare_oedb.py --base data             # OEDB QC flags and in-house/OEDB overlap
python scripts/build_unified_and_folds.py --base data  # unified_md.csv (and folds_inhouse.csv)
python scripts/build_kim_test_set.py --out data/LCE_test13.csv   # 13 external CE electrolytes
```

`build_unified_and_folds.py` regenerates `folds_inhouse.csv` deterministically; the copy
included here is the one used for all results in the paper.
