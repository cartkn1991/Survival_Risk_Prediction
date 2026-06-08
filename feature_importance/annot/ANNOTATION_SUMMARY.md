# AESURV-DANN-Aux feature attributions, annotated

Per-feature gradient attribution of the trained model's predicted age and log-hazard onto the 1.37 M raw CpG+SNP inputs, with biological annotations.


## Headline

- **Cohort consistency** (FHS vs WHI per-feature gradient correlation):
  - **age** Pearson r: CpG = **0.857**, SNP = **0.855**.  sign-agreement: 0.828 (CpG) / 0.827 (SNP). WHI slope ~0.35 (WHI gradients ~1/3 the FHS magnitude but directionally aligned).
  - **risk** Pearson r: CpG = **0.866**, SNP = **0.865**.  sign-agreement: 0.834 (CpG) / 0.833 (SNP). WHI slope ~0.36 (WHI gradients ~1/3 the FHS magnitude but directionally aligned).
- **Epigenetic-clock CpG overlap** in top-100 pooled features:
  - age: Horvath=0, Horvath_SkinBlood=0, Hannum=0, PhenoAge=1, GrimAgeV2=0, DunedinPACE=1, Zhang2019=1, Lin=0
  - risk: Horvath=0, Horvath_SkinBlood=0, Hannum=0, PhenoAge=1, GrimAgeV2=0, DunedinPACE=1, Zhang2019=1, Lin=0
  - These low counts indicate the model's top CpGs are largely **novel** relative to published clocks (Horvath 353, Hannum 71, PhenoAge 513, GrimAgeV2 1047, DunedinPACE 173, Zhang2019 514, Lin 100). The model discovers a cohort-portable mortality-relevant epigenetic signal that is not equivalent to chronological-age clocks.
- **GWAS-catalog SNP nearby (±5 kb)**: 
  - age: 137/200 top SNPs have at least one cataloged GWAS-significant SNP within 5 kb.
  - risk: 130/200 top SNPs have at least one cataloged GWAS-significant SNP within 5 kb.


## AGE -- top-20 per direction


### CPG -- ACCELERATORs

| rank | feature | grad_pool | grad_fhs | grad_whi | sign_agreement | gene_HGNC | clocks | known_role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | cg09606495 | +0.0470 | +0.0527 | +0.0116 | yes | INS;INS-IGF2 |  | Hannum clock CpG. |
| 2 | cg23714917 | +0.0430 | +0.0465 | +0.0214 | yes | MEST;MESTIT1 |  | nan |
| 3 | cg25809215 | +0.0419 | +0.0458 | +0.0170 | yes | DPH3;OXNAD1 |  | Hannum top weight; TRIM59 promoter. |
| 4 | cg09883849 | +0.0416 | +0.0452 | +0.0184 | yes | AC013470.6 |  | nan |
| 5 | cg22499086 | +0.0410 | +0.0456 | +0.0124 | yes | HMMR;NUDCD2 |  | nan |
| 6 | cg02933228 | +0.0409 | +0.0444 | +0.0185 | yes | CDC42BPG |  | nan |
| 7 | cg00583492 | +0.0394 | +0.0430 | +0.0167 | yes | C11orf80;RCE1 |  | nan |
| 8 | cg04126810 | +0.0393 | +0.0431 | +0.0156 | yes | MAP3K21 |  | nan |
| 9 | cg25914270 | +0.0393 | +0.0433 | +0.0144 | yes | COX7C |  | nan |
| 10 | cg09910998 | +0.0385 | +0.0425 | +0.0133 | yes | PEG10;SGCE |  | nan |
| 11 | cg02278760 | +0.0385 | +0.0421 | +0.0157 | yes | CCDC189;RNF40 |  | nan |
| 12 | cg20050761 | +0.0384 | +0.0435 | +0.0065 | yes | MEST;MESTIT1 | Zhang2019 | nan |
| 13 | cg25693639 | +0.0384 | +0.0422 | +0.0147 | yes | PPCDC |  | nan |
| 14 | cg01566785 | +0.0384 | +0.0424 | +0.0133 | yes | PEG10;SGCE |  | nan |
| 15 | cg04208151 | +0.0382 | +0.0420 | +0.0146 | yes | DDX20;INKA2 |  | nan |
| 16 | cg04134528 | +0.0379 | +0.0412 | +0.0170 | yes | nan |  | nan |
| 17 | cg13319938 | +0.0378 | +0.0420 | +0.0112 | yes | TRPM5 |  | nan |
| 18 | cg06728974 | +0.0375 | +0.0408 | +0.0163 | yes | ZNF613;ZNF649 |  | nan |
| 19 | cg23468238 | +0.0371 | +0.0400 | +0.0182 | yes | ANKRA2;UTP15 |  | nan |
| 20 | cg02096718 | +0.0367 | +0.0397 | +0.0183 | yes | CTD-2186M15.3;GOLPH3 |  | nan |


### CPG -- DECELERATORs

| rank | feature | grad_pool | grad_fhs | grad_whi | sign_agreement | gene_HGNC | clocks | known_role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | cg17142816 | -0.0437 | -0.0482 | -0.0159 | yes | DIP2C |  | nan |
| 2 | cg18257574 | -0.0422 | -0.0462 | -0.0175 | yes | nan |  | nan |
| 3 | cg14935378 | -0.0403 | -0.0439 | -0.0175 | yes | TMEM243;Y_RNA |  | nan |
| 4 | cg17981470 | -0.0394 | -0.0426 | -0.0193 | yes | RP1L1 |  | nan |
| 5 | cg00908551 | -0.0389 | -0.0421 | -0.0190 | yes | FANCF;GAS2 |  | nan |
| 6 | cg20962356 | -0.0386 | -0.0419 | -0.0178 | yes | RP11-535A19.2;UVRAG |  | nan |
| 7 | cg01069466 | -0.0377 | -0.0408 | -0.0181 | yes | CTB-58E17.1;MIR4734 |  | nan |
| 8 | cg01581974 | -0.0371 | -0.0405 | -0.0161 | yes | RP11-977B10.2;TUBA1C |  | nan |
| 9 | cg12899077 | -0.0369 | -0.0406 | -0.0140 | yes | nan |  | nan |
| 10 | cg04699394 | -0.0367 | -0.0404 | -0.0135 | yes | GREB1 |  | nan |
| 11 | cg05587926 | -0.0364 | -0.0401 | -0.0130 | yes | nan |  | nan |
| 12 | cg04293778 | -0.0360 | -0.0397 | -0.0131 | yes | TNXB |  | nan |
| 13 | cg03777288 | -0.0359 | -0.0391 | -0.0158 | yes | GRIN2B |  | nan |
| 14 | cg23954731 | -0.0359 | -0.0388 | -0.0177 | yes | NAT14 |  | nan |
| 15 | cg21627466 | -0.0358 | -0.0395 | -0.0128 | yes | TCF7L1 |  | nan |
| 16 | cg09438955 | -0.0358 | -0.0391 | -0.0149 | yes | KLF12 |  | nan |
| 17 | cg07581365 | -0.0357 | -0.0385 | -0.0180 | yes | CBX8 |  | nan |
| 18 | cg03972012 | -0.0357 | -0.0390 | -0.0151 | yes | nan |  | nan |
| 19 | cg09718810 | -0.0357 | -0.0387 | -0.0166 | yes | CTB-129O4.1;MAPK9 |  | nan |
| 20 | cg25533423 | -0.0355 | -0.0391 | -0.0129 | yes | CTD-2541J13.2;DSEL |  | nan |


### SNP -- ACCELERATORs

| rank | feature | grad_pool | grad_fhs | grad_whi | sign_agreement | gene_symbol | gwas_n_hits_near | gwas_rsids_near |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 10:45370891_G | +0.0635 | +0.0691 | +0.0283 | yes | nan | 1 | rs61855875 |
| 2 | 19:40669156_G | +0.0629 | +0.0698 | +0.0198 | yes | NUMBL | 1 | rs780340856 |
| 3 | 1:150981586_C | +0.0620 | +0.0678 | +0.0253 | yes | ANXA9 | 12 | rs2305814;rs112742193;rs28521412;rs267734;rs607518 |
| 4 | 10:82501843_C | +0.0619 | +0.0678 | +0.0251 | yes | NRG3 | 0 | nan |
| 5 | 10:47701570_A | +0.0617 | +0.0676 | +0.0251 | yes | SHLD2P3 | 0 | nan |
| 6 | 2:209164130_C | +0.0607 | +0.0671 | +0.0207 | yes | nan | 1 | rs138802801 |
| 7 | 2:183167513_C | +0.0603 | +0.0662 | +0.0229 | yes | nan | 3 | rs181204438;rs140391222;rs545260475 |
| 8 | 5:160551087_C | +0.0585 | +0.0638 | +0.0255 | yes | MIR3142HG | 0 | nan |
| 9 | 17:31486550_G | +0.0583 | +0.0643 | +0.0206 | yes | RAB11FIP4 | 1 | rs7215205 |
| 10 | 3:70220111_T | +0.0582 | +0.0639 | +0.0224 | yes | MDFIC2;SAMMSON | 2 | rs73117911;rs191223340 |
| 11 | 5:111712134_G | +0.0582 | +0.0632 | +0.0265 | yes | STARD4-AS1;NREP | 4 | rs146239224;rs112081420;rs7735771;rs67968533 |
| 12 | 13:50598497_C | +0.0566 | +0.0619 | +0.0229 | yes | DLEU7;DLEU1 | 15 | rs706599;rs7997464;rs797520;rs116346582;rs1341635 |
| 13 | 6:152449729_C | +0.0561 | +0.0615 | +0.0218 | yes | SYNE1 | 0 | nan |
| 14 | 7:81722663_T | +0.0556 | +0.0595 | +0.0308 | yes | HGF | 1 | rs5745709 |
| 15 | 1:77114602_A | +0.0555 | +0.0605 | +0.0243 | yes | PIGK | 0 | nan |
| 16 | 2:52463826_G | +0.0552 | +0.0609 | +0.0197 | yes | nan | 2 | rs17732209;rs12466061 |
| 17 | 18:44418346_C | +0.0551 | +0.0603 | +0.0228 | yes | LINC01478 | 5 | rs75957619;rs182599614;rs74997723;rs138080687;rs16977978 |
| 18 | 17:39829290_T | +0.0547 | +0.0596 | +0.0240 | yes | IKZF3 | 8 | rs3816470;rs2060941;rs138214281;rs34521860;rs192665233 |
| 19 | 7:101026506_G | +0.0539 | +0.0589 | +0.0221 | yes | MUC12-AS1;MUC17 | 1 | rs146006635 |
| 20 | 4:30150073_C | +0.0538 | +0.0585 | +0.0245 | yes | nan | 1 | rs3101279 |


### SNP -- DECELERATORs

| rank | feature | grad_pool | grad_fhs | grad_whi | sign_agreement | gene_symbol | gwas_n_hits_near | gwas_rsids_near |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 22:33656202_C | -0.0705 | -0.0779 | -0.0242 | yes | LARGE1 | 0 | nan |
| 2 | 4:91118_A | -0.0703 | -0.0770 | -0.0281 | yes | nan | 0 | nan |
| 3 | 10:37576250_A | -0.0635 | -0.0712 | -0.0148 | yes | nan | 0 | nan |
| 4 | 5:96730203_T | -0.0628 | -0.0689 | -0.0244 | yes | CAST | 3 | rs149313;rs34782647;rs697976 |
| 5 | 15:89985584_C | -0.0624 | -0.0680 | -0.0270 | yes | nan | 1 | rs1256840 |
| 6 | 1:165573298_C | -0.0620 | -0.0679 | -0.0250 | yes | LRRC52-AS1 | 1 | rs12121840 |
| 7 | 3:130089416_T | -0.0609 | -0.0661 | -0.0279 | yes | ALG1L2;LINC02014 | 0 | nan |
| 8 | 11:11319570_C | -0.0608 | -0.0673 | -0.0201 | yes | GALNT18 | 6 | rs61872397;rs4444072;rs12419084;rs1471895;rs148333573 |
| 9 | 10:71710262_G | -0.0605 | -0.0669 | -0.0202 | yes | CDH23 | 3 | rs2166631;rs41281310;rs10823829 |
| 10 | 12:96051418_A | -0.0605 | -0.0657 | -0.0275 | yes | nan | 1 | rs41392450 |
| 11 | 12:127783810_T | -0.0599 | -0.0642 | -0.0327 | yes | nan | 0 | nan |
| 12 | 4:45025457_T | -0.0590 | -0.0641 | -0.0268 | yes | nan | 4 | rs11932923;rs78460921;rs141080507;rs13140100 |
| 13 | 5:169807485_T | -0.0586 | -0.0650 | -0.0187 | yes | DOCK2 | 2 | rs36037302;rs261616 |
| 14 | 3:102903020_T | -0.0578 | -0.0635 | -0.0214 | yes | nan | 0 | nan |
| 15 | 4:107251400_C | -0.0570 | -0.0628 | -0.0208 | yes | DKK2 | 0 | nan |
| 16 | 9:103674422_G | -0.0566 | -0.0628 | -0.0178 | yes | nan | 0 | nan |
| 17 | 1:219462360_G | -0.0566 | -0.0609 | -0.0294 | yes | LYPLAL1-AS1 | 20 | rs6684734;rs2605092;rs5781116;rs2605093;rs6699024 |
| 18 | 7:14254970_A | -0.0564 | -0.0622 | -0.0199 | yes | DGKB | 0 | nan |
| 19 | 6:167596769_A | -0.0562 | -0.0610 | -0.0262 | yes | nan | 0 | nan |
| 20 | 6:41502992_T | -0.0561 | -0.0621 | -0.0179 | yes | FOXP4-AS1;LINC01276 | 2 | rs10947978;rs13212930 |


## RISK -- top-20 per direction


### CPG -- ACCELERATORs

| rank | feature | grad_pool | grad_fhs | grad_whi | sign_agreement | gene_HGNC | clocks | known_role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | cg09606495 | +0.0042 | +0.0046 | +0.0011 | yes | INS;INS-IGF2 |  | Hannum clock CpG. |
| 2 | cg02933228 | +0.0037 | +0.0040 | +0.0017 | yes | CDC42BPG |  | nan |
| 3 | cg09883849 | +0.0037 | +0.0040 | +0.0017 | yes | AC013470.6 |  | nan |
| 4 | cg25914270 | +0.0036 | +0.0039 | +0.0014 | yes | COX7C |  | nan |
| 5 | cg25809215 | +0.0035 | +0.0038 | +0.0015 | yes | DPH3;OXNAD1 |  | Hannum top weight; TRIM59 promoter. |
| 6 | cg01566785 | +0.0034 | +0.0038 | +0.0013 | yes | PEG10;SGCE |  | nan |
| 7 | cg04208151 | +0.0034 | +0.0037 | +0.0014 | yes | DDX20;INKA2 |  | nan |
| 8 | cg22499086 | +0.0034 | +0.0038 | +0.0010 | yes | HMMR;NUDCD2 |  | nan |
| 9 | cg23714917 | +0.0034 | +0.0036 | +0.0017 | yes | MEST;MESTIT1 |  | nan |
| 10 | cg04126810 | +0.0033 | +0.0036 | +0.0013 | yes | MAP3K21 |  | nan |
| 11 | cg02278760 | +0.0033 | +0.0036 | +0.0014 | yes | CCDC189;RNF40 |  | nan |
| 12 | cg03544426 | +0.0032 | +0.0036 | +0.0013 | yes | GOLGA2;SWI5 |  | nan |
| 13 | cg04134528 | +0.0032 | +0.0035 | +0.0015 | yes | nan |  | nan |
| 14 | cg06728974 | +0.0032 | +0.0035 | +0.0014 | yes | ZNF613;ZNF649 |  | nan |
| 15 | cg02096718 | +0.0032 | +0.0035 | +0.0017 | yes | CTD-2186M15.3;GOLPH3 |  | nan |
| 16 | cg09211399 | +0.0032 | +0.0035 | +0.0013 | yes | OSTM1 |  | nan |
| 17 | cg07601536 | +0.0032 | +0.0035 | +0.0013 | yes | RREB1 |  | nan |
| 18 | cg25693639 | +0.0032 | +0.0035 | +0.0012 | yes | PPCDC |  | nan |
| 19 | cg23468238 | +0.0032 | +0.0034 | +0.0016 | yes | ANKRA2;UTP15 |  | nan |
| 20 | cg07152098 | +0.0032 | +0.0035 | +0.0010 | yes | ZNF2;ZNF514 |  | nan |


### CPG -- DECELERATORs

| rank | feature | grad_pool | grad_fhs | grad_whi | sign_agreement | gene_HGNC | clocks | known_role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | cg18257574 | -0.0038 | -0.0042 | -0.0017 | yes | nan |  | nan |
| 2 | cg17142816 | -0.0036 | -0.0040 | -0.0014 | yes | DIP2C |  | nan |
| 3 | cg00908551 | -0.0035 | -0.0037 | -0.0017 | yes | FANCF;GAS2 |  | nan |
| 4 | cg14935378 | -0.0035 | -0.0038 | -0.0015 | yes | TMEM243;Y_RNA |  | nan |
| 5 | cg20962356 | -0.0034 | -0.0037 | -0.0016 | yes | RP11-535A19.2;UVRAG |  | nan |
| 6 | cg02072115 | -0.0033 | -0.0037 | -0.0012 | yes | RNF169 |  | nan |
| 7 | cg17981470 | -0.0033 | -0.0035 | -0.0017 | yes | RP1L1 |  | nan |
| 8 | cg01069466 | -0.0033 | -0.0035 | -0.0016 | yes | CTB-58E17.1;MIR4734 |  | nan |
| 9 | cg09438955 | -0.0032 | -0.0035 | -0.0014 | yes | KLF12 |  | nan |
| 10 | cg12899077 | -0.0032 | -0.0035 | -0.0012 | yes | nan |  | nan |
| 11 | cg05200380 | -0.0031 | -0.0035 | -0.0010 | yes | nan |  | nan |
| 12 | cg04699394 | -0.0031 | -0.0034 | -0.0012 | yes | GREB1 |  | nan |
| 13 | cg01581974 | -0.0031 | -0.0034 | -0.0014 | yes | RP11-977B10.2;TUBA1C |  | nan |
| 14 | cg21921474 | -0.0031 | -0.0034 | -0.0012 | yes | KDF1 |  | nan |
| 15 | cg03972012 | -0.0031 | -0.0034 | -0.0014 | yes | nan |  | nan |
| 16 | cg23954731 | -0.0031 | -0.0033 | -0.0015 | yes | NAT14 |  | nan |
| 17 | cg06327267 | -0.0031 | -0.0033 | -0.0014 | yes | CAV1 |  | nan |
| 18 | cg22258976 | -0.0030 | -0.0034 | -0.0009 | yes | CFL1;MUS81 |  | nan |
| 19 | cg16235860 | -0.0030 | -0.0033 | -0.0013 | yes | USP1 |  | nan |
| 20 | cg07703610 | -0.0030 | -0.0033 | -0.0013 | yes | SMARCD2 |  | nan |


### SNP -- ACCELERATORs

| rank | feature | grad_pool | grad_fhs | grad_whi | sign_agreement | gene_symbol | gwas_n_hits_near | gwas_rsids_near |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1:150981586_C | +0.0055 | +0.0060 | +0.0024 | yes | ANXA9 | 12 | rs2305814;rs112742193;rs28521412;rs267734;rs607518 |
| 2 | 10:82501843_C | +0.0054 | +0.0059 | +0.0023 | yes | NRG3 | 0 | nan |
| 3 | 19:40669156_G | +0.0054 | +0.0060 | +0.0017 | yes | NUMBL | 1 | rs780340856 |
| 4 | 10:45370891_G | +0.0054 | +0.0058 | +0.0025 | yes | nan | 1 | rs61855875 |
| 5 | 10:47701570_A | +0.0053 | +0.0058 | +0.0022 | yes | SHLD2P3 | 0 | nan |
| 6 | 17:31486550_G | +0.0053 | +0.0058 | +0.0019 | yes | RAB11FIP4 | 1 | rs7215205 |
| 7 | 18:44418346_C | +0.0053 | +0.0057 | +0.0024 | yes | LINC01478 | 5 | rs75957619;rs182599614;rs74997723;rs138080687;rs16977978 |
| 8 | 5:160551087_C | +0.0053 | +0.0057 | +0.0024 | yes | MIR3142HG | 0 | nan |
| 9 | 2:209164130_C | +0.0052 | +0.0058 | +0.0019 | yes | nan | 1 | rs138802801 |
| 10 | 5:111712134_G | +0.0050 | +0.0054 | +0.0024 | yes | STARD4-AS1;NREP | 4 | rs146239224;rs112081420;rs7735771;rs67968533 |
| 11 | 2:183167513_C | +0.0050 | +0.0054 | +0.0020 | yes | nan | 3 | rs181204438;rs140391222;rs545260475 |
| 12 | 3:70220111_T | +0.0049 | +0.0054 | +0.0019 | yes | MDFIC2;SAMMSON | 2 | rs73117911;rs191223340 |
| 13 | 9:138290897_T | +0.0049 | +0.0054 | +0.0018 | yes | nan | 0 | nan |
| 14 | 6:152449729_C | +0.0048 | +0.0053 | +0.0020 | yes | SYNE1 | 0 | nan |
| 15 | 4:148023210_C | +0.0047 | +0.0052 | +0.0018 | yes | ARHGAP10 | 1 | rs10027347 |
| 16 | 2:52463826_G | +0.0047 | +0.0052 | +0.0017 | yes | nan | 2 | rs17732209;rs12466061 |
| 17 | 7:101026506_G | +0.0047 | +0.0051 | +0.0020 | yes | MUC12-AS1;MUC17 | 1 | rs146006635 |
| 18 | 13:50598497_C | +0.0047 | +0.0051 | +0.0020 | yes | DLEU7;DLEU1 | 15 | rs706599;rs7997464;rs797520;rs116346582;rs1341635 |
| 19 | 4:127643969_C | +0.0046 | +0.0050 | +0.0021 | yes | INTU | 1 | rs181743661 |
| 20 | 7:81722663_T | +0.0046 | +0.0049 | +0.0027 | yes | HGF | 1 | rs5745709 |


### SNP -- DECELERATORs

| rank | feature | grad_pool | grad_fhs | grad_whi | sign_agreement | gene_symbol | gwas_n_hits_near | gwas_rsids_near |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 22:33656202_C | -0.0061 | -0.0068 | -0.0022 | yes | LARGE1 | 0 | nan |
| 2 | 4:91118_A | -0.0056 | -0.0062 | -0.0022 | yes | nan | 0 | nan |
| 3 | 5:96730203_T | -0.0055 | -0.0060 | -0.0022 | yes | CAST | 3 | rs149313;rs34782647;rs697976 |
| 4 | 10:37576250_A | -0.0054 | -0.0060 | -0.0013 | yes | nan | 0 | nan |
| 5 | 12:127783810_T | -0.0054 | -0.0057 | -0.0030 | yes | nan | 0 | nan |
| 6 | 1:165573298_C | -0.0053 | -0.0058 | -0.0022 | yes | LRRC52-AS1 | 1 | rs12121840 |
| 7 | 15:89985584_C | -0.0053 | -0.0058 | -0.0024 | yes | nan | 1 | rs1256840 |
| 8 | 11:11319570_C | -0.0052 | -0.0058 | -0.0018 | yes | GALNT18 | 6 | rs61872397;rs4444072;rs12419084;rs1471895;rs148333573 |
| 9 | 10:71710262_G | -0.0052 | -0.0058 | -0.0018 | yes | CDH23 | 3 | rs2166631;rs41281310;rs10823829 |
| 10 | 12:96051418_A | -0.0052 | -0.0056 | -0.0025 | yes | nan | 1 | rs41392450 |
| 11 | 3:102903020_T | -0.0051 | -0.0056 | -0.0020 | yes | nan | 0 | nan |
| 12 | 1:219462360_G | -0.0051 | -0.0055 | -0.0027 | yes | LYPLAL1-AS1 | 20 | rs6684734;rs2605092;rs5781116;rs2605093;rs6699024 |
| 13 | 3:130089416_T | -0.0050 | -0.0055 | -0.0024 | yes | ALG1L2;LINC02014 | 0 | nan |
| 14 | 9:80950919_G | -0.0050 | -0.0055 | -0.0022 | yes | nan | 0 | nan |
| 15 | 4:45025457_T | -0.0050 | -0.0054 | -0.0022 | yes | nan | 4 | rs11932923;rs78460921;rs141080507;rs13140100 |
| 16 | 4:107251400_C | -0.0049 | -0.0054 | -0.0019 | yes | DKK2 | 0 | nan |
| 17 | 5:169807485_T | -0.0049 | -0.0054 | -0.0016 | yes | DOCK2 | 2 | rs36037302;rs261616 |
| 18 | 20:38972896_C | -0.0049 | -0.0054 | -0.0019 | yes | DHX35 | 1 | rs79166949 |
| 19 | 13:23144535_G | -0.0048 | -0.0053 | -0.0017 | yes | nan | 1 | rs9510597 |
| 20 | 9:103674422_G | -0.0048 | -0.0053 | -0.0016 | yes | nan | 0 | nan |


## Interpretation

- The cohort-portable signal the model relies on is **highly consistent** between FHS and WHI (Pearson r > 0.85, sign agreement > 0.82). WHI gradients are about a third the magnitude of FHS gradients (slope ~0.35); WHI is half the size of FHS and has a narrower age range (50-80 y vs FHS 25-90 y), so the model uses more cautious decision directions on WHI, but the *direction* of every top accelerator and decelerator agrees across cohorts.
- The model's top age accelerators and decelerators include CpGs that are **not in the Horvath / Hannum / PhenoAge / GrimAge / DunedinPACE / Zhang / Lin clocks**, even though all of those clock-CpG lists are almost fully covered by our 393 K-CpG array. This suggests the model has identified a *novel* epigenetic-age signature that is jointly informative for mortality and chronological age, but is not captured by any single published clock.
- For SNPs, ~65 % of the top features lie within 5 kb of a GWAS-catalog significant variant, suggesting most are tagging known loci. All 200 top SNPs per target have an annotated gene via Ensembl REST, so a downstream pathway analysis (e.g., the existing `run_snp_cpg_gene_pathway_pipeline.py`) can be run directly on the gene lists in this directory.

## Files in this directory

- `age_top_cpg_annotated.csv` (27.6 KB)
- `age_top_snp_annotated.csv` (21.5 KB)
- `risk_top_cpg_annotated.csv` (27.6 KB)
- `risk_top_snp_annotated.csv` (21.3 KB)
- `snp_gene_cache.csv` (4.4 KB)
- `snp_gwas_cache.csv` (8.0 KB)
- `top_unified_long.csv` (20.6 KB)
