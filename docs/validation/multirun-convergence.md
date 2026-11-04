# Convergência do multi-run (medido)

Relatório gerado por `scripts/multirun_convergence.py report` a partir do protocolo e dos lotes; nenhum número é escrito à mão.

- modo: `measured`
- protocolo: `multirun-convergence-v1` sha256 `aface42da48da52b48f0465182f219ee1200e96ba7b0025f736cb35ba29f430f`, commit `19ead1526038ea2060d06d2bb42ca6f5055b3365` (2026-11-03T12:42:17-03:00)
- snapshot: `dev-2026-10-29-ibov` manifest_sha256 `b66a5764bdcb8d8a20cfc4e10f9ad4d7b14448e3c997fec8e597fdcaccd2d014`
- código: commit `9b3b375b2ec71266e28df890cafb5509eaf24ae0`, code_fingerprint `cb98e7756cbc0f57d5f846965c5254491dbdcc691e02313f04c86aa695d2c566`, experiment_fingerprint `83b200d38a26748892a082c7f3d316a8a2b15a05b36014cefc3c0e526ead4b40`
- seeds: 46641103, 42, 43, 44, 45; execuções por seletor: 150; prefixos: 30, 60, 100, 150

## Recomendações

### run_count: **data_insufficient**

Regra: por célula, `S(k)` e `M(k)` são a média e a mediana de trocados entre os pares de seeds no prefixo k; a célula achatou quando `S(100) - S(150) <= 0,50`; `k_min` é o menor candidato (30, 60, 100, 150) com S e M até 0,50 acima do último; o valor é o maior `k_min` entre as células, `data_insufficient` se alguma não achatou, e a recomendação é o máximo entre a regra cheia e cada seed deixada de fora.

- regra cheia: data_insufficient
- sem a seed 46641103: data_insufficient
- sem a seed 42: data_insufficient
- sem a seed 43: data_insufficient
- sem a seed 44: data_insufficient
- sem a seed 45: data_insufficient

| célula | curva (cheia) | achatou | k_min |
| --- | --- | --- | --- |
| arrojado-anchor/stocks | 30: S 4,40 / M 4,50, 60: S 2,50 / M 2,00, 100: S 2,00 / M 2,00, 150: S 1,40 / M 1,50 | não | - |
| arrojado-anchor/fiis | 30: S 3,70 / M 3,50, 60: S 2,30 / M 2,00, 100: S 2,20 / M 2,00, 150: S 1,30 / M 1,00 | não | - |
| premium-2026-11-02/stocks | 30: S 2,10 / M 2,00, 60: S 1,60 / M 1,50, 100: S 1,40 / M 1,00, 150: S 1,10 / M 1,00 | sim | 60 |
| premium-2026-11-02/fiis | 30: S 0,70 / M 1,00, 60: S 0,40 / M 0,00, 100: S 0,00 / M 0,00, 150: S 0,00 / M 0,00 | sim | 60 |

### Alvos do critério adaptativo: **use_fixed_run_count**

Regra: a 150 execuções, `target_jaccard` 0,75 (`jaccard_mean >=`) e `target_cv` 0,02 (`fitness_cv <=`) são `reachable` numa célula quando todas as seeds atingem, `unreachable` quando nenhuma, `mixed` no resto; `keep` só quando os dois vereditos são `reachable` em todas as células, senão `use_fixed_run_count`.

- target_jaccard: unreachable
- target_cv: unreachable
- fixed_run_count: 150

### Zona de proteção (K, δ)

Regra (AD-022): no prefixo recomendado, `K*` é o quantil 0,95 (CDF invertida) dos rankings de cada titular no lote da outra seed (pares ordenados; sem ranking conta como infinito; abaixo de n sobe para n; acima de 30 é nulo); `δ*` é o mesmo quantil de `|f_i(t) - f_j(t)|` nos pares de seeds, exato a partir das contagens e arredondado para cima em 0,01; vale o maior entre os perfis.

| seletor | k_offset | δ | prefixo |
| --- | --- | --- | --- |
| stocks | 6 | 0,11 | 150 |
| fiis | 3 | 0,11 | 150 |

| célula | K* | δ* | giro K=0 (média / máx) | giro no ponto recomendado (média / máx) |
| --- | --- | --- | --- | --- |
| arrojado-anchor/stocks | 21 | 0,10 | 1,40 / 2 | 0,40 / 1 |
| premium-2026-11-02/stocks | 14 | 0,11 | 1,10 / 2 | 0,10 / 1 |
| arrojado-anchor/fiis | 18 | 0,11 | 1,30 / 2 | 0,70 / 2 |
| premium-2026-11-02/fiis | 13 | 0,10 | 0,00 / 0 | 0,00 / 0 |

## Execução de produção de 2026-11-02

O lote `premium-2026-11-02` com a seed 46641103 refaz a execução `9e22cc16-cfda-495e-a8c9-c0d202ef32b6` sem o critério adaptativo. Mesmos tickers que os usuários receberam: ações não, FIIs não.

## Célula arrojado-anchor/stocks (n = 15)

### Curvas (média entre seeds, [mín, máx])

| k | fitness_cv | jaccard_mean |
| --- | --- | --- |
| 10 | 0,0529 [0,0380, 0,0690] | 0,1246 [0,1095, 0,1369] |
| 20 | 0,0594 [0,0522, 0,0671] | 0,1206 [0,1129, 0,1320] |
| 30 | 0,0568 [0,0525, 0,0613] | 0,1189 [0,1139, 0,1247] |
| 40 | 0,0589 [0,0489, 0,0638] | 0,1187 [0,1169, 0,1232] |
| 50 | 0,0588 [0,0486, 0,0631] | 0,1185 [0,1140, 0,1241] |
| 60 | 0,0578 [0,0489, 0,0618] | 0,1185 [0,1151, 0,1233] |
| 70 | 0,0568 [0,0474, 0,0614] | 0,1180 [0,1146, 0,1227] |
| 80 | 0,0560 [0,0502, 0,0595] | 0,1175 [0,1145, 0,1207] |
| 90 | 0,0546 [0,0493, 0,0585] | 0,1173 [0,1153, 0,1209] |
| 100 | 0,0551 [0,0488, 0,0613] | 0,1180 [0,1159, 0,1215] |
| 110 | 0,0552 [0,0492, 0,0588] | 0,1183 [0,1169, 0,1200] |
| 120 | 0,0545 [0,0485, 0,0582] | 0,1179 [0,1160, 0,1202] |
| 130 | 0,0549 [0,0492, 0,0569] | 0,1181 [0,1164, 0,1204] |
| 140 | 0,0554 [0,0504, 0,0583] | 0,1182 [0,1159, 0,1203] |
| 150 | 0,0554 [0,0504, 0,0583] | 0,1183 [0,1158, 0,1205] |

Parada adaptativa simulada (primeiro k em que a produção pararia): seed 46641103: -, seed 42: -, seed 43: -, seed 44: -, seed 45: -

### Frequências a 150 execuções (top 30 por seed)

| rank | seed 46641103 | seed 42 | seed 43 | seed 44 | seed 45 |
| --- | --- | --- | --- | --- | --- |
| 1 | EQPA3 0,940 | EQPA3 0,953 | EQPA3 0,927 | EQPA3 0,947 | EQPA3 0,953 |
| 2 | BBSE3 0,560 | BBSE3 0,513 | BMOB3 0,573 | BBSE3 0,580 | BBSE3 0,520 |
| 3 | BMOB3 0,460 | CXSE3 0,460 | BBSE3 0,560 | BMOB3 0,480 | BMOB3 0,460 |
| 4 | CXSE3 0,420 | BMOB3 0,420 | LOGG3 0,440 | VULC3 0,440 | CXSE3 0,427 |
| 5 | LOGG3 0,413 | LOGG3 0,400 | VULC3 0,407 | CXSE3 0,413 | VULC3 0,407 |
| 6 | VULC3 0,400 | VULC3 0,367 | CXSE3 0,320 | LOGG3 0,400 | LOGG3 0,407 |
| 7 | RDOR3 0,307 | RDOR3 0,280 | RDOR3 0,293 | JSLG3 0,273 | RDOR3 0,307 |
| 8 | FESA4 0,273 | EUCA4 0,280 | EUCA4 0,273 | RDOR3 0,267 | DEXP3 0,280 |
| 9 | JSLG3 0,247 | JSLG3 0,253 | FESA4 0,273 | FESA4 0,267 | FIQE3 0,253 |
| 10 | EUCA4 0,247 | CURY3 0,233 | FIQE3 0,260 | FIQE3 0,267 | JSLG3 0,247 |
| 11 | MDNE3 0,240 | FESA4 0,220 | JSLG3 0,253 | DEXP3 0,253 | EUCA4 0,247 |
| 12 | FIQE3 0,240 | FIQE3 0,213 | LAVV3 0,233 | EUCA4 0,227 | LAVV3 0,240 |
| 13 | LAVV3 0,233 | DEXP3 0,207 | MDNE3 0,207 | LAVV3 0,207 | FESA4 0,227 |
| 14 | DEXP3 0,213 | MDNE3 0,193 | ARML3 0,207 | ARML3 0,200 | MDNE3 0,213 |
| 15 | SMFT3 0,213 | TFCO4 0,187 | CURY3 0,200 | MDNE3 0,187 | ARML3 0,187 |
| 16 | CURY3 0,193 | ARML3 0,187 | BRSR6 0,200 | TFCO4 0,187 | CURY3 0,180 |
| 17 | DIRR3 0,180 | SUZB3 0,180 | TFCO4 0,187 | BRSR6 0,187 | SMFT3 0,180 |
| 18 | INTB3 0,180 | DIRR3 0,180 | SUZB3 0,180 | CURY3 0,167 | BRSR6 0,180 |
| 19 | AZZA3 0,180 | INTB3 0,180 | MNPR3 0,180 | ASAI3 0,167 | SUZB3 0,167 |
| 20 | TTEN3 0,173 | ASAI3 0,173 | AZZA3 0,173 | TTEN3 0,167 | DIRR3 0,167 |
| 21 | SUZB3 0,167 | LAVV3 0,167 | DEXP3 0,167 | SIMH3 0,160 | GMAT3 0,147 |
| 22 | ARML3 0,160 | GMAT3 0,167 | SMFT3 0,167 | SMFT3 0,153 | ANIM3 0,147 |
| 23 | TFCO4 0,140 | JHSF3 0,167 | GMAT3 0,167 | GMAT3 0,147 | MNPR3 0,147 |
| 24 | GMAT3 0,140 | AZZA3 0,167 | ABEV3 0,167 | INTB3 0,133 | INTB3 0,133 |
| 25 | ASAI3 0,140 | MNPR3 0,167 | INTB3 0,147 | RIAA3 0,133 | ABEV3 0,133 |
| 26 | BRSR3 0,133 | RADL3 0,167 | JHSF3 0,147 | MNPR3 0,133 | JHSF3 0,133 |
| 27 | ABEV3 0,127 | ABEV3 0,153 | DIRR3 0,140 | DIRR3 0,127 | AZZA3 0,133 |
| 28 | TOTS3 0,120 | BRSR3 0,133 | ASAI3 0,127 | ABEV3 0,120 | MATD3 0,133 |
| 29 | RIAA3 0,113 | TTEN3 0,127 | EVEN3 0,127 | MILS3 0,120 | RANI3 0,133 |
| 30 | RADL3 0,113 | SIMH3 0,127 | TTEN3 0,113 | MATD3 0,113 | TFCO4 0,127 |

### Giro entre seeds e dentro da seed (trocados)

| k | entre seeds média | mediana | mín | máx | Jaccard médio | dentro da seed média | mediana | máx |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 30 | 4,40 | 4,50 | 3 | 6 | 0,552 | 2,80 | 2 | 5 |
| 60 | 2,50 | 2,00 | 2 | 4 | 0,717 | 1,60 | 2 | 2 |
| 100 | 2,00 | 2,00 | 1 | 3 | 0,768 | 1,00 | 1 | 2 |
| 150 | 1,40 | 1,50 | 0 | 2 | 0,832 | 0,00 | 0 | 0 |

### Folga de ranking (f_n - f_(n+1))

| k | seed 46641103 | seed 42 | seed 43 | seed 44 | seed 45 | mediana |
| --- | --- | --- | --- | --- | --- | --- |
| 30 | 0,200 / 0,200 / 0,000 | 0,200 / 0,200 / 0,000 | 0,200 / 0,200 / 0,000 | 0,233 / 0,233 / 0,000 | 0,200 / 0,167 / 0,033 | 0,000 |
| 60 | 0,200 / 0,200 / 0,000 | 0,183 / 0,183 / 0,000 | 0,200 / 0,183 / 0,017 | 0,183 / 0,183 / 0,000 | 0,183 / 0,183 / 0,000 | 0,000 |
| 100 | 0,210 / 0,210 / 0,000 | 0,200 / 0,190 / 0,010 | 0,200 / 0,200 / 0,000 | 0,180 / 0,170 / 0,010 | 0,200 / 0,190 / 0,010 | 0,010 |
| 150 | 0,213 / 0,193 / 0,020 | 0,187 / 0,187 / 0,000 | 0,200 / 0,200 / 0,000 | 0,187 / 0,187 / 0,000 | 0,187 / 0,180 / 0,007 | 0,000 |

### Zona simulada a 150 execuções (giro médio / máximo nos pares ordenados)

| K | δ 0,000 | δ 0,010 | δ 0,020 | δ 0,030 | δ 0,050 | δ 0,075 | δ 0,100 | δ 0,150 | δ 0,200 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 (consenso puro) | 1,40 / 2 | 1,40 / 2 | 1,40 / 2 | 1,40 / 2 | 1,40 / 2 | 1,40 / 2 | 1,40 / 2 | 1,40 / 2 | 1,40 / 2 |
| 15 | 1,40 / 2 | 1,40 / 2 | 1,40 / 2 | 1,40 / 2 | 1,40 / 2 | 1,40 / 2 | 1,40 / 2 | 1,40 / 2 | 1,40 / 2 |
| 17 | 1,40 / 2 | 1,15 / 2 | 1,10 / 2 | 0,95 / 2 | 0,95 / 2 | 0,95 / 2 | 0,95 / 2 | 0,90 / 2 | 0,90 / 2 |
| 20 | 1,40 / 2 | 1,15 / 2 | 1,10 / 2 | 0,95 / 2 | 0,90 / 2 | 0,90 / 2 | 0,85 / 2 | 0,80 / 2 | 0,80 / 2 |
| 25 | 1,40 / 2 | 1,15 / 2 | 1,10 / 2 | 0,90 / 2 | 0,55 / 1 | 0,25 / 1 | 0,15 / 1 | 0,10 / 1 | 0,10 / 1 |
| 30 | 1,40 / 2 | 1,15 / 2 | 1,10 / 2 | 0,90 / 2 | 0,55 / 1 | 0,25 / 1 | 0,15 / 1 | 0,05 / 1 | 0,05 / 1 |

Ponto recomendado: K = 21, δ = 0,11: giro 0,40 / 1, contra 1,40 / 2 sem zona (K = 0).

## Célula arrojado-anchor/fiis (n = 15)

### Curvas (média entre seeds, [mín, máx])

| k | fitness_cv | jaccard_mean |
| --- | --- | --- |
| 10 | 0,0424 [0,0268, 0,0554] | 0,1901 [0,1549, 0,2190] |
| 20 | 0,0441 [0,0268, 0,0562] | 0,1927 [0,1750, 0,2050] |
| 30 | 0,0503 [0,0401, 0,0551] | 0,1876 [0,1783, 0,1961] |
| 40 | 0,0514 [0,0388, 0,0573] | 0,1899 [0,1833, 0,1986] |
| 50 | 0,0525 [0,0478, 0,0572] | 0,1899 [0,1856, 0,1971] |
| 60 | 0,0508 [0,0456, 0,0537] | 0,1903 [0,1866, 0,1976] |
| 70 | 0,0519 [0,0484, 0,0544] | 0,1896 [0,1863, 0,1967] |
| 80 | 0,0524 [0,0473, 0,0552] | 0,1905 [0,1869, 0,1978] |
| 90 | 0,0526 [0,0499, 0,0550] | 0,1905 [0,1877, 0,1970] |
| 100 | 0,0523 [0,0490, 0,0567] | 0,1901 [0,1866, 0,1949] |
| 110 | 0,0525 [0,0490, 0,0553] | 0,1903 [0,1865, 0,1954] |
| 120 | 0,0525 [0,0506, 0,0544] | 0,1896 [0,1847, 0,1949] |
| 130 | 0,0542 [0,0523, 0,0566] | 0,1899 [0,1864, 0,1937] |
| 140 | 0,0541 [0,0520, 0,0567] | 0,1903 [0,1879, 0,1938] |
| 150 | 0,0537 [0,0511, 0,0562] | 0,1899 [0,1881, 0,1937] |

Parada adaptativa simulada (primeiro k em que a produção pararia): seed 46641103: -, seed 42: -, seed 43: -, seed 44: -, seed 45: -

### Frequências a 150 execuções (top 30 por seed)

| rank | seed 46641103 | seed 42 | seed 43 | seed 44 | seed 45 |
| --- | --- | --- | --- | --- | --- |
| 1 | JCCJ11 0,847 | JCCJ11 0,793 | JCCJ11 0,853 | JCCJ11 0,860 | JCCJ11 0,820 |
| 2 | RBRF11 0,673 | KNCR11 0,627 | KNCR11 0,620 | MXRF11 0,680 | KNCR11 0,647 |
| 3 | MXRF11 0,647 | BTCR11 0,620 | BTCR11 0,613 | RBRF11 0,633 | BTCR11 0,567 |
| 4 | BTCR11 0,560 | RBRF11 0,613 | MXRF11 0,560 | BTCR11 0,607 | RBRF11 0,567 |
| 5 | KNCR11 0,527 | MXRF11 0,580 | RBRF11 0,520 | KNCR11 0,573 | MXRF11 0,560 |
| 6 | PLAG11 0,447 | PLAG11 0,467 | PLAG11 0,467 | PLAG11 0,513 | PLAG11 0,480 |
| 7 | PQDP11 0,420 | PQDP11 0,400 | HLOG11 0,467 | PQDP11 0,447 | PQDP11 0,473 |
| 8 | SAPI11 0,407 | TVRI11 0,393 | PQDP11 0,460 | HLOG11 0,413 | TRXB11 0,427 |
| 9 | TVRI11 0,407 | KNUQ11 0,373 | XPML11 0,440 | HGRE11 0,380 | HLOG11 0,420 |
| 10 | HLOG11 0,380 | HLOG11 0,367 | TRXB11 0,380 | TRXB11 0,360 | HGRE11 0,360 |
| 11 | KNUQ11 0,367 | SAPI11 0,360 | SAPI11 0,353 | XPML11 0,347 | XPML11 0,353 |
| 12 | TRXB11 0,360 | TRXB11 0,353 | TVRI11 0,340 | SAPI11 0,340 | SAPI11 0,353 |
| 13 | XPML11 0,353 | XPML11 0,347 | KNUQ11 0,327 | TVRI11 0,327 | CPOF11 0,347 |
| 14 | HGRE11 0,313 | BTLG11 0,300 | GGRC11 0,313 | KNUQ11 0,320 | KNUQ11 0,293 |
| 15 | GGRC11 0,307 | ALZR11 0,293 | BTLG11 0,307 | GGRC11 0,287 | TVRI11 0,293 |
| 16 | CPOF11 0,293 | RZAT11 0,293 | HGRE11 0,287 | CPUR11 0,280 | BTLG11 0,293 |
| 17 | BRCR11 0,287 | HGRE11 0,287 | CNES11 0,273 | HFOF11 0,247 | BRCR11 0,293 |
| 18 | CPUR11 0,267 | GGRC11 0,280 | HFOF11 0,267 | RZAT11 0,247 | IRIM11 0,293 |
| 19 | BTLG11 0,253 | CPUR11 0,280 | BRCR11 0,247 | BROF11 0,247 | GGRC11 0,280 |
| 20 | BROF11 0,253 | BRCR11 0,267 | CPUR11 0,247 | CPOF11 0,240 | KNRI11 0,273 |
| 21 | HFOF11 0,247 | BROF11 0,260 | ALZR11 0,247 | BRCR11 0,233 | ALZR11 0,247 |
| 22 | ALZR11 0,233 | CNES11 0,247 | CPOF11 0,240 | CNES11 0,233 | RZAT11 0,247 |
| 23 | CNES11 0,233 | HFOF11 0,240 | SNEL11 0,240 | BTLG11 0,227 | BROF11 0,233 |
| 24 | SNEL11 0,233 | CPOF11 0,227 | CPTS11 0,213 | IRIM11 0,227 | HFOF11 0,220 |
| 25 | KNRI11 0,220 | BBFO11 0,227 | RZAT11 0,200 | CPTS11 0,220 | VSHO11 0,213 |
| 26 | VSHO11 0,213 | FLMA11 0,220 | FLMA11 0,193 | FLMA11 0,200 | BBFO11 0,207 |
| 27 | BBFO11 0,207 | IRIM11 0,213 | VGIR11 0,193 | ALZR11 0,193 | RECT11 0,207 |
| 28 | VGIR11 0,193 | VSHO11 0,207 | VSHO11 0,187 | KNRI11 0,193 | CNES11 0,200 |
| 29 | RECT11 0,187 | VGIR11 0,207 | RECT11 0,187 | VSHO11 0,187 | KNHF11 0,180 |
| 30 | IRIM11 0,173 | SNEL11 0,200 | BROF11 0,180 | KNIP11 0,180 | CPUR11 0,167 |

### Giro entre seeds e dentro da seed (trocados)

| k | entre seeds média | mediana | mín | máx | Jaccard médio | dentro da seed média | mediana | máx |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 30 | 3,70 | 3,50 | 1 | 6 | 0,614 | 1,60 | 2 | 3 |
| 60 | 2,30 | 2,00 | 1 | 4 | 0,741 | 0,80 | 1 | 1 |
| 100 | 2,20 | 2,00 | 0 | 4 | 0,754 | 0,80 | 1 | 2 |
| 150 | 1,30 | 1,00 | 0 | 2 | 0,843 | 0,00 | 0 | 0 |

### Folga de ranking (f_n - f_(n+1))

| k | seed 46641103 | seed 42 | seed 43 | seed 44 | seed 45 | mediana |
| --- | --- | --- | --- | --- | --- | --- |
| 30 | 0,300 / 0,300 / 0,000 | 0,333 / 0,300 / 0,033 | 0,300 / 0,300 / 0,000 | 0,333 / 0,333 / 0,000 | 0,300 / 0,300 / 0,000 | 0,000 |
| 60 | 0,300 / 0,300 / 0,000 | 0,317 / 0,317 / 0,000 | 0,300 / 0,283 / 0,017 | 0,283 / 0,250 / 0,033 | 0,333 / 0,300 / 0,033 | 0,017 |
| 100 | 0,340 / 0,320 / 0,020 | 0,310 / 0,290 / 0,020 | 0,300 / 0,290 / 0,010 | 0,270 / 0,270 / 0,000 | 0,310 / 0,300 / 0,010 | 0,010 |
| 150 | 0,307 / 0,293 / 0,013 | 0,293 / 0,293 / 0,000 | 0,307 / 0,287 / 0,020 | 0,287 / 0,280 / 0,007 | 0,293 / 0,293 / 0,000 | 0,007 |

### Zona simulada a 150 execuções (giro médio / máximo nos pares ordenados)

| K | δ 0,000 | δ 0,010 | δ 0,020 | δ 0,030 | δ 0,050 | δ 0,075 | δ 0,100 | δ 0,150 | δ 0,200 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 (consenso puro) | 1,30 / 2 | 1,30 / 2 | 1,30 / 2 | 1,30 / 2 | 1,30 / 2 | 1,30 / 2 | 1,30 / 2 | 1,30 / 2 | 1,30 / 2 |
| 15 | 1,30 / 2 | 1,30 / 2 | 1,30 / 2 | 1,30 / 2 | 1,30 / 2 | 1,30 / 2 | 1,30 / 2 | 1,30 / 2 | 1,30 / 2 |
| 17 | 1,30 / 2 | 1,15 / 2 | 1,10 / 2 | 0,95 / 2 | 0,95 / 2 | 0,85 / 2 | 0,85 / 2 | 0,85 / 2 | 0,85 / 2 |
| 20 | 1,30 / 2 | 1,15 / 2 | 1,05 / 2 | 0,80 / 2 | 0,75 / 2 | 0,45 / 2 | 0,40 / 2 | 0,40 / 2 | 0,40 / 2 |
| 25 | 1,30 / 2 | 1,15 / 2 | 1,05 / 2 | 0,80 / 2 | 0,75 / 2 | 0,25 / 1 | 0,15 / 1 | 0,10 / 1 | 0,05 / 1 |
| 30 | 1,30 / 2 | 1,15 / 2 | 1,05 / 2 | 0,80 / 2 | 0,75 / 2 | 0,25 / 1 | 0,15 / 1 | 0,10 / 1 | 0,00 / 0 |

Ponto recomendado: K = 18, δ = 0,11: giro 0,70 / 2, contra 1,30 / 2 sem zona (K = 0).

## Célula premium-2026-11-02/stocks (n = 13)

### Curvas (média entre seeds, [mín, máx])

| k | fitness_cv | jaccard_mean |
| --- | --- | --- |
| 10 | 0,0634 [0,0379, 0,0907] | 0,1370 [0,1204, 0,1548] |
| 20 | 0,0600 [0,0353, 0,0768] | 0,1324 [0,1262, 0,1403] |
| 30 | 0,0585 [0,0357, 0,0866] | 0,1344 [0,1294, 0,1380] |
| 40 | 0,0570 [0,0418, 0,0773] | 0,1339 [0,1318, 0,1348] |
| 50 | 0,0556 [0,0432, 0,0725] | 0,1344 [0,1330, 0,1353] |
| 60 | 0,0543 [0,0420, 0,0678] | 0,1349 [0,1325, 0,1364] |
| 70 | 0,0571 [0,0520, 0,0680] | 0,1348 [0,1319, 0,1369] |
| 80 | 0,0567 [0,0522, 0,0670] | 0,1347 [0,1311, 0,1373] |
| 90 | 0,0567 [0,0525, 0,0688] | 0,1345 [0,1314, 0,1371] |
| 100 | 0,0561 [0,0512, 0,0683] | 0,1342 [0,1313, 0,1367] |
| 110 | 0,0556 [0,0508, 0,0667] | 0,1337 [0,1302, 0,1364] |
| 120 | 0,0555 [0,0505, 0,0652] | 0,1339 [0,1310, 0,1366] |
| 130 | 0,0551 [0,0501, 0,0637] | 0,1339 [0,1315, 0,1370] |
| 140 | 0,0550 [0,0496, 0,0642] | 0,1334 [0,1302, 0,1377] |
| 150 | 0,0548 [0,0491, 0,0635] | 0,1326 [0,1297, 0,1360] |

Parada adaptativa simulada (primeiro k em que a produção pararia): seed 46641103: -, seed 42: -, seed 43: -, seed 44: -, seed 45: -

### Frequências a 150 execuções (top 30 por seed)

| rank | seed 46641103 | seed 42 | seed 43 | seed 44 | seed 45 |
| --- | --- | --- | --- | --- | --- |
| 1 | BBSE3 0,687 | BBSE3 0,693 | BBSE3 0,647 | BBSE3 0,600 | LOGG3 0,707 |
| 2 | LOGG3 0,547 | LOGG3 0,587 | LOGG3 0,600 | LOGG3 0,573 | BBSE3 0,593 |
| 3 | VULC3 0,520 | RDOR3 0,467 | RDOR3 0,500 | RDOR3 0,560 | RDOR3 0,473 |
| 4 | RDOR3 0,500 | VULC3 0,433 | VULC3 0,467 | BRSR6 0,473 | BRSR6 0,460 |
| 5 | BRSR6 0,440 | FIQE3 0,433 | BRSR6 0,440 | FIQE3 0,407 | VULC3 0,440 |
| 6 | FIQE3 0,420 | BRSR6 0,380 | FIQE3 0,413 | VULC3 0,400 | SUZB3 0,413 |
| 7 | BMOB3 0,373 | SUZB3 0,380 | CXSE3 0,367 | FESA4 0,373 | BMOB3 0,373 |
| 8 | FESA4 0,360 | BMOB3 0,320 | SUZB3 0,347 | ISAE4 0,340 | FIQE3 0,353 |
| 9 | SUZB3 0,320 | EUCA4 0,300 | FESA4 0,313 | SUZB3 0,300 | FESA4 0,347 |
| 10 | CXSE3 0,313 | CXSE3 0,287 | BMOB3 0,300 | CXSE3 0,287 | ISAE4 0,313 |
| 11 | MILS3 0,287 | FESA4 0,267 | EUCA4 0,287 | BMOB3 0,280 | CXSE3 0,260 |
| 12 | EUCA4 0,253 | ISAE4 0,253 | ISAE4 0,267 | MILS3 0,240 | DEXP3 0,227 |
| 13 | CURY3 0,240 | CURY3 0,227 | CURY3 0,253 | CURY3 0,233 | EUCA4 0,213 |
| 14 | ISAE4 0,233 | LAVV3 0,227 | LAVV3 0,253 | EUCA4 0,213 | MILS3 0,207 |
| 15 | LAVV3 0,220 | DEXP3 0,220 | ABEV3 0,220 | LAVV3 0,207 | LAVV3 0,200 |
| 16 | DEXP3 0,193 | GMAT3 0,200 | MILS3 0,193 | ABEV3 0,200 | GMAT3 0,193 |
| 17 | LEVE3 0,167 | MILS3 0,193 | JHSF3 0,187 | DEXP3 0,180 | CURY3 0,180 |
| 18 | MDNE3 0,147 | JHSF3 0,193 | JSLG3 0,180 | JSLG3 0,167 | JSLG3 0,167 |
| 19 | ABEV3 0,147 | MOTV3 0,173 | DEXP3 0,173 | ASAI3 0,160 | ECOR3 0,153 |
| 20 | TTEN3 0,147 | DIRR3 0,167 | INTB3 0,147 | GMAT3 0,153 | INTB3 0,147 |
| 21 | DIRR3 0,133 | ASAI3 0,153 | TTEN3 0,147 | TAEE11 0,147 | TAEE4 0,147 |
| 22 | SHUL4 0,133 | PETR4 0,147 | ASAI3 0,133 | CPFE3 0,147 | ABEV3 0,133 |
| 23 | JSLG3 0,127 | JSLG3 0,133 | GMAT3 0,120 | MDNE3 0,140 | JHSF3 0,120 |
| 24 | JHSF3 0,120 | MDNE3 0,133 | MDNE3 0,120 | DIRR3 0,133 | MOTV3 0,120 |
| 25 | INTB3 0,120 | RECV3 0,127 | TAEE11 0,120 | RECV3 0,133 | DIRR3 0,113 |
| 26 | GMAT3 0,113 | CPFE3 0,127 | SBSP3 0,120 | INTB3 0,120 | SIMH3 0,113 |
| 27 | MOTV3 0,113 | ABEV3 0,120 | CPFE3 0,107 | MDIA3 0,120 | BRAP4 0,113 |
| 28 | SIMH3 0,107 | INTB3 0,120 | PETR4 0,107 | CMIN3 0,120 | RIAA3 0,107 |
| 29 | ASAI3 0,107 | TAEE3 0,120 | SHUL4 0,107 | SIMH3 0,113 | LEVE3 0,100 |
| 30 | TAEE11 0,107 | ECOR3 0,107 | SIMH3 0,100 | MATD3 0,107 | CMIG4 0,100 |

### Giro entre seeds e dentro da seed (trocados)

| k | entre seeds média | mediana | mín | máx | Jaccard médio | dentro da seed média | mediana | máx |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 30 | 2,10 | 2,00 | 1 | 3 | 0,727 | 1,20 | 1 | 2 |
| 60 | 1,60 | 1,50 | 1 | 3 | 0,784 | 1,00 | 1 | 3 |
| 100 | 1,40 | 1,00 | 1 | 2 | 0,808 | 0,40 | 0 | 1 |
| 150 | 1,10 | 1,00 | 0 | 2 | 0,847 | 0,00 | 0 | 0 |

### Folga de ranking (f_n - f_(n+1))

| k | seed 46641103 | seed 42 | seed 43 | seed 44 | seed 45 | mediana |
| --- | --- | --- | --- | --- | --- | --- |
| 30 | 0,233 / 0,233 / 0,000 | 0,267 / 0,267 / 0,000 | 0,233 / 0,233 / 0,000 | 0,267 / 0,267 / 0,000 | 0,233 / 0,233 / 0,000 | 0,000 |
| 60 | 0,250 / 0,250 / 0,000 | 0,267 / 0,250 / 0,017 | 0,267 / 0,250 / 0,017 | 0,250 / 0,233 / 0,017 | 0,200 / 0,183 / 0,017 | 0,017 |
| 100 | 0,240 / 0,230 / 0,010 | 0,260 / 0,230 / 0,030 | 0,260 / 0,250 / 0,010 | 0,260 / 0,220 / 0,040 | 0,230 / 0,220 / 0,010 | 0,010 |
| 150 | 0,240 / 0,233 / 0,007 | 0,227 / 0,227 / 0,000 | 0,253 / 0,253 / 0,000 | 0,233 / 0,213 / 0,020 | 0,213 / 0,207 / 0,007 | 0,007 |

### Zona simulada a 150 execuções (giro médio / máximo nos pares ordenados)

| K | δ 0,000 | δ 0,010 | δ 0,020 | δ 0,030 | δ 0,050 | δ 0,075 | δ 0,100 | δ 0,150 | δ 0,200 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 (consenso puro) | 1,10 / 2 | 1,10 / 2 | 1,10 / 2 | 1,10 / 2 | 1,10 / 2 | 1,10 / 2 | 1,10 / 2 | 1,10 / 2 | 1,10 / 2 |
| 13 | 1,10 / 2 | 1,10 / 2 | 1,10 / 2 | 1,10 / 2 | 1,10 / 2 | 1,10 / 2 | 1,10 / 2 | 1,10 / 2 | 1,10 / 2 |
| 15 | 1,10 / 2 | 0,95 / 2 | 0,95 / 2 | 0,70 / 1 | 0,70 / 1 | 0,60 / 1 | 0,60 / 1 | 0,55 / 1 | 0,55 / 1 |
| 18 | 1,10 / 2 | 0,95 / 2 | 0,95 / 2 | 0,70 / 1 | 0,55 / 1 | 0,30 / 1 | 0,20 / 1 | 0,05 / 1 | 0,05 / 1 |
| 23 | 1,10 / 2 | 0,95 / 2 | 0,95 / 2 | 0,70 / 1 | 0,55 / 1 | 0,30 / 1 | 0,15 / 1 | 0,00 / 0 | 0,00 / 0 |
| 30 | 1,10 / 2 | 0,95 / 2 | 0,95 / 2 | 0,70 / 1 | 0,55 / 1 | 0,30 / 1 | 0,15 / 1 | 0,00 / 0 | 0,00 / 0 |

Ponto recomendado: K = 19, δ = 0,11: giro 0,10 / 1, contra 1,10 / 2 sem zona (K = 0).

## Célula premium-2026-11-02/fiis (n = 13)

### Curvas (média entre seeds, [mín, máx])

| k | fitness_cv | jaccard_mean |
| --- | --- | --- |
| 10 | 0,0315 [0,0193, 0,0513] | 0,3788 [0,3712, 0,3874] |
| 20 | 0,0309 [0,0217, 0,0406] | 0,3796 [0,3701, 0,3888] |
| 30 | 0,0350 [0,0246, 0,0482] | 0,3851 [0,3675, 0,3982] |
| 40 | 0,0390 [0,0325, 0,0492] | 0,3884 [0,3764, 0,3972] |
| 50 | 0,0413 [0,0343, 0,0471] | 0,3871 [0,3728, 0,3930] |
| 60 | 0,0412 [0,0355, 0,0490] | 0,3859 [0,3756, 0,3921] |
| 70 | 0,0415 [0,0351, 0,0491] | 0,3856 [0,3778, 0,3885] |
| 80 | 0,0415 [0,0357, 0,0479] | 0,3864 [0,3803, 0,3913] |
| 90 | 0,0415 [0,0361, 0,0467] | 0,3864 [0,3813, 0,3905] |
| 100 | 0,0413 [0,0370, 0,0467] | 0,3862 [0,3836, 0,3901] |
| 110 | 0,0412 [0,0383, 0,0454] | 0,3859 [0,3834, 0,3903] |
| 120 | 0,0410 [0,0373, 0,0450] | 0,3858 [0,3818, 0,3897] |
| 130 | 0,0413 [0,0372, 0,0458] | 0,3865 [0,3839, 0,3907] |
| 140 | 0,0413 [0,0375, 0,0457] | 0,3865 [0,3823, 0,3913] |
| 150 | 0,0419 [0,0376, 0,0483] | 0,3862 [0,3820, 0,3906] |

Parada adaptativa simulada (primeiro k em que a produção pararia): seed 46641103: -, seed 42: -, seed 43: -, seed 44: -, seed 45: -

### Frequências a 150 execuções (top 30 por seed)

| rank | seed 46641103 | seed 42 | seed 43 | seed 44 | seed 45 |
| --- | --- | --- | --- | --- | --- |
| 1 | KNCR11 1,000 | MXRF11 0,993 | MXRF11 1,000 | MXRF11 1,000 | MXRF11 1,000 |
| 2 | MXRF11 0,993 | KNCR11 0,987 | KNCR11 1,000 | KNCR11 0,987 | KNCR11 0,993 |
| 3 | BTLG11 0,927 | BTLG11 0,907 | BTLG11 0,913 | BTLG11 0,920 | BTLG11 0,933 |
| 4 | BPML11 0,900 | BPML11 0,840 | BPML11 0,867 | BPML11 0,860 | BPML11 0,847 |
| 5 | ALZR11 0,780 | ALZR11 0,773 | GGRC11 0,747 | GGRC11 0,780 | ALZR11 0,713 |
| 6 | GGRC11 0,767 | GGRC11 0,727 | ALZR11 0,713 | ALZR11 0,727 | GGRC11 0,700 |
| 7 | XPML11 0,673 | XPML11 0,720 | XPML11 0,713 | XPML11 0,687 | XPML11 0,700 |
| 8 | KNRI11 0,580 | KNRI11 0,580 | KNRI11 0,567 | KNRI11 0,540 | KNRI11 0,593 |
| 9 | BRCR11 0,500 | BRCR11 0,507 | RECT11 0,527 | RBRF11 0,533 | RBRF11 0,533 |
| 10 | KNUQ11 0,500 | RBRF11 0,500 | BRCR11 0,493 | BRCR11 0,480 | RECT11 0,507 |
| 11 | CPTS11 0,433 | KNUQ11 0,453 | RBRF11 0,487 | RECT11 0,473 | KNUQ11 0,500 |
| 12 | RECT11 0,413 | RECT11 0,393 | KNUQ11 0,407 | KNUQ11 0,433 | BRCR11 0,473 |
| 13 | RBRF11 0,400 | CPTS11 0,360 | CPTS11 0,360 | CPTS11 0,373 | CPTS11 0,447 |
| 14 | VGIR11 0,280 | VGIR11 0,353 | IRIM11 0,293 | VGIR11 0,287 | VGIR11 0,287 |
| 15 | PSEC11 0,280 | IRIM11 0,287 | VGIR11 0,287 | IRIM11 0,267 | HGLG11 0,247 |
| 16 | HGLG11 0,240 | HGRU11 0,240 | HGLG11 0,280 | PSEC11 0,267 | KNIP11 0,227 |
| 17 | IRIM11 0,240 | PSEC11 0,233 | TGAR11 0,253 | HGLG11 0,247 | IRIM11 0,227 |
| 18 | KNIP11 0,220 | CPOF11 0,233 | PSEC11 0,227 | HGRE11 0,247 | PSEC11 0,227 |
| 19 | HGRE11 0,193 | TGAR11 0,213 | KNIP11 0,200 | TGAR11 0,233 | HGRU11 0,200 |
| 20 | TGAR11 0,180 | HGLG11 0,200 | VILG11 0,200 | HGRU11 0,220 | VILG11 0,187 |
| 21 | BROF11 0,180 | HGRE11 0,200 | BROF11 0,193 | KNIP11 0,193 | HGRE11 0,180 |
| 22 | VILG11 0,160 | VILG11 0,173 | HGRE11 0,187 | VILG11 0,180 | BROF11 0,167 |
| 23 | HGRU11 0,140 | BROF11 0,167 | HGRU11 0,173 | SNEL11 0,147 | CPOF11 0,160 |
| 24 | CPOF11 0,140 | KNIP11 0,160 | CPOF11 0,167 | CPOF11 0,140 | TGAR11 0,160 |
| 25 | SNEL11 0,140 | TVRI11 0,133 | KNHF11 0,120 | BROF11 0,127 | VINO11 0,153 |
| 26 | RBVA11 0,133 | SNEL11 0,120 | TRXF11 0,100 | HSLG11 0,100 | HSLG11 0,113 |
| 27 | HSLG11 0,120 | TRXF11 0,107 | VCJR11 0,093 | RBVA11 0,100 | KNHF11 0,100 |
| 28 | TRXF11 0,107 | RBVA11 0,093 | VINO11 0,087 | KNHF11 0,100 | SNEL11 0,093 |
| 29 | KNHF11 0,087 | KNHF11 0,087 | TVRI11 0,087 | VINO11 0,093 | RBVA11 0,093 |
| 30 | VINO11 0,080 | HSLG11 0,080 | HSLG11 0,073 | TVRI11 0,087 | TRXF11 0,087 |

### Giro entre seeds e dentro da seed (trocados)

| k | entre seeds média | mediana | mín | máx | Jaccard médio | dentro da seed média | mediana | máx |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 30 | 0,70 | 1,00 | 0 | 1 | 0,900 | 0,40 | 0 | 1 |
| 60 | 0,40 | 0,00 | 0 | 1 | 0,943 | 0,20 | 0 | 1 |
| 100 | 0,00 | 0,00 | 0 | 0 | 1,000 | 0,00 | 0 | 0 |
| 150 | 0,00 | 0,00 | 0 | 0 | 1,000 | 0,00 | 0 | 0 |

### Folga de ranking (f_n - f_(n+1))

| k | seed 46641103 | seed 42 | seed 43 | seed 44 | seed 45 | mediana |
| --- | --- | --- | --- | --- | --- | --- |
| 30 | 0,300 / 0,267 / 0,033 | 0,433 / 0,400 / 0,033 | 0,400 / 0,400 / 0,000 | 0,433 / 0,367 / 0,067 | 0,433 / 0,267 / 0,167 | 0,033 |
| 60 | 0,383 / 0,350 / 0,033 | 0,433 / 0,367 / 0,067 | 0,383 / 0,333 / 0,050 | 0,417 / 0,300 / 0,117 | 0,450 / 0,267 / 0,183 | 0,067 |
| 100 | 0,370 / 0,270 / 0,100 | 0,380 / 0,340 / 0,040 | 0,360 / 0,340 / 0,020 | 0,400 / 0,300 / 0,100 | 0,430 / 0,270 / 0,160 | 0,100 |
| 150 | 0,400 / 0,280 / 0,120 | 0,360 / 0,353 / 0,007 | 0,360 / 0,293 / 0,067 | 0,373 / 0,287 / 0,087 | 0,447 / 0,287 / 0,160 | 0,087 |

### Zona simulada a 150 execuções (giro médio / máximo nos pares ordenados)

| K | δ 0,000 | δ 0,010 | δ 0,020 | δ 0,030 | δ 0,050 | δ 0,075 | δ 0,100 | δ 0,150 | δ 0,200 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 (consenso puro) | 0,00 / 0 | 0,00 / 0 | 0,00 / 0 | 0,00 / 0 | 0,00 / 0 | 0,00 / 0 | 0,00 / 0 | 0,00 / 0 | 0,00 / 0 |
| 13 | 0,00 / 0 | 0,00 / 0 | 0,00 / 0 | 0,00 / 0 | 0,00 / 0 | 0,00 / 0 | 0,00 / 0 | 0,00 / 0 | 0,00 / 0 |
| 15 | 0,00 / 0 | 0,00 / 0 | 0,00 / 0 | 0,00 / 0 | 0,00 / 0 | 0,00 / 0 | 0,00 / 0 | 0,00 / 0 | 0,00 / 0 |
| 18 | 0,00 / 0 | 0,00 / 0 | 0,00 / 0 | 0,00 / 0 | 0,00 / 0 | 0,00 / 0 | 0,00 / 0 | 0,00 / 0 | 0,00 / 0 |
| 23 | 0,00 / 0 | 0,00 / 0 | 0,00 / 0 | 0,00 / 0 | 0,00 / 0 | 0,00 / 0 | 0,00 / 0 | 0,00 / 0 | 0,00 / 0 |
| 30 | 0,00 / 0 | 0,00 / 0 | 0,00 / 0 | 0,00 / 0 | 0,00 / 0 | 0,00 / 0 | 0,00 / 0 | 0,00 / 0 | 0,00 / 0 |

Ponto recomendado: K = 16, δ = 0,11: giro 0,00 / 0, contra 0,00 / 0 sem zona (K = 0).

## Custo (medido)

| lote | ações wall (s) | ações CPU (s) | FIIs wall (s) | FIIs CPU (s) | total wall (s) | total CPU (s) |
| --- | --- | --- | --- | --- | --- | --- |
| arrojado-anchor:46641103 | 3425,0 | 3337,2 | 5287,7 | 4922,3 | 8713,9 | 8260,9 |
| arrojado-anchor:42 | 3460,3 | 3379,6 | 5523,4 | 5166,5 | 8984,8 | 8547,3 |
| arrojado-anchor:43 | 3394,0 | 3310,6 | 5346,9 | 4987,8 | 8742,2 | 8299,8 |
| arrojado-anchor:44 | 3513,0 | 3434,1 | 5327,6 | 4962,6 | 8841,8 | 8398,0 |
| arrojado-anchor:45 | 3552,7 | 3468,4 | 5273,7 | 4915,1 | 8827,5 | 8384,8 |
| premium-2026-11-02:46641103 | 3661,1 | 3575,0 | 5484,2 | 5385,4 | 9147,0 | 8961,6 |
| premium-2026-11-02:42 | 3737,2 | 3654,6 | 5533,1 | 5439,3 | 9272,1 | 9095,2 |
| premium-2026-11-02:43 | 3861,9 | 3773,8 | 5551,7 | 5456,0 | 9415,4 | 9231,1 |
| premium-2026-11-02:44 | 3650,3 | 3567,0 | 5580,6 | 5481,8 | 9232,7 | 9050,1 |
| premium-2026-11-02:45 | 3737,2 | 3648,9 | 5395,6 | 5302,8 | 9134,5 | 8953,0 |

- CPU total: 24,22 CPU-horas (medido)
- tempo de parede do experimento: 9416,4 s com `jobs` 5
- máquina: AMD Ryzen 5 7600X 6-Core Processor (6 núcleos / 12 threads), fingerprint `60c16cfaee4ccce84be963e0f1b7107a7f2969287ef7c3b2e4f56ad2f16cb753`; AMD Ryzen 5 7600X 6-Core Processor (6 núcleos / 12 threads), fingerprint `7e180109c5fd263c7687373422045f6a29636375972351b195d017da41706955`

## Limitações

- Um único snapshot: isto é o piso do giro entre seeds, não o giro mês a mês, que a fase 2 mede no #46.
- 5 seeds: os 10 pares não são independentes.
- A zona simulada é o modelo da porta 3 (AD-022); o #46 precisa reproduzi-lo ou re-simular sobre estes lotes.
- Nenhuma afirmação fora da amostra: o relatório mede estabilidade, não qualidade.
- Outras configurações do GA (população, gerações, mutação) não foram medidas.
- SCORE NaN (porta 5, AD-025): produção deixa algumas ações com SCORE NaN e o GA as escolhe; um NaN desfaz a ordem por score de todo ticker empatado na mesma contagem, não só a dele. O consenso (as n primeiras posições de R) reproduz a produção exatamente, mas as posições além de n carregam a ordem arbitrária da produção dentro do empate, o que afeta o rank gap, o K* e as trocas da zona com δ = 0.
- Cobertura de perfis: selector score 1,000000 para `arrojado-anchor`, 0,697275 para `premium-2026-11-02`; nenhum perfil com selector score abaixo de 0,697275 (cerca de 0,70), e nenhum na metade conservador-moderado (selector score 0,5 ou menos), foi medido, enquanto o #46 aplica o maior `k_offset` e δ a todo perfil.
- Os lotes estão presos ao build que os produziu (Python 3.14.4, numpy 2.5.4, pandas 3.0.6; Python 3.14.4, numpy 2.5.4, pandas 3.0.6, copiados dos arquivos de custo); a aritmética do relatório é stdlib e não depende dele.
