# RQ3: effectiveness and overhead across window sizes

Analysis is descriptive: 8 projects, 18 window sizes (1–18).

## Equal-project averages (not pooled raw costs)

| Window | OCplx reduction (%) | SStubs reduction (%) | Maximum measured time (s) | Minimum peak RSS (MiB) | Maximum peak RSS (MiB) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0000 | 0.0000 | 0.012731 | 32.8281 | 43.9570 |
| 2 | 14.5464 | 12.0113 | 0.015030 | 32.8008 | 43.9141 |
| 3 | 24.3600 | 17.1966 | 0.023725 | 32.6523 | 43.8906 |
| 4 | 28.2015 | 21.3529 | 0.060359 | 32.6445 | 43.9141 |
| 5 | 33.9351 | 25.6067 | 0.059966 | 32.8945 | 43.8867 |
| 6 | 37.9994 | 27.8086 | 0.128884 | 32.6602 | 43.8867 |
| 7 | 40.9852 | 30.9065 | 0.254504 | 32.8711 | 43.8125 |
| 8 | 42.9132 | 33.0111 | 0.550814 | 32.7656 | 44.0352 |
| 9 | 45.6607 | 35.7507 | 1.289183 | 32.8477 | 44.1094 |
| 10 | 47.8812 | 37.1130 | 2.439970 | 33.0078 | 43.9727 |
| 11 | 50.0775 | 38.9353 | 5.115545 | 33.2852 | 43.9844 |
| 12 | 51.3788 | 41.3348 | 10.105629 | 33.3750 | 43.8711 |
| 13 | 54.0208 | 42.3774 | 20.127485 | 34.1016 | 43.8477 |
| 14 | 55.8293 | 43.5714 | 43.009961 | 35.1172 | 43.8828 |
| 15 | 57.5831 | 44.8766 | 81.733133 | 36.5859 | 45.0352 |
| 16 | 59.2237 | 45.0148 | 175.554705 | 40.1641 | 49.0859 |
| 17 | 59.7996 | 45.6827 | 380.901783 | 46.7422 | 66.3320 |
| 18 | 60.4594 | 46.7445 | 731.505802 | 51.0469 | 99.4102 |

## From window 10 to window 18

| Project | Additional OCplx reduction (%) | Additional SStubs reduction (%) | Measured runtime ratio | Peak RSS ratio |
|---|---:|---:|---:|---:|
| ANT | 46.3668 | 3.1250 | 62.5473 | 1.5465 |
| Email-spl | 0.0000 | 0.0000 | 95.4876 | 1.8110 |
| DNS | 85.9048 | 54.2056 | 129.9954 | 1.9732 |
| jboss | 36.8421 | 35.0000 | 151.7777 | 1.8592 |
| xml_security | 16.2745 | 16.8285 | 296.0257 | 2.5383 |
| jmeter | 12.1587 | 13.0579 | 278.0470 | 2.3671 |
| JHotDraw | 9.9021 | 5.9722 | 263.3689 | 2.2202 |
| xstream_spl | 9.3983 | 10.3004 | 314.1104 | 2.2128 |

## Deteriorations between adjacent window sizes

These do not violate the guarantee, which compares the optimized order with its own initial order, not different window sizes.

| Project | Metric | Window change | Score change |
|---|---|---:|---:|
| ANT | Ocplx | 3 → 4 | 8.54 → 8.59 |
| ANT | Ocplx | 9 → 10 | 5.63 → 5.78 |
| ANT | SStubs | 9 → 10 | 31 → 32 |
| ANT | SStubs | 15 → 16 | 27 → 31 |
| jboss | Ocplx | 5 → 6 | 3.31 → 3.43 |
| jboss | Ocplx | 14 → 15 | 2.0 → 2.01 |
| xml_security | Ocplx | 11 → 12 | 15.07 → 15.38 |
| jmeter | Ocplx | 16 → 17 | 24.27 → 24.29 |
