# UTS Machine Learning – Linear & Logistic Regression

- **Nama  :** Syarif Muhammad Kautsar
- **NIM   :** 241011400494
- **Kelas :** Reg B
- **random_state :** 94

## Ringkasan Hasil

### Bagian A – Linear Regression (`Ads_Data.csv`)
| Metrik | Training | Testing |
|---|---|---|
| MAE | 4,0062 | 6,2916 |
| RMSE | 5,1305 | 8,6867 |
| R² | 0,9607 | 0,8137 |

- Persamaan: ŷ = 27,6131 + 9,2416 · BiayaIklan
- Gradient descent: lr 0,001 terlalu lambat, lr 0,02 konvergen (sama dengan normal equation), lr 0,1 divergen (overflow → NaN).
- Prediksi: 4,5 juta → 69,20; 7 juta → 92,30; 12 juta → 138,51 (ekstrapolasi, kurang dapat dipercaya).

### Bagian B – Logistic Regression (`Email_Data.csv`)
- P(Spam) = σ(−10,1635 + 0,8950·JumlahLink + 0,8434·KataPromo)
- Testing (threshold 0,5): accuracy 0,8667, precision 0,8333, recall 0,8333, F1 0,8333
- Threshold dipilih: **0,7** (precision 1,0, recall 0,6667, accuracy 0,8667)

## Isi Repositori
- `tugas_UTS.ipynb` – notebook lengkap dengan output
- `Ads_Data.csv`, `Email_Data.csv` – dataset
