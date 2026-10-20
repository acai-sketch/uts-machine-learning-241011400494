{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "cell-00",
   "metadata": {},
   "source": [
    "# Tugas UTS: Linear Regression & Logistic Regression\n",
    "**Mata Kuliah:** Machine Learning\n",
    "\n",
    "Tugas ini menguji pemahaman Anda tentang materi **Simple Regression** (`simple_regression/regression_1.ipynb`) dan **Simple Classification** (`simple_classification/classification_1.ipynb`). Anda akan menerapkan algoritma yang sama pada **dataset baru**, lalu mengembangkannya dengan evaluasi dan eksperimen tambahan.\n",
    "\n",
    "**Identitas Mahasiswa**\n",
    "- Nama  : Syarif Muhammad Kautsar\n",
    "- NIM   : 241011400494\n",
    "- Kelas : Reg B"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cell-01",
   "metadata": {},
   "source": [
    "## 📌 Ketentuan Pengerjaan\n",
    "\n",
    "1. Kerjakan semua soal di notebook ini pada sel kode / sel markdown yang telah disediakan.\n",
    "2. **Model dan metrik evaluasi wajib diimplementasikan sendiri** dengan NumPy, seperti pada materi (`train_test_split`, `LinearRegression`, `LinearRegressionGD`, `LogisticRegression`, `accuracy_score`, `confusion_matrix`). Anda boleh menyalin implementasi dari materi.\n",
    "   - Library yang boleh dipakai: `numpy`, `pandas`, `matplotlib`.\n",
    "   - `scikit-learn` **tidak boleh** dipakai untuk model dan metrik, kecuali pada soal bonus untuk membandingkan hasil.\n",
    "3. **`random_state` = 2 digit terakhir NIM Anda** (contoh: NIM `...0347` → `random_state=47`; jika `00`, gunakan `100`). Karena itu, setiap mahasiswa akan mendapat hasil angka yang berbeda.\n",
    "4. Setiap soal yang meminta **penjelasan/analisis** dijawab dengan kalimat Anda sendiri pada sel markdown, dan harus merujuk ke angka/grafik hasil Anda.\n",
    "5. Notebook harus **sudah dijalankan** (*Run All*) sehingga semua output dan grafik terlihat saat dibuka di GitHub.\n",
    "\n",
    "## 📂 Dataset\n",
    "Dataset disediakan di folder ini:\n",
    "\n",
    "| File | Kolom | Keterangan |\n",
    "|---|---|---|\n",
    "| `Ads_Data.csv` | `BiayaIklan`, `Penjualan` | Biaya iklan per bulan (juta Rp) dan penjualan per bulan (juta Rp), 40 data |\n",
    "| `Email_Data.csv` | `JumlahLink`, `KataPromo`, `Spam` | Jumlah link dan jumlah kata promosi dalam sebuah email, serta label Spam (1) / Bukan Spam (0), 60 data |"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cell-02",
   "metadata": {},
   "source": [
    "## 📤 Cara Pengumpulan\n",
    "\n",
    "1. Buat **repositori baru** di akun GitHub Anda masing-masing dengan nama: `uts-machine-learning-<NIM>` (contoh: `uts-machine-learning-20230801047`).\n",
    "2. Atur repositori menjadi **Public**.\n",
    "3. Struktur repositori minimal:\n",
    "   ```\n",
    "   uts-machine-learning-<NIM>/\n",
    "   ├── README.md            # Nama, NIM, Kelas, dan ringkasan hasil (R², akurasi, dll)\n",
    "   ├── tugas_UTS.ipynb      # notebook ini, sudah dijalankan lengkap dengan output\n",
    "   ├── Ads_Data.csv\n",
    "   └── Email_Data.csv\n",
    "   ```\n",
    "4. Kumpulkan **link repositori GitHub** di halaman **UTS pada Mentari**.\n",
    "5. Batas waktu pengumpulan (hari terakhir periode UTS):\n",
    "\n",
    "| Kelas | Batas Pengumpulan |\n",
    "|---|---|\n",
    "| **Reg B** | **Jumat, 23 Oktober 2026, pukul 23:59 WIB** |\n",
    "| **Reg C** | **Sabtu, 24 Oktober 2026, pukul 23:59 WIB** |\n",
    "\n",
    "> ⚠️ Yang dinilai adalah **commit terakhir sebelum batas waktu**. Commit setelah batas waktu tidak akan dinilai. Pastikan link yang dikumpulkan bisa dibuka (repositori tidak *private*).\n",
    "\n",
    "## 🧮 Penilaian\n",
    "\n",
    "| Bagian | Bobot |\n",
    "|---|---|\n",
    "| Bagian A – Linear Regression | 35% |\n",
    "| Bagian B – Logistic Regression | 40% |\n",
    "| Bagian C – Pemahaman Konsep | 15% |\n",
    "| Kerapihan repositori, README, dan notebook (output lengkap, kode rapi & diberi komentar) | 10% |\n",
    "| Bonus – Perbandingan dengan scikit-learn | +5 |"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "cell-03",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-10-20T11:35:52.660165Z",
     "iopub.status.busy": "2026-10-20T11:35:52.659322Z",
     "iopub.status.idle": "2026-10-20T11:35:53.271703Z",
     "shell.execute_reply": "2026-10-20T11:35:53.270472Z"
    }
   },
   "outputs": [],
   "source": [
    "# import Library\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "RANDOM_STATE = 94   # 2 digit terakhir NIM 241011400494"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "cell-04",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-10-20T11:35:53.274064Z",
     "iopub.status.busy": "2026-10-20T11:35:53.273369Z",
     "iopub.status.idle": "2026-10-20T11:35:53.278793Z",
     "shell.execute_reply": "2026-10-20T11:35:53.277435Z"
    }
   },
   "outputs": [],
   "source": [
    "# Salin fungsi train_test_split dari materi di sini\n",
    "def train_test_split(X, y, test_size=0.2, random_state=None):\n",
    "    \"\"\"Membagi data menjadi train dan test secara acak (implementasi NumPy).\"\"\"\n",
    "    X = np.asarray(X)\n",
    "    y = np.asarray(y)\n",
    "    rng = np.random.RandomState(random_state)\n",
    "    n = X.shape[0]\n",
    "    idx = rng.permutation(n)\n",
    "    n_test = int(round(n * test_size))\n",
    "    test_idx, train_idx = idx[:n_test], idx[n_test:]\n",
    "    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cell-05",
   "metadata": {},
   "source": [
    "---\n",
    "# Bagian A – Linear Regression (35%)\n",
    "\n",
    "Sebuah toko online ingin mengetahui pengaruh **biaya iklan** terhadap **penjualan** setiap bulan. Gunakan dataset `Ads_Data.csv`."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cell-06",
   "metadata": {},
   "source": [
    "### Soal A1 – Load & Visualisasi Data\n",
    "1. Baca `Ads_Data.csv` menggunakan pandas dan tampilkan 5 data pertama serta jumlah baris datanya.\n",
    "2. Pisahkan `X` (`BiayaIklan`) dan `y` (`Penjualan`). Pastikan `X` berbentuk 2 dimensi `(n, 1)`.\n",
    "3. Buat *scatter plot* `BiayaIklan` vs `Penjualan` (lengkap dengan judul dan label sumbu)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "cell-07",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-10-20T11:35:53.282210Z",
     "iopub.status.busy": "2026-10-20T11:35:53.281063Z",
     "iopub.status.idle": "2026-10-20T11:35:53.433154Z",
     "shell.execute_reply": "2026-10-20T11:35:53.432164Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5 data pertama:\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>BiayaIklan</th>\n",
       "      <th>Penjualan</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.8</td>\n",
       "      <td>34.3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.9</td>\n",
       "      <td>38.8</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1.0</td>\n",
       "      <td>33.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1.3</td>\n",
       "      <td>36.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1.7</td>\n",
       "      <td>51.3</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   BiayaIklan  Penjualan\n",
       "0         0.8       34.3\n",
       "1         0.9       38.8\n",
       "2         1.0       33.2\n",
       "3         1.3       36.0\n",
       "4         1.7       51.3"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Jumlah baris data: 40\n",
      "Shape X: (40, 1) | Shape y: (40,)\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAmoAAAGwCAYAAAAZn0mrAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjgsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvwVt1zgAAAAlwSFlzAAAPYQAAD2EBqD+naQAAXBlJREFUeJzt3XlYVHX7P/D3DLIJAiLDlgMiLqS5FWqRKBilPuYDIaKmuWRamSagphaaGuWSC2YumT3aopIi9lj9tMcNHZcUN0pzRQErYHBhE0GdOb8//DI5MuAMzDBz4P26Lq6rOefMOffMR+n2s9wfiSAIAoiIiIjI4kjNHQARERER6cZEjYiIiMhCMVEjIiIislBM1IiIiIgsFBM1IiIiIgvFRI2IiIjIQjFRIyIiIrJQTNSIiIiILBQTNSIiIiILxUSNqJ6TSCSYPXu2ucPQW2pqKiQSCZKTk6u9bv369ZBIJMjMzKybwBqAkJAQhISEmPQZs2fPhkQiMekziOoTJmpEIlORoDz84+7ujtDQUOzYscPc4RlMIpFgwoQJ5g7DojzctlKpFN7e3njppZeQmppq7tCIqI41MncARFQzc+fOhZ+fHwRBQF5eHtavX49//etf+PHHH/Hyyy9rrrtz5w4aNeJfdbF58cUXMWLECAiCgKtXr2LlypXo3bs3fv75Z/Tr188kz/zf//5nkvsSUc3xtzeRSPXr1w+BgYGa12PGjIGHhwc2bdqklajZ2dmZIzyqpTZt2mD48OGa16+88go6duyIxMREkyVqNjY2JrkvEdUchz6J6gkXFxfY29tX6j17dI5aVlYWxo8fj7Zt28Le3h7NmjXDoEGDtOZ6XblyBRKJBEuXLq30nMOHD0MikWDTpk1636+2ysvL8fLLL8PZ2RmHDx+u8rr//ve/6N+/P7y9vWFrawt/f3989NFHUKlUWteFhITgqaeewh9//IHQ0FA0btwYTzzxBBYuXPjYWJ566imEhoZWOq5Wq/HEE08gKipKcywpKQnPPPMMmjRpAicnJ3To0AHLli0z4JP/o0OHDnBzc8PVq1c1x86fP4+oqCi4urrCzs4OgYGB2L59u9b7KobKDx06hLi4OMhkMjg4OOCVV15Bfn6+1rWPzlGrah5gxTzCh4diFQoFBg0aBB8fH9ja2kIulyM2NhZ37tx57Gdbt24devfuDXd3d9ja2qJdu3ZYtWpVpetatGiBl19+GQcPHkS3bt1gZ2eHli1b4ptvvnnsM4jEiokakUgVFhbi+vXryM/Px9mzZ/H222+jpKREqxdGl7S0NBw+fBhDhgzBZ599hrfeegt79uxBSEgISktLAQAtW7bE888/jw0bNlR6/4YNG9CkSROEh4frfb/auHPnDgYMGIDDhw9j9+7dCAoKqvLa9evXw9HREXFxcVi2bBmeeeYZzJo1C9OnT6907a1bt9C3b1906tQJixcvRkBAAKZNm/bYeX6DBw/GgQMHkJubq3X84MGD+PvvvzFkyBAAwK5duzB06FA0bdoUCxYswPz58xESEoJDhw7V4Ft4EO+tW7fQrFkzAMDZs2fx7LPP4ty5c5g+fToWL14MBwcHREREYNu2bZXeP3HiRKSnp+PDDz/E22+/jR9//NGocwO3bNmC0tJSvP3221i+fDn69OmD5cuXY8SIEY9976pVq+Dr64v3338fixcvhlwux/jx47FixYpK116+fBlRUVF48cUXsXjxYjRt2hSjRo3C2bNnjfZZiCyKQESism7dOgFApR9bW1th/fr1la4HIHz44Yea16WlpZWuOXLkiABA+OabbzTHvvjiCwGAcO7cOc2xu3fvCm5ubsLIkSMNvl9VAAjvvPOO5vW+ffsEAMKWLVuE4uJioVevXoKbm5tw6tQpnd/D1atXq43lzTffFBo3biyUlZVpjvXq1atSfOXl5YKnp6cwcODAauO9cOGCAEBYvny51vHx48cLjo6OmhgmTZokODk5Cffv33/sd/AoAMKYMWOE/Px8QalUCkePHhVeeOEFAYCwePFiQRAE4YUXXhA6dOig9bnUarUQFBQktG7dWnOs4nsKCwsT1Gq15nhsbKxgZWUlFBQUaH0vvXr1qvTeh79jQfinjfbt26c5puu7nzdvniCRSISsrCzNsQ8//FB49H89ut7bp08foWXLllrHfH19BQDCgQMHNMeUSqVga2srTJ48udI9iOoD9qgRidSKFSuwa9cu7Nq1C9999x1CQ0PxxhtvICUlpdr32dvba/773r17uHHjBlq1agUXFxecPHlScy46Ohp2dnZavWq//PILrl+/rtVrp+/9DFVYWIiXXnoJ58+fR2pqKjp37vzY9zwcS3FxMa5fv47g4GCUlpbi/PnzWtc6OjpqfQ4bGxt069YNV65cqfYZ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2W1A18KHh5l77hwREZkHEzXSyZDEoap6ZRW9b+lJCZg8ZSoOHTRub9DDixwuXbqE2bNnV5soPpqA/fvf/8bKlSuRkZEBf39/jB8/HjY2NkaNUV+6Fj48zJxz54iIyHw49Ek6PZw46PJw4qDvsN3Zs2eNFt+jw6yz58xBs9aBBg3TtmrdBrGxsfj8888RGxuLVq3bmG3Cvq6FDxXMOXeOiIjMi4ka6WRI4qBv79utW7eMEtujixy6DJ8NQa1Gy16D9ZrfZYmrK3UtfLCEuXNERGReTNRIJ0MSB31735o2bVrruHQtcrhfdhtAzYZpLWl1ZWRkpNbCh70fR+PY2qmwu6PktlZERA0UEzWqkr6Jg769b+3bt691TLqGWQ2pjWbpqysjIyNx9UoG9u3bh40bN2Lfvn24knGZSRoRUQPFxQRUrcjISISHh1e7M0FF71tUVBTSkxLQoscgOHr4oiQvC5kHtyD/QhqSk5Mhldb+3wW6hlmb+raHnYsHrh7YrLWYAag8TLt58+ZK73+YJayutLKyQkhIiNmeT0REloOJGj2WPolDRe9bTGyczpIXERERUCqVtY5F1+pIidQKbfuOQfr383BqUwJaButOFB8dpuXqSiIisnQSQRAEcwdhbkVFRXB2dkZhYSGcnJzMHY6oVbUvqFqthlKphLu7e6161lQqFfxa+qO8sQc6DdHuPcs9cxBnf0iE6m6Z5piPbwssXbJYM3RY3fsFtRrpSQmwu6PElYzLDX7ivrHajOoG20t82GbiYsz2MiTvYI8aGZWph+2qG2bN/W0vVHfLMGfOHLRu3bpWw7QNPUkjIiLLwESNRKe6YdatW7c+duL944ZpOXGfiIgsBRM1EiV9FjmY8v1ERER1gYkaiVZth1m5upKIiCwdZy8SERERWSgmakREREQWiokaERERkYViokZERERkoZioEREREVkoJmpEREREFoqJGhEREZGFYqJGREREZKGYqBERERFZKCZqRERERBaKiRoRERGRhWKiRkRERGShmKgRERERWSgmakREREQWiokaERERkYViokZERERkoZioEREREVkoJmpEREREFoqJGhEREZGFYqJGREREZKGYqBERERFZKCZqRERERBaKiRoRERGRhWKiRkRERGShRJWozZ8/HxKJBDExMZpjZWVleOedd9CsWTM4Ojpi4MCByMvLM1+Q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",
      "text/plain": [
       "<Figure size 700x450 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Jawaban A1\n",
    "df_ads = pd.read_csv(\"Ads_Data.csv\")\n",
    "print(\"5 data pertama:\")\n",
    "display(df_ads.head())\n",
    "print(\"Jumlah baris data:\", len(df_ads))\n",
    "\n",
    "# Pisahkan X (2 dimensi, shape (n, 1)) dan y\n",
    "X_ads = df_ads[[\"BiayaIklan\"]].values      # double bracket -> 2D\n",
    "y_ads = df_ads[\"Penjualan\"].values\n",
    "print(\"Shape X:\", X_ads.shape, \"| Shape y:\", y_ads.shape)\n",
    "\n",
    "# Scatter plot\n",
    "plt.figure(figsize=(7, 4.5))\n",
    "plt.scatter(X_ads, y_ads, color=\"steelblue\", edgecolor=\"k\")\n",
    "plt.title(\"Biaya Iklan vs Penjualan\")\n",
    "plt.xlabel(\"Biaya Iklan (juta Rp)\")\n",
    "plt.ylabel(\"Penjualan (juta Rp)\")\n",
    "plt.grid(alpha=0.3)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cell-08",
   "metadata": {},
   "source": [
    "### Soal A2 – Split Data\n",
    "Bagi data menjadi data training dan testing dengan `test_size=0.2` dan `random_state=RANDOM_STATE`. Tampilkan *shape* dari `X_train`, `X_test`, `y_train`, `y_test`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "cell-09",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-10-20T11:35:53.435439Z",
     "iopub.status.busy": "2026-10-20T11:35:53.434798Z",
     "iopub.status.idle": "2026-10-20T11:35:53.440500Z",
     "shell.execute_reply": "2026-10-20T11:35:53.439567Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "X_train: (32, 1)\n",
      "X_test : (8, 1)\n",
      "y_train: (32,)\n",
      "y_test : (8,)\n"
     ]
    }
   ],
   "source": [
    "# Jawaban A2\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X_ads, y_ads, test_size=0.2, random_state=RANDOM_STATE)\n",
    "\n",
    "print(\"X_train:\", X_train.shape)\n",
    "print(\"X_test :\", X_test.shape)\n",
    "print(\"y_train:\", y_train.shape)\n",
    "print(\"y_test :\", y_test.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cell-10",
   "metadata": {},
   "source": [
    "### Soal A3 – Training dengan Normal Equation\n",
    "1. Latih model `LinearRegression` (normal equation, seperti di materi) menggunakan data training.\n",
    "2. Tampilkan nilai **intercept** ($b$) dan **koefisien** ($m$), lalu tuliskan persamaan garis regresinya: $\\hat{y} = b + mX$.\n",
    "3. **Interpretasikan** arti nilai $m$ dalam konteks kasus ini (misal: \"setiap kenaikan biaya iklan 1 juta Rp, penjualan naik sebesar ...\")."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "cell-11",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-10-20T11:35:53.442612Z",
     "iopub.status.busy": "2026-10-20T11:35:53.441948Z",
     "iopub.status.idle": "2026-10-20T11:35:53.448442Z",
     "shell.execute_reply": "2026-10-20T11:35:53.447615Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Intercept (b)  : 27.6131\n",
      "Koefisien (m)  : 9.2416\n",
      "Persamaan      : y_hat = 27.6131 + 9.2416 * X\n"
     ]
    }
   ],
   "source": [
    "# Jawaban A3\n",
    "class LinearRegression:\n",
    "    \"\"\"Linear Regression dengan Normal Equation: theta = (X^T X)^-1 X^T y\"\"\"\n",
    "    def fit(self, X, y):\n",
    "        Xb = np.c_[np.ones(X.shape[0]), X]            # tambah kolom bias\n",
    "        theta = np.linalg.pinv(Xb.T @ Xb) @ Xb.T @ y\n",
    "        self.intercept_ = theta[0]\n",
    "        self.coef_ = theta[1:]\n",
    "        return self\n",
    "\n",
    "    def predict(self, X):\n",
    "        return self.intercept_ + X @ self.coef_\n",
    "\n",
    "lin = LinearRegression().fit(X_train, y_train)\n",
    "b_ne, m_ne = lin.intercept_, lin.coef_[0]\n",
    "print(f\"Intercept (b)  : {b_ne:.4f}\")\n",
    "print(f\"Koefisien (m)  : {m_ne:.4f}\")\n",
    "print(f\"Persamaan      : y_hat = {b_ne:.4f} + {m_ne:.4f} * X\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cell-12",
   "metadata": {},
   "source": [
    "**Jawaban A3 (interpretasi):**\n",
    "\n",
    "Persamaan regresi hasil normal equation: **ŷ = 27,6131 + 9,2416 · X**.\n",
    "\n",
    "Koefisien **m = 9,2416** berarti setiap kenaikan biaya iklan sebesar **1 juta Rp**, penjualan per bulan rata-rata naik sekitar **9,24 juta Rp** (dengan asumsi hubungan linear). Intercept **b = 27,6131** adalah penjualan dasar ketika biaya iklan 0. Karena biaya iklan terkecil pada data adalah 0,8 juta, nilai ini hanya berupa titik potong garis dan tidak perlu ditafsirkan terlalu harfiah."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cell-13",
   "metadata": {},
   "source": [
    "### Soal A4 – Evaluasi Model\n",
    "Pada materi belum ada evaluasi untuk model regresi. **Implementasikan sendiri** fungsi-fungsi berikut menggunakan NumPy:\n",
    "\n",
    "$$\n",
    "MAE = \\frac{1}{n}\\sum_{i=1}^{n}|y_i - \\hat{y}_i| \\qquad\n",
    "MSE = \\frac{1}{n}\\sum_{i=1}^{n}(y_i - \\hat{y}_i)^2 \\qquad\n",
    "RMSE = \\sqrt{MSE}\n",
    "$$\n",
    "\n",
    "$$\n",
    "R^2 = 1 - \\frac{\\sum_{i=1}^{n}(y_i - \\hat{y}_i)^2}{\\sum_{i=1}^{n}(y_i - \\bar{y})^2}\n",
    "$$\n",
    "\n",
    "1. Hitung MAE, MSE, RMSE, dan $R^2$ pada **data training** dan **data testing**.\n",
    "2. Tampilkan hasilnya dalam satu tabel (boleh pakai `pd.DataFrame`).\n",
    "3. Jelaskan arti nilai RMSE dan $R^2$ yang Anda dapatkan. Apakah model ini sudah baik?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "cell-14",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-10-20T11:35:53.450545Z",
     "iopub.status.busy": "2026-10-20T11:35:53.449759Z",
     "iopub.status.idle": "2026-10-20T11:35:53.464021Z",
     "shell.execute_reply": "2026-10-20T11:35:53.463105Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>MAE</th>\n",
       "      <th>MSE</th>\n",
       "      <th>RMSE</th>\n",
       "      <th>R2</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Training</th>\n",
       "      <td>4.0062</td>\n",
       "      <td>26.3219</td>\n",
       "      <td>5.1305</td>\n",
       "      <td>0.9607</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Testing</th>\n",
       "      <td>6.2916</td>\n",
       "      <td>75.4585</td>\n",
       "      <td>8.6867</td>\n",
       "      <td>0.8137</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "             MAE      MSE    RMSE      R2\n",
       "Training  4.0062  26.3219  5.1305  0.9607\n",
       "Testing   6.2916  75.4585  8.6867  0.8137"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Std Penjualan (data train): 25.8794\n"
     ]
    }
   ],
   "source": [
    "# Jawaban A4\n",
    "def mae(y_true, y_pred):\n",
    "    return np.mean(np.abs(y_true - y_pred))\n",
    "\n",
    "def mse(y_true, y_pred):\n",
    "    return np.mean((y_true - y_pred) ** 2)\n",
    "\n",
    "def rmse(y_true, y_pred):\n",
    "    return np.sqrt(mse(y_true, y_pred))\n",
    "\n",
    "def r2_score(y_true, y_pred):\n",
    "    ss_res = np.sum((y_true - y_pred) ** 2)\n",
    "    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)\n",
    "    return 1 - ss_res / ss_tot\n",
    "\n",
    "y_pred_train = lin.predict(X_train)\n",
    "y_pred_test = lin.predict(X_test)\n",
    "\n",
    "hasil_a4 = pd.DataFrame({\n",
    "    \"MAE\":  [mae(y_train, y_pred_train),  mae(y_test, y_pred_test)],\n",
    "    \"MSE\":  [mse(y_train, y_pred_train),  mse(y_test, y_pred_test)],\n",
    "    \"RMSE\": [rmse(y_train, y_pred_train), rmse(y_test, y_pred_test)],\n",
    "    \"R2\":   [r2_score(y_train, y_pred_train), r2_score(y_test, y_pred_test)],\n",
    "}, index=[\"Training\", \"Testing\"])\n",
    "display(hasil_a4.round(4))\n",
    "print(\"Std Penjualan (data train):\", round(np.std(y_train), 4))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cell-15",
   "metadata": {},
   "source": [
    "**Jawaban A4 (analisis):**\n",
    "\n",
    "| Data | MAE | MSE | RMSE | R² |\n",
    "|---|---|---|---|---|\n",
    "| Training | 4,0062 | 26,3219 | 5,1305 | 0,9607 |\n",
    "| Testing | 6,2916 | 75,4585 | 8,6867 | 0,8137 |\n",
    "\n",
    "**RMSE** adalah rata-rata kesalahan prediksi dalam satuan asli (juta Rp). Pada data testing RMSE = 8,69, artinya prediksi penjualan meleset sekitar 8,7 juta Rp, atau sekitar 10,5% dari rata-rata penjualan testing (82,7 juta). Nilai **R²** menunjukkan proporsi variasi penjualan yang dijelaskan oleh biaya iklan: 96,07% pada training dan 81,37% pada testing.\n",
    "\n",
    "Model ini **sudah cukup baik**, karena satu variabel (biaya iklan) saja mampu menjelaskan lebih dari 80% variasi penjualan di data yang belum pernah dilihat, dan RMSE testing (8,69) jauh lebih kecil dari simpangan baku penjualan (±25,9). Meski begitu, ada selisih antara training dan testing (R² 0,96 vs 0,81). Data testing hanya 8 titik, sehingga angkanya sensitif terhadap pembagian data. Selisih ini juga menunjukkan bahwa performa pada data baru akan sedikit lebih buruk daripada performa training."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cell-16",
   "metadata": {},
   "source": [
    "### Soal A5 – Gradient Descent & Learning Rate\n",
    "1. Latih model `LinearRegressionGD` (dari materi) dengan `n_iterations=5000` menggunakan **3 nilai learning rate**: `0.001`, `0.02`, dan `0.1`.\n",
    "2. Untuk setiap learning rate, tampilkan intercept dan koefisien akhirnya, lalu bandingkan dengan hasil normal equation (Soal A3).\n",
    "3. Buat grafik **loss (MSE) vs iterasi** untuk ketiga learning rate (boleh dalam satu grafik atau *subplot*; gunakan skala log jika perlu).\n",
    "4. Jelaskan apa yang terjadi pada masing-masing learning rate: mana yang **terlalu lambat**, mana yang **pas**, dan mana yang **gagal konvergen (divergen)**? Mengapa bisa terjadi?\n",
    "\n",
    "> *Tips: jika muncul `RuntimeWarning: overflow`, itu adalah bagian dari hasil eksperimen. Jelaskan artinya.*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "cell-17",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-10-20T11:35:53.466357Z",
     "iopub.status.busy": "2026-10-20T11:35:53.465694Z",
     "iopub.status.idle": "2026-10-20T11:35:54.287963Z",
     "shell.execute_reply": "2026-10-20T11:35:54.286887Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "--- learning rate = 0.001 ---\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Intercept: 25.2720 | Koefisien: 9.6112 | MSE akhir: 27.6648\n",
      "--- learning rate = 0.02 ---\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Intercept: 27.6131 | Koefisien: 9.2416 | MSE akhir: 26.3219"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "--- learning rate = 0.1 ---\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Intercept: nan | Koefisien: nan | MSE akhir: nan\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.12/dist-packages/numpy/_core/_methods.py:132: RuntimeWarning: overflow encountered in reduce\n",
      "  ret = umr_sum(arr, axis, dtype, out, keepdims, where=where)\n",
      "/tmp/ipykernel_119/1773970267.py:16: RuntimeWarning: overflow encountered in square\n",
      "  self.loss_history_.append(np.mean(error ** 2))     # MSE\n",
      "/tmp/ipykernel_119/1773970267.py:19: RuntimeWarning: overflow encountered in matmul\n",
      "  self.coef_      -= self.lr * (2 / n) * (X.T @ error)\n",
      "/tmp/ipykernel_119/1773970267.py:19: RuntimeWarning: invalid value encountered in subtract\n",
      "  self.coef_      -= self.lr * (2 / n) * (X.T @ error)\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Model</th>\n",
       "      <th>LR</th>\n",
       "      <th>Intercept</th>\n",
       "      <th>Koefisien</th>\n",
       "      <th>MSE akhir (train)</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Normal Equation</td>\n",
       "      <td>-</td>\n",
       "      <td>27.613127</td>\n",
       "      <td>9.241607</td>\n",
       "      <td>26.321944</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Gradient Descent</td>\n",
       "      <td>0.001</td>\n",
       "      <td>25.271975</td>\n",
       "      <td>9.611181</td>\n",
       "      <td>27.664807</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Gradient Descent</td>\n",
       "      <td>0.02</td>\n",
       "      <td>27.613127</td>\n",
       "      <td>9.241607</td>\n",
       "      <td>26.321944</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Gradient Descent</td>\n",
       "      <td>0.1</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              Model     LR  Intercept  Koefisien  MSE akhir (train)\n",
       "0   Normal Equation      -  27.613127   9.241607          26.321944\n",
       "1  Gradient Descent  0.001  25.271975   9.611181          27.664807\n",
       "2  Gradient Descent   0.02  27.613127   9.241607          26.321944\n",
       "3  Gradient Descent    0.1        NaN        NaN                NaN"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAABgEAAAGGCAYAAABSYiEeAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjgsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvwVt1zgAAAAlwSFlzAAAPYQAAD2EBqD+naQAAti1JREFUeJzs3Xl4TNcbB/DvzGTfE9mDiFDEvot9K0Kritp3P1SDVqxBEWqnVcTaoguldqq172utVYJWBZXdkkxkz8z9/UGmRhYzzOTO8v08T5527r1z7ntPJueY+95zjkQQBAFERERERERERERERGRypGIHQERERERERERERERE+sEkABERERERERERERGRiWISgIiIiIiIiIiIiIjIRDEJQERERERERERERERkopgEICIiIiIiIiIiIiIyUUwCEBERERERERERERGZKCYBiIiIiIiIiIiIiIhMFJMAREREREREREREREQmikkAIiIiIiIiIiIiIiITxSQAmZX169dDIpHg3r17YodSpGPHjkEikeDYsWNih0JERGSQ2KcTkTljG0iGpn379hgyZIjYYZCR+OGHH1CxYkVYWlrCxcUFANC8eXM0b95c1LgMzdu09c2bN0eVKlV0H5QRkkgkmD59uthhvJEyZcpgwIABqtf79u2Dg4MDkpKStC6LSQAi0rkzZ85g+vTpSE5OFjuUIimVSsyfPx8BAQGwsbFBtWrV8NNPP2n8/uTkZAwdOhQeHh6wt7dHixYtcPny5QKP3b17N2rVqgUbGxuULl0a06ZNQ25urtoxcXFxmDhxIlq0aAFHR0d+YSIiItGxT//P48ePsWDBAjRt2hQeHh5wcXFBgwYNsHnzZl1fDhEZCHNoA3XxHeT06dM4cOAAJkyYoNqWlwDK+7G2toaXlxeaN2+O2bNnF3gDK++G58WLFwEA1apVQ+nSpSEIQqHnbtSoEby8vJCbm4t79+6pnfPVn7lz56re17x5c7V9tra2qFatGhYvXgylUlngueRyOWbNmoU6derA2dkZ1tbW8Pf3R/fu3bF37161Y1+9/ld/Nm3apDq2TJkykEgkGDlyZL5z5pWzdevWQuvA2Ny6dQsDBgxAYGAg1qxZg9WrV4sdkl7ExsZi+vTpuHr1qtihFJv09HRERkaiTZs28PHxgaOjI2rWrIkVK1ZAoVDkO16btmvZsmWoVKkS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rK2thcDAQGHcuHFCSkpKvrhCQ0OFUqVKCZaWloK3t7fQqlUrYfXq1WrH7dq1SwgKChIsLCzy/VvhypUrQufOnVXn8vf3F7p16yYcPnxYdUxBbf3p06eFBg0aCLa2toKvr68wfvx4Yf/+/W/V1uqKNp97QdC87Vq3bp1QvXp1wd7eXnB0dBRatWolHDlyROO49NG2F9WGaPI7unv3rjBo0CAhMDBQsLGxEdzc3IQWLVoIhw4dUivL399fre0SBEFYsWKFYGdnJ8jlco3rQBAEQfKiMojIxC1evBijR4/Gw4cP1RYzIiIiIuPCPp2IzBnbQMNz8uRJNG/eHLdu3VLN10368+uvv+K9997DH3/8gapVq4odDhEVs5o1a6J58+b46quvtHofkwBEJigjI0NtlfLMzEzUrFkTCoUCf/31l4iRERERkTbYpxOROWMbaDxCQkJQsmTJt16zgV5v3LhxiImJwcaNG8UOhYiK2b59+9C1a1fcvXsXnp6eWr2XSQAiExQSEoLSpUujRo0aSElJwY8//ogbN25gw4YN6NWrl9jhERERkYbYpxOROWMbSEREpBtcGJjIBLVt2xbffPMNNmzYAIVCgaCgIGzatCnfwj5ERERk2NinE5E5YxtIRESkGxwJQERERERERERERERkoqRiB0BERERERERERERERPrBJAARERERERERERERkYky6zUBlEolYmNj4ejoCIlEInY4REREAABBEJCamgpfX19IpczXF4R9OBERGRr236/H/puIiAyNufTfZp0EiI2NRalSpcQOg4iIqED//vsvSpYsKXYYBol9OBERGSr234Vj/01ERIbK1Ptvs04CODo6Anj+S3Zycnrr8pRKJZKSkuDh4WHSmSNdYp1pj3WmPdaZ9lhn2tNlncnlcpQqVUrVT1F+uuzD+XnXHutMe6wz7bHOtMc60x777+LF/tt4sb6LH+u8eLG+i5+h1Lm59N9mnQTIG37o5OSksyRAZmYmnJyc2GBoiHWmPdaZ9lhn2mOdaU8fdcZh8oXTZR/Oz7v2WGfaY51pj3WmPdaZ9th/Fy/238aL9V38WOfFi/Vd/Aytzk29/xa/homIiIiIiIiIiIiISC+YBCAiIiIiIiIiIiIiMlFmmQSIjIxEUFAQ6tatK3YoREREpAX24URERERERETaMcskQGhoKKKionDhwgWxQyEi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",
      "text/plain": [
       "<Figure size 1500x400 with 3 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Jawaban A5\n",
    "class LinearRegressionGD:\n",
    "    \"\"\"Linear Regression dengan Gradient Descent (batch).\"\"\"\n",
    "    def __init__(self, learning_rate=0.01, n_iterations=1000):\n",
    "        self.lr = learning_rate\n",
    "        self.n_iter = n_iterations\n",
    "\n",
    "    def fit(self, X, y):\n",
    "        n = X.shape[0]\n",
    "        self.intercept_ = 0.0\n",
    "        self.coef_ = np.zeros(X.shape[1])\n",
    "        self.loss_history_ = []\n",
    "        for _ in range(self.n_iter):\n",
    "            y_hat = self.intercept_ + X @ self.coef_\n",
    "            error = y_hat - y\n",
    "            self.loss_history_.append(np.mean(error ** 2))     # MSE\n",
    "            # gradien MSE\n",
    "            self.intercept_ -= self.lr * (2 / n) * np.sum(error)\n",
    "            self.coef_      -= self.lr * (2 / n) * (X.T @ error)\n",
    "        return self\n",
    "\n",
    "    def predict(self, X):\n",
    "        return self.intercept_ + X @ self.coef_\n",
    "\n",
    "learning_rates = [0.001, 0.02, 0.1]\n",
    "models_gd = {}\n",
    "rows = [{\"Model\": \"Normal Equation\", \"LR\": \"-\", \"Intercept\": b_ne, \"Koefisien\": m_ne,\n",
    "         \"MSE akhir (train)\": mse(y_train, lin.predict(X_train))}]\n",
    "\n",
    "for lr in learning_rates:\n",
    "    print(f\"--- learning rate = {lr} ---\")\n",
    "    gd = LinearRegressionGD(learning_rate=lr, n_iterations=5000).fit(X_train, y_train)\n",
    "    models_gd[lr] = gd\n",
    "    print(f\"Intercept: {gd.intercept_:.4f} | Koefisien: {gd.coef_[0]:.4f} | MSE akhir: {gd.loss_history_[-1]:.4f}\")\n",
    "    rows.append({\"Model\": \"Gradient Descent\", \"LR\": lr, \"Intercept\": gd.intercept_,\n",
    "                 \"Koefisien\": gd.coef_[0], \"MSE akhir (train)\": gd.loss_history_[-1]})\n",
    "\n",
    "print()\n",
    "display(pd.DataFrame(rows))\n",
    "\n",
    "# Grafik loss vs iterasi\n",
    "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n",
    "for ax, lr in zip(axes, learning_rates):\n",
    "    loss = np.array(models_gd[lr].loss_history_, dtype=float)\n",
    "    loss[~np.isfinite(loss)] = np.nan          # nilai inf/nan tidak digambar\n",
    "    if lr < 0.05:\n",
    "        ax.plot(loss)\n",
    "        ax.set_yscale(\"log\")\n",
    "        ax.set_ylabel(\"MSE (skala log)\")\n",
    "    else:\n",
    "        # loss divergen terlalu besar untuk skala log matplotlib -> gambar log10(MSE)\n",
    "        ax.plot(np.log10(loss), color=\"tab:red\")\n",
    "        ax.set_ylabel(\"log10(MSE)\")\n",
    "        n_valid = np.isfinite(loss).sum()\n",
    "        ax.set_title(f\"learning rate = {lr} (DIVERGEN, overflow setelah ~{n_valid} iterasi)\")\n",
    "    if lr < 0.05:\n",
    "        ax.set_title(f\"learning rate = {lr}\")\n",
    "    ax.set_xlabel(\"Iterasi\")\n",
    "    ax.grid(alpha=0.3)\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cell-18",
   "metadata": {},
   "source": [
    "**Jawaban A5 (analisis):**\n",
    "\n",
    "| Model | Intercept | Koefisien | MSE akhir (train) |\n",
    "|---|---|---|---|\n",
    "| Normal Equation | 27,6131 | 9,2416 | 26,3219 |\n",
    "| GD, lr = 0,001 | 25,2720 | 9,6112 | 27,6648 |\n",
    "| GD, lr = 0,02 | 27,6131 | 9,2416 | 26,3219 |\n",
    "| GD, lr = 0,1 | NaN | NaN | NaN |\n",
    "\n",
    "- **lr = 0,001 (terlalu lambat).** Loss turun mulus, tetapi setelah 5000 iterasi parameter (25,27 dan 9,61) masih berbeda dari solusi normal equation (27,61 dan 9,24), dan MSE (27,66) masih lebih tinggi dari 26,32. Langkahnya terlalu kecil, sehingga butuh iterasi jauh lebih banyak untuk konvergen.\n",
    "- **lr = 0,02 (pas).** Hasilnya identik dengan normal equation sampai 4 desimal. Loss turun cepat dan sudah hampir mencapai nilai akhirnya (sekitar 26,32) pada iterasi ke-500-an, lalu datar.\n",
    "- **lr = 0,1 (divergen).** Loss justru membesar: sekitar 1,8×10¹⁸ pada iterasi ke-10, lalu *overflow* pada iterasi sekitar ke-209, sehingga parameter menjadi `NaN`. Karena nilainya terlalu besar untuk skala log biasa, pada grafik lr = 0,1 yang digambar adalah log10(MSE), yang terlihat naik terus sampai putus saat overflow.\n",
    "\n",
    "**Penyebabnya:** gradient descent untuk MSE hanya stabil jika *learning rate* lebih kecil dari 2/λmax, dengan λmax adalah nilai eigen terbesar dari (2/n)XᵀX. Pada data training ini λmax ≈ 63,1, sehingga batas stabilnya sekitar **0,032**. lr = 0,02 berada di bawah batas ini sehingga konvergen, sedangkan lr = 0,1 berada di atasnya. Setiap langkah melompati titik minimum dan mendarat lebih jauh dari sebelumnya, sehingga error terus membesar. Sementara itu lr = 0,001 stabil, tetapi laju konvergennya pada arah yang \"datar\" (nilai eigen terkecil ≈ 0,48) sangat lambat.\n",
    "\n",
    "**Arti `RuntimeWarning: overflow`:** nilai error dan gradien menjadi begitu besar sehingga melampaui batas bilangan *float64* (sekitar 1,8×10³⁰⁸) dan menjadi `inf`. Operasi berikutnya (`inf − inf`) menghasilkan `NaN`. Ini tanda bahwa training gagal konvergen karena learning rate terlalu besar."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cell-19",
   "metadata": {},
   "source": [
    "### Soal A6 – Visualisasi & Prediksi\n",
    "1. Buat grafik *scatter* data **testing** beserta **garis regresi** hasil model (Soal A3).\n",
    "2. Prediksi penjualan jika biaya iklan sebesar **4.5 juta**, **7 juta**, dan **12 juta**.\n",
    "3. Prediksi untuk 12 juta berada **di luar rentang data** (ekstrapolasi). Menurut Anda, seberapa bisa dipercaya hasil prediksi tersebut? Jelaskan."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "cell-20",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-10-20T11:35:54.290263Z",
     "iopub.status.busy": "2026-10-20T11:35:54.289661Z",
     "iopub.status.idle": "2026-10-20T11:35:54.411259Z",
     "shell.execute_reply": "2026-10-20T11:35:54.410328Z"
    }
   },
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAmoAAAGwCAYAAAAZn0mrAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjgsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvwVt1zgAAAAlwSFlzAAAPYQAAD2EBqD+naQAAfxBJREFUeJzt3XlYVOX7x/H3sCMIuKBAIpJL7ktqlmauZaZmmZVmplYu5W6LaWpp7ntaaWpl9bX8qqFZfVvMJdHKfd9L0UpwF0QCYeb8/jgxvxBURoEZ4PO6rrkunnPOnHPPzai3zznP81gMwzAQEREREZfj5uwARERERCRrKtREREREXJQKNREREREXpUJNRERExEWpUBMRERFxUSrURERERFyUCjURERERF6VCTURERMRFqVATERERcVEq1ETEabp37065cuWcHUYG5cqVo3v37s4OI8+8+eabWCwWZ4eRbzRt2pSmTZs6OwwpRFSoieSAhQsXYrFY7C8fHx/CwsJo1aoVs2bN4tKlSzd97p9//pk333yTixcv5kisMTExGWK93ismJuaWr3fy5EnefPNNdu7cecvnKgyOHTtGv379qFSpEkWKFKFIkSJUrVqVvn37snv3bmeHd13lypXL8P3x8/Pjrrvu4pNPPnF2aCL5lkVrfYrcuoULF9KjRw/GjBlDZGQkqampxMXFsW7dOlatWkXZsmVZuXIlNWvWdPjcU6dO5ZVXXuHYsWM50vt0+fJlli9fnmHbtGnT+PPPP5kxY0aG7Y8++ih+fn63dL2tW7dSv359Pvroo0w9VampqdhsNry9vW/pGjmpXLlyNG3alIULF+b5tb/++muefPJJPDw86NKlC7Vq1cLNzY2DBw8SFRXF8ePHOXbsGBERETl2zbS0NNLS0vDx8bnlc5UrV45ixYrx0ksvARAbG8uCBQs4fPgw8+bNo2fPnrd8DWe7cuUKAF5eXk6ORAoLD2cHIFKQtG7dmnr16tnbw4YNY82aNbRt25aHH36YAwcO4Ovr68QIwc/Pj6effjrDtsWLF3PhwoVM23Obp6dnnl7Plf3+++906tSJiIgIVq9eTWhoaIb9kyZN4r333sPNLWduhFy+fBk/Pz88PDzw8Mi5fwpuu+22DN+j7t27c/vttzNjxow8L9TSP2NOUoEmeU23PkVyWfPmzRk5ciTHjx/nP//5j3377t277f+I+fj4EBISwrPPPsu5c+fsx7z55pu88sorAERGRma6JfnRRx/RvHlzSpUqhbe3N1WrVmXOnDk5EndKSgpvvPEGFSpUwNvbm/DwcF599VVSUlIyHLdq1SruvfdegoKC8Pf354477mD48OEArFu3jvr16wPQo0cPe/zpvVVXP6OWflt26tSpzJs3j/Lly+Pt7U39+vXZsmVLphiXLl1K1apV8fHxoXr16ixfvjzbz70ZhsHYsWMpU6YMRYoUoVmzZuzbty/TcefPn+fll1+mRo0a+Pv7ExAQQOvWrdm1a1eG49atW4fFYmHJkiWMGzeOMmXK4OPjQ4sWLfjtt99uGM/kyZO5fPkyH330UaYiDcDDw4MBAwYQHh5u35ad7xD8/3No+/fv56mnnqJYsWLce++9Gfb92/V+p44KDg6mcuXK/P777xm222w2Zs6cSbVq1fDx8aF06dL07t2bCxcuZDruzTffJCwszP572r9/f6ZnCdMfP/jpp5948cUXKVWqFGXKlLHv//bbb2ncuDF+fn4ULVqUNm3aZPp9x8XF0aNHD8qUKYO3tzehoaG0b98+wyMAekZN8pp61ETyQNeuXRk+fDg//PCDvVdh1apVHD16lB49ehASEsK+ffuYN28e+/bt49dff8VisdChQwcOHz7M559/zowZMyhZsiRg/uMHMGfOHKpVq8bDDz+Mh4cHX331FS+++CI2m42+ffvedLw2m42HH36YDRs20KtXL6pUqcKePXuYMWMGhw8fZsWKFQDs27ePtm3bUrNmTcaMGYO3tze//fYbGzduBKBKlSqMGTOGUaNG0atXLxo3bgxAw4YNr3v9zz77jEuXLtG7d28sFguTJ0+mQ4cOHD161N4L98033/Dkk09So0YNJkyYwIULF3juuee47bbbsvUZR40axdixY3nooYd46KGH2L59Ow888ID91la6o0ePsmLFCh5//HEiIyM5deoU77//Pk2aNGH//v2EhYVlOH7ixIm4ubnx8ssvEx8fz+TJk+nSpQubNm26bjxff/01FSpUoEGDBtmKH7L3Hfq3xx9/nIoVKzJ+/Hiu9dTLjX6njkpLS+PPP/+kWLFiGbb37t3b/sjAgAEDOHbsGO+88w47duxg48aN9t/zsGHDmDx5Mu3ataNVq1bs2rWLVq1akZycnOX1XnzxRYKDgxk1ahSXL18G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2bMz5C+r/Lu7u2f6PS1dupS//vrr5j9AFl599VWKFCnCs88+y6lTpzLtvzoGR75DjsjO79RRQ4cO5dy5c/bv0xNPPIHVauWtt97KdGxaWpr9u9KiRQs8PDwyTTnzzjvvZPvarVq1IiAggPHjx5Oamppp/5kzZwBISkrKNJK0fPnyFC1a9KY/t0hOUI+aSA769ttvOXjwIGlpaZw6dYo1a9awatUqIiIiWLlypX3294CAAPszMKmpqdx222388MMPHDt2LNM569atC8Drr79Op06d8PT0pF27djzwwAN4eXnRrl07evfuTWJiIvPnz6dUqVLExsbe0ufo2rUrS5YsoU+fPqxdu5ZGjRphtVo5ePAgS5Ys4fvvv6devXqMGTOG9evX06ZNGyIiIjh9+jTvvfceZcqUsc/TVb58eYKCgpg7dy5FixbFz8+PBg0aZOt5qesZP3487du3p1GjRvTo0YMLFy7wzjvvUL169Syf9fq34OBgXn75ZSZMmEDbtm156KGH2LFjB99++22mXrK2bdsyZswYevToQcOGDdmzZw+LFi3K0BOXEypWrMhnn31G586dueOOO+wrExiGwbFjx/jss89wc3Ozzw3myHfIEdn5nTqqdevWVK9enenTp9O3b1+aNGlC7969mTBhAjt37uSBBx7A09OTI0eOsHTpUt5++206duxI6dKlGThwINOmTePhhx/mwQcfZNeuXfbfU3Z6awMCApgzZw5du3blzjvvpFOnTgQHB3PixAm++eYbGjVqxDvvvMPhw4dp0aIFTzzxBFWrVsXDw4Ply5dz6tQpOnXqdFOfWyRHOG28qUgBkj41QPrLy8vLCAkJMe6//37j7bffNhISEjK9588//zQeffRRIygoyAgMDDQef/xx4+TJkwZgvPHGGxmOfeutt4zbbrvNcHNzyzBVx8qVK42aNWsaPj4+Rrly5YxJkyYZH3744TWn87iWq6fnMAzDuHLlijFp0iSjWrVqhre3t1GsWDGjbt26xujRo434+HjDMAxj9erVRvv27Y2wsDDDy8vLCAsLMzp37mwcPnw4w7m+/PJLo2rVqoaHh0eGqTquNT1HVlMkZJWXxYsXG5UrVza8vb2N6tWrGytXrjQee+wxo3Llyjf8zFar1Rg9erQRGhpq+Pr6Gk2bNjX27t1rREREZJqe46WXXrIf16hRI+OXX37J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+9ZnVr2BO3fuxGKxEBMTw7p16+jRowfx8fH2nrD063766afUq1ePokWLEhISwlNPPcXp06dvmI8iRYoQEhJCREQEPXr0oGbNmqxatcq+Pz2mb775hpo1a+Lj48Pdd9/N3r17M5ynXbt2bN26ld9///2G1xSR7FGhJiI3JTExkf/85z9UqFCBEiVKZHmMzWajTJkyLF26lP379zNq1CiGDx/OkiVLALjvvvu4/fbb+fTTT+3vSU1NZdGiRTz77LMAJCcnU7duXb755hv27t1Lr1696Nq1K5s3b76puP/44w8aN25M9erVWbZsWZY9Wdm95scff4yfnx+bNm1i8uTJjBkzJkOBc7Xo6GgqVapE0aJFbyp2gIYNGzJz5kwCAgLsvZsvv/wyYOburbfeYteuXaxYsYKYmBi6d++e7XMbhkF0dDQHDx7Ey8sr0/5XXnmFadOmsWXLFoKDg2nXrh2pqan2/WXLlqV06dJER0ff9OcTkas4dUl4Eck3unXrZri7uxt+fn6Gn5+fARihoaHGtm3b7MccO3bMAIwdO3Zc8zx9+/Y1HnvsMXt70qRJRpUqVeztL774wvD39zcSExOveY42bdoYL7300jX3r1271gCMCxcuGIZhGB999JERGBhoHDx40AgPDzcGDBhg2Gy2ax6fnWs2adLEuPfeezMcU79+fWPo0KHXPMfAgQON5s2bZ9oOGMuXL79mLDt27DAA49ixYxk+z41s2bLFAIxLly5d85gmTZoYnp6ehp+fn+Hp6WkAho+Pj7Fx40b7MekxLV682L7t3Llzhq+vr/Hf//43w/nq1KljvPnmmzeMTUSyRz1qIpJtzZo1Y+fOnezcuZPNmzfTqlUrWrduzfHjx6/5nnfffZe6desSHByMv78/8+bN48SJE/b93bt357fffuPXX38FzAfln3jiCfz8/ADz9ulbb71FjRo1KF68OP7+/nz//fcZzpEdf//9N40bN6ZDhw68/fbbWCyWax6b3WvWrFkzQzs0NPS6txr//vtvfHx8HIrbEdu2baNdu3aULVuWokWL0qRJE4Ab5qpLly7s3LmTjRs30rp1a15//fUMgx3S3XPPPfafixcvzh133MGBAwcyHOPr60tSUlIOfBoRAd36FBEH+Pn5UaFCBSpUqED9+vVZsGABly9fZv78+Vkev3jxYl5++WWee+45fvjhB3bu3EmPHj24cuWK/ZhSpUrRrl07PvroI06dOsW3335rv+0JMGXKFN5++22GDh3K2rVr2blzJ61atcpwjuzw9vamZcuWfP311/z111/XPTa71/T09MzQtlgs152eomTJkly4cOG613ZzM/9aNv416vLftxev5fLly7Rq1YqAgAAWLVrEli1bWL58OcANcxUYGGj/nS5ZsoR33nmHH3/88YbXzMr58+cJDg6+qfeKSGYamiUiN81iseDm5sbff/+d5f6NGzfSsGFDXnzxRfu2rB40f/755+ncuTNlypShfPnyNGrUKMM52rdvz9NPPw2Yz70dPnyYqlWrOhSrm5sbn376KU899RTNmjVj3bp1hIWFXTPunLjm1erUqcOcOXMwDOOaPXrpRU5sbCzFihUDyDQvnZeXF1arNcO2gwcPcu7cOSZOnEh4eDgAW7dudThGf39/Bg4cyMsvv8yOHTsyxPnrr79StmxZAC5cuMDhw4epUqWKfX9ycjK///47derUcfi6IpI19aiJSLalpKQQFxdHXFwcBw4coH///iQmJtKuXbssj69YsSJbt27l+++/5/Dhw4wcOZItW7ZkOi69J2js2LH06NEj0zlWrVrFzz//zIEDB+jduzenTp26qfjd3d1ZtGgRtWrVonnz5plGm+bGNf+tWbNmJCYmsm/fvmseU6FCBcLDw3nzzTc5cuQI33zzDdOmTctwTLly5UhMTGT16tWcPXuWpKQkypYti5eXF7Nnz+bo0aOsXLmSt95666bi7N27N4cPH+aLL77IsH3MmDGsXr2avXv30r17d0qWLMkjjzxi3//rr7/i7e2d4RapiNwaFWoikm3fffcdoaGhhIaG0qBBA7Zs2cLSpUtp2rRplsf37t2bDh068OSTT9KgQQPOnTuXoXctnZubG927d8dqtfLMM89k2DdixAjuvPNOWrVqRdOmTQkJCclQHDjKw8ODzz//nGrVqtG8efMsnynL6WumK1GiBI8++uh1J4X19PTk888/5+DBg9SsWZNJkyYxduzYDMc0bNiQPn368OSTTxIcHMzkyZMJDg5m4cKFLF26lKpVqzJx4kSmTp16U3EWL16cZ555hjfffDPDrdyJEycycOBA6tatS1xcHF999VWG0aGff/45Xbp0oUiRIjd1XRHJzGIYDk4/LSKSC5577jnOnDnDypUrnR1Krtq9ezf3338/v//+O/7+/qSkpODj48OqVato2bKls8PL0rp162jWrBkXLly45ooIZ8+e5Y477mDr1q1ERkbmbYAiBZieURMRp4qPj2fPnj189tlnBb5IA+y9ZMeOHSMiIoKoqCjc3NyoXLmys0O7JTExMbz33nsq0kRymAo1EXGq9u3bs3nzZvr06cP999/v7HDyRPoktIMHD+azzz5j0qRJlClTxrlB3aJ69epRr149Z4chUuDo1qeIiIiIi9JgAhEREREXpUJNRERExEWpUBMRERFxUSrURERERFyUCjURERERF6VCTURERMRFqVATERERcVEq1ERERERclAo1ERERERf1f1WH51PwJWzhAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 700x450 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Biaya iklan  4.5 juta -> prediksi penjualan = 69.20 juta\n",
      "Biaya iklan  7.0 juta -> prediksi penjualan = 92.30 juta\n",
      "Biaya iklan 12.0 juta -> prediksi penjualan = 138.51 juta\n",
      "Rentang BiayaIklan pada data: 0.8 s/d 9.9\n"
     ]
    }
   ],
   "source": [
    "# Jawaban A6\n",
    "# 1. Scatter data testing + garis regresi\n",
    "x_line = np.linspace(0, 13, 100).reshape(-1, 1)\n",
    "plt.figure(figsize=(7, 4.5))\n",
    "plt.scatter(X_test, y_test, color=\"steelblue\", edgecolor=\"k\", label=\"Data testing\")\n",
    "plt.plot(x_line, lin.predict(x_line), color=\"red\", label=\"Garis regresi\")\n",
    "plt.title(\"Data Testing dan Garis Regresi\")\n",
    "plt.xlabel(\"Biaya Iklan (juta Rp)\")\n",
    "plt.ylabel(\"Penjualan (juta Rp)\")\n",
    "plt.legend()\n",
    "plt.grid(alpha=0.3)\n",
    "plt.show()\n",
    "\n",
    "# 2. Prediksi\n",
    "biaya_baru = np.array([[4.5], [7.0], [12.0]])\n",
    "pred_baru = lin.predict(biaya_baru)\n",
    "for x, p in zip(biaya_baru.ravel(), pred_baru):\n",
    "    print(f\"Biaya iklan {x:>4} juta -> prediksi penjualan = {p:.2f} juta\")\n",
    "print(\"Rentang BiayaIklan pada data:\", X_ads.min(), \"s/d\", X_ads.max())"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cell-21",
   "metadata": {},
   "source": [
    "**Jawaban A6 (analisis):**\n",
    "\n",
    "Prediksi penjualan (model normal equation):\n",
    "\n",
    "| Biaya iklan | Prediksi penjualan |\n",
    "|---|---|\n",
    "| 4,5 juta | 69,20 juta |\n",
    "| 7 juta | 92,30 juta |\n",
    "| 12 juta | 138,51 juta |\n",
    "\n",
    "Prediksi untuk **4,5 juta** dan **7 juta** berada dalam rentang data (0,8–9,9 juta), sehingga cukup dapat dipercaya. Prediksi untuk **12 juta** adalah **ekstrapolasi** (di luar rentang data) dan kurang dapat dipercaya. Model hanya belajar dari biaya iklan 0,8–9,9 juta, dan hasil 138,5 juta sudah melebihi penjualan tertinggi di data (121,1 juta). Garis lurus mengasumsikan hubungan linear tetap berlaku selamanya, padahal di dunia nyata efek iklan biasanya mengalami *diminishing returns* (penambahan iklan memberi tambahan penjualan yang makin kecil) atau terbatas oleh ukuran pasar dan stok. Tidak ada data yang membuktikan pola linear tetap berlaku di atas 9,9 juta, sehingga angka ini hanya boleh dipakai sebagai perkiraan kasar."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cell-22",
   "metadata": {},
   "source": [
    "---\n",
    "# Bagian B – Logistic Regression (40%)\n",
    "\n",
    "Sebuah layanan email ingin membuat filter **spam** sederhana berdasarkan **jumlah link** dan **jumlah kata promosi** (misal: \"gratis\", \"diskon\", \"promo\") di dalam email. Gunakan dataset `Email_Data.csv`."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cell-23",
   "metadata": {},
   "source": [
    "### Soal B1 – Load & Visualisasi Data\n",
    "1. Baca `Email_Data.csv`, tampilkan 5 data pertama, dan hitung **jumlah email Spam dan Bukan Spam**.\n",
    "2. Pisahkan `X` (`JumlahLink`, `KataPromo`) dan `y` (`Spam`).\n",
    "3. Buat *scatter plot* `JumlahLink` vs `KataPromo` dengan warna berbeda untuk Spam dan Bukan Spam (lengkap dengan legend)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "cell-24",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-10-20T11:35:54.413554Z",
     "iopub.status.busy": "2026-10-20T11:35:54.412922Z",
     "iopub.status.idle": "2026-10-20T11:35:54.590969Z",
     "shell.execute_reply": "2026-10-20T11:35:54.589615Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5 data pertama:\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>JumlahLink</th>\n",
       "      <th>KataPromo</th>\n",
       "      <th>Spam</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>6</td>\n",
       "      <td>7</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>10</td>\n",
       "      <td>8</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>8</td>\n",
       "      <td>7</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   JumlahLink  KataPromo  Spam\n",
       "0           0          0     0\n",
       "1           2          4     0\n",
       "2           6          7     1\n",
       "3          10          8     1\n",
       "4           8          7     1"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Jumlah Spam (1)       : 25\n",
      "Jumlah Bukan Spam (0) : 35\n",
      "Shape X: (60, 2) | Shape y: (60,)\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAmEAAAHWCAYAAAA/0l4bAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjgsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvwVt1zgAAAAlwSFlzAAAPYQAAD2EBqD+naQAAaZlJREFUeJzt3XlcVOX+B/DPOYMDyqbBsClbiGulhktquEWpmVmEZmlXs7q/m7uYW7mmRWqZtlB2W2yxRRHL1pvXVCg1CcXquoAEaIiCG4PLMDjz/P4wJodFthnOmeHzfr141Zxz5pzv4/c8hy/nPPOMJIQQICIiIqJGJSsdABEREVFTxCKMiIiISAEswoiIiIgUwCKMiIiISAEswoiIiIgUwCKMiIiISAEswoiIiIgUwCKMiIiISAEswoiIiIgUwCKMqAkYMGAABgwYUOf35ebmQpIkvPjii7YP6hqSJGHx4sX1fu/kyZNtGxARUSNgEUbUSNatWwdJkvDLL78oHUqj2bFjByRJQlJSktKhKKK6AvH555+HJEmYMGECzGZzrfd38OBBLF68GLm5uTaM8u9zs/zHzc0N7dq1w+TJk3Hq1CmbHouI/uaidABERJcvX4aLS9O4HL3wwgt45plnMG7cOLz99tuQ5dr/LXzw4EEsWbIEAwYMQFhYmM1je/bZZxEeHg6DwYAff/wRb7zxBr755hv8/vvvaNGihc2PR9TUNY2rHhGpmpubm9IhNIqVK1di3rx5+Mc//oF33323TgVYYxg6dCi6d+8OAHj88cfh4+ODVatW4YsvvsBDDz1U5XsuXrwId3f3xgyTyGmo6wpA1IRUN05r/PjxVnc5rh2X9frrr+PGG29EixYtcNddd+H48eMQQmDp0qVo06YNmjdvjhEjRuDs2bPXPbbRaMTChQsRFRUFb29vuLu7Izo6Gtu3b6/2PW+99RYiIiLg6uqKHj16IC0trb5Nr6TimLDFixdDkiQcPXoU48ePR8uWLeHt7Y1HH30Uly5dqnF/y5YtgyzLePXVV6vd5qabbsLAgQMrLTebzWjdujXi4uIsyz799FNERUXB09MTXl5euPnmm7FmzZo6tXHVqlWYPXs2xo4di/fee8+qAMvLy8PEiRPRvn17NG/eHD4+Phg5cqTVY8d169Zh5MiRAICBAwdaHh3u2LEDAPDFF19g2LBhCAoKgqurKyIiIrB06VKYTKY6xXmtQYMGAQBycnIAXD03PTw8kJ2djbvvvhuenp4YM2YMgKvF2MyZMxEcHAxXV1e0b98eL774IoQQVvssf0S7ceNGdOrUCc2bN0fv3r3x22+/AQDWrl2Ltm3bws3NDQMGDKjy0evGjRsRFRWF5s2bw9fXF2PHjkV+fn6920mkFN4JI3IQ69evh9FoxJQpU3D27FmsWLECo0aNwqBBg7Bjxw7MmTMHR48exauvvoqnnnoK7777brX70uv1ePvtt/HQQw/hiSeeQElJCd555x0MHjwYe/fuRdeuXa22//jjj1FSUoL/+7//gyRJWLFiBWJjY/HHH3+gWbNmdmvzqFGjEB4ejoSEBOzbtw9vv/02/Pz8sHz58mrfM3/+fDz//PNYu3YtnnjiiWq3e/DBB7F48WKcPHkSAQEBluU//vgjTpw4gdGjRwMAtm7dioceegh33HGH5biHDh3CTz/9hGnTptWqHWvWrMHMmTPx8MMPY926dZXugKWlpWHXrl0YPXo02rRpg9zcXLzxxhsYMGAADh48iBYtWqBfv36YOnUqXnnlFTz99NPo2LEjAFj+u27dOnh4eCA+Ph4eHh744YcfsHDhQuj1eqxcubJWcVaUnZ0NAPDx8bEsu3LlCgYPHozbb78dL774Ilq0aAEhBO69915s374djz32GLp27Yr//Oc/mDVrFvLz8/Hyyy9b7Tc1NRVbtmzBpEmTAAAJCQm45557MHv2bCQmJmLixIk4d+4cVqxYgQkTJuCHH36wvHfdunV49NFH0aNHDyQkJODUqVNYs2YNfvrpJ+zfvx8tW7asV1uJFCGIqFG89957AoBIS0sTQgjRv39/0b9//0rbjRs3ToSGhlpe5+TkCABCp9OJ8+fPW5bPmzdPABBdunQRZWVlluUPPfSQ0Gq1wmAwWJZVPNaVK1dEaWmp1XHPnTsn/P39xYQJEyod28fHR5w9e9ay/IsvvhAAxJdffnndNm/fvl0AEBs3brzudgDEokWLLK8XLVokAFjFIoQQ999/v/Dx8an03kmTJgkhhJg5c6aQZVmsW7fuuscTQogjR44IAOLVV1+1Wj5x4kTh4eEhLl26JIQQYtq0acLLy0tcuXKlxn1W1a7Q0FABQDz00EPV7qP8WNfavXu3ACA++OADy7KNGzcKAGL79u212sf//d//iRYtWlidC1UpPzf/+9//iqKiInH8+HHx6aefCh8fH9G8eXPx559/CiGunpsAxNy5c63e//nnnwsAYtmyZVbL4+LihCRJ4ujRo5ZlAISrq6vIycmxLFu7dq0AIAICAoRer7csLz/Hy7c1Go3Cz89P3HTTTeLy5cuW7b766isBQCxcuPC67SRSGz6OJHIQI0eOhLe3t+V1r169AABjx461GtTeq1cvGI3G6z6e0Wg00Gq1AK4+fjt79iyuXLmC7t27Y9++fZW2f/DBB9GqVSvL6+joaADAH3/80bBG1eBf//qX1evo6GicOXMGer3earkQApMnT8aaNWvw0UcfYdy4cTXuu127dujatSs+++wzyzKTyYSkpCQMHz4czZs3BwC0bNkSFy9exNatW+vVhvJPF4aHh0Oj0VS5TfmxAKCsrAxnzpxB27Zt0bJlyyrzUdM+SkpKcPr0aURHR+PSpUs4fPhwrfYRExMDnU6H4OBgjB49Gh4eHti8eTNat25ttd2TTz5p9fqbb76BRqPB1KlTrZbPnDkTQgh8++23VsvvuOMOq0fu5efyAw88AE9Pz0rLy8+zX375BYWFhZg4caLVOMJhw4ahQ4cO+Prrr2vVTiK14ONIIgcREhJi9bq8IAsODq5y+blz5667v/fffx8vvfQSDh8+jLKyMsvy8PDwGo9dXpDVdIyGut5xvby8LMs/+OADXLhwAW+88Ua1A8ir8uCDD+Lpp59Gfn4+WrdujR07dqCwsBAPPvigZZuJEydiw4YNGDp0KFq3bo277roLo0aNwpAhQ2p1jHHjxuHEiRN4/vnn4evrixkzZlTa5vLly0hISMB7772H/Px8q3FUxcXFtTrO//73P8yfPx8//PBDpSK1tvt4/fXX0a5dO7i4uMDf3x/t27ev9OjUxcUFbdq0sVqWl5eHoKAgqwIK+PtRaV5entXy+p7L5ftp3759pdg7dOiAH3/8seZGEqkI74QRKUSSpCqXVzeQurq7KNUtFxUGRF/ro48+wvjx4xEREYF33nkH3333HbZu3YpBgwZVOW9VfY5hC7U9bt++feHv74/XXnutxg8lXOvBBx+EEAIbN24EAGzYsAHe3t5WBZafnx8yMjKwZcsWy7inoUOH1upuG3C1aNmwYQP69++PmTNn4r333qu0zZQpU/Dcc89h1KhR2LBhA77//nts3boVPj4+tZpH7Pz58+jfvz8OHDiAZ599Fl9++SW2bt1qGcNW27nIevbsiZiYGAwYMAAdO3as8tObrq6uDf5Upy3PZSJHxjthRApp1apVlY/zKt41sIekpCTceOONSE5OtioGFy1aZPdj20Pbtm2xYsUKDBgwAEOGDMG2bdsq3ZWpSnh4OHr27InPPvsMkydPRnJyMu677z64urpabafVajF8+HAMHz4cZrMZEydOxNq1a7FgwQK0bdu2xuO4ublhy5YtGDhwIJ544gm0bNkS999/v2V9UlISxo0bh5deesmyzGAw4Pz581b7qa5w37FjB86cOYPk5GT069fPsrz8U432Fhoaiv/+978oKSmx+ncvfwwaGhpqs+MAwJEjRyyf3Cx35MgRmx2HqLHwThiRQiIiInD48GEUFRVZlh04cAA//fST3Y9dfsfh2jsMP//8M3bv3m33Y9vLLbfcgm+++QaHDh3C8OHDcfny5Vq978EHH8SePXvw7rvv4vTp01aPIgHgzJkzVq9lWcYtt9wCACgtLa11fF5eXvjuu+/Qtm1bPPTQQ9i2bZtlnUajqXS359VXX610V7R8Pq6KxVlV+TQajUhMTKx1fA1x9913w2Qy4bXXXrNa/vLLL0OSJAwdOtQmx+nevTv8/Pzw5ptvWv3bf/vttzh06BCGDRtmk+MQNRbeCSNSyIQJE7Bq1SoMHjwYjz32GAoLC/Hmm2+ic+fOlcb02No999yD5ORk3H///Rg2bBhycnLw5ptvolOnTrhw4YLNj7dp06YqB4ePGzeu0jighrjtttvwxRdf4O6770ZcXBw+//zzGqfQGDVqFJ566ik89dRTuOGGGxATE2O1/vHHH8fZs2cxaNAgtGnTBnl5eXj11VfRtWtXy5in2tLpdNi6dSv69u2L++67D9u2bUPPnj1xzz334MMPP4S3tzc6deqE3bt347///a/V1BAA0LVrV2g0GixfvhzFxcVwdXXFoEGD0KdPH7Rq1Qrjxo3D1KlTIUkSPvzww0Z7jDd8+HAMHDgQzzzzDHJzc9GlSxd8//33+OKLLzB9+nRERETY5DjNmjXD8uXL8eijj6J///546KGHLFNUhIWFVTnejkjNWIQRNZLyX4jldy06duyIDz74AAsXLkR8fDw6deqEDz/8EB9//LFlAk57GT9+PE6ePIm1a9fiP//5Dzp16oSPPvoIGzdutMuxP/300yqXDxgwwKZFGHB1gtENGzbggQcewCOPPIKPP/74umOY2rRpgz59+uCnn37C448/XqloGzt2LN566y0kJibi/PnzCAgIsMwxVp+xUcHBwfj+++8RHR2NoUOHIiUlBWvWrIFGo8H69ethMBjQt29f/Pe//8XgwYOt3hsQEIA333wTCQkJeOyxx2AymbB9+3YMGDAAX331FWbOnIn58+ejVatWGDt2LO64445K+7AHWZaxZcsWLFy4EJ999hnee+89hIWFYeXKlZg5c6ZNjzV+/Hi0aNECL7zwAubMmQN3d3fcf//9WL58OecII4cjCY54JGoUr7zyCqZNm4ajR4/a7M4AERE5Lo4JI2okaWlpcHd35+BhIiICwMeRRHa3adMm7NixA+vXr8fjjz9uNbEqERE1XXwcSWRn4eHhKCkpwf3334/Vq1dbPuFGRERNG4swIiIiIgVwTBgRERGRAliEERERESnA6UcIm81mnDhxAp6entV+5QcRERGRLQghUFJSgqCgoBrnEnT6IuzEiRM2nwySiIiI6HqOHz+ONm3aXHcbpy/Cyr9M9vjx4/Dy8rLLMcxmM4qKiqDT6eo1gzbZBvOgHsyFOjAP6sFcqIe9c6HX6xEcHGz1ZfbVcfoirPwRpJeXl12LMIPBAC8vL3YuBTEP6sFcqAPzoB7MhXo0Vi5qMwSKZwIRERGRAliEERERESmARRgRERGRAliEERERESmARRgRERGRAliEERERESmARRgRERGRAliEERERESmARRgRERGRApx+xnwiR5CVlYWSkpJq13t6eiIyMrIRIyIi4rXJ3hQtwlJSUrBy5Uqkp6ejoKAAmzdvxn333QcAKCsrw/z58/HNN9/gjz/+gLe3N2JiYvDCCy8gKChIybCJbCorKwvt2rWrcbvMzExe7Iio0fDaZH+KFmEXL15Ely5dMGHCBMTGxlqtu3TpEvbt24cFCxagS5cuOHfuHKZNm4Z7770Xv/zyi0IRE9le+V+ZywMDEaF1rbQ+21iKOQUF1/1rlIjI1nhtsj9Fi7ChQ4di6NChVa7z9vbG1q1brZa99tpr6NmzJ44dO4aQkJDGCJGo0URoXdHJzU3pMIiIrPDaZD8ONSasuLgYkiShZcuW1W5TWlqK0tJSy2u9Xg/g6remm81mu8RlNpshhLDb/ql2HDUPQgjIsgzIMoRcxWdlZBmyLDtU2xw1F86GeVAPR8yFM16bAPvnoi77dZgizGAwYM6cOXjooYfg5eVV7XYJCQlYsmRJpeVFRUUwGAx2ic1sNqO4uPjvE5YU4ah5MBgMiIqKgqTzw0WtttJ6yWhElK8PDAYDCgsLFYiw7hw1F86GeVAPR8yFM16bAPvnoi6PZx2iCCsrK8OoUaMghMAbb7xx3W3nzZuH+Ph4y2u9Xo/g4GDodLrrFm8NYTabIUkSdDqdw3QuZ+SoecjPz0d6ejpEaBjcq7jlLwwGpOflws3NDX5+fgpEWHeOmgtnwzyohyPmwhmvTYD9c+FWh0e3qi/CyguwvLw8/PDDDzUWUq6urnB1rTyAUP7rtqm9SJJk92NQzRwxD5IkXb19bTZDquo29l+P0svb5igcMRfOiHlQD0fLhbNemwD75qIu+1R1EVZegGVlZWH79u3w8fFROiQiu8k2ltZpORFRY+C1yX4ULcIuXLiAo0ePWl7n5OQgIyMDN9xwAwIDAxEXF4d9+/bhq6++gslkwsmTJwEAN9xwA7RVPJ8mckSenp4AgDkFBbXajoioMfDaZH+KFmG//PILBg4caHldPpZr3LhxWLx4MbZs2QIA6Nq1q9X7tm/fjgEDBjRWmER2FRkZiczMTM5KTUSqwmuT/SlahA0YMABCiGrXX28dkTPhRYyI1IjXJvtyrJF0RERERE6CRRgRERGRAliEERERESmARRgRERGRAliEERERESmARRgRERGRAliEERERESmARRgRERGRAliEERERESmARRgRERGRAliEERERESmARRgRERGRAliEERERESmARRgRERGRAliEERERESmARRgRERGRAliEERERESnARekAHJ3JZEJKSgqKioqg0+nQr18/aDQapcMiB5OVlYWSkpJq13t6eiIyMrIRI6q7a9sghIDBYEB+fj4kSQLgGG0gImvOcG1SMxZhDZCcnIzpM+KR/+dxREVFIT09Ha3bBGP1y6sQGxurdHjkILKystCuXbsat8vMzFTtxa5iG2RZtvQJs9lsWa7mNhCRNWe4Nqkdi7B6Sk5ORlxcHJpH9EDAmCnwbR+GgA65OLN7A+Li4pCUlMRCjGql/K/M5YGBiNC6VlqfbSzFnIKC6/41qrRKbZBlSDo/iNAwwGx2iDYQkTVnuDapHYuwejCZTJg+Ix7NI3rAN3Y+NJIEyUXANag9fGPn43TyMkyPn4kRI0bw0STVWoTWFZ3c3JQOo0HK2yBkGRe1Wri7uUG65k4YETkeZ7g2qRUH5tdDamoqjh/Lg9dtoyBJ1v+EkiTD67aROJ6Xi9TUVIUiJCIiIrVjEVYPBQUFAIBmutAq1zfzDbXajoiIiKgiFmH1EBgYCAAoK8qrcn3Z6Tyr7YiIiIgqYhFWD9HR0QgOCYV+zwYIYT3eRQgz9Hs2Ijg0DNHR0QpFSERERGrHgfn1oNFosPrlVYiLi8Pp5GVo2XskhEcYSk/k4vzujbicnYbVSUkclE91km0srdNyNbLEKsuQjEYIg8Hy6UgickzOcG1SKxZh9RQbG4ukpKSr84Stn4PWUVE4mZ6O1sEhWM3pKagOPD09AQBzahhDWL6dGlVsgyzLiPL1QXpertU8YWpuAxFZc4Zrk9qxCGuA2NhYjBgxgjPmU4NERkYiMzPToWelrtiG8hnz3dzcOGM+kYNyhmuT2rEIayCNRoP+/fujsLAQfn5+kGUOs6O6c4aL2LVtMJvN7BNETsAZrk1qxqsjERERkQJYhBEREREpgEUYERERkQJYhBEREREpgEUYERERkQJYhBEREREpgEUYERERkQJYhBEREREpgEUYERERkQJYhBEREREpgEUYERERkQJYhBEREREpgEUYERERkQJYhBEREREpgEUYERERkQJYhBEREREpgEUYERERkQJYhBEREREpwEXpAIgIyMrKQklJCUwmE/bv34/Tp0/D19cX3bp1g0ajgaenJyIjI5UOk6jRlPeJ6jhCn7i2DUIIGAwG5OfnQ5IkAI7RBrIvRYuwlJQUrFy5Eunp6SgoKMDmzZtx3333WdYLIbBo0SL8+9//xvnz59G3b1+88cYbPGnJqWRlZaFdu3Y1bpeZmclzn5oEZ+gTFdsgyzKioqKQnp4Os9lsWa7mNpD9KVqEXbx4EV26dMGECRMQGxtbaf2KFSvwyiuv4P3330d4eDgWLFiAwYMH4+DBg3Bzc1MgYiLbK/9LeXlgICK0rpXWZxtLMaeg4Lp3BYiciTP0iUptkGVIOj+I0DDAbHaINpD9KVqEDR06FEOHDq1ynRACq1evxvz58zFixAgAwAcffAB/f398/vnnGD16dGOGSmQ3JpMJABChdUWn6/xxUb4dUVNRU59wBOVtELKMi1ot3N3cIF1zJ4yaNtWOCcvJycHJkycRExNjWebt7Y1evXph9+7d1RZhpaWlKC0ttbzW6/UAALPZbHUL2JbMZjOEEHbbP9WOo+Zh//79kGUZkGUIuYrPysgyZFnG/v37ERUV1fgB1oOj5sLZOGoehBC16hNqblvFNghZhpCkv9vjAG1wVvbuF3XZr2qLsJMnTwIA/P39rZb7+/tb1lUlISEBS5YsqbS8qKgIBoPBtkH+xWw2o7i4+O9OR4pw1DwYDAZERUVB0vnholZbab1kNCLK1wcGgwGFhYUKRFh3jpoLZ+OoeXCGPlGxDUKSUNq6NQBAEsIh2uCs7N0v6vKIWbVFWH3NmzcP8fHxltd6vR7BwcHQ6XTw8vKyyzHNZjMkSYJOp3OoC52zcdQ8uLm5IT09HSI0DO5VPHoRBgPS83Lx+OOPw8/PT4EI685Rc+FsHDUP+fn5teoTbm5uqu0TFdtQfgfMPTMTktnsEG1wVvbuF3UZs67aIiwgIAAAcOrUKQQGBlqWnzp1Cl27dq32fa6urnB1rTyQU/7r1q+9SJJk92NQzRwxD926dbt6+9psrnqsyF+P0rt16+ZQ7XLEXDgjR8yDJEm16hPlbVOjqtogCQGp/LUDtMGZ2bNf1GWfqi3CwsPDERAQgG3btlmKLr1ej59//hlPPvmkssER2ZBGowFw9RNfVSlfXr4dUVNRU59wBJZYZRmS0QhhMFg+HUmkaBF24cIFHD161PI6JycHGRkZuOGGGxASEoLp06dj2bJliIyMtExRERQUZDWXGJGj8/T0BADMKSio1XZEzs4Z+kTFNsiyjChfH6Tn5VoN3FZzG8j+FC3CfvnlFwwcONDyunws17hx47Bu3TrMnj0bFy9exD//+U+cP38et99+O7777jvOEUZOJTIyEpmZmZwxn+gv1/aJ6qi9T1RsQ/mM+W5ubpwxnywkIYRQOgh70uv18Pb2RnFxsV0H5hcWFsLPz4/P9hXEPKgHc6EOzIN6MBfqYe9c1KXu4JlAREREpAAWYUREREQKYBFGREREpAAWYUREREQKYBFGREREpAAWYUREREQKYBFGREREpAAWYUREREQKYBFGREREpAAWYUREREQKYBFGREREpAAWYUREREQKYBFGREREpAAWYUREREQKYBFGREREpAAWYUREREQKYBFGREREpAAXpQMg5WRlZaGkpAQmkwn79+/H6dOn4evri27dukGj0cDT0xORkZFKh9mkmEwmpKamoqCgAIGBgYiOjoZGo1E6rFopP58AQAgBg8GA/Px8SJIEAA5xPl3bhqo4Qhu2bt2KwsJCy2uNRgOTyWR57efnhzvvvFOJ0MgBOUOfUDMWYU1UVlYW2rVrV+N2mZmZ7GCNJDk5GdNnxOP4sTzLsuCQUKx+eRViY2MVjKxmFc8nWZYRFRWF9PR0mM1my3I1n0/O0Ce2bt2Ku+66y/K6ujx8//33LMSoRs7QJ9SORVgTVf6XzfLAQERoXSutzzaWYk5BwXX/AiLbSU5ORlxcHJpH9EDA2ClopgtFWVEezuzZgLi4OCQlJam6EKt0PskyJJ0fRGgYYDY7xPnkDH2i/A5YTXm49k4ZUXWcoU+oHYuwJqr88USE1hWd3Nxq3I7sx2QyYfqMeDSP6AHf2PmQpKtDNV1bd4Bv7HycTl6G6fEzMWLECNU/miw/n4Qs46JWC3c3N0jX3IFxBDX1CUfgDHkg9XCGPqFWHJjfRO3fv9+m21H9paam4vixPHjdNspSgJWTJBlet43E8bxcpKamKhQhERHZA4uwJur06dM23Y7qr6CgAADQTBda5fpmvqFW2xERkXNgEdZE+fr62nQ7qr/AwEAAQFlRXpXry07nWW1HRETOgUVYE9WtWzebbkf1Fx0djeCQUOj3bIAQ1uN2hDBDv2cjgkPDEB0drVCERERkDxyY30SVD/DONpZWub58udoHgjsDjUaD1S+vQlxcHE4nL4PXbSPRzDcUZafzoN+zEZez07A6KckhcmE5n2QZktEIYTBYPpXnKGrqE47AGfJA6uEMfUKtWIQ1UZ6engCAOTWMMyrfjuwrNjYWSUlJV+cJ+2iWZXlwaBhWq3x6CqDy+STLMqJ8fZCel2s1P5Wazydn6BN+fn4Aas5D+XZE1+MMfULtJCGEUDoIe9Lr9fD29kZxcTG8vLzscgyz2YzCwkL4+flBlh3nCa+zzZjvqHm4lrPNmO/m5sYZ8xsZZ8xXJ0e9PjlDn6jI3rmoS93BIswGHLVzORvmQT2YC3VgHtSDuVAPNRVhPBOIiIiIFMAijIiIiEgBLMKIiIiIFMAijIiIiEgBLMKIiIiIFMAijIiIiEgBLMKIiIiIFMAijIiIiEgBLMKIiIiIFMAijIiIiEgBLMKIiIiIFMAijIiIiEgBLMKIiIiIFMAijIiIiEgBLMKIiIiIFMAijIiIiEgBLMKIiIiIFOCidACOzmQyISUlBUVFRdDpdOjXrx80Go3SYdWJyWRCamoqCgoKEBgYiOjoaIdrAxER2V5WVhZKSkqqXe/p6YnIyMhGjKh+rm2HEAIGgwH5+fmQJAmAcu1QdRFmMpmwePFifPTRRzh58iSCgoIwfvx4zJ8/3/IPp6Tk5GRMnxGP/D+PIyoqCunp6WjdJhirX16F2NhYpcOrlfI2HD+WZ1kWHBLqUG0gIiLby8rKQrt27WrcLjMzU9WFWMV2yLJs+Z1tNpsty5Voh6qLsOXLl+ONN97A+++/j86dO+OXX37Bo48+Cm9vb0ydOlXR2JKTkxEXF4fmET0QMGYKfNuHIaBDLs7s3oC4uDgkJSWpvoixasPYKWimC0VZUR7O7HGcNhARkX2U3zlaHhiICK1rpfXZxlLMKSi47p0yNajUDlmGpPODCA0DzGZF26HqImzXrl0YMWIEhg0bBgAICwvDJ598gr179yoal8lkwvQZ8Wge0QO+sfOhkSRILgKuQe3hGzsfp5OXYXr8TIwYMUK1j/UqtkGSrg4PdG3dwWHaQERE9hehdUUnNzelw2iw8nYIWcZFrRbubm6QrrkTpgRVF2F9+vTBW2+9hczMTLRr1w4HDhzAjz/+iFWrVlX7ntLSUpSWllpe6/V6AIDZbLa67dgQKSkpyP/zOALGTIFGkiBDQIK4+ikHSULL3iORv34OUlJS0L9/f5sc09YqtgEQf690kDZUZDabIYSwWZ6p/pgLdWAe1MMRcyGEgCzLgCxDyFV8jk+WIcuy6ttVsR1CliEk6e822bgdddmHqouwuXPnQq/Xo0OHDtBoNDCZTHjuuecwZsyYat+TkJCAJUuWVFpeVFQEg8Fgk7iKiooQFRUF3/ZhkFyuFl9tPAAJgBkCwiMMraOiUFRUhMLCQpsc09YqtqEiR2hDRWazGcXFxX93OFIMc6EOzIN6OGIuDAYDoqKiIOn8cFGrrbReMhoR5esDg8Gg6t8TFdshJAmlrVsDACQhbN6OujzWVHURtmHDBqxfvx4ff/wxOnfujIyMDEyfPh1BQUEYN25cle+ZN28e4uPjLa/1ej2Cg4Oh0+ng5eVlk7h0Oh3S09MR0CEXrkHtIUNAADh8DjBDQumJXJxMT4dOp4Ofn59NjmlrFdtQkSO0oSKz2QxJkqDT6RzmIuesmAt1YB7UwxFzkZ+fj/T0dIjQMLhX8ThSGAxIz8uFm5ubqn9PVGxH+R0w98xMSGazzdvhVodHt6ouwmbNmoW5c+di9OjRAICbb74ZeXl5SEhIqLYIc3V1hatr5QGE8l+3G22hX79+aN0mGGd2b4Bv7HxAkiBwtQAzCYHzuzeidXAI+vXrp9rOVrEN5WPCAEAIs0O0oSqSJNk011R/zIU6MA/q4Wi5kCTp6qM1s7nqsVN/DfMpb5daVdUOSQhI5a9t3I667EPVRdilS5cqNUaj0Sj+7Fmj0WD1y6sQFxeH08nL0LL3SAiPMJSeyMX53RtxOTsNq5OSVD2gvWIbvG4biWa+oSg7nQf9HsdoAxER2V+2sbROy9XKEq8sQzIaIQwGy6cjlaLqImz48OF47rnnEBISgs6dO2P//v1YtWoVJkyYoHRoiI2NRVJS0tV5wtbPQeuoKJxMT0fr4BCsdpCpHa5tw/GPZlmWB4eGOUwbiIjIPjw9PQEAcwoKarWdWlVshyzLiPL1QXpertVNHSXaIQkhKo/KVomSkhIsWLAAmzdvRmFhIYKCgvDQQw9h4cKF0FYxSLAqer0e3t7eKC4uttmYsGtxxnz1MJvNKCwshJ+fn6pvjTcFzIU6MA/q4ai5cOYZ893c3OwyY35d6g5VF2G2YO8iDHDczuVsmAf1YC7UgXlQD+ZCPeydi7rUHTwTiIiIiBTAIoyIiIhIASzCiIiIiBTAIoyIiIhIASzCiIiIiBTAIoyIiIhIASzCiIiIiBTAIoyIiIhIASzCiIiIiBTAIoyIiIhIASzCiIiIiBTAIoyIiIhIASzCiIiIiBTAIoyIiIhIASzCiIiIiBTAIoyIiIhIASzCiIiIiBTgonQApJysrCyUlJRUu97T0xORkZGNGBE5A6PRiMTERBQXF8Pb2xsTJ06EVqtVOqwm49p+LYSAwWBAfn4+JEkCwH7dmJgL9VBrLupdhGVnZ2P16tU4dOgQAKBTp06YNm0aIiIibBYc2U9WVhbatWtX43aZmZm8SFCtzZ49G6tWr4EwXUFUVBTS09Px1Ow5iJ8+DStWrFA6PKdXsV/LsmzJg9lstixnv7Y/5kI91JyLehVh//nPf3Dvvfeia9eu6Nu3LwDgp59+QufOnfHll1/izjvvtGmQZHvlfxEsDwxEhNa10vpsYynmFBRc904Z0bVmz56NlStXonlET7TqMwq+7cMQ0CEX53Z9hpUrVwIACzE7q9SvZRmSzg8iNAwwm9mvGxFzoR5qzkW9irC5c+dixowZeOGFFyotnzNnDoswBxKhdUUnNzelwyAHZzQasWr1GjSP6AndA/OhkSRILgKuQe2he2ABijYtxarVa7Bs2TI+mmwE5f1ayDIuarVwd3ODdM1f/NR4mAv1UGMu6jUw/9ChQ3jssccqLZ8wYQIOHjzY4KCIyLEkJibCVGaEd+9RkCTry4okyfDuPQqmsqtjxYiI6Kp6FWE6nQ4ZGRmVlmdkZMDPz6+hMRGRg8nOzgYANNOFVrm+mW+o1XZERFTPx5FPPPEE/vnPf+KPP/5Anz59AFwdE7Z8+XLEx8fbNEAiUr/yD+SUFeXBtXWHSuvLTudZbUdERPUswhYsWABPT0+89NJLmDdvHgAgKCgIixcvxtSpU20aIBGp38SJE/HU7Dko3v0ZdA8sAP762DcACGFG8e4N0DTTYuLEiQpGSUSkLvUqwiRJwowZMzBjxgzLpwk8PT1tGhg1jmxjaZ2WE1VFq9Uifvo0rFy5EkWblqJVn1EQHmEoPZGLc7s24HJ2GmbNmsVB+Y3E0n9lGZLRCGEwWD4FRo2LuVAPNeaiwZO1svhyTOV5m1NQUKvtiGpSPv3EqtVrcDInHa2jonAyPR2SxgWzZs3i9BSNoGK/lmUZUb4+SM/LtZoPif3a/pgL9VBzLiQhhKjrm86cOYOFCxdi+/btKCwstGoEAJw9e9ZmATaUXq+Ht7c3iouL4eXlZZdjmM1mFBYWws/PD7LsON8E5Wwz5jtqHpwNZ8xXVlUzg7u5uSk+M7g9mM1mGI1GpcOoVm5uLi5evGh5XVpaClfXv+dldHd3R1hYmAKRNT22zoVWq63290xd6o563Ql75JFHcPToUTz22GPw9/e3dG5yLM5yISZ10Wq1mDp1KgtihVzbr535DxOj0YicnJxKNwHUpkWLFgCuFsSurq6QZdnyO1MIgZycHCXDa1JsmQtZlhEeHt7gPzDrVYSlpqbixx9/RJcuXRp0cCIioroSQqCgoAAajQbBwcEOUWAKIXDlyhW4uLjwxoXCGpoLs9mMEydOoKCgACEhIQ3KZ72KsA4dOuDy5cv1PigREVF9XblyBZcuXUJQUJDl7obasQhTD1vkQqfT4cSJE7hy5QqaNWtW71jq9edDYmIinnnmGezcuRNnzpyBXq+3+iEiIrIXk8kEABxrSIopP/fKz8X6qtedsJYtW0Kv12PQoEFWy4UQkCSpwUERERHVhHeUSCm2OvfqVYSNGTMGzZo1w8cff8yB+URERET1UK8i7Pfff8f+/fvRvn17W8dDRERE17Fu3TpMnz4d58+fVzoU1TMajejUqRM++OADy9cs1uS7777D3LlzsW/fPrt/6KNee+/evTuOHz9u61iIiIic1mOPPWaZFkGSJPj4+GDIkCH49ddflQ6tVoqKivDkk08iJCQErq6uCAgIwODBg/HTTz8pHVq13nzzTYSHh1sVYGfPnsU//vEPeHt7o2XLlnjsscdw4cIFy/ohQ4agWbNmWL9+vd3jq1cRNmXKFEybNg3r1q1Deno6fv31V6sfIiIitTOZTNixYwc++eQT7Nixo1HGMw8ZMgQFBQUoKCjAtm3b4OLignvuucfux7WFBx54APv378f777+PzMxMbNmyBQMGDMCZM2eUDq1KQgi89tpreOyxx6yWjx07FgcPHsT333+Pr776CikpKfjnP/9ptc348ePxyiuvNEqQdSZJUqUfWZYt/1WT4uJiAUAUFxfb7Rgmk0kUFBQIk8lkt2NQzZgH9WAu1MFZ83D58mVx8OBBcfny5XrvY9OmTSI4JFQAsPwEh4SKTZs22TDSv5nNZvHII4+IESNGWC1PTU0VAERhYaEQQojt27cLAOLcuXOWbfbv3y8AiJycHCGEEO+9957w9va2rC8sLBRRUVHivvvuEwaDQRw9elTce++9ws/PT7i7u4vu3buLrVu3Wh03NDRUPPfcc+LRRx8VHh4eIjg4WKxdu7ba+M+dOycAiB07dly3nQBEYmKiGDJkiHBzcxPh4eFi48aNVtvMnj1bREZGiubNm4vw8HAxf/58YTQaLesXLVokunTpIt555x0RHBws3N3dxZNPPimuXLkili9fLvz9/YVOpxPLli27bixpaWlClmWh1+styw4ePCgAiN27dwuz2SyEEOLbb78VkiSJ/Px8y3Z5eXkCgDh69GiV+77eOViXuqNed8JycnIq/fzxxx+W/xIREalVcnIy4uLicEbrj4CxLyJ4xkYEjH0RZ7T+iIuLQ3JycqPEceHCBXz00Udo27YtfHx86rWP48ePIzo6GjfddBOSkpLg6uqKCxcu4O6778a2bduwf/9+DBkyBMOHD8exY8es3vvSSy+he/fu2L9/PyZOnIgnn3wSR44cqfI4Hh4e8PDwwOeff47S0ut/4fWCBQvwwAMP4MCBAxgzZgxGjx6NQ4cOWdZ7enpi3bp1OHjwINasWYN///vfePnll632kZ2djW+//RbfffcdPvnkE7zzzjsYNmwY/vzzT+zcuRPLly/H/Pnz8fPPP1cbR2pqKtq1a2f1nZC7d+9Gy5YtERUVZVkWExMDWZat9hUSEgJ/f3+kpqZet60NVmOZ5uB4J6zpYB7Ug7lQB2fNQ0PuhF25ckUEh4SKFm17ipDZW0TonK8sPyGzt4gWbXuK4NAwceXKFZvGXH4nTKPRCHd3d+Hu7i4AiMDAQJGenm7Zri53wg4fPiyCg4PF1KlTLXd1qtO5c2fx6quvWl6HhoaKsWPHWsXn5+cn3njjjWr3kZSUJFq1aiXc3NxEnz59xLx588SBAwestgEg/vWvf1kt69Wrl3jyySer3e/KlStFVFSU5fWiRYtEixYtrO5gDR48WISFhVmdy+3btxcJCQnV7nfatGli0KBBVsuee+450a5dO2E0Gq3+zXQ6nUhMTLTatlu3bmLx4sVV7lvRO2HA1Sp1ypQpiImJQUxMDKZOnYrs7GzbVIZERER2kJqaiuPH8uB12yhIkvWvQEmS4XXbSBzPy7XbHZCBAwciIyMDGRkZ2Lt3LwYPHoyhQ4ciLy+vTvu5fPkyoqOjERsbizVr1lhNFXXhwgU89dRT6NixI1q2bAkPDw8cOnSo0p2wW265xfL/kiQhICAAhYWF1R7zgQcewIkTJ7BlyxYMGTIEO3bswK233op169ZZbde7d+9Kr6+9E/bZZ5+hb9++CAgIgIeHB+bPn18ptrCwMKs7WP7+/ujUqZPVpxX9/f2vG+/ly5fh5uZW7fqaNG/eHJcuXar3+2ujXkXYf/7zH3Tq1Al79+7FLbfcgltuuQU///wzOnfujK1bt9o6RiIiIpsoKCgAADTThVa5vplvqNV2tubu7o62bduibdu26NGjB95++21cvHgR//73vwHAUmQIISzvKSsrq7QfV1dXxMTE4KuvvkJ+fr7VuqeeegqbN2/G888/j9TUVGRkZODmm2+G0Wi02q7i1+1IklTjF6K7ubnhzjvvxIIFC7Br1y6MHz8eixYtqnX7d+/ejTFjxuDuu+/GV199hf379+OZZ56pVWx1jdfX1xfnzp2zWlZVoXnlyhWcPXsWAQEBVsvPnj0LnU5X67bVR72KsLlz52LGjBn4+eefsWrVKqxatQo///wzpk+fjjlz5tg6RiIiIpsIDAwEAJQVVX3nqex0ntV29iZJEmRZtnwfc/kv/WuLwIyMjErvk2UZH374IaKiojBw4ECcOHHCsu6nn37C+PHjcf/99+Pmm29GQEAAcnNz7RJ/p06dcPHiRatle/bsqfS6Y8eOAIBdu3YhNDQUzzzzDLp3747IyMg63wWsrW7duuHw4cNWBW3v3r1x/vx57Nu3z7Lshx9+gNlsRq9evSzLDAYDsrOz0a1bN7vEVq5eRdihQ4cqfeQTACZMmICDBw82OCgiIiJ7iI6ORnBIKPR7NkAI67soQpih37MRwaFhiI6OtsvxS0tLcfLkSZw8eRKHDh3ClClTcOHCBQwfPhwA0LZtWwQHB2Px4sXIysrC119/jZdeeqnKfWk0Gqxfvx5dunTBoEGDcPLkSQBAZGQkkpOTkZGRgQMHDuDhhx+u8Q5XTc6cOYNBgwbho48+wq+//oqcnBxs3LgRK1aswIgRI6y23bhxI959911kZmZi0aJF2Lt3LyZPnmyJ7dixY/j000+RnZ2NV155BZs3b25QbNUZOHAgLly4gP/973+WZR07dsSQIUPwr3/9C3v37sVPP/2EyZMnY/To0QgKCrJst2fPHri6ulZ6tGpr9SrCdDpdlZV5RkYG/Pz8GhoTERGRXWg0Gqx+eRUuZ6fhdPIylOYfgrn0EkrzD+F08jJczk7D6lUvQaPR2OX43333HQIDAxEYGIhevXohLS0NGzduxIABAwBcfQz3ySef4PDhw7jllluwfPlyLFu2rNr9ubi44JNPPkHnzp0xaNAgFBYWYtWqVWjVqhX69OmD4cOHY/Dgwbj11lsbFLeHhwd69eqFl19+Gf369cNNN92EBQsW4IknnsBrr71mte2SJUvw6aef4pZbbsEHH3yATz75BJ06dQIA3HvvvZgxYwYmT56Mrl27YteuXViwYEGDYquOj48P7r///kqTrn700Udo3749YmJicPfdd+P222/HW2+9ZbXNJ598gjFjxqBFixZ2ic2ixqH7VViyZIlo2bKleOGFF0RKSopISUkRCQkJomXLluLZZ5+tzy7thp+ObDqYB/VgLtTBWfNgt3nCQsPsOk9YxU/kOSMAYvPmzUqHYXHgwAHh5+cnSkpKLMtqykVRUZG44YYbxB9//FHtfm316ch6fXfkggUL4OnpiZdeegnz5s0DAAQFBWHx4sWYOnWqjcpDx2AymZCSkoKioiLodDr069fPbn9BkfMzmUxITU1FQUEBAgMDER0d7XDnE/uEsrKyslBSUgKTyYT9+/fDYDDAzc0N3bp1g0ajgaenJyIjI5UOU3GxsbEYMWKEw/c3ur7yu4k5OTm4+eaba/We3NxcJCYmIjw83M7Roe53wsrKysT7778vTp48KYQQQq/XW83lYWt//vmnGDNmjLjhhhuEm5ubuOmmm0RaWlqt32/PO2Hlf0nJsix69OghZFm264zLdH2O/ld/Y8/gbQ/sE8rKzMy0On+uzcO1yzMzM5UOtUFscSessfFOmHrYIheK3QlzcXHBv/71L8ucH9fO42Fr586dQ9++fTFw4EB8++230Ol0yMrKQqtWrex2zNoqn3G5eUQPBIyZAt/2YQjokIszuzcgLi4OSUlJiI2NVTpMchBW59PYKWimC0VZUR7O7HGc84l9QnklJSUAgOWBgYjQugKyDEnnBxEaBpjNyDaWYk5BgWU7IlsT13wSkWomiXr8iw0YMADTp0/HfffdZ4eQ/jZ37lz89NNPDZo0T6/Xw9vbG8XFxfDy8rJJXCaTCeE3RuCM1h++sfOhkSR0bCVw6JwEkxA4nbwMPmWFyMk+ylvbjchsNqOwsBB+fn5WE/qpXcXz6doJJIUwO8T5xD6hDmlpaejZsyeSQsPQyc0NQpZxsX17uB85AslsxkGDAXF5udi7dy969OihdLj1ZjAYkJOTg/Dw8AZNxtmYhBC4cuUKXFxcrCZWpcZni1xc7xysS91RrzFhEydOxMyZM/Hnn38iKioK7u7uVuuvnYW3IbZs2YLBgwdj5MiR2LlzJ1q3bo2JEyfiiSeeqPY9paWlVt9rpdfrAVz9Bd3Qj+iWS0lJQf6fxxEwZgo0kgQZAhLE1Y+aShJa9h6J/PVzkJKSgv79+9vkmFQzs9kMIYTN8txYKp5PV58Y/cVBzif2CXXYv3//1T9AZBmi/EeSIMr/KJFlyLKM/fv3W313nqMp7+vlP46iPFZHitlZNTQX5edeVbVFXX4H1asIGz16NABYDcKXJAlCCEiSBJPJVJ/dVvLHH3/gjTfeQHx8PJ5++mmkpaVh6tSp0Gq1GDduXJXvSUhIwJIlSyotLyoqgsFgsElcRUVFiIqKgm/7MEguV3/RtPEAJABmCAiPMLSOikJRUdF1v1KBbMtsNqO4uBhCCIe6E1bxfKrIEc4n9gl1MBgMiIqKgqTzw0WtFkKSUNq6NQBAEgKS0YgoXx8YDAaHzkNZWRnMZjOuXLmCK1euKB1OrQghLL8beSdMWbbIxZUrV2A2m3HmzJlKM/nX5XF/vYqwnJyc+rytzsxmM7p3747nn38ewNXZb3///Xe8+eab1RZh8+bNQ3x8vOW1Xq9HcHAwdDqdzR5H6nQ6pKenI6BDLlyD2kOGgABw+BxghoTSE7k4mZ4OnU7HedMakdlshiRJ0Ol0DlWEVTyfKnKE84l9Qh3c3NyQnp4OERoG978eRwKAe2YmJLMZwmBAel4uHn/8cYfOg8FgQElJCVxcXODiUq9fY4qp+AublNOQXLi4uECWZfj4+FR6HFmXR+R1Pnv1ej0yMzNhNBrRs2dPu36vUmBgoGWCt3IdO3bEpk2bqn2Pq6srXF1dKy2X/7oNbwv9+vVD6zbBOLN7A3xj5wOSBIGrv2xMQuD87o1oHRyCfv36OVQx4AzKvwLEkf7dK55PFceEOcL5xD6hDt26dbv6KMRshvTXIxFJCEjlr/96dNKtWzeHzoMsy5AkyfLjCMqfFAG8E6Y0W+Si/Nyr6vdNXfpWnXphRkYGOnTogCFDhmD48OFo27Yt/vOf/9RlF3XSt29fHDlyxGpZZmYmQkOr/uLVxlJpxuUThyHKSlF64nCjzLhMzkXpGbxtgX1CHcr/fbONpThoMOCQwYDjRiMOGQw4aDAg21hqtR0RKawu82Lcddddok+fPmLXrl1i37594v777xdt27atyy7qZO/evcLFxUU899xzIisrS6xfv160aNFCfPTRR7XeR6PPE2bHGZfp+pxynjAHO5/YJ5TFecLUy1HnCSstLRURERHip59+qvV7/ve//4nWrVuLCxcu2DGy+lPTPGF1mqLC19cX33//veU7qM6fP48bbrgB58+ft9l4q4q++uorzJs3D1lZWQgPD0d8fPx1Px1ZkT2mqLgWZwdXD0edouJanDGfGqopzJjvqFNUFBQU4Nlnn8U333yDU6dOoVWrVujSpQsWLlyIvn37Kh1ilV555RV8+eWX2Lp1q2XZc889h6+//hoZGRnQarU4f/58pffFxcWhS5cudvteyIYQKpqiok5FmCzLOHnypNWATk9PT/z666+NM71/Pdi7CAOc45e/M2Ae1IO5UAdnzUNDi7DyQrU69ihUhRDo168fysrKkJCQgBtvvBGnTp3Ctm3b0LlzZ9x77702PZ4tCCHQvn17PPvss5ZZEQBg0aJFaNmyJf7880+88847VRZhX3/9NZ544gkcO3ZMdR+eUFMRVqfHkZIkie3bt4sDBw5Yftzd3cXXX39ttUxN+AXeTQfzoB7MhTo4ax4a8jiy4iPb6n5s/cj27NmzAoDYvn37dbcDIBITE8WQIUOEm5ubCA8PFxs3brTaZvbs2SIyMlI0b95chIeHi/nz5wuj0WhZv2jRItGlSxfxzjvviODgYOHu7i6efPJJceXKFbF8+XLh7+8vdDqdWLZs2XVjSUtLE7IsV/vVhO+9957w9vaucl1paalwdXUV//3vf697DCWo6XFkncvTO+64o9LkZvfcc49d5gkjIiKypUpf7VSBvb7aycPDAx4eHvj888/Ru3fvKj/FX27BggV44YUXsGbNGnz44YcYPXo0fvvtN3Ts2BHA1Tt169atQ1BQEH777Tc88cQT8PT0xOzZs/9uR3Y2vv32W3z33XfIzs5GXFwc/vjjD7Rr1w47d+7Erl27MGHCBMTExKBXr15VxpGamop27drV6+sJtVotunbtitTUVNxxxx11fn9TUacirLHmByMiIrKnCK0rOjXieDIXFxe8/fbbePLJJ7F27Vrceuut6N+/P0aPHl3pW2ZGjhyJxx9/HACwdOlSbN26Fa+++ioSExMBAPPnz7dsGxYWhqeeegqffvqpVRFmNpvx7rvvwtPTE506dcLAgQNx5MgRfPPNN5BlGe3bt8fy5cuxffv2aouwvLw8BAUF1bvNQUFByMvLq/f7m4I6FWFKTw1BRETkqGJjY3Hvvffixx9/xJ49e/Dtt99ixYoVePvttzF+/HjLdr1797Z6X+/evZGRkWF5/dlnn+GVV15BdnY2Lly4gCtXrlQaexQWFmZ1B8vf3x8ajcZqbKC/v/91vznh8uXLDfrgQ/PmzXHp0qV6v78paNBIzUuXLuHw4cP49ddfrX6IiIioMjc3N9x5551YsGABdu3ahfHjx2PRokW1fv/u3bsxZswY3H333fjqq6+wf/9+PPPMMzAajVbbVZwNXpKkKpdd73sOfX19ce7cuVrHVtHZs2ftOqG7M6hXEVZUVIR77rkHnp6e6Ny5M7p162b1Q0RERDXr1KkTLl68aLVsz549lV6XjwfbtWsXQkND8cwzz6B79+6IjIy02yO/bt264fDhw/X+kuvff/+dNUEN6lWETZ8+HefPn8fPP/+M5s2b47vvvsP777+PyMhIbNmyxdYxEhERObQzZ87grrvuwkcffYRff/0VOTk52LhxI1asWIERI0ZYbbtx40a8++67yMzMxKJFi7B3715MnjwZABAZGYljx47h008/RXZ2Nl555RVs3rzZLjEPHDgQFy5cwP/+9z+r5ceOHUNGRgaOHTsGk8mEjIwMZGRk4MKFC5ZtcnNzkZ+fj5iYGLvE5izqNXnHDz/8gC+++ALdu3eHLMsIDQ3FnXfeCS8vLyQkJGDYsGG2jpOIiMhmyr/CqbbLG8rDwwM9evTA6tWrkZ2djbKyMgQHB+OJJ57A008/bbXtkiVL8Omnn2LixIkIDAzEJ598Yvke5XvvvRczZszA5MmTUVpaimHDhmHBggVYvHixzWP28fHB/fffj/Xr1yMhIcGyfOHChXj//fctr8vvdm3fvh0DBgwAAHzyySe46667OJa8BnWarLWcl5cXfv31V4SFhSE0NBQff/wx+vbti5ycHHTu3FlVA/E4WWvTwTyoB3OhDs6ah4ZM1pqVlYV27drVuF1mZqZNJ2wVtZwgVJIkbN68Gffdd5/Njt0Qv/76K+68805kZ2fDw8OjVu8xGo2IjIy01AZqU9tcXI+tJmut152w9u3b48iRIwgLC0OXLl2wdu1ahIWF4c0330RgYGB9dklERGR3kZGRyMzMbPQZ8x3VLbfcguXLlyMnJwc333xzrd5z7NgxPP3006oswNSmXkXYtGnTUFBQAODq1xcMGTIE69evh1arxbp162wZHxERkU2xwKqba6fPqI22bduibdu29gnGydSrCBs7dqzl/6OiopCXl4fDhw8jJCQEvr6+NguOiIioKanvJxHJMdVrkMCzzz5rNe6rRYsWuPXWW+Hu7o5nn33WZsEREREROat6FWFLliyx+ihquUuXLmHJkiUNDoqIiKgmvGtESrHVuVevIqz8i7orOnDgAG644YYGB0VERFQdjUYDAJVmiSdqLOXnXvm5WF91GhPWqlUrSJIESZLQrl07q0LMZDLhwoUL+Ne//tWggIiIiK7HxcUFLVq0QFFREZo1a+YQ02/YYloEso2G5sJsNqOoqAgtWrSAi0u9htZb1Ondq1evhhACEyZMwJIlS+Dt7W1Zp9VqERYWVumLR4mIiGxJkiQEBgYiJyfHbl/ZY2tCCJjNZsiyzCJMYbbIhSzLCAkJaXAu61SEjRs3DgAQHh6OPn36VPoy0KbIZDIhJSUFRUVF0Ol06NevX4NvT1LdMQ/qwVyog7PnQavVIjIy0mEeSZrNZpw5cwY+Pj4OceeuXG5ubqXvtryWu7s7wsLCGi8gG7BFLrRarW3yKBro8uXLori42OpHTYqLiwUAu8S1adMmERwSKmRZFj169BCyLIvgkFCxadMmmx+Lqsc8qAdzoQ7Mg/qYTCZRUFAgTCaT0qHUWmZmpgBQ409mZqbSodaJvXNRl7qjXg8zL126hNmzZ2PDhg04c+ZMpfUmk6l+FaEDSU5ORlxcHJpH9EDAmCnwbR+GgA65OLN7A+Li4pCUlITY2Filw3R6zIN6MBfqwDyQrZR/q8DywEBEaF0rrc82lmJOQcF1v32Arq9e3x05adIkbN++HUuXLsUjjzyC119/Hfn5+Vi7di1eeOEFjBkzxh6x1os9vjvSZDIh/MYInNH6wzd2PjSShI6tBA6dk2ASAqeTl8GnrBA52Ued6va/2jAP6sFcqAPzoF6O+D2e+/btQ1RUFJJCw9Cpiu/oPGgwIC4vF+np6bj11lsViLB+7J2LutQd9Tr6l19+icTERDzwwANwcXFBdHQ05s+fj+effx7r16+vV9COJDU1FceP5cHrtlGQJOt/QkmS4XXbSBzPy0VqaqpCETYNzIN6MBfqwDwQOZZ6FWFnz57FjTfeCADw8vLC2bNnAQC33347UlJSbBedSpV/b2YzXWiV65v5hlptR/bBPKgHc6EOzAORY6lXEXbjjTciJycHANChQwds2LABwNU7ZNdOW+GsAgMDAQBlRVV/NLrsdJ7VdmQfzIN6MBfqwDwQOZZ6FWGPPvooDhw4AACYO3cuXn/9dbi5uWH69OlNYrLW6OhoBIeEQr9nA4QwW60Twgz9no0IDg1DdHS0QhE2DcyDejAX6sA8EDmWOhVhL7/8MgBgxowZmDp1KgAgJiYGhw8fxscff4yMjAx89913to9SZTQaDVa/vAqXs9NwOnkZSk8chigrRemJwzidvAyXs9OwetVLHPhqZ8yDejAX6sA8kD1kG0tx0GCo9JNtLFU6NMdXl7kv3NzcxPvvv1/lupKSEtG3b1/Rvn37uuzS7hp9nrDQMM7F08iYB/VgLtSBeVAfzhOmHmqaJ6xOU1QkJSXhkUcewWeffYZ7773XsvzChQsYOnQoTp06hR07diAoKMh2VWID2WOKims5+6zUjoJ5UA/mQh2YB3VxxCkqACArK+u684B5enoiMjKyESNqODVNUVHnecLefvttTJs2DV9//TUGDBiAixcvYsiQITh58iR27NiB1q1bNyh4W7N3EQY4budyNsyDejAX6sA8qAdzoR5qKsLqPGP+448/jrNnz2LEiBH44osvsHDhQpw4cQI7d+5UXQFGREREpFb1+tqi2bNn4+zZs7jjjjsQFhaGHTt2oE2bNraOjYiIiMhp1akIq/h9Y82aNYOvry+mTZtmtTw5ObnhkRERERE5sToVYRUnYn3ooYdsGgwRERFRU1GnIuy9996zVxxERERETQo/okFERESkABZhRERERApgEUZERESkABZhRERERApgEUZERESkABZhRERERApgEUZERESkABZhRERERApgEUZERESkABZhRERERApgEUZERESkABZhRERERApgEUZERESkABelAyDlmUwmpKamoqCgAIGBgYiOjoZGo1E6rDoxGo1ITExEcXExvL29MXHiRGi1WqXDapJMJhNSUlJQVFQEnU6Hfv36Odz5ROrhyNenrKwslJSUAACEEDAYDMjPz4ckSQAAT09PREZGKhkiKU04kISEBAFATJs2rdbvKS4uFgBEcXGx3eIymUyioKBAmEwmux3DXjZt2iSCQ0IFAMtPcEio2LRpk9Kh1dqsWbOEpplWyLIsevToIWRZFppmWjFr1iylQ2tyys+na3PhaOeTM3Hka5MQjn19yszMtIr72j5x7fLMzEylQ21y7N0v6lJ3OMydsLS0NKxduxa33HKL0qE4jeTkZMTFxaF5RA8EjJ2CZrpQlBXl4cyeDYiLi0NSUhJiY2OVDvO6Zs+ejZUrV6J5RE+06jMKvu3DENAhF+d2fYaVK1cCAFasWKFwlE2D1fk0ZoolF2d2O875ROrh6Nen8jtgywMDEaF1BWQZks4PIjQMMJuRbSzFnIICy3bUNElCCKF0EDW5cOECbr31ViQmJmLZsmXo2rUrVq9eXav36vV6eHt7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",
      "text/plain": [
       "<Figure size 700x500 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Jawaban B1\n",
    "df_email = pd.read_csv(\"Email_Data.csv\")\n",
    "print(\"5 data pertama:\")\n",
    "display(df_email.head())\n",
    "print(\"Jumlah Spam (1)       :\", (df_email[\"Spam\"] == 1).sum())\n",
    "print(\"Jumlah Bukan Spam (0) :\", (df_email[\"Spam\"] == 0).sum())\n",
    "\n",
    "X_em = df_email[[\"JumlahLink\", \"KataPromo\"]].values\n",
    "y_em = df_email[\"Spam\"].values\n",
    "print(\"Shape X:\", X_em.shape, \"| Shape y:\", y_em.shape)\n",
    "\n",
    "plt.figure(figsize=(7, 5))\n",
    "plt.scatter(X_em[y_em == 0, 0], X_em[y_em == 0, 1], c=\"tab:blue\", edgecolor=\"k\", label=\"Bukan Spam (0)\")\n",
    "plt.scatter(X_em[y_em == 1, 0], X_em[y_em == 1, 1], c=\"tab:red\", marker=\"s\", edgecolor=\"k\", label=\"Spam (1)\")\n",
    "plt.title(\"Jumlah Link vs Kata Promo\")\n",
    "plt.xlabel(\"JumlahLink\")\n",
    "plt.ylabel(\"KataPromo\")\n",
    "plt.legend()\n",
    "plt.grid(alpha=0.3)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cell-25",
   "metadata": {},
   "source": [
    "### Soal B2 – Split Data & Training\n",
    "1. Bagi data dengan `test_size=0.25` dan `random_state=RANDOM_STATE`.\n",
    "2. Latih model `LogisticRegression` (gradient descent, dari materi) dengan `learning_rate=0.1` dan `n_iterations=10000`.\n",
    "3. Tampilkan intercept ($b$) dan koefisien ($m_1, m_2$), lalu tuliskan persamaannya:\n",
    "   $$P(\\text{Spam}) = \\sigma(b + m_1 \\cdot \\text{JumlahLink} + m_2 \\cdot \\text{KataPromo})$$\n",
    "4. Buat grafik **log loss vs iterasi**. Apakah loss sudah turun dan stabil?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "cell-26",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-10-20T11:35:54.593395Z",
     "iopub.status.busy": "2026-10-20T11:35:54.593096Z",
     "iopub.status.idle": "2026-10-20T11:35:55.044475Z",
     "shell.execute_reply": "2026-10-20T11:35:55.043440Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train: (45, 2) | Test: (15, 2)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Intercept (b)       : -10.1635\n",
      "Koefisien m1 (Link) : 0.8950\n",
      "Koefisien m2 (Promo): 0.8434\n",
      "P(Spam) = sigmoid(-10.1635 + 0.8950*JumlahLink + 0.8434*KataPromo)\n",
      "Log loss awal: 0.6931 | akhir: 0.2844\n"
     ]
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 700x400 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Jawaban B2\n",
    "Xe_train, Xe_test, ye_train, ye_test = train_test_split(\n",
    "    X_em, y_em, test_size=0.25, random_state=RANDOM_STATE)\n",
    "print(\"Train:\", Xe_train.shape, \"| Test:\", Xe_test.shape)\n",
    "\n",
    "def sigmoid(z):\n",
    "    return 1 / (1 + np.exp(-z))\n",
    "\n",
    "class LogisticRegression:\n",
    "    \"\"\"Logistic Regression dengan Gradient Descent (log loss).\"\"\"\n",
    "    def __init__(self, learning_rate=0.1, n_iterations=1000):\n",
    "        self.lr = learning_rate\n",
    "        self.n_iter = n_iterations\n",
    "\n",
    "    def fit(self, X, y):\n",
    "        n = X.shape[0]\n",
    "        self.intercept_ = 0.0\n",
    "        self.coef_ = np.zeros(X.shape[1])\n",
    "        self.loss_history_ = []\n",
    "        eps = 1e-15\n",
    "        for _ in range(self.n_iter):\n",
    "            p = sigmoid(self.intercept_ + X @ self.coef_)\n",
    "            p_clip = np.clip(p, eps, 1 - eps)\n",
    "            self.loss_history_.append(-np.mean(y * np.log(p_clip) + (1 - y) * np.log(1 - p_clip)))\n",
    "            # gradien log loss\n",
    "            error = p - y\n",
    "            self.intercept_ -= self.lr * np.mean(error)\n",
    "            self.coef_      -= self.lr * (X.T @ error) / n\n",
    "        return self\n",
    "\n",
    "    def predict_proba(self, X):\n",
    "        return sigmoid(self.intercept_ + X @ self.coef_)\n",
    "\n",
    "    def predict(self, X, threshold=0.5):\n",
    "        return (self.predict_proba(X) >= threshold).astype(int)\n",
    "\n",
    "logreg = LogisticRegression(learning_rate=0.1, n_iterations=10000).fit(Xe_train, ye_train)\n",
    "b_lg = logreg.intercept_\n",
    "m1_lg, m2_lg = logreg.coef_\n",
    "print(f\"Intercept (b)       : {b_lg:.4f}\")\n",
    "print(f\"Koefisien m1 (Link) : {m1_lg:.4f}\")\n",
    "print(f\"Koefisien m2 (Promo): {m2_lg:.4f}\")\n",
    "print(f\"P(Spam) = sigmoid({b_lg:.4f} + {m1_lg:.4f}*JumlahLink + {m2_lg:.4f}*KataPromo)\")\n",
    "print(f\"Log loss awal: {logreg.loss_history_[0]:.4f} | akhir: {logreg.loss_history_[-1]:.4f}\")\n",
    "\n",
    "plt.figure(figsize=(7, 4))\n",
    "plt.plot(logreg.loss_history_)\n",
    "plt.title(\"Log Loss vs Iterasi\")\n",
    "plt.xlabel(\"Iterasi\")\n",
    "plt.ylabel(\"Log loss\")\n",
    "plt.grid(alpha=0.3)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cell-27",
   "metadata": {},
   "source": [
    "**Jawaban B2 (analisis):**\n",
    "\n",
    "Hasil training (lr = 0,1; 10.000 iterasi):\n",
    "\n",
    "**P(Spam) = σ(−10,1635 + 0,8950 · JumlahLink + 0,8434 · KataPromo)**\n",
    "\n",
    "Kedua koefisien positif: semakin banyak link dan kata promosi, semakin besar peluang email dianggap spam. Setiap tambahan 1 link menaikkan *odds* spam sekitar e^0,8950 ≈ 2,45 kali, dan setiap tambahan 1 kata promosi sekitar e^0,8434 ≈ 2,32 kali. Pengaruh keduanya hampir sama kuat, dengan jumlah link sedikit lebih besar.\n",
    "\n",
    "**Grafik log loss:** loss turun tajam dari 0,6931 (= ln 2, saat semua probabilitas 0,5) menjadi 0,3821 pada iterasi ke-500 dan 0,3045 pada iterasi ke-2000, lalu melandai sampai 0,2844 di akhir (0,2877 pada iterasi ke-5000). Jadi loss **sudah turun dan praktis stabil**, hanya masih menurun sangat pelan pada ujung iterasi."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cell-28",
   "metadata": {},
   "source": [
    "### Soal B3 – Evaluasi Model\n",
    "Selain `accuracy_score` dan `confusion_matrix` dari materi, **implementasikan sendiri** fungsi berikut:\n",
    "\n",
    "$$\n",
    "Precision = \\frac{TP}{TP + FP} \\qquad\n",
    "Recall = \\frac{TP}{TP + FN} \\qquad\n",
    "F1 = 2 \\cdot \\frac{Precision \\cdot Recall}{Precision + Recall}\n",
    "$$\n",
    "\n",
    "1. Hitung accuracy, confusion matrix, precision, recall, dan F1-score pada **data testing**.\n",
    "2. Jelaskan arti nilai **FP** (*false positive*) dan **FN** (*false negative*) pada kasus filter spam. Kesalahan mana yang menurut Anda **lebih merugikan pengguna email**? Mengapa?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "cell-29",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-10-20T11:35:55.046704Z",
     "iopub.status.busy": "2026-10-20T11:35:55.046067Z",
     "iopub.status.idle": "2026-10-20T11:35:55.062283Z",
     "shell.execute_reply": "2026-10-20T11:35:55.061195Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy : 0.8667\n",
      "Precision: 0.8333\n",
      "Recall   : 0.8333\n",
      "F1-score : 0.8333\n",
      "\n",
      "Confusion matrix (baris=aktual, kolom=prediksi):\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Prediksi 0</th>\n",
       "      <th>Prediksi 1</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Aktual 0 (Bukan Spam)</th>\n",
       "      <td>8</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Aktual 1 (Spam)</th>\n",
       "      <td>1</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                       Prediksi 0  Prediksi 1\n",
       "Aktual 0 (Bukan Spam)           8           1\n",
       "Aktual 1 (Spam)                 1           5"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "TN=8  FP=1  FN=1  TP=5\n"
     ]
    }
   ],
   "source": [
    "# Jawaban B3\n",
    "def accuracy_score(y_true, y_pred):\n",
    "    return np.mean(y_true == y_pred)\n",
    "\n",
    "def confusion_matrix(y_true, y_pred):\n",
    "    tn = np.sum((y_true == 0) & (y_pred == 0))\n",
    "    fp = np.sum((y_true == 0) & (y_pred == 1))\n",
    "    fn = np.sum((y_true == 1) & (y_pred == 0))\n",
    "    tp = np.sum((y_true == 1) & (y_pred == 1))\n",
    "    return np.array([[tn, fp], [fn, tp]])\n",
    "\n",
    "def precision_score(y_true, y_pred):\n",
    "    cm = confusion_matrix(y_true, y_pred)\n",
    "    tp, fp = cm[1, 1], cm[0, 1]\n",
    "    return tp / (tp + fp) if (tp + fp) > 0 else 0.0\n",
    "\n",
    "def recall_score(y_true, y_pred):\n",
    "    cm = confusion_matrix(y_true, y_pred)\n",
    "    tp, fn = cm[1, 1], cm[1, 0]\n",
    "    return tp / (tp + fn) if (tp + fn) > 0 else 0.0\n",
    "\n",
    "def f1_score(y_true, y_pred):\n",
    "    p, r = precision_score(y_true, y_pred), recall_score(y_true, y_pred)\n",
    "    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0\n",
    "\n",
    "ye_pred = logreg.predict(Xe_test)\n",
    "cm = confusion_matrix(ye_test, ye_pred)\n",
    "print(\"Accuracy :\", round(accuracy_score(ye_test, ye_pred), 4))\n",
    "print(\"Precision:\", round(precision_score(ye_test, ye_pred), 4))\n",
    "print(\"Recall   :\", round(recall_score(ye_test, ye_pred), 4))\n",
    "print(\"F1-score :\", round(f1_score(ye_test, ye_pred), 4))\n",
    "print(\"\\nConfusion matrix (baris=aktual, kolom=prediksi):\")\n",
    "display(pd.DataFrame(cm, index=[\"Aktual 0 (Bukan Spam)\", \"Aktual 1 (Spam)\"],\n",
    "                     columns=[\"Prediksi 0\", \"Prediksi 1\"]))\n",
    "print(f\"TN={cm[0,0]}  FP={cm[0,1]}  FN={cm[1,0]}  TP={cm[1,1]}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cell-30",
   "metadata": {},
   "source": [
    "**Jawaban B3 (analisis):**\n",
    "\n",
    "Hasil pada data testing (15 email): **accuracy = 0,8667**, **precision = 0,8333**, **recall = 0,8333**, **F1 = 0,8333**.\n",
    "\n",
    "| | Prediksi Bukan Spam | Prediksi Spam |\n",
    "|---|---|---|\n",
    "| **Aktual Bukan Spam** | TN = 8 | FP = 1 |\n",
    "| **Aktual Spam** | FN = 1 | TP = 5 |\n",
    "\n",
    "- **FP (false positive):** email yang sebenarnya *bukan spam* tetapi dianggap spam. Pada data ini ada 1 email.\n",
    "- **FN (false negative):** email yang sebenarnya *spam* tetapi lolos ke inbox. Pada data ini ada 1 email.\n",
    "\n",
    "Menurut saya **FP lebih merugikan**. Email penting yang salah masuk folder spam, misalnya tagihan, undangan wawancara, atau informasi akademik, bisa terlewat dan menimbulkan kerugian nyata. FN hanya membuat pengguna menerima satu email gangguan yang bisa dihapus manual. Karena itu, untuk filter spam, **precision** perlu mendapat perhatian lebih dibanding recall."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cell-31",
   "metadata": {},
   "source": [
    "### Soal B4 – Decision Boundary\n",
    "1. Hitung persamaan **decision boundary** (garis saat probabilitas = 0.5):\n",
    "   $$b + m_1 x_1 + m_2 x_2 = 0 \\;\\Rightarrow\\; x_2 = -\\frac{b + m_1 x_1}{m_2}$$\n",
    "2. Buat visualisasi data **testing** beserta *background* probabilitas (`contourf`) dan garis decision boundary, seperti di materi.\n",
    "3. Tunjukkan (beri tanda/penjelasan) data testing mana saja yang **salah diklasifikasikan**."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "cell-32",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-10-20T11:35:55.064181Z",
     "iopub.status.busy": "2026-10-20T11:35:55.063996Z",
     "iopub.status.idle": "2026-10-20T11:35:55.456311Z",
     "shell.execute_reply": "2026-10-20T11:35:55.452578Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Decision boundary: x2 = -(-10.1635 + 0.8950 * x1) / 0.8434\n",
      "                   x2 = 12.0513 + (-1.0612) * x1\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAqUAAAJpCAYAAACU8kv9AAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjgsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvwVt1zgAAAAlwSFlzAAAPYQAAD2EBqD+naQAA87ZJREFUeJzs3Xd4k1Ubx/FvuhctLYXSIlsQCmXLFFlF9lKmIFNERQFxIL4KgiiigoiigAIisoeIqAzZCDJlCcjeUEZpS+lun/eP0EjIaJKmzdPk/lxXrsumJydPaws/znPuc2sURVEQQgghhBDCgdwcfQFCCCGEEEJIKBVCCCGEEA4noVQIIYQQQjichFIhhBBCCOFwEkqFEEIIIYTDSSgVQgghhBAOJ6FUCCGEEEI4nIRSIYQQQgjhcBJKhRBCCCGEw0koFU7p/PnzaDQavv/+e6te17RpU5o2bZon1+TsypQpQ//+/R19GQ7Rv39/ypQp4+jLcHmu/DMohDOQUCryxPfff49Go9E9fHx8iIiIoFWrVkybNo27d+86+hJVZcuWLXrfL41GQ0hICPXr12fBggWOvjyRz/r376/3sxAQEEC5cuXo2rUrK1asICsry+a5Fy5cyNSpU+12rQ//rpt62Cu079y5k/fff5+4uDi7zCeEUA8PR1+AcG7jx4+nbNmypKenc/36dbZs2cKIESOYMmUKq1evplq1annyvqVLlyY5ORlPT0+rXrd+/fo8uR5LDRs2jMcffxyA27dvs2TJEvr06UNcXBxDhw516LWJ/OXt7c13330HQHJyMhcuXOCXX36ha9euNG3alJ9//pnAwECr5124cCFHjx5lxIgRdrnOJ598kvnz5+s99/zzz1O3bl1eeOEF3XMBAQF2eb+dO3cybtw4+vfvT+HChfU+9++//+LmJmstQhRUEkpFnmrTpg116tTRfTx69Gg2bdpE+/bt6dixI8ePH8fX19fu75u9OmstLy8vu1+LNRo3bkzXrl11H7/00kuUK1eOhQsXulQoTUlJwcvLy6UDhoeHB3369NF7bsKECXz88ceMHj2awYMHs2TJEgdd3X/KlStHuXLl9J578cUXKVeunMH15zVvb+98fT8hhH257p/4wmGaN2/Oe++9x4ULF/jxxx/1PnfixAm6du1KSEgIPj4+1KlTh9WrVxvMERcXx2uvvUaZMmXw9vbmkUceoW/fvty6dQswvqf0+vXrDBgwgEceeQRvb2/Cw8Pp1KkT58+f140xtqf0xo0bDBo0iLCwMHx8fKhevTrz5s3TG5P9fp999hmzZs2ifPnyeHt78/jjj7N3716bv1deXl4EBwfj4aH/78eMjAw++OAD3fuUKVOGd955h9TUVL1xGo2G999/32Deh/feZd+C/fPPPxk5ciRFixbF39+fLl26cPPmTb3XKorChAkTeOSRR/Dz86NZs2b8888/Bu8RGxvLG2+8QVRUFAEBAQQGBtKmTRsOHTqkNy5768LixYt59913KVGiBH5+fhw8eBCNRsPnn39uMPfOnTvRaDQsWrTI5Pcue94lS5bwzjvvULx4cfz9/enYsSOXLl3SG7t9+3a6detGqVKl8Pb2pmTJkrz22mskJycbzLtq1SqqVq2Kj48PVatW5aeffjL6/p999hkNGzakSJEi+Pr6Urt2bZYvX27yei319ttv89RTT7Fs2TJOnjype/7nn3+mXbt2RERE4O3tTfny5fnggw/IzMzUjWnatCm//vorFy5cMLitnpaWxpgxY6hduzZBQUH4+/vTuHFjNm/enOtrBrhy5QoDBw4kLCwMb29vqlSpwpw5cwzGffnll1SpUgU/Pz+Cg4OpU6cOCxcuBOD999/nzTffBKBs2bK6ryH7dzg3P9dZWVm8//77RERE6H6ujx07JvtUhchHslIqHOK5557jnXfeYf369QwePBiAf/75h0aNGlGiRAnefvtt/P39Wbp0KZ07d2bFihV06dIFgMTERBo3bszx48cZOHAgtWrV4tatW6xevZrLly8TGhpq9D2feeYZ/vnnH1599VXKlCnDjRs32LBhAxcvXjS53y05OZmmTZty+vRpXnnlFcqWLcuyZcvo378/cXFxDB8+XG/8woULuXv3LkOGDEGj0fDJJ5/w9NNPc/bsWYu2Ety9e1cXrGNjY3W3WmfPnq037vnnn2fevHl07dqV119/nd27dzNx4kSOHz9uMiRZ4tVXXyU4OJixY8dy/vx5pk6dyiuvvKK3IjdmzBgmTJhA27Ztadu2LQcOHOCpp54iLS1Nb66zZ8+yatUqunXrRtmyZYmJiWHmzJk0adKEY8eOERERoTf+gw8+wMvLizfeeIPU1FQqVapEo0aNWLBgAa+99pre2AULFlCoUCE6deqU49f04Ycf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P2b9+vX06NFDt5XA2NeQlpbG119/bfB6f39/o7fjjc2xe/dudu3alavrdXd355lnnmHFihVG/zH54O/27du39T7n5eVFZGQkiqKQnp6uu37Arh2dWrRogYeHB998843e81999ZXd3kMIkTNZKRV56vfff+fEiRNkZGQQExPDpk2b2LBhA6VLl2b16tV6B9xPnz6dJ554gqioKAYPHky5cuWIiYlh165dXL58WXe+5Ztvvsny5cvp1q0bAwcOpHbt2sTGxrJ69WpmzJihC28POnnyJC1atKB79+5ERkbi4eHBTz/9RExMDD179jR5/S+88AIzZ86kf//+7N+/nzJlyrB8+XL+/PNPpk6dqrenzx62b9+uC8/ZX9PWrVvp2bMnlSpVArThtF+/fsyaNUt323bPnj3MmzePzp0706xZM918zz//PC+++CLPPPMMLVu25NChQ6xbt86qlcoHFS1alDfeeIOJEyfSvn172rZty99//83vv/9uMGf79u0ZP348AwYMoGHDhhw5coQFCxYYHLRuib59+zJt2jQ2b97MpEmTrHptSEgITzzxBAMGDCAmJoapU6fy6KOP6o4iq1SpEuXLl+eNN97gypUrBAYGsmLFCqMrZBMnTqRdu3Y88cQTDBw4kNjYWN25mtkr2aD9h8GUKVNo3bo1zz77LDdu3GD69Ok8+uijHD582KLrzsjI0K3gp6SkcOHCBVavXs3hw4dp1qwZs2bN0o1t2LAhwcHB9OvXj2HDhqHRaJg/f77Rf6TVrl2bJUuWMHLkSB5//HECAgLo0KED7du3Z+XKlXTp0oV27dpx7tw5ZsyYQWRkpN7XZouPP/6YzZs3U69ePQYPHkxkZCSxsbEcOHCAP/74g9jYWACeeuopihcvTqNGjQgLC+P48eN89dVXtGvXTve7Vrt2bUB7vFXPnj3x9PSkQ4cOurBqi7CwMIYPH87kyZPp2LEjrVu35tChQ7qf67xYnRVCGJHv9f7CJWQfxZL98PLyUooXL660bNlS+eKLL/SO6HnQmTNnlL59+yrFixdXPD09lRIlSijt27dXli9frjfu9u3byiuvvKKUKFFC8fLyUh555BGlX79+uiNnHj4S6tatW8rQoUOVSpUqKf7+/kpQUJBSr149ZenSpXrzPnwklKIoSkxMjDJgwAAlNDRU8fLyUqKionTzZst+P2NHTmHi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",
      "text/plain": [
       "<Figure size 800x600 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Jumlah data testing salah klasifikasi: 2 dari 15\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>JumlahLink</th>\n",
       "      <th>KataPromo</th>\n",
       "      <th>Aktual</th>\n",
       "      <th>Prediksi</th>\n",
       "      <th>P(Spam)</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>9</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0.1573</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>3</td>\n",
       "      <td>9</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0.5279</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   JumlahLink  KataPromo  Aktual  Prediksi  P(Spam)\n",
       "0           1          9       1         0   0.1573\n",
       "1           3          9       0         1   0.5279"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Jawaban B4\n",
    "# 1. Persamaan decision boundary\n",
    "print(f\"Decision boundary: x2 = -({b_lg:.4f} + {m1_lg:.4f} * x1) / {m2_lg:.4f}\")\n",
    "print(f\"                   x2 = {-b_lg/m2_lg:.4f} + ({-m1_lg/m2_lg:.4f}) * x1\")\n",
    "\n",
    "# 2. Visualisasi data testing + background probabilitas + garis boundary\n",
    "x1_min, x1_max = 0 - 1, 10 + 1\n",
    "x2_min, x2_max = 0 - 1, 12 + 1\n",
    "xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 300), np.linspace(x2_min, x2_max, 300))\n",
    "grid = np.c_[xx1.ravel(), xx2.ravel()]\n",
    "prob_grid = logreg.predict_proba(grid).reshape(xx1.shape)\n",
    "\n",
    "salah = ye_pred != ye_test      # 3. data salah klasifikasi\n",
    "\n",
    "plt.figure(figsize=(8, 6))\n",
    "cs = plt.contourf(xx1, xx2, prob_grid, levels=20, cmap=\"RdBu_r\", alpha=0.6)\n",
    "plt.colorbar(cs, label=\"P(Spam)\")\n",
    "x_b = np.linspace(x1_min, x1_max, 100)\n",
    "plt.plot(x_b, -(b_lg + m1_lg * x_b) / m2_lg, \"k--\", linewidth=2, label=\"Decision boundary (P=0.5)\")\n",
    "plt.scatter(Xe_test[ye_test == 0, 0], Xe_test[ye_test == 0, 1], c=\"tab:blue\", edgecolor=\"k\", s=70, label=\"Bukan Spam (test)\")\n",
    "plt.scatter(Xe_test[ye_test == 1, 0], Xe_test[ye_test == 1, 1], c=\"tab:red\", marker=\"s\", edgecolor=\"k\", s=70, label=\"Spam (test)\")\n",
    "if salah.any():\n",
    "    plt.scatter(Xe_test[salah, 0], Xe_test[salah, 1], s=300, facecolors=\"none\",\n",
    "                edgecolors=\"lime\", linewidths=3, label=\"Salah klasifikasi\")\n",
    "plt.xlim(x1_min, x1_max); plt.ylim(x2_min, x2_max)\n",
    "plt.title(\"Decision Boundary pada Data Testing\")\n",
    "plt.xlabel(\"JumlahLink\"); plt.ylabel(\"KataPromo\")\n",
    "plt.legend(loc=\"upper center\", bbox_to_anchor=(0.5, -0.13), ncol=2)\n",
    "plt.show()\n",
    "\n",
    "print(\"Jumlah data testing salah klasifikasi:\", salah.sum(), \"dari\", len(ye_test))\n",
    "if salah.any():\n",
    "    display(pd.DataFrame({\n",
    "        \"JumlahLink\": Xe_test[salah, 0], \"KataPromo\": Xe_test[salah, 1],\n",
    "        \"Aktual\": ye_test[salah], \"Prediksi\": ye_pred[salah],\n",
    "        \"P(Spam)\": logreg.predict_proba(Xe_test[salah]).round(4)}))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cell-33",
   "metadata": {},
   "source": [
    "**Jawaban B4 (analisis):**\n",
    "\n",
    "Decision boundary: **x₂ = −(−10,1635 + 0,8950·x₁) / 0,8434 = 12,0513 − 1,0612·x₁**. Secara kasar, email diprediksi spam jika jumlah link ditambah jumlah kata promosi bernilai sekitar 12 atau lebih. Titik di atas/kanan garis diprediksi Spam, dan di bawah/kiri garis diprediksi Bukan Spam.\n",
    "\n",
    "Terdapat **2 dari 15** data testing yang salah diklasifikasikan (ditandai lingkaran hijau pada grafik):\n",
    "\n",
    "| JumlahLink | KataPromo | Aktual | Prediksi | P(Spam) | Jenis |\n",
    "|---|---|---|---|---|---|\n",
    "| 1 | 9 | 1 (Spam) | 0 | 0,1573 | FN |\n",
    "| 3 | 9 | 0 (Bukan Spam) | 1 | 0,5279 | FP |\n",
    "\n",
    "Email (1, 9) adalah spam, tetapi jumlah linknya hanya 1 sehingga skor totalnya berada jauh di sisi \"bukan spam\". Email (3, 9) berada sangat dekat dengan garis batas (P = 0,53), sehingga sedikit saja pergeseran garis sudah mengubah prediksinya."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cell-34",
   "metadata": {},
   "source": [
    "### Soal B5 – Eksperimen Threshold\n",
    "Secara default, model memprediksi Spam jika $P(\\text{Spam}) \\geq 0.5$.\n",
    "\n",
    "1. Gunakan `predict_proba` lalu buat prediksi dengan threshold **0.3**, **0.5**, dan **0.7**.\n",
    "2. Untuk setiap threshold, hitung confusion matrix, precision, dan recall. Tampilkan dalam satu tabel.\n",
    "3. Jelaskan pengaruh menaikkan/menurunkan threshold terhadap precision dan recall. Berdasarkan jawaban Anda di Soal B3, **threshold mana yang Anda pilih** untuk filter spam? Beri alasan."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "cell-35",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-10-20T11:35:55.459004Z",
     "iopub.status.busy": "2026-10-20T11:35:55.458049Z",
     "iopub.status.idle": "2026-10-20T11:35:55.471694Z",
     "shell.execute_reply": "2026-10-20T11:35:55.470544Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Threshold</th>\n",
       "      <th>TN</th>\n",
       "      <th>FP</th>\n",
       "      <th>FN</th>\n",
       "      <th>TP</th>\n",
       "      <th>Accuracy</th>\n",
       "      <th>Precision</th>\n",
       "      <th>Recall</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.3</td>\n",
       "      <td>7</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>5</td>\n",
       "      <td>0.8000</td>\n",
       "      <td>0.7143</td>\n",
       "      <td>0.8333</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.5</td>\n",
       "      <td>8</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>5</td>\n",
       "      <td>0.8667</td>\n",
       "      <td>0.8333</td>\n",
       "      <td>0.8333</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0.7</td>\n",
       "      <td>9</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>4</td>\n",
       "      <td>0.8667</td>\n",
       "      <td>1.0000</td>\n",
       "      <td>0.6667</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Threshold  TN  FP  FN  TP  Accuracy  Precision  Recall\n",
       "0        0.3   7   2   1   5    0.8000     0.7143  0.8333\n",
       "1        0.5   8   1   1   5    0.8667     0.8333  0.8333\n",
       "2        0.7   9   0   2   4    0.8667     1.0000  0.6667"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Jawaban B5\n",
    "proba_test = logreg.predict_proba(Xe_test)\n",
    "rows = []\n",
    "for th in [0.3, 0.5, 0.7]:\n",
    "    pred = (proba_test >= th).astype(int)\n",
    "    c = confusion_matrix(ye_test, pred)\n",
    "    rows.append({\"Threshold\": th, \"TN\": c[0, 0], \"FP\": c[0, 1], \"FN\": c[1, 0], \"TP\": c[1, 1],\n",
    "                 \"Accuracy\": accuracy_score(ye_test, pred),\n",
    "                 \"Precision\": precision_score(ye_test, pred),\n",
    "                 \"Recall\": recall_score(ye_test, pred)})\n",
    "tabel_th = pd.DataFrame(rows)\n",
    "display(tabel_th.round(4))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cell-36",
   "metadata": {},
   "source": [
    "**Jawaban B5 (analisis):**\n",
    "\n",
    "| Threshold | TN | FP | FN | TP | Accuracy | Precision | Recall |\n",
    "|---|---|---|---|---|---|---|---|\n",
    "| 0,3 | 7 | 2 | 1 | 5 | 0,8000 | 0,7143 | 0,8333 |\n",
    "| 0,5 | 8 | 1 | 1 | 5 | 0,8667 | 0,8333 | 0,8333 |\n",
    "| 0,7 | 9 | 0 | 2 | 4 | 0,8667 | 1,0000 | 0,6667 |\n",
    "\n",
    "**Pengaruh threshold:** menaikkan threshold membuat model lebih \"hati-hati\" memberi label spam. FP berkurang sehingga **precision naik** (0,71 → 0,83 → 1,0), tetapi lebih banyak spam yang lolos (FN naik) sehingga **recall turun** (0,83 → 0,83 → 0,67). Menurunkan threshold berefek sebaliknya: lebih banyak spam tertangkap, tetapi lebih banyak email normal salah dianggap spam sehingga precision turun. Jadi precision dan recall saling tukar (*trade-off*).\n",
    "\n",
    "**Threshold pilihan: 0,7.** Berdasarkan B3, FP (email penting masuk spam) lebih merugikan daripada FN. Threshold 0,7 membuat **tidak ada** email normal yang salah masuk spam (FP = 0, precision = 1,0) dengan akurasi yang tetap sama dengan threshold 0,5 (0,8667). Harganya, recall turun menjadi 0,67 (2 dari 6 spam lolos), yang menurut saya masih dapat diterima karena spam yang lolos hanya mengganggu. Perlu diingat bahwa test set hanya 15 email, sehingga angka ini hanya gambaran kasar."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cell-37",
   "metadata": {},
   "source": [
    "### Soal B6 – Prediksi Email Baru\n",
    "Prediksi kelas dan probabilitas spam untuk email berikut (gunakan threshold pilihan Anda dari Soal B5):\n",
    "\n",
    "| Email | JumlahLink | KataPromo |\n",
    "|---|---|---|\n",
    "| E1 | 1 | 2 |\n",
    "| E2 | 5 | 6 |\n",
    "| E3 | 9 | 10 |\n",
    "| E4 | 2 | 11 |"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "cell-38",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-10-20T11:35:55.473488Z",
     "iopub.status.busy": "2026-10-20T11:35:55.473320Z",
     "iopub.status.idle": "2026-10-20T11:35:55.485319Z",
     "shell.execute_reply": "2026-10-20T11:35:55.484266Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Email</th>\n",
       "      <th>JumlahLink</th>\n",
       "      <th>KataPromo</th>\n",
       "      <th>P(Spam)</th>\n",
       "      <th>Prediksi</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>E1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>0.0005</td>\n",
       "      <td>Bukan Spam</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>E2</td>\n",
       "      <td>5</td>\n",
       "      <td>6</td>\n",
       "      <td>0.3478</td>\n",
       "      <td>Bukan Spam</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>E3</td>\n",
       "      <td>9</td>\n",
       "      <td>10</td>\n",
       "      <td>0.9982</td>\n",
       "      <td>Spam</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>E4</td>\n",
       "      <td>2</td>\n",
       "      <td>11</td>\n",
       "      <td>0.7116</td>\n",
       "      <td>Spam</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  Email  JumlahLink  KataPromo  P(Spam)    Prediksi\n",
       "0    E1           1          2   0.0005  Bukan Spam\n",
       "1    E2           5          6   0.3478  Bukan Spam\n",
       "2    E3           9         10   0.9982        Spam\n",
       "3    E4           2         11   0.7116        Spam"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Jawaban B6\n",
    "THRESHOLD_PILIHAN = 0.7   # threshold pilihan dari Soal B5\n",
    "\n",
    "email_baru = pd.DataFrame({\"Email\": [\"E1\", \"E2\", \"E3\", \"E4\"],\n",
    "                           \"JumlahLink\": [1, 5, 9, 2],\n",
    "                           \"KataPromo\": [2, 6, 10, 11]})\n",
    "p_baru = logreg.predict_proba(email_baru[[\"JumlahLink\", \"KataPromo\"]].values)\n",
    "email_baru[\"P(Spam)\"] = p_baru.round(4)\n",
    "email_baru[\"Prediksi\"] = np.where(p_baru >= THRESHOLD_PILIHAN, \"Spam\", \"Bukan Spam\")\n",
    "display(email_baru)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cell-39",
   "metadata": {},
   "source": [
    "---\n",
    "# Bagian C – Pemahaman Konsep (15%)\n",
    "\n",
    "Jawab dengan kalimat Anda sendiri (singkat dan jelas, 3–6 kalimat per soal).\n",
    "\n",
    "1. Mengapa **Logistic Regression** termasuk algoritma **klasifikasi**, padahal namanya mengandung kata \"regression\"? Apa peran fungsi **sigmoid**?\n",
    "2. Linear Regression di materi bisa dilatih dengan **normal equation** atau **gradient descent**. Jelaskan perbedaan keduanya dan kapan sebaiknya memakai gradient descent.\n",
    "3. Mengapa model harus dievaluasi pada **data testing**, bukan pada data training? Apa yang dimaksud dengan **overfitting**?\n",
    "4. Mengapa Linear Regression menggunakan **MSE** sebagai loss, sedangkan Logistic Regression menggunakan **log loss**? (Petunjuk: perhatikan jenis output masing-masing model.)\n",
    "5. Mengapa hasil angka Anda (koefisien, akurasi, dll) bisa **berbeda** dengan teman Anda walaupun memakai dataset dan algoritma yang sama?"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cell-40",
   "metadata": {},
   "source": [
    "**Jawaban C1:**\n",
    "Logistic Regression termasuk klasifikasi karena outputnya adalah probabilitas kelas yang kemudian diubah menjadi label diskret (Spam/Bukan Spam) dengan threshold, bukan nilai kontinu. Kata \"regression\" berasal dari komponen awalnya yang berupa kombinasi linear z = b + m₁x₁ + m₂x₂. Fungsi **sigmoid** σ(z) = 1/(1+e⁻ᶻ) memetakan nilai z yang bisa berapa saja ke rentang (0, 1) sehingga bisa ditafsirkan sebagai peluang. Z positif besar menghasilkan peluang mendekati 1, z negatif besar mendekati 0, dan z = 0 menghasilkan tepat 0,5 (batas keputusan).\n",
    "\n",
    "**Jawaban C2:**\n",
    "**Normal equation** menghitung parameter optimal secara langsung (satu kali hitung) lewat rumus θ = (XᵀX)⁻¹Xᵀy, tanpa learning rate dan tanpa iterasi. **Gradient descent** mencari parameter secara bertahap dengan menggeser parameter berlawanan arah gradien loss selama banyak iterasi, dan hasilnya bergantung pada learning rate dan jumlah iterasi (seperti terlihat di A5). Gradient descent sebaiknya dipakai ketika data atau jumlah fitur sangat besar, karena membalik matriks XᵀX mahal (sekitar O(n³) terhadap jumlah fitur). Gradient descent juga dipakai ketika modelnya tidak punya solusi tertutup, seperti Logistic Regression.\n",
    "\n",
    "**Jawaban C3:**\n",
    "Data training sudah dipakai model untuk belajar, sehingga skor di data training cenderung terlalu optimis (model bisa saja hanya menghafal). Tujuan sebenarnya adalah performa pada data baru, yang hanya bisa diukur pada **data testing** yang tidak dilihat saat training. **Overfitting** adalah kondisi saat model terlalu menyesuaikan diri pada data training (termasuk noise) sehingga performanya bagus di training tetapi buruk di data baru. Pada A4, R² training 0,9607 lebih tinggi dari R² testing 0,8137, yang merupakan contoh selisih semacam itu.\n",
    "\n",
    "**Jawaban C4:**\n",
    "Linear Regression memprediksi nilai kontinu, sehingga ukuran kesalahan yang wajar adalah jarak kuadrat antara nilai prediksi dan nilai aktual (MSE). Logistic Regression mengeluarkan probabilitas p di antara 0 dan 1 untuk label biner. Ukuran yang cocok adalah **log loss** (cross-entropy), −[y·log(p) + (1−y)·log(1−p)], karena menghukum sangat berat prediksi yang yakin tetapi salah (misal p ≈ 0 padahal aslinya 1). Log loss juga berasal dari prinsip *maximum likelihood* untuk data biner, dan bersifat konveks bila dikombinasikan dengan sigmoid. Jika MSE dipakai dengan sigmoid, permukaan loss menjadi tidak konveks dan gradiennya mengecil saat prediksi salah-yakin, sehingga training lambat.\n",
    "\n",
    "**Jawaban C5:**\n",
    "Setiap mahasiswa memakai `random_state` berbeda (dua digit terakhir NIM), sehingga pembagian data training dan testing berbeda, termasuk titik data mana yang masuk ke test set. Akibatnya koefisien, error, dan akurasi berbeda walau dataset dan algoritmanya sama. Pada dataset sekecil ini (40 dan 60 data), satu atau dua titik berbeda saja sudah bisa mengubah hasil secara cukup terlihat. Itulah mengapa hasil evaluasi pada satu split sebaiknya tidak dianggap sebagai kebenaran mutlak."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cell-41",
   "metadata": {},
   "source": [
    "---\n",
    "# Bonus (+5) – Perbandingan dengan scikit-learn\n",
    "\n",
    "Latih ulang kedua model menggunakan `sklearn.linear_model.LinearRegression` dan `sklearn.linear_model.LogisticRegression` dengan data training yang **sama**. Bandingkan koefisien dan hasil evaluasinya dengan implementasi Anda. Jika ada perbedaan (terutama pada Logistic Regression), jelaskan penyebabnya.\n",
    "\n",
    "> *Petunjuk: baca dokumentasi parameter `penalty` / `C` pada `LogisticRegression` scikit-learn.*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "cell-42",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-10-20T11:35:55.487807Z",
     "iopub.status.busy": "2026-10-20T11:35:55.486952Z",
     "iopub.status.idle": "2026-10-20T11:35:56.457098Z",
     "shell.execute_reply": "2026-10-20T11:35:56.456043Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "== Linear Regression ==\n",
      "Buatan sendiri : b=27.613127, m=9.241607, R2 test=0.813662\n",
      "scikit-learn   : b=27.613127, m=9.241607, R2 test=0.813662\n",
      "\n",
      "== Logistic Regression ==\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.12/dist-packages/sklearn/linear_model/_logistic.py:1170: UserWarning: Setting penalty=None will ignore the C and l1_ratio parameters\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Intercept</th>\n",
       "      <th>m1 (Link)</th>\n",
       "      <th>m2 (Promo)</th>\n",
       "      <th>Accuracy test</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Buatan sendiri (tanpa regularisasi)</th>\n",
       "      <td>-10.1635</td>\n",
       "      <td>0.8950</td>\n",
       "      <td>0.8434</td>\n",
       "      <td>0.8667</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>sklearn default (L2, C=1.0)</th>\n",
       "      <td>-9.4055</td>\n",
       "      <td>0.8177</td>\n",
       "      <td>0.7843</td>\n",
       "      <td>0.8667</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>sklearn tanpa regularisasi (C=inf)</th>\n",
       "      <td>-10.8933</td>\n",
       "      <td>0.9650</td>\n",
       "      <td>0.8992</td>\n",
       "      <td>0.8667</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                     Intercept  m1 (Link)  m2 (Promo)  \\\n",
       "Buatan sendiri (tanpa regularisasi)   -10.1635     0.8950      0.8434   \n",
       "sklearn default (L2, C=1.0)            -9.4055     0.8177      0.7843   \n",
       "sklearn tanpa regularisasi (C=inf)    -10.8933     0.9650      0.8992   \n",
       "\n",
       "                                     Accuracy test  \n",
       "Buatan sendiri (tanpa regularisasi)         0.8667  \n",
       "sklearn default (L2, C=1.0)                 0.8667  \n",
       "sklearn tanpa regularisasi (C=inf)          0.8667  "
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Jawaban Bonus\n",
    "from sklearn.linear_model import LinearRegression as SkLinear\n",
    "from sklearn.linear_model import LogisticRegression as SkLogistic\n",
    "\n",
    "# --- Linear Regression ---\n",
    "sk_lin = SkLinear().fit(X_train, y_train)\n",
    "print(\"== Linear Regression ==\")\n",
    "print(f\"Buatan sendiri : b={lin.intercept_:.6f}, m={lin.coef_[0]:.6f}, R2 test={r2_score(y_test, lin.predict(X_test)):.6f}\")\n",
    "print(f\"scikit-learn   : b={sk_lin.intercept_:.6f}, m={sk_lin.coef_[0]:.6f}, R2 test={r2_score(y_test, sk_lin.predict(X_test)):.6f}\")\n",
    "\n",
    "# --- Logistic Regression ---\n",
    "sk_log_default = SkLogistic(max_iter=10000).fit(Xe_train, ye_train)                  # default: L2, C=1.0\n",
    "sk_log_nopen   = SkLogistic(C=np.inf, max_iter=10000).fit(Xe_train, ye_train)    # tanpa regularisasi (C=inf)\n",
    "\n",
    "print(\"\\n== Logistic Regression ==\")\n",
    "hasil_lg = pd.DataFrame({\n",
    "    \"Intercept\": [b_lg, sk_log_default.intercept_[0], sk_log_nopen.intercept_[0]],\n",
    "    \"m1 (Link)\": [m1_lg, sk_log_default.coef_[0, 0], sk_log_nopen.coef_[0, 0]],\n",
    "    \"m2 (Promo)\": [m2_lg, sk_log_default.coef_[0, 1], sk_log_nopen.coef_[0, 1]],\n",
    "    \"Accuracy test\": [accuracy_score(ye_test, logreg.predict(Xe_test)),\n",
    "                      accuracy_score(ye_test, sk_log_default.predict(Xe_test)),\n",
    "                      accuracy_score(ye_test, sk_log_nopen.predict(Xe_test))],\n",
    "}, index=[\"Buatan sendiri (tanpa regularisasi)\", \"sklearn default (L2, C=1.0)\", \"sklearn tanpa regularisasi (C=inf)\"])\n",
    "display(hasil_lg.round(4))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cell-43",
   "metadata": {},
   "source": [
    "**Jawaban Bonus (analisis):**\n",
    "\n",
    "**Linear Regression:** hasil scikit-learn **identik** dengan implementasi sendiri (b = 27,613127; m = 9,241607; R² testing = 0,813662), karena keduanya menyelesaikan masalah least squares yang sama secara eksak.\n",
    "\n",
    "**Logistic Regression:**\n",
    "\n",
    "| Model | Intercept | m₁ (Link) | m₂ (Promo) | Accuracy test |\n",
    "|---|---|---|---|---|\n",
    "| Buatan sendiri (tanpa regularisasi) | −10,1635 | 0,8950 | 0,8434 | 0,8667 |\n",
    "| sklearn default (L2, C = 1,0) | −9,4055 | 0,8177 | 0,7843 | 0,8667 |\n",
    "| sklearn tanpa regularisasi (C = ∞) | −10,8933 | 0,9650 | 0,8992 | 0,8667 |\n",
    "\n",
    "Koefisien sklearn default **berbeda** dari implementasi sendiri. Penyebab utamanya: `LogisticRegression` di scikit-learn secara default memakai **regularisasi L2 dengan C = 1,0**, yang menambahkan penalti pada besar koefisien sehingga koefisien \"diperkecil\" (0,82 dan 0,78 dibanding 0,90 dan 0,84). Implementasi sendiri tidak memakai regularisasi. Ketika regularisasi dimatikan (C = ∞), koefisien sklearn menjadi lebih dekat dengan milik kita, tetapi masih sedikit berbeda (0,97 dan 0,90). Ini karena data hampir terpisah linear, sehingga log loss terus menurun pelan dan koefisien optimal cenderung terus membesar. Gradient descent kita berhenti di 10.000 iterasi (loss masih turun sedikit), sedangkan solver sklearn berjalan lebih jauh hingga kriteria konvergensinya. Meskipun koefisiennya berbeda, **akurasi testing ketiganya sama (0,8667)**, karena garis keputusan yang dihasilkan hampir sama."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
