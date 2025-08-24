# 2025 Infodengue-Mosqlimate dengue Forecast Sprint


## **Team and Contributors**

* **Team:** GeoHealth Dengue Forecasting Team
* **Contributors:**

  * Xiang Chen – King Abdullah University of Science and Technology (KAUST)
  * Paula Moraga - King Abdullah University of Science and Technology (KAUST)

---

## **Repository Structure**

* **`lstm_forecast.py`** – Core script for LSTM-based dengue case forecasting for multiple tasks.
* **`output/`** – Stores prediction files for each state and task (Task 1, 2, 3).

---

## **Libraries and Dependencies**

* **Core:** `numpy`, `pandas`, `argparse`
* **Deep Learning:** `torch`
* **Others:** `os`

Install via:

```bash
conda install numpy pandas pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

---

## **Data and Variables**

* **Dataset:** `state_level_dengue_climate.csv`
* **Variables:**

  * `uf` (state), `epiweek`, `year`, `casos` (dengue cases)
  * Climate features:
    `temp_min`, `temp_med`, `temp_max`,
    `precip_min`, `precip_med`, `precip_max`,
    `pressure_min`, `pressure_med`, `pressure_max`,
    `rel_humid_min`, `rel_humid_med`, `rel_humid_max`,
    `thermal_range`, `rainy_days`
* **Preprocessing:**

  * Min-Max normalization applied per state on training period only.
  * Seasonal imputation for missing covariates beyond observed weeks:

    * For weeks after EW25, we use **the average value of the same epidemiological week across all previous years** for each climate variable.

---

## **Data Usage Restriction**

To respect the restriction of **using only data up to EW25 of the current year for training**, we:

* Used a Boolean mask (`train_1`, `train_2`, `train_3`) to select valid training data.
* For prediction periods (EW41 of current year → EW40 of next year), if climate variables were missing, we filled them with **historical averages of the same week across past years** (ensuring no future leakage).

---

## **Model Training**

* **Model:** `SimpleLSTM` (1–2 LSTM layers + Fully Connected layer).
* **Hyperparameters:**

  * Tunable: `epochs`, `hidden_dim`, `num_layers`, `window_size`
  * Example:

    ```bash
    python lstm_forecast.py \
      --epochs 100 \
      --num_layers 2 \
      --hidden_dim 128 \
      --window_size 12 \
      --data_file state_level_dengue_climate.csv \
      --results_dir output
    ```
* **Optimization:** Adam optimizer, learning rate = 0.001
* **Loss:** MSE

---

## **Predictive Uncertainty**

Prediction intervals computed using **Adaptive Conformal Prediction**:

* Based on residual quantiles in a sliding calibration window (default = 15 weeks).
* For each confidence level (50%, 80%, 90%, 95%), compute:

  ```python
  q = np.quantile(|error|, confidence_level)
  lower = prediction - q
  upper = prediction + q
  ```
* Applied for all tasks; extended Task 3 predictions use the same logic after merging.
