# Cours Intégré (CI) 2 : Outlier Detection, Cleaning Trade-offs, Imbalanced Data & Scikit-Learn Pipelines

---

## Prerequisites & Environment Setup

Each group (2–3 students) must work inside a dedicated Jupyter Notebook or VS Code interactive session.

```bash
# Verify Python environment
python3 --version  # >= 3.10

# Required libraries
pip install pandas numpy matplotlib seaborn scikit-learn
```

Make sure `hospital_readmissions_raw.csv` is in your working directory. 


---

## Step-by-Step Guided Lab Guide

---

### Part 1: Ingestion Handshake & Advanced Deduplication (00:00 – 00:25)

In real-world data engineering, raw data exports rarely arrive cleanly indexed. They contain both **exact byte-for-byte duplicates** (from network re-posts or repeated ETL syncs) and **semantic duplicates** (different tracking IDs assigned to the exact same physical event).

#### 1.1 Ingestion & Dimensional Handshake
Open your notebook. Ingest `hospital_readmissions_raw.csv` and inspect its shape, column names, and detected data types:

-> add python code

#### 1.2 Uncovering Dirty Formats in `lab_charges_eur`
Inspect a random sample of `lab_charges_eur`. Notice the presence of currency symbols (`€`), thousands separators (`,`), negative values, and missing tokens:

-> add python code

Create a cleaning function to parse `lab_charges_eur` into a clean numerical float:
1. Strip `€`, spaces, and commas.
2. Convert invalid tokens (`"N/A"`, `""`, whitespace) to `np.nan`.
3. Identify impossible negative charges and coerce them to `np.nan`.
4. Coerce `blood_glucose_mg_dl` to float via `pd.to_numeric`.

-> add python code

#### 1.3 Exact vs Semantic Duplicates
Deduplication is not just running `df.drop_duplicates()`. We must distinguish between two levels:
* **Level 1 (Exact row duplicates):** Identical values across *every* column.
* **Level 2 (Semantic / Key duplicates):** Two records share the same clinical event (`patient_id` and `admission_date`), but received different system-generated `admission_id` values due to billing retry scripts.

Execute both deduplication steps sequentially:

-> add python code

> **Discussion Question :**  
> If two rows share the same `patient_id` and `admission_date` but have different `lab_charges_eur`, what business logic should decide which record to keep (`first`, `last`, or `max`)?

#### 1.4 Defining the Dataset Grain
In a Markdown cell in your notebook, answer:
* **Question :** What is the exact unit of observation (the grain) of this table after deduplication?
* **Clinical context :** Can a single patient appear multiple times in this dataset across different dates? Why is that clinically meaningful for readmission risk?

---

### Part 2: The Outlier Detection Tournament — Z-Score vs IQR (00:25 – 00:50)

```
                       ┌──────────────────────────────────────────────┐
                       │          OUTLIER DETECTION METHODS           │
                       └──────────────────────┬───────────────────────┘
                                              │
               ┌──────────────────────────────┴──────────────────────────────┐
               ▼                                                             ▼
 ┌───────────────────────────┐                                 ┌───────────────────────────┐
 │     PARAMETRIC METHOD     │                                 │   NON-PARAMETRIC METHOD   │
 │         (Z-SCORE)         │                                 │           (IQR)           │
 ├───────────────────────────┤                                 ├───────────────────────────┤
 │ • Assumes Normal/Gaussian │                                 │ • No distribution shape   │
 │ • Sensitive to extremes   │                                 │   assumption              │
 │ • Threshold: |z| > 3.0    │                                 │ • Fences: Q1/Q3 ± 1.5*IQR │
 └───────────────────────────┘                                 └───────────────────────────┘
```

#### 2.1 The Parametric Rule: Z-Score
Recall the Z-score formula:
$$z_i = \frac{x_i - \mu}{\sigma}$$
Under a Gaussian distribution, the Empirical Rule states that $99.73\%$ of observations lie within $[-3\sigma, +3\sigma]$. Observations with $|z| > 3$ are classical statistical outliers.

Compute the Z-score for `systolic_bp` and plot its distribution with vertical lines at $\pm 3\sigma$:

-> add python code

* **Investigation :** Are the outliers detected in `systolic_bp` impossible physical measurements, or are they valid extreme clinical conditions (e.g., severe hypotension or hypertensive crises)?

#### 2.2 The Z-Score Trap on Skewed Features
Now compute the Z-scores for `emergency_visits_last_year` and `length_of_stay_days`. 
* Plot their histograms and calculate their skewness:
  $$\text{Skew} = \frac{n}{(n-1)(n-2)} \sum \left(\frac{x_i - \bar{x}}{s}\right)^3$$

-> add python code

> **Critical Thinking Question :**  
> Why does Z-score fail on heavily skewed distributions? How do extreme positive values artificially inflate both $\bar{x}$ and the sample standard deviation $s$, leading to the **masking effect** (under-detecting true outliers)?

#### 2.3 The Non-Parametric Rule: Tukey's IQR Fences
The Interquartile Range (IQR) relies on rank-based percentiles and is robust to extreme values:
$$\text{IQR} = Q_3 - Q_1$$
$$\text{Lower Fence} = Q_1 - 1.5 \times \text{IQR}, \qquad \text{Upper Fence} = Q_3 + 1.5 \times \text{IQR}$$

Calculate the IQR fences for `length_of_stay_days` and `emergency_visits_last_year`. Plot boxplots showing the outlier points outside the whiskers:

-> add python code

#### 2.4 Corrupted Outliers vs Valid Extreme Signals

Not all outliers are created equal. Distinguish between:
1. **Corrupted Outliers (Data Errors / Hardware Glitches):** Values that violate biological reality (e.g., `patient_age < 0` or `patient_age > 120`, `heart_rate_bpm <= 0` or `heart_rate_bpm > 300`).
2. **Valid Extreme Signals (Black Swans):** Real high-utilizer patients with 18 ER visits or €50,000 intensive care stays.

Audit the dataset for corrupted sensor/age values and set them to `np.nan` so they can be imputed properly later:

-> add python code

---

### Part 3: Imbalanced Target Awareness & The Golden Split (00:50 – 01:10)

#### 3.1 Auditing the Readmission Target
Calculate the percentage of positive cases for `readmitted_30d`:

-> add python code

#### 3.2 The Accuracy Paradox
Suppose a lazy dummy model simply predicts `0` (No Readmission) for every single patient. Calculate its accuracy:

-> add python code

* **Written Analysis :** Why is accuracy a dangerous and misleading metric in imbalanced healthcare or fraud domains? What happens if a hospital relies on a model with 94% accuracy that catches 0% of patients who will relapse?

#### 3.3 The Core Discipline: Stratified Train/Test Split FIRST

> 🔥 **THE GOLDEN RULE OF MACHINE LEARNING ENGINEERING :**  
> **NEVER compute outlier thresholds, IQR fences, or imputation statistics on the entire dataset.**  
> Splitting must occur **BEFORE** calculating any statistical parameters. Furthermore, for imbalanced targets, the split MUST be **stratified** to preserve the minority class ratio in both splits.

Separate features ($X$) and target ($y$). Perform an 80/20 train/test split stratified on `readmitted_30d`:

-> add python code

Verify that the readmission percentage is identical in `y_train` and `y_test`:

-> add python code

---

### Part 4: The Cleaning Trade-off Matrix (01:10 – 01:30)

Complete the preprocessing trade-off specification table before writing transformation code:

| Feature                      | Issue Detected                               | Action (Drop / Impute / Cap / Transform) | Justification (Statistical & Business) |
| :--------------------------- | :------------------------------------------- | :--------------------------------------- | :------------------------------------- |
| `patient_age`                | Corrupted negative & $>200$ values           | ...                                      | ...                                    |
| `heart_rate_bpm`             | Sensor zeros & 999 disconnects               | ...                                      | ...                                    |
| `blood_glucose_mg_dl`        | ~4.5% missing lab values                     | ...                                      | ...                                    |
| `emergency_visits_last_year` | Extreme valid right-tail counts              | ...                                      | ...                                    |
| `length_of_stay_days`        | Right-skewed distribution                    | ...                                      | ...                                    |
| `lab_charges_eur`            | High variance, missing values, extreme bills | ...                                      | ...                                    |
| `insurance_type`             | ~2.5% missing categorical values             | ...                                      | ...                                    |

> **Discussion Questions :**
> 1. When is it acceptable to **drop** a row versus **imputing** it?
> 2. When should you **cap/Winsorize** an outlier versus applying a **logarithmic transform** ($\log(1+x)$)?

---

### Part 5: Production Scikit-Learn Pipelines (01:30 – 01:50)

In CI-1, you manually fit imputers and manually called `.transform()` on train and test sets. In production systems, maintaining dozens of manual transformation lines leads to silent bugs and accidental leakage.

Instead, we encapsulate our entire cleaning logic into a single **Scikit-Learn `Pipeline`** and **`ColumnTransformer`**.

```
                           THE COMPOSITE CLEANING PIPELINE
                           
       ┌───────────────────────┐
       │   X_train (Raw)       │
       └──────────┬────────────┘
                  │
        ┌─────────┴─────────────────────────────────┐
        ▼                                           ▼
[ Numeric Features ]                      [ Categorical Features ]
        │                                           │
  SimpleImputer(median)                       SimpleImputer(most_frequent)
        │                                           │
  IQROutlierCapper(factor=1.5)                      │
        │                                           │
        └─────────────────┬─────────────────────────┘
                          │  ColumnTransformer
                          ▼
              ┌───────────────────────┐
              │  X_train_clean (Prep) │
              └───────────────────────┘
```

#### 5.1 Building a Custom Scikit-Learn Transformer: `IQROutlierCapper`

To use IQR capping inside an `sklearn.pipeline.Pipeline`, create a custom transformer that conforms to the Scikit-Learn API:
1. Inherit from `BaseEstimator` and `TransformerMixin`.
2. Implement `.fit(X, y=None)`: Computes $Q_1, Q_3$, and the lower/upper fences **exclusively from the training data**.
3. Implement `.transform(X)`: Clips values outside the learned fences using `np.clip()`.

Write the `IQROutlierCapper` class:

-> add python code

#### 5.2 Assembling the Preprocessing Pipeline
Define:
* `num_cols`: `['patient_age', 'heart_rate_bpm', 'systolic_bp', 'blood_glucose_mg_dl', 'emergency_visits_last_year', 'length_of_stay_days', 'lab_charges_eur']`
* `cat_cols`: `['primary_diagnosis', 'admission_urgency', 'insurance_type']`

Build a `ColumnTransformer` that:
1. Chains `SimpleImputer(strategy='median')` and `IQROutlierCapper(factor=1.5)` for numeric columns.
2. Applies `SimpleImputer(strategy='most_frequent')` for categorical columns.

-> add python code

#### 5.3 Fitting on Train, Transforming Both Sets
Fit the unified pipeline **exclusively on `X_train`**, then transform both `X_train` and `X_test`:

-> add python code

Verify that neither `X_train_clean` nor `X_test_clean` contains any null values, and that outliers were successfully bounded:

-> add python code

---

### Part 6: Synthesis, Deliverable Audit & Export (01:50 – 02:00)

#### 6.1 Recombining & Exporting Cleaned Splits
Recombine the processed feature arrays with their column names and the target series, then export them:

-> add python code

---

## Deliverables & Assessment

At the conclusion of the 2-hour session, each group must submit:
1. **The Completed Jupyter Notebook (`CI2_DEP_<Group_Name>.ipynb`)** containing:
   - Full sequential execution without errors from top to bottom.
   - Systematic EDA diagnostic markdown cells explaining the **exact and semantic duplicates**, and the **grain** of the table.
   - The comparative study between **Z-score** (Gaussian) and **IQR** (Skewed).
   - The stratified train/test split executed **before** computing any outlier boundaries or imputer statistics.
   - The trade-off specification table explaining when to drop, impute, cap, or transform.
   - The custom `IQROutlierCapper` class and the integrated `ColumnTransformer` Scikit-Learn pipeline.
   - The explanation of class imbalance and the **Accuracy Paradox**.
2. **Exported CSVs :** `train_cleaned.csv` and `test_cleaned.csv`.
