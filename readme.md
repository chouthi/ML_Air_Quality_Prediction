## OVERVIEW

This application predicts **PM2.5 concentration** in **Ho Chi Minh City** based on environmental features:

- CO (Carbon monoxide)
- O₃ (Ozone)
- NO₂ (Nitrogen dioxide)
- TSP (Total Suspended Particulates)
- SO₂ (Sulfur dioxide)
- Temperature
- Humidity

---

## 🚀 HOW TO RUN

### Step 1: Install dependencies

- Make sure you have Python installed.
- Then run:

```pip install -r requirements.txt ```

### Step 2: Navigate to the demo folder
- Run: ```cd ./src/demo/```

### Step 3: Run the app with Streamlit
- Run: ```python -m streamlit run app.py```

### Step 4: Open the app in browser
- After running, Streamlit will provide a link like: http://localhost:8501

### Step 5: Explore Sample Data & Test the Prediction

#### Sample Dataset Preview

Below is the original dataset used to train the prediction model:

![Sample Dataset](data/Air Quality Ho Chi Minh City.csv)

> You can copy values from this table (excluding PM2.5) and input them into the app to test the model’s prediction.

---

#### 🧪 Example: 3 Valid Sample Rows for Testing

| TSP     | O3       | CO       | NO2      | SO2     | Temperature | Humidity |
|---------|----------|----------|----------|---------|-------------|----------|
| 32.93571 | 55.43138 | 1330.451 | 112.7408 | 393     | 28.3619     | 63.1881  |
| 22.5217 | 53.7862  | 1322.293 | 114.3315 | 393     | 28.3000     | 65.1883  |
| 98.1778 | 131.7391 | 976.6467 | 125.4667 | 285.2889| 28.3667     | 65.9778  |


---

### Model Accuracy Disclaimer

This application includes 2 machine learning models for comparison purposes. Among them:

- ✅ **Random Forest (RF)** is our primary and most reliable model. It delivers stable and accurate PM2.5 predictions.
- ⚠️ **K-Nearest Neighbors (KNN)** is included for educational comparison only. Its predictions may be **less accurate** due to its simpler nature and sensitivity to noisy data.

> 👉 For best results, we recommend using the **Random Forest** model when testing the prediction functionality.
