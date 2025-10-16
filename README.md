# Daily-stock-price-predictor
# 📈 GRU-Based Multi-Company Stock Price Prediction

> **"Predict Tomorrow’s Market with Deep Learning (GRU)"**  
> A time-series forecasting project using **Gated Recurrent Units (GRU)** to predict stock prices for multiple companies.

---

## 🧠 Project Overview

This project focuses on predicting **future stock prices** using **Deep Learning** with **GRU (Gated Recurrent Unit)** models.  
Stock prices are inherently sequential — meaning today’s price is influenced by past days.  
Conventional models often fail to capture such dependencies, but **GRU networks**, a variant of RNNs, efficiently learn from historical trends.

Our approach uses a dataset containing **Open, High, Low, Close** prices for multiple companies and trains **individual GRU models** per company.  
The trained model then predicts future closing prices, allowing investors to assess **undervalued or overvalued** stocks.

---

## 🎯 Objectives

- Build a **deep learning-based time series prediction model** using GRUs.  
- Train separate models for each company’s data.  
- Predict the **future Close price** from historical data.  
- Visualize **actual vs predicted** price trends for insights.  
- Enable investors to make **data-driven decisions**.

---

## ⚙️ Key Features

| Feature | Description |
|----------|--------------|
| 🧩 Multi-Company Training | Separate GRU models trained for each company. |
| 🤖 Deep Learning (GRU) | Captures temporal dependencies in stock data. |
| 🔄 Sequential Data | Uses 60 previous days to predict the next day's price. |
| ⚖️ Scaled Data | MinMaxScaler ensures stable learning. |
| ⏸ Early Stopping | Prevents overfitting and saves best model weights. |
| 📊 Visualization | Graphs for **Actual vs Predicted** closing prices. |
| 💾 Model Saving | Each trained model saved in `.keras` format. |

---

## 📂 Dataset Description

- **File:** `final_destination.csv`  
- **Format:** CSV  
- **Required Columns:**
  - `Date` → Date of record (`YYYY-MM-DD`)
  - `company` → Company name (e.g., MRF, TATA, Reliance)
  - `Open`, `High`, `Low`, `Close` → Daily stock data

### Example Data

| Date | company | Open | High | Low | Close |
|------|----------|------|------|-----|-------|
| 2023-01-01 | MRF | 8123.5 | 8230.4 | 8100.2 | 8210.1 |
| 2023-01-02 | MRF | 8210.1 | 8300.8 | 8190.0 | 8295.2 |
| 2023-01-01 | TATA | 520.3 | 530.1 | 518.0 | 528.2 |

---

## 🧩 Working of the Solution

### 1️⃣ Data Preprocessing
- Converts `Date` to datetime and sorts chronologically.  
- Groups data per company.  
- Scales `Open`, `High`, `Low`, `Close` using **MinMaxScaler**.

### 2️⃣ Sequence Creation

Each sample represents 60 previous days of stock prices predicting the 61st day’s closing price.

```python
def create_sequences(data, time_steps=60):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i-time_steps:i])
        y.append(data[i, 3])  # 'Close' is at index 3
    return np.array(X), np.array(y)
3️⃣ Model Architecture
model = Sequential([
    GRU(100, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')


Layers Explanation:

GRU Layer (100 units): Learns temporal patterns.

Dropout Layer (0.2): Reduces overfitting.

Dense Layer (1): Outputs predicted price.

4️⃣ Model Training

Uses 80% data for training and 20% for testing.

Applies EarlyStopping to restore best weights when validation loss stops improving.

5️⃣ Prediction and Evaluation

Predicts future closing prices.

Inverses scaling to original values.

Plots actual vs predicted trends for each company.

