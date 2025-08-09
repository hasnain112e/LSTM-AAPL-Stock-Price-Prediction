# LSTM-AAPL-Stock-Price-Prediction
# 📈 AAPL Stock Price Prediction using LSTM

## 📌 Project Overview
This project demonstrates the use of **Long Short-Term Memory (LSTM)** neural networks for predicting Apple Inc. (AAPL) stock closing prices using historical data from **Yahoo Finance**.  
The aim is to show how LSTMs can model temporal dependencies in time-series financial data.

## 📊 Dataset
- **Source:** Yahoo Finance via `yfinance` library
- **Date Range:** 2018-01-01 to 2023-12-31
- **Target Variable:** Closing price (`Close` column)
- Missing values handled using forward-fill

## 🚀 Steps Performed
1. **Data Acquisition:**
   - Downloaded AAPL stock price data from Yahoo Finance
   - Selected only the 'Close' price

2. **Preprocessing:**
   - Normalized values to [0, 1] using MinMaxScaler
   - Created 60-day sequences to predict the next day's price
   - Reshaped data for LSTM input

3. **Model Architecture:**
   - LSTM (50 units, return_sequences=True)
   - Dropout (0.2)
   - LSTM (50 units, return_sequences=False)
   - Dropout (0.2)
   - Dense(1) output layer
   - Optimizer: Adam
   - Loss: Mean Squared Error

4. **Training & Evaluation:**
   - Train-test split: 80%-20% (no shuffle)
   - Epochs: 20
   - Batch size: 32
   - Evaluated with RMSE
   - Plotted Actual vs Predicted prices

5. **Visualization:**
   - Historical vs Predicted Price plot
   - Clear labels, title, and legend

## 🛠 Technologies Used
- Python
- NumPy, Pandas
- Matplotlib
- scikit-learn
- yfinance
- TensorFlow/Keras

## 📈 Output
- **Model Metric:** RMSE ~ (example value after training)
- Graph comparing actual and predicted prices

## 🔧 How to Run
```bash
# Clone the repository
git clone https://github.com/yourusername/LSTM-AAPL-Stock-Price-Prediction.git
cd LSTM-AAPL-Stock-Price-Prediction

# Install dependencies
pip install yfinance numpy pandas scikit-learn tensorflow matplotlib

# Run Jupyter Notebook
jupyter notebook lstm_aapl_stock_prediction.ipynb
