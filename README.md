# Wind Power Forecasting and Power Quality Disturbance Classification

This project investigates the impact of **hyperparameter tuning** on machine learning models for **time-series forecasting** and **classification** tasks in the energy domain.  
Two real-world applications are explored:

- **Wind Power Forecasting** – predicting power generation from time-series data  
- **Power Quality Disturbance Classification** – identifying disturbance events using the ZeroOne Power Quality dataset  

The work compares both **deep learning architectures** and **ensemble methods**, evaluating how tuning affects their performance.

---

## Models Implemented

### Deep Learning
- Multilayer Perceptron (MLP)  
- Convolutional Neural Network (CNN)  
- Long Short-Term Memory (LSTM)  
- CNN-LSTM hybrid  
- AutoML / KerasTuner for hyperparameter search  

### Ensemble Methods
- Random Forest  
- XGBoost  
- LightGBM  

---

## Workflow

1. Data preprocessing (scaling, normalization, train/test splitting)  
2. Model construction (neural networks + ensemble baselines)  
3. Hyperparameter tuning (learning rate, optimizers, dropout, layers, batch size, epochs)  
4. Training and evaluation using MAE/RMSE (forecasting) and Accuracy/F1 (classification)  
5. Comparison of tuned vs. untuned performance  

---

## How to Run

```bash
pip install -r requirements.txt

# Run classification example
python Classification/Classification_XGBoost_ZeroOne.py

# Run forecasting example
python Forecasting/Forecasting_CNN_LSTM.py
