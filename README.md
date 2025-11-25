# ⚡ Hybrid Option Pricer: Black-Scholes + Machine Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Finance](https://img.shields.io/badge/Quantitative-Finance-green)
![ML](https://img.shields.io/badge/Machine-Learning-orange)
![Status](https://img.shields.io/badge/Status-Educational-yellow)

## 📖 Overview

This project implements a **Hybrid Pricing Engine** for European Options. It combines the rigorous mathematical foundation of the **Black-Scholes-Merton (BSM)** model with the predictive power of **Machine Learning (Random Forest)** to correct market biases (such as the Volatility Smile).

The goal is to demonstrate how a data-driven approach can improve pricing accuracy by predicting the "spread" between the theoretical BSM price and the real Market price.

## 🚀 Key Features

* **Black-Scholes Engine:** Vectorized implementation of the BSM formula including Greeks calculation ($\Delta, \Gamma, \nu, \Theta, \rho$).
* **Volatility Solver:** Implementation of the **Newton-Raphson** algorithm to reverse-engineer Implied Volatility from market prices.
* **Real-Time Data Pipeline:** Automated fetching of option chains (Calls/Puts) using the `yfinance` API.
* **Hybrid ML Model:** Uses a **Random Forest Regressor** to predict the pricing error of a naive BSM model (Flat Volatility) and correct it.

## 🧮 Mathematical Background

### 1. The Pricing Formula
The theoretical price is calculated using the standard BSM equation:

$$C(S, t) = N(d_1)S_t - N(d_2)Ke^{-r(T-t)}$$

### 2. The Hybrid Approach
Standard BSM assumes constant volatility ($\sigma$), which fails to capture the "Volatility Smile" observed in markets.
Our Hybrid model defines the final price as:

$$Price_{Hybrid} = Price_{BS\_Naive}(\sigma_{mean}) + \epsilon_{ML}(Strike, Moneyness)$$

Where $\epsilon_{ML}$ is the error term predicted by the Random Forest model.

## 📊 Results

The model was tested on **NVDA** (NVIDIA) option chains.
By correcting the naive assumption of constant volatility, the Hybrid Model significantly reduces the Mean Squared Error (MSE).

| Model | MSE (Pricing Error) |
| :--- | :--- |
| **Black-Scholes (Naive)** | 0.3454 |
| **Hybrid (BS + ML)** | **0.2625** |
| **Improvement (Alpha)** | **~24%** |

*(Results may vary depending on real-time market data).*

## 🛠️ Installation & Usage

2.  **Libraries**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the script**
    ```bash
    python main.py
    ```

## 📂 Project Structure