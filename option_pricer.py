import numpy as np
from scipy.stats import norm
import yfinance as yf
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def black_scholes(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> dict:
    """
    Calcule le prix théorique d'une option Européenne et ses Grecques selon le modèle Black-Scholes-Merton.

    Args:
        S (float): Prix Spot de l'actif sous-jacent.
        K (float): Prix d'exercice (Strike).
        T (float): Temps jusqu'à maturité (en années).
        r (float): Taux d'intérêt sans risque (ex: 0.05 pour 5%).
        sigma (float): Volatilité de l'actif sous-jacent (ex: 0.20 pour 20%).
        option_type (str): 'call' ou 'put'.

    Returns:
        dict: Dictionnaire contenant le Prix, Delta, Gamma, Vega, Theta, Rho.
    """
    # Gestion du cas limite (Expiration immédiate)
    if T <= 1e-5:
        if option_type == 'call':
            price = max(S - K, 0)
            delta = 1.0 if S > K else 0.0
            rho = 0.0
        else:
            price = max(S - K, 0)
            delta = -1.0 if S < K else 0.0
            rho = 0.0
        return {
            "price": price, "delta": delta, "gamma": 0.0, 
            "vega": 0.0, "theta": 0.0, "rho": rho
        }
    
    # Calcul des d1 et d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Pré-calcul des lois normales
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    N_minus_d1 = norm.cdf(-d1)
    N_minus_d2 = norm.cdf(-d2)
    N_prime_d1 = norm.pdf(d1)

    # Calcul du Prix et des Grecques
    if option_type == 'call':
        price = S * N_d1 - K * np.exp(-r * T) * N_d2
        delta = N_d1
        theta = -(S * sigma * N_prime_d1) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * N_d2
        rho = K * T * np.exp(-r * T) * N_d2
    elif option_type == 'put':
        price = K * np.exp(-r * T) * N_minus_d2 - S * N_minus_d1
        delta = N_d1 - 1
        theta = -(S * sigma * N_prime_d1) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * N_minus_d2
        rho = -K * T * np.exp(-r * T) * N_minus_d2
    
    gamma = N_prime_d1 / (S * sigma * np.sqrt(T))
    vega = S * np.sqrt(T) * N_prime_d1
    
    return {
        "price": price, "delta": delta, "gamma": gamma,
        "vega": vega, "theta": theta, "rho": rho
    }

def volatility(market_price: float, S: float, K: float, T: float, r: float, option_type: str = "call") -> float:
    """
    Retrouve la Volatilité Implicite à partir du prix du marché via l'algorithme de Newton-Raphson.
    
    Résout f(sigma) = BS(sigma) - MarketPrice = 0

    Args:
        market_price (float): Le prix observé sur le marché.
        S, K, T, r: Paramètres standard BS.

    Returns:
        float: La volatilité implicite (sigma) ou None si l'algo ne converge pas.
    """
    sigma = 0.5 # Point de départ arbitraire
    tol = 1e-5  # Tolérance de précision
    max_iter = 100

    for i in range(max_iter):
        bs_metrics = black_scholes(S, K, T, r, sigma, option_type)
        price = bs_metrics['price']
        vega = bs_metrics['vega']

        diff = price - market_price

        if abs(diff) < tol:
            return sigma
        
        # Éviter la division par zéro si Vega est trop faible (option loin de la monnaie)
        if abs(vega) < 1e-8:
            return None

        # Mise à jour Newton-Raphson : x_{n+1} = x_n - f(x_n) / f'(x_n)
        sigma -= diff / vega
        
    return None

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    # 1. Configuration et Récupération des données
    stock_name = "NVDA"
    print(f"--- Récupération des données pour {stock_name} ---")
    
    stock = yf.Ticker(stock_name)
    S = stock.history(period="1d")['Close'].iloc[-1]

    # Tentative de récupérer une maturité à moyen terme (~1 mois)
    try:
        expiry = stock.options[4] 
    except IndexError:
        expiry = stock.options[-1] # Fallback
        
    opt = stock.option_chain(expiry)
    calls = opt.calls.copy()

    # Calcul du temps en années
    T_years = (pd.to_datetime(expiry) - pd.Timestamp.now()).days / 365.0
    if T_years <= 0: T_years = 1/365 # Sécurité
    
    r = 0.045 # Taux sans risque fixe (ex: US Treasury 10Y)

    print(f"Spot: ${S:.2f} | Expiry: {expiry} | T: {T_years:.4f} ans")

    # 2. Calculs Financiers (BS & IV)
    # Note : On utilise 'apply' pour vectoriser les calculs sur le DataFrame
    calls['BS_Price'] = calls.apply(lambda row: black_scholes(S, row['strike'], T_years, r, row['impliedVolatility'])['price'], axis=1)
    calls['Newton-Raphson'] = calls.apply(lambda row: volatility(row['lastPrice'], S, row['strike'], T_years, r), axis=1)

    # Nettoyage des données (suppression des échecs de convergence et penny options)
    calls = calls.dropna(subset=['Newton-Raphson', 'lastPrice'])
    calls = calls[calls['lastPrice'] > 0.5]

    # 3. Machine Learning (Modèle Hybride)
    print("--- Entraînement du Modèle Hybride ---")
    
    # Création du "Modèle Naïf" (Black-Scholes avec volatilité constante moyenne)
    # L'objectif est de simuler un modèle imparfait que le ML devra corriger
    vol_naive = calls['impliedVolatility'].mean() 
    calls['BS_Naive_Price'] = calls.apply(lambda row: black_scholes(S, row['strike'], T_years, r, vol_naive)['price'], axis=1)

    # Target (Y) : L'erreur du modèle Naïf par rapport au marché
    calls['Error_Naive'] = calls['lastPrice'] - calls['BS_Naive_Price']

    # Features (X) : Strike et Moneyness (S/K)
    calls['Moneyness'] = S / calls['strike']
    features = ['strike', 'Moneyness'] 

    X = calls[features]
    y = calls['Error_Naive']

    # Split Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    predicted_error = model.predict(X_test)

    # 4. Évaluation et Résultats
    market_price_test = calls.loc[X_test.index, 'lastPrice']
    
    # Performance du modèle Naïf
    price_naive = calls.loc[X_test.index, 'BS_Naive_Price']
    mse_naive = mean_squared_error(market_price_test, price_naive)

    # Performance du modèle Hybride (Naïf + Correction ML)
    final_price_prediction = calls.loc[X_test.index, 'BS_Naive_Price'] + predicted_error
    mse_hybrid = mean_squared_error(market_price_test, final_price_prediction)

    print(f"\n--- RÉSULTATS ---")
    print(f"MSE Black-Scholes Naïf (Baseline) : {mse_naive:.4f}")
    print(f"MSE Hybride (BS + ML Correction)  : {mse_hybrid:.4f}")
    print(f"Amélioration (Alpha)              : {(1 - mse_hybrid/mse_naive)*100:.2f}%")

    # 5. Visualisation
    plt.figure(figsize=(15, 6)) 

    # Plot 1 : Volatility Smile
    plt.subplot(1, 2, 1)
    plt.plot(calls['strike'], calls['Newton-Raphson'], 'o-', label='Implied Volatility (Newton)', color='orange', markersize=4)
    plt.axhline(y=calls['Newton-Raphson'].mean(), color='blue', linestyle='--', label='Mean Volatility')
    plt.title(f'Volatility Smile ({stock_name})')
    plt.xlabel('Strike')
    plt.ylabel('Implied Vol')
    plt.legend()
    plt.grid(True)

    # Plot 2 : Correction ML
    plt.subplot(1, 2, 2)
    subset = X_test.sort_values(by='strike')
    indices = subset.index

    plt.scatter(calls.loc[indices, 'strike'], calls.loc[indices, 'Error_Naive'], color='red', label='Erreur du Modèle Naïf', alpha=0.5)
    plt.plot(calls.loc[indices, 'strike'], model.predict(X.loc[indices]), color='green', linewidth=2, label='Correction prédite par le ML')

    plt.title('Capacité du ML à prédire le biais de BS')
    plt.xlabel('Strike')
    plt.ylabel('Pricing Error ($)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()