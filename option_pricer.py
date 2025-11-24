import numpy as np
from scipy.stats import norm

# Création de la fonction qui renvoie le prix et les greeks de l'option selon la formule de Black-Scholes

def black_scholes(S, K, T, r, sigma, option_type = 'call'):
    
    # Gestion du cas T = 0
    if T == 0:
        if option_type == 'call':
            price = max(S - K, 0)
            delta = 1.0 if S > K else 0.0
            gamma = 0.0
            vega = 0.0
            theta = - r * K * np.exp(- r * T) if S > K else 0.0
            rho = K * T * np.exp(- r * T) if S > K else 0.0
        else : 
            price = max(S - K, 0)
            delta = - 1.0 if S > K else 0.0
            gamma = 0.0
            vega = 0.0
            theta = r * K * np.exp(- r * T) if S > K else 0.0
            rho = - K * T * np.exp(- r * T) if S > K else 0.0
        return {
            "price" : price,
            "delta" : delta,
            "gamma" : gamma,
            "vega" : vega,
            "theta" : theta,
            "rho" : rho
        }
    
    # Détermination des variables d1 et d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Calcul de N et N'
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    N_minus_d1 = norm.cdf(- d1)
    N_minus_d2 = norm.cdf(- d2)

    N_prime_d1 = norm.pdf(- d2)

    # Calcul des greeks
    if option_type == 'call':
        price = S * N_d1 - K * np.exp(- r * T) * N_d2
        delta = N_d1
        gamma = N_prime_d1 / (S * sigma * np.sqrt(T))
        vega = S * np.sqrt(T) * N_prime_d1
        theta = - (S * sigma * N_prime_d1) / (2 * np.sqrt(T)) - r * K * np.exp(- r * T) * N_d2
        rho = K * T * np.exp(- r * T) * N_d2
    elif option_type == 'put':
        price = K * np.exp(- r * T) * N_minus_d2 - S * N_minus_d1
        delta = N_d1 - 1
        gamma = N_prime_d1 / (S * sigma * np.sqrt(T))
        vega = S * np.sqrt(T) * N_prime_d1
        theta = - (S * sigma * N_prime_d1) / (2 * np.sqrt(T)) + r * K * np.exp(- r * T) * N_minus_d2
        rho = - K * T * np.exp(- r * T) * N_minus_d2
    
    return {
            "price" : price,
            "delta" : delta,
            "gamma" : gamma,
            "vega" : vega,
            "theta" : theta,
            "rho" : rho
        }

# Test 
S = 100 
K =  100
T =  1
r =  0.05
sigma = 0.2

call = black_scholes(S, K, T, r, sigma, 'call')
put = black_scholes(S, K, T, r, sigma, 'put')

print("Call (S = 100, K = 100, T = 1, r 5%, vol = 20%)")
for key, value in call.items():
    print(f"{key.capitalize():<21} : {value: >10.4}")

print("\n" + "-" * 40 + "\n")

print("Put (S = 100, K = 1100, T = 1, r 5%, vol = 20%)")
for key, value in put.items():
    print(f"{key.capitalize():<21} : {value: >10.4}")
