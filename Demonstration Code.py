import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cmath
from matplotlib.widgets import Slider, Button
from scipy.optimize import brentq
from scipy.stats import norm

###############################################################################
# 1) MARKET DATA LOADING & OPTION PRICE EXTRACTION (FOR A SINGLE QUOTE DATE)
###############################################################################
data_file = "spx_eod_202307.txt"  # example file; adjust path as needed
delimiter = ","
df = pd.read_csv(data_file, sep=delimiter)
df.columns = [col.strip() for col in df.columns]
df["[QUOTE_DATE]"]  = pd.to_datetime(df["[QUOTE_DATE]"].str.strip())
df["[EXPIRE_DATE]"] = pd.to_datetime(df["[EXPIRE_DATE]"].str.strip())
df["[STRIKE]"]    = pd.to_numeric(df["[STRIKE]"], errors="coerce")
df["[C_IV]"]      = pd.to_numeric(df["[C_IV]"], errors="coerce")
if "[C_ASK]" not in df.columns:
    raise ValueError("Column [C_ASK] not found in data.")
if "[UNDERLYING_LAST]" not in df.columns:
    df["[UNDERLYING_LAST]"] = 100.0

# Compute tau in years.
df["tau"] = (df["[EXPIRE_DATE]"] - df["[QUOTE_DATE]"]).dt.days / 365.0

# Select a target quote date.
target_quote_date = pd.to_datetime("2023-07-03")
df_quote = df[df["[QUOTE_DATE]"] == target_quote_date].copy()
df_quote = df_quote[df_quote["tau"] > 0].copy()

# Use all available data from this quote date.
S = float(df_quote["[UNDERLYING_LAST]"].iloc[0])  # Underlying

# Define strike range.
min_strike = 3600
max_strike = 8000

# Define the log–price grid for Fourier inversion.
x_grid_cal = np.linspace(np.log(100), np.log(40000), 30000)

###############################################################################
# Helper function: Filter market data for a given target tau.
###############################################################################
def get_market_data(t_target, tol=0.03):
    df_filt = df_quote[(df_quote["tau"] >= t_target - tol) & (df_quote["tau"] <= t_target + tol)].copy()
    if df_filt.empty:
        idx = np.argmin(np.abs(df_quote["tau"] - t_target))
        df_filt = df_quote.iloc[[idx]]
    grouped = df_filt.groupby("[STRIKE]")["[C_ASK]"].mean().reset_index()
    strikes = grouped["[STRIKE]"].values
    call_prices = grouped["[C_ASK]"].values
    mask = (strikes >= min_strike) & (strikes <= max_strike)
    return strikes[mask], call_prices[mask]

###############################################################################
# 2) MULTI–FACTOR MODEL FUNCTIONS
###############################################################################
def D_func(u, t, alpha, sigma, delta):
    c_factor = 0.5 + 0.5j
    sqrt_delta = cmath.sqrt(delta)
    arg_arctan = (c_factor * alpha * sigma * cmath.sqrt(u)) / sqrt_delta
    val_arctan = cmath.atan(arg_arctan)
    arg_tan = c_factor * sqrt_delta * sigma * t * cmath.sqrt(u) + val_arctan
    return cmath.tan(arg_tan)

def F_func(u, t, alpha, sigma, gamma, delta):
    if u == 0:
        return 0
    special_root = (-1/np.sqrt(2) + 1j/np.sqrt(2)) * cmath.sqrt(u)
    tan_factor = cmath.tan(0.5 * special_root * t * alpha * sigma**2 - cmath.atan(special_root))
    term1 = 1j * gamma * (cmath.sqrt(u)*alpha + (-1/np.sqrt(2) + 1j/np.sqrt(2)) * gamma) * cmath.log(1j - tan_factor)
    term2 = gamma * (1j * cmath.sqrt(u)*alpha + (1/np.sqrt(2) + 1j/np.sqrt(2)) * gamma) * cmath.log(1j + tan_factor)
    term3 = -cmath.sqrt(u)*alpha * (t * (u*alpha**2 + 1j*gamma**2)*sigma**2 + 2*1j*gamma*cmath.log((1/np.sqrt(2) + 1j/np.sqrt(2))*gamma + cmath.sqrt(u)*alpha*tan_factor))
    return (term1 + term2 + term3) / (cmath.sqrt(u)*(u*alpha**3 + 1j*alpha*gamma**2)*sigma**2)

def Fdiff(u, t, alpha, sigma, gamma, delta):
    return F_func(u, t, alpha, sigma, gamma, delta) - F_func(u, 0, alpha, sigma, gamma, delta)

def B_func(u, t, alpha, sigma, delta):
    special_root = (-1/np.sqrt(2) + 1j/np.sqrt(2)) * cmath.sqrt(u)
    B = -1j*u*alpha + special_root * alpha * cmath.tan(0.5*special_root * t * sigma**2 * alpha - cmath.atan(special_root))
    return B

def A_func(u, t, beta, lam, alpha, sigma, gamma, delta):
    first_term = 1j * u * beta * alpha * t
    second_term = lam * Fdiff(u, t, alpha, sigma, gamma, delta)
    return first_term + second_term

def char_func(u, t, X, params):
    exponential_factor = 1j * u * X
    for sigma, lam, gamma, alpha, Y0 in params:
        beta  = lam/(alpha - gamma)
        delta = -0.5 * alpha * sigma**2
        Aval = A_func(u, t, beta, lam, alpha, sigma, gamma, delta)
        Bval = B_func(u, t, alpha, sigma, delta)
        exponential_factor += Aval + Bval * Y0
    return cmath.exp(exponential_factor)

###############################################################################
# Example usage / test
###############################################################################
def density_from_cf(x_vals, t, X, params, u_max=90, N=10000):
    """
    Computes the density f(x) by inverting the characteristic function:
      f(x) = 1/(2π) ∫_{-u_max}^{u_max} e^{-i u x} φ(u;t,X,Y) du.
    """
    u_grid = np.linspace(0, u_max, 1000)
    char_func_values = np.array([char_func(u, t, X, params) for u in u_grid], dtype=complex)
    fs = []
    for x in x_vals:
        integrand = np.exp(-1j * u_grid * x) * char_func_values
        integral = np.trapz(integrand, u_grid)
        fs.append(integral.real / np.pi)
    # Diagnostic martingale check:
    print("Diagnostic (char_func(-0.9999999j)): ", char_func(-0.9999999j, t, X, params))
    return fs

def price_call(K, r, T, x_vals, distribution):
    prices = np.exp(x_vals + r*T)
    payoffs = np.maximum(prices - K, 0)
    call_price = np.exp(-r*T) * np.trapz(payoffs * distribution, x_vals)
    return call_price

def bs_call_price(S, K, T, sigma, r=0.0):
    if sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

def implied_vol(target_price, S, K, T, r=0.0):
    def diff(sigma):
        return bs_call_price(S, K, T, sigma, r) - target_price
    try:
        iv = brentq(diff, 1e-6, 5.0)
    except ValueError:
        iv = np.nan
    return iv

###############################################################################
# Global Settings for Fourier Inversion
###############################################################################
t_default = 1.0  # Default maturity (τ)
X = np.log(S)
u_min, u_max, n_u = 0, 100, 1000
u_grid = np.linspace(u_min, u_max, n_u)

###############################################################################
# 4) INTERACTIVE PLOT WITH SLIDERS FOR TWO FACTORS
###############################################################################
# Factor 1 initial parameters: (sigma, lambda, gamma, alpha, Y0)
init_params_f1 = [0.1, 0.4, 4.0, 0.1, 2.0]
# Factor 2 initial parameters: (sigma, lambda, gamma, alpha, Y0)
init_params_f2 = [0.1, 0.4, 4.0, 0.1, 2.0]

# Global tau slider.
ax_tau = plt.axes([0.25, 0.93, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider_tau = Slider(ax_tau, r'τ', 0.2, 2, valinit=t_default)

# Factor 1 sliders.
ax_f1_sigma  = plt.axes([0.25, 0.87, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_f1_lambda = plt.axes([0.25, 0.82, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_f1_gamma  = plt.axes([0.25, 0.77, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_f1_alpha  = plt.axes([0.25, 0.72, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_f1_Y0     = plt.axes([0.25, 0.67, 0.65, 0.03], facecolor='lightgoldenrodyellow')

slider_f1_sigma  = Slider(ax_f1_sigma,  "F1 σ",   0.01, 0.5,  valinit=init_params_f1[0])
slider_f1_lambda = Slider(ax_f1_lambda, "F1 λ",   0.01, 3.0,  valinit=init_params_f1[1])
slider_f1_gamma  = Slider(ax_f1_gamma,  "F1 γ",   0.1,  50.0, valinit=init_params_f1[2])
slider_f1_alpha  = Slider(ax_f1_alpha,  "F1 α",  -10,    10,   valinit=init_params_f1[3])
slider_f1_Y0     = Slider(ax_f1_Y0,     "F1 Y0",  0.0,  5.0,  valinit=init_params_f1[4])

# Factor 2 sliders.
ax_f2_sigma  = plt.axes([0.25, 0.61, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_f2_lambda = plt.axes([0.25, 0.56, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_f2_gamma  = plt.axes([0.25, 0.51, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_f2_alpha  = plt.axes([0.25, 0.46, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_f2_Y0     = plt.axes([0.25, 0.41, 0.65, 0.03], facecolor='lightgoldenrodyellow')

slider_f2_sigma  = Slider(ax_f2_sigma,  "F2 σ",   0.01, 0.5,  valinit=init_params_f2[0])
slider_f2_lambda = Slider(ax_f2_lambda, "F2 λ",   0.01, 3.0,  valinit=init_params_f2[1])
slider_f2_gamma  = Slider(ax_f2_gamma,  "F2 γ",   0.1,  50.0, valinit=init_params_f2[2])
slider_f2_alpha  = Slider(ax_f2_alpha,  "F2 α",  -10,    10,   valinit=init_params_f2[3])
slider_f2_Y0     = Slider(ax_f2_Y0,     "F2 Y0",  0.0,  5.0,  valinit=init_params_f2[4])

###############################################################################
# 5) INITIAL PLOTTING
###############################################################################
# Get initial market data for tau = t_default.
market_strikes, market_call_prices = get_market_data(t_default, tol=0.03)
market_iv = np.array([implied_vol(c, S, K, t_default) for c, K in zip(market_call_prices, market_strikes)])

def get_factors():
    f1 = (slider_f1_sigma.val, slider_f1_lambda.val, slider_f1_gamma.val,
          slider_f1_alpha.val, slider_f1_Y0.val)
    f2 = (slider_f2_sigma.val, slider_f2_lambda.val, slider_f2_gamma.val,
          slider_f2_alpha.val, slider_f2_Y0.val)
    return [f1, f2]

r = 0
factors_list = get_factors()
f_vals = density_from_cf(x_grid_cal, t_default, X, factors_list)
model_prices = np.array([price_call(K, r, t_default, x_grid_cal, f_vals) for K in market_strikes])
model_iv = np.array([implied_vol(c, S, K, t_default) for c, K in zip(model_prices, market_strikes)])
log_density = np.array(f_vals)

# Create the main figure with three subplots.
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
plt.subplots_adjust(left=0.25, bottom=0.20)

l_market_cp, = ax1.plot(market_strikes, market_call_prices, 'bo', label="Market Call Prices")
l_model_cp,  = ax1.plot(market_strikes, model_prices, 'r--', lw=2, label="Model Call Prices")
ax1.set_ylabel("Call Price")
ax1.set_title(f"Call Prices (τ = {t_default:.2f} yrs, S = {S:.2f})")
ax1.legend()
ax1.grid(True)

l_market_iv, = ax2.plot(market_strikes, market_iv, 'bo', label="Market IV")
l_model_iv,  = ax2.plot(market_strikes, model_iv, 'r--', lw=2, label="Model IV")
ax2.set_xlabel("Strike")
ax2.set_ylabel("Implied Volatility")
ax2.set_title("Implied Volatility Comparison")
ax2.legend()
ax2.grid(True)

l_log_density, = ax3.plot(x_grid_cal, log_density, 'g-', lw=2, label="Log Density")
ax3.set_xlabel("Log-Price")
ax3.set_ylabel("Log Density")
ax3.set_title("Log Density Plot")
ax3.legend()
ax3.grid(True)

###############################################################################
# 6) UPDATE FUNCTION
###############################################################################
def update(val):
    tau_val = slider_tau.val
    new_market_strikes, new_market_call_prices = get_market_data(tau_val, tol=0.03)
    new_market_iv = np.array([implied_vol(c, S, K, tau_val) for c, K in zip(new_market_call_prices, new_market_strikes)])
    factors = get_factors()
    f_vals = density_from_cf(x_grid_cal, tau_val, X, factors)
    new_model_prices = np.array([price_call(K, r, tau_val, x_grid_cal, f_vals) for K in new_market_strikes])
    new_model_iv = np.array([implied_vol(c, S, K, tau_val) for c, K in zip(new_model_prices, new_market_strikes)])

    l_market_cp.set_data(new_market_strikes, new_market_call_prices)
    l_market_iv.set_data(new_market_strikes, new_market_iv)
    l_model_cp.set_data(new_market_strikes, new_model_prices)
    l_model_iv.set_data(new_market_strikes, new_model_iv)
    
    new_log_density = np.array(f_vals)
    l_log_density.set_data(x_grid_cal, new_log_density)
    
    ax1.relim(); ax1.autoscale_view()
    ax2.relim(); ax2.autoscale_view()
    ax3.relim(); ax3.autoscale_view()
    ax1.set_title(f"Call Prices (τ = {tau_val:.2f} yrs, S = {S:.2f})")
    fig.canvas.draw_idle()

# Register update callbacks.
slider_tau.on_changed(update)
for slider in [slider_f1_sigma, slider_f1_lambda, slider_f1_gamma, slider_f1_alpha, slider_f1_Y0,
               slider_f2_sigma, slider_f2_lambda, slider_f2_gamma, slider_f2_alpha, slider_f2_Y0]:
    slider.on_changed(update)

plt.show()
