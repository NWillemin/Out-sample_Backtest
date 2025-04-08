import yfinance as yf
import pandas as pd
import numpy as np
import scipy
from scipy.optimize import minimize
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from sklearn.covariance import LedoitWolf
import matplotlib.pyplot as plt

def compute_metrics(returns, rf):
    
    annual_factor = len(returns)*365.25/(returns.index[-1]-returns.index[0]).days

    avg_returns = returns.mean() * annual_factor
    
    vol = returns.std() * np.sqrt(annual_factor)
    sharpe_ratio = (avg_returns - rf) / vol
    downside_ret = returns[returns < 0]
    downside_dev = np.sqrt(np.sum(downside_ret**2)/len(returns)) * np.sqrt(annual_factor)
    sortino = (avg_returns - rf) / downside_dev

    omega = returns[returns > 0].sum() / -returns[returns < 0].sum()

    cumulative = (1 + returns).cumprod()
    max_DD = (cumulative / cumulative.expanding().max() - 1).min()

    # Historical VaR (Annualized)
    var_historical = returns.quantile(0.05)

    # Conditional VaR (CVaR) - Expected Shortfall (Annualized)
    cvar = returns[returns <= var_historical].mean()

    return avg_returns, vol, sharpe_ratio, sortino, omega, max_DD, cvar

def plot_cumulative_returns(portfolio_value):
    fig, ax = plt.subplots()
    portfolio_value.plot(ax=ax, title="Cumulative Portfolio Return")
    ax.set_ylabel("Portfolio Value")
    ax.set_xlabel("Date")
    ax.grid(True)
    return fig

def plot_weight_evolution(weights_pct):
    fig, ax = plt.subplots(figsize=(10, 5))
    weights_pct.plot.area(ax=ax, stacked=True)
    ax.set_title("Portfolio Weights Over Time")
    ax.set_ylabel("Weight")
    ax.set_xlabel("Date")
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle='--', alpha=0.5)
    return fig

def plot_average_weights(real_weights: pd.DataFrame):
    avg_weights = real_weights.mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    avg_weights.plot(kind='bar', ax=ax, title="Average Portfolio Weights Over Period")
    ax.set_ylabel("Average Weight")
    ax.set_xlabel("Asset")
    ax.set_ylim(0, 1)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    return fig
