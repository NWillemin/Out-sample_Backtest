import yfinance as yf
import pandas as pd
import numpy as np
import scipy
from scipy.optimize import minimize
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from sklearn.covariance import LedoitWolf
import matplotlib.pyplot as plt
def run_backtest(config):
    tickers = config["tickers"]
    exchanges = config["exchanges"]
    asset_classes = config["asset_classes"]
    interval = config["interval"]
    na_method = config["na_method"]
    ffill_exchanges = config["ffill_exchanges"]
    return_type = config["return_type"]
    lookback_unit, lookback_period = config["lookback"]
    rebalancing_unit, rebalancing_freq = config["rebalancing"]
    start_test = config["start_date"]
    end_test = config["end_date"]
    min_weights = config["min_weights"]
    max_weights = config["max_weights"]
    asset_class_constraints = config["asset_class_constraints"]
    use_ledoit = config["use_ledoit"]
    shrinkage_method = config["shrinkage_method"]
    objective = config["objective"]
    cov_matrix_type = config["cov_matrix_type"]
    mv_risk_aversion = config["mean_variance_risk_aversion"]
    last_weights = config['starting_weights']
    cst_last_weights = config['stable_starting_weights']
    benchmark = config['benchmark']
    initial_value = config['initial_value']
    bl_params = config["black_litterman_params"]
    bl_risk_aversion = bl_params.get("risk_aversion")
    tau = bl_params.get("tau")
    market_weights = bl_params.get("market_weights")
    opens1 = yf.download(tickers=tickers, period='max', interval=interval)["Open"]
    closes1 = yf.download(tickers=tickers, period='max', interval=interval)["Close"]
    valid_index = opens1.dropna().index

    if valid_index.empty:
        st.error("🚨 No data available. Please check your tickers and interval, and ensure all fields are filled in.")
        st.stop()  # Stops the Streamlit script to avoid further errors
    else:
        opens = opens1.loc[valid_index[0]:].copy()
        closes = closes1.loc[valid_index[0]:].copy()
    if interval=='1d':
        opensd = opens.copy()
        closesd = closes.copy()
        if na_method == 'dropna':
            opens=opens.dropna()
            closes=closes.dropna()
        elif len(ffill_exchanges) == len(exchanges):
            opens=opens.ffill()
            closes=closes.ffill()
            opens=opens.dropna()
            closes=closes.dropna()
            
        elif len(ffill_exchanges) < len(exchanges):
            assets_to_ffill = [t for t, e in zip(tickers, exchanges) if e in ffill_exchanges]
            mask = opens[assets_to_ffill].notna().any(axis=1)
            opens = opens.loc[mask]
            closes = closes.loc[mask]

            # Step 3: Apply forward-fill to the full DataFrame
            opens = opens.ffill()
            closes = closes.ffill()
            opens = opens.dropna()
            closes = closes.dropna()
        else:
            raise ValueError(f"The number of exchanges to retain when forward-filling cannot be greater than the total number of exchanges: {len(ffill_exchanges)} > {len(exchanges)}")
    else:
        opens2 = yf.download(tickers=tickers, period='max', interval='1d')["Open"]
        closes2 = yf.download(tickers=tickers, period='max', interval='1d')["Close"]
        opensd = opens2.loc[opens2.dropna().index[0]:].copy()
        closesd = closes2.loc[closes2.dropna().index[0]:].copy()
    if return_type=='arithmetic':
        returns = opens.pct_change().dropna()
    elif return_type=='logarithmic':
        returns = np.log(opens/opens.shift(1)).dropna()
    else:
        raise ValueError(f"Unknown return type: {return_type}")
    
    st_date = pd.Timestamp(start_test)
    et_date = pd.Timestamp(end_test)
    rb_dic = {'weeks': 7, 'days': 1}
    no_rb = (et_date-st_date).days//(rebalancing_freq*rb_dic[rebalancing_unit])
    start_trains = [st_date-relativedelta(**{lookback_unit: lookback_period})+relativedelta(days=1)+relativedelta(**{rebalancing_unit: rebalancing_freq})*x for x in range(no_rb)]
    if start_trains[0] < valid_index[0]:
        st.error(f"🚨 First rebalancing date ({first_rebalancing}) is too early. Please select a backtest start date that is later than {valid_index[0]+relativedelta(**{lookback_unit: lookback_period})} or reduce the lookback.")
        st.stop()
    start_trains = [returns.index[returns.index >= date].min() for date in start_trains]
    rb_dates = [st_date+relativedelta(**{rebalancing_unit: rebalancing_freq})*x for x in range(no_rb)]
    rb_dates = [opensd.dropna().index[opensd.dropna().index >= date].min() for date in rb_dates]
    end_tests = [opensd.ffill().index[opensd.ffill().index<date].max() for date in rb_dates[1:]]
    last_test = end_tests[-1]+relativedelta(**{rebalancing_unit: rebalancing_freq})
    last_test = opensd.ffill().index[opensd.ffill().index>=last_test].min()
    end_tests.append(last_test)

    def objective_sortino(weights, *args):
        avg_returns, returns_matrix, returns_type = args
        port_returns = np.dot(returns_matrix, weights)
        downside_returns = port_returns[port_returns < 0]  # Only take negative returns
        if downside_returns.size == 0:
            return np.inf
        downside_std = np.sqrt(np.sum(downside_returns**2)/len(port_returns))
        # Compute the exponentially weighted expected return from avg_returns
        exp_returns = np.dot(weights, avg_returns)
        if returns_type == 'logarithmic':
            exp_returns = np.exp(exp_returns) - 1
        # Adjust target return (here 0.003) as needed.
        sortino = exp_returns / downside_std
        # Since we want to maximize Sortino ratio, we minimize its negative.
        return -sortino
    
    def objective_sharpe(weights, *args):
        avg_returns, cov_matrix, returns_type = args
        exp_returns = np.dot(weights, avg_returns)
        if returns_type == 'logarithmic':
            exp_returns = np.exp(avg_returns) - 1
        port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        if port_std == 0:
            return np.inf
        return -exp_returns / port_std

    def objective_exp_sharpe(weights, *args):
        avg_returns, cov_matrix = args
        mu_p = np.dot(weights, avg_returns)
        sigma2_p = np.dot(weights.T, np.dot(cov_matrix, weights))
        exp_returns = np.exp(mu_p + 0.5 * sigma2_p) - 1
        port_std = np.sqrt((np.exp(sigma2_p) - 1) * np.exp(2 * mu_p + sigma2_p))
        return -exp_returns/port_std

    def objective_exp_mv(weights, *args):
        avg_returns, cov_matrix, risk_aversion = args
        mu_p = np.dot(weights, avg_returns)
        sigma2_p = np.dot(weights.T, np.dot(cov_matrix, weights))
        exp_returns = np.exp(mu_p + 0.5 * sigma2_p) - 1
        port_var = (np.exp(sigma2_p) - 1) * np.exp(2 * mu_p + sigma2_p)
        return -exp_returns + risk_aversion/2*port_var
    def objective_mv(weights, *args):
        avg_returns, cov_matrix, returns_type, risk_aversion = args
        exp_returns = np.dot(weights, avg_returns)
        if returns_type == 'logarithmic':
            exp_returns = np.exp(exp_returns)-1
        port_var = np.dot(weights.T, np.dot(cov_matrix, weights))
        return -exp_returns + risk_aversion/2*port_var
    bounds = [(min_weights[t], max_weights[t]) for t in tickers]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    for ac, (min_ac, max_ac) in asset_class_constraints.items():
        indices = [i for i, c in enumerate(asset_classes) if c == ac]

        # Lower bound: sum(weights in class) ≥ min_ac
        constraints.append({
            "type": "ineq",
            "fun": lambda w, indices=indices, min_ac=min_ac: np.sum(w[indices]) - min_ac
        })

        # Upper bound: sum(weights in class) ≤ max_ac
        constraints.append({
            "type": "ineq",
            "fun": lambda w, indices=indices, max_ac=max_ac: max_ac - np.sum(w[indices])
        })
    shrunk_returns_dict = {}
    shrunk_cov_dict = {}
    if shrinkage_method=='black-litterman':
        for i, rb_date in enumerate(rb_dates):
            window_returns = returns.loc[start_trains[i]:rb_date].to_numpy()
            cov_matrix = np.cov(window_returns, rowvar=False) if use_ledoit==False else LedoitWolf().fit(window_returns).covariance_
            mean_returns = window_returns.mean(axis=0)
            market_implied_returns = (bl_risk_aversion * cov_matrix @ market_weights).reshape(-1,1)
            num_assets = len(tickers)
            P = np.eye(num_assets)
            Q = mean_returns.reshape(-1, 1)
            omega = np.diag(np.diag(P @ cov_matrix @ P.T))
            M_inverse = np.linalg.inv(np.linalg.inv(tau * cov_matrix) + P.T @ np.linalg.inv(omega) @ P)
            bl_mean_returns = M_inverse @ (np.linalg.inv(tau * cov_matrix) @ market_implied_returns + P.T @ np.linalg.inv(omega) @ Q)
            shrunk_cov_dict[rb_date] = cov_matrix + M_inverse
            shrunk_returns_dict[rb_date] = bl_mean_returns.flatten()
    elif shrinkage_method == 'bayes-stein':
        for i, rb_date in enumerate(rb_dates):
            window_returns = returns.loc[start_trains[i]:rb_date].to_numpy()
            cov_matrix = np.cov(window_returns, rowvar=False) if use_ledoit==False else LedoitWolf().fit(window_returns).covariance_
            mean_returns = window_returns.mean(axis=0)
            T, N = window_returns.shape  # Time periods, number of assets
            # Compute inverse covariance matrix
            inv_cov_matrix = np.linalg.inv(cov_matrix)
            
            # Create vector of ones
            ones = np.ones((N, 1))
            
            # Compute minimum variance portfolio expected return (mu_G)
            mu_G = (ones.T @ inv_cov_matrix @ mean_returns) / (ones.T @ inv_cov_matrix @ ones)
            mu_G = mu_G.item()  # Convert from matrix to scalar
            
            # Compute shrinkage factor g
            diff = mean_returns.reshape(-1, 1) - mu_G * ones  # (N,1)
            g_numerator = N + 2
            g_denominator = (N + 2) + T * (diff.T @ inv_cov_matrix @ diff).item()
            g = g_numerator / g_denominator
            g = np.clip(g, 0.1, 0.6)
            # Compute shrunk mean returns
            shrunk_returns_dict[rb_date] = (1 - g) * mean_returns + g * mu_G * np.ones(N)
            
            # Compute shrinkage adjustment parameter phi
            phi = (N + 2) / (diff.T @ inv_cov_matrix @ diff).item()
            
            # Compute shrunk covariance matrix
            first_term = ((T + phi + 1) / (T + phi)) * cov_matrix
            second_term = (phi / (T * (T + phi + 1))) * ((ones @ ones.T) / (ones.T @ inv_cov_matrix @ ones).item())
            
            shrunk_cov_dict[rb_date] = first_term + second_term
    elif shrinkage_method == "james-stein":
        for i, rb_date in enumerate(rb_dates):
            window_returns = returns.loc[start_trains[i]:rb_date].to_numpy()
            sample_cov = np.cov(window_returns, rowvar=False) if use_ledoit==False else LedoitWolf().fit(window_returns).covariance_
            sample_mean = window_returns.mean(axis=0)
            n, p = window_returns.shape
            inv_cov = np.linalg.pinv(sample_cov)
            ones = np.ones(p)
            mvp_weights = inv_cov @ ones / (ones.T @ inv_cov @ ones)
            target_return = np.dot(mvp_weights, sample_mean)
            # Shrinkage intensity calculation
            sigma_inv = inv_cov
            # James-Stein positive-part adjustment [3][6]
            quadratic_term = (sample_mean - target_return).T @ sigma_inv @ (sample_mean - target_return)
            raw_delta = 1 - (p - 3) / (n * quadratic_term)
            delta = np.clip(raw_delta, 0.1, 0.6)  # Avoid overshrinking via positive-part
            shrunk_returns_dict[rb_date] = (1 - delta) * sample_mean + delta * target_return
    elif shrinkage_method=='none':
        for i, rb_date in enumerate(rb_dates):
            window_returns = returns.loc[start_trains[i]:rb_date].to_numpy()
            shrunk_returns_dict[rb_date] = window_returns.mean(axis=0)
    else:
        raise ValueError(f"Unknown return shrinkage method: {shrinkage_method}")
    weights_dict = {}
    if objective=='sortino':
        for i, rb_date in enumerate(rb_dates):
            window_returns = returns.loc[start_trains[i]:rb_date].to_numpy()
            avg_returns = shrunk_returns_dict[rb_date]
            result = minimize(
                objective_sortino,
                x0=last_weights,
                args=(avg_returns, window_returns, return_type),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            weights_dict[rb_date]=result.x
            if cst_last_weights==False:
                last_weights = result.x
    elif objective in ['exp_sharpe', 'sharpe', 'exp_mean-variance', 'mean-variance']:
        for i, rb_date in enumerate(rb_dates):
            avg_returns = shrunk_returns_dict[rb_date]
            if cov_matrix_type == 'shrinkage':
                cov_matrix = shrunk_cov_dict[rb_date]
            elif cov_matrix_type == 'ledoit':
                window_returns = returns.loc[start_trains[i]:rb_date].to_numpy()
                cov_matrix = LedoitWolf().fit(window_returns).covariance_
            elif cov_matrix_type == 'none':
                window_returns = returns.loc[start_trains[i]:rb_date].to_numpy()
                cov_matrix = np.cov(window_returns, rowvar=False)
            else:
                raise ValueError(f"Unknown covariance matrix type: {cov_matrix_type}")
            if objective=='exp_sharpe':
                result = minimize(
                    objective_exp_sharpe,
                    x0=last_weights,
                    args=(avg_returns, cov_matrix),
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints
                )
            elif objective=='sharpe':
                result = minimize(
                    objective_sharpe,
                    x0=last_weights,
                    args=(avg_returns, cov_matrix, return_type),
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints
                )
            elif objective=='exp_mean-variance':
                result = minimize(
                    objective_exp_mv,
                    x0=last_weights,
                    args=(avg_returns, cov_matrix, mv_risk_aversion),
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints
                )
            else:
                result = minimize(
                    objective_mv,
                    x0=last_weights,
                    args=(avg_returns, cov_matrix, return_type, mv_risk_aversion),
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints
                )
            weights_dict[rb_date] = result.x
            if cst_last_weights==False:
                last_weights = result.x
    else:
        raise ValueError(f"Unknown objective function: {objective}")
    
   
    """ Simulates out-of-sample portfolio performance. """
    opens_perf = opensd.ffill().dropna()
    closes_perf = closesd.ffill().dropna()
    portfolio_holdings = pd.DataFrame(index=rb_dates, columns=closes_perf.columns)
    portfolio_value = pd.Series(index=closes_perf.loc[rb_dates[0]:end_tests[-1]].index, dtype=float)
    real_weights = pd.DataFrame(index=closes_perf.loc[rb_dates[0]:end_tests[-1]].index, columns=closes_perf.columns)
    last_value = float(initial_value)
    last_value_bench = float(initial_value)
    portfolio_holdings_bench = portfolio_holdings.copy()
    portfolio_value_bench = portfolio_value.copy()
    real_weights_bench = real_weights.copy()
    weights_dict_bench = {rb_date: benchmark for rb_date in rb_dates}
    for i, rb_date in enumerate(rb_dates):
        if i > 0:
            last_value = (portfolio_holdings.iloc[i-1] * closes_perf.loc[rb_date]).sum()
            last_value_bench = (portfolio_holdings_bench.iloc[i-1] * closes_perf.loc[rb_date]).sum()

        # Compute asset holdings based on portfolio weights
        temp_assets = (last_value * weights_dict[rb_date]) / closes_perf.loc[rb_date]
        portfolio_holdings.iloc[i] = temp_assets
        # Slice from a to the row right before b

        # Compute portfolio value over test period
        portfolio_value.loc[rb_date:end_tests[i]] = (temp_assets * closes_perf.loc[rb_date:end_tests[i]]).sum(axis=1)
        real_weights.loc[rb_date:end_tests[i]] = closes_perf.loc[rb_date:end_tests[i]].mul(temp_assets, axis=1).div(portfolio_value.loc[rb_date:end_tests[i]], axis=0)
        temp_assets_bench = (last_value_bench * weights_dict_bench[rb_date]) / closes_perf.loc[rb_date]
        portfolio_holdings_bench.iloc[i] = temp_assets_bench
    
        portfolio_value_bench.loc[rb_date:end_tests[i]] = (
            closes_perf.loc[rb_date:end_tests[i]].mul(temp_assets_bench, axis=1).sum(axis=1)
        )
    
        real_weights_bench.loc[rb_date:end_tests[i]] = (
            closes_perf.loc[rb_date:end_tests[i]]
            .mul(temp_assets_bench, axis=1)
            .div(portfolio_value_bench.loc[rb_date:end_tests[i]], axis=0)
        )
    return portfolio_holdings, real_weights, portfolio_value, portfolio_holdings_bench, real_weights_bench, portfolio_value_bench

