import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy
from scipy.optimize import minimize
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from sklearn.covariance import LedoitWolf
import matplotlib.pyplot as plt
import os
import pickle
from performance_analysis import compute_metrics, plot_cumulative_returns, plot_weight_evolution, plot_average_weights
from Backtest_calc import run_backtest
# Parameters
st.title("Customizable Backtesting Configuration")

if "user" not in st.session_state:
    st.subheader("🔐 Login")
    st.info("You don’t need to create an account — just enter any username. "
            "If it’s your first time, a profile will be created automatically.")
    username = st.text_input("Username")
    if st.button("Login"):
        if username:
            st.session_state["user"] = username
            st.success(f"Welcome, {username}! Please click a second time to log in")
        else:
            st.error("Please enter a username.")
                     
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["Assets", "Settings", "Objectives", "Results"])
with tab1:
    st.header("1. Asset Configuration")
    st.markdown(
        ":information_source: **Note**: Tickers must be valid symbols from [Yahoo Finance](https://finance.yahoo.com), "
        "e.g., `AAPL`, `GOOGL`, `SPY`, or `BTC-USD`."
    )
    num_assets = st.number_input("Number of assets", min_value=1, step=1, value=1)
    num_exchanges = st.number_input("Number of exchanges", min_value=1, step=1, value=1, help="Only useful if some of your assets have differing trading days and you want to customize how the optimizer treats those days")
    suggested_classes = ["Stocks", "Bonds", "Crypto", "Commodities", "Real Estate", "Cash", "Volatility"]
    selected_suggested = st.multiselect(
        "Select asset classes to use:",
        options=suggested_classes,
        default=["Stocks", "Bonds"],
        help="Pick from common classes"
    )
    
    # 2. Let users type in custom asset classes
    custom_classes_input = st.text_input(
        "Add custom asset classes (comma-separated)",
        placeholder="e.g. AI, Green_Energy, Healthcare",
        help="Only useful if you want to set asset class constraints later"
    )
    
    # 3. Combine both into a list
    custom_classes = [cls.strip() for cls in custom_classes_input.split(",") if cls.strip()]
    asset_class_labels = selected_suggested + custom_classes
    tickers = []
    exchanges = []
    asset_classes = []
    equal_weight = np.round(1/num_assets, 2)
    col1, col2, col3 = st.columns(3)
    for i in range(num_assets):
        with col1:
            ticker = st.text_input(f"Ticker {i+1}", key=f"ticker_{i}")
        with col2:
            exchange = st.selectbox(
                f"Exchange for asset {i+1}",
                options=[f"Exchange {j+1}" for j in range(num_exchanges)],
                key=f"exchange_{i}"
                )   
        with col3:
            asset_class = st.selectbox(
                f"Asset class for asset {i+1}",
                options=asset_class_labels,
                key=f"class_{i}"
            )
        tickers.append(ticker)
        exchanges.append(exchange)
        asset_classes.append(asset_class)
    sorted_indices = sorted(range(len(tickers)), key=lambda i: tickers[i])
    tickers = [tickers[i] for i in sorted_indices]
    exchanges = [exchanges[i] for i in sorted_indices]
    asset_classes = [asset_classes[i] for i in sorted_indices]
with tab2:
    st.header("2. Interval and NaN Handling")
    interval = st.selectbox("Data interval", ["1d", "1wk"])
    ffill_exchanges = []
    if interval=='1d':
        na_handling_method = st.selectbox(
            "NaN handling method", 
            ["dropna", "ffill"], 
            help=(
                "Only relevant if assets have differing trading days (e.g., from different exchanges). "
                "`dropna` removes any day where at least one asset has missing data. "
                "`ffill` fills forward missing prices — you can then specify which exchanges define 'valid' trading days."
            )
        )

        if na_handling_method == "ffill":
            st.markdown("#### Select exchanges to retain when forward-filling:")
            unique_exchanges = list(set(exchanges))
            ffill_exchanges = st.multiselect(
                "Exchanges used to define trading days for forward-filling:",
                unique_exchanges,
                help=(
                    "Forward-fill will only be applied on days where these selected exchanges were active. "
                    "Useful if some assets (e.g., crypto) trade every day, while others don’t."
                )
            )
    else:
        na_handling_method = None
    st.header("3. Returns and Rolling Window")
    return_type = st.selectbox("Return type", ["arithmetic", "logarithmic"])
    lookback_unit = st.selectbox("Lookback unit", ["days", "weeks"])
    lookback_period = st.number_input("Lookback period", min_value=1, value=50)

    st.header("4. Rebalancing")
    rebalancing_unit = st.selectbox("Rebalancing unit", ["days", "weeks"])
    rebalancing_freq = st.number_input("Rebalancing frequency", min_value=1, value=4)
    start_date = st.date_input("Backtest start date", value=datetime.today() - timedelta(days=365))
    end_date = st.date_input("Backtest end date", value=datetime.today())
with tab3:
    st.header("5. Constraints")
    min_weights = {}
    max_weights = {}
    asset_class_constraints = {}
    for i, ticker in enumerate(tickers):
        min_w = st.number_input(f"Minimum weight for {ticker}", min_value=0.0, max_value=1.0, value=0.0, step=0.01, key=f"min_weight_{i}")
        max_w = st.number_input(f"Maximum weight for {ticker}", min_value=0.0, max_value=1.0, value=1.0, step=0.01, key=f"max_weight_{i}")
        min_weights[ticker] = min_w
        max_weights[ticker] = max_w

    for ac in sorted(set(asset_classes)):
        st.subheader(f"Asset Class Constraints - {ac}")
        min_ac = st.number_input(f"Minimum weight for asset class {ac}", min_value=0.0, max_value=1.0, value=0.0, step=0.01, key=f"min_class_{ac}")
        max_ac = st.number_input(f"Maximum weight for asset class {ac}", min_value=0.0, max_value=1.0, value=1.0, step=0.01, key=f"max_class_{ac}")
        asset_class_constraints[ac] = (min_ac, max_ac)

    st.header("6. Initial Weights for the Optimizer")
    st.markdown(
        """
        Define the **initial portfolio weights** that will be used by the optimizer at the **first rebalancing date**.
        These weights act as an initial guess (`x0`) for the optimization process.
        We recommend to use weights that meet the constraints you entered above.
        """
    )
    starting_weights = []
    for i, ticker in enumerate(tickers):
        initial_weight = st.number_input(f"Initial weight for {ticker}", min_value=0.0, max_value=1.0, value=equal_weight, step=0.01, key=f"initial_weight_{i}")
        starting_weights.append(initial_weight)
    starting_weights = np.array(starting_weights)
    stable_starting_weights = st.checkbox(
        "Use these initial weights at every rebalancing",
        value=False,
        help=(
            "✅ Checked: The optimizer will always use the same initial weights you entered above.\n\n"
            "❌ Unchecked: At each rebalancing after the first, the optimizer will use the portfolio's current "
            "weights (from the previous rebalance) as the new initial guess for optimization."
        )
    )

    st.header("7. Return Shrinkage Method")
    shrinkage_dict = {'None': 'none', 'Black-Litterman': 'black-litterman', 'Bayes-Stein': 'bayes-stein', 'James-Stein': 'james-stein'}
    display_shrinkage_method = st.selectbox("Return shrinkage method", list(shrinkage_dict.keys()))
    shrinkage_method = shrinkage_dict[display_shrinkage_method]
    if shrinkage_method in ['black-litterman', 'bayes-stein', 'james-stein']:
        use_ledoit = st.checkbox("Use Ledoit-Wolf covariance matrix to shrink returns")
    else:
        use_ledoit=None
    if shrinkage_method == "black-litterman":
        bl_risk_aversion = st.number_input("Risk aversion (Black-Litterman)", min_value=0.0, value=2.5)
        tau = st.number_input("Tau (Black-Litterman)", min_value=0.0, value=0.05)
        ref_weights = st.checkbox("Use custom market weights for market-implied excess returns")
        if ref_weights==False:
            market_weights = np.ones(num_assets)/num_assets
        else:
            market_weights = []
            for i in range(num_assets):
                market_weight = st.number_input(f"Market weight for {tickers[i]}", min_value=0.0, max_value=1.0, value=equal_weight, step=0.01, key=f"market_weight_{i}")
                market_weights.append(market_weight)
            market_weights = np.array(market_weights)
    st.header("8. Objective Function")
    if return_type == 'logarithmic':
        objectives_dict = {'Expected Sharpe Ratio': 'exp_sharpe', 'Classic Sharpe Ratio': 'sharpe', 'Sortino Ratio': 'sortino', 'Expected Markowitz Utility': 'exp_mean-variance', 'Markowitz Utility': 'mean-variance'}
    else:
        objectives_dict = {'Classic Sharpe Ratio': 'sharpe', 'Sortino Ratio': 'sortino', 'Markowitz Utility': 'mean-variance'}
    display_objective = st.selectbox("Objective function", list(objectives_dict.keys()))
    objective = objectives_dict[display_objective]

    if objective != "sortino":
        if shrinkage_method in ["none", "james-stein"]:
            cov_shrinkage_dict = {'None': 'none', 'Ledoit-Wolf': 'ledoit'}
            display_cov_matrix_type = st.selectbox("Covariance shrinkage method", list(cov_shrinkage_dict.keys()))
        else:
            cov_shrinkage_dict = {'None': 'none', 'Ledoit-Wolf': 'ledoit', display_shrinkage_method: 'shrinkage'}
            display_cov_matrix_type = st.selectbox("Covariance shrinkage method", list(cov_shrinkage_dict.keys()))
        
        cov_matrix_type = cov_shrinkage_dict[display_cov_matrix_type]
        
    if objective in ['exp_mean-variance', 'mean-variance']:
        mv_risk_aversion = st.number_input("Risk aversion (Markowitz Utility)", min_value=0.0, value=3.0)
with tab4:
    st.header("9. Performance Evaluation")
    return_type_perf = st.selectbox("Return type used for the computation of the portfolio's performance metrics", ["arithmetic", "logarithmic"])
    rf = st.number_input("Annual risk-free rate", min_value=0.0, value=0.01)
    ew_benchmark = st.checkbox("Use the equally-weighted portfolio as benchmark", value=True)
    if ew_benchmark == False:
        benchmark_weights = []
        for i in range(num_assets):
            benchmark_weight = st.number_input(f"Market weight for {tickers[i]}", min_value=0.0, max_value=1.0, value=equal_weight, step=0.01, key=f"bench_weight_{i}")
            benchmark_weights.append(benchmark_weight)
        benchmark_weights = np.array(benchmark_weights)
    else:
        benchmark_weights = np.ones(num_assets)/num_assets
    initial_value = st.number_input("Initial investment", min_value=0, value=1000)
    if "Crypto" in asset_classes:
        keep_all_days = st.checkbox("Keep the returns of Saturdays and Sundays (when only your crypto is trading) for the computation of the performance metrics")
    else:
        keep_all_days = True
    st.header("10. Results")
    if st.button('Run Backtest'):
        config = {
            "tickers": tickers,
            "exchanges": exchanges,
            "asset_classes": asset_classes,
            "interval": interval,
            "na_method": na_handling_method,
            "ffill_exchanges": ffill_exchanges,
            "return_type": return_type,
            "lookback": (lookback_unit, lookback_period),
            "rebalancing": (rebalancing_unit, rebalancing_freq),
            "start_date": start_date,
            "end_date": end_date,
            "min_weights": min_weights,
            "max_weights": max_weights,
            "asset_class_constraints": asset_class_constraints,
            "starting_weights": starting_weights,
            "stable_starting_weights": stable_starting_weights,
            "use_ledoit": use_ledoit,
            "shrinkage_method": shrinkage_method,
            "initial_value": initial_value,
            "black_litterman_params": {
                "risk_aversion": bl_risk_aversion if shrinkage_method == "black-litterman" else None,
                "tau": tau if shrinkage_method == "black-litterman" else None,
                "market_weights": market_weights if shrinkage_method == "black-litterman" else None,
            },
            "objective": objective,
            "cov_matrix_type": cov_matrix_type if objective != 'sortino' else None,
            "mean_variance_risk_aversion": mv_risk_aversion if objective in ['exp_mean-variance', 'mean-variance'] else None,
            "benchmark": benchmark_weights,
            "keep_all_days": keep_all_days,
        }
        portfolio_holdings, real_weights, portfolio_value, portfolio_holdings_bench, real_weights_bench, portfolio_value_bench = run_backtest(config)
        portfolio_value = portfolio_value.astype(float)
        portfolio_value_bench = portfolio_value_bench.astype(float)
        
        st.session_state["backtest_result"] = {
        "portfolio_value": portfolio_value,
        "real_weights": real_weights,
        "portfolio_value_bench": portfolio_value_bench,
        "real_weights_bench": real_weights_bench,
        "config": config,
        }
    if "backtest_result" in st.session_state:
        result = st.session_state["backtest_result"]
        
        portfolio_value = result["portfolio_value"].astype(float)
        portfolio_value_bench = result["portfolio_value_bench"].astype(float)
        real_weights = result["real_weights"]
        real_weights_bench = result["real_weights_bench"]
        config = result["config"]
    
        if return_type_perf == 'arithmetic':
            returns = portfolio_value.pct_change().dropna()
            returns_bench = portfolio_value_bench.pct_change().dropna()
        else:
            returns = np.log(portfolio_value / portfolio_value.shift(1)).dropna()
            returns_bench = np.log(portfolio_value_bench / portfolio_value_bench.shift(1)).dropna()
    
        # Metrics
        avg_returns, vol, sharpe, sortino, omega, max_dd, cvar = compute_metrics(returns, rf)
        bench_avg, bench_vol, bench_sharpe, bench_sortino, bench_omega, bench_max_dd, bench_cvar = compute_metrics(returns_bench, rf)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📈 Portfolio Performance")
            st.write(f"Average Return: {avg_returns:.2%}")
            st.write(f"Volatility: {vol:.2%}")
            st.write(f"Sharpe Ratio: {sharpe:.2f}")
            st.write(f"Sortino Ratio: {sortino:.2f}")
            st.write(f"Omega Ratio: {omega:.2f}")
            st.write(f"Max Drawdown: {max_dd:.2%}")
            st.write(f"CVaR: {cvar:.2%}")
        
        with col2:
            st.subheader("🎯 Benchmark Performance")
            st.write(f"Average Return: {bench_avg:.2%}")
            st.write(f"Volatility: {bench_vol:.2%}")
            st.write(f"Sharpe Ratio: {bench_sharpe:.2f}")
            st.write(f"Sortino Ratio: {bench_sortino:.2f}")
            st.write(f"Omega Ratio: {bench_omega:.2f}")
            st.write(f"Max Drawdown: {bench_max_dd:.2%}")
            st.write(f"CVaR: {bench_cvar:.2%}")
    
        st.subheader("Performance Chart")
        fig = plot_cumulative_returns(portfolio_value, portfolio_value_bench)
        st.pyplot(fig)
    
        st.subheader("Weight Allocation Over Time")
        tab_weights_port, tab_weights_bench = st.tabs(["Portfolio", "Benchmark"])
        
        with tab_weights_port:
            st.markdown("### Portfolio Weights Over Time")
            fig1 = plot_weight_evolution(real_weights)
            st.pyplot(fig1)
        
        with tab_weights_bench:
            st.markdown("### Benchmark Weights Over Time")
            fig1_bench = plot_weight_evolution(real_weights_bench)
            st.pyplot(fig1_bench)
    
        st.subheader("Average Weight of each Asset")
        fig2 = plot_average_weights(real_weights, real_weights_bench)
        st.pyplot(fig2)
        # Save results to file for current user
        user = st.session_state.get("user")
        if user:
            st.markdown("---")
            st.subheader("💾 Save This Backtest")
        
            save_result = st.checkbox("Save this backtest?", value=False)
        
            if save_result:
                custom_name = st.text_input("Enter a name for this backtest", placeholder="e.g. black_litterman_sharpe_ratio")
                if custom_name.strip() == "":
                    st.warning("⚠️ Please enter a valid name.")
                elif st.button("💾 Confirm and Save"):
                    summary_name = f"{custom_name.strip().replace(' ', '_')}.pkl"
                    os.makedirs(f"results/{user}", exist_ok=True)
                    filename = f"results/{user}/{summary_name}"
            
                    with open(filename, "wb") as f:
                        pickle.dump({
                            "config": config,
                            "portfolio_value": portfolio_value,
                            "portfolio_value_bench": portfolio_value_bench,
                            "real_weights": real_weights,
                            "metrics": (avg_returns, vol, sharpe, sortino, omega, max_dd, cvar),
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        }, f)
                        st.success(f"✅ Backtest saved as: `{summary_name}`")
    if "user" in st.session_state:
        user = st.session_state["user"]
        user_folder = f"results/{user}"
    
        if os.path.exists(user_folder):
            st.subheader("📂 View Past Backtests")
            files = sorted(os.listdir(user_folder), reverse=True)
    
            selected_file = st.selectbox("Select a past backtest to view:", files)
    
            if selected_file:
                with open(f"{user_folder}/{selected_file}", "rb") as f:
                    past = pickle.load(f)
    
                st.markdown(f"**Backtest run at:** {past['timestamp']}")
                metrics_names = ["Avg Return", "Volatility", "Sharpe", "Sortino", "Omega", "Max Drawdown", "CVaR"]
                metrics_values = past["metrics"]
                settings = past['config']
                
                st.markdown("📊 **Performance Metrics**")
                for name, value in zip(metrics_names, metrics_values):
                    st.write(f"{name}: {value:.2%}" if "Return" in name or "Volatility" in name or "CVar" in name or "Drawdown" in name else f"{name}: {value:.2f}")
                portfolio_value = past["portfolio_value"]
                portfolio_value_bench = past.get("portfolio_value_bench")  # None if not saved
                
                if portfolio_value_bench is not None:
                    st.pyplot(plot_cumulative_returns(portfolio_value, portfolio_value_bench))
                else:
                    st.pyplot(plot_cumulative_returns(portfolio_value))
                st.markdown("Configuration")
                st.write(settings)



    
