# 📊 Customizable Portfolio Backtesting App

This is a fully interactive Streamlit web application that allows users to run out-of-sample backtests on portfolios using different optimization techniques, asset classes, and constraints. It includes support for:

- ✅ Custom assets, exchanges, and asset classes
- 📈 Interval selection (`1d`, `1wk`)
- 🔄 NaN handling logic (forward-fill or drop)
- 🔢 Arithmetic or logarithmic return types
- 🧠 Optimization using:
  - Sharpe Ratio (classic & expected)
  - Sortino Ratio
  - Markowitz Utility (mean-variance & expected)
- 🧠 Return shrinkage: Black-Litterman, Bayes-Stein, James-Stein
- 📉 Covariance shrinkage (Ledoit-Wolf)
- 🔐 Login system with saved backtest history per user

---

## 🚀 Try the App

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://out-samplebacktest-y2stywv8ct4quep8qba45t.streamlit.app/)

---

## 📥 Installation (Local Use)

```bash
git clone https://github.com/NWillemin/Out-sample_Backtest.git
cd Out-sample_Backtest
pip install -r requirements.txt
streamlit run your_script.py
