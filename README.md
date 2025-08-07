# 🏦 Option Pricing Model Validation Platform

## Advanced Quantitative Finance • Model Risk Management • Regulatory Compliance

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-Educational-green.svg)](LICENSE)

> **Disclaimer**: The use of the name "Commerzbank" is solely for educational purposes. | *Die Verwendung des Namens der Commerzbank dient ausschließlich Bildungszwecken.*

---

## 🎯 Project Overview

This comprehensive **Option Pricing Model Validation Framework** demonstrates advanced quantitative finance techniques through the implementation, calibration, and validation of two prominent option pricing models:

- **Black-Scholes Model** - Classic geometric Brownian motion framework
- **Heston Stochastic Volatility Model** - Advanced stochastic volatility extension

The project showcases industry-standard model risk management practices, regulatory compliance techniques, and professional-grade financial engineering capabilities.

## 🚀 Key Features

### 📊 **Live Market Data Integration**
- **FRED API Integration** - Real-time 10-Year Treasury rates
- **Yahoo Finance API** - Live stock price data
- **Automated Data Validation** - Error handling and fallback mechanisms

### 🔬 **Advanced Mathematical Models**
- **Black-Scholes Implementation** - Complete Greeks calculation
- **Heston Model with FFT** - Efficient Fourier Transform pricing
- **Characteristic Function Approach** - High-performance computation
- **Numba JIT Compilation** - Optimized numerical performance

### ⚖️ **Comprehensive Validation Framework**
- **Cross-Validation Analysis** - 5-fold validation with statistical metrics
- **Out-of-Sample Testing** - Robust generalization assessment
- **Moneyness-Based Analysis** - ITM/ATM/OTM performance evaluation
- **Residual Analysis** - Model diagnostic techniques
- **Time Decay Validation** - Maturity-dependent performance

### 📈 **Professional Visualization**
- **Interactive 2D/3D Plots** - Plotly-based dynamic visualizations
- **Volatility Surface Rendering** - 3D option pricing surfaces
- **Real-time Parameter Exploration** - Interactive model calibration
- **Performance Dashboards** - Comprehensive metrics display

### 🎨 **Enterprise-Grade UI/UX**
- **Corporate Branding** - Professional design system
- **Responsive Layout** - Multi-device compatibility
- **Interactive Navigation** - Streamlined user experience
- **Real-time Updates** - Live data refresh capabilities

## 🏗️ Technical Architecture

### **Core Technologies**
```
├── Frontend Framework: Streamlit 1.28+
├── Numerical Computing: NumPy, SciPy
├── Performance Optimization: Numba JIT
├── Data Visualization: Plotly, Matplotlib
├── Financial Data: yfinance, FRED API
├── Mathematical Libraries: scipy.stats, scipy.optimize
└── Web Framework: Streamlit with custom CSS
```

### **Mathematical Implementation**
- **FFT-based Option Pricing** - O(N log N) complexity
- **MLE + MSE Hybrid Calibration** - Robust parameter estimation
- **Greeks Calculation** - Complete risk sensitivity analysis
- **Monte Carlo Validation** - Statistical model verification

## 📋 Project Structure

```
option-pricing-validation/
│
├── app.py                 # Main Streamlit application
├── instrument.png         # Workflow diagram
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (API keys)
├── README.md             # This file
└── .streamlit/
    └── config.toml       # Streamlit configuration
```

## 🛠️ Installation & Setup

### **Prerequisites**
- Python 3.8+
- FRED API Key (free registration at https://fred.stlouisfed.org/docs/api/)

### **Installation Steps**

1. **Clone the Repository**
```bash
git clone https://github.com/your-username/option-pricing-validation.git
cd option-pricing-validation
```

2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Environment Configuration**
```bash
# Create .env file
echo "FRED_API_KEY=your_fred_api_key_here" > .env
```

5. **Run the Application**
```bash
streamlit run app.py
```

## 📦 Dependencies

```python
streamlit>=1.28.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
plotly>=5.0.0
yfinance>=0.2.0
scipy>=1.7.0
numba>=0.56.0
requests>=2.25.0
python-dotenv>=0.19.0
```

## 🔧 Core Functionality

### **1. Black-Scholes Model**
```python
def european_BS(t, St, K, T, r, sigma, call):
    """
    European option pricing using Black-Scholes formula
    
    Parameters:
    - t: Current time
    - St: Current stock price
    - K: Strike price
    - T: Time to expiry
    - r: Risk-free rate
    - sigma: Volatility
    - call: 1 for call, 0 for put
    """
```

### **2. Heston Model**
```python
def european_Heston_FFT_fast(S0, K_array, T, r, gam0, kappa, lamb, sig_tilde, rho):
    """
    Vectorized Heston call pricing via FFT (Carr-Madan)
    
    Parameters:
    - S0: Initial stock price
    - K_array: Array of strike prices
    - gam0: Initial variance
    - kappa: Mean reversion speed
    - lamb: Risk premium parameter
    - sig_tilde: Volatility of volatility
    - rho: Correlation coefficient
    """
```

### **3. Model Calibration**
```python
def calibrate_heston(market_prices, strikes, S0, T, r):
    """
    Hybrid MLE + MSE calibration for Heston parameters
    
    Returns:
    - Optimal parameters: [v0, kappa, lambda, sigma_v, rho]
    - Negative log-likelihood value
    """
```

## 📊 Performance Metrics

The application provides comprehensive validation metrics:

| Metric | Formula | Purpose |
|--------|---------|---------|
| **RMSE** | √(Σ(observed - predicted)²/n) | Overall prediction accuracy |
| **MAE** | Σ\|observed - predicted\|/n | Average absolute error |
| **R²** | 1 - SS_res/SS_tot | Coefficient of determination |
| **MAPE** | (Σ\|observed - predicted\|/observed)/n × 100 | Percentage error |

## 🎯 Use Cases

### **Academic Applications**
- **Quantitative Finance Education** - Teaching advanced option pricing
- **Model Risk Management** - Demonstrating validation techniques
- **Financial Engineering** - Showcasing mathematical modeling

### **Professional Applications**
- **Risk Management** - Model validation frameworks
- **Trading Desk Tools** - Option pricing and Greeks calculation
- **Regulatory Compliance** - Basel III/FRTB validation standards

## 📈 Key Results & Insights

### **Model Performance Comparison**
- **Black-Scholes**: Superior generalization, stable cross-validation
- **Heston Model**: Better volatility smile capture, higher complexity
- **Validation Metrics**: Comprehensive out-of-sample testing results

### **Risk Management Insights**
- **VaR Calculation**: 1-day Value at Risk estimation
- **Expected Shortfall**: Tail risk assessment
- **Sensitivity Analysis**: Greeks-based risk decomposition

## 🔮 Future Enhancements

### **Model Extensions**
- [ ] **Jump-Diffusion Models** (Merton, Kou)
- [ ] **Local Volatility Models** (Dupire)
- [ ] **SABR Model Implementation**
- [ ] **Machine Learning Integration** (Neural Networks, Random Forests)

### **Technical Improvements**
- [ ] **Real-time Data Streaming** (WebSocket integration)
- [ ] **Portfolio-Level Analytics** (Multi-asset pricing)
- [ ] **Advanced Calibration** (Genetic algorithms, particle swarms)
- [ ] **Cloud Deployment** (AWS/Azure integration)

### **Regulatory Features**
- [ ] **Basel III Compliance** (FRTB standards)
- [ ] **Model Documentation** (Automated reporting)
- [ ] **Audit Trail** (Change tracking and versioning)

## 🏆 Professional Competencies Demonstrated

### **Quantitative Finance**
✅ **Stochastic Calculus** - SDE modeling and solution techniques  
✅ **Numerical Methods** - FFT, Monte Carlo, finite differences  
✅ **Risk Management** - VaR, ES, Greeks, stress testing  
✅ **Model Validation** - Cross-validation, backtesting, benchmarking  

### **Technical Skills**
✅ **Python Programming** - Object-oriented and functional programming  
✅ **Mathematical Libraries** - NumPy, SciPy, optimization algorithms  
✅ **Data Visualization** - Interactive dashboards, 3D plotting  
✅ **Web Development** - Streamlit, HTML/CSS, responsive design  

### **Financial Engineering**
✅ **Option Pricing Theory** - Black-Scholes, stochastic volatility  
✅ **Market Data Integration** - APIs, data validation, error handling  
✅ **Performance Optimization** - JIT compilation, vectorization  
✅ **Professional Documentation** - Code documentation, user guides  

## 📞 Contact & Contributing

### **Author**
**Your Name** - Quantitative Finance Professional  
📧 Email: your.email@domain.com  
💼 LinkedIn: [Your LinkedIn Profile]  
🌐 Portfolio: [Your Portfolio Website]  

### **Contributing**
Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🎓 Educational Notice

**Important**: This project is developed for educational and demonstration purposes. The use of the "Commerzbank" name and branding is solely for portfolio demonstration and does not represent any official affiliation with Commerzbank AG.

*Die Verwendung des Namens und der Marke "Commerzbank" dient ausschließlich Bildungs- und Demonstrationszwecken und stellt keine offizielle Verbindung zur Commerzbank AG dar.*

---

<div align="center">

**⭐ If you found this project helpful, please give it a star!**

[![GitHub stars](https://img.shields.io/github/stars/your-username/option-pricing-validation?style=social)](https://github.com/your-username/option-pricing-validation/stargazers)

</div>