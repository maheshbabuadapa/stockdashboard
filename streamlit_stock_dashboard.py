import streamlit as st

# Check for required packages and provide helpful error messages
try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    st.error("Plotly is required. Install with: pip install plotly")
    st.stop()

try:
    import pandas as pd
except ImportError:
    st.error("Pandas is required. Install with: pip install pandas")
    st.stop()

try:
    import yfinance as yf
except ImportError:
    st.error("yfinance is required. Install with: pip install yfinance")
    st.stop()

try:
    import numpy as np
except ImportError:
    st.error("NumPy is required. Install with: pip install numpy")
    st.stop()

try:
    from scipy import stats
except ImportError:
    st.error("SciPy is required. Install with: pip install scipy")
    st.stop()

import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# ========================= Peer Groups =========================
PEER_GROUPS = {
    "semiconductors": ["NVDA","AVGO","AMD","INTC","QCOM","TXN","ADI","NXPI","MU","MRVL","MCHP","ON","KLAC","LRCX","AMAT","ASML","TSM","ARM","TER","LSCC","SWKS","QRVO","GFS","WOLF"],
    "internet_platforms": ["AAPL","MSFT","GOOGL","META","AMZN","NFLX"],
    "enterprise_software_saas": ["ADBE","CRM","NOW","WDAY","SNOW","DDOG","TEAM","MDB","HUBS","INTU","ADSK","DOCU","OKTA","ZS"],
    "cybersecurity": ["PANW","CRWD","ZS","FTNT","OKTA","S","TENB","CHKP","CYBR"],
    "ecommerce_marketplaces": ["AMZN","SHOP","MELI","SE","BABA","PDD","JD","ETSY","EBAY","W"],
    "enterprise_infra_networking": ["CSCO","ANET","JNPR","HPE","DELL","NTNX","FFIV","GLW","UI","NTAP","PSTG"],
    "compute_servers_storage": ["SMCI","DELL","HPE","NTAP","PSTG","IBM","HPQ"],
    "it_services_consulting": ["ACN","IBM","EPAM","GLOB","INFY","WIT","CTSH","DXC"],
    "payments_fintech": ["V","MA","PYPL","SQ","AXP","FIS","GPN","COIN"],
    "telecom_wireless": ["TMUS","VZ","T"],
    "electric_utilities": ["SO","AEP","DUK","NEE","EXC","D","EIX","XEL","PPL","PEG"],
    "grocery_retail": ["WMT","COST","KR","TGT","ACI","SFM","GO","BJ","TSCDY","ADRNY"],
    "food_processing": ["GIS","K","KHC","CAG","MDLZ","HSY","MKC","CPB","HRL","TSN"],
    "banks_us_large": ["JPM","BAC","C","WFC","GS","MS","USB","PNC","TFC"]
}

# ========================= Enhanced Weights (5-Year Focus) =========================
METRIC_WEIGHTS = {
    # Valuation (40%) - reduced weight, more focus on historical performance
    'P/E Ratio': 0.10,
    'P/B Ratio': 0.08,
    'EV/EBITDA': 0.12,
    'P/S Ratio': 0.10,
    
    # Quality & Growth (35%) - enhanced with 5-year metrics
    'ROE': 0.10,
    'ROE_5Y_Avg': 0.05,  # New: 5-year average ROE
    'Profit Margin': 0.08,
    'Revenue Growth': 0.07,
    'Revenue_5Y_CAGR': 0.05,  # New: 5-year revenue CAGR
    
    # Performance & Risk (20%) - new 5-year metrics
    'Price_5Y_CAGR': 0.08,  # New: 5-year price CAGR
    'Volatility_5Y': 0.04,  # New: 5-year volatility (lower is better)
    'Max_Drawdown_5Y': 0.04,  # New: Maximum drawdown (lower is better)
    'Sharpe_Ratio_5Y': 0.04,  # New: 5-year Sharpe ratio
    
    # Financial Health (5%)
    'Debt to Equity': 0.03,
    'Current_Ratio_Stability': 0.02,  # New: Current ratio consistency
}

# Metrics used for coverage calculation
CORE_METRICS = ['P/E Ratio','P/B Ratio','EV/EBITDA','P/S Ratio',
                'ROE','Profit Margin','Revenue Growth','Debt to Equity',
                'Price_5Y_CAGR','Volatility_5Y','ROE_5Y_Avg','Revenue_5Y_CAGR']

# ========================= Enhanced Data Fetcher =========================
class EnhancedStockDataFetcher:
    @staticmethod
    def _safe_get_name(ticker):
        """Best effort to get company name."""
        name = None
        try:
            info = ticker.get_info()
            name = info.get('shortName') or info.get('longName')
        except Exception:
            try:
                info = ticker.info
                name = info.get('shortName') or info.get('longName')
            except Exception:
                pass
        return name

    @staticmethod
    def _calculate_5year_metrics(ticker, current_price):
        """Calculate comprehensive 5-year historical metrics."""
        try:
            # Get 5+ years of data to ensure we have enough
            hist = ticker.history(period="6y", interval="1d", auto_adjust=True)
            if hist.empty or len(hist) < 1000:  # Need substantial data
                return {}
            
            # Calculate price-based metrics
            closes = hist['Close'].dropna()
            if len(closes) < 1260:  # ~5 years of trading days
                return {}
            
            # 5-Year Price CAGR
            price_start = closes.iloc[0]
            price_end = current_price if current_price else closes.iloc[-1]
            years = len(closes) / 252  # approximate years
            price_5y_cagr = ((price_end / price_start) ** (1/years) - 1) * 100 if price_start > 0 else None
            
            # 5-Year Volatility (annualized)
            returns = closes.pct_change().dropna()
            volatility_5y = returns.std() * np.sqrt(252) * 100  # Annualized volatility %
            
            # Maximum Drawdown
            rolling_max = closes.expanding().max()
            drawdowns = (closes - rolling_max) / rolling_max
            max_drawdown_5y = abs(drawdowns.min()) * 100  # Convert to positive percentage
            
            # Sharpe Ratio (using risk-free rate approximation of 2%)
            if len(returns) > 0:
                excess_returns = returns - (0.02/252)  # Daily risk-free rate
                sharpe_5y = (excess_returns.mean() / excess_returns.std() * np.sqrt(252)) if excess_returns.std() > 0 else 0
            else:
                sharpe_5y = 0
            
            # Recent momentum metrics
            ret_6m = ret_12m = None
            if len(closes) > 126:
                ret_6m = (closes.iloc[-1] / closes.iloc[-126] - 1) * 100
            if len(closes) > 252:
                ret_12m = (closes.iloc[-1] / closes.iloc[-252] - 1) * 100
            
            # Volume metrics
            vols = hist['Volume'].dropna()
            avg_vol_10d = vols.tail(10).mean() if len(vols) >= 10 else None
            dollar_vol_10d = avg_vol_10d * price_end if avg_vol_10d and price_end else None
            
            return {
                'Price_5Y_CAGR': price_5y_cagr,
                'Volatility_5Y': volatility_5y,
                'Max_Drawdown_5Y': max_drawdown_5y,
                'Sharpe_Ratio_5Y': sharpe_5y,
                'Ret_6M_%': ret_6m,
                'Ret_12M_%': ret_12m,
                'AvgVol10d': avg_vol_10d,
                'DollarVol10d': dollar_vol_10d
            }
            
        except Exception as e:
            print(f"Error calculating 5-year metrics: {e}")
            return {}

    @staticmethod
    def _calculate_fundamental_trends(ticker):
        """Calculate 5-year fundamental trends using available financial data."""
        try:
            # Get financial statements
            financials = ticker.financials
            balance_sheet = ticker.balance_sheet
            
            if financials.empty:
                return {}
            
            metrics = {}
            
            # Revenue 5-Year CAGR
            try:
                revenues = financials.loc['Total Revenue'] if 'Total Revenue' in financials.index else None
                if revenues is not None and len(revenues) >= 2:
                    revenues = revenues.sort_index()  # Chronological order
                    if len(revenues) >= 4:  # Need at least 4 years of data
                        start_revenue = revenues.iloc[0]
                        end_revenue = revenues.iloc[-1]
                        years = len(revenues) - 1
                        if start_revenue > 0:
                            revenue_cagr = ((end_revenue / start_revenue) ** (1/years) - 1) * 100
                            metrics['Revenue_5Y_CAGR'] = revenue_cagr
            except Exception:
                pass
            
            # ROE 5-Year Average (approximate using available data)
            try:
                # Try to calculate ROE trend from available financial data
                if not balance_sheet.empty:
                    net_incomes = financials.loc['Net Income'] if 'Net Income' in financials.index else None
                    total_equity = balance_sheet.loc['Total Stockholder Equity'] if 'Total Stockholder Equity' in balance_sheet.index else None
                    
                    if net_incomes is not None and total_equity is not None:
                        # Align dates and calculate ROE for available periods
                        common_dates = net_incomes.index.intersection(total_equity.index)
                        if len(common_dates) >= 3:
                            roes = []
                            for date in common_dates:
                                if total_equity[date] > 0:
                                    roe = (net_incomes[date] / total_equity[date]) * 100
                                    roes.append(roe)
                            if roes:
                                metrics['ROE_5Y_Avg'] = np.mean(roes)
            except Exception:
                pass
            
            # Current Ratio Stability (consistency measure)
            try:
                if not balance_sheet.empty:
                    current_assets = balance_sheet.loc['Current Assets'] if 'Current Assets' in balance_sheet.index else None
                    current_liabilities = balance_sheet.loc['Current Liabilities'] if 'Current Liabilities' in balance_sheet.index else None
                    
                    if current_assets is not None and current_liabilities is not None:
                        ratios = []
                        for date in current_assets.index:
                            if date in current_liabilities.index and current_liabilities[date] > 0:
                                ratio = current_assets[date] / current_liabilities[date]
                                ratios.append(ratio)
                        
                        if len(ratios) >= 3:
                            # Stability measured as inverse of coefficient of variation
                            cv = np.std(ratios) / np.mean(ratios) if np.mean(ratios) > 0 else float('inf')
                            stability = max(0, 100 - cv * 100)  # Higher = more stable
                            metrics['Current_Ratio_Stability'] = stability
            except Exception:
                pass
            
            return metrics
            
        except Exception as e:
            print(f"Error calculating fundamental trends: {e}")
            return {}

    @staticmethod
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def get_stock_data(symbols):
        """Fetch comprehensive stock data with 5-year historical analysis."""
        data = []

        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, symbol in enumerate(symbols):
            try:
                status_text.text(f"Processing {symbol}... ({i+1}/{len(symbols)})")
                progress_bar.progress((i + 1) / len(symbols))
                
                ticker = yf.Ticker(symbol)

                # ---------- Current Price & Change ----------
                price = None
                prev_close = None
                change_pct = None
                
                try:
                    fi = getattr(ticker, "fast_info", {}) or {}
                    price = fi.get("last_price")
                    prev_close = fi.get("previous_close")
                except Exception:
                    pass

                if price is None:
                    try:
                        hist = ticker.history(period="2d", auto_adjust=False)
                        if not hist.empty:
                            price = float(hist["Close"].iloc[-1])
                    except Exception:
                        pass

                if prev_close is None:
                    try:
                        info_tmp = ticker.get_info()
                        if info_tmp and info_tmp.get("previousClose"):
                            prev_close = float(info_tmp["previousClose"])
                    except Exception:
                        pass

                if price is not None and prev_close:
                    change_pct = (price / prev_close - 1.0) * 100.0

                # ---------- Company Name ----------
                company_name = EnhancedStockDataFetcher._safe_get_name(ticker) or symbol

                # ---------- Current Fundamentals ----------
                info = {}
                try:
                    info = ticker.get_info()
                except Exception:
                    try:
                        info = ticker.info
                    except Exception:
                        info = {}

                # Basic valuation metrics
                pe_ratio = info.get('forwardPE') or info.get('trailingPE')
                if pe_ratio is not None and pe_ratio <= 0:
                    pe_ratio = None

                pb_ratio = info.get('priceToBook')
                ev_ebitda = info.get('enterpriseToEbitda')
                if ev_ebitda is not None and ev_ebitda <= 0:
                    ev_ebitda = None

                ps_ratio = info.get('priceToSalesTrailing12Months')
                dividend_yield = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0

                # Quality metrics
                roe = info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0
                profit_margin = info.get('profitMargins', 0) * 100 if info.get('profitMargins') else 0
                revenue_growth = info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') else 0
                debt_to_equity = info.get('debtToEquity', 0)

                # Additional metrics
                gross_margins = info.get('grossMargins', 0) * 100 if info.get('grossMargins') else 0
                operating_margins = info.get('operatingMargins', 0) * 100 if info.get('operatingMargins') else 0
                current_ratio = info.get('currentRatio', 0)

                # ---------- 5-Year Historical Metrics ----------
                historical_metrics = EnhancedStockDataFetcher._calculate_5year_metrics(ticker, price)
                fundamental_trends = EnhancedStockDataFetcher._calculate_fundamental_trends(ticker)

                # Combine all data
                stock_data = {
                    'Symbol': symbol,
                    'Name': company_name,
                    'Price': price,
                    'Change %': change_pct,

                    # Current Valuation
                    'P/E Ratio': pe_ratio,
                    'P/B Ratio': pb_ratio,
                    'EV/EBITDA': ev_ebitda,
                    'P/S Ratio': ps_ratio,
                    'Dividend Yield': dividend_yield,

                    # Current Quality
                    'ROE': roe,
                    'Profit Margin': profit_margin,
                    'Revenue Growth': revenue_growth,
                    'Debt to Equity': debt_to_equity,

                    # Additional Current Metrics
                    'Gross Margin': gross_margins,
                    'Operating Margin': operating_margins,
                    'Current Ratio': current_ratio,

                    # 5-Year Historical Performance
                    'Price_5Y_CAGR': historical_metrics.get('Price_5Y_CAGR'),
                    'Volatility_5Y': historical_metrics.get('Volatility_5Y'),
                    'Max_Drawdown_5Y': historical_metrics.get('Max_Drawdown_5Y'),
                    'Sharpe_Ratio_5Y': historical_metrics.get('Sharpe_Ratio_5Y'),

                    # 5-Year Fundamental Trends
                    'Revenue_5Y_CAGR': fundamental_trends.get('Revenue_5Y_CAGR'),
                    'ROE_5Y_Avg': fundamental_trends.get('ROE_5Y_Avg'),
                    'Current_Ratio_Stability': fundamental_trends.get('Current_Ratio_Stability'),

                    # Recent Performance & Liquidity
                    'Ret_6M_%': historical_metrics.get('Ret_6M_%'),
                    'Ret_12M_%': historical_metrics.get('Ret_12M_%'),
                    'AvgVol10d': historical_metrics.get('AvgVol10d'),
                    'DollarVol10d': historical_metrics.get('DollarVol10d')
                }

                data.append(stock_data)

            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                # Create minimal entry for failed stocks
                stock_data = {col: None for col in [
                    'Symbol', 'Name', 'Price', 'Change %',
                    'P/E Ratio', 'P/B Ratio', 'EV/EBITDA', 'P/S Ratio', 'Dividend Yield',
                    'ROE', 'Profit Margin', 'Revenue Growth', 'Debt to Equity',
                    'Gross Margin', 'Operating Margin', 'Current Ratio',
                    'Price_5Y_CAGR', 'Volatility_5Y', 'Max_Drawdown_5Y', 'Sharpe_Ratio_5Y',
                    'Revenue_5Y_CAGR', 'ROE_5Y_Avg', 'Current_Ratio_Stability',
                    'Ret_6M_%', 'Ret_12M_%', 'AvgVol10d', 'DollarVol10d'
                ]}
                stock_data['Symbol'] = symbol
                stock_data['Name'] = symbol
                data.append(stock_data)

        progress_bar.empty()
        status_text.empty()
        return pd.DataFrame(data)

# ========================= Enhanced Analyzer =========================
class Enhanced5YearAnalyzer:
    @staticmethod
    def _winsorize_series(s: pd.Series, lower=0.01, upper=0.99) -> pd.Series:
        x = s.dropna().astype(float)
        if len(x) < 3:
            return s
        lo = x.quantile(lower)
        hi = x.quantile(upper)
        return s.clip(lower=lo, upper=hi)

    @staticmethod
    def calculate_z_scores(df):
        """Enhanced Z-scores with 5-year metrics consideration."""
        df_scores = df.copy()

        # Lower is better (flip sign)
        lower_better_metrics = ['P/E Ratio', 'P/B Ratio', 'EV/EBITDA', 'P/S Ratio', 
                               'Debt to Equity', 'Volatility_5Y', 'Max_Drawdown_5Y']
        
        # Higher is better
        higher_better_metrics = ['ROE', 'Profit Margin', 'Revenue Growth', 'Dividend Yield',
                                'Price_5Y_CAGR', 'Sharpe_Ratio_5Y', 'Revenue_5Y_CAGR', 
                                'ROE_5Y_Avg', 'Current_Ratio_Stability']
        
        all_metrics = lower_better_metrics + higher_better_metrics

        for metric in all_metrics:
            if metric in df_scores.columns:
                # Winsorize to reduce outlier impact
                s_clean = Enhanced5YearAnalyzer._winsorize_series(df_scores[metric])
                
                valid = s_clean.dropna().astype(float) if not s_clean.dropna().empty else pd.Series(dtype=float)
                if len(valid) > 1:
                    mean_val = valid.mean()
                    std_val = valid.std()
                    if std_val > 0:
                        z = (s_clean - mean_val) / std_val
                        if metric in lower_better_metrics:
                            z = -z  # Flip for lower-is-better metrics
                        df_scores[f'{metric}_Z'] = z
                    else:
                        df_scores[f'{metric}_Z'] = 0
                else:
                    df_scores[f'{metric}_Z'] = 0
        
        return df_scores

    @staticmethod
    def calculate_comprehensive_scores(df):
        """Calculate comprehensive scores with 5-year weighting."""
        z_cols = [c for c in df.columns if c.endswith('_Z')]

        composite_scores = []
        valuation_scores = []
        quality_scores = []
        performance_scores = []

        for _, row in df.iterrows():
            total_score = total_weight = 0.0
            val_score = val_weight = 0.0
            qual_score = qual_weight = 0.0
            perf_score = perf_weight = 0.0

            for col in z_cols:
                metric = col.replace('_Z', '')
                if pd.notna(row[col]) and metric in METRIC_WEIGHTS:
                    w = METRIC_WEIGHTS[metric]
                    contrib = row[col] * w
                    total_score += contrib
                    total_weight += w

                    # Categorize metrics
                    if metric in ['P/E Ratio', 'P/B Ratio', 'EV/EBITDA', 'P/S Ratio']:
                        val_score += contrib; val_weight += w
                    elif metric in ['ROE', 'ROE_5Y_Avg', 'Profit Margin', 'Revenue Growth', 
                                  'Revenue_5Y_CAGR', 'Debt to Equity', 'Current_Ratio_Stability']:
                        qual_score += contrib; qual_weight += w
                    elif metric in ['Price_5Y_CAGR', 'Volatility_5Y', 'Max_Drawdown_5Y', 'Sharpe_Ratio_5Y']:
                        perf_score += contrib; perf_weight += w

            composite_scores.append(total_score / total_weight if total_weight else 0)
            valuation_scores.append(val_score / val_weight if val_weight else 0)
            quality_scores.append(qual_score / qual_weight if qual_weight else 0)
            performance_scores.append(perf_score / perf_weight if perf_weight else 0)

        df['Composite_Score'] = composite_scores
        df['Valuation_Score'] = valuation_scores
        df['Quality_Score'] = quality_scores
        df['Performance_5Y_Score'] = performance_scores

        # Sort by composite score and assign rankings
        df = df.sort_values('Composite_Score', ascending=False).reset_index(drop=True)
        df['Rank'] = np.arange(1, len(df) + 1)

        # Convert to 1-10 scales
        if len(df) > 1:
            df['Overall_Score'] = 10 - ((df['Rank'] - 1) * 9 / (len(df) - 1))

            # Individual component scores (1-10)
            for score_col, new_col in [('Valuation_Score', 'Value_Score_1_10'),
                                     ('Quality_Score', 'Quality_Score_1_10'),
                                     ('Performance_5Y_Score', 'Performance_Score_1_10')]:
                df_temp = df.sort_values(score_col, ascending=False).reset_index(drop=True)
                df_temp['Temp_Rank'] = np.arange(1, len(df_temp) + 1)
                df_temp[new_col] = 10 - ((df_temp['Temp_Rank'] - 1) * 9 / (len(df_temp) - 1))
                df = df.merge(df_temp[['Symbol', new_col]], on='Symbol', how='left')
        else:
            df['Overall_Score'] = 5.5
            df['Value_Score_1_10'] = 5.5
            df['Quality_Score_1_10'] = 5.5
            df['Performance_Score_1_10'] = 5.5

        # Ensure numeric columns
        numeric_cols = ['Overall_Score', 'Value_Score_1_10', 'Quality_Score_1_10', 'Performance_Score_1_10',
                       'P/E Ratio', 'P/B Ratio', 'EV/EBITDA', 'P/S Ratio', 'Dividend Yield',
                       'ROE', 'Profit Margin', 'Revenue Growth', 'Debt to Equity',
                       'Price', 'Change %', 'Price_5Y_CAGR', 'Volatility_5Y', 'Max_Drawdown_5Y',
                       'Sharpe_Ratio_5Y', 'Revenue_5Y_CAGR', 'ROE_5Y_Avg', 'Current_Ratio_Stability',
                       'Ret_6M_%', 'Ret_12M_%', 'AvgVol10d', 'DollarVol10d']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df

def style_dataframe(df, score_columns=None):
    """Apply styling to dataframe for better visualization"""
    if score_columns is None:
        score_columns = []
    
    def color_score(val):
        if pd.isna(val):
            return ''
        if val >= 8:
            return 'background-color: #d4edda'
        elif val >= 6:
            return 'background-color: #d1ecf1'
        elif val >= 4:
            return 'background-color: #fff3cd'
        else:
            return 'background-color: #f8d7da'
    
    def color_change(val):
        if pd.isna(val):
            return ''
        return 'color: green' if val >= 0 else 'color: red'
    
    styled = df.style
    
    for col in score_columns:
        if col in df.columns:
            styled = styled.applymap(color_score, subset=[col])
    
    if 'Change %' in df.columns:
        styled = styled.applymap(color_change, subset=['Change %'])
    
    return styled

def create_summary_cards(df_final, selected_group):
    """Create summary cards for the dashboard"""
    if df_final.empty:
        return
    
    top_stock = df_final.iloc[0]
    top_price_txt = f"${top_stock['Price']:,.2f}" if pd.notna(top_stock.get('Price')) else "n/a"
    top_5y_cagr = f"{top_stock.get('Price_5Y_CAGR', 0):.1f}%" if pd.notna(top_stock.get('Price_5Y_CAGR')) else "n/a"
    avg_score = df_final['Overall_Score'].mean()
    avg_5y_cagr = df_final['Price_5Y_CAGR'].mean()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Top Ranked Stock",
            value=f"{top_stock['Symbol']}",
            delta=f"Score: {top_stock['Overall_Score']:.1f}/10"
        )
        st.caption(f"Price: {top_price_txt}")
        st.caption(f"5Y CAGR: {top_5y_cagr}")
    
    with col2:
        st.metric(
            label="Group Average Score",
            value=f"{avg_score:.1f}/10",
            delta=f"Stocks: {len(df_final)}"
        )
        st.caption(f"Avg 5Y CAGR: {avg_5y_cagr:.1f}%" if pd.notna(avg_5y_cagr) else "Avg 5Y CAGR: n/a")
    
    with col3:
        avg_coverage = df_final['Coverage_%'].mean()
        high_quality_count = len(df_final[df_final['Coverage_%'] >= 80])
        st.metric(
            label="Analysis Quality",
            value=f"{avg_coverage:.0f}%",
            delta=f"High Quality: {high_quality_count}/{len(df_final)}"
        )
        st.caption("Enhanced w/ 5Y Data")

def main():
    st.set_page_config(
        page_title="Enhanced 5-Year Stock Analysis Dashboard",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🚀 Enhanced 5-Year Stock Analysis Dashboard")
    st.markdown("📈 Comprehensive analysis incorporating 5 years of historical data for better investment insights")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        selected_group = st.selectbox(
            "Select Peer Group:",
            options=list(PEER_GROUPS.keys()),
            format_func=lambda x: x.replace('_', ' ').title(),
            index=0
        )
        
        st.markdown("---")
        st.markdown("### Enhanced Metric Weights (5-Year Focus)")
        st.markdown("**VALUATION (40%)**")
        st.markdown("• P/E: 10% | P/B: 8% | EV/EBITDA: 12% | P/S: 10%")
        st.markdown("**QUALITY & GROWTH (35%)**")
        st.markdown("• ROE: 10% | 5Y Avg ROE: 5% | Profit Margin: 8%")
        st.markdown("• Revenue Growth: 7% | 5Y Revenue CAGR: 5%")
        st.markdown("**PERFORMANCE & RISK (20%)**")
        st.markdown("• 5Y Price CAGR: 8% | 5Y Volatility: 4%")
        st.markdown("• Max Drawdown: 4% | Sharpe Ratio: 4%")
        st.markdown("**FINANCIAL HEALTH (5%)**")
        st.markdown("• Debt/Equity: 3% | Current Ratio Stability: 2%")
        
        st.markdown("---")
        st.markdown("### Key Enhancements")
        st.markdown("• 5-year historical price performance")
        st.markdown("• Risk metrics: volatility, drawdowns")
        st.markdown("• Long-term fundamental trends")
        st.markdown("• Enhanced risk flags")
        
        if st.button("Clear Cache", help="Clear cached data to force refresh"):
            st.cache_data.clear()
            st.success("Cache cleared!")
    
    if not selected_group:
        st.error("Please select a peer group to begin analysis.")
        return
    
    symbols = PEER_GROUPS[selected_group]
    
    with st.spinner(f'Loading comprehensive 5-year analysis for {selected_group.replace("_", " ").title()}... This may take a few minutes.'):
        try:
            # Fetch data
            fetcher = EnhancedStockDataFetcher()
            df = fetcher.get_stock_data(symbols)
            
            if df.empty:
                st.error("No data could be fetched. Please try again.")
                return
            
            # Calculate scores
            analyzer = Enhanced5YearAnalyzer()
            df_with_scores = analyzer.calculate_z_scores(df)
            df_final = analyzer.calculate_comprehensive_scores(df_with_scores)
            
            # Calculate coverage
            df_final['Coverage_%'] = (df_final[CORE_METRICS].notna().mean(axis=1) * 100).round(0)
            
            # Enhanced Risk Flags
            def build_enhanced_flags(row):
                flags = []
                if pd.notna(row['Coverage_%']) and row['Coverage_%'] < 70:
                    flags.append("Low Coverage")
                if pd.notna(row.get('DollarVol10d')) and row['DollarVol10d'] < 5_000_000:
                    flags.append("Illiquid")
                if pd.notna(row.get('Volatility_5Y')) and row['Volatility_5Y'] > 50:
                    flags.append("High Vol")
                if pd.notna(row.get('Max_Drawdown_5Y')) and row['Max_Drawdown_5Y'] > 50:
                    flags.append("High Drawdown")
                if pd.notna(row.get('Price_5Y_CAGR')) and row['Price_5Y_CAGR'] < 0:
                    flags.append("Neg 5Y Returns")
                if pd.notna(row.get('Ret_6M_%')) and row['Ret_6M_%'] < -20:
                    flags.append("Weak Recent")
                return " | ".join(flags) if flags else "Clean"

            df_final['Risk_Flags'] = df_final.apply(build_enhanced_flags, axis=1)
            
            # Display summary cards
            create_summary_cards(df_final, selected_group)
            
            # Score interpretation
            st.markdown("---")
            st.markdown("**Score Interpretation:** 🟢 Excellent (8-10) | 🟡 Good (6-8) | 🟠 Fair (4-6) | 🔴 Poor (1-4)")
            st.markdown("⭐ Scores now incorporate 5-year trends: Price CAGR, Volatility, Revenue Growth, ROE Consistency")
            
            # Main analysis table
            st.markdown("## 📊 Enhanced Comprehensive Analysis (5-Year Historical Data)")
            
            display_columns = [
                'Symbol', 'Name', 'Price', 'Change %',
                'Overall_Score', 'Value_Score_1_10', 'Quality_Score_1_10', 'Performance_Score_1_10',
                'P/E Ratio', 'P/B Ratio', 'EV/EBITDA', 'Dividend Yield',
                'ROE', 'Profit Margin', 'Revenue Growth',
                'Price_5Y_CAGR', 'Revenue_5Y_CAGR', 'ROE_5Y_Avg',
                'Volatility_5Y', 'Sharpe_Ratio_5Y'
            ]
            
            available_columns = [c for c in display_columns if c in df_final.columns]
            df_display = df_final[available_columns].copy()
            
            # Rename columns for better display
            df_display = df_display.rename(columns={
                'Overall_Score': 'Overall Score',
                'Value_Score_1_10': 'Value Score',
                'Quality_Score_1_10': 'Quality Score',
                'Performance_Score_1_10': 'Performance Score',
                'Price_5Y_CAGR': '5Y Price CAGR %',
                'Revenue_5Y_CAGR': '5Y Rev CAGR %',
                'ROE_5Y_Avg': '5Y Avg ROE %',
                'Volatility_5Y': '5Y Volatility %',
                'Sharpe_Ratio_5Y': '5Y Sharpe'
            })
            
            # Apply styling and display
            score_cols = ['Overall Score', 'Value Score', 'Quality Score', 'Performance Score']
            styled_df = style_dataframe(df_display, score_cols)
            
            st.dataframe(styled_df, use_container_width=True, height=600)
            
            # Risk Analysis Table
            st.markdown("## 🛡️ Enhanced Risk Analysis (5-Year Historical Context)")
            st.markdown("🚨 Red flags: Low coverage, thin liquidity, high volatility, large drawdowns, poor long-term performance")
            
            df_risk = pd.DataFrame({
                'Symbol': df_final['Symbol'],
                'Name': df_final['Name'],
                'Coverage %': df_final['Coverage_%'],
                '$ Vol 10d ($M)': (df_final['DollarVol10d'] / 1_000_000.0).round(1),
                '5Y CAGR %': df_final['Price_5Y_CAGR'],
                '5Y Volatility %': df_final['Volatility_5Y'],
                '5Y Max DD %': df_final['Max_Drawdown_5Y'],
                '5Y Sharpe': df_final['Sharpe_Ratio_5Y'],
                '6M Ret %': df_final['Ret_6M_%'],
                'Risk Flags': df_final['Risk_Flags']
            })
            
            st.dataframe(df_risk, use_container_width=True, height=400)
            
            # Charts section
            st.markdown("## 📈 Visualizations")
            
            # Top row of charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Score bar chart
                fig_bar = px.bar(
                    df_final.head(15), x='Symbol', y='Overall_Score',
                    title=f'📊 Top 15 Overall Scores - {selected_group.replace("_", " ").title()}',
                    color='Overall_Score', 
                    color_continuous_scale=['red', 'orange', 'yellow', 'lightgreen', 'green'],
                    range_color=[1, 10], text='Overall_Score'
                )
                fig_bar.update_traces(texttemplate='%{text:.1f}', textposition='outside')
                fig_bar.update_layout(
                    xaxis_title="Stock Symbol", yaxis_title="Overall Score (10=Best)",
                    showlegend=False, height=400
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col2:
                # Performance vs Quality scatter
                fig_scatter = px.scatter(
                    df_final, x='Quality_Score_1_10', y='Performance_Score_1_10', 
                    text='Symbol', title=f'⚡ 5-Year Performance vs Quality - {selected_group.replace("_", " ").title()}',
                    color='Overall_Score', 
                    color_continuous_scale=['red', 'orange', 'yellow', 'lightgreen', 'green'],
                    size='Value_Score_1_10', size_max=20, 
                    hover_data=['Price_5Y_CAGR', 'Volatility_5Y']
                )
                fig_scatter.update_traces(textposition="top center")
                fig_scatter.update_layout(
                    xaxis_title="Quality Score (10=Best)", 
                    yaxis_title="5-Year Performance Score (10=Best)", 
                    height=400
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Bottom row of charts
            col3, col4 = st.columns(2)
            
            with col3:
                # Historical performance chart
                fig_hist = px.bar(
                    df_final.head(15), x='Symbol', y='Price_5Y_CAGR',
                    title=f'📈 5-Year Price CAGR % - Top 15 Stocks',
                    color='Price_5Y_CAGR', 
                    color_continuous_scale=['red', 'orange', 'yellow', 'lightgreen', 'green'],
                    text='Price_5Y_CAGR'
                )
                fig_hist.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig_hist.update_layout(
                    xaxis_title="Stock Symbol", yaxis_title="5-Year CAGR (%)",
                    showlegend=False, height=400
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col4:
                # Risk-return scatter
                fig_risk_return = px.scatter(
                    df_final, x='Volatility_5Y', y='Price_5Y_CAGR',
                    text='Symbol', title=f'🎯 Risk vs Return (5-Year) - {selected_group.replace("_", " ").title()}',
                    color='Sharpe_Ratio_5Y', 
                    color_continuous_scale=['red', 'orange', 'yellow', 'lightgreen', 'green'],
                    size='Overall_Score', size_max=20
                )
                fig_risk_return.update_traces(textposition="top center")
                fig_risk_return.update_layout(
                    xaxis_title="5-Year Volatility (%)", yaxis_title="5-Year CAGR (%)", 
                    height=400
                )
                st.plotly_chart(fig_risk_return, use_container_width=True)
            
            # Comprehensive radar chart
            st.markdown("### 🎯 Comprehensive Analysis - Top 5 Stocks")
            top_stocks = df_final.head(5)
            z_metrics = ['P/E Ratio_Z', 'ROE_Z', 'Revenue Growth_Z', 'Price_5Y_CAGR_Z', 
                        'Volatility_5Y_Z', 'Revenue_5Y_CAGR_Z', 'Sharpe_Ratio_5Y_Z']
            available_metrics = [m for m in z_metrics if m in df_final.columns]
            
            fig_radar = go.Figure()
            colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
            
            for i, (_, stock) in enumerate(top_stocks.iterrows()):
                values, labels = [], []
                for m in available_metrics:
                    v = stock.get(m)
                    if pd.notna(v):
                        values.append(float(v))
                        clean_label = m.replace('_Z', '').replace('_', ' ').replace('Price 5Y CAGR', '5Y Returns').replace('Volatility 5Y', 'Low Volatility')
                        labels.append(clean_label)
                
                if values and len(values) >= 3:  # Need at least 3 metrics for radar
                    values.append(values[0])  # Close the radar
                    labels.append(labels[0])
                    fig_radar.add_trace(go.Scatterpolar(
                        r=values, theta=labels, fill='toself', 
                        name=f"{stock['Symbol']} ({stock['Overall_Score']:.1f})",
                        line_color=colors[i % len(colors)]
                    ))

            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[-2.5, 2.5])),
                showlegend=True, 
                title=f"🎯 Comprehensive Analysis - Top 5 Stocks ({selected_group.replace('_', ' ').title()})",
                height=500
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
            
            # Download section
            st.markdown("## 📥 Download Data")
            col1, col2 = st.columns(2)
            
            with col1:
                csv_main = df_display.to_csv(index=False)
                st.download_button(
                    label="Download Main Analysis (CSV)",
                    data=csv_main,
                    file_name=f"{selected_group}_analysis.csv",
                    mime="text/csv"
                )
            
            with col2:
                csv_risk = df_risk.to_csv(index=False)
                st.download_button(
                    label="Download Risk Analysis (CSV)",
                    data=csv_risk,
                    file_name=f"{selected_group}_risk_analysis.csv",
                    mime="text/csv"
                )
            
        except Exception as e:
            st.error(f"Error in enhanced analysis: {str(e)}")
            st.exception(e)

if __name__ == '__main__':
    st.markdown("""
    ---
    ### 📊 Enhanced Metric Weights (5-Year Focus)
    
    **🏷️ VALUATION (40%)**
    - P/E Ratio: 10% | P/B Ratio: 8% | EV/EBITDA: 12% | P/S Ratio: 10%

    **📈 QUALITY & GROWTH (35%)**
    - ROE: 10% | 5Y Avg ROE: 5% | Profit Margin: 8%
    - Revenue Growth: 7% | 5Y Revenue CAGR: 5%

    **⚡ PERFORMANCE & RISK (20%)**
    - 5Y Price CAGR: 8% | 5Y Volatility: 4%
    - Max Drawdown: 4% | Sharpe Ratio: 4%

    **💰 FINANCIAL HEALTH (5%)**
    - Debt/Equity: 3% | Current Ratio Stability: 2%

    ### 🔍 Key Enhancements:
    - 5-year historical price performance analysis
    - Risk metrics: volatility, drawdowns, Sharpe ratios  
    - Long-term fundamental trend analysis
    - Enhanced risk flags and coverage analysis
    - Performance vs Quality matrix visualization

    ⚠️ **Note:** Initial load may take 2-3 minutes due to comprehensive 5-year data analysis
    """)
    
    main()
