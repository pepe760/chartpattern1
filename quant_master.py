#@title US Quant Master — S&P 500 Full Universe & Stage Analysis Edition
#!/usr/bin/env python3
"""
================================================================================
  US Quant Master — 15-Pattern Scientific Backtesting Engine
  ══════════════════════════════════════════════════════════════
  Features:
    ● S&P 500 Full Universe (Dynamic scraping from Wikipedia)
    ● 15 Chart Patterns (Bullish + Bearish) with OOP architecture
    ● Market Regime Filter: Analyzes SPY to determine Stages 1-4
    ● Trailing Stop Loss (Chandelier Exit)
    ● Comprehensive ZIP file export (CSV + PNGs)
================================================================================
"""

import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import warnings
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy import stats as sp_stats
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from abc import ABC, abstractmethod
from tqdm import tqdm
import zipfile
import os
import requests

warnings.filterwarnings('ignore')
np.random.seed(42)

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 0. DYNAMIC S&P 500 FETCHER                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def get_sp500_tickers() -> List[str]:
    """Scrape the current S&P 500 tickers from Wikipedia."""
    print("🌐 正在從 Wikipedia 獲取最新 S&P 500 成分股清單...")
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        html = requests.get(url, headers=headers).text
        table = pd.read_html(html)[0]
        tickers = table['Symbol'].tolist()
        # Yahoo Finance uses '-' instead of '.' for Berkshire and Brown-Forman
        tickers = [t.replace('.', '-') for t in tickers]
        print(f"✅ 成功獲取 {len(tickers)} 檔 S&P 500 成分股。")
        return tickers
    except Exception as e:
        print(f"❌ 獲取 S&P 500 清單失敗: {e}")
        print("⚠️ 降級使用預設前 20 大權值股進行測試。")
        return ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'BRK-B', 'LLY', 'AVGO', 'JPM', 
                'TSLA', 'UNH', 'V', 'XOM', 'MA', 'JNJ', 'PG', 'HD', 'COST', 'MRK']

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 1. CONFIGURATION                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
@dataclass
class Config:
    # ── Data ──
    # 使用 factory 動態載入 S&P 500
    tickers: List[str] = field(default_factory=get_sp500_tickers)
    years: int = 5
    market_proxy: str = 'SPY' 

    # ── Pattern Detection Thresholds ──
    peak_order: int = 10          
    peak_order_large: int = 20    
    hs_min_bars: int = 40
    hs_max_bars: int = 200
    hs_shoulder_tol: float = 0.04    
    hs_head_min_prom: float = 0.03   
    ihs_min_bars: int = 40
    ihs_max_bars: int = 200
    ihs_shoulder_tol: float = 0.04
    ihs_head_min_prom: float = 0.03
    dtop_min_bars: int = 20
    dtop_max_bars: int = 120
    dtop_tolerance: float = 0.03
    dbot_min_bars: int = 20
    dbot_max_bars: int = 120
    dbot_tolerance: float = 0.03
    wedge_min_bars: int = 20
    wedge_max_bars: int = 100
    wedge_min_touches: int = 4       
    round_min_bars: int = 30
    round_max_bars: int = 150
    round_r2_threshold: float = 0.70  
    flag_pole_min_gain: float = 0.08
    flag_pole_max_bars: int = 15
    flag_consol_min_bars: int = 3
    flag_consol_max_bars: int = 12
    flag_max_retrace: float = 0.50
    bear_flag_pole_min_drop: float = 0.08
    bear_flag_consol_min_bars: int = 3
    bear_flag_consol_max_bars: int = 12
    bear_flag_max_retrace: float = 0.50
    cup_min_bars: int = 30
    cup_max_bars: int = 200
    cup_min_depth: float = 0.12
    cup_max_depth: float = 0.35
    handle_max_bars: int = 25
    handle_max_retrace: float = 0.12
    vcp_min_contractions: int = 2
    vcp_contraction_ratio: float = 0.75
    vcp_final_max_range: float = 0.10
    vcp_window_bars: int = 10
    tri_min_bars: int = 15
    tri_max_bars: int = 80
    tri_flat_tol: float = 0.02       
    tri_min_touches: int = 3
    sym_tri_min_bars: int = 15
    sym_tri_max_bars: int = 80
    sym_tri_converge_rate: float = 0.02
    rect_min_bars: int = 15
    rect_max_bars: int = 80
    rect_range_tol: float = 0.03     

    # ── Backtest ──
    atr_period: int = 14
    sl_atr_mult: float = 2.0         
    tp_atr_mult: float = 3.0         
    max_hold_days: int = 30
    slippage_pct: float = 0.001
    commission_bps: float = 5.0      
    use_trailing_stop: bool = True   # 開啟移動停損

CFG = Config()

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 2. DATA ENGINE & STAGE ANALYSIS                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class DataEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def download(self) -> Dict[str, pd.DataFrame]:
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=self.cfg.years * 365)
        print("\n" + "=" * 70)
        print("📡 DATA DOWNLOAD & STAGE ANALYSIS ENGINE (S&P 500)")
        print("=" * 70)
        print(f"  📅 {start.strftime('%Y-%m-%d')} → {end.strftime('%Y-%m-%d')}")
        print(f"  ⚠️ 警告：正在處理 500 檔股票與計算技術指標，這大約需要 3-5 分鐘，請耐心等候。")
        
        all_data = {}
        # 確保大盤指標 (SPY) 第一個下載，以便計算 Stage
        tickers = [self.cfg.market_proxy] + [t for t in self.cfg.tickers if t != self.cfg.market_proxy]
        market_stages = pd.Series(dtype=int)

        for ticker in tqdm(tickers, desc="  ⏳ Downloading & Processing"):
            try:
                df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
                if df is not None and len(df) > 100:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    df = self._add_indicators(df)
                    
                    if ticker == self.cfg.market_proxy:
                        market_stages = self._calculate_market_stages(df)
                        df['Market_Stage'] = market_stages
                    else:
                        if market_stages.empty:
                            print(f"    ⚠️ {ticker} 略過：大盤指標載入失敗")
                            continue
                        df['Market_Stage'] = market_stages.reindex(df.index, method='ffill')
                    
                    all_data[ticker] = df
            except Exception as e:
                pass # 忽略下市或抓取失敗的少數股票以維持版面乾淨

        print(f"  ✅ 成功載入並計算 {len(all_data)} 檔股票數據")
        return all_data

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        c, h, l = df['Close'], df['High'], df['Low']
        tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(self.cfg.atr_period).mean()
        df['SMA30_W'] = c.rolling(150).mean() # 約 30 週均線
        df['SMA30_W_ROC'] = df['SMA30_W'].diff(20)
        return df

    def _calculate_market_stages(self, df: pd.DataFrame) -> pd.Series:
        """Stan Weinstein 階段分析 (1:底, 2:多, 3:頂, 4:空)"""
        stages = pd.Series(index=df.index, data=0)
        c, ma, roc = df['Close'], df['SMA30_W'], df['SMA30_W_ROC']

        for i in range(150, len(df)):
            price, ma_val, ma_roc = c.iloc[i], ma.iloc[i], roc.iloc[i]
            slope_thresh = ma_val * 0.005  

            if price > ma_val and ma_roc > slope_thresh:
                stages.iloc[i] = 2 # Bull
            elif price < ma_val and ma_roc < -slope_thresh:
                stages.iloc[i] = 4 # Bear
            elif price >= ma_val and abs(ma_roc) <= slope_thresh:
                prev = stages.iloc[i-1] if i > 0 else 0
                stages.iloc[i] = 1 if prev in (4, 1) else 3
            elif price < ma_val and abs(ma_roc) <= slope_thresh:
                 prev = stages.iloc[i-1] if i > 0 else 0
                 stages.iloc[i] = 3 if prev in (2, 3) else 1
            else:
                stages.iloc[i] = stages.iloc[i-1]
        return stages


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 3. BASE PATTERN DETECTOR & ALL 15 IMPLEMENTATIONS                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class PatternDetector(ABC):
    def __init__(self, cfg: Config): self.cfg = cfg
    @property
    @abstractmethod
    def name(self) -> str: pass
    @property
    @abstractmethod
    def direction(self) -> str: pass
    @abstractmethod
    def detect(self, df: pd.DataFrame, i: int) -> Optional[Dict]: pass

    def _find_peaks(self, data: np.ndarray, order: int = 10) -> np.ndarray:
        return signal.argrelextrema(data, np.greater_equal, order=order)[0]
    def _find_troughs(self, data: np.ndarray, order: int = 10) -> np.ndarray:
        return signal.argrelextrema(data, np.less_equal, order=order)[0]
    def _fit_trendline(self, xs: np.ndarray, ys: np.ndarray) -> Tuple[float, float, float]:
        if len(xs) < 2: return 0.0, 0.0, 0.0
        slope, intercept, r_val, _, _ = sp_stats.linregress(xs.astype(float), ys)
        return slope, intercept, r_val ** 2
    def _calc_atr(self, df: pd.DataFrame, i: int) -> float:
        atr = df['ATR'].iloc[i]
        return atr if not pd.isna(atr) and atr > 0 else df['Close'].iloc[i] * 0.02

class HeadAndShoulders(PatternDetector):
    @property
    def name(self): return "Head&Shoulders"
    @property
    def direction(self): return "bearish"
    def detect(self, df, i):
        cfg = self.cfg
        if i < cfg.hs_max_bars: return None
        highs, closes = df['High'].iloc[i-cfg.hs_max_bars:i+1].values, df['Close'].iloc[i-cfg.hs_max_bars:i+1].values
        peaks = self._find_peaks(highs, order=cfg.peak_order_large)
        if len(peaks) < 3: return None
        for p_start in range(max(0, len(peaks)-5), len(peaks)-2):
            for p1, p2, p3 in itertools.combinations(peaks[p_start:], 3):
                ls, hd, rs = highs[p1], highs[p2], highs[p3]
                if hd <= ls or hd <= rs or (hd-ls)/ls < cfg.hs_head_min_prom or abs(ls-rs)/max(ls, 1e-9) > cfg.hs_shoulder_tol: continue
                if not (cfg.hs_min_bars <= (p3-p1) <= cfg.hs_max_bars): continue
                troughs = self._find_troughs(highs[p1:p3+1], order=5)
                if len(troughs) < 1: continue
                neckline = np.mean([highs[p1+t] for t in troughs])
                if closes[-1] < neckline * 1.01:
                    atr = self._calc_atr(df, i)
                    return {'entry_price': closes[-1], 'tp': closes[-1]-atr*cfg.tp_atr_mult, 'sl': closes[-1]+atr*cfg.sl_atr_mult, 'direction': 'short'}
        return None

class InverseHeadAndShoulders(PatternDetector):
    @property
    def name(self): return "InvH&S"
    @property
    def direction(self): return "bullish"
    def detect(self, df, i):
        cfg = self.cfg
        if i < cfg.ihs_max_bars: return None
        lows, closes = df['Low'].iloc[i-cfg.ihs_max_bars:i+1].values, df['Close'].iloc[i-cfg.ihs_max_bars:i+1].values
        troughs = self._find_troughs(lows, order=cfg.peak_order_large)
        if len(troughs) < 3: return None
        for t_start in range(max(0, len(troughs)-5), len(troughs)-2):
            for t1, t2, t3 in itertools.combinations(troughs[t_start:], 3):
                ls, hd, rs = lows[t1], lows[t2], lows[t3]
                if hd >= ls or hd >= rs or (ls-hd)/max(ls,1e-9) < cfg.ihs_head_min_prom or abs(ls-rs)/max(ls,1e-9) > cfg.ihs_shoulder_tol: continue
                if not (cfg.ihs_min_bars <= (t3-t1) <= cfg.ihs_max_bars): continue
                peaks = self._find_peaks(lows[t1:t3+1], order=5)
                if len(peaks) < 1: continue
                neckline = np.mean([lows[t1+p] for p in peaks])
                if closes[-1] > neckline * 0.99:
                    atr = self._calc_atr(df, i)
                    return {'entry_price': closes[-1], 'tp': closes[-1]+atr*cfg.tp_atr_mult, 'sl': closes[-1]-atr*cfg.sl_atr_mult, 'direction': 'long'}
        return None

class DoubleTop(PatternDetector):
    @property
    def name(self): return "DoubleTop"
    @property
    def direction(self): return "bearish"
    def detect(self, df, i):
        cfg = self.cfg
        if i < cfg.dtop_max_bars: return None
        highs, closes = df['High'].iloc[i-cfg.dtop_max_bars:i+1].values, df['Close'].iloc[i-cfg.dtop_max_bars:i+1].values
        peaks = self._find_peaks(highs, order=cfg.peak_order)
        if len(peaks) < 2: return None
        for p1, p2 in zip(peaks[-4:], peaks[-3:]):
            if p2 <= p1 or not (cfg.dtop_min_bars <= (p2-p1) <= cfg.dtop_max_bars): continue
            t1, t2 = highs[p1], highs[p2]
            if abs(t1-t2)/max(t1, 1e-9) > cfg.dtop_tolerance: continue
            neckline = np.min(closes[p1:p2+1])
            if closes[-1] < neckline * 1.01:
                atr = self._calc_atr(df, i)
                return {'entry_price': closes[-1], 'tp': closes[-1]-atr*cfg.tp_atr_mult, 'sl': closes[-1]+atr*cfg.sl_atr_mult, 'direction': 'short'}
        return None

class DoubleBottom(PatternDetector):
    @property
    def name(self): return "DoubleBottom"
    @property
    def direction(self): return "bullish"
    def detect(self, df, i):
        cfg = self.cfg
        if i < cfg.dbot_max_bars: return None
        lows, closes = df['Low'].iloc[i-cfg.dbot_max_bars:i+1].values, df['Close'].iloc[i-cfg.dbot_max_bars:i+1].values
        troughs = self._find_troughs(lows, order=cfg.peak_order)
        if len(troughs) < 2: return None
        for t1, t2 in zip(troughs[-4:], troughs[-3:]):
            if t2 <= t1 or not (cfg.dbot_min_bars <= (t2-t1) <= cfg.dbot_max_bars): continue
            b1, b2 = lows[t1], lows[t2]
            if abs(b1-b2)/max(b1, 1e-9) > cfg.dbot_tolerance: continue
            neckline = np.max(closes[t1:t2+1])
            if closes[-1] > neckline * 0.99:
                atr = self._calc_atr(df, i)
                return {'entry_price': closes[-1], 'tp': closes[-1]+atr*cfg.tp_atr_mult, 'sl': closes[-1]-atr*cfg.sl_atr_mult, 'direction': 'long'}
        return None

class FallingWedge(PatternDetector):
    @property
    def name(self): return "FallingWedge"
    @property
    def direction(self): return "bullish"
    def detect(self, df, i):
        cfg = self.cfg
        for lookback in [cfg.wedge_max_bars, cfg.wedge_max_bars//2]:
            if i < lookback: continue
            highs, lows, closes = df['High'].iloc[i-lookback:i+1].values, df['Low'].iloc[i-lookback:i+1].values, df['Close'].iloc[i-lookback:i+1].values
            peaks, troughs = self._find_peaks(highs, cfg.peak_order), self._find_troughs(lows, cfg.peak_order)
            if len(peaks) < 2 or len(troughs) < 2 or len(peaks)+len(troughs) < cfg.wedge_min_touches: continue
            slope_h, inter_h, r2_h = self._fit_trendline(peaks, highs[peaks])
            slope_l, inter_l, r2_l = self._fit_trendline(troughs, lows[troughs])
            if slope_h >= 0 or slope_l >= 0 or slope_l >= slope_h or r2_h < 0.5 or r2_l < 0.5: continue
            if closes[-1] > slope_h * len(highs) + inter_h:
                atr = self._calc_atr(df, i)
                return {'entry_price': closes[-1], 'tp': closes[-1]+atr*cfg.tp_atr_mult, 'sl': closes[-1]-atr*cfg.sl_atr_mult, 'direction': 'long'}
        return None

class RisingWedge(PatternDetector):
    @property
    def name(self): return "RisingWedge"
    @property
    def direction(self): return "bearish"
    def detect(self, df, i):
        cfg = self.cfg
        for lookback in [cfg.wedge_max_bars, cfg.wedge_max_bars//2]:
            if i < lookback: continue
            highs, lows, closes = df['High'].iloc[i-lookback:i+1].values, df['Low'].iloc[i-lookback:i+1].values, df['Close'].iloc[i-lookback:i+1].values
            peaks, troughs = self._find_peaks(highs, cfg.peak_order), self._find_troughs(lows, cfg.peak_order)
            if len(peaks) < 2 or len(troughs) < 2 or len(peaks)+len(troughs) < cfg.wedge_min_touches: continue
            slope_h, inter_h, r2_h = self._fit_trendline(peaks, highs[peaks])
            slope_l, inter_l, r2_l = self._fit_trendline(troughs, lows[troughs])
            if slope_h <= 0 or slope_l <= 0 or slope_h >= slope_l or r2_h < 0.5 or r2_l < 0.5: continue
            if closes[-1] < slope_l * len(lows) + inter_l:
                atr = self._calc_atr(df, i)
                return {'entry_price': closes[-1], 'tp': closes[-1]-atr*cfg.tp_atr_mult, 'sl': closes[-1]+atr*cfg.sl_atr_mult, 'direction': 'short'}
        return None

class RoundingBottom(PatternDetector):
    @property
    def name(self): return "RoundingBottom"
    @property
    def direction(self): return "bullish"
    def detect(self, df, i):
        cfg = self.cfg
        for lb in [cfg.round_max_bars, cfg.round_max_bars//2]:
            if i < lb or lb < cfg.round_min_bars: continue
            closes = df['Close'].iloc[i-lb:i+1].values
            n = len(closes)
            a, b, c = np.polyfit(np.arange(n, dtype=float), closes, 2)
            if a <= 0: continue
            fitted = np.polyval([a, b, c], np.arange(n))
            r2 = 1 - np.sum((closes - fitted)**2) / max(np.sum((closes - np.mean(closes))**2), 1e-9)
            min_idx = -b / (2*a)
            if r2 < cfg.round_r2_threshold or min_idx < n*0.2 or min_idx > n*0.8: continue
            if closes[-1] > closes[0] * 0.98:
                atr = self._calc_atr(df, i)
                return {'entry_price': closes[-1], 'tp': closes[-1]+atr*cfg.tp_atr_mult, 'sl': closes[-1]-atr*cfg.sl_atr_mult, 'direction': 'long'}
        return None

class BullFlag(PatternDetector):
    @property
    def name(self): return "BullFlag"
    @property
    def direction(self): return "bullish"
    def detect(self, df, i):
        cfg = self.cfg
        lookback = cfg.flag_pole_max_bars + cfg.flag_consol_max_bars + 5
        if i < lookback: return None
        closes, highs, lows = df['Close'].values, df['High'].values, df['Low'].values
        for c_len in range(cfg.flag_consol_min_bars, cfg.flag_consol_max_bars+1):
            c_start = i - c_len
            if c_start < cfg.flag_pole_max_bars: continue
            c_high, c_low = np.max(highs[c_start:i+1]), np.min(lows[c_start:i+1])
            for p_len in range(5, cfg.flag_pole_max_bars+1):
                p_start = c_start - p_len
                if p_start < 0: continue
                p_bot, p_top = np.min(lows[p_start:c_start]), np.max(highs[p_start:c_start])
                gain = (p_top - p_bot) / max(p_bot, 1e-9)
                retrace = (p_top - c_low) / max(p_top - p_bot, 1e-9)
                c_range = (c_high - c_low) / max(c_low, 1e-9)
                if gain >= cfg.flag_pole_min_gain and retrace <= cfg.flag_max_retrace and c_range <= gain*0.6:
                    if closes[i] > c_high:
                        atr = self._calc_atr(df, i)
                        return {'entry_price': closes[i], 'tp': closes[i]+atr*cfg.tp_atr_mult, 'sl': closes[i]-atr*cfg.sl_atr_mult, 'direction': 'long'}
        return None

class BearFlag(PatternDetector):
    @property
    def name(self): return "BearFlag"
    @property
    def direction(self): return "bearish"
    def detect(self, df, i):
        cfg = self.cfg
        lookback = cfg.flag_pole_max_bars + cfg.bear_flag_consol_max_bars + 5
        if i < lookback: return None
        closes, highs, lows = df['Close'].values, df['High'].values, df['Low'].values
        for c_len in range(cfg.bear_flag_consol_min_bars, cfg.bear_flag_consol_max_bars+1):
            c_start = i - c_len
            if c_start < cfg.flag_pole_max_bars: continue
            c_high, c_low = np.max(highs[c_start:i+1]), np.min(lows[c_start:i+1])
            for p_len in range(5, cfg.flag_pole_max_bars+1):
                p_start = c_start - p_len
                if p_start < 0: continue
                p_top, p_bot = np.max(highs[p_start:c_start]), np.min(lows[p_start:c_start])
                drop = (p_top - p_bot) / max(p_top, 1e-9)
                retrace = (c_high - p_bot) / max(p_top - p_bot, 1e-9)
                c_range = (c_high - c_low) / max(c_low, 1e-9)
                if drop >= cfg.bear_flag_pole_min_drop and retrace <= cfg.bear_flag_max_retrace and c_range <= drop*0.6:
                    if closes[i] < c_low:
                        atr = self._calc_atr(df, i)
                        return {'entry_price': closes[i], 'tp': closes[i]-atr*cfg.tp_atr_mult, 'sl': closes[i]+atr*cfg.sl_atr_mult, 'direction': 'short'}
        return None

class CupWithHandle(PatternDetector):
    @property
    def name(self): return "CWH"
    @property
    def direction(self): return "bullish"
    def detect(self, df, i):
        cfg = self.cfg
        if i < cfg.cup_max_bars: return None
        closes = df['Close'].values
        for cup_len in [cfg.cup_max_bars, cfg.cup_max_bars//2, cfg.cup_min_bars]:
            if i < cup_len + cfg.handle_max_bars: continue
            c_start, c_end = i - cup_len - cfg.handle_max_bars, i - cfg.handle_max_bars
            if c_start < 0 or c_end <= c_start: continue
            cup_slice = closes[c_start:c_end+1]
            if len(cup_slice) < cfg.cup_min_bars: continue
            l_rim_idx = np.argmax(cup_slice[:len(cup_slice)//3+1])
            l_rim = cup_slice[l_rim_idx]
            c_bot_idx = l_rim_idx + np.argmin(cup_slice[l_rim_idx:])
            c_bot = cup_slice[c_bot_idx]
            if l_rim <= 0: continue
            depth = (l_rim - c_bot) / l_rim
            r_rim = np.max(cup_slice[c_bot_idx:])
            if depth < cfg.cup_min_depth or depth > cfg.cup_max_depth or r_rim < l_rim*0.9: continue
            handle_slice = closes[c_end:i+1]
            if len(handle_slice) < 2: continue
            h_high, h_low = np.max(handle_slice), np.min(handle_slice)
            if (h_high - h_low) / max(h_high, 1e-9) > cfg.handle_max_retrace: continue
            if closes[i] > max(h_high, r_rim) * 0.99:
                atr = self._calc_atr(df, i)
                return {'entry_price': closes[i], 'tp': closes[i]+atr*cfg.tp_atr_mult, 'sl': closes[i]-atr*cfg.sl_atr_mult, 'direction': 'long'}
        return None

class VCP(PatternDetector):
    @property
    def name(self): return "VCP"
    @property
    def direction(self): return "bullish"
    def detect(self, df, i):
        cfg = self.cfg
        total_bars = 4 * cfg.vcp_window_bars
        if i < total_bars: return None
        highs, lows, closes = df['High'].values, df['Low'].values, df['Close'].values
        ranges = []
        for j in range(4):
            s = i - total_bars + j * cfg.vcp_window_bars
            e = s + cfg.vcp_window_bars
            ranges.append(np.max(highs[s:e]) - np.min(lows[s:e]))
        c = closes[i]
        if c <= 0 or ranges[-1]/c > cfg.vcp_final_max_range: return None
        contractions = sum(1 for j in range(1, 4) if ranges[j] < ranges[j-1] * cfg.vcp_contraction_ratio)
        if contractions >= cfg.vcp_min_contractions and c >= np.max(highs[i-cfg.vcp_window_bars:i+1]) * 0.99:
            atr = self._calc_atr(df, i)
            return {'entry_price': c, 'tp': c+atr*cfg.tp_atr_mult, 'sl': c-atr*cfg.sl_atr_mult, 'direction': 'long'}
        return None

class AscendingTriangle(PatternDetector):
    @property
    def name(self): return "AscTriangle"
    @property
    def direction(self): return "bullish"
    def detect(self, df, i):
        cfg = self.cfg
        for lb in [cfg.tri_max_bars, cfg.tri_max_bars//2]:
            if i < lb or lb < cfg.tri_min_bars: continue
            highs, lows, closes = df['High'].iloc[i-lb:i+1].values, df['Low'].iloc[i-lb:i+1].values, df['Close'].iloc[i-lb:i+1].values
            peaks, troughs = self._find_peaks(highs, cfg.peak_order), self._find_troughs(lows, cfg.peak_order)
            if len(peaks) < cfg.tri_min_touches - 1 or len(troughs) < 2: continue
            p_vals = highs[peaks]
            if (np.max(p_vals) - np.min(p_vals)) / max(np.mean(p_vals), 1e-9) > cfg.tri_flat_tol: continue
            slope_l, _, r2_l = self._fit_trendline(troughs, lows[troughs])
            res = np.mean(p_vals)
            if slope_l > 0 and r2_l >= 0.4 and closes[-1] > res * 1.001:
                atr = self._calc_atr(df, i)
                return {'entry_price': closes[-1], 'tp': closes[-1]+atr*cfg.tp_atr_mult, 'sl': closes[-1]-atr*cfg.sl_atr_mult, 'direction': 'long'}
        return None

class DescendingTriangle(PatternDetector):
    @property
    def name(self): return "DescTriangle"
    @property
    def direction(self): return "bearish"
    def detect(self, df, i):
        cfg = self.cfg
        for lb in [cfg.tri_max_bars, cfg.tri_max_bars//2]:
            if i < lb or lb < cfg.tri_min_bars: continue
            highs, lows, closes = df['High'].iloc[i-lb:i+1].values, df['Low'].iloc[i-lb:i+1].values, df['Close'].iloc[i-lb:i+1].values
            peaks, troughs = self._find_peaks(highs, cfg.peak_order), self._find_troughs(lows, cfg.peak_order)
            if len(troughs) < cfg.tri_min_touches - 1 or len(peaks) < 2: continue
            t_vals = lows[troughs]
            if (np.max(t_vals) - np.min(t_vals)) / max(np.mean(t_vals), 1e-9) > cfg.tri_flat_tol: continue
            slope_h, _, r2_h = self._fit_trendline(peaks, highs[peaks])
            sup = np.mean(t_vals)
            if slope_h < 0 and r2_h >= 0.4 and closes[-1] < sup * 0.999:
                atr = self._calc_atr(df, i)
                return {'entry_price': closes[-1], 'tp': closes[-1]-atr*cfg.tp_atr_mult, 'sl': closes[-1]+atr*cfg.sl_atr_mult, 'direction': 'short'}
        return None

class SymmetricalTriangle(PatternDetector):
    @property
    def name(self): return "SymTriangle"
    @property
    def direction(self): return "neutral"
    def detect(self, df, i):
        cfg = self.cfg
        for lb in [cfg.sym_tri_max_bars, cfg.sym_tri_max_bars//2]:
            if i < lb or lb < cfg.sym_tri_min_bars: continue
            highs, lows, closes = df['High'].iloc[i-lb:i+1].values, df['Low'].iloc[i-lb:i+1].values, df['Close'].iloc[i-lb:i+1].values
            peaks, troughs = self._find_peaks(highs, cfg.peak_order), self._find_troughs(lows, cfg.peak_order)
            if len(peaks) < 2 or len(troughs) < 2: continue
            slope_h, inter_h, r2_h = self._fit_trendline(peaks, highs[peaks])
            slope_l, inter_l, r2_l = self._fit_trendline(troughs, lows[troughs])
            if slope_h >= 0 or slope_l <= 0 or abs(abs(slope_h)-abs(slope_l))/max(abs(slope_h),1e-9) > 0.5 or r2_h < 0.3 or r2_l < 0.3: continue
            u_end, l_end = slope_h*len(highs)+inter_h, slope_l*len(lows)+inter_l
            atr = self._calc_atr(df, i)
            if closes[-1] > u_end: return {'entry_price': closes[-1], 'tp': closes[-1]+atr*cfg.tp_atr_mult, 'sl': closes[-1]-atr*cfg.sl_atr_mult, 'direction': 'long'}
            if closes[-1] < l_end: return {'entry_price': closes[-1], 'tp': closes[-1]-atr*cfg.tp_atr_mult, 'sl': closes[-1]+atr*cfg.sl_atr_mult, 'direction': 'short'}
        return None

class Rectangle(PatternDetector):
    @property
    def name(self): return "Rectangle"
    @property
    def direction(self): return "neutral"
    def detect(self, df, i):
        cfg = self.cfg
        for lb in [cfg.rect_max_bars, cfg.rect_max_bars//2]:
            if i < lb or lb < cfg.rect_min_bars: continue
            highs, lows, closes = df['High'].iloc[i-lb:i+1].values, df['Low'].iloc[i-lb:i+1].values, df['Close'].iloc[i-lb:i+1].values
            peaks, troughs = self._find_peaks(highs, cfg.peak_order), self._find_troughs(lows, cfg.peak_order)
            if len(peaks) < 2 or len(troughs) < 2: continue
            p_vals, t_vals = highs[peaks], lows[troughs]
            if (np.max(p_vals)-np.min(p_vals))/max(np.mean(p_vals),1e-9) > cfg.rect_range_tol or \
               (np.max(t_vals)-np.min(t_vals))/max(np.mean(t_vals),1e-9) > cfg.rect_range_tol: continue
            res, sup = np.mean(p_vals), np.mean(t_vals)
            if not (0.02 <= (res-sup)/max(sup,1e-9) <= 0.20): continue
            atr = self._calc_atr(df, i)
            if closes[-1] > res * 1.001: return {'entry_price': closes[-1], 'tp': closes[-1]+atr*cfg.tp_atr_mult, 'sl': closes[-1]-atr*cfg.sl_atr_mult, 'direction': 'long'}
            if closes[-1] < sup * 0.999: return {'entry_price': closes[-1], 'tp': closes[-1]-atr*cfg.tp_atr_mult, 'sl': closes[-1]+atr*cfg.sl_atr_mult, 'direction': 'short'}
        return None

def build_all_detectors(cfg: Config) -> List[PatternDetector]:
    return [
        HeadAndShoulders(cfg), InverseHeadAndShoulders(cfg), DoubleTop(cfg), DoubleBottom(cfg),
        FallingWedge(cfg), RisingWedge(cfg), RoundingBottom(cfg), BullFlag(cfg), BearFlag(cfg),
        CupWithHandle(cfg), VCP(cfg), AscendingTriangle(cfg), DescendingTriangle(cfg),
        SymmetricalTriangle(cfg), Rectangle(cfg)
    ]

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 4. BACKTESTING ENGINE (Stage-Aware + Trailing Stop)                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class BacktestEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def run_single_pattern(self, detector: PatternDetector, all_data: Dict[str, pd.DataFrame]) -> Dict:
        trades = []
        # 改用 tqdm 追蹤這 500 檔股票的模式掃描進度
        for ticker, df in tqdm(all_data.items(), desc=f"  🔍 掃描 {detector.name:<15}", leave=False):
            n = len(df)
            e_idx = n - self.cfg.max_hold_days - 1
            s_idx = 250
            if s_idx >= e_idx: continue

            for i in range(s_idx, e_idx):
                market_stage = df['Market_Stage'].iloc[i]
                if pd.isna(market_stage) or market_stage == 0: continue

                sig = detector.detect(df, i)
                if sig is None: continue

                trade = self._simulate_trade(df, i, sig)
                if trade:
                    trade['ticker'] = ticker
                    trade['pattern'] = detector.name
                    trade['entry_date'] = df.index[i]
                    trade['market_stage'] = int(market_stage)
                    trades.append(trade)

        return self._compute_stats(trades, detector.name)

    def _simulate_trade(self, df: pd.DataFrame, entry_idx: int, sig: Dict) -> Optional[Dict]:
        entry, tp, initial_sl, direction = sig['entry_price'], sig['tp'], sig['sl'], sig['direction']
        max_bars = self.cfg.max_hold_days
        atr_val = df['ATR'].iloc[entry_idx] if not pd.isna(df['ATR'].iloc[entry_idx]) else entry * 0.02

        slip = entry * self.cfg.slippage_pct
        commission = entry * self.cfg.commission_bps / 10000
        entry = entry + slip + commission if direction == 'long' else entry - slip - commission

        current_sl, highest_high, lowest_low = initial_sl, entry, entry

        for j in range(1, max_bars + 1):
            idx = entry_idx + j
            if idx >= len(df): break
            h, l, c = df['High'].iloc[idx], df['Low'].iloc[idx], df['Close'].iloc[idx]

            if direction == 'long':
                if self.cfg.use_trailing_stop:
                    highest_high = max(highest_high, h)
                    current_sl = max(current_sl, highest_high - (atr_val * self.cfg.sl_atr_mult))
                if l <= current_sl:
                    pnl = (current_sl - entry) / entry
                    return {'pnl_pct': pnl * 100, 'exit_reason': 'SL', 'hold_days': j, 'success': pnl > 0}
                if h >= tp:
                    pnl = (tp - entry) / entry
                    return {'pnl_pct': pnl * 100, 'exit_reason': 'TP', 'hold_days': j, 'success': True}
            else:  
                if self.cfg.use_trailing_stop:
                    lowest_low = min(lowest_low, l)
                    current_sl = min(current_sl, lowest_low + (atr_val * self.cfg.sl_atr_mult))
                if h >= current_sl:
                    pnl = (entry - current_sl) / entry
                    return {'pnl_pct': pnl * 100, 'exit_reason': 'SL', 'hold_days': j, 'success': pnl > 0}
                if l <= tp:
                    pnl = (entry - tp) / entry
                    return {'pnl_pct': pnl * 100, 'exit_reason': 'TP', 'hold_days': j, 'success': True}

        if entry_idx + max_bars < len(df):
            exit_price = df['Close'].iloc[entry_idx + max_bars]
            pnl = (exit_price - entry) / entry if direction == 'long' else (entry - exit_price) / entry
            return {'pnl_pct': pnl * 100, 'exit_reason': 'TIME', 'hold_days': max_bars, 'success': pnl > 0}
        return None

    def _compute_stats(self, trades: List[Dict], name: str) -> Dict:
        if not trades: return {'pattern': name, 'total_trades': 0, 'win_rate': 0, 'avg_pnl': 0, 'stage_breakdown': {}}
        pnls = [t['pnl_pct'] for t in trades]
        
        stage_breakdown = {1: {'pnls': []}, 2: {'pnls': []}, 3: {'pnls': []}, 4: {'pnls': []}}
        for t in trades: stage_breakdown[t['market_stage']]['pnls'].append(t['pnl_pct'])
        for s in stage_breakdown:
            s_pnls = stage_breakdown[s]['pnls']
            stage_breakdown[s]['count'] = len(s_pnls)
            stage_breakdown[s]['win_rate'] = (sum(1 for p in s_pnls if p > 0) / len(s_pnls) * 100) if s_pnls else 0

        return {
            'pattern': name,
            'total_trades': len(trades),
            'win_rate': sum(1 for p in pnls if p > 0) / len(pnls) * 100,
            'avg_pnl': np.mean(pnls),
            'trades': trades,
            'stage_breakdown': stage_breakdown
        }

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 5. EXPORT & VISUALIZATION ENGINE                                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class ExportEngine:
    @staticmethod
    def plot_stage_analysis(all_results: List[Dict], save_path: str = 'stage_analysis.png'):
        valid_results = [r for r in all_results if r.get('total_trades', 0) > 0]
        if not valid_results: return
        patterns = [r['pattern'] for r in valid_results]
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('Pattern Win Rate by Market Stage (S&P 500)', fontsize=16, fontweight='bold')
        colors = ['#9E9E9E', '#4CAF50', '#FF9800', '#F44336'] 
        
        for i, stage in enumerate([1, 2, 3, 4]):
            ax = axes.flatten()[i]
            wrs = [r['stage_breakdown'][stage]['win_rate'] for r in valid_results]
            counts = [r['stage_breakdown'][stage]['count'] for r in valid_results]
            bars = ax.bar(patterns, wrs, color=colors[i], alpha=0.8)
            ax.axhline(y=50, color='black', linestyle='--', alpha=0.5)
            ax.set_title(f'Stage {stage} ({["Accumulation", "Advancing (Bull)", "Distribution", "Declining (Bear)"][i]})')
            ax.set_ylabel('Win Rate (%)')
            ax.set_ylim(0, 100)
            ax.tick_params(axis='x', rotation=45)
            for bar, count in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"N={count}", ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    @staticmethod
    def export_to_zip(all_results: List[Dict], filename: str = 'quant_master_results_sp500.zip'):
        print(f"\n📦 Preparing export bundle: {filename}")
        
        all_trades = []
        for r in all_results:
            trades_list = r.get('trades', [])
            all_trades.extend(trades_list)
        
        if all_trades:
            pd.DataFrame(all_trades).to_csv('all_trades.csv', index=False)
            print(f"  📝 成功彙整 {len(all_trades)} 筆交易紀錄。")
        else:
            print("  ⚠️ 警告：沒有偵測到任何交易紀錄。")
            
        with zipfile.ZipFile(filename, 'w') as zipf:
            if os.path.exists('all_trades.csv'):
                zipf.write('all_trades.csv')
                os.remove('all_trades.csv')
            
            # 打包所有的圖片檔案
            for f in os.listdir('.'):
                if f.endswith('.png'):
                    zipf.write(f)
        print(f"  ✅ 打包完成！檔名：{filename}")
      
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 6. MAIN EXECUTION                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def main():
    print("╔" + "═" * 68 + "╗")
    print("║  US Quant Master — S&P 500 Full Universe & Stage Analysis        ║")
    print("╚" + "═" * 68 + "╝")

    data_engine = DataEngine(CFG)
    all_data = data_engine.download()
    if not all_data: return

    detectors = build_all_detectors(CFG)
    engine = BacktestEngine(CFG)
    all_results = []

    print("\n" + "=" * 70)
    print("🔍 STAGE 1: Pattern Detection & Stage Tracking (Across ~500 Stocks)")
    print("=" * 70)

    for det in detectors:
        res = engine.run_single_pattern(det, all_data)
        all_results.append(res)
        print(f"  ✅ {det.name:<15} | 總交易次數: {res['total_trades']:<5} | 勝率: {res['win_rate']:.1f}%")
        if res['total_trades'] > 0:
            print(f"     └─ Stage 2 (牛市) 勝率: {res['stage_breakdown'][2]['win_rate']:.1f}% (N={res['stage_breakdown'][2]['count']})")
            print(f"     └─ Stage 4 (熊市) 勝率: {res['stage_breakdown'][4]['win_rate']:.1f}% (N={res['stage_breakdown'][4]['count']})")

    ExportEngine.plot_stage_analysis(all_results)
    ExportEngine.export_to_zip(all_results)

if __name__ == '__main__':
    main()
