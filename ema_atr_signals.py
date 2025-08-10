"""
TradeBot: Enhanced EMA50/EMA200 cross with ATR-based TP/SL signals
- Monitors ETHUSDT and BTCUSDT on 15m timeframe
- Uses ATR for dynamic take profit and stop loss calculations
- Simplified output showing only TP and SL prices
- Telegram alerts with AI chat functionality
"""

import time
import threading
import requests
import pandas as pd
import numpy as np
import os
from datetime import datetime
from binance.client import Client
import sys

# ========== CONFIG ==========
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "WSPoE9L04b8xukvqh7li4WT1TEqNnGob4ZCanZ4p2lTy2AbjYxzjV8q9lPcARRNI")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "J4PoDOoyBhpMTKIBBHGa1eCz8gNG19N85yTxfwFGG5tNcPUVgbrwKA51lox451Xg")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8459911523:AAHXBnaNywpvSNU59MhD6G0RqJn993VKhUU")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "6080078099")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
GROQ_MODEL = "llama3-8b-8192"

SYMBOLS = ["ETHUSDT", "BTCUSDT"]
TIMEFRAME = "15m"
EMA_SHORT = 50
EMA_LONG = 200
ATR_PERIOD = 14
CHECK_INTERVAL_SECONDS = 60
BINANCE_KLINES_LIMIT = 500

# Dynamic Risk Management Configuration
BASE_ATR_TP_MULTIPLIER = 2.0  # Base Take Profit multiplier
BASE_ATR_SL_MULTIPLIER = 1.0  # Base Stop Loss multiplier
RISK_PER_TRADE = 0.10  # 2% risk per trade
ACCOUNT_BALANCE = 3.0  # Default account balance (update with actual balance)
MIN_TP_MULTIPLIER = 1.5
MAX_TP_MULTIPLIER = 3.5
MIN_SL_MULTIPLIER = 0.5
MAX_SL_MULTIPLIER = 2.0
VOLATILITY_LOOKBACK = 20  # Periods to look back for volatility calculation

# Rate limiting
MAX_RETRIES = 3
RETRY_DELAY = 5
TELEGRAM_RATE_LIMIT = 1

# Telegram endpoints
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"

# ========== GLOBAL STATE ==========
client = None
last_signal = {sym: None for sym in SYMBOLS}
last_candle_time = {sym: None for sym in SYMBOLS}
telegram_update_offset = None
last_telegram_send = 0
shutdown_event = threading.Event()

# ========== UTILITIES ==========
def safe_request(func, max_retries=MAX_RETRIES, delay=RETRY_DELAY):
    """Wrapper for safe API requests with retries."""
    for attempt in range(max_retries):
        try:
            return func()
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise e
            print(f"[Request] Retry {attempt + 1}/{max_retries} after error: {e}")
            time.sleep(delay * (attempt + 1))
        except Exception as e:
            print(f"[Request] Unexpected error: {e}")
            raise e

def send_telegram_message(chat_id, text, parse_mode=None):
    """Send message to a Telegram chat id with rate limiting."""
    global last_telegram_send
    
    time_since_last = time.time() - last_telegram_send
    if time_since_last < TELEGRAM_RATE_LIMIT:
        time.sleep(TELEGRAM_RATE_LIMIT - time_since_last)
    
    def _send():
        url = f"{TELEGRAM_API_URL}/sendMessage"
        payload = {"chat_id": chat_id, "text": text}
        if parse_mode:
            payload["parse_mode"] = parse_mode
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
        return r
    
    try:
        safe_request(_send)
        last_telegram_send = time.time()
        return True
    except Exception as e:
        print(f"[Telegram] send error: {e}")
        return False

def fetch_klines(symbol, interval=TIMEFRAME, limit=BINANCE_KLINES_LIMIT):
    """Fetch klines from Binance with improved error handling."""
    def _fetch():
        return client.get_klines(symbol=symbol, interval=interval, limit=limit)
    
    try:
        data = safe_request(_fetch)
        if not data or len(data) < max(EMA_LONG, ATR_PERIOD + 10):
            print(f"[Binance] Insufficient data for {symbol}: got {len(data) if data else 0} candles")
            return None
            
        cols = ['open_time','open','high','low','close','volume','close_time','qav','trades','tb_base','tb_quote','ignore']
        df = pd.DataFrame(data, columns=cols)
        
        # Convert to numeric
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        
        # Validate data
        if df[['open', 'high', 'low', 'close']].isna().any().any():
            print(f"[Binance] Warning: Found NaN values in {symbol} price data")
            df = df.dropna(subset=['open', 'high', 'low', 'close'])
        
        return df if len(df) >= max(EMA_LONG, ATR_PERIOD + 10) else None
        
    except Exception as e:
        print(f"[Binance] error fetching klines for {symbol}: {e}")
        return None

def calculate_atr(df, period=ATR_PERIOD):
    """Calculate Average True Range (ATR)."""
    if df is None or len(df) < period + 1:
        return df
    
    try:
        # Calculate True Range components
        df['HL'] = df['high'] - df['low']
        df['HC'] = abs(df['high'] - df['close'].shift(1))
        df['LC'] = abs(df['low'] - df['close'].shift(1))
        
        # True Range is the maximum of the three
        df['TR'] = df[['HL', 'HC', 'LC']].max(axis=1)
        
        # ATR is the exponential moving average of True Range
        df['ATR'] = df['TR'].ewm(span=period, adjust=False).mean()
        
        # Clean up temporary columns
        df.drop(['HL', 'HC', 'LC', 'TR'], axis=1, inplace=True)
        
        return df
    except Exception as e:
        print(f"[ATR] calculation error: {e}")
        return df

def calculate_indicators(df, short_period=EMA_SHORT, long_period=EMA_LONG):
    """Calculate EMAs, ATR, and additional volatility indicators."""
    if df is None or len(df) < long_period:
        return None
    
    try:
        # Calculate EMAs
        df[f'EMA_{short_period}'] = df['close'].ewm(span=short_period, adjust=False).mean()
        df[f'EMA_{long_period}'] = df['close'].ewm(span=long_period, adjust=False).mean()
        
        # Calculate ATR
        df = calculate_atr(df)
        
        # Calculate volatility metrics for dynamic risk management
        df['price_change_pct'] = df['close'].pct_change().abs()
        df['volatility'] = df['price_change_pct'].rolling(window=VOLATILITY_LOOKBACK).std()
        df['atr_normalized'] = df['ATR'] / df['close']
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        return df
    except Exception as e:
        print(f"[Indicators] calculation error: {e}")
        return None

def calculate_risk_score(df):
    """Calculate a dynamic risk score based on market conditions."""
    try:
        # Get latest values
        current_volatility = df['volatility'].iloc[-1]
        current_atr_norm = df['atr_normalized'].iloc[-1]
        current_volume_ratio = df['volume_ratio'].iloc[-1]
        
        # Handle NaN values
        if pd.isna(current_volatility):
            current_volatility = df['volatility'].dropna().iloc[-1] if not df['volatility'].dropna().empty else 0.01
        if pd.isna(current_atr_norm):
            current_atr_norm = df['atr_normalized'].dropna().iloc[-1] if not df['atr_normalized'].dropna().empty else 0.01
        if pd.isna(current_volume_ratio):
            current_volume_ratio = 1.0
            
        # Calculate volatility percentile (higher = more volatile)
        vol_values = df['volatility'].dropna()
        vol_percentile = (vol_values <= current_volatility).sum() / len(vol_values) if len(vol_values) > 0 else 0.5
        
        # Calculate ATR percentile
        atr_values = df['atr_normalized'].dropna()
        atr_percentile = (atr_values <= current_atr_norm).sum() / len(atr_values) if len(atr_values) > 0 else 0.5
        
        # Volume factor (high volume = more reliable signal = lower risk)
        volume_factor = min(current_volume_ratio / 2.0, 1.0)  # Cap at 1.0
        
        # Combined risk score (0 = low risk, 1 = high risk)
        risk_score = (vol_percentile * 0.4 + atr_percentile * 0.4 + (1 - volume_factor) * 0.2)
        
        return max(0.0, min(1.0, risk_score))  # Clamp between 0 and 1
        
    except Exception as e:
        print(f"[Risk Score] Calculation error: {e}")
        return 0.5  # Default medium risk

def calculate_dynamic_multipliers(risk_score):
    """Calculate dynamic TP/SL multipliers based on risk score."""
    try:
        # High risk = smaller TP, tighter SL
        # Low risk = larger TP, wider SL
        
        # TP Multiplier: decreases with higher risk
        tp_multiplier = BASE_ATR_TP_MULTIPLIER * (1 + (0.5 - risk_score) * 0.75)
        tp_multiplier = max(MIN_TP_MULTIPLIER, min(MAX_TP_MULTIPLIER, tp_multiplier))
        
        # SL Multiplier: increases slightly with higher risk for protection
        sl_multiplier = BASE_ATR_SL_MULTIPLIER * (1 + risk_score * 0.5)
        sl_multiplier = max(MIN_SL_MULTIPLIER, min(MAX_SL_MULTIPLIER, sl_multiplier))
        
        return tp_multiplier, sl_multiplier
        
    except Exception as e:
        print(f"[Dynamic Multipliers] Calculation error: {e}")
        return BASE_ATR_TP_MULTIPLIER, BASE_ATR_SL_MULTIPLIER

def calculate_position_size(account_balance, entry_price, sl_price, risk_per_trade=RISK_PER_TRADE):
    """Calculate position size based on risk management."""
    try:
        # Calculate risk per share/unit
        risk_per_unit = abs(entry_price - sl_price)
        
        if risk_per_unit <= 0:
            return 0
            
        # Calculate total risk amount
        total_risk_amount = account_balance * risk_per_trade
        
        # Calculate position size
        position_size = total_risk_amount / risk_per_unit
        
        # Calculate position value
        position_value = position_size * entry_price
        
        return {
            'position_size': position_size,
            'position_value': position_value,
            'risk_amount': total_risk_amount,
            'risk_per_unit': risk_per_unit
        }
        
    except Exception as e:
        print(f"[Position Size] Calculation error: {e}")
        return {
            'position_size': 0,
            'position_value': 0,
            'risk_amount': 0,
            'risk_per_unit': 0
        }

def calculate_dynamic_tp_sl(entry_price, atr_value, signal_type, df):
    """Calculate dynamic take profit and stop loss based on market conditions."""
    try:
        # Calculate risk score
        risk_score = calculate_risk_score(df)
        
        # Get dynamic multipliers
        tp_multiplier, sl_multiplier = calculate_dynamic_multipliers(risk_score)
        
        # Calculate TP/SL prices
        if signal_type == "BUY":
            tp = entry_price + (atr_value * tp_multiplier)
            sl = entry_price - (atr_value * sl_multiplier)
        else:  # SELL
            tp = entry_price - (atr_value * tp_multiplier)
            sl = entry_price + (atr_value * sl_multiplier)
        
        # Calculate position sizing
        position_info = calculate_position_size(ACCOUNT_BALANCE, entry_price, sl)
        
        return {
            'tp_price': tp,
            'sl_price': sl,
            'tp_multiplier': tp_multiplier,
            'sl_multiplier': sl_multiplier,
            'risk_score': risk_score,
            'position_info': position_info
        }
        
    except Exception as e:
        print(f"[Dynamic TP/SL] Calculation error: {e}")
        # Fallback to static calculation
        if signal_type == "BUY":
            tp = entry_price + (atr_value * BASE_ATR_TP_MULTIPLIER)
            sl = entry_price - (atr_value * BASE_ATR_SL_MULTIPLIER)
        else:
            tp = entry_price - (atr_value * BASE_ATR_TP_MULTIPLIER)
            sl = entry_price + (atr_value * BASE_ATR_SL_MULTIPLIER)
            
        return {
            'tp_price': tp,
            'sl_price': sl,
            'tp_multiplier': BASE_ATR_TP_MULTIPLIER,
            'sl_multiplier': BASE_ATR_SL_MULTIPLIER,
            'risk_score': 0.5,
            'position_info': {'position_size': 0, 'position_value': 0, 'risk_amount': 0, 'risk_per_unit': 0}
        }

# Legacy function for backward compatibility
def calculate_tp_sl(entry_price, atr_value, signal_type):
    """Legacy function - use calculate_dynamic_tp_sl instead."""
    if signal_type == "BUY":
        tp = entry_price + (atr_value * BASE_ATR_TP_MULTIPLIER)
        sl = entry_price - (atr_value * BASE_ATR_SL_MULTIPLIER)
    else:
        tp = entry_price - (atr_value * BASE_ATR_TP_MULTIPLIER)
        sl = entry_price + (atr_value * BASE_ATR_SL_MULTIPLIER)
    
    return tp, sl

def check_and_send_signal(symbol):
    """Check EMA crossover and send ATR-based TP/SL signal."""
    global last_signal, last_candle_time
    
    try:
        df = fetch_klines(symbol)
        if df is None or len(df) < (EMA_LONG + 2):
            return
        
        df = calculate_indicators(df)
        if df is None:
            return
        
        # Check if we have valid indicator data
        ema_short_col = f'EMA_{EMA_SHORT}'
        ema_long_col = f'EMA_{EMA_LONG}'
        
        if ema_short_col not in df.columns or ema_long_col not in df.columns or 'ATR' not in df.columns:
            print(f"[Signal] Missing indicator columns for {symbol}")
            return
        
        # Get latest candle time
        latest_time = df['open_time'].iloc[-1]
        if last_candle_time.get(symbol) == latest_time:
            return
        
        # Check for crossover
        if len(df) < 2:
            return
            
        prev_short = df[ema_short_col].iloc[-2]
        prev_long = df[ema_long_col].iloc[-2]
        curr_short = df[ema_short_col].iloc[-1]
        curr_long = df[ema_long_col].iloc[-1]
        current_atr = df['ATR'].iloc[-1]
        
        # Validate values
        if pd.isna(prev_short) or pd.isna(prev_long) or pd.isna(curr_short) or pd.isna(curr_long) or pd.isna(current_atr):
            print(f"[Signal] Invalid indicator values for {symbol}")
            return
        
        signal = None
        if prev_short < prev_long and curr_short > curr_long:
            signal = "BUY"
        elif prev_short > prev_long and curr_short < curr_long:
            signal = "SELL"
        
        last_candle_time[symbol] = latest_time
        
        if signal and signal != last_signal.get(symbol):
            entry_price = df['close'].iloc[-1]
            
            # Use dynamic risk management
            risk_data = calculate_dynamic_tp_sl(entry_price, current_atr, signal, df)
            tp_price = risk_data['tp_price']
            sl_price = risk_data['sl_price']
            tp_mult = risk_data['tp_multiplier']
            sl_mult = risk_data['sl_multiplier']
            risk_score = risk_data['risk_score']
            position_info = risk_data['position_info']
            
            # Risk level indicator
            if risk_score < 0.3:
                risk_emoji = "🟢"
                risk_level = "LOW"
            elif risk_score < 0.7:
                risk_emoji = "🟡"
                risk_level = "MED"
            else:
                risk_emoji = "🔴"
                risk_level = "HIGH"
            
            # Simplified output - only TP and SL prices with risk info
            simple_msg = (
                f"🎯 {symbol} {signal} {risk_emoji}\n"
                f"Entry: {entry_price:.2f}\n"
                f"TP: {tp_price:.2f} ({tp_mult:.1f}x)\n"
                f"SL: {sl_price:.2f} ({sl_mult:.1f}x)"
            )
            
            # Detailed Telegram message with dynamic risk info
            detailed_msg = (
                f"{'📈' if signal == 'BUY' else '📉'} {signal} Signal {risk_emoji}\n"
                f"Coin: {symbol}\n"
                f"Entry: {entry_price:.2f} USDT\n"
                f"Take Profit: {tp_price:.2f} USDT ({tp_mult:.1f}x ATR)\n"
                f"Stop Loss: {sl_price:.2f} USDT ({sl_mult:.1f}x ATR)\n"
                f"ATR: {current_atr:.2f}\n"
                f"Risk Level: {risk_level} ({risk_score:.2f})\n"
                f"Position Size: {position_info['position_size']:.2f}\n"
                f"Risk Amount: ${position_info['risk_amount']:.2f}\n"
                f"Timeframe: {TIMEFRAME}"
            )
            
            # Send to Telegram
            success = send_telegram_message(CHAT_ID, detailed_msg)
            if not success:
                print(f"[Signal] Failed to send Telegram message for {symbol} {signal}")
            
            # Console output (simplified)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            console_line = f"{timestamp} - {simple_msg.replace(chr(10), ' | ')}"
            print(console_line)
            
            # File logging
            try:
                with open("atr_signals.txt", "a", encoding="utf-8") as f:
                    f.write(f"{timestamp} | {symbol} | {signal} | Entry: {entry_price:.2f} | TP: {tp_price:.2f} | SL: {sl_price:.2f} | ATR: {current_atr:.2f}\n")
            except Exception as e:
                print(f"[Log] Error writing to file: {e}")
            
            last_signal[symbol] = signal
            
    except Exception as e:
        print(f"[Signal] Unexpected error processing {symbol}: {e}")

def groq_chat_reply(user_text):
    """Send user_text to Groq with improved error handling."""
    if not user_text.strip():
        return "Please send a message."
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful trading assistant with knowledge of ATR-based TP/SL strategies. Keep responses concise and informative."},
            {"role": "user", "content": user_text}
        ],
        "max_tokens": 512,
        "temperature": 0.7
    }
    
    def _groq_request():
        r = requests.post(GROQ_CHAT_URL, json=payload, headers=headers, timeout=30)
        r.raise_for_status()
        return r
    
    try:
        r = safe_request(_groq_request)
        data = r.json()
        
        if "choices" in data and len(data["choices"]) > 0:
            msg = data["choices"][0].get("message", {}).get("content")
            if isinstance(msg, str) and msg.strip():
                return msg.strip()
        
        if "error" in data:
            error_msg = data["error"].get("message", "Unknown Groq API error")
            print(f"[Groq] API error: {error_msg}")
            return "Sorry, I encountered an API error. Please try again."
        
        return "Sorry, I couldn't get a proper response from Groq."
        
    except Exception as e:
        print(f"[Groq] error: {e}")
        return "Error: Unable to connect to AI service. Please try again later."

def telegram_polling_loop(poll_interval=2.0):
    """Telegram polling with better error handling."""
    global telegram_update_offset
    print("[Telegram] Polling loop started.")
    
    consecutive_errors = 0
    max_consecutive_errors = 5
    
    while not shutdown_event.is_set():
        try:
            params = {"timeout": 20}
            if telegram_update_offset:
                params["offset"] = telegram_update_offset
            
            r = requests.get(f"{TELEGRAM_API_URL}/getUpdates", params=params, timeout=25)
            r.raise_for_status()
            
            updates = r.json().get("result", [])
            consecutive_errors = 0
            
            for upd in updates:
                if shutdown_event.is_set():
                    break
                    
                telegram_update_offset = upd["update_id"] + 1
                msg = upd.get("message") or upd.get("edited_message")
                if not msg:
                    continue
                
                chat = msg.get("chat", {})
                chat_id = chat.get("id")
                user_text = msg.get("text", "")
                
                if not chat_id:
                    continue
                
                # Handle /start command
                if user_text and user_text.strip().lower() in ["/start", "start"]:
                    start_msg = (
                        "✅ Dynamic Risk Management TradeBot Activated\n"
                        f"Monitoring: {', '.join(SYMBOLS)}\n"
                        f"Timeframe: {TIMEFRAME}\n"
                        f"EMA: {EMA_SHORT}/{EMA_LONG}\n"
                        f"ATR Period: {ATR_PERIOD}\n"
                        f"Base TP: {BASE_ATR_TP_MULTIPLIER}x ATR (Dynamic: {MIN_TP_MULTIPLIER}-{MAX_TP_MULTIPLIER}x)\n"
                        f"Base SL: {BASE_ATR_SL_MULTIPLIER}x ATR (Dynamic: {MIN_SL_MULTIPLIER}-{MAX_SL_MULTIPLIER}x)\n"
                        f"Risk per Trade: {RISK_PER_TRADE*100}%\n"
                        f"Account Balance: ${ACCOUNT_BALANCE}\n\n"
                        "🎯 Features:\n"
                        "• Dynamic TP/SL based on volatility\n"
                        "• Automatic position sizing\n"
                        "• Risk level indicators\n"
                        "• Real-time market analysis\n\n"
                        "I will send TP/SL signals and answer trading questions."
                    )
                    send_telegram_message(chat_id, start_msg)
                    continue
                
                # Handle regular messages
                if user_text and user_text.strip():
                    send_telegram_message(chat_id, "⏳ Thinking...")
                    reply = groq_chat_reply(user_text)
                    send_telegram_message(chat_id, reply)
                    
        except requests.exceptions.ReadTimeout:
            continue
        except requests.exceptions.RequestException as e:
            consecutive_errors += 1
            print(f"[Telegram Polling] Network error ({consecutive_errors}/{max_consecutive_errors}): {e}")
            
            if consecutive_errors >= max_consecutive_errors:
                print("[Telegram Polling] Too many consecutive errors, stopping.")
                break
                
            time.sleep(min(poll_interval * consecutive_errors, 30))
            
        except Exception as e:
            consecutive_errors += 1
            print(f"[Telegram Polling] Unexpected error ({consecutive_errors}/{max_consecutive_errors}): {e}")
            
            if consecutive_errors >= max_consecutive_errors:
                print("[Telegram Polling] Too many consecutive errors, stopping.")
                break
                
            time.sleep(poll_interval)

def monitor_loop():
    """Monitoring loop with better error handling."""
    print("[Monitor] ATR-based signal monitor started.")
    consecutive_errors = 0
    max_consecutive_errors = 5
    
    while not shutdown_event.is_set():
        try:
            for sym in SYMBOLS:
                if shutdown_event.is_set():
                    break
                try:
                    check_and_send_signal(sym)
                except Exception as e:
                    print(f"[Monitor] Error processing {sym}: {e}")
            
            consecutive_errors = 0
            
            # Sleep with interruption check
            for _ in range(CHECK_INTERVAL_SECONDS):
                if shutdown_event.is_set():
                    break
                time.sleep(1)
                
        except Exception as e:
            consecutive_errors += 1
            print(f"[Monitor] Loop error ({consecutive_errors}/{max_consecutive_errors}): {e}")
            
            if consecutive_errors >= max_consecutive_errors:
                print("[Monitor] Too many consecutive errors, stopping.")
                break
                
            time.sleep(min(CHECK_INTERVAL_SECONDS, 30))

def signal_handler(signum, frame):
    """Handle shutdown gracefully."""
    print(f"\n[Shutdown] Received signal {signum}, shutting down...")
    shutdown_event.set()

def main():
    global client
    
    print("="*70)
    print("🎯 Dynamic Risk Management TradeBot Starting...")
    print(f"Monitoring: {SYMBOLS}")
    print(f"Timeframe: {TIMEFRAME}")
    print(f"EMA Periods: {EMA_SHORT}/{EMA_LONG}")
    print(f"ATR Period: {ATR_PERIOD}")
    print(f"Base TP: {BASE_ATR_TP_MULTIPLIER}x ATR (Dynamic: {MIN_TP_MULTIPLIER}-{MAX_TP_MULTIPLIER}x)")
    print(f"Base SL: {BASE_ATR_SL_MULTIPLIER}x ATR (Dynamic: {MIN_SL_MULTIPLIER}-{MAX_SL_MULTIPLIER}x)")
    print(f"Risk per Trade: {RISK_PER_TRADE*100}% | Account: ${ACCOUNT_BALANCE}")
    print(f"Volatility Lookback: {VOLATILITY_LOOKBACK} periods")
    print("🟢 Low Risk | 🟡 Med Risk | 🔴 High Risk")
    print("="*70)
    
    # Initialize Binance client
    try:
        client = Client(BINANCE_API_KEY, BINANCE_API_SECRET, {"timeout": 10})
        client.ping()
        print("[Binance] Connection successful")
    except Exception as e:
        print(f"[Binance] Failed to initialize client: {e}")
        sys.exit(1)
    
    # Setup signal handlers
    import signal
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start Telegram polling thread
    t_telegram = threading.Thread(target=telegram_polling_loop, daemon=True)
    t_telegram.start()
    
    time.sleep(2)
    
    # Send startup notification
    startup_msg = (
        "🚀 Dynamic Risk Management TradeBot Started\n"
        f"Symbols: {', '.join(SYMBOLS)}\n"
        f"Timeframe: {TIMEFRAME}\n"
        f"Dynamic TP: {MIN_TP_MULTIPLIER}-{MAX_TP_MULTIPLIER}x ATR\n"
        f"Dynamic SL: {MIN_SL_MULTIPLIER}-{MAX_SL_MULTIPLIER}x ATR\n"
        f"Risk Management: {RISK_PER_TRADE*100}% per trade\n"
        f"Check interval: {CHECK_INTERVAL_SECONDS}s\n\n"
        "🎯 Adaptive to market volatility & volume"
    )
    success = send_telegram_message(CHAT_ID, startup_msg)
    if success:
        print("[Info] Startup notification sent")
    else:
        print("[Info] Failed to send startup notification")
    
    try:
        monitor_loop()
    except KeyboardInterrupt:
        print("\n[Shutdown] Keyboard interrupt received")
    finally:
        print("[Shutdown] ATR TradeBot stopped.")
        shutdown_event.set()

if __name__ == "__main__":
    main()
