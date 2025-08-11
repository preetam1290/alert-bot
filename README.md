# Dynamic Risk Management Trading Bot

A sophisticated trading bot that monitors EMA crossovers with ATR-based dynamic TP/SL calculations.

## Features

- **Dynamic Risk Management**: Automatically adjusts take profit and stop loss based on market volatility
- **EMA50/EMA200 Crossover Strategy**: Uses exponential moving averages for signal generation  
- **ATR-Based Position Sizing**: Uses Average True Range for dynamic risk calculations
- **Multi-Symbol Support**: Monitors ETHUSDT and BTCUSDT
- **Telegram Integration**: Real-time alerts and AI chat functionality
- **Risk Level Indicators**: Visual indicators for market risk levels (🟢🟡🔴)

## Configuration

The bot uses the following timeframes and parameters:
- **Timeframe**: 15 minutes
- **EMA Periods**: 50/200
- **ATR Period**: 14
- **Risk per Trade**: 10%
- **Account Balance**: $3.00 (configurable)

## Environment Variables Required

Set these environment variables in Railway:

- `BINANCE_API_KEY`: Your Binance API key
- `BINANCE_API_SECRET`: Your Binance API secret  
- `TELEGRAM_BOT_TOKEN`: Your Telegram bot token
- `TELEGRAM_CHAT_ID`: Your Telegram chat ID
- `GROQ_API_KEY`: Groq API key for AI chat (optional)

## Deployment to Railway

1. Create a new Railway project
2. Connect your GitHub repository or deploy from local files
3. Set the required environment variables
4. Railway will automatically detect the Python app and deploy

## Local Development

```bash
pip install -r requirements.txt
python ema_atr_signals.py
```

## Risk Management

The bot implements sophisticated risk management:
- Dynamic TP/SL multipliers based on market volatility
- Position sizing based on account balance and risk percentage
- Real-time risk score calculation
- Volatility-adjusted entry and exit points

## Signal Format

Telegram alerts include:
- Entry price and direction (BUY/SELL)
- Take Profit and Stop Loss levels
- ATR multipliers used
- Risk level and score
- Recommended position size
