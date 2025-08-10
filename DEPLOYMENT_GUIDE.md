# EMA ATR Trading Bot - Deployment Guide

## Fixed Issues

✅ **Removed hardcoded API keys** - Now uses environment variables only
✅ **Updated requirements.txt** - Added missing dependencies with specific versions
✅ **Added environment validation** - Bot will not start without proper configuration
✅ **Fixed nixpacks configuration** - Updated to handle Python dependencies properly

## Required Environment Variables

Before deploying, you need to set these environment variables:

```bash
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
GROQ_API_KEY=your_groq_api_key
```

## Deployment Steps

### Option 1: Using Docker

1. Build the Docker image:
   ```bash
   docker build -t trading-bot .
   ```

2. Run with environment variables:
   ```bash
   docker run -e BINANCE_API_KEY="your_key" \
              -e BINANCE_API_SECRET="your_secret" \
              -e TELEGRAM_BOT_TOKEN="your_token" \
              -e TELEGRAM_CHAT_ID="your_chat_id" \
              -e GROQ_API_KEY="your_groq_key" \
              trading-bot
   ```

### Option 2: Using Nixpacks (Railway, etc.)

1. Set environment variables in your deployment platform:
   - `BINANCE_API_KEY`
   - `BINANCE_API_SECRET`
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`
   - `GROQ_API_KEY`

2. Deploy the project - nixpacks will automatically use the `nixpacks.toml` configuration

### Option 3: Local Development

1. Create a `.env` file (NOT included in git):
   ```bash
   BINANCE_API_KEY=your_binance_api_key
   BINANCE_API_SECRET=your_binance_api_secret
   TELEGRAM_BOT_TOKEN=your_telegram_bot_token
   TELEGRAM_CHAT_ID=your_telegram_chat_id
   GROQ_API_KEY=your_groq_api_key
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the bot:
   ```bash
   python ema_atr_signals.py
   ```

## Common Nixpacks Issues Fixed

### 1. Missing Dependencies
- **Issue**: `ta-lib` and other packages not in requirements.txt
- **Fix**: Updated requirements.txt with all necessary packages and versions

### 2. Build Process
- **Issue**: Nixpacks couldn't detect proper Python setup
- **Fix**: Added explicit nixPkgs in nixpacks.toml for Python 3.11

### 3. API Keys in Code
- **Issue**: Security risk and deployment failure
- **Fix**: Removed all hardcoded values, using environment variables only

## Troubleshooting

If deployment still fails:

1. **Check logs** for specific error messages
2. **Verify environment variables** are properly set
3. **Test locally first** with the same environment setup
4. **Check platform limits** - some platforms have restrictions on long-running processes

## Security Notes

- Never commit API keys to version control
- Use secure environment variable storage
- Consider using secrets management for production
- Regularly rotate API keys
- Monitor for unauthorized access

## Bot Features

- **Dynamic Risk Management**: Adjusts TP/SL based on market volatility
- **Real-time Monitoring**: ETHUSDT and BTCUSDT on 15m timeframe
- **Telegram Integration**: Sends alerts and responds to questions
- **Position Sizing**: Calculates optimal position sizes based on risk
- **ATR-based Signals**: Uses Average True Range for entry/exit points
