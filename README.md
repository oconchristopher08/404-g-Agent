# 404-g Agent 🤖

> Autonomous crypto intelligence agent that monitors real-time social sentiment, on-chain wallet activity, and emerging tokens to detect early market opportunities.

---

## Features

- **Sentiment Analysis** — Scans Twitter/X for crypto mentions and scores them with VADER NLP
- **On-Chain Monitoring** — Tracks whale wallet activity on Ethereum via Etherscan
- **Token Detection** — Monitors DexScreener for new/trending pairs with unusual volume or price action
- **Signal Merging** — Combines all sources into high-confidence alerts

---

## Project Structure

```
404-g-agent/
├── main.py                    # Entry point
├── config/
│   └── settings.py            # Env-based configuration
├── agents/
│   └── alpha_scout.py         # Core agent orchestrator
├── modules/
│   ├── sentiment/
│   │   └── analyzer.py        # Twitter sentiment scanner
│   ├── onchain/
│   │   └── wallet_monitor.py  # Whale wallet tracker
│   └── tokens/
│       └── detector.py        # DexScreener token detector
├── utils/
│   └── logger.py              # Loguru logger setup
├── tests/
│   └── test_agent.py          # Unit tests
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/oconchristopher08/404-g-agent.git
cd 404-g-agent

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Fill in your API keys in .env

# 5. Run the agent
python main.py
```

---

## Configuration

All settings are loaded from `.env`. See `.env.example` for the full list.

| Variable | Description |
|---|---|
| `TWITTER_API_KEY` | Twitter/X API credentials |
| `WEB3_PROVIDER_URL` | Infura/Alchemy RPC endpoint |
| `ETHERSCAN_API_KEY` | Etherscan API key |
| `DEXSCREENER_API_URL` | DexScreener base URL |
| `SCAN_INTERVAL_SECONDS` | How often to run a scan cycle |
| `SENTIMENT_THRESHOLD` | Min confidence score for alerts (0–1) |
| `WHALE_WALLET_MIN_USD` | Min USD value to flag a wallet tx |

---

## Running Tests

```bash
pytest tests/ -v --cov=.
```

---

## License

MIT
