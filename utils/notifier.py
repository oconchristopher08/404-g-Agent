"""
Notifier — delivers formatted alerts to a Telegram chat via the Bot API.
Falls back to logger-only when credentials are not configured.
"""

import aiohttp
from loguru import logger

_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


class Notifier:
    def __init__(self, bot_token: str, chat_id: str):
        self._token = bot_token
        self._chat_id = chat_id
        self._enabled = bool(bot_token and chat_id)
        if not self._enabled:
            logger.warning(
                "Telegram not configured (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID missing) "
                "— alerts will only appear in logs."
            )

    async def send_alert(self, signal: dict) -> None:
        """Format a signal dict and deliver it as a Telegram message."""
        message = self._format(signal)
        logger.info("🚨 ALERT: {}", message)
        if self._enabled:
            await self._post(message)

    def _format(self, signal: dict) -> str:
        source = signal.get("source", "unknown").upper()
        token = signal.get("token", "?")
        confidence = signal.get("confidence", 0)
        conf_pct = f"{confidence * 100:.0f}%"

        lines = [f"*[{source}] {token}* — confidence {conf_pct}"]

        if source == "DEXSCREENER":
            chain = signal.get("chain", "")
            price = signal.get("price_usd", "?")
            vol = signal.get("volume_24h")
            liq = signal.get("liquidity_usd")
            change = signal.get("price_change_24h")
            reasons = signal.get("reasons", [])
            url = signal.get("dexscreener_url", "")

            if chain:
                lines.append(f"Chain: `{chain}`")
            if price:
                lines.append(f"Price: `${price}`")
            if change is not None:
                lines.append(f"24h change: `{change}%`")
            if vol:
                lines.append(f"Volume 24h: `${float(vol):,.0f}`")
            if liq:
                lines.append(f"Liquidity: `${float(liq):,.0f}`")
            if reasons:
                lines.append("Signals: " + " · ".join(reasons))
            if url:
                lines.append(f"[View on DexScreener]({url})")

        elif source == "ONCHAIN":
            tx_type = signal.get("type", "transfer")
            wallet = signal.get("wallet", "?")
            value_usd = signal.get("value_usd")
            tx_hash = signal.get("tx_hash", "")
            to_addr = signal.get("to", "")

            lines.append(f"Type: `{tx_type}`")
            lines.append(f"Wallet: `{wallet[:10]}…`")
            if to_addr:
                lines.append(f"To: `{to_addr[:10]}…`")
            if value_usd:
                lines.append(f"Value: `${float(value_usd):,.0f}`")
            if tx_hash:
                lines.append(f"[View on Etherscan](https://etherscan.io/tx/{tx_hash})")

        elif source == "TWITTER":
            sentiment = signal.get("sentiment", "neutral")
            text = signal.get("text", "")
            likes = signal.get("likes", 0)
            rts = signal.get("retweets", 0)

            lines.append(f"Sentiment: `{sentiment}`")
            if likes or rts:
                lines.append(f"Engagement: {likes} likes · {rts} RTs")
            if text:
                lines.append(f'_"{text[:120]}"_')

        return "\n".join(lines)

    async def _post(self, text: str) -> None:
        url = _TELEGRAM_API.format(token=self._token)
        payload = {
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        logger.warning("Telegram delivery failed ({}): {}", resp.status, body[:200])
        except Exception as e:
            logger.error("Telegram post error: {}", e)
