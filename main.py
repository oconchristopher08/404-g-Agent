"""
404-g Agent — AlphaScout Crypto Intelligence Agent
Entry point: boots all modules and starts the scan loop.
"""

import asyncio
from loguru import logger
from config.settings import Settings
from agents.alpha_scout import AlphaScoutAgent
from utils.logger import setup_logger


async def main():
    settings = Settings()
    setup_logger(settings.LOG_LEVEL)
    logger.info("🚀 404-g Agent starting up...")

    agent = AlphaScoutAgent(settings)
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
