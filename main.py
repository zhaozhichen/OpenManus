import argparse
import asyncio

from app.agent.manus import Manus
from app.logger import logger


async def main():
    # Create and initialize Manus agent
    agent = await Manus.create(max_steps=10000)
    try:
        prompt = input("Enter your prompt: ")
        if not prompt.strip():
            logger.warning("Empty prompt provided.")
            return

        logger.warning("Processing your request...")
        await agent.run(prompt)
        logger.info("Request processing completed.")
    except KeyboardInterrupt:
        logger.warning("Operation interrupted.")
    finally:
        await agent.cleanup()
        logger.info("Manus agent cleanup completed.")


if __name__ == "__main__":
    asyncio.run(main())
