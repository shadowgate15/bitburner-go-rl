"""Setup the Bitburner game training environment using Playwright."""

import asyncio
import signal
from pathlib import Path

from playwright.async_api import async_playwright

SAVE_FILE = Path("/app/bitburnerSave.json.gz")
BITBURNER_URL = "https://bitburner-official.github.io/"


async def main() -> None:
    """Load Bitburner and import the save file through the game UI."""
    stop_event = asyncio.Event()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop_event.set)

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        # Load the game.
        await page.goto(BITBURNER_URL)
        await page.wait_for_load_state("networkidle")

        # Open the Options menu.
        await page.get_by_role("button", name="Options").click()

        # Click "Import Game" which triggers a file chooser.
        async with page.expect_file_chooser() as fc_info:
            await page.get_by_role(
                "menuitem", name="Import Game"
            ).click()
        file_chooser = await fc_info.value
        await file_chooser.set_files(str(SAVE_FILE))

        # Wait for the Confirm button and click it.
        await page.get_by_role("button", name="Confirm").click()

        # The game reloads after import; wait for it to settle.
        await page.wait_for_load_state("networkidle")

        # Keep the browser alive for RL training interactions.
        # Exit cleanly on SIGTERM or SIGINT.
        await stop_event.wait()
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
