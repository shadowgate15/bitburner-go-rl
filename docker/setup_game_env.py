"""Setup the Bitburner game training environment using Playwright."""

import asyncio
import base64
import signal
from pathlib import Path

from playwright.async_api import async_playwright

SAVE_DATA_FILE = Path("/app/save_data.b64")
BITBURNER_URL = "https://bitburner-official.github.io/"


async def main() -> None:
    """Load Bitburner and populate IndexDB with the baked-in save data."""
    raw = SAVE_DATA_FILE.read_text().strip()
    save_bytes: list[int] = list(base64.b64decode(raw)) if raw else []

    stop_event = asyncio.Event()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop_event.set)

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        # Navigate to the game so its origin is established before
        # touching IndexDB (IndexDB is scoped to the page origin).
        await page.goto(BITBURNER_URL)
        await page.wait_for_load_state("networkidle")

        # Open (or create) the IndexDB database and store the save data.
        await page.evaluate(
            """
            async (saveBytes) => {
                const db = await new Promise((resolve, reject) => {
                    const request = indexedDB.open('bitburnerSave', 1);
                    request.onupgradeneeded = (event) => {
                        const db = event.target.result;
                        if (!db.objectStoreNames.contains('savestring')) {
                            db.createObjectStore('savestring');
                        }
                    };
                    request.onsuccess =
                        (event) => resolve(event.target.result);
                    request.onerror = (event) => reject(event.target.error);
                });
                const tx = db.transaction('savestring', 'readwrite');
                const store = tx.objectStore('savestring');
                store.put(new Uint8Array(saveBytes), 'save');
                await new Promise((resolve, reject) => {
                    tx.oncomplete = resolve;
                    tx.onerror = (event) => reject(event.target.error);
                });
                db.close();
            }
            """,
            save_bytes,
        )

        # Reload so the game reads the newly written save on startup.
        await page.reload()
        await page.wait_for_load_state("networkidle")

        # Keep the browser alive for RL training interactions.
        # Exit cleanly on SIGTERM or SIGINT.
        await stop_event.wait()
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
