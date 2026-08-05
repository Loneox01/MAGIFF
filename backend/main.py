import time

from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from prompts import SYSTEM_PROMPT

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

client = OpenAI()


def main() -> None:
    print("Fantasy agent ready. Type 'exit' to quit.\n")

    while True:
        try:
            prompt = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if prompt.lower() in {"exit", "quit"}:
            break

        if not prompt:
            continue

        started_at = time.perf_counter()

        response = client.responses.create(
            model="gpt-5.6-terra",
            instructions=(
                SYSTEM_PROMPT
            ),
            # tools=[
            #     {
            #         "type": "web_search",
            #         "filters": {
            #             "allowed_domains": [
            #                 "nfl.com",
            #                 "espn.com",
            #                 "cbssports.com",
            #                 "profootballtalk.nbcsports.com",
            #             ]
            #         },
            #     }
            # ],
            input=prompt,
        )

        elapsed = time.perf_counter() - started_at

        print(f"\nAgent: {response.output_text}")
        print(f"\nLatency: {elapsed:.2f}s")

        if response.usage:
            print(f"Input tokens: {response.usage.input_tokens}")
            print(f"Output tokens: {response.usage.output_tokens}")

        print()


if __name__ == "__main__":
    main()