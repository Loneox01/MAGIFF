import json
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from prompts import SYSTEM_PROMPT
from tools.nfl import TOOL_HANDLERS
from tools.schemas import NFL_TOOLS


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

client = OpenAI()

TOOLS = [
    *NFL_TOOLS,
    # {
    #     "type": "web_search",
    #     "filters": {
    #         "allowed_domains": [
    #             "nfl.com",
    #             "espn.com",
    #             "cbssports.com",
    #             "profootballtalk.nbcsports.com",
    #         ]
    #     },
    # },
]


def run_agent(prompt: str):
    """Run model/tool turns until the model returns a final answer."""
    input_items = [{"role": "user", "content": prompt}]
    total_input_tokens = 0
    total_output_tokens = 0

    for _ in range(8):
        response = client.responses.create(
            model="gpt-5.6-terra",
            instructions=SYSTEM_PROMPT,
            tools=TOOLS,
            input=input_items,
        )

        if response.usage:
            total_input_tokens += response.usage.input_tokens
            total_output_tokens += response.usage.output_tokens

        # Preserve function calls and any reasoning items for the next model turn.
        input_items += response.output
        tool_calls = [
            item for item in response.output if item.type == "function_call"
        ]

        if not tool_calls:
            return response, total_input_tokens, total_output_tokens

        for call in tool_calls:
            print(f"\nTool: {call.name}({call.arguments})")

            try:
                arguments = json.loads(call.arguments)
                handler = TOOL_HANDLERS[call.name]
                result = handler(**arguments)
            except Exception as error:
                result = {"error": str(error)}

            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(result, default=str),
                }
            )

    raise RuntimeError("Tool-call limit reached before a final response")


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

        try:
            response, input_tokens, output_tokens = run_agent(prompt)
        except Exception as error:
            print(f"\nError: {error}\n")
            continue

        elapsed = time.perf_counter() - started_at

        print(f"\nAgent: {response.output_text}")
        print(f"\nLatency: {elapsed:.2f}s")
        print(f"Input tokens: {input_tokens}")
        print(f"Output tokens: {output_tokens}")
        print()


if __name__ == "__main__":
    main()
