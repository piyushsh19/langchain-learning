"""
Real-time currency conversion with LangChain tools.

Install:
    pip install -U langchain-core langchain-openai requests

Set environment variables:
    export OPENAI_API_KEY="your-openai-api-key"
    export EXCHANGE_RATE_API_KEY="your-exchangerate-api-key"

Run:
    python currency_conversion_tool.py

ExchangeRate-API endpoint:
    https://v6.exchangerate-api.com/v6/YOUR-API-KEY/pair/USD/INR
"""

import json
import os
from typing import Annotated, Any

import requests
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import InjectedToolArg, tool
from langchain_openai import ChatOpenAI


@tool
def get_conversion(base_currency: str, target_currency: str) -> dict[str, Any]:
    """Fetch the latest exchange rate for two currencies, such as USD and INR."""
    api_key = os.getenv("EXCHANGE_RATE_API_KEY")
    if not api_key:
        raise ValueError("Set the EXCHANGE_RATE_API_KEY environment variable.")

    base = base_currency.strip().upper()
    target = target_currency.strip().upper()
    url = (
        f"https://v6.exchangerate-api.com/v6/{api_key}/pair/"
        f"{base}/{target}"
    )

    response = requests.get(url, timeout=20)
    response.raise_for_status()
    data = response.json()

    if data.get("result") != "success":
        raise ValueError(f"ExchangeRate-API error: {data.get('error-type', data)}")

    return {
        "base_code": data["base_code"],
        "target_code": data["target_code"],
        "conversion_rate": data["conversion_rate"],
        "time_last_update_utc": data.get("time_last_update_utc"),
    }


@tool
def convert(
    base_currency_value: float,
    conversion_rate: Annotated[float, InjectedToolArg],
) -> float:
    """Convert an amount using the rate fetched by get_conversion."""
    return round(base_currency_value * conversion_rate, 2)


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Set the OPENAI_API_KEY environment variable.")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_with_tools = llm.bind_tools([get_conversion, convert])

    messages = [
        HumanMessage(
            content=(
                "Fetch the current USD-to-INR exchange rate, then convert 10 USD "
                "to INR using the convert tool. Include the rate and converted amount."
            )
        )
    ]

    conversion_rate = None
    base_code = target_code = None

    # Let the model request tools, execute them, and send results back.
    for _ in range(5):
        ai_message = llm_with_tools.invoke(messages)
        messages.append(ai_message)

        if not ai_message.tool_calls:
            print(ai_message.content)
            return

        # If both are requested together, fetch the rate before converting.
        calls = sorted(
            ai_message.tool_calls,
            key=lambda call: 0 if call["name"] == "get_conversion" else 1,
        )

        for call in calls:
            args = dict(call["args"])

            if call["name"] == "get_conversion":
                result = get_conversion.invoke(args)
                conversion_rate = float(result["conversion_rate"])
                base_code = result["base_code"]
                target_code = result["target_code"]
                result_text = json.dumps(result)

            elif call["name"] == "convert":
                if conversion_rate is None:
                    result_text = (
                        "No exchange rate is available yet. Call get_conversion first."
                    )
                else:
                    # InjectedToolArg is hidden from the model, so supply it here.
                    args["conversion_rate"] = conversion_rate
                    amount = float(args["base_currency_value"])
                    converted = convert.invoke(args)
                    result_text = json.dumps({
                        "base_currency_value": amount,
                        "base_currency": base_code,
                        "target_currency": target_code,
                        "conversion_rate": conversion_rate,
                        "converted_amount": converted,
                    })
            else:
                result_text = f"Unknown tool: {call['name']}"

            messages.append(
                ToolMessage(content=result_text, tool_call_id=call["id"])
            )

    raise RuntimeError("Stopped after 5 tool-calling rounds.")


if __name__ == "__main__":
    main()
