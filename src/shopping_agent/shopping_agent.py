from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:8b"

# --- Tools (Langchain @tool decorator) ---

@tool
def get_product_price(product: str) -> float:
    """Get the price of a product."""
    # In a real application, this function would query a database or an API.
    # Here, we return a mock price for demonstration purposes.
    mock_prices = {
        "laptop": 999.99,
        "smartphone": 699.99,
        "headphones": 199.99,
        "monitor": 299.99,
    }
    print(f"Fetching price for product: {product}")
    return mock_prices.get(product.lower(), 0.0)

@tool
def apply_discount(price: float, discount_percentage: float) -> float:
    """Apply a discount to the price."""
    discounted_price = price * (1 - discount_percentage / 100)
    print(f"Applying discount: {discount_percentage}% to price: {price}, discounted price: {discounted_price}")
    return discounted_price

# --- Agent Loop ---

@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools=[get_product_price, apply_discount]
    tools_dict = {tool.name: tool for tool in tools}

    llm = init_chat_model(MODEL, model_provider="ollama", temperature=0.2)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Questions: {question}")
    print("="*60)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant."
                "You have access to a product price tool and a discount tool."
                "STRICT RULES - you must follow these exactly:\n"
                "1. Never guess or assume any product price."
                "You MUST call get_product_price to get the real price of a product.\n."
                "2. Only call apply_discount after you have obtained "
                "the real price from get_product_price. Pass the exact price "
                "returned by get_product_price - do NOT pass a made-up number.\n"
                "3. Never calculate discount yourself using math."
                "always use the apply_discount tool to calculate the discounted price.\n"
                "4. If user does not specify discount percent"
                " ask them the percent - do NOT assume a default value.\n"
            )
        ),
        HumanMessage(content=question),
    ] 

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n -- Iteration {iteration} ---")

        ai_message = llm_with_tools.invoke(messages)

        tool_calls = ai_message.tool_calls

        # If no tool calls, this is the final answer
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content
        
        #Process only the first tool call - force one tool per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args",{})
        tool_call_id = tool_call.get("id")

        print(f" [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if(tool_to_use) is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        observation = tool_to_use.invoke(tool_args)

        print(f"  [Tool Result] {observation}")

        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )

    print("ERROR: Max iterations reached without final answer")
    return None


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop after applying 10percent discount?")


def main():
    print("Hello LangChain Agent (.bind_tools)!")
    print()

    result = run_agent(
        "What is the price of a laptop after applying 10percent discount?"
    )


if __name__ == "__main__":
    main()