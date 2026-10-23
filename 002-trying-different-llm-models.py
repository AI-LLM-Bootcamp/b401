import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from langchain_groq import ChatGroq

# Original version: llama3-70b-8192 and mixtral-8x7b-32768.
# Groq has since retired both. Below are two current production models.
# Check the up-to-date list at https://console.groq.com/docs/models
BIG_MODEL = "openai/gpt-oss-120b"
SMALL_MODEL = "openai/gpt-oss-20b"

# These models "think" before answering. With low reasoning effort they
# spend fewer tokens thinking and leave room for the answer itself.
REASONING = {"extra_body": {"reasoning_effort": "low"}}

bigChatModel = ChatGroq(
    model=BIG_MODEL,
    model_kwargs=REASONING,
)

smallChatModel = ChatGroq(
    model=SMALL_MODEL,
    model_kwargs=REASONING,
)

messages = [
    ("system", "You are an historian expert in the Kennedy family."),
    ("human", "How many members of the family died tragically?"),
]


def show(response):
    """Print the answer; if it comes back empty, show why."""
    if response.content.strip():
        print(response.content)
    else:
        print("(Empty answer)")
        print("finish_reason:", response.response_metadata.get("finish_reason"))
        print("token_usage:", response.response_metadata.get("token_usage"))


print("\n----------\n")

print(f"How many members of the family died tragically? - {BIG_MODEL} Response:")

print("\n----------\n")

bigResponse = bigChatModel.invoke(messages)

show(bigResponse)

print("\n----------\n")

print(f"How many members of the family died tragically? - {SMALL_MODEL} Response:")

print("\n----------\n")

smallResponse = smallChatModel.invoke(messages)

show(smallResponse)
