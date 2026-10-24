import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

# Original version: OpenAI() used the completions model gpt-3.5-turbo-instruct,
# which OpenAI has since retired. Today we use a chat model through ChatOpenAI.
from langchain_openai import ChatOpenAI

MODEL = "gpt-6-luna"  # small, fast and affordable current OpenAI model

# This version of langchain-openai sends temperature=0.7 by default,
# but recent OpenAI models only accept the default value (1).
# That is why each model below sets temperature=1 explicitly.

llmModel = ChatOpenAI(model=MODEL, temperature=1)

print("\n----------\n")

response = llmModel.invoke(
    "Tell me one fun fact about the Kennedy family."
)

print("Tell me one fun fact about the Kennedy family:")
print(response.content)

print("\n----------\n")

print("Streaming:")

for chunk in llmModel.stream(
    "Tell me one fun fact about the Kennedy family."
):
    print(chunk.content, end="", flush=True)

print("\n----------\n")

# The original created a "creative" model with temperature=0.9.
# Recent OpenAI models only accept temperature=1, so we use that.
creativeLlmModel = ChatOpenAI(model=MODEL, temperature=1)

response = creativeLlmModel.invoke(
    "Write a short 5 line poem about JFK"
)

print("Write a short 5 line poem about JFK:")
print(response.content)

print("\n----------\n")

chatModel = ChatOpenAI(model=MODEL, temperature=1)

messages = [
    ("system", "You are an historian expert in the Kennedy family."),
    ("human", "Tell me one curious thing about JFK."),
]
response = chatModel.invoke(messages)

print("Tell me one curious thing about JFK:")
print(response.content)

print("\n----------\n")

print("Streaming:")

for chunk in chatModel.stream(messages):
    print(chunk.content, end="", flush=True)

print("\n----------\n")
