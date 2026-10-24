import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

# Original version: OpenAI() used the completions model gpt-3.5-turbo-instruct,
# and ChatOpenAI used gpt-3.5-turbo-0125. OpenAI has since retired both.
# Today we use a current chat model through ChatOpenAI.
from langchain_openai import ChatOpenAI

MODEL = "gpt-6-luna"  # small, fast and affordable current OpenAI model

# This version of langchain-openai sends temperature=0.7 by default,
# but recent OpenAI models only accept the default value (1).
# That is why each model below sets temperature=1 explicitly.
llmModel = ChatOpenAI(model=MODEL, temperature=1)

chatModel = ChatOpenAI(model=MODEL, temperature=1)


from langchain_core.prompts import ChatPromptTemplate

from langchain_core.prompts import FewShotChatMessagePromptTemplate

examples = [
    {"input": "hi!", "output": "¡hola!"},
    {"input": "bye!", "output": "¡adiós!"},
]

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an English-Spanish translator."),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)

chain = final_prompt | chatModel

response = chain.invoke({"input": "Who was JFK?"})

print("\n----------\n")

print("Translate: Who was JFK?")
print(response.content)

print("\n----------\n")
