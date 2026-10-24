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

from langchain_core.prompts import PromptTemplate

from langchain.output_parsers.json import SimpleJsonOutputParser

json_prompt = PromptTemplate.from_template(
    "Return a JSON object with an `answer` key that answers the following question: {question}"
)

json_parser = SimpleJsonOutputParser()

# The parser reads the text of the chat model's reply, so the chain works the same way.
json_chain = json_prompt | llmModel | json_parser

response = json_chain.invoke({"question": "What is the biggest country?"})

print("What is the biggest country?")
print(response)

print("\n----------\n")


from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

# Define your desired data structure.
class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")
    
# Set up a parser
parser = JsonOutputParser(pydantic_object=Joke)

# Inject parser instructions into the prompt template.
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# Create a chain with the prompt and the parser
chain = prompt | chatModel | parser

response = chain.invoke({"query": "Tell me a joke."})

print("Tell me a joke in custom format defined by Pydantic:")
print(response)

print("\n----------\n")
