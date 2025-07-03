import os
from dotenv import load_dotenv
from langchain import hub
from langchain_openai import AzureChatOpenAI

# this may not be necessary if we start using LangSmith
import warnings
from langsmith.utils import LangSmithMissingAPIKeyWarning
warnings.filterwarnings("ignore", category=LangSmithMissingAPIKeyWarning)

# connect to the LLM
load_dotenv()
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    temperature=0
)

# we will want to tweak this default prompt based on testing
prompt = hub.pull("rlm/rag-prompt")

user_input = input("Please enter your input: ")
messages = prompt.invoke({"question": user_input, "context":""})
response = llm.invoke(messages)
print(response.content)