import os
from dotenv import load_dotenv

from langchain.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

from pydantic import BaseModel,Field

load_dotenv()

class AskChatbotSpecs(BaseModel):
    response: str=Field(description="The response to the question asked")


def AskChatBot(question,context,chat_history):
    llm=ChatMistralAI(
        model="mistral-small-latest",
        temperature=0.3,
        max_retries=2,
        api_key=os.environ["MISTRAL_API_KEY"]
        
    ).with_structured_output(AskChatbotSpecs)

    template="""
    You are provided with a question,context to answer question and history of messages.
    Your job is to answer the question.

    Question:{question}
    Context:
    {context}
    History of Messages:
    {chat_history}
    """

    prompt=ChatPromptTemplate.from_template(template=template,
                                            input_variable=["question","context","chat_history"])

    chain=prompt|llm

    result=chain.invoke({
        "question":question,
        "context":context,
        "chat_history":chat_history
    })

    answer=result.response

    return answer