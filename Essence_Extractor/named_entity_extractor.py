import os
from dotenv import load_dotenv

from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from pydantic import BaseModel,Field
from typing_extensions import List

load_dotenv()


class Named_Entity_Extractor_Specs(BaseModel):
    named_entity: List= Field(description="A list of named entities from the provided text.")



def Named_Entity_Extractor(text):
    llm=ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=os.environ["GOOGLE_GEMINI_API_KEY"]
    ).with_structured_output(Named_Entity_Extractor_Specs)

    template="""
    You are provied with a text.
    Your job is to extract named entities from it.
    Text:
    {text}
    """

    prompt=ChatPromptTemplate.from_template(template=template,
                                           input_variable=["text"])

    named_entity_extractor_chain=prompt|llm

    result=named_entity_extractor_chain.invoke(
        {
            "text":text
        }
    )

    named_entities=result.named_entity

    return named_entities