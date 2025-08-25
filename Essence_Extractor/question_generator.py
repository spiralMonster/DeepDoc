import os
from dotenv import load_dotenv

from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from pydantic import BaseModel,Field
from typing_extensions import Dict

load_dotenv()


class Question_Generator_Specs(BaseModel):
    questions: Dict =Field(description="""
    Generated questions for each named entity provided.
    """)


def Question_Generator(named_entities:list):
    llm=ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=os.environ["GOOGLE_GEMINI_API_KEY"]
        
    ).with_structured_output(Question_Generator_Specs)

    template="""
    You are provided with a list of named entities.
    For each entity, your job is to create a couple of questions regarding that entity such that we can get answer to those questions from       the text that belongs those entity.

    Note:
    - There should be atleast one question at most 3 question per each entity.
    - The question generated should be independent of other entities.

    Named Entities:
    {named_entities}
    """

    prompt=ChatPromptTemplate.from_template(template=template,
                                            input_variable=["named_entities"])

    question_generator_chain=prompt|llm

    result=question_generator_chain.invoke(
        {
            "named_entities":named_entities
        }
    )

    generated_questions=result.questions

    return generated_questions