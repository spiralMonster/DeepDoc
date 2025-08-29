import os
from dotenv import load_dotenv

from langchain.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

from pydantic import BaseModel,Field
from typing_extensions import List

load_dotenv()


class Question_Generator_Specs(BaseModel):
    questions: List =Field(description="The generated questions.")


def Question_Generator(named_entities:list):
    llm=ChatMistralAI(
        model="mistral-small-latest",
        temperature=0,
        max_retries=2,
        api_key=os.environ["MISTRAL_API_KEY"]
    ).with_structured_output(Question_Generator_Specs)

    template="""
    You are provided with a list of named entities.
    These named entities are extracted from a document.
    So your job is to generate possible questions regarding these entities such that we get the answer to those questions from the original      document.

    Example:
     Suppose the named entities extracted from a document are: [Google,India,AI]
     Then the possible questions for whom we can get the answer from the doc could be:
      - What is the Google's next project in India?
      - Is India ready for AI?
      - Is Google winning the AI race?

    Note:
    - There should be at least three questions and at most 5 questions.
     

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


if __name__=="__main__":
    named_entities=[
        "AI-threat",
        "Geoffrey Hinton",
        "Nobel Prize"
    ]

    generated_questions=Question_Generator(named_entities)
    print(generated_questions)

