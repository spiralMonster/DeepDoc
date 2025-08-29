import os
from dotenv import load_dotenv

from langchain.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

from pydantic import BaseModel,Field

load_dotenv()

class Context_Creator_Specs(BaseModel):
    context_document: str= Field(description="The document created.")


def Context_Creator(generated_questions):
    llm=ChatMistralAI(
        model="mistral-small-latest",
        temperature=0.0,
        max_retries=2,
        api_key=os.environ["MISTRAL_API_KEY"]
        
    ).with_structured_output(Context_Creator_Specs)

    template="""
    You are provided with a list of questions.
    Your job is to create a document of text from those questions.
    The main aim of doing so, is to use the created document to retrieve the results from vectorstore.

    Note:
     -The length of the document should be at least 25 words and at max 50 words.

    List of questions:
    {questions}
    """

    prompt=ChatPromptTemplate.from_template(template=template,
                                           input_variable=["questions"])

    context_creator_chain=prompt|llm

    results=context_creator_chain.invoke(
        {
            "questions":generated_questions
        }
    )

    context_doc=results.context_document

    return context_doc