import os
from dotenv import load_dotenv

from langchain.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

from pydantic import BaseModel,Field
from typing_extensions import List

load_dotenv()


class QueryGeneratorSpecs(BaseModel):
    queries: List[str]=Field(description="The generated queries.")


def GenerateQueries(persons_mentioned,organizations_mentioned,places_mentioned):
    llm=ChatMistralAI(
        model="mistral-small-latest",
        temperature=0.2,
        max_retries=2,
        api_key=os.environ["MISTRAL_API_KEY"]
        
    ).with_structured_output(QueryGeneratorSpecs)

    template="""
    You are provided with the name of persons,organizations and places from the document.
    Your job is to generate queries using those that can be used to search on the Internet.

    Note:
     - The query should be 5-7 words long.
     - Generate at least 3 and at most 5 queries.

    Persons Mentioned: {persons_mentioned}
    Organizations Mentioned : {organizations_mentioned}
    Places Mentioned: {places_mentioned}

    """

    prompt=ChatPromptTemplate.from_template(template=template,
                                            input_variable=["persons_mentioned","organizations_mentioned","places_mentioned"])

    query_gen_chain=prompt|llm
    results=query_gen_chain.invoke({
        "persons_mentioned":persons_mentioned,
        "organizations_mentioned":organizations_mentioned,
        "places_mentioned":places_mentioned
    })

    queries=results.queries

    return queries