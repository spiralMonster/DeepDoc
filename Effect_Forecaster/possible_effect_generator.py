import os
from dotenv import load_dotenv

from langchain.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

from pydantic import BaseModel,Field
from typing_extensions import TypedDict,Annotated,List

load_dotenv()

class Statement_Effect_Pair(TypedDict):
    statement: Annotated[str,...,"The statement"]
    effect: Annotated[str,...,"The effect due to statement"]


class Possible_Effect_Specs(BaseModel):
    possible_effects: List[Statement_Effect_Pair]=Field(description="List of possible statement-effect pair")



def Possible_Effect_Generator(document_essence,context):
    llm=ChatMistralAI(
        model="mistral-small-latest",
        temperature=0.2,
        max_retries=2,
        api_key=os.environ["MISTRAL_API_KEY"]
        
    ).with_structured_output(Possible_Effect_Specs)

    template="""
    You are provided with the important points in the document.
    Your job is:
     - Try to predict the effect due to the statements.

    For Example:
     - [
      "Statement": "regarding a war"
      "Effect": "Rise in the inflation rate for the countries in war"
       ]

     - [
      "Statement": "Product launched by a company"
      "Effect": "Rise in the stock price of company", "Loss for the competetive company"
       ]

     - [
       "Statement": "A Policy introduced by a government"
       "Effect": "Oppostion party opposing the statement"
       ]

    You will also be provided with the context from the Internet. So use it accordingly.

    Note:
     - Generate at least three and at most five statement-effect pairs.

    Document Points:
    {document_points}

    Context from Internet:
    {context}
    
    """

    prompt=ChatPromptTemplate.from_template(template=template,
                                            input_variable=["document_points","context"])

    effect_chain=prompt|llm
    results=effect_chain.invoke({
        "document_points":document_essence,
        "context":context
    })

    statement_effect=results.possible_effects

    return statement_effect