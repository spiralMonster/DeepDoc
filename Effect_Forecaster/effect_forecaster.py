import sys
import os
sys.path.append(os.path.join(os.getcwd(),"..","Essence_Extractor"))


from query_generator import GenerateQueries
from gather_context_from_internet import GatherContext
from possible_effect_generator import Possible_Effect_Generator


def Effect_Forecaster(document_essence,entities_extracted):
    print(f"[Effect Forecaster Pipeline started...]")

    persons_mentioned=entities_extracted["persons"]
    organizations_mentioned=entities_extracted["organization"]
    places_mentioned=entities_extracted["places"]

    internet_queries=GenerateQueries(
        persons_mentioned=persons_mentioned,
        organizations_mentioned=organizations_mentioned,
        places_mentioned=places_mentioned
    )
    print(f"[INFO] Internet Queries Generated")

    internet_context=[]
    for query in internet_queries:
        context=GatherContext(query)
        context="\n".join(context)

        internet_context.append(context)

    print(f"[INFO] Context from Internet Gathered")

    statement_effect_pair=Possible_Effect_Generator(
        document_essence=document_essence,
        context=internet_context
    )

    print(f"[INFO] Statement-Effect pair generated")

    print(f"[Effect Forecaster Pipeline Ended]")

    return statement_effect_pair

    
   
