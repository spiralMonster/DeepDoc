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

    # internet_queries=GenerateQueries(
    #     persons_mentioned=persons_mentioned,
    #     organizations_mentioned=organizations_mentioned,
    #     places_mentioned=places_mentioned
    # )
    # print(f"[INFO] Internet Queries Generated")

    # internet_context=[]
    # for query in internet_queries:
    #     context=GatherContext(query)
    #     context="\n".join(context)

    #     internet_context.append(context)

    print(f"[INFO] Context from Internet Gathered")

    statement_effect_pair=Possible_Effect_Generator(
        document_essence=document_essence,
        context=[]
    )

    print(f"[INFO] Statement-Effect pair generated")

    print(f"[Effect Forecaster Pipeline Ended]")

    return statement_effect_pair


if __name__=="__main__":
    sys.path.append(os.path.join(os.getcwd(),"..","Essence_Extractor"))
    sys.path.append(os.path.join(os.getcwd(),"..","Entity_Extractor"))
    
    from pathlib import Path
    import pandas as pd

    from extract_document_essence import Extract_Document_Essence
    from extract_entities import Extract_Entities

    data_store=Path(os.getcwd()).parent/"Data_Store"

    df=pd.read_csv(os.path.join(data_store,"doc_classification_data.csv"))
    
    document=df.iloc[0].Text
    print("The document is: ")
    print(document)

    doc_essence=Extract_Document_Essence(document)
    important_points=[]
    print("Document Essence:")
    for ind,point in enumerate(doc_essence):
        point=point.page_content
        point=point.strip()
        point=point.capitalize()
        point=f"{ind+1}. "+point

        print(point)
        important_points.append(point)

    extracted_entities=Extract_Entities(document)
    print("Extracted Entities:")
    print(extracted_entities)

    statement_effect_pair=Effect_Forecaster(
        document_essence=important_points,
        entities_extracted=extracted_entities
    )

    for inst in statement_effect_pair:
        print(f"Statement: {inst['statement']}")
        print(f"Effect: {inst['effect']}")

        print("\n")

    

    
   
