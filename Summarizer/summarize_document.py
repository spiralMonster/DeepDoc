import os
from dotenv import load_dotenv

from transformers import AutoTokenizer,AutoModelWithLMHead


def Summarize_Document(document,min_length=100,max_length=125):
    print(f"[Document Summarization Pipeline started...]")

    tokenizer=AutoTokenizer.from_pretrained("T5-base")
    print(f"[INFO] Tokenizer Loaded")

    model=AutoModelWithLMHead.from_pretrained("T5-base",return_dict=True)
    print(f"[INFO] Model Loaded")

    inputs=tokenizer.encode(
        "summarize: "+document,
        return_tensors="pt",
        max_length=max_length,
        truncation=False
    )
    print(f"[INFO] Document Tokenized")

    output=model.generate(
        inputs,
        min_length=min_length,
        max_length=max_length
    )
    print(f"[INFO] Model Prediction Completed")

    summary=tokenizer.decode(output[0],skip_special_tokens=True)
    print(f"[INFO] Document Summarized")

    print(f"[Summarization Pipeline completed...]")

    summary=summary.capitalize()
    return summary


if __name__=="__main__":
    
    from pathlib import Path
    import numpy as np
    import pandas as pd

    data_store=Path(os.getcwd()).parent/"Data_Store"
    df=pd.read_csv(
        os.path.join(data_store,"doc_classification_data.csv")
    )

    document=df.iloc[0].Text
    print("The document is: ")
    print(document)

    doc_summary=Summarize_Document(document)

    print("Document Summary: ")
    print(doc_summary)


    
    
    