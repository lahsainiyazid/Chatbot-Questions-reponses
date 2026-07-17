from pathlib import Path 
import re  
import json 
import os 
from langchain_community.document_loaders import Docx2txtLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
folder1=Path("/home/yazid/stage/data/traite/documents")
folder2=Path("/home/yazid/stage/data/traite2")
folders=[folder1,folder2]
documents=[]
for folder in folders:
    for file in folder.glob("*.docx"):
        try:
            loader=Docx2txtLoader(str(file))
            docs=loader.load()#docs because load returns  a list even tho we have just one document 
        except Exception as e:
            print(f"Error: {e}")
            continue 
        for doc in docs:
            try:
                doc.metadata["source"]=os.fsdecode(file.name)
                cleaned_content=re.sub(r'\n{3,}',"\n\n",doc.page_content)
                doc.page_content=cleaned_content.strip()
            except Exception as e:
                print(f"Error loading file {file.name} error {e}")
        documents.extend(docs)
splitter=RecursiveCharacterTextSplitter(chunk_size=1000,
                               chunk_overlap=150,
                               separators=["\n\n",
                                           "\n",
                                           ".",
                                           " ",
                                           ""])
chunks=splitter.split_documents(documents)
print(f"Total chunks:{len(chunks)}")
output=[]
for i,chunk in enumerate(chunks):
    updated_metadata={**chunk.metadata,
                      "id":i}
    output.append({"Content":chunk.page_content,"metadata":updated_metadata})
with open("complete_data_v3.json","w",encoding="utf-8") as f:
    json.dump(output,f,ensure_ascii=False,indent=2)
print("Saved chunks !")

