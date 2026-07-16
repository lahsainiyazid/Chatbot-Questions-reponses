from pathlib import Path 
import json 
from langchain_community.document_loaders import Docx2txtLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
folder=Path("/home/yazid/stage/data/traite/documents")
documents=[]
for file in folder.glob("*.docx"):
    try:
        loader=Docx2txtLoader(str(file))
        docs=loader.load()
        for doc in docs:
            doc.metadata["source"]=file.name 
        documents.extend(docs)
    except Exception as e:
        print(f"Error loading {file.name}:{e}")

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
for i ,chunk in enumerate(chunks):
    output.append({"id":i,"Content":chunk.page_content,"metadata":chunk.metadata})
with open("traite_v1_docs.json","w",encoding="utf-8") as f :
    json.dump(output,f,ensure_ascii=False,indent=2)
print("Saved chunks !")
