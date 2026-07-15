import json
import re 
from pathlib import Path 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
def clean_text (text :str)->str:
    text=text.replace("\n"," ").replace("\r"," ")
    text=re.sub(r'[\x00-\x1f\x7f-\x9f]',"",text)
    text=re.sub(r'\s+',"",text).strip()
    return text 
BASE_DIR=Path("/home/yazid/stage/data/Question_a_traiter/pdfs")
all_pdf_paths=list(BASE_DIR.rglob("*.pdf"))
print(f"Found {len(all_pdf_paths)} pdf files!")
all_pages=[]
for path in all_pdf_paths:
    if path.parent.name==BASE_DIR:
        doc_date="unassigned"
    else:
        doc_date=path.parent.name
    try:
        loader=PyPDFLoader(str(path))
        pages=loader.load()
        for page in pages:
            page.page_content=clean_text(page.page_content)
            page.metadata["folder_date"]=doc_date 
            page.metadata["file_name"]=path.name 
        all_pages.extend(pages)
    except Exception as e:
        print(f"Failed to load page {path.name} error :{e}")
text_splitter=RecursiveCharacterTextSplitter(chunk_size=800,
                                             chunk_overlap=100,
                                             add_start_index=True)
chunks=text_splitter.split_documents(all_pages)
my_chunks=[{"page_content":chunk.page_content,
            "metadata":chunk.metadata} for chunk in chunks]
with open("chunks_questions_v1.json","w",encoding="utf-8") as f:
    json.dump(my_chunks,f,ensure_ascii=False,indent=2)




