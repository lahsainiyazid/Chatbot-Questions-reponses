from pathlib import Path
import re  
import json
import os
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Option A: Using a raw string (recommended for Windows paths)
folder1 = Path(r"C:\Users\HP\Desktop\stage\traite")
print(folder1)
print(folder1.exists)
# Option B: Using forward slashes (also works natively on Windows)
# folder1 = Path("C:/Users/yazid/stage/data/traite/documents")

documents = []

for file in folder1.glob("*.docx"):
    try:
        loader = Docx2txtLoader(str(file))
        docs = loader.load()
    except Exception as e:
        print(f"Error: {e}")
        continue
   
    for doc in docs:
        try:
            doc.metadata["source"] = os.fsdecode(file.name)
            cleaned_content = re.sub(r'\n{3,}', "\n\n", doc.page_content)
            doc.page_content = cleaned_content.strip()
        except Exception as e:
            print(f"Error loading file {file.name} error {e}")
           
    documents.extend(docs)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", " ", ""]
)

chunks = splitter.split_documents(documents)
print(f"Total chunks:{len(chunks)}")

output = []
for i, chunk in enumerate(chunks):
    updated_metadata = {**chunk.metadata, "id": i}
    output.append({"Content": chunk.page_content, "metadata": updated_metadata})

output_file = folder1 / "complete_windows.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"Saved to {output_file}")
