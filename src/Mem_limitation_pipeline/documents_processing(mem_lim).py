import os
import re
from tqdm import tqdm

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings

from langchain_community.vectorstores import Chroma



class DocumentsProcessingDB:
    def __init__(self, path_dir, path_db,
                 chunk_sz=300, chunk_overlap=30) -> None:
        
        self.directory = path_dir
        self.db_path = path_db
        self.collection_name = "vectorized_documents"

        self.chunk_size = chunk_sz
        self.chunk_overlap = chunk_overlap
        self.text_splitter = CharacterTextSplitter(
            separator='.',
            chunk_size=chunk_sz,
            chunk_overlap=chunk_overlap
        )

        self.documents = []
        self.docs_split = []

        self.embedding_function = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base",
                                                      model_kwargs={"device": "cuda"})
        
        self.db = None
        self.retriever = None

    def start_db(self):
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)
            self.load_documents()
            self.split_documents()
            self.filter_noise_pages()
            self.populate_vector_db()
        else:
            self.load_db()

    def load_documents(self):
        files = os.listdir(self.directory)
        for file in tqdm(files, total=len(files), desc="Loading Documents"):
            print("Loading PDF: ", file)
            pdf_path = os.path.join(self.directory, file)
            loader = PyPDFLoader(pdf_path)
            self.documents.extend(loader.load())

    def filter_noise_pages(self):
        filtered_documents = []
        for doc in tqdm(self.docs_split, desc="Cleaning documents"):
            cleaned_text = self.clean_data(doc.page_content)
            doc.page_content = cleaned_text
            filtered_documents.append(doc)
        self.docs_split = filtered_documents

    def clean_data(self, text):
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text

    def split_documents(self):
        self.docs_split = self.text_splitter.split_documents(self.documents)
        

    def populate_vector_db(self):
        self.db = Chroma.from_documents(documents=self.docs_split, embedding=self.embedding_function, 
                                    persist_directory=self.db_path)
        self.db.persist()

        self.retriever = self.db.as_retriever(search_kargs={"k":3})

    def get_context_question(self, query):
        
        print("Question: ", query)
        return self.db.similarity_search(query)
    
    def load_db(self):
        self.db = Chroma(persist_directory=self.db_path,    
                            embedding_function=self.embedding_function)
        
        self.retriever = self.db.as_retriever(search_kargs={"k":3})
        
        print("Loading db", self.db)

if __name__ == "__main__":
    doc_reader = DocumentsProcessingDB("data/manuals/", 'data/vectorized_DB')
   
    doc_reader.start_db()
    result = doc_reader.get_context_question("How can I open the windows?")

    print("results: ", result)
