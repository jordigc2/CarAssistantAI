import os
import re
from tqdm import tqdm

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from langchain_community.vectorstores import Chroma


class DocumentsProcessingDB:
    def __init__(self, path_dir, path_db, model_name='t5-base',
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

        self.embedding_function = SentenceTransformerEmbeddings(
                                        model_name="all-MiniLM-L6-v2")
        
        self.db = None

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
        self.db = Chroma.from_documents(self.docs_split, self.embedding_function, 
                                    persist_directory=self.db_path)
        self.db.persist()

    def get_context_question(self, query):
        
        return self.db.similarity_search(query)
    
    def load_db(self):
        self.db = Chroma(persist_directory=self.db_path,    
                            embedding_function=self.embedding_function)
        
        print("Loading db", self.db)

if __name__ == "__main__":


    #Unit tests
    doc_reader = DocumentsProcessingDB("data/manuals/processed", 'data/vectorized_DB')
   
    doc_reader.start_db()

    question="How should children wear the seat belt?"
    result = doc_reader.get_context_question(question)

    print("Question: ", question)
    print("-----------")
    print("results: ", result)


    # print("Input Document: ", doc_reader.documents[0])
    # print("-----------")
    # print("Split Document1: ", doc_reader.docs_split[0])
    # print("Split Document1 Size: ", len(doc_reader.docs_split[0].page_content))
    # print("Split Document2: ", doc_reader.docs_split[1])
    # print("Split Document2 Size: ", len(doc_reader.docs_split[1].page_content))

    
