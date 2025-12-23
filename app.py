import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def load_documents(file_path: str):
    """Load text file and convert to LangChain documents"""
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(text)
    documents = [Document(page_content=chunk) for chunk in chunks]
    return documents

def create_vector_store(documents):
    """Create FAISS vector store from documents"""
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

def main():
    print("\n🔹 AI Knowledge Assistant (Semantic Search)")
    print("Type 'exit' to quit\n")

    # Load and process documents
    documents = load_documents("data/sample_docs.txt")
    vector_store = create_vector_store(documents)

    # Initialize LLM
    llm = ChatOpenAI(temperature=0)

    while True:
        query = input("Ask a question: ")

        if query.lower() == "exit":
            print("Goodbye!")
            break

        # Retrieve relevant documents
        docs = vector_store.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in docs])

        # Prompt
        prompt = f"""
        Answer the question using only the context below.

        Context:
        {context}

        Question:
        {query}
        """

        response = llm.invoke(prompt)
        print("\nAnswer:")
        print(response.content)
        print("-" * 50)

if __name__ == "__main__":
    main()
