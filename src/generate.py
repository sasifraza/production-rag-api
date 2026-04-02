from dotenv import load_dotenv
import os
from src.retrieve import retrieve_docs
from langchain_openai import ChatOpenAI

load_dotenv()

def generate_answer(query: str):
    docs = retrieve_docs(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    prompt = f"""
You are an assistant answering questions based ONLY on the provided context.

Context:
{context}

Question:
{query}

Instructions:
- Answer only from the context
- Be concise
- If the context partially answers the question, give the best partial answer
- Only say "I do not know" if the context contains no relevant information

Answer:
"""

    response = llm.invoke(prompt)
    return response.content, docs


if __name__ == "__main__":
    answer, docs = generate_answer("What is the objective of this study?")

    print("\nAnswer:")
    print(answer)

    print("\nSources:")
    for i, doc in enumerate(docs, 1):
        print(f"\nSource {i}:")
        print(doc.page_content)