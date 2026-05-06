# main.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)
import os
import asyncio
from dotenv import load_dotenv
from data_ingestion import retriever
load_dotenv()


if not os.environ.get("GOOGLE_MODEL"):
    raise ValueError("GOOGLE_MODEL not set")
if not os.environ.get("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not set")

llm = ChatGoogleGenerativeAI(model=os.environ["GOOGLE_MODEL"])
# Add retry at the Runnable level
llm_with_retry = llm.with_retry(
        stop_after_attempt=3,
        retry_if_exception_type=(ValueError,ConnectionError,TimeoutError,Exception),
)
system_prompt = '''
You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Context: 
{context} 
'''
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

async def main():
    print("Hello from rag-with-ragas!")
    # res = await llm_with_retry.ainvoke(
    #     [HumanMessage(content="Hi")],
    #     config=RunnableConfig(max_concurrency=10)  #  built-in concurrency control
    # )
    # print(res.content)
    question_answer_chain = create_stuff_documents_chain(llm_with_retry, prompt)
    chain = create_retrieval_chain(retriever, question_answer_chain) 
    query = input("Enter your query :")
    resp = await chain.ainvoke(
        {"input": query},
        config=RunnableConfig(max_concurrency=10)
    )
    # print(resp)
    print("===============")
    print("\nAnswer:\n", resp["answer"])



if __name__ == "__main__":
    asyncio.run(main())
