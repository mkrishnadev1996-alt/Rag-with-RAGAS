# main.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

model_name = os.getenv("GOOGLE_MODEL")
if not model_name:
    raise ValueError("GOOGLE_MODEL not set in environment")
llm = ChatGoogleGenerativeAI(model=os.environ["GOOGLE_MODEL"])


async def main():
    print("Hello from rag-with-ragas!")
    res = await llm.ainvoke([
        HumanMessage(content="Hi")
        ]
    )
    print(res.content)


if __name__ == "__main__":
    asyncio.run(main())
