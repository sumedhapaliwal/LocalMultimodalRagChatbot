import os
from typing import Iterable

from dotenv import load_dotenv
import chainlit as cl

from langchain_core.documents import Document as LCDocument
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableConfig
from langchain.schema import StrOutputParser
from langchain.callbacks.base import BaseCallbackHandler

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaLLM


# Load environment variables from .env
load_dotenv()
qdrant_url = os.getenv("QDRANT_URL_LOCALHOST")

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

llm = OllamaLLM(
    model="deepseek-llm:latest"
)


@cl.on_chat_start
async def on_chat_start():
    try:
        print("on_chat_start triggered")

        from langchain.prompts import ChatPromptTemplate
        from langchain.schema import StrOutputParser
        from langchain.schema.runnable import RunnablePassthrough

        template = """You are a document analysis expert. When tables are present in the context, preserve the structure and refer to specific rows/columns directly.

        Context:
        {context}

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        def format_docs(docs):
            return "\n\n".join([d.page_content for d in docs])

        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID)

        print(f"Qdrant URL: {qdrant_url}")
        print("Trying to load existing Qdrant collection...")

        vectorstore = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            collection_name="rag",  # Make sure this matches your `ingest.py`
            url=qdrant_url,
        )

        retriever = vectorstore.as_retriever()

        runnable = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        cl.user_session.set("runnable", runnable)
        print("Runnable set successfully")

    except Exception as e:
        print("❌ ERROR in on_chat_start():", e)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable
    msg = cl.Message(content="")

    # Callback handler to show document sources
    class PostMessageHandler(BaseCallbackHandler):
        def __init__(self, msg: cl.Message):
            BaseCallbackHandler.__init__(self)
            self.msg = msg
            self.sources = set()

        def on_retriever_end(self, documents, *, run_id, parent_run_id, **kwargs):
            for d in documents:
                source = d.metadata.get('source', 'unknown')
                page = d.metadata.get('page', 'N/A')
                self.sources.add((source, page))

        def on_llm_end(self, response, *, run_id, parent_run_id, **kwargs):
            if self.sources:
                sources_text = "\n".join([f"{source}#page={page}" for source, page in self.sources])
                self.msg.elements.append(
                    cl.Text(name="Sources", content=sources_text, display="inline")
                )

    # Stream the response from the chain
    async for chunk in runnable.astream(
        message.content,
        config=RunnableConfig(callbacks=[
            cl.LangchainCallbackHandler(),
            PostMessageHandler(msg)
        ]),
    ):
        await msg.stream_token(chunk)

    await msg.send()
