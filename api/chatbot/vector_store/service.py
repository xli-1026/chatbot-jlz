from typing import List, Optional

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.utilities.sql_database import SQLDatabase
from langchain.vectorstores.redis import Redis


embeddings = HuggingFaceEmbeddings()


def init_vectorstore(
    chat_id: str,
    user_id: str,
    database: SQLDatabase,
):
    tables = database.get_usable_table_names()
    docs = [
        Document(
            page_content=database.get_table_info([table]),
            metadata={"table_name": table},
        )
        for table in tables
    ]
    """
    text_splitter = RecursiveCharacterTextSplitter(
        # TODO: find the best fit chunk_size. Together with ContextualCompressionRetriever. Large chunk_size may exceed max token size
        chunk_size=1000,
        chunk_overlap=200,
        separators=[
            "",
            " ",
            ",",
            "\n",
            "\n\n",
        ],
    )
    """
    rds = Redis.from_documents(
        documents=docs,
        embedding=embeddings,
        redis_url="redis://127.0.0.1:6379/0",
        index_name=f"{user_id}:{chat_id}",
    )
    rds.write_schema("redis_schema.yaml")
    return rds


def load_vectorstore(
    chat_id: str,
    user_id: str,
):
    vectorstore = Redis.from_existing_index(
        embedding=embeddings,
        index_name=f"{user_id}:{chat_id}",
        redis_url="redis://localhost:6380/0",
    )
    return vectorstore
