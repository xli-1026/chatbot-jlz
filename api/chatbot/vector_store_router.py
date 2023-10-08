from typing import Annotated

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import BaseLLM, HuggingFaceTextGenInference
from langchain.memory import ConversationBufferWindowMemory, RedisChatMessageHistory
from langchain.vectorstores import Redis

from langchain.utilities.sql_database import SQLDatabase
from loguru import logger

from chatbot.callbacks import (
    StreamingLLMCallbackHandler,
    UpdateConversationCallbackHandler,
)
from chatbot.vector_store import init_vectorstore
from chatbot.config import settings
from chatbot.history import AppendSuffixHistory
from chatbot.prompts.vicuna import (
    prompt,
    human_prefix,
    ai_prefix,
    human_suffix,
    ai_suffix,
)
from chatbot.schemas import (
    ChatMessage,
)
from chatbot.utils import UserIdHeader

from chatbot.db_chain import TablegptSQLDatabaseChain
from chatbot.prompts.sql_prompt import MYSQL_PROMPT

router = APIRouter(
    prefix="/api",
    tags=["conversation"],
)
embeddings = HuggingFaceEmbeddings()


def get_message_history() -> RedisChatMessageHistory:
    return AppendSuffixHistory(
        url=str(settings.redis_om_url),
        user_suffix=human_suffix,
        ai_suffix=ai_suffix,
        session_id="sid",  # a fake session id as it is required
    )


def get_llm() -> BaseLLM:
    return HuggingFaceTextGenInference(
        inference_server_url=str(settings.inference_server_url),
        stop_sequences=["</s>", f"{human_prefix}:"],
        streaming=True,
    )


def get_sql_llm() -> BaseLLM:
    return HuggingFaceTextGenInference(
        inference_server_url=str(settings.isvc_coder_model_uri),
        max_new_tokens=512,
        top_k=None,
        top_p=None,
        typical_p=None,
        temperature=0.01,
        repetition_penalty=None,
        stop_sequences=["Question", "SQLResult"],
        streaming=True,
        callbacks=[],
    )


SCHEMA = "college_2"
vector_store_schema = {
    "text": [
        {
            "name": "table_name",
            "weight": 1,
            "no_stem": False,
            "withsuffixtrie": False,
            "no_index": False,
            "sortable": False,
        },
        {
            "name": "content",
            "weight": 1,
            "no_stem": False,
            "withsuffixtrie": False,
            "no_index": False,
            "sortable": False,
        },
    ],
    "vector": [
        {
            "name": "content_vector",
            "dims": 768,
            "algorithm": "FLAT",
            "datatype": "FLOAT32",
            "distance_metric": "COSINE",
            "initial_cap": 20000,
            "block_size": 1000,
        }
    ],
}  # Schema of the index and the vector schema


@router.post("/init", status_code=201)
async def create_vectorstore_from_files(
    payload: dict, userid: Annotated[str | None, UserIdHeader()] = None
):
    warehouse = SQLDatabase.from_uri(
        f"{settings.warehouse_uri}{SCHEMA}", sample_rows_in_table_info=2
    )
    vector_store = init_vectorstore(
        chat_id=payload["conversation_id"],
        user_id=userid,
        database=warehouse,
    )
    vector_store


@router.websocket("/chat")
async def generate(
    websocket: WebSocket,
    llm: Annotated[BaseLLM, Depends(get_llm)],
    sql_llm: Annotated[BaseLLM, Depends(get_sql_llm)],
    history: Annotated[RedisChatMessageHistory, Depends(get_message_history)],
    userid: Annotated[str | None, UserIdHeader()] = None,
):
    await websocket.accept()
    memory = ConversationBufferWindowMemory(
        human_prefix=human_prefix,
        ai_prefix=ai_prefix,
        memory_key="history",
        chat_memory=history,
    )
    # region vector store prepare

    warehouse = SQLDatabase.from_uri(f"{settings.warehouse_uri}{SCHEMA}")
    while True:
        try:
            payload: str = await websocket.receive_text()
            message = ChatMessage.parse_raw(payload)
            vector_store = Redis.from_existing_index(
                redis_url="redis://localhost:6379/0",
                embedding=embeddings,
                index_name=f"{userid}:{message.conversation}",
                schema="redis_schema.yaml",
            )
            # endregion
            db_chain = TablegptSQLDatabaseChain.from_llm(
                llm=llm,
                prompt=MYSQL_PROMPT,
                coder_llm=llm,
                vector_store=vector_store,
                db=warehouse,
                memory=memory,
                top_k=5,
                use_query_checker=False,
                return_direct=True,
            )
            history.session_id = f"{userid}:{message.conversation}"
            streaming_callback = StreamingLLMCallbackHandler(
                websocket, message.conversation
            )
            update_conversation_callback = UpdateConversationCallbackHandler(
                message.conversation
            )
            await db_chain.arun(
                message.content,
                callbacks=[streaming_callback, update_conversation_callback],
            )
        except WebSocketDisconnect:
            logger.info("websocket disconnected")
            return
        except Exception as e:
            logger.error(f"Something goes wrong, err: {e}")
