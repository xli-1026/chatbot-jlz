from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain.callbacks.base import AsyncCallbackHandler
from langchain.callbacks.manager import (
    CallbackManagerForLLMRun,
    AsyncCallbackManagerForLLMRun,
)
from langchain.chains import LLMChain
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.schema import BaseChatMessageHistory

# from chatbot.history import AppendSuffixHistory

# from tablegpt.websocket.schemas import ai_plain_text_message

empty_template: PromptTemplate = PromptTemplate(
    input_variables=["input"],
    template="""{input}""",
)


class PassThroughLLM(LLM):
    """PassThroughLLM pass inputs directly to outputs.
    This is because we sometimes need to send message directly to user, but we also want to leverage the LLMChain's history and callback system.
    """

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        return prompt

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    ) -> str:
        return prompt

    @property
    def _llm_type(self) -> str:
        return "pass-through"


class AsyncPassThroughHistoryCallbackHandler(AsyncCallbackHandler):
    """AsyncPassThroughHistoryCallbackHandler only adds outputs to history.
    Used for pass_through_chain, in which the input is directly sent to output."""

    def __init__(self, history: BaseChatMessageHistory):
        self.history = history

    async def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run when chain ends running."""
        # msg = ai_plain_text_message(text=outputs["response"])
        # self.history.add_ai_message(msg.json())
        pass


def pass_through_chain(
    history: BaseChatMessageHistory = None, callbacks=None
) -> LLMChain:
    # history_ch = AsyncPassThroughHistoryCallbackHandler(history=history)
    # cbs = [history_ch]
    cbs = []
    if callbacks is not None:
        cbs.extend(callbacks)
    llm = PassThroughLLM()
    return LLMChain(
        llm=llm,
        prompt=empty_template,
        output_key="response",
        callbacks=cbs,
    )
