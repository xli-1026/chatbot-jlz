from typing import Any, Dict, List, Optional, Union

from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
)
from langchain.chains.llm import LLMChain
from langchain.chains.sql_database.prompt import PROMPT, SQL_PROMPTS
from langchain.llms.base import LLM
from langchain.memory.chat_memory import BaseChatMemory
from langchain.schema import BasePromptTemplate, Document
from langchain.schema.language_model import BaseLanguageModel
from langchain.tools.sql_database.tool import QuerySQLCheckerTool
from langchain.utilities.sql_database import SQLDatabase
from langchain.schema.vectorstore import VectorStore, VectorStoreRetriever
from langchain_experimental.sql import SQLDatabaseChain
from langchain_experimental.sql.base import INTERMEDIATE_STEPS_KEY

from pydantic.v1 import root_validator

from chatbot.tools import (
    QuerySQLDatabaseTool,
    SQLGeneratorDatabaseTool,
    TableSelectTool,
)


class TablegptSQLDatabaseChain(SQLDatabaseChain):
    """A naive extention of SQLDatabaseChain that implements async run method."""

    coder_llm: BaseLanguageModel
    """LLM used to generate SQL commands only.
    It may use a different, over-fitted model.
    """
    gen_sql_tool: Optional[SQLGeneratorDatabaseTool] = None

    query_checker_tool: Optional[QuerySQLCheckerTool] = None

    query_tool: Optional[QuerySQLDatabaseTool] = None

    vector_store: VectorStore

    table_select_tool: Optional[TableSelectTool] = None

    @root_validator()
    def initialize_query_checker_tool(cls, values: dict[str, Any]) -> dict[str, Any]:
        values["gen_sql_tool"] = SQLGeneratorDatabaseTool(
            llm=values["coder_llm"], db=values["database"]
        )
        values["query_checker_tool"] = QuerySQLCheckerTool(
            db=values["database"], llm=values["llm_chain"].llm
        )
        values["query_tool"] = QuerySQLDatabaseTool(db=values["database"])
        values["table_select_tool"] = TableSelectTool(
            llm=values["coder_llm"],
            vector_store=values["vector_store"],
            db=values["database"],
        )
        return values

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        input_text = inputs[self.input_key]
        # input_text = f"{inputs[self.input_key]}\nSQLQuery:"
        await _run_manager.on_text(input_text, verbose=self.verbose)

        # If not present, then defaults to None which is all tables.
        # table_names_to_use = inputs.get("table_names_to_use")
        # table_info = self.database.get_table_info(table_names=table_names_to_use)
        llm_inputs = {
            "input": input_text,
            "top_k": str(self.top_k),
            "dialect": self.database.dialect,
            "stop": ["\nSQLResult:"],
        }
        intermediate_steps: List = []
        try:
            intermediate_steps.append(llm_inputs)  # input: sql generation
            docs: list[Document] = await self.table_select_tool.arun(
                tool_input={"llm_inputs": llm_inputs},
                callbacks=_run_manager.get_child(),
                tags=["coder"],
            )  # vector_store selected tables
            _tables = [doc.metadata["table_name"] for doc in docs]
            table_info = self.database.get_table_info(_tables)
            llm_inputs = {
                "input": input_text,
                "top_k": str(self.top_k),
                "dialect": self.database.dialect,
                "table_info": table_info,
                "stop": ["\nSQLResult:"],
            }
            sql_cmd = await self.gen_sql_tool.arun(
                tool_input={"llm_inputs": llm_inputs},
                callbacks=_run_manager.get_child(),
                tags=["coder"],
            )

            intermediate_steps.extend(docs)
            # output: sql generation
            intermediate_steps.append({"tables": docs})
            _tables = [doc.metadata["table_name"] for doc in docs]
            tables_str = ",".join(_tables)
            if self.return_direct:
                final_result = tables_str
            else:
                await _run_manager.on_text("\nAnswer:", verbose=self.verbose)
                result_str = str(result)
                if len(result_str) > 1000:
                    result_str = (
                        result_str[:1000]
                        + "...(The result is too large, here's the first 1000 characters)"
                    )
                # quote the command and result to minimize impact on final answer
                input_text += f"\n```sql\n{sql_cmd}\n```\nSQLResult: \n```\n{result_str}\n```\nAnswer:"
                llm_inputs["input"] = input_text
                intermediate_steps.append(llm_inputs)  # input: final answer
                final_result = await self._aanswer(llm_inputs, _run_manager)
                intermediate_steps.append(final_result)  # output: final answer
                await _run_manager.on_text(
                    final_result, color="green", verbose=self.verbose
                )
            chain_result: Dict[str, Any] = {self.output_key: final_result}
            if self.return_intermediate_steps:
                chain_result[INTERMEDIATE_STEPS_KEY] = intermediate_steps
            return chain_result
        except Exception as exc:
            # Append intermediate steps to exception, to aid in logging and later
            # improvement of few shot prompt seeds
            exc.intermediate_steps = intermediate_steps  # type: ignore
            raise exc

    async def _aanswer(
        self, llm_inputs: dict, run_manager: AsyncCallbackManagerForChainRun
    ):
        final_result = await self.llm_chain.apredict(
            callbacks=run_manager.get_child(),
            **llm_inputs,
        )
        return final_result

    def prep_inputs(self, inputs: Union[Dict[str, Any], Any]) -> Dict[str, str]:
        inputs = super().prep_inputs(inputs)
        if self.memory is not None and isinstance(self.memory, BaseChatMemory):
            input_str, _ = self.memory._get_input_output(
                inputs, {self.memory.output_key: "foo"}
            )
            self.memory.chat_memory.add_user_message(input_str)
        return inputs

    def prep_outputs(
        self,
        inputs: Dict[str, str],
        outputs: Dict[str, str],
        return_only_outputs: bool = False,
    ) -> Dict[str, str]:
        """Override this method to disable saving context to memory.
        We need to separatly save the input and output on chain starts and ends.
        """
        self._validate_outputs(outputs)
        if self.memory is not None and isinstance(self.memory, BaseChatMemory):
            _, output_str = self.memory._get_input_output(inputs, outputs)
            self.memory.chat_memory.add_ai_message(output_str)
        if return_only_outputs:
            return outputs
        else:
            return {**inputs, **outputs}

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        coder_llm: BaseLanguageModel,
        vector_store: VectorStore,
        db: SQLDatabase,
        prompt: Optional[BasePromptTemplate] = None,
        **kwargs: Any,
    ) -> SQLDatabaseChain:
        """Create a TablegptSQLDatabaseChain from an LLM and a database connection.

        *Security note*: Make sure that the database connection uses credentials
            that are narrowly-scoped to only include the permissions this chain needs.
            Failure to do so may result in data corruption or loss, since this chain may
            attempt commands like `DROP TABLE` or `INSERT` if appropriately prompted.
            The best way to guard against such negative outcomes is to (as appropriate)
            limit the permissions granted to the credentials used with this chain.
            This issue shows an example negative outcome if these steps are not taken:
            https://github.com/langchain-ai/langchain/issues/5923
        """
        prompt = prompt or SQL_PROMPTS.get(db.dialect, PROMPT)
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        return cls(
            vector_store=vector_store,
            llm_chain=llm_chain,
            coder_llm=coder_llm,
            database=db,
            **kwargs,
        )
