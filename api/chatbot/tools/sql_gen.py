from typing import Any, Optional

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from langchain.tools.base import BaseTool
from langchain.tools.sql_database.tool import BaseSQLDatabaseTool
from pydantic import Field, model_validator
from pydantic.v1 import root_validator


_mysql_coder_prompt = """You are a MySQL expert. Given an input question, create a syntactically correct MySQL query to run.
Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per MySQL. You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in backticks (`) to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
Pay attention to use NOW() function to get the current time. Custom function StartOfPeriod($Period, Period_Number) and EndOfPeriod($Period, Period_Number) indicates the start and end of the time window, $Period: ['YEAR','QUARTER','MONTH','WEEK','DAY']; Period_Number: Int, negative numbers represent the past, zero represents the present, and positive numbers represent the future.

Use the following format:

Question: Question here
SQLQuery: SQL Query to run

"""

CODER_PROMPT_SUFFIX = """Only use the following tables:
{table_info}.

Question: {input}
"""

MYSQL_CODER_PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "top_k"],
    template=_mysql_coder_prompt + CODER_PROMPT_SUFFIX,
)


class SQLGeneratorDatabaseTool(BaseSQLDatabaseTool, BaseTool):
    """Tool for generating SQL queries.
    This class is is currently not utilizing the `BaseSQLDatabaseTool.db`, but it may leverage it in the future to determine the dialect.
    """

    name = "sql_db_sql_gen"
    description = ""
    llm: BaseLanguageModel
    llm_chain: LLMChain = Field(init=False)

    # @model_validator(mode="before")
    @root_validator(pre=True)
    def initialize_llm_chain(cls, values: dict[str, Any]) -> dict[str, Any]:
        # TODO: allow custom prompt?
        values["llm_chain"] = LLMChain(
            llm=values.get("llm"),
            prompt=MYSQL_CODER_PROMPT,
        )
        return values

    def _run(
        self,
        llm_inputs: dict[str, Any],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the query, return the results or an error message."""
        sql_cmd: str = self.llm_chain.predict(
            callbacks=run_manager.get_child() if run_manager else None,
            **llm_inputs,
        )
        return sql_cmd.strip()

    async def _arun(
        self,
        llm_inputs: dict[str, Any],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the query, return the results or an error message."""
        sql_cmd: str = await self.llm_chain.apredict(
            callbacks=run_manager.get_child() if run_manager else None,
            **llm_inputs,
        )
        return sql_cmd.strip()
