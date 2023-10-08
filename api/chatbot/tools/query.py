import csv
import tempfile
from typing import Optional

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools.base import BaseTool
from langchain.tools.sql_database.tool import BaseSQLDatabaseTool


class QuerySQLDatabaseTool(BaseSQLDatabaseTool, BaseTool):
    """Tool for querying a SQL database.
    It will save the query result to an S3 object store.
    """

    name = "sql_db_query"
    description = """
    Input to this tool is a detailed and correct SQL query, output is a result from the database.
    If the query is not correct, an error message will be returned.
    If an error is returned, rewrite the query, check the query, and try again.
    """

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the query, return the results or an error message."""
        rows = self.db._execute(query)
        filename = ""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline=""
        ) as tmp:
            filename = tmp.name
            # if len(rows) == 1 and all(element is None for element in rows[0]):
                
            if rows:
                csv_writer = csv.writer(tmp)
                for row in rows:
                    csv_writer.writerow(row)
        return filename

    async def _arun(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the query, return the results or an error message."""
        return self._run(query, run_manager=run_manager)
