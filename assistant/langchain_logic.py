import os
import re
from functools import lru_cache
from django.conf import settings

from langchain_community.chat_models import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser

# === Setup ===
os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY

# === Model ===
llm = ChatOpenAI(model_name="gpt-4", temperature=0)

# === Session Memory ===
store = {}
def get_memory(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# === DB Connection ===
db_user = settings.DB_USER
db_password = settings.DB_PASSWORD
db_host = settings.DB_HOST
db_name = settings.DB_NAME
db_port = settings.DB_PORT
db_url = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
db = SQLDatabase.from_uri(db_url)

# === Schema Description Caching ===
@lru_cache()
def get_schema_description():
    return db.get_table_info()

# === Few-shot SQL Examples ===
EXAMPLES = """
Examples:

Q: List all customers from France.
A:
SELECT customerName FROM customers WHERE country = 'France';

Q: Which product has the highest price?
A:
SELECT productName FROM products ORDER BY buyPrice DESC LIMIT 1;

Q: What is the total payment amount received?
A:
SELECT SUM(amount) FROM payments;

Q: How many orders were placed by each customer?
A:
SELECT customerNumber, COUNT(*) as order_count FROM orders GROUP BY customerNumber;
"""

# === SQL Prompt Builder ===
def build_sql_prompt():
    schema = get_schema_description()
    return PromptTemplate.from_template(
        f"""You are an expert SQL assistant.
Use the schema and examples below to write a valid MySQL query for the user's question.
Only return the raw SQL query without explanation or markdown formatting.

Schema:
{schema}

{EXAMPLES}

Chat History:
{{chat_history}}

User: {{question}}
SQL:"""
    )

# === Strip markdown or extra explanation from SQL ===
def clean_sql_output(sql):
    sql = sql.strip()
    sql = re.sub(r"```sql|```", "", sql).strip()
    sql = re.split(r'\n(?=SELECT|WITH|INSERT|UPDATE|DELETE)', sql, maxsplit=1)[-1]
    return sql.strip()

# === Correct Common SQL Mistakes ===
def correct_common_sql_errors(query):
    corrections = {
        "customer_id": "customerNumber",
        "order_id": "orderNumber",
        "order_details": "orderdetails",
        "product_id": "productCode",
        "products.price": "products.buyPrice"
    }
    for wrong, right in corrections.items():
        query = query.replace(wrong, right)
    return query

# === Execute SQL ===
def execute_query(query):
    try:
        print(f"\nExecuting SQL Query:\n{query}")
        result = db.run(query)
        return result if result else {"info": "No results found."}
    except Exception as e:
        print(f"\nSQL Execution Error: {e}")
        return {"error": f"Query failed: {str(e)}"}

# === Rephrase Results ===
answer_prompt = PromptTemplate.from_template(
    """Given the user question, SQL query, and result, return a helpful, user-friendly answer.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer:"""
)
rephrase_chain = answer_prompt | llm | StrOutputParser()

# === Final Processor ===
def process_question(question, session_id="user-1"):
    sql_prompt = build_sql_prompt()
    memory_chain = RunnableWithMessageHistory(
        RunnableSequence(sql_prompt | llm | StrOutputParser()),
        get_session_history=get_memory,
        input_messages_key="question",
        history_messages_key="chat_history"
    )

    # 1. Generate SQL
    raw_sql = memory_chain.invoke(
        {"question": question},
        config={"configurable": {"session_id": session_id}}
    )
    clean_query = clean_sql_output(raw_sql)
    clean_query = correct_common_sql_errors(clean_query)

    # 2. Execute SQL
    sql_result = execute_query(clean_query)

    # 3. Rephrase answer
    return rephrase_chain.invoke({
        "question": question,
        "query": clean_query,
        "result": str(sql_result)
    })
