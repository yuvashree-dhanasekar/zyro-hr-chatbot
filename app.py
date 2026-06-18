import os
import re
import streamlit as st
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(page_title="Zyro Dynamics HR Help Desk", page_icon="🤖")
st.title("🤖 Zyro Dynamics HR Help Desk")
st.caption("Ask me anything about HR policies or your leave balance!")

@st.cache_resource
def load_pipeline():
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    loader = PyPDFDirectoryLoader("hr_policies/")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 7, "fetch_k": 20, "lambda_mult": 0.9},
    )
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    SYSTEM_PROMPT = """You are the Zyro Dynamics HR Help Desk assistant.
Answer the employee's question using ONLY the context below from official Zyro Dynamics HR policy documents.
Be concise and specific. If the answer is not in the context, respond with exactly:
"I can only answer HR-related questions from Zyro Dynamics policy documents."
Context:
{context}"""
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", "{question}")]
    )
    def format_docs(docs):
        return "\n\n".join(
            f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
            for d in docs
        )
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    leave_df = pd.read_csv("leave_records.csv")
    leave_df["start_date"] = pd.to_datetime(leave_df["start_date"])
    leave_df["end_date"] = pd.to_datetime(leave_df["end_date"])
    leave_df["days_taken"] = (leave_df["end_date"] - leave_df["start_date"]).dt.days + 1
    return vectorstore, rag_chain, leave_df

vectorstore, rag_chain, leave_df = load_pipeline()

def calculate_leave(employee_id, leave_type=None):
    emp = leave_df[leave_df["employee_id"].str.upper() == employee_id.upper()]
    if emp.empty:
        return f"No leave records found for employee ID '{employee_id}'."
    if leave_type:
        emp = emp[emp["leave_type"].str.lower() == leave_type.lower()]
        if emp.empty:
            return f"No {leave_type} records found for employee '{employee_id}'."
        return f"Employee {employee_id} has taken {emp['days_taken'].sum()} day(s) of {leave_type}."
    summary = emp.groupby("leave_type")["days_taken"].sum()
    lines = [f"Leave summary for **{employee_id}**:"]
    for lt, days in summary.items():
        lines.append(f"- {lt}: {days} day(s)")
    lines.append(f"\n**Total: {emp['days_taken'].sum()} day(s)**")
    return "\n".join(lines)

def smart_answer(question):
    q_lower = question.lower()
    emp_match = re.search(r"EMP\d+", question, re.IGNORECASE)
    if emp_match and any(w in q_lower for w in ["day", "leave", "taken", "off", "used", "balance"]):
        emp_id = emp_match.group(0).upper()
        leave_types = ["casual leave", "earned leave", "sick leave", "maternity leave", "paternity leave"]
        detected_type = next((lt for lt in leave_types if lt in q_lower), None)
        return calculate_leave(emp_id, detected_type)
    if any(w in q_lower for w in ["how many days", "days taken", "days off", "leave taken"]):
        return "Please provide your Employee ID (e.g. EMP001) so I can look up your leave records."
    scored = vectorstore.similarity_search_with_score(question, k=3)
    if not scored or min(score for _, score in scored) > 1.5:
        return "I can only answer HR-related questions from Zyro Dynamics policy documents."
    return rag_chain.invoke(question)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask your HR question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = smart_answer(prompt)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
