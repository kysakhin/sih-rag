import streamlit as st
from backend.main import get_vectordb, run_llm

st.title("APPLICATION TO QUERY YOUR FILES ")
st.subheader(" Works on a .txt file or a .pdf file using R.A.G")
st.write("[model : google/flan-t5-base]")
st.text("make sure that the pdf file and text file are well strucutured \nand only contain well spaced characters\nyou may also get an error if your API key is invalid\nthe model may take a while to generate responses :)\n \t\t\t\t\t\t -Harsh ")
#taking api key input
API_key = st.sidebar.text_input("PLEASE ENTER YOUR HUGGINGFACEHUB API KEY ", type = "password")



#taking the file input
uploaded_file = st.file_uploader("upload file", type=("txt","pdf"))
question = st.text_input(

    "Ask something about the file",
    placeholder="Can you give me a short summary?",
    disabled=not uploaded_file or not API_key,
)

if API_key and uploaded_file:
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())
else:
   st.info("no doc uploaded or no key uploaded")

if uploaded_file is not None:
    vectordb = get_vectordb(uploaded_file.name)
    if uploaded_file is None:
        st.error("either file is not uploaded or the file type is not supported ")
    elif vectordb is None:
        st.error("the file type is not supported ")
else:
     st.error("Either the file is not uploaded or the file type is not supported.")

#doing the spin thingy while generating a resp
with st.spinner("Generating response..."):
    if API_key and question:
        answer = run_llm(key= API_key, db= vectordb , query= question)
        st.write("### Answer")
        st.write(f"{answer['result']}")
        st.write("### Relevant source")
        rel_docs = answer['source_documents']
        for i, doc in enumerate(rel_docs):
            st.write(f"**{i+1}**: {doc.page_content}\n")
    else:
        st.info("_______________________")
