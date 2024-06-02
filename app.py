import streamlit as st
import pickle
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS


image='/Users/rohanwaghmare/intellipdfv2/ro3.jpeg'
with st.sidebar:
    st.title('IntelliPDF  🧠')
    st.markdown('''
    # About ✍🏻
    This app is an LLM-powered 💪🏻 chatbot built using:
    
    - [streamlit](https://streamlit.io/)   
    - [LangChain](https://python.langchain.com/)   
    - [OpenAI](https://platform.openai.com/docs/models)
    
    ### Check out more such cool projects on:    
    ''')
    
    
    col1,col2=st.columns(2)
    with col1:
        st.markdown('''
        
        - [github](https://github.com/ron103)      
        - [rohanwaghmare](https://rohanwaghmare.com)''')  
                
    
    with col2:
        st.image(image, caption='', width=80, use_column_width=False, clamp=False, channels="RGB", output_format="auto")
for _ in range(5):
    st.write("") 
    
def main():
    
    
    st.header('Chat with PDF 🤖')
    st.caption(':blue[Project under development as OpenAI API has exhausted quota]')
    load_dotenv()
    pdf = st.file_uploader('Upload your PDF', type='PDF')
    
    if pdf:
        pdf_reader = PdfReader(pdf)
        st.write(pdf_reader)
        
        txt=''
        for page in pdf_reader.pages:
            txt+=page.extract_text()
        #st.write(txt)

        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        chunks = text_splitter.split_text(text=txt)
        st.write(chunks)
        
        
        pdf_name=pdf.name[:-4]
        if os.path.exists(f'{pdf_name}.pkl'):
            with open(f'{pdf_name}.pkl','rb') as f:
                VectorStore = pickle.load(f)
            st.write('Embeddings Loaded from the Disk')
        else:
            embeddings=OpenAIEmbeddings()
            VectorStore=FAISS.from_texts(chunks, embedding=embeddings)
            with open(f'{pdf_name}.pkl', 'wb') as f:
                pickle.dump(VectorStore,f)
            st.write('Embeddings Computation Completed')
        
        query = st.text_input('Ask questions about your PDF file: ')
        
        st.write(query)
        
        if query:
            docs = VectorStore.similarity_search(query=query,k=3)
            st.write(docs)
            
            llm = OpenAI(model_name='gpt-3.5-turbo')
            chain = load_qa_chain(llm=llm,chain_type='stuff')
            with get_openai_callback() as cb:
                response=chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)
            
if __name__=='__main__':
    main()