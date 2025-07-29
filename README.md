📄 RAG Application By HMA  
This is a RAG (Retrieval-Augmented Generation) application built using Streamlit, LangChain, and Mistral AI, allowing users to upload documents and ask natural language questions. 
The app retrieves relevant information from the document using embeddings and answers using a language model.  

🚀 Features  
📂 Upload documents (.pdf, .docx, .txt, .csv, .html, .md, .json) 
✂️ Automatic document chunking and embedding using FAISS 
🔍 Semantic search using similarity-based retrieval 
🤖 Answers generated using MistralAI (mistral-small) 
🧠 Built with LangChain, FAISS, and Sentence Transformers 
🌐 Easy-to-use Streamlit web interface  

🛠 Tech Stack  
Streamlit –
Web UI LangChain – 
Retrieval + chaining FAISS – 
Vector store for similarity search Sentence Transformers – 
Embedding model (all-MiniLM-L6-v2) MistralAI – 
LLM for response generation - 

📦 Setup (Locally)  
pip install -r requirements.txt streamlit run main.py  

📁 Project Structure  
App/ ├── main.py         # Streamlit app entry point 
     ├── loader.py       # File loading logic
     ├── splitter.py     # Chunking documents
     ├── embedder.py     # Embedding model
     ├── requirements.txt  
     
☁️ Deployment 
To deploy this app on Streamlit Community Cloud: Upload your code to a public GitHub repository Go to https://streamlit.io/cloud Connect your repo and deploy the main.py file
