# RAG Application

## what this application does :

The RAG Application uses the Google FLAN-T5 model accesed using HuggingFace, LangChain is used for retrieval framework to provide answers from text and PDF files. Streamlit is used for the interface, it retrieves and generates required information from uploaded documents. You can check the GitHub repository below

## Functions :

- **File Support**: Handles `.txt` and `.pdf` file formats.
- **Generative Model**: uses the Google FLAN-T5 model from Hugging Face.
- **Retrieval-Augmented Generation**: uses RAG to get an answer
- **Streamlit Interface**: User-friendly interface for file uploads and querying.



### dependencies and what you'll need

- Python 3.8+
- Required Python libraries (listed in `requirements.txt`)
- Hugging Face API key

### how to install 

#### 1.Clone the repository:

   ```bash
   git clone https://github.com/Harshh1705/rag_application.git
```
##### 2. Navigate to the project directory:

   ```bash
   cd rag_application
  ```
#### 3.Install the required Python libraries:
```bash
pip install -r requirements.txt
```
## Useage:
#### To run the streamlit app enter this into the terminal:
```bash
  streamlit run interface.py
```
