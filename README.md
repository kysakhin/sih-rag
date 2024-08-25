# RAG Application

## Overview

The RAG (Retrieval-Augmented Generation) Application provides a streamlined interface for querying text or PDF files using retrieval-augmented generation techniques. It integrates a generative model with a retrieval mechanism to deliver contextually relevant answers based on the content of the uploaded files.

## Features

- **File Support**: Handles `.txt` and `.pdf` file formats.
- **Generative Model**: Utilizes the Google FLAN-T5 model from Hugging Face.
- **Retrieval-Augmented Generation**: Combines retrieval and generative techniques to enhance answer accuracy.
- **Streamlit Interface**: User-friendly web interface for file uploads and querying.

## Getting Started

### Prerequisites

- Python 3.8+
- Required Python libraries (listed in `requirements.txt`)
- Hugging Face API key

### Installation

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
