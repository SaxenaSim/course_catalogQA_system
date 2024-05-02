# course_catalogQA_system

This repository contains code for a Question Answering (QA) system designed to provide answers to questions related to a course catalog PDF document.

## Overview

The Course Catalog QA System is built using LangChain, an open-source Python library for building end-to-end language processing pipelines. The system is designed to extract relevant information from a course catalog PDF document and provide answers to user queries using a combination of text chunking, embeddings, retrieval, and language model-based question answering.

## Features

- **PDF Document Processing**: The system processes a course catalog PDF document to extract relevant information.
- **Text Chunking**: The document is divided into chunks for efficient processing.
- **Embeddings**: Text embeddings are generated using OpenAI's GPT-3.5 model.
- **Retrieval**: The system utilizes ChromaDB for document retrieval based on user queries.
- **Question Answering**: Answers to user queries are generated using LangChain's Question Answering capabilities.

## Files

- `data/`: Contains the input PDF document.
- `create_retriever.py`: Python script to process the PDF document, generate embeddings, and create a retriever.
- `validate_retrieval.py`: Python script to test the retriever by validating its functionality.
- `create_chain.py`: Python script to create a QA chain and answer user queries.

## Usage

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the create_retriever.py script to process the PDF document and generate embeddings:
   ```bash
   python create_retriever.py
   ```

3. Test the retriever using the validate_retrieval.py script:
   ```bash
   python validate_retrieval.py
   ```
   
4. Use the create_chain.py script to create a QA chain and answer user queries:
   ```bash
   python create_chain.py
   ```
