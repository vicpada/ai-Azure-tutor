This file provides information on Data Curation:

1. [Create dataset](01_Create_Dataset.ipynb)<br />
<em>This step crawls and scrapes data sources</em><br />
A jupiter notebook with steps to scrap data from five different [Azure Resources sites](https://docs.google.com/spreadsheets/d/1b_QcHNPBg34Q05FPsmRqzPW5XTUGmedaP_SLonyiMt4) 
It uses [Firecrawl](https://www.firecrawl.dev/) for the scrapping. The API kept failing and I end up scheduling and downloading the crawling from the site directly.
The files were uploaded into HF in `vicpada/AzureResources` into a zip file.

2. [Process files](02_Process_files.ipynb)<br />
<em>This step builds prepares JSONL</em><br />
A jupiter notebook with the steps to download the zip file create a JSONL files. One JSONL file for each site. For each entry from [Firecrawl](https://www.firecrawl.dev/) extracts a title, cleans the content, adds the token count, skips very small or extremely large files and generates a deterministic id so we can use for evaluation. Also, in the next step we will include the whole doc as context for those documents with token count less than 8000.

3. [Add context](03_Add_context.ipynb)<br />
<em>This step creates the chunks, adds context, and saves them into PKL files</em><br />
A jupiter notebook to download the JSONL files and build the documents for our RAG application.
It uses [SentenceSplitter](https://docs.llamaindex.ai/en/stable/api_reference/node_parsers/sentence_splitter/) with a <b>chunk size of 800</b> and <b>0 overlap</b>.
It also adds some context to each chunk to situate it within the overall document for the purposes of improving search retrieval of the chunk.
For each JSONL file a PKL file is created and uploaded to HF.
It uses `gpt-4.1-nano` for situating the chunk in the context for cost saving purposes and to finish processing in a timely manner. 
Also one site from the datasources was excluded as it was getting very expensive to process all.

4. [Finetune embedding](04_Finetune_Embedding.ipynb)<br />
<em>This step finetunes the embedding model</em><br />
A jupiter notebook with steps to download the PKL files and train an embedding model. Only 10,000 chunks were used for time and cost saving purposes as there were more than 400,000 nodes. Using [`generate_qa_embedding_pairs`](https://docs.llamaindex.ai/en/stable/module_guides/models/embeddings/) for generating training and validation data. 
The base model for the embedding is `local:BAAI/bge-small-en-v1.5`. <b>Hit Rate & MRR</b> were used for evaluation. The model is uploaded to HF.
Note: Finetuning and using the embedding model locally is a much cheaper way than using a online service. When dealing with a big number of nodes, and a big number of queries it does a big difference at the time of processing the nodes.


5. [Create test data](05_Create_Test_Data.ipynb)<br />
<em>Create Test data for evaluation of the model</em><br />
A jupiter notebook with steps to download the embedding model, the PKL files and use [`generate_question_context_pairs`](https://docs.llamaindex.ai/en/stable/examples/evaluation/QuestionGeneration/) to generate evaluation data. The json generated file is then uploaded to HF. It uses `Gemini-2.0-flash` to generate the eval data.


6. [Create Vector](06_Create_Vector.ipynb)<br />
<em>Create VectorDB and evaluate</em><br />
A jupiter notebook with steps to generate the VectorDB using the local embedding model. 
It calculates "hit_rate", "mrr", "precision", "recall", "ap" and "ndcg" metrics
The vector is saved as a PKL file and uploaded to HF. 

7. [Reranking](07_Reranking.ipynb)<br />
<em>Evaluate Cohere Rerank</em><br />
A jupiter notebook with steps to download the VectorDB, the embedding model and the evaluation dataset.
Then it evaluates the result with and without cohere. Setting the path for the final solution

