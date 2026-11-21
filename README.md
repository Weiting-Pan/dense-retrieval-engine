

\# Dense Retrieval Engine: Exact Search and HNSW Benchmark



A hands-on information retrieval project exploring dense vector representations, exact similarity search, and approximate nearest neighbor (ANN) retrieval using FAISS and HNSW.



The project implements a dense retrieval pipeline from scratch, evaluates retrieval effectiveness on the SciFact dataset, and benchmarks the trade-offs between search accuracy, latency, and index storage.



\## 1. Project Overview



Dense retrieval represents queries and documents as continuous vectors in a shared embedding space. Relevant documents are retrieved by comparing their vector representations.



However, exact nearest neighbor search requires comparing each query against every indexed document, which can become computationally expensive as the collection grows.



Approximate nearest neighbor (ANN) algorithms, such as HNSW, aim to reduce search cost while maintaining high agreement with exact search.



This project investigates three questions:



1\. How can dense retrieval be implemented using sentence embeddings and cosine similarity?

2\. How effectively does a pretrained embedding model retrieve relevant scientific documents?

3\. How does HNSW trade retrieval accuracy for search latency and index storage?



\## 2. Implementation



The project is organized into three Jupyter Notebooks.



\### Notebook 01: Dense Retrieval Basics



`notebooks/01\_dense\_retrieval\_basics.ipynb`



Implements the fundamental operations of dense retrieval using a small collection of five documents.



Key components:



\- Sentence embedding generation using Sentence Transformers

\- Manual cosine similarity computation

\- Top-K document ranking

\- A reusable dense retrieval function

\- L2 normalization

\- Vectorized similarity search using NumPy

\- Correctness verification between loop-based and vectorized implementations



\### Notebook 02: Retrieval Evaluation



`notebooks/02\_retrieval\_evaluation.ipynb`



Implements an exact dense retrieval pipeline using the SciFact dataset from BEIR.



Key components:



\- Loading the SciFact corpus, queries, and relevance judgments

\- Generating normalized 384-dimensional document and query embeddings

\- Exact similarity search using matrix multiplication

\- Efficient Top-K selection using `np.argpartition()`

\- Evaluating retrieval effectiveness using Recall@10 and nDCG@10

\- Saving embeddings and evaluation results



\### Notebook 03: ANN Benchmark



`notebooks/03\_ann\_benchmark.ipynb`



Benchmarks exact and approximate vector search using FAISS.



Key components:



\- Exact search with `IndexFlatIP`

\- Approximate search with `IndexHNSWFlat`

\- Verification against NumPy exact search

\- ANN Recall@10 evaluation

\- Search latency benchmarking

\- HNSW `efSearch` parameter sweep

\- Serialized index size comparison

\- Recall-latency trade-off visualization



\## 3. Dataset and Embedding Model



\*\*Dataset:\*\* SciFact (BEIR)



\- Documents: 5,183

\- Evaluation queries: 300



\*\*Embedding model:\*\* `sentence-transformers/all-MiniLM-L6-v2`



\- Embedding dimension: 384

\- Similarity metric: Cosine similarity

\- Embedding normalization: L2 normalization



For normalized embeddings, cosine similarity is equivalent to inner product:



$$

\\operatorname{cosine}(q,d)=\\hat{q}\\cdot\\hat{d}

$$



This allows the same vector representations to be used for NumPy-based exact search and FAISS inner product indexes.



\## 4. Experimental Results



\### 4.1 Exact Dense Retrieval Quality



The dense retrieval pipeline was evaluated against SciFact ground-truth relevance judgments.



| Metric | Result |

|---|---:|

| Recall@10 | 0.7833 |

| nDCG@10 | 0.6451 |



Recall@10 measures the proportion of relevant documents retrieved within the top 10 results.



nDCG@10 additionally considers the ranking positions of relevant documents.



\### 4.2 Exact Search vs. HNSW



We compare FAISS `IndexFlatIP` against `IndexHNSWFlat`.



HNSW configuration:



\- M = 16

\- efConstruction = 100

\- efSearch = 32



| Metric | IndexFlatIP | IndexHNSWFlat |

|---|---:|---:|

| Search time (ms/query) | 3.6973 | 0.4281 |

| ANN Recall@10 | 1.0000 | 0.9730 |

| Serialized index size (MiB) | 7.5923 | 8.3038 |



At `efSearch = 32`, HNSW achieved approximately \*\*8.64x faster search\*\* while recovering \*\*97.3% of the exact Top-10 nearest neighbors\*\*.



Search times represent average batch execution time divided by the number of queries, measured using a single FAISS CPU thread.



ANN Recall@10 measures agreement with exact nearest neighbor search. It is distinct from retrieval Recall@10, which uses dataset relevance judgments.



\### 4.3 HNSW Recall-Latency Trade-off



We evaluate different `efSearch` values to investigate the trade-off between approximate search accuracy and computational cost.



| efSearch | ANN Recall@10 | Search Time (ms/query) |

|---:|---:|---:|

| 10 | 0.8777 | 0.2016 |

| 16 | 0.9303 | 0.2521 |

| 32 | 0.9730 | 0.4002 |

| 64 | 0.9917 | 0.6347 |

| 128 | 0.9983 | 1.0796 |

| 256 | 1.0000 | 2.0681 |



!\[HNSW Recall-Latency Trade-off](results/hnsw\_recall\_latency.png)



Increasing `efSearch` generally improves ANN Recall@10 but increases search time.



The results illustrate how HNSW can be tuned to balance search quality and performance.



\### 4.4 Index Storage Comparison



| Index | Serialized Size |

|---|---:|

| IndexFlatIP | 7.5923 MiB |

| IndexHNSWFlat | 8.3038 MiB |



The HNSW index requires approximately 1.09x the serialized storage of the Flat index.



This overhead comes from the additional graph connectivity information maintained by HNSW.



These values represent serialized index sizes, not peak runtime memory usage.



\## 5. Project Structure



```text

dense\_retrieval/

├── notebooks/

│   ├── 01\_dense\_retrieval\_basics.ipynb

│   ├── 02\_retrieval\_evaluation.ipynb

│   └── 03\_ann\_benchmark.ipynb

├── data/

│   └── scifact\_embeddings.npy

├── results/

│   ├── exact\_retrieval\_metrics.json

│   ├── hnsw\_benchmark.csv

│   ├── ann\_benchmark\_summary.json

│   └── hnsw\_recall\_latency.png

├── requirements.txt

├── .gitignore

└── README.md

```



The embedding file is generated locally and excluded from Git tracking.



\## 6. Installation and Usage



\### Prerequisites



\- Python 3.11

\- JupyterLab

\- Required Python packages listed in `requirements.txt`



\### Installation



Clone the repository:



```bash

git clone <repository-url>

cd dense\_retrieval

```



Create and activate a Python environment, then install the dependencies:



```bash

python -m pip install -r requirements.txt

```



Launch JupyterLab:



```bash

jupyter lab

```



\### Execution Order



Run the notebooks in the following order:



1\. `01\_dense\_retrieval\_basics.ipynb`

2\. `02\_retrieval\_evaluation.ipynb`

3\. `03\_ann\_benchmark.ipynb`



Notebook 02 generates `data/scifact\_embeddings.npy`, which is required by Notebook 03.



The SciFact dataset and pretrained embedding model are downloaded automatically when the corresponding notebook code is executed for the first time.



\## 7. Key Findings



The experiments demonstrate that:



1\. Dense retrieval can be implemented using normalized sentence embeddings and vectorized similarity computation.

2\. Exact retrieval provides a reference for evaluating both retrieval effectiveness and approximate search correctness.

3\. HNSW can significantly reduce search time while maintaining high agreement with exact nearest neighbor results.

4\. Increasing `efSearch` improves ANN recall but increases search latency.

5\. Graph-based ANN indexes introduce additional storage overhead compared with flat vector indexes.



These findings provide a foundation for exploring more efficient retrieval systems, including graph-based indexing, index compression, and hybrid retrieval.



\## 8. Technologies



\- Python

\- NumPy

\- Pandas

\- Matplotlib

\- Sentence Transformers

\- Hugging Face Datasets

\- FAISS

\- JupyterLab



