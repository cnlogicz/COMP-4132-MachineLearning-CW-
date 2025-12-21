
# Machine Learning Coursework: Tolkien-Style Story Generation

## Project Overview
This project explores text generation through two distinct approaches based on J.R.R. Tolkien's *The Lord of the Rings* corpus:
1.  **Task 1 (Non-LLM):** A Character-level LSTM model designed to learn morphological rules and stylistic patterns from scratch.
2.  **Task 2 (LLM-based Agent):** A Multi-Agent System (RAG + LangGraph) capable of generating coherent, lore-accurate, and interactive narratives using a "Writer-Critic" cyclic workflow.

## Directory Structure
```text
MACHINEL_SUBMIT/
├── chroma_db/              # Persisted Vector Database for RAG (Lore & Style)
├── data/                   # Training data (LOTR trilogy txt files)
├── src/                    # Source code
│   ├── task1_lstm/         # Task 1: LSTM Implementation
│   │   ├── logs/           # Training logs
│   │   ├── best_lstm_model.pth  # Saved model weights (Best Val Loss)
│   │   ├── data_loader.py  # Data preprocessing & Sliding Window
│   │   ├── generate.py     # Text generation script
│   │   ├── lstm_config.py  # Hyperparameters
│   │   ├── model.py        # LSTM Architecture definition
│   │   ├── plot_loss.py    # Visualization of training curves
│   │   └── train.py        # Training loop implementation
│   ├── task2_llm/          # Task 2: RAG Agent Implementation
│   │   ├── agent_state.py  # LangGraph State definition
│   │   ├── config.py       # API Keys & LLM Configuration
│   │   ├── data_ingestion.py # Script to build Vector DB
│   │   ├── graph.py        # Graph construction (Nodes & Edges)
│   │   ├── nodes.py        # Core logic for Writer/Critic/Retriever
│   │   └── retriever.py    # Dual-track retrieval logic
│   ├── app.py              # Streamlit Web Interface (Task 2)
│   └── main_task2.py       # CLI Entry point for testing Task 2
├── .env                    # Environment variables (API Keys)
└── README.md               # This file

```

## Prerequisites & Installation

1. **Python Version**: Python 3.10+ is recommended.
2. **Dependencies**: Install the required packages.
```bash
pip install -r requirements1.txt
pip install -r requirements2.txt

```


*(Note: Key dependencies include `torch`, `langchain`, `langgraph`, `streamlit`, `chromadb`, `dashscope`.requirements1 for task1 and requirements2 for task2)*
3. **Environment Setup**:
Create a `.env` file in the root directory and add your Alibaba DashScope API Key:
```env
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx

```



## Task 1: Character-level LSTM

### How to Run

1. **Train the Model**:
To train the model from scratch (configured in `lstm_config.py`):
```bash
python src/task1_lstm/train.py

```


*Logs will be saved to `src/task1_lstm/logs/`.*
2. **Generate Text**:
To generate text using the trained model (`best_lstm_model.pth`):
```bash
python src/task1_lstm/generate.py

```


3. **Visualize Results**:
To plot the training and validation loss curves from logs:
```bash
python src/task1_lstm/plot_loss.py

```



## Task 2: RAG-based Interactive Agent

### How to Run

1. **Initialize Knowledge Base**:
Process the data and build the ChromaDB vector store (Lore & Style indices). *Note: `chroma_db/` is already included, run this only if you need to rebuild it.*
```bash
python src/task2_llm/data_ingestion.py

```


2. **Run the Interactive Interface**:
Launch the Streamlit web application:
```bash
streamlit run src/app.py

```


The app will open in your default browser (usually http://localhost:8501).

## Model Storage

Due to platform file size limits, the comprehensive training checkpoints and the pre-built vector database are hosted externally.

* **Download Link**: [Baidu Netdisk (百度网盘)](https://pan.baidu.com/s/1v79hfFouqbr9-pdIKLKsZw?pwd=8cnx)
* **Access Code**: `8cnx`

### 1. External Content Description
The link contains two main folders:

* **`task1_model/`**: Contains LSTM checkpoints trained with different Learning Rates (LR) for comparison experiments:
    * `best_lstm_model00005.pth`: Best model with **LR=0.0005** (Recommended/Default).
    * `best_lstm_model0002.pth`: Best model with **LR=0.002** (Used for stability comparison).
    * `latest_model*.pth`: The final checkpoints after the last epoch for respective LRs.

* **`chroma_db/`**: The complete, pre-embedded Vector Database for Task 2.
    * Includes the persisted SQLite3 database and binary index files (`*.bin`) for both the **Lore Index** and **Style Index**.

### 2. How to Use

#### For Task 1 (Reproducing Experiments)
The submission already includes the best default model (`best_lstm_model.pth`) in `src/task1_lstm/`. However, if you wish to evaluate the models with different learning rates mentioned in the report:
1.  Download the desired `.pth` file from the `task1_model` folder.
2.  Rename it to `best_lstm_model.pth`.
3.  Overwrite the existing file in:
    ```text
    src/task1_lstm/best_lstm_model.pth
    ```
4.  Run the generation script: `python src/task1_lstm/generate.py`

#### For Task 2 (Skipping Data Ingestion)
To run the Agent without rebuilding the vector database from scratch (which takes time):
1.  Download the entire `chroma_db` folder.
2.  Place it in the **root directory** of the project so that the structure looks like this:
    ```text
    MACHINEL_SUBMIT/
    ├── chroma_db/      <-- Place downloaded folder here
    ├── src/
    ├── .env
    └── ...
    ```
3.  Run the application directly: `streamlit run src/app.py`
## References

* **Dataset**: *The Lord of the Rings* Trilogy (J.R.R. Tolkien).
* **Course Material**: *Machine Learning Lab 4* (Base code structure for LSTM implementation).
* **Key Papers**:
    * **Task 1 (Character-Level Generation)**: Graves, A. (2013). *Generating Sequences With Recurrent Neural Networks*. arXiv preprint arXiv:1308.0850. (Foundational work on character-level text generation).
    * **Task 1 (Limitations)**: Fan, A., Lewis, M., & Dauphin, Y. (2018). *Hierarchical Neural Story Generation*. ACL 2018. (Discusses the long-range dependency limitations of non-hierarchical RNNs).
* **Libraries Used**: PyTorch, LangChain, LangGraph, Streamlit, ChromaDB.

