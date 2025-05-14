# Multiple Concepts and Cross-Attention Based Knowledge Graph Completion

To address the challenge of insufficient latent feature capture in low-dimensional embeddings for knowledge graph completion (KGC). Our MRCCA advances the field through three key innovations:&#x20;



1️⃣ Multi-Semantic Projection: Systematically decomposes entity/relation embeddings into multi-dimensional semantic representations via orthogonal feature space projections.&#x20;

2️⃣ Latent Feature Factorization: Hierarchically refines semantic units into disentangled subspaces using learnable transformation matrices, enabling multi-granularity feature extraction.&#x20;

3️⃣ Lightweight Cross-Attention Architecture: Dynamically models entity-relation interactions&#x20;



through parameter-shared attention layers while maintaining computational efficiency . Experimental validation on standard link prediction benchmarks (FB15k-237, WN18RR, and kinship) demonstrates state-of-the-art performance, with MRCCA achieving 5.2% MRR improvement over baseline models.



#### Running MRCCA

**STEP 1: Installation**

1.  Install [python](https://www.python.org/),. We use Python 3.9.&#x20;

2.  If you plan to use GPU computation, install [CUDA](https://developer.nvidia.com/cuda-downloads). conda install pytorch\=\=2.21.1 torchvision\=\=0.17.1 torchaudio\=\=2.2.1 cudatoolkit\=11.3 -c pytorch

3.  Download/clone the MRCCA code



**STEP 2: How to prepare  dataset**

1.  Download the KG datasets (FB15k-237, WN18RR, and kinship), including the training set, validation set, and test set. Place them under the data directory.



**STEP 4: Running **MRCCA

The model can be directly executed by running:

    python main.py

**Parameter Configuration**\
All hyperparameters and experimental settings are defined in `main.py`. Key adjustable parameters include:

*   **Embedding dimensions** (`embedding_dim`)

*   **Training epochs** (`num_epochs`)

*   **Batch size** (`batch_size`)

*   **Learning rate** (`learning_rate`)

**Notes**

1.  The script automatically loads the default configuration from `main.py`

2.  For custom experiments, modify the parameters directly in the source file

3.  No additional configuration files are required for basic usage

