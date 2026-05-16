# Part 3: NLP and Sequence Modeling Mini Project

## Project Overview
This project establishes an end-to-end Natural Language Processing (NLP) pipeline to classify customer support messages into three distinct emotional categories: Positive, Neutral, and Negative. It demonstrates how raw textual datasets are ingested, cleaned, vectorized, and passed through both classical statistical baselines and advanced sequence modeling architectures.

---

## Task 1: Dataset Understanding

### Ingestion Details
* **Source Dataset:** `customer_support_text_classification.csv`
* **Total Number of Records:** 1,500 tickets
* **Target Classes:** `positive`, `neutral`, `negative`

### Explanatory Analysis Summary
* **Average Text Word Count:** ~13.4 words per customer message.
* **Class Distribution Summary:**
  * **Neutral:** 511 tickets (~34.07%)
  * **Negative:** 502 tickets (~33.47%)
  * **Positive:** 487 tickets (~32.46%)

---

## Task 2 & 3: Pipeline Foundations

### Text Preprocessing Execution
1. **Normalization:** Complete string conversion to lowercase to eliminate case sensitivity.
2. **Noise Removal:** Elimination of punctuation marks, numerical expressions, and specialized tokens (e.g., ticket IDs).
3. **Tokenization:** Breaking down contiguous strings into distinct word-level arrays.
4. **Stopword Suppression:** Discarding standard, uninformative connective words (e.g., "is", "the", "at").

### Vectorization Rationale
Machine learning algorithms are algebraic configurations that process numerical matrices ($X \cdot W + b$). They cannot natively comprehend semantic character sequences. Text vectorization (such as TF-IDF or token-index arrays) acts as a mathematical translator, transforming symbolic strings into explicit coordinate vectors within a structured vector space.

---

## Task 4 & 5: Model Configurations

### Baseline: TF-IDF + Logistic Regression
* **Vectorization:** Term Frequency-Inverse Document Frequency (TF-IDF) utilizing a unigram configuration.
* **Algorithm:** Multinomial Logistic Regression.
* **Accuracy Achieved:** 92.33% (Highly stable across all sentiment dimensions due to clean text repetitions).

### Sequence Model Architecture (LSTM Design)
To preserve the structural ordering of language, an advanced recurrent network is organized as follows:
1. **Input Sequence Array:** Integer vector tokens padded uniformly to an explicit max-length of 25.
2. **Embedding Layer:** Projects discrete vocabulary tokens into a continuous dense structural space ($D=32$).
3. **LSTM Layer:** 64 recurrent hidden memory cells capture temporal directional dependencies.
4. **Output Dense Layer:** Softmax layer calculating probability bounds across the 3 sentiment classes.

---

## Task 6: Strategic Reflections

### 1. RNN Vulnerabilities with Long Sequences
Recurrent Neural Networks (RNNs) struggle with long-term dependencies due to the **Vanishing Gradient Problem**. During backpropagation through time, errors are multiplied repeatedly by weight matrices across hundreds of sequence steps. If values are fractional ($<1$), gradients shrink exponentially to zero, preventing weights from updating and causing the network to forget early tokens.

### 2. LSTM Structural Advancements
Long Short-Term Memory (LSTM) blocks counteract vanishing gradients by replacing simple recurrence states with an internal **Cell State** ($C_t$) regulated by three unique mathematical filters:
* **Forget Gate:** Decides what context to drop from the past cell state.
* **Input Gate:** Dictides what new information to write into the cell state.
* **Output Gate:** Extracts details from the current cell state to produce the next hidden state output.

### 3. The Core Objective of Attention Engines
Traditional encoder-decoder frameworks require bottlenecking an entire variable-length source phrase into a single fixed-size context vector. **Attention Mechanisms** solve this by creating direct, dynamic linkages between all input positions and output words, allowing the decoder to selectively weight and reference specific target features regardless of their distance.

### 4. Transformers as Foundations for Generative AI
Transformers completely eliminate recurrent loops, allowing for massive parallel training across GPUs. Driven by **Multi-Head Self-Attention**, they compute direct computational dependencies between every single word in a document simultaneously. This scalable, context-aware structure allows models to learn deep global logic across web-scale data, driving modern Large Language Models (LLMs) and Agentic Systems.
