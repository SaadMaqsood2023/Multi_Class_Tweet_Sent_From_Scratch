# TweetSent-LSTM  
**From-Scratch Multi-Class Sentiment Analysis on Social Media Text**

---

## Abstract
This repository presents a **complete, from-scratch implementation of a multi-class sentiment analysis system** using a manually implemented Long Short-Term Memory (LSTM) network.  
The objective of this project is to demonstrate a **fundamental understanding of deep learning architectures, sequence modeling, and training dynamics** without relying on high-level machine learning frameworks.

The model performs **three-class sentiment classification** (Positive, Neutral, Negative) on tweet-style text and is designed as a **research-oriented foundation** for further work in Natural Language Processing (NLP) and Deep Learning.

---

## Research Motivation
Most academic and industrial NLP pipelines rely heavily on abstraction layers provided by deep learning libraries. While efficient, these abstractions often obscure the **mathematical and algorithmic principles** behind sequence models.

This project was intentionally designed to:
- Expose the **internal mechanics of LSTM networks**
- Understand **gradient flow in recurrent architectures**
- Analyze **training stability and numerical behavior**
- Serve as a strong foundation for **advanced research in NLP and LLMs**

---

## Key Contributions
- Complete neural network implementation in **pure Python**
- Custom **Embedding Layer**
- Manual implementation of **LSTM gates and cell dynamics**
- **Softmax-based multi-class classification**
- Cross-entropy loss computation
- Gradient-based parameter updates
- Tweet-specific preprocessing pipeline
- Dataset-agnostic training support

---

## Model Architecture
Raw Text
↓
Text Normalization & Cleaning
↓
Tokenization & Vocabulary Mapping
↓
Trainable Embedding Layer
↓
LSTM Cell (From-Scratch)
↓
Final Hidden State
↓
Linear + Softmax Layer
↓
Sentiment Prediction


---

## Dataset Support
The system supports three data input formats:

### CSV Format
```csv
text,label
I love this product,1
The item is okay,0
Worst experience ever,-1
```

### TXT Format
```txt
I love this product|||1
The item is okay|||0
Worst experience ever|||-1
```

### Built-in Sample Dataset
If no external dataset is provided, the system automatically uses a balanced
internal dataset consisting of 60 tweet-style samples.

## Label Encoding
| Label | Sentiment |
| ----- | --------- |
| 1     | Positive  |
| 0     | Neutral   |
| -1    | Negative  |

## Implementation Details
###Components Implemented From Scratch
- Sigmoid, Tanh, Softmax
- Vector & Matrix operations
- Word Embeddings
- LSTM gates (Input, Forget, Output)
- Cell state and hidden state propagation
- Cross-entropy loss
- Gradient updates with clipping

## Library Used
This project deliberately avoids deep learning frameworks.
Python Standard Library Only:
math, random, re, csv

## Training Configuration
| Parameter        | Value |
| ---------------- | ----- |
| Embedding Size   | 32    |
| Hidden Units     | 48    |
| Epochs           | 80    |
| Max Sequence Len | 15    |
| Learning Rate    | 0.005 |

### Training reports:
Average loss per epoch
Classification accuracy

## Sample Inference
```txt
Tweet: "I love this product! Best purchase ever!"
Prediction: Positive (confidence: 92%)

Tweet: "The product arrived on time and works fine"
Prediction: Neutral (confidence: 63%)

Tweet: "Worst experience ever, very disappointed"
Prediction: Negative (confidence: 89%)
```

Significance for Graduate Research
This project demonstrates:
- Strong command of **deep learning fundamentals**
- Ability to reason about **recurrent neural architectures**
- Understanding of **training dynamics and numerical stability**
- Readiness for **research-level NLP and sequence modeling**
- A solid foundation for work on:
  - Attention mechanisms
  - Transformer models
  - Large Language Models (LLM)
  - Model interpretability and robustness

## Future Research Directions
- Bidirectional LSTM implementation
- Attention-augmented sequence modeling
- Transformer-based comparison studies
- Training on large-scale Twitter/X datasets
- Explainable AI for sentiment prediction
- Integration with modern embedding techniques

## Reproducibility
To train and evaluate the model:
```bash
python sentiment_analysis.py
```
The system automatically detects available datasets or falls back to the internal dataset.

## Author 
**Saad Maqsood**
Software Engineering Undergraduate
Research Interests: Deep Learning, NLP, Large Language Models

This repository is intended as a research-oriented portfolio project for 
MS / PhD applications in Artificial Intelligence and Computer Science.
