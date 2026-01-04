import math
import random
import re
import csv

# ================================
# 1. DATASET LOADING FUNCTIONS
# ================================
def load_dataset_from_csv(filename):
    """
    Load dataset from CSV file
    Expected format: text,label (where label is -1, 0, or 1)
    Example row: "I love this product",1
    """
    texts = []
    labels = []
    
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header if exists
            
            for row in reader:
                if len(row) >= 2:
                    texts.append(row[0])
                    label = int(row[1])
                    # Ensure label is -1, 0, or 1
                    if label not in [-1, 0, 1]:
                        print(f"Warning: Invalid label {label}, skipping row")
                        continue
                    labels.append(label)
        
        print(f"Loaded {len(texts)} samples from {filename}")
        return texts, labels
    
    except FileNotFoundError:
        print(f"File {filename} not found. Using sample dataset instead.")
        return None, None

def load_dataset_from_text(filename):
    """
    Load dataset from text file
    Expected format: Each line contains: text|||label
    Example: I love this product|||1
    """
    texts = []
    labels = []
    
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if '|||' in line:
                    text, label = line.split('|||')
                    label = int(label.strip())
                    if label not in [-1, 0, 1]:
                        print(f"Warning: Invalid label {label}, skipping row")
                        continue
                    texts.append(text.strip())
                    labels.append(label)
        
        print(f"Loaded {len(texts)} samples from {filename}")
        return texts, labels
    
    except FileNotFoundError:
        print(f"File {filename} not found. Using sample dataset instead.")
        return None, None

# ================================
# 2. SAMPLE DATASET (FALLBACK)
# ================================
def get_sample_dataset():
    """Sample tweet-style dataset with 3 classes: positive (1), neutral (0), negative (-1)"""
    texts = [
        # Positive samples (label = 1)
        "I absolutely love this product it is amazing",
        "Great quality fast shipping highly recommend",
        "So happy with my purchase exceeded expectations",
        "Best purchase ever five stars all the way",
        "Excellent service and fantastic product",
        "Love it works perfectly as described",
        "Amazing product exceeded my expectations",
        "Fantastic quality worth every penny",
        "Super happy great value for money",
        "Highly recommend great product love it",
        "Perfect exactly what I needed thanks",
        "Wonderful experience will buy again",
        "Best decision ever so satisfied",
        "Love this amazing quality and fast delivery",
        "Great product works like a charm",
        "Absolutely fantastic highly recommended",
        "So pleased with this purchase",
        "Excellent value money well spent",
        "Really happy great product love it",
        "Perfect product great service",
        
        # Neutral samples (label = 0)
        "The product arrived on time",
        "It is okay nothing special",
        "Average quality meets basic expectations",
        "The product works as described",
        "Standard quality for the price",
        "It does what it says acceptable",
        "Normal experience nothing remarkable",
        "The item is fine but not impressive",
        "Decent product neither good nor bad",
        "It is alright serves its purpose",
        "The quality is okay average product",
        "Received the item as expected",
        "Standard packaging and delivery",
        "The product is functional",
        "It works but nothing extraordinary",
        "Acceptable quality for daily use",
        "The item meets minimum requirements",
        "Average experience nothing to complain about",
        "It is what it is no surprises",
        "Neutral feelings about this purchase",
        
        # Negative samples (label = -1)
        "This is the worst thing I have ever bought",
        "Terrible experience will never buy again",
        "Complete waste of money very disappointed",
        "Do not buy this total scam",
        "Horrible quality broke after one day",
        "Awful experience terrible customer service",
        "Worst purchase of my life regret buying",
        "Garbage product do not waste your money",
        "Disappointed poor quality not as advertised",
        "Total disaster nothing worked properly",
        "Useless product returned immediately",
        "Terrible quality not worth the price",
        "Horrible do not recommend at all",
        "Worst experience ever very unhappy",
        "Awful waste of time and money",
        "Complete disappointment terrible product",
        "Not good at all very poor quality",
        "Rubbish total waste avoid this",
        "Bad quality broke immediately",
        "Terrible avoid at all costs"
    ]
    
    labels = [
        # 20 positive
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        # 20 neutral
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        # 20 negative
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
    ]
    
    return texts, labels

# Try to load from file, otherwise use sample
print("Attempting to load dataset...")
print("Supported formats:")
print("  1. CSV: 'tweets_dataset.csv' with columns: text,label")
print("  2. TXT: 'tweets_dataset.txt' with format: text|||label")
print("  Labels: 1 (positive), 0 (neutral), -1 (negative)")
print()

texts, labels = load_dataset_from_csv('tweet_reviews.csv')
# if texts is None:
#     texts, labels = load_dataset_from_text('tweets_dataset.txt')
if texts is None:
    print("Using built-in sample dataset (60 samples)")
    texts, labels = get_sample_dataset()

print(f"Dataset size: {len(texts)} samples")

# Count class distribution
pos_count = sum(1 for l in labels if l == 1)
neu_count = sum(1 for l in labels if l == 0)
neg_count = sum(1 for l in labels if l == -1)
print(f"Class distribution: Positive={pos_count}, Neutral={neu_count}, Negative={neg_count}")
print()

# ================================
# 3. MATH UTILITIES
# ================================
def sigmoid(x):
    """Sigmoid activation function"""
    return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))

def tanh(x):
    """Tanh activation function"""
    x = max(-500, min(500, x))
    return math.tanh(x)

def softmax(logits):
    """Softmax activation for multi-class classification"""
    # Subtract max for numerical stability
    max_logit = max(logits)
    exp_logits = [math.exp(x - max_logit) for x in logits]
    sum_exp = sum(exp_logits)
    return [x / sum_exp for x in exp_logits]

def dot_product(vec1, vec2):
    """Compute dot product of two vectors"""
    return sum(a * b for a, b in zip(vec1, vec2))

def matrix_vector_mult(matrix, vector):
    """Multiply matrix by vector
    matrix: list of lists, shape (output_size, input_size)
    vector: list, shape (input_size,)
    returns: list, shape (output_size,)
    """
    if not matrix:
        return []
    
    output_size = len(matrix)
    input_size = len(matrix[0]) if matrix else 0
    
    if len(vector) != input_size:
        raise ValueError(f"Matrix-vector dimension mismatch: matrix has {input_size} columns, vector has {len(vector)} elements")
    
    result = []
    for row in matrix:
        result.append(dot_product(row, vector))
    
    return result

def vector_add(vec1, vec2):
    """Add two vectors"""
    return [a + b for a, b in zip(vec1, vec2)]

def vector_mult(vec1, vec2):
    """Element-wise multiplication"""
    return [a * b for a, b in zip(vec1, vec2)]

def scalar_mult(scalar, vec):
    """Multiply vector by scalar"""
    return [scalar * v for v in vec]

# ================================
# 4. TEXT PREPROCESSING
# ================================
def clean_text(text):
    """Clean and normalize text for tweets/social media"""
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove user mentions and hashtags (keep the text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    # Remove special characters but keep spaces
    text = re.sub(r"[^a-z\s]", "", text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Clean all texts
print("Cleaning text data...")
texts = [clean_text(t) for t in texts]

# ================================
# 5. LABEL ENCODING
# ================================
def encode_label(label):
    """Convert label (-1, 0, 1) to one-hot encoding [neg, neu, pos]"""
    if label == -1:
        return [1, 0, 0]  # Negative
    elif label == 0:
        return [0, 1, 0]  # Neutral
    else:  # label == 1
        return [0, 0, 1]  # Positive

def decode_prediction(probs):
    """Convert probability distribution to label"""
    max_idx = probs.index(max(probs))
    if max_idx == 0:
        return -1  # Negative
    elif max_idx == 1:
        return 0   # Neutral
    else:
        return 1   # Positive

# ================================
# 6. VOCABULARY BUILDING
# ================================
def build_vocab(texts, min_freq=1):
    """Build vocabulary from texts with minimum frequency threshold"""
    word_counts = {}
    for text in texts:
        for word in text.split():
            word_counts[word] = word_counts.get(word, 0) + 1
    
    vocab = {"<PAD>": 0}
    idx = 1
    for word, count in word_counts.items():
        if count >= min_freq:
            vocab[word] = idx
            idx += 1
    
    return vocab

vocab = build_vocab(texts, min_freq=1)
vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size} unique words")

# ================================
# 7. EMBEDDING LAYER
# ================================
class Embedding:
    def __init__(self, vocab_size, embed_size):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.weights = [[random.gauss(0, 0.01) for _ in range(embed_size)] 
                       for _ in range(vocab_size)]
    
    def forward(self, word_idx):
        return self.weights[word_idx][:]
    
    def backward(self, word_idx, grad):
        for i in range(self.embed_size):
            self.weights[word_idx][i] -= grad[i]

# ================================
# 8. LSTM CELL
# ================================
class LSTMCell:
    def __init__(self, input_size, hidden_size, learning_rate=0.001):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lr = learning_rate
        
        # Input gate
        # W_xi shape: (hidden_size, input_size) - each row produces one hidden unit
        self.W_xi = [[random.gauss(0, 0.01) for _ in range(input_size)] 
                     for _ in range(hidden_size)]
        self.W_hi = [[random.gauss(0, 0.01) for _ in range(hidden_size)] 
                     for _ in range(hidden_size)]
        self.b_i = [0.0] * hidden_size
        
        # Forget gate
        self.W_xf = [[random.gauss(0, 0.01) for _ in range(input_size)] 
                     for _ in range(hidden_size)]
        self.W_hf = [[random.gauss(0, 0.01) for _ in range(hidden_size)] 
                     for _ in range(hidden_size)]
        self.b_f = [1.0] * hidden_size
        
        # Output gate
        self.W_xo = [[random.gauss(0, 0.01) for _ in range(input_size)] 
                     for _ in range(hidden_size)]
        self.W_ho = [[random.gauss(0, 0.01) for _ in range(hidden_size)] 
                     for _ in range(hidden_size)]
        self.b_o = [0.0] * hidden_size
        
        # Cell state
        self.W_xc = [[random.gauss(0, 0.01) for _ in range(input_size)] 
                     for _ in range(hidden_size)]
        self.W_hc = [[random.gauss(0, 0.01) for _ in range(hidden_size)] 
                     for _ in range(hidden_size)]
        self.b_c = [0.0] * hidden_size
    
    def forward(self, x, h_prev, c_prev):
        # Ensure input dimensions are correct
        if len(x) != self.input_size:
            raise ValueError(f"Input size mismatch: got {len(x)}, expected {self.input_size}")
        if len(h_prev) != self.hidden_size:
            raise ValueError(f"Hidden state size mismatch: got {len(h_prev)}, expected {self.hidden_size}")
        if len(c_prev) != self.hidden_size:
            raise ValueError(f"Cell state size mismatch: got {len(c_prev)}, expected {self.hidden_size}")
        
        # Input gate
        i_raw = vector_add(vector_add(matrix_vector_mult(self.W_xi, x),
                                      matrix_vector_mult(self.W_hi, h_prev)),
                          self.b_i)
        i_t = [sigmoid(val) for val in i_raw]
        
        # Forget gate
        f_raw = vector_add(vector_add(matrix_vector_mult(self.W_xf, x),
                                      matrix_vector_mult(self.W_hf, h_prev)),
                          self.b_f)
        f_t = [sigmoid(val) for val in f_raw]
        
        # Output gate
        o_raw = vector_add(vector_add(matrix_vector_mult(self.W_xo, x),
                                      matrix_vector_mult(self.W_ho, h_prev)),
                          self.b_o)
        o_t = [sigmoid(val) for val in o_raw]
        
        # Candidate cell state
        c_tilde_raw = vector_add(vector_add(matrix_vector_mult(self.W_xc, x),
                                           matrix_vector_mult(self.W_hc, h_prev)),
                                self.b_c)
        c_tilde = [tanh(val) for val in c_tilde_raw]
        
        # Cell state
        c_t = vector_add(vector_mult(f_t, c_prev), vector_mult(i_t, c_tilde))
        
        # Hidden state
        c_t_tanh = [tanh(val) for val in c_t]
        h_t = vector_mult(o_t, c_t_tanh)
        
        # Verify output dimensions
        if len(h_t) != self.hidden_size:
            raise ValueError(f"Output hidden state size mismatch: got {len(h_t)}, expected {self.hidden_size}")
        if len(c_t) != self.hidden_size:
            raise ValueError(f"Output cell state size mismatch: got {len(c_t)}, expected {self.hidden_size}")
        
        cache = {
            'x': x, 'h_prev': h_prev, 'c_prev': c_prev,
            'i_t': i_t, 'f_t': f_t, 'o_t': o_t,
            'c_tilde': c_tilde, 'c_t': c_t, 'h_t': h_t
        }
        
        return h_t, c_t, cache

# ================================
# 9. OUTPUT LAYER (Multi-class)
# ================================
class MultiClassOutputLayer:
    def __init__(self, hidden_size, num_classes=3, learning_rate=0.001):
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.lr = learning_rate
        # Initialize weights: shape (hidden_size, num_classes)
        self.W = []
        for i in range(hidden_size):
            row = [random.gauss(0, 0.01) for _ in range(num_classes)]
            self.W.append(row)
        self.b = [0.0] * num_classes
    
    def forward(self, h):
        """Forward pass with softmax"""
        # Ensure h has correct length
        if len(h) != self.hidden_size:
            raise ValueError(f"Hidden state size mismatch: got {len(h)}, expected {self.hidden_size}")
        
        logits = []
        for j in range(self.num_classes):
            # Compute h @ W[:, j] + b[j]
            logit = 0.0
            for i in range(self.hidden_size):
                logit += h[i] * self.W[i][j]
            logit += self.b[j]
            logits.append(logit)
        return softmax(logits)
    
    def backward(self, h, grad_output):
        """Backward pass and update weights"""
        # grad_output is the gradient of loss w.r.t. output probabilities
        # Update weights
        for i in range(self.hidden_size):
            for j in range(self.num_classes):
                self.W[i][j] -= self.lr * grad_output[j] * h[i]
        
        for j in range(self.num_classes):
            self.b[j] -= self.lr * grad_output[j]
        
        # Return gradient for hidden state
        grad_h = [sum(self.W[i][j] * grad_output[j] 
                     for j in range(self.num_classes))
                 for i in range(self.hidden_size)]
        return grad_h

# ================================
# 10. COMPLETE LSTM MODEL
# ================================
class SentimentLSTM:
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes=3, learning_rate=0.001):
        self.embedding = Embedding(vocab_size, embed_size)
        self.lstm = LSTMCell(embed_size, hidden_size, learning_rate)
        self.output_layer = MultiClassOutputLayer(hidden_size, num_classes, learning_rate)
        self.hidden_size = hidden_size
        self.num_classes = num_classes
    
    def forward(self, word_indices):
        h = [0.0] * self.hidden_size
        c = [0.0] * self.hidden_size
        
        caches = []
        embeddings = []
        
        # Handle empty sequences
        if not word_indices or all(idx == 0 for idx in word_indices):
            # Return zero hidden state for empty sequences
            output_probs = self.output_layer.forward(h)
            return output_probs, h, caches, embeddings, word_indices
        
        for word_idx in word_indices:
            x = self.embedding.forward(word_idx)
            embeddings.append(x)
            h, c, cache = self.lstm.forward(x, h, c)
            caches.append(cache)
        
        # Verify h has correct length before passing to output layer
        if len(h) != self.hidden_size:
            raise ValueError(f"LSTM hidden state size mismatch: got {len(h)}, expected {self.hidden_size}")
        
        output_probs = self.output_layer.forward(h)
        
        return output_probs, h, caches, embeddings, word_indices
    
    def compute_loss(self, y_pred, y_true):
        """Cross-entropy loss for multi-class classification"""
        eps = 1e-8
        loss = -sum(y_true[i] * math.log(y_pred[i] + eps) 
                   for i in range(self.num_classes))
        return loss
    
    def backward(self, y_pred, y_true, h_final, word_indices):
        """Backward pass"""
        # Gradient of cross-entropy loss w.r.t. softmax output
        grad_loss = [y_pred[i] - y_true[i] for i in range(self.num_classes)]
        
        # Gradient clipping
        grad_norm = math.sqrt(sum(g * g for g in grad_loss))
        if grad_norm > 1.0:
            grad_loss = [g / grad_norm for g in grad_loss]
        
        # Backprop through output layer
        grad_h = self.output_layer.backward(h_final, grad_loss)
        
        # Simplified embedding update
        for word_idx in word_indices:
            grad_embed = scalar_mult(0.0001, grad_h[:self.embedding.embed_size])
            self.embedding.backward(word_idx, grad_embed)

# ================================
# 11. TEXT TO INDICES
# ================================
def text_to_indices(text, vocab, max_len=15):
    """Convert text to list of word indices"""
    words = text.split()[:max_len]
    indices = [vocab.get(word, 0) for word in words]
    while len(indices) < max_len:
        indices.append(0)
    return indices

# ================================
# 12. TRAINING
# ================================
print("\n" + "="*50)
print("TRAINING 3-CLASS SENTIMENT ANALYSIS LSTM")
print("="*50)

# Hyperparameters
embed_size = 32
hidden_size = 48
learning_rate = 0.005
epochs = 80
max_len = 15

# Initialize model
random.seed(42)
model = SentimentLSTM(vocab_size, embed_size, hidden_size, num_classes=3, learning_rate=learning_rate)

# Training loop
print(f"\nTraining for {epochs} epochs...")
for epoch in range(epochs):
    total_loss = 0.0
    correct = 0
    
    # Shuffle data
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts_shuffled, labels_shuffled = zip(*combined)
    
    for text, label in zip(texts_shuffled, labels_shuffled):
        indices = text_to_indices(text, vocab, max_len)
        y_true = encode_label(label)
        
        y_pred, h_final, caches, embeddings, word_indices = model.forward(indices)
        loss = model.compute_loss(y_pred, y_true)
        total_loss += loss
        model.backward(y_pred, y_true, h_final, word_indices)
        
        prediction = decode_prediction(y_pred)
        if prediction == label:
            correct += 1
    
    accuracy = correct / len(texts) * 100
    avg_loss = total_loss / len(texts)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

# ================================
# 13. PREDICTION FUNCTION
# ================================
def predict_sentiment(text, model, vocab, max_len=15):
    """Predict sentiment of text"""
    cleaned = clean_text(text)
    indices = text_to_indices(cleaned, vocab, max_len)
    y_pred, _, _, _, _ = model.forward(indices)
    
    prediction = decode_prediction(y_pred)
    
    sentiment_map = {-1: "Negative", 0: "Neutral", 1: "Positive"}
    sentiment = sentiment_map[prediction]
    
    confidence = max(y_pred)
    
    return sentiment, confidence, y_pred

# ================================
# 14. TESTING
# ================================
print("\n" + "="*50)
print("TESTING ON SAMPLE TWEETS")
print("="*50)

test_samples = [
    "I love this product! Best purchase ever! #happy",
    "This is terrible, worst experience ever",
    "The product arrived on time and works fine",
    "@company your service is amazing, highly recommend!",
    "It is okay, nothing special but acceptable",
    "Complete waste of money, very disappointed",
    "Fantastic quality, exceeded expectations!",
    "Standard quality, meets basic requirements",
    "Awful product, broke after one day #disappointed",
    "The item is decent, neither good nor bad",
    "So happy with this! Works perfectly!",
    "Average experience, no complaints",
]

for sample in test_samples:
    sentiment, confidence, probs = predict_sentiment(sample, model, vocab)
    print(f"\nTweet: '{sample}'")
    print(f"Prediction: {sentiment} (confidence: {confidence:.2%})")
    print(f"Probabilities: Negative={probs[0]:.3f}, Neutral={probs[1]:.3f}, Positive={probs[2]:.3f}")

print("\n" + "="*50)
print("\nTo use your own dataset:")
print("1. CSV format: Create 'tweets_dataset.csv' with columns: text,label")
print("2. TXT format: Create 'tweets_dataset.txt' with lines: text|||label")
print("   Labels: 1 (positive), 0 (neutral), -1 (negative)")
print("   Example: I love this product|||1")
print("="*50)