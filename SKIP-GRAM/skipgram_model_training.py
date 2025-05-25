import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
import re
import time
import html
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from gensim.models import Word2Vec

# Download necessary NLTK resources
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

# Start timer
start_time = time.time()

# Create output directory for results
os.makedirs('model_results', exist_ok=True)

print("Loading the datasets...")
# Load the training dataset
df_train = pd.read_csv('../complete_cleaned_dataset.csv')
print(f"Training dataset loaded with shape: {df_train.shape}")

# Load the testing dataset
df_test = pd.read_csv('../test_dataset.csv')
print(f"Testing dataset loaded with shape: {df_test.shape}")

# Display class distribution for training data
print("\nTraining data class distribution (full dataset):")
print(df_train['type'].value_counts())

print("\nTesting data class distribution (full dataset):")
print(df_test['type'].value_counts())

# First, sample a subset of the original data to reduce processing time
print("\nReducing overall dataset size...")
# You can adjust these values to control how much data to use
sample_train_size = 45000  # Set to 40000 to use ~90% of the data, adjust as needed
sample_test_size = 6000    # Set to 6000 to use all test data, adjust as needed

# Sample the datasets if they're larger than the specified sizes
if len(df_train) > sample_train_size:
    df_train = df_train.sample(sample_train_size, random_state=42)
    print(f"Sampled training data to {sample_train_size} examples")

if len(df_test) > sample_test_size:
    df_test = df_test.sample(sample_test_size, random_state=42)
    print(f"Sampled testing data to {sample_test_size} examples")

# Create a more balanced dataset by keeping all directives and sampling other classes
print("\nCreating a balanced dataset...")

# Split the training data by class
df_train_directive = df_train[df_train['type'] == 'Directive']
df_train_regulation = df_train[df_train['type'] == 'Regulation']
df_train_decision = df_train[df_train['type'] == 'Decision']

# Keep all directives, sample 20% of other classes
df_train_regulation_sampled = df_train_regulation.sample(frac=0.20, random_state=42)
df_train_decision_sampled = df_train_decision.sample(frac=0.20, random_state=42)

# Combine the datasets
df_train = pd.concat([df_train_directive, df_train_regulation_sampled, df_train_decision_sampled])

# Do the same for test data
df_test_directive = df_test[df_test['type'] == 'Directive']
df_test_regulation = df_test[df_test['type'] == 'Regulation']
df_test_decision = df_test[df_test['type'] == 'Decision']

# Keep all directives, sample 20% of other classes
df_test_regulation_sampled = df_test_regulation.sample(frac=0.20, random_state=42)
df_test_decision_sampled = df_test_decision.sample(frac=0.20, random_state=42)

# Combine the datasets
df_test = pd.concat([df_test_directive, df_test_regulation_sampled, df_test_decision_sampled])

# Shuffle the data
df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
df_test = df_test.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nBalanced training data shape:", df_train.shape)
print("Balanced testing data shape:", df_test.shape)

print("\nBalanced training data class distribution:")
print(df_train['type'].value_counts())

print("\nBalanced testing data class distribution:")
print(df_test['type'].value_counts())

# Text preprocessing functions
# Define custom stopwords based on document type-specific terms
custom_stopwords = [
    # Decision-specific terms
    'decision', 'decisions',
    'this decision', 'decision is', 'decision that', 'decision shall',
    # Regulation-specific terms
    'regulation', 'regulations', 'this regulation', 'regulation that', 'the regulation',
    'the directive', 
    # Directive-specific terms
    'directive', 'directives', 'this directive', 'the directive', 'directive is', 
    'directive shall', 'directive to'
]

# Define text preprocessing functions
def remove_punctuation(text):
    if isinstance(text, str):
        return text.translate(str.maketrans('', '', string.punctuation))
    return ""

def remove_stopwords(text):
    if isinstance(text, str):
        stop_words = set(stopwords.words('english')).union(custom_stopwords)
        return ' '.join([word for word in text.split() if word.lower() not in stop_words])
    return ""

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # Decode HTML entities
    text = html.unescape(text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip().lower()
    
    return text

# Function to tokenize text
def tokenize_text(text):
    if not isinstance(text, str):
        return []
    
    # Tokenize
    tokens = nltk.word_tokenize(text.lower())
    
    # Filter tokens
    clean_tokens = [token for token in tokens 
                  if token.isalpha() and 
                  token.lower() not in stopwords.words('english') and
                  token not in custom_stopwords]
    
    return clean_tokens

print("\nPreparing and cleaning text features...")
# Process training data
print("Processing training data...")
# Handle NaN values before combining
df_train['header'] = df_train['header'].fillna('')
df_train['recitals'] = df_train['recitals'].fillna('')
df_train['main_body'] = df_train['main_body'].fillna('')

# Combine all text columns into a single feature for training data
df_train['combined_text'] = df_train['header'] + " " + df_train['recitals'] + " " + df_train['main_body']

# Process test data
print("Processing testing data...")
# Handle NaN values before combining
df_test['header'] = df_test['header'].fillna('')
df_test['recitals'] = df_test['recitals'].fillna('')
df_test['main_body'] = df_test['main_body'].fillna('')

# Combine all text columns into a single feature for test data
df_test['combined_text'] = df_test['header'] + " " + df_test['recitals'] + " " + df_test['main_body']

# Double-check for any remaining NaN values
if df_train['combined_text'].isnull().sum() > 0:
    print(f"Warning: {df_train['combined_text'].isnull().sum()} NaN values found in training data. Filling with empty strings.")
    df_train['combined_text'] = df_train['combined_text'].fillna('')

if df_test['combined_text'].isnull().sum() > 0:
    print(f"Warning: {df_test['combined_text'].isnull().sum()} NaN values found in test data. Filling with empty strings.")
    df_test['combined_text'] = df_test['combined_text'].fillna('')

# Clean text
df_train['combined_text'] = df_train['combined_text'].apply(clean_text)
df_test['combined_text'] = df_test['combined_text'].apply(clean_text)

# Tokenize the text for Skip-gram model
print("Tokenizing text for Skip-gram model...")
df_train['tokens'] = df_train['combined_text'].apply(tokenize_text)
df_test['tokens'] = df_test['combined_text'].apply(tokenize_text)

# Print statistics about training data
avg_train_length = df_train['combined_text'].str.len().mean()
max_train_length = df_train['combined_text'].str.len().max()
min_train_length = df_train['combined_text'].str.len().min()

print(f"\nTraining data statistics:")
print(f"Training data - Average text length: {avg_train_length:.2f} characters")
print(f"Training data - Max text length: {max_train_length} characters")
print(f"Training data - Min text length: {min_train_length} characters")

# Print statistics about test data
avg_test_length = df_test['combined_text'].str.len().mean()
max_test_length = df_test['combined_text'].str.len().max()
min_test_length = df_test['combined_text'].str.len().min()

print(f"Testing data - Average text length: {avg_test_length:.2f} characters")
print(f"Testing data - Min text length: {min_test_length} characters")

# Prepare features and target
X_train = df_train['combined_text']
y_train = df_train['type']
X_test = df_test['combined_text']
y_test = df_test['type']

# Encode the target variable
print("\nEncoding target variable...")
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

print("Encoded labels:")
for i, class_name in enumerate(label_encoder.classes_):
    print(f"{i}: {class_name}")

# Train Skip-gram model
print("\nTraining Skip-gram Word2Vec model...")
# Train Word2Vec using Skip-gram (sg=1)
w2v_model = Word2Vec(
    sentences=df_train['tokens'],
    vector_size=100,  # Dimensionality of the word vectors
    window=5,         # Maximum distance between current and predicted word
    min_count=2,      # Ignore words with fewer occurrences
    sg=1,             # Use Skip-gram (sg=1), not CBOW (sg=0)
    workers=4,        # Number of CPU cores
    epochs=5          # Number of iterations over the corpus
)

print(f"Skip-gram model trained on {len(w2v_model.wv.key_to_index)} words")
print(f"Vector size: {w2v_model.wv.vector_size}")

# Create document vectors from word vectors
def document_vector(tokens, model, vector_size=100):
    # Initialize empty vector
    doc_vector = np.zeros(vector_size)
    
    # Count valid tokens
    valid_tokens = 0
    
    # Sum embeddings for each token
    for token in tokens:
        if token in model.wv:
            doc_vector += model.wv[token]
            valid_tokens += 1
    
    # Average the embeddings
    if valid_tokens > 0:
        doc_vector /= valid_tokens
    
    return doc_vector

# Create document vectors for training and test sets
print("\nCreating document vectors using Skip-gram embeddings...")
X_train_skipgram = np.array([document_vector(tokens, w2v_model) for tokens in df_train['tokens']])
X_test_skipgram = np.array([document_vector(tokens, w2v_model) for tokens in df_test['tokens']])

print(f"Skip-gram features shape for training: {X_train_skipgram.shape}")
print(f"Skip-gram features shape for testing: {X_test_skipgram.shape}")

# Function to train and evaluate a model
def train_and_evaluate(model, name, X_train, X_test, y_train, y_test, needs_encoded=False):
    print(f"\n{'-'*20} Training {name} {'-'*20}")
    start = time.time()
    
    # Use encoded labels if needed
    train_y = y_train_encoded if needs_encoded else y_train
    test_y = y_test_encoded if needs_encoded else y_test
    
    # Train the model
    model.fit(X_train, train_y)
    train_time = time.time() - start
    
    # Make predictions
    if needs_encoded:
        y_pred_encoded = model.predict(X_test)
        y_pred = label_encoder.inverse_transform(y_pred_encoded)
    else:
        y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Print results
    print(f"{name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Training time: {train_time:.2f} seconds")
    
    # Save detailed report to file
    with open(f"model_results/{name.replace(' ', '_').lower()}_skipgram_report.txt", 'w') as f:
        f.write(f"{name} Results:\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Training time: {train_time:.2f} seconds\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred))
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {name} (Skip-gram)')
    plt.tight_layout()
    plt.savefig(f"model_results/{name.replace(' ', '_').lower()}_skipgram_confusion_matrix.png")
    plt.close()
    
    # Save the model
    joblib.dump(model, f"model_results/{name.replace(' ', '_').lower()}_skipgram_model.joblib")
    
    return {
        'name': name,
        'accuracy': accuracy,
        'training_time': train_time,
        'model': model
    }

# Train models using Skip-gram features
results = []

# 1. Multinomial Naive Bayes - need to make features non-negative
print("\nTraining Multinomial Naive Bayes model...")
# Add small constant and scale features to make them non-negative
X_train_mnb = X_train_skipgram - X_train_skipgram.min(axis=0)
X_test_mnb = X_test_skipgram - X_train_skipgram.min(axis=0)
mnb_model = MultinomialNB()
mnb_results = train_and_evaluate(mnb_model, "Multinomial Naive Bayes", X_train_mnb, X_test_mnb, y_train, y_test)
results.append(mnb_results)

# 2. SVM model
print("\nTraining SVM model...")
svm_model = LinearSVC(C=1.0, dual=False, class_weight='balanced', max_iter=2000)
svm_results = train_and_evaluate(svm_model, "SVM", X_train_skipgram, X_test_skipgram, y_train, y_test)
results.append(svm_results)

# 3. Gaussian Naive Bayes
print("\nTraining Gaussian Naive Bayes model...")
gnb_model = GaussianNB()
gnb_results = train_and_evaluate(gnb_model, "Gaussian Naive Bayes", X_train_skipgram, X_test_skipgram, y_train, y_test)
results.append(gnb_results)

# 4. Logistic Regression
print("\nTraining Logistic Regression model...")
lr_model = LogisticRegression(max_iter=1000, class_weight='balanced')
lr_results = train_and_evaluate(lr_model, "Logistic Regression", X_train_skipgram, X_test_skipgram, y_train, y_test)
results.append(lr_results)

# 5. Random Forest
print("\nTraining Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=5, random_state=42)
rf_results = train_and_evaluate(rf_model, "Random Forest", X_train_skipgram, X_test_skipgram, y_train, y_test)
results.append(rf_results)

# Find the best model
best_model = max(results, key=lambda x: x['accuracy'])

print("\n" + "="*70)
print("MODEL COMPARISON SUMMARY (Skip-gram Features)")
print("="*70)
print(f"{'Model':<25} {'Accuracy':<15} {'Training Time':<15}")
print("-"*70)
for result in sorted(results, key=lambda x: x['accuracy'], reverse=True):
    print(f"{result['name']:<25} {result['accuracy']:<15.4f} {result['training_time']:<15.2f}s")

print("\n" + "="*70)
print(f"BEST MODEL: {best_model['name']} with accuracy {best_model['accuracy']:.4f}")
print("="*70)

# Save the Skip-gram model and label encoder for future use
print("\nSaving Skip-gram model and label encoder for future use...")
w2v_model.save("model_results/skipgram_model.model")
joblib.dump(label_encoder, "model_results/label_encoder_skipgram.joblib")

# Calculate execution time
end_time = time.time()
execution_time = end_time - start_time
print(f"\nTotal execution time: {execution_time:.2f} seconds")
print("\nSkip-gram model training completed successfully. Results saved in 'model_results' directory.")
