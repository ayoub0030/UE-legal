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
from sklearn.feature_extraction.text import CountVectorizer

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

# Create a more balanced dataset by keeping all directives and sampling other classes
print("\nCreating a balanced dataset...")

# Split the training data by class
df_train_directive = df_train[df_train['type'] == 'Directive']
df_train_regulation = df_train[df_train['type'] == 'Regulation']
df_train_decision = df_train[df_train['type'] == 'Decision']

# Keep all directives, sample 10% of other classes
df_train_regulation_sampled = df_train_regulation.sample(frac=0.20, random_state=42)
df_train_decision_sampled = df_train_decision.sample(frac=0.20, random_state=42)

# Combine the datasets
df_train = pd.concat([df_train_directive, df_train_regulation_sampled, df_train_decision_sampled])

# Do the same for test data
df_test_directive = df_test[df_test['type'] == 'Directive']
df_test_regulation = df_test[df_test['type'] == 'Regulation']
df_test_decision = df_test[df_test['type'] == 'Decision']

# Keep all directives, sample 10% of other classes
df_test_regulation_sampled = df_test_regulation.sample(frac=0.20, random_state=42)
df_test_decision_sampled = df_test_decision.sample(frac=0.120, random_state=42)

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
    

    
    return text

# Function to tokenize text (without stemming)
def tokenize_and_clean(text):
    if not isinstance(text, str):
        return []
    
    # tokenize
    tokens = nltk.word_tokenize(text)
    
    # Filter tokens but skip stemming
    clean_tokens = tokens
    
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


# Tokenize the text for n-gram processing
print("Tokenizing text for N-gram processing...")
df_train['tokens'] = df_train['combined_text'].apply(tokenize_and_clean)
df_test['tokens'] = df_test['combined_text'].apply(tokenize_and_clean)

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

# Create N-gram features
print("\nCreating N-gram features...")
# We'll use unigrams and bigrams (1-gram and 2-gram)
ngram_vectorizer = CountVectorizer(
    ngram_range=(1, 2),  # Use both unigrams and bigrams
    max_features=2000,   # Limit to top 5000 features
    min_df=5,            # Ignore terms that appear in less than 5 documents
    stop_words=list(stopwords.words('english')) + custom_stopwords
)

# Fit and transform the training data
print("Fitting and transforming training data...")
X_train_ngrams = ngram_vectorizer.fit_transform(X_train)

# Transform the test data
print("Transforming test data...")
X_test_ngrams = ngram_vectorizer.transform(X_test)

print(f"N-gram features shape for training: {X_train_ngrams.shape}")
print(f"N-gram features shape for testing: {X_test_ngrams.shape}")

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
    with open(f"model_results/{name.replace(' ', '_').lower()}_ngram_report.txt", 'w') as f:
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
    plt.title(f'Confusion Matrix - {name} (N-grams)')
    plt.tight_layout()
    plt.savefig(f"model_results/{name.replace(' ', '_').lower()}_ngram_confusion_matrix.png")
    plt.close()
    
    # Save the model
    joblib.dump(model, f"model_results/{name.replace(' ', '_').lower()}_ngram_model.joblib")
    
    return {
        'name': name,
        'accuracy': accuracy,
        'training_time': train_time,
        'model': model
    }

# Train all models using N-gram features
results = []


# 2. SVM (LinearSVC is more efficient for text classification)
print("\nTraining SVM model...")
svm_model = LinearSVC(C=100, dual=False, class_weight='balanced', max_iter=2000)
svm_results = train_and_evaluate(svm_model, "SVM", X_train_ngrams, X_test_ngrams, y_train, y_test)
results.append(svm_results)




# Find the best model
best_model = max(results, key=lambda x: x['accuracy'])

print("\n" + "="*70)
print("MODEL COMPARISON SUMMARY (N-gram Features)")
print("="*70)
print(f"{'Model':<25} {'Accuracy':<15} {'Training Time':<15}")
print("-"*70)
for result in sorted(results, key=lambda x: x['accuracy'], reverse=True):
    print(f"{result['name']:<25} {result['accuracy']:<15.4f} {result['training_time']:<15.2f}s")

print("\n" + "="*70)
print(f"BEST MODEL: {best_model['name']} with accuracy {best_model['accuracy']:.4f}")
print("="*70)

# Save the vectorizer and label encoder for future use
print("\nSaving N-gram vectorizer and label encoder for future use...")
joblib.dump(ngram_vectorizer, "model_results/ngram_vectorizer.joblib")
joblib.dump(label_encoder, "model_results/label_encoder_ngram.joblib")

# Print top N-gram features for each class
print("\nTop N-gram Features for Each Class:")
feature_names = ngram_vectorizer.get_feature_names_out()
top_n = 20  # Number of top features to show

for i, class_name in enumerate(label_encoder.classes_):
    # Get the feature importances for this class (using log probabilities)
    feature_importances = mnb_model.feature_log_prob_[i]
    
    # Get the indices of top features
    top_indices = feature_importances.argsort()[-top_n:][::-1]
    
    # Print top features for this class
    print(f"\nTop {top_n} N-gram features for class '{class_name}':")
    for idx in top_indices:
        print(f"  - {feature_names[idx]} (log prob: {feature_importances[idx]:.4f})")

# Calculate execution time
end_time = time.time()
execution_time = end_time - start_time
print(f"\nTotal execution time: {execution_time:.2f} seconds")
print("\nAll models trained and evaluated successfully with N-gram features. Results saved in 'model_results' directory.")
