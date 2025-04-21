import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class EmailClassifier:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(max_features=5000)
        
    def preprocess_text(self, text):
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        words = nltk.word_tokenize(text)
        
        # Remove stopwords and lemmatize
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]
        
        return ' '.join(words)
    
    def load_and_preprocess_data(self, file_path):
        # Load the dataset
        df = pd.read_csv(file_path, encoding='latin-1')
        
        # Rename columns to match our expected format
        df = df.rename(columns={'v1': 'label', 'v2': 'text'})
        
        # Convert labels to consistent format
        df['label'] = df['label'].map({'ham': 'ham', 'spam': 'spam'})
        
        # Preprocess the text
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        
        return df
    
    def train_and_evaluate(self, df):
        # Split the data
        X = self.vectorizer.fit_transform(df['processed_text'])
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize classifiers
        classifiers = {
            'Naive Bayes': MultinomialNB(),
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, clf in classifiers.items():
            # Train the classifier
            clf.fit(X_train, y_train)
            
            # Make predictions
            y_pred = clf.predict(X_test)
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, pos_label='spam'),
                'recall': recall_score(y_test, y_pred, pos_label='spam'),
                'f1': f1_score(y_test, y_pred, pos_label='spam')
            }
            
            # Create confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            results[name] = {
                'metrics': metrics,
                'confusion_matrix': cm,
                'model': clf
            }
            
            # Save the best model
            if name == 'Random Forest':  # Using Random Forest as the default model
                joblib.dump(clf, 'spam_classifier.joblib')
                joblib.dump(self.vectorizer, 'vectorizer.joblib')
        
        return results
    
    def plot_confusion_matrix(self, cm, title):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'confusion_matrix_{title.lower().replace(" ", "_")}.png')
        plt.close()

def main():
    # Initialize the classifier
    classifier = EmailClassifier()
    
    # Load and preprocess the data
    df = classifier.load_and_preprocess_data('spam_ham_dataset.csv')
    
    # Train and evaluate models
    results = classifier.train_and_evaluate(df)
    
    # Print results and save confusion matrices
    for name, result in results.items():
        print(f"\n{name} Results:")
        for metric, value in result['metrics'].items():
            print(f"{metric}: {value:.4f}")
        
        classifier.plot_confusion_matrix(result['confusion_matrix'], name)

if __name__ == "__main__":
    main() 