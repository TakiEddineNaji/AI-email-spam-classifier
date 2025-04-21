"""
AI-Powered Email Spam Classifier - Streamlit Interface
Created by: Taki Eddine Naji
GitHub: https://github.com/TakyDN/
Date: 2024

This module provides a user-friendly interface for the email spam classifier
using Streamlit, featuring a dark theme and real-time classification.
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
from train_model import EmailClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, f1_score
import os
from typing import Dict, Tuple, Optional, Union

# Constants
REQUIRED_FILES = ['spam_classifier.joblib', 'vectorizer.joblib', 'spam_ham_dataset.csv']
DATASET_ENCODING = 'latin-1'
PLOT_COLORS = {
    'spam': '#ff6b6b',
    'ham': '#4CAF50',
    'roc': '#4CAF50',
    'diagonal': 'gray'
}

# Set page config
st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #1E1E1E;
    }
    /* Fix title color */
    h1 {
        color: white !important;
    }
    /* Fix footer color */
    footer {
        color: white !important;
    }
    .stTextInput>div>div>input {
        border-radius: 4px;
        background-color: #2D2D2D;
        color: white;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border-radius: 4px;
        border: none;
    }
    /* Fix text color */
    .stMarkdown {
        color: white;
    }
    .stTextInput>div>div>input {
        color: white;
    }
    .stTextArea>div>div>textarea {
        color: white;
        background-color: #2D2D2D;
    }
    /* Fix metric colors */
    .stMetric {
        color: white;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: white;
    }
    .stMetric [data-testid="stMetricLabel"] {
        color: white;
    }
    /* Fix footer text */
    .footer {
        color: white !important;
    }
    /* Fix results text */
    .element-container {
        color: white;
    }
    .stAlert {
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

def check_required_files() -> list:
    """Check if all required files exist."""
    return [f for f in REQUIRED_FILES if not os.path.exists(f)]

@st.cache_resource
def load_model() -> Tuple[Optional[EmailClassifier], Optional[object]]:
    """Load the trained model and vectorizer."""
    try:
        classifier = joblib.load('spam_classifier.joblib')
        vectorizer = joblib.load('vectorizer.joblib')
        return classifier, vectorizer
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        return None, None

@st.cache_data
def load_model_stats() -> Optional[Dict]:
    """Load and calculate model statistics and metrics."""
    try:
        if not os.path.exists('spam_ham_dataset.csv'):
            st.error("Dataset file not found. Please ensure 'spam_ham_dataset.csv' is in the current directory.")
            return None

        # Load and preprocess dataset
        df = pd.read_csv('spam_ham_dataset.csv', encoding=DATASET_ENCODING)
        df = df.rename(columns={'v1': 'label', 'v2': 'text'})
        
        # Calculate basic statistics
        stats = {
            'total_emails': len(df),
            'spam_count': (df['label'] == 'spam').sum(),
            'ham_count': (df['label'] == 'ham').sum(),
            'avg_length': df['text'].str.len().mean()
        }
        
        # Load model and calculate metrics
        classifier, vectorizer = load_model()
        if classifier and vectorizer:
            X = vectorizer.transform(df['text'])
            y_true = df['label'].values
            y_pred = classifier.predict(X)
            y_prob = classifier.predict_proba(X)[:, 1]
            
            # Calculate metrics
            y_true_binary = (y_true == 'spam').astype(int)
            fpr, tpr, _ = roc_curve(y_true_binary, y_prob)
            
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'f1_score': f1_score(y_true, y_pred, pos_label='spam'),
                'roc_auc': auc(fpr, tpr),
                'confusion_matrix': confusion_matrix(y_true, y_pred, labels=['ham', 'spam']),
                'fpr': fpr,
                'tpr': tpr
            }
            
            return {**stats, 'metrics': metrics}
        return None
    except Exception as e:
        st.error(f"Error loading model statistics: {str(e)}")
        return None

def create_pie_chart(stats: Dict) -> None:
    """Create and display pie chart for email distribution."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie([stats['spam_count'], stats['ham_count']], 
           labels=['Spam', 'Ham'], 
           autopct='%1.1f%%',
           colors=[PLOT_COLORS['spam'], PLOT_COLORS['ham']])
    ax.set_title('Email Distribution')
    st.pyplot(fig)

def create_confusion_matrix(metrics: Dict) -> None:
    """Create and display confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(metrics['confusion_matrix'], 
                annot=True, 
                fmt='d',
                cmap='Blues',
                xticklabels=['Ham', 'Spam'],
                yticklabels=['Ham', 'Spam'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

def create_roc_curve(metrics: Dict) -> None:
    """Create and display ROC curve."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(metrics['fpr'], metrics['tpr'], 
            color=PLOT_COLORS['roc'], 
            lw=2, 
            label=f'ROC curve (AUC = {metrics["roc_auc"]:.3f})')
    ax.plot([0, 1], [0, 1], 
            color=PLOT_COLORS['diagonal'], 
            lw=1, 
            linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    st.pyplot(fig)

def display_metrics(stats: Dict) -> None:
    """Display all model metrics and visualizations."""
    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Emails", f"{stats['total_emails']:,}")
    with col2:
        st.metric("Spam Emails", f"{stats['spam_count']:,}")
    with col3:
        st.metric("Ham Emails", f"{stats['ham_count']:,}")
    with col4:
        st.metric("Avg Length", f"{stats['avg_length']:.0f} chars")

    # Visualizations
    st.markdown("### 📊 Email Distribution")
    create_pie_chart(stats)

    st.markdown("### 📈 Model Performance")
    metrics = stats['metrics']
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    with col2:
        st.metric("F1-Score", f"{metrics['f1_score']:.2%}")
    with col3:
        st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

    st.markdown("### Confusion Matrix")
    create_confusion_matrix(metrics)

    st.markdown("### ROC Curve")
    create_roc_curve(metrics)

def classify_email(email_text: str, classifier: EmailClassifier, vectorizer: object) -> None:
    """Classify the input email and display results."""
    try:
        # Initialize EmailClassifier for preprocessing
        email_classifier = EmailClassifier()
        processed_text = email_classifier.preprocess_text(email_text)
        X = vectorizer.transform([processed_text])
        prediction = classifier.predict(X)[0]
        probability = classifier.predict_proba(X)[0]
        
        if prediction == 'spam':
            st.error(f"🔴 This email is classified as SPAM (confidence: {probability[1]:.2%})")
        else:
            st.success(f"✅ This email is classified as NOT SPAM (confidence: {probability[0]:.2%})")
        
        st.markdown("### Probability Distribution")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Not Spam", f"{probability[0]:.2%}")
        with col2:
            st.metric("Spam", f"{probability[1]:.2%}")
    except Exception as e:
        st.error(f"Error during classification: {str(e)}")

def main():
    """Main application function."""
    # Title and description
    st.title("📧 AI-Powered Email Spam Classifier")
    st.markdown("""
    This app uses machine learning to classify emails as spam or not spam. 
    Enter an email text below and click the 'Classify' button to get the prediction.
    """)

    # Check for missing files
    missing_files = check_required_files()
    if missing_files:
        st.error(f"Missing required files: {', '.join(missing_files)}")
        st.info("Please ensure all required files are present in the current directory.")
        return

    # Display model statistics and metrics
    st.markdown("### 📊 Model Statistics")
    stats = load_model_stats()
    
    if stats:
        display_metrics(stats)

    # Email classification interface
    st.markdown("### ✉️ Email Classification")
    email_text = st.text_area("Enter the email text:", height=200)

    if st.button("Classify Email"):
        if not email_text.strip():
            st.warning("Please enter some text to classify.")
            return

        classifier, vectorizer = load_model()
        if classifier and vectorizer:
            classify_email(email_text, classifier, vectorizer)
        else:
            st.error("Model not loaded. Please ensure model files are present.")

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: white;'>
            <p>Built and directed by Taki-Eddine Naji — AI-assisted for faster development and practical learning.</p>
            <p>GitHub: <a href='https://github.com/TakyDN/AI-email-spam-classifier' style='color: #4CAF50;'>https://github.com/TakyDN/</a></p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 