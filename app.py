import streamlit as st
import joblib
import numpy as np
import pandas as pd
from train_model import EmailClassifier
import matplotlib.pyplot as plt
import seaborn as sns

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

# Title and description
st.title("📧 AI-Powered Email Spam Classifier")
st.markdown("""
This app uses machine learning to classify emails as spam or not spam. 
Enter an email text below and click the 'Classify' button to get the prediction.
""")

# Load the model and vectorizer
@st.cache_resource
def load_model():
    try:
        classifier = joblib.load('spam_classifier.joblib')
        vectorizer = joblib.load('vectorizer.joblib')
        return classifier, vectorizer
    except:
        st.error("Model files not found. Please train the model first using train_model.py")
        return None, None

# Load model statistics
@st.cache_data
def load_model_stats():
    try:
        # Load the dataset to show some statistics
        df = pd.read_csv('spam_ham_dataset.csv', encoding='latin-1')
        df = df.rename(columns={'v1': 'label', 'v2': 'text'})
        
        # Calculate basic statistics
        total_emails = len(df)
        spam_count = (df['label'] == 'spam').sum()
        ham_count = (df['label'] == 'ham').sum()
        
        # Calculate average text length
        df['text_length'] = df['text'].str.len()
        avg_length = df['text_length'].mean()
        
        return {
            'total_emails': total_emails,
            'spam_count': spam_count,
            'ham_count': ham_count,
            'avg_length': avg_length,
            'spam_percentage': (spam_count / total_emails) * 100
        }
    except:
        return None

# Display model statistics
st.markdown("### 📊 Model Statistics")
stats = load_model_stats()
if stats:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Emails", f"{stats['total_emails']:,}")
    with col2:
        st.metric("Spam Emails", f"{stats['spam_count']:,}")
    with col3:
        st.metric("Ham Emails", f"{stats['ham_count']:,}")
    with col4:
        st.metric("Avg Length", f"{stats['avg_length']:.0f} chars")

    # Create a pie chart for spam/ham distribution
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie([stats['spam_count'], stats['ham_count']], 
           labels=['Spam', 'Ham'], 
           autopct='%1.1f%%',
           colors=['#ff6b6b', '#4CAF50'])
    ax.set_title('Email Distribution')
    st.pyplot(fig)

# Initialize the email classifier for preprocessing
email_classifier = EmailClassifier()

# Get user input
st.markdown("### ✉️ Email Classification")
email_text = st.text_area("Enter the email text:", height=200)

if st.button("Classify Email"):
    if email_text.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        # Load the model
        classifier, vectorizer = load_model()
        
        if classifier and vectorizer:
            # Preprocess the text
            processed_text = email_classifier.preprocess_text(email_text)
            
            # Vectorize the text
            X = vectorizer.transform([processed_text])
            
            # Make prediction
            prediction = classifier.predict(X)[0]
            probability = classifier.predict_proba(X)[0]
            
            # Display results
            if prediction == 'spam':
                st.error(f"🔴 This email is classified as SPAM (confidence: {probability[1]:.2%})")
            else:
                st.success(f"✅ This email is classified as NOT SPAM (confidence: {probability[0]:.2%})")
            
            # Show probability distribution
            st.markdown("### Probability Distribution")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Not Spam", f"{probability[0]:.2%}")
            with col2:
                st.metric("Spam", f"{probability[1]:.2%}")

# Add footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: white;'>
        <p>Built with ❤️ using Streamlit</p>
    </div>
    """, unsafe_allow_html=True) 