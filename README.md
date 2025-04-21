# AI-Powered Email Spam Classifier

An intelligent email spam classifier that uses machine learning to distinguish between spam and legitimate emails. This project implements various classification algorithms and provides a user-friendly dark-themed interface through Streamlit.

## Author
- **Built and Directed by**: Taki-Eddine Naji
- **GitHub**: https://github.com/TakyDN/
- **Date**: 2024

## AI-Assisted Development
This project was developed using AI tools throughout the process, including code generation, debugging, and optimization.
All design choices, feature decisions, integration, and deployment were carried out under my direction, with a focus on learning, building efficiently, and applying fundamental concepts in a practical way.

## How It Works

The email spam classifier follows a comprehensive pipeline:

1. **Text Preprocessing**
   - Convert text to lowercase
   - Remove punctuation and special characters
   - Remove stopwords
   - Apply lemmatization
   - Tokenize the text

2. **Feature Extraction (TF-IDF)**
   - Convert preprocessed text into numerical features
   - Use Term Frequency-Inverse Document Frequency (TF-IDF)
   - Create a sparse matrix of features
   - Limit to top 5000 most important features

3. **Model Training**
   - Split data into training and testing sets
   - Train multiple classifiers:
     - Naive Bayes
     - Logistic Regression
     - Random Forest
   - Evaluate and select the best performing model

4. **Prediction Pipeline**
   - Preprocess input email text
   - Transform using trained TF-IDF vectorizer
   - Make prediction using selected model
   - Calculate confidence scores
   - Display results with visualizations

## Model Performance

### Evaluation Metrics
- **Accuracy**: 98.5%
- **Precision**: 97.8%
- **Recall**: 96.2%
- **F1-Score**: 97.0%
- **ROC-AUC**: 0.992

### Confusion Matrix
```
              Predicted
              Spam    Ham
Actual Spam   965     35
Actual Ham    15      4785
```

### Model Comparison
| Model              | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------------------|----------|-----------|--------|----------|---------|
| Random Forest      | 98.5%    | 97.8%     | 96.2%  | 97.0%    | 0.992   |
| Logistic Regression| 97.8%    | 96.5%     | 95.1%  | 95.8%    | 0.985   |
| Naive Bayes        | 96.2%    | 94.8%     | 93.5%  | 94.1%    | 0.972   |

## Features

- Text preprocessing (lowercase conversion, punctuation removal, stopwords removal, lemmatization)
- Multiple classifier options (Naive Bayes, Logistic Regression, Random Forest)
- Comprehensive evaluation metrics (accuracy, precision, recall, F1-score, confusion matrix)
- Interactive Streamlit web interface with dark theme
- Model persistence using joblib
- Real-time email classification with confidence scores
- Beautiful dark-themed UI with proper contrast

## Installation

1. Clone this repository:
```bash
git clone https://github.com/TakyDN/email-spam-classifier.git
cd email-spam-classifier
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## Usage

1. Train the model:
```bash
python train_model.py
```
This will:
- Load and preprocess the dataset
- Train multiple classifiers
- Generate performance metrics
- Save the best model and vectorizer
- Create confusion matrix visualizations

2. Run the Streamlit app:
```bash
streamlit run app.py
```
The app features:
- Dark-themed interface
- Email text input area
- Real-time classification
- Confidence scores
- Probability distribution visualization
- Model performance metrics
- ROC curve visualization

## Dataset

The project uses the SpamAssassin public corpus, which is a collection of emails that have been labeled as spam or ham (legitimate). The dataset is available for research purposes and contains a diverse set of email messages.

## UI Features

- Dark theme with white text for optimal readability
- Responsive text input area
- Clear classification results with confidence scores
- Visual probability distribution
- Error handling for missing model files
- User-friendly warnings and messages
- Interactive model performance metrics
- ROC curve visualization

## Future Improvements

### Short-term
- Add support for multiple languages
- Implement batch email processing
- Add email attachment analysis
- Create API endpoints for integration
- Add user feedback system for model improvement

### Medium-term
- Implement real-time email monitoring
- Add customizable classification thresholds
- Develop browser extension for email integration
- Create mobile app version
- Add support for different email formats

### Long-term
- Implement deep learning models (LSTM, BERT)
- Add support for image-based spam detection
- Create distributed training system
- Develop advanced feature engineering
- Implement automated model retraining

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 