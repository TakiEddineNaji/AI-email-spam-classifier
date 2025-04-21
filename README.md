# AI-Powered Email Spam Classifier

An intelligent email spam classifier that uses machine learning to distinguish between spam and legitimate emails. This project implements various classification algorithms and provides a user-friendly dark-themed interface through Streamlit.

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
git clone https://github.com/yourusername/email-spam-classifier.git
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

## Dataset

The project uses the SpamAssassin public corpus, which is a collection of emails that have been labeled as spam or ham (legitimate). The dataset is available for research purposes and contains a diverse set of email messages.

## Model Performance

The model provides the following evaluation metrics:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

Results are saved as PNG files for visual analysis.

## UI Features

- Dark theme with white text for optimal readability
- Responsive text input area
- Clear classification results with confidence scores
- Visual probability distribution
- Error handling for missing model files
- User-friendly warnings and messages

## Future Enhancements

- Email inbox integration
- CSV export of classification results
- Real-time email monitoring
- Customizable classification thresholds
- Multi-language support
- API endpoint for integration with other applications
- Light/dark theme toggle
- Batch email processing
- Custom model training interface

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 