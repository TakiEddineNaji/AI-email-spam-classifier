import os
import pandas as pd
import requests
from zipfile import ZipFile
from io import BytesIO

def download_and_prepare_dataset():
    # URL for the SpamAssassin dataset
    url = "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham_2.tar.bz2"
    
    print("Downloading dataset...")
    response = requests.get(url)
    
    if response.status_code == 200:
        # Create a directory for the dataset if it doesn't exist
        if not os.path.exists('data'):
            os.makedirs('data')
        
        # Save the downloaded file
        with open('data/spam_dataset.tar.bz2', 'wb') as f:
            f.write(response.content)
        
        print("Dataset downloaded successfully!")
        
        # Note: The actual dataset processing would go here
        # For now, we'll create a sample CSV with the expected format
        sample_data = {
            'text': [
                'This is a legitimate email about our meeting tomorrow.',
                'WIN A FREE IPHONE! CLICK HERE NOW!!!',
                'Please review the attached documents for the project.',
                'URGENT: Your account has been compromised!',
                'Meeting reminder: Team sync at 2 PM today.'
            ],
            'label': ['ham', 'spam', 'ham', 'spam', 'ham']
        }
        
        df = pd.DataFrame(sample_data)
        df.to_csv('spam_ham_dataset.csv', index=False)
        print("Sample dataset created as spam_ham_dataset.csv")
        
    else:
        print(f"Failed to download dataset. Status code: {response.status_code}")

if __name__ == "__main__":
    download_and_prepare_dataset() 