import os
import sys
import nltk

# Add project root to python path for local execution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set NLTK data path to the local nltk_data directory included in the repo
nltk_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'nltk_data')
nltk.data.path.append(nltk_data_dir)

from app import create_app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True)
