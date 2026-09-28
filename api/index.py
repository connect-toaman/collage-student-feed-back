import os
import sys
import nltk

# Add project root to python path for local execution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set NLTK data path to /tmp for Vercel serverless environment
nltk_data_dir = os.environ.get('NLTK_DATA', '/tmp/nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)
try:
    nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
    nltk.download('punkt_tab', download_dir=nltk_data_dir, quiet=True)
    nltk.download('wordnet', download_dir=nltk_data_dir, quiet=True)
except Exception as e:
    print(f"Error downloading NLTK data: {e}")

from app import create_app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True)
