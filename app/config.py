import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'default-dev-key')
    
    # Use SQLite for local development, PostgreSQL for production
    # If on Vercel without DATABASE_URL, use /tmp/ to avoid Read-Only error
    default_db = 'sqlite:////tmp/feedback.db' if os.environ.get('VERCEL') else 'sqlite:///feedback.db'
    DATABASE_URL = os.environ.get('DATABASE_URL', default_db)
    
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
        
    SQLALCHEMY_DATABASE_URI = DATABASE_URL
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    ADMIN_PASSWORD_HASH = os.environ.get('ADMIN_PASSWORD_HASH', 'pbkdf2:sha256:600000$dummy$dummy')
