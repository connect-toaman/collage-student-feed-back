import os
import sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from app.config import Config
from app.database import db

class TestConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    SECRET_KEY = 'test-secret'
    ADMIN_PASSWORD_HASH = 'pbkdf2:sha256:600000$dummy$dummy'

@pytest.fixture
def client():
    app = create_app(TestConfig)
    with app.test_client() as client:
        with app.app_context():
            db.create_all()
        yield client

def test_home_page(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b"College Student Feedback Sentiment Analysis" in response.data

def test_submit_feedback(client):
    response = client.post('/predict', data={
        'student_name': 'Test User',
        'department': 'Computer Science',
        'category': 'Teacher',
        'feedback_text': 'This is a great test feedback!'
    }, follow_redirects=True)
    
    assert response.status_code == 200
    assert b"Your Feedback Analysis" in response.data
    
def test_submit_empty_feedback(client):
    response = client.post('/predict', data={
        'student_name': 'Test User',
        'department': '',
        'category': '',
        'feedback_text': ''
    }, follow_redirects=True)
    
    assert response.status_code == 200
    assert b"Please fill in all required fields" in response.data

def test_admin_login_page(client):
    response = client.get('/admin/login')
    assert response.status_code == 200
    assert b"Admin Login" in response.data

def test_dashboard_unauthorized(client):
    response = client.get('/admin/dashboard', follow_redirects=True)
    assert response.status_code == 200
    assert b"Admin Login" in response.data  # Redirects to login
