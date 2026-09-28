import os
import sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from app.config import Config
from app.database import db
from app.models import Feedback

class TestConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'

@pytest.fixture
def app():
    app = create_app(TestConfig)
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()

def test_feedback_insertion(app):
    with app.app_context():
        feedback = Feedback(
            student_name="Test Student",
            department="IT",
            category="Infrastructure",
            feedback_text="Good infrastructure.",
            sentiment="Positive",
            confidence=0.85
        )
        db.session.add(feedback)
        db.session.commit()
        
        saved_feedback = Feedback.query.first()
        assert saved_feedback is not None
        assert saved_feedback.student_name == "Test Student"
        assert saved_feedback.sentiment == "Positive"

def test_feedback_retrieval(app):
    with app.app_context():
        feedback1 = Feedback(department="CS", category="Teacher", feedback_text="Test 1", sentiment="Positive", confidence=0.9)
        feedback2 = Feedback(department="ME", category="Hostel", feedback_text="Test 2", sentiment="Negative", confidence=0.8)
        
        db.session.add_all([feedback1, feedback2])
        db.session.commit()
        
        feedbacks = Feedback.query.all()
        assert len(feedbacks) == 2
        assert feedbacks[0].department == "CS"
        assert feedbacks[1].department == "ME"
