from datetime import datetime
from app.database import db

class Feedback(db.Model):
    __tablename__ = 'feedback'
    
    id = db.Column(db.Integer, primary_key=True)
    student_name = db.Column(db.String(100), nullable=True)
    department = db.Column(db.String(100), nullable=False)
    category = db.Column(db.String(100), nullable=False)
    feedback_text = db.Column(db.Text, nullable=False)
    sentiment = db.Column(db.String(20), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    aspects = db.relationship('FeedbackAspect', backref='feedback_ref', lazy=True, cascade="all, delete-orphan")

    def to_dict(self):
        return {
            'id': self.id,
            'student_name': self.student_name,
            'department': self.department,
            'category': self.category,
            'feedback_text': self.feedback_text,
            'sentiment': self.sentiment,
            'confidence': self.confidence,
            'created_at': self.created_at.isoformat(),
            'aspects': [a.to_dict() for a in self.aspects]
        }

class FeedbackAspect(db.Model):
    __tablename__ = 'feedback_aspect'
    
    id = db.Column(db.Integer, primary_key=True)
    feedback_id = db.Column(db.Integer, db.ForeignKey('feedback.id'), nullable=False)
    aspect_name = db.Column(db.String(100), nullable=False)
    sentiment = db.Column(db.String(20), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    context = db.Column(db.Text, nullable=False)
    
    def to_dict(self):
        return {
            'aspect': self.aspect_name,
            'sentiment': self.sentiment,
            'confidence': self.confidence,
            'context': self.context
        }
