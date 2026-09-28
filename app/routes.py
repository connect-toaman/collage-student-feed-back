from flask import Blueprint, render_template, request, redirect, url_for, session, flash, Response, current_app, jsonify
from werkzeug.security import check_password_hash
from app.models import Feedback, FeedbackAspect
from app.database import db
from app.sentiment import predict_sentiment
from ml.aspect_sentiment import analyze_aspects
import csv
import io

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/about')
def about():
    return render_template('about.html')

@main.route('/feedback')
def feedback():
    return render_template('feedback.html')

@main.route('/predict', methods=['POST'])
def predict():
    student_name = request.form.get('student_name', '')
    department = request.form.get('department')
    category = request.form.get('category')
    feedback_text = request.form.get('feedback_text')

    if not feedback_text or not department or not category:
        flash("Please fill in all required fields.")
        return redirect(url_for('main.feedback'))

    # Run sentiment analysis
    prediction = predict_sentiment(feedback_text)
    sentiment = prediction.get('sentiment', 'Unknown')
    confidence = prediction.get('confidence', 0.0)
    probabilities = prediction.get('probabilities', {})
    
    # Run ABSA
    absa_result = analyze_aspects(feedback_text)

    # Save to database
    feedback = Feedback(
        student_name=student_name,
        department=department,
        category=category,
        feedback_text=feedback_text,
        sentiment=sentiment,
        confidence=confidence
    )
    
    db.session.add(feedback)
    
    try:
        db.session.flush() # To get feedback.id
        for asp in absa_result['aspects']:
            fa = FeedbackAspect(
                feedback_id=feedback.id,
                aspect_name=asp['aspect'],
                sentiment=asp['sentiment'],
                confidence=asp['confidence'],
                context=asp['context']
            )
            db.session.add(fa)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        flash("An error occurred while saving feedback.")
        return redirect(url_for('main.index'))

    return render_template('result.html', 
                           sentiment=sentiment, 
                           confidence=confidence,
                           probabilities=probabilities,
                           absa_result=absa_result,
                           feedback_text=feedback_text)

@main.route('/api/model-info', methods=['GET'])
def model_info():
    import json
    import os
    metadata_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model', 'model_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    return jsonify({"error": "Model metadata not found"}), 404

@main.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    
    prediction = predict_sentiment(data['text'])
    return jsonify(prediction)

@main.route('/api/analyze', methods=['POST'])
def api_analyze():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
        
    prediction = predict_sentiment(data['text'])
    absa_result = analyze_aspects(data['text'])
    
    return jsonify({
        "text": data['text'],
        "overall_sentiment": {
            "label": prediction.get('sentiment'),
            "confidence": prediction.get('confidence'),
            "probabilities": prediction.get('probabilities')
        },
        "aspect_based_analysis": absa_result
    })


@main.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        password = request.form.get('password')
        hashed_password = current_app.config['ADMIN_PASSWORD_HASH']
        
        # Super simple check for dummy env
        if 'dummy' in hashed_password and password == 'admin':
            session['admin_logged_in'] = True
            return redirect(url_for('main.dashboard'))
            
        if check_password_hash(hashed_password, password):
            session['admin_logged_in'] = True
            return redirect(url_for('main.dashboard'))
        else:
            flash("Invalid password.")
            
    return render_template('admin_login.html')


@main.route('/admin/logout', methods=['POST'])
def admin_logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('main.index'))


@main.route('/admin/dashboard')
def dashboard():
    if not session.get('admin_logged_in'):
        return redirect(url_for('main.admin_login'))
        
    feedbacks = Feedback.query.order_by(Feedback.created_at.desc()).all()
    
    total = len(feedbacks)
    positive = sum(1 for f in feedbacks if f.sentiment == 'Positive')
    neutral = sum(1 for f in feedbacks if f.sentiment == 'Neutral')
    negative = sum(1 for f in feedbacks if f.sentiment == 'Negative')
    
    # Calculate Aspect stats
    aspect_counts = {}
    aspect_sentiments = {}
    for f in feedbacks:
        for a in f.aspects:
            name = a.aspect_name
            sent = a.sentiment
            aspect_counts[name] = aspect_counts.get(name, 0) + 1
            if name not in aspect_sentiments:
                aspect_sentiments[name] = {'Positive': 0, 'Negative': 0, 'Neutral': 0, 'Total': 0}
            aspect_sentiments[name][sent] += 1
            aspect_sentiments[name]['Total'] += 1
            
    top_aspects = sorted(aspect_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    neg_aspects = []
    pos_aspects = []
    for aspect, stats in aspect_sentiments.items():
        if stats['Total'] > 0:
            neg_pct = (stats['Negative'] / stats['Total']) * 100
            pos_pct = (stats['Positive'] / stats['Total']) * 100
            neg_aspects.append((aspect, neg_pct))
            pos_aspects.append((aspect, pos_pct))
            
    top_neg_aspects = sorted(neg_aspects, key=lambda x: x[1], reverse=True)[:5]
    top_pos_aspects = sorted(pos_aspects, key=lambda x: x[1], reverse=True)[:5]
    
    return render_template('dashboard.html', 
                           feedbacks=feedbacks,
                           total=total,
                           positive=positive,
                           neutral=neutral,
                           negative=negative,
                           top_aspects=top_aspects,
                           top_neg_aspects=top_neg_aspects,
                           top_pos_aspects=top_pos_aspects)


@main.route('/admin/export')
def export_csv():
    if not session.get('admin_logged_in'):
        return redirect(url_for('main.admin_login'))
        
    feedbacks = Feedback.query.all()
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['ID', 'Student Name', 'Department', 'Category', 'Feedback', 'Sentiment', 'Confidence', 'Date'])
    
    for f in feedbacks:
        writer.writerow([
            f.id, 
            f.student_name, 
            f.department, 
            f.category, 
            f.feedback_text, 
            f.sentiment, 
            f.confidence, 
            f.created_at.strftime("%Y-%m-%d %H:%M:%S")
        ])
        
    response = Response(output.getvalue(), mimetype='text/csv')
    response.headers["Content-Disposition"] = "attachment; filename=feedback_export.csv"
    return response

@main.route('/admin/export_aspects')
def export_aspects_csv():
    if not session.get('admin_logged_in'):
        return redirect(url_for('main.admin_login'))
        
    feedbacks = Feedback.query.all()
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Feedback_ID', 'Feedback_Text', 'Overall_Sentiment', 'Overall_Confidence', 'Aspect', 'Aspect_Sentiment', 'Aspect_Confidence', 'Aspect_Context', 'Date'])
    
    for f in feedbacks:
        if not f.aspects:
            continue
        for a in f.aspects:
            writer.writerow([
                f.id, 
                f.feedback_text, 
                f.sentiment, 
                f.confidence,
                a.aspect_name,
                a.sentiment,
                a.confidence,
                a.context,
                f.created_at.strftime("%Y-%m-%d %H:%M:%S")
            ])
            
    response = Response(output.getvalue(), mimetype='text/csv')
    response.headers["Content-Disposition"] = "attachment; filename=aspects_export.csv"
    return response
