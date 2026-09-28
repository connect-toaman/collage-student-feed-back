# College Student Feedback Sentiment Analysis

## 1. Abstract
This project presents an automated College Student Feedback Sentiment Analysis system developed using Python, Flask, and Machine Learning. The application aims to collect feedback from students regarding various aspects of their college experience (teaching, infrastructure, administration) and automatically classify the sentiment of the feedback into Positive, Neutral, or Negative categories. The system utilizes Natural Language Processing (NLP) techniques and a Logistic Regression classification model to perform the sentiment analysis.

## 2. Introduction
Student feedback is a critical component of institutional growth and quality assurance in educational institutions. Traditionally, analyzing large volumes of unstructured textual feedback is labor-intensive and subjective. This project automates the process by employing Machine Learning to analyze the sentiment of submitted feedback in real-time, providing actionable insights to the college administration through an interactive dashboard.

## 3. Problem Statement
Manual analysis of student feedback is time-consuming, prone to human error, and difficult to scale. There is a need for an automated system that can efficiently collect textual feedback, understand the underlying sentiment of the student's message, and present aggregated analytics to administrators to help them make informed decisions.

## 4. Objectives
- To develop a responsive web application for collecting student feedback.
- To implement NLP techniques for text preprocessing and feature extraction.
- To train a Machine Learning model capable of classifying text sentiment.
- To create an admin dashboard for data visualization and export.
- To ensure the system is deployable to modern cloud platforms (Vercel).

## 5. Technology Used
- **Programming Language**: Python 3
- **Web Framework**: Flask
- **Machine Learning**: Scikit-Learn, NLTK
- **Frontend**: HTML5, CSS3, Vanilla JavaScript, Chart.js
- **Database**: SQLite (Local), PostgreSQL (Production)
- **Deployment**: Vercel

## 6. System Architecture & Data Flow
1. **User Input**: Student submits feedback via the web form.
2. **Backend Processing**: The Flask application receives the POST request.
3. **NLP Preprocessing**: The text is cleaned (lowercase, punctuation removal, lemmatization) while preserving negations.
4. **Feature Extraction**: TF-IDF Vectorizer converts the text into numerical features.
5. **Prediction**: The trained Logistic Regression model predicts the sentiment.
6. **Storage**: The feedback and prediction are saved to the database.
7. **Output**: The user sees the analysis result, and the admin dashboard updates with the new data.

## 7. Machine Learning Methodology
### 7.1 Dataset
A custom dataset consisting of realistic student feedback across various departments and categories is used for training the model. The data contains examples of Positive, Neutral, and Negative sentiments.

### 7.2 Text Preprocessing (NLP)
- **Tokenization**: Breaking text into individual words.
- **Lemmatization**: Reducing words to their base or root form.
- **Negation Preservation**: Important sentiment modifiers like "not" and "never" are preserved before removing special characters to maintain context.

### 7.3 TF-IDF Explanation
Term Frequency-Inverse Document Frequency (TF-IDF) is used to convert the textual data into numerical vectors. It evaluates how relevant a word is to a document in a collection of documents.

### 7.4 Logistic Regression Explanation
Logistic Regression is a statistical model used for classification. In this project, a multinomial Logistic Regression model is trained on the TF-IDF vectors to predict the probability of the feedback belonging to one of the three sentiment classes.

## 8. Results & Testing
The system underwent rigorous testing including:
- Unit testing for ML model predictions.
- Integration testing for Flask routes and database operations.
- UI/UX testing for responsiveness.
The model effectively classifies sentiments with high accuracy, and the admin dashboard successfully visualizes the distribution of feedback.

## 9. Limitations & Future Scope
**Limitations**: The model's vocabulary is limited to its training data. Extremely sarcastic or complex nuanced sentences might be misclassified.
**Future Scope**: Implement advanced Deep Learning models (like LSTMs or Transformers) for better contextual understanding, and expand the dataset to cover more diverse feedback scenarios.

## 10. Conclusion
The College Student Feedback Sentiment Analysis system provides an efficient, automated solution for institutions to gauge student satisfaction. By leveraging Machine Learning and a robust web architecture, the system transforms unstructured text into structured, actionable insights.
