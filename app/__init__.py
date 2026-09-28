from flask import Flask
from app.config import Config
from app.database import db

def create_app(config_class=Config):
    app = Flask(__name__.split('.')[0], 
                template_folder='../templates',
                static_folder='../static')
    app.config.from_object(config_class)

    db.init_app(app)

    with app.app_context():
        # Import models so they are registered with SQLAlchemy
        from app import models
        
        # Avoid creating tables in production postgres, assume migrations/alembic or pre-created tables.
        # But for this simple project, we can just create them if using SQLite
        if 'sqlite' in app.config.get('SQLALCHEMY_DATABASE_URI', ''):
            db.create_all()
            
        from app.routes import main
        app.register_blueprint(main)

    return app
