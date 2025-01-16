from flask import Flask

def create_app():
    """
    Create and configure the Flask app.
    """
    app = Flask(_name_)

    # Import and register the API routes
    from .routes import api_routes
    app.register_blueprint(api_routes)

    return app