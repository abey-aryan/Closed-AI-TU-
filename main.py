from app import create_app

# Create the Flask app instance
app = create_app()

if _name_== "_main_":
    app.run(host="0.0.0.0", port=5000, debug=True)
