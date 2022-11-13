import os
import logging
import random

from flask import Flask, render_template_string, request, jsonify, render_template, flash, redirect
from model_SLID import Spoken_Lang_Detection

app = Flask(__name__, template_folder='template', static_folder='template/static', 
)  

# define model path
model_path = 'saved_models/model.h5'
model = Spoken_Lang_Detection(model_path)

# create instance
logging.basicConfig(level=logging.INFO)

app.config['FILE_UPLOADS'] = "Testing_Data"


@app.route("/")
def index():
    """Provide simple health check route."""
    return render_template('index.html')

@app.route("/", methods=['POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file.')
            return redirect(request.url)
        file = request.files['file']   
        if file.filename == '':
            flash('No file selected for uploading.')
            return redirect(request.url)
        if file:
            classes = ['Arabic','Chinese','Portuguese']
            uploaded_file = request.files['file'] 
            filepath = os.path.join(app.config['FILE_UPLOADS'], uploaded_file.filename)
            predicted_class = model.predict(filepath)
            flash(classes[predicted_class])

           

def main():
    """Run the Flask app."""
    port=int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port = port, debug=True) 

if __name__ == "__main__":
    main()

