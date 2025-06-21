from flask import Flask, request, render_template, redirect, url_for, flash
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure random key in production

# Load model and class names
model = joblib.load('model.pkl')
class_names = ['setosa', 'versicolor', 'virginica']

# Upload folder setup
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    """Render the home page with the prediction form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle single prediction from form input"""
    try:
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])
    except ValueError:
        flash('Please enter valid numeric values for all features.')
        return redirect(url_for('home'))

    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)
    predicted_class = class_names[prediction[0]]

    return render_template('index.html', prediction=predicted_class)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """Handle batch predictions from uploaded CSV files"""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part in the request.')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected.')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            try:
                df = pd.read_csv(filepath)
                required_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
                if not all(col in df.columns for col in required_cols):
                    flash(f'CSV must contain columns: {", ".join(required_cols)}')
                    return redirect(request.url)
                
                features = df[required_cols].values
                preds = model.predict(features)
                df['prediction'] = [class_names[p] for p in preds]

            except Exception as e:
                flash(f'Error processing CSV file: {e}')
                return redirect(request.url)

            # Convert dataframe to HTML table string for proper rendering
            table_html = df.to_html(classes='data', header=True, index=False)
            return render_template('upload.html', tables=table_html)
        else:
            flash('Allowed file types are: csv')
            return redirect(request.url)

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
