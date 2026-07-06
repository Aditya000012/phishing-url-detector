from flask import Flask, render_template, request
import pickle
import os
from feature_extractor import extract_features

app = Flask(__name__)

# Load the trained RandomForest model at startup
MODEL_PATH = 'model.pkl'
model = None

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print("Warning: model.pkl not found. Predictions will not be available until trained.")

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        url = request.form.get('url', '').strip()
        if not url:
            prediction = {
                'error': 'Please provide a valid URL.'
            }
        elif model is None:
            prediction = {
                'error': 'The machine learning model could not be loaded. Ensure train_model.py has run successfully.'
            }
        else:
            try:
                # Extract features
                features = extract_features(url)
                
                # Convert to DataFrame to match training feature names and prevent warnings
                feature_cols = [
                    'url_length', 'has_ip', 'count_dots', 'count_hyphens_in_domain', 'count_hyphens_in_path',
                    'count_at_symbol', 'count_subdomains', 'has_https', 'has_double_slash_redirect', 'is_shortened',
                    'has_suspicious_words_in_domain', 'has_suspicious_words_in_path', 'has_port_number', 'path_depth',
                    'digit_ratio_in_domain'
                ]
                import pandas as pd
                features_df = pd.DataFrame([features], columns=feature_cols)
                
                # Predict class (1 = Phishing, 0 = Legitimate)
                pred_class = model.predict(features_df)[0]
                probs = model.predict_proba(features_df)[0]
                
                confidence = probs[pred_class] * 100
                label = "Phishing" if pred_class == 1 else "Legitimate"
                
                # Human-readable features list matching extraction order
                feature_names = [
                    'URL Length (total characters)',
                    'IP Address instead of Domain',
                    'Dot (.) Count',
                    'Hyphen Count in Domain',
                    'Hyphen Count in Path',
                    'Presence of "@" symbol',
                    'Subdomain Count (excluding TLD & domain)',
                    'Uses HTTPS protocol (https://)',
                    'Double Slash Redirection (// in path)',
                    'Known URL Shortening service',
                    'Contains suspicious words in Domain',
                    'Contains suspicious words in Path',
                    'Non-Standard Port Number specified',
                    'Path Depth (number of subfolders)',
                    'Digit Ratio in Domain Name'
                ]
                
                # Format feature values for display
                formatted_features = []
                for name, val in zip(feature_names, features):
                    # For binary features, display Yes/No
                    if name in [
                        'IP Address instead of Domain',
                        'Presence of "@" symbol',
                        'Uses HTTPS protocol (https://)',
                        'Double Slash Redirection (// in path)',
                        'Known URL Shortening service',
                        'Contains suspicious words in Domain',
                        'Contains suspicious words in Path',
                        'Non-Standard Port Number specified'
                    ]:
                        display_val = "Yes" if val == 1 else "No"
                    elif name == 'Digit Ratio in Domain Name':
                        display_val = f"{val:.4f}"
                    else:
                        display_val = str(val)
                    formatted_features.append((name, display_val))
                
                prediction = {
                    'url': url,
                    'label': label,
                    'confidence': f"{confidence:.1f}%",
                    'features': formatted_features,
                    'is_phishing': (pred_class == 1)
                }
            except Exception as e:
                prediction = {
                    'error': f"Failed to predict URL: {str(e)}"
                }
                
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    # Bind to 0.0.0.0 and port 5000 for local testing and container environments
    app.run(host='0.0.0.0', port=5000, debug=True)
