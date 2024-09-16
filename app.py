import pickle
import pandas as pd
import tldextract
from collections import Counter
import numpy as np
from urllib.parse import urlparse
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('best_random_forest_model.pkl', 'rb'))
print(type(model))


def extract_lexical_features(url):
    features = {}
    features['url_length'] = len(url)
    features['num_dots'] = url.count('.')
    features['num_slashes'] = url.count('/')
    features['num_hyphens'] = url.count('-')
    features['num_underscores'] = url.count('_')
    features['num_question_marks'] = url.count('?')
    features['num_ampersands'] = url.count('&')
    features['num_equals'] = url.count('=')
    features['num_at_symbols'] = url.count('@')
    features['num_hashes'] = url.count('#')
    features['num_percent_encodings'] = url.count('%')
    features['num_tildes'] = url.count('~')
    features['num_pluses'] = url.count('+')
    features['num_dollars'] = url.count('$')
    features['num_exclamations'] = url.count('!')
    features['num_commas'] = url.count(',')
    features['num_apostrophes'] = url.count("'")
    features['num_parentheses'] = url.count('(') + url.count(')')
    suspicious_words = ['login', 'secure', 'update', 'bank', 'free', 'win']
    features['suspicious_word_count'] = sum([1 for word in suspicious_words if word in url.lower()])
    extracted = tldextract.extract(url)
    features['domain'] = extracted.domain
    features['subdomain'] = extracted.subdomain
    features['suffix'] = extracted.suffix
    features['uses_https'] = 1 if url.startswith('https') else 0
    
    def calculate_entropy(s):
        p, lns = Counter(s), float(len(s))
        return -sum(count/lns * np.log2(count/lns) for count in p.values())
    
    features['entropy'] = calculate_entropy(url)
    features['num_digits'] = sum(c.isdigit() for c in url)
    features['num_special_chars'] = sum(not c.isalnum() for c in url)
    features['domain_length'] = len(extracted.domain)
    features['tld_length'] = len(extracted.suffix)
    features['num_path_segments'] = len(urlparse(url).path.split('/'))
    features['num_query_params'] = len(urlparse(url).query.split('&')) if urlparse(url).query else 0
    features['query_length'] = len(urlparse(url).query)
    features['first_path_segment_length'] = len(urlparse(url).path.split('/')[1]) if len(urlparse(url).path.split('/')) > 1 else 0
    features['proportion_numeric'] = sum(c.isdigit() for c in url) / len(url)
    features['proportion_alphabetic'] = sum(c.isalpha() for c in url) / len(url)
    features['proportion_special'] = sum(not c.isalnum() for c in url) / len(url)
    
    return features

def preprocess_url(url):
    features = extract_lexical_features(url)
    feature_vector = [
        features['url_length'],
        features['num_dots'],
        features['num_slashes'],
        features['num_hyphens'],
        features['num_underscores'],
        features['num_question_marks'],
        features['num_ampersands'],
        features['num_equals'],
        features['num_at_symbols'],
        features['num_hashes'],
        features['num_percent_encodings'],
        features['num_tildes'],
        features['num_pluses'],
        features['num_dollars'],
        features['num_exclamations'],
        features['num_commas'],
        features['num_apostrophes'],
        features['num_parentheses'],
        features['suspicious_word_count'],
        features['domain_length'],
        features['tld_length'],
        features['num_path_segments'],
        features['num_query_params'],
        features['query_length'],
        features['first_path_segment_length'],
        features['proportion_numeric'],
        features['proportion_alphabetic'],
        features['proportion_special']
    ]
    return feature_vector

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        url = request.form['url']
        # Process the URL and make a prediction
        result = predict_url(url)
        return render_template('result.html', result=result)
    return render_template('index.html')

def predict_url(url):
    features = preprocess_url(url)
    # Predict using the model
    prediction = model.predict([features])
    return 'Benign' if prediction[0] == 0 else 'Malicious'

@app.route('/how-it-works')
def how_it_works():
    return render_template('how_it_works.html')

if __name__ == '__main__':
    app.run(debug=True)
