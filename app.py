import json
import pandas as pd
import matplotlib.pyplot as plt
import os
from flask import Flask, request, jsonify, render_template
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Correct file path
file_path = 'C:\\Users\\Sajid Shah G\\Desktop\\Weather Test Data.csv'

# Load the weather data
weather_data = pd.read_csv(file_path)

if weather_data is not None:
    print("Weather data loaded successfully.")
else:
    print("Failed to load weather data.")

# Function to get weather summary for a city
def get_weather_summary(city):
    city_data = weather_data[weather_data['Location'].str.lower() == city.lower()]
    if city_data.empty:
        return {"error": "City not found."}
    summary = city_data.describe().to_dict()
    clean_summary = {
        key: {k: (v if pd.notna(v) else None) for k, v in value.items()}
        for key, value in summary.items()
    }
    return clean_summary

# Function to apply classification techniques and return accuracy and Decision Tree plot
def apply_classification():
    # Preprocess the data
    weather_data_clean = weather_data.dropna()  # Dropping rows with NaN values
    feature_columns = ['MinTemp', 'MaxTemp', 'Rainfall']  # Example features
    target_column = 'Location'  # Assuming 'Location' is the target for classification

    # Encode the categorical target variable (Location)
    label_encoder = LabelEncoder()
    weather_data_clean.loc[:, target_column] = label_encoder.fit_transform(weather_data_clean[target_column])

    # Define features (X) and target (y)
    X = weather_data_clean[feature_columns]
    y = weather_data_clean[target_column]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Decision Tree Classifier
    classifier = DecisionTreeClassifier(random_state=42)

    # Encode the target labels
    y_train = label_encoder.fit_transform(y_train)  # Encode the target labels
    classifier.fit(X_train, y_train)  # Train the classifier

    # Predict on test data
    y_pred = classifier.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred) * 100  # Convert to percentage

    # Create the Decision Tree plot
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_tree(classifier, filled=True, feature_names=feature_columns, class_names=label_encoder.classes_, ax=ax)
    
    # Save the figure to the static folder as a PNG file
    img_path = 'static/decision_tree.png'
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    fig.savefig(img_path)

    return accuracy, '/static/decision_tree.png'

# Main route for displaying the webpage
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/weather', methods=['POST'])
def weather():
    city = request.form.get('city')
    if not city:
        return jsonify({"error": "Please enter a city name."}), 400
    result = get_weather_summary(city)
    return jsonify(result)

@app.route('/suggestions', methods=['GET'])
def get_suggestions():
    query = request.args.get('query', '').strip().lower()
    if not query:
        return jsonify([])  

    matching_cities = weather_data['Location'].str.lower().unique()
    suggestions = [city for city in matching_cities if query in city]
    return jsonify(suggestions[:10])

@app.route('/classification', methods=['GET'])
def classification():
    accuracy, tree_plot = apply_classification()
    return jsonify({"accuracy": accuracy, "treePlot": tree_plot})

if __name__ == '__main__':
    app.run(debug=True)
