import json
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template_string
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

file_path = 'C:\\Users\\Sajid Shah G\\Desktop\\Weather Test Data.csv'

weather_data = pd.read_csv(file_path)

if weather_data is not None:
    print("Weather data loaded successfully.")
else:
    print("Failed to load weather data.")

app = Flask(__name__)

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
    weather_data_clean[target_column] = label_encoder.fit_transform(weather_data_clean[target_column])

    # Define features (X) and target (y)
    X = weather_data_clean[feature_columns]
    y = weather_data_clean[target_column]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply Decision Tree Classifier
    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)

    # Predict on test data
    y_pred = classifier.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred) * 100  # Convert to percentage

    # Create the Decision Tree plot
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_tree(classifier, filled=True, feature_names=feature_columns, class_names=label_encoder.classes_, ax=ax)
    
    # Save the figure to a BytesIO object
    img_io = io.BytesIO()
    FigureCanvas(fig).print_png(img_io)
    img_io.seek(0)
    
    # Encode the image to base64
    tree_plot = base64.b64encode(img_io.getvalue()).decode('utf8')

    return accuracy, tree_plot

# Main route for displaying the webpage
@app.route('/')
def index():
    html_code = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Weather Analysis</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            /* Add your existing styles here */
        </style>
    </head>
    <body>
        <div class="sidebar">
            <button>Home</button>
            <button>About</button>
            <button>Contact</button>
            <button>Help</button>
        </div>
        <div class="container">
            <h1>Weather Data Analysis</h1>
            <form id="weatherForm">
                <input type="text" name="city" id="city" placeholder="Enter City Name" required autocomplete="off">
                <button type="submit">Get Weather Summary</button>
            </form>
            <div id="suggestions"></div>
            <div id="result" style="margin-top: 20px;"></div>
            <div class="buttons-container">
                <button id="classificationBtn" class="chart-btn">Apply Classification</button>
            </div>
        </div>

        <div id="chart-container" style="display: none;">
            <canvas id="weatherChart"></canvas>
        </div>

        <div id="tree-plot-container" style="display: none;">
            <img id="tree-plot" src="" alt="Decision Tree Plot">
        </div>

        <script>
            document.getElementById('city').addEventListener('input', async (e) => {
                const query = e.target.value;
                const suggestionsDiv = document.getElementById('suggestions');
                if (query.length > 0) {
                    try {
                        const response = await fetch(`/suggestions?query=${query}`);
                        const suggestions = await response.json();
                        suggestionsDiv.innerHTML = '';
                        suggestions.forEach(suggestion => {
                            const suggestionItem = document.createElement('li');
                            suggestionItem.textContent = suggestion;
                            suggestionItem.onclick = () => {
                                document.getElementById('city').value = suggestion;
                                suggestionsDiv.innerHTML = '';
                            };
                            suggestionsDiv.appendChild(suggestionItem);
                        });
                    } catch (error) {
                        console.error('Error fetching suggestions:', error);
                    }
                } else {
                    suggestionsDiv.innerHTML = '';
                }
            });

            document.getElementById('weatherForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                const city = document.getElementById('city').value;
                const resultDiv = document.getElementById('result');
                resultDiv.innerHTML = 'Loading...';

                try {
                    const response = await fetch('/weather', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/x-www-form-urlencoded',
                        },
                        body: `city=${city}`
                    });
                    const data = await response.json();

                    if (data.error) {
                        resultDiv.innerHTML = `<p style="color: red;">${data.error}</p>`;
                    } else {
                        const table = document.createElement('table');
                        const thead = document.createElement('thead');
                        const headerRow = document.createElement('tr');
                        headerRow.innerHTML = `
                            <th>Metric</th>
                            <th>Min</th>
                            <th>Max</th>
                            <th>Mean</th>
                            <th>Std</th>
                        `;
                        thead.appendChild(headerRow);
                        table.appendChild(thead);

                        const tbody = document.createElement('tbody');
                        let hasValidData = false;

                        for (const [metric, values] of Object.entries(data)) {
                            if (values.min !== 'N/A' || values.max !== 'N/A' || values.mean !== 'N/A' || values.std !== 'N/A') {
                                hasValidData = true;
                                const row = document.createElement('tr');
                                row.innerHTML = `
                                    <td>${metric}</td>
                                    <td>${values.min || 'N/A'}</td>
                                    <td>${values.max || 'N/A'}</td>
                                    <td>${values.mean || 'N/A'}</td>
                                    <td>${values.std || 'N/A'}</td>
                                `;
                                tbody.appendChild(row);
                            }
                        }

                        if (hasValidData) {
                            table.appendChild(tbody);
                            resultDiv.innerHTML = '';
                            resultDiv.appendChild(table);
                            const chartButton = document.createElement('button');
                            chartButton.classList.add('chart-btn');
                            chartButton.textContent = 'Generate Chart';
                            resultDiv.appendChild(chartButton);

                            chartButton.addEventListener('click', () => {
                                generateChart(data);
                            });
                        } else {
                            resultDiv.innerHTML = '<p>No valid data to display.</p>';
                        }
                    }
                } catch (error) {
                    resultDiv.innerHTML = `<p style="color: red;">An error occurred: ${error.message}</p>`;
                }
            });

            // Handle Classification Button
            document.getElementById('classificationBtn').addEventListener('click', async () => {
                const resultDiv = document.getElementById('result');
                resultDiv.innerHTML = 'Applying classification...';

                try {
                    const response = await fetch('/classification', { method: 'GET' });
                    const data = await response.json();
                    resultDiv.innerHTML = `Classification Accuracy: ${data.accuracy}%`;

                    // Display the Decision Tree plot
                    const treePlotContainer = document.getElementById('tree-plot-container');
                    const treePlot = document.getElementById('tree-plot');
                    treePlot.src = 'data:image/png;base64,' + data.treePlot;
                    treePlotContainer.style.display = 'block';

                } catch (error) {
                    resultDiv.innerHTML = `<p style="color: red;">An error occurred: ${error.message}</p>`;
                }
            });

            function generateChart(data) {
                const chartContainer = document.getElementById('chart-container');
                chartContainer.style.display = 'block';
                const ctx = document.getElementById('weatherChart').getContext('2d');

                const labels = Object.keys(data);
                const minData = labels.map(label => data[label].min);
                const maxData = labels.map(label => data[label].max);
                const meanData = labels.map(label => data[label].mean);
                const stdData = labels.map(label => data[label].std);

                new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: labels,
                        datasets: [
                            {
                                label: 'Min',
                                data: minData,
                                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                                borderColor: 'rgba(75, 192, 192, 1)',
                                borderWidth: 1
                            },
                            {
                                label: 'Max',
                                data: maxData,
                                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                                borderColor: 'rgba(255, 99, 132, 1)',
                                borderWidth: 1
                            },
                            {
                                label: 'Mean',
                                data: meanData,
                                backgroundColor: 'rgba(153, 102, 255, 0.2)',
                                borderColor: 'rgba(153, 102, 255, 1)',
                                borderWidth: 1
                            },
                            {
                                label: 'Std',
                                data: stdData,
                                backgroundColor: 'rgba(255, 159, 64, 0.2)',
                                borderColor: 'rgba(255, 159, 64, 1)',
                                borderWidth: 1
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            legend: {
                                position: 'top',
                            },
                            tooltip: {
                                mode: 'index',
                                intersect: false,
                            },
                        },
                        scales: {
                            x: {
                                beginAtZero: true
                            }
                        }
                    }
                });
            }
        </script>
    </body>
    </html>
    """
    return render_template_string(html_code)

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
