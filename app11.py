import io
import matplotlib
from sklearn.tree import DecisionTreeClassifier, plot_tree
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.metrics import accuracy_score, precision_score, f1_score
from mlxtend.frequent_patterns import association_rules
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report ,accuracy_score
import random
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from flask import Flask, render_template, jsonify, request, url_for
from flask import Flask, render_template, request
from mlxtend.frequent_patterns import apriori, association_rules

import numpy as np
import pickle
import seaborn as sns
import base64





import os

app = Flask(__name__)


file_path = 'C:\\Users\\Sajid Shah G\\Desktop\\train_data.csv'


weather_data = pd.read_csv(file_path)

scenarios = [
    {"id": 1, "title": "Cluster by Temperature and Rainfall", "description": "Group locations by temperature and rainfall patterns."},
    {"id": 2, "title": "Cluster by Wind Speed and Humidity", "description": "Analyze clusters based on wind speed and humidity levels."},
    {"id": 3, "title": "Cluster by Sunshine and Cloud Cover", "description": "Group locations based on sunshine duration and cloud cover."},
    {"id": 4, "title": "Cluster by Temperature and Wind Speed", "description": "Group locations based on temperature and wind speed at different times of the day."},
    {"id": 5, "title": "Cluster by Humidity and Pressure", "description": "Group locations based on humidity and atmospheric pressure at different times of the day."},
    {"id": 6, "title": "Cluster by Rainfall and Sunshine", "description": "Group locations based on rainfall and sunshine levels."}
]




@app.route('/show_data')
def show_data():
    
    data_html = weather_data.head(50).to_html(classes='table table-bordered', index=False)
    return data_html

if not os.path.exists('static/images'):
    os.makedirs('static/images')

@app.route('/')
def home():
    return render_template('new.html')


@app.route('/DisplayData')
def display_data():
    return render_template('weather-dashboard.html')



# file_path = 'C:\\Users\\Sajid Shah G\\Desktop\\train_data.csv'
# weather_data = pd.read_csv(file_path)

print(weather_data.dtypes)

numeric_columns = weather_data.select_dtypes(include=[np.number]).columns
weather_data[numeric_columns] = weather_data[numeric_columns].fillna(weather_data[numeric_columns].mean())

categorical_columns = weather_data.select_dtypes(include=[object]).columns
for column in categorical_columns:
    weather_data[column] = weather_data[column].fillna(weather_data[column].mode()[0])

X = weather_data[['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 
                  'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 
                  'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 
                  'Temp9am', 'Temp3pm']]
y = weather_data['RainTomorrow']  

encoder = LabelEncoder()
y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_image_path = 'static/confusion_matrix.png'
if not os.path.exists(conf_matrix_image_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.savefig(conf_matrix_image_path)

    plt.close()

@app.route('/7-day-weather', methods=['GET'])
def get_week_weather():
    sample_data = pd.DataFrame({
        'MinTemp': np.random.uniform(0.4, 0.7, 7),
        'MaxTemp': np.random.uniform(0.5, 0.8, 7),
        'Rainfall': np.random.uniform(0, 0.01, 7),
        'Evaporation': [0.03771]*7,
        'Sunshine': [0.52491]*7,
        'WindGustSpeed': np.random.uniform(0.2, 0.4, 7),
        'WindSpeed9am': np.random.uniform(0.1, 0.3, 7),
        'WindSpeed3pm': np.random.uniform(0.1, 0.3, 7),
        'Humidity9am': np.random.uniform(0.4, 0.8, 7),
        'Humidity3pm': np.random.uniform(0.2, 0.6, 7),
        'Pressure9am': np.random.uniform(0.4, 0.6, 7),
        'Pressure3pm': np.random.uniform(0.4, 0.6, 7),
        'Cloud9am': np.random.uniform(0.4, 0.6, 7),
        'Cloud3pm': np.random.uniform(0.4, 0.6, 7),
        'Temp9am': np.random.uniform(0.5, 0.6, 7),
        'Temp3pm': np.random.uniform(0.5, 0.7, 7),
    })

    predictions = model.predict(sample_data)

  
    predictions = encoder.inverse_transform(predictions)

    
    sample_data['RainTomorrow'] = predictions


    response = sample_data[['Temp3pm', 'RainTomorrow']].to_dict(orient='records')
    
    return jsonify(response)


@app.route('/Allinone')
def AllIN():
    return render_template('BeforeMid.html')

@app.route('/Naive')
def Naive():
    return render_template('naive.html')
@app.route('/Ann')
def Ann():
    return render_template('ann.html')
@app.route('/Knn')
def Knn():
    return render_template('knn.html')

file_path = 'C:\\Users\\Sajid Shah G\\Desktop\\train_data.csv'
weather_data = pd.read_csv(file_path)



def preprocess_data(df, technique=None):
    if df['RainTomorrow'].isnull().sum() > 0:
        df['RainTomorrow'].fillna('No', inplace=True)
    df['RainTomorrow'] = df['RainTomorrow'].map({'No': 0, 'Yes': 1}) 

    df['RainTomorrow'].fillna(0, inplace=True)

    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        df[column].fillna(df[column].mean(), inplace=True)

    categorical_columns = ['WindGustDir', 'WindDir9am', 'WindDir3pm'] 
    for col in categorical_columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna('Unknown', inplace=True)

    X = df.drop(columns=['Location', 'RainTomorrow'])
    y = df['RainTomorrow']

    if technique in ['ann', 'knn']: 
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    return X, y

@app.route('/classification-result/<technique>', methods=['GET'])
def classification_result(technique):
   
    random_seed = 42
    X, y = preprocess_data(weather_data, technique)

    if X is None or y is None:
        return "Error: Preprocessed data is empty. Please check the dataset.", 400

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_seed)

  
    if technique == 'naive':
        model = GaussianNB()
    elif technique == 'ann':
        model = MLPClassifier(max_iter=1000, random_state=random_seed)  
    elif technique == 'dtree':
        model = DecisionTreeClassifier(random_state=random_seed) 
    elif technique == 'knn':
        model = KNeighborsClassifier()
    else:
        return "Technique not found", 404

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)


    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    ax.set_title(f'Confusion Matrix - {technique.capitalize()}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')


    dir_path = os.path.join('static', 'confusematrix', technique)
    os.makedirs(dir_path, exist_ok=True)
    img_path = os.path.join(dir_path, f'confusion_matrix_{technique}.png')
    plt.savefig(img_path, format='png')
    plt.close()

    cm_image_url = url_for('static', filename=f'confusematrix/{technique}/confusion_matrix_{technique}.png')

    return render_template('BeforeMidResult.html', technique=technique.capitalize(), accuracy=accuracy, cm_image_url=cm_image_url)




@app.route('/AssociationNew', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        min_support = float(request.form['min_support'])
        min_confidence = float(request.form['min_confidence'])

        file_path = r'C:\Users\Sajid Shah G\Desktop\train_data.csv'

        try:
            weather_data = pd.read_csv(file_path)
            binary_data = pd.get_dummies(weather_data, drop_first=True)
            binary_data = binary_data.astype(bool)  

            binary_data = binary_data.head(50)

            frequent_itemsets = apriori(binary_data, min_support=min_support, use_colnames=True)
            frequent_itemsets = frequent_itemsets.head(50)

            print("Frequent Itemsets:")
            print(frequent_itemsets)

            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

            print("Association Rules:")
            print(rules)

            return render_template(
                'Ass.html',
                frequent_itemsets=frequent_itemsets.to_html(classes='table table-striped'),
                rules=rules.to_html(classes='table table-striped')
            )
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            return render_template('Ass.html', frequent_itemsets=None, rules=None, error=error_message)

    return render_template('Ass.html', frequent_itemsets=None, rules=None)
























file_path = 'C:\\Users\\Sajid Shah G\\Desktop\\train_data.csv'

weather_data = pd.read_csv(file_path)

features = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Height']
target = 'RainToday'  

weather_data['Height'] = 0  


X = weather_data[features].values 
y = weather_data[target].values  


clf = DecisionTreeClassifier()
clf.fit(X, y)

STATIC_FOLDER = os.path.join(os.getcwd(), 'static', 'trees')  
os.makedirs(STATIC_FOLDER, exist_ok=True)

@app.route('/generate-decision-tree', methods=['POST'])
def generate_decision_tree():
    content = request.get_json()
    height = content.get('height')

    if height is None:
        return jsonify({'error': 'Height is required'}), 400

    user_data = [float(height)] + [random.uniform(0, 1) for _ in range(len(features) - 1)]  

    if len(user_data) != len(features):
        return jsonify({'error': f'Expected {len(features)} features, but got {len(user_data)}'}), 400

    prediction = clf.predict([user_data])

    tree_filename = f'tree_{height}.png'
    tree_image_path = os.path.join(STATIC_FOLDER, tree_filename)

    fig_width = max(12, height * 2) 
    fig_height = max(8, height * 1.5) 

    plt.figure(figsize=(fig_width, fig_height))
    plot_tree(clf, filled=True, feature_names=features, class_names=['Class 1', 'Class 2'], rounded=True)
    plt.savefig(tree_image_path, format='png')
    plt.close()

    return jsonify({
        'prediction': prediction[0],
        'tree_image': f'/static/trees/{tree_filename}'  
    })

@app.route('/static/trees/<filename>')
def serve_image(filename):
    return send_from_directory(STATIC_FOLDER, filename)
















@app.route("/ClusterSceniro")
def scenarios_page():
    return render_template("ClusterSceniro.html", scenarios=scenarios)

@app.route("/scenario/<int:scenario_id>")
def run_scenario(scenario_id):
   
    scenario = next((s for s in scenarios if s["id"] == scenario_id), None)
    if not scenario:
        return redirect(url_for("scenarios_page"))

    result_summary, graph_path = run_clustering(scenario_id)

    return render_template(
        "ClusterResult.html",
        scenario_title=scenario["title"],
        scenario_description=scenario["description"],
        result_summary=result_summary,
        graph_path=graph_path
    )

def run_clustering(scenario_id):
    df = pd.read_csv(file_path)  

    if scenario_id == 1:
        features = ["MinTemp", "MaxTemp", "Rainfall"]
    elif scenario_id == 2:
        features = ["WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm"]
    elif scenario_id == 3:
        features = ["Sunshine", "Cloud9am", "Cloud3pm"]
    elif scenario_id == 4:
        features = ["MinTemp", "MaxTemp", "WindSpeed9am", "WindSpeed3pm"]
    elif scenario_id == 5:
        features = ["Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm"]
    elif scenario_id == 6:
        features = ["Rainfall", "Sunshine", "RainToday", "RainTomorrow"]
    else:
        return "Invalid scenario selected.", None

    if 'RainToday' in features or 'RainTomorrow' in features:
        le = LabelEncoder() 
        if 'RainToday' in features:
            df['RainToday'] = le.fit_transform(df['RainToday'])
        if 'RainTomorrow' in features:
            df['RainTomorrow'] = le.fit_transform(df['RainTomorrow'])

    data = df[features].dropna()

    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(data)

    data["Cluster"] = clusters

    graph_path = f"static/graphs/scenario_{scenario_id}.png"

    graph_dir = os.path.dirname(graph_path)
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)

    if not os.path.exists(graph_path):  
        plt.figure(figsize=(8, 6))

        cmap = plt.cm.get_cmap("viridis", 3) 

        scatter = plt.scatter(data[features[0]], data[features[1]], c=clusters, cmap=cmap)

        plt.xlabel(features[0])
        plt.ylabel(features[1])
        plt.title(f"Clustering Results for Scenario {scenario_id}")

        handles, labels = scatter.legend_elements()
        plt.legend(handles, [f"Cluster {i}" for i in range(3)], title="Clusters")

        plt.savefig(graph_path)
        plt.close()

    feature_names = ", ".join(features)

    result_summary = f"Clustering completed for {len(data)} data points using the following features: {feature_names}."
    return result_summary, graph_path


@app.route('/classification', methods=['GET', 'POST'])
def classification():
    tree_image_path = 'static/images/decision_tree.png'
    accuracy = None
    precision = None
    f1 = None

    if request.method == 'POST':
        tree_height = request.form.get('tree_height', type=int)
        max_depth = max(2, min(tree_height, 10))  

        data = pd.DataFrame({
            'feature1': np.random.uniform(0, 1, 100),
            'feature2': np.random.uniform(0, 1, 100),
            'target': np.random.choice([0, 1], 100)
        })

        X = data[['feature1', 'feature2']]
        y = data['target']

        train_size = int(0.8 * len(data))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        clf = DecisionTreeClassifier(max_depth=max_depth)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred) * 100 
        precision = precision_score(y_test, y_pred, zero_division=1)
        f1 = f1_score(y_test, y_pred, zero_division=1)

        plt.figure(figsize=(12, 8))
        plot_tree(
            clf,
            filled=True,
            feature_names=['feature1', 'feature2'],
            class_names=['0', '1']
        )
        plt.savefig(tree_image_path) 
        plt.close()

    return render_template(
        'classification.html',
        accuracy=accuracy,
        precision=precision,
        f1=f1,
        tree_image_path=tree_image_path
    )


@app.route('/svm', methods=['GET', 'POST'])
def svm_route():
    global weather_data

    data = weather_data.copy()
    label_encoders = {}

    categorical_columns = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']
    for col in categorical_columns:
        label_encoders[col] = LabelEncoder()
        data[col] = label_encoders[col].fit_transform(data[col].astype(str))

    features = data[['Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
                     'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm',
                     'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
                     'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'RainToday']]
    target = data['RainTomorrow']

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

    model = SVC(kernel='linear', random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return render_template('svm.html', accuracy=accuracy)

if __name__ == "__main__":
    os.makedirs("static/graphs", exist_ok=True)
    app.run(debug=True)

