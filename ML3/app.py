from flask import Flask, render_template, request, jsonify
import pandas as pd
import traceback
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

# Load the data
try:
    df = pd.read_csv('unemployment_data_large.csv')
except Exception as e:
    print("Error loading CSV file:", str(e))
    traceback.print_exc()

# Preprocessing
label_encoder_gender = LabelEncoder()
label_encoder_qualification = LabelEncoder()
label_encoder_location = LabelEncoder()
label_encoder_field_of_study = LabelEncoder()

df['gender_encoded'] = label_encoder_gender.fit_transform(df['gender'])
df['qualification_encoded'] = label_encoder_qualification.fit_transform(df['qualification'])
df['location_encoded'] = label_encoder_location.fit_transform(df['location'])
df['field_of_study_encoded'] = label_encoder_field_of_study.fit_transform(df['field_of_study'])

# Ensure 'employability_status' is properly encoded as integers
df['employability_status_encoded'] = df['employability_status'].apply(lambda x: 1 if x == 'Employable' else 0)

# Features and Target
X = df[['age', 'gender_encoded', 'qualification_encoded', 'years_of_experience', 'location_encoded', 'field_of_study_encoded']].astype(float)
y = df['employability_status_encoded']  # Use the encoded target variable

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define parameter grid for RandomForest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create GridSearchCV object
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid,
                           cv=5,
                           scoring='accuracy',
                           verbose=2,
                           n_jobs=-1)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Get the best model
model = grid_search.best_estimator_

# Define a function to make recommendations
def recommend_skills(qualification, years_of_experience, field_of_study):
    if qualification == 'No Qualification':
        if field_of_study == 'No Field of Study':
            return "Consider enrolling in basic education programs or vocational training courses. Potential fields of study could include Information Technology, Healthcare, or Trade Skills."
        else:
            return f"Since you have experience or interest in {field_of_study}, consider pursuing formal education or certification in this area to improve your employability."
    elif qualification == 'Matric':
        return "Consider acquiring vocational training or a tertiary education degree."
    elif qualification == 'Diploma':
        return "Consider pursuing a specialized certification in your field."
    elif qualification == 'Degree':
        if years_of_experience == 0:
            return "Since you have a degree but no work experience, consider internships, volunteer work, or entry-level positions to gain practical experience."
        elif years_of_experience < 2:
            return "Consider gaining more practical experience or pursuing a relevant certification."
        else:
            return "Continue enhancing your skills with industry-relevant courses or consider advanced studies."
    elif qualification == 'Postgraduate':
        return "Leverage your advanced qualifications by seeking specialized roles or consulting opportunities in your field."
    else:
        return "Continue enhancing your skills with industry-relevant courses or consider advanced studies."

# Define a function to create visualizations
def create_visualizations():
    # Employability Distribution (Bar Plot)
    plt.figure(figsize=(8, 6))
    sns.countplot(x='employability_status', data=df, palette='viridis')
    plt.title('Employability Distribution')
    plt.xlabel('Employability')
    plt.ylabel('Count')
    plt.savefig('static/employability_distribution.png')
    
    # Recommendations Distribution (Pie Chart)
    recommendations = df['qualification'].apply(lambda q: recommend_skills(q, 0, 'Unknown'))
    recommendation_counts = recommendations.value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(recommendation_counts, labels=recommendation_counts.index, autopct='%1.1f%%', colors=sns.color_palette('viridis'))
    plt.title('Recommendations for Unemployable Youth')
    plt.savefig('static/recommendations_distribution.png')
        

# Generate visualizations once when the app starts
create_visualizations()

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        age = int(data['age'])
        gender = data['gender']
        qualification = data['qualification']
        years_of_experience = data['years_of_experience']
        location = data['location']
        field_of_study = data['field_of_study']

        # Convert "No Experience" to 0
        if years_of_experience == "No Experience":
            years_of_experience = 0
        else:
            years_of_experience = int(years_of_experience)

        if qualification == 'No Qualification' or field_of_study == 'No Field of Study':
            recommendation = recommend_skills(qualification, years_of_experience, field_of_study)
            employability = 'Unemployable'
        else:
            # Transform input data using the same encoders
            gender_encoded = label_encoder_gender.transform([gender])[0]
            qualification_encoded = label_encoder_qualification.transform([qualification])[0]
            location_encoded = label_encoder_location.transform([location])[0]
            field_of_study_encoded = label_encoder_field_of_study.transform([field_of_study])[0]

            input_data = [[age, gender_encoded, qualification_encoded, years_of_experience, location_encoded, field_of_study_encoded]]
            prediction = model.predict(input_data)[0]

            employability = 'Employable' if prediction == 1 else 'Unemployable'
            recommendation = recommend_skills(qualification, years_of_experience, field_of_study) if employability == 'Unemployable' else "No recommendation needed."

        return jsonify({'employability': employability, 'recommendation': recommendation})
    
    except Exception as e:
        error_message = f"Error processing the prediction: {str(e)}"
        print(error_message)
        traceback.print_exc()
        return jsonify({'error': error_message}), 500


if __name__ == '__main__':
    app.run(debug=True)
