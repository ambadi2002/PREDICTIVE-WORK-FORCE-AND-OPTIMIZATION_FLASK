from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('best_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve input values from the form
        features = [float(request.form.get(feature)) for feature in [
            'Age', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction', 
            'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 
            'NumCompaniesWorked', 'RelationshipSatisfaction', 'TotalWorkingYears', 
            'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 
            'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager',
            'TotalYearsInRoles', 'Satisfaction_Involvement']]
        
        # Categorical features
        business_travel = request.form.get('BusinessTravel')
        department = request.form.get('Department')
        education_field = request.form.get('EducationField')
        gender = request.form.get('Gender')
        job_role = request.form.get('JobRole')
        marital_status = request.form.get('MaritalStatus')
        overtime = request.form.get('OverTime')

        # Encode categorical features
        categorical_features = [business_travel, department, education_field, gender, job_role, marital_status, overtime]
        encoded_features = encode_categorical_features(categorical_features)
        
        final_features = np.hstack((features, encoded_features)).reshape(1, -1)

        # Check if feature size matches model expectation
        if final_features.shape[1] != model.n_features_in_:
            return f"Error: Expected {model.n_features_in_} features, but got {final_features.shape[1]} features."

        prediction = model.predict(final_features)[0]
        
        if prediction == 1:
            return render_template('result_yes.html', prediction_text='Attrition: Yes')
        else:
            return render_template('result_no.html', prediction_text='Attrition: No')
    except Exception as e:
        return f"An error occurred: {str(e)}"

def encode_categorical_features(categorical_features):
    # Assuming you have a pre-defined encoding function or strategy
    # Replace this with your actual encoding logic
    encoded = []
    mapping = {
        'BusinessTravel': {'Non-Travel': 0, 'Travel_Frequently': 1, 'Travel_Rarely': 2},
        'Department': {'Human Resources': 0, 'Research & Development': 1, 'Sales': 2},
        'EducationField': {'Human Resources': 0, 'Life Sciences': 1, 'Marketing': 2, 'Medical': 3, 'Other': 4, 'Technical Degree': 5},
        'Gender': {'Female': 0, 'Male': 1},
        'JobRole': {'Healthcare Representative': 0, 'Human Resources': 1, 'Laboratory Technician': 2, 'Manager': 3, 'Manufacturing Director': 4, 'Research Director': 5, 'Research Scientist': 6, 'Sales Executive': 7, 'Sales Representative': 8},
        'MaritalStatus': {'Divorced': 0, 'Married': 1, 'Single': 2},
        'OverTime': {'No': 0, 'Yes': 1}
    }
    
    for feature, category in zip(['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime'], categorical_features):
        encoded.append(mapping[feature][category])
    
    return encoded

if __name__ == '__main__':
    app.run(debug=True)
