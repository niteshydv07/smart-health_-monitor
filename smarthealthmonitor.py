import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Generate synthetic data
np.random.seed(42)

# Synthetic dataset creation
n_samples = 1000

data = {
    'age': np.random.randint(18, 80, n_samples),
    'sex': np.random.choice([0, 1], n_samples),  # 0: Female, 1: Male
    'heart_rate': np.random.randint(60, 100, n_samples),
    'calories_burned': np.random.randint(1500, 3500, n_samples),
    'steps': np.random.randint(1000, 20000, n_samples),
    'bmi': np.random.uniform(18.5, 35, n_samples),
    'risk': np.random.choice([0, 1], n_samples)  # 0: Low risk, 1: High risk
}

df = pd.DataFrame(data)

# Exploratory Data Analysis
plt.figure(figsize=(10, 6))
sns.histplot(df['age'], kde=True, bins=30)
plt.title('Age Distribution')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df['heart_rate'], kde=True, bins=30)
plt.title('Heart Rate Distribution')
plt.show()

# Data Preprocessing
X = df.drop('risk', axis=1)
y = df['risk']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Data Visualization
def plot_metric(metric):
    plt.figure(figsize=(10, 6))
    sns.histplot(df[metric], kde=True, bins=30)
    plt.title(f'{metric.capitalize()} Distribution')
    plt.show()

metrics = ['heart_rate', 'calories_burned', 'steps', 'bmi']
for metric in metrics:
    plot_metric(metric)

# Input method
def get_user_input():
    print("Please enter the following details:")
    age = int(input("Age: "))
    sex = int(input("Sex (0: Female, 1: Male): "))
    heart_rate = int(input("Heart Rate: "))
    calories_burned = int(input("Calories Burned: "))
    steps = int(input("Steps: "))
    bmi = float(input("BMI: "))
    
    user_input = {
        'age': age,
        'sex': sex,
        'heart_rate': heart_rate,
        'calories_burned': calories_burned,
        'steps': steps,
        'bmi': bmi
    }
    return user_input

# Get user input
user_input = get_user_input()

user_input_df = pd.DataFrame([user_input])
user_input_scaled = scaler.transform(user_input_df)

risk_prediction = model.predict(user_input_scaled)
risk_probability = model.predict_proba(user_input_scaled)

print(f'Predicted Risk: {"High" if risk_prediction[0] == 1 else "Low"}')
print(f'Risk Probability: {risk_probability[0][1]:.2f}')

# Visualizing the user input against the overall data
def plot_user_vs_data(metric, user_value):
    plt.figure(figsize=(10, 6))
    sns.histplot(df[metric], kde=True, bins=30)
    plt.axvline(user_value, color='red', linestyle='dashed', linewidth=2)
    plt.title(f'{metric.capitalize()} Distribution with User Input')
    plt.show()

for metric in user_input.keys():
    plot_user_vs_data(metric, user_input[metric])
