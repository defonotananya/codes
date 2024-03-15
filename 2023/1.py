# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Function to load dataset
def load_dataset(file_path):
    return pd.read_csv(file_path)

# Function for data preprocessing
def preprocess_data(df):
    # Handle missing values
    df = df.dropna()

    # Extract relevant features
    features = df[['Email Text']]

    return features, df['Email Type']

# Function for model training
def train_model(X_train, y_train):
    # Vectorize the content using CountVectorizer
    vectorizer = CountVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)

    # Initialize and train the Naive Bayes model
    nb_model = MultinomialNB()
    nb_model.fit(X_train_vectorized, y_train)

    return nb_model, vectorizer

# Function for model evaluation
def evaluate_model(model, X_test, y_test):
    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:\n", report)

    # Visualizations
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Email Type', data=df)
    plt.title('Distribution of Email Types')
    plt.show()

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=df['Email Type'].unique(), yticklabels=df['Email Type'].unique())
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Main execution
if __name__ == "__main__":
    # Load the dataset
    file_path = r"C:\Users\abtsn\OneDrive\Documents\GitHub\projects\cn\Phishing_Email.csv"
    df = load_dataset(file_path)

    # Preprocess data
    X, y = preprocess_data(df)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X['Email Text'], y, test_size=0.2, random_state=42)

    # Train the model
    trained_model, vectorizer = train_model(X_train, y_train)

    # Evaluate the model
    evaluate_model(trained_model, vectorizer.transform(X_test), y_test)
