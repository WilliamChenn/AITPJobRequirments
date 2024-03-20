from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import os

# Define a function to read resumes from text files in a directory
def read_resumes_from_files(directory):
    resumes = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), "r") as file:
                resumes.append(file.read())
    return resumes

# Specify the directory where the resumes are stored
resumes_directory = "/path/to/resumes"

# Read resumes from text files
resumes = read_resumes_from_files(resumes_directory)

# Labels indicating whether each resume is fit (1) or not fit (0) for the job
labels = [1, 0, 1, 1, 1]  # Assuming you have labels corresponding to each resume file

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(resumes, labels, test_size=0.2, random_state=42)

# Vectorize text data using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Initialize and train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_vectorized, y_train)

# Predictions on the test set
predictions = classifier.predict(X_test_vectorized)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Example prediction for a new resume
new_resume_text = "Experienced customer service representative with a background in retail."
new_resume_vectorized = vectorizer.transform([new_resume_text])
prediction_new_resume = classifier.predict(new_resume_vectorized)
print("\nPrediction for the new resume:", "Fit" if prediction_new_resume[0] == 1 else "Not fit")
