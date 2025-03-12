import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Sample dataset
emails = [
    {'text': 'Congratulations, you have won a lottery!', 'label': 'spam'},
    {'text': 'Hi, can we meet tomorrow?', 'label': 'not spam'},
    {'text': 'Win a free vacation to Bahamas!', 'label': 'spam'},
    {'text': 'Please find the attached report.', 'label': 'not spam'},
    {'text': 'Get cheap loans now!', 'label': 'spam'},
    {'text': 'Let me know your availability for the meeting.', 'label': 'not spam'}
]

# Convert to DataFrame
email_df = pd.DataFrame(emails)

# Split the dataset
data_train, data_test, label_train, label_test = train_test_split(
    email_df['text'], email_df['label'], test_size=0.3, random_state=42
)

# Vectorize the text data
vectorizer = CountVectorizer()
train_vectors = vectorizer.fit_transform(data_train)
test_vectors = vectorizer.transform(data_test)

# Train the Naive Bayes model
model = MultinomialNB()
model.fit(train_vectors, label_train)

# Predict and evaluate
predictions = model.predict(test_vectors)
accuracy = accuracy_score(label_test, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Predict new emails
def predict_spam(text):
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)
    return prediction[0]


