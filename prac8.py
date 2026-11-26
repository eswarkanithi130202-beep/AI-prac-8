from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Training dataset
messages = [
    "Win a free prize now",
    "Congratulations you won a lottery",
    "Claim your free gift today",
    "Get money quickly click here",
    "You have won a free ticket",
    "Hello how are you",
    "Are we meeting today",
    "Please call me later",
    "I will come to college tomorrow",
    "Thank you for your help"
]

labels = [
    "Spam", "Spam", "Spam", "Spam", "Spam",
    "Not Spam", "Not Spam", "Not Spam",
    "Not Spam", "Not Spam"
]

# Create and train the model
model = make_pipeline(
    CountVectorizer(),
    MultinomialNB()
)

model.fit(messages, labels)

# Test the application
print("Spam Message Detection System")

while True:
    text = input("\nEnter a message (or type exit): ")

    if text.lower() == "exit":
        break

    prediction = model.predict([text])[0]

    print("Prediction:", prediction)

    if prediction == "Spam":
        print("Explanation: The model detected word patterns "
              "associated with spam.")
    else:
        print("Explanation: The model classified the message "
              "as Not Spam based on learned word patterns.")
