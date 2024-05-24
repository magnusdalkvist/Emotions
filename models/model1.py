# Imports-------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import logging
#-------------------------------------------

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Dataset preparations-----------------------
logger.info("Loading dataset...")
df = pd.read_csv('data/emotions.csv')
df.drop(columns='Unnamed: 0', inplace=True)

def clean_text(text):
    text = text.lower()  # Lowercase text
    text = re.sub(r'\b\w{1,2}\b', '', text)  # Remove short words
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

logger.info("Cleaning text data...")
df['cleaned_text'] = df['text'].apply(clean_text)

# Mapping labels to readable form for visualization
emotion_map = {
    0: 'sadness',
    1: 'joy',
    2: 'love',
    3: 'anger',
    4: 'fear',
    5: 'surprise'
}
df['label_name'] = df['label'].map(emotion_map)

# Plotting the distribution of categories
logger.info("Plotting data distribution...")
background_color = '#5fa1bc'
sns.set_theme(style="whitegrid", rc={"axes.facecolor": background_color, 'figure.facecolor': background_color})
count = df['label_name'].value_counts()
fig, axs = plt.subplots(1, 2, figsize=(12, 6), facecolor=background_color)
palette = sns.color_palette("bright", len(count))
sns.set_palette(palette)
axs[0].pie(count, labels=count.index, autopct='%1.1f%%', startangle=140)
axs[0].set_title('Distribution of Categories', fontsize=15, fontweight='bold')
sns.barplot(x=count.index, y=count.values, ax=axs[1])
axs[1].set_title('Count of Categories', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()

# Converting labels back to numeric for training
df['label'] = df['label_name'].map({v: k for k, v in emotion_map.items()})

logger.info("Splitting data into train and test sets...")
X = df['cleaned_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorizing the text data
logger.info("Vectorizing text data...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
logger.info("Text data vectorized successfully.")

# Training the model
logger.info("Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000, verbose=1)
model.fit(X_train_tfidf, y_train)
logger.info("Model trained successfully.")

# Predicting and evaluating the model
logger.info("Evaluating the model...")
y_pred = model.predict(X_test_tfidf)

logger.info(f"Accuracy: {accuracy_score(y_test, y_pred)}")
logger.info("Classification Report:\n" + classification_report(y_test, y_pred, target_names=list(emotion_map.values())))


def predict_emotion(message, model, vectorizer, emotion_map):

    cleaned_message = clean_text(message)
    

    message_tfidf = vectorizer.transform([cleaned_message])
    
 
    prediction = model.predict(message_tfidf)
    

    predicted_emotion = emotion_map[prediction[0]]
    
    return predicted_emotion


new_message = "I feel good today, because i learned something new!"

# Predict the emotion of the new message
predicted_emotion = predict_emotion(new_message, model, vectorizer, emotion_map)

print(f"The predicted emotion for the message is: {predicted_emotion}")