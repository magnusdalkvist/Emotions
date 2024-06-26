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
import tkinter as tk
from tkinter import messagebox
from sklearn.utils.class_weight import compute_class_weight
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


# logger.info("Plotting data distribution...")
# background_color = '#5fa1bc'
# sns.set_theme(style="whitegrid", rc={"axes.facecolor": background_color, 'figure.facecolor': background_color})
# count = df['label_name'].value_counts()
# fig, axs = plt.subplots(1, 2, figsize=(12, 6), facecolor=background_color)
# palette = sns.color_palette("bright", len(count))
# sns.set_palette(palette)
# axs[0].pie(count, labels=count.index, autopct='%1.1f%%', startangle=140)
# axs[0].set_title('Distribution of Categories', fontsize=15, fontweight='bold')
# sns.barplot(x=count.index, y=count.values, ax=axs[1])
# axs[1].set_title('Count of Categories', fontsize=15, fontweight='bold')
# plt.tight_layout()
# plt.show()

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

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)



logger.info("Training Logistic Regression model with class weights...")
model = LogisticRegression(max_iter=1000, verbose=1, class_weight=dict(zip(np.unique(y_train), class_weights)))
model.fit(X_train_tfidf, y_train)
logger.info("Model trained successfully.")




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




#GUI SETUP-----------------------------------------

flag = True

def submit_input(event=None):
    global flag
    prediction_text = prediction_entry.get()
    if prediction_text.lower() == "exit":
        flag = False
        root.quit()
        return
    
    predicted_emotion = predict_emotion(prediction_text, model, vectorizer, emotion_map)
    output_text = f"Predicted Emotion: {predicted_emotion}\n"

    
    output_label.config(text=output_text)
    
    # Clear the entry field
    prediction_entry.delete(0, tk.END)


def reset_gui():
    output_label.config(text="")
    prediction_entry.delete(0, tk.END)
    prediction_entry.focus()

total_predictions = len(y_test)
correct_predictions = (y_test == y_pred).sum()
accuracy_percentage = (correct_predictions / total_predictions) * 100

while flag:
    root = tk.Tk()
    root.title("Prediction GUI")

    # Set font size for labels and buttons
    label_font = ("Arial", 24)
    button_font = ("Arial", 20)

    tk.Label(root, text=f"Model prediction accuracy: {accuracy_percentage:.2f}%\n\nPlease enter the prediction message, or \"exit\" to exit the loop.",
             font=label_font).pack(pady=5)
    prediction_entry = tk.Entry(root, font=label_font)
    prediction_entry.pack(pady=5)

    submit_button = tk.Button(root, text="Submit", command=submit_input, font=button_font)
    submit_button.pack(pady=10)

    reset_button = tk.Button(root, text="Reset", command=reset_gui, font=button_font)
    reset_button.pack(pady=10)

    output_label = tk.Label(root, text="", fg="blue", font=label_font)
    output_label.pack(pady=10)

    # Bind the <Return> key to submit_input function
    root.bind("<Return>", submit_input)

    root.mainloop()

    # Calculate accuracy for each prediction
    if not flag:
        break  # Exit the loop if the user exits

    # Display accuracy for the last prediction
    if y_test is not None:
        last_prediction_text = prediction_text
        last_predicted_emotion = predicted_emotion
        last_actual_emotion = emotion_map[y_test[-1]]
        last_prediction_accuracy = "Correct" if last_predicted_emotion == last_actual_emotion else "Incorrect"
        logger.info(f"Input: {last_prediction_text}, Predicted Emotion: {last_predicted_emotion}, Actual Emotion: {last_actual_emotion}, Accuracy: {last_prediction_accuracy}")
