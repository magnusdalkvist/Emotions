#Different feelings by category
# 0 - Sadness
# 1 - Joy
# 2 - Love
# 3 - Anger
# 4 - Fear
# 5 - Surprise

#Imports-------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#-------------------------------------------


#Dataset preparations-----------------------
df = pd.read_csv('data\emotions.csv')
df.drop(columns='Unnamed: 0', inplace=True)
#-------------------------------------------

#Making a DF with the real-life labeling for easy visualization
emotion_map = {
    0: 'sadness',
    1: 'joy',
    2: 'love',
    3: 'anger',
    4: 'fear',
    5: 'surprise'
}
df['label'] = df['label'].map(emotion_map)
#-------------------------------------------


#-------------------------------------------

#Checking the dataset with prints and plots-
print(df.head(10))
print(df.shape)
#416809, 2



background_color = '#5fa1bc'
sns.set_theme(style="whitegrid", rc={"axes.facecolor": background_color, 'figure.facecolor': background_color})
count = df['label'].value_counts()
fig, axs = plt.subplots(1, 2, figsize=(12, 6), facecolor=background_color)
palette = sns.color_palette("bright", len(count))
sns.set_palette(palette)
axs[0].pie(count, labels=count.index, autopct='%1.1f%%', startangle=140)
axs[0].set_title('Distribution of Categories', fontsize=15, fontweight='bold')
sns.barplot(x=count.index, y=count.values, ax=axs[1], palette=palette)
axs[1].set_title('Count of Categories', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()
#-------------------------------------------

print(df.head(10))