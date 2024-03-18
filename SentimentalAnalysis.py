import pandas as pd

dataset_path = "testdata.txt"
# Read the .txt file and create a DataFrame
with open(dataset_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()
    data = {'text': [line.strip() for line in lines]}

df = pd.DataFrame(data)



import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

#nltk.download('punkt')
#nltk.download('stopwords')

def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@\w+|\#','', text)  # Remove mentions and hashtags
    text = re.sub(r'\W', ' ', text)  # Remove non-alphanumeric characters
    text = text.lower()  # Convert to lower case
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)  # Remove single characters
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)  # Remove single characters at the start
    text = re.sub(r'\s+', ' ', text, flags=re.I)  # Replace multiple spaces with a single space
    text = re.sub(r'^b\s+', '', text)  # Remove prefixed 'b'
    text_tokens = word_tokenize(text)  # Tokenize
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words('english')]  # Remove stopwords
    text = ' '.join(tokens_without_sw)
    return text

# Apply the cleaning function to your data
df['cleaned_text'] = df['text'].apply(clean_text)



from textblob import TextBlob

def get_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'

df['sentiment'] = df['cleaned_text'].apply(get_sentiment)


import matplotlib.pyplot as plt
import seaborn as sns

# Visualize the sentiment distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='sentiment', data=df)
plt.title('Sentiment Distribution')
plt.show()


from wordcloud import WordCloud

def generate_word_cloud_for_sentiment(sentiment):
    text = ' '.join(df[df['sentiment'] == sentiment]['cleaned_text'])
    wordcloud = WordCloud(width=800, height=400, background_color ='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for {sentiment} Sentiment')
    plt.show()

generate_word_cloud_for_sentiment('Positive')
generate_word_cloud_for_sentiment('Negative')
generate_word_cloud_for_sentiment('Neutral')

print('')