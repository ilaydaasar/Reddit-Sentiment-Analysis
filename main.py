import praw
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import os
from transformers import pipeline
import nltk

# Stopwords (İngilizce + Türkçe)
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('turkish') + stopwords.words('english'))

# ------------------- Reddit Fonksiyonu -------------------
def get_reddit_posts(subreddit_name="iPhone", limit=100):
    reddit = praw.Reddit(
        client_id="D6UZCi1VbUKGc5mlUoPkog",
        client_secret="ERiOlEok4_VK-p7JWodmtXuIk99wrA",
        user_agent="SentimentAnalyzer by u/Emergency-Carob-6747"
    )

    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    for post in subreddit.hot(limit=limit):
        posts.append({
            "platform": "Reddit",
            "text": post.title,
            "score": post.score,
            "comments": post.num_comments
        })
    return pd.DataFrame(posts)

# ------------------- Ana İşlem -------------------
subreddits_input = input("Analiz edilecek subredditleri virgülle ayırarak girin (örn: iPhone,Android): ")
subreddit_list = [s.strip() for s in subreddits_input.split(",")]

pipe = pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis")

for subreddit_name in subreddit_list:
    print(f"\n🔹 Analiz ediliyor: {subreddit_name}")
    df = get_reddit_posts(subreddit_name, 100)

    if df.empty:
        print(f"{subreddit_name} için veri çekilemedi.")
        continue

    # Duygu analizi
    df['duygu_sonucu'] = df['text'].apply(lambda x: pipe(x[:512])[0])
    df['duygu'] = df['duygu_sonucu'].apply(lambda x: x['label'])
    df['oran'] = df['duygu_sonucu'].apply(lambda x: x['score'])

    # CSV kaydet
    dosya_adi = f'reddit_{subreddit_name}_analiz_sonuclari.csv'
    df.to_csv(dosya_adi, index=False)
    print(f"Sonuçlar kaydedildi: {dosya_adi}")

    # Duygu dağılımı
    sentiment_counts = df['duygu'].value_counts()

    # Pasta grafik
    plt.figure(figsize=(6, 6))
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%',
            startangle=90, colors=['#4CAF50', '#F44336', '#FFC107'])
    plt.title(f'{subreddit_name} Duygu Dağılımı (Pie Chart)')
    plt.show()

    # Bar chart
    plt.figure(figsize=(8, 5))
    sentiment_counts.plot(kind='bar', color=['#4CAF50', '#F44336', '#FFC107'])
    plt.title(f'{subreddit_name} Duygu Dağılımı (Bar Chart)')
    plt.xlabel("Duygu")
    plt.ylabel("Gönderi Sayısı")
    plt.show()

    # Kelime Bulutu (stopwords temizlenmiş)
    all_words = ' '.join(df['text'])
    wordcloud = WordCloud(
        width=800,
        height=500,
        random_state=21,
        max_font_size=110,
        background_color="white",
        stopwords=stop_words
    ).generate(all_words)

    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.title(f"{subreddit_name} - Kelime Bulutu")
    plt.show()
