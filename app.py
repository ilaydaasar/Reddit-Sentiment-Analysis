import praw
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os
from transformers import pipeline
import nltk
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()

# Stopwords (İngilizce + Türkçe)
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('turkish') + stopwords.words('english'))

# Ayarlar ve Sabitler
POST_LIMIT = 100
TEXT_TRUNCATION_LIMIT = 512
OUTPUT_DIR = "data"

# Reddit API bağlantısı
reddit = praw.Reddit(
    client_id=os.getenv("D6UZCi1VbUKGc5mlUoPkog"),
    client_secret=os.getenv("ERiOlEok4_VK-p7JWodmtXuIk99wrA"),
    user_agent="SentimentAnalyzer by u/Emergency-Carob-6747"
)

# ------------------- Reddit Fonksiyonu -------------------
def get_reddit_posts(subreddit_name):
    """Belirtilen subreddit'ten gönderi çeker."""
    posts = []
    try:
        subreddit = reddit.subreddit(subreddit_name)
        for post in subreddit.hot(limit=POST_LIMIT):
            posts.append({
                "platform": "Reddit",
                "text": post.title,
                "score": post.score,
                "comments": post.num_comments
            })
        return pd.DataFrame(posts)
    except Exception as e:
        print(f"Hata: {subreddit_name} için veri çekilemedi. {e}")
        return pd.DataFrame()

# ------------------- Görselleştirme Fonksiyonu -------------------
def visualize_results(df, subreddit_name):
    """Duygu dağılımı ve kelime bulutu grafiklerini oluşturur ve kaydeder."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Duygu dağılımı (Bar Chart)
    sentiment_counts = df['duygu'].value_counts()
    plt.figure(figsize=(8, 5))
    sentiment_counts.plot(kind='bar', color=['#4CAF50', '#F44336', '#FFC107'])
    plt.title(f'{subreddit_name} Duygu Dağılımı')
    plt.xlabel("Duygu")
    plt.ylabel("Gönderi Sayısı")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{subreddit_name}_duygu_dagilimi.png'))
    plt.show()

    # Kelime Bulutu
    all_words = ' '.join(df['text'])
    if all_words.strip():
        wordcloud = WordCloud(
            width=800,
            height=500,
            background_color="white",
            stopwords=stop_words
        ).generate(all_words)
        
        plt.figure(figsize=(10, 7))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis('off')
        plt.title(f"{subreddit_name} - Kelime Bulutu")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'{subreddit_name}_kelime_bulutu.png'))
        plt.show()
    else:
        print(f"Uyarı: {subreddit_name} için kelime bulutu oluşturulamadı (boş içerik).")


# ------------------- Ana İşlem -------------------
def main():
    pipe = pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis")
    subreddits_input = input("Analiz edilecek subredditleri virgülle ayırarak girin (örn: iPhone,Android): ")
    subreddit_list = [s.strip() for s in subreddits_input.split(",")]

    for subreddit_name in subreddit_list:
        print(f"\n🔹 Analiz ediliyor: {subreddit_name}")
        df = get_reddit_posts(subreddit_name)

        if df.empty:
            print(f"Hata: {subreddit_name} için veri bulunamadı veya çekilemedi.")
            continue

        # Duygu analizi
        df['duygu_sonucu'] = df['text'].apply(lambda x: pipe(x[:TEXT_TRUNCATION_LIMIT])[0])
        df['duygu'] = df['duygu_sonucu'].apply(lambda x: x['label'])
        df['oran'] = df['duygu_sonucu'].apply(lambda x: x['score'])

        # CSV kaydet
        dosya_adi = os.path.join(OUTPUT_DIR, f'reddit_{subreddit_name}_analiz_sonuclari.csv')
        df.to_csv(dosya_adi, index=False)
        print(f"Sonuçlar kaydedildi: {dosya_adi}")

        # Görselleştirme
        visualize_results(df, subreddit_name)

if __name__ == "__main__":
    main()