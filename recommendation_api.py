import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
from fastapi import FastAPI
import uvicorn
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

# --- THƯ VIỆN RAG ĐÃ ĐƯỢC SỬA ĐÚNG THEO LANGCHAIN MỚI ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # ← ĐÃ SỬA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Tải biến môi trường
load_dotenv()

warnings.filterwarnings('ignore')

# --- KHỞI TẠO APP FASTAPI ---
app = FastAPI()

# --- BIẾN TOÀN CỤC ---
df_ratings = None
df_movies = None
df_user_sim = None
df_content_sim = None
df_user = None
df_movie = None
norm_user_item = None

# Biến cho RAG
policy_vector_store = None


# === CÁC HÀM TIỀN XỬ LÝ ===
def transform_names_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return ''
    return ' '.join([p.strip() for p in text.split(',')])


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    words = []
    for word in text.split():
        word = re.sub('[^a-zA-Z0-9]', '', word)
        if word and word not in stop_words:
            words.append(lemmatizer.lemmatize(word))
    return ' '.join(words)


# === CÁC HÀM ĐỀ XUẤT ===
def get_user_similar_movies(user, similarity_threshold=0.1):
    global norm_user_item
    if df_user_sim is None or norm_user_item is None:
        return pd.DataFrame(columns=['movie_id', 'title', 'genre', 'user_similarity'])
    if user not in df_user_sim.index:
        return pd.DataFrame(columns=['movie_id', 'title', 'genre', 'user_similarity'])

    similar_users = df_user_sim.loc[df_user_sim[user] > similarity_threshold, user] \
        .drop(user).sort_values(ascending=False)

    if similar_users.empty:
        return pd.DataFrame(columns=['movie_id', 'title', 'genre', 'user_similarity'])

    target_user_movies = norm_user_item.loc[[user]].dropna(axis=1, how='all')
    similar_user_movies = norm_user_item.loc[similar_users.index].dropna(axis=1, how='all')

    # Loại bỏ phim người dùng đã xem
    cols_to_drop = [col for col in target_user_movies.columns if col in similar_user_movies.columns]
    similar_user_movies = similar_user_movies.drop(columns=cols_to_drop, errors='ignore')

    movie_score = {}
    for movie in similar_user_movies.columns:
        movie_ratings = similar_user_movies[movie]
        numerator = sum(similar_users[u] * movie_ratings[u] for u in similar_users.index if pd.notnull(movie_ratings[u]))
        denominator = sum(similar_users[u] for u in similar_users.index if pd.notnull(movie_ratings[u]))
        if denominator > 0:
            movie_score[movie] = numerator / denominator

    if not movie_score:
        return pd.DataFrame(columns=['movie_id', 'title', 'genre', 'user_similarity'])

    movie_score_df = pd.DataFrame(movie_score.items(), columns=['movie_id', 'user_similarity'])
    user_rec = pd.merge(df_movies[['movie_id', 'title', 'genre']], movie_score_df, on='movie_id', how='inner')
    return user_rec.sort_values(by='user_similarity', ascending=False).reset_index(drop=True)


def get_content_similar_movies(user_id):
    if df_user is None or df_content_sim is None or df_movie is None:
        return pd.DataFrame(columns=['title', 'genre', 'features', 'content_similarity'])

    df_current_user = df_user[df_user['userId'] == user_id]
    if df_current_user.empty:
        return pd.DataFrame(columns=['title', 'genre', 'features', 'content_similarity'])

    user_watched_movies = df_current_user['title'].tolist()
    user_mean_rating = df_current_user['rating'].mean()
    liked_movies = df_current_user[df_current_user['rating'] >= user_mean_rating]['title'].tolist()

    similar_movies = pd.DataFrame()
    for movie in liked_movies:
        if movie in df_content_sim.columns:
            temp = df_content_sim[movie].drop(labels=user_watched_movies, errors='ignore')
            similar_movies = pd.concat([similar_movies, temp.to_frame()], axis=1)

    if similar_movies.empty:
        return pd.DataFrame(columns=['title', 'genre', 'features', 'content_similarity'])

    similarity_scores = similar_movies.sum(axis=1)
    similarity_scores = similarity_scores.drop(labels=user_watched_movies, errors='ignore')
    top_recommendations = similarity_scores.sort_values(ascending=False).head(10).index

    result = df_movie[df_movie['title'].isin(top_recommendations)][['title', 'genre', 'features']].copy()
    result['content_similarity'] = result['title'].map(similarity_scores)
    return result.sort_values(by='content_similarity', ascending=False)


def hybrid_recommender(user_id, top_n=5):
    if df_movie is None or df_user is None:
        return pd.DataFrame(columns=['title', 'genre', 'features', 'similarity_score'])

    content_df = get_content_similar_movies(user_id)
    user_df = get_user_similar_movies(user_id, 0.1)

    # Merge 2 kết quả
    hybrid_df = pd.merge(
        content_df[['title', 'content_similarity']],
        user_df[['title', 'user_similarity']],
        on='title',
        how='outer'
    )

    hybrid_df['similarity_score'] = hybrid_df[['content_similarity', 'user_similarity']].mean(axis=1, skipna=True)
    hybrid_df = pd.merge(hybrid_df, df_movie[['title', 'genre', 'features']], on='title', how='left')

    # Loại bỏ phim đã xem
    watched = df_user[df_user['userId'] == user_id]['title'].tolist()
    hybrid_df = hybrid_df[~hybrid_df['title'].isin(watched)]

    hybrid_df = hybrid_df[['title', 'genre', 'features', 'similarity_score']]
    return hybrid_df.sort_values(by='similarity_score', ascending=False).head(top_n).reset_index(drop=True)


# === HÀM TẢI PDF (RAG) ===
def load_policy_data():
    global policy_vector_store
    print("Đang xử lý các file PDF Chính sách (RAG)...")

    pdf_files = ["regulations.pdf", "payment_policy.pdf"]
    all_documents = []

    for pdf_file in pdf_files:
        file_path = f"./{pdf_file}"
        if os.path.exists(file_path):
            print(f"   - Đang đọc: {pdf_file}")
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            all_documents.extend(docs)
        else:
            print(f"   Không tìm thấy: {pdf_file}")

    if not all_documents:
        print("Không có tài liệu PDF nào. Tính năng hỏi đáp chính sách sẽ bị tắt.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(all_documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    policy_vector_store = FAISS.from_documents(chunks, embeddings)
    print(f"Đã xử lý xong {len(chunks)} đoạn văn bản từ PDF.")


# === TẢI DỮ LIỆU + CHUẨN BỊ MÔ HÌNH ===
def load_data_and_prep_models():
    global df_ratings, df_movies, df_user_sim, df_content_sim, df_user, df_movie, norm_user_item

    print("Bắt đầu tải dữ liệu Recommendation...")
    try:
        db_url = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_DATABASE')}"
        engine = create_engine(db_url)
    except Exception as e:
        print(f"LỖI KẾT NỐI DB: {e}")
        return

    try:
        query_ratings = """
            SELECT b.user_id as "userId", s.movie_id as "movieId", 5 as rating 
            FROM bookings b 
            JOIN showtimes s ON b.showtime_id = s.showtime_id 
            WHERE s.start_time < NOW();
        """
        df_ratings = pd.read_sql(query_ratings, engine)
        df_movies = pd.read_sql("SELECT * FROM movies", engine)

        if df_ratings.empty or df_movies.empty:
            print("Không có dữ liệu ratings hoặc movies.")
        else:
            # Collaborative Filtering
            user_item = df_ratings.pivot_table(values='rating', index='userId', columns='movieId').fillna(0)
            norm_user_item = user_item.subtract(user_item.mean(axis=1), axis=0)
            user_similarity = cosine_similarity(norm_user_item)
            df_user_sim = pd.DataFrame(user_similarity, index=user_item.index, columns=user_item.index)

            # Content-Based
            nltk.download(['stopwords', 'wordnet', 'vader_lexicon'], quiet=True)

            df_movie = df_movies.copy()
            df_user = df_ratings.copy()

            df_movie['cast_members'] = df_movie['cast_members'].apply(lambda x: ' '.join(x).lower() if isinstance(x, (list, np.ndarray)) else '')
            df_movie['features_text'] = df_movie['features'].apply(lambda x: ' '.join(x).lower() if isinstance(x, (list, np.ndarray)) else '')
            df_movie['director'] = df_movie['director'].apply(transform_names_text)
            df_movie['genres_text'] = df_movie['genre'].apply(lambda x: ' '.join(str(x).split(',')).lower() if pd.notna(x) else '')

            df_movie['body_weighted'] = (
                (df_movie['description'].fillna('') + ' ') * 3 +
                (df_movie['features_text'].fillna('') + ' ') * 1 +
                (df_movie['cast_members'].fillna('') + ' ') * 3 +
                (df_movie['director'].fillna('') + ' ') * 3 +
                (df_movie['genres_text'].fillna('') + ' ') * 4
            )
            df_movie['body_weighted'] = df_movie['body_weighted'].apply(clean_text)

            sia = SentimentIntensityAnalyzer()
            df_movie['sentiment_score'] = df_movie['body_weighted'].apply(lambda x: sia.polarity_scores(x)['compound'])

            tfidf = TfidfVectorizer(min_df=1, max_df=0.85, ngram_range=(1, 2))
            tfidf_matrix = tfidf.fit_transform(df_movie['body_weighted'])
            sentiment_matrix = sparse.csr_matrix(df_movie['sentiment_score'].values.reshape(-1, 1)) * 0.2
            final_matrix = sparse.hstack([tfidf_matrix, sentiment_matrix]).tocsr()

            similarity_matrix = cosine_similarity(final_matrix)
            df_content_sim = pd.DataFrame(similarity_matrix, index=df_movie['title'], columns=df_movie['title'])

            df_user = pd.merge(df_user, df_movies[['movie_id', 'title', 'genre']], left_on='movieId', right_on='movie_id', how='left')

            print("Mô hình Recommendation đã sẵn sàng!")
    except Exception as e:
        print(f"Lỗi xử lý dữ liệu: {e}")

    load_policy_data()


@app.on_event("startup")
async def startup_event():
    load_data_and_prep_models()


@app.get("/recommendations/{user_id}")
async def get_recommendations(user_id: int):
    print(f"Đề xuất cho user_id: {user_id}")
    try:
        result = hybrid_recommender(user_id, top_n=5)
        if result.empty:
            return {"message": "Chưa có đủ dữ liệu để đề xuất phim."}
        return result.to_dict(orient='records')
    except Exception as e:
        return {"error": str(e)}


@app.get("/policy-search")
async def search_policy(query: str):
    if policy_vector_store is None:
        return {"message": "Hệ thống chính sách chưa sẵn sàng (thiếu file PDF)."}
    try:
        docs = policy_vector_store.similarity_search(query, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
        return {"query": query, "context": context}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
