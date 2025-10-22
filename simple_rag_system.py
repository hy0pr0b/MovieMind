import os
import pandas as pd
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv

class MovieMindRAG:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = None
        self.gemini_model = None

    def setup_gemini(self):
        if not self.api_key:
            return False
        genai.configure(api_key=self.api_key)
        self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        return True
        
    def load_letterboxd_data(self, folder_path: str):
        ratings_df = pd.read_csv(f"{folder_path}/ratings.csv")
        ratings_df['Watched'] = True
        
        if 'Rating' not in ratings_df.columns:
            ratings_df['Rating'] = 0.0

        review_path = f"{folder_path}/reviews.csv"
        if os.path.exists(review_path):
            try:
                reviews_df = pd.read_csv(review_path)
                ratings_df = ratings_df.merge(
                    reviews_df[['Name', 'Year', 'Review']],
                    on=['Name', 'Year'], how='left'
                )
            except Exception:
                ratings_df['Review'] = ''
        else:
            ratings_df['Review'] = ''

        df = ratings_df
        
        return df
    
    def create_movie_documents(self, df: pd.DataFrame) -> List[Dict]:
        documents = []
        
        for _, row in df.iterrows():
            name_val = str(row.get('Name', '') or '').strip()
            year_raw = row.get('Year')
            year_val = -1
            try:
                if pd.notna(year_raw):
                    year_val = int(year_raw)
            except Exception:
                year_val = -1
            rating_raw = row.get('Rating')
            rating_val = -1.0
            try:
                if pd.notna(rating_raw):
                    rating_val = float(rating_raw)
            except Exception:
                rating_val = -1.0

            text = f"Film: {name_val} ({year_val if year_val != -1 else 'Yıl yok'})"
            if rating_val >= 0:
                text += f" | Puan: {rating_val}/5"
            if pd.notna(row.get('Review')) and str(row['Review']).strip():
                text += f" | Yorum: {row['Review']}"
            
            documents.append({
                'id': f"{name_val}_{year_val}",
                'text': text,
                'metadata': {
                    'title': name_val,
                    'year': int(year_val),
                    'rating': float(rating_val),
                    'watched': bool(row.get('Watched', False))
                }
            })
        
        return documents
    
    def setup_vector_database(self, documents: List[Dict]):
        try:
            self.client.delete_collection("movies")
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection("movies")
        if not documents:
            return 0
        
        texts = [doc['text'] for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]
        ids = [doc['id'] for doc in documents]
        embeddings = self.embedding_model.encode(texts).tolist()
        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )
        return len(documents)
    
    def search_movies(self, query: str, n_results: int = 10) -> List[Dict]:
        if not self.collection:
            return []
        
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        movies = []
        if results and results['ids']:
            for i in range(len(results['ids'][0])):
                metadata = results['metadatas'][0][i]
                movies.append({
                    'title': metadata['title'],
                    'year': metadata['year'],
                    'rating': metadata['rating'],
                    'watched': metadata['watched']
                })
        return movies

    def get_recommendations(self, query: str, filters: Dict):
        if not self.gemini_model:
            similar_movies = self.search_movies(query, n_results=10)
            recommendations = "Gemini API key mevcut değil. Benzer filmler:\n"
            for movie in similar_movies:
                recommendations += f"- {movie['title']} ({movie['year']})\n"
            return {'success': True, 'recommendations': recommendations, 'similar_movies_found': len(similar_movies)}

        # Önce tüm filmlerden benzer olanları bul
        all_similar_movies = self.search_movies(query, n_results=50)
        
        # İzlenmiş filmleri filtrele (bunlar referans olacak)
        watched_movies_for_ai = [
            movie for movie in all_similar_movies 
            if movie['watched'] == True and movie['rating'] >= filters.get('min_rating', 0.0) and movie['year'] >= filters.get('year_min', 1900)
        ]

        # Eğer izlenmiş benzer film yoksa, tüm izlenmiş filmlerden yola çık
        if not watched_movies_for_ai:
            # Tüm izlenmiş filmleri al
            all_watched_results = self.collection.query(
                query_texts=[query], 
                n_results=100, 
                where={"watched": True}
            )
            
            if all_watched_results and all_watched_results['metadatas']:
                watched_movies_for_ai = []
                for i, metadata in enumerate(all_watched_results['metadatas'][0]):
                    if metadata.get('rating', 0) >= filters.get('min_rating', 0.0) and metadata.get('year', 0) >= filters.get('year_min', 1900):
                        watched_movies_for_ai.append({
                            'title': metadata['title'],
                            'year': metadata['year'],
                            'rating': metadata['rating'],
                            'watched': metadata['watched']
                        })

        return self.generate_recommendations_from_watched(query, watched_movies_for_ai)

    def generate_recommendations_from_watched(self, query: str, watched_movies: List[Dict]):
        if not self.gemini_model:
            return {'success': False, 'error': "Gemini modeli başlatılamadı."}

        if not watched_movies:
            return {'success': False, 'error': "İzlediğiniz film bulunamadı. Lütfen önce veri yükleyin."}

        top_movies = sorted(watched_movies, key=lambda x: x.get('rating', 0.0), reverse=True)[:5]
        avg_user_rating = sum(movie.get('rating', 0.0) for movie in top_movies) / len(top_movies) if top_movies else 0.0

        watched_titles = [f"{movie['title']} ({movie['year']})" for movie in watched_movies]
        
        all_watched_titles_from_db = []
        if self.collection:
            results = self.collection.query(
                query_texts=["watched movies"], 
                n_results=1000, 
                where={"watched": True}
            )
            if results and results['metadatas']:
                for metadata in results['metadatas'][0]:
                    all_watched_titles_from_db.append(f"{metadata['title']} ({metadata['year']})")

        all_watched_text = list(set(watched_titles + all_watched_titles_from_db))

        prompt = f"""
        Kullanıcı "{query}" türünde film önerileri istiyor.
        Kullanıcının izlediği ve beğendiği filmlerden bazıları (ortalama puanı {avg_user_rating:.1f}/5):
        {chr(10).join([f"- {movie['title']} ({movie['year']}) - Puan: {movie['rating']}/5" for movie in top_movies])}

        Bu filmlerden yola çıkarak, kullanıcının sevebileceği, henüz izlemediği 3-5 adet film önerisi yap.
        Önerilerini yaparken, kullanıcının izlediği filmlerin tarzını, türünü ve genel beğenisini göz önünde bulundur.
        Önerilerini kısa ve öz açıklamalarla birlikte sun.

        🚨 KRITIK UYARI: Aşağıdaki filmleri KESINLIKLE önerme (kullanıcı zaten izlemiş):
        {chr(10).join(all_watched_text)}
        Bu listedeki HİÇBİR filmi önerme! Sadece bu listede olmayan filmler öner.

        Önerilerini şu formatta sun:
        **Beğendiğiniz filmlerden yola çıkarak:**
        [Kısa bir giriş cümlesi]

        **Önerilerim:**
        - [Film Adı] ([Yıl]): [Kısa açıklama]
        - [Film Adı] ([Yıl]): [Kısa açıklama]
        - ...
        """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            recommendations = response.text
            
            return {'success': True, 'recommendations': recommendations, 'similar_movies_found': len(watched_movies)}
        except Exception as e:
            return {'success': False, 'error': f"Gemini modelinden öneri alınamadı: {e}"}