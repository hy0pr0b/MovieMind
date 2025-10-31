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
        
        # Set cache directory for Hugging Face models
        cache_dir = os.getenv("TRANSFORMERS_CACHE", "/app/.cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Use a more compatible model for Hugging Face Spaces
        try:
            self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2', cache_folder=cache_dir)
        except Exception as e:
            # Fallback to an even simpler model
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder=cache_dir)
        
        # Set up ChromaDB with proper cache directory
        cache_dir = os.getenv("CHROMA_CACHE_DIR", "/app/.cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = None
        self.gemini_model = None
        
        # Augmentation ayarları
        self.enable_document_augmentation = True
        self.enable_query_augmentation = True

    def setup_gemini(self):
        if not self.api_key:
            return False
        genai.configure(api_key=self.api_key)
        self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
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
    
    def augment_document(self, name: str, year: int, rating: float, review: str) -> List[str]:
        """
        Bir film dokümanını farklı açılardan yazarak çoğaltır (Document Augmentation)
        """
        augmented_texts = []
        
        # Varyasyon 1: Standart format
        text1 = f"Film: {name} ({year if year != -1 else 'Yıl yok'})"
        if rating >= 0:
            text1 += f" | Puan: {rating}/5"
        if review:
            text1 += f" | Yorum: {review}"
        augmented_texts.append(text1)
        
        # Varyasyon 2: Alternatif format
        text2 = f"{name} adlı {year if year != -1 else 'bilinmeyen yıl'} yapımı film"
        if rating >= 0:
            text2 += f" | Kullanıcı puanı: {rating}/5"
        if review:
            text2 += f" | İnceleme: {review}"
        augmented_texts.append(text2)
        
        # Varyasyon 3: Farklı açıdan
        text3 = f"Film adı: {name}"
        if year != -1:
            text3 += f" | Yapım yılı: {year}"
        if rating >= 0:
            text3 += f" | Değerlendirme: {rating} yıldız"
        if review:
            text3 += f" | Kullanıcı yorumu: {review}"
        augmented_texts.append(text3)
        
        # Varyasyon 4: Daha detaylı format
        if rating >= 0 and review:
            text4 = f"{name} ({year if year != -1 else 'Bilinmeyen yıl'}) - {rating}/5 puanlı bir film. İnceleme: {review}"
            augmented_texts.append(text4)
        
        return augmented_texts
    
    def augment_query(self, query: str) -> List[str]:
        """
        Kullanıcı sorgusunu zenginleştirir (Query Augmentation)
        """
        augmented_queries = [query]  # Orijinal sorgu her zaman dahil
        
        # Türkçe sinema terimleri eş anlamlıları
        genre_synonyms = {
            'aksiyon': ['aksiyon', 'action', 'macera', 'adventure', 'gerilim', 'thriller'],
            'komedi': ['komedi', 'comedy', 'mizah', 'güldürü'],
            'drama': ['drama', 'dramatik', 'duygusal'],
            'korku': ['korku', 'horror', 'gerilim', 'thriller', 'korku filmi'],
            'bilim kurgu': ['bilim kurgu', 'sci-fi', 'science fiction', 'gelecek', 'uzay'],
            'romantik': ['romantik', 'romance', 'aşk', 'romantik komedi'],
            'gerilim': ['gerilim', 'thriller', 'suspense', 'heyecanlı'],
        }
        
        query_lower = query.lower()
        
        # Eğer sorgu bir tür içeriyorsa, alternatif terimler ekle
        for genre, synonyms in genre_synonyms.items():
            if genre in query_lower:
                for synonym in synonyms[:2]:  # İlk 2 eş anlamlıyı ekle
                    if synonym != genre:
                        new_query = query.replace(genre, synonym)
                        if new_query not in augmented_queries:
                            augmented_queries.append(new_query)
        
        # Alternatif formülasyonlar
        if 'film' not in query_lower and 'movie' not in query_lower:
            augmented_queries.append(f"{query} film")
            augmented_queries.append(f"{query} filmleri")
        
        # Farklı soru formatları ekle
        question_variations = [
            query,
            f"{query} öner",
            f"{query} tavsiye et",
            f"{query} türünde film",
        ]
        
        for var in question_variations:
            if var not in augmented_queries:
                augmented_queries.append(var)
        
        # Maksimum 5 query döndür (performans için)
        return augmented_queries[:5]
    
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

            review_text = ''
            if pd.notna(row.get('Review')) and str(row['Review']).strip():
                review_text = str(row['Review']).strip()

            # Augmentation kullanılıyorsa, birden fazla doküman versiyonu oluştur
            if self.enable_document_augmentation:
                augmented_texts = self.augment_document(name_val, year_val, rating_val, review_text)
                # Her augment edilmiş versiyon için ayrı doküman oluştur
                for idx, text in enumerate(augmented_texts):
                    documents.append({
                        'id': f"{name_val}_{year_val}_aug{idx}",
                        'text': text,
                        'metadata': {
                            'title': name_val,
                            'year': int(year_val),
                            'rating': float(rating_val),
                            'watched': bool(row.get('Watched', False)),
                            'augmented': True,
                            'aug_index': idx
                        }
                    })
            else:
                # Augmentation kapalıysa, sadece standart format
                text = f"Film: {name_val} ({year_val if year_val != -1 else 'Yıl yok'})"
                if rating_val >= 0:
                    text += f" | Puan: {rating_val}/5"
                if review_text:
                    text += f" | Yorum: {review_text}"
                
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
        
        # Query augmentation kullanılıyorsa, birden fazla sorgu oluştur
        if self.enable_query_augmentation:
            augmented_queries = self.augment_query(query)
            # Her augment edilmiş sorgu için embedding oluştur
            query_embeddings = self.embedding_model.encode(augmented_queries).tolist()
        else:
            query_embeddings = self.embedding_model.encode([query]).tolist()
        
        # Tüm augment edilmiş sorgular için arama yap
        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results * 2  # Daha fazla sonuç al, sonra unique'leştir
        )
        
        # Sonuçları birleştir ve unique'leştir (aynı film tekrar göstermesin)
        seen_movies = set()
        movies = []
        
        if results and results['ids']:
            # Tüm sorgu sonuçlarını birleştir
            all_movies = []
            for query_idx in range(len(results['ids'])):
                if results['ids'][query_idx]:
                    for i in range(len(results['ids'][query_idx])):
                        metadata = results['metadatas'][query_idx][i]
                        movie_key = f"{metadata['title']}_{metadata['year']}"
                        
                        if movie_key not in seen_movies:
                            seen_movies.add(movie_key)
                            all_movies.append({
                                'title': metadata['title'],
                                'year': metadata['year'],
                                'rating': metadata['rating'],
                                'watched': metadata['watched']
                            })
            
            # Rating'e göre sırala ve n_results kadar döndür
            movies = sorted(all_movies, key=lambda x: x.get('rating', 0.0), reverse=True)[:n_results]
        
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
