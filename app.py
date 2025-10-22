import streamlit as st
import os
import pandas as pd

from simple_rag_system import MovieMindRAG

st.set_page_config(page_title="🎬 MovieMind - Akıllı Film Öneri", page_icon="🎬", layout="centered")

def main():
    st.title("🎬 MovieMind - Akıllı Film Öneri")
    st.caption("Letterboxd verilerinizle kişiselleştirilmiş film önerileri")

    if 'rag' not in st.session_state:
        st.session_state.rag = MovieMindRAG()

    gemini_available = st.session_state.rag.setup_gemini()
    if not gemini_available:
        st.info("💡 Gemini API key yoksa sadece benzer film listesi gösterilir (sistem çalışır)")

    st.subheader("1) Veriyi Yükle")
    
    upload_option = st.radio("Veri kaynağı seçin:", ["📁 Mevcut letterboxd/ klasörü", "📤 CSV dosyalarını yükle"], key="upload_option")
    
    if upload_option == "📤 CSV dosyalarını yükle":
        st.markdown("**Letterboxd'den export ettiğiniz CSV dosyalarını yükleyin:**")
        
        with st.expander("📋 Letterboxd'den nasıl export edilir?"):
            st.markdown("""
            1. **Letterboxd.com**'a giriş yapın
            2. **Settings** → **Data** → **Export** tıklayın
            3. **ratings.csv** dosyasını indirin (zorunlu)
            4. İsterseniz **reviews.csv** dosyasını da indirin (opsiyonel)
            5. Bu dosyaları aşağıdan yükleyin
            """)
        
        ratings_file = st.file_uploader("ratings.csv dosyasını yükleyin", type="csv", key="ratings")
        reviews_file = st.file_uploader("reviews.csv dosyasını yükleyin (opsiyonel)", type="csv", key="reviews")
        
        if st.button("🔧 İndeks Oluştur / Yenile", type="primary", key="create_index_csv"):
            if ratings_file is None:
                st.error("En az ratings.csv dosyasını yüklemelisiniz!")
            else:
                try:
                    temp_folder = "temp_letterboxd"
                    os.makedirs(temp_folder, exist_ok=True)
                    
                    with open(f"{temp_folder}/ratings.csv", "wb") as f:
                        f.write(ratings_file.getbuffer())
                    
                    if reviews_file is not None:
                        with open(f"{temp_folder}/reviews.csv", "wb") as f:
                            f.write(reviews_file.getbuffer())
                    
                    df = st.session_state.rag.load_letterboxd_data(temp_folder)
                    docs = st.session_state.rag.create_movie_documents(df)
                    count = st.session_state.rag.setup_vector_database(docs)
                    st.success(f"✅ {count} film indekse eklendi")
                    st.session_state.df = df
                    
                    import shutil
                    shutil.rmtree(temp_folder, ignore_errors=True)
                    
                except Exception as e:
                    st.error(f"Hata: {e}")
    
    else:  # Mevcut klasör
        letterboxd_folder = "letterboxd"
        if st.button("🔧 İndeks Oluştur / Yenile", type="primary", key="create_index_folder"):
            if not os.path.exists(letterboxd_folder):
                st.error("`letterboxd/` klasörü bulunamadı. ratings.csv (ve varsa reviews.csv) bu klasörde olmalı.")
            else:
                try:
                    df = st.session_state.rag.load_letterboxd_data(letterboxd_folder)
                    docs = st.session_state.rag.create_movie_documents(df)
                    count = st.session_state.rag.setup_vector_database(docs)
                    st.success(f"✅ {count} film indekse eklendi")
                    st.session_state.df = df
                except Exception as e:
                    st.error(f"Hata: {e}")

    if 'df' in st.session_state and st.session_state.df is not None:
        st.subheader("📊 Profil Analizi")
        df = st.session_state.df
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🎬 Toplam Film", len(df))
        with col2:
            avg_rating = df['Rating'].mean()
            st.metric("⭐ Ortalama Puan", f"{avg_rating:.1f}/5")
        with col3:
            high_rated = len(df[df['Rating'] >= 4.0])
            st.metric("🔥 Yüksek Puanlı", high_rated)
        with col4:
            recent_films = len(df[df['Year'] >= 2020])
            st.metric("📅 Son 4 Yıl", recent_films)
        
        if len(df) > 0:
            st.markdown("### 🌟 En Sevdiğin Filmler")
            top_movies = df[df['Rating'] >= 4.5].sort_values('Rating', ascending=False).head(5)
            if len(top_movies) > 0:
                for _, movie in top_movies.iterrows():
                    st.write(f"• **{movie['Name']}** ({movie['Year']}) - {movie['Rating']}/5")
            else:
                st.write("Henüz 4.5+ puan verdiğin film yok")
        

    st.markdown("---")
    st.subheader("🔍 Film Önerisi")
    st.markdown("İzlediğin filmlerden yola çıkarak öneri üretir.")

    query = st.text_input("Ne tür film arıyorsun?", placeholder="İstediğiniz film türünü yazın (örnek: aksiyon, komedi, dram, korku...)", key="main_query")
    
    year_min = st.number_input("En eski yıl", 1900, 2024, 2000, key="year_min_input")

    if st.button("🔍 Öneri Getir", type="primary", key="get_recommendations"):
        if not query.strip():
            st.warning("⚠️ Lütfen prompt kısmına film türü yazın!")
        elif not st.session_state.rag.collection:
            st.error("Önce veriyi yükleyin!")
        else:
            with st.spinner("Aranıyor..."):
                filters = {
                    "min_rating": 0.0,
                    "year_min": year_min,
                    "only_unwatched": True
                }
                result = st.session_state.rag.get_recommendations(query, filters)
            
            if result.get('success'):
                st.markdown("## 🎬 Film Önerileri")
                
                recommendations_text = result['recommendations']
                
                if "**Önerilerim:**" in recommendations_text:
                    parts = recommendations_text.split("**Önerilerim:**")
                    if len(parts) > 1:
                        st.markdown(parts[0])
                        
                        st.markdown("### 🎯 Size Özel Öneriler")
                        recommendations_part = parts[1].strip()
                        
                        lines = recommendations_part.split('\n')
                        for line in lines:
                            if line.strip() and not line.strip().startswith('*'):
                                st.markdown(f"• {line.strip()}")
                        
                    else:
                        st.markdown(recommendations_text)
                else:
                    st.markdown(recommendations_text)
                
            else:
                st.warning(f"❌ {result.get('error', 'Sonuç bulunamadı')}")

if __name__ == "__main__":
    main()