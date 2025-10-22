# 🎬 MovieMind - Akıllı Film Öneri Sistemi

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Letterboxd verilerinizle kişiselleştirilmiş film önerileri alın! RAG (Retrieval-Augmented Generation) teknolojisi kullanarak, izlediğiniz filmlerden yola çıkarak size özel film önerileri sunar.

## ✨ Özellikler

- 🎯 **Kişiselleştirilmiş Öneriler**: Letterboxd verilerinizden yola çıkarak size özel film önerileri
- 🤖 **AI Destekli**: Google Gemini AI ile akıllı öneri sistemi
- 📊 **RAG Teknolojisi**: Retrieval-Augmented Generation ile gelişmiş arama
- 🎨 **Modern Arayüz**: Streamlit ile kullanıcı dostu web arayüzü
- 📁 **Kolay Veri Yükleme**: Letterboxd CSV dosyalarınızı kolayca yükleyin
- 🔍 **Akıllı Filtreleme**: Yıl ve tür bazlı filtreleme seçenekleri

## 🚀 Hızlı Başlangıç

### Gereksinimler

- Python 3.8+
- Letterboxd hesabı (veri export için)

### Kurulum

1. **Repository'yi klonlayın:**
```bash
git clone https://github.com/KULLANICI_ADINIZ/MovieMind.git
cd MovieMind
```

2. **Gerekli paketleri yükleyin:**
```bash
pip install -r requirements.txt
```

3. **Google Gemini API Key alın:**
   - [Google AI Studio](https://makersuite.google.com/app/apikey) adresine gidin
   - API key oluşturun
   - Proje klasöründe `.env` dosyası oluşturun:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

4. **Uygulamayı çalıştırın:**
```bash
streamlit run app.py
```

5. **Tarayıcıda açın:**
   - `http://localhost:8501` adresine gidin

## 📋 Kullanım

### 1. Letterboxd Verilerinizi Hazırlayın

1. **Letterboxd.com**'a giriş yapın
2. **Settings** → **Data** → **Export** tıklayın
3. **ratings.csv** dosyasını indirin (zorunlu)
4. İsterseniz **reviews.csv** dosyasını da indirin (opsiyonel)

### 2. Verilerinizi Yükleyin

- Uygulamada **"📤 CSV dosyalarını yükle"** seçeneğini seçin
- **ratings.csv** dosyasını yükleyin
- **reviews.csv** dosyasını yükleyin (varsa)
- **"🔧 İndeks Oluştur / Yenile"** butonuna tıklayın

### 3. Film Önerisi Alın

- **"Ne tür film arıyorsun?"** kısmına istediğiniz türü yazın (örn: "aksiyon filmleri", "komedi", "dram")
- **"En eski yıl"** filtresini ayarlayın
- **"🔍 Öneri Getir"** butonuna tıklayın

## 🛠️ Teknik Detaylar

### Kullanılan Teknolojiler

- **Streamlit**: Web arayüzü
- **ChromaDB**: Vektör veritabanı
- **Sentence Transformers**: Embedding modeli
- **Google Gemini AI**: Doğal dil işleme
- **Pandas**: Veri işleme

### Proje Yapısı

```
MovieMind/
├── app.py                 # Ana Streamlit uygulaması
├── simple_rag_system.py   # RAG sistemi
├── requirements.txt       # Python bağımlılıkları
├── README.md             # Proje dokümantasyonu
├── .gitignore            # Git ignore dosyası
└── letterboxd/           # Örnek veri klasörü
    ├── ratings.csv
    └── reviews.csv
```

### RAG Sistemi Nasıl Çalışır?

1. **Veri Yükleme**: Letterboxd CSV dosyalarından film verileri yüklenir
2. **Embedding**: Film açıklamaları vektörlere dönüştürülür
3. **Vektör Veritabanı**: ChromaDB'de saklanır
4. **Arama**: Kullanıcı sorgusu benzer filmlerle eşleştirilir
5. **AI Önerisi**: Gemini AI ile kişiselleştirilmiş öneriler oluşturulur


## 🤝 Destek

Sorularınız için:
- **Issue** açın: [GitHub Issues](https://github.com/KULLANICI_ADINIZ/MovieMind/issues)
- **Email**: your-email@example.com


**⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!**
