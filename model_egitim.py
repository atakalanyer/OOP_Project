"""
* Amaç:
    Bu projede amacımız Eyüpsultanda bulunan 5 mahalledeki (Alibeyköy, Çırçır, Karadolap, Yeşilpınar, Akşemsettin) ev fiyatlarını farklı karmaşıklıktaki
    makine öğrenmesi modelleriyle tahmin etmek, validation kullanarak hiperparametre seviyesinde optimize etmek ve en başarılı modelin karar mekanizmasını
    görselleştirerek açıklamak. Son aşama olarak ise bu modeli bir web sitesine bağlayarak kullanıcılara ev fiyatlarını gösterebilmek.

* Kapsam:
    Bu kapsamda ilk aşamada Data Mining, Data Preprocessing, Feature Engineering yapacağız.
    Karşılaştırmalı Modelleme ve Optimizasyon
    Model Değerlendirme ve Açıklanabilirlik
    Website oluşturma.

* Adımlar:
    1. Instant Data Scraper tool kullanarak sahibinden sitesinden aradığımız kriterdeki ilanların verilerini mining etme.
    2. Veri Yükleme ve İnceleme
    3. Veri Preprocessing ve Feature Engineering
    4. RandomizedSearchCV ve K-Fold Cross Validation İle Model Eğitimi
    5. Performans Metriklerinin (CV R², Test R²) Karşılaştırılması ve Tabloştırılması
    6. Görsel Analiz ile Model Açıklanabilirliği
    7. Web...

* Kurulum:
    pip install numpy pandas matplotlib seaborn scikit-learn
"""