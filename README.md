# Eyüpsultan Konut Fiyat Tahmini | Machine Learning & Model Explainability

🇹🇷 Türkçe | 🇬🇧 [English](#english)

## 🇹🇷 Proje Hakkında

Bu proje, İstanbul'un Eyüpsultan ilçesindeki **Alibeyköy, Çırçır, Karadolap, Yeşilpınar ve Akşemsettin** mahallelerinde bulunan konutların fiyatlarını makine öğrenmesi algoritmalarıyla tahmin etmeyi amaçlamaktadır.

Sahibinden.com üzerindeki konut ilanlarından Instant Data Scraper kullanılarak elde edilen veriler; veri temizleme, ön işleme ve özellik mühendisliği aşamalarından geçirilmiştir. Ardından farklı regresyon modelleri RandomizedSearchCV ve K-Fold Cross Validation kullanılarak optimize edilmiş ve performansları karşılaştırılmıştır.

Projenin son aşamasında, modellerin tahmin performansları değerlendirilmiş ve özelliklerin tahminler üzerindeki etkilerini anlamak amacıyla görsel analizler gerçekleştirilmiştir.

## 🎯 Projenin Amaçları

- Gerçek konut ilanlarından elde edilen verilerle fiyat tahmini yapmak.
- Veri temizleme ve özellik mühendisliği tekniklerini uygulamak.
- Farklı makine öğrenmesi modellerinin performanslarını karşılaştırmak.
- Hiperparametre optimizasyonu ve çapraz doğrulama kullanmak.
- Model tahminlerini ve özellik önemini görselleştirerek yorumlamak.

## 📊 Veri Seti

Veri seti, Eyüpsultan'ın beş farklı mahallesindeki konut ilanlarından oluşturulmuştur.

Veri hazırlama sürecinde gerçekleştirilen başlıca işlemler:

- Eksik ve yinelenen kayıtların kontrol edilmesi.
- Fiyat sütununun sayısal formata dönüştürülmesi.
- Metrekare başına fiyat üzerinden tutarsız kayıtların incelenmesi.
- IQR yöntemiyle aykırı değerlerin belirlenmesi.
- Oda ve salon bilgilerinin ayrı özelliklere dönüştürülmesi.
- İlan başlıklarından anahtar kelime tabanlı özellikler çıkarılması.
- Mahalle bilgilerinin One-Hot Encoding ile dönüştürülmesi.

**Hedef değişken:** `fiyat_tl`

**Başlıca özellikler:** Brüt metrekare, bina yaşı, oda bilgileri, mahalle ve ilan başlıklarından türetilen özellikler.

> Veri toplama ve filtreleme sonrasında kalan kesin gözlem sayısı, veri temizleme işlemlerine bağlıdır.

## 🛠️ Kullanılan Teknolojiler

- **Python** — Programlama dili
- **Pandas & NumPy** — Veri işleme ve analizi
- **Matplotlib & Seaborn** — Veri görselleştirme
- **Scikit-learn** — Makine öğrenmesi, model değerlendirme ve optimizasyon
- **Instant Data Scraper** — Web üzerinden ilan verisi toplama

## 🤖 Kullanılan Modeller

1. Ridge Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Gradient Boosting Regressor

Hiperparametre optimizasyonunda `RandomizedSearchCV`, çapraz doğrulamada ise 5 katlı `KFold` kullanılmıştır.

## 📈 Model Performansları

Mevcut deney sonuçları aşağıda verilmiştir.

| Model | CV R² | Test R² | Test MAE (TL) |
|---|---:|---:|---:|
| Ridge Regression | 0.532 | 0.513 | 1,237,125 |
| Decision Tree | 0.521 | 0.570 | 1,125,905 |
| **Random Forest** | **0.606** | **0.648** | **1,053,250** |
| Gradient Boosting | 0.584 | 0.611 | 1,104,675 |

### Sonuçların Değerlendirilmesi

Mevcut deneylerde Random Forest, **0.648 Test R²** ve yaklaşık **1,053,250 TL Test MAE** ile en iyi test performansını göstermiştir.

- **R²:** Modelin hedef değişkendeki varyansı ne ölçüde açıkladığını gösterir.
- **MAE:** Tahminler ile gerçek konut fiyatları arasındaki ortalama mutlak farkı gösterir.

Bu sonuçlar, modellerin mevcut veri setindeki performansını yansıtmaktadır. Sonuçlar farklı veri setlerine veya dönemlere doğrudan genellenemez.

## 📉 Model Açıklanabilirliği ve Görsel Analiz

Model performansını ve tahmin davranışlarını incelemek için aşağıdaki analizler gerçekleştirilmiştir:

- Fiyat dağılımı ve mahallelere göre fiyat karşılaştırması
- Brüt metrekare ile fiyat arasındaki ilişkinin incelenmesi
- Gerçek fiyat ve tahmin edilen fiyat karşılaştırması
- Tahmin hatalarının dağılımı (Residual Analysis)
- Permutation Importance ile özellik öneminin ölçülmesi
- Partial Dependence Display ile özelliklerin tahminlere etkisinin incelenmesi
- Model davranışını daha anlaşılır kılmak için sığ bir karar ağacının görselleştirilmesi

## 🚀 Kurulum ve Kullanım

### 1. Depoyu klonlayın

```bash
git clone https://github.com/atakalanyer/Eyupsultan-House-Price-Prediction.git
cd Eyupsultan-House-Price-Prediction
```

> Depo adını GitHub'daki gerçek depo adınızla eşleştirin.

### 2. Gerekli kütüphaneleri yükleyin

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 3. Veri setini hazırlayın

CSV dosyanızı Python dosyasıyla aynı dizine yerleştirin. Kodda kullanılan dosya adlarının mevcut dosyalarla eşleştiğinden emin olun.

### 4. Projeyi çalıştırın

Notebook veya Python dosyanızı çalıştırarak veri ön işleme, model eğitimi, performans karşılaştırması ve görsel analiz aşamalarını gerçekleştirin.

## ⚠️ Sınırlılıklar

- Veri seti belirli mahallelerden ve belirli bir dönemde toplanan ilanlarla sınırlıdır.
- İlan fiyatları, gerçekleşmiş satış fiyatlarıyla aynı olmayabilir.
- Konut fiyatlarını etkileyen tüm değişkenler veri setinde bulunmayabilir.
- Model performansı veri kalitesine, örneklem büyüklüğüne ve seçilen özelliklere bağlıdır.

## 🔮 Gelecekteki Geliştirmeler

- Daha fazla konut ilanı ve yeni özelliklerle veri setini genişletmek.
- Farklı aykırı değer işleme yöntemlerini karşılaştırmak.
- Daha kapsamlı hiperparametre optimizasyonu yapmak.
- SHAP ile yerel ve genel model açıklanabilirliği eklemek.
- Modeli Flask veya Streamlit ile basit bir tahmin uygulamasına dönüştürmek.

## 👨‍💻 Geliştirici

**Ata Kalanyer**

GitHub: [@atakalanyer](https://github.com/atakalanyer)

---

<a id="english"></a>

# Eyüpsultan House Price Prediction | Machine Learning & Model Explainability

## 🇬🇧 About the Project

This project aims to predict residential property prices in five neighborhoods of Eyüpsultan, Istanbul, Türkiye: **Alibeyköy, Çırçır, Karadolap, Yeşilpınar, and Akşemsettin**.

Property listing data collected from Sahibinden.com using Instant Data Scraper was processed through data cleaning, preprocessing, and feature engineering. Four regression models were then optimized and compared using RandomizedSearchCV and K-Fold Cross Validation.

The project also includes visual analyses to evaluate prediction performance and investigate the contribution of individual features.

## 🎯 Project Objectives

- Predict property prices using real-world listing data.
- Apply data cleaning, preprocessing, and feature engineering techniques.
- Compare the performance of different machine learning regression models.
- Optimize hyperparameters and evaluate models using cross-validation.
- Visualize model predictions, errors, and feature importance.

## 📊 Dataset

The dataset consists of residential property listings from five neighborhoods in Eyüpsultan.

Key data preparation steps include:

- Checking missing values and duplicate records.
- Converting property prices into a numerical format.
- Identifying inconsistent records using price-per-square-meter information.
- Detecting outliers using the IQR method.
- Extracting room and living-room counts from room descriptions.
- Creating keyword-based features from listing titles.
- Encoding neighborhood categories using One-Hot Encoding.

**Target variable:** `fiyat_tl` (property price in Turkish lira)

**Main features:** Gross area, building age, room information, neighborhood, and features extracted from listing titles.

> The final number of observations depends on the data cleaning and filtering steps.

## 🛠️ Technologies Used

- **Python** — Programming language
- **Pandas & NumPy** — Data manipulation and analysis
- **Matplotlib & Seaborn** — Data visualization
- **Scikit-learn** — Machine learning, evaluation, and optimization
- **Instant Data Scraper** — Web-based listing data collection

## 🤖 Machine Learning Models

1. Ridge Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Gradient Boosting Regressor

Hyperparameter optimization was performed using `RandomizedSearchCV`, with 5-fold `KFold` cross-validation.

## 📈 Model Performance

The following results were obtained in the current experiments.

| Model | CV R² | Test R² | Test MAE (TRY) |
|---|---:|---:|---:|
| Ridge Regression | 0.532 | 0.513 | 1,237,125 |
| Decision Tree | 0.521 | 0.570 | 1,125,905 |
| **Random Forest** | **0.606** | **0.648** | **1,053,250** |
| Gradient Boosting | 0.584 | 0.611 | 1,104,675 |

### Results and Interpretation

Among the evaluated models, Random Forest achieved the best test performance, with a **Test R² of 0.648** and a **Test MAE of approximately TRY 1,053,250**.

- **R²:** Measures the proportion of variance in the target variable explained by the model.
- **MAE:** Measures the average absolute difference between predicted and actual property prices.

These results reflect performance on the current dataset and should not be assumed to generalize directly to other datasets or time periods.

## 📉 Model Explainability and Visualization

The following analyses were performed to investigate model performance and prediction behavior:

- Property price distributions and neighborhood-level comparisons
- Gross area versus property price analysis
- Actual versus predicted price plots
- Residual error distribution analysis
- Feature importance analysis using Permutation Importance
- Partial dependence analysis to examine feature effects on predictions
- Visualization of a shallow decision tree to illustrate simplified price-splitting behavior

## 🚀 Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/atakalanyer/Eyupsultan-House-Price-Prediction.git
cd Eyupsultan-House-Price-Prediction
```

> Replace the repository name with the actual GitHub repository name if it differs.

### 2. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 3. Prepare the dataset

Place the CSV dataset in the same directory as the Python script and ensure that the file paths match those used in the code.

### 4. Run the project

Run the Python script or notebook to execute the data preprocessing, model training, performance comparison, and visualization stages.

## ⚠️ Limitations

- The dataset is limited to listings from five neighborhoods and a specific collection period.
- Listing prices may differ from actual transaction prices.
- Some factors affecting property prices may not be available in the dataset.
- Model performance depends on data quality, sample size, and feature selection.

## 🔮 Future Improvements

- Expand the dataset with additional listings and relevant features.
- Compare alternative outlier handling methods.
- Perform more extensive hyperparameter optimization.
- Add SHAP for global and local model explainability.
- Develop a simple prediction interface using Flask or Streamlit.

## 👨‍💻 Author

**Ata Kalanyer**

GitHub: [@atakalanyer](https://github.com/atakalanyer)
