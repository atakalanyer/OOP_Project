"""
* Amaç:
    Bu projede amacımız Eyüpsultanda bulunan 5 mahalledeki (Alibeyköy, Çırçır, Karadolap, Yeşilpınar, Akşemsettin) ev fiyatlarını farklı karmaşıklıktaki
    makine öğrenmesi modelleriyle tahmin etmek, validation kullanarak hiperparametre seviyesinde optimize etmek ve en başarılı modelin karar mekanizmasını
    görselleştirerek açıklamak.

* Kapsam:
    Bu kapsamda ilk aşamada Data Mining, Data Preprocessing, Feature Engineering yapacağız.
    Karşılaştırmalı Modelleme ve Optimizasyon
    Model Değerlendirme ve Açıklanabilirlik

* Adımlar:
    1. Instant Data Scraper tool kullanarak sahibinden sitesinden aradığımız kriterdeki ilanların verilerini mining etme.
    2. Veri Yükleme ve İnceleme
    3. Veri Preprocessing ve Feature Engineering
    4. RandomizedSearchCV ve K-Fold Cross Validation İle Model Eğitimi
    5. Performans Metriklerinin (CV R², Test R²) Karşılaştırılması ve Tabloştırılması
    6. Görsel Analiz ile Model Açıklanabilirliği

* Kurulum:
    pip install numpy pandas matplotlib seaborn scikit-learn
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.tree import plot_tree

df = pd.read_csv("ev_verileri_eyup5.csv")

print(df.shape)

#Verimizdeki bina yaşı sütunundaki sayılar artan sıralı şekilde olduğu için modelimiz yanılmasın diye karıştırıyoruz. 
#frac 1 ibaresi verinin tamamını almamız için.
df_karistir = df.sample(frac=1, random_state=42).reset_index(drop=True)

df_karistir.to_csv("ev_verileri_eyup5.csv", index=False,encoding="utf-8-sig")

df_kopya = df_karistir.copy()
print(df_karistir.head())

print(df_kopya.info())

print(df_kopya.duplicated().sum())

def metin_normalize_et(series):

  return(
      series.astype(str)
      .str.strip()
      .str.lower()
      .str.replace("i̇", "i", regex=False)
      .str.replace("ı", "i", regex=False)
      .str.replace("ö", "o", regex=False)
      .str.replace("ü", "u", regex=False)
      .str.replace("ç", "c", regex=False)
      .str.replace("ş", "s", regex=False)
      .str.replace("ğ", "g", regex=False)
  )

kategorik_sutunlar = df.select_dtypes(include =["object","category","bool","string"]).columns.tolist()

for i in kategorik_sutunlar:
  if i in df_kopya.columns:
    df_kopya[i] = metin_normalize_et(df_kopya[i])

print("Normalizasyon sonrası tekrarlayan satır sayısı:", df_kopya.duplicated().sum())

#Tekrarlayan satırlardan emin olmak için ekranda gösteriyoruz.
tekrarlayan_satırlar = df_kopya[df_kopya.duplicated(keep=False)].sort_values(by=kategorik_sutunlar)
print(tekrarlayan_satırlar)

print(f"Temizlik öncesi satır sayısı: {len(df_kopya)}")

#Tekrarlayan satırları silme işlemi
df_kopya = df_kopya.drop_duplicates().reset_index(drop=True)

print(f"Temizlik sonrası satır sayısı: {len(df_kopya)}")

#NaN değer kontrolü
print(df_kopya.isnull().sum())

#STR olan fiyat sütununu normalize edip, int türüne dönüştürdük.
df_kopya["fiyat_tl"] = df_kopya["fiyat_tl"].str.replace(".", "", regex=False)
df_kopya["fiyat_tl"] = df_kopya["fiyat_tl"].str.replace("TL", "", regex=False)
df_kopya["fiyat_tl"] = df_kopya["fiyat_tl"].str.replace("tl", "", regex=False)
df_kopya["fiyat_tl"] = df_kopya["fiyat_tl"].astype(int)

# m² başına fiyat türetelim
df_kopya["fiyat_m2"] = df_kopya["fiyat_tl"] / df_kopya["m2_brut"]

#Mantıksız/ Hatalı verileri filtreleme
df_kopya = df_kopya[
    (df_kopya["fiyat_tl"] > 100000) &
    (df_kopya["m2_brut"] > 20) &
    (df_kopya["m2_brut"] < 1000) &
    (df_kopya["fiyat_m2"] >= 10000) & 
    (df_kopya["fiyat_m2"] <= 300000)
].reset_index(drop=True)

print(df_kopya.shape)

#Filtreleme ile verileri temizleyemedik. Aykırı değerleri bulalım.
Q1_fiyat = df_kopya["fiyat_tl"].quantile(0.25)
Q3_fiyat = df_kopya["fiyat_tl"].quantile(0.75)
IQR_fiyat = Q3_fiyat - Q1_fiyat

alt_sinir_fiyat = Q1_fiyat - 1.5 * IQR_fiyat
ust_sinir_fiyat = Q3_fiyat + 1.5 * IQR_fiyat

Q1_m2 = df_kopya["m2_brut"].quantile(0.25)
Q3_m2 = df_kopya["m2_brut"].quantile(0.75)
IQR_m2 = Q3_m2 - Q1_m2

alt_sinir_m2 = Q1_m2 - 1.5 * IQR_m2
ust_sinir_m2 = Q3_m2 + 1.5 * IQR_m2

df_kopya = df_kopya[
    (df_kopya["fiyat_tl"] >= alt_sinir_fiyat) & 
    (df_kopya["fiyat_tl"] <= ust_sinir_fiyat) &
    (df_kopya["m2_brut"] >= alt_sinir_m2) & 
    (df_kopya["m2_brut"] <= ust_sinir_m2)
].reset_index(drop=True)

print(df_kopya.shape)
#Temizlediğimiz aykırı değerleri temizleyip yeni bir csv dosyasına kaydedelim.
df_kopya.to_csv("ev_veriler_eyup5_aykiri_temizlendi.csv", index=False,encoding="utf-8-sig")

df = pd.read_csv("ev_veriler_eyup5_aykiri_temizlendi.csv")

print(df.shape)
print(df.head())
print(df.describe())

# Fiyatın dağılımı ve fiyatla en ilişkili sayısal değişkenler
plt.figure(figsize=(7, 4))
sns.histplot(df["fiyat_tl"], bins=30, kde=True)
plt.title("Fiyat Dağılımı")
plt.xlabel("Fiyat (TL)")
plt.show()

plt.figure(figsize=(7, 4))
sns.boxplot(data=df, x="mahalle", y="fiyat_tl")
plt.title("Mahallelere Göre Fiyat")
plt.xticks(rotation=20)
plt.show()

plt.figure(figsize=(7, 4))
sns.scatterplot(data=df, x="m2_brut", y="fiyat_tl", hue="mahalle")
plt.title("m² - Fiyat İlişkisi")
plt.show()

df_model = df.copy()

# Mahalle isimlerindeki "mh." / "mah." eklerini temizleyelim
df_model["mahalle"] = (df_model["mahalle"]
                       .str.replace(" mah.", "", regex=False)
                       .str.replace(" mh.", "", regex=False)
                       .str.strip())

# "2+1" gibi oda bilgisini oda ve salon olarak ayırıp sayıya çevirelim
oda_salon = df_model["oda_sayisi"].str.extract(r"(\d+(?:\.\d+)?)\+(\d+)").astype(float)
df_model["oda"] = oda_salon[0]
df_model["salon"] = oda_salon[1]
df_model["toplam_oda"] = df_model["oda"] + df_model["salon"]

# İlan başlığından işe yarayabilecek ipuçlarını (0/1 sütunlar) çıkaralım
anahtar_kelimeler = ["site", "sifir", "metro", "dubleks", "otopark", "asansor", "manzara", "havuz"]
for kelime in anahtar_kelimeler:
    df_model["baslik_" + kelime] = df_model["Baslik"].str.contains(kelime).astype(int)

# ÖNEMLİ: fiyat_m2 sütunu fiyat_tl'den türetildi. Hedefi bilerek tahmin yapmaya
# çalıştığımız için bunu özellik olarak kullanırsak model "kopya çekmiş" olur.
# Bu yüzden Baslik (ham metin), oda_sayisi (yukarıda sayıya çevirdik) ve fiyat_m2'yi atıyoruz.
df_model = df_model.drop(columns=["Baslik", "oda_sayisi", "fiyat_m2"])

# Kategorik olan mahalle için one-hot encoding
df_model = pd.get_dummies(df_model, columns=["mahalle"], drop_first=False, dtype=int)

print(df_model.shape)
print(df_model.isnull().sum().sum(), "eksik değer var")
print(df_model.head())

# Özellikler (X) ve hedef (y)
X = df_model.drop(columns=["fiyat_tl"]).astype(float)   # float: grafik fonksiyonları tam sayı sütunlarını sevmiyor
y = df_model["fiyat_tl"]

# Eğitim / test ayrımı: test seti en sonda sadece bir kez kullanılacak
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Eğitim:", X_train.shape, "Test:", X_test.shape)


kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Basitten karmaşığa doğru 4 model ve denenecek hiperparametreler
# (Ridge, ölçeklemeye duyarlı olduğu için Pipeline içinde StandardScaler ile kullanılıyor)
modeller = {
    "Ridge Regresyon": (
        Pipeline([("scaler", StandardScaler()), ("model", Ridge())]),
        {"model__alpha": np.logspace(-2, 3, 15)}   # 0.01 ile 1000 arasında 15 değer
    ),
    "Karar Ağacı": (
        DecisionTreeRegressor(random_state=42),
        {"max_depth": [3, 4, 5, 6, 8, 10, None],
         "min_samples_leaf": [1, 3, 5, 10, 20],
         "min_samples_split": [2, 5, 10, 20]}
    ),
    "Random Forest": (
        RandomForestRegressor(random_state=42),
        {"n_estimators": [100, 200, 300],
         "max_depth": [5, 8, 12, None],
         "min_samples_leaf": [1, 2, 5, 10],
         "max_features": ["sqrt", 0.5, 1.0]}
    ),
    "Gradient Boosting": (
        GradientBoostingRegressor(random_state=42),
        {"n_estimators": [100, 200, 300],
         "learning_rate": [0.01, 0.05, 0.1],
         "max_depth": [2, 3, 4],
         "subsample": [0.7, 0.85, 1.0]}
    ),
}

en_iyi_modeller = {}
sonuclar = []

for isim, (model, parametreler) in modeller.items():
    arama = RandomizedSearchCV(
        model,
        param_distributions=parametreler,
        n_iter=15,              # Ridge gibi küçük uzaylarda sklearn otomatik düşürür
        cv=kfold,
        scoring="r2",
        random_state=42,
        n_jobs=-1
    )
    arama.fit(X_train, y_train)

    en_iyi = arama.best_estimator_
    en_iyi_modeller[isim] = en_iyi

    # Test seti sadece burada, hiperparametre seçimi bittikten sonra kullanılıyor
    tahmin = en_iyi.predict(X_test)
    sonuclar.append({
        "Model": isim,
        "CV R²": arama.best_score_,
        "Test R²": r2_score(y_test, tahmin),
        "Test MAE (TL)": mean_absolute_error(y_test, tahmin),
        "En İyi Parametreler": arama.best_params_
    })
    print(isim, "tamamlandı")


tablo = pd.DataFrame(sonuclar).set_index("Model")
tablo["Fark (CV - Test)"] = tablo["CV R²"] - tablo["Test R²"]

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 80)
print(tablo[["CV R²", "Test R²", "Test MAE (TL)", "Fark (CV - Test)"]].round(3))
print()
print(tablo["En İyi Parametreler"])

# CV R² ve Test R² yan yana
tablo[["CV R²", "Test R²"]].plot(kind="bar", figsize=(8, 5), rot=15)
plt.title("Model Karşılaştırması (R²)")
plt.ylabel("R²")
plt.ylim(0, 1)
plt.show()

# CV R²'si en yüksek model "en başarılı" kabul edilir (seçimi test setine bakarak yapmıyoruz)
en_iyi_isim = tablo["CV R²"].idxmax()
en_iyi_model = en_iyi_modeller[en_iyi_isim]
print("En başarılı model:", en_iyi_isim)


# Görsel Analiz ile Model Açıklanabilirliği

y_tahmin = en_iyi_model.predict(X_test)

# 6.1 Gerçek - Tahmin grafiği
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_tahmin, alpha=0.5)
sinirlar = [y_test.min(), y_test.max()]
plt.plot(sinirlar, sinirlar, "r--")   # mükemmel tahmin çizgisi
plt.xlabel("Gerçek Fiyat (TL)")
plt.ylabel("Tahmin Edilen Fiyat (TL)")
plt.title(f"{en_iyi_isim}: Gerçek - Tahmin")
plt.show()

# 6.2 Hata (residual) dağılımı
hata = y_test - y_tahmin
plt.figure(figsize=(7, 4))
sns.histplot(hata, bins=30, kde=True)
plt.axvline(0, color="r", linestyle="--")
plt.title("Tahmin Hataları (Gerçek - Tahmin)")
plt.xlabel("Hata (TL)")
plt.show()

# 6.3 Özellik önemi: bir özelliğin değerlerini karıştırınca R² ne kadar düşüyor?
onem = permutation_importance(en_iyi_model, X_test, y_test,
                              n_repeats=20, random_state=42, scoring="r2")
onem_df = pd.DataFrame({"Özellik": X.columns, "Önem": onem.importances_mean}) \
            .sort_values("Önem", ascending=True)

plt.figure(figsize=(7, 6))
plt.barh(onem_df["Özellik"], onem_df["Önem"])
plt.title(f"{en_iyi_isim}: Özellik Önemi (R² düşüşü)")
plt.xlabel("Önem")
plt.show()

# 6.4 En önemli 2 özelliğin fiyata etkisi (Partial Dependence)
en_onemli = onem_df["Özellik"].iloc[::-1].head(2).tolist()
fig, ax = plt.subplots(figsize=(10, 4))
PartialDependenceDisplay.from_estimator(en_iyi_model, X_train, en_onemli, ax=ax)
plt.suptitle("Özelliklerin Tahmin Edilen Fiyata Etkisi")
plt.tight_layout()
plt.show()

# 6.5 Karar mekanizmasını gözle görmek için sığ (derinlik 3) bir karar ağacı
# Not: En iyi model ağaç topluluğu olduğu için tek bir ağaç gibi çizilemez.
# Bu ağaç, fiyatın hangi özelliklere göre nasıl bölündüğünü sadeleştirilmiş şekilde gösterir.
sade_agac = DecisionTreeRegressor(max_depth=3, min_samples_leaf=10, random_state=42)
sade_agac.fit(X_train, y_train)
print("Sade ağacın Test R²'si:", round(r2_score(y_test, sade_agac.predict(X_test)), 3))

plt.figure(figsize=(18, 8))
plot_tree(sade_agac, feature_names=X.columns, filled=True, rounded=True,
          impurity=False, precision=0, fontsize=10)   # value = o gruptaki ortalama fiyat (TL)
plt.title("Sadeleştirilmiş Karar Ağacı (max_depth=3)")
plt.show()