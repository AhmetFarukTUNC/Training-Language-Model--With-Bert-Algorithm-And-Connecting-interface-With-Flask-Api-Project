# Flask ve gerekli kütüphanelerin içe aktarılması
from flask import Flask, request, jsonify, render_template  # Flask web sunucusu ve JSON/HTML işlemleri
from transformers import BertTokenizer, BertForSequenceClassification  # Hugging Face BERT modeli ve tokenizer
import torch  # PyTorch ile model işlemleri
from flask_cors import CORS  # CORS politikası için (tarayıcıdan gelen isteklere izin vermek için)

# Flask uygulaması oluşturuluyor
app = Flask(__name__)
CORS(app)  # Tüm domainlerden gelen isteklere CORS izni verilir (frontend arayüzler için gerekli)

# --- MODELİ YÜKLEME BÖLÜMÜ ---
model_dir = "./results"  # Fine-tuned BERT modelinin ve tokenizer dosyalarının bulunduğu klasör

# Tokenizer ve model yükleniyor
tokenizer = BertTokenizer.from_pretrained(model_dir)  # tokenizer klasörünü model_dir içinden al
model = BertForSequenceClassification.from_pretrained(model_dir, trust_remote_code=True)  # fine-tuned model
model.eval()  # Model 'inference' moduna alınır (eğitimde değil, tahmin için)

# --- ROOT ENDPOINT ---
@app.route("/")
def home():
    # templates/index.html dosyasını render eder (bir HTML arayüzü)
    return render_template("index.html")

# --- PREDICT ENDPOINT ---
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # İstekten gelen JSON veriyi alıyoruz
        data = request.json
        text = data["text"]  # JSON'daki "text" alanını al

        # Gelen metni tokenizer ile BERT'e uygun tensor yapısına dönüştür
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        # Modeli kullanarak tahmin yap (gradient hesaplamaya gerek yok cunku tahmin yapıluyor eğitim kısmında değiliz)
        with torch.no_grad():
            outputs = model(**inputs)  # Model çıktısı (logits = ham skorlar)
            logits = outputs.logits  # Çıktı tensörü (sınıf skorları)
            predicted_class = torch.argmax(logits, dim=1).item()  # En yüksek skorlu sınıf (0 veya 1)
            confidence = torch.softmax(logits, dim=1).max().item()  # En yüksek sınıfın olasılığı (güven)

        # JSON yanıtı olarak sınıf ve güven oranı döndürülür
        return jsonify({
            "prediction": predicted_class,  # 0: negatif, 1: pozitif (veya başka sınıflar)
            "confidence": confidence        # Tahminin güven düzeyi (0-1 arası)
        })

    except Exception as e:
        # Hata durumunda JSON olarak hata mesajı döndürülür
        return jsonify({"error": str(e)})

# --- UYGULAMA SUNUCUSUNU ÇALIŞTIR ---
if __name__ == "__main__":
    # Flask sunucusunu başlat (debug=True geliştirici modudur)
    app.run(debug=True)
