import os
import torch
import librosa
import numpy as np
import parselmouth
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from scipy.spatial.distance import cosine
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# =========================================================
# 1. إعدادات الموديل والجهاز
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"

print("جاري تحميل الموديل العربي...")
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME, output_hidden_states=True).to(device)
model.eval()


def transcribe_audio(audio_path):
    speech, sr = librosa.load(audio_path, sr=16000)
    input_values = processor(speech, sampling_rate=16000, return_tensors="pt").input_values.to(device)
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription

def detect_pronunciation_errors(reference_text, predicted_text):
    errors = []
    ref_chars = list(reference_text)
    pred_chars = list(predicted_text)
    min_len = min(len(ref_chars), len(pred_chars))

    for i in range(min_len):
        if ref_chars[i] != pred_chars[i]:
            errors.append(f"{ref_chars[i]} -> {pred_chars[i]}")

    if len(ref_chars) > len(pred_chars):
        for i in range(min_len, len(ref_chars)):
            errors.append(f"{ref_chars[i]} -> محذوف")
    elif len(pred_chars) > len(ref_chars):
        for i in range(min_len, len(pred_chars)):
            errors.append(f"إضافة: {pred_chars[i]}")
    return errors

# =========================================================
# 2. دوال استخراج الميزات (نفس منطق الكود السابق)
# =========================================================

def extract_embedding(audio_path):
    speech, sr = librosa.load(audio_path, sr=16000)
    input_values = processor(speech, sampling_rate=16000, return_tensors="pt").input_values.to(device)
    with torch.no_grad():
        outputs = model(input_values)
        hidden_states = outputs.hidden_states[10]
        embedding = hidden_states.squeeze(0).cpu().numpy()
    return embedding

def compute_similarity(ref_emb, test_emb):
    min_len = min(len(ref_emb), len(test_emb))
    ref_emb, test_emb = ref_emb[:min_len], test_emb[:min_len]
    similarities = [1 - cosine(r, t) for r, t in zip(ref_emb, test_emb)]
    return np.mean(similarities)

def extract_speech_features(audio_path):
    snd = parselmouth.Sound(audio_path)
    pitch = snd.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    mean_pitch = np.mean(pitch_values[pitch_values > 0]) if any(pitch_values > 0) else 0
    intensity = snd.to_intensity()
    mean_intensity = np.mean(intensity.values)
    return [mean_pitch, mean_intensity]

# =========================================================
# 3. بناء مصفوفة البيانات (معدل ليدعم كلمات متعددة)
# =========================================================

def load_multi_word_dataset(dataset_path, references_folder):
    X = []
    y = []
        
    ref_embeddings = {}
    print("جاري تجهيز البصمات الصوتية للمراجع (5 كلمات)...")
    for i in range(1, 6): # الكلمات من 1 إلى 5
        ref_path = os.path.join(references_folder, f"{i}.wav")
        if os.path.exists(ref_path):
            ref_embeddings[str(i)] = extract_embedding(ref_path)

    print(f"جاري قراءة المجلدات للمتحدثين من {dataset_path}...")
        
    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)
        if os.path.isdir(folder_path):    
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".wav"):                    
                    word_id = file_name.replace("_N", "").replace(".wav", "")
                    
                    if word_id in ref_embeddings:
                        file_path = os.path.join(folder_path, file_name)
                        try:
                            emb = extract_embedding(file_path)
                            sim = compute_similarity(ref_embeddings[word_id], emb)
                            pitch, intensity = extract_speech_features(file_path)
                            
                            X.append([sim, pitch, intensity])                            
                            y.append(1 if "_N" in file_name else 0)
                        except Exception as e:
                            print(f"خطأ في معالجة الملف {file_name}: {e}")

    return np.array(X), np.array(y)

# =========================================================
# 4. التدريب (تعديل المسارات)
# =========================================================

DATASET_PATH = "ASMDD_Dataset_Folders" 
REFERENCES_FOLDER = "Reference_Voices" 

X_train, y_train = load_multi_word_dataset(DATASET_PATH, REFERENCES_FOLDER)

print("\n" + "="*30)
print("التحقق من المصفوفات المستخرجة:")
print("="*30)

print(f"\n1. شكل مصفوفة الميزات (X_train Shape): {X_train.shape}")
print("أول 5 عينات من X_train (Similarity, Pitch, Intensity):")
print(X_train[:35])

print(f"\n2. شكل مصفوفة الأهداف (y_train Shape): {y_train.shape}")
print("أول 20 تصنيف في y_train (0=سليم, 1=خطأ):")
print(y_train[:35])

print(f"\n3. إحصائيات سريعة:")
print(f"- إجمالي العينات: {len(y_train)}")
print(f"- عدد الأخطاء المكتشفة (_N): {np.sum(y_train)}")
print(f"- عدد النطق السليم: {len(y_train) - np.sum(y_train)}")
print("="*30 + "\n")
# ----------------------------

if len(X_train) > 0:

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    print(f"تم تجميع {len(X_train)} عينة. جاري تدريب النموذج...")
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train_scaled, y_train)

    print("النموذج جاهز الآن للتقييم!")

else:
    print("تنبيه: لم يتم استخراج أي بيانات، تأكد من مسار الداتاسيت.")

# =========================================================
# 5. دالة فحص متحدث جديد (يفحص الـ 5 كلمات ويعطي تقرير)
# =========================================================

def evaluate_speaker_complete(speaker_folder, references_folder):
    print(f"\n" + "="*45)
    print(f"--- تقرير فحص المتحدث في مجلد: {speaker_folder} ---")
    print("="*45)
    
    results = []
    
    if not os.path.exists(speaker_folder):
        print(f"خطأ: المجلد {speaker_folder} غير موجود.")
        return

    for i in range(1, 6):
        word_id = str(i)    
        test_file = os.path.join(speaker_folder, f"{word_id}.wav")
        if not os.path.exists(test_file):
            test_file = os.path.join(speaker_folder, f"{word_id}_N.wav")
            
        if os.path.exists(test_file):
            ref_file = os.path.join(references_folder, f"{word_id}.wav")
            
            if os.path.exists(ref_file):
                try:                 
                    emb_test = extract_embedding(test_file)
                    emb_ref = extract_embedding(ref_file)
                    
                    sim = compute_similarity(emb_ref, emb_test)
                    pitch, intensity = extract_speech_features(test_file)
                                        
                    sample = scaler.transform([[sim, pitch, intensity]])
                    prediction = classifier.predict(sample)[0]
                    
                    ref_text = transcribe_audio(ref_file)
                    pred_text = transcribe_audio(test_file)
                    char_errors = detect_pronunciation_errors(ref_text, pred_text)
                    
                    status = "⚠️ اضطراب نطق" if prediction == 1 else "✅ نطق سليم"
                    print(f"الكلمة {word_id}: {status.ljust(15)} | التشابه الصوتي: {sim*100:.2f}%")

                    if char_errors:
                        print(f"   ∟ أخطاء الحروف المكتشفة: {', '.join(char_errors)}")
                    else:
                        print(f"   ∟ لا توجد أخطاء في الأحرف.")


                    results.append(prediction)
                except Exception as e:
                    print(f"حدث خطأ أثناء فحص الكلمة {word_id}: {e}")
            else:
                print(f"تنبيه: ملف المرجع {word_id}.wav غير موجود في مجلد المراجع.")
        else:
            print(f"الكلمة {word_id}: ملف الصوت غير موجود في مجلد الطفل.")
    
    if results:
        error_count = sum(results)
        total_words = len(results)
        error_rate = (error_count / total_words) * 100
        
        print("-" * 45)
        print(f"النتيجة النهائية للمتحدث:")
        print(f"عدد الكلمات التي بها اضطراب: {error_count} من أصل {total_words}")
        print(f"نسبة الاضطراب الإجمالية: {error_rate:.1f}%")
        
        if error_rate >= 40:
            print("التقييم: نوصي بعرض الطفل على أخصائي تخاطب.")
        else:
            print("التقييم: حالة النطق مستقرة وضمن الحدود الطبيعية.")
    print("="*45 + "\n")

# ==========================================
# استدعاء الفحص (تغيير المسارات حسب مجلداتك)
# ==========================================

my_test_folder = "Test_Child" 

if os.path.exists(my_test_folder):
    evaluate_speaker_complete(my_test_folder, REFERENCES_FOLDER)
else:
    print(f"يرجى إنشاء مجلد باسم {my_test_folder} ووضع 5 ملفات صوتية فيه للتجربة.")