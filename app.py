import streamlit as st
import numpy as np
import librosa
import joblib
import tensorflow as tf
import os
import soundfile as sf
import tempfile
import matplotlib.pyplot as plt
from matplotlib import colormaps

# --------------------------------- PARTE 1: EXTRAIR FEATURES --------------------------------- #

# Carregar o modelo e o scaler
MODEL_PATH = r"c:\Users\arthu\OneDrive\Documentos\Faculdade p1\trilha\miniprojeto2\models\audio_emotion_model.keras"  # Example
SCALER_PATH = r"c:\Users\arthu\OneDrive\Documentos\Faculdade p1\trilha\miniprojeto2\models\scaler.joblib"                # Example

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Lista de emoções
EMOTIONS = ["angry", "calm", "disgust", "fear",
            "happy", "neutral", "sad", "surprise"]


# Função para extrair features
def extract_features(audio_path):
    data, sr = librosa.load(audio_path, sr=16000, mono=True)
    features = []

    # Para fazer essa parte apenas peguei o código feito no notebook de feature-extraction-and-modelling
    # substituindo as variáveis
    
    # Zero Crossing Rate
    # Extract the zcr here
    # features.extend(zcr)

    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data,),axis=1) #way for measuring smoothness of a signal is to calculate the number of zero-crossing within a segment of that signal.
    features = np.hstack((features, zcr))
    
    

    # Chroma STFT
    # Extract the chroma stft here
    # features.extend(chroma)
    chroma_stft = np.mean(librosa.feature.chroma_stft(y=data,sr=sr),axis=1) #Compute a chromagram from a waveform or power spectrogram
    features = np.hstack((features, chroma_stft))


    # MFCCs
    # Extract the mfccs here
    # features.extend(mfccs)
    mfcc = np.mean(librosa.feature.mfcc(y=data,sr=sr),axis=1)
    features = np.hstack((features,mfcc))


    # RMS
    # Extract the rms here
    # features.extend(rms)
    rms = np.mean(librosa.feature.rms(y=data),axis=1)
    features = np.hstack((features, rms))

    # Mel Spectrogram
    # Extract the mel here
    # features.extend(mel)
    mel = np.mean(librosa.feature.melspectrogram(y=data,sr=sr),axis=1) 
    features = np.hstack((features, mel))

    # Garantir que tenha exatamente 162 features (ou truncar/zerar)
    target_length = 162
    if len(features) < target_length:
        features.extend([0] * (target_length - len(features)))
    elif len(features) > target_length:
        features = features[:target_length]

    return np.array(features).reshape(1, -1)


# --------------------------------- PARTE 2: STREAMLIT --------------------------------- #

# Configuração do app Streamlit (Título e descrição)
# Code here
st.title("🎶 Detector de emoções em Áudio")
st.write("Envie um arquivo de áudio para análise!")


# Upload de arquivo de áudio (wav, mp3, ogg)
uploaded_file = st.file_uploader(
    "Escolha um arquivo de áudio...", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    # Salvar temporariamente o áudio
    # Code here
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav") # criar um arquivo temporário com nome visível
    temp_audio.write(uploaded_file.getvalue()) #escrevendo os dados do arquivo
    audio_path = temp_audio.name # caminho absoluto do arquivo temporário no sistema
    temp_audio.close() # fechando o arquivo 

    # Reproduzir o áudio enviado
    # Code here
    st.audio(uploaded_file)

    # Extrair features
    # Code here
    features = extract_features(audio_path)

    # Normalizar os dados com o scaler treinado
    # Code here
    features_scaled = scaler.transform(features)

    # Ajustar formato para o modelo
    # Code here
    features_final = np.expand_dims(features_scaled, axis=2)

    # Fazer a predição
    # Code here
    predictions = model.predict(features_final)
    emotion = EMOTIONS[np.argmax(predictions[0])]

    # Exibir o resultado
    # Code here
    st.success(f"🎭 Emoção detectada: {emotion}")

    # Exibir probabilidades (gráfico de barras)
    # Code here
    cores = plt.cm.tab20(np.linspace(0, 1, len(predictions)))
    classes = EMOTIONS
    plt.style.use('dark_background') # achei que o fundo branco ficava muito destoante da cor do site por isso utilizei essa função
    fig, ax = plt.subplots()
    ax.bar(classes,predictions[0],color=cores)
    st.pyplot(fig)
    # Remover o arquivo temporário
    # Code here
    os.remove(audio_path)
