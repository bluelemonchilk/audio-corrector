import streamlit as st
import numpy as np
from scipy.fft import fft, ifft
from scipy.io import wavfile
from scipy.signal import resample_poly
import io
from fractions import Fraction
import traceback

st.set_page_config(page_title="Аудиометрия и Коррекция Сигнала", layout="wide")
st.title("🎛️ Аудиометрия и Коррекция Сигнала")

def resample_audio(audio_signal, current_fs, target_fs):
    if current_fs == target_fs:
        return audio_signal
    frac = Fraction(target_fs, current_fs).limit_denominator(1000)
    return resample_poly(audio_signal, frac.numerator, frac.denominator)

def process_wav_file(uploaded_file):
    if uploaded_file is None:
        return None
    fs, data = wavfile.read(uploaded_file)
    audio = data[:, 0] if len(data.shape) > 1 else data
    return fs, audio

st.sidebar.header("Загрузка аудиофайлов")
file1 = st.sidebar.file_uploader("1. Оригинальный звук", type=["wav"])
file2 = st.sidebar.file_uploader("2. Искаженный звук", type=["wav"])
file3 = st.sidebar.file_uploader("3. Входной файл для восстановления", type=["wav"])

if file1 and file2 and file3:
    try:
        st.write("1 чтение...")
        fs1, a1 = process_wav_file(file1)
        fs2, a2 = process_wav_file(file2)
        fs3, a3 = process_wav_file(file3)
        st.write("2 ресемплинг...")
        target_fs = fs1
        a2 = resample_audio(a2, fs2, target_fs)
        a3 = resample_audio(a3, fs3, target_fs)
        st.write("3 нормализация...")
        a1f = a1.astype(np.float32)/32768.0
        a2f = a2.astype(np.float32)/32768.0
        a3f = a3.astype(np.float32)/32768.0
        st.write("4 синхронизация эталонов...")
        min_len = min(len(a1f), len(a2f))
        a1c = a1f[:min_len]
        a2c = a2f[:min_len]
        st.write("5 БПФ эталонов...")
        spec_orig = fft(a1c)
        spec_heard = fft(a2c)
        st.write("6 расчёт коэффициента...")
        alpha = 1e-6 * np.max(np.abs(spec_heard))**2
        gain = (spec_orig * np.conj(spec_heard)) / (np.abs(spec_heard)**2 + alpha)
        max_gain = 50.0
        gain[np.abs(gain) > max_gain] *= max_gain / np.abs(gain[np.abs(gain) > max_gain])
        st.write("7 БПФ целевого...")
        spec_target = fft(a3f)
        st.write("8 интерполяция...")
        # упростим интерполяцию через resize
        gain_resized = np.interp(np.linspace(0, 1, len(spec_target)), np.linspace(0, 1, len(gain)), gain)
        spec_result = spec_target * gain_resized
        st.write("9 обратное БПФ...")
        signal_rec = np.real(ifft(spec_result))
        signal_rec = np.clip(signal_rec, -1.0, 0.9999695)
        out = (signal_rec * 32768.0).astype(np.int16)
        buf = io.BytesIO()
        wavfile.write(buf, target_fs, out)
        buf.seek(0)
        st.success("✅ Все шаги успешно выполнены!")
        st.download_button("Скачать результат", buf, "out.wav", "audio/wav")
    except Exception as e:
        st.error(f"Ошибка: {e}")
        st.code(traceback.format_exc())
else:
    st.info("Загрузите три файла.")
