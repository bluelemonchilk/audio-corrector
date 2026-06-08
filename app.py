import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.io import wavfile
from scipy.signal import resample_poly
import io
from fractions import Fraction
import traceback

# Настройка страницы Streamlit
st.set_page_config(page_title="Аудиометрия и Коррекция Сигнала", layout="wide")

st.title("🎛️ Аудиометрия и Коррекция Сигнала")
st.write("Загрузите эталоны для калибровки искажения и любой по длине целевой файл для восстановления.")

# Функция надежного полифазного ресэмплинга
def resample_audio(audio_signal, current_fs, target_fs):
    if current_fs == target_fs:
        return audio_signal
    frac = Fraction(target_fs, current_fs).limit_denominator(1000)
    up = frac.numerator
    down = frac.denominator
    return resample_poly(audio_signal, up, down)

# Функция для обработки загруженного WAV файла
def process_wav_file(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        fs, data = wavfile.read(uploaded_file)
        if data.dtype != np.int16:
            st.error(f"Файл {uploaded_file.name} имеет формат {data.dtype}. Поддерживается только 16-bit WAV (int16).")
            return None
        audio = data[:, 0] if len(data.shape) > 1 else data
        return fs, audio
    except Exception as e:
        st.error(f"Ошибка при чтении файла {uploaded_file.name}: {e}")
        return None

# Функция интерполяции спектрального коэффициента под нужную длину
def interpolate_spectrum_coef(coef, target_len):
    current_len = len(coef)
    if current_len == target_len:
        return coef
    xp = np.linspace(0, 1, current_len)
    xnew = np.linspace(0, 1, target_len)
    real_interp = np.interp(xnew, xp, np.real(coef))
    imag_interp = np.interp(xnew, xp, np.imag(coef))
    return real_interp + 1j * imag_interp

# Боковая панель
st.sidebar.header("Загрузка аудиофайлов")
file1 = st.sidebar.file_uploader("1. Оригинальный звук", type=["wav"])
file2 = st.sidebar.file_uploader("2. Искаженный звук", type=["wav"])
file3 = st.sidebar.file_uploader("3. Входной файл для восстановления", type=["wav"])

if file1 and file2 and file3:
    try:
        st.success("✅ Файлы загружены. Начинаем расчёт...")
        
        # --- ШАГ 1: чтение файлов ---
        st.write("📖 Шаг 1/9: Чтение WAV файлов...")
        res1 = process_wav_file(file1)
        res2 = process_wav_file(file2)
        res3 = process_wav_file(file3)
        if not (res1 and res2 and res3):
            st.error("Не удалось прочитать один из файлов.")
            st.stop()
        fs1, audio1 = res1
        fs2, audio2 = res2
        fs3, audio3 = res3
        st.write("   ✓ файлы прочитаны.")
        
        # --- ШАГ 2: ресемплинг к единой частоте ---
        st.write("🔄 Шаг 2/9: Приведение частоты дискретизации...")
        target_fs = fs1
        audio2 = resample_audio(audio2, fs2, target_fs)
        audio3 = resample_audio(audio3, fs3, target_fs)
        st.write(f"   ✓ частота {target_fs} Гц.")
        
        # --- ШАГ 3: нормализация ---
        st.write("📊 Шаг 3/9: Нормализация в float32...")
        audio1_float = audio1.astype(np.float32) / 32768.0
        audio2_float = audio2.astype(np.float32) / 32768.0
        audio3_float = audio3.astype(np.float32) / 32768.0
        st.write("   ✓ нормализация выполнена.")
        
        # --- ШАГ 4: синхронизация длин калибровочных эталонов ---
        st.write("✂️ Шаг 4/9: Обрезка эталонов до минимальной длины...")
        min_len_calibration = min(len(audio1_float), len(audio2_float))
        audio1_calib = audio1_float[:min_len_calibration]
        audio2_calib = audio2_float[:min_len_calibration]
        st.write(f"   ✓ длина калибровки: {min_len_calibration} отсчётов.")
        
        # --- ШАГ 5: БПФ эталонов ---
        st.write("⚡ Шаг 5/9: Вычисление спектров эталонов (FFT)...")
        spectrum_original = fft(audio1_calib)
        spectrum_heard = fft(audio2_calib)
        st.write("   ✓ FFT выполнено.")
        
        # --- ШАГ 6: расчёт коэффициента усиления с регуляризацией ---
        st.write("🎛️ Шаг 6/9: Расчёт коэффициента коррекции...")
        max_gain = 50.0
        alpha = 1e-6 * np.max(np.abs(spectrum_heard))**2
        koeff_usilenia_base = (spectrum_original * np.conj(spectrum_heard)) / (np.abs(spectrum_heard)**2 + alpha)
        gain_magnitude = np.abs(koeff_usilenia_base)
        too_high_gain = gain_magnitude > max_gain
        koeff_usilenia_base[too_high_gain] = (koeff_usilenia_base[too_high_gain] / gain_magnitude[too_high_gain]) * max_gain
        st.write("   ✓ коэффициент готов.")
        
        # --- ШАГ 7: БПФ целевого файла ---
        st.write("🎵 Шаг 7/9: Вычисление спектра целевого файла...")
        len3 = len(audio3_float)
        spectrum_to_correct = fft(audio3_float)
        st.write(f"   ✓ длина целевого файла: {len3} отсчётов.")
        
        # --- ШАГ 8: интерполяция коэффициента и коррекция ---
        st.write("📈 Шаг 8/9: Интерполяция коэффициента и применение коррекции...")
        koeff_usilenia_stretched = interpolate_spectrum_coef(koeff_usilenia_base, len3)
        spectrum_result = spectrum_to_correct * koeff_usilenia_stretched
        st.write("   ✓ коррекция выполнена.")
        
        # --- ШАГ 9: обратное БПФ и сохранение ---
        st.write("🔊 Шаг 9/9: Обратное преобразование и формирование WAV...")
        signal_reconstructed = np.real(ifft(spectrum_result))
        recon_clipped = np.clip(signal_reconstructed, -1.0, 0.9999695)
        signal_to_save = (recon_clipped * 32768.0).astype(np.int16)
        
        wav_buffer = io.BytesIO()
        wavfile.write(wav_buffer, target_fs, signal_to_save)
        wav_buffer.seek(0)
        st.write("   ✓ готово!")
        
        # --- Вывод результатов ---
        st.subheader("🎵 Результат восстановления")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Исходный файл для коррекции:")
            st.audio(file3)
        with col2:
            st.write("💥 **Восстановленный файл:**")
            st.audio(wav_buffer, format="audio/wav")
            st.download_button(
                label="📥 Скачать восстановленный WAV",
                data=wav_buffer,
                file_name="reconstructed_sound.wav",
                mime="audio/wav"
            )
        
        st.markdown("---")
        st.subheader("📊 Визуализация сигналов и их спектров")
        
        # --- Графики ---
        spec_3_plot = np.abs(spectrum_to_correct[:len3 // 2])
        spec_res_plot = np.abs(spectrum_result[:len3 // 2])
        max_y_value = max(np.max(spec_3_plot), np.max(spec_res_plot)) * 1.05
        max_x_value = target_fs / 2
        
        fig, axs = plt.subplots(4, 2, figsize=(14, 20))
        t_calib = np.arange(min_len_calibration) / target_fs
        t3 = np.arange(len3) / target_fs
        freqs_calib = np.linspace(0, target_fs/2, min_len_calibration // 2)
        freqs3 = np.linspace(0, target_fs/2, len3 // 2)
        
        # Оригинал
        axs[0, 0].plot(t_calib, audio1_calib, color='green')
        axs[0, 0].set_title("Оригинальный эталон: A(t)")
        axs[0, 1].plot(freqs_calib, np.abs(spectrum_original[:min_len_calibration // 2]), color='green')
        axs[0, 1].set_title("Оригинальный эталон: спектр")
        
        # Искажённый
        axs[1, 0].plot(t_calib, audio2_calib, color='red')
        axs[1, 0].set_title("Искаженный эталон: A(t)")
        axs[1, 1].plot(freqs_calib, np.abs(spectrum_heard[:min_len_calibration // 2]), color='red')
        axs[1, 1].set_title("Искаженный эталон: спектр")
        
        # Целевой
        axs[2, 0].plot(t3, audio3_float, color='orange')
        axs[2, 0].set_title("Файл для коррекции: A(t)")
        axs[2, 1].plot(freqs3, spec_3_plot, color='orange')
        axs[2, 1].set_title("Файл для коррекции: спектр")
        axs[2, 1].set_ylim(0, max_y_value)
        axs[2, 1].set_xlim(0, max_x_value)
        
        # Восстановленный
        axs[3, 0].plot(t3, signal_reconstructed, color='blue')
        axs[3, 0].set_title("Восстановленный сигнал: A(t)")
        axs[3, 1].plot(freqs3, spec_res_plot, color='blue')
        axs[3, 1].set_title("Восстановленный сигнал: спектр")
        axs[3, 1].set_ylim(0, max_y_value)
        axs[3, 1].set_xlim(0, max_x_value)
        
        for ax in axs.flatten():
            ax.set_xlabel("Время (с)" if "A(t)" in ax.get_title() else "Частота (Гц)")
            ax.set_ylabel("Амплитуда")
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        
    except Exception as e:
        st.error("❌ Произошла ошибка при выполнении:")
        st.code(traceback.format_exc())
        st.stop()
else:
    st.info("💡 Загрузите калибровочные эталоны и аудиофайл для исправления.")
