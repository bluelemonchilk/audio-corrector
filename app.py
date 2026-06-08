import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from scipy.io import wavfile
from scipy.signal import resample_poly
import io
from fractions import Fraction

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
            
        # Конвертируем в моно (первый канал)
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
    st.success("Файлы загружены. Начинаем расчёт...")
    
    res1 = process_wav_file(file1)
    res2 = process_wav_file(file2)
    res3 = process_wav_file(file3)
    
    if res1 and res2 and res3:
        fs1, audio1 = res1
        fs2, audio2 = res2
        fs3, audio3 = res3
        
        # Шаг 1: Приводим все файлы к единой частоте дискретизации (оригинальной fs1)
        target_fs = fs1
        audio2 = resample_audio(audio2, fs2, target_fs)
        audio3 = resample_audio(audio3, fs3, target_fs)
        
        # Шаг 2: Нормализуем во float
        audio1_float = audio1.astype(np.float32) / 32768.0
        audio2_float = audio2.astype(np.float32) / 32768.0
        audio3_float = audio3.astype(np.float32) / 32768.0
        
        # Шаг 3: Синхронизируем длины ТОЛЬКО для калибровочных эталонов (1 и 2)
        min_len_calibration = min(len(audio1_float), len(audio2_float))
        audio1_calib = audio1_float[:min_len_calibration]
        audio2_calib = audio2_float[:min_len_calibration]
        
       # Шаг 4: Считаем спектр искажения по эталонам одинаковой длины
        spectrum_original = fft(audio1_calib)
        spectrum_heard = fft(audio2_calib)
        
        # --- ОБНОВЛЕННАЯ НАСТРОЙКА ФИЛЬТРАЦИИ ШУМА ---
        # Вы можете менять max_gain прямо здесь, теперь он будет работать!
        max_gain = 5.0  # Попробуйте поставить 5.0, 20.0, 50.0
        
        # Вместо деления на spectrum_heard + eps, используем регуляризацию Тихонова.
        # Она плавно гасит спектр там, где слышимый сигнал (знаменатель) слишком мал.
        # Параметр alpha подбирается автоматически от энергии сигнала
        #alpha = 0.001 * np.max(np.abs(spectrum_heard))**2
        
        # Физически это: (Original * Heard*) / (|Heard|^2 + alpha)
        koeff_usilenia_base = (spectrum_original * np.conj(spectrum_heard)) / (np.abs(spectrum_heard)**2 + 1e-12)
        
        # Ограничиваем пики коэффициента по амплитуде (наш max_gain)
        gain_magnitude = np.abs(koeff_usilenia_base)
        too_high_gain = gain_magnitude > max_gain
        
        # Там, где коэффициент превысил max_gain, мы аккуратно подрезаем его до max_gain, сохраняя фазу
        koeff_usilenia_base[too_high_gain] = (koeff_usilenia_base[too_high_gain] / gain_magnitude[too_high_gain]) * max_gain
        # ------------------------------------------------
        
        # Шаг 5: Берем третий файл целиком
        len3 = len(audio3_float)
        spectrum_to_correct = fft(audio3_float)
        
        # Шаг 6: Масштабируем коэффициент под длину третьего файла
        koeff_usilenia_stretched = interpolate_spectrum_coef(koeff_usilenia_base, len3)
        
        # Шаг 7: Коррекция спектра
        spectrum_result = spectrum_to_correct * koeff_usilenia_stretched
        
        # Шаг 8: Обратное преобразование Фурье
        signal_reconstructed = np.real(ifft(spectrum_result))
        
        # Сохранение результата
        recon_clipped = np.clip(signal_reconstructed, -1.0, 0.9999695)
        signal_to_save = (recon_clipped * 32768.0).astype(np.int16)
        
        wav_buffer = io.BytesIO()
        wavfile.write(wav_buffer, target_fs, signal_to_save)
        wav_buffer.seek(0)
        
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
        
        # Выделяем массивы для графиков спектров (только положительные частоты)
        spec_3_plot = np.abs(spectrum_to_correct[:len3 // 2])
        spec_res_plot = np.abs(spectrum_result[:len3 // 2])
        
        # --- НАХОДИМ ОБЩИЙ МАСШТАБ ДЛЯ 3 И 4 ГРАФИКОВ ---
        # По оси Y (максимальная амплитуда из двух сигналов с запасом 5% для красоты)
        max_y_value = max(np.max(spec_3_plot), np.max(spec_res_plot)) * 1.05
        # Максимальная частота по оси X
        max_x_value = target_fs / 2
        
        # Графики
        fig, axs = plt.subplots(4, 2, figsize=(14, 20))
        
        t_calib = np.arange(min_len_calibration) / target_fs
        t3 = np.arange(len3) / target_fs
        
        freqs_calib = np.linspace(0, target_fs/2, min_len_calibration // 2)
        freqs3 = np.linspace(0, target_fs/2, len3 // 2)
        
        # 1. Оригинал (калибровка)
        axs[0, 0].plot(t_calib, audio1_calib, color='green')
        axs[0, 0].set_title(f"Оригинальный эталон: A(t)")
        
        axs[0, 1].plot(freqs_calib, np.abs(spectrum_original[:min_len_calibration // 2]), color='green')
        axs[0, 1].set_title("Оригинальный эталон: спектр")
        
        # 2. Искаженный (калибровка)
        axs[1, 0].plot(t_calib, audio2_calib, color='red')
        axs[1, 0].set_title(f"Искаженный эталон: A(t)")
        
        axs[1, 1].plot(freqs_calib, np.abs(spectrum_heard[:min_len_calibration // 2]), color='red')
        axs[1, 1].set_title("Искаженный эталон: спектр")
        
        # 3. Входной файл для восстановления
        axs[2, 0].plot(t3, audio3_float, color='orange')
        axs[2, 0].set_title(f"Файл для коррекции: A(t)")
        
        axs[2, 1].plot(freqs3, spec_3_plot, color='orange')
        axs[2, 1].set_title("Файл для коррекции: спектр")
        axs[2, 1].set_ylim(0, max_y_value)  # Применяем общий масштаб по Y
        axs[2, 1].set_xlim(0, max_x_value)  # Применяем общий масштаб по X
        
        # 4. Восстановленный
        axs[3, 0].plot(t3, signal_reconstructed, color='blue')
        axs[3, 0].set_title(f"Восстановленный сигнал: A(t)")
        
        axs[3, 1].plot(freqs3, spec_res_plot, color='blue')
        axs[3, 1].set_title("Восстановленный сигнал: спектр")
        axs[3, 1].set_ylim(0, max_y_value)  # Применяем общий масштаб по Y
        axs[3, 1].set_xlim(0, max_x_value)  # Применяем общий масштаб по X
        
        for ax in axs.flatten():
            ax.set_xlabel("Время (с)" if "A(t)" in ax.get_title() else "Частота (Гц)")
            ax.set_ylabel("Амплитуда")
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)   
else:
    st.info("💡 Загрузите калибровочные эталоны и аудиофайл для исправления.")
