import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from scipy.io import wavfile
from scipy.signal import resample_poly, correlate, butter, sosfilt
from scipy.ndimage import gaussian_filter1d
import io

st.set_page_config(page_title="Аудиометрия и Коррекция Сигнала (с шумоподавлением)", layout="wide")
st.title("🎛️ Аудиометрия и Коррекция Сигнала — Улучшенная версия")
st.write("Загрузите эталоны (оригинал → искажённый) и целевой файл. Алгоритм использует регуляризацию, сглаживание и синхронизацию.")

# ----------------------------------------------
# Функции обработки
# ----------------------------------------------
def resample_audio(audio_signal, current_fs, target_fs):
    if current_fs == target_fs:
        return audio_signal
    from fractions import Fraction
    frac = Fraction(target_fs, current_fs).limit_denominator(1000)
    return resample_poly(audio_signal, frac.numerator, frac.denominator)

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

def align_signals(ref, degraded):
    """Выравнивание сигналов по времени с помощью взаимной корреляции."""
    corr = correlate(degraded, ref, mode='same')
    delay = np.argmax(np.abs(corr)) - len(ref)//2
    if delay > 0:
        degraded_aligned = degraded[delay:]
        ref_aligned = ref[:-delay] if delay > 0 else ref
    else:
        degraded_aligned = degraded[-delay:]
        ref_aligned = ref[:delay] if delay < 0 else ref
    min_len = min(len(ref_aligned), len(degraded_aligned))
    return ref_aligned[:min_len], degraded_aligned[:min_len]

def apply_window(signal, window_type='hanning'):
    if window_type == 'hanning':
        win = np.hanning(len(signal))
    elif window_type == 'hamming':
        win = np.hamming(len(signal))
    else:
        win = np.ones(len(signal))
    return signal * win, win

def wiener_filter(original_spectrum, degraded_spectrum, alpha):
    """Винеровская регуляризация: H = conj(D)*O / (|D|^2 + alpha*mean(|D|^2))"""
    mean_power = np.mean(np.abs(degraded_spectrum)**2)
    denom = np.abs(degraded_spectrum)**2 + alpha * mean_power
    coeff = (original_spectrum * np.conj(degraded_spectrum)) / denom
    return coeff

def smooth_coefficient(coeff, sigma):
    """Сглаживание амплитуды коэффициента, фаза сохраняется."""
    mag = np.abs(coeff)
    phase = np.angle(coeff)
    mag_smoothed = gaussian_filter1d(mag, sigma=sigma, mode='reflect')
    return mag_smoothed * np.exp(1j * phase)

def apply_frequency_cutoff(coeff, fs, cutoff_hz):
    """Принудительно устанавливаем коэффициент = 1 выше заданной частоты."""
    n = len(coeff)
    freqs = np.linspace(0, fs, n)
    coeff_copy = coeff.copy()
    coeff_copy[freqs > cutoff_hz] = 1.0
    return coeff_copy

def interpolate_coefficient(coeff, target_len):
    """Интерполяция комплексного коэффициента на новую длину (по амплитуде и фазе отдельно)."""
    current_len = len(coeff)
    if current_len == target_len:
        return coeff
    xp = np.linspace(0, 1, current_len)
    xnew = np.linspace(0, 1, target_len)
    mag = np.abs(coeff)
    phase = np.angle(coeff)
    mag_interp = np.interp(xnew, xp, mag)
    phase_interp = np.interp(xnew, xp, phase)
    return mag_interp * np.exp(1j * phase_interp)

# ----------------------------------------------
# Боковая панель с параметрами
# ----------------------------------------------
st.sidebar.header("⚙️ Настройки фильтрации")
alpha = st.sidebar.slider("Регуляризация (alpha)", 0.001, 0.2, 0.01, 0.001, format="%.3f")
sigma = st.sidebar.slider("Сглаживание коэффициента (sigma)", 0.5, 10.0, 3.0, 0.5)
cutoff_hz = st.sidebar.slider("Обрезать частоты выше (Гц)", 2000, 20000, 8000, 500)
use_window = st.sidebar.checkbox("Использовать оконное взвешивание (Ханна)", value=True)
post_filter = st.sidebar.checkbox("Постфильтр низких частот (убирает остаточный шум)", value=True)
sync_signals = st.sidebar.checkbox("Синхронизировать эталоны по времени", value=True)

st.sidebar.markdown("---")
st.sidebar.info("Настройки влияют на подавление шума. Начните с alpha=0.01, sigma=3.0, cutoff=8000 Гц.")

# ----------------------------------------------
# Загрузка файлов
# ----------------------------------------------
st.sidebar.header("Загрузка аудиофайлов")
file1 = st.sidebar.file_uploader("1. Оригинальный эталон (чистый)", type=["wav"])
file2 = st.sidebar.file_uploader("2. Искажённый эталон (после тракта)", type=["wav"])
file3 = st.sidebar.file_uploader("3. Целевой файл для восстановления", type=["wav"])

if file1 and file2 and file3:
    st.success("Файлы загружены. Выполняется обработка...")
    res1 = process_wav_file(file1)
    res2 = process_wav_file(file2)
    res3 = process_wav_file(file3)
    
    if res1 and res2 and res3:
        fs1, audio1 = res1
        fs2, audio2 = res2
        fs3, audio3 = res3
        
        target_fs = fs1  # привязываемся к частоте оригинала
        
        # Ресемплинг до единой частоты
        audio2 = resample_audio(audio2, fs2, target_fs)
        audio3 = resample_audio(audio3, fs3, target_fs)
        
        # Приведение к float [-1, 1)
        audio1_float = audio1.astype(np.float32) / 32768.0
        audio2_float = audio2.astype(np.float32) / 32768.0
        audio3_float = audio3.astype(np.float32) / 32768.0
        
        # ----- Синхронизация эталонов по времени -----
        if sync_signals:
            audio1_calib, audio2_calib = align_signals(audio1_float, audio2_float)
            st.info(f"Синхронизация выполнена. Длина эталонов: {len(audio1_calib)} отсчётов.")
        else:
            min_len = min(len(audio1_float), len(audio2_float))
            audio1_calib = audio1_float[:min_len]
            audio2_calib = audio2_float[:min_len]
        
        # ----- Применение окна для калибровки -----
        if use_window:
            audio1_win, win1 = apply_window(audio1_calib, 'hanning')
            audio2_win, win2 = apply_window(audio2_calib, 'hanning')
        else:
            audio1_win = audio1_calib
            audio2_win = audio2_calib
        
        # БПФ для калибровки
        spectrum_original = fft(audio1_win)
        spectrum_degraded = fft(audio2_win)
        
        # Винеровская регуляризация
        coeff_raw = wiener_filter(spectrum_original, spectrum_degraded, alpha)
        
        # Сглаживание амплитуды
        coeff_smoothed = smooth_coefficient(coeff_raw, sigma)
        
        # Обрезание высоких частот
        coeff_cut = apply_frequency_cutoff(coeff_smoothed, target_fs, cutoff_hz)
        
        # ----- Подготовка целевого сигнала -----
        len3 = len(audio3_float)
        if use_window:
            audio3_win, win3 = apply_window(audio3_float, 'hanning')
        else:
            audio3_win = audio3_float
            win3 = np.ones(len3)
        
        spectrum_target = fft(audio3_win)
        
        # Интерполяция коэффициента до длины целевого сигнала
        coeff_final = interpolate_coefficient(coeff_cut, len3)
        
        # Применение коррекции
        spectrum_result = spectrum_target * coeff_final
        signal_reconstructed = np.real(ifft(spectrum_result))
        
        # Компенсация окна (деление на окно, с защитой от деления на ноль)
        if use_window:
            win3[win3 == 0] = 1e-10
            signal_reconstructed = signal_reconstructed / win3
        
        # Постфильтрация для удаления остаточного шума
        if post_filter:
            # Фильтр Баттерворта 2-го порядка с частотой среза 0.95*Nyquist
            sos = butter(2, 0.95, btype='low', output='sos', fs=target_fs)
            signal_reconstructed = sosfilt(sos, signal_reconstructed)
        
        # Нормализация и квантование
        signal_reconstructed = np.clip(signal_reconstructed, -1.0, 0.9999695)
        signal_to_save = (signal_reconstructed * 32768.0).astype(np.int16)
        
        # Сохранение в буфер
        wav_buffer = io.BytesIO()
        wavfile.write(wav_buffer, target_fs, signal_to_save)
        wav_buffer.seek(0)
        
        # ----- Вывод результатов -----
        st.subheader("🎵 Результат восстановления")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Исходный целевой файл:")
            st.audio(file3)
        with col2:
            st.write("💥 **Восстановленный сигнал (с шумоподавлением):**")
            st.audio(wav_buffer, format="audio/wav")
            st.download_button(label="📥 Скачать WAV", data=wav_buffer, file_name="restored_audio.wav", mime="audio/wav")
        
        # ----- Визуализация (спектры и временные формы) -----
        try:
            fig, axs = plt.subplots(4, 2, figsize=(14, 20))
            t_calib = np.arange(len(audio1_calib)) / target_fs
            t_target = np.arange(len3) / target_fs
            freqs_calib = np.linspace(0, target_fs/2, len(audio1_calib)//2)
            freqs_target = np.linspace(0, target_fs/2, len3//2)
            
            # Спектры для калибровки
            spec_orig = np.abs(spectrum_original[:len(audio1_calib)//2])
            spec_degr = np.abs(spectrum_degraded[:len(audio1_calib)//2])
            spec_target = np.abs(spectrum_target[:len3//2])
            spec_result = np.abs(spectrum_result[:len3//2])
            
            # Временные формы
            axs[0,0].plot(t_calib, audio1_calib, color='green', linewidth=0.5)
            axs[0,0].set_title("Оригинальный эталон: A(t)")
            axs[0,1].plot(freqs_calib, spec_orig, color='green')
            axs[0,1].set_title("Оригинальный эталон: спектр")
            
            axs[1,0].plot(t_calib, audio2_calib, color='red', linewidth=0.5)
            axs[1,0].set_title("Искажённый эталон: A(t)")
            axs[1,1].plot(freqs_calib, spec_degr, color='red')
            axs[1,1].set_title("Искажённый эталон: спектр")
            
            axs[2,0].plot(t_target, audio3_float, color='orange', linewidth=0.5)
            axs[2,0].set_title("Целевой сигнал: A(t)")
            axs[2,1].plot(freqs_target, spec_target, color='orange')
            axs[2,1].set_title("Целевой сигнал: спектр")
            
            axs[3,0].plot(t_target, signal_reconstructed, color='blue', linewidth=0.5)
            axs[3,0].set_title("Восстановленный сигнал: A(t)")
            axs[3,1].plot(freqs_target, spec_result, color='blue')
            axs[3,1].set_title("Восстановленный сигнал: спектр")
            
            # Единые масштабы для спектров
            max_y = max(np.max(spec_orig), np.max(spec_degr), np.max(spec_target), np.max(spec_result)) * 1.05
            for i in range(4):
                axs[i,1].set_ylim(0, max_y)
                axs[i,1].set_xlim(0, target_fs/2)
                axs[i,1].grid(True, alpha=0.3)
                axs[i,0].grid(True, alpha=0.3)
                axs[i,0].set_xlabel("Время (с)")
                axs[i,1].set_xlabel("Частота (Гц)")
                axs[i,0].set_ylabel("Амплитуда")
                axs[i,1].set_ylabel("Амплитуда")
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.warning(f"Графики не построены: {e}")
else:
    st.info("💡 Загрузите все три файла (оригинал → искажённый → целевой) для начала работы.")
