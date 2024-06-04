from pytube import YouTube
from pydub import AudioSegment
import os
from tqdm import tqdm
from pydub.utils import which
from pydub.effects import speedup, low_pass_filter, high_pass_filter
import librosa
import numpy as np
import random

# Explicitly set the paths to ffmpeg and ffprobe in the environment
os.environ["PATH"] += os.pathsep + "D:\\ffmpeg\\bin"

# Check if ffmpeg and ffprobe are correctly found
ffmpeg_path = which("ffmpeg")
ffprobe_path = which("ffprobe")


def download_audio(youtube_url, output_path='.'):
    try:
        yt = YouTube(youtube_url)
        audio_stream = yt.streams.filter(only_audio=True).order_by('abr').desc().first()
        file_size = audio_stream.filesize
        output_file = os.path.join(output_path, audio_stream.default_filename)

        with tqdm(total=file_size, unit='B', unit_scale=True, desc='Downloading') as pbar:
            def progress(chunk, file_handle, bytes_remaining):
                pbar.update(file_size - bytes_remaining)

            yt.register_on_progress_callback(progress)
            audio_stream.download(output_path)

        wav_output_file = os.path.splitext(output_file)[0] + '.wav'
        audio = AudioSegment.from_file(output_file)
        audio.export(wav_output_file, format='wav')
        os.remove(output_file)
        print(f'Audio downloaded and converted to WAV successfully: {wav_output_file}')
        return wav_output_file

    except Exception as e:
        print(f'Error downloading audio: {e}')
        return None


def split_audio(audio_file, chunk_length_ms=1500):
    audio = AudioSegment.from_file(audio_file)
    chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
    return chunks


def detect_pitch(audio_segment):
    y = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
    sr = audio_segment.frame_rate
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch = np.mean(pitches[np.nonzero(pitches)])
    return pitch


def detect_tempo(audio_segment):
    y = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
    sr = audio_segment.frame_rate
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
    return tempo[0]


def change_pitch(audio_segment, semitones):
    return audio_segment._spawn(audio_segment.raw_data, overrides={
        "frame_rate": int(audio_segment.frame_rate * (2.0 ** (semitones / 12.0)))
    }).set_frame_rate(audio_segment.frame_rate)


def apply_speed_change(audio_segment, speed=1.0):
    # Ensure minimum chunk length for speedup
    if len(audio_segment) < 150:
        return audio_segment
    return speedup(audio_segment, playback_speed=speed)


def apply_reverb(audio_segment, reverb_level=50):
    audio_segment = low_pass_filter(audio_segment, reverb_level)
    audio_segment = high_pass_filter(audio_segment, reverb_level)
    return audio_segment


def manipulate_audio(chunks):
    manipulated_chunks = []
    for chunk in chunks:
        detected_pitch = detect_pitch(chunk)

        # Calculate random changes within ±10%
        pitch_change = random.uniform(0.5, 1.5)
        speed_change_factor = random.uniform(0.98, 1.02)
        speed_change = max(0.99, min(1.01, speed_change_factor))  # Limit speed change to 0.5x to 2.0x
        reverb_change = random.randint(45, 55)  # Assuming a base reverb level of 50

        chunk = change_pitch(chunk, semitones=pitch_change)
        chunk = apply_speed_change(chunk, speed=speed_change)
        chunk = apply_reverb(chunk, reverb_level=reverb_change)
        manipulated_chunks.append(chunk)
    return manipulated_chunks


def combine_audio(chunks):
    combined = AudioSegment.empty()
    for chunk in chunks:
        combined += chunk
    return combined


if __name__ == "__main__":
    youtube_url = 'https://www.youtube.com/watch?v=testCode'
    output_path = './downloads'
    os.makedirs(output_path, exist_ok=True)
    downloaded_file = download_audio(youtube_url, output_path)

    if downloaded_file:
        chunks = split_audio(downloaded_file, chunk_length_ms=500)
        manipulated_chunks = manipulate_audio(chunks)
        combined_audio = combine_audio(manipulated_chunks)
        combined_audio.export(os.path.join(output_path, 'manipulated_audio.wav'), format='wav')
        print(f'Manipulated audio saved successfully: {os.path.join(output_path, "manipulated_audio.wav")}')
