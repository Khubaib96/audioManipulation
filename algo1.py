from pytube import YouTube
from pydub import AudioSegment
import os
from tqdm import tqdm
from pydub.utils import which
from pydub.effects import speedup, low_pass_filter, high_pass_filter
import librosa
import numpy as np
import random
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set the FFmpeg path
os.environ["PATH"] += os.pathsep + "D:/audioManipulation-main/ffmpeg-master-latest-win64-gpl/bin"

# Check if ffmpeg and ffprobe are correctly found
ffmpeg_path = which("ffmpeg")
ffprobe_path = which("ffprobe")

if not ffmpeg_path:
    raise EnvironmentError("FFmpeg not found. Ensure it is installed and the path is correctly set.")

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
        logging.info(f'Audio downloaded and converted to WAV successfully: {wav_output_file}')
        return wav_output_file

    except Exception as e:
        logging.error(f'Error downloading audio: {e}')
        return None


def split_audio(audio_file, chunk_length_ms=1500):
    logging.info('Splitting audio into chunks...')
    audio = AudioSegment.from_file(audio_file)
    chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
    logging.info(f'Split audio into {len(chunks)} chunks.')
    return chunks


def detect_pitch(audio_segment):
    y = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
    sr = audio_segment.frame_rate
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch = np.mean(pitches[np.nonzero(pitches)])
    return pitch


def change_pitch(audio_segment, semitones=0):
    y = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
    sr = audio_segment.frame_rate
    y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones)
    return AudioSegment(
        y_shifted.tobytes(),
        frame_rate=sr,
        sample_width=audio_segment.sample_width,
        channels=audio_segment.channels
    ).set_frame_rate(audio_segment.frame_rate)


def apply_speed_change(audio_segment, speed=1.0):
    if len(audio_segment) < 150:
        return audio_segment
    return speedup(audio_segment, playback_speed=speed)


def apply_reverb(audio_segment, reverb_level=50):
    audio_segment = low_pass_filter(audio_segment, reverb_level)
    audio_segment = high_pass_filter(audio_segment, reverb_level)
    return audio_segment


def apply_random_changes(audio_segment, input_files):
    input_file = random.choice(input_files)
    input_audio = AudioSegment.from_file(input_file)
    input_audio_segment = input_audio[:len(audio_segment)]

    # Mix the audio_segment with the input audio segment
    mixed_segment = audio_segment.overlay(input_audio_segment)

    # Detect pitch and apply random changes
    detected_pitch = detect_pitch(mixed_segment)
    pitch_change = random.uniform(-2, 2)
    speed_change_factor = random.uniform(0.9, 1.1)
    reverb_change = random.randint(45, 55)

    mixed_segment = change_pitch(mixed_segment, semitones=pitch_change)
    mixed_segment = apply_speed_change(mixed_segment, speed=speed_change_factor)
    mixed_segment = apply_reverb(mixed_segment, reverb_level=reverb_change)

    # Apply additional random changes
    changes = [
        mixed_segment + random.randint(-5, 5),  # Volume change
    ]

    return random.choice(changes)


def manipulate_audio(chunks, input_files):
    logging.info('Applying random changes to audio chunks...')
    manipulated_chunks = []
    for i, chunk in enumerate(chunks):
        logging.info(f'Processing chunk {i + 1}/{len(chunks)}...')
        chunk = apply_random_changes(chunk, input_files)
        manipulated_chunks.append(chunk)
    logging.info('Finished applying random changes to audio chunks.')
    return manipulated_chunks


def combine_audio(chunks, original_audio):
    logging.info('Combining audio chunks into a single file...')
    combined = AudioSegment.empty()
    for i, chunk in enumerate(chunks):
        fade_in_duration = 10  # 10 ms fade in
        fade_out_duration = 10  # 10 ms fade out
        chunk = chunk.fade_in(fade_in_duration).fade_out(fade_out_duration)
        combined += chunk

    # Make sure original audio has higher amplitude
    original_audio = original_audio + 8  # Increase amplitude
    # combined = combined.overlay(original_audio)
    combined = original_audio.overlay(combined)

    logging.info('Finished combining audio chunks.')
    return combined


if __name__ == "__main__":
    youtube_url = 'https://www.youtube.com/watch?v=ts7D-hvTUqg&ab_channel=BillyRaffoulVEVO'  # Replace with your YouTube URL
    output_path = 'D:/audioManipulation-main/downloads'
    os.makedirs(output_path, exist_ok=True)

    downloaded_file = download_audio(youtube_url, output_path)

    input_dir_path = 'D:/audioManipulation-main/Input'
    input_files = [os.path.join(input_dir_path, f) for f in os.listdir(input_dir_path) if f.endswith('.mp3')]

    if downloaded_file:
        original_audio = AudioSegment.from_file(downloaded_file)
        chunks = split_audio(downloaded_file, chunk_length_ms=500)
        manipulated_chunks = manipulate_audio(chunks, input_files)
        combined_audio = combine_audio(manipulated_chunks, original_audio)
        combined_audio.export(os.path.join(output_path, 'manipulated_audio_algo1.wav'), format='wav')
        logging.info(f'Manipulated audio saved successfully: {os.path.join(output_path, "manipulated_audio_algo1.wav")}')
