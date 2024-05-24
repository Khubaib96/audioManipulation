from pytube import YouTube
from pydub import AudioSegment
import os
from tqdm import tqdm
from pydub.utils import which

# Explicitly set the paths to ffmpeg and ffprobe in the environment
os.environ["PATH"] += os.pathsep + "D:\\ffmpeg\\bin"

# Check if ffmpeg and ffprobe are correctly found
ffmpeg_path = which("ffmpeg")
ffprobe_path = which("ffprobe")


def download_audio(youtube_url, output_path='.'):
    try:
        # Create YouTube object
        yt = YouTube(youtube_url)

        # Get the highest quality audio stream available
        audio_stream = yt.streams.filter(only_audio=True).order_by('abr').desc().first()

        # Download the audio stream with a progress bar
        file_size = audio_stream.filesize
        output_file = os.path.join(output_path, audio_stream.default_filename)

        with tqdm(total=file_size, unit='B', unit_scale=True, desc='Downloading') as pbar:
            def progress(chunk, file_handle, bytes_remaining):
                pbar.update(file_size - bytes_remaining)

            yt.register_on_progress_callback(progress)
            audio_stream.download(output_path)

        # Convert to WAV format
        wav_output_file = os.path.splitext(output_file)[0] + '.wav'
        audio = AudioSegment.from_file(output_file)
        audio.export(wav_output_file, format='wav')

        # Remove the original downloaded file
        os.remove(output_file)

        print(f'Audio downloaded and converted to WAV successfully: {wav_output_file}')

    except Exception as e:
        print(f'Error downloading audio: {e}')


if __name__ == "__main__":
    # Example YouTube URL
    youtube_url = 'https://www.youtube.com/watch?v=Nnop2walGmM'

    # Output path where the audio file will be saved
    output_path = './downloads'

    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Call the download_audio function
    download_audio(youtube_url, output_path)
