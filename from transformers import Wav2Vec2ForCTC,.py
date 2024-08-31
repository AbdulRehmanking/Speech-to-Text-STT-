from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa
import torch

# Load pre-trained model and processor
model_name = "facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

def transcribe(audio_file):
    # Load and resample audio to 16000 Hz
    speech, _ = librosa.load(audio_file, sr=16000)  # Ensure resampling is enforced
    # Convert to tensor and process through the model
    input_values = processor(speech, return_tensors="pt", sampling_rate=16000).input_values
    # Perform inference
    with torch.no_grad():  # Ensuring no gradient calculations
        logits = model(input_values).logits
    # Decode the predicted ids to the final transcription
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])

    return transcription

# Usage
audio_file = '/content/OSR_us_000_0010_8k.wav'  # Make sure the path is correct and accessible
transcription = transcribe(audio_file)
print(transcription)
# Install necessary libraries


# Import libraries
from moviepy.editor import VideoFileClip
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Load pre-trained model and processor
model_name = "facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

# Function to check file type and process accordingly
def process_file(input_file):
    if input_file.endswith('.mp4'):  # Assuming video files are .mp4
        output_audio_file = input_file.replace('.mp4', '_audio.wav')
        extract_audio(input_file, output_audio_file)
        return output_audio_file
    elif input_file.endswith('.wav'):  # Assuming audio files are .wav
        return input_file
    else:
        raise ValueError("Unsupported file format. Please use MP4 for video or WAV for audio.")

# Function to extract audio from video
def extract_audio(video_file, output_audio_file):
    video = VideoFileClip(video_file)
    video.audio.write_audiofile(output_audio_file, codec='pcm_s16le')  # Writing as 16-bit PCM WAV
    video.close()

# Function to transcribe audio to text
def transcribe(audio_file):
    # Load and resample audio to 16000 Hz
    speech, _ = librosa.load(audio_file, sr=16000)  # Ensure resampling is enforced
    # Convert to tensor and process through the model
    input_values = processor(speech, return_tensors="pt", sampling_rate=16000).input_values
    # Perform inference
    with torch.no_grad():  # Ensuring no gradient calculations
        logits = model(input_values).logits
    # Decode the predicted ids to the final transcription
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription

# Specify your file here (audio or video)
input_file = '/content/file_example_MP4_480_1_5MG.mp4'  # Adjust path as necessary

# Process the file (extract audio if video)
processed_audio_file = process_file(input_file)

# Transcribe the processed audio file
transcription = transcribe(processed_audio_file)
print("Transcription:\n", transcription)
import os
import zipfile

def zip_files_in_directory(directory_path, output_zip_path):
    # Create a ZipFile object
    with zipfile.ZipFile(output_zip_path, 'w') as zipf:
        # Walk through the directory
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                # Create complete filepath of file in directory
                filepath = os.path.join(root, file)
                # Add file to zip
                zipf.write(filepath, arcname=os.path.relpath(filepath, directory_path))

# Specify the directory to zip and the output zip file path
directory_to_zip = '/content'  # Adjust this path to your directory of work
output_zip_file = '/content/my_work_archive.zip'  # The path where the zip file will be saved

# Call the function to zip the directory
zip_files_in_directory(directory_to_zip, output_zip_file)

print(f'Files zipped successfully into: {output_zip_file}')
