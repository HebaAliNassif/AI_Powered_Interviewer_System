#Reference:
#https://www.thepythoncode.com/article/using-speech-recognition-to-convert-speech-to-text-python


from distutils import extension
import os
import sys
import speech_recognition as sr
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from pydub.silence import split_on_silence

# a function that splits the audio file into chunks and applies speech recognition
def get_large_audio_transcription(path):
    """
    Splitting the large audio file into chunks
    and apply speech recognition on each of these chunks
    """
    # create a speech recognition object
    r = sr.Recognizer()
    # open the audio file using pydub
    sound = AudioSegment.from_wav(path)  
    # split audio sound where silence is 700 miliseconds or more and get chunks
    chunks = split_on_silence(sound,
        # experiment with this value for your target audio file
        min_silence_len = 500,
        # adjust this per requirement
        silence_thresh = sound.dBFS-14,
        # keep the silence for 1 second, adjustable as well
        keep_silence=500,
    )
    folder_name = "audio-chunks"
    # create a directory to store the audio chunks
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    whole_text = ""
    # process each chunk 
    for i, audio_chunk in enumerate(chunks, start=1):
        # export audio chunk and save it in
        # the `folder_name` directory.
        chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
        audio_chunk.export(chunk_filename, format="wav")
        # recognize the chunk
        with sr.AudioFile(chunk_filename) as source:
            audio_listened = r.record(source)
            # try converting it to text
            try:
                text = r.recognize_google(audio_listened)
            except sr.UnknownValueError as e:
                text = "."
                whole_text += text
                #print("Error:", str(e))
            else:
                text = f"{text.capitalize()}. "
                #print(chunk_filename, ":", text)
                whole_text += text
    # return the text for all chunks detected
    return whole_text
    
#a function that convert mp4 to wav
def convert_video_to_audio_moviepy(video_file, output_ext="wav"):
    filename, ext = os.path.splitext(video_file)
    clip = VideoFileClip(video_file)
    clip.audio.write_audiofile(f"{filename}.{output_ext}")

def test():
    file_path=input('Please enter the mp4 file path: ')
    filename, exten = os.path.splitext(file_path)
    while exten !=".mp4":
        print(exten)
        file_path=input('Please enter the mp4 file path correctly: ')
        filename, exten = os.path.splitext(file_path)
    
    convert_video_to_audio_moviepy(file_path)
    path = filename+".wav"
    print("Converting speech to text....")
    text=get_large_audio_transcription(path)
    print("Conversion complete check the SpeechToTextOutput.txt file")
    f= open("SpeechToTextOutput.txt","w+")
    f.write(text)
    f.close()    
