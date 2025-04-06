import marimo

__generated_with = "0.12.4"
app = marimo.App(width="medium")


@app.cell
def _():

    import librosa
    import librosa.display
    import numpy as np
    import seaborn as sns
    import marimo as mo
    import pandas as pd



    return librosa, mo, np, pd, sns


@app.cell
def _(librosa):
    # Load the audio file
    audio_file = 'test-sample.wav'
    y, sr = librosa.load(audio_file)

    return audio_file, sr, y


@app.cell
def _(librosa, y):

    # Convert the audio to a spectrogram or mel-spectrogram representation
    spectrogram = librosa.stft(y)

    return (spectrogram,)


@app.cell
def _(spectrogram):
    len(spectrogram)
    return


@app.cell
def _(spectrogram):
    unknown_len = len(spectrogram) 
    return (unknown_len,)


@app.cell
def _(mo, unknown_len):
    mo.md(
        f'''
            what is the significance of **{unknown_len}** being the length
    
        '''
    
    )
    return


@app.cell
def _():
    # current size n == 1025 and m == unknown


    return


@app.cell
def _(pd, spectrogram):
    wav_encoding = pd.DataFrame(spectrogram)
    return (wav_encoding,)


@app.cell
def _():
    return


@app.cell
def _(df, sns):
    sns.lineplot(x='x', y='y', data=df)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _(librosa, sns, sr, y):
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    sns
    return (mel_spectrogram,)


@app.cell
def _(librosa, np, sr, y):

    # Normalize the audio data
    y_normalized = y / np.max(np.abs(y))

    # Extract features
    pitch = librosa.feature.spectral_centroid(y=y, sr=sr)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    rhythm = librosa.feature.rms(y=y)
    return beat_frames, pitch, rhythm, tempo, y_normalized


@app.cell
def _(librosa, np, sr, y):

    # Normalize the audio data
    y_normalized = y / np.max(np.abs(y))

    # Extract features
    pitch = librosa.feature.spectral_centroid(y=y, sr=sr)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    rhythm = librosa.feature.rms(y=y)
    return beat_frames, pitch, rhythm, tempo, y_normalized


if __name__ == "__main__":
    app.run()
