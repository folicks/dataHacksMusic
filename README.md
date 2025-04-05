# dataHacksMusic
TODO :
- [ ]  document the tech support
- [ ] read in .wav files to ml data friendly format
- [ ]  get the genius words to match with the corresponding music
- [ ] goal get the words entered in the html(for now) to be entered as parameters for the model see below
- [ ]  use deep learning and reinforcement learning algorithms for generating songs, images, drawings, and other materials

#### Courtesy of the AI of how to make ai

Audio Preprocessing: First, you'll need to preprocess the .wav file to extract relevant features that can be used by the model. This may include:
Converting the audio to a spectrogram or mel-spectrogram representation
Normalizing the audio data
Extracting features such as pitch, tempo, and rhythm
Model Selection: Choose a suitable music generation model that can take the preprocessed audio features and generate a predictive version of the music. Some popular options include:
Generative Adversarial Networks (GANs)
Variational Autoencoders (VAEs)
Recurrent Neural Networks (RNNs)
Transformers
Model Training: Train the selected model on a dataset of music files, including the preprocessed .wav file. You can use a dataset of songs with similar styles or genres to the one you want to generate.
Text-to-Music Generation: Once the model is trained, you can use it to generate music based on the provided words. This may involve:
Converting the text into a numerical representation using techniques such as word embeddings or one-hot encoding
Passing the numerical representation through the model to generate a musical output
Post-processing the generated music to refine the output and ensure it meets your requirements
Song Generation: Use the generated music and provided words to create a song. You can use techniques such as:
Melody generation: use the model to generate a melody that fits the provided words and music
Lyric generation: use a separate model or algorithm to generate lyrics that fit the provided words and music
Music arrangement: use the generated melody and lyrics to create a full song arrangement, including harmony, rhythm, and instrumentation
Some popular tools and libraries for music generation and audio processing include:

Librosa: a Python library for audio signal processing
Music21: a Python library for music theory and analysis
Magenta: a TensorFlow-based library for music generation
Amper Music: a music generation platform that uses AI to create custom music tracks
