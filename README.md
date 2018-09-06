# Universial Karaoke CNN

When doing karaoke, I often wished to be able to choose freely from all existing songs. Unfortunately, for some songs (especially some of my favorite songs) there exists no actual karaoke version. Therefore I asked myself how to create my own karaoke version.

Frequently, a karaoke version can be created by taking the stereo track of the song and inverting one of the channels. As the main voice usually is evenly broadcasted across both channels, the inversion will cancel out the singer's voice. Sadly, this technique does not work for all songs. Often stereo effects like reverb or background vocals are still in place or the music gets altered as well. So, what if we could create a neural network that could differentiate between voices and background music in any kind of song?

I aimed to build a convolutional neural network (CNN) that can detect voices in the spectogram of a song. Instead of instantly predicting the karaoke version I decided to predict voice. I assume that different voices share more similarities across all songs than the background music. The background music usually changes a lot more depending on the effects, music style or instruments used in each song. If the CNN can successfully separate the voices from the music track, the seperated vocal track can be used to remove the vocals from the original song (by adding the inverted vocal track; or substracting the frequencies in the spectogram).

So far, the model can detect the voice snippets (time-wise) and the core frequencies and remove them from the musical track. The model struggles though when the vocal frequencies overlap with (loud) music parts. Even though the results are not perfect, I think that this project can be a good starting point for someone trying to apply a CNN to audio files.

A demonstration of the model's results can be found in the [demo.ipynb](demo.ipynb). It will additionally explain the structure of the training data and the model setup.

The following instructions will explain how to build the necessary data sets, how to train and how to run the model. All required python modules are listed in the [requirements.txt](requirements.txt).
All program files can be found in the [src folder](src). Before running the python commands below, make sure to navigate to said src folder. The global parameters that change the structure and behaviour of the model have been collected in the [global_settings.py](src/global_settings.py). Note that changes to the model prevent the trained model from loading properly. So, when doing your own experiments make sure to delete my trained model or change the folder where the model is created.

## Data Construction

The training data consists of 2 different types of data sets. The foundation of my data set is the CCMixter voice separation corpus created by Antoine Liutkus. The corpus was created for the publication "Kernel Additive Modelling for source separation" and can be found [here](https://members.loria.fr/ALiutkus/kam/). The data needs to be downloaded manually and has to be placed in the prepared (thus currently empty) [cc_mixter folder](ccmixter_corpus). To add the CCMixter data to the pool of training data, run:

```
python ccmixter_to_train_data.py
```

I decided to extend the CCMixter corpus with vocal only and music only songs from [youtube](https://youtube.com). The uris of the accoring songs can be found [here(vocal)](urls/vocals.csv) and [here(music)](urls/music.csv). To add these songs to the pool of training data, run:

```
python youtube_to_train_data.py
```

If the songs have been successfully added, you can find them in the "input_24k" folder. 

## Build and Run (Pre-)Training

For this project I used two training sets. A pre-training set, that only contains voice data and noise without music, and the main training set, containing voice, music and noise data. 
The pre-training data is "easier" data than the actual training data. It is used to ensure that the model builds an internal representation (similar to an auto-encoder) of voice without the "distraction" of music data. Only afterwards the model is trained on syntetically created song snippets.

To create both training sets, run:

```
python create_datasets.py
```

Be aware that you have to build the pool of training data (input_24k folder) beforehand.

To train the model on the pre-training data (not mandatory; but, as mentioned before, a good starting point), run:
```
python pre_train_model.py
```

To train the model on the actual training data, run:
```
python train_model.py
```
## Test Model

If you want to transform a song to its karaoke version, put your audio file in the \*test_music folder\*/originals (specified in [global_settings.py](src/global_settings.py); default: src/test_music/originals) and run:

```
python test_model.py
```

The predictions will be placed in \*test_music_folder\*/predictions. \*song_name\*_pred.wav files contain the predicted voice, while \*song_name\*_karaoke.wav files contain the karaoke versions.

## Built With

* [Librosa](http://librosa.github.io/) - Audio Processing
* [Tensorflow](https://www.tensorflow.org/) - Machine Learning
* [Matplotlib](https://matplotlib.org/) - Plotting

## License

This project is licensed under the GNU General Public License (GPLv3) - see the [LICENSE.md](LICENSE.md) file for details
