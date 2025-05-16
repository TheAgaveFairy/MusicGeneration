# MusicGeneration
Exploring models for music generation, eventually with the goal of the model responding to EEG signals to bring about relaxation. We use a transformer to generate tokenized music, and classify EEG signals for the eventual feedback loop.

## Inputs:
We want something lighter than PCM representations of music (MP3, WAV, FLAC), so we're looking namely at MIDI files. There are a couple of ways to extract information from these files that we focused on; a custom syntax could be generated like ABC if needed.

### Midi to CSV through Pandas DF
Input datasets get loaded into pretty_midi, and "rows" of desired data are extracted. This is saved in .csv, or manipulated as Pandas DataFrames. Outputs are sent to the desired folder to be loaded by our model and further processed for feeding into the GPT-2 tokenizer / model.

### Midi to 'ABC'
My peer found a standard called "ABC" for representing music that has some advantages over CSV for a few reasons; time stamps being not absolute but relative instead make it easier to join files and ensure output starts at zero. We don't seem to lose any functionality. I'm also under the impression that the note times being in float format isn't as good for the tokenized representation that most modern Transformer-based LLMs are best at dealing with. This would avoid having to potentially train our own tokenizer, etc, if that would even help.

While Pandas offered a nice API for quickly representing, augmenting, and handling data, this must now all be done from within the midi format via pretty_midi. I regret not doing this from the beginning, but I was less familiar with the API and midi format so it was easier to do in my own extracted version.

## Model
A simple GPT-2 from HuggingFace is employed. Implementations are available in C of this famous architecture, if needed by some follow-up work. It's a powerful model for its small size and a good architecture for our problem.

### Custom Tokenizer
We played around with a custom tokenizer for another class, with a vocabulary size of 600, 1200, and 3000. Since the ABC notation is small, 600 was the best. The only thing outside of that spec is any title label, e.g. "X: Debussy - Claire de Lune". I'm not sure how to best handle that. These results are not in this GitHub and wouldn't help anyways as they were trained for BERT. The point being, it should be easy to train a custom tokenizer and I think it's probably worthwhile.

## Data
We want the model to generate some notation / tokenization of music. MIDI as ABC (or make your own more compact version of ABC) is a great thing. Then the model can generate ABC, which we convert to MIDI, then synthesize to .WAV or some other PCM format.

The model will eventually want to respond to EEG signals. That means that you probably want to have some kinds of knobs and dials to teach the model to turn. Commonly, ideas such as "valance", "tempo", "energy" are correlated to subjective changes in mood in patients. Ideally, we'd have a labeled dataset with some kind of information like this, then use reinforcement learning to increase the relaxed state of the patient.

It is pretty commonly accepted what a calm EEG looks like, and we can train a simple classifier to judge how the patient is feeling. An approach I saw and have started to implement is known as "differential" entropy, where you use a band-pass filter for the alpha, beta, theta, etc. brain waves and use their relative powers to classify the patient's mood. There are no datasets I found that directly correlate EEG signals to music, but there are some for video stimulus and video games.

### EEG Response to Stressors

This is the really important thing to find...

https://www.kaggle.com/datasets/sigfest/database-for-emotion-recognition-system-gameemo 


### General MIDI Files

https://www.kaggle.com/datasets/soumikrakshit/classical-music-midi 
https://www.kaggle.com/datasets/imsparsh/musicnet-dataset 
https://github.com/lucasnfe/vgmidi - there's TONS of video game MIDI content out there. video game music is typically stressful though, according to a few papers. this seems to pull from some database I can't figure out

GET AS MUCH AS YOU CAN, IMHO!
