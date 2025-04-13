# MusicGeneration
exploring models for music generation. pretty_midi takes midi files and we convert to either a csv representation or the "ABC" standard to feed into GPT-2 for fine tuning on our task

## Inputs:
I used MusicNet and a dataset of free piano music. If you can find more, that probably will never hurt.

### Midi to CSV through Pandas DF
Input datasets get loaded into pretty_midi, and "rows" of desired data are extracted. This is saved in .csv, or manipulated as Pandas DataFrames. Outputs are sent to the desired folder to be loaded by our model and further processed for feeding into the GPT-2 tokenizer / model.

### Midi to 'ABC'
My peer found a standard called "ABC" for representing music that has some advantages over CSV for a few reasons; time stamps being not absolute but relative instead make it easier to join files and ensure output starts at zero. We don't seem to lose any functionality. I'm also under the impression that the note times being in float format isn't as good for the tokenized representation that most modern Transformer-based LLMs are best at dealing with. This would avoid having to potentially train our own tokenizer, etc, if that would even help.

While Pandas offered a nice API for quickly representing, augmenting, and handling data, this must now all be done from within the midi format via pretty_midi. I regret not doing this from the beginning, but I was less familiar with the API and midi format so it was easier to do in my own extracted version.
