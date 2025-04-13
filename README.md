# MusicGeneration
exploring models for music generation. pretty_midi takes midi files and we convert to either a csv representation or the "ABC" standard to feed into GPT-2 for fine tuning on our task

## Inputs:
I used MusicNet and a dataset of free piano music. If you can find more, that probably will never hurt.

### Midi to CSV through Pandas DF
It gets combined by the combiner and stored

### Midi to 'ABC'
My peer found a standard called "ABC" for representing music that has some advantages over CSV for a few reasons; time stamps being not absolute but relative instead make it easier to join files and ensure output starts at zero. We don't seem to lose any functionality. I'm also under the impression that the note times being in float format isn't as good for the tokenized representation that most modern Transformer-based LLMs are best at dealing with. This would avoid having to potentially train our own tokenizer, etc, if that would even help.
