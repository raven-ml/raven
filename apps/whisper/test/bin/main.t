The command line writes a transcript of each file in every format, named after
the file, and prints each segment as its window is decided.

  $ mkdir ranks; cp ../upstream/gpt2-1000.tiktoken ranks/gpt2.tiktoken
  $ M="--model ../upstream/transcript.pt --tokenizer ranks"
  $ cp ../librispeech/7902-96594-0025.flac talk.flac
  $ ../../bin/main.exe $M --seed 0 -o out talk.flac
  [00:00.000 --> 00:00.260]  used
  $ ls out
  talk.json
  talk.srt
  talk.tsv
  talk.txt
  talk.vtt
  $ cat out/talk.txt
  used
  $ cat out/talk.vtt
  WEBVTT
  
  00:00.000 --> 00:00.260
  used
  
  $ cat out/talk.srt
  1
  00:00:00,000 --> 00:00:00,260
  used
  
  $ cat out/talk.tsv
  start	end	text
  0	260	used
  $ cat out/talk.json
  {"text": " used\u0015\u0015\u0015\u0015\u0015\u0015", "segments": [{"id": 0, "seek": 0, "start": 0.0, "end": 0.26, "text": " used\u0015\u0015\u0015\u0015\u0015\u0015", "tokens": [1120, 973, 209, 209, 209, 209, 209, 209], "temperature": 0.0, "avg_logprob": -0.6695048544141982, "compression_ratio": 0.5555555555555556, "no_speech_prob": 0.000299664621707052}], "language": "en"}

One format, compiled on the host:

  $ ../../bin/main.exe $M --seed 0 --device cpu-jit --verbose false -f srt -o jit talk.flac
  $ ls jit
  talk.srt

Greedy decoding alone, without a ladder:

  $ ../../bin/main.exe $M --beam_size None --temperature_increment_on_fallback None --verbose false -f txt -o greedy talk.flac
  $ cat greedy/talk.txt
  used used

A file it cannot read is skipped, and the command fails.

  $ ../../bin/main.exe $M --verbose false -o out missing.flac talk.flac
  Skipping missing.flac due to missing.flac: No such file or directory
  [1]

Options it refuses:

  $ ../../bin/main.exe talk.flac
  main.exe: --model is required
  [2]
  $ ../../bin/main.exe $M -f doc talk.flac
  main.exe: wrong argument 'doc'; option '-f' expects one of: all txt vtt srt tsv json.
  main.exe [OPTION...] AUDIO...
  Options:
    --model FILE                          a checkpoint, as tiny.en.pt
    --tokenizer DIR                       the directory of gpt2.tiktoken, an English-only model's rank file, and multilingual.tiktoken, the others'
    --output_dir DIR                      where to write (.)
    -o DIR                                --output_dir
    --output_format {all|txt|vtt|srt|tsv|json}
                                          the format, every one by default
    -f {all|txt|vtt|srt|tsv|json}
                                          --output_format
    --verbose BOOL                        print each segment as it is decided (true)
    --device D                            cpu (eager, the default), cpu-jit (compiled on the host) or a GPU by name, compiled: cuda, nv, amd or metal
    --task {transcribe|translate}
                                          transcribe, or translate into English
    --language CODE                       the spoken language, detected by default
    --temperature T                       the first temperature (0)
    --temperature_increment_on_fallback T the temperatures' step to 1, or None (0.2)
    --best_of N                           draws at a positive temperature (5)
    --beam_size N                         beams at temperature 0, or None for greedy (5)
    --patience P                          a beam's patience (None: 1)
    --length_penalty A                    candidates over ((5 + length) / 6) ** A (None: over their length)
    --suppress_tokens IDS                 ids to suppress, comma-separated; -1 for the non-speech ones (-1)
    --initial_prompt TEXT                 text before the recording
    --carry_initial_prompt BOOL           begin every window's prompt with it (false)
    --condition_on_previous_text BOOL     prompt each window with the transcript before it (true)
    --compression_ratio_threshold X       fall back above this compression ratio, or None (2.4)
    --logprob_threshold X                 fall back below this average log probability, or None (-1)
    --no_speech_threshold X               silence above this no-speech probability, or None (0.6)
    --clip_timestamps S                   start,end,start,end,... in seconds, the last end the file's (0)
    --seed N                              the draws' key, from the system's entropy by default
    -help                                 Display this list of options
    --help                                Display this list of options
  [2]
  $ ../../bin/main.exe $M --clip_timestamps 5,1 talk.flac
  Skipping talk.flac due to Whisper.Transcript: clip (500, 100) after a clip ending at 0
  [1]

A rank file that does not fit the model. The directory ends in /, so that the
file's name, joined to it, reads the same on Windows:

  $ mkdir short; head -n 500 ranks/gpt2.tiktoken > short/gpt2.tiktoken
  $ ../../bin/main.exe --model ../upstream/transcript.pt --tokenizer short/ talk.flac
  Fatal error: exception Failure("short/gpt2.tiktoken: its ranks leave 599 languages in the model's 2608 tokens, not 99")
  [2]
