# Brot Examples

Learn Brot through progressively complex examples. Start with `01-encode-decode`
and work through the numbered examples in order.

## Examples

| Example | Concept | Key Functions |
|---------|---------|---------------|
| [`01-encode-decode`](./01-encode-decode/) | Text to IDs and back | `bpe`, `encode`, `decode` |
| [`02-encoding-fields`](./02-encoding-fields/) | Encoding metadata | `Encoding.ids`, `.tokens`, `.offsets` |
| [`03-normalizers`](./03-normalizers/) | Text normalization | `Normalizer.lowercase`, `.bert`, `.sequence` |
| [`04-pre-tokenizers`](./04-pre-tokenizers/) | Splitting before vocab | `Pre_tokenizer.whitespace`, `.bert`, `.sequence` |
| [`05-algorithms`](./05-algorithms/) | Algorithm comparison | `bpe`, `wordpiece`, `unigram`, `word_level`, `chars` |
| [`06-special-tokens`](./06-special-tokens/) | Special tokens and post-processing | `Post_processor.bert`, `.template`, `encode ~pair` |
| [`07-padding-truncation`](./07-padding-truncation/) | Batch preparation | `padding`, `truncation`, `encode_batch` |
| [`08-decoders`](./08-decoders/) | Tokens back to text | `Decoder.wordpiece`, `.bpe`, `.fuse`, `.sequence` |
| [`09-training`](./09-training/) | Train from scratch | `train_bpe`, `train_wordpiece`, `train_unigram` |
| [`10-bert-pipeline`](./10-bert-pipeline/) | Full BERT pipeline | All stages assembled end-to-end |

Advanced:

- [**x-gpt2-tokenizer**](./x-gpt2-tokenizer/): Loading a real GPT-2 tokenizer
  from HuggingFace model files

## Running Examples

All examples can be run with:

```bash
dune exec brot/examples/<name>/main.exe
```

For example:

```bash
dune exec brot/examples/01-encode-decode/main.exe
```

## Quick Reference

### Encode and Decode

```ocaml
open Brot

let tokenizer = bpe ~vocab:[("hello", 0); ...] ~merges:[...] () in
let encoding = encode tokenizer "hello world" in
let ids = Encoding.ids encoding in
let text = decode tokenizer ids
```

### Full Pipeline

```ocaml
let tokenizer =
  wordpiece ~vocab
    ~normalizer:(Normalizer.bert ~lowercase:true ())
    ~pre:(Pre_tokenizer.bert ())
    ~post:(Post_processor.bert ~cls:("[CLS]", 2) ~sep:("[SEP]", 3) ())
    ~decoder:(Decoder.wordpiece ())
    ~specials:(List.map special [ "[CLS]"; "[SEP]"; "[PAD]" ])
    ~pad_token:"[PAD]" ()
```

### Train from Text

```ocaml
let tokenizer =
  train_bpe (`Seq (List.to_seq texts)) ~vocab_size:1000
```
