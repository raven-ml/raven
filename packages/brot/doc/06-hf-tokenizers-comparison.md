# Brot vs. HuggingFace Tokenizers -- A Practical Comparison

This guide explains how Brot relates to Python's [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers), focusing on:

* How core concepts map (tokenizer types, pipeline stages, encoding results)
* Where the APIs feel similar vs. deliberately different
* How to translate common HuggingFace patterns into Brot

If you already use HuggingFace Tokenizers, this should be enough to become productive in Brot quickly.

---

## 1. Big-Picture Differences

| Aspect             | HuggingFace Tokenizers (Python)                      | Brot (OCaml)                                                                  |
| ------------------ | ---------------------------------------------------- | ----------------------------------------------------------------------------- |
| Language           | Python bindings over Rust                            | Native OCaml                                                                  |
| Core type          | `tokenizers.Tokenizer`                               | `Brot.t`                                                                      |
| Encoding result    | `tokenizers.Encoding`                                | `Encoding.t`                                                                  |
| Algorithms         | `BPE`, `WordPiece`, `Unigram`, `WordLevel`           | `Brot.bpe`, `Brot.wordpiece`, `Brot.unigram`, `Brot.word_level`, `Brot.chars` |
| Pipeline stages    | Mutable properties on `Tokenizer` object             | Immutable `~normalizer`, `~pre`, `~post`, `~decoder` args                     |
| Mutability         | Tokenizer is mutable (set properties after creation) | Tokenizer is immutable after creation                                         |
| HuggingFace compat | Native format                                        | Full `tokenizer.json` read/write via `from_file`/`save_pretrained`            |
| Training           | `Trainer` objects passed to `tokenizer.train()`      | `Brot.train_bpe`, `Brot.train_wordpiece`, etc.                                |
| Padding config     | `tokenizer.enable_padding()`                         | `~padding` arg on `encode`/`encode_batch`                                     |
| Truncation config  | `tokenizer.enable_truncation()`                      | `~truncation` arg on `encode`/`encode_batch`                                  |

**Brot semantics to know (read once):**
- Tokenizers are immutable. Pipeline components are set at construction time, not mutated after.
- `from_file` returns `(t, string) result`. Handle errors explicitly.
- Padding and truncation are per-call parameters, not global tokenizer state.
- Special tokens use a record type (`Brot.special`) with explicit control over stripping and normalization.
- `encode` returns `Encoding.t`; use `encode_ids` when you only need the ID array.

---

## 2. Loading Pretrained Tokenizers

### 2.1 From a tokenizer.json file

**HuggingFace**

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer.json")
```

**Brot**

<!-- $MDX skip -->
```ocaml
let tokenizer = Brot.from_file "tokenizer.json" |> Result.get_ok
```

Both read the same `tokenizer.json` format. Brot's `from_file` returns a `result` instead of raising an exception.

### 2.2 From vocabulary and merges files

**HuggingFace**

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE

tokenizer = Tokenizer(BPE.from_file("vocab.json", "merges.txt"))
```

**Brot**

<!-- $MDX skip -->
```ocaml
let tokenizer =
  Brot.from_model_file
    ~vocab:"vocab.json"
    ~merges:"merges.txt"
    ()
```

When `~merges` is omitted, Brot infers WordPiece instead of BPE.

### 2.3 Saving

**HuggingFace**

```python
tokenizer.save("tokenizer.json")
```

**Brot**

<!-- $MDX skip -->
```ocaml
Brot.save_pretrained tokenizer ~path:"./my_tokenizer"
```

`save_pretrained` creates `path/tokenizer.json` in HuggingFace format. Use `to_json` when you need the JSON value directly.

---

## 3. Encoding Text

### 3.1 Basic encoding

**HuggingFace**

```python
output = tokenizer.encode("Hello world!")
output.ids          # [101, 7592, 2088, 999, 102]
output.tokens       # ['[CLS]', 'hello', 'world', '!', '[SEP]']
output.offsets      # [(0, 0), (0, 5), (6, 11), (11, 12), (0, 0)]
output.type_ids     # [0, 0, 0, 0, 0]
output.attention_mask  # [1, 1, 1, 1, 1]
```

**Brot**

<!-- $MDX skip -->
```ocaml
let enc = Brot.encode tokenizer "Hello world!"
let ids   = Encoding.ids enc            (* int array *)
let toks  = Encoding.tokens enc         (* string array *)
let offs  = Encoding.offsets enc        (* (int * int) array *)
let types = Encoding.type_ids enc       (* int array *)
let mask  = Encoding.attention_mask enc (* int array *)
```

### 3.2 IDs only

**HuggingFace**

```python
ids = tokenizer.encode("Hello world!").ids
```

**Brot**

<!-- $MDX skip -->
```ocaml
let ids = Brot.encode_ids tokenizer "Hello world!"
```

`encode_ids` is a shortcut that avoids constructing the full `Encoding.t` when you only need token IDs.

### 3.3 Without special tokens

**HuggingFace**

```python
output = tokenizer.encode("Hello world!", add_special_tokens=False)
```

**Brot**

<!-- $MDX skip -->
```ocaml
let enc = Brot.encode tokenizer ~add_special_tokens:false "Hello world!"
```

---

## 4. Decoding

### 4.1 Basic decoding

**HuggingFace**

```python
text = tokenizer.decode([101, 7592, 2088, 999, 102])
text_clean = tokenizer.decode([101, 7592, 2088, 999, 102], skip_special_tokens=True)
```

**Brot**

<!-- $MDX skip -->
```ocaml
let text = Brot.decode tokenizer [| 101; 7592; 2088; 999; 102 |]
let text_clean =
  Brot.decode tokenizer ~skip_special_tokens:true
    [| 101; 7592; 2088; 999; 102 |]
```

### 4.2 Batch decoding

**HuggingFace**

```python
texts = tokenizer.decode_batch([[101, 7592, 102], [101, 2088, 102]])
```

**Brot**

<!-- $MDX skip -->
```ocaml
let texts =
  Brot.decode_batch tokenizer
    [ [| 101; 7592; 102 |]; [| 101; 2088; 102 |] ]
```

---

## 5. Batch Encoding

**HuggingFace**

```python
outputs = tokenizer.encode_batch(["Hello world!", "How are you?"])
# outputs is a list of Encoding objects
for enc in outputs:
    print(enc.ids)
```

**Brot**

<!-- $MDX skip -->
```ocaml
let encodings =
  Brot.encode_batch tokenizer
    [ "Hello world!"; "How are you?" ]

let () =
  List.iter
    (fun enc ->
      let ids = Encoding.ids enc in
      Array.iter (Printf.printf "%d ") ids;
      print_newline ())
    encodings
```

Both return a list of encoding objects, one per input.

---

## 6. Padding and Truncation

### 6.1 Padding

In HuggingFace, padding is global state on the tokenizer. In Brot, it is a per-call parameter.

**HuggingFace**

```python
tokenizer.enable_padding(
    direction="right",
    pad_id=0,
    pad_token="[PAD]",
    length=128,         # fixed length
)
output = tokenizer.encode("Hello")
# output.attention_mask shows 0s for padding positions
```

**Brot**

<!-- $MDX skip -->
```ocaml
let pad = Brot.padding ~pad_id:0 ~pad_token:"[PAD]" (`Fixed 128)
let enc = Brot.encode tokenizer ~padding:pad "Hello"
(* Encoding.attention_mask enc has 0s for padding positions *)
```

Padding strategies:

| HuggingFace                             | Brot                          |
| --------------------------------------- | ----------------------------- |
| `length=None` (pad to longest in batch) | `` `Batch_longest ``          |
| `length=128` (fixed)                    | `` `Fixed 128 ``              |
| `pad_to_multiple_of=8`                  | `` `To_multiple 8 ``          |
| `direction="left"`                      | `~direction:`Left`            |
| `direction="right"` (default)           | `~direction:`Right` (default) |

### 6.2 Truncation

**HuggingFace**

```python
tokenizer.enable_truncation(max_length=512, direction="right")
output = tokenizer.encode("Very long text ...")
```

**Brot**

<!-- $MDX skip -->
```ocaml
let trunc = Brot.truncation 512
let enc = Brot.encode tokenizer ~truncation:trunc "Very long text ..."
```

Truncation direction defaults to `` `Right `` in both libraries.

### 6.3 Combined padding and truncation

**HuggingFace**

```python
tokenizer.enable_padding(length=512, pad_token="[PAD]", pad_id=0)
tokenizer.enable_truncation(max_length=512)
outputs = tokenizer.encode_batch(texts)
```

**Brot**

<!-- $MDX skip -->
```ocaml
let pad = Brot.padding ~pad_token:"[PAD]" ~pad_id:0 (`Fixed 512)
let trunc = Brot.truncation 512
let encodings =
  Brot.encode_batch tokenizer ~padding:pad ~truncation:trunc texts
```

The key difference: Brot passes these as arguments, so different calls can use different settings without mutating the tokenizer.

---

## 7. Sentence Pairs

**HuggingFace**

```python
# Single pair
output = tokenizer.encode("premise", "hypothesis")
output.type_ids  # [0, 0, 0, 0, 1, 1, 1]  (with BERT post-processor)

# Batch of pairs
outputs = tokenizer.encode_batch([("premise1", "hyp1"), ("premise2", "hyp2")])
```

**Brot**

<!-- $MDX skip -->
```ocaml
(* Single pair *)
let enc = Brot.encode tokenizer ~pair:"hypothesis" "premise"
let type_ids = Encoding.type_ids enc  (* 0s for first, 1s for second *)

(* Batch of pairs *)
let encodings =
  Brot.encode_pairs_batch tokenizer
    [ ("premise1", "hyp1"); ("premise2", "hyp2") ]
```

Brot uses the `~pair` optional argument on `encode` for single pairs and a dedicated `encode_pairs_batch` for batches, instead of overloading the same function with tuples.

---

## 8. Special Tokens

### 8.1 Defining special tokens

**HuggingFace**

```python
from tokenizers import AddedToken

tokenizer.add_special_tokens([
    AddedToken("[CLS]", single_word=False, lstrip=False, rstrip=False),
    AddedToken("[SEP]", single_word=False, lstrip=False, rstrip=False),
    AddedToken("[PAD]", single_word=False, lstrip=False, rstrip=False),
])
```

**Brot**

<!-- $MDX skip -->
```ocaml
let tokenizer =
  Brot.bpe
    ~specials:[
      Brot.special "[CLS]";
      Brot.special "[SEP]";
      Brot.special "[PAD]";
    ]
    ~pad_token:"[PAD]"
    ~bos_token:"[CLS]"
    ~eos_token:"[SEP]"
    ()
```

In HuggingFace, special tokens are added after construction. In Brot, they are part of construction since tokenizers are immutable. The `special` function accepts optional `~single_word`, `~lstrip`, `~rstrip`, and `~normalized` parameters matching `AddedToken`.

### 8.2 Role tokens

**HuggingFace**

```python
tokenizer.pad_token       # "[PAD]"
tokenizer.cls_token       # "[CLS]"
tokenizer.sep_token       # "[SEP]"
tokenizer.unk_token       # "[UNK]"
```

**Brot**

<!-- $MDX skip -->
```ocaml
let pad = Brot.pad_token tokenizer  (* string option *)
let bos = Brot.bos_token tokenizer  (* string option *)
let eos = Brot.eos_token tokenizer  (* string option *)
let unk = Brot.unk_token tokenizer  (* string option *)
```

Brot uses `bos_token`/`eos_token` instead of `cls_token`/`sep_token` since these are model-agnostic roles. They return `option` instead of raising on missing tokens.

### 8.3 Special tokens mask

Both libraries provide a mask distinguishing special tokens from content tokens in the encoding:

**HuggingFace**

```python
output.special_tokens_mask  # [1, 0, 0, 0, 1]
```

**Brot**

<!-- $MDX skip -->
```ocaml
let mask = Encoding.special_tokens_mask enc  (* int array: 1 for special, 0 for content *)
```

---

## 9. Pipeline Components

Both libraries use the same four-stage pipeline: normalizer, pre-tokenizer, post-processor, decoder. The difference is how they are configured.

### 9.1 Normalizer

**HuggingFace**

```python
from tokenizers import normalizers

tokenizer.normalizer = normalizers.Sequence([
    normalizers.NFD(),
    normalizers.StripAccents(),
    normalizers.Lowercase(),
])
```

**Brot**

<!-- $MDX skip -->
```ocaml
let norm =
  Normalizer.sequence
    [ Normalizer.nfd; Normalizer.strip_accents; Normalizer.lowercase ]

let tokenizer = Brot.bpe ~normalizer:norm ()
```

Common normalizers:

| HuggingFace                         | Brot                                       |
| ----------------------------------- | ------------------------------------------ |
| `normalizers.NFC()`                 | `Normalizer.nfc`                           |
| `normalizers.NFD()`                 | `Normalizer.nfd`                           |
| `normalizers.NFKC()`                | `Normalizer.nfkc`                          |
| `normalizers.NFKD()`                | `Normalizer.nfkd`                          |
| `normalizers.Lowercase()`           | `Normalizer.lowercase`                     |
| `normalizers.StripAccents()`        | `Normalizer.strip_accents`                 |
| `normalizers.Strip()`               | `Normalizer.strip ()`                      |
| `normalizers.Replace(pattern, rep)` | `Normalizer.replace ~pattern ~replacement` |
| `normalizers.Prepend(s)`            | `Normalizer.prepend s`                     |
| `normalizers.BertNormalizer()`      | `Normalizer.bert ()`                       |
| `normalizers.ByteLevel()`           | `Normalizer.byte_level ()`                 |
| `normalizers.Sequence([...])`       | `Normalizer.sequence [...]`                |

### 9.2 Pre-tokenizer

**HuggingFace**

```python
from tokenizers import pre_tokenizers

tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
    pre_tokenizers.WhitespaceSplit(),
    pre_tokenizers.Punctuation(),
])
```

**Brot**

<!-- $MDX skip -->
```ocaml
let pre =
  Pre_tokenizer.sequence
    [ Pre_tokenizer.whitespace_split ();
      Pre_tokenizer.punctuation () ]

let tokenizer = Brot.bpe ~pre ()
```

Common pre-tokenizers:

| HuggingFace                            | Brot                                |
| -------------------------------------- | ----------------------------------- |
| `pre_tokenizers.Whitespace()`          | `Pre_tokenizer.whitespace ()`       |
| `pre_tokenizers.WhitespaceSplit()`     | `Pre_tokenizer.whitespace_split ()` |
| `pre_tokenizers.BertPreTokenizer()`    | `Pre_tokenizer.bert ()`             |
| `pre_tokenizers.ByteLevel()`           | `Pre_tokenizer.byte_level ()`       |
| `pre_tokenizers.Punctuation()`         | `Pre_tokenizer.punctuation ()`      |
| `pre_tokenizers.Digits()`              | `Pre_tokenizer.digits ()`           |
| `pre_tokenizers.Metaspace()`           | `Pre_tokenizer.metaspace ()`        |
| `pre_tokenizers.UnicodeScripts()`      | `Pre_tokenizer.unicode_scripts ()`  |
| `pre_tokenizers.CharDelimiterSplit(c)` | `Pre_tokenizer.char_delimiter c`    |
| `pre_tokenizers.Split(pattern, ...)`   | `Pre_tokenizer.split ~pattern ()`   |
| `pre_tokenizers.Sequence([...])`       | `Pre_tokenizer.sequence [...]`      |

### 9.3 Post-processor

**HuggingFace**

```python
from tokenizers import processors

tokenizer.post_processor = processors.BertProcessing(
    sep=("[SEP]", 102),
    cls=("[CLS]", 101),
)
```

**Brot**

<!-- $MDX skip -->
```ocaml
let post =
  Post_processor.bert
    ~sep:("[SEP]", 102)
    ~cls:("[CLS]", 101)
    ()

let tokenizer = Brot.bpe ~post ()
```

Common post-processors:

| HuggingFace                                                   | Brot                                                       |
| ------------------------------------------------------------- | ---------------------------------------------------------- |
| `processors.BertProcessing(sep, cls)`                         | `Post_processor.bert ~sep ~cls ()`                         |
| `processors.RobertaProcessing(sep, cls)`                      | `Post_processor.roberta ~sep ~cls ()`                      |
| `processors.ByteLevel()`                                      | `Post_processor.byte_level ()`                             |
| `processors.TemplateProcessing(single, pair, special_tokens)` | `Post_processor.template ~single ?pair ~special_tokens ()` |
| `processors.Sequence([...])`                                  | `Post_processor.sequence [...]`                            |

### 9.4 Decoder

**HuggingFace**

```python
from tokenizers import decoders

tokenizer.decoder = decoders.WordPiece(prefix="##")
```

**Brot**

<!-- $MDX skip -->
```ocaml
let dec = Decoder.wordpiece ~prefix:"##" ()
let tokenizer = Brot.wordpiece ~decoder:dec ()
```

Common decoders:

| HuggingFace                     | Brot                              |
| ------------------------------- | --------------------------------- |
| `decoders.BPEDecoder(suffix)`   | `Decoder.bpe ~suffix ()`          |
| `decoders.ByteLevel()`          | `Decoder.byte_level ()`           |
| `decoders.ByteFallback()`       | `Decoder.byte_fallback ()`        |
| `decoders.WordPiece(prefix)`    | `Decoder.wordpiece ~prefix ()`    |
| `decoders.Metaspace()`          | `Decoder.metaspace ()`            |
| `decoders.CTC()`                | `Decoder.ctc ()`                  |
| `decoders.Replace(pattern, by)` | `Decoder.replace ~pattern ~by ()` |
| `decoders.Strip()`              | `Decoder.strip ()`                |
| `decoders.Fuse()`               | `Decoder.fuse ()`                 |
| `decoders.Sequence([...])`      | `Decoder.sequence [...]`          |

### 9.5 Inspecting the pipeline

**HuggingFace**

```python
tokenizer.normalizer
tokenizer.pre_tokenizer
tokenizer.post_processor
tokenizer.decoder
```

**Brot**

<!-- $MDX skip -->
```ocaml
let norm = Brot.normalizer tokenizer     (* Normalizer.t option *)
let pre  = Brot.pre_tokenizer tokenizer  (* Pre_tokenizer.t option *)
let post = Brot.post_processor tokenizer (* Post_processor.t option *)
let dec  = Brot.decoder tokenizer        (* Decoder.t option *)
```

Brot returns `option` for each stage, since any stage can be absent.

---

## 10. Training Tokenizers

### 10.1 BPE training

**HuggingFace**

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

tokenizer = Tokenizer(BPE())
trainer = BpeTrainer(
    vocab_size=30000,
    min_frequency=2,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]"],
)
tokenizer.train(["corpus.txt"], trainer)
```

**Brot**

<!-- $MDX skip -->
```ocaml
let tokenizer =
  Brot.train_bpe
    (`Files [ "corpus.txt" ])
    ~vocab_size:30000
    ~min_frequency:2
    ~specials:[
      Brot.special "[UNK]";
      Brot.special "[CLS]";
      Brot.special "[SEP]";
      Brot.special "[PAD]";
    ]
    ~unk_token:"[UNK]"
    ~pad_token:"[PAD]"
```

Brot combines the `Tokenizer` + `Trainer` pattern into a single function call. Training data is passed as `` `Files `` (file paths) or `` `Seq `` (string sequence).

### 10.2 WordPiece training

**HuggingFace**

```python
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer

tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
trainer = WordPieceTrainer(vocab_size=30000, special_tokens=["[UNK]", "[PAD]"])
tokenizer.train(["corpus.txt"], trainer)
```

**Brot**

<!-- $MDX skip -->
```ocaml
let tokenizer =
  Brot.train_wordpiece
    (`Files [ "corpus.txt" ])
    ~vocab_size:30000
    ~unk_token:"[UNK]"
    ~specials:[ Brot.special "[UNK]"; Brot.special "[PAD]" ]
    ~pad_token:"[PAD]"
```

### 10.3 Unigram training

**HuggingFace**

```python
from tokenizers.models import Unigram
from tokenizers.trainers import UnigramTrainer

tokenizer = Tokenizer(Unigram())
trainer = UnigramTrainer(vocab_size=8000, special_tokens=["<unk>", "<pad>"])
tokenizer.train(["corpus.txt"], trainer)
```

**Brot**

<!-- $MDX skip -->
```ocaml
let tokenizer =
  Brot.train_unigram
    (`Files [ "corpus.txt" ])
    ~vocab_size:8000
    ~unk_token:"<unk>"
    ~specials:[ Brot.special "<unk>"; Brot.special "<pad>" ]
    ~pad_token:"<pad>"
```

### 10.4 Training from in-memory data

**HuggingFace**

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

tokenizer = Tokenizer(BPE())
trainer = BpeTrainer(vocab_size=1000)
tokenizer.train_from_iterator(
    ["Hello world", "How are you?", "Hello again"],
    trainer,
)
```

**Brot**

<!-- $MDX skip -->
```ocaml
let texts = [ "Hello world"; "How are you?"; "Hello again" ]
let tokenizer =
  Brot.train_bpe (`Seq (List.to_seq texts)) ~vocab_size:1000
```

### 10.5 Extending an existing tokenizer

**HuggingFace**

```python
# Load, then retrain with more data
tokenizer = Tokenizer.from_file("tokenizer.json")
trainer = BpeTrainer(vocab_size=50000)
tokenizer.train(["more_data.txt"], trainer)
```

**Brot**

<!-- $MDX skip -->
```ocaml
let base = Brot.from_file "tokenizer.json" |> Result.get_ok
let tokenizer =
  Brot.train_bpe ~init:base (`Files [ "more_data.txt" ]) ~vocab_size:50000
```

The `~init` parameter on training functions lets you extend an existing tokenizer with additional data.

---

## 11. Vocabulary Inspection

**HuggingFace**

```python
tokenizer.get_vocab()                 # dict: token -> id
tokenizer.get_vocab_size()            # int
tokenizer.token_to_id("[CLS]")        # int or None
tokenizer.id_to_token(101)            # str or None
```

**Brot**

<!-- $MDX skip -->
```ocaml
let v     = Brot.vocab tokenizer         (* (string * int) list *)
let size  = Brot.vocab_size tokenizer    (* int *)
let id    = Brot.token_to_id tokenizer "[CLS]"  (* int option *)
let token = Brot.id_to_token tokenizer 101       (* string option *)
```

`vocab` returns an association list instead of a dictionary. `token_to_id` and `id_to_token` return `option` instead of nullable values.

---

## 12. Quick Cheat Sheet

| Task                | HuggingFace Tokenizers                                      | Brot                                                             |
| ------------------- | ----------------------------------------------------------- | ---------------------------------------------------------------- |
| Load from file      | `Tokenizer.from_file("tokenizer.json")`                     | `Brot.from_file "tokenizer.json"`                                |
| Save to file        | `tokenizer.save("tokenizer.json")`                          | `Brot.save_pretrained tokenizer ~path:"./out"`                   |
| Encode text         | `tokenizer.encode("Hello")`                                 | `Brot.encode tokenizer "Hello"`                                  |
| Encode IDs only     | `tokenizer.encode("Hello").ids`                             | `Brot.encode_ids tokenizer "Hello"`                              |
| Encode batch        | `tokenizer.encode_batch(["a", "b"])`                        | `Brot.encode_batch tokenizer ["a"; "b"]`                         |
| Encode pair         | `tokenizer.encode("a", "b")`                                | `Brot.encode tokenizer ~pair:"b" "a"`                            |
| Encode pairs batch  | `tokenizer.encode_batch([("a","b"), ...])`                  | `Brot.encode_pairs_batch tokenizer [("a","b"); ...]`             |
| Decode              | `tokenizer.decode(ids)`                                     | `Brot.decode tokenizer ids`                                      |
| Decode batch        | `tokenizer.decode_batch([ids1, ids2])`                      | `Brot.decode_batch tokenizer [ids1; ids2]`                       |
| Get token IDs       | `output.ids`                                                | `Encoding.ids enc`                                               |
| Get tokens          | `output.tokens`                                             | `Encoding.tokens enc`                                            |
| Get attention mask  | `output.attention_mask`                                     | `Encoding.attention_mask enc`                                    |
| Get type IDs        | `output.type_ids`                                           | `Encoding.type_ids enc`                                          |
| Get offsets         | `output.offsets`                                            | `Encoding.offsets enc`                                           |
| Padding             | `tokenizer.enable_padding(length=128)`                      | `Brot.encode tokenizer ~padding:(Brot.padding (`Fixed 128)) ...` |
| Truncation          | `tokenizer.enable_truncation(max_length=512)`               | `Brot.encode tokenizer ~truncation:(Brot.truncation 512) ...`    |
| Vocab size          | `tokenizer.get_vocab_size()`                                | `Brot.vocab_size tokenizer`                                      |
| Token to ID         | `tokenizer.token_to_id("[CLS]")`                            | `Brot.token_to_id tokenizer "[CLS]"`                             |
| ID to token         | `tokenizer.id_to_token(101)`                                | `Brot.id_to_token tokenizer 101`                                 |
| Train BPE           | `tokenizer.train(files, BpeTrainer(...))`                   | `Brot.train_bpe (`Files files) ~vocab_size:30000`                |
| Train WordPiece     | `tokenizer.train(files, WordPieceTrainer(...))`             | `Brot.train_wordpiece (`Files files) ~vocab_size:30000`          |
| Train Unigram       | `tokenizer.train(files, UnigramTrainer(...))`               | `Brot.train_unigram (`Files files) ~vocab_size:8000`             |
| Train from iterator | `tokenizer.train_from_iterator(iter, trainer)`              | `Brot.train_bpe (`Seq seq) ~vocab_size:1000`                     |
| Set normalizer      | `tokenizer.normalizer = normalizers.Lowercase()`            | `Brot.bpe ~normalizer:Normalizer.lowercase ()`                   |
| Set pre-tokenizer   | `tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()`      | `Brot.bpe ~pre:(Pre_tokenizer.byte_level ()) ()`                 |
| Set post-processor  | `tokenizer.post_processor = processors.BertProcessing(...)` | `Brot.bpe ~post:(Post_processor.bert ~sep ~cls ()) ()`           |
| Set decoder         | `tokenizer.decoder = decoders.WordPiece()`                  | `Brot.bpe ~decoder:(Decoder.wordpiece ()) ()`                    |
| Add special tokens  | `tokenizer.add_special_tokens([AddedToken(...)])`           | Pass `~specials:[Brot.special "..."; ...]` at construction       |
