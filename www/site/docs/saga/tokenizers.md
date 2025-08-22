# Tokenizers

Saga provides multiple tokenization strategies suitable for different NLP tasks.

## Available Tokenizers

### Word-level Tokenization

The default tokenizer splits on whitespace and punctuation:

```ocaml
let tokens = Saga.tokenize "Hello, world!"
(* ["Hello"; ","; "world"; "!"] *)
```

### Character-level Tokenization

Split text into individual characters:

```ocaml
let tokens = Saga.tokenize ~method_:`Chars "Hi!"
(* ["H"; "i"; "!"] *)
```

### Regex-based Tokenization

Use custom patterns for fine-grained control:

```ocaml
let tokens = Saga.tokenize ~method_:(`Regex "\\w+|[^\\w\\s]+") "don't stop!"
(* ["don"; "'"; "t"; "stop"; "!"] *)
```

## Subword Tokenizers

### BPE (Byte Pair Encoding)

BPE learns merge rules from a corpus and applies them to split words into subword units:

```ocaml
(* Load pretrained BPE model *)
let bpe = Saga.Bpe.from_files 
  ~vocab:"vocab.json" 
  ~merges:"merges.txt" in

(* Tokenize text *)
let tokens = Saga.Bpe.tokenize bpe "Hello world"
(* Returns list of BPE tokens *)

(* Build BPE from scratch *)
let builder = Saga.Bpe.Builder.create () in
let builder = Saga.Bpe.Builder.vocab_and_merges builder vocab merges in
let bpe = Saga.Bpe.Builder.build builder
```

Configuration options:
- `cache_capacity`: Number of words to cache (default: 10000)
- `dropout`: BPE dropout for regularization during training
- `unk_token`: Token for unknown subwords
- `continuing_subword_prefix`: Prefix for non-initial subwords
- `end_of_word_suffix`: Suffix to mark word boundaries

### WordPiece

WordPiece uses a greedy longest-match-first algorithm, commonly used in BERT:

```ocaml
(* Create WordPiece tokenizer *)
let wp = Saga.Wordpiece.from_files ~vocab:"vocab.txt" in

(* Tokenize with WordPiece *)
let tokens = Saga.Wordpiece.tokenize wp "Hello world"
(* ["Hello"; "##world"] with ## prefix for subwords *)

(* Configure WordPiece *)
let config = {
  Saga.Wordpiece.vocab;
  unk_token = "[UNK]";
  max_input_chars_per_word = 100;
  continuing_subword_prefix = "##";
} in
let wp = Saga.Wordpiece.create config
```

## Tokenizer Pipelines

### Composing Tokenizers

Build complex tokenization pipelines:

```ocaml
open Saga.Tokenizer

(* Add normalization *)
let tokenizer = 
  words
  |> with_normalizer (Saga.normalize ~lowercase:true) in

(* Get token offsets for alignment *)
let tokens_with_offsets = run_with_offsets tokenizer text in
(* [("hello", 0, 5); ("world", 6, 11)] *)
```

### Custom Tokenizers

Implement your own tokenization logic:

```ocaml
let custom_tokenizer = Saga.Tokenizer.create
  ~name:"custom"
  ~tokenize:(fun text -> 
    (* Your tokenization logic *)
    String.split_on_char ' ' text
  )
```

## Vocabulary Integration

All tokenizers work seamlessly with vocabularies:

```ocaml
(* Build vocabulary from tokenized text *)
let tokens = Saga.tokenize text in
let vocab = Saga.vocab ~max_size:10000 tokens in

(* Encode using vocabulary *)
let encoded = Saga.encode ~vocab text in

(* Decode back to text *)
let decoded = Saga.decode vocab encoded
```

## Performance Considerations

### Caching

BPE and WordPiece tokenizers cache tokenization results:

```ocaml
(* Configure cache size *)
let bpe = Saga.Bpe.Builder.create ()
  |> Saga.Bpe.Builder.cache_capacity 50000
  |> Saga.Bpe.Builder.build

(* Clear cache when needed *)
Saga.Bpe.clear_cache bpe
```

### Batch Processing

Process multiple texts efficiently:

```ocaml
let texts = ["text1"; "text2"; "text3"] in
let tensor = Saga.encode_batch ~vocab ~max_len:128 texts
(* Processes all texts in parallel *)
```

## Best Practices

1. **Choose the right tokenizer**:
   - Word-level: Simple tasks, small vocabularies
   - BPE: General purpose, handles OOV well
   - WordPiece: BERT compatibility, good for transfer learning

2. **Normalize consistently**:
   ```ocaml
   let normalized = Saga.normalize 
     ~lowercase:true 
     ~strip_accents:true 
     text
   ```

3. **Handle special tokens properly**:
   ```ocaml
   (* Add special tokens to sequences *)
   let with_special = [bos_token] @ tokens @ [eos_token]
   ```

4. **Monitor vocabulary size**:
   ```ocaml
   let vocab = Saga.vocab 
     ~max_size:30000    (* Reasonable for most tasks *)
     ~min_freq:2        (* Filter rare tokens *)
     tokens
   ```