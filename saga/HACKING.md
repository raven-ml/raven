# Saga Developer Guide

## Architecture

Saga is a tokenization and text processing library for ML, providing fast tokenization (BPE, WordPiece), statistical language models, and text generation with composable sampling strategies.

### Core Components

- **[lib/saga.ml](lib/saga.ml)**: High-level tokenization API
- **[lib/tokenizers/](lib/tokenizers/)**: Tokenizer implementations (BPE, WordPiece, etc.)
- **[lib/lm.ml](lib/lm.ml)**: Statistical language models (n-grams)
- **[lib/sampler.ml](lib/sampler.ml)**: Advanced sampling and generation
- **[lib/io.ml](lib/io.ml)**: Efficient file I/O for text corpora

### Key Design Principles

1. **Simple API for common cases**: Direct functions with sensible defaults
2. **Composable processors**: Sampling strategies compose via function chaining
3. **Unicode-aware**: Proper text normalization and CJK handling
4. **Performance**: Efficient tokenization using linked structures and caching
5. **Nx integration**: Direct encoding to tensors for ML pipelines

## Tokenization

### Tokenizer Types

**Built-in methods:**
- `Words`: Whitespace + punctuation splitting (default)
- `Chars`: Character-level tokenization with Unicode awareness
- `Regex pat`: Custom regex pattern
- `BPE`: Byte Pair Encoding (subword tokenization)
- `WordPiece`: WordPiece algorithm (BERT-style)

### BPE Algorithm

BPE learns subword vocabulary through iterative merging:

**Training:**
1. Start with character vocabulary
2. Find most frequent byte pair in corpus
3. Merge pair into new token
4. Repeat for N merges

**Encoding (via linked list):**

```ocaml
type symbol = {
  mutable c : int;        (* Token ID *)
  mutable prev : int;     (* Previous symbol index *)
  mutable next : int;     (* Next symbol index *)
  mutable len : int;      (* Byte length *)
}

type word = {
  symbols : symbol array;
  mutable size : int;
}
```

**Why linked structure?**
- Merging: Update pointers, no array shifts
- Cache-friendly: Symbols stored contiguously
- O(1) merge operations

**Encoding process:**

```ocaml
let encode_word bpe word =
  (* 1. Initialize symbols from characters *)
  let w = init_word word in

  (* 2. Apply merges in order *)
  List.iter (fun (pair, new_token) ->
    merge_pair w pair new_token
  ) bpe.merges;

  (* 3. Extract tokens from linked list *)
  traverse_symbols w
```

**Caching:**
- Cache encoded words
- LRU eviction when full
- Massive speedup for repeated words

### WordPiece

Similar to BPE but:
- Uses likelihood-based scoring for merges
- Subword prefix: `##` for continuations
- Greedy longest-match encoding

### Vocabulary Management

```ocaml
type vocab = (string, int) Hashtbl.t  (* token -> id *)
type vocab_r = (int, string) Hashtbl.t  (* id -> token *)
```

**Building vocabulary:**

```ocaml
let build_vocab ~max_size ~min_freq tokens =
  (* 1. Count token frequencies *)
  let freqs = count_frequencies tokens in

  (* 2. Sort by frequency *)
  let sorted = sort_by_freq freqs in

  (* 3. Take top max_size tokens above min_freq *)
  let vocab_tokens = List.filter (fun (tok, freq) ->
    freq >= min_freq
  ) sorted |> List.take max_size in

  (* 4. Assign IDs *)
  assign_ids vocab_tokens
```

**Special tokens:**
- `<unk>`: Unknown words (ID 0)
- `<pad>`: Padding (ID 1)
- `<bos>`: Beginning of sequence
- `<eos>`: End of sequence

## Language Models

### N-gram Models

Statistical models using n-gram frequencies:

```ocaml
type 'a ngram_model = {
  n : int;                          (* N-gram order *)
  counts : ('a list, int) Hashtbl.t; (* N-gram counts *)
  context_counts : ('a list, int) Hashtbl.t; (* (N-1)-gram counts *)
  vocab : 'a list;                  (* Vocabulary *)
  smoothing : float;                (* Laplace smoothing *)
}
```

**Training:**

```ocaml
let train_ngram ~n texts =
  (* 1. Tokenize texts *)
  let tokenized = List.map tokenize texts in

  (* 2. Extract n-grams *)
  let ngrams = List.concat_map (extract_ngrams ~n) tokenized in

  (* 3. Count n-grams and contexts *)
  let counts = count_ngrams ngrams in
  let context_counts = count_ngrams (List.map (fun ng ->
    List.take (n-1) ng
  ) ngrams) in

  {n; counts; context_counts; vocab; smoothing}
```

**Probability estimation:**

```ocaml
(* P(w_n | w_1...w_{n-1}) with smoothing *)
let prob model context word =
  let ngram = context @ [word] in
  let ngram_count = Hashtbl.find_opt model.counts ngram in
  let context_count = Hashtbl.find_opt model.context_counts context in

  match ngram_count, context_count with
  | Some c_ngram, Some c_ctx ->
      (* Laplace smoothing *)
      float (c_ngram + 1) /. float (c_ctx + List.length model.vocab)
  | _ ->
      (* Unseen ngram: uniform smoothing *)
      model.smoothing /. float (List.length model.vocab)
```

### Perplexity

Measure of model quality:

```ocaml
let perplexity model text =
  let tokens = tokenize text in
  let log_prob = sum (List.map (fun ngram ->
    log (prob model (context_of ngram) (last ngram))
  ) (extract_ngrams ~n:model.n tokens)) in

  exp (-. log_prob /. float (List.length tokens))
```

Lower perplexity = better model.

## Text Generation

### Sampling Strategies

**Temperature scaling:**

```ocaml
let apply_temperature logits temp =
  Array.map (fun x -> x /. temp) logits
```

- `temp < 1`: Sharper distribution (more deterministic)
- `temp > 1`: Flatter distribution (more random)

**Top-k sampling:**

```ocaml
let top_k logits k =
  (* 1. Find top k logits *)
  let sorted = Array.to_list logits |> List.sort compare |> List.rev in
  let threshold = List.nth sorted k in

  (* 2. Mask others to -inf *)
  Array.map (fun x -> if x >= threshold then x else neg_infinity) logits
```

**Top-p (nucleus) sampling:**

```ocaml
let top_p logits p =
  (* 1. Sort by probability *)
  let probs = softmax logits in
  let sorted_indices = argsort probs in

  (* 2. Find cumulative mass threshold *)
  let rec find_threshold cumsum i =
    if cumsum >= p then i
    else find_threshold (cumsum +. probs.(sorted_indices.(i))) (i+1)
  in
  let threshold_idx = find_threshold 0. 0 in

  (* 3. Mask low-probability tokens *)
  ...
```

**Repetition penalty:**

```ocaml
let apply_repetition_penalty logits prev_tokens penalty =
  (* Divide logits of previously generated tokens *)
  List.iter (fun token_id ->
    logits.(token_id) <- logits.(token_id) /. penalty
  ) prev_tokens;
  logits
```

### Composable Processors

Processors transform logits:

```ocaml
type processor = float array -> float array

let compose p1 p2 logits =
  logits |> p1 |> p2

(* Build generation config *)
let config =
  identity
  |> with_temperature 0.9
  |> with_top_k 40
  |> with_repetition_penalty 1.1
```

**Generation loop:**

```ocaml
let generate ~model ~processors ~max_tokens =
  let rec loop tokens =
    if List.length tokens >= max_tokens then tokens
    else
      (* 1. Get logits from model *)
      let logits = model (Array.of_list tokens) in

      (* 2. Apply processors *)
      let processed = processors logits in

      (* 3. Sample next token *)
      let next_token = sample_categorical (softmax processed) in

      (* 4. Continue *)
      loop (tokens @ [next_token])
  in
  loop initial_tokens
```

## Development Workflow

### Building and Testing

```bash
# Build saga
dune build saga/

# Run tests
dune build saga/test/test_saga.exe && _build/default/saga/test/test_saga.exe

# Run examples
dune exec saga/example/ngram_model.exe
```

### Testing Tokenizers

```ocaml
let test_bpe () =
  (* Train BPE *)
  let texts = ["hello world"; "hello there"] in
  let bpe = train_bpe texts ~vocab_size:100 in

  (* Test encoding *)
  let tokens = encode_bpe bpe "hello" in
  (* Verify tokens *)
  Alcotest.(check (list string)) "tokens" expected tokens;

  (* Test roundtrip *)
  let decoded = decode_bpe bpe tokens in
  Alcotest.(check string) "roundtrip" "hello" decoded
```

### Testing Language Models

```ocaml
let test_ngram_probabilities () =
  let model = train_ngram ~n:2 ["the cat"; "the dog"] in

  (* Test probability *)
  let p = prob model ["the"] "cat" in
  Alcotest.(check (float 1e-6)) "p(cat|the)" 0.5 p;

  (* Test perplexity *)
  let pp = perplexity model "the cat" in
  assert (pp > 0. && pp < infinity)
```

## Adding Features

### New Tokenizer

Implement tokenizer interface:

```ocaml
module My_Tokenizer = struct
  type t = {
    (* tokenizer state *)
  }

  let create config = ...

  let encode t text =
    (* Return token IDs *)
    ...

  let decode t ids =
    (* Return text *)
    ...
end

(* Register in saga.ml *)
let tokenize ?(method_=`Words) text =
  match method_ with
  | `My_Tokenizer -> My_Tokenizer.tokenize text
  | ...
```

### New Sampling Strategy

Add processor:

```ocaml
let with_min_p threshold config =
  let processor logits =
    let probs = softmax logits in
    let max_prob = Array.fold_left max neg_infinity probs in
    Array.mapi (fun i p ->
      if p < threshold *. max_prob then neg_infinity else logits.(i)
    ) probs
  in
  compose_processor config processor
```

## Common Pitfalls

### Unicode Handling

Always use Unicode-aware functions:

```ocaml
(* Wrong: byte-level splitting *)
let chars = String.to_seq text |> List.of_seq

(* Correct: Unicode grapheme clusters *)
let chars = Unicode.graphemes text
```

### BPE Cache Invalidation

Cache uses word strings as keys:

```ocaml
(* Wrong: different normalization breaks cache *)
let t1 = encode bpe "Hello" in  (* Cached as "Hello" *)
let t2 = encode bpe "hello" in  (* Cache miss! *)

(* Correct: normalize before encoding *)
let normalize_and_encode bpe text =
  let normalized = String.lowercase_ascii text in
  encode bpe normalized  (* Cache hits *)
```

### Vocabulary OOV

Handle unknown tokens:

```ocaml
let encode_token vocab token =
  match Hashtbl.find_opt vocab token with
  | Some id -> id
  | None ->
      (* Return <unk> token ID *)
      Hashtbl.find vocab "<unk>"
```

### Sampling Degeneracy

Temperature 0 or top-k 1 = greedy:

```ocaml
(* Greedy decoding *)
let greedy logits =
  Array.fold_left_map (fun i x ->
    (i+1, if i = argmax logits then x else neg_infinity)
  ) 0 logits |> snd
```

## Performance

- **BPE caching**: Cache encoded words (major speedup)
- **Batch encoding**: Encode multiple texts in parallel
- **Vocabulary hash tables**: O(1) token lookups
- **Lazy tokenization**: Don't tokenize until needed
- **Tensor batching**: Use Nx for batch encoding to tensors

## Code Style

- **Optional parameters**: `?method_`, `?max_size`, `?min_freq`
- **Labeled arguments**: `~vocab`, `~n`, `~smoothing`
- **Errors**: `"function_name: error description"`
- **Documentation**: Explain algorithm choices and tradeoffs

## Related Documentation

- [CLAUDE.md](../CLAUDE.md): Project-wide conventions
- [README.md](README.md): User-facing documentation
- [nx/HACKING.md](../nx/HACKING.md): Tensor encoding
- Hugging Face Tokenizers documentation for algorithm reference
