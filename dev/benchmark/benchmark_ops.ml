[@@@warning "-32"]

open Rune
open Kaun
open Kaun.Layer

(* Common dimensions from makemore *)
let batch_size = 256
let seq_len = 16
let hidden_dim = 64
let vocab_size = 27
let num_heads = 4

(* Initialize dtype *)
let dtype = Rune.float32

(* Helper to create random tensor *)
let randn shape = Rune.randn dtype shape

(* Benchmark matrix multiplication *)
let bench_matmul () =
  let x = randn [| batch_size; seq_len; hidden_dim |] in
  let w = randn [| hidden_dim; hidden_dim |] in
  fun () -> Rune.matmul x w |> ignore

(* Benchmark embedding lookup *)
let bench_embedding () =
  (* Create random indices as float32 tensor with integer values *)
  (* Since randint doesn't work with float32, create random floats and floor them *)
  let indices =
    Rune.rand dtype [| batch_size; seq_len |]
    |> Rune.mul (Rune.scalar dtype (float_of_int vocab_size))
    |> Rune.floor
  in
  (* Create embedding layer *)
  let emb_layer = Layer.embedding ~vocab_size ~embed_dim:hidden_dim () in
  let params = emb_layer.init ~rngs:(Rng.key 42) ~dtype in
  fun () -> apply emb_layer params ~training:false indices |> ignore

(* Benchmark layer normalization *)
let bench_layernorm () =
  let x = randn [| batch_size; seq_len; hidden_dim |] in
  let ln = Layer.layer_norm ~dim:hidden_dim () in
  let params = ln.init ~rngs:(Rng.key 42) ~dtype in
  fun () -> apply ln params ~training:false x |> ignore

(* Benchmark multi-head attention *)
let bench_attention () =
  let x = randn [| batch_size; seq_len; hidden_dim |] in
  let head_dim = hidden_dim / num_heads in

  (* Projection weights *)
  let wq = randn [| hidden_dim; hidden_dim |] in
  let wk = randn [| hidden_dim; hidden_dim |] in
  let wv = randn [| hidden_dim; hidden_dim |] in
  let wo = randn [| hidden_dim; hidden_dim |] in

  (* Create causal mask *)
  let mask =
    let ones = Rune.ones dtype [| seq_len; seq_len |] in
    let upper = Rune.triu ~k:1 ones in
    Rune.mul upper (Rune.scalar dtype (-1e10))
  in

  fun () ->
    (* Project to Q, K, V *)
    let q =
      Rune.matmul x wq
      |> Rune.reshape [| batch_size; seq_len; num_heads; head_dim |]
      |> Rune.transpose ~axes:[| 0; 2; 1; 3 |]
    in
    let k =
      Rune.matmul x wk
      |> Rune.reshape [| batch_size; seq_len; num_heads; head_dim |]
      |> Rune.transpose ~axes:[| 0; 2; 1; 3 |]
    in
    let v =
      Rune.matmul x wv
      |> Rune.reshape [| batch_size; seq_len; num_heads; head_dim |]
      |> Rune.transpose ~axes:[| 0; 2; 1; 3 |]
    in

    (* Attention scores *)
    let scores =
      Rune.matmul q (Rune.transpose ~axes:[| 0; 1; 3; 2 |] k)
      |> Rune.div (Rune.scalar dtype (Float.sqrt (float_of_int head_dim)))
    in

    (* Apply mask *)
    let scores =
      Rune.add scores (Rune.reshape [| 1; 1; seq_len; seq_len |] mask)
    in

    (* Softmax *)
    let attn_weights = Rune.softmax ~axes:[| 3 |] scores in

    (* Apply attention *)
    let out =
      Rune.matmul attn_weights v
      |> Rune.transpose ~axes:[| 0; 2; 1; 3 |]
      |> Rune.contiguous
      |> Rune.reshape [| batch_size; seq_len; hidden_dim |]
    in

    (* Output projection *)
    Rune.matmul out wo |> ignore

(* Benchmark GRU cell *)
let bench_gru_cell () =
  let x = randn [| batch_size; hidden_dim |] in
  let h = randn [| batch_size; hidden_dim |] in

  (* GRU weights *)
  let w_ir = randn [| hidden_dim; hidden_dim |] in
  let w_hr = randn [| hidden_dim; hidden_dim |] in
  let b_r = randn [| hidden_dim |] in

  let w_iz = randn [| hidden_dim; hidden_dim |] in
  let w_hz = randn [| hidden_dim; hidden_dim |] in
  let b_z = randn [| hidden_dim |] in

  let w_in = randn [| hidden_dim; hidden_dim |] in
  let w_hn = randn [| hidden_dim; hidden_dim |] in
  let b_n = randn [| hidden_dim |] in

  fun () ->
    (* Reset gate *)
    let r =
      Rune.add (Rune.matmul x w_ir) (Rune.matmul h w_hr)
      |> Rune.add b_r |> Rune.sigmoid
    in

    (* Update gate *)
    let z =
      Rune.add (Rune.matmul x w_iz) (Rune.matmul h w_hz)
      |> Rune.add b_z |> Rune.sigmoid
    in

    (* New gate *)
    let n =
      Rune.add (Rune.matmul x w_in) (Rune.matmul (Rune.mul r h) w_hn)
      |> Rune.add b_n |> Rune.tanh
    in

    (* Update hidden state *)
    let h_new =
      Rune.add (Rune.mul (Rune.sub (Rune.scalar dtype 1.) z) n) (Rune.mul z h)
    in
    ignore h_new

(* Benchmark GRU sequence *)
let bench_gru_sequence () =
  let x = randn [| batch_size; seq_len; hidden_dim |] in
  let gru =
    Layer.gru ~input_size:hidden_dim ~hidden_size:hidden_dim
      ~return_sequences:true ~learned_init:true ()
  in
  let params = gru.init ~rngs:(Rng.key 42) ~dtype in
  fun () -> apply gru params ~training:false x |> ignore

(* Benchmark transformer block *)
let bench_transformer_block () =
  let x = randn [| batch_size; seq_len; hidden_dim |] in

  (* Create a single transformer decoder layer *)
  let decoder =
    Layer.transformer_decoder ~num_layers:1 ~embed_dim:hidden_dim ~num_heads
      ~mlp_hidden:(4 * hidden_dim) ()
  in

  let params = decoder.init ~rngs:(Rng.key 42) ~dtype in

  fun () -> apply decoder params ~training:false x |> ignore

let () =
  Printf.printf "%s\n" (String.make 60 '=');
  Printf.printf "OCaml/Rune Benchmark - Transformer Operations\n";
  Printf.printf "%s\n" (String.make 60 '=');
  Printf.printf "Configuration:\n";
  Printf.printf "  Batch size: %d\n" batch_size;
  Printf.printf "  Sequence length: %d\n" seq_len;
  Printf.printf "  Hidden dimension: %d\n" hidden_dim;
  Printf.printf "  Vocabulary size: %d\n" vocab_size;
  Printf.printf "  Number of heads: %d\n" num_heads;
  Printf.printf "%s\n" (String.make 60 '-');

  (* Configure benchmarks with 1 second time limit *)
  let config = Ubench.Config.(default |> time_limit 1.0 |> warmup 3 |> build) in

  (* Create benchmarks *)
  let benchmarks =
    [
      Ubench.bench "embedding" (bench_embedding ());
      Ubench.bench "matmul" (bench_matmul ());
      Ubench.bench "layernorm" (bench_layernorm ());
      Ubench.bench "attention" (bench_attention ());
      (* Ubench.bench "gru_cell" (bench_gru_cell ()); Ubench.bench
         "gru_sequence" (bench_gru_sequence ()); Ubench.bench
         "transformer_block" (bench_transformer_block ()); *)
    ]
  in

  (* Run benchmarks *)
  let _results = Ubench.run ~config benchmarks in
  ()
