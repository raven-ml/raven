(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* What Llama's walk visits: its leaf paths and reports, with a tied and an
   untied head. *)

open Windtrap

let cfg ~tied : Nx.host Llama.config =
  {
    vocab_size = 11;
    dim = 8;
    n_layers = 2;
    n_heads = 2;
    n_kv_heads = 1;
    head_dim = 4;
    hidden_dim = 16;
    norm_eps = 1e-5;
    rope = Kaun.Rope.make ~head_dim:4 ~context:64 ();
    tied;
  }

let visits s x =
  List.map (Format.asprintf "%a" Nx.Ptree.pp_visit) (Nx.Ptree.visits s x)

let leaf p = p ^ ": a leaf"
let linear p = [ leaf (p ^ ".w"); p ^ ".b: None" ]

let block i =
  let at s = Printf.sprintf "blocks.%d.%s" i s in
  List.concat
    [
      [ leaf (at "attn_norm.gamma") ];
      linear (at "attn.q");
      linear (at "attn.k");
      linear (at "attn.v");
      linear (at "attn.out");
      [ leaf (at "ffn_norm.gamma") ];
      linear (at "gate");
      linear (at "up");
      linear (at "down");
    ]

let body =
  List.concat
    [
      [ leaf "tok.table"; "blocks: length 2" ];
      block 0;
      block 1;
      [ leaf "norm.gamma" ];
    ]

let test_params () =
  let params = Nx.Ptree.instantiate (module Llama.Params) in
  equal ~msg:"a tied head" (list string) (body @ [ "head: None" ])
    (visits params (Llama.make ~key:(Nx.Rng.key 0) (cfg ~tied:true)));
  equal ~msg:"an untied head" (list string)
    (body @ ("head: Some" :: linear "head"))
    (visits params (Llama.make ~key:(Nx.Rng.key 0) (cfg ~tied:false)))

(* GGUF files *)

let le n bytes =
  String.init bytes (fun i -> Char.chr ((n lsr (8 * i)) land 0xff))

let u64 n = le n 8
let gstring s = u64 (String.length s) ^ s
let align n = (n + 31) / 32 * 32

(* Metadata values by the specification's table of types. *)
let kv_u32 key n = gstring key ^ le 4 4 ^ le n 4

let kv_f32 key x =
  gstring key ^ le 6 4 ^ le (Int32.to_int (Int32.bits_of_float x)) 4

let kv_string key s = gstring key ^ le 8 4 ^ gstring s

(* A GGUF file of [kvs] and [tensors], each a name, a type tag, a logical shape
   and its bytes, every data start aligned to 32. *)
let gguf_file kvs tensors =
  let offsets, _ =
    List.fold_left
      (fun (offs, next) (_, _, _, data) ->
        (align next :: offs, align next + String.length data))
      ([], 0) tensors
  in
  let offsets = List.rev offsets in
  let info (name, tag, shape, _) off =
    gstring name
    ^ le (Array.length shape) 4
    ^ String.concat "" (List.rev_map u64 (Array.to_list shape))
    ^ le tag 4 ^ u64 off
  in
  let header =
    "GGUF" ^ le 3 4
    ^ u64 (List.length tensors)
    ^ u64 (List.length kvs)
    ^ String.concat "" kvs
    ^ String.concat "" (List.map2 info tensors offsets)
  in
  let data = Buffer.create 4096 in
  List.iter2
    (fun (_, _, _, bytes) off ->
      Buffer.add_string data (String.make (off - Buffer.length data) '\000');
      Buffer.add_string data bytes)
    tensors offsets;
  let start = align (String.length header) in
  header
  ^ String.make (start - String.length header) '\000'
  ^ Buffer.contents data

let with_gguf kvs tensors f =
  let path = Filename.temp_file "llama" ".gguf" in
  Fun.protect
    ~finally:(fun () -> Sys.remove path)
    (fun () ->
      Out_channel.with_open_bin path (fun oc ->
          output_string oc (gguf_file kvs tensors));
      f (Nx_io.load_gguf path))

(* The storage formats, each with a range of integers it holds exactly at the
   scales [encode] writes: Q8_0 at d = 1, Q4_K at d = dmin = 1 with every
   sub-block's scale 1 and min 8, Q6_K at d = 1 with every scale 1. *)
type format = F32 | F16 | Q8_0 | Q4_K | Q6_K

let tag = function F32 -> 0 | F16 -> 1 | Q8_0 -> 8 | Q4_K -> 12 | Q6_K -> 14

let range = function
  | F32 | F16 -> (-1024, 1024)
  | Q8_0 -> (-128, 127)
  | Q4_K -> (-8, 7)
  | Q6_K -> (-32, 31)

let one = 0x3c00

(* The float16 bits of an integer of magnitude below 2048. *)
let f16_of_int v =
  if v = 0 then 0
  else
    let a = abs v in
    let rec log2 e = if a lsr (e + 1) = 0 then e else log2 (e + 1) in
    let e = log2 0 in
    (if v < 0 then 0x8000 else 0)
    lor ((e + 15) lsl 10)
    lor ((a lsl (10 - e)) land 0x3ff)

(* The bytes of a row of values in [format], by ggml's block layouts
   (ggml-common.h). *)
let encode format (row : int array) =
  let b = Buffer.create 256 in
  let byte n = Buffer.add_char b (Char.chr (n land 0xff)) in
  let half n = Buffer.add_string b (le n 2) in
  let blocks size f =
    for i = 0 to (Array.length row / size) - 1 do
      f (Array.sub row (i * size) size)
    done
  in
  (match format with
  | F32 ->
      Array.iter
        (fun v ->
          Buffer.add_string b
            (le (Int32.to_int (Int32.bits_of_float (Float.of_int v))) 4))
        row
  | F16 -> Array.iter (fun v -> half (f16_of_int v)) row
  | Q8_0 ->
      blocks 32 (fun y ->
          half one;
          Array.iter byte y)
  | Q4_K ->
      blocks 256 (fun y ->
          half one;
          half one;
          List.iter byte [ 1; 1; 1; 1; 8; 8; 8; 8; 0x81; 0x81; 0x81; 0x81 ];
          for c = 0 to 3 do
            for l = 0 to 31 do
              let q i = y.((64 * c) + i) + 8 in
              byte (q l lor (q (l + 32) lsl 4))
            done
          done)
  | Q6_K ->
      blocks 256 (fun y ->
          let q i = y.(i) + 32 in
          let ql = Bytes.make 128 '\000' and qh = Bytes.make 64 '\000' in
          let set bs i n = Bytes.set bs i (Char.chr n) in
          for h = 0 to 1 do
            for l = 0 to 31 do
              let q0 = q ((128 * h) + l) and q1 = q ((128 * h) + l + 32) in
              let q2 = q ((128 * h) + l + 64) and q3 = q ((128 * h) + l + 96) in
              set ql ((64 * h) + l) (q0 land 15 lor ((q2 land 15) lsl 4));
              set ql ((64 * h) + l + 32) (q1 land 15 lor ((q3 land 15) lsl 4));
              set qh
                ((32 * h) + l)
                ((q0 lsr 4)
                lor ((q1 lsr 4) lsl 2)
                lor ((q2 lsr 4) lsl 4)
                lor ((q3 lsr 4) lsl 6))
            done
          done;
          Buffer.add_bytes b ql;
          Buffer.add_bytes b qh;
          for _ = 1 to 16 do
            byte 1
          done;
          half one));
  Buffer.contents b

(* A model whose every width is a whole number of Q4_K and Q6_K blocks. *)
let dim = 256
let heads = 4
let kv_heads = 2
let head_dim = dim / heads
let hidden = 256
let vocab = 3
let n_layers = 3
let context = 32

let metadata =
  [
    kv_string "general.architecture" "llama";
    kv_u32 "llama.context_length" context;
    kv_u32 "llama.embedding_length" dim;
    kv_u32 "llama.block_count" n_layers;
    kv_u32 "llama.feed_forward_length" hidden;
    kv_u32 "llama.rope.dimension_count" head_dim;
    kv_u32 "llama.attention.head_count" heads;
    kv_u32 "llama.attention.head_count_kv" kv_heads;
    kv_f32 "llama.attention.layer_norm_rms_epsilon" 1e-5;
    kv_f32 "llama.rope.freq_base" 500.;
  ]

(* Each layer's projections, [q; k; v; out; gate; up; down], in formats that
   give the reordered q and k each format. *)
let layer_formats =
  [|
    [| Q8_0; Q4_K; Q6_K; F16; F32; Q8_0; Q4_K |];
    [| Q4_K; Q6_K; Q8_0; Q6_K; F16; Q4_K; Q8_0 |];
    [| Q6_K; F16; F32; Q4_K; Q8_0; Q6_K; F16 |];
  |]

(* llama.cpp's reordering of a head's q or k rows: its row 2j + c is
   HuggingFace's row c * head_dim / 2 + j. *)
let pair_rows ~heads rows (m : int array array) =
  let half = rows / heads / 2 in
  Array.init rows (fun r ->
      let h = r / (2 * half) and j = r mod (2 * half) / 2 and c = r mod 2 in
      m.((h * 2 * half) + (c * half) + j))

(* [convert st] is a model's GGUF tensors and the HuggingFace archive of the
   same values, drawn from [st]. *)
let convert st =
  let tensors = ref [] and hf = ref [] in
  let entry ?heads format ~gguf ~name shape =
    let lo, hi = range format in
    let rows = shape.(0)
    and cols = if Array.length shape = 1 then 1 else shape.(1) in
    let m =
      Array.init rows (fun _ ->
          Array.init cols (fun _ -> lo + Random.State.int st (hi - lo + 1)))
    in
    let stored =
      match heads with None -> m | Some heads -> pair_rows ~heads rows m
    in
    let bytes =
      if Array.length shape = 1 then
        encode format (Array.map (fun r -> r.(0)) stored)
      else String.concat "" (Array.to_list (Array.map (encode format) stored))
    in
    tensors := (gguf ^ ".weight", tag format, shape, bytes) :: !tensors;
    let values = Array.map Float.of_int (Array.concat (Array.to_list m)) in
    hf := (name ^ ".weight", Nx.P (Nx.create Nx.float32 shape values)) :: !hf
  in
  entry Q4_K ~gguf:"token_embd" ~name:"model.embed_tokens" [| vocab; dim |];
  for i = 0 to n_layers - 1 do
    let f = layer_formats.(i) in
    let gguf s = Printf.sprintf "blk.%d.%s" i s in
    let name s = Printf.sprintf "model.layers.%d.%s" i s in
    let matrix ?heads k g n shape =
      entry ?heads f.(k) ~gguf:(gguf g) ~name:(name n) shape
    in
    entry F32 ~gguf:(gguf "attn_norm") ~name:(name "input_layernorm") [| dim |];
    matrix ~heads 0 "attn_q" "self_attn.q_proj" [| dim; dim |];
    matrix ~heads:kv_heads 1 "attn_k" "self_attn.k_proj"
      [| kv_heads * head_dim; dim |];
    matrix 2 "attn_v" "self_attn.v_proj" [| kv_heads * head_dim; dim |];
    matrix 3 "attn_output" "self_attn.o_proj" [| dim; dim |];
    entry F32 ~gguf:(gguf "ffn_norm")
      ~name:(name "post_attention_layernorm")
      [| dim |];
    matrix 4 "ffn_gate" "mlp.gate_proj" [| hidden; dim |];
    matrix 5 "ffn_up" "mlp.up_proj" [| hidden; dim |];
    matrix 6 "ffn_down" "mlp.down_proj" [| dim; hidden |]
  done;
  entry F16 ~gguf:"output_norm" ~name:"model.norm" [| dim |];
  entry Q6_K ~gguf:"output" ~name:"lm_head" [| vocab; dim |];
  (List.rev !tensors, Nx_io.Archive.of_list !hf)

let host _ ~axis:_ = Nx.Host.on

(* Each leaf's name and values. *)
let leaves p =
  let a =
    Nx_io.Archive.of_value (Nx.Ptree.instantiate (module Llama.Params)) p
  in
  List.map
    (fun name ->
      let shape =
        match Nx_io.Archive.find name a with
        | Some (Nx.P t) -> Nx.shape t
        | None -> [||]
      in
      (name, Nx.to_array (Nx_io.Archive.float ~shape Nx.float64 name a)))
    (Nx_io.Archive.names a)

let test_import =
  prop ~count:8 "a converted file loads as the HuggingFace one" Gen.nat
    (fun seed ->
      let tensors, hf = convert (Random.State.make [| seed |]) in
      with_gguf metadata tensors (fun g ->
          let cfg = Llama.config_of_gguf g in
          equal
            (list (pair string (array float_exact)))
            (leaves (Llama.of_hf ~placement:host cfg Nx.float32 hf))
            (leaves (Llama.of_gguf ~placement:host cfg Nx.float32 g))))

let summary (c : _ Llama.config) =
  ( [
      c.vocab_size;
      c.dim;
      c.n_layers;
      c.n_heads;
      c.n_kv_heads;
      c.head_dim;
      c.hidden_dim;
      Kaun.Rope.context c.rope;
    ],
    c.tied )

let frequencies ~theta =
  Kaun.Rope.frequencies (Kaun.Rope.make ~theta ~head_dim ~context ())

let test_config () =
  let tensors, _ = convert (Random.State.make [| 0 |]) in
  with_gguf metadata tensors (fun g ->
      let cfg = Llama.config_of_gguf g in
      equal
        (pair (list int) bool)
        ( [ vocab; dim; n_layers; heads; kv_heads; head_dim; hidden; context ],
          false )
        (summary cfg);
      equal ~msg:"norm_eps" float_exact
        (Int32.float_of_bits (Int32.bits_of_float 1e-5))
        cfg.norm_eps;
      equal ~msg:"the standard schedule at the file's base" (array float_exact)
        (frequencies ~theta:500.)
        (Kaun.Rope.frequencies cfg.rope))

(* A tied file, without output.weight, that names its vocabulary and stores
   Llama 3's divisors of the standard frequencies. *)
let test_llama3_config () =
  let divisors =
    Array.init (head_dim / 2) (fun i -> Float.of_int (1 lsl (i mod 4)))
  in
  let tensors =
    [
      ( "token_embd.weight",
        tag F16,
        [| vocab; dim |],
        encode F16 (Array.make (vocab * dim) 0) );
      ( "rope_freqs.weight",
        tag F32,
        [| head_dim / 2 |],
        encode F32 (Array.map int_of_float divisors) );
    ]
  in
  with_gguf (kv_u32 "llama.vocab_size" 7 :: metadata) tensors (fun g ->
      let cfg = Llama.config_of_gguf g in
      equal
        (pair (list int) bool)
        ([ 7; dim; n_layers; heads; kv_heads; head_dim; hidden; context ], true)
        (summary cfg);
      equal ~msg:"the standard frequencies over the divisors"
        (array float_exact)
        (Array.map2 ( /. ) (frequencies ~theta:500.) divisors)
        (Kaun.Rope.frequencies cfg.rope))

let without key = List.filter (fun kv -> kv <> key) metadata

let test_config_errors =
  cases ~id:fst "metadata the import rejects"
    [
      ( "another architecture",
        ( kv_string "general.architecture" "gpt2"
          :: without (kv_string "general.architecture" "llama"),
          "llama GGUF: architecture gpt2, llama expected" ) );
      ( "a missing key",
        ( without (kv_u32 "llama.block_count" n_layers),
          "llama GGUF: missing llama.block_count" ) );
      ( "a rotary scaling",
        ( kv_string "llama.rope.scaling.type" "linear" :: metadata,
          "llama GGUF: unsupported llama.rope.scaling.type linear" ) );
      ( "a partial rotary",
        ( kv_u32 "llama.rope.dimension_count" 32
          :: without (kv_u32 "llama.rope.dimension_count" head_dim),
          "llama GGUF: llama.rope.dimension_count 32 rotates part of a head of \
           64" ) );
    ]
    (fun (_, (kvs, message)) ->
      let tensors, _ = convert (Random.State.make [| 0 |]) in
      with_gguf kvs tensors (fun g ->
          raises (Failure message) (fun () -> Llama.config_of_gguf g)))

(* A block format nx_quant does not decode, Q4_0 (tag 2, 18 bytes per 32
   values). *)
let test_unknown_format () =
  let tensors, _ = convert (Random.State.make [| 0 |]) in
  let v = "blk.0.attn_v.weight" in
  let tensors =
    List.map
      (fun ((name, _, shape, _) as t) ->
        if name <> v then t
        else
          (name, 2, shape, String.make (shape.(0) * shape.(1) / 32 * 18) '\000'))
      tensors
  in
  with_gguf metadata tensors (fun g ->
      let cfg = Llama.config_of_gguf g in
      raises
        (Failure
           (v
          ^ ": not a float16, bfloat16, float32, float64, Q8_0, Q4_K or Q6_K \
             entry"))
        (fun () -> Llama.of_gguf ~placement:host cfg Nx.float32 g))

let () =
  exit
    (run "llama"
       [
         group "structures" [ test "the parameters' walk" test_params ];
         group "gguf"
           [
             test_import;
             test "the configuration" test_config;
             test "a tied Llama 3 configuration" test_llama3_config;
             test_config_errors;
             test "a block format nx_quant does not decode" test_unknown_format;
           ];
       ])
