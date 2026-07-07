(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* GPT-2 scaffolding tests that do not need the 550MB pretrained weights: a
   tiny randomly initialised transformer with the same block structure as the
   gpt2 example, exercised in prompt mode, plus the byte-level BPE tokenizer
   against token ids produced by the reference tokenizer (tiktoken "gpt2"). *)

open Windtrap
open Tolk_frontend
module Embedding = Tolk_nn.Embedding
module Linear = Tolk_nn.Linear
module Layer_norm = Tolk_nn.Layer_norm
module State = Tolk_nn.State

(* Tiny transformer: same decomposition as examples/gpt2, scaled down. *)

let dim = 64
let n_heads = 2
let n_layers = 2

(* Like the real model's 50257, the vocabulary size is deliberately not a
   multiple of 16: a divisible vocabulary triggers the matrix-vector
   grouping optimization on devices with local memory, whose lowering
   currently miscompiles this head kernel (invalid RESHAPE). *)
let vocab_size = 251
let max_seq_len = 32
let head_dim = dim / n_heads

type block = {
  c_attn : Linear.t;
  c_proj : Linear.t;
  ln_1 : Layer_norm.t;
  c_fc : Linear.t;
  c_proj2 : Linear.t;
  ln_2 : Layer_norm.t;
}

type model = {
  wte : Embedding.t;
  wpe : Embedding.t;
  h : block list;
  ln_f : Layer_norm.t;
  lm_head : Linear.t;
}

let build () =
  {
    wte = Embedding.create vocab_size dim;
    wpe = Embedding.create max_seq_len dim;
    h =
      List.init n_layers (fun _ ->
          {
            c_attn = Linear.create dim (3 * dim);
            c_proj = Linear.create dim dim;
            ln_1 = Layer_norm.create dim;
            c_fc = Linear.create dim (4 * dim);
            c_proj2 = Linear.create (4 * dim) dim;
            ln_2 = Layer_norm.create dim;
          })
      ;
    ln_f = Layer_norm.create dim;
    lm_head = Linear.create ~bias:false dim vocab_size;
  }

let state_dict m =
  let linear p (l : Linear.t) =
    ((p ^ ".weight", l.weight)
    :: (match l.bias with Some b -> [ (p ^ ".bias", b) ] | None -> []))
  in
  let norm p (l : Layer_norm.t) =
    [ (p ^ ".weight", Option.get l.weight); (p ^ ".bias", Option.get l.bias) ]
  in
  [ ("wte.weight", m.wte.weight); ("wpe.weight", m.wpe.weight) ]
  @ List.concat
      (List.mapi
         (fun i b ->
           let p = Printf.sprintf "h.%d" i in
           linear (p ^ ".attn.c_attn") b.c_attn
           @ linear (p ^ ".attn.c_proj") b.c_proj
           @ linear (p ^ ".mlp.c_fc") b.c_fc
           @ linear (p ^ ".mlp.c_proj") b.c_proj2
           @ norm (p ^ ".ln_1") b.ln_1
           @ norm (p ^ ".ln_2") b.ln_2)
         m.h)
  @ norm "ln_f" m.ln_f
  @ linear "lm_head" m.lm_head

(* Deterministic pseudo-random weights: a fixed LCG seeded per parameter so
   the model is identical across runs. *)
let random_state_dict model =
  let state = ref 0x2545F491 in
  let next () =
    state := (!state * 1103515245) + 12345;
    let bits = (!state lsr 8) land 0xFFFF in
    (float_of_int bits /. 65535.0 *. 0.2) -. 0.1
  in
  List.map
    (fun (name, t) ->
      let n = Tensor.numel t in
      (name, Run.of_float_array ~shape:(Tensor.shape t) (Array.init n (fun _ -> next ()))))
    (state_dict model)

let attention (b : block) x mask =
  let bsz, seqlen =
    match Tensor.shape x with [ b; s; _ ] -> (b, s) | _ -> assert false
  in
  let xqkv =
    Movement.reshape (Linear.apply b.c_attn x)
      [ bsz; seqlen; 3; n_heads; head_dim ]
  in
  let sel i = Op.getitem xqkv Movement.[ All; All; I i; All; All ] in
  let tr t = Movement.transpose ~dim0:1 ~dim1:2 t in
  let out =
    Op.scaled_dot_product_attention ?attn_mask:mask (tr (sel 0)) (tr (sel 1))
      (tr (sel 2))
  in
  Linear.apply b.c_proj (Movement.reshape (tr out) [ bsz; seqlen; dim ])

let block_apply b x mask =
  let h = Elementwise.add x (attention b (Layer_norm.apply b.ln_1 x) mask) in
  Elementwise.contiguous
    (Elementwise.add h
       (Linear.apply b.c_proj2
          (Elementwise.gelu (Linear.apply b.c_fc (Layer_norm.apply b.ln_2 h)))))

let forward ?(temperature = 0.) m tokens =
  let seqlen = List.nth (Tensor.shape tokens) 1 in
  let pos =
    Movement.shrink
      (Movement.reshape (Op.arange max_seq_len) [ 1; -1 ])
      [ (0, 1); (0, seqlen) ]
  in
  let h =
    Elementwise.add (Embedding.apply m.wte tokens) (Embedding.apply m.wpe pos)
  in
  let mask =
    if seqlen > 1 then
      Some
        (Op.triu ~diagonal:1
           (Creation.full [ 1; 1; seqlen; seqlen ] (Tensor.Sfloat neg_infinity)))
    else None
  in
  let h = List.fold_left (fun h b -> block_apply b h mask) h m.h in
  let logits = Linear.apply m.lm_head (Layer_norm.apply m.ln_f h) in
  let logits = Op.getitem logits Movement.[ All; I (-1); All ] in
  if temperature < 1e-6 then Op.argmax ~axis:(-1) logits
  else
    Rand.multinomial
      (Op.softmax (Elementwise.div logits (Tensor.f temperature)))

let generate ?temperature m prompt count =
  let toks = ref prompt in
  for _ = 1 to count do
    let arr = Array.of_list !toks in
    let tokens = Run.of_int_array ~shape:[ 1; Array.length arr ] arr in
    toks := !toks @ [ Run.item_int (forward ?temperature m tokens) ]
  done;
  !toks

let model_tests =
  group "tiny model"
    [
      test "forward produces a token id" (fun () ->
          let m = build () in
          State.load_state_dict (state_dict m) (random_state_dict m);
          let tokens = Run.of_int_array ~shape:[ 1; 3 ] [| 10; 20; 30 |] in
          let out = forward m tokens in
          equal (list int) [ 1 ] (Tensor.shape out);
          let id = Run.item_int out in
          is_true (id >= 0 && id < vocab_size));
      test "prompt-mode generation is deterministic" (fun () ->
          let m = build () in
          State.load_state_dict (state_dict m) (random_state_dict m);
          let a = generate m [ 10; 20; 30 ] 4 in
          let b = generate m [ 10; 20; 30 ] 4 in
          equal (list int) a b;
          equal int 7 (List.length a));
      test "seeded sampling is deterministic and seed-sensitive" (fun () ->
          let m = build () in
          State.load_state_dict (state_dict m) (random_state_dict m);
          let sample seed =
            Rand.manual_seed seed;
            generate ~temperature:0.8 m [ 10; 20; 30 ] 4
          in
          let a = sample 42 in
          List.iter (fun t -> is_true (t >= 0 && t < vocab_size)) a;
          equal int 7 (List.length a);
          equal (list int) a (sample 42);
          is_true (sample 1337 <> a));
    ]

(* Tokenizer: gpt2's byte-level BPE loaded through brot must agree with the
   reference tokenizer. Expected ids generated once with tiktoken ("gpt2"). *)

let tokenizer_path =
  Filename.concat
    (try Sys.getenv "XDG_CACHE_HOME"
     with Not_found -> Filename.concat (Sys.getenv "HOME") ".cache")
    "tolk-gpt2/tokenizer.json"

let with_tokenizer f =
  if not (Sys.file_exists tokenizer_path) then
    skip ~reason:("tokenizer data not cached at " ^ tokenizer_path) ()
  else
    match Brot.from_file tokenizer_path with
    | Ok t -> f t
    | Error e -> failf "tokenizer load failed: %s" e

let tokenizer_tests =
  group "tokenizer"
    [
      test "encodes the reference prompts like tiktoken" (fun () ->
          with_tokenizer (fun t ->
              equal (array int)
                [| 2061; 318; 262; 3280; 284; 1204; 11; 262; 6881; 11; 290; 2279; 30 |]
                (Brot.encode_ids t
                   "What is the answer to life, the universe, and everything?");
              equal (array int) [| 15496; 13 |] (Brot.encode_ids t "Hello.")));
      test "decode round-trips" (fun () ->
          with_tokenizer (fun t ->
              let s = "What is the answer to life, the universe, and everything?" in
              equal string s (Brot.decode t (Brot.encode_ids t s));
              equal string "Hello." (Brot.decode t (Brot.encode_ids t "Hello."))));
    ]

let () = run "Tolk_gpt2" [ model_tests; tokenizer_tests ]
