(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Loading a checkpoint of gpt-oss's names and shapes, small and float: the
   query, key and value entries join into one projection grouped by key-value
   head, unpacked and Q8_0, and a tensor-parallel placement over two software
   devices, which cuts that projection between key-value heads, gives one
   device's logits. Each claim holds at one and at two key-value heads a
   device, and at one query head a key-value head. *)

open Windtrap
open Kaun

let head_dim = 8

(* Query and key-value heads. *)
let heads = [ (4, 2); (8, 4); (4, 4) ]

let label (n_heads, n_kv_heads) =
  Printf.sprintf "%d/%d heads" n_heads n_kv_heads

let config (n_heads, n_kv_heads) =
  {
    Gpt_oss.vocab_size = 16;
    dim = 32;
    layers = [ Gpt_oss.Sliding; Full ];
    window = 3;
    n_heads;
    n_kv_heads;
    head_dim;
    hidden_dim = 32;
    experts = 4;
    experts_per_token = 2;
    swiglu_limit = 7.0;
    norm_eps = 1e-5;
    rope = Rope.make ~head_dim ~context:16 ();
    attention_scale = 1.0 /. sqrt (float_of_int head_dim);
    tied = false;
  }

(* Floats in [-0.5, 0.5), from a pattern of the position and [seed]. *)
let floats seed s =
  let n = Array.fold_left ( * ) 1 s in
  Nx.create Nx.float32 s
    (Array.init n (fun i ->
         (Float.of_int ((i + (seed * 31)) * 7919 mod 97) /. 97.) -. 0.5))

(* The checkpoint's entries, each projection [outputs; inputs], the experts as
   floats [experts; inputs; outputs]. *)
let archive (cfg : _ Gpt_oss.config) =
  let n_heads = cfg.n_heads and n_kv_heads = cfg.n_kv_heads in
  let seed = ref 0 in
  let entry name s =
    incr seed;
    (name, Nx.P (floats !seed s))
  in
  let q_dim = n_heads * head_dim and kv_dim = n_kv_heads * head_dim in
  let d = cfg.dim and e = cfg.experts and h = cfg.hidden_dim in
  let linear name ~inputs ~outputs =
    [
      entry (name ^ ".weight") [| outputs; inputs |];
      entry (name ^ ".bias") [| outputs |];
    ]
  in
  let block i _ =
    let at = Printf.sprintf "model.layers.%d.%s" i in
    List.concat
      [
        [ entry (at "input_layernorm.weight") [| d |] ];
        linear (at "self_attn.q_proj") ~inputs:d ~outputs:q_dim;
        linear (at "self_attn.k_proj") ~inputs:d ~outputs:kv_dim;
        linear (at "self_attn.v_proj") ~inputs:d ~outputs:kv_dim;
        linear (at "self_attn.o_proj") ~inputs:q_dim ~outputs:d;
        [
          entry (at "self_attn.sinks") [| n_heads |];
          entry (at "post_attention_layernorm.weight") [| d |];
        ];
        linear (at "mlp.router") ~inputs:d ~outputs:e;
        [
          entry (at "mlp.experts.gate_up_proj") [| e; d; 2 * h |];
          entry (at "mlp.experts.gate_up_proj_bias") [| e; 2 * h |];
          entry (at "mlp.experts.down_proj") [| e; h; d |];
          entry (at "mlp.experts.down_proj_bias") [| e; d |];
        ];
      ]
  in
  Nx_io.Archive.of_list
    (List.concat
       [
         [ entry "model.embed_tokens.weight" [| cfg.vocab_size; d |] ];
         List.concat (List.mapi block cfg.layers);
         [
           entry "model.norm.weight" [| d |];
           entry "lm_head.weight" [| cfg.vocab_size; d |];
         ];
       ])

(* The rows [Gpt_oss.attn] states for layer [i]'s entries [leaf]: for each
   key-value head, its query heads, then its key head and its value head. *)
let grouped (cfg : _ Gpt_oss.config) archive i leaf =
  let per_kv = cfg.n_heads / cfg.n_kv_heads in
  let entry p =
    let name = Printf.sprintf "model.layers.%d.self_attn.%s%s" i p leaf in
    Nx.unpack Nx.float32 (Option.get (Nx_io.Archive.find name archive))
  in
  let q = entry "q_proj" and k = entry "k_proj" and v = entry "v_proj" in
  let rows x ~heads j = Nx.slice [ R (j * heads, (j + 1) * heads) ] x in
  Nx.concatenate ~axis:0
    (List.concat
       (List.init cfg.n_kv_heads (fun j ->
            [
              rows q ~heads:(per_kv * head_dim) j;
              rows k ~heads:head_dim j;
              rows v ~heads:head_dim j;
            ])))

let host _ ~axis:_ = Nx.Host.on
let values x = Nx.to_array (Nx.copy x)

(* [each f] is [f] at each of [heads]'s configurations, its archive and a
   message prefix. *)
let each f =
  List.iter
    (fun h ->
      let cfg = config h in
      f cfg (archive cfg) (fun s -> label h ^ ", " ^ s))
    heads

let test_unpacked () =
  each @@ fun cfg archive msg ->
  let grouped = grouped cfg archive in
  let p = Gpt_oss.of_hf ~placement:host cfg Nx.float32 archive in
  List.iteri
    (fun i (b : _ Gpt_oss.block) ->
      let msg s = msg (Printf.sprintf "layer %d: %s" i s) in
      match b.attn.qkv with
      | Float l ->
          equal ~msg:(msg "weight") (array float_exact)
            (values (grouped i ".weight"))
            (values l.w);
          equal ~msg:(msg "bias")
            (option (array float_exact))
            (Some (values (grouped i ".bias")))
            (Option.map values l.b)
      | Quant _ -> fail (msg "packed"))
    p.blocks

let test_q8_0 () =
  each @@ fun cfg archive msg ->
  let grouped = grouped cfg archive in
  let p = Gpt_oss.of_hf ~pack:Q8_0 ~placement:host cfg Nx.float32 archive in
  List.iteri
    (fun i (b : _ Gpt_oss.block) ->
      let msg s = msg (Printf.sprintf "layer %d: %s" i s) in
      match (b.attn.qkv, Nx_quant.quantize_q8_0 (grouped i ".weight")) with
      | Quant { w = Q8_0 w; b }, Q8_0 want ->
          equal ~msg:(msg "blocks") (array int) (values want.blocks)
            (values w.blocks);
          equal ~msg:(msg "bias")
            (option (array float_exact))
            (Some (values (grouped i ".bias")))
            (Option.map values b)
      | _ -> fail (msg "not Q8_0"))
    p.blocks

let memory k = Result.get_ok (Rig.software (Printf.sprintf "tp%d" k))

module S = (val Nx.devices [ memory 0; memory 1 ])

(* Columns and the pools cut along their outputs and heads: each device holds
   half the key-value heads and their query heads. Rows are cut along their
   inputs, or with [~rows:false] whole: a Q8_0 weight is quantised whole
   along its inputs. *)
let tensor_parallel ~rows role ~axis =
  match role with
  | Gpt_oss.Column | Kv_heads -> S.split ~axis
  | Row when rows -> S.split ~axis
  | Row | Whole | Experts -> S.on

let logits (type d) ~(on : d Nx.Placement.t) cfg p =
  let cfg = Gpt_oss.place_config on cfg in
  let ids = Nx.create Nx.int64 [| 1; 5 |] [| 3L; 1L; 4L; 1L; 5L |] in
  let h = Gpt_oss.hidden cfg p (Nx.place on ids) in
  values (Nx.place Nx.Host.on (Gpt_oss.logits cfg p h))

(* The output projection's sum is split over the two devices, so the logits
   agree within float32's reassociation of its terms. *)
let test_tensor_parallel ~rows pack () =
  each @@ fun cfg archive msg ->
  let one = Gpt_oss.of_hf ~pack ~placement:host cfg Nx.float32 archive in
  let placement = tensor_parallel ~rows in
  let split = Gpt_oss.of_hf ~pack ~placement cfg Nx.float32 archive in
  equal ~msg:(msg "logits")
    (array (float_rel ~rel:1e-5 ~abs:1e-6))
    (logits ~on:Nx.Host.on cfg one)
    (logits ~on:S.on cfg split)

let () =
  exit
    (run "gpt-oss loading"
       [
         group "query, key and value"
           [
             test "unpacked, joined by key-value head" test_unpacked;
             test "Q8_0, joined by key-value head" test_q8_0;
           ];
         group "tensor parallel"
           [
             test "unpacked gives one device's logits"
               (test_tensor_parallel ~rows:true Unpacked);
             test "Q8_0, rows whole, gives one device's logits"
               (test_tensor_parallel ~rows:false Q8_0);
           ];
       ])
