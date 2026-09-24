(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The jitted decode step of a GPT-2 124M shaped decoder: one token in, one
   token out, through key-value caches, with the state consumed and the sampled
   token read back on the host as a generate loop does. The stack is built here
   from kaun layers with zero weights: the step's cost does not depend on their
   values, and kaun ships no model to depend on.

   The cache lengths show how the step scales with the cache it carries, not
   only with the single position it writes. *)

open Kaun

let vocab = 50257
and positions = 1024
and embd = 768
and inner = 3072

let layers = 12
and heads = 12

let head_dim = embd / heads
let eps = 1e-5

type block = {
  ln1 : Nx.float32_t Layer_norm.t;
  attn : Nx.float32_t Attention.t;
  ln2 : Nx.float32_t Layer_norm.t;
  fc : Nx.float32_t Linear.t;
  proj : Nx.float32_t Linear.t;
}

type model = {
  wte : Nx.float32_t Embedding.t;
  wpe : Nx.float32_t Embedding.t;
  blocks : block list;
  ln_f : Nx.float32_t Layer_norm.t;
}

let model () =
  let zeros = Init.zeros in
  let linear ~inputs ~outputs =
    Linear.make ~w_init:zeros ~bias_init:zeros ~inputs ~outputs Nx.float32
  in
  let block () =
    {
      ln1 = Layer_norm.init ~dim:embd;
      attn =
        Attention.make ~w_init:zeros ~bias_init:zeros ~embed_dim:embd Nx.float32;
      ln2 = Layer_norm.init ~dim:embd;
      fc = linear ~inputs:embd ~outputs:inner;
      proj = linear ~inputs:inner ~outputs:embd;
    }
  in
  {
    wte = Embedding.make ~init:zeros ~vocab ~dim:embd Nx.float32;
    wpe = Embedding.make ~init:zeros ~vocab:positions ~dim:embd Nx.float32;
    blocks = List.init layers (fun _ -> block ());
    ln_f = Layer_norm.init ~dim:embd;
  }

let cached m caches index ids =
  let x =
    Nx.add
      (Embedding.apply m.wte ids)
      (Embedding.apply m.wpe (Cache_index.positions index))
  in
  let x, rev =
    List.fold_left2
      (fun (x, cs) b c ->
        let a, c =
          Attention.cached ~head_dim b.attn c index
            (Layer_norm.apply ~eps b.ln1 x)
        in
        let x = Nx.add x a in
        let mlp =
          Linear.apply b.proj
            (Fn.gelu_approx (Linear.apply b.fc (Layer_norm.apply ~eps b.ln2 x)))
        in
        (Nx.add x mlp, c :: cs))
      (x, []) m.blocks caches
  in
  (x, List.rev rev)

let logits m h =
  Nx.matmul
    (Layer_norm.apply ~eps m.ln_f h)
    (Nx.transpose m.wte.Embedding.table)

let cache ~slots =
  List.init layers (fun _ ->
      Attention.Cache.make ~slots ~kv_heads:heads ~head_dim Nx.float32)

(* A decode loop warmed past its two compilations. The returned thunk decodes
   one token at a fixed position in the middle of the cache, from the previous
   call's token and caches: every measured step is in range, whatever the number
   of samples. *)
let decoder params ~len =
  let caches = Nx.Ptree.list (Nx.Ptree.instantiate (module Attention.Cache)) in
  let step =
    Rune.jit
      Nx.Ptree.(
        tensor @-> Cache_index.ptree @-> consumes caches
        @@ returns (pair tensor caches))
      (fun token index caches ->
        let seq = (Nx.shape token).(1) in
        let h, caches = cached params caches index token in
        let last = Nx.slice [ A; I (seq - 1) ] h in
        (Nx.reshape [| 1; 1 |] (Nx.argmax ~axis:1 (logits params last)), caches))
  in
  let state =
    ref
      (step
         (Nx.zeros Nx.int32 [| 1; 8 |])
         (Cache_index.rows ~context:len [| 8 |])
         (cache ~slots:len))
  in
  let at = Nx.full Nx.int32 [| 1; 1 |] (Int32.of_int (len / 2)) in
  let middle =
    Cache_index.make ~pos:at
      ~table:(Nx.reshape [| 1; len |] (Nx.arange Nx.int32 0 len 1))
      ()
  in
  let advance () =
    let token, caches = !state in
    state := step token middle caches;
    ignore (Nx.item [ 0; 0 ] (fst !state) : int32)
  in
  advance ();
  advance

(* The decoder is built inside the measuring worker: a device handle does not
   survive the fork that isolates a case. *)
let case len =
  Thumper.bench_with_setup
    ~setup:(fun () -> decoder (model ()) ~len)
    (Printf.sprintf "decode step, cache %d" len)
    (fun advance -> advance ())

let lens = [ 256; 1024 ]

(* A forked worker cannot reach the GPU's compiler service either. A child
   process compiles the kernels first; the workers then load them from tolk's
   kernel cache. This process never touches the device. *)
let warm () =
  let exe = Sys.executable_name in
  let pid =
    Unix.create_process exe [| exe; "--warm" |] Unix.stdin Unix.stdout
      Unix.stderr
  in
  match Unix.waitpid [] pid with
  | _, Unix.WEXITED 0 -> ()
  | _ -> failwith "bench_decode: compiling the decode kernels failed"

let () =
  match Array.to_list Sys.argv with
  | [ _; "--warm" ] -> List.iter (fun len -> (decoder (model ()) ~len) ()) lens
  | argv ->
      let measures =
        match argv with
        | _ :: ("list" | "-h" | "--help" | "-V" | "--version") :: _ -> false
        | _ -> true
      in
      if measures then warm ();
      let budgets =
        [ Thumper.Budget.no_slower_than ~metric:Thumper.Metric.wall_time 0.05 ]
      in
      Thumper.run "kaun_decode" ~budgets
        [ Thumper.group "Gpt2" (List.map case lens) ]
