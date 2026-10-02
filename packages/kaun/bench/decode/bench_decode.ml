(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The jitted decode step of a GPT-2 124M shaped decoder: one token in, one
   token out, through key-value caches, with the state consumed and the sampled
   token read back on the host as a generate loop does. The stack is built here
   from kaun layers with zero weights: the step's cost does not depend on their
   values, and kaun ships no model to depend on. Each weight, and the position
   the step reads, is copied to storage of its own, as a loaded model's are: a
   constant is one element seen at every index, which a compiled call folds.

   The cache lengths show how the step scales with the cache it carries, not
   only with the single position it writes.

   gpt-oss-20b's own decode step and prefill are measured by the example's
   suite, examples/06-gpt-oss/bench; the routed products below guard its
   kernels.

   The cases compile in their setups. [--warm] runs each case's setup and one
   call in a process of its own, which fills tolk's disk cache: the cases then
   read their kernels back, and a forked worker, which cannot reach Metal's
   compiler, needs it. *)

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
  let zeros ~fan_in ~fan_out dtype shape =
    Nx.copy (Init.zeros ~fan_in ~fan_out dtype shape)
  in
  let norm () =
    let { Layer_norm.gamma; beta } = Layer_norm.init ~dim:embd in
    { Layer_norm.gamma = Nx.copy gamma; beta = Nx.copy beta }
  in
  let linear ~inputs ~outputs =
    Linear.make ~w_init:zeros ~bias_init:zeros ~inputs ~outputs Nx.float32
  in
  let block () =
    {
      ln1 = norm ();
      attn =
        Attention.make ~w_init:zeros ~bias_init:zeros ~embed_dim:embd Nx.float32;
      ln2 = norm ();
      fc = linear ~inputs:embd ~outputs:inner;
      proj = linear ~inputs:inner ~outputs:embd;
    }
  in
  {
    wte = Embedding.make ~init:zeros ~vocab ~dim:embd Nx.float32;
    wpe = Embedding.make ~init:zeros ~vocab:positions ~dim:embd Nx.float32;
    blocks = List.init layers (fun _ -> block ());
    ln_f = norm ();
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
         (Nx.zeros Nx.int64 [| 1; 8 |])
         (Cache_index.rows ~context:len [| 8 |])
         (cache ~slots:len))
  in
  let at = Nx.create Nx.int64 [| 1; 1 |] [| Int64.of_int (len / 2) |] in
  let middle =
    Cache_index.make ~pos:at
      ~table:(Nx.reshape [| 1; len |] (Nx.arange Nx.int64 0 len 1))
      ()
  in
  let advance () =
    let token, caches = !state in
    state := step token middle caches;
    ignore (Nx.item [ 0; 0 ] (fst !state) : int64)
  in
  advance ();
  advance

(* A case: its name, and its setup, which compiles the case's program and
   returns one synchronized call. A setup runs in the measuring worker, since a
   device handle does not survive the fork that isolates a case. *)
type case = { name : string; setup : unit -> unit -> unit }

let gpt2 =
  List.map
    (fun len ->
      {
        name = Printf.sprintf "decode step, cache %d" len;
        setup = (fun () -> decoder (model ()) ~len);
      })
    [ 256; 1024 ]

(* Routed quantised products at gpt-oss-20b's shapes: the gate and up projection
   of one layer's 32 experts, MXFP4 [[| 32; 5760; 2880 |]], applied to each
   token's 4 experts, compiled. One token is a decode step's product; a prompt's
   tokens share experts, 512 of them on a GPU and 64 on the host, where 512 take
   most of a minute a call. On the host, one token's product also runs eagerly.
   Zero codes: the product's cost does not depend on their values. Each operand
   is copied to storage of its own, as a model's are: a constant is one element
   seen at every index, which a compiled call folds. *)

let experts = 32
and per_token = 4
and outputs = 5760
and inputs = 2880

let compiled f = Rune.jit Nx.Ptree.(tensor @-> tensor @-> returns tensor) f
let eager f = f

let routed ~run ~tokens device =
  let place t = Nx.place (Nx.Placement.on device) (Nx.copy t) in
  let w =
    Nx_quant.mxfp4
      ~scales:(place (Nx.full Nx.uint8 [| experts; outputs; inputs / 32 |] 127))
      (place (Nx.zeros Nx.uint8 [| experts; outputs; inputs / 2 |]))
  in
  (* Each token's experts, distinct, spread over all of them. *)
  let ids =
    Nx.init Nx.int64 [| tokens; per_token |] (fun i ->
        Int64.of_int (((i.(0) * 7) + (i.(1) * 8)) mod experts))
  in
  let x = Nx.full Nx.float32 [| tokens; 1; 1; inputs |] 0.5 in
  let f = run (fun ids x -> Nx_quant.apply ~ids w x) in
  let ids = place ids and x = place x in
  ignore (f ids x);
  fun () -> ignore (f ids x)

let synchronize device = Nx_device.synchronize (Nx.Device.memory device)

(* Dense quantised products, one per format, at a decode step's shape: one token
   by a [[| 4096; 4096 |]] projection, compiled. Zero bytes, copied to storage
   of their own, as the routed product's are. *)

let dense = 4096

let formats =
  let zeros last place = place (Nx.zeros Nx.uint8 [| dense; last |]) in
  [
    ( "mxfp4",
      fun place ->
        Nx_quant.mxfp4
          ~scales:(zeros (dense / 32) place)
          (zeros (dense / 2) place) );
    ("q8_0", fun place -> Nx_quant.q8_0 (zeros (dense / 32 * 34) place));
    ("q4_k", fun place -> Nx_quant.q4_k (zeros (dense / 256 * 144) place));
    ("q6_k", fun place -> Nx_quant.q6_k (zeros (dense / 256 * 210) place));
  ]

let linear make device =
  let place t = Nx.place (Nx.Placement.on device) (Nx.copy t) in
  let w = make place in
  let f = Rune.jit' (Nx_quant.apply w) in
  let x = place (Nx.full Nx.float32 [| 1; dense |] 0.5) in
  ignore (f x);
  fun () -> ignore (f x)

(* A case of [product] on the device [device ()] opens, each call
   synchronized. *)
let on device name product =
  {
    name;
    setup =
      (fun () ->
        let d = device () in
        let call = product d in
        fun () ->
          call ();
          synchronize d);
  }

let routed_on device ~run ~tokens suffix =
  on device
    (Printf.sprintf "routed product, %d tokens%s" tokens suffix)
    (routed ~run ~tokens)

(* The routed products of each count of [tokens], then the dense ones. *)
let products ~tokens device =
  List.map (fun tokens -> routed_on device ~run:compiled ~tokens "") tokens
  @ List.map
      (fun (format, make) ->
        on device (Printf.sprintf "%s product, 1 token" format) (linear make))
      formats

let host =
  let device () = Nx.Device.host in
  products ~tokens:[ 1; 64 ] device
  @ [ routed_on device ~run:eager ~tokens:1 ", eager" ]

let metal =
  match Metal.device with
  | None -> []
  | Some device -> [ ("metal", products ~tokens:[ 1; 512 ] device) ]

(* The GPU: CUDA's first, else AMD's first. *)
let gpu =
  lazy (match Nx_cuda.get 0 with Ok d -> d | Error _ -> Nx_amd.device 0)

(* Whether a [vendor] GPU opens, asked of a fresh process: a GPU's driver must
   not be initialized before the fork that isolates a case. *)
let opens vendor =
  let exe = Sys.executable_name in
  let pid =
    Unix.create_process exe
      [| exe; "--" ^ vendor |]
      Unix.stdin Unix.stdout Unix.stderr
  in
  match Unix.waitpid [] pid with _, Unix.WEXITED 0 -> true | _ -> false

let quant () =
  let gpu =
    match List.find_opt opens [ "cuda"; "amd" ] with
    | None -> []
    | Some vendor ->
        [ (vendor, products ~tokens:[ 1; 512 ] (fun () -> Lazy.force gpu)) ]
  in
  (("host", host) :: metal) @ gpu

let bench c =
  Thumper.bench_with_setup ~setup:c.setup c.name (fun call -> call ())

let () =
  match Array.to_list Sys.argv with
  | [ _; "--warm" ] ->
      List.iter (fun c -> c.setup () ()) (gpt2 @ List.concat_map snd (quant ()))
  | [ _; "--cuda" ] -> exit (if Result.is_ok (Nx_cuda.get 0) then 0 else 1)
  | [ _; "--amd" ] -> exit (if Result.is_ok (Nx_amd.get 0) then 0 else 1)
  | _ ->
      (* The eager routed product takes over half a second a call, so its trial
         outlasts the default deadline. *)
      Thumper.run "kaun_decode"
        ~config:Thumper.Config.(default |> deadline 120.)
        ~budgets:[ Thumper.Budget.no_slower_than 0.05 ]
        [
          Thumper.group "Gpt2" (List.map bench gpt2);
          Thumper.group "Quant"
            (List.map
               (fun (name, cases) -> Thumper.group name (List.map bench cases))
               (quant ()));
        ]
      |> exit
