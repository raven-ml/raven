(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The jitted decode step of a GPT-2 124M shaped decoder: one token in, one
   token out, through key-value caches, with the state donated and the sampled
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

(* The decoder lies on the host. *)
type tensor = Nx.host Nx.float32_t

type block = {
  ln1 : (tensor, Nx.host) Layer_norm.t;
  attn : (tensor, Nx.host) Attention.t;
  ln2 : (tensor, Nx.host) Layer_norm.t;
  fc : (tensor, Nx.host) Linear.t;
  proj : (tensor, Nx.host) Linear.t;
}

type model = {
  wte : (tensor, Nx.host) Embedding.t;
  wpe : (tensor, Nx.host) Embedding.t;
  blocks : block list;
  ln_f : (tensor, Nx.host) Layer_norm.t;
}

let model () =
  let key = Nx.Rng.key 0 in
  let zeros key ~fan_in ~fan_out dtype shape =
    Nx.copy (Init.zeros key ~fan_in ~fan_out dtype shape)
  in
  let norm () =
    let { Layer_norm.gamma; beta } = Layer_norm.init ~dim:embd Nx.Host.on in
    { Layer_norm.gamma = Nx.copy gamma; beta = Nx.copy beta }
  in
  let linear ~inputs ~outputs =
    Linear.make ~w_init:zeros ~bias_init:zeros ~key ~inputs ~outputs Nx.float32
  in
  let block () =
    {
      ln1 = norm ();
      attn =
        Attention.make ~w_init:zeros ~bias_init:zeros ~key ~embed_dim:embd
          Nx.float32;
      ln2 = norm ();
      fc = linear ~inputs:embd ~outputs:inner;
      proj = linear ~inputs:inner ~outputs:embd;
    }
  in
  {
    wte = Embedding.make ~init:zeros ~key ~vocab ~dim:embd Nx.float32;
    wpe = Embedding.make ~init:zeros ~key ~vocab:positions ~dim:embd Nx.float32;
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
      Attention.Cache.make ~slots ~kv_heads:heads ~head_dim Nx.float32
        Nx.Host.on)

(* [lowered name f] is [f ()], refused when a compiled call in it ran
   evaluated: a compiled row that rune declines to lower would time
   evaluation and look like a regression of the compiler. *)
let lowered name f =
  let before = Rune.evaluations () in
  let y = f () in
  if Rune.evaluations () > before then
    failwith (name ^ ": a compiled call ran evaluated");
  y

(* A decode loop warmed past its two compilations. The returned thunk decodes
   one token at a fixed position in the middle of the cache, from the previous
   call's token and caches: every measured step is in range, whatever the number
   of samples. *)
let decoder params ~len =
  let caches = Nx.Ptree.list (Nx.Ptree.instantiate (module Attention.Cache)) in
  let compiled =
    Rune.jit
      Nx.Ptree.(
        tensor @-> v Cache_index.walk @-> caches
        @-> returns (pair tensor caches))
      (fun token index caches ->
        let seq = (Nx.shape token).(1) in
        let h, caches = cached params caches index token in
        let last = Nx.slice [ A; I (seq - 1) ] h in
        (Nx.reshape [| 1; 1 |] (Nx.argmax ~axis:1 (logits params last)), caches))
  in
  let step token index cs = compiled token index (Nx.Ptree.donate caches cs) in
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
  lowered "decode" advance;
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
   Each operand is copied to storage of its own, as a model's are: a constant is
   one element seen at every index, which a compiled call folds.

   The codes are random bytes from a fixed key, as a checkpoint's look, at the
   scale byte 127, a scale of one. A product's speed depends on its codes: on
   x86 a kernel runs a few percent faster or slower on zero codes than on random
   ones. *)

let experts = 32
and per_token = 4
and outputs = 5760
and inputs = 2880

(* [random shape] is uniformly random bytes of [shape], whose last axis is a
   multiple of 4, the same for every call. *)
let random shape =
  let last = Array.length shape - 1 in
  let words = Array.mapi (fun i n -> if i = last then n / 4 else n) shape in
  Nx.reshape shape
    (Nx.bitcast Nx.uint8 (Nx.Rng.bits ~key:(Nx.Rng.key 42) words))

(* How a product runs: compiled, or eagerly, which only the host does. *)
type run = Compiled | Eager

(* [own at t] is [t] copied to storage of its own, at [at]. *)
let own at t = Nx.place at (Nx.copy t)

let routed ~run ~tokens at =
  let w =
    Nx_quant.place at
      (Nx_quant.mxfp4
         ~scales:
           (Nx.copy (Nx.full Nx.uint8 [| experts; outputs; inputs / 32 |] 127))
         (Nx.copy (random [| experts; outputs; inputs / 2 |])))
  in
  (* Each token's experts, distinct, spread over all of them. *)
  let ids =
    Nx.init Nx.int64 [| tokens; per_token |] (fun i ->
        Int64.of_int (((i.(0) * 7) + (i.(1) * 8)) mod experts))
  in
  let x = Nx.full Nx.float32 [| tokens; 1; inputs |] 0.5 in
  let product ids x =
    Nx.map_segments ~segments:experts ids
      (fun owners rows -> Nx_quant.apply (Nx_quant.take ~axis:0 owners w) rows)
      x
  in
  let f =
    match run with
    | Eager -> product
    | Compiled ->
        Rune.jit Nx.Ptree.(tensor @-> tensor @-> returns tensor) product
  in
  let ids = own at ids and x = own at x in
  (match run with
  | Eager -> ignore (f ids x)
  | Compiled -> ignore (lowered "moe" (fun () -> f ids x)));
  fun () -> Nx.sync (f ids x)

(* Dense quantised products, one per format, at a decode step's shape: one token
   by a [[| 4096; 4096 |]] projection, compiled. MXFP4 is read from a
   checkpoint's bytes and from GGUF's blocks, whose codes a product reads
   through their pairing of values [j] and [j + 16]. Random codes at scales of
   one, copied to storage of their own, as the routed product's are. *)

let dense = 4096

(* A part of a block: fixed bytes, or a number of random ones. *)
type part = Fixed of int array | Random of int

(* The float16 [1.], little-endian. *)
let one = [| 0x00; 0x3c |]

(* [dense] rows of blocks of [per] values, each laid out as [parts], on the host
   in storage of their own. *)
let blocks ~per parts =
  let n = dense / per in
  let part = function
    | Random len -> random [| dense; n; len |]
    | Fixed b ->
        let len = Array.length b in
        Nx.broadcast_to [| dense; n; len |]
          (Nx.create Nx.uint8 [| 1; 1; len |] b)
  in
  let b = Nx.concatenate ~axis:2 (List.map part parts) in
  Nx.copy (Nx.reshape [| dense; -1 |] b)

let formats =
  [
    ( "mxfp4",
      fun () ->
        Nx_quant.mxfp4
          ~scales:(Nx.copy (Nx.full Nx.uint8 [| dense; dense / 32 |] 127))
          (Nx.copy (random [| dense; dense / 2 |])) );
    ( "mxfp4 blocks",
      fun () ->
        Nx_quant.mxfp4_blocks (blocks ~per:32 [ Fixed [| 127 |]; Random 16 ]) );
    ("q8_0", fun () -> Nx_quant.q8_0 (blocks ~per:32 [ Fixed one; Random 32 ]));
    ( "q4_k",
      fun () ->
        Nx_quant.q4_k (blocks ~per:256 [ Fixed one; Fixed one; Random 140 ]) );
    ("q6_k", fun () -> Nx_quant.q6_k (blocks ~per:256 [ Random 208; Fixed one ]));
  ]

let linear make at =
  let w = Nx_quant.place at (make ()) in
  let f = Rune.jit' (Nx_quant.apply w) in
  let x = own at (Nx.full Nx.float32 [| 1; dense |] 0.5) in
  ignore (lowered "linear" (fun () -> f x));
  fun () -> Nx.sync (f x)

(* A product made at a placement of any set: its one call, waited for. *)
type product = { at : 'd. 'd Nx.Placement.t -> unit -> unit }

(* A case of [product] on the set [devices ()] mints, over the devices it opens.
   A GPU's set has no kernels, so its products are compiled. *)
let on devices name product =
  {
    name;
    setup =
      (fun () ->
        let module D = (val devices () : Nx.Devices) in
        product.at D.on);
  }

let routed_on devices ~run ~tokens suffix =
  on devices
    (Printf.sprintf "routed product, %d tokens%s" tokens suffix)
    { at = (fun at -> routed ~run ~tokens at) }

(* The routed products of each count of [tokens], then the dense ones. *)
let products ~tokens devices =
  List.map (fun tokens -> routed_on devices ~run:Compiled ~tokens "") tokens
  @ List.map
      (fun (format, make) ->
        on devices
          (Printf.sprintf "%s product, 1 token" format)
          { at = (fun at -> linear make at) })
      formats

let host =
  let devices () = (module Nx.Host : Nx.Devices) in
  products ~tokens:[ 1; 64 ] devices
  @ [ routed_on devices ~run:Eager ~tokens:1 ", eager" ]

let get_ok = function Ok d -> d | Error e -> failwith e

(* The GPU: CUDA's first, else AMD's first. *)
let gpu =
  lazy
    (match Rig.open_ (Rig_cuda.gpu 0) with
    | Ok d -> d
    | Error _ -> get_ok (Rig.open_ (Rig_amd_amdgpu.gpu 0)))

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

(* Metal's cases where its device opens, asked of a fresh process too. *)
let metal () =
  if opens "metal" then
    [
      ( "metal",
        products ~tokens:[ 1; 512 ] (fun () ->
            Nx.devices [ get_ok (Rig.open_ (Rig_metal.gpu 0)) ]) );
    ]
  else []

let quant () =
  let gpu =
    match List.find_opt opens [ "cuda"; "amd" ] with
    | None -> []
    | Some vendor ->
        [
          ( vendor,
            products ~tokens:[ 1; 512 ] (fun () ->
                Nx.devices [ Lazy.force gpu ]) );
        ]
  in
  (("host", host) :: metal ()) @ gpu

(* A GPU case times its calls only: a GPU buffer's release allocates when the
   collector finalises it, so a call's count depends on the collection's phase.
   Its allocation is measured again once rig's release path allocates
   nothing. *)
let gpu_metrics = [ Thumper.Metric.wall_time ]

let bench c =
  Thumper.bench_with_setup ~setup:c.setup c.name (fun call -> call ())

let () =
  match Array.to_list Sys.argv with
  | [ _; "--warm" ] ->
      List.iter (fun c -> c.setup () ()) (gpt2 @ List.concat_map snd (quant ()))
  | [ _; "--cuda" ] ->
      exit (if Result.is_ok (Rig.open_ (Rig_cuda.gpu 0)) then 0 else 1)
  | [ _; "--amd" ] ->
      exit (if Result.is_ok (Rig.open_ (Rig_amd_amdgpu.gpu 0)) then 0 else 1)
  | [ _; "--metal" ] ->
      exit (if Result.is_ok (Rig.open_ (Rig_metal.gpu 0)) then 0 else 1)
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
               (fun (name, cases) ->
                 let metrics =
                   if name = "host" then None else Some gpu_metrics
                 in
                 Thumper.group ?metrics name (List.map bench cases))
               (quant ()));
        ]
      |> exit
