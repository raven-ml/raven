(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* raven's operations at TESSERA's shapes: every product of the four v2 students
   and of the 2B teacher, their attention, layer norm and softmax pooling, and
   the products, attention and GRU pooling of a training step. A pixel is a
   sequence of T observations, so a product has pixels × T rows. The students
   run in float32, the teacher in bfloat16; a training case runs the forward
   pass and the backward one, a pullback of a drawn cotangent.

   A case runs on the device TESSERA_DEVICE names: the host's eager kernels
   (cpu), or compiled by [Rune.jit] on the host (cpu-jit), a CUDA, NV, AMD or
   Metal GPU. Its inputs are normal draws made where it computes, and each call
   waits for its results. [rates] reports each case's time per call with its
   rate in TFLOP/s and GB/s. A forward and backward pass counts three forward
   passes of FLOPs, and its bytes are those of the arrays it reads and writes
   once. *)

let f32 = Nx.float32
let bf16 = Nx.bfloat16
let fl = Float.of_int

(* Targets *)

(* Where a case's arrays lie, and whether its calls compile. *)
type 'd target = { at : 'd Nx.Placement.t; jit : bool }

let compile t s f = if t.jit then Rune.jit s f else f

(* Normal draws of [shape] at [t], made there. *)
let draw t dt seed shape =
  let key = Nx.Rng.place t.at (Nx.Rng.key seed) in
  let normal key = Nx.Rng.normal_at t.at ~key dt shape in
  compile t Nx.Ptree.(Nx.Rng.ptree @-> returns tensor) normal key

let wait s y = Nx.Ptree.fold s (fun _ x () -> Nx.sync x) y ()

(* A case's call at any target, made from its inputs. *)
type make = { make : 'd. 'd target -> unit -> unit }

type case = {
  name : string;
  shape : string; (* the fields shapes.json gives the case *)
  flops : float;
  bytes : float;
  call : make;
}

(* Precisions *)

(* Tf32 stores float32 and lets products round their operands to TF32 on
   NVIDIA's tensor cores, a tolk setting. *)
type precision = F32 | Tf32 | Bf16
type pass = Fwd | Fwd_bwd

let precision_name = function F32 -> "f32" | Tf32 -> "tf32" | Bf16 -> "bf16"
let pass_name = function Fwd -> "fwd" | Fwd_bwd -> "fwd+bwd"
let size = function F32 | Tf32 -> 4. | Bf16 -> 2.
let passes = function Fwd -> 1. | Fwd_bwd -> 3.

(* The backward pass reads and writes as many arrays as the forward one. *)
let traffic = function Fwd -> 1. | Fwd_bwd -> 2.

let precise p (m : make) =
  match p with
  | F32 | Bf16 -> m
  | Tf32 ->
      let make t =
        let call = m.make t in
        fun () -> Tolk.Setting.(context [ B (allow_tf32, true) ] call)
      in
      { make }

(* Products *)

let product (type s) (dt : (float, s) Nx.dtype) ~m ~k ~n pass =
  let make t =
    let x = draw t dt 1 [| m; k |] and w = draw t dt 2 [| k; n |] in
    match pass with
    | Fwd ->
        let f =
          compile t Nx.Ptree.(tensor @-> tensor @-> returns tensor) Nx.matmul
        in
        fun () -> Nx.sync (f x w)
    | Fwd_bwd ->
        let g = draw t dt 3 [| m; n |] in
        let step x w g =
          let mm (x, w) = Nx.matmul x w in
          let y, back =
            Rune.vjp Nx.Ptree.(pair tensor tensor) Nx.Ptree.tensor mm (x, w)
          in
          let dx, dw = back g in
          (y, dx, dw)
        in
        let r = Nx.Ptree.(triple tensor tensor tensor) in
        let f =
          compile t Nx.Ptree.(tensor @-> tensor @-> tensor @-> returns r) step
        in
        fun () -> wait r (f x w g)
  in
  { make }

let gemm name ~m ~k ~n ?(pass = Fwd) p =
  let make =
    match p with
    | F32 | Tf32 -> product f32 ~m ~k ~n pass
    | Bf16 -> product bf16 ~m ~k ~n pass
  in
  {
    name;
    shape =
      Printf.sprintf "op=gemm M=%d K=%d N=%d dtype=%s pass=%s" m k n
        (precision_name p) (pass_name pass);
    flops = 2. *. fl m *. fl k *. fl n *. passes pass;
    bytes = size p *. fl ((m * k) + (k * n) + (m * n)) *. traffic pass;
    call = precise p make;
  }

(* Attention *)

(* [batch] pixels of [heads] heads over [len] steps, no mask. With [bias], each
   key's score gains a value of its own, standing for the log of its
   multiplicity when duplicated observations are computed once. *)
let attend (type s) (dt : (float, s) Nx.dtype) ~batch ~heads ~len ~head_dim
    ~bias pass =
  let make t =
    let shape = [| batch; heads; len; head_dim |] in
    let q = draw t dt 1 shape
    and k = draw t dt 2 shape
    and v = draw t dt 3 shape in
    let m = draw t f32 4 [| batch; 1; 1; len |] in
    let out m (q, k, v) =
      let score = if bias then Some (fun _ s -> Nx.add s m) else None in
      fst (Nx.attention ?score ~acc:f32 ~kv:(Nx.Dense { k; v }) q)
    in
    let qkv = Nx.Ptree.(triple tensor tensor tensor) in
    match pass with
    | Fwd ->
        let s = Nx.Ptree.(tensor @-> qkv @-> returns tensor) in
        let f = compile t s out in
        fun () -> Nx.sync (f m (q, k, v))
    | Fwd_bwd ->
        let g = draw t dt 5 shape in
        let step m qkv' g =
          let o, back = Rune.vjp qkv Nx.Ptree.tensor (out m) qkv' in
          (o, back g)
        in
        let r = Nx.Ptree.(pair tensor qkv) in
        let f =
          compile t Nx.Ptree.(tensor @-> qkv @-> tensor @-> returns r) step
        in
        fun () -> wait r (f m (q, k, v) g)
  in
  { make }

let heads = 4

let attention name ~batch ~len ~head_dim ?(bias = false) ?(pass = Fwd) p =
  let make =
    match p with
    | F32 | Tf32 -> attend f32 ~batch ~heads ~len ~head_dim ~bias pass
    | Bf16 -> attend bf16 ~batch ~heads ~len ~head_dim ~bias pass
  in
  (* The elements of q, k, v and o; [q kᵀ] and its product with [v] take
     [2 len] FLOPs for each element of q and of o. *)
  let qkvo = 4 * batch * heads * len * head_dim in
  {
    name;
    shape =
      Printf.sprintf
        "op=attention batch=%d heads=%d T=%d head_dim=%d bias=%b dtype=%s \
         pass=%s"
        batch heads len head_dim bias (precision_name p) (pass_name pass);
    flops = fl qkvo *. fl len *. passes pass;
    bytes = ((size p *. fl qkvo) +. (4. *. fl (batch * len))) *. traffic pass;
    call = precise p make;
  }

(* Norms and pooling *)

let layernorm name ~rows ~d =
  let make t =
    let x = draw t f32 1 [| rows; d |] in
    let p = Kaun.Layer_norm.make ~dim:d f32 t.at in
    let ln = Nx.Ptree.instantiate (module Kaun.Layer_norm) in
    let f =
      compile t
        Nx.Ptree.(ln @-> tensor @-> returns tensor)
        (fun p x -> Kaun.Layer_norm.apply p x)
    in
    fun () -> Nx.sync (f p x)
  in
  {
    name;
    shape = Printf.sprintf "op=layernorm rows=%d d=%d dtype=f32 pass=fwd" rows d;
    flops = 0.;
    bytes = 8. *. fl (rows * d);
    call = { make };
  }

(* The sum over steps of [x], [batch; len; d], weighed by the softmax of the
   steps' scores. The scores are the pool_query product's. *)
let softmax_pool name ~batch ~len ~d =
  let make t =
    let x = draw t f32 1 [| batch; len; d |]
    and s = draw t f32 2 [| batch; len; 1 |] in
    let pool s x = Nx.sum ~axes:[ 1 ] (Nx.mul (Nx.softmax ~axes:[ 1 ] s) x) in
    let f = compile t Nx.Ptree.(tensor @-> tensor @-> returns tensor) pool in
    fun () -> Nx.sync (f s x)
  in
  {
    name;
    shape =
      Printf.sprintf "op=softmax_pool batch=%d T=%d d=%d dtype=f32 pass=fwd"
        batch len d;
    flops = 0.;
    bytes = 4. *. fl ((batch * len * (d + 1)) + (batch * d));
    call = { make };
  }

(* v1's pooling: a GRU runs over the steps of the encoder's output [x], of
   shape [[batch; len; d]]. Its reset gate applies before the candidate's
   product: [tanh (W_ih x + W_hh (r ∘ h) + b_h)]. Its states, normalised,
   score each step, and the softmax of the scores weighs [x]. The input
   products of the three gates are one product over every step, the hidden
   products of the reset and update gates one per step. *)
module Gru_pool = struct
  type ('a, 'd) t = {
    input : ('a, 'd) Kaun.Linear.t;
    hidden : ('a, 'd) Kaun.Linear.t;
    candidate : ('a, 'd) Kaun.Linear.t;
    norm : ('a, 'd) Kaun.Layer_norm.t;
    query : ('a, 'd) Kaun.Linear.t;
  }

  let walk c p =
    let open Nx.Ptree.Walk in
    let input = field c "input" Kaun.Linear.walk p.input in
    let hidden = field c "hidden" Kaun.Linear.walk p.hidden in
    let candidate = field c "candidate" Kaun.Linear.walk p.candidate in
    let norm = field c "norm" Kaun.Layer_norm.walk p.norm in
    let query = field c "query" Kaun.Linear.walk p.query in
    { input; hidden; candidate; norm; query }

  let init d =
    let key = Nx.Rng.split ~n:4 (Nx.Rng.key 6) in
    let linear ?bias i o =
      Kaun.Linear.make ?bias ~key:key.(i) ~inputs:d ~outputs:o f32
    in
    {
      input = linear 0 (3 * d);
      hidden = linear ~bias:false 1 (2 * d);
      candidate = linear ~bias:false 2 d;
      norm = Kaun.Layer_norm.init ~dim:d Nx.Host.on;
      query = linear 3 1;
    }

  let apply p x =
    let open Kaun in
    let d = (Nx.shape x).(2) in
    let gates i j g = Nx.slice [ Nx.A; Nx.R (i * d, j * d) ] g in
    let step h g =
      let rz = Nx.sigmoid (Nx.add (gates 0 2 g) (Linear.apply p.hidden h)) in
      let r = gates 0 1 rz and z = gates 1 2 rz in
      let c = Linear.apply p.candidate (Nx.mul r h) in
      let h = Nx.lerp h (Nx.tanh (Nx.add (gates 2 3 g) c)) z in
      (h, h)
    in
    let gs = Nx.moveaxis 1 0 (Linear.apply p.input x) in
    let h0 = Nx.zeros_at (Nx.placement x) f32 [| (Nx.shape x).(0); d |] in
    let _, hs = Rune.scan' step ~init:h0 gs in
    let s =
      Linear.apply p.query (Layer_norm.apply p.norm (Nx.moveaxis 0 1 hs))
    in
    Nx.sum ~axes:[ 1 ] (Nx.mul (Nx.softmax ~axes:[ 1 ] s) x)
end

let gru_pool name ~batch ~len ~d =
  let make (type d) (t : d target) =
    let x = draw t f32 1 [| batch; len; d |]
    and g = draw t f32 2 [| batch; d |] in
    let gru = Nx.Ptree.instantiate (module Gru_pool) in
    let p = Nx.Ptree.place (module Gru_pool) t.at (Gru_pool.init d) in
    let step x p g =
      let apply (x, p) = Gru_pool.apply p x in
      let y, back =
        Rune.vjp Nx.Ptree.(pair tensor gru) Nx.Ptree.tensor apply (x, p)
      in
      (y, back g)
    in
    let r = Nx.Ptree.(pair tensor (pair tensor gru)) in
    let f = compile t Nx.Ptree.(tensor @-> gru @-> tensor @-> returns r) step in
    fun () -> wait r (f x p g)
  in
  {
    name;
    shape =
      Printf.sprintf
        "op=gru_pool batch=%d T=%d input=%d hidden=%d dtype=f32 pass=fwd+bwd"
        batch len d d;
    (* The six d × d products a step, on each step of each pixel. *)
    flops = 12. *. fl (d * d) *. fl (batch * len) *. passes Fwd_bwd;
    bytes = 4. *. fl ((batch * len * d) + (batch * d)) *. traffic Fwd_bwd;
    call = { make };
  }

(* Cases *)

(* A v2 student: post-norm encoder layers of 4 heads, the same for S2's 10 bands
   and S1's 2, each modality pooled, the two concatenated and reduced to a 128-d
   embedding. Inference runs chunks of 4096 pixels, at T 40 and 64 (S2's and
   S1's typical lengths) and 256 (the cap). *)
type student = { model : string; d : int; ffn : int }

let students =
  [
    { model = "nano"; d = 144; ffn = 384 };
    { model = "small"; d = 256; ffn = 1024 };
    { model = "medium"; d = 440; ffn = 1792 };
    { model = "large"; d = 640; ffn = 2560 };
  ]

let s2_bands = 10
let s1_bands = 2
let embedding = 128

let student ~pixels s =
  let d = s.d in
  let at t =
    let name op = Printf.sprintf "%s/%s/T%d" s.model op t and m = pixels * t in
    let self_attention ?bias name =
      attention ?bias name ~batch:pixels ~len:t ~head_dim:(d / heads) F32
    in
    [
      gemm (name "embed_s2") ~m ~k:s2_bands ~n:d F32;
      gemm (name "embed_s1") ~m ~k:s1_bands ~n:d F32;
      gemm (name "embed2") ~m ~k:d ~n:d F32;
      gemm (name "in_proj") ~m ~k:d ~n:(3 * d) F32;
      gemm (name "out_proj") ~m ~k:d ~n:d F32;
      gemm (name "ffn1") ~m ~k:d ~n:s.ffn F32;
      gemm (name "ffn2") ~m ~k:s.ffn ~n:d F32;
      gemm (name "pool_query") ~m ~k:d ~n:1 F32;
      self_attention (name "attention");
      self_attention ~bias:true (name "attention" ^ "/bias");
      layernorm (name "layernorm") ~rows:m ~d;
      softmax_pool (name "softmax_pool") ~batch:pixels ~len:t ~d;
    ]
  in
  List.concat_map at [ 40; 64; 256 ]
  @ [
      gemm (s.model ^ "/reduce1") ~m:pixels ~k:(2 * d) ~n:(4 * d) F32;
      gemm (s.model ^ "/reduce2") ~m:pixels ~k:(4 * d) ~n:embedding F32;
    ]

(* The 2B teacher in bfloat16: d 4096, heads of 1024, FFN 16384, its query, key
   and value projections apart (one case stands for the three), two tokens a
   pixel through the fusion layers, a 1024-d embedding. *)
let teacher ~pixels =
  let d = 4096 and ffn = 16384 in
  let at t =
    let name op = Printf.sprintf "teacher/%s/T%d" op t and m = pixels * t in
    [
      gemm (name "embed2") ~m ~k:d ~n:d Bf16;
      gemm (name "q_proj") ~m ~k:d ~n:d Bf16;
      gemm (name "out_proj") ~m ~k:d ~n:d Bf16;
      gemm (name "ffn1") ~m ~k:d ~n:ffn Bf16;
      gemm (name "ffn2") ~m ~k:ffn ~n:d Bf16;
      attention (name "attention") ~batch:pixels ~len:t ~head_dim:(d / heads)
        Bf16;
    ]
  in
  List.concat_map at [ 40; 64 ]
  @ [
      gemm "teacher/fusion_ffn1" ~m:(2 * pixels) ~k:d ~n:(2 * d) Bf16;
      gemm "teacher/reduce1" ~m:pixels ~k:(2 * d) ~n:(4 * d) Bf16;
      gemm "teacher/reduce2" ~m:pixels ~k:(4 * d) ~n:1024 Bf16;
    ]

(* A training step's products, forward and backward, at each precision. *)
let trained name ~m ~k ~n ps =
  List.map
    (fun p -> gemm (name ^ "/" ^ precision_name p) ~m ~k ~n ~pass:Fwd_bwd p)
    ps

(* Distilling the medium student: 4096 pixels of 40 steps. *)
let distillation ~pixels =
  let m = pixels * 40 and d = 440 and ffn = 1792 and ps = [ Bf16; F32 ] in
  trained "train/medium/in_proj/T40" ~m ~k:d ~n:(3 * d) ps
  @ trained "train/medium/ffn1/T40" ~m ~k:d ~n:ffn ps
  @ trained "train/medium/ffn2/T40" ~m ~k:ffn ~n:d ps
  @ [
      attention "train/medium/attention/T40/bf16" ~batch:pixels ~len:40
        ~head_dim:(d / heads) ~bias:true ~pass:Fwd_bwd Bf16;
    ]

(* v1.0's Barlow Twins step on one GPU: 2048 pixels of 40 steps, d 512, FFN
   4096, the 128-d representation projected to 16384 and through 16384 × 16384
   layers. *)
let barlow_twins ~pixels =
  let m = pixels * 40 and d = 512 and ffn = 4096 and wide = 16384 in
  let ps = [ F32; Tf32; Bf16 ] in
  trained "train/v1/in_proj/T40" ~m ~k:d ~n:(3 * d) ps
  @ trained "train/v1/ffn1/T40" ~m ~k:d ~n:ffn ps
  @ trained "train/v1/ffn2/T40" ~m ~k:ffn ~n:d ps
  @ trained "train/v1/projector_in" ~m:pixels ~k:128 ~n:wide ps
  @ trained "train/v1/projector_hidden" ~m:pixels ~k:wide ~n:wide ps
  @ [
      attention "train/v1/attention/T40/bf16" ~batch:pixels ~len:40
        ~head_dim:(d / heads) ~pass:Fwd_bwd Bf16;
      gru_pool "train/v1/gru_pool/T40" ~batch:pixels ~len:40 ~d;
    ]

(* Every case, with each count of pixels divided by [shrink]. *)
let cases ~shrink =
  let pixels n = max 1 (n / shrink) in
  List.concat_map (student ~pixels:(pixels 4096)) students
  @ teacher ~pixels:(pixels 512)
  @ distillation ~pixels:(pixels 4096)
  @ barlow_twins ~pixels:(pixels 2048)

(* Devices *)

let gpus =
  [
    ("cuda", fun () -> Rig.open_ (Rig_cuda.gpu 0));
    ("nv", fun () -> Rig.open_ (Rig_nv_nvidia.gpu 0));
    ("amd", fun () -> Rig.open_ (Rig_amd_amdgpu.gpu 0));
    ("metal", fun () -> Rig.open_ (Rig_metal.gpu 0));
  ]

let devices = "cpu" :: "cpu-jit" :: List.map fst gpus

(* [call device c] is [c]'s call on [device], its inputs made and the device
   opened in this process. *)
let call device c =
  match device with
  | "cpu" -> c.call.make { at = Nx.Host.on; jit = false }
  | "cpu-jit" -> c.call.make { at = Nx.Host.on; jit = true }
  | gpu -> (
      match (List.assoc gpu gpus) () with
      | Error e -> failwith e
      | Ok g ->
          let module D = (val Nx.devices [ g ]) in
          c.call.make { at = D.on; jit = true })

(* Whether [gpu] opens, asked of a fresh process: a GPU's driver must not be
   initialised before the fork that isolates a case. *)
let opens gpu =
  let exe = Sys.executable_name in
  let pid =
    Unix.create_process exe [| exe; "opens"; gpu |] Unix.stdin Unix.stdout
      Unix.stderr
  in
  match Unix.waitpid [] pid with _, Unix.WEXITED 0 -> true | _ -> false

(* The first GPU that opens, else the host. *)
let auto () =
  match List.find_opt (fun (g, _) -> opens g) gpus with
  | Some (g, _) -> g
  | None -> "cpu"

(* Runs *)

(* A case's trial makes its inputs and compiles its call in the worker, with one
   uncounted call: compiling may start domains, after which thumper cannot fork.
   The largest case reads and writes 14 GB, and a CPU call takes seconds. *)
let bench device c =
  Thumper.bench_with_setup c.name
    ~setup:(fun () ->
      let f = call device c in
      f ();
      f)
    (fun f -> f ())

let config =
  Thumper.Config.(
    default |> deadline 900. |> metrics [ Thumper.Metric.wall_time ])

(* Whether [s] holds [sub]. *)
let contains s sub =
  let n = String.length sub in
  let rec at i =
    i + n <= String.length s && (String.sub s i n = sub || at (i + 1))
  in
  at 0

let selected patterns cs =
  let matches c p = contains c.name p in
  if patterns = [] then cs
  else List.filter (fun c -> List.exists (matches c) patterns) cs

(* Each selected case called once, in this process: a check that every case
   runs, and on a GPU the warm pass that fills tolk's disk cache, which a worker
   forked on the Mac cannot fill. *)
let once device cs =
  List.iter
    (fun c ->
      (call device c) ();
      Gc.full_major ();
      Printf.printf "ok %s\n%!" c.name)
    cs

let median a = a.(Array.length a / 2)

let rates device cs =
  let group = Thumper.group device (List.map (bench device) cs) in
  let run =
    Thumper.measure ~config:Thumper.Config.(config |> samples 11) [ group ]
  in
  List.iter
    (fun c ->
      let id = device ^ "/" ^ c.name in
      match Thumper.Run.samples run ~case:id Thumper.Metric.wall_time with
      | None -> ()
      | Some s ->
          let t = median s in
          Printf.printf "%-40s %10.3f ms %8.2f TFLOP/s %8.1f GB/s\n%!" id
            (t *. 1e3)
            (c.flops /. t /. 1e12)
            (c.bytes /. t /. 1e9))
    cs

let usage =
  "usage: bench_tessera [--shrink K] (cases|once|warm|rates) [PATTERN...]\n\
  \       bench_tessera [check|bless|list] [THUMPER OPTIONS]\n\
   The variable TESSERA_DEVICE names the device: auto (the default, the first\n\
   GPU that opens, else cpu) or one of " ^ String.concat ", " devices
  ^ ".\n--shrink divides each count of pixels by K."

(* The device comes from the environment, so that the commands thumper's report
   hints at run on it too. *)
let device () =
  match Sys.getenv_opt "TESSERA_DEVICE" with
  | None | Some "auto" -> auto ()
  | Some d when List.mem d devices -> d
  | Some d -> failwith (Printf.sprintf "TESSERA_DEVICE: %S is not a device" d)

let () =
  let shrink, args =
    match List.tl (Array.to_list Sys.argv) with
    | "--shrink" :: k :: args -> (int_of_string k, args)
    | args -> (1, args)
  in
  let cs = cases ~shrink in
  match args with
  | [ "opens"; g ] ->
      exit (if Result.is_ok ((List.assoc g gpus) ()) then 0 else 1)
  | [ "cases" ] ->
      List.iter (fun c -> Printf.printf "%s %s\n" c.name c.shape) cs
  | "once" :: patterns -> once (device ()) (selected patterns cs)
  | "warm" :: patterns ->
      let d = device () in
      if d <> "cpu" then once d (selected patterns cs)
  | "rates" :: patterns -> rates (device ()) (selected patterns cs)
  | ("-h" | "--help") :: _ -> print_endline usage
  | _ when shrink <> 1 ->
      (* A baseline's rows are the cases at their own shapes. *)
      prerr_endline usage;
      exit 2
  | args ->
      let d = device () in
      let argv = Array.of_list (Sys.executable_name :: args) in
      exit
        (Thumper.run "tessera" ~config ~argv
           [ Thumper.group d (List.map (bench d) cs) ])
