(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* raven's operations at TESSERA's shapes: every product of the four v2 students
   and of the 2B teacher, their attention, layer norm and softmax pooling. A
   pixel is a sequence of T observations, so a product has pixels × T rows. The
   students run in float32, the teacher in bfloat16.

   A case runs on the device TESSERA_DEVICE names: the host's eager kernels
   (cpu), or compiled by [Rune.jit] on the host (cpu-jit), a CUDA, NV, AMD or
   Metal GPU. Its inputs are normal draws made where it computes, and each call
   waits for its results. [rates] reports each case's time per call with its
   rate in TFLOP/s and GB/s, its bytes those of the arrays it reads and writes
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

type precision = F32 | Bf16

let precision_name = function F32 -> "f32" | Bf16 -> "bf16"
let size = function F32 -> 4. | Bf16 -> 2.

(* Products *)

let product (type s) (dt : (float, s) Nx.dtype) ~m ~k ~n =
  let make t =
    let x = draw t dt 1 [| m; k |] and w = draw t dt 2 [| k; n |] in
    let f =
      compile t Nx.Ptree.(tensor @-> tensor @-> returns tensor) Nx.matmul
    in
    fun () -> Nx.sync (f x w)
  in
  { make }

let gemm name ~m ~k ~n p =
  let make =
    match p with F32 -> product f32 ~m ~k ~n | Bf16 -> product bf16 ~m ~k ~n
  in
  {
    name;
    shape =
      Printf.sprintf "op=gemm M=%d K=%d N=%d dtype=%s pass=fwd" m k n
        (precision_name p);
    flops = 2. *. fl m *. fl k *. fl n;
    bytes = size p *. fl ((m * k) + (k * n) + (m * n));
    call = make;
  }

(* Attention *)

(* [batch] pixels of [heads] heads over [len] steps, no mask. With [bias], each
   key's score gains a value of its own, standing for the log of its
   multiplicity when duplicated observations are computed once. *)
let attend (type s) (dt : (float, s) Nx.dtype) ~batch ~heads ~len ~head_dim
    ~bias =
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
    let f = compile t Nx.Ptree.(tensor @-> qkv @-> returns tensor) out in
    fun () -> Nx.sync (f m (q, k, v))
  in
  { make }

let heads = 4

let attention name ~batch ~len ~head_dim ?(bias = false) p =
  let make =
    match p with
    | F32 -> attend f32 ~batch ~heads ~len ~head_dim ~bias
    | Bf16 -> attend bf16 ~batch ~heads ~len ~head_dim ~bias
  in
  (* The elements of q, k, v and o; [q kᵀ] and its product with [v] take
     [2 len] FLOPs for each element of q and of o. *)
  let qkvo = 4 * batch * heads * len * head_dim in
  {
    name;
    shape =
      Printf.sprintf
        "op=attention batch=%d heads=%d T=%d head_dim=%d bias=%b dtype=%s \
         pass=fwd"
        batch heads len head_dim bias (precision_name p);
    flops = fl qkvo *. fl len;
    bytes = (size p *. fl qkvo) +. (4. *. fl (batch * len));
    call = make;
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

(* Every case, with each count of pixels divided by [shrink]. *)
let cases ~shrink =
  let pixels n = max 1 (n / shrink) in
  List.concat_map (student ~pixels:(pixels 4096)) students
  @ teacher ~pixels:(pixels 512)

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
