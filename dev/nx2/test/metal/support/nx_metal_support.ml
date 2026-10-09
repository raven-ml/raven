(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf

(* The machine's GPU lock *)

external lock : string -> string -> int = "nx_metal_test_lock"

let gpu_lock = "/tmp/raven-rig-gpu.lock"

(* The longest wait for the lock, in seconds: the machine's suites, from every
   checkout and user, take it in turn. *)
let gpu_wait = 300

let holder () =
  match In_channel.with_open_bin gpu_lock In_channel.input_all with
  | note -> String.trim note
  | exception Sys_error _ -> "a process that left no note"

(* [lock] naps 100 ms each time it is refused. *)
let rec take refused =
  match lock gpu_lock Sys.executable_name with
  | 0 -> ()
  | -1 when refused < gpu_wait * 10 -> take (refused + 1)
  | -1 ->
      failwith
        (strf "%s: still held after %d s, by %s" gpu_lock gpu_wait (holder ()))
  | errno -> failwith (strf "%s: errno %d" gpu_lock errno)

let hold_gpu () =
  if
    Sys.getenv_opt "RIG_GPU_LOCK_HELD" = None
    && Sys.file_exists "/System/Library/Frameworks/Metal.framework"
  then take 0

(* Devices *)

external metallib : unit -> string = "nx_metal_test_metallib"
external kernels : unit -> string array = "nx_metal_test_kernels"
external contents : nativeint -> int = "nx_metal_test_contents"

external view_at :
  ('a, 'b) Bigarray.kind ->
  int ->
  int ->
  ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t = "nx_metal_test_view"

type bytes_ba =
  (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

type pipelines =
  (int64, Bigarray.int64_elt, Bigarray.c_layout) Bigarray.Array1.t

external fill : unit -> nativeint = "nx_metal_test_fill"

external arg : nativeint -> pipelines -> string -> bytes_ba
  = "nx_metal_test_arg"

external span : bytes_ba -> int = "nx_metal_test_span"

external record :
  int -> int * int * int -> int * int * int -> string -> int -> string
  = "nx_metal_test_launch"

external harness_threads : unit -> int = "nx_metal_test_threads"

let kernels = kernels ()
let harness_threads = harness_threads ()

type t = {
  rig : Rig.t;
  image : Rig.Image.t;
  split : nativeint;
  pipelines : pipelines;
}

let get = function Ok x -> x | Error why -> failwith why

let open_ () =
  if Rig_metal.count () = 0 then None
  else
    let rig =
      get
        (Rig.open_
           (module Rig_metal)
           ~name:"METAL"
           (fun () -> Rig_metal.open_ 0))
    in
    let cap = Option.get (Rig.capability rig Rig_metal_abi.key) in
    let image = get (Rig.Image.load rig (metallib ())) in
    let pipelines =
      Bigarray.(Array1.create int64 c_layout (Array.length kernels))
    in
    Bigarray.Array1.fill pipelines 0L;
    Some { rig; image; split = cap.split; pipelines }

let rig t = t.rig

(* Operands *)

type operand = { buf : Rig.Buffer.t; host : int; bytes : int }

let operand t n =
  let buf = Rig.Buffer.create t.rig n in
  {
    buf;
    host = contents (Rig.Buffer.handle buf) + Rig.Buffer.offset buf;
    bytes = n;
  }

let address o = Rig.Buffer.address o.buf

let view k o =
  let v = view_at k o.host (o.bytes / Bigarray.kind_size_in_bytes k) in
  Gc.finalise_last (fun () -> ignore (Sys.opaque_identity o)) v;
  v

(* Runs *)

(* A run's records and the kernels they launch, by index. *)
type run = { records : string; kernels : int list }

let index k =
  match Array.find_index (String.equal k) kernels with
  | Some i -> i
  | None -> invalid_arg (strf "Nx_metal_support.launch: no kernel %S" k)

let threads = (harness_threads, 1, 1)

let launch ?(groups = (1, 1, 1)) ?(threads = threads) k ~addrs ~words =
  let k = index k in
  let n = (8 * List.length addrs) + (4 * List.length words) in
  let b = Bytes.make ((n + 7) / 8 * 8) '\000' in
  List.iteri (fun i a -> Bytes.set_int64_le b (8 * i) (Int64.of_int a)) addrs;
  let at = 8 * List.length addrs in
  List.iteri
    (fun i w -> Bytes.set_int32_le b (at + (4 * i)) (Int32.of_int w))
    words;
  let records =
    record k groups threads (Bytes.unsafe_to_string b) (List.length addrs)
  in
  { records; kernels = [ k ] }

let seq rs =
  {
    records = String.concat "" (List.map (fun r -> r.records) rs);
    kernels = List.concat_map (fun r -> r.kernels) rs;
  }

let prepare t run =
  (* A fill resolves nothing: each pipeline is made before the submission. *)
  let entry k =
    if t.pipelines.{k} = 0L then
      t.pipelines.{k} <-
        Int64.of_int (Option.get (Rig.Image.entry t.image kernels.(k)))
  in
  List.iter entry run.kernels;
  let a = arg t.split t.pipelines run.records in
  let work =
    Rig.Submission.Fill
      {
        fill = fill ();
        arg = Rig.Buffer.of_bigarray a;
        ring_units = 0;
        segment_bytes = 0;
      }
  in
  let s =
    Rig.Submission.make ~reads:0 ~writes:0 t.rig
      [| { queue = "COMPUTE:0"; after = [||]; work } |]
  in
  fun () ->
    let p = Rig.submit s ~reads:[||] ~writes:[||] ~waits:[||] in
    Rig.wait t.rig (Rig.Point.value p);
    span a

let run t r = prepare t r ()
let groups n = ((n + harness_threads - 1) / harness_threads, 1, 1)

(* Operand values *)

let drawn : type v s. (v, s) Nx_array.Dtype.t -> bool = function
  | Float64 | Float32 | Float16 | Bfloat16 | Int64 | Uint64 | Int32 | Uint32
  | Int16 | Uint16 | Int8 | Uint8 ->
      true
  | _ -> false

let generate ?(spread = 0) t o dt n ~seed =
  if not (drawn dt) then
    invalid_arg
      (strf "Nx_metal_support.generate: %s is not drawn"
         (Nx_array.Dtype.name dt));
  let words = [ n; Nx_array.Dtype.code dt; seed; spread ] in
  ignore
    (run t (launch "generate" ~groups:(groups n) ~addrs:[ address o ] ~words))

(* Probes *)

type floats = (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t
type words = (int32, Bigarray.int32_elt, Bigarray.c_layout) Bigarray.Array1.t
type counts = int * int * int

external probe_contract : int -> floats -> floats -> floats -> counts
  = "nx_metal_test_contract"

external probe_div_sqrt : int -> floats -> floats -> floats -> counts
  = "nx_metal_test_div_sqrt"

external probe_half : words -> words -> words -> words -> counts
  = "nx_metal_test_half"

external probe_codec : int -> words -> words -> words -> words -> counts
  = "nx_metal_test_codec"

let probe_contract ?(show = 0) = probe_contract show
let probe_div_sqrt ?(show = 0) = probe_div_sqrt show
let probe_codec dt = probe_codec (Nx_array.Dtype.code dt)

let probe ?(dtype = 0) t kernel in_ ~which n =
  let out = operand t (4 * n) in
  let addrs = [ address out; address in_ ] in
  ignore
    (run t
       (launch kernel ~groups:(groups n) ~addrs ~words:[ n; which; dtype; 0 ]));
  out
