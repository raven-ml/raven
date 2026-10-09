(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf

(* The machine's GPU lock *)

let hold_gpu () =
  if Sys.file_exists "/System/Library/Frameworks/Metal.framework" then
    Rig_gpu_lock.hold ()

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
external library : unit -> string * string array = "nx_metal_test_library"

(* The run's kernels, by index: nx.metal's, by their enum, then the harness's,
   by theirs. *)
let library_metallib, library_kernels = library ()
let harness_kernels = kernels ()
let kernels = Array.append library_kernels harness_kernels
let harness_threads = harness_threads ()

type t = {
  rig : Rig.t;
  library : Rig.Image.t;
  harness : Rig.Image.t;
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
    let library = get (Rig.Image.load rig library_metallib) in
    let harness = get (Rig.Image.load rig (metallib ())) in
    let pipelines =
      Bigarray.(Array1.create int64 c_layout (Array.length kernels))
    in
    Bigarray.Array1.fill pipelines 0L;
    Some { rig; library; harness; split = cap.split; pipelines }

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

(* A run's records, the kernels they launch, by index, and the memory they
   address beyond the caller's operands, such as their scratch. *)
type run = { records : string; kernels : int list; holds : operand list }

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
  { records; kernels = [ k ]; holds = [] }

let seq rs =
  {
    records = String.concat "" (List.map (fun r -> r.records) rs);
    kernels = List.concat_map (fun r -> r.kernels) rs;
    holds = List.concat_map (fun r -> r.holds) rs;
  }

let prepare t r =
  (* A fill resolves nothing: each pipeline is made before the submission. *)
  let entry k =
    let image =
      if k < Array.length library_kernels then t.library else t.harness
    in
    if t.pipelines.{k} = 0L then
      t.pipelines.{k} <-
        Int64.of_int (Option.get (Rig.Image.entry image kernels.(k)))
  in
  List.iter entry r.kernels;
  let a = arg t.split t.pipelines r.records in
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
  let run = Rig.Submission.Run.make () in
  fun () ->
    let p = Rig.submit s ~run ~reads:[||] ~writes:[||] ~waits:[||] in
    Rig.wait t.rig (Rig.Point.value p);
    ignore (Sys.opaque_identity r.holds);
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

(* Contract *)

type arg = { o : operand; dtype : int; strides : int * int * int }

let arg o dt strides = { o; dtype = Nx_array.Dtype.code dt; strides }
let arg_operand a = a.o
let gpu a = (address a.o, a.dtype, a.strides)
let cpu a = (a.o.host, a.dtype, a.strides)

(* batch, m, n, k, the accumulator's dtype code, and how many outputs a
   check reads, 0 for all. *)
type dims = int * int * int * int * int * int
type view = int * int * (int * int * int)

external plan_contract :
  dims -> view -> view -> view -> view option -> (string * int) option
  = "nx_metal_test_plan_contract"

external rebase : string -> int -> string = "nx_metal_test_rebase"
external record_entries : string -> int array = "nx_metal_test_entries"

external contract_error :
  dims -> view -> view -> view -> view option -> float * int
  = "nx_metal_test_contract_error"

external contract_wrong :
  dims -> view -> view -> view -> view option -> int * int
  = "nx_metal_test_contract_wrong"

let with_acc ?(samples = 0) acc (batch, m, n, k) =
  (batch, m, n, k, acc, samples)
let float32 = Nx_array.Dtype.(Any Float32)
let code (Nx_array.Dtype.Any dt) = Nx_array.Dtype.code dt

let plan_contract ?init ?(acc = float32) t dims ~a ~b ~out =
  let planned =
    plan_contract
      (with_acc (code acc) dims)
      (gpu a) (gpu b) (gpu out) (Option.map gpu init)
  in
  Fun.flip Option.map planned @@ fun (records, scratch) ->
  let kernels = Array.to_list (record_entries records) in
  if scratch = 0 then { records; kernels; holds = [] }
  else
    let s = operand t scratch in
    { records = rebase records (address s); kernels; holds = [ s ] }

let entries r = List.map (fun k -> kernels.(k)) r.kernels

let contract_error ?init ?samples dims ~a ~b ~out =
  let worst, at =
    contract_error
      (with_acc ?samples (code float32) dims)
      (cpu a) (cpu b) (cpu out) (Option.map cpu init)
  in
  (* windtrap's at_most orders NaN below every number: a NaN worst would
     pass a bound. *)
  if Float.is_nan worst then
    failwith "Nx_metal_support.contract_error: a NaN ratio";
  (worst, at)

let contract_wrong ?init ?samples ~acc dims ~a ~b ~out =
  contract_wrong
    (with_acc ?samples (code acc) dims)
    (cpu a) (cpu b) (cpu out) (Option.map cpu init)
