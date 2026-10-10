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
external harness_kernels : unit -> string array = "nx_metal_test_kernels"
external contents : nativeint -> int = "nx_metal_test_contents"

external view_at :
  ('a, 'b) Bigarray.kind ->
  int ->
  int ->
  ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t = "nx_metal_test_view"

external harness_threads : unit -> int = "nx_metal_test_threads"

let harness_kernels = harness_kernels ()
let harness_threads = harness_threads ()

type t = { rig : Rig.t; harness : Rig.Image.t }

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
    if not (Nx_metal.computes_on rig) then
      failwith "Nx_metal_support.open_: nx.metal does not compute on the GPU";
    Some { rig; harness = get (Rig.Image.load rig (metallib ())) }

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

let buffer o = o.buf

let view k o =
  let v = view_at k o.host (o.bytes / Bigarray.kind_size_in_bytes k) in
  Gc.finalise_last (fun () -> ignore (Sys.opaque_identity o)) v;
  v

(* Runs *)

(* A launch of a harness kernel: its grid, its parameters' bytes, and the
   operands whose addresses are its first words. *)
type launch = {
  kernel : string;
  groups : int * int * int;
  threads : int * int * int;
  params : string;
  addrs : operand list;
}

(* A call of nx.metal's contraction. *)
type call = {
  spec : Nx_kernel.Spec.contract Nx_kernel.Spec.t;
  dst : Nx_array.any;
  ops : Nx_array.any array;
}

type step = Launch of launch | Call of call
type run = step list

let threads = (harness_threads, 1, 1)

let launch ?(groups = (1, 1, 1)) ?(threads = threads) kernel ~addrs ~words =
  if not (Array.mem kernel harness_kernels) then
    invalid_arg (strf "Nx_metal_support.launch: no kernel %S" kernel);
  let at = 8 * List.length addrs in
  let n = at + (4 * List.length words) in
  let b = Bytes.make ((n + 7) / 8 * 8) '\000' in
  List.iteri
    (fun i w -> Bytes.set_int32_le b (at + (4 * i)) (Int32.of_int w))
    words;
  [ Launch { kernel; groups; threads; params = Bytes.to_string b; addrs } ]

let seq = List.concat

(* The submission of the launches [ls], in order, with its run and the buffers
   it writes: each operand a launch addresses, once. Every byte of each block
   is stored here, and the run serves this submission alone. *)
let submission t ls =
  let slots = ref [] in
  let slot o =
    match List.find_index (fun o' -> o'.buf == o.buf) (List.rev !slots) with
    | Some i -> i
    | None ->
        slots := o :: !slots;
        List.length !slots - 1
  in
  let parts =
    List.map
      (fun l ->
        let refs =
          List.mapi
            (fun i o -> { Rig.Submission.at = 8 * i; slot = slot o })
            l.addrs
        in
        {
          Rig.Submission.queue = "COMPUTE:0";
          after = [||];
          work =
            Launch
              {
                image = t.harness;
                kernel = l.kernel;
                params = String.length l.params;
                refs = Array.of_list refs;
              };
        })
      ls
  in
  let writes = Array.of_list (List.rev_map (fun o -> o.buf) !slots) in
  let sub =
    Rig.Submission.make ~reads:0 ~writes:(Array.length writes) t.rig
      (Array.of_list parts)
  in
  let run = Rig.Submission.Run.make () in
  List.iteri
    (fun j l ->
      let at = Rig.Submission.block sub j in
      let gx, gy, gz = l.groups and tx, ty, tz = l.threads in
      Rig.Submission.Run.groups run at gx gy gz;
      Rig.Submission.Run.threads run at tx ty tz;
      Rig.Submission.Run.shared run at 0;
      for q = 0 to (String.length l.params / 8) - 1 do
        Rig.Submission.Run.int64 run at (8 * q)
          (Int64.to_int (String.get_int64_le l.params (8 * q)))
      done)
    ls;
  (sub, run, writes)

let call_contract c =
  match Nx_metal.contract c.spec ~dst:c.dst c.ops with
  | Nx_array.Done -> ()
  | Declined ->
      failwith "Nx_metal_support: nx.metal declines a contraction it computed"
  | r -> Nx_array.refused "Nx_metal.contract" r (c.dst :: Array.to_list c.ops)

(* [r]'s steps as work: runs of launches as one submission each, calls as
   they are. *)
let rec works t = function
  | [] -> []
  | Call c :: r -> `Call c :: works t r
  | Launch _ :: _ as r ->
      let rec split acc = function
        | Launch l :: r -> split (l :: acc) r
        | r -> (List.rev acc, r)
      in
      let ls, r = split [] r in
      `Submit (submission t ls) :: works t r

let prepare t r =
  let ws = works t r in
  fun () ->
    let t0 = Rig.Profile.now () in
    List.iter
      (function
        | `Call c -> call_contract c
        | `Submit (sub, run, writes) ->
            ignore (Rig.submit sub ~run ~reads:[||] ~writes ~waits:[||]))
      ws;
    Rig.wait t.rig (Rig.submitted t.rig);
    ignore (Sys.opaque_identity r);
    Rig.Profile.now () - t0

let run t r = prepare t r ()

let call = function
  | [ Call c ] -> call_contract c
  | _ -> invalid_arg "Nx_metal_support.call: not one contraction"

let issue t ~count = function
  | [ Call c ] ->
      for _ = 1 to count do
        call_contract c
      done;
      Rig.wait t.rig (Rig.submitted t.rig)
  | _ -> invalid_arg "Nx_metal_support.issue: not one contraction"

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
  ignore (run t (launch "generate" ~groups:(groups n) ~addrs:[ o ] ~words))

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
  let addrs = [ out; in_ ] in
  ignore
    (run t
       (launch kernel ~groups:(groups n) ~addrs ~words:[ n; which; dtype; 0 ]));
  out

(* Contract *)

type arg = { o : operand; dtype : int; first : int; strides : int * int * int }

let arg ?(first = 0) o dt strides =
  { o; dtype = Nx_array.Dtype.code dt; first; strides }

let arg_operand a = a.o
let dtype c = Option.get (Nx_array.Dtype.of_code c)

let cpu a =
  let (Nx_array.Dtype.Any d) = dtype a.dtype in
  (a.o.host + (a.first * Nx_array.Dtype.bits d / 8), a.dtype, a.strides)

(* [a] as an array of [shape], its first element [a.first] elements into its
   memory. *)
let array a shape =
  let (Nx_array.Dtype.Any d) = dtype a.dtype in
  let s0, s1, s2 = a.strides in
  let layout =
    Nx_array.Layout.v ~offset:a.first ~strides:[| s0; s1; s2 |] shape
  in
  Nx_array.Any (Nx_array.v d layout a.o.buf)

(* batch, m, n, k, the accumulator's dtype code, and how many outputs a
   check reads, 0 for all. *)
type dims = int * int * int * int * int * int
type view = int * int * (int * int * int)

external contract_error :
  dims -> view -> view -> view -> view option -> float * int
  = "nx_metal_test_contract_error"

external contract_wrong :
  dims -> view -> view -> view -> view option -> int * int
  = "nx_metal_test_contract_wrong"

let with_acc ?(samples = 0) acc (batch, m, n, k) =
  (batch, m, n, k, acc, samples)

let float32 = Nx_array.Dtype.(Any Float32)

let plan_contract ?init ?(acc = float32) t (batch, m, n, k) ~a ~b ~out =
  let spec =
    Nx_kernel.Spec.contract
      ~batch:[| (0, 0) |]
      ~contracting:[| (2, 1) |]
      ~acc ~out:(dtype out.dtype) ~init:(Option.is_some init)
  in
  let c =
    {
      spec;
      dst = array out [| batch; m; n |];
      ops =
        Array.of_list
          (array a [| batch; m; k |]
          :: array b [| batch; k; n |]
          :: Option.to_list
               (Option.map (fun i -> array i [| batch; m; n |]) init));
    }
  in
  match Nx_metal.contract c.spec ~dst:c.dst c.ops with
  | Nx_array.Done ->
      Rig.wait t.rig (Rig.submitted t.rig);
      Some [ Call c ]
  | Declined -> None
  | r -> Nx_array.refused "Nx_metal.contract" r (c.dst :: Array.to_list c.ops)

let contract_error ?init ?samples dims ~a ~b ~out =
  let worst, at =
    contract_error
      (with_acc ?samples (Nx_array.Dtype.code Nx_array.Dtype.Float32) dims)
      (cpu a) (cpu b) (cpu out) (Option.map cpu init)
  in
  (* windtrap's at_most orders NaN below every number: a NaN worst would
     pass a bound. *)
  if Float.is_nan worst then
    failwith "Nx_metal_support.contract_error: a NaN ratio";
  (worst, at)

let contract_wrong ?init ?samples ~acc dims ~a ~b ~out =
  let (Nx_array.Dtype.Any acc) = acc in
  contract_wrong
    (with_acc ?samples (Nx_array.Dtype.code acc) dims)
    (cpu a) (cpu b) (cpu out) (Option.map cpu init)
