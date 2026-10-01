(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk
module B = Nx_device.Buffer
module C = Nx_cuda_device

let word v =
  let w = B.create Nx_device.host Nx_dtype.Scalar.Int64 1 in
  (B.bigarray Bigarray.int64 w).{0} <- Int64.of_nativeint v;
  w

(* The words host programs read, made once and kept for the life of the process:
   the driver's entry points, and each device's context, streams and status, and
   functions, with their programs. A word keeps its program loaded while a batch
   reads it, and a placeholder's storage can keep nothing else alive, so a
   loaded function and its module stay loaded until the process exits, even once
   no compiled schedule uses them, as tinygrad's functools.cache on
   CUDADevice.function keeps them. *)
type words = {
  handles : B.t; (* [context, compute stream, copy stream, status] *)
  functions : (string * string, Nx_device.Program.t * B.t) Hashtbl.t;
}

let lock = Mutex.create ()
let devices_words = ref []
let driver_words = Hashtbl.create 8
let stamp_word = lazy (word C.stamp)

let words d c =
  match List.assq_opt d !devices_words with
  | Some w -> w
  | None ->
      let handles = B.create Nx_device.host Nx_dtype.Scalar.Int64 4 in
      List.iteri
        (fun i v ->
          (B.bigarray Bigarray.int64 handles).{i} <- Int64.of_nativeint v)
        [ C.context c; C.compute c; C.copy c; 0n ];
      let w = { handles; functions = Hashtbl.create 16 } in
      devices_words := (d, w) :: !devices_words;
      w

let driver_word f =
  match Hashtbl.find_opt driver_words f with
  | Some b -> b
  | None -> (
      match C.driver_function f with
      | Some a ->
          let b = word a in
          Hashtbl.add driver_words f b;
          b
      | None -> invalid_arg ("the CUDA driver has no function " ^ f))

let function_word d w binary name =
  match Hashtbl.find_opt w.functions (binary, name) with
  | Some (_, b) -> b
  | None -> (
      match Nx_device.Program.load d ~binary ~name with
      | Ok p ->
          let b = word (Nx_device.Program.handle p) in
          Hashtbl.add w.functions (binary, name) (p, b);
          b
      | Error why -> failwith why)

(* The storage of the placeholders CUDA's commands name: those of the device
   [name], and the driver's entry points on the host. *)
let placeholder name d c u =
  let on_device =
    match Ops.device u with
    | Some (Single n) | Some (Multi [ n ]) -> n = name
    | _ -> false
  in
  Mutex.protect lock @@ fun () ->
  match Ops.tag u with
  | Some (Tuple [ String "cfunc"; String "cuda"; String f ]) ->
      Some (driver_word f)
  | _ when not on_device -> None
  | Some (String "cuda") -> Some (words d c).handles
  | Some (String "stamp") -> Some (Lazy.force stamp_word)
  | Some (Tuple [ String "function"; Bytes binary; String f ]) ->
      Some (function_word d (words d c) binary f)
  | _ -> None

(* The queues address the memory nx.device says the device reaches: other memory
   is copied through the host's. *)
let reaches d devices n =
  match List.assoc_opt n devices with
  | Some d' -> Nx_device.reaches d d'
  | None -> false

let queues ~host devices name d =
  Option.map
    (fun c ->
      ( Ops_cuda.queues ~host:(Lazy.force host) ~reaches:(reaches d devices),
        placeholder name d c,
        ignore ))
    (C.of_device d)
