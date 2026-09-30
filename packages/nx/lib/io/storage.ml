(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Tensors and the host buffers that hold their elements *)

module B = Nx_device.Buffer

let tensor dtype buffer shape =
  let view = Nx_array.View.create [| B.length buffer |] in
  Nx.reshape shape (Nx.Repr.host { Nx_array.dtype; view; buffer })

(* The elements of [x] in C order, in a host buffer: its storage when it is
   contiguous on the host. *)
let elements x =
  let x =
    match Nx.Repr.v x with Placed _ -> x | Host _ | Traced _ -> Nx.contiguous x
  in
  let b = Nx.Op.eval (Read x) in
  B.view b ~offset:0 (B.dtype b) (Nx.numel x)

(* The buffer of exactly [x]'s elements in C order, without a copy, when there
   is one: [x] is on the host, or its storage is one runtime buffer, and its
   view is a contiguous run of its storage that starts on a byte. *)
let run x =
  let run_in b v =
    let s = B.dtype b and n = Nx_array.View.numel v in
    let first = Nx_array.View.offset v * Nx_dtype.Scalar.bitsize s in
    if n > 0 && Nx_array.View.is_c_contiguous v && first mod 8 = 0 then
      Some (B.view b ~offset:(first / 8) s n)
    else None
  in
  match Nx.Repr.v x with
  | Host a -> run_in a.buffer a.view
  | Placed p -> (
      match Nx.Repr.Storage.buffers (Nx.Repr.Placed.storage p) with
      | [ b ] -> run_in b (Nx.Repr.Placed.view p)
      | _ | (exception Invalid_argument _) -> None)
  | Traced _ -> None

(* The value of [shape] whose elements, of [dtype], are [b]'s in C order,
   without a copy: a host value for a buffer of the host, and on [b]'s device
   otherwise. *)
let on_device (type a b) (dtype : (a, b) Nx_dtype.t) shape b : (a, b) Nx.t =
  let n = B.length b in
  if Array.fold_left ( * ) 1 shape <> n then
    invalid_arg
      (Printf.sprintf "Nx_io: shape %s for %d elements"
         (Nx_array.Shape.to_string shape) n);
  let s = B.dtype b in
  if not (Nx_dtype.Scalar.equal s (Nx_dtype.Scalar.of_dtype dtype)) then
    invalid_arg
      (Printf.sprintf "Nx_io: a %s buffer read as %s"
         (Nx_dtype.Scalar.to_string s)
         (Nx_dtype.to_string dtype));
  let d = B.device b in
  if Nx_device.equal d Nx_device.host then tensor dtype b shape
  else
    let p = Nx.Placement.device (Nx.Device.of_runtime d) in
    Nx.Repr.Placed.v p dtype (Nx_array.View.create shape)
      (Nx.Repr.Storage.v p [ b ])

let bytes b = B.bigarray Bigarray.int8_unsigned b

(* The bytes of the file at [path], read where they lie: the disk's mapping of
   its pages. Raises [Sys_error] if the file cannot be opened or mapped. *)
let file_bytes path =
  match Result.bind (B.of_file path) (B.borrow Nx_device.host) with
  | Ok b -> bytes b
  | Error why -> raise (Sys_error why)
