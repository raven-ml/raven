[@@@warning "-69"]

open Nx_core

type ('a, 'b) buffer = ('a, 'b) Internal.buffer

type ('a, 'b) t = ('a, 'b) Internal.t = {
  dtype : ('a, 'b) Dtype.dtype;
  buffer : ('a, 'b) buffer;
  view : View.view;
}

type context = Internal.context

let create_context () = Internal.{ pool = Parallel.get_or_setup_pool () }

(* *)

let const _ctx shape buffer =
  let nelems =
    if Array.length shape = 0 then 1 else Array.fold_left ( * ) 1 shape
  in
  if Bigarray.Array1.dim buffer <> nelems then
    invalid_arg "payload dim ≠ product(shape)";
  let view = View.make_view shape in
  let dtype = Dtype.dtype_of_kind (Bigarray.Array1.kind buffer) in
  { dtype; buffer; view }

let buffer _ctx dtype size =
  let kind = Dtype.kind_of_dtype dtype in
  let ba = Bigarray.Array1.create kind Bigarray.c_layout size in
  let view = View.make_view [| size |] in
  { dtype; buffer = ba; view }

let reshape _ctx t _new_shape =
  (* NoOp *)
  t

let expand _ctx t _new_shape =
  (* NoOp *)
  t

let add ctx a b out = Ops_binary.add ctx a b out
