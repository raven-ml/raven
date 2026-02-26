(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Dtype = Nx_core.Dtype

type t = (string * Ptree.tensor) list

let empty = []
let set ~name tensor t = (name, tensor) :: t
let find t ~name = List.assoc_opt name t

let get_float_exn (type l) ~ctx t ~name ~(dtype : (float, l) Rune.dtype) :
    (float, l) Rune.t =
  match find t ~name with
  | Some (Ptree.P x) -> (
      match Dtype.equal_witness dtype (Rune.dtype x) with
      | Some Type.Equal -> x
      | None ->
          invalid_arg
            (Printf.sprintf "%s: %s dtype mismatch (expected %s, got %s)" ctx
               name (Dtype.to_string dtype)
               (Dtype.to_string (Rune.dtype x))))
  | None -> invalid_arg (Printf.sprintf "%s: %s not found in context" ctx name)

let get_int32_exn ~ctx t ~name : (int32, Bigarray.int32_elt) Rune.t =
  match find t ~name with
  | Some tensor -> Ptree.Tensor.to_typed_exn Rune.int32 tensor
  | None -> invalid_arg (Printf.sprintf "%s: %s not found in context" ctx name)

let get_bool_exn ~ctx t ~name : (bool, Nx.bool_elt) Rune.t =
  match find t ~name with
  | Some tensor -> Ptree.Tensor.to_typed_exn Rune.bool tensor
  | None -> invalid_arg (Printf.sprintf "%s: %s not found in context" ctx name)
