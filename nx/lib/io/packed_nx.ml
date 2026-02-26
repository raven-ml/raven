(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf

type t = P : ('a, 'b) Nx.t -> t
type archive = (string, t) Hashtbl.t

let err_dtype_mismatch ~expected ~got =
  strf "dtype mismatch: expected %s, got %s" expected got

let to_typed : type a b. (a, b) Nx.dtype -> t -> (a, b) Nx.t =
 fun target (P nx) ->
  let source = Nx.dtype nx in
  match Nx_core.Dtype.equal_witness source target with
  | Some Type.Equal -> (nx : (a, b) Nx.t)
  | None ->
      let expected = Nx_core.Dtype.to_string target in
      let got = Nx_core.Dtype.to_string source in
      failwith (err_dtype_mismatch ~expected ~got)

let packed_shape (P nx) = Nx.shape nx
