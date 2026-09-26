(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

module U = Tolk_uop.Uop
module T = Tensor

let create t make =
  match T.device t with
  | None -> make (T.symbolic_shape t) None
  | Some (U.Single device) -> make (T.symbolic_shape t) (Some device)
  | Some (U.Multi devices) -> (
      match U.axis (T.uop t) with
      | None ->
          let value = make (T.symbolic_shape t) None in
          T.of_uop (U.copy ~src:(T.uop value) ~device:(U.Multi devices) ())
      | Some axis ->
          let shape = U.shard_shape (T.uop t) in
          let shards = List.map (fun device -> T.uop (make shape device)) devices in
          let range = U.range ~size:(U.const_int (List.length devices))
              ~axis:(-1) ~kind:Tolk_uop.Axis_type.Device () in
          T.of_uop (U.unshard ~src:(U.mstack shards) ~axes:[axis] ~ranges:[range] ()))
  | Some (U.Index _) -> invalid_arg "Like.create: unresolved device index"
