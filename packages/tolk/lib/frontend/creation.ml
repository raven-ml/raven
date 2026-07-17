(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk_uop
module U = Uop
module D = Dtype
module T = Tensor

let dtype_of_fill = function
  | T.Sint _ -> D.default_int
  | T.Sfloat _ -> D.default_float
  | T.Sbool _ -> D.bool

let broadcast_scalar dt fill shape =
  let v = T.of_uop (U.const (T.scalar_const dt fill)) in
  Movement.expand (Movement.reshape v (List.map (fun _ -> 1) shape)) shape

(* Clone [t] into a fresh buffer: an unallocated flat buffer viewed at [t]'s
   shape, written by a store effect. Realization allocates the storage and
   runs the fill, and in-place assignment then writes into it. *)
let clone t =
  let shape = T.shape t in
  let n = List.fold_left ( * ) 1 shape in
  let buf =
    U.buffer ~slot:(U.fresh_buffer_slot ()) ~dtype:(T.dtype t)
      ~shape:(T.shape_uop [ n ]) ()
  in
  let dst = U.reshape ~src:buf ~shape:(T.shape_uop shape) in
  T.of_uop (U.after ~src:dst ~deps:[ U.store ~dst ~value:(T.uop t) () ])

let full ?dtype ?(buffer = true) shape fill =
  let dt = match dtype with Some d -> d | None -> dtype_of_fill fill in
  let v = broadcast_scalar dt fill shape in
  if buffer then clone v else v

let zeros ?dtype ?buffer shape = full ?dtype ?buffer shape (T.Sfloat 0.0)
let ones ?dtype ?buffer shape = full ?dtype ?buffer shape (T.Sfloat 1.0)
let const_like t fill = broadcast_scalar (T.val_dtype t) fill (T.shape t)

let full_like ?dtype ?buffer t fill =
  let dt = match dtype with Some d -> d | None -> T.val_dtype t in
  full ~dtype:dt ?buffer (T.shape t) fill

let zeros_like ?dtype ?buffer t = full_like ?dtype ?buffer t (T.Sint 0)
let ones_like ?dtype ?buffer t = full_like ?dtype ?buffer t (T.Sint 1)
