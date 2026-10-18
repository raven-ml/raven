(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

external view_fields :
  batch:(int * int) iarray ->
  contracting:(int * int) iarray ->
  dst:('v, 's) Nx_array.t ->
  init:('v, 's) Nx_array.t option ->
  ('a, 'b) Nx_array.t ->
  ('c, 'e) Nx_array.t ->
  (int array, Nx_array.answer) result
  = "nx_kernel_support_view_byte" "nx_kernel_support_view"

external view_groups :
  batch:(int * int) iarray ->
  contracting:(int * int) iarray ->
  dst:('v, 's) Nx_array.t ->
  init:('v, 's) Nx_array.t option ->
  ('a, 'b) Nx_array.t ->
  ('c, 'e) Nx_array.t ->
  Nx_array.answer = "nx_kernel_support_groups_byte" "nx_kernel_support_groups"

external prog : Nx_kernel.Prog.t -> string = "nx_kernel_support_prog"

external op1 : Nx_kernel.Prog.op1 -> int = "nx_kernel_support_op1"
external op2 : Nx_kernel.Prog.op2 -> int = "nx_kernel_support_op2"
external tag : 'a -> int = "nx_kernel_support_tag"
external code : string -> int = "nx_kernel_support_code"
