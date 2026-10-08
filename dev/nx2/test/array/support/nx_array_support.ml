(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

external row : int -> (string * int * int) option = "nx_array_support_row"
external of_int64 : int -> int64 -> int = "nx_array_support_of_i64"
external of_uint64 : int -> int64 -> int = "nx_array_support_of_u64"
external layout : Nx_array.Layout.t -> int array = "nx_array_support_layout"
external add : 'z -> 'x -> 'y -> int = "nx_array_support_add" [@@noalloc]

external copy_into : ('v, 's) Nx_array.t -> ('v, 's) Nx_array.t -> int
  = "nx_array_copy"
[@@noalloc]

external int16_at :
  int ->
  int ->
  (int, Bigarray.int16_signed_elt, Bigarray.c_layout) Bigarray.Genarray.t
  = "nx_array_support_int16_at"

external collect : ('v, 's) Nx_array.t -> int = "nx_array_support_collect"

(* An io device over bigarrays *)

type bytes =
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

external blit_in : bytes -> int -> int -> int -> unit
  = "nx_array_support_blit_in"
[@@noalloc]

external blit_out : bytes -> int -> int -> int -> unit
  = "nx_array_support_blit_out"
[@@noalloc]

module Io = struct
  type t = unit
  type region = bytes

  exception Fault of string

  let region_key : region Type.Id.t = Type.Id.make ()
  let budget () = max_int
  let allocations = Atomic.make 0

  let alloc () n =
    Atomic.incr allocations;
    Some (Bigarray.Array1.create Bigarray.char Bigarray.c_layout n)

  let free () (_ : region) = ()
  let read () r ~at ~dst ~len = blit_out r at dst len
  let write () r ~at ~src ~len = blit_in r at src len
  let pages () (_ : region) = None
  let prefetch () (_ : region) ~at:(_ : int) ~len:(_ : int) = ()
  let stop () = ()
end

let io =
  lazy
    (match Rig.open_io (module Io) ~name:"nx2-io" (fun () -> Ok ()) with
    | Ok d -> d
    | Error e -> failwith e)

let io_device () = Lazy.force io
let io_allocations () = Atomic.get Io.allocations
