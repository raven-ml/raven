(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Typed host buffers for the tests that call the engine's C entry points with
   operand records of their own. *)

type ('a, 'b) t = {
  dtype : ('a, 'b) Nx_dtype.t;
  storage : Nx_device.Buffer.t;
  get : int -> 'a;
  set : int -> 'a -> unit;
}

let create dtype n =
  let storage = Nx_core.Elements.create dtype n in
  {
    dtype;
    storage;
    get = Nx_core.Elements.get dtype storage;
    set = Nx_core.Elements.set dtype storage;
  }

let dtype b = b.dtype
let storage b = b.storage
let length b = Nx_device.Buffer.length b.storage
let get b i = b.get i
let set b i v = b.set i v
let fill b v = Nx_core.Elements.fill b.dtype b.storage v
