(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf

(* A compiled program's parameters *)

module Tiny_elf = struct
  type param = {
    name : string option;
    addrspace : Dtype.addr_space;
    dtype : Dtype.t;
    shape : int list;
  }

  let iter_sig ?(offset = 0) signature =
    let place offset p =
      (* A buffer is passed by its address. *)
      let dt = if p.addrspace = Dtype.Alu then p.dtype else Dtype.Uint64 in
      let size = Dtype.itemsize dt in
      let offset = Helpers.round_up offset size in
      (offset + size, (offset, dt))
    in
    snd (List.fold_left_map place offset signature)

  let layout params =
    let param u =
      let addrspace = (Ops.param_arg_of u).addrspace in
      { name = None; addrspace; dtype = Ops.dtype u; shape = [] }
    in
    iter_sig (List.map param params)
end

(* Renderers *)

(* Each device has one renderer. *)
let renderer_of = function
  | "CPU" -> ("CLANG", Cstyle.clang)
  | "METAL" -> ("METAL", Cstyle.metal)
  | "CUDA" | "NV" -> ("CUDA", Cstyle.cuda)
  | "AMD" -> ("HIP", Cstyle.hip)
  | device -> invalid_arg (strf "no device named '%s'" device)

(* A renderer is a function of its name and target, which also key the programs
   it compiles, so each pair makes one. Domains compile concurrently. *)
let cache = Hashtbl.create 8
let lock = Mutex.create ()

let cached name make target =
  Mutex.protect lock @@ fun () ->
  match Hashtbl.find_opt cache (name, target) with
  | Some r -> Ok r
  | None -> (
      match make target with
      | r ->
          Hashtbl.add cache (name, target) r;
          Ok r
      | exception Invalid_argument e -> Error e)

let renderer (t : Helpers.Target.t) =
  let name, make = renderer_of t.device in
  if t.renderer <> "" && t.renderer <> name then
    Error (strf "%s has no renderer '%s'" t.device t.renderer)
  else cached name make t
