(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf

(* Compiled programs *)

module Tiny_elf = struct
  type param = {
    name : string option;
    addrspace : Dtype.addr_space;
    dtype : Dtype.t;
    shape : int list;
  }

  type t = {
    lib : string;
    name : string;
    target : Helpers.Target.t;
    signature : param list;
    profile_key : string option;
  }

  let of_program prg =
    let invalid () =
      invalid_arg
        (Format.asprintf "%a of %d sources is not a compiled program" Op.pp
           (Ops.op prg)
           (List.length (Ops.src prg)))
    in
    let param u =
      match (Ops.op u, Ops.arg u) with
      | Op.Param, Ops.Param p ->
          {
            name = p.name;
            addrspace = p.addrspace;
            dtype = Ops.dtype u;
            shape = Shape.max_shape u;
          }
      | _ -> invalid ()
    in
    match (Ops.op prg, Ops.arg prg, Ops.src prg) with
    | Op.Program, Ops.Program info, [ kernel; _; _; binary ] ->
        let lib =
          match Ops.arg binary with Ops.Bytes b -> b | _ -> invalid ()
        in
        let name =
          match Ops.arg kernel with
          | Ops.Kernel k -> Ops.function_name k
          | _ -> invalid ()
        in
        {
          lib;
          name;
          target = info.target;
          signature = List.map param (Shape.kernel_params prg);
          profile_key = Some (Ops.key prg);
        }
    | _ -> invalid ()

  (* Python's tuples: one element keeps its trailing comma. *)
  let pp_tuple pp_x ppf = function
    | [ x ] -> Format.fprintf ppf "(%a,)" pp_x x
    | xs ->
        let sep ppf () = Format.pp_print_string ppf ", " in
        Format.fprintf ppf "(%a)" (Format.pp_print_list ~pp_sep:sep pp_x) xs

  (* Arguments print as Python literals, and No_arg as None. *)
  let pp_literal f ppf x =
    Ops.pp_arg ppf (Option.fold ~none:Ops.No_arg ~some:f x)

  let string s = Ops.String s
  let bytes b = Ops.Bytes b

  let pp_param ppf (p : param) =
    Format.fprintf ppf "(%a, %a, %a, %a)" (pp_literal string) p.name
      Dtype.pp_addr_space p.addrspace Dtype.pp p.dtype
      (pp_tuple Format.pp_print_int)
      p.shape

  let pp ppf e =
    Format.fprintf ppf
      "TinyELF(lib=%a, name=%a, target=%a, signature=%a, profile_key=%a)"
      Ops.pp_arg (bytes e.lib) Ops.pp_arg (string e.name) Helpers.Target.pp
      e.target (pp_tuple pp_param) e.signature (pp_literal bytes) e.profile_key

  let iter_sig ?(offset = 0) signature =
    let place offset p =
      (* A buffer is passed by its address. *)
      let dt = if p.addrspace = Dtype.Alu then p.dtype else Dtype.Uint64 in
      let size = Dtype.itemsize dt in
      let offset = Helpers.round_up offset size in
      (offset + size, (offset, dt))
    in
    snd (List.fold_left_map place offset signature)
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
