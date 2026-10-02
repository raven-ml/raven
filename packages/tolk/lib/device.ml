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
    slot : int;
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
    let param_arg u =
      match (Ops.op u, Ops.arg u) with
      | Op.Param, Ops.Param p -> p
      | _ -> invalid ()
    in
    let param slot u =
      let p = param_arg u in
      {
        name = p.name;
        slot;
        dtype = Ops.dtype u;
        shape = Option.to_list p.size;
      }
    in
    match (Ops.op prg, Ops.arg prg, Ops.src prg) with
    | Op.Program, Ops.Program info, [ kernel; linear; _; binary ] ->
        let lib =
          match Ops.arg binary with Ops.Bytes b -> b | _ -> invalid ()
        in
        let name =
          match Ops.arg kernel with
          | Ops.Kernel k -> Ops.function_name k
          | _ -> invalid ()
        in
        (* Slots are compact, the buffers in the order of the program's globals,
           in which runtimes launch them: a kernel may use a sparse subset of
           its call's buffers. *)
        let buffer u =
          let slot = (param_arg u).slot in
          match List.find_index (Int.equal slot) info.globals with
          | Some j -> param j u
          | None ->
              invalid_arg (strf "buffer slot %d is not among the globals" slot)
        in
        let is_buffer u =
          Ops.op u = Op.Param && Ops.addrspace u <> Some Dtype.Alu
        in
        let nglobals = List.length info.globals in
        {
          lib;
          name;
          target = info.target;
          signature =
            List.map buffer (List.filter is_buffer (Ops.src linear))
            @ List.mapi (fun j v -> param (nglobals + j) v) info.vars;
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
    Format.fprintf ppf "(%a, %d, %a, %a)" (pp_literal string) p.name p.slot
      Dtype.pp p.dtype
      (pp_tuple Format.pp_print_int)
      p.shape

  let pp ppf e =
    Format.fprintf ppf
      "TinyELF(lib=%a, name=%a, target=%a, signature=%a, profile_key=%a)"
      Ops.pp_arg (bytes e.lib) Ops.pp_arg (string e.name) Helpers.Target.pp
      e.target (pp_tuple pp_param) e.signature (pp_literal bytes) e.profile_key

  let iter_sig ?(offset = 0) signature =
    let place offset p =
      let size = Dtype.itemsize p.dtype in
      let offset = Helpers.round_up offset size in
      (offset + size, (offset, p.dtype))
    in
    snd (List.fold_left_map place offset signature)
end

(* Renderers *)

let renderers = function
  | "CPU" -> [ ("CLANG", Cstyle.clang) ]
  | "METAL" -> [ ("METAL", Cstyle.metal) ]
  | "CUDA" | "NV" -> [ ("CUDA", Cstyle.cuda) ]
  | "AMD" -> [ ("HIP", Cstyle.hip) ]
  | device -> invalid_arg (strf "no device named '%s'" device)

(* A renderer is a function of its name and target, which also key the programs
   it compiles, so each pair makes one. Domains compile concurrently. *)
let cache = Hashtbl.create 8
let lock = Mutex.create ()

let cached name make target () =
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
  let candidates = renderers t.device in
  let error = strf "%s has no renderer '%s'" t.device t.renderer in
  Result.bind (Helpers.select_by_name ~error fst t.renderer candidates)
  @@ fun named ->
  Helpers.select_first_inited
    ~error:(strf "No renderer for %s is available" t.device)
    (List.map (fun (name, make) -> cached name make t) named)
