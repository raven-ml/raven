(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Names = Map.Make (String)

type t = Nx.packed Names.t

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt
let failf fmt = Printf.ksprintf failwith fmt

let shape_to_string s =
  "[" ^ String.concat "; " (Array.to_list (Array.map string_of_int s)) ^ "]"

let dtype_to_string = Nx_dtype.to_string

(* Formats build an archive from names they keep distinct and non-empty, and
   write one's bindings. *)

let empty = Names.empty
let mem = Names.mem
let add = Names.add
let bindings = Names.bindings

(* Constructors *)

let of_list entries =
  List.fold_left
    (fun t (name, x) ->
      if name = "" then invalid_arg "Nx_io.Archive.of_list: an empty name";
      if Names.mem name t then
        invalid_argf "Nx_io.Archive.of_list: %S is named twice" name;
      Names.add name x t)
    Names.empty entries

let union ts =
  let shared name _ _ =
    invalid_argf "Nx_io.Archive.union: %S is in two archives" name
  in
  List.fold_left (Names.union shared) Names.empty ts

(* Queries *)

let names t = List.map fst (Names.bindings t)
let find = Names.find_opt

(* Typed entries *)

let entry ~fn name t =
  match Names.find_opt name t with
  | Some x -> x
  | None -> failf "Nx_io.Archive.%s: %s: no entry in the archive" fn name

let check_shape ~fn name ~shape x =
  if Nx.shape x <> shape then
    failf "Nx_io.Archive.%s: %s: shape %s in the archive, %s asked for" fn name
      (shape_to_string (Nx.shape x))
      (shape_to_string shape)

let tensor (type a b) ~shape (dtype : (a, b) Nx.dtype) name t : (a, b) Nx.t =
  let fn = "tensor" in
  let (Nx.P x) = entry ~fn name t in
  check_shape ~fn name ~shape x;
  match Nx_dtype.equal_witness (Nx.dtype x) dtype with
  | Some Type.Equal -> x
  | None ->
      failf "Nx_io.Archive.%s: %s: %s in the archive, %s asked for" fn name
        (dtype_to_string (Nx.dtype x))
        (dtype_to_string dtype)

(* [convertible d] is [true] iff [float] converts to and from [d]. *)
let convertible (type a b) (d : (a, b) Nx.dtype) =
  match d with Float16 | BFloat16 | Float32 | Float64 -> true | _ -> false

let float (type b) ~shape (dtype : (float, b) Nx.dtype) name t : (float, b) Nx.t
    =
  let fn = "float" in
  if not (convertible dtype) then
    invalid_argf
      "Nx_io.Archive.%s: %s is not float16, bfloat16, float32 or float64" fn
      (dtype_to_string dtype);
  let (Nx.P x) = entry ~fn name t in
  check_shape ~fn name ~shape x;
  match Nx_dtype.equal_witness (Nx.dtype x) dtype with
  | Some Type.Equal -> x
  | None when convertible (Nx.dtype x) -> Nx.cast dtype x
  | None ->
      failf
        "Nx_io.Archive.%s: %s: %s in the archive, not float16, bfloat16, \
         float32 or float64"
        fn name
        (dtype_to_string (Nx.dtype x))

(* Values *)

(* [namer ~fn] names a value's leaves by their paths, distinct and non-empty. *)
let namer ~fn =
  let seen = Hashtbl.create 64 in
  fun path ->
    let name = Nx.Ptree.Path.to_string path in
    if name = "" then
      invalid_argf
        "Nx_io.Archive.%s: a leaf at the root has no name; put it under \
         Nx.Ptree.field"
        fn;
    if Hashtbl.mem seen name then
      invalid_argf "Nx_io.Archive.%s: %s: two leaves have this name" fn name;
    Hashtbl.add seen name ();
    name

let of_value s x =
  let name = namer ~fn:"of_value" in
  Nx.Ptree.fold s
    (fun path leaf t -> Names.add (name path) (Nx.P leaf) t)
    x Names.empty

(* [owns ns name] is [true] iff [name] lies in the namespace [ns]: [ns] itself
   or below it. The root's namespace holds every name. *)
let owns ns name =
  ns = "" || name = ns || String.starts_with ~prefix:(ns ^ ".") name

let to_value s ~like t =
  let fn = "to_value" in
  let name = namer ~fn in
  let named = Hashtbl.create 64 in
  let read (type a b) path (leaf : (a, b) Nx.t) : (a, b) Nx.t =
    let name = name path in
    Hashtbl.add named name ();
    match Names.find_opt name t with
    | None ->
        failf
          "Nx_io.Archive.%s: %s: no entry in the archive, a leaf in the value"
          fn name
    | Some (Nx.P x) -> (
        if Nx.shape x <> Nx.shape leaf then
          failf "Nx_io.Archive.%s: %s: shape %s in the archive, %s in the value"
            fn name
            (shape_to_string (Nx.shape x))
            (shape_to_string (Nx.shape leaf));
        match Nx_dtype.equal_witness (Nx.dtype x) (Nx.dtype leaf) with
        | Some Type.Equal -> x
        | None ->
            failf "Nx_io.Archive.%s: %s: %s in the archive, %s in the value" fn
              name
              (dtype_to_string (Nx.dtype x))
              (dtype_to_string (Nx.dtype leaf)))
  in
  let x = Nx.Ptree.map s read like in
  let ns = Nx.Ptree.Path.to_string (Nx.Ptree.prefix s) in
  Names.iter
    (fun entry _ ->
      if owns ns entry && not (Hashtbl.mem named entry) then
        failf
          "Nx_io.Archive.%s: %s: an entry in the archive, no leaf in the value"
          fn entry)
    t;
  x
