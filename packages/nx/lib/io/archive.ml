(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Names = Map.Make (String)
module D = Nx_array.Dtype

type 'd t = 'd Nx.packed Names.t

(* Formats build an archive from names they keep distinct and non-empty, and
   write one's bindings. *)

let empty = Names.empty
let add = Names.add
let bindings = Names.bindings

(* Constructors *)

let of_list entries =
  List.fold_left
    (fun t (name, x) ->
      if name = "" then invalid_arg "Nx_io.Archive.of_list: an empty name";
      if Names.mem name t then
        Fail.invalid_argf "Nx_io.Archive.of_list: %S is named twice" name;
      Names.add name x t)
    Names.empty entries

let union ts =
  let shared name _ _ =
    Fail.invalid_argf "Nx_io.Archive.union: %S is in two archives" name
  in
  List.fold_left (Names.union shared) Names.empty ts

(* Queries *)

let names t = List.map fst (Names.bindings t)
let find = Names.find_opt

(* Typed entries *)

let entry ~fn name t =
  match Names.find_opt name t with
  | Some x -> x
  | None -> Fail.failf "Nx_io.Archive.%s: %s: no entry in the archive" fn name

let check_shape ~fn name ~shape x =
  if Nx.shape x <> shape then
    Fail.failf "Nx_io.Archive.%s: %s: shape %s in the archive, %s asked for" fn
      name
      (Fail.pp_shape (Nx.shape x))
      (Fail.pp_shape shape)

let tensor (type v s d) ~shape (dt : (v, s) D.t) name (t : d t) : (v, s, d) Nx.t
    =
  let fn = "tensor" in
  let (Nx.P x) = entry ~fn name t in
  check_shape ~fn name ~shape x;
  match D.equal_witness (Nx.dtype x) dt with
  | Some Type.Equal -> x
  | None ->
      Fail.failf "Nx_io.Archive.%s: %s: %s in the archive, %s asked for" fn name
        (D.name (Nx.dtype x))
        (D.name dt)

(* [convertible dt] is [true] iff [float] converts to and from [dt]. *)
let convertible (type v s) (dt : (v, s) D.t) =
  match dt with
  | D.Float16 | D.Bfloat16 | D.Float32 | D.Float64 -> true
  | _ -> false

let float (type s d) ~shape (dt : (float, s) D.t) name (t : d t) :
    (float, s, d) Nx.t =
  let fn = "float" in
  if not (convertible dt) then
    Fail.invalid_argf
      "Nx_io.Archive.%s: %s is not float16, bfloat16, float32 or float64" fn
      (D.name dt);
  let (Nx.P x) = entry ~fn name t in
  check_shape ~fn name ~shape x;
  match D.equal_witness (Nx.dtype x) dt with
  | Some Type.Equal -> x
  | None when convertible (Nx.dtype x) -> Nx.cast dt x
  | None ->
      Fail.failf
        "Nx_io.Archive.%s: %s: %s in the archive, not float16, bfloat16, \
         float32 or float64"
        fn name
        (D.name (Nx.dtype x))

(* Values *)

(* [full prefix path] is the name of the array at [path] under [prefix]. *)
let full prefix path =
  let p = Nx.Ptree.Path.to_string path in
  if prefix = "" then p else if p = "" then prefix else prefix ^ "." ^ p

(* [namer ~fn prefix] names a value's arrays under [prefix], checking that the
   names are distinct and non-empty. *)
let namer ~fn prefix =
  let seen = Hashtbl.create 64 in
  fun path ->
    let name = full prefix path in
    if name = "" then
      Fail.invalid_argf
        "Nx_io.Archive.%s: an array at the root has no name; give it a ~prefix"
        fn;
    if Hashtbl.mem seen name then
      Fail.invalid_argf "Nx_io.Archive.%s: %s: two arrays have this name" fn
        name;
    Hashtbl.add seen name ();
    name

let of_value ?(prefix = "") s x =
  let name = namer ~fn:"of_value" prefix in
  Nx.Ptree.fold s
    (fun path a t -> Names.add (name path) (Nx.P a) t)
    x Names.empty

(* [owns prefix name] is [true] iff [name] is [prefix] or lies below it. The
   empty prefix holds every name. *)
let owns prefix name =
  prefix = "" || name = prefix || String.starts_with ~prefix:(prefix ^ ".") name

let to_value ?(prefix = "") s ~like t =
  let fn = "to_value" in
  (* Every name, entry and placement is checked before any entry moves. *)
  let name = namer ~fn prefix in
  let named = Hashtbl.create 64 in
  let check path a () =
    let name = name path in
    Hashtbl.replace named name ();
    (match Names.find_opt name t with
    | None ->
        Fail.failf
          "Nx_io.Archive.%s: %s: no entry in the archive, an array in the value"
          fn name
    | Some (Nx.P x) ->
        if Nx.shape x <> Nx.shape a then
          Fail.failf
            "Nx_io.Archive.%s: %s: shape %s in the archive, %s in the value" fn
            name
            (Fail.pp_shape (Nx.shape x))
            (Fail.pp_shape (Nx.shape a));
        if not (D.equal (Nx.dtype x) (Nx.dtype a)) then
          Fail.failf "Nx_io.Archive.%s: %s: %s in the archive, %s in the value"
            fn name
            (D.name (Nx.dtype x))
            (D.name (Nx.dtype a)))
  in
  Nx.Ptree.fold s check like ();
  Names.iter
    (fun entry _ ->
      if owns prefix entry && not (Hashtbl.mem named entry) then
        Fail.failf
          "Nx_io.Archive.%s: %s: an entry in the archive, no array in the value"
          fn entry)
    t;
  (* The checks above make the dtype there. *)
  let read path a =
    let x = Nx.unpack (Nx.dtype a) (Names.find (full prefix path) t) in
    Nx.place (Nx.placement a) x
  in
  Nx.Ptree.map s read like
