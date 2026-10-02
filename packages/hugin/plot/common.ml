(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let err fn fmt =
  Format.kasprintf (fun s -> invalid_arg ("Hugin." ^ fn ^ ": " ^ s)) fmt

let is_pos x = Float.is_finite x && x > 0.

let pp_shape ppf s =
  Format.fprintf ppf "[%a]"
    (Format.pp_print_seq
       ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
       Format.pp_print_int)
    (Array.to_seq s)

(* Ids and warnings *)

type id = Nx.Ptree.Path.t
type warning = id * string

let pp_id ppf id =
  match Nx.Ptree.Path.segments id with
  | [] -> Format.pp_print_string ppf "root"
  | _ -> Nx.Ptree.Path.pp ppf id

let pp_warning ppf (id, msg) = Format.fprintf ppf "@[%a: %s@]" pp_id id msg

let compare_seg (s : Nx.Ptree.Path.seg) (s' : Nx.Ptree.Path.seg) =
  match (s, s') with
  | Field a, Field b -> String.compare a b
  | Field _, Index _ -> -1
  | Index _, Field _ -> 1
  | Index i, Index j -> Int.compare i j

let compare_id id id' =
  List.compare compare_seg
    (Nx.Ptree.Path.segments id)
    (Nx.Ptree.Path.segments id')

let equal_tensor : type a b c d. (a, b) Nx.t -> (c, d) Nx.t -> bool =
 fun x y ->
  match Nx_dtype.equal_witness (Nx.dtype x) (Nx.dtype y) with
  | Some Type.Equal -> x == y
  | None -> false

let equal_strings = Array.equal String.equal
