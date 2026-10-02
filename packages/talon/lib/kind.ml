(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type ext = |

type _ t =
  | Bool : bool t
  | Int : int t
  | Float : float t
  | String : string t
  | Binary : Binary.t t
  | Date : Time.date t
  | Instant : Time.instant t
  | Span : Time.span t
  | List : 'a t -> 'a array t
  | Record : record t
  | Tensor : ('a, 'b) Nx.dtype -> ('a, 'b) Nx.t t
  | Ext : ext t

and record = { fields : (string * field) iarray }

and field =
  | Value : 'a t * 'a option -> field
  | Storage : 'a t * 'a option -> field

let bool = Bool
let int = Int
let float = Float
let string = String
let binary = Binary
let date = Date
let instant = Instant
let span = Span
let list k = List k
let tensor dt = Tensor dt

let rec equal_witness : type a b. a t -> b t -> (a, b) Stdlib.Type.eq option =
 fun k0 k1 ->
  match (k0, k1) with
  | Bool, Bool -> Some Equal
  | Int, Int -> Some Equal
  | Float, Float -> Some Equal
  | String, String -> Some Equal
  | Binary, Binary -> Some Equal
  | Date, Date -> Some Equal
  | Instant, Instant -> Some Equal
  | Span, Span -> Some Equal
  | Record, Record -> Some Equal
  | Ext, Ext -> Some Equal
  | List e0, List e1 -> (
      match equal_witness e0 e1 with Some Equal -> Some Equal | None -> None)
  | Tensor dt0, Tensor dt1 -> (
      match Nx_dtype.equal_witness dt0 dt1 with
      | Some Equal -> Some Equal
      | None -> None)
  | _ -> None

let rec has_ext : type a. a t -> bool = function
  | Ext -> true
  | List e -> has_ext e
  | _ -> false

let provably_equal k0 k1 = if has_ext k0 then None else equal_witness k0 k1

let rec pp : type a. Format.formatter -> a t -> unit =
 fun ppf -> function
  | Bool -> Format.pp_print_string ppf "bool"
  | Int -> Format.pp_print_string ppf "int"
  | Float -> Format.pp_print_string ppf "float"
  | String -> Format.pp_print_string ppf "string"
  | Binary -> Format.pp_print_string ppf "binary"
  | Date -> Format.pp_print_string ppf "date"
  | Instant -> Format.pp_print_string ppf "instant"
  | Span -> Format.pp_print_string ppf "span"
  | Record -> Format.pp_print_string ppf "record"
  | Ext -> Format.pp_print_string ppf "ext"
  | List e -> Format.fprintf ppf "list[%a]" pp e
  | Tensor dt -> Format.fprintf ppf "tensor[%s]" (Nx_dtype.to_string dt)
