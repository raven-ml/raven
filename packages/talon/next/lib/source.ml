(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type answer = Exact | Inexact | Unsupported

module Pred = struct
  type value = Value : 'a Type.t * 'a -> value

  type t =
    | Cmp of string * [ `Eq | `Ne | `Lt | `Le | `Gt | `Ge ] * value
    | In of string * value list
    | Null of string
    | Valid of string
    | And of t list
    | Or of t list
    | Not of t
end

type request = {
  columns : string list;
  filters : Pred.t list;
  limit : int option;
}

type reader = {
  next : unit -> (Table.t option, Error.t) result;
  close : unit -> unit;
}

type part = { rows : int option; open_ : unit -> (reader, Error.t) result }

type t = {
  name : string;
  schema : Schema.t;
  rows : int option;
  sorted : Order.t list;
  pushdown : Pred.t -> answer;
  parts : request -> (part list, Error.t) result;
}

let err fmt = Format.kasprintf invalid_arg ("Source.v: " ^^ fmt)

let has_control s =
  let rec loop i =
    i < String.length s
    &&
    let d = String.get_utf_8_uchar s i in
    let c = Uchar.to_int (Uchar.utf_decode_uchar d) in
    c < 0x20 || (0x7f <= c && c < 0xa0) || loop (i + Uchar.utf_decode_length d)
  in
  loop 0

let unsupported _ = Unsupported

let v ~name ~schema ?rows ?(sorted = []) ?(pushdown = unsupported) parts =
  if String.equal name "" then err "the name is empty";
  if not (String.is_valid_utf_8 name) then err "the name %S is not UTF-8" name;
  if has_control name then err "the name %S holds a control character" name;
  (match rows with
  | Some n when n < 0 -> err "the row count %d is negative" n
  | _ -> ());
  (match Order.check sorted schema with
  | [] -> ()
  | ps ->
      let pp ppf (k, p) =
        Format.fprintf ppf "@\n  %a: %a" Order.pp k Problem.pp p
      in
      err "~sorted:%a" (Format.pp_print_list ~pp_sep:(fun _ () -> ()) pp) ps);
  { name; schema; rows; sorted; pushdown; parts }
