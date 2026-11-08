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

  (* A column of the schema, read through a handle of its kind. *)
  type column = Column : 'a Kind.t * ('a, Expr.row) Expr.t -> column

  let column schema c =
    match Schema.find schema c with
    | Some (Type.Any ty) when not (Type.has_ext ty) ->
        let k = Type.kind ty in
        Some (Column (k, Expr.make (Handle (k, c))))
    | Some _ | None -> None

  (* [raw k v] is the value of [v] read as [k], and [lit k v] its literal,
     typed at [v]'s type, where the comparison is made. *)
  let raw (type a) (k : a Kind.t) (Value (t, v)) : a option =
    match Kind.provably_equal (Type.kind t) k with
    | Some Equal -> Some v
    | None -> None

  let lit (type a) (k : a Kind.t) (Value (t, v)) :
      (a, Expr.row) Expr.t option =
    match Kind.provably_equal (Type.kind t) k with
    | Some Equal -> Some (Expr.store t (Expr.make (Lit (k, v))))
    | None -> None

  let all f init ps =
    List.fold_left
      (fun acc p ->
        match (acc, p) with Some a, Some b -> Some (f a b) | _ -> None)
      (Some init) ps

  let rec filter schema p : (bool, Expr.row) Expr.t option =
    let on c f = Option.bind (column schema c) f in
    match p with
    | Cmp (c, op, v) ->
        on c @@ fun (Column (k, x)) ->
        Option.map
          (fun v ->
            match op with
            | `Eq -> Expr.(x = v)
            | `Ne -> Expr.(x <> v)
            | `Lt -> Expr.(x < v)
            | `Le -> Expr.(x <= v)
            | `Gt -> Expr.(x > v)
            | `Ge -> Expr.(x >= v))
          (lit k v)
    | In (c, []) -> on c @@ fun _ -> Some (Expr.bool false)
    | In (c, (Value (t, _) :: _ as vs)) ->
        (* The column is compared at its values' type, which contains its
           own: a float64 value is no float32 row's key unless it is one
           exactly. *)
        on c @@ fun (Column (k, x)) ->
        let at (Value (t', _) as v) =
          if Type.equal t' t then raw k v else None
        in
        let ws = List.filter_map at vs in
        if List.compare_lengths ws vs <> 0 then None
        else begin
          match Kind.provably_equal (Type.kind t) k with
          | Some Equal -> Some (Expr.is_in ws (Expr.store t x))
          | None -> None
        end
    | Null c -> on c @@ fun (Column (_, x)) -> Some (Expr.is_null x)
    | Valid c -> on c @@ fun (Column (_, x)) -> Some Expr.(not (is_null x))
    | And ps -> all Expr.( && ) (Expr.bool true) (List.map (filter schema) ps)
    | Or ps -> all Expr.( || ) (Expr.bool false) (List.map (filter schema) ps)
    | Not p -> Option.map Expr.not (filter schema p)
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
