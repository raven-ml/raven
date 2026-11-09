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

(* Hive paths *)

(* No specification fixes this layout: what a Hive directory name means is the
   convention its writers and readers share. A partition key and value are
   percent-encoded in one directory name, [key=value], and split at its first
   [=]. Decoding is by bytes, and only a pair of hex digits decodes. *)

let hive_null = "__HIVE_DEFAULT_PARTITION__"

let hex c =
  match c with
  | '0' .. '9' -> Some (Char.code c - Char.code '0')
  | 'a' .. 'f' -> Some (Char.code c - Char.code 'a' + 10)
  | 'A' .. 'F' -> Some (Char.code c - Char.code 'A' + 10)
  | _ -> None

let percent_decode s =
  if not (String.contains s '%') then s
  else begin
    let n = String.length s in
    let b = Buffer.create n in
    let rec go i =
      if i < n then begin
        match s.[i] with
        | '%' when i + 2 < n -> (
            match (hex s.[i + 1], hex s.[i + 2]) with
            | Some h, Some l ->
                Buffer.add_char b (Char.chr ((h * 16) + l));
                go (i + 3)
            | _ ->
                Buffer.add_char b '%';
                go (i + 1))
        | c ->
            Buffer.add_char b c;
            go (i + 1)
      end
    in
    go 0;
    Buffer.contents b
  end

(* [hive_keys path] is the [(key, value)] pairs of [path]'s directory
   segments, in order. Every segment but the last is a directory; a segment
   holding no [=] gives no key. *)
let hive_keys path =
  let segments = String.split_on_char '/' path in
  let directories =
    match List.rev segments with [] -> [] | _ :: dirs -> List.rev dirs
  in
  let rec go seen acc = function
    | [] -> Ok (List.rev acc)
    | segment :: rest -> (
        match String.index_opt segment '=' with
        | None -> go seen acc rest
        | Some i ->
            let key = percent_decode (String.sub segment 0 i)
            and raw =
              String.sub segment (i + 1) (String.length segment - i - 1)
            in
            let bad reason =
              Error
                (Error.v ~file:path
                   (Printf.sprintf "the directory %S is not a Hive pair: %s"
                      segment reason))
            in
            if String.equal key "" then bad "its key is empty"
            else if not (String.is_valid_utf_8 key) then
              bad "its key is not valid UTF-8 once decoded"
            else if List.mem key seen then bad "its key is repeated in the path"
            else if String.equal key "path" then
              bad "the column path holds the file's path; rename the directory"
            else begin
              let value =
                if String.equal raw hive_null then Ok None
                else
                  let v = percent_decode raw in
                  if String.is_valid_utf_8 v then Ok (Some v)
                  else bad "its value is not valid UTF-8 once decoded"
              in
              match value with
              | Error _ as e -> e
              | Ok v -> go (key :: seen) ((key, v) :: acc) rest
            end)
  in
  go [] [] directories

let hive paths =
  let rec decode acc = function
    | [] -> Ok (List.rev acc)
    | path :: rest -> (
        if not (String.is_valid_utf_8 path) then
          Error (Error.v ~file:path "the path is not valid UTF-8")
        else
          match hive_keys path with
          | Error _ as e -> e
          | Ok kv -> decode ((path, kv) :: acc) rest)
  in
  match decode [] paths with
  | Error _ as e -> e
  | Ok decoded -> (
      let keys =
        match decoded with [] -> [] | (_, kv) :: _ -> List.map fst kv
      in
      let same (path0, kv0) (path, kv) =
        let names = List.map fst kv in
        if List.equal String.equal (List.sort String.compare names) (List.sort String.compare keys)
        then Ok ()
        else
          Error
            (Error.v
               (Printf.sprintf
                  "%s and %s have different Hive keys: [%s] and [%s]"
                  path0 path
                  (String.concat "; " (List.map fst kv0))
                  (String.concat "; " names)))
      in
      let rec check first = function
        | [] -> Ok ()
        | row :: rest -> (
            match same first row with
            | Ok () -> check first rest
            | Error _ as e -> e)
      in
      let checked =
        match decoded with [] -> Ok () | first :: rest -> check first rest
      in
      match checked with
      | Error _ as e -> e
      | Ok () ->
          let rows = Array.of_list decoded in
          let column key =
            ( key,
              Column.of_options Type.string
                (Array.map (fun (_, kv) -> List.assoc key kv) rows) )
          in
          let path_column =
            ( "path",
              Column.v Type.string (Array.map fst rows) )
          in
          Ok (Table.v (path_column :: List.map column keys)))
