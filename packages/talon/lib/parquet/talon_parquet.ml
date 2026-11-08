(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon

(* [ty] is [None] for a decimal that no [with_type] declared. *)
type column = { leaf : Leaf.t; ty : Type.any option }
type format = column array

(* [with_bytes ?file b f] is [f] applied to the bytes of [b], mapped on the
   host, with the errors of the private modules as [Error]. *)
let with_bytes ?file b f =
  match Rig.Buffer.borrow Rig.host b with
  | exception Sys_error why -> Error (Error.v ?file why)
  | None -> Error (Error.v ?file "the host cannot map the bytes")
  | Some host -> (
      Rig.Buffer.read host Bigarray.int8_unsigned @@ fun bytes ->
      try Ok (f bytes)
      with Meta.Error { row_group; bytes; text; msg } ->
        Error (Error.v ?file ?row_group ?bytes ?text msg))

let open_ bytes =
  let m = Meta.footer bytes in
  let leaves = Leaf.of_schema m.schema in
  Array.iteri
    (fun g (rg : Meta.row_group) ->
      if Array.length rg.chunks <> Array.length leaves then
        Meta.fail ~row_group:g
          "the row group has %d column chunks for %d columns"
          (Array.length rg.chunks) (Array.length leaves);
      Array.iteri (fun i l -> Chunk.check m ~row_group:g i l) leaves)
    m.row_groups;
  (m, leaves)

let sniffed leaves =
  Array.map (fun leaf -> { leaf; ty = Leaf.default leaf }) leaves

let sniff b = with_bytes b @@ fun bytes -> sniffed (snd (open_ bytes))
let find f name = Array.find_opt (fun c -> String.equal c.leaf.name name) f

let index f name =
  match Array.find_index (fun c -> String.equal c.leaf.name name) f with
  | Some i -> i
  | None ->
      invalid_arg
        (Printf.sprintf "Talon_parquet.with_type: no column %S in the format."
           name)

let type_name (Type.Any t) = Format.asprintf "%a" Type.pp t

let with_type name t f =
  let i = index f name in
  let leaf = f.(i).leaf in
  if not (Leaf.reads_as leaf t) then
    invalid_arg
      (Format.asprintf
         "Talon_parquet.with_type: column %S (%a) reads as %a, not %s. Cast it \
          once read (Expr.cast)."
         name Leaf.pp leaf Leaf.pp_reads leaf (type_name t));
  let f = Array.copy f in
  f.(i) <- { leaf; ty = Some t };
  f

(* [scalars s] is the number of Unicode scalar values of the UTF-8 [s]. *)
let scalars s =
  let n = ref 0 in
  String.iter (fun c -> if Char.code c land 0xC0 <> 0x80 then incr n) s;
  !n

let pp_format ppf f =
  let left =
    Array.map
      (fun c ->
        match c.ty with
        | Some ty ->
            Format.asprintf "%a" Schema.pp (Schema.v [ (c.leaf.name, ty) ])
        | None -> Format.asprintf "%a undeclared" Type.pp_name c.leaf.name)
      f
  in
  let width = Array.fold_left (fun w s -> max w (scalars s)) 0 left in
  let n = Array.length f in
  Format.fprintf ppf "@[<v>parquet (%d column%s)" n (if n = 1 then "" else "s");
  Array.iteri
    (fun i c ->
      Format.fprintf ppf "@,  %s%s ← %a" left.(i)
        (String.make (width - scalars left.(i)) ' ')
        Leaf.pp c.leaf)
    f;
  Format.fprintf ppf "@]"

(* Statistics *)

let stamp (u : Type.unit_) t =
  match u with
  | S -> Time.of_s t
  | Ms -> Time.of_ms t
  | Us -> Time.of_us t
  | Ns -> Some (Time.of_ns t)

(* [decoder ty l] reads a statistic of the leaf [l], [PLAIN]-encoded, as a value
   of [ty], for the types whose Parquet order is talon's. *)
let decoder : type a. a Type.t -> Leaf.t -> (string -> a option) option =
 fun ty l ->
  let fixed (p : Meta.physical) n f =
    if l.physical <> p then None
    else Some (fun s -> if String.length s = n then f s else None)
  in
  let i32 s = Int32.to_int (String.get_int32_le s 0) in
  let i64 s = String.get_int64_le s 0 in
  let int = fixed Int32 4 (fun s -> Some (i32 s)) in
  let uint = fixed Int32 4 (fun s -> Some (i32 s land 0xFFFF_FFFF)) in
  let bytes f =
    match l.physical with
    | Byte_array | Fixed_len_byte_array -> Some (fun s -> Some (f s))
    | _ -> None
  in
  match ty with
  | Int8 -> int
  | Int16 -> int
  | Int32 -> int
  | Uint8 -> uint
  | Uint16 -> uint
  | Uint32 -> uint
  | Int64 ->
      fixed Int64 8 (fun s ->
          let v = i64 s in
          if Int64.(equal (of_int (to_int v)) v) then Some (Int64.to_int v)
          else None)
  | Float32 ->
      fixed Float 4 (fun s ->
          Some (Int32.float_of_bits (String.get_int32_le s 0)))
  | Float64 -> fixed Double 8 (fun s -> Some (Int64.float_of_bits (i64 s)))
  | Date -> fixed Int32 4 (fun s -> Time.Date.of_days (i32 s))
  | Datetime { unit_; _ } -> fixed Int64 8 (fun s -> stamp unit_ (i64 s))
  | String -> bytes Fun.id
  | Categorical _ -> bytes Fun.id
  | Binary -> bytes Binary.of_string
  | _ -> None

(* [bounds l ty s v] compares the minimum and the maximum of the statistics [s]
   of the leaf [l], read as [ty], with [v]. A float's bounds hold only when the
   chunk is known to hold no NaN, which Parquet leaves out of them. *)
let bounds l (Type.Any ty) (s : Meta.stats) (Source.Pred.Value (tv, v)) =
  match
    (tv, decoder ty l, Kind.provably_equal (Type.kind ty) (Type.kind tv))
  with
  | Categorical _, _, _ | _, None, _ | _, _, None -> None
  | _, Some read, Some Equal -> (
      let nan_free =
        match l.Leaf.physical with
        | Float | Double -> s.nans = Some 0
        | _ -> true
      in
      match (Option.bind s.min read, Option.bind s.max read) with
      | Some lo, Some hi when nan_free ->
          Some (Type.compare_value tv lo v, Type.compare_value tv hi v)
      | _ -> None)

(* [passes stats rows p] is [false] when the statistics [stats c] of each column
   [c], with its leaf and type, show that none of [rows] rows passes [p]. A
   comparison with null is null, so a column of nulls passes none. *)
let rec passes stats rows (p : Source.Pred.t) =
  let range c v k =
    match stats c with
    | None -> true
    | Some (_, _, (s : Meta.stats)) when s.nulls = Some rows -> false
    | Some (l, ty, s) -> (
        match bounds l ty s v with Some (lo, hi) -> k lo hi | None -> true)
  in
  let within lo hi = lo <= 0 && hi >= 0 in
  match p with
  | Cmp (c, op, v) ->
      range c v (fun lo hi ->
          match op with
          | `Eq -> within lo hi
          | `Ne -> true
          | `Lt -> lo < 0
          | `Le -> lo <= 0
          | `Gt -> hi > 0
          | `Ge -> hi >= 0)
  | In (c, vs) -> List.exists (fun v -> range c v within) vs
  | Null c -> (
      match stats c with Some (_, _, s) -> s.nulls <> Some 0 | None -> true)
  | Valid c -> (
      match stats c with Some (_, _, s) -> s.nulls <> Some rows | None -> true)
  | And ps -> List.for_all (passes stats rows) ps
  | Or ps -> List.exists (passes stats rows) ps
  | Not _ -> true

(* [usable f p] is [true] iff statistics can show that no row passes [p]. *)
let rec usable f (p : Source.Pred.t) =
  match p with
  | Cmp (c, _, _) | In (c, _) | Null c | Valid c -> Option.is_some (find f c)
  | And ps -> List.exists (usable f) ps
  | Or ps -> List.for_all (usable f) ps
  | Not _ -> false

(* In-scan filters

   The source applies a conjunct over one column of byte strings itself: on a
   dictionary-encoded chunk ({!Chunk.Coded}) once per word, the words' results
   then taken at the rows' codes, and on any other chunk on its rows. The
   conjunct is evaluated by talon, as the filter it is
   ({!Talon.Source.Pred.filter}); a row passes where it is [true]. *)

(* [truth e] is the output [true] where the filter [e] is, [false] where it
   is [false] or null. *)
let truth e = Expr.("keep" := if_ e (bool true) (bool false))

(* [in_scan schema p] is the column that [p] reads and the filter it is, when
   the source applies [p]: a predicate on one column of [schema] that reads as
   [string] or [binary]. *)
let in_scan schema (p : Source.Pred.t) =
  let rec columns acc : Source.Pred.t -> string list = function
    | Cmp (c, _, _) | In (c, _) | Null c | Valid c ->
        if List.mem c acc then acc else c :: acc
    | And ps | Or ps -> List.fold_left columns acc ps
    | Not p -> columns acc p
  in
  match columns [] p with
  | [ c ] -> (
      match Schema.find schema c with
      | Some (Type.Any (String | Binary)) ->
          Option.map (fun e -> (c, e)) (Source.Pred.filter schema p)
      | _ -> None)
  | _ -> None

(* [passing keep c rows] is where [keep] is [true] on the rows of the column
   [c], of [rows] rows, [keep] being a {!truth}. *)
let passing keep c rows =
  Query.of_table (Talon.v ~rows [ c ])
  |> Query.select [ keep ]
  |> Query.run |> Error.get_ok
  |> Fun.flip Talon.column "keep"
  |> Column.to_tensor Nx.bit

(* Sources *)

(* [column ~row_group name ty c] is the column [name] of type [ty] that the
   decoded chunk [c] lays out. *)
let column ~row_group name ty (c : Chunk.t) =
  let layout : Column.layout =
    match Chunk.flat c with
    | Fixed { valid; values } -> Fixed { validity = valid; values }
    | Varsize { valid; offsets; data } ->
        Varsize { validity = valid; offsets; child = Column.of_tensor data }
    | Coded _ -> assert false
  in
  match Column.of_layout ty layout with
  | Ok c -> c
  | Error (row, why) ->
      Meta.fail ~row_group "column %S: row %d: %s" name row why

(* A decoded chunk: the column [name] of type [ty] decoded as [chunk], its
   words as a column, with a null word after them that the null rows read,
   when it is coded and they are all values of [ty], and its column. Words
   that are not are read as the rows that read them, so that only a row
   fails, and an unused one fails nothing. *)
type decoded = {
  name : string;
  ty : Type.any;
  chunk : Chunk.t;
  words : Column.t option Lazy.t;
  column : Column.t Lazy.t;
}

let decoded ~row_group name ty (chunk : Chunk.t) =
  let words =
    lazy
      (match chunk with
      | Fixed _ | Varsize _ -> None
      | Coded { offsets; data; _ } -> (
          let n = Nx.dim 0 offsets - 1 in
          let last = Int64.of_int (n - 1) in
          let valid = Nx.less_s (Nx.arange Nx.int64 0 n 1) last in
          let words : Column.layout =
            Varsize
              {
                validity = Some (Nx.cast Nx.bit valid);
                offsets;
                child = Column.of_tensor data;
              }
          in
          match Column.of_layout ty words with
          | Ok words -> Some words
          | Error _ -> None))
  in
  { name; ty; chunk; words; column = lazy (column ~row_group name ty chunk) }

(* [filtered d rows e] is where [e], a {!truth}, is [true] on the [rows] rows
   of [d]: on its words, taken at its codes, or on its column. *)
let filtered d rows e =
  match (d.chunk, Lazy.force d.words) with
  | Coded { codes; _ }, Some words ->
      Nx.take codes (passing e (d.name, words) (Column.length words))
  | _ -> passing e (d.name, Lazy.force d.column) rows

(* [taken ~row_group ~rows at ds] is the batch of the rows [at] of the
   decoded chunks [ds], of [rows] rows. A coded chunk's codes are taken before its
   words are gathered; another chunk's rows once its column is built. *)
let taken ~row_group ~rows at ds =
  let coded d =
    match (d.chunk, Lazy.force d.words) with
    | Coded _, Some _ -> true
    | _ -> false
  in
  let flat =
    List.filter_map
      (fun d -> if coded d then None else Some (d.name, Lazy.force d.column))
      ds
  in
  let rest = Talon.take at (Talon.v ~rows flat) in
  let one d =
    match d.chunk with
    | Coded k when coded d ->
        let valid = Option.map (Nx.take at) k.valid in
        let codes = Nx.take at k.codes in
        let k = Chunk.Coded { k with valid; codes } in
        (d.name, column ~row_group d.name d.ty k)
    | _ -> (d.name, Talon.column rest d.name)
  in
  Talon.v ~rows:(Nx.dim 0 at) (List.map one ds)

(* [with_scratch shared f] is [f s] for [s] the scratch that the reads of one
   source share ({!Chunk.type-scratch}), taken from [shared], or one of its
   own if another part, on another domain, holds it. [s] goes back to [shared]
   however [f] ends, so that a failed read leaves it for the next. *)
let with_scratch shared f =
  let s =
    match Atomic.exchange shared None with
    | Some s -> s
    | None -> Chunk.scratch ()
  in
  match f s with
  | r ->
      Atomic.set shared (Some s);
      r
  | exception e ->
      Atomic.set shared (Some s);
      Printexc.raise_with_backtrace e (Printexc.get_raw_backtrace ())

(* [whole rows ds] is the batch of the decoded chunks [ds], of [rows] rows. *)
let whole rows ds =
  Talon.v ~rows (List.map (fun d -> (d.name, Lazy.force d.column)) ds)

(* [part ?file shared b m leaves g columns filters] is the row group [g] of [b]
   as one batch of [columns], each a name, the index of its leaf and its type,
   decoded with the scratch [shared], of the rows that pass [filters], each a
   column, as [columns] give one, and the filter on it. *)
let part ?file shared b m leaves g columns filters : Source.part =
  let rows = m.Meta.row_groups.(g).rows in
  let open_ () =
    let read = ref false in
    let next () =
      if !read then Ok None
      else begin
        read := true;
        with_bytes ?file b @@ fun bytes ->
        with_scratch shared @@ fun scratch ->
        (* Each column is decoded and built once, a filter's column being
           the request's or one of its own. *)
        let decode (name, i, ty) =
          let k = Chunk.read ~scratch bytes m ~row_group:g i leaves.(i) ty in
          decoded ~row_group:g name ty k
        in
        let ds = List.map decode columns in
        let pass (((name, _, _) as c), e) =
          let d =
            match List.find_opt (fun d -> d.name = name) ds with
            | Some d -> d
            | None -> decode c
          in
          filtered d rows e
        in
        match filters with
        | [] -> Some (whole rows ds)
        | f :: fs ->
            let keep =
              List.fold_left
                (fun acc f -> Nx.logical_and acc (pass f))
                (pass f) fs
            in
            let at = Nx.positions keep in
            if Nx.dim 0 at = rows then Some (whole rows ds)
            else Some (taken ~row_group:g ~rows at ds)
      end
    in
    Ok { Source.next; close = ignore }
  in
  { rows = (if filters = [] then Some rows else None); open_ }

(* [parts ?file shared types b r] are the parts of [b] for the request [r],
   whose columns read as [types] say, decoded with the scratch [shared]. *)
let parts ?file shared types b (r : Source.request) =
  with_bytes ?file b @@ fun bytes ->
  let m, leaves = open_ bytes in
  let leaf name =
    Array.find_index (fun (l : Leaf.t) -> String.equal l.name name) leaves
  in
  let ty name = List.assoc_opt name types in
  let requested name =
    let ty = Option.get (ty name) in
    match leaf name with
    | None -> Meta.fail "the file has no column %S" name
    | Some i when not (Leaf.reads_as leaves.(i) ty) ->
        Meta.fail "column %S reads as %s, not as the format's %s" name
          (Format.asprintf "%a" Leaf.pp_reads leaves.(i))
          (type_name ty)
    | Some i -> (name, i, ty)
  in
  let columns = List.map requested r.columns in
  (* A column's conjuncts are one filter, evaluated once. *)
  let add acc (c, e) =
    match List.assoc_opt c acc with
    | Some _ ->
        let join (c', e') = (c', if c' = c then Expr.( && ) e' e else e') in
        List.map join acc
    | None -> acc @ [ (c, e) ]
  in
  let filters =
    let schema = Schema.v types in
    List.fold_left add [] (List.filter_map (in_scan schema) r.filters)
    |> List.map (fun (c, e) -> (requested c, truth e))
  in
  let stats g name =
    match (leaf name, ty name) with
    | Some i, Some ty ->
        Option.map
          (fun (cm : Meta.column_meta) -> (leaves.(i), ty, cm.stats))
          m.row_groups.(g).chunks.(i).meta
    | _ -> None
  in
  let kept g (rg : Meta.row_group) =
    if List.for_all (passes (stats g) rg.rows) r.filters then
      Some (part ?file shared b m leaves g columns filters)
    else None
  in
  List.filter_map Fun.id (Array.to_list (Array.mapi kept m.row_groups))

(* [declared fn c] is [c]'s name and type, which a decimal must have been
   given. *)
let declared fn (c : column) =
  match c.ty with
  | Some ty -> (c.leaf.name, ty)
  | None ->
      invalid_arg
        (Format.asprintf
           "Talon_parquet.%s: column %S (%a) is a decimal, which reads as \
            float64, or as its unscaled int64 integers up to 18 digits. \
            Declare one with with_type."
           fn c.leaf.name Leaf.pp c.leaf)

let make fn ?file ?rows f b =
  let name =
    match file with
    | None -> "parquet"
    | Some p -> Format.asprintf "parquet %a" Type.pp_quoted p
  in
  let types = Array.to_list (Array.map (declared fn) f) in
  let schema = Schema.v types in
  let pushdown p =
    if Option.is_some (in_scan schema p) then Source.Exact
    else if usable f p then Inexact
    else Unsupported
  in
  let shared = Atomic.make None in
  Source.v ~name ~schema ?rows ~pushdown
    (parts ?file shared types b)

let source f b = make "source" f b

let file ?format path =
  match Rig_disk.of_file path with
  | Error why -> Error (Error.v why)
  | Ok b ->
      with_bytes ~file:path b (fun bytes ->
          let m, leaves = open_ bytes in
          let f = match format with Some f -> f | None -> sniffed leaves in
          make "file" ~file:path ~rows:m.rows f b)
