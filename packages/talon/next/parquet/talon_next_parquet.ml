(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon_next

(* [ty] is [None] for a decimal that no [with_type] declared. *)
type column = { leaf : Leaf.t; ty : Type.any option }
type format = column array

(* [with_bytes ?file b f] is [f] applied to the bytes of [b], mapped on the
   host, with the errors of the private modules as [Error]. *)
let with_bytes ?file b f =
  match Nx_device.Buffer.borrow Nx_device.host b with
  | Error why -> Error (Error.v ?file why)
  | Ok host ->
      let bytes = Nx_device.Buffer.bigarray Bigarray.int8_unsigned host in
      let r =
        try Ok (f bytes)
        with Meta.Error { row_group; bytes; text; msg } ->
          Error (Error.v ?file ?row_group ?bytes ?text msg)
      in
      (* The bigarray of a borrowed file does not keep its mapping alive. *)
      ignore (Sys.opaque_identity host);
      r

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
        (Printf.sprintf
           "Talon_next_parquet.with_type: no column %S in the format." name)

let type_name (Type.Any t) = Format.asprintf "%a" Type.pp t

let with_type name t f =
  let i = index f name in
  let leaf = f.(i).leaf in
  if not (Leaf.reads_as leaf t) then
    invalid_arg
      (Format.asprintf
         "Talon_next_parquet.with_type: column %S (%a) reads as %a, not %s. \
          Cast it once read (Expr.cast)."
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

let instant (u : Type.unit_) t =
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
  | Datetime { unit_; _ } -> fixed Int64 8 (fun s -> instant unit_ (i64 s))
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

(* Sources *)

let decode bytes m g i leaf name ty =
  let validity = Option.map Nx_bits.of_bool in
  let layout : Column.layout =
    match Chunk.read bytes m ~row_group:g i leaf ty with
    | Fixed { valid; values } -> Fixed { validity = validity valid; values }
    | Varsize { valid; offsets; data } ->
        Varsize
          { validity = validity valid; offsets; child = Column.of_tensor data }
  in
  match Column.of_layout ty layout with
  | Ok c -> c
  | Error (row, why) ->
      Meta.fail ~row_group:g "column %S: row %d: %s" name row why

(* [part ?file b m leaves g columns] is the row group [g] of [b] as one batch of
   [columns], each a name, the index of its leaf and its type. *)
let part ?file b m leaves g columns : Source.part =
  let open_ () =
    let read = ref false in
    let next () =
      if !read then Ok None
      else begin
        read := true;
        with_bytes ?file b @@ fun bytes ->
        let column (name, i, ty) =
          (name, decode bytes m g i leaves.(i) name ty)
        in
        Some
          (Talon_next.v ~rows:m.Meta.row_groups.(g).rows
             (List.map column columns))
      end
    in
    Ok { Source.next; close = ignore }
  in
  { rows = Some m.Meta.row_groups.(g).rows; open_ }

(* [parts ?file types b r] are the parts of [b] for the request [r], whose
   columns read as [types] say. *)
let parts ?file types b (r : Source.request) =
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
      Some (part ?file b m leaves g columns)
    else None
  in
  List.filter_map Fun.id (Array.to_list (Array.mapi kept m.row_groups))

(* [declared fn c] is [c]'s name and type, which a decimal must have been
   given. *)
let declared fn c =
  match c.ty with
  | Some ty -> (c.leaf.name, ty)
  | None ->
      invalid_arg
        (Format.asprintf
           "Talon_next_parquet.%s: column %S (%a) is a decimal, which reads as \
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
  let pushdown p = if usable f p then Source.Inexact else Unsupported in
  Source.v ~name ~schema:(Schema.v types) ?rows ~pushdown (parts ?file types b)

let source f b = make "source" f b

let file ?format path =
  match Nx_device.Buffer.of_file path with
  | Error why -> Error (Error.v why)
  | Ok b ->
      with_bytes ~file:path b (fun bytes ->
          let m, leaves = open_ bytes in
          let f = match format with Some f -> f | None -> sniffed leaves in
          make "file" ~file:path ~rows:m.rows f b)
