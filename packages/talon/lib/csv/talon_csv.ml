(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon
module Reader = Bytesrw.Bytes.Reader
module A1 = Bigarray.Array1

type column = { name : string; ty : Type.any; declared : bool }

type format = {
  sep : char;
  quote : char;
  header : bool;
  nulls : string list;
  columns : column array;
  sniffed : int option; (* The rows sniff inferred the types from. *)
}

let invalid fn fmt =
  Printf.ksprintf (fun s -> invalid_arg ("Talon_csv." ^ fn ^ ": " ^ s)) fmt

let type_name (Type.Any t) = Format.asprintf "%a" Type.pp t
let plural n word = Printf.sprintf "%d %s%s" n word (if n = 1 then "" else "s")
let quoted s = Format.asprintf "%a" Type.pp_quoted s

(* Formats *)

let is_break c = c = '\n' || c = '\r'

let check_names fn names =
  let seen = Hashtbl.create 16 in
  List.iter
    (fun n ->
      if not (String.is_valid_utf_8 n) then
        invalid fn "the column name %s is not UTF-8" (quoted n);
      if Hashtbl.mem seen n then
        invalid fn "two columns are named %s" (quoted n);
      Hashtbl.add seen n ())
    names

let reads (Type.Any t) =
  match t with
  | Bool | Int8 | Int16 | Int32 | Int64 | Uint8 | Uint16 | Uint32 | Uint64
  | Float16 | Float32 | Float64 | String | Binary | Categorical _ | Date
  | Datetime _ ->
      true
  | Clock _ | Duration _ | List _ | Record _ | Tensor _ | Ext _ -> false

let check_type fn name ty =
  if not (reads ty) then
    invalid fn "column %s: CSV does not read %s" (quoted name) (type_name ty)

let check_nulls fn ~seps ~quote nulls =
  List.iter
    (fun t ->
      if String.exists (fun c -> List.mem c seps || c = quote || is_break c) t
      then
        invalid fn
          "the null token %s holds a separator, the quote or a line break"
          (quoted t))
    nulls

let format ?(sep = ',') ?(quote = '"') ?(header = true) ?(nulls = []) columns =
  let fn = "format" in
  if columns = [] then invalid fn "a format has at least one column";
  if is_break sep || is_break quote || sep >= '\x80' || quote >= '\x80' then
    invalid fn
      "the separator and the quote are ASCII bytes other than a line break";
  if sep = quote then invalid fn "the separator and the quote are both %C" sep;
  check_nulls fn ~seps:[ sep ] ~quote nulls;
  check_names fn (List.map fst columns);
  List.iter (fun (name, ty) -> check_type fn name ty) columns;
  let columns =
    Array.of_list
      (List.map (fun (name, ty) -> { name; ty; declared = true }) columns)
  in
  { sep; quote; header; nulls; columns; sniffed = None }

let with_type name ty f =
  let fn = "with_type" in
  let rec index i =
    if i = Array.length f.columns then
      invalid fn "no column %s in the format" (quoted name)
    else if String.equal f.columns.(i).name name then i
    else index (i + 1)
  in
  let i = index 0 in
  check_type fn name ty;
  let columns = Array.copy f.columns in
  columns.(i) <- { name; ty; declared = true };
  { f with columns }

let thousands n =
  let s = string_of_int n and b = Buffer.create 16 in
  String.iteri
    (fun i c ->
      if i > 0 && (String.length s - i) mod 3 = 0 then Buffer.add_char b ',';
      Buffer.add_char b c)
    s;
  Buffer.contents b

(* [scalars s] is the number of Unicode scalar values of the UTF-8 [s]. *)
let scalars s =
  let n = ref 0 in
  String.iter (fun c -> if Char.code c land 0xC0 <> 0x80 then incr n) s;
  !n

let pp_format ppf f =
  let n = Array.length f.columns in
  Format.fprintf ppf "@[<v>csv (%s, separator %C, quote %C, %s"
    (plural n "column") f.sep f.quote
    (if f.header then "header" else "no header");
  if f.nulls <> [] then
    Format.fprintf ppf ", nulls [%s]"
      (String.concat "; " (List.map quoted f.nulls));
  Format.fprintf ppf ")";
  Option.iter
    (fun rows ->
      Format.fprintf ppf ", types sniffed from %s"
        (match rows with
        | 0 -> "no rows"
        | 1 -> "1 row"
        | n -> thousands n ^ " rows"))
    f.sniffed;
  let left =
    Array.map
      (fun c ->
        let (Type.Any ty) = c.ty in
        Format.asprintf "%a %a" Type.pp_name c.name Type.pp ty)
      f.columns
  in
  let width = Array.fold_left (fun w s -> max w (scalars s)) 0 left in
  Array.iteri
    (fun i c ->
      Format.fprintf ppf "@,  %s" left.(i);
      if f.sniffed <> None && c.declared then
        Format.fprintf ppf "%s declared"
          (String.make (width - scalars left.(i)) ' '))
    f.columns;
  Format.fprintf ppf "@]"

(* Errors *)

exception
  Failed of { at : (int * int) option; text : string option; msg : string }

let fail ?at ?text msg = raise (Failed { at; text; msg })
let fail_at s pos ?text msg = fail ~at:(Scan.locate s pos) ?text msg

(* [file] is the path the errors name. A [Sys_error] names its own. *)
let protect ?file f =
  let error ?line ?column ?text msg =
    Error (Error.v ?file ?line ?column ?text msg)
  in
  try f () with
  | Failed { at = Some (line, column); text; msg } ->
      error ~line ~column ?text msg
  | Failed { at = None; text; msg } -> error ?text msg
  | Scan.Error { line; column; msg } -> error ~line ~column msg
  | Bytesrw.Bytes.Stream.Error e -> error (Bytesrw.Bytes.Stream.error_message e)
  | Sys_error msg -> Error (Error.v msg)

(* Fields *)

let rec same_from b pos s i =
  i = String.length s
  || Bytes.unsafe_get b (pos + i) = String.unsafe_get s i
     && same_from b pos s (i + 1)

let rec is_token b pos len = function
  | [] -> false
  | t :: ts ->
      (String.length t = len && same_from b pos t 0) || is_token b pos len ts

let is_null nulls s r j =
  (not (Scan.quoted s r j))
  &&
  let len = Scan.len s r j in
  len = 0 || is_token (Scan.bytes s) (Scan.pos s r j) len nulls

let tensor a = Nx.of_bigarray (Bigarray.genarray_of_array1 a)

(* A binary column has no value to check, so [of_layout] cannot fail. *)
let binary ?validity offsets data =
  let length = A1.dim offsets - 1 in
  let validity = Option.map (fun v -> Nx_bits.v ~length (tensor v)) validity in
  Result.get_ok
    (Column.of_layout (Type.Any Type.binary)
       (Varsize
          {
            validity;
            offsets = tensor offsets;
            child = Column.of_tensor (tensor data);
          }))

let no_texts =
  binary
    (A1.init Bigarray.int64 Bigarray.c_layout 1 (Fun.const 0L))
    (A1.create Bigarray.int8_unsigned Bigarray.c_layout 0)

(* [texts ~quote ~nulls s j ~first ~rows] is the binary column of field [j] of
   the records [first] to [first + rows - 1] of [s]'s batch, its doubled quotes
   undoubled, null where the field is. *)
let texts ~quote ~nulls s j ~first ~rows =
  let b = Scan.bytes s in
  let total = ref 0 in
  for r = first to first + rows - 1 do
    total := !total + Scan.len s r j
  done;
  let data = A1.create Bigarray.int8_unsigned Bigarray.c_layout !total in
  let offsets = A1.create Bigarray.int64 Bigarray.c_layout (rows + 1) in
  A1.unsafe_set offsets 0 0L;
  let validity = ref None and o = ref 0 in
  (* The validity's bytes are made at the first null. *)
  let null i =
    let v =
      match !validity with
      | Some v -> v
      | None ->
          let v =
            A1.create Bigarray.int8_unsigned Bigarray.c_layout ((rows + 7) / 8)
          in
          A1.fill v 0xFF;
          validity := Some v;
          v
    in
    A1.unsafe_set v (i / 8) (A1.unsafe_get v (i / 8) land lnot (1 lsl (i mod 8)))
  in
  for i = 0 to rows - 1 do
    let r = first + i in
    if is_null nulls s r j then null i
    else begin
      let pos = Scan.pos s r j and len = Scan.len s r j in
      let quoted = Scan.quoted s r j in
      let k = ref pos in
      while !k < pos + len do
        let c = Bytes.unsafe_get b !k in
        A1.unsafe_set data !o (Char.code c);
        incr o;
        k := if quoted && c = quote then !k + 2 else !k + 1
      done
    end;
    A1.unsafe_set offsets (i + 1) (Int64.of_int !o)
  done;
  binary ?validity:!validity offsets (A1.sub data 0 !o)

(* Sniffing

   A column's type is the first candidate that each of its sampled values reads
   as, by [Column.parse] over the column's texts with its nulls and the rows
   outside the sample masked. *)

let candidates =
  Type.
    [
      Any bool;
      Any int64;
      Any float64;
      Any date;
      Any (datetime Us);
      Any (datetime Ns);
      Any (datetime ~zone:"UTC" Us);
      Any (datetime ~zone:"UTC" Ns);
    ]

let is_digit c = '0' <= c && c <= '9'

let after_sign b pos len =
  if len > 0 && (Bytes.get b pos = '-' || Bytes.get b pos = '+') then pos + 1
  else pos

(* [is_integer b pos len] is [true] iff the text is digits after an optional
   sign. *)
let is_integer b pos len =
  let first = after_sign b pos len and stop = pos + len in
  let rec digits i = i = stop || (is_digit (Bytes.get b i) && digits (i + 1)) in
  first < stop && digits first

(* [leading_zero b pos len] is [true] iff the text, after an optional sign,
   starts with a zero followed by a digit, as [007] and [-01.5] do. *)
let leading_zero b pos len =
  let i = after_sign b pos len in
  i + 1 < pos + len && Bytes.get b i = '0' && is_digit (Bytes.get b (i + 1))

(* A batch of sampled records: each column's texts, and for each of its rows,
   whether the field is a value, an integer, and a number with a leading
   zero. *)
type sampled = {
  first : int; (* The batch's first record in the sample. *)
  texts : Column.t array;
  value : bool array array;
  integer : bool array array;
  lead : bool array array;
}

let sampled ~nulls s ~first =
  let rows = Scan.rows s and cols = Scan.fields s 0 in
  let each f =
    Array.init cols (fun j ->
        Array.init rows (fun r ->
            f (Scan.bytes s) (Scan.pos s r j) (Scan.len s r j)))
  in
  {
    first;
    texts =
      Array.init cols (fun j -> texts ~quote:'"' ~nulls s j ~first:0 ~rows);
    value =
      Array.init cols (fun j ->
          Array.init rows (fun r -> not (is_null nulls s r j)));
    integer = each is_integer;
    lead = each leading_zero;
  }

let masked c mask =
  match Column.layout c with
  | Varsize { offsets; child; _ } ->
      let validity =
        Some (Nx_bits.of_bool (Nx.create Nx.bool [| Array.length mask |] mask))
      in
      Result.get_ok
        (Column.of_layout (Type.Any Type.binary)
           (Varsize { validity; offsets; child }))
  | Fixed _ | Children _ -> assert false

(* [reads_as ty b j keep] is [true] iff the values of column [j] of [b] at the
   records [keep] holds read as [ty]. Numbers have no leading zero, and an
   integer reads as [float64] only if [int64] holds it. *)
let reads_as ty b j keep =
  let mask extra =
    Array.init
      (Array.length b.value.(j))
      (fun r -> b.value.(j).(r) && keep (b.first + r) && extra r)
  in
  let parses ty m =
    (not (Array.mem true m))
    || Result.is_ok (Column.parse ty (masked b.texts.(j) m))
  in
  let m = mask (Fun.const true) in
  let number () = not (Array.exists2 ( && ) m b.lead.(j)) in
  match ty with
  | Type.Any Int64 -> number () && parses ty m
  | Type.Any Float64 ->
      number () && parses ty m
      && parses (Type.Any Type.int64) (mask (fun r -> b.integer.(j).(r)))
  | _ -> parses ty m

let reads_all batches ty j keep =
  List.for_all (fun b -> reads_as ty b j keep) batches

let infer batches j keep =
  let seen b =
    Array.exists Fun.id
      (Array.mapi (fun r v -> v && keep (b.first + r)) b.value.(j))
  in
  if not (List.exists seen batches) then Type.Any Type.string
  else
    Option.value ~default:(Type.Any Type.string)
      (List.find_opt (fun ty -> reads_all batches ty j keep) candidates)

let is_text = function Type.Any String -> true | _ -> false

let records ~sep sample f =
  let s = Scan.make ~sep ~quote:'"' (Reader.of_string sample) in
  while Scan.next s do
    for r = 0 to Scan.rows s - 1 do
      f s r
    done
  done

let separators = [ ','; '\t'; ';'; '|' ]

(* [width ~sep sample] is the number of fields of every record of [sample] split
   at [sep]. *)
let width ~sep sample =
  let n = ref (-1) in
  records ~sep sample (fun s r ->
      let k = Scan.fields s r in
      if !n < 0 then n := k
      else if k <> !n then
        let line, column = Scan.locate s (Scan.start s r 0) in
        raise
          (Scan.Error
             {
               line;
               column;
               msg =
                 Printf.sprintf
                   "no separator splits every record into the same number of \
                    fields: with %C, the record has %s, and the first one %d"
                   sep (plural k "field") !n;
             }));
  !n

(* A file that no separator splits is one column. One that a separator splits
   unevenly fails, so that it never reads as one column. *)
let separator sample =
  let widths =
    List.map
      (fun sep ->
        (sep, try Ok (width ~sep sample) with Scan.Error _ as e -> Error e))
      separators
  in
  let wider best (sep, w) =
    match (w, best) with
    | Ok w, Some (_, b) when w <= b -> best
    | Ok w, _ when w > 1 -> Some (sep, w)
    | _ -> best
  in
  match List.fold_left wider None widths with
  | Some (sep, _) -> sep
  | None -> (
      match
        List.find_map (function _, Error e -> Some e | _, Ok _ -> None) widths
      with
      | Some e -> raise e
      | None -> ',')

(* A field of the first record, which may name a column. *)
type name = { text : string; raw : string; at : int * int }

let check_names_read names =
  Array.iteri
    (fun j n ->
      let fail msg = fail ~at:n.at ~text:n.raw msg in
      if not (String.is_valid_utf_8 n.text) then
        fail "the column name is not UTF-8";
      for k = 0 to j - 1 do
        if String.equal names.(k).text n.text then
          fail "the header names two columns the same"
      done)
    names

let sniff_from ?file ~rows ~nulls r =
  protect ?file @@ fun () ->
  let sample = Scan.sample ~quote:'"' ~records:(rows + 1) r in
  let sep = separator sample in
  let s = Scan.make ~sep ~quote:'"' (Reader.of_string sample) in
  let names = ref [||] and batches = ref [] and n = ref 0 in
  while Scan.next s do
    if !n = 0 then
      names :=
        Array.init (Scan.fields s 0) (fun j ->
            {
              text = Scan.text s 0 j;
              raw = Scan.raw s 0 j;
              at = Scan.locate s (Scan.start s 0 j);
            });
    batches := sampled ~nulls s ~first:!n :: !batches;
    n := !n + Scan.rows s
  done;
  if !n = 0 then fail "the input holds no record";
  let batches = List.rev !batches and n = !n in
  let infer keep =
    Array.init (Array.length !names) (fun j -> infer batches j keep)
  in
  let types = infer (fun r -> r >= 1) in
  let reads_first j ty =
    is_text ty || reads_all batches ty j (fun r -> r = 0)
  in
  let header =
    Array.for_all is_text types
    || not (Array.for_all Fun.id (Array.mapi reads_first types))
  in
  let used = if header then n - 1 else min n rows in
  let types = if header then types else infer (fun r -> r < used) in
  if header then check_names_read !names;
  let name j =
    if header then !names.(j).text else Printf.sprintf "column_%d" (j + 1)
  in
  let columns =
    Array.mapi (fun j ty -> { name = name j; ty; declared = false }) types
  in
  Ok { sep; quote = '"'; header; nulls; columns; sniffed = Some used }

let sample_rows = 16384

let sniff ?(rows = sample_rows) ?(nulls = []) r =
  if rows < 1 then invalid "sniff" "rows is %d, not positive" rows;
  check_nulls "sniff" ~seps:separators ~quote:'"' nulls;
  sniff_from ~rows ~nulls r

(* Reading *)

let fields_error f s r =
  let n = Scan.fields s r and cols = Array.length f.columns in
  let at = if n > cols then Scan.start s r cols else Scan.stop s r (n - 1) in
  fail_at s at
    (Printf.sprintf "the record has %s, and the format %s" (plural n "field")
       (plural cols "column"))

let check_header f s =
  if Scan.fields s 0 <> Array.length f.columns then fields_error f s 0;
  Array.iteri
    (fun j c ->
      if not (String.equal (Scan.text s 0 j) c.name) then
        fail_at s (Scan.start s 0 j) ~text:(Scan.raw s 0 j)
          (Printf.sprintf "the header does not name column %d %s" (j + 1)
             (quoted c.name)))
    f.columns

(* A string column fails only on invalid UTF-8, so its type names its fix, and a
   column sniffed from no rows is a string column. *)
let value_error f s row j reason =
  let c = f.columns.(j) in
  let fix =
    match (f.sniffed, c.ty) with
    | _, Type.Any Bool ->
        "Declare the null token (~nulls), or read the column as string \
         (with_type) and map its spellings with an expression."
    | _, Type.Any String -> "Read the column as binary (with_type)."
    | Some rows, _ when not c.declared ->
        Printf.sprintf
          "The type was sniffed from %s. Declare the null token (~nulls) or \
           the type (with_type)."
          (if rows = 1 then "row 1" else "rows 1 to " ^ thousands rows)
    | _ -> "Declare the null token (~nulls) or another type (with_type)."
  in
  fail_at s (Scan.start s row j) ~text:(Scan.raw s row j)
    (Printf.sprintf "column %s: cannot read as %s: %s. %s" (quoted c.name)
       (type_name c.ty) reason fix)

(* A reading of a text: its scanner, the columns it reads, by position, and the
   rows it may still yield. *)
type reading = {
  f : format;
  s : Scan.t;
  read : int list;
  mutable header : bool;
  mutable left : int;
}

let reading f ~read ~limit r =
  let s = Scan.make ~sep:f.sep ~quote:f.quote r in
  { f; s; read; header = f.header; left = Option.value ~default:max_int limit }

(* A batch's failure is at its earliest row, then its leftmost column. *)
let batch { f; s; read; _ } ~first ~rows =
  let parse j =
    let texts = texts ~quote:f.quote ~nulls:f.nulls s j ~first ~rows in
    (j, Column.parse f.columns.(j).ty texts)
  in
  let parsed = List.map parse read in
  let earliest failure (j, p) =
    match (p, failure) with
    | Error (row, _), Some (r, _, _) when r <= row -> failure
    | Error (row, reason), _ -> Some (row, j, reason)
    | Ok _, _ -> failure
  in
  match List.fold_left earliest None parsed with
  | Some (row, j, reason) -> value_error f s (first + row) j reason
  | None ->
      Talon.v ~rows
        (List.map (fun (j, p) -> (f.columns.(j).name, Result.get_ok p)) parsed)

(* [next r] is the next batch of [r], which stops at its limit. A batch ends
   before a record of another number of fields, which then fails. *)
let rec next r =
  if r.left = 0 then None
  else if not (Scan.next r.s) then begin
    if r.header then fail "the input is empty, and the format has a header";
    None
  end
  else begin
    let first = if r.header then 1 else 0 in
    if r.header then check_header r.f r.s;
    r.header <- false;
    let rows = Scan.rows r.s and cols = Array.length r.f.columns in
    let last = ref first in
    while
      !last < rows && !last - first < r.left && Scan.fields r.s !last = cols
    do
      incr last
    done;
    let n = !last - first in
    let b = if n > 0 then Some (batch r ~first ~rows:n) else None in
    if !last < rows && n < r.left then fields_error r.f r.s !last;
    r.left <- r.left - n;
    match b with Some _ -> b | None -> next r
  end

let schema f =
  Schema.v (Array.to_list (Array.map (fun c -> (c.name, c.ty)) f.columns))

let decode f r =
  protect @@ fun () ->
  let r =
    reading f ~read:(List.init (Array.length f.columns) Fun.id) ~limit:None r
  in
  let rec batches acc =
    match next r with Some b -> batches (b :: acc) | None -> List.rev acc
  in
  match batches [] with
  | [] ->
      let empty c = (c.name, Result.get_ok (Column.parse c.ty no_texts)) in
      Ok (Talon.v (Array.to_list (Array.map empty f.columns)))
  | bs -> Ok (Talon.of_batches bs)

(* [open_ ()] is a reader and the function that closes it. *)
let source_of ~name ?file f open_ =
  let index n =
    let rec find j =
      if String.equal f.columns.(j).name n then j else find (j + 1)
    in
    find 0
  in
  let parts (q : Source.request) =
    let read = List.map index q.columns in
    let open_ () =
      protect ?file @@ fun () ->
      let r, close = open_ () in
      let r = reading f ~read ~limit:q.limit r in
      let next () = protect ?file (fun () -> Ok (next r)) in
      Ok { Source.next; close }
    in
    Ok [ { Source.rows = None; open_ } ]
  in
  Source.v ~name ~schema:(schema f) parts

let source f open_ = source_of ~name:"csv" f (fun () -> (open_ (), ignore))

let file ?format ?nulls path =
  let open_ () =
    let ic = open_in_bin path in
    (Reader.of_in_channel ic, fun () -> close_in_noerr ic)
  in
  let source f =
    let name = Format.asprintf "csv %a" Type.pp_quoted path in
    source_of ~name ~file:path f open_
  in
  match (format, nulls) with
  | Some _, Some _ ->
      invalid "file" "a format holds its null tokens, so ~nulls is for sniffing"
  | Some f, None -> Ok (source f)
  | None, nulls -> (
      let nulls = Option.value ~default:[] nulls in
      check_nulls "file" ~seps:separators ~quote:'"' nulls;
      match open_ () with
      | exception Sys_error msg -> Error (Error.v msg)
      | r, close ->
          Fun.protect ~finally:close @@ fun () ->
          Result.map source (sniff_from ~file:path ~rows:sample_rows ~nulls r))

(* Writing *)

(* A field's bytes: [get k] is its byte [k] of [len]. *)
type field = { get : int -> char; len : int }

let of_string s = { get = String.unsafe_get s; len = String.length s }

(* A field is quoted when reading it unquoted would not give it back: empty
   text, which is null unquoted, a null token, a leading quote, and the bytes
   that end or break an unquoted field. *)
let needs_quote f { get; len } =
  let token t =
    String.length t = len
    &&
    let rec same k =
      k = len || (get k = String.unsafe_get t k && same (k + 1))
    in
    same 0
  in
  let rec special k =
    k < len
    &&
    let c = get k in
    c = f.sep || c = f.quote || is_break c || special (k + 1)
  in
  len = 0 || special 0 || List.exists token f.nulls

let add_field b f ({ get; len } as x) =
  if not (needs_quote f x) then
    for k = 0 to len - 1 do
      Buffer.add_char b (get k)
    done
  else begin
    Buffer.add_char b f.quote;
    for k = 0 to len - 1 do
      let c = get k in
      if c = f.quote then Buffer.add_char b c;
      Buffer.add_char b c
    done;
    Buffer.add_char b f.quote
  end

(* [cells c] is the field of each row of the text column [c], [None] where it is
   null. *)
let cells c =
  match Column.layout c with
  | Fixed _ | Children _ -> assert false (* [Column.print] makes text. *)
  | Varsize { validity; offsets; child } ->
      let valid = Option.map (fun v -> Nx.to_array (Nx_bits.to_bool v)) validity
      and o = Nx.to_array offsets
      and a =
        Bigarray.array1_of_genarray
          (Nx.to_bigarray (Column.to_tensor Nx.uint8 child))
      in
      fun i ->
        if Option.fold ~none:false ~some:(fun v -> not v.(i)) valid then None
        else
          let pos = Int64.to_int o.(i) in
          let get k = Char.unsafe_chr (A1.unsafe_get a (pos + k)) in
          Some { get; len = Int64.to_int o.(i + 1) - pos }

let encode ?(eod = false) f q w =
  let s = schema f in
  if not (Schema.equal (Query.schema q) s) then
    invalid "encode" "the query's columns are not the format's: %s, against %s"
      (Format.asprintf "%a" Schema.pp (Query.schema q))
      (Format.asprintf "%a" Schema.pp s);
  protect @@ fun () ->
  let b = Buffer.create 65536 and line = ref 1 in
  let flush () =
    Bytesrw.Bytes.Writer.write_string w (Buffer.contents b);
    Buffer.clear b
  in
  (* A null is an empty field, but a record of one empty field is an empty line,
     which is no record: one column writes a null as a null token. *)
  let null =
    match (f.columns, f.nulls) with
    | [| _ |], t :: _ -> fun () -> Buffer.add_string b t
    | [| c |], [] ->
        fun () ->
          flush ();
          fail ~at:(!line, 1)
            (Printf.sprintf
               "column %s: a null in the only column is an empty line, which \
                is no record. Declare a null token (~nulls)."
               (quoted c.name))
    | _ -> ignore
  in
  if f.header then begin
    Array.iteri
      (fun j c ->
        if j > 0 then Buffer.add_char b f.sep;
        add_field b f (of_string c.name))
      f.columns;
    Buffer.add_char b '\n';
    incr line
  end;
  let batch () t =
    let cells =
      Array.map
        (fun c -> cells (Column.print (Talon.column t c.name)))
        f.columns
    in
    for i = 0 to Talon.rows t - 1 do
      for j = 0 to Array.length cells - 1 do
        if j > 0 then Buffer.add_char b f.sep;
        match cells.(j) i with None -> null () | Some x -> add_field b f x
      done;
      Buffer.add_char b '\n';
      incr line
    done;
    flush ()
  in
  let r = Query.fold q ~init:() batch in
  flush ();
  if eod && Result.is_ok r then Bytesrw.Bytes.Writer.write_eod w;
  r
