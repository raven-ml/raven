(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon_next
module Reader = Bytesrw.Bytes.Reader

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
  Printf.ksprintf (fun s -> invalid_arg ("Talon_next_csv." ^ fn ^ ": " ^ s)) fmt

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

let check_type fn name ty =
  if not (Columns.reads ty) then
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

exception Failed of Error.t

let fail_at s pos ?text msg =
  let line, column = Scan.locate s pos in
  raise (Failed (Error.v ~line ~column ?text msg))

let protect f =
  try f () with
  | Failed e -> Error e
  | Scan.Error { line; column; msg } -> Error (Error.v ~line ~column msg)
  | Bytesrw.Bytes.Stream.Error e ->
      Error (Error.v (Bytesrw.Bytes.Stream.error_message e))
  | Sys_error msg -> Error (Error.v msg)

(* Sniffing

   A text's forms are the types it reads as, as bits, with flags. A null reads
   as every type, and a column's forms are the meet of its fields' forms. *)

let bool_ = 1
let int_ = 2
let float_ = 4
let date_ = 8
let naive = 16
let zoned = 32
let null = 63
let fine = 64 (* Not a whole number of microseconds. *)
let wide = 128 (* Outside the nanoseconds' range. *)
let seen = 256 (* Not null. *)

let meet a b =
  let common = a land b and either = a lor b in
  common land null lor (either land lnot null)

(* [is_integer b pos len] is [true] iff the text is digits after an optional
   sign. *)
let is_integer b pos len =
  let stop = pos + len in
  let first =
    if len > 0 && (Bytes.get b pos = '-' || Bytes.get b pos = '+') then pos + 1
    else pos
  in
  let rec digits i =
    i = stop || (Char.Ascii.is_digit (Bytes.get b i) && digits (i + 1))
  in
  first < stop && digits first

let forms ~nulls ~ticks ~floats s r j =
  let b = Scan.bytes s and pos = Scan.pos s r j and len = Scan.len s r j in
  let reads parse =
    match parse b pos len with () -> true | exception Text.Invalid _ -> false
  in
  let number = not (Text.leading_zero b pos len) in
  let in_unit ~zoned u =
    reads (fun b p l -> Text.datetime u ~zoned b p l ticks 0)
  in
  let datetime ~zoned bit =
    if in_unit ~zoned Us then bit lor if in_unit ~zoned Ns then 0 else wide
    else if in_unit ~zoned Ns then bit lor fine
    else 0
  in
  if Columns.is_null nulls s r j then null
  else
    seen
    lor
    if reads (fun b p l -> ignore (Text.bool b p l)) then bool_
    else if number && reads (fun b p l -> Text.int64 b p l ticks 0) then
      int_ lor float_
    else if is_integer b pos len then 0
    else if number && reads (fun b p l -> Text.float b p l floats 0) then float_
    else if reads (fun b p l -> ignore (Text.date b p l)) then date_
    else
      let forms = datetime ~zoned:false naive in
      if forms <> 0 then forms else datetime ~zoned:true zoned

(* [infer forms] is the type that columns of [forms] read as, and its bit, [0]
   for [string]. *)
let infer forms : int * Type.any =
  let datetime bit zone =
    if forms land fine = 0 then (bit, Type.Any (Type.datetime ?zone Us))
    else if forms land wide = 0 then (bit, Type.Any (Type.datetime ?zone Ns))
    else (0, Type.Any Type.string)
  in
  if forms land seen = 0 then (0, Type.Any Type.string)
  else if forms land bool_ <> 0 then (bool_, Type.Any Type.bool)
  else if forms land int_ <> 0 then (int_, Type.Any Type.int64)
  else if forms land float_ <> 0 then (float_, Type.Any Type.float64)
  else if forms land date_ <> 0 then (date_, Type.Any Type.date)
  else if forms land naive <> 0 then datetime naive None
  else if forms land zoned <> 0 then datetime zoned (Some "UTC")
  else (0, Type.Any Type.string)

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
      let line, column = n.at in
      let fail msg = raise (Failed (Error.v ~line ~column ~text:n.raw msg)) in
      if not (String.is_valid_utf_8 n.text) then
        fail "the column name is not UTF-8";
      for k = 0 to j - 1 do
        if String.equal names.(k).text n.text then
          fail "the header names two columns the same"
      done)
    names

let sniff ?(rows = 16384) ?(nulls = []) r =
  if rows < 1 then invalid "sniff" "rows is %d, not positive" rows;
  check_nulls "sniff" ~seps:separators ~quote:'"' nulls;
  protect @@ fun () ->
  let sample = Scan.sample ~quote:'"' ~records:(rows + 1) r in
  let sep = separator sample in
  let ticks = Bigarray.(Array1.create int64 c_layout 1) in
  let floats = Bigarray.(Array1.create float64 c_layout 1) in
  let names = ref [||] and all = ref [] in
  records ~sep sample (fun s r ->
      if !all = [] then
        names :=
          Array.init (Scan.fields s r) (fun j ->
              {
                text = Scan.text s r j;
                raw = Scan.raw s r j;
                at = Scan.locate s (Scan.start s r j);
              });
      all :=
        Array.init (Scan.fields s r) (forms ~nulls ~ticks ~floats s r) :: !all);
  if !all = [] then raise (Failed (Error.v "the input holds no record"));
  let all = Array.of_list (List.rev !all) in
  let cols = Array.length !names and n = Array.length all in
  let meet_rows first last =
    Array.init cols (fun j ->
        let f = ref null in
        for r = first to last do
          f := meet !f all.(r).(j)
        done;
        !f)
  in
  let types = Array.map infer (meet_rows 1 (n - 1)) in
  let typed = Array.exists (fun (bit, _) -> bit <> 0) types in
  let header =
    not
      (typed
      && Array.for_all2
           (fun (bit, _) f -> bit = 0 || f land bit <> 0)
           types all.(0))
  in
  let used = if header then n - 1 else min n rows in
  let types =
    if header then types else Array.map infer (meet_rows 0 (used - 1))
  in
  if header then check_names_read !names;
  let name j =
    if header then !names.(j).text else Printf.sprintf "column_%d" (j + 1)
  in
  let columns =
    Array.mapi (fun j (_, ty) -> { name = name j; ty; declared = false }) types
  in
  Ok { sep; quote = '"'; header; nulls; columns; sniffed = Some used }

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

(* A batch's failure is at its earliest row, then its leftmost column. *)
let batch f readers s ~first ~rows =
  let failure = ref None in
  let read j c =
    match Columns.read c s j ~first ~rows with
    | column -> Some column
    | exception Columns.Invalid { row; reason } ->
        (match !failure with
        | Some (r, _, _) when r <= row -> ()
        | _ -> failure := Some (row, j, reason));
        None
  in
  let columns = Array.mapi read readers in
  match !failure with
  | Some (row, j, reason) -> value_error f s row j reason
  | None -> Array.map Option.get columns

module Private = struct
  type column = Columns.t =
    | Fixed of { valid : Nx.bool_t option; values : Nx.packed }
    | Varsize of {
        valid : Nx.bool_t option;
        offsets : Nx.int64_t;
        data : Nx.uint8_t;
      }

  let read f r =
    protect @@ fun () ->
    let s = Scan.make ~sep:f.sep ~quote:f.quote r in
    let readers =
      Array.map
        (fun c -> Columns.reader ~quote:f.quote ~nulls:f.nulls c.ty)
        f.columns
    in
    let cols = Array.length f.columns in
    let header = ref f.header and batches = ref [] in
    while Scan.next s do
      let first =
        if !header then begin
          check_header f s;
          1
        end
        else 0
      in
      header := false;
      let rows = Scan.rows s and last = ref first in
      while !last < rows && Scan.fields s !last = cols do
        incr last
      done;
      if !last > first then
        batches := batch f readers s ~first ~rows:(!last - first) :: !batches;
      if !last < rows then fields_error f s !last
    done;
    if !header then
      raise (Failed (Error.v "the input is empty, and the format has a header"));
    Ok (List.rev !batches)
end
