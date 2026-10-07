(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Binary and ASCII tables (FITS 4.0 §7): columns read as nx tensors, text
   and heap arrays as ragged arrays, each with the validity of its
   elements; and binary tables written from columns. *)

open Err
module A = Bigarray.Array1
module S = Nx_dtype.Scalar
module B = Nx_device.Buffer

let strf = Printf.sprintf

type bigbytes = Checksum.bigbytes
type layout = Array of int array | Lists | Text of int array

type column = {
  name : string;
  element : S.t;
  layout : layout;
  scaled : bool;
  cards : Header.t;
}

(* Fields *)

type tnull = No_null | Int_null of int64 | Text_null of string

(* Where a column's cells are in a row. A binary field reads [count]
   elements of its form; a heap field reads its descriptor. An ASCII field
   is [width] bytes from [start]. *)
type field =
  | Bin of { form : Bintable.form; start : int; count : int; width : int }
  | Ascii of { code : char; width : int; decimals : int; start : int }

type col = {
  column : column;
  number : int;
  field : field;
  tscal : float;
  tzero : float;
  tnull : tnull;
  string_width : int;  (** an A field's string length *)
  unit_text : string option;
  place : Err.place;
}

type t = {
  place : Err.place;
  rows : int;
  row_bytes : int;
  cols : col array;
  heap : Bintable.t option;  (** the binary layout, for its heap *)
  store : Hdu.store;
  ascii : bool;
}

let rows t = t.rows
let columns t = Array.to_list (Array.map (fun c -> c.column) t.cols)

(* Column keywords *)

let structure_roots =
  [ "TTYPE"; "TFORM"; "TDIM"; "TNULL"; "TSCAL"; "TZERO"; "TBCOL" ]

(* [numbered k] splits a column keyword into its name without the number,
   alternate letter kept, and the number: TCTYP3A is (TCTYPA, 3). *)
let numbered k =
  if not (Structure.column_numbered k) then None
  else
    let n = String.length k in
    let alt, k' =
      if
        n >= 2
        && k.[n - 1] >= 'A'
        && k.[n - 1] <= 'Z'
        && Structure.is_digit k.[n - 2]
      then (String.make 1 k.[n - 1], String.sub k 0 (n - 1))
      else ("", k)
    in
    let m = String.length k' in
    let i = ref m in
    while !i > 0 && Structure.is_digit k'.[!i - 1] do
      decr i
    done;
    if !i = 0 || !i = m then None
    else
      Some (String.sub k' 0 !i ^ alt, int_of_string (String.sub k' !i (m - !i)))

let rename r key =
  Printf.sprintf "%-8s" key ^ String.sub r 8 (String.length r - 8)

(* Each column's keywords other than its structure, the number removed. *)
let column_cards h nfields =
  let cards = Array.make (nfields + 1) [] in
  let records = Array.of_list (Header.records h) in
  let i = ref 0 in
  while !i < Array.length records do
    let span = Header.span_of records !i in
    (match Header.keyword records.(!i) with
    | Some (k, _) -> (
        match numbered k with
        | Some (root, n) when n >= 1 && n <= nfields ->
            if not (List.mem root structure_roots) then
              cards.(n) <-
                cards.(n)
                @ rename records.(!i) root
                  :: Array.to_list (Array.sub records (!i + 1) (span - 1))
        | _ -> ())
    | None -> ());
    i := !i + span
  done;
  Array.map (fun rs -> Header.of_records (Array.of_list rs)) cards

(* Description *)

let elt_element (e : Bintable.elt) : S.t =
  match e with
  | L -> Bool
  | X -> Bit
  | A | B -> UInt8
  | I -> Int16
  | J -> Int32
  | K -> Int64
  | E -> Float32
  | D -> Float64
  | C -> Complex64
  | M -> Complex128

let offset_of (e : Bintable.elt) =
  match e with
  | B -> Some ("-128", S.Int8)
  | I -> Some ("32768", UInt16)
  | J -> Some ("2147483648", UInt32)
  | K -> Some ("9223372036854775808", UInt64)
  | _ -> None

let numeric (e : Bintable.elt) =
  match e with B | I | J | K | E | D | C | M -> true | _ -> false

let tdim h k =
  match Header.find_struct Value.string k h with
  | Error e -> fail "%s" e
  | Ok None -> None
  | Ok (Some s) -> (
      let s = String.trim s in
      let n = String.length s in
      if n < 2 || s.[0] <> '(' || s.[n - 1] <> ')' then
        Hdu.card_fail h k (strf "%S is not (n1,n2,...)" s);
      let parts = String.split_on_char ',' (String.sub s 1 (n - 2)) in
      try
        Some
          (Array.of_list
             (List.map
                (fun p ->
                  match int_of_string_opt (String.trim p) with
                  | Some v when v >= 0 -> v
                  | _ -> raise Exit)
                parts))
      with Exit -> Hdu.card_fail h k (strf "%S is not (n1,n2,...)" s))

let product place what dims =
  Array.fold_left
    (fun acc n ->
      match Err.mul acc n with
      | Some p -> p
      | None -> fail_at place "%s overflows" what)
    1 dims

let max_cell_rank = 31

let ascii_tform s =
  let s = String.trim s in
  let n = String.length s in
  if n < 2 then None
  else
    let code = s.[0] in
    let rest = String.sub s 1 (n - 1) in
    let w, d =
      match String.index_opt rest '.' with
      | None -> (int_of_string_opt rest, Some 0)
      | Some i ->
          ( int_of_string_opt (String.sub rest 0 i),
            int_of_string_opt
              (String.sub rest (i + 1) (String.length rest - i - 1)) )
    in
    match (code, w, d) with
    | ('A' | 'I' | 'F' | 'E' | 'D'), Some w, Some d when w > 0 && d >= 0 ->
        Some (code, w, d)
    | _ -> None

let describe hdu =
  let h = Hdu.header hdu in
  let place = Hdu.place hdu in
  let store = Hdu.store hdu in
  let xt = Hdu.get_struct h Value.string "XTENSION" in
  let ascii = xt = "TABLE" in
  if not (ascii || xt = "BINTABLE") then
    fail_at place "a %s extension is no table" xt;
  let num h k ~default =
    match Header.find_struct Value.text k h with
    | Error e -> fail "%s" e
    | Ok None -> (default, float_of_string default)
    | Ok (Some t) -> (
        match Value.read Value.Float t None with
        | Ok x -> (t, x)
        | Error e -> Hdu.card_fail h k e)
  in
  let name i =
    match Header.find_struct Value.string (strf "TTYPE%d" i) h with
    | Ok (Some s) -> s
    | Ok None -> ""
    | Error e -> fail "%s" e
  in
  let col_place i n =
    Err.sub place
      (if n = "" then strf "column %d" i else strf "column %d (%s)" i n)
  in
  let unit_text i =
    match Header.find_struct Value.string (strf "TUNIT%d" i) h with
    | Ok s -> s
    | Error e -> fail "%s" e
  in
  if ascii then begin
    let row_bytes = Hdu.get_struct h Value.int "NAXIS1"
    and rows = Hdu.get_struct h Value.int "NAXIS2" in
    if row_bytes < 0 || rows < 0 then
      fail_at place "NAXIS1 or NAXIS2 is negative";
    (match Err.mul row_bytes rows with
    | Some b when b <= store.size -> ()
    | _ -> fail_at place "the rows pass the data unit");
    let n = Hdu.get_struct h Value.int "TFIELDS" in
    if n < 0 || n > 999 then fail_at place "TFIELDS %d is outside 0-999" n;
    let cards = column_cards h n in
    let cols =
      Array.init n (fun i ->
          let i = i + 1 in
          let k = strf "TFORM%d" i in
          let f = Hdu.get_struct h Value.string k in
          let code, width, decimals =
            match ascii_tform f with
            | Some x -> x
            | None ->
                Hdu.card_fail h k (strf "%S is not an ASCII table format" f)
          in
          let bk = strf "TBCOL%d" i in
          let bcol = Hdu.get_struct h Value.int bk in
          if bcol < 1 || bcol - 1 > row_bytes - width then
            Hdu.card_fail h bk
              (strf "the field of %d bytes at byte %d passes NAXIS1 %d" width
                 bcol row_bytes);
          let nm = name i in
          let element : S.t =
            match code with
            | 'A' -> UInt8
            | 'I' ->
                if width <= 2 then Int8
                else if width <= 4 then Int16
                else if width <= 9 then Int32
                else Int64
            | _ -> Float64
          in
          let tscal_t, tscal = num h (strf "TSCAL%d" i) ~default:"1" in
          let tzero_t, tzero = num h (strf "TZERO%d" i) ~default:"0" in
          let scaled =
            code <> 'A'
            && not (Decimal.equal tscal_t "1" && Decimal.equal tzero_t "0")
          in
          let tnull =
            match Header.find_struct Value.string (strf "TNULL%d" i) h with
            | Ok (Some s) -> Text_null s
            | Ok None -> No_null
            | Error e -> fail "%s" e
          in
          {
            column =
              {
                name = nm;
                element;
                layout = (if code = 'A' then Text [||] else Array [||]);
                scaled;
                cards = cards.(i);
              };
            number = i;
            field = Ascii { code; width; decimals; start = bcol - 1 };
            tscal;
            tzero;
            tnull;
            string_width = width;
            unit_text = unit_text i;
            place = col_place i nm;
          })
    in
    { place; rows; row_bytes; cols; heap = None; store; ascii = true }
  end
  else begin
    let bt = Bintable.layout h store in
    let n = Array.length bt.columns in
    let cards = column_cards h n in
    let cols =
      Array.map
        (fun (bc : Bintable.column) ->
          let i = bc.number in
          let cplace = col_place i bc.name in
          let form = bc.form in
          let dims =
            if form.heap = Row then tdim h (strf "TDIM%d" i) else None
          in
          let tscal_t, tscal = num h (strf "TSCAL%d" i) ~default:"1" in
          let tzero_t, tzero = num h (strf "TZERO%d" i) ~default:"0" in
          let one = Decimal.equal tscal_t "1"
          and zero = Decimal.equal tzero_t "0" in
          let element, scaled =
            if (not (numeric form.elt)) || (one && zero) then
              (elt_element form.elt, false)
            else
              match offset_of form.elt with
              | Some (off, e) when one && Decimal.equal tzero_t off -> (e, false)
              | _ -> (elt_element form.elt, true)
          in
          let string_width, layout, count =
            match (form.heap, form.elt) with
            | (P | Q), A -> (0, Text [||], 0)
            | (P | Q), _ -> (0, Lists, 0)
            | Row, A -> (
                match dims with
                | None -> (form.repeat, Text [||], form.repeat)
                | Some d when Array.length d = 0 ->
                    (form.repeat, Text [||], form.repeat)
                | Some d ->
                    let total = product cplace "TDIM's product" d in
                    if total > form.repeat then
                      Hdu.card_fail h (strf "TDIM%d" i)
                        (strf "its product %d passes the repeat %d" total
                           form.repeat);
                    let grid = Array.sub d 1 (Array.length d - 1) in
                    ( d.(0),
                      Text (Array.of_list (List.rev (Array.to_list grid))),
                      total ))
            | Row, _ -> (
                match dims with
                | None ->
                    ( 0,
                      Array
                        (if form.repeat = 1 then [||] else [| form.repeat |]),
                      form.repeat )
                | Some d ->
                    let total = product cplace "TDIM's product" d in
                    if total > form.repeat then
                      Hdu.card_fail h (strf "TDIM%d" i)
                        (strf "its product %d passes the repeat %d" total
                           form.repeat);
                    if Array.length d > max_cell_rank then
                      fail_at cplace "a cell of %d axes, past the 31 nx holds"
                        (Array.length d);
                    ( 0,
                      Array (Array.of_list (List.rev (Array.to_list d))),
                      total ))
          in
          let tnull =
            match form.elt with
            | B | I | J | K -> (
                let k = strf "TNULL%d" i in
                match Header.find_struct Value.text k h with
                | Error e -> fail "%s" e
                | Ok None -> No_null
                | Ok (Some t) -> (
                    let lo, hi =
                      match form.elt with
                      | B -> (0L, 255L)
                      | I -> (-32768L, 32767L)
                      | J -> (-2147483648L, 2147483647L)
                      | _ -> (Int64.min_int, Int64.max_int)
                    in
                    let t' =
                      if String.length t > 0 && t.[0] = '+' then
                        String.sub t 1 (String.length t - 1)
                      else t
                    in
                    match Int64.of_string_opt t' with
                    | Some v
                      when String.for_all
                             (fun c -> Structure.is_digit c || c = '-')
                             t' ->
                        if Int64.compare v lo < 0 || Int64.compare v hi > 0 then
                          Hdu.card_fail h k
                            (strf
                               "%Ld is outside the column's stored range %Ld \
                                to %Ld"
                               v lo hi);
                        Int_null v
                    | _ -> Hdu.card_fail h k (strf "%s is not an integer" t)))
            | _ -> No_null
          in
          let width =
            match form.elt with
            | X -> (count + 7) / 8
            | e -> count * Bintable.size e
          in
          {
            column =
              { name = bc.name; element; layout; scaled; cards = cards.(i) };
            number = i;
            field = Bin { form; start = bc.start; count; width };
            tscal;
            tzero;
            tnull;
            string_width;
            unit_text = unit_text i;
            place = cplace;
          })
        bt.columns
    in
    {
      place;
      rows = bt.rows;
      row_bytes = bt.row_bytes;
      cols;
      heap = Some bt;
      store;
      ascii = false;
    }
  end

let description : (t, string) result Type.Id.t = Type.Id.make ()

let of_hdu hdu =
  Hdu.derive description (fun h -> catch (fun () -> describe h)) hdu

let pp_layout ppf (c : col) =
  let shape s =
    String.concat "; " (Array.to_list (Array.map string_of_int s))
  in
  match c.column.layout with
  | Text [||] ->
      if c.string_width > 0 then Format.fprintf ppf "text [%d]" c.string_width
      else Format.fprintf ppf "text"
  | Text g -> Format.fprintf ppf "text [%d] [%s]" c.string_width (shape g)
  | Lists -> Format.fprintf ppf "%s list" (S.to_string c.column.element)
  | Array [||] -> Format.pp_print_string ppf (S.to_string c.column.element)
  | Array s ->
      Format.fprintf ppf "%s [%s]" (S.to_string c.column.element) (shape s)

let pp ppf t =
  let heap = match t.heap with Some b -> b.heap_size | None -> 0 in
  Format.fprintf ppf "@[<v>%s %d rows, %d columns, heap %d bytes"
    (if t.ascii then "TABLE" else "BINTABLE")
    t.rows (Array.length t.cols) heap;
  Array.iter
    (fun c ->
      let kind = Format.asprintf "%a" pp_layout c in
      let kind = if c.column.scaled then kind ^ " scaled" else kind in
      Format.fprintf ppf "@,%4d %-20s %-12s %-7s" c.number c.column.name kind
        (Option.value ~default:"" c.unit_text);
      match c.tnull with
      | Int_null v -> Format.fprintf ppf " TNULL %Ld" v
      | Text_null s -> Format.fprintf ppf " TNULL %S" s
      | No_null -> ())
    t.cols;
  Format.fprintf ppf "@]"

(* Lookup and rows *)

let find t name =
  match List.filter (fun c -> c.column.name = name) (Array.to_list t.cols) with
  | [ c ] -> c
  | l ->
      let names =
        String.concat ", "
          (List.map
             (fun c ->
               if c.column.name = "" then strf "%d unnamed" c.number
               else c.column.name)
             (Array.to_list t.cols))
      in
      if l = [] then
        fail_at t.place "no column is named %s; the columns are %s" name names
      else
        fail_at t.place
          "%d columns are named %s; read them with Table.read; the columns are \
           %s"
          (List.length l) name names

let row_range t rows =
  match rows with
  | None -> (0, t.rows)
  | Some (a, b) ->
      if a < 0 || a > b then
        invalid_arg
          (strf "Fits.Table: the rows (%d, %d) are not 0 <= start <= stop" a b);
      if b > t.rows then
        fail_at t.place "the rows (%d, %d) pass the table's %d" a b t.rows;
      (a, b)

let chunk_bytes = 1 lsl 22

(* [cells t cols (a, b)] copies, in one pass over rows [a, b), each column's
   field bytes, row after row: an ASCII field's text, a binary field's
   elements, or a heap field's descriptor. *)
let cells t cols (a, b) =
  let n = b - a in
  let width c =
    match c.field with
    | Bin { form = { heap = P; _ }; _ } -> 8
    | Bin { form = { heap = Q; _ }; _ } -> 16
    | Bin { width; _ } -> width
    | Ascii { width; _ } -> width
  in
  let start c =
    match c.field with Bin { start; _ } | Ascii { start; _ } -> start
  in
  let outs =
    List.map
      (fun c -> A.create Bigarray.int8_unsigned Bigarray.c_layout (n * width c))
      cols
  in
  if n > 0 && t.row_bytes > 0 then begin
    let step = Int.max 1 (chunk_bytes / t.row_bytes) in
    let host = Hdu.host_bytes (Int.min n step * t.row_bytes) in
    let rows = Hdu.bigbytes host in
    let r = ref a in
    while !r < b do
      let m = Int.min step (b - !r) in
      Hdu.copy_in t.store.buffer
        ~offset:(t.store.offset + (!r * t.row_bytes))
        host ~at:0 (m * t.row_bytes);
      List.iter2
        (fun c out ->
          let w = width c and s = start c in
          if w > 0 then
            for k = 0 to m - 1 do
              A.blit
                (A.sub rows ((k * t.row_bytes) + s) w)
                (A.sub out ((!r - a + k) * w) w)
            done)
        cols outs;
      r := !r + m
    done
  end;
  outs

(* Decoding cells *)

let host_of (a : bigbytes) =
  let b = Hdu.host_bytes (A.dim a) in
  A.blit a (Hdu.bigbytes b);
  b

(* The stored elements of a binary column's [n] cells: a tensor of shape
   [n] @ cell shape in the stored format (bytes for L, bools for X). *)
let stored c (cells : bigbytes) n shape : Nx.packed =
  match c.field with
  | Ascii _ -> assert false
  | Bin { form; count; _ } -> (
      let full = Array.append [| n |] shape in
      let be dtype = Image.of_big_endian dtype full (host_of cells) in
      match form.elt with
      | B | L ->
          Nx.P
            (Nx.reshape full
               (Nx.of_buffer Nx.uint8 [| A.dim cells |] (host_of cells)))
      | I -> Nx.P (be Nx.int16)
      | J -> Nx.P (be Nx.int32)
      | K -> Nx.P (be Nx.int64)
      | E -> Nx.P (be Nx.float32)
      | D -> Nx.P (be Nx.float64)
      | C ->
          let f =
            Image.of_big_endian Nx.float32
              (Array.append full [| 2 |])
              (host_of cells)
          in
          Nx.P (Nx.bitcast Nx.complex64 f)
      | M ->
          let f =
            Image.of_big_endian Nx.float64
              (Array.append full [| 2 |])
              (host_of cells)
          in
          Nx.P (Nx.bitcast Nx.complex128 f)
      | X ->
          let w = (count + 7) / 8 in
          let b = B.create Nx_device.host Nx_dtype.Scalar.Bool (n * count) in
          let o = B.bigarray Bigarray.int8_unsigned b in
          for r = 0 to n - 1 do
            for k = 0 to count - 1 do
              let byte = A.get cells ((r * w) + (k / 8)) in
              A.set o ((r * count) + k) ((byte lsr (7 - (k mod 8))) land 1)
            done
          done;
          Nx.P (Nx.reshape full (Nx.of_buffer Nx.bool [| n * count |] b))
      | A -> assert false)

let t_byte = Char.code 'T'
let f_byte = Char.code 'F'

(* The stored numbers as the element: offsets added, L bytes read as T. *)
let element c (s : Nx.packed) : Nx.packed =
  match (c.field, s) with
  | Bin { form = { elt = L; _ }; _ }, Nx.P s -> (
      match Nx.dtype s with UInt8 -> Nx.P (Nx.equal_s s t_byte) | _ -> Nx.P s)
  | Bin { form = { elt = X; _ }; _ }, Nx.P s -> Nx.P (Nx.cast Nx.bit s)
  | _ -> Image.with_offset c.column.element s

(* Where the stored elements are undefined: TNULL, NaN, a logical byte other
   than T or F. *)
let undefined c (Nx.P s) : Nx.bool_t option =
  let tnull (type a b) (s : (a, b) Nx.t) v : Nx.bool_t option =
    match Nx.dtype s with
    | UInt8 -> Some (Nx.equal_s s (Int64.to_int v))
    | Int16 -> Some (Nx.equal_s s (Int64.to_int v))
    | Int32 -> Some (Nx.equal_s s (Int64.to_int32 v))
    | Int64 -> Some (Nx.equal_s s v)
    | _ -> None
  in
  match c.field with
  | Bin { form = { elt = L; _ }; _ } -> (
      match Nx.dtype s with
      | UInt8 ->
          Some
            (Nx.logical_and (Nx.not_equal_s s t_byte) (Nx.not_equal_s s f_byte))
      | _ -> None)
  | Bin { form = { elt = E | D; _ }; _ } -> Some (Nx.isnan s)
  | Bin { form = { elt = C; _ }; _ } -> (
      match Nx.dtype s with
      | Complex64 ->
          Some
            (Nx.logical_or
               (Nx.isnan (Nx.real Nx.float32 s))
               (Nx.isnan (Nx.imag Nx.float32 s)))
      | _ -> None)
  | Bin { form = { elt = M; _ }; _ } -> (
      match Nx.dtype s with
      | Complex128 ->
          Some
            (Nx.logical_or
               (Nx.isnan (Nx.real Nx.float64 s))
               (Nx.isnan (Nx.imag Nx.float64 s)))
      | _ -> None)
  | _ -> ( match c.tnull with Int_null v -> tnull s v | _ -> None)

(* ASCII fields *)

let is_blank (cells : bigbytes) off w =
  let r = ref true in
  for i = 0 to w - 1 do
    if A.get cells (off + i) <> 32 then r := false
  done;
  !r

let field_string (cells : bigbytes) off w =
  String.init w (fun i -> Char.chr (A.get cells (off + i)))

(* §7.2.5: a real with an implied decimal point [d] places from the right
   when it has none, an exponent written E, D or as a bare sign. *)
let ascii_real s d =
  let s = String.trim s in
  let n = String.length s in
  (* split mantissa and exponent *)
  let rec find_exp i =
    if i >= n then None
    else
      match s.[i] with
      | 'E' | 'D' | 'e' | 'd' -> Some (i, i + 1)
      | ('+' | '-') when i > 0 -> Some (i, i)
      | _ -> find_exp (i + 1)
  in
  let mant, exp =
    match find_exp 0 with
    | Some (i, j) -> (String.sub s 0 i, String.sub s j (n - j))
    | None -> (s, "")
  in
  let mant =
    if String.contains mant '.' || d = 0 then mant
    else
      let sign, digits =
        if mant <> "" && (mant.[0] = '+' || mant.[0] = '-') then
          (String.make 1 mant.[0], String.sub mant 1 (String.length mant - 1))
        else ("", mant)
      in
      let digits =
        if String.length digits <= d then
          String.make (d - String.length digits + 1) '0' ^ digits
        else digits
      in
      let k = String.length digits - d in
      sign ^ String.sub digits 0 k ^ "." ^ String.sub digits k d
  in
  let text = if exp = "" then mant else mant ^ "E" ^ exp in
  if not (Value.real_grammar text) then None
  else
    match
      float_of_string_opt
        (String.map (function 'D' | 'd' -> 'E' | c -> c) text)
    with
    | Some x -> Some x
    | None -> None

(* An ASCII column's values in its element (int64 or float64 holding
   them), with undefined fields. *)
let ascii_values c (cells : bigbytes) n a : Nx.packed * Nx.bool_t =
  match c.field with
  | Bin _ -> assert false
  | Ascii { code; width = w; decimals; _ } ->
      let undef = Array.make n false in
      let null_text = match c.tnull with Text_null s -> Some s | _ -> None in
      let is_null off =
        match null_text with
        | Some s ->
            let padded =
              if String.length s >= w then String.sub s 0 w
              else s ^ String.make (w - String.length s) ' '
            in
            field_string cells off w = padded
        | None -> false
      in
      let row_place r = Err.sub c.place (strf "row %d" (a + r)) in
      if code = 'I' then begin
        let out = Array.make n 0L in
        for r = 0 to n - 1 do
          let off = r * w in
          if is_null off then undef.(r) <- true
          else if is_blank cells off w then ()
          else
            let s = String.trim (field_string cells off w) in
            let s' =
              if String.length s > 0 && s.[0] = '+' then
                String.sub s 1 (String.length s - 1)
              else s
            in
            let ok =
              s' <> ""
              && String.for_all (fun ch -> Structure.is_digit ch || ch = '-') s'
              && String.rindex_opt s' '-'
                 |> Option.fold ~none:true ~some:(fun i -> i = 0)
            in
            if not ok then fail_at (row_place r) "%S is not an integer" s;
            match Int64.of_string_opt s' with
            | Some v -> out.(r) <- v
            | None -> fail_at (row_place r) "%s is past int64" s
        done;
        (Nx.P (Nx.create Nx.int64 [| n |] out), Nx.create Nx.bool [| n |] undef)
      end
      else begin
        let out = Array.make n 0. in
        for r = 0 to n - 1 do
          let off = r * w in
          if is_null off then undef.(r) <- true
          else if is_blank cells off w then ()
          else
            let s = field_string cells off w in
            match ascii_real s decimals with
            | Some x -> out.(r) <- x
            | None ->
                fail_at (row_place r) "%S is not a FITS %c field"
                  (String.trim s) code
        done;
        ( Nx.P (Nx.create Nx.float64 [| n |] out),
          Nx.create Nx.bool [| n |] undef )
      end

(* Text *)

(* A cell's text: its bytes up to the first NUL, trailing spaces dropped. *)
let cell_text (a : bigbytes) off w =
  let stop = ref off in
  while !stop < off + w && A.get a !stop <> 0 do
    incr stop
  done;
  while !stop > off && A.get a (!stop - 1) = 32 do
    decr stop
  done;
  !stop - off

let ragged_of_texts (src : bigbytes) (spans : (int * int) list) =
  let total = List.fold_left (fun acc (_, l) -> acc + l) 0 spans in
  let values = Bytes.create total in
  let offsets = Array.make (List.length spans + 1) 0L in
  let pos = ref 0 in
  List.iteri
    (fun i (off, len) ->
      for k = 0 to len - 1 do
        Bytes.unsafe_set values (!pos + k)
          (Char.unsafe_chr (A.get src (off + k)))
      done;
      pos := !pos + len;
      offsets.(i + 1) <- Int64.of_int !pos)
    spans;
  let v =
    Nx.init Nx.uint8 [| total |] (fun i -> Char.code (Bytes.get values i.(0)))
  in
  Nx_ragged.v ~offsets:(Nx.create Nx.int64 [| Array.length offsets |] offsets) v

(* Heaps *)

(* The heap spans of a heap column's [n] descriptors, checked, whose stored
   bytes together fit the heap. *)
let descriptors ~first_row t c (cells : bigbytes) n =
  let bt = match t.heap with Some b -> b | None -> assert false in
  let bc =
    match c.field with
    | Bin { form; _ } ->
        { Bintable.number = c.number; name = c.column.name; form; start = 0 }
    | Ascii _ -> assert false
  in
  let rb =
    {
      bt with
      Bintable.row_bytes = (match bc.form.heap with Q -> 16 | _ -> 8);
    }
  in
  let budget = ref bt.heap_size in
  Array.init n (fun r ->
      let count, off = Bintable.descriptor ~row:(first_row + r) rb bc cells r in
      let bytes =
        match bc.form.elt with
        | X -> (count + 7) / 8
        | e -> count * Bintable.size e
      in
      if bytes > !budget then
        fail_at c.place "the descriptors name more bytes than the heap holds";
      budget := !budget - bytes;
      (count, off, bytes))

let heap_bytes t spans =
  let bt = match t.heap with Some b -> b | None -> assert false in
  let total = Array.fold_left (fun acc (_, _, b) -> acc + b) 0 spans in
  let out = A.create Bigarray.int8_unsigned Bigarray.c_layout total in
  let pos = ref 0 in
  Array.iter
    (fun (_, off, bytes) ->
      if bytes > 0 then
        A.blit (Bintable.heap bt off bytes) (A.sub out !pos bytes);
      pos := !pos + bytes)
    spans;
  out

(* A heap column's stored elements, flat, with the row offsets. *)
let heap_stored ~first_row t c cells n : Nx.packed * Nx.int64_t =
  let spans = descriptors ~first_row t c cells n in
  let total = Array.fold_left (fun acc (k, _, _) -> acc + k) 0 spans in
  let offsets = Array.make (n + 1) 0L in
  Array.iteri
    (fun i (k, _, _) ->
      offsets.(i + 1) <- Int64.add offsets.(i) (Int64.of_int k))
    spans;
  let form =
    match c.field with Bin { form; _ } -> form | Ascii _ -> assert false
  in
  let flat =
    {
      c with
      field =
        Bin
          {
            form = { form with heap = Row };
            start = 0;
            count = total;
            width = 0;
          };
    }
  in
  let bytes =
    match form.elt with
    | X ->
        (* each row's bits start on a byte: repack them contiguously *)
        let raw = heap_bytes t spans in
        let out =
          A.create Bigarray.int8_unsigned Bigarray.c_layout ((total + 7) / 8)
        in
        A.fill out 0;
        let k = ref 0 and pos = ref 0 in
        Array.iter
          (fun (cnt, _, bytes) ->
            for j = 0 to cnt - 1 do
              let bitv =
                (A.get raw (!pos + (j / 8)) lsr (7 - (j mod 8))) land 1
              in
              if bitv = 1 then
                A.set out (!k / 8)
                  (A.get out (!k / 8) lor (1 lsl (7 - (!k mod 8))));
              incr k
            done;
            pos := !pos + bytes)
          spans;
        out
    | _ -> heap_bytes t spans
  in
  let s =
    stored
      {
        flat with
        field =
          Bin
            {
              form = { form with heap = Row };
              start = 0;
              count = total;
              width = A.dim bytes;
            };
      }
      bytes 1 [| total |]
  in
  let (Nx.P s) = s in
  (Nx.P (Nx.reshape [| total |] s), Nx.create Nx.int64 [| n + 1 |] offsets)

let text_spans ~first_row t c (cells : bigbytes) n : bigbytes * (int * int) list
    =
  match (c.field, c.column.layout) with
  | Bin { form = { heap = P | Q; _ }; _ }, _ ->
      let spans = descriptors ~first_row t c cells n in
      let raw = heap_bytes t spans in
      let pos = ref 0 in
      let l =
        Array.to_list
          (Array.map
             (fun (_, _, bytes) ->
               let off = !pos in
               pos := !pos + bytes;
               (off, cell_text raw off bytes))
             spans)
      in
      (raw, l)
  | Bin { width; _ }, Text grid ->
      let per = Array.fold_left ( * ) 1 grid in
      let w = c.string_width in
      let l = ref [] in
      for r = 0 to n - 1 do
        for k = 0 to per - 1 do
          let off = (r * width) + (k * w) in
          l := (off, cell_text cells off w) :: !l
        done
      done;
      (cells, List.rev !l)
  | Ascii { width; _ }, _ ->
      ( cells,
        List.init n (fun r -> (r * width, cell_text cells (r * width) width)) )
  | Bin _, _ -> assert false

(* Reads *)

let scale c (Nx.P s) =
  let x = Nx.cast Nx.float64 s in
  Nx.fma (Nx.scalar Nx.float64 c.tscal) x (Nx.scalar Nx.float64 c.tzero)

let numeric_col c =
  match c.field with
  | Bin { form = { elt = B | I | J | K | E | D; _ }; _ } -> true
  | Ascii { code = 'I' | 'F' | 'E' | 'D'; _ } -> true
  | _ -> false

let array_shape c =
  match c.column.layout with Array s -> s | _ -> assert false

let not_array c =
  match c.column.layout with
  | Array _ -> ()
  | Lists -> fail_at c.place "a heap column; read it with Table.ragged"
  | Text _ -> fail_at c.place "a text column; read it with Table.ragged"

(* The element tensor and undefined mask of an Array column. *)
let array_read t c (a, b) =
  let n = b - a in
  match c.field with
  | Ascii _ ->
      let cells = List.hd (cells t [ c ] (a, b)) in
      let v, u = ascii_values c cells n a in
      (v, Some u)
  | Bin _ ->
      let cells = List.hd (cells t [ c ] (a, b)) in
      let s = stored c cells n (array_shape c) in
      (s, undefined c s)

let ascii_element c (Nx.P v) : Nx.packed =
  match c.column.element with
  | Int8 -> Nx.P (Nx.cast Nx.int8 v)
  | Int16 -> Nx.P (Nx.cast Nx.int16 v)
  | Int32 -> Nx.P (Nx.cast Nx.int32 v)
  | _ -> Nx.P v

let to_element c s =
  match c.field with Ascii _ -> ascii_element c s | Bin _ -> element c s

let raw (type a b) ?rows (dtype : (a, b) Nx.dtype) name hdu :
    ((a, b) Nx.t, string) result =
  catch (fun () ->
      let t = ok_or_fail (of_hdu hdu) in
      let c = find t name in
      not_array c;
      Image.check_holds c.place c.column.element (S.of_dtype dtype);
      let r = row_range t rows in
      let s, _ = array_read t c r in
      let (Nx.P e) = to_element c s in
      Nx.cast dtype e)

let physical (type b) c (dtype : (float, b) Nx.dtype) s u : (float, b) Nx.t =
  let v =
    if c.column.scaled then Nx.cast dtype (scale c s)
    else (
      Image.check_holds c.place c.column.element (S.of_dtype dtype);
      let (Nx.P e) = to_element c s in
      Nx.cast dtype e)
  in
  match u with
  | Some u -> Nx.where u (Nx.full dtype (Nx.shape v) Float.nan) v
  | None -> v

let values (type b) ?rows (dtype : (float, b) Nx.dtype) name hdu :
    ((float, b) Nx.t, string) result =
  catch (fun () ->
      let t = ok_or_fail (of_hdu hdu) in
      let c = find t name in
      not_array c;
      if not (numeric_col c) then
        fail_at c.place
          "a %s column has no physical values; read it with Table.raw"
          (S.to_string c.column.element);
      let r = row_range t rows in
      let s, u = array_read t c r in
      physical c dtype s u)

let validity_of u =
  match u with
  | None -> None
  | Some u ->
      if Nx.numel u = 0 || not (Nx.item [] (Nx.any u)) then None
      else Some (Nx.cast Nx.bit (Nx.logical_not u))

let validity ?rows name hdu =
  catch (fun () ->
      let t = ok_or_fail (of_hdu hdu) in
      let c = find t name in
      let r = row_range t rows in
      match c.column.layout with
      | Text _ -> None
      | Array _ -> validity_of (snd (array_read t c r))
      | Lists ->
          let cells = List.hd (cells t [ c ] r) in
          let s, _ = heap_stored ~first_row:(fst r) t c cells (snd r - fst r) in
          validity_of (undefined c s))

let ragged (type a b) ?rows (dtype : (a, b) Nx.dtype) name hdu :
    ((a, b) Nx_ragged.t, string) result =
  catch (fun () ->
      let t = ok_or_fail (of_hdu hdu) in
      let c = find t name in
      let r = row_range t rows in
      let n = snd r - fst r in
      let cells = List.hd (cells t [ c ] r) in
      match c.column.layout with
      | Array _ -> fail_at c.place "an array column; read it with Table.raw"
      | Text _ ->
          Image.check_holds c.place UInt8 (S.of_dtype dtype);
          let src, spans = text_spans ~first_row:(fst r) t c cells n in
          Nx_ragged.map (Nx.cast dtype) (ragged_of_texts src spans)
      | Lists ->
          Image.check_holds c.place c.column.element (S.of_dtype dtype);
          let s, offsets = heap_stored ~first_row:(fst r) t c cells n in
          let (Nx.P e) = element c s in
          Nx_ragged.v ~offsets (Nx.cast dtype e))

let unit ~scope name hdu =
  catch (fun () ->
      let t = ok_or_fail (of_hdu hdu) in
      let c = find t name in
      match c.unit_text with
      | None -> None
      | Some s -> (
          match Fits_unit.parse ~scope s with
          | Ok u -> Some u
          | Error e ->
              let h = Hdu.header hdu and k = strf "TUNIT%d" c.number in
              fail_at
                (Header.card_place h (List.hd (Header.cards h k)) k)
                "%s" e))

(* A column read whole: its layout's constructor names its data. *)
type data =
  | Array of { values : Nx.packed; validity : Nx.bit_t option }
  | Lists : {
      values : ('a, 'b) Nx_ragged.t;
      validity : Nx.bit_t option;
    }
      -> data
  | Text of (int, Nx.uint8_elt) Nx_ragged.t

let read ?rows hdu =
  catch (fun () ->
      let t = ok_or_fail (of_hdu hdu) in
      let r = row_range t rows in
      let n = snd r - fst r in
      let cols = Array.to_list t.cols in
      let all = cells t cols r in
      List.map2
        (fun c cells ->
          let data =
            match c.column.layout with
            | Text _ ->
                let src, spans = text_spans ~first_row:(fst r) t c cells n in
                Text (ragged_of_texts src spans)
            | Array shape -> (
                let s, u =
                  match c.field with
                  | Ascii _ ->
                      let v, u = ascii_values c cells n (fst r) in
                      (v, Some u)
                  | Bin _ ->
                      let s = stored c cells n shape in
                      (s, undefined c s)
                in
                let validity = validity_of u in
                if c.column.scaled then
                  Array { values = Nx.P (physical c Nx.float64 s u); validity }
                else
                  match to_element c s with
                  | Nx.P e -> Array { values = Nx.P e; validity })
            | Lists ->
                let s, offsets = heap_stored ~first_row:(fst r) t c cells n in
                let u = undefined c s in
                let validity = validity_of u in
                if c.column.scaled then
                  Lists
                    {
                      values = Nx_ragged.v ~offsets (physical c Nx.float64 s u);
                      validity;
                    }
                else
                  let (Nx.P e) = element c s in
                  Lists { values = Nx_ragged.v ~offsets e; validity }
          in
          (c.column.name, c.column.cards, data))
        cols all)

(* Writing *)

(* The TFORM code of a dtype, the TZERO that offsets it, and the cast that
   stores a dtype FITS has no column for. *)
let code_of (type a b) (dtype : (a, b) Nx.dtype) : Bintable.elt * string option
    =
  let no cast =
    invalid_arg
      (strf "Fits.Table.hdu: FITS has no column of %s; %s"
         (Nx_dtype.to_string dtype) cast)
  in
  match dtype with
  | Bool -> (L, None)
  | Bit -> (X, None)
  | UInt8 -> (B, None)
  | Int8 -> (B, Some "-128")
  | Int16 -> (I, None)
  | UInt16 -> (I, Some "32768")
  | Int32 -> (J, None)
  | UInt32 -> (J, Some "2147483648")
  | Int64 -> (K, None)
  | UInt64 -> (K, Some "9223372036854775808")
  | Float32 -> (E, None)
  | Float64 -> (D, None)
  | Complex64 -> (C, None)
  | Complex128 -> (M, None)
  | Int4 -> no "cast it to int8"
  | UInt4 -> no "cast it to uint8"
  | Float16 | BFloat16 | Float8_e4m3 | Float8_e5m2 -> no "cast it to float32"

let wider : Bintable.elt -> string = function
  | B -> "int16"
  | I -> "int32"
  | J -> "int64"
  | _ -> "a float"

let stored_min : Bintable.elt -> int64 = function
  | B -> 0L
  | I -> -32768L
  | J -> -2147483648L
  | _ -> Int64.min_int

(* The TNULL of an integer column whose cells [invalid] marks: the stored
   type's minimum unless a valid cell holds it, then the least stored value
   none holds. *)
(* The TNULL of an integer column whose valid cells hold the stored values
   [held] (those from the stored type's minimum on, which is where it lies):
   that minimum unless a valid cell holds it, then the least stored value
   none holds. *)
let choose_null cplace elt held =
  let max =
    match elt with
    | Bintable.B -> 255L
    | I -> 32767L
    | J -> 2147483647L
    | _ -> Int64.max_int
  in
  let rec gap x =
    if not (Hashtbl.mem held x) then x
    else if Int64.equal x max then
      fail_at cplace
        "its valid cells hold every stored value, leaving none for TNULL; \
         store it as %s"
        (wider elt)
    else gap (Int64.succ x)
  in
  gap (stored_min elt)

let be_of (type a b) (t : (a, b) Nx.t) : bigbytes =
  let b = Image.to_big_endian t in
  Hdu.bigbytes b

(* Rows of at most this many bytes make one slab of a write. *)
let slab_bytes = 1 lsl 22

let invalid_of validity n =
  match validity with
  | None -> Array.make n false
  | Some v ->
      Array.map not (Nx.to_array (Nx.reshape [| n |] (Nx.cast Nx.bool v)))

(* [encode_flat elt null t invalid] is the stored bytes of the flat [t],
   big-endian, its [invalid] elements written as the column's undefined
   value: [null] for integers, NaN for floats, a zero byte for logicals. *)
let encode_flat (type a b) (elt : Bintable.elt) null (t : (a, b) Nx.t)
    (invalid : bool array) : bigbytes =
  let n = Nx.numel t in
  let t = Nx.reshape [| n |] t in
  let any = Array.exists Fun.id invalid in
  let mask () = Nx.create Nx.bool [| n |] invalid in
  let nan_pairs f =
    if any then
      Nx.where
        (Nx.broadcast_to [| n; 2 |] (Nx.reshape [| n; 1 |] (mask ())))
        (Nx.full (Nx.dtype f) [| n; 2 |] Float.nan)
        f
    else f
  in
  match elt with
  | L ->
      let a = Nx.to_array (Nx.cast Nx.bool t) in
      let o = A.create Bigarray.int8_unsigned Bigarray.c_layout n in
      Array.iteri
        (fun i b ->
          A.set o i (if invalid.(i) then 0 else if b then t_byte else f_byte))
        a;
      o
  | X ->
      let a = Nx.to_array (Nx.cast Nx.bool t) in
      let o = A.create Bigarray.int8_unsigned Bigarray.c_layout n in
      Array.iteri (fun i b -> A.set o i (if b then 1 else 0)) a;
      o
  | E | D -> (
      match Nx.dtype t with
      | Float32 ->
          be_of
            (if any then
               Nx.where (mask ()) (Nx.full Nx.float32 [| n |] Float.nan) t
             else t)
      | Float64 ->
          be_of
            (if any then
               Nx.where (mask ()) (Nx.full Nx.float64 [| n |] Float.nan) t
             else t)
      | _ -> assert false)
  | C | M -> (
      match Nx.dtype t with
      | Complex64 -> be_of (nan_pairs (Nx.bitcast Nx.float32 t))
      | Complex128 -> be_of (nan_pairs (Nx.bitcast Nx.float64 t))
      | _ -> assert false)
  | B | I | J | K -> (
      let (Nx.P s) = Image.to_stored t in
      match null with
      | Some v when any -> (
          let m = mask () in
          match Nx.dtype s with
          | UInt8 ->
              be_of (Nx.where m (Nx.full Nx.uint8 [| n |] (Int64.to_int v)) s)
          | Int16 ->
              be_of (Nx.where m (Nx.full Nx.int16 [| n |] (Int64.to_int v)) s)
          | Int32 ->
              be_of (Nx.where m (Nx.full Nx.int32 [| n |] (Int64.to_int32 v)) s)
          | Int64 -> be_of (Nx.where m (Nx.full Nx.int64 [| n |] v) s)
          | _ -> be_of s)
      | _ -> be_of s)
  | A -> assert false

(* [scan_null cplace elt t validity] reads an integer column once, in slabs,
   for its TNULL: [None] unless a cell is undefined. Only stored values from
   the type's minimum to the minimum plus the valid cells' count can be the
   least one no valid cell holds, so only those are kept. *)
let scan_null (type a b) cplace elt (t : (a, b) Nx.t) validity =
  match (elt : Bintable.elt) with
  | B | I | J | K -> (
      match validity with
      | None -> None
      | Some v ->
          let n = Nx.numel t in
          let flat = Nx.reshape [| n |] t and vflat = Nx.reshape [| n |] v in
          let valid = ref 0 and any = ref false in
          let held = Hashtbl.create 64 in
          let step = Int.max 1 (slab_bytes / 8) in
          let lo = stored_min elt in
          let i = ref 0 in
          let pass f =
            i := 0;
            while !i < n do
              let j = Int.min n (!i + step) in
              f
                (Nx.slice [ Nx.R (!i, j) ] flat)
                (Nx.to_array
                   (Nx.cast Nx.bool (Nx.slice [ Nx.R (!i, j) ] vflat)));
              i := j
            done
          in
          pass (fun _ ok ->
              Array.iter (fun o -> if o then incr valid else any := true) ok);
          if not !any then None
          else begin
            let hi = Int64.add lo (Int64.of_int !valid) in
            pass (fun x ok ->
                let (Nx.P s) = Image.to_stored x in
                let st = Nx.to_array (Nx.cast Nx.int64 s) in
                Array.iteri
                  (fun k o ->
                    let y = st.(k) in
                    if o && Int64.compare y lo >= 0 && Int64.compare y hi <= 0
                    then Hashtbl.replace held y ())
                  ok);
            Some (choose_null cplace elt held)
          end)
  | X -> (
      match validity with
      | Some v when not (Nx.item [] (Nx.all (Nx.cast Nx.bool v))) ->
          invalid_arg
            "Fits.Table.hdu: a validity on bit data, which has no undefined \
             value"
      | _ -> None)
  | _ -> None

let pack_bits (bits : bigbytes) off count (dst : bigbytes) doff =
  for k = 0 to ((count + 7) / 8) - 1 do
    A.set dst (doff + k) 0
  done;
  for k = 0 to count - 1 do
    if A.get bits (off + k) = 1 then
      A.set dst
        (doff + (k / 8))
        (A.get dst (doff + (k / 8)) lor (1 lsl (7 - (k mod 8))))
  done

let tform_text repeat (elt : Bintable.elt) =
  strf "%d%c" repeat (Bintable.char_of_elt elt)

let common_keys ~zero ~null n =
  (match zero with
    | Some z ->
        [
          Structure.decimal (strf "TZERO%d" n) z;
          Structure.int (strf "TSCAL%d" n) 1;
        ]
    | None -> [])
  @
  match null with
  | Some v -> [ Structure.decimal (strf "TNULL%d" n) (Int64.to_string v) ]
  | None -> []

let be_put (dst : bigbytes) off v n =
  for i = 0 to n - 1 do
    A.set dst (off + i) ((v lsr (8 * (n - 1 - i))) land 0xFF)
  done

(* A column as the writer streams it: [width] bytes in each row, written
   for rows [a, b) by [cells], then [heap_size] bytes of the heap, written
   for rows [a, b) by [heap]. A heap column's descriptors depend on where its
   arrays start in the heap and on whether descriptors are Q, known once
   every column is scanned: [writer wide base] makes it. *)
type writer = {
  cells : int -> int -> bigbytes;
  heap : int -> int -> bigbytes;
  keys : int -> Structure.entry list;
}

type column_plan = {
  rows : int;
  width : bool -> int;  (** bytes per row, given Q descriptors *)
  heap_size : int;
  writer : wide:bool -> base:int -> writer;
}

let no_heap _ _ = A.create Bigarray.int8_unsigned Bigarray.c_layout 0

let array_plan cplace (Nx.P t) validity =
  let shape = Nx.shape t in
  if Array.length shape = 0 then
    invalid_arg "Fits.Table.hdu: a column of no rows axis";
  let rows = shape.(0) in
  let cell = Array.sub shape 1 (Array.length shape - 1) in
  let count = Array.fold_left ( * ) 1 cell in
  let elt, zero = code_of (Nx.dtype t) in
  let null = scan_null cplace elt t validity in
  let w =
    match elt with X -> (count + 7) / 8 | e -> count * Bintable.size e
  in
  let cells a b =
    let n = b - a in
    let slice = Nx.slice [ Nx.R (a, b) ] t in
    let invalid =
      invalid_of
        (Option.map (fun v -> Nx.slice [ Nx.R (a, b) ] v) validity)
        (n * count)
    in
    let bytes = encode_flat elt null slice invalid in
    match elt with
    | X ->
        let o = A.create Bigarray.int8_unsigned Bigarray.c_layout (n * w) in
        for r = 0 to n - 1 do
          pack_bits bytes (r * count) count o (r * w)
        done;
        o
    | _ -> bytes
  in
  let keys n =
    [ Structure.string (strf "TFORM%d" n) (tform_text count elt) ]
    @ (if Array.length cell >= 2 then
         [
           Structure.string (strf "TDIM%d" n)
             ("("
             ^ String.concat ","
                 (List.rev_map string_of_int (Array.to_list cell))
             ^ ")");
         ]
       else [])
    @ common_keys ~zero ~null n
  in
  {
    rows;
    width = (fun _ -> w);
    heap_size = 0;
    writer = (fun ~wide:_ ~base:_ -> { cells; heap = no_heap; keys });
  }

let lists_plan cplace (values : ('a, 'b) Nx_ragged.t) validity =
  let v = Nx_ragged.values values in
  if Nx.ndim v <> 1 then
    invalid_arg
      "Fits.Table.hdu: heap arrays hold scalars; its values are not 1-D";
  let offsets =
    Array.map Int64.to_int (Nx.to_array (Nx_ragged.offsets values))
  in
  let rows = Array.length offsets - 1 in
  let elt, zero = code_of (Nx.dtype v) in
  let null = scan_null cplace elt v validity in
  (* where each row's array starts in the column's heap *)
  let bytes_of len =
    match elt with X -> (len + 7) / 8 | e -> len * Bintable.size e
  in
  let starts = Array.make (rows + 1) 0 in
  let maxlen = ref 0 in
  for r = 0 to rows - 1 do
    let len = offsets.(r + 1) - offsets.(r) in
    maxlen := Int.max !maxlen len;
    starts.(r + 1) <- starts.(r) + bytes_of len
  done;
  let heap a b =
    let lo = offsets.(a) and hi = offsets.(b) in
    let slice = Nx.slice [ Nx.R (lo, hi) ] v in
    let invalid =
      invalid_of
        (Option.map (fun x -> Nx.slice [ Nx.R (lo, hi) ] x) validity)
        (hi - lo)
    in
    let bytes = encode_flat elt null slice invalid in
    match elt with
    | X ->
        let o =
          A.create Bigarray.int8_unsigned Bigarray.c_layout
            (starts.(b) - starts.(a))
        in
        for r = a to b - 1 do
          pack_bits bytes
            (offsets.(r) - lo)
            (offsets.(r + 1) - offsets.(r))
            o
            (starts.(r) - starts.(a))
        done;
        o
    | _ -> bytes
  in
  let writer ~wide ~base =
    let d = if wide then 8 else 4 in
    let cells a b =
      let o =
        A.create Bigarray.int8_unsigned Bigarray.c_layout ((b - a) * 2 * d)
      in
      for r = a to b - 1 do
        let at = (r - a) * 2 * d in
        be_put o at (offsets.(r + 1) - offsets.(r)) d;
        be_put o (at + d) (base + starts.(r)) d
      done;
      o
    in
    let keys n =
      Structure.string (strf "TFORM%d" n)
        (strf "1%s%c(%d)"
           (if wide then "Q" else "P")
           (Bintable.char_of_elt elt) !maxlen)
      :: common_keys ~zero ~null n
    in
    { cells; heap; keys }
  in
  {
    rows;
    width = (fun wide -> if wide then 16 else 8);
    heap_size = starts.(rows);
    writer;
  }

let text_plan cplace (r : (int, Nx.uint8_elt) Nx_ragged.t) =
  let v = Nx_ragged.values r in
  let offsets = Array.map Int64.to_int (Nx.to_array (Nx_ragged.offsets r)) in
  let rows = Array.length offsets - 1 in
  let text a b =
    let lo = offsets.(a) and hi = offsets.(b) in
    (lo, Nx.to_array (Nx.slice [ Nx.R (lo, hi) ] v))
  in
  (* one pass for the width and the bytes reading would not give back *)
  let w = ref 1 in
  let step = Int.max 1 (slab_bytes / 64) in
  let a = ref 0 in
  while !a < rows do
    let b = Int.min rows (!a + step) in
    let lo, bytes = text a.contents b in
    for i = !a to b - 1 do
      let s = offsets.(i) - lo and e = offsets.(i + 1) - lo in
      w := Int.max !w (e - s);
      for k = s to e - 1 do
        if bytes.(k) < 32 || bytes.(k) > 126 then
          fail_at
            (Err.sub cplace (strf "row %d" i))
            "byte %d is outside ASCII 32-126" bytes.(k)
      done;
      if e > s && bytes.(e - 1) = 32 then
        fail_at
          (Err.sub cplace (strf "row %d" i))
          "the text ends in a space, which reading drops"
    done;
    a := b
  done;
  let w = !w in
  let cells a b =
    let lo, bytes = text a b in
    let o = A.create Bigarray.int8_unsigned Bigarray.c_layout ((b - a) * w) in
    A.fill o 32;
    for i = a to b - 1 do
      let s = offsets.(i) - lo and e = offsets.(i + 1) - lo in
      for k = s to e - 1 do
        A.set o (((i - a) * w) + (k - s)) bytes.(k)
      done
    done;
    o
  in
  let keys n = [ Structure.string (strf "TFORM%d" n) (strf "%dA" w) ] in
  {
    rows;
    width = (fun _ -> w);
    heap_size = 0;
    writer = (fun ~wide:_ ~base:_ -> { cells; heap = no_heap; keys });
  }

(* [renumber key n] is the keyword of column [n] whose keyword without the
   number is [key]. *)
let renumber key n =
  let candidates =
    let l = String.length key in
    (key ^ string_of_int n)
    ::
    (if l >= 2 && key.[l - 1] >= 'A' && key.[l - 1] <= 'Z' then
       [
         String.sub key 0 (l - 1) ^ string_of_int n ^ String.make 1 key.[l - 1];
       ]
     else [])
  in
  List.find_opt
    (fun k -> String.length k <= 8 && numbered k = Some (key, n))
    candidates

let card_entries n cards =
  let records = Array.of_list (Header.records cards) in
  let out = ref [] in
  let i = ref 0 in
  while !i < Array.length records do
    let span = Header.span_of records !i in
    (match Header.keyword records.(!i) with
    | Some (k, _) when not (List.mem k structure_roots) -> (
        match renumber k n with
        | Some key ->
            let recs =
              rename records.(!i) key
              :: Array.to_list (Array.sub records (!i + 1) (span - 1))
            in
            out := Structure.verbatim key recs :: !out
        | None ->
            invalid_arg
              (strf "Fits.Table.hdu: %s is not a column keyword FITS numbers" k)
        )
    | _ -> ());
    i := !i + span
  done;
  List.rev !out

let hdu header columns =
  catch (fun () ->
      let ncols = List.length columns in
      let place i name =
        Err.sub Err.nowhere
          (if name = "" then strf "column %d" i
           else strf "column %d (%s)" i name)
      in
      (* one scan of each column: its TNULL, its text's width and checks,
         its heap arrays' places *)
      let plans =
        List.mapi
          (fun i (name, cards, data) ->
            let n = i + 1 in
            let cp = place n name in
            let plan =
              match data with
              | Array { values; validity } -> array_plan cp values validity
              | Text r -> text_plan cp r
              | Lists { values; validity } -> lists_plan cp values validity
            in
            (n, name, cards, plan))
          columns
      in
      let rows =
        match plans with
        | [] -> 0
        | (_, _, _, p) :: rest ->
            List.iter
              (fun (n, _, _, (p' : column_plan)) ->
                if p'.rows <> p.rows then
                  invalid_arg
                    (strf "Fits.Table.hdu: column %d has %d rows, column 1 %d" n
                       p'.rows p.rows))
              rest;
            p.rows
      in
      let heap_size =
        List.fold_left (fun acc (_, _, _, p) -> acc + p.heap_size) 0 plans
      in
      let wide = heap_size > 0x7FFFFFFF in
      let base = ref 0 in
      let writers =
        List.map
          (fun (n, name, cards, p) ->
            let w = p.writer ~wide ~base:!base in
            base := !base + p.heap_size;
            (n, name, cards, p, w))
          plans
      in
      let widths = List.map (fun (_, _, _, p, _) -> p.width wide) writers in
      let row_bytes = List.fold_left ( + ) 0 widths in
      let entries =
        List.concat_map
          (fun (n, name, cards, _, w) ->
            (if name = "" then []
             else [ Structure.string (strf "TTYPE%d" n) name ])
            @ w.keys n @ card_entries n cards)
          writers
      in
      let prefix =
        Structure.
          [
            string "XTENSION" "BINTABLE";
            int "BITPIX" 8;
            int "NAXIS" 2;
            int "NAXIS1" row_bytes;
            int "NAXIS2" rows;
            int "PCOUNT" heap_size;
            int "GCOUNT" 1;
            int "TFIELDS" ncols;
          ]
      in
      let h =
        Structure.apply ~owned:Structure.table_owned ~prefix ~others:entries
          header
      in
      (* the rows in slabs, each column's fields interleaved, then each heap
         column's arrays in slabs *)
      let step = Int.max 1 (slab_bytes / Int.max 1 row_bytes) in
      let stream (append : bigbytes -> unit) =
        let a = ref 0 in
        while !a < rows do
          let b = Int.min rows (!a + step) in
          let o =
            A.create Bigarray.int8_unsigned Bigarray.c_layout
              ((b - !a) * row_bytes)
          in
          let at = ref 0 in
          List.iter2
            (fun (_, _, _, _, w) width ->
              let cells = w.cells !a b in
              for r = 0 to b - !a - 1 do
                A.blit
                  (A.sub cells (r * width) width)
                  (A.sub o ((r * row_bytes) + !at) width)
              done;
              at := !at + width)
            writers widths;
          append o;
          a := b
        done;
        List.iter
          (fun (_, _, _, p, w) ->
            if p.heap_size > 0 then begin
              let a = ref 0 in
              while !a < rows do
                let b = Int.min rows (!a + step) in
                append (w.heap !a b);
                a := b
              done
            end)
          writers
      in
      let host (a : bigbytes) =
        let b = Hdu.host_bytes (A.dim a) in
        A.blit a (Hdu.bigbytes b);
        b
      in
      let data =
        Once.make (fun () ->
            let size = (rows * row_bytes) + heap_size in
            let all = A.create Bigarray.int8_unsigned Bigarray.c_layout size in
            let at = ref 0 in
            stream (fun a ->
                A.blit a (A.sub all !at (A.dim a));
                at := !at + A.dim a);
            Hdu.host_store (host all))
      in
      let write (sink : Hdu.sink) =
        stream (fun a -> sink.append (host a));
        h
      in
      Hdu.constructed
        ~stream:(Once.of_value { Hdu.provisional = h; write })
        h data)
