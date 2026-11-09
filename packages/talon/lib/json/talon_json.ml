(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon
module Reader = Bytesrw.Bytes.Reader
module A1 = Bigarray.Array1

type framing = Scan.framing = Lines | Array
type column = { name : string; ty : Type.any; declared : bool }

type format = {
  framing : framing;
  columns : column array;
  sniffed : int option; (* The rows sniff inferred the types from. *)
}

let invalid fn fmt =
  Printf.ksprintf (fun s -> invalid_arg ("Talon_json." ^ fn ^ ": " ^ s)) fmt

let type_name (Type.Any t) = Format.asprintf "%a" Type.pp t
let plural n word = Printf.sprintf "%d %s%s" n word (if n = 1 then "" else "s")
let quoted s = Format.asprintf "%a" Type.pp_quoted s

(* Formats *)

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

let rec reads (Type.Any t) =
  match t with
  | Bool | Int8 | Int16 | Int32 | Int64 | Uint8 | Uint16 | Uint32 | Uint64
  | Float16 | Float32 | Float64 | String | Binary | Categorical _ | Date
  | Datetime _ ->
      true
  | List e -> reads (Type.Any e)
  | Record fields -> List.for_all (fun (_, ty) -> reads ty) fields
  | Clock _ | Duration _ | Tensor _ | Ext _ -> false

let check_type fn name ty =
  if not (reads ty) then
    invalid fn "column %s: JSON does not read %s" (quoted name) (type_name ty)

let format ?(framing = Lines) columns =
  let fn = "format" in
  if columns = [] then invalid fn "a format has at least one column";
  check_names fn (List.map fst columns);
  List.iter (fun (name, ty) -> check_type fn name ty) columns;
  let columns =
    Array.of_list
      (List.map (fun (name, ty) -> { name; ty; declared = true }) columns)
  in
  { framing; columns; sniffed = None }

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
  Format.fprintf ppf "@[<v>json (%s, %s)" (plural n "column")
    (match f.framing with Lines -> "lines" | Array -> "array");
  Option.iter
    (fun rows ->
      Format.fprintf ppf ", types sniffed from %s"
        (match rows with 1 -> "1 row" | n -> thousands n ^ " rows"))
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

exception Encoding_failed of Error.t
exception Unwritable of string

let protect ?file f =
  try f () with
  | Scan.Error { line; column; byte; text; msg } ->
      Error (Error.v ?file ~line ~column ~bytes:(byte, byte) ?text msg)
  | Encoding_failed e -> Error e
  | Bytesrw.Bytes.Stream.Error e ->
      Error (Error.v ?file (Bytesrw.Bytes.Stream.error_message e))
  | Sys_error msg -> Error (Error.v msg)

(* Text columns *)

(* A binary column has no value to check, so [of_layout] cannot fail. Its
   validity is the bits of its rows in the bytes [validity]. *)
let binary ?validity offsets data =
  let length = Nx.numel offsets - 1 in
  let bits v =
    Nx.slice [ Nx.R (0, length) ] (Nx.reshape [| -1 |] (Nx.bitcast Nx.bit v))
  in
  let validity = Option.map bits validity in
  Result.get_ok
    (Column.of_layout (Type.Any Type.binary)
       (Varsize { validity; offsets; child = Column.of_tensor data }))

(* A column's arrays are host arrays ({!Nx_array.host}): rig's host memory,
   which paces the collector by the program's memory and is reused once
   collected. *)
let host dt n = Nx_array.host dt [| n |]
let write a f = Nx_array.write a (fun g -> f (Bigarray.array1_of_genarray g))
let value a = Nx.Repr.unsafe_of_array Nx.Host.v a

(* Reading

   A batch is read into one builder per column, by recursion over its type,
   whose depth the format bounds: a value nested deeper than its column's type
   is a value of another type, or the JSON text of a string column. Every scalar
   column builds its text: [Column.parse] then reads each batch as CSV's does,
   and a failure at a text maps back to its byte through [pos]. *)

type accepts = Literal | Integer | Number | Quoted | Any_value

type text = {
  accepts : accepts;
  mutable data : Bytes.t;
  mutable used : int;
  mutable ends : int array; (* Row [i]'s text ends at [ends.(i + 1)]. *)
  mutable pos : int array; (* Row [i]'s value starts at [pos.(i)]. *)
}

type col = {
  ty : Type.any;
  path : string;
  fix : string; (* What a value that fails to read should say to do. *)
  mutable n : int;
  mutable valid : Bytes.t; (* A bit per row, cleared at a null. *)
  mutable has_null : bool;
  shape : shape;
}

and shape =
  | Skip
  | Text of text
  | Seq of seq
  | Fields of fields

and seq = { child : col; mutable off : int array }

and fields = {
  names : string array;
  cols : col array;
  index : (string, int) Hashtbl.t;
  mutable guess : int; (* The column the next name likely is. *)
}

let accepts_of (Type.Any t) =
  match t with
  | Bool -> Literal
  | Int8 | Int16 | Int32 | Int64 | Uint8 | Uint16 | Uint32 | Uint64 -> Integer
  | Float16 | Float32 | Float64 -> Number
  | String | Binary -> Any_value
  | Categorical _ | Date | Datetime _ -> Quoted
  | List _ | Record _ | Clock _ | Duration _ | Tensor _ | Ext _ ->
      assert false (* [reads]. *)

let builder ~path ~fix ty shape =
  let valid = Bytes.make 16 '\xFF' in
  { ty; path; fix; n = 0; valid; has_null = false; shape }

let rec column_builder ~path ~fix ty =
  let (Type.Any t) = ty in
  match t with
  | List e ->
      let child = column_builder ~path:(path ^ "[]") ~fix (Type.Any e) in
      builder ~path ~fix ty (Seq { child; off = Array.make 16 0 })
  | Record fields ->
      let names = Array.of_list (List.map fst fields) in
      let cols =
        Array.of_list
          (List.map
             (fun (n, fty) -> column_builder ~path:(path ^ "." ^ n) ~fix fty)
             fields)
      in
      builder ~path ~fix ty (fields_shape names cols)
  | _ ->
      builder ~path ~fix ty
        (Text
           {
             accepts = accepts_of ty;
             data = Bytes.create 256;
             used = 0;
             ends = Array.make 16 0;
             pos = Array.make 16 0;
           })

and fields_shape names cols =
  let index = Hashtbl.create (Array.length names) in
  Array.iteri (fun i n -> Hashtbl.replace index n i) names;
  Fields { names; cols; index; guess = 0 }

(* [grow a n] is [a], longer if it has fewer than [n] elements. *)
let grow a n =
  if Array.length a >= n then a
  else begin
    let b = Array.make (max n (2 * Array.length a)) 0 in
    Array.blit a 0 b 0 (Array.length a);
    b
  end

let reserve l k =
  if l.used + k > Bytes.length l.data then begin
    let b = Bytes.create (max (l.used + k) (2 * Bytes.length l.data)) in
    Bytes.blit l.data 0 b 0 l.used;
    l.data <- b
  end

let push_valid c =
  while c.n / 8 >= Bytes.length c.valid do
    let n = Bytes.length c.valid in
    let v = Bytes.make (2 * n) '\xFF' in
    Bytes.blit c.valid 0 v 0 n;
    c.valid <- v
  done

let rec push_null c =
  push_valid c;
  c.has_null <- true;
  let byte = c.n / 8 in
  let bits = Char.code (Bytes.get c.valid byte) in
  Bytes.set c.valid byte (Char.chr (bits land lnot (1 lsl (c.n mod 8))));
  (match c.shape with
  | Skip -> ()
  | Fields f -> Array.iter push_null f.cols
  | Text l ->
      l.ends <- grow l.ends (c.n + 2);
      l.pos <- grow l.pos (c.n + 1);
      l.ends.(c.n + 1) <- l.used
  | Seq q ->
      q.off <- grow q.off (c.n + 2);
      q.off.(c.n + 1) <- q.child.n);
  c.n <- c.n + 1

(* [finish c l p] ends a value of [c] that starts at the input position [p]. *)
let finish c l p =
  push_valid c;
  l.ends <- grow l.ends (c.n + 2);
  l.pos <- grow l.pos (c.n + 1);
  l.pos.(c.n) <- p;
  l.ends.(c.n + 1) <- l.used;
  c.n <- c.n + 1

let name_is s name ~pos ~len =
  len = String.length name
  &&
  let b = Scan.bytes s in
  let rec same k =
    k = len
    || Bytes.unsafe_get b (pos + k) = String.unsafe_get name k && same (k + 1)
  in
  same 0

(* What a JSON value's first byte says it is. *)
let json_type = function
  | '"' -> "a string"
  | 't' | 'f' -> "a boolean"
  | '{' -> "an object"
  | '[' -> "an array"
  | _ -> "a number"

let is_value_start c =
  match c with
  | '"' | 't' | 'f' | '{' | '[' | '-' | '0' .. '9' -> true
  | _ -> false

let mismatch s c first what =
  if not (is_value_start first) then Scan.unexpected s "expected a value";
  Scan.fail s (Scan.pos s)
    (Printf.sprintf "column %s: %s does not read as %s, %s. %s" (quoted c.path)
       (json_type first) (type_name c.ty) what c.fix)

let add_string s l ~escaped =
  reserve l (Scan.text_len s);
  if escaped then l.used <- l.used + Scan.unescape s l.data l.used
  else begin
    Bytes.blit (Scan.bytes s) (Scan.text_pos s) l.data l.used (Scan.text_len s);
    l.used <- l.used + Scan.text_len s
  end

let add_raw s l ~from =
  let len = Scan.pos s - from in
  reserve l len;
  Bytes.blit (Scan.bytes s) from l.data l.used len;
  l.used <- l.used + len

let text_value s c l first =
  let p = Scan.pos s in
  (match (first, l.accepts) with
  | '"', (Quoted | Any_value) ->
      let escaped = Scan.string s in
      add_string s l ~escaped
  | ('t' | 'f'), Literal ->
      Scan.literal s (if first = 't' then "true" else "false");
      add_raw s l ~from:p
  | ('-' | '0' .. '9'), Integer ->
      if not (Scan.number s) then
        Scan.fail s p ~text:(Bytes.sub_string (Scan.bytes s) p (Scan.pos s - p))
          (Printf.sprintf
             "column %s: a number with a fraction or an exponent does not read \
              as %s. %s"
             (quoted c.path) (type_name c.ty) c.fix);
      add_raw s l ~from:p
  | ('-' | '0' .. '9'), Number ->
      ignore (Scan.number s);
      add_raw s l ~from:p
  | _, Any_value ->
      Scan.skip_value s;
      add_raw s l ~from:p
  | ( _,
      ( Literal | Integer | Number
      | Quoted ) ) ->
      let want =
        match l.accepts with
        | Literal -> "a boolean"
        | Integer | Number -> "a number"
        | Quoted | Any_value -> "a string"
      in
      mismatch s c first (Printf.sprintf "which holds %s" want));
  finish c l p

let rec read_value s c =
  let first = Scan.peek s in
  match c.shape with
  | _ when first = 'n' ->
      Scan.literal s "null";
      push_null c
  | Skip ->
      Scan.skip_value s;
      push_valid c;
      c.n <- c.n + 1
  | Text l -> text_value s c l first
  | Seq q -> seq s c q first
  | Fields f ->
      if first <> '{' then mismatch s c first "which holds an object";
      fields s c f

and seq s c q first =
  if first <> '[' then mismatch s c first "which holds an array";
  if Scan.enter s ']' then begin
    read_value s q.child;
    while Scan.continue_ s ']' do
      read_value s q.child
    done
  end;
  push_valid c;
  q.off <- grow q.off (c.n + 2);
  q.off.(c.n + 1) <- q.child.n;
  c.n <- c.n + 1

and fields s c f =
  let row = c.n in
  if Scan.enter s '}' then begin
    member s f ~row;
    while Scan.continue_ s '}' do
      member s f ~row
    done
  end;
  Array.iter (fun fc -> if fc.n = row then push_null fc) f.cols;
  push_valid c;
  c.n <- c.n + 1

and member s f ~row =
  let escaped = Scan.member s in
  let i =
    if
      (not escaped) && f.guess < Array.length f.names
      && name_is s f.names.(f.guess) ~pos:(Scan.text_pos s)
           ~len:(Scan.text_len s)
    then Some f.guess
    else Hashtbl.find_opt f.index (name_text s ~escaped)
  in
  match i with
  | None -> Scan.skip_value s
  | Some i ->
      let fc = f.cols.(i) in
      if fc.n > row then
        Scan.fail s (Scan.name_pos s)
          (Printf.sprintf "the object names %s twice" (quoted f.names.(i)));
      read_value s fc;
      f.guess <- i + 1

(* The text of the name [Scan.string] just read. *)
and name_text s ~escaped =
  if escaped then begin
    let b = Bytes.create (Scan.text_len s) in
    Bytes.sub_string b 0 (Scan.unescape s b 0)
  end
  else Bytes.sub_string (Scan.bytes s) (Scan.text_pos s) (Scan.text_len s)

(* Batches *)

let bits_of c ~rows =
  if not c.has_null then None
  else begin
    let a = host Nx.uint8 ((rows + 7) / 8) in
    write a (fun v ->
        for i = 0 to A1.dim v - 1 do
          A1.unsafe_set v i (Char.code (Bytes.get c.valid i))
        done;
        if rows mod 8 <> 0 then
          let last = rows / 8 in
          A1.unsafe_set v last
            (A1.unsafe_get v last land ((1 lsl (rows mod 8)) - 1)));
    Some (value a)
  end

let validity c ~rows =
  Option.map
    (fun v ->
      Nx.slice [ Nx.R (0, rows) ] (Nx.reshape [| -1 |] (Nx.bitcast Nx.bit v)))
    (bits_of c ~rows)

let reset_valid c =
  if c.has_null then begin
    Bytes.fill c.valid 0 (Bytes.length c.valid) '\xFF';
    c.has_null <- false
  end;
  c.n <- 0

(* A reading's failure is the earliest of its leaves' parse failures. *)
type earliest = {
  mutable failure : (int * string option * string) option;
}

let note e p ?text msg =
  match e.failure with
  | Some (q, _, _) when q <= p -> ()
  | _ -> e.failure <- Some (p, text, msg)

(* JSON's grammar has no text for an infinity, so one in a parsed column is a
   number beyond the type's range. Most batches hold none: the check on the
   column's values avoids boxing them, and only a hit reads its rows. *)
let first_infinity (Type.Any t) c =
  let find o =
    let rec go i =
      if i = Array.length o then None
      else
        match o.(i) with
        | Some x when Float.abs x = infinity -> Some i
        | _ -> go (i + 1)
    in
    go 0
  in
  let has_infinity () =
    match Column.layout c with
    | Fixed { values = Nx.P x; _ } -> (Nx.to_array (Nx.any (Nx.isinf x))).(0)
    | Varsize _ | Children _ -> false
  in
  match t with
  | Float16 when has_infinity () -> find (Column.options (Type.kind t) c)
  | Float32 when has_infinity () -> find (Column.options (Type.kind t) c)
  | Float64 when has_infinity () -> find (Column.options (Type.kind t) c)
  | _ -> None

let rec flush e c ~rows =
  match c.shape with
  | Skip -> None
  | Text l -> flush_text e c l ~rows
  | Seq q -> (
      let child_rows = if rows = 0 then 0 else q.off.(rows) in
      match flush e q.child ~rows:child_rows with
      | None -> None
      | Some child ->
          let offsets = host Nx.int64 (rows + 1) in
          write offsets (fun o ->
              A1.unsafe_set o 0 0L;
              for i = 1 to rows do
                A1.unsafe_set o i (Int64.of_int q.off.(i))
              done);
          Some
            (Result.get_ok
               (Column.of_layout c.ty
                  (Varsize
                     {
                       validity = validity c ~rows;
                       offsets = value offsets;
                       child;
                     }))))
  | Fields f ->
      let cols = Array.map (fun fc -> flush e fc ~rows) f.cols in
      if Array.exists Option.is_none cols then None
      else
        Some
          (Result.get_ok
             (Column.of_layout c.ty
                (Children
                   {
                     validity = validity c ~rows;
                     length = rows;
                     fields =
                       Array.to_list
                         (Array.mapi
                            (fun i col -> (f.names.(i), Option.get col))
                            cols);
                   })))

and flush_text e c l ~rows =
  let len = if rows = 0 then 0 else l.ends.(rows) in
  let data = host Nx.uint8 len and offsets = host Nx.int64 (rows + 1) in
  write data (fun d ->
      for i = 0 to len - 1 do
        A1.unsafe_set d i (Char.code (Bytes.unsafe_get l.data i))
      done);
  write offsets (fun o ->
      A1.unsafe_set o 0 0L;
      for i = 1 to rows do
        A1.unsafe_set o i (Int64.of_int l.ends.(i))
      done);
  let texts = binary ?validity:(bits_of c ~rows) (value offsets) (value data) in
  match Column.parse c.ty texts with
  | Error (row, reason) ->
      let text =
        Bytes.sub_string l.data l.ends.(row) (l.ends.(row + 1) - l.ends.(row))
      in
      note e l.pos.(row) ~text
        (Printf.sprintf "column %s: cannot read as %s: %s. %s" (quoted c.path)
           (type_name c.ty) reason c.fix);
      None
  | Ok col -> (
      match first_infinity c.ty col with
      | Some row ->
          note e l.pos.(row)
            (Printf.sprintf "column %s: the number exceeds %s. %s"
               (quoted c.path) (type_name c.ty) c.fix);
          None
      | None -> Some col)

let rec reset c =
  reset_valid c;
  match c.shape with
  | Skip -> ()
  | Text l -> l.used <- 0
  | Seq q -> reset q.child
  | Fields f -> Array.iter reset f.cols

(* A reading of a text: its scanner, one builder per format column (those not
   requested skip their values) and the rows it may still yield. *)
type reading = {
  f : format;
  s : Scan.t;
  top : col;
  read : int list; (* The requested columns, in the request's order. *)
  mutable left : int;
  mutable pending : exn option;
}

(* A failing value of a sniffed column says what the type came from. *)
let fix f c =
  match f.sniffed with
  | Some rows when not c.declared ->
      Printf.sprintf
        "The type was sniffed from %s. Declare another (with_type)."
        (if rows = 1 then "row 1" else "rows 1 to " ^ thousands rows)
  | _ -> "Declare another type (with_type)."

let reading f ~read ~limit r =
  let s = Scan.make f.framing r in
  let requested = Array.make (Array.length f.columns) false in
  List.iter (fun j -> requested.(j) <- true) read;
  let cols =
    Array.mapi
      (fun j c ->
        if requested.(j) then column_builder ~path:c.name ~fix:(fix f c) c.ty
        else builder ~path:c.name ~fix:"" c.ty Skip)
      f.columns
  in
  let names = Array.map (fun c -> c.name) f.columns in
  let top =
    builder ~path:"" ~fix:"" (Type.Any (Type.record []))
      (fields_shape names cols)
  in
  let left = Option.value ~default:max_int limit in
  { f; s; top; read; left; pending = None }

let check_row s =
  let first = Scan.peek s in
  if first = '\000' then Scan.unexpected s "expected a record";
  if first <> '{' then
    Scan.fail s (Scan.pos s)
      (Printf.sprintf "a row is a JSON object, and this is %s"
         (json_type first))

let row r =
  check_row r.s;
  match r.top.shape with
  | Fields f -> fields r.s r.top f
  | _ -> assert false

(* [next r] is the next batch of [r], which stops at its limit. A row that
   fails ends its batch: the rows before it are returned, and the next call
   raises, unless one of them fails first. *)
let rec next r =
  match r.pending with
  | Some e -> raise e
  | None ->
      if r.left = 0 || not (Scan.next r.s) then None
      else begin
        let rows = ref 0 in
        let pending =
          try
            while !rows < r.left && Scan.record r.s do
              row r;
              Scan.end_record r.s;
              incr rows
            done;
            None
          with Scan.Error _ as e -> Some e
        in
        r.pending <- pending;
        r.left <- r.left - !rows;
        match batch r ~rows:!rows with
        | Some t -> Some t
        | None -> next r
      end

and batch r ~rows =
  let e = { failure = None } in
  let f = match r.top.shape with Fields f -> f | _ -> assert false in
  let cols = List.map (fun j -> (j, flush e f.cols.(j) ~rows)) r.read in
  (match e.failure with
  | Some (p, text, msg) -> Scan.fail r.s p ?text msg
  | None -> ());
  reset r.top;
  if rows = 0 then None
  else
    Some
      (Talon.v ~rows
         (List.map (fun (j, c) -> (r.f.columns.(j).name, Option.get c)) cols))

let schema f =
  Schema.v (Array.to_list (Array.map (fun c -> (c.name, c.ty)) f.columns))

let decode f r =
  protect @@ fun () ->
  let read = List.init (Array.length f.columns) Fun.id in
  let reading = reading f ~read ~limit:None r in
  let rec batches acc =
    match next reading with
    | Some b -> batches (b :: acc)
    | None -> List.rev acc
  in
  match batches [] with
  | [] ->
      let e = { failure = None } in
      let empty c =
        let b = column_builder ~path:c.name ~fix:"" c.ty in
        (c.name, Option.get (flush e b ~rows:0))
      in
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

let source f open_ = source_of ~name:"json" f (fun () -> (open_ (), ignore))

(* Sniffing

   A column's type is the join of the types of its sampled values. A node holds
   what is known at one path. The nesting a type gets is bounded because every
   traversal of a type (this reader's builders, [Type.pp], the query layer)
   recurses on it, on the OCaml stack; a value past the bound reads whole as its
   JSON text, so the bound loses nothing. *)

let max_depth = 64

type guess =
  | Nothing
  | Boolean
  | Integral
  | Decimal
  | Textual
  | Listing of node
  | Object of obj

and node = { mutable g : guess }

and obj = {
  entries : (string, entry) Hashtbl.t;
  mutable order : (string * entry) list;
  mutable generation : int;
}

and entry = { node : node; mutable last : int }

let new_node () = { g = Nothing }

let set node g =
  node.g <-
    (match (node.g, g) with
    | Nothing, x -> x
    | Boolean, Boolean -> Boolean
    | Integral, Integral -> Integral
    | (Integral | Decimal), (Integral | Decimal) -> Decimal
    | _ -> Textual)

let observe_number s node =
  let p = Scan.pos s in
  let integer = Scan.number s in
  let text = Bytes.sub_string (Scan.bytes s) p (Scan.pos s - p) in
  if integer then
    set node
      (match Int64.of_string_opt text with
      | Some _ -> Integral
      | None -> Textual)
  else set node Decimal

let rec observe s node ~depth =
  match Scan.peek s with
  | 'n' -> Scan.literal s "null"
  | '"' ->
      ignore (Scan.string s);
      set node Textual
  | 't' ->
      Scan.literal s "true";
      set node Boolean
  | 'f' ->
      Scan.literal s "false";
      set node Boolean
  | '-' | '0' .. '9' -> observe_number s node
  | '{' when depth < max_depth -> observe_object s node ~depth
  | '[' when depth < max_depth -> observe_array s node ~depth
  | _ ->
      Scan.skip_value s;
      set node Textual

and observe_array s node ~depth =
  let elem =
    match node.g with
    | Listing e -> Some e
    | Nothing ->
        let e = new_node () in
        node.g <- Listing e;
        Some e
    | _ -> None
  in
  match elem with
  | None ->
      Scan.skip_value s;
      set node Textual
  | Some e ->
      if Scan.enter s ']' then begin
        observe s e ~depth:(depth + 1);
        while Scan.continue_ s ']' do
          observe s e ~depth:(depth + 1)
        done
      end

and observe_object s node ~depth =
  let o =
    match node.g with
    | Object o -> Some o
    | Nothing ->
        let o = { entries = Hashtbl.create 16; order = []; generation = 0 } in
        node.g <- Object o;
        Some o
    | _ -> None
  in
  match o with
  | None ->
      Scan.skip_value s;
      set node Textual
  | Some o ->
      o.generation <- o.generation + 1;
      if Scan.enter s '}' then begin
        observe_member s o ~depth;
        while Scan.continue_ s '}' do
          observe_member s o ~depth
        done
      end

and observe_member s o ~depth =
  let escaped = Scan.member s in
  let name = name_text s ~escaped in
  let e =
    match Hashtbl.find_opt o.entries name with
    | Some e -> e
    | None ->
        let e = { node = new_node (); last = 0 } in
        Hashtbl.add o.entries name e;
        o.order <- (name, e) :: o.order;
        e
  in
  if e.last = o.generation then
    Scan.fail s (Scan.name_pos s)
      (Printf.sprintf "the object names %s twice" (quoted name));
  e.last <- o.generation;
  observe s e.node ~depth:(depth + 1)

let rec type_of node =
  match node.g with
  | Nothing | Textual -> Type.Any Type.string
  | Boolean -> Type.Any Type.bool
  | Integral -> Type.Any Type.int64
  | Decimal -> Type.Any Type.float64
  | Listing e ->
      let (Type.Any t) = type_of e in
      Type.Any (Type.list t)
  | Object o when Hashtbl.length o.entries = 0 -> Type.Any Type.string
  | Object o ->
      Type.Any
        (Type.record
           (List.rev_map (fun (n, e) -> (n, type_of e.node)) o.order))

let sniff_from ?file ~rows r =
  protect ?file @@ fun () ->
  let framing, text = Scan.sample ~records:rows r in
  let s = Scan.make ~sampled:true framing (Reader.of_string text) in
  let top = new_node () and n = ref 0 in
  while Scan.next s do
    while !n < rows && Scan.record s do
      check_row s;
      observe_object s top ~depth:0;
      Scan.end_record s;
      incr n
    done
  done;
  match top.g with
  | Object o when o.order <> [] ->
      let columns =
        Array.of_list
          (List.rev_map
             (fun (name, e) ->
               { name; ty = type_of e.node; declared = false })
             o.order)
      in
      Ok { framing; columns; sniffed = Some !n }
  | Object _ -> Error (Error.v "no sampled row names a column")
  | _ -> Error (Error.v "the input holds no record")

let sample_rows = 16384

let sniff ?(rows = sample_rows) r =
  if rows < 1 then invalid "sniff" "rows is %d, not positive" rows;
  sniff_from ~rows r

let file ?format path =
  let open_ () =
    let ic = open_in_bin path in
    (Reader.of_in_channel ic, fun () -> close_in_noerr ic)
  in
  let source f =
    let name = Format.asprintf "json %a" Type.pp_quoted path in
    source_of ~name ~file:path f open_
  in
  match format with
  | Some f -> Ok (source f)
  | None -> (
      match open_ () with
      | exception Sys_error msg -> Error (Error.v msg)
      | r, close ->
          Fun.protect ~finally:close @@ fun () ->
          Result.map source (sniff_from ~file:path ~rows:sample_rows r))

(* Writing *)

let add_escaped b ~get ~len =
  Buffer.add_char b '"';
  for k = 0 to len - 1 do
    match get k with
    | '"' -> Buffer.add_string b "\\\""
    | '\\' -> Buffer.add_string b "\\\\"
    | '\b' -> Buffer.add_string b "\\b"
    | '\012' -> Buffer.add_string b "\\f"
    | '\n' -> Buffer.add_string b "\\n"
    | '\r' -> Buffer.add_string b "\\r"
    | '\t' -> Buffer.add_string b "\\t"
    | c when c < ' ' -> Printf.bprintf b "\\u%04x" (Char.code c)
    | c -> Buffer.add_char b c
  done;
  Buffer.add_char b '"'

(* [cells c] is the bytes of each row of the text column [c], [None] at a
   null. *)
let cells c =
  match Column.layout c with
  | Fixed _ | Children _ -> assert false (* [Column.print] makes text. *)
  | Varsize { validity; offsets; child } ->
      let valid = Option.map Nx.to_array validity
      and o = Nx.to_array offsets
      and a =
        Bigarray.array1_of_genarray
          (Nx.to_bigarray Bigarray.int8_unsigned
             (Column.to_tensor Nx.uint8 child))
      in
      fun i ->
        if Option.fold ~none:false ~some:(fun v -> not v.(i)) valid then None
        else
          let pos = Int64.to_int o.(i) in
          let get k = Char.unsafe_chr (A1.unsafe_get a (pos + k)) in
          Some (get, Int64.to_int o.(i + 1) - pos)

let valid_of c =
  match Column.validity c with
  | None -> Fun.const true
  | Some v ->
      let v = Nx.to_array v in
      fun i -> v.(i)

(* A float's text reads as an integer without a point or an exponent, so a
   column of whole floats would read back as integers. *)
let add_float b ~get ~len ~column =
  let c0 = get 0 in
  if c0 = 'n' || c0 = 'i' || (c0 = '-' && len > 1 && get 1 = 'i') then
    raise
      (Unwritable
         (Printf.sprintf
            "column %s: JSON has no text for an infinity or NaN. Filter them \
             out or replace them with null first."
            (quoted column)));
  let plain = ref true in
  for k = 0 to len - 1 do
    Buffer.add_char b (get k);
    if get k = '.' || get k = 'e' || get k = 'E' then plain := false
  done;
  if !plain then Buffer.add_string b ".0"

(* [writer b t c ~path] is the function that adds row [i] of the column [c], of
   type [t], to [b]. *)
let rec writer b (Type.Any t) c ~path =
  let nullable f i =
    if valid_of c i then f i else Buffer.add_string b "null"
  in
  (* [scalar add] adds a cell's text to [b]. *)
  let scalar c add =
    let cells = cells c in
    nullable (fun i ->
        match cells i with
        | None -> Buffer.add_string b "null"
        | Some (get, len) -> add ~get ~len)
  in
  match t with
  | Bool | Int8 | Int16 | Int32 | Int64 | Uint8 | Uint16 | Uint32 | Uint64 ->
      scalar (Column.print c) (fun ~get ~len ->
          for k = 0 to len - 1 do
            Buffer.add_char b (get k)
          done)
  | Float16 | Float32 | Float64 ->
      scalar (Column.print c) (fun ~get ~len ->
          add_float b ~get ~len ~column:path)
  | String | Categorical _ | Date | Datetime _ ->
      scalar (Column.print c) (fun ~get ~len -> add_escaped b ~get ~len)
  | Binary ->
      scalar c (fun ~get ~len ->
          if not (String.is_valid_utf_8 (String.init len get)) then
            raise
              (Unwritable
                 (Printf.sprintf
                    "column %s: a binary value is not valid UTF-8, which JSON \
                     text holds."
                    (quoted path)));
          add_escaped b ~get ~len)
  | List e ->
      let offsets, child =
        match Column.layout c with
        | Varsize { offsets; child; _ } -> (Nx.to_array offsets, child)
        | _ -> assert false
      in
      let element = writer b (Type.Any e) child ~path in
      nullable (fun i ->
          Buffer.add_char b '[';
          let first = Int64.to_int offsets.(i) in
          for j = first to Int64.to_int offsets.(i + 1) - 1 do
            if j > first then Buffer.add_char b ',';
            element j
          done;
          Buffer.add_char b ']')
  | Record fields ->
      let cols =
        match Column.layout c with
        | Children { fields; _ } -> List.map snd fields
        | _ -> assert false
      in
      let members =
        List.map2
          (fun (name, ty) col ->
            (name, writer b ty col ~path:(path ^ "." ^ name)))
          fields cols
      in
      nullable (fun i -> add_object b members i)
  | Clock _ | Duration _ | Tensor _ | Ext _ -> assert false (* [reads]. *)

and add_object b members i =
  Buffer.add_char b '{';
  List.iteri
    (fun k (name, w) ->
      if k > 0 then Buffer.add_char b ',';
      add_escaped b ~get:(String.get name) ~len:(String.length name);
      Buffer.add_char b ':';
      w i)
    members;
  Buffer.add_char b '}'

let encode ?(eod = false) f q w =
  let s = schema f in
  if not (Schema.equal (Query.schema q) s) then
    invalid "encode" "the query's columns are not the format's: %s, against %s"
      (Format.asprintf "%a" Schema.pp (Query.schema q))
      (Format.asprintf "%a" Schema.pp s);
  protect @@ fun () ->
  let b = Buffer.create 65536 and row = ref 0 in
  let flush () =
    Bytesrw.Bytes.Writer.write_string w (Buffer.contents b);
    Buffer.clear b
  in
  if f.framing = Array then Buffer.add_string b "[\n";
  let batch () t =
    let members =
      Array.to_list
        (Array.map
           (fun c ->
             (c.name, writer b c.ty (Talon.column t c.name) ~path:c.name))
           f.columns)
    in
    for i = 0 to Talon.rows t - 1 do
      let start = Buffer.length b in
      if f.framing = Array && !row > 0 then Buffer.add_string b ",\n";
      (try add_object b members i
       with Unwritable msg ->
         Buffer.truncate b start;
         flush ();
         let line = !row + if f.framing = Array then 2 else 1 in
         raise (Encoding_failed (Error.v ~line msg)));
      incr row;
      if f.framing = Lines then Buffer.add_char b '\n';
      if Buffer.length b >= 1 lsl 16 then flush ()
    done
  in
  let r = Query.fold q ~init:() batch in
  if Result.is_ok r && f.framing = Array then
    Buffer.add_string b (if !row = 0 then "]\n" else "\n]\n");
  flush ();
  if eod && Result.is_ok r then Bytesrw.Bytes.Writer.write_eod w;
  r
