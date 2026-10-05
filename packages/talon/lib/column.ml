(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A validity: bits set where a row holds a value, and the number of rows they
   leave null, [-1] until it is read. The count is a fact about the bits, so it
   lives with them: a column that keeps a validity keeps its count, columns that
   share one count it once, and a column with new bits has a new count. Two
   domains that read the count at once store the same number. *)
type validity = { bits : Nx.bit_t; nulls : int Atomic.t }

type t = {
  type_ : Type.any;
  length : int;
  validity : validity option;
  data : data;
}

and data =
  | Fixed of Nx.packed
  | Bytes of (int, Nx.uint8_elt) Nx_ragged.t
  | List of { offsets : Nx.int64_t; child : t }
  | Fields of t list

let err fmt = Format.kasprintf invalid_arg fmt
let type_ c = c.type_
let length c = c.length
let unread bits = { bits; nulls = Atomic.make (-1) }
let counted bits n = { bits; nulls = Atomic.make n }

(* [count length v] is the nulls of [v], a validity of [length] rows: read once,
   then kept. *)
let count length v =
  match Atomic.get v.nulls with
  | -1 ->
      let n = length - Int64.to_int (Nx.item [] (Nx.count v.bits)) in
      Atomic.set v.nulls n;
      n
  | n -> n

let null_count c =
  match c.validity with None -> 0 | Some v -> count c.length v

let known_zero c =
  match c.validity with None -> true | Some v -> Atomic.get v.nulls = 0

(* [known c] is [c]'s null count if it was read, without reading it. *)
let known c =
  match c.validity with
  | None -> Some 0
  | Some v -> ( match Atomic.get v.nulls with -1 -> None | n -> Some n)

let validity c = Option.map (fun v -> v.bits) c.validity
let data c = c.data
let has_type ty c = match c.type_ with Any t -> Type.equal t ty

type mask = validity

let mask bits = unread bits

let restrict m c =
  if Nx.shape m.bits <> [| c.length |] then
    err "Column.restrict: a mask of shape %a for %d rows" Nx.pp_shape
      (Nx.shape m.bits) c.length;
  match c.validity with Some _ -> c | None -> { c with validity = Some m }

(* A value its type does not hold raises [Refused] with the reason while a
   column is encoded. *)

exception Refused of string

let refused fmt = Format.kasprintf (fun r -> raise (Refused r)) fmt

(* Scalars

   A scalar type stores one element of [dtype] per row. [load] reads a stored
   value back; where it can fall outside the OCaml type, [outside] says why. *)

type 'a scalar =
  | Scalar : {
      dtype : ('b, 'c) Nx.dtype;
      store : 'a -> 'b;
      load : 'b -> 'a;
      outside : ('b -> string option) option;
    }
      -> 'a scalar

let cell ?outside dtype store load =
  Some (Scalar { dtype; store; load; outside })

let fits_int x = Int64.equal (Int64.of_int (Int64.to_int x)) x
let check_int pp ok x = if ok x then None else Some (pp x ^ " is outside int")

(* A temporal type's ticks, read back through [of_ticks], which is [None]
   outside {!Time}'s range. *)
let ticks ty (of_ticks : int64 -> 'a option) to_ticks =
  let outside x =
    match of_ticks x with
    | Some _ -> None
    | None ->
        Some (Format.asprintf "%a tick %Ld is outside the range" Type.pp ty x)
  in
  cell ~outside Nx.int64 to_ticks (fun x -> Option.get (of_ticks x))

let span_ticks ty : Type.unit_ -> Time.span scalar option = function
  | S -> ticks ty Time.Span.of_s Time.Span.to_s
  | Ms -> ticks ty Time.Span.of_ms Time.Span.to_ms
  | Us -> ticks ty Time.Span.of_us Time.Span.to_us
  | Ns -> cell Nx.int64 Time.Span.to_ns Time.Span.of_ns

let instant_ticks ty : Type.unit_ -> Time.instant scalar option = function
  | S -> ticks ty Time.of_s Time.to_s
  | Ms -> ticks ty Time.of_ms Time.to_ms
  | Us -> ticks ty Time.of_us Time.to_us
  | Ns -> cell Nx.int64 Time.to_ns Time.of_ns

let scalar : type a. a Type.t -> a scalar option = function
  | Bool -> cell Nx.bool Fun.id Fun.id
  | Int8 -> cell Nx.int8 Fun.id Fun.id
  | Int16 -> cell Nx.int16 Fun.id Fun.id
  | Int32 -> cell Nx.int32 Int32.of_int Int32.to_int
  | Int64 ->
      let outside = check_int (Printf.sprintf "%Ld") fits_int in
      cell ~outside Nx.int64 Int64.of_int Int64.to_int
  | Uint8 -> cell Nx.uint8 Fun.id Fun.id
  | Uint16 -> cell Nx.uint16 Fun.id Fun.id
  | Uint32 ->
      cell Nx.uint32 Int32.of_int (fun x -> Int32.to_int x land 0xffff_ffff)
  | Uint64 ->
      let ok x = Int64.compare x 0L >= 0 && fits_int x in
      let outside = check_int (Printf.sprintf "%Lu") ok in
      cell ~outside Nx.uint64 Int64.of_int Int64.to_int
  | Float16 -> cell Nx.float16 Fun.id Fun.id
  | Float32 -> cell Nx.float32 Fun.id Fun.id
  | Float64 -> cell Nx.float64 Fun.id Fun.id
  | Categorical d as ty ->
      let index =
        lazy
          (let index = Hashtbl.create (Iarray.length d) in
           Iarray.iteri (fun i s -> Hashtbl.add index s (Int32.of_int i)) d;
           index)
      in
      let code s =
        match Hashtbl.find_opt (Lazy.force index) s with
        | Some c -> c
        | None -> refused "%a does not hold %a" Type.pp ty Type.pp_quoted s
      in
      cell Nx.int32 code (fun c -> Iarray.get d (Int32.to_int c))
  | Date ->
      let store d = Int32.of_int (Time.Date.to_days d) in
      cell Nx.int32 store (fun x ->
          Option.get (Time.Date.of_days (Int32.to_int x)))
  | Clock u as ty -> span_ticks ty u
  | Duration u as ty -> span_ticks ty u
  | Datetime { unit_; _ } as ty -> instant_ticks ty unit_
  | String | Binary | List _ | Record _ | Tensor _ | Ext _ -> None

(* Making columns *)

(* [stores ty n d] is [true] iff [d] is [ty]'s storage for [n] rows. *)
let rec stores : type a. a Type.t -> int -> data -> bool =
 fun ty n d ->
  match (ty, d) with
  | Ext { storage; _ }, d -> stores storage n d
  | (String | Binary), Bytes r -> Nx_ragged.length r = n
  | List e, List { offsets; child } ->
      Nx.shape offsets = [| n + 1 |] && has_type e child
  | Record fields, Fields cs ->
      List.compare_lengths fields cs = 0
      && List.for_all2
           (fun (_, Type.Any t) c -> has_type t c && c.length = n)
           fields cs
  | Tensor (dt, cell), Fixed (P x) ->
      Nx_dtype.equal dt (Nx.dtype x)
      && Nx.shape x = Array.append [| n |] (Iarray.to_array cell)
  | _, Fixed (P x) -> (
      match scalar ty with
      | Some (Scalar s) ->
          Nx_dtype.equal s.dtype (Nx.dtype x) && Nx.shape x = [| n |]
      | None -> false)
  | _ -> false

(* [with_bits fn ty bits ~length data] is the column of [ty] whose validity is
   [bits], its count not yet read. *)
let with_bits fn type_ bits ~length data =
  (match bits with
  | Some b when Nx.shape b <> [| length |] ->
      err "Column.%s: a validity of shape %a for %d rows" fn Nx.pp_shape
        (Nx.shape b) length
  | _ -> ());
  { type_; length; validity = Option.map unread bits; data }

let make (Type.Any ty as type_) ?validity ~length data =
  if not (stores ty length data) then
    err "Column.make: the data is not %a's storage for %d rows" Type.pp ty
      length;
  with_bits "make" type_ validity ~length data

let with_data (Type.Any ty as type_) data c =
  if not (stores ty c.length data) then
    err "Column.with_data: the data is not %a's storage for %d rows" Type.pp ty
      c.length;
  { c with type_; data }

(* [retype ty c] is [c] as a column of [ty], a type with [c]'s storage: an
   extension type over [c]'s, or the reverse. *)
let rec retype : type a. a Type.t -> t -> t =
 fun ty c ->
  let data =
    match (ty, c.data) with
    | List e, List l -> List { l with child = retype e l.child }
    | Ext { storage; _ }, _ -> (retype storage c).data
    | _, d -> d
  in
  { c with type_ = Any ty; data }

(* Encoding

   A builder takes a type's values one row at a time and makes the column. The
   builders of lists and records prefix a refusal with the element or the
   field. *)

type 'a builder = { add : 'a option -> unit; finish : unit -> t }

(* Growable arrays *)
type 'a buf = { mutable items : 'a array; mutable len : int }

let buf x = { items = Array.make 16 x; len = 0 }

let push b x =
  if b.len = Array.length b.items then b.items <- Array.append b.items b.items;
  Array.unsafe_set b.items b.len x;
  b.len <- b.len + 1

let contents b = Array.sub b.items 0 b.len
let offsets_tensor b = Nx.create Nx.int64 [| b.len |] (contents b)

(* [check ty v] refuses a value [ty] does not hold. The builders of lists and
   records check their elements and fields, and a categorical's store checks its
   strings. *)
let check : type a. a Type.t -> a -> unit =
 fun ty ->
  match ty with
  | Categorical _ | List _ | Record _ | Ext _ -> ignore
  | _ ->
      let held = Type.holds ty in
      fun v ->
        if not (held v) then
          refused "%a does not hold %a" Type.pp ty (Type.pp_value ty) v

(* [rows ty ~null ~add ~data] is the builder that keeps the validity of each
   row, calls [null] or [add] to keep its value, and makes the column from [data
   ()]. *)
let rows ty ~null ~add ~data =
  let valid = buf true and nulls = ref 0 in
  let add = function
    | None ->
        null ();
        incr nulls;
        push valid false
    | Some v ->
        add v;
        push valid true
  in
  let finish () =
    let length = valid.len in
    let c = make (Any ty) ~length (data ()) in
    if !nulls = 0 then c
    else
      let bits =
        Nx.cast Nx.bit (Nx.create Nx.bool [| length |] (contents valid))
      in
      { c with validity = Some (counted bits !nulls) }
  in
  { add; finish }

let uint8_of_string s =
  let a = Bigarray.(Array1.create int8_unsigned c_layout (String.length s)) in
  String.iteri (fun i c -> Bigarray.Array1.unsafe_set a i (Char.code c)) s;
  Nx.of_bigarray (Bigarray.genarray_of_array1 a)

let bytes_builder ty raw =
  let check = check ty and bytes = Buffer.create 256 and offsets = buf 0L in
  let next () = push offsets (Int64.of_int (Buffer.length bytes)) in
  next ();
  let add v =
    check v;
    Buffer.add_string bytes (raw v);
    next ()
  in
  let data () =
    let values = uint8_of_string (Buffer.contents bytes) in
    Bytes (Nx_ragged.v ~offsets:(offsets_tensor offsets) values)
  in
  rows ty ~null:next ~add ~data

let scalar_builder ty (Scalar s) =
  let check = check ty and zero = Nx_dtype.zero s.dtype in
  let values = buf zero in
  let add v =
    check v;
    push values (s.store v)
  in
  let data () =
    Fixed (P (Nx.create s.dtype [| values.len |] (contents values)))
  in
  rows ty ~null:(fun () -> push values zero) ~add ~data

let rec builder : type a. a Type.t -> a builder =
 fun ty ->
  match ty with
  | String -> bytes_builder ty Fun.id
  | Binary -> bytes_builder ty (fun b -> (b :> string))
  | List e -> list_builder ty e
  | Record fields -> record_builder ty fields
  | Tensor (dt, cell) -> tensor_builder ty dt (Iarray.to_array cell)
  | Ext { storage; _ } ->
      let b = builder storage in
      let add = function None -> b.add None | Some (_ : Type.ext) -> . in
      { add; finish = (fun () -> retype ty (b.finish ())) }
  | _ -> (
      match scalar ty with
      | Some s -> scalar_builder ty s
      | None -> assert false)

and list_builder : type a. a array Type.t -> a Type.t -> a array builder =
 fun ty e ->
  let child = builder e and offsets = buf 0L and count = ref 0 in
  let next () = push offsets (Int64.of_int !count) in
  next ();
  let add_element j v =
    match child.add (Some v) with
    | () -> incr count
    | exception Refused r -> refused "element %d: %s" j r
  in
  let add vs =
    Array.iteri add_element vs;
    next ()
  in
  let data () =
    List { offsets = offsets_tensor offsets; child = child.finish () }
  in
  rows ty ~null:next ~add ~data

and record_builder :
    Record.t Type.t -> (string * Type.any) list -> Record.t builder =
 fun ty fields ->
  let names = List.map fst fields in
  let children = Array.of_list (List.map field_builder fields) in
  let null () = Array.iter (fun c -> c.add None) children in
  let add (r : Record.t) =
    if not (List.equal String.equal names (Record.names r)) then
      refused "%a does not hold a record of fields %a" Type.pp ty
        (Type.pp_list Type.pp_name)
        (Record.names r);
    Iarray.iteri (fun j (_, f) -> children.(j).add (Some f)) r.Kind.fields
  in
  let data () =
    Fields (Array.to_list (Array.map (fun c -> c.finish ()) children))
  in
  rows ty ~null ~add ~data

(* A field whose type is or holds an extension holds its storage's values in a
   [Storage] field. *)
and field_builder (name, Type.Any ft) =
  let (Any st) = Type.storage ft in
  stored_field name ~ext:(Kind.has_ext (Type.kind ft)) ft st

and stored_field : type a b.
    string -> ext:bool -> a Type.t -> b Type.t -> Kind.field builder =
 fun name ~ext ft st ->
  let b = builder st and k = Type.kind st in
  let mismatch : type c. c Kind.t -> unit =
   fun k' ->
    refused "field %a: %a does not hold a %a value" Type.pp_name name Type.pp ft
      Kind.pp k'
  in
  let put : type c. c Kind.t -> c option -> unit =
   fun k' v ->
    match Kind.equal_witness k' k with
    | None -> mismatch k'
    | Some Equal -> (
        match b.add v with
        | () -> ()
        | exception Refused r -> refused "field %a: %s" Type.pp_name name r)
  in
  let add = function
    | None | Some (Kind.Value (_, None) | Storage (_, None)) -> b.add None
    | Some (Kind.Value (k', v)) -> if ext then mismatch k' else put k' v
    | Some (Storage (k', v)) -> if ext then put k' v else mismatch k'
  in
  { add; finish = (fun () -> retype ft (b.finish ())) }

and tensor_builder : type a b.
    (a, b) Nx.t Type.t -> (a, b) Nx.dtype -> int array -> (a, b) Nx.t builder =
 fun ty dt cell ->
  let check = check ty and zero = Nx.zeros dt cell in
  let cells = buf zero in
  let add x =
    check x;
    push cells x
  in
  let data () =
    if cells.len = 0 then Fixed (P (Nx.zeros dt (Array.append [| 0 |] cell)))
    else Fixed (P (Nx.stack ~axis:0 (Array.to_list (contents cells))))
  in
  rows ty ~null:(fun () -> push cells zero) ~add ~data

let encode ty n f =
  if n < 0 then err "Column.encode: %d rows" n;
  let b = builder ty in
  let rec loop i =
    if i = n then Ok (b.finish ())
    else
      match b.add (f i) with
      | () -> loop (i + 1)
      | exception Refused r -> Error (i, r)
  in
  loop 0

(* Decoding

   A reader's [get] reads a row that is not null. Its [bad], absent for types
   whose stored values all read, says why a row's value is outside the OCaml
   type; the decoder asks it of every non-null row before any [get]. *)

type 'a reader = { get : int -> 'a; bad : (int -> string option) option }

let flags c = Option.map Nx.to_array (validity c)
let is_valid flags i = match flags with None -> true | Some f -> f.(i)
let int_offsets o = Array.map Int64.to_int (Nx.to_array o)

let bytes_reader (r : Strings.bytes) =
  let o = int_offsets (Nx_ragged.offsets r) in
  let a = Bigarray.array1_of_genarray (Nx.to_bigarray (Nx_ragged.values r)) in
  let get i =
    String.init (o.(i + 1) - o.(i)) (fun k -> Char.unsafe_chr a.{o.(i) + k})
  in
  { get; bad = None }

(* [first n f] is the first [Some] of [f 0], …, [f (n - 1)]. *)
let first n f =
  let rec loop i =
    if i = n then None else match f i with None -> loop (i + 1) | r -> r
  in
  loop 0

(* [make] keeps a column's data its type's storage, which leaves no other
   case. *)
let rec reader : type a. a Type.t -> t -> a reader =
 fun ty c ->
  match (ty, c.data) with
  | String, Bytes r -> bytes_reader r
  | Binary, Bytes r ->
      let r = bytes_reader r in
      { r with get = (fun i -> Binary.of_string (r.get i)) }
  | List e, List { offsets; child } -> list_reader e (int_offsets offsets) child
  | Record fields, Fields cs ->
      let fields = Array.of_list (List.map2 field_reader fields cs) in
      let get i =
        {
          Kind.fields =
            Iarray.init (Array.length fields) (fun j -> fields.(j).get i);
        }
      in
      let bad i =
        first (Array.length fields) (fun j ->
            match fields.(j).bad with Some bad -> bad i | None -> None)
      in
      let checked = Array.exists (fun f -> Option.is_some f.bad) fields in
      { get; bad = (if checked then Some bad else None) }
  | Tensor (dt, _), Fixed p ->
      let x = Nx.unpack dt p in
      { get = (fun i -> Nx.copy (Nx.get [ i ] x)); bad = None }
  | Ext _, _ ->
      let bad _ = Some "an extension value is read through its declaration" in
      { get = (fun _ -> assert false); bad = Some bad }
  | _, Fixed p -> (
      match scalar ty with
      | Some (Scalar s) ->
          let a = Nx.to_array (Nx.unpack s.dtype p) in
          let bad = Option.map (fun outside i -> outside a.(i)) s.outside in
          { get = (fun i -> s.load a.(i)); bad }
      | None -> assert false)
  | _ -> assert false

and list_reader : type a. a Type.t -> int array -> t -> a array reader =
 fun e o child ->
  let r = reader e child and flags = flags child in
  let get i = Array.init (o.(i + 1) - o.(i)) (fun k -> r.get (o.(i) + k)) in
  let element i k =
    let j = o.(i) + k in
    if not (is_valid flags j) then Some (Printf.sprintf "element %d is null" k)
    else
      match r.bad with
      | None -> None
      | Some bad -> Option.map (Printf.sprintf "element %d: %s" k) (bad j)
  in
  let bad i = first (o.(i + 1) - o.(i)) (element i) in
  let nulls = null_count child in
  { get; bad = (if nulls = 0 && r.bad = None then None else Some bad) }

and field_reader (name, Type.Any ft) c =
  let (Any st) = Type.storage ft in
  stored_reader name ~ext:(Kind.has_ext (Type.kind ft)) st c

and stored_reader : type a.
    string -> ext:bool -> a Type.t -> t -> (string * Kind.field) reader =
 fun name ~ext st c ->
  let r = reader st (retype st c) and flags = flags c and k = Type.kind st in
  let get i =
    let v = if is_valid flags i then Some (r.get i) else None in
    (name, if ext then Kind.Storage (k, v) else Kind.Value (k, v))
  in
  let field bad i =
    if not (is_valid flags i) then None
    else Option.map (Format.asprintf "field %a: %s" Type.pp_name name) (bad i)
  in
  { get; bad = Option.map field r.bad }

let decoder ty c =
  if not (has_type ty c) then
    err "Column.decoder: a column of %a is not one of %a"
      (fun ppf (Type.Any t) -> Type.pp ppf t)
      c.type_ Type.pp ty;
  let r = reader ty c and flags = flags c in
  let check bad i =
    if is_valid flags i then Option.map (fun why -> (i, why)) (bad i) else None
  in
  match Option.bind r.bad (fun bad -> first c.length (check bad)) with
  | Some e -> Error e
  | None -> Ok (fun i -> if is_valid flags i then Some (r.get i) else None)

(* OCaml values *)

let v ty vs =
  match encode ty (Array.length vs) (fun i -> Some vs.(i)) with
  | Ok c -> c
  | Error (row, why) -> err "Column.v: row %d: %s" row why

let of_options ty vs =
  match encode ty (Array.length vs) (Array.get vs) with
  | Ok c -> c
  | Error (row, why) -> err "Column.of_options: row %d: %s" row why

(* [read fn k c] decodes [c] as [k], raising as [fn]. *)
let read : type a. string -> a Kind.t -> t -> int -> a option =
 fun fn k c ->
  let (Any ty) = c.type_ in
  match Kind.provably_equal (Type.kind ty) k with
  | None when Kind.has_ext (Type.kind ty) ->
      err "Column.%s: no kind reads %a" fn Type.pp ty
  | None -> err "Column.%s: %a is not read as %a" fn Type.pp ty Kind.pp k
  | Some Equal -> (
      match decoder ty c with
      | Ok get -> get
      | Error (row, why) -> err "Column.%s: row %d: %s" fn row why)

let options k c = Array.init c.length (read "options" k c)

let values k c =
  let get = read "values" k c in
  let value i =
    match get i with
    | Some v -> v
    | None -> err "Column.values: row %d is null" i
  in
  Array.init c.length value

(* Tensors and bytes *)

let first_null c =
  let flags = Option.get (flags c) in
  Option.get (first c.length (fun i -> if flags.(i) then None else Some i))

(* [cells fn x] is the type of the rows of [x]: its dtype's scalar type for a
   1-D [x], and a tensor type of its cells otherwise. *)
let cells : type a b. string -> (a, b) Nx.t -> Type.any =
 fun fn x ->
  let shape = Nx.shape x and dt = Nx.dtype x in
  if Array.length shape > 1 then
    Any (Type.tensor dt (Array.sub shape 1 (Array.length shape - 1)))
  else
    match dt with
    | Bool -> Any Type.bool
    | Int8 -> Any Type.int8
    | Int16 -> Any Type.int16
    | Int32 -> Any Type.int32
    | Int64 -> Any Type.int64
    | UInt8 -> Any Type.uint8
    | UInt16 -> Any Type.uint16
    | UInt32 -> Any Type.uint32
    | UInt64 -> Any Type.uint64
    | Float16 -> Any Type.float16
    | Float32 -> Any Type.float32
    | Float64 -> Any Type.float64
    | BFloat16 | Float8_e4m3 | Float8_e5m2 | Int4 | UInt4 | Bit | Complex64
    | Complex128 ->
        err "Column.%s: no scalar type stores %a; make it 2-D" fn Nx_dtype.pp dt

let of_tensor ?validity x =
  let shape = Nx.shape x in
  if shape = [||] then err "Column.of_tensor: a scalar has no rows";
  let length = shape.(0) in
  let type_ = cells "of_tensor" x in
  with_bits "of_tensor" type_ validity ~length (Fixed (P x))

let to_tensor (type a b) (dt : (a, b) Nx.dtype) c : (a, b) Nx.t =
  let (Any ty) = c.type_ in
  match c.data with
  | Fixed (P x) when not (Nx_dtype.equal dt (Nx.dtype x)) ->
      err "Column.to_tensor: %a is stored as %a, not %a" Type.pp ty Nx_dtype.pp
        (Nx.dtype x) Nx_dtype.pp dt
  | Fixed _ when null_count c > 0 ->
      err "Column.to_tensor: row %d is null" (first_null c)
  | Fixed p -> Nx.unpack dt p
  | _ -> err "Column.to_tensor: %a is not stored one element per row" Type.pp ty

(* Structural operations *)

let rows_of_tensor x ~offset ~length =
  let range i d = if i = 0 then (offset, offset + length) else (0, d) in
  Nx.shrink (Array.mapi range (Nx.shape x)) x

let rec sub c ~offset ~length =
  if offset < 0 || length < 0 || offset + length > c.length then
    err "Column.sub: rows %d to %d of %d rows" offset (offset + length) c.length;
  (* A count of 0 or of every row holds for any rows. *)
  let validity =
    match (c.validity, known c) with
    | None, _ | _, Some 0 -> None
    | Some v, Some n when n = c.length ->
        Some (counted (Nx.shrink [| (offset, offset + length) |] v.bits) length)
    | Some v, _ ->
        Some (unread (Nx.shrink [| (offset, offset + length) |] v.bits))
  in
  let data =
    match c.data with
    | Fixed (P x) -> Fixed (P (rows_of_tensor x ~offset ~length))
    | Bytes r -> Bytes (Nx_ragged.sub r ~offset ~length)
    | List { offsets; child } ->
        let offsets = Nx.shrink [| (offset, offset + length + 1) |] offsets in
        List { offsets; child }
    | Fields cs -> Fields (List.map (sub ~offset ~length) cs)
  in
  { type_ = c.type_; length; validity; data }

(* Ragged arrays *)

let ragged (type a b) (dt : (a, b) Nx.dtype) c : (a, b) Nx_ragged.t =
  let (Any ty) = c.type_ in
  let mismatch stored =
    err "Column.ragged: %a is stored as %a, not %a" Type.pp ty Nx_dtype.pp
      stored Nx_dtype.pp dt
  in
  if null_count c > 0 then err "Column.ragged: row %d is null" (first_null c);
  match c.data with
  | Bytes r -> (
      match Nx_dtype.equal_witness Nx.uint8 dt with
      | Some Equal -> r
      | None -> mismatch Nx.uint8)
  | List { offsets; child = { data = Fixed (P x); _ } as child } ->
      if not (Nx_dtype.equal dt (Nx.dtype x)) then mismatch (Nx.dtype x);
      let r = Nx_ragged.v ~offsets (Nx.unpack dt (P x)) in
      (* Only the elements of the rows count: values outside the offsets are not
         the column's. *)
      (if null_count child > 0 then
         let first = Int64.to_int (Nx.item [ 0 ] offsets) in
         let last = Int64.to_int (Nx.item [ c.length ] offsets) in
         let elements = sub child ~offset:first ~length:(last - first) in
         if null_count elements > 0 then
           let j = Int64.of_int (first + first_null elements) in
           let ends = Nx.shrink [| (1, c.length + 1) |] offsets in
           let row =
             Nx.item [] (Nx.sum (Nx.cast Nx.int64 (Nx.less_equal_s ends j)))
           in
           err "Column.ragged: an element of row %Ld is null" row);
      r
  | _ ->
      err "Column.ragged: %a is not a list of %a, text or bytes" Type.pp ty
        Nx_dtype.pp dt

let of_ragged ?validity r =
  let values = Nx_ragged.values r in
  let (Type.Any e as cell) = cells "of_ragged" values in
  let child =
    {
      type_ = cell;
      length = Nx.dim 0 values;
      validity = None;
      data = Fixed (P values);
    }
  in
  let length = Nx_ragged.length r in
  with_bits "of_ragged"
    (Any (Type.list e))
    validity ~length
    (List { offsets = Nx_ragged.offsets r; child })

(* [gather_data indices c] is the data of [c]'s rows at [indices], zeros and
   empty rows outside [c]'s rows. *)
let rec gather_data indices c =
  match c.data with
  | Fixed (P x) -> Fixed (P (Nx.take ~axis:0 ~indices x))
  | Bytes r -> Bytes (Nx_ragged.take ~indices r)
  | List { offsets; child } ->
      let elements = Nx.arange Nx.int64 0 child.length 1 in
      let at = Nx_ragged.take ~indices (Nx_ragged.v ~offsets elements) in
      List
        {
          offsets = Nx_ragged.offsets at;
          child = gather (Nx_ragged.values at) child;
        }
  | Fields cs -> Fields (List.map (gather indices) cs)

and gather indices c =
  let validity =
    Option.map (fun v -> unread (Nx.take ~indices v.bits)) c.validity
  in
  {
    type_ = c.type_;
    length = Nx.dim 0 indices;
    validity;
    data = gather_data indices c;
  }

(* A permutation moves the nulls but keeps their number. *)
let permute p c =
  let validity =
    Option.map
      (fun v -> counted (Nx.take ~indices:p v.bits) (Atomic.get v.nulls))
      c.validity
  in
  { c with validity; data = gather_data p c }

(* [bounds offsets] is the first and last of [offsets]. *)
let bounds offsets =
  let last = Nx.dim 0 offsets - 1 in
  (Int64.to_int (Nx.item [ 0 ] offsets), Int64.to_int (Nx.item [ last ] offsets))

(* [rebase offsets first] is [offsets] less [first], over contiguous storage. *)
let rebase offsets first =
  if first = 0 then Nx.contiguous offsets
  else Nx.sub_s offsets (Int64.of_int first)

(* [exact offsets child] is [offsets] from [0] and the rows of [child] they
   cut. *)
let exact offsets child =
  let first, last = bounds offsets in
  let child =
    if first = 0 && last = child.length then child
    else sub child ~offset:first ~length:(last - first)
  in
  (rebase offsets first, child)

(* Canonical bits start at bit [0] of their storage and leave the bits past
   their length unset, so that equal bits have equal bytes. The check reads one
   byte. *)
let canonical_bits b =
  let n = Nx.numel b in
  Nx.contiguous b == b
  && (n land 7 = 0
     || Strings.reading ~by:"Column.canonical" b (fun bytes ->
         let bytes = Nx_device.Buffer.bigarray Bigarray.int8_unsigned bytes in
         Bigarray.Array1.get bytes (n / 8) lsr (n land 7) = 0))

let rec canonical c =
  let validity =
    match c.validity with
    | Some v when count c.length v = 0 -> None
    | Some v as kept when canonical_bits v.bits -> kept
    | Some v -> Some (counted (Nx.copy v.bits) (count c.length v))
    | None -> None
  in
  let data =
    match c.data with
    | Fixed (P x) as d ->
        let y = Nx.contiguous x in
        if y == x then d else Fixed (P y)
    | Bytes r as d ->
        let offsets = Nx_ragged.offsets r and values = Nx_ragged.values r in
        let first, last = bounds offsets in
        let offsets' = rebase offsets first in
        let values' =
          if first = 0 && last = Nx.dim 0 values then Nx.contiguous values
          else Nx.contiguous (Nx.shrink [| (first, last) |] values)
        in
        if offsets' == offsets && values' == values then d
        else Bytes (Nx_ragged.v ~offsets:offsets' values')
    | List { offsets; child } as d ->
        let offsets', child' = exact offsets child in
        let child' = canonical child' in
        if offsets' == offsets && child' == child then d
        else List { offsets = offsets'; child = child' }
    | Fields cs as d ->
        let cs' = List.map canonical cs in
        if List.for_all2 ( == ) cs cs' then d else Fields cs'
  in
  if validity == c.validity && data == c.data then c
  else { c with validity; data }

(* The rows of [cs] one after the other. Their validity is the parts' bits, ones
   for a part without, with the sum of their counts when every one is known. *)
let rec concat = function
  | [] -> invalid_arg "Column.concat: no column"
  | [ c ] -> c
  | c :: _ as cs ->
      let length = List.fold_left (fun n c -> n + c.length) 0 cs in
      let nulls =
        List.fold_left
          (fun n c ->
            match (n, known c) with Some n, Some k -> Some (n + k) | _ -> None)
          (Some 0) cs
      in
      let validity =
        if nulls = Some 0 then None
        else
          let bits c =
            match c.validity with
            | Some v -> v.bits
            | None -> Nx.ones Nx.bit [| c.length |]
          in
          let bits = Nx.concatenate ~axis:0 (List.map bits cs) in
          Some
            (match nulls with Some n -> counted bits n | None -> unread bits)
      in
      { type_ = c.type_; length; validity; data = join cs c.data }

and join cs = function
  | Fixed (P x) ->
      let part c =
        match c.data with
        | Fixed p -> Nx.unpack (Nx.dtype x) p
        | _ -> assert false
      in
      Fixed (P (Nx.concatenate ~axis:0 (List.map part cs)))
  | Bytes _ ->
      let part c = match c.data with Bytes r -> r | _ -> assert false in
      Bytes (Nx_ragged.concat (List.map part cs))
  | List _ ->
      let part c =
        match c.data with
        | List { offsets; child } -> exact offsets child
        | _ -> assert false
      in
      let parts = List.map part cs in
      let shift (base, tails) (offsets, child) =
        let tail = Nx.shrink [| (1, Nx.dim 0 offsets) |] offsets in
        (base + child.length, Nx.add_s tail (Int64.of_int base) :: tails)
      in
      let _, tails = List.fold_left shift (0, []) parts in
      let offsets =
        Nx.concatenate ~axis:0 (Nx.zeros Nx.int64 [| 1 |] :: List.rev tails)
      in
      List { offsets; child = concat (List.map snd parts) }
  | Fields fs ->
      let field i c =
        match c.data with Fields fs -> List.nth fs i | _ -> assert false
      in
      Fields (List.mapi (fun i _ -> concat (List.map (field i) cs)) fs)

(* Layouts *)

type layout =
  | Fixed of { validity : Nx.bit_t option; values : Nx.packed }
  | Varsize of { validity : Nx.bit_t option; offsets : Nx.int64_t; child : t }
  | Children of {
      validity : Nx.bit_t option;
      length : int;
      fields : (string * t) list;
    }

let refuse fmt = err ("Column.of_layout: " ^^ fmt)
let pp_any ppf (Type.Any t) = Type.pp ppf t

let fields_of ty =
  match Type.storage ty with Any (Record fs) -> fs | _ -> assert false

let layout c =
  let validity = validity c in
  match c.data with
  | Fixed values -> Fixed { validity; values }
  | Bytes r ->
      let values = Nx_ragged.values r in
      let child =
        {
          type_ = Any Type.uint8;
          length = Nx.dim 0 values;
          validity = None;
          data = Fixed (P values);
        }
      in
      Varsize { validity; offsets = Nx_ragged.offsets r; child }
  | List { offsets; child } -> Varsize { validity; offsets; child }
  | Fields cs ->
      let (Any ty) = c.type_ in
      let fields = List.map2 (fun (n, _) c -> (n, c)) (fields_of ty) cs in
      Children { validity; length = c.length; fields }

let check_validity validity n =
  match validity with
  | Some v when Nx.shape v <> [| n |] ->
      refuse "a validity of shape %a for %d rows" Nx.pp_shape (Nx.shape v) n
  | _ -> ()

let of_bits type_ validity ~length data =
  { type_; length; validity = Option.map unread validity; data }

(* [rows_of offsets child] is the number of rows that [offsets] cut from the
   rows of [child]. *)
let rows_of (offsets : Nx.int64_t) child =
  let shape = Nx.shape offsets in
  if Array.length shape <> 1 || shape.(0) = 0 then
    refuse "offsets of shape %a, not 1-D with an entry" Nx.pp_shape shape;
  let o = Bigarray.array1_of_genarray (Nx.to_bigarray offsets) in
  let n = shape.(0) - 1 in
  if Int64.compare o.{0} 0L < 0 then refuse "offsets start at %Ld" o.{0};
  for r = 0 to n - 1 do
    if Int64.compare o.{r + 1} o.{r} < 0 then
      refuse "offsets decrease at row %d" r
  done;
  if Int64.compare o.{n} (Int64.of_int child.length) > 0 then
    refuse "offsets end at %Ld, past the child's %d rows" o.{n} child.length;
  n

(* [rows_of_values ty x] is the rows of [x] if it lays out [ty], a scalar or
   tensor type. *)
let rows_of_values : type a. a Type.t -> Nx.packed -> int =
 fun ty (P x) ->
  let shape = Nx.shape x in
  let lays_out dt cell =
    Nx_dtype.equal dt (Nx.dtype x)
    && Array.length shape > 0
    && Array.sub shape 1 (Array.length shape - 1) = cell
  in
  let ok =
    match ty with
    | Tensor (dt, cell) -> lays_out dt (Iarray.to_array cell)
    | _ -> (
        match scalar ty with
        | Some (Scalar s) -> lays_out s.dtype [||]
        | None -> false)
  in
  if not ok then
    refuse "%a values of shape %a do not lay out %a" Nx_dtype.pp (Nx.dtype x)
      Nx.pp_shape shape Type.pp ty;
  shape.(0)

let ticks_per_day : Type.unit_ -> int64 = function
  | S -> 86_400L
  | Ms -> 86_400_000L
  | Us -> 86_400_000_000L
  | Ns -> 86_400_000_000_000L

(* [unheld ty c] is the first non-null row of the fixed-width column [c] whose
   stored value [ty] does not hold, and why. Every other fixed-width type holds
   all the values of its storage. *)
let unheld : type a. a Type.t -> t -> (int * string) option =
 fun ty c ->
  let first dt lo hi why =
    let x = match c.data with Fixed p -> Nx.unpack dt p | _ -> assert false in
    let bad = Nx.logical_or (Nx.less_s x lo) (Nx.greater_equal_s x hi) in
    let bad =
      match validity c with
      | Some v -> Nx.logical_and (Nx.cast Nx.bool v) bad
      | None -> bad
    in
    let rows = Nx.positions bad in
    if Nx.numel rows = 0 then None
    else
      let r = Int64.to_int (Nx.item [ 0 ] rows) in
      Some (r, Format.asprintf "%a %s" Type.pp ty (why (Nx.item [ r ] x)))
  in
  match ty with
  | Categorical d ->
      let n = Int32.of_int (Iarray.length d) in
      first Nx.int32 0l n (Printf.sprintf "has no code %ld")
  | Clock u ->
      first Nx.int64 0L (ticks_per_day u)
        (Printf.sprintf "tick %Ld is outside the day")
  | _ -> None

let held c = function Some e -> Error e | None -> Ok c

let rec of_layout : type a. a Type.t -> layout -> (t, int * string) result =
 fun ty l ->
  match (ty, l) with
  | Ext { storage; _ }, l -> Result.map (retype ty) (of_layout storage l)
  | (String | Binary), Varsize { validity; offsets; child } -> (
      if not (has_type Type.uint8 child) then
        refuse "a child of %a does not lay out %a" pp_any child.type_ Type.pp ty;
      if null_count child > 0 then
        refuse "a child with a null does not lay out %a" Type.pp ty;
      let length = rows_of offsets child in
      check_validity validity length;
      let values = match child.data with Fixed p -> p | _ -> assert false in
      let r = Nx_ragged.v ~offsets (Nx.unpack Nx.uint8 values) in
      let c = of_bits (Any ty) validity ~length (Bytes r) in
      match ty with
      | String -> held c (Strings.utf_8 ~by:"Column.of_layout" ?mask:validity r)
      | _ -> Ok c)
  | List e, Varsize { validity; offsets; child } ->
      if not (has_type e child) then
        refuse "a child of %a does not lay out %a" pp_any child.type_ Type.pp ty;
      let length = rows_of offsets child in
      check_validity validity length;
      Ok (of_bits (Any ty) validity ~length (List { offsets; child }))
  | Record fields, Children { validity; length; fields = cs } ->
      if not (List.equal String.equal (List.map fst fields) (List.map fst cs))
      then
        refuse "fields %a do not lay out %a"
          (Type.pp_list Type.pp_name)
          (List.map fst cs) Type.pp ty;
      if length < 0 then refuse "a length of %d" length;
      let field (n, Type.Any ft) (_, c) =
        if not (has_type ft c) then
          refuse "field %a of %a does not lay out %a" Type.pp_name n pp_any
            c.type_ Type.pp ty;
        if c.length <> length then
          refuse "field %a of %d rows for %d" Type.pp_name n c.length length
      in
      List.iter2 field fields cs;
      check_validity validity length;
      Ok (of_bits (Any ty) validity ~length (Fields (List.map snd cs)))
  | _, Fixed { validity; values } ->
      let length = rows_of_values ty values in
      check_validity validity length;
      let c = of_bits (Any ty) validity ~length (Fixed values) in
      held c (unheld ty c)
  | _, Varsize _ -> refuse "offsets and a child do not lay out %a" Type.pp ty
  | _, Children _ -> refuse "fields do not lay out %a" Type.pp ty

let of_layout (Type.Any ty) l = of_layout ty l
