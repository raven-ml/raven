(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type unit_ = S | Ms | Us | Ns
type ext = Kind.ext

type 'a t =
  | Bool : bool t
  | Int8 : int t
  | Int16 : int t
  | Int32 : int t
  | Int64 : int t
  | Uint8 : int t
  | Uint16 : int t
  | Uint32 : int t
  | Uint64 : int t
  | Float16 : float t
  | Float32 : float t
  | Float64 : float t
  | Decimal : { precision : int; scale : int } -> Decimal.t t
  | String : string t
  | Binary : Binary.t t
  | Categorical : string iarray -> string t
  | Date : Time.date t
  | Clock : unit_ -> Time.span t
  | Duration : unit_ -> Time.span t
  | Datetime : { unit_ : unit_; zone : string option } -> Time.instant t
  | List : 'a t -> 'a array t
  | Record : (string * any) list -> Record.t t
  | Tensor : ('a, 'b) Nx.dtype * int iarray -> ('a, 'b) Nx.t t
  | Ext : { name : string; metadata : string; storage : 's t } -> ext t

and any = Any : 'a t -> any

let err fmt = Format.kasprintf invalid_arg fmt

(* Constructors *)

let bool = Bool
let int8 = Int8
let int16 = Int16
let int32 = Int32
let int64 = Int64
let uint8 = Uint8
let uint16 = Uint16
let uint32 = Uint32
let uint64 = Uint64
let float16 = Float16
let float32 = Float32
let float64 = Float64

let decimal ~precision ~scale =
  if precision < 1 || precision > 18 then
    err "Type.decimal: precision %d is not in [1;18]" precision;
  if scale < 0 || scale > precision then
    err "Type.decimal: scale %d is not in [0;%d]" scale precision;
  Decimal { precision; scale }

let string = String
let binary = Binary

let categorical d =
  if
    (Array.length d > 0x7fff_ffff)
    [@mutate off "a dictionary of 2^31 strings is too large to test"]
  then err "Type.categorical: more than 2^31 - 1 strings";
  let seen = Hashtbl.create (Array.length d) in
  let add s =
    if not (String.is_valid_utf_8 s) then
      err "Type.categorical: %S is not UTF-8" s;
    if Hashtbl.mem seen s then err "Type.categorical: %S appears twice" s;
    Hashtbl.add seen s ()
  in
  Array.iter add d;
  Categorical (Iarray.of_array d)

let date = Date
let clock u = Clock u
let duration u = Duration u

let datetime ?zone unit_ =
  (match zone with
  | Some "" -> err "Type.datetime: empty zone"
  | Some z when not (String.is_valid_utf_8 z) ->
      err "Type.datetime: zone %S is not UTF-8" z
  | _ -> ());
  Datetime { unit_; zone }

let list t = List t

let record fields =
  let seen = Hashtbl.create (List.length fields) in
  let add (name, _) =
    if not (String.is_valid_utf_8 name) then
      err "Type.record: field name %S is not UTF-8" name;
    if Hashtbl.mem seen name then err "Type.record: duplicate field %S" name;
    Hashtbl.add seen name ()
  in
  List.iter add fields;
  Record fields

let tensor dt shape =
  if Array.length shape = 0 then err "Type.tensor: empty shape";
  Array.iter
    (fun n -> if n < 0 then err "Type.tensor: dimension %d is negative" n)
    shape;
  Tensor (dt, Iarray.of_array shape)

let ext (type a) ~name ?(metadata = "") (storage : a t) =
  if name = "" then err "Type.ext: empty name";
  if not (String.is_valid_utf_8 name) then
    err "Type.ext: name %S is not UTF-8" name;
  match storage with
  | Ext _ -> err "Type.ext: %S is stored as an extension type" name
  | _ -> Ext { name; metadata; storage }

(* Kinds *)

let rec kind : type a. a t -> a Kind.t = function
  | Bool -> Kind.Bool
  | Int8 -> Kind.Int
  | Int16 -> Kind.Int
  | Int32 -> Kind.Int
  | Int64 -> Kind.Int
  | Uint8 -> Kind.Int
  | Uint16 -> Kind.Int
  | Uint32 -> Kind.Int
  | Uint64 -> Kind.Int
  | Float16 -> Kind.Float
  | Float32 -> Kind.Float
  | Float64 -> Kind.Float
  | Decimal _ -> Kind.Decimal
  | String -> Kind.String
  | Categorical _ -> Kind.String
  | Binary -> Kind.Binary
  | Date -> Kind.Date
  | Clock _ -> Kind.Span
  | Duration _ -> Kind.Span
  | Datetime _ -> Kind.Instant
  | List e -> Kind.List (kind e)
  | Record _ -> Kind.Record
  | Tensor (dt, _) -> Kind.Tensor dt
  | Ext _ -> Kind.Ext

let rec has_ext : type a. a t -> bool = function
  | Ext _ -> true
  | List e -> has_ext e
  | Record fields -> List.exists (fun (_, Any t) -> has_ext t) fields
  | _ -> false

let rec has_float : type a. a t -> bool = function
  | Float16 | Float32 | Float64 -> true
  | List e -> has_float e
  | Record fields -> List.exists (fun (_, Any t) -> has_float t) fields
  | Tensor (dt, _) -> (
      match dt with
      | Float16 | Float32 | Float64 | BFloat16 | Float8_e4m3 | Float8_e5m2
      | Complex64 | Complex128 ->
          true
      | Int4 | UInt4 | Int8 | UInt8 | Int16 | UInt16 | Int32 | UInt32 | Int64
      | UInt64 | Bool ->
          false)
  | Ext { storage; _ } -> has_float storage
  | _ -> false

(* [storage t] is [t] with every extension it is or holds as list elements
   replaced by its storage type: the type of the values a record field of type
   [t] holds. *)
let rec storage : type a. a t -> any = function
  | Ext { storage = s; _ } -> storage s
  | List e ->
      let (Any e) = storage e in
      Any (List e)
  | t -> Any t

(* Names *)

(* A byte up to the space is a space or a control byte. *)
let is_bare_byte c =
  not
    (Char.code c <= 0x20
    || Char.code c = 0x7f
    || c = ',' || c = '[' || c = ']' || c = '"' || c = '\\')

let pp_quoted ppf s =
  let hex c = Format.fprintf ppf "\\x%02x" (Char.code c) in
  let rec loop i =
    if i < String.length s then
      let d = String.get_utf_8_uchar s i in
      if not (Uchar.utf_decode_is_valid d) then begin
        hex s.[i];
        loop (i + 1)
      end
      else begin
        (match s.[i] with
        | ('"' | '\\') as c -> Format.fprintf ppf "\\%c" c
        | c when Char.code c < 0x20 || Char.code c = 0x7f -> hex c
        | _ ->
            Format.pp_print_string ppf
              (String.sub s i (Uchar.utf_decode_length d)));
        loop (i + Uchar.utf_decode_length d)
      end
  in
  Format.pp_print_char ppf '"';
  loop 0;
  Format.pp_print_char ppf '"'

let pp_name ppf n =
  if n <> "" && String.for_all is_bare_byte n then Format.pp_print_string ppf n
  else pp_quoted ppf n

let pp_list pp ppf vs =
  let pp_sep ppf () = Format.fprintf ppf ";@ " in
  Format.fprintf ppf "@[<hov 1>[%a]@]" (Format.pp_print_list ~pp_sep pp) vs

let pp_sep ppf () = Format.pp_print_string ppf ", "

(* Values *)

let ns_per_unit = function
  | S -> 1_000_000_000L
  | Ms -> 1_000_000L
  | Us -> 1_000L
  | Ns -> 1L

let is_whole u ns = Int64.equal (Int64.rem ns (ns_per_unit u)) 0L
let ns_per_day = 86_400_000_000_000L
let in_range (lo : int) hi v = lo <= v && v <= hi

(* A finite float rounds to infinity from the midpoint between the format's
   largest finite value and the next power of two, ties going to the even
   significand: 65520 for binary16, 2^128 - 2^103 for binary32. *)
let float_holds limit v = (not (Float.is_finite v)) || Float.abs v < limit
let rec pow10 k = if k = 0 then 1L else Int64.mul 10L (pow10 (k - 1))

(* [d] is exact at [scale] in [precision] digits. Its unscaled value has at most
   18 digits, so [abs] cannot overflow, and scaling up compares against a
   smaller bound instead of multiplying. *)
let decimal_holds ~precision ~scale d =
  let u = Int64.abs (Decimal.unscaled d) and s = Decimal.scale d in
  if (s <= scale) [@mutate off "at s = scale both branches agree"] then
    Int64.compare u (pow10 (precision - (scale - s))) < 0
  else
    let p = pow10 (s - scale) in
    Int64.equal (Int64.rem u p) 0L
    && Int64.compare (Int64.div u p) (pow10 precision) < 0

let dictionary_index d =
  let index = Hashtbl.create (Iarray.length d) in
  Iarray.iteri (fun i s -> Hashtbl.add index s i) d;
  index

let names fields = Iarray.of_list (List.map fst fields)

let has_names names (r : Record.t) =
  Iarray.length names = Iarray.length r.Kind.fields
  && Iarray.for_all2 (fun n (n', _) -> String.equal n n') names r.Kind.fields

let has_shape shape v =
  let dims = Nx.shape v in
  let rec from i =
    i = Array.length dims || (dims.(i) = Iarray.get shape i && from (i + 1))
  in
  Array.length dims = Iarray.length shape && from 0

let rec holds : type a. a t -> a -> bool = function
  | Bool -> Fun.const true
  | Binary -> Fun.const true
  | Date -> Fun.const true
  | Int64 -> Fun.const true
  | Float64 -> Fun.const true
  | Int8 -> in_range (-0x80) 0x7f
  | Int16 -> in_range (-0x8000) 0x7fff
  | Int32 -> in_range (-0x8000_0000) 0x7fff_ffff
  | Uint8 -> in_range 0 0xff
  | Uint16 -> in_range 0 0xffff
  | Uint32 -> in_range 0 0xffff_ffff
  | Uint64 -> fun v -> v >= 0
  | Float16 -> float_holds 65520.
  | Float32 -> float_holds 0x1.ffffffp127
  | Decimal { precision; scale } -> decimal_holds ~precision ~scale
  | String -> String.is_valid_utf_8
  | Categorical d -> Hashtbl.mem (dictionary_index d)
  | Clock u ->
      fun d ->
        let ns = Time.Span.to_ns d in
        is_whole u ns
        && Int64.compare ns 0L >= 0
        && Int64.compare ns ns_per_day < 0
  | Duration u -> fun d -> is_whole u (Time.Span.to_ns d)
  | Datetime { unit_; _ } -> fun t -> is_whole unit_ (Time.to_ns t)
  | List e -> Array.for_all (holds e)
  | Record fields ->
      let names = names fields in
      let fields =
        Iarray.of_list (List.map (fun (_, Any t) -> holds_field t) fields)
      in
      fun r ->
        has_names names r
        && Iarray.for_all2 (fun h (_, f) -> h f) fields r.Kind.fields
  | Tensor (_, shape) -> has_shape shape
  | Ext _ -> ( function (_ : ext) -> .)

(* A field's value is that of the type with its extensions replaced by their
   storage, held in a [Storage] field exactly when the type is or contains an
   extension. *)
and holds_field : type a. a t -> Kind.field -> bool =
 fun t ->
  match storage t with Any st -> holds_stored ~ext:(Kind.has_ext (kind t)) st

(* A null field is held whatever its kind. *)
and holds_stored : type a. ext:bool -> a t -> Kind.field -> bool =
 fun ~ext t ->
  let held = holds t and k = kind t in
  let read : type b. b Kind.t -> b -> bool =
   fun k' v ->
    match Kind.equal_witness k' k with Some Equal -> held v | None -> false
  in
  function
  | Value (_, None) | Storage (_, None) -> true
  | Value (k', Some v) -> (not ext) && read k' v
  | Storage (k', Some v) -> ext && read k' v

(* OCaml rounds a double to binary32 to nearest, ties to even, but has no
   binary16 conversion that does: one through binary32 rounds twice. *)
let rec value : type a. a t -> a -> a option = function
  | Float16 -> Fun.const None
  | Float32 -> fun v -> Some (Int32.float_of_bits (Int32.bits_of_float v))
  | List e ->
      let value = value e in
      fun vs ->
        let vs = Array.map value vs in
        if Array.for_all Option.is_some vs then Some (Array.map Option.get vs)
        else None
  | Record _ as t -> if has_float t then Fun.const None else Option.some
  | _ -> Option.some

(* Order *)

let compare_float a b =
  match (Float.is_nan a, Float.is_nan b) with
  | true, true -> 0
  | true, false -> 1
  | false, true -> -1
  | false, false -> Float.compare a b

let compare_complex (a : Complex.t) (b : Complex.t) =
  match compare_float a.re b.re with 0 -> compare_float a.im b.im | c -> c

let compare_element : type a b. (a, b) Nx.dtype -> a -> a -> int = function
  | Float16 -> compare_float
  | Float32 -> compare_float
  | Float64 -> compare_float
  | BFloat16 -> compare_float
  | Float8_e4m3 -> compare_float
  | Float8_e5m2 -> compare_float
  | Int4 -> Int.compare
  | UInt4 -> Int.compare
  | Int8 -> Int.compare
  | UInt8 -> Int.compare
  | Int16 -> Int.compare
  | UInt16 -> Int.compare
  | Int32 -> Int32.compare
  | UInt32 -> Int32.unsigned_compare
  | Int64 -> Int64.compare
  | UInt64 -> Int64.unsigned_compare
  | Complex64 -> compare_complex
  | Complex128 -> compare_complex
  | Bool -> Bool.compare

let compare_arrays cmp a0 a1 =
  let n0 = Array.length a0 and n1 = Array.length a1 in
  let rec loop i =
    if i = n0 || i = n1 then Int.compare n0 n1
    else match cmp a0.(i) a1.(i) with 0 -> loop (i + 1) | c -> c
  in
  loop 0

let rec compare_value : type a. a t -> a -> a -> int = function
  | Bool -> Bool.compare
  | Int8 -> Int.compare
  | Int16 -> Int.compare
  | Int32 -> Int.compare
  | Int64 -> Int.compare
  | Uint8 -> Int.compare
  | Uint16 -> Int.compare
  | Uint32 -> Int.compare
  | Uint64 -> Int.compare
  | Float16 -> compare_float
  | Float32 -> compare_float
  | Float64 -> compare_float
  | Decimal _ -> Decimal.compare
  | String -> String.compare
  | Binary -> fun b0 b1 -> String.compare (b0 :> string) (b1 :> string)
  | Categorical d ->
      let index = dictionary_index d in
      let position s =
        match Hashtbl.find_opt index s with
        | Some i -> i
        | None -> err "Type.compare_value: %S is not in the dictionary" s
      in
      fun s0 s1 -> Int.compare (position s0) (position s1)
  | Date -> Time.Date.compare
  | Clock _ -> Time.Span.compare
  | Duration _ -> Time.Span.compare
  | Datetime _ -> Time.compare
  | List e -> compare_arrays (compare_value e)
  | Record fields ->
      let names = names fields in
      let fields =
        Array.of_list (List.map (fun (n, Any t) -> compare_field n t) fields)
      in
      let check r =
        if not (has_names names r) then
          err "Type.compare_value: the record's fields are not [%a]"
            (Format.pp_print_list ~pp_sep pp_name)
            (Iarray.to_list names)
      in
      fun r0 r1 ->
        check r0;
        check r1;
        let f0 = r0.Kind.fields and f1 = r1.Kind.fields in
        let rec loop i =
          if i = Array.length fields then 0
          else
            match
              fields.(i) (snd (Iarray.get f0 i)) (snd (Iarray.get f1 i))
            with
            | 0 -> loop (i + 1)
            | c -> c
        in
        loop 0
  | Tensor (dt, shape) ->
      (* Each comparison copies both tensors: this is the reference order for
         plan-time code, never a per-row kernel. *)
      let compare = compare_element dt in
      let elements v =
        if not (has_shape shape v) then
          err "Type.compare_value: a tensor does not have the type's shape";
        Nx.to_array v
      in
      fun v0 v1 -> compare_arrays compare (elements v0) (elements v1)
  | Ext _ -> ( fun (v : ext) _ -> match v with _ -> .)

and compare_field : type a. string -> a t -> Kind.field -> Kind.field -> int =
 fun name t ->
  match storage t with
  | Any st -> compare_stored name ~ext:(Kind.has_ext (kind t)) st

(* Null comes after every value. *)
and compare_stored : type a.
    string -> ext:bool -> a t -> Kind.field -> Kind.field -> int =
 fun name ~ext t ->
  let compare = compare_value t and k = kind t in
  let cast : type b. b Kind.t -> b -> a =
   fun k' v ->
    match Kind.equal_witness k' k with
    | Some Equal -> v
    | None ->
        err "Type.compare_value: field %S is %a, not %a" name Kind.pp k' Kind.pp
          k
  in
  let read : Kind.field -> a option = function
    | Value (_, None) | Storage (_, None) -> None
    | Value (k', Some v) when not ext -> Some (cast k' v)
    | Storage (k', Some v) when ext -> Some (cast k' v)
    | Value _ ->
        err
          "Type.compare_value: field %S has an extension type, which no plain \
           value holds"
          name
    | Storage _ ->
        err
          "Type.compare_value: field %S holds an extension's storage, but its \
           type has no extension"
          name
  in
  fun f0 f1 ->
    match (read f0, read f1) with
    | None, None -> 0
    | None, Some _ -> 1
    | Some _, None -> -1
    | Some v0, Some v1 -> compare v0 v1

(* Operands *)

let int_range : type a. a t -> (bool * int) option = function
  | Int8 -> Some (true, 8)
  | Int16 -> Some (true, 16)
  | Int32 -> Some (true, 32)
  | Int64 -> Some (true, 64)
  | Uint8 -> Some (false, 8)
  | Uint16 -> Some (false, 16)
  | Uint32 -> Some (false, 32)
  | Uint64 -> Some (false, 64)
  | _ -> None

let float_width : type a. a t -> int option = function
  | Float16 -> Some 16
  | Float32 -> Some 32
  | Float64 -> Some 64
  | _ -> None

let fineness = function S -> 0 | Ms -> 1 | Us -> 2 | Ns -> 3

let rec equal : type a b. a t -> b t -> bool =
 fun t0 t1 ->
  match (t0, t1) with
  | Bool, Bool -> true
  | Int8, Int8 -> true
  | Int16, Int16 -> true
  | Int32, Int32 -> true
  | Int64, Int64 -> true
  | Uint8, Uint8 -> true
  | Uint16, Uint16 -> true
  | Uint32, Uint32 -> true
  | Uint64, Uint64 -> true
  | Float16, Float16 -> true
  | Float32, Float32 -> true
  | Float64, Float64 -> true
  | String, String -> true
  | Binary, Binary -> true
  | Date, Date -> true
  | Decimal d0, Decimal d1 ->
      Int.equal d0.precision d1.precision && Int.equal d0.scale d1.scale
  | Categorical d0, Categorical d1 -> Iarray.equal String.equal d0 d1
  | Clock u0, Clock u1 -> u0 = u1
  | Duration u0, Duration u1 -> u0 = u1
  | Datetime d0, Datetime d1 ->
      d0.unit_ = d1.unit_ && Option.equal String.equal d0.zone d1.zone
  | List e0, List e1 -> equal e0 e1
  | Record f0, Record f1 ->
      let field (n0, Any t0) (n1, Any t1) = String.equal n0 n1 && equal t0 t1 in
      List.equal field f0 f1
  | Tensor (dt0, s0), Tensor (dt1, s1) ->
      Nx_dtype.equal dt0 dt1 && Iarray.equal Int.equal s0 s1
  | Ext e0, Ext e1 ->
      String.equal e0.name e1.name
      && String.equal e0.metadata e1.metadata
      && equal e0.storage e1.storage
  | _ -> false

let is_prefix d0 d1 =
  Iarray.length d0 <= Iarray.length d1
  && Iarray.equal String.equal d0 (Iarray.sub d1 ~pos:0 ~len:(Iarray.length d0))

(* [contains t u] is [true] iff [t] stores every value the scalar type [u]
   stores. *)
let contains : type a. a t -> a t -> bool =
 fun t u ->
  match (t, u) with
  | Decimal d1, Decimal d0 ->
      d0.scale <= d1.scale && d0.precision - d0.scale <= d1.precision - d1.scale
  | String, Categorical _ -> true
  | Categorical d1, Categorical d0 -> is_prefix d0 d1
  | Clock u1, Clock u0 -> fineness u0 <= fineness u1
  | _ -> (
      match (int_range t, int_range u, float_width t, float_width u) with
      | Some (st, wt), Some (su, wu), _, _ ->
          if Bool.equal st su then wu <= wt else st && wu < wt
      | _, _, Some wt, Some wu -> wu <= wt
      | _ -> equal t u)

let all_some l =
  List.fold_right
    (fun x acc ->
      match (x, acc) with Some x, Some l -> Some (x :: l) | _ -> None)
    l (Some [])

let rec transpose = function
  | [] | [] :: _ -> []
  | rows -> List.map List.hd rows :: transpose (List.map List.tl rows)

let rec common : type a. a t list -> a t option = function
  | [] -> None
  | t :: _ as ts -> (
      match t with
      | List _ -> common_lists ts
      | Record _ -> common_records ts
      | _ -> List.find_opt (fun t -> List.for_all (contains t) ts) ts)

(* [common] dispatches on the head of [ts], so [ts] is not empty. The indices
   are abstract, which hides from the checker that only [List] has an array
   index and only [Record] the index [Record.t]. *)
and common_lists : type a. a array t list -> a array t option =
 fun ts ->
  let element : a array t -> a t = function List e -> e | _ -> assert false in
  Option.map list (common (List.map element ts))

and common_records : Record.t t list -> Record.t t option =
 fun ts ->
  let fields : Record.t t -> (string * any) list = function
    | Record fields -> fields
    | _ -> assert false
  in
  let all = List.map fields ts in
  let names = List.map fst (List.hd all) in
  let same_names f = List.equal String.equal names (List.map fst f) in
  if not (List.for_all same_names all) then None
  else
    let columns = transpose (List.map (List.map snd) all) in
    Option.map
      (fun ts -> Record (List.combine names ts))
      (all_some (List.map common_any columns))

and common_any = function [] -> None | Any t :: _ as ts -> common_at t ts

and common_at : type a. a t -> any list -> any option =
 fun t ts ->
  let k = kind t in
  let cast (Any u) : a t option =
    match Kind.equal_witness (kind u) k with
    | Some Equal -> Some u
    | None -> None
  in
  match all_some (List.map cast ts) with
  | None -> None
  | Some ts -> Option.map (fun t -> Any t) (common ts)

(* Formatting *)

let unit_name = function S -> "s" | Ms -> "ms" | Us -> "us" | Ns -> "ns"

let rec pp : type a. Format.formatter -> a t -> unit =
 fun ppf t ->
  let str = Format.pp_print_string ppf in
  match t with
  | Bool -> str "bool"
  | Int8 -> str "int8"
  | Int16 -> str "int16"
  | Int32 -> str "int32"
  | Int64 -> str "int64"
  | Uint8 -> str "uint8"
  | Uint16 -> str "uint16"
  | Uint32 -> str "uint32"
  | Uint64 -> str "uint64"
  | Float16 -> str "float16"
  | Float32 -> str "float32"
  | Float64 -> str "float64"
  | String -> str "string"
  | Binary -> str "binary"
  | Date -> str "date"
  | Decimal { precision; scale } ->
      Format.fprintf ppf "decimal[%d, %d]" precision scale
  | Categorical d ->
      let n = Iarray.length d in
      let shown = Iarray.to_list (Iarray.sub d ~pos:0 ~len:(min n 8)) in
      Format.fprintf ppf "categorical[%a"
        (Format.pp_print_list ~pp_sep pp_quoted)
        shown;
      if n > 8 then Format.fprintf ppf ", … %d" n;
      str "]"
  | Clock u -> Format.fprintf ppf "clock[%s]" (unit_name u)
  | Duration u -> Format.fprintf ppf "duration[%s]" (unit_name u)
  | Datetime { unit_; zone = None } ->
      Format.fprintf ppf "datetime[%s]" (unit_name unit_)
  | Datetime { unit_; zone = Some z } ->
      Format.fprintf ppf "datetime[%s, %a]" (unit_name unit_) pp_name z
  | List e -> Format.fprintf ppf "list[%a]" pp e
  | Record fields ->
      let field ppf (n, Any t) = Format.fprintf ppf "%a %a" pp_name n pp t in
      Format.fprintf ppf "record[%a]"
        (Format.pp_print_list ~pp_sep field)
        fields
  | Tensor (dt, shape) ->
      let dims = Iarray.to_list (Iarray.map string_of_int shape) in
      Format.fprintf ppf "tensor[%s, %s]" (Nx_dtype.to_string dt)
        (String.concat "×" dims)
  | Ext { name; metadata; storage } ->
      Format.fprintf ppf "ext[%a" pp_name name;
      if metadata <> "" then Format.fprintf ppf " %a" pp_quoted metadata;
      Format.fprintf ppf ", %a]" pp storage
