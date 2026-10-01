(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon_next
open Windtrap

let ( let+ ) = Gen.( let+ )
let ( and+ ) = Gen.( and+ )
let edges pp lo hi g = Gen.frequency [ (5, g); (1, Gen.of_list ~pp [ lo; hi ]) ]

(* Types *)

let pp_any ppf (Type.Any t) = Type.pp ppf t
let ext t = Type.Any (Type.ext ~name:"units.mass" ~metadata:"kg" t)

let scalars =
  Type.
    [
      Any bool;
      Any int8;
      Any int16;
      Any int32;
      Any int64;
      Any uint8;
      Any uint16;
      Any uint32;
      Any uint64;
      Any float16;
      Any float32;
      Any float64;
      Any (decimal ~precision:5 ~scale:2);
      Any (decimal ~precision:18 ~scale:0);
      Any (decimal ~precision:18 ~scale:18);
      Any string;
      Any binary;
      Any (categorical [| "a"; "é"; "" |]);
      Any (categorical [||]);
      Any date;
      Any (clock S);
      Any (clock Ns);
      Any (duration Ms);
      Any (datetime Ns);
      Any (datetime ~zone:"UTC" S);
      Any (tensor Nx.float32 [| 2 |]);
      Any (tensor Nx.int8 [| 2; 0 |]);
      Any (tensor Nx.bool [| 1; 2 |]);
    ]

let names = [ "a"; "b"; "é"; "" ]
let scalar = Gen.of_list ~pp:pp_any scalars
let ext_scalar = Gen.map (fun (Type.Any t) -> ext t) scalar

(* An extension is drawn as often at the bottom as at the top, so that a record
   two deep can hold a record with an extension field. *)
let rec nested depth =
  let ext = ext_scalar in
  if depth = 0 then Gen.frequency [ (3, scalar); (1, ext) ]
  else
    let inner = nested (depth - 1) in
    let record n =
      let+ ts = Gen.list ~size:(Gen.int_range n n) inner in
      Type.Any
        (Type.record (List.combine (List.filteri (fun i _ -> i < n) names) ts))
    in
    Gen.frequency
      [
        (4, scalar);
        (2, Gen.map (fun (Type.Any t) -> Type.Any (Type.list t)) inner);
        (3, Gen.bind (Gen.int_range 0 3) record);
        (1, ext);
      ]

(* A record of a record with an extension field over [t]. *)
let ext_in_record (Type.Any t) =
  let (Type.Any e) = ext t in
  Type.Any (Type.record [ ("r", Any (Type.record [ ("m", Any e) ])) ])

(* Text, an extension and a record of a record with an extension field are rare
   in [nested 2], and suites cover them: each is also drawn directly, in one
   case of seven. *)
let type_ =
  Gen.with_pp pp_any
    (Gen.frequency
       [
         (4, nested 2);
         (1, Gen.constant (Type.Any Type.string));
         (1, ext_scalar);
         (1, Gen.map ext_in_record scalar);
       ])

(* Values *)

let int_in lo hi = edges Format.pp_print_int lo hi (Gen.int_range lo hi)

(* The binary16 value of the bits [b]. *)
let half b =
  let sign = if b land 0x8000 = 0 then 1. else -1. in
  let e = (b lsr 10) land 0x1f and m = b land 0x3ff in
  if e = 0x1f then if m = 0 then sign *. Float.infinity else Float.nan
  else if e = 0 then sign *. Float.ldexp (Float.of_int m) (-24)
  else sign *. Float.ldexp (Float.of_int (m + 0x400)) (e - 25)

let ns_per = function
  | Type.S -> 1_000_000_000L
  | Ms -> 1_000_000L
  | Us -> 1_000L
  | Ns -> 1L

(* The nanoseconds of whole ticks of [u] in [lo, hi] ticks. *)
let ticks u lo hi =
  let+ t =
    edges (fun ppf -> Format.fprintf ppf "%LdL") lo hi (Gen.int64_range lo hi)
  in
  Int64.mul t (ns_per u)

let all_ticks u =
  ticks u
    (Int64.div Int64.min_int (ns_per u))
    (Int64.div Int64.max_int (ns_per u))

let text =
  let piece = Gen.of_list [ ""; "a"; "é"; "日本"; "\000"; "𝄞"; "," ] in
  Gen.map (String.concat "") (Gen.list ~size:(Gen.int_range 0 3) piece)

let rec all = function
  | [] -> Gen.constant []
  | g :: gs ->
      let+ x = g and+ xs = all gs in
      x :: xs

let rec value : type a. a Type.t -> a Gen.t option = function
  | Bool -> Some Gen.bool
  | Int8 -> Some (int_in (-0x80) 0x7f)
  | Int16 -> Some (int_in (-0x8000) 0x7fff)
  | Int32 -> Some (int_in (-0x8000_0000) 0x7fff_ffff)
  | Int64 -> Some Gen.int
  | Uint8 -> Some (int_in 0 0xff)
  | Uint16 -> Some (int_in 0 0xffff)
  | Uint32 -> Some (int_in 0 0xffff_ffff)
  | Uint64 -> Some (int_in 0 max_int)
  | Float16 -> Some (Gen.map half (Gen.int_range 0 0xffff))
  | Float32 -> Some (Gen.map Int32.float_of_bits Gen.int32)
  | Float64 -> Some (edges Format.pp_print_float (-0.) Float.nan Gen.any_float)
  | Decimal { precision; scale } ->
      let max = Int64.pred (Int64.of_float (10. ** Float.of_int precision)) in
      let unscaled = Gen.int64_range (Int64.neg max) max in
      Some (Gen.map (fun unscaled -> Decimal.v ~unscaled ~scale) unscaled)
  | String -> Some text
  | Binary -> Some (Gen.map Binary.of_string Gen.string)
  | Categorical d when Iarray.length d = 0 -> None
  | Categorical d -> Some (Gen.of_list (Iarray.to_list d))
  | Date ->
      let lo = Int32.(to_int min_int) and hi = Int32.(to_int max_int) in
      let date n = Option.get (Time.Date.of_days n) in
      Some (Gen.map date (int_in lo hi))
  | Clock u ->
      let last = Int64.pred (Int64.div 86_400_000_000_000L (ns_per u)) in
      Some (Gen.map Time.Span.of_ns (ticks u 0L last))
  | Duration u -> Some (Gen.map Time.Span.of_ns (all_ticks u))
  | Datetime { unit_; _ } -> Some (Gen.map Time.of_ns (all_ticks unit_))
  | List e -> (
      match value e with
      | Some g -> Some (Gen.array ~size:(Gen.int_range 0 3) g)
      | None -> Some (Gen.constant [||]))
  | Record fields ->
      let field (name, Type.Any t) =
        let k = Type.kind t in
        let add v r = Record.add k name v r in
        match (Kind.provably_equal k k, value t) with
        | None, _ -> None
        | Some _, Some g -> Some (Gen.map add (Gen.option g))
        | Some _, None -> Some (Gen.constant (add None))
      in
      let fields = List.map field fields in
      if List.mem None fields then None
      else
        let build adds = List.fold_left ( |> ) Record.empty adds in
        Some (Gen.map build (all (List.map Option.get fields)))
  | Tensor (dt, shape) ->
      let n = Iarray.fold_left ( * ) 1 shape in
      let tensor xs =
        Nx.cast dt
          (Nx.create Nx.float64 (Iarray.to_array shape) (Array.of_list xs))
      in
      Some
        (Gen.map tensor
           (Gen.list ~size:(Gen.int_range n n)
              (Gen.map Float.of_int Gen.small_int)))
  | Ext _ -> None

let options ty =
  let row =
    match value ty with Some g -> Gen.option g | None -> Gen.constant None
  in
  let size =
    Gen.frequency
      [ (1, Gen.constant 0); (1, Gen.int_range 1 3); (3, Gen.int_range 0 40) ]
  in
  Gen.array ~size row

(* Tables *)

let pp_batches ppf t =
  let pp_sep ppf () = Format.pp_print_string ppf " + " in
  Format.fprintf ppf "batches of %a rows"
    (Format.pp_print_list ~pp_sep Format.pp_print_int)
    (List.map rows (batches t))

let split t =
  let n = rows t in
  let run a b = take (Nx.arange Nx.int64 a b 1) t in
  let rec runs = function
    | a :: (b :: _ as cuts) -> run a b :: runs cuts
    | _ -> []
  in
  let cut cuts =
    of_batches (runs (List.sort Int.compare ((0 :: cuts) @ [ n ])))
  in
  Gen.with_pp pp_batches
    (Gen.map cut (Gen.list ~size:(Gen.int_range 0 5) (Gen.int_range 0 n)))

(* Comparing and printing *)

let same_float a b =
  Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)
  || (Float.is_nan a && Float.is_nan b)

let rec equal_value : type a. a Type.t -> a -> a -> bool =
 fun ty a b ->
  match ty with
  | Float16 -> same_float a b
  | Float32 -> same_float a b
  | Float64 -> same_float a b
  | List e ->
      Array.length a = Array.length b && Array.for_all2 (equal_value e) a b
  | Record fields ->
      let field (name, Type.Any t) =
        let k = Type.kind t in
        Option.equal (equal_value t) (Record.field k name a)
          (Record.field k name b)
      in
      Record.names a = Record.names b && List.for_all field fields
  | _ -> Type.compare_value ty a b = 0

let pp_sep ppf () = Format.fprintf ppf ";@ "

let rec pp_value : type a. a Type.t -> Format.formatter -> a -> unit =
 fun ty ppf v ->
  let int = Format.pp_print_int and float ppf = Format.fprintf ppf "%h" in
  match ty with
  | Bool -> Format.pp_print_bool ppf v
  | Int8 -> int ppf v
  | Int16 -> int ppf v
  | Int32 -> int ppf v
  | Int64 -> int ppf v
  | Uint8 -> int ppf v
  | Uint16 -> int ppf v
  | Uint32 -> int ppf v
  | Uint64 -> int ppf v
  | Float16 -> float ppf v
  | Float32 -> float ppf v
  | Float64 -> float ppf v
  | Decimal _ -> Decimal.pp ppf v
  | String -> Type.pp_quoted ppf v
  | Categorical _ -> Type.pp_quoted ppf v
  | Binary -> Binary.pp ppf v
  | Date -> Time.Date.pp ppf v
  | Clock _ -> Time.Span.pp ppf v
  | Duration _ -> Time.Span.pp ppf v
  | Datetime _ -> Time.pp ppf v
  | Tensor _ -> Nx.pp ppf v
  | List e ->
      Format.fprintf ppf "@[<hov 1>[|%a|]@]"
        (Format.pp_print_array ~pp_sep (pp_value e))
        v
  | Record fields ->
      let field ppf (name, Type.Any t) =
        Format.fprintf ppf "%a = %a" Type.pp_name name (pp_option t)
          (Record.field (Type.kind t) name v)
      in
      Format.fprintf ppf "@[<hov 1>{%a}@]"
        (Format.pp_print_list ~pp_sep field)
        fields
  | Ext _ -> Format.pp_print_string ppf "<ext>"

and pp_option : type a. a Type.t -> Format.formatter -> a option -> unit =
 fun ty ppf -> function
  | None -> Format.pp_print_string ppf "None"
  | Some v -> Format.fprintf ppf "Some %a" (pp_value ty) v

let witness ty = Testable.make ~pp:(pp_value ty) ~equal:(equal_value ty)

type sample = Sample : 'a Type.t * 'a option array -> sample

let pp_sample ppf (Sample (ty, vs)) =
  Format.fprintf ppf "@[<hov 2>%a:@ [|%a|]@]" Type.pp ty
    (Format.pp_print_array ~pp_sep (pp_option ty))
    vs

let sample =
  Gen.with_pp pp_sample
    (Gen.bind type_ (fun (Type.Any ty) ->
         Gen.map (fun vs -> Sample (ty, vs)) (options ty)))
