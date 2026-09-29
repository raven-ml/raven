open Windtrap
open Tolk_next

let reprs =
  Dtype.
    [
      (Void, "dtypes.void");
      (Weak_int, "dtypes.weakint");
      (Bool, "dtypes.bool");
      (Int8, "dtypes.char");
      (Uint8, "dtypes.uchar");
      (Int16, "dtypes.short");
      (Uint16, "dtypes.ushort");
      (Int32, "dtypes.int");
      (Uint32, "dtypes.uint");
      (Int64, "dtypes.long");
      (Uint64, "dtypes.ulong");
      (Weak_float, "dtypes.weakfloat");
      (Fp8e4m3, "dtypes.fp8e4m3");
      (Fp8e5m2, "dtypes.fp8e5m2");
      (Fp8e4m3fnuz, "dtypes.fp8e4m3fnuz");
      (Fp8e5m2fnuz, "dtypes.fp8e5m2fnuz");
      (Float16, "dtypes.half");
      (Bfloat16, "dtypes.bfloat16");
      (Float32, "dtypes.float");
      (Float64, "dtypes.double");
    ]

(* Witnesses *)

let declared = List.map fst reprs
let pp_dtype ppf dt = Format.pp_print_string ppf (List.assoc dt reprs)

let alias dt =
  let repr = List.assoc dt reprs in
  String.sub repr 7 (String.length repr - 7)

let dtype =
  Testable.with_compare Dtype.compare (Testable.make ~pp:pp_dtype ~equal:( = ))

let pp_value ppf = function
  | `Bool b -> Format.pp_print_string ppf (if b then "True" else "False")
  | `Int n -> Z.pp_print ppf n
  | `Float f -> Testable.pp float_exact ppf f

let equal_value v0 v1 =
  match (v0, v1) with
  | `Bool b0, `Bool b1 -> Bool.equal b0 b1
  | `Int n0, `Int n1 -> Z.equal n0 n1
  | `Float f0, `Float f1 ->
      Int64.equal (Int64.bits_of_float f0) (Int64.bits_of_float f1)
      || (Float.is_nan f0 && Float.is_nan f1)
  | _ -> false

let value = Testable.make ~pp:pp_value ~equal:equal_value

let pp_const ppf = function
  | `Invalid -> Format.pp_print_string ppf "Invalid"
  | #Dtype.value as v -> pp_value ppf v

let equal_const c0 c1 =
  match (c0, c1) with
  | `Invalid, `Invalid -> true
  | (#Dtype.value as v0), (#Dtype.value as v1) -> equal_value v0 v1
  | _ -> false

let const = Testable.make ~pp:pp_const ~equal:equal_const
let z = Testable.make ~pp:Z.pp_print ~equal:Z.equal

(* Bounds *)

let int_bits =
  Dtype.
    [
      (Bool, (1, false));
      (Int8, (8, true));
      (Uint8, (8, false));
      (Int16, (16, true));
      (Uint16, (16, false));
      (Int32, (32, true));
      (Uint32, (32, false));
      (Int64, (64, true));
      (Uint64, (64, false));
      (Weak_int, (800, true));
    ]

let int_bounds dt =
  match List.assoc_opt dt int_bits with
  | Some (bits, true) ->
      let half = Z.shift_left Z.one (bits - 1) in
      (Z.neg half, Z.pred half)
  | Some (bits, false) when dt <> Dtype.Bool ->
      (Z.zero, Z.pred (Z.shift_left Z.one bits))
  | _ -> invalid_arg (Format.asprintf "%a is not an integer" pp_dtype dt)

(* Generators *)

let every = Gen.of_list ~pp:pp_dtype declared
let promotable = Gen.of_list ~pp:pp_dtype (List.tl declared)

let stored =
  Gen.of_list ~pp:pp_dtype
    (List.filter
       (fun dt -> not (List.mem dt Dtype.[ Void; Weak_int; Weak_float ]))
       declared)

let edges =
  List.concat_map
    (fun (dt, _) ->
      let lo, hi = int_bounds dt in
      [ Z.pred lo; lo; Z.succ lo; Z.pred hi; hi; Z.succ hi ])
    (List.remove_assoc Dtype.Bool int_bits)

let integer =
  let scaled =
    Gen.map
      (fun (n, shift) -> Z.shift_left (Z.of_int64 n) shift)
      (Gen.pair Gen.int64 (Gen.int_range 0 1036))
  in
  Gen.with_pp Z.pp_print
    (Gen.frequency
       [
         (2, Gen.map Z.of_int (Gen.int_range (-300) 300));
         (2, Gen.of_list edges);
         (3, Gen.map Z.of_int64 Gen.int64);
         (3, scaled);
       ])

let integer_in (lo, hi) =
  let within n = Z.add lo (Z.erem n (Z.succ (Z.sub hi lo))) in
  Gen.frequency
    [
      (1, Gen.of_list [ lo; Z.succ lo; Z.pred hi; hi ]);
      (1, Gen.of_list (List.filter (fun n -> Z.leq lo n && Z.leq n hi) edges));
      (4, Gen.map within integer);
    ]

let finite_float =
  Gen.frequency
    [
      (2, Gen.such_that Float.is_finite Gen.float);
      (2, Gen.float_range (-1e5) 1e5);
      (2, Gen.float_range (-2.) 2.);
      (1, Gen.float_range (-1e-3) 1e-3);
    ]

let float = Gen.frequency [ (1, Gen.any_float); (3, finite_float) ]

let value_of dt =
  Gen.with_pp pp_value
    (match dt with
    | Dtype.Void -> invalid_arg "Dtypes.value_of: void has no value"
    | Bool -> Gen.map (fun b -> `Bool b) Gen.bool
    | Weak_float -> Gen.map (fun f -> `Float f) float
    | dt when List.mem_assoc dt int_bits ->
        Gen.map (fun n -> `Int n) (integer_in (int_bounds dt))
    | dt -> Gen.map (fun f -> Dtype.truncate dt (`Float f)) float)

(* Golden cells *)

let of_cell s =
  match List.find_opt (fun (_, repr) -> String.equal repr s) reprs with
  | Some (dt, _) -> dt
  | None -> invalid_arg (Printf.sprintf "%S is not a data type" s)

let is_integer s =
  let digits = if String.starts_with ~prefix:"-" s then 1 else 0 in
  String.length s > digits
  && String.for_all
       (function '0' .. '9' -> true | _ -> false)
       (String.sub s digits (String.length s - digits))

let value_of_cell = function
  | "True" -> `Bool true
  | "False" -> `Bool false
  | "-nan" -> `Float (Float.neg Float.nan)
  | s when is_integer s -> `Int (Z.of_string s)
  | s -> (
      match float_of_string_opt s with
      | Some f -> `Float f
      | None -> invalid_arg (Printf.sprintf "%S is not a value" s))

let const_of_cell = function
  | "Invalid" -> `Invalid
  | s -> (value_of_cell s :> Dtype.const)

let consts_of_cell s =
  let n = String.length s in
  if n < 2 || s.[0] <> '[' || s.[n - 1] <> ']' then
    invalid_arg (Printf.sprintf "%S is not a list" s);
  match String.sub s 1 (n - 2) with
  | "" -> []
  | items ->
      List.map const_of_cell
        (String.split_on_char ',' items |> List.map String.trim)
