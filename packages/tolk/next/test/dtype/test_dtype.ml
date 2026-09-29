open Windtrap
open Tolk_next
open Dtypes

let lub2 a b = Dtype.least_upper [ a; b ]
let promotes a b = Dtype.equal (lub2 a b) b
let rejects f = raises_match (Exn.invalid_arg ?substring:None) f
let with_setting setting name f = Helpers.context [ B (setting, name) ] f

let bool_cell = function
  | "True" -> true
  | "False" -> false
  | s -> invalid_arg s

let fmt_cell = function "None" -> None | s -> Some s.[0]
let dtype_list dts = Gen.of_list ~pp:(Testable.pp dtype) dts

let as_float = function
  | `Float f -> f
  | v -> failf "%a is not a float" (Testable.pp value) v

(* [expect w read cell f] is that [f ()] is what [cell] reads as, or that it
   rejects its argument where tinygrad raises. *)
let raised cell = String.starts_with ~prefix:"raises " cell

let expect w read cell f =
  if raised cell then rejects f else equal w (read cell) (f ())

(* Constants *)

let nans =
  [
    Float.nan;
    Float.neg Float.nan;
    Int64.float_of_bits 0x7FF0_0000_0000_0001L;
    Int64.float_of_bits 0xFFF8_0000_DEAD_BEEFL;
  ]

let any_const =
  Gen.with_pp (Testable.pp const)
    (Gen.frequency
       [
         (1, Gen.constant `Invalid);
         (1, Gen.map (fun b -> `Bool b) Gen.bool);
         (3, Gen.map (fun n -> `Int n) integer);
         (2, Gen.map (fun f -> `Float f) Gen.any_float);
         (1, Gen.map (fun f -> `Float f) (Gen.of_list (0. :: -0. :: nans)));
       ])

let const_equality =
  Testable.make ~pp:(Testable.pp const) ~equal:Dtype.equal_const

let constants =
  group "constants"
    [
      test "every NaN is the same constant" (fun () ->
          List.iter
            (fun f -> equal const_equality (`Float Float.nan) (`Float f))
            nans);
      test "zero and negative zero are different constants" (fun () ->
          not_equal const_equality (`Float 0.) (`Float (-0.)));
      test "constants of different kinds differ" (fun () ->
          not_equal const_equality (`Int Z.one) (`Float 1.);
          not_equal const_equality (`Bool true) (`Int Z.one);
          not_equal const_equality `Invalid (`Bool false));
      prop "equal_const is an equivalence"
        (Gen.pair any_const any_const)
        (Law.equivalence const_equality);
      prop "equal_const is the witness's equality"
        (Gen.pair any_const any_const) (fun (c0, c1) ->
          equal bool (Testable.equal const c0 c1) (Dtype.equal_const c0 c1));
      test "every NaN hashes alike" (fun () ->
          List.iter
            (fun f ->
              equal int
                (Dtype.hash_const (`Float Float.nan))
                (Dtype.hash_const (`Float f)))
            nans);
      Golden.cases "const_repr.golden" (fun cell ->
          equal string (cell "printed")
            (Format.asprintf "%a" Dtype.pp_const (const_of_cell (cell "const"))));
    ]

let address_spaces =
  cases "address spaces print as their enum member" ~name:snd
    Dtype.
      [
        (Global, "AddrSpace.GLOBAL");
        (Local, "AddrSpace.LOCAL");
        (Reg, "AddrSpace.REG");
        (Alu, "AddrSpace.ALU");
      ]
    (fun (space, printed) ->
      equal string printed (Format.asprintf "%a" Dtype.pp_addr_space space))

(* Data types *)

let properties =
  Golden.cases "properties.golden" (fun cell ->
      let dt = of_cell (cell "dtype") in
      equal string (cell "dtype") (Format.asprintf "%a" Dtype.pp dt);
      equal int (int_of_string (cell "priority")) (Dtype.priority dt);
      equal int (int_of_string (cell "bitsize")) (Dtype.bitsize dt);
      equal int (int_of_string (cell "itemsize")) (Dtype.itemsize dt);
      equal string (cell "name") (Dtype.name dt);
      equal (option char) (fmt_cell (cell "fmt")) (Dtype.fmt dt);
      equal value (value_of_cell (cell "min")) (Dtype.min dt);
      equal value (value_of_cell (cell "max")) (Dtype.max dt);
      equal bool (bool_cell (cell "is_int")) (Dtype.is_int dt);
      equal bool (bool_cell (cell "is_float")) (Dtype.is_float dt);
      equal bool (bool_cell (cell "is_unsigned")) (Dtype.is_unsigned dt);
      equal bool (bool_cell (cell "is_bool")) (Dtype.is_bool dt))

let groups =
  Golden.cases "groups.golden" (fun cell ->
      let group =
        match cell "group" with
        | "fp8_ocp" -> Dtype.fp8_ocp
        | "fp8_fnuz" -> Dtype.fp8_fnuz
        | "fp8s" -> Dtype.fp8s
        | "floats" -> Dtype.floats
        | "uints" -> Dtype.uints
        | "sints" -> Dtype.sints
        | "ints" -> Dtype.ints
        | "weaks" -> Dtype.weaks
        | "all" -> Dtype.all
        | name -> invalid_arg name
      in
      equal (list dtype)
        (List.map of_cell (String.split_on_char ' ' (cell "members")))
        group)

(* The rows of properties.golden are tinygrad's sorted data types. *)
let sorted =
  List.map
    (fun cell -> of_cell (cell "dtype"))
    (Golden.rows "properties.golden")

let data_types =
  group "data types"
    [
      properties;
      groups;
      test "compare sorts as tinygrad does" (fun () ->
          equal (list dtype) sorted (List.sort Dtype.compare (List.rev sorted)));
      prop "compare is a total order"
        (Gen.triple every every every)
        (Law.order dtype);
      prop "equal data types hash alike" (Gen.pair every every) (fun (a, b) ->
          cover "equal" (Dtype.equal a b);
          if Dtype.equal a b then equal int (Dtype.hash a) (Dtype.hash b));
      prop "equal is structural equality" (Gen.pair every every) (fun (a, b) ->
          equal bool (a = b) (Dtype.equal a b));
      Golden.cases "finfo.golden" (fun cell ->
          equal (pair int int)
            (int_of_string (cell "exponent"), int_of_string (cell "mantissa"))
            (Dtype.finfo (of_cell (cell "dtype"))));
      cases "finfo rejects every data type but the floats of known width"
        ~name:alias
        (List.filter (fun dt -> not (List.mem dt Dtype.floats)) declared)
        (fun dt -> rejects (fun () -> Dtype.finfo dt));
    ]

(* Names and defaults *)

let names =
  group "names"
    [
      Golden.cases "names.golden" (fun cell ->
          equal (result dtype string)
            (Ok (of_cell (cell "dtype")))
            (Dtype.of_string (cell "name")));
      test "a name is read in any case" (fun () ->
          equal (result dtype string) (Ok Dtype.Float16)
            (Dtype.of_string "HALF");
          equal (result dtype string) (Ok Dtype.Float32)
            (Dtype.of_string "Float32");
          equal (result dtype string) (Ok Dtype.Uint64)
            (Dtype.of_string "ULong"));
      cases "the error names what is not a data type"
        ~name:(Printf.sprintf "%S")
        [
          "nonexistdtype";
          "f32";
          "bf16";
          " half ";
          "uint128";
          "floats";
          "is_float";
          "dtypes";
          "dtypes.floats";
        ] (fun s -> contains ~sub:s (require_error (Dtype.of_string s)));
      test "the empty string is not a data type" (fun () ->
          is_error (Dtype.of_string ""));
      prop "of_string reads what pp prints" every
        (Law.round_trip dtype string (Format.asprintf "%a" Dtype.pp) (fun s ->
             require_ok (Dtype.of_string s)));
    ]

let defaults =
  group "defaults"
    [
      cases "default_float is the float DEFAULT_FLOAT names" ~name:alias
        Dtype.floats (fun dt ->
          equal dtype dt
            (with_setting Helpers.default_float (alias dt) Dtype.default_float));
      test "DEFAULT_FLOAT is read in any case" (fun () ->
          equal dtype Dtype.Float16
            (with_setting Helpers.default_float "HALF" Dtype.default_float));
      cases "default_int is the integer DEFAULT_INT names" ~name:alias
        Dtype.ints (fun dt ->
          equal dtype dt
            (with_setting Helpers.default_int (alias dt) Dtype.default_int));
      cases "default_float rejects what is not a float of known width"
        ~name:Fun.id [ "int32"; "weakfloat"; "bool"; "void"; "typo" ]
        (fun name ->
          rejects (fun () ->
              with_setting Helpers.default_float name Dtype.default_float));
      cases "default_int rejects what is not an integer of known width"
        ~name:Fun.id [ "float32"; "weakint"; "bool"; "void"; "typo" ]
        (fun name ->
          rejects (fun () ->
              with_setting Helpers.default_int name Dtype.default_int));
      test "strong commits the weak data types at the current defaults"
        (fun () ->
          with_setting Helpers.default_int "int64" (fun () ->
              equal dtype Dtype.Int64 (Dtype.strong Dtype.Weak_int));
          with_setting Helpers.default_float "float16" (fun () ->
              equal dtype Dtype.Float16 (Dtype.strong Dtype.Weak_float)));
      Golden.cases "projections.golden" (fun cell ->
          let dt = of_cell (cell "dtype") in
          equal dtype (of_cell (cell "weak")) (Dtype.weak dt);
          equal dtype (of_cell (cell "strong")) (Dtype.strong dt);
          expect dtype of_cell (cell "least_upper_float") (fun () ->
              Dtype.least_upper_float dt);
          expect dtype of_cell (cell "sum_acc") (fun () -> Dtype.sum_acc dt);
          equal (option char)
            (fmt_cell (cell "storage_fmt"))
            (Dtype.storage_fmt dt));
    ]

(* Literals *)

let small_const =
  Gen.with_pp (Testable.pp const)
    (Gen.frequency
       [
         (1, Gen.constant `Invalid);
         (1, Gen.map (fun b -> `Bool b) Gen.bool);
         (3, Gen.map (fun n -> `Int (Z.of_int n)) (Gen.int_range (-1000) 1000));
         (2, Gen.map (fun f -> `Float f) Gen.any_float);
       ])

let commit_bounds =
  Gen.map
    (fun (a, b) -> (Z.min a b, Z.max a b))
    (Gen.such_that (fun (a, b) -> not (Z.equal a b)) (Gen.pair integer integer))

let literals =
  group "literals"
    [
      Golden.cases "of_const.golden" (fun cell ->
          equal dtype
            (of_cell (cell "dtype"))
            (Dtype.of_const (const_of_cell (cell "value"))));
      Golden.cases "of_consts.golden" (fun cell ->
          expect dtype of_cell (cell "dtype") (fun () ->
              Dtype.of_consts (consts_of_cell (cell "consts"))));
      prop "of_consts commits: its data type is never weak"
        (Gen.list small_const) (fun cs ->
          is_false (List.mem (Dtype.of_consts cs) Dtype.weaks));
      prop "of_consts ignores the order of the literals" (Gen.list small_const)
        (fun cs ->
          equal dtype (Dtype.of_consts cs) (Dtype.of_consts (List.rev cs)));
      Golden.cases "commit.golden" ~key:[ "lo"; "hi"; "default_int" ]
        (fun cell ->
          let lo = Z.of_string (cell "lo") and hi = Z.of_string (cell "hi") in
          let default_int =
            match cell "default_int" with
            | "None" -> None
            | s -> Some (of_cell s)
          in
          expect dtype of_cell (cell "dtype") (fun () ->
              Dtype.commit_int ?default_int lo hi));
      prop "commit_int holds its bounds, or falls back to int64" commit_bounds
        (fun (lo, hi) ->
          let dt = Dtype.commit_int lo hi in
          let min, max = int_bounds dt in
          let holds = Z.leq min lo && Z.leq hi max in
          cover "holds its bounds" holds;
          cover "falls back" (not holds);
          if not holds then equal dtype Dtype.Int64 dt);
    ]

(* Promotion *)

(* The least data type, by compare, that every one of [dts] promotes to. *)
let least_common_bound dts =
  List.filter
    (fun u -> List.for_all (fun d -> promotes d u) dts)
    (List.tl declared)
  |> List.sort Dtype.compare |> List.hd

let fold_bound dts = List.fold_left lub2 (List.hd dts) dts
let fp8_triple = Dtype.[ Fp8e4m3; Fp8e5m2; Bfloat16 ]

let promotion =
  group "promotion"
    [
      Golden.cases "least_upper.golden" ~key:[ "a"; "b" ] (fun cell ->
          equal dtype
            (of_cell (cell "least_upper"))
            (lub2 (of_cell (cell "a")) (of_cell (cell "b"))));
      Golden.cases "least_upper_triples.golden" ~key:[ "a"; "b"; "c" ]
        (fun cell ->
          equal dtype
            (of_cell (cell "least_upper"))
            (Dtype.least_upper
               (List.map (fun c -> of_cell (cell c)) [ "a"; "b"; "c" ])));
      test "least_upper is not a fold of pairwise bounds" (fun () ->
          equal dtype Dtype.Bfloat16 (Dtype.least_upper fp8_triple);
          equal dtype Dtype.Float32 (fold_bound fp8_triple));
      prop "least_upper is commutative"
        (Gen.pair promotable promotable)
        (Law.commutative dtype lub2);
      prop "a data type is its own least upper bound" promotable (fun dt ->
          equal dtype dt (Dtype.least_upper [ dt ]);
          equal dtype dt (Dtype.least_upper [ dt; dt ]);
          equal dtype dt (Dtype.least_upper [ dt; dt; dt ]));
      prop "bool is the bottom of the lattice" promotable
        (Law.neutral dtype lub2 Dtype.Bool);
      prop "float64 is the top of the lattice" promotable
        (Law.absorbing dtype lub2 Dtype.Float64);
      prop "least_upper is at least each of its inputs"
        (Gen.pair promotable promotable) (fun (a, b) ->
          at_least dtype ~than:a (lub2 a b);
          at_least dtype ~than:b (lub2 a b));
      prop "promotion is a partial order"
        (Gen.map
           (fun (a, x, y) -> (a, lub2 a x, lub2 (lub2 a x) y))
           (Gen.triple promotable promotable promotable))
        (Law.partial_order dtype promotes);
      prop "least_upper of a list is its least common bound"
        ~examples:[ fp8_triple ]
        (Gen.list ~size:(Gen.int_range 1 5) promotable)
        (fun dts ->
          let bound = Dtype.least_upper dts in
          cover "a list that a pairwise fold bounds wrongly"
            (not (Dtype.equal bound (fold_bound dts)));
          equal dtype (least_common_bound dts) bound);
      test "least_upper rejects the empty list" (fun () ->
          rejects (fun () -> Dtype.least_upper []));
      prop "least_upper rejects void" (Gen.list promotable) (fun dts ->
          rejects (fun () -> Dtype.least_upper (dts @ [ Dtype.Void ])));
      prop "least_upper_float is a float" promotable (fun dt ->
          is_true (Dtype.is_float (Dtype.least_upper_float dt)));
      cases "least_upper_float keeps a float whatever the default" ~name:alias
        Dtype.floats (fun default ->
          with_setting Helpers.default_float (alias default) (fun () ->
              List.iter
                (fun dt -> equal dtype dt (Dtype.least_upper_float dt))
                Dtype.floats));
      cases "least_upper_float takes an integer to the default float"
        ~name:alias Dtype.floats (fun default ->
          with_setting Helpers.default_float (alias default) (fun () ->
              List.iter
                (fun dt -> equal dtype default (Dtype.least_upper_float dt))
                Dtype.ints));
    ]

let lossless_cast =
  Gen.bind promotable (fun a ->
      Gen.bind
        (dtype_list
           (List.filter (Dtype.can_lossless_cast a) (List.tl declared)))
        (fun b -> Gen.map (fun v -> (a, b, v)) (value_of a)))

let lossless =
  group "lossless casts"
    [
      Golden.cases "lossless_cast.golden" ~key:[ "from"; "to" ] (fun cell ->
          equal bool
            (bool_cell (cell "lossless"))
            (Dtype.can_lossless_cast
               (of_cell (cell "from"))
               (of_cell (cell "to"))));
      prop "a lossless cast round-trips every value" lossless_cast
        (fun (a, b, v) ->
          Law.round_trip const const (Dtype.const b) (Dtype.const a)
            (v :> Dtype.const));
    ]

(* Casts *)

(* Where tolk.next departs from tinygrad, a value converts as tinygrad converts
   another, whose own golden row checks it.

   D9. bfloat16 rounds once from the double: a float that tinygrad's float32
   step rounds onto a bfloat16 tie converts as the bfloat16 above it.

   Excluded (README): CPython's refusal to convert an integer of 2^1024 or more
   to a float. Where tinygrad raises, such an integer converts as the infinity
   of its sign. *)
let bfloat16_ties = [ 1.0039062500000002; 1. +. 0x1p-8 +. 0x1p-40 ]

let as_tinygrad dt cell = function
  | `Float f when Dtype.equal dt Dtype.Bfloat16 && List.mem f bfloat16_ties ->
      Some (`Float 1.0078125)
  | `Int n when Dtype.is_float dt && Z.numbits n > 1024 && raised cell ->
      Some
        (`Float (if Z.sign n < 0 then Float.neg_infinity else Float.infinity))
  | _ -> None

(* [row w read cell dt v f] checks [f v] against the golden row of [dt] and [v],
   or, where D9 departs from tinygrad, against [f] of the value that [v]
   converts as, which its own row checks. *)
let row w read cell dt v f =
  match as_tinygrad dt cell v with
  | Some v' -> equal w (f v') (f v)
  | None -> expect w read cell (fun () -> f v)

(* [nearest_bfloat16 x r], for a float [x >= 0], is that [r] is the bfloat16
   nearest to [x], the one with an even last bit on a tie, or an infinity from
   the midpoint past the largest bfloat16. A bfloat16's neighbours are one unit
   away in the upper half of its float32 bits, the largest one's upper neighbour
   is a unit of its binade above it, and the midpoint of two neighbours is exact
   in a float. *)
let nearest_bfloat16 x r =
  let bfloat16 top = Int32.float_of_bits (Int32.of_int (top lsl 16)) in
  let above top =
    if top = 0x7F7F then bfloat16 top +. (bfloat16 top -. bfloat16 (top - 1))
    else bfloat16 (top + 1)
  in
  let midpoint a b = (a +. b) /. 2. in
  if r = Float.infinity then
    at_least float_exact ~than:(midpoint (bfloat16 0x7F7F) (above 0x7F7F)) x
  else begin
    let bits = Int32.to_int (Int32.bits_of_float r) land 0xFFFF_FFFF in
    equal ~msg:"a bfloat16" int 0 (bits land 0xFFFF);
    let top = bits lsr 16 in
    let low = if top = 0 then 0. else midpoint (bfloat16 (top - 1)) r in
    let high = midpoint r (above top) in
    at_least float_exact ~than:low x;
    at_most float_exact ~than:high x;
    if x = low || x = high then equal ~msg:"even on a tie" int 0 (top land 1)
  end

let any_value =
  Gen.with_pp (Testable.pp value)
    (Gen.frequency
       [
         (1, Gen.map (fun b -> `Bool b) Gen.bool);
         (2, Gen.map (fun n -> `Int n) integer);
         (3, Gen.map (fun f -> `Float f) Gen.any_float);
       ])

(* A data type and a value it truncates: an integer data type takes no float,
   and a float no integer beyond float64. *)
let truncatable =
  Gen.bind stored (fun dt ->
      let takes = function
        | `Float _ -> Dtype.is_float dt || Dtype.is_bool dt
        | `Int n -> (not (Dtype.is_float dt)) || Z.numbits n < 1000
        | `Bool _ -> true
      in
      Gen.map (fun v -> (dt, v)) (Gen.such_that takes any_value))

let truncated dt f = as_float (Dtype.truncate dt (`Float f))

let truncation =
  group "truncate"
    [
      Golden.cases "truncation.golden" ~key:[ "dtype"; "value" ] (fun cell ->
          let dt = of_cell (cell "dtype")
          and v = value_of_cell (cell "value") in
          row value value_of_cell (cell "truncated") dt v (Dtype.truncate dt));
      prop "truncation is idempotent" truncatable (fun (dt, v) ->
          Law.idempotent value (Dtype.truncate dt) v);
      prop "a weak data type truncates nothing"
        (Gen.pair (dtype_list Dtype.weaks) any_value)
        (fun (dt, v) -> equal value v (Dtype.truncate dt v));
      test "void truncates nothing" (fun () ->
          rejects (fun () -> Dtype.truncate Dtype.Void (`Int Z.zero)));
      prop "an integer wraps modulo two to its width"
        (Gen.pair (dtype_list Dtype.ints) integer)
        (fun (dt, n) ->
          let r =
            match Dtype.truncate dt (`Int n) with
            | `Int r -> r
            | _ -> fail "not an integer"
          in
          let lo, hi = int_bounds dt in
          is_true ~msg:"within bounds" (Z.leq lo r && Z.leq r hi);
          equal ~msg:"congruent" z Z.zero
            (Z.erem (Z.sub n r) (Z.shift_left Z.one (Dtype.bitsize dt))));
      prop "bfloat16 rounding is odd" Gen.any_float (fun f ->
          equal value
            (`Float (-.truncated Dtype.Bfloat16 f))
            (Dtype.truncate Dtype.Bfloat16 (`Float (-.f))));
      prop "bfloat16 rounds once, to the nearest, ties to even"
        ~examples:bfloat16_ties finite_float (fun f ->
          nearest_bfloat16 (Float.abs f)
            (truncated Dtype.Bfloat16 (Float.abs f)));
      cases "an infinity is NaN in an 8-bit float without infinities"
        ~name:alias
        Dtype.[ Fp8e4m3; Fp8e4m3fnuz; Fp8e5m2fnuz ]
        (fun dt ->
          is_true (Float.is_nan (truncated dt Float.infinity));
          is_true (Float.is_nan (truncated dt Float.neg_infinity)));
      prop "a finite float stays finite in an 8-bit float"
        (Gen.pair (dtype_list Dtype.fp8s) finite_float)
        (fun (dt, f) -> is_true (Float.is_finite (truncated dt f)));
      prop "rounding to a float is monotone"
        (Gen.triple (dtype_list Dtype.floats) finite_float finite_float)
        (fun (dt, a, b) ->
          Law.monotone float_exact float_exact (truncated dt) (a, b));
    ]

let storage =
  group "storage"
    [
      Golden.cases "truncation.golden" ~key:[ "dtype"; "value" ] (fun cell ->
          let dt = of_cell (cell "dtype")
          and v = value_of_cell (cell "value") in
          row value value_of_cell (cell "storage") dt v
            (Dtype.to_storage_scalar dt));
      Golden.cases "decode.golden" ~key:[ "dtype"; "storage" ] (fun cell ->
          equal value
            (value_of_cell (cell "value"))
            (Dtype.from_storage_scalar
               (of_cell (cell "dtype"))
               (value_of_cell (cell "storage"))));
      prop "storage round-trips every value"
        (Gen.bind stored (fun dt -> Gen.map (fun v -> (dt, v)) (value_of dt)))
        (fun (dt, v) ->
          Law.round_trip value value
            (Dtype.to_storage_scalar dt)
            (Dtype.from_storage_scalar dt)
            v);
      cases "every bit pattern decodes and encodes back, a NaN to a NaN"
        ~name:alias
        Dtype.(Bfloat16 :: fp8s)
        (fun dt ->
          for bits = 0 to (1 lsl Dtype.bitsize dt) - 1 do
            let msg = Printf.sprintf "0x%x" bits in
            let stored = `Int (Z.of_int bits) in
            let decoded = Dtype.from_storage_scalar dt stored in
            let back = Dtype.to_storage_scalar dt decoded in
            let f = as_float decoded in
            if Float.is_nan f then begin
              let g = as_float (Dtype.from_storage_scalar dt back) in
              is_true ~msg (Float.is_nan g);
              equal ~msg bool (Float.sign_bit f) (Float.sign_bit g)
            end
            else equal ~msg value stored back
          done);
      (* D10. A float8 NaN keeps its sign when decoded: tinygrad decodes e4m3's
         0xFF as a positive NaN. *)
      test "a NaN decodes with the sign of its bits" (fun () ->
          List.iter
            (fun (dt, bits, negative) ->
              let f =
                as_float (Dtype.from_storage_scalar dt (`Int (Z.of_int bits)))
              in
              is_true
                ~msg:(Printf.sprintf "0x%x is a NaN" bits)
                (Float.is_nan f);
              equal
                ~msg:(Printf.sprintf "0x%x's sign" bits)
                bool negative (Float.sign_bit f))
            Dtype.
              [
                (Fp8e4m3, 0x7F, false);
                (Fp8e4m3, 0xFF, true);
                (Fp8e5m2, 0x7D, false);
                (Fp8e5m2, 0xFD, true);
                (Fp8e5m2, 0xFF, true);
                (Bfloat16, 0x7FC0, false);
                (Bfloat16, 0xFFC0, true);
              ]);
      cases "an encoded float decodes only from an integer" ~name:alias
        Dtype.(Bfloat16 :: fp8s)
        (fun dt ->
          rejects (fun () -> Dtype.from_storage_scalar dt (`Float 1.));
          rejects (fun () -> Dtype.from_storage_scalar dt (`Bool true)));
    ]

let integer_bitcast =
  Gen.bind stored (fun a ->
      Gen.bind
        (dtype_list
           (List.filter
              (fun b -> Dtype.itemsize a = Dtype.itemsize b)
              Dtype.ints))
        (fun b -> Gen.map (fun v -> (a, b, v)) (value_of a)))

let bitcasts =
  group "bitcast"
    [
      Golden.cases "bitcasts.golden" ~key:[ "from"; "to"; "value" ] (fun cell ->
          let a = of_cell (cell "from") and b = of_cell (cell "to") in
          row value value_of_cell (cell "bitcast") a
            (value_of_cell (cell "value"))
            (Dtype.bitcast a b));
      prop "a bitcast to an integer and back is the identity" integer_bitcast
        (fun (a, b, v) ->
          Law.round_trip value value (Dtype.bitcast a b) (Dtype.bitcast b a) v);
      test "a bitcast keeps the item size" (fun () ->
          rejects (fun () ->
              Dtype.bitcast Dtype.Int8 Dtype.Float16 (`Int Z.one)));
      cases "a bitcast needs data types with storage" ~name:alias
        Dtype.[ Void; Weak_int; Weak_float ]
        (fun dt ->
          rejects (fun () -> Dtype.bitcast dt dt (`Int Z.one));
          rejects (fun () -> Dtype.bitcast Dtype.Int64 dt (`Int Z.one)));
    ]

(* A data type and a constant it takes: only a float or bool takes a NaN or an
   infinity, and only a float of known width is bounded. *)
let constable =
  Gen.bind every (fun dt ->
      let takes = function
        | `Float f -> Float.is_finite f || Dtype.is_float dt || Dtype.is_bool dt
        | _ -> true
      in
      Gen.map (fun c -> (dt, c)) (Gen.such_that takes small_const))

let consts =
  group "const"
    [
      Golden.cases "const.golden" ~key:[ "dtype"; "value" ] (fun cell ->
          let dt = of_cell (cell "dtype")
          and c = const_of_cell (cell "value") in
          match c with
          | #Dtype.value as v ->
              row const const_of_cell (cell "const") dt v (fun v ->
                  Dtype.const dt v)
          | `Invalid -> equal const `Invalid (Dtype.const dt c));
      prop "const is idempotent" constable (fun (dt, c) ->
          Law.idempotent const (Dtype.const dt) c);
    ]

let () =
  exit
    (run "tolk.next.dtype"
       [
         constants;
         address_spaces;
         data_types;
         names;
         defaults;
         literals;
         promotion;
         lossless;
         truncation;
         storage;
         bitcasts;
         consts;
       ])
