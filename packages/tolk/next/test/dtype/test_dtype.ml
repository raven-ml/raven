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

(* The same constant built anew: an integer from its digits, and a float from
   its bits. *)
let respell = function
  | `Int n -> `Int (Z.of_string (Z.to_string n))
  | `Float f -> `Float (Int64.float_of_bits (Int64.bits_of_float f))
  | c -> c

let const_equality =
  Testable.make ~pp:(Testable.pp const) ~equal:Dtype.equal_const

(* The witness's equality, with floats the same when their bits are. *)
let same_const c0 c1 =
  match (c0, c1) with
  | `Float f0, `Float f1 ->
      Int64.equal (Int64.bits_of_float f0) (Int64.bits_of_float f1)
  | _ -> Testable.equal const c0 c1

let constants =
  group "constants"
    [
      test "NaNs of different bits are different constants" (fun () ->
          List.iter
            (fun f ->
              if Int64.bits_of_float f <> Int64.bits_of_float Float.nan then
                not_equal const_equality (`Float Float.nan) (`Float f))
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
      prop "equal_const is equality of kinds and payloads, floats by their bits"
        (Gen.pair any_const any_const) (fun (c0, c1) ->
          equal bool (same_const c0 c1) (Dtype.equal_const c0 c1));
      prop "equal constants hash alike" any_const (fun c ->
          cover "a NaN" (match c with `Float f -> Float.is_nan f | _ -> false);
          cover "an integer built anew"
            (match c with `Int _ -> true | _ -> false);
          equal int (Dtype.hash_const c) (Dtype.hash_const (respell c)));
      Golden.cases "const_repr.golden" (fun cell ->
          equal string (cell "printed")
            (Format.asprintf "%a" Dtype.pp_const (const_of_cell (cell "const"))));
    ]

let after prefix s =
  if not (String.starts_with ~prefix s) then
    failf "%S has no %s prefix" s prefix;
  String.sub s (String.length prefix) (String.length s - String.length prefix)

let space = Testable.make ~pp:Dtype.pp_addr_space ~equal:( = )

let address_spaces =
  group "address spaces"
    [
      cases "an address space prints as its enum member" ~name:snd
        Dtype.
          [
            (Global, "AddrSpace.GLOBAL");
            (Local, "AddrSpace.LOCAL");
            (Reg, "AddrSpace.REG");
            (Alu, "AddrSpace.ALU");
          ]
        (fun (space, printed) ->
          equal string printed (Format.asprintf "%a" Dtype.pp_addr_space space));
      prop "addr_space_of_string reads the name pp prints after AddrSpace."
        (Gen.of_list ~pp:Dtype.pp_addr_space Dtype.[ Global; Local; Reg; Alu ])
        (Law.round_trip space string (Format.asprintf "%a" Dtype.pp_addr_space)
           (fun s ->
             require_ok (Dtype.addr_space_of_string (after "AddrSpace." s))));
      cases "the error names what is not an address space"
        ~name:(Printf.sprintf "%S")
        [ "global"; "Global"; "AddrSpace.GLOBAL"; " GLOBAL"; "SHARED" ]
        (fun s ->
          contains ~sub:s (require_error (Dtype.addr_space_of_string s)));
      test "the empty string is not an address space" (fun () ->
          is_error (Dtype.addr_space_of_string ""));
    ]

(* Data types *)

let properties =
  Golden.cases "properties.golden" (fun cell ->
      let dt = dtype_of_cell (cell "dtype") in
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
        (List.map dtype_of_cell (String.split_on_char ' ' (cell "members")))
        group)

(* The rows of properties.golden are tinygrad's sorted data types. *)
let sorted =
  List.map
    (fun cell -> dtype_of_cell (cell "dtype"))
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
            (Dtype.finfo (dtype_of_cell (cell "dtype"))));
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
            (Ok (dtype_of_cell (cell "dtype")))
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
          "dtypes.half";
          "dtypes.floats";
        ] (fun s -> contains ~sub:s (require_error (Dtype.of_string s)));
      test "the empty string is not a data type" (fun () ->
          is_error (Dtype.of_string ""));
      prop "of_string reads the name pp prints after dtypes." every
        (Law.round_trip dtype string (Format.asprintf "%a" Dtype.pp) (fun s ->
             require_ok (Dtype.of_string (after "dtypes." s))));
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
          let dt = dtype_of_cell (cell "dtype") in
          equal dtype (dtype_of_cell (cell "weak")) (Dtype.weak dt);
          equal dtype (dtype_of_cell (cell "strong")) (Dtype.strong dt);
          expect dtype dtype_of_cell (cell "least_upper_float") (fun () ->
              Dtype.least_upper_float dt);
          expect dtype dtype_of_cell (cell "sum_acc") (fun () ->
              Dtype.sum_acc dt);
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
            (dtype_of_cell (cell "dtype"))
            (Dtype.of_const (const_of_cell (cell "value"))));
      Golden.cases "of_consts.golden" (fun cell ->
          expect dtype dtype_of_cell (cell "dtype") (fun () ->
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
            | s -> Some (dtype_of_cell s)
          in
          expect dtype dtype_of_cell (cell "dtype") (fun () ->
              Dtype.commit_int ?default_int lo hi));
      cases "commit_int takes only an integer of known width as default"
        ~name:alias
        Dtype.[ Weak_int; Bool; Float32; Weak_float; Void ]
        (fun default_int ->
          rejects (fun () -> Dtype.commit_int ~default_int Z.zero Z.one));
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
            (dtype_of_cell (cell "least_upper"))
            (lub2 (dtype_of_cell (cell "a")) (dtype_of_cell (cell "b"))));
      Golden.cases "least_upper_triples.golden" ~key:[ "a"; "b"; "c" ]
        (fun cell ->
          equal dtype
            (dtype_of_cell (cell "least_upper"))
            (Dtype.least_upper
               (List.map (fun c -> dtype_of_cell (cell c)) [ "a"; "b"; "c" ])));
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
               (dtype_of_cell (cell "from"))
               (dtype_of_cell (cell "to"))));
      prop "a lossless cast round-trips every value" lossless_cast
        (fun (a, b, v) ->
          Law.round_trip const const (Dtype.const b) (Dtype.const a)
            (v :> Dtype.const));
    ]

(* Casts *)

(* The integer type whose words store [dt]. *)
let word_of dt =
  match Dtype.itemsize dt with
  | 1 -> Dtype.Uint8
  | 2 -> Dtype.Uint16
  | 4 -> Dtype.Uint32
  | _ -> Dtype.Int64

(* Where tolk.next departs from tinygrad, a value converts as tinygrad converts
   another, whose own golden row checks it.

   D9. Narrow float conversions are IEEE conversions: a float that tinygrad's
   float32 step rounds onto a bfloat16 tie converts as the bfloat16 above it.

   Excluded (README): CPython's refusal to convert an integer whose magnitude
   rounds to 2^1024 or more (at least 2^1024 - 2^970) to a float. Where tinygrad
   raises, such an integer converts to a double as the infinity of its sign, and
   to a narrower float as the greatest double does: the finite value it is
   overflows every narrower float alike. *)
let beyond_doubles = Z.sub (Z.shift_left Z.one 1024) (Z.shift_left Z.one 970)
let bfloat16_ties = [ 1.0039062500000002; 1. +. 0x1p-8 +. 0x1p-40 ]

let as_tinygrad dt cell = function
  | `Float f when Dtype.equal dt Dtype.Bfloat16 && List.mem f bfloat16_ties ->
      Some (`Float 1.0078125)
  | `Int n
    when Dtype.is_float dt && Z.geq (Z.abs n) beyond_doubles && raised cell ->
      let big =
        if Dtype.(equal dt Float64 || equal dt Weak_float) then Float.infinity
        else Float.max_float
      in
      Some (`Float (if Z.sign n < 0 then -.big else big))
  | _ -> None

(* Storage moves a NaN's bits: every word encodes back to itself through its
   data type. tinygrad departs from that on NaN words, and a row gives the word
   itself where

   D10. A float8 NaN keeps its sign when decoded: tinygrad encodes e4m3's 0xFF
   back as 0x7F;

   D20. Storage keeps an e5m2 NaN's payload: tinygrad encodes every e5m2 NaN as
   0x7F of its sign;

   Excluded (README): CPython's struct packing every float16 NaN as the
   canonical 0x7e00, of its sign, and its conversions between float32 and double
   quieting a signalling NaN, which tinygrad's bfloat16 storage goes through. *)
let nan_word dt word =
  let bits mask = Z.to_int (Z.logand word (Z.of_int mask)) in
  match dt with
  | Dtype.Fp8e4m3 -> bits 0xFF = 0xFF
  | Fp8e5m2 -> bits 0x7C = 0x7C && bits 0x03 <> 0
  | Float16 -> bits 0x7C00 = 0x7C00 && bits 0x3FF <> 0
  | Bfloat16 -> bits 0x7F80 = 0x7F80 && bits 0x7F <> 0
  | _ -> false

let reencoded dt word tinygrad =
  if nan_word dt word then `Int word else tinygrad

(* D20: the e5m2 storage of a NaN is its quiet code, 0x7E of its sign, where
   tinygrad stores 0x7F. *)
let e5m2_nan = function
  | `Float f when Float.is_nan f ->
      Some (`Int (Z.of_int (if Float.sign_bit f then 0xFE else 0x7E)))
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
          let dt = dtype_of_cell (cell "dtype")
          and v = value_of_cell (cell "value") in
          row value value_of_cell (cell "truncated") dt v (Dtype.truncate dt));
      prop "truncation is idempotent" truncatable (fun (dt, v) ->
          Law.idempotent value (Dtype.truncate dt) v);
      prop "a weak data type truncates nothing"
        (Gen.pair (dtype_list Dtype.weaks) any_value)
        (fun (dt, v) -> equal value v (Dtype.truncate dt v));
      test "truncate rejects void" (fun () ->
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
      test "an integer rounds to a narrower float once, from its value"
        (fun () ->
          (* Past 2^53 the double rounds an integer onto a tie of the narrower
             float, which it would then round to even. *)
          let z = Z.of_string "9042383626829825" in
          equal value
            (Dtype.bitcast Uint16 Bfloat16 (`Int (Z.of_int 0x5a01)))
            (Dtype.truncate Bfloat16 (`Int z));
          let z = Z.(shift_left one 60 + shift_left one 36 + one) in
          equal value
            (`Float (Float.ldexp 1. 60 +. Float.ldexp 1. 37))
            (Dtype.truncate Float32 (`Int z)));
      cases "an infinity is NaN in an 8-bit float without infinities"
        ~name:alias
        Dtype.[ Fp8e4m3; Fp8e4m3fnuz; Fp8e5m2fnuz ]
        (fun dt ->
          let signed = Dtype.equal dt Dtype.Fp8e4m3 in
          List.iter
            (fun inf ->
              let f = truncated dt inf in
              is_true (Float.is_nan f);
              equal ~msg:"sign" bool (signed && inf < 0.) (Float.sign_bit f))
            [ Float.infinity; Float.neg_infinity ]);
      cases "truncate quiets a signalling NaN and keeps its sign" ~name:alias
        Dtype.floats (fun dt ->
          let w = word_of dt in
          List.iter
            (fun bits ->
              let msg = Printf.sprintf "0x%LX" bits in
              let negative = Int64.compare bits 0L < 0 in
              let v = Dtype.truncate dt (`Float (Int64.float_of_bits bits)) in
              is_true ~msg (Float.is_nan (as_float v));
              let word =
                match Dtype.bitcast dt w v with `Int n -> n | _ -> fail msg
              in
              let has mask = not (Z.equal (Z.logand word mask) Z.zero) in
              let bit n = Z.shift_left Z.one n in
              match dt with
              | Dtype.Fp8e4m3fnuz | Fp8e5m2fnuz ->
                  equal ~msg z (Z.of_int 0x80) word
              | Fp8e4m3 | Fp8e5m2 ->
                  equal ~msg z (Z.of_int (if negative then 0xFF else 0x7F)) word
              | Float64 ->
                  is_true ~msg:"quiet" (has (bit 51));
                  equal ~msg:"sign" bool negative (Z.sign word < 0)
              | _ ->
                  let bits = 8 * Dtype.itemsize dt in
                  let mantissa = snd (Dtype.finfo dt) in
                  is_true ~msg:"quiet" (has (bit (mantissa - 1)));
                  equal ~msg:"sign" bool negative (has (bit (bits - 1))))
            [
              0x7FF0_0000_0000_0001L;
              0x7FF4_0000_0000_0000L;
              0xFFF0_0000_0000_0001L;
            ]);
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
          let dt = dtype_of_cell (cell "dtype")
          and v = value_of_cell (cell "value") in
          match (dt, e5m2_nan v) with
          | Dtype.Fp8e5m2, Some word ->
              equal value word (Dtype.to_storage_scalar dt v)
          | _ ->
              row value value_of_cell (cell "storage") dt v
                (Dtype.to_storage_scalar dt));
      Golden.cases "decode.golden" ~key:[ "dtype"; "storage" ] (fun cell ->
          equal value
            (value_of_cell (cell "value"))
            (Dtype.from_storage_scalar
               (dtype_of_cell (cell "dtype"))
               (value_of_cell (cell "storage"))));
      Golden.cases "reencode.golden" ~key:[ "dtype"; "word" ] (fun cell ->
          let dt = dtype_of_cell (cell "dtype")
          and word = Z.of_string (cell "word") in
          let unsigned =
            if Dtype.itemsize dt = 1 then Dtype.Uint8 else Dtype.Uint16
          in
          equal value
            (reencoded dt word (value_of_cell (cell "reencoded")))
            (Dtype.bitcast dt unsigned (Dtype.bitcast unsigned dt (`Int word))));
      prop "storage round-trips every value"
        (Gen.bind stored (fun dt -> Gen.map (fun v -> (dt, v)) (value_of dt)))
        (fun (dt, v) ->
          Law.round_trip value value
            (Dtype.to_storage_scalar dt)
            (Dtype.from_storage_scalar dt)
            v);
      cases "a bitcast through a float gives back every 8- and 16-bit word"
        ~name:alias
        Dtype.
          [
            Fp8e4m3;
            Fp8e5m2;
            Fp8e4m3fnuz;
            Fp8e5m2fnuz;
            Int8;
            Float16;
            Bfloat16;
            Int16;
          ]
        (fun dt ->
          let unsigned =
            if Dtype.itemsize dt = 1 then Dtype.Uint8 else Dtype.Uint16
          in
          for word = 0 to (1 lsl (8 * Dtype.itemsize dt)) - 1 do
            let word = `Int (Z.of_int word) in
            equal
              ~msg:(Format.asprintf "%a" (Testable.pp value) word)
              value word
              (Dtype.bitcast dt unsigned (Dtype.bitcast unsigned dt word))
          done);
      prop "a bitcast through a float gives back every 32- and 64-bit word"
        ~examples:
          [
            (Dtype.Float32, `Int (Z.of_int 0x7F80_0001));
            (Dtype.Float32, `Int (Z.of_int 0xFFBF_FFFF));
            (Dtype.Float64, `Int (Z.of_int64 0x7FF0_0000_0000_0001L));
            (Dtype.Float64, `Int (Z.of_int64 0xFFF7_FFFF_FFFF_FFFFL));
          ]
        (Gen.bind
           (dtype_list Dtype.[ Float32; Float64; Int32 ])
           (fun dt -> Gen.map (fun v -> (dt, v)) (value_of (word_of dt))))
        (fun (dt, word) ->
          let w = word_of dt in
          equal value word (Dtype.bitcast dt w (Dtype.bitcast w dt word)));
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
          let a = dtype_of_cell (cell "from")
          and b = dtype_of_cell (cell "to") in
          let v = value_of_cell (cell "value") in
          match (a, e5m2_nan v) with
          | Dtype.Fp8e5m2, Some word ->
              equal value
                (Dtype.bitcast Dtype.Uint8 b word)
                (Dtype.bitcast a b v)
          | _ ->
              row value value_of_cell (cell "bitcast") a v (Dtype.bitcast a b));
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
          let dt = dtype_of_cell (cell "dtype")
          and c = const_of_cell (cell "value") in
          match c with
          | #Dtype.value as v ->
              row const const_of_cell (cell "const") dt v (fun v ->
                  Dtype.const dt v)
          | `Invalid -> equal const `Invalid (Dtype.const dt c));
      prop "const is idempotent" constable (fun (dt, c) ->
          Law.idempotent const (Dtype.const dt) c);
      cases "const keeps the bits of every 8- and 16-bit float word" ~name:alias
        Dtype.[ Fp8e4m3; Fp8e5m2; Fp8e4m3fnuz; Fp8e5m2fnuz; Float16; Bfloat16 ]
        (fun dt ->
          let w = word_of dt in
          for word = 0 to (1 lsl (8 * Dtype.itemsize dt)) - 1 do
            let word = `Int (Z.of_int word) in
            match Dtype.const dt (Dtype.bitcast w dt word) with
            | #Dtype.value as v ->
                equal
                  ~msg:(Format.asprintf "%a" (Testable.pp value) word)
                  value word (Dtype.bitcast dt w v)
            | `Invalid -> fail "Invalid"
          done);
    ]

(* Values *)

(* Excluded (README): CPython's refusal to convert an integer whose magnitude
   rounds to 2^1024 or more to a float. Where Python raises, such an integer
   operates with a float as the infinity of its sign. *)
let to_infinity = function
  | `Int n when Z.geq (Z.abs n) beyond_doubles ->
      `Float (if Z.sign n < 0 then Float.neg_infinity else Float.infinity)
  | v -> v

(* [arithmetic cell op a b] is that [op a b] is what [cell] reads as. Where
   Python divides by zero, it raises [Division_by_zero]; where Python refuses a
   huge integer, it is [op] on the infinity of its sign. *)
let arithmetic cell op a b =
  if String.equal cell "raises ZeroDivisionError" then
    raises Division_by_zero (fun () -> op a b)
  else if raised cell then
    match op (to_infinity a) (to_infinity b) with
    | expected -> equal value expected (op a b)
    | exception Division_by_zero -> raises Division_by_zero (fun () -> op a b)
  else equal value (value_of_cell cell) (op a b)

let magnitude =
  Testable.with_compare Dtype.Value.compare
    (Testable.make ~pp:(Testable.pp value) ~equal:(fun a b ->
         Dtype.Value.compare a b = 0))

let operand =
  Gen.with_pp (Testable.pp value)
    (Gen.frequency
       [
         (1, Gen.map (fun b -> `Bool b) Gen.bool);
         (3, Gen.map (fun n -> `Int n) integer);
         (1, Gen.map (fun n -> `Int (Z.of_int n)) (Gen.int_range (-3) 3));
         (3, Gen.map (fun f -> `Float f) Gen.any_float);
         ( 1,
           Gen.map
             (fun f -> `Float f)
             (Gen.of_list [ 0.; -0.; 1.; -1.; 0x1p53 ]) );
       ])

let number =
  Gen.such_that
    (function `Float f -> not (Float.is_nan f) | _ -> true)
    operand

let integers = Gen.map (fun n -> `Int n) integer

let is_zero = function
  | `Int n -> Z.equal n Z.zero
  | `Bool b -> not b
  | `Float f -> Stdlib.( = ) f 0.

let is_nan = function `Float f -> Float.is_nan f | _ -> false

let values =
  let open Dtype.Value in
  group "values"
    [
      Golden.cases "values.golden" ~key:[ "a"; "b" ] (fun cell ->
          let a = value_of_cell (cell "a") and b = value_of_cell (cell "b") in
          equal bool (bool_cell (cell "lt")) (a < b);
          equal bool (bool_cell (cell "le")) (a <= b);
          equal bool (bool_cell (cell "eq")) (a = b);
          equal bool (bool_cell (cell "ne")) (a <> b);
          equal ~msg:"min" value (value_of_cell (cell "min")) (min a b);
          equal ~msg:"max" value (value_of_cell (cell "max")) (max a b);
          arithmetic (cell "add") ( + ) a b;
          arithmetic (cell "sub") ( - ) a b;
          arithmetic (cell "mul") ( * ) a b;
          arithmetic (cell "floordiv") ( // ) a b;
          arithmetic (cell "mod") ( % ) a b);
      Golden.cases "negated.golden" (fun cell ->
          equal value
            (value_of_cell (cell "negated"))
            (-value_of_cell (cell "a")));
      test "of_int is an integer" (fun () ->
          equal value (`Int (Z.of_int (-7))) (of_int (-7));
          equal value (`Int (Z.of_int Stdlib.max_int)) (of_int Stdlib.max_int));
      prop "compare is a total order by magnitude"
        (Gen.triple operand operand operand)
        (Law.order magnitude);
      prop "compare agrees with < and = away from NaN" (Gen.pair number number)
        (fun (a, b) ->
          let c = compare a b in
          equal ~msg:"<" bool (a < b) (Stdlib.( < ) c 0);
          equal ~msg:"=" bool (a = b) (Stdlib.( = ) c 0));
      prop "NaN is less than every other value and unordered by <" number
        (fun v ->
          let nan = `Float Float.nan in
          less int ~than:0 (compare nan v);
          equal int 0 (compare nan nan);
          is_false (nan < v || v < nan || nan = v || nan <= v || v >= nan);
          is_true (nan <> v));
      prop "the comparisons are derived from < and =" (Gen.pair operand operand)
        (fun (a, b) ->
          equal ~msg:"<>" bool (not (a = b)) (a <> b);
          equal ~msg:"<=" bool (a < b || a = b) (a <= b);
          equal ~msg:">" bool (b < a) (a > b);
          equal ~msg:">=" bool (b <= a) (a >= b));
      prop "min and max return an argument, the least and the greatest"
        (Gen.pair number number) (fun (a, b) ->
          let lo = min a b and hi = max a b in
          is_true ~msg:"min is an argument" (lo == a || lo == b);
          is_true ~msg:"max is an argument" (hi == a || hi == b);
          at_most magnitude ~than:a lo;
          at_most magnitude ~than:b lo;
          at_least magnitude ~than:a hi;
          at_least magnitude ~than:b hi);
      prop "addition and multiplication are commutative"
        (Gen.pair operand operand) (fun (a, b) ->
          Law.commutative value ( + ) (a, b);
          Law.commutative value ( * ) (a, b));
      prop "integer arithmetic is exact" (Gen.pair integers integers)
        (fun (a, b) ->
          match (a, b) with
          | `Int m, `Int n ->
              equal ~msg:"+" value (`Int (Z.add m n)) (a + b);
              equal ~msg:"-" value (`Int (Z.sub m n)) (a - b);
              equal ~msg:"*" value (`Int (Z.mul m n)) (a * b)
          | _ -> fail "not integers");
      prop "multiplication distributes over addition on integers"
        (Gen.triple integers integers integers)
        (Law.distributive value ( * ) ~over:( + ));
      prop "float arithmetic is the float's"
        (Gen.pair Gen.any_float Gen.any_float) (fun (x, y) ->
          equal ~msg:"+" value (`Float (Float.add x y)) (`Float x + `Float y);
          equal ~msg:"-" value (`Float (Float.sub x y)) (`Float x - `Float y);
          equal ~msg:"*" value (`Float (Float.mul x y)) (`Float x * `Float y));
      (* An integer zero has no sign, so -0.0 - 0 is -0.0 while -0.0 + -0 is
         0.0. *)
      prop "subtraction adds the negation of anything but an integer zero"
        (Gen.pair operand
           (Gen.such_that
              (function
                | `Int n -> not (Z.equal n Z.zero)
                | `Bool b -> b
                | `Float _ -> true)
              operand))
        (fun (a, b) -> equal value (a + -b) (a - b));
      prop "negation is an involution on integers and floats"
        (Gen.such_that (function `Bool _ -> false | _ -> true) operand)
        (Law.involutive value ( ~- ));
      prop "integer division and modulo rebuild the dividend"
        (Gen.pair integers
           (Gen.such_that
              (function `Int n -> not (Z.equal n Z.zero) | _ -> true)
              integers))
        (fun (a, b) -> equal value a ((a // b * b) + (a % b)));
      prop "an integer remainder is less than the divisor and of its sign"
        (Gen.pair integers
           (Gen.such_that
              (function `Int n -> not (Z.equal n Z.zero) | _ -> true)
              integers))
        (fun (a, b) ->
          match (a % b, b) with
          | `Int r, `Int n ->
              is_true ~msg:"sign"
                (Stdlib.( = ) (Z.sign r) 0 || Stdlib.( = ) (Z.sign r) (Z.sign n));
              is_true ~msg:"size" (Z.lt (Z.abs r) (Z.abs n))
          | _ -> fail "not integers");
      (* Moving a remainder to the divisor's sign adds the divisor, which can
         round up to it: 5e-324 % -4.450246113776542e-308 is the divisor. *)
      prop "a float remainder is at most the divisor and of its sign"
        ~examples:[ (5e-324, -4.450246113776542e-308) ]
        (Gen.pair finite_float
           (Gen.such_that (fun f -> Stdlib.( <> ) f 0.) finite_float))
        (fun (x, y) ->
          let r = as_float (`Float x % `Float y) in
          equal ~msg:"sign" bool (Float.sign_bit y) (Float.sign_bit r);
          is_true ~msg:"size" (Stdlib.( <= ) (Float.abs r) (Float.abs y)));
      cases "a zero divisor raises Division_by_zero"
        ~name:(Format.asprintf "%a" (Testable.pp value))
        [ `Int Z.zero; `Bool false; `Float 0.; `Float (-0.) ]
        (fun zero ->
          raises Division_by_zero (fun () -> `Int (Z.of_int 7) // zero);
          raises Division_by_zero (fun () -> `Float 1.5 % zero));
      prop "a bool counts as an integer" (Gen.pair Gen.bool operand)
        (fun (b, v) ->
          let n = `Int (if b then Z.one else Z.zero) in
          equal ~msg:"+" value (n + v) (`Bool b + v);
          equal ~msg:"*" value (n * v) (`Bool b * v);
          equal ~msg:"-" value (-n) (-`Bool b);
          if not (is_zero v) then begin
            equal ~msg:"//" value (n // v) (`Bool b // v);
            equal ~msg:"%" value (n % v) (`Bool b % v)
          end);
    ]

let conversions =
  let open Dtype.Value in
  group "conversions"
    [
      Golden.cases "conversions.golden" (fun cell ->
          let v = value_of_cell (cell "value") in
          (match cell "float" with
          | c when raised c ->
              equal ~msg:"to_float" value (to_infinity v) (`Float (to_float v))
          | c ->
              equal ~msg:"to_float" value (value_of_cell c)
                (`Float (to_float v)));
          (match cell "int" with
          | c when raised c ->
              rejects (fun () -> to_z v);
              rejects (fun () -> to_int v)
          | c ->
              let n = Z.of_string c in
              equal ~msg:"to_z" z n (to_z v);
              if Z.fits_int n then
                equal ~msg:"to_int" int (Z.to_int n) (to_int v)
              else rejects (fun () -> to_int v));
          equal ~msg:"to_bool" bool (bool_cell (cell "bool")) (to_bool v));
      prop "to_float of a float is itself" Gen.any_float (fun f ->
          equal value (`Float f) (`Float (to_float (`Float f))));
      prop "to_z truncates a float towards zero" finite_float (fun f ->
          equal z (Z.of_float (Float.trunc f)) (to_z (`Float f)));
      prop "to_z of an integer is itself, and to_int where it fits" integer
        (fun n ->
          equal ~msg:"to_z" z n (to_z (`Int n));
          if Z.fits_int n then
            equal ~msg:"to_int" int (Z.to_int n) (to_int (`Int n))
          else rejects (fun () -> to_int (`Int n)));
      cases "to_int takes exactly the integers of int" ~name:Z.to_string
        Z.
          [
            of_int Stdlib.min_int;
            of_int Stdlib.max_int;
            pred (of_int Stdlib.min_int);
            succ (of_int Stdlib.max_int);
          ]
        (fun n ->
          if Z.fits_int n then equal int (Z.to_int n) (to_int (`Int n))
          else rejects (fun () -> to_int (`Int n)));
      prop "to_bool is being unequal to zero" operand (fun v ->
          equal bool (is_nan v || not (v = `Int Z.zero)) (to_bool v));
      prop "to_float preserves the order of values" (Gen.pair operand operand)
        (fun (a, b) ->
          assume (not (is_nan a || is_nan b));
          is_true ((not (a <= b)) || Stdlib.( <= ) (to_float a) (to_float b)));
    ]

let () =
  exit
    (run "Tolk_next.Dtype"
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
         values;
         conversions;
       ])
