(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Quantities: a structure that reports its unit's canonical text then walks its
   payload; conversion by one rounded factor or a raise; the algebra's units and
   payloads; and behaviour under rune's transformations. *)

open Windtrap
open Ymir_units
open Ymir_units_test

let km = Unit.(kilo metre)
let f32 xs = Nx.create Nx.float32 [| Array.length xs |] xs
let f64 xs = Nx.create Nx.float64 [| Array.length xs |] xs
let i32 xs = Nx.create Nx.int32 [| Array.length xs |] xs
let floats = Nx.to_array
let bits = array float_exact

let q32 : Nx.float32_t Quantity.t Nx.Ptree.t =
  Nx.Ptree.instantiate (module Quantity)

let q64 : Nx.float64_t Quantity.t Nx.Ptree.t =
  Nx.Ptree.instantiate (module Quantity)

let qi32 : Nx.int32_t Quantity.t Nx.Ptree.t =
  Nx.Ptree.instantiate (module Quantity)

(* [is_payload x y] states that [y] is the tensor [x] itself. *)
let is_payload x y =
  satisfies ~claim:"is the payload itself" pass (fun y -> x == y) y

(* [renamed fn msg] is [msg], a message of [Unit.ratio], naming [fn]. *)
let renamed fn msg =
  let prefix = "Unit.ratio" in
  let n = String.length prefix in
  fn ^ String.sub msg n (String.length msg - n)

(* Structure *)

let visit =
  let equal a b =
    match (a, b) with
    | Nx.Ptree.Leaf p, Nx.Ptree.Leaf q -> Nx.Ptree.Path.equal p q
    | Report (p, r), Report (q, s) -> Nx.Ptree.Path.equal p q && r = s
    | _ -> false
  in
  Testable.make ~pp:Nx.Ptree.pp_visit ~equal

let root = Nx.Ptree.Path.root
let at name = Nx.Ptree.Path.v [ Field name ]

type 'a star = { flux : 'a Quantity.t; parallax : 'a Quantity.t }

module Star = struct
  type 'a t = 'a star

  let walk c s =
    let open Nx.Ptree.Walk in
    let flux = field c "flux" Quantity.walk s.flux in
    let parallax = field c "parallax" Quantity.walk s.parallax in
    { flux; parallax }
end

let mas = Unit.(milli arcsecond)
let jansky = Unit.(decimal "1e-26" * kilogram / (second ** 2))

let star =
  {
    flux = Quantity.v jansky (f32 [| 3. |]);
    parallax = Quantity.v mas (f32 [| 0.5 |]);
  }

let structure =
  group "A quantity reports one string"
    [
      prop "a quantity's visits are its unit's text, then its payload" units
        (fun u ->
          equal (list visit)
            [ Report (root, Case (Unit.to_string u)); Leaf root ]
            (Nx.Ptree.visits q32 (Quantity.v u (f32 [| 1. |]))));
      test "a record of quantities reports each field's unit at its path"
        (fun () ->
          equal (list visit)
            [
              Report (at "flux", Case "1e-26 kg s^-2");
              Leaf (at "flux");
              Report (at "parallax", Case "1/648000000 pi rad");
              Leaf (at "parallax");
            ]
            (Nx.Ptree.visits (Nx.Ptree.instantiate (module Star)) star));
      test "cast changes the payloads' dtype and keeps the units" (fun () ->
          let s = Nx.Ptree.cast (module Star) Nx.float64 star in
          equal unit jansky (Quantity.unit s.flux);
          equal unit mas (Quantity.unit s.parallax);
          equal bits [| 0.5 |] (floats (Quantity.value mas s.parallax)));
      test "a payload map keeps the units" (fun () ->
          let s =
            Nx.Ptree.Payload.map (module Star) (fun _ x -> Nx.mul_s x 2.) star
          in
          equal unit jansky (Quantity.unit s.flux);
          equal bits [| 6. |] (floats (Quantity.value jansky s.flux)));
      prop
        "a list of optional quantities reports, per element, its presence, \
         then its unit's text and its payload"
        Gen.(list ~size:(int_range 0 4) (option units))
        (fun us ->
          let s = Nx.Ptree.(list (option q32)) in
          let xs =
            List.map (Option.map (fun u -> Quantity.v u (f32 [| 1. |]))) us
          in
          let at i = Nx.Ptree.Path.v [ Index i ] in
          let element i u =
            let present = Nx.Ptree.Report (at i, Present (Option.is_some u)) in
            match u with
            | None -> [ present ]
            | Some u ->
                [ present; Report (at i, Case (Unit.to_string u)); Leaf (at i) ]
          in
          cover "a quantity is present" (List.exists Option.is_some us);
          equal (list visit)
            (Report (root, Length (List.length us))
            :: List.concat (List.mapi element us))
            (Nx.Ptree.visits s xs));
      prop "two spellings of a unit report the same visits" units (fun u ->
          equal (list visit)
            (Nx.Ptree.visits q32 (Quantity.v u (f32 [| 1. |])))
            (Nx.Ptree.visits q32 (Quantity.v (rebuild u) (f32 [| 1. |]))));
      test "a quantity of a pair reports its unit once, then the pair's leaves"
        (fun () ->
          let s =
            Nx.Ptree.nest (module Quantity) Nx.Ptree.(pair tensor tensor)
          in
          let q =
            Nx.Ptree.Payload.map
              (module Quantity)
              (fun _ x -> (x, Nx.neg x))
              (Quantity.v km (f32 [| 1. |]))
          in
          let at i = Nx.Ptree.Path.v [ Index i ] in
          equal (list visit)
            [ Report (root, Case "1e3 m"); Leaf (at 0); Leaf (at 1) ]
            (Nx.Ptree.visits s q));
    ]

(* Constructors and units *)

let constructors =
  group "Quantity.v"
    [
      test "is the payload in the unit" (fun () ->
          let x = f32 [| 1.; 2. |] in
          let q = Quantity.v km x in
          equal unit km (Quantity.unit q);
          is_payload x (Quantity.value km q));
      test "refuses a bool tensor" (fun () ->
          raises (Invalid_argument "Quantity.v: a bool tensor has no unit")
            (fun () -> Quantity.v km (Nx.create Nx.bool [| 1 |] [| true |])));
      test "refuses a bit tensor" (fun () ->
          raises (Invalid_argument "Quantity.v: a bit tensor has no unit")
            (fun () ->
              Quantity.v km (Nx.create Nx.bit [| 9 |] (Array.make 9 true))));
      test "times and per change the unit and keep the payload" (fun () ->
          let x = f32 [| 2. |] in
          let q = Quantity.v km x in
          let t = Quantity.times Unit.second q in
          let p = Quantity.per Unit.second q in
          equal unit Unit.(km * second) (Quantity.unit t);
          equal unit Unit.(km / second) (Quantity.unit p);
          is_payload x (Quantity.value (Quantity.unit t) t);
          is_payload x (Quantity.value (Quantity.unit p) p));
      test "times and per accept any payload" (fun () ->
          let q =
            Quantity.times km
              (Nx.Ptree.Payload.map
                 (module Quantity)
                 (fun _ _ -> "label")
                 (Quantity.v Unit.one (f32 [| 1. |])))
          in
          equal unit km (Quantity.unit q));
    ]

(* Conversion *)

(* Dimensionless numbers to scale a unit by: integers, powers of ten, π, roots
   and their reciprocals, and factors past each float format's range. *)
let numbers =
  Unit.
    [
      one;
      int 1000;
      int 3 / int 1000;
      decimal "1e-35";
      decimal "1e40";
      pi;
      root 3 (int 2);
      (pi ** -2) / int 7;
      int 2 ** 200;
      int 2 ** -149;
      int 2 ** -150;
      decimal "65520";
    ]

let payloads =
  [|
    0.;
    -0.;
    1.;
    -1.;
    0.1;
    3.5e38;
    1e-45;
    Float.nan;
    Float.infinity;
    Float.neg_infinity;
    65504.;
    1e-7;
  |]

let number = Gen.of_list ~pp:Unit.pp numbers

(* [one_multiply d x u w] states that [x] of dtype [d] in [u], read in [w], is
   itself, [x] times the rounded factor, or raises as the factor does. *)
let one_multiply (type b) (d : (float, b) Nx.dtype) x u w =
  let x = Nx.cast d x in
  let q = Quantity.v u x in
  if Unit.equal u w then is_payload x (Quantity.value w q)
  else
    match Unit.ratio d u w with
    | f ->
        cover "a conversion multiplies" true;
        equal bits (floats (Nx.mul_s x f)) (floats (Quantity.value w q))
    | exception Invalid_argument msg ->
        cover "a conversion raises" true;
        raises
          (Invalid_argument (renamed "Quantity.value" msg))
          (fun () -> Quantity.value w q)

let conversion =
  group "A conversion is one rounded multiply, or it raises"
    [
      prop "value multiplies by the rounded factor, or raises as ratio does"
        Gen.(pair units number)
        (fun (u, n) ->
          let x = f64 payloads in
          let w =
            match within (fun () -> Unit.(u * n)) with
            | Some w -> w
            | None -> reject ()
          in
          one_multiply Nx.float64 x u w;
          one_multiply Nx.float32 x u w;
          one_multiply Nx.float16 x u w;
          one_multiply Nx.bfloat16 x u w;
          one_multiply Nx.float8_e4m3 x u w);
      test "km to m multiplies by 1000" (fun () ->
          equal bits [| 1000.; -2500.; -0. |]
            (floats
               (Quantity.value Unit.metre
                  (Quantity.v km (f32 [| 1.; -2.5; -0. |])))));
      test "hours to seconds multiplies by 3600" (fun () ->
          equal bits [| 7200. |]
            (floats
               (Quantity.value Unit.second
                  (Quantity.v Unit.hour (f64 [| 2. |])))));
      test "degrees to radians multiplies by pi/180 rounded once" (fun () ->
          equal bits [| 0x1.1df46a2529d39p-6 |]
            (floats
               (Quantity.value Unit.radian
                  (Quantity.v Unit.degree (f64 [| 1. |])))));
      test "a value in its own unit is the payload itself" (fun () ->
          let x = f32 [| Float.nan |] in
          is_payload x (Quantity.value km (Quantity.v km x)));
      test "units that do not convert raise naming their quotient" (fun () ->
          let electron = Unit.symbol "electron" in
          raises
            (Invalid_argument
               "Quantity.value: electron s^-1 does not convert to 1e-32 kg \
                s^-2: their quotient keeps electron kg^-1 s") (fun () ->
              Quantity.value
                Unit.(decimal "1e-32" * kilogram / (second ** 2))
                (Quantity.v Unit.(electron / second) (f32 [| 1. |]))));
      test "a factor that is 0 in the payload's dtype raises" (fun () ->
          let u = Unit.(decimal "1e-35" * kilogram / (second ** 2)) in
          raises
            (Invalid_argument
               "Quantity.value: the factor from 1e-35 kg s^-2 to kg s^-2 is \
                1e-35, which is 0 in float16") (fun () ->
              Quantity.value
                Unit.(kilogram / (second ** 2))
                (Quantity.v u (Nx.create Nx.float16 [| 1 |] [| 1. |]))));
      test "a bool payload raises, even in its own unit" (fun () ->
          let q =
            Nx.Ptree.cast
              (module Quantity)
              Nx.bool
              (Quantity.v km (f32 [| 1. |]))
          in
          raises (Invalid_argument "Quantity.value: bool holds no factor")
            (fun () -> Quantity.value km q));
      test "convert is the value in the new unit, its errors naming convert"
        (fun () ->
          let q = Quantity.convert Unit.metre (Quantity.v km (f32 [| 2. |])) in
          equal unit Unit.metre (Quantity.unit q);
          equal bits [| 2000. |] (floats (Quantity.value Unit.metre q));
          raises_match
            (Exn.invalid_arg
               ~substring:"Quantity.convert: m does not convert to s")
            (fun () -> Quantity.convert Unit.second q));
    ]

(* Complex payloads *)

let c64 zs = Nx.create Nx.complex64 [| Array.length zs |] zs
let c128 zs = Nx.create Nx.complex128 [| Array.length zs |] zs

(* [parts zs] is the real and imaginary parts of [zs], compared as [float_exact]
   does: NaN equals NaN and [0.] differs from [-0.]. *)
let parts zs = Array.map (fun (z : Complex.t) -> (z.re, z.im)) zs
let complex_w = array (pair float_exact float_exact)

let qc64 : Nx.complex64_t Quantity.t Nx.Ptree.t =
  Nx.Ptree.instantiate (module Quantity)

let zs =
  Complex.
    [|
      { re = Float.infinity; im = 1. };
      { re = 1.; im = Float.neg_infinity };
      { re = -0.; im = 2. };
      { re = Float.nan; im = 0. };
    |]

let scaled =
  Complex.
    [|
      { re = Float.infinity; im = 1000. };
      { re = 1000.; im = Float.neg_infinity };
      { re = -0.; im = 2000. };
      { re = Float.nan; im = 0. };
    |]

let complex =
  group "Complex payloads"
    [
      test "complex64 scales each part by the real factor" (fun () ->
          equal complex_w (parts scaled)
            (parts
               (Nx.to_array
                  (Quantity.value Unit.metre (Quantity.v km (c64 zs))))));
      xfail ~reason:"rune's jit compiles no complex tensor"
        (test "complex64 converts under jit as eager does" (fun () ->
             let value =
               Rune.jit
                 Nx.Ptree.(qc64 @-> returns tensor)
                 (Quantity.value Unit.metre)
             in
             equal complex_w (parts scaled)
               (parts (Nx.to_array (value (Quantity.v km (c64 zs)))))));
      test "complex64 converts under vmap as eager does" (fun () ->
          let convert =
            Rune.vmap
              Nx.Ptree.(qc64 @-> returns qc64)
              (Quantity.convert Unit.metre)
          in
          let q = Quantity.v km (Nx.reshape [| 2; 2 |] (c64 zs)) in
          let r = convert q in
          equal unit Unit.metre (Quantity.unit r);
          equal complex_w (parts scaled)
            (parts (Nx.to_array (Quantity.value Unit.metre r))));
      test "complex128 scales each part by the real factor" (fun () ->
          equal complex_w (parts scaled)
            (parts
               (Nx.to_array
                  (Quantity.value Unit.metre (Quantity.v km (c128 zs))))));
    ]

(* Conversion against the mpmath goldens *)

(* Each golden line is a unit, a product of primes and pi, and its number
   rounded to a float format: a factor, or 0, a subnormal or an overflow, which
   a conversion refuses. A float payload of that format, and a complex payload
   of that component, read in 1 is the payload times the factor or raises. *)

type golden_outcome = Factor of float | Refused of string
type float_dtype = F : (float, 'b) Nx.dtype -> float_dtype
type complex_dtype = C : (Complex.t, 'b) Nx.dtype -> complex_dtype

let golden_dtype = function
  | "float64" -> (F Nx.float64, Some (C Nx.complex128))
  | "float32" -> (F Nx.float32, Some (C Nx.complex64))
  | "float16" -> (F Nx.float16, None)
  | "bfloat16" -> (F Nx.bfloat16, None)
  | "float8_e4m3" -> (F Nx.float8_e4m3, None)
  | "float8_e5m2" -> (F Nx.float8_e5m2, None)
  | name -> failwith ("unknown golden format " ^ name)

let golden_factor token =
  let split c s =
    match String.index_opt s c with
    | None -> None
    | Some i ->
        Some (String.sub s 0 i, String.sub s (i + 1) (String.length s - i - 1))
  in
  let base, e = Option.value (split '^' token) ~default:(token, "1") in
  let n, d =
    match split '/' e with
    | None -> (int_of_string e, 1)
    | Some (n, d) -> (int_of_string n, int_of_string d)
  in
  ((if base = "pi" then Pi else Int (int_of_string base)), n, d)

let golden_outcome = function
  | "zero" -> Refused "which is 0 in"
  | "subnormal" -> Refused "which is subnormal in"
  | "overflow" -> Refused "which overflows"
  | hex -> Factor (float_of_string hex)

let golden_lines =
  In_channel.with_open_text "golden/ratio.golden" In_channel.input_lines
  |> List.map (fun line ->
      match List.rev (String.split_on_char ' ' line) with
      | result :: "=" :: rest -> (
          match List.rev rest with
          | name :: tokens ->
              (line, name, List.map golden_factor tokens, golden_outcome result)
          | [] -> failwith ("malformed golden line: " ^ line))
      | _ -> failwith ("malformed golden line: " ^ line))

let specials = Float.[| 1.; -1.; 0.; -0.; infinity; neg_infinity; nan |]

let complex_specials =
  Complex.
    [|
      { re = 1.; im = -1. };
      { re = -0.; im = 0. };
      { re = Float.infinity; im = Float.nan };
      { re = Float.neg_infinity; im = 1. };
      { re = Float.nan; im = -0. };
    |]

(* [refusal u which d] is the message of [Quantity.value Unit.one] refusing a
   payload of dtype [d] in [u]. *)
let refusal u which d =
  Printf.sprintf "Quantity.value: the factor from %s to %s is %s, %s %s"
    (Unit.to_string u) (Unit.to_string Unit.one) (Unit.to_string u) which
    (Nx_dtype.to_string d)

let golden_real (type b) (d : (float, b) Nx.dtype) u outcome =
  let x = Nx.create d [| Array.length specials |] specials in
  let q = Quantity.v u x in
  match outcome with
  | Factor f ->
      (* Each special times a value of [d] is exact in float64 and in [d]. *)
      equal ~msg:(Nx_dtype.to_string d) bits
        (Array.map (fun x -> x *. f) (floats x))
        (floats (Quantity.value Unit.one q))
  | Refused which ->
      raises
        (Invalid_argument (refusal u which d))
        (fun () -> Quantity.value Unit.one q)

let golden_complex (type b) (d : (Complex.t, b) Nx.dtype) u outcome =
  let z = Nx.create d [| Array.length complex_specials |] complex_specials in
  let q = Quantity.v u z in
  match outcome with
  | Factor f ->
      equal ~msg:(Nx_dtype.to_string d) complex_w
        (Array.map
           (fun (re, im) -> (re *. f, im *. f))
           (parts complex_specials))
        (parts (Nx.to_array (Quantity.value Unit.one q)))
  | Refused which ->
      raises
        (Invalid_argument (refusal u which d))
        (fun () -> Quantity.value Unit.one q)

let goldens =
  cases
    ~name:(fun (line, _, _, _) -> line)
    "Conversion by the mpmath goldens' factors" golden_lines
    (fun (_, name, factors, outcome) ->
      let u = product factors in
      let F d, complex = golden_dtype name in
      golden_real d u outcome;
      Option.iter (fun (C c) -> golden_complex c u outcome) complex)

(* Errors *)

let errors =
  group "Every stated error"
    [
      test "a factor that is subnormal in the payload's dtype" (fun () ->
          raises
            (Invalid_argument
               "Quantity.value: the factor from 1e-5 to 1 is 1e-5, which is \
                subnormal in float16") (fun () ->
              Quantity.value Unit.one
                (Quantity.v
                   Unit.(int 10 ** -5)
                   (Nx.create Nx.float16 [| 1 |] [| 1. |]))));
      test "a factor at the payload's overflow tie" (fun () ->
          raises
            (Invalid_argument
               "Quantity.value: the factor from 6552e1 to 1 is 6552e1, which \
                overflows float16") (fun () ->
              Quantity.value Unit.one
                (Quantity.v (Unit.int 65520)
                   (Nx.create Nx.float16 [| 1 |] [| 1. |]))));
      test "a factor whose evaluation is past the budget" (fun () ->
          let u =
            Unit.((int 16777289 ** 1000000) / (int 16777259 ** 1000000))
          in
          raises
            (Invalid_argument
               "Quantity.value: the factor from 16777259^-1000000 \
                16777289^1000000 to 1 is 16777259^-1000000 16777289^1000000, \
                whose evaluation needs a natural wider than 65536 bits")
            (fun () -> Quantity.value Unit.one (Quantity.v u (f64 [| 1. |]))));
      test "a quotient whose exponent leaves int" (fun () ->
          raises
            (Invalid_argument "Quantity.value: the exponent of pi leaves int")
            (fun () ->
              Quantity.value
                Unit.(pi ** -1)
                (Quantity.v Unit.(pi ** max_int) (f64 [| 1. |]))));
      test "convert of units that do not convert" (fun () ->
          raises
            (Invalid_argument
               "Quantity.convert: m does not convert to s: their quotient \
                keeps m s^-1") (fun () ->
              Quantity.convert Unit.second
                (Quantity.v Unit.metre (f32 [| 1. |]))));
      test "sub of units that do not convert" (fun () ->
          raises
            (Invalid_argument
               "Quantity.sub: s does not convert to m: their quotient keeps \
                m^-1 s") (fun () ->
              Quantity.sub
                (Quantity.v Unit.metre (f32 [| 1. |]))
                (Quantity.v Unit.second (f32 [| 1. |]))));
      test "map2 of units that do not convert" (fun () ->
          raises
            (Invalid_argument
               "Quantity.map2: s does not convert to m: their quotient keeps \
                m^-1 s") (fun () ->
              Quantity.map2 Nx.add
                (Quantity.v Unit.metre (f32 [| 1. |]))
                (Quantity.v Unit.second (f32 [| 1. |]))));
    ]

(* Bool payloads *)

let as_bool q = Nx.Ptree.cast (module Quantity) Nx.bool q
let no_unit = Invalid_argument "Quantity.v: a bool tensor has no unit"

let bools =
  group "A bool tensor has no unit"
    [
      test "v refuses an empty bool tensor" (fun () ->
          raises no_unit (fun () ->
              Quantity.v km (Nx.create Nx.bool [| 0 |] [||])));
      test "v refuses a bool scalar" (fun () ->
          raises no_unit (fun () -> Quantity.v km (Nx.scalar Nx.bool false)));
      test "map refuses a function that returns an empty bool tensor" (fun () ->
          raises
            (Invalid_argument "Quantity.map: the function returns a bool tensor")
            (fun () ->
              Quantity.map
                (fun _ -> Nx.create Nx.bool [| 0 |] [||])
                (Quantity.v km (f32 [| 1. |]))));
      test "value raises on a bool payload a payload map built" (fun () ->
          let q =
            Nx.Ptree.Payload.map
              (module Quantity)
              (fun _ x -> Nx.isnan x)
              (Quantity.v km (f32 [| 1. |]))
          in
          raises (Invalid_argument "Quantity.value: bool holds no factor")
            (fun () -> Quantity.value Unit.metre q));
      test "convert raises on a bool payload, naming convert" (fun () ->
          raises (Invalid_argument "Quantity.convert: bool holds no factor")
            (fun () ->
              Quantity.convert km (as_bool (Quantity.v km (f32 [| 1. |])))));
      test "map refuses a function that returns a bit tensor" (fun () ->
          raises
            (Invalid_argument "Quantity.map: the function returns a bit tensor")
            (fun () ->
              Quantity.map
                (fun x -> Nx.cast Nx.bit (Nx.isnan x))
                (Quantity.v km (f32 [| 1. |]))));
      test "value and convert raise on a bit payload" (fun () ->
          let q =
            Nx.Ptree.cast
              (module Quantity)
              Nx.bit
              (Quantity.v km (f32 [| 1. |]))
          in
          raises (Invalid_argument "Quantity.value: bit holds no factor")
            (fun () -> Quantity.value km q);
          raises (Invalid_argument "Quantity.convert: bit holds no factor")
            (fun () -> Quantity.convert Unit.metre q));
      test "add, sub and map2 raise on bool payloads, each naming itself"
        (fun () ->
          let a = as_bool (Quantity.v km (f32 [| 1. |])) in
          let b = as_bool (Quantity.v km (f32 [| 0. |])) in
          raises (Invalid_argument "Quantity.add: bool holds no factor")
            (fun () -> Quantity.add a b);
          raises (Invalid_argument "Quantity.sub: bool holds no factor")
            (fun () -> Quantity.sub a b);
          raises (Invalid_argument "Quantity.map2: bool holds no factor")
            (fun () -> Quantity.map2 Nx.logical_and a b));
    ]

(* Integer payloads *)

(* An integer conversion by [f] holds the elements in the dtype's range divided
   by [f], rounded inwards: [hi] converts and [hi + 1] overflows, [lo] converts
   and [lo - 1] overflows. *)
type int_case =
  | Case : {
      dtype : ('a, 'b) Nx.dtype;
      from : Unit.t;
      ok : 'a array;  (** Elements that convert. *)
      product : 'a array;  (** Their products. *)
      bad : 'a;  (** An element that overflows. *)
      factor : string;
    }
      -> int_case

let two_m = Unit.(int 2 * metre)

let small dtype ~lo ~hi =
  Case
    {
      dtype;
      from = two_m;
      ok = [| lo; hi; 0 |];
      product = [| 2 * lo; 2 * hi; 0 |];
      bad = hi + 1;
      factor = "2";
    }

let int_cases =
  [
    small Nx.int8 ~lo:(-64) ~hi:63;
    Case
      {
        dtype = Nx.int8;
        from = two_m;
        ok = [| 1 |];
        product = [| 2 |];
        bad = -65;
        factor = "2";
      };
    small Nx.uint8 ~lo:0 ~hi:127;
    small Nx.int4 ~lo:(-4) ~hi:3;
    Case
      {
        dtype = Nx.int4;
        from = two_m;
        ok = [| 1 |];
        product = [| 2 |];
        bad = -5;
        factor = "2";
      };
    small Nx.uint4 ~lo:0 ~hi:7;
    (* -8 / 3 rounds inwards to -2: -2 converts to -6 and -3 overflows. *)
    Case
      {
        dtype = Nx.int4;
        from = Unit.(int 3 * metre);
        ok = [| -2; 2 |];
        product = [| -6; 6 |];
        bad = -3;
        factor = "3";
      };
    (* The largest factor int4 holds: only 1, 0 and -1 convert. *)
    Case
      {
        dtype = Nx.int4;
        from = Unit.(int 7 * metre);
        ok = [| 1; -1; 0 |];
        product = [| 7; -7; 0 |];
        bad = -2;
        factor = "7";
      };
    Case
      {
        dtype = Nx.uint4;
        from = Unit.(int 15 * metre);
        ok = [| 1; 0 |];
        product = [| 15; 0 |];
        bad = 2;
        factor = "15";
      };
    small Nx.int16 ~lo:(-16384) ~hi:16383;
    small Nx.uint16 ~lo:0 ~hi:32767;
    Case
      {
        dtype = Nx.int32;
        from = km;
        ok = [| -2147483l; 2147483l |];
        product = [| -2147483000l; 2147483000l |];
        bad = 2147484l;
        factor = "1000";
      };
    Case
      {
        dtype = Nx.int32;
        from = km;
        ok = [| 0l |];
        product = [| 0l |];
        bad = -2147484l;
        factor = "1000";
      };
    Case
      {
        dtype = Nx.uint32;
        from = km;
        ok = [| 4294967l |];
        product = [| -296l (* 4294967000 *) |];
        bad = 4294968l;
        factor = "1000";
      };
    Case
      {
        dtype = Nx.int64;
        from = km;
        ok = [| -9223372036854775L; 9223372036854775L |];
        product = [| -9223372036854775000L; 9223372036854775000L |];
        bad = 9223372036854776L;
        factor = "1000";
      };
    Case
      {
        dtype = Nx.uint64;
        from = km;
        ok = [| 18446744073709551L |];
        product = [| -616L (* 18446744073709551000 *) |];
        bad = 18446744073709552L;
        factor = "1000";
      };
    Case
      {
        dtype = Nx.int16;
        from = two_m;
        ok = [| 0 |];
        product = [| 0 |];
        bad = -16385;
        factor = "2";
      };
    (* The largest factor each signed dtype holds: only 1, 0 and -1 convert. *)
    Case
      {
        dtype = Nx.int8;
        from = Unit.(int 127 * metre);
        ok = [| 1; -1; 0 |];
        product = [| 127; -127; 0 |];
        bad = -2;
        factor = "127";
      };
    Case
      {
        dtype = Nx.int32;
        from = Unit.(int 2147483647 * metre);
        ok = [| 1l; -1l |];
        product = [| 2147483647l; -2147483647l |];
        bad = 2l;
        factor = "2147483647";
      };
    Case
      {
        dtype = Nx.int64;
        (* 2^63 - 1 *)
        from =
          Unit.(
            (int 7 ** 2)
            * int 73 * int 127 * int 337 * int 92737 * int 649657 * metre);
        ok = [| 1L; -1L |];
        product = [| Int64.max_int; Int64.neg Int64.max_int |];
        bad = -2L;
        factor = "9223372036854775807";
      };
    (* int64 by 2^62 reaches min_int exactly. *)
    Case
      {
        dtype = Nx.int64;
        from = Unit.((int 2 ** 62) * metre);
        ok = [| -2L; 1L |];
        product = [| Int64.min_int; 4611686018427387904L |];
        bad = 2L;
        factor = "4611686018427387904";
      };
    Case
      {
        dtype = Nx.int64;
        from = Unit.((int 2 ** 62) * metre);
        ok = [| 0L |];
        product = [| 0L |];
        bad = -3L;
        factor = "4611686018427387904";
      };
    Case
      {
        dtype = Nx.int64;
        from = two_m;
        ok = [| -4611686018427387904L; 4611686018427387903L |];
        product = [| Int64.min_int; Int64.pred Int64.max_int |];
        bad = Int64.min_int;
        factor = "2";
      };
    (* Unsigned payloads past the signed range: their elements and products are
       bit patterns, compared without sign. *)
    Case
      {
        dtype = Nx.uint32;
        from = two_m;
        ok = [| 2147483647l; 0l |];
        product = [| -2l (* 4294967294 *); 0l |];
        bad = Int32.min_int (* 2^31 *);
        factor = "2";
      };
    Case
      {
        dtype = Nx.uint32;
        from = two_m;
        ok = [| 0l |];
        product = [| 0l |];
        bad = -1l (* 2^32 - 1 *);
        factor = "2";
      };
    Case
      {
        dtype = Nx.uint32;
        from = Unit.(int 4294967295 * metre);
        ok = [| 1l; 0l |];
        product = [| -1l (* 4294967295 *); 0l |];
        bad = 2l;
        factor = "4294967295";
      };
    Case
      {
        dtype = Nx.uint64;
        from = two_m;
        ok = [| Int64.max_int; 0L |];
        product = [| -2L (* 2^64 - 2 *); 0L |];
        bad = Int64.min_int (* 2^63 *);
        factor = "2";
      };
    Case
      {
        dtype = Nx.uint64;
        from = two_m;
        ok = [| 0L |];
        product = [| 0L |];
        bad = -1L (* 2^64 - 1 *);
        factor = "2";
      };
    Case
      {
        dtype = Nx.uint64;
        (* 2^64 - 1 *)
        from = Unit.(int 4294967295 * int 4294967297 * metre);
        ok = [| 1L; 0L |];
        product = [| -1L; 0L |];
        bad = 2L;
        factor = "18446744073709551615";
      };
  ]

(* Factors one past each integer dtype's largest value. *)
type unheld = Unheld : ('a, 'b) Nx.dtype * Unit.t * 'a -> unheld

let unheld =
  Unit.
    [
      Unheld (Nx.int4, int 8, 0);
      Unheld (Nx.uint4, int 16, 0);
      Unheld (Nx.int8, int 128, 0);
      Unheld (Nx.uint8, int 256, 0);
      Unheld (Nx.int16, int 32768, 0);
      Unheld (Nx.uint16, int 65536, 0);
      Unheld (Nx.int32, int 2 ** 31, 0l);
      Unheld (Nx.uint32, int 2 ** 32, 0l);
      Unheld (Nx.int64, int 2 ** 63, 0L);
      Unheld (Nx.uint64, int 2 ** 64, 0L);
    ]

let unheld_name (Unheld (d, n, _)) =
  Format.asprintf "%s by %a" (Nx_dtype.to_string d) Unit.pp n

let unheld_case (Unheld (d, n, zero)) =
  let from = Unit.(n * metre) in
  raises
    (Invalid_argument
       (Printf.sprintf
          "Quantity.value: the factor from %s to m is %s, which %s does not \
           hold"
          (Unit.to_string from) (Unit.to_string n) (Nx_dtype.to_string d)))
    (fun () ->
      Quantity.value Unit.metre
        (Quantity.v from (Nx.create d [| 1 |] [| zero |])))

let int_name (Case c) =
  Format.asprintf "%s by %s, refusing %s"
    (Nx_dtype.to_string c.dtype)
    c.factor
    (Nx.to_string (Nx.create c.dtype [| 1 |] [| c.bad |]))

let article dt = if String.starts_with ~prefix:"int" dt then "an" else "a"

let int_case (Case c) =
  let vec xs = Nx.create c.dtype [| Array.length xs |] xs in
  let into = Unit.metre in
  let q = Quantity.v c.from (vec c.ok) in
  equal string
    (Nx.to_string (vec c.product))
    (Nx.to_string (Quantity.value into q));
  let n = Array.length c.ok in
  let dt = Nx_dtype.to_string c.dtype in
  raises
    (Invalid_argument
       (Printf.sprintf
          "Quantity.value: element [%d] of %s %s payload overflows converting \
           %s to m (factor %s)"
          n (article dt) dt (Unit.to_string c.from) c.factor))
    (fun () ->
      Quantity.value into
        (Quantity.v c.from (vec (Array.append c.ok [| c.bad |]))))

let integers =
  group "Integer payloads"
    [
      cases ~name:int_name "the range divided by the factor converts" int_cases
        int_case;
      cases ~name:unheld_name "a factor past the dtype raises" unheld
        unheld_case;
      test "add converts an int4 operand into the first's unit" (fun () ->
          equal (array int) [| 7; -8 |]
            (Nx.to_array
               (Quantity.value Unit.metre
                  (Quantity.add
                     (Quantity.v Unit.metre
                        (Nx.create Nx.int4 [| 2 |] [| 1; -2 |]))
                     (Quantity.v two_m (Nx.create Nx.int4 [| 2 |] [| 3; -3 |]))))));
      test "convert names itself in an overflow" (fun () ->
          raises
            (Invalid_argument
               "Quantity.convert: element [0] of an int8 payload overflows \
                converting 2 m to m (factor 2)") (fun () ->
              Quantity.convert Unit.metre
                (Quantity.v two_m (Nx.create Nx.int8 [| 2 |] [| 64; 0 |]))));
      test "map2 names itself for a factor that is not an integer" (fun () ->
          raises
            (Invalid_argument
               "Quantity.map2: the factor from m to 1e3 m is 1e-3, which is \
                not an integer") (fun () ->
              Quantity.map2 Nx.add
                (Quantity.v km (i32 [| 1l |]))
                (Quantity.v Unit.metre (i32 [| 1l |]))));
      test "jit raises the uint32 overflow at 2^31 that eager raises" (fun () ->
          let qu32 : Nx.uint32_t Quantity.t Nx.Ptree.t =
            Nx.Ptree.instantiate (module Quantity)
          in
          let value =
            Rune.jit
              Nx.Ptree.(qu32 @-> returns tensor)
              (Quantity.value Unit.metre)
          in
          let q =
            Quantity.v two_m
              (Nx.create Nx.uint32 [| 2 |] [| 2147483647l; Int32.min_int |])
          in
          let msg =
            "Quantity.value: element [1] of a uint32 payload overflows \
             converting 2 m to m (factor 2)"
          in
          raises (Invalid_argument msg) (fun () -> Quantity.value Unit.metre q);
          raises (Invalid_argument msg) (fun () -> value q));
      test "jit raises the int4 overflow that eager raises" (fun () ->
          let qi4 : Nx.int4_t Quantity.t Nx.Ptree.t =
            Nx.Ptree.instantiate (module Quantity)
          in
          let value =
            Rune.jit
              Nx.Ptree.(qi4 @-> returns tensor)
              (Quantity.value Unit.metre)
          in
          let q = Quantity.v two_m (Nx.create Nx.int4 [| 3 |] [| 3; -4; 4 |]) in
          let msg =
            "Quantity.value: element [2] of an int4 payload overflows \
             converting 2 m to m (factor 2)"
          in
          raises (Invalid_argument msg) (fun () -> Quantity.value Unit.metre q);
          raises (Invalid_argument msg) (fun () -> value q));
      test "a scalar that overflows is named as the payload" (fun () ->
          raises
            (Invalid_argument
               "Quantity.value: an int32 payload overflows converting 1e3 m to \
                m (factor 1000)") (fun () ->
              Quantity.value Unit.metre
                (Quantity.v km (Nx.scalar Nx.int32 3_000_000l))));
      test "the index of a matrix element is its row and column" (fun () ->
          raises
            (Invalid_argument
               "Quantity.value: element [1; 0] of an int32 payload overflows \
                converting 1e3 m to m (factor 1000)") (fun () ->
              Quantity.value Unit.metre
                (Quantity.v km
                   (Nx.create Nx.int32 [| 2; 2 |]
                      [| 1l; 2l; 3_000_000l; -3_000_000l |]))));
      test "a factor that is not an integer raises" (fun () ->
          raises
            (Invalid_argument
               "Quantity.value: the factor from m to 1e3 m is 1e-3, which is \
                not an integer") (fun () ->
              Quantity.value km (Quantity.v Unit.metre (i32 [| 1000l |]))));
      test "a factor the dtype does not hold raises" (fun () ->
          raises
            (Invalid_argument
               "Quantity.value: the factor from 1e3 m to m is 1e3, which int8 \
                does not hold") (fun () ->
              Quantity.value Unit.metre
                (Quantity.v km (Nx.create Nx.int8 [| 1 |] [| 0 |]))));
    ]

(* Maps *)

let maps =
  group "Maps"
    [
      test "map applies the function in the quantity's unit" (fun () ->
          let q = Quantity.map Nx.sum (Quantity.v km (f32 [| 1.; 2. |])) in
          equal unit km (Quantity.unit q);
          equal bits [| 3. |]
            (floats (Nx.reshape [| 1 |] (Quantity.value km q))));
      test "map refuses a function that returns a bool tensor" (fun () ->
          raises
            (Invalid_argument "Quantity.map: the function returns a bool tensor")
            (fun () ->
              Quantity.map (fun x -> Nx.isnan x) (Quantity.v km (f32 [| 1. |]))));
      test "map2 converts its second argument to the first's unit" (fun () ->
          let q =
            Quantity.map2 Nx.maximum
              (Quantity.v km (f32 [| 1.; 1. |]))
              (Quantity.v Unit.metre (f32 [| 500.; 2000. |]))
          in
          equal unit km (Quantity.unit q);
          equal bits [| 1.; 2. |] (floats (Quantity.value km q)));
      test "add converts to the first unit" (fun () ->
          let q =
            Quantity.add
              (Quantity.v km (f64 [| 1. |]))
              (Quantity.v Unit.metre (f64 [| 1. |]))
          in
          equal unit km (Quantity.unit q);
          equal bits [| 1.001 |] (floats (Quantity.value km q)));
      test "sub converts to the first unit" (fun () ->
          let q =
            Quantity.sub
              (Quantity.v Unit.metre (f64 [| 1. |]))
              (Quantity.v km (f64 [| 1. |]))
          in
          equal unit Unit.metre (Quantity.unit q);
          equal bits [| -999. |] (floats (Quantity.value Unit.metre q)));
      test "add of two data sets' symbols raises" (fun () ->
          let pix scope = Unit.scoped ~scope "pix" in
          raises
            (Invalid_argument
               "Quantity.add: pix{sha256:9f2c41#SCI} does not convert to \
                pix{sha256:07ab3e#SCI}: their quotient keeps \
                pix{sha256:07ab3e#SCI}^-1 pix{sha256:9f2c41#SCI}") (fun () ->
              Quantity.add
                (Quantity.v (pix "sha256:07ab3e#SCI") (f32 [| 1. |]))
                (Quantity.v (pix "sha256:9f2c41#SCI") (f32 [| 1. |]))));
      test "an integer add converts only to the finer unit" (fun () ->
          let a = Quantity.v Unit.metre (i32 [| 1l |]) in
          let b = Quantity.v km (i32 [| 2l |]) in
          equal (array int32) [| 2001l |]
            (Nx.to_array (Quantity.value Unit.metre (Quantity.add a b)));
          raises
            (Invalid_argument
               "Quantity.add: the factor from m to 1e3 m is 1e-3, which is not \
                an integer") (fun () -> Quantity.add b a));
      test "map2 names itself in its errors" (fun () ->
          raises_match
            (Exn.invalid_arg ~substring:"Quantity.map2: s does not convert to m")
            (fun () ->
              Quantity.map2 Nx.add
                (Quantity.v Unit.metre (f32 [| 1. |]))
                (Quantity.v Unit.second (f32 [| 1. |]))));
    ]

(* Algebra *)

let algebra =
  group "Algebra"
    [
      test "mul multiplies units and payloads" (fun () ->
          let q =
            Quantity.mul
              (Quantity.v km (f32 [| 3. |]))
              (Quantity.v Unit.hertz (f32 [| 2. |]))
          in
          equal unit Unit.(km / second) (Quantity.unit q);
          equal bits [| 6. |] (floats (Quantity.value Unit.(km / second) q)));
      test "div divides units and payloads, truncating integers" (fun () ->
          let q =
            Quantity.div
              (Quantity.v km (i32 [| -7l |]))
              (Quantity.v Unit.second (i32 [| 2l |]))
          in
          equal unit Unit.(km / second) (Quantity.unit q);
          equal (array int32) [| -3l |]
            (Nx.to_array (Quantity.value Unit.(km / second) q)));
      test "pow raises unit and payload" (fun () ->
          let q = Quantity.pow 3 (Quantity.v km (f32 [| -2.; 0.5 |])) in
          equal unit Unit.(km ** 3) (Quantity.unit q);
          equal bits [| -8.; 0.125 |] (floats (Quantity.value Unit.(km ** 3) q)));
      test "pow 0 is one in the unit 1, NaN included" (fun () ->
          let q = Quantity.pow 0 (Quantity.v km (f32 [| Float.nan; 0. |])) in
          equal unit Unit.one (Quantity.unit q);
          equal bits [| 1.; 1. |] (floats (Quantity.value Unit.one q)));
      test "a negative pow is the reciprocal" (fun () ->
          let q = Quantity.pow (-2) (Quantity.v km (f32 [| -2.; 0. |])) in
          equal unit Unit.(km ** -2) (Quantity.unit q);
          equal bits [| 0.25; Float.infinity |]
            (floats (Quantity.value Unit.(km ** -2) q)));
      test "pow keeps an odd exponent past the dtype's integers" (fun () ->
          (* 17 is not a float8_e4m3 value, nor 2049 a float16 one. *)
          let e4m3 = Nx.create Nx.float8_e4m3 [| 1 |] [| -1. |] in
          let h = Nx.create Nx.float16 [| 1 |] [| -1. |] in
          equal bits [| -1. |]
            (floats
               (Quantity.value Unit.one
                  (Quantity.pow 17 (Quantity.v Unit.one e4m3))));
          equal bits [| -1. |]
            (floats
               (Quantity.value Unit.one
                  (Quantity.pow 2049 (Quantity.v Unit.one h)))));
      test "an integer pow wraps" (fun () ->
          let q =
            Quantity.pow 3 (Quantity.v Unit.one (i32 [| 100000l; -2l |]))
          in
          equal (array int32) [| -1530494976l; -8l |]
            (Nx.to_array (Quantity.value Unit.one q)));
      test "a negative pow of an integer payload raises" (fun () ->
          raises
            (Invalid_argument
               "Quantity.pow: -1 is negative and the payload is int32")
            (fun () -> Quantity.pow (-1) (Quantity.v km (i32 [| 1l |]))));
      test "root of an odd order is the real root, negative included" (fun () ->
          let q =
            Quantity.root 3 (Quantity.v Unit.(km ** 3) (f64 [| -8.; 8. |]))
          in
          equal unit km (Quantity.unit q);
          equal bits [| -2.; 2. |] (floats (Quantity.value km q)));
      test "root of a float narrower than float32 does not round the exponent"
        (fun () ->
          (* 27 is 28 in float8_e4m3, whose cube root 3.04 rounds to 3; 0.001 is
             0.00100040435791015625 in float16, whose cube root 0.100013 rounds
             to 0.10003662109375. *)
          let root d x =
            floats
              (Quantity.value Unit.one
                 (Quantity.root 3
                    (Quantity.v Unit.one (Nx.create d [| 1 |] [| x |]))))
          in
          equal bits [| 3. |] (root Nx.float8_e4m3 27.);
          equal bits [| 0.10003662109375 |] (root Nx.float16 0.001));
      test "root passes NaN and -0 as Nx.pow gives them" (fun () ->
          let x = f64 [| Float.nan; -0. |] in
          equal bits
            (floats (Nx.pow_s x (1. /. 3.)))
            (floats
               (Quantity.value Unit.one
                  (Quantity.root 3 (Quantity.v Unit.one x))));
          equal bits
            (floats (Nx.pow_s x 0.5))
            (floats
               (Quantity.value Unit.one
                  (Quantity.root 2 (Quantity.v Unit.one x)))));
      test "root of an even order of a negative element raises" (fun () ->
          raises
            (Invalid_argument
               "Quantity.root: element [1] of a float32 payload is below 0, \
                whose root of order 2 is not real") (fun () ->
              Quantity.root 2 (Quantity.v Unit.hertz (f32 [| 4.; -1. |]))));
      test "root of order 0 raises" (fun () ->
          raises (Invalid_argument "Quantity.root: 0 is below 1") (fun () ->
              Quantity.root 0 (Quantity.v km (f32 [| 1. |]))));
    ]

(* Hostile exponents *)

let inf = Float.infinity
let ninf = Float.neg_infinity
let nan = Float.nan

(* The float dtypes whose products overflow to infinity. float8_e5m2 holds
   infinity but its products saturate, and float8_e4m3 holds none. *)
let ieee_dtypes = [ F Nx.float64; F Nx.float32; F Nx.float16; F Nx.bfloat16 ]
let infinite_dtypes = ieee_dtypes @ [ F Nx.float8_e5m2 ]

let in_dtype (type b) (d : (float, b) Nx.dtype) xs =
  Nx.create d [| Array.length xs |] xs

(* [powers dtypes n xs ys] states that [pow n] takes [xs] to [ys] in [Unit.one]
   in each of [dtypes]. *)
let powers dtypes n xs ys =
  List.iter
    (fun (F d) ->
      let q = Quantity.pow n (Quantity.v Unit.one (in_dtype d xs)) in
      equal unit Unit.one (Quantity.unit q);
      equal ~msg:(Nx_dtype.to_string d) bits ys
        (floats (Quantity.value Unit.one q)))
    dtypes

(* [roots dtypes n u xs ys] states that [root n] takes [xs] in [u] to [ys] in
   [Unit.root n u] in each of [dtypes]. *)
let roots dtypes n u xs ys =
  List.iter
    (fun (F d) ->
      let q = Quantity.root n (Quantity.v u (in_dtype d xs)) in
      let msg = Nx_dtype.to_string d in
      equal ~msg unit (Unit.root n u) (Quantity.unit q);
      equal ~msg bits ys (floats (Quantity.value (Unit.root n u) q)))
    dtypes

(* An integer's inverse modulo 2^k: for odd [a], a^(2^62 - 1) is a^-1 modulo 2^k
   for k <= 64, since a's order modulo 2^k divides 2^62. 3^-1 is 0xAB..AB and
   5^-1 is 0xCD..CD. *)
let int_powers =
  [
    test "int32" (fun () ->
        equal (array int32)
          [| 1l; -1l; 0l; -1431655765l; -858993459l; 0l |]
          (Nx.to_array
             (Quantity.value Unit.one
                (Quantity.pow max_int
                   (Quantity.v Unit.one (i32 [| 1l; -1l; 2l; 3l; 5l; 0l |]))))));
    test "int8 and uint8" (fun () ->
        let pow d xs =
          Nx.to_array
            (Quantity.value Unit.one
               (Quantity.pow max_int
                  (Quantity.v Unit.one (Nx.create d [| Array.length xs |] xs))))
        in
        equal (array int) [| -85; -51; -1; 0 |] (pow Nx.int8 [| 3; 5; -1; 2 |]);
        equal (array int) [| 171; 205 |] (pow Nx.uint8 [| 3; 5 |]));
    test "int16" (fun () ->
        equal (array int) [| -21845 |]
          (Nx.to_array
             (Quantity.value Unit.one
                (Quantity.pow max_int
                   (Quantity.v Unit.one (Nx.create Nx.int16 [| 1 |] [| 3 |]))))));
    test "int64 and uint64" (fun () ->
        let pow d =
          Nx.to_array
            (Quantity.value Unit.one
               (Quantity.pow max_int
                  (Quantity.v Unit.one (Nx.create d [| 2 |] [| 3L; 5L |]))))
        in
        let inverses = [| -6148914691236517205L; -3689348814741910323L |] in
        equal (array int64) inverses (pow Nx.int64);
        equal (array int64) inverses (pow Nx.uint64));
  ]

let exponents =
  group "Hostile exponents"
    [
      test "pow max_int keeps signs, overflows to infinity and underflows to 0"
        (fun () ->
          powers ieee_dtypes max_int
            [| 1.; -1.; 2.; 0.5; 0.; -0.; inf; ninf; nan |]
            [| 1.; -1.; inf; 0.; 0.; -0.; inf; ninf; nan |];
          powers [ F Nx.float8_e5m2 ] max_int
            [| 1.; -1.; 0.5; 0.; -0.; inf; ninf; nan |]
            [| 1.; -1.; 0.; 0.; -0.; inf; ninf; nan |];
          powers [ F Nx.float8_e4m3 ] max_int
            [| 1.; -1.; 0.5; 0.; -0.; nan |]
            [| 1.; -1.; 0.; 0.; -0.; nan |]);
      test "pow min_int is the reciprocal of an even power" (fun () ->
          powers ieee_dtypes min_int
            [| 1.; -1.; 2.; 0.5; 0.; -0.; inf; ninf; nan |]
            [| 1.; 1.; 0.; inf; inf; inf; 0.; 0.; nan |];
          powers [ F Nx.float8_e5m2 ] min_int
            [| 1.; -1.; 0.5; 0.; -0.; inf; ninf; nan |]
            [| 1.; 1.; inf; inf; inf; 0.; 0.; nan |];
          powers [ F Nx.float8_e4m3 ] min_int [| 1.; -1.; nan |]
            [| 1.; 1.; nan |]);
      test "pow -1 is the reciprocal, signed zeros and infinities included"
        (fun () ->
          powers infinite_dtypes (-1)
            [| 4.; 0.; -0.; inf; ninf; nan |]
            [| 0.25; inf; ninf; 0.; -0.; nan |]);
      test "pow 1 is the payload's values in the same unit" (fun () ->
          let xs = [| 3.; -0.; 0.; inf; ninf; nan; 0.1 |] in
          List.iter
            (fun (F d) ->
              let x = in_dtype d xs in
              let q = Quantity.pow 1 (Quantity.v km x) in
              equal unit km (Quantity.unit q);
              equal ~msg:(Nx_dtype.to_string d) bits (floats x)
                (floats (Quantity.value km q)))
            (F Nx.float8_e4m3 :: infinite_dtypes));
      prop "pow's unit is the algebra's power, or raises as the algebra does"
        Gen.(
          pair units
            (frequency
               [
                 (3, int_range (-4) 4);
                 ( 1,
                   of_list ~pp:Format.pp_print_int
                     [ max_int; min_int; max_int - 1; min_int + 1 ] );
               ]))
        (fun (u, n) ->
          let x = f32 [| 1. |] in
          match Unit.(u ** n) with
          | w ->
              cover "the power is a unit" true;
              equal unit w (Quantity.unit (Quantity.pow n (Quantity.v u x)))
          | exception e ->
              cover "the power leaves the algebra" true;
              raises e (fun () -> Quantity.pow n (Quantity.v u x)));
      test
        "a narrow float overflows as its dtype stores it: to infinity, or to \
         the float8 formats' largest value" (fun () ->
          powers [ F Nx.float16 ] 2 [| 300. |] [| inf |];
          powers [ F Nx.bfloat16 ] 2 [| 0x1p64 |] [| inf |];
          powers [ F Nx.float8_e5m2 ] 2 [| 256.; -256. |] [| 57344.; 57344. |];
          powers [ F Nx.float8_e4m3 ] 3 [| 16.; -16. |] [| 448.; -448. |]);
      group "pow max_int of an integer wraps to the inverse modulo 2^k"
        int_powers;
      test "pow min_int of an integer payload raises" (fun () ->
          raises
            (Invalid_argument
               "Quantity.pow: -4611686018427387904 is negative and the payload \
                is uint64") (fun () ->
              Quantity.pow min_int
                (Quantity.v Unit.one (Nx.create Nx.uint64 [| 1 |] [| 1L |]))));
      test "pow 0 of an integer payload is one" (fun () ->
          equal (array int32) [| 1l; 1l |]
            (Nx.to_array
               (Quantity.value Unit.one
                  (Quantity.pow 0 (Quantity.v km (i32 [| 0l; Int32.min_int |]))))));
      test "root min_int and root -1 raise" (fun () ->
          raises
            (Invalid_argument "Quantity.root: -4611686018427387904 is below 1")
            (fun () -> Quantity.root min_int (Quantity.v km (f32 [| 1. |])));
          raises (Invalid_argument "Quantity.root: -1 is below 1") (fun () ->
              Quantity.root (-1) (Quantity.v km (f32 [| 1. |]))));
      test "root 1 is the payload's values in the same unit" (fun () ->
          let xs = [| nan; -0.; 0.; ninf; inf; -3.; 0.5 |] in
          List.iter
            (fun (F d) ->
              let x = in_dtype d xs in
              roots [ F d ] 1 km xs (floats x))
            (F Nx.float8_e4m3 :: infinite_dtypes));
      test "root max_int, odd, is the real root, -0 taking +0 as pow gives"
        (fun () ->
          let u = Unit.(second ** 3) in
          roots infinite_dtypes max_int u
            [| 0.; -0.; 1.; -1.; inf; ninf; nan |]
            [| 0.; 0.; 1.; -1.; inf; ninf; nan |];
          roots [ F Nx.float8_e4m3 ] max_int u
            [| 0.; -0.; 1.; -1.; nan |]
            [| 0.; 0.; 1.; -1.; nan |]);
      test "root max_int - 1, even, passes 0, -0, infinity and NaN" (fun () ->
          let n = max_int - 1 in
          roots infinite_dtypes n km
            [| 0.; -0.; 1.; inf; nan |]
            [| 0.; 0.; 1.; inf; nan |];
          roots [ F Nx.float8_e4m3 ] n km [| 0.; -0.; 1.; nan |]
            [| 0.; 0.; 1.; nan |]);
      test "an even root of a negative scalar raises in every float dtype"
        (fun () ->
          List.iter
            (fun (F d) ->
              let dt = Nx_dtype.to_string d in
              raises
                (Invalid_argument
                   (Printf.sprintf
                      "Quantity.root: a %s payload is below 0, whose root of \
                       order 4611686018427387902 is not real"
                      dt))
                (fun () ->
                  Quantity.root (max_int - 1)
                    (Quantity.v km (Nx.scalar d (-1.)))))
            (F Nx.float8_e4m3 :: infinite_dtypes));
      test "an even root of -infinity raises naming its element" (fun () ->
          raises
            (Invalid_argument
               "Quantity.root: element [1] of a bfloat16 payload is below 0, \
                whose root of order 2 is not real") (fun () ->
              Quantity.root 2
                (Quantity.v km (in_dtype Nx.bfloat16 [| nan; ninf |]))));
      test "root of an exact power in every float dtype" (fun () ->
          roots
            (F Nx.float8_e4m3 :: infinite_dtypes)
            3
            Unit.(km ** 3)
            [| -8.; 8.; 1. |] [| -2.; 2.; 1. |];
          roots
            (F Nx.float8_e4m3 :: infinite_dtypes)
            2
            Unit.(km ** 2)
            [| 4.; 16.; 0.25 |] [| 2.; 4.; 0.5 |]);
    ]

(* Formatting *)

let formatting =
  group "Quantity.pp"
    [
      test "formats the payload then the unit's canonical text" (fun () ->
          expect
            (Format.asprintf "%a" Quantity.pp
               (Quantity.v Unit.(km / second) (f32 [| 1.; 2.5 |])))
          @@ __POS_OF__ {|[1, 2.5] 1e3 m s^-1|});
    ]

(* Rune *)

let rune =
  group "Under rune"
    [
      test "jit converts as eager does, tracing once per unit" (fun () ->
          let traces = ref 0 in
          let value =
            Rune.jit
              Nx.Ptree.(q32 @-> returns tensor)
              (fun q ->
                incr traces;
                Quantity.value Unit.metre q)
          in
          let x = f32 [| 1.; 0.1; 3.5e35 |] in
          List.iter
            (fun u ->
              let q = Quantity.v u x in
              equal bits
                (floats (Quantity.value Unit.metre q))
                (floats (value q)))
            Unit.[ km; centi metre; astronomical_unit; km ];
          equal int 3 !traces);
      prop "jit and eager agree bit for bit on a factor with pi"
        Gen.(array ~size:(constant 4) any_float)
        (let parsec = Unit.(int 648000 / pi * astronomical_unit) in
         let into = Unit.(parsec / mega (int 31557600 * second)) in
         let value =
           Rune.jit Nx.Ptree.(q32 @-> returns tensor) (Quantity.value into)
         in
         fun xs ->
           let q = Quantity.v Unit.(km / second) (f32 xs) in
           equal bits (floats (Quantity.value into q)) (floats (value q)));
      test "jit raises an integer overflow when the call returns" (fun () ->
          let value =
            Rune.jit
              Nx.Ptree.(qi32 @-> returns tensor)
              (Quantity.value Unit.metre)
          in
          raises
            (Invalid_argument
               "Quantity.value: element [1] of an int32 payload overflows \
                converting 1e3 m to m (factor 1000)") (fun () ->
              value (Quantity.v km (i32 [| 1l; 2147484l |]))));
      test "jit raises an even root of a negative element" (fun () ->
          let root =
            Rune.jit Nx.Ptree.(q32 @-> returns q32) (Quantity.root 2)
          in
          raises
            (Invalid_argument
               "Quantity.root: element [0] of a float32 payload is below 0, \
                whose root of order 2 is not real") (fun () ->
              root (Quantity.v Unit.hertz (f32 [| -4. |]))));
      test "vmap carries the unit and names an overflow by its lane's index"
        (fun () ->
          let f =
            Rune.vmap
              Nx.Ptree.(qi32 @-> returns qi32)
              (Quantity.convert Unit.metre)
          in
          let q =
            Quantity.v km (Nx.create Nx.int32 [| 2; 2 |] [| 1l; 2l; 3l; 4l |])
          in
          let r = f q in
          equal unit Unit.metre (Quantity.unit r);
          equal (array int32)
            [| 1000l; 2000l; 3000l; 4000l |]
            (Nx.to_array (Quantity.value Unit.metre r));
          raises
            (Invalid_argument
               "Quantity.convert: element [1] of an int32 payload overflows \
                converting 1e3 m to m (factor 1000)") (fun () ->
              f
                (Quantity.v km
                   (Nx.create Nx.int32 [| 2; 2 |] [| 1l; 2l; 3l; 3_000_000l |]))));
      test "scan refuses a carry whose unit changes" (fun () ->
          raises_match
            (Exn.invalid_arg
               ~substring:
                 {|case "m" in the carry the step returned, case "1e3 m" in the carry it received|})
            (fun () ->
              Rune.scan q32 Nx.Ptree.tensor Nx.Ptree.tensor
                ~f:(fun c x -> (Quantity.convert Unit.metre c, x))
                ~init:(Quantity.v km (Nx.scalar Nx.float32 1.))
                (f32 [| 1.; 2. |])));
      test "scan accepts a carry converted back to the initial unit" (fun () ->
          let init = Quantity.v km (Nx.scalar Nx.float32 1.) in
          let step c x =
            let c =
              Quantity.add
                (Quantity.convert Unit.metre c)
                (Quantity.v Unit.metre x)
            in
            (Quantity.convert (Quantity.unit init) c, x)
          in
          let c, _ =
            Rune.scan q32 Nx.Ptree.tensor Nx.Ptree.tensor ~f:step ~init
              (f32 [| 500.; 500. |])
          in
          equal unit km (Quantity.unit c);
          (* 1 km + 500 m + 500 m, rounded in float32 at each step. *)
          equal
            (array (float 1e-6))
            [| 2. |]
            (floats (Nx.reshape [| 1 |] (Quantity.value km c))));
      test "jvp tangents carry their primals' units" (fun () ->
          let x = Quantity.v km (f64 [| 3. |]) in
          let t = Quantity.v km (f64 [| 1. |]) in
          let y, dy = Rune.jvp q64 q64 (Quantity.pow 2) x t in
          equal unit Unit.(km ** 2) (Quantity.unit y);
          equal unit Unit.(km ** 2) (Quantity.unit dy);
          equal bits [| 6. |] (floats (Quantity.value Unit.(km ** 2) dy)));
      test "jvp refuses a tangent in another unit" (fun () ->
          let x = Quantity.v km (f64 [| 3. |]) in
          raises_match
            (Exn.invalid_arg
               ~substring:
                 {|case "1e3 m" in the parameters, case "m" in the tangents|})
            (fun () ->
              Rune.jvp q64 q64 (Quantity.pow 2) x
                (Quantity.convert Unit.metre x)));
      test "grad of bare tensors through a conversion is the factor" (fun () ->
          let loss x = Nx.sum (Quantity.value Unit.metre (Quantity.v km x)) in
          equal bits [| 1000.; 1000. |]
            (floats (Rune.grad' loss (f64 [| 1.; -2. |]))));
      test "grad through an odd root is finite below zero" (fun () ->
          let loss x =
            Nx.sum
              (Quantity.value Unit.one
                 (Quantity.root 3 (Quantity.v Unit.one x)))
          in
          (* d/dx x^(1/3) = 1/(3 x^(2/3)) is 1/12 at -8 and 8. *)
          equal
            (array (float 1e-15))
            [| 1. /. 12.; 1. /. 12. |]
            (floats (Rune.grad' loss (f64 [| -8.; 8. |]))));
    ]

(* Rune laws *)

(* Lengths whose factors to the metre are integers, decimals, pi and a root,
   each finite and normal in float32 and float64. *)
let lengths =
  Unit.
    [
      metre;
      km;
      centi metre;
      astronomical_unit;
      int 3 / int 1000 * metre;
      pi * metre;
      root 3 (int 2) * metre;
    ]

let length = Gen.of_list ~pp:Unit.pp lengths

(* [outcome f] is the unit and payload of [f ()], or the message it raised. *)
let outcome f =
  match f () with
  | q ->
      Ok
        ( Unit.to_string (Quantity.unit q),
          floats (Quantity.value (Quantity.unit q) q) )
  | exception Invalid_argument msg -> Error msg

type op = {
  name : string;
  w : float array testable;
  eager :
    Nx.float32_t Quantity.t ->
    Nx.float32_t Quantity.t ->
    Nx.float32_t Quantity.t;
  jitted :
    Nx.float32_t Quantity.t ->
    Nx.float32_t Quantity.t ->
    Nx.float32_t Quantity.t;
}

(* [ulps k] compares float32 arrays within [k] ulps relative, NaN equal to
   NaN. *)
let ulps k =
  let rel = Float.ldexp (Float.of_int k) (-23) in
  let close a b =
    a = b
    || (Float.is_nan a && Float.is_nan b)
    || Float.abs (a -. b) <= rel *. Float.max (Float.abs a) (Float.abs b)
  in
  array
    (Testable.make ~pp:(fun ppf x -> Format.fprintf ppf "%h" x) ~equal:close)

let op ?(w = bits) name f =
  {
    name;
    w;
    eager = f;
    jitted = Rune.jit Nx.Ptree.(q32 @-> q32 @-> returns q32) f;
  }

let ops =
  [
    op "convert" (fun a _ -> Quantity.convert Unit.metre a);
    op "add" Quantity.add;
    op "sub" Quantity.sub;
    op "mul" Quantity.mul;
    op "div" Quantity.div;
    op "map2" (Quantity.map2 (fun x y -> Nx.sub (Nx.mul_s x 2.) y));
    op "map" (fun a _ -> Quantity.map Nx.sum a);
    op "pow 3" (fun a _ -> Quantity.pow 3 a);
    op "pow -2" (fun a _ -> Quantity.pow (-2) a);
    (* Nx.pow's rounding is not specified, and may differ between eager and
       compiled kernels. *)
    op ~w:(ulps 2) "root 3" (fun a _ -> Quantity.root 3 a);
    op ~w:(ulps 2) "root 2" (fun a _ -> Quantity.root 2 a);
    op "times" (fun a b -> Quantity.times (Quantity.unit b) a);
    op "per" (fun a b -> Quantity.per (Quantity.unit b) a);
  ]

let pp_op ppf o = Format.pp_print_string ppf o.name
let row q i = Quantity.map (Nx.get [ i ]) q

(* [chain q] is a chain of the algebra on [x] in km, with odd roots of elements
   below zero: (1000 x m)^2 (x - 0.25)^(1/3) km^(1/3) in m^(7/3), whose factor
   from m^2 km^(1/3) is 10. *)
let m7_3 = Unit.(root 3 (metre ** 7))

let chain q =
  let shift = Quantity.v Unit.metre (Nx.full Nx.float64 [| 3 |] 250.) in
  Quantity.convert m7_3
    (Quantity.mul
       (Quantity.pow 2 (Quantity.convert Unit.metre q))
       (Quantity.root 3 (Quantity.sub q shift)))

let chain_at xs = floats (Quantity.value m7_3 (chain (Quantity.v km (f64 xs))))

(* Lengths in km at least 0.25 from the chain's kink at 0.25. *)
let away =
  Gen.(
    array ~size:(constant 3)
      (let+ x = float_range 0.5 3. and+ negative = bool in
       if negative then -.x else x))

(* [central f xs dx] is the derivative of [f] at [xs] along [dx] by a central
   difference. *)
let central f xs dx =
  let h = 1e-6 in
  let at s = f (Array.mapi (fun i x -> x +. (s *. h *. dx.(i))) xs) in
  Array.map2 (fun a b -> (a -. b) /. (2. *. h)) (at 1.) (at (-1.))

let close = array (float_rel ~rel:1e-6 ~abs:0.)

(* [scan_step c x] adds [x] to the carry in metres and returns the carry in its
   own unit, with their product. *)
let scan_step c x =
  ( Quantity.convert (Quantity.unit c)
      (Quantity.add (Quantity.convert Unit.metre c) x),
    Quantity.mul c x )

let scan init xs = Rune.scan q64 q64 q64 ~f:scan_step ~init xs

let scan_jitted =
  Rune.jit Nx.Ptree.(q64 @-> q64 @-> returns (pair q64 q64)) scan

let rune_laws =
  group "Rune laws"
    [
      prop "jit equals eager for every operation, raises included"
        Gen.(
          quad (of_list ~pp:pp_op ops) (pair length length)
            (array ~size:(constant 3) any_float)
            (array ~size:(constant 3) any_float))
        (fun (o, (u, w), xs, ys) ->
          let a = Quantity.v u (f32 xs) and b = Quantity.v w (f32 ys) in
          equal
            (result (pair string o.w) string)
            (outcome (fun () -> o.eager a b))
            (outcome (fun () -> o.jitted a b)));
      prop "vmap agrees with the stack of its rows"
        Gen.(
          triple (pair length length)
            (array ~size:(constant 6) float)
            (array ~size:(constant 6) float))
        (fun ((u, w), xs, ys) ->
          let m2 = Unit.(metre ** 2) in
          let f a b =
            Quantity.convert m2
              (Quantity.add (Quantity.mul a a) (Quantity.mul b b))
          in
          let batch xs = Nx.create Nx.float32 [| 3; 2 |] xs in
          let a = Quantity.v u (batch xs) and b = Quantity.v w (batch ys) in
          let r = Rune.vmap Nx.Ptree.(q32 @-> q32 @-> returns q32) f a b in
          let rows = List.init 3 (fun i -> f (row a i) (row b i)) in
          equal unit m2 (Quantity.unit r);
          equal bits
            (floats (Nx.stack (List.map (Quantity.value m2) rows)))
            (floats (Quantity.value m2 r)));
      prop "grad of bare tensors wrapped inside is the central difference" away
        (fun xs ->
          let loss x = Nx.sum (Quantity.value m7_3 (chain (Quantity.v km x))) in
          let g = Rune.grad' loss (f64 xs) in
          let sum xs = [| Array.fold_left ( +. ) 0. (chain_at xs) |] in
          let fd =
            Array.init 3 (fun i ->
                (central sum xs
                   (Array.init 3 (fun j -> if i = j then 1. else 0.))).(0))
          in
          equal close fd (floats g));
      prop "jvp is the central difference, in the result's unit"
        Gen.(pair away away)
        (fun (xs, dx) ->
          let y, dy =
            Rune.jvp q64 q64 chain
              (Quantity.v km (f64 xs))
              (Quantity.v km (f64 dx))
          in
          equal unit m7_3 (Quantity.unit y);
          equal unit m7_3 (Quantity.unit dy);
          equal close (central chain_at xs dx) (floats (Quantity.value m7_3 dy)));
      prop
        "scan carrying a quantity is the fold of its body, eager and compiled"
        Gen.(
          triple length
            (array ~size:(constant 2) float)
            (array ~size:(constant 8) float))
        (fun (u, init, xs) ->
          let init = Quantity.v u (f64 init) in
          let xs = Quantity.v Unit.metre (Nx.create Nx.float64 [| 4; 2 |] xs) in
          let fold_c, fold_ys =
            List.fold_left
              (fun (c, ys) i ->
                let c, y = scan_step c (row xs i) in
                (c, y :: ys))
              (init, []) (List.init 4 Fun.id)
          in
          let um = Unit.(u * metre) in
          let expected_ys =
            floats (Nx.stack (List.rev_map (Quantity.value um) fold_ys))
          in
          let check (c, ys) =
            equal unit u (Quantity.unit c);
            equal unit um (Quantity.unit ys);
            equal bits
              (floats (Quantity.value u fold_c))
              (floats (Quantity.value u c));
            equal bits expected_ys (floats (Quantity.value um ys))
          in
          check (scan init xs);
          check (scan_jitted init xs));
    ]

(* Gradients *)

(* A record of quantities in two dimensions, each in a drawn unit. *)
type 'a body = { size : 'a Quantity.t; temperature : 'a Quantity.t }

module Body = struct
  type 'a t = 'a body

  let walk c b =
    let open Nx.Ptree.Walk in
    let size = field c "size" Quantity.walk b.size in
    let temperature = field c "temperature" Quantity.walk b.temperature in
    { size; temperature }
end

let body : Nx.float64_t body Nx.Ptree.t = Nx.Ptree.instantiate (module Body)
let mk = Unit.(milli kelvin)
let temperatures = [ Unit.kelvin; mk ]
let temperature = Gen.of_list ~pp:Unit.pp temperatures
let payload q = floats (Quantity.value (Quantity.unit q) q)

let pp_body ppf b =
  Format.fprintf ppf "@[<v>size %a@,temperature %a@]" Quantity.pp b.size
    Quantity.pp b.temperature

(* Payloads of 0 or of a magnitude whose products with the lengths' factors stay
   normal. *)
let payloads =
  Gen.(
    array ~size:(constant 3)
      (one_of
         [
           constant 0.;
           (let+ x = float_range 1e-3 3. and+ negative = bool in
            if negative then -.x else x);
         ]))

(* Draws [n] bodies in one pair of units: a point and its directions. *)
let bodies n =
  let open Gen in
  with_pp
    (Format.pp_print_list pp_body)
    (let+ u = length
     and+ w = temperature
     and+ vs = list ~size:(constant n) (pair payloads payloads) in
     List.map
       (fun (s, t) ->
         { size = Quantity.v u (f64 s); temperature = Quantity.v w (f64 t) })
       vs)

(* Each field read in a unit of its own. *)
let radiated b =
  let r = Quantity.value Unit.metre b.size in
  let t = Quantity.value Unit.kelvin b.temperature in
  Nx.sum (Nx.mul (Nx.mul r r) (Nx.sin t))

(* Bilinear in the fields, so its derivative in one field does not depend on
   that field's payload: converting the field leaves the point's other
   coordinates, and the derivative per metre or kelvin, bit for bit. *)
let bilinear b =
  Nx.sum
    (Nx.mul
       (Quantity.value Unit.metre b.size)
       (Quantity.value Unit.kelvin b.temperature))

(* Each unit pair a field converts between: every pair of lengths, and kelvin
   and millikelvin both ways. *)
type change = Size of Unit.t * Unit.t | Temperature of Unit.t * Unit.t

let pp_change ppf = function
  | Size (u, v) -> Format.fprintf ppf "size %a -> %a" Unit.pp u Unit.pp v
  | Temperature (u, v) ->
      Format.fprintf ppf "temperature %a -> %a" Unit.pp u Unit.pp v

let changes =
  List.concat_map (fun u -> List.map (fun v -> Size (u, v)) lengths) lengths
  @ [ Temperature (Unit.kelvin, mk); Temperature (mk, Unit.kelvin) ]

(* [rel_ulps k] compares within [k] ulps of the larger magnitude. *)
let rel_ulps k =
  array (float_rel ~rel:(float_of_int k *. epsilon_float) ~abs:0.)

let gradients =
  group "Gradients"
    [
      prop "a gradient paired with a tangent is the derivative along it"
        (bodies 2) (fun bs ->
          let x, t = match bs with [ x; t ] -> (x, t) | _ -> assert false in
          let g = Rune.grad body radiated x in
          let _, d = Rune.jvp body Nx.Ptree.tensor radiated x t in
          let scale =
            Nx.Ptree.dot body Nx.float64
              (Nx.Ptree.map body (fun _ t -> Nx.abs t) g)
              (Nx.Ptree.map body (fun _ t -> Nx.abs t) t)
          in
          let tol = 1e3 *. epsilon_float *. (1. +. Nx.item [] scale) in
          equal (float tol) (Nx.item [] d)
            (Nx.item [] (Nx.Ptree.dot body Nx.float64 g t)));
      prop "a gradient is held in its parameters' units" (bodies 1) (fun bs ->
          let x = List.hd bs in
          let g = Rune.grad body radiated x in
          equal (list visit) (Nx.Ptree.visits body x) (Nx.Ptree.visits body g));
      (* [g] at [x] and [g'] at [x'], [x] with one field converted from [u] to
         [v] by the factor [r]: [g'] is the derivative per [v], [g / r], and
         [value v g] is [r g], [r^2] times it. A gradient and [value] are each
         within 1.5 ulps of their exact products with the units' factors, and
         each product or quotient by [r] rounds [r] and the result: 4 ulps for
         the first claim, 3 for the second and 8 for the third. *)
      prop "converting a parameter divides its gradient by the factor"
        Gen.(pair (of_list ~pp:pp_change changes) (bodies 1))
        (fun (change, bs) ->
          let x = List.hd bs in
          let x, x', get =
            match change with
            | Size (u, v) ->
                let x = { x with size = Quantity.convert u x.size } in
                (x, { x with size = Quantity.convert v x.size }, fun b -> b.size)
            | Temperature (u, v) ->
                let x =
                  { x with temperature = Quantity.convert u x.temperature }
                in
                ( x,
                  { x with temperature = Quantity.convert v x.temperature },
                  fun b -> b.temperature )
          in
          let u = Quantity.unit (get x) and v = Quantity.unit (get x') in
          let r = Unit.ratio Nx.float64 u v in
          let g = get (Rune.grad body bilinear x) in
          let g' = get (Rune.grad body bilinear x') in
          equal ~msg:"per v" (rel_ulps 4)
            (Array.map (fun p -> p /. r) (payload g))
            (payload g');
          equal ~msg:"value v" (rel_ulps 3)
            (Array.map (fun p -> p *. r) (payload g))
            (floats (Quantity.value v g));
          equal ~msg:"r^2 apart" (rel_ulps 8)
            (Array.map (fun p -> p *. r *. r) (payload g'))
            (floats (Quantity.value v g)));
    ]

(* Accuracy *)

(* Float formats, as IEEE 754 and the float8 specifications state them:
   precision in bits, least normal exponent, largest finite value. *)
type format = { dtype : dtype; prec : int; emin : int; max : float }
and dtype = D : (float, 'b) Nx.dtype -> dtype

let pp_format ppf { dtype = D d; _ } =
  Format.pp_print_string ppf (Nx_dtype.to_string d)

let float32 =
  {
    dtype = D Nx.float32;
    prec = 24;
    emin = -126;
    max = Float.ldexp 0xffffffp0 104;
  }

let narrow_formats =
  [
    float32;
    { dtype = D Nx.float16; prec = 11; emin = -14; max = 65504. };
    {
      dtype = D Nx.bfloat16;
      prec = 8;
      emin = -126;
      max = Float.ldexp 0xffp0 120;
    };
    { dtype = D Nx.float8_e4m3; prec = 4; emin = -6; max = 448. };
  ]

(* [ulp f x] is the unit in the last place of [x] in the format [f], its
   subnormals included. *)
let ulp f x =
  let _, e = Float.frexp x in
  Float.ldexp 1. (Int.max (e - 1) f.emin - f.prec + 1)

(* [read f x] is [x] rounded to [f] and read back. *)
let read { dtype = D d; _ } x = Nx.item [ 0 ] (Nx.create d [| 1 |] [| x |])

let ulps =
  Testable.make
    ~pp:(fun ppf x -> Format.fprintf ppf "%g ulps" x)
    ~equal:Float.equal
  |> Testable.with_compare Float.compare

let magnitude =
  Gen.(
    let+ m = float_range 1. 10.
    and+ e = int_range (-45) 38
    and+ negative = bool in
    let x = m *. (10. ** Float.of_int e) in
    if negative then -.x else x)

let accuracy =
  group "Accuracy"
    [
      prop "value is within 1.5 ulps of the exact product"
        Gen.(triple (of_list ~pp:pp_format narrow_formats) number magnitude)
        (fun (f, n, x) ->
          let u = Unit.(n * metre) in
          let x = read f x in
          let { dtype = D d; _ } = f in
          match Unit.ratio d u Unit.metre with
          | exception Invalid_argument _ -> reject ()
          | _ ->
              let exact = x *. Unit.ratio Nx.float64 u Unit.metre in
              if Float.abs exact > f.max then reject ();
              let r =
                Nx.item [ 0 ]
                  (Quantity.value Unit.metre
                     (Quantity.v u (Nx.create d [| 1 |] [| x |])))
              in
              cover "a subnormal result"
                (Float.abs exact < Float.ldexp 1. f.emin && exact <> 0.);
              at_most ulps ~than:1.5 (Float.abs (r -. exact) /. ulp f exact));
      prop "a float32 root is within one ulp of the float64 root"
        Gen.(pair (of_list ~pp:Format.pp_print_int [ 3; 4; 5; 7 ]) magnitude)
        ~examples:[ (3, 1e-30); (3, 1e-38); (3, -1e-38); (5, 3e38) ]
        (fun (n, x) ->
          let x = read float32 (if n land 1 = 0 then Float.abs x else x) in
          let reference =
            Float.copy_sign (Float.pow (Float.abs x) (1. /. Float.of_int n)) x
          in
          let r =
            Nx.item [ 0 ]
              (Quantity.value Unit.one
                 (Quantity.root n (Quantity.v Unit.one (f32 [| x |]))))
          in
          at_most ulps ~than:1.
            (Float.abs (r -. reference) /. ulp float32 reference));
    ]

let () =
  exit
    (run "Quantity"
       [
         structure;
         constructors;
         conversion;
         goldens;
         errors;
         complex;
         integers;
         bools;
         maps;
         algebra;
         exponents;
         accuracy;
         formatting;
         rune;
         rune_laws;
         gradients;
       ])
