(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx_cuda.mli: a kind has its nx_kinds.h function's bits, nx.cpu's, but
   for a NaN's sign and payload, at every float and complex dtype. Each
   kind of one and of two operands runs on GPU 0 through CUDA over 4096
   seeded elements, over specials where the dtype has them, and, at the
   complex dtypes, over every pair of numbers whose parts are each one of
   a few specials: there [w log z] meets infinities, zeros and NaNs in both
   parts, as [(0.5 + 0.5i) ** (inf + inf i)] does. Every test skips where
   CUDA opens no GPU. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module Support = Nx_kernels_support

let cuda () =
  match Support.gpus () with
  | b :: _ -> b
  | [] -> skip ~reason:"CUDA opens no GPU" ()

let terms = 4096

(* A part's specials as bits: zeros, a half, ones, two, infinities and
   NaNs of both signs. *)
let parts32 =
  List.map Int64.of_int32
    [
      0x00000000l; 0x80000000l; 0x3F000000l; 0x3F800000l; 0xBF800000l;
      0x40000000l; 0x7F800000l; 0xFF800000l; 0x7FC00000l; 0xFFC00000l;
    ]

let parts64 =
  List.map Int64.bits_of_float
    [ 0.; -0.; 0.5; 1.; -1.; 2.; infinity; neg_infinity ]
  @ [ 0x7FF8000000000000L; 0xFFF8000000000000L ]

(* Every number of two parts from [parts], each part [w] bytes. *)
let numbers w parts =
  let set b o v =
    if w = 4 then Bytes.set_int32_le b o (Int64.to_int32 v)
    else Bytes.set_int64_le b o v
  in
  List.concat_map
    (fun re ->
      List.map
        (fun im ->
          let b = Bytes.create (2 * w) in
          set b 0 re;
          set b w im;
          Bytes.to_string b)
        parts)
    parts

(* Each number of [numbers] beside each: [x] runs through the numbers
   slowly, [y] fast. *)
let grid (type s) (dt : (Complex.t, s) D.t) =
  let ns =
    match dt with
    | D.Complex64 -> numbers 4 parts32
    | D.Complex128 -> numbers 8 parts64
  in
  let k = List.length ns in
  let array f =
    let data = String.concat "" (List.init (k * k) f) in
    A.v dt (L.contiguous [| k * k |]) (Rig.Buffer.of_string data)
  in
  let at i = List.nth ns i in
  Kernels_cases.Pair (array (fun i -> at (i / k)), array (fun i -> at (i mod k)))

type row = Row : string * Kernels_cases.pair -> row

let kind_dtypes =
  List.filter (fun (D.Any dt) -> D.is D.Float dt || D.is D.Complex dt) D.all

let rows =
  List.concat_map
    (fun (D.Any dt) ->
      let name = D.name dt in
      let seeded seed = Kernels_cases.seeded dt [| terms |] seed in
      let specials =
        if Kernels_cases.specials dt = [] then []
        else
          [
            Row
              ( name ^ " specials",
                Kernels_cases.Pair
                  ( Kernels_cases.specials_only dt terms 1,
                    Kernels_cases.specials_only dt terms 2 ) );
          ]
      in
      Row (name ^ " seeded", Kernels_cases.Pair (seeded 1, seeded 2))
      :: specials)
    kind_dtypes
  @ [
      Row ("complex64 grid", grid D.Complex64);
      Row ("complex128 grid", grid D.Complex128);
    ]

(* The parts of [x]'s complex elements, as floats. *)
let complex_parts (type v s) (x : (v, s) A.t) : Complex.t array =
  let all (y : (Complex.t, s) A.t) =
    let n = Iarray.fold_left ( * ) 1 (L.shape (A.layout y)) in
    Array.init n (fun i -> A.get y (Iarray.of_array [| i |]))
  in
  match A.dtype x with
  | D.Complex64 -> all x
  | D.Complex128 -> all x
  | _ -> [||]

(* Whether [z ** w] at nx_pow_raw's two-part [w] takes [w log z]'s
   imaginary part from [inf - inf] while its real part is [-inf]. *)
let vanishing (z : Complex.t) (w : Complex.t) =
  let lr = Float.log (Float.hypot z.re z.im) and li = Float.atan2 z.im z.re in
  let a = w.re *. li and b = w.im *. lr in
  w.re <> 0. && w.im <> 0.
  && Float.abs a = infinity
  && Float.abs b = infinity
  && Float.is_nan (a +. b)
  && (w.re *. lr) -. (w.im *. li) = neg_infinity

let law (Row (_, (Kernels_cases.Pair (x, y) as p))) =
  let zs = complex_parts x and ws = complex_parts y in
  cover "a complex pow whose w log z is -inf + i (inf - inf)"
    (Array.exists2 vanishing zs ws);
  let b = cuda () in
  Kernels_cases.law_map1 b (Kernels_cases.Case x);
  Kernels_cases.law_map2 b p

let () =
  exit
    (run "nx.cuda's kinds"
       [
         group "kinds"
           [
             prop ~count:1 ~examples:rows
               "every kind has nx.cpu's bits at float and complex dtypes"
               (Gen.of_list
                  ~pp:(fun ppf (Row (name, _)) -> Format.pp_print_string ppf name)
                  rows)
               law;
           ];
       ])
