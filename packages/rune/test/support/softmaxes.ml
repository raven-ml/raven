(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap

module Sm = Nx_softmax_support

(* Cases *)

(* A dtype with the unit roundoff of the format it computes in, its own
   ([0.] where it is that format) and its least normal float. *)
type dtype = Dt : (float, 's) Nx.dtype * float * float * float -> dtype

let dtypes =
  [
    Dt (Nx.float64, 0x1p-53, 0., 0x1p-1022);
    Dt (Nx.float32, 0x1p-24, 0., 0x1p-126);
    Dt (Nx.float16, 0x1p-24, 0x1p-11, 0x1p-14);
    Dt (Nx.bfloat16, 0x1p-24, 0x1p-8, 0x1p-126);
  ]

type special = Plain | Neg_inf | Nan | Inf

type case = {
  dt : dtype;
  s : int array;
  xs : float array;
  offset : float;
  special : special;
  scale : float;
}

let pp_float ppf v = Format.fprintf ppf "%h" v

let gen =
  let open Gen in
  let* dt =
    of_list ~pp:(fun ppf (Dt (d, _, _, _)) -> Nx.pp_dtype ppf d) dtypes
  in
  let* r = int_range 1 3 in
  let* k =
    frequency
      [
        (2, constant 1);
        (5, of_list ~pp:Format.pp_print_int [ 2; 5; 16; 17; 1025; 3000 ]);
      ]
  in
  let* bound = of_list ~pp:pp_float [ 1.; 60.; 300. ] in
  let* xs = array ~size:(constant (r * k)) (float_range (-.bound) bound) in
  let* offset =
    frequency
      [ (3, constant 0.); (1, of_list ~pp:pp_float [ 1e3; -1e4; 3e4 ]) ]
  in
  let* special =
    frequency
      [
        (4, constant Plain); (3, constant Neg_inf); (2, constant Nan);
        (1, constant Inf);
      ]
  in
  let* holes = array ~size:(constant (r * k)) (int_range 0 3) in
  let+ scale = of_list ~pp:pp_float [ 1.; -0.5; 3.; 0.1 ] in
  (* A quarter of the elements [-∞], or a NaN or [+∞] in every other
     lane. *)
  let xs =
    Array.mapi
      (fun i x ->
        let lane = i / k and j = i mod k in
        let marked = lane mod 2 = 0 && j = lane * 7 mod k in
        match special with
        | Plain -> x +. offset
        | Neg_inf -> if holes.(i) = 0 then Float.neg_infinity else x +. offset
        | Nan -> if marked then Float.nan else x +. offset
        | Inf -> if marked then Float.infinity else x +. offset)
      xs
  in
  { dt; s = [| r; k |]; xs; offset; special; scale }

(* The law *)

let law (module S : Nx.Devices) ~f64 ~chain ~exp_ulps c =
  let (Dt (dt, u, ud, normal)) = c.dt in
  if Nx.Dtype.equal dt Nx.float64 then assume f64;
  let r = c.s.(0) and k = c.s.(1) in
  cover "a narrow dtype" (ud > 0.);
  cover "one element a lane" (k = 1);
  cover "lanes far from 0" (c.offset <> 0.);
  cover "a scale other than 1" (c.scale <> 1.);
  cover "-inf elements" (c.special = Neg_inf);
  let x = Nx.cast dt (Nx.create Nx.float64 c.s c.xs) in
  let xs = Nx.to_array (Nx.cast Nx.float64 x) in
  let scale = c.scale in
  let y = Rune.jit' (fun x -> Nx.softmax ~scale x) (Nx.place S.on x) in
  let got = Nx.to_array (Nx.cast Nx.float64 (Nx.place Nx.Host.on y)) in
  let b = if chain then Stdlib.max (Sm.beta k) (k - 1) else Sm.beta k in
  for lane = 0 to r - 1 do
    let i0 = lane * k in
    let exact = Sm.lane ~scale (Array.sub xs i0 k) in
    cover "a NaN lane" (Option.is_none exact);
    match exact with
    | None ->
        for i = i0 to i0 + k - 1 do
          if not (Float.is_nan got.(i)) then
            failf "softmax at %d of a NaN lane: %h" i got.(i)
        done
    | Some l ->
        Sm.covers ~normal l;
        Sm.check u ~narrow:ud ~normal ~exp_ulps ~b ~n:k ~scale l
          ~at:(fun j -> i0 + j)
          got
  done
