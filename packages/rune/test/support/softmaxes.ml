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

(* A program: a dtype and [r] lanes of [k] scaled by [scale]. Its softmax and
   log_softmax compile once for each set the law is applied to, at the first
   case that draws the program. *)
type program = { dt : dtype; r : int; k : int; scale : float }

(* Lanes of one element, of a few, of 16 and 17 about a tile's edge, of 1025,
   one past 1024, and of a few thousand. *)
let lengths = [| 1; 2; 5; 16; 17; 1025; 3000 |]
let scales = [| 1.; -0.5; 3.; 0.1 |]

(* Every dtype at every lane length; at each length, the dtypes take one to
   three lanes and every scale. *)
let programs =
  let n = Array.length lengths in
  Array.init
    (List.length dtypes * n)
    (fun i ->
      let dt = i / n and j = i mod n in
      {
        dt = List.nth dtypes dt;
        r = 1 + ((j + dt) mod 3);
        k = lengths.(j);
        scale = scales.((j + dt) mod Array.length scales);
      })

type case = {
  program : int;
  xs : float array;
  offset : float;
  special : special;
}

let pp_float ppf v = Format.fprintf ppf "%h" v

let pp_dtype ppf (Dt (dt, _, _, _)) = Nx.pp_dtype ppf dt

let pp_case ppf c =
  let p = programs.(c.program) in
  Format.fprintf ppf "%a [%d; %d], scale %h: %a" pp_dtype p.dt p.r p.k p.scale
    (Format.pp_print_array
       ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
       pp_float)
    c.xs

(* [program]'s case of elements [xs] about [offset]: with [Neg_inf], the
   elements whose [holes] is [0], a quarter, are [-∞]; with [Nan] or [Inf], one
   element in every other lane is a NaN or [+∞]. *)
let case program xs offset special holes =
  let k = programs.(program).k in
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
  { program; xs; offset; special }

let gen =
  let open Gen in
  with_pp pp_case
  @@ let* program = int_range 0 (Array.length programs - 1) in
     let { r; k; _ } = programs.(program) in
     let* bound = of_list ~pp:pp_float [ 1.; 60.; 300. ] in
     let* xs = array ~size:(constant (r * k)) (float_range (-.bound) bound) in
     let* offset =
       frequency
         [ (3, constant 0.); (1, of_list ~pp:pp_float [ 1e3; -1e4; 3e4 ]) ]
     in
     let* special =
       frequency
         [
           (4, constant Plain);
           (3, constant Neg_inf);
           (2, constant Nan);
           (1, constant Inf);
         ]
     in
     let+ holes = array ~size:(constant (r * k)) (int_range 0 3) in
     case program xs offset special holes

(* Each program once, its elements spread over [[-60, 60]], the offsets and
   special values taken in turn. *)
let examples =
  let offsets = [| 0.; 1e3; 0.; -1e4 |]
  and specials = [| Plain; Neg_inf; Nan; Inf |] in
  List.init (Array.length programs) (fun i ->
      let { r; k; _ } = programs.(i) in
      let n = r * k in
      let xs = Array.init n (fun j -> float_of_int (j * 37 mod 121) -. 60.) in
      case i xs
        offsets.(i mod 4)
        specials.(i / 4 mod 4)
        (Array.init n (fun j -> j mod 4)))

(* The law *)

(* [xs] rounded to [p]'s dtype, as float64 numbers. *)
let rounded p xs =
  match p.dt with
  | Dt (dt, _, _, _) ->
      let x = Nx.create Nx.float64 [| p.r; p.k |] xs in
      Nx.to_array (Nx.cast Nx.float64 (Nx.cast dt x))

let is_float64 (Dt (dt, _, _, _)) = Nx.Dtype.equal dt Nx.float64
let roundoffs (Dt (_, u, ud, normal)) = (u, ud, normal)

(* [p]'s softmax, or log_softmax with [log], compiled at [on], as a function of
   [p]'s elements as float64 numbers, which it rounds to [p]'s dtype. *)
let compiled on ~log p =
  match p.dt with
  | Dt (dt, _, _, _) ->
      let scale = p.scale and s = [| p.r; p.k |] in
      lazy
        (let f =
           Rune.jit' (fun x ->
               if log then Nx.log_softmax ~scale x else Nx.softmax ~scale x)
         in
         fun xs ->
           let x = Nx.place on (Nx.cast dt (Nx.create Nx.float64 s xs)) in
           Nx.to_array (Nx.cast Nx.float64 (Nx.place Nx.Host.on (f x))))

let law_at softmax log_softmax ~f64 ~chain ~exp_ulps ?log_ulps c =
  let p = programs.(c.program) in
  let u, ud, normal = roundoffs p.dt in
  if is_float64 p.dt then assume f64;
  let r = p.r and k = p.k in
  cover "a narrow dtype" (ud > 0.);
  cover "one element a lane" (k = 1);
  cover "lanes far from 0" (c.offset <> 0.);
  cover "a scale other than 1" (p.scale <> 1.);
  cover "-inf elements" (c.special = Neg_inf);
  let xs = rounded p c.xs in
  let scale = p.scale in
  let got = Lazy.force softmax.(c.program) c.xs in
  (* [log_ulps] and the compiled [log_softmax]. *)
  let log =
    Option.map
      (fun lu -> (lu, Lazy.force log_softmax.(c.program) c.xs))
      log_ulps
  in
  let b = if chain then Stdlib.max (Sm.beta k) (k - 1) else Sm.beta k in
  for lane = 0 to r - 1 do
    let i0 = lane * k in
    let exact = Sm.lane ~scale (Array.sub xs i0 k) in
    cover "a NaN lane" (Option.is_none exact);
    match exact with
    | None ->
        for i = i0 to i0 + k - 1 do
          if not (Float.is_nan got.(i)) then
            failf "softmax at %d of a NaN lane: %h" i got.(i);
          Option.iter
            (fun (_, lsm) ->
              if not (Float.is_nan lsm.(i)) then
                failf "log_softmax at %d of a NaN lane: %h" i lsm.(i))
            log
        done
    | Some l -> (
        Sm.covers ~normal l;
        let at j = i0 + j in
        Sm.check u ~narrow:ud ~normal ~exp_ulps ~b ~n:k ~scale l ~at got;
        match log with
        | None -> ()
        | Some (log_ulps, lsm) ->
            Sm.check_log u ~narrow:ud ~exp_ulps ~log_ulps ~b ~n:k ~scale l ~at
              lsm)
  done

let law (module S : Nx.Devices) =
  law_at
    (Array.map (compiled S.on ~log:false) programs)
    (Array.map (compiled S.on ~log:true) programs)
