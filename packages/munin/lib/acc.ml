(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The finite values are summed with Neumaier's compensation, scaled by
   2^-[shift]: when the sum, or the sum with its compensation, would overflow,
   both are scaled down by 2^-[step] (exactly, being powers of two), so that a
   mean of values near the largest float stays finite. The compensation counts
   too: [max_float] plus two 2^969 rounds to [max_float] each time, leaving
   2^970 in the compensation, and the two together round to infinity. *)
type finite = {
  least : Metric.sample;
  greatest : Metric.sample;
  n : int;
  sum : float;
  comp : float;
  shift : int;
}

let step = 64

let rec add_scaled f x =
  let x' = Float.ldexp x (-f.shift) in
  let t = f.sum +. x' in
  let comp =
    if Float.abs f.sum >= Float.abs x' then f.comp +. (f.sum -. t +. x')
    else f.comp +. (x' -. t +. f.sum)
  in
  if Float.is_finite (t +. comp) then { f with sum = t; comp }
  else
    add_scaled
      {
        f with
        sum = Float.ldexp f.sum (-step);
        comp = Float.ldexp f.comp (-step);
        shift = f.shift + step;
      }
      x

type t = {
  count : int;
  last : Metric.sample option;
  finite : finite option;
  non_finite : (int * Metric.sample) option;
}

let empty = { count = 0; last = None; finite = None; non_finite = None }

let add a (s : Metric.sample) =
  let x = s.value in
  let finite, non_finite =
    if Float.is_finite x then
      let f =
        match a.finite with
        | None ->
            { least = s; greatest = s; n = 1; sum = x; comp = 0.; shift = 0 }
        | Some f ->
            let f = add_scaled f x in
            {
              (* Strict comparisons keep the earliest of equals, and treat -0.
                 and 0. as equal. *)
              least = (if x < f.least.value then s else f.least);
              greatest = (if x > f.greatest.value then s else f.greatest);
              n = f.n + 1;
              sum = f.sum;
              comp = f.comp;
              shift = f.shift;
            }
      in
      (Some f, a.non_finite)
    else
      ( a.finite,
        Some
          (match a.non_finite with
          | None -> (1, s)
          | Some (k, first) -> (k + 1, first)) )
  in
  { count = a.count + 1; last = Some s; finite; non_finite }

(* The exact mean lies between the least and greatest values, so clamping the
   rounded one to them only brings it closer: the rounded mean of five copies
   of [max_float] falls a ulp below it. *)
let mean f =
  let m = Float.ldexp ((f.sum +. f.comp) /. Float.of_int f.n) f.shift in
  if m < f.least.value then f.least.value
  else if m > f.greatest.value then f.greatest.value
  else m

let stats a =
  match a.last with
  | None -> None
  | Some last ->
      Some
        {
          Metric.count = a.count;
          last;
          finite =
            Option.map
              (fun f ->
                {
                  Metric.least = f.least;
                  greatest = f.greatest;
                  mean = mean f;
                })
              a.finite;
          non_finite = a.non_finite;
        }
