(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t =
  | Rows of { before : int; after : int }
  | Times of { on : string; before : Time.span; after : Time.span }

(* [positive_sum a b] is [true] iff [a + b > 0], without overflow: the first two
   branches only keep [Int64.add] from overflowing, so where one bound moves by
   one value, the last branch agrees. *)
let positive_sum a b =
  if (a > 0L && b > 0L) [@mutate off "an overflow guard"] then true
  else if (a <= 0L && b <= 0L) [@mutate off "an overflow guard"] then false
  else Int64.add a b > 0L

let rows ~before ~after =
  (* Two OCaml [int]s sum in [int64] without overflow. *)
  if Int64.add (Int64.of_int before) (Int64.of_int after) < 0L then
    Format.kasprintf invalid_arg
      "Window.rows: ~before:%d ~after:%d is empty for every row" before after;
  Rows { before; after }

let time ?(after = Time.Span.ns 0) ~before on =
  if not (positive_sum (Time.Span.to_ns before) (Time.Span.to_ns after)) then
    Format.kasprintf invalid_arg
      "Window.time: ~before:%a ~after:%a is empty for every row" Time.Span.pp
      before Time.Span.pp after;
  Times { on; before; after }

let equal w0 w1 =
  match (w0, w1) with
  | Rows r0, Rows r1 ->
      Int.equal r0.before r1.before && Int.equal r0.after r1.after
  | Times t0, Times t1 ->
      String.equal t0.on t1.on
      && Time.Span.equal t0.before t1.before
      && Time.Span.equal t0.after t1.after
  | (Rows _ | Times _), _ -> false

let pp ppf = function
  | Rows { before; after } ->
      let pp_int ppf n =
        if n < 0 then Format.fprintf ppf "(%d)" n else Format.pp_print_int ppf n
      in
      Format.fprintf ppf "rows ~before:%a ~after:%a" pp_int before pp_int after
  | Times { on; before; after } ->
      if Int64.equal (Time.Span.to_ns after) 0L then
        Format.fprintf ppf "time ~before:%a %S" Time.Span.pp before on
      else
        Format.fprintf ppf "time ~after:%a ~before:%a %S" Time.Span.pp after
          Time.Span.pp before on
