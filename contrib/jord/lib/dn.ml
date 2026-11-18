(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf
let none x = Nx.full_at (Nx.placement x) Nx.bool (Nx.shape x) false

let refuse fn x what =
  invalid_arg
    (strf "%s: %s is not %s dtype" fn (Nx.Dtype.name (Nx.dtype x)) what)

(* Numbers *)

(* Where the DNs [x] equal [n]. An integer dtype holds [n] when [n] is an
   integer of its range, which the powers of 2 bound exactly. *)
let equal (type v s) fn (x : (v, s, _) Nx.t) n =
  let dt = Nx.dtype x in
  match Nx.Dtype.kind dt with
  | Float when Float.is_nan n -> Nx.isnan x
  | Float ->
      let v = Nx.Dtype.of_float dt n in
      if v = n then Nx.equal_s x v else none x
  | (Signed | Unsigned) as k ->
      let w = Nx.Dtype.bits dt in
      let lo, hi =
        match k with
        | Signed -> (-.Float.ldexp 1. (w - 1), Float.ldexp 1. (w - 1))
        | _ -> (0., Float.ldexp 1. w)
      in
      if Float.is_integer n && lo <= n && n < hi then
        Nx.equal_s x (Nx.Dtype.of_float dt n)
      else none x
  | Complex | Boolean -> refuse fn x "a real"

let mem (type v s) fn ns (x : (v, s, _) Nx.t) =
  match Nx.Dtype.kind (Nx.dtype x) with
  | Complex | Boolean -> refuse fn x "a real"
  | Float | Signed | Unsigned ->
      List.fold_left (fun m n -> Nx.logical_or m (equal fn x n)) (none x) ns

(* Bits *)

let ints (type v s) fn (x : (v, s, _) Nx.t) =
  match Nx.Dtype.kind (Nx.dtype x) with
  | Signed | Unsigned -> ()
  | Float | Complex | Boolean ->
      invalid_arg
        (strf
           "%s: %s is not an integer dtype; mask NaN DNs as fill \
            (Nx.isnan), then cast the rest to the file's integer dtype"
           fn
           (Nx.Dtype.name (Nx.dtype x)))

(* The 0-d value of [x]'s dtype whose set bits are [bits]: a shift of 1 in
   that dtype, so that the top bit of a signed dtype is its sign, and a bit
   at or past the width is none, as [Nx.lshift] gives 0 there. *)
let pattern x bits =
  let dt = Nx.dtype x in
  let one = Nx.scalar_like x (Nx.Dtype.one dt) in
  List.fold_left
    (fun m b -> Nx.bitwise_or m (Nx.lshift one b))
    (Nx.scalar_like x (Nx.Dtype.zero dt))
    bits

let width x = Nx.Dtype.bits (Nx.dtype x)

let any fn bits x =
  ints fn x;
  let zero = Nx.Dtype.zero (Nx.dtype x) in
  Nx.not_equal_s (Nx.bitwise_and x (pattern x bits)) zero

(* Where the DNs' [bits] are set at [set] alone. A bit of [set] past the
   width is set in no DN. *)
let holds x ~bits ~set =
  if List.exists (fun b -> b >= width x) set then none x
  else Nx.equal (Nx.bitwise_and x (pattern x bits)) (pattern x set)

let all fn bits x =
  ints fn x;
  holds x ~bits ~set:bits

let field fn ~offset ~length vs x =
  ints fn x;
  let bits = List.init length (( + ) offset) in
  let set v = List.filter (fun b -> (v lsr (b - offset)) land 1 = 1) bits in
  List.fold_left
    (fun m v ->
      if v < 0 || v lsr length <> 0 then m
      else Nx.logical_or m (holds x ~bits ~set:(set v)))
    (none x) vs
