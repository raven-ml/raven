(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops
open Shape
module V = Dtype.Value

exception Not_a_number

let number : Dtype.const -> V.t = function
  | #Dtype.value as v -> v
  | `Invalid -> raise_notrace Not_a_number

let rule p f =
  Pattern_matcher.rule p (fun m -> try f m with Not_a_number -> None)

let zero = V.of_int 0
let num u = number (value u)

(* Python's integers in node arithmetic: [lit n] is the weak literal an integer
   becomes, and [sum] starts from the integer 0, as Python's does. *)
let lit (n : V.t) = const (n :> Dtype.const)
let sum us = List.fold_left add (lit zero) us
let size u = Nodes.cardinal (backward_slice ~calls:Skip u)
let zero_like u = const_like u (zero :> Dtype.const)

(* [first rules] is the result of the first rule that applies. *)
let first = List.find_map (fun rule -> rule ())

(* itertools.product: the first list varies slowest. *)
let rec product = function
  | [] -> Seq.return []
  | xs :: rest ->
      Seq.flat_map
        (fun x -> Seq.map (List.cons x) (product rest))
        (List.to_seq xs)

(* Memoised on index nodes, for as long as the node lives; compilation runs on
   domains, so the table is locked. *)
module Memo = Ephemeron.K1.Make (struct
  type nonrec t = t

  let equal = ( == )
  let hash = hash
end)

let memo = Memo.create 64
let memo_lock = Mutex.create ()

(* PARAM // c is irreducible *)
let irreducible x y =
  match arg x with
  | Param { multiple_of = Some m; _ } ->
      op x = Op.Param && op y = Op.Const && V.(of_int m % num y = zero)
  | _ -> false

let rec fold_divmod_general d =
  match Mutex.protect memo_lock (fun () -> Memo.find_opt memo d) with
  | Some ret -> ret
  | None ->
      let ret = fold d in
      Mutex.protect memo_lock (fun () -> Memo.replace memo d ret);
      ret

and fold d =
  let x = nth d 0 and y = nth d 1 and is_mod = op d = Op.Floormod in
  if V.(vmin y = zero && vmax y = zero) then raise Division_by_zero;
  let xdiv = O.(x // y) in
  let q = vmin xdiv in
  (* x // y is constant *)
  if V.(q = vmax xdiv) then
    Some
      (if is_mod then O.(x - (lit q * y))
       else const_like xdiv (q :> Dtype.const))
  else if irreducible x y then if is_mod then Some (zero_like d) else None
  else
    let x_peeled, const = pop_const x in
    let const = number const and uops_no_const = split_uop x_peeled Op.Add in
    (* Constant denominator rules, for a constant c > 0 *)
    let c = if op y = Op.Const then num y else zero in
    (* nested_div: (x % (k * c)) // c is (x // c) % k for k > 0; the mod case is
       remove_nested_mod's *)
    let nested_div () =
      if is_mod || op x <> Op.Floormod then None
      else
        match divides (nth x 1) (V.to_z c) with
        | Some k when V.(vmin k > zero) -> Some O.(nth x 0 // y % k)
        | _ -> None
    in
    (* remove_nested_mod in a sum: (a % 4 + b) % 2 is (a + b) % 2 *)
    let remove_nested_mod () =
      let unnest u =
        if op u = Op.Floormod && Option.is_some (divides (nth u 1) (V.to_z c))
        then nth u 0
        else u
      in
      let xs = List.map unnest uops_no_const in
      if (not is_mod) || List.equal ( == ) xs uops_no_const then None
      else Some O.((usum (List.hd xs) (List.tl xs) + lit const) % y)
    in
    let folds () =
      (* The shared decomposition: const_factor u divides u *)
      let factors = List.map (fun u -> `Int (const_factor u)) uops_no_const in
      let terms =
        List.map2
          (fun u f -> Option.get (divides u (V.to_z f)))
          uops_no_const factors
      in
      (* fold_divmod_congruence: fold if x is congruent to an expression whose
         range is within one period of c. A lone term (a binary numerator that
         crosses one period) or an exact f % c = c // 2 tie tries both signs of
         the remainder; otherwise the smaller keeps the product small. *)
      let congruence () =
        let lone = List.compare_length_with terms 1 = 0 in
        let choices f =
          (* r is in [0, c), so r is the smaller in magnitude iff r <= c - r *)
          let r = V.(f % c) in
          if V.(r * of_int 2 = c) || lone then [ r; V.(r - c) ]
          else [ (if V.(r <= c - r) then r else V.(r - c)) ]
        in
        let fold rems =
          let rem =
            O.(
              sum (List.map2 (fun r v -> lit r * v) rems terms)
              + lit V.(const % c))
          in
          let q = V.(vmin rem // c) in
          if V.(q <> vmax rem // c) then None
          else if is_mod then Some O.(rem - lit V.(q * c))
          else
            let quotient (f, r) v = O.(lit V.((f - r) // c) * v) in
            let quotients =
              List.map2 quotient (List.combine factors rems) terms
            in
            Some O.(sum quotients + lit V.(const // c) + lit q)
        in
        Seq.find_map fold (product (List.map choices factors))
      in
      (* gcd_with_remainder: factor out the common gcd of the numerator *)
      let gcd_with_remainder () =
        let g =
          `Int
            (List.fold_left (fun g f -> Bigint.gcd g (V.to_z f)) (V.to_z c) factors)
        in
        if V.(g <= of_int 1) then None
        else
          let x_g = simplify (Option.get (divides x_peeled (V.to_z g))) in
          let new_x = O.(x_g + lit V.(const // g % (c // g))) in
          if V.(vmin new_x < zero) then None
          else if is_mod then
            Some O.((new_x % lit V.(c // g) * lit g) + lit V.(const % g))
          else Some O.((new_x // lit V.(c // g)) + lit V.(const // c))
      in
      (* nest_by_factor: x // c is (x // f) // (c // f), and x % c is (x // f %
         (c // f)) * f + x % f. The division holds for x of any sign; the
         remainder's reconstruction needs x >= 0 *)
      let nest_by_factor () =
        let divisor u f =
          let f = V.(max f (-f)) in
          if op u <> Op.Const && V.(of_int 1 < f && f < c && c % f = zero) then
            Some f
          else None
        in
        let divs =
          List.sort_uniq V.compare
            (List.filter_map Fun.id (List.map2 divisor uops_no_const factors))
        in
        let remainder newxs div =
          let part f t =
            if V.(f % div = zero) then [] else [ O.(lit V.(f % div) * t) ]
          in
          let parts = List.concat (List.map2 part factors terms) in
          let parts =
            if V.(const % div = zero) then parts
            else parts @ [ const_like x (V.(const % div) :> Dtype.const) ]
          in
          let b = match parts with [] -> zero_like x | b :: bs -> usum b bs in
          if V.(zero <= vmin b && vmax b < div) then
            let r = O.((newxs % lit V.(c // div) * lit div) + b) in
            Some (size r, r)
          else None
        in
        let result div =
          match fold_divmod_general O.(x // lit div) with
          | Some newxs when not is_mod ->
              Some (size newxs, O.(newxs // lit V.(c // div)))
          | Some newxs when V.(vmin x >= zero && vmin newxs >= zero) ->
              remainder newxs div
          | _ -> None
        in
        let smaller (n0, r0) (n1, r1) =
          if n1 < n0 then (n1, r1) else (n0, r0)
        in
        match List.filter_map result divs with
        | [] -> None
        | r :: rs -> Some (snd (List.fold_left smaller r rs))
      in
      first [ congruence; gcd_with_remainder; nest_by_factor ]
    in
    (* Variable denominator and fallback rules *)
    let all_uops = split_uop x Op.Add in
    (* divide_by_gcd: x // y is (x // gcd) // (y // gcd) *)
    let divide_by_gcd () =
      let g = simplify (gcd (all_uops @ [ y ])) in
      if op g = Op.Const && V.(num g = of_int 1) then None
      else
        let ret =
          alu
            (Option.get (divide_exact x g))
            (op d)
            [ Option.get (divide_exact y g) ]
        in
        Some (if is_mod then O.(ret * g) else ret)
    in
    (* factor_remainder: (d * x + y) // d is x + y // d *)
    let factor_remainder () =
      let split u (quo, rem) =
        let f = `Int (const_factor u) in
        match divide_exact u y with
        | Some q -> (q :: quo, rem)
        | None when op y = Op.Const && V.(f % c <> f) ->
            let t = Option.get (divides u (V.to_z f)) in
            let q = if is_mod then zero_like u else O.(t * lit V.(f // c)) in
            (q :: quo, O.(t * lit V.(f % c)) :: rem)
        | None -> (quo, u :: rem)
      in
      match List.fold_right split all_uops ([], []) with
      | [], _ -> None
      | quo, rem ->
          let new_x = O.(sum rem + zero_like x) in
          if V.(vmin new_x < zero) then None
          else
            Some (if is_mod then O.(new_x % y) else O.((new_x // y) + sum quo))
    in
    let fallback () =
      if V.(vmin y < zero || vmin x < zero) then None else factor_remainder ()
    in
    let constant_rules =
      if V.(c > zero) then [ nested_div; remove_nested_mod; folds ] else []
    in
    first (constant_rules @ [ divide_by_gcd; fallback ])

let floor_ops = Op.Set.of_list [ Op.Floordiv; Op.Floormod ]

let div_and_mod_symbolic =
  Pattern_matcher.v
    (fun () -> [
      (* Fast inline rules *)
      (* (x // c + a) // d is (x + a * c) // (c * d) for d > 0, where
           nothing wraps *)
      rule
        Upat.(((var "x" // cvar "c") + cvar "a") // cvar "d")
        (fun m ->
          let x = m "x" and c = m "c" and a = m "a" and d = m "d" in
          let ac = V.(vmin a * vmin c) and cd = V.(vmin c * vmin d) in
          let values =
            V.[ vmin a; vmin c; vmin d; ac; cd; vmin x + ac; vmax x + ac ]
          in
          if V.(vmin d > zero) && exact (dtype x) values then
            Some O.((x + (a * c)) // (c * d))
          else None);
      (* (x + c) // d is (x + c % d) // d + c // d, and (x + c) % d is (x + c %
         d) % d: the multiple of d leaves the constant, for any d <> 0 *)
      rule
        (Upat.v ~op:floor_ops
           ~src:
             [
               Upat.(var "x" ~dtype:[ Dtype.Weak_int ] + cvar "c");
               Upat.cvar "d";
             ]
           ~name:"n" ())
        (fun m ->
          let x = m "x" and c = num (m "c") and d = m "d" in
          let dv = num d in
          if V.(dv = zero || c % dv = c) then None
          else if op (m "n") = Op.Floordiv then
            Some O.(((x + lit V.(c % dv)) // d) + lit V.(c // dv))
          else Some O.((x + lit V.(c % dv)) % d));
      (* Slow rules *)
      rule (Upat.v ~op:floor_ops ~dtype:[ Dtype.Weak_int ] ~name:"d" ())
        (fun m -> fold_divmod_general (m "d"));
    ])
