(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Integer division and modulo simplification.

   Two layers of rules:

   1. Fast inline rules: algebraic identities keyed off vmin/vmax
      bounds.

   2. [fold_divmod_general_rec]: multi-strategy folder for non-trivial cases.
      Strategies are tried in order: [cancel_divmod], [nested_div],
      [remove_nested_mod], [fold_divmod_congruence],
      [gcd_with_remainder], [nest_by_factor], [divide_by_gcd],
      [factor_remainder]. Some strategies that require the [simplify]
      hook to make progress fall back to partial results. *)

let int64_min_int = Int64.of_int min_int
let int64_max_int = Int64.of_int max_int

let int_of_int64_checked n =
  if Int64.compare n int64_min_int < 0
     || Int64.compare n int64_max_int > 0
  then None
  else Some (Int64.to_int n)

let add_checked a b =
  int_of_int64_checked (Int64.add (Int64.of_int a) (Int64.of_int b))

let sub_checked a b =
  int_of_int64_checked (Int64.sub (Int64.of_int a) (Int64.of_int b))

let abs_checked n = if n = min_int then None else Some (abs n)

let mul_checked a b =
  if a = 0 || b = 0 then Some 0
  else if a = 1 then Some b
  else if b = 1 then Some a
  else if a = -1 then if b = min_int then None else Some (-b)
  else if b = -1 then if a = min_int then None else Some (-a)
  else
    match abs_checked a, abs_checked b with
    | Some aa, Some bb when aa <= max_int / bb -> Some (a * b)
    | _ -> None

let option_all xs =
  let rec loop acc = function
    | [] -> Some (List.rev acc)
    | Some x :: xs -> loop (x :: acc) xs
    | None :: _ -> None
  in
  loop [] xs

let gcd_int_checked a b =
  let abs64 n = if Int64.compare n 0L < 0 then Int64.neg n else n in
  let rec loop a b =
    if b = 0L then abs64 a else loop b (Int64.rem a b)
  in
  loop (Int64.of_int a) (Int64.of_int b) |> int_of_int64_checked

let floor_div_checked x d =
  if d = 0 then None
  else
    let x = Int64.of_int x and d = Int64.of_int d in
    let q = Int64.div x d and r = Int64.rem x d in
    let q =
      if r <> 0L && ((Int64.compare r 0L < 0) <> (Int64.compare d 0L < 0))
      then Int64.sub q 1L
      else q
    in
    int_of_int64_checked q

let floor_mod_checked x d =
  match floor_div_checked x d with
  | None -> None
  | Some q ->
      let r =
        Int64.sub (Int64.of_int x)
          (Int64.mul (Int64.of_int q) (Int64.of_int d))
      in
      int_of_int64_checked r

let floordiv x y = Uop.alu_binary ~op:Ops.Floordiv ~lhs:x ~rhs:y
let floormod x y = Uop.alu_binary ~op:Ops.Floormod ~lhs:x ~rhs:y

let floordiv_pat x y = Upat.alu [ x; y ] Ops.Floordiv

(* Rule 1: (x//c + a)//d  ->  (x + a*c)//(c*d)
   when d > 0. The identity is valid for negative constant [c] under
   floor-division semantics. *)
let rule_nested_div =
  let open Upat in
  let x = var "x" and c = cvar ~name:"c" ()
  and a = cvar ~name:"a" () and d = cvar ~name:"d" () in
  floordiv_pat O.(floordiv_pat x c + a) d => fun bs ->
    let x = bs $ "x" and c = bs $ "c" in
    let a = bs $ "a" and d = bs $ "d" in
    if Uop.vmin d > 0
    then Some (floordiv Uop.O.(x + (a * c)) Uop.O.(c * d))
    else None

(* Rule 1b: (x+c)//d  ->  ((x+c%d)//d + c//d) when c%d != c and d > 0. *)
let rule_add_const_div =
  let open Upat in
  let x = var_dtype "x" (exact_dtype Dtype.index)
  and c = cvar ~name:"c" ()
  and d = cvar ~name:"d" () in
  let n = alu ~name:"n" [ x; c ] Ops.Add in
  floordiv_pat n d => fun bs ->
    let x = bs $ "x" and c = bs $ "c" and d = bs $ "d" in
    match Uop.const_int_value c, Uop.const_int_value d with
    | Some cv, Some dv when dv > 0 ->
        (match floor_mod_checked cv dv, floor_div_checked cv dv with
         | Some c_mod_v, Some c_div_v when c_mod_v <> cv ->
             let c_mod = Uop.const_like c c_mod_v in
             let c_div = Uop.const_like c c_div_v in
             Some Uop.O.(floordiv (x + c_mod) d + c_div)
         | _ -> None)
    | _ -> None

(* fold_divmod_general

   General folder for div/mod. Strategies producing nested div/mod
   ([fold_divmod_congruence], [nest_by_factor]) assume [Uop.simplify]
   is installed to reach a fixed point; without it the results are
   under-simplified but still correct. *)

(* cancel_divmod: [x // y] takes a single value over the whole range.
   The quotient bound is read from the constructed [x // y] node so the
   check uses the same reasoning as everywhere else, including divisors
   that are only bounded on one side. *)
let try_cancel_divmod d_op x y =
  let xdiv = floordiv x y in
  let q = Uop.vmin xdiv in
  if q <> Uop.vmax xdiv then None
  else if d_op = Ops.Floormod then Some Uop.O.(x - (Uop.const_like x q * y))
  else Some (Uop.const_like xdiv q)

(* remove_nested_mod for ADD: (a%4 + b) % 2 -> (a+b) % 2 when the inner
   mod divisor divides [c]. *)
let try_remove_nested_mod d_op x y c =
  if d_op <> Ops.Floormod then None
  else
    let x_peeled, const = Uop.pop_const x in
    let uops = Uop.split_uop x_peeled Ops.Add in
    let changed = ref false in
    let new_xs = List.map (fun u ->
      if Uop.op u = Ops.Floormod && Array.length (Uop.src u) = 2 then
        let inner_d = (Uop.src u).(1) in
        match Uop.divides inner_d c with
        | Some _ ->
            changed := true;
            (Uop.src u).(0)
        | None -> u
      else u) uops
    in
    if not !changed then None
    else
      let sum = Uop.usum new_xs in
      let sum_c =
        if const = 0 then sum
        else Uop.O.(sum + Uop.const_like sum const)
      in
      Some (floormod sum_c y)

(* Split [x] into its addition terms (with the constant popped off) and
   compute the [(factored_term, factor)] decomposition used by the
   constant-denominator strategies. Returns [(uops_no_const, const,
   terms, factors)]. [terms.(i) * factors.(i)] is provably equal to
   [uops_no_const.(i)]; [terms.(i)] is [None] when the divisibility could
   not be proved syntactically. *)
let decompose_numerator x =
  let x_peeled, const = Uop.pop_const x in
  let uops = Uop.split_uop x_peeled Ops.Add in
  let factors = List.map Uop.const_factor uops in
  let terms =
    List.map2 (fun u f ->
      if f = 0 then None else Uop.divides u f) uops factors in
  (uops, const, terms, factors)

(* fold_divmod_congruence: pick per-term residues [r_i] of [factor_i] mod
   [c] and check whether the residual sum lies in a single period of
   [c]. If so, divide or modulo simplifies to a closed form using the
   [(factor - r) / c] multiples plus a constant. When a factor's residue
   is exactly [c/2], both signs of the residue are tried since either
   may fit. *)
let try_fold_divmod_congruence d_op x y c =
  let _uops, const, terms, factors = decompose_numerator x in
  if List.exists (fun t -> t = None) terms then None
  else
    let unwrap_terms = List.map (function Some t -> t | None -> assert false) terms in
    (* Per-factor residue choices. *)
    let choice f =
      match floor_mod_checked f c with
      | Some r ->
          (match sub_checked r c with
           | None -> None
           | Some r_minus_c ->
               if List.length terms = 1 then Some [ r; r_minus_c ]
               else
                 (match mul_checked r 2 with
                  | Some twice when twice = c -> Some [ r; r_minus_c ]
                  | Some _ ->
                      (match abs_checked r, abs_checked r_minus_c with
                       | Some ar, Some arc ->
                           if ar <= arc then Some [ r ] else Some [ r_minus_c ]
                       | _ -> None)
                  | None -> None))
      | _ -> None
    in
    match option_all (List.map choice factors), floor_mod_checked const c with
    | None, _ | _, None -> None
    | Some choices, Some const_mod_c ->
        (* Cartesian product of per-factor choices. *)
        let rec product = function
          | [] -> [ [] ]
          | xs :: rest ->
              let rest_prod = product rest in
              List.concat_map
                (fun x -> List.map (fun r -> x :: r) rest_prod)
                xs
        in
        let combos = product choices in
        let result = ref None in
        List.iter
          (fun rems ->
            if !result <> None then ()
            else
              let r_terms =
                List.map2
                  (fun r v -> Uop.O.(Uop.const_like v r * v))
                  rems unwrap_terms
              in
              let const_u = Uop.const_like x const_mod_c in
              let rem_sum = Uop.usum (r_terms @ [ const_u ]) in
              let rmin = Uop.vmin rem_sum and rmax = Uop.vmax rem_sum in
              match floor_div_checked rmin c, floor_div_checked rmax c with
              | Some k, Some kmax
                when rmin <> min_int && rmax <> max_int && k = kmax ->
                  if d_op = Ops.Floormod then
                    (match mul_checked k c with
                     | Some offset_v ->
                         let offset = Uop.const_like y offset_v in
                         result := Some Uop.O.(rem_sum - offset)
                     | None -> ())
                  else
                    let coeffs =
                      List.map2
                        (fun r f ->
                          match sub_checked f r with
                          | None -> None
                          | Some diff -> floor_div_checked diff c)
                        rems factors
                    in
                    (match option_all coeffs, floor_div_checked const c with
                     | Some coeffs, Some const_div ->
                         (match add_checked const_div k with
                          | Some const_part_v ->
                              let quot_terms =
                                List.map2
                                  (fun coeff v ->
                                    Uop.O.(Uop.const_like v coeff * v))
                                  coeffs unwrap_terms
                              in
                              let const_part = Uop.const_like x const_part_v in
                              result :=
                                Some (Uop.usum (quot_terms @ [ const_part ]))
                          | None -> ())
                     | _ -> ())
              | _ -> ())
          combos;
        !result

(* nested_div: (x % (k*c)) // c -> (x // c) % k when k > 0. The modulo
   counterpart is handled by remove_nested_mod. *)
let try_nested_div d_op x y c =
  if d_op <> Ops.Floordiv then None
  else if Uop.op x <> Ops.Floormod || Array.length (Uop.src x) <> 2 then None
  else
    let inner_x = (Uop.src x).(0) and inner_d = (Uop.src x).(1) in
    match Uop.divides inner_d c with
    | Some k when Uop.vmin k > 0 -> Some (floormod (floordiv inner_x y) k)
    | _ -> None

(* gcd_with_remainder: factor a common GCD out of numerator. *)
let try_gcd_remainder d_op x y c =
  let x_peeled, const = Uop.pop_const x in
  let uops = Uop.split_uop x_peeled Ops.Add in
  let factors = List.map Uop.const_factor uops in
  match
    List.fold_left
      (fun acc f ->
        match acc with
        | None -> None
        | Some g -> gcd_int_checked g f)
      (Some c) factors
  with
  | None -> None
  | Some g when g <= 1 -> None
  | Some g ->
    match Uop.divides x_peeled g with
    | None -> None
    | Some xp_g ->
        let xp_g = Uop.simplify xp_g in
        match
          floor_div_checked c g,
          floor_div_checked const g,
          floor_mod_checked const g,
          floor_div_checked const c
        with
        | Some c_over_g, Some const_div_g, Some const_rem, Some const_quot ->
            (match floor_mod_checked const_div_g c_over_g with
             | None -> None
             | Some const_mod ->
                 let new_x =
                   if const_mod = 0 then xp_g
                   else Uop.O.(xp_g + Uop.const_like xp_g const_mod)
                 in
                 if Uop.vmin new_x < 0 then None
                 else if d_op = Ops.Floormod then
                   let divisor = Uop.const_like y c_over_g in
                   let factor = Uop.const_like y g in
                   let offset = Uop.const_like y const_rem in
                   Some Uop.O.(floormod new_x divisor * factor + offset)
                 else
                   let divisor = Uop.const_like y c_over_g in
                   let offset = Uop.const_like y const_quot in
                   Some Uop.O.(floordiv new_x divisor + offset))
        | _ -> None

(* divide_by_gcd: variable-denominator fallback. x op y -> (x/g) op (y/g)
   where g = UOp.gcd(x_terms..., y). *)
let try_divide_by_gcd d_op x y =
  let all_uops = Uop.split_uop x Ops.Add in
  let gcd_all = Uop.gcd (all_uops @ [ y ]) |> Uop.simplify in
  match Uop.const_int_value gcd_all with
  | Some 1 -> None
  | _ ->
      (match Uop.divide_exact x gcd_all, Uop.divide_exact y gcd_all with
       | Some x_g, Some y_g ->
           let ret = Uop.alu_binary ~op:d_op ~lhs:x_g ~rhs:y_g in
           if d_op = Ops.Floormod
           then Some Uop.O.(ret * gcd_all)
           else Some ret
       | _ -> None)

(* factor_remainder: (d*x+y) op d -> x + y op d / x for div, y op d for mod. *)
let try_factor_remainder d_op x y =
  if Uop.vmin x < 0 || Uop.vmin y < 0 then None
  else
    let all_uops = Uop.split_uop x Ops.Add in
    let quo = ref [] and rem = ref [] in
    let failed = ref false in
    List.iter (fun u ->
      if !failed then ()
      else match Uop.divide_exact u y with
      | Some q -> quo := q :: !quo
      | None ->
          (match Uop.const_int_value y, Uop.const_factor u with
           | Some y_c, c when y_c > 0 ->
               (match floor_mod_checked c y_c with
                | Some c_mod when c_mod <> c ->
                    (match Uop.divides u c with
                     | Some u_c ->
                         let rem_coeff = Uop.const_like y c_mod in
                         rem := Uop.O.(u_c * rem_coeff) :: !rem;
                         if d_op = Ops.Floordiv then
                           (match floor_div_checked c y_c with
                            | Some c_div ->
                                let quo_coeff = Uop.const_like y c_div in
                                quo := Uop.O.(u_c * quo_coeff) :: !quo
                            | None -> failed := true)
                         else quo := Uop.const_like u 0 :: !quo
                     | None -> failed := true)
                | _ -> rem := u :: !rem)
           | _ -> rem := u :: !rem)
    ) all_uops;
    if !failed || !quo = [] then None
    else
      let rem_sum =
        match List.rev !rem with
        | [] -> Uop.const_like x 0
        | xs -> Uop.usum xs
      in
      if Uop.vmin rem_sum < 0 then None
      else if d_op = Ops.Floormod
      then Some (floormod rem_sum y)
      else Some Uop.O.(floordiv rem_sum y + Uop.usum (List.rev !quo))

(* nest_by_factor: for each non-trivial common factor [f] of a numerator
   term that also divides [c], try rewriting [x op c] via [x / f] and a
   recursive fold. Picks the smallest result by backward-slice size. *)
let rec try_nest_by_factor d_op x y c =
  let uops, const, terms, factors = decompose_numerator x in
  (* Collect candidate divisors: [|f|] for non-constant terms with
     [1 < |f| < c] and [f] dividing [c]. *)
  let divs =
    List.fold_left2 (fun acc u f ->
      match abs_checked f, floor_mod_checked c f with
      | Some af, Some 0
        when Uop.op u <> Ops.Const
             && af > 1 && af < c && f <> 0 && not (List.mem af acc) ->
          af :: acc
      | _ -> acc) [] uops factors
  in
  let build_mod_candidate newxs div_ =
    let b_parts = ref [] in
    let failed = ref false in
    List.iter2 (fun f t_opt ->
      match floor_mod_checked f div_, t_opt with
      | Some r, Some t when r <> 0 ->
          b_parts := Uop.O.(Uop.const_like t r * t) :: !b_parts
      | Some _, _ -> ()
      | None, _ -> failed := true) factors terms;
    match !failed, floor_mod_checked const div_, floor_div_checked c div_ with
    | false, Some const_r, Some c_div_v ->
        if const_r <> 0 then b_parts := Uop.const_like x const_r :: !b_parts;
        let b =
          match !b_parts with
          | [] -> Uop.const_like x 0
          | xs -> Uop.usum (List.rev xs)
        in
        if Uop.vmin b >= 0 && Uop.vmax b < div_ then
          let c_div = Uop.const_like x c_div_v in
          let factor_u = Uop.const_like x div_ in
          Some Uop.O.(floormod newxs c_div * factor_u + b)
        else None
    | _ -> None
  in
  let try_div best div_ =
    let div_u = Uop.const_like x div_ in
    let x_div = floordiv x div_u in
    match fold_divmod_general_rec x_div with
    | None -> best
    | Some newxs ->
        let candidate_and_size =
          if d_op = Ops.Floordiv then
            (match floor_div_checked c div_ with
             | Some c_div ->
                 let divisor = Uop.const_like y c_div in
                 Some
                   (List.length (Uop.backward_slice newxs),
                    floordiv newxs divisor)
             | None -> None)
          else if Uop.vmin x >= 0 && Uop.vmin newxs >= 0 then
            (match build_mod_candidate newxs div_ with
             | Some r -> Some (List.length (Uop.backward_slice r), r)
             | None -> None)
          else None
        in
        (match candidate_and_size with
         | None -> best
         | Some (size, r) ->
             match best with
             | None -> Some (size, r)
             | Some (bsize, _) when size < bsize -> Some (size, r)
             | _ -> best)
  in
  match List.fold_left try_div None divs with
  | Some (_, r) -> Some r
  | None -> None

(* Try strategies in sequence, stopping at the first [Some]. *)
and first_some fs =
  match fs with
  | [] -> None
  | f :: rest ->
      (match f () with Some _ as r -> r | None -> first_some rest)

and try_variable_fallbacks d_op x y =
  first_some [
    (fun () -> try_divide_by_gcd d_op x y);
    (fun () -> try_factor_remainder d_op x y);
  ]

(* Run each strategy in order; stop at the first hit. *)
and fold_divmod_general_rec (d : Uop.t) : Uop.t option =
  let d_op = Uop.op d in
  if d_op <> Ops.Floordiv && d_op <> Ops.Floormod then None
  else if Array.length (Uop.src d) <> 2 then None
  else
    let x = (Uop.src d).(0) and y = (Uop.src d).(1) in
    if Uop.vmin y = 0 && Uop.vmax y = 0 then raise Division_by_zero
    else
      match try_cancel_divmod d_op x y with
      | Some r -> Some r
      | None ->
          (match Uop.const_int_value y with
           | Some c when c > 0 ->
               first_some [
                 (fun () -> try_nested_div d_op x y c);
                 (fun () -> try_remove_nested_mod d_op x y c);
                 (fun () -> try_fold_divmod_congruence d_op x y c);
                 (fun () -> try_gcd_remainder d_op x y c);
                 (fun () -> try_nest_by_factor d_op x y c);
                 (fun () -> try_variable_fallbacks d_op x y);
               ]
           | _ -> try_variable_fallbacks d_op x y)

let rule_fold_divmod_general =
  let open Upat in
  ops ~dtype:Dtype.index ~name:"d" [ Ops.Floordiv; Ops.Floormod ]
  => fun bs -> fold_divmod_general_rec (bs $ "d")

let div_and_mod_symbolic : Upat.Pattern_matcher.t =
  Upat.Pattern_matcher.make [
    rule_nested_div;
    rule_add_const_div;
    rule_fold_divmod_general;
  ]
