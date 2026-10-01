(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type leaves = Nx.packed list

type request = {
  req_carry : leaves;
  req_xs : leaves;
  req_step : leaves -> leaves -> leaves * leaves;
  req_reverse : bool;
}

type result = { r_carry : leaves; r_ys : leaves }

exception Not_staged

let rec stack = function
  | [] | [] :: _ -> []
  | steps ->
      let (Nx.P y0) = List.hd (List.hd steps) in
      let dtype = Nx.dtype y0 in
      let column = List.map (fun y -> Nx.unpack dtype (List.hd y)) steps in
      Nx.P (Nx.stack ~axis:0 column) :: stack (List.map List.tl steps)

(* A carry of its own per step: a compiled call that writes the loop out stores
   each step's carry, so the next step reads its storage and no kernel nests as
   deep as the loop is long. An eager carry is already stored. *)
let own (Nx.P c) =
  match Nx.Repr.v c with
  | Traced _ -> Nx.P (Nx.copy c)
  | Host _ | Placed _ -> Nx.P c

let fold r =
  let (Nx.P x) = List.hd r.req_xs in
  let n = (Nx.shape x).(0) in
  let ys = Array.make n [] in
  let carry = ref r.req_carry in
  for k = 0 to n - 1 do
    let i = if r.req_reverse then n - 1 - k else k in
    let row =
      List.map (fun (Nx.P x) -> Nx.P (Nx.slice [ Nx.I i ] x)) r.req_xs
    in
    let c, y = r.req_step !carry row in
    carry := List.map own c;
    ys.(i) <- y
  done;
  { r_carry = !carry; r_ys = stack (Array.to_list ys) }

(* Transformed scans *)

let rec split n l =
  if n = 0 then ([], l)
  else
    match l with
    | x :: l ->
        let a, b = split (n - 1) l in
        (x :: a, b)
    | [] -> invalid_arg "Scan.split: too few leaves"

let fixpoint active attempt =
  let exception Grow of bool list in
  let rec run active =
    let grow next =
      if List.exists2 (fun a n -> n && not a) active next then
        raise_notrace (Grow (List.map2 ( || ) active next))
    in
    match attempt ~grow active with
    | r -> r
    | exception Grow active -> run active
  in
  run active
