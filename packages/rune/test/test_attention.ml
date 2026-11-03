(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Nx.attention under rune's transformations, which give its operation the
   meaning of the operations it composes, drawn over dense and paged keys,
   lengths, grouped heads, and masks and scores that capture arrays:
   - Rune.jit against the eager kernel, each within nx.mli's bound of the
     exact result, rune.mli's constants for the compiled one;
   - Rune.vmap against each lane eagerly, bit for bit: a vmapped axis is a
     lead axis of the one operation, and nx.cpu's rows are their own;
   - Rune.jvp against central differences, and Rune.grad against them.
   Values are finite: a NaN or an infinity at a hidden key gives the
   compiled and differentiated results NaN where eager's is finite, as
   nx.mli and rune.mli state. *)

open Windtrap
module T = Nx.Ptree

let u = 0x1p-24
let numel = Array.fold_left ( * ) 1

type mask = Unmasked | Causal | Captured

type case = {
  hk : int;
  groups : int;
  tq : int;
  tk : int;
  dk : int;
  dv : int;
  page : int option;  (** Paged by pages of this many keys. *)
  table : int array;  (** Two lead indices' entries in [[-1, 3]]. *)
  lengths : int array option;
  mask : mask;
  alibi : bool;
  seed : float;
}

let pp_case ppf c =
  Format.fprintf ppf
    "hk %d groups %d tq %d tk %d dk %d dv %d page %s lengths %s mask %s \
     alibi %b seed %g"
    c.hk c.groups c.tq c.tk c.dk c.dv
    (match c.page with Some p -> string_of_int p | None -> "none")
    (match c.lengths with
    | Some l -> String.concat "," (Array.to_list (Array.map string_of_int l))
    | None -> "none")
    (match c.mask with
    | Unmasked -> "none"
    | Causal -> "causal"
    | Captured -> "captured")
    c.alibi c.seed

let case =
  let open Gen in
  let* hk = int_range 1 2 in
  let* groups = int_range 1 2 in
  let* tq = int_range 1 4 in
  let* page =
    frequency [ (2, constant None); (1, map Option.some (int_range 1 3)) ]
  in
  let* blocks = int_range 1 4 in
  let tk = match page with Some p -> p * blocks | None -> blocks * 3 in
  let* table = array ~size:(constant (2 * blocks)) (int_range (-1) 3) in
  let* lengths =
    frequency
      [
        (1, constant None);
        (1, map Option.some (array ~size:(constant 2) (int_range 0 (tk + 1))));
      ]
  in
  let* mask = of_list [ Unmasked; Causal; Captured ] in
  let* alibi = bool in
  let* dk = int_range 1 4 in
  let* dv = int_range 1 3 in
  let+ seed = float_range 0. 10. in
  { hk; groups; tq; tk; dk; dv; page; table; lengths; mask; alibi; seed }

let draw c seed s =
  Nx.create Nx.float32 s
    (Array.init (numel s) (fun k ->
         Float.sin ((Float.of_int k *. 2.3) +. seed +. c.seed)))

let hq c = c.hk * c.groups

(* [c]'s call on [q] of [lead] axes, its keys and values [k] and [v] and
   its captures drawn from [c], its table's and lengths' entries from the
   first [lead]. *)
let call c ~lead q k v =
  let hq = hq c in
  let n = numel lead in
  let lengths =
    Option.map
      (fun l -> Nx.create Nx.int64 lead (Array.map Int64.of_int (Array.sub l 0 n)))
      c.lengths
  in
  let mask =
    match c.mask with
    | Unmasked -> None
    | Causal -> Some Nx.causal
    | Captured ->
        let m = Nx.greater_equal_s (draw c 7. [| hq; c.tq; c.tk |]) (-0.5) in
        Some (fun _ -> m)
  in
  let score =
    if not c.alibi then None
    else
      let slopes = Nx.abs (draw c 8. [| hq; 1; 1 |]) in
      Some
        (fun (p : Nx.host Nx.pair) s ->
          Nx.(Infix.(s - (cast float32 (p.q - p.k) * slopes))))
  in
  let kv =
    match c.page with
    | None -> Nx.Dense { k; v }
    | Some p ->
        let pages = c.tk / p in
        let table = Array.map Int64.of_int (Array.sub c.table 0 (n * pages)) in
        Nx.Paged { k; v; table = Nx.create Nx.int64 (Array.append lead [| pages |]) table }
  in
  Nx.attention ?lengths ?mask ?score ~kv q

(* Keys and values for [lead]: dense ones of its axes, or pools of three
   pages. *)
let keys c lead =
  match c.page with
  | None ->
      ( draw c 2. (Array.append lead [| c.hk; c.tk; c.dk |]),
        draw c 3. (Array.append lead [| c.hk; c.tk; c.dv |]) )
  | Some p -> (draw c 2. [| 3; c.hk; p; c.dk |], draw c 3. [| 3; c.hk; p; c.dv |])

(* The sum of the bounds nx.mli states for two computations of [c], one
   within the kernel's [ε] and one within a compiled log-sum-exp's, both
   generous to first order: values lie in [[-1, 1]], so a score is at most
   [a = |scale| dk] plus ALiBi's slopes times [tq + tk], [b], and [η] at
   most [(2 dk + 4) u a + 2u b]. *)
let tolerances c lse =
  let scale = 1. /. Float.sqrt (Float.of_int c.dk) in
  let a = scale *. Float.of_int c.dk in
  let b = a +. Float.of_int (c.tq + c.tk) in
  let eta = (Float.of_int ((2 * c.dk) + 4) *. u *. a) +. (2. *. u *. b) in
  let ln = Float.log (Float.of_int (max 1 c.tk)) in
  let n = (c.tk + 63) / 64 in
  let kernel =
    ((Float.of_int ((13 * n) + 9) +. (5. *. ln)) *. u) +. (u *. Float.abs lse)
  in
  let compiled =
    ((Float.of_int ((2 * c.tk) + 20) +. (10. *. ln)) *. u)
    +. (u *. Float.abs lse)
  in
  let w eps = (2. *. eta) +. eps +. (((2. *. b) +. ln +. 5.) *. u) in
  let gamma = Float.of_int (c.tk + 1) *. 2. *. u in
  ((2. *. eta) +. kernel +. compiled, w kernel +. w compiled +. (2. *. gamma))

let law_jit c =
  cover "paged" (c.page <> None);
  cover "lengths" (c.lengths <> None);
  cover "a captured mask" (c.mask = Captured);
  let lead = [| 2 |] in
  let k, v = keys c lead in
  let q = draw c 1. [| 2; hq c; c.tq; c.dk |] in
  let o, lse = call c ~lead q k v in
  let o' = Rune.jit' (fun q -> fst (call c ~lead q k v)) q in
  let lse' = Rune.jit' (fun q -> snd (call c ~lead q k v)) q in
  let o = Nx.to_array o and o' = Nx.to_array o' in
  let lse = Nx.to_array lse and lse' = Nx.to_array lse' in
  Array.iteri
    (fun i l ->
      if l = Float.neg_infinity then
        equal ~msg:"an empty row's lse" float_exact l lse'.(i)
      else begin
        let tl, to_ = tolerances c l in
        at_most ~msg:"lse" float_exact ~than:tl (Float.abs (l -. lse'.(i)));
        for d = 0 to c.dv - 1 do
          let j = (i * c.dv) + d in
          at_most ~msg:"o" float_exact ~than:to_ (Float.abs (o.(j) -. o'.(j)))
        done
      end)
    lse

let bits a = Array.map Int64.bits_of_float (Nx.to_array a)

(* Lanes of query rows over keys and values the lanes share. *)
let law_vmap c =
  cover "paged" (c.page <> None);
  cover "lengths" (c.lengths <> None);
  cover "a captured mask" (c.mask = Captured);
  let k, v = keys c [||] in
  let q = draw c 1. [| 2; hq c; c.tq; c.dk |] in
  let lane q = fst (call c ~lead:[||] q k v) in
  let each = Nx.stack ~axis:0 [ lane (Nx.get [ 0 ] q); lane (Nx.get [ 1 ] q) ] in
  equal ~msg:"o" (array int64) (bits each) (bits (Rune.vmap' lane q))

(* [<w, f'(q) t>] against central differences of step [h]: within 1e-2 of
   the derivative's magnitude, which the step's truncation and float32's
   rounding over [h] stay below on these values. *)
let law_jvp c =
  cover "paged" (c.page <> None);
  cover "lengths" (c.lengths <> None);
  let lead = [| 2 |] in
  let k, v = keys c lead in
  let q = draw c 1. [| 2; hq c; c.tq; c.dk |] in
  let t = draw c 9. (Nx.shape q) in
  let w = draw c 10. [| 2; hq c; c.tq; c.dv |] in
  let f q = Nx.sum (Nx.mul w (fst (call c ~lead q k v))) in
  let _, d = Rune.jvp' f q t in
  let h = 1e-2 in
  let at s = Nx.item [] (f (Nx.add q (Nx.mul_s t s))) in
  let fd = (at h -. at (-.h)) /. (2. *. h) in
  let d = Nx.item [] d in
  at_most ~msg:"directional derivative" float_exact
    ~than:(1e-2 *. (1. +. Float.abs d))
    (Float.abs (d -. fd))

let test_grad () =
  let w =
    Nx.create Nx.float64 [| 2; 4; 5; 2 |]
      (Array.init 80 (fun k -> Float.sin (Float.of_int k)))
  in
  let slopes = Nx.create Nx.float32 [| 4; 1; 1 |] [| 0.1; 0.2; 0.3; 0.4 |] in
  let lengths = Nx.create Nx.int64 [| 2 |] [| 6L; 7L |] in
  let loss = function
    | [ q; k; v ] ->
        let c x = Nx.cast Nx.float32 x in
        let o, _ =
          Nx.attention ~lengths ~mask:Nx.causal
            ~score:(fun p s ->
              Nx.(Infix.(s - (cast float32 (p.q - p.k) * slopes))))
            ~kv:(Nx.Dense { k = c k; v = c v })
            (c q)
        in
        Nx.sum (Nx.mul w (Nx.cast Nx.float64 o))
    | _ -> invalid_arg "loss"
  in
  let f s seed =
    Nx.create Nx.float64 s
      (Array.init (numel s) (fun k -> Float.sin ((Float.of_int k *. 2.3) +. seed)))
  in
  let params = [ f [| 2; 4; 5; 3 |] 1.; f [| 2; 2; 7; 3 |] 2.; f [| 2; 2; 7; 2 |] 3. ] in
  equal ~msg:"gradient" (result unit string) (Ok ())
    (Rune.check_grads ~eps:1e-2 ~tol:2e-3 T.(list tensor) loss params)

let () =
  exit
    (run "rune attention"
       [
         prop ~count:12 "jit within the bounds of eager"
           (Gen.with_pp pp_case case) law_jit;
         prop ~count:60 "vmap is each lane, bit for bit"
           (Gen.with_pp pp_case case) law_vmap;
         prop ~count:60 "jvp by central differences"
           (Gen.with_pp pp_case case) law_jvp;
         test "gradients by central differences" test_grad;
       ])
