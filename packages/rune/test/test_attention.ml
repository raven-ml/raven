(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Nx.attention under rune's transformations, drawn over dense and paged keys,
   lengths, grouped heads, and masks and scores that capture arrays: - the eager
   derivative past a row's length, against the mask that hides the same pairs; -
   Rune.jvp against central differences, and Rune.grad against them; - Rune.vmap
   against each lane eagerly, bit for bit: a vmapped axis is a lead axis of the
   one operation, and nx.cpu's rows are their own; - Rune.jit's results, as
   storage its kernels write; - Rune.jit against the eager kernel, each within
   nx.mli's bound of the exact result, rune.mli's constants for the compiled
   one, against a float64 reference at every dtype, a sum of zeros +0, paged
   against dense (Attentions, Paged_attentions), and over keys cut across two
   devices against one device; - Rune.grad compiled against eager, and forward
   mode against reverse mode, through the derivative's construct and its
   transpose. Each law draws float32, float64, and a float32 query accumulating
   in float64. A compiled law draws its cases on its examples' forms, which
   compile together, a form's program once a run. Values are finite: a NaN or an
   infinity at a hidden key gives the compiled and differentiated results NaN
   where eager's is finite, as nx.mli and rune.mli state. *)

open Windtrap
module T = Nx.Ptree
module Batch = Rune_support.Batch
module Attentions = Rune_support.Attentions
module Paged = Rune_support.Paged_attentions

let numel = Array.fold_left ( * ) 1

(* A dtype's unit roundoff, [u] in the bounds below. *)
let unit_roundoff (type s) (dt : (float, s) Nx.dtype) =
  match dt with Float64 -> 0x1p-53 | _ -> 0x1p-24

(* A query's dtype and its accumulator's. *)
type dtypes = D : (float, 'a) Nx.dtype * (float, 'b) Nx.dtype -> dtypes

let dtypes_name (D (dt, acc)) =
  if Nx.Dtype.name dt = Nx.Dtype.name acc then Nx.Dtype.name dt
  else Nx.Dtype.name dt ^ " in " ^ Nx.Dtype.name acc

(* float32, float64, and a float32 query in a float64 accumulator. *)
let pairs =
  [
    D (Nx.float32, Nx.float32);
    D (Nx.float64, Nx.float64);
    D (Nx.float32, Nx.float64);
  ]

let is_float64 (D (dt, _)) = Nx.Dtype.name dt = "float64"

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
    "hk %d groups %d tq %d tk %d dk %d dv %d page %s table %s lengths %s mask \
     %s alibi %b seed %g"
    c.hk c.groups c.tq c.tk c.dk c.dv
    (match c.page with Some p -> string_of_int p | None -> "none")
    (String.concat "," (Array.to_list (Array.map string_of_int c.table)))
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

let draw dt c seed s =
  Nx.create dt s
    (Array.init (numel s) (fun k ->
         Float.sin ((Float.of_int k *. 2.3) +. seed +. c.seed)))

let hq c = c.hk * c.groups

(* [c]'s call on [q] of [lead] axes, its keys and values [k] and [v] and
   its captures drawn from [c], its table's and lengths' entries from the
   first [lead], accumulating in [acc]. *)
let call (type a) (acc : (float, a) Nx.dtype) c ~lead
    (q : (float, 's, Nx.host) Nx.t) k v =
  let dt = Nx.dtype q in
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
        let m = Nx.greater_equal_s (draw dt c 7. [| hq; c.tq; c.tk |]) (-0.5) in
        Some (fun _ -> m)
  in
  let score =
    if not c.alibi then None
    else
      let slopes = Nx.abs (draw acc c 8. [| hq; 1; 1 |]) in
      Some
        (fun (p : Nx.host Nx.pair) s ->
          Nx.(Infix.(s - (cast acc (p.q - p.k) * slopes))))
  in
  let kv =
    match c.page with
    | None -> Nx.Dense { k; v }
    | Some p ->
        let pages = c.tk / p in
        let table = Array.map Int64.of_int (Array.sub c.table 0 (n * pages)) in
        Nx.Paged { k; v; table = Nx.create Nx.int64 (Array.append lead [| pages |]) table }
  in
  Nx.attention ?lengths ?mask ?score ~acc ~kv q

(* Keys and values for [lead]: dense ones of its axes, or pools of three
   pages. *)
let keys dt c lead =
  match c.page with
  | None ->
      ( draw dt c 2. (Array.append lead [| c.hk; c.tk; c.dk |]),
        draw dt c 3. (Array.append lead [| c.hk; c.tk; c.dv |]) )
  | Some p ->
      (draw dt c 2. [| 3; c.hk; p; c.dk |], draw dt c 3. [| 3; c.hk; p; c.dv |])

(* The sum of the bounds nx.mli states for two computations of [c] in
   [acc], to first order in its [u], of a query of [dt]: one with nx.cpu's [ε]
   (Nx_kernels_support.cpu) and its exp of 5u, and one with
   rune.mli's compiled [Nx.logsumexp],
   [(1.37 (tk - 1) + 2E + 1 + 3 ln tk) u + u |lse|] for [E = 9] ulps, and
   its exp of 9 ulps, 19u, in [w]. Values lie in [[-1, 1]], so a score is
   at most [a = |scale| dk] plus ALiBi's slopes times [tq + tk], [b], and
   [η] at most [(2 dk + 4) u a + 2u b]. *)
let tolerances dt acc c lse =
  let u = unit_roundoff acc and uq = unit_roundoff dt in
  let scale = 1. /. Float.sqrt (Float.of_int c.dk) in
  let a = scale *. Float.of_int c.dk in
  let b = a +. Float.of_int (c.tq + c.tk) in
  let eta = (Float.of_int ((2 * c.dk) + 4) *. u *. a) +. (2. *. u *. b) in
  let ln = Float.log (Float.of_int (max 1 c.tk)) in
  let kernel =
    (Option.get Nx_kernels_support.cpu.attention).epsilon
      (Nx_array.Dtype.Any acc) (max 1 c.tk) lse
  in
  let compiled =
    (((1.37 *. Float.of_int (c.tk - 1)) +. 19. +. (3. *. ln)) *. u)
    +. (u *. Float.abs lse)
  in
  let w ~exp eps = (2. *. eta) +. eps +. (((2. *. b) +. ln +. exp) *. u) in
  let gamma = Float.of_int (c.tk + 1) *. 2. *. u in
  (* Each [o], at most 1, rounds once into [dt]: [u{_q} |o|] each. *)
  ( (2. *. eta) +. kernel +. compiled,
    w ~exp:5. kernel +. w ~exp:19. compiled +. (2. *. gamma) +. (2. *. uq) )

(* A case that reaches every cover of each law on every run: paged, with
   lengths and a captured mask, which [law_cut] makes causal. And dense
   keys of two query rows and three keys under a captured mask and ALiBi,
   whose host kernel converts the differences [q - k], int64s of two
   32-bit halves, to float32 side by side: LLVM 20's straight-line
   vectorizer narrowed the unsigned low halves to 8 bits, giving scores
   off by about [2^32] slopes. *)
let examples =
  [
    {
      hk = 1;
      groups = 1;
      tq = 2;
      tk = 3;
      dk = 1;
      dv = 2;
      page = None;
      table = [| 0; 0 |];
      lengths = None;
      mask = Captured;
      alibi = true;
      seed = 0.;
    };
    {
      hk = 1;
      groups = 2;
      tq = 2;
      tk = 6;
      dk = 3;
      dv = 2;
      page = Some 2;
      table = [| 0; 1; -1; 2; 3; 0 |];
      lengths = Some [| 3; 6 |];
      mask = Captured;
      alibi = true;
      seed = 1.5;
    };
  ]

(* The forms a law draws its cases on: [examples], then each mask, with ALiBi
   and without, over dense keys and over pages of one, two and three keys,
   decode rows and prefills of up to five, more rows than keys under a causal
   mask and a captured one, one and two key heads of one and two query heads
   each, one to four blocks of keys of one to four dimensions, values of one to
   three, lengths of no key, of every key and past them. *)
let forms =
  examples
  @ [
      {
        hk = 2;
        groups = 1;
        tq = 4;
        tk = 12;
        dk = 4;
        dv = 3;
        page = None;
        table = [| 0; 0 |];
        lengths = Some [| 12; 5 |];
        mask = Causal;
        alibi = false;
        seed = 3.;
      };
      {
        hk = 2;
        groups = 2;
        tq = 1;
        tk = 9;
        dk = 1;
        dv = 1;
        page = Some 3;
        table = [| 2; -1; 0; 3; 1; 2 |];
        lengths = None;
        mask = Unmasked;
        alibi = true;
        seed = 6.;
      };
      {
        hk = 1;
        groups = 2;
        tq = 3;
        tk = 9;
        dk = 2;
        dv = 1;
        page = None;
        table = [| 0; 0 |];
        lengths = Some [| 0; 10 |];
        mask = Unmasked;
        alibi = false;
        seed = 4.;
      };
      {
        hk = 2;
        groups = 2;
        tq = 1;
        tk = 6;
        dk = 4;
        dv = 3;
        page = None;
        table = [| 0; 0 |];
        lengths = None;
        mask = Unmasked;
        alibi = true;
        seed = 5.;
      };
      {
        hk = 1;
        groups = 1;
        tq = 4;
        tk = 4;
        dk = 3;
        dv = 2;
        page = Some 1;
        table = [| 3; 0; -1; 1; 2; 1; 0; 3 |];
        lengths = Some [| 4; 2 |];
        mask = Unmasked;
        alibi = false;
        seed = 7.;
      };
      {
        hk = 2;
        groups = 1;
        tq = 3;
        tk = 6;
        dk = 1;
        dv = 3;
        page = None;
        table = [| 0; 0 |];
        lengths = Some [| 6; 0 |];
        mask = Causal;
        alibi = true;
        seed = 8.;
      };
      {
        hk = 1;
        groups = 2;
        tq = 1;
        tk = 2;
        dk = 2;
        dv = 1;
        page = Some 2;
        table = [| -1; 1 |];
        lengths = None;
        mask = Causal;
        alibi = false;
        seed = 9.;
      };
      {
        hk = 2;
        groups = 2;
        tq = 4;
        tk = 12;
        dk = 3;
        dv = 2;
        page = Some 3;
        table = [| 1; 3; -1; 0; 2; 2; 3; 0 |];
        lengths = Some [| 13; 7 |];
        mask = Causal;
        alibi = true;
        seed = 2.;
      };
      {
        hk = 2;
        groups = 2;
        tq = 2;
        tk = 12;
        dk = 4;
        dv = 1;
        page = None;
        table = [| 0; 0 |];
        lengths = Some [| 1; 12 |];
        mask = Captured;
        alibi = false;
        seed = 4.5;
      };
      {
        hk = 1;
        groups = 1;
        tq = 3;
        tk = 2;
        dk = 1;
        dv = 3;
        page = Some 1;
        table = [| 3; -1; 0; 2 |];
        lengths = Some [| 0; 2 |];
        mask = Captured;
        alibi = false;
        seed = 8.5;
      };
      {
        hk = 1;
        groups = 2;
        tq = 5;
        tk = 3;
        dk = 2;
        dv = 2;
        page = None;
        table = [| 0; 0 |];
        lengths = Some [| 3; 1 |];
        mask = Causal;
        alibi = false;
        seed = 9.5;
      };
    ]

(* [forms] in each of [dts]. *)
let typed dts forms =
  List.concat_map (fun d -> List.map (fun c -> (d, c)) forms) dts

let pp_typed ppf (d, c) = Format.fprintf ppf "%s %a" (dtypes_name d) pp_case c

(* A case of [c]'s generator, in one of [dts]. *)
let any_of dts c = Gen.with_pp pp_typed (Gen.pair (Gen.of_list dts) c)

(* A case of the form, dtypes included, of one of [cases]: its values, table
   entries and lengths drawn anew. *)
let redrawn cases =
  let open Gen in
  let* d, c = of_list cases in
  let* seed = float_range 0. 10. in
  let* table =
    array ~size:(constant (Array.length c.table)) (int_range (-1) 3)
  in
  let+ lengths =
    match c.lengths with
    | None -> constant None
    | Some l ->
        map Option.some
          (array ~size:(constant (Array.length l)) (int_range 0 (c.tk + 1)))
  in
  (d, { c with seed; table; lengths })

type out = O | Lse

(* [c]'s call compiled over its query, the program holding its keys, values and
   captures, returning [o] or [lse] as [out] says: the program of a user who
   reads only that one. Rune.jit within the sum of eager's bound and the
   compiled call's ([tolerances]) of eager. *)
let law_jit_call out (D (dt, acc), c) =
  let lead = [| 2 |] in
  let k, v = keys dt c lead in
  let q = draw dt c 1. [| 2; hq c; c.tq; c.dk |] in
  let f q = call acc c ~lead q k v in
  match out with
  | O -> Batch.call T.tensor q T.tensor (fun q -> fst (f q)) Nx.to_array
  | Lse -> Batch.call T.tensor q T.tensor (fun q -> snd (f q)) Nx.to_array

let law_jit (D (dt, acc), c) compiled =
  cover "paged" (c.page <> None);
  cover "lengths" (c.lengths <> None);
  cover "a captured mask" (c.mask = Captured);
  let lead = [| 2 |] in
  let k, v = keys dt c lead in
  let q = draw dt c 1. [| 2; hq c; c.tq; c.dk |] in
  let o, lse = call acc c ~lead q k v in
  let o = Nx.to_array o and lse = Nx.to_array lse in
  let o', lse' = compiled () in
  Array.iteri
    (fun i l ->
      if l = Float.neg_infinity then
        equal ~msg:"an empty row's lse" float_exact l lse'.(i)
      else begin
        let tl, to_ = tolerances dt acc c l in
        at_most ~msg:"lse" float_exact ~than:tl (Float.abs (l -. lse'.(i)));
        for d = 0 to c.dv - 1 do
          let j = (i * c.dv) + d in
          at_most ~msg:"o" float_exact ~than:to_ (Float.abs (o.(j) -. o'.(j)))
        done
      end)
    lse

let bits a = Array.map Int64.bits_of_float (Nx.to_array a)

(* Keys and values cut across two devices along their keys, Rune.jit against the
   eager call on one device, and the gradient of [Σ o] within 1e-3 relative of
   eager's: each part's coordinates are the whole's, so a causal or captured
   mask hides the same pairs. nx.mli's bound over parts doubles [η] and takes
   [ε] three times the parts' plus a Logsumexp of two: within four times the
   one-device sum of bounds. *)
module Two = (val Nx.devices [ Nx_support.memory 0; Nx_support.memory 1 ])

(* [c] cut: dense keys of twice its length, without ALiBi, a captured mask made
   causal. *)
let cut c =
  let c = { c with page = None; tk = 2 * c.tk; alibi = false } in
  if c.mask = Captured then { c with mask = Causal } else c

(* [c]'s call of [q], dense keys and values [k] and [v] of one lead axis of two
   indices, on any devices, and the loss [Σ o] of the query, keys and values. *)
let cut_call (type a) (acc : (float, a) Nx.dtype) c =
  let f (type d) (q : (float, _, d) Nx.t) k v =
    let lengths =
      Option.map
        (fun l ->
          Nx.place (Nx.beside q)
            (Nx.create Nx.int64 [| 2 |]
               (Array.map Int64.of_int (Array.sub l 0 2))))
        c.lengths
    in
    let mask = if c.mask = Causal then Some Nx.causal else None in
    Nx.attention ?lengths ?mask ~acc ~kv:(Nx.Dense { k; v }) q
  in
  let loss = function
    | [ q; k; v ] -> Nx.sum (fst (f q k v))
    | _ -> invalid_arg "loss"
  in
  (f, loss)

(* [c] cut: its query, keys and values placed over [Two], its keys and values
   split along the keys, and its call and loss. *)
let cut_parts dt acc c =
  let c = cut c in
  let k, v = keys dt c [| 2 |] in
  let q = draw dt c 1. [| 2; hq c; c.tq; c.dk |] in
  let split x = Nx.place (Nx.Placement.split ~axis:2 Two.v) x in
  let f, loss = cut_call acc c in
  ((Nx.place Two.on q, split k, split v), f, loss)

let host x = Nx.to_array (Nx.place Nx.Host.on x)

(* [c] cut, compiled over its parts: [o] and [lse]. *)
let law_cut_call (D (dt, acc), c) =
  let x, f, _ = cut_parts dt acc c in
  Batch.call
    T.(triple tensor tensor tensor)
    x
    T.(pair tensor tensor)
    (fun (q, k, v) -> f q k v)
    (fun (o, lse) -> (host o, host lse))

(* [c] cut, compiled over its parts: the gradient of [Σ o]. *)
let law_cut_grad_call (D (dt, acc), c) =
  let x, _, loss = cut_parts dt acc c in
  Batch.call
    T.(triple tensor tensor tensor)
    x
    T.(list tensor)
    (fun (q, k, v) -> Rune.grad T.(list tensor) loss [ q; k; v ])
    (List.map host)

let law_cut (D (dt, acc), c) compiled =
  cover "lengths" (c.lengths <> None);
  let c = cut c in
  cover "causal" (c.mask = Causal);
  cover "a key on each device" (c.tk = 2);
  cover "causal rows past the keys" (c.mask = Causal && c.tq > c.tk);
  let k, v = keys dt c [| 2 |] in
  let q = draw dt c 1. [| 2; hq c; c.tq; c.dk |] in
  let f, loss = cut_call acc c in
  let o, lse = f q k v in
  let g = Rune.grad T.(list tensor) loss [ q; k; v ] in
  let (o', lse'), g' = compiled () in
  (* The gradient of [Σ o], compiled over the parts, against eager's. *)
  List.iter2
    (fun g g' ->
      Array.iteri
        (fun i x ->
          at_most ~msg:"gradient" float_exact
            ~than:(1e-3 *. (1. +. Float.abs x))
            (Float.abs (x -. g'.(i))))
        (Nx.to_array g))
    g g';
  let o = Nx.to_array o and lse = Nx.to_array lse in
  Array.iteri
    (fun i l ->
      if l = Float.neg_infinity then
        equal ~msg:"an empty row's lse" float_exact l lse'.(i)
      else begin
        let tl, to_ = tolerances dt acc c l in
        at_most ~msg:"lse" float_exact ~than:(4. *. tl)
          (Float.abs (l -. lse'.(i)));
        for d = 0 to c.dv - 1 do
          let j = (i * c.dv) + d in
          at_most ~msg:"o" float_exact ~than:(4. *. to_)
            (Float.abs (o.(j) -. o'.(j)))
        done
      end)
    lse

(* The forms the cut law draws its cases on: [examples], their captured masks
   made causal, then dense keys of two key heads under lengths and no mask,
   [forms]' unmasked prefill under lengths, unmasked decode row and causal
   decode row of two pages, and a causal prefill of four rows over one key, a
   key on each device once cut: keys of 2 to 24 cut in two, decode rows and
   prefills of up to four, causal rows past the keys, each mask with lengths and
   without, each pair of one and two key and query heads. *)
let cut_forms =
  examples
  @ [ { (List.nth forms 2) with mask = Unmasked } ]
  @ List.map (List.nth forms) [ 4; 5; 8 ]
  @ [
      {
        hk = 1;
        groups = 2;
        tq = 4;
        tk = 1;
        dk = 2;
        dv = 1;
        page = None;
        table = [| 0; 0 |];
        lengths = Some [| 2; 1 |];
        mask = Causal;
        alibi = false;
        seed = 10.5;
      };
    ]

(* Lanes of query rows over keys and values the lanes share. *)
let law_vmap (D (dt, acc), c) =
  cover "paged" (c.page <> None);
  cover "lengths" (c.lengths <> None);
  cover "a captured mask" (c.mask = Captured);
  let k, v = keys dt c [||] in
  let q = draw dt c 1. [| 2; hq c; c.tq; c.dk |] in
  let lane q = fst (call acc c ~lead:[||] q k v) in
  let each =
    Nx.stack ~axis:0 [ lane (Nx.get [ 0 ] q); lane (Nx.get [ 1 ] q) ]
  in
  equal ~msg:"o" (array int64) (bits each) (bits (Rune.vmap' lane q))

(* Lanes of pools of their own over one query: vmap reads them as one pool of
   every lane's pages, each lane's table entries renumbered into it and one out
   of range, -1 or past the pages, kept out of range. *)
let law_vmap_pools (D (dt, acc), c) =
  let page = Option.value c.page ~default:2 in
  let blocks = Array.length c.table / 2 in
  let table = Array.copy c.table in
  table.(0) <- -1;
  table.(blocks - 1) <- 3;
  let c =
    { c with page = Some page; tk = page * blocks; table; lengths = None }
  in
  cover "an entry out of range in a lane's table"
    (Array.exists (fun e -> e < 0 || e >= 3) (Array.sub c.table 0 blocks));
  let pools = draw dt c 2. [| 2; 3; c.hk; page; c.dk + c.dv |] in
  let q = draw dt c 1. [| hq c; c.tq; c.dk |] in
  let lane pool =
    let k = Nx.slice [ A; A; A; R (0, c.dk) ] pool in
    let v = Nx.slice [ A; A; A; R (c.dk, c.dk + c.dv) ] pool in
    fst (call acc c ~lead:[||] q k v)
  in
  let each =
    Nx.stack ~axis:0 [ lane (Nx.get [ 0 ] pools); lane (Nx.get [ 1 ] pools) ]
  in
  equal ~msg:"o" (array int64) (bits each) (bits (Rune.vmap' lane pools))

(* [<w, f'(q) t>] against central differences of step [h], within [tol (1 +
   |d|)]. A float32 [f], whose rounding over [h] dominates, takes [h = 1e-2] and
   [tol = 1e-2]. In float64, to first order in [u = 2{^-53}], the difference is
   within [e / h + h² F / 6] of the derivative, [e] bounding [f]'s computed
   error and [F] its third derivative: - [e]. Values lie in [[-1, 1]], [dk <=
   4], [tk <= 12], [tq <= 4] and ALiBi's slopes are at most 1, so [a = √dk <= 2]
   and [b <= a + 11]. nx.mli's bound has [η <= 12u a + 2u b <= 50u], [ε <= (22 +
   7 ln 12) u + u (b + ln 12) <= 55u] and [(2b + ln tk + 5) u <= 34u]: [w <=
   189u], and with [γ(13, 2u)] and [o]'s rounding each element of [o] is within
   [216u]. [f] sums at most 96 products by [|w| <= 1], each rounded, within
   [γ(95)]: with the rounding of [q + h t], [e <= 32000u <= 3.6e-12]. - [F].
   Along [t] each score moves by [g = scale t·k], [|g| <= √dk <= 2]. For a
   complex step [|σ| <= ρ = π / 8] the weights' terms turn by at most [π / 4],
   so their sum is at least [cos (π / 4)] of their moduli' and [|o| <= √2];
   Cauchy's estimate gives [|o'''| <= 3! √2 / ρ³ <= 140], and [F <= 96 · 140].
   At [h = 1e-5] that is [3.6e-7 + 2.3e-7]; [d]'s own error is of [e]'s order
   times the derivative's, far below: [tol = 1e-6], where float32's rounding of
   [o] alone, [2{^-24}] each, would give [0.6]. *)
let law_jvp (D (dt, acc), c) =
  let h, tol =
    if is_float64 (D (dt, acc)) then (1e-5, 1e-6) else (1e-2, 1e-2)
  in
  cover "paged" (c.page <> None);
  cover "lengths" (c.lengths <> None);
  let lead = [| 2 |] in
  let k, v = keys dt c lead in
  let q = draw dt c 1. [| 2; hq c; c.tq; c.dk |] in
  let t = draw dt c 9. (Nx.shape q) in
  let w = draw dt c 10. [| 2; hq c; c.tq; c.dv |] in
  let f q = Nx.sum (Nx.mul w (fst (call acc c ~lead q k v))) in
  let _, d = Rune.jvp' f q t in
  let at s = Nx.item [] (f (Nx.add q (Nx.mul_s t s))) in
  let fd = (at h -. at (-.h)) /. (2. *. h) in
  let d = Nx.item [] d in
  at_most ~msg:"directional derivative" float_exact
    ~than:(tol *. (1. +. Float.abs d))
    (Float.abs (d -. fd))

(* Rune.grad of a loss over attention against central differences, its
   query, keys and values float64. In float32, each cast down and
   accumulating in float32, within [2e-3] at a step of [1e-2], set by hand
   as rounding dominates. In float64 the difference is within
   [e / h + h² F / 6] of the derivative, as [law_jvp] derives, along
   check_grads' directions, of entries at most [2.1]:
   - [e]. [dk = 3], [tk = 7] and slopes of at most [0.4] give [a <= √3]
     and [b <= 4.2]: [η <= 26u], [ε <= 42u], [(2b + ln tk + 5) u <= 16u],
     [w <= 110u], each of the 80 elements of [o] within [127u], and the
     loss, with its products, its sum and the parameters' rounding, within
     [e <= 19000u <= 2.1e-12].
   - [F]. A score is [scale (q + σ t_q)·(k + σ t_k)], of slope at most
     [scale · 3 · 4.2 <= 7.3] and curvature at most [scale · 3 · 4.41
     <= 7.7]; at [|σ| <= ρ = 0.095] its terms turn by at most
     [0.76 < π / 4], a value is at most [1.2], so [|o| <= 1.2 / cos 0.76
     <= 1.67] and [|o'''| <= 3! · 1.67 / ρ³ <= 11700]: [F <= 80 · 11700].
   At [h = 2e-6] that is [1.05e-6 + 6.3e-7]: [tol = 2e-6], where float32's
   rounding of [o] alone, [2{^-24}] each, would give about [2.4]. Lengths
   [0] and [3], a row of no key and rows of fewer keys, only lower these
   bounds. *)
let grads (type s) (dt : (float, s) Nx.dtype) ~eps ~tol lens () =
  let w =
    Nx.create Nx.float64 [| 2; 4; 5; 2 |]
      (Array.init 80 (fun k -> Float.sin (Float.of_int k)))
  in
  let slopes = Nx.create dt [| 4; 1; 1 |] [| 0.1; 0.2; 0.3; 0.4 |] in
  let lengths = Nx.create Nx.int64 [| 2 |] lens in
  let loss = function
    | [ q; k; v ] ->
        let c x = Nx.cast dt x in
        let o, _ =
          Nx.attention ~lengths ~mask:Nx.causal
            ~score:(fun p s -> Nx.(Infix.(s - (cast dt (p.q - p.k) * slopes))))
            ~acc:dt
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
    (Rune.check_grads ~eps ~tol T.(list tensor) loss params)

(* [grads]'s loss, in float64, against its slopes as well: a float capture's
   cotangent. As [grads] derives, along directions of entries at most [2.1],
   a slope moves a score by at most [6 · 2.1] more, so [ρ = 0.035] and
   [|o'''| <= 3! · 1.67 / ρ³ <= 235000]: at [h = 2e-6] that is [1.05e-6 +
   1.3e-5], [tol = 2e-5]. *)
let capture_grads () =
  let w =
    Nx.create Nx.float64 [| 2; 4; 5; 2 |]
      (Array.init 80 (fun k -> Float.sin (Float.of_int k)))
  in
  let lengths = Nx.create Nx.int64 [| 2 |] [| 6L; 7L |] in
  let loss = function
    | [ q; k; v; slopes ] ->
        let o, _ =
          Nx.attention ~lengths ~mask:Nx.causal
            ~score:(fun p s ->
              Nx.(Infix.(s - (cast Nx.float64 (p.q - p.k) * slopes))))
            ~acc:Nx.float64
            ~kv:(Nx.Dense { k; v })
            q
        in
        Nx.sum (Nx.mul w o)
    | _ -> invalid_arg "loss"
  in
  let f s seed =
    Nx.create Nx.float64 s
      (Array.init (numel s) (fun k -> Float.sin ((Float.of_int k *. 2.3) +. seed)))
  in
  let slopes = Nx.create Nx.float64 [| 4; 1; 1 |] [| 0.1; 0.2; 0.3; 0.4 |] in
  let params =
    [ f [| 2; 4; 5; 3 |] 1.; f [| 2; 2; 7; 3 |] 2.; f [| 2; 2; 7; 2 |] 3.; slopes ]
  in
  equal ~msg:"gradient" (result unit string) (Ok ())
    (Rune.check_grads ~eps:2e-6 ~tol:2e-5 T.(list tensor) loss params)

(* The same law through a page table of three entries a lead index, of two
   keys a page, pages [0, 2, 0] and [1, -1, 2]: page 0 is read twice by the
   first lead index and page 2 by both, so each cotangent is a sum, and the
   entry [-1], out of range, hides its keys. In float64, [lse]'s cotangent
   non-zero. As [grads] derives, whose [e / h + h² F / 6] is [1.05e-6 + 6.2e-7]
   for [o], the 40 elements of [lse], of magnitude at most [4.4], add [8e-7]:
   [tol = 3e-6]. *)
let paged_grads () =
  let f s seed =
    Nx.create Nx.float64 s
      (Array.init (numel s) (fun k -> Float.sin ((Float.of_int k *. 2.3) +. seed)))
  in
  let table = Nx.create Nx.int64 [| 2; 3 |] [| 0L; 2L; 0L; 1L; -1L; 2L |] in
  let w = f [| 2; 4; 5; 2 |] 5. and wl = f [| 2; 4; 5 |] 6. in
  let loss = function
    | [ q; k; v ] ->
        let o, lse =
          Nx.attention ~acc:Nx.float64 ~kv:(Nx.Paged { k; v; table }) q
        in
        Nx.add (Nx.sum (Nx.mul w o)) (Nx.sum (Nx.mul wl lse))
    | _ -> invalid_arg "loss"
  in
  let params =
    [ f [| 2; 4; 5; 3 |] 1.; f [| 3; 2; 2; 3 |] 2.; f [| 3; 2; 2; 2 |] 3. ]
  in
  equal ~msg:"gradient" (result unit string) (Ok ())
    (Rune.check_grads ~eps:2e-6 ~tol:3e-6 T.(list tensor) loss params)

(* [c]'s query, keys, values and the slopes its score captures, in [acc], and a
   loss over its [o] and [lse] of them, [lse]'s cotangent non-zero. *)
let grad_loss (type a) (acc : (float, a) Nx.dtype) c =
  let lead = [| 2 |] in
  let k, v = keys acc c lead in
  let q = draw acc c 1. [| 2; hq c; c.tq; c.dk |] in
  let slopes = Nx.abs (draw acc c 8. [| hq c; 1; 1 |]) in
  let w = draw acc c 10. [| 2; hq c; c.tq; c.dv |] in
  let w2 = draw acc c 11. [| 2; hq c; c.tq |] in
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
        let m = Nx.greater_equal_s (draw acc c 7. [| hq c; c.tq; c.tk |]) (-0.5) in
        Some (fun _ -> m)
  in
  let kv k v =
    match c.page with
    | None -> Nx.Dense { k; v }
    | Some p ->
        let pages = c.tk / p in
        let table = Array.map Int64.of_int (Array.sub c.table 0 (n * pages)) in
        Nx.Paged
          { k; v; table = Nx.create Nx.int64 (Array.append lead [| pages |]) table }
  in
  let loss = function
    | [ q; k; v; slopes ] ->
        let score (p : Nx.host Nx.pair) s =
          Nx.(Infix.(s - (cast acc (p.q - p.k) * slopes)))
        in
        let o, lse = Nx.attention ?lengths ?mask ~score ~acc ~kv:(kv k v) q in
        let finite x = Nx.where (Nx.isfinite x) x (Nx.zeros_like x) in
        Nx.add (Nx.sum (Nx.mul w o)) (Nx.sum (Nx.mul w2 (finite lse)))
    | _ -> invalid_arg "loss"
  in
  ([ q; k; v; slopes ], loss)

let law_grad_call (D (_, acc), c) =
  let c = { c with alibi = true } in
  let xs, loss = grad_loss acc c in
  Batch.call
    T.(list tensor)
    xs
    T.(list tensor)
    (Rune.grad T.(list tensor) loss)
    (List.map Nx.to_array)

(* The gradient of [grad_loss] in the query, keys, values and slopes, by
   [Attention_linear] transposed: compiled against eager, and reverse mode
   against forward mode along a direction. Each computes the same sums of
   products of the pair loop in another order: in float64 within 1e-9 relative,
   in float32 within 1e-3, set by hand. *)
let law_grad ((D (_, acc) as d), c) compiled =
  cover "paged" (c.page <> None);
  cover "lengths" (c.lengths <> None);
  let tol = if is_float64 d then 1e-9 else 1e-3 in
  let c = { c with alibi = true } in
  let xs, loss = grad_loss acc c in
  let g = Rune.grad T.(list tensor) loss xs in
  (* Eager forward against eager reverse first, so that a failure of the
     compiled comparison below lies on the compiled side. *)
  let ts =
    List.mapi (fun i x -> draw acc c (20. +. Float.of_int i) (Nx.shape x)) xs
  in
  let _, d = Rune.jvp T.(list tensor) T.tensor loss xs ts in
  let d = Nx.item [] d in
  let dot =
    List.fold_left2 (fun a g t -> a +. Nx.item [] (Nx.sum (Nx.mul g t))) 0. g ts
  in
  at_most ~msg:"forward against reverse" float_exact
    ~than:(tol *. (1. +. Float.abs dot))
    (Float.abs (d -. dot));
  List.iter2
    (fun (name, g) g' ->
      Array.iteri
        (fun i x ->
          at_most
            ~msg:(Printf.sprintf "compiled gradient in %s at %d" name i)
            float_exact
            ~than:(tol *. (1. +. Float.abs x))
            (Float.abs (x -. g'.(i))))
        (Nx.to_array g))
    (List.combine [ "q"; "k"; "v"; "slopes" ] g)
    (compiled ())

(* The eager derivative under lengths computes over groups of lead indices,
   those whose lengths round up to one power of two of keys, at most [tk], and
   puts their results back in the lead's order. Covered: groups that interleave,
   a group of one row among others, and a run of several rows of a group after
   its first row, whose results lie inside the group's rather than at their
   start. [lens] are clamped into [[0, tk]]. *)
let arrangements tk lens =
  let group l =
    let rec up c = if c >= l then c else up (2 * c) in
    if l = 0 then 0 else min tk (up 1)
  in
  let g = Array.map group lens in
  let n = Array.length g in
  let rows x = Array.fold_left (fun k y -> if y = x then k + 1 else k) 0 g in
  let first x =
    let rec go i = if g.(i) = x then i else go (i + 1) in
    go 0
  in
  (* Each maximal run of one group: its start and its rows. *)
  let rec runs i =
    if i = n then []
    else
      let rec stop j = if j < n && g.(j) = g.(i) then stop (j + 1) else j in
      let j = stop i in
      (i, j - i) :: runs j
  in
  let runs = runs 0 in
  let groups = List.sort_uniq compare (Array.to_list g) in
  cover "groups that interleave" (List.length runs > List.length groups);
  cover "a group of one row among others"
    (List.length groups > 1 && List.exists (fun x -> rows x = 1) groups);
  cover "a run of several rows after its group's first"
    (List.exists (fun (i, k) -> k > 1 && first g.(i) < i) runs)

(* Lengths hide the keys past each lead index's length as a mask of the same
   pairs does: the gradient in the query, keys, values and slopes and the
   tangent along a direction, with lengths, whose eager derivative computes over
   each lead index's keys, against the mask's, whose derivative computes the
   whole pair loop, over two lead axes. The same sums of products, without some
   zero terms and in another order: in float64 within 1e-9 relative, in float32
   within 1e-3, set by hand as in [law_grad]. *)
let law_lengths ((D (_, acc) as d), c) =
  let tol = if is_float64 d then 1e-9 else 1e-3 in
  let c = { c with page = None; alibi = true } in
  let lens = Option.value c.lengths ~default:[| 0; c.tk |] in
  let clamped = Array.map (fun l -> max 0 (min c.tk l)) lens in
  Array.iter
    (fun l ->
      cover "a row of no key" (l = 0);
      cover "a row of every key" (l = c.tk);
      cover "a row of some keys" (l > 0 && l < c.tk))
    clamped;
  arrangements c.tk clamped;
  let lead = [| 1; Array.length lens |] in
  let with_lead s = Array.append lead s in
  let k, v = keys acc c lead in
  let q = draw acc c 1. (with_lead [| hq c; c.tq; c.dk |]) in
  let slopes = Nx.abs (draw acc c 8. [| hq c; 1; 1 |]) in
  let w = draw acc c 10. (with_lead [| hq c; c.tq; c.dv |]) in
  let w2 = draw acc c 11. (with_lead [| hq c; c.tq |]) in
  let lengths = Nx.create Nx.int64 lead (Array.map Int64.of_int lens) in
  let len =
    Nx.create Nx.int64 (with_lead [| 1; 1; 1 |]) (Array.map Int64.of_int clamped)
  in
  let captured =
    Nx.greater_equal_s (draw acc c 7. [| hq c; c.tq; c.tk |]) (-0.5)
  in
  let mask =
    match c.mask with
    | Unmasked -> None
    | Causal -> Some Nx.causal
    | Captured -> Some (fun _ -> captured)
  in
  (* [mask] with the lengths' pairs hidden by a captured [len]. *)
  let hiding (p : Nx.host Nx.pair) =
    let within = Nx.less p.k len in
    match c.mask with
    | Unmasked -> within
    | Causal ->
        Nx.logical_and within
          (Nx.less_equal p.k (Nx.add p.q (Nx.sub_s len (Int64.of_int c.tq))))
    | Captured -> Nx.logical_and within captured
  in
  let loss ?lengths mask = function
    | [ q; k; v; slopes ] ->
        let score (p : Nx.host Nx.pair) s =
          Nx.(Infix.(s - (cast acc (p.q - p.k) * slopes)))
        in
        let o, lse =
          Nx.attention ?lengths ?mask ~score ~acc ~kv:(Nx.Dense { k; v }) q
        in
        let finite x = Nx.where (Nx.isfinite x) x (Nx.zeros_like x) in
        Nx.add (Nx.sum (Nx.mul w o)) (Nx.sum (Nx.mul w2 (finite lse)))
    | _ -> invalid_arg "loss"
  in
  let cut = loss ~lengths mask and whole = loss (Some hiding) in
  let close msg a b =
    let a = Nx.to_array a and b = Nx.to_array b in
    Array.iteri
      (fun i x ->
        at_most ~msg float_exact
          ~than:(tol *. (1. +. Float.abs x))
          (Float.abs (x -. b.(i))))
      a
  in
  let xs = [ q; k; v; slopes ] in
  List.iter2 (close "gradient")
    (Rune.grad T.(list tensor) whole xs)
    (Rune.grad T.(list tensor) cut xs);
  let ts = List.mapi (fun i x -> draw acc c (20. +. Float.of_int i) (Nx.shape x)) xs in
  close "tangent"
    (snd (Rune.jvp T.(list tensor) T.tensor whole xs ts))
    (snd (Rune.jvp T.(list tensor) T.tensor cut xs ts))

(* Rows of no key, of some and of every key, under each mask, rows
   whose groups interleave, in runs of one row and of several within a
   larger group, and rows of one group below every key. *)
let length_examples =
  List.map
    (fun (mask, lengths) ->
      {
        hk = 1;
        groups = 2;
        tq = 3;
        tk = 9;
        dk = 2;
        dv = 2;
        page = None;
        table = [| 0; 0 |];
        lengths = Some lengths;
        mask;
        alibi = true;
        seed = 0.5;
      })
    [
      (Causal, [| 0; 9 |]);
      (Captured, [| 4; 10 |]);
      (Unmasked, [| 1; 0 |]);
      (Causal, [| 1; 9; 2; 9; 1; 9; 5; 9 |]);
      (Causal, [| 1; 1; 9; 1; 1; 9; 9; 2 |]);
      (Causal, [| 3; 4 |]);
    ]

(* Ten lead indices' lengths in [[0, tk + 1]], over dense keys of [tk]
   from 3 to 12. In one case each of [arrangements]' covers holds with a
   chance above 0.55 at each [tk] (measured over 10^5 draws), so the cases
   of a run all miss one with a chance below 10^-10. *)
let length_case =
  let open Gen in
  with_pp pp_case
    (let* c = case in
     let* tk = int_range 3 12 in
     let+ lengths = array ~size:(constant 10) (int_range 0 (tk + 1)) in
     { c with page = None; tk; lengths = Some lengths })

(* The most elements a result of an operation of [f ()] holds, a view's
   excepted: the largest array [f] stores. *)
let stored f =
  let most = ref 0 in
  let rule : type r. Nx.Prim.interpretation -> by:string -> r Nx.Prim.t -> r =
   fun _ ~by op ->
    let r = Nx.Prim.eval ~by op in
    (match op with
    | Move _ -> ()
    | _ ->
        Nx.Prim.iter_results
          (fun y -> most := max !most (numel (Nx.shape y)))
          op r);
    r
  in
  Nx.Prim.interpret ~name:"stored" Extent rule (fun _ -> ignore (f ()));
  !most

(* The eager derivative computes over each lead index's keys before its
   length: of 64 keys, lengths 3 and 0 leave it no array of the whole pair
   loop, in reverse mode and in forward mode. *)
let test_padded_work () =
  let lead = [| 2 |] and hq = 2 and tq = 4 and tk = 64 in
  let f s seed =
    Nx.create Nx.float64 s
      (Array.init (numel s) (fun k -> Float.sin ((Float.of_int k *. 1.3) +. seed)))
  in
  let lengths = Nx.create Nx.int64 lead [| 3L; 0L |] in
  let loss = function
    | [ q; k; v ] ->
        let o, _ =
          Nx.attention ~lengths ~mask:Nx.causal ~acc:Nx.float64
            ~kv:(Nx.Dense { k; v }) q
        in
        Nx.sum o
    | _ -> invalid_arg "loss"
  in
  let xs = [ f [| 2; hq; tq; 3 |] 1.; f [| 2; 1; tk; 3 |] 2.; f [| 2; 1; tk; 2 |] 3. ] in
  let pairs = 2 * hq * tq * tk in
  less ~msg:"reverse mode" Testable.int ~than:pairs
    (stored (fun () -> Rune.grad T.(list tensor) loss xs));
  let ts = List.map Nx.ones_like xs in
  less ~msg:"forward mode" Testable.int ~than:pairs
    (stored (fun () -> Rune.jvp T.(list tensor) T.tensor loss xs ts))

module One = (val Nx.devices [ Nx_support.memory 0 ])

(* Under lengths of no key the eager gradient lies where its operands lie,
   on a set of its own, and is zero. *)
let test_no_key_placement () =
  let f s seed =
    Nx.place One.on
      (Nx.create Nx.float64 s
         (Array.init (numel s) (fun k ->
              Float.sin ((Float.of_int k *. 1.3) +. seed))))
  in
  let lengths = Nx.place One.on (Nx.create Nx.int64 [| 2 |] [| 0L; 0L |]) in
  let loss = function
    | [ q; k; v ] ->
        let o, _ =
          Nx.attention ~lengths ~acc:Nx.float64 ~kv:(Nx.Dense { k; v }) q
        in
        Nx.sum o
    | _ -> invalid_arg "loss"
  in
  let xs = [ f [| 2; 2; 3; 3 |] 1.; f [| 2; 1; 4; 3 |] 2.; f [| 2; 1; 4; 2 |] 3. ] in
  List.iter2
    (fun x g ->
      equal ~msg:"placement" (Nx_support.placement ()) (Nx.placement x)
        (Nx.placement g);
      equal ~msg:"gradient" (Testable.array float_exact)
        (Array.make (Nx.numel x) 0.) (Nx.to_array g))
    xs
    (Rune.grad T.(list tensor) loss xs)

(* A length acts as its clamp into [[0, tk]], int64's extremes included:
   the eager gradient under lengths [max_int] and [min_int] has the bits of
   the gradient under [tk] and [0]. *)
let test_length_extremes () =
  let tk = 4 in
  let f s seed =
    Nx.create Nx.float64 s
      (Array.init (numel s) (fun k -> Float.sin ((Float.of_int k *. 1.3) +. seed)))
  in
  let loss lengths = function
    | [ q; k; v ] ->
        let o, _ =
          Nx.attention ~lengths ~mask:Nx.causal ~acc:Nx.float64
            ~kv:(Nx.Dense { k; v }) q
        in
        Nx.sum o
    | _ -> invalid_arg "loss"
  in
  let xs = [ f [| 2; 2; 3; 3 |] 1.; f [| 2; 1; tk; 3 |] 2.; f [| 2; 1; tk; 2 |] 3. ] in
  let grad lens =
    Rune.grad T.(list tensor) (loss (Nx.create Nx.int64 [| 2 |] lens)) xs
  in
  List.iter2
    (fun a b -> equal ~msg:"gradient" (Testable.array Testable.int64) (bits a) (bits b))
    (grad [| Int64.of_int tk; 0L |])
    (grad [| Int64.max_int; Int64.min_int |])

(* Values and keys past a row's length are never read, by the derivative
   either: NaN there leaves every cotangent and tangent finite, eagerly and
   compiled, and the keys' and values' past each length zero. *)
let test_past_length () =
  Rune_support.Compiled.here ();
  let lead = [| 2 |] and tk = 8 and lens = [| 3; 5 |] in
  let f s seed =
    Nx.create Nx.float64 s
      (Array.init (numel s) (fun k -> Float.sin ((Float.of_int k *. 1.7) +. seed)))
  in
  let poisoned s seed =
    let x = Nx.to_array (f s seed) in
    let per = numel (Array.sub s 1 3) and d = s.(3) in
    Array.iteri
      (fun i _ -> if i / per |> fun b -> i mod per / d mod tk >= lens.(b) then x.(i) <- Float.nan)
      x;
    Nx.create Nx.float64 s x
  in
  let lengths = Nx.create Nx.int64 lead (Array.map Int64.of_int lens) in
  let loss = function
    | [ q; k; v ] ->
        let o, lse =
          Nx.attention ~lengths ~mask:Nx.causal ~acc:Nx.float64
            ~kv:(Nx.Dense { k; v }) q
        in
        let finite x = Nx.where (Nx.isfinite x) x (Nx.zeros_like x) in
        Nx.add (Nx.sum o) (Nx.sum (finite lse))
    | _ -> invalid_arg "loss"
  in
  let xs =
    [ f [| 2; 2; 4; 3 |] 1.; poisoned [| 2; 1; tk; 3 |] 2.; poisoned [| 2; 1; tk; 2 |] 3. ]
  in
  let finite msg x =
    Array.iter
      (fun v -> equal ~msg Testable.bool true (Float.is_finite v))
      (Nx.to_array x)
  in
  List.iter (finite "eager gradient") (Rune.grad T.(list tensor) loss xs);
  List.iter (finite "compiled gradient")
    (Rune.jit T.(list tensor @-> returns (list tensor)) (Rune.grad T.(list tensor) loss) xs);
  let ts = List.map (fun x -> Nx.ones_like x) xs in
  finite "tangent" (snd (Rune.jvp T.(list tensor) T.tensor loss xs ts))

(* A tile of the derivative's kernels holds a whole number of steps of every
   core's [N] and [K], as either operand order names them: the kernels loop
   [tile / k] times, so a tile shorter than a step would run no product and
   leave each cotangent zero. CDNA 4's K of 32 is the case no machine here
   runs. *)
let tile_steps () =
  let open Tolk in
  let cores =
    List.concat
      Tc.[ cuda_sm75; cuda_sm80; cuda_sm89; amd_rdna3; amd_rdna4; amd_cdna3; amd_cdna4; metal ]
  in
  let check tc =
    let n, _, k = Tc.dims tc in
    List.iter
      (fun wide ->
        let t = Rune.attention_tile tc ~wide in
        at_least ~msg:"keys of a tile" Testable.int ~than:(if wide then 16 else 32) t;
        equal ~msg:"whole steps of N" Testable.int 0 (t mod n);
        equal ~msg:"whole steps of K" Testable.int 0 (t mod k))
      [ false; true ]
  in
  List.iter
    (fun tc ->
      check tc;
      check (Tc.transpose tc))
    cores

(* Heads of every size, in float32 and bfloat16. *)
let heads = Attentions.heads Nx.Dtype.[ Any Nx.float32; Any Nx.bfloat16 ]

let head_id (c : Attentions.t) =
  let (Nx.Dtype.Any dt) = c.dtype in
  Printf.sprintf "%s dk %d dv %d" (Nx.Dtype.name dt) c.dk c.dv

let wide = [ D (Nx.float32, Nx.float32); D (Nx.float64, Nx.float64) ]

(* Each compiled law's examples, its forms in each pair of dtypes, compile
   together, and its drawn cases on their forms. A law whose examples make a few
   dozen kernels in all compiles them as one program, which keeps every worker
   compiling to its end: on fudge 36 examples of 57 kernels took 2.3 s as one
   program and 2.8 s as three. *)
let pair_forms = typed pairs forms

(* [Batch.law] of two calls of a case, each call's examples in programs of their
   own: the two may share intermediates, which one program would compute once. A
   case not of [examples] compiles each call alone. *)
let apart ~per call call' examples check =
  let t = Batch.table ~per call examples in
  let t' = Batch.table ~per call' examples in
  let alone call c () = List.hd (Batch.run [ call c ]) in
  fun c ->
    let r = Option.value (List.assq_opt c t) ~default:(alone call c) in
    let r' = Option.value (List.assq_opt c t') ~default:(alone call' c) in
    check c (fun () -> (r (), r' ()))

let law_jit =
  let per = List.length pair_forms in
  apart ~per (law_jit_call O) (law_jit_call Lse) pair_forms law_jit

let cut_examples = typed pairs cut_forms
let law_cut = apart ~per:9 law_cut_call law_cut_grad_call cut_examples law_cut
let grad_examples = typed wide forms
let law_grad = Batch.law ~per:8 law_grad_call grad_examples law_grad

(* Every target's examples: those of a GPU kernel's shapes are
   test_jit_gpu's. *)
let float64_examples =
  let dts =
    Nx.Dtype.[ Any Nx.float16; Any Nx.bfloat16; Any Nx.float32; Any Nx.float64 ]
  in
  Attentions.examples dts @ Attentions.edge_examples dts

let float64_law =
  Batch.law
    ~per:(List.length float64_examples)
    (Attentions.law_call (module Nx.Host))
    float64_examples
    (Attentions.law_check ?composed:None ?wide:None)

let composed_law =
  Batch.law ~per:16
    (Attentions.law_call (module Nx.Host))
    heads
    (Attentions.law_check ~composed:true ?wide:None)

let zero_dtypes = Nx.Dtype.[ Any Nx.float32; Any Nx.bfloat16; Any Nx.float16 ]

let zero_sums =
  Batch.law ~per:3
    (Attentions.zero_sums_call (module Nx.Host))
    zero_dtypes
    (fun _ bits -> Attentions.zero_sums_check (bits ()))

(* float32 alone: the host's kernel computes every dtype in its accumulator
   alike. *)
let paged_law =
  let host paged = Paged.law_call ~paged (module Nx.Host) in
  apart
    ~per:(List.length Paged.host_examples)
    (host true) (host false) Paged.host_examples Paged.law_check

(* Each dtype's gradient against central differences, under lengths that hold a
   row of no key or none. *)
let central =
  [
    (D (Nx.float32, Nx.float32), [| 6L; 7L |]);
    (D (Nx.float64, Nx.float64), [| 6L; 7L |]);
    (D (Nx.float32, Nx.float32), [| 0L; 3L |]);
    (D (Nx.float64, Nx.float64), [| 0L; 3L |]);
  ]

let central_id (d, lens) =
  if lens.(0) = 0L then dtypes_name d ^ ", a row of no key" else dtypes_name d

let central_grads ((D (dt, _) as d), lens) =
  if is_float64 d then grads dt ~eps:2e-6 ~tol:2e-6 lens ()
  else grads dt ~eps:1e-2 ~tol:2e-3 lens ()

let () =
  let case = Gen.with_pp pp_case case in
  exit
    (run "rune attention"
       [
         group "past a row's length"
           [
             test "the derivative reads no key or value there" test_past_length;
             test "the eager derivative stores no pair there" test_padded_work;
             test "lengths of no key keep the gradient where it lies"
               test_no_key_placement;
             test "a length acts as its clamp" test_length_extremes;
             prop
               ~examples:(typed wide length_examples)
               ~count:60 "lengths hide what a mask of their pairs hides"
               (any_of wide length_case) law_lengths;
           ];
         group "jvp by central differences"
           [
             prop ~examples:pair_forms ~count:180 "of a weighted sum of o"
               (any_of pairs case) law_jvp;
           ];
         group "gradients by central differences"
           [
             cases "of the query, keys and values" ~id:central_id central
               central_grads;
             test "of a float capture" capture_grads;
             test "through a page table" paged_grads;
           ];
         group "vmap is each lane, bit for bit"
           [
             prop ~examples:pair_forms ~count:180
               "over keys and values the lanes share" (any_of pairs case)
               law_vmap;
             prop ~examples:pair_forms ~count:180 "over lanes' own pools"
               (any_of pairs case) law_vmap_pools;
           ];
         group "results"
           [
             (* The host takes no tensor core: its tiles hold 16 keys at any
                number of value columns. *)
             test
               "of values of no column are the full call's lse and gradient \
                bits" (fun () ->
                 Rune_support.Compiled.here ();
                 ignore
                   (Attentions.lse_only
                      (module Nx.Host)
                      ~bits:true ~rows:2 ~dims:110 (Nx.Dtype.Any Nx.float32)));
             test "are the storage the kernels wrote" (fun () ->
                 Rune_support.Compiled.here ();
                 Attentions.uncopied
                   (module Nx.Host)
                   ~grad:false (Nx.Dtype.Any Nx.float32));
             test "returned twice are apart" (fun () ->
                 Rune_support.Compiled.here ();
                 Attentions.apart (module Nx.Host) (Nx.Dtype.Any Nx.float32));
             test "stacked are written by their kernels" (fun () ->
                 Rune_support.Compiled.here ();
                 Attentions.stacked (module Nx.Host) (Nx.Dtype.Any Nx.float32));
             test "returned twice into two donated arguments are apart"
               (fun () ->
                 Rune_support.Compiled.here ();
                 Attentions.donated_twice
                   (module Nx.Host)
                   (Nx.Dtype.Any Nx.float32));
             test "are written into a donated argument they do not read"
               (fun () ->
                 Rune_support.Compiled.here ();
                 Attentions.donated (module Nx.Host) (Nx.Dtype.Any Nx.float32));
           ];
         group "jit within the bounds of eager"
           [
             prop ~examples:pair_forms ~count:36 "over dense and paged keys"
               (Gen.with_pp pp_typed (redrawn pair_forms))
               law_jit;
           ];
         group "jit within nx's bound of a float64 reference"
           [
             prop ~examples:float64_examples ~count:6 "every dtype"
               (Gen.with_pp Attentions.pp (Attentions.redrawn float64_examples))
               float64_law;
             cases "a sum of zeros is +0 at rows of one run of keys"
               ~id:(fun (Nx.Dtype.Any dt) -> Nx.Dtype.name dt)
               zero_dtypes zero_sums;
             cases "heads of 36 to 1024 dimensions, against the composition"
               ~id:head_id heads composed_law;
             prop ~examples:Paged.host_examples ~count:4
               "paged keys and values give dense ones' bits"
               (Gen.with_pp Paged.pp (Paged.redrawn Paged.host_examples))
               paged_law;
           ];
         group "jit over keys cut across two devices"
           [
             prop ~examples:cut_examples ~count:36
               "is within four times one device's bounds"
               (Gen.with_pp pp_typed (redrawn cut_examples))
               law_cut;
           ];
         group "the gradient compiled and in forward mode"
           [
             prop ~examples:grad_examples ~count:24 "is eager reverse mode's"
               (Gen.with_pp pp_typed (redrawn grad_examples))
               law_grad;
           ];
         group "the derivative's tiles"
           [ test "hold whole steps of each tensor core" tile_steps ];
       ])
