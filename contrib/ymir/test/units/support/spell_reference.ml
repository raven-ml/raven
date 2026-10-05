(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A reference for [Vocabulary.spell], written from its statement over plain
   maps from coordinates to rational exponents. It shares no code with the
   library: it tries every candidate the statement names and keeps the
   cheapest. It raises [Overflow] when its own int arithmetic overflows; a
   property that compares against it discards such a case. *)

open Ymir_units

exception Overflow

(* Rationals *)

let rec gcd a b = if b = 0 then abs a else gcd b (a mod b)

let mul a b =
  let p = a * b in
  if a <> 0 && (p / a <> b || p = min_int) then raise Overflow else p

let add a b =
  let s = a + b in
  if (a > 0 && b > 0 && s < 0) || (a < 0 && b < 0 && s >= 0) || s = min_int then
    raise Overflow
  else s

let q n d =
  if d = 0 then invalid_arg "Spell_reference.q";
  let g = gcd n d in
  let n = n / g and d = d / g in
  if d < 0 then (-n, -d) else (n, d)

let ( +/ ) (a, b) (c, d) = q (add (mul a d) (mul c b)) (mul b d)
let ( */ ) (a, b) (c, d) = q (mul a c) (mul b d)
let neg (a, b) = (-a, b)
let ( -/ ) x y = x +/ neg y
let inv (a, b) = q b a
let ( // ) x y = x */ inv y
let qabs (a, b) = (abs a, b)
let is_zero (a, _) = a = 0
let is_int (_, b) = b = 1
let qcompare x y = compare (fst (x -/ y)) 0

(* Units as maps *)

type key = Prime of int | Pi | Sym of string * string option | D25

module K = Map.Make (struct
  type t = key

  let compare = compare
end)

let of_unit u =
  List.fold_left
    (fun m (t, n, d) ->
      let k =
        match (t : Unit.term) with
        | Prime p -> Prime p
        | Pi -> Pi
        | Symbol { name; scope } -> Sym (name, scope)
      in
      K.add k (q n d) m)
    K.empty (Unit.terms u)

let get k m = Option.value ~default:(0, 1) (K.find_opt k m)

let mulu a b =
  K.union
    (fun _ x y ->
      let s = x +/ y in
      if is_zero s then None else Some s)
    a b

let pw a e = if is_zero e then K.empty else K.map (fun x -> x */ e) a
let divu a b = mulu a (pw b (-1, 1))
let syms u = K.filter (fun k _ -> match k with Sym _ -> true | _ -> false) u

(* What a power of ten cannot give: π, primes other than 2 and 5, and the
   excess of 2's exponent over 5's. *)
let numcoords u =
  let c =
    K.filter
      (fun k _ ->
        match k with Pi -> true | Prime p -> p <> 2 && p <> 5 | _ -> false)
      u
  in
  let d = get (Prime 2) u -/ get (Prime 5) u in
  if is_zero d then c else K.add D25 d c

(* [ten u] is [Some k] when [u]'s number is 10^k, symbols aside. *)
let ten u = if K.is_empty (numcoords u) then Some (get (Prime 5) u) else None
let coords u = K.union (fun _ a _ -> Some a) (syms u) (numcoords u)
let equal a b = K.equal (fun x y -> x = y) a b

(* Entries *)

type kind = Number | Single | Head | Dimensionless

type entry = {
  i : int;
  name : string;
  pre : bool;
  u : (int * int) K.t;
  kind : kind;
  k : int * int;
}

let classify es =
  List.mapi
    (fun i (name, p, unit) ->
      let u = of_unit unit in
      let kind =
        if not (K.is_empty (numcoords u)) then Number
        else
          match K.cardinal (syms u) with
          | 0 -> Dimensionless
          | 1 -> Single
          | _ -> Head
      in
      let k = Option.value ~default:(0, 1) (ten u) in
      { i; name; pre = p = Vocabulary.Prefixable; u; kind; k })
    es

let prefixes =
  [
    (-30, [ "q" ]);
    (-27, [ "r" ]);
    (-24, [ "y" ]);
    (-21, [ "z" ]);
    (-18, [ "a" ]);
    (-15, [ "f" ]);
    (-12, [ "p" ]);
    (-9, [ "n" ]);
    (-6, [ "\xce\xbc"; "\xc2\xb5"; "u" ]);
    (-3, [ "m" ]);
    (-2, [ "c" ]);
    (-1, [ "d" ]);
    (1, [ "da" ]);
    (2, [ "h" ]);
    (3, [ "k" ]);
    (6, [ "M" ]);
    (9, [ "G" ]);
    (12, [ "T" ]);
    (15, [ "P" ]);
    (18, [ "E" ]);
    (21, [ "Z" ]);
    (24, [ "Y" ]);
    (27, [ "R" ]);
    (30, [ "Q" ]);
  ]

let collides cls p name =
  let spellings = List.assoc p prefixes in
  List.exists (fun s -> List.exists (fun c -> c.name = s ^ name) cls) spellings

(* [prefix_ok cls c p] is [true] iff [c] may carry the prefix of 10^p. *)
let prefix_ok cls c p =
  List.mem_assoc p prefixes && c.pre && not (collides cls p c.name)

let word p c (n, d) : Vocabulary.word =
  { prefix = p; symbol = c.name; num = n; den = d }

(* Lexicographic order on lists of comparisons. *)
let rec lex = function [] -> 0 | 0 :: cs -> lex cs | c :: _ -> c
let b2i b = if b then 1 else 0

let argmin cmp = function
  | [] -> None
  | x :: xs ->
      Some (List.fold_left (fun b y -> if cmp y b < 0 then y else b) x xs)

(* A word's exponent is an integer when all of [u]'s exponents are. *)
let allowed u e = is_int e || not (K.for_all (fun _ x -> is_int x) u)

(* Step 2: one word *)

let one_word cls u =
  let cand c =
    let co = coords c.u in
    match K.min_binding_opt co with
    | None -> None
    | Some (key0, v0) -> (
        let e = get key0 (coords u) // v0 in
        if is_zero e || not (allowed u e) then None
        else if not (equal (coords u) (pw co e)) then None
        else
          match ten (divu u (pw c.u e)) with
          | None -> None
          | Some t ->
              let p =
                if is_zero t then Some 0
                else
                  let pf = t // e in
                  if is_int pf && prefix_ok cls c (fst pf) then Some (fst pf)
                  else None
              in
              Option.map
                (fun p ->
                  let exact =
                    K.for_all
                      (fun k _ -> match k with Sym _ -> true | _ -> false)
                      c.u
                  in
                  let rank =
                    [ b2i (p <> 0); b2i (not exact); b2i (p <> 0 && fst e < 0) ]
                  in
                  (rank, qabs e, c.i, word p c e))
                p)
  in
  let cmp (r, a, i, _) (r', a', i', _) =
    lex [ compare r r'; qcompare a a'; compare i i' ]
  in
  argmin cmp (List.filter_map cand cls) |> Option.map (fun (_, _, _, w) -> w)

(* Step 3: words *)

let solve_numbers cls u =
  let nu = numcoords u in
  if K.is_empty nu then [ [] ]
  else
    let nums = List.filter (fun c -> c.kind = Number) cls in
    let single c =
      let nc = numcoords c.u in
      let key, v = K.min_binding nc in
      let e = get key nu // v in
      if (not (is_zero e)) && equal (pw nc e) nu then Some [ (c, e) ] else None
    in
    match List.filter_map single nums with
    | _ :: _ as s -> s
    | [] ->
        let rec pairs = function
          | [] -> []
          | x :: xs -> List.map (fun y -> (x, y)) xs @ pairs xs
        in
        let solve (a, b) =
          let na = numcoords a.u and nb = numcoords b.u in
          let keys =
            List.concat_map
              (fun m -> List.map fst (K.bindings m))
              [ na; nb; nu ]
            |> List.sort_uniq compare
          in
          List.find_map
            (fun (k1, k2) ->
              let a1 = get k1 na and b1 = get k1 nb and c1 = get k1 nu in
              let a2 = get k2 na and b2 = get k2 nb and c2 = get k2 nu in
              let det = (a1 */ b2) -/ (a2 */ b1) in
              if is_zero det then None
              else
                let qa = ((c1 */ b2) -/ (c2 */ b1)) // det in
                let qb = ((a1 */ c2) -/ (a2 */ c1)) // det in
                if
                  (not (is_zero qa))
                  && (not (is_zero qb))
                  && equal (mulu (pw na qa) (pw nb qb)) nu
                then Some [ (a, qa); (b, qb) ]
                else None)
            (pairs keys)
        in
        List.filter_map solve (pairs nums)

let symbol_word cls s x =
  let cands =
    List.filter_map
      (fun c ->
        if c.kind = Single && K.mem s c.u then
          let e = x // get s c.u in
          Some
            ( [ b2i (not (is_zero c.k)); b2i (not (is_int e)) ],
              qabs e,
              c.i,
              (c, e) )
        else None)
      cls
  in
  let cmp (r, a, i, _) (r', a', i', _) =
    lex [ compare r r'; qcompare a a'; compare i i' ]
  in
  argmin cmp cands |> Option.map (fun (_, _, _, w) -> w)

type role = N | H | S

let compound cls u =
  let candidates nwords =
    let r = List.fold_left (fun r (c, e) -> divu r (pw c.u e)) u nwords in
    let heads =
      None
      :: List.concat_map
           (fun c ->
             if c.kind <> Head then []
             else
               List.filter_map
                 (fun (s, hs) ->
                   let x = get s r in
                   if is_zero x then None else Some (Some (c, x // hs)))
                 (K.bindings (syms c.u)))
           cls
    in
    List.filter_map
      (fun head ->
        let r2 =
          match head with None -> r | Some (h, e) -> divu r (pw h.u e)
        in
        let sws =
          List.map (fun (s, x) -> symbol_word cls s x) (K.bindings (syms r2))
        in
        if List.mem None sws then None
        else
          let sws = List.filter_map Fun.id sws in
          let rest =
            List.fold_left (fun r (c, e) -> divu r (pw c.u e)) r2 sws
          in
          let exponents =
            List.map snd nwords
            @ (match head with None -> [] | Some (_, e) -> [ e ])
            @ List.map snd sws
          in
          match ten rest with
          | Some t
            when K.is_empty (syms rest)
                 && is_int t
                 && List.for_all (allowed u) exponents ->
              let t = fst t in
              let words =
                List.map (fun (c, e) -> (c, e, N)) nwords
                @ (match head with None -> [] | Some (h, e) -> [ (h, e, H) ])
                @ List.map (fun (c, e) -> (c, e, S)) sws
              in
              let words =
                List.stable_sort
                  (fun (c, e, k) (c', e', k') ->
                    compare (fst e < 0, k = S, c.i) (fst e' < 0, k' = S, c'.i))
                  words
              in
              let out =
                Array.of_list (List.map (fun (c, e, _) -> word 0 c e) words)
              in
              let rec place j = function
                | [] -> false
                | (c, e, k) :: ws ->
                    let alts =
                      if k <> S then [ c ]
                      else
                        let s, a = K.min_binding (syms c.u) in
                        List.filter
                          (fun d ->
                            d.kind = Single
                            && K.cardinal (syms d.u) = 1
                            && get s (syms d.u) = a)
                          cls
                        |> List.stable_sort (fun d d' ->
                            compare
                              (not (is_zero d.k), d.i)
                              (not (is_zero d'.k), d'.i))
                    in
                    let try_alt d =
                      let pf = ((t, 1) // e) -/ d.k +/ c.k in
                      if not (is_int pf) then false
                      else
                        let p = fst pf in
                        if p <> 0 && not (prefix_ok cls d p) then false
                        else (
                          out.(j) <- word p d e;
                          true)
                    in
                    List.exists try_alt alts || place (j + 1) ws
              in
              let decade = if t = 0 || place 0 words then 0 else t in
              let out = Array.to_list out in
              let items = List.length out + b2i (decade <> 0) in
              let size =
                List.fold_left
                  (fun acc (w : Vocabulary.word) -> acc +/ (abs w.num, w.den))
                  (b2i (decade <> 0), 1)
                  out
              in
              let hi = match head with None -> -1 | Some (h, _) -> h.i in
              let cost =
                ( items,
                  b2i (Option.is_some head),
                  size,
                  b2i (decade <> 0),
                  hi,
                  List.map (fun (c, _) -> c.i) nwords )
              in
              Some (cost, ({ decade; words = out } : Vocabulary.spelling))
          | Some _ | None -> None)
      heads
  in
  let cmp ((n, h, s, d, hi, ns), _) ((n', h', s', d', hi', ns'), _) =
    lex
      [
        compare n n';
        compare h h';
        qcompare s s';
        compare d d';
        compare hi hi';
        compare ns ns';
      ]
  in
  List.concat_map candidates (solve_numbers cls u)
  |> argmin cmp |> Option.map snd

(* [spell entries u] is what [Vocabulary.spell (Vocabulary.v entries) u]
   must be. *)
let spell entries unit : Vocabulary.spelling option =
  let cls = classify entries in
  let u = of_unit unit in
  match List.find_opt (fun c -> equal c.u u) cls with
  | Some c -> Some { decade = 0; words = [ word 0 c (1, 1) ] }
  | None -> (
      match one_word cls u with
      | Some w -> Some { decade = 0; words = [ w ] }
      | None -> compound cls u)
