(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t =
  | Blocking of (Table.t -> Table.t -> (Table.t, Eval.failure) result)
  | Streaming of {
      batch : Table.t -> Table.t -> (Table.t, Eval.failure) result;
      last : Table.t -> Table.t -> int -> (Table.t, Eval.failure) result;
    }

let ( let* ) = Result.bind
let arange n = Nx.arange Nx.int64 0 n 1
let none n = Nx.full Nx.int64 [| n |] (-1L)
let ints n k = Nx.full Nx.int64 [| n |] (Int64.of_int k)

(* [exclusive c] is the sum of the entries of [c] before each one. *)
let exclusive c = Nx.sub (Nx.cumsum c) c

(* [assemble q kind l li r ri keys] is [q]'s rows: each column is the one [keys]
   give it, or else [l]'s taken at [li], or else [r]'s taken at [ri]. The side a
   [kind] of join pads, the left of a full join and the right of a left or full
   one, is null where an index is outside its rows, as [-1] is, and those rows
   are found once for all its columns. *)
let assemble q (kind : Join.kind) l li r ri keys =
  let found pads idx t =
    if not pads then None
    else
      Some
        (lazy
          (let inside =
             Nx.logical_and
               (Nx.greater_equal_s idx 0L)
               (Nx.less_s idx (Int64.of_int (Table.rows t)))
           in
           Column.mask (Nx.cast Nx.bit inside)))
  in
  let found_l = found (kind = Full) li l
  and found_r = found (kind = Left || kind = Full) ri r in
  let side found idx c =
    let g = Column.gather idx c in
    match found with
    | Some found when Option.is_none (Column.validity g) ->
        Column.restrict (Lazy.force found) g
    | _ -> g
  in
  let column n =
    match List.assoc_opt n keys with
    | Some c -> c
    | None -> (
        match Schema.find (Table.schema l) n with
        | Some _ -> side found_l li (Table.column l n)
        | None -> side found_r ri (Table.column r n))
  in
  let s = Query.schema q in
  Table.batch s ~rows:(Nx.dim 0 li)
    (Array.of_list (List.map column (Schema.names s)))

(* Assertions *)

let phrase : Join.count -> string = function
  | Any -> "any"
  | At_most_one -> "at most one"
  | One -> "one"
  | At_least_one -> "at least one"

(* [check side count keys counts] fails at the first row of [side] whose number
   of matches [counts] [count] does not allow, naming the row's [keys]. *)
let check side (count : Join.count) keys counts =
  let bad =
    match count with
    | Any -> None
    | At_most_one -> Some (fun c -> Nx.greater_s c 1L)
    | One -> Some (fun c -> Nx.not_equal_s c 1L)
    | At_least_one -> Some (fun c -> Nx.less_s c 1L)
  in
  match bad with
  | None -> Ok ()
  | Some bad ->
      let c = Lazy.force counts in
      let at = Nx.positions (bad c) in
      if Nx.numel at = 0 then Ok ()
      else
        let row = Int64.to_int (Nx.item [ 0 ] at) in
        let key ppf (n, k) =
          Format.fprintf ppf "%a is %t" Type.pp_quoted n (Form.pp_cell k row)
        in
        let whose ppf = function
          | [] -> ()
          | ks ->
              Format.fprintf ppf " whose %a"
                (Format.pp_print_list
                   ~pp_sep:(fun ppf () -> Format.pp_print_string ppf " and ")
                   key)
                ks
        in
        let why =
          Format.asprintf "the %s row%a matches %Ld rows, not %s" side whose
            keys (Nx.item [ row ] c) (phrase count)
        in
        Error { Eval.row; cause = Data (Error.v (why ^ ".")) }

let named t ns = List.map (fun n -> (n, Table.column t n)) ns

(* Equality joins

   The keys of both sides are coded at once: {!Key.groups} numbers the groups of
   equal keys, and the right rows of each group, in order, are one row of
   [runs]. A left row's matches are its group's row. *)

(* [left_pairs counts matched] pairs each left row with its matches, or with
   [-1] if it has none. *)
let left_pairs counts matched =
  let one = Nx.maximum_s counts 1L in
  let li = Nx.positions one and inner = Nx.positions counts in
  let rank =
    Nx.sub (arange (Nx.dim 0 inner)) (Nx.take ~indices:inner (exclusive counts))
  in
  let at = Nx.add (Nx.take ~indices:inner (exclusive one)) rank in
  let ri =
    Nx.scatter ~axis:0 ~indices:at ~values:matched (none (Nx.dim 0 li))
  in
  (li, ri)

let equality q kind (each_left, each_right) left right eqs =
  let widen (ln, rn) =
    let tl = Option.get (Schema.find (Query.schema left) ln)
    and tr = Option.get (Schema.find (Query.schema right) rn) in
    let t = Option.get (Join.common_type tl tr) in
    (ln, rn, Eval.widen tl t, Eval.widen tr t)
  in
  let atoms = List.map widen eqs in
  fun l r ->
    let n = Table.rows l and m = Table.rows r in
    let keys =
      List.map
        (fun (ln, rn, wl, wr) ->
          (ln, Column.concat [ wl (Table.column l ln); wr (Table.column r rn) ]))
        atoms
    in
    let g = Key.groups (List.map snd keys) in
    let ids lo hi = Nx.slice [ Nx.R (lo, hi) ] g.ids in
    let runs =
      Nx_ragged.of_ids ~segments:(Nx.dim 0 g.first) (ids n (n + m)) (arange m)
    in
    let hits = Nx_ragged.take ~indices:(ids 0 n) runs in
    let counts = Nx_ragged.lengths hits and matched = Nx_ragged.values hits in
    let rcounts =
      lazy (Nx.reduce_segments `Add ~segments:m matched (Nx.ones_like matched))
    in
    let* () =
      check "left" each_left (named l (List.map fst eqs)) (lazy counts)
    in
    let* () = check "right" each_right (named r (List.map snd eqs)) rcounts in
    let pairs li ri = Ok (assemble q kind l li r ri []) in
    match (kind : Join.kind) with
    | Inner -> pairs (Nx.positions counts) matched
    | Semi -> pairs (Nx.positions (Nx.greater_s counts 0L)) (arange 0)
    | Anti -> pairs (Nx.positions (Nx.equal_s counts 0L)) (arange 0)
    | Left ->
        let li, ri = left_pairs counts matched in
        pairs li ri
    | Full ->
        (* A right row without a match follows, its keys in the left's. *)
        let li, ri = left_pairs counts matched in
        let u = Nx.positions (Nx.equal_s (Lazy.force rcounts) 0L) in
        let at = Nx.concatenate ~axis:0 [ li; Nx.add_s u (Int64.of_int n) ] in
        let coalesced =
          List.map (fun (ln, k) -> (ln, Column.gather at k)) keys
        in
        let li = Nx.concatenate ~axis:0 [ li; none (Nx.dim 0 u) ]
        and ri = Nx.concatenate ~axis:0 [ ri; u ] in
        Ok (assemble q kind l li r ri coalesced)

(* Position and all joins *)

let position q kind (each_left, each_right) l r =
  let n = Table.rows l and m = Table.rows r in
  let below k rows = lazy (Nx.cast Nx.int64 (Nx.less_s (arange rows) k)) in
  let* () = check "left" each_left [] (below (Int64.of_int m) n) in
  let* () = check "right" each_right [] (below (Int64.of_int n) m) in
  let idx =
    match (kind : Join.kind) with
    | Inner | Semi -> arange (Int.min n m)
    | Left -> arange n
    | Full -> arange (Int.max n m)
    | Anti -> Nx.arange Nx.int64 (Int.min n m) n 1
  in
  Ok (assemble q kind l idx r idx [])

let cross q kind (each_left, each_right) =
  let batch r b =
    let n = Table.rows b and m = Table.rows r in
    let* () = check "left" each_left [] (lazy (ints n m)) in
    let kept keep = if keep then arange n else arange 0 in
    let li, ri =
      match (kind : Join.kind) with
      | (Inner | Left | Full) when m > 0 ->
          let p = arange (n * m) in
          (Nx.div_s p (Int64.of_int m), Nx.mod_s p (Int64.of_int m))
      | Inner -> (arange 0, arange 0)
      | Left | Full -> (arange n, none n)
      | Semi -> (kept (m > 0), arange 0)
      | Anti -> (kept (m = 0), arange 0)
    in
    Ok (assemble q kind b li r ri [])
  in
  let last r e n =
    let m = Table.rows r in
    let* () = check "right" each_right [] (lazy (ints m n)) in
    if kind = Full && n = 0 then Ok (assemble q kind e (none m) r (arange m) [])
    else Ok (assemble q kind e (arange 0) r (arange 0) [])
  in
  Streaming { batch; last }

let compile q =
  match Query.node q with
  | Join { kind; each_left; each_right; on; left; right } -> (
      let counts = (each_left, each_right) in
      let eq = function Join.Eq (l, r) -> Some (l, r) | Position -> None in
      match (on :> Join.atom list) with
      | [] -> cross q kind counts
      | [ Position ] -> Blocking (position q kind counts)
      | atoms ->
          let eqs = List.filter_map eq atoms in
          Blocking (equality q kind counts left right eqs))
  | _ -> invalid_arg "Join_run.compile: not a join"
