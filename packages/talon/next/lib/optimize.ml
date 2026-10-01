(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The optimizer runs four passes over the whole plan. [fold] folds the
   constants of every expression. [push] moves conjuncts and slices toward the
   sources, leaving above each source the filters that reach it. [prune] narrows
   each step to the columns read above it and gives each place that reads a
   source its own request. [share] makes equal steps one value. *)

type conjunct = (bool, Expr.row) Expr.t

let mem_id c cs = List.exists (Expr.same c) cs
let union ns ms = ns @ List.filter (fun m -> not (List.mem m ns)) ms
let subset ns ms = List.for_all (fun n -> List.mem n ms) ns

let sat_add a b =
  if (a > max_int - b) [@mutate off "equivalent: a + b is max_int there"] then
    max_int
  else a + b

(* [at_shape e] is [e] at any shape: shapes are erased in nodes. *)
let at_shape e = Expr.typed (Expr.typing e) (Expr.node e)

(* Conjuncts *)

(* [conjuncts p] is the operands of the bound predicate [p]'s [&&]s, in
   order. *)
let rec conjuncts (p : conjunct) =
  match Expr.node p with
  | Logic (And, a, b) -> conjuncts (at_shape a) @ conjuncts (at_shape b)
  | _ -> [ p ]

(* [filters cs q] is [q] filtered by the conjuncts [cs], in order. *)
let filters cs q =
  let rec conjoin c = function
    | [] -> c
    | d :: ds -> Expr.conj c (conjoin d ds)
  in
  match cs with
  | [] -> q
  | c :: cs -> Query.make (Filter { predicate = conjoin c cs; input = q })

(* [column e] is the column that the bound expression [e] reads, if it reads one
   unchanged. *)
let column e =
  match Expr.node e with
  | Handle (_, n) | Read (_, n) | Ext_handle (_, n) -> Some n
  | _ -> None

(* [literal e] is the typing and the value of the bound literal [e]. *)
let literal (type a s) (e : (a, s) Expr.t) : (a Expr.typing * a) option =
  match Expr.node e with
  | Lit (_, v) | Const v -> Some (Expr.typing e, v)
  | _ -> None

(* [value t v] is the value [v] of the typing [t] as a predicate holds it: as
   stored at its type, or at its storage type. *)
let value : type a. a Expr.typing -> a -> Source.Pred.value option =
 fun t v ->
  let stored ty v =
    Option.map (fun v -> Source.Pred.Value (ty, v)) (Type.value ty v)
  in
  match t with
  | Column ty -> stored ty v
  | Extension d -> stored d.storage (d.enc v)
  | Value -> None

(* [compared t0 t1 v] is [v], of the typing [t1], at the type in which a column
   of typing [t0] is compared with it. *)
let compared : type a.
    a Expr.typing -> a Expr.typing -> a -> Source.Pred.value option =
 fun t0 t1 v ->
  match (t0, t1) with
  | Column ty0, Column ty1 -> (
      match (Type.common [ ty0; ty1 ], Type.value ty1 v) with
      | Some ty, Some v -> value (Column ty) v
      | _ -> None)
  | _, Extension _ -> value t1 v
  | _ -> None

let mirror : Expr.compare -> Expr.compare = function
  | `Lt -> `Gt
  | `Le -> `Ge
  | `Gt -> `Lt
  | `Ge -> `Le
  | (`Eq | `Ne) as op -> op

(* [pred p] is the source predicate that the bound predicate [p] is, if it is
   one. *)
let rec pred : type s. (bool, s) Expr.t -> Source.Pred.t option =
 fun p ->
  let both f a b =
    match (pred a, pred b) with
    | Some a, Some b -> Some (f [ a; b ])
    | _ -> None
  in
  match Expr.node p with
  | Compare (o, a, b) -> (
      let atom c op col lit =
        match (Expr.typing col, literal lit) with
        | tc, Some (tl, v) ->
            Option.map (fun v -> Source.Pred.Cmp (c, op, v)) (compared tc tl v)
        | _, None -> None
      in
      match (column a, column b) with
      | Some c, None -> atom c o a b
      | None, Some c -> atom c (mirror o) b a
      | _ -> None)
  | Is_in (vs, a) -> (
      let t = Expr.typing a in
      match column a with
      | Some c ->
          let values = List.filter_map (value t) vs in
          if List.compare_lengths values vs = 0 then Some (In (c, values))
          else None
      | None -> None)
  | Is_null a -> Option.map (fun c -> Source.Pred.Null c) (column a)
  | Not a -> (
      match Expr.node a with
      | Is_null x -> Option.map (fun c -> Source.Pred.Valid c) (column x)
      | _ -> Option.map (fun p -> Source.Pred.Not p) (pred a))
  | Logic (And, a, b) -> both (fun ps -> Source.Pred.And ps) a b
  | Logic (Or, a, b) -> both (fun ps -> Source.Pred.Or ps) a b
  | _ -> None

(* Constants *)

let truth (p : conjunct) =
  match Expr.node p with
  | Lit (_, b) -> Some (Some b)
  | Null -> Some None
  | _ -> None

let fold_outputs outputs =
  List.map
    (fun (n, Expr.Packed e) -> (n, Expr.Packed (Expr.fold_constants e)))
    outputs

(* [fold q] is [q] with the constants of its expressions folded, a filter of
   [true] gone and one of [false] or null an empty slice. *)
let rec fold q =
  match Query.node q with
  | Filter { predicate; input } -> (
      let predicate = Expr.fold_constants predicate and input = fold input in
      match truth predicate with
      | Some (Some true) -> input
      | Some (Some false | None) ->
          Query.make (Slice { offset = 0; length = 0; input })
      | None -> Query.make (Filter { predicate; input }))
  | n ->
      let folded : Query.node =
        match n with
        | Select r -> Select { r with outputs = fold_outputs r.outputs }
        | Derive r -> Derive { r with outputs = fold_outputs r.outputs }
        | Aggregate r -> Aggregate { r with outputs = fold_outputs r.outputs }
        | n -> n
      in
      Query.make (Query.map_inputs fold folded)

(* Pushing conjuncts and slices *)

let row_local outputs =
  List.for_all (fun (_, Expr.Packed e) -> Expr.row_local e) outputs

(* [through ~derive outputs c] is [c] reading the input of the [select] or
   [derive] of [outputs] if each column it reads is one of the input's,
   unchanged. *)
let through ~derive outputs c =
  let input n =
    match List.assoc_opt n outputs with
    | Some (Expr.Packed e) -> column e
    | None -> if derive then Some n else None
  in
  let rec pairs = function
    | [] -> Some []
    | n :: ns -> (
        match (input n, pairs ns) with
        | Some m, Some ps -> Some ((n, m) :: ps)
        | _ -> None)
  in
  Option.map
    (fun ps -> Expr.rename (fun n -> List.assoc n ps) c)
    (pairs (Expr.reads c))

(* Pending steps are the filters and slices on their way down, the top one
   first. They keep their order: one passes a step only if every one below it
   does. A filter's conjuncts that can fail stay above the next filter's, which
   remove rows they would fail on, so filters merge only otherwise. *)
type pending = Sliced of int * int | Filtered of conjunct list

let restore q = function
  | Sliced (offset, length) -> Query.make (Slice { offset; length; input = q })
  | Filtered cs -> filters cs q

(* [above staying q] is [q] under the pending steps [staying], the bottom one
   first. *)
let above staying q = List.fold_left restore q staying

(* [under_slice (offset, length) pending] is [pending] over a slice: two slices
   from the start merge. *)
let under_slice (offset, length) pending =
  match List.rev pending with
  | Sliced (o, l) :: upper when o >= 0 && offset >= 0 ->
      List.rev (Sliced (sat_add offset o, max 0 (min l (length - o))) :: upper)
  | _ -> pending @ [ Sliced (offset, length) ]

(* [under_filter own pending] is [pending] over a filter of the conjuncts [own]:
   the conjuncts of a filter directly above merge with [own], but those that can
   fail. *)
let under_filter own pending =
  match List.rev pending with
  | Filtered cs :: upper -> (
      match List.partition Expr.can_fail cs with
      | [], passing -> List.rev (Filtered (own @ passing) :: upper)
      | failing, passing ->
          List.rev (Filtered (own @ passing) :: Filtered failing :: upper))
  | _ -> pending @ [ Filtered own ]

(* [settle ~slices route pending] is [(passing, staying)]: the pending steps,
   top first, that pass a step, each conjunct sent by [route] to an input of the
   step, and those, bottom first, that stay above it. Slices pass iff [slices],
   into every input. *)
let settle ~slices route pending =
  let rec up passing = function
    | [] -> (passing, [])
    | (Sliced _ as s) :: upper ->
        if slices then up ((fun _ -> Some s) :: passing) upper
        else (passing, s :: upper)
    | Filtered cs :: upper -> (
        let routed = List.map (fun c -> (c, route c)) cs in
        let into input =
          match
            List.filter_map
              (function _, Some (i, c) when i = input -> Some c | _ -> None)
              routed
          with
          | [] -> None
          | cs -> Some (Filtered cs)
        in
        match List.filter (fun (_, r) -> Option.is_none r) routed with
        | [] -> up (into :: passing) upper
        | stuck -> (into :: passing, Filtered (List.map fst stuck) :: upper))
  in
  let passing, staying = up [] (List.rev pending) in
  ((fun input -> List.filter_map (fun into -> into input) passing), staying)

(* [push pending q] is [q] under the [pending] filters and slices, each moved as
   far toward the sources as the rules let it. *)
let rec push pending q =
  let past ~slices route rebuild =
    let passing, staying = settle ~slices route pending in
    above staying (rebuild passing)
  in
  let only route c = Option.map (fun c -> (`Input, c)) (route c) in
  let past_one ~slices route rebuild input =
    past ~slices (only route) (fun passing ->
        rebuild (push (passing `Input) input))
  in
  match Query.node q with
  | Of_table _ | Of_source _ -> above (List.rev pending) q
  | Filter { predicate; input } ->
      let own = conjuncts predicate in
      if List.for_all Expr.row_local own then
        push (under_filter own pending) input
      else
        above (List.rev pending)
          (Query.make (Filter { predicate; input = push [] input }))
  | Slice { offset; length; input } ->
      push (under_slice (offset, length) pending) input
  | Sort { keys; input } ->
      past_one ~slices:false Option.some
        (fun input -> Query.make (Sort { keys; input }))
        input
  | Select { outputs; input } when row_local outputs ->
      past_one ~slices:true
        (through ~derive:false outputs)
        (fun input -> Query.make (Select { outputs; input }))
        input
  | Derive { outputs; input } when row_local outputs ->
      past_one ~slices:true
        (through ~derive:true outputs)
        (fun input -> Query.make (Derive { outputs; input }))
        input
  | Aggregate ({ by = _ :: _ as by; input; _ } as r) ->
      let keys c =
        let reads = Expr.reads c in
        subset reads by
        && List.for_all
             (fun n ->
               match Schema.find (Query.schema input) n with
               | Some (Type.Any t) -> not (Type.has_float t)
               | None -> false)
             reads
      in
      past_one ~slices:false
        (fun c -> if keys c then Some c else None)
        (fun input -> Query.make (Aggregate { r with input }))
        input
  | Unnest { columns; input } ->
      let passes c =
        (not (Expr.can_fail c))
        && not (List.exists (fun n -> List.mem n columns) (Expr.reads c))
      in
      past_one ~slices:false
        (fun c -> if passes c then Some c else None)
        (fun input -> Query.make (Unnest { columns; input }))
        input
  | Append { input; rest } ->
      let passing, staying =
        settle ~slices:false (fun c -> Some (`Input, c)) pending
      in
      let limit =
        match staying with
        | Sliced (o, l) :: _ when o >= 0 -> [ Sliced (0, sat_add o l) ]
        | _ -> []
      in
      let into q = push (limit @ passing `Input) q in
      above staying
        (Query.make (Append { input = into input; rest = into rest }))
  | Join ({ kind; each_left = Any; each_right = Any; on; left; right } as r)
    when kind <> Join.Full
         && not
              (List.exists
                 (function Join.Position -> true | _ -> false)
                 (on :> Join.atom list)) ->
      let lefts = Schema.names (Query.schema left)
      and rights = Schema.names (Query.schema right) in
      let ordering =
        List.exists
          (function Join.Closest _ | Nearest _ -> true | _ -> false)
          (on :> Join.atom list)
      in
      let route c =
        let reads = Expr.reads c in
        if Expr.can_fail c then None
        else if subset reads lefts then Some (`Left, c)
        else if
          kind = Join.Inner && (not ordering) && subset reads rights
          && not (List.exists (fun n -> List.mem n lefts) reads)
        then Some (`Right, c)
        else None
      in
      past ~slices:false route (fun passing ->
          Query.make
            (Join
               {
                 r with
                 left = push (passing `Left) left;
                 right = push (passing `Right) right;
               }))
  | n -> above (List.rev pending) (Query.make (Query.map_inputs (push []) n))

(* Sources *)

(* [chain q] is [Some (fs, leaf)] if [q] is row-local filters directly over a
   source [leaf], [fs] their conjuncts, the lowest filter's first. A filter that
   is not row-local ends the chain below it: handing the source a conjunct above
   it would change its frame. *)
let rec chain q =
  match Query.node q with
  | Of_source _ -> Some ([], q)
  | Filter { predicate; input } ->
      let cs = conjuncts predicate in
      if List.for_all Expr.row_local cs then
        Option.map (fun (fs, leaf) -> (fs @ [ cs ], leaf)) (chain input)
      else None
  | _ -> None

(* Columns *)

let keep columns q =
  let outputs =
    List.filter_map
      (fun (n, Type.Any ty) ->
        if List.mem n columns then Some (n, Expr.Packed (Expr.read ty n))
        else None)
      (Schema.columns (Query.schema q))
  in
  if List.compare_lengths outputs (Schema.columns (Query.schema q)) = 0 then q
  else Query.make (Select { outputs; input = q })

let reads_of outputs =
  List.fold_left
    (fun acc (_, Expr.Packed e) -> union acc (Expr.reads e))
    [] outputs

let is_identity outputs input =
  List.equal String.equal (List.map fst outputs)
    (Schema.names (Query.schema input))
  && List.for_all (fun (n, Expr.Packed e) -> column e = Some n) outputs

(* [unkept q] is the input of [q] if [q] keeps some of its columns unchanged,
   which a step that reads its input by name and passes no column through reads
   as well. *)
let unkept q =
  match Query.node q with
  | Select { outputs; input }
    when List.for_all (fun (n, Expr.Packed e) -> column e = Some n) outputs ->
      input
  | _ -> q

let cond_columns (on : Join.cond) =
  List.fold_left
    (fun (ls, rs) -> function
      | Join.Eq (l, r)
      | Compare (_, l, r)
      | Closest { left = l; right = r; _ }
      | Nearest { left = l; right = r; _ } ->
          (l :: ls, r :: rs)
      | Position -> (ls, rs))
    ([], [])
    (on :> Join.atom list)

(* [prune ~ordered need q] is [q] with each step narrowed to the columns [need]
   of its result and those its steps read, and each source given its request.
   Unless [ordered], the order of [q]'s columns is not observed, so a [derive]
   need not read a column that it replaces to keep its place. *)
let rec prune ~ordered need q =
  match chain q with
  | Some (fs, leaf) -> place need fs leaf
  | None -> (
      match Query.node q with
      | Of_table _ | Of_source _ -> q
      | Filter { predicate; input } ->
          let input =
            prune ~ordered (union need (Expr.reads predicate)) input
          in
          Query.make (Filter { predicate; input })
      | Sort { keys; input } ->
          let keys_read = List.map (fun (k : Order.t) -> k.name) keys in
          Query.make
            (Sort { keys; input = prune ~ordered (union need keys_read) input })
      | Slice { offset; length; input } ->
          limit offset length (prune ~ordered need input)
      | Select { outputs; input } ->
          let outputs = List.filter (fun (n, _) -> List.mem n need) outputs in
          let input = unkept (prune ~ordered:false (reads_of outputs) input) in
          if is_identity outputs input then input
          else Query.make (Select { outputs; input })
      | Derive { outputs; input } ->
          let outputs = List.filter (fun (n, _) -> List.mem n need) outputs in
          let kept =
            List.filter
              (fun n ->
                Option.is_some (Schema.find (Query.schema input) n)
                && (ordered || not (List.mem_assoc n outputs)))
              need
          in
          let input = prune ~ordered (union kept (reads_of outputs)) input in
          if List.is_empty outputs then input
          else Query.make (Derive { outputs; input })
      | Aggregate { by; outputs; input } ->
          let outputs = List.filter (fun (n, _) -> List.mem n need) outputs in
          let input =
            unkept (prune ~ordered:false (union by (reads_of outputs)) input)
          in
          Query.make (Aggregate { by; outputs; input })
      | Join ({ on; left; right; _ } as r) ->
          let lefts = Schema.names (Query.schema left) in
          let ls, rs = cond_columns on in
          let left_need =
            union (List.filter (fun n -> List.mem n lefts) need) ls
          in
          let right_need =
            union
              (List.filter
                 (fun n ->
                   (not (List.mem n lefts))
                   && Option.is_some (Schema.find (Query.schema right) n))
                 need)
              rs
          in
          Query.make
            (Join
               {
                 r with
                 left = prune ~ordered left_need left;
                 right = prune ~ordered right_need right;
               })
      | Append { input; rest } ->
          let side ~ordered q = keep need (prune ~ordered need q) in
          Query.make
            (Append
               { input = side ~ordered input; rest = side ~ordered:false rest })
      | Unnest { columns; input } ->
          Query.make
            (Unnest
               { columns; input = prune ~ordered (union need columns) input }))

(* [place need fs leaf] is the filters of the conjuncts [fs] over [leaf], a
   source, read with this place's request. The place offers the source the
   conjuncts that are predicates, its request's and its filters', and those the
   source applies exactly go. *)
and place need fs leaf =
  let source, request =
    match Query.node leaf with
    | Of_source r -> (r.source, r.filters)
    | _ -> assert false (* [chain] ends at a source. *)
  in
  let offered =
    List.fold_left
      (fun acc c -> if mem_id c acc then acc else acc @ [ c ])
      []
      (request @ List.filter (fun c -> Option.is_some (pred c)) (List.concat fs))
  in
  let answered =
    List.map (fun c -> (c, source.pushdown (Option.get (pred c)))) offered
  in
  let with_answer f =
    List.filter_map (fun (c, a) -> if f a then Some c else None) answered
  in
  let handed = with_answer (fun a -> a <> Source.Unsupported)
  and exact = with_answer (fun a -> a = Source.Exact) in
  let fs = List.map (List.filter (fun c -> not (mem_id c exact))) fs in
  let read =
    List.fold_left
      (fun acc cs ->
        List.fold_left (fun acc c -> union acc (Expr.reads c)) acc cs)
      need fs
  in
  let columns =
    List.filter (fun n -> List.mem n read) (Schema.names source.schema)
  in
  let leaf =
    Query.make (Of_source { source; columns; filters = handed; limit = None })
  in
  List.fold_left (fun q cs -> filters cs q) leaf fs

(* [limit offset length input] is [input] sliced, the slice giving the source
   directly below it its limit. A slice from the end of a source that states its
   rows and is handed no conjunct counts from the start. *)
and limit offset length input =
  let sliced = Query.make (Slice { offset; length; input }) in
  match Query.node input with
  | Of_source r -> (
      match r.source.rows with
      | _ when offset >= 0 ->
          let limit = Some (sat_add offset length) in
          Query.make
            (Slice
               {
                 offset;
                 length;
                 input = Query.make (Of_source { r with limit });
               })
      | Some rows when List.is_empty r.filters ->
          let first = max 0 (rows + offset) in
          let last = min rows (sat_add (rows + offset) length) in
          limit first (max 0 (last - first)) input
      | Some _ | None -> sliced)
  | _ -> sliced

(* Sharing *)

let share q =
  let canonical = ref [] in
  let rec go q =
    let q = Query.make (Query.map_inputs go (Query.node q)) in
    match List.find_opt (Query.same q) !canonical with
    | Some c -> c
    | None ->
        canonical := q :: !canonical;
        q
  in
  go q

(* Optimizing *)

(* [once q] is [q] optimized by one run of the passes. *)
let once q =
  let need = Schema.names (Query.schema q) in
  share (prune ~ordered:true need (push [] (fold q)))

(* [steps q] is the number of distinct steps of [q]. *)
let steps q =
  let seen = ref [] in
  let rec visit q =
    if not (List.memq q !seen) then begin
      seen := q :: !seen;
      List.iter visit (Query.inputs q)
    end
  in
  visit q;
  List.length !seen

(* Removing a step that nothing reads can let a conjunct or a slice that stopped
   at it move further, so [query] repeats the passes until the plan no longer
   changes. A round after the first changes the plan only past steps that the
   round before removed, so a plan of n steps settles within n + 2 rounds. *)
let query q =
  let bound = steps q + 2 in
  let rec fix rounds q =
    assert (rounds <= bound);
    let q' = once q in
    if Query.same q' q then q'
    else
      fix
        ((rounds + 1)
         [@mutate off "the count matters only to a plan that never settles"])
        q'
  in
  let optimized = fix 1 q in
  assert (Schema.equal (Query.schema optimized) (Query.schema q));
  optimized
