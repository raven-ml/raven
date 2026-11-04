(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Nx.Dtype
module L = Nx_array.Layout
module Ptree = Nx.Ptree

let report = Compile.report

(* Keys *)

type place = At : 'd Nx.Placement.t -> place

(* What a trace depends on of a leaf: its abstract value. *)
type leaf = { dtype : D.any; shape : int array; at : place }

let leaf (Nx.P x) =
  let f = Nx.Prim.form x in
  {
    dtype = D.Any f.dtype;
    shape = Nx.shape x;
    at = At f.placement;
  }

let same_place (At p) (At q) = Nx.Placement.equal p q

let same_leaf a b =
  (match (a.dtype, b.dtype) with D.Any x, D.Any y -> D.equal x y)
  && a.shape = b.shape && same_place a.at b.at

(* What a call's program depends on of its arguments and of where the running
   values of the totals it threads lie, where a call found them elsewhere than a
   trace without them took them: each total's number and placement. The answers
   to the questions a trace asks of the call's context are its entry's
   ({!current}). *)
type key = {
  skeleton : Ptree.Skeleton.t;
  leaves : leaf array;
  totals : (int * place) array;
}

let same_key k k' =
  Ptree.Skeleton.equal k.skeleton k'.skeleton
  && Array.length k.leaves = Array.length k'.leaves
  && Array.for_all2 same_leaf k.leaves k'.leaves
  && Array.length k.totals = Array.length k'.totals
  && Array.for_all2
       (fun (a, p) (b, q) -> a = b && same_place p q)
       k.totals k'.totals

let hash k =
  Array.fold_left
    (fun h l -> (h * 31) + Hashtbl.hash l.shape)
    (Ptree.Skeleton.hash k.skeleton)
    k.leaves

(* A leaf's parts, in the order a retrace names the first that differs. *)
let parts l =
  let (D.Any dt) = l.dtype in
  [
    D.name dt;
    Format.asprintf "shape %a" Nx.pp_shape l.shape;
    (let (At p) = l.at in
     Format.asprintf "at %a" Nx.Placement.pp p);
  ]

(* The first difference between the key [k] and the previous key [k']. *)
let difference paths k k' =
  match
    Ptree.Skeleton.diff ~this:"here" k.skeleton ~that:"in the previous key"
      k'.skeleton
  with
  | Some m -> m
  | None ->
      let place (At p) = Format.asprintf "at %a" Nx.Placement.pp p in
      (* A running total that lies elsewhere, or one the previous key did not
         place. *)
      let total () =
        let at k id =
          Array.find_map
            (fun (j, p) -> if j = id then Some p else None)
            k.totals
        in
        let ids = Array.append k.totals k'.totals in
        match
          Array.find_map
            (fun (id, _) ->
              let a = at k id and b = at k' id in
              match (a, b) with
              | Some p, Some q when same_place p q -> None
              | _ ->
                  let name = Option.fold ~none:"not taken" ~some:place in
                  Some
                    (Printf.sprintf
                       "a running total: %s here, %s in the previous key"
                       (name a) (name b)))
            ids
        with
        | Some m -> m
        | None -> "an equal key"
      in
      let rec first i paths =
        if i >= Array.length k.leaves then total ()
        else
          let l = k.leaves.(i) and l' = k'.leaves.(i) in
          let path, rest =
            match paths with p :: rest -> (p, rest) | [] -> ("a leaf", [])
          in
          if same_leaf l l' then first (i + 1) rest
          else
            match
              List.find_opt
                (fun (a, b) -> not (String.equal a b))
                (List.combine (parts l) (parts l'))
            with
            | Some (a, b) ->
                Printf.sprintf "%s: %s here, %s in the previous key" path a b
            | None -> path ^ ": another placement"
      in
      first 0 paths

(* Programs *)

(* A key's programs, the [n]th traced under [(key, n)]. *)
module Programs = Cache.Make (struct
  type t = key * int

  let equal (k, n) (k', n') = n = n' && same_key k k'
  let hash (k, n) = hash k + n
end)

type probe = Probe : 'a Program.probe -> probe

let probe_id (Probe p) = Program.probe_id p

type 'y entry = {
  body : Program.t;
  like : 'y;
  site : Program.site;
  axes : Program.axis list;
  lanes : int list;
  asked : probe list;
  probes : (int * bool) list;
}

(* A map's lanes, [-1] for none. *)
let count = function Some n -> n | None -> -1

(* The lanes of each map of [axes] around the call. *)
let lanes axes = List.map (fun a -> count (Program.bound a)) axes

(* Whether the maps of [axes] around the call have [ns] lanes, [lanes axes =
   ns] without building the list: a call compares them on every hit. *)
let rec same_lanes axes ns =
  match (axes, ns) with
  | [], [] -> true
  | a :: axes, n :: ns -> count (Program.bound a) = n && same_lanes axes ns
  | [], _ :: _ | _ :: _, [] -> false

(* Whether a collect of each probe of [ps] is open around the call as [os]
   says, by each probe's id. *)
let rec same_opened ps os =
  match (ps, os) with
  | [], [] -> true
  | Probe p :: ps, (id, o) :: os ->
      Program.probe_id p = id && Program.opened p = o && same_opened ps os
  | [], _ :: _ | _ :: _, [] -> false

(* Whether the call answers each question [e]'s trace asked as the trace was
   answered, so that [e]'s program is the call's: its trace would ask the same
   questions in the same order. *)
let current e = same_lanes e.axes e.lanes && same_opened e.asked e.probes

(* [xs] then each of [ys] that [same] finds in none before it, in order. *)
let union same xs ys =
  List.fold_left
    (fun acc y -> if List.exists (same y) acc then acc else acc @ [ y ])
    xs ys

(* The programs of a key, in the order traced. *)
module Families = Cache.Make (struct
  type t = key

  let equal = same_key
  let hash = hash
end)

(* The key of the latest call that looked one up in the table, and its programs.
   A call that finds a current one there reads it without the tables' locks. *)
type 'y last = { key : key; family : 'y entry list Atomic.t }

type 'y t = {
  name : string;
  beam : int option;
  parallel : int option;
  table : 'y entry Programs.t;
  families : 'y entry list Atomic.t Families.t;
  last : 'y last option Atomic.t;
}

let create ~name ?beam ?parallel () =
  {
    name;
    beam;
    parallel;
    table = Programs.create ();
    families = Families.create ();
    last = Atomic.make None;
  }

(* The stand-in of a leaf: its abstract value, in C order. *)
let stand_in (Nx.P x) =
  let f = Nx.Prim.form x in
  Program.F { f with layout = L.contiguous (L.shape f.layout) }

let stand_ins leaves = Array.of_list (List.map stand_in leaves)

(* Offers as outputs

   A compiled program offers as an evaluated one would: each emission becomes
   outputs of the program, which the call offers once it has run. The program is
   traced again with its emissions caught. A scan whose step emits is a scan of
   the step traced so, its emissions more outputs stacked over its rows; a call,
   a remat and a custom rule run their function in place, as the lowering does.
   Another construct whose programs emit, and a program that checks on the
   host, whose failure there would lose its outputs, are evaluated. Off the
   host the call's check raises at a later wait, so every emission is
   offered. *)

exception Kept

(* Whether [p] or a program it carries checks: by a [Check], or by an
   iterate's [max] or a switch's index, which the call's check holds too. *)
let rec checks (p : Program.t) =
  Array.exists
    (function
      | Program.Op { op = Nx.Prim.Check _; _ } -> true
      | Op _ -> false
      | Do { c = Program.Iterate _ | Program.Switch _; _ } -> true
      | Do { c; _ } -> List.exists checks (Program.carried c))
    p.eqns

(* An emission's arrays, its mask last. *)
let arrays (Program.Emission e) =
  Array.append e.values (Array.of_list (Option.to_list e.mask))

(* [e] over [vs], its arrays then its mask's. *)
let over (Program.Emission e) vs =
  let n = Array.length e.values in
  let mask = if Option.is_some e.mask then Some vs.(n) else None in
  Program.Emission { e with values = Array.sub vs 0 n; mask }

(* A scan's emission [e] whose arrays stack the rows' along a new axis 0: the
   rows merged after its lanes, in the order they ran. *)
let rowed ~reverse (Program.Emission e) =
  let row (Program.V y) =
    let y = if reverse then Nx.flip ~axes:[ 0 ] y else y in
    let y = Nx.moveaxis 0 e.lead y in
    let s = Nx.shape y in
    let l = e.lead in
    let shape =
      Array.concat
        [
          Array.sub s 0 l;
          [| s.(l) * s.(l + 1) |];
          Array.sub s (l + 2) (Array.length s - l - 2);
        ]
    in
    Program.V (Nx.reshape shape y)
  in
  over (Program.Emission e) (Array.map row (arrays (Program.Emission e)))

(* [f vs] traced over [forms] with its emissions as outputs, before the running
   values out of the totals [adds] it threads: the program and its emissions, in
   order, each over the variables its arrays are. *)
let rec lifted ~name ~adds forms f =
  let ems = ref [] in
  let body vs =
    let catch (type a) (e : a Effect.t) :
        ((a, _) Effect.Deep.continuation -> _) option =
      match e with
      | Program.Construct { c = Some (Emit em) } ->
          Some
            (fun k ->
              ems := em :: !ems;
              Effect.Deep.continue k [||])
      | _ -> None
    in
    let out =
      Effect.Deep.try_with
        (fun () ->
          Program.gathering (fun () -> converted ~name (fun () -> f vs)))
        () { effc = catch }
    in
    let na = Array.length adds in
    let n = Array.length out - na in
    Array.concat
      ([ Array.sub out 0 n ] @ List.rev_map arrays !ems @ [ Array.sub out n na ])
  in
  let q, captures = Trace.trace Jit ~name forms body in
  if captures <> [||] || q.Program.adds <> [||] then raise Kept;
  let q =
    Program.make ~owner:q.owner ~params:q.params (Array.to_list q.eqns)
      ~outputs:q.outputs ~adds ~slots:q.slots
  in
  (q, List.rev !ems)

(* [f ()] with each construct whose programs emit performed so that its
   emissions are performed here. *)
and converted ~name f =
  let effc (type a) (e : a Effect.t) :
      ((a, _) Effect.Deep.continuation -> _) option =
    match e with
    | Program.Construct { c = Some c }
      when List.exists (fun (p : Program.t) -> p.emits) (Program.carried c) -> (
        match c with
        | Call _ | Remat _ | Custom _ ->
            Some (fun k -> Effect.Deep.discontinue k Program.Evaluate)
        | Scan { trips = None; _ } ->
            Some (fun k -> Effect.Deep.continue k (scan ~name c))
        | _ -> Some (fun k -> Effect.Deep.discontinue k Kept))
    | _ -> None
  in
  Effect.Deep.try_with f () { effc }

(* A scan whose step emits: the scan of its step with its emissions as more
   outputs, then each emission of their rows. *)
and scan ~name (c : Program.construct) : Program.value array =
  match c with
  | Scan s ->
      let step = s.step in
      let nc = Array.length s.init and na = Array.length step.adds in
      let ny = Array.length step.outputs - nc - na in
      let forms =
        Array.concat
          [
            Array.map Program.form s.init;
            Array.map Loops.row s.xs;
            Array.map Program.form s.captures;
            Array.map
              (fun (Program.Addition (_, f, _)) -> Program.F f)
              step.adds;
          ]
      in
      let step', ems =
        lifted ~name ~adds:step.adds forms (fun vs ->
            let n = Array.length vs - na in
            Program.eval ~totals:(Array.sub vs n na) step (Array.sub vs 0 n))
      in
      let r = Program.perform (Scan { s with step = step' }) in
      let at = ref (nc + ny) in
      List.iter
        (fun em ->
          let n = Array.length (arrays em) in
          let e = rowed ~reverse:s.reverse (over em (Array.sub r !at n)) in
          at := !at + n;
          ignore (Program.perform (Emit e)))
        ems;
      Array.append
        (Array.sub r 0 (nc + ny))
        (Array.sub r !at (Array.length r - !at))
  | _ -> invalid_arg "Rune: a scan's emissions"

(* Whether [body] computes on the host's set: its first argument's, or else
   its first result's. *)
let on_host (body : Program.t) forms =
  let firsts = Array.append forms (Array.map Program.form body.outputs) in
  Array.length firsts = 0
  ||
  let (Program.F f) = firsts.(0) in
  Option.is_some (Nx.Placement.same f.placement Nx.Host.on)

(* [body] over arguments of [forms] with its emissions as outputs, or [None]
   where it is to be evaluated. *)
let offered name (body : Program.t) forms =
  if (not body.emits) || (checks body && on_host body forms) then None
  else
    let na = Array.length body.adds in
    match
      lifted ~name ~adds:body.adds forms (fun vs ->
          let n = Array.length vs - na in
          Program.eval ~totals:(Array.sub vs n na) body (Array.sub vs 0 n))
    with
    | lifted -> Some lifted
    | exception Kept -> None

(* [out] of [body]'s outputs and then [ems]' arrays before its running values
   out, each emission performed: [body]'s outputs. *)
let delivered (body : Program.t) ems out =
  if ems = [] then out
  else
    let n = Array.length body.outputs - Array.length body.adds in
    let at = ref n in
    List.iter
      (fun em ->
        let k = Array.length (arrays em) in
        let e = over em (Array.sub out !at k) in
        at := !at + k;
        ignore (Program.perform (Emit e)))
      ems;
    Array.append (Array.sub out 0 n)
      (Array.sub out !at (Array.length out - !at))

(* [compiled name body forms] runs [body] compiled, made the first time it runs,
   or evaluated where the compiler declines it. A declined program that runs a
   callback is evaluated outlined: each stretch without one is a compiled call
   of its own. *)
let rec compiled ~beam ~parallel name body forms =
  let made = Atomic.make None and lock = Mutex.create () in
  let get args =
    match Atomic.get made with
    | Some c -> c
    | None -> (
        Mutex.protect lock @@ fun () ->
        match Atomic.get made with
        | Some c -> c
        | None ->
            let p, ems =
              match offered name body forms with
              | Some (p, ems) -> (p, ems)
              | None -> (body, [])
            in
            let c = Compile.make ~name ?beam ?parallel ~first:args p forms in
            Result.iter_error (report "evaluates its program uncompiled: %s") c;
            let q =
              if Result.is_ok c then p
              else
                let compile = compiled ~beam ~parallel name in
                Program.outline ~name ~compile p
            in
            let r = (q, q != p, ems, c) in
            Atomic.set made (Some r);
            r)
  in
  fun args ->
    let p, fresh, ems, c = get args in
    let out =
      match c with
      | Ok c -> Compile.run c args
      | Error _ ->
          Compile.evaluated_call ();
          Compile.evaluate ~fresh ~name p args
    in
    delivered body ems out

let find ?(totals = [||]) t leaves skeleton ~paths trace =
  let key =
    {
      skeleton;
      leaves = Array.of_list (List.map leaf leaves);
      totals =
        Array.map
          (fun (Program.Addition (u, (f : _ Nx.Prim.form), _)) ->
            (Program.total_id u, At f.placement))
          totals;
    }
  in
  let make () =
    let forms = stand_ins leaves in
    let body, like, asked = trace forms totals in
    (* Each question the trace asked, once, in the order first asked. *)
    let maps =
      List.filter_map
        (function Program.Bound_asked (a, n) -> Some (a, n) | Asked _ -> None)
        asked
      |> union (fun (a, _) (b, _) -> Program.same_axis a b) []
    in
    let probed =
      List.filter_map
        (function
          | Program.Asked (p, o) -> Some (Probe p, o) | Bound_asked _ -> None)
        asked
      |> union (fun (p, _) (q, _) -> probe_id p = probe_id q) []
    in
    let compile = compiled ~beam:t.beam ~parallel:t.parallel t.name in
    let site = Program.site ~name:t.name ~compile body forms in
    let axes =
      union Program.same_axis (List.map fst maps) (Program.axes site)
    in
    let answer a =
      match List.find_opt (fun (b, _) -> Program.same_axis a b) maps with
      | Some (_, n) -> count n
      | None -> count (Program.bound a)
    in
    {
      body;
      like;
      site;
      axes;
      lanes = List.map answer axes;
      asked = List.map fst probed;
      probes = List.map (fun (p, o) -> (probe_id p, o)) probed;
    }
  in
  let previous = Atomic.get t.last in
  (* The first of [family]'s programs that is current. A key's programs are
     numbered as they are traced: a call that finds none current traces under
     the next number, so that a call of another domain with the same answers
     waits for that trace and then finds it current. A program found that is not
     current joined [family] after the call looked, so each turn sees one more,
     and a call traces only where no program answers as it does. *)
  let rec found family =
    let es = Atomic.get family in
    match List.find_opt current es with
    | Some e -> e
    | None ->
        let miss () =
          match es with
          | [] ->
              Option.iter
                (fun l ->
                  report "retrace: %s" (difference (paths ()) key l.key))
                previous
          | _ when List.for_all (fun e -> lanes e.axes <> e.lanes) es ->
              report "retrace: a map around the call differs"
          | _ -> report "retrace: a probe's collect is open or closed"
        in
        let made () =
          let e = make () in
          let rec add () =
            let es = Atomic.get family in
            if not (Atomic.compare_and_set family es (es @ [ e ])) then add ()
          in
          add ();
          e
        in
        let e = Programs.find t.table (key, List.length es) ~miss made in
        if current e then e else found family
  in
  match previous with
  | Some l when same_key l.key key -> found l.family
  | _ ->
      let family =
        Families.find t.families key ~miss:ignore (fun () -> Atomic.make [])
      in
      Atomic.set t.last (Some { key; family });
      found family
