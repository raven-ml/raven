(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Program
module P = Nx_kernel.Prog
module Prim = Nx.Prim

type kind = Body | Jit | Tape | Staged

(* A total the program threads: the variable of its running value in, and its
   running value where the trace is. *)
type thread =
  | Thread : {
      total : ('v, 's, 'd) total;
      into : ('v, 's, 'd) Nx.Prim.form;
      var : int;
      mutable now : ('v, 's, 'd) Nx.t;
    }
      -> thread

type state = {
  kind : kind;
  name : string;
  mutable next : int;  (** The next variable's number. *)
  mutable eqns : eqn list;  (** Newest first. *)
  mutable captures : (value * int) list;
      (** Each capture and its variable's number, newest first. *)
  mutable threads : thread list;  (** Newest first. *)
  hints : addition array;
      (** Where a total's running value in lies, where the caller knows. *)
}

let fresh st i form =
  let n = st.next in
  st.next <- n + 1;
  Nx.Prim.traced i form (Var n)

(* The number of [x]'s variable, if [x] is a capture already. *)
let captured st x =
  List.find_map (fun (y, n) -> if same y x then Some n else None) st.captures

(* [x] as an equation of [i]'s reads it: [i]'s variable, a constant, or the
   variable of a capture. *)
let operand (type v s d) st i (x : (v, s, d) Nx.t) : (v, s, d) Nx.t =
  match Nx.Prim.owner x with
  | None -> x
  | Some j when j == i -> x
  | Some _ -> (
      match st.kind with
      | Jit ->
          invalid_arg
            (st.name
           ^ ": the function captures a value another transformation traces; \
              pass it as an argument")
      | Body | Tape | Staged -> (
          match captured st (V x) with
          | Some n -> Nx.Prim.traced i (Nx.Prim.form x) (Var n)
          | None ->
              st.captures <- (V x, st.next) :: st.captures;
              fresh st i (Nx.Prim.form x)))

let record_op (type r) st i ~by (op : r Nx.Prim.t) : r =
  let op = Nx.Prim.map (fun x -> operand st i x) op in
  let base = st.next and n = ref 0 in
  let r =
    Nx.Prim.results ~by
      (fun k form ->
        n := Int.max !n (k + 1);
        Nx.Prim.traced i form (Var (base + k)))
      op
  in
  st.next <- base + !n;
  st.eqns <- Op { by; op; base } :: st.eqns;
  r

(* Partial evaluation

   The tape tracer records only what reads its variables. A loop's outputs that
   read none of them compute at the call, outward, over the loads they read; the
   others are recorded over theirs, the values they recompute. *)

let owns i x = match Nx.Prim.owner x with Some j -> j == i | None -> false

let load_owned i : 'd Prim.load -> bool = function
  | Plain x -> owns i x
  | Padded { x; _ } -> owns i x

(* Outputs [outs] of a program over the loads they read, the loads [kept]. *)
type part = { prog : P.t; outs : int array; kept : int array }

let part (p : P.t) outs =
  let keep = Scalar.reads p ~outs in
  let kept =
    Array.of_list
      (List.filter (fun j -> keep.(j)) (List.init (Array.length keep) Fun.id))
  in
  { prog = Scalar.prune p ~keep ~outs; outs; kept }

(* A program's outputs split by whether they read a load of [mask]:
   [unknown.(k)] for output [k], and the parts of each side, where it has
   outputs. *)
type plan = {
  unknown : bool array;
  known : part option;
  recorded : part option;
}

let plan (p : P.t) owned =
  let deps = Scalar.depends p owned in
  let unknown = Array.map (fun o -> deps.(o)) (Iarray.to_array p.outs) in
  let side flag =
    let outs =
      List.filter
        (fun k -> unknown.(k) = flag)
        (List.init (Array.length unknown) Fun.id)
    in
    match outs with [] -> None | outs -> Some (part p (Array.of_list outs))
  in
  { unknown; known = side false; recorded = side true }

(* Plans per domain, found by their program and its owned loads, as jvp's are:
   the last one found, then a table emptied when it holds [entries]. *)
type entry = { p : P.t; m : int; plan : plan }

type cache = {
  mutable last : entry option;
  table : (P.t * int, plan) Hashtbl.t;
}

let entries = 4096

let cache =
  Domain.DLS.new_key (fun () -> { last = None; table = Hashtbl.create 64 })

let plan_of i (p : P.t) (loads : 'd Prim.load iarray) =
  let n = Iarray.length loads in
  let owned () = Array.init n (fun j -> load_owned i (Iarray.get loads j)) in
  if n >= Sys.int_size - 1 then plan p (owned ())
  else
    let m = ref 0 in
    for j = n - 1 downto 0 do
      m := (!m lsl 1) lor Bool.to_int (load_owned i (Iarray.get loads j))
    done;
    let m = !m in
    let c = Domain.DLS.get cache in
    match c.last with
    | Some e when e.p == p && e.m = m -> e.plan
    | _ ->
        let plan =
          match Hashtbl.find_opt c.table (p, m) with
          | Some plan -> plan
          | None ->
              if Hashtbl.length c.table >= entries then Hashtbl.reset c.table;
              let plan = plan p (owned ()) in
              Hashtbl.add c.table (p, m) plan;
              plan
        in
        c.last <- Some { p; m; plan };
        plan

let kept part loads =
  Iarray.init (Array.length part.kept) (fun j -> Iarray.get loads part.kept.(j))

(* Whether a loop over [loads], which it has, keeps a known part that reads
   none of them: computed apart, that part would be a creation with no operand
   to lie beside, so the loop is recorded whole. *)
let unread plan loads =
  Iarray.length loads > 0
  && match plan.known with Some p -> Array.length p.kept = 0 | None -> false

(* A map's outputs split in two maps' and how their results give its own. *)
type ('d, 'r) outs_split =
  | O :
      ('d, 'k) Prim.outs * ('d, 'u) Prim.outs * ('k -> 'u -> 'r)
      -> ('d, 'r) outs_split

let rec split_outs : type d r.
    (d, r) Prim.outs -> bool array -> int -> (d, r) outs_split =
 fun outs unknown k ->
  match outs with
  | [] -> O ([], [], fun () () -> ())
  | dt :: rest ->
      let (O (ko, uo, f)) = split_outs rest unknown (k + 1) in
      if unknown.(k) then O (ko, dt :: uo, fun kr (y, ur) -> (y, f kr ur))
      else O (dt :: ko, uo, fun (y, kr) ur -> (y, f kr ur))

(* A reduction's reductions split likewise, each renumbered to its side's
   program by [index]. *)
type ('d, 'r) reds_split =
  | R :
      ('d, 'k) Prim.reductions * ('d, 'u) Prim.reductions * ('k -> 'u -> 'r)
      -> ('d, 'r) reds_split

let output : type d a. (d, a) Prim.reduction -> int = function
  | Monoid (_, k, _) | Logsumexp (k, _) | Moments (k, _) | Arg (_, k, _) -> k

let renumber : type d a. int -> (d, a) Prim.reduction -> (d, a) Prim.reduction =
 fun k -> function
  | Monoid (m, _, dt) -> Monoid (m, k, dt)
  | Logsumexp (_, dt) -> Logsumexp (k, dt)
  | Moments (_, dt) -> Moments (k, dt)
  | Arg (e, _, dt) -> Arg (e, k, dt)

let rec split_reds : type d r.
    (d, r) Prim.reductions -> bool array -> (int -> int) -> (d, r) reds_split =
 fun rs unknown index ->
  match rs with
  | [] -> R ([], [], fun () () -> ())
  | r :: rest ->
      let (R (kr, ur, f)) = split_reds rest unknown index in
      let k = output r in
      let r = renumber (index k) r in
      if unknown.(k) then R (kr, r :: ur, fun kv (y, uv) -> (y, f kv uv))
      else R (r :: kr, ur, fun (y, kv) uv -> (y, f kv uv))

(* The position of each output of [part] among its outputs. *)
let positions n = function
  | None -> Array.make n (-1)
  | Some part ->
      let a = Array.make n (-1) in
      Array.iteri (fun i o -> a.(o) <- i) part.outs;
      a

let map_parts (type d r) st i ~by layout prog (outs : (d, r) Prim.outs)
    (loads : d Prim.load iarray) at : r =
  let plan = plan_of i prog loads in
  match (outs, plan.known, plan.recorded) with
  | _ when unread plan loads ->
      record_op st i ~by (Map { layout; prog; outs; loads; at })
  | [ a; b ], Some k, Some u when not plan.unknown.(0) ->
      (* A primal and its tangent, as forward mode makes them: without the
         general split. *)
      let y, () =
        Prim.eval ~by
          (Map
             {
               layout;
               prog = k.prog;
               outs = [ a ];
               loads = kept k loads;
               at = None;
             })
      in
      let t, () =
        record_op st i ~by
          (Map
             {
               layout;
               prog = u.prog;
               outs = [ b ];
               loads = kept u loads;
               at = None;
             })
      in
      (y, (t, ()))
  | _ ->
      let (O (ko, uo, f)) = split_outs outs plan.unknown 0 in
      let known : type k. (d, k) Prim.outs -> k = function
        | [] -> ()
        | _ :: _ as ko ->
            let part = Option.get plan.known in
            (* With no loads the map is a creation, at its own placement. *)
            Prim.eval ~by
              (Map
                 {
                   layout;
                   prog = part.prog;
                   outs = ko;
                   loads = kept part loads;
                   at = (if Iarray.length loads = 0 then at else None);
                 })
      in
      let recorded : type u. (d, u) Prim.outs -> u = function
        | [] -> ()
        | _ :: _ as uo ->
            let part = Option.get plan.recorded in
            record_op st i ~by
              (Map
                 {
                   layout;
                   prog = part.prog;
                   outs = uo;
                   loads = kept part loads;
                   at = None;
                 })
      in
      let kr = known ko in
      f kr (recorded uo)

let reduce_parts (type d r) st i ~by layout axes prog
    (reductions : (d, r) Prim.reductions) (loads : d Prim.load iarray) : r =
  let plan = plan_of i prog loads in
  if unread plan loads then
    record_op st i ~by (Reduce { layout; axes; prog; reductions; loads })
  else
    let n = Array.length plan.unknown in
    let k_at = positions n plan.known and u_at = positions n plan.recorded in
    let index k = if plan.unknown.(k) then u_at.(k) else k_at.(k) in
    let (R (kr, ur, f)) = split_reds reductions plan.unknown index in
    let known : type k. (d, k) Prim.reductions -> k = function
      | [] -> ()
      | _ :: _ as kr ->
          let part = Option.get plan.known in
          Prim.eval ~by
            (Reduce
               {
                 layout;
                 axes;
                 prog = part.prog;
                 reductions = kr;
                 loads = kept part loads;
               })
    in
    let recorded : type u. (d, u) Prim.reductions -> u = function
      | [] -> ()
      | _ :: _ as ur ->
          let part = Option.get plan.recorded in
          record_op st i ~by
            (Reduce
               {
                 layout;
                 axes;
                 prog = part.prog;
                 reductions = ur;
                 loads = kept part loads;
               })
    in
    let kv = known kr in
    f kv (recorded ur)

let scan_part (type d r) st i ~by layout axis prog
    (reduction : (d, r) Prim.reduction) (loads : d Prim.load iarray)
    (carry : d Prim.any option) : r =
  let plan = plan_of i prog loads in
  let carried = match carry with Some (Any c) -> owns i c | None -> false in
  if carried || plan.unknown.(output reduction) || unread plan loads then
    record_op st i ~by (Scan { layout; axis; prog; reduction; loads; carry })
  else
    let part = Option.get plan.known in
    let at = positions (Array.length plan.unknown) plan.known in
    let reduction = renumber at.(output reduction) reduction in
    Prim.eval ~by
      (Scan
         {
           layout;
           axis;
           prog = part.prog;
           reduction;
           loads = kept part loads;
           carry;
         })

let rule (type r) st i ~by (op : r Nx.Prim.t) : r =
  match (st.kind, op) with
  | Tape, Map { layout; prog; outs; loads; at } ->
      map_parts st i ~by layout prog outs loads at
  | Tape, Reduce { layout; axes; prog; reductions; loads } ->
      reduce_parts st i ~by layout axes prog reductions loads
  | Tape, Scan { layout; axis; prog; reduction; loads; carry } ->
      scan_part st i ~by layout axis prog reduction loads carry
  | _ -> record_op st i ~by op

let owned i (V x) = owns i x

(* Whether [i]'s handler answers [c]: a construct over its variables, or any of
   the extent of a tracer that stages every operation. A map's operation with
   no such map around is left to its default, which raises. An addition reaches
   only a tape: the others answer for the totals' running values
   ({!thread}). *)
let claims st i = function
  | Add (_, x) -> st.kind = Tape && owns i x
  (* An abstract operator is recorded, to be filled in, whatever it reads. *)
  | Operator _ -> true
  | (Lanes (a, _) | Lane_index (a, _)) when bound a = None -> false
  | c -> st.kind = Jit || st.kind = Staged || List.exists (owned i) (operands c)

(* The running value of [t] at the shape of [f] where the trace is: its
   thread's, made where the trace has none, its running value in a parameter of
   [f]'s shape. The parameter lies where the trace's hints place it, as a jit
   does where a call found the running value, and otherwise where [f], the
   first addition, does. *)
let thread (type v s d) st i (t : (v, s, d) total) (f : (v, s, d) Nx.Prim.form)
    : (v, s, d) Nx.t =
  let shape = Iarray.to_array (Nx_array.Layout.shape f.layout) in
  let rec find : thread list -> (v, s, d) Nx.t option = function
    | [] -> None
    | Thread th :: rest -> (
        match same_total th.total t with
        | Some Type.Equal
          when Iarray.to_array (Nx_array.Layout.shape th.into.layout) = shape ->
            Some th.now
        | _ -> find rest)
  in
  match find st.threads with
  | Some x -> x
  | None ->
      let rec hinted : addition list -> d Nx.Placement.t option = function
        | [] -> None
        | Addition (u, into, _) :: rest -> (
            match same_total u t with
            | Some Type.Equal
              when Iarray.to_array (Nx_array.Layout.shape into.layout) = shape
              ->
                Some into.placement
            | _ -> hinted rest)
      in
      let placement =
        match hinted (Array.to_list st.hints) with
        | Some p -> p
        | None -> f.placement
      in
      let into =
        {
          f with
          layout = Nx_array.Layout.contiguous (Iarray.of_array shape);
          placement;
        }
      in
      let var = st.next in
      let now = fresh st i into in
      st.threads <- Thread { total = t; into; var; now } :: st.threads;
      now

(* Whether the trace holds a running value of [t], and where it lies. *)
let held st t =
  List.exists
    (fun (Thread th) -> Option.is_some (same_total th.total t))
    st.threads

let placement_of (type v s d) st (t : (v, s, d) total) : d Nx.Placement.t =
  let rec go : thread list -> d Nx.Placement.t = function
    | [] -> invalid_arg "Rune: a total the trace threads no running value of"
    | Thread th :: rest -> (
        match same_total th.total t with
        | Some Type.Equal -> Nx.placement th.now
        | None -> go rest)
  in
  go st.threads

(* [t]'s running value becomes [x], [x] as the trace reads it. *)
let replace (type v s d) st i (t : (v, s, d) total) (x : (v, s, d) Nx.t) =
  let shape = Nx.shape x in
  let rec find = function
    | [] ->
        invalid_arg "Rune: a running value of a total the trace does not hold"
    | Thread th :: rest -> (
        match same_total th.total t with
        | Some Type.Equal
          when Iarray.to_array (Nx_array.Layout.shape th.into.layout) = shape ->
            th.now <- operand st i x
        | _ -> find rest)
  in
  find st.threads

let record st i = function
  | Add _ -> invalid_arg (st.name ^ ": a tangent is added to a total")
  | c ->
      let c = map_operands { f = (fun x -> operand st i x) } c in
      let forms = forms c in
      let base = st.next in
      let r = Array.map (fun (F f) -> V (fresh st i f)) forms in
      st.eqns <- Do { c; base } :: st.eqns;
      r

(* Partial evaluation of programs

   The tape tracer splits a loop or a remat over its variables as it splits a
   map: what reads none of them computes at the call, the rest is recorded. *)

let nots = Array.map not

(* A residual of a scan's step: a capture of the step, a row, or a value of a
   step that the known loop stacks. *)
type residual = Capture of int | Row of int | Stacked

let index_of v vs =
  let r = ref None in
  Array.iteri (fun j w -> if !r = None && same v w then r := Some j) vs;
  !r

let rec answer st i c =
  match (st.kind, c) with
  | Tape, Scan { step; length; trips; reverse; init; xs; captures; seeds } ->
      scan_parts st i step ~length ~trips ~reverse ~init ~xs ~captures ~seeds
  | Tape, Remat { body; args; seeds } -> remat_parts st i body args ~seeds
  | Tape, Call { body; args; site; seeds } ->
      call_parts st i body args site ~seeds
  | Tape, Iterate _ -> iterate_parts st i c
  | Tape, Switch _ -> switch_parts st i c
  | _ -> record st i c

(* A scan over the tape's variables is two scans, as JAX's partial evaluation of
   a scan gives: a known scan, computed at the call, of the carries and outputs
   that read none of them, which stacks the step's residuals as more outputs;
   and a recorded linear scan of the others, reading the residuals as rows. A
   carry is unknown where its init is or where the step makes it so, a fixpoint.
   A residual that is a capture or a row of the step is read where it lies, not
   stacked. *)
and scan_parts st i (step : Program.t) ~length ~trips ~reverse ~init ~xs
    ~captures ~seeds =
  let name = st.name in
  let nc = Array.length init and nx = Array.length xs in
  let ny = Array.length step.outputs - nc - Array.length step.adds in
  let ux = Array.map (owned i) xs and ucap = Array.map (owned i) captures in
  let rows = Array.map Loops.row xs and cforms = Array.map form captures in
  let rec fix u (cf : form array) =
    let gained = Array.make nc false and uy = ref [||] in
    let kinds = ref [||] and tape = ref None in
    let kforms =
      Array.concat
        [
          Loops.pick (nots u) cf;
          Loops.pick (nots ux) rows;
          Loops.pick (nots ucap) cforms;
        ]
    and uforms =
      Array.concat
        [ Loops.pick u cf; Loops.pick ux rows; Loops.pick ucap cforms ]
    in
    let count m = Array.length (Loops.pick m m) in
    let known kvs =
      match
        Loops.split
          [ nc - count u; nx - count ux; Array.length captures - count ucap ]
          kvs
      with
      | [ kc; kr; kk ] ->
          let stash = ref [||] in
          let p, res =
            trace_in Tape ~name uforms (fun inner uvs ->
                match Loops.split [ count u; count ux; count ucap ] uvs with
                | [ uc; ur; uk ] ->
                    let args =
                      Array.concat
                        [
                          Loops.merge u uc kc;
                          Loops.merge ux ur kr;
                          Loops.merge ucap uk kk;
                        ]
                    in
                    (* The totals' running values are the known trace's: a
                       tangent added to one raises as it is handed back. *)
                    let out =
                      threaded step (fun totals -> eval ~totals step args)
                    in
                    let mine (V y) = owns inner y in
                    let c = Array.sub out 0 nc and y = Array.sub out nc ny in
                    Array.iteri
                      (fun k v -> gained.(k) <- (not u.(k)) && mine v)
                      c;
                    uy := Array.map mine y;
                    (* A carry that gained a variable is traced again: its known
                       output stands in meanwhile. *)
                    let kc' =
                      Array.mapi
                        (fun k v -> if gained.(k) then args.(k) else v)
                        c
                    in
                    stash :=
                      Array.append
                        (Loops.pick (nots u) kc')
                        (Loops.pick (nots !uy) y);
                    Array.append (Loops.pick u c) (Loops.pick !uy y)
                | _ -> invalid_arg "Rune: a scan's parts")
          in
          tape := Some p;
          kinds :=
            Array.map
              (fun r ->
                match index_of r kk with
                | Some j -> Capture j
                | None -> (
                    match index_of r kr with Some j -> Row j | None -> Stacked))
              res;
          let stacked =
            Loops.pick (Array.map (fun k -> k = Stacked) !kinds) res
          in
          Array.append !stash stacked
      | _ -> invalid_arg "Rune: a scan's parts"
    in
    let kprog, own = trace Body ~name kforms known in
    if Array.exists Fun.id gained then fix (Array.map2 ( || ) u gained) cf
    else (u, !uy, kprog, own, Option.get !tape, !kinds)
  in
  let u = Array.map (owned i) init in
  let u, uy, kprog, own, tape, kinds = fix u (Array.map form init) in
  let ku = nots u and kuy = nots uy in
  let kcaps = Loops.pick (nots ucap) captures in
  let kxs = Loops.pick (nots ux) xs in
  let kr =
    perform
      (Scan
         {
           step = kprog;
           length;
           trips;
           reverse;
           init = Loops.pick ku init;
           xs = kxs;
           captures = Array.append kcaps own;
           seeds;
         })
  in
  let nkc = Array.length (Loops.pick ku ku)
  and nky = Array.length (Loops.pick kuy kuy) in
  let is k = Array.map (fun r -> r = k) kinds in
  let nst = Array.length (Loops.pick (is Stacked) kinds) in
  let stacked = Array.sub kr (nkc + nky) nst in
  let sums =
    Array.sub kr (nkc + nky + nst) (Array.length kr - nkc - nky - nst)
  in
  let nuc = Array.length (Loops.pick u u)
  and nuy = Array.length (Loops.pick uy uy) in
  let lr =
    if nuc + nuy = 0 then [||]
    else begin
      (* The tape's parameters: its unknown carries, rows and captures, then the
         residuals; the linear step's: the carries, the stacked residuals, the
         residual rows and the unknown rows, then the unknown captures and the
         residual captures. *)
      let nur = Array.length (Loops.pick ux ux) in
      let nuk = Array.length (Loops.pick ucap ucap) in
      let nu = nuc + nur + nuk in
      let res p =
        List.filter
          (fun j -> p kinds.(j))
          (List.init (Array.length kinds) Fun.id)
      in
      let stacked_r = res (fun k -> k = Stacked) in
      let rows_r = res (function Row _ -> true | _ -> false) in
      let caps_r = res (function Capture _ -> true | _ -> false) in
      let range a n = List.init n (fun j -> a + j) in
      let order =
        List.concat
          [
            range 0 nuc;
            List.map (fun j -> nu + j) stacked_r;
            List.map (fun j -> nu + j) rows_r;
            range nuc nur;
            range (nuc + nur) nuk;
            List.map (fun j -> nu + j) caps_r;
          ]
      in
      let at j =
        match kinds.(j) with
        | Row r -> kxs.(r)
        | Capture c -> kcaps.(c)
        | Stacked -> invalid_arg "Rune: a stacked residual"
      in
      let linear =
        Scan
          {
            step = Loops.reorder tape (Array.of_list order);
            length;
            trips;
            reverse;
            init = Loops.pick u init;
            xs =
              Array.concat
                [
                  stacked; Array.of_list (List.map at rows_r); Loops.pick ux xs;
                ];
            captures =
              Array.append (Loops.pick ucap captures)
                (Array.of_list (List.map at caps_r));
            seeds = [||];
          }
      in
      record st i linear
    end
  in
  let next a =
    let k = ref (-1) in
    fun () ->
      incr k;
      a.(!k)
  in
  let kn = next kr and un = next lr in
  let carry = Array.map (fun u -> if u then un () else kn ()) u in
  let ys = Array.map (fun u -> if u then un () else kn ()) uy in
  Array.concat [ carry; ys; sums ]

(* An iterate over the tape's variables is a known iterate, computed at the
   call, of the carries and outputs that read none of them, which stacks the
   step's residuals over the steps it takes, and a recorded linear scan of the
   others over those steps, reading the residuals as rows. Its stop reads known
   values alone. The stacks hold the steps taken where the known iterate runs at
   the call, its operands no tracer's and no map read in its programs; and
   otherwise [max] rows, the linear scan running the rows of the steps taken. A
   residual that is a capture of the step is read where it lies. *)
and iterate_parts st i (c : construct) =
  match c with
  | Iterate it ->
      let name = st.name in
      let nc = Array.length it.init in
      let ny = Array.length it.step.outputs - nc - Array.length it.step.adds in
      let captures = it.step_captures in
      let ucap = Array.map (owned i) captures in
      if Array.exists (owned i) it.until_captures then
        invalid_arg (name ^ ": an iterate's stop reads a tangent");
      let cforms = Array.map form captures in
      let rec fix u (cf : form array) =
        let gained = Array.make nc false and uy = ref [||] in
        let kinds = ref [||] and tape = ref None in
        let kforms =
          Array.append (Loops.pick (nots u) cf) (Loops.pick (nots ucap) cforms)
        and uforms = Array.append (Loops.pick u cf) (Loops.pick ucap cforms) in
        let count m = Array.length (Loops.pick m m) in
        let known kvs =
          match
            Loops.split [ nc - count u; Array.length captures - count ucap ] kvs
          with
          | [ kc; kk ] ->
              let stash = ref [||] in
              let p, res =
                trace_in Tape ~name uforms (fun inner uvs ->
                    match Loops.split [ count u; count ucap ] uvs with
                    | [ uc; uk ] ->
                        let args =
                          Array.append (Loops.merge u uc kc)
                            (Loops.merge ucap uk kk)
                        in
                        let out =
                          threaded it.step (fun totals ->
                              eval ~totals it.step args)
                        in
                        let mine (V y) = owns inner y in
                        let c = Array.sub out 0 nc
                        and y = Array.sub out nc ny in
                        Array.iteri
                          (fun k v -> gained.(k) <- (not u.(k)) && mine v)
                          c;
                        uy := Array.map mine y;
                        let kc' =
                          Array.mapi
                            (fun k v -> if gained.(k) then args.(k) else v)
                            c
                        in
                        stash :=
                          Array.append
                            (Loops.pick (nots u) kc')
                            (Loops.pick (nots !uy) y);
                        Array.append (Loops.pick u c) (Loops.pick !uy y)
                    | _ -> invalid_arg "Rune: an iterate's parts")
              in
              tape := Some p;
              kinds :=
                Array.map
                  (fun r ->
                    match index_of r kk with
                    | Some j -> Capture j
                    | None -> Stacked)
                  res;
              Array.append !stash
                (Loops.pick (Array.map (fun k -> k = Stacked) !kinds) res)
          | _ -> invalid_arg "Rune: an iterate's parts"
        in
        let kprog, own = trace Body ~name kforms known in
        if Array.exists Fun.id gained then fix (Array.map2 ( || ) u gained) cf
        else (u, !uy, kprog, own, Option.get !tape, !kinds)
      in
      let u = Array.map (owned i) it.init in
      let u, uy, kprog, own, tape, kinds = fix u (Array.map form it.init) in
      let ku = nots u and kuy = nots uy in
      (* The stop over the known carries: an unknown one it does not read. *)
      let kuntil, uown =
        trace Body ~name
          (Array.append
             (Loops.pick ku (Array.map form it.init))
             (Array.map form it.until_captures))
          (fun vs ->
            let nk = Array.length (Loops.pick ku ku) in
            let kc = Array.sub vs 0 nk
            and caps = Array.sub vs nk (Array.length vs - nk) in
            let filler =
              Array.map
                (fun v -> Loops.filled ~zero:true (form v))
                (Loops.pick u it.init)
            in
            threaded it.until (fun totals ->
                eval ~totals it.until
                  (Array.append (Loops.merge u filler kc) caps)))
      in
      if reads_params it.until (Array.mapi (fun k _ -> u.(k)) it.init) then
        invalid_arg (name ^ ": an iterate's stop reads a tangent");
      let kinit = Loops.pick ku it.init in
      let kcaps = Loops.pick (nots ucap) captures in
      let operands =
        Array.concat [ kinit; kcaps; own; it.until_captures; uown; it.seeds ]
      in
      let rows =
        match it.rows with
        | Some r -> r
        | None ->
            let untraced (V x) = Option.is_none (Nx.Prim.owner x) in
            if
              Array.for_all untraced operands
              && (not (staging ()))
              && (not (reads_lanes kprog))
              && not (reads_lanes kuntil)
            then Taken
            else Max
      in
      let stopped =
        Option.map
          (fun j ->
            let k = ref (-1) in
            Array.iteri (fun l ku -> if ku && l <= j then incr k) ku;
            !k)
          it.stopped
      in
      let kr =
        perform
          (Iterate
             {
               it with
               step = kprog;
               until = kuntil;
               rows = Some rows;
               stopped;
               init = kinit;
               step_captures = Array.append kcaps own;
               until_captures = Array.append it.until_captures uown;
             })
      in
      let nkc = Array.length kinit
      and nky = Array.length (Loops.pick kuy kuy) in
      let nst =
        Array.length (Loops.pick (Array.map (fun k -> k = Stacked) kinds) kinds)
      in
      let trips = kr.(nkc) in
      let stacked = Array.sub kr (nkc + 1 + nky) nst in
      let sums =
        Array.sub kr
          (nkc + 1 + nky + nst)
          (Array.length kr - nkc - 1 - nky - nst)
      in
      let nuc = Array.length (Loops.pick u u)
      and nuy = Array.length (Loops.pick uy uy) in
      let lr =
        if nuc + nuy = 0 then [||]
        else begin
          (* The tape's parameters: its unknown carries and captures, then the
             residuals; the linear step's: the carries, the stacked residuals,
             then the unknown captures and the residual captures. *)
          let nuk = Array.length (Loops.pick ucap ucap) in
          let nu = nuc + nuk in
          let res p =
            List.filter
              (fun j -> p kinds.(j))
              (List.init (Array.length kinds) Fun.id)
          in
          let stacked_r = res (fun k -> k = Stacked) in
          let caps_r = res (function Capture _ -> true | _ -> false) in
          let range a n = List.init n (fun j -> a + j) in
          let order =
            List.concat
              [
                range 0 nuc;
                List.map (fun j -> nu + j) stacked_r;
                range nuc nuk;
                List.map (fun j -> nu + j) caps_r;
              ]
          in
          let at j =
            match kinds.(j) with
            | Capture c -> kcaps.(c)
            | Stacked | Row _ -> invalid_arg "Rune: a stacked residual"
          in
          let length, trips =
            match rows with
            | Taken -> (count trips, None)
            | Max -> (it.max, Some trips)
          in
          record st i
            (Scan
               {
                 step = Loops.reorder tape (Array.of_list order);
                 length;
                 trips;
                 reverse = false;
                 init = Loops.pick u it.init;
                 xs = stacked;
                 captures =
                   Array.append (Loops.pick ucap captures)
                     (Array.of_list (List.map at caps_r));
                 seeds = [||];
               })
        end
      in
      let next a =
        let k = ref (-1) in
        fun () ->
          incr k;
          a.(!k)
      in
      let kn = next kr and un = next lr in
      let carry = Array.map (fun u -> if u then un () else kn ()) u in
      if Option.is_none it.rows then Array.append carry sums
      else
        let _ = kn () in
        let ys = Array.map (fun u -> if u then un () else kn ()) uy in
        Array.concat [ carry; [| trips |]; ys; sums ]
  | _ -> invalid_arg "Rune: an iterate's parts"

(* A switch over the tape's variables is two switches on its index, as JAX's
   partial evaluation of a cond gives: a known switch, computed at the call, of
   the results no branch computes from them and every branch's residuals; and a
   recorded linear switch of the others over the residuals. The branches'
   residuals share slots by form, only the taken branch's being live, and a
   branch leaves the slots it does not fill zeros. Each branch is split traced
   as the switch's are, every operation staged. *)
and switch_parts st i (c : construct) =
  match c with
  | Switch sw ->
      let name = st.name in
      let branches = sw.branches in
      let ucap = Array.map (owned i) sw.captures in
      let cforms = Array.map form sw.captures in
      let kforms = Loops.pick (nots ucap) cforms
      and uforms = Loops.pick ucap cforms in
      let no = Array.length (switched branches) in
      (* Branch [b] split, the results [unknown] says unknown so: its known
         program over the known captures, to its known results, its residuals
         and its running totals out, and its own captures; its linear program
         over the unknown captures then its residuals; which results it leaves
         unknown; and how many residuals it has. With [fill = (at, slots)],
         the known program's residuals are the [slots] instead, residual [r]
         at slot [at.(r)], each slot it leaves zeros made in its trace. *)
      let split ?fill (b : Program.t) ~unknown =
        let lin = ref None and uy = ref [||] and nr = ref 0 in
        let known kvs =
          let kn = ref [||] in
          let p, res =
            trace_in Tape ~name uforms (fun inner uvs ->
                let args = Loops.merge ucap uvs kvs in
                let out = threaded b (fun totals -> eval ~totals b args) in
                let y = Array.sub out 0 no in
                uy := Array.mapi (fun j (V v) -> unknown.(j) || owns inner v) y;
                kn := Loops.pick (nots !uy) y;
                Loops.pick !uy y)
          in
          lin := Some p;
          nr := Array.length res;
          match fill with
          | None -> Array.append !kn res
          | Some (at, slots) ->
              let filled = Array.make (Array.length slots) None in
              Array.iteri (fun r j -> filled.(j) <- Some res.(r)) at;
              Array.append !kn
                (Array.mapi
                   (fun j v ->
                     match v with
                     | Some v -> v
                     | None -> Loops.filled ~zero:true slots.(j))
                   filled)
        in
        let kprog, own = trace Staged ~name kforms known in
        (kprog, own, Option.get !lin, !uy, !nr)
      in
      let none = Array.make no false in
      let unknown =
        Array.fold_left
          (fun acc b ->
            let _, _, _, uy, _ = split b ~unknown:none in
            Array.map2 ( || ) acc uy)
          none branches
      in
      let parts = Array.map (fun b -> split b ~unknown) branches in
      let nk = Array.length (Loops.pick (nots unknown) unknown) in
      (* The residual slots: each a form, a branch's residual taking the first
         slot of its form it has not filled. *)
      let slots = ref [] in
      let same_form (F a) (F b) =
        Nx.Dtype.equal a.dtype b.dtype
        && Nx_array.Layout.shape a.layout = Nx_array.Layout.shape b.layout
        && Nx.Placement.equal a.placement b.placement
      in
      let assign (kprog, _, _, _, nr) =
        let used = ref [] in
        Array.init nr (fun r ->
            let f = form kprog.outputs.(nk + r) in
            let rec go j = function
              | [] ->
                  slots := !slots @ [ f ];
                  j
              | g :: rest ->
                  if same_form f g && not (List.mem j !used) then j
                  else go (j + 1) rest
            in
            let j = go 0 !slots in
            used := j :: !used;
            j)
      in
      let at = Array.map assign parts in
      let slots = Array.of_list !slots in
      let nslot = Array.length slots in
      (* Each known program to its known results, the slots, then its running
         totals out; each linear one over the unknown captures then the
         slots. *)
      let parts =
        Array.mapi (fun k b -> split ~fill:(at.(k), slots) b ~unknown) branches
      in
      let known = Array.map (fun (kprog, own, _, _, _) -> (kprog, own)) parts in
      let kprogs, kown = unite known ~params:(Array.length kforms) in
      let kcaps = Loops.pick (nots ucap) sw.captures in
      let kr =
        perform
          (Switch
             { sw with branches = kprogs; captures = Array.append kcaps kown })
      in
      let ky = Array.sub kr 0 nk and sv = Array.sub kr nk nslot in
      let sums = Array.sub kr (nk + nslot) (Array.length kr - nk - nslot) in
      let nu = Array.length uforms in
      let results =
        if nk = no then ky
        else
          let lins =
            Array.mapi
              (fun k (_, _, lin, _, _) ->
                widen lin
                  ~at:
                    (Array.append (Array.init nu Fun.id)
                       (Array.map (fun j -> nu + j) at.(k)))
                  (nu + nslot))
              parts
          in
          let lr =
            record st i
              (Switch
                 {
                   name = sw.name;
                   index = sw.index;
                   branches = lins;
                   captures = Array.append (Loops.pick ucap sw.captures) sv;
                   seeds = [||];
                 })
          in
          Loops.merge unknown lr ky
      in
      Array.append results sums
  | _ -> invalid_arg "Rune: a switch's parts"

(* A remat over the tape's variables computes at the call the outputs that read
   none of them, its function partially evaluated once, and records itself over
   its arguments: transposing it evaluates its function again, as JAX's
   checkpoint keeps its inputs as residuals. The record of that first evaluation
   is dropped, so the tape keeps the arguments alone. *)
and remat_parts st i (body : Program.t) args ~seeds =
  let ua = Array.map (owned i) args in
  let known = ref [||] in
  let _ =
    trace_in Tape ~name:st.name
      (Array.map form (Loops.pick ua args))
      (fun inner uvs ->
        let out =
          eval_seeded body ~seeds
            (Loops.merge ua uvs (Loops.pick (nots ua) args))
        in
        known :=
          Array.map
            (fun (V y as v) -> if owns inner y then None else Some v)
            out;
        [||])
  in
  if Array.for_all Option.is_some !known then Array.map Option.get !known
  else
    let r = record st i (Remat { body; args; seeds }) in
    Array.mapi (fun k v -> match v with Some v -> v | None -> r.(k)) !known

(* A call over the tape's variables is two calls, as JAX's partial evaluation
   of a compiled call gives: a known call, computed at the call, of the results
   that read none of them, which returns the residuals the rest reads as more
   results and threads the call's totals; and a recorded call of the rest,
   linear in the tape's variables, over them and the residuals. Both are
   derived once per derivation at the call's site and compiled. *)
and call_parts st i (body : Program.t) args site ~seeds =
  let name = st.name in
  let ua = Array.map (owned i) args in
  let kargs = Loops.pick (nots ua) args and uargs = Loops.pick ua args in
  let made () =
    let unknown = ref [||] and linear = ref None in
    let known kvs =
      let kys = ref [||] in
      let tape, res =
        trace_in Tape ~name (Array.map form uargs) (fun inner uvs ->
            (* The totals' running values are the known trace's: a tangent
               added to one raises as it is handed back. *)
            let ys =
              threaded body (fun totals ->
                  eval ~totals body (Loops.merge ua uvs kvs))
            in
            let mine (V y) = owns inner y in
            unknown := Array.map mine ys;
            kys := Loops.pick (nots !unknown) ys;
            Loops.pick !unknown ys)
      in
      linear :=
        Some (tape, Array.append (Array.map form uargs) (Array.map form res));
      Array.append !kys res
    in
    let kprog, _ = trace Jit ~name (Array.map form kargs) known in
    ((kprog, Array.map form kargs), !unknown, !linear)
  in
  let d =
    Program.derive site (Known { unknown = ua }) (Array.map form args) made
  in
  let kr =
    perform (Call { body = d.program; args = kargs; site = d.at; seeds })
  in
  let nk = Array.length (Loops.pick (nots d.flags) d.flags) in
  let ns = Array.length d.program.adds in
  let nres = Array.length kr - nk - ns in
  let res = Array.sub kr nk nres and sums = Array.sub kr (nk + nres) ns in
  let ur =
    match d.linear with
    | Some (l, at) when Array.exists Fun.id d.flags ->
        let args = Array.append uargs res in
        record st i (Call { body = l; args; site = at; seeds = [||] })
    | _ -> [||]
  in
  Array.append (Loops.merge d.flags ur (Array.sub kr 0 nk)) sums

and trace ?totals kind ~name inputs f =
  trace_in ?totals kind ~name inputs (fun _ vs -> f vs)

and trace_in ?(totals = [||]) kind ~name inputs f =
  let st =
    {
      kind;
      name;
      next = Array.length inputs;
      eqns = [];
      captures = [];
      threads = [];
      hints = totals;
    }
  in
  let owner = ref None in
  let body i =
    owner := Some i;
    let params =
      Array.mapi (fun k (F form) -> V (Nx.Prim.traced i form (Var k))) inputs
    in
    let outs = f i params in
    Array.map (fun (V y) -> V (operand st i y)) outs
  in
  (* The handler lies outside the interpretation, as its rule does: what it
     computes goes to the interpretations around the tracer. *)
  let effc (type a) (e : a Effect.t) :
      ((a, _) Effect.Deep.continuation -> _) option =
    match (e, !owner) with
    | Construct (Custom _ as c), Some i
      when st.kind = Tape && List.exists (owned i) (operands c) ->
        (* A custom rule over a tape's variables, as a tangent map that applies
           one to its tangents: its primal runs where it was performed, and its
           operations are recorded. *)
        Some (fun k -> Effect.Deep.discontinue k Evaluate)
    | Construct (Call _ as c), Some i when claims st i c && st.kind = Jit ->
        (* Inlined: the default runs where the call was performed, which issues
           the call's equations here. *)
        Some (fun k -> Effect.Deep.discontinue k Evaluate)
    | Construct c, Some i when claims st i c ->
        Some (fun k -> Effect.Deep.continue k (answer st i c))
    | Construct (Linear { apply = None; forms; _ }), Some _ when st.kind = Tape
      ->
        (* A pullback's tangents over none of the tape's variables: linear in
           its tangents, which are known, so zeros. A loop's step derived under
           reverse mode makes them where a carry's tangent is a zero it
           instantiated. *)
        Some
          (fun k ->
            Effect.Deep.continue k (Array.map (Loops.filled ~zero:false) forms))
    | Staging, Some _ when st.kind = Jit || st.kind = Staged ->
        Some (fun k -> Effect.Deep.continue k true)
    (* A traced program threads the totals its extent adds to. *)
    | Running (t, f), Some i when st.kind <> Tape ->
        Some (fun k -> Effect.Deep.continue k (Some (thread st i t f)))
    (* A trace holds a total's running value where one around it lies, at its
       own where it threads one; a jit's program, kept for later calls, holds it
       where none does. *)
    | Where t, Some _ when st.kind <> Tape ->
        Some
          (fun k ->
            Effect.Deep.continue k
              (match Program.where t with
              | Nowhere when st.kind = Jit -> At None
              | Nowhere -> Nowhere
              | At _ as a ->
                  if held st t then At (Some (placement_of st t)) else a))
    | Replace (t, x), Some i when st.kind <> Tape ->
        Some
          (fun k ->
            replace st i t x;
            Effect.Deep.continue k ())
    | Replace (_, x), Some i when st.kind = Tape && owns i x ->
        Some
          (fun k ->
            Effect.Deep.discontinue k
              (Invalid_argument (st.name ^ ": a tangent is added to a total")))
    | _ -> None
  in
  let reach =
    match kind with Body | Tape -> Nx.Prim.Values | Jit | Staged -> Extent
  in
  let traced () =
    Effect.Deep.match_with
      (fun () ->
        Nx.Prim.interpret ~name reach (fun i ~by op -> rule st i ~by op) body)
      ()
      { retc = Fun.id; exnc = raise; effc }
  in
  (* A tape's primal computes at the call, so its draws are the scope's. *)
  let outs =
    match kind with
    | Tape -> traced ()
    | Body | Jit | Staged -> Nx.Rng.refusing name traced
  in
  let owner = Option.get !owner in
  let captures = List.rev st.captures and threads = List.rev st.threads in
  let params =
    Array.concat
      [
        Array.init (Array.length inputs) Fun.id;
        Array.of_list (List.map snd captures);
        Array.of_list (List.map (fun (Thread th) -> th.var) threads);
      ]
  in
  let outs =
    Array.append outs
      (Array.of_list (List.map (fun (Thread th) -> V th.now) threads))
  in
  let adds =
    Array.of_list
      (List.map
         (fun (Thread th) ->
           let out = Nx.Prim.form th.now in
           Addition
             ( th.total,
               th.into,
               {
                 out with
                 layout =
                   Nx_array.Layout.contiguous (Nx_array.Layout.shape out.layout);
               } ))
         threads)
  in
  let program =
    Program.make ~owner ~params (List.rev st.eqns) ~outputs:outs ~adds
      ~slots:st.next
  in
  (program, Array.of_list (List.map fst captures))
