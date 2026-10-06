(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Nx.Op
module Repr = Nx.Repr

let one x = [ Nx.P x ]

let results : type r. r Nx.Op.t -> r -> Nx.packed list =
 fun op r ->
  match[@warning "@4@8"] op with
  | Unary _ -> one r
  | Binary _ -> one r
  | Compare _ -> one r
  | Where _ -> one r
  | Fma _ -> one r
  | Reduce _ -> one r
  | Scan _ -> one r
  | Arg_reduce _ -> one r
  | Sort _ -> one r
  | Argsort _ -> one r
  | Group _ -> one r
  | Pad _ -> one r
  | Cat _ -> one r
  | Convert _ -> one r
  | Threefry _ -> one r
  | Gather _ -> one r
  | Scatter _ -> one r
  | Update _ -> one r
  | Unfold _ -> one r
  | Fold _ -> one r
  | Matmul _ -> one r
  | Fft _ -> one r
  | Rfft _ -> one r
  | Irfft _ -> one r
  | Contiguous _ -> one r
  | Cholesky _ -> one r
  | Solve_triangular _ -> one r
  | Move _ -> one r
  | Place _ -> one r
  | Qr _ ->
      let a, b = r in
      [ Nx.P a; Nx.P b ]
  | Lu _ ->
      let a, b, c = r in
      [ Nx.P a; Nx.P b; Nx.P c ]
  | Svd _ ->
      let a, b, c = r in
      [ Nx.P a; Nx.P b; Nx.P c ]
  | Eig _ ->
      let a, b = r in
      Nx.P a :: Option.to_list (Option.map (fun b -> Nx.P b) b)
  | Eigh _ ->
      let a, b = r in
      Nx.P a :: Option.to_list (Option.map (fun b -> Nx.P b) b)
  | Read _ | Check _ -> []

(* A record names each value of its run by its position: the inputs first, then
   the results of its entries in order. A value of the run is found by its
   identity, which a traced or a placed value carries; a host value is held
   weakly and compared physically. *)
type t = {
  inputs : int;
  mutable size : int;
  mutable entries : entry list;  (** The last first. *)
  traced : (int, int) Hashtbl.t;
  placed : (int, int) Hashtbl.t;
  mutable hosts : (Obj.t Weak.t * int) list;
  placeholders : (int, Nx.packed) Hashtbl.t;
  mutable outputs : Nx.packed list;  (** The function's results, as operands. *)
  mutable current : Nx.packed option array option;
      (** The values of the replay that runs. *)
  mutable answering : int;
      (** How many constructs the recorder answers: what their answers run is
          their entry. *)
  mutable passing : int;
      (** How many operations and constructs of an answer's own code pass the
          recorder: an answer marks them as they leave it, so that a handler
          around the call, which runs outside the answer while it runs, is
          recorded. *)
  mutable operator : (Nx.packed list -> Nx.packed list) option;
      (** The operator a replay of a [linear_solve]'s record applies. *)
}

and entry =
  | Op : { op : 'r Nx.Op.t; names : int list } -> entry
  | First : { c : ('a, 'b) Nx.t Construct.t; name : int } -> entry
      (** A collective or a detach, answered around the run. *)
  | Loop : {
      carry : Nx.packed list;
      trips : trips;
      step : t;  (** The step's record, over its carry, its row and its trip. *)
      rows : Nx.packed list;  (** The outputs of a replay that takes no step. *)
      names : int list;
    }
      -> entry
  | Remat : {
      p : 'p Nx.Ptree.t;
      q : 'q Nx.Ptree.t;
      args : 'p;
      like : 'q;
      f : t;
      recomputed : bool;
      names : int list;
    }
      -> entry
  | Custom : {
      q : 'q Nx.Ptree.t;
      like : 'q;
      c : 'q rule;
      names : int list;
    }
      -> entry
  | Root : {
      x : 'x Nx.Ptree.t;
      like : 'x;
      solve : t;
      residual : t option;  (** [None] when no derivative ran it. *)
      linear_solve : t option;
      names : int list;
    }
      -> entry
  | Apply : { args : Nx.packed list; names : int list } -> entry
      (** An application of the operator a [linear_solve] receives. *)

and trips =
  | Rows of { xs : Nx.packed list; reverse : bool }
  | Until of { until : t; max : int; failure : int array -> string }

and 'q rule =
  | Jvp_rule : {
      p : 'p Nx.Ptree.t;
      args : 'p;
      rule : t option;  (** [None] when the rule's value came with it. *)
      map : t option;  (** [None] when no derivative applied the map. *)
      value : 'q option;
    }
      -> 'q rule
  | Vjp_rule : {
      p : 'p Nx.Ptree.t;
      args : 'p;
      rule : t;
      pullback : 'q -> 'p;
          (** The last run's pullback, a function of cotangents. *)
    }
      -> 'q rule

type (_, _) Repr.node +=
  | Name : { record : t; index : int } -> ('a, 'b) Repr.node

let create inputs =
  {
    inputs;
    size = 0;
    entries = [];
    traced = Hashtbl.create 64;
    placed = Hashtbl.create 8;
    hosts = [];
    placeholders = Hashtbl.create 16;
    outputs = [];
    current = None;
    answering = 0;
    passing = 0;
    operator = None;
  }

let add r (Nx.P x) =
  let i = r.size in
  r.size <- i + 1;
  (match Repr.v x with
  | Traced t -> Hashtbl.replace r.traced (Repr.Traced.id t) i
  | Placed p -> Hashtbl.replace r.placed (Repr.Placed.id p) i
  | Host _ ->
      let w = Weak.create 1 in
      Weak.set w 0 (Some (Obj.repr x));
      r.hosts <- (w, i) :: r.hosts);
  i

let name_of (type a b) r (x : (a, b) Nx.t) =
  match Repr.v x with
  | Traced t -> (
      match Repr.Traced.node t with
      | Name { record; index } when record == r -> Some index
      | _ -> Hashtbl.find_opt r.traced (Repr.Traced.id t))
  | Placed p -> Hashtbl.find_opt r.placed (Repr.Placed.id p)
  | Host _ ->
      List.find_map
        (fun (w, i) ->
          match Weak.get w 0 with
          | Some v when v == Obj.repr x -> Some i
          | Some _ | None -> None)
        r.hosts

let placeholder (type a b) r i (x : (a, b) Nx.t) : (a, b) Nx.t =
  match Hashtbl.find_opt r.placeholders i with
  | Some p -> Nx.unpack (Nx.dtype x) p
  | None ->
      let p =
        Repr.Traced.v ~context:(Repr.context x) ~view:(Repr.view x)
          (Nx.placement x) (Nx.dtype x) (Nx.shape x)
          (Name { record = r; index = i })
      in
      Hashtbl.add r.placeholders i (Nx.P p);
      p

let rename r : mapper =
  {
    f =
      (fun x ->
        match name_of r x with Some i -> placeholder r i x | None -> x);
  }

let leaves (m : mapper) = List.map (fun (Nx.P x) -> Nx.P (m.f x))
let flat p v = fst (Nx.Ptree.flatten p v)
let mapped p (m : mapper) v = Nx.Ptree.map p (fun _ x -> m.f x) v
let names r l = List.map (add r) l

let first : type a b.
    mapper -> (a, b) Nx.t Construct.t -> (a, b) Nx.t Construct.t =
 fun m c ->
  match[@warning "@4@8"] c with
  | Lanes (axis, x) -> Lanes (axis, m.f x)
  | Detach x -> Detach (m.f x)
  | Lane_index _ -> c
  | Loop _ | Compiled _ | Remat _ | Barrier _ | Custom _ | Root _ | At_map _
  | Lane_count _ | Add _ ->
      assert false (* A first-order construct of tensor result. *)

(* [enclose r s] names in [r] the values of [r]'s run that [s], the record of a
   function a construct of [r]'s run carries, reads from outside: [s] holds
   names, never values of the run around it. *)
let rec enclose r s =
  let m = rename r in
  let sub = Option.iter (enclose r) in
  let entry = function
    | Op { op; names } -> Op { op = map_operands m op; names }
    | First { c; name } -> First { c = first m c; name }
    | Loop l ->
        enclose r l.step;
        let trips =
          match l.trips with
          | Rows rows -> Rows { rows with xs = leaves m rows.xs }
          | Until u ->
              enclose r u.until;
              l.trips
        in
        Loop { l with carry = leaves m l.carry; trips }
    | Remat e ->
        enclose r e.f;
        Remat { e with args = mapped e.p m e.args }
    | Custom ({ q; c = Jvp_rule j; _ } as e) ->
        sub j.rule;
        sub j.map;
        let value = Option.map (mapped q m) j.value in
        Custom
          { e with c = Jvp_rule { j with args = mapped j.p m j.args; value } }
    | Custom ({ c = Vjp_rule v; _ } as e) ->
        enclose r v.rule;
        Custom { e with c = Vjp_rule { v with args = mapped v.p m v.args } }
    | Root e ->
        enclose r e.solve;
        sub e.residual;
        sub e.linear_solve;
        Root e
    | Apply { args; names } -> Apply { args = leaves m args; names }
  in
  s.entries <- List.map entry s.entries;
  s.outputs <- leaves m s.outputs

(* Recording *)

(* [answered r f] is [f ()], an answer of [r]'s: [r] records none of what it
   runs, which the answer's entry covers. The operations and constructs [f]
   performs leave it through an installation that marks them; a handler around
   the call that runs while [f] does, such as a key scope computing a key, runs
   outside [f] and is recorded. *)
let answered r f =
  let through g =
    r.passing <- r.passing + 1;
    Fun.protect ~finally:(fun () -> r.passing <- r.passing - 1) g
  in
  let run : type o. o Nx.Op.t -> o = fun op -> through (fun () -> eval op) in
  let call : type c. c Construct.t -> c Construct.answer option =
   fun c ->
    match[@warning "@4@8"] c with
    | Lanes _ | Lane_index _ | Detach _ ->
        Some
          (Construct.value (fun () -> through (fun () -> Construct.perform c)))
    | Loop _ | Compiled _ | Remat _ | Barrier _ | Custom _ | Root _ | At_map _
    | Lane_count _ | Add _ ->
        None
  in
  r.answering <- r.answering + 1;
  Fun.protect ~finally:(fun () -> r.answering <- r.answering - 1) @@ fun () ->
  Construct.install { op = Some { run; claims = (fun _ -> true) }; call } f

let rec run : type a. Nx.packed list -> (unit -> a) -> a * t =
 fun inputs f -> running inputs (fun _ -> f ())

and running : type a. Nx.packed list -> (t -> a) -> a * t =
 fun inputs f ->
  let r = create (List.length inputs) in
  List.iter (fun x -> ignore (add r x)) inputs;
  let y = recording r (fun () -> f r) in
  (y, r)

(* [kept cell inputs f outs] is [f ()] run recorded at [inputs], its record,
   whose outputs are [outs] of its result, kept in [cell]: every run of a
   function is the same program, so the last one's record serves every
   replay. *)
and kept : type a.
    t option ref ->
    Nx.packed list ->
    (unit -> a) ->
    (a -> Nx.packed list) ->
    a * t =
 fun cell inputs f outs ->
  let y, s = run inputs f in
  s.outputs <- leaves (rename s) (outs y);
  cell := Some s;
  (y, s)

(* [recording r f] is [f ()] with its operations and the constructs that leave
   it recorded in [r]. A construct that carries a function is answered: it is
   performed with each function recorded each time it runs, and its entry holds
   the records. A compiled call runs inline. *)
and recording : type a. t -> (unit -> a) -> a =
 fun r f ->
  let run : type o. o Nx.Op.t -> o =
   fun op ->
    if r.passing > 0 then eval op
    else
      let named = map_operands (rename r) op in
      let y = eval op in
      r.entries <-
        Op { op = named; names = names r (results op y) } :: r.entries;
      y
  in
  let entry : type a b. (a, b) Nx.t Construct.t -> (a, b) Nx.t Construct.answer
      =
   fun c ->
    let named = first (rename r) c in
    Construct.value (fun () ->
        let y = Construct.perform c in
        if r.passing = 0 then
          r.entries <- First { c = named; name = add r (Nx.P y) } :: r.entries;
        y)
  in
  let here = Construct.here in
  let call : type c. c Construct.t -> c Construct.answer option =
   fun c ->
    match[@warning "@4@8"] c with
    | Lanes _ -> Some (entry c)
    | Lane_index _ -> Some (entry c)
    | Detach _ -> Some (entry c)
    | Loop q -> Some (here (fun () -> loop r q))
    | Remat { p; q; f; args; recomputed } ->
        Some (here (fun () -> remat r p q f args recomputed))
    | Custom rule -> Some (here (fun () -> custom r rule))
    | Root { x; residual; solve; linear_solve } ->
        Some (here (fun () -> root r x residual solve linear_solve))
    | Compiled { f; args; _ } ->
        Some (here (fun () -> recording r (fun () -> f args)))
    | Barrier _ | At_map _ | Lane_count _ | Add _ -> None
  in
  Construct.install { op = Some { run; claims = (fun _ -> true) }; call } f

and loop r (q : Trips.request) : Trips.result =
  let step = ref None and stop = ref None in
  (* The step's record takes its trip as an input, so that a replay at trip [k]
     draws as trip [k] drew. *)
  let req_step trip c x =
    fst
      (kept step
         (c @ x @ [ Nx.P trip ])
         (fun () -> q.req_step trip c x)
         (fun (c', y) -> c' @ y))
  in
  let req_trips : Trips.trips =
    match q.req_trips with
    | Rows _ as rows -> rows
    | Until u ->
        let until c =
          fst (kept stop c (fun () -> u.until c) (fun v -> [ Nx.P v ]))
        in
        Until { u with until }
  in
  let result =
    answered r (fun () ->
        Construct.perform (Loop { q with req_step; req_trips }))
  in
  (* A loop that took no step runs it once at its final carry, as its first
     trip, its additions dropped, so that its record replays at other inputs,
     where its outputs have rows. A step that draws computes the loop's key from
     the place the loop took at its call, and takes none. *)
  let result =
    match !step with
    | Some _ -> result
    | None ->
        let _, ys =
          answered r (fun () ->
              Total.discarding (fun () ->
                  req_step (Nx.scalar Nx.int32 0l) result.r_carry []))
        in
        { result with r_ys = Trips.no_rows ys }
  in
  let m = rename r in
  let trips =
    match (q.req_trips, !stop) with
    | Rows { xs; reverse }, _ -> Rows { xs = leaves m xs; reverse }
    | Until { max; failure; _ }, Some until ->
        enclose r until;
        Until { until; max; failure }
    | Until _, None -> assert false (* A stop is tested before any step. *)
  in
  let carry = leaves m q.req_carry in
  let names = names r (result.r_carry @ result.r_ys) in
  let step = Option.get !step in
  enclose r step;
  (* The step's outputs are names, whose metadata alone [no_rows] reads. *)
  let _, ys = Trips.split (List.length q.req_carry) step.outputs in
  r.entries <-
    Loop { carry; trips; step; rows = Trips.no_rows ys; names } :: r.entries;
  result

and remat : type p q.
    t -> p Nx.Ptree.t -> q Nx.Ptree.t -> (p -> q) -> p -> bool -> q =
 fun r p q f args recomputed ->
  let cell = ref None in
  let f a = fst (kept cell (flat p a) (fun () -> f a) (flat q)) in
  let y =
    answered r (fun () ->
        Construct.perform (Remat { p; q; f; args; recomputed }))
  in
  let f =
    Option.get !cell
    (* Every answer runs the function. *)
  in
  enclose r f;
  let m = rename r in
  let args = mapped p m args and names = names r (flat q y) in
  r.entries <-
    Remat { p; q; args; like = mapped q m y; f; recomputed; names } :: r.entries;
  y

and custom : type q. t -> q Construct.rule -> q =
 fun r c ->
  let m = rename r in
  match c with
  | Jvp_rule { p; q; rule; args; value } ->
      let ruled = ref None and mapped_ = ref None in
      let rule a =
        let (y, map), s =
          kept ruled (flat p a) (fun () -> rule a) (fun (y, _) -> flat q y)
        in
        let map da =
          let dy, ms = kept mapped_ (flat p da) (fun () -> map da) (flat q) in
          enclose s ms;
          dy
        in
        (y, map)
      in
      let y =
        answered r (fun () ->
            Construct.perform (Custom (Jvp_rule { p; q; rule; args; value })))
      in
      Option.iter (enclose r) !ruled;
      Option.iter (enclose r) !mapped_;
      let args = mapped p m args and value = Option.map (mapped q m) value in
      let names = names r (flat q y) in
      let c = Jvp_rule { p; args; rule = !ruled; map = !mapped_; value } in
      r.entries <- Custom { q; like = mapped q m y; c; names } :: r.entries;
      y
  | Vjp_rule { p; q; rule; args } ->
      let ruled = ref None and pulled = ref None in
      let rule a =
        let (y, pullback), _ =
          kept ruled (flat p a) (fun () -> rule a) (fun (y, _) -> flat q y)
        in
        pulled := Some pullback;
        (y, pullback)
      in
      let y =
        answered r (fun () ->
            Construct.perform (Custom (Vjp_rule { p; q; rule; args })))
      in
      let ruled =
        Option.get !ruled
        (* Every answer runs the rule. *)
      in
      enclose r ruled;
      let names = names r (flat q y) in
      let c =
        Vjp_rule
          {
            p;
            args = mapped p m args;
            rule = ruled;
            pullback = Option.get !pulled;
          }
      in
      r.entries <- Custom { q; like = mapped q m y; c; names } :: r.entries;
      y

and root : type x.
    t -> x Nx.Ptree.t -> (x -> x) -> (unit -> x) -> ((x -> x) -> x -> x) -> x =
 fun r x residual solve linear_solve ->
  let solved = ref None and residuals = ref None and solves = ref None in
  let solve () = fst (kept solved [] solve (flat x)) in
  let residual v =
    fst (kept residuals (flat x v) (fun () -> residual v) (flat x))
  in
  let linear_solve op b =
    let y, s =
      running (flat x b) (fun s -> linear_solve (fun v -> apply s x op v) b)
    in
    s.outputs <- leaves (rename s) (flat x y);
    solves := Some s;
    y
  in
  let y =
    answered r (fun () ->
        Construct.perform (Root { x; residual; solve; linear_solve }))
  in
  let solve =
    Option.get !solved
    (* Every answer runs the solve. *)
  in
  enclose r solve;
  Option.iter (enclose r) !residuals;
  Option.iter (enclose r) !solves;
  let names = names r (flat x y) in
  r.entries <-
    Root
      {
        x;
        like = mapped x (rename r) y;
        solve;
        residual = !residuals;
        linear_solve = !solves;
        names;
      }
    :: r.entries;
  y

(* [apply s x op v] is [op v], recorded in [s] as an application of the operator
   its function receives, which a replay replaces. *)
and apply : type x. t -> x Nx.Ptree.t -> (x -> x) -> x -> x =
 fun s x op v ->
  if s.answering > 0 then op v
  else
    let args = leaves (rename s) (flat x v) in
    let y = answered s (fun () -> op v) in
    s.entries <- Apply { args; names = names s (flat x y) } :: s.entries;
    y

(* Replaying *)

let resolve : type a b. (a, b) Nx.t -> (a, b) Nx.t =
 fun x ->
  match Repr.v x with
  | Traced t -> (
      match Repr.Traced.node t with
      | Name { record; index } -> (
          match record.current with
          | Some values -> (
              match values.(index) with
              | Some v -> Nx.unpack (Nx.dtype x) v
              | None -> assert false (* An entry names a value made before. *))
          | None -> assert false (* A name is read inside its replay. *))
      | _ -> x)
  | Host _ | Placed _ -> x

let resolver = { f = resolve }

(* The replays that run, innermost first, on this domain: a pullback a replay
   gives reads the values of each. *)
let active = Domain.DLS.new_key (fun () -> ref [])

let with_current r values f =
  let saved = r.current and stack = Domain.DLS.get active in
  let outer = !stack in
  r.current <- Some values;
  stack := (r, values) :: outer;
  Fun.protect
    ~finally:(fun () ->
      r.current <- saved;
      stack := outer)
    f

(* [substitution (r, values)] is the replay of [r] at [values] as a
   substitution: a name of [r], or a value of [r]'s run, by its replayed
   value. *)
let substitution (r, values) f =
  let owner = { Construct.owns = (fun x -> Option.is_some (name_of r x)) } in
  let s : mapper =
    {
      f =
        (fun x ->
          match name_of r x with
          | Some i -> (
              match values.(i) with
              | Some v -> Nx.unpack (Nx.dtype x) v
              | None -> assert false (* Every name has its value. *))
          | None -> x);
    }
  in
  Construct.substituting owner s f

let missing () =
  invalid_arg
    "Rune: a construct's replay applies a function that no transformation ran \
     when it was recorded"

let rec evaluate r inputs =
  if List.compare_length_with inputs r.inputs <> 0 then
    invalid_arg "Record.replay: the inputs differ from the run's in number";
  let values = Array.make r.size None in
  List.iteri (fun i x -> values.(i) <- Some x) inputs;
  let set names made =
    List.iter2 (fun i v -> values.(i) <- Some v) names made
  in
  let perform = function
    | Op { op; names } ->
        let op = map_operands resolver op in
        set names (results op (eval op))
    | First { c; name } ->
        values.(name) <- Some (Nx.P (Construct.perform (first resolver c)))
    | Loop { carry; trips; step; rows; names } ->
        let n = List.length carry in
        let req_trips : Trips.trips =
          match trips with
          | Rows { xs; reverse } -> Rows { xs = leaves resolver xs; reverse }
          | Until { until; max; failure } ->
              let until c = Nx.unpack Nx.bool (List.hd (outputs until c)) in
              Until { until; max; failure }
        in
        let req_step trip c x =
          Trips.split n (outputs step (c @ x @ [ Nx.P trip ]))
        in
        let result =
          Construct.loop
            { req_carry = leaves resolver carry; req_trips; req_step }
        in
        let ys = match result.r_ys with [] -> rows | ys -> ys in
        set names (result.r_carry @ ys)
    | Remat { p; q; args; like; f; recomputed; names } ->
        let f a = Nx.Ptree.rebuild q ~like (outputs f (flat p a)) in
        let args = mapped p resolver args in
        set names
          (flat q (Construct.perform (Remat { p; q; f; args; recomputed })))
    | Custom { q; like; c; names } ->
        let rebuild l = Nx.Ptree.rebuild q ~like l in
        let c : _ Construct.rule =
          match c with
          | Jvp_rule { p; args; rule; map; value } ->
              let rule a =
                let rs = match rule with Some rs -> rs | None -> missing () in
                let values = evaluate rs (flat p a) in
                let y =
                  with_current rs values (fun () ->
                      rebuild (leaves resolver rs.outputs))
                in
                let map da =
                  let ms =
                    match map with Some ms -> ms | None -> missing ()
                  in
                  with_current rs values (fun () ->
                      rebuild (outputs ms (flat p da)))
                in
                (y, map)
              in
              let value = Option.map (mapped q resolver) value in
              Jvp_rule { p; q; rule; args = mapped p resolver args; value }
          | Vjp_rule { p; args; rule = rs; pullback } ->
              let rule a =
                let values = evaluate rs (flat p a) in
                let y =
                  with_current rs values (fun () ->
                      rebuild (leaves resolver rs.outputs))
                in
                (* The pullback reads the values of the rule's run and of the
                   runs around it, which the replays give. *)
                let replays = (rs, values) :: !(Domain.DLS.get active) in
                let pullback ct =
                  List.fold_left
                    (fun f replay () -> substitution replay f)
                    (fun () -> pullback ct)
                    replays ()
                in
                (y, pullback)
              in
              Vjp_rule { p; q; rule; args = mapped p resolver args }
        in
        set names (flat q (Construct.perform (Custom c)))
    | Root { x; like; solve; residual; linear_solve; names } ->
        let rebuild l = Nx.Ptree.rebuild x ~like l in
        let solve () = rebuild (outputs solve []) in
        let residual v =
          match residual with
          | Some rs -> rebuild (outputs rs (flat x v))
          | None -> missing ()
        in
        let linear_solve op b =
          match linear_solve with
          | Some ls ->
              ls.operator <- Some (fun l -> flat x (op (rebuild l)));
              Fun.protect ~finally:(fun () -> ls.operator <- None) @@ fun () ->
              rebuild (outputs ls (flat x b))
          | None -> missing ()
        in
        set names
          (flat x
             (Construct.perform (Root { x; residual; solve; linear_solve })))
    | Apply { args; names } -> (
        match r.operator with
        | Some op -> set names (op (leaves resolver args))
        | None -> assert false (* A linear_solve's replay gives its operator. *)
        )
  in
  with_current r values (fun () -> List.iter perform (List.rev r.entries));
  values

and replay : type a. t -> Nx.packed list -> (unit -> a) -> a =
 fun r inputs f ->
  let values = evaluate r inputs in
  with_current r values (fun () -> substitution (r, values) f)

and outputs s inputs = replay s inputs (fun () -> leaves resolver s.outputs)
