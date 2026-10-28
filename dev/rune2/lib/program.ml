(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Nx.Dtype

type value = V : ('v, 's, 'd) Nx.t -> value
type form = F : ('v, 's, 'd) Nx.Prim.form -> form

let form (V x) = F (Nx.Prim.form x)

(* [v] at [dt] and [set]'s brand. A program binds each variable to a value its
   equation computed at the variable's type, but holds it as a [value], so the
   type is read back from the run-time facts nx checks: the dtype and the set. A
   value is read back at the dtype and set of the form its stand-in was traced
   at, so a failure is a binding at the wrong variable, or a trace at other
   forms than the evaluation's. *)
let cast (type v s d) (dt : (v, s) D.t) (set : d Nx.devices) (V y) :
    (v, s, d) Nx.t =
  match Nx.Prim.coerce dt set y with
  | Some y -> y
  | None when D.equal (Nx.dtype y) dt ->
      invalid_arg
        (Format.asprintf "Rune: a %a value on %a where one on %a was expected"
           Nx.pp_dtype dt Nx.Placement.pp (Nx.placement y) Nx.Placement.pp
           (Nx.Placement.on set))
  | None ->
      invalid_arg
        (Format.asprintf "Rune: a %a value where %a was expected" Nx.pp_dtype
           (Nx.dtype y) Nx.pp_dtype dt)

(* [x] at [f]'s type where it has [f]'s dtype and lies on its set, and [v] at
   [f]'s type, which raises where it does not. *)
let coerce_at (type v s d) (f : (v, s, d) Nx.Prim.form) x :
    (v, s, d) Nx.t option =
  Nx.Prim.coerce f.dtype (Nx.Placement.devices f.placement) x

let cast_at (type v s d) (f : (v, s, d) Nx.Prim.form) v : (v, s, d) Nx.t =
  cast f.dtype (Nx.Placement.devices f.placement) v

(* [v] at [x]'s type. *)
let like x v = cast (Nx.dtype x) (Nx.Placement.devices (Nx.placement x)) v

(* Whether [a] and [b] are one value. *)
let same (V a) (V b) =
  let set = Nx.Placement.devices (Nx.placement a) in
  match Nx.Prim.coerce (Nx.dtype a) set b with
  | Some b -> a == b
  | None -> false

(* Names *)

(* A map's name is a number no other name has. *)
type axis = int

let axes = Atomic.make 0
let axis () = Atomic.fetch_and_add axes 1
let same_axis = Int.equal

type ('v, 's, 'd) total = { id : ('v, 's, 'd) Nx.t Type.Id.t }

let total () = { id = Type.Id.make () }
let same_total t u = Type.Id.provably_equal t.id u.id
let total_id t = Type.Id.uid t.id

(* Derivations

   A site keeps a program per derivation, keyed by the derivation, the
   abstract values of the derived program's arguments, and the lanes of each
   map the site's program names, which its derived programs' forms depend
   on. *)

type derivation =
  | Jvp of { tangents : bool array; reverse : bool }
  | Vmap of { lanes : bool array; n : int; named : bool }
  | Known of { unknown : bool array }
  | Transpose of { linear : bool array; cts : bool array }

(* A form's abstract value: its dtype, shape and placement. *)
let same_abstract (F f) (F g) =
  D.equal f.dtype g.dtype
  && Nx_array.Layout.shape f.layout = Nx_array.Layout.shape g.layout
  && Nx.Placement.equal f.placement g.placement

let same_derivation a b =
  match (a, b) with
  | Jvp a, Jvp b -> a.tangents = b.tangents && a.reverse = b.reverse
  | Vmap a, Vmap b -> a.lanes = b.lanes && a.n = b.n && a.named = b.named
  | Known a, Known b -> a.unknown = b.unknown
  | Transpose a, Transpose b -> a.linear = b.linear && a.cts = b.cts
  | (Jvp _ | Vmap _ | Known _ | Transpose _), _ -> false

type key = { derivation : derivation; forms : form array; lanes : int list }

module Table = Cache.Make (struct
  type t = key

  let equal a b =
    same_derivation a.derivation b.derivation
    && Array.length a.forms = Array.length b.forms
    && Array.for_all2 same_abstract a.forms b.forms
    && a.lanes = b.lanes

  let hash k =
    let tag =
      match k.derivation with
      | Jvp { tangents; _ } -> Hashtbl.hash (0, tangents)
      | Vmap { lanes; n; _ } -> Hashtbl.hash (1, lanes, n)
      | Known { unknown } -> Hashtbl.hash (2, unknown)
      | Transpose { linear; cts; _ } -> Hashtbl.hash (3, linear, cts)
    in
    Array.fold_left
      (fun h (F f) ->
        (h * 31) + Hashtbl.hash (Nx_array.Layout.shape f.layout))
      tag k.forms
end)

(* Programs *)

type ('v, 's, 'd) Nx.Prim.payload += Var : int -> ('v, 's, 'd) Nx.Prim.payload

type addition =
  | Addition :
      ('v, 's, 'd) total * ('v, 's, 'd) Nx.Prim.form * ('v, 's, 'd) Nx.Prim.form
      -> addition

type t = {
  owner : Nx.Prim.interpretation;
  params : int array;
  eqns : eqn array;
  outputs : value array;
  adds : addition array;
  slots : int;
}

and eqn =
  | Op : { by : string; op : 'r Nx.Prim.t; base : int } -> eqn
  | Do : { c : construct; base : int } -> eqn

and construct =
  | Detach : ('v, 's, 'd) Nx.t -> construct
  | Lanes : axis * ('v, 's, 'd) Nx.t -> construct
  | Lane_index : axis * 'd Nx.Placement.t -> construct
  | Add : ('v, 's, 'd) total * ('v, 's, 'd) Nx.t -> construct
  | Scan : {
      step : t;
      length : int;
      trips : value option;
      reverse : bool;
      init : value array;
      xs : value array;
      captures : value array;
      seeds : value array;
    }
      -> construct
  | Iterate : {
      name : string;
      step : t;
      until : t;
      max : int;
      rows : rows option;
      stopped : int option;
      init : value array;
      step_captures : value array;
      until_captures : value array;
      seeds : value array;
    }
      -> construct
  | Call : {
      body : t;
      args : value array;
      site : site;
      seeds : value array;
    }
      -> construct
  | Custom : {
      name : string;
      primal : t;
      derivative : derivative;
      ny : int;
      args : value array;
      captures : value array;
      residuals : bool;
      applied : bool;
      seeds : value array;
    }
      -> construct
  | Linear : {
      name : string;
      apply : t option;
      transpose : t;
      args : value array;
      captures : value array;
      residuals : value array;
      forms : form array;
    }
      -> construct
  | Remat : { body : t; args : value array; seeds : value array } -> construct
  | Switch : switch -> construct
  | Operator : {
      id : int;
      transposed : bool;
      lanes : (int * bool array) list;
      args : value array;
      data : value array;
      forms : form array;
    }
      -> construct
  | Root : {
      name : string;
      system : system;
      solution : value array;
      theta : value array;
      captures : value array;
    }
      -> construct
  | Solve : {
      name : string;
      system : system;
      transposed : bool;
      b : value array;
      data : value array;
      captures : value array;
    }
      -> construct

and switch = {
  name : string;
  index : value;
  branches : t array;
  captures : value array;
  seeds : value array;
}

and system = {
  id : int;
  operator :
    transposed:bool -> lanes:(int * bool array) list -> t * value array;
  residual : t;
  jacobian : t;
  transpose : t;
  solve : t;
}

and derivative = Tangent of t | Pullback of t
and rows = Taken | Max

and site = {
  name : string;
  compile : t -> form array -> value array -> value array;
  compiled : (value array -> value array) option;
  axes : axis list;
  iterates : bool;
  customs : bool;
  table : derived Table.t;
}

and derived = {
  program : t;
  at : site;
  flags : bool array;
  linear : (t * site) option;
}

let make ~owner ~params eqns ~outputs ~adds ~slots =
  { owner; params; eqns = Array.of_list eqns; outputs; adds; slots }

type _ Effect.t += Construct : construct -> value array Effect.t

exception Evaluate

type _ Effect.t += Bound : axis -> int option Effect.t

let bound a =
  match Effect.perform (Bound a) with
  | n -> n
  | exception Effect.Unhandled (Bound _) -> None

type _ Effect.t += Staging : bool Effect.t

let staging () =
  match Effect.perform Staging with
  | b -> b
  | exception Effect.Unhandled Staging -> false

(* Sites *)

(* The programs [c] carries. *)
let carried = function
  | Scan { step; _ } -> [ step ]
  | Iterate { until; step; _ } -> [ until; step ]
  | Call { body; _ } | Remat { body; _ } -> [ body ]
  | Switch { branches; _ } -> Array.to_list branches
  | Custom { primal; derivative = Tangent d | Pullback d; _ } -> [ primal; d ]
  | Linear { apply; transpose; _ } -> Option.to_list apply @ [ transpose ]
  | Root { system; _ } | Solve { system; _ } ->
      [ system.residual; system.jacobian; system.transpose; system.solve ]
  | Detach _ | Lanes _ | Lane_index _ | Add _ | Operator _ -> []

(* The maps [p] names, and those the programs it carries name, in the order
   first named. *)
let named p =
  let found = ref [] in
  let see a = if not (List.mem a !found) then found := !found @ [ a ] in
  let rec go (p : t) =
    Array.iter
      (function
        | Op _ -> ()
        | Do { c; _ } ->
            (match c with Lanes (a, _) | Lane_index (a, _) -> see a | _ -> ());
            List.iter go (carried c))
      p.eqns
  in
  go p;
  !found

(* Whether [p] or a program it carries holds a construct [pred] holds of. *)
let rec holds pred (p : t) =
  Array.exists
    (function
      | Op _ -> false
      | Do { c; _ } -> pred c || List.exists (holds pred) (carried c))
    p.eqns

let iterate = function Iterate _ -> true | _ -> false
let custom = function Custom _ | Linear _ -> true | _ -> false

(* A site of [body] with the compiling of [s], no compiled form and no derived
   program. *)
let bare s body =
  {
    s with
    compiled = None;
    axes = named body;
    iterates = holds iterate body;
    customs = holds custom body;
    table = Table.create ();
  }

let site ~name ~compile p forms =
  {
    name;
    compile;
    compiled = Some (compile p forms);
    axes = named p;
    iterates = holds iterate p;
    customs = holds custom p;
    table = Table.create ();
  }

let axes s = s.axes
let iterates s = s.iterates
let customs s = s.customs
let made = Domain.DLS.new_key (fun () -> ref 0)
let derivations () = !(Domain.DLS.get made)

let derive ?map s derivation forms make =
  let lanes =
    List.map
      (fun a ->
        let n =
          match map with
          | Some (b, n) when same_axis a b -> Some n
          | _ -> bound a
        in
        match n with Some n -> n | None -> -1)
      s.axes
  in
  let key = { derivation; forms; lanes } in
  Table.find s.table key ~miss:ignore (fun () ->
      let (program, pf), flags, linear = make () in
      incr (Domain.DLS.get made);
      let sub (p, forms) = (p, site ~name:s.name ~compile:s.compile p forms) in
      let program, at = sub (program, pf) in
      { program; at; flags; linear = Option.map sub linear })

(* Operands *)

type map = { f : 'v 's 'd. ('v, 's, 'd) Nx.t -> ('v, 's, 'd) Nx.t }

let map_values m = Array.map (fun (V x) -> V (m.f x))

let map_operands m = function
  | Detach x -> Detach (m.f x)
  | Lanes (a, x) -> Lanes (a, m.f x)
  | Lane_index _ as c -> c
  | Add (t, x) -> Add (t, m.f x)
  | Scan s ->
      Scan
        {
          s with
          trips = Option.map (fun (V x) -> V (m.f x)) s.trips;
          init = map_values m s.init;
          xs = map_values m s.xs;
          captures = map_values m s.captures;
          seeds = map_values m s.seeds;
        }
  | Iterate s ->
      Iterate
        {
          s with
          init = map_values m s.init;
          step_captures = map_values m s.step_captures;
          until_captures = map_values m s.until_captures;
          seeds = map_values m s.seeds;
        }
  | Call c ->
      Call { c with args = map_values m c.args; seeds = map_values m c.seeds }
  | Custom c ->
      Custom
        {
          c with
          args = map_values m c.args;
          captures = map_values m c.captures;
          seeds = map_values m c.seeds;
        }
  | Linear c ->
      Linear
        {
          c with
          args = map_values m c.args;
          captures = map_values m c.captures;
          residuals = map_values m c.residuals;
        }
  | Remat r ->
      Remat { r with args = map_values m r.args; seeds = map_values m r.seeds }
  | Operator o ->
      Operator { o with args = map_values m o.args; data = map_values m o.data }
  | Root r ->
      Root
        {
          r with
          solution = map_values m r.solution;
          theta = map_values m r.theta;
          captures = map_values m r.captures;
        }
  | Solve s ->
      Solve
        {
          s with
          b = map_values m s.b;
          data = map_values m s.data;
          captures = map_values m s.captures;
        }
  | Switch s ->
      Switch
        {
          s with
          index =
            (let (V i) = s.index in
             V (m.f i));
          captures = map_values m s.captures;
          seeds = map_values m s.seeds;
        }

let operands = function
  | Detach x -> [ V x ]
  | Lanes (_, x) -> [ V x ]
  | Add (_, x) -> [ V x ]
  | Lane_index _ -> []
  | Scan { trips; init; xs; captures; seeds; _ } ->
      Option.to_list trips
      @ Array.to_list (Array.concat [ init; xs; captures; seeds ])
  | Iterate { init; step_captures; until_captures; seeds; _ } ->
      Array.to_list
        (Array.concat [ init; step_captures; until_captures; seeds ])
  | Call { args; seeds; _ } | Remat { args; seeds; _ } ->
      Array.to_list (Array.append args seeds)
  | Custom { args; captures; seeds; _ } ->
      Array.to_list (Array.concat [ args; captures; seeds ])
  | Linear { args; captures; residuals; _ } ->
      Array.to_list (Array.concat [ args; captures; residuals ])
  | Operator { args; data; _ } -> Array.to_list (Array.append args data)
  | Root { solution; theta; captures; _ } ->
      Array.to_list (Array.concat [ solution; theta; captures ])
  | Solve { b; data; captures; _ } ->
      Array.to_list (Array.concat [ b; data; captures ])
  | Switch { index; captures; seeds; _ } ->
      index :: Array.to_list (Array.append captures seeds)

let rec map_programs f = function
  | (Detach _ | Lanes _ | Lane_index _ | Add _ | Operator _) as c -> c
  | Scan s -> Scan { s with step = f s.step }
  | Iterate s -> Iterate { s with step = f s.step; until = f s.until }
  | Call c ->
      let body = f c.body in
      Call { c with body; site = bare c.site body }
  | Custom c ->
      let derivative =
        match c.derivative with
        | Tangent t -> Tangent (f t)
        | Pullback p -> Pullback (f p)
      in
      Custom { c with primal = f c.primal; derivative }
  | Linear c ->
      Linear { c with apply = Option.map f c.apply; transpose = f c.transpose }
  | Remat r -> Remat { r with body = f r.body }
  | Root r -> Root { r with system = map_system f r.system }
  | Solve s -> Solve { s with system = map_system f s.system }
  | Switch s -> Switch { s with branches = Array.map f s.branches }

(* A system's programs at its construct's level; the operator's, a lane's,
   stay. *)
and map_system f s =
  {
    s with
    residual = f s.residual;
    jacobian = f s.jacobian;
    transpose = f s.transpose;
    solve = f s.solve;
  }

(* Forms *)

(* Outputs [i] to [j - 1]. *)
let slice a i j = Array.sub a i (j - i)
let contiguous shape = Nx_array.Layout.contiguous (Iarray.of_array shape)

let shape_of (f : _ Nx.Prim.form) =
  Iarray.to_array (Nx_array.Layout.shape f.layout)

(* The form of a value of [f]'s with a leading axis of [n]. *)
let stacked n (F f) =
  F
    {
      f with
      layout = contiguous (Array.append [| n |] (shape_of f));
      placement = Nx.Placement.with_leading_axis f.placement;
    }

(* The running totals out [p]'s outputs end with. *)
let tail p =
  let n = Array.length p.outputs in
  slice p.outputs (n - Array.length p.adds) n

(* A loop's sums: one per total and shape its programs add to, in the order
   their programs first take them, each the chain of that total's additions in
   the order they are made, as an eager loop would make them. With each
   program's totals, the sum each feeds. *)
let sums programs =
  let found = ref [] in
  (* A sum's addition and form are its first addition's. *)
  let index (Addition (t, _, out) as a) =
    let f = { out with layout = contiguous (shape_of out) } in
    let shape = shape_of f in
    let rec go k = function
      | [] ->
          found := !found @ [ (a, F f) ];
          k
      | (Addition (u, _, _), F g) :: rest ->
          if Option.is_some (same_total t u) && shape_of g = shape then k
          else go (k + 1) rest
    in
    go 0 !found
  in
  let feeds = List.map (fun p -> Array.map index p.adds) programs in
  let adds, forms = List.split !found in
  (Array.of_list adds, Array.of_list forms, feeds)

let feeds step =
  match sums [ step ] with
  | adds, forms, [ feeds ] -> (adds, forms, feeds)
  | _ -> invalid_arg "Rune: a step's sums"

(* The additions a construct's results end with: its sums, one per total and
   shape its programs thread, which it starts from its seeds; then for a custom
   rule its tangent map's running totals out, chains from the identity. *)
let additions = function
  | Detach _ | Lanes _ | Lane_index _ | Add _ | Operator _ | Root _ | Solve _ ->
      [||]
  | Scan { step = p; _ } | Call { body = p; _ } | Remat { body = p; _ } ->
      let adds, _, _ = sums [ p ] in
      adds
  | Iterate { step; until; _ } ->
      let adds, _, _ = sums [ until; step ] in
      adds
  | Switch { branches; _ } ->
      let adds, _, _ = sums (Array.to_list branches) in
      adds
  | Custom { primal; derivative = Tangent t; _ } ->
      let adds, _, _ = sums [ primal ] in
      Array.append adds t.adds
  | Custom { primal; derivative = Pullback _; _ } ->
      let adds, _, _ = sums [ primal ] in
      adds
  | Linear { apply = Some a; _ } -> a.adds
  | Linear { apply = None; _ } -> [||]

(* How many of a construct's additions it starts from its seeds. *)
let seeded_count = function
  | Custom { primal; _ } ->
      let adds, _, _ = sums [ primal ] in
      Array.length adds
  | Linear _ -> 0
  | c -> Array.length (additions c)

(* [c] starting its sums from [seeds]. *)
let with_seeds c seeds =
  match c with
  | Scan s -> Scan { s with seeds }
  | Iterate s -> Iterate { s with seeds }
  | Call s -> Call { s with seeds }
  | Remat s -> Remat { s with seeds }
  | Switch s -> Switch { s with seeds }
  | Custom s -> Custom { s with seeds }
  | c -> c

let no_map name =
  invalid_arg (name ^ ": no map named by the axis is around the call")

let rec forms = function
  | Detach x -> [| form (V x) |]
  | Lanes (a, x) -> (
      match bound a with
      | Some n -> [| stacked n (form (V x)) |]
      | None -> no_map "Rune.lanes")
  | Lane_index (_, placement) ->
      [| F { dtype = Nx.int32; layout = contiguous [||]; placement } |]
  | Add _ -> [||]
  | Scan { step; length; trips; init; _ } ->
      let nc = Array.length init and no = Array.length step.outputs in
      let na = Array.length step.adds in
      let carry =
        if length = 0 then Array.map form init
        else if Option.is_some trips then Array.map form init
        else Array.map form (slice step.outputs 0 nc)
      in
      let ys = Array.map form (slice step.outputs nc (no - na)) in
      let _, sums, _ = sums [ step ] in
      Array.concat [ carry; Array.map (stacked length) ys; sums ]
  | Iterate { name; step; until; max; rows; init; _ } ->
      let _, sums, _ = sums [ until; step ] in
      let carry = Array.map form init in
      let stacked =
        match rows with
        | None -> [||]
        | Some Taken ->
            invalid_arg
              (name ^ ": an iterate stacking the steps it takes has no form")
        | Some Max ->
            let nc = Array.length init in
            let no = Array.length step.outputs - Array.length step.adds in
            (* The count lies where the stop does, where a map whose lanes stop
               apart counts each lane's steps. *)
            let (F u) = form until.outputs.(0) in
            let trips =
              F
                {
                  dtype = Nx.int32;
                  layout = contiguous [||];
                  placement = Nx.Placement.uncut u.placement;
                }
            in
            Array.append [| trips |]
              (Array.map (stacked max)
                 (Array.map form (slice step.outputs nc no)))
      in
      Array.concat [ carry; stacked; sums ]
  | Call { body = p; _ } | Remat { body = p; _ } -> results p
  | Custom { primal; derivative = Tangent t; _ } ->
      Array.append (results primal) (Array.map form (tail t))
  | Custom { primal; derivative = Pullback _; _ } -> results primal
  | Linear { apply = Some a; forms; _ } ->
      Array.append forms (Array.map form (tail a))
  | Linear { apply = None; forms; _ } -> forms
  | Operator { forms; _ } -> forms
  | Root { solution; _ } -> Array.map form solution
  | Solve { system; b; _ } ->
      Array.map form (slice system.solve.outputs 0 (Array.length b))
  | Switch { branches; _ } ->
      let _, sums, _ = sums (Array.to_list branches) in
      Array.append (switched branches) sums

(* A switch's results' forms, its first branch's: every branch's agree. *)
and switched branches =
  let p = branches.(0) in
  Array.map form
    (slice p.outputs 0 (Array.length p.outputs - Array.length p.adds))

(* The forms of [p]'s results, then its sums'. *)
and results p =
  let n = Array.length p.outputs - Array.length p.adds in
  let _, sums, _ = sums [ p ] in
  Array.append (Array.map form (slice p.outputs 0 n)) sums

(* Values of a form *)

(* The additive identity that keeps every sign: [-0.], and [0] where a dtype has
   no negative zero. *)
let neg_zero (type v s) (dt : (v, s) Nx.dtype) : v =
  match D.kind dt with
  | Float -> -0.
  | Complex -> { Complex.re = -0.; im = -0. }
  | Signed | Unsigned | Boolean -> D.zero dt

let filled ~zero (F f) =
  let v = if zero then D.zero f.dtype else neg_zero f.dtype in
  V (Nx.full_at f.placement f.dtype (shape_of f) v)

let add (V a) b = V (Nx.add a (like a b))

(* Rows each read at [f], the form their output was traced at. *)
let stack (F f) vs =
  let set = Nx.Placement.devices f.placement in
  V (Nx.stack ~axis:0 (List.map (cast f.dtype set) vs))

(* Running totals *)

type _ Effect.t +=
  | Running :
      ('v, 's, 'd) total * ('v, 's, 'd) Nx.Prim.form
      -> ('v, 's, 'd) Nx.t option Effect.t
  | Replace : ('v, 's, 'd) total * ('v, 's, 'd) Nx.t -> unit Effect.t

let running t f =
  match Effect.perform (Running (t, f)) with
  | r -> r
  | exception Effect.Unhandled (Running _) -> None

let replace t x = Effect.perform (Replace (t, x))

type 'd where = Nowhere | At of 'd Nx.Placement.t option
type _ Effect.t += Where : ('v, 's, 'd) total -> 'd where Effect.t

let where t =
  match Effect.perform (Where t) with
  | w -> w
  | exception Effect.Unhandled (Where _) -> Nowhere

type _ Effect.t +=
  | Accepts : ('v, 's, 'd) total * ('v, 's, 'd) Nx.Prim.form -> unit Effect.t

let accepts t f =
  match Effect.perform (Accepts (t, f)) with
  | () -> ()
  | exception Effect.Unhandled (Accepts _) -> ()

(* What answers for a running value takes any addition to it; otherwise the
   addition reaches a collect, which may refuse it. *)
let admits t f = if Option.is_none (running t f) then accepts t f

(* Evaluation *)

(* An unbound slot. *)
let hole = V (Nx.zeros Nx.bool [||] : (bool, Nx.bool_elt, Nx.host) Nx.t)

(* Whether [until]'s output, a boolean of one element at any set, holds. *)
let stops (V y) =
  match D.equal_witness (Nx.dtype y) Nx.bool with
  | Some Type.Equal -> (Nx.item (List.init (Nx.ndim y) (fun _ -> 0)) y : bool)
  | None ->
      invalid_arg
        (Format.asprintf "Rune: until gave a %a value, not a boolean"
           Nx.pp_dtype (Nx.dtype y))

(* A running value computed on a set its total's running value does not lie
   on. *)
let unset (V y) =
  invalid_arg
    (Format.asprintf
       "Rune.Total.add: a running total takes a value on %a, as the lanes of a \
        map held apart compute it, which its running value's set cannot hold; \
        give the collect's zero that placement"
       Nx.Placement.pp (Nx.placement y))

let rec eval ?totals p args =
  let na = Array.length p.adds in
  let totals =
    match totals with
    | Some ts -> ts
    | None ->
        Array.map (fun (Addition (_, f, _)) -> filled ~zero:false (F f)) p.adds
  in
  let args = Array.append args totals in
  if Array.length args <> Array.length p.params || Array.length totals <> na
  then
    invalid_arg
      (Printf.sprintf "Rune: a program of %d parameters applied to %d values"
         (Array.length p.params) (Array.length args));
  let env = Array.make p.slots hole in
  Array.iteri (fun k slot -> env.(slot) <- args.(k)) p.params;
  let owner = p.owner in
  let lookup (type v s d) (x : (v, s, d) Nx.t) : (v, s, d) Nx.t =
    match Nx.Prim.payload owner x with Some (Var n) -> like x env.(n) | _ -> x
  in
  let m = { f = lookup } in
  let at = ref 0 in
  let bind (type v s d) (y : (v, s, d) Nx.t) =
    env.(!at) <- V y;
    incr at
  in
  Array.iter
    (function
      | Op { by; op; base } ->
          let op = Nx.Prim.map lookup op in
          let r = Nx.Prim.eval ~by op in
          at := base;
          Nx.Prim.iter_results bind op r
      | Do { c; base } ->
          let r = perform (map_operands m c) in
          Array.blit r 0 env base (Array.length r))
    p.eqns;
  map_values m p.outputs

and perform c =
  match Effect.perform (Construct c) with
  | r -> r
  | exception Effect.Unhandled (Construct c') when c' == c -> default c
  | exception Evaluate -> default c

and default = function
  | Detach x -> [| V x |]
  | Lanes _ -> no_map "Rune.lanes"
  | Lane_index _ -> no_map "Rune.lane_index"
  | Add _ -> [||]
  | Scan { step; length; trips; reverse; init; xs; captures; seeds } ->
      scan step ~length ~trips ~reverse ~init ~xs ~captures ~seeds
  | Iterate
      {
        name;
        step;
        until;
        max;
        rows;
        stopped;
        init;
        step_captures;
        until_captures;
        seeds;
      } ->
      iterate name step ~until ~max ~rows ~stopped ~init ~step_captures
        ~until_captures ~seeds
  | Call { body; args; site; seeds } ->
      let untraced (V x) = Option.is_none (Nx.Prim.owner x) in
      from_seeds body seeds (fun totals ->
          match site.compiled with
          | Some run
            when Array.for_all untraced args
                 && Array.for_all untraced totals
                 && not (staging ()) ->
              run (Array.append args totals)
          | _ -> eval ~totals body args)
  | Custom { primal; derivative; ny; args; captures; residuals; seeds; _ } ->
      let args = Array.append args captures in
      let nr = Array.length primal.outputs - ny - Array.length primal.adds in
      (* A tangent map's additions, which no differentiation made. *)
      let tadds =
        match derivative with
        | Tangent t -> Array.map (fun v -> filled ~zero:false (form v)) (tail t)
        | Pullback _ -> [||]
      in
      if residuals then
        Array.append
          (from_seeds primal seeds (fun totals -> eval ~totals primal args))
          tadds
      else
        let p = values_only primal ~ny in
        let out = from_seeds p seeds (fun totals -> eval ~totals p args) in
        Array.concat
          [
            slice out 0 ny;
            Array.make nr hole;
            slice out ny (Array.length out);
            tadds;
          ]
  | Linear { apply = Some a; args; captures; _ } ->
      eval a (Array.append args captures)
  | Linear { name; apply = None; _ } ->
      invalid_arg
        (name
       ^ ": a custom_vjp rule has no forward derivative; give the function a \
          custom_jvp rule")
  | Remat { body; args; seeds } ->
      from_seeds body seeds (fun totals -> eval ~totals body args)
  | Switch { name; index; branches; captures; seeds } ->
      let n = Array.length branches in
      let k = count index in
      if k < 0 || k >= n then
        invalid_arg
          (Printf.sprintf "%s: index %d outside 0 .. %d" name k (n - 1));
      let _, forms, feeds = sums (Array.to_list branches) in
      let acc = accumulators forms seeds in
      let p = branches.(k) in
      let out = evaluate p (List.nth feeds k) acc captures in
      let results = slice out 0 (Array.length out - Array.length p.adds) in
      Array.append results acc
  | Operator _ ->
      invalid_arg
        "Rune.root: linear_solve's operator was applied after linear_solve \
         returned"
  | Root { solution; _ } -> solution
  | Solve { system; transposed; b; data; captures; _ } ->
      let body ~transposed:t ~lanes =
        system.operator ~transposed:(t <> transposed) ~lanes
      in
      let p = fill system.solve ~id:system.id body in
      slice (eval p (Array.concat [ b; data; captures ])) 0 (Array.length b)

(* [p] with the operator [id]'s applications calls of [body]'s programs. *)
and fill (p : t) ~id body =
  let eqn = function
    | Do { c = Operator o; base } when o.id = id ->
        let body, captures = body ~transposed:o.transposed ~lanes:o.lanes in
        let args = Array.concat [ o.args; o.data; captures ] in
        let site = evaluated ~name:"Rune.root" body in
        Do { c = Call { body; args; site; seeds = [||] }; base }
    | Do { c; base } ->
        Do { c = map_programs (fun q -> fill q ~id body) c; base }
    | e -> e
  in
  { p with eqns = Array.map eqn p.eqns }

(* A site of [body] that evaluates it, as each program derived from it. *)
and evaluated ~name body =
  let compile p _ vs =
    let n = Array.length vs - Array.length p.adds in
    eval ~totals:(slice vs n (Array.length vs)) p (slice vs 0 n)
  in
  {
    name;
    compile;
    compiled = None;
    axes = named body;
    iterates = holds (function Iterate _ -> true | _ -> false) body;
    customs = holds custom body;
    table = Table.create ();
  }

and eval_seeded p ~seeds args =
  from_seeds p seeds (fun totals -> eval ~totals p args)

(* [f totals], [p]'s outputs from its totals' running values in [totals], its
   sums started from [seeds] and ended where its running values out are. *)
and from_seeds p seeds f =
  let _, forms, feeds = sums [ p ] in
  let feeds = List.hd feeds and acc = accumulators forms seeds in
  let out = feeding p feeds acc f in
  Array.append (slice out 0 (Array.length out - Array.length feeds)) acc

(* [f totals], [p]'s outputs from its running totals in [totals], each taken
   from the innermost collect or tracer that answers for its total and handed
   back its running value out, the results alone. A total none answers for, or
   whose running value [p] cannot take, starts from the identity, and its end is
   added to it. *)
and threaded ?lanes p f =
  threading ?lanes p.adds (fun totals ->
      let out = f totals in
      let n = Array.length out - Array.length p.adds in
      (slice out n (Array.length out), slice out 0 n))

and threading :
    'a. ?lanes:int -> addition array -> (value array -> value array * 'a) -> 'a
    =
 fun ?lanes adds f ->
  let at (type v s d) (f : (v, s, d) Nx.Prim.form) : (v, s, d) Nx.Prim.form =
    match lanes with
    | None -> f
    | Some n ->
        {
          f with
          layout = contiguous (Array.append [| n |] (shape_of f));
          placement = Nx.Placement.with_leading_axis f.placement;
        }
  in
  (* Each total's running value in, and how its running value out is handed
     back: at the form [p] gave it, or at the running value's set, where a map
     whose lanes stop apart computed it there. *)
  let ins =
    Array.map
      (fun (Addition (t, into, out)) ->
        let into = at into and out = at out in
        let r = running t into in
        let typed (V y as v) =
          match coerce_at out y with
          | Some y -> y
          | None -> (
              match r with
              | Some r -> (
                  match coerce_at (Nx.Prim.form r) y with
                  | Some y -> y
                  | None -> unset v)
              | None -> unset v)
        in
        match r with
        | Some r when Option.is_some (coerce_at into r) ->
            ( V r,
              fun v ->
                let y = typed v in
                fun () -> replace t y )
        | _ ->
            if Option.is_none r then accepts t out;
            ( filled ~zero:false (F into),
              fun v ->
                let y = typed v in
                fun () -> add_to t y ))
      adds
  in
  let outs, r = f (Array.map fst ins) in
  (* Every running value out is read back before any is delivered, so that
     one that raises delivers none. *)
  let deliveries = Array.mapi (fun k (_, back) -> back outs.(k)) ins in
  Array.iter (fun deliver -> deliver ()) deliveries;
  r

(* [x] added to the innermost collect or tracer that answers for [t], as its
   running value plus [x], and otherwise performed as an addition. *)
and add_to : type v s d. (v, s, d) total -> (v, s, d) Nx.t -> unit =
 fun t x ->
  match running t (Nx.Prim.form x) with
  | Some r -> replace t (Nx.add r x)
  | None -> ignore (perform (Add (t, x)))

(* [p], a custom rule's primal, with its results and running totals alone. Where
   its residuals are its parameters, constants or results, nothing reads only
   them, and every equation is kept. *)
and values_only p ~ny =
  let no = Array.length p.outputs and na = Array.length p.adds in
  let owner = p.owner in
  let free (V r) =
    match Nx.Prim.payload owner r with
    | Some (Var n) ->
        Array.mem n p.params
        || Array.exists (fun y -> same y (V r)) (slice p.outputs 0 ny)
    | _ -> true
  in
  if Array.for_all free (slice p.outputs ny (no - na)) then
    { p with outputs = Array.append (slice p.outputs 0 ny) (tail p) }
  else prune p ~outputs:(Array.init ny Fun.id)

(* A loop's sums start from its seeds, the running totals of the collects they
   reach, or else at the identity, so that a loop that runs no step adds nothing
   a total can tell. *)
and accumulators forms seeds =
  if seeds = [||] then Array.map (filled ~zero:false) forms
  else Array.copy seeds

(* [f totals], [p]'s outputs from its totals' running values in [totals], which
   run from the sums [acc] they feed, each sum's end in [acc] once it has run. A
   total starts from its sum where [p]'s running value in can take it, and
   otherwise from the identity, its end then added to the sum. *)
and feeding p feeds acc f =
  let ins =
    Array.mapi
      (fun k (Addition (_, into, _)) ->
        let (V a as v) = acc.(feeds.(k)) in
        match coerce_at into a with
        | Some _ -> (v, true)
        | None -> (filled ~zero:false (F into), false))
      p.adds
  in
  let out = f (Array.map fst ins) in
  let base = Array.length out - Array.length feeds in
  Array.iteri
    (fun k j ->
      let o = out.(base + k) in
      acc.(j) <- (if snd ins.(k) then o else add acc.(j) o))
    feeds;
  out

and evaluate p feeds acc args =
  feeding p feeds acc (fun totals -> eval ~totals p args)

and scan step ~length ~trips ~reverse ~init ~xs ~captures ~seeds =
  let nc = Array.length init in
  let no = Array.length step.outputs and na = Array.length step.adds in
  let ny = no - nc - na in
  let _, forms, feeds = sums [ step ] in
  let feeds = List.hd feeds and acc = accumulators forms seeds in
  let t = match trips with None -> length | Some v -> count v in
  if t < 0 || t > length then
    invalid_arg (Printf.sprintf "Rune: a scan of %d rows runs %d" length t);
  let carry = ref init and ys = Array.make_matrix ny length None in
  for s = 0 to t - 1 do
    let i = if reverse then t - 1 - s else s in
    let row = Array.map (fun (V x) -> V (Nx.get [ i ] x)) xs in
    let out =
      evaluate step feeds acc (Array.concat [ !carry; row; captures ])
    in
    carry := slice out 0 nc;
    for j = 0 to ny - 1 do
      ys.(j).(i) <- Some out.(nc + j)
    done
  done;
  let ys =
    Array.mapi (fun j rows -> stacked_rows step.outputs.(nc + j) rows) ys
  in
  Array.concat [ !carry; ys; acc ]

(* The rows of an output traced as [y], each computed or a zero, stacked. *)
and stacked_rows y rows =
  let f = form y in
  if rows = [||] then filled ~zero:true (stacked 0 f)
  else
    stack f
      (Array.to_list
         (Array.map (function Some v -> v | None -> filled ~zero:true f) rows))

(* An [int32] count. *)
and count (V x) =
  match D.equal_witness (Nx.dtype x) Nx.int32 with
  | Some Type.Equal -> Int32.to_int (Nx.item [] x : int32)
  | None -> invalid_arg "Rune: a count that is not an int32"

and iterate name step ~until ~max ~rows ~stopped ~init ~step_captures
    ~until_captures ~seeds =
  let nc = Array.length init in
  let ny = Array.length step.outputs - nc - Array.length step.adds in
  let _, forms, feeds = sums [ until; step ] in
  let until_feeds = List.nth feeds 0 and step_feeds = List.nth feeds 1 in
  let acc = accumulators forms seeds in
  let ys = ref [] in
  let last = ref None in
  let rec go k carry =
    let out =
      evaluate until until_feeds acc (Array.append carry until_captures)
    in
    last := Some out.(0);
    if stops out.(0) then (k, carry)
    else if k = max then
      invalid_arg
        (Printf.sprintf "%s: until is still false after max = %d steps%s" name
           max
           (match stopped with Some j -> running_lane carry.(j) | None -> ""))
    else
      let out =
        evaluate step step_feeds acc (Array.append carry step_captures)
      in
      ys := slice out nc (nc + ny) :: !ys;
      go (k + 1) (slice out 0 nc)
  in
  let k, carry = go 0 init in
  match rows with
  | None -> Array.append carry acc
  | Some rows ->
      let taken = Array.of_list (List.rev !ys) in
      let length = match rows with Taken -> k | Max -> max in
      let stacked =
        Array.init ny (fun j ->
            stacked_rows
              step.outputs.(nc + j)
              (Array.init length (fun i ->
                   if i < k then Some taken.(i).(j) else None)))
      in
      (* The count lies where the stop does. *)
      let (V s) = Option.get !last in
      let trips =
        V
          (Nx.scalar_at
             (Nx.Placement.uncut (Nx.placement s))
             Nx.int32 (Int32.of_int k))
      in
      Array.concat [ carry; [| trips |]; stacked; acc ]

(* The lanes a map's iterate names when it runs out of steps: of its lanes'
   stops [flags], the first still false, each axis's position. *)
and running_lane (V flags) =
  match D.equal_witness (Nx.dtype flags) Nx.bool with
  | None -> ""
  | Some Type.Equal -> (
      let shape = Nx.shape flags in
      let all = (Nx.to_array flags : bool array) in
      let rec first i =
        if i >= Array.length all then None
        else if not all.(i) then Some i
        else first (i + 1)
      in
      match first 0 with
      | None -> ""
      | Some i ->
          let index = Array.make (Array.length shape) 0 and rest = ref i in
          for d = Array.length shape - 1 downto 0 do
            index.(d) <- !rest mod shape.(d);
            rest := !rest / shape.(d)
          done;
          String.concat ""
            (Array.to_list (Array.map (Printf.sprintf ", in lane %d") index)))

(* Pruning *)

and prune p ~outputs =
  let owner = p.owner in
  let live = Array.make p.slots false in
  let mark (type v s d) (x : (v, s, d) Nx.t) =
    match Nx.Prim.payload owner x with
    | Some (Var n) -> live.(n) <- true
    | _ -> ()
  in
  let outputs =
    Array.append (Array.map (fun k -> p.outputs.(k)) outputs) (tail p)
  in
  Array.iter (fun (V y) -> mark y) outputs;
  let n = Array.length p.eqns in
  let base = function Op { base; _ } | Do { base; _ } -> base in
  let kept = ref [] in
  for e = n - 1 downto 0 do
    let eq = p.eqns.(e) in
    let b = base eq in
    let next = if e + 1 < n then base p.eqns.(e + 1) else p.slots in
    let read = ref false in
    for v = b to next - 1 do
      if live.(v) then read := true
    done;
    let keep =
      match eq with Op { op = Check _; _ } -> true | Op _ | Do _ -> !read
    in
    if keep then begin
      kept := eq :: !kept;
      match eq with
      | Op { op; _ } ->
          let (Nx.Prim.Operands xs) = Nx.Prim.operands op in
          List.iter (fun (Nx.Prim.Any x) -> mark x) xs
      | Do { c; _ } -> List.iter (fun (V x) -> mark x) (operands c)
    end
  done;
  { p with eqns = Array.of_list !kept; outputs }

(* A total's running value, where a collect or a tracer answers for it. *)
type seed = Seed : ('v, 's, 'd) total * ('v, 's, 'd) Nx.t -> seed | Unseeded

(* A loop asks each total it adds to for its running value and chains its
   additions from it, then hands the chain's end back; a total none answers for
   receives the chain as one addition. Another construct's running totals out,
   chains from the identity, are added to their totals. *)
let run c =
  let adds = additions c in
  let ns = seeded_count c in
  let seeds =
    Array.mapi
      (fun k (Addition (t, _, f)) ->
        if k >= ns then Unseeded
        else match running t f with Some v -> Seed (t, v) | None -> Unseeded)
      adds
  in
  (* An addition a collect would refuse raises before the construct runs, so
     that it delivers none, also where no trace met the addition. *)
  Array.iteri
    (fun k (Addition (t, _, f)) ->
      match seeds.(k) with Unseeded -> admits t f | Seed _ -> ())
    adds;
  let c =
    if Array.for_all (fun s -> s = Unseeded) seeds then c
    else
      let fs = forms c in
      let base = Array.length fs - Array.length adds in
      with_seeds c
        (Array.init ns (fun k ->
             match seeds.(k) with
             | Seed (_, v) -> V v
             | Unseeded -> filled ~zero:false fs.(base + k)))
  in
  let r = perform c in
  let base = Array.length r - Array.length adds in
  (* Every sum is read back before any is delivered. *)
  let deliveries =
    Array.mapi
      (fun k (Addition (t, _, f)) ->
        let set = Nx.Placement.devices f.placement in
        match seeds.(k) with
        | Seed (u, v) -> (
            (* The chain's end lies where its seed does. *)
            match same_total u t with
            | Some Type.Equal ->
                let y = like v r.(base + k) in
                fun () -> replace t y
            | None -> invalid_arg "Rune: a seed of another total")
        | Unseeded ->
            let y = cast f.dtype set r.(base + k) in
            fun () -> add_to t y)
      adds
  in
  Array.iter (fun deliver -> deliver ()) deliveries;
  r

(* Printing

   One naming context covers a program and the programs it carries: each
   program's variables are numbered from where the previous one's end, so no two
   programs share a name, and the constants every one of them reads are numbered
   once, in the order they are first read, and declared at the top. *)

type names = {
  mutable constants : value list;
  mutable axes : axis list;  (** The maps named, in the order first seen. *)
  mutable next : int;
  mutable bases : (t * int) list;
}

let rec mentions a (p : t) =
  Array.exists
    (function
      | Op _ -> false
      | Do { c = Lanes (b, _) | Lane_index (b, _); _ } when same_axis a b ->
          true
      | Do { c; _ } -> List.exists (mentions a) (carried c))
    p.eqns

let widen (p : t) ~at n =
  let np = Array.length p.params - Array.length p.adds in
  let slot = Array.make n (-1) in
  Array.iteri (fun k j -> slot.(j) <- p.params.(k)) at;
  let next = ref p.slots in
  let fresh s =
    if s >= 0 then s
    else begin
      incr next;
      !next - 1
    end
  in
  let params = Array.map fresh slot in
  {
    p with
    params = Array.append params (slice p.params np (Array.length p.params));
    slots = !next;
  }

let unite traced ~params =
  let union = ref [] in
  let position v =
    let rec go j = function
      | [] ->
          union := !union @ [ v ];
          j
      | u :: rest -> if same u v then j else go (j + 1) rest
    in
    go 0 !union
  in
  let ats =
    Array.map
      (fun (_, own) ->
        Array.append (Array.init params Fun.id)
          (Array.map (fun v -> params + position v) own))
      traced
  in
  let captures = Array.of_list !union in
  let n = params + Array.length captures in
  (Array.map2 (fun (p, _) at -> widen p ~at n) traced ats, captures)

let rec reads_lanes ?(index = true) (p : t) =
  Array.exists
    (function
      | Op _ -> false
      | Do { c = Lanes _; _ } -> true
      | Do { c = Lane_index _; _ } -> index
      | Do { c; _ } -> List.exists (reads_lanes ~index) (carried c))
    p.eqns

let reads_params p mask =
  let q =
    prune p
      ~outputs:
        (Array.init (Array.length p.outputs - Array.length p.adds) Fun.id)
  in
  let read = ref false in
  let see (type v s d) (x : (v, s, d) Nx.t) =
    match Nx.Prim.payload q.owner x with
    | Some (Var n) ->
        Array.iteri (fun k m -> if m && q.params.(k) = n then read := true) mask
    | _ -> ()
  in
  Array.iter
    (function
      | Op { op; _ } ->
          let (Nx.Prim.Operands xs) = Nx.Prim.operands op in
          List.iter (fun (Nx.Prim.Any x) -> see x) xs
      | Do { c; _ } -> List.iter (fun (V x) -> see x) (operands c))
    q.eqns;
  Array.iter (fun (V x) -> see x) q.outputs;
  !read

let owned (p : t) (V x) =
  match Nx.Prim.payload p.owner x with Some (Var _) -> true | _ -> false

let rec register names (p : t) =
  (* A program carried twice prints once at the base it first took. *)
  if not (List.mem_assq p names.bases) then begin
    names.bases <- (p, names.next) :: names.bases;
    names.next <- names.next + p.slots
  end;
  let axis a =
    if not (List.mem a names.axes) then names.axes <- names.axes @ [ a ]
  in
  let see v =
    if (not (owned p v)) && not (List.exists (same v) names.constants) then
      names.constants <- names.constants @ [ v ]
  in
  Array.iter
    (function
      | Op { op; _ } ->
          let (Operands xs) = Nx.Prim.operands op in
          List.iter (fun (Nx.Prim.Any x) -> see (V x)) xs
      | Do { c; _ } ->
          (match c with Lanes (a, _) | Lane_index (a, _) -> axis a | _ -> ());
          List.iter see (operands c);
          List.iter (register names) (carried c))
    p.eqns;
  Array.iter see p.outputs

let base names p = List.assq p names.bases

let pp_name names (p : t) ppf (V x as v) =
  match Nx.Prim.payload p.owner x with
  | Some (Var n) -> Format.fprintf ppf "%%%d" (base names p + n)
  | _ ->
      let rec find k = function
        | [] -> k
        | u :: rest -> if same u v then k else find (k + 1) rest
      in
      Format.fprintf ppf "c%d" (find 0 names.constants)

let comma ppf () = Format.pp_print_string ppf ", "

(* A map's number in the printout: the order the program first names it. *)
let map_number names a =
  let rec find k = function
    | [] -> k
    | b :: rest -> if b = a then k else find (k + 1) rest
  in
  find 0 names.axes

let pp_names names p ppf vs =
  Format.pp_print_list ~pp_sep:comma (pp_name names p) ppf vs

let pp_form ppf (F f) =
  let shape = Iarray.to_array (Nx_array.Layout.shape f.layout) in
  Format.fprintf ppf "%a %a at %a" Nx.pp_dtype f.dtype Nx.pp_shape shape
    Nx.Placement.pp f.placement

(* The forms of an operation's results, as eager computation gives them. *)
let result_forms (p : t) ~by op =
  let forms = ref [] in
  ignore
    (Nx.Prim.results ~by
       (fun k f ->
         forms := (k, F f) :: !forms;
         Nx.Prim.traced p.owner f (Var k))
       op);
  List.map snd (List.sort (fun (a, _) (b, _) -> Int.compare a b) !forms)

(* The variables a construct's results bind: from [base] to the next equation's,
   those its program's parameters take aside. *)
let construct_results (p : t) i base =
  let next =
    if i + 1 < Array.length p.eqns then
      match p.eqns.(i + 1) with Op { base; _ } | Do { base; _ } -> base
    else p.slots
  in
  List.filter
    (fun n -> not (Array.mem n p.params))
    (List.init (Int.max 0 (next - base)) (fun k -> base + k))

let rec pp_program names ~indent ppf (p : t) =
  let pad = String.make indent ' ' in
  let b = base names p in
  let var ppf n = Format.fprintf ppf "%%%d" (b + n) in
  let na = Array.length p.adds in
  let ni = Array.length p.params - na in
  let params = Array.to_list p.params in
  Format.fprintf ppf "%sprogram (%a)" pad
    (Format.pp_print_list ~pp_sep:comma var)
    (List.filteri (fun i _ -> i < ni) params);
  if na > 0 then
    Format.fprintf ppf ", totals (%a)"
      (Format.pp_print_list ~pp_sep:comma var)
      (List.filteri (fun i _ -> i >= ni) params);
  Array.iteri (fun i e -> pp_eqn names ~indent p i ppf e) p.eqns;
  let outs = Array.to_list p.outputs in
  let n = List.length outs - na in
  Format.fprintf ppf "@\n%s  return %a" pad (pp_names names p)
    (List.filteri (fun i _ -> i < n) outs);
  if na > 0 then
    Format.fprintf ppf "@\n%s  totals %a" pad (pp_names names p)
      (List.filteri (fun i _ -> i >= n) outs)

and pp_eqn names ~indent (p : t) i ppf eqn =
  let pad = String.make (indent + 2) ' ' in
  let b = base names p in
  let lhs ppf results =
    match results with
    | [] -> ()
    | _ ->
        Format.fprintf ppf "%a = "
          (Format.pp_print_list ~pp_sep:comma (fun ppf (n, form) ->
               Format.fprintf ppf "%%%d : %a" (b + n) pp_form form))
          results
  in
  match eqn with
  | Op { by; op; base } ->
      let forms = result_forms p ~by op in
      let (Operands xs) = Nx.Prim.operands op in
      Format.fprintf ppf "@\n%s%a%a" pad lhs
        (List.mapi (fun k f -> (base + k, f)) forms)
        Nx.Prim.pp op;
      if xs <> [] then
        Format.fprintf ppf "@\n%s  with %a" pad
          (Format.pp_print_list ~pp_sep:comma (fun ppf (k, v) ->
               Format.fprintf ppf "x%d = %a" k (pp_name names p) v))
          (List.mapi (fun k (Nx.Prim.Any x) -> (k, V x)) xs)
  | Do { c; base } ->
      let ns = construct_results p i base in
      let forms =
        match forms c with
        | fs when Array.length fs = List.length ns -> Array.to_list fs
        | _ | (exception Invalid_argument _) -> []
      in
      let results =
        if forms = [] then List.map (fun n -> (n, None)) ns
        else List.map2 (fun n f -> (n, Some f)) ns forms
      in
      Format.fprintf ppf "@\n%s%a%a" pad
        (fun ppf rs ->
          match rs with
          | [] -> ()
          | _ ->
              Format.fprintf ppf "%a = "
                (Format.pp_print_list ~pp_sep:comma (fun ppf (n, f) ->
                     match f with
                     | Some f ->
                         Format.fprintf ppf "%%%d : %a" (b + n) pp_form f
                     | None -> Format.fprintf ppf "%%%d" (b + n)))
                rs)
        results
        (pp_construct names ~indent:(indent + 2) p)
        c

and pp_construct names ~indent (p : t) ppf c =
  let pad = String.make indent ' ' in
  let value = pp_name names p and values = pp_names names p in
  let program ppf q =
    Format.fprintf ppf "@\n%a" (pp_program names ~indent:(indent + 4)) q
  in
  let seeded_from ppf seeds =
    if seeds <> [||] then
      Format.fprintf ppf ", seeds (%a)" values (Array.to_list seeds)
  in
  match c with
  | Detach x -> Format.fprintf ppf "detach %a" value (V x)
  | Lanes (a, x) ->
      Format.fprintf ppf "lanes of map %d %a" (map_number names a) value (V x)
  | Lane_index (a, _) ->
      Format.fprintf ppf "lane_index of map %d" (map_number names a)
  | Add (_, x) -> Format.fprintf ppf "add %a" value (V x)
  | Scan { step; length; trips; reverse; init; xs; captures; seeds } ->
      Format.fprintf ppf "scan %d rows%a%s, init (%a), xs (%a), captures (%a)"
        length
        (fun ppf -> function
          | Some t -> Format.fprintf ppf ", the first %a" value t | None -> ())
        trips
        (if reverse then " last first" else "")
        values (Array.to_list init) values (Array.to_list xs) values
        (Array.to_list captures);
      seeded_from ppf seeds;
      Format.fprintf ppf "@\n%s  step%a" pad program step
  | Iterate { step; until; max; rows; init; step_captures; until_captures; _ }
    ->
      Format.fprintf ppf
        "iterate at most %d%s, init (%a), captures (%a) and (%a)@\n\
         %s  until%a@\n\
         %s  step%a"
        max
        (match rows with
        | None -> ""
        | Some Taken -> ", stacking the steps taken"
        | Some Max -> ", stacking max steps")
        values (Array.to_list init) values
        (Array.to_list until_captures)
        values
        (Array.to_list step_captures)
        pad program until pad program step
  | Call { body; args; seeds; _ } ->
      Format.fprintf ppf "call (%a)%a@\n%s  body%a" values (Array.to_list args)
        seeded_from seeds pad program body
  | Remat { body; args; seeds } ->
      Format.fprintf ppf "remat (%a)%a@\n%s  body%a" values (Array.to_list args)
        seeded_from seeds pad program body
  | Switch { index; branches; captures; seeds; _ } ->
      Format.fprintf ppf "switch %a (%a)%a" values [ index ] values
        (Array.to_list captures) seeded_from seeds;
      Array.iteri
        (fun k b -> Format.fprintf ppf "@\n%s  branch %d%a" pad k program b)
        branches
  | Operator { transposed; lanes; args; data; _ } ->
      Format.fprintf ppf "operator%s%s (%a) of (%a)"
        (if transposed then " transposed" else "")
        (String.concat ""
           (List.map (fun (n, _) -> Printf.sprintf " over %d lanes" n) lanes))
        values (Array.to_list args) values (Array.to_list data)
  | Root { name; system; solution; theta; captures } ->
      Format.fprintf ppf "%s (%a) of (%a), captures (%a)@\n%s  residual%a" name
        values (Array.to_list solution) values (Array.to_list theta) values
        (Array.to_list captures) pad program system.residual
  | Solve { name; system; transposed; b; data; captures } ->
      Format.fprintf ppf "%s solve%s (%a) of (%a), captures (%a)@\n%s  solve%a"
        name
        (if transposed then " transposed" else "")
        values (Array.to_list b) values (Array.to_list data) values
        (Array.to_list captures) pad program system.solve
  | Custom { name; primal; derivative; args; captures; seeds; _ } ->
      let kind, d =
        match derivative with
        | Tangent d -> ("tangent map", d)
        | Pullback d -> ("pullback", d)
      in
      Format.fprintf ppf "%s (%a), captures (%a)%a@\n%s  primal%a@\n%s  %s%a"
        name values (Array.to_list args) values (Array.to_list captures)
        seeded_from seeds pad program primal pad kind program d
  | Linear { name; apply; transpose; args; captures; residuals; _ } ->
      Format.fprintf ppf "linear %s (%a), captures (%a), residuals (%a)" name
        values (Array.to_list args) values (Array.to_list captures) values
        (Array.to_list residuals);
      Option.iter
        (fun a -> Format.fprintf ppf "@\n%s  function%a" pad program a)
        apply;
      Format.fprintf ppf "@\n%s  transpose%a" pad program transpose

let pp ppf p =
  let names = { constants = []; axes = []; next = 0; bases = [] } in
  register names p;
  List.iteri
    (fun k v -> Format.fprintf ppf "c%d : %a@\n" k pp_form (form v))
    names.constants;
  pp_program names ~indent:0 ppf p
