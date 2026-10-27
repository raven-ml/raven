(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)


module D = Nx_array.Dtype
module L = Nx_array.Layout
module M = Nx_array.Move
module P = Nx_kernel.Prog
module E = P.E

let full d = { M.start = 0; count = d; step = 1 }
let numel = Array.fold_left ( * ) 1
let counts w = Array.map (fun (r : M.range) -> r.count) w

let floats (D.Any dt) =
  match D.kind dt with
  | D.Float | D.Complex -> true
  | D.Signed | D.Unsigned | D.Boolean -> false

let identity d = P.of_node ~ins:[| d |] (In 0)

let binary : Nx_kernel.monoid -> P.binary = function
  | Sum -> Add
  | Prod -> Mul
  | Max -> Maximum
  | Min -> Minimum

(* The position of device [k] among [p]'s. *)
let position p k =
  Option.get (Array.find_index (( = ) k) (Grid.devices (Devices.grid p)))

(* The window of operand [i], of [shape], that a device reads by [rule]: the
   window its window [w] of the results reads ({!Route.window}), but [tile f]
   along an axis of the folded factor [f]. *)
let reads rule i shape (w : M.range array) tile =
  match (rule : Route.rule) with
  | Elementwise -> w
  | Axes t | Into t ->
      let window = Route.window rule ~operand:i shape w in
      Array.mapi
        (fun a r ->
          match t.(i).(a) with
          | Route.Reduced f -> tile f
          | Becomes _ | Scanned _ | Regroups _ | Whole -> r)
        window

(* Groups *)

(* The devices of a placement, by position, that hold one window of the
   results off the folded axes, its [key]: its [members]; their [tiles],
   distinct windows along the folded axes that hold terms, in order; and
   the first member holding each, its [holder]. *)
type group = {
  members : int list;
  tiles : M.range array array;
  holder : int array;
}

let groups n ~key ~tile =
  let keys = Array.init n key and tiles = Array.init n tile in
  let all = List.init n Fun.id in
  let group j =
    let members = List.filter (fun i -> keys.(i) = keys.(j)) all in
    let holding =
      List.filter (fun i -> numel (counts tiles.(i)) > 0) members
    in
    let ts =
      Array.of_list
        (List.sort_uniq compare (List.map (fun i -> tiles.(i)) holding))
    in
    let holder =
      Array.map (fun t -> List.find (fun i -> tiles.(i) = t) holding) ts
    in
    { members; tiles = ts; holder }
  in
  let first j = List.find (fun i -> keys.(i) = keys.(j)) all = j in
  List.map group (List.filter first all)

(* Contraction pieces *)

let rec gcd a b = if b = 0 then a else gcd b (a mod b)

(* The ranges [lo, hi) splits into, in order, each within one block of
   every length of [rs] or from one of their common edges to another:
   [lo, hi) itself where [rs] is empty. *)
let pieces rs lo hi =
  let l = List.fold_left (fun l r -> l / gcd l r * r) 1 rs in
  let rec from x =
    if x >= hi then []
    else
      let body = hi / l * l in
      let y =
        if x mod l = 0 && body > x then body
        else List.fold_left (fun y r -> Int.min y (((x / r) + 1) * r)) hi rs
      in
      { M.start = x; count = y - x; step = 1 } :: from y
  in
  from lo

(* The parts of [tile], its ranges along the contracting pairs, each cut
   into {!pieces} of [runs], the pair's scale blocks' lengths, in C
   order. *)
let parts runs (tile : M.range array) =
  let rec from p =
    if p = Array.length tile then [ [] ]
    else
      let rest = from (p + 1) and r = tile.(p) in
      List.concat_map
        (fun x -> List.map (fun u -> x :: u) rest)
        (pieces runs.(p) r.start (r.start + r.count))
  in
  List.map Array.of_list (from 0)

(* Operations of folds *)

(* [op], an operation of a fold on [here], one device, where each of its
   operands lies: by {!Exec.local} where it maps operands that lie beside
   each other, else by windows over the route that reads each operand on
   [here] and puts each result there. One of no operand, a creation, computes
   there too. *)
let apply : type r d. by:string -> d Devices.placement -> r Value.prim -> r =
 fun ~by here op ->
  if Exec.is_local op then Exec.local ~by op
  else
    let here = Devices.rebrand here in
    let (Prim.Operands xs) = Prim.operands op in
    let operands = Array.make (List.length xs) here in
    let route = { Route.operands; result = here; compute = Windows } in
    Exec.windows ~by route (Prim.layouts ~by op) op

(* Folds of partials *)

module Make (A : sig
  val by : string
end) =
struct
  let by = A.by
  let apply here op = apply ~by here op

  (* [op] on the device [x] lies on. *)
  let on x op = apply (Prim.placement x) op

  (* Values *)

  (* [x] on [here]. *)
  let move (type v s d) (here : d Devices.placement) (x : (v, s, d) Value.t) :
      (v, s, d) Value.t =
    if Devices.equal (Prim.placement x) here then x
    else Exec.place ~by here x

  let moves here vs =
    List.map (fun (Value.Any x) -> Value.Any (move here x)) vs

  let reshape (type v s d) shape (x : (v, s, d) Value.t) : (v, s, d) Value.t =
    if Prim.has_shape x shape then x
    else on x (Value.Move (Reshape (Iarray.of_array shape), x))

  (* Elements [start, start + count) of [x] along [axis]. *)
  let slice (type v s d) axis start count (x : (v, s, d) Value.t) :
      (v, s, d) Value.t =
    let r = Array.map full (Prim.shape x) in
    if start = 0 && count = r.(axis).count then x
    else begin
      r.(axis) <- { M.start; count; step = 1 };
      on x (Value.Move (Slice (Iarray.of_array r), x))
    end

  (* [x] of [shape], its axes of extent 1 stretched. *)
  let spread (type v s d) shape (x : (v, s, d) Value.t) : (v, s, d) Value.t =
    if Prim.has_shape x shape then x
    else on x (Value.Move (Broadcast (Iarray.of_array shape), x))

  let cast (type d) (D.Any dt) (Value.Any x : d Value.any) : d Value.any =
    match D.equal_witness (Prim.dtype x) dt with
    | Some Type.Equal -> Value.Any x
    | None -> Value.Any (fst (on x (Prim.op1 ~by Cast dt x)))

  (* [x]'s window [w] from its array on its set's device [k], at [at]. *)
  let view (type v s d) k (at : d Devices.placement) w (x : (v, s, d) Value.t)
      : (v, s, d) Value.t =
    Value.Array { at; a = Place.view ~by x k w; dead = Value.Live }

  let view_load (type d) k at w (l : d Value.load) : d Value.load =
    match l with
    | Plain x -> Plain (view k at w x)
    | Padded p -> Padded { p with x = view k at w p.x }

  (* Output [o] of [prog], of [term], over [loads] in the window [w] of a
     loop of [layout], on [here]: the load itself where [prog] is its
     identity. *)
  let terms (type d) here layout (prog : P.t) o (D.Any term)
      (loads : d Value.load iarray) w : d Value.any =
    match loads with
    | [| Plain x |]
      when P.is_identity prog && D.equal (Prim.dtype x) term ->
        Value.Any x
    | _ ->
        let one =
          P.subst ~outs:[| o |] ~ins:prog.ins (fun _ -> None) prog
        in
        let map = Value.Map { layout; prog = one; outs = Value.[ term ]; loads } in
        Value.Any (fst (apply here (Prim.within map w)))

  (* The value at [p] of [shape] whose array on each of [p]'s devices, in
     order, is [vs]'s. *)
  let at_placement (type d) (p : d Devices.placement) shape
      (vs : d Value.any array) : d Value.any =
    let (Value.Any x) = vs.(0) in
    let dt = Prim.dtype x in
    let array v = Iarray.get (Prim.arrays (Prim.expect dt v)) 0 in
    Value.Any (Prim.of_arrays p ~shape (Iarray.of_array (Array.map array vs)))

  (* The results of [r] at [p], of [shape], from each device's values, in
     [p]'s order. *)
  let result (type d a) (r : (d, a) Value.reduction) p shape
      (per : d Value.any list array) : a =
    let nth i = at_placement p shape (Array.map (fun vs -> List.nth vs i) per) in
    match r with
    | Monoid (_, _, dt) -> Prim.expect dt (nth 0)
    | Logsumexp (_, dt) -> Prim.expect dt (nth 0)
    | Moments (_, dt) -> (Prim.expect dt (nth 0), Prim.expect dt (nth 1))
    | Arg (_, _, dt) -> (Prim.expect dt (nth 0), Prim.expect D.Int64 (nth 1))

  let listed : type d a. (d, a) Value.reduction -> a -> d Value.any list =
    function
    | Monoid _ -> fun v -> [ Value.Any v ]
    | Logsumexp _ -> fun v -> [ Value.Any v ]
    | Moments _ -> fun (m, v) -> [ Value.Any m; Value.Any v ]
    | Arg _ -> fun (e, p) -> [ Value.Any e; Value.Any p ]

  (* Maps *)

  (* The map over [vs], of one shape, its inputs in order, to [exprs]. *)
  let map (type d) (vs : d Value.any list) exprs : d Value.any list =
    let dtype (Value.Any x : d Value.any) = D.Any (Prim.dtype x) in
    let prog = P.of_exprs ~ins:(Iarray.of_list (List.map dtype vs)) (Iarray.of_list exprs) in
    let dts = List.map (P.dtype prog) (Iarray.to_list prog.outs) in
    let (Prim.Outputs (outs, read)) = Prim.outputs dts in
    let (Value.Any x) = List.hd vs in
    let layout = L.contiguous (Iarray.of_array (Prim.shape x)) in
    let plain (Value.Any x : d Value.any) = Value.Plain x in
    let loads = Iarray.of_list (List.map plain vs) in
    read (on x (Value.Map { layout; prog; outs; loads }))

  (* Folds *)

  (* The partials [value i], [i < n], of the terms one device each computes,
     moved to [here] and combined in order by [node] of two, the earlier
     first. *)
  let fold node here value n =
    let acc = ref (moves here (value 0)) in
    for i = 1 to n - 1 do
      acc := node !acc (moves here (value i))
    done;
    !acc

  (* The [n] partials [value i] combined by [node] on [here j] for each member
     [j] of [g], each given to [settle j]. *)
  let combine here g n value node settle =
    List.iter (fun j -> settle j (fold node (here j) value n)) g.members

  (* The tile of each device of [route]'s result, in order: its window along
     [axes] of operand [0], of [shape], where [route] reads it. The route
     reads every operand alike along them. *)
  let tiles (route : 'd Route.t) shape axes =
    let q = route.operands.(0) in
    Array.map
      (fun k ->
        let w = Devices.window ~by q shape (position q k) in
        Array.map (fun a -> w.(a)) axes)
      (Grid.devices (Devices.grid route.result))

  (* Kinds *)

  (* Partials [a] then [b], lists of one length, combined element by
     element by [k]: exact over integers and booleans, one rounding per
     element over floats. *)
  let exact (k : P.binary) a b =
    let n = List.length a in
    let one i (Value.Any x) =
      let dt = Prim.dtype x in
      E.any (E.binary k (E.input i dt) (E.input (i + n) dt))
    in
    map (a @ b) (List.mapi one a)

  (* The mean of [n] terms from their sum [s], in the carrier. *)
  let mean n s =
    let (Value.Any x) = s in
    let c = Prim.dtype x in
    let count = E.const c (D.of_float c (Float.of_int n)) in
    map [ s ] [ E.any (E.fdiv (E.input 0 c) count) ]

  (* [Moments] into [dt] of [n] terms from their mean [m], in the carrier,
     and [ds], the sums of their deviations [d] from it and of [d²]:
     [(Σ d² - Σ d (Σ d / n)) / n], [+0] where that is negative or [Σ d²]
     where that is [+∞]. A NaN mean makes every deviation, so the variance,
     a NaN. *)
  let moments (D.Any dt) n m ds =
    let (Value.Any x) = m in
    let c = Prim.dtype x in
    let m = E.input 0 c and s1 = E.input 1 c and s2 = E.input 2 c in
    let count = E.const c (D.of_float c (Float.of_int n)) in
    let zero = E.const c (D.zero c)
    and inf = E.const c (D.of_float c Float.infinity) in
    let q = E.(fdiv (s2 - (s1 * fdiv s1 count)) count) in
    let v = E.(where (s2 = inf) s2 (where (q < zero) zero q)) in
    map (Value.Any x :: ds) [ E.any (E.cast dt m); E.any (E.cast dt v) ]

  (* The values [x] orders by: itself, or a complex number's parts. *)
  let keys (type v s d) (x : (v, s, d) Value.t) : d Value.any list =
    let parts (type w r) (pt : (w, r) D.t) =
      let pairs = on x (Value.Bitcast (pt, x)) in
      let shape = Prim.shape x in
      let half j = Value.Any (reshape shape (slice (Array.length shape) j 1 pairs)) in
      [ half 0; half 1 ]
    in
    match Prim.dtype x with
    | D.Complex64 -> parts D.Float32
    | D.Complex128 -> parts D.Float64
    | _ -> [ Value.Any x ]

  (* Of two ranked partials of the extreme [e], each its extreme and its
     position in the whole's order, the first extreme: a NaN before any
     number, the smaller position between two NaNs; else the extreme in
     Prog's order, [-0] below [+0] and a complex number by its parts, the
     smaller position between two of one bits. *)
  let first_of (e : Nx_kernel.extreme) a b =
    match (a, b) with
    | [ Value.Any ea; pa ], [ eb; pb ] ->
        let dt = Prim.dtype ea in
        let eb = Prim.expect dt eb in
        let ka = keys ea and kb = keys eb in
        let count = List.length ka in
        let n = 2 + count in
        let (Value.Any k0) = List.hd ka in
        let kdt = Prim.dtype k0 in
        let fl = floats (D.Any kdt) in
        let before x y =
          if fl then E.(x < y || (x = y && recip x < recip y)) else E.(x < y)
        in
        let same x y = if fl then E.(x = y && recip x = recip y) else E.(x = y) in
        let key base j = E.input (base + 2 + j) kdt in
        (* Whether keys of [u] are below those of [v], from key [j]. *)
        let rec below u v j =
          let lt = before (key u j) (key v j) in
          if j = count - 1 then lt
          else
            let rest = below u v (j + 1) in
            E.(lt || (same (key u j) (key v j) && rest))
        in
        let rec tie j =
          let s = same (key 0 j) (key n j) in
          if j = count - 1 then s
          else
            let rest = tie (j + 1) in
            E.(s && rest)
        in
        let past = match e with Max -> below 0 n 0 | Min -> below n 0 0 in
        let at = n + 1 in
        let earlier = E.(input at D.Int64 < input 1 D.Int64) in
        let wins =
          match D.kind dt with
          | D.Float | D.Complex ->
              let nan i = E.(input i dt <> input i dt) in
              let numbers =
                E.(where (nan n) (const D.Bool true) (past || (tie 0 && earlier)))
              in
              E.(where (nan 0) (nan n && earlier) numbers)
          | D.Signed | D.Unsigned | D.Boolean -> E.(past || (tie 0 && earlier))
        in
        let pick i t = E.any (E.where wins (E.input (n + i) t) (E.input i t)) in
        map
          ((Value.Any ea :: pa :: ka) @ (Value.Any eb :: pb :: kb))
          [ pick 0 dt; pick 1 D.Int64 ]
    | _ -> invalid_arg "Partial.first_of: an extreme and its position each"

  (* [pos] shifted by [s]. *)
  let shift s pos =
    if s = 0 then pos
    else
      let s = E.const D.Int64 (Int64.of_int s) in
      List.hd (map [ pos ] [ E.any E.(input 0 D.Int64 + s) ])

  (* [pos], positions in C order of the folded indices of a window [w] of
     the folded extents [whole], as positions in C order of the whole's. *)
  let renumber (w : M.range array) whole pos =
    if Array.for_all2 (fun (r : M.range) n -> r.start = 0 && r.count = n) w whole
    then pos
    else
      let int64 n = E.const D.Int64 (Int64.of_int n) in
      let p = E.input 0 D.Int64 in
      let global = ref (int64 0) and inner = ref 1 and outer = ref 1 in
      for i = Array.length w - 1 downto 0 do
        let digit = E.((p / int64 !inner) mod int64 w.(i).count) in
        global := E.(!global + ((digit + int64 w.(i).start) * int64 !outer));
        inner := !inner * w.(i).count;
        outer := !outer * whole.(i)
      done;
      List.hd (map [ pos ] [ E.any !global ])

  (* Reductions *)

  let reduce (type d r) (route : unit Route.t) rule layout axes prog
      (rs : (d, r) Value.reductions) (loads : d Value.load iarray) : r =
    let shape = Iarray.to_array (L.shape layout)
    and ax = Iarray.to_array axes in
    let p : d Devices.placement = Devices.rebrand route.result in
    let set = Devices.set p and ks = Grid.devices (Devices.grid p) in
    let out = Prim.reduced shape axes in
    let whole = Array.map (fun a -> shape.(a)) ax in
    let outw j = Devices.window ~by p out j in
    let here j = Devices.one set ks.(j) in
    let tiles =
      let (Value.Any x) = Prim.load_any (Iarray.get loads 0) in
      tiles route (Prim.shape x) ax
    in
    let along j a = tiles.(j).(Option.get (Array.find_index (( = ) a) ax)) in
    let loop j =
      let w = outw j in
      let before a =
        Array.fold_left (fun n b -> if b < a then n + 1 else n) 0 ax
      in
      Array.init (Array.length shape) (fun a ->
          if Array.mem a ax then along j a else w.(a - before a))
    in
    let viewed j =
      Iarray.mapi
        (fun i l ->
          let (Value.Any x) = Prim.load_any l in
          view_load ks.(j) (here j) (reads rule i (Prim.shape x) (outw j) (along j)) l)
        loads
    in
    let local j prog rs =
      let op =
        Value.Reduce { layout; axes; prog; reductions = rs; loads = viewed j }
      in
      apply (here j) (Prim.within op (loop j))
    in
    let gs = groups (Array.length ks) ~key:outw ~tile:(Array.get tiles) in
    let term o = P.dtype prog (Iarray.get prog.outs o) in
    let one (type a) (r : (d, a) Value.reduction) : a =
      let per = Array.make (Array.length ks) [] in
      let store settle j vs = per.(j) <- settle j vs in
      (* One partial per tile, by [unit] on its holder, combined in tile
         order on each member and given to [settle] with the group. *)
      let by_tiles unit node settle =
        List.iter
          (fun g ->
            let values = Array.map unit g.holder in
            combine here g (Array.length g.tiles) (Array.get values) node
              (store (settle g)))
          gs
      in
      let (D.Any c) = Nx_kernel.carrier (term (snd (Prim.reduction r))) in
      match r with
      | Monoid (m, o, dt) ->
          let unit j =
            let v, () = local j prog Value.[ Monoid (m, o, c) ] in
            [ Value.Any v ]
          in
          let settle _ _ vs = [ cast (D.Any dt) (List.hd vs) ] in
          by_tiles unit (exact (binary m)) settle;
          result r p out per
      | Arg (e, o, dt) ->
          let (D.Any t) = term o in
          let unit j =
            let (x, q), () = local j prog Value.[ Arg (e, o, t) ] in
            [ Value.Any x; renumber tiles.(j) whole (Value.Any q) ]
          in
          let settle _ _ = function
            | [ x; q ] -> [ cast (D.Any dt) x; q ]
            | _ -> invalid_arg "Partial.reduce: an extreme and its position"
          in
          by_tiles unit (first_of e) settle;
          result r p out per
      | Logsumexp (o, dt) ->
          (* The logsumexp of the tiles' logsumexps, laid out as
             [outputs; tiles]. *)
          let unit j =
            let v, () = local j prog Value.[ Logsumexp (o, c) ] in
            [ Value.Any v ]
          in
          let settle _ j vs =
            let n = numel (counts (outw j)) and t = List.length vs in
            let piece k v =
              let r = [| full n; { M.start = k; count = 1; step = 1 } |] in
              (Iarray.of_array r, reshape [| n; 1 |] (Prim.expect c v))
            in
            let pieces = List.mapi piece vs
            and shape : int iarray = [| n; t |] in
            let x =
              apply (here j)
                (Value.Assemble { dtype = c; shape; fill = D.zero c; pieces })
            in
            let lse, () =
              apply (here j)
                (Value.Reduce
                   {
                     layout = L.contiguous shape;
                     axes = [| 1 |];
                     prog = identity (D.Any c);
                     reductions = Value.[ Logsumexp (0, dt) ];
                     loads = [| Plain x |];
                   })
            in
            [ Value.Any (reshape (counts (outw j)) lse) ]
          in
          by_tiles unit ( @ ) settle;
          result r p out per
      | Moments (o, dt) ->
          (* Two passes: the mean from the tiles' sums, then the tiles'
             sums of the deviations from it and of their squares, each
             combined in tile order. *)
          let (D.Any t) = term o in
          let n = numel whole in
          let xs =
            Array.init (Array.length ks) (fun j ->
                lazy
                  (terms (here j) layout prog o (D.Any t) (viewed j)
                     (loop j)))
          in
          let x j = Prim.expect t (Lazy.force xs.(j)) in
          let over j prog rs loads =
            let layout = L.contiguous (Iarray.of_array (Prim.shape (x j))) in
            apply (here j)
              (Value.Reduce { layout; axes; prog; reductions = rs; loads })
          in
          let total j =
            let s, () =
              over j (identity (D.Any t)) Value.[ Monoid (Sum, 0, c) ]
                [| Plain (x j) |]
            in
            [ Value.Any s ]
          in
          (* [per.(j)] holds the mean the first round gave device [j]. *)
          let centred j =
            let shape = Prim.shape (x j) in
            let units =
              Array.mapi (fun a e -> if Iarray.mem a axes then 1 else e) shape
            in
            let m = Prim.expect c (List.hd per.(j)) in
            let m = spread shape (reshape units m) in
            let d = E.(cast c (input 0 t) - input 1 c) in
            let ins : D.any iarray = [| D.Any t; D.Any c |] in
            let prog = P.of_exprs ~ins [| E.any d; E.any E.(d * d) |] in
            let s1, (s2, ()) =
              over j prog Value.[ Monoid (Sum, 0, c); Monoid (Sum, 1, c) ]
                [| Plain (x j); Plain m |]
            in
            [ Value.Any s1; Value.Any s2 ]
          in
          by_tiles total (exact Add) (fun _ _ s -> mean n (List.hd s));
          by_tiles centred (exact Add) (fun _ j ds ->
              moments (D.Any dt) n (List.hd per.(j)) ds);
          result r p out per
    in
    let rec each : type q. (d, q) Value.reductions -> q = function
      | [] -> ()
      | r :: rest ->
          let a = one r in
          (a, each rest)
    in
    each rs

  (* Scans *)

  let scan (type d a) (route : unit Route.t) rule layout axis prog
      (r : (d, a) Value.reduction) (loads : d Value.load iarray) : a =
    let shape = Iarray.to_array (L.shape layout) in
    let p : d Devices.placement = Devices.rebrand route.result in
    let set = Devices.set p and ks = Grid.devices (Devices.grid p) in
    let win j = Devices.window ~by p shape j in
    let here j = Devices.one set ks.(j) in
    let viewed j =
      let tile _ = invalid_arg "Partial.scan: a reduced axis" in
      Iarray.mapi
        (fun i l ->
          let (Value.Any x) = Prim.load_any l in
          view_load ks.(j) (here j) (reads rule i (Prim.shape x) (win j) tile) l)
        loads
    in
    let o = snd (Prim.reduction r) in
    let term = P.dtype prog (Iarray.get prog.outs o) in
    (* A device's terms over its window. *)
    let own j = terms (here j) layout prog o term (viewed j) (win j) in
    (* [x]'s terms reduced by [rs], their row of extent 1 along the axis. *)
    let total (Value.Any x : d Value.any) rs =
      let w = Prim.shape x in
      let split =
        Array.concat
          [
            Array.sub w 0 axis;
            [| 1; w.(axis) |];
            Array.sub w (axis + 1) (Array.length w - axis - 1);
          ]
      in
      let loads : d Value.load iarray = [| Value.Plain (reshape split x) |] in
      on x
        (Value.Reduce
           {
             layout = L.contiguous (Iarray.of_array split);
             axes = [| axis + 1 |];
             prog = identity term;
             reductions = rs;
             loads;
           })
    in
    let scanned (Value.Any x : d Value.any) reduction carry =
      let prog = identity (D.Any (Prim.dtype x)) in
      let layout = L.contiguous (Iarray.of_array (Prim.shape x)) in
      on x
        (Value.Scan { layout; axis; prog; reduction; loads = [| Plain x |]; carry })
    in
    (* By kind: a tile's partial, of its terms [x] from term [at] of the
       axis, how two combine, the earlier first, and the scan of [x] from the
       partials of the tiles before it. *)
    let partial, node, continued =
      match r with
      | Monoid (m, _, dt) ->
          (* Each tile continues from the tiles before it. *)
          let (D.Any c) = Nx_kernel.carrier term in
          let partial x _ =
            let v, () = total x Value.[ Monoid (m, 0, c) ] in
            [ Value.Any v ]
          in
          let continued carry x _ =
            [ Value.Any (scanned x (Monoid (m, 0, dt)) (Option.map List.hd carry)) ]
          in
          (partial, exact (binary m), continued)
      | Arg (e, _, dt) ->
          (* Each tile scans alone, its first extremes then combined with
             those before it. *)
          let partial x at =
            let (v, q), () = total x Value.[ Arg (e, 0, dt) ] in
            [ Value.Any v; shift at (Value.Any q) ]
          in
          let continued carry x at =
            let v, q = scanned x (Arg (e, 0, dt)) None in
            let local = [ Value.Any v; shift at (Value.Any q) ] in
            match carry with
            | None -> local
            | Some cv ->
                let wide (Value.Any c) = Value.Any (spread (Prim.shape v) c) in
                first_of e (List.map wide cv) local
          in
          (partial, first_of e, continued)
      | Logsumexp _ | Moments _ ->
          invalid_arg "Partial.scan: a scan that reads its axis whole"
    in
    let per = Array.make (Array.length ks) [] in
    let off j =
      let w = Array.copy (win j) in
      w.(axis) <- full 0;
      w
    in
    let tile j = [| (win j).(axis) |] in
    let group g =
      let start t = g.tiles.(t).(0).start in
      let slabs = Array.map (fun j -> lazy (own j)) g.holder in
      let partials =
        Array.mapi (fun t s -> lazy (partial (Lazy.force s) (start t))) slabs
      in
      let member j =
        let w = win j in
        if w.(axis).count = 0 then
          (* An empty window scans alone. *)
          let op =
            Value.Scan { layout; axis; prog; reduction = r; loads = viewed j; carry = None }
          in
          let y = apply (here j) (Prim.within op w) in
          per.(j) <- moves (here j) (listed r y)
        else
          let t = Option.get (Array.find_index (fun tl -> tl.(0) = w.(axis)) g.tiles) in
          let x = if j = g.holder.(t) then Lazy.force slabs.(t) else own j in
          let carry =
            if t = 0 then None
            else Some (fold node (here j) (fun i -> Lazy.force partials.(i)) t)
          in
          per.(j) <- moves (here j) (continued carry x (start t))
      in
      List.iter member g.members
    in
    List.iter group (groups (Array.length ks) ~key:off ~tile);
    result r p shape per

  (* Contractions *)

  let contract (type v s d a b c e) (route : unit Route.t) rule ~batch
      ~contracting ~acc (out : (v, s) D.t) (a : (a, b, d) Value.t)
      (b : (c, e, d) Value.t) (init : (v, s, d) Value.t option)
      (a_scale : d Value.scale option) (b_scale : d Value.scale option) :
      (v, s, d) Value.t =
    let shape =
      Prim.contract_shape ~by ~batch ~contracting ~segments:None ~acc a b init
        a_scale b_scale
    in
    let pairs = Iarray.to_array contracting in
    let p : d Devices.placement = Devices.rebrand route.result in
    let set = Devices.set p and ks = Grid.devices (Devices.grid p) in
    let resw j = Devices.window ~by p shape j in
    let here j = Devices.one set ks.(j) in
    let tiles = tiles route (Prim.shape a) (Array.map fst pairs) in
    (* Operand [i] on device [j], its window along the pairs [r], none for
       [init]. *)
    let read (type w q) j i r (x : (w, q, d) Value.t) =
      let w = reads rule i (Prim.shape x) (resw j) (Array.get r) in
      view ks.(j) (here j) w x
    in
    (* The scale [sc] of [x], operand [i], on device [j]: the blocks that
       [x]'s window along the pairs [r] spans, [side] naming [x]'s axis of a
       pair. *)
    let scale j i side x r (sc : d Value.scale) : d Value.scale =
      let blocks (type w q) (v : (w, q, d) Value.t) =
        let w = reads rule i (Prim.shape v) (resw j) (Array.get r) in
        Array.iteri
          (fun k pr ->
            let ax = side pr in
            let n = Prim.dim x ax / Prim.dim v ax and q = r.(k) in
            let lo = q.M.start / n and hi = (q.start + q.count + n - 1) / n in
            w.(ax) <- { M.start = lo; count = hi - lo; step = 1 })
          pairs;
        view ks.(j) (here j) w v
      in
      match sc with
      | Exponent v -> Exponent (blocks v)
      | Value v -> Value (blocks v)
    in
    let first = if Option.is_some init then 3 else 2 in
    let b_at = first + Bool.to_int (Option.is_some a_scale) in
    let run x ax = function
      | Some sc -> [ Prim.dim x ax / (Prim.scale_shape sc).(ax) ]
      | None -> []
    in
    let runs =
      Array.map (fun (i, k) -> run a i a_scale @ run b k b_scale) pairs
    in
    let (D.Any ad) = acc in
    (* The contraction of a part [r] of a tile on device [j] from no
       [init], in the accumulator. *)
    let part j r =
      [
        Value.Any
          (apply (here j)
             (Value.Contract
                {
                  batch;
                  contracting;
                  acc;
                  out = ad;
                  a = read j 0 r a;
                  b = read j 1 r b;
                  init = None;
                  a_scale = Option.map (scale j first fst a r) a_scale;
                  b_scale = Option.map (scale j b_at snd b r) b_scale;
                  segments = None;
                }));
      ]
    in
    let per = Array.make (Array.length ks) [] in
    let group g =
      let values =
        Array.concat
          (Array.to_list
             (Array.mapi
                (fun t tl ->
                  Array.of_list (List.map (part g.holder.(t)) (parts runs tl)))
                g.tiles))
      in
      let settle j s =
        let s =
          match init with
          | None -> s
          | Some i -> exact Add s [ cast acc (Value.Any (read j 2 [||] i)) ]
        in
        per.(j) <- [ cast (D.Any out) (List.hd s) ]
      in
      combine here g (Array.length values) (Array.get values)
        (exact Add) settle
    in
    let tile = Array.get tiles in
    List.iter group (groups (Array.length ks) ~key:resw ~tile);
    Prim.expect out (at_placement p shape (Array.map List.hd per))
end

let run (type r) ~by route (op : r Value.prim) : r =
  let module F = Make (struct
    let by = by
  end) in
  let rule = Prim.rule op in
  match op with
  | Value.Reduce { layout; axes; prog; reductions; loads } ->
      F.reduce route rule layout axes prog reductions loads
  | Value.Scan { layout; axis; prog; reduction; loads; carry = None } ->
      F.scan route rule layout axis prog reduction loads
  | Value.Contract
      {
        batch;
        contracting;
        acc;
        out;
        a;
        b;
        init;
        a_scale;
        b_scale;
        segments = None;
      } ->
      F.contract route rule ~batch ~contracting ~acc out a b init a_scale
        b_scale
  | Value.Contract { segments = Some _; _ } ->
      invalid_arg "Partial.run: a segmented contraction by partials"
  | Value.Scan { carry = Some _; _ }
  | Value.Map _ | Value.Gather _ | Value.Scatter _ | Value.Sort _
  | Value.Fft _ | Value.Linalg _ | Value.Fold _ | Value.Assemble _
  | Value.Copy _ | Value.Move _ | Value.Bitcast _ | Value.Place _
  | Value.Check _ ->
      invalid_arg "Partial.run: an operation no route computes by partials"
