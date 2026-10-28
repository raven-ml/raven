(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Program
module P = Nx_kernel.Prog
module D = Nx_array.Dtype
module L = Nx_array.Layout
module Move = Nx_array.Move
module Prim = Nx.Prim

type t = {
  name : string;
  tape : Program.t;
  ins : int;
  captures : value array;
  outs : int option array;
  fn : bool;
      (** Whether the tape is a function traced as it is ({!trace}), not a
          derivative. *)
}

let tangible x = Scalar.tangible (D.Any (Nx.dtype x))

(* Linearize *)

let linearize ~name xs f =
  let forms =
    Array.of_list
      (List.filter_map
         (fun (V x as v) -> if tangible x then Some (form v) else None)
         (Array.to_list xs))
  in
  let primals = ref [||] and outs = ref [||] in
  let tape, captures =
    Trace.trace Tape ~name forms (fun ts ->
        let tape =
          if Array.length ts = 0 then None
          else
            let (V t) = ts.(0) in
            Nx.Prim.owner t
        in
        Jvp.run ~name ?tape (fun i ->
            let next = ref 0 in
            let seed (V x as v) =
              if not (tangible x) then v
              else begin
                let t = ts.(!next) in
                incr next;
                V (Jvp.seed i x (like x t))
              end
            in
            let ys = f (Array.map seed xs) in
            primals := Array.map (fun (V y) -> V (Jvp.primal i y)) ys;
            let tangents = ref [] and k = ref 0 in
            outs :=
              Array.map
                (fun (V y) ->
                  match Jvp.tangent_of i y with
                  | None -> None
                  | Some t ->
                      tangents := V t :: !tangents;
                      incr k;
                      Some (!k - 1))
                ys;
            Array.of_list (List.rev !tangents)))
  in
  ( !primals,
    { name; tape; ins = Array.length forms; captures; outs = !outs; fn = false }
  )

(* [f] traced over stand-ins of [xs]: linear in them where it is linear. *)
let trace ~name xs f =
  let tape, captures = Trace.trace Tape ~name (Array.map form xs) f in
  let outs = Array.init (Array.length tape.outputs) Option.some in
  { name; tape; ins = Array.length xs; captures; outs; fn = true }

let apply l ts =
  if Array.length ts <> l.ins then
    invalid_arg
      (Printf.sprintf "%s: %d tangents for %d parameters" l.name
         (Array.length ts) l.ins);
  let out = eval l.tape (Array.append ts l.captures) in
  Array.map (Option.map (fun j -> out.(j))) l.outs

(* Brands

   A cotangent is held as a value of any brand and read back at the brand of the
   values it meets, which lie on its set. *)

(* [x] at [r]'s brand. *)
let rebrand (type v s w t d e) (r : (w, t, e) Nx.t) (x : (v, s, d) Nx.t) :
    (v, s, e) Nx.t =
  let set = Nx.Placement.devices (Nx.placement r) in
  match Prim.coerce (Nx.dtype x) set x with
  | Some y -> y
  | None ->
      invalid_arg
        (Format.asprintf "Rune: a cotangent on %a meets a value on %a"
           Nx.Placement.pp (Nx.placement x) Nx.Placement.pp (Nx.placement r))

(* [x] at [dt]. *)
let with_dtype (type v s w t e) (dt : (v, s) D.t) (x : (w, t, e) Nx.t) :
    (v, s, e) Nx.t =
  match D.equal_witness (Nx.dtype x) dt with
  | Some Type.Equal -> x
  | None ->
      invalid_arg
        (Format.asprintf "Rune: a %a cotangent where %a was expected"
           Nx.pp_dtype (Nx.dtype x) Nx.pp_dtype dt)

(* The value whose brand [vs] meet at: the first. *)
let reference = function [] -> invalid_arg "Rune: no value" | v :: _ -> v

(* A contribution, and whether it is memory of its own in C order. *)
type ct = { v : value; fresh : bool }

let fresh v = { v; fresh = true }
let held v = { v; fresh = false }

let sum a b =
  match reference [ a.v; b.v ] with
  | V r ->
      let (V x) = a.v in
      let (V y) = b.v in
      fresh (V (Nx.add (rebrand r x) (with_dtype (Nx.dtype x) (rebrand r y))))

(* Outputs of dtypes known at run time *)

type 'd outs = Outs : ('d, 'r) Prim.outs * ('r -> 'd Prim.any list) -> 'd outs

let rec outs_of : type d. D.any list -> d outs = function
  | [] -> Outs ([], fun () -> [])
  | D.Any dt :: rest ->
      let (Outs (o, read)) = outs_of rest in
      Outs (dt :: o, fun (y, r) -> Prim.Any y :: read r)

let rec out_dtypes : type d r. (d, r) Prim.outs -> D.any list = function
  | [] -> []
  | dt :: rest -> D.Any dt :: out_dtypes rest

(* Transposes *)

(* What a rule reads: whether an operand is linear, a constant operand's value,
   and where a contribution goes. *)
type env = {
  name : string;
  lin : 'v 's 'd. ('v, 's, 'd) Nx.t -> bool;
  const : 'v 's 'd. ('v, 's, 'd) Nx.t -> ('v, 's, 'd) Nx.t;
  give : 'v 's 'd. ('v, 's, 'd) Nx.t -> ct -> unit;
  fn : bool;
}

(* Only a custom_jvp tangent map puts on a derivative's tape an operation that
   is not linear in its tangents; on a function's, the function does. *)
let nonlinear env what =
  if env.fn then
    invalid_arg
      (Printf.sprintf "%s: %s is not linear in the function's argument" env.name
         what)
  else
    invalid_arg
      (Printf.sprintf
         "%s: a custom_jvp tangent map applies %s to a tangent; a tangent map \
          must be linear in its tangents"
         env.name what)

let shape_of layout = Iarray.to_array (L.shape layout)
let identity dt = P.of_node ~ins:[| dt |] (Op1 (Copy, dt, 0))

let load_lin env : 'd Prim.load -> bool = function
  | Plain x -> env.lin x
  | Padded { x; _ } -> env.lin x

let load_value env : 'd Prim.load -> value = function
  | Plain x -> V (env.const x)
  | Padded { x; _ } -> V (env.const x)

(* A constant load at [r]'s brand. *)
let load_at env r : 'd Prim.load -> 'e Prim.load = function
  | Plain x -> Plain (rebrand r (env.const x))
  | Padded { x; fill; pad } -> Padded { x = rebrand r (env.const x); fill; pad }

(* A loop's transpose: its program, the constant loads it reads and the linear
   ones that receive a contribution, by position. *)
type plan = { tp : P.t option; constants : int array; received : int array }

let positions flags =
  Array.of_list
    (List.filter (fun j -> flags.(j)) (List.init (Array.length flags) Fun.id))

let transpose_prog env prog linear cts =
  let tp, received =
    try Scalar.transpose prog ~linear ~cts
    with Scalar.Nonlinear k -> nonlinear env k
  in
  {
    tp;
    constants = positions (Array.map not linear);
    received = positions received;
  }

(* Plans per domain, found as jvp's are: the last one, then a table emptied when
   it holds [entries]. A key is the program, its linear loads as bits and each
   cotangent's output and dtype as a number. *)
type key = { p : P.t; mask : int; codes : int array }

type cache = {
  mutable last : (key * plan) option;
  table : (key, plan) Hashtbl.t;
}

let entries = 4096

let cache =
  Domain.DLS.new_key (fun () -> { last = None; table = Hashtbl.create 64 })

let code (k, D.Any dt) = (k lsl 8) lor D.code dt

let same_codes codes cts =
  let n = Array.length codes in
  n = Array.length cts
  &&
  let rec go k = k = n || (codes.(k) = code cts.(k) && go (k + 1)) in
  go 0

let transposed env prog linear cts =
  let n = Array.length linear in
  if n >= Sys.int_size - 1 then transpose_prog env prog linear cts
  else
    let mask = ref 0 in
    for j = n - 1 downto 0 do
      mask := (!mask lsl 1) lor Bool.to_int linear.(j)
    done;
    let mask = !mask in
    let c = Domain.DLS.get cache in
    match c.last with
    | Some (k, plan) when k.p == prog && k.mask = mask && same_codes k.codes cts
      ->
        plan
    | _ ->
        let key = { p = prog; mask; codes = Array.map code cts } in
        let plan =
          match Hashtbl.find_opt c.table key with
          | Some plan -> plan
          | None ->
              if Hashtbl.length c.table >= entries then Hashtbl.reset c.table;
              let plan = transpose_prog env prog linear cts in
              Hashtbl.add c.table key plan;
              plan
        in
        c.last <- Some (key, plan);
        plan

(* Loops *)

(* The transpose of a loop over [loads] whose outputs [k] receive the cotangents
   [cts], each of the loop's shape: one map over the constant loads and the
   cotangents, each of its results a contribution to a linear load, a padded
   one's folded back to its value's shape. *)
let sweep (type d) env layout prog (loads : d Prim.load iarray) cts =
  let n = Iarray.length loads in
  let linear = Array.init n (fun j -> load_lin env (Iarray.get loads j)) in
  let spec = Array.map (fun (k, dt, _) -> (k, dt)) cts in
  match transposed env prog linear spec with
  | { tp = None; _ } -> ()
  | { tp = Some tp; constants; received } -> (
      (* The constants and the cotangents meet at the first one's brand. *)
      let first =
        if Array.length constants > 0 then
          load_value env (Iarray.get loads constants.(0))
        else
          let _, _, c = cts.(0) in
          c
      in
      match first with
      | V r -> (
          let nc = Array.length constants in
          let loads' =
            Iarray.init
              (nc + Array.length cts)
              (fun i ->
                if i < nc then load_at env r (Iarray.get loads constants.(i))
                else
                  let _, _, V c = cts.(i - nc) in
                  Prim.Plain (rebrand r c))
          in
          let give l (Prim.Any y) =
            match l with
            | Prim.Plain x -> env.give x (fresh (V y))
            | Padded { x; pad; _ } ->
                let shape = Iarray.of_array (Nx.shape x) in
                let y = Prim.eval ~by:env.name (Fold { shape; pad; x = y }) in
                env.give x (fresh (V y))
          in
          let map outs =
            Prim.Map { layout; prog = tp; outs; loads = loads'; at = None }
          in
          match received with
          | [| j |] -> (
              (* One contribution, the common case, without a list. *)
              match Iarray.get loads j with
              | Prim.Plain x as l ->
                  let y, () = Prim.eval ~by:env.name (map [ Nx.dtype x ]) in
                  give l (Prim.Any y)
              | Padded { x; _ } as l ->
                  let y, () = Prim.eval ~by:env.name (map [ Nx.dtype x ]) in
                  give l (Prim.Any y))
          | outs ->
              let outs = Array.map (Iarray.get loads) outs in
              let dts =
                Array.to_list
                  (Array.map
                     (function
                       | Prim.Plain x -> D.Any (Nx.dtype x)
                       | Padded { x; _ } -> D.Any (Nx.dtype x))
                     outs)
              in
              let (Outs (o, read)) = outs_of dts in
              let ys = read (Prim.eval ~by:env.name (map o)) in
              List.iteri (fun k y -> give outs.(k) y) ys))

(* A reduction's cotangent over the loop, along the reduced axes broadcast. *)
let spread layout axes (V c) =
  let loop = shape_of layout in
  let kept = Array.mapi (fun a e -> if Iarray.mem a axes then 1 else e) loop in
  V (Nx.broadcast_to loop (Nx.reshape kept c))

(* The cotangents of a reduction's outputs, from those of its results [cts pos]:
   a sum's spread, a mean's divided by the count first. *)
let reduce_cts (type d r) env layout axes (rs : (d, r) Prim.reductions) cts =
  let loop = shape_of layout in
  let n = Iarray.fold_left (fun n a -> n * loop.(a)) 1 axes in
  let rec go : type r. (d, r) Prim.reductions -> int -> _ list =
   fun rs pos ->
    match rs with
    | [] -> []
    | Monoid (Sum, k, dt) :: rest -> (
        let rest = go rest (pos + 1) in
        match cts pos with
        | None -> rest
        | Some c -> (k, D.Any dt, spread layout axes c) :: rest)
    | Moments (k, dt) :: rest -> (
        if Option.is_some (cts (pos + 1)) then nonlinear env "a variance";
        let rest = go rest (pos + 2) in
        match cts pos with
        | None -> rest
        | Some (V c) ->
            (* Divided in the carrier of [c]'s dtype, which counts more terms
               than a narrow float holds, and rounded once. *)
            let cd = Nx.dtype c in
            let (D.Any w) = Nx_kernel.carrier (D.Any cd) in
            let count = D.of_float w (Float.of_int n) in
            let mean = Nx.cast cd (Nx.div_s (Nx.cast w c) count) in
            (k, D.Any dt, spread layout axes (V mean)) :: rest)
    | Monoid (Prod, _, _) :: rest -> skip "a product" rest pos 1
    | Monoid ((Max | Min), _, _) :: rest -> skip "an extreme" rest pos 1
    | Logsumexp _ :: rest -> skip "a log-sum-exp" rest pos 1
    | Arg _ :: rest -> skip "an extreme" rest pos 2
  and skip : type r. string -> (d, r) Prim.reductions -> int -> int -> _ =
   fun what rest pos n ->
    if Option.is_some (cts pos) then nonlinear env what;
    go rest (pos + n)
  in
  go rs 0

(* The additive identity that keeps every sign. *)
let neg_zero (type v s) (dt : (v, s) D.t) : v =
  match D.kind dt with
  | Float -> -0.
  | Complex -> { Complex.re = -0.; im = -0. }
  | Signed | Unsigned | Boolean -> D.zero dt

(* The sums of [w]'s elements from each index on, along [axis]: a scan of the
   axis reversed, carried from [-0.], so that a sum of one term is that term,
   sign of zero included. *)
let from_each (type v s e) ~by ~axis (w : (v, s, e) Nx.t) : (v, s, e) Nx.t =
  let dt = Nx.dtype w in
  let reversed = Nx.flip ~axes:[ axis ] w in
  let shape = Nx.shape reversed in
  let one = Array.mapi (fun a d -> if a = axis then 1 else d) shape in
  let (D.Any c) = Nx_kernel.carrier (D.Any dt) in
  let here = Nx.beside w in
  let carry = Nx.full_at here c one (neg_zero c) in
  let r =
    Prim.eval ~by
      (Scan
         {
           layout = L.contiguous (Iarray.of_array shape);
           axis;
           prog = identity (D.Any dt);
           reduction = Monoid (Sum, 0, dt);
           loads = [| Plain reversed |];
           carry = Some (Any carry);
         })
  in
  Nx.flip ~axes:[ axis ] r

let scan_rule (type d r) env layout axis prog
    (reduction : (d, r) Prim.reduction) (loads : d Prim.load iarray)
    (carry : d Prim.any option) cts =
  let running k dt w =
    let r = from_each ~by:env.name ~axis w in
    sweep env layout prog loads [| (k, dt, V r) |];
    r
  in
  match reduction with
  | Monoid (Sum, k, dt) ->
      Option.iter
        (fun (V w) ->
          let r = running k (D.Any dt) w in
          (* The carry is a term of every element along the axis: its cotangent
             is the first slice's. *)
          Option.iter
            (fun (Prim.Any c) ->
              if env.lin c && Nx.dim axis r > 0 then
                let first =
                  Nx.slice
                    (List.init (Nx.ndim r) (fun a ->
                         if a = axis then Nx.R (0, 1) else Nx.A))
                    r
                in
                env.give c (held (V (Nx.cast (Nx.dtype c) first))))
            carry)
        (cts 0)
  | Moments (k, dt) ->
      if Option.is_some (cts 1) then nonlinear env "a running variance";
      Option.iter
        (fun (V w) ->
          let n = Nx.dim axis w and r = Nx.ndim w in
          let along = Array.init r (fun a -> if a = axis then n else 1) in
          let here = Nx.beside w in
          (* Each cotangent divided by its count and summed from each index on
             in the carrier of its dtype, which counts more terms than a
             narrow float holds, and rounded once. *)
          let cd = Nx.dtype w in
          let (D.Any c) = Nx_kernel.carrier (D.Any cd) in
          let count = Nx.arange_at here Nx.int32 1 (n + 1) 1 in
          let count = Nx.reshape along (Nx.cast c count) in
          let mean = Nx.div (Nx.cast c w) count in
          let r = Nx.cast cd (from_each ~by:env.name ~axis mean) in
          sweep env layout prog loads [| (k, D.Any dt, V r) |])
        (cts 0)
  | Monoid ((Prod | Max | Min), _, _) | Logsumexp _ | Arg _ ->
      if Option.is_some (cts 0) then
        nonlinear env "a running product or extreme"

(* Indexing *)

(* The positions of [x]'s elements in C order, in its shape, whole on each of
   its devices. *)
let iota x =
  let here = Nx.beside x in
  Nx.reshape (Nx.shape x)
    (Nx.arange_at here Nx.int64 0 (Array.fold_left ( * ) 1 (Nx.shape x)) 1)

let gather ~by ~axis idx x = Prim.eval ~by (Gather { axis; idx; x })

let scatter ~by combine ~unique ~axis idx updates into =
  Prim.eval ~by (Scatter { combine; unique; axis; idx; updates; into })

let gather_rule env ~axis idx x (V w) =
  match reference [ V (env.const idx); V w ] with
  | V r ->
      let idx = rebrand r (env.const idx) in
      let ct = rebrand r (with_dtype (Nx.dtype x) w) in
      let into = rebrand r (Nx.zeros_like x) in
      let y = scatter ~by:env.name Add ~unique:false ~axis idx ct into in
      env.give x (fresh (V y))

let scatter_rule env (combine : Nx_kernel.combine) ~unique ~axis idx updates
    into (V w) =
  match reference [ V (env.const idx); V w ] with
  | V r -> (
      let by = env.name in
      let idx = rebrand r (env.const idx) in
      let ct = rebrand r (with_dtype (Nx.dtype into) w) in
      let zeros x = rebrand r (Nx.zeros_like x) in
      match combine with
      | Add ->
          if env.lin updates then
            env.give updates (fresh (V (gather ~by ~axis idx ct)));
          if env.lin into then env.give into (held (V ct))
      | Set ->
          if env.lin into then begin
            let y = scatter ~by Set ~unique ~axis idx (zeros updates) ct in
            env.give into (fresh (V y))
          end;
          if env.lin updates then begin
            let g = gather ~by ~axis idx ct in
            if unique then env.give updates (fresh (V g))
            else
              (* Of the updates to one target, the last in C order wins. *)
              let iota = rebrand r (iota updates) in
              let at = Nx.placement into and s = Nx.shape into in
              let none = rebrand r (Nx.full_at at Nx.int64 s (-1L)) in
              let won = scatter ~by Set ~unique:false ~axis idx iota none in
              let wins = Nx.equal (gather ~by ~axis idx won) iota in
              env.give updates (fresh (V (Nx.where wins g (Nx.zeros_like g))))
          end
      | Max | Min ->
          if env.lin updates || env.lin into then
            nonlinear env "a scatter by extreme")

(* Movements *)

let window ~axis ~step : Move.window = { axis; size = 1; step; dilation = 1 }

(* A slice's transpose: its cotangent read back in the slice's order, then
   folded into the operand's shape, as the adjoint of the load that crops each
   axis to the slice's span and windows it by its step. *)
let unslice env x (ranges : Move.range iarray) w =
  let s = Nx.shape x and r = Nx.ndim x in
  let ranges = Iarray.to_array ranges in
  let reversed =
    List.filter
      (fun a -> ranges.(a).step < 0 && ranges.(a).count > 1)
      (List.init r Fun.id)
  in
  let w = if reversed = [] then w else Nx.flip ~axes:reversed w in
  let lo = Array.make r 0 and hi = Array.make r 0 in
  let windows = ref [] in
  Array.iteri
    (fun a ({ start; count; step } : Move.range) ->
      let step = if count > 1 then Int.abs step else 1 in
      let first =
        if List.mem a reversed then start + ((count - 1) * ranges.(a).step)
        else start
      in
      let span = if count = 0 then 0 else ((count - 1) * step) + 1 in
      lo.(a) <- -first;
      hi.(a) <- -(s.(a) - first - span);
      if step > 1 then windows := window ~axis:a ~step :: !windows)
    ranges;
  let windows = List.rev !windows in
  let pad : Nx_kernel.pad =
    {
      lo = Iarray.of_array lo;
      hi = Iarray.of_array hi;
      interior = Iarray.init r (fun _ -> 0);
      windows = Iarray.of_list windows;
    }
  in
  let w =
    if windows = [] then w
    else
      Nx.reshape
        (Array.append (Nx.shape w) (Array.make (List.length windows) 1))
        w
  in
  let shape = Iarray.of_array s in
  ignore env;
  Prim.eval ~by:env.name (Fold { shape; pad; x = w })

let move_rule env (mv : Move.t) x (V w) =
  let w = with_dtype (Nx.dtype x) w and s = Nx.shape x in
  let c =
    match mv with
    | Reshape _ -> held (V (Nx.reshape s w))
    | Permute p ->
        let inv = Array.make (Iarray.length p) 0 in
        Iarray.iteri (fun i q -> inv.(q) <- i) p;
        held (V (Nx.transpose ~axes:(Array.to_list inv) w))
    | Broadcast b ->
        (* Summed over each axis that repeats the operand. *)
        let b = Iarray.to_array b in
        let lead = Array.length b - Array.length s in
        let repeats a = b.(a) <> 1 && (a < lead || s.(a - lead) = 1) in
        let axes = List.filter repeats (List.init (Array.length b) Fun.id) in
        if axes = [] then held (V (Nx.reshape s w))
        else fresh (V (Nx.reshape s (Nx.sum ~axes w)))
    | Slice ranges -> fresh (V (unslice env x ranges w))
    | Window windows ->
        let r = Nx.ndim x in
        let zeros = Iarray.init r (fun _ -> 0) in
        let pad : Nx_kernel.pad =
          { lo = zeros; hi = zeros; interior = zeros; windows }
        in
        let shape = Iarray.of_array s in
        fresh (V (Prim.eval ~by:env.name (Fold { shape; pad; x = w })))
  in
  env.give x c

(* Assemblies *)

let interval ({ start; count; step } : Move.range) =
  let last = start + ((count - 1) * step) in
  (Int.min start last, Int.max start last)

(* Whether two regions may share an element: their spans meet on every axis. *)
let overlap a b =
  Iarray.for_all2
    (fun (x : Move.range) (y : Move.range) ->
      x.count > 0 && y.count > 0
      &&
      let x0, x1 = interval x and y0, y1 = interval y in
      x0 <= y1 && y0 <= x1)
    a b

let assemble_rule (type v s) env (dtype : (v, s) D.t) shape pieces (V w) =
  let w = with_dtype dtype w in
  let whole =
    Iarray.map
      (fun d -> ({ start = 0; count = d; step = 1 } : Move.range))
      shape
  in
  (* Each piece's region of the cotangent, from the last piece back, each region
     zeroed for the pieces before that it overlaps. *)
  let rec go running = function
    | [] -> ()
    | (region, x) :: earlier ->
        if env.lin x then
          env.give x
            (held (V (Prim.eval ~by:env.name (Move (Slice region, running)))));
        let running =
          if List.exists (fun (r, _) -> overlap r region) earlier then
            let counts = Iarray.map (fun (r : Move.range) -> r.count) region in
            let here = Nx.beside running in
            let zeros = Nx.zeros_at here dtype (Iarray.to_array counts) in
            Prim.eval ~by:env.name
              (Assemble
                 {
                   dtype;
                   shape;
                   fill = D.zero dtype;
                   pieces = [ (whole, running); (region, zeros) ];
                 })
          else running
        in
        go running earlier
  in
  go w (List.rev pieces)

(* Contractions

   [y[B, I, J] = Σ_K a[B, I, K] b[B, K, J]] is linear in each operand: the
   cotangent of one is the contraction of the other with [y]'s, its axes moved
   back to the operand's. A segmented contraction's transposes are segmented by
   the same offsets: [a]'s of [Rows] by rows, [b]'s by contracting each group's
   rows, and each of [Contracting]'s by rows of the groups' axis. *)

(* [y] whose axis [r] is the operand's axis [order.(r)], in the operand's
   order. *)
let reorder order y =
  let perm = Array.make (Array.length order) 0 in
  Array.iteri (fun r q -> perm.(q) <- r) order;
  Nx.transpose ~axes:(Array.to_list perm) y

(* A block scale's factors, each repeated over its block of [shape], in [dt]. *)
let expanded (type v s e) (dt : (v, s) D.t) shape (scale : e Prim.scale) :
    (v, s, e) Nx.t =
  let blocks : (float, D.float64_elt, e) Nx.t =
    match scale with
    | Value s -> Nx.cast Nx.float64 s
    | Exponent e ->
        (* Code [e] scales by [2^(e - 127)], and [255] is NaN. *)
        let e = Nx.cast Nx.float64 e in
        Nx.where
          (Nx.equal e (Nx.scalar_like e 255.))
          (Nx.full_like e Float.nan)
          (Nx.exp2 (Nx.sub e (Nx.scalar_like e 127.)))
  in
  let es = Nx.shape blocks in
  let r = Array.length shape in
  let per a = if es.(a) = 0 then 0 else shape.(a) / es.(a) in
  let split = Array.concat (List.init r (fun a -> [| es.(a); 1 |])) in
  let wide = Array.concat (List.init r (fun a -> [| es.(a); per a |])) in
  Nx.cast dt (Nx.reshape shape (Nx.broadcast_to wide (Nx.reshape split blocks)))

let position x l =
  let rec go i = function
    | [] -> invalid_arg "Rune: an axis outside a contraction"
    | y :: l -> if x = y then i else go (i + 1) l
  in
  go 0 l

let scale_values env : 'd Prim.scale option -> value list = function
  | Some (Value s) -> [ V (env.const s) ]
  | Some (Exponent e) -> [ V (env.const e) ]
  | None -> []

let scale_at env r : 'd Prim.scale option -> 'e Prim.scale option = function
  | Some (Value s) -> Some (Value (rebrand r (env.const s)))
  | Some (Exponent e) -> Some (Exponent (rebrand r (env.const e)))
  | None -> None

let contract_rule env ~batch ~con ~acc ~a ~b ~a_scale ~b_scale ~segments w =
  let batch = Iarray.to_list batch and con = Iarray.to_list con in
  let nb = List.length batch in
  let paired side x l = List.exists (fun p -> side p = x) l in
  let rows_b =
    match segments with Some (Nx_kernel.Rows r) -> Some r.b_axis | _ -> None
  in
  let free side rank skip =
    List.filter
      (fun x ->
        (not (paired side x batch || paired side x con)) && Some x <> skip)
      (List.init rank Fun.id)
  in
  let ra = Nx.ndim a and rb = Nx.ndim b in
  let a_free = free fst ra None and b_free = free snd rb rows_b in
  let a_con = List.filter (fun x -> paired fst x con) (List.init ra Fun.id) in
  let b_con = List.filter (fun x -> paired snd x con) (List.init rb Fun.id) in
  let to_b x = snd (List.find (fun (y, _) -> y = x) con) in
  let to_a x = fst (List.find (fun (_, y) -> y = x) con) in
  let lead =
    match segments with Some (Nx_kernel.Contracting _) -> 1 | _ -> 0
  in
  (* The cotangent's axes: the groups', the batch, then [a]'s and [b]'s free
     ones. *)
  let na = List.length a_free in
  let ct_batch i = lead + i and ct_a m = lead + nb + m in
  let ct_b m = lead + nb + na + m in
  let batch_a = List.map fst batch and batch_b = List.map snd batch in
  let offsets =
    match segments with
    | Some (Nx_kernel.Rows r) -> [ V (env.const r.offsets) ]
    | Some (Contracting r) -> [ V (env.const r.offsets) ]
    | None -> []
  in
  if env.lin a && env.lin b then
    nonlinear env "a contraction with another tangent";
  let other = if env.lin a then V (env.const b) else V (env.const a) in
  match
    reference
      ((w :: other :: offsets) @ scale_values env a_scale
     @ scale_values env b_scale)
  with
  | V r ->
      let by = env.name in
      let offsets =
        match segments with
        | Some (Nx_kernel.Rows s) -> rebrand r (env.const s.offsets)
        | Some (Contracting s) -> rebrand r (env.const s.offsets)
        | None ->
            Nx.zeros_at (Nx.beside r) Nx.int64 [| 0 |]
      in
      let contract ~batch ~contracting ~segments x x_scale y y_scale out =
        Prim.eval ~by
          (Contract
             {
               batch = Iarray.of_list batch;
               contracting = Iarray.of_list contracting;
               acc;
               out;
               a = x;
               b = y;
               init = None;
               a_scale = x_scale;
               b_scale = y_scale;
               segments;
             })
      in
      (* A contraction's result is its own memory in C order, its axes reordered
         a view of it. *)
      let give x scale order y =
        let order = Array.of_list order in
        let fresh =
          ref
            (Array.for_all2 ( = ) order
               (Array.init (Array.length order) Fun.id))
        in
        let y = reorder order y in
        let y =
          match scale_at env r scale with
          | None -> y
          | Some s ->
              fresh := true;
              Nx.mul y (expanded (Nx.dtype x) (Nx.shape x) s)
        in
        env.give x { v = V y; fresh = !fresh }
      in
      let (V w) = w in
      if env.lin a then begin
        let b' = rebrand r (env.const b) in
        let ct = rebrand r w in
        let pairs = List.mapi (fun i bi -> (ct_batch i, bi)) batch_b in
        let meets = List.mapi (fun m bx -> (ct_b m, bx)) b_free in
        let bs = scale_at env r b_scale in
        match segments with
        | None ->
            let y =
              contract ~batch:pairs ~contracting:meets ~segments:None ct None b'
                bs (Nx.dtype a)
            in
            give a a_scale (batch_a @ a_free @ List.map to_a b_con) y
        | Some (Rows s) ->
            let segments =
              Some
                (Nx_kernel.Rows
                   {
                     offsets;
                     a_axis = ct_a (position s.a_axis a_free);
                     b_axis = s.b_axis;
                   })
            in
            let y =
              contract ~batch:pairs ~contracting:meets ~segments ct None b' bs
                (Nx.dtype a)
            in
            give a a_scale (batch_a @ a_free @ List.map to_a b_con) y
        | Some (Contracting s) ->
            (* Each index of the pair reads its group's slice of [y]. *)
            let segments =
              Some
                (Nx_kernel.Rows
                   { offsets; a_axis = snd (List.nth con s.pair); b_axis = 0 })
            in
            let flip l = List.map (fun (c, x) -> (x, c)) l in
            let y =
              contract ~batch:(flip pairs) ~contracting:(flip meets) ~segments
                b' bs ct None (Nx.dtype a)
            in
            give a a_scale (batch_a @ List.map to_a b_con @ a_free) y
      end
      else if env.lin b then begin
        let a' = rebrand r (env.const a) in
        let ct = rebrand r w in
        let pairs = List.mapi (fun i ai -> (ai, ct_batch i)) batch_a in
        let meets = List.mapi (fun m ax -> (ax, ct_a m)) a_free in
        let segments, groups =
          match segments with
          | None -> (None, [])
          | Some (Rows s) ->
              (* Each group's rows sum into its slice of [b]. *)
              ( Some
                  (Nx_kernel.Contracting
                     { offsets; pair = position s.a_axis a_free }),
                [ s.b_axis ] )
          | Some (Contracting s) ->
              ( Some
                  (Nx_kernel.Rows
                     { offsets; a_axis = fst (List.nth con s.pair); b_axis = 0 }),
                [] )
        in
        let y =
          contract ~batch:pairs ~contracting:meets ~segments a'
            (scale_at env r a_scale) ct None (Nx.dtype b)
        in
        give b b_scale (groups @ batch_b @ List.map to_b a_con @ b_free) y
      end

(* Fourier transforms

   With [<u, v> = Re Σ u v], a complex transform's transpose is the same
   transform, its matrix symmetric. Each bin of a real transform but bin 0 and,
   for an even [n], bin [n/2] stands for two points, a conjugate pair. *)

(* [c] at each of [n/2 + 1] bins along [x]'s [axis], in [x]'s dtype: [1] at bin
   0 and, for an even [n], bin [n/2], [c] at the others. *)
let bins (type v s d) ~axis (x : (v, s, d) Nx.t) n c : (v, s, d) Nx.t =
  let k = (n / 2) + 1 and r = Nx.ndim x in
  let here = Nx.beside x in
  let j = Nx.arange_at here Nx.int32 0 k 1 in
  let at b = Nx.equal_s j (Int32.of_int b) in
  let ends = if n mod 2 = 0 then Nx.logical_or (at 0) (at (n / 2)) else at 0 in
  let w =
    Nx.where ends
      (Nx.ones_at here Nx.float64 [| k |])
      (Nx.full_at here Nx.float64 [| k |] c)
  in
  let shape = Array.init r (fun a -> if a = axis then k else 1) in
  Nx.reshape shape (Nx.cast (Nx.dtype x) w)

let last axes = Iarray.get axes (Iarray.length axes - 1)

let fft_rule (type d r) env (f : (d, r) Prim.fft) (V w) =
  let by = env.name in
  match f with
  | C2c { direction; axes; x } ->
      if env.lin x then
        let ct = with_dtype (Nx.dtype x) w in
        env.give x
          (fresh (V (Prim.eval ~by (Fft (C2c { direction; axes; x = ct })))))
  | R2c { dtype; axes; x } ->
      if env.lin x then
        (* [x̄_j = Re Σ_k w_k e^(-2πi jk/n)] over the bins: the real inverse
           transform of [conj w], bin [k] weighted by [1 / c_k]. *)
        let n = Nx.dim (last axes) x in
        let v = Nx.conjugate (with_dtype dtype w) in
        let v = Nx.mul v (bins ~axis:(last axes) v n 0.5) in
        let dtype = Nx.dtype x in
        env.give x
          (fresh (V (Prim.eval ~by (Fft (C2r { dtype; n; axes; x = v })))))
  | C2r { dtype; n; axes; x } ->
      if env.lin x then
        (* [x̄_k = c_k Σ_j w_j e^(+2πi jk/n)]: the conjugate of the forward real
           transform of [w], bin [k] weighted by [c_k]. *)
        let y =
          Prim.eval ~by
            (Fft (R2c { dtype = Nx.dtype x; axes; x = with_dtype dtype w }))
        in
        let y = Nx.conjugate y in
        env.give x (fresh (V (Nx.mul y (bins ~axis:(last axes) y n 2.))))

(* Complex values and their parts' pairs: bitcasts between them are linear. *)
let same_coordinates (D.Any a) (D.Any b) =
  match (a, b) with
  | D.Complex64, D.Float32 | D.Float32, D.Complex64 -> true
  | D.Complex128, D.Float64 | D.Float64, D.Complex128 -> true
  | _ -> false

(* The rules *)

let rule (type r) env (op : r Prim.t) (cts : int -> value option) =
  let by = env.name in
  let first f = Option.iter f (cts 0) in
  match op with
  | Map { layout; prog; outs; loads; _ } ->
      let dts = Array.of_list (out_dtypes outs) in
      let given =
        List.filter_map
          (fun k -> Option.map (fun c -> (k, dts.(k), c)) (cts k))
          (List.init (Array.length dts) Fun.id)
      in
      if given <> [] then sweep env layout prog loads (Array.of_list given)
  | Reduce { layout; axes; prog; reductions; loads } ->
      let given = reduce_cts env layout axes reductions cts in
      if given <> [] then sweep env layout prog loads (Array.of_list given)
  | Scan { layout; axis; prog; reduction; loads; carry } ->
      scan_rule env layout axis prog reduction loads carry cts
  | Gather { axis; idx; x } ->
      first (fun w -> if env.lin x then gather_rule env ~axis idx x w)
  | Scatter { combine; unique; axis; idx; updates; into } ->
      first (scatter_rule env combine ~unique ~axis idx updates into)
  | Sort { x; _ } ->
      if env.lin x && (Option.is_some (cts 0) || Option.is_some (cts 1)) then
        nonlinear env "a sort"
  | Fold { pad; x; shape = _ } ->
      first (fun (V w) ->
          if env.lin x then
            let dt = Nx.dtype x in
            let ct = with_dtype dt w in
            let layout = L.contiguous (Iarray.of_array (Nx.shape x)) in
            let y, () =
              Prim.eval ~by
                (Map
                   {
                     layout;
                     prog = identity (D.Any dt);
                     outs = [ dt ];
                     loads = [| Padded { x = ct; fill = D.zero dt; pad } |];
                     at = None;
                   })
            in
            env.give x (fresh (V y)))
  | Assemble { dtype; shape; pieces; fill = _ } ->
      first (assemble_rule env dtype shape pieces)
  | Contract
      {
        batch;
        contracting;
        acc;
        a;
        b;
        init;
        a_scale;
        b_scale;
        segments;
        out = _;
      } ->
      first (fun w ->
          Option.iter (fun i -> if env.lin i then env.give i (held w)) init;
          if env.lin a || env.lin b then
            contract_rule env ~batch ~con:contracting ~acc ~a ~b ~a_scale
              ~b_scale ~segments w)
  | Fft f -> first (fft_rule env f)
  | Linalg (Solve_triangular { triangle; adjoint; unit_diagonal; a; b }) ->
      first (fun (V w) ->
          if env.lin a then nonlinear env "a triangular solve's matrix";
          if env.lin b then
            match reference [ V (env.const a); V w ] with
            | V r ->
                (* [x = op(a)⁻¹ b] transposes to [op(a)⁻ᵀ w], unconjugated: the
                   solve by [op(a)]'s other transpose, through conjugates for a
                   complex [a]. *)
                let a = rebrand r (env.const a) in
                let ct = rebrand r (with_dtype (Nx.dtype b) w) in
                let solve b =
                  Prim.eval ~by
                    (Linalg
                       (Solve_triangular
                          {
                            triangle;
                            adjoint = not adjoint;
                            unit_diagonal;
                            a;
                            b;
                          }))
                in
                let y =
                  if D.is D.Complex (Nx.dtype b) then
                    Nx.conjugate (solve (Nx.conjugate ct))
                  else solve ct
                in
                env.give b (fresh (V y)))
  | Linalg _ ->
      let (Prim.Operands xs) = Prim.operands op in
      if List.exists (fun (Prim.Any x) -> env.lin x) xs then
        nonlinear env "a factorisation"
  | Copy x -> first (fun w -> if env.lin x then env.give x (held w))
  | Move (mv, x) -> first (fun w -> if env.lin x then move_rule env mv x w)
  | Bitcast (dt, x) ->
      first (fun (V w) ->
          if env.lin x then begin
            if not (same_coordinates (D.Any dt) (D.Any (Nx.dtype x))) then
              nonlinear env "a bitcast";
            (* A complex number's pairing with its parts' is [Re (u z) = u_re
               z_re - u_im z_im], so each bitcast's transpose conjugates. *)
            let w = with_dtype dt w in
            let y =
              if D.is D.Complex (Nx.dtype x) then
                Nx.conjugate (Prim.eval ~by (Bitcast (Nx.dtype x, w)))
              else Prim.eval ~by (Bitcast (Nx.dtype x, Nx.conjugate w))
            in
            env.give x (fresh (V y))
          end)
  | Place (_, x) ->
      first (fun (V w) ->
          if env.lin x then
            let p = Nx.placement x in
            env.give x (held (V (Prim.eval ~by (Place (p, w))))))
  | Check _ -> ()

(* [lanes a]'s transpose, as JAX's all_gather's: the calling lane's row of the
   sum over the lanes of their cotangents [w] of every lane's [x]. *)
let lanes_transpose a (x : ('v, 's, 'd) Nx.t) (V w) =
  let w = with_dtype (Nx.dtype x) w in
  let total =
    Nx.sum ~axes:[ 0 ] (like w (Program.perform (Lanes (a, w))).(0))
  in
  let placement = Nx.beside w in
  let index = (Program.perform (Lane_index (a, placement))).(0) in
  let own = cast Nx.int32 (Nx.Placement.devices placement) index in
  let row = Nx.reshape [| 1 |] (Nx.cast Nx.int64 own) in
  V (Nx.reshape (Nx.shape x) (Nx.take ~axis:0 row total))

(* A linear call's transpose: its transpose program over the results'
   cotangents, zeros for one with none, and its residuals, its outputs given to
   the arguments. Where the call has its function, the transpose is itself a
   linear call whose transpose is that function, so that transposing it again
   applies the function. The additions reach the scopes around the backward
   pass. *)
let linear_call env ~name ~apply ~transpose ~args ~captures ~residuals ~forms
    cts =
  let ct k (F f) =
    match cts k with
    | Some c -> V (cast f.dtype (Nx.Placement.devices f.placement) c)
    | None -> V (Nx.zeros_at f.placement f.dtype (shape_of f.layout))
  in
  let cts = Array.mapi ct forms in
  let const = Array.map (fun (V r) -> V (env.const r)) in
  let out =
    match apply with
    | Some f ->
        Program.run
          (Linear
             {
               name;
               apply = Some transpose;
               transpose = f;
               args = cts;
               captures = const residuals;
               residuals = const captures;
               forms = Array.map form args;
             })
    | None ->
        Program.threaded transpose (fun totals ->
            Program.eval ~totals transpose (Array.append cts (const residuals)))
  in
  Array.iteri (fun k (V a) -> if env.lin a then env.give a (held out.(k))) args

(* Transposition *)

let next_bases (p : Program.t) =
  let n = Array.length p.eqns in
  let base = function Op { base; _ } | Do { base; _ } -> base in
  Array.init n (fun e -> if e + 1 < n then base p.eqns.(e + 1) else p.slots)

(* [p]'s parameters' cotangents, given its outputs' [cts]: [None] for a
   parameter bound to a constant ([consts.(k) = Some v]) or one that receives no
   contribution. *)
let rec walk ~name ~fn (p : Program.t) consts cts =
  let owner = p.owner in
  let bound = Array.make p.slots None in
  Array.iteri (fun k c -> bound.(p.params.(k)) <- c) consts;
  let acc = Array.make p.slots None in
  let var x = match Prim.payload owner x with Some (Var n) -> n | _ -> -1 in
  let lin x =
    let n = var x in
    n >= 0 && Option.is_none bound.(n)
  in
  let const (type v s d) (x : (v, s, d) Nx.t) : (v, s, d) Nx.t =
    let n = var x in
    if n < 0 then x
    else
      match bound.(n) with
      | Some v -> like x v
      | None -> invalid_arg (name ^ ": a tangent is read as a constant")
  in
  let give x c =
    let n = var x in
    acc.(n) <- Some (match acc.(n) with None -> c | Some a -> sum a c)
  in
  let env = { name; lin; const; give; fn } in
  Array.iteri
    (fun k c ->
      match (c, p.outputs.(k)) with
      | Some c, V y -> if lin y then give y (held c)
      | None, _ -> ())
    cts;
  let acc_at base k = Option.map (fun c -> c.v) acc.(base + k) in
  let next = next_bases p in
  let reached base n =
    let rec go k = k < n && (Option.is_some acc.(base + k) || go (k + 1)) in
    go 0
  in
  let cts_at base n k =
    if k < n then Option.map (fun c -> c.v) acc.(base + k) else None
  in
  for e = Array.length p.eqns - 1 downto 0 do
    match p.eqns.(e) with
    | Op { op; base; _ } ->
        let n = next.(e) - base in
        if reached base n then rule env op (cts_at base n)
    | Do { c = Lanes (a, x); base } -> (
        match acc.(base) with
        | Some c when lin x -> give x (fresh (lanes_transpose a x c.v))
        | _ -> ())
    | Do
        {
          c =
            Linear { name; apply; transpose; args; captures; residuals; forms };
          base;
        } ->
        if reached base (Array.length forms) then
          linear_call env ~name ~apply ~transpose ~args ~captures ~residuals
            ~forms (acc_at base)
    | Do
        {
          c = Scan { step; length; trips; reverse; init; xs; captures; _ };
          base;
        } ->
        let n = next.(e) - base in
        if reached base n then
          scan_rule env step ~length ~trips ~reverse ~init ~xs ~captures
            (cts_at base n)
    | Do { c = Operator o; base } ->
        (* An operator's transpose is its transposed operator. *)
        let n = Array.length o.forms in
        if reached base n then begin
          let cts =
            Array.mapi
              (fun k f ->
                match acc_at base k with
                | Some c -> c
                | None -> Loops.filled ~zero:true f)
              o.forms
          in
          let data = Array.map (fun (V d) -> V (const d)) o.data in
          let r =
            Program.perform
              (Operator
                 { o with transposed = not o.transposed; args = cts; data })
          in
          Array.iteri (fun k (V a) -> if lin a then give a (fresh r.(k))) o.args
        end
    | Do { c = Solve sv; base } ->
        (* A linear solve's transpose solves the transposed system. *)
        let n = Array.length sv.b in
        if reached base n then begin
          let cts =
            Array.mapi
              (fun k (V b) ->
                match acc_at base k with
                | Some c -> c
                | None -> Loops.filled ~zero:true (form (V b)))
              sv.b
          in
          let const = Array.map (fun (V d) -> V (env.const d)) in
          let r =
            Program.perform
              (Solve
                 {
                   sv with
                   transposed = not sv.transposed;
                   b = cts;
                   data = const sv.data;
                   captures = const sv.captures;
                 })
          in
          Array.iteri (fun k (V b) -> if lin b then give b (fresh r.(k))) sv.b
        end
    | Do { c = Remat { body; args; _ }; base } ->
        let n = next.(e) - base in
        if reached base n then remat_rule env body args (cts_at base n)
    | Do { c = Call { body; args; site; _ }; base } ->
        let n = next.(e) - base in
        if reached base n then call_rule env body args site (cts_at base n)
    | Do { c = Switch _ as c; base } ->
        let n = next.(e) - base in
        if reached base n then switch_rule env c (cts_at base n)
    | Do { base; _ } ->
        if reached base (next.(e) - base) then
          invalid_arg (name ^ ": a loop has no transpose yet")
  done;
  Array.mapi
    (fun k c -> if Option.is_none c then acc.(p.params.(k)) else None)
    consts

(* A linear scan's transpose is the scan of its step's transpose over the same
   rows in the other order, as JAX transposes a scan: its carry holds the
   cotangents of the carries that receive one, a fixpoint, and the running sums
   of the linear captures' cotangents; its rows are the constant rows and the
   outputs' cotangents. A carry or a sum starts from the identity that keeps
   every sign, so that a single contribution is itself. *)
and scan_rule env (step : Program.t) ~length ~trips ~reverse ~init ~xs ~captures
    cts =
  let nc = Array.length init and nx = Array.length xs in
  let ncap = Array.length captures in
  let ny = Array.length step.outputs - nc - Array.length step.adds in
  let lin (V x) = env.lin x and const (V x) = V (env.const x) in
  let ct_c = Array.init nc cts
  and ct_y = Array.init ny (fun j -> cts (nc + j)) in
  if length = 0 then
    Array.iteri
      (fun k (V x) ->
        match ct_c.(k) with
        | Some c when env.lin x -> env.give x (held c)
        | _ -> ())
      init
  else
    let lx = Array.map lin xs and lk = Array.map lin captures in
    let cxs = Array.map const (Loops.pick (Array.map not lx) xs) in
    let ckk = Array.map const (Loops.pick (Array.map not lk) captures) in
    let gy = Array.map Option.is_some ct_y in
    let ys = Array.map Option.get (Loops.pick gy ct_y) in
    let nlk = Array.length (Loops.pick lk lk) in
    let caps = Loops.pick lk captures in
    let start k =
      match ct_c.(k) with
      | Some c -> c
      | None -> Loops.filled ~zero:false (form init.(k))
    in
    let rec fix m a (cf : form array) (af : form array) =
      let gained = Array.make nc false and gained_a = Array.make nlk false in
      let got = Array.make nc false in
      let has_row = ref [||] in
      let forms =
        Array.concat
          [
            Loops.pick m cf;
            Loops.pick a af;
            Array.map Loops.row cxs;
            Array.map Loops.row ys;
            Array.map form ckk;
          ]
      in
      let count m = Array.length (Loops.pick m m) in
      let lengths =
        [
          count m; count a; Array.length cxs; Array.length ys; Array.length ckk;
        ]
      in
      let transposed vs =
        match Loops.split lengths vs with
        | [ cc; ca; rx; ry; kk ] ->
            let some = Array.map Option.some in
            let none n = Array.make n None in
            let consts =
              Array.concat
                [
                  none nc;
                  Loops.merge lx
                    (none (Array.length (Loops.pick lx lx)))
                    (some rx);
                  Loops.merge lk (none nlk) (some kk);
                ]
            in
            let out_cts =
              Array.concat
                [
                  Loops.merge m (some cc) (none (nc - Array.length cc));
                  Loops.merge gy (some ry) (none (ny - Array.length ry));
                  none (Array.length step.adds);
                ]
            in
            let r =
              walk ~name:env.name ~fn:env.fn step consts out_cts
            in
            let carry =
              Array.init nc (fun k ->
                  gained.(k) <- (not m.(k)) && Option.is_some r.(k);
                  got.(k) <- Option.is_some r.(k);
                  match r.(k) with
                  | Some c -> Some c.v
                  | None ->
                      if m.(k) then Some (Loops.filled ~zero:false cf.(k))
                      else None)
            in
            let carry = Array.map Option.get (Loops.pick m carry) in
            let rcaps = Loops.pick lk (Array.sub r (nc + nx) ncap) in
            let accs = ref (Array.to_list ca) in
            let caps' =
              Array.mapi
                (fun j rc ->
                  gained_a.(j) <- (not a.(j)) && Option.is_some rc;
                  if not a.(j) then None
                  else
                    let acc = List.hd !accs in
                    accs := List.tl !accs;
                    match rc with
                    | Some c -> Some (sum (fresh acc) c).v
                    | None -> Some acc)
                rcaps
            in
            let caps' = Array.map Option.get (Loops.pick a caps') in
            let rrows = Loops.pick lx (Array.sub r nc nx) in
            has_row := Array.map Option.is_some rrows;
            Array.concat
              [
                carry;
                caps';
                Array.map
                  (fun c -> (Option.get c).v)
                  (Loops.pick !has_row rrows);
              ]
        | _ -> invalid_arg "Rune: a scan's transpose"
      in
      let prog, own = Trace.trace Body ~name:env.name forms transposed in
      if Array.exists Fun.id gained || Array.exists Fun.id gained_a then
        fix (Array.map2 ( || ) m gained) (Array.map2 ( || ) a gained_a) cf af
      else (prog, own, m, got, a, !has_row, cf)
    in
    let m = Array.map Option.is_some ct_c and a = Array.make nlk false in
    let cf = Array.init nc (fun k -> form (start k)) in
    let af = Array.map form caps in
    let prog, own, m, got, a, has_row, cf = fix m a cf af in
    let init' =
      Array.mapi
        (fun k f ->
          match ct_c.(k) with Some c -> c | None -> Loops.filled ~zero:false f)
        cf
    in
    let r =
      Program.perform
        (Scan
           {
             step = prog;
             length;
             trips = Option.map const trips;
             reverse = not reverse;
             init =
               Array.append (Loops.pick m init')
                 (Array.map (Loops.filled ~zero:false)
                    (Loops.pick a (Array.map form caps)));
             xs = Array.append cxs ys;
             captures = Array.append ckk own;
             seeds = [||];
           })
    in
    let next = ref (-1) in
    let take () =
      incr next;
      r.(!next)
    in
    (* A carry's init receives no cotangent where the step's transpose gives the
       carry none: only the last row's could, and it is a zero; but where the
       rows that run are counted, none may run. *)
    Array.iteri
      (fun k (V x) ->
        if m.(k) then
          let c = take () in
          if (got.(k) || Option.is_some trips) && env.lin x then
            env.give x (held c))
      init;
    Array.iteri (fun j (V x) -> if a.(j) then env.give x (held (take ()))) caps;
    let lxs = Loops.pick lx xs in
    Array.iteri
      (fun j (V x) -> if has_row.(j) then env.give x (held (take ())))
      lxs

(* A linear switch's transpose is the switch, on the same index, of its
   branches' transposes, as JAX transposes a cond: each branch's over the
   constants and the results' cotangents, to the linear captures'. *)
and switch_rule env (sw : construct) cts =
  match sw with
  | Switch { name; index; branches; captures; _ } ->
      let la = Array.map (fun (V x) -> env.lin x) captures in
      let consts =
        Array.map
          (fun (V x) -> V (env.const x))
          (Loops.pick (Array.map not la) captures)
      in
      let lins = Loops.pick la captures in
      let no = Array.length (Program.switched branches) in
      let ct_y = Array.init no cts in
      let gy = Array.map Option.is_some ct_y in
      let ys = Array.map Option.get (Loops.pick gy ct_y) in
      let nc = Array.length consts and ny = Array.length ys in
      let forms = Array.map form (Array.append consts ys) in
      let lforms = Array.map form lins in
      let transposed (b : Program.t) vs =
        let cs = Array.sub vs 0 nc and gs = Array.sub vs nc ny in
        let tape, tcaps =
          Trace.trace Tape ~name:env.name lforms (fun ts ->
              eval b (Loops.merge la ts cs))
        in
        let nl = Array.length lins in
        let tconsts =
          Array.append (Array.make nl None) (Array.map Option.some tcaps)
        in
        let gys =
          let n = ref (-1) in
          Array.map
            (fun g ->
              if g then begin
                incr n;
                Some gs.(!n)
              end
              else None)
            gy
        in
        let r =
          walk ~name:env.name ~fn:env.fn tape tconsts
            (Array.init (Array.length tape.outputs) (fun j ->
                 if j < no then gys.(j) else None))
        in
        Array.mapi
          (fun k f ->
            match r.(k) with Some c -> c.v | None -> Loops.filled ~zero:true f)
          lforms
      in
      let traced =
        Array.map
          (fun b -> Trace.trace Staged ~name:env.name forms (transposed b))
          branches
      in
      let progs, own = unite traced ~params:(Array.length forms) in
      let (V ix) = index in
      let r =
        Program.perform
          (Switch
             {
               name;
               index = V (env.const ix);
               branches = progs;
               captures = Array.concat [ consts; ys; own ];
               seeds = [||];
             })
      in
      Array.iteri (fun k (V x) -> env.give x (fresh r.(k))) lins
  | _ -> invalid_arg "Rune: a switch's transpose"

(* A remat's transpose linearizes its function again at its constant arguments,
   then transposes that, so that the forward pass kept its arguments alone.
   Where a tracer stages it, it is the remat of that transpose over the
   constant arguments and the cotangents given, differentiated, as JAX
   transposes a checkpoint: compiled, it computes the function's values again
   apart from the forward pass's, once the cotangents are computed. Elsewhere it
   computes where it is, as that remat would. *)
and remat_rule env (body : Program.t) args cts =
  let name = env.name in
  let la = Array.map (fun (V x) -> env.lin x) args in
  let consts =
    Array.map
      (fun (V x) -> V (env.const x))
      (Loops.pick (Array.map not la) args)
  in
  let lins = Loops.pick la args in
  let n = Array.length lins in
  let given = Array.init (Array.length body.outputs) cts in
  let gc = Array.map Option.is_some given in
  (* The linear arguments' cotangents at the constant arguments [cs], given the
     cotangents [ws]. *)
  let transposed cs ws =
    let tape, captures =
      Trace.trace Tape ~name (Array.map form lins) (fun ts ->
          eval body (Loops.merge la ts cs))
    in
    let tconsts =
      Array.append (Array.make n None) (Array.map Option.some captures)
    in
    let out_cts =
      Loops.merge gc (Array.map Option.some ws)
        (Array.make (Array.length gc - Array.length ws) None)
    in
    Array.sub (walk ~name ~fn:env.fn tape tconsts out_cts) 0 n
  in
  let ws = Array.map Option.get (Loops.pick gc given) in
  if not (Program.staging ()) then
    let r = transposed consts ws in
    Array.iteri (fun k (V x) -> Option.iter (env.give x) r.(k)) lins
  else
    let nc = Array.length consts and got = ref [||] in
    let staged vs =
      let r =
        transposed (Array.sub vs 0 nc) (Array.sub vs nc (Array.length vs - nc))
      in
      got := Array.map Option.is_some r;
      Array.map (fun c -> (Option.get c).v) (Loops.pick !got r)
    in
    let ins = Array.append consts ws in
    let prog, own = Trace.trace Staged ~name (Array.map form ins) staged in
    let r =
      Program.perform
        (Remat
           {
             body = prog;
             args = Array.append ins own;
             seeds = [||];
             differentiated = true;
           })
    in
    let next = ref 0 in
    Array.iteri
      (fun k (V x) ->
        if !got.(k) then begin
          env.give x (held r.(!next));
          incr next
        end)
      lins

(* A linear call's transpose is the call of its program's transpose, as JAX
   transposes a compiled call's jaxpr: derived once per derivation at the
   call's site, over the constant arguments and the cotangents given, traced as
   a compiled function is, and compiled. *)
and call_rule env (body : Program.t) args site cts =
  let name = env.name in
  let la = Array.map (fun (V x) -> env.lin x) args in
  let nl = Array.map not la in
  let consts = Array.map (fun (V x) -> V (env.const x)) (Loops.pick nl args) in
  let na = Array.length body.adds in
  let no = Array.length body.outputs - na in
  let given = Array.init no cts in
  let gc = Array.map Option.is_some given in
  let ws = Array.map Option.get (Loops.pick gc given) in
  let ins = Array.append consts ws in
  let forms = Array.map form ins in
  let made () =
    let got = ref [||] in
    let transposed vs =
      let nc = Array.length consts in
      let cs = Array.sub vs 0 nc in
      let ws = Array.sub vs nc (Array.length vs - nc) in
      let none n = Array.make n None in
      let bound =
        Loops.merge la
          (none (Array.length args - nc))
          (Array.map Option.some cs)
      in
      let out_cts =
        Array.append
          (Loops.merge gc
             (Array.map Option.some ws)
             (none (no - Array.length ws)))
          (none na)
      in
      let r =
        walk ~name:env.name ~fn:env.fn body bound out_cts
      in
      let lin = Loops.pick la r in
      got := Array.map Option.is_some lin;
      Array.map (fun c -> (Option.get c).v) (Loops.pick !got lin)
    in
    let prog, _ = Trace.trace Jit ~name forms transposed in
    ((prog, forms), !got, None)
  in
  let derivation =
    Program.Transpose { linear = la; cts = gc }
  in
  let d = Program.derive site derivation forms made in
  let r =
    Program.run (Call { body = d.program; args = ins; site = d.at; seeds = [||] })
  in
  let next = ref 0 in
  Array.iteri
    (fun k (V x) ->
      if d.flags.(k) then begin
        env.give x (held r.(!next));
        incr next
      end)
    (Loops.pick la args)

let transpose l cts =
  let p = l.tape in
  let consts =
    Array.init (Array.length p.params) (fun k ->
        if k < l.ins then None else Some l.captures.(k - l.ins))
  in
  let out_cts = Array.make (Array.length p.outputs) None in
  Array.iteri
    (fun k c ->
      match (l.outs.(k), c) with
      | Some j, Some c -> out_cts.(j) <- Some c
      | _ -> ())
    cts;
  (* A backward pass offers nothing. *)
  Program.silenced (fun () ->
      Array.sub (walk ~name:l.name ~fn:l.fn l.tape consts out_cts) 0 l.ins)
