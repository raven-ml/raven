(* Tests of Tolk.Simplify: each rewrite its interface states, on hand-built
   graphs, on the cases and kernels gen/codegen/simplify.py records, and the
   laws that the reductions' rewrites keep a value and the range rewrites keep
   what a kernel writes. *)

open Windtrap
open Tolk
module Pm = Ops.Pattern_matcher

let uop = Uops.uop
let i n = `Int (Bigint.of_int n)

(* Passes *)

let rewrite m u =
  Ops.graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:() u (After_sources m)

let flatten u = rewrite Simplify.pm_flatten_range u
let unparent u = rewrite Simplify.pm_reduce_unparented u
let collapse u = rewrite Simplify.pm_reduce_collapse u
let reduce_simplify u = rewrite Simplify.pm_reduce_simplify u
let load_collapse u = rewrite Simplify.pm_load_collapse u
let unpin u = rewrite Simplify.pm_unpin_effects u

(* The range passes read a context, which starts empty; the pipeline flattens
   the ranges of what they build. *)
let split u =
  Ops.graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:(Ops.Tbl.create 8) u
    (After_sources
       (Pm.append Simplify.pm_split_ranges
          (Pm.with_ctx Simplify.pm_flatten_range)))

let simplify u =
  Ops.graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:(Ops.Tbl.create 8) u
    (After_sources
       (Pm.append
          (Pm.with_ctx Simplify.pm_flatten_range)
          Simplify.pm_simplify_ranges))

(* The symbolic pass the pipeline runs before simplifying ranges, which removes
   the guards that hold everywhere. *)
let initial_symbolic u =
  rewrite (Pm.append Symbolic.sym Simplify.pm_flatten_range) u

(* Builders *)

let range ?(axis_type = Ops.Axis_type.Weak) n axis =
  Ops.range ~axis_type (Int n) [ axis ]

let var ?dtype name lo hi = Ops.variable ?dtype name (i lo) (i hi)
let buf ?(slot = 0) ?(size = 1024) dt = Shape.param ~shape:[ Int size ] slot dt
let kernel us = Ops.sink ~kernel:(Ops.kernel_info ()) us
let f32 x = Ops.float ~dtype:Float32 x
let i32 n = Ops.int ~dtype:Int32 n
let sum u rs = Ops.reduce u Add rs
let store_at idx x = Ops.store (Ops.index (buf Float32) [ idx ]) x
let one = f32 1.
let two = f32 2.
let zero = Ops.float 0.

let gated_load valid idx =
  Ops.load (Ops.index (buf Float32) [ Shape.valid idx valid ]) []

let ranges u =
  List.filter (fun n -> Ops.op n = Range) (Ops.toposort ~calls:Enter u)

let size r = Bigint.to_int (Shape.to_z (Ops.nth r 0))
let sizes u = List.sort compare (List.map size (ranges u))

let count op u =
  List.length
    (List.filter (fun n -> Ops.op n = op) (Ops.toposort ~calls:Enter u))

(* [sums_to v u] is the claim that [u] has no range left and is worth [v]. *)
let sums_to ?(vars = []) v u =
  equal (list int) [] (sizes u);
  equal Dtypes.const v (Interpreter.eval ~vars u)

(* Bindings

   The laws that a rewrite keeps a value, or what a kernel writes, evaluate a
   graph and its rewrite at bindings of the leaves the graph reads: variables
   and scalar parameters within their bounds, the ranges it runs inside below
   their sizes, and the elements of storage in [-3, 12], so that an index read
   from memory falls on both sides of a small table's bounds. Floats are drawn
   as integers, so that sums are exact whatever order they add in.

   A leaf takes one choice: its lower bound, its upper bound, or a value
   between. Storage takes one choice for all its elements: all at the lower
   bound, all at the upper, or a cycle through every value from a drawn
   start. *)

type choice = Low | High | At of int

let pp_choice ppf = function
  | Low -> Format.pp_print_string ppf "low"
  | High -> Format.pp_print_string ppf "high"
  | At n -> Format.pp_print_int ppf n

let choice =
  Gen.(
    frequency
      [
        (1, constant Low);
        (1, constant High);
        (4, map (fun n -> At n) (int_range 0 1_000_000));
      ])

let pick c lo hi =
  match c with Low -> lo | High -> hi | At n -> lo + (n mod (hi - lo + 1))

type binding = {
  vars : (string * Dtype.value) list;
  params : (int * Dtype.value) list;
  buffers : (int * Dtype.value array) list;
}

let eval b u =
  Interpreter.eval ~vars:b.vars ~params:b.params ~buffers:b.buffers u

(* The leaves [u] reads, and the ranges it runs inside, each after the leaves
   its value depends on. *)
let leaves u =
  let free = Ops.ranges u in
  List.filter
    (fun n ->
      match Ops.op n with
      | Range -> Ops.Nodes.mem n free
      | Param -> true
      | _ -> false)
    (Ops.toposort ~calls:Enter u)

let element_lo = -3
let element_hi = 12

let value_of dt v : Dtype.value =
  match (dt : Dtype.t) with
  | Bool -> `Bool (v = 1)
  | dt when Dtype.is_float dt -> `Float (float_of_int v)
  | _ -> i v

let bound_int (v : Dtype.value) =
  match v with
  | `Int z -> Bigint.to_int z
  | `Float f -> int_of_float f
  | `Bool b -> Bool.to_int b

(* [binding choices u] binds the leaves of [u], one choice each, or is [None] if
   a range [u] runs inside is empty. *)
let binding choices u =
  let bind b (u, c) =
    Option.bind b (fun b ->
        match (Ops.op u, Ops.arg u) with
        | Param, Param { vmin_vmax = Some (lo, hi); dtype; name = Some name; _ }
          ->
            if List.mem_assoc name b.vars then Some b
            else
              let v = value_of dtype (pick c (bound_int lo) (bound_int hi)) in
              Some { b with vars = (name, v) :: b.vars }
        | Param, Param { slot; dtype; _ } when Shape.max_shape u = [] ->
            if List.mem_assoc slot b.params then Some b
            else
              let v = value_of dtype (pick c 0 9) in
              Some { b with params = (slot, v) :: b.params }
        | Param, Param { slot; dtype; _ } ->
            if List.mem_assoc slot b.buffers then Some b
            else
              let span = element_hi - element_lo + 1 in
              let element j =
                match c with
                | Low -> element_lo
                | High -> element_hi
                | At n -> element_lo + ((n + (7 * j)) mod span)
              in
              let elements =
                Array.init (Shape.max_numel u) (fun j ->
                    value_of dtype (element j))
              in
              Some { b with buffers = (slot, elements) :: b.buffers }
        | Range, _ -> (
            let name = Option.get (Interpreter.name u) in
            if List.mem_assoc name b.vars then Some b
            else
              match eval b (Ops.nth u 0) with
              | `Int n when Bigint.(n > zero) ->
                  let v = i (pick c 0 (Bigint.to_int n - 1)) in
                  Some { b with vars = (name, v) :: b.vars }
              | _ -> None)
        | _ -> Some b)
  in
  List.fold_left bind
    (Some { vars = []; params = []; buffers = [] })
    (List.combine (leaves u) choices)

let pp_binding ppf b =
  let pp_one ppf (name, v) = Format.fprintf ppf "%s=%a" name Dtype.pp_const v in
  Format.pp_print_list
    ~pp_sep:(fun ppf () -> Format.fprintf ppf ", ")
    pp_one ppf (List.rev b.vars)

(* A count times a value that is no sum of zeros may be [-0.] where the sum is
   [0.]: zeros compare by value. *)
let by_value = function `Float f -> `Float (f +. 0.) | v -> v

let keeps_value name b before after =
  let msg = Format.asprintf "%s at %a" name pp_binding b in
  equal ~msg Dtypes.const (by_value (eval b before)) (by_value (eval b after))

let write = triple int int Dtypes.value

let kernel_writes b u =
  List.map
    (fun (s, i, v) -> (s, i, by_value v))
    (Interpreter.writes ~vars:b.vars ~params:b.params ~buffers:b.buffers u)

let keeps_writes name b before after =
  let msg = Format.asprintf "%s at %a" name pp_binding b in
  equal ~msg (list write) (kernel_writes b before) (kernel_writes b after)

(* Tables

   A table is named graphs. A law over a table binds every graph at each case,
   so that each run checks every graph at fresh bindings; its two examples bind
   every leaf at its lower bound, then at its upper. A rewrite reads no leaf its
   graph does not. *)

type table = { names : string list; graphs : Ops.t list }

let table named = { names = List.map fst named; graphs = List.map snd named }
let arities t = List.map (fun u -> List.length (leaves u)) t.graphs

let choices_of t =
  Gen.with_pp
    (fun ppf css ->
      Format.pp_print_list
        ~pp_sep:(fun ppf () -> Format.pp_print_string ppf " | ")
        (Format.pp_print_list ~pp_sep:Format.pp_print_space pp_choice)
        ppf css)
    (List.fold_right
       (fun n rest ->
         Gen.(
           let+ cs = list ~size:(constant n) choice and+ rest = rest in
           cs :: rest))
       (arities t) (Gen.constant []))

let extremes t =
  List.map
    (fun c -> List.map (fun n -> List.init n (fun _ -> c)) (arities t))
    [ Low; High ]

(* [over ~count claim t pass check] is the law [claim] over [t]: at each case,
   [check name binding graph (pass graph)] for each graph of [t]. A graph inside
   an empty range checks nothing, so the case is discarded. *)
let over ~count claim t pass check =
  let rewritten = lazy (List.map pass t.graphs) in
  prop ~count claim ~examples:(extremes t) (choices_of t) (fun css ->
      let bindings = List.map2 binding css t.graphs in
      assume (List.for_all Option.is_some bindings);
      List.iteri
        (fun k b ->
          check (List.nth t.names k) (Option.get b) (List.nth t.graphs k)
            (List.nth (Lazy.force rewritten) k))
        bindings)

(* The cases of a golden table: each source of its input golden, named by its
   row. *)
let recorded_cases matcher =
  let inputs = Ops.src (Golden.sink (matcher ^ "_input.golden")) in
  List.map
    (fun cell -> (cell "case", List.nth inputs (int_of_string (cell "src"))))
    (Golden.rows (matcher ^ ".golden"))

let rewrites_as_recorded matcher pass =
  let input = lazy (Ops.src (Golden.sink (matcher ^ "_input.golden"))) in
  let output = lazy (Ops.src (Golden.sink (matcher ^ "_output.golden"))) in
  Golden.cases (matcher ^ ".golden") ~key:[ "case" ] (fun cell ->
      let n = int_of_string (cell "src") in
      equal uop
        (List.nth (Lazy.force output) n)
        (pass (List.nth (Lazy.force input) n)))

let recorded name result pass =
  Golden.graph
    (name ^ "_" ^ result ^ ".golden")
    (fun () -> pass (Golden.sink (name ^ ".golden")))

let recorded_kernels names =
  table (List.map (fun n -> (n, Golden.sink (n ^ ".golden"))) names)

let keeps_values matcher pass =
  over ~count:10 "keeps the value of each recorded case"
    (table (recorded_cases matcher))
    pass keeps_value

let keeps_case_writes ?(prepared = Fun.id) matcher pass =
  over ~count:4 "keeps what each recorded case writes"
    (table (List.map (fun (n, u) -> (n, prepared u)) (recorded_cases matcher)))
    pass keeps_writes

(* Generated kernels: a store inside one to three ranges, at an index that takes
   each range, or its remainder or quotient by a divisor of its size, with
   strides in some order, possibly guarded below a constant; the value stored
   tells the iterations apart, and reads nothing. *)
type term = Plain | Remainder of int | Quotient of int

let pp_kernel ppf (dims, order, guard) =
  let pp_dim ppf (n, t) =
    match t with
    | Plain -> Format.fprintf ppf "%d" n
    | Remainder c -> Format.fprintf ppf "%d%%%d" n c
    | Quotient c -> Format.fprintf ppf "%d//%d" n c
  in
  Format.fprintf ppf "ranges [%a], strides in order [%a]%a"
    (Format.pp_print_list
       ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
       pp_dim)
    dims
    (Format.pp_print_list
       ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
       Format.pp_print_int)
    order
    (fun ppf -> function
      | None -> () | Some (j, c) -> Format.fprintf ppf ", guarded r%d < %d" j c)
    guard

let generated_kernels =
  let open Gen in
  let dim =
    let* n = of_list [ 2; 3; 4; 6; 8; 12 ] in
    let divisors =
      List.filter (fun c -> n mod c = 0 && c > 1 && c < n) [ 2; 3; 4; 6 ]
    in
    let+ t =
      one_of
        (constant Plain
        :: List.concat_map
             (fun c -> [ constant (Remainder c); constant (Quotient c) ])
             divisors)
    in
    (n, t)
  in
  let* dims = list ~size:(int_range 1 3) dim in
  let* order = Gen.permutation (List.init (List.length dims) Fun.id) in
  let+ guard =
    option
      (let* j = int_range 0 (List.length dims - 1) in
       let n, _ = List.nth dims j in
       let+ c = int_range 1 (n - 1) in
       (j, c))
  in
  (dims, order, guard)

let generated_kernels = Gen.with_pp pp_kernel generated_kernels

let kernel_of (dims, order, guard) =
  let rs = List.mapi (fun axis (n, _) -> range n axis) dims in
  let term r (_, t) =
    match t with
    | Plain -> r
    | Remainder c -> Ops.O.(r % Ops.int c)
    | Quotient c -> Ops.O.(r // Ops.int c)
  in
  let terms = List.map2 term rs dims in
  let index, _ =
    List.fold_left
      (fun (index, stride) j ->
        (Ops.O.(index + (List.nth terms j * Ops.int stride)), stride * 13))
      (Ops.int 0, 1)
      (List.rev order)
  in
  let value, _ =
    List.fold_left
      (fun (value, weight) r ->
        (Ops.O.(value + (r * Ops.int weight)), weight * 17))
      (Ops.int 0, 1)
      rs
  in
  let index =
    match guard with
    | None -> index
    | Some (j, c) -> Shape.valid index Ops.O.(List.nth rs j < Ops.int c)
  in
  initial_symbolic
    (kernel
       [
         Ops.end_
           (Ops.store (Ops.index (buf Int32) [ index ]) (Ops.cast value Int32))
           rs;
       ])

(* What a generated kernel writes, which reads nothing. *)
let writes u = Interpreter.writes ~buffers:[ (0, [||]) ] u

(* Ranges *)

let flattening =
  let r0 = range 3 0 in
  group "pm_flatten_range"
    [
      test "ranges keep the order they are listed in" (fun () ->
          let late =
            Ops.range ~axis_type:Weak (Sym Ops.O.(r0 + Ops.int 1)) [ 4 ]
          in
          let u =
            Ops.end_ (store_at Ops.O.((r0 * Ops.int 4) + late) one) [ late; r0 ]
          in
          equal uop u (flatten u));
      group "flattens each case as its golden states"
        [ rewrites_as_recorded "flatten_range" flatten ];
      keeps_case_writes "flatten_range" flatten;
    ]

let merged_size u = sizes (simplify (kernel [ u ]))

let simplifying =
  let r0 = range 3 0 in
  let s0 = range 3 3 and s1 = range 4 4 in
  group "pm_simplify_ranges"
    [
      test "a range whose end is read after another range stays apart from it"
        (fun () ->
          let after_r0 =
            Ops.range ~axis_type:Weak (Sym (Ops.after (Ops.int 4) [ r0 ])) [ 1 ]
          in
          let u = Ops.end_ (store_at (Ops.int 0) one) [ r0; after_r0 ] in
          equal int 2 (List.length (ranges (simplify (kernel [ u ])))));
      test "reductions that share ranges merge them in all of them" (fun () ->
          let flat = Ops.cast Ops.O.((s0 * Ops.int 4) + s1) Float32 in
          let shared =
            Ops.O.(sum flat [ s0; s1 ] + sum (flat * two) [ s0; s1 ])
          in
          equal (list int) [ 12 ] (merged_size (store_at (Ops.int 0) shared)));
      test "reductions that share ranges keep them when one would divide"
        (fun () ->
          let flat = Ops.cast Ops.O.((s0 * Ops.int 4) + s1) Float32 in
          let apart = Ops.O.(Ops.cast s0 Float32 + Ops.cast s1 Float32) in
          let shared = Ops.O.(sum flat [ s0; s1 ] + sum apart [ s0; s1 ]) in
          equal (list int) [ 3; 4 ] (merged_size (store_at (Ops.int 0) shared)));
      cases "a range of a typed integer shrinks to its guard as a weak one"
        ~id:(Format.asprintf "%a" Dtype.pp)
        Dtype.[ Int32; Int64; Uint32 ]
        (fun dt ->
          let r = Ops.range ~axis_type:Weak ~dtype:dt (Int 204) [ 0 ] in
          let guard = Ops.O.(r < Ops.int ~dtype:dt 4) in
          equal (list int) [ 4 ]
            (merged_size (Ops.end_ (gated_load guard r) [ r ])));
      group "simplifies each case and kernel as its golden states"
        [
          rewrites_as_recorded "simplify_ranges" simplify;
          recorded "elementwise" "simplified" simplify;
          recorded "embedding_ranges" "simplified" simplify;
          recorded "matmul" "simplified" simplify;
          recorded "conv" "simplified" simplify;
          recorded "shrink_single_guard" "simplified" simplify;
          recorded "shrink_picks_max_guard" "simplified" simplify;
          recorded "shrink_guard_ge_max" "simplified" simplify;
          recorded "shrink_unguarded_elsewhere" "simplified" simplify;
          recorded "shrink_used_in_reduce" "simplified" simplify;
          recorded "shrink_to_single_iteration" "simplified" simplify;
          recorded "shrink_store_where_invalid" "simplified" simplify;
          recorded "shrink_store_where_invalid_flipped" "simplified" simplify;
        ];
      keeps_case_writes "simplify_ranges" ~prepared:initial_symbolic simplify;
      over ~count:4 "keeps what each recorded kernel writes"
        (recorded_kernels
           [ "elementwise"; "embedding_ranges"; "matmul"; "conv" ])
        simplify keeps_writes;
      prop "keeps what a kernel writes" generated_kernels (fun g ->
          let u = kernel_of g in
          let simplified = simplify u in
          cover "ranges merge"
            (List.length (ranges simplified) < List.length (ranges u));
          cover "a range shrinks"
            (List.length (ranges simplified) = List.length (ranges u)
            && List.fold_left ( + ) 0 (sizes simplified)
               < List.fold_left ( + ) 0 (sizes u));
          equal (list write) (writes u) (writes simplified));
    ]

let splitting =
  let r n = range n 0 in
  let modulo r c =
    kernel [ Ops.end_ (store_at r Ops.O.(r % Ops.int c)) [ r ] ]
  in
  group "pm_split_ranges"
    [
      test
        "the outer part is numbered 0 and the inner 1 under the range's \
         identity" (fun () ->
          equal
            (list (pair (list int) int))
            [ ([ 0; 0 ], 4); ([ 0; 1 ], 2) ]
            (List.sort compare
               (List.map
                  (fun r -> (Ops.axis_id r, size r))
                  (ranges (split (modulo (r 8) 2))))));
      cases "a range of a typed integer splits as a weak one"
        ~id:(Format.asprintf "%a" Dtype.pp)
        Dtype.[ Int32; Int64; Uint32 ]
        (fun dt ->
          let r = Ops.range ~axis_type:Weak ~dtype:dt (Int 8) [ 0 ] in
          let u =
            kernel [ Ops.end_ (store_at r Ops.O.(r % Ops.int 2)) [ r ] ]
          in
          equal (list int) [ 2; 4 ] (sizes (split u)));
      cases "a remainder by a constant that is not an integer leaves the range"
        ~id:(Format.asprintf "%a" Ops.pp)
        [ Ops.bool true; Ops.float 2. ]
        (fun c ->
          let r = r 8 in
          let u =
            kernel [ Ops.end_ (store_at r (Ops.alu r Mod [ c ])) [ r ] ]
          in
          equal uop u (split u));
      group "splits each case and kernel as its golden states"
        [
          rewrites_as_recorded "split_ranges" split;
          recorded "transpose" "split" split;
          recorded "repeat" "split" split;
        ];
      keeps_case_writes "split_ranges" split;
      over ~count:4 "keeps what each recorded kernel writes"
        (recorded_kernels [ "transpose"; "repeat" ])
        split keeps_writes;
      prop "keeps what a kernel writes" generated_kernels (fun g ->
          let u = kernel_of g in
          let split = split u in
          cover "a range splits"
            (List.length (ranges split) > List.length (ranges u));
          equal (list write) (writes u) (writes split));
    ]

(* Effects *)

(* The storage each store of [u] writes through. *)
let stored_through u =
  List.filter_map
    (fun n -> if Ops.op n = Store then Some (Ops.nth (Ops.nth n 0) 0) else None)
    (Ops.toposort ~calls:Enter u)

let effects =
  let r = range 8 0 in
  let out = buf Float32 in
  (* A store at [r / 8] over [r] of size 8: the symbolic rules fold its index to
     [0], and the store still runs once per step of [r]. *)
  let eighth =
    kernel
      [
        Ops.end_
          (Ops.store (Ops.index out [ Ops.O.(r // Ops.int 8) ]) one)
          [ r ];
      ]
  in
  group "pin_effects and pm_unpin_effects"
    [
      test "an effect whose index loses a range keeps reading after it"
        (fun () ->
          let folded =
            unpin (rewrite Symbolic.sym (Simplify.pin_effects eighth))
          in
          equal (list uop) [ Ops.after out [ r ] ] (stored_through folded));
      prop "pm_unpin_effects undoes pin_effects where no range was removed"
        generated_kernels (fun g ->
          let u = kernel_of g in
          let pinned = Simplify.pin_effects u in
          cover "pin_effects reads storage after ranges"
            (not (Ops.equal u pinned));
          equal uop u (unpin pinned));
    ]

(* Reductions *)

(* Reductions of a parameter over a range it ignores, whose size is a variable
   from 0 to a bound, as a symbolic shape's is: each operation, value and
   bound. *)
let reductions_by_size =
  List.concat_map
    (fun rop ->
      List.concat_map
        (fun x -> List.map (fun hi -> (rop, x, hi)) [ 1; 2; 4 ])
        [ Float.infinity; Float.neg_infinity; Float.nan; 1.5; 0. ])
    [ Op.Add; Op.Mul; Op.Max ]

let pp_reduction (rop, x, hi) =
  Format.asprintf "%a of %g below n <= %d" Op.pp rop x hi

(* The reduction is evaluated at every size up to its bound, 0 included. A
   maximum of NaN may be NaN or the identity (Tolk.Op.Reduce), so both the
   reduction and its rewrite are one of the two; every other value is one. *)
let reduces_by_size (rop, x, hi) =
  let n = var "n" 0 hi and value = Shape.param 0 Float32 in
  let r = Ops.range ~axis_type:Weak (Sym n) [ 0 ] in
  let u = Ops.reduce value rop [ r ] in
  let at size u =
    let vars = [ ("n", i size) ] and params = [ (0, `Float x) ] in
    by_value (Interpreter.eval ~vars ~params u)
  in
  let either = function
    | `Float f -> Float.is_nan f || f = Float.neg_infinity
    | _ -> false
  in
  List.iter
    (fun size ->
      let msg = Printf.sprintf "at n = %d" size in
      if rop = Op.Max && Float.is_nan x && size > 0 then
        satisfies ~msg ~claim:"NaN or the identity"
          (pair Dtypes.const Dtypes.const)
          (fun (a, b) -> either a && either b)
          (at size u, at size (unparent u))
      else equal ~msg Dtypes.const (at size u) (at size (unparent u)))
    (List.init (hi + 1) Fun.id)

(* Generated sums: a value over a range of size 0 to 8, selected by bounds of
   the range, as the rules that count a sum see them. *)
type bound = Constant of int | Variable | Outer

type selection =
  | Below of bound
  | Above of bound
  | Between of bound * bound
  | Shifted of int * bound
  | Scaled of int * bound
  | Mask of bound

type summand = Constant_value | Variable_value | Outer_value | Range_value

let pp_sum ppf (n, s, v, dt) =
  let pp_bound ppf = function
    | Constant c -> Format.pp_print_int ppf c
    | Variable -> Format.pp_print_string ppf "b"
    | Outer -> Format.pp_print_string ppf "o"
  in
  let pp_selection ppf = function
    | Below b -> Format.fprintf ppf "r < %a" pp_bound b
    | Above b -> Format.fprintf ppf "!(r < %a)" pp_bound b
    | Between (lo, hi) ->
        Format.fprintf ppf "%a <= r < %a" pp_bound lo pp_bound hi
    | Shifted (k, b) -> Format.fprintf ppf "r + %d < %a" k pp_bound b
    | Scaled (k, b) -> Format.fprintf ppf "r * %d < %a" k pp_bound b
    | Mask b -> Format.fprintf ppf "x * cast(r < %a)" pp_bound b
  in
  let summand = function
    | Constant_value -> "3"
    | Variable_value -> "v"
    | Outer_value -> "o"
    | Range_value -> "r"
  in
  Format.fprintf ppf "sum over r < %d of %s where %a, %a" n (summand v)
    pp_selection s Dtype.pp dt

let bound =
  Gen.frequency
    [
      (3, Gen.map (fun c -> Constant c) (Gen.int_range (-2) 12));
      (1, Gen.constant Variable);
      (1, Gen.constant Outer);
    ]

let selection ~mask =
  let open Gen in
  one_of
    ([
       map (fun b -> Below b) bound;
       map (fun b -> Above b) bound;
       map (fun (lo, hi) -> Between (lo, hi)) (pair bound bound);
       map (fun (k, b) -> Shifted (k, b)) (pair (int_range (-3) 3) bound);
       map (fun (k, b) -> Scaled (k, b)) (pair (int_range 1 3) bound);
     ]
    @ if mask then [ map (fun b -> Mask b) bound ] else [])

let sum_size = Gen.(one_of [ constant 0; int_range 1 8 ])
let sum_dtype = Gen.of_list [ Dtype.Float32; Int32 ]

let sums =
  let open Gen in
  with_pp pp_sum
    (quad sum_size (selection ~mask:true)
       (of_list [ Constant_value; Variable_value; Outer_value; Range_value ])
       sum_dtype)

(* Sums of a value that is no function of the range, which a closed form
   computes: a float product by a mask is left, as it is no selection at an
   infinity or a negative value. *)
let closed_sums =
  let open Gen in
  with_pp pp_sum
    (let* dt = sum_dtype in
     let+ n = sum_size
     and+ s = selection ~mask:(not (Dtype.is_float dt))
     and+ v = of_list [ Constant_value; Variable_value; Outer_value ] in
     (n, s, v, dt))

let sum_of (n, s, v, dt) =
  let r = range n 0 and o = range 6 1 in
  let bound = function
    | Constant c -> Ops.int c
    | Variable -> var "b" (-2) 12
    | Outer -> o
  in
  let value =
    match v with
    | Constant_value -> Ops.int ~dtype:dt 3
    | Variable_value -> var ~dtype:dt "v" (-4) 9
    | Outer_value -> Ops.cast o dt
    | Range_value -> Ops.cast r dt
  in
  let zero = Ops.const (`Int Bigint.zero) in
  let select c = Ops.where c value zero in
  let body =
    match s with
    | Below b -> select Ops.O.(r < bound b)
    | Above b -> Ops.where Ops.O.(r < bound b) zero value
    | Between (lo, hi) ->
        select Ops.O.(Ops.logical_not (r < bound lo) land (r < bound hi))
    | Shifted (k, b) -> select Ops.O.(r + Ops.int k < bound b)
    | Scaled (k, b) -> select Ops.O.(r * Ops.int k < bound b)
    | Mask b -> Ops.O.(value * Ops.cast (r < bound b) dt)
  in
  sum body [ r ]

(* A generated sum with choices for its leaves. *)
let bound_sums =
  let open Gen in
  let* s = sums in
  let+ cs = list ~size:(constant (List.length (leaves (sum_of s)))) choice in
  (s, cs)

let sum_keeps_value pass (s, cs) =
  let u = sum_of s in
  match binding cs u with
  | Some b -> keeps_value "the sum" b u (pass u)
  | None -> assume false

let unparented =
  let r0 = range 4 0 and r1 = range 5 1 in
  let x = Ops.cast r0 Float32 in
  let reduce src = Ops.v ~src ~arg:(Reduce { op = Add; num_axes = 0 }) Reduce in
  group "pm_reduce_unparented"
    [
      test "a reduction by another operation is left" (fun () ->
          let bits = Ops.cast r0 Int32 in
          let u =
            Ops.v ~src:[ bits; r0; r1 ]
              ~arg:(Reduce { op = Or; num_axes = 0 })
              Reduce
          in
          equal uop u (unparent u));
      test "a constant source, a range of one trip folded, is dropped"
        (fun () ->
          equal uop (sum x [ r0 ]) (unparent (reduce [ x; r0; Ops.int 0 ])));
      test "a source that is neither a range nor a constant is refused"
        (fun () ->
          raises_match (Exn.invalid_arg ?sub:None) (fun () ->
              unparent (reduce [ x; Ops.O.(r0 + Ops.int 1) ])));
      cases "a reduction keeps its value over a range that may run no time"
        ~id:pp_reduction reductions_by_size reduces_by_size;
      group "removes each case's ranges as its golden states"
        [ rewrites_as_recorded "reduce_unparented" unparent ];
      keeps_values "reduce_unparented" unparent;
    ]

let collapsing =
  let r = range 10 0 in
  let u8 = var ~dtype:Uint8 "u" 0 255 and v8 = var ~dtype:Uint8 "v" 1 255 in
  let same_value pass g vars params =
    equal Dtypes.const
      (Interpreter.eval ~vars ~params g)
      (Interpreter.eval ~vars ~params (pass g))
  in
  group "pm_reduce_collapse"
    [
      test "a comparison of a product is solved by a rounded-up division"
        (fun () ->
          equal uop
            Ops.O.(r < Ops.int 5)
            (collapse Ops.O.(r * Ops.int 3 < Ops.int 15));
          equal uop
            Ops.O.(r < Ops.int 6)
            (collapse Ops.O.(r * Ops.int 3 < Ops.int 16)));
      cases "a comparison of an unsigned sum or product is left, as it wraps"
        ~id:fst
        (let x = var ~dtype:Uint32 "x" 1 100
         and y = var ~dtype:Uint32 "y" 1 100
         and c = var ~dtype:Uint32 "c" 1 100 in
         Ops.O.[ ("x + y < c", x + y < c); ("x * y < c", x * y < c) ])
        (fun (_, lt) -> equal uop lt (collapse lt));
      test "a comparison of an int32 sum is solved for x" (fun () ->
          let x = var ~dtype:Int32 "x" 1 100
          and y = var ~dtype:Int32 "y" 1 100
          and c = var ~dtype:Int32 "c" 1 100 in
          equal uop Ops.O.(x < c - y) (collapse Ops.O.(x + y < c)));
      test "x + y < c is solved only where nothing wraps" (fun () ->
          same_value collapse
            Ops.O.(u8 + v8 < Ops.int 50)
            [ ("u", i 10); ("v", i 100) ]
            []);
      test "x * y < c is solved only where nothing wraps" (fun () ->
          same_value collapse
            Ops.O.(u8 * v8 < Ops.int 50)
            [ ("u", i 100); ("v", i 3) ]
            []);
      test "a float x + y < c is left, where moving y rounds" (fun () ->
          let fvar name = var ~dtype:Float32 name 0 1 in
          let lt = Ops.O.(fvar "x" + fvar "y" < fvar "c") in
          equal uop lt (collapse lt));
      test "a count by variable bounds is clamped at zero" (fun () ->
          let lo = var "lo" 0 20 and hi = var "hi" 0 20 in
          let between = Ops.O.(Ops.logical_not (r < lo) land (r < hi)) in
          let u = collapse (sum (Ops.where between two zero) [ r ]) in
          sums_to ~vars:[ ("lo", i 7); ("hi", i 3) ] (`Float 0.) u;
          sums_to ~vars:[ ("lo", i 3); ("hi", i 30) ] (`Float 14.) u);
      test
        "a float sum over an empty part of a range is +0., whatever the value"
        (fun () ->
          let r = range 4 0 and k = var "k" 0 4 and x = Shape.param 0 Float32 in
          let u = sum (Ops.where Ops.O.(r < k) x zero) [ r ] in
          List.iter
            (fun (kv, xv) ->
              same_value collapse u [ ("k", i kv) ] [ (0, `Float xv) ])
            [ (0, Float.infinity); (0, Float.nan); (2, 1.5) ]);
      test
        "an integer product by a comparison cast from a boolean is a selection"
        (fun () ->
          let c = Ops.O.(r < Ops.int 3) in
          equal uop
            (Ops.where c (i32 2) (Ops.int 0))
            (collapse Ops.O.(i32 2 * Ops.cast c Int32)));
      test "a float product by a comparison cast from a boolean is no selection"
        (fun () ->
          let u = Ops.O.(two * Ops.cast Ops.O.(r < Ops.int 3) Float32) in
          equal int 0 (count Where (collapse u)));
      test "a float product by a boolean mask keeps its value" (fun () ->
          let x = Shape.param 0 Float32 and g = Shape.param 1 Bool in
          let u = Ops.O.(x * Ops.cast g Float32) in
          List.iter
            (fun xv ->
              same_value collapse u [] [ (0, `Float xv); (1, `Bool false) ])
            [ -1.; Float.infinity ]);
      test "a float sum lifted past a parameter is +0. where it is false"
        (fun () ->
          let r = range 4 0
          and x = Shape.param 0 Float32
          and g = Shape.param 1 Bool in
          let u = sum (Ops.where Ops.O.(g land (r < Ops.int 3)) x zero) [ r ] in
          List.iter
            (fun xv ->
              same_value collapse u [] [ (0, `Float xv); (1, `Bool false) ])
            [ -1.; Float.infinity ]);
      test "a sum over a range of size 1 is its value where the range is 0"
        (fun () ->
          let once = range 1 2 in
          sums_to (`Float 2.)
            (collapse
               (sum (Ops.where Ops.O.(once < Ops.int 1) two zero) [ once ])));
      group "collapses each case as its golden states"
        [ rewrites_as_recorded "reduce_collapse" collapse ];
      keeps_values "reduce_collapse" collapse;
      prop ~count:300 "keeps the value of a sum" bound_sums
        (sum_keeps_value collapse);
    ]

let reduce_simplifying =
  let r = range 10 0 and o = range 6 1 in
  group "pm_reduce_simplify"
    [
      test "a sum whose value is a function of its range has a closed form"
        (fun () ->
          let u =
            reduce_simplify (sum (Ops.where Ops.O.(r < o) two zero) [ r ])
          in
          equal int 0 (count Reduce u);
          (* Two for each step of [r] below [o], at each value of [o]. *)
          equal (list Dtypes.const)
            (List.init 6 (fun k -> `Float (2. *. float_of_int k)))
            (List.init 6 (fun k -> Interpreter.eval ~vars:[ ("r1", i k) ] u)));
      test "a sum left with a range after collapsing is left" (fun () ->
          let u =
            sum
              (Ops.where Ops.O.(r < Ops.int 3) (Ops.cast r Float32) zero)
              [ r ]
          in
          equal uop u (reduce_simplify u));
      test "a sum over two ranges collapses each in turn" (fun () ->
          let s = range 4 2 in
          sums_to (`Float 12.)
            (reduce_simplify
               (sum (Ops.where Ops.O.(r < Ops.int 3) one zero) [ r; s ])));
      test "a sum holding a store is left" (fun () ->
          let u = sum (Ops.after two [ store_at r one ]) [ r ] in
          equal uop u (reduce_simplify u));
      test "a sum holding a reduction is left while that reduction stays"
        (fun () ->
          let s = range 4 2 in
          let inner = sum (Ops.cast Ops.O.(s + r) Float32) [ s ] in
          let u = sum inner [ r ] in
          equal uop u (reduce_simplify u));
      cases "a sum over a range of size 1 is its value where it is 0, of"
        ~id:(Format.asprintf "%a" Dtype.pp)
        Dtype.[ Weak_int; Int32; Int64; Uint32 ]
        (fun dt ->
          let once = Ops.range ~axis_type:Weak ~dtype:dt (Int 1) [ 2 ] in
          let below = Ops.O.(once < Ops.int ~dtype:dt 1) in
          sums_to (`Float 2.)
            (reduce_simplify (sum (Ops.where below two zero) [ once ])));
      test "a maximum is left" (fun () ->
          let u =
            Ops.reduce (Ops.where Ops.O.(r < Ops.int 3) two zero) Max [ r ]
          in
          equal uop u (reduce_simplify u));
      test "an unparented range is removed" (fun () ->
          (* Ten times the value where the range runs, the identity where it
             does not. *)
          let counted =
            Ops.where
              Ops.O.(Ops.int 0 < Ops.int 10)
              Ops.O.(f32 3. * Ops.int 10)
              (Shape.const_like (f32 3.) (Ops.identity_element Add Float32))
          in
          equal uop counted (reduce_simplify (sum (f32 3.) [ r ])));
      group "computes each case and kernel as its golden states"
        [
          rewrites_as_recorded "reduce_simplify" reduce_simplify;
          recorded "arange" "collapsed" reduce_simplify;
          recorded "arange_sum" "collapsed" reduce_simplify;
          recorded "arange_index" "collapsed" reduce_simplify;
          recorded "triu" "collapsed" reduce_simplify;
          recorded "one_hot" "collapsed" reduce_simplify;
          recorded "sum_zero" "collapsed" reduce_simplify;
          recorded "arange_transposed" "collapsed" reduce_simplify;
          recorded "cumsum" "collapsed" reduce_simplify;
        ];
      keeps_values "reduce_simplify" reduce_simplify;
      prop ~count:300 "keeps the value of a sum" bound_sums
        (sum_keeps_value reduce_simplify);
      prop "computes a sum of a value that is no function of its range"
        closed_sums (fun s ->
          equal int 0 (count Reduce (reduce_simplify (sum_of s))));
    ]

(* A sum over [r] of [table] at [r], where [r] is [label]: [table] at [label]
   where [label] is within the range, [0] elsewhere. The table's elements are [j
   * j + 1], none of them [0]. *)
let table_size = 10
let table = buf ~size:table_size Float32

let elements =
  [ (0, Array.init table_size (fun j -> `Float (float_of_int ((j * j) + 1)))) ]

let label = var "label" (-3) 12
let labels = List.init 16 (fun k -> k - 3)

let selected r =
  sum (Ops.where Ops.O.(label <> r) zero (Ops.index table [ r ])) [ r ]

let at_labels u =
  List.map
    (fun v -> Interpreter.eval ~vars:[ ("label", i v) ] ~buffers:elements u)
    labels

(* What a sum over a range of [n] steps selects at each label. *)
let selected_values n =
  List.map
    (fun v ->
      `Float (if v >= 0 && v < n then float_of_int ((v * v) + 1) else 0.))
    labels

let load_collapsing_value =
  let r = range table_size 0 and s = range 4 1 in
  group "load_collapse"
    [
      cases "is None for" ~id:fst
        [
          ( "a maximum",
            Ops.reduce
              (Ops.where Ops.O.(label <> r) zero (Ops.index table [ r ]))
              Max [ r ] );
          ( "a sum over two ranges",
            sum (Ops.cast Ops.O.(r + s) Float32) [ r; s ] );
          ("a sum no form computes", sum (Ops.index table [ r ]) [ r ]);
        ]
        (fun (_, u) -> is_none ~pp:Ops.pp (Simplify.load_collapse u));
      cases "computes a sum selected where an index equals a range of size"
        ~id:string_of_int [ table_size; 1 ] (fun n ->
          let u =
            require_some (Simplify.load_collapse (selected (range n 2)))
          in
          equal (list int) [] (sizes u);
          equal (list Dtypes.const) (selected_values n) (at_labels u));
    ]

let load_collapsing =
  let r = range table_size 0 in
  group "pm_load_collapse"
    [
      test "selects the element at an index within the range, and zero outside"
        (fun () ->
          equal (list Dtypes.const)
            (selected_values table_size)
            (at_labels (load_collapse (selected r))));
      test "a sum whose index is read at its range adds each trip it selects"
        (fun () ->
          let id = Ops.index (buf ~slot:1 ~size:10 Int32) [ r ] in
          let u =
            sum
              (Ops.where
                 Ops.O.(id <> Ops.cast r Int32)
                 zero (Ops.index table [ r ]))
              [ r ]
          in
          (* The index equals the range at 0, 2 and 4, whose elements add to 1 +
             5 + 17. *)
          let ids = Array.map i [| 0; 5; 2; 1; 4; 0; 9; 3; 3; 8 |] in
          equal Dtypes.const (`Float 23.)
            (Interpreter.eval ~buffers:((1, ids) :: elements) (load_collapse u)));
      test "a comparison of a shifted index read from memory is solved for it"
        (fun () ->
          let loaded =
            Ops.cast
              (Ops.index (buf ~slot:1 ~size:4 Int32) [ Ops.int 0 ])
              Weak_int
          in
          equal uop
            Ops.O.(loaded < Ops.int 20 - Ops.int 5)
            (load_collapse Ops.O.(loaded + Ops.int 5 < Ops.int 20)));
      test "a comparison of a shifted index of a committed type is left"
        (fun () ->
          let loaded = Ops.index (buf ~slot:1 ~size:4 Int32) [ Ops.int 0 ] in
          let u = Ops.O.(loaded + i32 5 < i32 20) in
          equal uop u (load_collapse u));
      (* The selected value compares two elements read at the index, [x + 1 =
         c]: neither runs inside a range once the range is the index, so the sum
         is not lifted out of the equality, which would lift it back. *)
      test "an equality of two values read at the selected index is reached"
        (fun () ->
          let a = buf ~slot:1 ~size:10 Uint8
          and b = buf ~slot:2 ~size:10 Uint8 in
          let at t = Ops.cast (Ops.index t [ r ]) Int32 in
          let hit = Ops.cast (Ops.eq Ops.O.(at b + i32 1) (at a)) Float32 in
          let u = sum (Ops.where Ops.O.(label <> r) zero hit) [ r ] in
          let bytes l = Array.of_list (List.map (fun n -> i n) l) in
          let buffers =
            [
              (1, bytes [ 1; 2; 3; 4; 5; 6; 7; 8; 9; 10 ]);
              (2, bytes [ 0; 2; 2; 4; 4; 6; 6; 8; 8; 10 ]);
            ]
          in
          let at_label v u =
            Interpreter.eval ~vars:[ ("label", i v) ] ~buffers u
          in
          let collapsed = load_collapse u in
          equal (list int) [] (sizes collapsed);
          List.iter
            (fun v -> equal Dtypes.const (at_label v u) (at_label v collapsed))
            [ 0; 1; 2; 9 ]);
      test "x + y <> c under a narrowing cast keeps its value" (fun () ->
          let y = var ~dtype:Int32 "y" 250 260 in
          let x = Ops.cast r Int32 in
          let hit = Ops.O.(Ops.cast (x + y) Uint8 <> Ops.int ~dtype:Uint8 2) in
          let counted = sum (Ops.where hit (f32 1.) zero) [ r ] in
          let vars = [ ("y", i 250) ] in
          equal Dtypes.const
            (Interpreter.eval ~vars counted)
            (Interpreter.eval ~vars (load_collapse counted)));
      group "collapses each case and kernel as its golden states"
        [
          rewrites_as_recorded "load_collapse" load_collapse;
          recorded "gather" "collapsed" load_collapse;
          recorded "embedding" "collapsed" load_collapse;
        ];
      keeps_values "load_collapse" load_collapse;
    ]

let () =
  exit
    (run "Tolk.Simplify"
       [
         flattening;
         simplifying;
         splitting;
         effects;
         unparented;
         collapsing;
         reduce_simplifying;
         load_collapsing_value;
         load_collapsing;
       ])
