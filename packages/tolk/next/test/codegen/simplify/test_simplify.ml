(* Tests of Tolk_next.Simplify: each rewrite its interface states, the law that
   the reductions' rewrites keep a value, the kernels tinygrad compiles, and the
   cases tinygrad's rewrites were recorded on. *)

open Windtrap
open Tolk_next
open Common

let one = f32 1.
let two = f32 2.
let zero = Ops.float 0.

(* Flattening *)

let flattening =
  let r0 = range 3 0 and r1 = range 4 1 in
  let s0 = reduce_range 3 2 and s1 = reduce_range 4 3 in
  let value = Ops.cast Ops.O.((s0 * Ops.int 4) + s1) Float32 in
  let at = store_at Ops.O.((r0 * Ops.int 4) + r1) one in
  group "pm_flatten_range"
    [
      test "a reduction's sources become the ranges they run inside" (fun () ->
          equal uop
            (sum value [ s0; s1 ])
            (flatten (sum value [ Ops.O.((s0 * Ops.int 4) + s1) ])));
      test "an end's sources become the ranges they run inside" (fun () ->
          equal uop
            (Ops.end_ at [ r0; r1 ])
            (flatten (Ops.end_ at [ Ops.O.((r0 * Ops.int 4) + r1) ])));
      test "a range listed twice is listed once" (fun () ->
          equal uop (sum value [ s0; s1 ]) (flatten (sum value [ s0; s1; s0 ]));
          equal uop
            (Ops.end_ at [ r0; r1 ])
            (flatten (Ops.end_ at [ r0; r1; Ops.O.(r0 + Ops.int 1) ])));
      test "ranges keep the order they are listed in" (fun () ->
          let late = Ops.range (Sym Ops.O.(r0 + Ops.int 1)) [ 4 ] in
          let u =
            Ops.end_ (store_at Ops.O.((r0 * Ops.int 4) + late) one) [ late; r0 ]
          in
          equal uop u (flatten u));
      test "a source that runs inside no range is dropped" (fun () ->
          equal (list uop) [ at ]
            (Ops.src (flatten (Ops.v ~src:[ at; Ops.int 3 ] End))));
    ]

(* Range simplification *)

let merged_size u = sizes (simplify (kernel [ u ]))

let merging =
  let r0 = range 3 0 and r1 = range 4 1 and r2 = range 5 2 in
  let s0 = reduce_range 3 3 and s1 = reduce_range 4 4 in
  let summed value rs =
    store_at (Ops.int 0) (sum (Ops.cast value Float32) rs)
  in
  group "pm_simplify_ranges › merging"
    [
      test "adjacent ranges indexed contiguously merge into their product"
        (fun () ->
          equal (list int) [ 12 ]
            (merged_size
               (Ops.end_
                  (store_at Ops.O.((r0 * Ops.int 4) + r1) one)
                  [ r0; r1 ])));
      test "the merged range keeps the first range's identity" (fun () ->
          let merged =
            simplify
              (kernel
                 [
                   Ops.end_
                     (store_at Ops.O.((r0 * Ops.int 4) + r1) one)
                     [ r0; r1 ];
                 ])
          in
          equal
            (list (list int))
            [ [ 0 ] ]
            (List.map Ops.axis_id (ranges merged)));
      test "ranges a kernel does not read merge" (fun () ->
          equal (list int) [ 60 ]
            (merged_size (Ops.end_ (store_at (Ops.int 0) one) [ r0; r1; r2 ])));
      test "ranges whose merge adds a division stay" (fun () ->
          equal (list int) [ 3; 4 ]
            (merged_size (Ops.end_ (store_at r0 one) [ r0; r1 ])));
      test "an end merges only ranges next to each other" (fun () ->
          let index = Ops.O.((r0 * Ops.int 20) + (r2 * Ops.int 4) + r1) in
          equal (list int) [ 3; 4; 5 ]
            (merged_size (Ops.end_ (store_at index one) [ r0; r1; r2 ])));
      test "a reduction merges its ranges in either order" (fun () ->
          equal (list int) [ 12 ]
            (merged_size (summed Ops.O.((s1 * Ops.int 3) + s0) [ s0; s1 ])));
      test "ranges of different axis types stay" (fun () ->
          let loop = range ~axis_type:Loop 4 5 in
          equal (list int) [ 3; 4 ]
            (merged_size
               (Ops.end_
                  (store_at Ops.O.((r0 * Ops.int 4) + loop) one)
                  [ r0; loop ])));
      test "ranges reduced by different reductions stay" (fun () ->
          let inner = sum (Ops.cast s0 Float32) [ s0 ] in
          let outer = sum Ops.O.(inner + Ops.cast s1 Float32) [ s1 ] in
          equal (list int) [ 3; 4 ] (merged_size (store_at (Ops.int 0) outer)));
    ]

let shrinking =
  let r = range 204 0 in
  let shrunk u = sizes (simplify (kernel [ u ])) in
  group "pm_simplify_ranges › shrinking"
    [
      test "a range every index guards below a constant shrinks to it"
        (fun () ->
          equal (list int) [ 4 ]
            (shrunk (Ops.end_ (gated_load Ops.O.(r < Ops.int 4) r) [ r ])));
      test "a range shrinks to the greatest of its guards" (fun () ->
          let load c = gated_load Ops.O.(r < Ops.int c) r in
          equal (list int) [ 8 ]
            (shrunk (Ops.end_ (Ops.sink [ load 4; load 8 ]) [ r ])));
      test "a range an index reads unguarded stays" (fun () ->
          let plain = Ops.load (Ops.index (buf ~slot:1 Float32) [ r ]) [] in
          let value = Ops.O.(gated_load (r < Ops.int 4) r + plain) in
          equal (list int) [ 204 ] (shrunk (Ops.end_ value [ r ])));
      test "a guard by a variable is no guard" (fun () ->
          equal (list int) [ 204 ]
            (shrunk (Ops.end_ (gated_load Ops.O.(r < var "c" 1 8) r) [ r ])));
      test "a reduction's range stays" (fun () ->
          let s = reduce_range 5 1 in
          let value =
            Ops.O.(Ops.cast s Float32 + gated_load (s < Ops.int 2) s)
          in
          equal (list int) [ 5 ]
            (shrunk (store_at (Ops.int 0) (sum value [ s ]))));
      test "each range of a conjunction of guards shrinks" (fun () ->
          let q = range 8 1 in
          let valid = Ops.O.((r < Ops.int 4) land (q < Ops.int 2)) in
          let index = Ops.O.((r * Ops.int 8) + q) in
          equal (list int) [ 2; 4 ]
            (shrunk (Ops.end_ (gated_load valid index) [ r; q ])));
      test "a store's gate is no guard" (fun () ->
          let store =
            Ops.store
              ~gate:Ops.O.(r < Ops.int 200)
              (Ops.index (buf Float32) [ r ])
              one
          in
          equal (list int) [ 204 ] (shrunk (Ops.end_ store [ r ])));
      test "a sink without kernel information is left as it is" (fun () ->
          let u =
            Ops.sink [ Ops.end_ (gated_load Ops.O.(r < Ops.int 4) r) [ r ] ]
          in
          equal uop u (simplify u));
      test "the kernel's sink empties the context" (fun () ->
          let ctx = Ops.Tbl.create 8 in
          let u =
            kernel [ Ops.end_ (gated_load Ops.O.(r < Ops.int 4) r) [ r ] ]
          in
          ignore (Ops.graph_rewrite ~ctx u Simplify.pm_simplify_ranges);
          equal int 0 (Ops.Tbl.length ctx));
    ]

(* Range splitting *)

let splitting =
  let r n = range n 0 in
  let modulo r c =
    kernel [ Ops.end_ (store_at r Ops.O.(r % Ops.int c)) [ r ] ]
  in
  let split_sizes u = sizes (split u) in
  group "pm_split_ranges"
    [
      test "a range taken modulo a divisor of its size splits in two" (fun () ->
          equal (list int) [ 2; 4 ] (split_sizes (modulo (r 8) 2));
          equal (list int) [ 3; 4 ] (split_sizes (modulo (r 12) 4)));
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
      test "a range taken modulo a number that does not divide its size stays"
        (fun () -> equal (list int) [ 7 ] (split_sizes (modulo (r 7) 3)));
      test "a truncating remainder is no modulo" (fun () ->
          let r = r 12 in
          let u =
            kernel
              [ Ops.end_ (store_at r (Ops.alu r Cmod [ Ops.int 4 ])) [ r ] ]
          in
          equal uop u (split u));
      test "a range of symbolic size stays" (fun () ->
          let r = Ops.range (Sym (var "n" 1 16)) [ 0 ] in
          let u = modulo r 2 in
          equal uop u (split u));
      test "warp and device ranges stay" (fun () ->
          List.iter
            (fun axis_type ->
              let u = modulo (range ~axis_type 8 0) 2 in
              equal uop u (split u))
            [ Ops.Axis_type.Warp; Device ]);
      test "a sink without kernel information is left as it is" (fun () ->
          let r = r 8 in
          let u =
            Ops.sink [ Ops.end_ (store_at r Ops.O.(r % Ops.int 2)) [ r ] ]
          in
          equal uop u (split u));
    ]

(* Reductions *)

let unparented =
  let r0 = reduce_range 4 0 and r1 = reduce_range 5 1 in
  let x = Ops.cast r0 Float32 in
  group "pm_reduce_unparented"
    [
      test "a sum over a range its value ignores is multiplied by its size"
        (fun () ->
          equal uop
            Ops.O.(sum x [ r0 ] * Ops.int 5)
            (unparent (sum x [ r0; r1 ])));
      test "a product is raised to its size" (fun () ->
          equal uop
            (Ops.pow (Ops.reduce x Mul [ r0 ]) (Ops.int 5))
            (unparent (Ops.reduce x Mul [ r0; r1 ])));
      test "a maximum drops the range" (fun () ->
          equal uop (Ops.reduce x Max [ r0 ])
            (unparent (Ops.reduce x Max [ r0; r1 ])));
      test "a reduction over no range its value reads is its value, scaled"
        (fun () ->
          equal uop Ops.O.(f32 3. * Ops.int 4) (unparent (sum (f32 3.) [ r0 ])));
      test "a reduction over ranges its value reads is left" (fun () ->
          let u = sum Ops.O.(x + Ops.cast r1 Float32) [ r0; r1 ] in
          equal uop u (unparent u));
      test "a reduction by another operation is left" (fun () ->
          let bits = Ops.cast r0 Int32 in
          let u =
            Ops.v ~src:[ bits; r0; r1 ]
              ~arg:(Reduce { op = Or; num_axes = 0 })
              Reduce
          in
          equal uop u (unparent u));
      test "a source that is not a range is refused" (fun () ->
          raises_match (Exn.invalid_arg ?substring:None) (fun () ->
              unparent
                (Ops.v
                   ~src:[ x; Ops.O.(r0 + Ops.int 1) ]
                   ~arg:(Reduce { op = Add; num_axes = 0 })
                   Reduce)));
    ]

(* [sums_to v u] is the claim that [u] has no range left and is worth [v]. *)
let sums_to ?(vars = []) v u =
  equal (list int) [] (sizes u);
  equal Dtypes.const v (Interpreter.eval ~vars u)

let collapsing =
  let r = reduce_range 10 0 in
  group "pm_reduce_collapse"
    [
      test "a comparison of a sum is solved for its range" (fun () ->
          equal uop
            Ops.O.(r < Ops.int 5)
            (collapse Ops.O.(r + Ops.int 2 < Ops.int 7)));
      test "a comparison of a product is solved by a rounded-up division"
        (fun () ->
          equal uop
            Ops.O.(r < Ops.int 5)
            (collapse Ops.O.(r * Ops.int 3 < Ops.int 15));
          equal uop
            Ops.O.(r < Ops.int 6)
            (collapse Ops.O.(r * Ops.int 3 < Ops.int 16)));
      test "a comparison by a range-dependent term is left" (fun () ->
          let o = range 6 1 in
          let u = Ops.O.(r + o < Ops.int 7) in
          equal uop u (collapse u));
      test "a product by a factor that can be zero is left" (fun () ->
          let u =
            Ops.O.(r * var ~dtype:Int32 "y" 0 20 < var ~dtype:Int32 "x" 0 20)
          in
          equal uop u (collapse u));
      test "a sum of a value below a bound is the bound times the value"
        (fun () ->
          sums_to (`Float 6.)
            (collapse (sum (Ops.where Ops.O.(r < Ops.int 3) two zero) [ r ])));
      test "a sum of a value above a bound counts the rest" (fun () ->
          sums_to (`Float 14.)
            (collapse (sum (Ops.where Ops.O.(r < Ops.int 3) zero two) [ r ])));
      test "a sum between two bounds counts what lies between" (fun () ->
          let between =
            Ops.O.(Ops.logical_not (r < Ops.int 2) land (r < Ops.int 7))
          in
          sums_to (`Float 10.)
            (collapse (sum (Ops.where between two zero) [ r ])));
      test "a bound beyond the range counts the whole range" (fun () ->
          sums_to (`Float 20.)
            (collapse (sum (Ops.where Ops.O.(r < Ops.int 30) two zero) [ r ])));
      test "a count by variable bounds is clamped at zero" (fun () ->
          let lo = var "lo" 0 20 and hi = var "hi" 0 20 in
          let between = Ops.O.(Ops.logical_not (r < lo) land (r < hi)) in
          let u = collapse (sum (Ops.where between two zero) [ r ]) in
          sums_to ~vars:[ ("lo", i 7); ("hi", i 3) ] (`Float 0.) u;
          sums_to ~vars:[ ("lo", i 3); ("hi", i 30) ] (`Float 14.) u);
      test "a sum of a selected zero of a committed type is left" (fun () ->
          let u = sum (Ops.where Ops.O.(r < Ops.int 3) two (f32 0.)) [ r ] in
          equal uop u (collapse u));
      test "a sum of a sum is the sum of the sums" (fun () ->
          let x = Ops.cast r Float32 in
          equal uop
            Ops.O.(sum x [ r ] + Ops.float 20.)
            (collapse (sum Ops.O.(x + two) [ r ])));
      test "a product by a comparison cast from a boolean is a selection"
        (fun () ->
          let c = Ops.O.(r < Ops.int 3) in
          equal uop (Ops.where c two zero)
            (collapse Ops.O.(two * Ops.cast c Float32)));
      test "a parameter guarding a sum is lifted out of it" (fun () ->
          let p = Ops.param 0 Bool in
          let u =
            collapse
              (sum (Ops.where Ops.O.(p land (r < Ops.int 3)) two zero) [ r ])
          in
          equal (list int) [] (sizes u);
          equal Dtypes.const (`Float 0.)
            (Interpreter.eval ~params:[ (0, `Bool false) ] u);
          equal Dtypes.const (`Float 6.)
            (Interpreter.eval ~params:[ (0, `Bool true) ] u));
    ]

let reduce_simplifying =
  let r = reduce_range 10 0 and o = range 6 1 in
  group "pm_reduce_simplify"
    [
      test "a sum whose value is a function of its range has a closed form"
        (fun () ->
          let u =
            reduce_simplify (sum (Ops.where Ops.O.(r < o) two zero) [ r ])
          in
          equal int 0 (count Reduce u);
          equal Dtypes.const (`Float 8.)
            (Interpreter.eval ~vars:[ ("r1", i 4) ] u));
      test "a closed form reads the values outside the sum it replaces"
        (fun () ->
          let u =
            reduce_simplify (sum (Ops.where Ops.O.(r < o) two zero) [ r ])
          in
          is_true (Ops.Nodes.mem o (Ops.backward_slice_with_self u)));
      test "a sum left with a range after collapsing is left" (fun () ->
          let u =
            sum
              (Ops.where Ops.O.(r < Ops.int 3) (Ops.cast r Float32) zero)
              [ r ]
          in
          equal uop u (reduce_simplify u));
      test "a sum over two ranges collapses each in turn" (fun () ->
          let s = reduce_range 4 2 in
          sums_to (`Float 12.)
            (reduce_simplify
               (sum (Ops.where Ops.O.(r < Ops.int 3) one zero) [ r; s ])));
      test "a sum holding a store is left" (fun () ->
          let u = sum (Ops.after two [ store_at r one ]) [ r ] in
          equal uop u (reduce_simplify u));
      test "a sum holding a reduction is left while that reduction stays"
        (fun () ->
          let s = reduce_range 4 2 in
          let inner = sum (Ops.cast Ops.O.(s + r) Float32) [ s ] in
          let u = sum inner [ r ] in
          equal uop u (reduce_simplify u));
      test "a sum over a range of size 1 is refused, the range folding to 0"
        (fun () ->
          let one = reduce_range 1 2 in
          raises_match (Exn.invalid_arg ?substring:None) (fun () ->
              reduce_simplify
                (sum (Ops.where Ops.O.(one < Ops.int 0) two zero) [ one ])));
      test "a maximum is left" (fun () ->
          let u =
            Ops.reduce (Ops.where Ops.O.(r < Ops.int 3) two zero) Max [ r ]
          in
          equal uop u (reduce_simplify u));
      test "an unparented range is removed" (fun () ->
          equal uop
            Ops.O.(f32 3. * Ops.int 10)
            (reduce_simplify (sum (f32 3.) [ r ])));
    ]

let load_collapsing =
  let r = reduce_range 10 0 in
  let table = buf ~size:10 Float32 in
  let elements =
    [ (0, Array.init 10 (fun j -> `Float (float_of_int (j * j)))) ]
  in
  let selected index =
    sum (Ops.where Ops.O.(index <> r) zero (Ops.index table [ r ])) [ r ]
  in
  let label = var "label" (-3) 12 in
  let at v u = Interpreter.eval ~vars:[ ("label", i v) ] ~buffers:elements u in
  group "pm_load_collapse"
    [
      test "a sum of the value an index selects is the value at that index"
        (fun () ->
          let u = load_collapse (selected label) in
          equal (list int) [] (sizes u);
          equal Dtypes.const (`Float 49.) (at 7 u));
      test "an index outside the range selects zero" (fun () ->
          let u = load_collapse (selected label) in
          equal Dtypes.const (`Float 0.) (at (-1) u);
          equal Dtypes.const (`Float 0.) (at 10 u);
          equal Dtypes.const (`Float 0.) (at 12 u));
      test "a sum whose index is its range is left" (fun () ->
          let u = selected r in
          equal uop u (load_collapse u));
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
      test "a comparison of a shifted variable is left" (fun () ->
          let u = Ops.O.(var "v" 0 9 + Ops.int 5 < Ops.int 20) in
          equal uop u (load_collapse u));
    ]

(* Laws *)

(* The cases of a matcher: each source of its input golden, named by its table,
   with what tinygrad makes of it. *)
let recorded_cases matcher =
  let input = lazy (Ops.src (Golden.sink (matcher ^ "_input.golden")))
  and output = lazy (Ops.src (Golden.sink (matcher ^ "_output.golden"))) in
  fun cell ->
    let n = int_of_string (cell "src") in
    (List.nth (Lazy.force input) n, List.nth (Lazy.force output) n)

let parity matcher pass =
  let case = recorded_cases matcher in
  Golden.cases (matcher ^ ".golden") ~key:[ "case" ] (fun cell ->
      let input, output = case cell in
      equal uop output (pass input))

let keeps_values matcher pass =
  let case = recorded_cases matcher in
  group
    (matcher ^ " keeps the value")
    [
      Golden.cases (matcher ^ ".golden") ~key:[ "case" ] (fun cell ->
          let input, _ = case cell in
          keeps_value ~name:(cell "case") input (pass input));
    ]

(* Generated sums: a value over a range of size 0 or 2 to 8, selected by bounds
   of the range, as the rules that count a sum see them. A range of size 1 folds
   to 0 inside the collapse, which then refuses its reduction. *)
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

let sums =
  let open Gen in
  let bound =
    frequency
      [
        (3, map (fun c -> Constant c) (int_range (-2) 12));
        (1, constant Variable);
        (1, constant Outer);
      ]
  in
  let selection =
    one_of
      [
        map (fun b -> Below b) bound;
        map (fun b -> Above b) bound;
        map (fun (lo, hi) -> Between (lo, hi)) (pair bound bound);
        map (fun (k, b) -> Shifted (k, b)) (pair (int_range (-3) 3) bound);
        map (fun (k, b) -> Scaled (k, b)) (pair (int_range 1 3) bound);
        map (fun b -> Mask b) bound;
      ]
  in
  let summand =
    of_list [ Constant_value; Variable_value; Outer_value; Range_value ]
  in
  with_pp pp_sum
    (quad
       (one_of [ constant 0; int_range 2 8 ])
       selection summand
       (of_list [ Dtype.Float32; Int32 ]))

let sum_of (n, s, v, dt) =
  let r = reduce_range n 0 and o = range 6 1 in
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
  let zero = Ops.const (`Int Z.zero) in
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

let keeps_kernel_writes ?(prepared = Fun.id) matcher pass =
  let case = recorded_cases matcher in
  group
    (matcher ^ " keeps the writes")
    [
      Golden.cases (matcher ^ ".golden") ~key:[ "case" ] (fun cell ->
          let input = prepared (fst (case cell)) in
          keeps_writes ~name:(cell "case") input (pass input));
    ]

(* Generated kernels: a store inside one to three ranges, at an index that takes
   each range, or its remainder or quotient by a divisor of its size, with
   strides in some order, possibly guarded below a constant; the value stored
   tells the iterations apart. *)
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
    | Some (j, c) -> Ops.valid index Ops.O.(List.nth rs j < Ops.int c)
  in
  initial_symbolic
    (kernel
       [
         Ops.end_
           (Ops.store (Ops.index (buf Int32) [ index ]) (Ops.cast value Int32))
           rs;
       ])

let laws =
  group "laws"
    [
      keeps_kernel_writes "flatten_range" flatten;
      keeps_kernel_writes "split_ranges" split;
      keeps_kernel_writes "simplify_ranges" ~prepared:initial_symbolic simplify;
      prop "pm_split_ranges keeps what a kernel writes" generated_kernels
        (fun g ->
          let u = kernel_of g in
          let split = split u in
          cover "a range splits"
            (List.length (ranges split) > List.length (ranges u));
          keeps_writes ~name:"kernel" u split);
      prop "pm_simplify_ranges keeps what a kernel writes" generated_kernels
        (fun g ->
          let u = kernel_of g in
          let simplified = simplify u in
          cover "ranges merge"
            (List.length (ranges simplified) < List.length (ranges u));
          cover "a range shrinks"
            (List.length (ranges simplified) = List.length (ranges u)
            && List.fold_left ( + ) 0 (sizes simplified)
               < List.fold_left ( + ) 0 (sizes u));
          keeps_writes ~name:"kernel" u simplified);
      keeps_values "reduce_unparented" unparent;
      keeps_values "reduce_collapse" collapse;
      keeps_values "reduce_simplify" reduce_simplify;
      keeps_values "load_collapse" load_collapse;
      prop "pm_reduce_collapse keeps the value of a sum" sums (fun s ->
          let u = sum_of s in
          keeps_value ~name:"sum" u (collapse u));
      prop "pm_reduce_simplify keeps the value of a sum" sums (fun s ->
          let u = sum_of s in
          keeps_value ~name:"sum" u (reduce_simplify u));
      prop "pm_reduce_simplify computes a sum of a function of its range" sums
        (fun ((_, _, v, _) as drawn) ->
          assume (v <> Range_value);
          let u = reduce_simplify (sum_of drawn) in
          equal int 0 (count Reduce u));
    ]

(* Goldens *)

let recorded name result pass =
  Golden.graph
    (name ^ "_" ^ result ^ ".golden")
    (fun () -> pass (Golden.sink (name ^ ".golden")))

let recorded_writes name pass =
  test (name ^ " keeps its writes") (fun () ->
      let u = Golden.sink (name ^ ".golden") in
      keeps_writes ~count:2 ~name u (pass u))

let kernels =
  group "tinygrad's kernels"
    [
      recorded "gather" "collapsed" load_collapse;
      recorded "embedding" "collapsed" load_collapse;
      recorded "transpose" "split" split;
      recorded "repeat" "split" split;
      recorded "elementwise" "simplified" simplify;
      recorded "embedding_ranges" "simplified" simplify;
      recorded "matmul" "simplified" simplify;
      recorded "conv" "simplified" simplify;
      recorded "arange" "collapsed" reduce_simplify;
      recorded "arange_sum" "collapsed" reduce_simplify;
      recorded "arange_index" "collapsed" reduce_simplify;
      recorded "triu" "collapsed" reduce_simplify;
      recorded "one_hot" "collapsed" reduce_simplify;
      recorded "sum_zero" "collapsed" reduce_simplify;
      recorded "arange_transposed" "collapsed" reduce_simplify;
      recorded "cumsum" "collapsed" reduce_simplify;
      recorded_writes "transpose" split;
      recorded_writes "repeat" split;
      recorded_writes "elementwise" simplify;
      recorded_writes "embedding_ranges" simplify;
      recorded_writes "matmul" simplify;
      recorded_writes "conv" simplify;
    ]

(* test_simplify_valid_idx.py::TestRangeShrink, recorded where each kernel
   reaches the range simplification. *)
let range_shrink =
  group "TestRangeShrink"
    (List.map
       (fun name -> recorded name "simplified" simplify)
       [
         "shrink_single_guard";
         "shrink_picks_max_guard";
         "shrink_guard_ge_max";
         "shrink_unguarded_elsewhere";
         "shrink_used_in_reduce";
         "shrink_to_single_iteration";
         "shrink_store_where_invalid";
         "shrink_store_where_invalid_flipped";
       ])

let cases =
  group "tinygrad's rewrites"
    [
      parity "flatten_range" flatten;
      parity "split_ranges" split;
      parity "simplify_ranges" simplify;
      parity "reduce_unparented" unparent;
      parity "reduce_collapse" collapse;
      parity "reduce_simplify" reduce_simplify;
      parity "load_collapse" load_collapse;
    ]

let () =
  exit
    (run "Tolk_next.Simplify"
       [
         flattening;
         merging;
         shrinking;
         splitting;
         unparented;
         collapsing;
         reduce_simplifying;
         load_collapsing;
         laws;
         kernels;
         range_shrink;
         cases;
       ])
