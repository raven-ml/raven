(* Tests of Tolk.Uop_weak: each pass commits weak data types as tinygrad's does,
   leaves no weak width behind, and keeps the values it commits. *)

open Windtrap
open Tolk

let i n = `Int (Bigint.of_int n)
let int = Ops.int
let float = Ops.float
let pow2 n = Int.shift_left 1 n
let i32 n = Ops.int ~dtype:Int32 n

let var ?(dtype = Dtype.Weak_int) name lo hi =
  Ops.variable ~dtype name (i lo) (i hi)

let small ?(name = "x") dtype = var ~dtype name 0 10
let weak = var "w" 0 4
let flag = Ops.variable ~dtype:Bool "c" (`Bool false) (`Bool true)

let fvar ?(dtype = Dtype.Float32) name hi =
  Ops.variable ~dtype name (`Float 0.) (`Float hi)

let rewrite pm u =
  Ops.graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:() u (After_sources pm)

(* A golden holds a graph and its rewrite by one pass, with the setting
   DEFAULT_FLOAT at [default_float] (default float). *)
let rewrites ?(default_float = "float") pm name u =
  Golden.graph (name ^ ".golden") (fun () ->
      Setting.context
        [ B (Setting.default_float, default_float) ]
        (fun () -> Ops.sink [ u; rewrite pm u ]))

let loaded_float =
  Ops.load (Ops.index (Ops.param ~shape:[ Int 1 ] 0 Float32) [ i32 0 ]) []

(* commit_weak_consts *)

let commit_weak_consts =
  let product = Ops.O.(fvar ~dtype:Float16 "h" 1. * float 2.0) in
  let committed u dt = Ops.sink [ u; Uop_weak.commit_weak_consts u dt ] in
  group "commit_weak_consts"
    [
      Golden.graph "commit_consts_at_a_stated_width.golden" (fun () ->
          committed product (Some Float16));
      Golden.graph "commit_consts_leaves_weak_expressions.golden" (fun () ->
          committed
            (Ops.v Op.Add ~src:[ Ops.O.(weak + int 1); int 3 ])
            (Some Int32));
      test "commits nothing without a width" (fun () ->
          equal Uops.uop product (Uop_weak.commit_weak_consts product None));
    ]

(* pm_commit_weak *)

let commit ?default_float = rewrites ?default_float Uop_weak.pm_commit_weak

let pm_commit_weak =
  group "pm_commit_weak"
    [
      test
        "a 64-bit unsigned cast of a weak expression keeps its value past a \
         float's precision" (fun () ->
          let b = var "b" 0 (pow2 40) in
          let u = Ops.cast Ops.O.((b * int 3) + int 1) Uint64 in
          equal Dtypes.const
            (i ((3 * (pow2 30 + 1)) + 1))
            (Interpreter.eval
               ~vars:[ ("b", i (pow2 30 + 1)) ]
               (rewrite Uop_weak.pm_lower_weak
                  (rewrite Uop_weak.pm_commit_weak u))));
      commit "peer_keeps_a_derivable_literal_bare" Ops.O.(small Int8 + int 3);
      commit "peer_rounds_a_derivable_literal"
        Ops.O.(loaded_float * float (-0.9999999893980771));
      commit "peer_commits_a_weak_expression"
        Ops.O.(small Int8 + (weak + int 1));
      commit "weak_sources_stay_weak_without_a_committed_peer"
        (Ops.v Op.Add ~src:[ int 1; float 1.0 ]);
      commit "where_keeps_a_weak_arm_bare"
        (Ops.v Op.Where
           ~src:[ Ops.bool true; Ops.cast (float 2.0) Float16; float 1.0 ]);
      commit "shift_commits_its_weak_operand"
        Ops.O.(int 0xFFFF lsl var ~dtype:Uint32 "s" 0 16);
      commit ~default_float:"half" "store_commits_its_value_at_the_destination"
        (let dst =
           Ops.index (Ops.param ~shape:[ Int 1 ] 0 Bfloat16) [ i32 0 ]
         in
         Ops.store ~gate:(Ops.bool true) dst (float 5.0));
      commit "cast_widens_a_weak_expression"
        (Ops.cast Ops.O.(weak + int 1) Int64);
      commit "cast_never_narrows_below_the_bounds"
        (Ops.cast Ops.O.(int (pow2 32) + int 1) Int8);
      commit "cast_never_narrows_below_the_default_float"
        (Ops.cast Ops.O.(float 1.0 + float 2.0) Float16);
      commit "cast_to_another_kind_commits_nothing"
        (Ops.cast Ops.O.(weak + int 1) Float32);
      commit "cast_to_a_weak_type_commits_nothing"
        (Ops.cast Ops.O.(weak + int 1) Weak_float);
      commit "cast_commits_operands_at_their_own_bounds"
        (Ops.cast
           (Ops.v Op.Cdiv ~src:[ var "n" 0 (pow2 40); int (pow2 40) ])
           Int32);
      commit ~default_float:"half" "cast_anchors_a_mixed_expression_at_the_cast"
        (Ops.cast Ops.O.(i32 1 + float 1.0) Float32);
    ]

(* pm_lower_weak *)

let lower ?default_float name u =
  rewrites ?default_float Uop_weak.pm_lower_weak name (Ops.sink [ u ])

let pm_lower_weak =
  let range = Ops.range (Int 16) [ 0 ] in
  let gated_index size hi =
    let idx = Ops.variable ~dtype:Int64 "i" (i 0) (`Int hi) in
    Ops.index (Ops.param ~shape:[ Int size ] 0 Float32) [ Ops.valid idx flag ]
  in
  let max_uint64 = Bigint.(pred (shift_left one 64)) in
  let custom = Ops.v Op.Custom ~arg:(Code { code = "n"; dtype = Weak_int }) in
  group "pm_lower_weak"
    [
      lower "lower_an_int_to_int32" Ops.O.(int 1 + int 2);
      lower "lower_an_int_beyond_int32_to_int64" Ops.O.(int (pow2 32) + int 1);
      lower "lower_an_int_beyond_int64_to_uint64" (Ops.const (`Int max_uint64));
      test "an int no 64-bit integer holds cannot be lowered" (fun () ->
          raises_match (Exn.invalid_arg ?substring:None) (fun () ->
              rewrite Uop_weak.pm_lower_weak
                (Ops.sink [ Ops.const (`Int (Bigint.succ max_uint64)) ])));
      lower ~default_float:"half" "lower_a_float_to_the_default_float"
        Ops.O.(float 1.5 * float 2.0);
      lower "lower_range_arithmetic" Ops.O.(range * int 4);
      lower "lower_a_comparison" Ops.O.(range < int 8);
      lower "lower_a_where" (Ops.where flag weak (int 3));
      lower "lower_a_stack" (Ops.stack [ weak; int 3 ]);
      lower "lower_a_special" Ops.O.(Ops.special (Int 8) "gidx0" * int 2);
      lower "lower_a_unary_float" (Ops.exp2 (float 2.0));
      lower "lower_a_weak_int_cast_of_a_bool_as_a_conversion"
        Ops.O.(Ops.cast flag Weak_int + int 1);
      lower "lower_a_weak_int_cast_of_a_float_as_a_conversion"
        Ops.O.(Ops.cast (fvar "f" 9.) Weak_int + int 1);
      lower "lower_a_weak_int_cast_of_an_int_as_a_restatement"
        Ops.O.(Ops.cast (small Int16) Weak_int + int 1);
      lower "lower_a_weak_float_cast_of_a_float_as_a_restatement"
        (Ops.exp2 (Ops.cast (fvar ~dtype:Float16 "f" 9.) Weak_float));
      lower "lower_stacked_weak_casts_as_two_conversions"
        (Ops.cast (Ops.cast (Ops.float ~dtype:Float32 1.5) Weak_int) Weak_float);
      lower "lower_stacked_weak_casts_of_a_weak_value_one_at_a_time"
        Ops.O.(Ops.cast (Ops.cast weak Weak_float) Weak_int + int 1);
      lower "lower_around_a_weak_node_it_does_not_lower" Ops.O.(custom + int 1);
      lower "lower_stacked_weak_casts_of_a_node_it_does_not_lower"
        (Ops.cast (Ops.cast custom Weak_float) Weak_int);
      lower "lower_keeps_weak_storage_outside_registers_of_scalars"
        (Ops.v Op.Param ~arg:(Param (Ops.param_arg ~size:4 ~slot:0 Weak_int)));
      lower "lower_a_weak_variable_to_its_default" Ops.O.(weak + int 1);
      lower "lower_a_weak_expression_under_a_committed_consumer"
        Ops.O.(small Int32 * (weak + int 1));
      lower "lower_a_gated_long_index_into_a_small_buffer_to_int32"
        (gated_index 8 (Bigint.of_int 7));
      lower "lower_a_gated_long_index_into_the_largest_int32_buffer_to_int32"
        (gated_index (pow2 31) (Bigint.of_int (pow2 31 - 1)));
      lower "lower_a_gated_long_index_into_a_buffer_past_int32_keeps_int64"
        (gated_index (pow2 31 + 1) (Bigint.of_int (pow2 31)));
      lower "lower_a_gated_long_index_into_a_huge_buffer_keeps_int64"
        (gated_index (pow2 33) (Bigint.shift_left Bigint.one 32));
      lower "lower_a_gated_shrink_to_the_width_its_bounds_need"
        (let buf = Ops.param ~shape:[ Int (pow2 31 + 64) ] 0 Float32 in
         let n = var "i" 0 (pow2 28) in
         let offset = Ops.valid Ops.O.(n * int 24) Ops.O.(n < int (pow2 28)) in
         Ops.v Op.Shrink ~src:[ buf; offset; int 4 ]);
      lower "lower_a_register_buffer_size"
        (Ops.placeholder ~slot:0 ~addrspace:Reg [ 4 ] Float32);
    ]

(* pm_uncast_const *)

let uncast = rewrites Uop_weak.pm_uncast_const

let pm_uncast_const =
  let u8 = small ~name:"u" Uint8 in
  group "pm_uncast_const"
    [
      uncast "uncast_a_committed_literal" Ops.O.(small Int32 + i32 1);
      uncast "uncast_keeps_literals_with_no_committed_peer"
        Ops.O.(i32 1 + i32 2);
      uncast "uncast_keeps_a_shifted_literal"
        (Ops.v Op.Shl ~src:[ i32 1; var ~dtype:Uint32 "s" 0 31 ]);
      uncast "uncast_keeps_a_cast_of_an_expression"
        Ops.O.(small Int32 + Ops.cast (weak + int 1) Int32);
      uncast "uncast_keeps_a_shifted_literal_of_its_peer_s_type"
        (Ops.v Op.Shl ~src:[ i32 1; var ~dtype:Int32 "s" 0 31 ]);
      uncast "uncast_keeps_a_literal_that_widens_a_comparison"
        (Ops.v Op.Cmplt ~src:[ small Int16; i32 1 ]);
      uncast "uncast_drops_a_cast_that_fits"
        Ops.O.(u8 < Ops.cconst Uint8 (i 44));
    ]

(* pm_cast_const *)

let cast_consts = rewrites Uop_weak.pm_cast_const

let pm_cast_const =
  group "pm_cast_const"
    [
      cast_consts "cast_consts_state_each_edge_width"
        (Ops.sink
           [
             Ops.O.(small ~name:"i" Int32 + int 1);
             Ops.O.(fvar "f" 10. + int 1);
             Ops.bool true;
           ]);
      cast_consts "cast_consts_give_underivable_literals_their_default"
        (Ops.sink [ Ops.O.(int 1 + int (pow2 40)) ]);
      cast_consts "cast_consts_leave_invalid_bare"
        (Ops.sink [ Ops.valid (Ops.cast weak Int32) flag ]);
    ]

(* Laws *)

(* An integer expression over weak and committed variables, as data, so that a
   failing case prints and shrinks. Its values fit an int32 wherever its
   variables are. *)
type expr =
  | Var of int
  | Const of int
  | Add of expr * expr
  | Mul of expr * int
  | Div of expr * int
  | Mod of expr * int
  | Max of expr * expr
  | Cast of expr * Dtype.t
  | Where of expr * expr * expr

let variables =
  [|
    var "w0" 0 15;
    var "w1" (-8) 8;
    var ~dtype:Int32 "x" 0 100;
    var ~dtype:Int16 "y" (-5) 5;
  |]

let rec build = function
  | Var k -> variables.(k)
  | Const c -> int c
  | Add (e0, e1) -> Ops.O.(build e0 + build e1)
  | Mul (e, c) -> Ops.O.(build e * int c)
  | Div (e, c) -> Ops.O.(build e // int c)
  | Mod (e, c) -> Ops.O.(build e % int c)
  | Max (e0, e1) -> Ops.maximum (build e0) (build e1)
  | Cast (e, dt) -> Ops.cast (build e) dt
  | Where (c, e0, e1) -> Ops.where Ops.O.(build c < int 3) (build e0) (build e1)

let rec pp_expr ppf = function
  | Var k -> Format.pp_print_string ppf (Ops.expr variables.(k))
  | Const c -> Format.pp_print_int ppf c
  | Add (e0, e1) -> Format.fprintf ppf "(%a + %a)" pp_expr e0 pp_expr e1
  | Mul (e, c) -> Format.fprintf ppf "(%a * %d)" pp_expr e c
  | Div (e, c) -> Format.fprintf ppf "(%a // %d)" pp_expr e c
  | Mod (e, c) -> Format.fprintf ppf "(%a %% %d)" pp_expr e c
  | Max (e0, e1) -> Format.fprintf ppf "max(%a, %a)" pp_expr e0 pp_expr e1
  | Cast (e, dt) ->
      Format.fprintf ppf "%a.cast(%a)" pp_expr e (Testable.pp Dtypes.dtype) dt
  | Where (c, e0, e1) ->
      Format.fprintf ppf "(%a < 3).where(%a, %a)" pp_expr c pp_expr e0 pp_expr
        e1

let gen_expr =
  let open Gen in
  let small_nonzero =
    map (fun k -> if k >= 0 then k + 1 else k) (int_range (-4) 4)
  in
  let leaf =
    frequency
      [
        (4, map (fun k -> Var k) (int_range 0 3));
        (1, map (fun c -> Const c) (int_range (-9) 9));
      ]
  in
  let dtype =
    of_list ~pp:(Testable.pp Dtypes.dtype)
      Dtype.[ Weak_int; Int32; Int64; Int16 ]
  in
  let rec expr depth =
    if depth = 0 then leaf
    else
      let sub = expr (depth - 1) in
      frequency
        [
          (2, leaf);
          (3, map (fun (e0, e1) -> Add (e0, e1)) (pair sub sub));
          (2, map (fun (e, c) -> Mul (e, c)) (pair sub small_nonzero));
          (1, map (fun (e, c) -> Div (e, c)) (pair sub small_nonzero));
          (1, map (fun (e, c) -> Mod (e, c)) (pair sub small_nonzero));
          (1, map (fun (e0, e1) -> Max (e0, e1)) (pair sub sub));
          (2, map (fun (e, dt) -> Cast (e, dt)) (pair sub dtype));
          (1, map (fun (c, e0, e1) -> Where (c, e0, e1)) (triple sub sub sub));
        ]
  in
  with_pp
    (Format.pp_print_list ~pp_sep:Format.pp_print_space pp_expr)
    (list ~size:(int_range 1 3) (expr 3))

let gen_point =
  Gen.map
    (fun (w0, w1, x, y) ->
      [ ("w0", i w0); ("w1", i w1); ("x", i x); ("y", i y) ])
    (Gen.quad (Gen.int_range 0 15) (Gen.int_range (-8) 8) (Gen.int_range 0 100)
       (Gen.int_range (-5) 5))

let weak_non_constant u =
  List.mem (Ops.dtype u) Dtype.weaks && Ops.op u <> Op.Const

let values_of env sink = List.map (Interpreter.eval ~vars:env) (Ops.src sink)
let values = list Dtypes.const

(* The consumers of a bare weak constant other than [`Invalid], but for the
   casts that state its width. *)
let bare_consumers sink =
  let bare s =
    Ops.op s = Op.Const
    && List.mem (Ops.dtype s) Dtype.weaks
    && not (Ops.is_invalid s)
  in
  List.filter
    (fun u -> Ops.op u <> Op.Cast && List.exists bare (Ops.src u))
    (Ops.toposort ~calls:Enter sink)

let integer_type =
  Gen.of_list ~pp:(Testable.pp Dtypes.dtype)
    Dtype.[ Int8; Int16; Int32; Int64; Uint8; Uint16; Uint32; Uint64 ]

let laws =
  let sink es = Ops.sink (List.map build es) in
  group "laws"
    [
      prop "pm_lower_weak leaves no weak width but a literal's" gen_expr
        (fun es ->
          let lowered = rewrite Uop_weak.pm_lower_weak (sink es) in
          equal (list Uops.uop) []
            (List.filter weak_non_constant (Ops.toposort ~calls:Enter lowered)));
      prop "pm_lower_weak keeps the values of what it lowers"
        (Gen.pair gen_expr gen_point) (fun (es, env) ->
          let u = sink es in
          equal values (values_of env u)
            (values_of env (rewrite Uop_weak.pm_lower_weak u)));
      prop "pm_commit_weak keeps the values of what it commits"
        (Gen.pair gen_expr gen_point) (fun (es, env) ->
          let u = sink es in
          equal values (values_of env u)
            (values_of env (rewrite Uop_weak.pm_commit_weak u)));
      prop "pm_commit_weak computes an integer cast in integers"
        (Gen.pair gen_expr integer_type) (fun (es, dt) ->
          let committed =
            rewrite Uop_weak.pm_commit_weak
              (Ops.sink (List.map (fun e -> Ops.cast (build e) dt) es))
          in
          equal (list Uops.uop) []
            (List.filter
               (fun u -> Dtype.is_float (Ops.dtype u))
               (Ops.toposort ~calls:Enter committed)));
      prop "pm_cast_const states the width of every constant" gen_expr
        (fun es ->
          equal (list Uops.uop) []
            (bare_consumers (rewrite Uop_weak.pm_cast_const (sink es))));
      prop "pm_cast_const is idempotent" gen_expr (fun es ->
          Law.idempotent Uops.uop (rewrite Uop_weak.pm_cast_const) (sink es));
      prop "pm_cast_const keeps the values of what it casts"
        (Gen.pair gen_expr gen_point) (fun (es, env) ->
          let u = sink es in
          equal values (values_of env u)
            (values_of env (rewrite Uop_weak.pm_cast_const u)));
    ]

let () =
  exit
    (run "Tolk.Uop_weak"
       [
         commit_weak_consts;
         pm_commit_weak;
         pm_lower_weak;
         pm_uncast_const;
         pm_cast_const;
         laws;
       ])
