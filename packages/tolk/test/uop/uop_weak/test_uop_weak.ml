(* Tests of Tolk.Uop_weak: each pass gives weak data types the widths its
   interface states, leaves no weak width behind, and keeps the values it
   commits. *)

open Windtrap
open Tolk

let i n = `Int (Bigint.of_int n)
let z n = `Int n
let int = Ops.int
let float = Ops.float
let pow2 n = Int.shift_left 1 n
let big n = Bigint.shift_left Bigint.one n
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

let rejects f u = raises_match (Exn.invalid_arg ?sub:None) (fun () -> f u)

(* A golden holds a graph and its rewrite by one pass, with the setting
   DEFAULT_FLOAT at [default_float] (default float). *)
let rewrites ?(default_float = "float") pm name u =
  Golden.graph (name ^ ".golden") (fun () ->
      Setting.context
        [ B (Setting.default_float, default_float) ]
        (fun () -> Ops.sink [ u; rewrite pm u ]))

let loaded_float =
  Ops.load (Ops.index (Shape.param ~shape:[ Int 1 ] 0 Float32) [ i32 0 ]) []

(* The first node of [op] under [u]. *)
let first op u =
  List.find (fun n -> Ops.op n = op) (Ops.toposort ~calls:Enter u)

(* Expressions *)

(* An expression as data, so that a failing case prints and shrinks: integer
   expressions over integer variables, and float expressions over float
   variables and literals, NaN and -0. among them. *)
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
  | Fvar of int
  | Fconst of float
  | Fadd of expr * expr
  | Exp2 of expr
  | Convert of expr * Dtype.t
  | Big of Bigint.t

(* Every value of these fits a 64-bit integer, wherever the variables are: [b]
   reaches past a float's precision, [n] past an int32. *)
let narrow =
  [|
    var "w0" 0 15;
    var "w1" (-8) 8;
    var ~dtype:Int32 "x" 0 100;
    var ~dtype:Int16 "y" (-5) 5;
    var "b" 0 (pow2 40);
    var "n" (pow2 31 - 3) (pow2 31 + 2);
  |]

(* [narrow], and weak variables at the ends of 64 bits, whose sums leave them:
   for laws on widths alone. *)
let wide =
  Array.append narrow
    [|
      Ops.variable "u"
        (z Bigint.(pred (pred (big 63))))
        (z (Bigint.succ (big 63)));
      Ops.variable "v"
        (z (Bigint.neg (big 63)))
        (z Bigint.(add (neg (big 63)) (of_int 3)));
    |]

let floats = [| fvar "f" 10.; fvar ~dtype:Float16 "h" 4. |]

let rec build vars = function
  | Var k -> vars.(k)
  | Const c -> int c
  | Add (e0, e1) -> Ops.O.(build vars e0 + build vars e1)
  | Mul (e, c) -> Ops.O.(build vars e * int c)
  | Div (e, c) -> Ops.O.(build vars e // int c)
  | Mod (e, c) -> Ops.O.(build vars e % int c)
  | Max (e0, e1) -> Ops.maximum (build vars e0) (build vars e1)
  | Cast (e, dt) | Convert (e, dt) -> Ops.cast (build vars e) dt
  | Where (c, e0, e1) ->
      Ops.where Ops.O.(build vars c < int 3) (build vars e0) (build vars e1)
  | Fvar k -> floats.(k)
  | Fconst x -> float x
  | Big n -> Ops.const (`Int n)
  | Fadd (e0, e1) -> Ops.O.(build vars e0 + build vars e1)
  | Exp2 e -> Ops.exp2 (build vars e)

let pp_dtype = Testable.pp Dtypes.dtype

let rec pp_expr vars ppf = function
  | Var k -> Format.pp_print_string ppf (Ops.expr vars.(k))
  | Const c -> Format.pp_print_int ppf c
  | Add (e0, e1) | Fadd (e0, e1) ->
      Format.fprintf ppf "(%a + %a)" (pp_expr vars) e0 (pp_expr vars) e1
  | Mul (e, c) -> Format.fprintf ppf "(%a * %d)" (pp_expr vars) e c
  | Div (e, c) -> Format.fprintf ppf "(%a // %d)" (pp_expr vars) e c
  | Mod (e, c) -> Format.fprintf ppf "(%a %% %d)" (pp_expr vars) e c
  | Max (e0, e1) ->
      Format.fprintf ppf "max(%a, %a)" (pp_expr vars) e0 (pp_expr vars) e1
  | Cast (e, dt) | Convert (e, dt) ->
      Format.fprintf ppf "%a.cast(%a)" (pp_expr vars) e pp_dtype dt
  | Where (c, e0, e1) ->
      Format.fprintf ppf "(%a < 3).where(%a, %a)" (pp_expr vars) c
        (pp_expr vars) e0 (pp_expr vars) e1
  | Fvar k -> Format.pp_print_string ppf (Ops.expr floats.(k))
  | Fconst x -> Format.fprintf ppf "%h" x
  | Big n -> Bigint.pp_print ppf n
  | Exp2 e -> Format.fprintf ppf "exp2(%a)" (pp_expr vars) e

let small_nonzero =
  Gen.map (fun k -> if k >= 0 then k + 1 else k) (Gen.int_range (-4) 4)

let int_dtype = Gen.of_list ~pp:pp_dtype Dtype.[ Weak_int; Int32; Int64; Int16 ]

(* A variable of [vars], or a small constant. *)
let leaf_of vars =
  Gen.frequency
    [
      (4, Gen.map (fun k -> Var k) (Gen.int_range 0 (Array.length vars - 1)));
      (1, Gen.map (fun c -> Const c) (Gen.int_range (-9) 9));
    ]

(* [integers leaf depth] draws an integer expression over [leaf]. *)
let rec integers leaf depth =
  let open Gen in
  if depth = 0 then leaf
  else
    let sub = integers leaf (depth - 1) in
    frequency
      [
        (2, leaf);
        (3, map (fun (e0, e1) -> Add (e0, e1)) (pair sub sub));
        (2, map (fun (e, c) -> Mul (e, c)) (pair sub small_nonzero));
        (1, map (fun (e, c) -> Div (e, c)) (pair sub small_nonzero));
        (1, map (fun (e, c) -> Mod (e, c)) (pair sub small_nonzero));
        (1, map (fun (e0, e1) -> Max (e0, e1)) (pair sub sub));
        (2, map (fun (e, dt) -> Cast (e, dt)) (pair sub int_dtype));
        (1, map (fun (c, e0, e1) -> Where (c, e0, e1)) (triple sub sub sub));
      ]

(* [reals vars depth] draws a float expression, its integer parts over
   [vars]. *)
let rec reals vars depth =
  let open Gen in
  let leaf =
    frequency
      [
        (2, map (fun k -> Fvar k) (int_range 0 1));
        ( 2,
          map
            (fun x -> Fconst x)
            (of_list [ 1.5; 0.1; -0.; Float.nan; infinity; 65504. ]) );
      ]
  in
  if depth = 0 then leaf
  else
    let sub = reals vars (depth - 1) in
    let float_dtype =
      of_list ~pp:pp_dtype Dtype.[ Weak_float; Float32; Float16 ]
    in
    frequency
      [
        (2, leaf);
        (2, map (fun (e0, e1) -> Fadd (e0, e1)) (pair sub sub));
        (1, map (fun e -> Exp2 e) sub);
        ( 2,
          map
            (fun (e, dt) -> Convert (e, dt))
            (pair (integers (leaf_of vars) (depth - 1)) float_dtype) );
      ]

let roots vars gen =
  Gen.with_pp
    (Format.pp_print_list ~pp_sep:Format.pp_print_space (pp_expr vars))
    (Gen.list ~size:(Gen.int_range 1 3) gen)

(* Integer expressions over [narrow], whose values fit 64 bits. *)
let integer_graph = roots narrow (integers (leaf_of narrow) 3)

(* Integer and float expressions over [wide], and integer conversions of
   floats. *)
let mixed_graph =
  let int_of_float =
    Gen.map
      (fun (e, dt) -> Convert (e, dt))
      (Gen.pair (reals wide 2)
         (Gen.of_list ~pp:pp_dtype Dtype.[ Weak_int; Int32 ]))
  in
  roots wide
    (Gen.frequency
       [ (3, integers (leaf_of wide) 3); (2, reals wide 3); (1, int_of_float) ])

let default_float = Gen.of_list ~pp:Format.pp_print_string [ "float"; "half" ]

(* A value of each of [narrow]'s variables: its bounds and their inner
   neighbours, or any value between. *)
let point =
  let value u =
    let lo = Bigint.to_int (Dtype.Value.to_z (Ops.vmin u))
    and hi = Bigint.to_int (Dtype.Value.to_z (Ops.vmax u)) in
    Gen.frequency
      [ (1, Gen.of_list [ lo; lo + 1; hi - 1; hi ]); (2, Gen.int_range lo hi) ]
  in
  Array.fold_right
    (fun u rest ->
      Gen.map
        (fun (v, rest) -> (Ops.expr u, i v) :: rest)
        (Gen.pair (value u) rest))
    narrow (Gen.constant [])

(* [integer_graph] with a point, printed. *)
let at_a_point =
  Gen.with_pp
    (fun ppf (es, env) ->
      Format.fprintf ppf "%a@ at %a"
        (Format.pp_print_list ~pp_sep:Format.pp_print_space (pp_expr narrow))
        es
        (Format.pp_print_list ~pp_sep:Format.pp_print_space (fun ppf (n, v) ->
             Format.fprintf ppf "%s = %a" n (Testable.pp Dtypes.value) v))
        env)
    (Gen.pair integer_graph point)

(* [mixed_graph] under a default float, printed. *)
let under_a_default =
  Gen.with_pp
    (fun ppf (es, df) ->
      Format.fprintf ppf "%a@ with DEFAULT_FLOAT=%s"
        (Format.pp_print_list ~pp_sep:Format.pp_print_space (pp_expr wide))
        es df)
    (Gen.pair mixed_graph default_float)

(* u + v, from u near 2^63 and v near -2^63, is within [-2, 4]: operands that
   only a 64-bit unsigned and a signed integer hold, meeting near 0. *)
let opposite =
  Gen.frequency
    [
      (3, Gen.of_list [ Add (Var 6, Var 7); Add (Var 7, Var 6) ]);
      (1, Gen.map (fun c -> Const c) (Gen.int_range (-9) 9));
    ]

(* A value of u and of v, each at its bounds or between. *)
let opposite_point =
  let near base =
    Gen.map (fun k -> `Int Bigint.(add base (of_int k))) (Gen.int_range 0 3)
  in
  Gen.map
    (fun (u, v) -> [ ("u", u); ("v", v) ])
    (Gen.pair (near Bigint.(pred (pred (big 63)))) (near (Bigint.neg (big 63))))

let pp_env ppf env =
  Format.pp_print_list ~pp_sep:Format.pp_print_space
    (fun ppf (n, v) ->
      Format.fprintf ppf "%s = %a" n (Testable.pp Dtypes.value) v)
    ppf env

(* Expressions over u + v, at a value of u and v. *)
let opposite_signs =
  Gen.with_pp
    (fun ppf (es, env) ->
      Format.fprintf ppf "%a@ at %a"
        (Format.pp_print_list ~pp_sep:Format.pp_print_space (pp_expr wide))
        es pp_env env)
    (Gen.pair (roots wide (integers opposite 2)) opposite_point)

(* (u + v < 3).where (1, 0) and (u + v) // 2 as named examples: at u + v = -2,
   which an unsigned sum reads as 2^64 - 2, and at u + v = 4, which a float32
   sum reads as 0. *)
let at_minus_two =
  [
    ("u", `Int Bigint.(pred (pred (big 63)))); ("v", `Int (Bigint.neg (big 63)));
  ]

let at_four =
  [
    ("u", `Int (Bigint.succ (big 63)));
    ("v", `Int Bigint.(add (neg (big 63)) (of_int 3)));
  ]

let below_three =
  ([ Where (Add (Var 6, Var 7), Const 1, Const 0) ], at_minus_two)

let below_three_at_four = (fst below_three, at_four)

(* Nodes over u alone, at each of u's bounds: u // 2 reads u's signedness beside
   its own signed width; u + w1, w1 signed, computes at u's unsigned width; (u -
   2^63 + 3 < 3) is u < 2^63. *)
let at_u_bounds =
  let u_at lo =
    [
      ("u", `Int Bigint.(if lo then pred (pred (big 63)) else succ (big 63)));
      ("w1", `Int (Bigint.of_int (if lo then 8 else -8)));
    ]
  in
  List.concat_map
    (fun e -> [ ([ e ], u_at true); ([ e ], u_at false) ])
    [
      Div (Var 6, 2);
      Where (Add (Var 6, Big Bigint.(sub (of_int 3) (big 63))), Const 1, Const 0);
      Mul (Var 6, 1);
      Add (Var 6, Var 1);
    ]

let halved = ([ Div (Add (Var 6, Var 7), 2) ], at_minus_two)
let sink vars es = Ops.sink (List.map (build vars) es)
let values_of env sink = List.map (Interpreter.eval ~vars:env) (Ops.src sink)
let values = list Dtypes.const
let nodes u = Ops.toposort ~calls:Enter u
let has_op op u = List.exists (fun n -> Ops.op n = op) (nodes u)
let is_weak u = List.mem (Ops.dtype u) Dtype.weaks
let weak_non_constant u = is_weak u && Ops.op u <> Op.Const

(* The consumers of a bare weak constant other than [`Invalid], but for the
   casts that state its width. *)
let bare_consumers sink =
  let bare s = Ops.op s = Op.Const && is_weak s && not (Ops.is_invalid s) in
  List.filter
    (fun u -> Ops.op u <> Op.Cast && List.exists bare (Ops.src u))
    (nodes sink)

let with_default_float df f =
  Setting.context [ B (Setting.default_float, df) ] f

(* What the laws' graphs reach. *)
let has_weak_int_past_int32 u =
  List.exists
    (fun n ->
      Dtype.equal (Ops.dtype n) Weak_int
      && Ops.op n <> Op.Const
      && not (Dtype.equal (Ops.commit_dtype ~default_int:Int32 n) Int32))
    (nodes u)

let has_weak_float u =
  List.exists
    (fun n -> Dtype.equal (Ops.dtype n) Weak_float && Ops.op n <> Op.Const)
    (nodes u)

let at_a_bound env =
  Array.exists
    (fun u ->
      let v = Dtype.Value.to_z (List.assoc (Ops.expr u) env) in
      List.exists
        (fun b -> Bigint.equal v (Dtype.Value.to_z b))
        [ Ops.vmin u; Ops.vmax u ])
    narrow

(* commit_weak_consts *)

let commit_weak_consts =
  let product = Ops.O.(fvar ~dtype:Float16 "h" 1. * float 2.0) in
  let committed u dt = Ops.sink [ u; Uop_weak.commit_weak_consts u dt ] in
  group "commit_weak_consts"
    [
      group "converts each graph's weak constants as its golden states"
        [
          Golden.graph "commit_consts_at_a_stated_width.golden" (fun () ->
              committed product (Some Float16));
          Golden.graph "commit_consts_leaves_weak_expressions.golden" (fun () ->
              committed
                (Ops.v Op.Add ~src:[ Ops.O.(weak + int 1); int 3 ])
                (Some Int32));
        ];
      test "commits nothing without a width" (fun () ->
          equal Uops.uop product (Uop_weak.commit_weak_consts product None));
    ]

(* pm_commit_weak *)

let commit ?default_float = rewrites ?default_float Uop_weak.pm_commit_weak

let commit_values =
  prop "keeps the value of each integer node it rewrites" at_a_point
    (fun (es, env) ->
      let u = sink narrow es in
      cover "a weak operand beside a committed peer"
        (List.exists
           (fun n ->
             List.exists weak_non_constant (Ops.src n)
             && List.exists
                  (fun s -> (not (is_weak s)) && Dtype.is_int (Ops.dtype s))
                  (Ops.src n))
           (nodes u));
      cover "a committed cast over a weak expression"
        (List.exists
           (fun n ->
             Ops.op n = Op.Cast
             && (not (is_weak n))
             && weak_non_constant (Ops.nth n 0))
           (nodes u));
      equal values (values_of env u)
        (values_of env (rewrite Uop_weak.pm_commit_weak u)))

let integer_type =
  Gen.of_list ~pp:pp_dtype
    Dtype.[ Int8; Int16; Int32; Int64; Uint8; Uint16; Uint32; Uint64 ]

(* DIVERGENCES D44: (b * 3 + 1).cast(Uint64), b weak below 2^40, commits at a
   signed width and an unsigned one, which meet only at a float. *)
let d44 = ([ Add (Mul (Var 4, 3), Const 1) ], Dtype.Uint64)

let integer_casts =
  prop "computes an integer cast in integers" ~count:300 ~examples:[ d44 ]
    (Gen.pair (roots wide (integers (leaf_of wide) 3)) integer_type)
    (fun (es, dt) ->
      let casts = List.map (fun e -> Ops.cast (build wide e) dt) es in
      cover "a 64-bit unsigned cast over a weak expression committed signed"
        (Dtype.equal dt Uint64
        && List.exists
             (fun e ->
               let u = build wide e in
               weak_non_constant u
               && not
                    (Dtype.is_unsigned (Ops.commit_dtype ~default_int:Int32 u)))
             es);
      let committed = rewrite Uop_weak.pm_commit_weak (Ops.sink casts) in
      equal (list Uops.uop) []
        (List.filter (fun u -> Dtype.is_float (Ops.dtype u)) (nodes committed)))

let commit_opposite =
  prop "keeps the value of a cast over a 64-bit unsigned and a signed operand"
    ~examples:[ (halved, Dtype.Int64); (below_three, Int64) ]
    (Gen.pair opposite_signs integer_type)
    (fun ((es, env), dt) ->
      let u = Ops.sink (List.map (fun e -> Ops.cast (build wide e) dt) es) in
      cover "a cast of a division of a sum near 0"
        (has_op Op.Truncdiv u || has_op Op.Div u);
      equal values (values_of env u)
        (values_of env (rewrite Uop_weak.pm_commit_weak u)))

(* [committed_shift ~lo ~hi k] is the count of the uint8 255 shifted by a weak
   count over [lo, hi], once committed, and the shift's value, both at [k]. *)
let committed_shift ~lo ~hi w =
  let x = var ~dtype:Uint8 "x" 0 255 and k = Ops.variable "k" (z lo) (z hi) in
  let committed = rewrite Uop_weak.pm_commit_weak (Ops.alu x Shl [ k ]) in
  let vars = [ ("x", i 255); ("k", w) ] in
  ( Interpreter.eval ~vars (Ops.nth (first Op.Shl committed) 1),
    Interpreter.eval ~vars committed )

let pm_commit_weak =
  group "pm_commit_weak"
    [
      group "rewrites each graph as its golden states"
        [
          commit "peer_keeps_a_derivable_literal_bare"
            Ops.O.(small Int8 + int 3);
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
          commit ~default_float:"half"
            "store_commits_its_value_at_the_destination"
            (let dst =
               Ops.index (Shape.param ~shape:[ Int 1 ] 0 Bfloat16) [ i32 0 ]
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
               (Ops.v Op.Truncdiv ~src:[ var "n" 0 (pow2 40); int (pow2 40) ])
               Int32);
          commit ~default_float:"half"
            "cast_anchors_a_mixed_expression_at_the_cast"
            (Ops.cast Ops.O.(i32 1 + float 1.0) Float32);
        ];
      (* A count of the width or more shifts every bit of a uint8 out: the count
         reads min (k, 8), and the shift keeps its value, 255 << k modulo
         2^8. *)
      cases "a weak count its shift's type does not hold saturates at the width"
        ~id:(fun (k, _) -> Printf.sprintf "k = %d" k)
        [
          (0, (0, 255));
          (7, (7, 128));
          (8, (8, 0));
          (255, (8, 0));
          (1000, (8, 0));
        ]
        (fun (k, (count, shifted)) ->
          equal
            (pair Dtypes.const Dtypes.const)
            (i count, i shifted)
            (committed_shift ~lo:Bigint.zero ~hi:(Bigint.of_int 1000) (i k)));
      test "a count past 64 bits, all at or past the width, is the width"
        (fun () ->
          equal Dtypes.const (i 8)
            (fst (committed_shift ~lo:(Bigint.of_int 8) ~hi:(big 70) (i 9))));
      test "a count past 64 bits that may be below the width is refused"
        (fun () -> rejects (committed_shift ~lo:Bigint.zero ~hi:(big 70)) (i 3));
      test "a cast of a weak shift by a huge count saturates the count"
        (fun () ->
          let huge = Ops.const (z (big 64)) in
          let u = Ops.cast Ops.O.(weak lsl huge) Int32 in
          equal Dtypes.const (i 0)
            (Interpreter.eval
               ~vars:[ ("w", i 3) ]
               (rewrite Uop_weak.pm_commit_weak u)));
      test "a weak exponent keeps its value past its base's type" (fun () ->
          let x = Shape.param 0 Dtype.Uint16 in
          let e = Ops.O.(weak + int 65535) in
          let u = rewrite Uop_weak.pm_commit_weak (Ops.alu x Op.Pow [ e ]) in
          equal Dtypes.const (i 0)
            (Interpreter.eval ~vars:[ ("w", i 1) ] ~params:[ (0, i 0) ] u));
      (* w^(2^40) modulo 2^32: 0 for an even w; 1 for 3, whose powers modulo
         2^32 repeat every 2^30. *)
      cases "a cast of a weak power computes it" ~id:(Printf.sprintf "w = %d")
        [ 0; 1; 2; 3 ] (fun w ->
          let e = Ops.const (z (big 40)) in
          let u = Ops.cast (Ops.alu weak Op.Pow [ e ]) Int32 in
          equal Dtypes.const
            (i (if w mod 2 = 0 then 0 else 1))
            (Interpreter.eval
               ~vars:[ ("w", i w) ]
               (rewrite Uop_weak.pm_commit_weak u)));
      test "a cast of a weak power leaves its exponent weak" (fun () ->
          let e = Ops.const (z (big 40)) in
          let u = Ops.cast (Ops.alu weak Op.Pow [ e ]) Int32 in
          let pow = first Op.Pow (rewrite Uop_weak.pm_commit_weak u) in
          equal Dtypes.dtype Weak_int (Ops.dtype (Ops.nth pow 1)));
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
      (* A host vector of doubles stores a stack of weak constants through one
         cast: committed narrower and converted, a signalling NaN would come out
         quiet. *)
      test "a cast of a stack of weak constants commits each at the cast's type"
        (fun () ->
          let snan = Int64.float_of_bits 0x7FF0_0000_2000_0000L in
          let u =
            Ops.cast (Shape.stack [ float snan; float (-.snan) ]) Float64
          in
          let lowered =
            rewrite Uop_weak.pm_lower_weak (rewrite Uop_weak.pm_commit_weak u)
          in
          let double x = Ops.const ~dtype:Float64 (`Float x) in
          equal Uops.uop
            (Ops.cast (Shape.stack [ double snan; double (-.snan) ]) Float64)
            lowered);
      commit_values;
      commit_opposite;
      integer_casts;
      test
        "a weak constant under a committed cast that no 64-bit integer holds \
         is refused" (fun () ->
          rejects
            (rewrite Uop_weak.pm_commit_weak)
            (Ops.cast Ops.O.(weak + Ops.const (z (big 70))) Int64));
    ]

(* pm_lower_weak *)

let lower ?default_float name u =
  rewrites ?default_float Uop_weak.pm_lower_weak name (Ops.sink [ u ])

let lower_widths =
  prop "leaves no weak width but a literal's" ~count:300 under_a_default
    (fun (es, df) ->
      let u = sink wide es in
      cover "a weak float lowered" (has_weak_float u);
      cover "a weak integer committed past Int32" (has_weak_int_past_int32 u);
      cover "at the default float half" (df = "half");
      let lowered =
        with_default_float df (fun () -> rewrite Uop_weak.pm_lower_weak u)
      in
      equal (list Uops.uop) [] (List.filter weak_non_constant (nodes lowered)))

let lower_values =
  prop "keeps the value of each integer node it rewrites" at_a_point
    (fun (es, env) ->
      let u = sink narrow es in
      cover "a variable at a bound" (at_a_bound env);
      cover "a weak integer committed past Int32" (has_weak_int_past_int32 u);
      equal values (values_of env u)
        (values_of env (rewrite Uop_weak.pm_lower_weak u)))

(* u * 2, u weak just below 2^63 + 2: its bounds pass every 64-bit integer, and
   u commits at Uint64. *)
let past_64_bits = [ Mul (Var 6, 2) ]

let lower_integers =
  prop "computes an integer node in integers" ~examples:[ past_64_bits ]
    (roots wide (integers (leaf_of wide) 3))
    (fun es ->
      let u = sink wide es in
      cover "a node whose bounds no 64-bit integer holds"
        (List.exists
           (fun n -> weak_non_constant n && Ops.fits_no_int n)
           (nodes u));
      equal (list Uops.uop) []
        (List.filter
           (fun n -> Dtype.is_float (Ops.dtype n))
           (nodes (rewrite Uop_weak.pm_lower_weak u))))

let lower_opposite =
  prop "keeps the value of a node over a 64-bit unsigned and a signed operand"
    ~examples:([ below_three; below_three_at_four; halved ] @ at_u_bounds)
    opposite_signs
    (fun (es, env) ->
      let u = sink wide es in
      cover "a selection by a sum near 0" (has_op Op.Where u);
      cover "a division of a sum near 0"
        (has_op Op.Truncdiv u || has_op Op.Div u);
      equal values (values_of env u)
        (values_of env (rewrite Uop_weak.pm_lower_weak u)))

(* The type a weak constant lowers to. *)
let lowered_type n =
  Ops.dtype
    (List.hd
       (Ops.src (rewrite Uop_weak.pm_lower_weak (Ops.sink [ Ops.const (z n) ]))))

let pm_lower_weak =
  let range = Ops.range ~axis_type:Weak (Int 16) [ 0 ] in
  let gated_index size hi =
    let idx = Ops.variable ~dtype:Int64 "i" (i 0) (`Int hi) in
    Ops.index
      (Shape.param ~shape:[ Int size ] 0 Float32)
      [ Shape.valid idx flag ]
  in
  let max_uint64 = Bigint.(pred (shift_left one 64)) in
  let custom =
    Ops.call ~inline:true
      (Ops.custom_function ~dtype:Weak_int ~fmt:true "n" [])
      []
  in
  group "pm_lower_weak"
    [
      group "rewrites each graph as its golden states"
        [
          lower "lower_an_int_to_int32" Ops.O.(int 1 + int 2);
          lower "lower_an_int_beyond_int32_to_int64"
            Ops.O.(int (pow2 32) + int 1);
          lower "lower_an_int_beyond_int64_to_uint64"
            (Ops.const (`Int max_uint64));
          lower ~default_float:"half" "lower_a_float_to_the_default_float"
            Ops.O.(float 1.5 * float 2.0);
          lower "lower_range_arithmetic" Ops.O.(range * int 4);
          lower "lower_a_comparison" Ops.O.(range < int 8);
          lower "lower_a_where" (Ops.where flag weak (int 3));
          lower "lower_a_stack" (Shape.stack [ weak; int 3 ]);
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
            (Ops.cast
               (Ops.cast (Ops.float ~dtype:Float32 1.5) Weak_int)
               Weak_float);
          lower "lower_stacked_weak_casts_of_a_weak_value_one_at_a_time"
            Ops.O.(Ops.cast (Ops.cast weak Weak_float) Weak_int + int 1);
          lower "lower_around_a_weak_node_it_does_not_lower"
            Ops.O.(custom + int 1);
          lower "lower_stacked_weak_casts_of_a_node_it_does_not_lower"
            (Ops.cast (Ops.cast custom Weak_float) Weak_int);
          lower "lower_keeps_weak_storage_outside_registers_of_scalars"
            (Ops.v Op.Param
               ~src:[ Ops.size_src (Some 4) ]
               ~arg:(Param (Ops.param_arg ~slot:0 Weak_int)));
          lower "lower_a_weak_variable_to_its_default" Ops.O.(weak + int 1);
          lower "lower_a_weak_expression_under_a_committed_consumer"
            Ops.O.(small Int32 * (weak + int 1));
          lower "lower_a_gated_long_index_into_a_small_buffer_to_int32"
            (gated_index 8 (Bigint.of_int 7));
          lower
            "lower_a_gated_long_index_into_the_largest_int32_buffer_to_int32"
            (gated_index (pow2 31) (Bigint.of_int (pow2 31 - 1)));
          lower "lower_a_gated_long_index_into_a_buffer_past_int32_keeps_int64"
            (gated_index (pow2 31 + 1) (Bigint.of_int (pow2 31)));
          lower "lower_a_gated_long_index_into_a_huge_buffer_keeps_int64"
            (gated_index (pow2 33) (Bigint.shift_left Bigint.one 32));
          lower "lower_a_gated_shrink_to_the_width_its_bounds_need"
            (let buf = Shape.param ~shape:[ Int (pow2 31 + 64) ] 0 Float32 in
             let n = var "i" 0 (pow2 28) in
             let offset =
               Shape.valid Ops.O.(n * int 24) Ops.O.(n < int (pow2 28))
             in
             Ops.v Op.Shrink ~src:[ buf; offset; int 4 ]);
          lower "lower_keeps_a_register_buffer_size_bare"
            (Shape.placeholder ~slot:0 ~addrspace:Reg [ 4 ] Float32);
        ];
      (* the first of Int32, Int64 and Uint64 that holds the value *)
      cases "a weak integer constant takes the first 64-bit type that holds it"
        ~id:(fun (n, _) -> Bigint.to_string n)
        Bigint.
          [
            (pred (big 31), Dtype.Int32);
            (neg (big 31), Int32);
            (big 31, Int64);
            (pred (neg (big 31)), Int64);
            (pred (big 63), Int64);
            (neg (big 63), Int64);
            (big 63, Uint64);
            (pred (big 64), Uint64);
          ]
        (fun (n, dt) -> equal Dtypes.dtype dt (lowered_type n));
      cases "a weak integer constant no 64-bit integer holds is refused"
        ~id:Bigint.to_string
        Bigint.[ big 64; pred (neg (big 63)) ]
        (fun n -> rejects lowered_type n);
      test
        "a weak expression whose bounds no 64-bit integer holds commits at \
         Int64" (fun () ->
          let v = Ops.variable "v" (z (Bigint.neg (big 64))) (z (big 64)) in
          let lowered =
            rewrite Uop_weak.pm_lower_weak (Ops.sink [ Ops.O.(v + int 1) ])
          in
          equal Dtypes.dtype Int64 (Ops.dtype (first Op.Add lowered)));
      (* a variable of the one value 2^65 is no constant: it has a width *)
      test
        "an expression whose bounds are one value no 64-bit integer holds \
         commits at Int64" (fun () ->
          let p = Ops.variable "p" (z (big 65)) (z (big 65)) in
          let lowered =
            rewrite Uop_weak.pm_lower_weak (Ops.sink [ Ops.O.(p * int 2) ])
          in
          equal Dtypes.dtype Int64 (Ops.dtype (first Op.Mul lowered)));
      (* a scalar parameter of slot 0 passed by value, bounded to an int32 *)
      test
        "a weak parameter passed by value is passed at its default, cast back"
        (fun () ->
          let param dt =
            Ops.v Op.Param
              ~src:[ Ops.size_src None ]
              ~arg:
                (Param
                   (Ops.param_arg
                      ~vmin_vmax:(i 0, i 100)
                      ~addrspace:Alu ~slot:0 dt))
          in
          equal Uops.uop
            (Ops.cast (param Int32) Weak_int)
            (rewrite Uop_weak.pm_lower_weak (param Weak_int)));
      lower_widths;
      lower_values;
      lower_opposite;
      lower_integers;
    ]

(* pm_cast_const *)

let cast_consts = rewrites Uop_weak.pm_cast_const

let cast_const_widths =
  prop "states the width of every constant" mixed_graph (fun es ->
      let u = sink wide es in
      let bare n = Ops.op n = Op.Const && is_weak n in
      cover "a weak constant beside a committed peer"
        (List.exists
           (fun n ->
             List.exists bare (Ops.src n)
             && List.exists (fun s -> not (is_weak s)) (Ops.src n))
           (nodes u));
      cover "a weak constant with no committed peer"
        (List.exists
           (fun n ->
             Ops.op n <> Op.Cast
             && List.exists bare (Ops.src n)
             && List.for_all is_weak (Ops.src n))
           (nodes u));
      equal (list Uops.uop) []
        (bare_consumers (rewrite Uop_weak.pm_cast_const u)))

let cast_const_idempotent =
  prop "is idempotent" mixed_graph (fun es ->
      let u = sink wide es in
      cover "a graph it changes" (rewrite Uop_weak.pm_cast_const u != u);
      Law.idempotent Uops.uop (rewrite Uop_weak.pm_cast_const) u)

(* pm_cast_const runs last, on a graph pm_lower_weak lowered. *)
let cast_const_values =
  prop "keeps the value of each integer node of a lowered graph" at_a_point
    (fun (es, env) ->
      let u = rewrite Uop_weak.pm_lower_weak (sink narrow es) in
      cover "a weak constant committed"
        (List.exists (fun n -> Ops.op n = Op.Const && is_weak n) (nodes u));
      equal values (values_of env u)
        (values_of env (rewrite Uop_weak.pm_cast_const u)))

let pm_cast_const =
  group "pm_cast_const"
    [
      group "rewrites each graph as its golden states"
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
            (Ops.sink [ Shape.valid (Ops.cast weak Int32) flag ]);
        ];
      test
        "a weak constant with no committed peer that no 64-bit integer holds \
         is refused" (fun () ->
          rejects
            (rewrite Uop_weak.pm_cast_const)
            (Ops.sink [ Ops.O.(Ops.const (z (big 70)) + int 1) ]));
      cast_const_widths;
      cast_const_idempotent;
      cast_const_values;
    ]

(* The passes in order *)

let all_passes u =
  rewrite Uop_weak.pm_cast_const
    (rewrite Uop_weak.pm_lower_weak (rewrite Uop_weak.pm_commit_weak u))

let in_order =
  group "commit, lower, then cast constants"
    [
      prop "leave every width stated" ~count:300 under_a_default
        (fun (es, df) ->
          let u = sink wide es in
          cover "a weak float" (has_weak_float u);
          cover "a weak integer committed past Int32"
            (has_weak_int_past_int32 u);
          let done_ = with_default_float df (fun () -> all_passes u) in
          equal (list Uops.uop) [] (List.filter weak_non_constant (nodes done_));
          equal (list Uops.uop) [] (bare_consumers done_));
      prop "keep the value of each integer node" at_a_point (fun (es, env) ->
          let u = sink narrow es in
          cover "a weak integer committed past Int32"
            (has_weak_int_past_int32 u);
          equal values (values_of env u) (values_of env (all_passes u)));
    ]

let () =
  exit
    (run "Tolk.Uop_weak"
       [
         in_order;
         commit_weak_consts;
         pm_commit_weak;
         pm_lower_weak;
         pm_cast_const;
       ])
