(* Tests of Tolk.Decomp_op: fast_idiv divides, threefry2x32 is the
   Threefry-2x32-20 hash, and each pattern rewrites as tinygrad's does and keeps
   the value of what it rewrites. *)

open Windtrap
open Tolk
open Dtypes

let i n = `Int (Bigint.of_int n)

let var ?(dtype = Dtype.Int32) name lo hi =
  Ops.variable ~dtype name (`Int lo) (`Int hi)

let v ?dtype name lo hi = var ?dtype name (Bigint.of_int lo) (Bigint.of_int hi)
let param ?(slot = 0) dt = Ops.param slot dt
let target = Result.get_ok (Helpers.Target.of_string "")

(* [everything] widens to any type; [nothing] has no type to widen to. *)
let everything = Renderer.v target
let nothing = Renderer.v ~native:(fun _ -> false) target

let renderer_named = function
  | "all" -> everything
  | "none" -> nothing
  | name -> invalid_arg name

let at vars u = Interpreter.eval ~vars u
let ops = Op.Set.of_list

(* A golden holds the sink of its inputs and that sink rewritten. *)
let rewrites file ~rewrite inputs =
  Golden.graph (file ^ ".golden") (fun () ->
      let s = Ops.sink inputs in
      Ops.sink [ s; rewrite s ])

(* fast_idiv *)

let dtype_named name = dtype_of_cell ("dtypes." ^ name)

let grid =
  Golden.cases ~key:[ "dtype"; "vmin"; "vmax"; "d"; "renderer" ]
    "fast_idiv_grid.golden" (fun cell ->
      let x =
        var
          ~dtype:(dtype_named (cell "dtype"))
          "x"
          (Bigint.of_string (cell "vmin"))
          (Bigint.of_string (cell "vmax"))
      in
      let r =
        Decomp_op.fast_idiv
          (renderer_named (cell "renderer"))
          x
          (Bigint.of_string (cell "d"))
      in
      equal string (cell "result")
        (match r with
        | None -> "None"
        | Some u -> Render.render ~simplify:false u))

(* The numerators a quotient is checked at: the ends of the range, each side of
   the first multiples of [d], and points spread between. *)
let numerators vmax d =
  let spread = List.init 16 (fun k -> Bigint.(vmax * of_int k / of_int 15)) in
  [
    Bigint.zero;
    Bigint.one;
    Bigint.pred d;
    d;
    Bigint.succ d;
    Bigint.(d + d);
    Bigint.pred vmax;
    vmax;
  ]
  @ spread
  |> List.filter (fun n -> Bigint.(geq n zero && leq n vmax))

let divides vmax d u =
  List.iter
    (fun n ->
      equal
        ~msg:(Format.asprintf "%a / %a" Bigint.pp_print n Bigint.pp_print d)
        const
        (`Int (Bigint.div n d))
        (at [ ("x", `Int n) ] u))
    (numerators vmax d)

let grid_divides =
  let rows =
    List.filter
      (fun cell -> cell "result" <> "None" && cell "vmin" = "0")
      (Golden.rows "fast_idiv_grid.golden")
  in
  let key cell = cell "dtype" ^ " " ^ cell "renderer" in
  let keys = List.sort_uniq compare (List.map key rows) in
  cases ~name:Fun.id "fast_idiv is division on the grid's dividends" keys
    (fun k ->
      List.iter
        (fun cell ->
          if key cell = k then
            let vmax = Bigint.of_string (cell "vmax")
            and d = Bigint.of_string (cell "d") in
            let x =
              var ~dtype:(dtype_named (cell "dtype")) "x" Bigint.zero vmax
            in
            let u =
              require_some
                (Decomp_op.fast_idiv (renderer_named (cell "renderer")) x d)
            in
            divides vmax d u)
        rows)

let integers =
  Dtype.[ Int8; Uint8; Int16; Uint16; Int32; Uint32; Int64; Uint64; Weak_int ]

(* [below hi] draws an integer in [0, hi], of every magnitude alike, with the
   ends and their neighbours. *)
let below hi =
  let open Gen in
  let* bits = int_range 1 (max 1 (Bigint.numbits hi)) in
  let* edge = int_range 0 7 in
  let+ lo = int64 and+ hi_bits = int64 in
  let r =
    Bigint.(
      extract
        (logor (of_int64_unsigned lo)
           (shift_left (of_int64_unsigned hi_bits) 64))
        0 bits)
  in
  match edge with
  | 0 -> Bigint.zero
  | 1 -> hi
  | 2 -> Bigint.max Bigint.zero (Bigint.pred hi)
  | _ -> Bigint.min r hi

let top dt =
  match dt with
  | Dtype.Weak_int -> Bigint.(pred (shift_left one 64))
  | _ -> snd (int_bounds dt)

let division_case =
  let open Gen in
  let* dt = of_list ~pp:Dtype.pp integers in
  let* vmax = below (top dt) in
  let vmax = Bigint.max vmax Bigint.one in
  let* d = map Bigint.succ (below Bigint.(max vmax (of_int 1000))) in
  let+ n = below vmax and+ wide = bool in
  (dt, vmax, d, n, wide)

let pp_division_case ppf (dt, vmax, d, n, wide) =
  Format.fprintf ppf "%a x in [0, %a], %a / %a%s" Dtype.pp dt Bigint.pp_print
    vmax Bigint.pp_print n Bigint.pp_print d
    (if wide then "" else ", nothing to widen to")

let fast_idiv_divides =
  prop ~count:500 "fast_idiv x d is x / d wherever it applies"
    (Gen.with_pp pp_division_case division_case) (fun (dt, vmax, d, n, wide) ->
      let x = var ~dtype:dt "x" Bigint.zero vmax in
      let r = if wide then everything else nothing in
      match Decomp_op.fast_idiv r x d with
      | None -> cover "declines" true
      | Some u ->
          cover "divides" true;
          cover "widens"
            (List.exists
               (fun u -> Ops.op u = Cast)
               (Ops.toposort ~calls:Enter u));
          equal const (`Int (Bigint.div n d)) (at [ ("x", `Int n) ] u))

(* A golden holds the dividend and its quotient. *)
let quotient file x d =
  Golden.graph (file ^ ".golden") (fun () ->
      Ops.sink
        [ x; require_some (Decomp_op.fast_idiv everything x (Bigint.of_int d)) ])

let fast_idiv =
  let x = v "x" 0 1000 in
  group "fast_idiv"
    [
      grid;
      grid_divides;
      fast_idiv_divides;
      quotient "fast_idiv_multiplies_and_shifts" x 7;
      quotient "fast_idiv_shifts_out_powers_of_two" (v "x" 0 (1 lsl 20)) 448;
      quotient "fast_idiv_widens" (v ~dtype:Int16 "x" 0 1000) 3;
      quotient "fast_idiv_folds_a_small_dividend" (v "x" 0 6) 7;
      test "fast_idiv declines a divisor that is not positive" (fun () ->
          is_none (Decomp_op.fast_idiv everything x Bigint.zero);
          is_none (Decomp_op.fast_idiv everything x (Bigint.of_int (-3))));
      test "fast_idiv declines a dividend that can be negative" (fun () ->
          is_none
            (Decomp_op.fast_idiv everything (v "x" (-1) 100) (Bigint.of_int 7)));
      test "fast_idiv of a dividend below the divisor is zero of its type"
        (fun () ->
          let u =
            require_some (Decomp_op.fast_idiv everything x (Bigint.of_int 1001))
          in
          equal Uops.uop (Ops.const_like x (i 0)) u);
      test "fast_idiv divides by a divisor beyond every integer type" (fun () ->
          let u =
            require_some
              (Decomp_op.fast_idiv everything (v ~dtype:Int64 "x" 0 100)
                 Bigint.(shift_left one 70))
          in
          equal const (i 0) (at [ ("x", i 100) ] u));
    ]

(* threefry2x32 *)

(* [hash ~counter ~key] is the Threefry-2x32-20 hash of the counter words [(c0,
   c1)] under the key words [(k0, k1)], as the words [(r0, r1)]; the first word
   of each pair is the low half of its Uint64. *)
let hash ~counter:(c0, c1) ~key:(k0, k1) =
  let word64 lo hi =
    `Int Bigint.(logor (of_int lo) (shift_left (of_int hi) 32))
  in
  let u = Decomp_op.threefry2x32 (param Uint64) (param ~slot:1 Uint64) in
  match Interpreter.eval ~params:[ (0, word64 c0 c1); (1, word64 k0 k1) ] u with
  | `Int r -> Bigint.(to_int (extract r 0 32), to_int (extract r 32 32))
  | c -> failf "a hash is an integer, not %a" (Testable.pp const) c

let words = pair int int

(* Random123's known-answer vectors for threefry2x32_20 (kat_vectors), as
   counter, key and result words. *)
let random123 =
  [
    ((0, 0), (0, 0), (0x6b200159, 0x99ba4efe));
    ( (0xffffffff, 0xffffffff),
      (0xffffffff, 0xffffffff),
      (0x1cb996fc, 0xbb002be7) );
    ( (0x243f6a88, 0x85a308d3),
      (0x13198a2e, 0x03707344),
      (0xc4923a9c, 0x483df7a0) );
  ]

(* JAX's threefry_2x32 with the key (0, 1337) over the counters 0 to 19, the
   reference of tinygrad's test_threefry_against_reference: the counter words
   (n, n + 10) hash to the values n and n + 10. *)
let jax =
  [
    (0, (2221762175, 3544324138));
    (1, (1752107825, 1436466838));
    (2, (653745012, 2169858556));
    (3, (1967534793, 2570072943));
    (4, (1395205442, 2387150698));
    (5, (3840423848, 3678370550));
    (6, (2159346757, 2911697663));
    (7, (603508235, 403244401));
    (8, (3319473678, 2560861638));
    (9, (3363866483, 1692360114));
  ]

let hash_of_constants =
  Decomp_op.threefry2x32 (Ops.int ~dtype:Uint64 5) (Ops.int ~dtype:Uint64 10)

let threefry =
  group "threefry2x32"
    [
      Golden.graph "threefry.golden" (fun () ->
          Ops.sink
            [ Decomp_op.threefry2x32 (param Uint64) (param ~slot:1 Uint64) ]);
      cases
        ~name:(fun ((c0, c1), (k0, k1), _) ->
          Printf.sprintf "counter %08x %08x key %08x %08x" c0 c1 k0 k1)
        "Random123's known answers" random123
        (fun (counter, key, expected) ->
          equal words expected (hash ~counter ~key));
      cases
        ~name:(fun (n, _) -> Printf.sprintf "counter %d %d" n (n + 10))
        "JAX's values under the key (0, 1337)" jax
        (fun (n, expected) ->
          equal words expected (hash ~counter:(n, n + 10) ~key:(0, 1337)));
      test "the hash of constants is a Uint64" (fun () ->
          equal dtype Dtype.Uint64 (Ops.dtype hash_of_constants));
      test "the hash of constants simplifies to a constant" (fun () ->
          let alu u = Op.Set.mem (Ops.op u) Op.Set.alu in
          equal (list Uops.uop) []
            (List.filter alu
               (Ops.toposort ~calls:Enter (Ops.simplify hash_of_constants))));
      (* Folding reads committed constants at their width, so the fold
         wraps each 32-bit word as the hash does. *)
      test "the hash of constants folds to its value" (fun () ->
          equal const
            (`Int (Bigint.of_string "6264663365535751564"))
            (Interpreter.eval (Ops.simplify hash_of_constants)));
    ]

(* Simplifying patterns *)

let wide = var ~dtype:Uint64 "w" Bigint.zero Bigint.(pred (shift_left one 64))
let two_to_63 = Ops.const ~dtype:Uint64 (`Int Bigint.(shift_left one 63))

let floordivs () =
  let p = v "p" 0 100 and n = v "n" (-100) 0 and m = v "m" (-10) 10 in
  let d = v "d" 1 7 in
  Ops.O.
    [
      p // int 3;
      n // v "q" (-7) (-1);
      m // int 3;
      m // d;
      p // d;
      m // int 8;
      m // int 1;
      m // int (-4);
      wide // two_to_63;
      wide // Ops.const (`Int Bigint.(shift_left one 63));
      wide // Ops.const (`Int Bigint.(shift_left one 64));
      v ~dtype:Weak_int "k" (-10) 10 // int 8;
      v "a" 0 3 // v "b" 0 3;
      v "c" (-3) 0 // v "e" (-3) 0;
      (Ops.maximum (v "x" (-5) 5) (int 0) + int 1) // int 3;
    ]

let floormods () =
  let p = v "p" 0 100 and n = v "n" (-100) 0 and m = v "m" (-10) 10 in
  Ops.O.
    [
      p % int 3;
      n % v "q" (-7) (-1);
      m % int 3;
      m % v "d" 1 7;
      m % int 4;
      m % int 1;
      m % int (-4);
      wide % two_to_63;
      wide % Ops.const (`Int Bigint.(shift_left one 63));
      wide % Ops.const (`Int Bigint.(shift_left one 64));
      v ~dtype:Weak_int "k" (-10) 10 % int 8;
      v "a" 0 3 % v "b" 0 3;
      v "c" (-3) 0 % v "e" (-3) 0;
    ]

let threefries () = [ Ops.alu (param Uint64) Threefry [ param ~slot:1 Uint64 ] ]

let simplifying_sets =
  [
    ("none", ops []);
    ("shr_and", ops [ Shr; And ]);
    ("shr_and_threefry", ops [ Shr; And; Threefry ]);
  ]

let simplify_with set s =
  Ops.graph_rewrite ~calls:Skip ~ctx:() s (Decomp_op.simplifying_patterns set)

let simplifying_graphs =
  List.concat_map
    (fun (family, inputs) ->
      List.map
        (fun (name, set) ->
          rewrites
            (Printf.sprintf "simplifying_%s_%s" family name)
            ~rewrite:(simplify_with set) (inputs ()))
        simplifying_sets)
    [
      ("floordiv", floordivs); ("floormod", floormods); ("threefry", threefries);
    ]

(* A floor division or remainder of drawn operands: the dividend's type and
   range, the divisor a constant or a variable of the same type, and a point in
   the ranges. *)
let floor_case =
  let open Gen in
  let* dt =
    of_list ~pp:Dtype.pp
      Dtype.[ Int8; Int16; Int32; Int64; Uint8; Uint32; Uint64; Weak_int ]
  in
  let lo, hi =
    match dt with
    | Weak_int -> (Bigint.of_int (-1000), Bigint.of_int 1000)
    | _ -> int_bounds dt
  in
  let point = map (fun k -> Bigint.(lo + k)) (below Bigint.(hi - lo)) in
  let* a0 = point in
  let* a1 = point in
  let* b = such_that (fun b -> not (Bigint.equal b Bigint.zero)) point in
  let* constant = bool in
  let* floor_mod = bool in
  let* shr_and = bool in
  let+ a =
    map (fun k -> Bigint.(min a0 a1 + k)) (below Bigint.(abs (a1 - a0)))
  in
  (dt, (Bigint.min a0 a1, Bigint.max a0 a1), a, b, constant, floor_mod, shr_and)

let pp_floor_case ppf (dt, (lo, hi), a, b, constant, floor_mod, shr_and) =
  Format.fprintf ppf "%a a=%a in [%a, %a] %s %s%a%s" Dtype.pp dt Bigint.pp_print
    a Bigint.pp_print lo Bigint.pp_print hi
    (if floor_mod then "%" else "//")
    (if constant then "" else "b=")
    Bigint.pp_print b
    (if shr_and then " with Shr and And" else "")

let floor_rewrites_keep_values =
  prop ~count:500 "floor divisions and remainders keep their values"
    (Gen.with_pp pp_floor_case floor_case)
    (fun (dt, (lo, hi), a, b, constant, floor_mod, shr_and) ->
      (* the least integer divided by -1 overflows its type *)
      assume
        (not
           (Bigint.equal b Bigint.minus_one
           && Bigint.equal a (fst (int_bounds dt))));
      let x = var ~dtype:dt "a" lo hi in
      let y, vars =
        if constant then (Ops.const (`Int b), [])
        else (var ~dtype:dt "b" b b, [ ("b", `Int b) ])
      in
      let e =
        if floor_mod then Ops.mod_ x y else Ops.div ~rounding:`Floor x y
      in
      let set = if shr_and then ops [ Shr; And ] else ops [] in
      let q = Bigint.fdiv a b in
      let expected = if floor_mod then Bigint.(a - (b * q)) else q in
      let r = simplify_with set e in
      cover "rewritten" (not (Ops.equal r e));
      equal const (`Int expected) (at (("a", `Int a) :: vars) r))

let simplifying =
  group "simplifying_patterns"
    (simplifying_graphs
    @ [
        floor_rewrites_keep_values;
        test "a target with Threefry keeps it" (fun () ->
            let t = Ops.sink (threefries ()) in
            equal Uops.uop t (simplify_with (ops [ Threefry ]) t));
      ])

(* Late patterns *)

let late_with ?(disable_fast_idiv = true) ?(renderer = everything) set s =
  Ops.graph_rewrite ~calls:Skip ~ctx:renderer s
    (Decomp_op.late_patterns ~disable_fast_idiv set)

let fparam ?slot () = param ?slot Float32

let maxes () =
  [
    Ops.maximum (v "x" (-5) 5) (v "y" (-5) 5);
    Ops.maximum (fparam ()) (fparam ~slot:1 ());
    Ops.maximum (v ~dtype:Int16 "s" (-5) 5) (v "y" (-5) 5);
  ]

let logical () =
  let a = param Bool and b = param ~slot:1 Bool in
  [ Ops.bitwise_and (Ops.logical_not a) (Ops.logical_not b) ]

let muls () =
  let x = v "x" (-100) 100 and k = v ~dtype:Weak_int "k" (-100) 100 in
  Ops.O.
    [
      x * int 8;
      x * int 1;
      x * int 3;
      x * int (-8);
      k * int 16;
      v ~dtype:Uint32 "u" 0 100 * int 4;
      fparam () * float 4.;
    ]

let cdivs () =
  let u = v ~dtype:Uint32 "u" 0 1000 and p = v "p" 0 1000 in
  let m = v "m" (-1000) 1000 in
  let cdiv x d = Ops.alu x Cdiv [ d ] and cmod x d = Ops.alu x Cmod [ d ] in
  let big =
    var ~dtype:Uint32 "x" Bigint.zero Bigint.(pred (shift_left one 32))
  in
  [
    cdiv u (Ops.int 8);
    cdiv u (Ops.int 1);
    cdiv p (Ops.int 8);
    cdiv m (Ops.int 8);
    cdiv m (Ops.int 1);
    cdiv p (Ops.int 7);
    cmod p (Ops.int 7);
    cdiv m (Ops.int 7);
    cmod m (Ops.int 7);
    cmod p (v "d" 1 9);
    cdiv p (Ops.int (-3));
    cmod p (Ops.int 0);
    cmod wide (Ops.int 3);
    cdiv big (Ops.int 7);
    cmod big (Ops.int 7);
    cdiv (v ~dtype:Weak_int "k" 0 100) (Ops.int 8);
    cmod m (Ops.int 4);
    cdiv (v ~dtype:Int64 "l" 0 100)
      (Ops.const (`Int (Bigint.of_int64 Int64.max_int)));
  ]

let negations () =
  let x = v "x" (-5) 5 and y = v "y" (-5) 5 in
  let f = fparam () and g = fparam ~slot:1 () in
  let neg u = Ops.alu u Neg [] in
  Ops.O.[ x * int (-1); f * float (-1.); x + neg y; neg y + x; f + neg g ]

let comparisons () =
  let x = v "x" (-10) 10 and y = v "y" (-10) 10 in
  let u = v ~dtype:Uint32 "u" 0 10 in
  let c3 = Ops.int ~dtype:Int32 3 and c5 = Ops.int ~dtype:Int32 5 in
  Ops.O.
    [
      Ops.logical_not (x < int 5);
      Ops.logical_not (int 5 < x);
      Ops.logical_not (u < int 5);
      x * int (-1) < y * int 3;
      x * int (-1) < int 5;
      (int 3 < x) land (x < int 5);
      (x < int 5) land (int 3 < x);
      (int 3 < x) land (x < int 6);
      (int 3 < u) land (u < int 5);
      (c3 < x) land (x < c5);
      Ops.logical_not (x <> y);
      Ops.logical_not (fparam () <> fparam ~slot:1 ());
    ]

let extremes () =
  let x = param Int64 and y = param ~slot:1 Int64 in
  let c n = Ops.const (`Int n) in
  let lo = Bigint.of_int64 Int64.min_int
  and hi = Bigint.of_int64 Int64.max_int in
  let two_to_80 = Bigint.(shift_left one 80) in
  Ops.O.
    [
      Ops.logical_not (x < c lo);
      Ops.logical_not (c hi < x);
      x * int (-1) < c lo;
      x * int (-1) < y * c lo;
      (c hi < x) land (x < c (Bigint.succ lo));
      (c lo < x) land (x < c Bigint.(lo + of_int 2));
      (c Bigint.(hi - of_int 2) < x) land (x < c hi);
      (c (Bigint.pred two_to_80) < x) land (x < c (Bigint.succ two_to_80));
    ]

let mulaccs () =
  let a = v "a" (-5) 5 and b = v "b" (-5) 5 and c = v "c" (-5) 5 in
  let f = fparam () and g = fparam ~slot:1 () and h = fparam ~slot:2 () in
  Ops.O.
    [
      (a * b) + c;
      c + (a * b);
      (f * g) + h;
      Ops.alu a Shl [ Ops.int ~dtype:Int32 3 ] + c;
      (a * int 4) + c;
    ]

let divisions () =
  let a = fparam () and b = fparam ~slot:1 () in
  let one_over b = Ops.alu (Ops.float ~dtype:Float32 1.) Fdiv [ b ] in
  Ops.O.
    [ Ops.reciprocal b; a * one_over b; one_over b * a; a * Ops.reciprocal b ]

let late_families =
  [
    ( "max",
      maxes,
      [ ("none", []); ("cmplt", [ Op.Cmplt ]); ("max_cmplt", [ Max; Cmplt ]) ]
    );
    ("logical", logical, [ ("none", []); ("or", [ Or ]) ]);
    ("mul", muls, [ ("none", []); ("shl", [ Shl ]) ]);
    ("cdiv", cdivs, [ ("none", []); ("shr", [ Shr ]) ]);
    ( "negation",
      negations,
      [ ("none", []); ("neg", [ Neg ]); ("neg_sub", [ Neg; Sub ]) ] );
    ( "comparison",
      comparisons,
      [ ("none", []); ("cmplt", [ Cmplt ]); ("cmpeq", [ Cmpeq ]) ] );
    ("extremes", extremes, [ ("cmplt", [ Cmplt ]) ]);
    ( "mulacc",
      mulaccs,
      [ ("none", []); ("mulacc", [ Mulacc ]); ("mulacc_shl", [ Mulacc; Shl ]) ]
    );
    ("division", divisions, [ ("none", []); ("fdiv", [ Fdiv ]) ]);
  ]

let late_graphs =
  List.concat_map
    (fun (family, inputs, sets) ->
      List.map
        (fun (name, set) ->
          rewrites
            (Printf.sprintf "late_%s_%s" family name)
            ~rewrite:(late_with (ops set))
            (inputs ()))
        sets)
    late_families
  @ [
      rewrites "late_cdiv_shr_fast_idiv"
        ~rewrite:(late_with ~disable_fast_idiv:false (ops [ Shr ]))
        (cdivs ());
      rewrites "late_cdiv_shr_fast_idiv_narrow"
        ~rewrite:
          (late_with ~disable_fast_idiv:false ~renderer:nothing (ops [ Shr ]))
        (cdivs ());
    ]

(* A truncating division or remainder by a positive constant, of a dividend of
   any integer type and range, lowered with Shr and fast_idiv. *)
let trunc_case =
  let open Gen in
  let* dt = of_list ~pp:Dtype.pp integers in
  let lo, hi =
    match dt with
    | Weak_int -> (Bigint.of_int (-100_000), Bigint.of_int 100_000)
    | _ -> int_bounds dt
  in
  let point = map (fun k -> Bigint.(lo + k)) (below Bigint.(hi - lo)) in
  let* a0 = point in
  let* a1 = point in
  let top = hi in
  let lo, hi = (Bigint.min a0 a1, Bigint.max a0 a1) in
  let* power = int_range 0 (min 12 (Bigint.numbits top - 1)) in
  let* d =
    frequency
      [
        (1, constant Bigint.(shift_left one power));
        ( 3,
          map Bigint.succ
            (below (Bigint.min (Bigint.pred top) (Bigint.of_int 100_000))) );
      ]
  in
  let* remainder = bool in
  let* narrow = bool in
  let+ a = map (fun k -> Bigint.(lo + k)) (below Bigint.(hi - lo)) in
  (dt, (lo, hi), a, d, remainder, narrow)

let pp_trunc_case ppf (dt, (lo, hi), a, d, remainder, narrow) =
  Format.fprintf ppf "%a a=%a in [%a, %a] %s %a%s" Dtype.pp dt Bigint.pp_print a
    Bigint.pp_print lo Bigint.pp_print hi
    (if remainder then "cmod" else "cdiv")
    Bigint.pp_print d
    (if narrow then ", nothing to widen to" else "")

let truncations_keep_values =
  prop ~count:500 "truncating divisions and remainders keep their values"
    (Gen.with_pp pp_trunc_case trunc_case)
    (fun (dt, (lo, hi), a, d, remainder, narrow) ->
      let x = var ~dtype:dt "a" lo hi in
      let e =
        Ops.alu x (if remainder then Cmod else Cdiv) [ Ops.const (`Int d) ]
      in
      let renderer = if narrow then nothing else everything in
      let r = late_with ~disable_fast_idiv:false ~renderer (ops [ Shr ]) e in
      cover "rewritten" (not (Ops.equal r e));
      equal const
        (`Int (if remainder then Bigint.rem a d else Bigint.div a d))
        (at [ ("a", `Int a) ] r))

(* The integer expressions the other late rules rewrite, over x, y and z in
   [-50, 50]. *)
let late_expressions =
  let x = v "x" (-50) 50 and y = v "y" (-50) 50 and z = v "z" (-50) 50 in
  let neg u = Ops.alu u Neg [] in
  Ops.O.
    [
      ("max x y", Ops.maximum x y);
      ("not (x < 7)", Ops.logical_not (x < int 7));
      ("not (7 < x)", Ops.logical_not (int 7 < x));
      ("x * -1 < y * 3", x * int (-1) < y * int 3);
      ("x * -1 < 7", x * int (-1) < int 7);
      ("6 < x and x < 8", (int 6 < x) land (x < int 8));
      ("not (x <> y)", Ops.logical_not (x <> y));
      ("x * y + z", (x * y) + z);
      ("x << 3 + z", Ops.alu x Shl [ Ops.int ~dtype:Int32 3 ] + z);
      ("x * 8", x * int 8);
      ("x + neg y", x + neg y);
      ("x * -1", x * int (-1));
    ]

let every_late_op = ops [ Cmplt; Or; Shl; Shr; Neg; Sub; Cmpeq; Mulacc; Fdiv ]

let late_rules_keep_values =
  let point = Gen.int_range (-50) 50 in
  prop ~count:300 "the late rules keep the values of integer expressions"
    Gen.(
      triple
        (of_list
           ~pp:(fun ppf (name, _) -> Format.pp_print_string ppf name)
           late_expressions)
        (pair point point) point)
    (fun ((_, e), (x, y), z) ->
      let vars = [ ("x", i x); ("y", i y); ("z", i z) ] in
      let r = late_with every_late_op e in
      cover "rewritten" (not (Ops.equal r e));
      equal const (at vars e) (at vars r))

(* The divisions of the old tolk's run of constant integer division, lowered as
   code generation lowers them: the simplifying patterns, then the late ones
   with fast_idiv, for a target with Shr and And. *)
let lowered_divisions =
  let divisors = [ 2; 3; 4; 6; 7; 8; 19; 65537; 2147483647 ] in
  let lower e =
    late_with ~disable_fast_idiv:false
      (ops [ Shr; And ])
      (simplify_with (ops [ Shr; And ]) e)
  in
  cases
    ~name:(fun (dt, _) -> Format.asprintf "%a" Dtype.pp dt)
    "lowered divisions compute truncating and floor quotients and remainders"
    [
      ( Dtype.Uint32,
        [
          0;
          1;
          2;
          3;
          6;
          7;
          18;
          19;
          20;
          65536;
          65537;
          2147483647;
          2147483648;
          4294967294;
          4294967295;
        ] );
      ( Dtype.Int32,
        [ -2147483648; -65537; -20; -19; -7; -1; 0; 1; 7; 2147483647 ] );
    ]
    (fun (dt, values) ->
      let lo, hi = int_bounds dt in
      let x = var ~dtype:dt "x" lo hi in
      List.iter
        (fun divisor ->
          let d = Ops.int divisor in
          List.iter
            (fun n ->
              let a = Bigint.of_int n and b = Bigint.of_int divisor in
              let check name e expected =
                equal
                  ~msg:(Printf.sprintf "%d %s %d" n name divisor)
                  const (`Int expected)
                  (at [ ("x", `Int a) ] (lower e))
              in
              let q = Bigint.fdiv a b in
              check "cdiv" (Ops.alu x Cdiv [ d ]) (Bigint.div a b);
              check "cmod" (Ops.alu x Cmod [ d ]) (Bigint.rem a b);
              check "//" (Ops.div ~rounding:`Floor x d) q;
              check "%" (Ops.mod_ x d) Bigint.(a - (b * q)))
            values)
        divisors)

let late =
  group "late_patterns"
    (late_graphs
    @ [ truncations_keep_values; late_rules_keep_values; lowered_divisions ])

(* Constants other than integers *)

(* The rules that read a constant divisor, factor or bound take an integer: a
   boolean, a float or [`Invalid] there is declined (README, CPython's treatment
   of constants other than integers). The nodes are built as given, without
   promotion. *)
let non_integers =
  let x = v "x" 0 100 in
  let has op u =
    List.exists (fun n -> Ops.op n = op) (Ops.toposort ~calls:Enter u)
  in
  let keeps rewrite e = equal Uops.uop e (rewrite e) in
  let shr_and = ops [ Shr; And ] in
  group "constants other than integers are declined"
    [
      test "the mask rule declines a remainder by True" (fun () ->
          let r =
            simplify_with shr_and (Ops.alu x Floormod [ Ops.bool true ])
          in
          is_false (has And r));
      test "the shift rules decline 2.0" (fun () ->
          is_false
            (has Shr
               (simplify_with shr_and (Ops.alu x Floordiv [ Ops.float 2. ])));
          keeps (late_with (ops [ Shl ])) (Ops.alu x Mul [ Ops.float 2. ]);
          keeps (late_with (ops [ Shr ])) (Ops.alu x Cdiv [ Ops.float 2. ]));
      test "fast_idiv declines True, a float and Invalid" (fun () ->
          List.iter
            (fun d ->
              keeps
                (late_with ~disable_fast_idiv:false (ops [ Shr ]))
                (Ops.alu x Cdiv [ d ]))
            [ Ops.bool true; Ops.float 7.; Ops.invalid ]);
      test "the one-integer-between rule declines Invalid" (fun () ->
          keeps
            (late_with (ops [ Cmplt ]))
            Ops.O.(Ops.alu Ops.invalid Cmplt [ x ] land (x < int 5)));
    ]

let () =
  exit
    (run "Tolk.Decomp_op"
       [ fast_idiv; threefry; simplifying; late; non_integers ])
