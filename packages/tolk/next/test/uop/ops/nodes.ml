(* Nodes: axis types, arguments, identity, data types, graphs, shapes and
   ranges. *)

open Windtrap
open Tolk_next
open Common

let str pp x = Format.asprintf "%a" pp x

(* Axis types *)

let axis_types =
  Ops.Axis_type.
    [
      Device;
      Global;
      Warp;
      Local;
      Weak;
      Reduce;
      Upcast;
      Unroll;
      Placeholder;
      Loop;
    ]

let axis_type_of_cell s =
  Result.get_ok (Ops.Axis_type.of_string (after "AxisType." s))

let color_of_cell : string -> Helpers.color = function
  | "green" -> Green
  | "blue" -> Blue
  | "CYAN" -> Bright_cyan
  | "cyan" -> Cyan
  | "WHITE" -> Bright_white
  | "red" -> Red
  | "yellow" -> Yellow
  | "magenta" -> Magenta
  | s -> invalid_arg s

let color =
  Testable.make
    ~pp:(fun ppf _ -> Format.pp_print_string ppf "<color>")
    ~equal:( = )

let axis_type_group =
  group "Axis_type"
    [
      Golden.cases "axis_types.golden" (fun cell ->
          let a = axis_type_of_cell (cell "axis_type") in
          equal string (cell "axis_type") (str Ops.Axis_type.pp a);
          expect string Fun.id (cell "letter") (fun () ->
              Ops.Axis_type.letter a);
          expect color color_of_cell (cell "color") (fun () ->
              Ops.Axis_type.color a);
          expect int int_of_string (cell "position") (fun () ->
              Ops.Axis_type.position a));
      test "of_string reads back the name pp formats, without its prefix"
        (fun () ->
          List.iter
            (fun a ->
              let name = after "AxisType." (str Ops.Axis_type.pp a) in
              is_true ~msg:name (Ops.Axis_type.of_string name = Ok a))
            axis_types);
      test "of_string rejects a name in lower case, naming it" (fun () ->
          match Ops.Axis_type.of_string "reduce" with
          | Ok _ -> fail "read reduce"
          | Error e -> contains ~sub:"reduce" e);
      test "compare follows the declaration order" (fun () ->
          List.iteri
            (fun i a ->
              List.iteri
                (fun j b ->
                  equal ~msg:(str Ops.Axis_type.pp a) int
                    (Int.compare i j |> Int.min 1 |> Int.max (-1))
                    (Ops.Axis_type.compare a b |> Int.min 1 |> Int.max (-1)))
                axis_types)
            axis_types);
    ]

(* Devices *)

let devices =
  group "devices"
    [
      test "pp_device formats a name quoted, and several as a tuple" (fun () ->
          equal string "'CPU'" (str Ops.pp_device (Single "CPU"));
          equal string "('CPU:0', 'CPU:1')"
            (str Ops.pp_device (Multi [ "CPU:0"; "CPU:1" ]));
          equal string "('CPU:0',)" (str Ops.pp_device (Multi [ "CPU:0" ])));
      test "equal_device tells one device from one of several" (fun () ->
          is_true (Ops.equal_device (Single "CPU") (Single "CPU"));
          is_false (Ops.equal_device (Single "CPU") (Multi [ "CPU" ]));
          is_false (Ops.equal_device (Multi [ "A"; "B" ]) (Multi [ "B"; "A" ])));
    ]

(* Arguments *)

let reprs = lazy (Goldens.reprs ())

let arguments =
  group "arguments"
    [
      Golden.cases "reprs.golden" (fun cell ->
          let u = List.assoc (cell "name") (Lazy.force reprs) in
          equal string (cell "arg") (str Ops.pp_arg (Ops.arg u));
          equal string (cell "tag")
            (match Ops.tag u with None -> "None" | Some t -> str Ops.Tag.pp t));
      test "reprs.golden names every argument the suite builds" (fun () ->
          equal
            (slist string String.compare)
            (List.map (fun cell -> cell "name") (Golden.rows "reprs.golden"))
            (List.map fst (Lazy.force reprs)));
      test "each payload formats as its argument does" (fun () ->
          List.iter
            (fun (name, u) ->
              let payload =
                match Ops.arg u with
                | Param p -> Some (str Ops.pp_param_arg p)
                | Kernel k -> Some (str Ops.pp_kernel_info k)
                | Program p -> Some (str Ops.pp_program_info p)
                | Call c -> Some (str Ops.pp_call_info c)
                | Bufferize b -> Some (str Ops.pp_bufferize_opts b)
                | _ -> None
              in
              Option.iter
                (fun text ->
                  equal ~msg:name string (str Ops.pp_arg (Ops.arg u)) text)
                payload)
            (Lazy.force reprs));
      test "pp_estimates formats the record" (fun () ->
          equal string "Estimates(ops=0, lds=0, mem=0)"
            (str Ops.pp_estimates { ops = Int 0; lds = Int 0; mem = Int 0 }));
      test "param_arg defaults to global storage with no flags" (fun () ->
          let p = Ops.param_arg ~slot:3 Dtype.Float32 in
          is_true (p.addrspace = Some Dtype.Global);
          is_false p.volatile;
          is_false p.bind_on_realize;
          is_true (p.size = None && p.vmin_vmax = None && p.multiple_of = None);
          is_true (p.name = None && p.device = None && p.bound = None));
      test "kernel_info defaults to the kernel named test" (fun () ->
          let k = Ops.kernel_info () in
          equal string "test" k.name;
          is_true
            (k.applied_opts = [] && k.opts_to_apply = None && k.estimates = None);
          equal int 0 k.beam);
      test "function_name is the name as an identifier" (fun () ->
          List.iter
            (fun (name, expected) ->
              equal ~msg:name string expected
                (Ops.function_name (Ops.kernel_info ~name ())))
            [
              ("", "");
              ("9abc", "9abc");
              ("kernel name", "kernel20name");
              ("a-b.c", "a2Db2Ec");
              ("\xC3\xA9", "E9");
              ("\027[31mred\027[0m", "red");
            ]);
      test "equal_arg tells apart arguments that differ in one field" (fun () ->
          is_false
            (Ops.equal_arg
               (Range { axis_id = [ 0 ]; axis_type = Reduce })
               (Range { axis_id = [ 0 ]; axis_type = Upcast }));
          is_false
            (Ops.equal_arg
               (Range { axis_id = [ 0 ]; axis_type = Reduce })
               (Range { axis_id = [ 1 ]; axis_type = Reduce }));
          is_false
            (Ops.equal_arg
               (Reduce { op = Op.Add; num_axes = 1 })
               (Reduce { op = Op.Add; num_axes = 2 }));
          is_false
            (Ops.equal_arg
               (Code { code = "a"; dtype = Void })
               (Code { code = "a"; dtype = Int32 }));
          is_false
            (Ops.equal_arg
               (Queue { devices = [ "A" ]; queue = "q" })
               (Queue { devices = [ "B" ]; queue = "q" }));
          is_false (Ops.equal_arg (Flips [ true ]) (Flips [ false ]));
          is_false (Ops.equal_arg (Shard 0) (Axes [ 0 ]));
          is_true
            (Ops.equal_arg
               (Queue { devices = [ "A" ]; queue = "q" })
               (Queue { devices = [ "A" ]; queue = "q" })));
      test "equal_arg compares constants by their bits and nodes physically"
        (fun () ->
          is_false
            (Ops.equal_arg
               (Const (`Float Float.nan))
               (Const (`Float (-.Float.nan))));
          is_false (Ops.equal_arg (Const (`Float 0.)) (Const (`Float (-0.))));
          is_false (Ops.equal_arg (Const (i 1)) (Const (`Bool true)));
          let a = var "a" 0 1 in
          let k u =
            Ops.Program
              { (Ops.program_info_of_sink (Ops.sink [])) with vars = [ u ] }
          in
          is_true (Ops.equal_arg (k a) (k (var "a" 0 1)));
          is_false (Ops.equal_arg (k a) (k (var "b" 0 1))));
    ]

(* Printing *)

let printing =
  group "printing"
    [
      Golden.text "pretty.golden" (fun () ->
          let print u =
            let b = Buffer.create 256 in
            let ppf = Format.formatter_of_buffer b in
            Format.pp_set_margin ppf 10_000;
            Format.fprintf ppf "%a@?" Ops.pp u;
            Buffer.contents b
          in
          String.concat "\n\n" (List.map print (Goldens.printed_nodes ()))
          ^ "\n");
    ]

(* Tags *)

let gen_tag =
  let open Gen in
  let leaf =
    frequency
      [
        (1, map (fun b -> Ops.Tag.Bool b) bool);
        (1, map (fun n -> Ops.Tag.Int n) small_int);
        (1, map (fun s -> Ops.Tag.String s) (of_list [ ""; "x"; "mergeable" ]));
        (1, map (fun dt -> Ops.Tag.Dtype dt) (of_list Dtypes.declared));
        (1, map (fun s -> Ops.Tag.Bytes s) (of_list [ ""; "\x00a'" ]));
      ]
  in
  with_pp Ops.Tag.pp
    (frequency
       [
         (3, leaf);
         (1, map (fun l -> Ops.Tag.Tuple l) (list ~size:(int_range 0 3) leaf));
       ])

let tag = Testable.make ~pp:Ops.Tag.pp ~equal:Ops.Tag.equal

(* A pair of tags, equal a quarter of the time: the second is then a fresh copy
   of the first. *)
let gen_tag_pair =
  let rec copy : Ops.Tag.t -> Ops.Tag.t = function
    | String s -> String (String.init (String.length s) (String.get s))
    | Bytes s -> Bytes (String.init (String.length s) (String.get s))
    | Tuple l -> Tuple (List.map copy l)
    | t -> t
  in
  Gen.frequency
    [
      (1, Gen.map (fun t -> (t, copy t)) gen_tag); (3, Gen.pair gen_tag gen_tag);
    ]

let tags =
  group "Tag"
    [
      prop "equal is an equivalence" (Gen.pair gen_tag gen_tag)
        (Law.equivalence tag);
      prop "hash agrees with equal" gen_tag_pair (fun (t0, t1) ->
          cover "equal tags" (Ops.Tag.equal t0 t1);
          if Ops.Tag.equal t0 t1 then
            equal int (Ops.Tag.hash t0) (Ops.Tag.hash t1));
      test "pp formats a literal as Python writes it" (fun () ->
          List.iter
            (fun (t, text) -> equal string text (str Ops.Tag.pp t))
            Ops.Tag.
              [
                (Bool true, "True");
                (Int 1, "1");
                (String "mergeable", "'mergeable'");
                (Dtype Int32, "dtypes.int");
                (Tuple [ Int 0; Dtype Int32 ], "(0, dtypes.int)");
                (Tuple [ Int 0 ], "(0,)");
                (Tuple [], "()");
                (Bytes "\x00a'", "b\"\\x00a'\"");
              ]);
    ]

(* Queue calls *)

let hcq ?(skip_wait = false) device : Ops.hcq_info =
  let zero : Ops.estimates = { ops = Int 0; lds = Int 0; mem = Int 0 } in
  {
    device;
    kernels = [];
    estimates = zero;
    nargs = 0;
    table = -1;
    inputs = [];
    slots = [];
    host_deps = [];
    written_bufs = [];
    skip_wait;
  }

let hcq_calls =
  group "queue calls"
    [
      test "pp_hcq_info formats every field, as the record's repr" (fun () ->
          equal string
            "HCQInfo(device=('AMD',), kernels=(), estimates=Estimates(ops=0, \
             lds=0, mem=0), nargs=0, table=-1, inputs=(), slots=(), \
             host_deps=(), written_bufs=(), skip_wait=False)"
            (Format.asprintf "%a"
               (fun ppf ->
                 Format.pp_set_margin ppf 10_000;
                 Ops.pp_hcq_info ppf)
               (hcq [ "AMD" ])));
      test "a call's queue data is part of its identity" (fun () ->
          let call aux = Ops.call ?aux (Ops.sink []) [] in
          is_true (call (Some (hcq [ "AMD" ])) == call (Some (hcq [ "AMD" ])));
          is_false
            (call (Some (hcq [ "AMD" ]))
            == call (Some (hcq ~skip_wait:true [ "AMD" ])));
          is_false (call (Some (hcq [ "AMD" ])) == call None));
    ]

(* Identity *)

(* Random expressions over two variables, built from a recipe so that each can
   be built twice. *)
type recipe =
  | Leaf of int
  | Konst of int
  | Bin of Op.t * recipe * recipe
  | Neg of recipe

let rec pp_recipe ppf = function
  | Leaf n -> Format.fprintf ppf "v%d" n
  | Konst n -> Format.pp_print_int ppf n
  | Bin (o, a, b) ->
      Format.fprintf ppf "(%a %a %a)" Op.pp o pp_recipe a pp_recipe b
  | Neg a -> Format.fprintf ppf "-%a" pp_recipe a

let gen_recipe =
  let open Gen in
  let rec go depth =
    if depth = 0 then
      frequency
        [
          (2, map (fun n -> Leaf n) (int_range 0 1));
          (1, map (fun n -> Konst n) (int_range (-3) 3));
        ]
    else
      frequency
        [
          (1, go 0);
          ( 3,
            let* o =
              of_list ~pp:Op.pp
                Op.
                  [
                    Add;
                    Mul;
                    Max;
                    Cmplt;
                    Cmpne;
                    Xor;
                    And;
                    Floordiv;
                    Floormod;
                    Cdiv;
                    Cmod;
                  ]
            in
            let* a = go (depth - 1) in
            let+ b = go (depth - 1) in
            Bin (o, a, b) );
          (1, map (fun a -> Neg a) (go (depth - 1)));
        ]
  in
  with_pp pp_recipe (bind (int_range 0 4) go)

let leaves () = [| var "v0" (-8) 8; var "v1" 1 5 |]

let rec build vars = function
  | Leaf n -> vars.(n)
  | Konst n -> Ops.int ~dtype:Dtype.Int32 n
  | Neg a -> Ops.neg (build vars a)
  | Bin (((Cmplt | Cmpne) as o), a, b) ->
      Ops.cast (Ops.alu (build vars a) o [ build vars b ]) Dtype.Int32
  | Bin (o, a, b) -> Ops.alu (build vars a) o [ build vars b ]

let identity =
  group "identity"
    [
      prop "building a graph twice gives the same node" gen_recipe (fun r ->
          is_true (build (leaves ()) r == build (leaves ()) r));
      prop "structurally different graphs are different nodes"
        (Gen.pair gen_recipe gen_recipe) (fun (r0, r1) ->
          assume (r0 <> r1);
          let vars = leaves () in
          let normal r = Graph.to_string (build vars r) in
          equal bool (normal r0 = normal r1) (build vars r0 == build vars r1));
      prop "equal is an equivalence" (Gen.pair gen_recipe gen_recipe)
        (fun (r0, r1) ->
          let vars = leaves () in
          Law.equivalence uop (build vars r0, build vars r1));
      prop "compare is a total order that agrees with equal"
        (Gen.triple gen_recipe gen_recipe gen_recipe) (fun (r0, r1, r2) ->
          let vars = leaves () in
          Law.order uop (build vars r0, build vars r1, build vars r2));
      prop "hash agrees with equal on live nodes" gen_recipe (fun r ->
          let u0 = build (leaves ()) r in
          let u1 = build (leaves ()) r in
          equal int (Ops.hash u0) (Ops.hash u1);
          is_true (Sys.opaque_identity u0 == u1));
      test "zero and negative zero are different nodes" (fun () ->
          is_false (Ops.float 0. == Ops.float (-0.)));
      (* D27. A NaN constant keeps its bits, where tinygrad makes every NaN
         constant the same. *)
      test "NaN constants of different bits are different nodes" (fun () ->
          is_false
            (Ops.float Float.nan
            == Ops.float (Int64.float_of_bits 0x7FF4_0000_0000_0000L));
          is_false (Ops.float Float.nan == Ops.float (-.Float.nan));
          is_true (Ops.float Float.nan == Ops.float Float.nan));
      test "an integer and a boolean of equal value are different nodes"
        (fun () ->
          is_false (Ops.int 1 == Ops.bool true);
          is_false (Ops.int 0 == Ops.float 0.));
      test "a tag is part of the node" (fun () ->
          let a = Ops.int 1 in
          is_false (a == Ops.rtag a);
          is_true (Ops.rtag a == Ops.rtag ~tag:(Bool true) a);
          is_false (Ops.rtag ~tag:(String "x") a == Ops.rtag ~tag:(String "y") a));
      test "replace is the node itself when nothing changes" (fun () ->
          let a = Ops.O.(var "a" 0 4 + Ops.int 1) in
          is_true (Ops.replace a == a);
          is_true
            (Ops.replace ~op:Op.Add ~src:(Ops.src a) ~arg:Ops.No_arg a == a);
          is_true
            (Ops.replace ~op:Op.Mul a
            == Ops.alu (Ops.nth a 0) Op.Mul [ Ops.nth a 1 ]);
          is_true (Ops.replace ~tag:None (Ops.rtag a) == a));
      test "nth is the ith source, and rejects one past the last" (fun () ->
          let a = var "a" 0 4 and b = var "b" 0 4 in
          equal uop b (Ops.nth Ops.O.(a + b) 1);
          rejects (fun () -> Ops.nth Ops.O.(a + b) 2);
          rejects (fun () -> Ops.nth Ops.O.(a + b) (-1)));
      test "a node nothing references is collected" (fun () ->
          let cell = Weak.create 2 in
          let[@inline never] build () =
            let r = Ops.range (Int 9127) [ 9127 ] in
            let c = Ops.O.(r < int 4) in
            ignore
              ( Ops.ranges c,
                Ops.vmin r,
                Ops.shape r,
                Ops.backward_slice c,
                Ops.device c,
                Ops.addrspace c );
            Weak.set cell 0 (Some r);
            Weak.set cell 1 (Some (Ops.end_ c [ r ]))
          in
          build ();
          Gc.full_major ();
          Gc.full_major ();
          is_false ~msg:"a range" (Weak.check cell 0);
          is_false ~msg:"an end" (Weak.check cell 1));
      test "arguments that differ in one field are different nodes" (fun () ->
          let est ops lds mem : Ops.estimates =
            { ops = Int ops; lds = Int lds; mem = Int mem }
          in
          let kernel ?(beam = 0) e =
            Ops.sink ~kernel:(Ops.kernel_info ~estimates:e ~beam ()) []
          in
          is_false (kernel (est 1 2 3) == kernel (est 1 2 4));
          is_false (kernel (est 1 2 3) == kernel (est 1 3 3));
          is_false (kernel (est 1 2 3) == kernel ~beam:1 (est 1 2 3));
          let program target =
            Ops.v
              ~arg:
                (Program
                   { (Ops.program_info_of_sink (Ops.sink [])) with target })
              Op.Program
          in
          let cpu = Result.get_ok (Helpers.Target.parse "CPU") in
          let cuda = Result.get_ok (Helpers.Target.parse "CUDA") in
          is_false (program cpu == program cuda);
          is_false
            (Ops.range ~axis_type:Reduce (Int 4) [ 0 ]
            == Ops.range ~axis_type:Upcast (Int 4) [ 0 ]));
      test "Tbl is keyed by identity" (fun () ->
          let t = Ops.Tbl.create 4 in
          Ops.Tbl.replace t (var "a" 0 4) 1;
          Ops.Tbl.replace t (var "a" 0 4) 2;
          Ops.Tbl.replace t (var "a" 0 5) 3;
          equal int 2 (Ops.Tbl.length t);
          equal (option int) (Some 2) (Ops.Tbl.find_opt t (var "a" 0 4)));
      test "nodes built on several domains at once are the same node" (fun () ->
          let build () =
            Array.init 200 (fun n -> Ops.O.(var "d" 0 1000 * Ops.int n))
          in
          let domains = List.init 4 (fun _ -> Domain.spawn build) in
          let results = List.map Domain.join domains in
          let first = List.hd results in
          List.iter
            (fun r ->
              Array.iteri
                (fun n u -> is_true ~msg:(string_of_int n) (u == first.(n)))
                r)
            results);
    ]

let structure_order =
  group "compare_structure"
    [
      test "ignores tags" (fun () ->
          let a = Ops.int 1 in
          equal int 0
            (Ops.compare_structure a (Ops.rtag ~tag:(String "tagged") a));
          equal int 0
            (Ops.compare_structure (Ops.rtag ~tag:(String "tagged") a) a));
      test "is zero exactly on equal graphs, when untagged" (fun () ->
          let a = Ops.int 1 and b = Ops.int 2 in
          List.iter
            (fun (x, y) -> equal bool (x == y) (Ops.compare_structure x y = 0))
            [
              (a, Ops.int 1);
              (a, b);
              (a, Ops.O.(a + b));
              (Ops.O.(a + b), Ops.O.(b + a));
              (Ops.sink [ a; b ], Ops.sink [ a; b ]);
            ]);
      test "tells apart deep graphs that differ only at the bottom" (fun () ->
          let rec deepen n u =
            if n = 0 then u else deepen (n - 1) (Ops.v ~src:[ u; u ] Op.Add)
          in
          let a = deepen 256 (Ops.int 1) and b = deepen 256 (Ops.int 2) in
          let c = Ops.compare_structure a b in
          not_equal int 0 c;
          equal int (-c) (Ops.compare_structure b a));
      prop "is antisymmetric and transitive"
        (Gen.triple gen_recipe gen_recipe gen_recipe) (fun (r0, r1, r2) ->
          let vars = leaves () in
          let by_structure =
            Testable.with_compare Ops.compare_structure
              (Testable.make ~pp:Ops.pp ~equal:(fun a b ->
                   Ops.compare_structure a b = 0))
          in
          Law.order by_structure (build vars r0, build vars r1, build vars r2));
    ]

let keys =
  group "key"
    [
      prop "is equal for equal graphs" gen_recipe (fun r ->
          equal string
            (Ops.key (build (leaves ()) r))
            (Ops.key (build (leaves ()) r)));
      prop "differs for different graphs" (Gen.pair gen_recipe gen_recipe)
        (fun (r0, r1) ->
          let vars = leaves () in
          let u0 = build vars r0 and u1 = build vars r1 in
          assume (u0 != u1);
          not_equal string (Ops.key u0) (Ops.key u1));
      test "tells apart an operation, a type, an argument and a source"
        (fun () ->
          let x = var "x" 0 4 and y = var "y" 0 4 in
          let a = Ops.O.(x + y) in
          List.iter
            (fun (what, b) ->
              not_equal ~msg:what string (Ops.key a) (Ops.key b))
            [
              ("operation", Ops.O.(x * y));
              ("type", Ops.cast a Int64);
              ("argument", Ops.O.(x + var "y" 0 5));
              ("source", Ops.O.(y + x));
            ];
          equal int 32 (String.length (Ops.key a)));
      test "ignores tags" (fun () ->
          let a = Ops.O.(var "a" 0 4 + Ops.int 1) in
          equal string (Ops.key a) (Ops.key (Ops.rtag ~tag:(String "t") a)));
      test "ignores a call's auxiliary data" (fun () ->
          let call aux = Ops.call ?aux (Ops.sink []) [] in
          equal string
            (Ops.key (call None))
            (Ops.key (call (Some (hcq [ "AMD" ])))));
      test "tells arguments apart" (fun () ->
          not_equal string
            (Ops.key (Ops.sink ~kernel:(Ops.kernel_info ()) []))
            (Ops.key (Ops.sink ~kernel:(Ops.kernel_info ~beam:3 ()) [])));
    ]

(* Data types *)

let dtype_or_raise cell =
  if raised cell then cell else "dtypes." ^ Dtypes.alias (dtype_of_cell cell)

let operand dt =
  if Dtype.equal dt Bool then flag "x"
  else Ops.variable ~dtype:dt "x" (i 0) (i 1)

let data_types =
  group "data types"
    [
      Golden.cases "dtypes_of.golden" ~key:[ "op"; "a"; "b" ] (fun cell ->
          let o = op_of_cell (cell "op") in
          let srcs =
            match (o, cell "b") with
            | _, "-" -> [ operand (dtype_of_cell (cell "a")) ]
            | Op.Where, b ->
                [
                  operand (dtype_of_cell (cell "a"));
                  operand (dtype_of_cell b);
                  operand Float32;
                ]
            | _, b ->
                [
                  operand (dtype_of_cell (cell "a")); operand (dtype_of_cell b);
                ]
          in
          expect dtype dtype_of_cell (cell "dtype") (fun () ->
              Ops.dtype_of o srcs No_arg));
      test "a constant's data type is its literal's" (fun () ->
          equal dtype Bool (Ops.dtype_of Op.Const [] (Const (`Bool true)));
          equal dtype Weak_float (Ops.dtype_of Op.Const [] (Const (f 3.)));
          equal dtype Weak_int (Ops.dtype_of Op.Const [] (Const (i 3)));
          equal dtype Bool (Ops.dtype_of Op.Const [] (Const `Invalid)));
      test "an operation whose type is its argument's needs that argument"
        (fun () ->
          rejects (fun () -> Ops.v Op.Const);
          rejects (fun () -> Ops.v ~src:[ Ops.int 1 ] Op.Cast);
          rejects (fun () -> Ops.v ~src:[ Ops.int 1 ] ~arg:(Shard 3) Op.Bitcast);
          rejects (fun () -> Ops.v Op.Param);
          rejects (fun () -> Ops.v Op.Custom));
      test "a transcendental of Invalid is boolean" (fun () ->
          equal dtype Bool (Ops.dtype (Ops.alu Ops.invalid Op.Sqrt [])));
      test "effects are void, and a call returns its CallInfo's type" (fun () ->
          List.iter
            (fun o ->
              equal ~msg:(str Op.pp o) dtype Void (Ops.dtype_of o [] No_arg))
            Op.
              [
                Sink;
                Linear;
                Program;
                Source;
                Backedge;
                Barrier;
                Group;
                If;
                Endif;
                Noop;
              ];
          equal dtype Int32
            (Ops.dtype
               (Ops.call ~ret_dtype:Int32 (Ops.custom_function "f" []) [])));
      test "promo_dtype is the shared type, or the least upper one" (fun () ->
          equal dtype Float32
            (Ops.promo_dtype [ fvar "x"; fvar ~dtype:Float16 "y" ]);
          equal dtype Int32
            (Ops.promo_dtype [ var ~dtype:Int8 "a" 0 1; var "b" 0 1 ]);
          equal dtype Weak_int (Ops.promo_dtype [ Ops.int 1; Ops.int 2 ]));
      Golden.cases "identities.golden" ~key:[ "op"; "dtype" ] (fun cell ->
          equal const
            (const_of_cell (cell "identity"))
            (Ops.identity_element
               (op_of_cell (cell "op"))
               (dtype_of_cell (cell "dtype"))));
      test "identity_element rejects an operation without one" (fun () ->
          rejects (fun () -> Ops.identity_element Op.Sub Int32);
          rejects (fun () -> Ops.identity_element Op.Where Int32));
    ]

(* Graphs *)

(* A random DAG: node [n] adds two of the earlier ones. *)
let gen_dag =
  Gen.(
    list ~size:(int_range 1 12) (pair nat nat)
    |> with_pp (fun ppf l ->
        Format.pp_print_list
          (fun ppf (a, b) -> Format.fprintf ppf "(%d,%d)" a b)
          ppf l))

let dag edges =
  let nodes = ref [ var "leaf0" 0 1; var "leaf1" 0 1 ] in
  List.iter
    (fun (a, b) ->
      let n = List.length !nodes in
      let pick k = List.nth !nodes (n - 1 - (k mod n)) in
      nodes := !nodes @ [ Ops.v ~src:[ pick a; pick b ] Op.Add ])
    edges;
  Ops.sink !nodes

let index_of l =
  let t = Ops.Tbl.create 16 in
  List.iteri (fun n u -> Ops.Tbl.replace t u n) l;
  Ops.Tbl.find t

let graphs =
  group "graphs"
    [
      prop "toposort lists each node once, after its sources" gen_dag
        (fun edges ->
          let order = Ops.toposort (dag edges) in
          let at = index_of order in
          equal int (List.length order)
            (Ops.Tbl.length
               (let t = Ops.Tbl.create 16 in
                List.iter (fun u -> Ops.Tbl.replace t u ()) order;
                t));
          List.iter
            (fun u ->
              List.iter (fun s -> less int ~than:(at u) (at s)) (Ops.src u))
            order);
      prop "toposort reaches exactly what the sources reach" gen_dag
        (fun edges ->
          let root = dag edges in
          let rec reach acc u =
            if List.memq u acc then acc
            else List.fold_left reach (u :: acc) (Ops.src u)
          in
          equal int
            (List.length (reach [] root))
            (List.length (Ops.toposort root)));
      test "toposort finishes sources left to right, then the node" (fun () ->
          let a = var "a" 0 4 and b = var "b" 0 4 in
          let c = Ops.int 2 in
          equal uops
            [ a; b; Ops.O.(a + b); c; Ops.O.((a + b) * c) ]
            (Ops.toposort Ops.O.((a + b) * c)));
      test "toposort enters only the nodes the gate accepts" (fun () ->
          let a = var "a" 0 4 and b = var "b" 0 4 in
          let s = Ops.O.(a + b) in
          equal uops
            [ Ops.O.(s * s) ]
            (Ops.toposort ~gate:(fun u -> u != s) Ops.O.(s * s));
          equal uops [] (Ops.toposort ~gate:(fun _ -> false) s));
      test "toposort does not enter call bodies without enter_calls" (fun () ->
          let body = Ops.sink [ var "inside" 0 1 ] in
          let arg = var "arg" 0 1 in
          let c = Ops.call body [ arg ] in
          is_true (List.memq body (Ops.toposort c));
          is_false (List.memq body (Ops.toposort ~enter_calls:false c));
          is_true (List.memq arg (Ops.toposort ~enter_calls:false c)));
      test "topovisit applies its function once per node, sources first"
        (fun () ->
          let a = var "a" 0 4 in
          let s = Ops.O.(a + a) in
          let seen = ref [] in
          let cache = Ops.Tbl.create 4 in
          let depth u =
            seen := u :: !seen;
            List.length !seen
          in
          equal int 2 (Ops.topovisit s depth cache);
          equal uops [ a; s ] (List.rev !seen);
          equal int 2 (Ops.topovisit s depth cache);
          equal int 2 (List.length !seen));
      test "backward_slice is the reached nodes without the root or call bodies"
        (fun () ->
          let leaf = Ops.param 97 Int32 in
          let branch = Ops.O.(leaf + Ops.int 3) in
          let root = Ops.sink [ branch; leaf; branch ] in
          equal uops
            (List.filter (fun u -> u != root) (Ops.toposort root))
            (Ops.Nodes.to_list (Ops.backward_slice root));
          equal uops
            (root :: Ops.Nodes.to_list (Ops.backward_slice root))
            (Ops.Nodes.to_list (Ops.backward_slice_with_self root));
          let body = Ops.sink [ var "inside" 0 1 ] in
          is_false (Ops.Nodes.mem body (Ops.backward_slice (Ops.call body []))));
      test
        "op_in_backward_slice_with_self looks at the node and what it reaches"
        (fun () ->
          let e = Ops.O.(var "a" 0 4 + Ops.int 1) in
          is_true (Ops.op_in_backward_slice_with_self e [ Op.Add ]);
          is_true (Ops.op_in_backward_slice_with_self e [ Op.Param; Op.Mul ]);
          is_false (Ops.op_in_backward_slice_with_self e [ Op.Mul ]));
      test "bool_slice is the boolean nodes reached" (fun () ->
          let a = var "a" 0 4 in
          let c = Ops.O.(a < Ops.int 2) in
          let d = Ops.O.(c land flag "p") in
          equal (slist uop Ops.compare)
            [ c; flag "p"; d ]
            (Ops.Nodes.to_list (Ops.bool_slice (Ops.where d a (Ops.int 0))));
          equal int 0 (Ops.Nodes.cardinal (Ops.bool_slice a)));
      test "split_uop is the operands of a tree of one operation" (fun () ->
          let a = var "a" 0 4 and b = var "b" 0 4 and c = var "c" 0 4 in
          equal uops [ a; b; c ] (Ops.split_uop Ops.O.(a + b + c) Op.Add);
          equal uops
            [ a; Ops.O.(b * c) ]
            (Ops.split_uop Ops.O.(a + (b * c)) Op.Add);
          equal uops [ a ] (Ops.split_uop a Op.Mul));
    ]

(* Shapes *)

let t_var () = weak_var "t" 1 10

let shapes =
  group "shapes"
    [
      test "broadcast_shape aligns right and keeps the size that is not 1"
        (fun () ->
          equal shape
            (ints [ 4; 8 ])
            (Ops.broadcast_shape [ ints [ 4; 8 ]; ints [ 4; 1 ] ]);
          equal shape
            (ints [ 4; 8 ])
            (Ops.broadcast_shape [ ints [ 4; 8 ]; ints [ 1; 8 ] ]);
          equal shape
            (ints [ 4; 8 ])
            (Ops.broadcast_shape [ ints [ 4; 8 ]; ints [ 8 ] ]);
          equal shape
            (ints [ 4; 8 ])
            (Ops.broadcast_shape [ ints [ 4; 8 ]; [] ]);
          equal shape (ints [ 0 ])
            (Ops.broadcast_shape [ ints [ 0 ]; ints [ 1 ] ]);
          let t = t_var () in
          equal shape [ Int 1; Sym t ]
            (Ops.broadcast_shape [ [ Int 1; Sym t ]; [ Int 1; Sym t ] ]);
          equal shape [ Sym t ] (Ops.broadcast_shape [ ints [ 1 ]; [ Sym t ] ]));
      test "broadcast_shape rejects two sizes other than 1" (fun () ->
          rejects (fun () -> Ops.broadcast_shape [ ints [ 3 ]; ints [ 2 ] ]);
          rejects (fun () ->
              Ops.broadcast_shape
                [ [ Sym (t_var ()) ]; [ Sym (weak_var "s" 1 10) ] ]));
      test "broadcast_axes is the axes broadcasting adds or expands" (fun () ->
          equal (list int) []
            (Ops.broadcast_axes (ints [ 4; 8 ]) (ints [ 4; 8 ]));
          equal (list int) [ 0 ]
            (Ops.broadcast_axes (ints [ 8 ]) (ints [ 4; 8 ]));
          equal (list int) [ 0; 1 ] (Ops.broadcast_axes [] (ints [ 4; 8 ]));
          equal (list int) [ 0; 2 ]
            (Ops.broadcast_axes (ints [ 3; 1 ]) (ints [ 4; 3; 8 ]));
          equal (list int) []
            (Ops.broadcast_axes (ints [ 1; 8 ]) (ints [ 1; 8 ]));
          rejects (fun () -> Ops.broadcast_axes (ints [ 4; 8 ]) (ints [ 8 ])));
      test "broadcast_axes compares symbolic sizes" (fun () ->
          let t = t_var () in
          equal (list int) []
            (Ops.broadcast_axes [ Sym t; Int 8 ] [ Sym t; Int 8 ]);
          equal (list int) [ 0 ]
            (Ops.broadcast_axes (ints [ 1; 8 ]) [ Sym t; Int 8 ]));
      test "an elementwise operation broadcasts its sources' shapes" (fun () ->
          let row = Ops.expand (Ops.float 1.) (ints [ 4; 8 ]) in
          equal shape
            (ints [ 4; 8 ])
            (Ops.shape Ops.O.(row + Ops.expand (Ops.float 2.) (ints [ 4; 1 ])));
          equal shape
            (ints [ 4; 8 ])
            (Ops.shape Ops.O.(row + Ops.expand (Ops.float 2.) (ints [ 1; 8 ])));
          equal shape
            (ints [ 4; 8 ])
            (Ops.shape Ops.O.(row * Ops.expand (Ops.float 2.) (ints [ 8 ])));
          equal shape (ints [ 4; 8 ]) (Ops.shape Ops.O.(row * float 2.)));
      test "an elementwise operation keeps a symbolic shape" (fun () ->
          let t = t_var () in
          let sym = Ops.expand (Ops.float 1.) [ Int 1; Int 1; Sym t ] in
          equal shape [ Int 1; Int 1; Sym t ] (Ops.shape Ops.O.(sym + sym)));
      test "an elementwise operation rejects sources that do not broadcast"
        (fun () ->
          let a = Ops.param ~shape:(ints [ 2 ]) 812 Float32
          and b = Ops.param ~shape:(ints [ 3 ]) 813 Float32 in
          rejects (fun () -> Ops.shape Ops.O.(a + b)));
      test "shape_opt is None for effects and program structure" (fun () ->
          List.iter
            (fun u -> is_none ~msg:(str Op.pp (Ops.op u)) (Ops.shape_opt u))
            [
              Ops.sink [];
              Ops.group [ Ops.int 1; Ops.int 2 ];
              Ops.v ~arg:(Code { code = "nop"; dtype = Void }) Op.Custom;
              Ops.v ~arg:(Code { code = "nop"; dtype = Void }) Op.Ins;
              Ops.call (Ops.sink []) [];
              Ops.custom_function "f" [];
            ];
          rejects (fun () -> Ops.shape (Ops.sink [])));
      test
        "a typed instruction is a scalar, and a custom node broadcasts its \
         sources" (fun () ->
          equal shape []
            (Ops.shape
               (Ops.v
                  ~arg:(Code { code = "mov"; dtype = Int32 })
                  ~src:[ Ops.int 1 ]
                  Op.Ins));
          let pair = Ops.consts [ i 1; i 2 ] in
          equal shape (ints [ 2 ])
            (Ops.shape
               (Ops.v
                  ~arg:(Code { code = "x"; dtype = Int32 })
                  ~src:[ pair ] Op.Customi)));
      test "a stack prepends its length" (fun () ->
          let vec = Ops.consts [ i 5; i 6; i 7 ] in
          equal shape (ints [ 3 ]) (Ops.shape vec);
          equal shape
            (ints [ 2; 3 ])
            (Ops.shape (Ops.stack [ vec; Ops.consts [ i 8; i 9; i 10 ] ]));
          equal shape [] (Ops.shape (Ops.v Op.Stack));
          equal dtype Void (Ops.dtype (Ops.v Op.Stack)));
      test "an expand prepends its sizes" (fun () ->
          let base = Ops.param ~shape:(ints [ 4; 5 ]) 0 Int32 in
          equal shape
            (ints [ 3; 4; 5 ])
            (Ops.shape (Ops.mop base (Expand (ints [ 3 ]))));
          is_true (Ops.mop base (Expand []) == base));
      test
        "a bitcast rescales the last axis, and rejects a size that does not \
         divide" (fun () ->
          equal shape (ints [ 1 ])
            (Ops.shape
               (Ops.bitcast (Ops.param ~shape:(ints [ 4 ]) 1 Int8) Int32));
          equal shape (ints [ 8 ])
            (Ops.shape
               (Ops.bitcast (Ops.param ~shape:(ints [ 2 ]) 1 Int32) Int8));
          rejects (fun () ->
              Ops.shape
                (Ops.bitcast (Ops.param ~shape:(ints [ 3 ]) 0 Int8) Int32)));
      test "binary code is a vector of its bytes" (fun () ->
          let bin = Ops.v ~arg:(Bytes "code") Op.Binary in
          equal dtype Uint8 (Ops.dtype bin);
          equal shape (ints [ 4 ]) (Ops.shape bin));
      test
        "an index takes its indices' shapes, then the source's remaining axes"
        (fun () ->
          let p = Ops.param ~shape:(ints [ 4; 5; 6 ]) 0 Float32 in
          equal shape (ints [ 5; 6 ]) (Ops.shape (Ops.index p [ Ops.int 1 ]));
          equal shape
            (ints [ 2; 6 ])
            (Ops.shape (Ops.index p [ Ops.int 1; Ops.consts [ i 0; i 1 ] ])));
      test "an index past the source's axes takes only its indices' shapes"
        (fun () ->
          equal shape []
            (Ops.shape (Ops.index (Ops.param 0 Float32) [ Ops.int 0 ]));
          equal shape []
            (Ops.shape
               (Ops.index
                  (Ops.param ~shape:(ints [ 4 ]) 0 Float32)
                  [ Ops.int 0; Ops.int 1 ])));
      test "a stage puts its ranges' sizes in front" (fun () ->
          let r = Ops.range (Int 3) [ 0 ] in
          equal shape
            (ints [ 3; 2 ])
            (Ops.shape (Ops.bufferize (Ops.consts [ f 1.; f 2. ]) [ r ])));
      test "a matrix multiply-accumulate has the accumulator's shape" (fun () ->
          let a = Ops.param ~shape:(ints [ 8 ]) 0 Float16
          and b = Ops.param ~shape:(ints [ 8 ]) 1 Float16 in
          equal shape (ints [ 4 ])
            (Ops.shape
               (Ops.wmma a b
                  ~acc:(Ops.param ~shape:(ints [ 4 ]) 2 Float32)
                  ~dims:(8, 16, 16) ~threads:32)));
      test
        "a reduction drops its leading axes, and rejects more axes than its \
         source has" (fun () ->
          let p = Ops.param ~shape:(ints [ 2; 3; 4 ]) 0 Float32 in
          equal shape
            (ints [ 3; 4 ])
            (Ops.shape
               (Ops.v ~src:[ p ]
                  ~arg:(Reduce { op = Op.Add; num_axes = 1 })
                  Op.Reduce));
          rejects (fun () ->
              Ops.shape
                (Ops.v ~src:[ p ]
                   ~arg:(Reduce { op = Op.Add; num_axes = 4 })
                   Op.Reduce)));
      test "ndim, numel, max_shape and max_numel read the shape" (fun () ->
          let n = weak_var "n" 1 8 in
          let p = Ops.param ~shape:(ints [ 3; 8 ]) 0 Float32 in
          equal int 2 (Ops.ndim p);
          equal (list int) [ 3; 8 ] (Ops.max_shape p);
          equal int 24 (Ops.max_numel p);
          equal (list int) [ 3; 8 ] (Ops.to_max_shape [ Int 3; Sym n ]);
          equal sint (Int 12)
            (Ops.numel (Ops.param ~shape:(ints [ 3; 4 ]) 1 Float32));
          equal sint (Int 1) (Ops.numel (Ops.param 2 Float32)));
      test "max_shape takes a symbolic size's greatest value" (fun () ->
          let n = weak_var "n" 1 8 in
          let p = Ops.param ~shape:[ Int 3; Sym n ] 0 Float32 in
          equal (list int) [ 3; 8 ] (Ops.max_shape p);
          equal int 24 (Ops.max_numel p));
      test "placeholder rejects a size past the largest int" (fun () ->
          rejects (fun () ->
              Ops.placeholder ~slot:0 [ 1 lsl 32; 1 lsl 32 ] Float32));
      test "the shape of a deep graph needs no deep recursion" (fun () ->
          let rec deepen n u =
            if n = 0 then u else deepen (n - 1) Ops.O.(u + u)
          in
          let p = Ops.param ~shape:(ints [ 2; 3 ]) 0 Float32 in
          equal shape (ints [ 2; 3 ]) (Ops.shape (deepen 10_000 p)));
      test "max_numel is 0 when an axis is empty" (fun () ->
          equal int 0
            (Ops.max_numel
               (Ops.placeholder ~slot:0 [ 1 lsl 32; 1 lsl 32; 0 ] Float32)));
      test "broadcast_shape rejects nothing to broadcast" (fun () ->
          rejects (fun () -> Ops.broadcast_shape []));
      test "disallow_broadcast rejects sources of different shapes" (fun () ->
          let a = Ops.param ~shape:(ints [ 2; 3 ]) 0 Float32
          and b = Ops.param ~shape:(ints [ 3 ]) 1 Float32 in
          Helpers.context
            [ B (Helpers.disallow_broadcast, true) ]
            (fun () -> rejects (fun () -> Ops.shape Ops.O.(a + b)));
          equal shape (ints [ 2; 3 ]) (Ops.shape Ops.O.(a + b)));
      test "a node passing its source through keeps its shape" (fun () ->
          let p =
            Ops.param ~device:(Single "CPU") ~shape:(ints [ 2; 3 ]) 0 Float32
          in
          List.iter
            (fun u ->
              equal
                ~msg:(str Op.pp (Ops.op u))
                shape
                (ints [ 2; 3 ])
                (Ops.shape u))
            [
              Ops.after p [ Ops.sink [] ];
              Ops.copy_to_device p (Single "CUDA");
              Ops.v ~src:[ p ] Op.Detach;
              Ops.v ~src:[ p ] Op.Contiguous_backward;
              Ops.v ~src:[ p ] Op.Noop;
            ];
          is_none (Ops.shape_opt (Ops.v Op.Noop)));
      test "a bitcast of a scalar or to an equal size keeps the shape"
        (fun () ->
          equal shape [] (Ops.shape (Ops.bitcast (Ops.param 0 Int32) Float32));
          equal shape (ints [ 4 ])
            (Ops.shape
               (Ops.bitcast (Ops.param ~shape:(ints [ 4 ]) 0 Int32) Float32));
          is_none
            (Ops.shape_opt
               (Ops.v ~src:[ Ops.sink [] ] ~arg:(Dtype Int32) Op.Bitcast)));
      test "a reshape of a no-op has its argument's shape" (fun () ->
          equal shape
            (ints [ 2; 3 ])
            (Ops.shape
               (Ops.v
                  ~src:
                    [
                      Ops.v Op.Noop;
                      Ops.v ~src:[ Ops.int 2; Ops.int 3 ] Op.Stack;
                    ]
                  Op.Reshape)));
      test "an axis out of range is rejected" (fun () ->
          let p = Ops.param ~shape:(ints [ 2; 1 ]) 0 Float32 in
          rejects (fun () -> Ops.squeeze ~axis:2 p);
          rejects (fun () -> Ops.squeeze ~axis:(-3) p);
          rejects (fun () -> Ops.flatten ~start:5 p));
      test "a movement of a node without a shape is rejected" (fun () ->
          rejects (fun () ->
              Ops.shape (Ops.v ~src:[ Ops.sink []; Ops.int 1 ] Op.Expand)));
      test "a shrink past the largest int is rejected, not wrapped" (fun () ->
          let p = Ops.param ~shape:(ints [ 2; 3 ]) 0 Float32 in
          rejects (fun () ->
              Ops.shape
                (Ops.mop p (Shrink [ (Int max_int, Int 2); (Int 0, Int 3) ]))));
      test "sint_to_uop is the literal, or the node itself" (fun () ->
          equal uop (Ops.int 3) (Ops.sint_to_uop (Int 3));
          equal uop (Ops.int ~dtype:Int32 3)
            (Ops.sint_to_uop ~dtype:Int32 (Int 3));
          let n = weak_var "n" 1 8 in
          equal uop n (Ops.sint_to_uop (Sym n)));
      test "a movement checks its argument against its source's shape"
        (fun () ->
          let p = Ops.param ~shape:(ints [ 2; 3 ]) 0 Float32 in
          rejects (fun () -> Ops.shape (Ops.mop p (Pad [ (Int 0, Int 2) ])));
          rejects (fun () -> Ops.shape (Ops.mop p (Shrink [ (Int 0, Int 2) ])));
          rejects (fun () ->
              Ops.shape (Ops.mop p (Pad [ (Int (-1), Int 3); (Int 0, Int 3) ])));
          rejects (fun () ->
              Ops.shape
                (Ops.mop p (Shrink [ (Int 0, Int (-1)); (Int 0, Int 3) ])));
          rejects (fun () -> Ops.shape (Ops.mop p (Permute [ 0; 0 ])));
          rejects (fun () -> Ops.shape (Ops.mop p (Permute [ 0 ])));
          rejects (fun () -> Ops.shape (Ops.mop p (Reshape (ints [ 5 ]))));
          rejects (fun () -> Ops.shape (Ops.mop p (Reshape (ints [ -6; -1 ]))));
          rejects (fun () ->
              Ops.shape (Ops.mop p (Shrink [ (Int 1, Int 2); (Int 0, Int 3) ])));
          rejects (fun () ->
              Ops.shape (Ops.mop p (Pad [ (Int 3, Int 2); (Int 0, Int 3) ])));
          rejects (fun () -> Ops.shape (Ops.mop p (Flip [ true ]))));
      test "a symbolic reshape is accepted unless its sizes provably differ"
        (fun () ->
          let n = weak_var "n" 1 8 in
          let p = Ops.param ~shape:[ Sym n ] 0 Float32 in
          equal shape [ Sym n; Int 1 ]
            (Ops.shape (Ops.mop p (Reshape [ Sym n; Int 1 ]))));
    ]

(* Ranges *)

let list_ranges u = Ops.Nodes.to_list (Ops.ranges u)

let ranges =
  group "ranges"
    [
      test
        "range_start is the first range source of the operations that end \
         ranges" (fun () ->
          List.iter
            (fun (o, expected) ->
              equal ~msg:(str Op.pp o) (option int) expected (Ops.range_start o))
            Op.
              [
                (Stage, Some 1);
                (Reduce, Some 1);
                (End, Some 1);
                (Call, Some 1);
                (Linear, Some 0);
                (Add, None);
                (After, None);
              ]);
      test "an end closes its ranges, and an after ordered on it too" (fun () ->
          let r = Ops.range (Int 10) [ 0 ] in
          let c = Ops.O.(r + int 1) in
          is_true (Ops.Nodes.mem r (Ops.ranges c));
          let e = Ops.end_ (Ops.v Op.Noop) [ r ] in
          is_false (Ops.Nodes.mem r (Ops.ranges e));
          is_false (Ops.Nodes.mem r (Ops.ranges (Ops.after c [ e ]))));
      test "ranges lists the node first, then its sources' ranges" (fun () ->
          let r0 = Ops.range (Int 4) [ 0 ] in
          let r1 = Ops.range ~src:[ r0 ] (Int 4) [ 1 ] in
          equal uops [ r1; r0 ] (list_ranges r1);
          equal int 2 (Ops.Nodes.cardinal (Ops.ranges r1));
          equal uops [ r0; r1 ]
            (Ops.Nodes.fold (fun u acc -> u :: acc) (Ops.ranges r1) []));
      test "a range runs inside itself" (fun () ->
          let r = Ops.range (Int 10) [ 0 ] in
          equal uops [ r ] (list_ranges r));
      test "a call to an external function keeps its arguments' ranges"
        (fun () ->
          let r = Ops.range ~dtype:Int32 (Int 4) [ 0 ] in
          let c =
            Ops.call ~ret_dtype:Int32
              (Ops.custom_function "external" [ Ops.int ~dtype:Uint64 0 ])
              [ Ops.O.(r + int 1) ]
          in
          equal uops [ r ] (list_ranges c));
      test "a backedge closes its loop and keeps the condition's other ranges"
        (fun () ->
          let outer = Ops.range (Int 4) [ 0 ] and inner = Ops.loop 1 in
          let e =
            Ops.backedge (Ops.int 1) ~loop:inner ~cond:Ops.O.(outer < int 2)
          in
          equal uops [ outer ] (list_ranges e);
          equal uops [ outer ]
            (list_ranges (Ops.after Ops.O.(outer + int 1) [ e ])));
      test "an after of a barrier over an end closes the ended range" (fun () ->
          let range axis = Ops.range (Int 4) [ axis ] in
          let first = range 1 and closed = range 2 and last = range 3 in
          let barrier = Ops.barrier (Ops.end_ closed [ closed ]) [] in
          let root =
            Ops.after Ops.O.(first + closed + last) [ barrier; barrier ]
          in
          equal uops [ first; last ] (list_ranges root));
      test "a call to a function without arguments ends its range arguments"
        (fun () ->
          let r = Ops.range (Int 4) [ 0 ] in
          equal uops []
            (list_ranges (Ops.call (Ops.custom_function "f" []) [ r ]));
          equal uops [ r ]
            (list_ranges
               (Ops.call (Ops.custom_function "f" [ Ops.int 0 ]) [ r ])));
      test "an end closes its range only, not the ranges it runs inside"
        (fun () ->
          let outer = Ops.range (Int 4) [ 0 ] in
          let inner = Ops.range ~src:[ outer ] (Int 4) [ 1 ] in
          equal uops [ outer ]
            (list_ranges (Ops.end_ Ops.O.(outer + inner) [ inner ])));
      test "a linear program closes the ranges it lays out" (fun () ->
          let r = Ops.range (Int 4) [ 0 ] in
          is_false (Ops.Nodes.mem r (Ops.ranges (Ops.v ~src:[ r ] Op.Linear))));
      test "a barrier closes what its sources close" (fun () ->
          let r = Ops.range (Int 4) [ 0 ] in
          let ended = Ops.end_ Ops.O.(r + int 1) [ r ] in
          equal uops [ r ] (Ops.ended_ranges (Ops.barrier ended [])));
      test "ended_ranges of an unshard is its sharding ranges" (fun () ->
          let d = Ops.range ~axis_type:Device (Int 2) [ -1 ] in
          let u =
            Ops.unshard ~ranges:[ d ]
              (Ops.param ~shape:(ints [ 4 ]) 0 Float32)
              [ 0 ]
          in
          equal uops [ d ] (Ops.ended_ranges u));
      test "axis_id and axis_type read a range and reject anything else"
        (fun () ->
          let r = Ops.range ~axis_type:Reduce (Int 4) [ 1; 0 ] in
          equal (list int) [ 1; 0 ] (Ops.axis_id r);
          is_true (Ops.axis_type r = Reduce);
          rejects (fun () -> Ops.axis_id (Ops.int 1));
          rejects (fun () -> Ops.axis_type (Ops.int 1)));
      test
        "range_str writes the identity with underscores and m for a negative \
         part" (fun () ->
          equal string "0_1" (Ops.range_str (Ops.range (Int 4) [ 0; 1 ]));
          equal string "m1"
            (Ops.range_str (Ops.range ~axis_type:Device (Int 4) [ -1 ]));
          equal string "2_m3" (Ops.range_str (Ops.range (Int 4) [ 2; -3 ])));
      test "range_str paints in the axis colour on request" (fun () ->
          let r = Ops.range ~axis_type:Reduce (Int 4) [ 3 ] in
          equal string (Helpers.colored Red "3") (Ops.range_str ~color:true r));
      test "multirange_str sorts by argument and pads to printed columns"
        (fun () ->
          let r0 = Ops.range ~axis_type:Global (Int 4) [ 0; 1 ]
          and r1 = Ops.range ~axis_type:Upcast (Int 4) [ 0; 0 ] in
          equal string "0_0,0_1" (Ops.multirange_str [ r0; r1 ]);
          equal string "0_0,0_1   " (Ops.multirange_str ~pad:10 [ r0; r1 ]);
          equal int 10
            (Helpers.ansilen
               (Ops.multirange_str ~color:true ~pad:10 [ r0; r1 ])));
    ]

let groups =
  [
    axis_type_group;
    devices;
    arguments;
    printing;
    tags;
    hcq_calls;
    identity;
    structure_order;
    keys;
    data_types;
    graphs;
    shapes;
    ranges;
  ]
