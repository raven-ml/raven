(* Nodes: axis types, arguments, identity, data types, graphs, shapes and
   ranges. *)

open Windtrap
open Tolk
open Common

let str pp x = Format.asprintf "%a" pp x

(* Axis types *)

let axis_types =
  Ops.Axis_type.[ Global; Local; Warp; Weak; Loop; Upcast; Placeholder ]

let axis_type = Testable.make ~pp:Ops.Axis_type.pp ~equal:Ops.Axis_type.equal
let show_axis_type a = str Ops.Axis_type.pp a

let axis_type_of_cell s =
  Result.get_ok (Ops.Axis_type.of_string (after "AxisType." s))

(* A colour as axis_types.golden writes it: bright ones in upper case. *)
let color_name : Helpers.color -> string = function
  | Green -> "green"
  | Blue -> "blue"
  | Cyan -> "cyan"
  | Bright_cyan -> "CYAN"
  | Bright_white -> "WHITE"
  | Red -> "red"
  | Yellow -> "yellow"
  | Magenta -> "magenta"
  | _ -> "another colour"

(* [named ~cell f a] is that [f a] is the golden's [cell], or raises where
   tinygrad raises. *)
let named ~cell name_of f a =
  if raised cell then rejects (fun () -> f a)
  else equal string cell (name_of (f a))

let axis_type_group =
  group "Axis_type"
    [
      group "names each axis type and orders it as axis_types.golden states"
        [
          Golden.cases "axis_types.golden" (fun cell ->
              let a = axis_type_of_cell (cell "axis_type") in
              equal string (cell "axis_type") (show_axis_type a);
              named ~cell:(cell "letter") Fun.id Ops.Axis_type.letter a;
              named ~cell:(cell "color") color_name Ops.Axis_type.color a;
              equal ~msg:"axis types before it" int
                (int_of_string (cell "nesting"))
                (List.length
                   (List.filter
                      (fun b -> Ops.Axis_type.compare b a < 0)
                      axis_types)));
        ];
      cases "equal holds of an axis type and itself alone" ~id:show_axis_type
        axis_types (fun a ->
          equal (list axis_type) [ a ]
            (List.filter (Ops.Axis_type.equal a) axis_types));
      cases "compare orders axis types by their declaration order"
        ~id:show_axis_type axis_types (fun a ->
          let i =
            Option.get (List.find_index (Ops.Axis_type.equal a) axis_types)
          in
          let by sign =
            List.filter (fun b -> sign (Ops.Axis_type.compare b a)) axis_types
          in
          equal
            (triple (list axis_type) (list axis_type) (list axis_type))
            ( List.filteri (fun j _ -> j < i) axis_types,
              [ a ],
              List.filteri (fun j _ -> j > i) axis_types )
            (by (fun c -> c < 0), by (fun c -> c = 0), by (fun c -> c > 0)));
      cases "of_string reads back the name pp formats, without its prefix"
        ~id:show_axis_type axis_types (fun a ->
          equal (result axis_type string) (Ok a)
            (Ops.Axis_type.of_string (after "AxisType." (show_axis_type a))));
      cases "of_string of anything but a name is an error that names it"
        ~id:(Printf.sprintf "%S")
        [ "weak"; "Weak"; "AxisType.WEAK"; " WEAK"; "THREAD" ] (fun s ->
          contains ~sub:s (require_error (Ops.Axis_type.of_string s)));
      test "of_string of the empty string is an error" (fun () ->
          is_error ~pp:Ops.Axis_type.pp (Ops.Axis_type.of_string ""));
    ]

(* Devices *)

let pp_devices =
  group "pp_device"
    [
      cases "writes the name as pp_arg writes a string"
        ~id:(fun (name, _) -> Printf.sprintf "%S" name)
        [
          ("CPU", "'CPU'");
          ("it's", "\"it's\"");
          ("a\"b", "'a\"b'");
          ("it's \"x\"", "'it\\'s \"x\"'");
          ("a\\b", "'a\\\\b'");
          ("a\nb\tc\rd", "'a\\nb\\tc\\rd'");
          ("\001\127", "'\\x01\\x7f'");
          ("\xC3\xA9", "'\xC3\xA9'");
        ]
        (fun (name, literal) -> equal string literal (str Ops.pp_device name));
    ]

let disk_devices =
  group "is_disk_device"
    [
      test "is a name that is DISK, in any case, up to a colon" (fun () ->
          let names =
            [
              "DISK";
              "disk";
              "DiSk:1";
              "DISK:a:b";
              "DISK:";
              "DISKS";
              "DISK1";
              "CPU";
              "";
              ":DISK";
              "AMD:DISK";
            ]
          in
          equal (list string)
            [ "DISK"; "disk"; "DiSk:1"; "DISK:a:b"; "DISK:" ]
            (List.filter Ops.is_disk_device names));
    ]

(* Arguments *)

(* A parameter's argument, compared field by field: the defaults hold no node or
   float. *)
let param_w =
  Testable.make ~pp:(fun ppf p -> Ops.pp_arg ppf (Param p)) ~equal:( = )

let param_rule =
  Gen.triple
    (Gen.of_list ~pp:(Testable.pp dtype) Dtype.all)
    (Gen.int_range (-1) 17)
    (Gen.of_list ~pp:Format.pp_print_int [ 0; 1; 2; 3; 4; 8; 16; 32 ])

(* The phase and alignment [param_arg] keeps, or [None] where it raises. *)
let kept ~phase ~align dt =
  match Ops.param_arg ~slot:0 ~phase ~align dt with
  | p -> Some (p.phase, p.align)
  | exception Invalid_argument _ -> None

let params =
  group "param_arg"
    [
      test "defaults to global storage with no flags, phase 0 and alignment 16"
        (fun () ->
          equal param_w
            {
              slot = 3;
              dtype = Float32;
              vmin_vmax = None;
              name = None;
              addrspace = Global;
              device = None;
              volatile = false;
              bind_on_realize = false;
              bound = None;
              phase = 0;
              align = 16;
            }
            (Ops.param_arg ~slot:3 Float32));
      prop
        ~count:(List.length Dtype.all * 19 * 8)
        "takes an alignment that is a power of two up to 16, and a phase below \
         it that is a multiple of the element's size or of the alignment, \
         whichever is less"
        param_rule
        (fun (dt, phase, align) ->
          let allowed =
            List.mem align [ 1; 2; 4; 8; 16 ]
            && 0 <= phase && phase < align
            && phase mod Int.min (Dtype.itemsize dt) align = 0
          in
          equal
            (option (pair int int))
            (if allowed then Some (phase, align) else None)
            (kept ~phase ~align dt));
    ]

let estimates =
  group "pp_estimates"
    [
      test "writes the record as a call" (fun () ->
          equal string "Estimates(ops=0, lds=0, mem=0)"
            (str Ops.pp_estimates { ops = Int 0; lds = Int 0; mem = Int 0 }));
    ]

let pp_kernel_info ppf (k : Ops.kernel_info) =
  Format.fprintf ppf
    "{ name = %S; applied_opts = %d; opts_to_apply = %s; estimates = %s; beam \
     = %d; optimized = %b; kernel = %b; split = %s }"
    k.name
    (List.length k.applied_opts)
    (match k.opts_to_apply with
    | None -> "None"
    | Some l -> string_of_int (List.length l))
    (match k.estimates with None -> "None" | Some e -> str Ops.pp_estimates e)
    k.beam k.optimized k.kernel
    (match k.split with None -> "None" | Some _ -> "Some _")

(* A kernel's argument, compared field by field: the defaults hold no node. *)
let kernel_info_w = Testable.make ~pp:pp_kernel_info ~equal:( = )

let kernel_infos =
  group "kernel_info"
    [
      test "defaults to a launched kernel named test, not optimised" (fun () ->
          equal kernel_info_w
            {
              name = "test";
              applied_opts = [];
              opts_to_apply = None;
              estimates = None;
              beam = 0;
              optimized = false;
              kernel = true;
              split = None;
            }
            (Ops.kernel_info ()));
    ]

let function_names =
  group "function_name"
    [
      prop "is the name as an identifier"
        (Gen.string_of ~size:(Gen.int_range 0 8)
           (Gen.of_list
              [
                'a';
                'Z';
                '9';
                '_';
                ' ';
                '-';
                '.';
                '\027';
                '[';
                'm';
                '\xC3';
                '\xA9';
              ]))
        ~examples:
          [
            ""; "9abc"; "kernel name"; "a-b.c"; "\xC3\xA9"; "\027[31mred\027[0m";
          ]
        (fun name ->
          equal string
            (Helpers.to_function_name name)
            (Ops.function_name (Ops.kernel_info ~name ())));
    ]

let equal_args =
  group "equal_arg"
    [
      (* Two arguments redrawn in one field are equal exactly when the field's
         two values are, by an equality of their own. A custom function's
         argument is on the body its call calls. The rarest field's differing
         draw has a share of 0.4%: 6000 cases miss it with a probability under
         10^-9. *)
      prop ~count:6000 "is equality of every field" Args.redrawn
        (fun (field, u, v, same) ->
          List.iter
            (fun f -> cover f (String.equal f field && not same))
            Args.field_names;
          let holder w =
            if String.starts_with ~prefix:"custom." field then Ops.nth w 0
            else w
          in
          let a0 = Ops.arg (holder u) and a1 = Ops.arg (holder v) in
          if same then equal arg a0 a1 else not_equal arg a0 a1);
      cases
        "compares constants by their bits, nodes physically, and kinds apart"
        ~id:(fun (what, _, _, _) -> what)
        (let a = var "a" 0 1 in
         let program u =
           Ops.Program
             { (Shape.program_info_of_sink (Ops.sink [])) with params = [ u ] }
         in
         Ops.
           [
             ( "NaNs of other bits",
               Const (f Float.nan),
               Const (f (-.Float.nan)),
               false );
             ("0. and -0.", Const (f 0.), Const (f (-0.)), false);
             ("1 and true", Const (i 1), Const (`Bool true), false);
             ("a device and a string", Device "CPU", String "CPU", false);
             ("programs of one node", program a, program (var "a" 0 1), true);
             ("programs of two nodes", program a, program (var "b" 0 1), false);
           ])
        (fun (_, a0, a1, same) ->
          if same then equal arg a0 a1 else not_equal arg a0 a1);
    ]

let reprs = lazy (Goldens.reprs ())

(* [nines n] is the integer of [n] nines. *)
let nines n = Bigint.pred (Bigint.pow (Bigint.of_int 10) n)

let pp_args =
  group "pp_arg"
    [
      group "writes each argument as reprs.golden states"
        [
          Golden.cases "reprs.golden"
            ~differs:
              [
                ("D80", [ "name=kernel"; "name=kernel with opts" ]);
                ("D77", [ "name=bufferize"; "name=bufferize without device" ]);
              ]
            (fun cell ->
              let u = List.assoc (cell "name") (Lazy.force reprs) in
              equal string (cell "arg") (str Ops.pp_arg (Ops.arg u)));
        ];
      test "every argument this suite builds has a row of reprs.golden"
        (fun () ->
          equal
            (slist string String.compare)
            (List.map (fun cell -> cell "name") (Golden.rows "reprs.golden"))
            (List.map fst (Lazy.force reprs)));
      cases "writes a parameter's fields that differ from their defaults"
        ~id:fst
        [
          ( "phase",
            ( Ops.param_arg ~slot:0 ~phase:4 Float32,
              "ParamArg(0, dtypes.f32, phase=4)" ) );
          ( "align",
            ( Ops.param_arg ~slot:0 ~align:4 Float32,
              "ParamArg(0, dtypes.f32, align=4)" ) );
          ("none", (Ops.param_arg ~slot:0 Float32, "ParamArg(0, dtypes.f32)"));
        ]
        (fun (_, (p, text)) -> equal string text (str Ops.pp_arg (Param p)));
      test "writes a byte of bytes past 0x7f as x and two hexadecimal digits"
        (fun () ->
          equal string "b'\\x7f\\x80\\xff'"
            (str Ops.pp_arg (Bytes "\x7f\x80\xff")));
      cases "rejects an integer of more than 4300 digits" ~id:string_of_int
        [ 4300; 4301 ] (fun digits ->
          let write () = str Ops.pp_arg (Const (`Int (nines digits))) in
          if digits > 4300 then rejects write
          else equal int digits (String.length (write ())));
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
                    Div;
                    Mod;
                    Truncdiv;
                    Truncmod;
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

exception Interrupted

(* [interrupted k f] is [f ()] with [Interrupted] raised at its [k]th
   allocation, where Ctrl-C's [Sys.Break] or a test runner's timeout can land.
   Is [None] when it was raised, else [Some n] with [n] the allocations [f]
   made. *)
let interrupted k f =
  let armed = ref false and n = ref 0 in
  let alloc _ =
    if !armed then begin
      incr n;
      if !n = k then begin
        armed := false;
        raise Interrupted
      end
    end;
    None
  in
  let tracker =
    { Gc.Memprof.null_tracker with alloc_minor = alloc; alloc_major = alloc }
  in
  let profile = Gc.Memprof.start ~sampling_rate:1. ~callstack_size:0 tracker in
  Fun.protect
    ~finally:(fun () ->
      armed := false;
      Gc.Memprof.stop ();
      Gc.Memprof.discard profile)
    (fun () ->
      match
        armed := true;
        f ()
      with
      | _ ->
          armed := false;
          Some !n
      | exception Interrupted -> None)

(* Hash-consing under churn, against keys. A world counts its collections: a
   node's reference is its key and the collections before it. Nodes dropped
   before a build can still read live when it walks their slots, and die by the
   next collection, so a key built again after it walks dead slots before
   reaching its live node. Two keys, comparisons drawn twice as often as the
   other calls, and programs of 50 calls, so that about half the cases compare
   one key built on each side of a collection: 40 cases all miss it with a
   probability under 0.55^40, 4e-11. *)
let churn_node k = Ops.v ~arg:(String (Printf.sprintf "churn key %d" k)) Op.Noop

let[@inline never] [@local never] drop n =
  for i = 0 to n - 1 do
    ignore
      (Sys.opaque_identity
         (Ops.v ~arg:(String (Printf.sprintf "churn dropped %d" i)) Op.Noop))
  done

let churn =
  let world = abstract "w" and node = abstract "u" in
  let same (ka, ea) (kb, eb) =
    cover "one key built on each side of a collection" (ka = kb && ea <> eb);
    cover "two keys" (ka <> kb);
    ka = kb
  in
  stateful ~count:40 ~steps:50
    "a node is any equal live node, across collections and churn"
    [
      command "world" (Gen.unit @-> makes world) (fun () -> ref 0) ignore;
      command "build"
        (world ^-> Gen.int_range 0 1 @-> makes node)
        (fun w k -> (k, !w))
        (fun () k -> churn_node k);
      command "same" (node ^-> node ^-> returns bool) same ( == );
      command "same" (node ^-> node ^-> returns bool) same ( == );
      command "drop" (Gen.int_range 0 2000 @-> returns unit) ignore drop;
      command "collect"
        (world ^-> returns unit)
        (fun w -> incr w)
        (fun () -> Gc.full_major ());
    ]

(* Nodes built on two domains, by key: a table's nodes are its own, never built
   before it, so the branches race to build each first. *)
type built = { table : int; nodes : Ops.t list Atomic.t array }

let tables_made = Atomic.make 0

let new_built () =
  {
    table = Atomic.fetch_and_add tables_made 1;
    nodes = Array.init 4 (fun _ -> Atomic.make []);
  }

let built_node t k =
  Ops.v ~arg:(String (Printf.sprintf "table %d key %d" t.table k)) Op.Noop

let build_recorded t k =
  let u = built_node t k in
  let rec push () =
    let l = Atomic.get t.nodes.(k) in
    if not (Atomic.compare_and_set t.nodes.(k) l (u :: l)) then push ()
  in
  push ()

(* [one_recorded t k] is the node of [k] built now and the nodes recorded for
   it, which [one] judges on the test's domain. *)
let one_recorded t k = (built_node t k, Atomic.get t.nodes.(k))

let one () _ = function
  | Ok (u, recorded) ->
      cover "two builds recorded for one key" (List.length recorded >= 2);
      equal uops (List.map (fun _ -> u) recorded) recorded
  | Error e -> raise e

(* [one_built vars r] is [r] built twice over the same leaves. *)
let one_built r = (build (leaves ()) r, build (leaves ()) r)
let sign c = Int.compare c 0

(* The variables of [prefix], so that two calls build disjoint graphs. *)
let leaves_of prefix = [| var (prefix ^ "0") (-8) 8; var (prefix ^ "1") 1 5 |]

(* Leaves no earlier call made, so that no node over them is alive before. *)
let fresh_leaves =
  let made = ref 0 in
  fun () ->
    incr made;
    leaves_of (Printf.sprintf "fresh%d_" !made)

(* A constant past the 4300 digits Bigint writes. *)
let unwritable =
  Ops.const (`Int (Bigint.pred (Bigint.pow (Bigint.of_int 10) 4301)))

let identity =
  group "Identity"
    [
      churn;
      stateful ~domains:2 "nodes built on two domains at once are one node"
        (let built = abstract "t" in
         [
           command "table" (Gen.unit @-> makes built) ignore new_built;
           command "build"
             (built ^-> Gen.int_range 0 3 @-> returns unit)
             (fun () _ -> ())
             build_recorded;
           command "one"
             (built ^-> Gen.int_range 0 3 @-> judges (pair uop uops))
             one one_recorded;
         ]);
      prop "building a graph twice gives the same node" gen_recipe (fun r ->
          let u0, u1 = one_built r in
          equal uop u0 u1);
      prop "structurally different graphs are different nodes"
        (Gen.pair gen_recipe gen_recipe) (fun (r0, r1) ->
          let vars = leaves () in
          let normal r = Graph.to_string (build vars r) in
          let u0 = build vars r0 and u1 = build vars r1 in
          if String.equal (normal r0) (normal r1) then (
            cover "one structure" true;
            equal uop u0 u1)
          else (
            cover "two structures" true;
            not_equal uop u0 u1));
      cases "constants of other bits or kinds are different nodes"
        ~id:(fun (what, _, _) -> what)
        [
          ("0. and -0.", Ops.float 0., Ops.float (-0.));
          ( "NaNs of other payloads",
            Ops.float Float.nan,
            Ops.float (Int64.float_of_bits 0x7FF4_0000_0000_0000L) );
          ("NaNs of other signs", Ops.float Float.nan, Ops.float (-.Float.nan));
          ("1 and true", Ops.int 1, Ops.bool true);
          ("0 and 0.", Ops.int 0, Ops.float 0.);
        ]
        (fun (_, u0, u1) -> not_equal uop u0 u1);
      test "a NaN constant is one node with its own bits" (fun () ->
          equal uop (Ops.float Float.nan) (Ops.float Float.nan));
      cases "v keeps the order of a commutative operation's operands"
        ~id:(fun (what, _, _) -> what)
        (let i1 = weak_var "idx1" 0 3 and i2 = weak_var "idx2" 0 3 in
         [
           ("a sum", Ops.O.(i1 + i2), Ops.O.(i2 + i1));
           ("a product", Ops.O.(i1 * i2), Ops.O.(i2 * i1));
         ])
        (fun (_, u0, u1) -> not_equal uop u0 u1);
      test "a node nothing references is collected" (fun () ->
          let cell = Weak.create 2 in
          let[@inline never] [@local never] build () =
            let r = Ops.range ~axis_type:Weak (Int 9127) [ 9127 ] in
            let c = Ops.O.(r < int 4) in
            ignore
              ( Ops.ranges c,
                Ops.vmin r,
                Shape.shape r,
                Ops.backward_slice ~calls:Skip c,
                Ops.device c,
                Ops.addrspace c );
            Weak.set cell 0 (Some r);
            Weak.set cell 1 (Some (Ops.end_ c [ r ]))
          in
          build ();
          Gc.full_major ();
          Gc.full_major ();
          equal (list string) []
            (List.filter_map
               (fun (what, i) -> if Weak.check cell i then Some what else None)
               [ ("a range", 0); ("an end", 1) ]));
      test "a node built again after its first was collected is one node"
        (fun () ->
          let node k =
            Ops.v ~arg:(String (Printf.sprintf "again %d" k)) Op.Noop
          in
          let[@inline never] [@local never] build () =
            ignore (List.init 1000 node)
          in
          build ();
          Gc.full_major ();
          Gc.full_major ();
          let first = List.init 1000 node in
          let again = List.init 1000 node in
          equal uops first again);
    ]

(* The literals that tell whether one holds its type's value: -1, the type's
   bounds and past them, the width's power and its neighbours, its negation, and
   one no integer type holds. *)
let width_literals dt =
  let lo, hi = Dtypes.int_bounds dt in
  let wide = Bigint.shift_left Bigint.one (Dtype.bitsize dt) in
  Bigint.
    [
      minus_one;
      zero;
      one;
      lo;
      pred lo;
      hi;
      succ hi;
      pred wide;
      wide;
      succ wide;
      neg wide;
      shift_left one 80;
    ]

let literal z = Ops.const (`Int z)
let literal_value u i = Ops.value (Ops.nth u i)
let at_type dt z = (Dtype.truncate dt (`Int z) :> Dtype.const)

let construct =
  let reading_ops =
    Op.
      [
        Add;
        Sub;
        Mul;
        And;
        Or;
        Xor;
        Max;
        Div;
        Mod;
        Truncdiv;
        Truncmod;
        Cmplt;
        Cmpne;
        Cmpeq;
      ]
  in
  let on_ints f = List.iter f Dtype.ints in
  let variable dt =
    let lo, hi = Dtypes.int_bounds dt in
    Ops.variable ~dtype:dt "x" (`Int lo) (`Int hi)
  in
  group "v"
    [
      test "an operation whose type is its argument's needs that argument"
        (fun () ->
          rejects (fun () -> Ops.v Op.Const);
          rejects (fun () -> Ops.v ~src:[ Ops.int 1 ] Op.Cast);
          rejects (fun () ->
              Ops.v ~src:[ Ops.int 1 ] ~arg:(Device "CPU") Op.Bitcast);
          rejects (fun () -> Ops.v Op.Param);
          rejects (fun () -> Ops.v Op.Ins));
      test "a literal holds its value at the operation's integer type"
        (fun () ->
          on_ints (fun dt ->
              let x = variable dt in
              List.iter
                (fun z ->
                  let want = at_type dt z in
                  let holds name u i =
                    equal
                      ~msg:
                        (Format.asprintf "%s %a %s" name Dtype.pp dt
                           (Bigint.to_string z))
                      const want (literal_value u i)
                  in
                  List.iter
                    (fun op ->
                      let name = Format.asprintf "%a" Op.pp op in
                      holds name (Ops.v op ~src:[ x; literal z ]) 1;
                      holds name (Ops.v op ~src:[ literal z; x ]) 0)
                    reading_ops;
                  let cond =
                    Ops.variable ~dtype:Bool "c" (`Bool false) (`Bool true)
                  in
                  holds "a selection"
                    (Ops.v Op.Where ~src:[ cond; literal z; x ])
                    1;
                  holds "a multiply-add"
                    (Ops.v Op.Mulacc ~src:[ x; x; literal z ])
                    2;
                  holds "a cast"
                    (Ops.v Op.Cast ~src:[ literal z ] ~arg:(Dtype dt))
                    0)
                (width_literals dt)));
      test "a 64-bit key of a random bits operation holds its value" (fun () ->
          let k = variable Uint64 in
          equal const
            (`Int (Bigint.of_int 1))
            (literal_value
               (Ops.v Op.Threefry
                  ~src:
                    [
                      k; literal (Bigint.shift_left Bigint.one 64 |> Bigint.succ);
                    ])
               1));
      test "a shift's count and an exponent keep their value" (fun () ->
          on_ints (fun dt ->
              let x = variable dt in
              List.iter
                (fun z ->
                  if Bigint.sign z >= 0 then
                    List.iter
                      (fun op ->
                        equal const (`Int z)
                          (literal_value (Ops.v op ~src:[ x; literal z ]) 1))
                      Op.[ Shl; Shr; Pow ])
                (width_literals dt)));
      test "a literal base of a power is read at the power's type" (fun () ->
          let y = variable Uint8 in
          equal const
            (`Int (Bigint.of_int 44))
            (literal_value
               (Ops.v Op.Pow ~src:[ literal (Bigint.of_int 300); y ])
               0));
      test "a weak expression, a float operation and no peer keep their values"
        (fun () ->
          let x = variable Uint8 in
          let w =
            Ops.variable "w" (`Int Bigint.zero) (`Int (Bigint.of_int 1000))
          in
          let expr = Ops.O.(w + Ops.int 300) in
          equal uop expr (Ops.nth (Ops.v Op.Add ~src:[ x; expr ]) 1);
          let f = Ops.variable ~dtype:Float32 "f" (`Float 0.) (`Float 1.) in
          equal const
            (`Int (Bigint.of_int 300))
            (literal_value
               (Ops.v Op.Add ~src:[ f; literal (Bigint.of_int 300) ])
               1);
          equal const
            (`Int (Bigint.of_int 300))
            (literal_value
               (Ops.v Op.Add
                  ~src:
                    [ literal (Bigint.of_int 300); literal (Bigint.of_int 7) ])
               0));
      test "an exception at any allocation of v keeps the nodes built"
        (fun () ->
          let node tag = Ops.v ~arg:(String tag) Op.Noop in
          let tag name n = Printf.sprintf "%s %d" name n in
          let built = ref [] in
          let build tag = built := (tag, node tag) :: !built in
          List.iter build (List.init 256 (tag "kept"));
          (* Building a batch of new nodes raises at its [k]th allocation, for
             each [k] up to the batch's last. The batches fill the shards, so
             some raises land as a shard's buckets or table grow and move the
             nodes held in [built]. *)
          let rec from k =
            if k > 10_000 then fail "a batch makes over 10000 allocations";
            let batch =
              List.init 32 (fun j -> tag "interrupted" ((k * 32) + j))
            in
            let made = interrupted k (fun () -> List.iter build batch) in
            match made with Some n when n < k -> () | _ -> from (k + 1)
          in
          from 1;
          let tags, nodes = List.split !built in
          equal uops nodes (List.map node tags));
    ]

let replacing =
  group "replace"
    [
      test "is the node itself when nothing changes" (fun () ->
          let a = Ops.O.(var "a" 0 4 + Ops.int 1) in
          equal uop a (Ops.replace a);
          equal uop a
            (Ops.replace ~op:Op.Add ~src:(Ops.src a) ~arg:Ops.No_arg a));
      test "is the node with the fields given" (fun () ->
          let a = Ops.O.(var "a" 0 4 + Ops.int 1) in
          equal uop
            (Ops.alu (Ops.nth a 0) Op.Mul [ Ops.nth a 1 ])
            (Ops.replace ~op:Op.Mul a));
    ]

let equality =
  group "equal"
    [
      prop "is an equivalence" (Gen.pair gen_recipe gen_recipe) (fun (r0, r1) ->
          let vars = leaves () in
          Law.equivalence uop (build vars r0, build vars r1));
    ]

let ordering =
  group "compare"
    [
      prop "is a total order that agrees with equal"
        (Gen.triple gen_recipe gen_recipe gen_recipe) (fun (r0, r1, r2) ->
          let vars = leaves () in
          Law.order uop (build vars r0, build vars r1, build vars r2));
    ]

let hashing =
  group "hash"
    [
      prop "agrees with equal on live nodes" gen_recipe (fun r ->
          let u0, u1 = one_built r in
          equal int (Ops.hash u0) (Ops.hash u1));
    ]

let printing =
  group "pp"
    [
      group "writes each node as pretty.golden states"
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
        ];
    ]

(* Functions that raise as [pp_arg] does where they write an argument: each
   compares the constant with another one. *)
let writers =
  [
    ("pp", fun u -> ignore (str Ops.pp u));
    ("compare_structure", fun u -> ignore (Ops.compare_structure u (Ops.int 1)));
    ( "structure_ranks",
      fun u -> ignore (Ops.structure_ranks (Ops.sink [ u; Ops.int 1 ]) u) );
    ("key", fun u -> ignore (Ops.key u));
  ]

let raising =
  group "pp_arg's raise"
    [
      cases "raises where they write an integer pp_arg rejects" ~id:fst writers
        (fun (_, write) -> rejects (fun () -> write unwritable));
    ]

(* A few expressions and a chain of one operation by a constant over the first
   of them: each step of the chain falls next to the last in the order. *)
let gen_ranked =
  Gen.(
    pair
      (list ~size:(int_range 1 4) gen_recipe)
      (triple
         (of_list ~pp:Op.pp Op.[ Add; Mul; Max ])
         (int_range (-2) 2) (int_range 0 400)))

let structure_order =
  group "compare_structure"
    [
      cases "is zero exactly on equal graphs"
        ~id:(fun (what, _, _) -> what)
        (let a = Ops.int 1 and b = Ops.int 2 in
         [
           ("one constant", a, Ops.int 1);
           ("two constants", a, b);
           ("a constant and a sum", a, Ops.O.(a + b));
           ("sums of swapped operands", Ops.O.(a + b), Ops.O.(b + a));
           ("one sink", Ops.sink [ a; b ], Ops.sink [ a; b ]);
         ])
        (fun (_, x, y) ->
          if x == y then equal int 0 (Ops.compare_structure x y)
          else not_equal int 0 (Ops.compare_structure x y));
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
      (* Two graphs over fresh variables, built in one order and then in the
         other: [compare]'s order may follow the build, this one may not. *)
      prop "does not depend on the order nodes were built in"
        (Gen.pair gen_recipe gen_recipe) (fun (r0, r1) ->
          let a = fresh_leaves () in
          let u0 = build a r0 in
          let u1 = build a r1 in
          let b = fresh_leaves () in
          let w1 = build b r1 in
          let w0 = build b r0 in
          cover "two graphs" (u0 != u1);
          equal int
            (sign (Ops.compare_structure u0 u1))
            (sign (Ops.compare_structure w0 w1)));
    ]

let ranks =
  group "structure_ranks"
    [
      prop "numbers a graph's nodes in compare_structure's order" gen_ranked
        (fun (recipes, (o, k, length)) ->
          cover "a chain of one operation past 128 steps" (length > 128);
          let vars = leaves () in
          let rec chain n =
            if n = 0 then List.hd recipes else Bin (o, chain (n - 1), Konst k)
          in
          let sink =
            Ops.sink (List.map (build vars) (chain length :: recipes))
          in
          let rank = Ops.structure_ranks sink in
          let order a b = Int.compare (rank a) (rank b) in
          let sorted =
            List.stable_sort Ops.compare_structure
              (Ops.toposort ~calls:Enter sink)
          in
          (* The sign of each step along the order, by both. *)
          let rec steps = function
            | a :: (b :: _ as rest) ->
                (sign (Ops.compare_structure a b), sign (order a b))
                :: steps rest
            | _ -> []
          in
          let steps = steps sorted in
          equal (list int) (List.map fst steps) (List.map snd steps));
      test "numbers NaNs of different bits alike" (fun () ->
          let nan bits = Ops.float (Int64.float_of_bits bits) in
          let a = nan 0x7FF8_0000_0000_0000L
          and b = nan 0x7FF8_0000_0000_0001L in
          let x = fvar "x" in
          let rec chain n u =
            if n = 0 then u else chain (n - 1) Ops.O.(u + x)
          in
          let sa = chain 3 a and sb = chain 3 b in
          let rank = Ops.structure_ranks (Ops.sink [ sa; sb ]) in
          let order a b = Int.compare (rank a) (rank b) in
          not_equal uop a b;
          equal int 0 (order sa sb);
          equal int 0 (order a b);
          equal int 0 (order Ops.O.(a + x) Ops.O.(b + x)));
      test "raises Not_found on a node the graph does not reach" (fun () ->
          let rank = Ops.structure_ranks Ops.O.(var "a" 0 4 + int 1) in
          raises Not_found (fun () -> rank (var "b" 0 4)));
    ]

(* Short texts of quotes, escapes and bytes past ASCII, often equal. *)
let texts =
  Gen.string_of ~size:(Gen.int_range 0 2)
    (Gen.of_list [ 'a'; '\''; '\000'; '\xff' ])

let keys =
  group "key"
    [
      (* A collection per case: ten graphs share one. *)
      prop ~count:20
        "is equal for equal graphs, built before a collection and after"
        (Gen.list ~size:(Gen.int_range 1 10) gen_recipe)
        (fun rs ->
          let[@inline never] [@local never] keys () =
            List.map (fun r -> Ops.key (build (leaves_of "collected") r)) rs
          in
          let before = keys () in
          Gc.full_major ();
          equal (list string) before (keys ()));
      prop "differs for different graphs" (Gen.pair gen_recipe gen_recipe)
        (fun (r0, r1) ->
          let vars = leaves () in
          let u0 = build vars r0 and u1 = build vars r1 in
          assume (u0 != u1);
          not_equal string (Ops.key u0) (Ops.key u1));
      cases "tells apart an operation, a type, an argument and a source" ~id:fst
        (let x = var "x" 0 4 and y = var "y" 0 4 in
         [
           ("operation", Ops.O.(x * y));
           ("type", Ops.cast Ops.O.(x + y) Int64);
           ("argument", Ops.O.(x + var "y" 0 5));
           ("source", Ops.O.(y + x));
         ])
        (fun (_, b) ->
          let x = var "x" 0 4 and y = var "y" 0 4 in
          not_equal string (Ops.key Ops.O.(x + y)) (Ops.key b));
      prop "tells apart sources and binaries by their text"
        (Gen.pair texts texts)
        ~examples:[ ("a", "a") ]
        (fun (t0, t1) ->
          cover "equal texts" (String.equal t0 t1);
          List.iter
            (fun (what, node) ->
              let k0 = Ops.key (node t0) and k1 = Ops.key (node t1) in
              if String.equal t0 t1 then equal ~msg:what string k0 k1
              else not_equal ~msg:what string k0 k1)
            [
              ("source", fun t -> Ops.v Source ~arg:(String t));
              ("binary", fun t -> Ops.v Binary ~arg:(Bytes t));
            ]);
      test "tells apart NaN constants of different bits" (fun () ->
          let nan bits = Ops.float (Int64.float_of_bits bits) in
          let quiet = nan 0x7FF8_0000_0000_0000L in
          List.iter
            (fun (what, b) ->
              not_equal ~msg:what string (Ops.key quiet) (Ops.key (nan b)))
            [
              ("signalling", 0x7FF0_0000_2000_0000L);
              ("payload", 0x7FF8_0000_0000_0001L);
              ("sign", 0xFFF8_0000_0000_0000L);
            ];
          equal string (Ops.key quiet) (Ops.key (nan 0x7FF8_0000_0000_0000L)));
      (* A node redrawn in one field of its argument keys apart, but for a
         call's write marks, which the key leaves out. *)
      (* The rarest field's differing draw, stage.device's, has a share of
         about 0.4%: 6000 cases all miss it with a probability of about
         0.996^6000, 3.6e-11. *)
      prop ~count:6000
        "two nodes of one key are one node, but for a call's marks"
        Args.neighbours (fun (field, u, v) ->
          List.iter
            (fun f -> cover f (String.equal f field && u != v))
            Args.field_names;
          if u == v || String.equal field "call.write" then
            equal string (Ops.key u) (Ops.key v)
          else not_equal string (Ops.key u) (Ops.key v));
      prop ~count:3000 "two nodes drawn apart share a key only as one node"
        Args.pairs (fun (u, v) ->
          let unmarked w =
            match Ops.arg w with
            | Call c -> Ops.replace w ~arg:(Call { c with write = [] })
            | _ -> w
          in
          if unmarked u == unmarked v then (
            cover "one node, but for a call's marks" true;
            equal string (Ops.key u) (Ops.key v))
          else (
            cover "two nodes of one kind" true;
            not_equal string (Ops.key u) (Ops.key v)));
      test "tells apart arguments a repr prints alike" (fun () ->
          let open Args in
          let stage keep =
            Ops.v ~src:[ buffer ]
              ~arg:(Bufferize { device = None; addrspace = Global; keep })
              Stage
          in
          let program ?(params = []) ?(interface = "") ?(indices = "") () =
            let one = [ Ops.Int 1; Int 1; Int 1 ] in
            let target : Helpers.Target.t =
              { device = "CPU"; renderer = ""; arch = ""; interface; indices }
            in
            Ops.v
              ~arg:
                (Program
                   {
                     global_size = one;
                     local_size = one;
                     params;
                     globals = [];
                     outs = [];
                     ins = [];
                     target;
                   })
              Program
          in
          let estimated u =
            Ops.sink
              ~kernel:
                (Ops.kernel_info
                   ~estimates:{ ops = Sym u; lds = Int 0; mem = Int 0 }
                   ())
              [ x ]
          in
          List.iter
            (fun (what, u, v) ->
              not_equal ~msg:what string (Ops.key u) (Ops.key v))
            [
              ("a parameter's NaNs", bounded quiet, bounded signalling);
              ("what a stage keeps", stage Whole, stage Argument);
              ( "a program's parameter",
                program ~params:[ bounded quiet ] (),
                program ~params:[ bounded signalling ] () );
              ( "a kernel's estimate",
                estimated (bounded quiet),
                estimated (bounded signalling) );
              ( "a program's target",
                program ~interface:"a:b" (),
                program ~interface:"a" ~indices:"b" () );
              ( "where a program's target's fields end",
                program ~interface:"ab" (),
                program ~interface:"a" ~indices:"b" () );
            ]);
    ]

(* [u] with each node [arg] gives an argument taking it. *)
let with_args arg u =
  Ops.substitute ~calls:Enter ~pass:Once u
    (List.filter_map
       (fun n -> Option.map (fun a -> (n, Ops.replace n ~arg:a)) (arg n))
       (Ops.toposort ~calls:Enter u))

let keys_with =
  group "key_with"
    [
      prop "is the key of the graph with the arguments given" gen_recipe
        (fun r ->
          let vars = leaves () in
          let u = build vars r in
          let renamed = Ops.arg (var "w" (-8) 8) in
          let arg n = if n == vars.(0) then Some renamed else None in
          cover "renames" (List.memq vars.(0) (Ops.toposort ~calls:Enter u));
          equal string (Ops.key (with_args arg u)) (Ops.key_with ~arg u));
      test "is the key of a NaN constant of the bits given" (fun () ->
          let nan bits = Ops.float (Int64.float_of_bits bits) in
          let quiet = nan 0x7FF8_0000_0000_0000L in
          let u = Ops.O.(fvar "x" + quiet) in
          let payload = Ops.arg (nan 0x7FF8_0000_0000_0001L) in
          let arg n = if n == quiet then Some payload else None in
          equal string (Ops.key (with_args arg u)) (Ops.key_with ~arg u);
          not_equal string (Ops.key u) (Ops.key_with ~arg u));
      test "is the key of a source and a binary of the texts given" (fun () ->
          let source = Ops.v Source ~arg:(String "a")
          and binary = Ops.v Binary ~arg:(Bytes "\x7fELF") in
          let u = Ops.sink [ source; binary ] in
          let arg n =
            if n == source then Some (Ops.String "b'")
            else if n == binary then Some (Ops.Bytes "\x00\xff")
            else None
          in
          equal string (Ops.key (with_args arg u)) (Ops.key_with ~arg u);
          not_equal string (Ops.key u) (Ops.key_with ~arg u));
      test "is the key when no argument is given" (fun () ->
          let u = Ops.O.(var "x" 0 4 + Ops.int 1) in
          equal string (Ops.key u) (Ops.key_with ~arg:(fun _ -> None) u));
      prop "calls arg once on each node" gen_recipe (fun r ->
          let u = build (leaves ()) r in
          let calls = Ops.Tbl.create 16 in
          let arg n =
            Ops.Tbl.replace calls n
              (1 + Option.value ~default:0 (Ops.Tbl.find_opt calls n));
            None
          in
          ignore (Ops.key_with ~arg u);
          let nodes = Ops.toposort ~calls:Enter u in
          equal (list int)
            (List.map (fun _ -> 1) nodes)
            (List.map
               (fun n -> Option.value ~default:0 (Ops.Tbl.find_opt calls n))
               nodes));
    ]

let tables =
  group "Tbl"
    [
      test "is keyed by identity" (fun () ->
          let t = Ops.Tbl.create 4 in
          Ops.Tbl.replace t (var "a" 0 4) 1;
          Ops.Tbl.replace t (var "a" 0 4) 2;
          Ops.Tbl.replace t (var "a" 0 5) 3;
          equal int 2 (Ops.Tbl.length t);
          equal (option int) (Some 2) (Ops.Tbl.find_opt t (var "a" 0 4)));
    ]

(* Data types *)

let operand dt =
  if Dtype.equal dt Bool then flag "x"
  else Ops.variable ~dtype:dt "x" (i 0) (i 1)

let transcendentals = Op.[ Exp2; Log2; Sin; Sqrt; Recip ]

let data_types =
  group "dtype_of"
    [
      group "derives each operation's type as dtypes_of.golden states"
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
                      operand (dtype_of_cell (cell "a"));
                      operand (dtype_of_cell b);
                    ]
              in
              let derive () = Ops.dtype_of o srcs No_arg in
              if raised (cell "dtype") then rejects derive
              else equal dtype (dtype_of_cell (cell "dtype")) (derive ()));
        ];
      (* The graph format's table of constants: a constant's type follows from
         its kind. *)
      cases "is a constant's weak type of its kind, Bool for Invalid"
        ~id:(fun (c, _) -> str Ops.pp_arg (Const c))
        [
          (`Bool true, Dtype.Bool);
          (f 3., Weak_float);
          (i 3, Weak_int);
          (`Invalid, Bool);
        ]
        (fun (c, dt) -> equal dtype dt (Ops.dtype_of Op.Const [] (Const c)));
      cases "is Bool for a transcendental of invalid" ~id:(str Op.pp)
        transcendentals (fun o ->
          equal dtype Bool (Ops.dtype (Ops.alu Ops.invalid o [])));
      cases "is Void for an effect" ~id:(str Op.pp)
        Op.
          [
            Store;
            Sink;
            Linear;
            Program;
            Source;
            Backedge;
            Barrier;
            If;
            Endif;
            Noop;
          ]
        (fun o -> equal dtype Void (Ops.dtype_of o [] No_arg));
    ]

let promoting =
  group "promo_dtype"
    [
      prop "is the type its nodes share, or their least upper bound"
        (Gen.list ~size:(Gen.int_range 1 3) Dtypes.promotable)
        (fun dts ->
          let shared =
            match dts with
            | d :: rest when List.for_all (Dtype.equal d) rest -> Some d
            | _ -> None
          in
          cover "a shared type" (Option.is_some shared);
          cover "several types" (Option.is_none shared);
          equal dtype
            (Option.value shared ~default:(Dtype.least_upper dts))
            (Ops.promo_dtype (List.map operand dts)));
    ]

(* An operation that has an identity, a data type and a value of the type its
   accumulator takes. *)
let accumulated =
  Gen.(
    with_pp
      (fun ppf (o, dt, x) ->
        Format.fprintf ppf "(%a, %a, %a)" Op.pp o (Testable.pp dtype) dt
          (Testable.pp value) x)
      (let* o = of_list ~pp:Op.pp Op.[ Add; Mul; Max ] in
       let* dt =
         of_list ~pp:(Testable.pp dtype)
           (Dtype.Weak_int :: Dtype.Weak_float :: Dtype.all)
       in
       let+ x = Dtypes.value_of (Dtype.strong dt) in
       (o, dt, x)))

let is_nan = function `Float x -> Float.is_nan x | _ -> false
let is_neg_zero = function `Float x -> x = 0. && Float.sign_bit x | _ -> false

(* [op e x] and [op x e] are [x], but for the two exceptions the .mli states:
   [0. + -0.] is [0.], and [Max] of the identity and a NaN gives either. *)
let neutral (o, dt, x) =
  cover "-0. under Add" (o = Op.Add && is_neg_zero x);
  cover "a NaN under Max" (o = Op.Max && is_nan x);
  cover "a NaN under Add or Mul" (o <> Op.Max && is_nan x);
  let e = Ops.identity_element o dt and x = (x :> Dtype.const) in
  let apply a b = Ops.exec_alu o (Dtype.strong dt) [ a; b ] in
  List.iter
    (fun r ->
      if o = Op.Add && is_neg_zero x then equal const (`Float 0.) r
      else if o = Op.Max && is_nan x then mem const r [ e; x ]
      else equal const x r)
    [ apply e x; apply x e ]

let identities =
  group "identity_element"
    [
      group "is each operation's identity as identities.golden states"
        [
          Golden.cases "identities.golden" ~key:[ "op"; "dtype" ]
            ~differs:[ ("D165", [ "op=Ops.MAX dtype=dtypes.weakint" ]) ]
            (fun cell ->
              equal const
                (const_of_cell (cell "identity"))
                (Ops.identity_element
                   (op_of_cell (cell "op"))
                   (dtype_of_cell (cell "dtype"))));
        ];
      prop
        "is neutral for its operation on every value of the accumulator's type"
        ~examples:
          [
            (Op.Add, Dtype.Float32, `Float (-0.));
            (Op.Max, Dtype.Float64, `Float Float.nan);
            (Op.Mul, Dtype.Float16, `Float Float.nan);
          ]
        accumulated neutral;
      cases "raises for an operation other than Add, Mul and Max"
        ~id:(str Op.pp)
        (List.filter
           (fun o -> not (List.mem o Op.[ Add; Mul; Max ]))
           (Op.Set.to_list Op.Set.all))
        (fun o -> rejects (fun () -> Ops.identity_element o Int32));
    ]

(* Graphs *)

(* A random graph over two variables: each step adds two of the nodes before it,
   or calls a body that multiplies two of them on the second. A call has no
   value, so no later step reads it. *)
let gen_graph =
  Gen.(
    list ~size:(int_range 1 12) (triple nat nat bool)
    |> with_pp (fun ppf l ->
        Format.pp_print_list
          (fun ppf (a, b, call) -> Format.fprintf ppf "(%d,%d,%b)" a b call)
          ppf l))

let graph steps =
  let nodes = ref [ var "leaf0" 0 1; var "leaf1" 0 1 ] and calls = ref [] in
  List.iter
    (fun (a, b, call) ->
      let n = List.length !nodes in
      let pick k = List.nth !nodes (n - 1 - (k mod n)) in
      if call then
        calls :=
          Ops.call (Ops.sink [ Ops.O.(pick a * pick b) ]) [ pick b ] :: !calls
      else nodes := !nodes @ [ Ops.v ~src:[ pick a; pick b ] Op.Add ])
    steps;
  Ops.sink (!nodes @ List.rev !calls)

let walked_srcs ~calls u =
  match (calls, Ops.op u) with
  | Ops.Skip, Op.Call -> List.tl (Ops.src u)
  | _ -> Ops.src u

(* The order ops.mli states: a depth-first walk of the sources, in order,
   through the nodes [gate] accepts, each listed when the walk finishes it. *)
let reference_toposort ?(gate = fun _ -> true) ~calls root =
  let seen = Ops.Tbl.create 16 and order = ref [] in
  let rec visit u =
    if (not (Ops.Tbl.mem seen u)) && gate u then begin
      Ops.Tbl.replace seen u ();
      List.iter visit (walked_srcs ~calls u);
      order := u :: !order
    end
  in
  visit root;
  List.rev !order

(* [x] is [u], or a source of a node a walk from [u] enters through nodes other
   than [x] that [gate] accepts. *)
let reference_reaches ?(gate = fun _ -> true) ~calls u x =
  u == x
  || List.exists
       (fun n -> List.memq x (walked_srcs ~calls n))
       (reference_toposort ~gate:(fun n -> n != x && gate n) ~calls u)

let has_call root = List.exists (fun u -> Ops.op u = Op.Call) (Ops.src root)

let pp_calls ppf (c : Ops.calls) =
  Format.pp_print_string ppf (match c with Enter -> "Enter" | Skip -> "Skip")

let gen_walk =
  Gen.(triple gen_graph bool (of_list ~pp:pp_calls Ops.[ Enter; Skip ]))

(* The nodes of the graph of [steps], each a root, and a gate that refuses every
   third of them if [gated]. *)
let walks steps gated =
  let root = graph steps in
  let nodes = reference_toposort ~calls:Enter root in
  let at = Ops.Tbl.create 16 in
  List.iteri (fun n u -> Ops.Tbl.replace at u n) nodes;
  let gate u = (not gated) || Ops.Tbl.find at u mod 3 <> 1 in
  (root, nodes, gate)

let call_covers root calls =
  cover "a call's body left out" (calls = Ops.Skip && has_call root);
  cover "a call's body walked" (calls = Ops.Enter && has_call root)

let walk_errors =
  group "Graphs"
    [
      test
        "a walk an exception ends at any allocation leaves later walks their \
         results" (fun () ->
          (* A chain of 40 sums outgrows a walk's first rows and pairs, so some
             raises land as they grow. A walk the exception ends gives its
             arrays back, and later walks take them: a walk in another's gate
             holds its arrays while the other holds its own, so an array given
             back twice would be both walks'. *)
          let leaf = var "leaf" 0 4 and other = var "other" 0 4 in
          let rec chain n u =
            if n = 0 then u else chain (n - 1) Ops.O.(u + Ops.int n)
          in
          let root = chain 40 leaf in
          let walk ?gate u = Ops.toposort ?gate ~calls:Enter u in
          let rewrite () =
            Ops.substitute ~calls:Enter ~pass:Fixed_point root [ (leaf, other) ]
          in
          let order = walk root and renamed = rewrite () in
          let below = Ops.Tbl.create 64 in
          List.iter (fun u -> Ops.Tbl.replace below u (walk u)) order;
          let gate u =
            equal uops (Ops.Tbl.find below u) (walk u);
            true
          in
          let rec from k =
            if k > 100_000 then fail "the walks make over 100000 allocations";
            let made =
              interrupted k (fun () ->
                  ignore (walk root);
                  ignore (rewrite ()))
            in
            equal uops order (walk ~gate root);
            equal uop renamed (rewrite ());
            match made with Some n when n < k -> () | _ -> from (k + 1)
          in
          from 1);
    ]

let toposorts =
  group "toposort"
    [
      prop
        "is the nodes a depth-first walk of the sources finishes, through \
         those the gate accepts"
        gen_walk (fun (steps, gated, calls) ->
          let root, nodes, gate = walks steps gated in
          call_covers root calls;
          cover "a gate that refuses nodes" gated;
          List.iter
            (fun u ->
              equal uops
                (reference_toposort ~gate ~calls u)
                (Ops.toposort ~gate ~calls u))
            nodes);
    ]

let topovisits =
  group "topovisit"
    [
      (* A cache that holds every fourth node beforehand, if [cached]. *)
      prop
        "applies its function to the nodes a walk enters outside the cache, \
         each after its sources, once" (Gen.pair gen_graph Gen.bool)
        (fun (steps, cached) ->
          let root = graph steps in
          let nodes = reference_toposort ~calls:Enter root in
          let cache = Ops.Tbl.create 16 in
          List.iteri
            (fun n u ->
              if cached && n mod 4 = 0 then Ops.Tbl.replace cache u (-1))
            nodes;
          let outside u = not (Ops.Tbl.mem cache u) in
          let expected = reference_toposort ~gate:outside ~calls:Enter root in
          cover "a cached node" cached;
          let visited = ref [] in
          let f u =
            visited := u :: !visited;
            List.length !visited
          in
          let result = Ops.topovisit root f cache in
          equal uops expected (List.rev !visited);
          equal
            (list (pair uop int))
            (List.mapi (fun n u -> (u, n + 1)) expected)
            (List.map (fun u -> (u, Ops.Tbl.find cache u)) expected);
          equal int (Ops.Tbl.find cache root) result;
          equal int result (Ops.topovisit root f cache);
          equal int (List.length expected) (List.length !visited));
    ]

let slices =
  group "backward_slice"
    [
      prop "is toposort's nodes without the root" gen_graph (fun steps ->
          let root = graph steps in
          cover "a call" (has_call root);
          (* Both walks of each node in turn: a slice kept for one must not
             answer the other. *)
          List.iter
            (fun u ->
              List.iter
                (fun calls ->
                  equal uops
                    (List.filter
                       (fun n -> n != u)
                       (reference_toposort ~calls u))
                    (Ops.Nodes.to_list (Ops.backward_slice ~calls u)))
                Ops.[ Skip; Enter; Skip ])
            (reference_toposort ~calls:Enter root));
    ]

let slices_with_self =
  group "backward_slice_with_self"
    [
      prop "is the root, then its backward_slice" gen_walk
        (fun (steps, _, calls) ->
          let root, nodes, _ = walks steps false in
          call_covers root calls;
          List.iter
            (fun u ->
              equal uops
                (u :: Ops.Nodes.to_list (Ops.backward_slice ~calls u))
                (Ops.Nodes.to_list (Ops.backward_slice_with_self ~calls u)))
            nodes);
    ]

let ops_in_slices =
  group "op_in_backward_slice_with_self"
    [
      prop "is whether an operation of the list is one of the slice's" gen_walk
        (fun (steps, _, calls) ->
          let root, nodes, _ = walks steps false in
          call_covers root calls;
          let lists =
            Op.[ [ Add ]; [ Mul ]; [ Param ]; [ Sink ]; [ Mul; Call ] ]
          in
          List.iter
            (fun u ->
              let ops_of n = Ops.op n in
              let slice = reference_toposort ~calls u in
              equal
                (list (list op))
                (List.filter
                   (fun ops ->
                     List.exists (fun n -> List.mem (ops_of n) ops) slice)
                   lists)
                (List.filter
                   (Ops.op_in_backward_slice_with_self ~calls u)
                   lists))
            nodes);
    ]

let reaching =
  group "reaches"
    [
      prop
        "is whether a node is the root or a source of a node the walk enters, \
         through nodes other than it that the gate accepts"
        gen_walk (fun (steps, gated, calls) ->
          let root, nodes, gate = walks steps gated in
          call_covers root calls;
          cover "a gate that refuses nodes" gated;
          List.iter
            (fun u ->
              equal uops
                (List.filter (reference_reaches ~gate ~calls u) nodes)
                (List.filter (Ops.reaches ~gate ~calls u) nodes))
            nodes);
    ]

let splitting =
  group "split_uop"
    [
      cases "is the operands of the tree of the operation, left to right"
        ~id:(fun (what, _, _, _) -> what)
        (let a = var "a" 0 4 and b = var "b" 0 4 and c = var "c" 0 4 in
         [
           ("a chain of sums", Ops.O.(a + b + c), Op.Add, [ a; b; c ]);
           ( "a sum of a product",
             Ops.O.(a + (b * c)),
             Op.Add,
             [ a; Ops.O.(b * c) ] );
           ("a sum of sums", Ops.O.(a + b + (c + a)), Op.Add, [ a; b; c; a ]);
           ("another operation", a, Op.Mul, [ a ]);
         ])
        (fun (_, u, o, expected) -> equal uops expected (Ops.split_uop u o));
    ]

(* Shapes *)

let broadcasting =
  let t = weak_var "t" 1 10 in
  group "broadcast_shape"
    [
      cases
        "aligns shapes to the right, each axis the size of its sources not 1"
        ~id:(fun (what, _, _) -> what)
        [
          ("one shape", [ ints [ 2; 3 ] ], ints [ 2; 3 ]);
          ("a 1 of the second", [ ints [ 4; 8 ]; ints [ 4; 1 ] ], ints [ 4; 8 ]);
          ("a 1 of the first", [ ints [ 4; 1 ]; ints [ 1; 8 ] ], ints [ 4; 8 ]);
          ("a shorter shape", [ ints [ 4; 8 ]; ints [ 8 ] ], ints [ 4; 8 ]);
          ("the empty shape", [ ints [ 4; 8 ]; [] ], ints [ 4; 8 ]);
          ("an axis of 0", [ ints [ 0 ]; ints [ 1 ] ], ints [ 0 ]);
          ( "three shapes",
            [ ints [ 3; 1; 1 ]; ints [ 1; 4; 1 ]; ints [ 5 ] ],
            ints [ 3; 4; 5 ] );
          ("all 1", [ ints [ 1; 1 ]; ints [ 1 ] ], ints [ 1; 1 ]);
          ( "a symbolic size twice",
            [ [ Int 1; Sym t ]; [ Int 1; Sym t ] ],
            [ Int 1; Sym t ] );
          ("a symbolic size against 1", [ ints [ 1 ]; [ Sym t ] ], [ Sym t ]);
        ]
        (fun (_, shapes, expected) ->
          equal shape expected (Ops.broadcast_shape shapes));
      cases "raises on an axis of two sizes other than 1"
        ~id:(fun (what, _) -> what)
        [
          ("3 and 2", [ ints [ 3 ]; ints [ 2 ] ]);
          ("3 and 0", [ ints [ 3; 1 ]; ints [ 0; 1 ] ]);
          ("two symbolic sizes", [ [ Sym t ]; [ Sym (weak_var "s" 1 10) ] ]);
        ]
        (fun (_, shapes) -> rejects (fun () -> Ops.broadcast_shape shapes));
      test "of no shape is the empty shape" (fun () ->
          equal shape [] (Ops.broadcast_shape []));
    ]

let max_shapes =
  group "to_max_shape"
    [
      test "replaces each symbolic size by its greatest value" (fun () ->
          let t = weak_var "t" 1 10 in
          equal (list int) [ 3; 10; 20; 0 ]
            (Ops.to_max_shape [ Int 3; Sym t; Sym Ops.O.(t * int 2); Int 0 ]));
    ]

let sints =
  group "sint_to_uop"
    [
      test "is an integer's literal, of the type given" (fun () ->
          equal uop (Ops.int 3) (Ops.sint_to_uop (Int 3));
          equal uop (Ops.int ~dtype:Int32 3)
            (Ops.sint_to_uop ~dtype:Int32 (Int 3)));
      test "is a symbolic size's node" (fun () ->
          let n = weak_var "n" 1 8 in
          equal uop n (Ops.sint_to_uop (Sym n)));
    ]

(* Ranges *)

let list_ranges u = Ops.Nodes.to_list (Ops.ranges u)
let weak_range ?dtype axis = Ops.range ~axis_type:Weak ?dtype (Int 4) [ axis ]

let range_starts =
  group "range_start"
    [
      cases "is 1 for Stage, Reduce and End, 0 for Linear, None otherwise"
        ~id:(str Op.pp) (Op.Set.to_list Op.Set.all) (fun o ->
          let expected =
            match o with
            | Stage | Reduce | End -> Some 1
            | Linear -> Some 0
            | _ -> None
          in
          equal (option int) expected (Ops.range_start o));
    ]

let ended =
  let r = weak_range 0 and outer = weak_range 1 and loop = Ops.loop 2 in
  let ended = Ops.end_ Ops.O.(r + int 1) [ r ] in
  group "ended_ranges"
    [
      cases "is the ranges a node closes"
        ~id:(fun (what, _, _) -> what)
        [
          ( "an end, its ranges",
            Ops.end_ Ops.O.(r + outer) [ r; outer ],
            [ r; outer ] );
          ( "a reduction, its ranges",
            Ops.reduce Ops.O.(r + int 1) Op.Add [ r ],
            [ r ] );
          ("a stage, its ranges", Ops.bufferize Ops.O.(r + int 1) [ r ], [ r ]);
          ("a linear program, its sources", Ops.v ~src:[ r ] Op.Linear, [ r ]);
          ("a barrier, what its sources close", Ops.barrier ended [], [ r ]);
          ( "an after, what its effects close",
            Ops.after Ops.O.(outer + int 1) [ ended ],
            [ r ] );
          ( "a backedge, its loop",
            Ops.backedge (Ops.int 1) ~loop ~cond:Ops.O.(outer < int 2),
            [ loop ] );
          ( "a call, none",
            Ops.call (Ops.custom_function "f" []) [ Ops.O.(r + int 1) ],
            [] );
          ("a sum, none", Ops.O.(r + int 1), []);
        ]
        (fun (_, u, expected) -> equal uops expected (Ops.ended_ranges u));
    ]

let ranges =
  let range axis = weak_range axis in
  group "ranges"
    [
      cases
        "is the node if a range, then those its sources run inside, minus \
         those it closes"
        ~id:(fun (what, _, _) -> what)
        (let r = range 0 and outer = range 1 in
         let inner =
           Ops.range ~axis_type:Weak
             (Sym (Ops.after (Ops.int 4) [ outer ]))
             [ 2 ]
         in
         let ended = Ops.end_ (Ops.v Op.Noop) [ r ] in
         let loop = Ops.loop 3 in
         let backedge =
           Ops.backedge (Ops.int 1) ~loop ~cond:Ops.O.(outer < int 2)
         in
         let first = range 4 and closed = range 5 and last = range 6 in
         let barrier = Ops.barrier (Ops.end_ closed [ closed ]) [] in
         [
           ("a range", r, [ r ]);
           ("a sum over a range", Ops.O.(r + int 1), [ r ]);
           ("a range bounded after another", inner, [ inner; outer ]);
           ("an end of its range", ended, []);
           ( "an after ordered after an end",
             Ops.after Ops.O.(r + int 1) [ ended ],
             [] );
           ( "an end of an inner range",
             Ops.end_ Ops.O.(outer + inner) [ inner ],
             [ outer ] );
           ("a linear program", Ops.v ~src:[ r ] Op.Linear, []);
           ( "an after of a barrier over an end",
             Ops.after Ops.O.(first + closed + last) [ barrier; barrier ],
             [ first; last ] );
           ("a backedge", backedge, [ outer ]);
           ( "an after ordered after a backedge",
             Ops.after Ops.O.(outer + int 1) [ backedge ],
             [ outer ] );
           ( "a call to an external function",
             Ops.call
               (Ops.custom_function ~dtype:Int32 "external" [])
               [ Ops.int ~dtype:Uint64 0; Ops.O.(r + int 1) ],
             [ r ] );
           ( "a call to an external function of a source",
             Ops.call (Ops.custom_function "f" [ Ops.int 0 ]) [ r ],
             [ r ] );
           ( "a call to a kernel",
             Ops.call (Ops.sink [ var "inside" 0 1 ]) [ r ],
             [ r ] );
         ])
        (fun (_, u, expected) -> equal uops expected (list_ranges u));
    ]

let not_ranges = [ ("a constant", Ops.int 1); ("a variable", var "v" 0 4) ]

let axis_ids =
  group "axis_id"
    [
      cases "is a range's identity"
        ~id:(fun (ids, _) -> str Format.(pp_print_list pp_print_int) ids)
        [
          ([ 0 ], Ops.Axis_type.Weak); ([ 1; 0 ], Loop); ([ -2; 3; 1 ], Upcast);
        ]
        (fun (ids, a) ->
          equal (list int) ids
            (Ops.axis_id (Ops.range ~axis_type:a (Int 4) ids)));
      cases "raises on anything else" ~id:fst not_ranges (fun (_, u) ->
          rejects (fun () -> Ops.axis_id u));
    ]

let axis_types_of =
  group "axis_type"
    [
      cases "is a range's role" ~id:show_axis_type axis_types (fun a ->
          equal axis_type a
            (Ops.axis_type (Ops.range ~axis_type:a (Int 4) [ 0 ])));
      cases "raises on anything else" ~id:fst not_ranges (fun (_, u) ->
          rejects (fun () -> Ops.axis_type u));
    ]

(* Bounds of a range [bounded b] and what [bound_loads] and [known_at_launch]
   make of each, from their definitions: a bound reads memory through a load, or
   an index into global or workgroup storage; a launch knows a bound that reads
   no range and no memory; effects an after orders a bound after are no part of
   it. *)
let bounds_rows =
  let i = weak_range 0 in
  let x = Shape.param ~shape:[ Int 4 ] 0 Int32 in
  let y = Shape.param ~shape:[ Int 4 ] 1 Int32 in
  let one = Ops.int ~dtype:Int32 1 in
  let stores = Ops.end_ (Ops.store (Ops.index x [ i ]) one) [ i ] in
  let copy read =
    Ops.end_ (Ops.store (Ops.index x [ i ]) (read (Ops.index y [ i ]))) [ i ]
  in
  let local =
    Ops.v
      ~arg:(Param (Ops.param_arg ~addrspace:Local ~slot:2 Int32))
      ~src:[ Ops.size_src (Some 4) ]
      Op.Param
  in
  let register =
    Ops.v
      ~arg:(Param (Ops.param_arg ~addrspace:Reg ~slot:3 Int32))
      ~src:[ Ops.size_src (Some 4) ]
      Op.Param
  in
  let at p = Ops.cast (Ops.index p [ Ops.int 0 ]) Weak_int in
  let bounded_by_memory = Ops.range ~axis_type:Weak (Sym (at x)) [ 2 ] in
  [
    ("a constant", Ops.int 4, false, true);
    ("a variable", var "n" 0 8, false, true);
    ( "a variable after a loop that stores",
      Ops.after (var "n" 0 8) [ stores ],
      false,
      true );
    ( "a variable after a loop that loads",
      Ops.after (var "n" 0 8) [ copy (fun at -> Ops.load at []) ],
      false,
      true );
    ( "a variable after a loop that indexes",
      Ops.after (var "n" 0 8) [ copy Fun.id ],
      false,
      true );
    ("a range", Ops.O.(i + int 1), false, false);
    ("a constant after a range", Ops.after (Ops.int 4) [ i ], false, true);
    ("a variable after a range", Ops.after (var "n" 0 8) [ i ], false, true);
    ( "a constant after a range bounded by memory",
      Ops.after (Ops.int 4) [ bounded_by_memory ],
      false,
      true );
    ( "a load",
      Ops.cast (Ops.load (Ops.index x [ Ops.int 0 ]) []) Weak_int,
      true,
      false );
    ("an index into global storage", at x, true, false);
    ("an index into workgroup storage", at local, true, false);
    ("an index into registers", at register, false, true);
    ( "a sum of a load",
      Ops.O.(
        Ops.cast (Ops.load (Ops.index x [ Ops.int 0 ]) []) Weak_int + int 1),
      true,
      false );
  ]

let bounded b = Ops.range ~axis_type:Weak (Sym b) [ 1 ]

(* The rows' names of the ranges [pred] holds of: each bound's range, then a
   loop, which has no bound, alone and ordered after a range, as the linearizer
   orders a loop after the effects before it. *)
let holds pred expected =
  let loop = Ops.loop 7 in
  let after_range =
    Ops.replace ~src:[ Ops.after (Ops.nth loop 0) [ weak_range 0 ] ] loop
  in
  let rows =
    List.map
      (fun (what, b, reads, known) -> (what, bounded b, reads, known))
      bounds_rows
    @ [
        ("no bound, a loop's", loop, false, false);
        ("no bound, a loop's ordered after a range", after_range, false, false);
      ]
  in
  let names f =
    List.filter_map
      (fun (what, r, reads, known) ->
        if f r reads known then Some what else None)
      rows
  in
  equal (list string)
    (names (fun _ reads known -> expected reads known))
    (names (fun r _ _ -> pred r))

let loads =
  group "bound_loads"
    [
      test "is true of the ranges whose bound reads memory" (fun () ->
          holds Ops.bound_loads (fun reads _ -> reads));
      cases "raises on anything else" ~id:fst not_ranges (fun (_, u) ->
          rejects (fun () -> Ops.bound_loads u));
    ]

let launches =
  group "known_at_launch"
    [
      test "is true of the ranges whose bound reads no range and no memory"
        (fun () -> holds Ops.known_at_launch (fun _ known -> known));
      cases "raises on anything else" ~id:fst not_ranges (fun (_, u) ->
          rejects (fun () -> Ops.known_at_launch u));
    ]

(* A range's argument: its axis type and a short identity, so that drawn
   arguments often share a type, an identity or a prefix. *)
let gen_range_arg =
  Gen.(
    pair
      (of_list ~pp:Ops.Axis_type.pp axis_types)
      (list ~size:(int_range 1 3) (int_range (-1) 1)))

(* Two arguments, the second drawn apart, the first's identity extended or the
   first itself, in either order, so that a prefix and one argument are each
   over a fifth of the cases: 100 cases all miss either under 10^-9. *)
let gen_range_args =
  Gen.(
    let extended (a, ids) =
      map
        (fun more -> (a, ids @ more))
        (list ~size:(int_range 1 2) (int_range (-1) 1))
    in
    let pp_arg ppf (a, ids) =
      Format.fprintf ppf "(%a, [%a])" Ops.Axis_type.pp a
        Format.(
          pp_print_list
            ~pp_sep:(fun ppf () -> pp_print_string ppf "; ")
            pp_print_int)
        ids
    in
    with_pp
      (fun ppf (r0, r1) -> Format.fprintf ppf "(%a, %a)" pp_arg r0 pp_arg r1)
      (let* r0 = gen_range_arg in
       let* r1 =
         frequency
           [ (2, gen_range_arg); (1, extended r0); (1, constant ~pp:pp_arg r0) ]
       in
       let+ swap = bool in
       if swap then (r1, r0) else (r0, r1)))

let range_of (a, ids) = Ops.range ~axis_type:a (Int 4) ids

(* A strict prefix of an identity of one axis type. *)
let prefix ((a0, ids0), (a1, ids1)) =
  Ops.Axis_type.equal a0 a1
  && List.length ids0 < List.length ids1
  && List.filteri (fun n _ -> n < List.length ids0) ids1 = ids0

let range_args =
  group "compare_range_arg"
    [
      prop
        "orders by axis type, then identity, lexicographically, a prefix first"
        gen_range_args (fun ((a0, ids0), (a1, ids1)) ->
          let pair = ((a0, ids0), (a1, ids1)) in
          cover "one axis type" (Ops.Axis_type.equal a0 a1);
          cover "a prefix" (prefix pair || prefix ((a1, ids1), (a0, ids0)));
          cover "one argument" (Ops.Axis_type.equal a0 a1 && ids0 = ids1);
          let expected =
            match Ops.Axis_type.compare a0 a1 with
            | 0 -> List.compare Int.compare ids0 ids1
            | c -> c
          in
          equal int (sign expected)
            (sign
               (Ops.compare_range_arg
                  (range_of (a0, ids0))
                  (range_of (a1, ids1)))));
      cases "raises on anything else" ~id:fst not_ranges (fun (_, u) ->
          let r = weak_range 0 in
          rejects (fun () -> Ops.compare_range_arg u r);
          rejects (fun () -> Ops.compare_range_arg r u));
    ]

let range_strs =
  group "range_str"
    [
      cases
        "writes the identity with underscores, a negative part as m and its \
         magnitude"
        ~id:fst
        [ ("0_1", [ 0; 1 ]); ("m1", [ -1 ]); ("2_m3", [ 2; -3 ]); ("7", [ 7 ]) ]
        (fun (text, ids) ->
          equal string text
            (Ops.range_str (Ops.range ~axis_type:Loop (Int 4) ids)));
      cases "is in the axis colour if color" ~id:show_axis_type
        (List.filter (fun a -> a <> Ops.Axis_type.Placeholder) axis_types)
        (fun a ->
          equal string
            (Helpers.colored (Ops.Axis_type.color a) "3")
            (Ops.range_str ~color:true (Ops.range ~axis_type:a (Int 4) [ 3 ])));
    ]

let multirange_strs =
  group "multirange_str"
    [
      test "writes each range sorted by argument, separated by commas"
        (fun () ->
          let r0 = Ops.range ~axis_type:Global (Int 4) [ 0; 1 ]
          and r1 = Ops.range ~axis_type:Upcast (Int 4) [ 0; 0 ]
          and r2 = Ops.range ~axis_type:Global (Int 4) [ 0 ] in
          equal string "0,0_1,0_0" (Ops.multirange_str [ r1; r0; r2 ]));
      test "pads to printed columns, colours aside" (fun () ->
          let r0 = Ops.range ~axis_type:Global (Int 4) [ 0; 1 ]
          and r1 = Ops.range ~axis_type:Upcast (Int 4) [ 0; 0 ] in
          equal string "0_1,0_0   " (Ops.multirange_str ~pad:10 [ r1; r0 ]);
          equal int 10
            (Helpers.ansilen
               (Ops.multirange_str ~color:true ~pad:10 [ r0; r1 ]));
          equal string "0_1,0_0" (Ops.multirange_str ~pad:3 [ r1; r0 ]));
    ]

(* Nodes, through the slices the walks keep *)

let node_sets =
  group "Nodes"
    [
      prop "agree with the list of their nodes in the order they joined"
        gen_graph (fun steps ->
          let root = graph steps in
          let nodes = reference_toposort ~calls:Enter root in
          List.iter
            (fun u ->
              let s = Ops.backward_slice_with_self ~calls:Enter u in
              let l = Ops.Nodes.to_list s in
              equal uops (List.rev l) (Ops.Nodes.fold List.cons s []);
              equal int (List.length l) (Ops.Nodes.cardinal s);
              equal uops
                (List.filter (fun n -> List.memq n l) nodes)
                (List.filter (fun n -> Ops.Nodes.mem n s) nodes))
            nodes);
    ]

let groups =
  [
    axis_type_group;
    pp_devices;
    disk_devices;
    params;
    estimates;
    kernel_infos;
    function_names;
    equal_args;
    pp_args;
    identity;
    construct;
    replacing;
    equality;
    ordering;
    hashing;
    printing;
    raising;
    structure_order;
    ranks;
    keys;
    keys_with;
    tables;
    node_sets;
    data_types;
    promoting;
    identities;
    walk_errors;
    toposorts;
    topovisits;
    slices;
    slices_with_self;
    ops_in_slices;
    reaching;
    splitting;
    broadcasting;
    max_shapes;
    sints;
    range_starts;
    ended;
    ranges;
    axis_ids;
    axis_types_of;
    loads;
    launches;
    range_args;
    range_strs;
    multirange_strs;
  ]
