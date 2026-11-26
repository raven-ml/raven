(* The graphs of gen/uop/ops.py, built with the constructors: each golden is the
   graph tinygrad builds for the same calls. *)

open Tolk
open Common
open Ops
open Shape

let a () = var "a" 0 10
let b () = var "b" 0 10
let x () = fvar "x"
let y () = fvar "y"
let p () = flag "p"

(* [u], a define, named [name]. *)
let named u name = replace u ~arg:(Param { (param_arg_of u) with name = Some name })
let param ?device shape slot dt = param ?device ~shape:(ints shape) slot dt

let typed_constants () =
  sink
    [
      int ~dtype:Int32 3;
      bool true;
      float 1.5;
      int ~dtype:Float32 2;
      float ~dtype:Int32 1.5;
      bool ~dtype:Bool true;
      const ~dtype:Float32 `Invalid;
      int ~dtype:Int8 300;
      cconst Bool (`Bool true);
      consts [ i 1; f 2.5; `Bool true ];
      consts ~dtype:Int8 [ i 1; i 2 ];
    ]

let ccast () =
  sink
    [
      ccast (int 3) Float32; ccast (a ()) Int64; ccast (int ~dtype:Int32 3) Int8;
    ]

let subtraction () =
  let a = a () and b = b () in
  sink O.[ a - b; a - int 1; int 1 - a; x () - float 1.5 ]

let negation () =
  sink O.[ ~-(a ()); ~-(x ()); ~-(p ()); ~-(Ops.int ~dtype:Uint8 1) ]

let weak_promotion () =
  sink
    O.
      [
        a () + int 1;
        a () + float 1.5;
        x () + int 2;
        fvar ~dtype:Float16 "x" + a ();
        int 1 + float 2.5;
        var ~dtype:Int8 "a" 0 10 + var ~dtype:Uint8 "b" 0 10;
        var ~dtype:Int8 "a" 0 10 + int 1;
        expand (int 1) (ints [ 4 ]) + param [ 4 ] 0 Float32;
      ]

let division () =
  let a = a () and b = b () and x = x () and y = y () in
  sink
    O.
      [
        a / b;
        a // b;
        a % b;
        x / y;
        x // y;
        x % y;
        div ~rounding:`Trunc a b;
        div ~rounding:`Trunc x y;
        fmod a b;
        fmod x y;
        p () / a;
        int 8 // a;
        int 9 / x;
      ]

let constant_division () =
  let a = a () and x = x () and p = p () in
  sink
    O.
      [
        a / int 3;
        a // int 3;
        int 7 // a;
        a % int 3;
        div ~rounding:`Trunc a (int 3);
        fmod a (int 3);
        x // float 2.;
        x % float 2.;
        div ~rounding:`Trunc x (float 2.);
        fmod x (float 2.);
        p // bool true;
        p % bool true;
        fmod p (bool true);
        a // float 2.;
        a % float 2.;
      ]

let comparisons () =
  let a = a () and b = b () in
  sink
    O.
      [
        a < b;
        a > b;
        a <= b;
        a >= b;
        ne a b;
        eq a b;
        a < int 3;
        int 3 < a;
        x () < int 1;
      ]

let bitwise () =
  let a = a () and b = b () and u = var ~dtype:Uint8 "u" 0 10 and p = p () in
  sink
    O.
      [
        a land b;
        a lor b;
        a lxor b;
        lnot a;
        lnot u;
        lnot p;
        a lsl int 2;
        a lsr int 1;
        int 1 lsl a;
        p land bool true;
        p lor bool false;
        logical_not p;
        logical_not a;
      ]

let extrema () =
  sink
    O.
      [
        maximum (a ()) (b ());
        minimum (a ()) (b ());
        minimum (x ()) (y ());
        minimum (var ~dtype:Uint8 "u" 0 10) (int 3);
        maximum (x ()) (int 0);
        minimum (a ()) (float 1.5);
      ]

let selection () =
  let c = O.(a () < b ()) in
  sink
    O.
      [
        where c (a ()) (int 0);
        where c (float 1.5) (x ());
        where c (int 1) (int 2);
        where c (fvar ~dtype:Float16 "x") (y ());
      ]

let unary () =
  let a = a () and x = x () in
  sink
    [
      sqrt a;
      sqrt x;
      exp2 a;
      log2 x;
      reciprocal x;
      reciprocal a;
      trunc x;
      floor x;
      sqrt (int 4);
      exp2 (cast x Float16);
    ]

let powers () =
  sink
    O.
      [
        pow (x ()) (int 2);
        pow (x ()) (y ());
        pow (a ()) (int 2);
        pow (x ()) (float 0.5);
        pow (float 2.) (a ());
        pow (a ()) (float 0.5);
      ]

let sums_and_products () =
  let p = p () and q = flag "q" in
  sink
    O.
      [
        usum (a ()) [ b (); var "c" 0 10 ];
        uprod (a ()) [ b (); int 2 ];
        usum p [ q ];
        uprod p [ q ];
        usum (a ()) [];
      ]

let casts () =
  let a = a () in
  sink
    [
      cast a Int32;
      cast a Float32;
      cast (cast a Float32) Float32;
      bitcast a Uint32;
      bitcast a Int32;
      bitcast (x ()) Int32;
      cast (int 1) Bool;
    ]

let bitcasts () =
  sink
    [
      bitcast (param [ 4 ] 0 Uint8) Uint16;
      bitcast (param [ 4 ] 1 Uint16) Uint8;
      bitcast (param [ 4 ] 2 Float32) Uint32;
      stack [ int 1; int 2 ];
    ]

let kernel_nodes () =
  let buf = param [ 16 ] 0 Float32 in
  let r = range ~axis_type:Weak (Int 16) [ 0 ] in
  let red = reduce (load (index buf [ r ]) []) Op.Add [ r ] in
  let gidx = special (Int 4) "gidx0" in
  let st =
    store
      ~gate:O.(gidx < int 2)
      (index (param [ 4 ] 1 Float32) [ gidx ])
      O.(red + float 1.)
  in
  let triple = consts [ i 1; i 2; i 3 ] in
  sink
    [
      end_ st [ r ];
      end_ st [];
      after red [ st ];
      after red [];
      barrier st [];
      loop 3;
      range ~axis_type:Upcast ~dtype:Int32 (Int 8) [ 2 ];
      backedge (int 1) ~loop:(loop 1) ~cond:O.(r < int 2);
      index buf [ int 3 ];
      index triple [ int 1 ];
      index triple [ int ~dtype:Int32 1 ];
      group [ st; st ];
      group [ st ];
    ]

let validity () =
  let r = range ~axis_type:Weak (Int 10) [ 0 ] in
  let v = valid r O.(r < int 5) in
  sink
    [
      v;
      get_idx v;
      get_valid v;
      get_idx r;
      get_valid r;
      get_valid invalid;
      get_idx (stack [ v; r ]);
      get_valid (stack [ v; r ]);
    ]

let sets () =
  let p = param [ 4 ] 0 Float32 and r = range ~axis_type:Weak (Int 4) [ 0 ] in
  let q = reshape p (ints [ 2; 2 ]) in
  let idx = index p [ r ] in
  sink
    [ set ~ends:[ r ] idx (const_like idx (f 1.)); set q (const_like q (f 2.)) ]

let instructions () =
  let x = index (param [ 4 ] 0 Float32) [ int 0 ] in
  sink
    [
      ins x "v_mov";
      ins ~src:[ x; x ] ~dtype:Float16 x "v_add";
      ins ~dtype:Void x "nop";
    ]

let wmmas () =
  let a = param [ 8 ] 0 Float16
  and b = param [ 8 ] 1 Float16
  and c = param [ 4 ] 2 Float32 in
  sink
    [
      wmma a b ~acc:c ~dims:(8, 16, 16) ~threads:32;
      wmma
        ~upcast_axes:
          ( [ ((Upcast, [ 0 ]), 2) ],
            [ ((Upcast, [ 1 ]), 2) ],
            [ ((Upcast, [ 2 ]), 2) ] )
        a b ~acc:c ~dims:(8, 16, 16) ~threads:32;
    ]

let substitution () =
  let a = a () and b = b () and c = var "c" 0 10 and x = x () in
  let e = O.(a + int 4 + (a + int 5)) in
  let sin u = alu u Op.Sin [] in
  sink
    [
      substitute ~calls:Skip ~pass:Fixed_point e [ (a, b) ];
      substitute ~calls:Skip ~pass:Fixed_point e [ (a, a) ];
      substitute ~calls:Skip ~pass:Fixed_point O.(a + int 4) [ (a, c) ];
      substitute ~calls:Skip ~pass:Fixed_point (sin (sin x)) [ (sin x, sqrt x) ];
    ]

(* Each golden and the graph built for it. *)
let all =
  [
    ("typed_constants", typed_constants);
    ("ccast", ccast);
    ("subtraction", subtraction);
    ("negation", negation);
    ("weak_promotion", weak_promotion);
    ("division", division);
    ("constant_division", constant_division);
    ("comparisons", comparisons);
    ("bitwise", bitwise);
    ("extrema", extrema);
    ("selection", selection);
    ("unary", unary);
    ("powers", powers);
    ("sums_and_products", sums_and_products);
    ("casts", casts);
    ("bitcasts", bitcasts);
    ("kernel_nodes", kernel_nodes);
    ("validity", validity);
    ("sets", sets);
    ("instructions", instructions);
    ("wmmas", wmmas);
    ("substitution", substitution);
  ]

(* The nodes of pretty.golden, in order. *)
let printed_nodes () =
  let forty_two = int 42 and one = int 1 and n = weak_var "n" 1 8 in
  [
    forty_two;
    O.(forty_two + int 3);
    O.(forty_two + forty_two);
    O.((one + one) * (one + one));
    n;
    sink ~kernel:(kernel_info ()) O.[ n + int 1; n * int 2 ];
    range ~axis_type:Weak (Int 4) [ 0 ];
    int ~dtype:Int32 3;
    float 1.5;
    param [ 2; 3 ] 0 Float32;
  ]

(* Arguments, by the name reprs.golden gives them. *)

let reprs () =
  let n = weak_var "n" 1 10 in
  let cpu = param ~device:"CPU" [ 4 ] 0 Float32 in
  [
    ("none", v Op.Noop);
    ("int", int 42);
    ("negative int", int (-3));
    ("huge int", const (`Int (Bigint.shift_left Bigint.one 100)));
    ("float", float 1.5);
    ("negative zero", float (-0.));
    ("nan", float Float.nan);
    ("inf", float Float.infinity);
    ("bool", bool true);
    ("invalid", invalid);
    ("dtype", cast (int 1) Float16);
    ("range", range ~axis_type:Weak (Int 4) [ 0 ]);
    ("range with sub-axes", range ~axis_type:Upcast (Int 4) [ 1; 0 ]);
    ("negative range", range ~axis_type:Loop (Int 4) [ -1 ]);
    ( "reduce",
      v
        ~src:[ expand (float 1.) (ints [ 4 ]) ]
        ~arg:(Reduce { op = Op.Add; num_axes = 1 })
        Op.Reduce );
    ("special", special (Int 8) "gidx0");
    ("permute", permute (param [ 2; 3 ] 0 Float32) [ 1; 0 ]);
    ("flip", flip (param [ 2; 3 ] 0 Float32) [ 1 ]);
    ("device", copy_to_device cpu "CUDA");
    ("custom", call (custom_function ~fmt:true "barrier();" []) []);
    ("ins", v ~arg:(Code { code = "s_endpgm"; dtype = Void }) Op.Ins);
    ("binary", v ~arg:(Bytes "\x7fELF\x00\n'\"") Op.Binary);
    ("source", v ~arg:(String "int x = 'a';\n") Op.Source);
    ("param", param [ 256 ] 3 Float32);
    ("scalar param", Shape.param 1 Int32);
    ("variable", n);
    ("bound variable", bind n (i 3));
    ( "named local buffer",
      named (placeholder ~slot:2 ~addrspace:Local [ 4 ] Int32) "buf" );
    ( "volatile param",
      Shape.param ~shape:(ints [ 4 ]) ~volatile:true ~device:"CPU" 0
        Uint32 );
    ("alloc", alloc ~slot:7 ~device:"CPU" (ints [ 4 ]) Float32);
    ("kernel", sink ~kernel:(kernel_info ()) []);
    ( "kernel with opts",
      sink
        ~kernel:
          (kernel_info ~name:"kern"
             ~applied_opts:
               [
                 Opt.Split
                   { axis = 0; amount = 4; target = Upcast; top = false };
                 Opt.Split { axis = 1; amount = 16; target = Local; top = true };
                 Opt.Tc { axis = 0; tc_select = -1; tc_opt = 2; use_tc = 1 };
                 Opt.Padto { axis = 1; amount = 32 };
                 Opt.Swap { axis = 0; with_axis = 1 };
               ]
             ~opts_to_apply:[ Opt.Padto { axis = 0; amount = 4 } ]
             ~estimates:{ ops = Int 1; lds = Int 2; mem = Int 3 }
             ~beam:2 ())
        [] );
    ( "bufferize",
      bufferize
        ~opts:
          { device = Some "CPU"; addrspace = Local; keep = Whole }
        (float 1.) [] );
    ( "bufferize without device",
      bufferize
        ~opts:{ device = None; addrspace = Global; keep = Removable }
        (float 1.) [] );
    ("call", call ~name:"f" (sink []) []);
    ( "call returning",
      call ~precompile:true (custom_function ~dtype:Int32 "f" []) [] );
    ( "program",
      v
        ~src:[ sink ~kernel:(kernel_info ()) [] ]
        ~arg:
          (Program
             {
               global_size = ints [ 4; 1; 1 ];
               local_size = ints [ 8; 1; 1 ];
               params = [];
               globals = [ 0; 1 ];
               outs = [ 0 ];
               ins = [ 1 ];
               target =
                 {
                   device = "CPU";
                   renderer = "CLANG";
                   arch = "";
                   interface = "";
                   indices = "";
                 };
             })
        Op.Program );
    ( "wmma",
      wmma
        ~upcast_axes:
          ( [ ((Upcast, [ 0 ]), 2); ((Upcast, [ 1 ]), 2) ],
            [ ((Upcast, [ 2 ]), 2) ],
            [ ((Upcast, [ 3 ]), 2) ] )
        (param [ 8 ] 0 Float16) (param [ 8 ] 1 Float16)
        ~acc:(param [ 4 ] 2 Float32) ~dims:(8, 16, 16) ~threads:32 );
  ]
