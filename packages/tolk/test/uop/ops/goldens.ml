(* The graphs of gen/uop/ops.py, built with the constructors: each golden is the
   graph tinygrad builds for the same calls. *)

open Tolk
open Common
open Ops

let a () = var "a" 0 10
let b () = var "b" 0 10
let x () = fvar "x"
let y () = fvar "y"
let p () = flag "p"
let pair (lo, hi) : (sint * sint) option = Some (Int lo, Int hi)
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

let movement () =
  let p = param [ 2; 3; 4 ] 0 Float32 in
  sink
    [
      reshape p (ints [ 6; 4 ]);
      reshape p (ints [ -1; 2 ]);
      reshape p (ints [ 2; 3; 4 ]);
      permute p [ 2; 0; 1 ];
      permute p [ 0; -1; 1 ];
      permute p [ 0; 1; 2 ];
      flip p [ 0; 2 ];
      flip p [ -1 ];
      shrink p [ pair (0, 1); None; pair (1, 3) ];
      shrink p [ None; None; None ];
      shrink_to p [ Some (Int 1); None; Some (Int 2) ];
      pad p [ pair (1, 0); None; pair (0, 2) ];
      pad_to p [ None; Some (Int 5); Some (Int 6) ];
      flatten p;
      flatten ~start:1 p;
      flatten ~start:0 ~stop:1 p;
      unflatten (reshape p (ints [ 2; 12 ])) 1 (ints [ 3; 4 ]);
    ]

let expansion () =
  let p = param [ 3; 1 ] 0 Float32 and s = param [ 1; 4; 1 ] 1 Float32 in
  sink
    [
      expand p (ints [ 2; 3; 1 ]);
      expand p (ints [ 3; 5 ]);
      expand p (ints [ 2; 3; 5 ]);
      expand p (ints [ -1; 5 ]);
      expand s (ints [ 2; 4; 3 ]);
      expand (float 1.) (ints [ 4; 8 ]);
      expand (float 1.) [];
    ]

let squeezes () =
  let p = param [ 1; 3; 1; 2 ] 0 Float32 in
  sink
    [
      squeeze p;
      squeeze ~axis:0 p;
      squeeze ~axis:2 p;
      squeeze ~axis:1 p;
      squeeze ~axis:(-2) p;
    ]

let axes () =
  let p = param [ 2; 3; 4 ] 0 Float32 in
  sink
    ([
       unsqueeze p 0;
       unsqueeze p 2;
       unsqueeze p (-1);
       transpose p 1 0;
       transpose p 0 2;
       transpose p (-1) (-2);
     ]
    @ split ~axis:1 p [ 1; 2 ]
    @ split ~axis:(-1) p [ 2; 2 ]
    @ split p [ 2 ])

let stacks () =
  let a = param [ 2; 3 ] 0 Float32 and b = param [ 2; 3 ] 1 Float32 in
  let h = param [ 2; 3 ] 2 Float16 in
  sink
    [
      stack [ a; b ];
      stack ~axis:1 [ a; b ];
      stack ~axis:(-1) [ a; b ];
      stack [ a; h ];
      stack [ a; b; a ];
      stack [ invalid; cast (float 1.) Float32 ];
      broadcast (int 1) 3;
      broadcast (int 1) 1;
    ]

let concatenation () =
  let a = param [ 2; 3 ] 0 Float32 and b = param [ 2; 3 ] 1 Float32 in
  let c = param [ 4; 3 ] 2 Float32 in
  sink
    [
      cat a [ b ];
      cat ~axis:1 a [ b ];
      cat a [ c ];
      cat a [ c; b ];
      cat ~axis:(-1) a [ b ];
    ]

let pools () =
  let p = param [ 2; 5; 6 ] 0 Float32 in
  sink
    [
      pool p [ 3 ];
      pool ~stride:[ 2 ] p [ 3 ];
      pool ~stride:[ 1; 2 ] ~dilation:[ 2; 1 ] p [ 2; 2 ];
      pool p [ 5; 6 ];
      pool ~stride:[ 3 ] p [ 2 ];
      repeat p [ 2; 1; 1 ];
      repeat p [ 3; 1; 1; 2 ];
    ]

let running () =
  let p = param [ 3; 5 ] 0 Float32 in
  sink [ cumalu p 1 Op.Add; cumalu p 0 Op.Max; cumalu p (-1) Op.Mul ]

let aranges () =
  sink
    [
      arange 5;
      arange ~start:2 ~step:3 9;
      arange ~start:4 ~step:(-1) 0;
      arange ~dtype:Int64 3;
      arange ~dtype:Float32 3;
      arange 0;
    ]

let padding () =
  let p = param [ 4; 4 ] 0 Float32 in
  sink
    [
      pad ~value:(f 0.) p [ pair (1, 1); pair (2, 0) ];
      pad ~value:(f 1.5) p [ pair (1, 1); None ];
      pad p [ pair (-1, 2); pair (0, -2) ];
      pad_to ~value:(f 2.) p [ Some (Int 6); None ];
      pad ~value:(f (-1.)) p [ pair (1, -1); pair (0, 2) ];
    ]

let reductions () =
  let p = param [ 2; 3; 4 ] 0 Float32 and q = param [ 2; 1; 4 ] 1 Float32 in
  sink
    [
      rop p Op.Add [ 0 ];
      rop p Op.Max [ 2; 0 ];
      rop p Op.Mul [ 1 ];
      rop q Op.Add [ 1 ];
      rop q Op.Add [ 1; 2 ];
      rop p Op.Add [];
    ]

let constants_like () =
  let p = param [ 2; 3 ] 0 Float32 and r = range (Int 4) [ 0 ] in
  sink
    [
      const_like p (i 0);
      const_like ~dtype:Bool p (`Bool true);
      const_like r (i 3);
      const_like (int 1) (i 2);
      vconst_like
        (stack [ float ~dtype:Float16 1.; float ~dtype:Float16 2. ])
        (i 0);
    ]

let kernel_nodes () =
  let buf = param [ 16 ] 0 Float32 in
  let r = range (Int 16) [ 0 ] in
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
  let r = range (Int 10) [ 0 ] in
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

let contraction () =
  let r0 = range ~axis_type:Upcast (Int 2) [ 0 ]
  and r1 = range ~axis_type:Upcast (Int 3) [ 1 ] in
  sink
    [
      contract O.((r0 * int 3) + r1) [ r0; r1 ]; contract O.(r0 + int 1) [ r0 ];
    ]

let storage () =
  sink
    [
      param ~device:(Single "CPU") [ 2; 3; 4 ] 2 Float32;
      Ops.param 3 Float16;
      Ops.param ~shape:(ints [ 4 ])
        ~vmin_vmax:(f 0., f 1.)
        ~name:"w" ~volatile:true 4 Float32;
      placeholder ~slot:5 [ 2; 3 ] Weak_int;
      placeholder ~slot:6 ~addrspace:Local [ 8 ] Float32;
      placeholder ~slot:7 ~addrspace:Reg ~tag:(String "acc") [ 4 ] Float16;
      alloc ~slot:9 [] Int32;
      alloc ~slot:10
        ~device:(Multi [ "CPU:0"; "CPU:1" ])
        ~axis:0
        (ints [ 4; 2 ])
        Float32;
      view_as (new_buffer ~slot:11 (Single "CPU") 4 Float32) (ints [ 2; 2 ]);
    ]

let multi = Multi [ "CPU:0"; "CPU:1" ]

let storage_like () =
  let p = param ~device:(Single "CPU") [ 2; 3 ] 0 Float32 in
  let v = weak_var "n" 1 8 in
  let s = unshard (param ~device:multi [ 4; 3 ] 1 Float32) [ 0 ] in
  sink
    [
      param_like p 5;
      param_like v 6;
      param_like (bind v (i 3)) 7;
      param_like s 8;
      placeholder_like p 9;
      alloc_like ~slot:10 p;
      alloc_like ~slot:11 ~addrspace:Local s;
    ]

let sets () =
  let p = param [ 4 ] 0 Float32 and r = range (Int 4) [ 0 ] in
  let q = reshape p (ints [ 2; 2 ]) in
  let idx = index p [ r ] in
  sink
    [ set ~ends:[ r ] idx (const_like idx (f 1.)); set q (const_like q (f 2.)) ]

let shards () =
  let p = param ~device:(Single "CPU") [ 4; 6 ] 0 Float32 in
  let m = param ~device:multi [ 4; 6 ] 1 Float32 in
  let devices = [ "CPU:0"; "CPU:1" ] in
  sink
    [
      shard p devices;
      unshard m [ 0 ];
      unshard
        ~ranges:
          [
            range ~axis_type:Device (Int 2) [ -1 ];
            range ~axis_type:Local (Int 3) [ -2 ];
          ]
        m [ 1; 0 ];
      mselect m 0;
      mstack (mselect m 0) [ mselect m 1 ];
      mstack (mselect m 0) [];
      allreduce m Op.Add multi;
      copy_to_device p (Single "CUDA");
      copy_to_device m (Multi [ "CUDA:0"; "CUDA:1" ]);
      copy_to_device ~shard:1 m (Single "CPU");
    ]

let calls () =
  let r = range (Int 4) [ 0 ] in
  let idx = index (param [ 4 ] 0 Float32) [ r ] in
  let body = sink [ end_ (store idx (const_like idx (f 1.))) [ r ] ] in
  let a = param ~device:(Single "CPU") [ 4 ] 0 Float32 in
  sink
    [
      call body [ a ];
      call ~name:"fill" ~precompile:true body [ a ];
      call ~ret_dtype:Int32 (custom_function "f" [ int ~dtype:Uint64 0 ]) [];
      store_call a (param ~device:(Single "CPU") [ 4 ] 1 Float32);
    ]

let custom_kernels () =
  let a = param ~device:(Single "CPU") [ 4 ] 0 Float32 in
  let b = param ~device:(Single "CPU") [ 4 ] 1 Float32 in
  let copy = function
    | [ x; y ] ->
        let r = range (Int 4) [ 0 ] in
        sink [ end_ (store (index x [ r ]) (load (index y [ r ]) [])) [ r ] ]
    | _ -> invalid_arg "copy takes two placeholders"
  in
  sink (custom_kernel [ a; b ] copy)

let variables_and_binding () =
  let n = weak_var "n" 1 8 and m = weak_var ~multiple_of:2 "m" 0 4 in
  let e = O.((bind n (i 3) * bind m (i 2)) + n) in
  let all_unbound, _ = unbind_all e in
  sink
    ([
       bind n (i 3);
       unbound (bind n (i 3));
       unbound (rtag ~tag:(String "t") (bind n (i 3)));
       all_unbound;
     ]
    @ variables e
    @ variables O.(range ~axis_type:Device (Int 4) [ -1 ] + n))

let getaddrs () =
  let p = param ~device:(Single "CPU") [ 4 ] 0 Float32 in
  sink
    [
      getaddr p;
      getaddr ~device:"CPU:1" p;
      getaddr ~device:"CPU" (v ~arg:(Bytes "ab") Op.Binary);
      getaddr (int 1);
      getaddr (after p [ sink [] ]);
    ]

let instructions () =
  let x = index (param [ 4 ] 0 Float32) [ int 0 ] in
  sink
    [
      ins x "v_mov";
      ins ~src:[ x; x ] ~dtype:Float16 x "v_add";
      ins ~dtype:Void ~tag:None (rtag ~tag:(Int 1) x) "nop";
    ]

let wmmas () =
  let a = param [ 8 ] 0 Float16
  and b = param [ 8 ] 1 Float16
  and c = param [ 4 ] 2 Float32 in
  sink
    [
      wmma a b ~acc:c ~dims:(8, 16, 16) ~threads:32;
      wmma
        ~upcast_axes:([ ([ 0 ], 2) ], [ ([ 1 ], 2) ], [ ([ 2 ], 2) ])
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
      substitute ~calls:Skip ~pass:Fixed_point
        (replace ~tag:(Some (Tag.Int 1)) O.(a + int 4))
        [ (a, c) ];
      substitute ~calls:Skip ~pass:Fixed_point (sin (sin x)) [ (sin x, sqrt x) ];
    ]

let clones () =
  let p = param ~device:(Single "CPU") [ 2; 3 ] 0 Float32 in
  sink
    [
      clone p;
      clone ~device:(Single "CUDA") p;
      empty_like p;
      empty_like ~dtype:Float16 ~device:(Single "CUDA") p;
      empty ~device:(Single "CPU") (ints [ 2; 3 ]) Int32;
    ]

let outputs () =
  let a = param ~device:(Single "CPU") [ 4 ] 0 Float32 in
  sink
    (call_with_outputs ~name:"two" O.[ a + float 1.; a * float 2. ] [ a ]
    @ [ call_with_output ~precompile:true a [ a ] ])

(* Symbolic sizes *)

let symbolic_storage () =
  let n = weak_var "n" 1 8 in
  sink
    [
      Ops.param ~shape:[ Int 2; Sym n ] 0 Float32;
      Ops.param ~shape:[ Sym n ] 1 Int32;
      alloc ~slot:8 [ Int 2; Sym n ] Weak_float;
    ]

let symbolic_shards () =
  sink
    [
      shard ~axis:1
        (param ~device:(Single "CPU") [ 4; 6 ] 0 Float32)
        [ "CPU:0"; "CPU:1" ];
    ]

let symbolic_shard_slices () =
  let p = param [ 4; 6 ] 0 Float32 in
  sink
    [
      shard_slice p 1 (range ~axis_type:Device (Int 2) [ -1 ]);
      shard_slice p 0 (range ~axis_type:Local (Int 4) [ 0 ]);
    ]

let symbolic_outputs () =
  let a = param ~device:(Single "CPU") [ 4 ] 0 Float32 in
  let n = weak_var "n" 1 4 in
  let b = Ops.param ~device:(Single "CPU") ~shape:[ Sym n ] 1 Float32 in
  let formal =
    Ops.param ~vmin_vmax:(i 1, i 8) ~name:"d" ~addrspace:(Some Alu) 0 Int32
  in
  sink
    (call_with_outputs ~output_pos:[ 0 ] O.[ b + float 1. ] [ a; b ]
    @ [
        call_with_output
          (expand (cast (float 1.) Float32) [ Sym formal ])
          [ int 5 ];
        empty ~device:(Single "CPU") [ Int 2; Sym n ] Int32;
      ])

let hcq_calls () =
  let b = new_buffer ~slot:3 (Single "AMD") 4 Float32 in
  let est : estimates = { ops = Int 1; lds = Int 2; mem = Int 3 } in
  let kernel =
    {
      devices = [ "AMD" ];
      name = "k";
      estimates = est;
      stamps = [ 0; 1 ];
      profile_key = Some "key";
      input_slots = [ 0 ];
      outs = [ 0 ];
      ins = [ 0 ];
    }
  in
  let info =
    {
      device = [ "AMD" ];
      kernels = [ kernel ];
      estimates = est;
      nargs = 2;
      table = 1;
      inputs = [ (b, 0, "GLOBAL") ];
      slots = [ ("AMD", 1) ];
      written_bufs = [ b ];
      writes = [ b ];
      copies = [];
    }
  in
  let zero : estimates = { ops = Int 0; lds = Int 0; mem = Int 0 } in
  let bare =
    {
      device = [ "AMD" ];
      kernels = [];
      estimates = zero;
      nargs = 0;
      table = -1;
      inputs = [];
      slots = [];
      written_bufs = [];
      writes = [];
      copies = [];
    }
  in
  sink
    [
      call ~aux:info (custom_function "f" []) [ b ];
      call ~aux:bare (custom_function "f" []) [ b ];
    ]

(* Storage made without a slot takes a number no other storage has. [minted]
   renumbers those of [sink] from 1000, in the order the graph lists them, as
   gen/uop/ops.py numbers tinygrad's. *)
let minted sink =
  let slots =
    List.fold_left
      (fun acc u ->
        match (op u, arg u) with
        | Op.Alloc, Param p when not (List.mem_assoc p.slot acc) ->
            (p.slot, 1000 + List.length acc) :: acc
        | _ -> acc)
      []
      (toposort ~calls:Enter sink)
  in
  let renumber =
    Pattern_matcher.v (fun () ->
        [
          Pattern_matcher.rule (Upat.op ~name:"x" Op.Alloc) (fun m ->
              let x = m "x" in
              match arg x with
              | Param p ->
                  Some
                    (replace
                       ~arg:(Param { p with slot = List.assoc p.slot slots })
                       x)
              | _ -> None);
        ])
  in
  graph_rewrite ~calls:Skip ~pass:Once ~ctx:() sink (After_sources renumber)

(* Each golden, the graph built for it, and whether it mints slots. *)
let all =
  [
    ("typed_constants", typed_constants, false);
    ("ccast", ccast, false);
    ("subtraction", subtraction, false);
    ("negation", negation, false);
    ("weak_promotion", weak_promotion, false);
    ("division", division, false);
    ("constant_division", constant_division, false);
    ("comparisons", comparisons, false);
    ("bitwise", bitwise, false);
    ("extrema", extrema, false);
    ("selection", selection, false);
    ("unary", unary, false);
    ("powers", powers, false);
    ("sums_and_products", sums_and_products, false);
    ("casts", casts, false);
    ("bitcasts", bitcasts, false);
    ("movement", movement, false);
    ("expansion", expansion, false);
    ("squeezes", squeezes, false);
    ("axes", axes, false);
    ("stacks", stacks, false);
    ("concatenation", concatenation, false);
    ("pools", pools, false);
    ("running", running, false);
    ("aranges", aranges, false);
    ("padding", padding, false);
    ("reductions", reductions, false);
    ("constants_like", constants_like, false);
    ("kernel_nodes", kernel_nodes, false);
    ("validity", validity, false);
    ("contraction", contraction, false);
    ("storage", storage, false);
    ("storage_like", storage_like, false);
    ("sets", sets, false);
    ("shards", shards, false);
    ("calls", calls, false);
    ("custom_kernels", custom_kernels, false);
    ("variables_and_binding", variables_and_binding, false);
    ("getaddrs", getaddrs, false);
    ("instructions", instructions, false);
    ("wmmas", wmmas, false);
    ("substitution", substitution, false);
    ("hcq_calls", hcq_calls, false);
    ("clones", clones, true);
    ("outputs", outputs, true);
    ("symbolic_storage", symbolic_storage, false);
    ("symbolic_shards", symbolic_shards, false);
    ("symbolic_shard_slices", symbolic_shard_slices, false);
    ("symbolic_outputs", symbolic_outputs, true);
  ]

(* The nodes of pretty.golden, in order. *)
let printed_nodes () =
  let forty_two = int 42 and one = int 1 and n = weak_var "n" 1 8 in
  [
    forty_two;
    O.(forty_two + int 3);
    O.(forty_two + forty_two);
    O.((one + one) * (one + one));
    rtag ~tag:(String "x") forty_two;
    rtag ~tag:(Tuple [ String "y"; Int 1 ]) forty_two;
    n;
    sink ~kernel:(kernel_info ()) O.[ n + int 1; n * int 2 ];
    range ~axis_type:Reduce (Int 4) [ 0 ];
    int ~dtype:Int32 3;
    float 1.5;
    param [ 2; 3 ] 0 Float32;
  ]

(* Arguments and tags, by the name reprs.golden gives them. *)

let reprs () =
  let n = weak_var "n" 1 10 in
  let m () = param ~device:multi [ 4 ] 0 Float32 in
  let cpu = param ~device:(Single "CPU") [ 4 ] 0 Float32 in
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
    ("range", range ~axis_type:Reduce (Int 4) [ 0 ]);
    ("range with sub-axes", range ~axis_type:Upcast (Int 4) [ 1; 0 ]);
    ("negative range", range ~axis_type:Device (Int 4) [ -1 ]);
    ( "reduce",
      v
        ~src:[ expand (float 1.) (ints [ 4 ]) ]
        ~arg:(Reduce { op = Op.Add; num_axes = 1 })
        Op.Reduce );
    ("special", special (Int 8) "gidx0");
    ("permute", permute (param [ 2; 3 ] 0 Float32) [ 1; 0 ]);
    ("flip", flip (param [ 2; 3 ] 0 Float32) [ 1 ]);
    ("device", copy_to_device cpu (Single "CUDA"));
    ("devices", copy_to_device cpu multi);
    ("mselect", mselect (m ()) 1);
    ("unshard", unshard (m ()) [ 0 ]);
    ("allreduce", allreduce (m ()) Op.Add multi);
    ("custom", v ~arg:(Code { code = "barrier();"; dtype = Void }) Op.Custom);
    ("ins", v ~arg:(Code { code = "s_endpgm"; dtype = Void }) Op.Ins);
    ("binary", v ~arg:(Bytes "\x7fELF\x00\n'\"") Op.Binary);
    ("source", v ~arg:(String "int x = 'a';\n") Op.Source);
    ("param", param [ 256 ] 3 Float32);
    ("scalar param", Ops.param 1 Int32);
    ("variable", n);
    ("bound variable", bind n (i 3));
    ( "named local buffer",
      placeholder ~slot:2 ~addrspace:Local ~tag:(String "buf") [ 4 ] Int32 );
    ( "volatile param",
      Ops.param ~shape:(ints [ 4 ]) ~volatile:true ~device:(Single "CPU") 0
        Uint32 );
    ("alloc", alloc ~slot:7 ~device:(Single "CPU") (ints [ 4 ]) Float32);
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
          { device = Some (Single "CPU"); addrspace = Local; keep = Whole }
        (float 1.) [] );
    ( "bufferize without device",
      bufferize
        ~opts:{ device = None; addrspace = Global; keep = Removable }
        (float 1.) [] );
    ("call", call ~name:"f" (sink []) []);
    ( "call returning",
      call ~ret_dtype:Int32 ~precompile:true (custom_function "f" []) [] );
    ( "program",
      v
        ~src:[ sink ~kernel:(kernel_info ()) [] ]
        ~arg:
          (Program
             {
               global_size = ints [ 4; 1; 1 ];
               local_size = ints [ 8; 1; 1 ];
               vars = [];
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
        ~upcast_axes:([ ([ 0 ], 2); ([ 1 ], 2) ], [ ([ 2 ], 2) ], [ ([ 3 ], 2) ])
        (param [ 8 ] 0 Float16) (param [ 8 ] 1 Float16)
        ~acc:(param [ 4 ] 2 Float32) ~dims:(8, 16, 16) ~threads:32 );
    ("bytes tag", rtag ~tag:(Bytes "\x00a'") (int 2));
    ( "tagged",
      rtag
        ~tag:(Tuple [ String "x"; Int 1; Bool true; Dtype Int32; Tuple [] ])
        (int 1) );
  ]
