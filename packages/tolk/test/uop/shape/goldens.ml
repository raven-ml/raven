(* The graphs of gen/uop/shape.py, built with the constructors: each golden is
   the graph tinygrad builds for the same calls. *)

open Tolk
open Common
open Ops
open Shape

let var = Simplification.var

(* [u], a define, named [name]. *)
let named u name = replace u ~arg:(Param { (param_arg_of u) with name = Some name })
let pair (lo, hi) : (sint * sint) option = Some (Int lo, Int hi)
let param ?device shape slot dt = param ?device ~shape:(ints shape) slot dt

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
  let p = param [ 2; 3 ] 0 Float32
  and r = range ~axis_type:Weak (Int 4) [ 0 ] in
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
      param ~device:"CPU" [ 2; 3; 4 ] 2 Float32;
      Shape.param 3 Float16;
      Shape.param ~shape:(ints [ 4 ])
        ~vmin_vmax:(f 0., f 1.)
        ~name:"w" ~volatile:true 4 Float32;
      placeholder ~slot:5 [ 2; 3 ] Weak_int;
      placeholder ~slot:6 ~addrspace:Local [ 8 ] Float32;
      named (placeholder ~slot:7 ~addrspace:Reg [ 4 ] Float16) "acc";
      alloc ~slot:9 [] Int32;
      view_as (new_buffer ~slot:11 "CPU" 4 Float32) (ints [ 2; 2 ]);
    ]

let storage_like () =
  let p = param ~device:"CPU" [ 2; 3 ] 0 Float32 in
  let v = var "n" 1 8 in
  sink
    [
      param_like p 5;
      param_like v 6;
      param_like (bind v (i 3)) 7;
      placeholder_like p 9;
      alloc_like ~slot:10 p;
      alloc_like ~slot:11 ~addrspace:Local p;
    ]

let calls () =
  let r = range ~axis_type:Weak (Int 4) [ 0 ] in
  let idx = index (param [ 4 ] 0 Float32) [ r ] in
  let body = sink [ end_ (store idx (const_like idx (f 1.))) [ r ] ] in
  let a = param ~device:"CPU" [ 4 ] 0 Float32 in
  sink
    [
      call body [ a ];
      call ~name:"fill" ~precompile:true body [ a ];
      call (custom_function ~dtype:Int32 "f" [ int ~dtype:Uint64 0 ]) [];
      store_call a (param ~device:"CPU" [ 4 ] 1 Float32);
    ]

let custom_kernels () =
  let a = param ~device:"CPU" [ 4 ] 0 Float32 in
  let b = param ~device:"CPU" [ 4 ] 1 Float32 in
  let copy = function
    | [ x; y ] ->
        let r = range ~axis_type:Weak (Int 4) [ 0 ] in
        sink [ end_ (store (index x [ r ]) (load (index y [ r ]) [])) [ r ] ]
    | _ -> invalid_arg "copy takes two placeholders"
  in
  sink (custom_kernel [ a; b ] copy)

let variables_and_binding () =
  let n = var "n" 1 8 and m = var "m" 0 4 in
  let e = O.((bind n (i 3) * bind m (i 2)) + n) in
  let all_unbound, _ = unbind_all e in
  sink ([ bind n (i 3); unbound (bind n (i 3)); all_unbound ] @ variables e)

let clones () =
  let p = param ~device:"CPU" [ 2; 3 ] 0 Float32 in
  sink
    [
      clone p;
      clone ~device:"CUDA" p;
      empty_like p;
      empty_like ~dtype:Float16 ~device:"CUDA" p;
      empty ~device:"CPU" (ints [ 2; 3 ]) Int32;
    ]

let outputs () =
  let a = param ~device:"CPU" [ 4 ] 0 Float32 in
  sink
    (call_with_outputs ~name:"two" O.[ a + float 1.; a * float 2. ] [ a ]
    @ [ call_with_output ~precompile:true a [ a ] ])

(* Symbolic sizes *)

let symbolic_storage () =
  let n = var "n" 1 8 in
  sink
    [
      Shape.param ~shape:[ Int 2; Sym n ] 0 Float32;
      Shape.param ~shape:[ Sym n ] 1 Int32;
      alloc ~slot:8 [ Int 2; Sym n ] Weak_float;
    ]

let symbolic_outputs () =
  let a = param ~device:"CPU" [ 4 ] 0 Float32 in
  let n = var "n" 1 4 in
  let b = Shape.param ~device:"CPU" ~shape:[ Sym n ] 1 Float32 in
  let formal =
    Shape.param ~vmin_vmax:(i 1, i 8) ~name:"d" ~addrspace:Alu 0 Int32
  in
  sink
    (call_with_outputs ~output_pos:[ 0 ] O.[ b + float 1. ] [ a; b ]
    @ [
        call_with_output
          (expand (cast (float 1.) Float32) [ Sym formal ])
          [ int 5 ];
        empty ~device:"CPU" [ Int 2; Sym n ] Int32;
      ])

(* Storage made without a slot takes a number no other storage has. [minted]
   renumbers those of [sink] from 1000, in the order the graph lists them, as
   gen/uop/shape.py numbers tinygrad's. *)
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
    ("contraction", contraction, false);
    ("storage", storage, false);
    ("storage_like", storage_like, false);
    ("calls", calls, false);
    ("custom_kernels", custom_kernels, false);
    ("variables_and_binding", variables_and_binding, false);
    ("clones", clones, true);
    ("outputs", outputs, true);
    ("symbolic_storage", symbolic_storage, false);
    ("symbolic_outputs", symbolic_outputs, true);
  ]
