(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let sizes = [ 512 ]
let cat_tags = [ "cat" ]

let group_name category size dtype =
  Printf.sprintf "%s / %dx%d %s" category size size dtype

let ops_f32 ~size =
  let shape = [| size; size |] in
  let a = Nx.rand Nx.Float32 shape in
  let b = Nx.rand Nx.Float32 shape in
  let cond = Nx.less a b in
  let transposed_a = Nx.transpose a in
  let transposed_b = Nx.transpose b in
  let offset_a = Nx.shrink [| (1, size - 1); (0, size) |] a in
  let offset_b = Nx.shrink [| (1, size - 1); (0, size) |] b in
  let indices =
    Nx.create Nx.Int32 [| size |]
      (Array.init size (fun i -> Int32.of_int ((i * 37) mod size)))
  in
  [
    Thumper.group (group_name "elementwise" size "f32")
      [
        Thumper.bench "Add" (fun () -> Nx.add a b);
        Thumper.bench "Sub" (fun () -> Nx.sub a b);
        Thumper.bench "Mul" (fun () -> Nx.mul a b);
        Thumper.bench "Div" (fun () -> Nx.div a b);
        Thumper.bench "Maximum" (fun () -> Nx.maximum a b);
        Thumper.bench "Minimum" (fun () -> Nx.minimum a b);
        Thumper.bench "Less" (fun () -> Nx.less a b);
        Thumper.bench "Where" (fun () -> Nx.where cond a b);
      ];
    Thumper.group (group_name "unary" size "f32")
      [
        Thumper.bench "Neg" (fun () -> Nx.neg a);
        Thumper.bench "Abs" (fun () -> Nx.abs a);
        Thumper.bench "Sqrt" (fun () -> Nx.sqrt a);
        Thumper.bench "Exp" (fun () -> Nx.exp a);
        Thumper.bench "Log" (fun () -> Nx.log a);
        Thumper.bench "Sin" (fun () -> Nx.sin a);
        Thumper.bench "Cos" (fun () -> Nx.cos a);
      ];
    Thumper.group (group_name "reduction and scan" size "f32")
      [
        Thumper.bench "Sum" (fun () -> Nx.sum a);
        Thumper.bench "Sum axis 0" (fun () -> Nx.sum ~axes:[ 0 ] a);
        Thumper.bench "Sum axis 1" (fun () -> Nx.sum ~axes:[ 1 ] a);
        Thumper.bench "Max axis 1" (fun () -> Nx.max ~axes:[ 1 ] a);
        Thumper.bench "Cumsum axis 1" (fun () -> Nx.cumsum ~axis:1 a);
        Thumper.bench "Argmax axis 1" (fun () -> Nx.argmax ~axis:1 a);
      ];
    Thumper.group (group_name "structural" size "f32")
      [
        Thumper.bench "Matmul" (fun () -> Nx.matmul a b);
        Thumper.bench ~tags:cat_tags "Cat axis 0" (fun () ->
            Nx.concatenate ~axis:0 [ a; b ]);
        Thumper.bench ~tags:cat_tags "Cat axis 1" (fun () ->
            Nx.concatenate ~axis:1 [ a; b ]);
        Thumper.bench ~tags:cat_tags "Cat offset views axis 0" (fun () ->
            Nx.concatenate ~axis:0 [ offset_a; offset_b ]);
        Thumper.bench ~tags:cat_tags "Cat transposed views axis 1" (fun () ->
            Nx.concatenate ~axis:1 [ transposed_a; transposed_b ]);
        Thumper.bench "Contiguous transpose" (fun () ->
            Nx.contiguous transposed_a);
        Thumper.bench "Pad" (fun () ->
            Nx.pad [| (1, 1); (1, 1) |] 0.0 a);
        Thumper.bench "Cast f32 to f64" (fun () -> Nx.cast Nx.Float64 a);
        Thumper.bench "Take rows" (fun () ->
            Nx.take ~axis:0 ~indices a);
        Thumper.bench "Sort rows" (fun () -> Nx.sort a);
        Thumper.bench "Rand" (fun () -> Nx.rand Nx.Float32 shape);
        Thumper.bench "Randn" (fun () -> Nx.randn Nx.Float32 shape);
      ];
  ]

let ops_f64 ~size =
  let shape = [| size; size |] in
  let a = Nx.rand Nx.Float64 shape in
  let b = Nx.rand Nx.Float64 shape in
  let cond = Nx.less a b in
  let transposed_a = Nx.transpose a in
  let transposed_b = Nx.transpose b in
  let offset_a = Nx.shrink [| (1, size - 1); (0, size) |] a in
  let offset_b = Nx.shrink [| (1, size - 1); (0, size) |] b in
  let indices =
    Nx.create Nx.Int32 [| size |]
      (Array.init size (fun i -> Int32.of_int ((i * 37) mod size)))
  in
  [
    Thumper.group (group_name "elementwise" size "f64")
      [
        Thumper.bench "Add" (fun () -> Nx.add a b);
        Thumper.bench "Sub" (fun () -> Nx.sub a b);
        Thumper.bench "Mul" (fun () -> Nx.mul a b);
        Thumper.bench "Div" (fun () -> Nx.div a b);
        Thumper.bench "Maximum" (fun () -> Nx.maximum a b);
        Thumper.bench "Minimum" (fun () -> Nx.minimum a b);
        Thumper.bench "Less" (fun () -> Nx.less a b);
        Thumper.bench "Where" (fun () -> Nx.where cond a b);
      ];
    Thumper.group (group_name "unary" size "f64")
      [
        Thumper.bench "Neg" (fun () -> Nx.neg a);
        Thumper.bench "Abs" (fun () -> Nx.abs a);
        Thumper.bench "Sqrt" (fun () -> Nx.sqrt a);
        Thumper.bench "Exp" (fun () -> Nx.exp a);
        Thumper.bench "Log" (fun () -> Nx.log a);
        Thumper.bench "Sin" (fun () -> Nx.sin a);
        Thumper.bench "Cos" (fun () -> Nx.cos a);
      ];
    Thumper.group (group_name "reduction and scan" size "f64")
      [
        Thumper.bench "Sum" (fun () -> Nx.sum a);
        Thumper.bench "Sum axis 0" (fun () -> Nx.sum ~axes:[ 0 ] a);
        Thumper.bench "Sum axis 1" (fun () -> Nx.sum ~axes:[ 1 ] a);
        Thumper.bench "Max axis 1" (fun () -> Nx.max ~axes:[ 1 ] a);
        Thumper.bench "Cumsum axis 1" (fun () -> Nx.cumsum ~axis:1 a);
        Thumper.bench "Argmax axis 1" (fun () -> Nx.argmax ~axis:1 a);
      ];
    Thumper.group (group_name "structural" size "f64")
      [
        Thumper.bench "Matmul" (fun () -> Nx.matmul a b);
        Thumper.bench ~tags:cat_tags "Cat axis 0" (fun () ->
            Nx.concatenate ~axis:0 [ a; b ]);
        Thumper.bench ~tags:cat_tags "Cat axis 1" (fun () ->
            Nx.concatenate ~axis:1 [ a; b ]);
        Thumper.bench ~tags:cat_tags "Cat offset views axis 0" (fun () ->
            Nx.concatenate ~axis:0 [ offset_a; offset_b ]);
        Thumper.bench ~tags:cat_tags "Cat transposed views axis 1" (fun () ->
            Nx.concatenate ~axis:1 [ transposed_a; transposed_b ]);
        Thumper.bench "Contiguous transpose" (fun () ->
            Nx.contiguous transposed_a);
        Thumper.bench "Pad" (fun () ->
            Nx.pad [| (1, 1); (1, 1) |] 0.0 a);
        Thumper.bench "Cast f64 to f32" (fun () -> Nx.cast Nx.Float32 a);
        Thumper.bench "Take rows" (fun () ->
            Nx.take ~axis:0 ~indices a);
        Thumper.bench "Sort rows" (fun () -> Nx.sort a);
        Thumper.bench "Rand" (fun () -> Nx.rand Nx.Float64 shape);
        Thumper.bench "Randn" (fun () -> Nx.randn Nx.Float64 shape);
      ];
  ]

let benchmarks () =
  List.concat_map
    (fun size -> ops_f32 ~size @ ops_f64 ~size)
    sizes
