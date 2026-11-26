(* Tests of Tolk.Indexing: which nodes are gathers, each movement as index
   arithmetic, the selection of a lane, and the ranges run_rangeify gives a
   tensor graph. *)

open Windtrap
open Tolk

let z n = `Int (Bigint.of_int n)
let size shape = List.fold_left ( * ) 1 shape
let ints = List.map (fun n -> Ops.Int n)

let range ?(axis_type = Ops.Axis_type.Weak) n axis =
  Ops.range ~axis_type n [ axis ]

let ranges shape = List.mapi (fun axis n -> range (Int n) axis) shape

let all op u =
  List.filter (fun n -> Ops.op n = op) (Ops.toposort ~calls:Enter u)

let cpu = "CPU"
let p slot shape = Shape.param ~device:cpu ~shape:(ints shape) slot Float32

(* [index slot n] is a value of [n] integers, the storage of [slot] read as
   indices. *)
let index slot n =
  Ops.cast (Shape.param ~device:cpu ~shape:[ Int n ] slot Int32) Weak_int

(* is_gather *)

let gather u = satisfies ~claim:"a gather" Uops.uop Indexing.is_gather (u ())

let not_gather u =
  satisfies ~claim:"not a gather" Uops.uop
    (fun u -> not (Indexing.is_gather u))
    (u ())

let is_gather =
  group "is_gather"
    [
      cases "an index of a tensor by a value with axes is a gather" ~id:fst
        [
          ( "by a value of one axis",
            fun () -> Ops.index (p 1 [ 4; 3 ]) [ index 2 5 ] );
          ( "by a value of two axes",
            fun () ->
              Ops.index (p 1 [ 4 ])
                [ Shape.reshape (index 2 6) (ints [ 2; 3 ]) ] );
        ]
        (fun (_, u) -> gather u);
      cases "an index by scalars is not a gather" ~id:fst
        [
          ("by a constant", fun () -> Ops.index (p 1 [ 4; 3 ]) [ Ops.int 1 ]);
          ("by a range", fun () -> Ops.index (p 1 [ 4 ]) [ range (Int 4) 0 ]);
          ( "by a variable",
            fun () -> Ops.index (p 1 [ 4 ]) [ Ops.variable "i" (z 0) (z 3) ] );
        ]
        (fun (_, u) -> not_gather u);
      cases "a node other than an index is not a gather" ~id:fst
        [
          ("storage", fun () -> p 1 [ 4 ]);
          ("a movement", fun () -> Shape.reshape (p 1 [ 4 ]) (ints [ 2; 2 ]));
        ]
        (fun (_, u) -> not_gather u);
    ]

(* apply_movement_op *)

let n () = Ops.variable "n" (z 1) (z 8)

let moved file f =
  Golden.graph file (fun () ->
      let in_shape, m, idxs = f () in
      Ops.sink (Indexing.apply_movement_op in_shape m idxs))

let movement_goldens =
  group "indexes each movement as its golden states"
    [
      moved "movement_shrink.golden" (fun () ->
          ( ints [ 8; 6 ],
            Ops.Shrink [ (Int 2, Int 4); (Int 0, Int 6) ],
            ranges [ 4; 6 ] ));
      moved "movement_permute.golden" (fun () ->
          (ints [ 2; 3; 4 ], Ops.Permute [ 2; 0; 1 ], ranges [ 4; 2; 3 ]));
      moved "movement_expand.golden" (fun () ->
          (ints [ 3 ], Ops.Expand (ints [ 2; 4 ]), ranges [ 2; 4; 3 ]));
      moved "movement_expand_symbolic.golden" (fun () ->
          let n = n () in
          ( ints [ 3 ],
            Ops.Expand [ Sym n ],
            [ range (Sym n) 0; range (Int 3) 1 ] ));
      moved "movement_pad.golden" (fun () ->
          ( ints [ 4; 4 ],
            Ops.Pad [ (Int 1, Int 7); (Int 0, Int 4) ],
            ranges [ 7; 4 ] ));
      moved "movement_pad_end.golden" (fun () ->
          ( ints [ 4; 4 ],
            Ops.Pad [ (Int 0, Int 6); (Int 0, Int 4) ],
            ranges [ 6; 4 ] ));
      moved "movement_pad_symbolic.golden" (fun () ->
          let n_plus_2 = Ops.O.(n () + int 2) in
          ( [ Sym (n ()); Int 4 ],
            Ops.Pad [ (Int 1, Sym n_plus_2); (Int 0, Int 4) ],
            [ range (Sym n_plus_2) 0; range (Int 4) 1 ] ));
      moved "movement_reshape_flatten.golden" (fun () ->
          (ints [ 2; 3 ], Ops.Reshape (ints [ 6 ]), ranges [ 6 ]));
      moved "movement_reshape_unflatten.golden" (fun () ->
          (ints [ 6 ], Ops.Reshape (ints [ 2; 3 ]), ranges [ 2; 3 ]));
      moved "movement_reshape_regroup.golden" (fun () ->
          (ints [ 4; 6 ], Ops.Reshape (ints [ 3; 8 ]), ranges [ 3; 8 ]));
      moved "movement_reshape_unit_axes.golden" (fun () ->
          (ints [ 1; 4; 1 ], Ops.Reshape (ints [ 4 ]), ranges [ 4 ]));
      moved "movement_reshape_symbolic.golden" (fun () ->
          let n4 = Ops.O.(n () * int 4) in
          ([ Sym (n ()); Int 4 ], Ops.Reshape [ Sym n4 ], [ range (Sym n4) 0 ]));
      moved "movement_reshape_of_flipped.golden" (fun () ->
          ( ints [ 12 ],
            Ops.Reshape (ints [ 3; 4 ]),
            [ Ops.O.(int 2 - range (Int 3) 0); range (Int 4) 1 ] ));
      moved "movement_reshape_of_padded.golden" (fun () ->
          let r0 = range (Int 3) 0 in
          ( ints [ 8 ],
            Ops.Reshape (ints [ 2; 4 ]),
            [
              Shape.valid Ops.O.(r0 - int 1) Ops.O.(r0 >= int 1);
              range (Int 4) 1;
            ] ));
    ]

(* The index law of a movement: the source, stored row-major, read at the index
   that apply_movement_op gives for an element of the result, is that element of
   the moved source as the movement defines it (Tensors), and an element that a
   pad adds is one whose index is invalid. The source's element [j] holds [j +
   1], so that a padded [0] is told from a read.

   The composition law: the index of two movements of one kind, applied in turn,
   reads what the index of the one movement they make reads. *)

let concrete u =
  List.map
    (function Ops.Int n -> n | Sym _ -> invalid_arg "a symbolic shape")
    (Shape.shape u)

let source in_shape = Shape.param ~shape:[ Int (size in_shape) ] 0 Int32
let elements in_shape = Array.init (size in_shape) (fun j -> z (j + 1))
let view in_shape = Shape.reshape (source in_shape) (ints in_shape)
let shape_of in_shape m = concrete (Shape.mop (view in_shape) m)

let rec coords = function
  | [] -> [ [] ]
  | n :: rest ->
      List.concat_map
        (fun i -> List.map (fun c -> i :: c) (coords rest))
        (List.init n Fun.id)

(* [reads in_shape out_shape idxs] is what the source reads at [idxs], an index
   into it from the ranges of [out_shape], at each element of [out_shape] in
   row-major order. *)
let reads in_shape out_shape idxs =
  let offset =
    List.fold_left2
      (fun acc n i -> Ops.O.((acc * int n) + i))
      (Ops.int 0) in_shape idxs
  in
  let read = Ops.index (source in_shape) [ offset ] in
  let names = List.map (fun r -> Option.get (Interpreter.name r)) in
  List.map
    (fun c ->
      let vars = List.combine (names (ranges out_shape)) (List.map z c) in
      Interpreter.eval ~vars ~buffers:[ (0, elements in_shape) ] read)
    (coords out_shape)

(* [moved in_shape m] is the elements of the movement [m] of the source. The
   source holds no zero, so a zero is an element a pad adds, which a read at an
   invalid index gives. *)
let moved in_shape m =
  let moved = Shape.mop (view in_shape) m in
  Array.to_list (Tensors.eval ~buffers:[ (0, elements in_shape) ] moved)

let apply in_shape m idxs = Indexing.apply_movement_op (ints in_shape) m idxs

let index_law (in_shape, m) =
  let out_shape = shape_of in_shape m in
  cover "an empty axis" (List.mem 0 in_shape || List.mem 0 out_shape);
  cover "an element is read" (not (List.mem 0 out_shape));
  equal (list Dtypes.const) (moved in_shape m)
    (reads in_shape out_shape (apply in_shape m (ranges out_shape)))

(* A pad's law says something only where the pad adds elements. *)
let pad_law (in_shape, m) =
  cover "an element is padded" (List.mem (`Int Bigint.zero) (moved in_shape m));
  index_law (in_shape, m)

let composite (m1 : Ops.movement) (m2 : Ops.movement) : Ops.movement =
  let int = function Ops.Int n -> n | Sym _ -> invalid_arg "symbolic" in
  match (m1, m2) with
  | Reshape _, Reshape s -> Reshape s
  | Expand a1, Expand a2 -> Expand (a2 @ a1)
  | Pad p1, Pad p2 ->
      Pad
        (List.map2
           (fun (b1, _) (b2, s) -> (Ops.Int (int b1 + int b2), s))
           p1 p2)
  | Shrink b1, Shrink b2 ->
      Shrink
        (List.map2
           (fun (s1, _) (s2, n) -> (Ops.Int (int s1 + int s2), n))
           b1 b2)
  | Permute p1, Permute p2 -> Permute (List.map (List.nth p1) p2)
  | _ -> invalid_arg "movements of two kinds"

let composition_law (in_shape, m1, m2) =
  let mid = shape_of in_shape m1 in
  let out_shape = shape_of mid m2 in
  cover "an empty axis" (List.mem 0 in_shape || List.mem 0 out_shape);
  cover "an element is read" (not (List.mem 0 out_shape));
  let rs = ranges out_shape in
  equal (list Dtypes.const)
    (reads in_shape out_shape (apply in_shape (composite m1 m2) rs))
    (reads in_shape out_shape (apply in_shape m1 (apply mid m2 rs)))

(* Generators: shapes of up to three axes of up to four elements, an axis empty
   one time in seven, and a movement of each kind that fits a shape. *)

let pp_sint ppf = function
  | Ops.Int n -> Format.pp_print_int ppf n
  | Sym u -> Format.pp_print_string ppf (Render.render u)

let pp_list pp ppf l =
  Format.fprintf ppf "[%a]"
    (Format.pp_print_list ~pp_sep:(fun ppf () -> Format.fprintf ppf "; ") pp)
    l

let pp_movement ppf (m : Ops.movement) =
  let pp_pair ppf (a, b) = Format.fprintf ppf "(%a, %a)" pp_sint a pp_sint b in
  match m with
  | Reshape s -> Format.fprintf ppf "Reshape %a" (pp_list pp_sint) s
  | Expand s -> Format.fprintf ppf "Expand %a" (pp_list pp_sint) s
  | Pad p -> Format.fprintf ppf "Pad %a" (pp_list pp_pair) p
  | Shrink b -> Format.fprintf ppf "Shrink %a" (pp_list pp_pair) b
  | Permute o -> Format.fprintf ppf "Permute %a" (pp_list Format.pp_print_int) o

let rec each = function
  | [] -> Gen.constant []
  | g :: gs -> Gen.map (fun (x, xs) -> x :: xs) (Gen.pair g (each gs))

let axis = Gen.frequency [ (1, Gen.constant 0); (6, Gen.int_range 1 4) ]
let shape = Gen.list ~size:(Gen.int_range 1 3) axis
let divisors n = List.filter (fun d -> n mod d = 0) (List.init n succ)

(* [factors k n] draws [k] axes of [n] elements in all; of none, at least one
   axis is empty. *)
let rec factors k n =
  let open Gen in
  if k = 1 then constant [ n ]
  else if n = 0 then
    let* d = axis in
    let+ rest = factors (k - 1) (if d = 0 then 1 else 0) in
    d :: rest
  else
    let* d = of_list (divisors n) in
    let+ rest = factors (k - 1) (n / d) in
    d :: rest

let movement kind in_shape : Ops.movement Gen.t =
  let open Gen in
  let per_axis f = each (List.map f in_shape) in
  match kind with
  | `Reshape ->
      let* k = int_range 1 3 in
      let+ s = factors k (size in_shape) in
      Ops.Reshape (ints s)
  | `Expand ->
      let+ added = list ~size:(int_range 0 2) (int_range 1 3) in
      Ops.Expand (ints added)
  | `Pad ->
      let+ p =
        per_axis (fun n ->
            let+ before = int_range 0 2 and+ after = int_range 0 2 in
            (Ops.Int before, Ops.Int (n + before + after)))
      in
      Ops.Pad p
  | `Shrink ->
      let empty n =
        let+ start = int_range 0 n in
        (Ops.Int start, Ops.Int 0)
      in
      let nonempty n =
        let* start = int_range 0 (n - 1) in
        let+ length = int_range 1 (n - start) in
        (Ops.Int start, Ops.Int length)
      in
      let+ b =
        per_axis (fun n ->
            if n = 0 then empty n
            else frequency [ (1, empty n); (6, nonempty n) ])
      in
      Ops.Shrink b
  | `Permute ->
      let+ o = permutation (List.init (List.length in_shape) Fun.id) in
      Ops.Permute o

let one kind =
  Gen.(
    let* s = shape in
    let+ m = movement kind s in
    (s, m))
  |> Gen.with_pp (fun ppf (s, m) ->
      Format.fprintf ppf "%a of %a" pp_movement m
        (pp_list Format.pp_print_int)
        s)

let two kind =
  Gen.(
    let* s = shape in
    let* m1 = movement kind s in
    let+ m2 = movement kind (shape_of s m1) in
    (s, m1, m2))
  |> Gen.with_pp (fun ppf (s, m1, m2) ->
      Format.fprintf ppf "%a of %a of %a" pp_movement m2 pp_movement m1
        (pp_list Format.pp_print_int)
        s)

(* One memo gives, for each of several movements of any kind, at a first call
   and at a repeat, what no memo gives: over the result's ranges, and over its
   first element's constant index. *)
let memo_law moves =
  let kind (m : Ops.movement) =
    match m with
    | Reshape _ -> 0
    | Expand _ -> 1
    | Pad _ -> 2
    | Shrink _ -> 3
    | Permute _ -> 4
  in
  cover "movements of two kinds"
    (List.length
       (List.sort_uniq compare (List.map (fun (_, m) -> kind m) moves))
    > 1);
  let calls =
    List.concat_map
      (fun (in_shape, m) ->
        let out_shape = shape_of in_shape m in
        List.map
          (fun idxs -> (in_shape, m, idxs))
          [ ranges out_shape; List.map (fun _ -> Ops.int 0) out_shape ])
      moves
  in
  let memo = Indexing.memo () in
  let check msg (in_shape, m, idxs) =
    equal (list Uops.uop) ~msg (apply in_shape m idxs)
      (Indexing.apply_movement_op ~memo (ints in_shape) m idxs)
  in
  List.iter (check "first") calls;
  List.iter (check "repeat") calls

let mixed_movements =
  Gen.(
    list ~size:(int_range 2 6)
      (let* kind = of_list [ `Shrink; `Permute; `Expand; `Pad; `Reshape ] in
       one kind))

let apply_movement_op =
  group "apply_movement_op"
    [
      movement_goldens;
      prop "a shrink's index reads the element it moves" (one `Shrink) index_law;
      prop "a permute's index reads the element it moves" (one `Permute)
        index_law;
      prop "an expand's index reads the element it moves" (one `Expand)
        index_law;
      prop "a pad's index reads the element it moves" (one `Pad) pad_law;
      prop "a reshape's index reads the element it moves" (one `Reshape)
        index_law;
      prop "the indices of two shrinks compose" (two `Shrink) composition_law;
      prop "the indices of two permutes compose" (two `Permute) composition_law;
      prop "the indices of two expands compose" (two `Expand) composition_law;
      prop "the indices of two pads compose" (two `Pad) composition_law;
      prop "the indices of two reshapes compose" (two `Reshape) composition_law;
      prop
        "with a memo, each index is the one without it, at a first call and a \
         repeat"
        mixed_movements memo_law;
    ]

(* stack_select *)

(* [lanes n] is [n] lanes, lane [k] holding [k]. The index ranges far beyond
   them, so that its bounds fold no selection. *)
let lanes n = List.init n (fun k -> Ops.float ~dtype:Float32 (float_of_int k))
let lane_index () = Ops.variable "i" (z (-4096)) (z 4096)

let selected n i =
  Interpreter.eval
    ~vars:[ ("i", z i) ]
    (Indexing.stack_select (lane_index ()) (lanes n))

let pp_lanes_at ppf (n, i) = Format.fprintf ppf "%d lanes at %d" n i

let outside =
  Gen.(
    let* n = int_range 1 40 in
    let+ i = one_of [ int_range (-1000) (-1); int_range n (n + 1000) ] in
    (n, i))
  |> Gen.with_pp pp_lanes_at

let stack_select =
  group "stack_select"
    [
      test "of two lanes is where (i mod 2 < 1) l0 l1" (fun () ->
          let i = lane_index () and l0 = Ops.float 1. and l1 = Ops.float 2. in
          equal Uops.uop
            (Ops.where Ops.O.(i % int 2 < int 1) l0 l1)
            (Indexing.stack_select i [ l0; l1 ]));
      prop "selects the lane at each index" ~examples:[ 1; 2; 3; 257 ]
        (Gen.int_range 1 40) (fun n ->
          List.iter
            (fun i ->
              equal Dtypes.const ~msg:(Printf.sprintf "at %d" i)
                (`Float (float_of_int i))
                (selected n i))
            (List.init n Fun.id));
      prop "an index outside the lanes selects one of them" outside
        (fun (n, i) ->
          cover "below the lanes" (i < 0);
          cover "above the lanes" (i >= n);
          satisfies ~claim:"one of the lanes" Dtypes.const
            (fun v ->
              List.init n (fun k -> `Float (float_of_int k)) |> List.mem v)
            (selected n i));
      test "raises on no lanes" (fun () ->
          let i = lane_index () in
          raises_match Exn.invalid_arg (fun () -> Indexing.stack_select i []));
    ]

(* run_rangeify *)

let program name = Golden.sink (name ^ ".golden")

let recorded =
  group "rangeifies each recorded graph as its golden states"
    (List.map
       (fun name ->
         Golden.graph (name ^ "_rangeified.golden") (fun () ->
             Indexing.run_rangeify ~cores:Cores.host (program name)))
       [
         "elementwise_three";
         "mulacc";
         "binop_reshape";
         "binop_permute";
         "shared_sum";
         "reduce_unary";
         "reduce_reshape_binop";
         "reduce_permute_binop";
         "reduce_permute_nofuse";
         "permute_through_reshape";
         "shrink_fuse";
         "multistage_reduce";
         "reduce_shrink";
         "contiguous_add";
         "reshape_chain";
         "children_dont_push";
         "rmsnorm";
         "add";
         "sum";
         "sum_all";
         "matmul";
         "double_matmul";
         "matmul_relu_cat";
         "einsum";
         "where";
         "cast_half";
         "permute";
         "flip";
         "shrink";
         "pad";
         "reshape";
         "reshape_split";
         "expand";
         "outer";
         "pad_reshape";
         "flip_reshape";
         "pad_reduce";
         "repeat";
         "roll";
         "interpolate";
         "broadcast_reduce";
         "softmax";
         "layernorm";
         "standardize";
         "diamond";
         "attention";
         "argmax";
         "two_consumers";
         "two_consumers_permuted";
         "shared_view";
         "conv";
         "conv_bn_relu";
         "maxpool";
         "avgpool";
         "cat";
         "stack";
         "stack_eight";
         "stack_twelve";
         "tensor_of_list";
         "arange";
         "cumsum";
         "cumsum_rows";
         "triu";
         "embedding";
         "gather";
         "two_gathers";
         "sort";
         "topk";
         "select_no_lane";
         "select_no_lane_rows";
         "contiguous";
         "assign";
         "assign_permuted";
         "assign_double_diamond";
         "assign_reads_behind_other_after";
         "setitem";
         "custom_kernel";
         "variable_shrink";
         "variable_offset";
         "variable_data_and_shape";
         "variable_stack";
         "copy";
         "bitcast_copy";
         (* hand-built graphs *)
         "expand_by_range";
         "expand_by_range_stored";
         "scalar_and_wide_uses";
       ])

(* The law that ranges keep what a program writes: where the rangeified graph is
   one kernel that reads only what it was given, the writes of its stores
   (Interpreter) are what the program's tensors write (Tensors). Storage [s]
   holds small integers, so that sums are exact in any order. *)

let filled u =
  List.filter_map
    (fun n ->
      match (Ops.op n, Ops.arg n) with
      | (Param | Buffer | Alloc), Param { slot; dtype; _ }
        when Shape.max_shape n <> [] ->
          let size = Shape.max_numel n in
          let element j : Dtype.value =
            let k = (((j * 7) + (slot * 3)) mod 11) - 3 in
            if Dtype.is_float dtype then `Float (float_of_int k)
            else if Dtype.equal dtype Bool then `Bool (k > 0)
            else z k
          in
          Some (slot, Array.init size element)
      | _ -> None)
    (Ops.toposort ~calls:Enter u)

let write = triple int int Dtypes.value

let keeps_writes ?buffers sink =
  let buffers = Option.value buffers ~default:(filled sink) in
  equal (list write)
    (Tensors.writes ~buffers sink)
    (Interpreter.writes ~buffers (Indexing.run_rangeify ~cores:Cores.host sink))

let writes =
  cases "keeps what a recorded program of one kernel writes" ~id:Fun.id
    [
      "elementwise_three";
      "mulacc";
      "binop_reshape";
      "binop_permute";
      "shared_sum";
      "reduce_unary";
      "reduce_reshape_binop";
      "reduce_permute_binop";
      "reduce_permute_nofuse";
      "permute_through_reshape";
      "shrink_fuse";
      "multistage_reduce";
      "reduce_shrink";
      "reshape_chain";
      "add";
      "sum";
      "sum_all";
      "matmul";
      "einsum";
      "where";
      "cast_half";
      "permute";
      "flip";
      "shrink";
      "pad";
      "reshape";
      "reshape_split";
      "expand";
      "outer";
      "pad_reshape";
      "flip_reshape";
      "pad_reduce";
      "repeat";
      "roll";
      "conv";
      "avgpool";
      "cat";
      "stack";
      "stack_eight";
      "stack_twelve";
      "arange";
      "cumsum";
      "cumsum_rows";
      "assign";
      "assign_permuted";
    ] (fun name -> keeps_writes (program name))

(* Hand-built graphs, each the store of one value into the storage of slot [0]
   on the CPU, that state one rule of run_rangeify at a time. *)

let stored value =
  let out =
    Shape.param ~device:cpu ~shape:[ Int (size (concrete value)) ] 0 Float32
  in
  Ops.sink
    [
      Ops.after out [ Ops.store (Shape.reshape out (Shape.shape value)) value ];
    ]

let rangeified value = Indexing.run_rangeify ~cores:Cores.host (stored value)
let axis_type = Testable.make ~pp:Ops.Axis_type.pp ~equal:Ops.Axis_type.equal

let range_of r =
  (Ops.axis_id r, Ops.axis_type r, Bigint.to_int (Shape.to_z (Ops.nth r 0)))

let ranges_of u = List.sort compare (List.map range_of (all Range u))
let ranges_witness = list (triple (list int) axis_type int)

(* The ranges a stage stores its source over. *)
let staged_over s = List.length (List.tl (Ops.src s))
let exp2 = Ops.exp2
let ops = list (Testable.make ~pp:Op.pp ~equal:( = ))

(* The ops of the values stored whole under [value]. *)
let staged value =
  List.map (fun s -> Ops.op (Ops.nth s 0)) (all Stage (rangeified value))

(* A sum over the last axis of 8, kept as an axis of one element. *)
let row_sums () =
  Shape.reshape (Shape.rop (p 1 [ 4; 8 ]) Add [ 1 ]) (ints [ 4; 1 ])

let opts u =
  List.map
    (fun s ->
      match Ops.arg s with Bufferize o -> o | _ -> fail "a stage has options")
    (all Stage u)

let opts_witness =
  Testable.make ~pp:(fun ppf o -> Ops.pp_arg ppf (Bufferize o)) ~equal:( = )

(* [into_dest value] stores [value dest] into [dest], a 4 by 4 view of the
   storage of slot [0]. *)
let into_dest value =
  let out = Shape.param ~device:cpu ~shape:[ Int 16 ] 0 Float32 in
  let dest = Shape.reshape out (ints [ 4; 4 ]) in
  Ops.sink [ Ops.after out [ Ops.store dest (value dest) ] ]

(* The op of the value each store stores, and whether it is read from a
   stage. *)
let stored_values u =
  List.map
    (fun st ->
      let v = Ops.nth st 1 in
      if Ops.op v = Index && Ops.op (Ops.nth v 0) = Stage then
        (true, Ops.op (Ops.nth (Ops.nth v 0) 0))
      else (false, Ops.op v))
    (all Store u)

let staged_op = list (pair bool (Testable.make ~pp:Op.pp ~equal:( = )))

(* Stacks of constants. *)
let constants n =
  Shape.stack
    (List.init n (fun k ->
         Ops.float ~dtype:Float32 (float_of_int ((3 * k) + 7))))

let rec bits n = if n = 0 then 0 else 1 + bits (n lsr 1)

(* The greatest number of float selections on a path from [u] to a leaf. *)
let depth u =
  let depths = Ops.Tbl.create 64 in
  List.iter
    (fun n ->
      let below =
        List.fold_left (fun d s -> max d (Ops.Tbl.find depths s)) 0 (Ops.src n)
      in
      let own = if Ops.op n = Where && Ops.dtype n = Float32 then 1 else 0 in
      Ops.Tbl.replace depths n (below + own))
    (Ops.toposort ~calls:Enter u);
  Ops.Tbl.find depths u

let run_rangeify =
  group "run_rangeify"
    [
      recorded;
      writes;
      test "a node stored whole gets a new range per axis of size other than 1"
        (fun () ->
          equal ranges_witness
            [ ([ 0 ], Weak, 4); ([ 1 ], Weak, 3) ]
            (ranges_of (rangeified (exp2 (p 1 [ 4; 1; 3 ])))));
      test "a node stored whole gets a range for an axis of size 0" (fun () ->
          equal ranges_witness
            [ ([ 0 ], Weak, 0) ]
            (ranges_of (rangeified (exp2 (p 1 [ 0 ])))));
      (* The store's axes of 4 and 3 take the walk's first new ranges, 0 and 1.
         The stored expand's leading axis has the size of the golden's loop
         range 7, and its axis of 3 takes the next new range, 2. *)
      test "a stored axis whose size is a range is indexed by that range"
        (fun () ->
          let u =
            Indexing.run_rangeify ~cores:Cores.host
              (program "expand_by_range_stored")
          in
          equal
            (list (list (list int)))
            [ [ [ 7 ]; [ 2 ] ] ]
            (List.map
               (fun s -> List.map Ops.axis_id (List.tl (Ops.src s)))
               (all Stage u)));
      test "a node whose consumers index it differently is stored on every axis"
        (fun () ->
          let x = exp2 (p 1 [ 4; 4 ]) in
          let stages =
            all Stage (rangeified Ops.O.(x + Shape.permute x [ 1; 0 ]))
          in
          equal (list int) [ 2 ] (List.map staged_over stages));
      test
        "a node whose consumers index it differently on one axis is stored on \
         every axis" (fun () ->
          let x = exp2 (p 1 [ 4; 4 ]) in
          let stages = all Stage (rangeified Ops.O.(x + Shape.flip x [ 1 ])) in
          equal (list int) [ 2 ] (List.map staged_over stages));
      test "an elementwise node that an operation broadcasts is not stored"
        (fun () ->
          equal ops [] (staged Ops.O.(exp2 (p 1 [ 4; 1 ]) + p 2 [ 4; 8 ])));
      test "a reduction that an operation broadcasts is stored" (fun () ->
          equal ops [ Reduce ] (staged Ops.O.(row_sums () + p 2 [ 4; 8 ])));
      test "a reduction below a broadcast elementwise node is stored" (fun () ->
          equal ops [ Reduce ]
            (staged Ops.O.(exp2 (row_sums ()) + p 2 [ 4; 8 ])));
      test "an elementwise node that an expand broadcasts is stored" (fun () ->
          let e = Shape.expand (exp2 (p 1 [ 4 ])) (ints [ 3; 4 ]) in
          equal ops [ Exp2 ] (staged Ops.O.(e + p 2 [ 3; 4 ])));
      test "a node that an expand by a range broadcasts is not stored"
        (fun () ->
          equal (list Uops.uop) []
            (all Stage
               (Indexing.run_rangeify ~cores:Cores.host
                  (program "expand_by_range"))));
      test "a node whose consumers index it alike takes their ranges" (fun () ->
          let x = exp2 (p 1 [ 4; 4 ]) in
          equal (list Uops.uop) [] (all Stage (rangeified Ops.O.(x + x))));
      (* [x] is read at the index of the sum's range by both pads, valid below 4
         through one and below 3 through the other. *)
      test
        "a node whose consumers index it alike but for validity takes their \
         ranges" (fun () ->
          let x = exp2 (p 1 [ 4 ]) in
          let padded = Shape.pad x [ Some (Int 0, Int 2) ] in
          let cut =
            Shape.pad
              (Shape.shrink x [ Some (Int 0, Int 3) ])
              [ Some (Int 0, Int 3) ]
          in
          equal (list Uops.uop) [] (all Stage (rangeified Ops.O.(padded + cut))));
      test "new ranges are weak, numbered from 0 as the walk creates them"
        (fun () ->
          equal ranges_witness
            [ ([ 0 ], Weak, 4); ([ 1 ], Weak, 8) ]
            (ranges_of (rangeified (Shape.rop (p 1 [ 4; 8 ]) Add [ 1 ]))));
      (* The kernel split stages it, where the kernel's indices show it reads
         other elements than it writes. *)
      test "a value stored into a destination it reads is not stored whole"
        (fun () ->
          equal staged_op
            [ (false, Add) ]
            (stored_values
               (Indexing.run_rangeify ~cores:Cores.host
                  (into_dest (fun dest ->
                       Ops.O.(Shape.permute dest [ 1; 0 ] + float 1.))))));
      test
        "a value stored into a destination it does not read is not stored whole"
        (fun () ->
          equal staged_op
            [ (false, Add) ]
            (stored_values
               (Indexing.run_rangeify ~cores:Cores.host
                  (into_dest (fun _ -> Ops.O.(p 1 [ 4; 4 ] + float 1.))))));
      test
        "a stage of a computed value lives on its source's device, where later \
         passes may inline it" (fun () ->
          let x = exp2 (p 1 [ 4; 4 ]) in
          equal (list opts_witness)
            [ { device = Some cpu; addrspace = Global; keep = Removable } ]
            (opts (rangeified Ops.O.(x + Shape.permute x [ 1; 0 ]))));
      test "a stored store is closed by an end over its ranges" (fun () ->
          let ends = all End (rangeified (exp2 (p 1 [ 4; 3 ]))) in
          equal (list int) [ 2 ]
            (List.map (fun e -> List.length (Ops.src e) - 1) ends));
      test "a gather reads its source at the element its index holds" (fun () ->
          let table = Array.init 4 (fun k -> `Float (float_of_int (k + 1))) in
          let rows = Array.map z [| 3; 0; 2; 2; 1 |] in
          keeps_writes
            ~buffers:[ (0, Array.make 5 (`Float 0.)); (1, table); (2, rows) ]
            (stored (Ops.index (p 1 [ 4 ]) [ index 2 5 ])));
      test "a reduction of leading axes becomes a reduction over their ranges"
        (fun () ->
          let reductions =
            all Reduce (rangeified (Shape.rop (p 1 [ 4; 8 ]) Add [ 1 ]))
          in
          equal
            (list (pair ops (Testable.make ~pp:Ops.pp_arg ~equal:( = ))))
            [ ([ Range ], Reduce { op = Add; num_axes = 0 }) ]
            (List.map
               (fun r -> (List.map Ops.op (List.tl (Ops.src r)), Ops.arg r))
               reductions));
      test "a reduction leaves out the ranges of its axes of one element"
        (fun () ->
          let reductions =
            all Reduce (rangeified (Shape.rop (p 1 [ 4; 1; 8 ]) Add [ 1; 2 ]))
          in
          equal
            (list (list int))
            [ [ 8 ] ]
            (List.map
               (fun r ->
                 List.map
                   (fun s -> Bigint.to_int (Shape.to_z (Ops.nth s 0)))
                   (List.tl (Ops.src r)))
               reductions));
      test "a pad becomes a selection of its source and of 0" (fun () ->
          let u = rangeified (Shape.pad (p 1 [ 4 ]) [ Some (Int 1, Int 2) ]) in
          equal (list Uops.uop) [] (all Pad u);
          equal (list Dtypes.const)
            [ `Float 0. ]
            (List.filter_map
               (fun w ->
                 if Ops.dtype w = Float32 then Some (Ops.value (Ops.nth w 2))
                 else None)
               (all Where u)));
      cases "an index of a value by scalars reads the element they select"
        ~id:fst
        [
          ("1 of 3", fun v -> Ops.index v [ Ops.int 1 ]);
          ( "0 of 3 padded by 1 each side",
            fun v ->
              Ops.index (Shape.pad v [ Some (Int 1, Int 1) ]) [ Ops.int 0 ] );
          ( "2 of 3 padded by 1 each side",
            fun v ->
              Ops.index (Shape.pad v [ Some (Int 1, Int 1) ]) [ Ops.int 2 ] );
          ( "4 of 3 padded by 1 each side",
            fun v ->
              Ops.index (Shape.pad v [ Some (Int 1, Int 1) ]) [ Ops.int 4 ] );
        ]
        (fun (_, index) ->
          keeps_writes (stored (index Ops.O.(p 1 [ 3 ] + float 1.))));
      test "a stack becomes a selection of its sources" (fun () ->
          let u =
            rangeified (Shape.stack [ p 1 [ 4 ]; p 2 [ 4 ]; p 3 [ 4 ] ])
          in
          equal (list Uops.uop) []
            (List.filter (fun s -> Ops.dtype s = Float32) (all Stack u)));
      cases "a stack of constants writes each of them" ~id:string_of_int
        [ 1; 8; 9; 17; 100 ] (fun n -> keeps_writes (stored (constants n)));
      (* A tree of selections on the bits of an index below [n] has a level per
         bit of [n - 1]. *)
      cases "a stack's selection nests to a depth logarithmic in its sources"
        ~id:string_of_int [ 2; 8; 9; 17; 1024 ] (fun n ->
          at_most int ~than:(bits (n - 1)) (depth (rangeified (constants n))));
      test "movements are removed" (fun () ->
          let u =
            rangeified
              (Shape.flip
                 (Shape.permute
                    (Shape.shrink (p 1 [ 6; 4 ]) [ Some (Int 1, Int 5); None ])
                    [ 1; 0 ])
                 [ 0 ])
          in
          equal (list Uops.uop) []
            (List.filter
               (fun n -> Op.Set.mem (Ops.op n) Op.Set.movement)
               (Ops.toposort ~calls:Enter u)));
      test "a stage of a value placed nowhere lives on the sink's device"
        (fun () ->
          let c =
            exp2 (Shape.expand (Ops.float ~dtype:Float32 2.) (ints [ 4; 4 ]))
          in
          equal (list opts_witness)
            [ { device = Some cpu; addrspace = Global; keep = Removable } ]
            (opts (rangeified Ops.O.(c + Shape.permute c [ 1; 0 ]))));
      (* [c] and the sum are each stored whole, the sum since the two stores
         index it by ranges of their own, so each has a stage on each device. *)
      test "a stage read on two devices is a stage on each" (fun () ->
          let c =
            exp2 (Shape.expand (Ops.float ~dtype:Float32 2.) (ints [ 4; 4 ]))
          in
          let value = Ops.O.(c + Shape.permute c [ 1; 0 ]) in
          let into device slot =
            let out = Shape.param ~device ~shape:[ Int 16 ] slot Float32 in
            Ops.after out
              [ Ops.store (Shape.reshape out (ints [ 4; 4 ])) value ]
          in
          let u =
            Indexing.run_rangeify ~cores:Cores.host
              (Ops.sink [ into cpu 0; into "CPU:1" 1 ])
          in
          equal
            (slist (option string) compare)
            [ Some cpu; Some cpu; Some "CPU:1"; Some "CPU:1" ]
            (List.map (fun (o : Ops.bufferize_opts) -> o.device) (opts u)));
      test "a pad that gets no ranges stays a pad" (fun () ->
          let pad = Shape.pad (p 1 [ 4 ]) [ Some (Int 1, Int 2) ] in
          equal (list Uops.uop) [ pad ]
            (all Pad
               (Indexing.run_rangeify ~cores:Cores.host (Ops.sink [ pad ]))));
      test "a reshape that gets no ranges stays a reshape" (fun () ->
          let reshape = Shape.reshape (p 1 [ 4; 3 ]) (ints [ 12 ]) in
          mem Uops.uop reshape
            (all Reshape
               (Indexing.run_rangeify ~cores:Cores.host (Ops.sink [ reshape ]))));
      test "raises on a reduction of leading axes without ranges" (fun () ->
          let sink = Ops.sink [ Shape.rop (p 1 [ 4; 8 ]) Add [ 1 ] ] in
          raises_match Exn.invalid_arg (fun () ->
              Indexing.run_rangeify ~cores:Cores.host sink));
      test "raises on a gather of two indices" (fun () ->
          let sink =
            stored (Ops.index (p 1 [ 4; 3 ]) [ index 2 5; index 3 5 ])
          in
          raises_match Exn.invalid_arg (fun () ->
              Indexing.run_rangeify ~cores:Cores.host sink));
    ]

let () =
  exit
    (run "Tolk.Indexing"
       [ is_gather; apply_movement_op; stack_select; run_rangeify ])
