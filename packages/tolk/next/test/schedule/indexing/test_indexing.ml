(* Tests of Tolk_next.Indexing: each movement as index arithmetic, its laws over
   generated shapes, the ranges of the programs tinygrad schedules and what they
   write, and the rules of run_rangeify. *)

open Windtrap
open Tolk_next

let z n = `Int (Z.of_int n)
let size shape = List.fold_left ( * ) 1 shape
let ints = List.map (fun n -> Ops.Int n)

let range ?(axis_type = Ops.Axis_type.Weak) n axis =
  Ops.range ~axis_type n [ axis ]

let ranges shape = List.mapi (fun axis n -> range (Int n) axis) shape
let n = Ops.variable "n" (z 1) (z 8)

(* Movements *)

let moved file in_shape m idxs =
  Golden.graph file (fun () ->
      Ops.sink (Indexing.apply_movement_op in_shape m idxs))

let movements =
  let r0 = range (Int 3) 0 and r1 = range (Int 4) 1 in
  let n_plus_2 = Ops.O.(n + Ops.int 2) in
  group "apply_movement_op"
    [
      moved "movement_shrink.golden"
        (ints [ 8; 6 ])
        (Shrink [ (Int 2, Int 4); (Int 0, Int 6) ])
        (ranges [ 4; 6 ]);
      moved "movement_permute.golden"
        (ints [ 2; 3; 4 ])
        (Permute [ 2; 0; 1 ])
        (ranges [ 4; 2; 3 ]);
      moved "movement_flip.golden"
        (ints [ 4; 5 ])
        (Flip [ true; false ])
        (ranges [ 4; 5 ]);
      moved "movement_expand.golden" (ints [ 3 ])
        (Expand (ints [ 2; 4 ]))
        (ranges [ 2; 4; 3 ]);
      moved "movement_expand_symbolic.golden" (ints [ 3 ]) (Expand [ Sym n ])
        [ range (Sym n) 0; range (Int 3) 1 ];
      moved "movement_pad.golden"
        (ints [ 4; 4 ])
        (Pad [ (Int 1, Int 7); (Int 0, Int 4) ])
        (ranges [ 7; 4 ]);
      moved "movement_pad_end.golden"
        (ints [ 4; 4 ])
        (Pad [ (Int 0, Int 6); (Int 0, Int 4) ])
        (ranges [ 6; 4 ]);
      moved "movement_pad_symbolic.golden" [ Sym n; Int 4 ]
        (Pad [ (Int 1, Sym n_plus_2); (Int 0, Int 4) ])
        [ range (Sym n_plus_2) 0; range (Int 4) 1 ];
      moved "movement_reshape_flatten.golden"
        (ints [ 2; 3 ])
        (Reshape (ints [ 6 ]))
        (ranges [ 6 ]);
      moved "movement_reshape_unflatten.golden" (ints [ 6 ])
        (Reshape (ints [ 2; 3 ]))
        (ranges [ 2; 3 ]);
      moved "movement_reshape_regroup.golden"
        (ints [ 4; 6 ])
        (Reshape (ints [ 3; 8 ]))
        (ranges [ 3; 8 ]);
      moved "movement_reshape_unit_axes.golden"
        (ints [ 1; 4; 1 ])
        (Reshape (ints [ 4 ]))
        (ranges [ 4 ]);
      moved "movement_reshape_symbolic.golden" [ Sym n; Int 4 ]
        (Reshape [ Sym Ops.O.(n * Ops.int 4) ])
        [ range (Sym Ops.O.(n * Ops.int 4)) 0 ];
      moved "movement_reshape_of_flipped.golden" (ints [ 12 ])
        (Reshape (ints [ 3; 4 ]))
        [ Ops.O.(Ops.int 2 - r0); r1 ];
      moved "movement_reshape_of_padded.golden" (ints [ 8 ])
        (Reshape (ints [ 2; 4 ]))
        [ Ops.valid Ops.O.(r0 - Ops.int 1) Ops.O.(r0 >= Ops.int 1); r1 ];
    ]

(* Movement laws

   The index law of a movement: the source, stored row-major, read at the index
   that apply_movement_op gives for an element of the result, is that element of
   the moved source as the movement defines it (Tensors), and an element that a
   pad adds is one whose index is invalid. The source's element [j] holds [j +
   1], so that a padded [0] is told from a read.

   The composition law: the index of two movements of one kind, applied in turn,
   reads what the index of the one movement they make reads. *)

let concrete u =
  List.map
    (function Ops.Int n -> n | Sym _ -> invalid_arg "a symbolic shape")
    (Ops.shape u)

let source in_shape = Ops.param ~shape:[ Int (size in_shape) ] 0 Int32
let elements in_shape = Array.init (size in_shape) (fun j -> z (j + 1))
let view in_shape = Ops.reshape (source in_shape) (ints in_shape)
let shape_of in_shape m = concrete (Ops.mop (view in_shape) m)

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
      (fun acc n i -> Ops.O.((acc * Ops.int n) + i))
      (Ops.int 0) in_shape idxs
  in
  let read = Ops.index (source in_shape) [ offset ] in
  let names = List.map (fun r -> Option.get (Interpreter.name r)) in
  List.map
    (fun c ->
      let vars = List.combine (names (ranges out_shape)) (List.map z c) in
      Interpreter.eval ~vars ~buffers:[ (0, elements in_shape) ] read)
    (coords out_shape)

let defined in_shape m =
  let moved = Ops.mop (view in_shape) m in
  match Tensors.eval ~buffers:[ (0, elements in_shape) ] moved with
  | [ elements ] ->
      List.map
        (function `Int v when Z.equal v Z.zero -> `Invalid | v -> v)
        (Array.to_list elements)
  | _ -> fail "a movement of a value on one device is on one device"

let apply in_shape m idxs = Indexing.apply_movement_op (ints in_shape) m idxs

(* A pad's law says something only where the pad adds elements. *)
let index_law kind (in_shape, m) =
  let out_shape = shape_of in_shape m in
  let expected = defined in_shape m in
  if kind = `Pad then cover "an element is padded" (List.mem `Invalid expected);
  equal (list Dtypes.const) expected
    (reads in_shape out_shape (apply in_shape m (ranges out_shape)))

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
  | Flip f1, Flip f2 -> Flip (List.map2 ( <> ) f1 f2)
  | _ -> invalid_arg "movements of two kinds"

let composition_law (in_shape, m1, m2) =
  let mid = shape_of in_shape m1 in
  let out_shape = shape_of mid m2 in
  let rs = ranges out_shape in
  equal (list Dtypes.const)
    (reads in_shape out_shape (apply in_shape (composite m1 m2) rs))
    (reads in_shape out_shape (apply in_shape m1 (apply mid m2 rs)))

(* Generators: shapes of up to three axes of up to four elements, and a movement
   of each kind that fits a shape. *)

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
  | Flip f -> Format.fprintf ppf "Flip %a" (pp_list Format.pp_print_bool) f

let rec each = function
  | [] -> Gen.constant []
  | g :: gs -> Gen.map (fun (x, xs) -> x :: xs) (Gen.pair g (each gs))

let shape = Gen.list ~size:(Gen.int_range 1 3) (Gen.int_range 1 4)
let divisors n = List.filter (fun d -> n mod d = 0) (List.init n succ)

let rec factors k n =
  let open Gen in
  if k = 1 then constant [ n ]
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
      let+ b =
        per_axis (fun n ->
            let* start = int_range 0 (n - 1) in
            let+ length = int_range 1 (n - start) in
            (Ops.Int start, Ops.Int length))
      in
      Ops.Shrink b
  | `Permute ->
      let+ o = permutation (List.init (List.length in_shape) Fun.id) in
      Ops.Permute o
  | `Flip ->
      let+ f = per_axis (fun _ -> bool) in
      Ops.Flip f

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

let kinds =
  [
    ("shrink", `Shrink);
    ("permute", `Permute);
    ("flip", `Flip);
    ("expand", `Expand);
    ("pad", `Pad);
    ("reshape", `Reshape);
  ]

let movement_laws =
  group "apply_movement_op › laws"
    (List.concat_map
       (fun (name, kind) ->
         [
           prop
             (Printf.sprintf "a %s's index reads the element it moves" name)
             (one kind) (index_law kind);
           prop
             (Printf.sprintf "the indices of two %ss compose" name)
             (two kind) composition_law;
         ])
       kinds)

(* Recorded programs *)

let programs =
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
    "contiguous";
    "assign";
    "assign_permuted";
    "assign_double_diamond";
    "setitem";
    "custom_kernel";
    "variable_shrink";
    "variable_offset";
    "variable_data_and_shape";
    "variable_stack";
    "copy";
    "bitcast_copy";
    "shard_add";
    "shard_sum";
    "shard_sum_1";
    "shard_sum_ring";
    "shard_sum_ring_1";
    "shard_matmul";
    (* hand-built graphs *)
    "expand_by_range";
    "expand_by_range_stored";
    "mstack_of_values";
    "scalar_and_wide_uses";
  ]

let program name = Golden.sink (name ^ ".golden")

let recorded =
  group "run_rangeify › recorded graphs"
    (List.map
       (fun name ->
         Golden.graph (name ^ "_rangeified.golden") (fun () ->
             Indexing.run_rangeify (program name)))
       programs)

let chomp s =
  if String.ends_with ~suffix:"\n" s then String.sub s 0 (String.length s - 1)
  else s

(* [plain line] is [line] without its colours, and [yellow line] the line of a
   stored node with each of its axes in yellow. *)
let plain line =
  let b = Buffer.create (String.length line) in
  let rec go k =
    if k < String.length line then
      if line.[k] = '\027' then go (String.index_from line k 'm' + 1)
      else (
        Buffer.add_char b line.[k];
        go (k + 1))
  in
  go 0;
  Buffer.contents b

let yellow line =
  if not (String.starts_with ~prefix:"***" line) then line
  else
    match String.split_on_char '[' line with
    | [] -> line
    | head :: axes ->
        let axis part =
          match String.index_opt part ']' with
          | Some j ->
              Helpers.colored Yellow (String.sub part 0 j)
              ^ String.sub part j (String.length part - j)
          | None -> part
        in
        String.concat "[" (head :: List.map axis axes)

let printout name =
  Golden.text (name ^ "_debug.golden") (fun () ->
      Helpers.context
        [ Helpers.B (Helpers.no_color, true) ]
        (fun () -> ignore (Indexing.run_rangeify ~debug:true (program name)));
      chomp (output ()))

let debug =
  group "run_rangeify › debug"
    [
      printout "pad_reduce";
      printout "two_consumers";
      printout "shared_view";
      printout "softmax";
      test "in colour, the axes a node is stored over are yellow" (fun () ->
          Helpers.context
            [ Helpers.B (Helpers.no_color, false) ]
            (fun () ->
              ignore
                (Indexing.run_rangeify ~debug:true (program "two_consumers")));
          let lines = String.split_on_char '\n' (chomp (output ())) in
          equal (list text) (List.map (fun l -> yellow (plain l)) lines) lines);
      test "without debug, nothing is printed" (fun () ->
          ignore (Indexing.run_rangeify (program "softmax"));
          equal string "" (output ()));
    ]

(* What programs write

   The law that ranges keep what a program writes: where the rangeified graph is
   one kernel that reads only what it was given, the writes of its stores
   (Interpreter) are what the program's tensors write (Tensors). Storage [s]
   holds small integers, so that sums are exact in any order. *)

let devices = function Some (Ops.Multi l) -> List.length l | _ -> 1

let filled u =
  List.filter_map
    (fun n ->
      match (Ops.op n, Ops.arg n) with
      | ( (Param | Buffer | Alloc),
          Param { slot; size = Some size; dtype; device; _ } ) ->
          let element j : Dtype.value =
            let k = (((j * 7) + (slot * 3)) mod 11) - 3 in
            if Dtype.is_float dtype then `Float (float_of_int k)
            else if Dtype.equal dtype Bool then `Bool (k > 0)
            else z k
          in
          Some (slot, Array.init (size * devices device) element)
      | _ -> None)
    (Ops.toposort u)

let write = triple int int Dtypes.value

let zeros =
  List.map (fun (s, i, v) ->
      (s, i, match v with `Float f -> `Float (f +. 0.) | v -> v))

let keeps_writes name =
  test (name ^ " writes what its tensors write") (fun () ->
      let sink = program name in
      let buffers = filled sink in
      equal (list write)
        (zeros (Tensors.writes ~buffers sink))
        (zeros (Interpreter.writes ~buffers (Indexing.run_rangeify sink))))

let writes =
  group "run_rangeify › writes"
    (List.map keeps_writes
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
       ])

(* Rules

   Hand-built graphs, each the store of one value into the storage of slot [0]
   on the CPU, that state one rule of run_rangeify at a time. *)

let cpu = Ops.Single "CPU"
let p slot shape = Ops.param ~device:cpu ~shape:(ints shape) slot Float32

let stored value =
  let out =
    Ops.param ~device:cpu ~shape:[ Int (size (concrete value)) ] 0 Float32
  in
  Ops.sink
    [ Ops.after out [ Ops.store (Ops.reshape out (Ops.shape value)) value ] ]

let rangeified value = Indexing.run_rangeify (stored value)
let all op u = List.filter (fun n -> Ops.op n = op) (Ops.toposort u)
let count op u = List.length (all op u)
let axis_type = Testable.make ~pp:Ops.Axis_type.pp ~equal:Ops.Axis_type.equal

let range_of r =
  (Ops.axis_id r, Ops.axis_type r, Z.to_int (Ops.to_z (Ops.nth r 0)))

let ranges_of u = List.sort compare (List.map range_of (all Range u))
let ranges_witness = list (triple (list int) axis_type int)

(* The ranges a stage stores its source over. *)
let staged_over s = List.length (List.tl (Ops.src s))
let exp2 = Ops.exp2

let new_ranges =
  group "run_rangeify › new ranges"
    [
      test "a stored node gets a range per axis, and a unit axis none"
        (fun () ->
          equal ranges_witness
            [ ([ 0 ], Weak, 4); ([ 1 ], Weak, 3) ]
            (ranges_of (rangeified (exp2 (p 1 [ 4; 1; 3 ])))));
      test "an empty axis gets a range" (fun () ->
          equal ranges_witness
            [ ([ 0 ], Weak, 0) ]
            (ranges_of (rangeified (exp2 (p 1 [ 0 ])))));
      test "the axes a reduction reduces get reduce ranges, numbered after"
        (fun () ->
          equal ranges_witness
            [ ([ 0 ], Weak, 4); ([ 1 ], Reduce, 8) ]
            (ranges_of (rangeified (Ops.rop (p 1 [ 4; 8 ]) Add [ 1 ]))));
      test "consumers that index a node alike share its ranges" (fun () ->
          let x = exp2 (p 1 [ 4; 4 ]) in
          equal int 0 (count Stage (rangeified Ops.O.(x + x))));
      test "consumers that index a node differently store it on every axis"
        (fun () ->
          let x = exp2 (p 1 [ 4; 4 ]) in
          let stages =
            all Stage (rangeified Ops.O.(x + Ops.permute x [ 1; 0 ]))
          in
          equal (list int) [ 2 ] (List.map staged_over stages));
      test
        "consumers that index a node differently on one axis store it on every \
         axis" (fun () ->
          let x = exp2 (p 1 [ 4; 4 ]) in
          let stages = all Stage (rangeified Ops.O.(x + Ops.flip x [ 1 ])) in
          equal (list int) [ 2 ] (List.map staged_over stages));
    ]

let below_broadcasts =
  let sum = Ops.reshape (Ops.rop (p 1 [ 4; 8 ]) Add [ 1 ]) (ints [ 4; 1 ]) in
  let staged value =
    List.map (fun s -> Ops.op (Ops.nth s 0)) (all Stage (rangeified value))
  in
  let ops = list (Testable.make ~pp:Op.pp ~equal:( = )) in
  group "run_rangeify › below a broadcast"
    [
      test "an elementwise node that an operation broadcasts is not stored"
        (fun () ->
          equal ops [] (staged Ops.O.(exp2 (p 1 [ 4; 1 ]) + p 2 [ 4; 8 ])));
      test "a reduction that an operation broadcasts is stored" (fun () ->
          equal ops [ Reduce ] (staged Ops.O.(sum + p 2 [ 4; 8 ])));
      test "a reduction below a broadcast elementwise node is stored" (fun () ->
          equal ops [ Reduce ] (staged Ops.O.(exp2 sum + p 2 [ 4; 8 ])));
      test "an elementwise node that an expand broadcasts is stored" (fun () ->
          let e = Ops.expand (exp2 (p 1 [ 4 ])) (ints [ 3; 4 ]) in
          equal ops [ Exp2 ] (staged Ops.O.(e + p 2 [ 3; 4 ])));
    ]

let storage =
  let out = Ops.param ~device:cpu ~shape:[ Int 16 ] 0 Float32 in
  let dest = Ops.reshape out (ints [ 4; 4 ]) in
  let assign value = Ops.sink [ Ops.after out [ Ops.store dest value ] ] in
  (* The op of the value a store stores, and whether it is read from a stage. *)
  let stored_value u =
    List.map
      (fun st ->
        let v = Ops.nth st 1 in
        if Ops.op v = Index && Ops.op (Ops.nth v 0) = Stage then
          (true, Ops.op (Ops.nth (Ops.nth v 0) 0))
        else (false, Ops.op v))
      (all Store u)
  in
  let staged_op = list (pair bool (Testable.make ~pp:Op.pp ~equal:( = ))) in
  group "run_rangeify › stores"
    [
      test "a value stored into storage it reads is stored whole first"
        (fun () ->
          let value = Ops.O.(Ops.permute dest [ 1; 0 ] + Ops.float 1.) in
          equal staged_op
            [ (true, Add) ]
            (stored_value (Indexing.run_rangeify (assign value))));
      test "a value that does not read its destination is not" (fun () ->
          let value = Ops.O.(p 1 [ 4; 4 ] + Ops.float 1.) in
          equal staged_op
            [ (false, Add) ]
            (stored_value (Indexing.run_rangeify (assign value))));
      test "a stored store is closed by an end over its ranges" (fun () ->
          let ends = all End (rangeified (exp2 (p 1 [ 4; 3 ]))) in
          equal (list int) [ 2 ]
            (List.map (fun e -> List.length (Ops.src e) - 1) ends));
    ]

(* Stacks of constants: the selection that replaces a stack writes each of its
   sources, selects the last at a negative index, and nests its comparisons to a
   depth logarithmic in their number: a chain of at most [8], below one
   comparison per halving. *)

let constants n =
  Ops.stack
    (List.init n (fun k ->
         Ops.float ~dtype:Float32 (float_of_int ((3 * k) + 7))))

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
    (Ops.toposort u);
  Ops.Tbl.find depths u

let stacks =
  group "run_rangeify › stacks"
    (List.map
       (fun n ->
         test (Printf.sprintf "a stack of %d constants writes each" n)
           (fun () ->
             let sink = stored (constants n) in
             equal (list write)
               (zeros (Tensors.writes sink))
               (zeros (Interpreter.writes (Indexing.run_rangeify sink)))))
       [ 1; 8; 9; 17; 100 ]
    @ List.map
        (fun n ->
          test
            (Printf.sprintf
               "a stack of %d constants selects the last at a negative index, \
                in logarithmic depth"
               n) (fun () ->
              let u = Indexing.run_rangeify (stored (constants n)) in
              let value = Ops.nth (List.hd (all Store u)) 1 in
              let r0 = List.hd (all Range u) in
              equal Dtypes.const
                (`Float (float_of_int ((3 * (n - 1)) + 7)))
                (Interpreter.eval
                   ~vars:[ (Option.get (Interpreter.name r0), z (-1)) ]
                   value);
              let halvings = Float.(to_int (ceil (log2 (of_int n /. 8.)))) in
              at_most int ~than:(7 + max 0 halvings + 1) (depth u)))
        [ 2; 8; 9; 17; 1024 ])

let rewrites =
  let x = exp2 (p 1 [ 4; 4 ]) in
  let opts u =
    List.map
      (fun s ->
        match Ops.arg s with
        | Bufferize o -> o
        | _ -> fail "a stage has options")
      (all Stage u)
  in
  let opts_witness = Testable.make ~pp:Ops.pp_bufferize_opts ~equal:( = ) in
  let movements = Op.[ Reshape; Expand; Pad; Shrink; Permute; Flip ] in
  group "run_rangeify › rewrites"
    [
      test "a stage lives on its source's device and may be inlined" (fun () ->
          equal (list opts_witness)
            [ { device = Some cpu; addrspace = Global; removable = true } ]
            (opts (rangeified Ops.O.(x + Ops.permute x [ 1; 0 ]))));
      test "a stage of a value placed nowhere lives on the sink's device"
        (fun () ->
          let c =
            exp2 (Ops.expand (Ops.float ~dtype:Float32 2.) (ints [ 4; 4 ]))
          in
          equal (list opts_witness)
            [ { device = Some cpu; addrspace = Global; removable = true } ]
            (opts (rangeified Ops.O.(c + Ops.permute c [ 1; 0 ]))));
      test "a reduction of leading axes becomes a reduction over ranges"
        (fun () ->
          let reductions =
            all Reduce (rangeified (Ops.rop (p 1 [ 4; 8 ]) Add [ 1 ]))
          in
          equal
            (list (list (Testable.make ~pp:Op.pp ~equal:( = ))))
            [ [ Range ] ]
            (List.map
               (fun r -> List.map Ops.op (List.tl (Ops.src r)))
               reductions);
          equal (list int) [ 0 ]
            (List.map
               (fun r ->
                 match Ops.arg r with
                 | Reduce { num_axes; _ } -> num_axes
                 | _ -> -1)
               reductions));
      test "a pad becomes a selection of its source and of zero" (fun () ->
          let u = rangeified (Ops.pad (p 1 [ 4 ]) [ Some (Int 1, Int 2) ]) in
          equal int 0 (count Pad u);
          equal (list Dtypes.const)
            [ `Float 0. ]
            (List.filter_map
               (fun w ->
                 if Ops.dtype w = Float32 then Some (Ops.value (Ops.nth w 2))
                 else None)
               (all Where u)));
      test "a stack becomes a selection of its sources" (fun () ->
          let u = rangeified (Ops.stack [ p 1 [ 4 ]; p 2 [ 4 ]; p 3 [ 4 ] ]) in
          equal int 0
            (List.length
               (List.filter (fun s -> Ops.dtype s = Float32) (all Stack u))));
      test "movements are removed" (fun () ->
          let u =
            rangeified
              (Ops.flip
                 (Ops.permute
                    (Ops.shrink (p 1 [ 6; 4 ]) [ Some (Int 1, Int 5); None ])
                    [ 1; 0 ])
                 [ 0 ])
          in
          equal int 0
            (List.length
               (List.filter
                  (fun n -> List.mem (Ops.op n) movements)
                  (Ops.toposort u))));
      test "calls, afters and shards get no ranges" (fun () ->
          ignore (Indexing.run_rangeify ~debug:true (program "shard_sum"));
          let printed = output () in
          List.iter
            (fun op -> not_contains ~sub:op printed)
            [ "Ops.CALL"; "Ops.AFTER"; "Ops.MSELECT"; "Ops.MSTACK" ]);
      test "a pad that gets no ranges stays a pad" (fun () ->
          let u = Ops.sink [ Ops.pad (p 1 [ 4 ]) [ Some (Int 1, Int 2) ] ] in
          equal int 1 (count Pad (Indexing.run_rangeify u)));
      test "a source of a gather across devices is stored on its device"
        (fun () ->
          equal (list opts_witness)
            [
              {
                device = Some (Single "CPU:0");
                addrspace = Global;
                removable = true;
              };
              {
                device = Some (Single "CPU:1");
                addrspace = Global;
                removable = true;
              };
            ]
            (opts (Indexing.run_rangeify (program "mstack_of_values"))));
      test "an expand by a range ends none of its source's ranges" (fun () ->
          equal int 0
            (count Stage (Indexing.run_rangeify (program "expand_by_range"))));
      test "a stored axis whose size is a range is indexed by that range"
        (fun () ->
          let u = Indexing.run_rangeify (program "expand_by_range_stored") in
          equal
            (list (list (list int)))
            [ [ [ 7 ]; [ 2 ] ] ]
            (List.map
               (fun s -> List.map Ops.axis_id (List.tl (Ops.src s)))
               (all Stage u)));
      test "a reduction of leading axes without ranges is refused" (fun () ->
          raises_match Exn.invalid_arg (fun () ->
              Indexing.run_rangeify
                (Ops.sink [ Ops.rop (p 1 [ 4; 8 ]) Add [ 1 ] ])));
    ]

let () =
  exit
    (run "Indexing"
       [
         movements;
         movement_laws;
         recorded;
         debug;
         writes;
         new_ranges;
         below_broadcasts;
         storage;
         stacks;
         rewrites;
       ])
