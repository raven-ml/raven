(* Tests of Tolk_next.Prepare: what prepare_rangeify makes of the graphs
   tinygrad schedules, the law that it keeps what a program writes, the rules of
   pm_mops and their law, and contiguous_view. *)

open Windtrap
open Tolk_next

let uop = Uops.uop
let program name = Golden.sink (name ^ ".golden")
let prepare = Prepare.prepare_rangeify
let size shape = List.fold_left ( * ) 1 shape
let ints = List.map (fun n -> Ops.Int n)

(* Recorded graphs *)

let programs =
  [
    "add";
    "add_nothing";
    "alias_of_placed";
    "arange";
    "assign";
    "assign_bitcast";
    "assign_bitcast_wider";
    "assign_broadcast";
    "assign_cross_device";
    "assign_deviceless_const";
    "assign_disjoint_self";
    "assign_double_bitcast";
    "assign_flipped_self";
    "assign_own_contents";
    "assign_permuted_self";
    "assign_reshaped";
    "assign_reshaped_self";
    "assign_shifted_self";
    "assign_shrink_then_bitcast";
    "assign_shrunk_in_place";
    "assign_shrunk_self";
    "assign_to_disk";
    "assign_to_disk_1";
    "assign_to_function_output";
    "assign_twice";
    "attention";
    "below_threshold";
    "bitcast_half";
    "bitcast_long";
    "bitcast_narrow";
    "bitcast_on_disk";
    "bitcast_wide";
    "cat";
    "clone";
    "contiguous";
    "contiguous_backward";
    "contiguous_of_storage";
    "contiguous_permuted";
    "conv";
    "copy";
    "copy_computed";
    "copy_from_disk";
    "copy_into_storage";
    "copy_permuted_from_disk";
    "copy_same_device";
    "copy_staged_view_from_disk";
    "copy_view";
    "cumsum";
    "custom_kernel";
    "detach";
    "disk_staged_view";
    "embedding";
    "flip_of_other";
    "hazard_behind_other_after";
    "inline_function";
    "inline_sharded";
    "inline_symbolic";
    "matmul";
    "max_of_nothing";
    "no_split";
    "pad";
    "placed_after_read";
    "placed_through_view";
    "precompiled_function";
    "precompiled_function_1";
    "setitem";
    "setitem_tensor";
    "shard_add";
    "shard_gather";
    "shard_sum";
    "shard_sum_1";
    "sharded_output_of_whole_storage";
    "softmax";
    "sort";
    "split_at_threshold";
    "split_expanded";
    "split_max";
    "split_prime";
    "split_rows";
    "split_sum";
    "split_unit_axis";
    "stack";
    "store_ordered_before";
    "sum";
    "sum_of_nothing";
    "variable_shrink";
    "where";
  ]

(* The settings a program was recorded under. *)
let settings = function
  | "no_split" -> [ Helpers.B (Helpers.split_reduceop, false) ]
  | _ -> []

let prepared name =
  Helpers.context (settings name) (fun () -> prepare (program name))

let recorded =
  group "prepare_rangeify › recorded"
    (List.map
       (fun name ->
         let file = name ^ "_prepared.golden" in
         Golden.graph file (fun () ->
             Uops.numbered_like (Golden.sink file) (prepared name)))
       programs)

(* Values

   The law that preparing keeps what a program writes into the storage it is
   given, its parameters and buffers (Tensors); call-local storage ({!Op.Alloc})
   is scratch, which preparing may remove or add. Memory holds small integers,
   the same on each device of a replicated buffer and different on each device
   of a sharded one. *)

let devices = function Some (Ops.Multi l) -> List.length l | _ -> 1

(* [element dtype k] is [k], from [-3] to [7], as a [dtype] value; shifted to be
   non-negative in an unsigned type. *)
let element dtype k : Dtype.value =
  if Dtype.is_float dtype then `Float (float_of_int k)
  else if Dtype.equal dtype Bool then `Bool (k > 0)
  else if Dtype.is_unsigned dtype then `Int (Z.of_int (k + 3))
  else `Int (Z.of_int k)

let storage u =
  List.filter_map
    (fun n ->
      match (Ops.op n, Ops.arg n) with
      | (Param | Buffer | Alloc), Param p when p.addrspace <> Some Alu ->
          Some (n, p)
      | _ -> None)
    (Ops.toposort u)

let filled u =
  let sharded =
    List.filter_map
      (fun n ->
        if Ops.op n = Unshard then Some (Ops.storage_base (Ops.nth n 0))
        else None)
      (Ops.toposort u)
  in
  List.map
    (fun (n, (p : Ops.param_arg)) ->
      let size = Option.value p.size ~default:1 in
      let at j =
        let j = if List.memq n sharded then j else j mod size in
        element p.dtype ((((j * 7) + (p.slot * 3)) mod 11) - 3)
      in
      (p.slot, Array.init (size * devices p.device) at))
    (storage u)

let close = Testable.float_rel ~rel:1e-6 ~abs:0.

let value =
  Testable.make ~pp:(Testable.pp Dtypes.value) ~equal:(fun v0 v1 ->
      match (v0, v1) with
      | `Float x, `Float y -> Testable.equal close x y
      | v0, v1 -> Testable.equal Dtypes.value v0 v1)

let write = triple int int value

let keeps_writes name =
  test (name ^ " writes what it wrote before") (fun () ->
      let sink = program name in
      let buffers = filled sink in
      let given =
        List.filter_map
          (fun (n, (p : Ops.param_arg)) ->
            if Ops.op n = Alloc then None else Some p.slot)
          (storage sink)
      in
      let into_given = List.filter (fun (s, _, _) -> List.mem s given) in
      equal (list write)
        (into_given (Tensors.writes ~buffers sink))
        (into_given (Tensors.writes ~buffers (prepared name))))

(* Programs the law does not apply to: a call of a kernel or of a compiled
   schedule, which Tensors does not run, and a symbolic shape. *)
let unevaluated =
  [
    "custom_kernel";
    "precompiled_function";
    "precompiled_function_1";
    "variable_shrink";
    "inline_symbolic";
  ]

let values =
  group "prepare_rangeify › values"
    (List.filter_map
       (fun name ->
         if List.mem name unevaluated then None else Some (keeps_writes name))
       programs)

(* pm_mops

   Movements move towards the storage they view. The law: an index of a movement
   chain, rewritten, reads through the storage the element that the movements
   place there (Tensors). Pads are left out of the law: the validity a pad gives
   an index lasts only while the index carries it (Indexing.apply_movement_op),
   which [pm_mops › pads] states. Storage element [j] holds [j + 1], so that a
   padded [0] is told from a read. *)

let mops u = Ops.graph_rewrite ~ctx:() u Prepare.pm_mops

let range ?(axis_type = Ops.Axis_type.Loop) n axis =
  Ops.range ~axis_type (Int n) [ axis ]

let ranges shape = List.mapi (fun axis n -> range n axis) shape
let cpu = Ops.Single "CPU"
let flat n = Ops.param ~device:cpu ~shape:[ Int n ] 1 Float32
let stored shape = Ops.reshape (flat (size shape)) (ints shape)

let copy_in =
  Ops.store (flat 4) (Ops.param ~device:cpu ~shape:[ Int 4 ] 2 Float32)

let pm_mops_rules =
  let x = stored [ 4; 8 ] in
  let r0 = range 8 0 and r1 = range 4 1 in
  group "pm_mops › rules"
    [
      test "an index of a movement indexes its source at the mapped indices"
        (fun () ->
          let mapped =
            Indexing.apply_movement_op
              (ints [ 4; 8 ])
              (Permute [ 1; 0 ])
              [ r0; r1 ]
          in
          let flat_index =
            Indexing.apply_movement_op [ Int 32 ]
              (Reshape (ints [ 4; 8 ]))
              mapped
          in
          equal uop
            (Ops.index (flat 32) flat_index)
            (mops (Ops.index (Ops.permute x [ 1; 0 ]) [ r0; r1 ])));
      test
        "an index of a reshape's leading axes indexes its source when the \
         trailing axes are kept" (fun () ->
          let a = range 2 0 and b = range 2 1 in
          let lead =
            Indexing.apply_movement_op [ Int 4 ]
              (Reshape (ints [ 2; 2 ]))
              [ a; b ]
          in
          equal uop (Ops.index x lead)
            (mops (Ops.index (Ops.reshape x (ints [ 2; 2; 8 ])) [ a; b ])));
      test "an index of a reshape whose trailing axes change is left as it is"
        (fun () ->
          let u = Ops.index (Ops.reshape x (ints [ 2; 16 ])) [ range 2 0 ] in
          equal uop u (mops u));
      test "an index of a reshape's added leading axis is its source" (fun () ->
          let row = stored [ 8 ] in
          equal uop row
            (mops (Ops.index (Ops.reshape row (ints [ 1; 8 ])) [ range 1 0 ])));
      test "a movement after effects is the movement of its source after them"
        (fun () ->
          let u = flat 32 in
          equal uop
            (Ops.reshape (Ops.after u [ copy_in ]) (ints [ 4; 8 ]))
            (mops (Ops.after (Ops.reshape u (ints [ 4; 8 ])) [ copy_in ])));
      test "an index after effects is the index of its source after them"
        (fun () ->
          let u = flat 32 in
          equal uop
            (Ops.index (Ops.after u [ copy_in ]) [ r0 ])
            (mops (Ops.after (Ops.index u [ r0 ]) [ copy_in ])));
      test "an end of a movement is the end of its source" (fun () ->
          let u = flat 32 in
          equal uop (Ops.end_ u [ r0 ])
            (mops (Ops.end_ (Ops.reshape u (ints [ 4; 8 ])) [ r0 ])));
    ]

let concrete u =
  List.map
    (function Ops.Int n -> n | Sym _ -> fail "a generated shape is concrete")
    (Ops.shape u)

let rec coords = function
  | [] -> [ [] ]
  | n :: rest ->
      List.concat_map
        (fun i -> List.map (fun c -> i :: c) (coords rest))
        (List.init n Fun.id)

let pp_list pp ppf l =
  Format.fprintf ppf "[%a]"
    (Format.pp_print_list ~pp_sep:(fun ppf () -> Format.fprintf ppf "; ") pp)
    l

let pp_sint ppf = function
  | Ops.Int n -> Format.pp_print_int ppf n
  | Sym u -> Format.pp_print_string ppf (Render.render u)

let pp_movement ppf (m : Ops.movement) =
  let pair ppf (a, b) = Format.fprintf ppf "(%a, %a)" pp_sint a pp_sint b in
  match m with
  | Reshape s -> Format.fprintf ppf "reshape %a" (pp_list pp_sint) s
  | Expand s -> Format.fprintf ppf "expand %a" (pp_list pp_sint) s
  | Pad p -> Format.fprintf ppf "pad %a" (pp_list pair) p
  | Shrink b -> Format.fprintf ppf "shrink %a" (pp_list pair) b
  | Permute o -> Format.fprintf ppf "permute %a" (pp_list Format.pp_print_int) o
  | Flip f -> Format.fprintf ppf "flip %a" (pp_list Format.pp_print_bool) f

let rec each = function
  | [] -> Gen.constant []
  | g :: gs -> Gen.map (fun (x, xs) -> x :: xs) (Gen.pair g (each gs))

let divisors n = List.filter (fun d -> n mod d = 0) (List.init n succ)

let rec factors k n =
  let open Gen in
  if k = 1 then constant [ n ]
  else
    let* d = of_list (divisors n) in
    let+ rest = factors (k - 1) (n / d) in
    d :: rest

(* A movement of one of [kinds] that fits [shape]. *)
let movement kinds shape : Ops.movement Gen.t =
  let open Gen in
  let per_axis f = each (List.map f shape) in
  let* kind = of_list kinds in
  match kind with
  | `Reshape ->
      let* k = int_range 1 3 in
      let+ s = factors k (size shape) in
      Ops.Reshape (ints s)
  | `Expand ->
      let+ added = list ~size:(int_range 1 2) (int_range 1 3) in
      Ops.Expand (ints added)
  | `Shrink ->
      let+ b =
        per_axis (fun n ->
            let* start = int_range 0 (n - 1) in
            let+ length = int_range 1 (n - start) in
            (Ops.Int start, Ops.Int length))
      in
      Ops.Shrink b
  | `Permute ->
      let+ o = permutation (List.init (List.length shape) Fun.id) in
      Ops.Permute o
  | `Flip ->
      let+ f = per_axis (fun _ -> bool) in
      Ops.Flip f

(* A shape of up to three axes of up to four elements, and up to three movements
   of one of [kinds], in turn. *)
let chains kinds =
  let open Gen in
  let rec moves shape k =
    if k = 0 then constant []
    else
      let* m = movement kinds shape in
      let next = concrete (Ops.mop (stored shape) m) in
      let+ rest = moves next (k - 1) in
      m :: rest
  in
  (let* shape = list ~size:(int_range 1 3) (int_range 1 4) in
   let* k = int_range 1 3 in
   let+ ms = moves shape k in
   (shape, ms))
  |> with_pp (fun ppf (shape, ms) ->
      Format.fprintf ppf "%a of %a" (pp_list pp_movement) ms
        (pp_list Format.pp_print_int)
        shape)

let chain = chains [ `Reshape; `Expand; `Shrink; `Permute; `Flip ]
let moved (shape, ms) = List.fold_left Ops.mop (stored shape) ms
let iota n = Array.init n (fun j -> `Float (float_of_int (j + 1)))

let tensor_values memory u =
  match Tensors.eval ~buffers:memory u with
  | [ elements ] -> Array.to_list elements
  | _ -> fail "a movement of a value on one device is on one device"

(* [reads memory rs u cs] is [u] at each binding [cs] of the ranges [rs]. *)
let reads memory rs u cs =
  let names = List.map (fun r -> Option.get (Interpreter.name r)) rs in
  List.map
    (fun c ->
      let vars = List.combine names (List.map (fun i -> `Int (Z.of_int i)) c) in
      Interpreter.eval ~vars ~buffers:memory u)
    cs

let reads_what_it_moves (shape, ms) =
  let u = moved (shape, ms) in
  let out_shape = concrete u in
  let rs = ranges out_shape in
  let read = mops (Ops.index u rs) in
  let memory = [ (1, iota (size shape)) ] in
  equal (list Dtypes.const) (tensor_values memory u)
    (reads memory rs read (coords out_shape))

let pm_mops_laws =
  group "pm_mops › laws"
    [
      prop "an index of movements reads the element they move" chain
        reads_what_it_moves;
    ]

(* The validity a pad gives an index lasts only while the index expression
   carries it: a later movement whose simplification makes the index constant on
   an axis drops it, and a caller masks the padded values itself. Each case is
   one tinygrad HEAD gives the same index for: the read drops the gate, and
   masked by the pad's condition it reads what the movements place. *)

let pad_validity name (shape, ms) mask =
  test ("a pad's validity lasts only while the index carries it: " ^ name)
    (fun () ->
      let u = moved (shape, ms) in
      let rs = ranges (concrete u) in
      let read = mops (Ops.index u rs) in
      let masked = Ops.where (mask rs) read (Ops.float ~dtype:Float32 0.) in
      let memory = [ (1, iota (size shape)) ] in
      let cs = coords (concrete u) in
      let expected = tensor_values memory u in
      is_true ~msg:"an element is padded" (List.mem (`Float 0.) expected);
      is_false ~msg:"the gate is dropped"
        (List.mem `Invalid (reads memory rs read cs));
      equal (list Dtypes.const) expected (reads memory rs masked cs))

let pads =
  let nth k rs = List.nth rs k in
  group "pm_mops › pads"
    [
      pad_validity "a reshape to an axis of one element"
        ([ 1 ], [ Reshape (ints [ 1 ]); Pad [ (Int 0, Int 2) ] ])
        (fun rs -> Ops.O.(nth 0 rs < int 1));
      pad_validity "an expand that drops the leading index"
        ([ 2 ], [ Expand (ints [ 2 ]); Pad [ (Int 1, Int 3); (Int 0, Int 2) ] ])
        (fun rs -> Ops.O.(nth 0 rs >= int 1));
      pad_validity "a shrink onto an axis of one element"
        ( [ 3; 2 ],
          [
            Pad [ (Int 0, Int 3); (Int 2, Int 6) ];
            Shrink [ (Int 0, Int 1); (Int 0, Int 3) ];
          ] )
        (fun rs -> Ops.O.(nth 1 rs >= int 2));
    ]

(* contiguous_view

   The laws: a view found, [Some (b, offset)], has its elements, in row-major
   order, [b]'s from [offset] on (Tensors), [b] holding distinct elements; and
   reshapes, shrinks and permutes are found exactly when their elements are such
   a run. Two runs are not found, as in tinygrad, whose rewrite of the index
   does not reach them: flips that cancel across a reshape, and a read within
   one copy of an expanded value. *)

let view_witness = option (pair uop int)

let contiguous_views =
  let b = flat 32 in
  let x = Ops.reshape b (ints [ 4; 8 ]) in
  let bytes = Ops.bitcast b Uint8 in
  let case name expected u =
    test name (fun () ->
        equal view_witness expected (Prepare.contiguous_view u))
  in
  group "contiguous_view"
    [
      case "storage is its own view from 0" (Some (b, 0)) b;
      case "a reshape keeps the order" (Some (b, 0)) x;
      case "a shrink of leading rows starts at their first element"
        (Some (b, 8))
        (Ops.shrink x [ Some (Int 1, Int 3); None ]);
      case "a shrink of inner columns skips elements" None
        (Ops.shrink x [ None; Some (Int 2, Int 4) ]);
      case "a permute reorders" None (Ops.permute x [ 1; 0 ]);
      case "a flip reorders" None (Ops.flip x [ 1 ]);
      (let x = Ops.reshape b (ints [ 2; 1; 16 ]) in
       case "flips that cancel across a reshape are not found" None
         (Ops.flip (Ops.reshape (Ops.flip x [ 0 ]) (ints [ 2; 16 ])) [ 0 ]));
      (let x = stored [ 1; 2; 4; 2 ] in
       case "a read within one copy of an expanded value is not found" None
         (Ops.shrink
            (Ops.reshape (Ops.expand x (ints [ 2; 2; 4; 2 ])) (ints [ 32 ]))
            [ Some (Int 6, Int 9) ]));
      case "an expand repeats" None
        (Ops.expand (Ops.reshape b (ints [ 1; 32 ])) (ints [ 2; 32 ]));
      case "a pad adds elements" None (Ops.pad b [ Some (Int 1, Int 0) ]);
      case "a bitcast to bytes is a view of the storage" (Some (b, 0)) bytes;
      case "an element is a view at its offset"
        (Some (b, 11))
        (Ops.shrink x [ Some (Int 1, Int 2); Some (Int 3, Int 4) ]);
      case "an empty view is no view" None
        (Ops.shrink x [ Some (Int 2, Int 2); None ]);
      (let ints = Ops.param ~device:cpu ~shape:[ Int 8 ] 1 Int32 in
       let ordered =
         Ops.after ints
           [ Ops.store ints (Ops.param ~device:cpu ~shape:[ Int 8 ] 2 Int32) ]
       in
       case
         "a view of storage after its stores is a view of the ordered storage"
         (Some (ordered, 2))
         (Ops.shrink ordered [ Some (Int 2, Int 5) ]));
      (let rows = Ops.variable "rows" (`Int (Z.of_int 1)) (`Int (Z.of_int 2)) in
       let m = Ops.reshape (flat 6) (ints [ 3; 2 ]) in
       case "a view of a symbolic size is no view" None
         (Ops.shrink m
            [ Some (Int 1, Sym Ops.O.(int 1 + rows)); Some (Int 0, Int 2) ]));
      case "two permutes that cancel are a view"
        (Some (b, 0))
        (Ops.permute (Ops.permute x [ 1; 0 ]) [ 1; 0 ]);
      case "an expand of an axis of one element is a view"
        (Some (b, 0))
        (Ops.expand (Ops.reshape x (ints [ 1; 4; 8 ])) (ints [ 1; 4; 8 ]));
      case "bytes on whole elements start at their element"
        (Some (b, 1))
        (Ops.shrink bytes [ Some (Int 4, Int 12) ]);
      case "a bitcast of a view with an axis of one element is a view"
        (Some (b, 0))
        (Ops.bitcast (Ops.reshape b (ints [ 1; 32 ])) Uint8);
      case "bytes within an element are a view of the bytes"
        (Some (bytes, 2))
        (Ops.shrink bytes [ Some (Int 2, Int 6) ]);
    ]

let is_run memory elements =
  let n = Array.length elements and m = Array.length memory in
  let at off = Array.for_all2 ( = ) elements (Array.sub memory off n) in
  List.find_opt at (List.init (max 0 (m - n + 1)) Fun.id)

(* [viewed chain] is the view found for the movement of [chain], and the run of
   its storage that its elements are, if they are one. *)
let viewed (shape, ms) =
  let u = moved (shape, ms) in
  let memory = iota (size shape) in
  let elements =
    match Tensors.eval ~buffers:[ (1, memory) ] u with
    | [ e ] ->
        Array.map (function #Dtype.value as v -> v | `Invalid -> `Float nan) e
    | _ -> fail "a movement of a value on one device is on one device"
  in
  let run =
    Option.map (fun off -> (flat (size shape), off)) (is_run memory elements)
  in
  (Prepare.contiguous_view u, run)

let found_is_a_run chain =
  match viewed chain with
  | None, _ -> ()
  | (Some _ as found), run ->
      cover "a view" true;
      equal view_witness run found

let found_exactly_when_a_run chain =
  let found, run = viewed chain in
  cover "a view" (Option.is_some run);
  equal view_witness run found

let contiguous_view_laws =
  group "contiguous_view › laws"
    [
      prop "a view found is exactly a run of its storage" chain found_is_a_run;
      prop
        "reshapes, shrinks and permutes are found exactly when they are a run"
        (chains [ `Reshape; `Shrink; `Permute ])
        found_exactly_when_a_run;
    ]

(* Rules

   Hand-built functions on the CPU, each stating one rule of prepare_rangeify:
   an output of sixteen elements in slot [0], inputs in the next slots. *)

let out = Ops.param ~device:cpu ~shape:[ Int 16 ] 0 Float32
let input slot = Ops.param ~device:cpu ~shape:[ Int 16 ] slot Float32
let stores value = Ops.sink [ Ops.after out [ Ops.store out value ] ]
let has op u = List.exists (fun n -> Ops.op n = op) (Ops.toposort u)
let rejects f = raises_match (Exn.invalid_arg ?substring:None) f

let calls u =
  List.filter_map
    (fun n ->
      match Ops.arg n with
      | Call c when Ops.op n = Call -> Some c.name
      | _ -> None)
    (Ops.toposort u)

let copy_body =
  Ops.sink
    [ Ops.store (input 0) (Ops.param ~device:cpu ~shape:[ Int 16 ] 1 Float32) ]

let outputs =
  let value = Ops.exp2 (input 1) in
  group "prepare_rangeify › outputs"
    [
      test
        "a value computed into new storage of the output's size is computed \
         into the output" (fun () ->
          let scratch = Ops.alloc ~device:cpu [ Int 16 ] Float32 in
          let computed = Ops.after scratch [ Ops.store scratch value ] in
          let u = prepare (stores computed) in
          is_false (has Alloc u);
          equal (list (option string)) [] (calls u));
      test "a materialised value is computed into the output" (fun () ->
          let u =
            prepare (Ops.sink [ Ops.store out (Ops.v Stage ~src:[ value ]) ])
          in
          is_false (has Stage u);
          is_false (has Alloc u));
      test "a value that is neither is stored as it is" (fun () ->
          equal uop (stores value) (prepare (stores value)));
      test "a materialised value that reads the output is not placed there"
        (fun () ->
          let reads_out = Ops.v Stage ~src:[ Ops.O.(out + float 1.) ] in
          is_true (has Alloc (prepare (Ops.sink [ Ops.store out reads_out ]))));
      test "a value materialised into two outputs is placed in the first"
        (fun () ->
          let other = Ops.param ~device:cpu ~shape:[ Int 16 ] 2 Float32 in
          let staged = Ops.v Stage ~src:[ value ] in
          let u =
            prepare (Ops.sink [ Ops.store out staged; Ops.store other staged ])
          in
          equal
            (list (triple int int Dtypes.value))
            (List.concat_map
               (fun slot ->
                 List.init 16 (fun j ->
                     (slot, j, `Float (Float.pow 2. (float_of_int j)))))
               [ 0; 2 ])
            (Tensors.writes
               ~buffers:
                 [ (1, Array.init 16 (fun j -> `Float (float_of_int j))) ]
               u));
      test "every read of storage placed in an output reads the output"
        (fun () ->
          let scratch = Ops.alloc ~device:cpu [ Int 16 ] Float32 in
          let other = Ops.param ~device:cpu ~shape:[ Int 16 ] 2 Float32 in
          let computed = Ops.after scratch [ Ops.store scratch value ] in
          let u =
            prepare
              (Ops.sink
                 [
                   Ops.store out computed;
                   Ops.after other
                     [ Ops.store other Ops.O.(scratch + float 1.) ];
                 ])
          in
          is_false (has Alloc u));
      test "a value materialised into a view of an output is computed into it"
        (fun () ->
          let half = Ops.shrink out [ Some (Int 0, Int 8) ] in
          let staged = Ops.v Stage ~src:[ Ops.exp2 (flat 8) ] in
          let u = prepare (Ops.sink [ Ops.store half staged ]) in
          is_false (has Stage u);
          is_false (has Alloc u));
      test "an output that stores new sharded storage takes its place"
        (fun () ->
          let two = Ops.Multi [ "CPU:0"; "CPU:1" ] in
          let target =
            Ops.unshard (Ops.param ~device:two ~shape:[ Int 8 ] 0 Float32) [ 0 ]
          in
          let fresh =
            Ops.unshard (Ops.new_buffer ~slot:5 two 8 Float32) [ 0 ]
          in
          is_false (has Buffer (prepare (Ops.sink [ Ops.store target fresh ]))));
      test "an output that stores new storage takes the storage's place"
        (fun () ->
          let fresh = Ops.new_buffer ~slot:5 cpu 16 Float32 in
          is_false (has Buffer (prepare (Ops.sink [ Ops.store out fresh ]))));
    ]

let calls_inline =
  group "prepare_rangeify › inline calls"
    [
      test "an inline call is replaced by its body on its arguments" (fun () ->
          let u = prepare (Ops.sink [ Ops.call copy_body [ out; input 1 ] ]) in
          is_false (has Call u);
          equal
            (list (triple int int Dtypes.value))
            (List.init 16 (fun j -> (0, j, `Float (float_of_int j))))
            (Tensors.writes
               ~buffers:
                 [ (1, Array.init 16 (fun j -> `Float (float_of_int j))) ]
               u));
      test "an argument of another size than its parameter is refused"
        (fun () ->
          let short = Ops.param ~device:cpu ~shape:[ Int 8 ] 1 Float32 in
          let call = Ops.sink [ Ops.call copy_body [ out; short ] ] in
          rejects (fun () -> prepare call));
      test "an argument of another type than its parameter is refused"
        (fun () ->
          let ints = Ops.param ~device:cpu ~shape:[ Int 16 ] 1 Int32 in
          let call = Ops.sink [ Ops.call copy_body [ out; ints ] ] in
          rejects (fun () -> prepare call));
      test "an inline body's own storage is renamed" (fun () ->
          let scratch = Ops.alloc ~slot:7 ~device:cpu [ Int 16 ] Float32 in
          let body =
            Ops.sink
              [
                Ops.store (input 0)
                  (Ops.after scratch [ Ops.store scratch (input 1) ]);
              ]
          in
          let slots u =
            List.filter_map
              (fun n ->
                match Ops.arg n with
                | Param p when Ops.op n = Alloc -> Some p.slot
                | _ -> None)
              (Ops.toposort u)
          in
          let renamed =
            slots (prepare (Ops.sink [ Ops.call body [ out; input 1 ] ]))
          in
          equal int 1 (List.length renamed);
          not_equal (list int) [ 7 ] renamed);
      test
        "an argument that is the first part of larger storage is passed as it \
         is" (fun () ->
          let wide = Ops.param ~device:cpu ~shape:[ Int 32 ] 1 Float32 in
          let first = Ops.shrink wide [ Some (Int 0, Int 16) ] in
          let u = prepare (Ops.sink [ Ops.call copy_body [ out; first ] ]) in
          equal
            (list (triple int int Dtypes.value))
            (List.init 16 (fun j -> (0, j, `Float (float_of_int j))))
            (Tensors.writes
               ~buffers:
                 [ (1, Array.init 32 (fun j -> `Float (float_of_int j))) ]
               u));
      test "an argument of another shape than its parameter is passed flat"
        (fun () ->
          let square = Ops.reshape (input 1) (ints [ 4; 4 ]) in
          let u = prepare (Ops.sink [ Ops.call copy_body [ out; square ] ]) in
          equal
            (list (triple int int Dtypes.value))
            (List.init 16 (fun j -> (0, j, `Float (float_of_int j))))
            (Tensors.writes
               ~buffers:
                 [ (1, Array.init 16 (fun j -> `Float (float_of_int j))) ]
               u));
      test "a shaped argument of a scalar parameter is refused" (fun () ->
          let scalar = Ops.param 1 Float32 in
          let body =
            Ops.sink [ Ops.store (input 0) Ops.O.(input 0 + scalar) ]
          in
          let call = Ops.sink [ Ops.call body [ out; input 1 ] ] in
          rejects (fun () -> prepare call));
    ]

let earliest =
  group "prepare_rangeify › earliest rewrites"
    [
      test "an allreduce is a call of its own, named allreduce" (fun () ->
          let two = Ops.Multi [ "CPU:0"; "CPU:1" ] in
          let shared = Ops.param ~device:two ~shape:[ Int 16 ] 0 Float32 in
          let whole = Ops.param ~device:two ~shape:[ Int 16 ] 1 Float32 in
          let u =
            prepare
              (Ops.sink
                 [
                   Ops.after shared
                     [ Ops.store shared (Ops.allreduce whole Add two) ];
                 ])
          in
          is_false (has Allreduce u);
          equal (list (option string)) [ Some "allreduce" ] (calls u));
      test
        "a split reduction keeps its first reduction's output within 2^22 \
         elements" (fun () ->
          let wide =
            Ops.param ~device:cpu ~shape:(ints [ 65536; 32768 ]) 1 Float32
          in
          let total = Ops.param ~device:cpu ~shape:[ Int 65536 ] 0 Float32 in
          let u =
            prepare
              (Ops.sink
                 [
                   Ops.after total [ Ops.store total (Ops.rop wide Add [ 1 ]) ];
                 ])
          in
          equal (list int)
            [ 1 lsl 22 ]
            (List.filter_map
               (fun n ->
                 match Ops.arg n with
                 | Param { size = Some s; _ } when Ops.op n = Alloc -> Some s
                 | _ -> None)
               (Ops.toposort u)));
      test "a detach and a gradient marker are their source" (fun () ->
          let value = Ops.exp2 (input 1) in
          let marked =
            Ops.v Contiguous_backward ~src:[ Ops.v Detach ~src:[ value ] ]
          in
          equal uop (stores value) (prepare (stores marked)));
      test "a copy to the device its value is on is the value" (fun () ->
          equal uop
            (stores (input 1))
            (prepare (stores (Ops.copy_to_device (input 1) cpu))));
      test "a sink's sources lose their movements and bitcasts" (fun () ->
          let stored = Ops.after out [ Ops.store out (input 1) ] in
          equal uop (Ops.sink [ stored ])
            (prepare
               (Ops.sink
                  [ Ops.bitcast (Ops.reshape stored (ints [ 4; 4 ])) Int32 ])));
      test "a split takes the largest divisor from 256 down" (fun () ->
          let total = Ops.param ~device:cpu ~shape:[] 0 Float32 in
          let n = 128 * 257 in
          let u =
            prepare
              (Ops.sink
                 [
                   Ops.after total
                     [ Ops.store total (Ops.rop (flat n) Add [ 0 ]) ];
                 ])
          in
          equal (list int) [ 128 ]
            (List.filter_map
               (fun n ->
                 match Ops.arg n with
                 | Param { size = Some s; _ } when Ops.op n = Alloc -> Some s
                 | _ -> None)
               (Ops.toposort u)));
      test
        "a value that permutes other storage and reads its destination is \
         stored directly" (fun () ->
          let square u = Ops.reshape u (ints [ 4; 4 ]) in
          let dest = square out in
          let value = Ops.O.(Ops.permute (square (input 1)) [ 1; 0 ] + dest) in
          let u =
            prepare (Ops.sink [ Ops.after out [ Ops.store dest value ] ])
          in
          is_false (has Stage u);
          is_false (has Alloc u));
      test "a split is announced at debug level 3" (fun () ->
          let total = Ops.param ~device:cpu ~shape:[] 0 Float32 in
          let sum = Ops.rop (flat 65536) Add [ 0 ] in
          let u = Ops.sink [ Ops.after total [ Ops.store total sum ] ] in
          Helpers.context
            [ Helpers.B (Helpers.debug, 3) ]
            (fun () -> ignore (prepare u));
          contains ~sub:"split 256: (65536,) -> (256, 256) -> ()" (output ()));
      test "a reduction of a symbolic shape is not split" (fun () ->
          let v =
            Ops.variable "v" (`Int (Z.of_int 1)) (`Int (Z.of_int 65536))
          in
          let part = Ops.shrink (flat 65536) [ Some (Int 0, Sym v) ] in
          let total = Ops.param ~device:cpu ~shape:[] 0 Float32 in
          let u =
            Ops.sink
              [ Ops.after total [ Ops.store total (Ops.rop part Add [ 0 ]) ] ]
          in
          is_false (has Alloc (prepare u)));
      test "a materialisation of storage is the storage" (fun () ->
          equal uop
            (stores (input 1))
            (prepare (stores (Ops.v Stage ~src:[ input 1 ]))));
      test "a copy to another device than its destination's is stored first"
        (fun () ->
          let elsewhere =
            Ops.param ~device:(Single "CPU:1") ~shape:[ Int 16 ] 0 Float32
          in
          let copied = Ops.copy_to_device (input 1) (Single "CPU:2") in
          let u =
            prepare
              (Ops.sink [ Ops.after elsewhere [ Ops.store elsewhere copied ] ])
          in
          equal
            (list (Testable.make ~pp:Ops.pp_device ~equal:Ops.equal_device))
            [ Single "CPU:2" ]
            (List.filter_map
               (fun n -> if Ops.op n = Alloc then Ops.device n else None)
               (Ops.toposort u)));
      test "the second of two equal stores into one storage is dropped"
        (fun () ->
          let v = input 1 in
          let first = Ops.after out [ Ops.store out v ] in
          equal uop (Ops.sink [ first ])
            (prepare (Ops.sink [ Ops.after first [ Ops.store first v ] ])));
      test "a store into a bitcast of storage stores the value bitcast"
        (fun () ->
          let v = Ops.param ~device:cpu ~shape:[ Int 16 ] 1 Int32 in
          equal uop
            (stores (Ops.bitcast v Float32))
            (prepare (Ops.sink [ Ops.store (Ops.bitcast out Int32) v ])));
      test "a bitcast on a disk keeps its size" (fun () ->
          let disk =
            Ops.param ~device:(Single "DISK:/tmp/tolk-next") ~shape:[ Int 64 ] 1
              Uint8
          in
          let u =
            prepare (stores (Ops.copy_to_device (Ops.bitcast disk Float32) cpu))
          in
          is_true
            (List.exists
               (fun n -> Ops.op n = Bitcast && Ops.dtype (Ops.nth n 0) = Uint8)
               (Ops.toposort u)));
      test "a value with an empty axis is zero" (fun () ->
          let none = Ops.param ~device:cpu ~shape:[ Int 0 ] 0 Float32 in
          let nothing = Ops.param ~device:cpu ~shape:[ Int 0 ] 1 Float32 in
          let u =
            prepare
              (Ops.sink
                 [ Ops.after none [ Ops.store none (Ops.exp2 nothing) ] ])
          in
          is_false (has Exp2 u));
      test "a no-op among a sink's effects is dropped" (fun () ->
          let noop = Ops.v Noop in
          let with_noop =
            Ops.sink [ noop; Ops.after out [ Ops.store out (input 1) ] ]
          in
          equal uop (stores (input 1)) (prepare with_noop));
    ]

let () =
  exit
    (run "Tolk_next.Prepare"
       [
         recorded;
         values;
         pm_mops_rules;
         pm_mops_laws;
         pads;
         contiguous_views;
         contiguous_view_laws;
         outputs;
         calls_inline;
         earliest;
       ])
