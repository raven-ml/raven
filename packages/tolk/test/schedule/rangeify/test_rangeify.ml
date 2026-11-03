(* Tests of Tolk.Rangeify: the kernel graphs tinygrad makes of the graphs it
   schedules, their kernel counts, the law that a kernel graph writes what its
   tensor graph does, and the rules of get_kernel_graph. *)

open Windtrap
open Tolk

let program name = Golden.sink (name ^ ".golden")

(* The settings a program was recorded under. *)
let settings = function
  | "many_inputs_limited" | "many_matrices_limited" | "many_cubes_limited"
  | "many_sums_limited" ->
      [ Setting.B (Setting.max_kernel_buffers, 4) ]
  | _ -> []

let kernel_graph name =
  Setting.context (settings name) (fun () ->
      Rangeify.get_kernel_graph (program name))

(* Recorded graphs *)

(* Gathers: tensor graphs of UOps in which an index by an integer value with a
   shape, clamped into range, reads rows of its source. *)
let gathers =
  [
    "index_rows";
    "index_rows_computed";
    "index_read_twice";
    "index_under_reduce";
    "index_broadcast";
    "index_view_source";
    "index_zip";
    "index_of_index";
    "index_zero_fill";
    "index_stored_zero_fill";
    "index_assign_self";
  ]

let programs =
  [
    "add";
    "allow_push_permutes";
    "arange";
    "arange_sum";
    "argmax";
    "assign";
    "assign_double_diamond";
    "assign_permuted";
    "assigned_contiguous";
    "assigned_read_twice";
    "attention";
    "base_change_expand_pad";
    "base_change_pad_expand";
    "binop_permute";
    "binop_reshape";
    "cast_half";
    "cat";
    "children_dont_push";
    "clone";
    "contiguous";
    "contiguous_add";
    "conv";
    "copy";
    "crossed_pairs";
    "cumsum";
    "custom_kernel";
    "custom_kernel_in_place";
    "custom_kernel_of_views";
    "custom_kernel_permuted";
    "disk_bitcast_to";
    "disk_store";
    "disk_store_1";
    "disk_view_to";
    "div_collapse";
    "double_matmul";
    "elementwise_three";
    "embedding";
    "swiglu_down";
    "exp_dead_axis";
    "exp_cheap_consumer";
    "rope_decode";
    "normed_vecmat";
    "normed_vecmats";
    "normed_matmul";
    "gather_broadcast";
    "gather_read_twice";
    "gather_rotary";
    "empty_sum";
    "expand_before_cast";
    "expand_kept";
    "expand_staged";
    "finite_checks";
    "full_invalid";
    "inline_function";
    "invalids_read";
    "layernorm";
    "long_cumsum";
    "many_cubes_limited";
    "many_inputs";
    "many_inputs_limited";
    "many_matrices_limited";
    "many_sums_limited";
    "matmul";
    "maxpool";
    "mean_kept_broadcast";
    "mulacc";
    "multimatmul";
    "multireduce_diffops_parallel";
    "multireduce_midreduce_nochase";
    "multireduce_parallel";
    "multireduce_push_shrink_chase";
    "multistage_reduce";
    "pad";
    "pad_reduce_safe";
    "pad_reduce_unsafe";
    "padded_pair";
    "padded_twice";
    "partial_fuse";
    "partially_invalid";
    "permute_arange";
    "permute_through_reshape";
    "precompiled_function";
    "precompiled_function_1";
    "preserve_multistage_reduce";
    "push_pads_elementwise";
    "read_around_assign";
    "read_at_offsets";
    "reduce_broadcast_not_recomputed";
    "reduce_expand_child";
    "reduce_expand_reduce";
    "reduce_ext_reduce_child";
    "reduce_multiple_paths";
    "reduce_permute_binop";
    "reduce_permute_nofuse";
    "reduce_reshape_binop";
    "reduce_same_size";
    "reduce_shrink";
    "reduce_unary";
    "reshape_chain";
    "setitem";
    "setitem_column";
    "setitem_cube";
    "setitem_tensor";
    "shared_sum";
    "shrink_fuse";
    "shrink_pad_unsafe";
    "softmax";
    "sort";
    "stack";
    "stage_of_assigned";
    "std";
    "sum";
    "sum_all";
    "sum_all_broadcast";
    "sum_kept_broadcast";
    "symbolic_contiguous";
    "prefix_sum_broadcast";
    "symbolic_kept";
    "tiled_chain";
    "two_outputs";
    "ugly_reduceop_pairing";
    "unit_axis_staged";
    "variable_offset";
    "variable_read_twice";
    "variable_reduce";
    "variable_same";
    "variable_shrink";
    "variable_staged";
    "flipped_store";
    "store_into_flipped";
    "variable_flipped_store";
    "variable_two";
    "where";
    "windowed_chain";
    "windowed_pair";
    "read_by_two";
    "long_read_by_two";
    "scalar_read_by_two";
    "long_scalar_read_by_two";
    "scalar_chain";
    "routed_product_2";
    "routed_product_64";
    "routed_product_wide";
    "blocked_product_16";
    "dense_decoded_64";
    "windowed_scaled";
    "windowed_scaled_long";
    "zero_size_children";
  ]
  @ gathers

let recorded =
  group "get_kernel_graph › recorded"
    (List.map
       (fun name ->
         Golden.graph (name ^ "_kernels.golden") (fun () -> kernel_graph name))
       programs)

let is_kernel u = Ops.op u = Call && Ops.op (Ops.nth u 0) = Sink
let kernels u = List.length (List.filter is_kernel (Ops.toposort ~calls:Skip u))

let counts =
  Golden.cases "kernel_counts.golden" (fun cell ->
      equal int
        (int_of_string (cell "kernels"))
        (kernels (kernel_graph (cell "program"))))

let gather_counts =
  Golden.cases "gather_kernel_counts.golden" (fun cell ->
      equal int
        (int_of_string (cell "kernels"))
        (kernels (kernel_graph (cell "program"))))

(* Values

   The law that a kernel graph writes what its tensor graph does: running its
   kernels in order (Kernel_graphs) leaves in the storage the function is given,
   its parameters and buffers, what the tensor graph writes there (Tensors).
   Memory holds small integers. *)

let element dtype k : Dtype.value =
  if Dtype.is_float dtype then `Float (float_of_int k)
  else if Dtype.equal dtype Bool then `Bool (k > 0)
  else if Dtype.is_unsigned dtype then `Int (Bigint.of_int (k + 3))
  else `Int (Bigint.of_int k)

let storage u =
  List.filter_map
    (fun n ->
      match (Ops.op n, Ops.arg n) with
      | (Param | Buffer | Alloc), Param p when p.addrspace <> Alu ->
          Some (n, p)
      | _ -> None)
    (Ops.toposort ~calls:Enter u)

let filled u =
  List.map
    (fun (n, (p : Ops.param_arg)) ->
      let size = Shape.max_numel n in
      let at j = element p.dtype ((((j * 7) + (p.slot * 3)) mod 11) - 3) in
      (p.slot, Array.init size at))
    (storage u)

let close = Testable.float_rel ~rel:1e-6 ~abs:0.

(* Floats agree up to the rounding of float32, since kernels reduce in another
   order than the tensor graph does, and NaN is NaN. *)
let value =
  Testable.make ~pp:(Testable.pp Dtypes.value) ~equal:(fun v0 v1 ->
      match (v0, v1) with
      | `Float x, `Float y ->
          (Float.is_nan x && Float.is_nan y) || Testable.equal close x y
      | v0, v1 -> Testable.equal Dtypes.value v0 v1)

let write = triple int int value

let given u =
  List.filter_map
    (fun (n, (p : Ops.param_arg)) ->
      if Ops.op n = Alloc then None else Some p.slot)
    (storage u)

(* A program of more than 30,000 elements, or a scan of a thousand, takes
   seconds to run element by element. *)
let heavy = [ "long_cumsum"; "preserve_multistage_reduce" ]

let keeps_writes name =
  (if List.mem name heavy then slow else Fun.id)
  @@ test (name ^ " writes what its tensors write") (fun () ->
      let sink = program name in
      let buffers = filled sink in
      let into_given = List.filter (fun (s, _, _) -> List.mem s (given sink)) in
      equal (list write)
        (into_given (Tensors.writes ~buffers sink))
        (into_given (Kernel_graphs.writes ~buffers (kernel_graph name))))

(* Programs the law does not apply to: custom kernels, whose bodies keep their
   views until they are compiled, a compiled schedule, and symbolic shapes,
   which Tensors does not run. *)
let unevaluated =
  [
    "custom_kernel";
    "custom_kernel_in_place";
    "custom_kernel_of_views";
    "custom_kernel_permuted";
    "precompiled_function_1";
    "symbolic_contiguous";
    "symbolic_kept";
    "variable_offset";
    "variable_read_twice";
    "variable_reduce";
    "variable_same";
    "variable_two";
    "variable_shrink";
    "variable_staged";
    "variable_flipped_store";
  ]

let values =
  group "get_kernel_graph › values"
    (List.filter_map
       (fun name ->
         if List.mem name unevaluated then None else Some (keeps_writes name))
       programs)

(* Kernel graphs

   What every kernel graph is, stated over the recorded ones: each kernel's
   storage parameters are numbered from [0] and match its call's arguments, its
   ranges are numbered from [0], the arguments are storage without views, the
   storage it makes is never of a weak type, and each is read after it is
   written. *)

let calls u = List.filter is_kernel (Ops.toposort ~calls:Skip u)

let storage_params body =
  List.filter_map
    (fun n ->
      match (Ops.op n, Ops.arg n) with
      | Param, Param p when p.addrspace <> Alu -> Some (n, p)
      | _ -> None)
    (Ops.toposort ~calls:Enter body)

let rec storage_of u =
  match Ops.op u with
  | After -> storage_of (Ops.nth u 0)
  | Param | Buffer | Alloc -> Some u
  | _ -> None

let kernel_graph_law name claim check =
  test (name ^ " " ^ claim) (fun () -> check (kernel_graph name))

let numbered u =
  List.iter
    (fun c ->
      let params = storage_params (Ops.nth c 0) in
      let slots =
        List.sort_uniq compare
          (List.map (fun (_, (p : Ops.param_arg)) -> p.slot) params)
      in
      let args = List.tl (Ops.src c) in
      equal ~msg:"slots" (list int) (List.init (List.length slots) Fun.id) slots;
      at_most ~msg:"arguments" int ~than:(List.length args) (List.length slots))
    (calls u)

let ranges_from_zero u =
  List.iter
    (fun c ->
      let ids =
        List.sort_uniq compare
          (List.filter_map
             (fun n ->
               if Ops.op n <> Range then None
               else
                 match Ops.axis_id n with
                 | id :: _ when id >= 0 -> Some id
                 | _ -> None)
             (Ops.toposort ~calls:Enter (Ops.nth c 0)))
      in
      equal ~msg:"range ids" (list int) (List.init (List.length ids) Fun.id) ids)
    (calls u)

let arguments_are_storage u =
  List.iter
    (fun c ->
      List.iter
        (fun a ->
          match (Ops.op a, storage_of a) with
          | (Param | Buffer | Alloc | After), Some _ -> ()
          | _ when Ops.op a = Param -> ()
          | op, _ -> failf "a call's argument is a %a" Op.pp op)
        (List.tl (Ops.src c)))
    (calls u)

let made_storage u =
  List.filter (fun n -> Ops.op n = Alloc) (Ops.toposort ~calls:Skip u)

let strong_storage u =
  List.iter
    (fun a -> is_false ~msg:"weak storage" (List.mem (Ops.dtype a) Dtype.weaks))
    (made_storage u)

(* [reads_after a arg] is [true] iff the argument [arg] is [a] after effects. *)
let reads_after a arg = Ops.op arg = After && Ops.nth arg 0 == a

(* Storage that no kernel writes, because every value stored into it was
   invalid, may still be read. *)
let read_after_written u =
  let written a =
    List.exists
      (fun n -> Ops.op n = After && Ops.nth n 0 == a)
      (Ops.toposort ~calls:Skip u)
  in
  List.iter
    (fun a ->
      if written a then
        is_true
          ~msg:(Graph.to_string (Ops.sink [ a ]))
          (List.exists
             (fun c -> List.exists (reads_after a) (List.tl (Ops.src c)))
             (calls u)))
    (made_storage u)

let structure =
  group "get_kernel_graph › kernel graphs"
    (List.concat_map
       (fun name ->
         [
           kernel_graph_law name "numbers each kernel's storage from 0" numbered;
           kernel_graph_law name "numbers each kernel's ranges from 0"
             ranges_from_zero;
           kernel_graph_law name "passes storage to its kernels"
             arguments_are_storage;
           kernel_graph_law name "makes storage of committed types"
             strong_storage;
           kernel_graph_law name "reads each storage it makes after writing it"
             read_after_written;
         ])
       programs)

let chomp s =
  if String.ends_with ~suffix:"\n" s then String.sub s 0 (String.length s - 1)
  else s

let debug =
  group "get_kernel_graph › debug"
    [
      Golden.text "softmax_debug.golden" (fun () ->
          Setting.context
            [
              Setting.B (Setting.debug_rangeify, true);
              Setting.B (Setting.no_color, true);
            ]
            (fun () -> ignore (kernel_graph "softmax"));
          chomp (output ()));
      test "without debug_rangeify, nothing is printed" (fun () ->
          ignore (kernel_graph "softmax");
          equal string "" (output ()));
    ]

let spec =
  group "get_kernel_graph › spec"
    [
      test "with spec checks on, every recorded kernel graph passes them"
        (fun () ->
          Setting.context
            [ Setting.B (Setting.spec, 1) ]
            (fun () ->
              List.iter (fun name -> ignore (kernel_graph name)) programs));
    ]

(* Rules

   Hand-built functions on the CPU, prepared, then turned into kernel graphs: an
   output of sixteen elements in slot [0], inputs in the next slots. *)

let cpu = "CPU"
let ints = List.map (fun n -> Ops.Int n)

let input ?(shape = [ 4; 4 ]) slot =
  Shape.reshape
    (Shape.param ~device:cpu
       ~shape:[ Int (List.fold_left ( * ) 1 shape) ]
       slot Float32)
    (ints shape)

let out = Shape.param ~device:cpu ~shape:[ Int 16 ] 0 Float32

let stores value =
  Ops.sink
    [
      Ops.after out [ Ops.store (Shape.reshape out (Shape.shape value)) value ];
    ]

let schedule sink = Rangeify.get_kernel_graph (Prepare.prepare_rangeify sink)
let kernels_of sink = kernels (schedule sink)

let sum_of us =
  List.fold_left (fun acc u -> Ops.O.(acc + u)) (List.hd us) (List.tl us)

let read_twice v = Ops.O.(v + Shape.permute v [ 1; 0 ])

let rules =
  group "get_kernel_graph › rules"
    [
      test "a value read twice and reading three storages is recomputed"
        (fun () ->
          equal int 1
            (kernels_of
               (stores (read_twice (sum_of (List.map input [ 1; 2; 3 ]))))));
      test "a value read twice and reading four storages is stored" (fun () ->
          equal int 2
            (kernels_of
               (stores (read_twice (sum_of (List.map input [ 1; 2; 3; 4 ]))))));
      test "a store of storage into itself runs no kernel" (fun () ->
          equal int 0
            (kernels
               (Rangeify.get_kernel_graph
                  (Ops.sink [ Ops.after out [ Ops.store out out ] ]))));
      test "a store of invalid values runs no kernel" (fun () ->
          let invalid =
            Shape.expand (Ops.const ~dtype:Float32 `Invalid) (ints [ 16 ])
          in
          equal int 0
            (kernels_of (Ops.sink [ Ops.after out [ Ops.store out invalid ] ])));
      test "a kernel accesses at most max_kernel_buffers storages" (fun () ->
          let u =
            Setting.context
              [ Setting.B (Setting.max_kernel_buffers, 4) ]
              (fun () -> kernel_graph "many_inputs_limited")
          in
          List.iter
            (fun c -> at_most int ~than:4 (List.length (List.tl (Ops.src c))))
            (calls u));
      test "without a limit, one kernel reads every storage" (fun () ->
          equal int 1 (kernels (kernel_graph "many_inputs")));
      test "a materialisation of a symbolic size takes its greatest size"
        (fun () ->
          let v =
            Ops.variable "v" (`Int (Bigint.of_int 1)) (`Int (Bigint.of_int 16))
          in
          let r = Ops.range ~axis_type:Loop (Sym v) [ 0 ] in
          let r' = Ops.range ~axis_type:Loop (Sym v) [ 1 ] in
          let opts : Ops.bufferize_opts =
            { device = Some cpu; addrspace = Global; keep = Whole }
          in
          let staged =
            Ops.bufferize ~opts (Ops.exp2 (Ops.index out [ r ])) [ r ]
          in
          let sink =
            Ops.sink
              [
                Ops.after out
                  [
                    Ops.end_
                      (Ops.store (Ops.index out [ r' ])
                         (Ops.index staged [ r' ]))
                      [ r' ];
                  ];
              ]
          in
          equal (list int) [ 16 ]
            (List.filter_map
               (fun n ->
                 if Ops.op n = Alloc then Some (Shape.max_numel n) else None)
               (Ops.toposort ~calls:Enter (Rangeify.get_kernel_graph sink))));
    ]

(* Outputs that share a computation

   Stored values of one shape and device whose computations share a node
   reached through no view, or one view of a computation, and neither of which
   reads the other, run as one kernel that writes both. A double word's high
   and low parts are one chain's two ends. Two views of one computation read
   it at different indices: slices of it run as kernels of their own. *)

let out2 = Shape.param ~device:cpu ~shape:[ Int 16 ] 4 Float32

let stores2 ?(size = 16) v w =
  let at slot = Shape.param ~device:cpu ~shape:[ Int size ] slot Float32 in
  let store o v =
    Ops.after o [ Ops.store (Shape.reshape o (Shape.shape v)) v ]
  in
  Ops.sink [ store (at 0) v; store (at 4) w ]

(* Knuth's two-sum of [a] and [b]: the rounded sum and its error. *)
let two_sum a b =
  let open Ops.O in
  let s = a + b in
  let bb = s - a in
  (s, a - (s - bb) + (b - bb))

let writes sink =
  let buffers = filled sink in
  equal (list write)
    (Tensors.writes ~buffers sink)
    (Kernel_graphs.writes ~buffers (schedule sink))

let shared =
  let hi, lo = two_sum (input 1) (input 2) in
  let x = input 1 in
  let v = Ops.O.(input 1 * input 2) in
  group "get_kernel_graph › outputs that share a computation"
    [
      test "a two-sum's sum and error run as one kernel" (fun () ->
          equal int 1 (kernels_of (stores2 hi lo)));
      test "a two-sum's sum and error are written as their tensors write them"
        (fun () -> writes (stores2 hi lo));
      test "outputs that share only a load run as two kernels" (fun () ->
          equal int 2 (kernels_of (stores2 Ops.O.(x + x) Ops.O.(x * x))));
      test "outputs that share a reduction run as one kernel" (fun () ->
          let m = Shape.rop (input 1) Max [ 1 ] in
          let sink =
            stores2 ~size:4 Ops.O.(m + float 1.) Ops.O.(m * float 2.)
          in
          equal int 1 (kernels_of sink);
          writes sink);
      test "outputs that share a broadcast reduction read it from its kernel"
        (fun () ->
          let m =
            Shape.expand (Shape.rop (input 1) Max [ 1 ]) (ints [ 4; 4 ])
          in
          let sink = stores2 Ops.O.(m + input 2) Ops.O.(m * input 3) in
          equal int 2 (kernels_of sink);
          writes sink);
      test "outputs with a reduction each run as two kernels" (fun () ->
          let sink =
            stores2 ~size:4 (Shape.rop v Add [ 1 ]) (Shape.rop v Max [ 1 ])
          in
          equal int 2 (kernels_of sink);
          writes sink);
      test "slices of one product through two views run as two kernels"
        (fun () ->
          let rows lo = Shape.shrink v [ Some (Int lo, Int (lo + 2)); None ] in
          let sink = stores2 ~size:8 (rows 0) (rows 2) in
          equal int 2 (kernels_of sink);
          writes sink);
      test "slices that also share a computation run as two kernels"
        (fun () ->
          let rows lo = Shape.shrink v [ Some (Int lo, Int (lo + 2)); None ] in
          let e = Ops.exp2 (input ~shape:[ 2; 4 ] 3) in
          let sink = stores2 ~size:8 Ops.O.(rows 0 + e) Ops.O.(rows 2 * e) in
          equal int 2 (kernels_of sink);
          writes sink);
      test "outputs that read a product through one view run as one kernel"
        (fun () ->
          let w = Shape.shrink v [ Some (Int 1, Int 3); None ] in
          let sink =
            stores2 ~size:8 Ops.O.(w + float 1.) Ops.O.(w * float 2.)
          in
          equal int 1 (kernels_of sink);
          writes sink);
      test "outputs of shapes that differ run as two kernels" (fun () ->
          let w = Shape.reshape v (ints [ 2; 8 ]) in
          equal int 2 (kernels_of (stores2 v Ops.O.(w * w))));
      test "an output that reads what another writes runs after it" (fun () ->
          let first =
            Ops.after out [ Ops.store (Shape.reshape out (ints [ 4; 4 ])) v ]
          in
          let read = Shape.reshape first (ints [ 4; 4 ]) in
          let sink =
            Ops.sink
              [
                first;
                Ops.after out2
                  [
                    Ops.store
                      (Shape.reshape out2 (ints [ 4; 4 ]))
                      Ops.O.((v * v) + read);
                  ];
              ]
          in
          equal int 2 (kernels_of sink);
          writes sink);
      test "an output that overwrites what another reads runs apart" (fun () ->
          let old = Shape.reshape out2 (ints [ 4; 4 ]) in
          let sink = stores2 Ops.O.(v + old) Ops.O.(v * v) in
          equal int 2 (kernels_of sink);
          writes sink);
      test "outputs reading a storage before and after its overwrite run apart"
        (fun () ->
          let a = Shape.param ~device:cpu ~shape:[ Int 16 ] 5 Float32 in
          let written = Ops.after a [ Ops.store a Ops.O.(a * a) ] in
          let at u = Shape.reshape u (ints [ 4; 4 ]) in
          let store o v = Ops.after o [ Ops.store (at o) v ] in
          let sink =
            Ops.sink
              [
                written;
                store out Ops.O.(v + at a);
                store out2 Ops.O.((v * v) + at written);
              ]
          in
          equal int 3 (kernels_of sink);
          writes sink);
      test "outputs that run apart are checked once, however much they share"
        (fun () ->
          let old = Shape.reshape out2 (ints [ 4; 4 ]) in
          List.iter
            (fun n ->
              let v =
                Seq.fold_left (fun u _ -> Ops.exp2 u) v (Seq.init n Fun.id)
              in
              let sink = stores2 Ops.O.(v + old) Ops.O.(v * v) in
              let before = Indexing.store_checks () in
              equal int 2 (kernels_of sink);
              equal
                ~msg:(Printf.sprintf "%d shared operations" (n + 1))
                int 1
                (Indexing.store_checks () - before))
            [ 0; 64 ]);
      test "two pairs of outputs that each read the other pair run as three"
        (fun () ->
          (* [a] and [b] share [p], [c] and [d] share [q], [b] reads [c] and
             [d] reads [a]: one kernel of each pair would run both before and
             after the other. *)
          let at u = Shape.reshape u (ints [ 4; 4 ]) in
          let store o v = Ops.after o [ Ops.store (at o) v ] in
          let out3 = Shape.param ~device:cpu ~shape:[ Int 16 ] 6 Float32 in
          let out4 = Shape.param ~device:cpu ~shape:[ Int 16 ] 7 Float32 in
          let p = Ops.O.(input 1 * input 2) and q = Ops.O.(input 1 - input 2) in
          let a = store out p and c = store out3 q in
          let sink =
            Ops.sink
              [
                a;
                store out2 Ops.O.((p * p) + at c);
                c;
                store out4 Ops.O.((q * q) + at a);
              ]
          in
          equal int 3 (kernels_of sink);
          writes sink);
    ]

(* Stores that read their destination

   A store whose value reads its destination only at the element it writes,
   through the destination itself and elementwise operations, runs as one
   kernel when no other store reads the destination: each element is loaded
   before it is overwritten. Read at another element or in a reduction, or
   read by another store, the value is staged. A sum in the value reading a
   storage keeps the stage stored, so that the stage is seen in the kernel
   counts. *)

let own = Shape.reshape out (ints [ 4; 4 ])

let reduced =
  Shape.reshape
    (Shape.rop (input ~shape:[ 4; 4; 4 ] 1) Add [ 2 ])
    (ints [ 4; 4 ])

let self_reads =
  group "get_kernel_graph › stores that read their destination"
    [
      test
        "a value that reads its destination at the element written is one \
         kernel" (fun () ->
          let sink = stores Ops.O.((own * float 2.) + Ops.exp2 own + reduced) in
          equal int 1 (kernels_of sink);
          writes sink);
      test
        "a value of five storages that reads its destination at the element \
         written is one kernel" (fun () ->
          let others = sum_of (List.map input [ 1; 2; 3; 4 ]) in
          let sink = stores Ops.O.(own + others) in
          equal int 1 (kernels_of sink);
          writes sink);
      test "a value that reads its destination at another element is staged"
        (fun () ->
          let sink = stores Ops.O.(Shape.permute own [ 1; 0 ] + reduced) in
          equal int 2 (kernels_of sink);
          writes sink);
      (* Two kernels: the row sums, which read the destination before the
         store overwrites it, and the store. *)
      test "a value that reduces its destination reads the sums first"
        (fun () ->
          let sums = Shape.reshape (Shape.rop own Add [ 1 ]) (ints [ 4; 1 ]) in
          let sink =
            stores Ops.O.(own + Shape.expand sums (ints [ 4; 4 ]) + reduced)
          in
          equal int 2 (kernels_of sink);
          writes sink);
      test
        "a value that reads a slice of its destination at the element written \
         is one kernel" (fun () ->
          let rows u = Shape.shrink u [ Some (Ops.Int 0, Ops.Int 2); None ] in
          let dest = rows own in
          let sink =
            Ops.sink
              [ Ops.after out [ Ops.store dest Ops.O.(dest + rows reduced) ] ]
          in
          equal int 1 (kernels_of sink);
          writes sink);
      test
        "a value that reads its destination through a bitcast of one size is \
         one kernel" (fun () ->
          let bits = Ops.bitcast own Int32 in
          let sink =
            stores Ops.O.(Ops.bitcast (bits lsr Ops.bitcast reduced Int32) Float32)
          in
          equal int 1 (kernels_of sink);
          writes sink);
      (* Through the expand, the four rows of [dest] are one element each of
         [row], and each stores the same sum: an element loaded after another
         row stored it would add the column's sum again. *)
      test "a value stored into a broadcast of its destination is staged"
        (fun () ->
          let row = Shape.param ~device:cpu ~shape:[ Int 4 ] 5 Float32 in
          let wide u = Shape.expand u (ints [ 4; 4 ]) in
          let dest = wide (Shape.reshape row (ints [ 1; 4 ])) in
          let sums = wide (Shape.rop (input 1) Add [ 0 ]) in
          let sink =
            Ops.sink [ Ops.after row [ Ops.store dest Ops.O.(dest + sums) ] ]
          in
          equal int ~msg:"the sums, the stage and its store" 3
            (List.length
            (List.filter_map
               (fun u ->
                 if Ops.op u = End && Ops.op (Ops.nth u 0) = Store then
                   Some (List.length (Ops.src u) - 1)
                 else None)
               (Ops.toposort ~calls:Enter (schedule sink))));
          writes sink);
      (* Two kernels: the other store reads the destination before its
         overwrite. *)
      test "a value whose destination another store reads runs after it"
        (fun () ->
          let sink =
            Ops.sink
              [
                Ops.after out [ Ops.store own Ops.O.(own + reduced) ];
                Ops.after out2
                  [
                    Ops.store
                      (Shape.reshape out2 (ints [ 4; 4 ]))
                      Ops.O.(own * float 3.);
                  ];
              ]
          in
          equal int 2 (kernels_of sink);
          writes sink);
    ]

(* Reductions that share a domain

   Reductions whose outputs a kernel indexes alike and that reduce axes of the
   same sizes run one loop when neither reads the other: the nodes their sources
   share are computed once. *)

let small = Shape.param ~device:cpu ~shape:[ Int 4 ] 0 Float32

let stores_small value =
  Ops.sink
    [
      Ops.after small
        [ Ops.store (Shape.reshape small (Shape.shape value)) value ];
    ]

(* The loops each kernel reduces over: the ranges its reductions close. *)
let reduce_loops sink =
  let closed u =
    if Ops.op u = Reduce then
      List.filter (fun r -> Ops.op r = Range) (List.tl (Ops.src u))
    else []
  in
  List.map
    (fun c ->
      List.length
        (List.sort_uniq Ops.compare
           (List.concat_map closed (Ops.toposort ~calls:Skip (Ops.nth c 0)))))
    (calls (schedule sink))

let siblings =
  let v = Ops.O.(input 1 * input 2) in
  group "get_kernel_graph › reductions that share a domain"
    [
      test "a sum and a maximum of one computation run one loop" (fun () ->
          let sink =
            stores_small Ops.O.(Shape.rop v Add [ 1 ] + Shape.rop v Max [ 1 ])
          in
          equal (list int) [ 1 ] (reduce_loops sink);
          writes sink);
      test "a reduction that reads another of its domain runs its own loop"
        (fun () ->
          let x = input 1 in
          let sums = Shape.reshape (Shape.rop x Add [ 1 ]) (ints [ 4; 1 ]) in
          let centred = Ops.O.(x - Shape.expand sums (ints [ 4; 4 ])) in
          let sink =
            stores_small (Shape.rop Ops.O.(centred * centred) Add [ 1 ])
          in
          equal int 2 (List.fold_left ( + ) 0 (reduce_loops sink));
          writes sink);
      test "reductions of axes of different sizes run a loop each" (fun () ->
          let w = input ~shape:[ 4; 8 ] 3 in
          let sink =
            stores_small Ops.O.(Shape.rop v Add [ 1 ] + Shape.rop w Add [ 1 ])
          in
          equal (list int) [ 2 ] (reduce_loops sink);
          writes sink);
    ]

(* Generated programs

   The law of scheduling end to end: a function of up to three inputs of up to
   three axes, through a few operations, materialisations among them, prepared
   and turned into a kernel graph, writes into its output what its tensor graph
   does. It exercises Indexing, Prepare and Rangeify together. *)

type step =
  | Add_self
  | Scale of int
  | Add_input
  | Sum of int
  | Max of int
  | Sum_max of int
  | Center of int
  | Rotate of int
  | Flip of int
  | Pad of int
  | Shrink of int
  | Unsqueeze of int
  | Expand
  | Stage
  | Read_twice

let pp_step ppf = function
  | Add_self -> Format.fprintf ppf "x + x"
  | Scale k -> Format.fprintf ppf "x * %d" k
  | Add_input -> Format.fprintf ppf "x + an input"
  | Sum k -> Format.fprintf ppf "sum %d" k
  | Max k -> Format.fprintf ppf "max %d" k
  | Sum_max k -> Format.fprintf ppf "sum %d + max %d" k k
  | Center k -> Format.fprintf ppf "x - sum %d" k
  | Rotate k -> Format.fprintf ppf "rotate by %d" k
  | Flip k -> Format.fprintf ppf "flip %d" k
  | Pad k -> Format.fprintf ppf "pad %d" k
  | Shrink k -> Format.fprintf ppf "shrink %d" k
  | Unsqueeze k -> Format.fprintf ppf "unsqueeze %d" k
  | Expand -> Format.fprintf ppf "expand"
  | Stage -> Format.fprintf ppf "materialise"
  | Read_twice -> Format.fprintf ppf "x + x transposed"

let step =
  Gen.(
    let* k = int_range 0 7 in
    of_list ~pp:pp_step
      [
        Add_self;
        Scale (k - 3);
        Add_input;
        Sum k;
        Max k;
        Sum_max k;
        Center k;
        Rotate k;
        Flip k;
        Pad k;
        Shrink k;
        Unsqueeze k;
        Expand;
        Stage;
        Read_twice;
      ])

let pp_program ppf (shape, steps) =
  Format.fprintf ppf "[%s], then %a"
    (String.concat "; " (List.map string_of_int shape))
    (Format.pp_print_list
       ~pp_sep:(fun ppf () -> Format.fprintf ppf ", ")
       pp_step)
    steps

let program =
  Gen.(
    let* shape = list ~size:(int_range 1 3) (int_range 1 4) in
    let+ steps = list ~size:(int_range 1 5) step in
    (shape, steps))
  |> Gen.with_pp pp_program

let dims u =
  List.map
    (function Ops.Int n -> n | Sym _ -> fail "a concrete shape")
    (Shape.shape u)

let pick k = function
  | [] -> None
  | l -> Some (List.nth l (k mod List.length l))

(* The function [shape, steps] draws, and its inputs' memory. *)
let build (shape, steps) =
  let memory = ref [] in
  let fresh shape =
    let slot = List.length !memory + 1 in
    let size = List.fold_left ( * ) 1 shape in
    memory :=
      ( slot,
        Array.init size (fun j ->
            element Float32 ((((j * 7) + (slot * 3)) mod 11) - 3)) )
      :: !memory;
    input ~shape slot
  in
  let apply v step =
    let shape = dims v in
    let rank = List.length shape in
    let on a f = List.mapi (fun b n -> if a = b then f n else None) shape in
    let axes = List.init rank Fun.id in
    match step with
    | Add_self -> Ops.O.(v + v)
    | Scale k -> Ops.O.(v * float (float_of_int k))
    | Add_input when List.length !memory < 3 -> Ops.O.(v + fresh shape)
    | Sum k when rank > 0 -> Shape.rop v Add [ k mod rank ]
    | Max k when rank > 0 -> Shape.rop v Max [ k mod rank ]
    | Sum_max k when rank > 0 ->
        Ops.O.(Shape.rop v Add [ k mod rank ] + Shape.rop v Max [ k mod rank ])
    | Center k when rank > 0 ->
        let a = k mod rank in
        let kept = List.mapi (fun b n -> if b = a then 1 else n) shape in
        let sums = Shape.reshape (Shape.rop v Add [ a ]) (ints kept) in
        Ops.O.(v - Shape.expand sums (ints shape))
    | Rotate k when rank > 0 ->
        Shape.permute v (List.init rank (fun i -> (i + k) mod rank))
    | Flip k -> (
        match pick k axes with Some a -> Shape.flip v [ a ] | None -> v)
    | Pad k -> (
        match pick k axes with
        | Some a -> Shape.pad v (on a (fun _ -> Some (Ops.Int 1, Ops.Int 2)))
        | None -> v)
    | Shrink k -> (
        match pick k (List.filter (fun a -> List.nth shape a > 1) axes) with
        | Some a -> Shape.shrink v (on a (fun n -> Some (Ops.Int 1, Ops.Int n)))
        | None -> v)
    | Unsqueeze k ->
        let p = k mod (rank + 1) in
        Shape.reshape v
          (ints
             (List.filteri (fun a _ -> a < p) shape
             @ (1 :: List.filteri (fun a _ -> a >= p) shape)))
    | Expand -> Shape.expand v (ints (2 :: shape))
    | Stage -> Ops.v Stage ~src:[ v ]
    | Read_twice when rank = 2 && List.nth shape 0 = List.nth shape 1 ->
        Ops.O.(v + Shape.permute v [ 1; 0 ])
    | _ -> v
  in
  let value = List.fold_left apply (fresh shape) steps in
  let size = List.fold_left ( * ) 1 (dims value) in
  let result = Shape.param ~device:cpu ~shape:[ Int size ] 0 Float32 in
  let sink =
    Ops.sink
      [
        Ops.after result
          [ Ops.store (Shape.reshape result (Shape.shape value)) value ];
      ]
  in
  (sink, !memory)

(* Whether two reductions of a kernel reduce over one range. *)
let shares_a_loop kernels =
  List.exists
    (fun c ->
      let reduces =
        List.filter
          (fun u -> Ops.op u = Reduce)
          (Ops.toposort ~calls:Skip (Ops.nth c 0))
      in
      List.exists
        (fun r ->
          List.exists
            (fun r' ->
              r != r'
              && List.exists
                   (fun u -> List.memq u (List.tl (Ops.src r')))
                   (List.tl (Ops.src r)))
            reduces)
        reduces)
    (calls kernels)

let writes_what_it_computes ?(cover = fun _ _ -> ()) drawn =
  let sink, buffers = build drawn in
  let kernels = schedule sink in
  let expected = Tensors.writes ~buffers sink in
  cover "several kernels" (List.length (calls kernels) > 1);
  cover "reductions that share a loop" (shares_a_loop kernels);
  cover "a value is written" (expected <> []);
  equal (list write) expected (Kernel_graphs.writes ~buffers kernels)

(* Programs the law once failed on. A sum over a padded axis of size 1 folds
   into the value times its own validity, which reads the input at an invalid
   index where it is padding: that read is zero. *)
let counterexamples = [ ([ 1 ], [ Pad 0; Unsqueeze 0; Pad 0; Sum 0 ]) ]

let laws =
  group "get_kernel_graph › laws"
    [
      prop "a scheduled function writes what its tensors compute" program
        (writes_what_it_computes ~cover);
      cases "a scheduled function writes what its tensors compute, for"
        ~id:(Format.asprintf "%a" pp_program)
        counterexamples writes_what_it_computes;
    ]

(* Loops of calls *)

(* A scan of three trips over parameters: a carry [c] of four floats, updated in
   place, and rows of four of [xs] and [ys]. Each trip's call stores [c * 2]
   into its row of [ys] and adds its row of [xs] to [c]. The call's body is
   already scheduled, as calls inside a program are scheduled before it. *)
let scan_loop () =
  let k = 4 and n = 3 in
  let p slot = Shape.param ~device:cpu ~shape:[ Int k ] slot Float32 in
  let body =
    Schedule.create_schedule
      (schedule
         (Ops.sink
            [
              Ops.store (p 0) Ops.O.(p 0 + p 1);
              Ops.store (p 2) Ops.O.(p 0 * float 2.);
            ]))
  in
  let c = Shape.param ~device:cpu ~shape:[ Int k ] 0 Float32
  and xs = Shape.param ~device:cpu ~shape:[ Int (n * k) ] 1 Float32
  and ys = Shape.param ~device:cpu ~shape:[ Int (n * k) ] 2 Float32 in
  let r = Ops.range ~axis_type:Loop (Int n) [ 100 ] in
  let row b =
    Shape.shrink b
      [ Some (Sym Ops.O.(r * int k), Sym Ops.O.((r * int k) + int k)) ]
  in
  let e =
    Ops.end_ (Ops.call ~precompile:true body [ c; row xs; row ys ]) [ r ]
  in
  (Ops.sink [ Ops.after c [ e ]; Ops.after ys [ e ] ], r)

let loops =
  group "get_kernel_graph › loops of calls"
    [
      test "a loop of a precompiled call is no kernel" (fun () ->
          let sink, _ = scan_loop () in
          equal int 0 (kernels_of sink));
      test "a loop's call reads views that move with its range" (fun () ->
          let sink, r = scan_loop () in
          let ends =
            List.filter
              (fun u -> Ops.op u = End)
              (Ops.toposort ~calls:Skip (schedule sink))
          in
          match ends with
          | [ e ] ->
              let call = Ops.nth e 0 in
              equal bool ~msg:"the end closes the range" true
                (List.memq r (List.tl (Ops.src e)));
              equal (list bool) ~msg:"which arguments move"
                [ false; true; true ]
                (List.map
                   (fun a -> Ops.Nodes.mem r (Ops.ranges a))
                   (Ops.src_without_body call))
          | ends -> failf "%d ends of calls" (List.length ends));
    ]

(* Buffer states

   A kernel reading a buffer in two states takes a parameter for each, which
   its call passes in the state it names. *)

let states =
  let flat slot = Shape.param ~device:cpu ~shape:[ Int 16 ] slot Float32 in
  let x = flat 1 and y = flat 2 in
  let assigned = Ops.after x [ Ops.store x y ] in
  let rows u = Shape.reshape u (ints [ 4; 4 ]) in
  let i32 n = Ops.const ~dtype:Int32 (`Int (Bigint.of_int n)) in
  let at =
    Ops.cast
      (Ops.maximum
         (Ops.minimum (Shape.param ~device:cpu ~shape:[ Int 4 ] 3 Int32) (i32 3))
         (i32 0))
      Weak_int
  in
  let reads_what_tensors_read sink () =
    let buffers = filled sink in
    let into_given = List.filter (fun (s, _, _) -> List.mem s (given sink)) in
    equal (list write)
      (into_given (Tensors.writes ~buffers sink))
      (into_given (Kernel_graphs.writes ~buffers (schedule sink)))
  in
  let one_state name sink =
    test name
      (reads_what_tensors_read (Ops.sink (Ops.src sink @ [ assigned ])))
  in
  group "get_kernel_graph › buffer states"
    [
      one_state
        "a value read beside a later state of its buffer is stored first"
        (stores Ops.O.(x + Ops.float ~dtype:Float32 1. + assigned));
      one_state "a buffer read beside its later state is copied first"
        (stores Ops.O.(x + assigned));
      one_state "rows gathered across a store are read before it"
        (stores
           Ops.O.(Ops.index (rows x) [ at ] + Ops.index (rows assigned) [ at ]));
    ]

(* Cost *)

(* [centred n] is [n] centrings in sequence: the row sums of a value are
   subtracted from it. A value read by its sums and by its difference is staged,
   and each row sum's stage is weighed for removal by asking whether its
   reduction reads a buffer, which the stage it reduces answers at once. Chains
   of other lengths read other inputs, and share no node. *)
let centred n =
  let rec link x k =
    if k = n then x
    else
      let sums = Shape.expand (Shape.rop x Add [ 1 ]) (ints [ 4; 4 ]) in
      link Ops.O.(x - sums) (k + 1)
  in
  stores (link (input n) 0)

(* [words f] is the words [f ()] allocates. *)
let words f =
  let before = Gc.minor_words () in
  ignore (Sys.opaque_identity (f ()));
  Gc.minor_words () -. before

(* [halvings d] is [d] levels over [16] rows of [2^d] values: each level
   multiplies the even and the odd elements of the one before, which it reads
   at two positions. Inlined into its reader, a level is two copies of the one
   before, so a kernel of all [d] levels would compute [2^d] products a
   row. *)
let halvings d =
  let rec level x m =
    if m = 1 then x
    else
      let pairs = Shape.reshape x (ints [ 16; m / 2; 2 ]) in
      let half i =
        Shape.reshape
          (Shape.shrink pairs [ None; None; Some (Ops.Int i, Ops.Int (i + 1)) ])
          (ints [ 16; m / 2 ])
      in
      level Ops.O.(half 0 * half 1) (m / 2)
  in
  let m = 1 lsl d in
  stores (Shape.reshape (level (input ~shape:[ 16; m ] 1) m) (ints [ 16 ]))

(* The nodes of the largest kernel of [sink]'s kernel graph. *)
let largest sink =
  List.fold_left
    (fun n c -> max n (List.length (Ops.toposort ~calls:Skip (Ops.nth c 0))))
    0
    (calls (schedule sink))

(* Work linear in the length is [a * n + b] with [b >= 0], so twice the length
   costs at most twice the work. The tenth of slack covers tables that double
   their capacity at different lengths in the two runs. *)
let cost =
  group "get_kernel_graph › cost"
    [
      test "the work on a chain of staged values is linear in its length"
        (fun () ->
          let work n =
            let sink = centred n in
            words (fun () -> schedule sink)
          in
          let short = work 100 and long = work 200 in
          less float_exact ~than:(2.2 *. short) long);
      (* Inlined, four more levels make the largest kernel sixteen times
         larger. Stored once its copies are many, a level starts a kernel
         again, and the depth only shifts where the kernels end: at most one
         level's doubling. *)
      test "levels each read twice make kernels that stop growing with depth"
        (fun () ->
          let shallow = largest (halvings 12) and deep = largest (halvings 16) in
          less int ~than:(4 * shallow) deep);
    ]

(* Copies

   Hand-built kernel graphs of stages that a later pass may inline back. A
   stage's value inlined is copied to each distinct index it is read at, and
   stays stored when the nodes its ranges reach times the copies beyond the
   first exceed [budget]. *)

let budget = 360

let removable : Ops.bufferize_opts =
  { device = Some cpu; addrspace = Global; keep = Removable }

let loop n id = Ops.range ~axis_type:Loop (Int n) [ id ]
let source = Shape.param ~device:cpu ~shape:[ Int 256 ] 1 Float32

(* [chain k x] is [k] steps of [y * y + x] from [x]: [2k] nodes over [x]'s. *)
let chain k x =
  let rec go y k = if k = 0 then y else go Ops.O.((y * y) + x) (k - 1) in
  go x k

(* The kernels of a graph that stores, at each [i] of [16], the sum of [reads
   i]. *)
let kernels_reading reads =
  let o = Shape.param ~device:cpu ~shape:[ Int 16 ] 0 Float32 in
  let i = loop 16 9 in
  let store = Ops.store (Ops.index o [ i ]) (sum_of (reads i)) in
  kernels
    (Rangeify.get_kernel_graph
       (Ops.sink [ Ops.after o [ Ops.end_ store [ i ] ] ]))

(* [staged v] is [v] of [source] at [s], staged over [s] of [size] (default
   [32]). [staged (chain k)] copies [2k + 2] nodes: the chain, [s] and the read
   of [source] at [s]. *)
let staged ?(size = Ops.Int 32) v =
  let s = Ops.range ~axis_type:Loop size [ 0 ] in
  Ops.bufferize ~opts:removable (v (Ops.index source [ s ])) [ s ]

(* The chain of [n] copied nodes, [n] even. *)
let chain_of n = chain ((n / 2) - 1)

(* The most copied nodes, and the fewest past them, that [c] copies of a value
   compute again within the budget: an even [n] with [n * (c - 1)] at most
   [budget], and the next even one. *)
let within c = budget / (c - 1) / 2 * 2
let past c = within c + 2

let at k i = [ Ops.O.(i + int k) ]
let once v i = [ Ops.index v (at 0 i) ]
let twice v i = [ Ops.index v (at 0 i); Ops.index v (at 16 i) ]

let copies =
  group "get_kernel_graph › copies"
    [
      test "a value of 1200 nodes read at one index is inlined" (fun () ->
          equal int 1 (kernels_reading (once (staged (chain 600)))));
      test "a value read twice whose second copy is the budget is inlined"
        (fun () ->
          equal int 1 (kernels_reading (twice (staged (chain_of (within 2))))));
      test "a value read twice whose second copy passes the budget is stored"
        (fun () ->
          equal int 2
            (kernels_reading (twice (staged (chain_of (past 2))))));
      (* The chain reads [source] at [0] alone: an index of the stage copies
         none of it. *)
      test "a value of 600 nodes outside its ranges read twice is inlined"
        (fun () ->
          let outside = chain 300 (Ops.index source [ Ops.O.int 0 ]) in
          let v = staged (fun x -> Ops.O.(x + outside)) in
          equal int 1 (kernels_reading (twice v)));
      (* [b] and [a] share the range [s]: the reads of [b] count at [s], not as
         reads of [a], which copies [a] once more and not 16 times. *)
      test "reads of a stage sharing a range are not the other's" (fun () ->
          let s = loop 48 0 and t = loop 2 1 in
          let a =
            Ops.bufferize ~opts:removable
              (chain 20 (Ops.index source [ Ops.O.((s * int 2) + t) ]))
              [ s; t ]
          in
          let b =
            let x = Ops.index source [ s ] in
            Ops.bufferize ~opts:removable Ops.O.(x * x) [ s ]
          in
          let reads i =
            Ops.index a [ i; Ops.O.int 0 ]
            :: Ops.index a [ i; Ops.O.int 1 ]
            :: List.init 15 (fun k -> Ops.index b (at k i))
          in
          equal int 1 (kernels_reading reads));
    ]

(* Recomputation

   A stage inlined into a reduction is computed at each trip of the reductions
   its read's indices run in, at most as often as each element is read, and
   stays stored when the nodes its ranges reach times the copies its reads make
   for one output beyond the first exceed [budget]. *)

(* The kernels of a graph that stores, at each [i] of [16], the sum over [k] of
   [width] of [read i k]. *)
let kernels_reducing ~width read =
  let o = Shape.param ~device:cpu ~shape:[ Int 16 ] 0 Float32 in
  let i = loop 16 9 and k = Ops.range (Int width) [ 10 ] in
  let store = Ops.store (Ops.index o [ i ]) (Ops.reduce (read i k) Add [ k ]) in
  kernels
    (Rangeify.get_kernel_graph
       (Ops.sink [ Ops.after o [ Ops.end_ store [ i ] ] ]))

let windows' read = kernels_reducing ~width:16 read

(* [windows v] reads [v] at [i + k] for [k] of [16], as a running sum's windows
   do: 256 trips over a stage of 32 elements, 8 an element. *)
let windows v = windows' (fun i k -> Ops.index v [ Ops.O.(i + k) ])

let recomputation =
  group "get_kernel_graph › recomputation"
    [
      (* [k] of [2] at [2i + k] reads each of the 32 elements once. *)
      test "a value of 1200 nodes reduced once an element is inlined" (fun () ->
          let read v i k = Ops.index v [ Ops.O.((i * int 2) + k) ] in
          equal int 1 (kernels_reducing ~width:2 (read (staged (chain 600)))));
      test "a cheap value reduced in windows is inlined" (fun () ->
          equal int 1 (windows (staged (chain 2))));
      test "a value reduced in windows whose work is under the budget is inlined"
        (fun () -> equal int 1 (windows (staged (chain_of (within 8)))));
      test "a value reduced in windows whose work passes the budget is stored"
        (fun () -> equal int 2 (windows (staged (chain_of (past 8)))));
      (* A value of its positions alone reads no storage. *)
      test "a value of no storage reduced in windows past the budget is stored"
        (fun () ->
          let s = Ops.range ~axis_type:Loop (Int 32) [ 0 ] in
          let v = chain_of (past 8) (Ops.cast s Float32) in
          equal int 2 (windows (Ops.bufferize ~opts:removable v [ s ])));
      (* Each window computes the copied nodes 8 times an element, and the two
         16 times. *)
      test "two windowed reads of a stage add up" (fun () ->
          let v = staged (chain_of (within 8)) in
          let read i k =
            Ops.O.(Ops.index v [ i + k ] + Ops.index v [ i + k + int 1 ])
          in
          equal int 2 (kernels_reducing ~width:16 read));
      (* At 32 elements the stage is read 8 times an element; at 31 it would
         be read more than 8 times, past the budget. *)
      test "a value of symbolic size counts at its largest" (fun () ->
          let n =
            Ops.variable "n" (`Int (Bigint.of_int 31)) (`Int (Bigint.of_int 32))
          in
          equal int 1 (windows (staged ~size:(Sym n) (chain_of (within 8)))));
      (* [b] of [4] is an axis the value does not vary along, which the cleanups
         drop before inlining: 32 elements, read 8 times each. *)
      test "a value reduced in windows counts no axis it does not vary along"
        (fun () ->
          let b = Ops.range ~axis_type:Loop (Int 4) [ 2 ] and s = loop 32 0 in
          let v =
            Ops.bufferize ~opts:removable
              (chain_of (past 8) (Ops.index source [ s ]))
              [ b; s ]
          in
          equal int 2
            (windows' (fun i k -> Ops.index v [ Ops.O.int 0; Ops.O.(i + k) ])));
      (* A window of 2 over a stage of 4 read at 16 outputs reads each element
         8 times, of which the window computes it again twice. *)
      test "a read counts at most the trips of its reductions" (fun () ->
          let v = staged ~size:(Int 4) (chain_of (past 8)) in
          let read i k = Ops.index v [ Ops.O.((i + k) % int 4) ] in
          equal int 1 (kernels_reducing ~width:2 read));
      (* Inlined, the value's loads are gated in the padding and its arithmetic
         runs at each of the 16 trips, 4 an element. *)
      test "a value reduced through padding counts the padding's trips"
        (fun () ->
          let v = staged ~size:(Int 4) (chain_of (past 4)) in
          let read _ k = Ops.index v [ Shape.valid k Ops.O.(k < int 4) ] in
          equal int 2 (kernels_reducing ~width:16 read));
      (* [a] is read at its reader [b]'s own index, and [b] in windows: [a] is
         inlined into [b] first, whose copies then count [a]'s nodes. *)
      test "a value inlined into a windowed stage counts in its copies"
        (fun () ->
          let a = staged (chain_of (past 8)) and s = loop 32 1 in
          let b =
            Ops.bufferize ~opts:removable
              Ops.O.(Ops.index a [ s ] + Ops.index source [ s ])
              [ s ]
          in
          equal int 2 (windows b));
      (* [i mod 4] and [i] valid below [4] read a stage of 4 elements at 16
         outputs, each computing it once. *)
      test "a value broadcast through a reshape is inlined" (fun () ->
          let v = staged ~size:(Int 4) (chain 200) in
          equal int 1
            (kernels_reading (fun i -> [ Ops.index v [ Ops.O.(i % int 4) ] ])));
      test "a value read through padding is inlined" (fun () ->
          let v = staged ~size:(Int 4) (chain 200) in
          let at i = Shape.valid i Ops.O.(i < int 4) in
          equal int 1 (kernels_reading (fun i -> [ Ops.index v [ at i ] ])));
    ]

(* The work bound

   A stage read once, at two indices, in windows or under a sum that selects
   one expert's matrix, stays stored exactly when the nodes its ranges reach
   times its copies beyond the first exceed [budget]. *)

(* The kernels of a graph that stores, at each output [r] of [experts * rows *
   2], the sum over [k] of 4 of [x] at [(r / 2, k)] times the sum over the
   experts [e] of [v] at [(e, r mod 2, k)] where [e] is the expert [ids] gives
   the row [r / 2]: a routed product, whose rows gather their expert's matrix
   by a sum that selects one trip. *)
let kernels_routed ~experts ~rows v =
  let n = experts * rows in
  let o = Shape.param ~device:cpu ~shape:[ Int (2 * n) ] 0 Float32 in
  let ids = Shape.param ~device:cpu ~shape:[ Int n ] 2 Int32 in
  let x = Shape.param ~device:cpu ~shape:[ Int (4 * n) ] 3 Float32 in
  let r = loop (2 * n) 9 and k = Ops.range (Int 4) [ 10 ] in
  let e = Ops.range (Int experts) [ 11 ] in
  let row = Ops.O.(r // int 2) in
  let selected = Ops.eq (Ops.index ids [ row ]) (Ops.cast e Dtype.Int32) in
  let read = Ops.index v [ e; Ops.O.(r % int 2); k ] in
  let gathered =
    Ops.reduce (Ops.where selected read (Ops.O.float 0.)) Add [ e ]
  in
  let product =
    Ops.reduce Ops.O.(gathered * Ops.index x [ (row * int 4) + k ]) Add [ k ]
  in
  let store = Ops.store (Ops.index o [ r ]) product in
  kernels
    (Rangeify.get_kernel_graph
       (Ops.sink [ Ops.after o [ Ops.end_ store [ r ] ] ]))

(* [routed_staged ~experts v] is [v] of [source] at [8e + 4j + k], staged over
   [e] of [experts], [j] of 2 and [k] of 4. [routed_staged ~experts (chain k)]
   copies [2k + 8] nodes. *)
let routed_staged ~experts v =
  let e = loop experts 0 and j = loop 2 1 and k = loop 4 2 in
  let at = Ops.O.((e * int 8) + (j * int 4) + k) in
  Ops.bufferize ~opts:removable (v (Ops.index source [ at ])) [ e; j; k ]

type reading =
  | Once
  | Twice
  | Windows of int  (** of [w] trips over 16 outputs: [w / 2] an element *)
  | Routed of int * int  (** experts and rows an expert *)

let pp_reading ppf = function
  | Once -> Format.fprintf ppf "once"
  | Twice -> Format.fprintf ppf "twice"
  | Windows w -> Format.fprintf ppf "windows of %d" w
  | Routed (e, r) -> Format.fprintf ppf "%d rows for each of %d experts" r e

(* The copies a reading makes of each element: a routed row computes its
   expert's matrix once, at most once a trip of [k]. *)
let repeats = function
  | Once -> 1
  | Twice -> 2
  | Windows w -> max 1 (w / 2)
  | Routed (_, rows) -> min 4 rows

(* The nodes of a staged chain of [k] steps that its ranges reach. *)
let copied reading k =
  match reading with Routed _ -> (2 * k) + 8 | _ -> (2 * k) + 2

let kernels_of_reading reading k =
  match reading with
  | Once -> kernels_reading (once (staged (chain k)))
  | Twice -> kernels_reading (twice (staged (chain k)))
  | Windows w ->
      kernels_reducing ~width:w (fun i j -> Ops.index (staged (chain k)) [ Ops.O.(i + j) ])
  | Routed (experts, rows) ->
      kernels_routed ~experts ~rows (routed_staged ~experts (chain k))

(* A reading, and chain lengths whose work falls on either side of the
   budget. *)
let reading_gen =
  Gen.(
    let* reading =
      one_of
        [
          constant Once;
          constant Twice;
          map (fun w -> Windows (1 lsl w)) (int_range 1 4);
          map
            (fun (e, r) -> Routed (e, r))
            (pair (of_list [ 2; 4; 8; 16 ]) (of_list [ 1; 2; 4 ]));
        ]
    in
    let c = repeats reading in
    let+ k = int_range 0 (if c = 1 then 64 else (budget / (c - 1)) + 8) in
    (reading, k))
  |> Gen.with_pp (fun ppf (r, k) -> Format.fprintf ppf "%a, chain %d" pp_reading r k)

let work_bound =
  group "get_kernel_graph › work bound"
    [
      prop "a stage is stored when its copies beyond the first pass the budget"
        reading_gen (fun (reading, k) ->
          let work = copied reading k * (repeats reading - 1) in
          let stored = work > budget in
          cover "read once" (reading = Once);
          cover "read twice" (reading = Twice);
          cover "read in windows" (match reading with Windows _ -> true | _ -> false);
          cover "routed, stored" ((match reading with Routed _ -> true | _ -> false) && stored);
          cover "routed, inlined" ((match reading with Routed (_, r) -> r > 1 | _ -> false) && not stored);
          cover "stored" stored;
          equal int (if stored then 2 else 1) (kernels_of_reading reading k));
    ]

(* Dropped digits

   A read whose index reads an output range only through floor divisions and
   remainders by constants drops the range's other digits: the outputs that
   differ only there read the value at one index, and a tile of them shares
   it, as a broadcast does. Its copies leave that reuse out. A window, a
   reduction range, a gather, a selection, a digit read twice in one component
   or also by another, and a constant that does not divide what it applies to
   stay counted. *)

let wide = Shape.param ~device:cpu ~shape:[ Int 4096 ] 1 Float32

(* [stage sizes v] is [v] of [wide] at the flat index of ranges of [sizes],
   staged over them. *)
let stage sizes v =
  let rs = List.mapi (fun i n -> loop n i) sizes in
  let at =
    List.fold_left2 (fun a r n -> Ops.O.((a * int n) + r)) (Ops.O.int 0) rs sizes
  in
  Ops.bufferize ~opts:removable (v (Ops.index wide [ at ])) rs

(* The kernels of a graph that stores, at each [r] of [rows] of 256 outputs,
   the sum over [k] of 16 of [read r k]. *)
let kernels_rows ~rows read =
  let o = Shape.param ~device:cpu ~shape:[ Int 256 ] 0 Float32 in
  let r = Ops.range ~axis_type:Loop rows [ 9 ] in
  let k = Ops.range (Int 16) [ 10 ] in
  let store = Ops.store (Ops.index o [ r ]) (Ops.reduce (read r k) Add [ k ]) in
  kernels
    (Rangeify.get_kernel_graph
       (Ops.sink [ Ops.after o [ Ops.end_ store [ r ] ] ]))

(* A stage of [sizes] read at [index r k] by [kernels_rows]: [copied] nodes,
   past the budget at the copies counted without the reuse a dropped digit
   shares, and [shared] if that reuse is all of them. *)
type digits = {
  name : string;
  rows : Ops.sint;
  sizes : int list;
  index : Ops.t -> Ops.t -> Ops.t list;
  copied : int;
  shared : bool;
}

let ids = Shape.param ~device:cpu ~shape:[ Int 4 ] 2 Int32
let gather r = Ops.cast (Ops.index ids [ Ops.O.(r // int 16) ]) Weak_int

(* The counts beside each row are the copies of each element without the
   shared reuse: the outputs and trips that read it. *)
let digit_cases =
  let open Ops.O in
  let n = Ops.variable "n" (`Int (Bigint.of_int 49)) (`Int (Bigint.of_int 64)) in
  [
    (* 16 *)
    { name = "[r / 16; k]"; rows = Int 64; sizes = [ 4; 16 ];
      index = (fun r k -> [ r // int 16; k ]); copied = past 16; shared = true };
    (* 16 *)
    { name = "[r mod 16; k]"; rows = Int 256; sizes = [ 16; 16 ];
      index = (fun r k -> [ r % int 16; k ]); copied = past 16; shared = true };
    (* 4 *)
    { name = "[r / 64; r mod 16; k]"; rows = Int 256; sizes = [ 4; 16; 16 ];
      index = (fun r k -> [ r // int 64; r % int 16; k ]); copied = past 4;
      shared = true };
    (* 16, the trips of [k] *)
    { name = "[(r / 16) mod 4; k]"; rows = Int 256; sizes = [ 4; 16 ];
      index = (fun r k -> [ (r // int 16) % int 4; k ]); copied = past 16;
      shared = true };
    (* 16 *)
    { name = "[r / 16 + k]"; rows = Int 64; sizes = [ 19 ];
      index = (fun r k -> [ (r // int 16) + k ]); copied = past 16;
      shared = false };
    (* 4 *)
    { name = "[r; k / 4]"; rows = Int 16; sizes = [ 16; 4 ];
      index = (fun r k -> [ r; k // int 4 ]); copied = past 4; shared = false };
    (* 16 *)
    { name = "[ids (r / 16); k]"; rows = Int 64; sizes = [ 4; 16 ];
      index = (fun r k -> [ gather r; k ]); copied = past 16; shared = false };
    (* 64 outputs over 28 rows: 2.3 *)
    { name = "[(r / 16) * 4 + r mod 16; k]"; rows = Int 64; sizes = [ 28; 16 ];
      index = (fun r k -> [ ((r // int 16) * int 4) + (r % int 16); k ]);
      copied = 300; shared = false };
    (* 64 outputs over 15 rows: 4.3 *)
    { name = "[r mod 8 + r / 8; k]"; rows = Int 64; sizes = [ 15; 16 ];
      index = (fun r k -> [ (r % int 8) + (r // int 8); k ]); copied = 120;
      shared = false };
    (* 16 outputs over 3 rows: 5.3 *)
    { name = "[r mod 3; k] of 16 rows"; rows = Int 16; sizes = [ 3; 16 ];
      index = (fun r k -> [ r % int 3; k ]); copied = 90; shared = false };
    (* 64 outputs over 124 elements, each read at 16 trips: 8.3 *)
    { name = "[r / 16; r mod 16 + k]"; rows = Int 64; sizes = [ 4; 31 ];
      index = (fun r k -> [ r // int 16; (r % int 16) + k ]); copied = 60;
      shared = false };
    (* 50 outputs over 4 rows: 12.5 *)
    { name = "[r / 16; k] of 50 rows"; rows = Int 50; sizes = [ 4; 16 ];
      index = (fun r k -> [ r // int 16; k ]); copied = 40; shared = false };
    (* 16, at the largest size *)
    { name = "[r / 16; k] of a symbolic size"; rows = Sym n; sizes = [ 4; 16 ];
      index = (fun r k -> [ r // int 16; k ]); copied = past 16; shared = false };
  ]

(* The kernels of a graph that stores, at each point of ranges of [rows], the
   sum over [k] of 16 of [v] read at [read] of those ranges and [k]. *)
let kernels_out ~rows read v =
  let o = Shape.param ~device:cpu ~shape:[ Int 4096 ] 0 Float32 in
  let rs = List.mapi (fun i n -> loop n (20 + i)) rows in
  let k = Ops.range (Int 16) [ 10 ] in
  let at =
    List.fold_left2 (fun a r n -> Ops.O.((a * int n) + r)) (Ops.O.int 0) rs rows
  in
  let value = Ops.reduce (Ops.index v (read rs k)) Add [ k ] in
  let store = Ops.store (Ops.index o [ at ]) value in
  kernels
    (Rangeify.get_kernel_graph
       (Ops.sink [ Ops.after o [ Ops.end_ store rs ] ]))

(* The digit of [d] a read drops from its outputs: of [4 * d] rows read at
   [r / d] or at [r mod d], or of [4 * d * 4] read at [r / (4 d)] and
   [r mod 4]. *)
type form = Low | High | Middle

let pp_form ppf f =
  Format.pp_print_string ppf
    (match f with Low -> "low" | High -> "high" | Middle -> "middle")

(* The outputs split at the digit, and the read of them that names no digit. *)
let split form d =
  match form with Low | High -> [ 4; d ] | Middle -> [ 4; d; 4 ]

let split_read form rs k =
  match (form, rs) with
  | Low, [ g; _ ] -> [ g; k ]
  | High, [ _; j ] -> [ j; k ]
  | Middle, [ g; _; l ] -> [ g; l; k ]
  | _ -> invalid_arg "split_read"

let merged_read form d r k =
  let wd = 4 * d in
  let open Ops.O in
  match form with
  | Low -> [ r // int d; k ]
  | High -> [ r % int d; k ]
  | Middle -> [ r // int wd; r % int 4; k ]

let digit_stage form d n =
  let sizes =
    match form with
    | Low -> [ 4; 16 ]
    | High -> [ d; 16 ]
    | Middle -> [ 4; 4; 16 ]
  in
  stage sizes (chain_of n)

(* The copies of each element counted without the digit's reuse: the trips
   of [k], 16, or the outputs that read it, fewer. *)
let digit_copies form d = match form with High -> 4 | Low | Middle -> min d 16

(* A form, its digit's size and a stage's copied nodes, an even number either
   side of the budget at the copies counted without the digit's reuse. *)
let digit_gen =
  Gen.(
    let* form = of_list [ Low; High; Middle ] in
    let* d = of_list [ 2; 3; 16; 64 ] in
    let+ n = int_range 1 (budget / (digit_copies form d - 1)) in
    (form, d, 2 * n))
  |> Gen.with_pp (fun ppf (f, d, n) ->
         Format.fprintf ppf "%a digit of %d, %d nodes" pp_form f d n)

let digits =
  group "get_kernel_graph › dropped digits"
    [
      cases "a dropped digit's reuse is no copy" ~id:(fun c -> c.name)
        digit_cases (fun c ->
          let v = stage c.sizes (chain_of c.copied) in
          equal int
            (if c.shared then 1 else 2)
            (kernels_rows ~rows:c.rows (fun r k -> Ops.index v (c.index r k))));
      prop "a read of merged outputs decides as one of them split" digit_gen
        (fun (form, d, n) ->
          let past = n * (digit_copies form d - 1) > budget in
          cover "a low digit dropped, past the budget" (form = Low && past);
          cover "a high digit dropped, past the budget" (form = High && past);
          cover "a middle digit dropped, past the budget" (form = Middle && past);
          cover "under the budget" (not past);
          let v = digit_stage form d n in
          let rows = split form d in
          equal int
            (kernels_out ~rows (fun rs k -> split_read form rs k) v)
            (kernels_out
               ~rows:[ List.fold_left ( * ) 1 rows ]
               (fun rs k -> merged_read form d (List.hd rs) k)
               v));
      (* [r mod 2] reads a digit of the rows, which the sum over the experts
         also runs in, selecting one: 4 copies. *)
      test "a digit a selection runs in stays counted" (fun () ->
          let v = routed_staged ~experts:2 (chain 60) in
          equal int 2 (kernels_routed ~experts:2 ~rows:4 v));
    ]

(* Selections

   A sum over one range that selects one trip, where an index equals the
   range, is computed at the selected index alone: a stage it reads counts the
   ranges that index runs in, as a read at that index would. A factor over
   another range inside the sum broadcasts the selected value along that
   range, as it would the gathered one. *)

type selecting = Selected | Gathered

(* The kernels of a graph that stores, at each output [r] of [experts * rows *
   2] and [m] of [width], the sum over [k] of 4 of [x] at [(r / 2, k)] times
   [v] at the row's expert [ids (r / 2)], [r mod 2] and [k], times [y] at [m]:
   the expert's matrix read by a sum over the experts that selects one and
   multiplies [y] inside it ([Selected]), or at the expert's index
   ([Gathered]). *)
let kernels_scaled ~experts ~rows ~width form v =
  let n = experts * rows in
  let o = Shape.param ~device:cpu ~shape:[ Int (2 * n * width) ] 0 Float32 in
  let ids = Shape.param ~device:cpu ~shape:[ Int n ] 2 Int32 in
  let x = Shape.param ~device:cpu ~shape:[ Int (4 * n) ] 3 Float32 in
  let y = Shape.param ~device:cpu ~shape:[ Int width ] 4 Float32 in
  let r = loop (2 * n) 9 and m = loop width 12 in
  let k = Ops.range (Int 4) [ 10 ] in
  let row = Ops.O.(r // int 2) in
  let scale = Ops.index y [ m ] in
  let term =
    match form with
    | Selected ->
        let e = Ops.range (Int experts) [ 11 ] in
        let selected =
          Ops.eq (Ops.index ids [ row ]) (Ops.cast e Dtype.Int32)
        in
        let read = Ops.index v [ e; Ops.O.(r % int 2); k ] in
        Ops.reduce
          (Ops.where selected Ops.O.(read * scale) (Ops.O.float 0.))
          Add [ e ]
    | Gathered ->
        let e = Ops.cast (Ops.index ids [ row ]) Weak_int in
        Ops.O.(Ops.index v [ e; r % int 2; k ] * scale)
  in
  let product =
    Ops.reduce Ops.O.(term * Ops.index x [ (row * int 4) + k ]) Add [ k ]
  in
  let at = Ops.O.((r * int width) + m) in
  let store = Ops.store (Ops.index o [ at ]) product in
  kernels
    (Rangeify.get_kernel_graph
       (Ops.sink [ Ops.after o [ Ops.end_ store [ r; m ] ] ]))

(* Experts, rows an expert, the width of the factor's range and a chain's
   steps, its copied nodes either side of the budget at a row's copies of its
   expert's matrix, [min 4 rows], and at those times the width. *)
let scaled_gen =
  Gen.(
    let* experts = of_list [ 2; 4; 8 ] in
    let* rows = of_list [ 1; 1; 2; 4 ] in
    let* width = of_list [ 1; 2; 4; 8 ] in
    let+ k = int_range 0 200 in
    (experts, rows, width, k))
  |> Gen.with_pp (fun ppf (e, r, w, k) ->
         Format.fprintf ppf "%d rows for each of %d experts, width %d, chain %d"
           r e w k)

(* Own ranges

   A stage's copies are counted at each axis of its ranges and the least taken,
   so reads of stages that share a range pool there. The ranges that indexing
   gives a stage that a later pass may remove are new to it, but an axis whose
   size is a range, which it indexes by that range: each such stage has a range
   of its own, where its reads are its own, and the least is no pool. *)

(* [drawn]'s program, its value stored again, doubled, into a second output,
   so that the two stores share their loops. *)
let two_outputs drawn =
  let sink, _ = build drawn in
  let store =
    List.find (fun u -> Ops.op u = Op.Store) (Ops.toposort ~calls:Skip sink)
  in
  let value = Ops.nth store 1 in
  let size = List.fold_left ( * ) 1 (dims value) in
  let twice = Shape.param ~device:cpu ~shape:[ Int size ] 9 Float32 in
  Ops.sink
    [
      Ops.nth sink 0;
      Ops.after twice
        [
          Ops.store
            (Shape.reshape twice (Shape.shape value))
            Ops.O.(value * float 2.);
        ];
    ]

let removable_stage u =
  match Ops.arg u with
  | Bufferize { keep = Whole; _ } -> false
  | Bufferize _ -> Ops.op u = Op.Stage
  | _ -> false

let own_ranges =
  group "get_kernel_graph › own ranges"
    [
      prop ~count:400 "a removable stage has a range no other stage has"
        program
        (fun drawn ->
          let indexed =
            Indexing.run_rangeify
              (Prepare.prepare_rangeify (two_outputs drawn))
          in
          let nodes = Ops.toposort ~calls:Skip indexed in
          let stages = List.filter (fun u -> Ops.op u = Op.Stage) nodes in
          let ranges_of u =
            List.filter (fun r -> Ops.op r = Op.Range) (List.tl (Ops.src u))
          in
          let live u =
            List.filter
              (fun r -> Ops.Nodes.mem r (Ops.ranges (Ops.nth u 0)))
              (ranges_of u)
          in
          let removable =
            List.filter (fun u -> removable_stage u && live u <> []) stages
          in
          let stores = List.filter (fun u -> Ops.op u = Op.Store) nodes in
          cover "two removable stages" (List.length removable >= 2);
          cover "stores sharing a loop"
            (List.exists
               (fun a ->
                 List.exists
                   (fun b ->
                     a != b
                     && Ops.Nodes.fold
                          (fun r found ->
                            found || Ops.Nodes.mem r (Ops.ranges b))
                          (Ops.ranges a) false)
                   stores)
               stores);
          let pooled u =
            let others = List.filter (fun v -> v != u) stages in
            let own r =
              not (List.exists (fun v -> List.memq r (ranges_of v)) others)
            in
            not (List.exists own (live u))
          in
          equal int ~msg:"removable stages with no range of their own" 0
            (List.length (List.filter pooled removable)));
    ]

let selections =
  group "get_kernel_graph › selections"
    [
      prop "a read a sum selects decides as one at the selected index"
        scaled_gen (fun (experts, rows, width, k) ->
          let work c = copied (Routed (experts, rows)) k * (c - 1) in
          let at_index = work (min 4 rows) > budget in
          let at_sum = work (min 4 (rows * width)) > budget in
          cover "a factor's range inside the sum" (width > 1);
          cover "past the budget at the sum's ranges, within at the index's"
            (at_sum && not at_index);
          cover "past the budget at the index's" at_index;
          let v = routed_staged ~experts (chain k) in
          let count form = kernels_scaled ~experts ~rows ~width form v in
          equal int (count Gathered) (count Selected));
    ]

(* Dead reads

   A staged value read at an index in padding is read nowhere. Here [s]'s two
   elements are summed into positions [0] and [1] of three, and only position
   [2] is kept: the sum there reads [s] at no index, beside a read of [s] at
   [0] that stages it. *)

let dead_read_sink () =
  let s = Ops.O.(input ~shape:[ 2 ] 1 * input ~shape:[ 2 ] 1) in
  let pad_end u n =
    let r = List.length (Shape.shape u) in
    let at i = if i = r - 1 then Some (Ops.Int 0, Ops.Int n) else None in
    Shape.pad u (List.init r at)
  in
  let x = Shape.reshape s (ints [ 2; 1; 1 ]) in
  let x = Shape.reshape (pad_end x 1) (ints [ 2; 2 ]) in
  let x = Shape.reshape (pad_end x 2) (ints [ 8 ]) in
  let x = Shape.reshape (pad_end x 1) (ints [ 3; 3 ]) in
  let x = Shape.rop x Add [ 0 ] in
  let x = Shape.shrink x [ Some (Ops.Int 2, Ops.Int 3) ] in
  let total = Ops.O.(x + Shape.shrink s [ Some (Ops.Int 0, Ops.Int 1) ]) in
  let o = Shape.param ~device:cpu ~shape:[ Int 1 ] 0 Float32 in
  Ops.sink
    [ Ops.after o [ Ops.store (Shape.reshape o (Shape.shape total)) total ] ]

let dead_reads =
  group "get_kernel_graph › dead reads"
    [
      test "a staged value read only in padding reads 0 there" (fun () ->
          writes (dead_read_sink ()));
    ]

(* Values that several kernels compute

   A value of no range that [k] kernels compute is stored once, in one
   element, when the nodes of its value that each kernel past the first
   computes again, [(k - 1) n], exceed [budget], [n] counting its computed
   nodes down to storage and to the values stored below it. Under the budget,
   each kernel computes it. A weak value is never stored, and below it, as
   below a load the kernels share, the value beneath is weighed. *)

(* The one element of slot [slot], of type [dt]. *)
let scalar ?(dt = Dtype.Float32) slot =
  Shape.reshape (Shape.param ~device:cpu ~shape:[ Int 1 ] slot dt) []

(* A sink whose outputs, of [sizes], each read the scalar of [reads] for it:
   output [j] stores [reads j] broadcast over its size plus an input. The
   outputs differ in size, so each is a kernel of its own. *)
let read_by sizes reads =
  let output j n =
    let o = Shape.param ~device:cpu ~shape:[ Int n ] (10 + j) Float32 in
    let x = Shape.param ~device:cpu ~shape:[ Int n ] (20 + j) Float32 in
    let v = Shape.expand (Shape.reshape (reads j) [ Int 1 ]) [ Int n ] in
    Ops.after o [ Ops.store o Ops.O.(v + x) ]
  in
  Ops.sink (List.mapi output sizes)

(* [steps k] is [k] steps of [y * y + x] from the scalar [x]: [2k] computed
   nodes. *)
let steps k = chain k (scalar 1)

(* The types of the storage a kernel graph makes. *)
let made_types sink =
  List.filter_map
    (fun n -> if Ops.op n = Alloc then Some (Ops.dtype n) else None)
    (Ops.toposort ~calls:Skip (schedule sink))

(* [t_(j+1) = t_j / 2 + 1], each [t_j] of [m] an output of its own: [t_j] is
   computed in the kernels of [t_j] to [t_m]. *)
let outputs_chain m =
  let rec go j t acc =
    if j > m then Ops.sink (List.rev acc)
    else
      let t = Ops.O.((t * float 0.5) + float 1.) in
      let o = Shape.param ~device:cpu ~shape:[ Int 1 ] (1 + j) Float32 in
      go (j + 1) t (Ops.after o [ Ops.store (Shape.reshape o []) t ] :: acc)
  in
  go 1 (scalar 0) []

let shared_scalars =
  group "get_kernel_graph › values that several kernels compute"
    [
      test "a value of no range read by two kernels within the budget is \
            computed in each" (fun () ->
          let sink = read_by [ 16; 8 ] (fun _ -> steps (budget / 2)) in
          equal int 2 (kernels_of sink));
      test "a value of no range read by two kernels past the budget is stored \
            once" (fun () ->
          let sink = read_by [ 16; 8 ] (fun _ -> steps ((budget / 2) + 1)) in
          equal int 3 (kernels_of sink);
          equal (list Dtypes.dtype) [ Float32 ] (made_types sink);
          writes sink);
      test "a value read by three kernels counts two copies past the first"
        (fun () ->
          let v k = read_by [ 16; 8; 4 ] (fun _ -> steps k) in
          equal int ~msg:"within" 3 (kernels_of (v (budget / 4)));
          equal int ~msg:"past" 4 (kernels_of (v ((budget / 4) + 1))));
      (* The index is a weak integer of the sum's low bits: the sum, an int32,
         is stored and each read converts it again. Tensors evaluates no index
         computed from storage, so the values are not compared. *)
      test "a weak value is never stored, and the value below it is" (fun () ->
          let x = scalar ~dt:Int32 1 in
          let t = chain ((budget / 2) + 1) x in
          let at =
            Ops.cast
              (Ops.bitwise_and t
                 (Ops.const ~dtype:Int32 (`Int (Bigint.of_int 3))))
              Weak_int
          in
          let table j =
            Shape.param ~device:cpu ~shape:[ Int 4 ] (30 + j) Float32
          in
          let sink = read_by [ 16; 8 ] (fun j -> Ops.index (table j) [ at ]) in
          equal int 3 (kernels_of sink);
          equal (list Dtypes.dtype) [ Int32 ] (made_types sink));
      (* One load of one table at the sum's low bits is shared by both
         readers: the load is no candidate, and the sum below it is. *)
      test "a value read through a load several kernels share is stored once"
        (fun () ->
          let t = chain ((budget / 2) + 1) (scalar ~dt:Int32 1) in
          let at =
            Ops.cast
              (Ops.bitwise_and t
                 (Ops.const ~dtype:Int32 (`Int (Bigint.of_int 3))))
              Weak_int
          in
          let table = Shape.param ~device:cpu ~shape:[ Int 4 ] 30 Float32 in
          let sink = read_by [ 16; 8 ] (fun _ -> Ops.index table [ at ]) in
          equal int 3 (kernels_of sink);
          equal (list Dtypes.dtype) [ Int32 ] (made_types sink));
      (* [l], two nodes over [h], is read by two kernels and [h] by a third:
         stored, [h] leaves [l] two nodes, which its two kernels compute
         again. *)
      test "a value cheap above the values stored below it is computed in each"
        (fun () ->
          let h = steps ((budget / 2) + 1) in
          let l = Ops.O.((h * float 2.) + float 1.) in
          let sink = read_by [ 16; 8; 4 ] (fun j -> if j < 2 then l else h) in
          equal int 4 (kernels_of sink);
          writes sink);
      (* Without the rule, the last output computes the whole chain and the
         kernels hold [m^2] nodes. Past the budget's first few hundred
         outputs, a kernel computes at most the budget again. *)
      test "the work on a chain of outputs is linear in its length" (fun () ->
          let work m =
            let sink = outputs_chain m in
            words (fun () -> schedule sink)
          in
          let short = work 1000 and long = work 2000 in
          less float_exact ~than:(2.2 *. short) long);
      test "a chain of outputs makes kernels that stop growing with its length"
        (fun () ->
          let shallow = largest (outputs_chain 1000)
          and deep = largest (outputs_chain 2000) in
          at_most int ~than:shallow deep);
    ]

let () =
  exit
    (run "Tolk.Rangeify"
       [
         recorded;
         counts;
         gather_counts;
         values;
         structure;
         laws;
         debug;
         spec;
         rules;
         shared;
         self_reads;
         siblings;
         loops;
         states;
         cost;
         copies;
         recomputation;
         work_bound;
         digits;
         selections;
         own_ranges;
         dead_reads;
         shared_scalars;
       ])
