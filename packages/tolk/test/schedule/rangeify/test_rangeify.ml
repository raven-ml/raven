(* Tests of Tolk.Rangeify: the kernel graphs tinygrad makes of the graphs it
   schedules, their kernel counts, the law that a kernel graph writes what its
   tensor graph does, and the rules of get_kernel_graph. *)

open Windtrap
open Tolk

let program name = Golden.sink (name ^ ".golden")

(* The settings a program was recorded under. *)
let settings = function
  | "many_inputs_limited" | "many_matrices_limited" | "many_cubes_limited"
  | "many_sums_limited" | "many_sharded_limited" ->
      [ Helpers.B (Helpers.max_kernel_buffers, 4) ]
  | _ -> []

let kernel_graph name =
  Helpers.context (settings name) (fun () ->
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
    "invalids_sharded";
    "layernorm";
    "long_cumsum";
    "many_cubes_limited";
    "many_inputs";
    "many_inputs_limited";
    "many_matrices_limited";
    "many_sharded_limited";
    "many_sums_limited";
    "matmul";
    "maxpool";
    "mean_kept_broadcast";
    "mesh_add";
    "mesh_add_1";
    "mesh_sum";
    "mesh_sum_1";
    "mesh_to_one";
    "mesh_to_one_1";
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
    "padded_twice";
    "partial_fuse";
    "partially_invalid";
    "permute_arange";
    "permute_through_reshape";
    "precompiled_function";
    "precompiled_function_1";
    "preserve_multistage_reduce";
    "push_pads_elementwise";
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
    "replicated_reshape";
    "reshape_chain";
    "setitem";
    "setitem_column";
    "setitem_cube";
    "setitem_tensor";
    "shard_add";
    "shard_gather";
    "shard_matmul";
    "shard_of_computed";
    "shard_sum";
    "shard_sum_1";
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
    "two_outputs";
    "ugly_reduceop_pairing";
    "unit_axis_staged";
    "variable_offset";
    "variable_read_twice";
    "variable_reduce";
    "variable_same";
    "variable_shrink";
    "variable_staged";
    "variable_two";
    "where";
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

let kernels u =
  List.length (List.filter is_kernel (Ops.toposort ~enter_calls:false u))

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

let devices = function Some (Ops.Multi l) -> List.length l | _ -> 1

let element dtype k : Dtype.value =
  if Dtype.is_float dtype then `Float (float_of_int k)
  else if Dtype.equal dtype Bool then `Bool (k > 0)
  else if Dtype.is_unsigned dtype then `Int (Bigint.of_int (k + 3))
  else `Int (Bigint.of_int k)

let storage u =
  List.filter_map
    (fun n ->
      match (Ops.op n, Ops.arg n) with
      | (Param | Buffer | Alloc), Param p when p.addrspace <> Some Alu ->
          Some (n, p)
      | _ -> None)
    (Ops.toposort u)

let filled u =
  List.map
    (fun (_, (p : Ops.param_arg)) ->
      let size = Option.value p.size ~default:1 in
      let at j = element p.dtype ((((j * 7) + (p.slot * 3)) mod 11) - 3) in
      (p.slot, Array.init (size * devices p.device) at))
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
  (if List.mem name heavy then slow else test)
    (name ^ " writes what its tensors write") (fun () ->
      let sink = program name in
      let buffers = filled sink in
      let into_given = List.filter (fun (s, _, _) -> List.mem s (given sink)) in
      equal (list write)
        (into_given (Tensors.writes ~buffers sink))
        (into_given (Kernel_graphs.writes ~buffers (kernel_graph name))))

(* Programs the law does not apply to: custom kernels, whose bodies keep their
   views until they are compiled, a compiled schedule, and symbolic shapes,
   which Tensors does not run; programs on several devices, which Kernel_graphs
   does not run, are left out as well. *)
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
  ]

let on_several_devices u =
  List.exists
    (fun n -> match Ops.device n with Some (Multi _) -> true | _ -> false)
    (Ops.toposort u)

let values =
  group "get_kernel_graph › values"
    (List.filter_map
       (fun name ->
         if List.mem name unevaluated || on_several_devices (program name) then
           None
         else Some (keeps_writes name))
       programs)

(* Kernel graphs

   What every kernel graph is, stated over the recorded ones: each kernel's
   storage parameters are numbered from [0] and match its call's arguments, its
   ranges are numbered from [0], the arguments are storage without views, the
   storage it makes is never of a weak type, and each is read after it is
   written. *)

let calls u = List.filter is_kernel (Ops.toposort ~enter_calls:false u)

let storage_params body =
  List.filter_map
    (fun n ->
      match (Ops.op n, Ops.arg n) with
      | Param, Param p when p.addrspace <> Some Alu -> Some (n, p)
      | _ -> None)
    (Ops.toposort body)

let rec storage_of u =
  match Ops.op u with
  | After -> storage_of (Ops.nth u 0)
  | Param | Buffer | Alloc | Mstack | Mselect -> Some u
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

(* A device range a kernel reads from its arguments keeps the id [-1] that marks
   it. *)
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
             (Ops.toposort (Ops.nth c 0)))
      in
      equal ~msg:"range ids" (list int) (List.init (List.length ids) Fun.id) ids)
    (calls u)

let arguments_are_storage u =
  List.iter
    (fun c ->
      List.iter
        (fun a ->
          match (Ops.op a, storage_of a) with
          | (Param | Buffer | Alloc | After | Mstack | Mselect), Some _ -> ()
          | _ when Ops.op a = Param -> ()
          | op, _ -> failf "a call's argument is a %a" Op.pp op)
        (List.tl (Ops.src c)))
    (calls u)

let made_storage u =
  List.filter (fun n -> Ops.op n = Alloc) (Ops.toposort ~enter_calls:false u)

let strong_storage u =
  List.iter
    (fun a -> is_false ~msg:"weak storage" (List.mem (Ops.dtype a) Dtype.weaks))
    (made_storage u)

(* [reads_after a arg] is [true] iff the argument [arg] is [a] after effects,
   directly or as one device's part of a gathered value. *)
let rec reads_after a arg =
  match Ops.op arg with
  | After -> Ops.nth arg 0 == a
  | Mstack | Mselect -> List.exists (reads_after a) (Ops.src arg)
  | _ -> false

(* Storage that no kernel writes, because every value stored into it was
   invalid, may still be read. *)
let read_after_written u =
  let written a =
    List.exists
      (fun n -> Ops.op n = After && Ops.nth n 0 == a)
      (Ops.toposort ~enter_calls:false u)
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
          Helpers.context
            [
              Helpers.B (Helpers.debug_rangeify, true);
              Helpers.B (Helpers.no_color, true);
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
          Helpers.context
            [ Helpers.B (Helpers.spec, 1) ]
            (fun () ->
              List.iter (fun name -> ignore (kernel_graph name)) programs));
    ]

(* Rules

   Hand-built functions on the CPU, prepared, then turned into kernel graphs: an
   output of sixteen elements in slot [0], inputs in the next slots. *)

let cpu = Ops.Single "CPU"
let ints = List.map (fun n -> Ops.Int n)

let input ?(shape = [ 4; 4 ]) slot =
  Ops.reshape
    (Ops.param ~device:cpu
       ~shape:[ Int (List.fold_left ( * ) 1 shape) ]
       slot Float32)
    (ints shape)

let out = Ops.param ~device:cpu ~shape:[ Int 16 ] 0 Float32

let stores value =
  Ops.sink
    [ Ops.after out [ Ops.store (Ops.reshape out (Ops.shape value)) value ] ]

let schedule sink = Rangeify.get_kernel_graph (Prepare.prepare_rangeify sink)
let kernels_of sink = kernels (schedule sink)
let rejects ~because f = raises_match (Exn.invalid_arg ~substring:because) f

let sum_of us =
  List.fold_left (fun acc u -> Ops.O.(acc + u)) (List.hd us) (List.tl us)

let read_twice v = Ops.O.(v + Ops.permute v [ 1; 0 ])

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
            Ops.expand (Ops.const ~dtype:Float32 `Invalid) (ints [ 16 ])
          in
          equal int 0
            (kernels_of (Ops.sink [ Ops.after out [ Ops.store out invalid ] ])));
      test "a kernel accesses at most max_kernel_buffers storages" (fun () ->
          let u =
            Helpers.context
              [ Helpers.B (Helpers.max_kernel_buffers, 4) ]
              (fun () -> kernel_graph "many_inputs_limited")
          in
          List.iter
            (fun c -> at_most int ~than:4 (List.length (List.tl (Ops.src c))))
            (calls u));
      test "without a limit, one kernel reads every storage" (fun () ->
          equal int 1 (kernels (kernel_graph "many_inputs")));
      test "a kernel reading one storage in two states is refused" (fun () ->
          let a = Ops.param ~device:cpu ~shape:[ Int 16 ] 1 Float32 in
          let written =
            Ops.after a
              [
                Ops.store a (Ops.param ~device:cpu ~shape:[ Int 16 ] 2 Float32);
              ]
          in
          let sink =
            Ops.sink [ Ops.after out [ Ops.store out Ops.O.(a + written) ] ]
          in
          rejects ~because:"cycle" (fun () -> Rangeify.get_kernel_graph sink));
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
                 match Ops.arg n with
                 | Param { size = Some s; _ } when Ops.op n = Alloc -> Some s
                 | _ -> None)
               (Ops.toposort (Rangeify.get_kernel_graph sink))));
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
        Rotate k;
        Flip k;
        Pad k;
        Shrink k;
        Unsqueeze k;
        Expand;
        Stage;
        Read_twice;
      ])

let program =
  Gen.(
    let* shape = list ~size:(int_range 1 3) (int_range 1 4) in
    let+ steps = list ~size:(int_range 1 5) step in
    (shape, steps))
  |> Gen.with_pp (fun ppf (shape, steps) ->
      Format.fprintf ppf "[%s], then %a"
        (String.concat "; " (List.map string_of_int shape))
        (Format.pp_print_list
           ~pp_sep:(fun ppf () -> Format.fprintf ppf ", ")
           pp_step)
        steps)

let dims u =
  List.map
    (function Ops.Int n -> n | Sym _ -> fail "a concrete shape")
    (Ops.shape u)

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
    | Sum k when rank > 0 -> Ops.rop v Add [ k mod rank ]
    | Max k when rank > 0 -> Ops.rop v Max [ k mod rank ]
    | Rotate k when rank > 0 ->
        Ops.permute v (List.init rank (fun i -> (i + k) mod rank))
    | Flip k -> (
        match pick k axes with Some a -> Ops.flip v [ a ] | None -> v)
    | Pad k -> (
        match pick k axes with
        | Some a -> Ops.pad v (on a (fun _ -> Some (Ops.Int 1, Ops.Int 2)))
        | None -> v)
    | Shrink k -> (
        match pick k (List.filter (fun a -> List.nth shape a > 1) axes) with
        | Some a -> Ops.shrink v (on a (fun n -> Some (Ops.Int 1, Ops.Int n)))
        | None -> v)
    | Unsqueeze k ->
        let p = k mod (rank + 1) in
        Ops.reshape v
          (ints
             (List.filteri (fun a _ -> a < p) shape
             @ (1 :: List.filteri (fun a _ -> a >= p) shape)))
    | Expand -> Ops.expand v (ints (2 :: shape))
    | Stage -> Ops.v Stage ~src:[ v ]
    | Read_twice when rank = 2 && List.nth shape 0 = List.nth shape 1 ->
        Ops.O.(v + Ops.permute v [ 1; 0 ])
    | _ -> v
  in
  let value = List.fold_left apply (fresh shape) steps in
  let size = List.fold_left ( * ) 1 (dims value) in
  let result = Ops.param ~device:cpu ~shape:[ Int size ] 0 Float32 in
  let sink =
    Ops.sink
      [
        Ops.after result
          [ Ops.store (Ops.reshape result (Ops.shape value)) value ];
      ]
  in
  (sink, !memory)

let schedules_what_it_computes drawn =
  let sink, buffers = build drawn in
  let kernels = schedule sink in
  let expected = Tensors.writes ~buffers sink in
  cover "several kernels" (List.length (calls kernels) > 1);
  cover "a value is written" (expected <> []);
  equal (list write) expected (Kernel_graphs.writes ~buffers kernels)

let laws =
  group "get_kernel_graph › laws"
    [
      prop "a scheduled function writes what its tensors compute" program
        schedules_what_it_computes;
    ]

(* Loops of calls *)

(* A scan of three trips over parameters: a carry [c] of four floats, updated in
   place, and rows of four of [xs] and [ys]. Each trip's call stores [c * 2]
   into its row of [ys] and adds its row of [xs] to [c]. The call's body is
   already scheduled, as calls inside a program are scheduled before it. *)
let scan_loop () =
  let k = 4 and n = 3 in
  let p slot = Ops.param ~device:cpu ~shape:[ Int k ] slot Float32 in
  let body =
    Schedule.create_schedule
      (schedule
         (Ops.sink
            [
              Ops.store (p 0) Ops.O.(p 0 + p 1);
              Ops.store (p 2) Ops.O.(p 0 * float 2.);
            ]))
  in
  let c = Ops.param ~device:cpu ~shape:[ Int k ] 0 Float32
  and xs = Ops.param ~device:cpu ~shape:[ Int (n * k) ] 1 Float32
  and ys = Ops.param ~device:cpu ~shape:[ Int (n * k) ] 2 Float32 in
  let r = Ops.range ~axis_type:Loop (Int n) [ 100 ] in
  let row b =
    Ops.shrink b
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
              (Ops.toposort ~enter_calls:false (schedule sink))
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

   A kernel reads each buffer in one state: a value read beside a later state of
   a buffer it reads is stored first. No program rune emits needs the rule yet,
   so each of these is expected to fail until it lands. *)

let states =
  let flat slot = Ops.param ~device:cpu ~shape:[ Int 16 ] slot Float32 in
  let x = flat 1 and y = flat 2 in
  let assigned = Ops.after x [ Ops.store x y ] in
  let rows u = Ops.reshape u (ints [ 4; 4 ]) in
  let i32 n = Ops.const ~dtype:Int32 (`Int (Bigint.of_int n)) in
  let at =
    Ops.cast
      (Ops.maximum
         (Ops.minimum (Ops.param ~device:cpu ~shape:[ Int 4 ] 3 Int32) (i32 3))
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
    xfail ~reason:"a kernel may read a buffer in two states"
      (test name (reads_what_tensors_read sink))
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
      let sums = Ops.expand (Ops.rop x Add [ 1 ]) (ints [ 4; 4 ]) in
      link Ops.O.(x - sums) (k + 1)
  in
  stores (link (input n) 0)

(* [words f] is the words [f ()] allocates. *)
let words f =
  let before = Gc.minor_words () in
  ignore (Sys.opaque_identity (f ()));
  Gc.minor_words () -. before

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
         loops;
         states;
         cost;
       ])
