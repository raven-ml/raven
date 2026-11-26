(* Tests of Tolk.Rangeify: the kernel graphs get_kernel_graph makes of the
   programs test/gen/schedule/rangeify.py records, the law that a kernel graph
   writes what its tensor graph writes, and get_kernel_graph's steps in the
   order its interface states them. *)

open Windtrap
open Tolk

(* Recorded programs *)

let recorded name = Golden.sink (name ^ ".golden")

(* The settings a program was recorded under. *)
let settings = function
  | "many_inputs_limited" | "many_matrices_limited" | "many_cubes_limited"
  | "many_sums_limited" ->
      [ Setting.B (Setting.max_kernel_buffers, 4) ]
  | _ -> []

let kernel_graph name =
  Setting.context (settings name) (fun () ->
      Rangeify.get_kernel_graph ~cores:Cores.host (recorded name))

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
    "assign_flipped_beside_call";
    "assign_flipped_beside_call_1";
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
    "selected_scaled";
    "blocked_product_16";
    "dense_decoded_64";
    "windowed_scaled";
    "windowed_scaled_long";
    "zero_size_children";
  ]
  @ gathers

let goldens =
  group
    "get_kernel_graph › makes each recorded program's kernel graph as its \
     golden states"
    (List.map
       (fun name ->
         Golden.graph (name ^ "_kernels.golden") (fun () -> kernel_graph name))
       programs)

(* Kernels and storage *)

let is_kernel u = Ops.op u = Call && Ops.op (Ops.nth u 0) = Sink
let calls u = List.filter is_kernel (Ops.toposort ~calls:Skip u)
let kernels u = List.length (calls u)

(* The sizes of the storage [graph] makes ({!Op.Alloc}), smallest first: the
   stages it keeps stored. *)
let new_storage graph =
  List.sort compare
    (List.filter_map
       (fun n -> if Ops.op n = Alloc then Some (Shape.max_numel n) else None)
       (Ops.toposort ~calls:Skip graph))

(* The types of the storage [graph] makes. *)
let new_types graph =
  List.filter_map
    (fun n -> if Ops.op n = Alloc then Some (Ops.dtype n) else None)
    (Ops.toposort ~calls:Skip graph)

let kernel_name c =
  match Ops.arg (Ops.nth c 0) with Kernel k -> Some k.name | _ -> None

(* Whether the call [c] of [name]'s kernel graph is of a kernel its program gave
   as code, compiled on its own: such a kernel keeps its views until it is
   compiled, and its call may pass a run of storage, which it takes where the
   run starts. *)
let given_as_code name c =
  match kernel_name c with
  | None -> false
  | Some k ->
      List.exists
        (fun n -> Ops.op n = Call && kernel_name n = Some k)
        (Ops.toposort ~calls:Skip (recorded name))

let storage_params body =
  List.filter_map
    (fun n ->
      match (Ops.op n, Ops.arg n) with
      | Param, Param p when p.addrspace <> Alu -> Some (n, p)
      | _ -> None)
    (Ops.toposort ~calls:Enter body)

(* What a kernel graph writes

   Running a kernel graph's kernels (Kernel_graphs) leaves in the storage the
   function is given, its parameters and buffers, what its tensor graph writes
   there (Tensors). Memory holds small integers. *)

let element dtype k : Dtype.value =
  if Dtype.is_float dtype then `Float (float_of_int k)
  else if Dtype.equal dtype Bool then `Bool (k > 0)
  else if Dtype.is_unsigned dtype then `Int (Bigint.of_int (k + 3))
  else `Int (Bigint.of_int k)

let storage u =
  List.filter_map
    (fun n ->
      match (Ops.op n, Ops.arg n) with
      | (Param | Buffer | Alloc), Param p when p.addrspace <> Alu -> Some (n, p)
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

(* The slots of the storage [u] is given: its parameters and buffers. *)
let given u =
  List.filter_map
    (fun (n, (p : Ops.param_arg)) ->
      if Ops.op n = Alloc then None else Some p.slot)
    (storage u)

let writes_given name =
  test name (fun () ->
      let sink = recorded name in
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
    "assign_flipped_beside_call_1";
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

let recorded_writes =
  List.filter_map
    (fun name ->
      if List.mem name unevaluated then None else Some (writes_given name))
    programs

(* Hand-built functions

   Functions on the CPU, prepared, then turned into kernel graphs: an output of
   sixteen elements in slot [0], inputs in the next slots. *)

let cpu = "CPU"
let ints = List.map (fun n -> Ops.Int n)

(* The storage of slot [slot]: sixteen floats. *)
let flat slot = Shape.param ~device:cpu ~shape:[ Int 16 ] slot Float32

let input ?(shape = [ 4; 4 ]) slot =
  Shape.reshape
    (Shape.param ~device:cpu
       ~shape:[ Int (List.fold_left ( * ) 1 shape) ]
       slot Float32)
    (ints shape)

let dims u =
  List.map
    (function Ops.Int n -> n | Sym _ -> fail "a concrete shape")
    (Shape.shape u)

(* An output of [slot] that stores [v] whole. *)
let output slot v =
  let size = List.fold_left ( * ) 1 (dims v) in
  let o = Shape.param ~device:cpu ~shape:[ Int size ] slot Float32 in
  Ops.after o [ Ops.store (Shape.reshape o (Shape.shape v)) v ]

let stores value = Ops.sink [ output 0 value ]

(* [cores] are what each device's kernels are taken to be: the host's by
   default, whose loops run a kernel's points in turn. *)
let schedule ?(cores = Cores.host) sink =
  Rangeify.get_kernel_graph ~cores (Prepare.prepare_rangeify sink)

let kernels_of ?cores sink = kernels (schedule ?cores sink)

(* The tests' storage on a device taken to run its kernels' points at once. *)
let at_once = Cores.at_once

let writes ?cores sink =
  let buffers = filled sink in
  equal (list write)
    (Tensors.writes ~buffers sink)
    (Kernel_graphs.writes ~buffers (schedule ?cores sink))

let sum_of us =
  List.fold_left (fun acc u -> Ops.O.(acc + u)) (List.hd us) (List.tl us)

let read_twice v = Ops.O.(v + Shape.permute v [ 1; 0 ])

(* [positions ~slot ~reads n] is [reads] positions loaded from the int32 storage
   of [slot] and clamped into [0, n), as a gather's indices are. *)
let positions ~slot ~reads n =
  let i32 k = Ops.const ~dtype:Int32 (`Int (Bigint.of_int k)) in
  let p = Shape.param ~device:cpu ~shape:[ Int reads ] slot Int32 in
  Ops.cast (Ops.maximum (Ops.minimum p (i32 (n - 1))) (i32 0)) Weak_int

(* Generated functions

   A function of up to three inputs of up to three axes, through a few
   operations, materialisations among them. A value along the way may also be
   returned, stored whole into an output of its own while later steps go on
   computing from it: where rangeify stores the value anyway, the output then
   holds it. *)

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
  | Return

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
  | Return -> Format.fprintf ppf "return"

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
        Return;
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

let pick k = function
  | [] -> None
  | l -> Some (List.nth l (k mod List.length l))

(* The first slot of the returned values' outputs, past the inputs'. *)
let returned_slot = 10

(* The function [shape, steps] draws, and its inputs' memory. *)
let build (shape, steps) =
  let memory = ref [] and returned = ref [] in
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
    | Return ->
        let slot = returned_slot + List.length !returned in
        returned := output slot v :: !returned;
        v
    | _ -> v
  in
  let value = List.fold_left apply (fresh shape) steps in
  (Ops.sink (output 0 value :: List.rev !returned), !memory)

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

(* Whether two kernels take a returned value's output: one writes it and another
   reads it, so the output holds a stage. *)
let output_read kernels =
  let rec base a = if Ops.op a = After then base (Ops.nth a 0) else a in
  let slots c =
    List.filter_map
      (fun a ->
        match Ops.arg (base a) with
        | Param p when p.slot >= returned_slot -> Some p.slot
        | _ -> None)
      (List.tl (Ops.src c))
  in
  let counts = Hashtbl.create 4 in
  List.iter
    (fun c ->
      List.iter
        (fun s ->
          Hashtbl.replace counts s
            (1 + Option.value ~default:0 (Hashtbl.find_opt counts s)))
        (List.sort_uniq compare (slots c)))
    (calls kernels);
  Hashtbl.fold (fun _ n acc -> acc || n > 1) counts false

(* Functions the law once failed on. A sum over a padded axis of size 1 folds
   into the value times its own validity, which reads the input at an invalid
   index where it is padding: that read is zero. A maximum of a broadcast sum is
   the sum, stored once for the broadcast and then read only by the store of the
   output, which copies it whole. Row sums returned and read broadcast are
   stored into their output, whose state the graph once named as it was before
   that store. *)
let counterexamples =
  [
    ([ 1 ], [ Pad 0; Unsqueeze 0; Pad 0; Sum 0 ]);
    ([ 2 ], [ Sum 0; Expand; Max 0 ]);
    ([ 4; 4 ], [ Sum 1; Return; Center 0 ]);
  ]

(* About one function in ten stores a stage into its output. *)
let generated_writes =
  prop ~count:1000 ~examples:counterexamples "for a generated function" program
    (fun drawn ->
      let sink, buffers = build drawn in
      let kernels = schedule sink in
      let expected = Tensors.writes ~buffers sink in
      cover "several kernels" (List.length (calls kernels) > 1);
      cover "an output another kernel reads" (output_read kernels);
      cover "reductions that share a loop" (shares_a_loop kernels);
      cover "a value is written" (expected <> []);
      equal (list write) expected (Kernel_graphs.writes ~buffers kernels))

(* Kernels *)

(* The movements in the bodies of the kernels [name]'s graph made. *)
let movements name =
  let graph = kernel_graph name in
  List.concat_map
    (fun c ->
      if given_as_code name c then []
      else
        List.filter
          (fun n -> Op.Set.mem (Ops.op n) Op.Set.movement)
          (Ops.toposort ~calls:Enter (Ops.nth c 0)))
    (calls graph)

let kernels_group =
  group "get_kernel_graph › kernels"
    [
      cases "are sinks carrying kernel information, in each recorded program"
        ~id:Fun.id programs (fun name ->
          let bare =
            List.filter
              (fun c ->
                match Ops.arg (Ops.nth c 0) with Kernel _ -> false | _ -> true)
              (calls (kernel_graph name))
          in
          equal (list Uops.uop) [] bare);
      cases "that the graph made hold no movements, in each recorded program"
        ~id:Fun.id programs (fun name ->
          equal (list Uops.uop) [] (movements name));
    ]

(* Products a core takes

   [Σ_i a_i · b_iᵀ] of three [m × k] by [n × k] matrices of [dt] on a CUDA
   device, each product summed in float32 and rounded to [dt], as rune lowers a
   contraction, the three added. Given sm_89's cores, bfloat16 products of 64
   rows each take a kernel of their own, the two stored as the bfloat16 their
   sum reads; products of 8 rows, which stream without a core, float32 products,
   which take one only under TF32, and a schedule given the host's cores keep
   one kernel of three sums. Where one factor is int8 codes decoded to float32,
   as a Q8_0 matrix's, the heuristic takes a core past 1 row where the factor
   that streams is decoded: the one of fewer free elements, or the first at a
   tie. 16 and 24 rows of bfloat16 by codes of 64 take a kernel each, as do
   codes of 24 by 24 rows and codes of 24 by codes of 64; 12 rows by codes of
   64, which the core pads past its bound, 24 stored rows, or 24 by codes of 24,
   keep one, as do codes of 17 by codes of 64, whose rows the core pads past its
   bound. *)

type factor = A | B

let cuda = "CUDA"

let sm_89_cores () =
  let target =
    {
      Helpers.Target.device = cuda;
      renderer = "CUDA";
      arch = "sm_89";
      interface = "";
      indices = "";
    }
  in
  [ (cuda, Heuristic.cores (Cstyle.without_compiler target)) ]

let sum_of_products ?(dt = Dtype.Bfloat16) ?(coded = []) m n k =
  let matrix factor slot rows =
    let dt = if List.mem factor coded then Dtype.Int8 else dt in
    Shape.reshape
      (Shape.param ~device:cuda ~shape:[ Int (rows * k) ] slot dt)
      (ints [ rows; k ])
  in
  let product i =
    let spread x shape = Shape.expand (Shape.reshape x (ints shape)) in
    let a = spread (matrix A (1 + i) m) [ m; 1; k ] (ints [ m; n; k ])
    and b = spread (matrix B (4 + i) n) [ 1; n; k ] (ints [ m; n; k ]) in
    let p = Ops.O.(Ops.cast a Float32 * Ops.cast b Float32) in
    Ops.cast (Shape.reshape (Shape.rop p Add [ 2 ]) (ints [ m; n ])) dt
  in
  let value = Ops.O.(product 0 + product 1 + product 2) in
  let out = Shape.param ~device:cuda ~shape:[ Int (m * n) ] 0 dt in
  Ops.sink
    [
      Ops.after out [ Ops.store (Shape.reshape out (Shape.shape value)) value ];
    ]

let on_sm_89 sink = kernels (schedule ~cores:(sm_89_cores ()) sink)

(* The types every kernel of [graph] stores. *)
let stored_types graph =
  List.sort_uniq Dtype.compare
    (List.filter_map
       (fun u ->
         if Ops.op u = Store then Some (Ops.dtype (Ops.nth u 1)) else None)
       (Ops.toposort ~calls:Enter graph))

let products_apart =
  group "get_kernel_graph › products a core takes"
    [
      test "bfloat16 products of 64 rows run a kernel each" (fun () ->
          equal int 3 (on_sm_89 (sum_of_products 64 64 64)));
      test "bfloat16 products of 64 rows are stored narrow" (fun () ->
          let g = schedule ~cores:(sm_89_cores ()) (sum_of_products 64 64 64) in
          equal (list Dtypes.dtype) [ Dtype.Bfloat16 ] (stored_types g));
      test "products of 8 rows, which take no core, share one kernel" (fun () ->
          equal int 1 (on_sm_89 (sum_of_products 8 64 64)));
      test "products of 16 rows by codes of 64 run a kernel each" (fun () ->
          equal int 3 (on_sm_89 (sum_of_products ~coded:[ B ] 16 64 64)));
      test "products of 12 rows by codes of 64 share one kernel" (fun () ->
          equal int 1 (on_sm_89 (sum_of_products ~coded:[ B ] 12 64 64)));
      test "products of 24 rows by codes of 64 run a kernel each" (fun () ->
          equal int 3 (on_sm_89 (sum_of_products ~coded:[ B ] 24 64 64)));
      test "products of codes of 64 by 24 rows run a kernel each" (fun () ->
          equal int 3 (on_sm_89 (sum_of_products ~coded:[ A ] 64 24 64)));
      test "stored products of 24 rows share one kernel" (fun () ->
          equal int 1 (on_sm_89 (sum_of_products 24 64 64)));
      test "products of 24 rows by codes of 24 share one kernel" (fun () ->
          equal int 1 (on_sm_89 (sum_of_products ~coded:[ B ] 24 24 64)));
      test "products of codes of 24 by 24 rows run a kernel each" (fun () ->
          equal int 3 (on_sm_89 (sum_of_products ~coded:[ A ] 24 24 64)));
      test "products of codes of 24 by codes of 64 run a kernel each" (fun () ->
          equal int 3 (on_sm_89 (sum_of_products ~coded:[ A; B ] 24 64 64)));
      test "products of codes of 17 by codes of 64 share one kernel" (fun () ->
          equal int 1 (on_sm_89 (sum_of_products ~coded:[ A; B ] 17 64 64)));
      test "float32 products share one kernel without TF32" (fun () ->
          equal int 1 (on_sm_89 (sum_of_products ~dt:Float32 64 64 64)));
      test "products share one kernel given the host's cores" (fun () ->
          equal int 1 (kernels_of (sum_of_products 64 64 64)));
    ]

(* Outputs that share a computation

   Stored values of one shape and device whose computations share a node reached
   through no view, or one view of a computation, and neither of which reads the
   other, run as one kernel that writes both. A double word's high and low parts
   are one chain's two ends. Two views of one computation read it at different
   indices: slices of it run as kernels of their own. *)

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

let two_sum_outputs () =
  let hi, lo = two_sum (input 1) (input 2) in
  stores2 hi lo

let product_of_inputs () = Ops.O.(input 1 * input 2)

let shared_reduction () =
  let m = Shape.rop (input 1) Max [ 1 ] in
  stores2 ~size:4 Ops.O.(m + float 1.) Ops.O.(m * float 2.)

let shared_broadcast_reduction () =
  let m = Shape.expand (Shape.rop (input 1) Max [ 1 ]) (ints [ 4; 4 ]) in
  stores2 Ops.O.(m + input 2) Ops.O.(m * input 3)

let two_reductions () =
  let v = product_of_inputs () in
  stores2 ~size:4 (Shape.rop v Add [ 1 ]) (Shape.rop v Max [ 1 ])

(* Rows [lo] and [lo + 1] of the product of inputs. *)
let rows_of_product lo =
  Shape.shrink (product_of_inputs ()) [ Some (Int lo, Int (lo + 2)); None ]

let two_slices () = stores2 ~size:8 (rows_of_product 0) (rows_of_product 2)

let two_slices_sharing () =
  let e = Ops.exp2 (input ~shape:[ 2; 4 ] 3) in
  stores2 ~size:8 Ops.O.(rows_of_product 0 + e) Ops.O.(rows_of_product 2 * e)

let one_view_twice () =
  let w = rows_of_product 1 in
  stores2 ~size:8 Ops.O.(w + float 1.) Ops.O.(w * float 2.)

let square u = Shape.reshape u (ints [ 4; 4 ])

let reads_what_another_writes () =
  let v = product_of_inputs () in
  let first = Ops.after (flat 0) [ Ops.store (square (flat 0)) v ] in
  Ops.sink
    [
      first;
      Ops.after (flat 4)
        [ Ops.store (square (flat 4)) Ops.O.((v * v) + square first) ];
    ]

let overwrites_what_another_reads () =
  let v = product_of_inputs () in
  stores2 Ops.O.(v + square (flat 4)) Ops.O.(v * v)

let read_around_overwrite () =
  let v = product_of_inputs () in
  let a = flat 5 in
  let written = Ops.after a [ Ops.store a Ops.O.(a * a) ] in
  let store o v = Ops.after o [ Ops.store (square o) v ] in
  Ops.sink
    [
      written;
      store (flat 0) Ops.O.(v + square a);
      store (flat 4) Ops.O.((v * v) + square written);
    ]

(* [a] and [b] share [p], [c] and [d] share [q], [b] reads [c] and [d] reads
   [a]: one kernel of each pair would run both before and after the other. *)
let crossed_pairs () =
  let store o v = Ops.after o [ Ops.store (square o) v ] in
  let p = Ops.O.(input 1 * input 2) and q = Ops.O.(input 1 - input 2) in
  let a = store (flat 0) p and c = store (flat 6) q in
  Ops.sink
    [
      a;
      store (flat 4) Ops.O.((p * p) + square c);
      c;
      store (flat 7) Ops.O.((q * q) + square a);
    ]

let shared =
  group "get_kernel_graph › outputs that share a computation"
    [
      test "a two-sum's sum and error run as one kernel" (fun () ->
          equal int 1 (kernels_of (two_sum_outputs ())));
      test "outputs that store their own storage back are no kernel" (fun () ->
          (* [flip u] flips [u]'s sign bit, through a mask the stores share:
             flipped twice, each stores its storage's own value. *)
          let a = flat 0 and b = flat 4 in
          let sign =
            Shape.const_like ~dtype:Uint32 a (`Int (Bigint.of_int 0x80000000))
          in
          let flip u =
            Ops.bitcast Ops.O.(Ops.bitcast u Uint32 lxor sign) Float32
          in
          let back o = Ops.after o [ Ops.store o (flip (flip o)) ] in
          equal int 0 (kernels_of (Ops.sink [ back a; back b ])));
      test "outputs that share only a load run as two kernels" (fun () ->
          let x = input 1 in
          equal int 2 (kernels_of (stores2 Ops.O.(x + x) Ops.O.(x * x))));
      test "outputs that share a reduction run as one kernel" (fun () ->
          equal int 1 (kernels_of (shared_reduction ())));
      test "outputs that share a broadcast reduction read it from its kernel"
        (fun () -> equal int 2 (kernels_of (shared_broadcast_reduction ())));
      test "outputs with a reduction each run as two kernels" (fun () ->
          equal int 2 (kernels_of (two_reductions ())));
      test "slices of one product through two views run as two kernels"
        (fun () -> equal int 2 (kernels_of (two_slices ())));
      test "slices that also share a computation run as two kernels" (fun () ->
          equal int 2 (kernels_of (two_slices_sharing ())));
      test "outputs that read a product through one view run as one kernel"
        (fun () -> equal int 1 (kernels_of (one_view_twice ())));
      test "outputs of shapes that differ run as two kernels" (fun () ->
          let v = product_of_inputs () in
          let w = Shape.reshape v (ints [ 2; 8 ]) in
          equal int 2 (kernels_of (stores2 v Ops.O.(w * w))));
      test "an output that reads what another writes runs after it" (fun () ->
          equal int 2 (kernels_of (reads_what_another_writes ())));
      test "an output that overwrites what another reads runs apart" (fun () ->
          equal int 2 (kernels_of (overwrites_what_another_reads ())));
      test "outputs reading a storage before and after its overwrite run apart"
        (fun () -> equal int 3 (kernels_of (read_around_overwrite ())));
      test "two pairs of outputs that each read the other pair run as three"
        (fun () -> equal int 3 (kernels_of (crossed_pairs ())));
    ]

(* Indexing's checks of stores, which get_kernel_graph runs through
   Indexing.run_rangeify. [n] exponentials of the product two outputs share, the
   second of which overwrites what the first reads, so that they run apart. *)
let store_checks =
  group "Indexing.store_checks"
    [
      cases
        "counts one check of outputs that run apart, however much they share"
        ~id:(Printf.sprintf "%d shared operations") [ 1; 65 ] (fun n ->
          let v =
            Seq.fold_left
              (fun u _ -> Ops.exp2 u)
              (product_of_inputs ())
              (Seq.init (n - 1) Fun.id)
          in
          let sink = stores2 Ops.O.(v + square (flat 4)) Ops.O.(v * v) in
          let before = Indexing.store_checks () in
          ignore (schedule sink);
          equal int 1 (Indexing.store_checks () - before));
    ]

(* Reductions that share a domain

   Reductions whose outputs a kernel indexes alike and that reduce axes of the
   same sizes run one loop when neither reads the other: the nodes their sources
   share are computed once. *)

let stores_small value =
  let small = Shape.param ~device:cpu ~shape:[ Int 4 ] 0 Float32 in
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

let sum_and_max () =
  let v = product_of_inputs () in
  stores_small Ops.O.(Shape.rop v Add [ 1 ] + Shape.rop v Max [ 1 ])

let centred_squares () =
  let x = input 1 in
  let sums = Shape.reshape (Shape.rop x Add [ 1 ]) (ints [ 4; 1 ]) in
  let centred = Ops.O.(x - Shape.expand sums (ints [ 4; 4 ])) in
  stores_small (Shape.rop Ops.O.(centred * centred) Add [ 1 ])

let sums_of_two_widths () =
  let w = input ~shape:[ 4; 8 ] 3 in
  stores_small
    Ops.O.(Shape.rop (product_of_inputs ()) Add [ 1 ] + Shape.rop w Add [ 1 ])

let siblings =
  group "get_kernel_graph › reductions that share a domain"
    [
      test "a sum and a maximum of one computation run one loop" (fun () ->
          equal (list int) [ 1 ] (reduce_loops (sum_and_max ())));
      test "a reduction that reads another of its domain runs its own loop"
        (fun () ->
          equal int 2
            (List.fold_left ( + ) 0 (reduce_loops (centred_squares ()))));
      test "reductions of axes of different sizes run a loop each" (fun () ->
          equal (list int) [ 2 ] (reduce_loops (sums_of_two_widths ())));
    ]

(* Own ranges, which Indexing.run_rangeify gives

   A stage's copies are counted at each axis of its ranges and the least taken,
   so reads of stages that share a range pool there. The ranges that indexing
   gives a stage that a later pass may remove are new to it, but an axis whose
   size is a range, which it indexes by that range: each such stage has a range
   of its own, where its reads are its own, and the least is no pool. *)

(* [drawn]'s function, its value stored again, doubled, into a second output, so
   that the two stores share their loops. *)
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
  group "Indexing.run_rangeify"
    [
      prop ~count:400 "gives a removable stage a range no other stage has"
        program (fun drawn ->
          let indexed =
            Indexing.run_rangeify ~cores:Cores.host
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
          equal (list Uops.uop) [] (List.filter pooled removable));
    ]

(* Storage that need not exist *)

let loop n id = Ops.range ~axis_type:Loop (Int n) [ id ]

let whole : Ops.bufferize_opts =
  { device = Some cpu; addrspace = Global; keep = Whole }

let removable : Ops.bufferize_opts = { whole with keep = Removable }

(* The storage of slot 1: 256 floats. *)
let source () = Shape.param ~device:cpu ~shape:[ Int 256 ] 1 Float32

(* A store into the output, at each [i] of [16], of [read i]. *)
let store_each read =
  let i = loop 16 1 in
  Ops.sink
    [
      Ops.after (flat 0)
        [ Ops.end_ (Ops.store (Ops.index (flat 0) [ i ]) (read i)) [ i ] ];
    ]

let needless =
  group "get_kernel_graph › removes storage that need not exist"
    [
      (* A materialisation, which no pass inlines, of a constant. *)
      test "a stage of a constant is the constant" (fun () ->
          let s =
            Ops.bufferize ~opts:whole
              (Ops.float ~dtype:Float32 2.)
              [ loop 16 0 ]
          in
          let sink = store_each (fun i -> Ops.index s [ i ]) in
          equal (list int) []
            (new_storage (Rangeify.get_kernel_graph ~cores:Cores.host sink)));
      (* A materialisation of a read of storage at its own ranges. *)
      test
        "a stage of a read of storage at the stage's own ranges is that storage"
        (fun () ->
          let s = loop 16 0 in
          let staged =
            Ops.bufferize ~opts:whole (Ops.index (source ()) [ s ]) [ s ]
          in
          let sink = store_each (fun i -> Ops.index staged [ i ]) in
          equal (list int) []
            (new_storage (Rangeify.get_kernel_graph ~cores:Cores.host sink)));
      test "a store of storage into itself runs no kernel" (fun () ->
          let out = flat 0 in
          equal int 0
            (kernels
               (Rangeify.get_kernel_graph ~cores:Cores.host
                  (Ops.sink [ Ops.after out [ Ops.store out out ] ]))));
    ]

(* The storages an inlined stage reads

   A value that reads more than three storages stays stored, unless each of its
   readers is stored as it is, as a store is, and its reads compute each of its
   elements at most once in all, none in a reduction: inlined, it then loads no
   storage more often than storing it does. A value read where it is broadcast
   stays stored. [four shape] sums four inputs of [shape]; [part v axis i] is
   element [i] of [v]'s [axis], that axis dropped. *)

let four shape = sum_of (List.map (input ~shape) [ 1; 2; 3; 4 ])

let part v axis i =
  let s = Shape.shape v in
  let bounds =
    List.mapi
      (fun a _ -> if a = axis then Some (Ops.Int i, Ops.Int (i + 1)) else None)
      s
  in
  Shape.reshape (Shape.shrink v bounds) (List.filteri (fun a _ -> a <> axis) s)

let four_read_in_halves () =
  let p = four [ 16; 2 ] in
  stores Ops.O.(part p 1 0 * part p 1 1)

(* The materialised product is read twice, so it is staged, and kept whole: it
   is a reader stored as it is. *)
let four_materialised_in_halves () =
  let p = four [ 4; 4; 2 ] in
  stores (read_twice (Ops.contiguous Ops.O.(part p 2 0 * part p 2 1)))

(* [p]'s halves are read by their sum and difference, each read twice and
   staged: inlined, [p] would make each read four storages. *)
let four_read_by_stages () =
  let p = four [ 4; 4; 2 ] in
  let h0 = part p 2 0 and h1 = part p 2 1 in
  stores Ops.O.(read_twice (h0 + h1) + read_twice (h0 - h1))

let four_reduced_in_halves () =
  let p = four [ 4; 16; 2 ] in
  stores (Shape.rop Ops.O.(part p 2 0 * part p 2 1) Add [ 0 ])

let four_broadcast () =
  let p = four [ 2; 4 ] in
  let row i =
    Shape.expand (Shape.reshape (part p 0 i) (ints [ 1; 4 ])) (ints [ 4; 4 ])
  in
  stores Ops.O.((input 5 * row 0) + row 1)

let storages =
  group "get_kernel_graph › the storages an inlined stage reads"
    [
      test "a value read twice and reading three storages is recomputed"
        (fun () ->
          let v = sum_of (List.map input [ 1; 2; 3 ]) in
          equal (list int) [] (new_storage (schedule (stores (read_twice v)))));
      test "a value read twice and reading four storages is stored" (fun () ->
          let v = sum_of (List.map input [ 1; 2; 3; 4 ]) in
          equal (list int) [ 16 ]
            (new_storage (schedule (stores (read_twice v)))));
      test "a value of four storages a store reads in halves is one kernel"
        (fun () -> equal int 1 (kernels_of (four_read_in_halves ())));
      test
        "a value of four storages a materialisation reads in halves is \
         computed there" (fun () ->
          equal (list int) [ 16 ]
            (new_storage (schedule (four_materialised_in_halves ()))));
      test "a value of four storages that stages read stays stored" (fun () ->
          equal (list int) [ 32 ]
            (new_storage (schedule (four_read_by_stages ()))));
      test "a value of four storages a reduction reads in halves stays stored"
        (fun () ->
          equal (list int) [ 128 ]
            (new_storage (schedule (four_reduced_in_halves ()))));
      test "a value of four storages read where it is broadcast stays stored"
        (fun () ->
          equal (list int) [ 8 ] (new_storage (schedule (four_broadcast ()))));
    ]

(* Broadcast exponentials

   A value read where it is broadcast and whose computing runs a transcendental
   function stays stored, whether the function is an operation or spelled in
   elementwise operations, as a compiler that restates a library's exponential
   emits it: inlined, each element of the ranges it does not vary along would
   compute it again. On a device whose kernels run their points at once it is
   computed where it is read when each point computes it once, and stays stored
   where a thread would compute an element again: in a reduction every output of
   which reads it, or through a stage that reduction may inline. [broadcast v]
   multiplies the vector [v] of 4 elements into each of the 4 rows of a matrix,
   and [vecmat v] by a matrix of 16 rows, each of whose outputs reads all of
   [v]. [weighed ~n v] multiplies [v] of [n] elements into each of the 4 rows of
   a stored matrix, and the [4 n] products into a product whose 16 outputs each
   read all of them, as attention's sinks weigh its output before its output
   projection: the weighed matrix reads two storages and every output reads it,
   so it stays stored whatever [v] costs. *)

let row v = Shape.reshape v (ints [ 1; 4 ])
let broadcast v = stores Ops.O.(input 5 * Shape.expand (row v) (ints [ 4; 4 ]))

let vecmat v =
  let m = input ~shape:[ 16; 4 ] 5 in
  stores (Shape.rop Ops.O.(m * Shape.expand (row v) (ints [ 16; 4 ])) Add [ 1 ])

let weighed ?(n = 4) v =
  let rows = Shape.expand (Shape.reshape v (ints [ 1; n ])) (ints [ 4; n ]) in
  let k = 4 * n in
  let w = Ops.O.(input ~shape:[ 4; n ] 5 * rows) in
  let w = Shape.expand (Shape.reshape w (ints [ 1; k ])) (ints [ 16; k ]) in
  stores (Shape.rop Ops.O.(w * input ~shape:[ 16; k ] 6) Add [ 1 ])

let spelled () = Transcendental.xexp2 (input ~shape:[ 4 ] 1)

(* [products n] is [n] products of a vector by itself: [n] operations. *)
let products n =
  let x = input ~shape:[ 4 ] 1 in
  let rec go v n = if n = 0 then v else go Ops.O.(v * x) (n - 1) in
  go x n

let cheap () =
  let x = input ~shape:[ 4 ] 1 in
  Ops.O.((x * x) + x)

(* Of two storages, 16 elements of 41 operations: past the budget, and under an
   exponential's work. *)
let two_storages_weighed () =
  let x = Ops.O.(input ~shape:[ 16 ] 1 + input ~shape:[ 16 ] 2) in
  let rec go v k = if k = 0 then v else go Ops.O.(v * x) (k - 1) in
  weighed ~n:16 (go x (Rangeify.transcendental_work () - 2))

let made ?cores sink = new_storage (schedule ?cores sink)

let broadcast_exponentials =
  group "get_kernel_graph › broadcast exponentials"
    [
      test "an exponential of elementwise operations broadcast stays stored"
        (fun () -> equal (list int) [ 4 ] (made (broadcast (spelled ()))));
      test
        "an exponential of elementwise operations every output reads stays \
         stored" (fun () -> equal (list int) [ 4 ] (made (vecmat (spelled ()))));
      test "a value of an exponential's work broadcast stays stored" (fun () ->
          let v = products (Rangeify.transcendental_work ()) in
          equal (list int) [ 4 ] (made (broadcast v)));
      test
        "a value one operation under an exponential's work broadcast is inlined"
        (fun () ->
          let v = products (Rangeify.transcendental_work () - 1) in
          equal (list int) [] (made (broadcast v)));
      test "a cheap value broadcast is inlined" (fun () ->
          equal (list int) [] (made (broadcast (cheap ()))));
      test "where points run at once, an exponential broadcast is inlined"
        (fun () ->
          equal (list int) [] (made ~cores:at_once (broadcast (spelled ()))));
      test
        "an exponential broadcast stays stored where only another device runs \
         points at once" (fun () ->
          let gpu = List.map (fun (_, c) -> ("GPU", c)) at_once in
          let cores = Cores.host @ gpu in
          equal (list int) [ 4 ] (made ~cores (broadcast (spelled ()))));
      test "an exponential broadcast stays stored on a device no cores name"
        (fun () ->
          equal (list int) [ 4 ] (made ~cores:[] (broadcast (spelled ()))));
      test
        "where points run at once, a value of an exponential's work every \
         output reads stays stored" (fun () ->
          let v = products (Rangeify.transcendental_work ()) in
          equal (list int) [ 4 ] (made ~cores:at_once (vecmat v)));
      test
        "an exponential a stored matrix reads where it broadcasts it stays \
         stored" (fun () ->
          equal (list int) [ 4; 16 ] (made (weighed (spelled ()))));
      test
        "where points run at once, an exponential a stored matrix reads where \
         it broadcasts it is computed there" (fun () ->
          equal (list int) [ 16 ] (made ~cores:at_once (weighed (spelled ()))));
      test
        "a value of two storages a stored matrix reads where it broadcasts it \
         is computed there" (fun () ->
          equal (list int) [ 64 ] (made (two_storages_weighed ())));
    ]

(* Gathers read where broadcast

   A value whose gathers are broadcast, and would compute more elements inlined
   than it holds, each element read once for each row of the broadcast, is read
   where it is broadcast, as a value broadcast directly is: one whose computing
   runs a transcendental function or that reads more than three storages stays
   stored. One whose gathers would compute fewer is inlined where the cost check
   allows. [gathered ~rows v] reads two elements of [v], at positions loaded
   from storage and clamped into range, and broadcasts them along [rows] rows of
   a sum. *)

let length v =
  match Shape.shape v with [ Ops.Int n ] -> n | _ -> fail "a vector"

let gathered ?(rows = 3) v =
  let at = positions ~slot:7 ~reads:2 (length v) in
  let g = Shape.reshape (Ops.index v [ at ]) (ints [ 1; 2 ]) in
  let o = Shape.param ~device:cpu ~shape:[ Int (rows * 2) ] 0 Float32 in
  let value =
    Ops.O.(Shape.expand g (ints [ rows; 2 ]) + input ~shape:[ rows; 2 ] 6)
  in
  Ops.sink
    [ Ops.after o [ Ops.store (Shape.reshape o (ints [ rows; 2 ])) value ] ]

let exp_of n = Ops.exp2 (input ~shape:[ n ] 1)

let gathered_broadcast =
  group "get_kernel_graph › gathers read where broadcast"
    [
      test "a gathered exponential read where it is broadcast stays stored"
        (fun () -> equal (list int) [ 2 ] (made (gathered (exp_of 2))));
      test
        "a gathered value of four storages read where it is broadcast stays \
         stored" (fun () ->
          equal (list int) [ 2 ] (made (gathered (four [ 2 ]))));
      test
        "an exponential of a larger table gathered and broadcast along few \
         rows is computed where it is read" (fun () ->
          equal (list int) [] (made (gathered (exp_of 16))));
      test
        "an exponential of a larger table gathered and broadcast along many \
         rows stays stored" (fun () ->
          equal (list int) [ 16 ] (made (gathered ~rows:64 (exp_of 16))));
      test
        "where points run at once, a gathered exponential read where it is \
         broadcast is computed where it is read" (fun () ->
          equal (list int) [] (made ~cores:at_once (gathered (exp_of 2))));
    ]

(* Gathers that read more than their table holds

   A value whose gathers, with no broadcast above them, read more elements than
   it holds computes some elements again inlined, as a broadcast value does: one
   whose computing runs a transcendental function or an exponential's work stays
   stored, and so does one of several storages that a reduction reads through
   its reduced ranges alone. [taken ~reads v] reads [reads] elements of [v] at
   positions loaded from storage and adds an input to each; [taken_twice ~reads
   v] adds two such gathers of [v]; [summed ~reads v] sums one into a scalar. *)

let store_reads ~reads value =
  let o = Shape.param ~device:cpu ~shape:[ Int reads ] 0 Float32 in
  Ops.sink [ Ops.after o [ Ops.store o value ] ]

let taken ~reads v =
  let g = Ops.index v [ positions ~slot:7 ~reads (length v) ] in
  store_reads ~reads Ops.O.(g + input ~shape:[ reads ] 6)

let taken_twice ~reads v =
  let g slot = Ops.index v [ positions ~slot ~reads (length v) ] in
  store_reads ~reads Ops.O.(g 7 + g 8)

let summed ~reads v =
  let g = Ops.index v [ positions ~slot:7 ~reads (length v) ] in
  store_reads ~reads:1 (Shape.reshape (Shape.rop g Add [ 0 ]) (ints [ 1 ]))

let two_inputs () = Ops.O.(input ~shape:[ 4 ] 1 + input ~shape:[ 4 ] 2)

let gathered_again =
  group "get_kernel_graph › gathers that read more than their table holds"
    [
      test
        "a gathered exponential read at more positions than it holds stays \
         stored" (fun () ->
          equal (list int) [ 2 ] (made (taken ~reads:16 (exp_of 2))));
      test
        "where points run at once, a gathered exponential read at more \
         positions than it holds stays stored" (fun () ->
          equal (list int) [ 2 ]
            (made ~cores:at_once (taken ~reads:16 (exp_of 2))));
      test
        "a gathered value of an exponential's work read at more positions than \
         it holds stays stored" (fun () ->
          let v = products (Rangeify.transcendental_work ()) in
          equal (list int) [ 4 ] (made (taken ~reads:16 v)));
      test
        "a gathered exponential read at one position more than it holds stays \
         stored" (fun () ->
          equal (list int) [ 4 ] (made (taken ~reads:5 (exp_of 4))));
      test
        "a gathered exponential read at as many positions as it holds is \
         computed where it is read" (fun () ->
          equal (list int) [] (made (taken ~reads:4 (exp_of 4))));
      test
        "a gathered exponential read at fewer positions than it holds is \
         computed where it is read" (fun () ->
          equal (list int) [] (made (taken ~reads:2 (exp_of 16))));
      test
        "an exponential two gathers read at more positions together than it \
         holds stays stored" (fun () ->
          equal (list int) [ 4 ] (made (taken_twice ~reads:3 (exp_of 4))));
      test
        "a cheap gathered value read at more positions than it holds is \
         computed where it is read" (fun () ->
          equal (list int) [] (made (taken ~reads:16 (cheap ()))));
      test
        "a cheap gathered value of two storages a sum reads at more positions \
         than it holds stays stored" (fun () ->
          equal (list int) [ 4 ] (made (summed ~reads:16 (two_inputs ()))));
      test
        "a cheap gathered value of one storage a sum reads at more positions \
         than it holds is computed where it is read" (fun () ->
          equal (list int) [] (made (summed ~reads:16 (cheap ()))));
    ]

(* Values a reduction reads again

   A value of more than one storage that a reduction reads along a range it
   sums, at outputs its read does not count, as a product's output columns read
   the rows of its left operand, stays stored when its elements times the work
   of one pass the budget and it holds no more bytes than it reads: inlined,
   each output would compute it again. [scaled rows k] is a [rows] by [k] matrix
   times its rows' sums of squares, as a normalisation's map reads its input and
   each row's scale, which stays stored as a reduction; [gained rows k] also
   multiplies it by a third storage. [product rows k n v] stores [v] times a [k]
   by [n] matrix. *)

let scaled rows k =
  let x = input ~shape:[ rows; k ] 1 in
  let ss = Shape.rop Ops.O.(x * x) Add [ 1 ] in
  let ss = Shape.reshape ss (ints [ rows; 1 ]) in
  Ops.O.(x * Shape.expand ss (ints [ rows; k ]))

let gained rows k =
  let g = Shape.expand (input ~shape:[ 1; k ] 4) (ints [ rows; k ]) in
  Ops.O.(scaled rows k * g)

(* [outer rows k] is a column of [rows] elements times a row of [k]: two
   storages of [rows + k] elements. *)
let outer rows k =
  let column = Shape.expand (input ~shape:[ rows; 1 ] 1) (ints [ rows; k ]) in
  Ops.O.(column * Shape.expand (input ~shape:[ 1; k ] 4) (ints [ rows; k ]))

let times_matrix rows k n v =
  let w = Shape.expand (input ~shape:[ 1; k; n ] 2) (ints [ rows; k; n ]) in
  let vs =
    Shape.expand (Shape.reshape v (ints [ rows; k; 1 ])) (ints [ rows; k; n ])
  in
  output 0 (Shape.rop Ops.O.(vs * w) Add [ 1 ])

let product rows k n v = Ops.sink [ times_matrix rows k n v ]

(* [product] and the greatest of each of [v]'s columns: a second reader at other
   indices, which makes [v] a stage of its own. *)
let product_and_max rows k n v =
  let m = Shape.rop (Shape.permute v [ 1; 0 ]) Max [ 1 ] in
  Ops.sink [ times_matrix rows k n v; output 5 m ]

(* The rows of the [rows] by [k] matrix [v] at [g] positions loaded from storage
   and clamped into range, as a mixture of experts gathers the rows each expert
   takes. *)
let gathered_rows ~rows ~k g v =
  let id = positions ~slot:3 ~reads:g rows in
  let row = Shape.expand (Shape.reshape id (ints [ g; 1 ])) (ints [ g; k ]) in
  let col =
    Shape.expand
      (Shape.reshape (Shape.arange ~dtype:Weak_int k) (ints [ 1; k ]))
      (ints [ g; k ])
  in
  let at = Shape.reshape Ops.O.((row * int k) + col) (ints [ g * k ]) in
  let flat = Shape.reshape v (ints [ rows * k ]) in
  Shape.reshape (Ops.index flat [ at ]) (ints [ g; k ])

let reread =
  group "get_kernel_graph › values a reduction reads again"
    [
      cases "a scaled matrix a product's columns read stays stored"
        ~id:(Printf.sprintf "%d rows") [ 12; 64; 512 ] (fun rows ->
          equal (list int)
            [ rows; rows * 2880 ]
            (made (product rows 2880 5120 (scaled rows 2880))));
      (* The map's work is one product an element: computing its [2 k] elements
         again is within the budget up to [k = 180]. *)
      test "a scaled matrix whose work is the budget is inlined" (fun () ->
          equal (list int) [ 2 ] (made (product 2 180 16 (scaled 2 180))));
      test "a scaled matrix of work past the budget stays stored" (fun () ->
          equal (list int) [ 2; 362 ] (made (product 2 181 16 (scaled 2 181))));
      (* The outer product of 2 and 181 elements does 362 multiplications, past
         the budget, as the scaled matrix's work does. It holds 362 elements and
         reads 183. *)
      test "a matrix holding more bytes than it reads is inlined" (fun () ->
          equal (list int) [] (made (product 2 181 16 (outer 2 181))));
      (* Inlined into the matrix the product's columns read, the exponential
         would be computed at each of the matrix's 16 elements, and the matrix
         stored in its place. *)
      test
        "where points run at once, an exponential a matrix broadcasts to a \
         product stays stored" (fun () ->
          let rows =
            Shape.expand
              (Shape.reshape (spelled ()) (ints [ 1; 4 ]))
              (ints [ 4; 4 ])
          in
          equal (list int) [ 4 ]
            (made ~cores:at_once (product 4 4 8 Ops.O.(input 5 * rows))));
      (* Inlined into the gathered rows, the scaled matrix would bring them its
         three storages and the positions' to read again, and the buffer limit
         would store the rows, four times the matrix. *)
      test "a scaled matrix whose gathered rows a product reads stays stored"
        (fun () ->
          let rows = 512 and k = 2880 and g = 2048 in
          let v = gathered_rows ~rows ~k g (gained rows k) in
          equal (list int) [ rows; rows * k ] (made (product_and_max g k 64 v)));
    ]

(* Copies of a stage a store reads

   Hand-built kernel graphs of stages that a later pass may inline back. A
   stage's value inlined is copied to each distinct index it is read at, and
   stays stored when the nodes its ranges reach times the copies beyond the
   first exceed [budget]. *)

let budget = 360

(* [chain k x] is [k] steps of [y * y + x] from [x]: [2k] nodes over [x]'s. *)
let chain k x =
  let rec go y k = if k = 0 then y else go Ops.O.((y * y) + x) (k - 1) in
  go x k

(* The kernel graph of a graph that stores, at each [i] of [16], the sum of
   [reads i]. *)
let sum_reads reads =
  let o = Shape.param ~device:cpu ~shape:[ Int 16 ] 0 Float32 in
  let i = loop 16 9 in
  let store = Ops.store (Ops.index o [ i ]) (sum_of (reads i)) in
  Rangeify.get_kernel_graph ~cores:Cores.host
    (Ops.sink [ Ops.after o [ Ops.end_ store [ i ] ] ])

(* [staged v] is [v] of [source] at [s], staged over [s] of [size] (default
   [32]). [staged (chain k)] copies [2k + 2] nodes: the chain, [s] and the read
   of [source] at [s]. *)
let staged ?(size = Ops.Int 32) v =
  let s = Ops.range ~axis_type:Loop size [ 0 ] in
  Ops.bufferize ~opts:removable (v (Ops.index (source ()) [ s ])) [ s ]

(* The chain of [n] copied nodes, [n] even. *)
let chain_of n = chain ((n / 2) - 1)

(* The most copied nodes, and the fewest past them, that [c] copies of a value
   compute again within the budget: an even [n] with [n * (c - 1)] at most
   [budget], and the next even one. *)
let within c = budget / (c - 1) / 2 * 2
let past c = within c + 2
let at k i = [ Ops.O.(i + int k) ]

(* The position at [s] of [positions], of [n] (default [256]) elements, loaded
   from storage. *)
let loaded_at ?(n = 256) s =
  let positions = Shape.param ~device:cpu ~shape:[ Int n ] 2 Int32 in
  Ops.cast (Ops.index positions [ s ]) Weak_int

let once v i = [ Ops.index v (at 0 i) ]
let twice v i = [ Ops.index v (at 0 i); Ops.index v (at 16 i) ]

(* [halvings d] is [d] levels over [16] rows of [2^d] values: each level
   multiplies the even and the odd elements of the one before, which it reads at
   two positions. Inlined into its reader, a level is two copies of the one
   before, so a kernel of all [d] levels would compute [2^d] products a row. *)
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

let copies =
  group "get_kernel_graph › copies of a stage a store reads"
    [
      test "a value of 1200 nodes read at one index is inlined" (fun () ->
          equal (list int) []
            (new_storage (sum_reads (once (staged (chain 600))))));
      (* The chain reads [source] at [0] alone: an index of the stage copies
         none of it. *)
      test "a value of 600 nodes outside its ranges read twice is inlined"
        (fun () ->
          let outside = chain 300 (Ops.index (source ()) [ Ops.O.int 0 ]) in
          let v = staged (fun x -> Ops.O.(x + outside)) in
          equal (list int) [] (new_storage (sum_reads (twice v))));
      (* [b] and [a] share the range [s]: the reads of [b] count at [s], not as
         reads of [a], which copies [a] once more and not 16 times. *)
      test "reads of a stage sharing a range are not the other's" (fun () ->
          let s = loop 48 0 and t = loop 2 1 in
          let a =
            Ops.bufferize ~opts:removable
              (chain 20 (Ops.index (source ()) [ Ops.O.((s * int 2) + t) ]))
              [ s; t ]
          in
          let b =
            let x = Ops.index (source ()) [ s ] in
            Ops.bufferize ~opts:removable Ops.O.(x * x) [ s ]
          in
          let reads i =
            Ops.index a [ i; Ops.O.int 0 ]
            :: Ops.index a [ i; Ops.O.int 1 ]
            :: List.init 15 (fun k -> Ops.index b (at k i))
          in
          equal (list int) [] (new_storage (sum_reads reads)));
      (* Inlined, four more levels make the largest kernel sixteen times larger.
         Stored once its copies are many, a level starts a kernel again, and the
         depth only shifts where the kernels end: at most one level's
         doubling. *)
      test "levels each read twice make kernels that stop growing with depth"
        (fun () ->
          let shallow = largest (halvings 12)
          and deep = largest (halvings 16) in
          less int ~than:(4 * shallow) deep);
    ]

(* Copies of a stage a reduction reads

   A stage inlined into a reduction is computed at each trip of the reductions
   its read's indices run in, at most as often as each element is read, and
   stays stored when the nodes its ranges reach times the copies its reads make
   for one output beyond the first exceed [budget]. *)

(* The kernel graph of a graph that stores, at each [i] of [16], the sum over
   [k] of [width] of [read i k]. *)
let reducing ~width read =
  let o = Shape.param ~device:cpu ~shape:[ Int 16 ] 0 Float32 in
  let i = loop 16 9 and k = Ops.range ~axis_type:Weak (Int width) [ 10 ] in
  let store = Ops.store (Ops.index o [ i ]) (Ops.reduce (read i k) Add [ k ]) in
  Rangeify.get_kernel_graph ~cores:Cores.host
    (Ops.sink [ Ops.after o [ Ops.end_ store [ i ] ] ])

(* [windows v] reads [v] at [i + k] for [k] of [16], as a running sum's windows
   do: 256 trips over a stage of 32 elements, 8 an element. *)
let windows v = reducing ~width:16 (fun i k -> Ops.index v [ Ops.O.(i + k) ])

(* [read_200 v] reads [v] at 200 distinct indices. *)
let read_200 v =
  sum_reads (fun i -> List.init 200 (fun k -> Ops.index v (at k i)))

let recomputation =
  group "get_kernel_graph › copies of a stage a reduction reads"
    [
      (* [k] of [2] at [2i + k] reads each of the 32 elements once. *)
      test "a value of 1200 nodes reduced once an element is inlined" (fun () ->
          let read v i k = Ops.index v [ Ops.O.((i * int 2) + k) ] in
          equal (list int) []
            (new_storage (reducing ~width:2 (read (staged (chain 600))))));
      test "a cheap value reduced in windows is inlined" (fun () ->
          equal (list int) [] (new_storage (windows (staged (chain 2)))));
      (* A value of its positions alone reads no storage. *)
      test "a value of no storage reduced in windows past the budget is stored"
        (fun () ->
          let s = Ops.range ~axis_type:Loop (Int 32) [ 0 ] in
          let v = chain_of (past 8) (Ops.cast s Float32) in
          equal (list int) [ 32 ]
            (new_storage (windows (Ops.bufferize ~opts:removable v [ s ]))));
      (* Each window computes the copied nodes 8 times an element, and the two
         16 times. *)
      test "two windowed reads of a stage add up" (fun () ->
          let v = staged (chain_of (within 8)) in
          let read i k =
            Ops.O.(Ops.index v [ i + k ] + Ops.index v [ i + k + int 1 ])
          in
          equal (list int) [ 32 ] (new_storage (reducing ~width:16 read)));
      (* At 32 elements the stage is read 8 times an element; at 31 it would be
         read more than 8 times, past the budget. *)
      test "a value of symbolic size counts at its largest" (fun () ->
          let n =
            Ops.variable "n" (`Int (Bigint.of_int 31)) (`Int (Bigint.of_int 32))
          in
          let v = staged ~size:(Sym n) (chain_of (within 8)) in
          equal (list int) [] (new_storage (windows v)));
      (* [b] of [4] is an axis the value does not vary along, which the cleanups
         drop before inlining: 32 elements, read 8 times each. *)
      test "a value reduced in windows counts no axis it does not vary along"
        (fun () ->
          let b = Ops.range ~axis_type:Loop (Int 4) [ 2 ] and s = loop 32 0 in
          let v =
            Ops.bufferize ~opts:removable
              (chain_of (past 8) (Ops.index (source ()) [ s ]))
              [ b; s ]
          in
          let read i k = Ops.index v [ Ops.O.int 0; Ops.O.(i + k) ] in
          equal (list int) [ 32 ] (new_storage (reducing ~width:16 read)));
      (* A window of 2 over a stage of 4 read at 16 outputs reads each element 8
         times, of which the window computes it again twice. *)
      test "a read counts at most the trips of its reductions" (fun () ->
          let v = staged ~size:(Int 4) (chain_of (past 8)) in
          let read i k = Ops.index v [ Ops.O.((i + k) % int 4) ] in
          equal (list int) [] (new_storage (reducing ~width:2 read)));
      (* Inlined, the value's loads are gated in the padding and its arithmetic
         runs at each of the 16 trips, 4 an element. *)
      test "a value reduced through padding counts the padding's trips"
        (fun () ->
          let v = staged ~size:(Int 4) (chain_of (past 4)) in
          let read _ k = Ops.index v [ Shape.valid k Ops.O.(k < int 4) ] in
          equal (list int) [ 4 ] (new_storage (reducing ~width:16 read)));
      (* [a] is read at its reader [b]'s own index, and [b] in windows: [a] is
         inlined into [b] first, whose copies then count [a]'s nodes. *)
      test "a value inlined into a windowed stage counts in its copies"
        (fun () ->
          let a = staged (chain_of (past 8)) and s = loop 32 1 in
          let b =
            Ops.bufferize ~opts:removable
              Ops.O.(Ops.index a [ s ] + Ops.index (source ()) [ s ])
              [ s ]
          in
          equal (list int) [ 32 ] (new_storage (windows b)));
      (* [i mod 4] and [i] valid below [4] read a stage of 4 elements at 16
         outputs, each computing it once. *)
      test "a value broadcast through a reshape is inlined" (fun () ->
          let v = staged ~size:(Int 4) (chain 200) in
          let read i = [ Ops.index v [ Ops.O.(i % int 4) ] ] in
          equal (list int) [] (new_storage (sum_reads read)));
      test "a value read through padding is inlined" (fun () ->
          let v = staged ~size:(Int 4) (chain 200) in
          let read i = [ Ops.index v [ Shape.valid i Ops.O.(i < int 4) ] ] in
          equal (list int) [] (new_storage (sum_reads read)));
      (* 200 reads at distinct indices copy a reversal of [source] 199 times
         past the first, past the budget: the load computes nothing storing
         would save. *)
      test "a view of storage read past the budget is inlined" (fun () ->
          let s = loop 216 0 in
          let flipped = Ops.index (source ()) [ Ops.O.(int 255 - s) ] in
          let v = Ops.bufferize ~opts:removable flipped [ s ] in
          equal (list int) [] (new_storage (read_200 v)));
      (* A position loaded from storage is computed: 200 reads copy the load of
         [positions] and its conversion 199 times past the first. *)
      test "a gather at loaded positions read past the budget is stored"
        (fun () ->
          let s = loop 216 0 in
          let gathered = Ops.index (source ()) [ loaded_at s ] in
          let v = Ops.bufferize ~opts:removable gathered [ s ] in
          equal (list int) [ 216 ] (new_storage (read_200 v)));
      (* Comparisons of positions are values where they are stored: 200 reads
         copy the stage's 4 nodes 199 times past the first. *)
      test "a mask of positions read past the budget is stored" (fun () ->
          let s = loop 216 0 in
          let v =
            Ops.bufferize ~opts:removable
              Ops.O.((s < int 100) land (int 3 < s))
              [ s ]
          in
          let read i =
            List.init 200 (fun k -> Ops.cast (Ops.index v (at k i)) Float32)
          in
          equal (list int) [ 216 ] (new_storage (sum_reads read)));
      (* 120 stores write [out] at [positions (16 k + i)] through a stage of its
         gather there, whose copies pass the budget: each position takes the
         value stored there. *)
      test "stores through a stage of their destination write the destination"
        (fun () ->
          let n = 2048 in
          let o = Shape.param ~device:cpu ~shape:[ Int n ] 0 Float32 in
          let s = loop n 0 and i = loop 16 9 in
          let gathered = Ops.index o [ loaded_at ~n s ] in
          let view = Ops.bufferize ~opts:removable gathered [ s ] in
          let store k =
            Ops.end_
              (Ops.store
                 (Ops.index view (at (16 * k) i))
                 (Ops.index (source ()) [ i ]))
              [ i ]
          in
          let graph =
            Rangeify.get_kernel_graph ~cores:Cores.host
              (Ops.sink [ Ops.after o (List.init 120 store) ])
          in
          (* A permutation of the [n] positions. *)
          let position j = j * 7 mod n in
          let buffers =
            [
              (0, Array.make n (`Float 0.));
              (1, Array.init 256 (fun j -> `Float (float_of_int j)));
              (2, Array.init n (fun j -> `Int (Bigint.of_int (position j))));
            ]
          in
          let expected =
            List.sort compare
              (List.concat
                 (List.init 120 (fun k ->
                      List.init 16 (fun i ->
                          let at = position ((16 * k) + i) in
                          (0, at, `Float (float_of_int i))))))
          in
          equal (list write) expected
            (List.filter
               (fun (slot, _, _) -> slot = 0)
               (Kernel_graphs.writes ~buffers graph)));
    ]

(* The bound on copies

   A stage read once, at two indices, in windows or under a sum that selects one
   expert's matrix, stays stored exactly when the nodes its ranges reach times
   its copies beyond the first exceed [budget]. *)

(* The kernel graph of a graph that stores, at each output [r] of [experts *
   rows * 2], the sum over [k] of 4 of [x] at [(r / 2, k)] times the sum over
   the experts [e] of [v] at [(e, r mod 2, k)] where [e] is the expert [ids]
   gives the row [r / 2]: a routed product, whose rows gather their expert's
   matrix by a sum that selects one trip. *)
let routed_item ~experts ~rows v =
  let n = experts * rows in
  let o = Shape.param ~device:cpu ~shape:[ Int (2 * n) ] 0 Float32 in
  let ids = Shape.param ~device:cpu ~shape:[ Int n ] 2 Int32 in
  let x = Shape.param ~device:cpu ~shape:[ Int (4 * n) ] 3 Float32 in
  let r = loop (2 * n) 9 and k = Ops.range ~axis_type:Weak (Int 4) [ 10 ] in
  let e = Ops.range ~axis_type:Weak (Int experts) [ 11 ] in
  let row = Ops.O.(r // int 2) in
  let selected = Ops.eq (Ops.index ids [ row ]) (Ops.cast e Dtype.Int32) in
  let read = Ops.index v [ e; Ops.O.(r % int 2); k ] in
  let gathered =
    Ops.reduce (Ops.where selected read (Ops.O.float 0.)) Add [ e ]
  in
  let product =
    Ops.reduce Ops.O.(gathered * Ops.index x [ (row * int 4) + k ]) Add [ k ]
  in
  Ops.after o [ Ops.end_ (Ops.store (Ops.index o [ r ]) product) [ r ] ]

let routed ~experts ~rows v =
  Rangeify.get_kernel_graph ~cores:Cores.host
    (Ops.sink [ routed_item ~experts ~rows v ])

(* [routed_staged ~experts v] is [v] of [source] at [8e + 4j + k], staged over
   [e] of [experts], [j] of 2 and [k] of 4. [routed_staged ~experts (chain k)]
   copies [2k + 8] nodes. *)
let routed_staged ?(opts = removable) ~experts v =
  let e = loop experts 0 and j = loop 2 1 and k = loop 4 2 in
  let at = Ops.O.((e * int 8) + (j * int 4) + k) in
  Ops.bufferize ~opts (v (Ops.index (source ()) [ at ])) [ e; j; k ]

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

(* The elements of the stage a reading reads. *)
let stage_size = function Routed (experts, _) -> 8 * experts | _ -> 32

let read_by_reading reading k =
  match reading with
  | Once -> sum_reads (once (staged (chain k)))
  | Twice -> sum_reads (twice (staged (chain k)))
  | Windows w ->
      reducing ~width:w (fun i j ->
          Ops.index (staged (chain k)) [ Ops.O.(i + j) ])
  | Routed (experts, rows) ->
      routed ~experts ~rows (routed_staged ~experts (chain k))

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
  |> Gen.with_pp (fun ppf (r, k) ->
      Format.fprintf ppf "%a, chain %d" pp_reading r k)

(* The chains of the copies within the budget and past it: read twice, 360 and
   362 nodes; in windows of 16, 8 an element, 50 and 52. *)
let bound_examples =
  [
    (Routed (2, 2), 200);
    (Routed (2, 2), 0);
    (Twice, 179);
    (Twice, 180);
    (Windows 16, 24);
    (Windows 16, 25);
  ]

let work_bound =
  group "get_kernel_graph › an inlinable stage"
    [
      prop "is stored when its copies beyond the first pass 360 nodes"
        ~examples:bound_examples reading_gen (fun (reading, k) ->
          let work = copied reading k * (repeats reading - 1) in
          let stored = work > budget in
          let is_routed = match reading with Routed _ -> true | _ -> false in
          cover "read once" (reading = Once);
          cover "read twice" (reading = Twice);
          cover "read in windows"
            (match reading with Windows _ -> true | _ -> false);
          cover "routed, stored" (is_routed && stored);
          cover "routed, inlined"
            ((match reading with Routed (_, r) -> r > 1 | _ -> false)
            && not stored);
          cover "stored" stored;
          equal (list int)
            (if stored then [ stage_size reading ] else [])
            (new_storage (read_by_reading reading k)));
    ]

(* Dropped digits

   A read whose index reads an output range only through floor divisions and
   remainders by constants drops the range's other digits: the outputs that
   differ only there read the value at one index, and a tile of them shares it,
   as a broadcast does. Its copies leave that reuse out. A window, a reduction
   range, a gather, a selection, a digit read twice in one component or also by
   another, and a constant that does not divide what it applies to stay
   counted. *)

(* The storage of slot 1: 4096 floats. *)
let wide () = Shape.param ~device:cpu ~shape:[ Int 4096 ] 1 Float32

(* [stage sizes v] is [v] of [wide] at the flat index of ranges of [sizes],
   staged over them. *)
let stage sizes v =
  let rs = List.mapi (fun i n -> loop n i) sizes in
  let at =
    List.fold_left2
      (fun a r n -> Ops.O.((a * int n) + r))
      (Ops.O.int 0) rs sizes
  in
  Ops.bufferize ~opts:removable (v (Ops.index (wide ()) [ at ])) rs

(* The kernel graph of a graph that stores, at each [r] of [rows] of 256
   outputs, the sum over [k] of 16 of [read r k]. *)
let by_rows ~rows read =
  let o = Shape.param ~device:cpu ~shape:[ Int 256 ] 0 Float32 in
  let r = Ops.range ~axis_type:Loop rows [ 9 ] in
  let k = Ops.range ~axis_type:Weak (Int 16) [ 10 ] in
  let store = Ops.store (Ops.index o [ r ]) (Ops.reduce (read r k) Add [ k ]) in
  Rangeify.get_kernel_graph ~cores:Cores.host
    (Ops.sink [ Ops.after o [ Ops.end_ store [ r ] ] ])

(* A stage of [sizes] read at [index r k] by [by_rows]: [copied] nodes, past the
   budget at the copies counted without the reuse a dropped digit shares, and
   [shared] if that reuse is all of them. *)
type digits = {
  name : string;
  rows : int;
  sizes : int list;
  index : Ops.t -> Ops.t -> Ops.t list;
  copied : int;
  shared : bool;
}

(* The row of [r]'s expert, loaded from the four ids of slot 2. *)
let gather r =
  let ids = Shape.param ~device:cpu ~shape:[ Int 4 ] 2 Int32 in
  Ops.cast (Ops.index ids [ Ops.O.(r // int 16) ]) Weak_int

(* The counts beside each row are the copies of each element without the shared
   reuse: the outputs and trips that read it. *)
let digit_cases =
  let open Ops.O in
  [
    (* 16 *)
    {
      name = "[r / 16; k]";
      rows = 64;
      sizes = [ 4; 16 ];
      index = (fun r k -> [ r // int 16; k ]);
      copied = past 16;
      shared = true;
    };
    (* 16 *)
    {
      name = "[r mod 16; k]";
      rows = 256;
      sizes = [ 16; 16 ];
      index = (fun r k -> [ r % int 16; k ]);
      copied = past 16;
      shared = true;
    };
    (* 4 *)
    {
      name = "[r / 64; r mod 16; k]";
      rows = 256;
      sizes = [ 4; 16; 16 ];
      index = (fun r k -> [ r // int 64; r % int 16; k ]);
      copied = past 4;
      shared = true;
    };
    (* 16, the trips of [k] *)
    {
      name = "[(r / 16) mod 4; k]";
      rows = 256;
      sizes = [ 4; 16 ];
      index = (fun r k -> [ r // int 16 % int 4; k ]);
      copied = past 16;
      shared = true;
    };
    (* 16 *)
    {
      name = "[r / 16 + k]";
      rows = 64;
      sizes = [ 19 ];
      index = (fun r k -> [ (r // int 16) + k ]);
      copied = past 16;
      shared = false;
    };
    (* 4 *)
    {
      name = "[r; k / 4]";
      rows = 16;
      sizes = [ 16; 4 ];
      index = (fun r k -> [ r; k // int 4 ]);
      copied = past 4;
      shared = false;
    };
    (* 16 *)
    {
      name = "[ids (r / 16); k]";
      rows = 64;
      sizes = [ 4; 16 ];
      index = (fun r k -> [ gather r; k ]);
      copied = past 16;
      shared = false;
    };
    (* 64 outputs over 28 rows: 2.3 *)
    {
      name = "[(r / 16) * 4 + r mod 16; k]";
      rows = 64;
      sizes = [ 28; 16 ];
      index = (fun r k -> [ (r // int 16 * int 4) + (r % int 16); k ]);
      copied = 300;
      shared = false;
    };
    (* 64 outputs over 15 rows: 4.3 *)
    {
      name = "[r mod 8 + r / 8; k]";
      rows = 64;
      sizes = [ 15; 16 ];
      index = (fun r k -> [ (r % int 8) + (r // int 8); k ]);
      copied = 120;
      shared = false;
    };
    (* 16 outputs over 3 rows: 5.3 *)
    {
      name = "[r mod 3; k] of 16 rows";
      rows = 16;
      sizes = [ 3; 16 ];
      index = (fun r k -> [ r % int 3; k ]);
      copied = 90;
      shared = false;
    };
    (* 64 outputs over 124 elements, each read at 16 trips: 8.3 *)
    {
      name = "[r / 16; r mod 16 + k]";
      rows = 64;
      sizes = [ 4; 31 ];
      index = (fun r k -> [ r // int 16; (r % int 16) + k ]);
      copied = 60;
      shared = false;
    };
    (* 50 outputs over 4 rows: 12.5 *)
    {
      name = "[r / 16; k] of 50 rows";
      rows = 50;
      sizes = [ 4; 16 ];
      index = (fun r k -> [ r // int 16; k ]);
      copied = 40;
      shared = false;
    };
  ]

(* The kernel graph of a graph that stores, at each point of ranges of [rows],
   the sum over [k] of 16 of [v] read at [read] of those ranges and [k]. *)
let by_outputs ~rows read v =
  let o = Shape.param ~device:cpu ~shape:[ Int 4096 ] 0 Float32 in
  let rs = List.mapi (fun i n -> loop n (20 + i)) rows in
  let k = Ops.range ~axis_type:Weak (Int 16) [ 10 ] in
  let at =
    List.fold_left2 (fun a r n -> Ops.O.((a * int n) + r)) (Ops.O.int 0) rs rows
  in
  let value = Ops.reduce (Ops.index v (read rs k)) Add [ k ] in
  let store = Ops.store (Ops.index o [ at ]) value in
  Rangeify.get_kernel_graph ~cores:Cores.host
    (Ops.sink [ Ops.after o [ Ops.end_ store rs ] ])

(* The digit of [d] a read drops from its outputs: of [4 * d] rows read at [r /
   d] or at [r mod d], or of [4 * d * 4] read at [r / (4 d)] and [r mod 4]. *)
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

(* The copies of each element counted without the digit's reuse: the trips of
   [k], 16, or the outputs that read it, fewer. *)
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

(* Each form's digit dropped past the budget: 362 nodes copied 1, 122 copied 3
   and 182 copied 2 times beyond the first. *)
let past_each_form = [ (Low, 2, 362); (High, 2, 122); (Middle, 3, 182) ]

let digits =
  group "get_kernel_graph › dropped digits"
    [
      cases "a dropped digit's reuse is no copy"
        ~id:(fun c -> c.name)
        digit_cases
        (fun c ->
          let v = stage c.sizes (chain_of c.copied) in
          let size = List.fold_left ( * ) 1 c.sizes in
          equal (list int)
            (if c.shared then [] else [ size ])
            (new_storage
               (by_rows ~rows:(Int c.rows) (fun r k ->
                    Ops.index v (c.index r k)))));
      (* [r / 16] of 49 to 64 rows: 16 copies at the largest size. *)
      test "outputs of a symbolic size stay counted at their largest" (fun () ->
          let n =
            Ops.variable "n" (`Int (Bigint.of_int 49)) (`Int (Bigint.of_int 64))
          in
          let v = stage [ 4; 16 ] (chain_of (past 16)) in
          let read r k = Ops.index v Ops.O.[ r // int 16; k ] in
          equal (list int) [ 64 ] (new_storage (by_rows ~rows:(Sym n) read)));
      prop "a read of merged outputs decides as one of them split"
        ~examples:past_each_form digit_gen (fun (form, d, n) ->
          let past = n * (digit_copies form d - 1) > budget in
          cover "a low digit dropped, past the budget" (form = Low && past);
          cover "a high digit dropped, past the budget" (form = High && past);
          cover "a middle digit dropped, past the budget" (form = Middle && past);
          cover "under the budget" (not past);
          let v = digit_stage form d n in
          let rows = split form d in
          equal (list int)
            (new_storage
               (by_outputs ~rows (fun rs k -> split_read form rs k) v))
            (new_storage
               (by_outputs
                  ~rows:[ List.fold_left ( * ) 1 rows ]
                  (fun rs k -> merged_read form d (List.hd rs) k)
                  v)));
      (* [r mod 2] reads a digit of the rows, which the sum over the experts
         also runs in, selecting one: 4 copies. *)
      test "a digit a selection runs in stays counted" (fun () ->
          let v = routed_staged ~experts:2 (chain 60) in
          equal (list int) [ 16 ] (new_storage (routed ~experts:2 ~rows:4 v)));
    ]

(* Selections

   A sum over one range that selects one trip, where an index equals the range,
   is computed at the selected index alone: a stage it reads counts the ranges
   that index runs in, as a read at that index would. A factor over another
   range inside the sum broadcasts the selected value along that range, as it
   would the gathered one. *)

type selecting = Selected | Gathered

(* The kernel graph of a graph that stores, at each output [r] of [experts *
   rows * 2] and [m] of [width], the sum over [k] of 4 of [x] at [(r / 2, k)]
   times [v] at the row's expert [ids (r / 2)], [r mod 2] and [k], times [y] at
   [m]: the expert's matrix read by a sum over the experts that selects one and
   multiplies [y] inside it ([Selected]), or at the expert's index
   ([Gathered]). *)
let scaled_product ~experts ~rows ~width form v =
  let n = experts * rows in
  let o = Shape.param ~device:cpu ~shape:[ Int (2 * n * width) ] 0 Float32 in
  let ids = Shape.param ~device:cpu ~shape:[ Int n ] 2 Int32 in
  let x = Shape.param ~device:cpu ~shape:[ Int (4 * n) ] 3 Float32 in
  let y = Shape.param ~device:cpu ~shape:[ Int width ] 4 Float32 in
  let r = loop (2 * n) 9 and m = loop width 12 in
  let k = Ops.range ~axis_type:Weak (Int 4) [ 10 ] in
  let row = Ops.O.(r // int 2) in
  let scale = Ops.index y [ m ] in
  let term =
    match form with
    | Selected ->
        let e = Ops.range ~axis_type:Weak (Int experts) [ 11 ] in
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
  Rangeify.get_kernel_graph ~cores:Cores.host
    (Ops.sink [ Ops.after o [ Ops.end_ store [ r; m ] ] ])

(* Experts, rows an expert, the width of the factor's range and a chain's steps,
   its copied nodes either side of the budget at a row's copies of its expert's
   matrix, [min 4 rows], and at those times the width. *)
let scaled_gen =
  Gen.(
    let* experts = of_list [ 2; 4; 8 ] in
    let* rows = of_list [ 1; 1; 2; 4 ] in
    let* width = of_list [ 1; 2; 4; 8 ] in
    let+ k = int_range 0 200 in
    (experts, rows, width, k))
  |> Gen.with_pp (fun ppf (e, r, w, k) ->
      Format.fprintf ppf "%d rows for each of %d experts, width %d, chain %d" r
        e w k)

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
          let made form =
            new_storage (scaled_product ~experts ~rows ~width form v)
          in
          equal (list int) (made Gathered) (made Selected));
    ]

(* A spelled exponential selected and broadcast

   A spelled exponential of 2 experts' matrices of 2 by 4, staged as a value
   read where it is broadcast ({!Ops.type-keep}), that one store reads through a
   sum selecting each row's expert ({!routed_item}'s product) and another reads
   broadcast along [m] of 2: the selecting sum's read is left to the budget,
   which inlines it, and the broadcast read keeps the stage stored. *)

let kept_exponential () =
  routed_staged
    ~opts:{ removable with keep = Broadcast }
    ~experts:2 Transcendental.xexp2

(* A store of [s] times [y] at [m] of 2: a read of [s] where it is broadcast
   along [m]. *)
let broadcast_item s =
  let o = Shape.param ~device:cpu ~shape:[ Int 32 ] 5 Float32 in
  let y = Shape.param ~device:cpu ~shape:[ Int 2 ] 4 Float32 in
  let e = loop 2 13 and j = loop 2 14 and k = loop 4 15 and m = loop 2 16 in
  let at = Ops.O.((((e * int 8) + (j * int 4) + k) * int 2) + m) in
  let value = Ops.O.(Ops.index s [ e; j; k ] * Ops.index y [ m ]) in
  Ops.after o [ Ops.end_ (Ops.store (Ops.index o [ at ]) value) [ e; j; k; m ] ]

let ranged items = Rangeify.get_kernel_graph ~cores:Cores.host (Ops.sink items)

let selected_and_broadcast =
  group "get_kernel_graph › a spelled exponential selected and broadcast"
    [
      test "is inlined where a selecting sum alone reads it" (fun () ->
          let s = kept_exponential () in
          equal (list int) []
            (new_storage (ranged [ routed_item ~experts:2 ~rows:2 s ])));
      test "stays stored for its broadcast read" (fun () ->
          let s = kept_exponential () in
          equal (list int) [ 16 ]
            (new_storage
               (ranged [ routed_item ~experts:2 ~rows:2 s; broadcast_item s ])));
    ]

(* Values that several kernels compute

   A value of no range that [k] kernels compute is stored once, in one element,
   when the nodes of its value that each kernel past the first computes again,
   [(k - 1) n], exceed [budget], [n] counting its computed nodes down to storage
   and to the values stored below it. Under the budget, each kernel computes it.
   A weak value is never stored, and below it, as below a load the kernels
   share, the value beneath is weighed. *)

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

let past_the_budget_twice () =
  read_by [ 16; 8 ] (fun _ -> steps ((budget / 2) + 1))

(* [l], two nodes over [h], is read by two kernels and [h] by a third: stored,
   [h] leaves [l] two nodes, which its two kernels compute again. *)
let cheap_above () =
  let h = steps ((budget / 2) + 1) in
  let l = Ops.O.((h * float 2.) + float 1.) in
  read_by [ 16; 8; 4 ] (fun j -> if j < 2 then l else h)

(* The low bits of a chain past the budget over the int32 scalar of slot 1, a
   weak integer: the chain is an int32, and each read converts it again. *)
let low_bits () =
  let t = chain ((budget / 2) + 1) (scalar ~dt:Int32 1) in
  let three = Ops.const ~dtype:Int32 (`Int (Bigint.of_int 3)) in
  Ops.cast (Ops.bitwise_and t three) Weak_int

let table slot = Shape.param ~device:cpu ~shape:[ Int 4 ] slot Float32

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
      test
        "a value of no range read by two kernels within the budget is computed \
         in each" (fun () ->
          let sink = read_by [ 16; 8 ] (fun _ -> steps (budget / 2)) in
          equal (list int) [] (made sink));
      test
        "a value of no range read by two kernels past the budget is stored \
         once, in one element" (fun () ->
          equal (list int) [ 1 ] (made (past_the_budget_twice ())));
      (* Three kernels make two copies past the first: [2 (2k)] nodes. *)
      test "a value three kernels read within the budget is computed in each"
        (fun () ->
          let sink = read_by [ 16; 8; 4 ] (fun _ -> steps (budget / 4)) in
          equal (list int) [] (made sink));
      test "a value three kernels read past the budget is stored once"
        (fun () ->
          let sink = read_by [ 16; 8; 4 ] (fun _ -> steps ((budget / 4) + 1)) in
          equal (list int) [ 1 ] (made sink));
      test "a weak value is never stored, and the value below it is" (fun () ->
          let sink =
            read_by [ 16; 8 ] (fun j ->
                Ops.index (table (30 + j)) [ low_bits () ])
          in
          equal (list Dtypes.dtype) [ Int32 ] (new_types (schedule sink)));
      (* One load of one table at the sum's low bits is shared by both readers:
         the load is no candidate, and the sum below it is. *)
      test "a value read through a load several kernels share is stored once"
        (fun () ->
          let sink =
            read_by [ 16; 8 ] (fun _ -> Ops.index (table 30) [ low_bits () ])
          in
          equal (list Dtypes.dtype) [ Int32 ] (new_types (schedule sink)));
      test "a value cheap above the values stored below it is computed in each"
        (fun () -> equal (list int) [ 1 ] (made (cheap_above ())));
      (* Without the rule, the last output computes the whole chain and the
         kernels hold [m^2] nodes. Past the budget's first few hundred outputs,
         a kernel computes at most the budget again. *)
      test "a chain of outputs makes kernels that stop growing with its length"
        (fun () ->
          let shallow = largest (outputs_chain 1000)
          and deep = largest (outputs_chain 2000) in
          at_most int ~than:shallow deep);
    ]

(* Max_kernel_buffers *)

let buffer_limit =
  group "get_kernel_graph › max_kernel_buffers"
    [
      (* Each of these programs is recorded under a limit of 4. *)
      cases "of 4 keeps each kernel to 4 storages" ~id:Fun.id
        [
          "many_inputs_limited";
          "many_matrices_limited";
          "many_cubes_limited";
          "many_sums_limited";
        ] (fun name ->
          let accessed =
            List.map
              (fun c -> List.length (List.tl (Ops.src c)))
              (calls (kernel_graph name))
          in
          equal (list int) [] (List.filter (fun n -> n > 4) accessed));
      test "at 0 lets one kernel read every storage" (fun () ->
          let graph =
            Setting.context
              [ Setting.B (Setting.max_kernel_buffers, 0) ]
              (fun () ->
                Rangeify.get_kernel_graph ~cores:Cores.host
                  (recorded "many_inputs"))
          in
          equal int 1 (kernels graph));
    ]

(* Stores that read their destination

   A store whose value reads its destination only at the element it writes,
   through the destination itself and elementwise operations, runs as one kernel
   when no other store reads the destination: each element is loaded before it
   is overwritten. Read at another element or in a reduction, or read by another
   store, the value is staged. A sum in the value reading a storage keeps the
   stage stored, so that the stage is seen in the kernel counts. *)

let own () = square (flat 0)

let reduced () =
  Shape.reshape
    (Shape.rop (input ~shape:[ 4; 4; 4 ] 1) Add [ 2 ])
    (ints [ 4; 4 ])

let reads_own_element () =
  stores Ops.O.((own () * float 2.) + Ops.exp2 (own ()) + reduced ())

let five_reading_own () =
  stores Ops.O.(own () + sum_of (List.map input [ 1; 2; 3; 4 ]))

let reads_other_element () =
  stores Ops.O.(Shape.permute (own ()) [ 1; 0 ] + reduced ())

let reduces_own () =
  let sums = Shape.reshape (Shape.rop (own ()) Add [ 1 ]) (ints [ 4; 1 ]) in
  stores Ops.O.(own () + Shape.expand sums (ints [ 4; 4 ]) + reduced ())

let reads_own_slice () =
  let rows u = Shape.shrink u [ Some (Ops.Int 0, Ops.Int 2); None ] in
  let dest = rows (own ()) in
  Ops.sink
    [ Ops.after (flat 0) [ Ops.store dest Ops.O.(dest + rows (reduced ())) ] ]

let reads_own_bits () =
  let bits = Ops.bitcast (own ()) Int32 in
  stores Ops.O.(Ops.bitcast (bits lsr Ops.bitcast (reduced ()) Int32) Float32)

let read_by_another_store () =
  Ops.sink
    [
      Ops.after (flat 0) [ Ops.store (own ()) Ops.O.(own () + reduced ()) ];
      Ops.after (flat 4)
        [ Ops.store (square (flat 4)) Ops.O.(own () * float 3.) ];
    ]

let self_reads =
  group "get_kernel_graph › stores that read their destination"
    [
      test
        "a value that reads its destination at the element written is one \
         kernel" (fun () -> equal int 1 (kernels_of (reads_own_element ())));
      test
        "a value of five storages that reads its destination at the element \
         written is one kernel" (fun () ->
          equal int 1 (kernels_of (five_reading_own ())));
      test "a value that reads its destination at another element is staged"
        (fun () -> equal (list int) [ 16 ] (made (reads_other_element ())));
      (* The row sums read the destination before the store overwrites it. *)
      test "a value that reduces its destination reads the sums first"
        (fun () -> equal (list int) [ 4 ] (made (reduces_own ())));
      test
        "a value that reads a slice of its destination at the element written \
         is one kernel" (fun () ->
          equal int 1 (kernels_of (reads_own_slice ())));
      test
        "a value that reads its destination through a bitcast of one size is \
         one kernel" (fun () -> equal int 1 (kernels_of (reads_own_bits ())));
      (* Two kernels: the other store reads the destination before its
         overwrite. *)
      test "a value whose destination another store reads runs after it"
        (fun () -> equal int 2 (kernels_of (read_by_another_store ())));
    ]

(* Storage it makes *)

(* The storage [graph] makes and writes that no call reads after the write.
   Storage no kernel writes, because every value stored into it was invalid, may
   still be read. *)
let unread graph =
  let nodes = Ops.toposort ~calls:Skip graph in
  let after_of a u = Ops.op u = After && Ops.nth u 0 == a in
  let written a = List.exists (after_of a) nodes in
  let read a =
    List.exists
      (fun c -> List.exists (after_of a) (List.tl (Ops.src c)))
      (calls graph)
  in
  List.filter (fun a -> Ops.op a = Alloc && written a && not (read a)) nodes

let made_storage =
  group "get_kernel_graph › storage it makes"
    [
      cases "is of a committed type, in each recorded program" ~id:Fun.id
        programs (fun name ->
          let weak dt = List.mem dt Dtype.weaks in
          equal (list Dtypes.dtype) []
            (List.filter weak (new_types (kernel_graph name))));
      cases "is read after it is written, in each recorded program" ~id:Fun.id
        programs (fun name ->
          equal (list Uops.uop) [] (unread (kernel_graph name)));
      test "of a symbolic size holds its greatest size" (fun () ->
          let v =
            Ops.variable "v" (`Int (Bigint.of_int 1)) (`Int (Bigint.of_int 16))
          in
          let out = flat 0 in
          let r = Ops.range ~axis_type:Loop (Sym v) [ 0 ] in
          let r' = Ops.range ~axis_type:Loop (Sym v) [ 1 ] in
          let staged =
            Ops.bufferize ~opts:whole (Ops.exp2 (Ops.index out [ r ])) [ r ]
          in
          let store =
            Ops.store (Ops.index out [ r' ]) (Ops.index staged [ r' ])
          in
          let sink = Ops.sink [ Ops.after out [ Ops.end_ store [ r' ] ] ] in
          equal (list int) [ 16 ] (new_storage (ranged (Ops.src sink))));
    ]

let invalid_writes =
  group "get_kernel_graph › drops invalid writes"
    [
      test "a store of invalid values runs no kernel" (fun () ->
          let invalid =
            Shape.expand (Ops.const ~dtype:Float32 `Invalid) (ints [ 16 ])
          in
          let out = flat 0 in
          equal int 0
            (kernels_of (Ops.sink [ Ops.after out [ Ops.store out invalid ] ])));
    ]

(* Calls *)

(* The slots of [c]'s storage parameters, each once, smallest first. *)
let slots c =
  List.sort_uniq compare
    (List.map
       (fun (_, (p : Ops.param_arg)) -> p.slot)
       (storage_params (Ops.nth c 0)))

let range_ids c =
  List.sort_uniq compare
    (List.filter_map
       (fun n ->
         if Ops.op n <> Range then None
         else
           match Ops.axis_id n with
           | id :: _ when id >= 0 -> Some id
           | _ -> None)
       (Ops.toposort ~calls:Enter (Ops.nth c 0)))

let from_zero l = List.init (List.length l) Fun.id

let calls_group =
  group "get_kernel_graph › calls"
    [
      cases
        "number a kernel's storage from 0, one parameter for each argument, in \
         each recorded program"
        ~id:Fun.id programs (fun name ->
          let cs = calls (kernel_graph name) in
          equal
            (list (list int))
            (List.map (fun c -> from_zero (List.tl (Ops.src c))) cs)
            (List.map slots cs));
      cases "name no device in a kernel's parameters, in each recorded program"
        ~id:Fun.id programs (fun name ->
          let placed =
            List.concat_map
              (fun c ->
                List.filter_map
                  (fun (n, (p : Ops.param_arg)) ->
                    if p.device = None then None else Some n)
                  (storage_params (Ops.nth c 0)))
              (calls (kernel_graph name))
          in
          equal (list Uops.uop) [] placed);
      cases "number a kernel's ranges from 0, in each recorded program"
        ~id:Fun.id programs (fun name ->
          let ids = List.map range_ids (calls (kernel_graph name)) in
          equal (list (list int)) (List.map from_zero ids) ids);
    ]

(* Buffer states

   A kernel reading a buffer in two states takes a parameter for each, which its
   call passes in the state it names. [beside_assigned sink] is [sink] with [x],
   slot 1, assigned [y], slot 2, as its last item. *)

let assigned () = Ops.after (flat 1) [ Ops.store (flat 1) (flat 2) ]
let beside_assigned sink = Ops.sink (Ops.src sink @ [ assigned () ])

let states =
  group "get_kernel_graph › buffer states"
    [
      test "a value read beside a later state of its buffer is stored first"
        (fun () ->
          let one = Ops.float ~dtype:Float32 1. in
          writes (beside_assigned (stores Ops.O.(flat 1 + one + assigned ()))));
      test "a buffer read beside its later state is copied first" (fun () ->
          writes (beside_assigned (stores Ops.O.(flat 1 + assigned ()))));
      test "rows gathered across a store are read before it" (fun () ->
          let at = positions ~slot:3 ~reads:4 4 in
          let rows u = Ops.index (square u) [ at ] in
          writes
            (beside_assigned
               (stores Ops.O.(rows (flat 1) + rows (assigned ())))));
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

(* The end of the one loop of calls in [sink]'s kernel graph. *)
let loop_end sink =
  match
    List.filter
      (fun u -> Ops.op u = End)
      (Ops.toposort ~calls:Skip (schedule sink))
  with
  | [ e ] -> e
  | ends -> failf "%d ends of calls" (List.length ends)

let loops =
  group "get_kernel_graph › loops of calls"
    [
      test "a loop of a precompiled call is no kernel" (fun () ->
          let sink, _ = scan_loop () in
          equal int 0 (kernels_of sink));
      test "a loop of a call ends over the loop's range" (fun () ->
          let sink, r = scan_loop () in
          equal (list Uops.uop) [ r ] (List.tl (Ops.src (loop_end sink))));
      test "a loop's call reads views that move with its range" (fun () ->
          let sink, r = scan_loop () in
          let call = Ops.nth (loop_end sink) 0 in
          equal (list bool) [ false; true; true ]
            (List.map
               (fun a -> Ops.Nodes.mem r (Ops.ranges a))
               (Ops.src_without_body call)));
    ]

(* Arguments *)

let rec storage_of u =
  match Ops.op u with
  | After -> storage_of (Ops.nth u 0)
  | Param | Buffer | Alloc -> Some u
  | _ -> None

(* The arguments of [name]'s calls that are neither storage nor a run of it
   passed to a kernel the program gave as code. *)
let not_storage name =
  let argument c a =
    match (Ops.op a, storage_of a) with
    | (Param | Buffer | Alloc | After), Some _ -> false
    | Param, _ -> false
    | _ -> not (given_as_code name c && Shape.run_of_storage a)
  in
  List.concat_map
    (fun c -> List.filter (argument c) (List.tl (Ops.src c)))
    (calls (kernel_graph name))

let arguments =
  group "get_kernel_graph › arguments"
    [
      cases "are storage without views, in each recorded program" ~id:Fun.id
        programs (fun name -> equal (list Uops.uop) [] (not_storage name));
    ]

let spec =
  group "get_kernel_graph › with Setting.spec at 1"
    [
      cases "each recorded program's kernel graph passes Spec.kernel_graph"
        ~id:Fun.id programs (fun name ->
          Setting.context
            [ Setting.B (Setting.spec, 1) ]
            (fun () -> ignore (kernel_graph name)));
    ]

let raising =
  group "get_kernel_graph › raises Invalid_argument"
    [
      test "if a stage has no elements" (fun () ->
          let s = loop 0 0 and i = loop 0 1 in
          let v = Ops.exp2 (Ops.index (source ()) [ s ]) in
          let staged = Ops.bufferize ~opts:whole v [ s ] in
          let out = flat 0 in
          let store =
            Ops.store (Ops.index out [ i ]) (Ops.index staged [ i ])
          in
          let sink = Ops.sink [ Ops.after out [ Ops.end_ store [ i ] ] ] in
          raises_match (Exn.invalid_arg ~sub:"a stage has no elements")
            (fun () -> Rangeify.get_kernel_graph ~cores:Cores.host sink));
      test "if a value is ordered after effects with no storage to hold it"
        (fun () ->
          let written = Ops.after (flat 0) [ Ops.store (flat 0) (flat 1) ] in
          let two = Ops.after (Ops.float ~dtype:Float32 2.) [ written ] in
          let out = flat 4 and i = loop 16 1 in
          let store = Ops.store (Ops.index out [ i ]) two in
          let sink = Ops.sink [ Ops.after out [ Ops.end_ store [ i ] ] ] in
          raises_match (Exn.invalid_arg ~sub:"a value ordered after effects")
            (fun () -> Rangeify.get_kernel_graph ~cores:Cores.host sink));
    ]

(* Allocation *)

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

(* [words f] is the words [f ()] allocates. A full collection first drops the
   nodes earlier tests left dead, which [f] would otherwise find instead of
   allocating: under seed s1:c7ab5af49669d437 the chain of 1000 outputs
   allocated 12.3M words without it and 13.1M with it (13.1M run alone), a ratio
   of 2.21 against 2.10 (2.10 alone). *)
let words f =
  Gc.full_major ();
  let before = Gc.minor_words () in
  ignore (Sys.opaque_identity (f ()));
  Gc.minor_words () -. before

(* Allocation linear in the length is [a * n + b] with [b >= 0], so twice the
   length allocates at most twice the words. The tenth of slack covers tables
   that double their capacity at different lengths in the two runs. *)
let linear words_of n =
  let short = words_of n and long = words_of (2 * n) in
  less float_exact ~than:(2.2 *. short) long

let allocation =
  group "get_kernel_graph › words allocated"
    [
      test
        "twice the steps of a chain of staged values allocate at most 2.2 \
         times the words to schedule" (fun () ->
          linear
            (fun n ->
              let sink = centred n in
              words (fun () -> schedule sink))
            100);
      (* Without the rule of values several kernels compute, the last output
         computes the whole chain and the kernels hold [m^2] nodes. *)
      test
        "twice the outputs of a chain allocate at most 2.2 times the words to \
         schedule" (fun () ->
          linear
            (fun m ->
              let sink = outputs_chain m in
              words (fun () -> schedule sink))
            1000);
    ]

(* Transcendental_work *)

(* A baseline, from this build: [Transcendental.xexp2] of a float32 load is 45
   elementwise operations before simplification and 42 after, none of them a
   multiply-add or a position. The tests of broadcast exponentials state what
   the value decides. *)
let transcendental =
  group "transcendental_work"
    [
      test "is the work of tolk's float32 exponential, simplified" (fun () ->
          expect (string_of_int (Rangeify.transcendental_work ()))
          @@ __POS_OF__ {|42|});
    ]

(* The writes law over hand-built functions *)

(* A staged value read at an index in padding is read nowhere. Here [s]'s two
   elements are summed into positions [0] and [1] of three, and only position
   [2] is kept: the sum there reads [s] at no index, beside a read of [s] at [0]
   that stages it. *)
let dead_read () =
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
  Ops.sink [ output 0 total ]

let beside_gpu () = Cores.host @ List.map (fun (_, c) -> ("GPU", c)) at_once
let work () = Rangeify.transcendental_work ()

(* Each row names its function, the cores its schedule is given, and the
   function. *)
let hand_built =
  let host = Cores.host in
  [
    ("a two-sum's sum and error", host, two_sum_outputs);
    ("outputs that share a reduction", host, shared_reduction);
    ( "outputs that share a broadcast reduction",
      host,
      shared_broadcast_reduction );
    ("outputs with a reduction each", host, two_reductions);
    ("slices of one product through two views", host, two_slices);
    ("slices that also share a computation", host, two_slices_sharing);
    ("outputs that read a product through one view", host, one_view_twice);
    ("an output that reads what another writes", host, reads_what_another_writes);
    ( "an output that overwrites what another reads",
      host,
      overwrites_what_another_reads );
    ( "outputs reading a storage before and after its overwrite",
      host,
      read_around_overwrite );
    ("two pairs of outputs that each read the other pair", host, crossed_pairs);
    ("a sum and a maximum of one computation", host, sum_and_max);
    ("a reduction that reads another of its domain", host, centred_squares);
    ("reductions of axes of different sizes", host, sums_of_two_widths);
    ( "a value of four storages a store reads in halves",
      host,
      four_read_in_halves );
    ( "a value of four storages a materialisation reads in halves",
      host,
      four_materialised_in_halves );
    ("a value of four storages that stages read", host, four_read_by_stages);
    ( "a value of four storages a reduction reads in halves",
      host,
      four_reduced_in_halves );
    ("a value of four storages read where it is broadcast", host, four_broadcast);
    ( "an exponential of elementwise operations broadcast",
      host,
      fun () -> broadcast (spelled ()) );
    ( "an exponential of elementwise operations every output reads",
      host,
      fun () -> vecmat (spelled ()) );
    ( "a value of an exponential's work broadcast",
      host,
      fun () -> broadcast (products (work ())) );
    ( "a value one operation under an exponential's work broadcast",
      host,
      fun () -> broadcast (products (work () - 1)) );
    ("a cheap value broadcast", host, fun () -> broadcast (cheap ()));
    ( "an exponential broadcast, its points at once",
      at_once,
      fun () -> broadcast (spelled ()) );
    ( "an exponential broadcast where only another device runs points at once",
      beside_gpu (),
      fun () -> broadcast (spelled ()) );
    ( "a value of an exponential's work every output reads, its points at once",
      at_once,
      fun () -> vecmat (products (work ())) );
    ( "an exponential a stored matrix reads where it broadcasts it",
      host,
      fun () -> weighed (spelled ()) );
    ( "an exponential a stored matrix reads where it broadcasts it, its points \
       at once",
      at_once,
      fun () -> weighed (spelled ()) );
    ( "a value of two storages a stored matrix reads where it broadcasts it",
      host,
      two_storages_weighed );
    ( "a gathered exponential read where it is broadcast",
      host,
      fun () -> gathered (exp_of 2) );
    ( "a gathered value of four storages read where it is broadcast",
      host,
      fun () -> gathered (four [ 2 ]) );
    ( "an exponential of a larger table gathered and broadcast along few rows",
      host,
      fun () -> gathered (exp_of 16) );
    ( "an exponential of a larger table gathered and broadcast along many rows",
      host,
      fun () -> gathered ~rows:64 (exp_of 16) );
    ( "a gathered exponential read where it is broadcast, its points at once",
      at_once,
      fun () -> gathered (exp_of 2) );
    ( "a gathered exponential read at more positions than it holds",
      host,
      fun () -> taken ~reads:16 (exp_of 2) );
    ( "a gathered exponential read at more positions than it holds, its points \
       at once",
      at_once,
      fun () -> taken ~reads:16 (exp_of 2) );
    ( "a gathered value of an exponential's work read at more positions than \
       it holds",
      host,
      fun () -> taken ~reads:16 (products (work ())) );
    ( "a gathered exponential read at one position more than it holds",
      host,
      fun () -> taken ~reads:5 (exp_of 4) );
    ( "a gathered exponential read at as many positions as it holds",
      host,
      fun () -> taken ~reads:4 (exp_of 4) );
    ( "a gathered exponential read at fewer positions than it holds",
      host,
      fun () -> taken ~reads:2 (exp_of 16) );
    ( "an exponential two gathers read at more positions together than it holds",
      host,
      fun () -> taken_twice ~reads:3 (exp_of 4) );
    ( "a cheap gathered value read at more positions than it holds",
      host,
      fun () -> taken ~reads:16 (cheap ()) );
    ( "a cheap gathered value of two storages a sum reads at more positions \
       than it holds",
      host,
      fun () -> summed ~reads:16 (two_inputs ()) );
    ( "a cheap gathered value of one storage a sum reads at more positions \
       than it holds",
      host,
      fun () -> summed ~reads:16 (cheap ()) );
    ( "a scaled matrix a product reads",
      host,
      fun () -> product 4 8 4 (scaled 4 8) );
    ( "a matrix holding more bytes than it reads, a product reads",
      host,
      fun () -> product 2 181 16 (outer 2 181) );
    ( "a value that reads its destination at the element written",
      host,
      reads_own_element );
    ( "a value of five storages that reads its destination at the element \
       written",
      host,
      five_reading_own );
    ( "a value that reads its destination at another element",
      host,
      reads_other_element );
    ("a value that reduces its destination", host, reduces_own);
    ( "a value that reads a slice of its destination at the element written",
      host,
      reads_own_slice );
    ( "a value that reads its destination through a bitcast of one size",
      host,
      reads_own_bits );
    ( "a value whose destination another store reads",
      host,
      read_by_another_store );
    ( "a value of no range read by two kernels past the budget",
      host,
      past_the_budget_twice );
    ("a value cheap above the values stored below it", host, cheap_above);
    ("a staged value read only in padding, which reads 0 there", host, dead_read);
  ]

let writes_law =
  group "get_kernel_graph › writes what its tensor graph writes"
    [
      group "for each recorded program" recorded_writes;
      generated_writes;
      cases "for a hand-built function"
        ~id:(fun (name, _, _) -> name)
        hand_built
        (fun (_, cores, sink) -> writes ~cores (sink ()));
    ]

let () =
  exit
    (run "Tolk.Rangeify"
       [
         goldens;
         writes_law;
         kernels_group;
         products_apart;
         shared;
         siblings;
         needless;
         storages;
         broadcast_exponentials;
         gathered_broadcast;
         gathered_again;
         reread;
         copies;
         recomputation;
         work_bound;
         digits;
         selections;
         selected_and_broadcast;
         shared_scalars;
         buffer_limit;
         self_reads;
         made_storage;
         invalid_writes;
         calls_group;
         states;
         loops;
         arguments;
         spec;
         raising;
         allocation;
         transcendental;
         own_ranges;
         store_checks;
       ])
