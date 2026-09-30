(* Tests of Tolk_next.Multi: what multi_pm makes of the programs tinygrad
   schedules on several devices and of kernels sharded across threads, the law
   that its rewrite keeps what a program computes, and each of its rules. *)

open Windtrap
open Tolk_next

let uop = Uops.uop
let multi u = Ops.graph_rewrite ~ctx:() u Multi.multi_pm
let program name = Golden.sink (name ^ ".golden")

(* Recorded graphs *)

let programs =
  [
    "add";
    "add_scalar";
    "add_whole";
    "add_broadcast";
    "add_resharded";
    "add_resharded_1";
    "add_replicated_scalar";
    "add_four";
    "add_eight";
    "where";
    "cast_half";
    "bitcast";
    "sum_sharded_axis";
    "sum_sharded_axis_1";
    "sum_other_axis";
    "sum_all";
    "sum_all_1";
    "max_sharded_axis";
    "max_sharded_axis_1";
    "allreduce_cast";
    "allreduce_cast_1";
    "allreduce_no_cast";
    "allreduce_no_cast_1";
    "allreduce_cast_bfloat16";
    "allreduce_cast_bfloat16_1";
    "allreduce_cast_float";
    "allreduce_cast_float_1";
    "explicit_allreduce";
    "explicit_allreduce_1";
    "reshape_split";
    "reshape_inner";
    "expand";
    "permute";
    "pad";
    "flip";
    "shrink_other_axis";
    "shrink_one_shard";
    "shrink_element";
    "shrink_chained";
    "shrink_rows";
    "reshape_then_shrink";
    "add_two_partitions";
    "variable_shrink";
    "stack";
    "cat";
    "repeat";
    "grid_add";
    "grid_add_1";
    "grid_sum_all";
    "grid_sum_all_1";
    "grid_sum_other_axis";
    "grid_sum_other_axis_1";
    "grid_matmul";
    "grid_matmul_1";
    "grid_to_one";
    "grid_to_one_1";
    "matmul_rows";
    "matmul_columns";
    "matmul_contracted";
    "matmul_contracted_1";
    "matmul_resharded";
    "matmul_resharded_1";
    "matmul_resharded_2";
    "double_matmul";
    "softmax";
    "softmax_sharded_axis";
    "softmax_sharded_axis_1";
    "softmax_sharded_axis_2";
    "layernorm";
    "rmsnorm";
    "attention";
    "conv";
    "embedding";
    "arange";
    "cumsum";
    "sort";
    "interpolate";
    "batchnorm_stats";
    "batchnorm_stats_1";
    "shard";
    "replicate";
    "gather";
    "gather_four";
    "select_first";
    "reshard_devices";
    "reshard_devices_1";
    "contiguous";
    "assign";
    "assign_shard";
    "setitem_rows";
    "setitem_row";
    "setitem_columns";
    "setitem_stride";
    "setitem_replicated_value";
    "setitem_sharded_value";
    "shard_invalids";
    "custom_kernel";
    "inline_function";
  ]

let kernels =
  [
    "fragment_blocks";
    "fragment_strided";
    "alu_scalar";
    "alu_whole";
    "store_value";
    "store_value_two_axes";
    "store_load";
  ]

(* The settings a program was recorded under. *)
let settings = function
  | "allreduce_no_cast" -> [ Helpers.B (Helpers.allreduce_cast, false) ]
  | _ -> []

let rewritten name =
  Golden.graph (name ^ "_multi.golden") (fun () ->
      Helpers.context (settings name) (fun () -> multi (program name)))

let recorded =
  group "multi_pm › recorded"
    [
      group "programs" (List.map rewritten programs);
      group "kernels" (List.map rewritten kernels);
    ]

(* Values

   The law that the rewrite keeps what a program computes: a sharded value is
   the whole that its devices' parts reassemble into (Tensors), so a program
   writes the same before and after its operations move to the shards. Memory
   holds small integers, the same on each device of a replicated buffer and
   different on each device of a sharded one, so that a shard read in place of
   another is seen. *)

let size shape = List.fold_left ( * ) 1 shape
let devices = function Some (Ops.Multi l) -> List.length l | _ -> 1

let sharded_storage u =
  List.filter_map
    (fun n ->
      if Ops.op n = Unshard then Some (Ops.storage_base (Ops.nth n 0)) else None)
    (Ops.toposort u)

let element dtype k : Dtype.value =
  if Dtype.is_float dtype then `Float (float_of_int k)
  else if Dtype.equal dtype Bool then `Bool (k > 0)
  else `Int (Z.of_int k)

let filled u =
  let sharded = sharded_storage u in
  List.filter_map
    (fun n ->
      match (Ops.op n, Ops.arg n) with
      | ( (Param | Buffer | Alloc),
          Param { slot; size; dtype; device; addrspace; _ } )
        when addrspace <> Some Alu ->
          let size = Option.value size ~default:1 in
          let per_device = List.memq n sharded in
          let at j =
            let j = if per_device then j else j mod size in
            element dtype ((((j * 7) + (slot * 3)) mod 11) - 3)
          in
          Some (slot, Array.init (size * devices device) at)
      | _ -> None)
    (Ops.toposort u)

(* A reduction across devices adds its terms in another order than one over the
   whole, so floats agree up to the rounding of float32. *)
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
      let rewritten = Helpers.context (settings name) (fun () -> multi sink) in
      equal (list write)
        (Tensors.writes ~buffers sink)
        (Tensors.writes ~buffers rewritten))

(* Programs the law does not apply to: a call of a kernel, whose ranges Tensors
   does not run, and an allreduce of a sharded value, which reduces the shards
   in place, so that no whole value holds its result. *)
let unevaluated = [ "custom_kernel"; "explicit_allreduce" ]

let concrete u =
  List.for_all
    (fun n ->
      List.for_all
        (function Ops.Int _ -> true | Sym _ -> false)
        (Option.value ~default:[] (Ops.shape_opt n)))
    (Ops.toposort u)

let values =
  group "multi_pm › values"
    (List.filter_map
       (fun name ->
         if List.mem name unevaluated || not (concrete (program name)) then None
         else Some (keeps_writes name))
       programs)

(* Generated programs: a buffer of up to three axes sharded on two, four or
   eight devices, along one axis or, on four or eight devices, along the first
   two as a grid of two rows by two or four columns, then a few operations
   chosen among those multi_pm moves to the shards. An operation that does not
   fit the value, such as a flip when every axis is sharded, is left out, so
   that multi_pm accepts every program. The law: each device of the rewritten
   value holds the value computed whole. *)

type step =
  | Add_self
  | Scale of int
  | Add_whole
  | Add_row
  | Sum of int
  | Max of int
  | Rotate of int
  | Flip of int
  | Pad of int
  | Shrink of int
  | Unsqueeze of int
  | Expand
  | Cast
  | Select of int

let pp_step ppf = function
  | Add_self -> Format.fprintf ppf "x + x"
  | Scale k -> Format.fprintf ppf "x * %d" k
  | Add_whole -> Format.fprintf ppf "x + a whole value"
  | Add_row -> Format.fprintf ppf "x + a broadcast row"
  | Sum k -> Format.fprintf ppf "sum %d" k
  | Max k -> Format.fprintf ppf "max %d" k
  | Rotate k -> Format.fprintf ppf "rotate by %d" k
  | Flip k -> Format.fprintf ppf "flip %d" k
  | Pad k -> Format.fprintf ppf "pad %d" k
  | Shrink k -> Format.fprintf ppf "shrink %d" k
  | Unsqueeze k -> Format.fprintf ppf "unsqueeze %d" k
  | Expand -> Format.fprintf ppf "expand"
  | Cast -> Format.fprintf ppf "cast to float"
  | Select k -> Format.fprintf ppf "select shard %d" k

type drawn = {
  n : int;
  shape : int list;
  axis : int;
  grid : bool;
  steps : step list;
}

let pp_drawn ppf d =
  Format.fprintf ppf "[%s] on %d devices, sharded on %s, then %a"
    (String.concat "; " (List.map string_of_int d.shape))
    d.n
    (if d.grid then Printf.sprintf "a grid of 2 by %d on axes 0 and 1" (d.n / 2)
     else Printf.sprintf "axis %d" d.axis)
    (Format.pp_print_list
       ~pp_sep:(fun ppf () -> Format.fprintf ppf ", ")
       pp_step)
    d.steps

let step =
  Gen.(
    let* k = int_range 0 7 in
    of_list ~pp:pp_step
      [
        Add_self;
        Scale (k - 3);
        Add_whole;
        Add_row;
        Sum k;
        Max k;
        Rotate k;
        Flip k;
        Pad k;
        Shrink k;
        Unsqueeze k;
        Expand;
        Cast;
        Select k;
      ])

let rec per_axis f a rank =
  if a = rank then Gen.constant []
  else
    Gen.map (fun (x, xs) -> x :: xs) (Gen.pair (f a) (per_axis f (a + 1) rank))

let drawn =
  Gen.(
    let* n = of_list [ 2; 4; 8 ] in
    let* rank = int_range 1 3 in
    let* axis = int_range 0 (rank - 1) in
    let* grid = if n > 2 && rank >= 2 then bool else constant false in
    let parts a =
      if grid then match a with 0 -> 2 | 1 -> n / 2 | _ -> 1
      else if a = axis then n
      else 1
    in
    let* shape =
      per_axis (fun a -> map (fun k -> k * parts a) (int_range 1 3)) 0 rank
    in
    let+ steps = list ~size:(int_range 1 4) step in
    { n; shape; axis; grid; steps })
  |> Gen.with_pp pp_drawn

let ints = List.map (fun n -> Ops.Int n)

let dims u =
  List.map
    (function Ops.Int n -> n | Sym _ -> fail "a generated shape is concrete")
    (Ops.shape u)

let pick k = function
  | [] -> None
  | l -> Some (List.nth l (k mod List.length l))

(* The program [d] draws: its value, the axes that are sharded across devices,
   and its memory. *)
let build d =
  let names = Ops.Multi (List.init d.n (Printf.sprintf "CPU:%d")) in
  let memory = ref [] in
  let storage ~sharded dtype shape =
    let slot = List.length !memory + 1 and size = size shape in
    let at j =
      let j = if sharded then j else j mod size in
      element dtype ((((j * 7) + (slot * 3)) mod 11) - 3)
    in
    memory := (slot, Array.init (size * d.n) at) :: !memory;
    Ops.reshape (Ops.new_buffer ~slot names size dtype) (ints shape)
  in
  let base =
    if d.grid then
      let r = Ops.range ~axis_type:Device (Int d.n) [ -1 ] in
      let columns = d.n / 2 in
      let shard =
        List.mapi
          (fun a n -> match a with 0 -> n / 2 | 1 -> n / columns | _ -> n)
          d.shape
      in
      Ops.unshard
        ~ranges:Ops.O.[ r // int columns; r % int columns ]
        (storage ~sharded:true Int32 shard)
        [ 0; 1 ]
    else
      let shard =
        List.mapi (fun a n -> if a = d.axis then n / d.n else n) d.shape
      in
      Ops.unshard (storage ~sharded:true Int32 shard) [ d.axis ]
  in
  let apply (v, axes) step =
    let shape = dims v and dtype = Ops.dtype v in
    let rank = List.length shape in
    let free =
      List.filter (fun a -> not (List.mem a axes)) (List.init rank Fun.id)
    in
    let on a f = List.mapi (fun b n -> if a = b then f n else None) shape in
    let reduce op k =
      if rank = 0 then (v, axes)
      else
        let a = k mod rank in
        let kept =
          List.filter_map
            (fun b -> if b = a then None else Some (if b > a then b - 1 else b))
            axes
        in
        if List.mem a axes && List.length axes > 1 then (v, axes)
        else (Ops.rop v op [ a ], if List.mem a axes then [] else kept)
    in
    match step with
    | Add_self -> (Ops.O.(v + v), axes)
    | Scale k -> (Ops.O.(v * int k), axes)
    | Add_whole -> (Ops.O.(v + storage ~sharded:false dtype shape), axes)
    | Add_row when rank > 0 ->
        let last = List.nth shape (rank - 1) in
        let row =
          Ops.reshape
            (storage ~sharded:false dtype [ last ])
            (ints (List.mapi (fun a n -> if a = rank - 1 then n else 1) shape))
        in
        (Ops.O.(v + Ops.expand row (ints shape)), axes)
    | Sum k -> reduce Add k
    | Max k -> reduce Max k
    | Rotate k when rank > 0 ->
        let order = List.init rank (fun i -> (i + k) mod rank) in
        ( Ops.permute v order,
          List.map (fun b -> (b - k + (8 * rank)) mod rank) axes )
    | Flip k -> (
        match pick k free with
        | Some a -> (Ops.flip v [ a ], axes)
        | None -> (v, axes))
    | Pad k -> (
        match pick k free with
        | Some a ->
            (Ops.pad v (on a (fun _ -> Some (Ops.Int 1, Ops.Int 2))), axes)
        | None -> (v, axes))
    | Shrink k -> (
        match pick k (List.filter (fun a -> List.nth shape a > 1) free) with
        | Some a ->
            (Ops.shrink v (on a (fun n -> Some (Ops.Int 1, Ops.Int n))), axes)
        | None -> (v, axes))
    | Unsqueeze k ->
        let p = k mod (rank + 1) in
        let shape =
          List.filteri (fun a _ -> a < p) shape
          @ (1 :: List.filteri (fun a _ -> a >= p) shape)
        in
        ( Ops.reshape v (ints shape),
          List.map (fun b -> if b >= p then b + 1 else b) axes )
    | Expand -> (Ops.expand v (ints (2 :: shape)), List.map succ axes)
    | Cast -> (Ops.cast v Float32, axes)
    | Select k -> (
        match axes with
        | [ a ] when not d.grid ->
            let part = List.nth shape a / d.n in
            let start = k mod d.n * part in
            ( Ops.shrink v
                (on a (fun _ -> Some (Ops.Int start, Ops.Int (start + part)))),
              [] )
        | _ -> (v, axes))
    | Add_row | Rotate _ -> (v, axes)
  in
  let v, axes =
    List.fold_left apply (base, if d.grid then [ 0; 1 ] else [ d.axis ]) d.steps
  in
  (v, axes, !memory)

let device k = function [ t ] -> t | ts -> List.nth ts k

(* [refused_by_tinygrad d] is [true] if [d] selects a shard of a product with
   zero that a cast or a reduction follows. tinygrad's rewrite refuses it:
   simplifying the copy of the shard folds the value to a constant, and a
   constant has no shards to select. *)
let refused_by_tinygrad d =
  let folds = function Cast | Sum _ | Max _ -> true | _ -> false in
  let rec after_zero = function
    | [] -> false
    | Scale 0 :: rest -> then_folded rest || after_zero rest
    | _ :: rest -> after_zero rest
  and then_folded = function
    | [] -> false
    | step :: rest when folds step ->
        List.exists (function Select _ -> true | _ -> false) rest
    | _ :: rest -> then_folded rest
  in
  after_zero d.steps

let holds_whole d =
  assume (not (refused_by_tinygrad d));
  let v, axes, buffers = build d in
  cover "a value stays sharded" (axes <> []);
  cover "a sharded axis is reduced across devices"
    (List.exists (fun n -> Ops.op n = Allreduce) (Ops.toposort (multi v)));
  let whole = Tensors.eval ~buffers v in
  let rewritten = Tensors.eval ~buffers (multi v) in
  List.iteri
    (fun k _ ->
      equal
        ~msg:(Printf.sprintf "device %d" k)
        (array Dtypes.const) (device k whole) (device k rewritten))
    (if List.length rewritten > List.length whole then rewritten else whole)

(* Counterexamples found before: one device's shard of a sum with a whole value,
   whose part of the whole is taken at the device range. *)
let examples =
  [
    {
      n = 2;
      shape = [ 2 ];
      axis = 0;
      grid = false;
      steps = [ Add_whole; Select 0 ];
    };
    {
      n = 2;
      shape = [ 2; 1 ];
      axis = 0;
      grid = false;
      steps = [ Add_row; Select 0 ];
    };
  ]

let refused_as_in_tinygrad name d =
  test (name ^ " is refused, as in tinygrad") (fun () ->
      let v, _, _ = build d in
      raises_match
        (Exn.invalid_arg
           ~substring:"a shard selection needs a value on several devices")
        (fun () -> multi v))

let laws =
  group "multi_pm › laws"
    [
      prop ~examples "a rewritten value holds the value computed whole" drawn
        holds_whole;
      refused_as_in_tinygrad "a shard of a product with zero cast to a float"
        {
          n = 2;
          shape = [ 2; 1; 1 ];
          axis = 0;
          grid = false;
          steps = [ Scale 0; Cast; Select 0 ];
        };
      refused_as_in_tinygrad "a shard of a sum of a product with zero"
        {
          n = 2;
          shape = [ 2; 2; 1 ];
          axis = 0;
          grid = false;
          steps = [ Scale 0; Sum 1; Select 0 ];
        };
    ]

(* Rules

   Hand-built graphs, each stating one rule on values of eight or four elements,
   sharded on two devices along one axis. *)

let cpu k = Ops.Single (Printf.sprintf "CPU:%d" k)
let two = Ops.Multi [ "CPU:0"; "CPU:1" ]
let four = Ops.Multi [ "CPU:0"; "CPU:1"; "CPU:2"; "CPU:3" ]

let storage ?(devices = two) ?(dtype = Dtype.Float32) slot shape =
  Ops.reshape (Ops.new_buffer ~slot devices (size shape) dtype) (ints shape)

(* A value of [shard]'s shape times the devices along [axis]. *)
let sharded ?devices ?dtype slot shard axis =
  Ops.unshard (storage ?devices ?dtype slot shard) [ axis ]

let shard u = Ops.nth u 0
let range u = Ops.nth u 1
let unshard u axes like = Ops.unshard ~ranges:[ range like ] u axes
let a = sharded 1 [ 2; 8 ] 0
let b = sharded 2 [ 2; 8 ] 0
let whole = storage 3 [ 4; 8 ]

(* [refused ~because u] checks that multi_pm refuses [u], which is built before,
   for a reason that mentions [because]. *)
let refused ~because u =
  raises_match (Exn.invalid_arg ~substring:because) (fun () -> multi u)

let has op u = List.exists (fun n -> Ops.op n = op) (Ops.toposort u)

let arithmetic =
  group "multi_pm › arithmetic"
    [
      test "an operation on values sharded alike computes on their shards"
        (fun () ->
          equal uop
            (unshard Ops.O.(shard a + shard b) [ 0 ] a)
            (multi Ops.O.(a + b)));
      test "a scalar source is kept as it is" (fun () ->
          equal uop
            (unshard Ops.O.(shard a * float 2.) [ 0 ] a)
            (multi Ops.O.(a * float 2.)));
      test "a source of the whole shape takes the shard's part of itself"
        (fun () ->
          equal uop
            (unshard
               Ops.O.(shard a + Ops.shard_slice whole 0 (range a))
               [ 0 ] a)
            (multi Ops.O.(a + whole)));
      test "a broadcast scalar is broadcast over the shard" (fun () ->
          let c =
            Ops.mop (Ops.float ~dtype:Float32 3.) (Expand (ints [ 4; 8 ]))
          in
          let expands u =
            List.filter_map
              (fun n -> if Ops.op n = Expand then Some (Ops.shape n) else None)
              (Ops.toposort u)
          in
          equal
            (list (list uop))
            [ List.map Ops.sint_to_uop (ints [ 2; 8 ]) ]
            (List.map (List.map Ops.sint_to_uop)
               (expands (multi Ops.O.(a + c)))));
      test "sources sharded differently are resharded on the result's axis"
        (fun () ->
          let across = sharded 2 [ 4; 4 ] 1 in
          let u = multi Ops.O.(a + across) in
          equal (list int) [ 1 ] (List.map fst (Ops.sharding u));
          is_true (has Allreduce u));
      test "a stack of values sharded alike stacks their shards" (fun () ->
          equal uop
            (unshard (Ops.stack [ shard a; shard b ]) [ 1 ] a)
            (multi (Ops.stack [ a; b ])));
      test "a stack of values sharded differently is resharded" (fun () ->
          let across = sharded 2 [ 4; 4 ] 1 in
          let memory =
            [
              (1, Array.init 32 (fun j -> `Float (float_of_int j)));
              (2, Array.init 32 (fun j -> `Float (float_of_int (100 + j))));
            ]
          in
          let stacked = Ops.stack [ a; across ] in
          let whole = List.hd (Tensors.eval ~buffers:memory stacked) in
          List.iter
            (fun v -> equal (array Dtypes.const) whole v)
            (Tensors.eval ~buffers:memory (multi stacked)));
      test "a source of lower rank sharded on the result's axis keeps its shard"
        (fun () ->
          let columns = sharded 1 [ 4; 4 ] 1 and row = sharded 2 [ 4 ] 0 in
          let memory =
            [
              (1, Array.init 32 (fun j -> `Float (float_of_int j)));
              (2, Array.init 8 (fun j -> `Float (float_of_int (100 * j))));
            ]
          in
          let sum = Ops.O.(columns + row) in
          let whole = List.hd (Tensors.eval ~buffers:memory sum) in
          List.iter
            (fun v -> equal (array Dtypes.const) whole v)
            (Tensors.eval ~buffers:memory (multi sum)));
      test "values on different devices are refused when resharded" (fun () ->
          let elsewhere =
            sharded ~devices:(Ops.Multi [ "CPU:2"; "CPU:3" ]) 2 [ 4; 4 ] 1
          in
          refused ~because:"devices" Ops.O.(a + elsewhere));
    ]

let reductions =
  let half = sharded ~dtype:Float16 4 [ 2; 8 ] 0 in
  let sum_up u = Ops.rop (Ops.cast u Float32) Add [ 0 ] in
  let cast_sum settings =
    Helpers.context
      [ Helpers.B (Helpers.allreduce_cast, settings) ]
      (fun () -> multi (sum_up half))
  in
  group "multi_pm › reductions"
    [
      test "a reduction of the sharded axis reduces each shard, then across"
        (fun () ->
          equal uop
            (Ops.allreduce (Ops.rop (shard a) Add [ 0 ]) Add two)
            (multi (Ops.rop a Add [ 0 ])));
      test "a reduction of other axes reduces each shard and stays sharded"
        (fun () ->
          equal uop
            (unshard (Ops.rop (shard a) Add [ 1 ]) [ 0 ] a)
            (multi (Ops.rop a Add [ 1 ])));
      test "a value cast up from a half crosses the devices as a half"
        (fun () ->
          let local = sum_up (shard half) in
          equal uop
            (Ops.cast (Ops.allreduce (Ops.cast local Float16) Add two) Float32)
            (cast_sum true));
      test "a value cast up from a bfloat16 crosses the devices as a bfloat16"
        (fun () ->
          let bf = sharded ~dtype:Bfloat16 4 [ 2; 8 ] 0 in
          let local = sum_up (shard bf) in
          equal uop
            (Ops.cast (Ops.allreduce (Ops.cast local Bfloat16) Add two) Float32)
            (Helpers.context
               [ Helpers.B (Helpers.allreduce_cast, true) ]
               (fun () -> multi (sum_up bf))));
      test "without allreduce_cast, it crosses in the type it is reduced in"
        (fun () ->
          equal uop
            (Ops.allreduce (sum_up (shard half)) Add two)
            (cast_sum false));
      test "a reduction of some sharded axes but not all is refused" (fun () ->
          let r = Ops.range ~axis_type:Device (Int 4) [ -1 ] in
          let grid =
            Ops.unshard
              ~ranges:Ops.O.[ r // int 2; r % int 2 ]
              (storage ~devices:four 1 [ 2; 4 ])
              [ 0; 1 ]
          in
          refused ~because:"several axes" (Ops.rop grid Add [ 0 ]));
      test "an allreduce of a sharded value reduces its shards" (fun () ->
          equal uop
            (unshard (Ops.allreduce (shard a) Add two) [ 0 ] a)
            (multi (Ops.allreduce a Add two)));
      test "an allreduce of a value that is not sharded is left for later"
        (fun () ->
          let red = Ops.allreduce whole Add two in
          equal uop red (multi red));
    ]

let movements =
  let over_four = sharded ~devices:four 1 [ 1; 8 ] 0 in
  group "multi_pm › movements"
    [
      test "a reshape keeps a sharded axis whole" (fun () ->
          equal uop
            (unshard (Ops.reshape (shard a) (ints [ 1; 2; 8 ])) [ 0 ] a)
            (multi (Ops.reshape a (ints [ 2; 2; 8 ]))));
      test "a reshape that moves elements between shards is refused" (fun () ->
          refused ~because:"between shards"
            (Ops.reshape (sharded 1 [ 4; 4 ] 1) (ints [ 32 ])));
      test "a reshape to an axis its shard count does not divide is refused"
        (fun () ->
          refused ~because:"between shards"
            (Ops.reshape over_four (ints [ 2; 16 ])));
      test "an expand moves the sharded axis behind the new axes" (fun () ->
          equal uop
            (unshard (Ops.mop (shard a) (Expand (ints [ 3 ]))) [ 1 ] a)
            (multi (Ops.mop a (Expand (ints [ 3 ])))));
      test "a permute moves the sharded axis" (fun () ->
          equal uop
            (unshard (Ops.permute (shard a) [ 1; 0 ]) [ 1 ] a)
            (multi (Ops.permute a [ 1; 0 ])));
      test "a pad of another axis pads each shard" (fun () ->
          equal uop
            (unshard
               (Ops.mop (shard a) (Pad [ (Int 0, Int 2); (Int 1, Int 11) ]))
               [ 0 ] a)
            (multi (Ops.pad a [ None; Some (Int 1, Int 2) ])));
      test "a pad after a sharded axis is refused" (fun () ->
          refused ~because:"pad" (Ops.pad a [ Some (Int 0, Int 2); None ]));
      test "a pad of a sharded axis is refused" (fun () ->
          refused ~because:"pad" (Ops.pad a [ Some (Int 1, Int 1); None ]));
      test "a flip of another axis flips each shard" (fun () ->
          equal uop
            (unshard (Ops.flip (shard a) [ 1 ]) [ 0 ] a)
            (multi (Ops.flip a [ 1 ])));
      test "a flip of a sharded axis is refused" (fun () ->
          refused ~because:"flip" (Ops.flip a [ 0 ]));
      test "a shrink of another axis shrinks each shard" (fun () ->
          equal uop
            (unshard
               (Ops.mop (shard a) (Shrink [ (Int 0, Int 2); (Int 2, Int 4) ]))
               [ 0 ] a)
            (multi (Ops.shrink a [ None; Some (Int 2, Int 6) ])));
      test "a shrink to one device's shard places that shard on every device"
        (fun () ->
          let memory =
            [ (1, Array.init 32 (fun j -> `Float (float_of_int j))) ]
          in
          let u = multi (Ops.shrink a [ Some (Int 2, Int 4); None ]) in
          let part = Array.init 16 (fun j -> `Float (float_of_int (16 + j))) in
          is_false (has Unshard u);
          equal
            (list (array Dtypes.const))
            [ part; part ]
            (Tensors.eval ~buffers:memory u));
      test "a shrink to one shard of a grid's axis is refused" (fun () ->
          let r = Ops.range ~axis_type:Device (Int 4) [ -1 ] in
          let grid =
            Ops.unshard
              ~ranges:Ops.O.[ r // int 2; r % int 2 ]
              (storage ~devices:four 1 [ 2; 4 ])
              [ 0; 1 ]
          in
          refused ~because:"shrink"
            (Ops.shrink grid [ Some (Int 0, Int 2); None ]));
      test "a shrink of part of a sharded axis is refused" (fun () ->
          refused ~because:"shrink" (Ops.shrink a [ Some (Int 1, Int 3); None ]));
    ]

(* Values sharded across the threads of a workgroup: register storage of eight
   by eight, each of eight threads holding eight rows of sixty-four. *)

let threads = Ops.range ~axis_type:Local (Int 8) [ 0 ]
let rows = Ops.range ~axis_type:Loop (Int 8) [ 1 ]
let cols = Ops.range ~axis_type:Loop (Int 8) [ 2 ]
let registers = Ops.placeholder ~addrspace:Reg [ 8; 8 ] Float32
let fragment = Ops.unshard ~ranges:[ threads ] registers [ 0 ]

let fragments =
  group "multi_pm › fragments"
    [
      test "an index into a thread's block of rows is an index into its shard"
        (fun () ->
          equal uop
            (Ops.index registers [ rows; cols ])
            (multi
               (Ops.index fragment Ops.O.[ (threads * int 8) + rows; cols ])));
      test "an index into every eighth row is an index into its shard"
        (fun () ->
          equal uop
            (Ops.index registers [ rows; cols ])
            (multi
               (Ops.index fragment Ops.O.[ threads + (rows * int 8); cols ])));
      test "an index into rows another thread holds is refused" (fun () ->
          refused ~because:"crosses"
            (Ops.index fragment Ops.O.[ threads + rows; cols ]));
      test "an index one past a thread's every eighth row is refused" (fun () ->
          refused ~because:"crosses"
            (Ops.index fragment
               Ops.O.[ threads + (rows * int 8) + int 1; cols ]));
      test "an index into every sixteenth row is refused" (fun () ->
          refused ~because:"crosses"
            (Ops.index fragment Ops.O.[ threads + (rows * int 16); cols ]));
      test "a shrink to one shard of another thread's rows is refused"
        (fun () ->
          refused ~because:"shrink"
            (Ops.shrink fragment [ Some (Int 8, Int 16); None ]));
      test "a shrink to a thread's own shard removes its sharding" (fun () ->
          let own = Ops.O.(threads * int 8) in
          equal uop
            (Ops.mop registers (Shrink [ (Int 0, Int 8); (Int 0, Int 8) ]))
            (multi
               (Ops.mop fragment (Shrink [ (Sym own, Int 8); (Int 0, Int 8) ]))));
    ]

(* A tile of four by twelve, sharded across two by three threads: register
   storage of two by four per thread. *)

let tile_rows = Ops.range ~axis_type:Local (Int 2) [ 0 ]
let tile_cols = Ops.range ~axis_type:Local (Int 3) [ 1 ]
let tile_part = Ops.placeholder ~addrspace:Reg [ 2; 4 ] Float32
let tile = Ops.unshard ~ranges:[ tile_rows; tile_cols ] tile_part [ 0; 1 ]
let sharding = list (pair int uop)

let tiles =
  group "multi_pm › two sharded axes"
    [
      test "a reshape divides each sharded axis by its own count (D21)"
        (fun () ->
          equal uop
            (Ops.unshard ~ranges:[ tile_rows; tile_cols ]
               (Ops.reshape tile_part (ints [ 2; 1; 4 ]))
               [ 0; 1 ])
            (multi (Ops.reshape tile (ints [ 4; 3; 4 ]))));
      test "a reshape of a mesh of 2 by 4 devices keeps each tile (D21)"
        (fun () ->
          let eight = Ops.Multi (List.init 8 (Printf.sprintf "CPU:%d")) in
          let r = Ops.range ~axis_type:Device (Int 8) [ -1 ] in
          let mesh =
            Ops.unshard
              ~ranges:Ops.O.[ r // int 4; r % int 4 ]
              (storage ~devices:eight ~dtype:Int32 1 [ 2; 3 ])
              [ 0; 1 ]
          in
          let v = Ops.O.(Ops.reshape mesh (ints [ 4; 12; 1 ]) * int 2) in
          let memory = [ (1, Array.init 48 (fun j -> `Int (Z.of_int j))) ] in
          let whole = List.hd (Tensors.eval ~buffers:memory v) in
          List.iter
            (fun got -> equal (array Dtypes.const) whole got)
            (Tensors.eval ~buffers:memory (multi v)));
      test "an operation with a whole value takes its tile of it" (fun () ->
          let whole = Ops.placeholder ~slot:1 [ 4; 12 ] Float32 in
          let part =
            Ops.shard_slice (Ops.shard_slice whole 0 tile_rows) 1 tile_cols
          in
          equal uop
            (Ops.unshard ~ranges:[ tile_rows; tile_cols ]
               Ops.O.(tile_part + part)
               [ 0; 1 ])
            (multi Ops.O.(tile + whole)));
      test "a permute keeps each range with its axis" (fun () ->
          let u = multi (Ops.permute tile [ 1; 0 ]) in
          equal sharding [ (0, tile_cols); (1, tile_rows) ] (Ops.sharding u);
          equal uop (Ops.permute tile_part [ 1; 0 ]) (Ops.nth u 0));
      test "a shrink to one axis's own shard keeps the other's sharding"
        (fun () ->
          let own = Ops.O.(tile_rows * int 2) in
          equal uop
            (Ops.unshard ~ranges:[ tile_cols ]
               (Ops.mop tile_part (Shrink [ (Int 0, Int 2); (Int 0, Int 4) ]))
               [ 1 ])
            (multi
               (Ops.mop tile (Shrink [ (Sym own, Int 2); (Int 0, Int 12) ]))));
    ]

let device = Testable.make ~pp:Ops.pp_device ~equal:Ops.equal_device

(* Replicated storage without movements, which a selection would move past. *)
let flat = Ops.new_buffer ~slot:3 two 32 Float32

let copies =
  let one = storage ~devices:(cpu 0) 1 [ 4; 8 ] in
  group "multi_pm › copies"
    [
      test "a copy of a sharded value to one device joins its shards" (fun () ->
          let u = multi (Ops.copy_to_device a (cpu 0)) in
          is_false (has Unshard u);
          equal (option device) (Some (cpu 0)) (Ops.device u));
      test "a copy of a sharded value to several devices sums its placed shards"
        (fun () ->
          let u = multi (Ops.copy_to_device a four) in
          equal (list device) [ four ]
            (List.filter_map
               (fun n ->
                 match Ops.arg n with
                 | Allreduce { device; _ } -> Some device
                 | _ -> None)
               (Ops.toposort u)));
      test "a copy of a value on one device to several is a copy to each"
        (fun () ->
          equal uop
            (Ops.mstack
               (Ops.copy_to_device one (cpu 0))
               [ Ops.copy_to_device one (cpu 1) ])
            (multi (Ops.copy_to_device one two)));
      test "a copy of a value that simplifies to no device is it on each"
        (fun () ->
          let zeros =
            Ops.O.(
              Ops.new_buffer ~slot:1 (cpu 0) 4 Int32 * Ops.int ~dtype:Int32 0)
          in
          let s = Ops.simplify zeros in
          equal (option device) None (Ops.device s);
          equal uop (Ops.mstack s [ s ]) (multi (Ops.copy_to_device zeros two)));
      test "a copy of a replicated value to one device copies its first"
        (fun () ->
          equal uop
            (Ops.copy_to_device (Ops.mselect flat 0) (cpu 1))
            (multi (Ops.copy_to_device flat (cpu 1))));
      test "a copy of a replicated value to its first device is its first"
        (fun () ->
          equal uop (Ops.mselect flat 0)
            (multi (Ops.copy_to_device flat (cpu 0))));
    ]

let selections =
  let first = storage ~devices:(cpu 0) 1 [ 4 ] in
  let second = storage ~devices:(cpu 1) 2 [ 4 ] in
  group "multi_pm › shard selections"
    [
      test "a selection of a gather is its source" (fun () ->
          equal uop second (multi (Ops.mselect (Ops.mstack first [ second ]) 1)));
      test "a selection of a movement moves the selection" (fun () ->
          equal uop
            (Ops.reshape (Ops.mselect flat 1) (ints [ 4; 8 ]))
            (multi (Ops.mselect (Ops.reshape flat (ints [ 4; 8 ])) 1)));
      test
        "a selection of a movement by the device range takes the selected \
         device's position (D23)" (fun () ->
          let d = Ops.range ~axis_type:Device (Int 2) [ -1 ] in
          let at start = Ops.mop flat (Shrink [ (start, Int 16) ]) in
          equal uop
            (Ops.mop (Ops.mselect flat 1) (Shrink [ (Int 16, Int 16) ]))
            (multi (Ops.mselect (at (Sym Ops.O.(d * int 16))) 1)));
      test "a selection of an operation selects its sources on several devices"
        (fun () ->
          equal uop
            Ops.O.(Ops.mselect flat 1 + float 1.)
            (multi (Ops.mselect Ops.O.(flat + float 1.) 1)));
      test
        "a shrink of a gather shrinks each source at its device, materialised \
         there" (fun () ->
          let d = Ops.range ~axis_type:Device (Int 2) [ -1 ] in
          let moved = Ops.copy_to_device first (cpu 1) in
          let at k u = Ops.mop u (Shrink [ (Int k, Int 2) ]) in
          equal uop
            (Ops.mstack
               (Ops.contiguous (at 0 first))
               [ Ops.copy_to_device (at 1 first) (cpu 1) ])
            (multi
               (Ops.mop
                  (Ops.mstack first [ moved ])
                  (Shrink [ (Sym d, Int 2) ]))));
    ]

let effects =
  let dest = sharded 5 [ 2; 8 ] 0 in
  group "multi_pm › stores and calls"
    [
      test "a store of a value sharded as its destination stores each shard"
        (fun () ->
          let own =
            Ops.mop (shard dest) (Shrink [ (Int 0, Int 2); (Int 0, Int 8) ])
          in
          equal uop
            (unshard
               (Ops.after (shard dest) [ Ops.store own (shard a) ])
               [ 0 ] a)
            (multi (Ops.after dest [ Ops.store dest a ])));
      test "a store of a sharded value into a whole one stores into its part"
        (fun () ->
          equal uop
            (Ops.store (Ops.shard_slice whole 0 (range a)) (shard a))
            (multi (Ops.store whole a)));
      test "a store of a whole value into a sharded destination stores its part"
        (fun () ->
          equal uop
            (Ops.store (shard dest) (Ops.shard_slice whole 0 (range a)))
            (multi (Ops.store dest whole)));
      test "a gated store into a sharded destination keeps a scalar gate"
        (fun () ->
          let gate = Ops.bool true in
          equal uop
            (Ops.store ~gate (shard dest) (Ops.shard_slice whole 0 (range a)))
            (multi (Ops.store ~gate dest whole)));
      test "a call of a compiled function passes its arguments' shards"
        (fun () ->
          let body =
            Ops.sink
              [
                Ops.store
                  (Ops.param ~shape:[ Int 16 ] 0 Float32)
                  (Ops.param ~shape:[ Int 16 ] 1 Float32);
              ]
          in
          equal uop
            (Ops.call ~precompile:true body [ shard dest; shard a ])
            (multi (Ops.call ~precompile:true body [ dest; a ])));
    ]

let passthrough =
  let through name wrap =
    test (name ^ " passes the shard through") (fun () ->
        equal uop (unshard (wrap (shard a)) [ 0 ] a) (multi (wrap a)))
  in
  group "multi_pm › passthrough"
    [
      through "a cast" (fun u -> Ops.cast u Float16);
      through "a bitcast" (fun u -> Ops.bitcast u Int32);
      through "a stage" (fun u -> Ops.v Stage ~src:[ u ]);
      through "a detach" (fun u -> Ops.v Detach ~src:[ u ]);
      through "a contiguous_backward" (fun u ->
          Ops.v Contiguous_backward ~src:[ u ]);
      through "an after" (fun u ->
          Ops.after u [ Ops.store (storage 6 [ 4 ]) (storage 7 [ 4 ]) ]);
    ]

let () =
  exit
    (run "Tolk_next.Multi"
       [
         recorded;
         values;
         laws;
         arithmetic;
         reductions;
         movements;
         fragments;
         tiles;
         copies;
         selections;
         effects;
         passthrough;
       ])
