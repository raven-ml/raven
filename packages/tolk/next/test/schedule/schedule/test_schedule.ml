(* Tests of Tolk_next.Schedule: the schedules tinygrad makes of the programs it
   realizes and the order of their kernels, the laws that a schedule writes what
   its program computes, the schedule cache, and the rules of the module. *)

open Windtrap
open Tolk_next

let uop = Uops.uop
let program name = Golden.sink (name ^ ".golden")

(* Recorded graphs *)

let programs =
  [
    "add";
    "assign";
    "assign_bitcast";
    "assign_double_diamond";
    "assign_permuted_self";
    "attention";
    "cat";
    "chained_functions";
    "clone";
    "contiguous";
    "conv";
    "copy";
    "copy_computed";
    "copy_one";
    "copy_view";
    "custom_kernel";
    "disk_store";
    "disk_to";
    "disk_view_to";
    "double_matmul";
    "embedding";
    "full_invalid";
    "inline_function";
    "layernorm";
    "matmul";
    "mesh_sum";
    "precompiled_function";
    "precompiled_scalar";
    "read_then_overwrite";
    "reduce_multiple_paths";
    "setitem";
    "shard_add";
    "shard_matmul";
    "shard_sum";
    "shard_to_one";
    "softmax";
    "sort";
    "split_sum";
    "sum";
    "two_outputs";
    "variable_offset";
    "variable_reduce";
    "variable_same";
    "variable_shrink";
    "variable_two";
    "variable_unused";
  ]

let kernel_graphs =
  [
    "add_kernels";
    "assign_bitcast_kernels";
    "assign_double_diamond_kernels";
    "assign_kernels";
    "assign_permuted_self_kernels";
    "attention_kernels";
    "cat_kernels";
    "chained_functions_kernels";
    "chained_functions_kernels_1";
    "clone_kernels";
    "contiguous_kernels";
    "conv_kernels";
    "copy_computed_kernels";
    "copy_kernels";
    "copy_one_kernels";
    "copy_view_kernels";
    "custom_kernel_kernels";
    "disk_store_kernels";
    "disk_store_kernels_1";
    "disk_to_kernels";
    "disk_view_to_kernels";
    "double_matmul_kernels";
    "embedding_kernels";
    "full_invalid_kernels";
    "inline_function_kernels";
    "layernorm_kernels";
    "matmul_kernels";
    "mesh_sum_kernels";
    "mesh_sum_kernels_1";
    "precompiled_function_kernels";
    "precompiled_function_kernels_1";
    "precompiled_scalar_kernels";
    "precompiled_scalar_kernels_1";
    "read_then_overwrite_kernels";
    "reduce_multiple_paths_kernels";
    "setitem_kernels";
    "shard_add_kernels";
    "shard_matmul_kernels";
    "shard_sum_kernels";
    "shard_sum_kernels_1";
    "shard_to_one_kernels";
    "softmax_kernels";
    "sort_kernels";
    "split_sum_kernels";
    "sum_kernels";
    "two_outputs_kernels";
    "variable_offset_kernels";
    "variable_reduce_kernels";
    "variable_same_kernels";
    "variable_shrink_kernels";
    "variable_two_kernels";
    "variable_unused_kernels";
  ]

(* The kernel graph [<p>_kernels<n>] of the program [<p>] is recorded with the
   schedule [<p>_schedule<n>]. [split name] is [(<p>, <n>)]. *)
let split name =
  let k = "_kernels" in
  let rec at i =
    if String.sub name i (String.length k) = k then i else at (i - 1)
  in
  let i = at (String.length name - String.length k) in
  let rest = i + String.length k in
  (String.sub name 0 i, String.sub name rest (String.length name - rest))

let schedule_of name =
  let program, n = split name in
  program ^ "_schedule" ^ n

let linears =
  group "create_linear_with_vars › recorded"
    (List.map
       (fun name ->
         let file = name ^ "_linear.golden" in
         Golden.graph file (fun () ->
             Uops.numbered_like (Golden.sink file)
               (fst (Schedule.create_linear_with_vars (program name)))))
       programs)

let var_vals =
  Golden.cases "var_vals.golden" (fun cell ->
      let _, vals =
        Schedule.create_linear_with_vars (program (cell "program"))
      in
      let written =
        String.concat " "
          (List.map
             (fun (n, v) -> Printf.sprintf "%s=%d" n v)
             (List.sort compare vals))
      in
      equal string (cell "var_vals") (if written = "" then "-" else written))

let schedules =
  group "create_schedule › recorded"
    (List.map
       (fun name ->
         Golden.graph
           (schedule_of name ^ ".golden")
           (fun () -> Schedule.create_schedule (program name)))
       kernel_graphs)

(* Values

   The law that a schedule runs its kernels in an order that keeps what they
   write: running the calls of create_schedule's schedule in order
   (Kernel_graphs.linear_writes) writes what the kernel graph does, each kernel
   reading its arguments in the states they name (Kernel_graphs.writes). And end
   to end: the schedule of a program, unplanned, writes into the program's
   buffers what its tensors compute (Tensors). Memory holds small integers. *)

let element dtype k : Dtype.value =
  if Dtype.is_float dtype then `Float (float_of_int k)
  else if Dtype.equal dtype Bool then `Bool (k > 0)
  else if Dtype.is_unsigned dtype then `Int (Z.of_int (k + 3))
  else `Int (Z.of_int k)

let given u =
  List.filter_map
    (fun n ->
      match (Ops.op n, Ops.arg n) with
      | (Param | Buffer), Param p when p.addrspace <> Some Alu -> Some (n, p)
      | _ -> None)
    (Ops.toposort ~enter_calls:false u)

let filled u =
  List.map
    (fun (_, (p : Ops.param_arg)) ->
      let devices =
        match p.device with Some (Multi ds) -> List.length ds | _ -> 1
      in
      let size = devices * Option.value p.size ~default:1 in
      let at j = element p.dtype ((((j * 7) + (p.slot * 3)) mod 11) - 3) in
      (p.slot, Array.init size at))
    (given u)

let close = Testable.float_rel ~rel:1e-6 ~abs:0.

let value =
  Testable.make ~pp:(Testable.pp Dtypes.value) ~equal:(fun v0 v1 ->
      match (v0, v1) with
      | `Float x, `Float y ->
          (Float.is_nan x && Float.is_nan y) || Testable.equal close x y
      | v0, v1 -> Testable.equal Dtypes.value v0 v1)

let write = triple int int value

let on_several_devices u =
  List.exists
    (fun n -> match Ops.device n with Some (Multi _) -> true | _ -> false)
    (Ops.toposort u)

let calls_only_kernels u =
  List.for_all
    (fun n -> Ops.op n <> Call || Ops.op (Ops.nth n 0) = Sink)
    (Ops.toposort ~enter_calls:false u)

(* Custom kernels keep their views until they are compiled, which the
   Interpreter does not run. The call of assign_bitcast passes a buffer and a
   bitcast of it as two arguments, one memory that storage, a memory per slot,
   cannot alias. *)
let apart_from_values =
  [
    "custom_kernel_kernels";
    "custom_kernel";
    "assign_bitcast_kernels";
    "assign_bitcast";
  ]

(* The pairs [k=v] recorded in [file] for the row whose first column [by] is
   [name], [k] read by [key] and [v] an integer. *)
let recorded key file by name column =
  let row = List.find (fun cell -> cell by = name) (Golden.rows file) in
  match row column with
  | "-" -> []
  | pairs ->
      List.map
        (fun kv ->
          match String.split_on_char '=' kv with
          | [ k; v ] -> (key k, `Int (Z.of_string v))
          | _ -> failf "a recorded pair %s" kv)
        (String.split_on_char ' ' pairs)

let vars_of program =
  recorded Fun.id "var_vals.golden" "program" program "var_vals"

(* A kernel graph is the body of a call, whose scalar arguments it reads as
   parameters. *)
let runs_in_order name =
  test (name ^ ": the schedule writes what the kernel graph does") (fun () ->
      let kg = program name in
      let buffers = filled kg
      and vars = vars_of (fst (split name))
      and params =
        recorded int_of_string "arguments.golden" "graph" name "arguments"
      in
      equal (list write)
        (Kernel_graphs.writes ~vars ~params ~buffers kg)
        (Kernel_graphs.linear_writes ~vars ~params ~buffers
           (Schedule.create_schedule kg)))

let ordered =
  group "create_schedule › values"
    (List.filter_map
       (fun name ->
         let kg = program name in
         if
           List.mem name apart_from_values
           || on_several_devices kg
           || not (calls_only_kernels kg)
         then None
         else Some (runs_in_order name))
       kernel_graphs)

let unplanned big = fst (Schedule.create_linear_with_vars ~capturing:true big)

let computes big =
  let buffers = filled big in
  let vars =
    List.map
      (fun (n, v) -> (n, `Int (Z.of_int v)))
      (snd (Schedule.create_linear_with_vars big))
  in
  let slots = List.map (fun (_, (p : Ops.param_arg)) -> p.slot) (given big) in
  let into_given = List.filter (fun (s, _, _) -> List.mem s slots) in
  equal (list write)
    (Tensors.writes ~buffers big)
    (into_given (Kernel_graphs.linear_writes ~vars ~buffers (unplanned big)))

(* Tensors runs no movement that a variable reaches. *)
let has_variables u =
  List.exists
    (fun n ->
      match (Ops.op n, Ops.arg n) with
      | Param, Param { addrspace = Some Alu; name = Some _; _ } -> true
      | _ -> false)
    (Ops.toposort u)

let realized =
  group "create_linear_with_vars › values"
    (List.filter_map
       (fun name ->
         let big = program name in
         if
           List.mem name apart_from_values
           || on_several_devices big || has_variables big
           || not (calls_only_kernels (unplanned big))
         then None
         else
           Some
             (test (name ^ " writes what its tensors compute") (fun () ->
                  computes big)))
       programs)

(* Planning: under capturing, the schedule is left unplanned for the jit, which
   plans it as create_linear_with_vars does, holding the program's buffers. *)

let buffers_of big =
  List.filter (fun n -> Ops.op n = Buffer) (Ops.toposort ~enter_calls:false big)

let planned_later name =
  test (name ^ ": planned later, the captured schedule is the planned one")
    (fun () ->
      let big = program name in
      let planned = fst (Schedule.create_linear_with_vars big) in
      let captured =
        Memory.memory_plan_rewrite ~held_bufs:(buffers_of big) (unplanned big)
      in
      equal uop planned (Uops.numbered_like planned captured))

let capturing =
  group "create_linear_with_vars › capturing" (List.map planned_later programs)

let checked () =
  List.iter
    (fun name ->
      let big = program name in
      let plain = fst (Schedule.create_linear_with_vars ~capturing:true big) in
      let under_spec =
        Helpers.context
          [ B (Helpers.spec, 1) ]
          (fun () -> fst (Schedule.create_linear_with_vars ~capturing:true big))
      in
      equal uop ~msg:name plain (Uops.numbered_like plain under_spec))
    programs

let spec =
  group "create_linear_with_vars › spec"
    [
      test "with SPEC=1, every recorded program passes and schedules alike"
        checked;
    ]

(* The schedule cache

   A body is keyed by its structure: its buffers are parameters, so only a
   program's computation tells two bodies apart. [unique ~constant name] is the
   program [name] with its [constant] (default 1.0), in call bodies too,
   replaced by one no earlier call made, so that its bodies are new to the cache
   whatever ran before. *)

let constants = Atomic.make 0

let unique ?(constant = 1.0) name =
  let x = 1000. +. float_of_int (Atomic.fetch_and_add constants 1) in
  Ops.substitute ~enter_calls:true (program name)
    [ (Ops.float constant, Ops.float x) ]

let with_settings ~debug ~scache f =
  Helpers.context [ B (Helpers.debug, debug); B (Helpers.scache, scache) ] f

(* The lines scheduling printed, among what else it printed: each [(n, ms,
   verdict, key)] of a line [scheduled %5d kernels in %8.2f ms | <verdict>
   <key>]. *)
let timed_reports () =
  let report line =
    match Str.bounded_split_delim (Str.regexp_string " | ") line 2 with
    | [ head; tail ] ->
        let n, ms =
          Scanf.sscanf head "scheduled %d kernels in %f ms%!" (fun n ms ->
              (n, ms))
        in
        equal string
          (Printf.sprintf "scheduled %5d kernels in %8.2f ms" n ms)
          head;
        let k = String.length tail - 8 in
        equal string " " (String.sub tail (k - 1) 1);
        (n, ms, String.sub tail 0 (k - 1), String.sub tail k 8)
    | _ -> failf "a line of scheduling is %S" line
  in
  List.map report
    (List.filter
       (String.starts_with ~prefix:"scheduled ")
       (String.split_on_char '\n' (output ())))

let reports () =
  List.map (fun (n, _, verdict, key) -> (n, verdict, key)) (timed_reports ())

let report = triple int string string

let schedule_twice ~scache big =
  with_settings ~debug:3 ~scache (fun () ->
      ignore (Schedule.create_linear_with_vars big);
      ignore (Schedule.create_linear_with_vars big));
  reports ()

let miss_then_hit () =
  match schedule_twice ~scache:1 (unique "assign") with
  | [ (_, _, key); _ ] as lines ->
      is_true ~msg:"the key is 8 hex digits"
        (String.for_all
           (function '0' .. '9' | 'a' .. 'f' -> true | _ -> false)
           key);
      equal (list report)
        [ (1, "CACHE MISS", key); (1, " cache hit", key) ]
        lines
  | lines -> failf "scheduling twice printed %d lines" (List.length lines)

let hit_is_the_miss () =
  let big = unique "contiguous" in
  let miss = fst (Schedule.create_linear_with_vars big) in
  let hit = fst (Schedule.create_linear_with_vars big) in
  equal uop miss (Uops.numbered_like miss hit)

let uncached () =
  match schedule_twice ~scache:0 (unique "assign") with
  | [ (_, _, key); _ ] as lines ->
      equal (list report)
        [ (1, "CACHE MISS", key); (1, "CACHE MISS", key) ]
        lines
  | lines -> failf "scheduling twice printed %d lines" (List.length lines)

let keys_differ () =
  with_settings ~debug:3 ~scache:1 (fun () ->
      ignore (Schedule.create_linear_with_vars (unique "assign"));
      ignore (Schedule.create_linear_with_vars (unique "assign")));
  match reports () with
  | [ (_, v0, k0); (_, v1, k1) ] ->
      equal (list string) [ "CACHE MISS"; "CACHE MISS" ] [ v0; v1 ];
      not_equal string k0 k1
  | lines -> failf "scheduling two bodies printed %d lines" (List.length lines)

let debug_one () =
  with_settings ~debug:1 ~scache:1 (fun () ->
      ignore (Schedule.create_linear_with_vars (unique "assign"));
      ignore (Schedule.create_linear_with_vars (unique "contiguous")));
  match reports () with
  | [ (n, verdict, _) ] -> equal (pair int string) (2, "CACHE MISS") (n, verdict)
  | lines -> failf "scheduling printed %d lines" (List.length lines)

let quiet () =
  with_settings ~debug:0 ~scache:1 (fun () ->
      ignore (Schedule.create_linear_with_vars (unique "contiguous")));
  equal string "" (output ())

let timed () =
  let big = unique "contiguous" in
  let start = Unix.gettimeofday () in
  with_settings ~debug:1 ~scache:1 (fun () ->
      ignore (Schedule.create_linear_with_vars big));
  let elapsed = (Unix.gettimeofday () -. start) *. 1000. in
  match timed_reports () with
  | [ (_, ms, _, _) ] ->
      at_least float_exact ~than:0. ms;
      (* the time is printed to a hundredth *)
      at_most float_exact ~than:(elapsed +. 0.005) ms
  | lines -> failf "scheduling printed %d lines" (List.length lines)

(* The slots of call-local storage are no part of a body's key. *)
let renumbered () =
  let big = unique "precompiled_function" in
  let alloc =
    List.find (fun n -> Ops.op n = Alloc) (Ops.toposort ~enter_calls:false big)
  in
  let in_slot slot =
    match Ops.arg alloc with
    | Param p ->
        Ops.substitute big
          [ (alloc, Ops.replace ~arg:(Param { p with slot }) alloc) ]
    | _ -> fail "call-local storage without its argument"
  in
  let verdicts big =
    with_settings ~debug:3 ~scache:1 (fun () ->
        ignore (Schedule.create_linear_with_vars big));
    List.map (fun (_, verdict, _) -> verdict) (reports ())
  in
  is_true ~msg:"the first is new" (List.mem "CACHE MISS" (verdicts (in_slot 0)));
  equal (list string) [ " cache hit"; " cache hit" ] (verdicts (in_slot 9))

(* tinygrad prints, for chained_functions, the function's body missing and then
   hitting twice, and the program's body missing. *)
let chained () =
  with_settings ~debug:3 ~scache:1 (fun () ->
      ignore (Schedule.create_linear_with_vars (unique "chained_functions")));
  match reports () with
  | [ (_, _, f); _; _; (_, _, program) ] as lines ->
      not_equal string f program;
      equal (list report)
        [
          (3, "CACHE MISS", f);
          (3, " cache hit", f);
          (3, " cache hit", f);
          (3, "CACHE MISS", program);
        ]
        lines
  | lines -> failf "scheduling printed %d lines" (List.length lines)

(* variable_shrink binds v to 3 and multiplies by 2.0. *)
let rebinding () =
  let big = unique ~constant:2.0 "variable_shrink" in
  let v = List.find Ops.is_bound_var (Ops.toposort big) in
  let five =
    Ops.substitute big [ (v, Ops.bind (Ops.unbound v) (`Int (Z.of_int 5))) ]
  in
  let vals =
    with_settings ~debug:3 ~scache:1 (fun () ->
        ignore (Schedule.create_linear_with_vars big);
        snd (Schedule.create_linear_with_vars five))
  in
  equal (list (pair string int)) [ ("v", 5) ] vals;
  equal (list string)
    [ "CACHE MISS"; " cache hit" ]
    (List.map (fun (_, verdict, _) -> verdict) (reports ()))

let in_domains bigs =
  List.map Domain.join
    (List.map
       (fun big ->
         Domain.spawn (fun () -> fst (Schedule.create_linear_with_vars big)))
       bigs)

let one_body_at_once () =
  let big = unique "softmax" in
  let results = in_domains (List.init 4 (fun _ -> big)) in
  let first = List.hd results in
  List.iteri
    (fun k r ->
      equal uop
        ~msg:(Printf.sprintf "domain %d" k)
        first
        (Uops.numbered_like first r))
    results;
  let held = Ops.backward_slice_with_self big in
  let made =
    List.concat_map
      (fun r ->
        List.filter_map
          (fun n ->
            match Ops.arg n with
            | Param p when Ops.op n = Buffer && not (Ops.Nodes.mem n held) ->
                Some p.slot
            | _ -> None)
          (Ops.toposort r))
      results
  in
  at_least ~msg:"each domain makes buffers" int ~than:(List.length results)
    (List.length made);
  equal ~msg:"each domain makes buffers of its own" int (List.length made)
    (List.length (List.sort_uniq compare made));
  with_settings ~debug:3 ~scache:1 (fun () ->
      ignore (Schedule.create_linear_with_vars big));
  match reports () with
  | [ (_, verdict, _) ] -> equal string " cache hit" verdict
  | lines -> failf "scheduling printed %d lines" (List.length lines)

let bodies_at_once () =
  let bigs = List.init 4 (fun _ -> unique "softmax") in
  List.iteri
    (fun k (big, r) ->
      let alone =
        with_settings ~debug:0 ~scache:0 (fun () ->
            fst (Schedule.create_linear_with_vars big))
      in
      equal uop
        ~msg:(Printf.sprintf "domain %d" k)
        alone
        (Uops.numbered_like alone r))
    (List.combine bigs (in_domains bigs))

let cache =
  group "create_linear_with_vars › cache"
    [
      test "a body scheduled again is a hit under the key it missed on"
        miss_then_hit;
      test "a hit is the schedule the miss made" hit_is_the_miss;
      test "with SCACHE=0 every schedule misses" uncached;
      test "two bodies miss under two keys" keys_differ;
      test "DEBUG=1 prints only schedules of several kernels" debug_one;
      test "DEBUG=0 prints nothing" quiet;
      test "the time printed is the time scheduling took" timed;
      test "call-local storage in other slots is the same body" renumbered;
      test "a function called three times is scheduled once" chained;
      test "a variable bound to another value is the same body" rebinding;
      test "domains scheduling one body at once schedule it alike"
        one_body_at_once;
      test "domains scheduling bodies at once schedule each as alone"
        bodies_at_once;
    ]

(* Rules *)

(* The kernel graph of clone: a kernel that copies its argument 1 into its
   argument 0. *)
let clone = program "clone_kernels"

let copy_call =
  List.find (fun n -> Ops.op n = Call) (Ops.toposort ~enter_calls:false clone)

let copy_kernel = Ops.nth copy_call 0

let x, y =
  match Ops.src copy_call with
  | [ _; x; y ] -> (x, y)
  | _ -> failwith "clone's call takes two arguments"

let copies ~into ~from = Ops.replace ~src:[ copy_kernel; into; from ] copy_call

let swap () =
  (* each kernel overwrites what the other reads as it was given *)
  let a = copies ~into:x ~from:y and b = copies ~into:y ~from:x in
  let kg = Ops.sink [ Ops.after x [ a ]; Ops.after y [ b ] ] in
  raises_match (Exn.invalid_arg ~substring:"cycle") (fun () ->
      Schedule.create_schedule kg)

let chain () =
  (* the second kernel reads what the first wrote, and overwrites what the first
     read *)
  let a = copies ~into:x ~from:y in
  let b = copies ~into:y ~from:(Ops.after x [ a ]) in
  let kg = Ops.sink [ Ops.after x [ a ]; Ops.after y [ b ] ] in
  equal (list uop)
    [ a; copies ~into:y ~from:x ]
    (Ops.src (Schedule.create_schedule kg))

let overwrite () =
  (* the second kernel writes the state the first left *)
  let z =
    match Ops.arg y with
    | Param p -> Ops.replace ~arg:(Param { p with slot = 2 }) y
    | _ -> fail "a parameter without its argument"
  in
  let a = copies ~into:x ~from:y in
  let b = copies ~into:(Ops.after x [ a ]) ~from:z in
  let kg = Ops.sink [ Ops.after (Ops.after x [ a ]) [ b ] ] in
  equal (list uop)
    [ a; copies ~into:x ~from:z ]
    (Ops.src (Schedule.create_schedule kg))

(* [b] reads x both before and after [a] overwrites it: it runs after [a]. *)
let read_across () =
  let a = copies ~into:x ~from:y in
  let z =
    match Ops.arg y with
    | Param p -> Ops.replace ~arg:(Param { p with slot = 2 }) y
    | _ -> fail "a parameter without its argument"
  in
  let b = Ops.replace ~src:[ copy_kernel; z; Ops.after x [ a ]; x ] copy_call in
  let kg = Ops.sink [ Ops.after z [ b ] ] in
  equal (list uop)
    [ a; Ops.replace ~src:[ copy_kernel; z; x; x ] copy_call ]
    (Ops.src (Schedule.create_schedule kg))

let not_an_effect () =
  let kg = Ops.sink [ Ops.after x [ Ops.float 2.0 ] ] in
  let const = Format.asprintf "%a" Op.pp Const in
  raises_match (Exn.invalid_arg ~substring:const) (fun () ->
      Schedule.create_schedule kg)

let not_storage () =
  let kg = Ops.sink [ Ops.after x [ copies ~into:x ~from:(Ops.float 2.0) ] ] in
  raises_match (Exn.invalid_arg ~substring:"buffer state") (fun () ->
      Schedule.create_schedule kg)

let loop = Ops.range ~axis_type:Loop (Int 4) [ 0 ]

let ended () =
  let a = copies ~into:x ~from:y in
  let kg = Ops.sink [ Ops.after x [ Ops.end_ a [ loop ] ] ] in
  equal (list uop) [ a ] (Ops.src (Schedule.create_schedule kg))

let ended_store () =
  let kg =
    Ops.sink [ Ops.after x [ Ops.end_ (Ops.store x (Ops.float 2.0)) [ loop ] ] ]
  in
  raises_match (Exn.invalid_arg ?substring:None) (fun () ->
      Schedule.create_schedule kg)

let bound_argument () =
  let n =
    Ops.bind
      (Ops.variable "n" (`Int (Z.of_int 1)) (`Int (Z.of_int 8)))
      (`Int (Z.of_int 3))
  in
  let a = Ops.replace ~src:[ copy_kernel; x; y; n ] copy_call in
  let kg = Ops.sink [ Ops.after x [ a ] ] in
  equal (list uop)
    [ copies ~into:x ~from:y ]
    (Ops.src (Schedule.create_schedule kg))

let ordering =
  group "create_schedule › rules"
    [
      test "a kernel's argument that is no storage is refused" not_storage;
      test "an end of a call schedules the call" ended;
      test "a kernel reading storage before and after a write runs after it"
        read_across;
      test "a kernel overwriting a state runs after the kernel that made it"
        overwrite;
      test "an end of something other than a call is refused" ended_store;
      test "a bound variable among a kernel's arguments leaves its call"
        bound_argument;
      test "kernels that overwrite each other's inputs are a cycle" swap;
      test "a kernel runs after the one it reads and before its overwriter"
        chain;
      test "an effect that is not a call, end, store or After is refused"
        not_an_effect;
    ]

let flattening () =
  let linear = Schedule.create_schedule (program "softmax_kernels") in
  match Ops.src linear with
  | [ c0; c1; c2 ] ->
      let nested =
        Ops.replace
          ~src:
            [
              Ops.replace ~src:[ c0; Ops.replace ~src:[ c1 ] linear ] linear; c2;
            ]
          linear
      in
      equal uop linear
        (Ops.graph_rewrite ~ctx:() nested Schedule.pm_flatten_linear)
  | calls -> failf "softmax schedules %d kernels" (List.length calls)

let flat () =
  let linear = Schedule.create_schedule (program "softmax_kernels") in
  equal uop linear (Ops.graph_rewrite ~ctx:() linear Schedule.pm_flatten_linear)

let flatten =
  group "pm_flatten_linear"
    [
      test "Linears nested in Linears are inlined in order" flattening;
      test "a flat Linear stays" flat;
    ]

(* is_store_after *)

let cpu = Ops.Single "CPU"
let buffer = Ops.new_buffer ~slot:1 cpu 16 Float32
let alloc = Ops.alloc ~device:cpu [ Int 16 ] Float32
let into p = Ops.store (Ops.index p [ Ops.int 0 ]) (Ops.float 1.0)

let store_after =
  let is_store_after name expected u =
    test name (fun () -> equal bool expected (Schedule.is_store_after u))
  in
  group "is_store_after"
    [
      is_store_after "an After of a store into a buffer" true
        (Ops.after buffer [ into buffer ]);
      is_store_after "an After of a call on a buffer" true
        (Ops.after buffer [ copy_call ]);
      is_store_after "an After of a store into call-local storage" true
        (Ops.after alloc [ into alloc ]);
      is_store_after "an After of a call on call-local storage" false
        (Ops.after alloc [ copy_call ]);
      is_store_after "an After of a call, then a store, on call-local storage"
        false
        (Ops.after alloc [ copy_call; into alloc ]);
      is_store_after "an After of a call on moved call-local storage" false
        (Ops.after (Ops.reshape alloc [ Int 4; Int 4 ]) [ copy_call ]);
      is_store_after "a store" false (into buffer);
      is_store_after "a buffer" false buffer;
    ]

(* contiguous_mops_to_view *)

let grid = Ops.reshape (Ops.new_buffer ~slot:1 cpu 64 Float32) [ Int 8; Int 8 ]
let rows = Ops.shrink grid [ Some (Int 2, Int 6); None ]
let columns = Ops.shrink grid [ None; Some (Int 2, Int 6) ]
let to_cpu1 u = Ops.copy_to_device u (Single "CPU:1")

let bytes =
  Ops.bitcast
    (Ops.shrink (Ops.new_buffer ~slot:1 cpu 64 Uint8) [ Some (Int 8, Int 24) ])
    Float32

(* A value of shape [4; 8] sharded on axis 0 over two devices. *)
let sharded_grid =
  List.find
    (fun n -> Ops.op n = Unshard)
    (Ops.toposort (program "shard_to_one"))

let sharded = Ops.reshape sharded_grid [ Int 32 ]

(* [through_buffer view] checks that [view] reads its buffer only through a
   reshape, shrink, bitcast or reassembly of shards. *)
let rec through_buffer view =
  match Ops.op view with
  | Buffer -> ()
  | Reshape | Shrink | Bitcast | Unshard -> through_buffer (Ops.nth view 0)
  | op -> failf "the view reads through a %a" Op.pp op

let values = list (array Dtypes.const)

let copies_a_view c =
  let viewed =
    require_some (Schedule.contiguous_mops_to_view c (Ops.nth c 0))
  in
  equal bool ~msg:"the copy stays" true (Ops.op viewed = Copy);
  through_buffer (Ops.nth viewed 0);
  let buffers = filled c in
  equal values (Tensors.eval ~buffers c) (Tensors.eval ~buffers viewed)

(* The copy holds the whole value on each device, and the view, sharded, is the
   whole value. *)
let shards_a_view () =
  let c = Ops.copy_to_device sharded (Multi [ "CPU"; "CPU:1" ]) in
  let view = require_some (Schedule.contiguous_mops_to_view c sharded) in
  equal bool ~msg:"sharded again" true (Ops.op view = Unshard);
  through_buffer view;
  let buffers = filled c in
  match Tensors.eval ~buffers view with
  | [ whole ] ->
      List.iteri
        (fun k on_k ->
          equal (array Dtypes.const)
            ~msg:(Printf.sprintf "device %d" k)
            whole on_k)
        (Tensors.eval ~buffers c)
  | parts -> failf "the view is %d values" (List.length parts)

let views =
  group "contiguous_mops_to_view"
    [
      test "a copy of contiguous rows reads a view of them" (fun () ->
          copies_a_view (to_cpu1 rows));
      test "a copy of a bitcast of contiguous bytes reads a view of them"
        (fun () -> copies_a_view (to_cpu1 bytes));
      test
        "a copy of a sharded value onto its devices is a view of each shard, \
         sharded again"
        shards_a_view;
      test "columns are not contiguous" (fun () ->
          equal (option uop) None
            (Schedule.contiguous_mops_to_view (to_cpu1 columns) columns));
      test "a permutation is not contiguous" (fun () ->
          let moved = Ops.permute grid [ 1; 0 ] in
          equal (option uop) None
            (Schedule.contiguous_mops_to_view (to_cpu1 moved) moved));
      test "a copy to one device of a sharded value has no view" (fun () ->
          equal (option uop) None
            (Schedule.contiguous_mops_to_view
               (Ops.copy_to_device sharded cpu)
               sharded));
      test "a store into a bitcast of contiguous bytes stores into a view"
        (fun () ->
          let value = Ops.new_buffer ~slot:2 cpu 4 Float32 in
          let c = Ops.store bytes value in
          let viewed =
            require_some (Schedule.contiguous_mops_to_view c bytes)
          in
          equal bool ~msg:"the store stays" true (Ops.op viewed = Store);
          equal uop ~msg:"the value stays" value (Ops.nth viewed 1);
          let view = Ops.nth viewed 0 in
          through_buffer view;
          let buffers = filled c in
          equal values
            (Tensors.eval ~buffers bytes)
            (Tensors.eval ~buffers view));
      test
        "a copy of a sharded value whose shards are not contiguous has no view"
        (fun () ->
          let moved = Ops.permute sharded_grid [ 1; 0 ] in
          equal (option uop) None
            (Schedule.contiguous_mops_to_view
               (Ops.copy_to_device moved (Multi [ "CPU"; "CPU:1" ]))
               moved));
      test "a copy of part of a sharded axis has no view" (fun () ->
          let part = Ops.shrink sharded_grid [ Some (Int 0, Int 2); None ] in
          equal (option uop) None
            (Schedule.contiguous_mops_to_view
               (Ops.copy_to_device part (Multi [ "CPU"; "CPU:1" ]))
               part));
      test "a symbolic shape has no view" (fun () ->
          let v = Ops.variable "n" (`Int (Z.of_int 1)) (`Int (Z.of_int 8)) in
          let prefix = Ops.shrink grid [ Some (Int 0, Sym v); None ] in
          equal (option uop) None
            (Schedule.contiguous_mops_to_view (to_cpu1 prefix) prefix));
      test "any other operation is the view itself" (fun () ->
          let view =
            require_some (Schedule.contiguous_mops_to_view rows rows)
          in
          equal (list int) [ 4; 8 ]
            (List.map
               (function Ops.Int n -> n | Sym _ -> fail "a symbolic size")
               (Ops.shape view));
          let buffers = filled rows in
          equal values (Tensors.eval ~buffers rows) (Tensors.eval ~buffers view));
    ]

(* Variables and devices *)

let bound_var name =
  List.find
    (fun n -> Ops.is_bound_var n && Ops.expr n = name)
    (Ops.toposort (program "variable_two"))

(* variable_two binds v to 4 and w to 7; [rebound value] binds, in w's place,
   another variable named v, of another range, to [value]. *)
let rebound value =
  Ops.substitute (program "variable_two")
    [
      ( bound_var "w",
        Ops.bind
          (Ops.variable "v" (`Int (Z.of_int 0)) (`Int (Z.of_int 10)))
          (`Int (Z.of_int value)) );
    ]

(* add reads the buffers of slots 2 and 3. *)
let several_devices () =
  let big = program "add" in
  let moved b =
    match Ops.arg b with
    | Param ({ slot = 3; _ } as p) ->
        Some
          ( b,
            Ops.replace ~arg:(Param { p with device = Some (Single "CPU:1") }) b
          )
    | _ -> None
  in
  let input = require_some (List.find_map moved (buffers_of big)) in
  let across = Ops.substitute big [ input ] in
  raises_match (Exn.invalid_arg ~substring:"same device") (fun () ->
      Schedule.create_linear_with_vars across)

let variables =
  group "create_linear_with_vars › rules"
    [
      test "two variables of one name bound to two values are refused"
        (fun () ->
          let big = rebound 7 in
          raises_match (Exn.invalid_arg ~substring:"bind mismatch on v")
            (fun () -> Schedule.create_linear_with_vars big));
      test "two variables of one name bound to one value are one" (fun () ->
          equal
            (list (pair string int))
            [ ("v", 4) ]
            (snd (Schedule.create_linear_with_vars (rebound 4))));
      test "a kernel on buffers of two devices is refused" several_devices;
    ]

let () =
  exit
    (run "Tolk_next.Schedule"
       [
         linears;
         var_vals;
         schedules;
         ordered;
         realized;
         capturing;
         spec;
         cache;
         ordering;
         flatten;
         store_after;
         views;
         variables;
       ])
