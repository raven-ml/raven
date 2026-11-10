(* Tests of Tolk.Schedule: the schedules tinygrad makes of the programs it
   realizes and the order of their kernels, the laws that a schedule writes what
   its program computes, the schedule cache, and the rules of the module. *)

open Windtrap
open Tolk

let uop = Uops.uop
let program name = Golden.sink (name ^ ".golden")

(* A setting the suite declares to reach output, of which scheduling knows
   nothing: the caches key on it all the same. It is declared before a child
   runs, so that the child keys on it too. *)
let declared = Setting.int ~reach:Output "TOLK_TEST_SCHEDULE" 0

(* A child of the schedules-on-disk tests schedules matmul, before the suite's
   own schedules run. *)
let () =
  Disk_cache.play
    [
      ( "schedule",
        fun () ->
          let linear, _ =
            Schedule.create_linear_with_vars ~capturing:true (program "matmul")
          in
          print_string (Graph.to_string linear) );
    ]

(* Recorded graphs *)

(* after_part_of_older_write is recorded as a kernel graph alone. Its two
   stores go into one element each of one buffer: here each is a call of a
   one-element view of the buffer, and the two run as one launch, where the
   recorded linear indexes the whole buffer in a kernel per store. *)
let programs =
  [
    "add";
    "add_row_from_self";
    "after_older_and_new_write";
    "after_older_write";
    "after_sum";
    "assign";
    "assign_bitcast";
    "assign_double_diamond";
    "assign_flipped_from_self";
    "assign_permuted_self";
    "assign_reads_behind_other_after";
    "attention";
    "call_given_one_state_twice";
    "cat";
    "chained_functions";
    "clone";
    "contiguous";
    "conv";
    "copy";
    "copy_computed";
    "copy_of_overwritten";
    "copy_one";
    "copy_view";
    "custom_kernel";
    "custom_kernel_stage";
    "custom_kernel_stage_table";
    "custom_kernel_variable_arg";
    "disk_store";
    "disk_to";
    "disk_view_to";
    "double_matmul";
    "embedding";
    "full_invalid";
    "inline_function";
    "layernorm";
    "matmul";
    "precompiled_function";
    "precompiled_scalar";
    "read_beside_overwrite";
    "read_join_of_assigned";
    "read_join_of_overwritten";
    "read_overwritten_through";
    "read_then_overwrite";
    "readers_after_older_write";
    "reduce_axis_and_range";
    "reduce_multiple_paths";
    "reduce_singleton_axis_and_range";
    "roll_from_self";
    "select_no_lane";
    "select_no_lane_rows";
    "select_one_lane";
    "setitem";
    "softmax";
    "sort";
    "split_sum";
    "stage_free_range_reg";
    "stage_no_range_reg";
    "stage_of_overwritten";
    "stage_padded_slice";
    "sum";
    "swap_through_values";
    "two_outputs";
    "variable_offset";
    "variable_reduce";
    "variable_same";
    "variable_shrink";
    "variable_two";
    "variable_unused";
    "write_over_written";
  ]

let kernel_graphs =
  [
    "add_kernels";
    "add_row_from_self_kernels";
    "after_older_and_new_write_kernels";
    "after_older_write_kernels";
    "after_part_of_older_write_kernels";
    "after_sum_kernels";
    "assign_bitcast_kernels";
    "assign_double_diamond_kernels";
    "assign_flipped_from_self_kernels";
    "assign_kernels";
    "assign_permuted_self_kernels";
    "assign_reads_behind_other_after_kernels";
    "attention_kernels";
    "call_given_one_state_twice_kernels";
    "call_given_one_state_twice_kernels_1";
    "cat_kernels";
    "chained_functions_kernels";
    "chained_functions_kernels_1";
    "clone_kernels";
    "contiguous_kernels";
    "conv_kernels";
    "copy_computed_kernels";
    "copy_kernels";
    "copy_of_overwritten_kernels";
    "copy_one_kernels";
    "copy_view_kernels";
    "custom_kernel_kernels";
    "custom_kernel_stage_kernels";
    "custom_kernel_stage_table_kernels";
    "custom_kernel_variable_arg_kernels";
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
    "precompiled_function_kernels";
    "precompiled_function_kernels_1";
    "precompiled_scalar_kernels";
    "precompiled_scalar_kernels_1";
    "read_beside_overwrite_kernels";
    "read_join_of_assigned_kernels";
    "read_join_of_overwritten_kernels";
    "read_overwritten_through_kernels";
    "read_then_overwrite_kernels";
    "readers_after_older_write_kernels";
    "reduce_axis_and_range_kernels";
    "reduce_multiple_paths_kernels";
    "reduce_singleton_axis_and_range_kernels";
    "roll_from_self_kernels";
    "select_no_lane_kernels";
    "select_no_lane_rows_kernels";
    "select_one_lane_kernels";
    "setitem_kernels";
    "softmax_kernels";
    "sort_kernels";
    "split_sum_kernels";
    "stage_free_range_reg_kernels";
    "stage_no_range_reg_kernels";
    "stage_of_overwritten_kernels";
    "stage_padded_slice_kernels";
    "sum_kernels";
    "swap_through_values_kernels";
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
  else if Dtype.is_unsigned dtype then `Int (Bigint.of_int (k + 3))
  else `Int (Bigint.of_int k)

let given u =
  List.filter_map
    (fun n ->
      match (Ops.op n, Ops.arg n) with
      | (Param | Buffer), Param p when p.addrspace <> Alu -> Some (n, p)
      | _ -> None)
    (Ops.toposort ~calls:Skip u)

let filled u =
  List.map
    (fun (n, (p : Ops.param_arg)) ->
      let size = Shape.max_numel n in
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

let calls_only_kernels u =
  List.for_all
    (fun n -> Ops.op n <> Call || Ops.op (Ops.nth n 0) = Sink)
    (Ops.toposort ~calls:Skip u)

(* Custom kernels keep their views until they are compiled, which the
   Interpreter does not run. The call of assign_bitcast passes a buffer and a
   bitcast of it as two arguments, one memory that storage, a memory per slot,
   cannot alias. The call of custom_kernel_variable_arg passes a variable,
   which the kernel graph's writes hold no storage for. The functions of
   custom_kernel_stage and custom_kernel_stage_table load, and those of
   stage_padded_slice, stage_free_range_reg and stage_no_range_reg stage,
   which the Interpreter does not evaluate; it sums no range of its own beside
   a sum's leading axes, as reduce_axis_and_range and
   reduce_singleton_axis_and_range do. The engine's suite runs them. *)
let apart_from_values =
  [
    "custom_kernel_kernels";
    "custom_kernel";
    "custom_kernel_variable_arg_kernels";
    "custom_kernel_variable_arg";
    "custom_kernel_stage_kernels";
    "custom_kernel_stage";
    "custom_kernel_stage_table_kernels";
    "custom_kernel_stage_table";
    "reduce_axis_and_range_kernels";
    "reduce_axis_and_range";
    "reduce_singleton_axis_and_range_kernels";
    "reduce_singleton_axis_and_range";
    "stage_padded_slice_kernels";
    "stage_padded_slice";
    "stage_free_range_reg_kernels";
    "stage_free_range_reg";
    "stage_no_range_reg_kernels";
    "stage_no_range_reg";
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
          | [ k; v ] -> (key k, `Int (Bigint.of_string v))
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
           List.mem name apart_from_values || not (calls_only_kernels kg)
         then None
         else Some (runs_in_order name))
       kernel_graphs)

let unplanned big = fst (Schedule.create_linear_with_vars ~capturing:true big)

let computes big =
  let buffers = filled big in
  let vars =
    List.map
      (fun (n, v) -> (n, `Int (Bigint.of_int v)))
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
      | Param, Param { addrspace = Alu; name = Some _; _ } -> true
      | _ -> false)
    (Ops.toposort ~calls:Enter u)

let realized =
  group "create_linear_with_vars › values"
    (List.filter_map
       (fun name ->
         let big = program name in
         if
           List.mem name apart_from_values
           || has_variables big
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
  List.filter (fun n -> Ops.op n = Buffer) (Ops.toposort ~calls:Skip big)

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
        Setting.context
          [ B (Setting.spec, 1) ]
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
  Ops.substitute ~calls:Enter ~pass:Fixed_point (program name)
    [ (Ops.float constant, Ops.float x) ]

let with_settings ~debug ~scache f =
  Setting.context [ B (Setting.debug, debug); B (Setting.scache, scache) ] f

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
    List.find (fun n -> Ops.op n = Alloc) (Ops.toposort ~calls:Skip big)
  in
  let in_slot slot =
    match Ops.arg alloc with
    | Param p ->
        Ops.substitute ~calls:Skip ~pass:Fixed_point big
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

(* Storage that a builder numbers from the process's counter, such as a staged
   loop's carry, can sit in the body of a precompiled call, where it stays
   storage. [counted g] is precompiled_function [g] as such a body: its output a
   parameter of the call, its input storage in a slot the counter gives now. *)
let counted g =
  let storage slot =
    List.find
      (fun n ->
        match Ops.arg n with
        | Param p -> Ops.op n = Buffer && p.slot = slot
        | _ -> false)
      (Ops.toposort ~calls:Skip g)
  in
  let out = storage 4 and input = storage 3 in
  let fresh =
    match Ops.arg input with
    | Param p ->
        Ops.replace input ~arg:(Param { p with slot = Ops.unique_num () })
    | _ -> fail "storage without its argument"
  in
  let body =
    Ops.substitute ~calls:Skip ~pass:Fixed_point g
      [ (out, Shape.param_like out 0); (input, fresh) ]
  in
  (Ops.sink [ Ops.after out [ Ops.call ~precompile:true body [ out ] ] ], fresh)

(* The second schedule hits under the first's keys, and holds its own
   storage. *)
let counted_alike () =
  let g = unique "precompiled_function" in
  let first, mine = counted g in
  ignore (Ops.unique_num ());
  let second, theirs = counted g in
  let linear =
    with_settings ~debug:3 ~scache:1 (fun () ->
        ignore (Schedule.create_linear_with_vars first);
        fst (Schedule.create_linear_with_vars ~capturing:true second))
  in
  let lines = reports () in
  let half = List.length lines / 2 in
  let missed = List.filteri (fun i _ -> i < half) lines in
  equal (list report)
    (List.map (fun (n, _, key) -> (n, " cache hit", key)) missed)
    (List.filteri (fun i _ -> i >= half) lines);
  let held = Ops.toposort ~calls:Enter linear in
  equal ~msg:"the second schedule holds its storage, not the first's"
    (pair bool bool) (true, false)
    (List.memq theirs held, List.memq mine held)

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
  let v = List.find Shape.is_bound_var (Ops.toposort ~calls:Enter big) in
  let five =
    Ops.substitute ~calls:Skip ~pass:Fixed_point big
      [ (v, Shape.bind (Shape.unbound v) (`Int (Bigint.of_int 5))) ]
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
  let held = Ops.backward_slice_with_self ~calls:Skip big in
  let made =
    List.concat_map
      (fun r ->
        List.filter_map
          (fun n ->
            match Ops.arg n with
            | Param p when Ops.op n = Buffer && not (Ops.Nodes.mem n held) ->
                Some p.slot
            | _ -> None)
          (Ops.toposort ~calls:Enter r))
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

(* Domains scheduling one body at once schedule each of its calls once: the
   others wait for the schedule and print a hit. *)
let one_body_once () =
  let big = unique "assign" in
  with_settings ~debug:3 ~scache:1 (fun () ->
      ignore (in_domains (List.init 4 (fun _ -> big))));
  let lines = reports () in
  let keys = List.sort_uniq compare (List.map (fun (_, _, k) -> k) lines) in
  let missed =
    List.filter_map
      (fun (_, verdict, k) -> if verdict = "CACHE MISS" then Some k else None)
      lines
  in
  equal ~msg:"lines" int (4 * List.length keys) (List.length lines);
  equal ~msg:"misses" (list string) keys (List.sort compare missed)

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

(* A setting that shapes a schedule keeps a body scheduled under another value
   from being returned, the suite's declared setting among them. *)
let separates setting () =
  let big = unique "assign" in
  with_settings ~debug:3 ~scache:1 (fun () ->
      ignore (Schedule.create_linear_with_vars big);
      Setting.context [ setting ] (fun () ->
          ignore (Schedule.create_linear_with_vars big)));
  equal (list string)
    [ "CACHE MISS"; "CACHE MISS" ]
    (List.map (fun (_, verdict, _) -> verdict) (reports ()))

let cache =
  group "create_linear_with_vars › cache"
    [
      group "a body scheduled under one setting misses under another"
        (List.map
           (fun (name, setting) -> test name (separates setting))
           Setting.
             [
               ("SPLIT_REDUCEOP", B (split_reduceop, false));
               ("MAX_KERNEL_BUFFERS", B (max_kernel_buffers, 8));
               ("DEFAULT_FLOAT", B (default_float, "half"));
               ("DEFAULT_INT", B (default_int, "long"));
               ("a setting its caller declares", B (declared, 1));
             ]);
      test "a body scheduled again is a hit under the key it missed on"
        miss_then_hit;
      test "a hit is the schedule the miss made" hit_is_the_miss;
      test "with SCACHE=0 every schedule misses" uncached;
      test "two bodies miss under two keys" keys_differ;
      test "DEBUG=1 prints only schedules of several kernels" debug_one;
      test "DEBUG=0 prints nothing" quiet;
      test "the time printed is the time scheduling took" timed;
      test "call-local storage in other slots is the same body" renumbered;
      test "storage numbered in another counter state is the same body"
        counted_alike;
      test "a function called three times is scheduled once" chained;
      test "a variable bound to another value is the same body" rebinding;
      test "domains scheduling one body at once schedule it alike"
        one_body_at_once;
      test "domains scheduling one body at once schedule it once"
        one_body_once;
      test "domains scheduling bodies at once schedule each as alone"
        bodies_at_once;
    ]

(* Rules *)

(* The kernel graph of clone: a kernel that copies its argument 1 into its
   argument 0. *)
let clone = program "clone_kernels"

let copy_call =
  List.find (fun n -> Ops.op n = Call) (Ops.toposort ~calls:Skip clone)

let copy_kernel = Ops.nth copy_call 0

let x, y =
  match Ops.src copy_call with
  | [ _; x; y ] -> (x, y)
  | _ -> failwith "clone's call takes two arguments"

let copies ~into ~from = Ops.replace ~src:[ copy_kernel; into; from ] copy_call

(* Running the schedule of [kg] writes what [kg] does. *)
let keeps kg =
  let buffers = filled kg in
  equal (list write)
    (Kernel_graphs.writes ~buffers kg)
    (Kernel_graphs.linear_writes ~buffers (Schedule.create_schedule kg))

let swap () =
  (* each kernel overwrites what the other reads as it was given: one reads a
     copy *)
  let a = copies ~into:x ~from:y and b = copies ~into:y ~from:x in
  let kg = Ops.sink [ Ops.after x [ a ]; Ops.after y [ b ] ] in
  keeps kg;
  let allocs =
    List.filter (fun n -> Ops.op n = Alloc)
      (Ops.toposort ~calls:Skip (Schedule.create_schedule kg))
  in
  equal int 1 (List.length allocs)

(* The kernel graph of read_overwritten_through: [b + (a * 3)], where [b] read
   [a] before [a] was assigned. The last kernel reads [a] as it was and waits,
   through [a * 3], for the kernel that overwrites it. *)
let read_overwritten () = keeps (program "read_overwritten_through_kernels")

let chain () =
  (* the second kernel reads what the first wrote, and overwrites what the first
     read *)
  let a = copies ~into:x ~from:y in
  let b = copies ~into:y ~from:(Ops.after x [ a ]) in
  let kg = Ops.sink [ Ops.after x [ a ]; Ops.after y [ b ] ] in
  equal (list uop)
    [ a; copies ~into:y ~from:x ]
    (Ops.src (Schedule.create_schedule kg))

(* [y] in slot [slot], with [f] applied to its argument. *)
let like_y ?(f = Fun.id) slot =
  match Ops.arg y with
  | Param p -> Ops.replace ~arg:(Param (f { p with slot })) y
  | _ -> failwith "a parameter without its argument"

(* A third and a fourth parameter. *)
let z = like_y 2
let w = like_y 3

(* A call of the copy kernel into [into] from [from], passed [rest] too. *)
let reads ~into ~from rest =
  Ops.replace ~src:(copy_kernel :: into :: from :: rest) copy_call

let overwrite () =
  (* the second kernel writes the state the first left *)
  let a = copies ~into:x ~from:y in
  let b = copies ~into:(Ops.after x [ a ]) ~from:z in
  let kg = Ops.sink [ Ops.after (Ops.after x [ a ]) [ b ] ] in
  equal (list uop)
    [ a; copies ~into:x ~from:z ]
    (Ops.src (Schedule.create_schedule kg))

(* [b] reads x both before and after [a] overwrites it, and reads in its
   argument 1 the state after. *)
let read_across () =
  let a = copies ~into:x ~from:y in
  let b = Ops.replace ~src:[ copy_kernel; z; Ops.after x [ a ]; x ] copy_call in
  keeps (Ops.sink [ Ops.after x [ a ]; Ops.after z [ b ] ])

(* As [read_across], [b] reading in its argument 1 the state before. *)
let read_across_before () =
  let a = copies ~into:x ~from:y in
  let b = Ops.replace ~src:[ copy_kernel; z; x; Ops.after x [ a ] ] copy_call in
  keeps (Ops.sink [ Ops.after x [ a ]; Ops.after z [ b ] ])

(* [r], listed under [x]'s write beside [a], reads [x] without storing into
   it: it reads [x] as it was, so it runs before [a]. *)
let read_in_write order () =
  let a = copies ~into:x ~from:y in
  let r = copies ~into:z ~from:x in
  keeps (Ops.sink [ Ops.after x (order [ a; r ]); Ops.after z [ r ] ])

(* Two kernels read [x] as it was and wait for its overwrite: both read one
   copy. *)
let read_by_two () =
  let a = copies ~into:x ~from:y in
  let b = reads ~into:z ~from:x [ Ops.after x [ a ] ] in
  let c = reads ~into:w ~from:x [ Ops.after x [ a ] ] in
  let kg =
    Ops.sink [ Ops.after x [ a ]; Ops.after z [ b ]; Ops.after w [ c ] ]
  in
  keeps kg;
  let allocs =
    List.filter (fun n -> Ops.op n = Alloc)
      (Ops.toposort ~calls:Skip (Schedule.create_schedule kg))
  in
  equal int 1 (List.length allocs)

(* One kernel reads [x] and [y] as they were and waits for the overwrite of
   each. *)
let reads_two () =
  let a = copies ~into:x ~from:w and b = copies ~into:y ~from:w in
  let c = reads ~into:z ~from:x [ y; Ops.after x [ a ]; Ops.after y [ b ] ] in
  keeps
    (Ops.sink [ Ops.after x [ a ]; Ops.after y [ b ]; Ops.after z [ c ] ])

(* [b] reads [x] as it was and waits for its overwrite [a]; [e] reads [y] as
   it was and waits, through [b], for its overwrite [d], which reads what [b]
   wrote: the copy for [e] leaves [b] reading its own copy. *)
let read_through_a_copy () =
  let a = copies ~into:x ~from:y in
  let b = reads ~into:z ~from:x [ Ops.after x [ a ] ] in
  let d = copies ~into:y ~from:(Ops.after z [ b ]) in
  let e = reads ~into:w ~from:y [ Ops.after y [ d ] ] in
  keeps
    (Ops.sink
       [
         Ops.after x [ a ];
         Ops.after z [ b ];
         Ops.after y [ d ];
         Ops.after w [ e ];
       ])

(* The copy of a state lies as its storage does. *)
let copy_keeps_phase () =
  let x = like_y ~f:(fun p -> { p with phase = 4 }) 0 in
  let a = copies ~into:x ~from:y in
  let b = reads ~into:z ~from:x [ Ops.after x [ a ] ] in
  let kg = Ops.sink [ Ops.after x [ a ]; Ops.after z [ b ] ] in
  keeps kg;
  let phases =
    List.filter_map
      (fun n ->
        match (Ops.op n, Ops.arg n) with
        | Alloc, Param p -> Some p.phase
        | _ -> None)
      (Ops.toposort ~calls:Skip (Schedule.create_schedule kg))
  in
  equal (list int) [ 4 ] phases

let not_an_effect () =
  let kg = Ops.sink [ Ops.after x [ Ops.float 2.0 ] ] in
  let const = Format.asprintf "%a" Op.pp Const in
  raises_match (Exn.invalid_arg ~sub:const) (fun () ->
      Schedule.create_schedule kg)

let not_storage () =
  let kg = Ops.sink [ Ops.after x [ copies ~into:x ~from:(Ops.float 2.0) ] ] in
  raises_match (Exn.invalid_arg ~sub:"buffer state") (fun () ->
      Schedule.create_schedule kg)

let loop = Ops.range ~axis_type:Loop (Int 4) [ 0 ]

let ended () =
  let a = copies ~into:x ~from:y in
  let kg = Ops.sink [ Ops.after x [ Ops.end_ a [ loop ] ] ] in
  equal (list uop)
    [ Ops.end_ a [ loop ] ]
    (Ops.src (Schedule.create_schedule kg))

(* A loop runs after the call that writes what it reads, and before the call
   that overwrites what it reads. *)
let loop_after_writer () =
  let w = copies ~into:y ~from:x in
  let l = Ops.end_ (copies ~into:x ~from:(Ops.after y [ w ])) [ loop ] in
  let kg = Ops.sink [ Ops.after x [ l ] ] in
  equal (list uop)
    [ w; Ops.end_ (copies ~into:x ~from:y) [ loop ] ]
    (Ops.src (Schedule.create_schedule kg))

let loop_before_overwriter () =
  let l = Ops.end_ (copies ~into:x ~from:y) [ loop ] in
  let o = copies ~into:y ~from:(Ops.after x [ l ]) in
  let kg = Ops.sink [ Ops.after x [ l ]; Ops.after y [ o ] ] in
  equal (list uop)
    [ l; copies ~into:y ~from:x ]
    (Ops.src (Schedule.create_schedule kg))

let ended_store () =
  let kg =
    Ops.sink [ Ops.after x [ Ops.end_ (Ops.store x (Ops.float 2.0)) [ loop ] ] ]
  in
  raises_match (Exn.invalid_arg ?sub:None) (fun () ->
      Schedule.create_schedule kg)

let bound_argument () =
  let n =
    Shape.bind
      (Ops.variable "n" (`Int (Bigint.of_int 1)) (`Int (Bigint.of_int 8)))
      (`Int (Bigint.of_int 3))
  in
  let a = Ops.replace ~src:[ copy_kernel; x; y; n ] copy_call in
  let kg = Ops.sink [ Ops.after x [ a ] ] in
  equal (list uop)
    [ copies ~into:x ~from:y ]
    (Ops.src (Schedule.create_schedule kg))

(* Order by storage

   Ordered by storage, a ready kernel runs by its number, the order the walk of
   the sink meets it, and one that frees storage and on no device makes more
   than it frees runs before it; otherwise ready kernels run as they become
   ready. Storage that kernels write lives from the first kernel that takes
   it to the last. Each parameter below holds as many elements as [y]. *)

let param slot = like_y slot

(* [a] makes [t], [c] makes [w] from [y], which no kernel writes, and [b] makes
   [z] from [t], its last reader: [b] makes as much as it frees. *)
let frees_first () =
  let t = param 4 and w = param 5 and z = param 6 in
  let a = copies ~into:t ~from:y in
  let c = copies ~into:w ~from:y in
  let b = copies ~into:z ~from:(Ops.after t [ a ]) in
  let kg = Ops.sink [ Ops.after t [ a ]; Ops.after w [ c ]; Ops.after z [ b ] ] in
  equal (list uop)
    [ a; copies ~into:z ~from:t; c ]
    (Ops.src (Schedule.create_schedule ~by_storage:true kg));
  keeps kg

(* {!frees_first}'s kernels, ordered as they become ready: [b] waits for [a],
   which [c] does not. *)
let as_ready () =
  let t = param 4 and w = param 5 and z = param 6 in
  let a = copies ~into:t ~from:y in
  let c = copies ~into:w ~from:y in
  let b = copies ~into:z ~from:(Ops.after t [ a ]) in
  let kg = Ops.sink [ Ops.after t [ a ]; Ops.after w [ c ]; Ops.after z [ b ] ] in
  equal (list uop)
    [ a; c; copies ~into:z ~from:t ]
    (Ops.src (Schedule.create_schedule kg));
  keeps kg

(* [p] makes a value that [s1] and [s2] each read into a piece of their own,
   which [c1] and [c2] read: each piece is read before the next is made. *)
let pieces_interleave () =
  let p = param 4 and s1 = param 5 and s2 = param 6 in
  let o1 = param 7 and o2 = param 8 in
  let made = copies ~into:p ~from:y in
  let pv = Ops.after p [ made ] in
  let k1 = copies ~into:s1 ~from:pv and k2 = copies ~into:s2 ~from:pv in
  let r1 = copies ~into:o1 ~from:(Ops.after s1 [ k1 ]) in
  let r2 = copies ~into:o2 ~from:(Ops.after s2 [ k2 ]) in
  let kg =
    Ops.sink
      [ Ops.after o1 [ r1 ]; Ops.after o2 [ r2 ]; pv; Ops.after s1 [ k1 ]; Ops.after s2 [ k2 ] ]
  in
  equal (list uop)
    [
      made;
      copies ~into:s1 ~from:p;
      copies ~into:o1 ~from:s1;
      copies ~into:s2 ~from:p;
      copies ~into:o2 ~from:s2;
    ]
    (Ops.src (Schedule.create_schedule ~by_storage:true kg));
  keeps kg

(* [ms] makes [s] from [t], then [k] reads [t] last and makes [n], and [r]
   reads [s] last into [o], which [m] made: both free, [k] as much as it
   makes and [r] making nothing, so [r] runs first though [k] comes before
   it. *)
let only_frees_first () =
  let t = param 4 and s = param 5 and o = param 6 and n = param 7 in
  let m = copies ~into:o ~from:y in
  let mt = copies ~into:t ~from:y in
  let tv = Ops.after t [ mt ] in
  let ms = copies ~into:s ~from:tv in
  let k = copies ~into:n ~from:tv in
  let ov = Ops.after o [ m ] in
  let r = copies ~into:ov ~from:(Ops.after s [ ms ]) in
  let kg =
    Ops.sink
      [ ov; tv; Ops.after s [ ms ]; Ops.after n [ k ]; Ops.after ov [ r ] ]
  in
  equal (list uop)
    [
      m;
      mt;
      copies ~into:s ~from:t;
      copies ~into:o ~from:s;
      copies ~into:n ~from:t;
    ]
    (Ops.src (Schedule.create_schedule ~by_storage:true kg));
  keeps kg

let by_storage =
  group "create_schedule › order by storage"
    [
      test "a kernel that frees storage runs before one that only makes it"
        frees_first;
      test "a kernel that only frees runs before one that also makes storage"
        only_frees_first;
      test "pieces of one value are each read before the next is made"
        pieces_interleave;
      test "without it, ready kernels run as they become ready" as_ready;
    ]

let ordering =
  group "create_schedule › rules"
    [
      test "a kernel's argument that is no storage is refused" not_storage;
      test "an end of a call over a loop schedules the call in its loop" ended;
      test "a loop runs after the call that writes what it reads"
        loop_after_writer;
      test "a loop runs before the call that overwrites what it reads"
        loop_before_overwriter;
      test "a kernel reading storage before and after a write reads both"
        read_across;
      test "a kernel reading storage before and after a write reads it before"
        read_across_before;
      test "a kernel of a write that reads its state and stores elsewhere reads \
            it as it was"
        (read_in_write Fun.id);
      test "a kernel of a write that reads its state, listed first, reads it as \
            it was"
        (read_in_write List.rev);
      test "two kernels that wait for the overwrite of a state read one copy"
        read_by_two;
      test "a kernel that waits for the overwrites of two states reads both"
        reads_two;
      test "a kernel that waits through a copying kernel reads its state"
        read_through_a_copy;
      test "the copy of a state lies at the state's phase" copy_keeps_phase;
      test "a kernel overwriting a state runs after the kernel that made it"
        overwrite;
      test "an end of something other than a call is refused" ended_store;
      test "a bound variable among a kernel's arguments leaves its call"
        bound_argument;
      test "kernels that overwrite each other's inputs read one copy" swap;
      test "a kernel that waits for the overwrite of a state it reads reads it"
        read_overwritten;
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
        (Ops.graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:() nested
           (After_sources Schedule.pm_flatten_linear))
  | calls -> failf "softmax schedules %d kernels" (List.length calls)

let flat () =
  let linear = Schedule.create_schedule (program "softmax_kernels") in
  equal uop linear
    (Ops.graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:() linear
       (After_sources Schedule.pm_flatten_linear))

let flatten =
  group "pm_flatten_linear"
    [
      test "Linears nested in Linears are inlined in order" flattening;
      test "a flat Linear stays" flat;
    ]

(* is_store_after *)

let cpu = "CPU"
let buffer = Ops.new_buffer ~slot:1 cpu 16 Float32
let alloc = Shape.alloc ~device:cpu [ Int 16 ] Float32
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
        (Ops.after (Shape.reshape alloc [ Int 4; Int 4 ]) [ copy_call ]);
      is_store_after "a store" false (into buffer);
      is_store_after "a buffer" false buffer;
    ]

(* contiguous_mops_to_view *)

let grid = Shape.reshape (Ops.new_buffer ~slot:1 cpu 64 Float32) [ Int 8; Int 8 ]
let rows = Shape.shrink grid [ Some (Int 2, Int 6); None ]
let columns = Shape.shrink grid [ None; Some (Int 2, Int 6) ]
let to_cpu1 u = Ops.copy_to_device u "CPU:1"

let bytes =
  Ops.bitcast
    (Shape.shrink (Ops.new_buffer ~slot:1 cpu 64 Uint8) [ Some (Int 8, Int 24) ])
    Float32

(* [through_buffer view] checks that [view] reads its buffer only through a
   reshape, shrink or bitcast. *)
let rec through_buffer view =
  match Ops.op view with
  | Buffer -> ()
  | Reshape | Shrink | Bitcast -> through_buffer (Ops.nth view 0)
  | op -> failf "the view reads through a %a" Op.pp op

let values = array Dtypes.const

(* [n] float32s of [c]. *)
let full n c =
  Shape.expand
    (Shape.reshape (Ops.cast (Ops.float c) Float32) [ Int 1 ])
    [ Int n ]

let copies_a_view c =
  let viewed =
    require_some (Schedule.contiguous_mops_to_view c (Ops.nth c 0))
  in
  equal bool ~msg:"the copy stays" true (Ops.op viewed = Copy);
  through_buffer (Ops.nth viewed 0);
  let buffers = filled c in
  equal values (Tensors.eval ~buffers c) (Tensors.eval ~buffers viewed)

(* A view of a buffer of 24 floats that starts [phase] bytes past a 16-byte
   boundary, through up to three movements of shapes known here. *)
let gen_view =
  let open Gen in
  let rec each = function
    | [] -> constant []
    | g :: gs ->
        let+ x = g and+ xs = each gs in
        x :: xs
  in
  let dims u =
    List.map (function Ops.Int n -> n | Sym _ -> 0) (Shape.shape u)
  in
  let movement u =
    let shape = dims u in
    let axes = List.init (List.length shape) Fun.id in
    one_of
      [
        (let+ bounds =
           each
             (List.map
                (fun n ->
                  let* start = int_range 0 (n - 1) in
                  let+ stop = int_range (start + 1) n in
                  Some (Ops.Int start, Ops.Int stop))
                shape)
         in
         Shape.shrink u bounds);
        (let+ order = permutation axes in
         Shape.permute u order);
        (let+ flipped = subsequence axes in
         Shape.flip u flipped);
        constant (Shape.reshape u [ Int (List.fold_left ( * ) 1 shape) ]);
        (let+ pad = int_range 0 2 in
         Shape.pad u (List.map (fun _ -> Some (Ops.Int pad, Ops.Int 0)) shape));
      ]
  in
  let rec moves u k =
    if k = 0 then constant u
    else
      let* u = movement u in
      moves u (k - 1)
  in
  (let* phase = of_list [ 0; 4; 8; 12 ] in
   let* shape = of_list [ [ 24 ]; [ 4; 6 ]; [ 2; 3; 4 ] ] in
   let* k = int_range 1 3 in
   let b = Ops.new_buffer ~slot:1 ~phase cpu 24 Float32 in
   moves (Shape.reshape b (List.map (fun n -> Ops.Int n) shape)) k)
  |> with_pp (Testable.pp uop)

(* A stage's alignment and phase hold of the storage it gets: a view of the
   buffer when the stage reads a contiguous run of it, and storage of its own on
   a boundary otherwise. *)
let stage_phase_holds v =
  let stage = Ops.contiguous v in
  assume (Ops.op stage = Stage);
  let align, phase = Shape.storage_phase stage in
  let holds what (a, p) =
    equal int
      ~msg:(what ^ ": its alignment, up to the stage's")
      align (min a align);
    equal int
      ~msg:(what ^ ": its phase, modulo the stage's alignment")
      phase (p mod align)
  in
  holds "storage of its own" (16, 0);
  match Schedule.contiguous_mops_to_view stage v with
  | Some view ->
      cover "a view" true;
      holds "the view" (Shape.storage_phase view)
  | None -> cover "a view" false

let views =
  group "contiguous_mops_to_view"
    [
      test "a copy of contiguous rows reads a view of them" (fun () ->
          copies_a_view (to_cpu1 rows));
      test "a copy of a bitcast of contiguous bytes reads a view of them"
        (fun () -> copies_a_view (to_cpu1 bytes));
      test "columns are not contiguous" (fun () ->
          equal (option uop) None
            (Schedule.contiguous_mops_to_view (to_cpu1 columns) columns));
      test "a permutation is not contiguous" (fun () ->
          let moved = Shape.permute grid [ 1; 0 ] in
          equal (option uop) None
            (Schedule.contiguous_mops_to_view (to_cpu1 moved) moved));
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
        "stores into runs of a buffer at a bound start store into views of \
         one alignment, whatever the start"
        (fun () ->
          let buffer = Ops.new_buffer ~slot:6 cpu 16 Float32 in
          let value = Ops.new_buffer ~slot:2 cpu 8 Float32 in
          let at first =
            let s =
              Ops.variable "s" (`Int (Bigint.of_int 0)) (`Int (Bigint.of_int 8))
            in
            let start = Shape.bind s (`Int (Bigint.of_int first)) in
            let run =
              Shape.shrink buffer
                [ Some (Ops.Sym start, Ops.Sym Ops.O.(start + int 8)) ]
            in
            let viewed =
              require_some
                (Schedule.contiguous_mops_to_view (Ops.store run value) run)
            in
            equal bool ~msg:"the store stays" true (Ops.op viewed = Store);
            let view = Ops.nth viewed 0 in
            through_buffer view;
            Shape.storage_phase view
          in
          equal (pair int int) ~msg:"known to a float, 4 bytes in" (4, 0)
            (at 1);
          equal (pair int int) ~msg:"known to a float, 12 bytes in" (4, 0)
            (at 3));
      test "a store into contiguous rows of a buffer stores into a view of them"
        (fun () ->
          let value = Ops.new_buffer ~slot:2 cpu 32 Float32 in
          let c = Ops.store rows (Shape.reshape value [ Int 4; Int 8 ]) in
          let viewed =
            require_some (Schedule.contiguous_mops_to_view c rows)
          in
          equal bool ~msg:"the store stays" true (Ops.op viewed = Store);
          let view = Ops.nth viewed 0 in
          through_buffer view;
          let buffers = filled c in
          equal values
            (Tensors.eval ~buffers rows)
            (Tensors.eval ~buffers view));
      test "stores of one shape into two runs of a buffer share a kernel"
        (fun () ->
          let buffer = Ops.new_buffer ~slot:6 cpu 16 Float32 in
          let run lo = Shape.shrink buffer [ Some (Int lo, Int (lo + 8)) ] in
          let doubled slot =
            let v = Ops.new_buffer ~slot cpu 8 Float32 in
            Ops.add v v
          in
          let sink =
            Ops.sink
              [
                Ops.after buffer
                  [
                    Ops.store (run 0) (doubled 7);
                    Ops.store (run 8) (doubled 8);
                  ];
              ]
          in
          let linear, _ =
            Schedule.create_linear_with_vars ~capturing:true sink
          in
          (* The two stores are of one state: one launch runs them. *)
          match Ops.src linear with
          | [ l ] when Shape.is_launch (Ops.nth l 0) -> (
              match Shape.launch_members l with
              | [ a; b ] ->
                  equal uop ~msg:"the kernels" (Ops.nth a 0) (Ops.nth b 0)
              | ms -> failf "%d members, not 2" (List.length ms))
          | es -> failf "%d entries, not one launch" (List.length es));
      test "a store into rows of a buffer after a store stores into a run of \
            that state" (fun () ->
          let buffer = Ops.new_buffer ~slot:3 cpu 64 Float32 in
          let first = Ops.new_buffer ~slot:4 cpu 64 Float32 in
          let state = Ops.after buffer [ Ops.store buffer first ] in
          let rows =
            Shape.shrink
              (Shape.reshape state [ Int 8; Int 8 ])
              [ Some (Int 2, Int 6); None ]
          in
          let value = Ops.new_buffer ~slot:5 cpu 32 Float32 in
          let c = Ops.store rows (Shape.reshape value [ Int 4; Int 8 ]) in
          let viewed =
            require_some (Schedule.contiguous_mops_to_view c rows)
          in
          let rec named v =
            match Ops.op v with
            | Reshape | Shrink | Bitcast -> named (Ops.nth v 0)
            | _ -> v
          in
          let view = Ops.nth viewed 0 in
          equal uop ~msg:"the state" state (named view);
          let buffers = filled c in
          equal values
            (Tensors.eval ~buffers rows)
            (Tensors.eval ~buffers view));
      (* The first and the last store are of one value into one run, over
         two states: each is a store of its own, the last landing over the
         second's value. *)
      test "a store repeated over a later state stores again" (fun () ->
          let buffer = Ops.new_buffer ~slot:3 cpu 8 Float32 in
          let write state c =
            let run = Shape.shrink state [ Some (Int 6, Int 8) ] in
            Ops.after state [ Ops.store run (full 2 c) ]
          in
          computes (Ops.sink [ write (write (write buffer 3.5) 1.) 3.5 ]));
      (* A copy of a run of a state, then a store into that run: the copy
         reads the state, not the storage under it. *)
      test "a read of the run a store stores into reads the state" (fun () ->
          let buffer = Ops.new_buffer ~slot:3 cpu 8 Float32 in
          let other = Ops.new_buffer ~slot:4 cpu 4 Float32 in
          let state = Ops.after buffer [ Ops.store buffer (full 8 2.) ] in
          let run = Shape.shrink state [ Some (Int 4, Int 8) ] in
          computes
            (Ops.sink
               [
                 Ops.after state [ Ops.store run (full 4 5.) ];
                 Ops.after other [ Ops.store other run ];
               ]));
      (* A store that reads a state shares a value with a store of a marked
         write over that state: one kernel of both would read the elements
         the write's other store writes, whichever runs first. *)
      test "a store of a value a write over its read state shares runs first"
        (fun () ->
          let buffer = Ops.new_buffer ~slot:3 cpu 8 Float32 in
          let other = Ops.new_buffer ~slot:4 cpu 4 Float32 in
          let w = Ops.new_buffer ~slot:5 cpu 4 Float32 in
          let state = Ops.after buffer [ Ops.store buffer (full 8 1.) ] in
          let run a = Shape.shrink state [ Some (Int a, Int (a + 4)) ] in
          let shared = Ops.O.(w * w) in
          computes
            (Ops.sink
               [
                 Ops.after_disjoint state
                   [
                     Ops.store (run 0) (full 4 5.);
                     Ops.store (run 4) shared;
                   ];
                 Ops.after other [ Ops.store other Ops.O.(run 2 + shared) ];
               ]));
      test "a store into rows of a buffer of a value that reads it has no view"
        (fun () ->
          let buffer = Ops.new_buffer ~slot:9 cpu 16 Float32 in
          let run lo = Shape.shrink buffer [ Some (Int lo, Int (lo + 8)) ] in
          let c = Ops.store (run 1) (run 0) in
          equal (option uop) None (Schedule.contiguous_mops_to_view c (run 1)));
      (* The value's kernel reads [other], which a kernel that runs before it
         stores: no kernel reads and writes the buffer. *)
      test
        "a store into rows of a buffer of a value that reads it through other \
         storage stores into a view" (fun () ->
          let buffer = Ops.new_buffer ~slot:9 cpu 16 Float32 in
          let run lo = Shape.shrink buffer [ Some (Int lo, Int (lo + 8)) ] in
          let other = Ops.new_buffer ~slot:10 cpu 8 Float32 in
          let doubled =
            Ops.after other [ Ops.store other Ops.O.(run 0 * float 2.) ]
          in
          let c = Ops.store (run 1) Ops.O.(doubled + float 1.) in
          let viewed =
            require_some (Schedule.contiguous_mops_to_view c (run 1))
          in
          equal bool ~msg:"the store stays" true (Ops.op viewed = Store);
          let view = Ops.nth viewed 0 in
          through_buffer view;
          let buffers = filled c in
          equal values
            (Tensors.eval ~buffers (run 1))
            (Tensors.eval ~buffers view));
      test "a store into columns of storage has no view" (fun () ->
          let value = Ops.new_buffer ~slot:2 cpu 32 Float32 in
          let c = Ops.store columns (Shape.reshape value [ Int 8; Int 4 ]) in
          equal (option uop) None (Schedule.contiguous_mops_to_view c columns));
      test "a symbolic shape has no view" (fun () ->
          let v =
            Ops.variable "n" (`Int (Bigint.of_int 1)) (`Int (Bigint.of_int 8))
          in
          let prefix = Shape.shrink grid [ Some (Int 0, Sym v); None ] in
          equal (option uop) None
            (Schedule.contiguous_mops_to_view (to_cpu1 prefix) prefix));
      test "a stage of doubles one past a boundary is known to start on 8 bytes"
        (fun () ->
          let doubles = Ops.new_buffer ~slot:1 cpu 8 Float64 in
          let stage =
            Ops.contiguous (Shape.shrink doubles [ Some (Int 1, Int 8) ])
          in
          equal (pair int int) (8, 0) (Shape.storage_phase stage));
      test
        "a stage of rows padded apart, starting one double past a boundary, is \
         storage of its own on one" (fun () ->
          let doubles = Ops.new_buffer ~slot:1 cpu 3 Float64 in
          let rows =
            Shape.reshape
              (Shape.shrink doubles [ Some (Int 1, Int 3) ])
              [ Int 2; Int 1 ]
          in
          let padded = Shape.pad rows [ None; Some (Int 0, Int 1) ] in
          let stage = Ops.contiguous (Shape.reshape padded [ Int 4 ]) in
          equal (pair int int) (16, 0) (Shape.storage_phase stage));
      prop
        "a stage's alignment and phase hold of the storage it gets, a view or \
         its own"
        gen_view stage_phase_holds;
      test "any other operation is the view itself" (fun () ->
          let view =
            require_some (Schedule.contiguous_mops_to_view rows rows)
          in
          equal (list int) [ 4; 8 ]
            (List.map
               (function Ops.Int n -> n | Sym _ -> fail "a symbolic size")
               (Shape.shape view));
          let buffers = filled rows in
          equal values (Tensors.eval ~buffers rows) (Tensors.eval ~buffers view));
    ]

(* Variables and devices *)

let bound_var name =
  List.find
    (fun n -> Shape.is_bound_var n && Ops.expr n = name)
    (Ops.toposort ~calls:Enter (program "variable_two"))

(* variable_two binds v to 4 and w to 7; [rebound value] binds, in w's place,
   another variable named v, of another range, to [value]. *)
let rebound value =
  Ops.substitute ~calls:Skip ~pass:Fixed_point (program "variable_two")
    [
      ( bound_var "w",
        Shape.bind
          (Ops.variable "v" (`Int (Bigint.of_int 0)) (`Int (Bigint.of_int 10)))
          (`Int (Bigint.of_int value)) );
    ]

(* add reads the buffers of slots 2 and 3. *)
let several_devices () =
  let big = program "add" in
  let moved b =
    match Ops.arg b with
    | Param ({ slot = 3; _ } as p) ->
        Some
          ( b,
            Ops.replace ~arg:(Param { p with device = Some "CPU:1" }) b
          )
    | _ -> None
  in
  let input = require_some (List.find_map moved (buffers_of big)) in
  let across = Ops.substitute ~calls:Skip ~pass:Fixed_point big [ input ] in
  raises_match (Exn.invalid_arg ~sub:"same device") (fun () ->
      Schedule.create_linear_with_vars across)

(* Rows of [values] on each of [devices], each summed against a table of no
   device, [2^j] along the row, as a rotary embedding reads its angles: the
   table computes a transcendental function and is broadcast over the rows, so
   it is stored, and every device's sums read the one table. *)
let table_sums devices =
  let m = 8 and k = 4 in
  let table =
    Shape.expand
      (Shape.reshape (Ops.exp2 (Shape.arange ~dtype:Float32 k)) [ Int 1; Int k ])
      [ Int m; Int k ]
  in
  let n = List.length devices in
  let sums slot d =
    let values =
      Shape.reshape (Ops.new_buffer ~slot d (m * k) Float32) [ Int m; Int k ]
    in
    let out = Ops.new_buffer ~slot:(slot + n) d m Float32 in
    let total = Shape.rop (Ops.mul values table) Op.Add [ 1 ] in
    Ops.after out [ Ops.store out (Shape.reshape total [ Int m ]) ]
  in
  Ops.sink (List.mapi sums devices)

(* The devices of the storage each call of [linear] reads and writes. *)
let call_devices linear =
  List.map
    (fun call ->
      List.sort_uniq String.compare
        (List.filter_map Ops.device (List.tl (Ops.src call))))
    (Ops.src linear)

(* On one device, the table is stored once and read by the sums; on two, each
   device stores its own and its sums read it there. Calls are compared in no
   order. *)
let deviceless_stage () =
  let calls devices =
    List.sort compare
      (call_devices
         (fst (Schedule.create_linear_with_vars (table_sums devices))))
  in
  equal (list (list string)) ~msg:"one device" [ [ "CPU" ]; [ "CPU" ] ]
    (calls [ "CPU" ]);
  equal (list (list string)) ~msg:"two devices"
    [ [ "CPU" ]; [ "CPU" ]; [ "CPU:1" ]; [ "CPU:1" ] ]
    (calls [ "CPU"; "CPU:1" ])

(* Kernels each storing windows of four floats of a buffer, each window
   starting at a variable [a] or [b] of [0, 12] bound to a value: [b] at 5,
   [a] at 1 plus [b] at 2, and [a] at 3 plus [b] at 5. The schedule meets [b]
   at 5 before [a] at 3. *)
let windows_at () =
  let x = Ops.new_buffer ~slot:0 cpu 16 Float32 in
  let at name value =
    let v =
      Ops.variable name (`Int (Bigint.of_int 0)) (`Int (Bigint.of_int 12))
    in
    let start = Shape.bind v (`Int (Bigint.of_int value)) in
    Shape.shrink x [ Some (Ops.Sym start, Ops.Sym Ops.O.(start + int 4)) ]
  in
  let stored slot value =
    let out = Ops.new_buffer ~slot cpu 4 Float32 in
    Ops.after out [ Ops.store out value ]
  in
  Ops.sink
    [
      stored 1 (at "b" 5);
      stored 2 (Ops.add (at "a" 1) (at "b" 2));
      stored 3 (Ops.add (at "a" 3) (at "b" 5));
    ]

let variables =
  group "create_linear_with_vars › rules"
    [
      test
        "two variables of one name bound to two values are passed by the \
         calls that read them"
        (fun () ->
          let linear, vars = Schedule.create_linear_with_vars (rebound 7) in
          equal (list (pair string int)) ~msg:"no run binds v" [] vars;
          let passed call =
            List.filter_map
              (fun s ->
                match Ops.arg s with
                | Param { bound = Some x; _ } when Shape.is_bound_var s ->
                    Some (Dtype.Value.to_int x)
                | _ -> None)
              (Ops.src_without_body call)
          in
          equal (list (list int)) ~msg:"each call its own value" [ [ 4 ]; [ 7 ] ]
            (List.map passed (Ops.src linear));
          let named call =
            List.filter_map
              (fun v -> match Ops.arg v with Param p -> p.name | _ -> None)
              (Shape.variables (Ops.body call))
          in
          equal (list (list string)) ~msg:"read by no name" [ []; [] ]
            (List.map named (Ops.src linear)));
      test "two variables of one name bound to one value are one" (fun () ->
          equal
            (list (pair string int))
            [ ("v", 4) ]
            (snd (Schedule.create_linear_with_vars (rebound 4))));
      test
        "kernels that read two names bound to several values alike are one \
         program, whatever order the schedule met the values in"
        (fun () ->
          let linear, _ = Schedule.create_linear_with_vars (windows_at ()) in
          let scalars k =
            List.filter
              (fun u -> Ops.op u = Param && Ops.addrspace u = Some Dtype.Alu)
              (Ops.toposort ~calls:Skip k)
          in
          let kernels =
            List.filter_map
              (fun n ->
                if Ops.op n = Call && Ops.op (Ops.nth n 0) = Sink then
                  Some (Ops.nth n 0)
                else None)
              (Ops.toposort ~calls:Skip linear)
          in
          match List.filter (fun k -> List.length (scalars k) = 2) kernels with
          | [ a; b ] -> equal uop ~msg:"the kernels" a b
          | ks -> failf "%d kernels of two scalars, not 2" (List.length ks));
      test "a kernel on buffers of two devices is refused" several_devices;
      test "one expression on two devices of a target is one kernel" (fun () ->
          let on slot d =
            let x = Ops.new_buffer ~slot d 4 Float32 in
            let out = Ops.new_buffer ~slot:(slot + 2) d 4 Float32 in
            Ops.after out [ Ops.store out Ops.O.((x * x) + float 1.) ]
          in
          let linear, _ =
            Schedule.create_linear_with_vars
              (Ops.sink [ on 0 "CPU:0"; on 1 "CPU:1" ])
          in
          match Ops.src linear with
          | [ a; b ] -> equal uop ~msg:"the kernels" (Ops.body a) (Ops.body b)
          | calls -> failf "%d calls, not 2" (List.length calls));
      test
        "a stored value of no device lives on each device whose kernels read \
         it"
        deviceless_stage;
    ]

(* Loops of calls *)

(* A scan of [n] trips, three by default, around a range numbered [axis]: a
   carry [c] of four floats, updated in place, and rows of four of [xs] and
   [ys]. Each trip stores [c * 2] into its row of [ys], then
   adds its row of [xs] to [c]. *)
let scan_loop ?(axis = 100) ?(n = 3) () =
  let k = 4 in
  let p slot = Shape.param ~shape:[ Int k ] ~device:cpu slot Float32 in
  let body =
    Ops.sink
      [
        Ops.store (p 0) Ops.O.(p 0 + p 1);
        Ops.store (p 2) Ops.O.(p 0 * float 2.);
      ]
  in
  let c = Ops.new_buffer cpu k Float32
  and xs = Ops.new_buffer cpu (n * k) Float32
  and ys = Ops.new_buffer cpu (n * k) Float32 in
  let r = Ops.range ~axis_type:Loop (Int n) [ axis ] in
  let row b =
    Shape.shrink b
      [ Some (Sym Ops.O.(r * int k), Sym Ops.O.((r * int k) + int k)) ]
  in
  let e =
    Ops.end_ (Ops.call ~precompile:true body [ c; row xs; row ys ]) [ r ]
  in
  (Ops.sink [ Ops.after c [ e ]; Ops.after ys [ e ] ], r, (c, xs, ys))

let ops_of us = List.map (fun u -> Op.name (Ops.op u)) us

(* [moving r a] is whether the argument [a] is a view that moves with [r]. *)
let moving r a = Ops.Nodes.mem r (Ops.ranges a)

let scan_linear () =
  let big, r, (c, xs, ys) = scan_loop () in
  let linear, _ = Schedule.create_linear_with_vars ~capturing:true big in
  match Ops.src linear with
  | [ e ] -> (
      equal (list string) ~msg:"a loop" [ "END" ] (ops_of [ e ]);
      equal (list uop) ~msg:"over the range" [ r ] (List.tl (Ops.src e));
      let body = Ops.nth e 0 in
      equal (list string) ~msg:"of a linear" [ "LINEAR" ] (ops_of [ body ]);
      match Ops.src body with
      | [ y; c' ] ->
          let storage a = Ops.buf_uop (Ops.base a) in
          equal (list uop) ~msg:"the first call stores the row of ys" [ ys ]
            (List.filter_map
               (fun a -> if moving r a then Some (storage a) else None)
               (Ops.src_without_body y));
          equal (list uop) ~msg:"the second adds the row of xs to c" [ c; xs ]
            (List.map storage (Ops.src_without_body c'));
          equal (list bool) ~msg:"whose row moves" [ false; true ]
            (List.map (moving r) (Ops.src_without_body c'))
      | calls -> failf "%d calls in the loop" (List.length calls))
  | entries -> failf "%d entries" (List.length entries)

(* A loop of three trips around a call whose body stores its scalar parameter 0,
   the argument [2r + 1], into its row of ys, its parameter 1. *)
let trip_linear () =
  let index = Shape.param ~addrspace:Alu 0 Int32
  and y = Shape.param ~shape:[ Int 1 ] ~device:cpu 1 Int32 in
  let body = Ops.sink [ Ops.store y (Shape.reshape index [ Int 1 ]) ] in
  let ys = Ops.new_buffer cpu 12 Int32 in
  let r = Ops.range ~axis_type:Loop (Int 3) [ 100 ] in
  let trip = Ops.O.((r * int 2) + int 1) in
  let row =
    Shape.shrink ys
      [ Some (Sym Ops.O.(r * int 4), Sym Ops.O.((r * int 4) + int 1)) ]
  in
  let e = Ops.end_ (Ops.call ~precompile:true body [ trip; row ]) [ r ] in
  let linear, _ =
    Schedule.create_linear_with_vars ~capturing:true
      (Ops.sink [ Ops.after ys [ e ] ])
  in
  match Ops.src linear with
  | [ e ] -> (
      equal (list string) ~msg:"a loop" [ "END" ] (ops_of [ e ]);
      match Ops.src (Ops.nth e 0) with
      | [ k ] ->
          equal uop ~msg:"the trip, past the row" trip
            (List.nth (Ops.src_without_body k) 1);
          equal (list uop) ~msg:"the kernel's variable, in the trip's slot"
            [ Shape.param ~addrspace:Alu 1 Int32 ]
            (Shape.variables (Ops.body k));
          equal bool ~msg:"the kernel reads no range" false
            (Ops.Nodes.mem r (Ops.backward_slice ~calls:Skip (Ops.body k)))
      | calls -> failf "%d calls in the loop" (List.length calls))
  | entries -> failf "%d entries" (List.length entries)

(* Two loops of three trips, each around a call whose body stores its scalar
   parameter 0 into its row of its ys, the argument the variable v bound to 4
   in one loop and 7 in the other. *)
let bound_loops_linear () =
  let loop (axis, value) =
    let index = Shape.param ~addrspace:Alu 0 Int32
    and y = Shape.param ~shape:[ Int 1 ] ~device:cpu 1 Int32 in
    let body = Ops.sink [ Ops.store y (Shape.reshape index [ Int 1 ]) ] in
    let ys = Ops.new_buffer cpu 12 Int32 in
    let r = Ops.range ~axis_type:Loop (Int 3) [ axis ] in
    let v =
      Shape.bind
        (Ops.variable ~dtype:Int32 "v" (Dtype.Value.of_int 0)
           (Dtype.Value.of_int 10))
        (`Int (Bigint.of_int value))
    in
    let row =
      Shape.shrink ys
        [ Some (Sym Ops.O.(r * int 4), Sym Ops.O.((r * int 4) + int 1)) ]
    in
    Ops.after ys [ Ops.end_ (Ops.call ~precompile:true body [ v; row ]) [ r ] ]
  in
  let linear, vars =
    Schedule.create_linear_with_vars ~capturing:true
      (Ops.sink (List.map loop [ (100, 4); (101, 7) ]))
  in
  equal (list (pair string int)) ~msg:"no run binds v" [] vars;
  let passed e =
    match Ops.src (Ops.nth e 0) with
    | [ k ] ->
        let slot v =
          match Ops.arg v with
          | Param p -> (p.name, p.slot)
          | _ -> failf "a variable without a parameter"
        in
        equal
          (list (pair (option string) int))
          ~msg:"the kernel's value, nameless, past its row" [ (None, 1) ]
          (List.map slot (Shape.variables (Ops.body k)));
        (match Ops.arg (List.nth (Ops.src_without_body k) 1) with
        | Param { bound = Some x; _ } -> Dtype.Value.to_int x
        | _ -> failf "the call passes no bound variable")
    | calls -> failf "%d calls in the loop" (List.length calls)
  in
  equal (list int) ~msg:"each loop's call its value" [ 4; 7 ]
    (List.sort compare (List.map passed (Ops.src linear)))

(* The values each kernel of [linear], at any depth, binds its named
   variables to, by name, where a name is bound to two. *)
let names_bound_twice linear =
  List.concat_map
    (fun c ->
      if Ops.op c <> Call || Ops.op (Ops.body c) <> Sink then []
      else
        let binds =
          List.sort_uniq compare
            (List.filter_map
               (fun u ->
                 match (Ops.op u, Ops.arg u) with
                 | Param, Param { name = Some n; bound = Some v; _ } ->
                     Some (n, Dtype.Value.to_int v)
                 | _ -> None)
               (Ops.toposort ~calls:Skip (Ops.body c)))
        in
        let twice (n, _) =
          List.length (List.filter (fun (m, _) -> m = n) binds) > 1
        in
        List.filter twice binds)
    (Ops.toposort ~calls:Skip linear)

(* A loop of two trips inside a loop's body of three, whose stores add a
   variable [k] bound to 0 in one and 1 in the other, beside a store of the
   outer body that binds it to 2. *)
let nested_bound () =
  let k = Ops.variable "k" (`Int Bigint.zero) (`Int (Bigint.of_int 2)) in
  let at j = Ops.cast (Shape.bind k (`Int (Bigint.of_int j))) Float32 in
  let q slot = Shape.param ~shape:[ Int 4 ] ~device:cpu slot Float32 in
  let inner =
    Ops.sink
      [
        Ops.store (q 0) Ops.O.((q 0 * float 2.) + at 0);
        Ops.store (q 1) Ops.O.(Ops.exp2 (q 1) + at 1);
      ]
  in
  let r' = Ops.range ~axis_type:Loop (Int 2) [ 101 ]
  and r = Ops.range ~axis_type:Loop (Int 3) [ 100 ] in
  let e' = Ops.end_ (Ops.call ~precompile:true inner [ q 0; q 1 ]) [ r' ] in
  let outer =
    Ops.sink
      [
        Ops.after (q 0) [ e' ];
        Ops.after (q 1) [ e' ];
        Ops.store (q 2) Ops.O.(q 2 + at 2);
      ]
  in
  let bufs = List.init 3 (fun _ -> Ops.new_buffer cpu 4 Float32) in
  let e = Ops.end_ (Ops.call ~precompile:true outer bufs) [ r ] in
  let big = Ops.sink (List.map (fun b -> Ops.after b [ e ]) bufs) in
  equal (list (pair string int)) ~msg:"names bound twice in one kernel" []
    (names_bound_twice (unplanned big));
  computes big

(* The scan's body holds a loop of its own: each row of xs, of eight floats, is
   added to the carry in two halves, one inner trip each. *)
let nested_linear () =
  let k = 4 and n = 3 in
  let q slot = Shape.param ~shape:[ Int k ] ~device:cpu slot Float32 in
  let inner = Ops.sink [ Ops.store (q 0) Ops.O.(q 0 + q 1) ] in
  let p0 = Shape.param ~shape:[ Int k ] ~device:cpu 0 Float32
  and p1 = Shape.param ~shape:[ Int (2 * k) ] ~device:cpu 1 Float32 in
  let r' = Ops.range ~axis_type:Loop (Int 2) [ 101 ]
  and r = Ops.range ~axis_type:Loop (Int n) [ 100 ] in
  let row r w b =
    Shape.shrink b
      [ Some (Sym Ops.O.(r * int w), Sym Ops.O.((r * int w) + int w)) ]
  in
  let outer =
    Ops.sink
      [
        Ops.after p0
          [
            Ops.end_
              (Ops.call ~precompile:true inner [ p0; row r' k p1 ])
              [ r' ];
          ];
      ]
  in
  let c = Ops.new_buffer cpu k Float32
  and xs = Ops.new_buffer cpu (n * 2 * k) Float32 in
  let e =
    Ops.end_ (Ops.call ~precompile:true outer [ c; row r (2 * k) xs ]) [ r ]
  in
  let linear, _ =
    Schedule.create_linear_with_vars ~capturing:true
      (Ops.sink [ Ops.after c [ e ] ])
  in
  match Ops.src linear with
  | [ e ] -> (
      equal (list uop) ~msg:"the outer loop" [ r ] (List.tl (Ops.src e));
      match Ops.src (Ops.nth e 0) with
      | [ e' ] ->
          equal (list string) ~msg:"holds a loop" [ "END" ] (ops_of [ e' ]);
          equal (list uop) ~msg:"the inner loop" [ r' ] (List.tl (Ops.src e'))
      | entries -> failf "%d entries in the outer loop" (List.length entries))
  | entries -> failf "%d entries" (List.length entries)

(* Whoever makes a loop numbers its range from a counter of its own process.
   Five trips keep the body apart from the other tests' loops. *)
let renumbered_loop () =
  let verdicts axis =
    let big, _, _ = scan_loop ~axis ~n:5 () in
    with_settings ~debug:3 ~scache:1 (fun () ->
        ignore (Schedule.create_linear_with_vars ~capturing:true big));
    List.map (fun (_, verdict, _) -> verdict) (reports ())
  in
  is_true ~msg:"the first is new" (List.mem "CACHE MISS" (verdicts 100));
  is_true ~msg:"under another number, every body hits"
    (List.for_all (String.equal " cache hit") (verdicts 7))

(* A scan's body holding two loops of six trips, around ranges numbered [a] and
   [b]: each adds a row of its own buffer to the carry. *)
let two_loops (a, b) =
  let k = 4 and n = 6 in
  let q slot = Shape.param ~shape:[ Int k ] ~device:cpu slot Float32 in
  let inner = Ops.sink [ Ops.store (q 0) Ops.O.(q 0 + q 1) ] in
  let p0 = Shape.param ~shape:[ Int k ] ~device:cpu 0 Float32 in
  let loop axis slot =
    let r = Ops.range ~axis_type:Loop (Int n) [ axis ] in
    let p = Shape.param ~shape:[ Int (n * k) ] ~device:cpu slot Float32 in
    let row =
      Shape.shrink p
        [ Some (Sym Ops.O.(r * int k), Sym Ops.O.((r * int k) + int k)) ]
    in
    Ops.end_ (Ops.call ~precompile:true inner [ p0; row ]) [ r ]
  in
  let outer = Ops.sink [ Ops.after p0 [ loop a 1; loop b 2 ] ] in
  let c = Ops.new_buffer cpu k Float32
  and xs = Ops.new_buffer cpu (n * k) Float32
  and ys = Ops.new_buffer cpu (n * k) Float32 in
  let e = Ops.call ~precompile:true outer [ c; xs; ys ] in
  Ops.sink [ Ops.after c [ e ] ]

(* Under some of these numberings, a loop's number by order is the other loop's
   number: the two trade numbers, or one takes the other's and the other a new
   one. Each run makes four schedules: the inner body once per loop, the outer
   body, and the function. *)
let renumbered_loops () =
  let verdicts axes =
    with_settings ~debug:3 ~scache:1 (fun () ->
        ignore
          (Schedule.create_linear_with_vars ~capturing:true (two_loops axes)));
    List.map (fun (_, verdict, _) -> verdict) (reports ())
  in
  ignore (verdicts (7, 8));
  List.iter
    (fun ((a, b) as axes) ->
      equal (list string)
        ~msg:(Printf.sprintf "numbered %d, %d" a b)
        (List.init 4 (fun _ -> " cache hit"))
        (verdicts axes))
    [ (1, 0); (0, 1); (1, 2); (2, 1) ]

(* A body that hits under another numbering is given back its own loops. *)
let own_loops () =
  let axes linear =
    List.filter_map
      (fun u -> if Ops.op u = Range then Some (Ops.axis_id u) else None)
      (Ops.toposort ~calls:Enter linear)
  in
  let schedule loops =
    with_settings ~debug:0 ~scache:1 (fun () ->
        fst (Schedule.create_linear_with_vars ~capturing:true (two_loops loops)))
  in
  ignore (schedule (7, 8));
  let held = axes (schedule (13, 14)) in
  equal (list (list int)) ~msg:"its own loops" [ [ 13 ]; [ 14 ] ]
    (List.filter (fun a -> List.mem a [ [ 13 ]; [ 14 ] ]) held
    |> List.sort_uniq compare);
  equal (list (list int)) ~msg:"none of the first's" []
    (List.filter (fun a -> List.mem a [ [ 7 ]; [ 8 ] ]) held)

let loops =
  group "create_linear_with_vars › loops of calls"
    [
      test "a loop of a precompiled call is a loop of its body's calls"
        scan_linear;
      test "a loop's call passes its kernel the trip as a scalar argument"
        trip_linear;
      test
        "loops whose calls read one name at two values each pass their own"
        bound_loops_linear;
      test "a loop inside a loop's body keeps its own end" nested_linear;
      test
        "a loop inside a loop's body passes the values of one name it binds \
         apart"
        nested_bound;
      test "a loop whose range has another number is the same body"
        renumbered_loop;
      test "a body whose loops trade numbers is the same body" renumbered_loops;
      test "a body that hits under other numbers keeps its own loops" own_loops;
    ]

(* Schedules on disk

   A child schedules matmul with DEBUG at 3 and prints the schedule after the
   line scheduling printed, whose verdict says whether it was made or read
   back. *)

(* Whether the child read its schedule back, and the schedule. *)
let scheduled ?(env = []) db =
  match
    Disk_cache.child ~env:(("DEBUG", "3") :: env) ~cachedb:db "schedule"
  with
  | Ok out ->
      let lines = String.split_on_char '\n' out in
      let reports, graph =
        List.partition (String.starts_with ~prefix:"scheduled ") lines
      in
      let hit =
        List.exists
          (fun l -> Str.string_match (Str.regexp ".*| +cache hit") l 0)
          reports
      in
      Ok (hit, String.concat "\n" graph)
  | Error err -> Error err

let read_back db =
  match scheduled db with
  | Ok (hit, graph) ->
      equal bool ~msg:"read back" true hit;
      graph
  | Error err -> failf "the child failed: %s" err

let made db =
  match scheduled db with
  | Ok (hit, graph) ->
      equal bool ~msg:"made" false hit;
      graph
  | Error err -> failf "the child failed: %s" err

let reads_back () =
  let db = Disk_cache.fresh () in
  let graph = made db in
  equal string graph (read_back db)

let misses_on env () =
  let db = Disk_cache.fresh () in
  ignore (made db);
  match scheduled ~env db with
  | Ok (hit, _) -> equal bool ~msg:"read back" false hit
  | Error err -> failf "the child failed: %s" err

let recovers damage () =
  let db = Disk_cache.fresh () in
  let graph = made db in
  Disk_cache.damage db damage;
  equal string ~msg:"made again" graph (made db);
  equal string ~msg:"then read back" graph (read_back db)

let ignores_other_builds () =
  let db = Disk_cache.fresh () in
  ignore (made db);
  Disk_cache.damage db Disk_cache.of_another_build;
  ignore (made db)

let races () =
  let db = Disk_cache.fresh () in
  let children =
    List.init 4 (fun _ ->
        Disk_cache.start ~env:[ ("DEBUG", "3") ] ~cachedb:db "schedule")
  in
  let graphs = List.map Disk_cache.finish children in
  let graph = read_back db in
  List.iter
    (fun g ->
      match g with
      | Ok out ->
          equal string graph
            (String.concat "\n"
               (List.filter
                  (fun l -> not (String.starts_with ~prefix:"scheduled " l))
                  (String.split_on_char '\n' out)))
      | Error err -> failf "a child failed: %s" err)
    graphs

let memory_only () =
  let db = Disk_cache.fresh () in
  let env = [ ("SCACHE", "1") ] in
  (match scheduled ~env db with
  | Ok (hit, _) -> equal bool ~msg:"made" false hit
  | Error err -> failf "the child failed: %s" err);
  equal (list string) ~msg:"entries" [] (Disk_cache.entries db);
  match scheduled ~env db with
  | Ok (hit, _) -> equal bool ~msg:"made again" false hit
  | Error err -> failf "the child failed: %s" err

(* Another value of each setting and variable that shapes what compilation
   makes, as the environment holds it: a flag flipped, a number plus one, and a
   string from this table, which a string declared later must join. *)
let other_values () =
  let strings =
    [
      ("DEFAULT_FLOAT", "half");
      ("DEFAULT_INT", "long");
      ("EMULATED_DTYPES", "long");
      ("JITBEAM", "1");
    ]
  in
  List.map
    (fun (name, value) ->
      match (value, int_of_string_opt value, float_of_string_opt value) with
      | "true", _, _ -> (name, "0")
      | "false", _, _ -> (name, "1")
      | _, Some n, _ -> (name, string_of_int (n + 1))
      | _, None, Some x -> (name, string_of_float (x +. 1.))
      | _, None, None -> (
          match List.assoc_opt name strings with
          | Some other -> (name, other)
          | None -> failf "%s holds a string: give it another value" name))
    (Setting.shaping ())

let on_disk =
  group "create_linear_with_vars › schedules are kept on disk"
    [
      test "a schedule made by one process is read back by the next" reads_back;
      group "a schedule made under one setting is not read back under another"
        (List.map
           (fun (name, value) -> test name (misses_on [ (name, value) ]))
           (other_values ()));
      group "a damaged entry is made anew, and replaced"
        [
          test "truncated" (recovers Disk_cache.truncated);
          test "holding no schedule" (recovers Disk_cache.not_a_graph);
        ];
      test "an entry of another build of the library is not read back"
        ignores_other_builds;
      test "processes making one schedule at once all get it" races;
      test "with SCACHE at 1, nothing is kept on disk" memory_only;
    ]

(* Copies into runs

   A copy stored into a run of consecutive elements of storage on its device
   lands there: the schedule is one copy into a view of the storage, from the
   run's first element. A copy stored into elements that are no run lands in
   storage of its own, which a kernel stores. *)

(* A call's body and destination, as their operations, and the first element
   and size of a destination that shrinks a buffer, read through a bitcast of
   the shrink. *)
let destination c =
  let name u = Format.asprintf "%a" Op.pp (Ops.op u) in
  let d = Ops.nth c 1 in
  let shrunk = if Ops.op d = Bitcast then Ops.nth d 0 else d in
  let run =
    if Ops.op shrunk <> Shrink then []
    else
      match Shape.marg shrunk with
      | Shrink [ (Ops.Int lo, Ops.Int k) ] -> [ (lo, k) ]
      | _ -> []
  in
  (name (Ops.nth c 0), name d, run)

(* The calls of the schedule that stores into [into], a view of a buffer of 64
   floats on CPU:1, a copy of a buffer of the host, each as {!destination}
   gives it. *)
let copied_into ?(dt = Dtype.Float32) into =
  let dst = Ops.new_buffer "CPU:1" 64 Float32 in
  let view = into (Shape.reshape dst [ Int 8; Int 8 ]) in
  let n = List.fold_left ( * ) 1 (List.map (function Ops.Int k -> k | Sym _ -> 0) (Shape.shape view)) in
  let src = Ops.new_buffer "CPU" n dt in
  let value =
    Ops.copy_to_device (Shape.reshape src (Shape.shape view)) "CPU:1"
  in
  let linear, _ =
    Schedule.create_linear_with_vars ~capturing:true
      (Ops.sink [ Ops.after dst [ Ops.store view value ] ])
  in
  List.map destination (Ops.src linear)

(* {!copied_into}'s calls for a buffer of 4 floats on CPU:1 filled by a copy
   from the host, then element 2 of it stored with a copy of one element. *)
let one_copied () =
  let dst = Ops.new_buffer "CPU:1" 4 Float32 in
  let whole = Ops.new_buffer "CPU" 4 Float32 in
  let one = Ops.new_buffer "CPU" 1 Float32 in
  let filled = Ops.after dst [ Ops.store dst (Ops.copy_to_device whole "CPU:1") ] in
  let into = Shape.shrink filled [ Some (Int 2, Int 3) ] in
  let value = Ops.copy_to_device one "CPU:1" in
  let linear, _ =
    Schedule.create_linear_with_vars ~capturing:true
      (Ops.sink [ Ops.after filled [ Ops.store into value ] ])
  in
  List.map destination (Ops.src linear)

let calls = list (triple string string (list (pair int int)))
let store = Format.asprintf "%a" Op.pp Store
let sink = Format.asprintf "%a" Op.pp Sink
let shrink = Format.asprintf "%a" Op.pp Shrink
let buffer = Format.asprintf "%a" Op.pp Buffer
let bitcast = Format.asprintf "%a" Op.pp Bitcast

let runs =
  group "copies into runs"
    [
      test "a copy stored into rows of storage on its device copies into them"
        (fun () ->
          equal calls
            [ (store, shrink, [ (16, 16) ]) ]
            (copied_into (fun g -> Shape.shrink g [ Some (Int 2, Int 4); None ])));
      test "a copy stored into a first row copies into it" (fun () ->
          equal calls
            [ (store, shrink, [ (0, 16) ]) ]
            (copied_into (fun g -> Shape.shrink g [ Some (Int 0, Int 2); None ])));
      test "a copy of one element stored into filled storage copies into it"
        (fun () ->
          equal calls
            [ (store, buffer, []); (store, shrink, [ (2, 1) ]) ]
            (one_copied ()));
      test "a copy stored into columns lands in storage of its own" (fun () ->
          match
            copied_into (fun g -> Shape.shrink g [ None; Some (Int 2, Int 4) ])
          with
          | [ (copy, _, _); (kernel, into, _) ] ->
              equal (triple string string string) (store, sink, buffer)
                (copy, kernel, into)
          | calls -> failf "%d calls, not a copy and a kernel" (List.length calls));
      test "a copy stored into a run of a bitcast copies into a view of it"
        (fun () ->
          equal calls
            [ (store, bitcast, [ (16, 16) ]) ]
            (copied_into ~dt:Dtype.Int32 (fun g ->
                 Shape.shrink (Ops.bitcast g Int32) [ Some (Int 2, Int 4); None ])));
    ]

(* Calls that run as one

   Kernel calls of one device and domain, with no reduction, that neither
   touches what the other writes, run as one call: in a loop's body whatever
   their domain, at the top level when they compute one element. A call moves up
   to its group's first call only past calls it does not conflict with. *)

(* The one element of a buffer of [dt] on [device]. *)
let element ?(device = cpu) ?(dt = Dtype.Float32) () =
  Shape.reshape (Ops.new_buffer device 1 dt) []

(* The store of [v] into [o], as a program writes it. *)
let into o v = Ops.after o [ Ops.store o v ]

(* The calls of [big]'s schedule, unplanned. *)
let calls_of big = Ops.src (unplanned big)

(* The calls in the body of [big]'s one loop. *)
let calls_in_loop big =
  match calls_of big with
  | [ e ] when Ops.op e = End -> Ops.src (Ops.nth e 0)
  | entries -> failf "%d entries, not one loop" (List.length entries)

(* A scan of three trips whose body stores [stores], each a parameter slot of
   [size] floats and the value stored into it, made of the parameters by
   [value]: [p k] is parameter [k]. Each slot is a buffer of its own. *)
let looped ?(size = 4) ?(sizes = []) stores =
  let size_of k = Option.value (List.assoc_opt k sizes) ~default:size in
  let p k = Shape.param ~shape:[ Int (size_of k) ] ~device:cpu k Float32 in
  let slots =
    List.sort_uniq compare
      (List.concat_map (fun (k, reads, _) -> k :: reads) stores)
  in
  let body =
    Ops.sink (List.map (fun (k, _, value) -> Ops.store (p k) (value p)) stores)
  in
  let bufs =
    List.map (fun k -> (k, Ops.new_buffer cpu (size_of k) Float32)) slots
  in
  let n = 1 + List.fold_left max 0 slots in
  let args =
    List.init n (fun k ->
        match List.assoc_opt k bufs with
        | Some b -> b
        | None -> Ops.new_buffer cpu 1 Float32)
  in
  let r = Ops.range ~axis_type:Loop (Int 3) [ 100 ] in
  let e = Ops.end_ (Ops.call ~precompile:true body args) [ r ] in
  Ops.sink
    (List.map (fun (k, _, _) -> Ops.after (List.nth args k) [ e ]) stores)

(* [two_chains n] is [n] outputs of one element in two chains: output [j] reads
   output [j - 2]. Each output joins the one before it, which it does not read,
   and waits for the pair before. *)
let two_chains n =
  let x = element () in
  let rec go j prev prev' acc =
    if j = n then Ops.sink (List.rev acc)
    else
      let o = into (element ()) Ops.O.(prev + float 1.) in
      go (j + 1) prev' o (o :: acc)
  in
  go 0 x x []

(* [words f] is the words [f ()] allocates. *)
let words f =
  let before = Gc.minor_words () in
  ignore (Sys.opaque_identity (f ()));
  Gc.minor_words () -. before

(* [add_one ~read ~write] is the call of a kernel of one element that stores
   one more than the element of [read] into [write]. *)
let add_one =
  let kernel =
    lazy
      (let x = element () and o = element () in
       match calls_of (Ops.sink [ into o Ops.O.(x + float 1.) ]) with
       | [ call ] ->
           let slot_of b =
             let rec go i = function
               | [] -> failf "no argument"
               | a :: rest ->
                   if Ops.storage_base a == b then i else go (i + 1) rest
             in
             go 0 (Ops.src_without_body call)
           in
           (call, slot_of (Ops.storage_base o), slot_of (Ops.storage_base x))
       | calls -> failf "%d calls" (List.length calls))
  in
  fun ~read ~write ->
    let call, written, read_slot = Lazy.force kernel in
    let args = Array.make 2 read in
    args.(written) <- write;
    args.(read_slot) <- read;
    Ops.replace call ~src:(Ops.body call :: Array.to_list args)

(* [chained_calls n] is a schedule of [n] calls, as {!two_chains} schedules them
   unmerged: call [j] adds one to the element call [j - 2] writes. *)
let chained_calls n =
  let bufs = Array.init (n + 2) (fun _ -> Ops.new_buffer cpu 1 Float32) in
  Ops.v Op.Linear
    ~src:(List.init n (fun j -> add_one ~read:bufs.(j) ~write:bufs.(j + 2)))

(* A program of outputs of one type and shape: each output [j] stores [x + 2],
   [x * 2] or [max x 2] of its own input or, where its source names one, of an
   earlier output, which it then waits for. *)
type outputs = {
  dt : Dtype.t;
  shape : int list;
  steps : (int option * int) list;  (** the earlier output read, the op *)
}

let outputs_gen =
  let open Gen in
  let step j =
    pair
      (if j = 0 then constant None
       else
         frequency
           [ (2, constant None); (1, map Option.some (int_range 0 (j - 1))) ])
      (int_range 0 2)
  in
  let steps n =
    let rec go j acc =
      if j = n then map List.rev acc
      else go (j + 1) (bind acc (fun acc -> map (fun s -> s :: acc) (step j)))
    in
    go 0 (constant [])
  in
  let dts = [ Dtype.Float32; Float64; Int32; Int64; Uint8; Int8 ] in
  map
    (fun ((dt, shape), steps) -> { dt; shape; steps })
    (pair
       (pair (of_list dts)
          (frequency
             [
               (6, constant []);
               (1, constant [ 1 ]);
               (1, constant [ 0 ]);
               (1, constant [ 3 ]);
             ]))
       (bind (int_range 1 6) steps))

let outputs_sink o =
  let n = List.fold_left ( * ) 1 o.shape in
  let shaped b = Shape.reshape b (List.map (fun d -> Ops.Int d) o.shape) in
  let two = Ops.const ~dtype:o.dt (`Int (Bigint.of_int 2)) in
  let apply k x =
    match k with
    | 0 -> Ops.add x two
    | 1 -> Ops.mul x two
    | _ -> Ops.maximum x two
  in
  let made =
    List.fold_left
      (fun made (r, k) ->
        let x =
          match r with
          | Some j -> List.nth made j
          | None -> shaped (Ops.new_buffer cpu n o.dt)
        in
        let out = shaped (Ops.new_buffer cpu n o.dt) in
        made @ [ into out (apply k x) ])
      [] o.steps
  in
  Ops.sink made

(* The calls a schedule's entry runs: a launch's members, or itself. *)
let members_of call =
  if Shape.is_launch (Ops.body call) then
    List.length (Shape.launch_members call)
  else 1

let merges_keep_values o =
  let big = outputs_sink o in
  let calls = calls_of big in
  let joined = List.map members_of calls in
  cover "two calls run as one" (List.mem 2 joined);
  cover "three or more calls run as one" (List.exists (fun k -> k >= 3) joined);
  cover "a call waits for another"
    (List.exists (fun (r, _) -> r <> None) o.steps);
  cover "outputs of several elements" (List.fold_left ( * ) 1 o.shape > 1);
  let buffers = filled big in
  let vars = [] in
  equal
    (list (triple int int Dtypes.value))
    (Tensors.writes ~buffers big)
    (Kernel_graphs.linear_writes ~vars ~buffers (unplanned big))

(* A loop of three trips whose body stores into slots [0] to [n - 1] in turn,
   [n] the steps: step [j] stores [x + 2], [x * 2] or [max x 2] into slot [j],
   [x] the sum of the slots it reads of its slot's size (its own slot if
   none), each read after the step that stores it when that step is earlier,
   plus, when it reads the trip, the scalar argument [2r + 1]. *)
type loop_program = {
  sizes : int list;  (** each slot's elements *)
  loop_steps : (int list * int * bool) list;
      (** the slots read, the op, whether it reads the trip *)
}

let loop_gen =
  let open Gen in
  let size = frequency [ (4, constant 4); (1, constant 2) ] in
  bind (int_range 2 4) (fun m ->
      let step =
        triple
          (list ~size:(int_range 0 2) (int_range 0 (m - 1)))
          (int_range 0 2)
          (frequency [ (2, constant false); (1, constant true) ])
      in
      map
        (fun (sizes, loop_steps) -> { sizes; loop_steps })
        (pair
           (list ~size:(constant m) size)
           (list ~size:(int_range 1 m) step)))

let loop_sink prog =
  let m = List.length prog.sizes in
  let size k = List.nth prog.sizes k in
  let p k = Shape.param ~shape:[ Int (size k) ] ~device:cpu k Float32 in
  let trip = Shape.param ~addrspace:Alu m Int32 in
  let two = Ops.float 2. in
  let apply k x =
    match k with
    | 0 -> Ops.add x two
    | 1 -> Ops.mul x two
    | _ -> Ops.maximum x two
  in
  let stores =
    List.fold_left
      (fun stores (reads, k, trips) ->
        let j = List.length stores in
        let read i =
          if i < j then Ops.after (p i) [ List.nth stores i ] else p i
        in
        let reads = List.filter (fun i -> size i = size j) reads in
        let x =
          match List.map read reads with
          | [] -> p j
          | r :: rs -> List.fold_left Ops.add r rs
        in
        let x =
          if not trips then x
          else
            Ops.add x
              (Shape.expand
                 (Shape.reshape (Ops.cast trip Float32) [ Int 1 ])
                 [ Int (size j) ])
        in
        stores @ [ Ops.store (p j) (apply k x) ])
      [] prog.loop_steps
  in
  let bufs = List.init m (fun k -> Ops.new_buffer cpu (size k) Float32) in
  let r = Ops.range ~axis_type:Loop (Int 3) [ 100 ] in
  let args = bufs @ [ Ops.O.((r * int 2) + int 1) ] in
  let e = Ops.end_ (Ops.call ~precompile:true (Ops.sink stores) args) [ r ] in
  Ops.sink (List.mapi (fun j _ -> Ops.after (List.nth bufs j) [ e ]) stores)

let loop_merges_keep_values prog =
  let big = loop_sink prog in
  let n = List.length prog.loop_steps in
  cover "calls in the loop run as one"
    (List.length (calls_in_loop big) < n);
  cover "a step overwrites a slot an earlier step reads"
    (List.exists
       (fun (j, (reads, _, _)) ->
         List.exists (fun i -> i > j && i < n) reads)
       (List.mapi (fun j s -> (j, s)) prog.loop_steps));
  cover "a step reads the trip"
    (List.exists (fun (_, _, t) -> t) prog.loop_steps);
  cover "slots of two sizes"
    (List.exists (fun s -> s <> List.hd prog.sizes) prog.sizes);
  computes big

(* A loop of three trips whose body stores into slots [0] to [n - 1], [n] the
   steps, and reads slot [n]: step [j] stores [x * 2], [exp2 x] or
   [max x 2] into slot [j], [x] its own slot or [exp2] of slot [n], which
   steps share, plus, where it binds one, a variable [k] bound to its value
   and cast to a float. *)
type bound_step = { op : int; shared : bool; value : int option }

let bound_gen =
  let open Gen in
  let step =
    map
      (fun ((op, shared), value) -> { op; shared; value })
      (pair
         (pair (int_range 0 2) bool)
         (frequency
            [ (1, constant None); (3, map Option.some (int_range 0 2)) ]))
  in
  list ~size:(int_range 2 4) step

let bound_sink steps =
  let n = List.length steps in
  let p k = Shape.param ~shape:[ Int 4 ] ~device:cpu k Float32 in
  let k = Ops.variable "k" (`Int Bigint.zero) (`Int (Bigint.of_int 2)) in
  let stored j s =
    let x = if s.shared then Ops.exp2 (p n) else p j in
    let x =
      match s.op with
      | 0 -> Ops.O.(x * float 2.)
      | 1 -> Ops.exp2 x
      | _ -> Ops.maximum x (Ops.float 2.)
    in
    match s.value with
    | None -> x
    | Some v ->
        Ops.O.(x + Ops.cast (Shape.bind k (`Int (Bigint.of_int v))) Float32)
  in
  let body =
    Ops.sink (List.mapi (fun j s -> Ops.store (p j) (stored j s)) steps)
  in
  let bufs = List.init (n + 1) (fun _ -> Ops.new_buffer cpu 4 Float32) in
  let r = Ops.range ~axis_type:Loop (Int 3) [ 100 ] in
  let e = Ops.end_ (Ops.call ~precompile:true body bufs) [ r ] in
  Ops.sink (List.mapi (fun j _ -> Ops.after (List.nth bufs j) [ e ]) steps)

let joins_keep_bindings steps =
  let big = bound_sink steps in
  let values =
    List.sort_uniq compare (List.filter_map (fun s -> s.value) steps)
  in
  let calls = calls_in_loop big in
  (* The values a call passes for the variables its body bound. *)
  let passed c =
    List.sort_uniq compare
      (List.filter_map
         (fun a ->
           match Ops.arg a with
           | Param { bound = Some v; _ } when Shape.is_bound_var a ->
               Some (Dtype.Value.to_int v)
           | _ -> None)
         (Ops.src_without_body c))
  in
  cover "a call passes one name at two values"
    (List.exists (fun c -> List.length (passed c) > 1) calls);
  cover "a loop's kernels read a name of one value" (List.length values = 1);
  cover "stores that share a computation bind one name to two values"
    (List.length values > 1
    && List.length
         (List.sort_uniq compare
            (List.filter_map
               (fun s -> if s.shared then s.value else None)
               steps))
       > 1);
  equal (list (pair string int)) ~msg:"names bound twice in one kernel" []
    (names_bound_twice (unplanned big));
  computes big

let merged_calls =
  group "create_linear_with_vars › launches"
    [
      test "independent calls of one domain in a loop run as one" (fun () ->
          let big =
            looped
              [
                (0, [ 0 ], fun p -> Ops.O.(p 0 * float 2.));
                (1, [ 1 ], fun p -> Ops.exp2 (p 1));
              ]
          in
          equal int 1 (List.length (calls_in_loop big)));
      test "calls that bind one name to two values run as one" (fun () ->
          (* Each adds [k], bound to 0 in one and 1 in the other: the call
             passes each value in a slot of its own. *)
          let k = Ops.variable "k" (`Int Bigint.zero) (`Int Bigint.one) in
          let plus j u =
            Ops.O.(u + Ops.cast (Shape.bind k (`Int (Bigint.of_int j))) Float32)
          in
          let big j =
            looped
              [
                (0, [ 0 ], fun p -> plus 0 Ops.O.(p 0 * float 2.));
                (1, [ 1 ], fun p -> plus j (Ops.exp2 (p 1)));
              ]
          in
          equal ~msg:"bound apart" int 1 (List.length (calls_in_loop (big 1)));
          equal ~msg:"bound alike" int 1 (List.length (calls_in_loop (big 0)));
          computes (big 1));
      test "loop bodies that differ in their bound values alone are one"
        (fun () ->
          let k = Ops.variable "k" (`Int Bigint.zero) (`Int Bigint.one) in
          let body j =
            let big =
              looped
                [
                  ( 0,
                    [ 0 ],
                    fun p ->
                      let v = Shape.bind k (`Int (Bigint.of_int j)) in
                      Ops.O.(p 0 + Ops.cast v Float32) );
                ]
            in
            List.map Ops.body (calls_in_loop big)
          in
          equal (list uop) (body 0) (body 1));
      test "calls of one element at the top level run as one" (fun () ->
          let x = element () and y = element () in
          let o = element () and o' = element () in
          let big =
            Ops.sink
              [ into o Ops.O.(x * float 2.); into o' Ops.O.(y + float 1.) ]
          in
          equal int 1 (List.length (calls_of big)));
      test "calls of many elements at the top level run apart" (fun () ->
          let buf () = Ops.new_buffer cpu 4 Float32 in
          let x = buf () and y = buf () and o = buf () and o' = buf () in
          let big =
            Ops.sink
              [ into o Ops.O.(x * float 2.); into o' Ops.O.(y + float 1.) ]
          in
          equal int 2 (List.length (calls_of big)));
      test "a call that reads what another writes runs apart" (fun () ->
          let x = element () and o = element () and o' = element () in
          let first = into o Ops.O.(x * float 2.) in
          let big = Ops.sink [ first; into o' Ops.O.(first + float 1.) ] in
          equal int 2 (List.length (calls_of big)));
      test "a call that overwrites what another reads runs apart" (fun () ->
          let big =
            looped
              [
                (0, [ 0; 1 ], fun p -> Ops.O.(p 0 + p 1));
                (1, [ 1 ], fun p -> Ops.O.(p 1 * float 2.));
              ]
          in
          equal int 2 (List.length (calls_in_loop big)));
      test "calls that write disjoint runs of one buffer run as one launch" (fun () ->
          let x = element () and y = element () in
          let o = Ops.new_buffer cpu 2 Float32 in
          let at i =
            Shape.reshape (Shape.shrink o [ Some (Int i, Int (i + 1)) ]) []
          in
          let big =
            Ops.sink
              [
                Ops.after o
                  [
                    Ops.store (at 0) Ops.O.(x * float 2.);
                    Ops.store (at 1) Ops.O.(y + float 1.);
                  ];
              ]
          in
          equal int 1 (List.length (calls_of big)));
      test "a call with a reduction joins a launch" (fun () ->
          let big =
            looped
              ~sizes:[ (2, 16) ]
              [
                ( 0,
                  [ 2 ],
                  fun p ->
                    Shape.reshape
                      (Shape.rop
                         (Shape.reshape (p 2) [ Int 4; Int 4 ])
                         Add [ 1 ])
                      [ Int 4 ] );
                (1, [ 1 ], fun p -> Ops.O.(p 1 * float 2.));
              ]
          in
          equal int 1 (List.length (calls_in_loop big)));
      test "calls on two devices run apart" (fun () ->
          let on device = element ~device () in
          let big =
            Ops.sink
              [
                into (on "CPU") Ops.O.(on "CPU" * float 2.);
                into (on "CPU:1") Ops.O.(on "CPU:1" + float 1.);
              ]
          in
          equal int 2 (List.length (calls_of big)));
      (* [b] reads what [a] writes and [d] what [b] writes; [c] is independent:
         it joins [a], and [d] cannot. *)
      test "a call moves up only past calls it does not conflict with"
        (fun () ->
          let x = element () and y = element () in
          let o k =
            List.nth [ element (); element (); element (); element () ] k
          in
          let a = into (o 0) Ops.O.(x * float 2.) in
          let b = into (o 1) Ops.O.(a + float 1.) in
          let c = into (o 2) Ops.O.(y * float 3.) in
          let d = into (o 3) Ops.O.(b * float 4.) in
          let calls = calls_of (Ops.sink [ a; b; c; d ]) in
          equal (list int) [ 2; 1; 1 ] (List.map members_of calls));
      test "a schedule of pairs of calls runs each pair as one" (fun () ->
          equal int 50 (List.length (calls_of (two_chains 100))));
      (* Work linear in the length is [a * n + b] with [b >= 0], so twice the
         length costs at most twice the work, and a tenth for tables that double
         their capacity at different lengths. *)
      test "the launches' work on a long schedule is linear in its length"
        (fun () ->
          let work n =
            let linear = chained_calls n in
            words (fun () -> Schedule.launches ~held:[] linear)
          in
          let short = work 1000 and long = work 2000 in
          equal int ~msg:"pairs" 500
            (List.length
               (Ops.src (Schedule.launches ~held:[] (chained_calls 1000))));
          less float_exact ~than:(2.2 *. short) long);
      (* Each output reads an earlier one, whose store reads the one before
         it: the whole schedule's work, not only its merge's. *)
      test "the work on a schedule of many outputs is linear in their number"
        (fun () ->
          let work n = words (fun () -> unplanned (two_chains n)) in
          let short = work 2000 and long = work 4000 in
          less float_exact ~than:(2.2 *. short) long);
      prop "outputs run as one write what their tensors compute, bit for bit"
        outputs_gen merges_keep_values;
      prop "calls of a loop run as one write what their tensors compute"
        loop_gen loop_merges_keep_values;
      prop
        "calls and stores of a loop that bind one name to several values \
         join, and no kernel binds a name to two"
        bound_gen joins_keep_bindings;
      test
        "a call of one element moves up only where no storage of many \
         elements starts living earlier" (fun () ->
          let buf n = Ops.new_buffer cpu n Float32 in
          let x = buf 1 and o = buf 1 and y = buf 1 in
          let calls t ~held =
            let linear =
              Ops.v Op.Linear
                ~src:[ add_one ~read:x ~write:o; add_one ~read:y ~write:t ]
            in
            List.length
              (Ops.src (Schedule.launches ~held:([ x; o; y ] @ held) linear))
          in
          let big = buf 4 in
          equal int ~msg:"storage of many elements" 2 (calls big ~held:[]);
          equal int ~msg:"given storage of many elements" 1
            (calls big ~held:[ big ]);
          equal int ~msg:"storage of one element" 1 (calls (buf 1) ~held:[]));
      test "calls that read one buffer read one parameter of their launch"
        (fun () ->
          let x = element () and o = element () and o' = element () in
          let big =
            Ops.sink
              [
                into o Ops.O.(Ops.exp2 x * float 2.);
                into o' Ops.O.(Ops.exp2 x + float 1.);
              ]
          in
          match calls_of big with
          | [ call ] ->
              equal int ~msg:"arguments" 3
                (List.length (Ops.src_without_body call))
          | calls -> failf "%d calls" (List.length calls));
    ]

(* Hazards

   Programs that read storage in one state and write it in another within one
   realize. Each writes what its tensors compute, read as the program states
   them (Tensors): a read of storage, through a join too, reads the state it
   names, whatever writes over that state later. Buffers hold four floats
   unless given another size. *)

let f32 ?(n = 4) slot = Ops.new_buffer ~slot cpu n Float32
let assign x v = Ops.after x [ Ops.store x v ]
let fl = Ops.float
let doubled u = Ops.O.(u * fl 2.)

(* The program [outputs] writes what its tensors compute. *)
let keeps_values outputs =
  let big = Ops.sink outputs in
  let buffers = filled big in
  let slots = List.map (fun (_, (p : Ops.param_arg)) -> p.slot) (given big) in
  let into_given = List.filter (fun (s, _, _) -> List.mem s slots) in
  let linear =
    with_settings ~debug:0 ~scache:0 (fun () -> unplanned big)
  in
  equal (list write)
    (Tensors.writes ~buffers big)
    (into_given (Kernel_graphs.linear_writes ~buffers linear));
  linear

(* [y] doubled, and [x] joined after it: [x] as it was, read once [y] is
   written. *)
let joined x y =
  let y1 = assign y Ops.O.(y * fl 2.) in
  (y1, Ops.after x [ y1 ])

(* A reader of a join waits for the writer of the state it holds: [x] is
   written by a kernel that waits for a chain of three, and [v] reads [x],
   joined after [y]'s write. *)
let join_waits_for_writer () =
  let x = f32 1 and y = f32 2 and t = f32 3 in
  let q =
    List.fold_left
      (fun q _ ->
        Ops.contiguous
          Ops.O.(Shape.expand (Shape.reshape (Shape.rop q Add [ 0 ]) [ Int 1 ]) [ Int 4 ] * fl 0.5))
      t [ 1; 2; 3 ]
  in
  let x1 = assign x Ops.O.(x + q) in
  let y1, xo = joined x1 y in
  ignore (keeps_values [ assign (f32 4) Ops.O.(xo * fl 2.); y1 ])

(* A reader of a join of [x] as it was waits for what the join waits for,
   whose writer reads [x] after an overwrite: it reads [x] as it was. *)
let join_over_overwritten () =
  let x = f32 1 and y = f32 2 in
  let x1 = assign x Ops.O.(x + fl 1.) in
  let y1 = assign y Ops.O.(y + x1) in
  let xo = Ops.after x [ y1 ] in
  ignore (keeps_values [ assign (f32 3) Ops.O.(xo * fl 2.); x1; y1 ])

(* A write over a join is a write over the state it holds: a kernel reading
   that state runs before it. *)
let write_over_join () =
  let x = f32 1 and y = f32 2 in
  let y1, xo = joined x y in
  let z = Ops.contiguous Ops.O.(y1 * fl 3.) in
  let r = assign (f32 3) Ops.O.(x + z) in
  ignore (keeps_values [ r; assign xo Ops.O.(xo + fl 5.) ])

(* Two joins over one state are no two writes over it. *)
let two_joins () =
  let x = f32 1 and y = f32 2 and z = f32 3 in
  let y1, x1 = joined x y in
  let z1, x2 = joined x z in
  ignore (keeps_values [ assign (f32 4) Ops.O.(x1 + x2); y1; z1 ])

(* A store into [x] of a value read from [x] joined after [y]'s write, at the
   element it writes, takes [x] once. *)
let join_one_parameter () =
  let x = f32 1 and y = f32 2 in
  let y1, xo = joined x y in
  let linear = keeps_values [ assign x Ops.O.((xo * fl 2.) + fl 1.); y1 ] in
  let into_x =
    List.filter
      (fun c -> List.exists (fun a -> Ops.buf_uop a == x) (Ops.src_without_body c))
      (Ops.src linear)
  in
  equal (list int) ~msg:"the kernel storing into x takes it once" [ 1 ]
    (List.map
       (fun c ->
         List.length
           (List.filter (fun a -> Ops.buf_uop a == x) (Ops.src_without_body c)))
       into_x)

(* A store through a view of a state, listed over a later state of that
   storage, is a write of the later state: it reads that state, older values
   beside it unchanged. *)
let store_names_older_state () =
  let x = f32 1 in
  let x1 = assign x Ops.O.(x * fl 2.) in
  let run a b = Shape.shrink x [ Some (Int a, Int b) ] in
  let run1 a b = Shape.shrink x1 [ Some (Int a, Int b) ] in
  let x2 = Ops.after x1 [ Ops.store (run 2 4) Ops.O.(run1 0 2 + fl 1.) ] in
  ignore (keeps_values [ x2 ])

(* A store through a run of a join is a store into the state the join holds,
   waiting for what the join waits for. *)
let store_through_join () =
  let x = f32 1 and y = f32 2 in
  let s1 = Ops.store y Ops.O.(y * fl 2.) in
  let xo = Ops.after x [ s1 ] in
  let run = Shape.shrink xo [ Some (Int 0, Int 2) ] in
  let x1 = Ops.after x [ Ops.store run (fl 3.) ] in
  ignore (keeps_values [ x1; Ops.after y [ s1 ] ])

(* An after listing only a store into other storage is the state it is over,
   read once the store has landed, one element or more. *)
let after_listing_foreign_store () =
  List.iter
    (fun n ->
      let x = f32 ~n 1 and y = f32 ~n 2 and z = f32 ~n 3 in
      let s1 = Ops.store y Ops.O.(y * fl 2.) in
      let xo = Ops.after x [ s1 ] in
      ignore (keeps_values [ assign z Ops.O.(xo + fl 1.); Ops.after y [ s1 ] ]))
    [ 1; 4 ]

(* Stores into one other storage listed in an after land in the order listed,
   as they do listed in that storage's own. *)
let after_listing_foreign_stores () =
  let x = f32 1 and y = f32 2 and z = f32 3 in
  let ss = [ Ops.store y Ops.O.(y * fl 2.); Ops.store y Ops.O.(y * fl 3.) ] in
  let xo = Ops.after x ss in
  ignore (keeps_values [ assign z Ops.O.(xo + fl 1.); Ops.after y ss ])

let joins =
  group "create_linear_with_vars › joins"
    [
      test "a reader of a join waits for the writer of the state it holds"
        join_waits_for_writer;
      test "a reader of a join of an overwritten state reads it as it was"
        join_over_overwritten;
      test "a write over a join runs after the readers of the state it holds"
        write_over_join;
      test "two joins over one state are no two writes" two_joins;
      test "a store reading its destination through a join takes it once"
        join_one_parameter;
      test "a store naming an older state, listed over a later one"
        store_names_older_state;
      test "a store through a run of a join" store_through_join;
      test "an after listing only a store into other storage is that state"
        after_listing_foreign_store;
      test "stores into one other storage land as listed"
        after_listing_foreign_stores;
    ]

(* [x]'s first two elements, staged, read after [x] is overwritten: they are
   [x]'s as it was. *)
let stage_of_written () =
  let x = f32 1 in
  let first_two u = Shape.shrink u [ Some (Int 0, Int 2) ] in
  let lo = Ops.contiguous (first_two x) in
  let x1 = assign x Ops.O.((x * fl 2.) + fl 7.) in
  let r =
    assign (f32 ~n:2 2) Ops.O.(lo + first_two (Ops.contiguous Ops.O.(x1 * fl 3.)))
  in
  ignore (keeps_values [ r; x1 ])

(* Halves of [x] swapped through stages of each. *)
let swap_halves () =
  let x = f32 1 in
  let half lo u = Shape.shrink u [ Some (Int lo, Int (lo + 2)) ] in
  let a = Ops.contiguous (half 0 x) and b = Ops.contiguous (half 2 x) in
  let x1 = Ops.after x [ Ops.store (half 0 x) b ] in
  let x2 = Ops.after x1 [ Ops.store (half 2 x1) a ] in
  ignore (keeps_values [ x2 ])

let views_of_written =
  group "create_linear_with_vars › views of written buffers"
    [
      test "a stage of a run of a buffer the call overwrites reads it as it was"
        stage_of_written;
      test "halves of a buffer swapped through stages swap" swap_halves;
    ]

(* Calls given one state twice

   A call that writes a state through one argument and reads it through
   another reads it as it was before the call, as [memmove] reads its
   source: a copy of it, made before the call, at the arguments it reads. *)

(* The copies of states the schedules made while [f] ran, each body
   scheduled anew. *)
let copies_while f =
  let before = Schedule.copies () in
  let x = with_settings ~debug:0 ~scache:0 f in
  (x, Schedule.copies () - before)

(* [keeps_values outputs], and the copies its schedule made. *)
let copied outputs = snd (copies_while (fun () -> ignore (keeps_values outputs)))
let p4 slot = Shape.param ~shape:[ Int 4 ] ~device:cpu slot Float32

(* A precompiled call storing [f] of its parameter 1 into its parameter 0,
   given [x] at both. *)
let into_self f x =
  Ops.after x [ Ops.call ~precompile:true (Ops.sink [ Ops.store (p4 0) (f (p4 1)) ]) [ x; x ] ]

let flipped u = Shape.flip u [ 0 ]

let shifted u =
  Shape.shrink (Shape.pad u [ Some (Int 1, Int 0) ]) [ Some (Int 0, Int 4) ]

let call_aliases (name, f) =
  test ("a call storing " ^ name ^ " of a state into it reads it as it was")
    (fun () -> equal int ~msg:"copies" 1 (copied [ into_self f (f32 1) ]))

(* Given two buffers, the call copies nothing. *)
let call_apart () =
  let x = f32 1 in
  let c =
    Ops.call ~precompile:true (Ops.sink [ Ops.store (p4 0) (flipped (p4 1)) ]) [ x; f32 2 ]
  in
  equal int ~msg:"copies" 0 (copied [ Ops.after x [ c ] ])

(* A kernel reversing its argument 1 into its argument 0, by element. *)
let reverse = function
  | [ o; i ] ->
      let r = Ops.range ~axis_type:Weak (Int 4) [ 0 ] in
      Ops.sink
        [ Ops.end_ (Ops.store (Ops.index o [ r ]) (Ops.index i [ Ops.O.(int 3 - r) ])) [ r ] ]
  | _ -> failwith "two arguments"

(* A custom kernel given [x] twice reverses [x]. Its body becomes the
   stores of a kernel, which reads [x] at other elements than it writes: it
   stores a stage of its value, and no copy is made. *)
let kernel_aliases () =
  let x = f32 1 in
  let big =
    Ops.sink [ List.hd (Shape.custom_kernel [ x; x ] reverse) ]
  in
  let buffers = filled big in
  let linear, made =
    copies_while (fun () -> unplanned big)
  in
  let given = Array.to_list (List.assoc 1 buffers) in
  equal (list write) ~msg:"x reversed"
    (List.mapi (fun i v -> (1, i, v)) (List.rev given))
    (List.filter
       (fun (s, _, _) -> s = 1)
       (Kernel_graphs.linear_writes ~buffers linear));
  equal int ~msg:"kernels" 2 (List.length (Ops.src linear));
  equal int ~msg:"copies" 0 made

(* A loop of three trips around a call storing [x] reversed, plus one, into
   [x]: each trip reads [x] as the trip before left it. *)
let loop_aliases () =
  let x = f32 1 in
  let r = Ops.range ~axis_type:Loop (Int 3) [ 100 ] in
  let body = Ops.sink [ Ops.store (p4 0) Ops.O.(flipped (p4 1) + fl 1.) ] in
  let e = Ops.end_ (Ops.call ~precompile:true body [ x; x ]) [ r ] in
  let linear, made =
    copies_while (fun () -> keeps_values [ Ops.after x [ e ] ])
  in
  equal int ~msg:"copies" 1 made;
  match Ops.src linear with
  | [ e ] ->
      equal (list string) ~msg:"a loop of a linear" [ "END"; "LINEAR" ]
        (ops_of [ e; Ops.nth e 0 ]);
      equal int ~msg:"of the copy and the call" 2
        (List.length (Ops.src (Ops.nth e 0)))
  | entries -> failf "%d entries" (List.length entries)

(* A call whose body does not show what it writes, given [x] twice and
   ordered as writing it, is refused. *)
let opaque_aliases () =
  let x = f32 1 in
  let c = Ops.call (Ops.custom_function "f" [ p4 0; p4 1 ]) [ x; x ] in
  raises_match (Exn.invalid_arg ~sub:"distinct storage") (fun () ->
      Schedule.create_schedule (Ops.sink [ Ops.after x [ c ] ]))

(* A precompiled call whose body calls a body storing the reverse of its
   parameter 1 into its parameter 0, passing its own parameters 0 and 1 and
   ordering its parameter 0 after it, given [x] at both: its body writes [x]
   at depth. *)
let nested_aliases () =
  let inner =
    Ops.sink [ Ops.store (p4 0) (flipped (p4 1)) ]
  in
  let outer =
    Ops.sink
      [ Ops.after (p4 0) [ Ops.call ~precompile:true inner [ p4 0; p4 1 ] ] ]
  in
  let x = f32 1 in
  equal (option (list int)) ~msg:"the outer body writes its parameter 0"
    (Some [ 0 ]) (Shape.written_params outer);
  equal int ~msg:"copies" 1
    (copied [ Ops.after x [ Ops.call ~precompile:true outer [ x; x ] ] ])

let aliased =
  group "create_linear_with_vars › calls given one state twice"
    [
      test "a call whose body writes through a call it makes reads a copy"
        nested_aliases;
      call_aliases ("the reverse", flipped);
      call_aliases ("a shift", shifted);
      call_aliases ("one plus each element", fun u -> Ops.O.(u + fl 1.));
      test "a call given two buffers copies nothing" call_apart;
      test "a custom kernel given a buffer twice reverses it" kernel_aliases;
      test "a loop of a call given a state twice reads it as each trip starts"
        loop_aliases;
      test "a call whose writes do not show, given a state twice, is refused"
        opaque_aliases;
    ]

(* Orders and copies

   Each program writes what its tensors compute, and its schedule copies a
   state only to break a cycle of its kernels: a kernel that must run both
   before and after another. A sum is a kernel of its own, which reads its
   source before any overwrite: no copy. *)

let three = fl 3.

(* [name], the program [outputs] and the copies its schedule makes. *)
let ordered_case (name, copies, outputs) =
  test (Printf.sprintf "%s: %d %s" name copies (if copies = 1 then "copy" else "copies"))
    (fun () -> equal int ~msg:"copies" copies (copied (outputs ())))

let first_half u = Shape.shrink u [ Some (Int 0, Int 2) ]
let second_half u = Shape.shrink u [ Some (Int 2, Int 4) ]
let summed u = Shape.expand (Shape.reshape (Shape.rop u Add [ 0 ]) [ Int 1 ]) [ Int 4 ]

let cases =
  [
    ( "a value of x read beside x's overwrite, waiting for it",
      1,
      fun () ->
        let x = f32 1 in
        let x1 = assign x Ops.O.(x + fl 1.) in
        [ assign (f32 2) Ops.O.(doubled x + Ops.contiguous Ops.O.(x1 * three)); x1 ] );
    ( "x flipped read beside x's overwrite, waiting for it",
      1,
      fun () ->
        let x = f32 1 in
        let x1 = assign x Ops.O.(x + fl 1.) in
        [ assign (f32 2) Ops.O.(Shape.flip x [ 0 ] + Ops.contiguous Ops.O.(x1 * three)); x1 ] );
    ( "x summed read beside x's overwrite, waiting for it",
      0,
      fun () ->
        let x = f32 1 in
        let x1 = assign x Ops.O.(x + fl 1.) in
        [ assign (f32 2) Ops.O.(summed x + Ops.contiguous Ops.O.(x1 * three)); x1 ] );
    ( "x and x overwritten in one kernel",
      1,
      fun () ->
        let x = f32 1 in
        let x1 = assign x Ops.O.(x + fl 1.) in
        [ assign (f32 2) Ops.O.(doubled x + x1); x1 ] );
    ( "x read, then overwritten",
      0,
      fun () ->
        let x = f32 1 in
        let b = Ops.contiguous (doubled x) in
        [ assign (f32 2) b; assign x Ops.O.(b + fl 1.) ] );
    ( "x and y swapped through values",
      1,
      fun () ->
        let x = f32 1 and y = f32 2 in
        [ assign x Ops.O.(y * fl 1.); assign y Ops.O.(x * fl 1.) ] );
    ( "x and y swapped through stages",
      0,
      fun () ->
        let x = f32 1 and y = f32 2 in
        let tx = Ops.contiguous Ops.O.(x * fl 1.)
        and ty = Ops.contiguous Ops.O.(y * fl 1.) in
        [ assign x ty; assign y tx ] );
    ( "x, y and z rotated through values",
      1,
      fun () ->
        let x = f32 1 and y = f32 2 and z = f32 3 in
        [
          assign x Ops.O.(y * fl 1.);
          assign y Ops.O.(z * fl 1.);
          assign z Ops.O.(x * fl 1.);
        ] );
    ( "x from y, y from z",
      0,
      fun () ->
        let x = f32 1 and y = f32 2 and z = f32 3 in
        [ assign x Ops.O.(y * fl 1.); assign y Ops.O.(z * fl 1.) ] );
    ( "x and y each reading both",
      1,
      fun () ->
        let x = f32 1 and y = f32 2 in
        [ assign x Ops.O.(y + x); assign y Ops.O.(x - y) ] );
    ( "x and y each reading the other's sum",
      0,
      fun () ->
        let x = f32 1 and y = f32 2 in
        [ assign x Ops.O.(x + summed y); assign y Ops.O.(y - summed x) ] );
    ( "x read through a join, y's write waiting for nothing of x",
      0,
      fun () ->
        let x = f32 1 and y = f32 2 in
        let y1 = assign y (doubled y) in
        [ assign (f32 3) Ops.O.(Ops.after x [ y1 ] + y1); y1 ] );
    ( "x's halves swapped through values",
      0,
      fun () ->
        let x = f32 1 in
        let lo = Ops.O.(first_half x * fl 1.) and hi = Ops.O.(second_half x * fl 1.) in
        let x1 = Ops.after x [ Ops.store (first_half x) hi ] in
        [ Ops.after x1 [ Ops.store (second_half x1) lo ] ] );
    ( "x's first half overwritten, its second read beside, waiting for it",
      1,
      fun () ->
        let x = f32 1 in
        let x1 = Ops.after x [ Ops.store (first_half x) Ops.O.(first_half x * fl 0. + fl 9.) ] in
        let late = Ops.contiguous Ops.O.(second_half x1 * three) in
        [ assign (f32 ~n:2 2) Ops.O.(second_half x + late); x1 ] );
    (* Two stores computing [a * 2] share their loops, b's first write and
       c's overwrite: the group runs after b's second write, which reads c as
       it was and overwrites what the group wrote, and before it. One copy, a
       baseline: the grouping makes the cycle, which the program does not
       have. *)
    ( "b from a, then from c, as c from a",
      1,
      fun () ->
        let a = f32 1 and b = f32 2 and c = f32 3 in
        let b1 = assign b (doubled a) in
        [ assign b1 (doubled c); assign c (doubled a) ] );
  ]

let relation =
  group "create_linear_with_vars › orders and copies" (List.map ordered_case cases)

(* Two writes over one state *)

let fork () =
  let x = f32 1 in
  raises_match (Exn.invalid_arg ~sub:"two writes over one state") (fun () ->
      ignore
        (keeps_values
           [ assign x (doubled x); assign x Ops.O.(x + fl 10.) ]))

(* A loop of a call reads x as it was given, and waits, through the
   storage it adds into, for x's overwrite: it reads a copy, made outside the
   loop, once. *)
let loop_reader () =
  let x = f32 1 and o = f32 2 in
  let x1 = assign x Ops.O.(x + fl 1.) in
  let o0 = assign o Ops.O.(x1 * fl 1.) in
  let r = Ops.range ~axis_type:Loop (Int 3) [ 100 ] in
  let body = Ops.sink [ Ops.store (p4 0) Ops.O.(p4 0 + p4 1) ] in
  let e = Ops.end_ (Ops.call ~precompile:true body [ o0; x ]) [ r ] in
  let linear, made =
    copies_while (fun () -> keeps_values [ Ops.after o0 [ e ]; x1 ])
  in
  equal int ~msg:"copies" 1 made;
  match List.filter (fun e -> Ops.op e = End) (Ops.src linear) with
  | [ e ] ->
      equal int ~msg:"the loop runs its call alone" 1
        (List.length (Ops.src (Ops.nth e 0)))
  | ends -> failf "%d loops" (List.length ends)

let kernels_apart () =
  let x = f32 1 and y = f32 2 and z = f32 3 in
  let outs k = Shape.custom_kernel k reverse in
  let a = outs [ x; z ] and b = outs [ y; z ] in
  equal (list uop) ~msg:"the argument read is itself" [ z; z ]
    [ List.nth a 1; List.nth b 1 ];
  let big = Ops.sink [ List.hd a; List.hd b ] in
  equal int ~msg:"copies" 0 (snd (copies_while (fun () -> unplanned big)))

(* Two calls, each writing [x] as it is given. *)
let fork_of_calls () =
  let x = f32 1 in
  let call v = Ops.call ~precompile:true (Ops.sink [ Ops.store (p4 0) v ]) [ x ] in
  raises_match (Exn.invalid_arg ~sub:"two writes over one state") (fun () ->
      ignore
        (keeps_values
           [ Ops.after x [ call (fl 1.) ]; Ops.after x [ call (fl 2.) ] ]))

(* A loop of three trips adding one into [x]'s storage, given [x] once
   assigned, its after naming the storage alone, as a loop's carry is built:
   it writes over the state it is given. A reader of that state beside runs
   before it. *)
let loop_after_storage () =
  let x = f32 1 in
  let x1 = assign x (doubled x) in
  let r = Ops.range ~axis_type:Loop (Int 3) [ 100 ] in
  let body = Ops.sink [ Ops.store (p4 0) Ops.O.(p4 0 + fl 1.) ] in
  let e = Ops.end_ (Ops.call ~precompile:true body [ x1 ]) [ r ] in
  let seen = assign (f32 2) Ops.O.(x1 * fl 3.) in
  equal int ~msg:"copies" 0 (copied [ Ops.after x [ e ]; seen ])

(* [x] stored [y] doubled twice: two states of one value. A loop of two trips
   adding its argument 1 into its argument 0, given the second state at 0 and
   the first at 1, reads the first as it was: each trip adds [2y]. *)
let loop_equal_states () =
  let x = f32 1 and y = f32 2 in
  let x1 = assign x (doubled y) in
  let x2 = assign x1 (doubled y) in
  let r = Ops.range ~axis_type:Loop (Int 2) [ 100 ] in
  let body = Ops.sink [ Ops.store (p4 0) Ops.O.(p4 0 + p4 1) ] in
  let e = Ops.end_ (Ops.call ~precompile:true body [ x2; x1 ]) [ r ] in
  ignore (keeps_values [ Ops.after x2 [ e ] ])

(* [x]'s two values [a] and [b] stored crossed into the columns of [p] and
   [q], 4 rows by 2, as two assemblies [[a; b]] and [[b; a]] along a new last
   axis are: the stores of one value, [a] into [p]'s first column and [q]'s
   second, are one kernel, which both [p]'s and [q]'s writes list, and so are
   [b]'s. The two kernels, of one write, run as one launch. *)
let crossed () =
  let x = f32 1 and p = f32 ~n:8 2 and q = f32 ~n:8 3 in
  let column b j =
    Shape.shrink
      (Shape.reshape b [ Int 4; Int 2 ])
      [ None; Some (Int j, Int (j + 1)) ]
  in
  let col v = Shape.reshape v [ Int 4; Int 1 ] in
  let a = col Ops.O.(x * fl 0.25) and b = col Ops.O.(x + fl 1.) in
  let assembled buf first second =
    Ops.after_disjoint buf
      [ Ops.store (column buf 0) first; Ops.store (column buf 1) second ]
  in
  let linear = keeps_values [ assembled p a b; assembled q b a ] in
  match Ops.src linear with
  | [ l ] when Shape.is_launch (Ops.nth l 0) ->
      equal int ~msg:"one kernel a column" 2
        (List.length (Shape.launch_members l))
  | es -> failf "%d entries, not one launch" (List.length es)

(* [a] stored into [p]'s first column and [q]'s second, one kernel (as
   [crossed] makes it), [c] into [q]'s first, and into [p]'s second [t]'s
   new value, which reads [q] once stored: the two stores into [p] are one
   write, and the second waits for the first through [t]'s kernel, so they
   cannot run as one launch. *)
let waits_through_another () =
  let x = f32 1 and p = f32 ~n:8 2 and q = f32 ~n:8 3 and t = f32 ~n:8 4 in
  let column b j =
    Shape.shrink
      (Shape.reshape b [ Int 4; Int 2 ])
      [ None; Some (Int j, Int (j + 1)) ]
  in
  let col v = Shape.reshape v [ Int 4; Int 1 ] in
  let a = col Ops.O.(x * fl 0.25) and c = col Ops.O.(x + fl 1.) in
  let q1 =
    Ops.after q [ Ops.store (column q 1) a; Ops.store (column q 0) c ]
  in
  let t1 = assign t Ops.O.(q1 + fl 1.) in
  let p1 =
    Ops.after p
      [
        Ops.store (column p 0) a;
        Ops.store (column p 1) (column Ops.O.(t1 * fl 2.) 0);
      ]
  in
  let linear = keeps_values [ p1; t1 ] in
  let writes_p m = List.memq p (Ops.src_without_body m) in
  List.iter
    (fun e ->
      if Ops.op e = Call && Shape.is_launch (Ops.nth e 0) then
        equal int ~msg:"members of a launch writing p" 1
          (List.length (List.filter writes_p (Shape.launch_members e))))
    (Ops.src linear)

let writes =
  group "create_linear_with_vars › writes"
    [
      test "a loop given two states of one value reads the older as it was"
        loop_equal_states;
      test "two calls writing one state are refused" fork_of_calls;
      test "a loop after its storage writes over the state it is given"
        loop_after_storage;
      test "two writes over one state are refused" fork;
      test "a loop reading a state a kernel it waits for overwrites reads a copy"
        loop_reader;
      test "custom kernels reading one buffer write nothing over it"
        kernels_apart;
      test "stores crossed into two buffers write both" crossed;
      test "stores of one write, one waiting for the other, run apart"
        waits_through_another;
    ]

(* Laws

   Programs drawn over three buffers of one size: each statement stores into a
   buffer, whole or a half of it, a constant plus a value read from the
   buffers' current states, their older states, joins of them, stages of
   them, flipped or summed, or repeats the statement before it. The schedule
   writes what the tensors compute. *)

type statement = {
  kind : [ `Store | `Call | `Loop ];
      (* A store of the value, or a precompiled call given the target's state
         and the states read, which adds them into it, once or on each of two
         trips of a loop. *)
  target : int;
  half : bool; (* Into the first half, of an even size, once written. *)
  older : bool;
      (* Over the state before the target's current one, beside the write
         that made the current one. *)
  constant : int;
  reads : (int * int * [ `Plain | `Flip | `Sum | `Stage | `Join of int ]) list;
      (* Each read: a buffer, how many states back, and how. *)
  bare : bool;
      (* A store no After lists, which the sink lists itself: it names no
         state, and the statements after it read the state it is over. *)
  pair : bool;
      (* A store listed with a second, of the value plus one, after it. *)
}

let gen_program ?(siblings = false) () =
  let open Gen in
  let gen_read =
    let* b = int_range 0 2 in
    let* back = int_range 0 2 in
    let+ how =
      one_of
        [
          constant `Plain; constant `Flip; constant `Sum; constant `Stage;
          map (fun j -> `Join j) (int_range 0 2);
        ]
    in
    (b, back, how)
  in
  let gen_statement =
    let some ?(always = false) k =
      if siblings || always then
        frequency [ (k, constant false); (1, constant true) ]
      else constant false
    in
    let* kind =
      frequency
        [ (3, constant `Store); (1, constant `Call); (1, constant `Loop) ]
    in
    let* target = int_range 0 2 in
    let* half = bool in
    let* older =
      if siblings then frequency [ (3, constant false); (1, constant true) ]
      else constant false
    in
    let* constant = int_range 1 2 in
    let* reads = list ~size:(int_range 1 3) gen_read in
    let* bare = some 4 in
    let+ pair = some ~always:true 3 in
    { kind; target; half; older; constant; reads; bare; pair }
  in
  let* n = int_range 1 5 in
  let* first = gen_statement in
  (* [None] repeats the statement before. *)
  let+ rest =
    list ~size:(int_range 0 4)
      (frequency [ (3, map Option.some gen_statement); (1, constant None) ])
  in
  let repeat (prev, acc) s =
    let s = Option.value s ~default:prev in
    (s, s :: acc)
  in
  (n, List.rev (snd (List.fold_left repeat (first, [ first ]) rest)))

let pp_program ppf (n, statements) =
  let how = function
    | `Plain -> "" | `Flip -> " flipped" | `Sum -> " summed"
    | `Stage -> " staged" | `Join j -> Printf.sprintf " joined after %d" j
  in
  Format.fprintf ppf "%d elements:" n;
  List.iter
    (fun s ->
      Format.fprintf ppf "@ %s%s%d%s%s := %d + %s;"
        (if s.bare then "bare " else if s.pair then "pair " else "")
        (match s.kind with `Store -> "" | `Call -> "call " | `Loop -> "loop ")
        s.target
        (if s.half then "[half]" else "")
        (if s.older then "@-1" else "")
        s.constant
        (String.concat " + "
           (List.map
              (fun (b, back, h) -> Printf.sprintf "%d@-%d%s" b back (how h))
              s.reads)))
    statements

(* The program's outputs, whether it reads an older state, through a join or
   not, whether a call or a loop is given its target's state twice, and whether
   a store stores the value the store before it stored into its storage. A
   write over an older state keeps the state it is beside among the
   outputs. *)
let program (n, statements) =
  let bufs = Array.init 3 (fun i -> f32 ~n (i + 1)) in
  (* Each buffer's states, newest first. *)
  let states = Array.map (fun b -> [ b ]) bufs in
  let old = ref false and joined = ref false and aliased = ref false in
  let again = ref false in
  let loops = ref 0 in
  let p i = Shape.param ~shape:[ Int n ] ~device:cpu i Float32 in
  (* A precompiled call adding its parameters past 0, halved, into its
     parameter 0. *)
  let adds cur args =
    let ps = List.mapi (fun i _ -> p (i + 1)) args in
    let v =
      List.fold_left (fun acc u -> Ops.O.(acc + (u * fl 0.5))) (p 0) ps
    in
    if List.memq cur args then aliased := true;
    Ops.call ~precompile:true (Ops.sink [ Ops.store (p 0) v ]) (cur :: args)
  in
  let state b back =
    let l = states.(b) in
    if back > 0 && List.length l > back then old := true;
    List.nth l (min back (List.length l - 1))
  in
  let read (b, back, how) =
    let s = state b back in
    match how with
    | `Plain -> s
    | `Flip -> Shape.flip s [ 0 ]
    | `Sum ->
        Shape.expand (Shape.reshape (Shape.rop s Add [ 0 ]) [ Int 1 ]) [ Int n ]
    | `Stage -> Ops.contiguous Ops.O.(s * fl 2.)
    | `Join j ->
        let dep = List.hd states.(j) in
        if Ops.op dep = After then begin
          joined := true;
          Ops.after s [ dep ]
        end
        else s
  in
  let beside = ref [] and bares = ref [] in
  List.iter
    (fun st ->
      let cur =
        match states.(st.target) with
        | latest :: prior :: _ when st.older ->
            beside := latest :: !beside;
            prior
        | latest :: _ -> latest
        | [] -> assert false
      in
      let value =
        List.fold_left
          (fun acc r -> Ops.O.(acc + read r))
          (fl (Float.of_int st.constant)) st.reads
      in
      let states_read =
        List.map (fun (b, back, _) -> state b back) st.reads
      in
      let next =
        (* Into a buffer as given, a half would be a view of it (D182), which
           Kernel_graphs does not run. *)
        if st.kind = `Call then Ops.after cur [ adds cur states_read ]
        else if st.kind = `Loop then begin
          (* Each loop its own range. *)
          incr loops;
          let r = Ops.range ~axis_type:Loop (Int 2) [ 900 + !loops ] in
          Ops.after cur [ Ops.end_ (adds cur states_read) [ r ] ]
        end
        else if st.half && n mod 2 = 0 && Ops.op cur = After then
          let h u = Shape.shrink u [ Some (Int 0, Int (n / 2)) ] in
          Ops.after cur [ Ops.store (h cur) (h value) ]
        else if st.bare then begin
          bares := Ops.store cur value :: !bares;
          cur
        end
        else if st.pair then
          Ops.after cur
            [ Ops.store cur value; Ops.store cur Ops.O.(value + fl 1.) ]
        else begin
          (match Ops.src cur with
          | [ _; s ] when Ops.op cur = After && Ops.op s = Store ->
              if Ops.nth s 1 == value then again := true
          | _ -> ());
          assign cur value
        end
      in
      if next != cur then states.(st.target) <- next :: states.(st.target))
    statements;
  let kinds = List.map (fun st -> st.kind) statements in
  ( Array.to_list (Array.map List.hd states)
    @ List.rev !beside @ List.rev !bares,
    !old,
    !joined,
    !aliased,
    !again,
    kinds )

let values_law =
  prop "a schedule writes what its tensors compute"
    (Gen.with_pp pp_program (gen_program ()))
    (fun p ->
      let outputs, old, joined, aliased, again, kinds = program p in
      let _, made =
        copies_while (fun () -> ignore (keeps_values outputs))
      in
      cover "an older state read" old;
      cover "a join read" joined;
      cover "a call" (List.mem `Call kinds);
      cover "a loop of a call" (List.mem `Loop kinds);
      cover "a call given its target's state twice" aliased;
      cover "two stores of one value into one storage" again;
      cover "a copy" (made > 0);
      cover "no copy" (made = 0))

(* The order of a sink's items orders nothing: a program whose outputs are
   listed the other way round, or with a group listing one of its writes first
   or last, writes what it writes, or all are refused as two writes over one
   state, which their order would otherwise decide. A write no After lists is
   over the state its destination names. *)

let outcome outputs =
  let big = Ops.sink outputs in
  let buffers = filled big in
  let slots = List.map (fun (_, (p : Ops.param_arg)) -> p.slot) (given big) in
  let into_given = List.filter (fun (s, _, _) -> List.mem s slots) in
  match with_settings ~debug:0 ~scache:0 (fun () -> unplanned big) with
  | linear -> Some (into_given (Kernel_graphs.linear_writes ~buffers linear))
  | exception Invalid_argument m
    when String.starts_with ~prefix:"two writes over one state" m ->
      None

let sink_order_law =
  prop "the order of a sink's items changes no result"
    (Gen.with_pp pp_program (gen_program ~siblings:true ()))
    (fun p ->
      let outputs, _, joined, _, _, kinds = program p in
      let got = outcome outputs in
      cover "two writes over one state" (got = None);
      cover "computed" (got <> None);
      cover "a write over an older state computed"
        (got <> None && List.exists (fun (st : statement) -> st.older) (snd p));
      cover "a join read" joined;
      cover "a call" (List.mem `Call kinds);
      let statements = snd p in
      cover "a write no After lists computed"
        (got <> None && List.exists (fun (st : statement) -> st.bare) statements);
      cover "a write listed second computed"
        (got <> None && List.exists (fun (st : statement) -> st.pair) statements);
      equal (option (list write)) got (outcome (List.rev outputs));
      (* A group listing a write orders nothing, wherever the sink has it: the
         first write an output lists, and the last one. *)
      match List.find_opt (fun o -> Ops.op o = After) outputs with
      | Some o ->
          List.iter
            (fun d ->
              let g = Ops.group [ d ] in
              equal (option (list write)) got (outcome (g :: outputs));
              equal (option (list write)) got (outcome (outputs @ [ g ])))
            (let first = Ops.nth o 1
             and last = List.nth (Ops.src o) (List.length (Ops.src o) - 1) in
             cover "a group of a write listed second" (first != last);
             if first == last then [ first ] else [ first; last ])
      | None -> ())

(* Kernel graphs drawn over four storages of four floats. Each write overwrites
   the current states of one storage or two with one kernel, which copies a
   state into each, or with two, which copy one into each half: the Afters of
   both storages then list both. A kernel copies from a state of any storage,
   possibly older or read through a join, and is passed one more such state
   beside. A copy is made exactly when the kernels, related as below, form a
   cycle: a kernel follows the writers of the states it is given and what the
   joins it reads through wait for, and precedes every other kernel of each
   write over a state it reads; a kernel writing a state that it reads
   through another argument precedes itself. *)

let gen_graph =
  let open Gen in
  let gen_ref =
    let* b = int_range 0 3 in
    let* back = int_range 0 2 in
    let+ join = option (int_range 0 3) in
    (b, back, join)
  in
  let gen_kernel =
    let* from = gen_ref in
    let+ beside = option gen_ref in
    (from, beside)
  in
  let gen_write =
    let* into = int_range 0 3 in
    let* also =
      frequency [ (3, constant None); (1, map Option.some (int_range 1 3)) ]
    in
    let+ kernels =
      list ~size:(frequency [ (3, constant 1); (1, constant 2) ]) gen_kernel
    in
    let targets =
      match also with
      | Some d -> [ into; (into + d) mod 4 ]
      | None -> [ into ]
    in
    (targets, kernels)
  in
  list ~size:(int_range 1 6) gen_write

let pp_graph ppf ws =
  let r (b, back, join) =
    Printf.sprintf "%d@-%d%s" b back
      (match join with Some j -> Printf.sprintf " after %d" j | None -> "")
  in
  let kernel (from, beside) =
    r from ^ match beside with Some b -> ", " ^ r b | None -> ""
  in
  List.iter
    (fun (targets, kernels) ->
      Format.fprintf ppf "@ %s := %s;"
        (String.concat "," (List.map string_of_int targets))
        (String.concat " | " (List.map kernel kernels)))
    ws

(* A parameter of the copy kernel's, of slot [slot]. *)
let kernel_param slot =
  let p =
    List.find
      (fun n ->
        match (Ops.op n, Ops.arg n) with
        | Param, Param p -> p.slot = 0
        | _ -> false)
      (Ops.toposort ~calls:Skip copy_kernel)
  in
  match Ops.arg p with
  | Param a -> Ops.replace ~arg:(Param { a with slot }) p
  | _ -> assert false

(* A kernel that copies its argument [n] into its arguments [0] to [n - 1], at
   the elements [lo] to [lo + count - 1]. *)
let copies_into n ~lo ~count =
  let r = Ops.range ~axis_type:Weak (Int count) [ 0 ] in
  let i = Ops.O.(r + int lo) in
  let from = Ops.index (kernel_param n) [ i ] in
  Ops.replace
    ~src:
      (List.init n (fun t ->
           Ops.end_ (Ops.store (Ops.index (kernel_param t) [ i ]) from) [ r ]))
    copy_kernel

(* The graph, and whether its kernels form a cycle. *)
let graph ws =
  let storage = [| x; y; z; w |] in
  let size = Shape.max_numel x in
  let states = Array.map (fun s -> [ s ]) storage in
  let state (b, back, join) =
    let l = states.(b) in
    let s = List.nth l (min back (List.length l - 1)) in
    match join with
    | Some j when Ops.op (List.hd states.(j)) = After ->
        Ops.after s [ List.hd states.(j) ]
    | _ -> s
  in
  (* Each kernel, the states it overwrites, and the states it reads with the
     kernels it waits for through joins; each write, its kernels and the
     states they overwrite. *)
  let kernels = ref [] and writes = ref [] in
  List.iter
    (fun (targets, ks) ->
      let curs = List.map (fun b -> List.hd states.(b)) targets in
      let n = List.length targets and count = size / List.length ks in
      let made =
        List.mapi
          (fun j (from, beside) ->
            let read = List.map state (from :: Option.to_list beside) in
            let body = copies_into n ~lo:(j * count) ~count in
            let k =
              Ops.replace ~src:((body :: curs) @ read) copy_call
            in
            kernels := (k, curs, read) :: !kernels;
            k)
          ks
      in
      writes := (made, curs) :: !writes;
      List.iter
        (fun b -> states.(b) <- Ops.after (List.hd states.(b)) made :: states.(b))
        targets)
    ws;
  let kernels = List.rev !kernels in
  let rec value s = if Ops.is_join s then value (Ops.nth s 0) else s in
  let rec waits s =
    (if Ops.op s = After && not (Ops.is_join s) then List.tl (Ops.src s)
     else [])
    @ if Ops.is_join s then List.concat_map waits (Ops.src s) else []
  in
  (* Edges from a kernel to those that must run after it. *)
  let succ (k, curs, read) =
    let war =
      List.concat_map
        (fun s ->
          List.concat_map
            (fun (made, curs') ->
              if List.memq (value s) curs' then
                List.filter (fun t -> t != k) made
              else [])
            !writes)
        read
    in
    let own = List.exists (fun s -> List.memq (value s) curs) read in
    war
    @ (if own then [ k ] else [])
    @ List.filter_map
        (fun (t, curs', read') ->
          if List.exists (fun s -> List.memq k (waits s)) (curs' @ read') then
            Some t
          else None)
        kernels
  in
  let succs = List.map (fun ((k, _, _) as e) -> (k, succ e)) kernels in
  (* A cycle: a kernel reached again from itself. *)
  let rec reaches seen k t =
    List.exists
      (fun n -> n == t || ((not (List.memq n seen)) && reaches (n :: seen) n t))
      (List.assq k succs)
  in
  let cycle = List.exists (fun (k, _, _) -> reaches [] k k) kernels in
  (Ops.sink (Array.to_list (Array.map List.hd states)), cycle)

let copy_iff_cycle =
  prop "a schedule copies a state iff its kernels form a cycle"
    (Gen.with_pp pp_graph gen_graph)
    (fun ws ->
      let kg, cycle = graph ws in
      let allocs =
        List.filter (fun n -> Ops.op n = Alloc)
          (Ops.toposort ~calls:Skip (Schedule.create_schedule kg))
      in
      cover "a cycle" cycle;
      cover "no cycle" (not cycle);
      cover "a kernel that two Afters list beside another kernel"
        (List.exists
           (fun (targets, ks) -> List.length targets = 2 && List.length ks = 2)
           ws);
      equal bool ~msg:"a copy" cycle (allocs <> []);
      keeps kg)

(* Repros

   The programs that decided the relation, each through its tensors: its
   values are its tensors', and its schedule makes the copies given, or the
   program raises. A count other than the relation's own reading of a
   program is the kernel split's: a sum or a stage is a kernel of its own,
   which reads its source before any overwrite, and a custom kernel's body is
   rangeified, where a stage keeps it right. *)

module O = Ops.O

let slots = Atomic.make 100
let fresh_slot () = Atomic.fetch_and_add slots 1
let pf ?(n = 4) slot = Shape.param ~shape:[ Int n ] ~device:cpu slot Float32
let numel u = Shape.max_numel u
let flat u = Shape.reshape u [ Int (numel u) ]

(* [v] stored into a buffer of its own. *)
let out v =
  let b = Ops.new_buffer ~slot:(fresh_slot ()) cpu (numel v) (Ops.dtype v) in
  Ops.after b [ Ops.store (Shape.reshape b (Shape.shape v)) v ]

(* [v] stored into a run of [x]'s state. *)
let into x lo hi v =
  Ops.after x [ Ops.store (Shape.shrink x [ Some (Int lo, Int hi) ]) v ]

let span lo hi u = Shape.shrink u [ Some (Int lo, Int hi) ]
let flip u = Shape.flip u [ 0 ]
let summed u = Shape.expand (Shape.reshape (Shape.rop (flat u) Add [ 0 ]) [ Int 1 ]) [ Int (numel u) ]
let padded u = span 0 (numel u) (Shape.pad u [ Some (Int 1, Int 0) ])
let rows u = Shape.reshape u [ Int 2; Int 2 ]
let k c = fl c

(* [a] at the elements [idx] holds, clamped into [a]: a gather. *)
let gather a idx =
  let n = numel a in
  let i = Ops.cast idx Int32 in
  let i =
    Ops.maximum (Ops.minimum i (Ops.int ~dtype:Int32 (n - 1))) (Ops.int ~dtype:Int32 0)
  in
  Ops.index a [ Ops.cast i Weak_int ]

(* A precompiled call storing [f] of its parameters past 0 into its parameter
   0, given [dst] there and [args] after. *)
let calls ?(n = 4) dst args f =
  let ps = List.mapi (fun i _ -> pf ~n (i + 1)) args in
  Ops.after dst
    [ Ops.call ~precompile:true (Ops.sink [ Ops.store (pf ~n 0) (f ps) ]) (dst :: args) ]

let two () = (f32 1, f32 2)
let three_bufs () = (f32 1, f32 2, f32 3)

type expect = Copies of int | Raises of string

let c_ n name f = (name, Copies n, f)

let repros =
  let c n name f = (name, Copies n, f) in
  [
    (* A stale read fused beside its overwrite, waiting for it. *)
    c 1 "t1 repro" (fun () ->
        let a = f32 1 in let b = doubled a in let a1 = assign a O.(a + k 1.) in
        [ out O.(b + Ops.contiguous O.(a1 * three)); a1 ]);
    c 1 "t1 flip" (fun () ->
        let a = f32 1 in let b = doubled (flip a) in let a1 = assign a O.(a + k 1.) in
        [ out O.(b + Ops.contiguous O.(a1 * three)); a1 ]);
    c 1 "t1 pad" (fun () ->
        let a = f32 1 in let b = doubled (padded a) in let a1 = assign a O.(a + k 1.) in
        [ out O.(b + Ops.contiguous O.(a1 * three)); a1 ]);
    c 1 "t1 gather" (fun () ->
        let a = f32 1 and i = f32 2 in let b = doubled (gather a i) in
        let a1 = assign a O.(a + k 1.) in
        [ out O.(b + Ops.contiguous O.(a1 * three)); a1 ]);
    c 1 "t1 chain" (fun () ->
        let a = f32 1 in let b = doubled a in let a1 = assign a O.(a + k 1.) in
        let a2 = assign a1 O.(a1 * k 10.) in
        [ out O.(b + Ops.contiguous O.(a2 * three)); a2 ]);
    c 0 "t1 reduce" (fun () ->
        let a = f32 1 in let b = summed a in let a1 = assign a O.(a + k 1.) in
        [ out O.(b + Ops.contiguous O.(a1 * three)); a1 ]);
    c 1 "t1 beside" (fun () ->
        let a = f32 1 in let a1 = assign a O.(a + k 1.) in [ out O.(doubled a + a1); a1 ]);
    c 2 "t1 twobufs" (fun () ->
        let a, cc = two () in let b = O.(doubled a + cc) in
        let a1 = assign a O.(a + k 1.) in let c1 = assign cc O.(cc + a1) in
        [ out O.(b + Ops.contiguous O.(c1 * three)); a1; c1 ]);
    c 1 "t2 view_read" (fun () ->
        let a = f32 1 in let b = doubled (Shape.permute (rows a) [ 1; 0 ]) in
        let a1 = assign a O.(a + k 1.) in
        [ out O.(b + Ops.contiguous O.(rows a1 * three)); a1 ]);
    (* Into a written state, which a store indexes whole (D182). *)
    c 1 "t2 view_assign" (fun () ->
        let a = f32 1 and v = f32 ~n:2 2 in let a0 = assign a O.(a + k 1.) in
        let b = doubled a0 in let a1 = into a0 0 2 v in
        [ out O.(b + Ops.contiguous O.(a1 * three)); a1 ]);
    c 1 "t2 gated" (fun () ->
        let a = f32 1 in let a1 = assign a O.(a * k 10.) in
        let m = Ops.contiguous O.(a1 > k 15.) in
        [ out O.(Ops.where m a (k 0.) + a1); a1 ]);
    c 1 "t2 gather_index_waits" (fun () ->
        let a = f32 1 in let a1 = assign a (flip a) in
        let idx = Ops.contiguous O.(a1 - k 1.) in
        [ out O.(gather a idx + a1); a1 ]);
    c 1 "t2 gather_index_waits_fused" (fun () ->
        let a = f32 1 in let a1 = assign a (flip a) in
        [ out (gather a O.(a1 - k 1.)); a1 ]);
    c 1 "t11 bitcast_old" (fun () ->
        let x = Ops.new_buffer ~slot:1 cpu 4 Int32 in
        let old = Ops.cast (Ops.bitcast x Uint32) Int32 in
        let x1 = assign x O.(x + Ops.int ~dtype:Int32 1) in
        [ out O.(old + Ops.contiguous O.(x1 * Ops.int ~dtype:Int32 3)); x1 ]);
    c 1 "t11 reduce_view_old" (fun () ->
        let x = f32 1 in let x1 = assign x (doubled x) in
        [ out O.(Shape.rop (rows x) Add [ 1 ] + Ops.contiguous (span 0 2 x1)); x1 ]);
    (* Calls and precompiled readers. *)
    c 1 "t2 function_reader" (fun () ->
        let a = f32 1 in let a1 = assign a O.(a + k 1.) in
        let na = Ops.contiguous O.(a1 * k 1.) in
        [ calls (f32 9) [ a; na ] (function [ x; y ] -> O.(x + (y * three)) | _ -> assert false); a1 ]);
    c 0 "t12 precompiled_then_new" (fun () ->
        let a = f32 1 in let o = calls (f32 9) [ a ] (function [ x ] -> doubled x | _ -> assert false) in
        let a1 = assign a O.(a + k 1.) in
        [ out O.(o + Ops.contiguous O.(a1 * three)); a1 ]);
    c 1 "B6, B8: a call reading x as it was beside x assigned" (fun () ->
        let a = f32 1 in let a1 = assign a O.(a + k 1.) in
        [ calls (f32 9) [ a; a1 ] (function [ x; y ] -> O.(x + (y * three)) | _ -> assert false); a1 ]);
    (* A destination waiting for the write. *)
    c 1 "t15 dst_state_waits" (fun () ->
        let a, cc = two () in let stale = O.(a + k 10.) in
        let a1 = assign a (doubled a) in let c1 = assign cc O.(a1 + k 0.) in
        [ a1; assign c1 stale ]);
    c 1 "t15 dst_state_waits_partial" (fun () ->
        let a, cc = two () in let stale = O.(span 2 4 a + k 10.) in
        let a1 = assign a (doubled a) in let c1 = assign cc O.(a1 + k 0.) in
        [ a1; into c1 0 2 stale ]);
    (* One kernel reading two or three states. *)
    c 2 "B2 states 0, 1 and 2" (fun () ->
        let a = f32 1 in let s0 = O.(a * k 1.) in let a1 = assign a O.(a + k 1.) in
        let s1 = O.(a1 * k 1.) in let a2 = assign a1 O.(a1 * k 10.) in
        [ out O.(s0 + s1 + a2); a2 ]);
    c 0 "B3 a sum beside" (fun () ->
        let a = f32 1 in let a1 = assign a O.(a + k 1.) in [ out O.(summed a + a1); a1 ]);
    c 1 "B4 a product summed" (fun () ->
        let a = f32 1 in let a1 = assign a O.(a + k 1.) in
        [ out (summed O.(doubled a * a1)); a1 ]);
    c 1 "B5 a flip beside" (fun () ->
        let a = f32 1 in let a1 = assign a O.(a * k 10.) in [ out O.(flip a + a1); a1 ]);
    c 1 "t3 pad_old" (fun () ->
        let a = f32 1 in let a1 = assign a O.(a + k 1.) in
        [ out O.(padded a + Ops.contiguous O.(a1 * three)); a1 ]);
    c 0 "t3 sum_old" (fun () ->
        let a = f32 1 in let a1 = assign a O.(a + k 1.) in
        [ out O.(summed a + Ops.contiguous O.(a1 * three)); a1 ]);
    c 1 "t3 old_beside_new_sum" (fun () ->
        let a = f32 1 in let a1 = assign a O.(a + k 1.) in [ out (summed O.(a * a1)); a1 ]);
    c 1 "t3 matmul_old_new" (fun () ->
        let a = f32 1 in let a1 = assign a O.(a + k 1.) in
        let x = Shape.expand (Shape.reshape (rows a) [ Int 2; Int 2; Int 1 ]) [ Int 2; Int 2; Int 2 ]
        and y = Shape.expand (Shape.reshape (rows a1) [ Int 1; Int 2; Int 2 ]) [ Int 2; Int 2; Int 2 ] in
        [ out (Shape.rop O.(x * y) Add [ 1 ]); a1 ]);
    c 1 "t3 assign_from_old_elsewhere" (fun () ->
        let a = f32 1 in let a1 = assign a (doubled a) in [ assign a1 O.(flip a + a1) ]);
    c 1 "t5 g2d_contig" (fun () ->
        let a = f32 1 in let a1 = assign a (flip a) in
        let idx = Ops.contiguous O.(a1 - k 1.) in
        [ out O.(gather a idx + a1); a1 ]);
    c 1 "t6 a transpose beside" (fun () ->
        let a = f32 1 in let a1 = assign a O.(flip a + k 1.) in
        [ out O.(Shape.permute (rows a) [ 1; 0 ] + rows a1); a1 ]);
    (* Swaps, rotations, crossed reads. *)
    c 1 "S1 swap" (fun () ->
        let a, b = two () in [ assign a O.(b * k 1.); assign b O.(a * k 1.) ]);
    c 0 "S2 swap through stages" (fun () ->
        let a, b = two () in let na = Ops.contiguous O.(b * k 1.) and nb = Ops.contiguous O.(a * k 1.) in
        [ assign a na; assign b nb ]);
    c 1 "S3 swap through a view" (fun () ->
        let a, b = two () in [ assign a b; assign b (flat (rows a)) ]);
    c 0 "S4 a = b, then b = a" (fun () ->
        let a, b = two () in let a1 = assign a b in [ a1; assign b a1 ]);
    c 1 "S5 swap through a value" (fun () ->
        let a, b = two () in let t = O.(a * k 1.) in [ assign a b; assign b t ]);
    c 0 "S6 swap through a stage" (fun () ->
        let a, b = two () in let t = Ops.contiguous O.(a * k 1.) in [ assign a b; assign b t ]);
    c 1 "S7 crossed, each reading both" (fun () ->
        let a, b = two () in [ assign a O.(b + a); assign b O.(a - b) ]);
    c 0 "S8 halves swapped through values" (fun () ->
        let x = f32 1 in let lo = O.(span 0 2 x * k 1.) and hi = O.(span 2 4 x * k 1.) in
        let x1 = into x 0 2 hi in [ into x1 2 4 lo ]);
    c 0 "S10 a flip in place" (fun () -> let x = f32 1 in [ assign x (flip x) ]);
    c 1 "R1 rotation" (fun () ->
        let a, b, cc = three_bufs () in
        [ assign a O.(b * k 1.); assign b O.(cc * k 1.); assign cc O.(a * k 1.) ]);
    c 0 "R2 rotation through stages" (fun () ->
        let a, b, cc = three_bufs () in
        let na = Ops.contiguous O.(b * k 1.) and nb = Ops.contiguous O.(cc * k 1.)
        and nc = Ops.contiguous O.(a * k 1.) in
        [ assign a na; assign b nb; assign cc nc ]);
    c 1 "R3 rotation through one value" (fun () ->
        let a, b, cc = three_bufs () in let t = O.(a * k 1.) in
        [ assign a b; assign b cc; assign cc t ]);
    c 0 "R4 a chain" (fun () ->
        let a, b, cc = three_bufs () in [ assign a O.(b * k 1.); assign b O.(cc * k 1.) ]);
    c 1 "C1 t13 crossed" (fun () ->
        let a, b, cc = three_bufs () in let sa = O.(a + k 0.) and sb = O.(b + k 0.) in
        let a1 = assign a (doubled a) in let c1 = assign cc O.(a1 + sb) in
        let b1 = assign b O.(b * three) in
        [ a1; b1; c1; out O.(sa + b1) ]);
    c 0 "C2 t13 crossed through stages" (fun () ->
        let a, b, cc = three_bufs () in
        let sa = Ops.contiguous O.(a + k 0.) and sb = Ops.contiguous O.(b + k 0.) in
        let a1 = assign a (doubled a) in let c1 = assign cc O.(a1 + sb) in
        let b1 = assign b O.(b * three) in
        [ a1; b1; c1; out O.(sa + b1) ]);
    (* Joins. *)
    c 0 "t7 a join beside the state it joins" (fun () ->
        let x, y = two () in let y1 = assign y (doubled y) in
        [ out (Ops.after x [ y1 ]); out O.(x + y1); y1 ]);
    c 0 "t9 a join read" (fun () ->
        let x, y = two () in let y1 = assign y (doubled y) in
        [ out O.(Ops.after x [ y1 ] + k 1.); out O.(x + y1) ]);
    c 0 "J2 a join over an assigned state" (fun () ->
        let x, y = two () in let x1 = assign x O.(x + k 1.) in let y1 = assign y (doubled y) in
        [ out O.(Ops.after x1 [ y1 ] + k 0.); y1 ]);
    c 0 "J3 a join, its dependency's writer reading x" (fun () ->
        let x, y = two () in let y1 = assign y O.(doubled y + x) in
        [ out (Ops.after x [ y1 ]); out O.(x + y1) ]);
    c 0 "joinflip c2: a store over x of x joined, at its element" (fun () ->
        let x, y = two () in let y1 = assign y (doubled y) in
        [ assign x O.((Ops.after x [ y1 ] * k 2.) + k 1.); y1 ]);
    c 0 "joinflip c1: a store over x of x joined, flipped" (fun () ->
        let x, y = two () in let y1 = assign y (doubled y) in
        [ assign x (flip (Ops.after x [ y1 ])); y1 ]);
    c 0 "joinflip c3: x joined and x flipped" (fun () ->
        let x, y = two () in let y1 = assign y (doubled y) in
        [ assign x O.(Ops.after x [ y1 ] + flip x); y1 ]);
    c 0 "s6 a join after a state on another device" (fun () ->
        let x = f32 1 and y = Ops.new_buffer ~slot:2 "CPU:1" 4 Float32 in
        let y1 = assign y (doubled y) and x1 = assign x O.(x + k 1.) in
        [ out O.(Ops.after x1 [ y1 ] + k 0.); y1 ]);
    c 0 "s6 nested joins over an old state" (fun () ->
        let x, y, z = three_bufs () in let x1 = assign x O.(x + k 1.) in
        let y1 = assign y O.(y + x1) in let z1 = assign z O.(z + y1) in
        [ out O.(Ops.after (Ops.after x [ y1 ]) [ z1 ] * k 1.); x1; y1; z1 ]);
    c 0 "s6 a join read, then its state overwritten" (fun () ->
        let x, y = two () in let y1 = assign y (doubled y) in
        let x1 = assign x O.(x + k 1.) in let w = O.(Ops.after x1 [ y1 ] * three) in
        [ out w; assign x1 O.((x1 * k 0.) + k 9.); y1 ]);
    c 0 "s2 a join staged" (fun () ->
        let x, y = two () in let y1 = assign y (doubled y) and x1 = assign x O.(x + k 1.) in
        [ out O.(Ops.contiguous (Ops.after x1 [ y1 ]) + k 0.); y1 ]);
    c 0 "s2 a join over an old state, staged" (fun () ->
        let x, y = two () in let x1 = assign x O.(x + k 1.) in let y1 = assign y O.(y + x1) in
        [ out O.(Ops.contiguous (Ops.after x [ y1 ]) * k 1.); x1; y1 ]);
    (* Disjoint partial overwrites. *)
    c 1 "t14 todo" (fun () ->
        let x = f32 ~n:1 1 and y = f32 ~n:1 2 in let xe = O.(x + k 10.) in
        let x1 = assign x (doubled x) in let y1 = assign y O.(y + x1) in
        [ x1; y1; out O.(y1 + xe) ]);
    c 1 "t14 disjoint" (fun () ->
        let x = f32 1 and y = f32 ~n:2 2 in let x0 = assign x O.(x + k 1.) in
        let stale = O.(span 2 4 x0 + k 10.) in
        let x1 = into x0 0 2 (f32 ~n:2 3) in let y1 = assign y O.(y + span 0 2 x1) in
        [ x1; y1; out O.(y1 + stale) ]);
    (* Views of written buffers. *)
    c 0 "V5 a stage of a run read after its buffer's overwrite" (fun () ->
        let x = f32 1 in let lo = Ops.contiguous (span 0 2 x) in
        let x1 = assign x O.((x * k 0.) + k 7.) in
        [ out O.(lo + span 0 2 (Ops.contiguous O.(x1 * k 1.))); x1 ]);
    c 0 "S9 halves swapped through stages" (fun () ->
        let x = f32 1 in let lo = Ops.contiguous (span 0 2 x) and hi = Ops.contiguous (span 2 4 x) in
        let x1 = into x 0 2 hi in [ into x1 2 4 lo ]);
    (* In-kernel: at the element written, no stage; elsewhere, a stage. *)
    c 0 "a := a * 2 + 1" (fun () -> let a = f32 1 in [ assign a O.((a * k 2.) + k 1.) ]);
    c 0 "an SGD step" (fun () ->
        let w, g = two () in [ assign w O.(w - (g * k 0.1)) ]);
    c 0 "an Adam step" (fun () ->
        let w, m, v = three_bufs () and g = f32 4 in
        let m1 = assign m O.((m * k 0.9) + (g * k 0.1)) in
        let v1 = assign v O.((v * k 0.999) + (g * g * k 0.001)) in
        [ m1; v1; assign w O.(w - (m1 * k 0.001 / (Ops.sqrt v1 + k 1e-8))) ]);
    c 0 "a[i] := sum over j of a[i] * w[j]" (fun () ->
        let a, w = two () in
        let prod = O.(Shape.expand (Shape.reshape a [ Int 4; Int 1 ]) [ Int 4; Int 4 ]
                      * Shape.expand (Shape.reshape w [ Int 1; Int 4 ]) [ Int 4; Int 4 ]) in
        [ assign a (Shape.rop prod Add [ 1 ]) ]);
    c 0 "elsewhere: a flip" (fun () -> let a = f32 1 in [ assign a O.(flip a + k 1.) ]);
    c 0 "elsewhere: a roll" (fun () ->
        let a = f32 1 in [ assign a (flat (Shape.permute (rows a) [ 1; 0 ])) ]);
    c 0 "elsewhere: a + a's sum" (fun () -> let a = f32 1 in [ assign a O.(a + summed a) ]);
    c 0 "elsewhere: a gather of a" (fun () ->
        let a, i = two () in [ assign a (gather a i) ]);
    (* Grouping. *)
    c 1 "group3" (fun () ->
        let a, b, cc = three_bufs () in let b1 = assign b (doubled a) in
        [ assign b1 (doubled cc); assign cc (doubled a) ]);
    (* Two writes over one state. *)
    ("F1 two assigns over one state", Raises "two writes over one state",
     fun () -> let a = f32 1 in [ assign a (doubled a); assign a O.(a + k 10.) ]);
    ("F3 a kernel reading two writes over one state", Raises "two writes over one state",
     fun () -> let a = f32 1 in [ out O.(assign a (doubled a) + assign a O.(a + k 10.)) ]);
    ("F4 an assign over the old state's view", Raises "two writes over one state",
     fun () -> let a = f32 1 in
       [ assign a (doubled a); Ops.after a [ Ops.store (rows a) (Shape.reshape (f32 2) [ Int 2; Int 2 ]) ] ]);
  ]

(* s7: an after of [a] listing a call that writes it and [y] once doubled:
   a write that also waits. *)
let mixed_after read_old =
  let a, b = two () and y = f32 3 in
  let c = calls a [ b ] (function [ x ] -> O.(x + k 1.) | _ -> assert false) in
  let c = Ops.nth c 1 in
  let y1 = assign y (doubled y) in
  let am = Ops.after a [ c; y1 ] in
  [ out (if read_old then O.(am + y) else O.(am + k 0.)); y1 ]

let mixed =
  [
    c_ 0 "s7 an after of a call and a join, read" (fun () -> mixed_after false);
    c_ 0 "s7 an after of a call and a join, read beside y as it was" (fun () ->
        mixed_after true);
  ]

(* s2: a custom kernel given views of one buffer of eight floats,
   [o.(r) := i.(at r)] for its [n] elements, writes what reading [i] before any
   write gives, as [expected] gives it of the buffer's values. Views start on
   a 16-byte boundary, as a kernel's parameters lie. *)
let kernel_views ~n ~at ~args expected () =
  let kernel = function
    | [ o; i ] ->
        let r = Ops.range ~axis_type:Weak (Int n) [ 0 ] in
        Ops.sink
          [ Ops.end_ (Ops.store (Ops.index o [ r ]) (Ops.index (flat i) [ at r ])) [ r ] ]
    | _ -> assert false
  in
  let a = f32 ~n:8 1 in
  let o = List.hd (Shape.custom_kernel (args a) kernel) in
  let big = Ops.sink [ Ops.after a [ Ops.nth o 1 ] ] in
  let buffers = filled big in
  let x = List.assoc 1 buffers in
  let linear, _ = copies_while (fun () -> unplanned big) in
  let want = expected x in
  equal (list write)
    (List.filter
       (fun (_, i, v) -> v <> x.(i))
       (List.mapi (fun i v -> (1, i, v)) (Array.to_list want)))
    (List.filter
       (fun (s, i, v) -> s = 1 && v <> x.(i))
       (Kernel_graphs.linear_writes ~buffers linear))

(* A back edge around a call copying [x] into itself, element by element, given
   [x] at both arguments: each trip reads [x] as the trip starts, through a copy
   made in the loop. *)
let backedge_aliases () =
  let r = Ops.range ~axis_type:Loop (Int 4) [ 7 ] in
  let flag = Ops.new_buffer ~slot:9 cpu 1 Bool in
  let e = Ops.backedge (copies ~into:x ~from:x) ~loop:r ~cond:flag in
  let made = Schedule.copies () in
  let linear = Schedule.create_schedule (Ops.sink [ Ops.after x [ e ] ]) in
  equal int ~msg:"copies" 1 (Schedule.copies () - made);
  match Ops.src linear with
  | [ b ] ->
      equal (list string) ~msg:"a back edge of a linear" [ "BACKEDGE"; "LINEAR" ]
        (ops_of [ b; Ops.nth b 0 ]);
      equal int ~msg:"of the copy and the call" 2
        (List.length (Ops.src (Ops.nth b 0)))
  | entries -> failf "%d entries" (List.length entries)

let view_rows =
  [
    test "s2: a custom kernel writes a[0:4] := flip a[0:4], a view written"
      (kernel_views ~n:4 ~at:(fun r -> Ops.O.(int 3 - r))
         ~args:(fun a -> [ span 0 4 a; a ])
         (fun x -> Array.init 8 (fun i -> if i < 4 then x.(3 - i) else x.(i))));
    test "s2: a custom kernel writes a := a[4:8] tiled and flipped, a view read"
      (kernel_views ~n:8 ~at:(fun r -> Ops.O.(int 3 - (r % int 4)))
         ~args:(fun a -> [ a; span 4 8 a ])
         (fun x -> Array.init 8 (fun i -> x.(4 + 3 - (i mod 4)))));
    test "s2: a custom kernel writes a[0:4] := a[4:8], disjoint views"
      (kernel_views ~n:4 ~at:(fun r -> r)
         ~args:(fun a -> [ span 0 4 a; span 4 8 a ])
         (fun x -> Array.init 8 (fun i -> if i < 4 then x.(4 + i) else x.(i))));
  ]

let repro (name, expect, outputs) =
  test name (fun () ->
      match expect with
      | Copies n -> equal int ~msg:"copies" n (copied (outputs ()))
      | Raises sub ->
          raises_match (Exn.invalid_arg ~sub) (fun () -> ignore (copied (outputs ()))))

let repro_tests =
  group "create_linear_with_vars › repros"
    (List.map repro (repros @ mixed)
    @ view_rows
    @ [
        test "a back edge of a call given a state twice copies it each trip"
          backedge_aliases;
      ])

(* A value ordered after a store but held by no storage: the store runs, and
   the value is what it names, whatever ran before its reader
   (runtime/test_schedule.py, test_value_order_dependency_requires_storage). *)
let boundary =
  group "create_linear_with_vars › boundaries"
    [
      test "a constant ordered after a store is the constant, and the store runs"
        (fun () ->
          let c = Ops.new_buffer ~slot:1 cpu 1 Int32
          and o = Ops.new_buffer ~slot:2 cpu 1 Int32 in
          let stored =
            Ops.store (Ops.index c [ Ops.int 0 ]) (Ops.int ~dtype:Int32 5)
          in
          let v = Ops.after (Ops.int ~dtype:Int32 1) [ stored ] in
          let big =
            Ops.sink [ Ops.after o [ Ops.store o (Shape.reshape v [ Int 1 ]) ] ]
          in
          computes big);
    ]

(* Two writes over one state

   Two writes over one state, neither listed over the state the other makes
   nor reading what the other left, are refused: which lands last would be
   the order of the sink's items. Listed in one After, or one over the After
   of the other, they are ordered. *)

let run_of x lo hi = Shape.shrink x [ Some (Int lo, Int hi) ]

let siblings =
  let refused big =
    raises_match
      (Exn.invalid_arg ~sub:"two writes over one state")
      (fun () -> Schedule.create_linear_with_vars big)
  in
  group "create_linear_with_vars › two writes over one state"
    [
      test "Afters over overlapping runs of one state are refused" (fun () ->
          let x = f32 ~n:8 1 in
          let over lo hi v =
            Ops.after (run_of x lo hi) [ Ops.store (run_of x lo hi) (fl v) ]
          in
          refused (Ops.sink [ over 0 6 1.; over 2 8 2. ]);
          refused (Ops.sink [ over 2 8 2.; over 0 6 1. ]));
      test "a read of a state beside another write over the state under it is \
            refused" (fun () ->
          let x = f32 ~n:8 1 and y = f32 ~n:8 2 in
          let a = Ops.after x [ Ops.store (run_of x 0 4) (fl 1.) ] in
          let b =
            Ops.after (run_of x 4 8) [ Ops.store (run_of x 4 8) (fl 2.) ]
          in
          refused (Ops.sink [ b; assign y Ops.O.(a + fl 10.) ]);
          refused (Ops.sink [ assign y Ops.O.(a + fl 10.); b ]));
      test "the refusal names both writes and how to order them" (fun () ->
          let x = f32 ~n:8 1 in
          raises
            (Invalid_argument
               "two writes over one state of the storage in slot 1: the \
                store into elements 0 to 3 and the store into elements 0 to 7 \
                are each over it; list one over the After of the other, or \
                list both in one After")
            (fun () ->
              Schedule.create_linear_with_vars
                (Ops.sink
                   [
                     assign x (fl 1.);
                     Ops.after x [ Ops.store (run_of x 0 4) (fl 2.) ];
                   ])));
      test "the same writes listed in one After are ordered as listed"
        (fun () ->
          let x = f32 ~n:8 1 and y = f32 ~n:8 2 in
          let a =
            Ops.after x
              [
                Ops.store (run_of x 0 4) (fl 1.);
                Ops.store (run_of x 4 8) (fl 2.);
              ]
          in
          computes (Ops.sink [ assign y Ops.O.(a + fl 10.) ]));
      test "a write over a state beside one it reads the result of is ordered"
        (fun () ->
          let x = f32 ~n:8 1 and y = f32 ~n:8 2 in
          let a = assign (run_of x 0 8) (fl 1.) in
          let y1 = assign y Ops.O.(a + fl 1.) in
          computes
            (Ops.sink [ a; Ops.after x [ Ops.store x Ops.O.(y1 * fl 2.) ] ]));
      test "a write a group or a value lists is ordered by its states alone"
        (fun () ->
          let x = f32 ~n:4 1 and y = f32 ~n:4 2 and z = f32 ~n:4 3 in
          let x1 = assign x (fl 1.) in
          let k = Ops.store x (fl 2.) in
          let x2 = Ops.after x1 [ k ] in
          let y1 = assign y (Ops.after (fl 3.) [ k ]) in
          let g = Ops.group [ k; Ops.store z (fl 5.) ] in
          List.iter
            (fun items ->
              let got = outcome items in
              equal bool ~msg:"computed" true (got <> None);
              equal (option (list write)) got (outcome (List.rev items)))
            [ [ x2; y1 ]; [ x2; g ] ]);
      test "an effect listed after another lands after it, whichever the sink \
            reaches first" (fun () ->
          let x = f32 1 and y = f32 2 in
          let e = Ops.store x (fl 2.) in
          let a = Ops.after x [ Ops.store x (fl 1.); e ] in
          let g = Ops.group [ e; Ops.store y (fl 5.) ] in
          let twos = Some (List.init 4 (fun i -> (1, i, `Float 2.))) in
          equal (option (list write)) twos (outcome [ a ]);
          equal (option (list write)) (outcome [ a ]) (outcome [ e; a ]);
          List.iter
            (fun items ->
              let got = outcome items in
              equal bool ~msg:"computed" true (got <> None);
              equal (option (list write)) got (outcome (List.rev items)))
            [ [ a; g ]; [ a; e ] ]);
      test "writes no After lists, over one state, are refused" (fun () ->
          let x = f32 1 in
          let a = assign x (fl 1.) in
          refused (Ops.sink [ Ops.store x (fl 1.); Ops.store x (fl 2.) ]);
          refused (Ops.sink [ Ops.store x (fl 2.); Ops.store x (fl 1.) ]);
          refused (Ops.sink [ a; Ops.store x (fl 2.) ]);
          refused (Ops.sink [ Ops.store x (fl 2.); a ]));
      test "a write no After lists is over the state its destination names"
        (fun () ->
          let x = f32 1 and y = f32 2 in
          let a = assign x (fl 1.) in
          let twos = Some (List.init 4 (fun i -> (1, i, `Float 2.))) in
          equal (option (list write)) twos
            (outcome [ a; Ops.store a Ops.O.(a + fl 1.) ]);
          equal (option (list write)) twos
            (outcome [ a; Ops.store x Ops.O.(a + fl 1.) ]);
          List.iter
            (fun items ->
              let big = Ops.sink items in
              computes big;
              equal (option (list write)) (outcome items)
                (outcome (List.rev items)))
            [
              [ a; Ops.store a Ops.O.(a + fl 1.) ];
              [ a; Ops.store x Ops.O.(a + fl 1.) ];
              [ Ops.store x (fl 1.); Ops.store y (fl 2.) ];
            ]);
      test "an After listing a write of other storage names its own state"
        (fun () ->
          let x = f32 ~n:4 1 and y = f32 ~n:4 2 and out = f32 ~n:4 3 in
          let x1 = assign x (fl 1.) in
          let x2 = assign x1 Ops.O.(x1 + fl 1.) in
          let r = Ops.after x1 [ Ops.store y (fl 5.) ] in
          let o = assign out Ops.O.(r + fl 0.) in
          let into_out items =
            match outcome items with
            | Some w -> List.filter (fun (s, _, _) -> s = 3) w
            | None -> failf "refused"
          in
          let ones = List.init 4 (fun i -> (3, i, `Float 1.)) in
          equal (list write) ones (into_out [ x2; o ]);
          equal (list write) ones (into_out [ o; x2 ]));
      test "an After over storage names the states its listed effects read"
        (fun () ->
          (* A loop's carry: its storage written before the loop, the loop
             listed over the storage itself. *)
          let x = f32 ~n:4 1 and y = f32 ~n:4 2 and out = f32 ~n:4 3 in
          let x1 = assign x (fl 1.) in
          let r = Ops.after x [ Ops.store y Ops.O.(x1 + fl 4.) ] in
          let o = assign out Ops.O.(r + fl 0.) in
          let items = [ o ] in
          computes (Ops.sink items);
          let ones = List.init 4 (fun i -> (3, i, `Float 1.)) in
          match outcome items with
          | Some w ->
              equal (list write) ones (List.filter (fun (s, _, _) -> s = 3) w)
          | None -> failf "refused");
      test "a write ordered after another through a value lands after it"
        (fun () ->
          let x = f32 ~n:4 1 and y = f32 ~n:4 2 in
          let m = Ops.store x (fl 3.) in
          let v = Ops.after y [ m ] in
          let items =
            [ Ops.after x [ m ]; Ops.after x [ Ops.store x Ops.O.(v + fl 1.) ] ]
          in
          computes (Ops.sink items);
          equal (option (list write)) (outcome items) (outcome (List.rev items)));
      test "a write listed over a state ordered after it is refused" (fun () ->
          let x = f32 ~n:4 1 and y = f32 ~n:4 2 in
          let e = Ops.store x (fl 1.) in
          let p = Ops.after x [ Ops.store x Ops.O.(Ops.after y [ e ] + fl 1.) ] in
          raises_match
            (Exn.invalid_arg ~sub:"an effect is listed over a state that needs it")
            (fun () ->
              Schedule.create_linear_with_vars (Ops.sink [ Ops.after p [ e ] ])));
      test "a kernel graph with two writes over one state is refused"
        (fun () ->
          raises_match
            (Exn.invalid_arg ~sub:"two writes over one state")
            (fun () ->
              Schedule.create_schedule
                (Golden.sink "write_over_written_kernels.golden")));
    ]

(* Calls of bodies precompiled or inlined, at any depth. *)
let call_writes =
  let param n slot = Shape.param ~shape:[ Int n ] ~device:cpu slot Float32 in
  let x = f32 ~n:8 1 and y = f32 ~n:8 2 in
  group "create_linear_with_vars › calls' writes"
    [
      test "an argument a call writes is its storage as the call runs"
        (fun () ->
          let body =
            Ops.sink [ Ops.store (param 8 0) Ops.O.(param 8 1 + param 8 0) ]
          in
          let twos = assign x (fl 2.) in
          List.iter
            (fun precompile ->
              (* [x] names the state before the store, the call runs after
                 it. *)
              let c = Ops.call ~precompile body [ x; Ops.O.(fl 3. + twos) ] in
              let items = [ Ops.after x [ c ] ] in
              computes (Ops.sink items);
              equal (option (list write))
                (Some (List.init 8 (fun i -> (1, i, `Float 7.))))
                (outcome items))
            [ false; true ]);
      test "a body that inlines a call writes what the call writes" (fun () ->
          let inner =
            Ops.sink [ Ops.store (param 4 0) Ops.O.(param 4 1 + fl 1.) ]
          in
          let run lo slot = Shape.shrink (param 8 slot) [ Some (Int lo, Int (lo + 4)) ] in
          let outer = Ops.sink [ Ops.call ~precompile:false inner [ run 4 0; run 0 1 ] ] in
          (* [y] holds 3, -1, 6, 2 first ([filled]). *)
          let fours = List.mapi (fun i v -> (1, 4 + i, `Float v)) [ 4.; 0.; 7.; 3. ] in
          List.iter
            (fun precompile ->
              let items =
                [ Ops.after x [ Ops.call ~precompile outer [ x; y ] ] ]
              in
              computes (Ops.sink items);
              equal (option (list write)) (Some fours) (outcome items))
            [ false; true ]);
      test "a loop of a call in a body reads its rows as each trip starts" (fun () ->
          let b = f32 ~n:16 3 in
          let add1 =
            Ops.sink [ Ops.store (param 4 0) Ops.O.(param 4 1 + fl 1.) ]
          in
          let whole = param 16 0 in
          let r = Ops.range ~axis_type:Loop (Int 4) [ 100 ] in
          let row =
            Shape.shrink whole
              [ Some (Sym Ops.O.(r * int 4), Sym Ops.O.((r * int 4) + int 4)) ]
          in
          let flipped = Shape.flip (Shape.shrink whole [ Some (Int 4, Int 8) ]) [ 0 ] in
          let c = Ops.call ~precompile:true add1 [ row; flipped ] in
          let body = Ops.sink [ Ops.after whole [ Ops.end_ c [ r ] ] ] in
          List.iter
            (fun precompile ->
              computes
                (Ops.sink [ Ops.after b [ Ops.call ~precompile body [ b ] ] ]))
            [ true; false ]);
    ]

let () =
  exit
    (run "Tolk.Schedule"
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
         by_storage;
         flatten;
         store_after;
         views;
         runs;
         variables;
         loops;
         merged_calls;
         on_disk;
         joins;
         views_of_written;
         aliased;
         relation;
         writes;
         boundary;
         siblings;
         call_writes;
         group "create_linear_with_vars › laws" [ values_law; sink_order_law ];
         group "create_schedule › laws" [ copy_iff_cycle ];
         repro_tests;
       ])
