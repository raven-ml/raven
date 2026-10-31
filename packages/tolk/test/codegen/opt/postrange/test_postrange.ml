(* Tests of Tolk.Postrange: a kernel optimised by a sequence of optimisations is
   the kernel tinygrad makes, a refused optimisation leaves the scheduler as it
   was, and optimising keeps what the kernel writes. *)

open Windtrap
open Tolk
module K = Postrange.Scheduler

(* Goldens *)

let renderers =
  List.map
    (fun cell -> (cell "renderer", Kernel_opts.renderer_of_row cell))
    (Golden.rows "renderers.golden")

let renderer name = List.assoc name renderers
let kernels = Hashtbl.create 64

let kernel name =
  match Hashtbl.find_opt kernels name with
  | Some k -> k
  | None ->
      let k = Golden.sink (name ^ ".golden") in
      Hashtbl.add kernels name k;
      k

type case = {
  name : string;
  kernel : string;
  renderer : string;
  opts : Opt.t list;
  settings : Setting.binding list;
  refused : (string * int) option;
      (** How tinygrad refuses an optimisation, and how many apply before it. *)
}

let case_of_row cell =
  {
    name = cell "case";
    kernel = cell "kernel";
    renderer = cell "renderer";
    opts = Kernel_opts.opts_of_cell (cell "opts");
    settings = Kernel_opts.settings_of_cell (cell "context");
    refused =
      (match cell "outcome" with
      | "ok" -> None
      | error -> Some (error, int_of_string (cell "refused")));
  }

let recorded_cases = List.map case_of_row (Golden.rows "cases.golden")
let case name = List.find (fun c -> c.name = name) recorded_cases

(* The goldens were recorded without colours. *)
let recorded ?(settings = []) f =
  Setting.context (Setting.B (Setting.no_color, true) :: settings) f

let info ast = match Ops.arg ast with Kernel k -> k | _ -> Ops.kernel_info ()

let asking opts ast =
  Ops.replace ~arg:(Kernel { (info ast) with opts_to_apply = Some opts }) ast

let never_hand_coded _ =
  fail "apply_opts chose optimisations for a kernel that asks for its own"

let optimize c =
  recorded ~settings:c.settings (fun () ->
      Postrange.apply_opts ~hand_coded:never_hand_coded
        (asking c.opts (kernel c.kernel))
        (renderer c.renderer))

(* The scheduler apply_opts optimises: its weak output axes made global. *)
let scheduler c =
  let k = K.v (kernel c.kernel) (renderer c.renderer) in
  K.convert_loop_to_global k;
  k

let axis_type = Testable.make ~pp:Ops.Axis_type.pp ~equal:Ops.Axis_type.equal

let keeps_writes before after =
  equal (list Kernel_opts.write)
    (Kernel_opts.writes before)
    (Kernel_opts.writes after)

let raises_invalid_arg f = raises_match (fun e -> Exn.invalid_arg e) f

(* [apply k o] applies [o], which must apply. *)
let apply ?append_opt k o =
  match K.apply_opt ?append_opt k o with
  | Ok axes -> axes
  | Error msg -> failf "%a does not apply: %s" Opt.pp o msg

(* An optimisation that does not apply, tinygrad's KernelOptError, is an
   [Error]. tinygrad raises ValueError where a condition it checks cannot be
   decided, and the port Invalid_argument. *)
let refuses_as error f =
  match error with
  | "KernelOptError" -> is_true ~msg:"refused" (Result.is_error (f ()))
  | "ValueError" -> raises_invalid_arg f
  | error -> failf "no refusal %s" error

let split ?(top = false) axis amount target =
  Opt.Split { axis; amount; target; top }

(* Printing as tinygrad's tables do *)

let repr_list pp l =
  Format.asprintf "[%a]"
    (Format.pp_print_list ~pp_sep:(fun ppf () -> Format.fprintf ppf ", ") pp)
    l

let sint ppf = function
  | Ops.Int n -> Format.pp_print_int ppf n
  | Sym u -> Format.pp_print_string ppf (Render.render u)

let size r =
  match Ops.vmax r with
  | `Int n -> Bigint.succ n
  | v -> failf "a range has the bound %a" (Testable.pp Dtypes.value) v

let axis_type_name t =
  let s = Format.asprintf "%a" Ops.Axis_type.pp t in
  String.sub s 9 (String.length s - 9)

let made_cell rngs =
  String.concat " "
    (List.map
       (fun r ->
         Printf.sprintf "%s:%s:%s"
           (String.concat "_" (List.map string_of_int (Ops.axis_id r)))
           (Bigint.to_string (size r))
           (axis_type_name (Ops.axis_type r)))
       rngs)

let escaped s = String.concat "\\e" (String.split_on_char '\027' s)

(* Optimising *)

let optimised =
  group "apply_opts optimises a kernel as tinygrad does"
    (List.map
       (fun c ->
         match c.refused with
         | None -> Golden.graph (c.name ^ ".golden") (fun () -> optimize c)
         | Some _ ->
             (* apply_opts raises Invalid_argument where apply_opt refuses *)
             test (c.name ^ " is refused") (fun () ->
                 raises_invalid_arg (fun () -> optimize c)))
       recorded_cases)

let leaves_as_it_was c (error, refused) =
  recorded ~settings:c.settings (fun () ->
      let k = scheduler c in
      List.iteri (fun i o -> if i < refused then ignore (apply k o)) c.opts;
      let ast = K.ast k and applied = K.applied_opts k in
      refuses_as error (fun () -> K.apply_opt k (List.nth c.opts refused));
      equal Uops.uop ast (K.ast k);
      equal (list Kernel_opts.opt) applied (K.applied_opts k))

let refusals =
  let refused =
    List.filter_map
      (fun c -> Option.map (fun n -> (c, n)) c.refused)
      recorded_cases
  in
  cases "a refused optimisation leaves the scheduler as it was"
    ~id:(fun (c, _) -> c.name)
    refused
    (fun (c, n) -> leaves_as_it_was c n)

let axes =
  group "the scheduler's axes after optimising are tinygrad's"
    [
      Golden.cases "axes.golden" (fun cell ->
          let c = case (cell "case") in
          recorded ~settings:c.settings (fun () ->
              let k = scheduler c in
              let made = List.fold_left (fun _ o -> apply k o) [] c.opts in
              let ints = repr_list Format.pp_print_int in
              let count column n =
                equal ~msg:column int (int_of_string (cell column)) n
              in
              let list column l =
                equal ~msg:column string (cell column) (ints l)
              in
              count "shape_len" (K.shape_len k);
              equal ~msg:"full_shape" string (cell "full_shape")
                (repr_list sint (K.full_shape k));
              equal ~msg:"axis_types" string (cell "axis_types")
                (repr_list Ops.Axis_type.pp (K.axis_types k));
              list "reduce_axes" (K.axes_of ~reduce:true k []);
              list "upcastable_dims" (K.upcastable_dims k);
              list "upcastable_reduce_dims" (K.upcastable_dims ~reduce:true k);
              equal ~msg:"upcast_size" string (cell "upcast_size")
                (Format.asprintf "%a" sint (K.upcast_size k));
              count "group_for_reduces" (K.group_for_reduces k);
              count "reduceops" (List.length (K.reduceops k));
              count "bufs" (List.length (K.bufs k));
              equal ~msg:"colored_shape" string (cell "colored_shape")
                (K.colored_shape k);
              equal ~msg:"made" string (cell "made") (made_cell made)));
    ]

(* [uncoloured s] is the escaped text [s] without its colour codes. *)
let uncoloured s =
  let b = Buffer.create (String.length s) in
  let n = String.length s in
  let rec go i =
    if i < n then
      if i + 2 < n && String.sub s i 3 = "\\e[" then
        go (String.index_from s i 'm' + 1)
      else (
        Buffer.add_char b s.[i];
        go (i + 1))
  in
  go 0;
  Buffer.contents b

let colors =
  group
    "the shape of a kernel is coloured by the roles of its axes, and its name \
     is not"
    [
      Golden.cases "colors.golden" (fun cell ->
          Setting.context
            [ B (Setting.no_color, false) ]
            (fun () ->
              let k =
                K.v (kernel (cell "kernel")) (renderer (cell "renderer"))
              in
              K.convert_loop_to_global k;
              List.iter
                (fun o -> ignore (apply k o))
                (Kernel_opts.opts_of_cell (cell "opts"));
              equal string (cell "colored_shape") (escaped (K.colored_shape k));
              equal string
                (uncoloured (cell "name"))
                (escaped (info (K.get_optimized_ast k)).name)));
    ]

(* Schedulers *)

let sum_rows_64 () = K.v (kernel "sum_rows_64") (renderer "cpu")
let upcast_4 = split 0 4 Upcast

(* Kernels over [a[r0 * 8 + r1]] whose reduction, or end, closes
   [closes r0 r1]. *)
let closing ~by_end closes =
  let r0 = Ops.range (Int 4) [ 0 ] and r1 = Ops.range (Int 8) [ 1 ] in
  let out = Shape.param ~shape:[ Int 32 ] 0 Float32
  and a = Shape.param ~shape:[ Int 32 ] 1 Float32 in
  let i = Ops.O.((r0 * Ops.int 8) + r1) in
  let x = Ops.index a [ i ] in
  let closer =
    if by_end then Ops.end_ (Ops.store (Ops.index out [ i ]) x) (closes r0 r1)
    else
      Ops.store
        (Ops.index out [ Ops.int 0 ])
        (Ops.reduce x Op.Add (closes r0 r1))
  in
  Ops.sink ~kernel:(Ops.kernel_info ()) [ closer ]

let closings =
  List.concat_map
    (fun (what, closes) ->
      [
        ("a reduction closing " ^ what, closing ~by_end:false closes);
        ("an end closing " ^ what, closing ~by_end:true closes);
      ])
    [
      ("distinct ranges", fun r0 r1 -> [ r0; r1 ]);
      ( "an expression of ranges",
        fun r0 r1 -> [ Ops.O.((r0 * Ops.int 8) + r1) ] );
      ("a range twice", fun r0 r1 -> [ r0; r1; r0 ]);
    ]

let flattened ast =
  Ops.graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:() ast
    (After_sources Simplify.pm_flatten_range)

let schedulers =
  group "Scheduler"
    [
      cases "v's kernel has the ranges its reductions and ends close flattened"
        ~id:fst closings (fun (_, ast) ->
          equal Uops.uop (flattened ast) (K.ast (K.v ast (renderer "cpu"))));
      cases "v's kernel of a recorded kernel is it flattened"
        ~id:(fun c -> c.name) recorded_cases (fun c ->
          let ast = kernel c.kernel in
          equal Uops.uop (flattened ast)
            (K.ast (K.v ast (renderer c.renderer))));
      cases "a copy's kernel after optimising is the kernel flattened"
        ~id:(fun c -> c.name)
        (List.filter (fun c -> c.refused = None) recorded_cases)
        (fun c ->
          recorded ~settings:c.settings (fun () ->
              let k = scheduler c in
              List.iter (fun o -> ignore (apply k o)) c.opts;
              equal Uops.uop (flattened (K.ast k)) (K.ast (K.copy k))));
      test "v reads the optimisations its kernel's argument lists" (fun () ->
          let k = K.v (Golden.sink "matmul_0.golden") (renderer "metal") in
          equal (list Kernel_opts.opt) [ split 1 32 Local ] (K.applied_opts k));
      test "an optimisation applied to a copy leaves the original as it was"
        (fun () ->
          let k = sum_rows_64 () in
          let ast = K.ast k and copy = K.copy k in
          ignore (apply copy upcast_4);
          equal Uops.uop ast (K.ast k);
          equal (list Kernel_opts.opt) [] (K.applied_opts k));
      test "an optimisation applied to the original leaves its copy as it was"
        (fun () ->
          let k = sum_rows_64 () in
          let copy = K.copy k in
          ignore (apply k upcast_4);
          equal Uops.uop (K.ast (sum_rows_64 ())) (K.ast copy);
          equal (list Kernel_opts.opt) [] (K.applied_opts copy));
      test "apply_opt records each optimisation it applies, in order" (fun () ->
          let k = sum_rows_64 () in
          ignore (apply k upcast_4);
          ignore (apply k (split 2 4 Upcast));
          equal (list Kernel_opts.opt)
            [ upcast_4; split 2 4 Upcast ]
            (K.applied_opts k));
      test "apply_opt ~append_opt:false leaves the optimisation unrecorded"
        (fun () ->
          let k = sum_rows_64 () in
          ignore (apply ~append_opt:false k upcast_4);
          equal (list Kernel_opts.opt) [] (K.applied_opts k));
      test "get_optimized_ast names the kernel name_override" (fun () ->
          let k = sum_rows_64 () in
          ignore (apply k upcast_4);
          let ast = K.get_optimized_ast ~name_override:"fused" k in
          equal string "fused" (info ast).name;
          equal (list Kernel_opts.opt) [ upcast_4 ] (info ast).applied_opts;
          equal (option string) (Some "1")
            (Option.map (Format.asprintf "%a" Ops.Tag.pp) (Ops.tag ast)));
      test
        "convert_loop_to_global leaves a kernel for a renderer without locals"
        (fun () ->
          let k = sum_rows_64 () in
          let ast = K.ast k in
          K.convert_loop_to_global k;
          equal Uops.uop ast (K.ast k));
    ]

(* Splitting *)

let shifts =
  let axis n = List.nth (K.rngs (sum_rows_64 ())) n in
  let shifted ?top ?new_rng n amount target =
    let k = sum_rows_64 () in
    let made = K.shift_to ?top ?new_rng k (axis n) amount target in
    (k, made)
  in
  let targets = [ (0, Opt.Upcast); (0, Local); (1, Upcast); (1, Local) ] in
  let splits =
    List.concat_map
      (fun (n, target) ->
        List.concat_map
          (fun amount ->
            [ (n, target, amount, false); (n, target, amount, true) ])
          [ 2; 4; 8; 16; 32; 64 ])
      targets
  in
  let name (n, target, amount, top) =
    Printf.sprintf "axis %d by %d to %s%s" n amount
      (match target with Opt.Upcast -> "upcast" | Local -> "local")
      (if top then " from the top" else "")
  in
  group "shift_to"
    [
      test "splits an axis into its quotient and a new axis of the amount"
        (fun () ->
          let k, (quotient, amount) = shifted 0 4 Upcast in
          equal Dtypes.z (Bigint.of_int 16) (size quotient);
          equal Dtypes.z (Bigint.of_int 4) (size amount);
          equal axis_type Weak (Ops.axis_type quotient);
          equal axis_type Upcast (Ops.axis_type amount);
          let slice = Ops.backward_slice ~calls:Skip (K.ast k) in
          equal ~msg:"the axis is gone" bool false
            (Ops.Nodes.mem (axis 0) slice);
          equal ~msg:"its quotient is used" bool true
            (Ops.Nodes.mem quotient slice);
          equal ~msg:"the new axis is used" bool true
            (Ops.Nodes.mem amount slice));
      test "makes the new axis of the range it is given" (fun () ->
          let new_rng = Ops.range ~axis_type:Upcast (Int 4) [ 99 ] in
          let _, (_, made) = shifted ~new_rng 0 4 Upcast in
          equal Uops.uop new_rng made);
      test "raises Invalid_argument on a target its axis's role cannot split to"
        (fun () ->
          let k = K.v (kernel "custom_gemm") (renderer "cpu") in
          let loop =
            List.find
              (fun r -> Ops.op r = Range && Ops.axis_type r = Loop)
              (Ops.toposort ~calls:Skip (K.ast k))
          in
          raises_invalid_arg (fun () -> K.shift_to k loop 4 Upcast));
      test "raises Invalid_argument on an amount of 1" (fun () ->
          raises_invalid_arg (fun () -> shifted 0 1 Upcast));
      test "raises Invalid_argument on an amount that does not divide the axis"
        (fun () -> raises_invalid_arg (fun () -> shifted 0 3 Upcast));
      cases "keeps the kernel's writes, once flattened" ~id:name splits
        (fun (n, target, amount, top) ->
          let k, _ = shifted ~top n amount target in
          keeps_writes (kernel "sum_rows_64") (K.get_optimized_ast k));
    ]

let split_targets =
  group "split_targets"
    [
      test "upcasts come from global, local and weak axes" (fun () ->
          equal (list axis_type) [ Global; Local; Weak ]
            (Postrange.split_targets Upcast));
      test "locals come from global and weak axes" (fun () ->
          equal (list axis_type) [ Global; Weak ]
            (Postrange.split_targets Local));
    ]

(* Choosing optimisations *)

let applied ast = (info ast).applied_opts
let never_searched _ = fail "apply_opts searched a kernel it must not optimise"

let upcasting k =
  ignore (apply k upcast_4);
  k

let without_opts ast =
  Ops.replace ~arg:(Kernel { (info ast) with opts_to_apply = None }) ast

let dispatch =
  let cpu = renderer "cpu" and metal = renderer "metal" in
  group "apply_opts"
    [
      test "returns a kernel it has tagged as it is" (fun () ->
          let optimized = Golden.sink "matmul_0.golden" in
          equal Uops.uop optimized
            (Postrange.apply_opts ~hand_coded:never_hand_coded optimized metal));
      test
        "hand-optimises a kernel that asks for nothing, after making its weak \
         outputs global" (fun () ->
          let calls = ref 0 in
          let hand_coded k =
            incr calls;
            equal (list axis_type) [ Global; Weak ] (K.axis_types k);
            equal (list int) [ 1 ] (K.axes_of ~reduce:true k []);
            upcasting k
          in
          let ast =
            Postrange.apply_opts ~hand_coded (kernel "sum_rows_64") metal
          in
          equal int 1 !calls;
          equal (list Kernel_opts.opt) [ upcast_4 ] (applied ast));
      test "hand-optimises a kernel without kernel information" (fun () ->
          let bare = Ops.replace ~arg:No_arg (kernel "sum_rows_64") in
          let ast =
            recorded (fun () ->
                Postrange.apply_opts ~hand_coded:upcasting bare cpu)
          in
          equal (list Kernel_opts.opt) [ upcast_4 ] (applied ast);
          equal string "r_16_4_64" (info ast).name);
      test "searches with beam instead, when given" (fun () ->
          let ast =
            Postrange.apply_opts ~beam:upcasting ~hand_coded:never_hand_coded
              (kernel "sum_rows_64") cpu
          in
          equal (list Kernel_opts.opt) [ upcast_4 ] (applied ast));
      test "applies nothing to a kernel that asks for no optimisation"
        (fun () ->
          let ast =
            Postrange.apply_opts ~beam:never_searched
              ~hand_coded:never_hand_coded
              (asking [] (kernel "sum_rows_64"))
              cpu
          in
          equal (list Kernel_opts.opt) [] (applied ast));
      test "noopt turns the hand-coded optimisations off" (fun () ->
          let ast =
            Setting.context
              [ B (Setting.noopt, true) ]
              (fun () ->
                Postrange.apply_opts ~hand_coded:never_hand_coded
                  (kernel "sum_rows_64") cpu)
          in
          equal (list Kernel_opts.opt) [] (applied ast));
      test "does not hand-optimise a kernel optimised before" (fun () ->
          let ast = kernel "sum_rows_64" in
          let optimized =
            Ops.replace
              ~arg:(Kernel { (info ast) with applied_opts = [ upcast_4 ] })
              ast
          in
          equal (list Kernel_opts.opt) [ upcast_4 ]
            (applied
               (Postrange.apply_opts ~hand_coded:never_hand_coded optimized cpu)));
      test "does not hand-optimise a kernel that buffers values" (fun () ->
          let ast = without_opts (kernel "stage_then_reduce") in
          equal (list Kernel_opts.opt) []
            (applied
               (Postrange.apply_opts ~hand_coded:never_hand_coded ast metal)));
      test "keeps a kernel's name other than test" (fun () ->
          let ast = kernel "sum_rows_64" in
          let named =
            Ops.replace
              ~arg:
                (Kernel
                   { (info ast) with name = "fused"; opts_to_apply = Some [] })
              ast
          in
          equal string "fused"
            (info (Postrange.apply_opts ~hand_coded:never_hand_coded named cpu))
              .name);
      test "names a kernel named test after its reduction and its axes"
        (fun () ->
          let ast =
            recorded (fun () ->
                Postrange.apply_opts ~hand_coded:never_hand_coded
                  (asking [] (kernel "sum_rows_64"))
                  cpu)
          in
          equal string "r_64_64" (info ast).name);
    ]

(* Laws *)

(* The interpreter evaluates neither a matrix multiply-accumulate, which a
   tensor core used in full makes, nor buffered values, loops or hardware
   indices. *)
let evaluable c =
  c.refused = None
  && (not
        (List.exists
           (function Opt.Tc { use_tc = 1; _ } -> true | _ -> false)
           c.opts))
  && not
       (Ops.op_in_backward_slice_with_self ~calls:Skip (kernel c.kernel)
          [ Op.Stage; Op.Backedge; Op.Special ])

let iterations c =
  List.fold_left
    (fun n r -> Bigint.mul n (size r))
    Bigint.one
    (K.rngs (scheduler c))

(* Kernels of more than 2^18 iterations are pinned by their goldens alone. *)
let keeps_kernel_writes =
  let evaluable = List.filter evaluable recorded_cases in
  let within lo hi =
    List.filter
      (fun c ->
        let n = iterations c in
        Bigint.(of_int lo < n && n <= of_int hi))
      evaluable
  in
  let keeps c = keeps_writes (kernel c.kernel) (optimize c) in
  group "optimising keeps a kernel's writes"
    [
      cases "small kernels" ~id:(fun c -> c.name) (within 0 (1 lsl 12)) keeps;
      slow
      @@ cases "large kernels"
           ~id:(fun c -> c.name)
           (within (1 lsl 12) (1 lsl 18))
           keeps;
    ]

(* The optimisation fuzzer: random optimisations of small kernels, each either
   refused, leaving the kernel as it was, or applied, keeping its writes. A
   tensor core is shaped for, without a matrix multiply-accumulate that the
   interpreter cannot evaluate. *)

let fuzzed_kernels =
  [
    "row_sum";
    "matmul_4x4";
    "cumsum";
    "max_then_sum";
    "masked_sum";
    "strided_conv";
    "sum_and_max";
    "flip_pad_sum";
    "where_max_multioutput";
    "tc_metal_0";
  ]

let fuzzed_opt =
  let open Gen in
  let axis = int_range (-1) 6
  and amount = of_list [ 0; 1; 2; 3; 4; 5; 8; 16; 32 ] in
  with_pp Opt.pp
    (one_of
       [
         (let+ axis
          and+ amount
          and+ target = of_list [ Opt.Upcast; Local ]
          and+ top = bool in
          Opt.Split { axis; amount; target; top });
         (let+ axis and+ amount in
          Opt.Padto { axis; amount });
         (let+ axis and+ with_axis = int_range (-1) 6 in
          Opt.Swap { axis; with_axis });
         (let+ axis = int_range 0 3 and+ tc_opt = int_range 0 2 in
          Opt.Tc { axis; tc_select = -1; tc_opt; use_tc = 2 });
       ])

let fuzzed_program =
  let open Gen in
  let pp ppf (name, ren, opts) =
    Format.fprintf ppf "%s on %s: %a" name ren
      (Format.pp_print_list
         ~pp_sep:(fun ppf () -> Format.fprintf ppf "; ")
         Opt.pp)
      opts
  in
  with_pp pp
    (triple (of_list fuzzed_kernels)
       (of_list [ "cpu"; "metal" ])
       (list ~size:(int_range 1 6) fuzzed_opt))

let fuzz (name, ren, opts) =
  let k = K.v (kernel name) (renderer ren) in
  K.convert_loop_to_global k;
  List.iter
    (fun o ->
      let ast = K.ast k and applied = K.applied_opts k in
      match K.apply_opt k o with
      | Ok _ -> ()
      | Error _ ->
          equal Uops.uop ast (K.ast k);
          equal (list Kernel_opts.opt) applied (K.applied_opts k))
    opts;
  cover "an optimisation applies" (K.applied_opts k <> []);
  cover "a tensor core applies"
    (List.exists (function Opt.Tc _ -> true | _ -> false) (K.applied_opts k));
  keeps_writes (kernel name) (K.get_optimized_ast k)

let fuzzer =
  slow
  @@ prop ~count:400
       "each random optimisation is refused as a whole or keeps the kernel's \
        writes"
       fuzzed_program fuzz

(* A scheduler's axis queries, between optimisations and copies, answer as
   those of a fresh copy of it, which has computed nothing yet. *)
let queries =
  let sched = abstract "k" in
  let kernels = [| "sum_rows_64"; "matmul_0"; "custom_gemm" |] in
  let opt i =
    let amount = List.nth [ 2; 4; 8 ] (i / 4 mod 3) in
    split (i mod 4) amount (if i >= 12 then Opt.Local else Upcast)
  in
  let make i = K.v (kernel kernels.(i)) (renderer "metal") in
  let fresh f k = f (K.copy k) in
  let ok k o = Result.is_ok (K.apply_opt k o) in
  let applies o k =
    let r = ok k o in
    cover "an optimisation applies" r;
    r
  in
  let copy k =
    cover "an optimised scheduler is copied" (K.applied_opts k <> []);
    K.copy k
  in
  let shape k = repr_list sint (K.full_shape k) in
  let ints = list int and uops = list Uops.uop in
  let query name w f =
    command name (sched ^-> returns w) (fresh f) f
  in
  stateful ~count:60 ~steps:12
    "a scheduler's axes answer as a fresh copy's, whatever came before"
    [
      command "v"
        (Gen.int_range 0 (Array.length kernels - 1) @-> makes sched)
        make make;
      command "copy" (sched ^-> makes sched) K.copy copy;
      command "apply_opt"
        (Gen.map opt (Gen.int_range 0 23) @-> sched ^-> returns bool)
        (fun o k -> ok k o)
        applies;
      command "apply_opt"
        (Gen.map opt (Gen.int_range 0 23) @-> sched ^-> returns bool)
        (fun o k -> ok k o)
        applies;
      query "rngs" uops K.rngs;
      query "full_shape" string shape;
      query "reduce_ranges" uops K.reduce_ranges;
      query "reduceops" uops K.reduceops;
      query "reduce axes" ints (fun k -> K.axes_of ~reduce:true k []);
      query "upcastable_dims" ints (fun k -> K.upcastable_dims k);
      query "upcast axes" ints (fun k -> K.axes_of k [ Upcast ]);
    ]

let () =
  exit
    (Windtrap.run "Tolk.Postrange"
       [
         optimised;
         refusals;
         axes;
         colors;
         schedulers;
         queries;
         shifts;
         split_targets;
         dispatch;
         keeps_kernel_writes;
         fuzzer;
       ])
