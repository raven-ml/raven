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
      test "get_optimized_ast names the kernel name_override, optimised"
        (fun () ->
          let k = sum_rows_64 () in
          ignore (apply k upcast_4);
          let ast = K.get_optimized_ast ~name_override:"fused" k in
          equal string "fused" (info ast).name;
          equal (list Kernel_opts.opt) [ upcast_4 ] (info ast).applied_opts;
          equal bool true (info ast).optimized);
      test
        "convert_loop_to_global leaves a kernel for a renderer without locals"
        (fun () ->
          let k = sum_rows_64 () in
          let ast = K.ast k in
          K.convert_loop_to_global k;
          equal Uops.uop ast (K.ast k));
    ]

(* Loops bounded by an axis

   A loop runs one count of trips, so an axis its bound reads cannot become a
   thread's lanes, where each lane would count its own. *)

(* [out[i * 8 + n] = j * j] over [j < x[i]], for four [i] and eight [n]. *)
let bounded_rows () =
  let i = Ops.range (Int 4) [ 0 ] and n = Ops.range (Int 8) [ 1 ] in
  let x = Shape.param ~shape:[ Int 4 ] 1 Int32
  and out = Shape.param ~shape:[ Int 32 ] 0 Int32 in
  let count = Ops.load (Ops.index x [ i ]) [] in
  let j = Ops.range ~axis_type:Loop ~dtype:Int32 (Sym count) [ 2 ] in
  let at = Ops.index out [ Ops.O.((i * Ops.int 8) + n) ] in
  let st = Ops.end_ (Ops.store at Ops.O.(j * j)) [ j ] in
  (i, n, Ops.sink ~kernel:(Ops.kernel_info ()) [ Ops.end_ st [ i; n ] ])

let position r k =
  match List.find_index (fun u -> u == r) (K.rngs k) with
  | Some i -> i
  | None -> failf "%s is not an axis" (Ops.range_str r)

(* [ast] with a loop beside its stores, over [j < x[r]] for its first axis
   [r], storing [j] at [r]. *)
let with_bounded_loop k =
  let ast = K.ast k in
  let r = List.hd (K.rngs k) in
  let x = Shape.param ~shape:[ Int 4096 ] 10 Int32
  and y = Shape.param ~shape:[ Int 4096 ] 11 Int32 in
  let count = Ops.load (Ops.index x [ r ]) [] in
  let j = Ops.range ~axis_type:Loop ~dtype:Int32 (Sym count) [ 99 ] in
  let loop =
    Ops.end_ (Ops.end_ (Ops.store (Ops.index y [ r ]) j) [ j ]) [ r ]
  in
  Ops.replace ast ~src:(Ops.src ast @ [ loop ])

let tc = Opt.Tc { axis = 0; tc_select = -1; tc_opt = 0; use_tc = 1 }

let bounded_loops =
  group "loops bounded by an axis"
    [
      test "read_by_bounds is the axes a loop's bound reads" (fun () ->
          let i, _, ast = bounded_rows () in
          equal (list Uops.uop) [ i ]
            (K.read_by_bounds (K.v ast (renderer "cpu"))));
      test "upcastable_dims leaves out an axis a loop's bound reads" (fun () ->
          let i, n, ast = bounded_rows () in
          let k = K.v ast (renderer "cpu") in
          equal (list int) [ position n k ]
            (List.filter
               (fun a -> a = position i k || a = position n k)
               (K.upcastable_dims k)));
      test
        "apply_opt refuses an upcast of an axis a loop's bound reads, leaving \
         the kernel as it was" (fun () ->
          let i, _, ast = bounded_rows () in
          let k = K.v ast (renderer "cpu") in
          let before = K.ast k in
          equal
            (result (list Uops.uop) string)
            (Error "a loop's bound reads axis 0")
            (K.apply_opt k (split (position i k) 4 Upcast));
          equal Uops.uop before (K.ast k);
          equal (list Kernel_opts.opt) [] (K.applied_opts k));
      test "apply_opt upcasts an axis no loop's bound reads" (fun () ->
          let _, n, ast = bounded_rows () in
          let k = K.v ast (renderer "cpu") in
          is_ok ~pp:Format.pp_print_string
            (K.apply_opt k (split (position n k) 4 Upcast)));
      test "apply_opt splits an axis a loop's bound reads into threads"
        (fun () ->
          let i, _, ast = bounded_rows () in
          let k = K.v ast (renderer "cuda") in
          is_ok ~pp:Format.pp_print_string
            (K.apply_opt k (split (position i k) 2 Local)));
      test "a group is refused where a loop's bound reads an axis" (fun () ->
          let k = K.v (kernel "sum_rows_64") (renderer "cuda") in
          let group k =
            let axis = List.hd (K.axes_of ~reduce:true k []) in
            K.apply_opt k (split axis 4 Local)
          in
          is_ok ~pp:Format.pp_print_string (group (K.copy k));
          equal
            (result (list Uops.uop) string)
            (Error "groups need loops whose bounds read no axis")
            (group (K.v (with_bounded_loop k) (renderer "cuda"))));
      test "a tensor core is refused where a loop's bound reads an axis"
        (fun () ->
          let k = K.v (kernel "tc_metal_0") (renderer "metal") in
          is_ok ~pp:Format.pp_print_string (K.apply_opt (K.copy k) tc);
          let bounded = K.v (with_bounded_loop k) (renderer "metal") in
          equal
            (result (list Uops.uop) string)
            (Error "tensor cores need loops whose bounds read no axis")
            (K.apply_opt bounded tc));
    ]

(* A range's integer type *)

(* [out[i] = i] over 3 [i] of type [dt]. *)
let typed_rows dt =
  let i = Ops.range ~dtype:dt (Int 3) [ 0 ] in
  let out = Shape.param ~shape:[ Int 4 ] 0 Int32 in
  Ops.sink ~kernel:(Ops.kernel_info ())
    [ Ops.end_ (Ops.store (Ops.index out [ i ]) (Ops.cast i Int32)) [ i ] ]

let range_types =
  cases "apply_opt pads a range of a typed integer as a weak one"
    ~id:(Format.asprintf "%a" Dtype.pp)
    Dtype.[ Int32; Int64; Uint32 ]
    (fun dt ->
      let k = K.v (typed_rows dt) (renderer "cpu") in
      is_ok ~pp:Format.pp_print_string
        (K.apply_opt k (Opt.Padto { axis = 0; amount = 4 }));
      equal (list (option int)) [ Some 4 ]
        (List.map
           (function Ops.Int n -> Some n | Sym _ -> None)
           (K.full_shape k)))

(* Axes a call or custom code reads

   A call, or code in the target's language, a statement or an expression,
   takes one pointer or value for each source, so an axis they read cannot
   become a thread's lanes, nor take padding. A call or a statement runs once
   for each point of the axes they read, so an axis they do not read cannot
   take threads; an expression has no effect, and every thread may compute
   it. *)

(* An opaque operation of kind [o] on [&x[r]]: [f(&x[r])], the code
   [{0}[0] = 1;], or [z[0] = {0}[0]] of the expression [{0}[0]]. *)
let opaque o x r =
  match o with
  | Op.Call ->
      let p = Shape.param ~shape:[ Int 1 ] 0 Int32 in
      let f =
        Ops.sink
          [ Ops.store (Ops.index p [ Ops.int 0 ]) (Ops.int ~dtype:Int32 1) ]
      in
      Ops.call ~name:"f" f [ Ops.index x [ r ] ]
  | Op.Customi ->
      let z = Shape.param ~shape:[ Int 1 ] 12 Int32 in
      Ops.store
        (Ops.index z [ Ops.int 0 ])
        (Ops.v Customi
           ~src:[ Ops.index x [ r ] ]
           ~arg:(Code { code = "{0}[0]"; dtype = Int32 }))
  | _ ->
      Ops.v Custom
        ~src:[ Ops.index x [ r ] ]
        ~arg:(Code { code = "{0}[0] = 1;"; dtype = Void })

let opaques = [ Op.Call; Op.Custom; Op.Customi ]

(* The opaque operations that run as effects: an expression has none. *)
let effects = [ Op.Call; Op.Custom ]
let opaque_id o = Format.asprintf "%a" Op.pp o

(* [o] on [&x[i]] and [out[i * 8 + n] = n], for four [i] and eight [n]. *)
let called_rows o =
  let i = Ops.range (Int 4) [ 0 ] and n = Ops.range (Int 8) [ 1 ] in
  let x = Shape.param ~shape:[ Int 4 ] 1 Int32
  and out = Shape.param ~shape:[ Int 32 ] 0 Int32 in
  let at = Ops.index out [ Ops.O.((i * Ops.int 8) + n) ] in
  let st = Ops.store at (Ops.cast n Int32) in
  ( i,
    n,
    Ops.sink ~kernel:(Ops.kernel_info ())
      [ Ops.end_ (Ops.group [ opaque o x i; st ]) [ i; n ] ] )

(* [ast] with [o] beside its stores, on [&y[r]] for its first axis [r]. *)
let with_opaque o k =
  let ast = K.ast k in
  let r = List.hd (K.rngs k) in
  let y = Shape.param ~shape:[ Int 4096 ] 11 Int32 in
  Ops.replace ast ~src:(Ops.src ast @ [ Ops.end_ (opaque o y r) [ r ] ])

let refused axis = Error (Printf.sprintf "a call or custom code reads axis %d" axis)

let called_axes =
  let opaque_cases name f = cases name ~id:opaque_id opaques f in
  let effect_cases name f = cases name ~id:opaque_id effects f in
  group "axes a call or custom code reads"
    [
      opaque_cases "read_by_opaque is the axes its sources read" (fun o ->
          let i, _, ast = called_rows o in
          equal (list Uops.uop) [ i ]
            (K.read_by_opaque (K.v ast (renderer "cpu"))));
      opaque_cases "upcastable_dims leaves out an axis its sources read"
        (fun o ->
          let i, n, ast = called_rows o in
          let k = K.v ast (renderer "cpu") in
          equal (list int)
            [ position n k ]
            (List.filter
               (fun a -> a = position i k || a = position n k)
               (K.upcastable_dims k)));
      opaque_cases
        "apply_opt refuses an upcast of an axis its sources read, leaving the \
         kernel as it was" (fun o ->
          let i, _, ast = called_rows o in
          let k = K.v ast (renderer "cpu") in
          let before = K.ast k in
          equal
            (result (list Uops.uop) string)
            (refused (position i k))
            (K.apply_opt k (split (position i k) 4 Upcast));
          equal Uops.uop before (K.ast k);
          equal (list Kernel_opts.opt) [] (K.applied_opts k));
      opaque_cases "apply_opt refuses a pad of an axis its sources read"
        (fun o ->
          let i, _, ast = called_rows o in
          let k = K.v ast (renderer "cpu") in
          equal
            (result (list Uops.uop) string)
            (refused (position i k))
            (K.apply_opt k (Opt.Padto { axis = position i k; amount = 8 })));
      opaque_cases "apply_opt upcasts and pads an axis its sources do not read"
        (fun o ->
          let _, n, ast = called_rows o in
          let ok opt =
            let k = K.v ast (renderer "cpu") in
            is_ok ~pp:Format.pp_print_string
              (K.apply_opt k (opt (position n k)))
          in
          ok (fun axis -> split axis 4 Upcast);
          ok (fun axis -> Opt.Padto { axis; amount = 3 }));
      opaque_cases "apply_opt splits an axis its sources read into threads"
        (fun o ->
          let i, _, ast = called_rows o in
          let k = K.v ast (renderer "cuda") in
          is_ok ~pp:Format.pp_print_string
            (K.apply_opt k (split (position i k) 2 Local)));
      effect_cases "unread_by_opaque is the axes its sources do not read"
        (fun o ->
          let _, n, ast = called_rows o in
          equal (list Uops.uop) [ n ]
            (K.unread_by_opaque (K.v ast (renderer "cuda"))));
      effect_cases
        "convert_loop_to_global gives workgroups only to an axis its sources \
         read" (fun o ->
          let _, _, ast = called_rows o in
          let k = K.v ast (renderer "cuda") in
          K.convert_loop_to_global k;
          equal string "[AxisType.GLOBAL, AxisType.WEAK]"
            (repr_list Ops.Axis_type.pp (K.axis_types k)));
      effect_cases "apply_opt refuses a local split of an axis its sources do not \
                    read" (fun o ->
          let _, n, ast = called_rows o in
          let k = K.v ast (renderer "cuda") in
          equal
            (result (list Uops.uop) string)
            (Error (Printf.sprintf "a call or custom code does not read axis %d"
                      (position n k)))
            (K.apply_opt k (split (position n k) 2 Local)));
      effect_cases "a group is refused over a reduce axis its sources do not read"
        (fun o ->
          let k = K.v (kernel "sum_rows_64") (renderer "cuda") in
          let group k =
            let axis = List.hd (K.axes_of ~reduce:true k []) in
            K.apply_opt k (split axis 4 Local)
          in
          is_ok ~pp:Format.pp_print_string (group (K.copy k));
          let axis = List.hd (K.axes_of ~reduce:true k []) in
          equal
            (result (list Uops.uop) string)
            (Error (Printf.sprintf "a call or custom code does not read axis %d" axis))
            (group (K.v (with_opaque o k) (renderer "cuda"))));
      effect_cases "a tensor core is refused in a kernel with one" (fun o ->
          let k = K.v (kernel "tc_metal_0") (renderer "metal") in
          is_ok ~pp:Format.pp_print_string (K.apply_opt (K.copy k) tc);
          let refused o' =
            equal
              (result (list Uops.uop) string)
              (Error "tensor cores need a kernel without calls or custom code")
              (K.apply_opt (K.v o' (renderer "metal")) tc)
          in
          refused (with_opaque o k);
          (* one that reads no axis *)
          let y = Shape.param ~shape:[ Int 1 ] 11 Int32 in
          let ast = K.ast k in
          refused (Ops.replace ast ~src:(Ops.src ast @ [ opaque o y (Ops.int 0) ])));
      test "an expression's unread axes take threads" (fun () ->
          let _, n, ast = called_rows Op.Customi in
          let k = K.v ast (renderer "cuda") in
          equal (list Uops.uop) [] (K.unread_by_opaque k);
          is_ok ~pp:Format.pp_print_string
            (K.apply_opt (K.copy k) (split (position n k) 2 Local));
          K.convert_loop_to_global k;
          equal string "[AxisType.GLOBAL, AxisType.GLOBAL]"
            (repr_list Ops.Axis_type.pp (K.axis_types k)));
      test "a tensor core is refused where an expression reads an axis"
        (fun () ->
          let k = K.v (kernel "tc_metal_0") (renderer "metal") in
          equal
            (result (list Uops.uop) string)
            (Error "tensor cores need custom expressions that read no axis")
            (K.apply_opt
               (K.v (with_opaque Op.Customi k) (renderer "metal"))
               tc);
          let y = Shape.param ~shape:[ Int 1 ] 11 Int32 in
          let ast = K.ast k in
          is_ok ~pp:Format.pp_print_string
            (K.apply_opt
               (K.v
                  (Ops.replace ast
                     ~src:(Ops.src ast @ [ opaque Op.Customi y (Ops.int 0) ]))
                  (renderer "metal"))
               tc));
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

(* [out[i] = inp[i]] over 256 elements [i], through what its builder chose for
   its threads: workgroup memory, a register, a barrier after its store, a
   matrix multiply-accumulate, custom code, a call whose body holds a barrier;
   or through none of them. *)
let fixed_by reason =
  let open Ops.O in
  let buffer slot = Shape.param ~shape:[ Int 256 ] slot Float32 in
  let i = Ops.range (Int 256) [ 0 ] in
  let at b j = Ops.index b [ j ] in
  let value = Ops.load (at (buffer 1) i) [] in
  let through space n j =
    let b = Shape.placeholder ~slot:10 ~addrspace:space [ n ] Float32 in
    Ops.load (at (Ops.after b [ Ops.store (at b j) value ]) j) []
  in
  let store v = Ops.store (at (buffer 0) i) v in
  let effects =
    match reason with
    | "workgroup memory" -> store (through Local 256 i)
    | "a register" -> store (through Reg 1 (int 0))
    | "a barrier" -> Ops.barrier (store value) []
    | "a matrix multiply-accumulate" ->
        let lanes n =
          Shape.stack (List.init n (fun _ -> Ops.cast value Bfloat16))
        in
        let acc = Shape.stack (List.init 4 (fun _ -> value)) in
        let d =
          Ops.wmma (lanes 8) (lanes 4) ~acc ~dims:(8, 16, 16) ~threads:32
        in
        store (Ops.index d [ Ops.int ~dtype:Int32 0 ])
    | "custom code" ->
        Ops.group
          [
            store value;
            Ops.v Op.Custom ~arg:(Code { code = "f();"; dtype = Void });
          ]
    | "a call whose body holds a barrier" ->
        let p = Shape.param ~shape:[ Int 1 ] 0 Float32 in
        let fence =
          Ops.sink [ Ops.barrier (Ops.store (at p (int 0)) (Ops.float 1.)) [] ]
        in
        Ops.group
          [ store value; Ops.call ~name:"fence" fence [ at (buffer 2) i ] ]
    | _ -> store value
  in
  Ops.sink ~kernel:(Ops.kernel_info ()) [ Ops.end_ effects [ i ] ]

let fixing =
  [
    "workgroup memory";
    "a register";
    "a barrier";
    "a matrix multiply-accumulate";
    "custom code";
    "a call whose body holds a barrier";
  ]

let dispatch =
  let cpu = renderer "cpu" and metal = renderer "metal" in
  group "apply_opts"
    [
      test "returns a kernel already optimised as it is" (fun () ->
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
      cases
        "neither hand-optimises nor searches a kernel whose builder fixed its \
         threads with"
        ~id:Fun.id fixing (fun reason ->
          let k = fixed_by reason and cuda = renderer "cuda" in
          let optimise ?beam () =
            applied
              (Postrange.apply_opts ?beam ~hand_coded:never_hand_coded k cuda)
          in
          equal (list Kernel_opts.opt) [] (optimise ());
          equal (list Kernel_opts.opt) [] (optimise ~beam:never_searched ()));
      test "hand-optimises the same kernel through none of them" (fun () ->
          let calls = ref 0 in
          let hand_coded k =
            incr calls;
            k
          in
          let cuda = renderer "cuda" in
          ignore (Postrange.apply_opts ~hand_coded (fixed_by "none") cuda);
          equal int 1 !calls);
      test
        "applies the optimisations a kernel whose builder fixed its threads \
         asks for" (fun () ->
          let c = case "loop_counter_upcast" in
          equal (list Kernel_opts.opt) c.opts (applied (optimize c)));
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
      query "read_by_bounds" uops K.read_by_bounds;
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
         bounded_loops;
         range_types;
         called_axes;
         queries;
         shifts;
         split_targets;
         dispatch;
         keeps_kernel_writes;
         fuzzer;
       ])
