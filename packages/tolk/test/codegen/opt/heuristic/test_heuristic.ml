(* Tests of Tolk.Heuristic: the hand-coded optimisations of a real kernel are
   tinygrad's on each renderer and under each tensor core setting, they leave
   the scheduler they are given as it was, and they keep what the kernel
   writes. *)

open Windtrap
open Tolk
module K = Postrange.Scheduler

(* Goldens *)

let renderers =
  List.map
    (fun cell -> (cell "renderer", Kernel_opts.renderer_of_row cell))
    (Golden.rows "renderers.golden")

let renderer name = List.assoc name renderers
let kernels = Hashtbl.create 32

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
  settings : Setting.binding list;
  environment : string;
      (** The switch of the matrix-vector layouts, [MV], which is read once from
          the environment: a case under it runs in a process of its own (see
          dune), tagged by {!process}. *)
  opts : Opt.t list;  (** The optimisations tinygrad chose. *)
}

let case_of_row cell =
  let is_environment pair = String.starts_with ~prefix:"MV" pair in
  let pairs = String.split_on_char ' ' (cell "context") in
  {
    name = cell "case";
    kernel = cell "kernel";
    renderer = cell "renderer";
    settings =
      Kernel_opts.settings_of_cell
        (String.concat " "
           (List.filter (fun p -> not (is_environment p)) pairs));
    environment = String.concat " " (List.filter is_environment pairs);
    opts = Kernel_opts.opts_of_cell (cell "opts");
  }

let recorded_cases = List.map case_of_row (Golden.rows "cases.golden")
let processes = [ ("", ""); ("MV=0", "mv_0") ]

(* The cases of the process that runs with [environment], in a group tagged
   after it, which first checks that the process has it. *)
let in_process environment tests =
  let in_environment =
    List.filter (fun c -> c.environment = environment) recorded_cases
  in
  match List.assoc environment processes with
  | "" -> group "in the default environment" (tests in_environment)
  | tag ->
      let has_environment () =
        List.iter
          (fun pair ->
            match String.split_on_char '=' pair with
            | [ name; value ] ->
                equal ~msg:name (option string) (Some value)
                  (Sys.getenv_opt name)
            | _ -> failf "no variable %s" pair)
          (String.split_on_char ' ' environment)
      in
      Windtrap.tag tag
      @@ group environment
           (test
              ("the suite runs this group with " ^ environment)
              has_environment
           :: tests in_environment)

(* The goldens were recorded without colours. *)
let optimize c =
  Setting.context
    (Setting.B (Setting.no_color, true) :: c.settings)
    (fun () ->
      Postrange.apply_opts ~hand_coded:Heuristic.hand_coded_optimizations
        (kernel c.kernel) (renderer c.renderer))

let size r =
  match Ops.vmax r with
  | `Int n -> Bigint.succ n
  | v -> failf "a range has the bound %a" (Testable.pp Dtypes.value) v

(* Choosing *)

(* A golden's kernel information holds the optimisations applied, so the
   optimised kernel states them too. *)
let chosen =
  let tests cases =
    [
      group "the optimised kernel"
        (List.map
           (fun c -> Golden.graph (c.name ^ ".golden") (fun () -> optimize c))
           cases);
    ]
  in
  group "the optimisations chosen are tinygrad's"
    (List.map (fun (environment, _) -> in_process environment tests) processes)

let leaves_its_argument =
  let kernels =
    List.sort_uniq String.compare (List.map (fun c -> c.kernel) recorded_cases)
  in
  cases "hand_coded_optimizations leaves the scheduler it is given as it was"
    ~id:Fun.id kernels (fun name ->
      (* Metal's case, or the host's for a kernel only the host optimises. *)
      let plain =
        List.filter
          (fun c -> c.kernel = name && c.settings = [] && c.environment = "")
          recorded_cases
      in
      let c =
        match List.find_opt (fun c -> c.renderer = "metal") plain with
        | Some c -> c
        | None -> List.hd plain
      in
      let k = K.v (kernel name) (renderer c.renderer) in
      K.convert_loop_to_global k;
      let ast = K.ast k in
      let optimized = Heuristic.hand_coded_optimizations k in
      equal Uops.uop ast (K.ast k);
      equal (list Kernel_opts.opt) [] (K.applied_opts k);
      equal (list Kernel_opts.opt) c.opts (K.applied_opts optimized))

(* Laws *)

(* The interpreter cannot evaluate a matrix multiply-accumulate, which a tensor
   core used in full makes. The cases of the default process are enough, and
   of AVX2's those whose optimisations differ from SSE's. *)
let evaluable c =
  let as_sse c' =
    c'.kernel = c.kernel && c'.renderer = "cpu" && c'.settings = c.settings
    && c'.opts = c.opts
  in
  c.environment = ""
  && (c.renderer <> "avx2" || not (List.exists as_sse recorded_cases))
  && not
       (List.exists
          (function Opt.Tc { use_tc = 1; _ } -> true | _ -> false)
          c.opts)

let iterations c =
  let k = K.v (kernel c.kernel) (renderer c.renderer) in
  List.fold_left (fun n r -> Bigint.mul n (size r)) Bigint.one (K.rngs k)

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
  let keeps c =
    equal (list Kernel_opts.write)
      (Kernel_opts.writes (kernel c.kernel))
      (Kernel_opts.writes (optimize c))
  in
  group "the hand-coded optimisations keep a kernel's writes"
    [
      cases "small kernels" ~id:(fun c -> c.name) (within 0 (1 lsl 12)) keeps;
      slow
      @@ cases "large kernels"
           ~id:(fun c -> c.name)
           (within (1 lsl 12) (1 lsl 18))
           keeps;
    ]

(* Streaming *)

let bigint = Testable.make ~pp:Bigint.pp_print ~equal:Bigint.equal

(* The cases of a GPU renderer in the default process, their kernel as the
   heuristic receives it and as it leaves it. *)
let gpu_cases =
  List.filter_map
    (fun c ->
      if
        c.renderer = "cpu" || c.renderer = "avx2" || c.settings <> []
        || c.environment <> ""
      then
        None
      else
        let k = K.v (kernel c.kernel) (renderer c.renderer) in
        K.convert_loop_to_global k;
        Some (c, k, Heuristic.hand_coded_optimizations k))
    recorded_cases

let ranges_at k ~reduce types =
  List.map (List.nth (K.rngs k)) (K.axes_of ~reduce k types)

let accesses o =
  List.filter
    (fun u -> Ops.op u = Op.Index)
    (Ops.Nodes.to_list (Ops.backward_slice_with_self ~calls:Skip o))

let buffers o = List.map (fun a -> Ops.nth a 0) (accesses o)

(* The operands of the product [k]'s first reduce sums, if it sums one. *)
let operands k =
  match K.reduceop k with
  | Some r when Ops.op (Ops.nth r 0) = Op.Mul ->
      let m = Ops.nth r 0 in
      Some (Ops.nth m 0, Ops.nth m 1)
  | _ -> None

(* The outputs that share each element of [o] along the axes of [k] of a role
   in [types]: an axis [o] does not read counts its size, one it reads only
   as its quotient by [d] counts [d]. *)
let reuse k types o =
  let along r =
    if not (Ops.Nodes.mem r (Ops.ranges o)) then size r
    else
      let readers =
        List.filter
          (fun u -> List.memq r (Ops.src u))
          (Ops.Nodes.to_list (Ops.backward_slice_with_self ~calls:Skip o))
      in
      let divisor u =
        if Ops.op u = Op.Div && Ops.op (Ops.nth u 1) = Op.Const then
          match Ops.value (Ops.nth u 1) with `Int d -> Some d | _ -> None
        else None
      in
      match List.map divisor readers with
      | Some d :: rest when List.for_all (Option.equal Bigint.equal (Some d)) rest
        ->
          d
      | _ -> Bigint.one
  in
  List.fold_left
    (fun n r -> Bigint.mul n (along r))
    Bigint.one
    (ranges_at k ~reduce:false types)

(* Whether [o] streams a contraction: its threads split a contracted axis
   without grouping from the top or a tensor core. *)
let streams o =
  ranges_at o ~reduce:true [ Local ] <> []
  && not
       (List.exists
          (function
            | Opt.Tc _ | Opt.Split { top = true; _ } -> true | _ -> false)
          (K.applied_opts o))

(* The main access of the streamed operand [w] of [o]: the one with a range of
   unit stride whose ranges of unit or constant stride that a contraction
   closes span the most bytes. *)
let main_access o w =
  let red = ranges_at o ~reduce:true [] in
  let terms a = Ops.split_uop (Ops.get_idx (Ops.nth a 1)) Op.Add in
  let range t =
    if Ops.op t = Op.Range then Some t
    else if Ops.op t = Op.Mul && Ops.op (Ops.nth t 1) = Op.Const
            && Ops.op (Ops.nth t 0) = Op.Range
    then Some (Ops.nth t 0)
    else None
  in
  let bytes a =
    List.fold_left
      (fun n t ->
        match range t with
        | Some r when List.memq r red -> Bigint.mul n (size r)
        | _ -> n)
      (Bigint.of_int (Dtype.itemsize (Ops.dtype a)))
      (terms a)
  in
  List.fold_left
    (fun best a ->
      match best with
      | Some b when Bigint.compare (bytes b) (bytes a) >= 0 -> best
      | _ -> Some a)
    None
    (List.filter
       (fun a -> List.exists (fun t -> Ops.op t = Op.Range) (terms a))
       (accesses w))

(* Whether the main access of [o] in [k] reads a contracted range at unit
   stride: of two operands of equal reuse, that one is streamed. *)
let along_reduce k o =
  match main_access k o with
  | None -> false
  | Some a ->
      let red = ranges_at k ~reduce:true [] in
      List.exists
        (fun t -> Ops.op t = Op.Range && List.memq t red)
        (Ops.split_uop (Ops.get_idx (Ops.nth a 1)) Op.Add)

(* The streamed operand of [k] and its reuse, and the same operand in the
   kernel [o] the heuristic makes of [k]. *)
let streamed k o =
  match (operands k, operands o) with
  | Some (a, b), Some (a', b') ->
      let ra = reuse k [ Global ] a and rb = reuse k [ Global ] b in
      let c = Bigint.compare ra rb in
      let w, r =
        if c < 0 || (c = 0 && (along_reduce k a || not (along_reduce k b)))
        then (a, ra)
        else (b, rb)
      in
      let same x = buffers x = buffers w in
      if same a' then Some (w, r, a') else if same b' then Some (w, r, b') else None
  | _ -> None

let rec largest_divisor n at_most =
  if at_most <= 1 then 1
  else if Bigint.(equal (rem n (of_int at_most)) zero) then at_most
  else largest_divisor n (at_most - 1)

let streamed_cases_all =
  List.filter_map
    (fun (c, k, o) ->
      if streams o then Option.map (fun s -> (c, k, o, s)) (streamed k o)
      else None)
    gpu_cases

(* The cases streamed along the reduce, the rows layout, whose lanes and
   unroll the laws below state. *)
let streamed_cases =
  List.filter (fun (_, k, _, (w, _, _)) -> along_reduce k w) streamed_cases_all

let id (c, _, _, _) = c.name

(* Each element of the streamed operand feeds [r] outputs, and a thread holds
   the largest share of them up to 16 that divides [r]: what is left of its
   reuse, outside the thread's lanes, is [r] divided by that share, each of
   them a pass over it. *)
let loads_once_a_tile =
  cases "the streamed operand's reuse outside a thread is its reuse over its tile"
    ~id streamed_cases_all (fun (_, _, o, (_, r, w')) ->
      let tile = largest_divisor r 16 in
      equal bigint
        (Bigint.div r (Bigint.of_int tile))
        (reuse o [ Global; Local ] w'))

(* The value of the index [i] with the ranges of [at] at their values and
   every other range and value it reads at 0. *)
let value_at i at =
  let zero = Ops.const (`Int Bigint.zero) in
  let others =
    List.filter_map
      (fun u ->
        if Ops.op u = Op.Index then Some (u, zero)
        else if Ops.op u = Op.Range && not (List.mem_assq u at) then
          Some (u, zero)
        else None)
      (Ops.Nodes.to_list (Ops.backward_slice ~calls:Skip i))
  in
  let at =
    List.map (fun (r, v) -> (r, Ops.const (`Int (Bigint.of_int v)))) at
  in
  Shape.sym_infer (Sym (Ops.substitute ~calls:Skip ~pass:Fixed_point i (at @ others))) []

(* The stride of the range [r] in the index [i]: how far its address moves
   when [r] moves from 0 to 1, if it moves. *)
let stride r i =
  if not (Ops.Nodes.mem r (Ops.backward_slice ~calls:Skip i)) then None
  else
    match value_at i [ (r, 1) ] - value_at i [] with
    | 0 -> None
    | s -> Some (Bigint.of_int s)

(* The bytes one load of the main access of [w] reads across the SIMD group
   of [o]'s first [simd_lanes] threads, the first local axis the fastest,
   each thread reading the vector of the upcast ranges of the index of lesser
   stride than its local ones, its other ranges and the values it reads at
   0. *)
let simd_lanes = 32

let group_bytes o w =
  match main_access o w with
  | None -> failf "the streamed operand has no access of a unit stride"
  | Some a ->
      let i = Ops.get_idx (Ops.nth a 1) in
      let locals = List.map (List.nth (K.rngs o)) (K.axes_of o [ Local ]) in
      let strides rs = List.filter_map (fun r -> stride r i) rs in
      let least =
        List.fold_left
          (fun m s -> if Bigint.compare s m < 0 then s else m)
          (Bigint.of_int max_int) (strides locals)
      in
      let vector =
        List.filter
          (fun r ->
            match stride r i with
            | Some s -> Bigint.compare s least < 0
            | None -> false)
          (List.map (List.nth (K.rngs o)) (K.axes_of o [ Upcast ]))
      in
      (* thread [t] of the group: its index along each local axis *)
      let thread t =
        fst
          (List.fold_left
             (fun (acc, t) r ->
               let n = Bigint.to_int (size r) in
               ((r, t mod n) :: acc, t / n))
             ([], t) locals)
      in
      let threads =
        min simd_lanes
          (List.fold_left (fun n r -> n * Bigint.to_int (size r)) 1 locals)
      in
      let rec combos = function
        | [] -> [ [] ]
        | r :: rest ->
            List.concat_map
              (fun c -> List.init (Bigint.to_int (size r)) (fun v -> (r, v) :: c))
              (combos rest)
      in
      let width = Dtype.itemsize (Ops.dtype a) in
      List.concat_map
        (fun t ->
          List.concat_map
            (fun v ->
              let first = width * value_at i (thread t @ v) in
              List.init width (fun b -> first + b))
            (combos vector))
        (List.init threads Fun.id)

(* A SIMD group's load of the streamed operand touches the fewest 32-byte
   sectors, memory's unit of transfer, that hold its bytes, one more where
   they start inside a sector. *)
let sector = 32

let coalesces =
  cases
    "a SIMD group's load of an operand streamed along the reduce reads whole \
     sectors"
    ~id streamed_cases
    (fun (_, _, o, (_, _, w')) ->
      let bytes = List.sort_uniq Int.compare (group_bytes o w') in
      let sectors =
        List.sort_uniq Int.compare (List.map (fun b -> b / sector) bytes)
      in
      less Testable.int
        ~than:(((List.length bytes + sector - 1) / sector) + 2)
        (List.length sectors))

(* A thread unrolls the loop of its streamed operand's contracted ranges of
   least stride, what the lanes leave of the reduce, by its largest divisor
   up to 8, or 4 when it holds rows that reuse each element: the unrolled
   lanes times the loop they leave, whose stride in the main access is theirs
   times their count, or the loop itself where nothing is unrolled. *)
let unrolls_in_flight =
  cases "the unroll is the largest divisor of its loop up to 8, 4 with reuse"
    ~id streamed_cases (fun (_, k, o, (_, r, w')) ->
      let cap = if Bigint.(equal r one) then 8 else 4 in
      let ids = List.map Ops.axis_id (K.rngs k) in
      let a =
        match main_access o w' with
        | Some a -> a
        | None -> failf "the streamed operand has no access of a unit stride"
      in
      let i = Ops.get_idx (Ops.nth a 1) in
      let unrolled =
        List.filter
          (fun u ->
            (not (List.mem (Ops.axis_id u) ids)) && Option.is_some (stride u i))
          (ranges_at o ~reduce:true [ Upcast ])
      in
      let contracted =
        match K.reduceop o with Some r -> List.tl (Ops.src r) | None -> []
      in
      let loops =
        List.filter_map
          (fun l ->
            if List.memq l contracted then
              Option.map (fun s -> (s, l)) (stride l i)
            else None)
          (ranges_at o ~reduce:true [ Weak; Loop ])
      in
      match unrolled with
      | [] -> (
          match List.sort (fun (s, _) (t, _) -> Bigint.compare s t) loops with
          | [] -> ()
          | (_, l) :: _ -> equal Testable.int 1 (largest_divisor (size l) cap))
      | u :: _ ->
          let n = size u in
          let s = Option.get (stride u i) in
          let left =
            List.fold_left
              (fun m (t, l) ->
                if Bigint.equal t (Bigint.mul n s) then size l else m)
              Bigint.one loops
          in
          equal Testable.int
            (largest_divisor (Bigint.mul n left) cap)
            (Bigint.to_int n))

(* A contraction of little reuse streams, the quantised matrices of a block
   range of unit stride among them, before a tensor core up to a reuse of 32
   when it sums 256 values or more and does at most 64 multiply-adds per byte
   of its streamed operand's buffers; past that reuse, with a shorter sum, or
   with blocks of rows that share their matrices past 64 a byte, the core
   takes it. *)
let applies =
  let expect = [
    ("q8_0_blocks_cuda", true); ("gguf_mxfp4_cuda", true);
    ("mxfp4_m12_cuda", true); ("vecmat_decoded_cuda", true);
    ("gpt_oss_gate_up_cuda", true); ("rows_m4_cuda", true);
    ("rows_m12_cuda", true); ("rows_m32_cuda", true);
    ("rows_m32_amd", true); ("shared_keys_cuda", true);
    ("routed_tiles_cuda", false); ("rows_short_cuda", false);
    ("routed_own_cuda", true); ("routed_shared_cuda", false);
    ("routed_at_bound_cuda", true); ("routed_past_bound_cuda", false);
  ] in
  cases "a contraction streams up to a reuse of 32, before a tensor core"
    ~id:fst expect (fun (name, streamed) ->
      let _, _, o = List.find (fun (c, _, _) -> c.name = name) gpu_cases in
      equal Testable.bool streamed (streams o))

(* Kernels rune compiles for CUDA, recorded without optimisations: products of
   one token by a Q8_0, an MXFP4 and a bfloat16 matrix, a prompt's blocks of 16
   rows each multiplying a bfloat16 matrix of its own, and gpt-oss-20b's gate
   and up and down products of a 512-token prompt, 158 blocks of 16 rows
   sharing 32 experts' MXFP4 matrices. The decode products and the blocks of
   their own do at most 8 multiply-adds per byte of the matrix and stream; the
   shared experts do 149 and take a tensor core. *)
let recorded_kernels =
  let graph path =
    let file = Filename.concat (Filename.dirname Sys.executable_name) path in
    Graph.of_string (In_channel.with_open_bin file In_channel.input_all)
  in
  let stream name = "../../../../bench/stream/" ^ name ^ ".graph" in
  [
    ("q8_0-1", stream "q8_0-1.0", true);
    ("mxfp4-1", stream "mxfp4-1.1", true);
    ("dense-bf16-1", stream "dense-bf16-1.0", true);
    ("routed-512-bf16", stream "routed-512-bf16.0", true);
    ("prefill_gate_up", "prefill_gate_up.graph", false);
    ("prefill_down", "prefill_down.graph", false);
  ]
  |> List.map (fun (name, path, streamed) -> (name, lazy (graph path), streamed))

(* The scheduler that the optimisations of [k] start from, after the lowering
   before them, as a search is handed it. *)
let handed k ren =
  let k =
    match Ops.arg k with
    | Ops.Kernel info ->
        Ops.replace k ~arg:(Kernel { info with opts_to_apply = None; beam = 1 })
    | _ -> k
  in
  let handed = ref None in
  let search _ s =
    handed := Some (K.copy s);
    s
  in
  ignore (Codegen.full_rewrite_to_sink ~beam:search k ren);
  Option.get !handed

let recorded_order =
  cases
    "a recorded contraction streams before a tensor core up to 64 \
     multiply-adds per byte of its streamed operand"
    ~id:(fun (name, _, _) -> name)
    recorded_kernels
    (fun (_, k, streamed) ->
      let o =
        Heuristic.hand_coded_optimizations
          (handed (Lazy.force k) (renderer "cuda"))
      in
      let cored =
        List.exists (function Opt.Tc _ -> true | _ -> false) (K.applied_opts o)
      in
      equal Testable.bool streamed (streams o);
      equal Testable.bool (not streamed) cored)

(* The host's budget *)

(* Vector register files, as a count of registers of a number of bytes: SSE's,
   AVX2's, NEON's, and AVX-512's at its full width. *)
let files = [ (16, 16); (16, 32); (32, 16); (32, 64) ]
let file_id (count, bytes) = Printf.sprintf "%dx%d" count bytes
let registers (count, bytes) = { Renderer.count; bytes }

let host file =
  let row =
    List.find
      (fun cell -> cell "renderer" = "cpu")
      (Golden.rows "renderers.golden")
  in
  Kernel_opts.renderer_of_row (fun column ->
      if column = "registers" then file_id file else row column)

(* The kernels of lorenz_simple's step that the tolk bench times, without the
   optimisations a search chose for them. *)
let lorenz =
  let graph name =
    let file =
      Filename.concat
        (Filename.dirname Sys.executable_name)
        ("../../../../bench/" ^ name ^ ".graph")
    in
    let k =
      Graph.of_string (In_channel.with_open_bin file In_channel.input_all)
    in
    match Ops.arg k with
    | Ops.Kernel info ->
        Ops.replace k ~arg:(Kernel { info with opts_to_apply = None })
    | _ -> k
  in
  List.map
    (fun name -> (name, lazy (graph name)))
    [
      "r_128_16_3_4_4_3_3_400_400"; "r_32_256_50_4_4_2_4_4"; "r_32_3_2_4_3_400";
    ]

let host_kernels =
  List.map
    (fun name -> (name, lazy (kernel name)))
    (List.sort_uniq String.compare
       (List.map (fun c -> c.kernel) recorded_cases))
  @ lorenz

let optimized k file =
  let s = K.v k (host file) in
  K.convert_loop_to_global s;
  Heuristic.hand_coded_optimizations s

(* The ranges a reduce closes that are axes of its kernel [s]. *)
let closing s u = List.filter (fun r -> List.memq r (K.rngs s)) (List.tl (Ops.src u))
let is_upcast r = Ops.axis_type r = Ops.Axis_type.Upcast

(* The output lanes of [s]: the product of its upcast axes no reduce closes. *)
let output_lanes s =
  List.fold_left
    (fun n i ->
      match List.nth (K.full_shape s) i with Ops.Int m -> n * m | _ -> n)
    1
    (K.axes_of ~reduce:false s [ Ops.Axis_type.Upcast ])

(* The shapes of the budget's clauses that a kernel has once optimised for
   [file]. *)
let shapes k file =
  let before = K.v k (host file) in
  let s = optimized k file in
  let reduces = K.reduceops s in
  let trips u =
    List.fold_left
      (fun n r ->
        match Ops.vmax r with `Int m -> n * (Bigint.to_int m + 1) | _ -> n)
      1 (closing before u)
  in
  List.filter_map
    (fun (label, holds) -> if holds then Some label else None)
    [
      ("no reduce", reduces = []);
      ("one reduce", List.length reduces = 1);
      ( "reduces sharing a loop",
        List.exists
          (fun u ->
            List.exists
              (fun v ->
                u != v
                && List.exists
                     (fun r -> List.memq r (closing s v))
                     (closing s u))
              reduces)
          reduces );
      ( "a short reduce beside a long one",
        List.exists (fun u -> trips u <= 32) (K.reduceops before)
        && List.exists (fun u -> trips u > 32) (K.reduceops before) );
      ( "a partial unroll",
        List.exists
          (fun u ->
            List.exists is_upcast (closing s u)
            && List.exists (fun r -> not (is_upcast r)) (closing s u))
          reduces );
      ( "a whole unroll",
        List.exists
          (fun u -> closing s u <> [] && List.for_all is_upcast (closing s u))
          reduces );
    ]

(* [k] with its 32-bit floats made 64-bit ones. *)
let retyped k =
  let text = Graph.to_string k in
  let f32 = "dtypes.f32" and f64 = "dtypes.f64" in
  let n = String.length f32 and b = Buffer.create (String.length text) in
  let rec go i =
    if i >= String.length text then ()
    else if i + n <= String.length text && String.sub text i n = f32 then begin
      Buffer.add_string b f64;
      go (i + n)
    end
    else begin
      Buffer.add_char b text.[i];
      go (i + 1)
    end
  in
  go 0;
  Graph.of_string (Buffer.contents b)

(* Registers held by kernels of known optimisations on AVX2's file, 16 of 32
   bytes, from the count's statement: a register holds 8 floats or 4 doubles,
   no more than a store writes at consecutive addresses, and a reduce that no
   longer loops holds nothing. *)
let held_cases =
  let up axis amount =
    Opt.Split { axis; amount; target = Opt.Upcast; top = false }
  in
  let avx2 = (16, 32) in
  cases "on AVX2 a kernel's accumulators hold"
    ~id:(fun (name, _, _, _) -> name)
    [
      (* 16 floats stored at consecutive addresses by 4 rows, the matmul's
         reduce unrolled by 4 and still looping: 64 lanes, 8 a register. *)
      ("8 registers for a 16 by 4 tile of floats", "matmul", false,
       ([ up 1 16; up 0 4; up 4 4 ], 8));
      (* 8 doubles by 4 rows: 32 lanes, 4 a register. *)
      ("8 registers for an 8 by 4 tile of doubles", "matmul", true,
       ([ up 1 8; up 0 4; up 4 4 ], 8));
      (* 4 rows, which the store writes 128 floats apart: one lane a
         register. *)
      ("4 registers for 4 lanes stored apart", "matmul", false, ([ up 0 4 ], 4));
      (* Both sums of 4 unrolled whole: nothing loops. *)
      ("nothing once no reduce loops", "tangent_products", false,
       ([ up 2 4; up 1 4; up 5 0 ], 0));
      ("nothing for a reduce unrolled whole", "max_rows", false,
       ([ up 0 4; up 2 0 ], 0));
    ]
    (fun (_, name, doubles, (opts, expected)) ->
      let k = kernel name in
      let k = if doubles then retyped k else k in
      let s = K.v k (host avx2) in
      K.convert_loop_to_global s;
      List.iter
        (fun o ->
          match K.apply_opt s o with
          | Ok _ -> ()
          | Error e -> failf "%a: %s" Opt.pp o e)
        opts;
      equal int expected (Heuristic.held_registers s (registers avx2)))

let budget_rows =
  List.concat_map
    (fun (name, k) -> List.map (fun file -> (name, k, file)) files)
    host_kernels

let budget =
  group "the register budget"
    [
      cases
        "the accumulators of the hottest loop take at most half the registers"
        ~id:(fun (name, _, file) -> name ^ " " ^ file_id file)
        budget_rows
        (fun (name, k, file) ->
          let s = optimized (Lazy.force k) file in
          at_most ~msg:name int ~than:(fst file / 2)
            (Heuristic.held_registers s (registers file)));
      test "the kernels the budget is checked on have each of its shapes"
        (fun () ->
          let found =
            List.sort_uniq String.compare
              (List.concat_map
                 (fun (_, k, file) -> shapes (Lazy.force k) file)
                 budget_rows)
          in
          equal (list string)
            [
              "a partial unroll";
              "a short reduce beside a long one";
              "a whole unroll";
              "no reduce";
              "one reduce";
              "reduces sharing a loop";
            ]
            found);
    ]

(* A file of no fewer registers, each no narrower, holds no fewer lanes. *)
let more_registers =
  let wider (c0, b0) (c1, b1) = c0 <= c1 && b0 <= b1 && (c0, b0) <> (c1, b1) in
  let rows =
    List.concat_map
      (fun (name, k) ->
        List.concat_map
          (fun f ->
            List.filter_map
              (fun g -> if wider f g then Some (name, k, f, g) else None)
              files)
          files)
      host_kernels
  in
  cases "on the host a larger register file never holds fewer output lanes"
    ~id:(fun (name, _, f, g) ->
      Printf.sprintf "%s %s %s" name (file_id f) (file_id g))
    rows
    (fun (_, k, f, g) ->
      let k = Lazy.force k in
      at_most int
        ~than:(output_lanes (optimized k g))
        (output_lanes (optimized k f)))


(* The axis and amount of [s]'s first optimisation, when it is an upcast. *)
let first_upcast s =
  match K.applied_opts s with
  | Opt.Split { axis; amount; target = Opt.Upcast; _ } :: _ ->
      Some (axis, amount)
  | _ -> None

(* Whether the axis [axis] of [k] holds a multiple of 32 elements, so that an
   upcast of floats by 16 leaves an axis of doubles to fill. *)
let divisible_by_32 k axis =
  match List.nth (K.full_shape k) axis with
  | Ops.Int n -> n mod 32 = 0
  | _ -> false

let doubles =
  let rows =
    List.concat_map
      (fun (name, k) ->
        List.filter_map
          (fun file ->
            let k = Lazy.force k in
            let s = optimized k file in
            let floats =
              K.reduceops s <> []
              && List.for_all
                   (fun u -> Dtype.equal (Ops.dtype u) Float32)
                   (K.reduceops s)
            in
            let input = K.v k (host file) in
            K.convert_loop_to_global input;
            match first_upcast s with
            | Some (axis, 16)
              when floats && divisible_by_32 input axis
                   && Heuristic.held_registers s (registers file)
                      = fst file / 2 ->
                Some (name, k, file, s)
            | _ -> None)
          files)
      host_kernels
  in
  cases
    "on the host a reduce of doubles over a stored axis of a multiple of 32 \
     takes half the lanes of floats, in the registers floats fill"
    ~id:(fun (name, _, file, _) -> name ^ " " ^ file_id file)
    rows
    (fun (_, k, file, s) ->
      let d = optimized (retyped k) file in
      equal
        (option (pair int int))
        (Option.map (fun (axis, n) -> (axis, n / 2)) (first_upcast s))
        (first_upcast d);
      equal int
        (Heuristic.held_registers s (registers file))
        (Heuristic.held_registers d (registers file)))

let () =
  exit
    (Windtrap.run "Tolk.Heuristic"
       [
         chosen;
         leaves_its_argument;
         keeps_kernel_writes;
         group "streaming"
           [
             loads_once_a_tile;
             coalesces;
             unrolls_in_flight;
             applies;
             recorded_order;
           ];
         group "the host's budget"
           [ held_cases; budget; more_registers; doubles ];
       ])
