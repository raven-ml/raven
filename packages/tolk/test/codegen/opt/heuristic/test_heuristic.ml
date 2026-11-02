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
   core used in full makes. The cases of the default process are enough. *)
let evaluable c =
  c.environment = ""
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
      if c.renderer = "cpu" || c.settings <> [] || c.environment <> "" then
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
   when it sums 256 values or more; past that reuse, or with a shorter sum,
   the core takes it. *)
let applies =
  let expect = [
    ("q8_0_blocks_cuda", true); ("gguf_mxfp4_cuda", true);
    ("mxfp4_m12_cuda", true); ("vecmat_decoded_cuda", true);
    ("gpt_oss_gate_up_cuda", true); ("rows_m4_cuda", true);
    ("rows_m12_cuda", true); ("rows_m32_cuda", true);
    ("rows_m32_amd", true); ("shared_keys_cuda", true);
    ("routed_tiles_cuda", false); ("rows_short_cuda", false);
  ] in
  cases "a contraction streams up to a reuse of 32, before a tensor core"
    ~id:fst expect (fun (name, streamed) ->
      let _, _, o = List.find (fun (c, _, _) -> c.name = name) gpu_cases in
      equal Testable.bool streamed (streams o))

let () =
  exit
    (Windtrap.run "Tolk.Heuristic"
       [
         chosen;
         leaves_its_argument;
         keeps_kernel_writes;
         group "streaming"
           [ loads_once_a_tile; coalesces; unrolls_in_flight; applies ];
       ])
