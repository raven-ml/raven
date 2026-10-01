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
  settings : Helpers.binding list;
  environment : string;
      (** The settings of the matrix-vector layout, [MV] and [MV_*], which are
          read once from the environment: a case under them runs in a process of
          its own (see dune), tagged by {!process}. *)
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

let processes =
  [
    ("", "");
    ("MV=0", "mv_0");
    ("MV_BLOCKSIZE=1 MV_ROWS_PER_THREAD=1", "mv_block_rows_1");
    ("MV_BLOCKSIZE=1 MV_THREADS_PER_ROW=1 MV_ROWS_PER_THREAD=1", "mv_sizes_1");
  ]

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
      group ~tags:[ tag ] environment
        (test ("the suite runs this group with " ^ environment) has_environment
        :: tests in_environment)

(* The goldens were recorded without colours. *)
let optimize c =
  Helpers.context
    (Helpers.B (Helpers.no_color, true) :: c.settings)
    (fun () ->
      Postrange.apply_opts ~hand_coded:Heuristic.hand_coded_optimizations
        (kernel c.kernel) (renderer c.renderer))

let applied ast =
  match Ops.arg ast with
  | Kernel k -> k.applied_opts
  | _ -> failf "%a is not a kernel" Ops.pp ast

let size r =
  match Ops.vmax r with
  | `Int n -> Bigint.succ n
  | v -> failf "a range has the bound %a" (Testable.pp Dtypes.value) v

(* Choosing *)

let chosen =
  let tests cases =
    [
      Windtrap.cases
        ~name:(fun c -> c.name)
        "applied_opts" cases
        (fun c -> equal (list Kernel_opts.opt) c.opts (applied (optimize c)));
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
  cases ~name:Fun.id
    "hand_coded_optimizations leaves the scheduler it is given as it was"
    kernels (fun name ->
      let k = K.v (kernel name) (renderer "metal") in
      K.convert_loop_to_global k;
      let ast = K.ast k in
      let optimized = Heuristic.hand_coded_optimizations k in
      equal Uops.uop ast (K.ast k);
      equal (list Kernel_opts.opt) [] (K.applied_opts k);
      equal (list Kernel_opts.opt)
        (List.filter_map
           (fun c ->
             if
               c.kernel = name && c.renderer = "metal" && c.settings = []
               && c.environment = ""
             then Some c.opts
             else None)
           recorded_cases
        |> List.hd)
        (K.applied_opts optimized))

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
      cases ~name:(fun c -> c.name) "small kernels" (within 0 (1 lsl 12)) keeps;
      cases ~tags:[ "slow" ]
        ~name:(fun c -> c.name)
        "large kernels"
        (within (1 lsl 12) (1 lsl 18))
        keeps;
    ]

let () =
  exit
    (Windtrap.run "Tolk.Heuristic"
       [ chosen; leaves_its_argument; keeps_kernel_writes ])
