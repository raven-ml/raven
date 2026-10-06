(* Tests of Tolk.Jit: the schedules tinygrad captures from the functions its
   tests jit, lowered as tinygrad lowers them, and the laws of a lowering over
   those captures and over drawn ones: its replays leave what the captured
   schedule leaves when it runs unplanned, whatever the inputs and variables;
   its memory plan never places a held buffer or an input in an arena; and it is
   deterministic. *)

open Windtrap
open Tolk
module Engine = Tolk_engine

let uop = Uops.uop
let plain f = Setting.context [ B (Setting.no_color, true) ] f

let param_arg u =
  match Ops.arg u with
  | Param p -> p
  | _ -> invalid_arg "storage has a parameter argument"

let device_name u =
  match (param_arg u).device with
  | Some (Single d) -> d
  | _ -> invalid_arg "storage on one device"

let storage u =
  List.filter (fun n -> Ops.op n = Buffer) (Ops.toposort ~calls:Skip u)

(* The parameters of a lowered schedule that each run binds: those its calls
   take, and the inputs of its batches. *)
let parameters u =
  let nodes = Ops.toposort ~calls:Skip u in
  let batch_inputs n =
    match Ops.arg n with
    | Call { aux = Some info; _ } -> List.map (fun (p, _, _) -> p) info.inputs
    | _ -> []
  in
  List.sort_uniq Ops.compare
    (List.filter
       (fun n ->
         Ops.op n = Param && Ops.tag n = None && (param_arg n).slot >= 0)
       (nodes @ List.concat_map batch_inputs nodes))

(* The parameter the [i]th input [u] becomes: of its type, device and size. *)
let parameter i u =
  Call.param
    ~shape:[ Int (Shape.max_numel u) ]
    ?device:(Ops.device u) i (Ops.dtype u)

(* The unbound variables the calls of a schedule read, with their bounds. *)
let variables linear =
  List.sort_uniq compare
    (List.filter_map
       (fun n ->
         match Ops.arg n with
         | Param
             {
               name = Some v;
               addrspace = Some Alu;
               bound = None;
               vmin_vmax = Some (lo, hi);
               _;
             } ->
             Some (v, (Dtype.Value.to_int lo, Dtype.Value.to_int hi))
         | _ -> None)
       (Ops.toposort ~calls:Enter linear))

(* The laws, for a captured schedule [captured] holding [held] and reading
   [inputs], and its lowering [lowered]. *)

(* The storage [lowered] reaches is the storage [captured] reaches that it holds
   and that is no input, and arenas, which [captured] does not reach. *)
let plans_around ~held ~inputs captured lowered =
  let reached = storage captured in
  let kept =
    List.filter (fun b -> List.memq b held && not (List.memq b inputs)) reached
  in
  let arenas, rest =
    List.partition (fun b -> not (List.memq b reached)) (storage lowered)
  in
  equal (slist uop Ops.compare) kept rest;
  arenas

(* The [i]th input is the parameter of slot [i] if [captured] reaches it, and no
   other parameter is bound. *)
let binds_inputs ~inputs captured lowered =
  let reached = storage captured in
  equal (slist uop Ops.compare)
    (List.concat
       (List.mapi
          (fun i u -> if List.memq u reached then [ parameter i u ] else [])
          inputs))
    (parameters lowered)

(* Lowering twice gives the same schedule, up to the numbers of the storage and
   placeholders it makes. *)
let lowers_the_same lower =
  let first = lower () in
  equal uop first
    (Uops.numbered_like first (Uops.placeholders_like first (lower ())))

(* Replays

   A captured schedule runs unplanned, compiled as it is and linked with its
   held buffers and inputs bound; its lowering is linked with its held buffers
   bound and runs with its inputs in their slots. A replay is runs in turn, each
   [(seed, vars)], as the calls of a jitted function are: the first starts from
   the contents its seed draws for each held buffer and input, and each later
   one from what the runs before it left, with new inputs drawn from its seed.
   It is observed by what the held buffers and inputs hold after the last
   run. *)

type replay = (int * (string * int) list) list -> Dtype.value array list

let element dtype k : Dtype.value =
  if Dtype.is_float dtype then `Float (float_of_int k)
  else if Dtype.equal dtype Bool then `Bool (k > 0)
  else if Dtype.is_unsigned dtype then `Int (Bigint.of_int (k + 3))
  else `Int (Bigint.of_int k)

let start seed u =
  Array.init (Shape.max_numel u) (fun j ->
      element (Ops.dtype u)
        ((((j * 7) + ((param_arg u).slot * 3) + seed) mod 11) - 3))

(* The held buffers and inputs of [captured], each with a buffer of its
   device. *)
let observed ~devices ~held ~inputs captured =
  let reached = storage captured in
  List.map
    (fun u ->
      ( u,
        Run.buffer (devices (device_name u)).Engine.device (Ops.dtype u)
          (start 0 u) ))
    (List.sort_uniq Ops.compare
       (inputs @ List.filter (fun b -> List.memq b reached) held))

let replay ~inputs bufs run : replay =
 fun runs ->
  List.iteri
    (fun k (seed, vars) ->
      List.iter
        (fun (u, b) ->
          if k = 0 || List.memq u inputs then
            Nx_device.Buffer.copy
              ~src:(Run.buffer Nx_device.host (Ops.dtype u) (start seed u))
              ~dst:b)
        bufs;
      run vars)
    runs;
  List.map (fun (u, b) -> Run.values (Ops.dtype u) b) bufs

let unplanned ~devices ~held ~inputs captured =
  let bufs = observed ~devices ~held ~inputs captured in
  let s =
    Engine.link ~devices
      ~bound:(List.map (fun (u, b) -> (u, [ b ])) bufs)
      (plain (fun () ->
           Hcq2.compile_linear ~profile:Unstamped
             ~devices:(fun n -> (devices n).compiler)
             captured))
  in
  replay ~inputs bufs (fun vars -> Engine.run ~vars s [||])

let replayed ~devices ~held ~inputs captured lowered =
  let bufs = observed ~devices ~held ~inputs captured in
  let reached = storage lowered in
  let bound = List.filter (fun (u, _) -> List.memq u reached) bufs in
  let slots = Array.of_list (List.map (fun u -> [ List.assq u bufs ]) inputs) in
  let s =
    Engine.link ~devices
      ~bound:(List.map (fun (u, b) -> (u, [ b ])) bound)
      lowered
  in
  replay ~inputs bufs (fun vars -> Engine.run ~vars s slots)

let contents = list (array Dtypes.value)

(* Recorded captures

   A case's goldens are its captured schedule, the sink of the buffers the
   capture holds, the sink of its inputs in the order of their slots, and its
   lowering, for Clang on x86_64 on the devices of the Hcq2 suite: the host,
   PYTHON, a host device that holds Python's data, and CPU:1 to CPU:3 with the
   NULL device's queues. The lowering's binaries are its sources' bytes. *)

let captured case = Golden.sink (case ^ ".golden")
let held case = Ops.src (Golden.sink (case ^ "_held.golden"))
let inputs case = Ops.src (Golden.sink (case ^ "_inputs.golden"))

let recorded_target =
  {
    Helpers.Target.device = "CPU";
    renderer = "";
    arch = "x86_64,x86-64";
    interface = "";
    indices = "";
  }

let recorded_devices () =
  let events = Null_queue.events () in
  function
  | "CPU:1" | "CPU:2" | "CPU:3" ->
      let queues =
        {
          Hcq2.commands = Null_queue.commands events;
          copy_queue = true;
          submission = Buffered;
          host = "CPU";
          reaches = (fun _ -> true);
        }
      in
      { Hcq2.target = recorded_target; queues = Some queues }
  | _ -> { Hcq2.target = recorded_target; queues = None }

let lower ?search ?(devices = recorded_devices ()) case =
  plain (fun () ->
      Jit.jit_lower ?search ~profile:Unstamped ~devices ~held_bufs:(held case)
        ~inputs:(inputs case) (captured case))

(* The cases of tinygrad's jit tests, by the file they come from. *)

let test_jit =
  [
    "input_view";
    "chain_of_three";
    "add";
    "assign";
    "assign_int8";
    "copyin";
    "clone";
    "transfers";
    "several_devs";
    "view_bitcast";
    "multiple_outputs";
    "weight";
    "assign_input";
    "lazy_grad";
    "copy_inside";
    "weights_copy";
    "weights_independent_copy";
    "weights_kernel";
    "held_constant";
    "accumulator";
    "split_simple";
    "split_cpu";
    "split_cpu_several";
    "split_multidev";
    "split_multidev_xfer";
    "split_multidev_copy";
  ]

let test_jit_cases =
  [ "explicit"; "implicit_input"; "implicit_output"; "implicit_io" ]

let test_jit_footguns =
  [
    "sum";
    "two_kernels";
    "cat_window";
    "shift_window";
    "slice_assign";
    "masked_select";
    "nonzero";
  ]

let test_symbolic_jit =
  [
    "plus1";
    "inner_bound_var_view";
    "plus1_pad_view";
    "plus1_pad";
    "symbolic_add";
    "symbolic_matmul";
    "mixed_with_no_symbol_kernel";
    "symbolic_attention";
    "cat_dim0";
    "cat_dim1";
    "cat_dim0_two_vars";
    "cat_dim1_two_vars";
    "two_vars_plus1_ij";
    "two_vars_plus1_ji";
    "symbolic_shrink";
    "symbolic_slice";
    "slice_var_shape";
    "ones_sum";
    "mean";
    "mean0";
    "mean1";
    "mean_2d";
    "mean_2d0";
    "mean_2d1";
    "var";
    "var0";
    "var1";
    "var_2d";
    "var_2d0";
    "var_2d1";
  ]

let cases = test_jit @ test_jit_cases @ test_jit_footguns @ test_symbolic_jit

(* One case of each kind runs by default: kernels on the host, an input that a
   kernel writes, held buffers without inputs, copies to a device with queues, a
   held constant of Python's data, and a variable. Compiling the others takes
   seconds. *)
let by_default =
  [ "add"; "assign"; "implicit_io"; "transfers"; "held_constant"; "plus1" ]

let per_case name tests =
  group name
    (List.map
       (fun case ->
         group
           ~tags:(if List.mem case by_default then [] else [ "slow" ])
           case (tests case))
       cases)

(* A capture binds the variables its calls read. *)
let captures =
  Golden.cases "captures.golden" (fun cell ->
      let names =
        if cell "vars" = "-" then []
        else
          List.map
            (fun w -> List.hd (String.split_on_char '=' w))
            (String.split_on_char ' ' (cell "vars"))
      in
      equal (list string) names
        (List.map fst (variables (captured (cell "case")))))

(* Profile keys are left out of the comparison. *)
let lowers_as_recorded case =
  let file = case ^ "_lowered.golden" in
  test file (fun () ->
      let golden = Golden.sink file in
      let recorded u = Graph.to_string (Uops.without_profile_keys u) in
      equal text (recorded golden)
        (lower case |> Uops.binaries_as_sources
        |> Uops.placeholders_like golden
        |> Uops.numbered_like golden |> recorded))

let recorded =
  per_case "jit_lower › recorded" (fun case ->
      [
        lowers_as_recorded case;
        test "makes each input the parameter of its slot" (fun () ->
            binds_inputs ~inputs:(inputs case) (captured case) (lower case));
        test "places no held buffer or input in an arena" (fun () ->
            ignore
              (plans_around ~held:(held case) ~inputs:(inputs case)
                 (captured case) (lower case)));
        test "lowers the same twice" (fun () ->
            lowers_the_same (fun () -> lower case));
      ])

(* Replays of recorded captures

   A case replays on the host and the NULL devices, PYTHON being a host device,
   with each variable over the values tinygrad's tests replay it with, the
   capture's binding first. *)

let null = lazy (Null_device.devices ())

let on_null = function
  | "PYTHON" -> Engine.device [ ("PYTHON", Nx_device.host) ] "PYTHON"
  | name -> (Lazy.force null) name

(* The values tinygrad's tests replay each variable with. *)
let replayed_values = [ ("i", (1, 4)); ("j", (2, 4)); ("pos", (0, 3)) ]

let bindings case =
  List.fold_right
    (fun (v, _) acc ->
      let lo, hi = List.assoc v replayed_values in
      Gen.map
        (fun (x, rest) -> (v, x) :: rest)
        (Gen.pair (Gen.int_range lo hi) acc))
    (variables (captured case))
    (Gen.constant [])

let capture_binding case =
  let row =
    List.find (fun cell -> cell "case" = case) (Golden.rows "captures.golden")
  in
  if row "vars" = "-" then []
  else
    List.map
      (fun w ->
        match String.split_on_char '=' w with
        | [ v; x ] -> (v, int_of_string x)
        | _ -> invalid_arg ("no binding " ^ w))
      (String.split_on_char ' ' (row "vars"))

let pp_run ppf (seed, vars) =
  Format.fprintf ppf "seed %d%a" seed
    (Format.pp_print_list (fun ppf (v, x) -> Format.fprintf ppf " %s=%d" v x))
    vars

let pp_runs =
  Format.pp_print_list ~pp_sep:(fun ppf () -> Format.fprintf ppf "; ") pp_run

let replays_what_its_capture_leaves case =
  let replays =
    lazy
      (let held = held case and inputs = inputs case and c = captured case in
       let lowered = lower ~devices:(fun n -> (on_null n).compiler) case in
       ( unplanned ~devices:on_null ~held ~inputs c,
         replayed ~devices:on_null ~held ~inputs c lowered ))
  in
  prop ~count:10
    ~examples:[ [ (0, capture_binding case) ] ]
    "replays what its capture leaves, run unplanned"
    (Gen.with_pp pp_runs
       (Gen.list ~size:(Gen.int_range 1 3) (Gen.pair Gen.nat (bindings case))))
    (fun runs ->
      let unplanned, replayed = Lazy.force replays in
      equal contents (unplanned runs) (replayed runs))

let replays =
  per_case "replay › recorded" (fun case ->
      [ replays_what_its_capture_leaves case ])

(* Drawn captures

   A drawn capture is a schedule of calls over buffers of four or seventy
   floats, each on the host or on CPU:1, with compiled kernels, on devices that
   run their calls one by one. Its first buffers are inputs and the next
   constants, which the capture holds; no call writes either. Each call writes a
   buffer it does not read, new or written by an earlier call: a kernel on one
   device adds one to a buffer, or adds two, or a copy moves a buffer to the
   other device. It holds the last buffer written, and those its draw keeps. *)

type call = Inc of int * int | Add of int * int * int | Copy of int * int

(* A buffer's device and number of floats. *)
type kind = string * int

let other = function "CPU" -> "CPU:1" | _ -> "CPU"

type drawn = {
  kinds : kind array; (* Of each buffer, by number. *)
  n_inputs : int;
  n_constants : int;
  calls : call list; (* Each writing its first buffer. *)
  kept : int list; (* The held buffers that calls write. *)
}

let pp_drawn ppf d =
  let b ppf i = Format.fprintf ppf "b%d" i in
  let buffers = Format.pp_print_list ~pp_sep:Format.pp_print_space b in
  let call ppf = function
    | Inc (d, s) -> Format.fprintf ppf "%a := %a + 1" b d b s
    | Add (d, s0, s1) -> Format.fprintf ppf "%a := %a + %a" b d b s0 b s1
    | Copy (d, s) -> Format.fprintf ppf "%a := copy %a" b d b s
  in
  Format.fprintf ppf "@[<v>";
  Array.iteri
    (fun i (device, n) ->
      Format.fprintf ppf "%a: %d floats on %s@," b i n device)
    d.kinds;
  Format.fprintf ppf "inputs %a, constants %a, kept %a@," buffers
    (List.init d.n_inputs Fun.id)
    buffers
    (List.init d.n_constants (fun i -> d.n_inputs + i))
    buffers d.kept;
  Format.pp_print_list call ppf d.calls;
  Format.fprintf ppf "@]"

(* [interpret inputs constants steps keeps] is the capture whose inputs and
   constants are of [inputs] and [constants] kinds, and whose calls follow
   [steps]: each [(op, a, b, fresh)] reads the [a]th readable buffer, and for an
   addition the [b]th other of its kind, and writes a new buffer if [fresh] or
   if no written buffer fits, else the [b]th that fits. Each [keeps] keeps the
   written buffer of its position. *)
let interpret inputs constants steps keeps =
  let kinds = ref (Array.of_list (inputs @ constants)) in
  let readable = ref (List.init (Array.length !kinds) Fun.id) in
  let written = ref [] in
  let pick l k = List.nth l (k mod List.length l) in
  let kind i = !kinds.(i) in
  let destination k reads fresh b =
    let fits =
      List.filter (fun w -> kind w = k && not (List.mem w reads)) !written
    in
    if fresh || fits = [] then (
      kinds := Array.append !kinds [| k |];
      Array.length !kinds - 1)
    else pick fits b
  in
  let step (op, a, b, fresh) =
    let src = pick !readable a in
    let ((device, n) as k) = kind src in
    let others = List.filter (fun r -> r <> src && kind r = k) !readable in
    let call =
      match (op, others) with
      | 2, _ -> Copy (destination (other device, n) [ src ] fresh b, src)
      | 1, _ :: _ ->
          let s1 = pick others b in
          Add (destination k [ src; s1 ] fresh b, src, s1)
      | _ -> Inc (destination k [ src ] fresh b, src)
    in
    let (Inc (dst, _) | Add (dst, _, _) | Copy (dst, _)) = call in
    if not (List.mem dst !written) then written := !written @ [ dst ];
    if not (List.mem dst !readable) then readable := !readable @ [ dst ];
    call
  in
  let calls = List.map step steps in
  let kept =
    List.filteri
      (fun i _ -> Option.value (List.nth_opt keeps i) ~default:false)
      !written
  in
  {
    kinds = !kinds;
    n_inputs = List.length inputs;
    n_constants = List.length constants;
    calls;
    kept =
      List.sort_uniq compare
        (List.nth !written (List.length !written - 1) :: kept);
  }

let drawn =
  let open Gen in
  let kind = pair (of_list [ "CPU"; "CPU:1" ]) (of_list [ 4; 70 ]) in
  let step =
    let+ op = int_range 0 2 and+ a = nat and+ b = nat and+ fresh = bool in
    (op, a, b, fresh)
  in
  with_pp pp_drawn
    (let+ inputs = list ~size:(int_range 1 3) kind
     and+ constants = list ~size:(int_range 0 2) kind
     and+ steps = list ~size:(int_range 1 8) step
     and+ keeps = list ~size:(int_range 0 8) bool in
     interpret inputs constants steps keeps)

(* The kernels, compiled for the host, of each size. *)

let clang = lazy (Cstyle.clang (Engine.target Nx_device.host))

let compiled name f =
  let kernel n =
    let i = Ops.range (Int n) [ 0 ] in
    let at slot = Ops.index (Call.placeholder ~slot [ n ] Float32) [ i ] in
    Codegen.to_program
      (Ops.sink
         ~kernel:(Ops.kernel_info ~name:(Printf.sprintf "%s_%d" name n) ())
         [ Ops.end_ (Ops.store (at 0) (f at)) [ i ] ])
      (Lazy.force clang)
  in
  lazy (List.map (fun n -> (n, kernel n)) [ 4; 70 ])

let inc =
  compiled "inc" (fun at -> Ops.add (at 1) (Ops.float ~dtype:Float32 1.))

let add = compiled "add" (fun at -> Ops.add (at 1) (at 2))
let unqueued = Engine.device (Run.devices ())

(* The captured schedule of [d], with the buffers it holds and its inputs. *)
let capture_of d =
  let nodes =
    Array.map
      (fun (device, n) -> Ops.new_buffer (Single device) n Float32)
      d.kinds
  in
  let program kernels dst =
    List.assoc (snd d.kinds.(dst)) (Lazy.force kernels)
  in
  let call = function
    | Inc (dst, s) -> Ops.call (program inc dst) [ nodes.(dst); nodes.(s) ]
    | Add (dst, s0, s1) ->
        Ops.call (program add dst) [ nodes.(dst); nodes.(s0); nodes.(s1) ]
    | Copy (dst, s) -> Call.store_call nodes.(dst) nodes.(s)
  in
  let held =
    List.map (Array.get nodes)
      (List.init d.n_constants (fun i -> d.n_inputs + i) @ d.kept)
  in
  ( Ops.v Op.Linear ~src:(List.map call d.calls),
    held,
    List.init d.n_inputs (Array.get nodes) )

let lower_drawn (linear, held, inputs) =
  Jit.jit_lower ~profile:Unstamped
    ~devices:(fun n -> (unqueued n).compiler)
    ~held_bufs:held ~inputs linear

let replays_drawn d =
  let ((linear, held, inputs) as c) = capture_of d in
  let lowered = lower_drawn c in
  let arenas = plans_around ~held ~inputs linear lowered in
  let planned =
    List.filter
      (fun b -> not (List.memq b held || List.memq b inputs))
      (storage linear)
  in
  cover "an arena holds several buffers"
    (List.length arenas < List.length planned);
  cover "a copy" (List.exists (function Copy _ -> true | _ -> false) d.calls);
  let runs = [ (0, []); (5, []) ] in
  equal contents
    (unplanned ~devices:unqueued ~held ~inputs linear runs)
    (replayed ~devices:unqueued ~held ~inputs linear lowered runs)

let drawn_laws =
  group "jit_lower › drawn"
    [
      prop "replays what its capture leaves, run unplanned" drawn replays_drawn;
      prop "places no held buffer or input in an arena" drawn (fun d ->
          let ((linear, held, inputs) as c) = capture_of d in
          ignore (plans_around ~held ~inputs linear (lower_drawn c)));
      prop "makes each input the parameter of its slot" drawn (fun d ->
          let ((linear, _, inputs) as c) = capture_of d in
          binds_inputs ~inputs linear (lower_drawn c));
      prop "lowers the same twice" drawn (fun d ->
          let c = capture_of d in
          lowers_the_same (fun () -> lower_drawn c));
    ]

let () = exit (run "Jit" [ captures; recorded; replays; drawn_laws ])
