open Windtrap
open Tolk

(* Goldens hold arbitrary text as OCaml string literals. *)
let literal cell = Scanf.sscanf cell "%S%!" Fun.id

(* [claim] checked on each row of a golden. *)
let as_tinygrad ?key claim file check =
  group claim [ Golden.cases ?key file check ]

(* The golden's text for an exception tinygrad raised. *)
let raised cell =
  List.mem cell [ "ValueError"; "ZeroDivisionError"; "RuntimeError" ]

let refuses f = raises_match (fun e -> Exn.invalid_arg e) f
let context = Setting.context
let shaping = Setting.shaping

(* Fresh variable names: a setting is declared once per process. *)
let fresh =
  let n = ref 0 in
  fun () ->
    incr n;
    Printf.sprintf "TOLK_TEST_VARIABLE_%d" !n

(* The initial value of a setting made by [make] from a fresh variable holding
   [value]. *)
let declared_with value make =
  let key = fresh () in
  setenv key value;
  Setting.value (make key)

let int_setting key = Setting.int ~reach:Process key 0
let float_setting key = Setting.float ~reach:Process key 0.

(* The environment *)

let parses_like_tinygrad make w of_cell column cell =
  let value = Some (literal (cell "input")) in
  let expected = cell column in
  if raised expected then refuses (fun () -> declared_with value make)
  else equal w (of_cell expected) (declared_with value make)

let environment =
  group "Environment"
    [
      as_tinygrad "an int setting reads an integer as tinygrad does"
        "env.golden"
        (parses_like_tinygrad int_setting int int_of_string "int");
      as_tinygrad "a float setting reads a number as tinygrad does" "env.golden"
        (parses_like_tinygrad float_setting float_exact float_of_string "float");
      (* Python accepts these, but tolk does not port CPython's Unicode leniency
         in int() and float() (README, Exclusions). *)
      cases ~name:(Printf.sprintf "%S")
        "int and float settings refuse non-ASCII digits and white space"
        [
          "\u{0665}";
          "1\u{0665}";
          "\u{00A0}5\u{00A0}";
          "\u{0085}5";
          "5\u{3000}";
          "\u{2028}1.5";
        ] (fun value ->
          refuses (fun () -> declared_with (Some value) int_setting);
          refuses (fun () -> declared_with (Some value) float_setting));
      test "an int setting refuses an integer beyond int's range" (fun () ->
          refuses (fun () ->
              declared_with (Some "9223372036854775807") int_setting));
      test "an int setting starts from its default when its variable is unset"
        (fun () ->
          equal int 5
            (declared_with None (fun key -> Setting.int ~reach:Process key 5)));
      test "an int setting starts from its variable when set" (fun () ->
          equal int 12
            (declared_with (Some " 12 ") (fun key ->
                 Setting.int ~reach:Process key 5)));
      test "an int setting refuses a variable that is no integer" (fun () ->
          refuses (fun () -> declared_with (Some "abc") int_setting));
      test "a float setting starts from its default when its variable is unset"
        (fun () ->
          equal float_exact 0.5
            (declared_with None (fun key ->
                 Setting.float ~reach:Process key 0.5)));
      cases
        ~name:(fun (value, _) -> Printf.sprintf "%S" value)
        "a switch is on iff its variable holds a nonzero integer"
        [
          ("0", false);
          ("1", true);
          ("2", true);
          ("-1", true);
          (" 0 ", false);
          ("00", false);
        ]
        (fun (value, on) ->
          equal bool on
            (declared_with (Some value) (fun key ->
                 Setting.bool ~reach:Process key (not on))));
      test "a switch starts from its default when its variable is unset"
        (fun () ->
          let switch default key = Setting.bool ~reach:Process key default in
          equal bool true (declared_with None (switch true));
          equal bool false (declared_with None (switch false)));
      test "a switch refuses a variable that is no integer" (fun () ->
          refuses (fun () ->
              declared_with (Some "true") (fun key ->
                  Setting.bool ~reach:Process key false)));
      test "a string setting starts from its variable as written" (fun () ->
          let word key = Setting.string ~reach:Process key "d" in
          equal string " x " (declared_with (Some " x ") word);
          equal string "" (declared_with (Some "") word);
          equal string "d" (declared_with None word));
      test "an optional setting holds its variable's integer, if set" (fun () ->
          let opt key = Setting.int_option ~reach:Process key in
          equal (option int) (Some (-3)) (declared_with (Some " -3 ") opt);
          equal (option int) None (declared_with None opt);
          refuses (fun () -> declared_with (Some "") opt));
    ]

(* Declarations *)

let declaration =
  group "declarations"
    [
      test "a setting reads its variable once, when declared" (fun () ->
          let key = fresh () in
          setenv key (Some "1");
          let v = int_setting key in
          setenv key (Some "2");
          equal int 1 (Setting.value v));
      test "a setting whose variable is refused is not declared" (fun () ->
          let key = fresh () in
          setenv key (Some "abc");
          refuses (fun () -> int_setting key);
          setenv key (Some "4");
          equal int 4 (Setting.value (int_setting key)));
      test "key is the name of the setting's variable" (fun () ->
          let key = fresh () in
          equal string key (Setting.key (Setting.bool ~reach:Process key false)));
      test "a name is declared once, whatever the setting's type and reach"
        (fun () ->
          let key = fresh () in
          ignore (Setting.int ~reach:Output key 0);
          refuses (fun () -> Setting.int ~reach:Output key 0);
          refuses (fun () -> Setting.bool ~reach:Process key false);
          refuses (fun () -> Setting.float ~reach:Output key 0.);
          refuses (fun () -> Setting.string ~reach:Process key "");
          refuses (fun () -> Setting.int_option ~reach:Output key));
      test "a library setting's name cannot be declared again" (fun () ->
          refuses (fun () -> Setting.int ~reach:Process "DEBUG" 0));
      test "a setting declared inside a context stays declared after it"
        (fun () ->
          let key = fresh () in
          context [ B (Setting.debug, 1) ] (fun () -> ignore (int_setting key));
          refuses (fun () -> int_setting key));
    ]

(* Contexts *)

let level = Setting.int ~reach:Output "TOLK_TEST_LEVEL" 0
let label = Setting.string ~reach:Output "TOLK_TEST_LABEL" "default"
let value = Setting.value

type scope = { bound : int; raises : bool; inner : scope list }

let rec pp_scope ppf s =
  Format.fprintf ppf "@[<hv 1>{ bound = %d;@ raises = %b;@ inner = [%a] }@]"
    s.bound s.raises
    (Format.pp_print_list
       ~pp_sep:(fun ppf () -> Format.fprintf ppf ";@ ")
       pp_scope)
    s.inner

let rec gen_scope depth =
  let open Gen in
  let inner =
    if depth = 0 then constant []
    else list ~size:(int_range 0 3) (gen_scope (depth - 1))
  in
  (let+ bound = small_int and+ raises = bool and+ inner in
   { bound; raises; inner })
  |> with_pp pp_scope

(* Runs [s] inside a context whose [level] is [outer], checking that [level] is
   [s.bound] within [s] and [outer] again after it, returned or raised. *)
let rec enter outer s =
  (try
     context
       [ B (level, s.bound) ]
       (fun () ->
         equal int s.bound (value level);
         List.iter (enter s.bound) s.inner;
         if s.raises then raise Exit)
   with Exit -> ());
  equal int outer (value level)

let seen_on_another_domain v = Domain.join (Domain.spawn (fun () -> value v))

let unseen_while_another_domain_binds () =
  let bound = Atomic.make false and released = Atomic.make false in
  let binder =
    Domain.spawn (fun () ->
        context
          [ B (level, 7) ]
          (fun () ->
            Atomic.set bound true;
            while not (Atomic.get released) do
              Domain.cpu_relax ()
            done))
  in
  while not (Atomic.get bound) do
    Domain.cpu_relax ()
  done;
  let seen = value level in
  Atomic.set released true;
  Domain.join binder;
  equal int 0 seen

let exited_contexts_release_values () =
  let weak = Weak.create 1 in
  let bind () =
    let bound = String.make 16 'x' in
    Weak.set weak 0 (Some bound);
    context [ B (label, bound) ] (fun () -> equal string bound (value label))
  in
  bind ();
  Gc.full_major ();
  is_true ~msg:"the bound value was collected"
    (Option.is_none (Weak.get weak 0))

let contexts =
  group "context"
    [
      test "is the value of its function" (fun () ->
          equal int 3 (context [] (fun () -> 3)));
      test "binds a setting for the extent of its function" (fun () ->
          context [ B (level, 1) ] (fun () -> equal int 1 (value level));
          equal int 0 (value level));
      test "restores a setting when its function raises" (fun () ->
          raises Exit (fun () ->
              context [ B (level, 1) ] (fun () -> raise Exit));
          equal int 0 (value level));
      test "restores nested bindings in order" (fun () ->
          context
            [ B (level, 1) ]
            (fun () ->
              context
                [ B (level, 2) ]
                (fun () ->
                  context
                    [ B (level, 3); B (label, "inner") ]
                    (fun () ->
                      equal int 3 (value level);
                      equal string "inner" (value label));
                  equal int 2 (value level);
                  equal string "default" (value label));
              equal int 1 (value level));
          equal int 0 (value level));
      test "binds settings of different types together" (fun () ->
          context
            [ B (level, 4); B (label, "x"); B (Setting.noopt, true) ]
            (fun () ->
              equal (triple int string bool) (4, "x", true)
                (value level, value label, value Setting.noopt)));
      test
        "gives a setting bound twice its later binding, and restores the first \
         value" (fun () ->
          context
            [ B (level, 20); B (level, 30) ]
            (fun () -> equal int 30 (value level));
          equal int 0 (value level));
      test "restores recursive bindings" (fun () ->
          let rec recurse n =
            context [ B (level, n) ] (fun () -> if n > 0 then recurse (n - 1))
          in
          recurse 3;
          equal int 0 (value level));
      prop "binds within its extent and restores after it, returned or raised"
        (Gen.list ~size:(Gen.int_range 1 3) (gen_scope 3))
        (List.iter (enter 0));
      test "binds for the domains spawned while it runs" (fun () ->
          context
            [ B (level, 7) ]
            (fun () -> equal int 7 (seen_on_another_domain level)));
      test "is not seen by the other domains" unseen_while_another_domain_binds;
      test "binds and restores on the domain that runs it" (fun () ->
          equal int 7
            (Domain.join
               (Domain.spawn (fun () ->
                    context [ B (level, 7) ] (fun () -> value level))));
          equal int 0 (value level));
      test "keeps no bound value once it returns" exited_contexts_release_values;
    ]

(* Library settings *)

type setting = S : 'a Setting.t * ('a -> string) -> setting

let key (S (v, _)) = Setting.key v
let shown (S (v, show)) = show (value v)

(* A setting as tinygrad prints its value: switches are integers. *)
let number v = S (v, string_of_int)
let switch v = S (v, fun on -> if on then "1" else "0")
let word v = S (v, Fun.id)
let optional v = S (v, Option.fold ~none:"" ~some:string_of_int)

let settings =
  [
    number Setting.debug;
    number Setting.beam;
    optional Setting.jitbeam;
    switch Setting.noopt;
    switch Setting.no_color;
    number Setting.use_tc;
    number Setting.tc_select;
    number Setting.tc_opt;
    number Setting.tc_min_globals;
    number Setting.transcendental;
    switch Setting.split_reduceop;
    switch Setting.no_memory_planner;
    number Setting.ring;
    number Setting.all2all;
    switch Setting.allreduce_cast;
    number Setting.allreduce_node_ndevs;
    number Setting.cachelevel;
    switch Setting.ignore_beam_cache;
    switch Setting.disable_fast_idiv;
    number Setting.max_kernel_buffers;
    S (Setting.emulated_dtypes, String.concat ",");
    word Setting.default_float;
    word Setting.default_int;
    number Setting.parallel;
    number Setting.spec;
    switch Setting.check_oob;
    switch Setting.debug_rangeify;
    switch Setting.tuple_order;
    switch Setting.ccache;
    switch Setting.allow_tf32;
    number Setting.scache;
    switch Setting.disallow_broadcast;
    word Setting.sum_dtype;
    switch Setting.late_allreduce;
    number Setting.ring_allreduce_threshold;
    number Setting.reduceop_split_threshold;
    number Setting.reduceop_split_size;
    optional Setting.hcq_num_sdma;
    switch Setting.mv;
    switch Setting.dmc;
    switch Setting.allow_half8;
    switch Setting.expand_ssa;
    switch Setting.aligned;
    number Setting.waves_per_sh;
    switch Setting.beam_padto;
    number Setting.beam_uops_max;
    number Setting.beam_upcast_max;
    number Setting.beam_local_max;
    S (Setting.beam_min_progress, string_of_float);
    switch Setting.beam_estimate;
    switch Setting.beam_strict_mode;
    switch Setting.beam_log_surpass_max;
    number Setting.beam_debug;
    word Setting.cc;
    word Setting.cuda_path;
    word Setting.rocm_path;
    switch Setting.assert_compile;
    number Setting.rewrite_stack_limit;
    switch Setting.debug_linearize;
    word Setting.dbgtv;
  ]

(* The settings whose readers tolk does not port, as its README lists them. *)
let not_ported =
  (* They gate openpilot's pass and image paths. *)
  [ "OPENPILOT_HACKS"; "FLOAT16" ]
  (* Jit capture, kernel runs and allocation are rune's. *)
  @ [ "CAPTURING"; "MAX_BUFFER_SIZE"; "VALIDATE_WITH_CPU" ]
  (* tolk cannot open a device. *)
  @ [ "ALLOW_DEVICE_USAGE" ]
  (* A device's target follows from the device alone (D95). *)
  @ [ "DEV" ]
  (* Their other readers are not ported. *)
  @ [
      "IMAGE";
      "JIT";
      "WINO";
      "TRACEMETA";
      "TRAINING";
      "LRU";
      "HCQ2";
      "FUSE_OPTIM";
      "USE_ATOMICS";
      "CAPTURE_PROCESS_REPLAY";
      "NULL_ALLOW_COPYOUT";
      "VIZ";
      "PROFILE";
    ]

(* The golden leaves out the settings whose default it cannot record, and
   tinygrad's helpers do not declare those that tinygrad reads with [getenv]
   where it uses them, which hold the defaults [Setting] states: [JITBEAM]'s
   stands for [BEAM]'s value. *)
let unrecorded = [ "PARALLEL"; "NO_COLOR" ]

let getenv_defaults =
  [
    ("JITBEAM", "");
    ("SUM_DTYPE", "float32");
    ("LATE_ALLREDUCE", "1");
    ("RING_ALLREDUCE_THRESHOLD", "256000");
    ("REDUCEOP_SPLIT_THRESHOLD", "32768");
    ("REDUCEOP_SPLIT_SIZE", "22");
    ("HCQ_NUM_SDMA", "");
    ("MV", "1");
    ("DMC", "0");
    ("ALLOW_HALF8", "0");
    ("EXPAND_SSA", "0");
    ("ALIGNED", "1");
    ("WAVES_PER_SH", "0");
    ("BEAM_PADTO", "0");
    ("BEAM_UOPS_MAX", "3000");
    ("BEAM_UPCAST_MAX", "256");
    ("BEAM_LOCAL_MAX", "1024");
    ("BEAM_MIN_PROGRESS", "0.01");
    ("BEAM_ESTIMATE", "1");
    ("BEAM_STRICT_MODE", "0");
    ("BEAM_LOG_SURPASS_MAX", "0");
    ("BEAM_DEBUG", "0");
    ("CC", "clang");
    ("CUDA_PATH", "");
    ("ROCM_PATH", "/opt/rocm");
    ("ASSERT_COMPILE", "0");
    ("REWRITE_STACK_LIMIT", "250000");
    ("DEBUG_LINEARIZE", "0");
    ("DBGTV", "");
  ]

let tinygrad_settings = Golden.rows "settings.golden"
let tinygrad_keys = List.map (fun cell -> cell "key") tinygrad_settings
let ported s = List.mem (key s) tinygrad_keys

let unless_set key =
  if Option.is_some (Sys.getenv_opt key) then
    skip ~reason:(key ^ " is set in the environment") ()

(* A default that is not tinygrad's: schedules are kept on disk. *)
let diverging = [ ("SCACHE", "2") ]

let holds_tinygrad_default s =
  unless_set (key s);
  let cell =
    List.find (fun cell -> String.equal (cell "key") (key s)) tinygrad_settings
  in
  let default =
    Option.value (List.assoc_opt (key s) diverging) ~default:(cell "default")
  in
  equal string default (shown s)

(* What the caches key on *)

(* The library's settings whose change can change what a compilation returns:
   the caches key on them. The others print, check, keep, look up, work in
   parallel or raise; among them [CC], [CUDA_PATH] and [ROCM_PATH], read when a
   compiler is made, whose cache key names its tools. *)
let output_settings =
  [
    "ALIGNED";
    "ALL2ALL";
    "ALLOW_HALF8";
    "ALLOW_TF32";
    "ALLREDUCE_CAST";
    "ALLREDUCE_NODE_NDEVS";
    "BEAM";
    "BEAM_ESTIMATE";
    "BEAM_LOCAL_MAX";
    "BEAM_MIN_PROGRESS";
    "BEAM_PADTO";
    "BEAM_UOPS_MAX";
    "BEAM_UPCAST_MAX";
    "DEFAULT_FLOAT";
    "DEFAULT_INT";
    "DISABLE_FAST_IDIV";
    "DMC";
    "EMULATED_DTYPES";
    "EXPAND_SSA";
    "HCQ_NUM_SDMA";
    "JITBEAM";
    "LATE_ALLREDUCE";
    "MAX_KERNEL_BUFFERS";
    "MV";
    "NOOPT";
    "NO_MEMORY_PLANNER";
    "REDUCEOP_SPLIT_SIZE";
    "REDUCEOP_SPLIT_THRESHOLD";
    "RING";
    "RING_ALLREDUCE_THRESHOLD";
    "SPLIT_REDUCEOP";
    "SUM_DTYPE";
    "TC";
    "TC_MIN_GLOBALS";
    "TC_OPT";
    "TC_SELECT";
    "TRANSCENDENTAL";
    "TUPLE_ORDER";
    "WAVES_PER_SH";
  ]

(* Each exported setting of output bound to a value other than its default. *)
let other_outputs =
  Setting.
    [
      B (beam, 1);
      B (jitbeam, Some 0);
      B (noopt, true);
      B (use_tc, 2);
      B (tc_select, 0);
      B (tc_opt, 1);
      B (tc_min_globals, 1);
      B (transcendental, 2);
      B (split_reduceop, false);
      B (no_memory_planner, true);
      B (ring, 2);
      B (all2all, 1);
      B (allreduce_cast, false);
      B (allreduce_node_ndevs, 2);
      B (disable_fast_idiv, false);
      B (max_kernel_buffers, 8);
      B (emulated_dtypes, [ "half" ]);
      B (default_float, "half");
      B (default_int, "long");
      B (tuple_order, false);
      B (allow_tf32, true);
    ]

let keyed key = List.assoc_opt key (shaping ())
let entries = list (pair string string)

(* The fewest words one of ten calls of [f] allocates on the minor heap. *)
let words f =
  let fewest = ref max_int in
  for _ = 1 to 10 do
    let before = Gc.minor_words () in
    ignore (Sys.opaque_identity (f ()));
    fewest := min !fewest (int_of_float (Gc.minor_words () -. before))
  done;
  !fewest

(* [shaping ()] read again after [f] is the list read before it. *)
let unchanged_by f =
  let before = shaping () in
  f ();
  satisfies ~claim:"the list read before" entries
    (fun l -> l == before)
    (shaping ())

let shown_as value make =
  let key = fresh () in
  setenv key value;
  ignore (make key);
  keyed key

let cache_keys =
  group "shaping"
    [
      test "holds a setting's current value on the calling domain" (fun () ->
          let key = fresh () in
          let v = Setting.int ~reach:Output key 1 in
          equal (option string) ~msg:"declared" (Some "1") (keyed key);
          equal (option string) ~msg:"in a context" (Some "4")
            (context [ B (v, 4) ] (fun () -> keyed key)));
      test "holds a setting's value again once a context ends" (fun () ->
          let key = fresh () in
          let v = Setting.int ~reach:Output key 1 in
          ignore (keyed key);
          context [ B (v, 4) ] (fun () -> ignore (keyed key));
          equal (option string) (Some "1") (keyed key));
      test "holds a setting declared after it was read" (fun () ->
          ignore (shaping ());
          let key = fresh () in
          ignore (Setting.bool ~reach:Output key true);
          equal (option string) (Some "true") (keyed key));
      test "shows each type's value as text" (fun () ->
          let shown value make = shown_as (Some value) make in
          equal (option string) ~msg:"int" (Some "6")
            (shown " 6 " (fun key -> Setting.int ~reach:Output key 2));
          equal (option string) ~msg:"switch" (Some "false")
            (shown "0" (fun key -> Setting.bool ~reach:Output key true));
          equal (option string) ~msg:"float" (Some "0x1.4p+1")
            (shown " 2.5 " (fun key -> Setting.float ~reach:Output key 0.));
          equal (option string) ~msg:"string" (Some " a,b ")
            (shown " a,b " (fun key -> Setting.string ~reach:Output key ""));
          equal (option string) ~msg:"optional" (Some "3")
            (shown "3" (fun key -> Setting.int_option ~reach:Output key));
          equal (option string) ~msg:"unset optional" (Some "")
            (shown_as None (fun key -> Setting.int_option ~reach:Output key)));
      test "tells apart floats one ulp apart" (fun () ->
          let float key = Setting.float ~reach:Output key 0. in
          not_equal (option string)
            (shown_as (Some "0.1") float)
            (shown_as (Some "0.10000000000000002") float));
      test "a domain spawned in a context holds the values it spawned with"
        (fun () ->
          let key = fresh () in
          let v = Setting.int ~reach:Output key 1 in
          ignore (keyed key);
          let inside =
            context
              [ B (v, 4) ]
              (fun () ->
                ignore (keyed key);
                Domain.spawn (fun () -> keyed key))
          in
          let after = Domain.spawn (fun () -> keyed key) in
          equal (option string) ~msg:"spawned in the context" (Some "4")
            (Domain.join inside);
          equal (option string) ~msg:"spawned after it" (Some "1")
            (Domain.join after));
      test "leaves out a setting that reaches only the process" (fun () ->
          equal (option string) None
            (shown_as (Some "1") (fun key ->
                 Setting.bool ~reach:Process key false)));
      test "holds the library's settings of output, and those alone" (fun () ->
          let library k = not (String.starts_with ~prefix:"TOLK_TEST_" k) in
          equal (list string) output_settings
            (List.filter library (List.map fst (shaping ()))));
      cases
        ~name:(fun (Setting.B (v, _)) -> Setting.key v)
        "changes a library setting's entry when the setting changes"
        other_outputs
        (fun (B (v, x)) ->
          let before = keyed (Setting.key v) in
          not_equal (option string) before
            (context [ B (v, x) ] (fun () -> keyed (Setting.key v))));
      test "is sorted by name" (fun () ->
          let names = List.map fst (shaping ()) in
          equal (list string) (List.sort compare names) names);
      test "is the same list while no setting that reaches output changes"
        (fun () ->
          let v = Setting.int ~reach:Output (fresh ()) 1 in
          unchanged_by (fun () -> ignore (shaping ()));
          unchanged_by (fun () -> context [ B (Setting.debug, 3) ] ignore);
          unchanged_by (fun () ->
              ignore (Setting.int ~reach:Process (fresh ()) 0));
          unchanged_by (fun () ->
              ignore
                (Domain.join
                   (Domain.spawn (fun () -> context [ B (v, 2) ] shaping)))));
      test "is read without allocating while nothing changes" (fun () ->
          ignore (shaping ());
          equal int (words (fun () -> [])) (words shaping));
    ]

let library_settings =
  group "settings"
    [
      test "are tinygrad's, except those tolk does not port" (fun () ->
          equal
            (slist string String.compare)
            not_ported
            (List.filter
               (fun k -> not (List.exists (fun s -> key s = k) settings))
               tinygrad_keys));
      test "missing from the golden are those whose default it cannot record"
        (fun () ->
          equal
            (slist string String.compare)
            (unrecorded @ List.map fst getenv_defaults)
            (List.filter_map
               (fun s -> if ported s then None else Some (key s))
               settings));
      cases ~name:key
        "hold tinygrad's default when their variable is unset, but SCACHE"
        (List.filter ported settings)
        holds_tinygrad_default;
      cases ~name:fst
        "read by tinygrad where it uses them hold the default Setting states \
         when their variable is unset"
        getenv_defaults (fun (k, default) ->
          unless_set k;
          equal string default
            (shown (List.find (fun s -> String.equal (key s) k) settings)));
      test "no_color is off when NO_COLOR is unset" (fun () ->
          unless_set "NO_COLOR";
          equal bool false (value Setting.no_color));
      test "parallel is between one and the domains the runtime recommends"
        (fun () ->
          unless_set "PARALLEL";
          at_least int ~than:1 (value Setting.parallel);
          at_most int
            ~than:(Domain.recommended_domain_count ())
            (value Setting.parallel));
    ]

(* Processes

   What a process reads when it starts is observed in a child: this suite,
   started again with [role] set, plays that role and exits. *)

let role = "TOLK_TEST_SETTING_ROLE"

let play = function
  | "settings" ->
      List.iter (fun s -> Printf.printf "%s=%s\n" (key s) (shown s)) settings
  | role -> failwith ("unknown role " ^ role)

type ended = { status : Unix.process_status; out : string; err : string }

(* [start ~env part args] starts playing [part] with [args] in an environment
   where each [(name, Some v)] of [env] is set and each [(name, None)] unset. *)
let start ?(env = []) part args =
  let unchanged binding =
    match String.index_opt binding '=' with
    | Some i -> not (List.mem_assoc (String.sub binding 0 i) env)
    | None -> true
  in
  let set =
    List.filter_map
      (fun (k, v) -> Option.map (fun v -> k ^ "=" ^ v) v)
      ((role, Some part) :: env)
  in
  let environment =
    Array.of_list
      (set @ List.filter unchanged (Array.to_list (Unix.environment ())))
  in
  let out = temp_file () and err = temp_file () in
  let fd file = Unix.openfile file [ Unix.O_WRONLY ] 0 in
  let out_fd = fd out and err_fd = fd err in
  let argv = Array.of_list (Sys.executable_name :: args) in
  let pid =
    Unix.create_process_env Sys.executable_name argv environment Unix.stdin
      out_fd err_fd
  in
  Unix.close out_fd;
  Unix.close err_fd;
  (pid, out, err)

let finish (pid, out, err) =
  let _, status = Unix.waitpid [] pid in
  let read file = In_channel.with_open_bin file In_channel.input_all in
  { status; out = read out; err = read err }

let child ?env part args = finish (start ?env part args)

let status =
  Testable.make
    ~pp:(fun ppf s ->
      match s with
      | Unix.WEXITED n -> Format.fprintf ppf "exited %d" n
      | Unix.WSIGNALED n -> Format.fprintf ppf "killed by signal %d" n
      | Unix.WSTOPPED n -> Format.fprintf ppf "stopped by signal %d" n)
    ~equal:( = )

let succeeds ended =
  equal ~msg:ended.err status (Unix.WEXITED 0) ended.status;
  ended.out

let settings_read env =
  let pair line =
    match String.index_opt line '=' with
    | Some i ->
        Some
          ( String.sub line 0 i,
            String.sub line (i + 1) (String.length line - i - 1) )
    | None -> None
  in
  List.filter_map pair
    (String.split_on_char '\n' (succeeds (child ~env "settings" [])))

let read_at_start env expected =
  let read = settings_read env in
  equal
    (list (pair string (option string)))
    (List.map (fun (k, v) -> (k, Some v)) expected)
    (List.map (fun (k, _) -> (k, List.assoc_opt k read)) expected)

let fails_at_start env =
  let ended = child ~env "settings" [] in
  not_equal status (Unix.WEXITED 0) ended.status;
  contains ~sub:"Invalid_argument" ended.err

let startup =
  group "startup"
    [
      test "reads each setting from its variable" (fun () ->
          read_at_start
            [
              ("DEBUG", Some "3");
              ("NOOPT", Some "2");
              ("CCACHE", Some "0");
              ("TC", Some "2");
              ("DEFAULT_FLOAT", Some "half");
              ("PARALLEL", Some "0");
            ]
            [
              ("DEBUG", "3");
              ("NOOPT", "1");
              ("CCACHE", "0");
              ("TC", "2");
              ("DEFAULT_FLOAT", "half");
              ("PARALLEL", "0");
            ]);
      test
        "reads EMULATED_DTYPES as names separated by commas, dropping empty \
         ones" (fun () ->
          read_at_start
            [ ("EMULATED_DTYPES", Some ",half,,bfloat16, ") ]
            [ ("EMULATED_DTYPES", "half,bfloat16, ") ]);
      test "starts whatever DEV holds" (fun () ->
          read_at_start
            [ ("DEV", Some "PCI+NV+CUDA"); ("DEBUG", Some "3") ]
            [ ("DEBUG", "3") ]);
      test "fails on a setting that is no integer" (fun () ->
          fails_at_start [ ("DEBUG", Some "two") ]);
    ]

let () =
  match Sys.getenv_opt role with
  | Some r -> play r
  | None ->
      exit
        (run "Tolk.Setting"
           [
             environment;
             declaration;
             contexts;
             cache_keys;
             library_settings;
             startup;
           ])
