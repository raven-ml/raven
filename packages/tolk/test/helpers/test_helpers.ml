open Windtrap
open Tolk.Helpers

(* Goldens hold arbitrary text as OCaml string literals. *)
let literal cell = Scanf.sscanf cell "%S%!" Fun.id

(* [claim] checked on each row of a golden. *)
let as_tinygrad ?key claim file check =
  group claim [ Golden.cases ?key file check ]

(* The golden's text for an exception tinygrad raised. *)
let raised cell =
  List.mem cell [ "ValueError"; "ZeroDivisionError"; "RuntimeError" ]

let refuses f = raises_match (fun e -> Exn.invalid_arg e) f

(* Witnesses and generators *)

let pp_target ppf (t : Target.t) =
  Format.fprintf ppf
    "{ device = %S; renderer = %S; arch = %S; interface = %S; indices = %S }"
    t.device t.renderer t.arch t.interface t.indices

let target_w = Testable.make ~pp:pp_target ~equal:( = )

(* An integer drawn across the edges of [int]. *)
let any_int =
  Gen.frequency
    [
      (6, Gen.small_int);
      (2, Gen.int);
      ( 1,
        Gen.of_list ~pp:Format.pp_print_int
          [ min_int; min_int + 1; max_int - 1; max_int ] );
    ]

let utf_8 uchars =
  let b = Buffer.create 16 in
  List.iter (Buffer.add_utf_8_uchar b) uchars;
  Buffer.contents b

let gen_text =
  Gen.map utf_8 (Gen.list Gen.uchar)
  |> Gen.with_pp (fun ppf s -> Format.fprintf ppf "%S" s)

let ascii_letter = Gen.char_range 'a' 'z'

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
  Context_var.value (make key)

let int_setting key = Context_var.int ~reach:Process key 0
let float_setting key = Context_var.float ~reach:Process key 0.

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
            (declared_with None (fun key ->
                 Context_var.int ~reach:Process key 5)));
      test "an int setting starts from its variable when set" (fun () ->
          equal int 12
            (declared_with (Some " 12 ") (fun key ->
                 Context_var.int ~reach:Process key 5)));
      test "an int setting refuses a variable that is no integer" (fun () ->
          refuses (fun () -> declared_with (Some "abc") int_setting));
      test "a float setting starts from its default when its variable is unset"
        (fun () ->
          equal float_exact 0.5
            (declared_with None (fun key ->
                 Context_var.float ~reach:Process key 0.5)));
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
                 Context_var.bool ~reach:Process key (not on))));
      test "a switch starts from its default when its variable is unset"
        (fun () ->
          let switch default key =
            Context_var.bool ~reach:Process key default
          in
          equal bool true (declared_with None (switch true));
          equal bool false (declared_with None (switch false)));
      test "a switch refuses a variable that is no integer" (fun () ->
          refuses (fun () ->
              declared_with (Some "true") (fun key ->
                  Context_var.bool ~reach:Process key false)));
      test "a string setting starts from its variable as written" (fun () ->
          let word key = Context_var.string ~reach:Process key "d" in
          equal string " x " (declared_with (Some " x ") word);
          equal string "" (declared_with (Some "") word);
          equal string "d" (declared_with None word));
      test "an optional setting holds its variable's integer, if set" (fun () ->
          let opt key = Context_var.int_option ~reach:Process key in
          equal (option int) (Some (-3)) (declared_with (Some " -3 ") opt);
          equal (option int) None (declared_with None opt);
          refuses (fun () -> declared_with (Some "") opt));
    ]

(* Declarations *)

let declaration =
  group "Context_var"
    [
      test "a setting reads its variable once, when declared" (fun () ->
          let key = fresh () in
          setenv key (Some "1");
          let v = int_setting key in
          setenv key (Some "2");
          equal int 1 (Context_var.value v));
      test "a setting whose variable is refused is not declared" (fun () ->
          let key = fresh () in
          setenv key (Some "abc");
          refuses (fun () -> int_setting key);
          setenv key (Some "4");
          equal int 4 (Context_var.value (int_setting key)));
      test "key is the name of the setting's variable" (fun () ->
          let key = fresh () in
          equal string key
            (Context_var.key (Context_var.bool ~reach:Process key false)));
      test "a name is declared once, whatever the setting's type and reach"
        (fun () ->
          let key = fresh () in
          ignore (Context_var.int ~reach:Output key 0);
          refuses (fun () -> Context_var.int ~reach:Output key 0);
          refuses (fun () -> Context_var.bool ~reach:Process key false);
          refuses (fun () -> Context_var.float ~reach:Output key 0.);
          refuses (fun () -> Context_var.string ~reach:Process key "");
          refuses (fun () -> Context_var.int_option ~reach:Output key));
      test "a library setting's name cannot be declared again" (fun () ->
          refuses (fun () -> Context_var.int ~reach:Process "DEBUG" 0));
      test "a setting declared inside a context stays declared after it"
        (fun () ->
          let key = fresh () in
          context [ B (debug, 1) ] (fun () -> ignore (int_setting key));
          refuses (fun () -> int_setting key));
    ]

(* Contexts *)

let level = Context_var.int ~reach:Output "TOLK_TEST_LEVEL" 0
let label = Context_var.string ~reach:Output "TOLK_TEST_LABEL" "default"
let value = Context_var.value

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
            [ B (level, 4); B (label, "x"); B (noopt, true) ]
            (fun () ->
              equal (triple int string bool) (4, "x", true)
                (value level, value label, value noopt)));
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

type setting = S : 'a Context_var.t * ('a -> string) -> setting

let key (S (v, _)) = Context_var.key v
let shown (S (v, show)) = show (value v)

(* A setting as tinygrad prints its value: switches are integers. *)
let number v = S (v, string_of_int)
let switch v = S (v, fun on -> if on then "1" else "0")
let word v = S (v, Fun.id)
let optional v = S (v, Option.fold ~none:"" ~some:string_of_int)

let settings =
  [
    number debug;
    number beam;
    switch noopt;
    switch no_color;
    number use_tc;
    number tc_select;
    optional tc_opt;
    number tc_min_globals;
    number transcendental;
    switch split_reduceop;
    switch no_memory_planner;
    number ring;
    number all2all;
    switch allreduce_cast;
    number allreduce_node_ndevs;
    number cachelevel;
    switch ignore_beam_cache;
    switch disable_fast_idiv;
    number max_kernel_buffers;
    S (emulated_dtypes, String.concat ",");
    word default_float;
    word default_int;
    number parallel;
    number spec;
    switch check_oob;
    switch debug_rangeify;
    switch tuple_order;
    switch ccache;
    switch allow_tf32;
    number scache;
    switch disallow_broadcast;
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

(* The golden leaves out the settings whose default it cannot record. *)
let unrecorded = [ "PARALLEL"; "NO_COLOR" ]
let tinygrad_settings = Golden.rows "settings.golden"
let tinygrad_keys = List.map (fun cell -> cell "key") tinygrad_settings
let ported s = List.mem (key s) tinygrad_keys

let unless_set key =
  if Option.is_some (Sys.getenv_opt key) then
    skip ~reason:(key ^ " is set in the environment") ()

(* Defaults that are not tinygrad's: schedules are kept on disk, and TC_OPT
   leaves its default to its readers, the search's being 2. *)
let diverging = [ ("SCACHE", "2"); ("TC_OPT", "") ]

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
          let v = Context_var.int ~reach:Output key 1 in
          equal (option string) ~msg:"declared" (Some "1") (keyed key);
          equal (option string) ~msg:"in a context" (Some "4")
            (context [ B (v, 4) ] (fun () -> keyed key)));
      test "holds a setting's value again once a context ends" (fun () ->
          let key = fresh () in
          let v = Context_var.int ~reach:Output key 1 in
          ignore (keyed key);
          context [ B (v, 4) ] (fun () -> ignore (keyed key));
          equal (option string) (Some "1") (keyed key));
      test "holds a setting declared after it was read" (fun () ->
          ignore (shaping ());
          let key = fresh () in
          ignore (Context_var.bool ~reach:Output key true);
          equal (option string) (Some "true") (keyed key));
      test "shows each type's value as text" (fun () ->
          let shown value make = shown_as (Some value) make in
          equal (option string) ~msg:"int" (Some "6")
            (shown " 6 " (fun key -> Context_var.int ~reach:Output key 2));
          equal (option string) ~msg:"switch" (Some "false")
            (shown "0" (fun key -> Context_var.bool ~reach:Output key true));
          equal (option string) ~msg:"float" (Some "0x1.4p+1")
            (shown " 2.5 " (fun key -> Context_var.float ~reach:Output key 0.));
          equal (option string) ~msg:"string" (Some " a,b ")
            (shown " a,b " (fun key -> Context_var.string ~reach:Output key ""));
          equal (option string) ~msg:"optional" (Some "3")
            (shown "3" (fun key -> Context_var.int_option ~reach:Output key));
          equal (option string) ~msg:"unset optional" (Some "")
            (shown_as None (fun key -> Context_var.int_option ~reach:Output key)));
      test "tells apart floats one ulp apart" (fun () ->
          let float key = Context_var.float ~reach:Output key 0. in
          not_equal (option string)
            (shown_as (Some "0.1") float)
            (shown_as (Some "0.10000000000000002") float));
      test "a domain spawned in a context holds the values it spawned with"
        (fun () ->
          let key = fresh () in
          let v = Context_var.int ~reach:Output key 1 in
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
                 Context_var.bool ~reach:Process key false)));
      test "leaves out the library's settings of the process" (fun () ->
          List.iter
            (fun key -> equal (option string) ~msg:key None (keyed key))
            [
              "DEBUG";
              "BEAM";
              "JITBEAM";
              "CACHELEVEL";
              "SCACHE";
              "CCACHE";
              "PARALLEL";
            ]);
      test "is sorted by name" (fun () ->
          let names = List.map fst (shaping ()) in
          equal (list string) (List.sort compare names) names);
      test "is the same list while no setting that reaches output changes"
        (fun () ->
          let v = Context_var.int ~reach:Output (fresh ()) 1 in
          unchanged_by (fun () -> ignore (shaping ()));
          unchanged_by (fun () -> context [ B (debug, 3) ] ignore);
          unchanged_by (fun () ->
              ignore (Context_var.int ~reach:Process (fresh ()) 0));
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
            unrecorded
            (List.filter_map
               (fun s -> if ported s then None else Some (key s))
               settings));
      cases ~name:key
        "hold tinygrad's default when their variable is unset, but SCACHE and \
         TC_OPT"
        (List.filter ported settings)
        holds_tinygrad_default;
      test "no_color is off when NO_COLOR is unset" (fun () ->
          unless_set "NO_COLOR";
          equal bool false (value no_color));
      test "parallel is between one and the domains the runtime recommends"
        (fun () ->
          unless_set "PARALLEL";
          at_least int ~than:1 (value parallel);
          at_most int
            ~than:(Domain.recommended_domain_count ())
            (value parallel));
    ]

(* Processes

   What a process reads when it starts is observed in a child: this suite,
   started again with [role] set, plays that role and exits. *)

let role = "TOLK_TEST_HELPERS_ROLE"

let play = function
  | "settings" ->
      List.iter (fun s -> Printf.printf "%s=%s\n" (key s) (shown s)) settings;
      Printf.printf "cache_dir=%s\ncachedb=%s\n" cache_dir cachedb
  | "put" -> Diskcache.put ~table:Sys.argv.(1) Sys.argv.(2) Sys.argv.(3)
  | "get" ->
      print_string
        (Option.value ~default:"<none>"
           (Diskcache.get ~table:Sys.argv.(1) Sys.argv.(2)))
  | "write" ->
      let table = Sys.argv.(1)
      and payload = String.make (1 lsl 20) Sys.argv.(2).[0] in
      for _ = 1 to 20 do
        Diskcache.put ~table "k" payload
      done
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

let macos () =
  let uname = Unix.open_process_args_in "uname" [| "uname"; "-s" |] in
  Fun.protect
    ~finally:(fun () -> ignore (Unix.close_process_in uname))
    (fun () -> String.equal "Darwin" (String.trim (In_channel.input_all uname)))

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
      test "puts the cache in tolk under XDG_CACHE_HOME" (fun () ->
          read_at_start
            [ ("XDG_CACHE_HOME", Some "/xdg"); ("CACHEDB", None) ]
            [ ("cache_dir", "/xdg/tolk"); ("cachedb", "/xdg/tolk/cache") ]);
      test
        "puts the cache in the platform's cache directory without \
         XDG_CACHE_HOME" (fun () ->
          let home =
            if macos () then "/home/Library/Caches/tolk"
            else "/home/.cache/tolk"
          in
          read_at_start
            [
              ("XDG_CACHE_HOME", None); ("HOME", Some "/home"); ("CACHEDB", None);
            ]
            [ ("cache_dir", home); ("cachedb", home ^ "/cache") ]);
      test "makes the default cache directory absolute" (fun () ->
          read_at_start
            [ ("XDG_CACHE_HOME", Some "relative"); ("CACHEDB", None) ]
            [
              ("cachedb", Filename.concat (Sys.getcwd ()) "relative/tolk/cache");
            ]);
      test "reads the cache's directory from CACHEDB" (fun () ->
          read_at_start [ ("CACHEDB", Some "/db") ] [ ("cachedb", "/db") ]);
    ]

(* Targets *)

let gen_field chars =
  Gen.string_of ~size:(Gen.int_range 0 4)
    (Gen.of_list ~pp:Format.pp_print_char chars)

let upper =
  List.init 26 (fun i -> Char.chr (Char.code 'A' + i)) @ [ '0'; '9'; '_' ]

let any_case = upper @ [ 'a'; 'x'; ','; '.'; '-' ]

(* Targets whose fields hold no separator, with their device and renderer in
   upper case, as parse makes them. *)
let gen_target =
  (let open Gen in
   let+ device = gen_field upper
   and+ renderer = gen_field upper
   and+ arch = gen_field any_case
   and+ interface = gen_field any_case
   and+ indices = gen_field any_case in
   Target.{ device; renderer; arch; interface; indices })
  |> Gen.with_pp pp_target

let parsed_like_tinygrad cell =
  let input = cell "input" in
  match Target.of_string input with
  | Error e when raised (cell "device") -> contains ~sub:(cell "names") e
  | Error e -> failf "tinygrad parses %S, parse fails with %S" input e
  | Ok t ->
      equal target_w
        Target.
          {
            device = cell "device";
            renderer = cell "renderer";
            arch = cell "arch";
            interface = cell "interface";
            indices = cell "indices";
          }
        t;
      equal string (cell "to_string") (Format.asprintf "%a" Target.pp t)

let targets =
  group "Target"
    [
      as_tinygrad "of_string reads a target as tinygrad does" "targets.golden"
        parsed_like_tinygrad;
      prop "of_string reads back what pp writes" gen_target
        (Law.round_trip target_w string (Format.asprintf "%a" Target.pp)
           (fun s -> require_ok ~pp:Format.pp_print_string (Target.of_string s)));
    ]

(* Integers and lists *)

let divides_like_tinygrad cell =
  let x = int_of_string (cell "x") and y = int_of_string (cell "y") in
  let check column f =
    match cell column with
    | "None" -> ()
    | "ZeroDivisionError" ->
        raises ~msg:column Division_by_zero (fun () -> f x y)
    | expected -> equal ~msg:column int (int_of_string expected) (f x y)
  in
  check "floordiv" floordiv;
  check "floormod" floormod;
  check "ceildiv" ceildiv;
  check "round_up" round_up

(* A divisor other than 0, and no quotient beyond int's range. *)
let gen_division =
  Gen.such_that
    (fun (x, y) -> y <> 0 && not (x = min_int && y = -1))
    (Gen.pair any_int any_int)

module Caseless = struct
  type t = string

  let equal a b =
    String.equal (String.lowercase_ascii a) (String.lowercase_ascii b)

  let hash s = Hashtbl.hash (String.lowercase_ascii s)
end

(* Keeps each element that no earlier element equals. *)
let first_occurrences equal l =
  List.rev
    (List.fold_left
       (fun kept x -> if List.exists (equal x) kept then kept else x :: kept)
       [] l)

let gen_words =
  Gen.list (Gen.of_list ~pp:Format.pp_print_string [ "a"; "A"; "b"; "B"; "c" ])

let gen_permutation =
  Gen.bind (Gen.int_range 0 8) (fun n ->
      Gen.permutation ~pp:Format.pp_print_int (List.init n Fun.id))

let stable_argsort l =
  List.map snd
    (List.stable_sort
       (fun (a, _) (b, _) -> compare a b)
       (List.mapi (fun i x -> (x, i)) l))

let integers =
  group "integers"
    [
      as_tinygrad ~key:[ "x"; "y" ] "divide as tinygrad does" "division.golden"
        divides_like_tinygrad;
      prop
        "floordiv and floormod split x into y times a quotient and a remainder \
         of y's sign"
        gen_division (fun (x, y) ->
          equal int x ((floordiv x y * y) + floormod x y);
          satisfies ~claim:"between 0 and y, y excluded" int
            (fun r -> if y > 0 then 0 <= r && r < y else y < r && r <= 0)
            (floormod x y));
      prop "ceildiv x y is minus the floor of -x over y"
        (Gen.pair Gen.small_int (Gen.such_that (( <> ) 0) Gen.small_int))
        (fun (x, y) -> equal int (-(floordiv (-x) y)) (ceildiv x y));
      prop "round_up is the least multiple of a positive divisor at least x"
        (Gen.pair Gen.small_int (Gen.int_range 1 100))
        (fun (x, y) ->
          let r = round_up x y in
          equal ~msg:"a multiple" int 0 (floormod r y);
          at_least int ~than:x r;
          less int ~than:(x + y) r);
      test "round_up keeps a multiple and rounds up a negative number"
        (fun () ->
          equal (list int)
            [ 0; -4; 8; 8; 24984; 25056 ]
            [
              round_up (-3) 4;
              round_up (-4) 4;
              round_up 6 4;
              round_up 8 4;
              round_up 232 24984;
              round_up 24984 232;
            ]);
      prop "data64 splits an integer into its high and low 32 bits" any_int
        (Law.round_trip int (pair int int) data64 (fun (hi, lo) ->
             (hi lsl 32) lor lo));
      prop "lo32 is below 2^32 and hi32 keeps the sign" any_int (fun x ->
          equal (pair int int) (hi32 x, lo32 x) (data64 x);
          equal (pair int int) (lo32 x, hi32 x) (data64_le x);
          is_true ~msg:"0 <= lo32 < 2^32" (lo32 x >= 0 && lo32 x < 1 lsl 32);
          equal ~msg:"hi32 is negative iff x is" bool (x < 0) (hi32 x < 0));
      test "prod of nothing is 1" (fun () -> equal int 1 (prod []));
      test "prod multiplies" (fun () -> equal int 30 (prod [ 2; 3; 5 ]));
      prop "prod maps concatenation to multiplication"
        (Gen.pair (Gen.list Gen.small_int) (Gen.list Gen.small_int))
        (Law.homomorphic (list int) int prod ( @ ) ( * ));
      prop "dedup keeps each element at its first occurrence" gen_words
        (fun l ->
          equal (list string)
            (first_occurrences Caseless.equal l)
            (dedup (module Caseless) l));
      prop "dedup is idempotent" gen_words
        (Law.idempotent (list string) (dedup (module Caseless)));
      prop
        "argsort orders positions by element, keeping equal elements in order"
        (Gen.list (Gen.int_range 0 4))
        (fun l -> equal (list int) (stable_argsort l) (argsort l));
      prop "argsort inverts a permutation" gen_permutation
        (Law.involutive (list int) argsort);
      cases
        ~name:(fun (l, _) -> Format.asprintf "%a" (Testable.pp (list int)) l)
        "all_same is true iff every element equals the first"
        [
          ([], true);
          ([ 1 ], true);
          ([ 1; 1; 1 ], true);
          ([ 1; 2; 1 ], false);
          ([ 1; 1; 2 ], false);
        ]
        (fun (l, same) -> equal bool same (all_same Int.equal l));
      test "get_single_element is the element of a one-element list" (fun () ->
          equal int 4 (get_single_element [ 4 ]));
      test "get_single_element refuses other lists" (fun () ->
          refuses (fun () -> get_single_element []);
          refuses (fun () -> get_single_element [ 1; 2 ]));
    ]

(* Selection *)

let selects_like_tinygrad cell =
  let names =
    match cell "candidates" with "" -> [] | s -> String.split_on_char ',' s
  in
  let candidates = List.mapi (fun i name -> (i, name)) names in
  let outcome =
    Result.map
      (fun selected ->
        String.concat "," (List.map (fun (i, _) -> string_of_int i) selected))
      (select_by_name ~error:"no match" snd (cell "query") candidates)
  in
  let expected =
    match cell "error" with "" -> Ok (cell "selected") | error -> Error error
  in
  equal (result string string) expected outcome

let initializes name calls () =
  calls := name :: !calls;
  Ok name

let fails message calls () =
  calls := message :: !calls;
  Error message

let first_inited candidates =
  let calls = ref [] in
  let outcome =
    select_first_inited ~error:"no candidate"
      (List.map (fun c -> c calls) candidates)
  in
  (outcome, List.rev !calls)

let tried = pair (result string string) (list string)

let selection =
  group "selection"
    [
      as_tinygrad ~key:[ "candidates"; "query" ]
        "select_by_name selects as tinygrad does" "selection.golden"
        selects_like_tinygrad;
      test "select_first_inited takes the first candidate and tries no other"
        (fun () ->
          equal tried (Ok "a", [ "a" ])
            (first_inited [ initializes "a"; initializes "b" ]));
      test "select_first_inited tries the next candidate after a failure"
        (fun () ->
          equal tried
            (Ok "b", [ "a failed"; "b" ])
            (first_inited [ fails "a failed"; initializes "b" ]));
      test "select_first_inited reports the only candidate's error as it is"
        (fun () ->
          equal tried
            (Error "a failed", [ "a failed" ])
            (first_inited [ fails "a failed" ]));
      test "select_first_inited reports every candidate's error, one per line"
        (fun () ->
          equal tried
            ( Error "no candidate\na failed\nb failed",
              [ "a failed"; "b failed" ] )
            (first_inited [ fails "a failed"; fails "b failed" ]));
      test "select_first_inited without candidates reports its error" (fun () ->
          equal tried (Error "no candidate", []) (first_inited []));
      test
        "select_first_inited lets a candidate's exception escape, trying no \
         other" (fun () ->
          let calls = ref [] in
          raises Exit (fun () ->
              select_first_inited ~error:"no candidate"
                [ (fun () -> raise Exit); initializes "b" calls ]);
          equal (list string) [] !calls);
    ]

(* Terminal text *)

let text_input cell = literal (cell "input")

let colors =
  [
    (Black, "black");
    (Red, "red");
    (Green, "green");
    (Yellow, "yellow");
    (Blue, "blue");
    (Magenta, "magenta");
    (Cyan, "cyan");
    (White, "white");
    (Bright_black, "BLACK");
    (Bright_red, "RED");
    (Bright_green, "GREEN");
    (Bright_yellow, "YELLOW");
    (Bright_blue, "BLUE");
    (Bright_magenta, "MAGENTA");
    (Bright_cyan, "CYAN");
    (Bright_white, "WHITE");
  ]

let identifier = function
  | 'a' .. 'z' | 'A' .. 'Z' | '0' .. '9' | '_' -> true
  | _ -> false

let gen_color =
  Gen.of_list ~pp:(fun ppf (_, n) -> Format.pp_print_string ppf n) colors

let with_color f = context [ B (no_color, false) ] f

let colored_like_tinygrad cell =
  let color =
    fst (List.find (fun (_, n) -> String.equal n (cell "color")) colors)
  in
  let background =
    bool_of_string (String.lowercase_ascii (cell "background"))
  in
  with_color (fun () ->
      equal string (literal (cell "colored")) (colored ~background color "x"))

let strip_escapes = String.map (fun c -> if c = '\027' then '[' else c)
let without_escape = Gen.map strip_escapes gen_text

let terminal_text =
  group "terminal text"
    [
      as_tinygrad ~key:[ "color"; "background" ]
        "colored paints as tinygrad does" "colors.golden" colored_like_tinygrad;
      test "colored leaves text alone under no_color" (fun () ->
          context
            [ B (no_color, true) ]
            (fun () -> equal string "x" (colored ~background:true Red "x")));
      as_tinygrad ~key:[ "seconds"; "w" ]
        "time_to_str writes a duration as tinygrad does" "durations.golden"
        (fun cell ->
          equal string (cell "time_to_str")
            (time_to_str
               ~w:(int_of_string (cell "w"))
               (float_of_string (cell "seconds"))));
      test "time_to_str is 8 columns wide by default" (fun () ->
          equal string "  500.00ms" (time_to_str 0.5));
      as_tinygrad "size_to_str writes a size as tinygrad does" "sizes.golden"
        (fun cell ->
          equal string (cell "size_to_str")
            (size_to_str (int_of_string (cell "bytes"))));
      as_tinygrad "ansistrip strips as tinygrad does" "text.golden" (fun cell ->
          equal string
            (literal (cell "ansistrip"))
            (ansistrip (text_input cell)));
      as_tinygrad "ansilen counts as tinygrad does" "text.golden" (fun cell ->
          equal int (int_of_string (cell "ansilen")) (ansilen (text_input cell)));
      as_tinygrad "to_function_name names as tinygrad does" "text.golden"
        (fun cell ->
          equal string (cell "to_function_name")
            (to_function_name (text_input cell)));
      as_tinygrad "strip_parens strips as tinygrad does" "text.golden"
        (fun cell ->
          equal string
            (literal (cell "strip_parens"))
            (strip_parens (text_input cell)));
      prop "ansistrip undoes colored" (Gen.pair gen_color without_escape)
        (fun ((color, _), s) ->
          with_color (fun () -> equal string s (ansistrip (colored color s))));
      prop "ansilen of text without escapes counts its characters"
        (Gen.list Gen.uchar
        |> Gen.map (fun uchars ->
            (List.length uchars, strip_escapes (utf_8 uchars)))
        |> Gen.with_pp (fun ppf (_, s) -> Format.fprintf ppf "%S" s))
        (fun (characters, s) -> equal int characters (ansilen s));
      prop "ansipad appends the spaces that reach the width"
        (Gen.pair (Gen.string_of ascii_letter) (Gen.int_range (-2) 12))
        (fun (s, w) ->
          equal string
            (s ^ String.make (max 0 (w - String.length s)) ' ')
            (ansipad s w));
      test "ansipad does not count escape sequences" (fun () ->
          with_color (fun () ->
              equal string
                (colored Red "ab" ^ "  ")
                (ansipad (colored Red "ab") 4)));
      prop "to_function_name makes a string of letters, digits and underscores"
        gen_text (fun s ->
          satisfies ~claim:"letters, digits and underscores" string
            (String.for_all identifier)
            (to_function_name s));
      prop "to_function_name keeps an identifier" gen_text
        (Law.idempotent string to_function_name);
      prop "strip_parens removes parentheses around a balanced expression"
        (Gen.list
           (Gen.of_list ~pp:Format.pp_print_string
              [ "a"; "+"; "(a)"; "(a+(b))" ]))
        (fun parts ->
          let s = String.concat "" parts in
          equal string s (strip_parens ("(" ^ s ^ ")")));
      cases
        ~name:(fun (st, cnt, _) -> Printf.sprintf "%d %s" cnt st)
        "pluralize counts"
        [
          ("kernel", 1, "1 kernel");
          ("kernel", 0, "0 kernels");
          ("kernel", 2, "2 kernels");
          ("kernel", -1, "-1 kernels");
        ]
        (fun (st, cnt, expected) -> equal string expected (pluralize st cnt));
    ]

(* Disk cache *)

let fresh_table =
  let n = ref 0 in
  fun () ->
    incr n;
    Printf.sprintf "table%d" !n

let rec files dir =
  List.concat_map
    (fun name ->
      let path = Filename.concat dir name in
      if Sys.is_directory path then files path else [ path ])
    (Array.to_list (Sys.readdir dir))

(* Runs [damage] on every file of the cache when it holds one entry, then reads
   the entry back. *)
let damaged damage =
  Diskcache.clear ();
  let table = fresh_table () in
  Diskcache.put ~table "k" (String.make 4096 'v');
  List.iter damage (files cachedb);
  (match Diskcache.get ~table "k" with
  | exception Failure _ -> ()
  | None -> fail "get is None"
  | Some v -> failf "get is a value of %d bytes" (String.length v));
  Diskcache.clear ()

(* The directory of [table]'s entries, found as the one its first entry
   creates. *)
let table_directory table =
  let before = files cachedb in
  Diskcache.put ~table "k" "v";
  let entry = List.find (fun f -> not (List.mem f before)) (files cachedb) in
  Filename.dirname entry

(* Files the cache did not write, some named like its own: a table directory
   holding a file whose name is no digest, directories whose names are not a
   digest and a version, and a plain file where a table directory would be. *)
let clear_keeps_foreign_files () =
  let write path =
    Out_channel.with_open_bin path (fun oc -> output_string oc "foreign")
  in
  let under dir name =
    let dir = Filename.concat cachedb dir in
    if not (Sys.file_exists dir) then Sys.mkdir dir 0o755;
    Filename.concat dir name
  in
  let digest = String.make 32 'a' in
  let lookalike = table_directory (fresh_table ()) in
  Diskcache.clear ();
  if Sys.file_exists lookalike then Sys.rmdir lookalike;
  let live = table_directory (fresh_table ()) in
  let foreign =
    [
      Filename.concat cachedb "foreign.txt";
      under "foreign" digest;
      under (digest ^ "_") digest;
      under (digest ^ "x1") digest;
      under (String.make 32 'z' ^ "_1") digest;
      Filename.concat live (String.make 32 'z');
      Filename.concat live (digest ^ "x");
      lookalike;
    ]
  in
  List.iter write foreign;
  Diskcache.clear ();
  equal (list bool)
    (List.map (fun _ -> true) foreign)
    (List.map Sys.file_exists foreign);
  List.iter Sys.remove foreign

let truncate file = Unix.truncate file ((Unix.stat file).st_size / 2)

let overwrite file =
  Out_channel.with_open_bin file (fun oc -> output_string oc "not an entry")

let writers_never_tear () =
  let table = fresh_table () and size = 1 lsl 16 in
  let whole v =
    String.length v = size
    && String.for_all (Char.equal v.[0]) v
    && String.contains "abcd" v.[0]
  in
  let reads = Array.make 4 [] in
  let writer i () =
    let payload = String.make size "abcd".[i] in
    reads.(i) <-
      List.init 20 (fun _ ->
          Diskcache.put ~table "k" payload;
          Diskcache.get ~table "k")
  in
  List.iter Domain.join (List.init 4 (fun i -> Domain.spawn (writer i)));
  Array.iter
    (List.iter
       (satisfies ~claim:"one writer's whole value" (option string)
          (Option.fold ~none:false ~some:whole)))
    reads

let processes_never_tear () =
  let table = fresh_table () in
  let writers =
    List.map (fun fill -> start "write" [ table; fill ]) [ "a"; "b"; "c"; "d" ]
  in
  List.iter (fun w -> ignore (succeeds (finish w))) writers;
  let entry = require_some (Diskcache.get ~table "k") in
  equal int (1 lsl 20) (String.length entry);
  is_true ~msg:"one writer's whole value"
    (String.for_all (Char.equal entry.[0]) entry)

let stays_inside_cachedb () =
  Diskcache.put ~table:(fresh_table ()) "k" "v";
  let around () =
    List.sort String.compare
      (Array.to_list (Sys.readdir (Filename.dirname cachedb)))
  in
  let before = around () in
  Diskcache.put ~table:"../outside" "../../outside" "v";
  equal (option string) (Some "v")
    (Diskcache.get ~table:"../outside" "../../outside");
  equal (list string) before (around ())

(* The model of the cache: a table of the entries, reset when the cache is
   cleared. *)
let entries : (string * string, string) Hashtbl.t = Hashtbl.create 16
let cache = abstract "cache"

let tables =
  Gen.of_list ~pp:Format.pp_print_string
    [ "a"; "A"; "b"; "test_gfx1010:xnack-"; "long" ^ String.make 300 't' ]

let keys = Gen.of_list ~pp:Format.pp_print_string [ "k"; "K"; ""; "k/../k" ]
let values = Gen.string_of ~size:(Gen.int_range 0 8) Gen.char
let disabled f = context [ B (cachelevel, 0) ] f

let model =
  [
    command "clear"
      (Gen.unit @-> makes cache)
      (fun () -> Hashtbl.reset entries)
      Diskcache.clear;
    command "put"
      (tables @-> keys @-> values @-> cache ^-> returns unit)
      (fun table key v () -> Hashtbl.replace entries (table, key) v)
      (fun table key v () -> Diskcache.put ~table key v);
    command "get"
      (tables @-> keys @-> cache ^-> returns (option string))
      (fun table key () -> Hashtbl.find_opt entries (table, key))
      (fun table key () -> Diskcache.get ~table key);
    command "clear again"
      (cache ^-> returns unit)
      (fun () -> Hashtbl.reset entries)
      Diskcache.clear;
    command "put while disabled"
      (tables @-> keys @-> values @-> cache ^-> returns unit)
      (fun _ _ _ () -> ())
      (fun table key v () -> disabled (fun () -> Diskcache.put ~table key v));
    command "get while disabled"
      (tables @-> keys @-> cache ^-> returns (option string))
      (fun _ _ () -> None)
      (fun table key () -> disabled (fun () -> Diskcache.get ~table key));
  ]

let diskcache =
  match Sys.getenv_opt "CACHEDB" with
  | None ->
      group "Diskcache"
        [
          test "runs under CACHEDB, since it clears the cache" (fun () ->
              skip ~reason:"CACHEDB is unset" ());
        ]
  | Some _ ->
      group "Diskcache"
        [
          stateful ~count:40 "behaves as a table of entries per table" model;
          test "get is None for a table never written" (fun () ->
              equal (option string) None
                (Diskcache.get ~table:(fresh_table ()) "k"));
          test "put replaces the value of a key" (fun () ->
              let table = fresh_table () in
              Diskcache.put ~table "hello" "world";
              Diskcache.put ~table "hello" "world2";
              equal (option string) (Some "world2")
                (Diskcache.get ~table "hello"));
          prop ~count:30 "get reads back any key and value put"
            (Gen.pair Gen.string Gen.string) (fun (key, v) ->
              let table = fresh_table () in
              Diskcache.put ~table key v;
              equal (option string) (Some v) (Diskcache.get ~table key));
          test "a megabyte value reads back" (fun () ->
              let table = fresh_table ()
              and v = String.init (1 lsl 20) (fun i -> Char.chr (i land 255)) in
              Diskcache.put ~table "k" v;
              equal (option string) (Some v) (Diskcache.get ~table "k"));
          test "keys and tables that look like paths stay inside cachedb"
            stays_inside_cachedb;
          test "clear removes the entries of every table" (fun () ->
              let a = fresh_table () and b = fresh_table () in
              Diskcache.put ~table:a "k" "v";
              Diskcache.put ~table:b "k" "v";
              Diskcache.clear ();
              Diskcache.clear ();
              equal
                (pair (option string) (option string))
                (None, None)
                (Diskcache.get ~table:a "k", Diskcache.get ~table:b "k"));
          test "clear keeps the files that are not entries"
            clear_keeps_foreign_files;
          cases ~name:(Printf.sprintf "cachelevel=%d")
            "a disabled cache neither reads nor writes" [ 0; -1 ] (fun level ->
              let table = fresh_table () in
              Diskcache.put ~table "kept" "v";
              context
                [ B (cachelevel, level) ]
                (fun () ->
                  Diskcache.put ~table "dropped" "v";
                  equal (option string) None (Diskcache.get ~table "kept"));
              equal
                (pair (option string) (option string))
                (Some "v", None)
                (Diskcache.get ~table "kept", Diskcache.get ~table "dropped"));
          test "an entry put by another process reads back" (fun () ->
              let table = fresh_table () in
              ignore (succeeds (child "put" [ table; "k"; "remote" ]));
              equal (option string) (Some "remote") (Diskcache.get ~table "k"));
          test "another process reads an entry put back" (fun () ->
              let table = fresh_table () in
              Diskcache.put ~table "k" "getme";
              equal string "getme" (succeeds (child "get" [ table; "k" ])));
          test "writers on several domains never tear an entry"
            writers_never_tear;
          test "writers in several processes never tear an entry"
            processes_never_tear;
          test "get fails on a truncated entry" (fun () -> damaged truncate);
          test "get fails on an entry that is no entry" (fun () ->
              damaged overwrite);
        ]

(* Programs *)

let fails_with message f =
  raises_match (function Failure m -> m = message | _ -> false) f

(* The object of [src], compiled by the first of [compilers] that can, if
   any. *)
let compiled ?(flags = "") compilers src =
  let obj = Filename.temp_file "tolk" ".o" in
  List.find_map
    (fun cc ->
      match
        system ~input:src (Printf.sprintf "%s %s -c -x c - -o %s" cc flags obj)
      with
      | _ -> Some (In_channel.with_open_bin obj In_channel.input_all)
      | exception Failure _ -> None)
    compilers

let last_line s = List.hd (List.rev (String.split_on_char '\n' (String.trim s)))

let programs =
  group "programs"
    [
      test "system is the output of a command" (fun () ->
          equal string "hello" (system "echo hello"));
      test "system splits a command at runs of white space" (fun () ->
          equal string "a b" (system "echo  a\t b"));
      test "system gives the input on standard input" (fun () ->
          equal string "abc" (system ~input:"abc" "cat"));
      test "system is standard output and standard error, stripped" (fun () ->
          equal string "out\nerr"
            (system ~input:"echo; echo out; echo err 1>&2; echo" "sh"));
      test "system fails with the exit code and the output" (fun () ->
          fails_with "system: 'sh' failed with exit code 3\nwhy" (fun () ->
              system ~input:"echo why; exit 3" "sh"));
      test "system fails on a program that does not exist" (fun () ->
          raises_match
            (function Failure _ -> true | _ -> false)
            (fun () -> system "tolk-no-such-program"));
      test "system reports its output's size and time from DEBUG=1" (fun () ->
          ignore (output ());
          ignore (context [ B (debug, 1) ] (fun () -> system "echo hello"));
          contains ~sub:"system: 'echo hello' returned 5 bytes in" (output ()));
      test "cpu_objdump prints the instructions of an object" (fun () ->
          match
            compiled [ "cc"; "clang" ] "int answer(void) { return 42; }"
          with
          | None -> skip ~reason:"no C compiler" ()
          | Some lib ->
              ignore (output ());
              cpu_objdump lib;
              contains ~sub:"Disassembly of section" (output ()));
      test
        "amdgpu_disassemble prints a code object up to its end, without padding"
        (fun () ->
          match
            compiled ~flags:"--target=amdgcn-amd-amdhsa -mcpu=gfx1100 -nogpulib"
              [
                "/opt/homebrew/opt/llvm/bin/clang";
                "/opt/rocm/llvm/bin/clang";
                "clang";
              ]
              "__attribute__((amdgpu_kernel)) void k(void) {}"
          with
          | None -> skip ~reason:"no clang for AMD GPUs" ()
          | Some lib -> (
              match find_llvm_objdump () with
              | exception Failure why -> skip ~reason:why ()
              | _ ->
                  ignore (output ());
                  amdgpu_disassemble lib;
                  contains ~sub:"s_endpgm" (last_line (output ()))));
    ]

let () =
  match Sys.getenv_opt role with
  | Some r -> play r
  | None ->
      exit
        (run "Tolk.Helpers"
           [
             environment;
             declaration;
             contexts;
             cache_keys;
             library_settings;
             startup;
             targets;
             integers;
             selection;
             terminal_text;
             diskcache;
             programs;
           ])
