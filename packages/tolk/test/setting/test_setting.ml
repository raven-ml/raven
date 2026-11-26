open Windtrap
open Tolk

let context = Setting.context
let shaping = Setting.shaping
let value = Setting.value
let refuses f = raises_match (fun e -> Exn.invalid_arg e) f

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

(* A library setting with its testable and a value. *)
type row = D : 'a Setting.t * 'a testable * 'a -> row

let key (D (v, _, _)) = Setting.key v
let shown (D (_, t, x)) = Testable.to_string t x
let current (D (v, t, _)) = Testable.to_string t (value v)

(* Each library setting with the default its documentation states, but
   [parallel], whose default depends on the host. *)
let defaults =
  [
    D (Setting.debug, int, 0);
    D (Setting.beam, int, 0);
    D (Setting.jitbeam, option int, None);
    D (Setting.noopt, bool, false);
    D (Setting.no_color, bool, false);
    D (Setting.use_tc, int, 1);
    D (Setting.tc_select, int, -1);
    D (Setting.tc_opt, int, 2);
    D (Setting.tc_min_globals, int, 0);
    D (Setting.stage, bool, true);
    D (Setting.transcendental, int, 1);
    D (Setting.split_reduceop, bool, true);
    D (Setting.no_memory_planner, bool, false);
    D (Setting.cachelevel, int, 2);
    D (Setting.cachesize, int, 5368709120);
    D (Setting.ignore_beam_cache, bool, false);
    D (Setting.disable_fast_idiv, bool, true);
    D (Setting.max_kernel_buffers, int, 0);
    D (Setting.emulated_dtypes, list string, []);
    D (Setting.default_float, string, "float32");
    D (Setting.default_int, string, "int32");
    D (Setting.spec, int, 1);
    D (Setting.check_oob, bool, false);
    D (Setting.tuple_order, bool, true);
    D (Setting.ccache, bool, true);
    D (Setting.allow_tf32, bool, false);
    D (Setting.scache, int, 2);
    D (Setting.disallow_broadcast, bool, false);
    D (Setting.reduceop_split_threshold, int, 32768);
    D (Setting.reduceop_split_size, int, 22);
    D (Setting.mv, bool, true);
    D (Setting.dmc, bool, false);
    D (Setting.expand_ssa, bool, false);
    D (Setting.beam_padto, bool, false);
    D (Setting.beam_uops_max, int, 3000);
    D (Setting.beam_upcast_max, int, 256);
    D (Setting.beam_local_max, int, 1024);
    D (Setting.beam_min_progress, float_exact, 0.01);
    D (Setting.beam_estimate, bool, true);
    D (Setting.beam_strict_mode, bool, false);
    D (Setting.beam_log_surpass_max, bool, false);
    D (Setting.beam_debug, int, 0);
    D (Setting.cc, string, "clang");
    D (Setting.cuda_path, string, "");
    D (Setting.rocm_path, string, "/opt/rocm");
    D (Setting.assert_compile, bool, false);
    D (Setting.rewrite_stack_limit, int, 250000);
    D (Setting.debug_linearize, bool, false);
    D (Setting.dbgtv, string, "");
  ]

(* Children

   What a program reads when it starts is observed in a child: this suite
   started again with [role] set, which plays that role and exits. *)

let role = "TOLK_TEST_SETTING_ROLE"

type ended = { status : Unix.process_status; out : string; err : string }

(* [child ~env part args] plays [part] with [args] in an environment where each
   [(name, Some v)] of [env] is set and each [(name, None)] unset. *)
let child ?(env = []) part args =
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
  let _, status = Unix.waitpid [] pid in
  (* in text mode: on Windows a child's lines end in CRLF *)
  let read file = In_channel.with_open_text file In_channel.input_all in
  { status; out = read out; err = read err }

let status =
  Testable.make
    ~pp:(fun ppf s ->
      match s with
      | Unix.WEXITED n -> Format.fprintf ppf "exited %d" n
      | Unix.WSIGNALED n -> Format.fprintf ppf "killed by signal %d" n
      | Unix.WSTOPPED n -> Format.fprintf ppf "stopped by signal %d" n)
    ~equal:( = )

(* The library settings' values a child started with [env] holds, each as
   [defaults] prints it, and [PARALLEL]'s. *)
let values_at_start env =
  let ended = child ~env "values" [] in
  equal ~msg:ended.err status (Unix.WEXITED 0) ended.status;
  let entry line =
    match String.index_opt line '\t' with
    | Some i ->
        Some
          ( String.sub line 0 i,
            String.sub line (i + 1) (String.length line - i - 1) )
    | None -> None
  in
  List.filter_map entry (String.split_on_char '\n' ended.out)

let read_at_start env expected =
  let read = values_at_start env in
  equal
    (list (pair string (option string)))
    (List.map (fun d -> (key d, Some (shown d))) expected)
    (List.map (fun d -> (key d, List.assoc_opt (key d) read)) expected)

(* A program that raises at start-up exits 2, as OCaml's runtime does on an
   uncaught exception. *)
let fails_at_start ~sub env =
  let ended = child ~env "values" [] in
  equal status (Unix.WEXITED 2) ended.status;
  contains ~sub:"Invalid_argument" ended.err;
  contains ~sub ended.err

(* The values of a child started with every library variable unset. *)
let at_defaults =
  lazy
    (values_at_start
       (("PARALLEL", None) :: List.map (fun d -> (key d, None)) defaults))

(* Settings of each kind, declared from a variable by name, each value shown as
   the golden writes it. *)
let kinds =
  [
    ("int", fun key -> string_of_int (Setting.value (int_setting key)));
    ( "float",
      fun key ->
        Format.asprintf "%a" Dtype.pp_const
          (`Float (Setting.value (float_setting key))) );
    ("string", fun key -> Setting.value (Setting.string ~reach:Process key "d"));
    ( "int_option",
      fun key ->
        match Setting.value (Setting.int_option ~reach:Process key) with
        | Some n -> string_of_int n
        | None -> "None" );
  ]

let empty = "TOLK_TEST_EMPTY_VARIABLE"

(* [read_empty kind] is the value a [kind] setting reads from an empty variable,
   shown, or why it refuses it. The variable is bound in a child's environment,
   since on Windows a process that sets a variable to the empty string unbinds
   it. *)
let read_empty kind =
  let ended = child ~env:[ (empty, Some "") ] "declare" [ kind ] in
  if ended.status = Unix.WEXITED 0 then Ok ended.out else Error ended.err

let refuses_empty kind =
  contains ~sub:"Invalid_argument"
    (require_error ~pp:Format.pp_print_string (read_empty kind))

let play = function
  | "values" ->
      List.iter (fun d -> Printf.printf "%s\t%s\n" (key d) (current d)) defaults;
      Printf.printf "PARALLEL\t%d\n" (value Setting.parallel)
  | "declare" -> print_string (List.assoc Sys.argv.(1) kinds empty)
  | role -> failwith ("unknown role " ^ role)

(* The environment *)

let environment =
  group "environment"
    [
      test "DEV is read by no setting: a program starts whatever it holds"
        (fun () ->
          read_at_start
            [ ("DEV", Some "PCI+NV+CUDA"); ("DEBUG", Some "3") ]
            [ D (Setting.debug, int, 3) ]);
    ]

let initial_value =
  group "initial value"
    [
      test "is read from the variable once, when the setting is declared"
        (fun () ->
          let key = fresh () in
          setenv key (Some "1");
          let v = int_setting key in
          setenv key (Some "2");
          equal int 1 (Setting.value v));
    ]

(* Settings *)

let constructors =
  group "constructors"
    [
      test
        "raise on a name declared before, whatever the setting's type and reach"
        (fun () ->
          let key = fresh () in
          ignore (Setting.int ~reach:Output key 0);
          refuses (fun () -> Setting.int ~reach:Output key 0);
          refuses (fun () -> Setting.bool ~reach:Process key false);
          refuses (fun () -> Setting.float ~reach:Output key 0.);
          refuses (fun () -> Setting.string ~reach:Process key "");
          refuses (fun () -> Setting.int_option ~reach:Output key));
      test "raise on the name of a library setting" (fun () ->
          refuses (fun () -> Setting.int ~reach:Process "DEBUG" 0));
      test "raise on a name declared inside a context that has ended" (fun () ->
          let key = fresh () in
          context [ B (Setting.debug, 1) ] (fun () -> ignore (int_setting key));
          refuses (fun () -> int_setting key));
      (* Python's int() and float() also read Unicode digits and white space; a
         setting refuses them, as the constructors state, rather than carry
         Unicode's tables of digits and spaces. *)
      cases "raise on non-ASCII digits and white space"
        ~id:(Printf.sprintf "%S")
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
      cases "raise on an integer out of int's range" ~id:Fun.id
        [
          (* max_int + 1 and min_int - 1 *)
          "4611686018427387904";
          "-4611686018427387905";
          "9223372036854775807";
        ] (fun value ->
          refuses (fun () -> declared_with (Some value) int_setting);
          refuses (fun () ->
              declared_with (Some value) (fun key ->
                  Setting.bool ~reach:Process key false));
          refuses (fun () ->
              declared_with (Some value) (fun key ->
                  Setting.int_option ~reach:Process key)));
      test "declare nothing when they raise" (fun () ->
          let key = fresh () in
          setenv key (Some "abc");
          refuses (fun () -> int_setting key);
          setenv key (Some "4");
          equal int 4 (Setting.value (int_setting key)));
    ]

(* The golden's text for an exception tinygrad raised. *)
let raised cell =
  List.mem cell [ "ValueError"; "ZeroDivisionError"; "RuntimeError" ]

(* Goldens hold arbitrary text as OCaml string literals. *)
let literal cell = Scanf.sscanf cell "%S%!" Fun.id

(* [reads ~kind make w of_cell column cell] checks the row [cell] of env.golden:
   a setting made by [make] from a variable holding the row's input holds the
   value of [column], or refuses the input where the golden raised. *)
let reads ~kind make w of_cell column cell =
  let input = literal (cell "input") and expected = cell column in
  match (input, raised expected) with
  | "", true -> refuses_empty kind
  | "", false ->
      equal string expected
        (require_ok ~pp:Format.pp_print_string (read_empty kind))
  | _, true -> refuses (fun () -> declared_with (Some input) make)
  | _, false -> equal w (of_cell expected) (declared_with (Some input) make)

let int_ =
  group "int"
    [
      group "reads a variable as its golden states"
        [
          Golden.cases "env.golden"
            (reads ~kind:"int" int_setting int int_of_string "int");
        ];
      test "starts from its default when its variable is unset" (fun () ->
          equal int 5
            (declared_with None (fun key -> Setting.int ~reach:Process key 5)));
    ]

let bool_ =
  group "bool"
    [
      cases "is true iff its variable holds a nonzero integer"
        ~id:(fun (value, _) -> Printf.sprintf "%S" value)
        [
          ("0", false);
          ("1", true);
          ("2", true);
          ("-1", true);
          (" 0 ", false);
          ("00", false);
          (" -0 ", false);
          ("+1", true);
          ("1_0", true);
        ]
        (fun (value, on) ->
          equal bool on
            (declared_with (Some value) (fun key ->
                 Setting.bool ~reach:Process key (not on))));
      cases "starts from its default when its variable is unset"
        ~id:string_of_bool [ true; false ] (fun default ->
          equal bool default
            (declared_with None (fun key ->
                 Setting.bool ~reach:Process key default)));
      test "refuses a variable that is no integer" (fun () ->
          refuses (fun () ->
              declared_with (Some "true") (fun key ->
                  Setting.bool ~reach:Process key false)));
    ]

(* The float a decimal reads as: the nearest, but on Windows, whose C library
   may read a decimal that no float equals as a neighbour of the nearest, that
   or a neighbour. *)
let as_read =
  let near x y =
    Float.equal x y
    || Float.equal (Float.succ x) y
    || Float.equal (Float.pred x) y
  in
  if Sys.win32 then Testable.make ~pp:(Testable.pp float_exact) ~equal:near
  else float_exact

let float_ =
  group "float"
    [
      group "reads a variable as its golden states"
        [
          Golden.cases "env.golden"
            (reads ~kind:"float" float_setting as_read float_of_string "float");
        ];
      cases "reads a decimal that a float equals as that float"
        ~id:(fun (input, _) -> Printf.sprintf "%S" input)
        [
          ("0.5", 0.5);
          ("-0.0", -0.);
          ("10.5", 10.5);
          ("1e3", 1000.);
          ("2.5e-1", 0.25);
          ("0.125e3", 125.);
          ("1e22", 1e22);
          ("9007199254740992", 9007199254740992.);
        ]
        (fun (input, x) ->
          equal float_exact x (declared_with (Some input) float_setting));
      test "starts from its default when its variable is unset" (fun () ->
          equal float_exact 0.5
            (declared_with None (fun key ->
                 Setting.float ~reach:Process key 0.5)));
    ]

let string_ =
  let word key = Setting.string ~reach:Process key "d" in
  group "string"
    [
      test "is its variable as written" (fun () ->
          equal string " x " (declared_with (Some " x ") word));
      test "is \"\" for a variable set to the empty string" (fun () ->
          equal string ""
            (require_ok ~pp:Format.pp_print_string (read_empty "string")));
      test "starts from its default when its variable is unset" (fun () ->
          equal string "d" (declared_with None word));
    ]

let int_option =
  let opt key = Setting.int_option ~reach:Process key in
  group "int_option"
    [
      test "is Some n for a variable that holds the integer n" (fun () ->
          equal (option int) (Some (-3)) (declared_with (Some " -3 ") opt));
      test "is None for an unset variable" (fun () ->
          equal (option int) None (declared_with None opt));
      test "refuses an empty variable" (fun () -> refuses_empty "int_option");
    ]

let key_ =
  group "key"
    [
      test "is the name of the setting's variable" (fun () ->
          let key = fresh () in
          equal string key (Setting.key (Setting.bool ~reach:Process key false)));
    ]

(* Contexts *)

let level = Setting.int ~reach:Output (fresh ()) 0
let label = Setting.string ~reach:Output (fresh ()) "default"

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

let spin_until flag =
  while not (Atomic.get flag) do
    Domain.cpu_relax ()
  done

(* [f ()] on a domain spawned inside a context that binds [v] to [x], run once
   that context has ended. *)
let spawned_within v x f =
  let ended = Atomic.make false in
  let d =
    context
      [ B (v, x) ]
      (fun () ->
        Domain.spawn (fun () ->
            spin_until ended;
            f ()))
  in
  Atomic.set ended true;
  Domain.join d

let unseen_while_another_domain_binds () =
  let bound = Atomic.make false and released = Atomic.make false in
  let binder =
    Domain.spawn (fun () ->
        context
          [ B (level, 7) ]
          (fun () ->
            Atomic.set bound true;
            spin_until released))
  in
  spin_until bound;
  let seen = value level in
  Atomic.set released true;
  Domain.join binder;
  equal int 0 seen

let releases_bound_values () =
  let weak = Weak.create 1 in
  let bind () =
    let bound = String.make 16 'x' in
    Weak.set weak 0 (Some bound);
    context [ B (label, bound) ] (fun () -> equal string bound (value label))
  in
  bind ();
  Gc.full_major ();
  is_none ~pp:Format.pp_print_string (Weak.get weak 0)

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
      test
        "binds for the domains spawned while it runs, which keep the value \
         they started with" (fun () ->
          equal int 7 (spawned_within level 7 (fun () -> value level)));
      test "is not seen by the other domains" unseen_while_another_domain_binds;
      test "binds and restores on the domain that runs it" (fun () ->
          equal int 7
            (Domain.join
               (Domain.spawn (fun () ->
                    context [ B (level, 7) ] (fun () -> value level))));
          equal int 0 (value level));
      test "leaves the calling domain holding none of the bound values"
        releases_bound_values;
    ]

(* What the caches key on *)

let keyed key = List.assoc_opt key (shaping ())
let entries = list (pair string string)

(* The text [shaping] gives [v] bound to [x]. *)
let text v x = List.assoc (Setting.key v) (Setting.shaping_under [ B (v, x) ])

(* [shaping ()] read again after [f] is the list read before it. *)
let unchanged_by f =
  let before = shaping () in
  f ();
  satisfies ~claim:"the list read before" entries
    (fun l -> l == before)
    (shaping ())

let shaping_under =
  group "shaping_under"
    [
      test "is shaping in the context of its bindings" (fun () ->
          let v = Setting.int ~reach:Output (fresh ()) 1 in
          let p = Setting.int ~reach:Process (fresh ()) 1 in
          let bs = Setting.[ B (v, 4); B (p, 3); B (v, 5) ] in
          equal entries (context bs shaping) (Setting.shaping_under bs));
      test "binds nothing" (fun () ->
          let v = Setting.int ~reach:Output (fresh ()) 1 in
          unchanged_by (fun () -> ignore (Setting.shaping_under [ B (v, 4) ]));
          equal int 1 (Setting.value v));
    ]

let float_out = Setting.float ~reach:Output (fresh ()) 0.
let int_out = Setting.int ~reach:Output (fresh ()) 0
let int_option_out = Setting.int_option ~reach:Output (fresh ())
let bool_out = Setting.bool ~reach:Output (fresh ()) false
let string_out = Setting.string ~reach:Output (fresh ()) ""

(* [emulated_dtypes] is the setting of lists of strings; [text] shows it under a
   binding without binding it. *)
let names_out = Setting.emulated_dtypes

(* [x] and [y] have the same text only if [eq x y]. *)
let only_if_equal eq v (x, y) =
  let unequal = not (eq x y) in
  cover "unequal values" unequal;
  if unequal then not_equal string (text v x) (text v y)

let pp_pair t ppf (x, y) =
  Format.fprintf ppf "(%a, %a)" (Testable.pp t) x (Testable.pp t) y

(* Pairs of floats drawn as bits, the second often a neighbour of the first or
   its negation. *)
let float_pairs =
  let open Gen in
  let any = map Int64.float_of_bits int64 in
  (let* x = any in
   let+ y =
     one_of
       [
         constant (Float.succ x);
         constant (Float.pred x);
         constant (Float.neg x);
         any;
       ]
   in
   (x, y))
  |> with_pp (pp_pair float_exact)

let int_pairs =
  let open Gen in
  (let* x = int in
   let+ y = one_of [ constant (x + 1); constant (-x); int ] in
   (x, y))
  |> with_pp (pp_pair Windtrap.int)

(* Lists of names drawn from the bytes that separate or quote them. *)
let names_pairs =
  let open Gen in
  let name =
    string_of ~size:(int_range 0 3) (of_list [ 'a'; ','; '"'; '\\' ])
  in
  let names = list ~size:(int_range 0 3) name in
  pair names names |> with_pp (pp_pair Windtrap.(list string))

let printable s = String.for_all (fun c -> c >= ' ' && c <= '~') s

let in_printable_ascii v x =
  satisfies ~claim:"printable ASCII" string printable (text v x)

(* The fewest words one of ten calls of [f] allocates on the minor heap. *)
let words f =
  let fewest = ref max_int in
  for _ = 1 to 10 do
    let before = Gc.minor_words () in
    ignore (Sys.opaque_identity (f ()));
    fewest := min !fewest (int_of_float (Gc.minor_words () -. before))
  done;
  !fewest

let shaping_ =
  group "shaping"
    [
      test "holds a setting's current value on the calling domain" (fun () ->
          let key = fresh () in
          let v = Setting.int ~reach:Output key 1 in
          equal (option string)
            (Some (text v 4))
            (context [ B (v, 4) ] (fun () -> keyed key)));
      test "holds a setting declared after it was read" (fun () ->
          ignore (shaping ());
          let key = fresh () in
          let v = Setting.bool ~reach:Output key true in
          equal (option string) (Some (text v true)) (keyed key));
      test "leaves out a setting that reaches Process" (fun () ->
          let key = fresh () in
          ignore (Setting.bool ~reach:Process key false);
          equal (option string) None (keyed key));
      test "is sorted by name" (fun () ->
          let names = List.map fst (shaping ()) in
          equal (list string) (List.sort compare names) names);
      test "writes a number in hexadecimal" (fun () ->
          equal string "0x1.4p+1" (text float_out 2.5));
      test "writes a string as an OCaml string literal" (fun () ->
          equal string {|" a,\"b "|} (text string_out {| a,"b |}));
      test "writes a list's strings as OCaml string literals separated by ','"
        (fun () ->
          equal string {|"half","b,f"|} (text names_out [ "half"; "b,f" ]));
      test "writes None as \"\"" (fun () ->
          equal string "" (text int_option_out None));
      prop
        "gives two floats the same text only if they are equal, any two NaNs \
         counting as equal"
        ~examples:
          [
            (0.1, 0.10000000000000002);
            (max_float, infinity);
            (-.max_float, neg_infinity);
            (0., 5e-324);
            (Float.nan, infinity);
            (Float.nan, 0.);
            (-0., 0.);
            (Float.nan, Int64.float_of_bits 0x7ff8000000000001L);
            (Float.nan, -.Float.nan);
          ]
        float_pairs
        (fun (x, y) ->
          cover "neighbours" (Float.equal (Float.succ x) y);
          only_if_equal Float.equal float_out (x, y));
      prop "gives two integers the same text only if they are equal"
        ~examples:[ (max_int, min_int); (-1, 1) ]
        int_pairs
        (only_if_equal Int.equal int_out);
      prop "gives two optional integers the same text only if they are equal"
        ~examples:[ (None, Some 0) ]
        (Gen.pair (Gen.option Gen.int) (Gen.option Gen.int)
        |> Gen.with_pp (pp_pair (option int)))
        (only_if_equal (Option.equal Int.equal) int_option_out);
      test "gives true and false different texts" (fun () ->
          not_equal string (text bool_out true) (text bool_out false));
      prop "gives two strings the same text only if they are equal"
        ~examples:[ ("", {|""|}); ("a", {|"a"|}) ]
        (Gen.pair Gen.string Gen.string |> Gen.with_pp (pp_pair string))
        (only_if_equal String.equal string_out);
      prop "gives two lists of strings the same text only if they are equal"
        ~examples:
          [ ([ "half,bfloat16" ], [ "half"; "bfloat16" ]); ([], [ "" ]) ]
        names_pairs
        (only_if_equal (List.equal String.equal) names_out);
      prop "writes every text in printable ASCII"
        ~examples:[ ("a\nDEFAULT_INT=b", [ "\n" ]); ("\x00\xff", [ "\r" ]) ]
        (Gen.pair Gen.string (Gen.list Gen.string)
        |> Gen.with_pp (fun ppf (s, l) ->
            Format.fprintf ppf "(%S, %a)" s (Testable.pp (list string)) l))
        (fun (s, l) ->
          cover "a string with a byte that is no printable ASCII"
            (not (printable s));
          in_printable_ascii string_out s;
          in_printable_ascii names_out l);
      test "holds the values a domain spawned in a context started with"
        (fun () ->
          let key = fresh () in
          let v = Setting.int ~reach:Output key 1 in
          ignore (keyed key);
          equal (option string)
            (Some (text v 4))
            (spawned_within v 4 (fun () -> keyed key)));
      test "holds the restored values in a domain spawned after a context ends"
        (fun () ->
          let key = fresh () in
          let v = Setting.int ~reach:Output key 1 in
          context [ B (v, 4) ] (fun () -> ignore (keyed key));
          equal (option string)
            (Some (text v 1))
            (Domain.join (Domain.spawn (fun () -> keyed key))));
      test "is the same list across a context that binds no setting of Output"
        (fun () ->
          unchanged_by (fun () -> context [ B (Setting.debug, 3) ] ignore));
      test "is the same list across the declaration of a setting of Process"
        (fun () ->
          unchanged_by (fun () ->
              ignore (Setting.int ~reach:Process (fresh ()) 0)));
      test "is the same list while another domain binds a setting" (fun () ->
          let v = Setting.int ~reach:Output (fresh ()) 1 in
          unchanged_by (fun () ->
              ignore
                (Domain.join
                   (Domain.spawn (fun () -> context [ B (v, 2) ] shaping)))));
      test "is the same list inside a context that binds a setting to its value"
        (fun () ->
          let v = Setting.int ~reach:Output (fresh ()) 1 in
          let before = shaping () in
          satisfies ~claim:"the list read before" entries
            (fun l -> l == before)
            (context [ B (v, 1) ] shaping));
      test "is the list a context found once it returns" (fun () ->
          let v = Setting.int ~reach:Output (fresh ()) 1 in
          unchanged_by (fun () ->
              context [ B (v, 2) ] (fun () -> ignore (shaping ()))));
      test "is the list a context found once it raises" (fun () ->
          let v = Setting.int ~reach:Output (fresh ()) 1 in
          unchanged_by (fun () ->
              try context [ B (v, 2) ] (fun () -> raise Exit) with Exit -> ()));
      test "is the list nested contexts found once they end" (fun () ->
          let v = Setting.int ~reach:Output (fresh ()) 1 in
          unchanged_by (fun () ->
              context
                [ B (v, 2); B (Setting.beam, 3) ]
                (fun () -> context [ B (v, 1) ] (fun () -> ignore (shaping ())))));
      test "holds a setting declared in a context once the context ends"
        (fun () ->
          let key = fresh () in
          ignore (shaping ());
          let v =
            context [ B (level, 5) ] (fun () -> Setting.int ~reach:Output key 2)
          in
          equal (option string) (Some (text v 2)) (keyed key));
      test "allocates nothing while no setting changes" (fun () ->
          ignore (shaping ());
          equal int (words (fun () -> [])) (words shaping));
    ]

(* Tolk's settings *)

(* Each library setting whose documentation says it reaches Output, bound to a
   value other than its default. *)
let outputs =
  Setting.
    [
      B (allow_tf32, true);
      B (beam, 1);
      B (beam_estimate, false);
      B (beam_local_max, 512);
      B (beam_min_progress, 0.5);
      B (beam_padto, true);
      B (beam_uops_max, 100);
      B (beam_upcast_max, 16);
      B (default_float, "half");
      B (default_int, "long");
      B (disable_fast_idiv, false);
      B (dmc, true);
      B (emulated_dtypes, [ "half" ]);
      B (expand_ssa, true);
      B (jitbeam, Some 0);
      B (max_kernel_buffers, 8);
      B (mv, false);
      B (noopt, true);
      B (no_memory_planner, true);
      B (reduceop_split_size, 10);
      B (reduceop_split_threshold, 4);
      B (split_reduceop, false);
      B (stage, false);
      B (use_tc, 2);
      B (tc_min_globals, 1);
      B (tc_opt, 1);
      B (tc_select, 0);
      B (transcendental, 2);
      B (tuple_order, false);
    ]

let bound_key (Setting.B (v, _)) = Setting.key v

(* A caller's environment may hold the value bound: then the entry has no other
   value to change to. *)
let changes_its_entry (Setting.B (v, x)) =
  if Setting.value v = x then
    skip ~reason:(Setting.key v ^ " holds the value bound") ();
  let before = keyed (Setting.key v) in
  not_equal (option string) before
    (context [ B (v, x) ] (fun () -> keyed (Setting.key v)))

(* Each library setting's variable as its documentation names it. *)
let variables =
  Setting.
    [
      ("DEBUG", key debug);
      ("BEAM", key beam);
      ("JITBEAM", key jitbeam);
      ("NOOPT", key noopt);
      ("NO_COLOR", key no_color);
      ("TC", key use_tc);
      ("TC_SELECT", key tc_select);
      ("TC_OPT", key tc_opt);
      ("TC_MIN_GLOBALS", key tc_min_globals);
      ("STAGE", key stage);
      ("TRANSCENDENTAL", key transcendental);
      ("SPLIT_REDUCEOP", key split_reduceop);
      ("NO_MEMORY_PLANNER", key no_memory_planner);
      ("CACHELEVEL", key cachelevel);
      ("CACHESIZE", key cachesize);
      ("IGNORE_BEAM_CACHE", key ignore_beam_cache);
      ("DISABLE_FAST_IDIV", key disable_fast_idiv);
      ("MAX_KERNEL_BUFFERS", key max_kernel_buffers);
      ("EMULATED_DTYPES", key emulated_dtypes);
      ("DEFAULT_FLOAT", key default_float);
      ("DEFAULT_INT", key default_int);
      ("PARALLEL", key parallel);
      ("SPEC", key spec);
      ("CHECK_OOB", key check_oob);
      ("TUPLE_ORDER", key tuple_order);
      ("CCACHE", key ccache);
      ("ALLOW_TF32", key allow_tf32);
      ("SCACHE", key scache);
      ("DISALLOW_BROADCAST", key disallow_broadcast);
      ("REDUCEOP_SPLIT_THRESHOLD", key reduceop_split_threshold);
      ("REDUCEOP_SPLIT_SIZE", key reduceop_split_size);
      ("MV", key mv);
      ("DMC", key dmc);
      ("EXPAND_SSA", key expand_ssa);
      ("BEAM_PADTO", key beam_padto);
      ("BEAM_UOPS_MAX", key beam_uops_max);
      ("BEAM_UPCAST_MAX", key beam_upcast_max);
      ("BEAM_LOCAL_MAX", key beam_local_max);
      ("BEAM_MIN_PROGRESS", key beam_min_progress);
      ("BEAM_ESTIMATE", key beam_estimate);
      ("BEAM_STRICT_MODE", key beam_strict_mode);
      ("BEAM_LOG_SURPASS_MAX", key beam_log_surpass_max);
      ("BEAM_DEBUG", key beam_debug);
      ("CC", key cc);
      ("CUDA_PATH", key cuda_path);
      ("ROCM_PATH", key rocm_path);
      ("ASSERT_COMPILE", key assert_compile);
      ("REWRITE_STACK_LIMIT", key rewrite_stack_limit);
      ("DEBUG_LINEARIZE", key debug_linearize);
      ("DBGTV", key dbgtv);
    ]

let library_settings =
  group "tolk's settings"
    [
      cases "read the variable their documentation names" ~id:fst variables
        (fun (documented, key) -> equal string documented key);
      cases
        "hold the default their documentation states when their variable is \
         unset"
        ~id:key defaults (fun d ->
          equal (option string)
            (Some (shown d))
            (List.assoc_opt (key d) (Lazy.force at_defaults)));
      test
        "parallel is between 1 and the domains the runtime recommends when \
         PARALLEL is unset" (fun () ->
          let n =
            int_of_string
              (require_some
                 (List.assoc_opt "PARALLEL" (Lazy.force at_defaults)))
          in
          at_least int ~than:1 n;
          at_most int ~than:(Domain.recommended_domain_count ()) n);
      test "those that reach Output are the library's entries of shaping"
        (fun () ->
          let library k = not (String.starts_with ~prefix:"TOLK_TEST_" k) in
          equal (list string)
            (List.sort compare (List.map bound_key outputs))
            (List.filter library (List.map fst (shaping ()))));
      cases
        "those that reach Output change their entry of shaping as they change"
        ~id:bound_key outputs changes_its_entry;
      test "read their variables when the program starts" (fun () ->
          read_at_start
            [
              ("DEBUG", Some "3");
              ("NOOPT", Some "2");
              ("CCACHE", Some "0");
              ("TC", Some "2");
              ("DEFAULT_FLOAT", Some "half");
            ]
            [
              D (Setting.debug, int, 3);
              D (Setting.noopt, bool, true);
              D (Setting.ccache, bool, false);
              D (Setting.use_tc, int, 2);
              D (Setting.default_float, string, "half");
            ]);
      test "parallel reads PARALLEL when the program starts" (fun () ->
          equal (option string) (Some "0")
            (List.assoc_opt "PARALLEL"
               (values_at_start [ ("PARALLEL", Some "0") ])));
      test "emulated_dtypes reads names separated by ',', dropping empty ones"
        (fun () ->
          read_at_start
            [ ("EMULATED_DTYPES", Some ",half,,bfloat16, ") ]
            [
              D
                ( Setting.emulated_dtypes,
                  list string,
                  [ "half"; "bfloat16"; " " ] );
            ]);
      test "a program fails when it starts if a setting's variable is refused"
        (fun () -> fails_at_start ~sub:"DEBUG" [ ("DEBUG", Some "two") ]);
      test "cachesize refuses a negative CACHESIZE when the program starts"
        (fun () -> fails_at_start ~sub:"CACHESIZE" [ ("CACHESIZE", Some "-1") ]);
    ]

let () =
  match Sys.getenv_opt role with
  | Some r -> play r
  | None ->
      exit
        (run "Tolk.Setting"
           [
             environment;
             initial_value;
             constructors;
             int_;
             bool_;
             float_;
             string_;
             int_option;
             key_;
             contexts;
             shaping_under;
             shaping_;
             library_settings;
           ])
