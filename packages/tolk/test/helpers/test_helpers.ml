open Windtrap
open Tolk
open Tolk.Helpers

(* Goldens hold arbitrary text as OCaml string literals. *)
let literal cell = Scanf.sscanf cell "%S%!" Fun.id

(* The golden's text for an exception its generator saw raised. *)
let raised cell =
  List.mem cell [ "ValueError"; "ZeroDivisionError"; "RuntimeError" ]

let refuses f = raises_match (fun e -> Exn.invalid_arg e) f

let unix_only f () =
  if Sys.win32 then skip ~reason:"no owners or modes" () else f ()

let read file = In_channel.with_open_bin file In_channel.input_all
let write file s = Out_channel.with_open_bin file (fun oc -> output_string oc s)

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

(* Processes

   What a process reads when it starts is observed in a child: this suite,
   started again with [role] set, plays that role and exits. *)

let role = "TOLK_TEST_HELPERS_ROLE"
let printed = function Some v -> v | None -> "<none>"
let of_printed = function "<none>" -> None | v -> Some v

let play = function
  | "cache" -> Printf.printf "cache_dir=%s\ncachedb=%s\n" cache_dir cachedb
  | "put" -> Diskcache.put ~table:Sys.argv.(1) Sys.argv.(2) Sys.argv.(3)
  | "get" ->
      print_string (printed (Diskcache.get ~table:Sys.argv.(1) Sys.argv.(2)))
  | "put then get" ->
      let table = Sys.argv.(1) in
      Diskcache.put ~table Sys.argv.(2) Sys.argv.(2);
      print_string (printed (Diskcache.get ~table Sys.argv.(3)))
  | "get, put, get" ->
      let table = Sys.argv.(1) and key = Sys.argv.(2) in
      let first = Diskcache.get ~table key in
      Diskcache.put ~table key Sys.argv.(3);
      let second = Diskcache.get ~table key in
      Printf.printf "%s\n%s" (printed first) (printed second)
  | "put and get after a clear" ->
      let table = Sys.argv.(1) in
      Diskcache.put ~table "k" "v";
      Diskcache.clear ();
      (match Diskcache.put ~table "k" "v" with
      | () -> print_endline "put"
      | exception Sys_error _ -> print_endline "Sys_error");
      print_string (printed (Diskcache.get ~table "k"))
  | "fill" ->
      for i = 1 to 4 do
        Diskcache.put ~table:Sys.argv.(1)
          (Sys.argv.(2) ^ string_of_int i)
          (String.make 4096 'v')
      done
  | "system" -> print_string (system "cat")
  | "find_llvm_objdump" -> (
      match find_llvm_objdump () with
      | path -> print_string path
      | exception Failure m -> print_string ("Failure: " ^ m))
  | role -> failwith ("unknown role " ^ role)

type ended = { status : Unix.process_status; out : string; err : string }

(* [start ~env ~stdin part args] starts playing [part] with [args], reading the
   file [stdin] if given, in an environment where each [(name, Some v)] of [env]
   is set and each [(name, None)] unset. *)
let start ?(env = []) ?stdin part args =
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
  let fd flags file = Unix.openfile file flags 0 in
  let out_fd = fd [ O_WRONLY ] out and err_fd = fd [ O_WRONLY ] err in
  let in_fd = Option.map (fd [ O_RDONLY ]) stdin in
  let argv = Array.of_list (Sys.executable_name :: args) in
  let pid =
    Unix.create_process_env Sys.executable_name argv environment
      (Option.value in_fd ~default:Unix.stdin)
      out_fd err_fd
  in
  List.iter Unix.close (out_fd :: err_fd :: Option.to_list in_fd);
  (pid, out, err)

let finish (pid, out, err) =
  let _, status = Unix.waitpid [] pid in
  (* in text mode: on Windows a child's lines end in CRLF *)
  let read file = In_channel.with_open_text file In_channel.input_all in
  { status; out = read out; err = read err }

let child ?env ?stdin part args = finish (start ?env ?stdin part args)

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

(* Asserts that a child started in [env] reads the values [expected]. *)
let read_at_start env expected =
  let binding line =
    match String.index_opt line '=' with
    | Some i ->
        Some
          ( String.sub line 0 i,
            String.sub line (i + 1) (String.length line - i - 1) )
    | None -> None
  in
  let read =
    List.filter_map binding
      (String.split_on_char '\n' (succeeds (child ~env "cache" [])))
  in
  equal
    (list (pair string (option string)))
    (List.map (fun (k, v) -> (k, Some v)) expected)
    (List.map (fun (k, _) -> (k, List.assoc_opt k read)) expected)

let macos () =
  (not Sys.win32)
  &&
  let uname = Unix.open_process_args_in "uname" [| "uname"; "-s" |] in
  Fun.protect
    ~finally:(fun () -> ignore (Unix.close_process_in uname))
    (fun () -> String.equal "Darwin" (String.trim (In_channel.input_all uname)))

(* [path dir names] is [names] joined under [dir] by the system's separator. *)
let path dir names = List.fold_left Filename.concat dir names

(* Target *)

let gen_field chars =
  Gen.string_of ~size:(Gen.int_range 0 4)
    (Gen.of_list ~pp:Format.pp_print_char chars)

let upper =
  List.init 26 (fun i -> Char.chr (Char.code 'A' + i)) @ [ '0'; '9'; '_' ]

let any_case = upper @ [ 'a'; 'x'; ','; '.'; '-' ]

(* Targets whose fields hold no separator, with their device and renderer in
   upper case, as of_string makes them. *)
let gen_target =
  (let open Gen in
   let+ device = gen_field upper
   and+ renderer = gen_field upper
   and+ arch = gen_field any_case
   and+ interface = gen_field any_case
   and+ indices = gen_field any_case in
   Target.{ device; renderer; arch; interface; indices })
  |> Gen.with_pp pp_target

(* Targets whose fields hold separators and pieces of each other, so that two of
   them often format alike. *)
let gen_split =
  (let open Gen in
   let few l = of_list ~pp:Format.pp_print_string l in
   let split = few [ ""; "a"; "b"; "ab"; "a:"; ":" ] in
   let+ renderer = few [ ""; "Y" ]
   and+ arch = few [ ""; "Z" ]
   and+ interface = split
   and+ indices = split in
   Target.{ device = "X"; renderer; arch; interface; indices })
  |> Gen.with_pp pp_target

let text t = Format.asprintf "%a" Target.pp t

(* The two targets that pp formats alike as [a:b+X]: those read from [a:b:+X]
   and from [a:b+X]. *)
let from_a_b_colon =
  Target.
    { device = "X"; renderer = ""; arch = ""; interface = "a:b"; indices = "" }

let from_a_b =
  Target.
    { device = "X"; renderer = ""; arch = ""; interface = "a"; indices = "b" }

let read_as_its_golden cell =
  let input = cell "input" in
  match Target.of_string input with
  | Error e when raised (cell "device") -> contains ~sub:(cell "names") e
  | Error e -> failf "the golden reads %S, of_string fails with %S" input e
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
        t

let written_as_its_golden cell =
  match Target.of_string (cell "input") with
  | Error _ -> ()
  | Ok t -> equal string (cell "to_string") (text t)

let key_tells_apart (t0, t1) =
  cover "one target" (t0 = t1);
  cover "two targets pp writes alike"
    (t0 <> t1 && String.equal (text t0) (text t1));
  if t0 = t1 then equal string (Target.key t0) (Target.key t1)
  else not_equal string (Target.key t0) (Target.key t1)

let targets =
  group "Target"
    [
      group "of_string reads each target as its golden states"
        [ Golden.cases "targets.golden" read_as_its_golden ];
      prop "of_string reads back what pp writes" gen_target
        (Law.round_trip target_w string text (fun s ->
             require_ok ~pp:Format.pp_print_string (Target.of_string s)));
      group "pp writes each target as its golden states"
        [ Golden.cases "targets.golden" written_as_its_golden ];
      test "pp writes a:b:+X's target as a:b+X, which reads back as a:b+X's"
        (fun () ->
          equal (pair target_w target_w) (from_a_b_colon, from_a_b)
            ( require_ok (Target.of_string "a:b:+X"),
              require_ok (Target.of_string (text from_a_b_colon)) ));
      prop "key tells targets apart, as pp does not"
        ~examples:[ (from_a_b_colon, from_a_b); (from_a_b, from_a_b) ]
        (Gen.pair gen_split gen_split)
        key_tells_apart;
    ]

(* Integers and lists *)

let gen_words =
  Gen.list (Gen.of_list ~pp:Format.pp_print_string [ "a"; "A"; "b"; "B"; "c" ])

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

let stable_argsort l =
  List.map snd
    (List.stable_sort
       (fun (a, _) (b, _) -> compare a b)
       (List.mapi (fun i x -> (x, i)) l))

(* Lists with equal elements, and permutations of [0 .. n-1]. *)
let gen_argsort =
  Gen.one_of
    [
      Gen.list (Gen.int_range 0 4);
      Gen.bind (Gen.int_range 0 8) (fun n ->
          Gen.permutation ~pp:Format.pp_print_int (List.init n Fun.id));
    ]

let is_permutation l = List.sort compare l = List.init (List.length l) Fun.id

(* The exact result of [f] on [x] and [y], if it fits an int. *)
let exact f x y =
  let r = f (Bigint.of_int x) (Bigint.of_int y) in
  if Bigint.fits_int r then Some (Bigint.to_int r) else None

let exact_floormod x y = exact (fun x y -> Bigint.(x - (fdiv x y * y))) x y
let exact_round_up x amt = exact (fun x amt -> Bigint.(cdiv x amt * amt)) x amt

(* [f x y] is its exact result where that fits an int; a result that does not is
   unspecified. *)
let is_exact ~exact f (x, y) =
  Option.iter (fun r -> equal int r (f x y)) (exact x y)

let gen_division = Gen.pair any_int (Gen.such_that (( <> ) 0) any_int)

let positive =
  Gen.frequency
    [
      (6, Gen.int_range 1 100);
      (2, Gen.such_that (fun n -> n > 0) Gen.int);
      (1, Gen.of_list ~pp:Format.pp_print_int [ 1; max_int - 1; max_int ]);
    ]

(* Pairs whose rounding wraps unless done with care. *)
let division_edges =
  [ (1, min_int); (min_int, min_int); (min_int, -1); (2, max_int) ]

let division_golden column f =
  Golden.cases ~key:[ "x"; "y" ] "division.golden" (fun cell ->
      let x = int_of_string (cell "x") and y = int_of_string (cell "y") in
      match cell column with
      | "None" -> ()
      | "ZeroDivisionError" -> raises Division_by_zero (fun () -> f x y)
      | expected -> equal int (int_of_string expected) (f x y))

let integers =
  group "Integers and lists"
    [
      prop "prod is the product of the elements, 1 if there are none"
        ~examples:[ []; [ 2; 3; 5 ] ]
        (Gen.list Gen.small_int)
        (fun l -> equal int (List.fold_left ( * ) 1 l) (prod l));
      prop "dedup keeps each element at its first occurrence" gen_words
        (fun l ->
          equal (list string)
            (first_occurrences Caseless.equal l)
            (dedup (module Caseless) l));
      prop
        "argsort orders positions by element, keeping equal elements in order, \
         so it inverts a permutation"
        gen_argsort (fun l ->
          cover "a permutation of two or more"
            (List.length l >= 2 && is_permutation l);
          cover "equal elements"
            (List.length (List.sort_uniq compare l) < List.length l);
          equal (list int) (stable_argsort l) (argsort l));
      cases "all_same is true iff every element equals the first"
        ~id:(fun (l, _) -> Format.asprintf "%a" (Testable.pp (list int)) l)
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
      cases "get_single_element refuses a list of another length"
        ~id:(fun l -> Format.asprintf "%a" (Testable.pp (list int)) l)
        [ []; [ 1; 2 ] ]
        (fun l -> refuses (fun () -> get_single_element l));
      group "ceildiv rounds num / amt up, as its golden states"
        [ division_golden "ceildiv" ceildiv ];
      prop "ceildiv is num / amt rounded up, where that fits an int"
        ~examples:division_edges gen_division
        (is_exact ~exact:(exact Bigint.cdiv) ceildiv);
      group
        "round_up is the least multiple of a positive amt at least num, as its \
         golden states"
        [ division_golden "round_up" round_up ];
      prop
        "round_up is the least multiple of a positive amt at least num, where \
         that fits an int"
        ~examples:[ (2, max_int); (max_int, 1); (min_int, max_int) ]
        (Gen.pair any_int positive)
        (is_exact ~exact:exact_round_up round_up);
      cases "round_up keeps a multiple and rounds up a negative number"
        ~id:(fun (num, amt, _) -> Printf.sprintf "round_up %d %d" num amt)
        [
          (-3, 4, 0);
          (-4, 4, -4);
          (6, 4, 8);
          (8, 4, 8);
          (232, 24984, 24984);
          (24984, 232, 25056);
        ]
        (fun (num, amt, r) -> equal int r (round_up num amt));
      group "floordiv rounds x / y down, 0 if y is 0, as its golden states"
        [ division_golden "floordiv" floordiv ];
      prop "floordiv is x / y rounded down, where that fits an int"
        ~examples:division_edges gen_division
        (is_exact ~exact:(exact Bigint.fdiv) floordiv);
      group "floormod has the sign of y, x if y is 0, as its golden states"
        [ division_golden "floormod" floormod ];
      prop "floormod is x - floordiv x y * y" ~examples:division_edges
        gen_division
        (is_exact ~exact:exact_floormod floormod);
      prop "lo32 is the low 32 bits" any_int (fun x ->
          equal int (x land 0xFFFF_FFFF) (lo32 x));
      prop "hi32 is x shifted right by 32 bits, keeping its sign" any_int
        (fun x -> equal int (x asr 32) (hi32 x));
      prop "data64 is (hi32 x, lo32 x)" any_int (fun x ->
          equal (pair int int) (hi32 x, lo32 x) (data64 x));
      prop "data64_le is (lo32 x, hi32 x)" any_int (fun x ->
          equal (pair int int) (lo32 x, hi32 x) (data64_le x));
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

let pp_color ppf = function
  | Rgb (r, g, b) -> Format.fprintf ppf "Rgb (%d, %d, %d)" r g b
  | c -> Format.pp_print_string ppf (List.assoc c colors)

(* The sixteen colours, and 24-bit ones. *)
let gen_color =
  Gen.one_of
    [
      Gen.of_list ~pp:pp_color (List.map fst colors);
      (let open Gen in
       let+ r = int_range 0 255
       and+ g = int_range 0 255
       and+ b = int_range 0 255 in
       Rgb (r, g, b));
    ]
  |> Gen.with_pp pp_color

let with_color f = Setting.context [ B (Setting.no_color, false) ] f

let colored_as_its_golden cell =
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

let gen_identifier =
  Gen.string_of
    (Gen.one_of
       [
         Gen.char_range 'a' 'z';
         Gen.char_range 'A' 'Z';
         Gen.char_range '0' '9';
         Gen.of_list ~pp:Format.pp_print_char [ '_' ];
       ])

let terminal_text =
  group "Terminal text"
    [
      group
        "colored paints each colour, or its background, as its golden states"
        [
          Golden.cases ~key:[ "color"; "background" ] "colors.golden"
            colored_as_its_golden;
        ];
      test "colored paints a 24-bit colour, or its background" (fun () ->
          with_color (fun () ->
              equal (pair string string)
                ( "\027[38;2;20;104;111mx\027[0m",
                  "\027[48;2;255;255;192mx\027[0m" )
                ( colored (Rgb (0x14, 0x68, 0x6f)) "x",
                  colored ~background:true (Rgb (0xff, 0xff, 0xc0)) "x" )));
      test "colored leaves text alone under no_color" (fun () ->
          Setting.context
            [ B (Setting.no_color, true) ]
            (fun () -> equal string "x" (colored ~background:true Red "x")));
      group "time_to_str writes each duration as its golden states"
        [
          Golden.cases ~key:[ "seconds"; "w" ] "durations.golden" (fun cell ->
              equal string (cell "time_to_str")
                (time_to_str
                   ~w:(int_of_string (cell "w"))
                   (float_of_string (cell "seconds"))));
        ];
      test "time_to_str is 8 columns wide by default" (fun () ->
          equal string "  500.00ms" (time_to_str 0.5));
      group "size_to_str writes each size as its golden states"
        [
          Golden.cases "sizes.golden" (fun cell ->
              equal string (cell "size_to_str")
                (size_to_str (int_of_string (cell "bytes"))));
        ];
      group "ansistrip strips each text as its golden states"
        [
          Golden.cases "text.golden" (fun cell ->
              equal string
                (literal (cell "ansistrip"))
                (ansistrip (text_input cell)));
        ];
      prop "ansistrip undoes colored"
        (Gen.triple gen_color Gen.bool without_escape)
        (fun (color, background, s) ->
          with_color (fun () ->
              equal string s (ansistrip (colored ~background color s))));
      group "ansilen counts each text as its golden states"
        [
          Golden.cases "text.golden" (fun cell ->
              equal int
                (int_of_string (cell "ansilen"))
                (ansilen (text_input cell)));
        ];
      prop "ansilen of text without escapes counts its characters"
        (Gen.list Gen.uchar
        |> Gen.map (fun uchars ->
            (List.length uchars, strip_escapes (utf_8 uchars)))
        |> Gen.with_pp (fun ppf (_, s) -> Format.fprintf ppf "%S" s))
        (fun (characters, s) -> equal int characters (ansilen s));
      prop "ansipad appends the spaces that bring its ansilen to the width"
        (Gen.pair gen_text (Gen.int_range (-2) 12))
        (fun (s, w) ->
          equal string
            (s ^ String.make (max 0 (w - ansilen s)) ' ')
            (ansipad s w));
      test "ansipad does not count escape sequences" (fun () ->
          with_color (fun () ->
              equal string
                (colored Red "ab" ^ "  ")
                (ansipad (colored Red "ab") 4)));
      group "strip_parens strips each text as its golden states"
        [
          Golden.cases "text.golden" (fun cell ->
              equal string
                (literal (cell "strip_parens"))
                (strip_parens (text_input cell)));
        ];
      prop "strip_parens removes parentheses around a balanced expression"
        (Gen.list
           (Gen.of_list ~pp:Format.pp_print_string
              [ "a"; "+"; "(a)"; "(a+(b))" ]))
        (fun parts ->
          let s = String.concat "" parts in
          equal string s (strip_parens ("(" ^ s ^ ")")));
      cases "pluralize writes cnt and st, with an s unless cnt is 1"
        ~id:(fun (st, cnt, _) -> Printf.sprintf "%d %s" cnt st)
        [
          ("kernel", 1, "1 kernel");
          ("kernel", 0, "0 kernels");
          ("kernel", 2, "2 kernels");
          ("kernel", -1, "-1 kernels");
        ]
        (fun (st, cnt, expected) -> equal string expected (pluralize st cnt));
      group "to_function_name names each text as its golden states"
        [
          Golden.cases "text.golden" (fun cell ->
              equal string (cell "to_function_name")
                (to_function_name (text_input cell)));
        ];
      prop "to_function_name makes a string of letters, digits and underscores"
        gen_text (fun s ->
          satisfies ~claim:"letters, digits and underscores" string
            (String.for_all identifier)
            (to_function_name s));
      prop "to_function_name keeps an identifier" gen_identifier (fun s ->
          equal string s (to_function_name s));
    ]

(* Disk cache *)

let cache_dir_group =
  group "cache_dir"
    [
      test "is tolk in XDG_CACHE_HOME" (fun () ->
          read_at_start
            [ ("XDG_CACHE_HOME", Some "/xdg") ]
            [ ("cache_dir", path "/xdg" [ "tolk" ]) ]);
      test
        "without XDG_CACHE_HOME, is tolk in the platform's caches of the home, \
         USERPROFILE on Windows and HOME elsewhere" (fun () ->
          let home = if Sys.win32 then "/profile" else "/home" in
          let caches =
            if macos () then [ "Library"; "Caches" ] else [ ".cache" ]
          in
          read_at_start
            [
              ("XDG_CACHE_HOME", None);
              ("HOME", Some "/home");
              ("USERPROFILE", Some "/profile");
            ]
            [ ("cache_dir", path home (caches @ [ "tolk" ])) ]);
    ]

(* Owners and modes *)

let describe file =
  match Unix.lstat file with
  | exception Unix.Unix_error (e, _, _) -> file ^ ": " ^ Unix.error_message e
  | st ->
      Printf.sprintf "%s: mode %o, owner %d, this user %d" file st.st_perm
        st.st_uid (Unix.geteuid ())

let not_root f () =
  if Unix.geteuid () = 0 then skip ~reason:"root writes every directory" ();
  f ()

(* [/] is root's, and no other user can write it, unless this process runs in a
   user namespace that does not map the system's root: there [/] shows as the
   overflow user's, another user's. *)
let root_owner () =
  if Sys.win32 then skip ~reason:"no owners or modes" ();
  (Unix.stat "/").st_uid

let dir_of_mode mode =
  let dir = temp_dir () in
  Unix.chmod dir mode;
  dir

let safe_dir_group =
  group "safe_dir"
    [
      cases
        "a directory of this user's is safe iff no other user can write it, or \
         its sticky bit is set"
        ~id:(fun (mode, _) -> Printf.sprintf "%o" mode)
        [
          (0o700, true);
          (0o755, true);
          (0o770, false);
          (0o707, false);
          (0o777, false);
          (0o1777, true);
          (0o1770, true);
        ]
        (fun (mode, safe) ->
          unix_only
            (fun () ->
              let dir = dir_of_mode mode in
              equal ~msg:(describe dir) bool safe (safe_dir dir))
            ());
      test "the system's directory is safe" (fun () ->
          if root_owner () <> 0 then skip ~reason:"/ is not root's here" ();
          equal ~msg:(describe "/") bool true (safe_dir "/"));
      test "another user's directory is not safe" (fun () ->
          let owner = root_owner () in
          if owner = 0 || owner = Unix.geteuid () then
            skip ~reason:"/ is root's or this user's here" ();
          equal ~msg:(describe "/") bool false (safe_dir "/"));
      test "a link to a safe directory is safe"
        (unix_only (fun () ->
             let link = Filename.concat (temp_dir ()) "link" in
             Unix.symlink (dir_of_mode 0o700) link;
             equal ~msg:(describe link) bool true (safe_dir link)));
      test "a file is not safe" (fun () ->
          let file = temp_file () in
          equal ~msg:(describe file) bool false (safe_dir file));
      test "a path that does not exist is not safe" (fun () ->
          let missing = Filename.concat (temp_dir ()) "missing" in
          equal ~msg:(describe missing) bool false (safe_dir missing));
    ]

(* Asserts that [open_owned] opens nothing at [path]. *)
let refused path =
  let opened = open_owned path in
  Option.iter Unix.close opened;
  is_none ~msg:(describe path) opened

let file_of_mode mode =
  let file = temp_file () in
  write file "x";
  Unix.chmod file mode;
  file

let open_owned_group =
  group "open_owned"
    [
      test "opens a regular file of this user's that no other user can write"
        (fun () ->
          let file = file_of_mode 0o644 in
          let fd = require_some ~msg:(describe file) (open_owned file) in
          let ic = Unix.in_channel_of_descr fd in
          equal string "x"
            (Fun.protect
               ~finally:(fun () -> close_in ic)
               (fun () -> In_channel.input_all ic)));
      cases "refuses a file another user can write" ~id:(Printf.sprintf "%o")
        [ 0o620; 0o602; 0o666 ] (fun mode ->
          unix_only (fun () -> refused (file_of_mode mode)) ());
      test "refuses a file reached through a link"
        (unix_only (fun () ->
             let link = Filename.concat (temp_dir ()) "link" in
             Unix.symlink (file_of_mode 0o600) link;
             refused link));
      test "refuses a directory" (fun () -> refused (temp_dir ()));
      test "refuses a pipe, without waiting for a writer"
        (unix_only (fun () ->
             let fifo = Filename.concat (temp_dir ()) "fifo" in
             Unix.mkfifo fifo 0o600;
             refused fifo));
      (* A file of another user's, root's, that no other user can write. *)
      test "refuses a file of another owner" (fun () ->
          let file = "/etc/passwd" in
          match Unix.stat file with
          | exception Unix.Unix_error _ -> skip ~reason:"no /etc/passwd" ()
          | st when Sys.win32 || st.st_uid = Unix.geteuid () ->
              skip ~reason:"/etc/passwd is this user's" ()
          | _ -> refused file);
      test "raises Unix_error on a path that does not exist" (fun () ->
          let missing = Filename.concat (temp_dir ()) "missing" in
          raises_match
            (function Unix.Unix_error (ENOENT, _, _) -> true | _ -> false)
            (fun () -> open_owned missing));
    ]

(* Windows holds a path of at most 259 characters, and the cache's files lie at
   most 101 beneath its directory: an entry of a generation, its longest name,
   is put and read in a child under a cachedb of [length] characters. *)
let put_under_cachedb length =
  let root = temp_dir () in
  let root =
    if Filename.is_relative root then Filename.concat (Sys.getcwd ()) root
    else root
  in
  let room = length - String.length root - 1 in
  if room < 1 then
    skip ~reason:("the temporary directory is too deep: " ^ root) ();
  let cachedb = Filename.concat root (String.make room 'd') in
  succeeds
    (child
       ~env:[ ("CACHEDB", Some cachedb) ]
       "put and get after a clear" [ "t" ])

let cachedb_group =
  group "cachedb"
    [
      test "is CACHEDB when set" (fun () ->
          read_at_start [ ("CACHEDB", Some "/db") ] [ ("cachedb", "/db") ]);
      test "defaults to cache in cache_dir" (fun () ->
          read_at_start
            [ ("XDG_CACHE_HOME", Some "/xdg"); ("CACHEDB", None) ]
            [ ("cachedb", path "/xdg" [ "tolk"; "cache" ]) ]);
      test "defaults to a directory made absolute" (fun () ->
          read_at_start
            [ ("XDG_CACHE_HOME", Some "relative"); ("CACHEDB", None) ]
            [
              ("cachedb", path (Sys.getcwd ()) [ "relative"; "tolk"; "cache" ]);
            ]);
      test
        "an entry of a generation is kept in a cache whose path has 158 \
         characters, what the cache's names leave of Windows' 259" (fun () ->
          equal string "put\nv" (put_under_cachedb 158));
      test
        "on Windows, under a longer cachedb, a put after a clear fails and a \
         get misses" (fun () ->
          if not Sys.win32 then skip ~reason:"paths are not bounded at 259" ();
          equal string "Sys_error\n<none>" (put_under_cachedb 159));
    ]

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

let directories () =
  if not (Sys.file_exists cachedb) then []
  else
    List.filter Sys.is_directory
      (List.map (Filename.concat cachedb) (Array.to_list (Sys.readdir cachedb)))

(* The entries of the cache: the files of its directories. *)
let entries_on_disk () = List.concat_map files (directories ())

(* An empty cache: cleared, and without the tables' directories, which a clear
   keeps and every later clear reads. Every test of the cache starts from one
   and leaves one. *)
let empty_cache () =
  Diskcache.clear ();
  List.iter
    (fun dir -> try Sys.rmdir dir with Sys_error _ -> ())
    (directories ())

let cache_test name body =
  bracket ~setup:empty_cache ~teardown:empty_cache name body

(* [leaves_empty f x] is [f x], which leaves the cache empty however it ends:
   for a case of a property or a row of a table. *)
let leaves_empty f x = Fun.protect ~finally:empty_cache (fun () -> f x)

(* [put_entry ~table key v] puts [v] at [key] in [table], a key not yet there,
   and is the file of its entry. *)
let put_entry ~table key v =
  let before = entries_on_disk () in
  Diskcache.put ~table key v;
  List.find (fun f -> not (List.mem f before)) (entries_on_disk ())

(* The directory of [table]'s entries, found as the one its first entry
   creates. *)
let table_directory table = Filename.dirname (put_entry ~table "k" "v")

(* The bytes the files under [cachedb] hold. *)
let cache_bytes () =
  if not (Sys.file_exists cachedb) then 0
  else List.fold_left (fun n f -> n + (Unix.stat f).st_size) 0 (files cachedb)

let sized limit f = Setting.context [ B (Setting.cachesize, limit) ] f
let disabled f = Setting.context [ B (Setting.cachelevel, 0) ] f

(* The model of the cache: one table of the entries, reset when the cache is
   cleared. The cache is one, so its model is one table, not one per [cache]
   value: two clears make two values that both stand for it. windtrap runs each
   program from no value and warns that a module's state carries over from run
   to run: every program starts with a clear, the only command that makes a
   [cache], and that clear resets the table. On two domains windtrap replays the
   calls on the reference along each order; this assumes each replay runs the
   prefix's reference calls again, from that first clear. *)
let entries : (string * string, string) Hashtbl.t = Hashtbl.create 16
let cache = abstract "cache" ~release:empty_cache

let tables =
  Gen.of_list ~pp:Format.pp_print_string
    [ "a"; "A"; "b"; "test_gfx1010:xnack-"; "long" ^ String.make 300 't' ]

let keys = Gen.of_list ~pp:Format.pp_print_string [ "k"; "K"; ""; "k/../k" ]
let values = Gen.string_of ~size:(Gen.int_range 0 8) Gen.char

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

(* [during cond f action] runs [action], running [f] once on its domain at its
   first allocation where [cond ()] holds. *)
let during cond f action =
  let me = Domain.self () and armed = ref false and ran = ref false in
  let alloc _ =
    if !armed && Domain.self () = me && (not !ran) && cond () then begin
      ran := true;
      f ()
    end;
    None
  in
  let tracker =
    { Gc.Memprof.null_tracker with alloc_minor = alloc; alloc_major = alloc }
  in
  let profile = Gc.Memprof.start ~sampling_rate:1. tracker in
  Gc.minor ();
  armed := true;
  Fun.protect
    ~finally:(fun () ->
      armed := false;
      Gc.Memprof.stop ();
      Gc.Memprof.discard profile)
    action

(* A clear has removed one of two entries and not the other when another domain
   or process, [race ~table ~left], puts a new entry and then reads the other,
   [left], and the clear goes on once it returns. Whatever order the calls take,
   that read sees the other entry only if the put came before the clear, so the
   new entry cannot also be there after the clear. *)
let clear_orders_calls race =
  let table = fresh_table () in
  let first = put_entry ~table "x1" "old" in
  let second = put_entry ~table "x2" "old" in
  let split () = Sys.file_exists first <> Sys.file_exists second in
  let left () = if Sys.file_exists first then "x1" else "x2" in
  let seen = ref None in
  during split
    (fun () -> seen := Some (race ~table ~left:(left ())))
    Diskcache.clear;
  let seen = require_some ~msg:"the calls ran during the clear" !seen in
  satisfies ~claim:"the results of some order of the calls"
    (pair (option string) (option string))
    (fun (seen, after) -> not (Option.is_some seen && Option.is_some after))
    (seen, Diskcache.get ~table "new")

let put_get_on_a_domain ~table ~left =
  Domain.join
    (Domain.spawn (fun () ->
         Diskcache.put ~table "new" "new";
         Diskcache.get ~table left))

let put_get_in_a_process ~table ~left =
  of_printed (succeeds (child "put then get" [ table; "new"; left ]))

(* A put of [a] at a key that holds [o] is held once its write has begun: a file
   other than the entry's is in the table's directory, or the entry's file has
   changed. There the other side, [other ~table], gets the key, puts [b] and
   gets it again. [a] is shorter than [b], so two writes that share a file leave
   a mixture. *)
let o = String.make 300 'o'
let a = String.make 100 'a'
let b = String.make 400 'b'

let put_held other =
  let table = fresh_table () in
  let entry = put_entry ~table "k" o in
  let dir = Filename.dirname entry and old = read entry in
  let begun () =
    Sys.readdir dir <> [| Filename.basename entry |]
    || try read entry <> old with Sys_error _ -> true
  in
  let seen = ref None in
  during begun
    (fun () -> seen := Some (other ~table))
    (fun () -> Diskcache.put ~table "k" a);
  let first, second = require_some ~msg:"the calls ran during the put" !seen in
  let whole l v = List.exists (fun x -> v = Some x) l in
  satisfies ~claim:"o or a, whole" (option string) (whole [ o; a ]) first;
  satisfies ~claim:"a or b, whole" (option string) (whole [ a; b ]) second;
  satisfies ~claim:"a or b, whole, and a if the second get read a"
    (option string)
    (fun v -> whole [ a; b ] v && (second <> Some a || v = Some a))
    (Diskcache.get ~table "k")

let get_put_get_on_a_domain ~table =
  Domain.join
    (Domain.spawn (fun () ->
         let first = Diskcache.get ~table "k" in
         Diskcache.put ~table "k" b;
         (first, Diskcache.get ~table "k")))

let get_put_get_in_a_process ~table =
  match
    String.split_on_char '\n'
      (succeeds (child "get, put, get" [ table; "k"; b ]))
  with
  | [ first; second ] -> (of_printed first, of_printed second)
  | lines -> failf "the child printed %d lines" (List.length lines)

let truncate file = Unix.truncate file ((Unix.stat file).st_size / 2)
let overwrite file = write file "not an entry"

(* Runs [damage] on the file of an entry, then reads the entry back. *)
let damaged damage =
  let table = fresh_table () in
  damage (put_entry ~table "k" (String.make 4096 'v'));
  match Diskcache.get ~table "k" with
  | exception Failure _ -> ()
  | None -> fail "get is None"
  | Some v -> failf "get is a value of %d bytes" (String.length v)

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

(* A link planted where a table's directory stands: a put refuses to write
   through it, and a clear removes nothing where it points. *)
let keeps_out_of_a_linked_table () =
  let table = fresh_table () in
  let dir = table_directory table in
  Diskcache.clear ();
  Array.iter (fun f -> Sys.remove (Filename.concat dir f)) (Sys.readdir dir);
  Sys.rmdir dir;
  let elsewhere = temp_dir () in
  write (Filename.concat elsewhere (String.make 32 'a')) "kept";
  Unix.symlink elsewhere dir;
  Fun.protect
    ~finally:(fun () -> Sys.remove dir)
    (fun () ->
      raises_match Exn.sys_error (fun () -> Diskcache.put ~table "k" "v");
      Diskcache.clear ();
      equal (list string) ~msg:"where the link points"
        [ String.make 32 'a' ]
        (Array.to_list (Sys.readdir elsewhere)))

let disabled_cache level =
  let table = fresh_table () in
  Diskcache.put ~table "kept" "v";
  Setting.context
    [ B (Setting.cachelevel, level) ]
    (fun () ->
      Diskcache.put ~table "dropped" "v";
      equal (option string) None (Diskcache.get ~table "kept"));
  equal
    (pair (option string) (option string))
    (Some "v", None)
    (Diskcache.get ~table "kept", Diskcache.get ~table "dropped")

(* Entries no other user wrote: an entry another user could have written,
   planted, linked or moved, reads as none, and a put replaces it. *)

(* What [get] reads of an entry [v] at the key [k] of a table once [f ~table]
   has run on its file, then once a put has replaced it with [v2]. *)
let after_tamper f =
  let table = fresh_table () in
  let file = put_entry ~table "k" "v" in
  f ~table file;
  let tampered = Diskcache.get ~table "k" in
  Diskcache.put ~table "k" "v2";
  (tampered, Diskcache.get ~table "k")

let refused_entry = (None, Some "v2")
let tampered = pair (option string) (option string)

let refuses_a_link () =
  equal tampered refused_entry
    (after_tamper (fun ~table:_ file ->
         let target = temp_file () in
         write target (read file);
         Sys.remove file;
         Unix.symlink target file))

let reads_by_mode (mode, readable) =
  equal tampered
    (if readable then (Some "v", Some "v2") else refused_entry)
    (after_tamper (fun ~table:_ file -> Unix.chmod file mode))

(* This user's own entry, moved by a writer of the cache to another key's or
   another table's place, or written as a library of another version would, or
   from before a clear, moved to where the next generation keeps it. *)
let refuses_another_key () =
  equal tampered refused_entry
    (after_tamper (fun ~table file ->
         Unix.rename (put_entry ~table "j" "a") file))

let refuses_another_table () =
  let source = put_entry ~table:(fresh_table ()) "k" "a" in
  equal tampered refused_entry
    (after_tamper (fun ~table:_ file -> Unix.rename source file))

let refuses_another_version () =
  equal tampered refused_entry
    (after_tamper (fun ~table:_ file ->
         write file (Disk_cache.of_another_version (read file))))

let refuses_an_earlier_generation () =
  let table = fresh_table () in
  let old = read (put_entry ~table "k" "v") in
  Diskcache.clear ();
  write (put_entry ~table "k" "now") old;
  equal (option string) None (Diskcache.get ~table "k")

(* A put where a link was planted for its entry replaces the link, writing
   nothing where it points, and its entry no other user can read. *)
let puts_past_a_link () =
  let table = fresh_table () in
  let file = put_entry ~table "k" "v" in
  let target = temp_file () in
  write target "kept";
  Sys.remove file;
  Unix.symlink target file;
  Diskcache.put ~table "k" "v2";
  equal
    (triple string (option string) int)
    ("kept", Some "v2", 0o600)
    (read target, Diskcache.get ~table "k", (Unix.lstat file).st_perm land 0o777)

(* A read holds an entry open, shared for deletion, while a put renames a new
   entry over it: on Windows a handle that does not share deletion makes that
   rename fail. *)
let put_over_a_held_entry () =
  let table = fresh_table () in
  let entry = put_entry ~table "k" "old" in
  let fd = Unix.openfile entry [ O_RDONLY; O_SHARE_DELETE ] 0 in
  Fun.protect
    ~finally:(fun () -> Unix.close fd)
    (fun () -> Diskcache.put ~table "k" "new");
  equal (option string) (Some "new") (Diskcache.get ~table "k")

let limits = Gen.of_list ~pp:Format.pp_print_int [ 1; 100; 4096; 1 lsl 16 ]

let lengths =
  Gen.list ~size:(Gen.int_range 1 40)
    (Gen.frequency [ (4, Gen.int_range 0 4096); (1, Gen.int_range 0 70_000) ])

let holds_its_size (limit, lengths) =
  empty_cache ();
  let table = fresh_table () in
  cover "past the bound" (List.fold_left ( + ) 0 lengths > limit);
  sized limit (fun () ->
      List.iteri
        (fun i n ->
          Diskcache.put ~table (string_of_int i) (String.make n 'v');
          at_most int ~than:limit (cache_bytes ()))
        lengths)

(* [ago file s] sets [file]'s times [s] seconds back. *)
let ago file s =
  let t = Unix.gettimeofday () -. s in
  Unix.utimes file t t

let entry_value = String.make 4096 'v'

(* Puts entries of 4 KiB at keys of one length, so that every entry has one
   size, until one shrinks the files of the cache, which a bound of 64 KiB
   reaches within 100; is the bytes the files then hold. *)
let fill_until_trimmed table =
  let rec fill i before =
    if i = 100 then fail "100 puts of 4 KiB never trimmed the cache";
    Diskcache.put ~table (Printf.sprintf "%03d" i) entry_value;
    let after = cache_bytes () in
    if after >= before then fill (i + 1) after else after
  in
  fill 0 (cache_bytes ())

(* The keys of [keys] that [table] holds. *)
let held table keys =
  List.filter (fun k -> Option.is_some (Diskcache.get ~table k)) keys

(* Use is ordered by file time: "read" is used an hour ago and read back now,
   "unread" used half an hour ago. *)
let keeps_recently_used () =
  let table = fresh_table () in
  sized (1 lsl 16) (fun () ->
      ago (put_entry ~table "read" entry_value) 3600.;
      ago (put_entry ~table "unread" entry_value) 1800.;
      ignore (require_some (Diskcache.get ~table "read"));
      ignore (fill_until_trimmed table);
      equal (list string) [ "read" ] (held table [ "read"; "unread" ]))

(* The trim keeps entries while they and the new one hold four fifths of the
   bound, so one more of its entries, of one size, would take them past it. The
   cache counts its size in a file of its own, of up to a byte per KiB of the
   bound. *)
let trims_to_four_fifths () =
  let table = fresh_table () and limit = 1 lsl 16 in
  let size = (Unix.stat (put_entry ~table "000" entry_value)).st_size in
  let after = sized limit (fun () -> fill_until_trimmed table) in
  let four_fifths = limit - (limit / 5) in
  at_most int ~than:four_fifths after;
  greater int ~than:four_fifths (after + size + (limit / 1024))

(* [write_aside table bytes s] is a file named like an entry of [table] being
   written, of [bytes] bytes, last written [s] seconds ago. *)
let write_aside table bytes s =
  let file =
    Filename.concat (table_directory table) (String.make 32 'a' ^ ".1.tmp")
  in
  write file (String.make bytes 'v');
  ago file s;
  file

(* A write a killed process left, last written two hours ago. *)
let removes_stale_writes () =
  let table = fresh_table () and limit = 1 lsl 16 in
  let stale = write_aside table (limit / 2) 7200. in
  ignore (sized limit (fun () -> fill_until_trimmed table));
  equal (list string) [] (List.filter Sys.file_exists [ stale ]);
  at_most int ~than:limit (cache_bytes ())

(* A write another process has in progress, last written a minute ago. *)
let keeps_writes_in_progress () =
  let table = fresh_table () in
  let file = write_aside table 4096 60. in
  ignore (sized (1 lsl 16) (fun () -> fill_until_trimmed table));
  equal string (String.make 4096 'v') (read file);
  Sys.remove file

(* A put into a table whose directory the process may not write fails, as a put
   on a full disk does. Entries used up to 16 minutes ago fill the cache past a
   bound of 64 KiB, the oldest under the key "0". *)
let trims_before_writing () =
  let full = fresh_table () and locked = fresh_table () in
  let dir = table_directory locked in
  for i = 0 to 15 do
    ago
      (put_entry ~table:full (string_of_int i) entry_value)
      (float_of_int ((16 - i) * 60))
  done;
  Unix.chmod dir 0o555;
  Fun.protect
    ~finally:(fun () -> Unix.chmod dir 0o755)
    (fun () ->
      raises_match Exn.sys_error (fun () ->
          sized (1 lsl 16) (fun () ->
              Diskcache.put ~table:locked "k2" entry_value)));
  equal (option string) None (Diskcache.get ~table:full "0")

let skips_an_oversized_entry () =
  let table = fresh_table () in
  sized 4096 (fun () ->
      Diskcache.put ~table "kept" "v";
      let before = cache_bytes () in
      Diskcache.put ~table "big" (String.make 8192 'v');
      let bytes = cache_bytes () in
      equal
        (triple int (option string) (option string))
        (before, None, Some "v")
        (bytes, Diskcache.get ~table "big", Diskcache.get ~table "kept"))

let unbounded () =
  let table = fresh_table () in
  let keys = List.init 32 string_of_int in
  sized 0 (fun () ->
      List.iter (fun k -> Diskcache.put ~table k entry_value) keys);
  equal (list string) keys (held table keys)

(* Each process puts less than the bound; together they put more. *)
let processes_share_the_bound () =
  let table = fresh_table () and limit = 1 lsl 16 in
  for i = 1 to 10 do
    ignore
      (succeeds
         (child
            ~env:[ ("CACHESIZE", Some (string_of_int limit)) ]
            "fill"
            [ table; string_of_int i ]));
    at_most int ~than:limit (cache_bytes ())
  done

(* The file in which the puts of all processes count the cache's size. *)
let size_file = Filename.concat cachedb "f7bd60b75b29d79b660a2859395c1a24_size"

(* No put can count while a directory stands at the size's file. *)
let stores_uncounted () =
  if not (Sys.file_exists cachedb) then Sys.mkdir cachedb 0o755;
  Sys.mkdir size_file 0o755;
  let table = fresh_table () in
  let stored =
    Fun.protect
      ~finally:(fun () -> Sys.rmdir size_file)
      (fun () ->
        sized (1 lsl 16) (fun () -> Diskcache.put ~table "k" "v");
        Diskcache.get ~table "k")
  in
  equal (option string) (Some "v") stored

(* Files the cache did not write, some named like its own: a table directory
   holding a file whose name is no digest, directories whose names are not a
   digest and a version, a plain file where a table directory would be, and a
   file named [size]. *)
let keeps_foreign_files () =
  let under dir name =
    let dir = Filename.concat cachedb dir in
    if not (Sys.file_exists dir) then Sys.mkdir dir 0o755;
    Filename.concat dir name
  in
  let digest = String.make 32 'a' in
  let lookalike = table_directory (fresh_table ()) in
  Diskcache.clear ();
  Sys.rmdir lookalike;
  let live = table_directory (fresh_table ()) in
  let foreign =
    [
      Filename.concat cachedb "foreign.txt";
      Filename.concat cachedb "size";
      under "foreign" digest;
      under (digest ^ "_") digest;
      under (digest ^ "x1") digest;
      under (String.make 32 'z' ^ "_1") digest;
      Filename.concat live (String.make 32 'z');
      Filename.concat live (digest ^ "x");
      lookalike;
    ]
  in
  List.iter (fun f -> write f "foreign") foreign;
  Diskcache.put ~table:(fresh_table ()) "k" "v";
  Diskcache.clear ();
  let lost f = not (Sys.file_exists f && read f = "foreign") in
  equal (list string) [] (List.filter lost foreign);
  List.iter Sys.remove foreign

(* An entry that cannot be removed: on Windows one held open without sharing
   deletion, elsewhere one in a table directory the process may not write. *)
let clear_leaves_held_entries () =
  if (not Sys.win32) && Unix.geteuid () = 0 then
    skip ~reason:"every directory is writable" ();
  let kept = fresh_table () and free = fresh_table () in
  let entry = put_entry ~table:kept "k" "v" in
  Diskcache.put ~table:free "k" "v";
  let hold, release =
    if Sys.win32 then
      (* Opened without sharing deletion, as a program other than the cache
         may. *)
      let fd = Unix.openfile entry [ O_RDONLY ] 0 in
      ((fun () -> ()), fun () -> Unix.close fd)
    else
      let dir = Filename.dirname entry in
      ((fun () -> Unix.chmod dir 0o555), fun () -> Unix.chmod dir 0o755)
  in
  hold ();
  Fun.protect ~finally:release Diskcache.clear;
  equal (option string) None (Diskcache.get ~table:free "k")

(* The directory of a table stays through a clear, so that a put in another
   process that has made it can still write there. *)
let clear_keeps_tables () =
  let dir = table_directory (fresh_table ()) in
  Diskcache.clear ();
  satisfies ~claim:"a directory" string
    (fun d -> Sys.file_exists d && Sys.is_directory d)
    dir

(* A write in progress a minute old stays through a clear, one two hours old
   does not. *)
let clear_keeps_writes_in_progress () =
  let young = write_aside (fresh_table ()) 4096 60. in
  let stale = write_aside (fresh_table ()) 4096 7200. in
  Diskcache.clear ();
  equal (list string) [ young ] (List.filter Sys.file_exists [ young; stale ]);
  Sys.remove young

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
          (* Programs are short: the branches, five calls each, are what the
             test is for, so at most 4 calls run before and after them. windtrap
             0.3.1 leaves each command out of a case with probability 1/4, so
             with the lengths it draws a case calls a given command with odds
             0.577: no case of 28 calls it with odds 0.423^28, 2e-10 over the
             five that run in branches. The rate is windtrap 0.3.1's draw, which
             windtrap.mli leaves out of its contract: a new windtrap needs the
             count checked again. *)
          stateful ~count:28 ~steps:4 ~domains:2
            "behaves as a table of entries on two domains, whatever the order \
             of the calls"
            model;
          cache_test "get is None for a table never written" (fun () ->
              equal (option string) None
                (Diskcache.get ~table:(fresh_table ()) "k"));
          cache_test "put replaces the value of a key" (fun () ->
              let table = fresh_table () in
              Diskcache.put ~table "hello" "world";
              Diskcache.put ~table "hello" "world2";
              equal (option string) (Some "world2")
                (Diskcache.get ~table "hello"));
          prop ~count:30 "get reads back any key and value put"
            (Gen.pair Gen.string Gen.string)
            ( leaves_empty @@ fun (key, v) ->
              Diskcache.put ~table:"values" key v;
              equal (option string) (Some v) (Diskcache.get ~table:"values" key)
            );
          cache_test "a megabyte value reads back" (fun () ->
              let table = fresh_table ()
              and v = String.init (1 lsl 20) (fun i -> Char.chr (i land 255)) in
              Diskcache.put ~table "k" v;
              equal (option string) (Some v) (Diskcache.get ~table "k"));
          cache_test "an entry put by another process reads back" (fun () ->
              let table = fresh_table () in
              ignore (succeeds (child "put" [ table; "k"; "remote" ]));
              equal (option string) (Some "remote") (Diskcache.get ~table "k"));
          cache_test "another process reads an entry put back" (fun () ->
              let table = fresh_table () in
              Diskcache.put ~table "k" "getme";
              equal string "getme" (succeeds (child "get" [ table; "k" ])));
          cache_test
            "while a put writes, another domain reads whole entries, and the \
             last writer wins" (fun () -> put_held get_put_get_on_a_domain);
          cache_test
            "while a put writes, another process reads whole entries, and the \
             last writer wins" (fun () -> put_held get_put_get_in_a_process);
          cache_test "get fails on a truncated entry" (fun () ->
              damaged truncate);
          cache_test "get fails on an entry that is no entry" (fun () ->
              damaged overwrite);
          cache_test "an entry of another version reads as none"
            refuses_another_version;
          cache_test "keys and tables that look like paths stay inside cachedb"
            stays_inside_cachedb;
          cache_test
            "puts and clears keep out of a link for a table's directory"
            (unix_only keeps_out_of_a_linked_table);
          cases "a disabled cache neither reads nor writes"
            ~id:(Printf.sprintf "cachelevel=%d")
            [ 0; -1 ]
            (leaves_empty disabled_cache);
          cache_test "an entry reached through a link reads as none"
            (unix_only refuses_a_link);
          cases "an entry reads only if no other user can write it"
            ~id:(fun (mode, _) -> Printf.sprintf "%o" mode)
            [
              (0o600, true);
              (0o644, true);
              (0o620, false);
              (0o602, false);
              (0o666, false);
            ]
            (leaves_empty (fun row ->
                 unix_only (fun () -> reads_by_mode row) ()));
          cache_test "an entry moved from another key reads as none"
            refuses_another_key;
          cache_test "an entry moved from another table reads as none"
            refuses_another_table;
          cache_test "an entry from before a clear reads as none"
            refuses_an_earlier_generation;
          cache_test "a put past a planted link writes its own entry"
            (unix_only puts_past_a_link);
          cache_test "a put replaces an entry that a read holds open"
            put_over_a_held_entry;
          prop ~count:30 "the files of the cache never hold more than cachesize"
            (Gen.pair limits lengths)
            (leaves_empty holds_its_size);
          cache_test
            "a put past cachesize removes entries used least recently first"
            keeps_recently_used;
          cache_test
            "a put past cachesize leaves entries that hold four fifths of \
             cachesize with its own"
            trims_to_four_fifths;
          cache_test
            "a put that cannot write removes entries past cachesize first"
            (unix_only (not_root trims_before_writing));
          cache_test
            "a put past cachesize removes the writes a killed process left"
            removes_stale_writes;
          cache_test
            "a put past cachesize keeps another process's write in progress"
            keeps_writes_in_progress;
          cache_test "a put of an entry alone past cachesize writes nothing"
            skips_an_oversized_entry;
          cache_test "a cachesize of 0 keeps every entry" unbounded;
          cache_test "a put under a negative cachesize raises Invalid_argument"
            (fun () ->
              refuses (fun () ->
                  sized (-1) (fun () ->
                      Diskcache.put ~table:(fresh_table ()) "k" "v")));
          cache_test "processes putting in turn keep the cache within cachesize"
            processes_share_the_bound;
          cache_test "a put that cannot count the cache's size stores its entry"
            stores_uncounted;
          cache_test "clear removes the entries of every table" (fun () ->
              let a = fresh_table () and b = fresh_table () in
              Diskcache.put ~table:a "k" "v";
              Diskcache.put ~table:b "k" "v";
              Diskcache.clear ();
              Diskcache.clear ();
              equal
                (pair (option string) (option string))
                (None, None)
                (Diskcache.get ~table:a "k", Diskcache.get ~table:b "k"));
          cache_test "clear keeps another process's write in progress"
            clear_keeps_writes_in_progress;
          cache_test "clear leaves the entries it cannot remove"
            clear_leaves_held_entries;
          cache_test "clear keeps the directories of the tables"
            clear_keeps_tables;
          cache_test "put and clear keep the files that are not entries"
            keeps_foreign_files;
          cache_test
            "a put and a get on another domain come before or after a clear"
            (fun () -> clear_orders_calls put_get_on_a_domain);
          cache_test
            "a put and a get in another process come before or after a clear"
            (fun () -> clear_orders_calls put_get_in_a_process);
        ]

(* Programs *)

(* The bytes of the object of [src], compiled by the first of [compilers] that
   can, if any. *)
let compiled ?(flags = "") compilers src =
  let obj = temp_file ~suffix:".o" () in
  List.find_map
    (fun cc ->
      match
        system ~input:src (Printf.sprintf "%s %s -c -x c - -o %s" cc flags obj)
      with
      | _ -> Some (read obj)
      | exception Failure _ -> None)
    compilers

let last_line s = List.hd (List.rev (String.split_on_char '\n' (String.trim s)))

let llvm_objdump () =
  match find_llvm_objdump () with
  | path -> path
  | exception Failure why -> skip ~reason:why ()

let programs =
  group "Programs"
    [
      test "system is the output of a command" (fun () ->
          equal string "hello" (system "echo hello"));
      test "system splits a command at runs of white space" (fun () ->
          equal string "a b" (system "echo  a\t b"));
      test "system gives the input on standard input" (fun () ->
          equal string "abc" (system ~input:"abc" "cat"));
      test "system without input gives the process's standard input" (fun () ->
          let input = temp_file () in
          write input "abc";
          equal string "abc" (succeeds (child ~stdin:input "system" [])));
      test "system is standard output and standard error, stripped" (fun () ->
          equal string "out\nerr"
            (system ~input:"echo; echo out; echo err 1>&2; echo" "sh"));
      test "system fails naming the command, its exit code and its output"
        (fun () ->
          match system ~input:"echo why; exit 3" "sh" with
          | _ -> fail "system returned"
          | exception Failure m -> in_order ~subs:[ "sh"; "3"; "why" ] m);
      test "system fails naming a program that does not exist" (fun () ->
          raises_match (Exn.failure ~sub:"tolk-no-such-program") (fun () ->
              system "tolk-no-such-program"));
      test "system prints how many bytes it returned, from DEBUG=1" (fun () ->
          ignore (output ());
          ignore
            (Setting.context
               [ B (Setting.debug, 1) ]
               (fun () -> system "echo hello"));
          contains ~sub:"5 bytes" (output ()));
      test "cpu_objdump prints the instructions of an object" (fun () ->
          match
            compiled [ "cc"; "clang" ] "int answer(void) { return 42; }"
          with
          | None -> skip ~reason:"no C compiler" ()
          | Some lib ->
              ignore (output ());
              cpu_objdump lib;
              contains ~sub:"Disassembly of section" (output ()));
      test "cpu_objdump fails on bytes objdump cannot read" (fun () ->
          (match system "objdump --version" with
          | _ -> ()
          | exception Failure _ -> skip ~reason:"no objdump" ());
          raises_match (Exn.failure ~sub:"objdump") (fun () ->
              cpu_objdump "not an object"));
      test "find_llvm_objdump fails without llvm-objdump" (fun () ->
          if macos () then skip ~reason:"Homebrew's path is the answer" ();
          if Sys.file_exists "/opt/rocm/llvm/bin/llvm-objdump" then
            skip ~reason:"ROCm's llvm-objdump is installed" ();
          starts_with ~prefix:"Failure: "
            (succeeds
               (child
                  ~env:[ ("PATH", Some (temp_dir ())) ]
                  "find_llvm_objdump" [])));
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
          | Some lib ->
              ignore (llvm_objdump ());
              ignore (output ());
              amdgpu_disassemble lib;
              contains ~sub:"s_endpgm" (last_line (output ())));
      test "amdgpu_disassemble fails on bytes llvm-objdump cannot read"
        (fun () ->
          ignore (llvm_objdump ());
          raises_match (Exn.failure ~sub:"llvm-objdump") (fun () ->
              amdgpu_disassemble "not an object"));
    ]

(* run_program *)

let script text =
  let file = temp_file ~suffix:".sh" () in
  write file text;
  file

let pp_run ppf = function
  | Ok (Some (Unix.WEXITED n)) -> Format.fprintf ppf "Ok (Some (WEXITED %d))" n
  | Ok (Some (WSIGNALED n)) -> Format.fprintf ppf "Ok (Some (WSIGNALED %d))" n
  | Ok (Some (WSTOPPED n)) -> Format.fprintf ppf "Ok (Some (WSTOPPED %d))" n
  | Ok None -> Format.pp_print_string ppf "Ok None"
  | Error e -> Format.fprintf ppf "Error %S" e

let run_w = Testable.make ~pp:pp_run ~equal:( = )

(* A shell started from a thread that blocks SIGTERM, as a domain Worker spawns
   does, sends itself SIGTERM: it ends by it unless it started with the signal
   blocked. *)
let starts_unblocked () =
  if Sys.win32 then skip ~reason:"Windows has no signal masks" ();
  let out = temp_file () in
  let old = Unix.sigprocmask Unix.SIG_BLOCK [ Sys.sigterm ] in
  let status =
    Fun.protect
      ~finally:(fun () -> ignore (Unix.sigprocmask Unix.SIG_SETMASK old))
      (fun () ->
        run_program
          ~stdin:(script "kill -TERM $$\necho blocked\n")
          ~stdout:out ~stderr:out "sh" [])
  in
  equal ~msg:(read out) run_w (Ok (Some (WSIGNALED Sys.sigterm))) status

(* A shell that records its process, signals its parent and sleeps: the
   handler's exception abandons the wait. The signal is held until the wait,
   however early the shell sends it. *)
let abandons () =
  if Sys.win32 then skip ~reason:"Windows has no SIGUSR1" ();
  let pid_file = temp_file () and out = temp_file () in
  let sh =
    script
      (Printf.sprintf "echo $$ > %s\nkill -USR1 $PPID\nexec sleep 30\n"
         (Filename.quote pid_file))
  in
  let handler = Sys.Signal_handle (fun _ -> raise Exit) in
  let previous = Sys.signal Sys.sigusr1 handler in
  Fun.protect
    ~finally:(fun () -> Sys.set_signal Sys.sigusr1 previous)
    (fun () ->
      raises Exit (fun () ->
          run_program ~stdin:sh ~stdout:out ~stderr:out "sh" []));
  let pid = int_of_string (String.trim (read pid_file)) in
  raises_match
    (function Unix.Unix_error (ESRCH, _, _) -> true | _ -> false)
    (fun () -> Unix.kill pid 0)

let run_program_group =
  group "run_program"
    [
      test "is the status the program ends with" (fun () ->
          let out = temp_file () in
          equal run_w (Ok (Some (WEXITED 3)))
            (run_program ~stdin:(script "exit 3\n") ~stdout:out ~stderr:out "sh"
               []));
      test "is Ok None when the status is lost, as when SIGCHLD is ignored"
        (fun () ->
          if Sys.win32 then skip ~reason:"Windows has no SIGCHLD" ();
          let out = temp_file () in
          let previous = Sys.signal Sys.sigchld Sys.Signal_ignore in
          equal run_w (Ok None)
            (Fun.protect
               ~finally:(fun () -> Sys.set_signal Sys.sigchld previous)
               (fun () ->
                 run_program ~stdin:(script "exit 0\n") ~stdout:out ~stderr:out
                   "sh" [])));
      test "empties the files of standard output and standard error first"
        (fun () ->
          let out = temp_file () and err = temp_file () in
          write out "an earlier output, longer";
          write err "an earlier error, longer";
          ignore
            (require_ok
               (run_program
                  ~stdin:(script "echo out; echo err 1>&2\n")
                  ~stdout:out ~stderr:err "sh" []));
          equal (pair string string) ("out\n", "err\n") (read out, read err));
      test "writes standard output and standard error in order to one file"
        (fun () ->
          let out = temp_file () in
          ignore
            (require_ok
               (run_program
                  ~stdin:(script "echo 1; echo 2 1>&2; echo 3; echo 4 1>&2\n")
                  ~stdout:out ~stderr:out "sh" []));
          equal string "1\n2\n3\n4\n" (read out));
      test "starts the program with no signal blocked" starts_unblocked;
      test
        "kills and reaps a program whose wait a handler's exception cuts short"
        abandons;
      test "is Error naming a program that cannot start" (fun () ->
          let out = temp_file () in
          contains ~sub:"tolk-no-such-program"
            (require_error
               ~pp:(Testable.pp (option status))
               (run_program ~stdout:out ~stderr:out "tolk-no-such-program" [])));
      test "raises Unix_error when a file cannot be opened" (fun () ->
          let out = temp_file () in
          let missing = Filename.concat (temp_dir ()) "missing" in
          raises_match
            (function Unix.Unix_error (ENOENT, _, _) -> true | _ -> false)
            (fun () ->
              run_program ~stdin:missing ~stdout:out ~stderr:out "sh" []));
    ]

(* Memo *)

let memo_keys = [| "a"; "b" |]

let key_index =
  Gen.of_list
    ~pp:(fun ppf i -> Format.pp_print_string ppf memo_keys.(i))
    [ 0; 1 ]

(* A memo, and per key how many times its [make] has run; its model, the values
   of its keys. *)
let memo = abstract "memo"

(* A make that counts its runs and is the count before its own: [0] in every
   call iff it runs once per key. It spins a little, so that two domains making
   one key overlap. *)
let counting made i () =
  for _ = 1 to 1000 do
    Domain.cpu_relax ()
  done;
  Atomic.fetch_and_add made.(i) 1

let value_of model i = Hashtbl.find_opt model memo_keys.(i)

let memo_model =
  [
    command "create"
      (Gen.unit @-> makes memo)
      (fun () -> Hashtbl.create 2)
      (fun () -> (Memo.create (), Array.init 2 (fun _ -> Atomic.make 0)));
    command "get"
      (key_index @-> memo ^-> returns int)
      (fun i model ->
        match value_of model i with
        | Some v -> v
        | None ->
            Hashtbl.replace model memo_keys.(i) 0;
            0)
      (fun i (m, made) -> Memo.get m memo_keys.(i) (counting made i));
    command "get with a make that raises"
      (key_index @-> memo ^-> returns int)
      (fun i model ->
        match value_of model i with Some v -> v | None -> raise Exit)
      (fun i (m, _) -> Memo.get m memo_keys.(i) (fun () -> raise Exit));
    command "try_get"
      (key_index @-> memo ^-> judges (option int))
      (fun i model outcome ->
        match outcome with
        | Ok (Some v) ->
            equal int (Option.value (value_of model i) ~default:0) v;
            Hashtbl.replace model memo_keys.(i) v
        | Ok None ->
            equal ~msg:"None only while the key has no value" (option int) None
              (value_of model i)
        | Error e -> raise e)
      (fun i (m, made) -> Memo.try_get m memo_keys.(i) (counting made i));
  ]

(* Another domain makes the key's value until [release] is set. *)
let held_try_get () =
  let m = Memo.create () in
  let started = Atomic.make false and release = Atomic.make false in
  let maker =
    Domain.spawn (fun () ->
        Memo.get m "k" (fun () ->
            Atomic.set started true;
            while not (Atomic.get release) do
              Domain.cpu_relax ()
            done;
            1))
  in
  while not (Atomic.get started) do
    Domain.cpu_relax ()
  done;
  let during = Memo.try_get m "k" (fun () -> 2) in
  Atomic.set release true;
  let made = Domain.join maker in
  equal
    (triple (option int) int (option int))
    (None, 1, Some 1)
    (during, made, Memo.try_get m "k" (fun () -> 2))

let memo_group =
  group "Memo"
    [
      (* windtrap 0.3.1 leaves each command out of a case with probability 1/4;
         with up to 20 calls around the branches, no case of 20 calls a given
         command with odds 0.282^20, 3e-11 over the three that run in branches.
         The rate is windtrap 0.3.1's draw, which windtrap.mli leaves out of its
         contract: a new windtrap needs the count checked again. *)
      stateful ~count:20 ~domains:2
        "get makes each key's value once, whatever the order of the calls"
        memo_model;
      test "get after a make that raised makes the value" (fun () ->
          let m = Memo.create () in
          raises Exit (fun () -> Memo.get m "k" (fun () -> raise Exit));
          equal int 1 (Memo.get m "k" (fun () -> 1)));
      test "try_get makes a key no one is making, which then holds its value"
        (fun () ->
          let m = Memo.create () in
          let made = Memo.try_get m "k" (fun () -> 7) in
          equal
            (pair (option int) int)
            (Some 7, 7)
            (made, Memo.get m "k" (fun () -> 8)));
      test
        "try_get is None, without waiting, while another domain makes the key"
        held_try_get;
    ]

let () =
  match Sys.getenv_opt role with
  | Some r -> play r
  | None ->
      exit
        (run "Tolk.Helpers"
           [
             targets;
             integers;
             terminal_text;
             cache_dir_group;
             safe_dir_group;
             open_owned_group;
             cachedb_group;
             diskcache;
             programs;
             run_program_group;
             memo_group;
           ])
