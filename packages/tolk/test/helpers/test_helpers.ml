open Windtrap
open Tolk
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

(* Processes

   What a process reads when it starts is observed in a child: this suite,
   started again with [role] set, plays that role and exits. *)

let role = "TOLK_TEST_HELPERS_ROLE"

let play = function
  | "cache" -> Printf.printf "cache_dir=%s\ncachedb=%s\n" cache_dir cachedb
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
  | "fill" ->
      for i = 1 to 4 do
        Diskcache.put ~table:Sys.argv.(1)
          (Sys.argv.(2) ^ string_of_int i)
          (String.make 4096 'v')
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
  (* in text mode: on Windows a child's lines end in CRLF *)
  let read file = In_channel.with_open_text file In_channel.input_all in
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

let cache_read env =
  let pair line =
    match String.index_opt line '=' with
    | Some i ->
        Some
          ( String.sub line 0 i,
            String.sub line (i + 1) (String.length line - i - 1) )
    | None -> None
  in
  List.filter_map pair
    (String.split_on_char '\n' (succeeds (child ~env "cache" [])))

let read_at_start env expected =
  let read = cache_read env in
  equal
    (list (pair string (option string)))
    (List.map (fun (k, v) -> (k, Some v)) expected)
    (List.map (fun (k, _) -> (k, List.assoc_opt k read)) expected)

let macos () =
  let uname = Unix.open_process_args_in "uname" [| "uname"; "-s" |] in
  Fun.protect
    ~finally:(fun () -> ignore (Unix.close_process_in uname))
    (fun () -> String.equal "Darwin" (String.trim (In_channel.input_all uname)))

(* [path dir names] is [names] joined under [dir] by the system's separator. *)
let path dir names = List.fold_left Filename.concat dir names

let cache_directory =
  group "cache directory"
    [
      test "puts the cache in tolk under XDG_CACHE_HOME" (fun () ->
          read_at_start
            [ ("XDG_CACHE_HOME", Some "/xdg"); ("CACHEDB", None) ]
            [
              ("cache_dir", path "/xdg" [ "tolk" ]);
              ("cachedb", path "/xdg" [ "tolk"; "cache" ]);
            ]);
      test
        "puts the cache in the platform's cache directory without \
         XDG_CACHE_HOME" (fun () ->
          let home =
            path "/home"
              [ (if macos () then "Library/Caches" else ".cache"); "tolk" ]
          in
          (* Windows' home is USERPROFILE, elsewhere it is HOME *)
          read_at_start
            [
              ("XDG_CACHE_HOME", None);
              ("HOME", Some "/home");
              ("USERPROFILE", Some "/home");
              ("CACHEDB", None);
            ]
            [ ("cache_dir", home); ("cachedb", path home [ "cache" ]) ]);
      test "makes the default cache directory absolute" (fun () ->
          read_at_start
            [ ("XDG_CACHE_HOME", Some "relative"); ("CACHEDB", None) ]
            [
              ("cachedb", path (Sys.getcwd ()) [ "relative"; "tolk"; "cache" ]);
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

let with_color f = Setting.context [ B (Setting.no_color, false) ] f

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
          Setting.context
            [ B (Setting.no_color, true) ]
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

(* The entries of the cache: the files of its directories. *)
let entries_on_disk () =
  if not (Sys.file_exists cachedb) then []
  else
    List.concat_map
      (fun name ->
        let dir = Filename.concat cachedb name in
        if Sys.is_directory dir then files dir else [])
      (Array.to_list (Sys.readdir cachedb))

(* Runs [damage] on every entry of the cache when it holds one, then reads the
   entry back. *)
let damaged damage =
  Diskcache.clear ();
  let table = fresh_table () in
  Diskcache.put ~table "k" (String.make 4096 'v');
  List.iter damage (entries_on_disk ());
  (match Diskcache.get ~table "k" with
  | exception Failure _ -> ()
  | None -> fail "get is None"
  | Some v -> failf "get is a value of %d bytes" (String.length v));
  Diskcache.clear ()

(* [put_entry ~table key v] puts [v] at [key] in [table], a key not yet there,
   and is the file of its entry. *)
let put_entry ~table key v =
  let before = entries_on_disk () in
  Diskcache.put ~table key v;
  List.find (fun f -> not (List.mem f before)) (entries_on_disk ())

(* The directory of [table]'s entries, found as the one its first entry
   creates. *)
let table_directory table = Filename.dirname (put_entry ~table "k" "v")

(* Files the cache did not write, some named like its own: a table directory
   holding a file whose name is no digest, directories whose names are not a
   digest and a version, a plain file where a table directory would be, and a
   file named [size]. *)
let keeps_foreign_files () =
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
  List.iter write foreign;
  Diskcache.put ~table:(fresh_table ()) "k" "v";
  Diskcache.clear ();
  let kept f =
    Sys.file_exists f
    && In_channel.with_open_bin f In_channel.input_all = "foreign"
  in
  equal (list bool) (List.map (fun _ -> true) foreign) (List.map kept foreign);
  List.iter Sys.remove foreign

(* The bytes the files under [cachedb] hold. *)
let cache_bytes () =
  if not (Sys.file_exists cachedb) then 0
  else List.fold_left (fun n f -> n + (Unix.stat f).st_size) 0 (files cachedb)

let sized limit f = Setting.context [ B (Setting.cachesize, limit) ] f
let limits = Gen.of_list ~pp:Format.pp_print_int [ 1; 100; 4096; 1 lsl 16 ]

let lengths =
  Gen.list ~size:(Gen.int_range 1 40)
    (Gen.frequency [ (4, Gen.int_range 0 4096); (1, Gen.int_range 0 70_000) ])

let holds_its_size (limit, lengths) =
  Diskcache.clear ();
  let table = fresh_table () in
  cover "past the bound" (List.fold_left ( + ) 0 lengths > limit);
  sized limit (fun () ->
      List.iteri
        (fun i n ->
          Diskcache.put ~table (string_of_int i) (String.make n 'v');
          at_most int ~than:limit (cache_bytes ()))
        lengths);
  Diskcache.clear ()

(* [ago file s] sets [file]'s times [s] seconds back. *)
let ago file s =
  let t = Unix.gettimeofday () -. s in
  Unix.utimes file t t

(* Puts 4 KiB entries until one shrinks the cache, within 64 KiB. *)
let fill_until_trimmed table =
  let v = String.make 4096 'v' in
  let rec fill i before =
    Diskcache.put ~table (string_of_int i) v;
    let after = cache_bytes () in
    if after >= before && i < 100 then fill (i + 1) after
  in
  fill 0 (cache_bytes ())

(* Use is ordered by file time: "read" is used an hour ago and read back now,
   "unread" used half an hour ago. *)
let keeps_recently_used () =
  Diskcache.clear ();
  let table = fresh_table () and v = String.make 4096 'v' in
  sized (1 lsl 16) (fun () ->
      ago (put_entry ~table "read" v) 3600.;
      ago (put_entry ~table "unread" v) 1800.;
      ignore (require_some (Diskcache.get ~table "read"));
      fill_until_trimmed table;
      let kept key = Option.is_some (Diskcache.get ~table key) in
      equal (pair bool bool) (false, true) (kept "unread", kept "read"));
  Diskcache.clear ()

(* [write_aside table bytes s] is a file named like an entry of [table] being
   written, of [bytes] bytes, last written [s] seconds ago. *)
let write_aside table bytes s =
  let write =
    Filename.concat (table_directory table) (String.make 32 'a' ^ ".1.tmp")
  in
  Out_channel.with_open_bin write (fun oc ->
      output_string oc (String.make bytes 'v'));
  ago write s;
  write

(* A write a killed process left, last written two hours ago. *)
let removes_stale_writes () =
  Diskcache.clear ();
  let table = fresh_table () and limit = 1 lsl 16 in
  ignore (write_aside table (limit / 2) 7200.);
  sized limit (fun () -> fill_until_trimmed table);
  at_most int ~than:limit (cache_bytes ());
  Diskcache.clear ()

(* A write another process has in progress, last written a minute ago. *)
let keeps_writes_in_progress () =
  Diskcache.clear ();
  let table = fresh_table () in
  let write = write_aside table 4096 60. in
  sized (1 lsl 16) (fun () -> fill_until_trimmed table);
  equal bool true (Sys.file_exists write);
  Sys.remove write;
  Diskcache.clear ()

(* The file in which the puts of all processes count the cache's size. *)
let size_file = Filename.concat cachedb "f7bd60b75b29d79b660a2859395c1a24_size"

(* No put can count while a directory stands at the size's file. *)
let stores_uncounted () =
  Diskcache.clear ();
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
  equal (option string) (Some "v") stored;
  Diskcache.clear ()

(* A put into a table whose directory the process may not write fails, as a put
   on a full disk does. Entries used up to 16 minutes ago fill the cache past a
   bound of 64 KiB, the oldest under the key "0". *)
let trims_before_writing () =
  if Sys.win32 || Unix.geteuid () = 0 then
    skip ~reason:"every directory is writable" ()
  else begin
    Diskcache.clear ();
    let full = fresh_table () and locked = fresh_table () in
    let v = String.make 4096 'v' and dir = table_directory locked in
    for i = 0 to 15 do
      ago
        (put_entry ~table:full (string_of_int i) v)
        (float_of_int ((16 - i) * 60))
    done;
    Unix.chmod dir 0o555;
    Fun.protect
      ~finally:(fun () -> Unix.chmod dir 0o755)
      (fun () ->
        raises_match
          (function Sys_error _ -> true | _ -> false)
          (fun () ->
            sized (1 lsl 16) (fun () -> Diskcache.put ~table:locked "k2" v)));
    equal (option string) None (Diskcache.get ~table:full "0");
    Diskcache.clear ()
  end

(* A write in progress a minute old stays through a clear, one two hours old
   does not. *)
let clear_keeps_writes_in_progress () =
  let kept = Sys.file_exists in
  let young = write_aside (fresh_table ()) 4096 60. in
  let stale = write_aside (fresh_table ()) 4096 7200. in
  Diskcache.clear ();
  equal (pair bool bool) (true, false) (kept young, kept stale);
  Sys.remove young;
  Diskcache.clear ()

(* An entry that cannot be removed: on Windows one held open without sharing
   deletion, elsewhere one in a table directory the process may not write. *)
let clear_leaves_held_entries () =
  if (not Sys.win32) && Unix.geteuid () = 0 then
    skip ~reason:"every directory is writable" ()
  else begin
    let held = fresh_table () and free = fresh_table () in
    let entry = put_entry ~table:held "k" "v" in
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
    equal (option string) None (Diskcache.get ~table:free "k");
    Diskcache.clear ()
  end

(* The directory of a table stays through a clear, so that a put in another
   process that has made it can still write there. *)
let clear_keeps_tables () =
  let dir = table_directory (fresh_table ()) in
  Diskcache.clear ();
  equal bool true (Sys.file_exists dir && Sys.is_directory dir)

(* A read and a mark hold an entry open, shared for deletion, while a put
   renames a new entry over it: on Windows a handle that does not share deletion
   makes that rename fail. *)
let put_over_a_held_entry () =
  let table = fresh_table () in
  let entry = put_entry ~table "k" "old" in
  let fd = Unix.openfile entry [ O_RDONLY; O_SHARE_DELETE ] 0 in
  Fun.protect
    ~finally:(fun () -> Unix.close fd)
    (fun () -> Diskcache.put ~table "k" "new");
  equal (option string) (Some "new") (Diskcache.get ~table "k")

let unbounded () =
  Diskcache.clear ();
  let table = fresh_table () and v = String.make 4096 'v' in
  let keys = List.init 32 string_of_int in
  sized 0 (fun () -> List.iter (fun k -> Diskcache.put ~table k v) keys);
  equal
    (list (option string))
    (List.map (fun _ -> Some v) keys)
    (List.map (Diskcache.get ~table) keys);
  Diskcache.clear ()

let refuses_negative_size () =
  let ended =
    child ~env:[ ("CACHESIZE", Some "-1") ] "put" [ fresh_table (); "k"; "v" ]
  in
  equal ~msg:ended.err status (Unix.WEXITED 2) ended.status;
  contains ~sub:"CACHESIZE" ended.err;
  refuses (fun () ->
      sized (-1) (fun () -> Diskcache.put ~table:(fresh_table ()) "k" "v"))

(* Each process puts less than the bound; together they put more. *)
let processes_share_the_bound () =
  Diskcache.clear ();
  let table = fresh_table () and limit = 1 lsl 16 in
  for i = 1 to 10 do
    ignore
      (succeeds
         (child
            ~env:[ ("CACHESIZE", Some (string_of_int limit)) ]
            "fill"
            [ table; string_of_int i ]));
    at_most int ~than:limit (cache_bytes ())
  done;
  Diskcache.clear ()

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
let disabled f = Setting.context [ B (Setting.cachelevel, 0) ] f

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
          test "put and clear keep the files that are not entries"
            keeps_foreign_files;
          test "clear leaves the entries it cannot remove"
            clear_leaves_held_entries;
          test "clear keeps the directories of the tables" clear_keeps_tables;
          test "a put replaces an entry that a read holds open"
            put_over_a_held_entry;
          test "clear keeps another process's write in progress"
            clear_keeps_writes_in_progress;
          cases ~name:(Printf.sprintf "cachelevel=%d")
            "a disabled cache neither reads nor writes" [ 0; -1 ] (fun level ->
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
          prop ~count:30 "the files of the cache never hold more than cachesize"
            (Gen.pair limits lengths) holds_its_size;
          test "a put past cachesize removes entries used least recently first"
            keeps_recently_used;
          test "a put past cachesize removes the writes a killed process left"
            removes_stale_writes;
          test "a put past cachesize keeps another process's write in progress"
            keeps_writes_in_progress;
          test "a put that cannot write removes entries past cachesize first"
            trims_before_writing;
          test "a put that cannot count the cache's size stores its entry"
            stores_uncounted;
          test "a cachesize of 0 keeps every entry" unbounded;
          test "a negative cachesize is refused" refuses_negative_size;
          test "processes putting in turn keep the cache within cachesize"
            processes_share_the_bound;
          test "get fails on a truncated entry" (fun () -> damaged truncate);
          test "get fails on an entry that is no entry" (fun () ->
              damaged overwrite);
        ]

(* Calls on two domains, each branch on its own: an entry is put, read and
   cleared while another domain puts, reads and clears. *)
let diskcache_domains =
  match Sys.getenv_opt "CACHEDB" with
  | None -> group "Diskcache on two domains" []
  | Some _ ->
      group "Diskcache on two domains"
        [
          stateful ~count:4 ~domains:2
            "behaves as a table of entries, whatever the order of the calls"
            model;
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
          ignore
            (Setting.context
               [ B (Setting.debug, 1) ]
               (fun () -> system "echo hello"));
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
             cache_directory;
             targets;
             integers;
             selection;
             terminal_text;
             diskcache;
             programs;
             diskcache_domains;
           ])
