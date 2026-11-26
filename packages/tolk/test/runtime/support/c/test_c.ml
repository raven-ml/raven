open Windtrap
open Tolk

(* Trees of files *)

let elf = "\x7fELF" ^ String.make 60 '\000'
let linker_script = "INPUT(-lc)\n"

(* [under root rel] is [rel], a relative path written with [/], under [root],
   each part joined as the system joins them. *)
let under root rel =
  List.fold_left Filename.concat root (String.split_on_char '/' rel)

let rec mkdir_p dir =
  if not (Sys.file_exists dir) then (
    mkdir_p (Filename.dirname dir);
    Unix.mkdir dir 0o755)

let write file contents =
  Out_channel.with_open_bin file (fun oc -> output_string oc contents)

(* [make root (path, kind)] makes [path] under [root] as [kind]: [elf], [text],
   [dir] or [@target], a symbolic link to [target]. *)
let make root (path, kind) =
  let file = under root path in
  mkdir_p (Filename.dirname file);
  match kind with
  | "elf" -> write file elf
  | "text" -> write file linker_script
  | "dir" -> Unix.mkdir file 0o755
  | link when link.[0] = '@' ->
      Unix.symlink (String.sub link 1 (String.length link - 1)) file
  | kind -> invalid_arg ("no entry kind " ^ kind)

let entry s =
  match String.index_opt s '=' with
  | Some i -> (String.sub s 0 i, String.sub s (i + 1) (String.length s - i - 1))
  | None -> invalid_arg ("no entry " ^ s)

(* The golden's cells: [-] is no word. *)
let words = function "-" -> [] | cell -> String.split_on_char ' ' cell

(* The platform whose search findlib runs, as the golden names it. *)
let platform =
  match Platform.system with
  | "macosx" -> "darwin"
  | "linux" -> "linux"
  | system -> system

let on platform' =
  if platform' <> platform then
    skip ~reason:(Printf.sprintf "%s searches as %s" platform platform') ()

(* The variable that overrides the search for [lib]. *)
let name_path lib =
  String.uppercase_ascii (String.map (function '-' -> '_' | c -> c) lib)
  ^ "_PATH"

(* Names of libraries that no system holds, a new one per call. *)
let fresh =
  let n = ref 0 in
  fun () ->
    incr n;
    Printf.sprintf "tolktest%d" !n

(* multiarch *)

(* Debian's multiarch tuple of each machine [uname -m] names, from Debian's
   table of tuples (wiki.debian.org/Multiarch/Tuples), for the machines OCaml
   compiles natively for on Linux. *)
let debian_tuples =
  [
    ("x86_64", "x86_64-linux-gnu");
    ("aarch64", "aarch64-linux-gnu");
    ("ppc64le", "powerpc64le-linux-gnu");
    ("riscv64", "riscv64-linux-gnu");
    ("s390x", "s390x-linux-gnu");
  ]

let machine () =
  match Unix.open_process_args_in "uname" [| "uname"; "-m" |] with
  | exception Unix.Unix_error _ -> skip ~reason:"uname does not run" ()
  | ic ->
      let m = String.trim (In_channel.input_all ic) in
      ignore (Unix.close_process_in ic);
      m

let multiarch =
  group "multiarch"
    [
      test "is Debian's tuple of the host's machine" (fun () ->
          on "linux";
          let m = machine () in
          match List.assoc_opt m debian_tuples with
          | None -> skip ~reason:("no Debian tuple listed for " ^ m) ()
          | Some tuple -> equal string tuple C.multiarch);
    ]

(* findlib *)

let finds_as_golden cell =
  on (cell "platform");
  let root = temp_dir () and lib = cell "lib" in
  List.iter (fun e -> make root (entry e)) (words (cell "tree"));
  setenv (name_path lib)
    (match cell "name_path" with
    | "-" -> None
    | {|""|} -> Some ""
    | path -> Some (under root path));
  setenv "LD_LIBRARY_PATH"
    (match cell "ld_library_path" with
    | "-" -> None
    | dirs ->
        String.split_on_char ':' dirs
        |> List.map (function "" -> "" | d -> under root d)
        |> String.concat ":" |> Option.some);
  (* An absolute path of the golden is one under [root]. *)
  let paths =
    List.map
      (fun p ->
        if p.[0] = '/' then under root (String.sub p 1 (String.length p - 1))
        else p)
      (words (cell "paths"))
  in
  let extra_paths = List.map (under root) (words (cell "extra_paths")) in
  let expected =
    match cell "found" with "None" -> None | found -> Some (under root found)
  in
  equal (option string) expected (C.findlib ~extra_paths lib paths)

let first_elf_in_name_order () =
  if platform = "darwin" then skip ~reason:"macOS takes no ELF file" ();
  let root = temp_dir () and lib = fresh () in
  let so v = Printf.sprintf "lib%s.so%s" lib v in
  List.iter (fun v -> make root ("d/" ^ so v, "elf")) [ ".2"; ".10"; ".1.5" ];
  equal (option string)
    (Some (under root ("d/" ^ so ".1.5")))
    (C.findlib ~extra_paths:[ under root "d" ] lib [ lib ])

(* A tree of directories [d0] to [d3], each holding some of a library's
   candidate names on the platform, as files or directories, and two lists of
   directories to search, [d9] missing. A candidate is a prefix, a suffix and
   the kinds of entry it is made as. *)
let candidates =
  match platform with
  | "darwin" ->
      [
        ("lib", ".dylib", [ "text"; "dir" ]);
        ("", ".dylib", [ "text" ]);
        ("", "", [ "text"; "dir" ]);
      ]
  | _ ->
      [
        ("lib", ".so", [ "elf"; "text"; "dir" ]);
        ("lib", ".so.1", [ "elf"; "text" ]);
        ("lib", ".so.1a", [ "elf" ]);
      ]

let candidate lib c =
  let prefix, suffix, _ = List.nth candidates c in
  prefix ^ lib ^ suffix

type layout = {
  files : (int * int * string) list;  (** Directory, candidate, kind. *)
  searches : int list * int list;
}

let pp_layout ppf l =
  let pp_dirs =
    Format.(pp_print_list ~pp_sep:pp_print_space (fun ppf -> fprintf ppf "d%d"))
  in
  List.iter
    (fun (d, c, kind) ->
      Format.fprintf ppf "d%d/%s=%s@ " d (candidate "P" c) kind)
    l.files;
  Format.fprintf ppf "searching [%a] then [%a]" pp_dirs (fst l.searches) pp_dirs
    (snd l.searches)

let gen_layout =
  let open Gen in
  let file =
    let* d = int_range 0 3 and+ c = int_range 0 (List.length candidates - 1) in
    let _, _, kinds = List.nth candidates c in
    let+ kind = of_list kinds in
    (d, c, kind)
  in
  (* Few entries are a library: on Linux only an ELF [libp.so] or [libp.so.1],
     about one entry in four. The tree holds up to a dozen entries and a search
     draws a directory once more for each entry it holds, so that on every
     platform each cover of the law below holds in one case in seven or more. *)
  (let* files = list ~size:(int_range 0 12) file in
   (* A name is made once in a directory. *)
   let files =
     List.sort_uniq (fun (d, c, _) (d', c', _) -> compare (d, c) (d', c')) files
   in
   let dirs = [ 0; 1; 2; 3; 9 ] @ List.map (fun (d, _, _) -> d) files in
   let search = list ~size:(int_range 0 4) (of_list dirs) in
   let+ searches = pair search search in
   { files; searches })
  |> with_pp pp_layout

(* A library in directory [d]: the platform's first candidate, as its first
   kind. *)
let library_in d =
  let _, _, kinds = List.hd candidates in
  (d, 0, List.hd kinds)

(* The layouts each cover needs, run whatever the seed. *)
let found_second_only = { files = [ library_in 1 ]; searches = ([ 0 ], [ 1 ]) }

let found_in_both =
  { files = [ library_in 0; library_in 1 ]; searches = ([ 0 ], [ 1 ]) }

let first_of_directories l =
  setenv "LD_LIBRARY_PATH" None;
  let root = temp_dir () and lib = fresh () in
  List.iter
    (fun (d, c, kind) ->
      make root (Printf.sprintf "d%d/%s" d (candidate lib c), kind))
    l.files;
  let dirs =
    List.map (fun d -> Filename.concat root (Printf.sprintf "d%d" d))
  in
  let find extra_paths = C.findlib ~extra_paths lib [ lib ] in
  let first f f' = match f with Some _ -> f | None -> f' in
  let a, b = (dirs (fst l.searches), dirs (snd l.searches)) in
  let found_a = find a <> None and found_b = find b <> None in
  cover "found in the second search only" ((not found_a) && found_b);
  cover "found in both searches" (found_a && found_b);
  Law.homomorphic (list string) (option string) find ( @ ) first (a, b)

let findlib =
  group "findlib"
    [
      group "finds in each tree the file the golden names"
        [
          Golden.cases ~key:[ "platform"; "case" ] "findlib_trees.golden"
            finds_as_golden;
        ];
      test "the first ELF file in the order of names is found"
        first_elf_in_name_order;
      prop "searching two lists of directories finds the first of each search"
        ~examples:[ found_second_only; found_in_both ]
        gen_layout first_of_directories;
      test "a name nowhere is not found" (fun () ->
          let lib = fresh () in
          equal (option string) None (C.findlib lib [ lib ]));
      test "no paths find nothing, even with a directory that holds the name"
        (fun () ->
          let root = temp_dir () and lib = fresh () in
          make root ("d/lib" ^ lib ^ ".dylib", "text");
          make root ("d/lib" ^ lib ^ ".so", "elf");
          equal (option string) None
            (C.findlib ~extra_paths:[ under root "d" ] lib []));
      test "MTLCompiler is found among macOS's private frameworks" (fun () ->
          on "darwin";
          equal (option string)
            (Some
               "/System/Library/PrivateFrameworks/MTLCompiler.framework/MTLCompiler")
            (C.findlib "MTLCompiler" [ "MTLCompiler" ]));
      (* Every Linux with a C toolchain has a shared C library: glibc's
         [libc.so.6], under [/lib/x86_64-linux-gnu] on Debian, or musl's
         [libc.so]. musl has no math library apart from it. *)
      test "the C library is found among Linux's library directories" (fun () ->
          on "linux";
          let file = require_some (C.findlib "c" [ "c" ]) in
          starts_with ~prefix:"libc.so" (Filename.basename file));
    ]

(* identity *)

(* A second of 2023, and times a tenth and four tenths into it, which a time
   kept in whole seconds, rounded or cut, tells apart from none. *)
let second = 1_700_000_000.
let early = second +. 0.1
let late = second +. 0.4

(* [library ~at] is a new library file, last modified at [at]. *)
let library ?(at = second) () =
  let file = Filename.concat (temp_dir ()) "libx.so" in
  write file elf;
  Unix.utimes file at at;
  file

let identity =
  group "identity"
    [
      test "no library is none" (fun () ->
          equal string "none" (C.identity None));
      test "a library names its path" (fun () ->
          let file = library () in
          contains ~sub:file (C.identity (Some file)));
      test "a library rewritten at another size changes its identity" (fun () ->
          let file = library () in
          let before = C.identity (Some file) in
          write file (elf ^ elf);
          Unix.utimes file second second;
          not_equal string before (C.identity (Some file)));
      test
        "a library rewritten at its size within a second changes its identity"
        (fun () ->
          let file = library ~at:early () in
          if Float.is_integer (Unix.stat file).st_mtime then
            skip ~reason:"the file system keeps whole seconds" ();
          let before = C.identity (Some file) in
          Unix.utimes file late late;
          not_equal string before (C.identity (Some file)));
      test "a library replaced by a rename changes its identity" (fun () ->
          let file = library () in
          let before = C.identity (Some file) in
          let copy = file ^ ".new" in
          write copy elf;
          Unix.utimes copy second second;
          Sys.rename copy file;
          not_equal string before (C.identity (Some file)));
      test "a library that cannot be found is its path" (fun () ->
          let absent = Filename.concat (temp_dir ()) "libx.so" in
          equal string absent (C.identity (Some absent)));
    ]

let () = exit (run "Tolk.C" [ multiarch; findlib; identity ])
