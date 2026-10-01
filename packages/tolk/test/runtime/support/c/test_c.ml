open Windtrap
open Tolk

(* Trees of files *)

let elf = "\x7fELF" ^ String.make 60 '\000'
let linker_script = "INPUT(-lc)\n"

let rec mkdir_p dir =
  if not (Sys.file_exists dir) then (
    mkdir_p (Filename.dirname dir);
    Unix.mkdir dir 0o755)

let write file contents =
  Out_channel.with_open_bin file (fun oc -> output_string oc contents)

(* [make root (path, kind)] makes [path] under [root] as [kind]: [elf], [text],
   [dir] or [@target], a symbolic link to [target]. *)
let make root (path, kind) =
  let file = Filename.concat root path in
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

(* Libraries named once per process, since a variable's read may be
   remembered. *)
let fresh =
  let n = ref 0 in
  fun () ->
    incr n;
    Printf.sprintf "tolktest%d" !n

(* tinygrad's findlib *)

let finds_as_tinygrad cell =
  on (cell "platform");
  let root = temp_dir () and lib = cell "lib" in
  List.iter (fun e -> make root (entry e)) (words (cell "tree"));
  let under = Filename.concat root in
  setenv (name_path lib)
    (match cell "name_path" with
    | "-" -> None
    | {|""|} -> Some ""
    | path -> Some (under path));
  setenv "LD_LIBRARY_PATH"
    (match cell "ld_library_path" with
    | "-" -> None
    | dirs ->
        String.split_on_char ':' dirs
        |> List.map (function "" -> "" | d -> under d)
        |> String.concat ":" |> Option.some);
  let paths =
    List.map
      (fun p -> if p.[0] = '/' then root ^ p else p)
      (words (cell "paths"))
  in
  let extra_paths = List.map under (words (cell "extra_paths")) in
  let expected =
    match cell "found" with "None" -> None | found -> Some (under found)
  in
  equal (option string) expected (C.findlib ~extra_paths lib paths)

let tinygrad =
  group "as tinygrad"
    [
      Golden.cases ~key:[ "platform"; "case" ] "findlib_trees.golden"
        finds_as_tinygrad;
    ]

(* Candidates of one directory *)

let first_elf_in_name_order () =
  on "linux";
  let root = temp_dir () and lib = fresh () in
  let so v = Printf.sprintf "lib%s.so%s" lib v in
  List.iter
    (fun name -> make root ("d/" ^ name, "elf"))
    [ so ".2"; so ".10"; so ".1.5" ];
  equal (option string)
    (Some (Filename.concat root ("d/" ^ so ".1.5")))
    (C.findlib ~extra_paths:[ Filename.concat root "d" ] lib [ lib ])

let first_elf_after_a_linker_script () =
  on "linux";
  let root = temp_dir () and lib = fresh () in
  make root (Printf.sprintf "d/lib%s.so" lib, "text");
  make root (Printf.sprintf "d/lib%s.so.1" lib, "elf");
  make root (Printf.sprintf "d/lib%s.so.2" lib, "elf");
  equal (option string)
    (Some (Filename.concat root (Printf.sprintf "d/lib%s.so.1" lib)))
    (C.findlib ~extra_paths:[ Filename.concat root "d" ] lib [ lib ])

let one_directory =
  group "one directory"
    [
      test "the first ELF file in the order of names is found"
        first_elf_in_name_order;
      test "a linker script first in the order of names is passed over"
        first_elf_after_a_linker_script;
    ]

(* The search is the first of its directories *)

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
  let search = list ~size:(int_range 0 3) (of_list [ 0; 1; 2; 3; 9 ]) in
  (let+ files = list ~size:(int_range 0 6) file
   and+ searches = pair search search in
   (* A name is made once in a directory. *)
   let files =
     List.sort_uniq (fun (d, c, _) (d', c', _) -> compare (d, c) (d', c')) files
   in
   { files; searches })
  |> with_pp pp_layout

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
  cover "found in the second search only" (find a = None && find b <> None);
  cover "found in both searches" (find a <> None && find b <> None);
  Law.homomorphic (list string) (option string) find ( @ ) first (a, b)

let search_order =
  group "search order"
    [
      prop "searching two lists of directories finds the first of each search"
        gen_layout first_of_directories;
    ]

(* The absent library *)

let absent =
  group "absence"
    [
      test "a name nowhere is not found" (fun () ->
          let lib = fresh () in
          is_none (C.findlib lib [ lib ]));
      test "no paths find nothing, even with a directory that holds the name"
        (fun () ->
          let root = temp_dir () and lib = fresh () in
          make root ("d/lib" ^ lib ^ ".dylib", "text");
          make root ("d/lib" ^ lib ^ ".so", "elf");
          is_none (C.findlib ~extra_paths:[ Filename.concat root "d" ] lib []));
    ]

(* The system's directories *)

let system =
  group "the system's directories"
    [
      test "MTLCompiler is found among macOS's private frameworks" (fun () ->
          on "darwin";
          equal (option string)
            (Some
               "/System/Library/PrivateFrameworks/MTLCompiler.framework/MTLCompiler")
            (C.findlib "MTLCompiler" [ "MTLCompiler" ]));
      test "the math library is found among Linux's library directories"
        (fun () ->
          on "linux";
          let file = require_some (C.findlib "m" [ "m" ]) in
          starts_with ~affix:"libm.so" (Filename.basename file));
    ]

(* multiarch *)

let multiarch =
  group "multiarch"
    [
      test "names a Linux machine" (fun () ->
          on "linux";
          contains ~sub:"-linux-" C.multiarch);
    ]

let () =
  exit
    (run "Tolk.C"
       [ tinygrad; one_directory; search_order; absent; system; multiarch ])
