(* v3.ml CC...: prints the flags that compile nx.cpu's x86-64-v3 units
   (lib/cpu/dune) with the C compiler CC, as an S-expression for dune's
   :include.

   gcc for Windows or Cygwin also takes -Wa,-muse-unaligned-vector-move.
   It keeps 32-byte vectors on the stack at addresses it assumes 32-byte
   aligned, where the Windows x64 ABI aligns the stack to 16 bytes only
   (gcc bug 54412), and an aligned move to such an address faults. The
   assembler's option (binutils 2.38) writes each aligned vector move as
   an unaligned one, which runs as fast on aligned data. Such a gcc whose
   assembler refuses it fails the build: without it, the units crash. *)

let v3 = [ "-mavx2"; "-mfma"; "-mf16c"; "-mbmi2" ]
let unaligned = "-Wa,-muse-unaligned-vector-move"

(* The declaration keeps the unit from being empty, which -Wpedantic -Werror
   refuses. *)
let windows_gcc =
  "#if defined(__clang__) || !defined(__GNUC__)\n\
   #error not gcc\n\
   #endif\n\
   #if !defined(_WIN32) && !defined(__CYGWIN__)\n\
   #error not for Windows\n\
   #endif\n\
   int f(void);\n"

let unit = "int f(void);\n"

(* Whether [cc] compiles [source] with [flags]. *)
let compiles cc flags source =
  let c = Filename.temp_file "v3" ".c" in
  let o = Filename.temp_file "v3" ".o" in
  Out_channel.with_open_text c (fun oc -> output_string oc source);
  let cmd =
    Filename.quote_command (List.hd cc) ~stdout:Filename.null
      ~stderr:Filename.null
      (List.tl cc @ flags @ [ "-c"; c; "-o"; o ])
  in
  let ok = Sys.command cmd = 0 in
  Sys.remove c;
  if Sys.file_exists o then Sys.remove o;
  ok

let () =
  let cc = Array.to_list (Array.sub Sys.argv 1 (Array.length Sys.argv - 1)) in
  let flags =
    if not (compiles cc [] windows_gcc) then v3
    else if compiles cc (v3 @ [ unaligned ]) unit then v3 @ [ unaligned ]
    else begin
      prerr_endline
        ("nx.cpu: gcc for Windows needs an assembler that takes " ^ unaligned
       ^ " (binutils 2.38 or later)");
      exit 1
    end
  in
  print_string ("(" ^ String.concat " " flags ^ ")")
