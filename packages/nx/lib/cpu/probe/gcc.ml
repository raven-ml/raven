(* gcc.ml CC...: prints gcc's vectoriser settings (lib/cpu/dune) when the C
   compiler CC is gcc, as a native or a cross compiler names it, and accepts
   them, else no flags, as an S-expression for dune's :include. *)

let flags =
  [ "-fvect-cost-model=dynamic"; "--param=max-tree-if-conversion-phi-args=16" ]

(* The declaration keeps the unit from being empty, which -Wpedantic -Werror
   refuses. *)
let source =
  "#if !defined(__GNUC__) || defined(__clang__)\n\
   #error not gcc\n\
   #endif\n\
   int f(void);\n"

let () =
  let cc = Array.to_list (Array.sub Sys.argv 1 (Array.length Sys.argv - 1)) in
  let c = Filename.temp_file "gcc" ".c" in
  let o = Filename.temp_file "gcc" ".o" in
  Out_channel.with_open_text c (fun oc -> output_string oc source);
  let cmd =
    Filename.quote_command (List.hd cc) ~stdout:Filename.null
      ~stderr:Filename.null
      (List.tl cc @ flags @ [ "-c"; c; "-o"; o ])
  in
  let gcc = Sys.command cmd = 0 in
  Sys.remove c;
  if Sys.file_exists o then Sys.remove o;
  print_string (if gcc then "(" ^ String.concat " " flags ^ ")" else "()")
