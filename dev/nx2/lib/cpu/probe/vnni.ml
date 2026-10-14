(* vnni.ml CC...: prints the flags that enable AVX-VNNI-INT8 when the
   C compiler CC accepts them on a use of VPDPBSSD, else no flags, as an
   S-expression for dune's :include. *)

let flag = "-mavxvnniint8"

let source =
  "#include <immintrin.h>\n\
   __m256i f(__m256i c, __m256i a, __m256i b) {\n\
  \  return _mm256_dpbssd_epi32(c, a, b);\n\
   }\n"

let () =
  let cc = Array.to_list (Array.sub Sys.argv 1 (Array.length Sys.argv - 1)) in
  let c = Filename.temp_file "vnni" ".c" in
  let o = Filename.temp_file "vnni" ".o" in
  Out_channel.with_open_text c (fun oc -> output_string oc source);
  let cmd =
    Filename.quote_command (List.hd cc) ~stdout:Filename.null
      ~stderr:Filename.null
      (List.tl cc @ [ flag; "-c"; c; "-o"; o ])
  in
  let ok = Sys.command cmd = 0 in
  Sys.remove c;
  if Sys.file_exists o then Sys.remove o;
  print_string (if ok then "(" ^ flag ^ ")" else "()")
