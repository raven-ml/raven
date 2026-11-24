(* gcc.ml CC...: prints gcc's vectoriser settings (lib/cpu/dune) when the C
   compiler CC is gcc, as a native or a cross compiler names it, and accepts
   them, else no flags, as an S-expression for dune's :include.

   gcc 12's vectoriser fuses the products and sums of a complex product into
   fused multiply-adds whatever -ffp-contract says (gcc bug 107647), so nx.cpu's
   transforms and factorisations gave other bits than on every other target.
   gcc's fix guards each pattern that fused, so a gcc that fuses this product
   fuses the others. Where it fuses one that holds no fma call, the vectoriser
   is turned off, at a cost in speed. The product is compiled for x86-64 only,
   with the FMA nx.cpu requires there: on arm64 they would fuse through FCMA,
   which nx.cpu's flags leave off. *)

let flags =
  [ "-fvect-cost-model=dynamic"; "--param=max-tree-if-conversion-phi-args=16" ]

let unfused = [ "-fno-tree-vectorize" ]

(* The declaration keeps the unit from being empty, which -Wpedantic -Werror
   refuses. *)
let source =
  "#if !defined(__GNUC__) || defined(__clang__)\n\
   #error not gcc\n\
   #endif\n\
   int f(void);\n"

(* Fast Fourier transforms' chirp, x c for complex x and c, its parts
   interleaved. *)
let product =
  "#if !defined(__x86_64__)\n\
   #error not x86-64\n\
   #endif\n\
   __attribute__((target(\"avx2,fma\")))\n\
   void f(double *restrict r, const double *restrict x,\n\
  \       const double *restrict c, long n) {\n\
  \  for (long j = 0; j < n; j++) {\n\
  \    double re = x[2 * j], im = x[2 * j + 1];\n\
  \    r[2 * j] = re * c[2 * j] - im * c[2 * j + 1];\n\
  \    r[2 * j + 1] = re * c[2 * j + 1] + im * c[2 * j];\n\
  \  }\n\
   }\n"

(* Whether [cc] compiles [source] with [flags], and its output, an object or
   with -S assembly. *)
let compile cc flags source =
  let c = Filename.temp_file "gcc" ".c" in
  let o = Filename.temp_file "gcc" ".o" in
  Out_channel.with_open_text c (fun oc -> output_string oc source);
  let cmd =
    Filename.quote_command (List.hd cc) ~stdout:Filename.null
      ~stderr:Filename.null
      (List.tl cc @ flags @ [ "-c"; c; "-o"; o ])
  in
  let ok = Sys.command cmd = 0 in
  let out =
    if ok then In_channel.with_open_bin o In_channel.input_all else ""
  in
  Sys.remove c;
  if Sys.file_exists o then Sys.remove o;
  (ok, out)

(* Whether a line of assembly is an instruction of x86-64's fused multiply-adds:
   vfmadd, vfmsub, vfnmadd, vfnmsub and their addsub forms. *)
let fused line =
  let line = String.trim line in
  String.starts_with ~prefix:"vfm" line
  || String.starts_with ~prefix:"vfnm" line

let fuses cc =
  let ok, asm =
    compile cc (flags @ [ "-O3"; "-ffp-contract=off"; "-S" ]) product
  in
  ok && List.exists fused (String.split_on_char '\n' asm)

let () =
  let cc = Array.to_list (Array.sub Sys.argv 1 (Array.length Sys.argv - 1)) in
  let gcc, _ = compile cc flags source in
  let flags = if gcc && fuses cc then flags @ unfused else flags in
  print_string (if gcc then "(" ^ String.concat " " flags ^ ")" else "()")
