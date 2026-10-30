(* isa.ml ISA CC...: prints the flags that compile ISA's checksum path, as
   an S-expression for dune's :include, when the C compiler CC accepts
   them on a use of every intrinsic the path's unit calls; else no flags,
   and the unit compiles empty (lib/checksum_paths.h). The flags and
   intrinsics postdate GCC 4.4 (PCLMULQDQ) to GCC 11 (AVX-VNNI).

   gcc for Windows or Cygwin compiles an AVX2 path with
   -Wa,-muse-unaligned-vector-move too, or not at all. It keeps 32-byte
   vectors on the stack at addresses it assumes 32-byte aligned, where the
   Windows x64 ABI aligns the stack to 16 bytes only (gcc bug 54412), and
   an aligned move to such an address faults. The assembler's option
   (binutils 2.38) writes each aligned vector move as an unaligned one,
   which runs as fast on aligned data. *)

let isas =
  [
    ( "pclmul",
      [ "-msse4.1"; "-mpclmul" ],
      "#include <immintrin.h>\n\
       __m128i f(__m128i a, __m128i b, __m128i m) {\n\
      \  return _mm_blendv_epi8(_mm_shuffle_epi8(a, m),\n\
      \                         _mm_clmulepi64_si128(a, b, 0x11), m);\n\
       }\n" );
    ( "vpclmul",
      [ "-mavx2"; "-mpclmul"; "-mvpclmulqdq" ],
      "#include <immintrin.h>\n\
       __m128i f(__m256i a, __m256i b, __m128i x) {\n\
      \  __m256i z = _mm256_inserti128_si256(_mm256_setzero_si256(), x, 0);\n\
      \  __m256i c = _mm256_xor_si256(_mm256_clmulepi64_epi128(a, b, 0x00), z);\n\
      \  return _mm_clmulepi64_si128(_mm256_castsi256_si128(c),\n\
      \                              _mm256_extracti128_si256(c, 1), 0x11);\n\
       }\n" );
    ( "avx2",
      [ "-mavx2" ],
      "#include <immintrin.h>\n\
       __m256i f(__m256i a, __m256i w) {\n\
      \  return _mm256_add_epi32(_mm256_sad_epu8(a, w),\n\
      \    _mm256_madd_epi16(_mm256_maddubs_epi16(a, w), _mm256_set1_epi16(1)));\n\
       }\n" );
    ( "vnni",
      [ "-mavx2"; "-mavxvnni" ],
      "#include <immintrin.h>\n\
       __m256i f(__m256i c, __m256i a, __m256i w) {\n\
      \  return _mm256_dpbusd_avx_epi32(c, a, w);\n\
       }\n" );
    ( "arm",
      [ "-march=armv8-a+crc+crypto" ],
      "#include <arm_acle.h>\n\
       #include <arm_neon.h>\n\
       poly128_t f(poly64x2_t a, poly64x2_t b, unsigned *c) {\n\
      \  *c = __crc32d(*c, 1);\n\
      \  return vmull_high_p64(a, b) ^ vmull_p64(vgetq_lane_p64(a, 0), 1);\n\
       }\n" );
  ]

let unaligned = "-Wa,-muse-unaligned-vector-move"

let windows_gcc =
  "#if defined(__clang__) || !defined(__GNUC__)\n\
   #error not gcc\n\
   #endif\n\
   #if !defined(_WIN32) && !defined(__CYGWIN__)\n\
   #error not for Windows\n\
   #endif\n\
   int f(void);\n"

(* Whether [cc] compiles [source] with [flags]. *)
let compiles name cc flags source =
  let c = Filename.temp_file name ".c" in
  let o = Filename.temp_file name ".o" in
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
  let isa = Sys.argv.(1) in
  let cc = Array.to_list (Array.sub Sys.argv 2 (Array.length Sys.argv - 2)) in
  let _, flags, source =
    List.find (fun (name, _, _) -> name = isa) isas
  in
  let flags =
    if List.mem "-mavx2" flags && compiles isa cc [] windows_gcc then
      flags @ [ unaligned ]
    else flags
  in
  let ok = compiles isa cc flags source in
  print_string (if ok then "(" ^ String.concat " " flags ^ ")" else "()")
