open Windtrap
open Tolk
module Compiler = Renderer.Compiler

(* The host's target, as the engine gives it. *)
let host_target = Tolk_engine.target Rig.host

(* Sources *)

let increment = "int increment(int x) { return x + 1; }"

let add =
  {|void add(float *restrict a, const float *restrict b, const float *restrict c) {
  for (int i = 0; i < 16; i++) a[i] = b[i] + c[i];
}|}

let square_root = "float root(float x) { return __builtin_sqrtf(x); }"

(* A loop over a count the function takes, which Clang's loop vectorizer packs
   into vectors of four floats. *)
let add_n =
  {|void add_n(float *restrict a, const float *restrict b, const float *restrict c, int n) {
  for (int i = 0; i < n; i++) a[i] = b[i] + c[i];
}
|}

(* The length past which a source compiles without the loop vectorizer, as
   compiler_cpu.mli states it. *)
let long_source = 400_000

(* [add_n] padded with a comment to [n] bytes. *)
let add_n_of n =
  let pad = n - String.length add_n - String.length "/**/" in
  add_n ^ "/*" ^ String.make pad ' ' ^ "*/"

(* A loop that exits on its data, as the renderer writes a loop of no bound:
   it counts a word's set bits by clearing the lowest until none is left, the
   loop that loop idiom recognition replaces by a population count. *)
let bit_count =
  {|void count(unsigned char* restrict args) {
  unsigned long x = *((unsigned long*)args);
  int c = 0;
  for (;;) { if (!(x != 0)) { break; } x &= x - 1; c++; }
  *((int*)(args + 8)) = c;
}
|}

(* A function that needs more registers than a processor has, so that the
   register allocator reaches for every register it may use. *)
let pressure =
  let n = 30 in
  let b = Buffer.create 4096 in
  Buffer.add_string b "void press(volatile long *a) {\n";
  for i = 0 to n - 1 do
    Printf.bprintf b "  long v%d = a[%d];\n" i i
  done;
  for i = 0 to n - 1 do
    Printf.bprintf b "  a[%d] = %s;\n" i
      (String.concat " ^ "
         (List.init n (fun j -> Printf.sprintf "(v%d * %d)" j (i + j + 1))))
  done;
  Buffer.add_string b "}\n";
  Buffer.contents b

(* Part of the kernel rune compiles for a uint4 sort along the last axis of a
   strided view: four gathers at clamped indices, two of 2 minus the clamp and
   two of 7 minus it, and the nibbles they read, selected in pairs. *)
let clamped_gathers =
  {|typedef unsigned char unsigned_char2 __attribute__((aligned(2),ext_vector_type(2)));
typedef long long2 __attribute__((aligned(16),ext_vector_type(2)));
typedef unsigned long unsigned_long2 __attribute__((aligned(16),ext_vector_type(2)));
typedef signed char signed_char2 __attribute__((aligned(2),ext_vector_type(2)));
__attribute__((min_vector_width(512))) void r_3_2(unsigned char* restrict data0_3, long* restrict data1_8, unsigned char* restrict data2_8) {
  long val2 = (*(data1_8+5));
  int alu2 = (((int)((val2&3l)))^-1);
  int alu3 = ((alu2<-3)?-3:alu2);
  unsigned char val3 = (*(data2_8+(2-(alu3^-1))));
  long val4 = (*(data1_8+6));
  int alu4 = (((int)((val4&3l)))^-1);
  int alu5 = ((alu4<-3)?-3:alu4);
  unsigned char val5 = (*(data2_8+(2-(alu5^-1))));
  long val6 = (*(data1_8+0));
  int alu6 = (((int)((val6&3l)))^-1);
  int alu7 = ((alu6<-3)?-3:alu6);
  unsigned char val7 = (*(data2_8+(7-(alu7^-1))));
  long val8 = (*(data1_8+1));
  int alu8 = (((int)((val8&3l)))^-1);
  int alu9 = ((alu8<-3)?-3:alu8);
  unsigned char val9 = (*(data2_8+(7-(alu9^-1))));
  unsigned_char2 cast0 = (unsigned_char2){((unsigned char)(0u)),((unsigned char)(4u))};
  unsigned_char2 alu12 = (__builtin_convertvector((__builtin_bit_cast(unsigned_long2, (long2)(((long2){val6,val8}&3l)))<3ul), signed_char2)?((unsigned_char2){val7,val9}&((unsigned char)(15u))):((unsigned char)(0u)));
  unsigned_char2 alu13 = ((alu12&((unsigned char)(15u)))<<cast0);
  unsigned_char2 alu16 = (__builtin_convertvector((__builtin_bit_cast(unsigned_long2, (long2)(((long2){val2,val4}&3l)))<3ul), signed_char2)?((unsigned_char2){val3,val5}&((unsigned char)(15u))):((unsigned char)(0u)));
  unsigned_char2 alu17 = ((alu16&((unsigned char)(15u)))<<cast0);
  *(data0_3+0) = (alu13[0]+alu13[1]);
  *(data0_3+2) = (alu17[0]+alu17[1]);
}|}

(* Part of the kernel rune compiles for casts from float16: in its first row
   the bits of each half as a double, in the others the half converted to a
   short, saturated, 0 at NaN. *)
let half_casts =
  {|void casts(unsigned long* restrict out, __fp16* restrict in, int n) {
  for (int row = 0; row < 21; row++) {
    for (int i = 0; i < n; i++) {
      __fp16 h = in[i];
      float f = ((float)(h));
      _Bool below = (f<-32768.0f);
      _Bool within = (f<32768.0f);
      float clamped = (((f!=f)|(below|(within==0)))?0.0f:f);
      short s = (below?((short)(-32768)):(within?((short)(clamped)):((short)(32767))));
      short s2 = ((f==f)?s:((short)(0)));
      unsigned long d = __builtin_bit_cast(unsigned long, (double)(h));
      out[(row*230)+i] = ((row==0)?d:((unsigned long)(__builtin_bit_cast(unsigned short, s2))));
    }
  }
}|}

(* The kernels of kernels.golden, each a linear program named after its kernel,
   whose sources are the kernel's nodes in order. *)
let kernels = lazy (Ops.src (Golden.sink "kernels.golden"))

let kernel name =
  let named l =
    match Ops.arg l with Ops.String s -> s = name | _ -> false
  in
  Ops.src (List.find named (Lazy.force kernels))

let target arch =
  {
    Helpers.Target.device = "CPU";
    renderer = "CLANG";
    arch;
    interface = "";
    indices = "";
  }

let rendered arch name = (Cstyle.clang (target arch)).render (kernel name)

(* Compiling *)

let on_macos_arm64 () =
  if Platform.system <> "macosx" || Platform.architecture <> "arm64" then
    skip ~reason:"not an Apple processor" ()

let host_machine =
  match Platform.architecture with
  | "amd64" -> "x86_64"
  | "riscv" -> "riscv64"
  | machine -> machine

let has_infix ~affix s =
  let n = String.length affix in
  let rec at i =
    i + n <= String.length s && (String.sub s i n = affix || at (i + 1))
  in
  at 0

(* [compile arch src] is [src] compiled for [arch], or skips the test if the
   installed Clang has no backend for [arch]'s machine. *)
let compile arch src =
  let clang = Compiler_cpu.clang arch in
  match Compiler.compile clang src with
  | lib -> lib
  | exception Compiler.Compile_error msg
    when List.exists
           (fun affix -> has_infix ~affix msg)
           [ "No available targets"; "unable to create target" ] ->
      skip ~reason:("Clang has no backend for " ^ arch) ()

(* [disassembly arch src] is what disassembling [src]'s object prints, or skips
   the test if the installed objdump cannot read [arch]'s machine. *)
let disassembly arch src =
  let lib = compile arch src in
  ignore (output ());
  (try Compiler.disassemble (Compiler_cpu.clang arch) lib
   with Failure msg when has_infix ~affix:"can't disassemble" msg ->
     skip ~reason:("objdump cannot read " ^ arch) ());
  flush stdout;
  output ()

(* The words of a disassembly, split at everything but letters and digits, so
   that the register [x18] is a word and the offset [#0x18] is not. *)
let words text =
  let is_word = function
    | 'a' .. 'z' | 'A' .. 'Z' | '0' .. '9' -> true
    | _ -> false
  in
  let b = Buffer.create 16 and ws = ref [] in
  let flush () =
    if Buffer.length b > 0 then (
      ws := Buffer.contents b :: !ws;
      Buffer.clear b)
  in
  String.iter
    (fun c -> if is_word c then Buffer.add_char b c else flush ())
    text;
  flush ();
  !ws

let mentions word text = List.mem word (words text)

let uses_vectors text =
  List.exists
    (fun w ->
      String.length w > 1 && w.[0] = 'v' && w.[1] >= '0' && w.[1] <= '9')
    (words text)

(* ELF headers *)

let u8 s i = Char.code s.[i]
let u16 s i = u8 s i lor (u8 s (i + 1) lsl 8)

let is_relocatable_elf64 ~machine lib =
  equal string "\x7fELF" (String.sub lib 0 4);
  equal ~msg:"class" int 2 (u8 lib 4);
  equal ~msg:"byte order" int 1 (u8 lib 5);
  equal ~msg:"object type" int 1 (u16 lib 16);
  equal ~msg:"machine" int machine (u16 lib 18)

let elf_machine = function
  | "x86_64" -> 62
  | "arm64" -> 183
  | "riscv64" -> 243
  | m -> invalid_arg ("no ELF machine for " ^ m)

(* Objects *)

let objects =
  group "objects"
    [
      cases "compiles to a relocatable ELF object for" ~id:Fun.id
        [ "x86_64,x86-64"; "arm64,generic"; "riscv64,rv64g" ] (fun arch ->
          let machine = List.hd (String.split_on_char ',' arch) in
          is_relocatable_elf64 ~machine:(elf_machine machine)
            (compile arch increment));
      test "native is the host's processor" (fun () ->
          is_relocatable_elf64 ~machine:(elf_machine host_machine)
            (compile (host_machine ^ ",native") increment));
      test "native on riscv64 is rv64g" (fun () ->
          equal string
            (compile "riscv64,rv64g" increment)
            (compile "riscv64,native" increment));
      test "one source compiles to the same bytes" (fun () ->
          let arch = host_machine ^ ",native" in
          equal string (compile arch add) (compile arch add));
      cases "a square root is one instruction on" ~id:fst
        [ ("x86_64,x86-64", "sqrtss"); ("arm64,generic", "fsqrt") ]
        (fun (arch, instruction) ->
          let text = disassembly arch square_root in
          is_true ~msg:text (mentions instruction text);
          not_contains ~sub:"sqrtf" text);
    ]

(* Several sources at once *)

let ccache_off f = Setting.context [ B (Setting.ccache, false) ] f

(* Sources of each length class, the loop vectorizer's on and off. *)
let pool =
  [
    ("increment", increment);
    ("add", add);
    ("square_root", square_root);
    ("add_n", add_n);
    ("pressure", pressure);
    ("clamped_gathers", clamped_gathers);
    ("half_casts", half_casts);
    ("long", add_n_of (long_source + 1));
  ]

let alone =
  lazy
    (let clang = Compiler_cpu.clang (host_machine ^ ",native") in
     List.map (fun (name, src) -> (name, ccache_off (fun () -> Compiler.compile clang src))) pool)

let batches =
  group "several sources"
    [
      prop "each object is the one its source compiles to alone" ~count:25
        Gen.(
          list ~size:(int_range 2 6)
            (of_list ~pp:Format.pp_print_string (List.map fst pool)))
        (fun names ->
          cover "a source twice" (List.length (List.sort_uniq compare names) < List.length names);
          cover "a long source among short ones"
            (List.mem "long" names && List.exists (( <> ) "long") names);
          let clang = Compiler_cpu.clang (host_machine ^ ",native") in
          let objs =
            ccache_off (fun () ->
                Compiler.compile_all clang
                  (List.map (fun n -> List.assoc n pool) names))
          in
          equal (list string)
            (List.map (fun n -> List.assoc n (Lazy.force alone)) names)
            objs);
      test "a source that names its own file compiles to one object" (fun () ->
          let clang = Compiler_cpu.clang (host_machine ^ ",native") in
          let named = "const char *file(void) { return __FILE__; }" in
          let objs () =
            ccache_off (fun () -> Compiler.compile_all clang [ add; named ])
          in
          let first = List.nth (objs ()) 1 in
          equal string ~msg:"again" first (List.nth (objs ()) 1);
          equal string ~msg:"alone" first
            (ccache_off (fun () -> Compiler.compile clang named)));
      test "a rejected source among others raises its own diagnostics"
        (fun () ->
          let clang = Compiler_cpu.clang (host_machine ^ ",native") in
          raises_match
            (function
              | Compiler.Compile_error msg ->
                  has_infix ~affix:"undeclared_name" msg
                  && not (has_infix ~affix:"other_name" msg)
              | _ -> false)
            (fun () ->
              ccache_off (fun () ->
                  Compiler.compile_all clang
                    [
                      increment;
                      "int f(void) { return undeclared_name; }";
                      "int g(void) { return other_name; }";
                    ])));
    ]

(* A compile abandoned

   A stand-in for Clang: asked its version or what it runs, it states itself as
   its compiler proper. Run as that, it records its process, leaves a file of its
   own beside its first input, as Clang leaves its temporary object, signals
   its parent and sleeps, so that the compile is abandoned while it runs. *)
let stand_in ~pid_file =
  Printf.sprintf
    {|#!/bin/sh
if [ "$1" != "-cc1" ]; then
  echo "clang version 19.0.0" >&2
  printf ' "%%s" "-cc1" "-o" "-" "-"\n' "$0" >&2
  exit 0
fi
for a; do rsp=$a; done
input=$(head -n 1 "${rsp#@}" | tr -d '"')
echo $$ > %s
: > "$(dirname "$input")/-.o-temporary"
kill -USR1 $PPID
exec sleep 30
|}
    (Filename.quote pid_file)

let abandoned =
  test "a compile abandoned by an exception leaves its Clang ended and no files"
    (fun () ->
      if Sys.win32 then skip ~reason:"a POSIX shell stands in for Clang" ();
      let dir = Filename.temp_dir "stand_in" "" in
      let pid_file = Filename.concat dir "pid"
      and clang = Filename.concat dir "clang"
      and tmp = Filename.concat dir "tmp" in
      Out_channel.with_open_gen [ Open_wronly; Open_creat ] 0o755 clang
        (fun oc -> output_string oc (stand_in ~pid_file));
      Unix.mkdir tmp 0o700;
      let saved = Filename.get_temp_dir_name () in
      let handler = Sys.Signal_handle (fun _ -> raise Exit) in
      let previous = Sys.signal Sys.sigusr1 handler in
      Fun.protect
        ~finally:(fun () ->
          Sys.set_signal Sys.sigusr1 previous;
          Filename.set_temp_dir_name saved)
        (fun () ->
          Filename.set_temp_dir_name tmp;
          raises Exit (fun () ->
              Setting.context
                [ B (Setting.cc, clang); B (Setting.ccache, false) ]
                (fun () ->
                  Compiler.compile_all
                    (Compiler_cpu.clang "x86_64,x86-64")
                    [ increment ])));
      let pid =
        int_of_string
          (String.trim (In_channel.with_open_bin pid_file In_channel.input_all))
      in
      raises_match
        (function Unix.Unix_error (ESRCH, _, _) -> true | _ -> false)
        (fun () -> Unix.kill pid 0);
      equal (array string) [||] (Sys.readdir tmp);
      List.iter Sys.remove [ pid_file; clang ];
      List.iter Sys.rmdir [ tmp; dir ])

(* tinygrad's tests *)

let tinygrad =
  group "as tinygrad"
    [
      cases "the add kernel moves vectors with vmov iff AVX is on:" ~id:fst
        [ ("x86_64,x86-64,avx", true); ("x86_64,x86-64,-avx", false) ]
        (fun (arch, vmov) ->
          let text = disassembly arch (rendered arch "add") in
          equal ~msg:text bool vmov (has_infix ~affix:"vmov" text));
      test "a half addition on an Apple processor converts nothing" (fun () ->
          on_macos_arm64 ();
          let arch = "arm64,native" in
          not_contains ~sub:"fcvt" (disassembly arch (rendered arch "half_add")));
    ]

(* Execution *)

let floats xs = Array.map (fun x -> `Float x) xs
let values = array Dtypes.value
let host = lazy (Cstyle.clang host_target)
let loaded name = Run.program (Lazy.force host) (kernel name)

let execution =
  group "execution on the host"
    [
      test "a compiled kernel adds two buffers" (fun () ->
          let a = Array.init 16 Float.of_int
          and b = Array.init 16 (fun i -> Float.of_int (100 - i)) in
          let out =
            Run.on_host (loaded "add") [ (1, floats a); (2, floats b) ]
          in
          equal values (floats (Array.make 16 100.)) (List.assoc 0 out));
      test "a compiled square root runs without a library" (fun () ->
          let squares = Array.init 16 (fun i -> Float.of_int (i * i)) in
          let out = Run.on_host (loaded "sqrt") [ (1, floats squares) ] in
          equal values (floats (Array.init 16 Float.of_int)) (List.assoc 0 out));
      test "a product and a sum round twice, never fused" (fun () ->
          (* (1 + 2^-12)^2 rounds to 1 + 2^-11, a tie to even, so the sum with
             -(1 + 2^-11) is 0; a fused multiply-add keeps the 2^-24. *)
          let x = 1. +. 0x1p-12 in
          let out =
            Run.on_host (loaded "muladd")
              [
                (1, floats (Array.make 16 x));
                (2, floats (Array.make 16 x));
                (3, floats (Array.make 16 (-.(1. +. 0x1p-11))));
              ]
          in
          equal values (floats (Array.make 16 0.)) (List.assoc 0 out));
    ]

(* The baseline

   x86-64's baseline processor has neither F16C nor FMA (x86-64 psABI,
   §3.1.1), so a host program compiled for it converts halves by the
   conversions each object defines and calls the process's [fmaf]. Every
   x86-64 host runs such a program. *)

let x86_host () =
  if Rig.instruction_set Rig.host <> "x86_64" then
    skip ~reason:"not an x86-64 host" ()

(* Under Windows the object calls the process's functions by Windows'
   convention ([Compiler_cpu.win64]). *)
let baseline =
  target ((if Sys.win32 then "x86_64-windows" else "x86_64") ^ ",x86-64")

(* The program of out[i] = f in[i], over [n] elements of [src] into [dst]. *)
let elementwise ~name ~n src dst f =
  let i = Ops.range ~axis_type:Weak (Int n) [ 0 ] in
  let at slot dt = Ops.index (Shape.placeholder ~slot [ n ] dt) [ i ] in
  let sink =
    Ops.sink ~kernel:(Ops.kernel_info ~name ())
      [ Ops.end_ (Ops.store (at 0 dst) (f (at 1 src))) [ i ] ]
  in
  Codegen.to_program sink (Cstyle.clang baseline)

let ints xs = Array.map (fun x -> `Int (Bigint.of_int x)) xs

let to_ints vs =
  Array.map (function `Int b -> Bigint.to_int b | _ -> fail "not an integer") vs

(* [prg] run once on [xs], its output read back as integers. *)
let run_ints prg xs =
  to_ints (List.assoc 0 (Run.on_host prg [ (1, ints xs) ]))

(* IEEE 754's binary16, by its bits: a finite half's value, exact in a
   double. *)
let half_value h =
  let e = (h lsr 10) land 0x1f and m = h land 0x3ff in
  let v =
    if e = 0 then Float.ldexp (Float.of_int m) (-24)
    else Float.ldexp (Float.of_int (m lor 0x400)) (e - 25)
  in
  if h land 0x8000 <> 0 then -.v else v

let f32_bits v = Int32.to_int (Int32.bits_of_float v) land 0xffff_ffff

(* The float32 bits of half [h]: its value, exact, and a NaN made quiet with
   its payload, as compiler_cpu.mli states. *)
let widened h =
  let sign = (h land 0x8000) lsl 16 and m = h land 0x3ff in
  if (h lsr 10) land 0x1f <> 0x1f then f32_bits (half_value h)
  else sign lor 0x7f800000 lor (m lsl 13) lor if m <> 0 then 0x400000 else 0

(* The half bits of the float32 bits [x], rounded to nearest with ties to
   even (IEEE 754 §4.3.1): the nearer of the two finite halves around it, the
   one of even bits at a tie, and an infinity at or past 65520, halfway
   between the greatest half and the next power of two. *)
let narrowed x =
  let sign = (x lsr 16) land 0x8000 and a = x land 0x7fff_ffff in
  if a > 0x7f800000 then sign lor 0x7e00 lor ((a lsr 13) land 0x3ff)
  else
    let v = Int32.float_of_bits (Int32.of_int a) in
    if v >= 65520. then sign lor 0x7c00
    else
      (* The greatest finite half at most [v], by bisection over their bits,
         which order them. *)
      let rec below lo hi =
        if hi - lo <= 1 then lo
        else
          let mid = (lo + hi) / 2 in
          if half_value mid <= v then below mid hi else below lo mid
      in
      let lo = below 0 0x7c00 in
      let hi = lo + 1 in
      let dlo = v -. half_value lo and dhi = half_value hi -. v in
      let h = if dlo < dhi || (dlo = dhi && lo land 1 = 0) then lo else hi in
      sign lor h

let every_half = Array.init 65536 Fun.id

(* Every half widened, the midpoints between finite halves and their float32
   neighbours, the edges of overflow, and the negatives of all. *)
let narrowing_inputs =
  let positive =
    List.init 0x7c00 (fun h ->
        let mid = f32_bits ((half_value h +. half_value (h + 1)) /. 2.) in
        [ widened h; mid; mid - 1; mid + 1 ])
    |> List.concat
  in
  let edges = [ 0x7f800000; 0x7fc00000; 0x7f800001; 0x7fbfffff ] in
  let all = edges @ positive in
  Array.of_list (all @ List.map (fun x -> x lor 0x8000_0000) all)

let half_casts_baseline =
  group "a program for x86-64's baseline runs here"
    [
      test "every half widens to its float32" (fun () ->
          x86_host ();
          let prg =
            elementwise ~name:"widen" ~n:65536 Dtype.Uint16 Dtype.Uint32
              (fun x ->
                Ops.bitcast
                  (Ops.cast (Ops.bitcast x Dtype.Float16) Dtype.Float32)
                  Dtype.Uint32)
          in
          equal (array int) (Array.map widened every_half)
            (run_ints prg every_half));
      test "float32s narrow to the nearest half, ties to even" (fun () ->
          x86_host ();
          let xs = narrowing_inputs in
          let prg =
            elementwise ~name:"narrow" ~n:(Array.length xs) Dtype.Uint32
              Dtype.Uint16 (fun x ->
                Ops.bitcast
                  (Ops.cast (Ops.bitcast x Dtype.Float32) Dtype.Float16)
                  Dtype.Uint16)
          in
          equal (array int) (Array.map narrowed xs) (run_ints prg xs));
      test "a fused multiply-add rounds once" (fun () ->
          x86_host ();
          (* (1 + 2^-12)^2 - (1 + 2^-11) is 2^-24, exact in a float; each
             product and sum rounding on its own gives 0. *)
          let x = 1. +. 0x1p-12 in
          let n = 4 in
          let i = Ops.range ~axis_type:Weak (Int n) [ 0 ] in
          let at slot = Ops.index (Shape.placeholder ~slot [ n ] Float32) [ i ] in
          let sink =
            Ops.sink
              ~kernel:(Ops.kernel_info ~name:"fused" ())
              [
                Ops.end_
                  (Ops.store (at 0) (Ops.alu (at 1) Op.Mulacc [ at 2; at 3 ]))
                  [ i ];
              ]
          in
          let prg = Codegen.to_program sink (Cstyle.clang baseline) in
          let out =
            Run.on_host prg
              [
                (1, floats (Array.make n x));
                (2, floats (Array.make n x));
                (3, floats (Array.make n (-.(1. +. 0x1p-11))));
              ]
          in
          equal values (floats (Array.make n 0x1p-24)) (List.assoc 0 out));
    ]

(* Features *)

let features =
  group "features"
    [
      cases "a half addition on arm64 converts iff fp16 is off:" ~id:fst
        [ ("arm64,generic", true); ("arm64,generic,fp16", false) ]
        (fun (arch, fcvt) ->
          let text = disassembly arch (rendered arch "half_add") in
          equal ~msg:text bool fcvt (mentions "fcvt" text));
      test "arm64 vectorizes with its SIMD registers" (fun () ->
          is_true (uses_vectors (disassembly "arm64,generic" add)));
      test "-simd on arm64 disables them" (fun () ->
          let text = disassembly "arm64,generic,-simd" add in
          is_false ~msg:text (uses_vectors text));
      test "a feature on riscv64 is an extension the processor gains" (fun () ->
          not_equal string
            (compile "riscv64,rv64g" add)
            (compile "riscv64,rv64g,c" add));
      (* Clang 18 crashed splitting a vector of more than 16 halves that it
         negates, takes the absolute value of or copies a sign to, for these
         processors, which prefer 256-bit vectors. *)
      cases "a kernel over more than 16 halves compiles for AVX-512"
        ~id:(fun (arch, name) -> arch ^ " " ^ name)
        (List.concat_map
           (fun arch ->
             List.map
               (fun name -> (arch, name))
               [ "half_neg"; "half_abs"; "half_copysign"; "half_min" ])
           [
             "x86_64,skylake-avx512";
             "x86_64,icelake-server";
             "x86_64,sapphirerapids";
           ])
        (fun (arch, name) ->
          is_relocatable_elf64 ~machine:(elf_machine "x86_64")
            (compile arch (rendered arch name)));
      (* LLVM 18's straight-line vectorizer narrowed the four indices into
         invalid IR for these processors, and Clang 18 crashed on it. *)
      cases "four clamped gathers compile for AVX-512 on" ~id:Fun.id
        [ "x86_64,x86-64-v4"; "x86_64,skylake-avx512"; "x86_64,znver4" ]
        (fun arch ->
          is_relocatable_elf64 ~machine:(elf_machine "x86_64")
            (compile arch clamped_gathers));
      (* The straight-line vectorizer of LLVM 19 and of Apple's Clang 17
         narrowed four lanes of rune's binary search into invalid IR for these
         processors, which the verifier rejects. *)
      cases "a binary search of four lanes compiles for AVX2 and AVX-512 on"
        ~id:Fun.id
        [ "x86_64,x86-64-v3"; "x86_64,x86-64-v4"; "x86_64,znver4" ]
        (fun arch ->
          let src =
            In_channel.with_open_bin "vectorized_search.c" In_channel.input_all
          in
          is_relocatable_elf64 ~machine:(elf_machine "x86_64")
            (ccache_off (fun () -> compile arch src)));
      (* Clang 18 crashed on this loop for these processors, which have
         AVX-512's half arithmetic. *)
      cases "a loop of half casts compiles with AVX-512's half arithmetic on"
        ~id:Fun.id
        [ "x86_64,sapphirerapids"; "x86_64,x86-64-v4,avx512fp16" ]
        (fun arch ->
          is_relocatable_elf64 ~machine:(elf_machine "x86_64")
            (compile arch half_casts));
      (* SSE2 adds four floats with [addps] and one with [addss]: a packed
         loop adds its last floats one at a time. *)
      cases
        "a loop over a count it takes is packed into vectors iff its source \
         is at most 400000 bytes long:"
        ~id:(fun (n, _) -> string_of_int n)
        [
          (String.length add_n, [ "addps"; "addss" ]);
          (long_source, [ "addps"; "addss" ]);
          (long_source + 1, [ "addss" ]);
          (2 * long_source, [ "addss" ]);
        ]
        (fun (n, adds) ->
          let src = if n = String.length add_n then add_n else add_n_of n in
          equal int ~msg:"the source's length" n (String.length src);
          let text = disassembly "x86_64,x86-64" src in
          equal (list string) adds
            (List.filter (fun w -> mentions w text) [ "addps"; "addss" ]));
      test "a loop that exits on its data stays a loop: idioms are off"
        (fun () ->
          let text = disassembly "x86_64,x86-64-v3" bit_count in
          equal (list string) []
            (List.filter (fun w -> mentions w text) [ "popcnt" ]));
      test "arm64 leaves x18 alone" (fun () ->
          let text = disassembly "arm64,generic" pressure in
          is_false ~msg:"x18" (mentions "x18" text);
          is_false ~msg:"w18" (mentions "w18" text));
    ]

(* Flags *)

let llvm18 = "Ubuntu clang version 18.1.3 (1ubuntu1)"
let x86_64_elf = "--target=x86_64-none-unknown-elf"

let windows =
  [
    "--target=x86_64-w64-windows-gnu-elf";
    "-mno-stack-arg-probe";
    "-mno-red-zone";
  ]

let llvm18_off = [ "-mno-avx10.1-512"; "-mno-avx10.1-256"; "-mno-avx512fp16" ]

let flags =
  group "flags"
    [
      cases "under LLVM 18, x86_64 ends by disabling AVX10.1 and avx512fp16:"
        ~id:fst
        [
          ("x86_64,native", [ x86_64_elf; "-march=native" ] @ llvm18_off);
          ( "x86_64,graniterapids,avx10.1-256",
            [ x86_64_elf; "-march=graniterapids"; "-mavx10.1-256" ]
            @ llvm18_off );
          ( "x86_64,sapphirerapids,avx512fp16,-avx",
            [ x86_64_elf; "-march=sapphirerapids"; "-mavx512fp16"; "-mno-avx" ]
            @ llvm18_off );
          ("x86_64-windows,native", windows @ [ "-march=native" ] @ llvm18_off);
        ]
        (fun (arch, expected) ->
          equal (list string) expected
            (Compiler_cpu.flags ~version:llvm18 arch));
      test "LLVM's own release of 18 is LLVM 18" (fun () ->
          equal (list string)
            ([ x86_64_elf; "-march=native" ] @ llvm18_off)
            (Compiler_cpu.flags ~version:"clang version 18.1.8"
               "x86_64,native"));
      cases "x86_64,native takes its own arguments alone under" ~id:Fun.id
        [
          "Debian clang version 19.1.7 (3+b1)";
          "Homebrew clang version 20.1.8";
          "clang version 17.0.6";
          "Apple clang version 18.0.0 (clang-1800.0.31.1)";
          "";
        ]
        (fun version ->
          equal (list string)
            [ x86_64_elf; "-march=native" ]
            (Compiler_cpu.flags ~version "x86_64,native"));
      cases "another machine under LLVM 18 takes its own arguments alone:"
        ~id:fst
        [
          ( "arm64,generic,fp16",
            [
              "--target=arm64-none-unknown-elf";
              "-ffixed-x18";
              "-mcpu=generic+fp16";
            ] );
          ( "riscv64,rv64g,c",
            [ "--target=riscv64-none-unknown-elf"; "-march=rv64g_c" ] );
        ]
        (fun (arch, expected) ->
          equal (list string) expected
            (Compiler_cpu.flags ~version:llvm18 arch));
      cases "a tune field tunes, after the processor and its features:"
        ~id:fst
        [
          ( "x86_64,x86-64-v3,tune=native",
            [ x86_64_elf; "-march=x86-64-v3"; "-mtune=native" ] );
          ( "x86_64,x86-64,avx,tune=znver4",
            [ x86_64_elf; "-march=x86-64"; "-mavx"; "-mtune=znver4" ] );
          ( "arm64,generic,tune=native",
            [
              "--target=arm64-none-unknown-elf";
              "-ffixed-x18";
              "-mcpu=generic";
              "-mtune=native";
            ] );
        ]
        (fun (arch, expected) ->
          equal (list string) expected
            (Compiler_cpu.flags ~version:"clang version 19.1.7" arch));
      test "tuning for this processor adds none of its instructions"
        (fun () ->
          x86_host ();
          let src =
            "float f(float a, float b, float c) { return __builtin_fmaf(a, \
             b, c); }"
          in
          let text = disassembly "x86_64,x86-64,tune=native" src in
          is_false ~msg:text (has_infix ~affix:"vfmadd" text));
      test "a malformed architecture is refused" (fun () ->
          raises_match
            (Exn.invalid_arg ~sub:"'x86_64'")
            (fun () -> Compiler_cpu.flags ~version:llvm18 "x86_64"));
    ]

(* Architecture strings *)

let architectures =
  group "architectures"
    [
      cases "an architecture of fewer than two fields is refused, named:"
        ~id:(Printf.sprintf "%S") [ ""; "x86_64"; "arm64" ] (fun arch ->
          raises_match
            (Exn.invalid_arg ~sub:("'" ^ arch ^ "'"))
            (fun () -> Compiler_cpu.clang arch));
      cases "another machine is refused, named:" ~id:fst
        [
          ("mips,native", "mips");
          ("aarch64,native", "aarch64");
          ("X86_64,native", "X86_64");
        ]
        (fun (arch, machine) ->
          raises_match (Exn.invalid_arg ~sub:machine) (fun () ->
              Compiler_cpu.clang arch));
    ]

(* Errors *)

let errors =
  group "errors"
    [
      test "a rejected source raises Compile_error with Clang's diagnostics"
        (fun () ->
          match
            Compiler.compile
              (Compiler_cpu.clang (host_machine ^ ",native"))
              "int f(void) { return undeclared_name; }"
          with
          | _ -> fail "Clang accepted an undeclared name"
          | exception Compiler.Compile_error msg ->
              in_order ~subs:[ "error"; "undeclared_name" ] msg);
    ]

(* With the straight-line vectorizer on: CC is read once, when the first
   compile runs, so these tests run in a process of their own, where CC names
   a script that runs Clang with that vectorizer on, whatever turns it off
   before. *)

let invalid_ir =
  tag "slp-clang"
  @@ group "invalid IR"
       [
         test
           "a source that LLVM 19's straight-line vectorizer makes invalid \
            raises Compile_error naming its function" (fun () ->
             if Sys.win32 then skip ~reason:"CC is a shell script" ();
             let stated =
               Helpers.system (Setting.value Setting.cc ^ " --version")
             in
             if
               has_infix ~affix:"Apple" stated
               || not (has_infix ~affix:"clang version 19." stated)
             then skip ~reason:"not a Clang of LLVM 19" ();
             let src =
               In_channel.with_open_bin "vectorized_search.c"
                 In_channel.input_all
             in
             match
               ccache_off (fun () ->
                   Compiler.compile (Compiler_cpu.clang "x86_64,x86-64-v3") src)
             with
             | _ -> fail "Clang compiled the invalid IR"
             | exception Compiler.Compile_error msg ->
                 contains ~sub:"E_16_2_4" msg);
       ]

(* Compile time *)

(* rune's host kernel of 1024 lanes gathered from 32 loaded rows: LLVM's
   interleaved load combine took Clang 19 274 s of 293 on it for apple-m1.
   rune's maximum of 32 float64 lanes in a loop of 32 trips: LLVM 23's loop
   idiom recognition did not finish it in 300 s. *)
let compile_time =
  group "compile time"
    [
      test ~timeout:60.
        "an arm64 kernel of 1024 lanes gathered from 32 rows compiles within \
         a minute" (fun () ->
          let src =
            In_channel.with_open_bin "gathered_lanes.c" In_channel.input_all
          in
          let lib = ccache_off (fun () -> compile "arm64,apple-m1" src) in
          is_relocatable_elf64 ~machine:(elf_machine "arm64") lib);
      cases ~timeout:60.
        "a host maximum of 32 float64 lanes in a loop of 32 trips compiles \
         within a minute for"
        ~id:Fun.id [ "x86_64,x86-64-v3"; "arm64,apple-m1" ]
        (fun arch ->
          let src =
            In_channel.with_open_bin "max_chain.c" In_channel.input_all
          in
          let machine = List.hd (String.split_on_char ',' arch) in
          let lib = ccache_off (fun () -> compile arch src) in
          is_relocatable_elf64 ~machine:(elf_machine machine) lib);
    ]

(* Without Clang: CC is read once, when the first compile runs, so these tests
   run in a process of their own, where CC names a program that does not
   exist. *)

let missing_clang = "/nonexistent/clang"

(* The macros of each processor name its features; Clang's statement names the
   processor alone. *)
let registers =
  let file arch =
    Option.map
      (fun (r : Renderer.registers) -> (r.count, r.bytes))
      (Compiler_cpu.registers arch)
  in
  cases "the vector registers, their number and bytes, of" ~id:fst
    [
      ("x86_64,x86-64", Some (16, 16));
      ("x86_64,x86-64-v3", Some (16, 32));
      ("x86_64,x86-64-v4", Some (32, 32));
      ("x86_64,znver4", Some (32, 32));
      ("x86_64-windows,x86-64-v3", Some (16, 32));
      ("arm64,apple-m1", Some (32, 16));
      ("riscv64,rv64g", None);
    ]
    (fun (arch, expected) -> equal (option (pair int int)) expected (file arch))

let without_clang =
  tag "no-clang"
  @@ group "without Clang"
       [
         test "a compile raises Compile_error naming the program CC names"
           (fun () ->
             raises_match
               (function
                 | Compiler.Compile_error msg ->
                     has_infix ~affix:missing_clang msg
                 | _ -> false)
               (fun () ->
                 Compiler.compile
                   (Compiler_cpu.clang (host_machine ^ ",native"))
                   increment));
         test "an x86_64 processor has SSE's 16 registers of 16 bytes"
           (fun () ->
             equal
               (option (pair int int))
               (Some (16, 16))
               (Option.map
                  (fun (r : Renderer.registers) -> (r.count, r.bytes))
                  (Compiler_cpu.registers "x86_64,x86-64-v3")));
         test "a cached object is served without a Clang that runs" (fun () ->
             let clang = Compiler_cpu.clang (host_machine ^ ",native") in
             let table = Option.get (Compiler.cachekey clang) in
             Helpers.Diskcache.put ~table increment "cached object";
             equal string "cached object"
               (Compiler.compile_cached clang increment));
         test "an object cached by the default compiler is not served under CC"
           (fun () ->
             let cachedb = Helpers.cachedb in
             let table =
               match
                 Disk_cache.child ~env:[ ("CC", "clang") ] ~cachedb "table"
               with
               | Ok table -> table
               | Error err ->
                   failf "the default compiler named no table: %s" err
             in
             let src = increment ^ "\n/* compiled by clang */\n" in
             Helpers.Diskcache.put ~table src "clang's object";
             raises_match
               (function Compiler.Compile_error _ -> true | _ -> false)
               (fun () ->
                 Compiler.compile_cached
                   (Compiler_cpu.clang (host_machine ^ ",native"))
                   src));
       ]

(* Disassembly *)

let disassembly_ =
  group "disassembly"
    [
      test "prints what objdump prints of the object" (fun () ->
          let text = disassembly "x86_64,x86-64" increment in
          in_order ~subs:[ "elf64-x86-64"; "<increment>:" ] text);
    ]

(* The cache *)

(* The table of the host's compiler, as a process of its own names it. *)
let host_table () =
  print_string
    (Option.get
       (Compiler.cachekey (Compiler_cpu.clang (host_machine ^ ",native"))))

(* Hints the release of LLVM that [TOLK_TEST_HINT] names for the Clang [CC]
   runs, as a process that found it would. On Windows a variable set to the
   empty string reads as unset, so an unset [TOLK_TEST_HINT] names the empty
   release. *)
let hint () =
  Helpers.Diskcache.put ~table:"clang_llvm"
    (Setting.value Setting.cc)
    (Option.value ~default:"" (Sys.getenv_opt "TOLK_TEST_HINT"))

let parts = [ ("table", host_table); ("hint", hint) ]

(* [table_in ()] is the table a process names, from a directory of its own. *)
let table_in () =
  let cachedb = Filename.concat (Sys.getcwd ()) "unused_cache" in
  chdir (temp_dir ());
  Disk_cache.child ~cachedb "table"

let is_hex c = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')

(* [named ~prefix table] checks that [table] is [prefix] then a digest. *)
let named ~prefix table =
  let n = Int.min (String.length prefix) (String.length table) in
  equal string ~msg:"the name" prefix (String.sub table 0 n);
  let digest = String.sub table n (String.length table - n) in
  equal int ~msg:"the digest's length" 32 (String.length digest);
  equal (list char) ~msg:"the digest's other characters" []
    (List.filter (fun c -> not (is_hex c)) (List.of_seq (String.to_seq digest)))

let cache =
  group "cache"
    [
      cases "objects are cached in the table of the architecture and a digest"
        ~id:fst
        [
          ("x86_64,znver2", "compile_clang_obj_x86_64_znver2_");
          ( "x86_64,znver2,avx,-avx512f",
            "compile_clang_obj_x86_64_znver2_avx_-avx512f_" );
          ("arm64,native", "compile_clang_obj_arm64_native_");
        ]
        (fun (arch, prefix) ->
          named ~prefix
            (Option.get (Compiler.cachekey (Compiler_cpu.clang arch))));
      test "the table does not depend on the working directory" (fun () ->
          let a = table_in () in
          equal Disk_cache.outcome a (table_in ()));
      test "the table does not depend on the release a process is hinted"
        (fun () ->
          let cachedb = Disk_cache.fresh () in
          let first = Disk_cache.child ~cachedb "table" in
          List.iter
            (fun release ->
              (match
                 Disk_cache.child ~env:[ ("TOLK_TEST_HINT", release) ] ~cachedb
                   "hint"
               with
              | Ok _ -> ()
              | Error err -> failf "the hint was not kept: %s" err);
              equal Disk_cache.outcome ~msg:release first
                (Disk_cache.child ~cachedb "table"))
            [ "17"; "18"; "19"; "20"; ""; "not a release" ]);
      test "with ccache off, no release is hinted" (fun () ->
          let cachedb = Disk_cache.fresh () in
          (match
             Disk_cache.child ~env:[ ("CCACHE", "0") ] ~cachedb "table"
           with
          | Ok _ -> ()
          | Error err -> failf "the table was not named: %s" err);
          equal (list string) []
            (Disk_cache.entries ~table:"clang_llvm" cachedb));
      test "the table is named with ccache off" (fun () ->
          let table () =
            Compiler.cachekey (Compiler_cpu.clang "arm64,native")
          in
          equal (option string) (table ()) (ccache_off table));
    ]

let () =
  Disk_cache.play parts;
  exit
    (run "Tolk.Compiler_cpu"
       [
         objects;
         batches;
         abandoned;
         tinygrad;
         execution;
         half_casts_baseline;
         features;
         flags;
         registers;
         architectures;
         errors;
         invalid_ir;
         compile_time;
         without_clang;
         disassembly_;
         cache;
       ])
