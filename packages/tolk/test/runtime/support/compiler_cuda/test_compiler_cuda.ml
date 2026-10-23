open Windtrap
open Tolk
module Compiler = Renderer.Compiler

let kernel =
  {|extern "C" __global__ void add(float *a, const float *b) {
  a[threadIdx.x] += b[threadIdx.x];
}|}

let has_infix ~affix s =
  let n = String.length affix in
  let rec at i =
    i + n <= String.length s && (String.sub s i n = affix || at (i + 1))
  in
  at 0

let rejection f =
  match f () with _ -> None | exception Compiler.Compile_error msg -> Some msg

(* NVRTC's absence is a Compile_error that names the variable to set. *)
let absent msg = has_infix ~affix:"NVRTC_PATH" msg

(* [with_nvrtc f] is [f ()], or skips the test if NVRTC is not on the
   machine. *)
let with_nvrtc f =
  match f () with
  | v -> v
  | exception Compiler.Compile_error msg when absent msg -> skip ~reason:msg ()

(* The cache *)

let ccache_off f = Setting.context [ B (Setting.ccache, false) ] f

(* [named ~prefix table] checks that [table] is [prefix] then a digest. *)
let named ~prefix table =
  let n = Int.min (String.length prefix) (String.length table) in
  equal string ~msg:"the name" prefix (String.sub table 0 n);
  equal int ~msg:"the digest's length" 32 (String.length table - n)

let table c = Option.get (Compiler.cachekey c)

let cache =
  group "cache"
    [
      test
        "binaries are cached in the table of cuda, the architecture and a \
         digest" (fun () ->
          named ~prefix:"compile_cuda_sm_89_"
            (table (Compiler_cuda.nvrtc "sm_89")));
      test "cubins and PTX are cached in different tables" (fun () ->
          not_equal string
            (table (Compiler_cuda.nvrtc "sm_89"))
            (table (Compiler_cuda.nvrtc ~ptx:false "sm_89")));
      test "cache_key names the table" (fun () ->
          named ~prefix:"compile_nv_sm_120_"
            (table (Compiler_cuda.nvrtc ~ptx:false ~cache_key:"nv" "sm_120")));
      test "the table is named with ccache off" (fun () ->
          let table () = Compiler.cachekey (Compiler_cuda.nvrtc "sm_89") in
          equal (option string) (table ()) (ccache_off table));
    ]

(* Without NVRTC on the machine *)

let without_nvrtc =
  group "without NVRTC on the machine"
    [
      test "a compile raises Compile_error naming nvrtc and NVRTC_PATH"
        (fun () ->
          match
            rejection (fun () ->
                Compiler.compile (Compiler_cuda.nvrtc "sm_89") kernel)
          with
          | None -> skip ~reason:"NVRTC is on the machine" ()
          | Some msg -> in_order ~subs:[ "nvrtc"; "NVRTC_PATH" ] msg);
    ]

(* Workers *)

(* [with_signal s behavior f] is [f ()] run with [behavior] for [s]. *)
let with_signal s behavior f =
  let previous = Sys.signal s behavior in
  Fun.protect ~finally:(fun () -> Sys.set_signal s previous) f

(* [f ()] with SIGPIPE's default action, which ends the process: a process
   may start with it ignored. *)
let sigpipe_kills f = with_signal Sys.sigpipe Signal_default f

(* The pids of this process's children, which pgrep lists, in order; the test
   skips where pgrep does not run. *)
let children () =
  let pid = string_of_int (Unix.getpid ()) in
  match Unix.open_process_args_in "pgrep" [| "pgrep"; "-P"; pid |] with
  | exception Unix.Unix_error _ -> skip ~reason:"pgrep does not run" ()
  | ic -> (
      let listed = In_channel.input_all ic in
      match Unix.close_process_in ic with
      | WEXITED (0 | 1) ->
          List.sort compare
            (List.filter_map int_of_string_opt
               (String.split_on_char '\n' listed))
      | _ -> skip ~reason:"pgrep failed" ())

(* Kills this process's children, idle workers, and waits for each to end. *)
let kill_children () =
  List.iter
    (fun pid ->
      Unix.kill pid Sys.sigkill;
      ignore (Unix.waitpid [] pid))
    (children ())

(* A library that does not load: these tests run in a process of their own,
   where NVRTC_PATH names a file that is no library, which each worker fails to
   load. *)

let load_failure () =
  rejection (fun () -> Compiler.compile (Compiler_cuda.nvrtc "sm_89") kernel)

let not_a_library =
  group ~tags:[ "no-library" ] "a library that does not load"
    [
      test "making a compiler loads nothing, and a compile raises Compile_error"
        (fun () ->
          let msg = require_some (load_failure ()) in
          in_order ~subs:[ "nvrtc"; "not_a_library" ] msg);
      test "every compile raises the same Compile_error" (fun () ->
          let first = load_failure () in
          equal (option string) first (load_failure ());
          is_some first);
      test "compiles from several domains at once raise it" (fun () ->
          let domains = List.init 4 (fun _ -> Domain.spawn load_failure) in
          let msgs = List.map Domain.join domains in
          is_some (List.hd msgs);
          List.iter (equal (option string) (List.hd msgs)) msgs);
      test "a compile in turn is served by an idle worker" (fun () ->
          ignore (load_failure ());
          let workers = children () in
          not_equal (list int) [] workers;
          ignore (load_failure ());
          equal (list int) workers (children ()));
      test "a compile after its idle workers are killed raises the same error"
        (fun () ->
          let first = load_failure () in
          kill_children ();
          equal (list int) [] (children ());
          equal (option string) first (sigpipe_kills load_failure));
      test "a cached binary is served without loading NVRTC" (fun () ->
          let nvrtc = Compiler_cuda.nvrtc "sm_89" in
          let table = Option.get (Compiler.cachekey nvrtc) in
          Helpers.Diskcache.put ~table kernel "cached binary";
          equal string "cached binary" (Compiler.compile_cached nvrtc kernel));
    ]

(* Disassembly *)

let disassembly =
  group "disassembly"
    [
      test "a PTX that ptxas cannot assemble prints why" (fun () ->
          Compiler.disassemble (Compiler_cuda.nvrtc "sm_89") "not ptx";
          flush stdout;
          contains ~sub:"ptxas -arch=sm_89 " (output ()));
      test "a cubin that nvdisasm cannot read prints why" (fun () ->
          Compiler.disassemble
            (Compiler_cuda.nvrtc ~ptx:false "sm_89")
            "not a cubin";
          flush stdout;
          let text = output () in
          contains ~sub:"nvdisasm " text;
          not_contains ~sub:"ptxas -arch" text);
    ]

(* NVRTC *)

let smoke =
  group "NVRTC"
    [
      slow "compiles a kernel to PTX" (fun () ->
          let ptx =
            with_nvrtc (fun () ->
                Compiler.compile (Compiler_cuda.nvrtc "sm_89") kernel)
          in
          in_order ~subs:[ ".target sm_89"; ".entry add" ] ptx);
      slow "compiles a kernel to a cubin" (fun () ->
          let cubin =
            with_nvrtc (fun () ->
                Compiler.compile (Compiler_cuda.nvrtc ~ptx:false "sm_89") kernel)
          in
          starts_with ~affix:"\x7fELF" cubin);
      slow "a rejected source raises Compile_error with NVRTC's log" (fun () ->
          match
            Compiler.compile
              (Compiler_cuda.nvrtc "sm_89")
              "extern \"C\" __global__ void f() { undeclared_name = 1; }"
          with
          | _ -> fail "NVRTC accepted an undeclared name"
          | exception Compiler.Compile_error msg when absent msg ->
              skip ~reason:msg ()
          | exception Compiler.Compile_error msg ->
              in_order
                ~subs:[ "NVRTC_ERROR_COMPILATION"; "undeclared_name" ]
                msg);
      slow "compiles from several domains at once" (fun () ->
          let compile arch () =
            Compiler.compile (Compiler_cuda.nvrtc arch) kernel
          in
          ignore (with_nvrtc (compile "sm_80"));
          let domains =
            List.map
              (fun arch -> Domain.spawn (compile arch))
              [ "sm_75"; "sm_80"; "sm_86"; "sm_89" ]
          in
          List.iter2
            (fun arch ptx -> contains ~sub:(".target " ^ arch) ptx)
            [ "sm_75"; "sm_80"; "sm_86"; "sm_89" ]
            (List.map Domain.join domains));
    ]

(* A compile in a process of its own *)

(* A valid kernel on which the assembler of NVRTC 13.3 and 13.4 crashes, for
   sm_80 to sm_89 with [--fmad=false]: deep chains of fused multiply-adds inside
   square roots, under the register limit of 1,024 threads. *)
let crashes_assembler =
  {|extern "C" __global__ void __launch_bounds__(1024) test(float* data0_65536, float* data1_65536, float* data4_128, float* data7_128, float* data9_1920, float* data10_65536) {
  int lidx0 = threadIdx.x;
  int lidx1 = threadIdx.y;
  int lidx2 = threadIdx.z;
  int alu0 = ((lidx1<<7));
  int alu1 = (lidx0<<2);
  int alu65 = (alu0+alu1);
  float4 val23 = (*((float4*)((data1_65536+alu65))));
  float4 val24 = (*((float4*)((data4_128))));
  float4 val25 = (*((float4*)((data7_128))));
  float4 val27 = (*((float4*)((data9_1920+(alu1)))));
  float4 val28 = (*((float4*)((data9_1920))));
  float4 val29 = (*((float4*)((1280))));
  float4 val30 = (*((float4*)((1408))));
  float4 val31 = (*((float4*)((1536))));
  float4 val32 = (*((float4*)((1664))));
  float4 val33 = (*((float4*)((1792))));
  float4 val34 = (*((float4*)((256))));
  float4 val35 = (*((float4*)((384))));
  float4 val36 = (*((float4*)((512))));
  float4 val37 = (*((float4*)((640))));
  float4 val38 = (*((float4*)((768))));
  float4 val39 = (*((float4*)((896))));
  float4 val40 = (*((float4*)((data9_1920))));
  float4 cast0 = make_float4((((val24.x+sqrt(((val24.x*val24.x)+4.0f)))*((__fmaf_rn(0.0f,val33.x,__fmaf_rn(0.0f,val32.x,__fmaf_rn(0.0f,val31.x,__fmaf_rn(0.0f,val30.x,__fmaf_rn(0.0f,val29.x,__fmaf_rn(0.0f,val27.x,__fmaf_rn(0.0f,val39.x,__fmaf_rn(0.0f,val38.x,__fmaf_rn(0.0f,val37.x,(val36.x)))))))))))))),((((((val24.y))))*(((val25.y)-val23.y)+(__fmaf_rn(0.0f,val33.y,__fmaf_rn(0.0f,val32.y,__fmaf_rn(0.0f,val31.y,__fmaf_rn(0.0f,val30.y,__fmaf_rn(0.0f,val29.y,__fmaf_rn(0.0f,val27.y,__fmaf_rn(0.0f,val39.y,__fmaf_rn(0.0f,val38.y,__fmaf_rn(0.0f,val37.y,__fmaf_rn(0.0f,val36.y,(val40.y))))))))))))))),((((((val24.z))))*(((val25.z)-val23.z)+(__fmaf_rn(0.0f,val33.z,__fmaf_rn(0.0f,val32.z,__fmaf_rn(0.0f,val31.z,__fmaf_rn(0.0f,val30.z,__fmaf_rn(0.0f,val29.z,__fmaf_rn(0.0f,val27.z,__fmaf_rn(0.0f,val39.z,__fmaf_rn(0.0f,val38.z,__fmaf_rn(0.0f,val37.z,__fmaf_rn(0.0f,val36.z,__fmaf_rn(0.0f,val35.z,__fmaf_rn(0.0f,val34.z,(val28.z))))))))))))))))),(0.0f+((sqrt(((val24.w))))*(((val25.w)-val23.w)+(__fmaf_rn(0.0f,val33.w,__fmaf_rn(0.0f,val32.w,__fmaf_rn(0.0f,val31.w,__fmaf_rn(0.0f,val30.w,__fmaf_rn(0.0f,val29.w,__fmaf_rn(0.0f,val27.w,__fmaf_rn(0.0f,val39.w,__fmaf_rn(0.0f,val38.w,__fmaf_rn(0.0f,val37.w,__fmaf_rn(0.0f,val36.w,__fmaf_rn(0.0f,val35.w,__fmaf_rn(0.0f,val34.w,(val28.w))))))))))))))))));
  bool alu70 = (lidx2==0);
  if (alu70) {
    *((float4*)((data0_65536+alu65))) = cast0;
    *((float4*)((data10_65536))) = cast0;
  }
}|}

let cubin src = Compiler.compile (Compiler_cuda.nvrtc ~ptx:false "sm_89") src

(* The files in NVRTC's cache of binaries, the directory the test's dune file
   names in CUDA_CACHE_PATH. *)
let nvrtc_cache () =
  let dir = Sys.getenv "CUDA_CACHE_PATH" in
  let rec files dir =
    Array.to_list (Sys.readdir dir)
    |> List.concat_map (fun f ->
        let path = Filename.concat dir f in
        if Sys.is_directory path then files path else [ path ])
  in
  if Sys.file_exists dir then files dir else []

let process =
  group "compiles in processes of their own"
    [
      slow
        "a source that crashes NVRTC's assembler compiles, or raises \
         Compile_error naming the signal"
        (fun () ->
          match cubin crashes_assembler with
          | lib -> starts_with ~affix:"\x7fELF" lib
          | exception Compiler.Compile_error msg when absent msg ->
              skip ~reason:msg ()
          | exception Compiler.Compile_error msg ->
              in_order ~subs:[ "nvrtc worker was killed by SIG"; "without a reply" ]
                msg);
      slow "compiles of one kernel are equal, in turn and from several domains"
        (fun () ->
          let lib = with_nvrtc (fun () -> cubin kernel) in
          equal string lib (cubin kernel);
          let domains =
            List.init 4 (fun _ -> Domain.spawn (fun () -> cubin kernel))
          in
          List.iter (equal string lib) (List.map Domain.join domains));
      slow "a compile after a crash returns its binary" (fun () ->
          let lib = with_nvrtc (fun () -> cubin kernel) in
          ignore (rejection (fun () -> cubin crashes_assembler));
          equal string lib (cubin kernel));
      slow "a compile after its idle workers are killed returns its binary"
        (fun () ->
          let lib = with_nvrtc (fun () -> cubin kernel) in
          kill_children ();
          equal string lib (sigpipe_kills (fun () -> cubin kernel)));
      slow "a worker started with SIGCHLD ignored returns its binary" (fun () ->
          let lib = with_nvrtc (fun () -> cubin kernel) in
          kill_children ();
          equal string lib
            (with_signal Sys.sigchld Signal_ignore (fun () -> cubin kernel)));
      slow "compiles leave NVRTC's cache of binaries empty" (fun () ->
          ignore (with_nvrtc (fun () -> cubin kernel));
          ignore (Compiler.compile (Compiler_cuda.nvrtc "sm_89") kernel);
          equal (list string) [] (nvrtc_cache ()));
    ]

(* A process that cannot start: these tests run in a process of their own, where
   NVRTC_PATH names a file, and forbid it to start any other for the rest of its
   life. *)

external forbid_spawn : unit -> unit = "tolk_test_forbid_spawn"

let no_spawn =
  group ~tags:[ "no-spawn" ] "a process that cannot start"
    [
      test "a compile raises Failure naming the worker" (fun () ->
          forbid_spawn ();
          raises_match (Exn.failure ~substring:"nvrtc worker") (fun () ->
              cubin kernel));
    ]

let () =
  exit
    (run "Tolk.Compiler_cuda"
       [
         cache; without_nvrtc; not_a_library; disassembly; smoke; process; no_spawn;
       ])
