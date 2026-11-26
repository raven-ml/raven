open Windtrap
open Tolk
module Compiler = Renderer.Compiler

let kernel =
  {|extern "C" __attribute__((global)) void add(float *a, const float *b) {
  a[0] += b[0];
}|}

(* A loop that exits on its data, written as the renderer writes a loop of no
   bound, its test at the end: it counts a nonzero word's set bits by clearing
   the lowest until none is left, which loop idiom recognition replaces by a
   population count. *)
let bit_count =
  {|extern "C" __attribute__((global)) void count(unsigned long *a, int *c) {
  unsigned long x = a[0];
  int n = 0;
  if (x != 0) {
    for (;;) {
      x &= x - 1;
      n++;
      if (!(x != 0)) { break; }
    }
  }
  c[0] = n;
}|}

(* The branches of [text], a disassembly, that jump backward: an [s_cbranch_*]
   whose 16-bit word offset, printed unsigned, is negative. *)
let backward_branches text =
  String.split_on_char '\n' text
  |> List.filter_map (fun line ->
      match String.split_on_char ' ' (String.trim line) with
      | op :: offset :: _ when String.starts_with ~prefix:"s_cbranch" op -> (
          match int_of_string_opt offset with
          | Some o when o >= 0x8000 -> Some op
          | _ -> None)
      | _ -> None)

(* Instructions, which HIP would reject. *)
let assembly = ".text\n.globl f\n.type f,@function\nf:\n  s_endpgm\n"

let has_infix ~affix s =
  let n = String.length affix in
  let rec at i =
    i + n <= String.length s && (String.sub s i n = affix || at (i + 1))
  in
  at 0

let rejection f =
  match f () with _ -> None | exception Compiler.Compile_error msg -> Some msg

(* comgr's absence is a Failure, raised when a compiler is made, that names the
   variable to set. *)
let absent msg = has_infix ~affix:"COMGR_PATH" msg

(* [with_comgr f] is [f ()], or skips the test if comgr is not on the
   machine. *)
let with_comgr f =
  match f () with
  | v -> v
  | exception Failure msg when absent msg -> skip ~reason:msg ()

let hip arch = with_comgr (fun () -> Compiler_amd.hip arch)
let compile ?(arch = "gfx1100") src = Compiler.compile (hip arch) src

(* An intrinsic of CDNA's matrix cores, which code generation for RDNA cannot
   select: LLVM ends the process that compiles it. *)
let unselectable =
  {|extern "C" __attribute__((device)) float mfma(float, float, float)
  __asm("llvm.amdgcn.mfma.f32.32x32x1f32");
extern "C" __attribute__((global)) void f(float *a) { a[0] = mfma(a[1], a[2], a[3]); }|}

(* A call of an intrinsic with an argument it does not take: IR that LLVM's
   verifier rejects and that code generation compiles without it, as it would IR
   that an optimization makes invalid. *)
let invalid_ir =
  {|extern "C" __attribute__((device)) int wid(int) __asm("llvm.amdgcn.workitem.id.x");
extern "C" __attribute__((global)) void f(int *a) { a[0] = wid(a[1]); }|}

(* Workers *)

(* The pids of process [pid]'s children, which pgrep lists, in order; the test
   skips where pgrep does not run. *)
let children_of pid =
  let pid = string_of_int pid in
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

let children () = children_of (Unix.getpid ())

(* Kills this process's children, idle workers, and waits for each to end. *)
let kill_children () =
  List.iter
    (fun pid ->
      Unix.kill pid Sys.sigkill;
      ignore (Unix.waitpid [] pid))
    (children ())

(* [remove path] removes the file or the directory tree at [path], following no
   symbolic link. *)
let rec remove path =
  match (Unix.lstat path).st_kind with
  | S_DIR ->
      Array.iter (fun n -> remove (Filename.concat path n)) (Sys.readdir path);
      Unix.rmdir path
  | _ -> Unix.unlink path

(* A directory of the run's own, made this process's temporary directory, where
   workers' temporary directories are made, and AMD_COMGR_CACHE_DIR's parent,
   where comgr would keep its own cache: what a compile leaves behind is seen
   there. The workers idle before it are killed, so that every later one starts
   under it. *)
let scratch =
  fixture ~teardown:remove (fun () ->
      let dir = Filename.temp_dir "tolk-test-comgr" "" in
      Filename.set_temp_dir_name dir;
      Unix.putenv "AMD_COMGR_CACHE_DIR" (Filename.concat dir "comgr-cache");
      kill_children ();
      dir)

let comgr_cache () = Filename.concat (scratch ()) "comgr-cache"

(* The paths under [dir] of comgr's temporary directories, named comgr-XXXXXX,
   which a compile that aborts leaves behind. *)
let rec comgr_files dir =
  List.concat_map
    (fun name ->
      let path = Filename.concat dir name in
      if String.starts_with ~prefix:"comgr-" name then [ path ]
      else
        match (Unix.lstat path).st_kind with
        | S_DIR -> comgr_files path
        | _ -> [])
    (Array.to_list (Sys.readdir dir))

(* [kernel]'s code object from a worker that has compiled nothing else: idle
   workers are killed first, so that a compile in turn is compared with one that
   no earlier compile can have touched. *)
let fresh_reference () =
  ignore (hip "gfx1100");
  kill_children ();
  compile kernel

(* A library that does not load: these tests run in a process of their own,
   where COMGR_PATH names a file that is no library, which the compiler finds
   and each worker fails to load. *)

let load_failure () =
  rejection (fun () -> Compiler.compile (Compiler_amd.hip "gfx1100") kernel)

let no_library = tag "no-library"

(* The table *)

let table c = Option.get (Compiler.cachekey c)

(* [named ~prefix table] checks that [table] is [prefix] then a digest. *)
let named ~prefix table =
  let n = Int.min (String.length prefix) (String.length table) in
  equal string ~msg:"the name" prefix (String.sub table 0 n);
  equal int ~msg:"the digest's length" 32 (String.length table - n)

let ccache_off f = Setting.context [ B (Setting.ccache, false) ] f

(* [path] made a copy of the file that is no library. *)
let copy_of_no_library path =
  let bytes = In_channel.with_open_bin "not_a_library" In_channel.input_all in
  Out_channel.with_open_bin path (fun oc -> output_string oc bytes)

let table_tests =
  group "the table"
    [
      no_library
      @@ test
           "code objects are cached in the table of hip, the architecture and \
            a digest" (fun () ->
             named ~prefix:"compile_hip_gfx1100_"
               (table (Compiler_amd.hip "gfx1100")));
      no_library
      @@ test "the table is named with ccache off" (fun () ->
          let table () = Compiler.cachekey (Compiler_amd.hip "gfx1100") in
          equal (option string) (table ()) (ccache_off table));
      no_library
      @@ test "a library that changes changes the table" (fun () ->
          let file = Filename.concat (temp_dir ()) "libamd_comgr.so" in
          copy_of_no_library file;
          setenv "COMGR_PATH" (Some file);
          let before = table (Compiler_amd.hip "gfx1100") in
          Out_channel.with_open_bin file (fun oc -> output_string oc "changed");
          not_equal string before (table (Compiler_amd.hip "gfx1100")));
    ]

(* Kernels the HIP renderer writes

   LLVM folds a float32 converted exactly from an integer and narrowed to
   bfloat16 into one conversion from the integer, and narrows a widened bool to
   an [i1], from which AMD's backend selects no conversion to bfloat16: the
   compile aborts. *)

(* The kernel storing in slot 0, as bfloat16, [f] of each of 64 elements of slot
   1, of [src], named so that code generation lowers it as a launched kernel. *)
let to_bf16 src f =
  let n = 64 in
  let r = Ops.range ~axis_type:Weak (Int n) [ 0 ] in
  let at slot dt = Ops.index (Shape.param ~shape:[ Int n ] slot dt) [ r ] in
  let v = f (Ops.load (at 1 src) []) in
  Ops.sink
    ~kernel:(Ops.kernel_info ~name:"to_bf16" ())
    [ Ops.end_ (Ops.store (at 0 Dtype.Bfloat16) v) [ r ] ]

let directly x = Ops.cast x Bfloat16
let through_float32 x = directly (Ops.cast x Float32)
let through_narrow x = directly (Decomp_dtype.narrow x Float32)

let casts_to_bf16 =
  List.concat_map
    (fun arch ->
      List.concat_map
        (fun src ->
          List.map
            (fun (name, route) -> (arch, src, name, route))
            [
              ("directly", directly);
              ("through a float32", through_float32);
              ("through narrow", through_narrow);
            ])
        Dtype.[ Bool; Int8; Uint8; Int16; Uint16; Int32; Uint32; Int64; Uint64 ])
    [ "gfx1100"; "gfx1201"; "gfx942"; "gfx950" ]

(* The binary of [prg], a program {!Codegen.compile} completed: its fourth
   source. *)
let binary prg =
  match Ops.arg (Ops.nth prg 3) with
  | Bytes b -> b
  | _ -> invalid_arg "a program's fourth source is its binary"

let compiles_to_bf16 (arch, src, _, route) =
  let target =
    {
      Helpers.Target.device = "AMD";
      renderer = "HIP";
      arch;
      interface = "";
      indices = "";
    }
  in
  let ren = with_comgr (fun () -> Cstyle.hip target) in
  let prg = Codegen.compile (Codegen.linearize (to_bf16 src route) ren) ren in
  starts_with ~prefix:"\x7fELF" (binary prg)

let casts_to_bf16_compile =
  cases "compiles each integer or bool cast to bfloat16 the HIP renderer writes"
    ~id:(fun (arch, src, name, _) ->
      Format.asprintf "%s %a %s" arch Dtype.pp src name)
    casts_to_bf16 compiles_to_bf16

let hip_tests =
  group "hip"
    [
      test "compiles a kernel to a code object" (fun () ->
          let lib = compile kernel in
          starts_with ~prefix:"\x7fELF" lib;
          contains ~sub:"add" lib);
      test "assembles a source whose first line is .text" (fun () ->
          starts_with ~prefix:"\x7fELF" (compile assembly));
      casts_to_bf16_compile;
      test "a loop that exits on its data is computed as written" (fun () ->
          let hip = hip "gfx1201" in
          let lib = Compiler.compile hip bit_count in
          ignore (output ());
          Compiler.disassemble hip lib;
          flush stdout;
          let text = output () in
          equal (list string) ~msg:"bit counts" []
            (List.filter
               (fun w -> has_infix ~affix:w text)
               [ "s_bcnt1"; "v_bcnt" ]);
          not_equal (list string) ~msg:"backward branches" []
            (backward_branches text));
      test "invalid IR raises Compile_error with the verifier's finding"
        (fun () ->
          ignore (hip "gfx1100");
          (match Compiler_amd.comgr_version () with
          | None -> fail "comgr compiled but reports no version"
          | Some v when v < (2, 8) ->
              skip ~reason:"comgr before 2.8 compiles without the verifier" ()
          | Some _ -> ());
          match compile invalid_ir with
          | _ -> fail "comgr compiled IR that LLVM's verifier rejects"
          | exception Compiler.Compile_error msg ->
              in_order
                ~subs:
                  [
                    "Intrinsic has incorrect argument type";
                    "Broken module found";
                  ]
                msg);
      table_tests;
      no_library
      @@ test
           "finds libamd_comgr.so in lib of the directory Setting.rocm_path \
            names" (fun () ->
             let rocm = temp_dir () in
             let lib = Filename.concat rocm "lib" in
             Unix.mkdir lib 0o755;
             let file = Filename.concat lib "libamd_comgr.so" in
             copy_of_no_library file;
             setenv "COMGR_PATH" None;
             let hip =
               Setting.context
                 [ B (Setting.rocm_path, rocm) ]
                 (fun () -> Compiler_amd.hip "gfx1100")
             in
             match Compiler.compile hip kernel with
             | _ -> fail "a file that is no library compiled"
             | exception Compiler.Compile_error msg ->
                 in_order ~subs:[ "comgr"; file ] msg);
      test "without comgr, making a compiler raises Failure naming COMGR_PATH"
        (fun () ->
          match Compiler_amd.hip "gfx1100" with
          | _ -> skip ~reason:"comgr is on the machine" ()
          | exception Failure msg ->
              in_order ~subs:[ "comgr"; "COMGR_PATH" ] msg);
      test "a rejected source raises Compile_error with comgr's step and log"
        (fun () ->
          match
            compile
              "extern \"C\" __attribute__((global)) void f() { undeclared_name \
               = 1; }"
          with
          | _ -> fail "comgr accepted an undeclared name"
          | exception Compiler.Compile_error msg ->
              in_order ~subs:[ "compile failed"; "undeclared_name" ] msg);
      test
        "a source that code generation rejects raises Compile_error with \
         comgr's step and log" (fun () ->
          let src =
            {|extern "C" __attribute__((global)) void f(int *a) {
  asm volatile("v_not_an_instruction v0");
  a[0] = 1;
}|}
          in
          match compile src with
          | _ -> fail "code generation assembled an unknown instruction"
          | exception Compiler.Compile_error msg ->
              in_order
                ~subs:[ "code generation failed"; "cannot compile inline asm" ]
                msg);
      cases "a source comgr rejects raises Compile_error"
        ~id:(Printf.sprintf "%S")
        [
          "";
          "\xff\xfe\x7fELF\000\001";
          "}{";
          String.sub kernel 0 40;
          ".text\n  not_an_instruction v0\n";
        ]
        (fun src ->
          let hip = hip "gfx1100" in
          raises_match
            (function Compiler.Compile_error _ -> true | _ -> false)
            (fun () -> Compiler.compile hip src));
      test
        "a source that ends its process raises Compile_error with how it ended \
         and LLVM's error" (fun () ->
          match compile unselectable with
          | _ -> fail "code generation selected a CDNA intrinsic for RDNA"
          | exception Compiler.Compile_error msg ->
              satisfies string ~claim:"says how the worker ended"
                (fun msg ->
                  has_infix ~affix:"comgr worker was killed by" msg
                  || has_infix ~affix:"comgr worker exited with" msg)
                msg;
              in_order ~subs:[ "without a reply"; "LLVM ERROR" ] msg);
      no_library
      @@ test
           "a library that does not load raises Compile_error naming the file"
           (fun () ->
             let msg = require_some (load_failure ()) in
             in_order ~subs:[ "comgr"; "not_a_library" ] msg);
      (* This test runs in a process of its own, where COMGR_PATH names a file,
         and forbids it to start any other for the rest of its life. *)
      tag "no-spawn"
      @@ test "a process that cannot start raises Failure naming the worker"
           (fun () ->
             Spawn.forbid ();
             raises_match (Exn.failure ~sub:"comgr worker") (fun () ->
                 Compiler.compile (Compiler_amd.hip "gfx1100") kernel));
      test "disassembles a code object as llvm-objdump does" (fun () ->
          let hip = hip "gfx1100" in
          let lib = Compiler.compile hip kernel in
          ignore (output ());
          Compiler.disassemble hip lib;
          flush stdout;
          contains ~sub:"s_endpgm" (output ()));
    ]

(* Compiles in processes of their own *)

let archs = [ "gfx90a"; "gfx942"; "gfx1100"; "gfx1201" ]

(* A worker whose parent dies *)

(* [kernel] including the FIFO [fifo]: its compile waits in the preprocessor for
   a writer to open [fifo], then for the writer to close it. *)
let waiting fifo = Printf.sprintf "#include \"%s\"\n%s" fifo kernel

(* The temporary directory of worker [pid], its program's second argument, as
   Linux's /proc/PID/cmdline lists them. *)
let temp_of pid =
  let cmdline = Printf.sprintf "/proc/%d/cmdline" pid in
  match In_channel.with_open_bin cmdline In_channel.input_all with
  | s -> (
      match String.split_on_char '\000' s with
      | _ :: _ :: temp :: _ -> Some temp
      | _ -> None)
  | exception Sys_error _ -> None

(* Whether process [pid] has ended: there is none, or it waits for its parent to
   collect it (Linux's /proc says so). *)
let ended pid =
  match Unix.kill pid 0 with
  | exception Unix.Unix_error (ESRCH, _, _) -> true
  | () -> (
      let stat = Printf.sprintf "/proc/%d/stat" pid in
      match In_channel.with_open_bin stat In_channel.input_all with
      | exception Sys_error _ -> true
      | s -> (
          match String.rindex_opt s ')' with
          | Some i -> i + 2 < String.length s && s.[i + 2] = 'Z'
          | None -> false))

(* Returns once process [pid] has ended, asking every millisecond: no clock
   decides, and WINDTRAP_TIMEOUT ends a wait for an end that never comes. *)
let rec until_ended pid =
  if not (ended pid) then begin
    Unix.sleepf 0.001;
    until_ended pid
  end

(* In the forked parent: forks a process that sleeps until killed, holding what
   this one holds, and writes its pid to [out]. *)
let fork_sleeper out =
  match Unix.fork () with
  | 0 ->
      Unix.close out;
      Unix.sleepf 60.;
      Unix._exit 0
  | pid ->
      let s = string_of_int pid in
      ignore (Unix.write_substring out s 0 (String.length s));
      Unix.close out

(* The pid that [fork_sleeper] writes to [input], once it has. *)
let read_sleeper input =
  let pid = In_channel.input_all (Unix.in_channel_of_descr input) in
  Unix.close input;
  require_some ~msg:"the forked process" (int_of_string_opt pid)

(* The parent compiles [waiting fifo]; the test's open of [fifo] for writing
   returns once the compile has opened it, which then reads until the test
   closes it. The forked process holds the parent's end of the worker's input
   open, so only the parent's death ends the worker, and comgr's files in the
   worker's directory go with it. *)
let parent_dies_in_a_compile () =
  ignore (compile kernel);
  let fifo = Filename.concat (temp_dir ()) "include.fifo" in
  (match Unix.mkfifo fifo 0o600 with
  | () -> ()
  | exception (Unix.Unix_error _ | Invalid_argument _) ->
      skip ~reason:"no FIFOs here" ());
  flush_all ();
  let input, out = Unix.pipe ~cloexec:true () in
  let parent =
    match Unix.fork () with
    | exception Invalid_argument _ -> skip ~reason:"no fork here" ()
    | 0 ->
        Unix.close input;
        (try
           ignore (compile kernel);
           fork_sleeper out;
           ignore (compile (waiting fifo))
         with _ -> ());
        Unix._exit 0
    | pid -> pid
  in
  Unix.close out;
  let sleeper = read_sleeper input in
  Fun.protect ~finally:(fun () -> Unix.kill sleeper Sys.sigkill) @@ fun () ->
  let writer = Unix.openfile fifo [ O_WRONLY; O_CLOEXEC ] 0 in
  Fun.protect ~finally:(fun () -> Unix.close writer) @@ fun () ->
  let workers = List.filter (fun pid -> pid <> sleeper) (children_of parent) in
  let temps = List.filter_map temp_of workers in
  Unix.kill parent Sys.sigkill;
  ignore (Unix.waitpid [] parent);
  not_equal (list int) ~msg:"the parent's workers" [] workers;
  List.iter until_ended workers;
  not_equal (list string) [] temps;
  equal (list string) ~msg:"their temporary directories" []
    (List.filter Sys.file_exists temps)

let process =
  group "compiles in processes of their own"
    [
      test
        "kernels compiled on several domains at once are the bytes they are \
         compiled to in turn" (fun () ->
          let in_turn = List.map (fun arch -> compile ~arch kernel) archs in
          let domains =
            List.map
              (fun arch -> Domain.spawn (fun () -> compile ~arch kernel))
              archs
          in
          equal (list string) in_turn (List.map Domain.join domains));
      no_library
      @@ test "a compile in turn is served by the idle worker, with its answer"
           (fun () ->
             let first = load_failure () in
             let workers = children () in
             not_equal (list int) [] workers;
             equal (option string) first (load_failure ());
             equal (list int) ~msg:"workers" workers (children ()));
      test "compiles in turn for other architectures are each one's" (fun () ->
          let lib = fresh_reference () in
          let other = compile ~arch:"gfx1201" kernel in
          not_equal string lib other;
          equal string lib (compile kernel);
          equal string other (compile ~arch:"gfx1201" kernel));
      test "a compile after a rejected source returns its code object"
        (fun () ->
          let lib = fresh_reference () in
          raises_match
            (function Compiler.Compile_error _ -> true | _ -> false)
            (fun () -> compile "}{");
          equal string lib (compile kernel));
      test
        "a compile after a source that ends its process returns its code object"
        (fun () ->
          let lib = fresh_reference () in
          raises_match
            (function Compiler.Compile_error _ -> true | _ -> false)
            (fun () -> compile unselectable);
          equal string lib (compile kernel));
      test "a compile writes nothing to comgr's own cache" (fun () ->
          let cache = comgr_cache () in
          ignore (compile kernel);
          ignore (compile ~arch:"gfx1201" kernel);
          let entries =
            if Sys.file_exists cache then Array.to_list (Sys.readdir cache)
            else []
          in
          equal (list string) [] entries);
      test
        "a worker in a compile whose parent is killed ends and removes its \
         directory, though a process forked from the parent lives"
        parent_dies_in_a_compile;
      test "a compile that ends its process leaves no temporary file" (fun () ->
          let dir = scratch () in
          raises_match
            (function Compiler.Compile_error _ -> true | _ -> false)
            (fun () -> compile unselectable);
          equal (list string) [] (comgr_files dir));
    ]

let () = exit (run "Tolk.Compiler_amd" [ hip_tests; process ])
