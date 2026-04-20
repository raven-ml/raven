(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk

(* FFI Externals *)

external cpu_alloc : int -> nativeint = "caml_tolk_cpu_alloc"
external cpu_free : nativeint -> unit = "caml_tolk_cpu_free"
external cpu_copyin : nativeint -> bytes -> unit = "caml_tolk_cpu_copyin"
external cpu_copyout : bytes -> nativeint -> unit = "caml_tolk_cpu_copyout"
external exec_alloc : int -> nativeint = "caml_tolk_cpu_jit_alloc"
external exec_free : nativeint -> int -> unit = "caml_tolk_cpu_jit_free"
external exec_write : nativeint -> bytes -> unit = "caml_tolk_cpu_jit_write"

external exec_call : nativeint -> nativeint array -> int64 array -> unit
  = "caml_tolk_cpu_jit_call"

external link_symbol_raw : string array -> string -> nativeint
  = "caml_tolk_cpu_jit_link_symbol"

type loaded_program = { base : nativeint; entry : nativeint; size : int }

let link_symbol ?(libs = []) name =
  let libs = Array.of_list libs in
  link_symbol_raw libs name

let load_program ~name ~lib =
  let prepared =
    Elf_cpu_loader.load ~link_symbol ~entry:name lib
  in
  let size = Elf_cpu_loader.alloc_size prepared in
  let base = exec_alloc size in
  try
    let image = Elf_cpu_loader.link ~base prepared in
    exec_write base image;
    let entry =
      Nativeint.add base
        (Nativeint.of_int (Elf_cpu_loader.entry_offset prepared))
    in
    { base; entry; size }
  with exn ->
    exec_free base size;
    raise exn

let unload_program loaded = exec_free loaded.base loaded.size

(* Allocator *)

(* Tinygrad uses mmap (MAP_ANON | MAP_SHARED) for CPU buffers. Tolk uses
   calloc/free for simplicity. Mmap becomes relevant for shared-memory IPC or
   very large allocations; calloc suffices for single-process CPU execution. *)
let raw_allocator =
  let alloc size spec =
    match spec.Device.Buffer_spec.external_ptr with
    | Some ptr -> ptr
    | None -> cpu_alloc size
  in
  let free buf _size spec =
    match spec.Device.Buffer_spec.external_ptr with
    | Some _ -> ()
    | None -> cpu_free buf
  in
  {
    Device.Allocator.alloc;
    free;
    copyin = cpu_copyin;
    copyout = cpu_copyout;
    addr = Fun.id;
    offset = None;
    transfer = None;
    supports_transfer = false;
    copy_from_disk = None;
    supports_copy_from_disk = false;
  }

(* Execution Queue *)

(* Tinygrad uses recursive CPUWorker threads: each worker can spawn sub-workers
   for parallel kernel execution. Tolk uses a flat two-tier model: a single
   Thread dispatches tasks from the queue, and a shared Domain pool provides
   true parallelism for multi-threaded kernels. Domains (not Threads) are used
   for the pool because OCaml 5 Domains run on separate OS threads with
   independent minor heaps, giving actual CPU parallelism for kernel
   execution. *)
module Cpu_queue = struct
  type pool_job = Run of (unit -> unit) | Stop

  type pool = {
    tasks : pool_job Queue.t;
    mutex : Mutex.t;
    cond : Condition.t;
    mutable workers : unit Domain.t list;
  }

  let pool_create () =
    {
      tasks = Queue.create ();
      mutex = Mutex.create ();
      cond = Condition.create ();
      workers = [];
    }

  let rec pool_worker_loop pool =
    Mutex.lock pool.mutex;
    while Queue.is_empty pool.tasks do
      Condition.wait pool.cond pool.mutex
    done;
    let job = Queue.take pool.tasks in
    Mutex.unlock pool.mutex;
    match job with
    | Stop -> ()
    | Run fn ->
        fn ();
        pool_worker_loop pool

  let pool_start_worker pool = Domain.spawn (fun () -> pool_worker_loop pool)

  (* Only called from the single dispatch thread (worker), so no lock needed. *)
  let pool_ensure pool count =
    let existing = List.length pool.workers in
    if count > existing then
      let new_workers =
        List.init (count - existing) (fun _ -> pool_start_worker pool)
      in
      pool.workers <- pool.workers @ new_workers

  let pool_enqueue pool job =
    Mutex.lock pool.mutex;
    Queue.add (Run job) pool.tasks;
    Condition.signal pool.cond;
    Mutex.unlock pool.mutex

  let pool_shutdown pool =
    match pool.workers with
    | [] -> ()
    | workers ->
        Mutex.lock pool.mutex;
        List.iter (fun _ -> Queue.add Stop pool.tasks) workers;
        Condition.broadcast pool.cond;
        Mutex.unlock pool.mutex;
        List.iter Domain.join workers;
        pool.workers <- []

  type work = {
    entry : nativeint;
    bufs : nativeint array;
    vals : int64 array;
    threads : int;
    core_id_index : int option;
  }

  type task = Work of work | Stop

  type t = {
    tasks : task Queue.t;
    mutex : Mutex.t;
    cond : Condition.t;
    pool : pool;
    mutable worker_thread : unit Domain.t option;
    mutable pending : int;
    mutable error : exn option;
  }

  let run_kernel task tid =
    let vals = Array.copy task.vals in
    (match task.core_id_index with
    | None -> ()
    | Some idx ->
        if idx >= 0 && idx < Array.length vals then
          vals.(idx) <- Int64.of_int tid);
    exec_call task.entry task.bufs vals

  (* Fan out kernel execution across the Domain pool. Thread 0 runs on the
     dispatch thread; threads 1..N-1 are enqueued to pool workers. The dispatch
     thread blocks until all threads complete, propagating the first error. *)
  let run_task t task =
    let threads = max 1 task.threads in
    if threads = 1 then run_kernel task 0
    else (
      pool_ensure t.pool (threads - 1);
      let remaining = ref threads in
      let mutex = Mutex.create () in
      let cond = Condition.create () in
      let error : exn option ref = ref None in
      let record_error exn =
        Mutex.lock mutex;
        if !error = None then error := Some exn;
        Mutex.unlock mutex
      in
      let finish () =
        Mutex.lock mutex;
        remaining := !remaining - 1;
        if !remaining = 0 then Condition.signal cond;
        Mutex.unlock mutex
      in
      let run tid =
        (try run_kernel task tid with exn -> record_error exn);
        finish ()
      in
      for tid = 1 to threads - 1 do
        pool_enqueue t.pool (fun () -> run tid)
      done;
      run 0;
      Mutex.lock mutex;
      while !remaining > 0 do
        Condition.wait cond mutex
      done;
      let task_error = !error in
      Mutex.unlock mutex;
      match task_error with None -> () | Some exn -> raise exn)

  let worker t =
    let rec loop () =
      Mutex.lock t.mutex;
      while Queue.is_empty t.tasks do
        Condition.wait t.cond t.mutex
      done;
      let task = Queue.take t.tasks in
      Mutex.unlock t.mutex;
      match task with
      | Stop -> ()
      | Work work ->
          let error =
            try
              run_task t work;
              None
            with exn -> Some exn
          in
          Mutex.lock t.mutex;
          (match error with
          | None -> ()
          | Some exn -> if t.error = None then t.error <- Some exn);
          t.pending <- t.pending - 1;
          Condition.broadcast t.cond;
          Mutex.unlock t.mutex;
          loop ()
    in
    loop ()

  let create () =
    let pool = pool_create () in
    let t =
      {
        tasks = Queue.create ();
        mutex = Mutex.create ();
        cond = Condition.create ();
        pool;
        worker_thread = None;
        pending = 0;
        error = None;
      }
    in
    let worker_thread = Domain.spawn (fun () -> worker t) in
    t.worker_thread <- Some worker_thread;
    t

  let exec t ~entry ~bufs ~vals ~threads ~core_id_index =
    let task = Work { entry; bufs; vals; threads; core_id_index } in
    Mutex.lock t.mutex;
    Queue.add task t.tasks;
    t.pending <- t.pending + 1;
    Condition.signal t.cond;
    Mutex.unlock t.mutex

  let synchronize t =
    Mutex.lock t.mutex;
    while t.pending > 0 do
      Condition.wait t.cond t.mutex
    done;
    let error = t.error in
    t.error <- None;
    Mutex.unlock t.mutex;
    match error with None -> () | Some exn -> raise exn

  let shutdown t =
    match t.worker_thread with
    | None -> ()
    | Some worker ->
        Mutex.lock t.mutex;
        Queue.add Stop t.tasks;
        Condition.signal t.cond;
        Mutex.unlock t.mutex;
        Domain.join worker;
        t.worker_thread <- None;
        pool_shutdown t.pool
end

(* Device Registration *)

let create name =
  let clang =
    Compiler.make ~name:"CLANG" ~cachekey:"compile_clang_jit"
      ~compile:Compiler_cpu.compile_clang ()
  in
  let state = Cpu_queue.create () in
  at_exit (fun () -> Cpu_queue.shutdown state);
  let runtime entry_name lib ~runtimevars =
    let loaded = load_program ~name:entry_name ~lib in
    let entry = loaded.entry in
    let core_id_index = List.assoc_opt "core_id" runtimevars in
    let call bufs ~global ~local:_ ~vals ~wait ~timeout:_ =
      let threads = match core_id_index with
        | Some _ -> max 1 global.(0)
        | None -> 1
      in
      Cpu_queue.exec state ~entry ~bufs ~vals ~threads ~core_id_index;
      if wait then begin
        Cpu_queue.synchronize state;
        None
      end else
        None
    in
    let free () = unload_program loaded in
    Device.{ call; free }
  in
  let synchronize () = Cpu_queue.synchronize state in
  let renderer = Renderer.with_compiler clang Cstyle.clang in
  let renderer_set = Device.Renderer_set.make [renderer, None] in
  let allocator =
    Device.Allocator.Pack (Device.Lru_allocator.wrap raw_allocator)
  in
  Device.make ~name ~allocator ~renderer_set ~runtime ~synchronize ()
