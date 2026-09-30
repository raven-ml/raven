open Tolk_next

external submit_address : unit -> nativeint = "tolk_null_submit_address"
external dlsym : string -> nativeint = "tolk_null_dlsym"
external take : unit -> (string * float) list = "tolk_null_take"
external set_latency : float -> unit = "tolk_null_set_latency"
external outstanding : unit -> int = "tolk_null_outstanding"
external finished : int -> unit = "tolk_null_finished"
external read : nativeint -> int = "tolk_null_load"
external write : nativeint -> int -> unit = "tolk_null_store"
external copy : nativeint -> nativeint -> int -> unit = "tolk_null_copy"
external call : nativeint -> int array -> int array -> unit = "tolk_null_call"

(* Devices *)

(* Run's test devices, whose queues this module runs. A queue that hangs fails
   in ten seconds. *)
let nx_devices =
  lazy
    (List.filter_map
       (fun (name, d) ->
         if name = "CPU" then None
         else begin
           Nx_device.set_timeout d 10_000;
           Some (name, d)
         end)
       (Run.devices ()))

let device name =
  match List.assoc_opt name (Lazy.force nx_devices) with
  | Some d -> d
  | None -> invalid_arg (Printf.sprintf "%s is no NULL device" name)

(* Programs, loaded when their queue is encoded, by key *)

let lock = Mutex.create ()
let programs : (string, Nx_device.Program.t) Hashtbl.t = Hashtbl.create 16
let events = Null_queue.events ()

let loaded prg = Mutex.protect lock (fun () -> Hashtbl.find programs (Ops.key prg))

let load prg =
  let key = Ops.key prg in
  if not (Mutex.protect lock (fun () -> Hashtbl.mem programs key)) then begin
    let binary =
      match Ops.arg (List.nth (Ops.src prg) 3) with
      | Bytes b -> b
      | _ -> invalid_arg "a compiled program's fourth source is its binary"
    in
    let name = (Device.Tiny_elf.of_program prg).name in
    match Nx_device.Program.load Nx_device.host ~binary ~name with
    | Ok p -> Mutex.protect lock (fun () -> Hashtbl.replace programs key p)
    | Error why -> failwith why
  end

(* The commands *)

let u64 n = Ops.int ~dtype:Uint64 n

let commands q =
  let c = Null_queue.commands events q in
  let device = List.hd (Hcq2.Queue.devices q) in
  let exec call prg =
    load prg;
    c.exec call prg
  in
  let copy dst src n =
    ignore
      (Hcq2.Queue.q q [ u64 Null_queue.copy; Ops.getaddr ~device dst; Ops.getaddr ~device src; u64 n ])
  in
  let submit cmdbuf =
    let head = Ops.cast (Ops.load (Ops.index cmdbuf [ Ops.int 0 ]) []) Uint64 in
    Hcq2.ccall ~host:device ~lib:"null" "tolk_null_submit"
      [ Ops.getaddr ~device cmdbuf; u64 (Ops.max_numel cmdbuf * Dtype.itemsize (Ops.dtype cmdbuf)); head ]
  in
  { c with exec; copy; submit }

(* The words that hold the addresses of C functions, by device and function. *)
let function_words = Hashtbl.create 8

let function_word name f =
  match Hashtbl.find_opt function_words (name, f) with
  | Some b -> b
  | None ->
      let b = Nx_device.Buffer.create (device name) UInt64 1 in
      let host = Result.get_ok (Nx_device.Buffer.borrow Nx_device.host b) in
      let address = if f = "tolk_null_submit" then submit_address () else dlsym f in
      (Nx_device.Buffer.bigarray Bigarray.int64 host).{0} <- Int64.of_nativeint address;
      Hashtbl.add function_words (name, f) b;
      b

let placeholder name u =
  match Ops.tag u with
  | Some (Tuple [ String "cfunc"; String _; String f ]) -> Some (function_word name f)
  | _ -> None

(* Running queues *)

let failure = Atomic.make None

let with_latency s f =
  set_latency s;
  Fun.protect ~finally:(fun () -> set_latency 0.) f

(* A queue being run: its command words, the next one's index, and when it may
   start. *)
type queue = { words : int array; mutable pc : int; start : float }

let words_of s = Array.init (String.length s / 8) (fun k -> Int64.to_int (String.get_int64_le s (8 * k)))
let addr n = Nativeint.of_int n

let exec kernargs nargs event =
  let prg = Null_queue.program events event in
  let words = Array.init nargs (fun k -> read (addr (kernargs + (8 * k)))) in
  let signature = (Device.Tiny_elf.of_program prg).signature in
  let buffers = List.length (List.filter (fun (p : Device.Tiny_elf.param) -> p.shape <> []) signature) in
  call (Nx_device.Program.handle (loaded prg)) (Array.sub words 0 buffers)
    (Array.sub words buffers (nargs - buffers))

(* Runs [q]'s next command, and is [false] if it must wait. *)
let step q =
  let w k = q.words.(q.pc + k) in
  let runs =
    match w 0 with
    | op when op = Null_queue.wait -> read (addr (w 1)) >= w 2
    | op when op = Null_queue.exec -> exec (w 1) (w 2) (w 3); true
    | op when op = Null_queue.copy -> copy (addr (w 1)) (addr (w 2)) (w 3); true
    | op when op = Null_queue.store -> write (addr (w 1)) (w 2); true
    | op when op = Null_queue.timestamp -> write (addr (w 1)) (Nx_device.Profile.now ()); true
    | op ->
        failwith
          (Printf.sprintf "command %d of %d has the unknown code %d" (q.pc / 4) (Array.length q.words / 4) op)
  in
  if runs then q.pc <- q.pc + 4;
  runs

let stop = Atomic.make false

let rec serve queues =
  if not (Atomic.get stop) then begin
    let now = Unix.gettimeofday () in
    let fresh =
      List.map (fun (s, start) -> { words = words_of s; pc = 0; start }) (take ())
    in
    let queues = queues @ fresh in
    let progressed = ref false in
    List.iter
      (fun q ->
        if q.start <= now then
          while q.pc < Array.length q.words && step q do
            progressed := true
          done)
      queues;
    let left = List.filter (fun q -> q.pc < Array.length q.words) queues in
    finished (List.length queues - List.length left);
    let queues = left in
    if not !progressed then Unix.sleepf 0.00005;
    serve queues
  end

let rec served queues =
  match serve queues with
  | () -> ()
  | exception e ->
      Atomic.set failure (Some (Printexc.to_string e));
      finished (outstanding ());
      served []

let server =
  lazy
    (let d = Domain.spawn (fun () -> served []) in
     at_exit (fun () ->
         Atomic.set stop true;
         Domain.join d))

let synchronize () =
  Lazy.force server;
  while outstanding () > 0 && Atomic.get failure = None do
    Unix.sleepf 0.0001
  done;
  (match Atomic.get failure with Some why -> failwith why | None -> ());
  List.iter (fun (_, d) -> Nx_device.synchronize d) (Lazy.force nx_devices)

(* Devices, as the engine runs work on them *)

let devices ?(copy_queue = true) () =
  Lazy.force server;
  function
  | "CPU" -> Tolk_next_engine.device [ ("CPU", Nx_device.host) ] "CPU"
  | name ->
      let d = device name in
      let queues = { Hcq2.commands; copy_queue; host = "CPU"; reaches = (fun _ -> true) } in
      {
        Tolk_next_engine.device = d;
        compiler = { target = Tolk_next_engine.target d; queues = Some queues };
        placeholder = placeholder name;
        submitting = ignore;
      }
