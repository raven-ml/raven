(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Any domain may call any function, as the interface says. The C state is
   written by [make], then by [room] and [submit] under the caller's turn;
   [local] holds the device's lock while it grows the local memory, and [stop]
   while it marks the device stopped; regions' [live] flags only detect
   misuse. *)

module D = Defs
module Abi = Rig_nv_abi
module Packet = Abi.Packet
module Method = Abi.Method
module Cubin = Abi.Cubin

let strf = Printf.sprintf
let invalid_argf fmt = Printf.ksprintf invalid_arg fmt
let ( let* ) = Result.bind

exception Fault of string

(* Paths *)

type params =
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

type rm = {
  release : int;
  client : int;
  alloc : parent:int -> int -> params option -> (int, string) result;
  control : int -> int -> params option -> (unit, string) result;
  free : parent:int -> int -> (unit, string) result;
}

type gpu = {
  channel_class : int;
  compute_class : int;
  copy_class : int;
  sm_version : int;
  gpcs : int;
  tpcs_per_gpc : int;
  sms_per_tpc : int;
  warps_per_sm : int;
}

type 'm memory = { address : int; host : int option; handle : int; data : 'm }

type 'm path = {
  key : 'm Type.Id.t;
  index : int;
  rm : rm;
  device : int;
  subdevice : int;
  vaspace : int;
  gpu : gpu;
  budget : int;
  doorbell : int;
  alloc : [ `Gpu | `Bar | `System ] -> int -> 'm memory option;
  map_host : (int -> int -> 'm memory option) option;
  reaches : int -> bool;
  map_peer : 'm memory -> 'm memory option;
  free : 'm memory -> unit;
  register : int -> (unit, string) result;
  unregister : int -> (unit, string) result;
  check : unit -> unit;
  hang_ms : int option;
  stop : unit -> [ `Stopped | `Unknown ];
}

let is_gpu ~vendor ~class_ = vendor = 0x10de && class_ lsr 16 = 0x03

(* The memory [m] of [n] bytes a path answered to the function [fn], below
   [address_limit] as the path's interface promises: a path that breaks it is
   given [m] back. *)
let address_limit = 1 lsl 40

let below (p : _ path) fn n = function
  | Some (m : _ memory) when m.address + n > address_limit ->
      p.free m;
      invalid_argf "%s: the path answered memory at 0x%x, past 2^40" fn
        m.address
  | m -> m

let path_alloc p fn kind n = below p fn n (p.alloc kind n)

(* Constants *)

let page = 4096

(* The device's words: the timeline word, then the channels' join words, as
   rig_nv_stubs.h's JOIN_GPU lays them out. *)
let words_bytes = 8 * 3

(* Each channel's ring holds [entries] entries, and its segment ring
   [segment_bytes] bytes, both powers of two: a submission takes at least one
   entry of each channel it uses, so thousands may be in flight. *)
let entries = 16384
let segment_bytes = 1 lsl 20

(* The addresses at which kernels see their shared and local memory, above 2^40,
   where no memory the device maps lies. *)
let shared_window = 0x7294_0000_0000
let local_window = 0x7293_0000_0000

(* rig_nv_stubs.h's templates, by index. *)
let t_acquire = 0
let t_release = 1
let t_copy_release = 2
let t_copy = 3
let t_local = 4
let t_setup = 5
let t_setup_copy = 6
let t_invalidate = 7
let t_idle = 8

(* A pending local memory is one word: its address, below 2^40, and its bytes
   per cluster in units of 32 KiB above them. *)
let local_address_bits = 40
let local_unit_shift = 15

(* The multiprocessors whose errors the RM reports, at most. *)
let sm_errors =
  let _, _, n = D.Sm_error_states.sm_error_state_array in
  n

(* The C state *)

external create : int -> int -> int -> int -> int = "caml_rig_nv_create"
[@@noalloc]

external destroy : int -> unit = "caml_rig_nv_destroy" [@@noalloc]

external set_channel : int -> int -> int array -> bool = "caml_rig_nv_channel"
[@@noalloc]

external set_doorbell : int -> int -> unit = "caml_rig_nv_doorbell" [@@noalloc]

external set_template :
  int -> int -> string -> (int * int Packet.word) list -> unit
  = "caml_rig_nv_template"

external set_entry : int -> int -> int -> unit = "caml_rig_nv_entry" [@@noalloc]
external set_bar : int -> int -> unit = "caml_rig_nv_bar" [@@noalloc]
external bar_live : int -> int -> unit = "caml_rig_nv_bar_live" [@@noalloc]
external zero : int -> int -> unit = "caml_rig_nv_zero" [@@noalloc]

external offer_local : int -> int -> int -> bool = "caml_rig_nv_offer_local"
[@@noalloc]

external set_local : int -> int -> unit = "caml_rig_nv_set_local" [@@noalloc]
external pending_local : int -> int = "caml_rig_nv_pending_local" [@@noalloc]
external owe_invalidate : int -> unit = "caml_rig_nv_owe_invalidate" [@@noalloc]
external read_word : int -> int = "caml_rig_nv_signaled" [@@noalloc]

external notification : int -> int -> int = "caml_rig_nv_notification"
[@@noalloc]

external watch : int -> int -> int -> bool = "caml_rig_nv_watch"
external last : int -> int = "caml_rig_nv_last" [@@noalloc]
external now_ms : unit -> int = "caml_rig_nv_now_ms" [@@noalloc]
external raise_word : int -> unit = "caml_rig_nv_raise" [@@noalloc]
external end_channels : int -> unit = "caml_rig_nv_end" [@@noalloc]
external room_entry_address : unit -> int = "caml_rig_nv_room_entry" [@@noalloc]

external submit_entry_address : unit -> int = "caml_rig_nv_submit_entry"
[@@noalloc]

(* RM parameters *)

external get16 : params -> int -> int = "%caml_bigstring_get16"
external get32 : params -> int -> int32 = "%caml_bigstring_get32"
external get64 : params -> int -> int64 = "%caml_bigstring_get64"
external set16 : params -> int -> int -> unit = "%caml_bigstring_set16"
external set32 : params -> int -> int32 -> unit = "%caml_bigstring_set32"
external set64 : params -> int -> int64 -> unit = "%caml_bigstring_set64"

let params n =
  let p = Bigarray.Array1.create Bigarray.char Bigarray.c_layout n in
  Bigarray.Array1.fill p '\000';
  p

(* A field is (byte offset, bytes); the generated tables hold widths of 1, 2, 4
   and 8 bytes only. *)
let get p (at, n) =
  match n with
  | 1 -> Char.code (Bigarray.Array1.get p at)
  | 2 -> get16 p at
  | 4 -> Int32.to_int (get32 p at) land 0xffff_ffff
  | _ -> Int64.to_int (get64 p at)

let set p (at, n) v =
  match n with
  | 1 -> Bigarray.Array1.set p at (Char.unsafe_chr (v land 0xff))
  | 2 -> set16 p at (v land 0xffff)
  | 4 -> set32 p at (Int32.of_int v)
  | _ -> set64 p at (Int64.of_int v)

(* Element [i] of the array field (offset, bytes of an element, elements), and
   the field [f] of that element. *)
let elt (at, n, _) i = (at + (i * n), n)
let elt_field (at, n, _) i (f, fn) = (at + (i * n) + f, fn)
let bits (lo, n) v = (v land ((1 lsl n) - 1)) lsl lo

(* Devices *)

(* A region is memory its path gave, through the GPU's BAR or not, or the
   timeline word. *)
type kind = Path | Bar | Word

type 'm dev = {
  path : 'm path;
  self : int;
  arch : string;
  error_names : (int * string) list;
  capability : Abi.Gpu.t;
  word : 'm reg;
  owned : 'm memory list;
  bar : bool;
  group : int;
  debugger : int;
  channels : int list;
  compute_channel : int;
  progress : progress Atomic.t;
  local_lock : Mutex.t;
  mutable stopped : bool;
  mutable per_thread : int;
  mutable local_current : 'm memory option;
  mutable local_pending : ('m memory * int) option;
  mutable local_retired : ('m memory * int) list;
}

(* The word as [sleep] last saw it, whether the device was idle then, and since
   when, in milliseconds of the monotonic clock. *)
and progress = { seen : int; idle : bool; since : int }

and 'm reg = {
  dev : 'm dev;
  mem : 'm memory;
  bytes : int;
  kind : kind;
  live : bool Atomic.t;
}

type t = T : 'm dev -> t
type region = R : 'm reg -> region

let key : t Type.Id.t = Type.Id.make ()

(* [r], if it is a region of [d]. *)
let mine : type m. m dev -> region -> m reg option =
 fun d (R r) ->
  if r.dev.self <> d.self then None
  else
    match Type.Id.provably_equal d.path.key r.dev.path.key with
    | Some Type.Equal -> Some r
    | None -> None

(* Facts *)

(* The architecture of SM version [v]: its major and minor numbers, but for
   Blackwell's 0xa04, which is sm_120. *)
let arch_of v =
  if v = 0xa04 then "sm_120"
  else
    let minor = v land 0xff in
    strf "sm_%d%d"
      ((v lsr 8) land 0xff)
      (if minor > 0xf then minor lsr 4 else minor)

(* The SASS version of SM version [v], as cubins state it: major and minor in
   one byte each nibble. *)
let sass_of v = ((v land 0xf00) lsr 4) lor (v land 0xf)
let arch (T d) = d.arch
let budget (T d) = d.path.budget
let queues (T _) = [ "COMPUTE:0"; "COPY:0" ]
let completion (T _) = `Store
let waits_on (T _) = function `Store | `Host -> true | `Object -> false

(* rig_nv_ring.c's MAX_WAITS. *)
let max_waits (T _) = 256
let blocks (T _) = `Returns
let maps_host (T d) = Option.is_some d.path.map_host

type capability = Abi.Gpu.t

let capability (T d) = d.capability
let capability_key = Abi.Gpu.key
let self (T d) = Nativeint.of_int d.self

(* Gives [m] back to its path. The device holds [m] no more whatever the path
   answers: a path that fails to take it back keeps it. *)
let give d m = try d.path.free m with Fault _ -> ()

(* Local memory *)

(* Moves the pending local memory to current once a submission took it. The
   current one serves the values before that submission, all at most the last
   value submitted once the taking is seen: it retires with that value. *)
let settle d =
  match d.local_pending with
  | Some (m, packed) when pending_local d.self <> packed ->
      let reached = last d.self in
      Option.iter
        (fun c -> d.local_retired <- (c, reached) :: d.local_retired)
        d.local_current;
      d.local_current <- Some m;
      d.local_pending <- None
  | Some _ | None -> ()

(* Frees the retired local memories no work uses: those whose successor's value
   is reached. *)
let retire d =
  let seen = read_word d.self in
  let done_, kept = List.partition (fun (_, v) -> v <= seen) d.local_retired in
  d.local_retired <- kept;
  List.iter (fun (m, _) -> give d m) done_

(* Makes [packed] the pending local memory, freeing a pending one that no
   submission took. *)
let publish d m packed =
  (match d.local_pending with
  | Some (old, old_packed) when offer_local d.self old_packed packed ->
      give d old
  | Some _ | None ->
      settle d;
      set_local d.self packed);
  d.local_pending <- Some (m, packed)

let local d n =
  Mutex.protect d.local_lock @@ fun () ->
  if d.stopped then Error "the device is stopped"
  else begin
    settle d;
    retire d;
    let l = Abi.Local_memory.make d.capability n in
    if l.per_thread <= d.per_thread then Ok ()
    else
      match path_alloc d.path "Rig_nv.capability" `Gpu l.bytes with
      | exception Fault why -> Error why
      | None ->
          Error (strf "no GPU memory for %d bytes of local memory" l.bytes)
      | Some m ->
          publish d m
            (m.address
            lor ((l.per_tpc lsr local_unit_shift) lsl local_address_bits));
          d.per_thread <- l.per_thread;
          Ok ()
  end

(* Templates *)

(* Sets template [k] of [self] to [p]: the words of [p] with a hole for every
   value [known] does not give. *)
let template self k ~known p =
  let words, holes = Packet.template known p in
  set_template self k words holes

let unknown _ = None
let known v = Some (Int64.of_int v)

let templates self (g : gpu) =
  (* The writer's values, by slot: two addresses or an address and a value, then
     a size. *)
  let a = 0 and b = 1 and n = 2 in
  let system = Packet.System in
  template self t_acquire ~known:unknown (Method.acquire a b);
  template self t_release ~known:unknown (Method.release system a b);
  template self t_copy_release ~known:unknown (Method.copy_release system a b);
  template self t_copy ~known:unknown (Method.copy ~dst:a ~src:b n);
  template self t_local ~known:unknown (Method.local_memory a ~per_tpc:b);
  template self t_setup ~known
    (Method.set_object Method.Compute g.compute_class
    @ Method.local_memory_window local_window
    @ Method.shared_memory_window shared_window);
  template self t_setup_copy ~known (Method.set_object Method.Copy g.copy_class);
  template self t_invalidate ~known (Method.invalidate_caches system);
  template self t_idle ~known Method.wait_for_idle;
  (* An entry is its segment's address plus a constant plus its words times
     another: the address is a term's value, and the words a field of their
     own. *)
  let entry words =
    let e = Abi.Gpfifo.entry 0 ~offset:0 ~words in
    Int64.to_int (String.get_int64_le (Packet.encode Int64.of_int e) 0)
  in
  set_entry self (entry 0) (entry 1 - entry 0)

(* Opening *)

(* A channel's place in the block of channel memory: the two rings, the two
   USERDs a page each, then the two segment rings. *)
let ring_bytes = 8 * entries
let ring_at q = q * ring_bytes
let userd_at q = (2 * ring_bytes) + (q * page)
let segments_at q = (2 * ring_bytes) + (2 * page) + (q * segment_bytes)
let block_bytes = segments_at 2
let in_rm what = Result.map_error (fun e -> strf "%s: %s" what e)

let host what (m : _ memory) =
  match m.host with
  | Some h -> Ok h
  | None -> Error (strf "the %s is not mapped for the host" what)

(* The device's objects and memory, each given back by [taken] if a later step
   fails. *)
let start (type m) (p : m path) (module R : D.RELEASE) ~taken =
  let rm = p.rm in
  let alloc kind bytes what =
    match path_alloc p "Rig_nv.make" kind bytes with
    | None -> Error (strf "no memory for the %s" what)
    | Some m ->
        taken (fun () -> p.free m);
        Ok m
  in
  let new_object ~parent cls what fill size =
    let q = params size in
    fill q;
    let* h = in_rm what (rm.alloc ~parent cls (Some q)) in
    taken (fun () -> ignore (rm.free ~parent h));
    Ok h
  in
  let* words = alloc `System page "timeline word" in
  let* words_host = host "timeline word" words in
  zero words_host words_bytes;
  let* block = alloc `System block_bytes "channels' rings" in
  let* block_host = host "channels' rings" block in
  let notifier q =
    let* m = alloc `System page (strf "error notifier of channel %d" q) in
    let* h = host "error notifier" m in
    zero h page;
    Ok (m, h)
  in
  let* compute_notifier = notifier 0 in
  let* copy_notifier = notifier 1 in
  let bar = path_alloc p "Rig_nv.make" `Bar page in
  Option.iter (fun m -> taken (fun () -> p.free m)) bar;
  let* group =
    new_object ~parent:p.device D.kepler_channel_group_a "the channel group"
      (fun q ->
        set q R.Channel_group_alloc.engine_type D.nv2080_engine_type_graphics)
      R.Channel_group_alloc.sizeof
  in
  let* ctxshare =
    let module C = D.Ctxshare_alloc in
    new_object ~parent:group D.fermi_context_share_a "the context share"
      (fun q ->
        set q C.h_va_space p.vaspace;
        set q C.flags
          (bits D.nv_ctxshare_allocation_flags_subcontext
             D.nv_ctxshare_allocation_flags_subcontext_async))
      C.sizeof
  in
  let self =
    create words_host words.address
      (fst D.Notification.info32)
      (fst D.Notification.status)
  in
  if self = 0 then Error "no host memory for the device's state"
  else
    let () = taken (fun () -> destroy self) in
    let channel q engine (notifier, notifier_host) =
      let module G = R.Gpfifo_alloc in
      let* ch =
        new_object ~parent:group p.gpu.channel_class (strf "channel %d" q)
          (fun f ->
            set f G.gp_fifo_offset (block.address + ring_at q);
            set f G.gp_fifo_entries entries;
            set f G.h_object_error notifier.handle;
            set f G.h_object_buffer block.handle;
            set f (elt G.h_userd_memory 0) block.handle;
            set f (elt G.userd_offset 0) (userd_at q);
            set f G.h_context_share ctxshare)
          G.sizeof
      in
      let* engine_object =
        in_rm "an engine" (rm.alloc ~parent:ch engine None)
      in
      let token = params D.Work_submit_token.sizeof in
      set token D.Work_submit_token.work_submit_token 0xffff_ffff;
      let* () =
        in_rm "the work submit token"
          (rm.control ch D.nvc36f_ctrl_cmd_gpfifo_get_work_submit_token
             (Some token))
      in
      let* () = in_rm (strf "registering channel %d" q) (p.register ch) in
      taken (fun () -> ignore (p.unregister ch));
      let ints =
        [|
          block_host + ring_at q;
          entries;
          block_host + userd_at q + fst D.Userd.gp_put;
          get token D.Work_submit_token.work_submit_token;
          block_host + segments_at q;
          block.address + segments_at q;
          segment_bytes;
          notifier_host;
        |]
      in
      if not (set_channel self q ints) then
        Error "no host memory for the channels' state"
      else Ok (ch, engine_object)
    in
    let* compute, compute_engine =
      channel 0 p.gpu.compute_class compute_notifier
    in
    let* copy, _ = channel 1 p.gpu.copy_class copy_notifier in
    let* debugger =
      let module A = D.Nv83de_alloc in
      new_object ~parent:p.device D.gt200_debugger "the debugger"
        (fun q ->
          set q A.h_app_client rm.client;
          set q A.h_class3d_object compute_engine)
        A.sizeof
    in
    let schedule = params R.Group_schedule.sizeof in
    set schedule R.Group_schedule.b_enable 1;
    let* () =
      in_rm "scheduling the channels"
        (rm.control group D.nva06c_ctrl_cmd_gpfifo_schedule (Some schedule))
    in
    let boost = params D.Perf_boost.sizeof in
    set boost D.Perf_boost.duration D.nv2080_ctrl_perf_boost_duration_infinite;
    set boost D.Perf_boost.flags
      (bits D.nv2080_ctrl_perf_boost_flags_cuda
         D.nv2080_ctrl_perf_boost_flags_cuda_yes
      lor bits D.nv2080_ctrl_perf_boost_flags_cuda_priority
            D.nv2080_ctrl_perf_boost_flags_cuda_priority_high
      lor bits D.nv2080_ctrl_perf_boost_flags_cmd
            D.nv2080_ctrl_perf_boost_flags_cmd_boost_to_max);
    let* () =
      in_rm "raising the GPU's clocks"
        (rm.control p.subdevice D.nv2080_ctrl_cmd_perf_boost (Some boost))
    in
    set_doorbell self p.doorbell;
    templates self p.gpu;
    let* () =
      match bar with
      | None -> Ok ()
      | Some m ->
          let* h = host "BAR page" m in
          Ok (set_bar self h)
    in
    let owned = [ block; fst compute_notifier; fst copy_notifier ] in
    let g = p.gpu in
    let rec d =
      {
        path = p;
        self;
        arch = arch_of g.sm_version;
        error_names = R.robust_channel_errors;
        capability =
          {
            Abi.Gpu.compute_class = g.compute_class;
            sass_version = sass_of g.sm_version;
            gpcs = g.gpcs;
            tpcs_per_gpc = g.tpcs_per_gpc;
            sms_per_tpc = g.sms_per_tpc;
            warps_per_sm = g.warps_per_sm;
            shared_window;
            local_window;
            local = (fun n -> local d n);
          };
        word =
          {
            dev = d;
            mem = words;
            bytes = 8;
            kind = Word;
            live = Atomic.make true;
          };
        owned = (match bar with Some m -> m :: owned | None -> owned);
        bar = Option.is_some bar;
        group;
        debugger;
        channels = [ compute; copy ];
        compute_channel = compute;
        progress = Atomic.make { seen = 0; idle = true; since = 0 };
        local_lock = Mutex.create ();
        stopped = false;
        per_thread = 0;
        local_current = None;
        local_pending = None;
        local_retired = [];
      }
    in
    Ok (T d)

let make p =
  Option.iter
    (fun n ->
      if n < 1 then
        invalid_argf "Rig_nv.make: a hang bound of %d ms, expected at least 1" n)
    p.hang_ms;
  match D.release p.rm.release with
  | None ->
      Error
        (strf "the RM's release %d is none of %s" p.rm.release
           (String.concat ", " (List.map string_of_int D.releases)))
  | Some release -> (
      let undo = ref [] in
      let taken f = undo := f :: !undo in
      (* A failure giving something back does not hide the open's. *)
      let give_back () =
        List.iter (fun f -> try f () with Fault _ -> ()) !undo
      in
      match start p release ~taken with
      | exception Fault why ->
          give_back ();
          Error why
      | Error _ as e ->
          give_back ();
          e
      | Ok _ as d -> d)

(* Memory *)

let region d kind bytes m =
  R { dev = d; mem = m; bytes; kind; live = Atomic.make true }

let alloc (T d) kind n =
  if n < 1 then invalid_argf "Rig_nv.alloc: %d bytes, expected at least 1" n;
  let p = d.path and fn = "Rig_nv.alloc" in
  match kind with
  | `Device -> Option.map (region d Path n) (path_alloc p fn `Gpu n)
  | `Pinned -> Option.map (region d Path n) (path_alloc p fn `System n)
  | `Mapped -> (
      match if d.bar then path_alloc p fn `Bar n else None with
      | Some m ->
          bar_live d.self 1;
          Some (region d Bar n m)
      | None -> Option.map (region d Path n) (path_alloc p fn `System n))

let free (T d) r =
  match mine d r with
  | None -> invalid_arg "Rig_nv.free: the region is not the device's"
  | Some r -> (
      if not (Atomic.compare_and_set r.live true false) then
        invalid_arg "Rig_nv.free: the region was freed";
      give d r.mem;
      if r.kind = Bar then bar_live d.self (-1))

let address (R r) = Some r.mem.address
let handle (R r) = Nativeint.of_int r.mem.address
let host (R r) = r.mem.host

let peer (T d) (T d') =
  d.self <> d'.self
  &&
  match Type.Id.provably_equal d.path.key d'.path.key with
  | Some Type.Equal -> d.path.reaches d'.path.index
  | None -> false

let map_peer (T d) (T d') r =
  if d.self = d'.self then
    invalid_arg "Rig_nv.map_peer: the two devices are one";
  match mine d' r with
  | None -> invalid_arg "Rig_nv.map_peer: the region is not the other device's"
  | Some r when not (Atomic.get r.live) ->
      invalid_arg "Rig_nv.map_peer: the region was freed"
  | Some r -> (
      match Type.Id.provably_equal d.path.key d'.path.key with
      | None -> None
      | Some Type.Equal ->
          let m = d.path.map_peer r.mem in
          Option.map (region d Path r.bytes)
            (below d.path "Rig_nv.map_peer" r.bytes m))

let map_host (T d) a n =
  if n < 1 then invalid_argf "Rig_nv.map_host: %d bytes, expected at least 1" n;
  match d.path.map_host with
  | None -> None
  | Some map ->
      let m = below d.path "Rig_nv.map_host" n (map a n) in
      Option.map (region d Path n) m

(* Images *)

(* An image is a cubin laid over a region of its device: the address of the
   region's first byte. *)
type image = {
  owner : int;
  base : int;
  cubin : Cubin.t;
  loaded : bool Atomic.t;
}

(* The image of [c] for an upload at [base]: its object's image, zeros up to its
   size, and its relocations' patches. *)
let image_bytes c ~base =
  let o = Cubin.elf c in
  let b = Bytes.make (Cubin.size c) '\000' in
  let put (s : Rig_elf.section) =
    match s.offset with
    | Some off -> Bytes.blit_string o.file s.at b off s.length
    | None -> ()
  in
  Iarray.iter put o.sections;
  let patch (at, p) = Bytes.blit_string p 0 b at (String.length p) in
  List.iter patch (Cubin.patches c ~base);
  Bytes.unsafe_to_string b

(* The cubin [c] laid over the region [r] of [d]. A cubin may load where another
   one's code was, so the compute channel owes an instruction cache
   invalidation. *)
let lay d c (R r) =
  let base = r.mem.address in
  owe_invalidate d.self;
  ( { owner = d.self; base; cubin = c; loaded = Atomic.make true },
    image_bytes c ~base )

let image (T d) bin =
  let* c = Cubin.of_string bin in
  Ok (`Place (Cubin.size c, lay d c))

let entry c name =
  if not (Atomic.get c.loaded) then
    invalid_arg "Rig_nv.entry: the image was unloaded";
  Option.map
    (fun (k : Cubin.kernel) -> c.base + k.code)
    (Cubin.kernel c.cubin name)

let unload (T d) c =
  if c.owner <> d.self then
    invalid_arg "Rig_nv.unload: the image is another device's";
  if not (Atomic.compare_and_set c.loaded true false) then
    invalid_arg "Rig_nv.unload: the image was unloaded"

(* Work *)

let room_entry = Nativeint.of_int (room_entry_address ())
let submit_entry = Nativeint.of_int (submit_entry_address ())

(* Timeline *)

let word (T d) = R d.word
let signaled (T d) = read_word d.self

let fault_name table v =
  match List.assoc_opt v table with Some n -> n | None -> strf "0x%x" v

(* The errors the RM wrote into the channels' notifiers when it stopped them,
   such as for a fault of a channel's own methods. *)
(* The RM writes an error of the channel group into each channel's notifier: a
   line for each different one. *)
let channel_errors d =
  let error q =
    let x = notification d.self q in
    if x = 0 then None
    else
      let code = x lsr 16 and status = x land 0xffff in
      Some
        (strf "channel error %d (%s), status 0x%x" code
           (fault_name d.error_names code)
           status)
  in
  match List.filter_map error [ 0; 1 ] with
  | [ a; b ] when a = b -> [ a ]
  | errors -> errors

(* The faults the multiprocessors or the MMU reported to the RM, one per
   line. *)
let sm_errors_of d =
  let module S = D.Sm_error_states in
  let module E = D.Sm_error_state in
  let rm = d.path.rm in
  let p = params S.sizeof in
  set p S.h_target_channel d.compute_channel;
  set p S.num_s_ms_to_read sm_errors;
  match
    rm.control d.debugger D.nv83de_ctrl_cmd_debug_read_all_sm_error_states
      (Some p)
  with
  | Error e -> [ strf "reading the multiprocessors' errors: %s" e ]
  | Ok () when get p S.mmu_fault_valid <> 0 -> (
      let module M = D.Mmu_fault_info in
      let module F = D.Mmu_fault_entry in
      let m = params M.sizeof in
      match
        rm.control d.debugger D.nv83de_ctrl_cmd_debug_read_mmu_fault_info
          (Some m)
      with
      | Error e -> [ strf "reading the MMU's faults: %s" e ]
      | Ok () ->
          List.init (get m M.count) (fun i ->
              let f x = elt_field M.mmu_fault_info_list i x in
              strf "MMU fault: 0x%X | %s | %s"
                (get m (f F.fault_address))
                (fault_name D.fault_types (get m (f F.fault_type)))
                (fault_name D.access_types (get m (f F.access_type)))))
  | Ok () ->
      List.filter_map
        (fun i ->
          let f x = elt_field S.sm_error_state_array i x in
          let global = get p (f E.hww_global_esr)
          and warp = get p (f E.hww_warp_esr) in
          if global = 0 && warp = 0 then None
          else
            Some
              (strf "SM %d fault: esr=0x%x warp_esr=0x%x warp_pc=0x%x" i global
                 warp
                 (get p (f E.hww_warp_esr_pc64))))
        (List.init sm_errors Fun.id)

let check_faults d =
  d.path.check ();
  match channel_errors d @ sm_errors_of d with
  | [] -> ()
  | report -> raise (Fault (String.concat "\n" report))

(* How long [sleep] may watch the word [w] under the hang bound [hang]. The
   clock restarts when the word moved or the device was idle at the last look,
   and runs on while the same value stays outstanding. *)
let bounded d w ~still_ms hang =
  let now = now_ms () and p = Atomic.get d.progress in
  let idle = last d.self <= w in
  if idle || p.idle || p.seen <> w then begin
    Atomic.set d.progress { seen = w; idle; since = now };
    if idle then still_ms else Int.min still_ms hang
  end
  else
    let left = p.since + hang - now in
    if left <= 0 then raise (Fault (strf "no progress for %d ms" hang));
    Int.min still_ms left

let sleep (T d) ~seen ~still_ms =
  if read_word d.self = seen then begin
    check_faults d;
    let ms =
      match d.path.hang_ms with
      | None -> still_ms
      | Some hang -> bounded d seen ~still_ms hang
    in
    if watch d.self seen ms then check_faults d
  end

(* Loss *)

let stop (T d) =
  Mutex.protect d.local_lock (fun () -> d.stopped <- true);
  let ok f =
    match f () with Ok () -> true | Error _ | (exception Fault _) -> false
  in
  let rm = d.path.rm in
  let unregistered =
    List.for_all Fun.id
      (List.map (fun ch -> ok (fun () -> d.path.unregister ch)) d.channels)
  in
  let freed =
    ok (fun () -> rm.free ~parent:d.path.device d.debugger)
    && ok (fun () -> rm.free ~parent:d.path.device d.group)
  in
  let path = try d.path.stop () with Fault _ -> `Unknown in
  if (unregistered && freed) || path = `Stopped then begin
    raise_word d.self;
    end_channels d.self;
    (* The work no longer runs: what it used goes back, and a failure to give
       some back leaves the device stopped. *)
    List.iter (give d) d.owned;
    Mutex.protect d.local_lock (fun () ->
        Option.iter (give d) d.local_current;
        Option.iter (fun (m, _) -> give d m) d.local_pending;
        List.iter (fun (m, _) -> give d m) d.local_retired)
  end
    (* The RM stopped the channels on a fault: nothing runs, though their
       objects stay. Otherwise their own releases bring the word up as their
       work ends; a word raised here could be lowered by a late release. *)
  else if channel_errors d <> [] then raise_word d.self
