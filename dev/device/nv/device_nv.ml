(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Any domain may call any function, as the interface says. The C state is
   written by [make], then by [room] and [submit] under the caller's turn;
   [local] holds the device's lock while it grows the local memory; regions'
   [live] flags only detect misuse. *)

module D = Defs
module Abi = Device_nv_abi
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

type 'm memory = {
  address : int;
  host : nativeint option;
  handle : int;
  data : 'm;
}

type 'm path = {
  key : 'm Type.Id.t;
  rm : rm;
  device : int;
  subdevice : int;
  vaspace : int;
  gpu : gpu;
  budget : int;
  doorbell : nativeint;
  alloc : [ `Gpu | `Bar | `System ] -> int -> 'm memory option;
  map_host : nativeint -> int -> 'm memory option;
  map_peer : 'm memory -> 'm memory option;
  free : 'm memory -> unit;
  register : int -> (unit, string) result;
  unregister : int -> (unit, string) result;
}

let is_gpu ~vendor ~class_ = vendor = 0x10de && class_ lsr 16 = 0x03

(* Constants *)

let page = 4096

(* Each channel's ring holds [entries] entries, and its segment ring
   [segment_bytes] bytes, both powers of two: a submission takes at least one
   entry of each channel it uses, so thousands may be in flight. *)
let entries = 16384
let segment_bytes = 1 lsl 20

(* The addresses at which kernels see their shared and local memory, above 2^40,
   where no memory the device maps lies. *)
let shared_window = 0x7294_0000_0000
let local_window = 0x7293_0000_0000

(* device_nv_ring.c's MAX_WAITS and MAX_PARTS. *)
let max_waits = 256
let max_parts = 65535

(* device_nv_stubs.h's templates, by index, and their bounds. *)
let t_acquire = 0
let t_release = 1
let t_copy_release = 2
let t_copy = 3
let t_local = 4
let t_setup = 5
let t_setup_copy = 6
let t_invalidate = 7
let template_words = 16
let template_holes = 6
let hole_ops = 3

(* A pending local memory is one word: its address, below 2^40, and its bytes
   per cluster in units of 32 KiB above them. *)
let local_address_bits = 40
let local_unit_shift = 15

(* The multiprocessors whose errors the RM reports, at most. *)
let sm_errors =
  let _, _, n = D.Sm_error_states.sm_error_state_array in
  n

(* The C state *)

external create : int -> int -> int -> int -> int = "caml_device_nv_create"

external set_channel : int -> int -> int array -> bool
  = "caml_device_nv_channel"

external set_doorbell : int -> int -> unit = "caml_device_nv_doorbell"

external set_template : int -> int -> int array -> int array -> unit
  = "caml_device_nv_template"

external set_entry : int -> int -> int -> unit = "caml_device_nv_entry"
external set_bar : int -> int -> unit = "caml_device_nv_bar"
external bar_live : int -> int -> unit = "caml_device_nv_bar_live"
external zero : int -> int -> unit = "caml_device_nv_zero"
external offer_local : int -> int -> int -> bool = "caml_device_nv_offer_local"
external set_local : int -> int -> unit = "caml_device_nv_set_local"
external pending_local : int -> int = "caml_device_nv_pending_local"
external local_placed : int -> int = "caml_device_nv_local_placed"
external owe_invalidate : int -> unit = "caml_device_nv_owe_invalidate"
external read_word : int -> int = "caml_device_nv_signaled" [@@noalloc]
external last : int -> int = "caml_device_nv_last" [@@noalloc]
external notification : int -> int -> int = "caml_device_nv_notification"
external watch : int -> int -> int -> bool = "caml_device_nv_watch"
external raise_word : int -> unit = "caml_device_nv_raise"
external end_channels : int -> unit = "caml_device_nv_end"
external room_entry_address : unit -> int = "caml_device_nv_room_entry"
external submit_entry_address : unit -> int = "caml_device_nv_submit_entry"

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

type kind = Allocation | Bar | Mapping | Word | Code

type 'm dev = {
  path : 'm path;
  self : int;
  arch : string;
  release : (module D.RELEASE);
  capability : Abi.Gpu.t;
  word : 'm reg;
  owned : 'm memory list;
  bar : bool;
  group : int;
  debugger : int;
  channels : int list;
  compute_channel : int;
  stopped : bool Atomic.t;
  local_lock : Mutex.t;
  mutable per_thread : int;
  mutable local_current : 'm memory option;
  mutable local_pending : ('m memory * int) option;
  mutable local_retired : ('m memory * int) list;
}

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

let arch (T d) = d.arch
let machine (T _) = None
let budget (T d) = d.path.budget
let queues (T _) = [ "COMPUTE:0"; "COPY:0" ]
let completion (T _) = `Store
let waits_on (T _) = function `Store | `Host -> true | `Object -> false
let blocks (T _) = `Returns

type capability = Abi.Gpu.t

let capability (T d) = d.capability
let capability_key = Abi.Gpu.key
let self (T d) = Nativeint.of_int d.self

(* Local memory *)

(* Moves the pending local memory to current once a submission took it, retiring
   the current one from the value that placed its successor. *)
let settle d =
  match d.local_pending with
  | Some (m, packed) when pending_local d.self <> packed ->
      let placed = local_placed d.self in
      Option.iter
        (fun c -> d.local_retired <- (c, placed) :: d.local_retired)
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
  List.iter (fun (m, _) -> d.path.free m) done_

(* Makes [packed] the pending local memory, freeing a pending one that no
   submission took. *)
let publish d m packed =
  (match d.local_pending with
  | Some (old, old_packed) when offer_local d.self old_packed packed ->
      d.path.free old
  | Some _ | None ->
      settle d;
      set_local d.self packed);
  d.local_pending <- Some (m, packed)

let local d n =
  Mutex.protect d.local_lock @@ fun () ->
  if Atomic.get d.stopped then Error "the device is stopped"
  else begin
    settle d;
    retire d;
    let l = Abi.Local_memory.make d.capability n in
    if l.per_thread <= d.per_thread then Ok ()
    else
      match d.path.alloc `Gpu l.bytes with
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

(* The ints of a hole as caml_device_nv_template reads them: its index, slot,
   width and number of operations, then each operation's shift (an addition if
   [0]) and addend. *)
let hole_ints (at, (word : int Packet.word)) =
  let rec ops : int Packet.term -> int * (int * int64) list = function
    | Packet.Value s -> (s, [])
    | Packet.Add (t, n) ->
        let s, l = ops t in
        (s, l @ [ (0, n) ])
    | Packet.Shift (t, n) ->
        let s, l = ops t in
        (s, l @ [ (n, 0L) ])
  in
  let wide, term =
    match word with
    | Packet.W32 t -> (0, t)
    | Packet.W64 t -> (1, t)
    | Packet.Dword _ -> invalid_arg "Device_nv: a template hole holds no value"
  in
  let slot, l = ops term in
  if List.length l > hole_ops then
    invalid_arg "Device_nv: a template hole takes too many operations";
  let op (shift, n) =
    let x = Int64.to_int n in
    if Int64.of_int x <> n then
      invalid_arg "Device_nv: a template's addend exceeds an int";
    [| shift; x |]
  in
  let pad = List.init (hole_ops - List.length l) (fun _ -> (0, 0L)) in
  Array.concat ([| at; slot; wide; List.length l |] :: List.map op (l @ pad))

(* Sets template [k] of [self] to [p]: the words of [p] with a hole for every
   value [known] does not give. *)
let template self k ~known p =
  let bytes, holes = Packet.template known p in
  let words =
    Array.init
      (String.length bytes / 4)
      (fun i ->
        Int32.to_int (String.get_int32_le bytes (4 * i)) land 0xffff_ffff)
  in
  if Array.length words > template_words || List.length holes > template_holes
  then invalid_arg "Device_nv: a template exceeds its bounds";
  set_template self k words (Array.concat (List.map hole_ints holes))

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
  | Some h -> Ok (Nativeint.to_int h)
  | None -> Error (strf "the %s is not mapped for the host" what)

(* The device's objects and memory, each given back by [taken] if a later step
   fails. *)
let start (type m) (p : m path) (module R : D.RELEASE) ~taken =
  let rm = p.rm in
  let alloc kind bytes what =
    match p.alloc kind bytes with
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
  zero words_host 24;
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
  let bar = p.alloc `Bar page in
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
    set_doorbell self (Nativeint.to_int p.doorbell);
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
        release = (module R : D.RELEASE);
        capability =
          {
            Abi.Gpu.compute_class = g.compute_class;
            sass_version =
              ((g.sm_version land 0xf00) lsr 4) lor (g.sm_version land 0xf);
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
        stopped = Atomic.make false;
        local_lock = Mutex.create ();
        per_thread = 0;
        local_current = None;
        local_pending = None;
        local_retired = [];
      }
    in
    Ok (T d)

let make p =
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

let reg d kind bytes m =
  { dev = d; mem = m; bytes; kind; live = Atomic.make true }

let region d kind bytes m = R (reg d kind bytes m)

let alloc (T d) kind n =
  if n < 1 then invalid_argf "Device_nv.alloc: %d bytes, expected at least 1" n;
  match kind with
  | `Device -> Option.map (region d Allocation n) (d.path.alloc `Gpu n)
  | `Pinned -> Option.map (region d Allocation n) (d.path.alloc `System n)
  | `Mapped -> (
      match if d.bar then d.path.alloc `Bar n else None with
      | Some m ->
          bar_live d.self 1;
          Some (region d Bar n m)
      | None -> Option.map (region d Allocation n) (d.path.alloc `System n))

let free (T d) r =
  match mine d r with
  | None -> invalid_arg "Device_nv.free: the region is not the device's"
  | Some r -> (
      match r.kind with
      | Word -> invalid_arg "Device_nv.free: the timeline word is never freed"
      | Code -> invalid_arg "Device_nv.free: an image's code is freed by unload"
      | Mapping -> invalid_arg "Device_nv.free: a mapping is ended by unmap"
      | Allocation | Bar ->
          if not (Atomic.compare_and_set r.live true false) then
            invalid_arg "Device_nv.free: the region was freed";
          d.path.free r.mem;
          if r.kind = Bar then bar_live d.self (-1))

let address (R r) = Some r.mem.address
let handle (R r) = Nativeint.of_int r.mem.address
let host (R r) = r.mem.host

let map_peer (T d) (T d') r =
  if d.self = d'.self then
    invalid_arg "Device_nv.map_peer: the two devices are one";
  match mine d' r with
  | None ->
      invalid_arg "Device_nv.map_peer: the region is not the other device's"
  | Some r when not (Atomic.get r.live) ->
      invalid_arg "Device_nv.map_peer: the region was freed or unmapped"
  | Some r -> (
      match Type.Id.provably_equal d.path.key d'.path.key with
      | None -> None
      | Some Type.Equal ->
          Option.map (region d Mapping r.bytes) (d.path.map_peer r.mem))

let map_host (T d) a n =
  if n < 1 then
    invalid_argf "Device_nv.map_host: %d bytes, expected at least 1" n;
  Option.map (region d Mapping n) (d.path.map_host a n)

let unmap (T d) r =
  match mine d r with
  | None -> invalid_arg "Device_nv.unmap: the region is not the device's"
  | Some r when r.kind <> Mapping ->
      invalid_arg "Device_nv.unmap: the region is no mapping"
  | Some r ->
      if not (Atomic.compare_and_set r.live true false) then
        invalid_arg "Device_nv.unmap: the region was unmapped";
      d.path.free r.mem

(* Images *)

type image = I : 'm img -> image
and 'm img = { code : 'm reg; cubin : Cubin.t }

(* The image of [c] for an upload at [base]: its object's image, zeros up to its
   size, and its relocations' patches. *)
let image_bytes c ~base =
  let o = Cubin.elf c in
  let b = Bytes.make (Cubin.size c) '\000' in
  let put (s : Device_elf.section) =
    match s.offset with
    | Some off -> Bytes.blit_string o.file s.at b off s.length
    | None -> ()
  in
  Iarray.iter put o.sections;
  let patch (at, p) = Bytes.blit_string p 0 b at (String.length p) in
  List.iter patch (Cubin.patches c ~base);
  Bytes.unsafe_to_string b

(* The image of the cubin [c] laid over [m], its code region, and the bytes that
   go there. A cubin may load where another one's code was, so the compute
   channel owes an instruction cache invalidation. *)
let lay d c m =
  let code = reg d Code (Cubin.size c) m in
  owe_invalidate d.self;
  (I { code; cubin = c }, R code, image_bytes c ~base:m.address)

let image (T d) bin =
  let* c = Cubin.of_string bin in
  let size = Cubin.size c in
  match d.path.alloc `Gpu size with
  | None -> Error (strf "no GPU memory for %d bytes of code" size)
  | Some m ->
      let img, code, bytes = lay d c m in
      Ok (img, Some (code, bytes))

let entry (I c) name =
  if not (Atomic.get c.code.live) then
    invalid_arg "Device_nv.entry: the image was unloaded";
  Option.map
    (fun (k : Cubin.kernel) -> c.code.mem.address + k.code)
    (Cubin.kernel c.cubin name)

let unload (T d) (I c) =
  match mine d (R c.code) with
  | None -> invalid_arg "Device_nv.unload: the image is another device's"
  | Some r ->
      if not (Atomic.compare_and_set r.live true false) then
        invalid_arg "Device_nv.unload: the image was unloaded";
      d.path.free r.mem

(* Work *)

(* A part is its device and the ints caml_device_nv_room reads: its queue, its
   copy's destination, offset, source, offset and bytes, its number of words,
   its words, then its [after] indices. *)
type part = { owner : int; ints : int array }

let fields = 7
let words_at = 6

let part (T d) ~queue ?(after = [||]) w =
  let q =
    match queue with
    | "COMPUTE:0" -> 0
    | "COPY:0" -> 1
    | _ ->
        invalid_argf "Device_nv.part: queue %S, expected COMPUTE:0 or COPY:0"
          queue
  in
  let check_after j =
    if j < 0 then invalid_argf "Device_nv.part: after index %d is negative" j
  in
  Array.iter check_after after;
  let part copy words =
    let head = Array.append [| q |] copy in
    let ints = Array.concat [ head; [| Array.length words |]; words; after ] in
    { owner = d.self; ints }
  in
  match w with
  | `Fill _ -> invalid_arg "Device_nv.part: the device runs no fill"
  | `Words ws ->
      if Array.length ws mod 2 <> 0 then
        invalid_argf "Device_nv.part: %d words, expected two per entry"
          (Array.length ws);
      let check x =
        if x < 0 || x > 0xffff_ffff then
          invalid_argf "Device_nv.part: word 0x%x exceeds 32 bits" x
      in
      Array.iter check ws;
      part [| 0; 0; 0; 0; 0 |] ws
  | `Copy ((dst, o), (src, o'), n) ->
      if q <> 1 then
        invalid_arg "Device_nv.part: copies run on COPY:0, not COMPUTE:0";
      if n < 0 then invalid_argf "Device_nv.part: a copy of %d bytes" n;
      let side what r o =
        match mine d r with
        | Some r when Atomic.get r.live ->
            if o < 0 || o + n > r.bytes then
              invalid_argf
                "Device_nv.part: the copy's %s range [%d, %d) lies outside its \
                 %d bytes"
                what o (o + n) r.bytes;
            r.mem.address
        | Some _ | None ->
            invalid_argf
              "Device_nv.part: the copy's %s is no live region of the device"
              what
      in
      let dst = side "destination" dst o and src = side "source" src o' in
      part [| dst; o; src; o'; n |] [||]

let check_parts name d ps =
  if Array.length ps > max_parts then
    invalid_argf "Device_nv.%s: %d parts, expected at most %d" name
      (Array.length ps) max_parts;
  let check i p =
    if p.owner <> d.self then
      invalid_argf "Device_nv.%s: part %d is another device's" name i;
    let after = fields + p.ints.(words_at) in
    for k = after to Array.length p.ints - 1 do
      if p.ints.(k) >= i then
        invalid_argf
          "Device_nv.%s: part %d runs after part %d, expected an earlier part"
          name i p.ints.(k)
    done
  in
  Array.iteri check ps

external room_parts : int -> part array -> int = "caml_device_nv_room"

external submit_parts : int -> int -> int array -> part array -> int
  = "caml_device_nv_submit"

(* nx_edge.h's codes. *)
let fits = 0
let later = 1

let room (T d) ps =
  check_parts "room" d ps;
  let r = room_parts d.self ps in
  if r = fits then `Fits else if r = later then `Later else `Never

let submit (T d) ~v ~waits ~handles:_ ps =
  let expected = last d.self + 1 in
  if v <> expected then
    invalid_argf "Device_nv.submit: value %d, expected %d" v expected;
  if Array.length waits > max_waits then
    invalid_argf "Device_nv.submit: %d waits, expected at most %d"
      (Array.length waits) max_waits;
  check_parts "submit" d ps;
  let w = Array.make (2 * Array.length waits) 0 in
  let wait i = function
    | `Word, at, value ->
        w.(2 * i) <- at;
        w.((2 * i) + 1) <- value
    | (`Equal | `Object), _, _ ->
        invalid_arg "Device_nv.submit: the device waits only with `Word"
  in
  Array.iteri wait waits;
  if room_parts d.self ps <> fits then
    invalid_arg "Device_nv.submit: the parts do not fit the rings now";
  ignore (submit_parts d.self v w ps : int);
  `Ok

let room_entry = Nativeint.of_int (room_entry_address ())
let submit_entry = Nativeint.of_int (submit_entry_address ())

(* Timeline *)

let word (T d) = R d.word
let signaled (T d) = read_word d.self

let fault_name table v =
  match List.assoc_opt v table with Some n -> n | None -> strf "0x%x" v

(* The errors the RM wrote into the channels' notifiers when it stopped them,
   such as for a fault of a channel's own methods. *)
let channel_errors d =
  let (module R : D.RELEASE) = d.release in
  let error q =
    let x = notification d.self q in
    if x = 0 then None
    else
      let code = x lsr 16 and status = x land 0xffff in
      Some
        (strf "channel error %d (%s), status 0x%x" code
           (fault_name R.robust_channel_errors code)
           status)
  in
  List.filter_map error [ 0; 1 ]

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
  match channel_errors d @ sm_errors_of d with
  | [] -> ()
  | report -> raise (Fault (String.concat "\n" report))

let sleep (T d) ~seen ~still_ms =
  if read_word d.self = seen then begin
    check_faults d;
    if watch d.self seen still_ms then check_faults d
  end

(* Loss *)

let stop (T d) =
  Mutex.protect d.local_lock (fun () -> Atomic.set d.stopped true);
  let ok = function Ok () -> true | Error _ -> false in
  let rm = d.path.rm in
  let unregistered =
    List.for_all Fun.id
      (List.map
         (fun ch -> try ok (d.path.unregister ch) with Fault _ -> false)
         d.channels)
  in
  let freed =
    ok (rm.free ~parent:d.path.device d.debugger)
    && ok (rm.free ~parent:d.path.device d.group)
  in
  if not (unregistered && freed) then `Unknown
  else begin
    raise_word d.self;
    end_channels d.self;
    (* The work no longer runs: what it used goes back, and a failure to give
       some back leaves the answer as it is. *)
    let give m = try d.path.free m with Fault _ -> () in
    List.iter give d.owned;
    Mutex.protect d.local_lock (fun () ->
        Option.iter give d.local_current;
        Option.iter (fun (m, _) -> give m) d.local_pending;
        List.iter (fun (m, _) -> give m) d.local_retired);
    `Stopped
  end
