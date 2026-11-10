(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

external row : int -> (string * int * int) option = "nx_array_support_row"
external decode : int -> int -> int = "nx_array_support_decode"
external of_int64 : int -> int64 -> int = "nx_array_support_of_i64"
external of_uint64 : int -> int64 -> int = "nx_array_support_of_u64"
external descriptor : ('v, 's) Nx_array.t -> int array
  = "nx_array_support_descriptor"

let answers =
  Nx_array.
    [
      ("Done", Done);
      ("Declined", Declined);
      ("Wrong_dtype", Wrong_dtype);
      ("Dead_buffer", Dead_buffer);
      ("Off_host", Off_host);
      ("Held_exclusive", Held_exclusive);
      ("Read_only", Read_only);
      ("Repeated_elements", Repeated_elements);
      ("Overlapping", Overlapping);
      ("Shape_mismatch", Shape_mismatch);
    ]

let pp_answer ppf a =
  Format.pp_print_string ppf (fst (List.find (fun (_, b) -> a = b) answers))

external codes : unit -> int array = "nx_array_support_codes"
external max_numel : unit -> int = "nx_array_support_max_numel"
external code : Nx_array.answer -> int = "%identity"
external add : 'z -> 'x -> 'y -> Nx_array.answer = "nx_array_support_add"

external read_all : Nx_array.any array -> Nx_array.answer
  = "nx_array_support_read_all"

external copy_into :
  ('v, 's) Nx_array.t -> ('v, 's) Nx_array.t -> Nx_array.answer
  = "nx_array_copy"

external of_array_into : ('v, 's) Nx_array.t -> 'v array -> Nx_array.answer
  = "nx_array_of_array"

external int16_at :
  int ->
  int ->
  (int, Bigarray.int16_signed_elt, Bigarray.c_layout) Bigarray.Genarray.t
  = "nx_array_support_int16_at"

external collect : ('v, 's) Nx_array.t -> Nx_array.answer
  = "nx_array_support_collect"

(* An io device over bigarrays *)

type bytes =
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

external blit_in : bytes -> int -> int -> int -> unit
  = "nx_array_support_blit_in"
[@@noalloc]

external blit_out : bytes -> int -> int -> int -> unit
  = "nx_array_support_blit_out"
[@@noalloc]

module Io = struct
  type region = bytes

  let free (_ : region) = ()
  let read r ~at ~dst ~len = blit_out r at dst len
  let write r ~at ~src ~len = blit_in r at src len
  let pages (_ : region) = None
  let prefetch (_ : region) ~at:(_ : int) ~len:(_ : int) = ()
end

(* Regions whose pages the host maps: the bigarray they are. *)
module Mapped = struct
  include Io

  let pages (r : region) = Some r
end

let io =
  lazy (match Rig.open_io "nx-io" with Ok d -> d | Error e -> failwith e)

let io_device () = Lazy.force io

let region n =
  let r = Bigarray.Array1.create Bigarray.char Bigarray.c_layout n in
  Bigarray.Array1.fill r '\000';
  r

let read_only n =
  Rig.Buffer.of_io (io_device ()) (module Io) (region n) ~access:Read n

let on_io (type v s) (a : (v, s) Nx_array.t) =
  let src = Nx_array.buffer a in
  let n = Rig.Buffer.length src in
  let b =
    Rig.Buffer.of_io (io_device ()) (module Io) (region n) ~access:Read_write n
  in
  Rig.Buffer.copy ~src ~dst:b;
  Nx_array.v (Nx_array.dtype a) (Nx_array.layout a) b

let mapped_region n =
  let r = region n in
  let b =
    Rig.Buffer.of_io (io_device ()) (module Mapped) r ~access:Read_write n
  in
  (b, r)

(* A device whose memory the host does not address *)

let off_host_polled =
  lazy
    (Rig_support.Polled.open_ ~host_visible:false ~runs:`Itself "nx-off-host")

let off_host () = fst (Lazy.force off_host_polled)

let off_host_allocations () =
  List.length (Rig_support.Polled.allocs (snd (Lazy.force off_host_polled)))

(* Late, a device over host memory whose work runs only when a wait sleeps on
   it: its submit queues the work, its sleep and its stop run it. *)

external late_new : nativeint -> nativeint = "nx_array_support_late_new"
external late_publish : nativeint -> unit = "nx_array_support_late_publish"
external late_signaled : nativeint -> int = "nx_array_support_late_signaled"
external late_word : nativeint -> int = "nx_array_support_late_word"
external malloc : int -> int = "nx_array_support_malloc"
external free : int -> unit = "nx_array_support_free"
external store_fill : unit -> nativeint = "nx_array_support_store"

module Driver = struct
  type t = {
    self : nativeint;
    fault : string option Atomic.t;
    allocated : (int * Rig_edge.memory option) Atomic.t;
        (* The regions [alloc] made, and the kind of the latest. *)
  }

  (* [raw] is what [malloc] gave, [0] for the word. *)
  type region = { at : int; raw : int }
  type image = unit

  exception Fault of string

  let key : t Type.Id.t = Type.Id.make ()
  let capability_key : unit Type.Id.t = Type.Id.make ()

  let facts d =
    {
      Rig_edge.arch = "late";
      budget = max_int;
      queues = [ { name = "COMPUTE:0"; runs = [ Fill ] } ];
      copy_engine = None;
      completion = Host;
      waits = { stores = false; hosts = false };
      may_block = false;
      hang_ms = None;
      maps_host = true;
      host_addresses = true;
      mapped = false;
      (* [alloc] rounds each region's start up to 64 bytes. *)
      alignment = 64;
      cache_bytes = None;
      counters = [];
      capability = Capability (capability_key, ());
      word = { at = late_word d.self; raw = 0 };
      edge = d.self;
      bus = None;
    }

  let alloc d m n =
    let raw = malloc n in
    if raw = 0 then None
    else begin
      let rec count () =
        let (k, _) as old = Atomic.get d.allocated in
        if not (Atomic.compare_and_set d.allocated old (k + 1, Some m)) then
          count ()
      in
      count ();
      Some { at = (raw + 63) land lnot 63; raw }
    end

  let free _ r = if r.raw <> 0 then free r.raw

  let locate r =
    {
      Rig_edge.address = Some r.at;
      host = Some r.at;
      handle = Nativeint.of_int r.at;
    }

  let peer _ _ = false
  let map_peer _ _ _ = Rig_edge.Never
  (* Host pages lie at their own address; [raw = 0] frees nothing. *)
  let map_host _ p _ = Rig_edge.Maps { at = p; raw = 0 }

  let dma _ _ ~fn:_ = Error "late gives no driver its memory"
  let flush _ = ()
  let image _ _ = Error "late loads no code"
  let entry () _ = Rig_edge.Absent
  let prepare () _ = ()
  let unload _ () = ()

  (* Its room never answers RIG_GROW. *)
  let grow _ = Rig_edge.Grown
  let signaled d = late_signaled d.self

  let sleep d ~seen:_ ~still_ms:_ =
    match Atomic.get d.fault with
    | Some why -> raise (Fault why)
    | None -> late_publish d.self

  let stop d ~fault:_ = late_publish d.self
end

module Late = struct
  type t = Driver.t

  let open_ name =
    let fault = Atomic.make None in
    let made = ref None in
    let make ~word =
      let allocated = Atomic.make (0, None) in
      let t = { Driver.self = late_new word; fault; allocated } in
      made := Some t;
      Ok t
    in
    match Rig.open_ { name; driver = (module Driver); make } with
    | Ok d -> (d, Option.get !made)
    | Error e -> failwith e

  let fault (t : t) why = Atomic.set t.fault (Some why)

  let allocations (t : t) =
    let n, m = Atomic.get t.allocated in
    (~count:n, ~last:m)
end

let store = store_fill ()

let write b s =
  let n = String.length s in
  let arg = Bytes.create (16 + n) in
  Bytes.set_int64_le arg 0 0L;
  Bytes.set_int64_le arg 8 (Int64.of_int n);
  Bytes.blit_string s 0 arg 16 n;
  let work =
    Rig.Submission.fill
      ~refs:[| Rig.Submission.write 0 ~at:0 |]
      store
      ~params:(4 * ((16 + n + 3) / 4))
  in
  let s = Rig.Submission.make (Rig.Buffer.device b) [| work |] in
  let run = Rig.Submission.take s in
  let block = Rig.Submission.block s 0 in
  for i = 0 to ((Bytes.length arg + 3) / 4) - 1 do
    let at = 4 * i in
    let word = ref 0 in
    for j = 0 to min 3 (Bytes.length arg - at - 1) do
      word := !word lor (Char.code (Bytes.get arg (at + j)) lsl (8 * j))
    done;
    Rig.Submission.Run.int32 run block at !word
  done;
  ignore (Rig.submit run ~buffers:[| b |])
