(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The records rig's modules share: devices, allocations, buffers, claims,
    images, holds and profile events.

    The types live here, apart from the modules that work on them, because
    they refer to each other, as a device holds its cache of entries and an
    entry its owner device. This module says what each field is. The module
    that writes a field states its invariants: {!Dev} for a device's state,
    {!Memory} for entries, caches and claim words, {!Claim} for claims.

    The C stubs read some fields by position, so their order is fixed.
    [rig_memory.c] reads, for the readers and claims [rig.h] declares, the
    fields of {!buffer} and {!claim}, those of {!memory} up to [root], those
    of {!entry} up to [pages], and those of {!device} up to [machine], and takes
    [Here] to be {!machine}'s first constant constructor. [rig.h]'s
    [enum rig_access] lists {!access}'s constructors in order. A change of
    either changes the C too. *)

(** The type for a driver's module with its types exposed. *)
type ('a, 'r, 'i) dm =
  (module Rig_edge.Driver
     with type t = 'a
      and type region = 'r
      and type image = 'i)

(** The type for an io library's module with its types exposed. *)
type ('a, 'r) im = (module Rig_edge.Io with type t = 'a and type region = 'r)

(** The type for a region of a driver, with the module and the device that
    hold it, and the witness of its type, which equates the regions of one
    device. *)
type region =
  | Region : {
      m : ('a, 'r, 'i) dm;
      h : 'a;
      r : 'r;
      rid : 'r Type.Id.t;
    }
      -> region

(** The type for a region of an io library, with the module and the device
    that hold it. *)
type io_region = Io_region : { m : ('a, 'r) im; h : 'a; r : 'r } -> io_region

type token
(** The type for custom blocks whose collection puts a release record on a
    release list ({!Memory.token}). *)

(** The type for what a device is: the host, a driver's device with its
    module, handle and region witness, or an io device with its module and
    handle. *)
type kind =
  | Host
  | Driver : { m : ('a, 'r, 'i) dm; h : 'a; rid : 'r Type.Id.t } -> kind
  | Io : { m : ('a, 'r) im; h : 'a } -> kind

(** The type for how a device's word advances ({!Rig_edge.completion}), an
    object as an int. *)
type completion = Store | Object of int | Host_writes

(** {!Rig.Buffer.access}. *)
type access = Read | Read_write

(** The type for the host memory a memory record keeps reachable: none, a
    heap bigarray, whose collection returns its bytes to the host's budget,
    or the caller's bigarray. *)
type keep =
  | Nothing
  | Heap of
      (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
  | Bigarray : ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> keep

(** The type for who made an io allocation: its device, or its library. *)
type origin = Made | Given

(** The type for what an allocation's bytes are: a driver's region of
    memory of some {!Rig.Buffer.kind}, an io region, or host memory that its
    keeper frees, held until the uses a borrow made are reached ([Kept
    Nothing] for none). *)
type backing =
  | Driver_memory of { memory : Rig_edge.memory; region : region }
  | Io_memory of { origin : origin; region : io_region }
  | Kept of keep

(** The type for the end of a Stopped device's timeline word: read still,
    its readers moved to the C record's copy at a count of minor
    collections, or given back to the driver ({!Dev}). *)
type word_end = Read | Moved of int | Given

(** The type for the word as a wait under a hang bound last saw it: its
    value, whether the device was idle then, and since when, in milliseconds
    of the monotonic clock. *)
type progress = { seen : int; idle : bool; since : int }

type c
(** The type for custom blocks that hold a device's C record
    ([struct rig_device] in [rig_stubs.h]), which is freed once neither its
    block nor a prepared submission holds it. *)

(** The type for devices: {!Rig.t}. A device is its facts, fixed at open
    and kept for the life of the process, and its {!live} resources, which
    it drops at its end ({!Dev.section-loss}). *)
type device = {
  index : int;  (** Its index in the table of devices, [0] for the host. *)
  name : string;
      (** The opener's name, followed by ["@machine"] on another machine. *)
  machine : machine;  (** The machine it belongs to. *)
  kind : kind;  (** What it is, with the module and handle that run it. *)
  arch : string;  (** Its architecture, as its driver names it. *)
  maps_host : bool;  (** Whether it maps host memory. *)
  host_addresses : bool;  (** Whether the host addresses all its regions. *)
  key : int;
      (** The uid of its driver's key or its io library's region key, [-1]
          for the host. *)
  memory_device : bool;  (** Whether it is a memory device. *)
  capability : Rig_edge.capability option;
      (** What compiled code needs of it. *)
  mutable budget : int; [@atomic]  (** {!Rig.budget}. *)
  mutable live : live option;
      (** Its resources, [None] once it ended. Read without a lock. *)
}

(** The type for a device's resources: its C record, its lock, its memory
    and the facts only its work needs. *)
and live = {
  c : c;  (** Its C record. *)
  queues : Rig_edge.queue array;  (** Its queues, as its driver lists them. *)
  copy_engine : int option;
      (** Its copy engine's index in [queues], if it has one. *)
  completion : completion;  (** How its word advances. *)
  waits : Rig_edge.waits;
      (** Whose words its queues wait on, and the most per submit. *)
  hang_ms : int option;  (** Its hang bound. *)
  mutable progress : progress; [@atomic]
      (** The word as the last wait under [hang_ms] saw it, written by that
          wait without the lock. *)
  word : int;  (** The word's host address, [0] if it has none. *)
  word_region : region option;  (** The region its word lies in. *)
  fault : exn -> string option;
      (** The reason of a driver fault, [None] for another exception. *)
  release : int;  (** Its C release list, in its C record. *)
  lock : Lock.t;
      (** Guards the mutable fields after it. The atomic ones among them are
          written under it and also read without it, as by a drain that
          finds nothing to do. *)
  mutable drains : int;
      (** The drains in flight on it, [-1] once its resources are freed. *)
  mutable entries : int; [@atomic]
      (** The entries of its memory not given back; the host counts none. *)
  mutable used : int; [@atomic]
      (** Its own bytes in live buffers, code and the cache. *)
  mutable cached : int; [@atomic]  (** The bytes in its cache. *)
  cache : entry list Hashtbl.Make(Int).t;
      (** Its cache, by [bytes * 4 + kind]. *)
  mutable retiring : entry list; [@atomic]
      (** Collected entries that wait for their points or their bigarrays. *)
  mutable pending : (int * pending) list * (int * pending) list; [@atomic]
      (** Releases by nondecreasing value of it, replaced whole: the front in
          order, the back reversed, and the front empty only when both are. *)
  mutable pairs : int array;
      (** By producer index, how its queues wait for the producer's values:
          [min_int] undecided, [-1] on the host, otherwise the producer's
          object or its word's address as this device maps it. Grown by
          replacement, written under the lock, and read without it. *)
  mutable pair_maps : (int * region) list;
      (** Its mappings of producers' words, by producer index, kept until
          the producer's word ends. *)
  mapped : entry Hashtbl.Make(Int).t;
      (** The entries it maps, by [id], whose mappings end with its word
          unless the entry is freed first. *)
  mutable word_end : word_end;  (** How far the end of its word went. *)
  mutable afters : (int * (unit -> unit)) list;
      (** Functions a wait runs once it finds their value reached. *)
}

(** The type for a device's machine: this process's, another one of which
    the device is the host, by the name {!Rig.open_host} gave it, or the
    other machine whose host is the device given. *)
and machine = Here | Hosts of string | On of device

(** The type for an allocation's entry: what its owner gives back once its
    memory record is collected and its points are reached. *)
and entry = {
  owner : device;  (** The device whose memory it is. *)
  backing : backing;  (** What its bytes are. *)
  bytes : int;  (** Its size. *)
  access : access;  (** What its memory admits. *)
  stamps : int;  (** Its C stamps ({!Memory.section-stamps}). *)
  mutable maps : mapping list;
      (** Other devices' mappings of it. Guarded by [owner]'s lock. *)
  mutable unmaps : int; [@atomic]
      (** The unmaps left before a freed entry goes back to its driver. *)
  mutable pages : pages;
      (** An io allocation's pages, asked at its first borrow. *)
  mutable proxy : int;
      (** The C proxy of the bigarrays over it, [0] before the first. *)
  id : int;  (** Unique for the life of the process. *)
}

(** The type for an io allocation's pages: not asked yet, a bigarray over
    them, or none. *)
and pages =
  | Unasked
  | Pages of
      (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
  | No_pages

(** The type for a device's mapping of an allocation: the device, its
    region, and the address and handle its work uses. *)
and mapping = { on : device; map : region; at : int; by : nativeint }

(** The type for releases that wait for a value of their device, which
    covers the work that may use them without naming them. *)
and pending =
  | Free of entry
  | Unmap of region * entry option
      (** A mapping, and the freed entry it maps, given back after its last
          unmap. *)
  | Unload of loaded * entry option

(** The type for an image: one a driver loaded, with its module and handle,
    or code the host linked ({!Host_code}). *)
and loaded =
  | Loaded : { m : ('a, 'r, 'i) dm; h : 'a; i : 'i } -> loaded
  | Linked of linked

(** The type for code the host linked: the executable memory that holds it,
    which the collector unmaps, and the address of each function in it. *)
and linked = { code : code; functions : (string * int) list }

(** The type for executable memory, a custom block that unmaps it once
    collected. *)
and code

(** The type for an allocation's claim, which every memory record over it
    shares ({!Claim}). *)
type claim = {
  mutable count : int; [@atomic]
      (** The claim word ({!Memory.section-claims}). *)
  mutable generation : int; [@atomic]
      (** Raised by each consumption, which kills the buffers of older
          ones. *)
  mutable why : string;  (** The last consumption's reason. *)
}

type memory = {
  dev : device;  (** The device that reaches the allocation through it. *)
  bytes : int;  (** The allocation's size. *)
  host : int;  (** Host address, [-1] if the host does not address it. *)
  address : int;  (** As [dev]'s work addresses it, [-1] if none. *)
  handle : nativeint;  (** The driver object that names it on [dev]. *)
  claim : claim;  (** The allocation's claim. *)
  mutable entry : entry;
      (** The allocation's entry, {!Memory.no_entry} for host memory no
          device borrowed. *)
  mutable root : memory;
      (** The owner's record for a borrow; the record itself otherwise, set
          once made. *)
  mutable token : token;
      (** Puts [entry] on [dev]'s release list once collected; none for a
          borrow. *)
  keep : keep;  (** The host memory it keeps reachable. *)
}
(** The type for a device's record of an allocation: the owner's, which is
    its own root, or a borrow's. *)

type buffer = { mem : memory; offset : int; length : int; generation : int }
(** {!Rig.Buffer.t}: live while [generation] is its claim's. *)

type image = { idev : device; loaded : loaded; itoken : token }
(** {!Rig.Image.t}: its device, what the driver loaded, and the token that
    unloads it once collected. *)

type hold = { hstamps : int; htoken : token }
(** The hold of the value a submission keeps ({!Rig.Submission.make}): the
    stamps of the submission's submits, and the token that releases the
    value once collected. *)

(** {!Rig.Profile.event}. *)
type event =
  | Span of {
      device : device;
      lane : string;
      name : string;
      start : int;
      stop : int;
    }
  | Allocation of { device : device; time : int; allocated : int }
  | Load of { image : image; binary : string; time : int }
  | Counters of {
      device : device;
      name : string;
      start : int;
      stop : int;
      counters : (string * int array) list;
    }
  | Trace of {
      device : device;
      name : string;
      start : int;
      stop : int;
      part : int;
      data : string;
    }
  | Overwritten of { device : device; time : int; runs : int }
  | Copy of { src : device; dst : device; bytes : int; start : int; stop : int }

(** The type for a hold's release: its stamps, the value it keeps, and the
    generation of forks it was made in. *)
type hold_release =
  | Hold_release : { stamps : int; value : 'a; generation : int }
      -> hold_release

(** The type for what a release list holds. *)
type released =
  | Memory of entry
  | Image of loaded * entry option  (** The image and its code's memory. *)
  | Release of hold_release
