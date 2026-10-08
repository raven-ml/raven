(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The records the core's modules share.

    [rig_memory.c] reads by position, for the readers and claims [rig.h]
    declares, the fields of {!buffer} and {!claim}, those of {!memory} up to
    [root], and those of {!entry} up to [held]: a change of their order changes
    it too. *)

type ('a, 'r, 'i) dm =
  (module Sigs.Driver with type t = 'a and type region = 'r and type image = 'i)

type ('a, 'r) im = (module Sigs.Io with type t = 'a and type region = 'r)

(** The type for a region of a driver, with the module and the device that hold
    it, and the witness of its type, which equates the regions of one device. *)
type region =
  | Region : {
      m : ('a, 'r, 'i) dm;
      h : 'a;
      r : 'r;
      rid : 'r Type.Id.t;
    }
      -> region

type io_region = Io_region : { m : ('a, 'r) im; h : 'a; r : 'r } -> io_region
type capability = Capability : 'c Type.Id.t * 'c -> capability

type token
(** The type for custom blocks whose collection puts a release record on a
    release list ({!Memory.token}). *)

type kind =
  | Host
  | Driver : { m : ('a, 'r, 'i) dm; h : 'a; rid : 'r Type.Id.t } -> kind
  | Io : { m : ('a, 'r) im; h : 'a } -> kind

(** The type for how a device's word advances ({!Rig.Driver.completion}). *)
type completion = Store | Object of int | Host_writes

(** The type for a memory's kind: {!Rig.Buffer.memory}'s three, host memory its
    keeper frees (a heap bigarray or the caller's), io memory its device made,
    and io memory its library gave. *)
type memory_kind = Device | Pinned | Mapped | Host_kept | Io_made | Io_given

(** {!Rig.Buffer.access}. *)
type access = Read | Read_write

(** The type for the host memory a memory record keeps reachable: none, a heap
    bigarray with the token that returns its bytes to the host's budget, or the
    caller's bigarray. *)
type keep =
  | Nothing
  | Heap of
      (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
      * token
  | Bigarray : ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> keep

type device = {
  index : int;
  name : string;
  machine : string option;
  kind : kind;
  c : int;  (** The C record, 0 for the host. *)
  arch : string;
  queues : string array;
  copy_queue : string option;  (** Its first copy queue, which runs copies. *)
  completion : completion;
  waits_store : bool;
  waits_object : bool;
  waits_host : bool;
  max_waits : int;  (** The waits in its queue one submission carries. *)
  maps_host : bool;  (** Whether it maps host memory. *)
  word : int;  (** The word's host address, 0 behind a transport. *)
  word_region : region option;
  key : int;
      (** The uid of its driver's key or its io library's region key, [-1] for
          the host. *)
  memory_device : bool;
  fault : exn -> string option;  (** The driver's faults. *)
  capability : capability option;
  release : int;  (** The C release list. *)
  lock : Lock.t;
      (** Guards everything mutable below. The atomic fields are written under
          it and read without it too, such as by a drain that finds nothing to
          do. *)
  mutable budget : int; [@atomic]
  mutable used : int; [@atomic]
      (** Own bytes in live buffers, code and cache. *)
  mutable cached : int; [@atomic]
  cache : entry list Hashtbl.Make(Int).t;  (** By [bytes * 8 + kind]. *)
  mutable retiring : entry list; [@atomic]
      (** Waiting for other devices' uses. *)
  mutable pending : (int * pending) list; [@atomic]
      (** Waiting for its own value. *)
  mutable pairs : int array;
      (** By producer index, how its queue waits for the producer's values: [0]
          undecided, [-1] on the host, otherwise the producer's object or its
          word's address as this device maps it. Replaced whole, and read
          without the lock. *)
  mutable pair_maps : region list;
      (** The mappings of producers' words, kept for good. *)
  mutable afters : (int * (unit -> unit)) list;
}

and entry = {
  owner : device;
  memory : memory_kind;
  bytes : int;
  region : region option;
  io_region : io_region option;
  access : access;
  mutable stamps : int;  (** The C stamps, a hold's once held; 0 none. *)
  mutable held : bool;  (** In a hold. Guarded by [owner]'s lock. *)
  mutable own : int;  (** The memory's own stamps. *)
  mutable maps : mapping list;  (** Other devices' mappings of it. *)
  mutable unmaps : int; [@atomic]
      (** The unmaps left before a dead memory is given back. *)
  mutable pages : pages;
      (** An io memory's pages, asked at its first borrow. *)
  mutable proxy : int;
      (** The C proxy of the bigarrays over the memory, 0 before the first. *)
  mutable kept : keep;
      (** Host memory a device borrowed: its bytes, held until its uses are
          reached. *)
}

and pages =
  | Unasked
  | Pages of
      (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
  | No_pages

and mapping = { on : device; map : region; at : int; by : nativeint }

(** The type for releases that wait for a value of their device, which covers
    the work that may use them without naming them. *)
and pending =
  | Free of entry
  | Unmap of region * entry option
      (** A mapping, and the dead memory it maps, given back after its last
          unmap. *)
  | Unload of image * entry option

and image = Image : { m : ('a, 'r, 'i) dm; h : 'a; i : 'i } -> image

type claim = {
  mutable count : int; [@atomic]  (** The claim word, laid out in {!Memory}. *)
  mutable generation : int; [@atomic]
      (** Raised by each consumption: a buffer of an older one is dead. *)
  mutable why : string;  (** The last consumption's reason. *)
}

type memory = {
  dev : device;
  bytes : int;
  host : int;  (** Host address, [-1] if the host does not address it. *)
  address : int;  (** As [dev]'s work addresses it, [-1] if none. *)
  handle : nativeint;
  claim : claim;
  mutable entry : entry;
  mutable root : memory;
      (** The memory a borrow maps; the record itself otherwise, set once made.
      *)
  mutable token : token;
  keep : keep;
}
(** The type for memory: owned, or a borrow of [root]'s on [dev]. *)

type buffer = { mem : memory; offset : int; length : int; generation : int }
(** {!Rig.Buffer.t}: live while [generation] is its claim's. *)

type program = { pdev : device; image : image; ptoken : token }
type hold = { hstamps : int; members : buffer list; htoken : token }

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
  | Load of { program : program; binary : string; time : int }
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

(** The type for what a release list holds. *)
type released =
  | Memory of entry
  | Program of image * entry option  (** The image and its code's memory. *)
  | Release of { stamps : int; release : unit -> unit }  (** A hold's. *)
