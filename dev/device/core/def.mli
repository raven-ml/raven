(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The records the core's modules share. *)

module Scalar = Device_dtype.Scalar

type ('a, 'r, 'i) dm =
  (module Sigs.Driver with type t = 'a and type region = 'r and type image = 'i)

type ('a, 'r) im = (module Sigs.Io with type t = 'a and type region = 'r)

(* A region of a driver, with the module and the device that hold it. *)
type region = Region : { m : ('a, 'r, 'i) dm; h : 'a; r : 'r } -> region
type io_region = Io_region : { m : ('a, 'r) im; h : 'a; r : 'r } -> io_region
type capability = Capability : 'c Type.Id.t * 'c -> capability

(* A custom block whose collection puts a release record on a device's release
   list. *)
type token

type kind =
  | Host
  | Driver : { m : ('a, 'r, 'i) dm; h : 'a } -> kind
  | Io : { m : ('a, 'r) im; h : 'a } -> kind

(* How a device's word advances. *)
type completion = Store | Object of int | Host_writes

type device = {
  index : int;
  name : string;
  machine : string option;
  kind : kind;
  c : int;  (** The C record, 0 for a host and an io device. *)
  arch : string;
  queues : string array;
  completion : completion;
  waits_store : bool;
  waits_object : bool;
  waits_host : bool;
  word : int;  (** The word's host address, 0 behind a transport. *)
  word_region : region option;
  key : int;  (** The driver's key's uid, [-1] for a host or io device. *)
  memory_device : bool;
  fault : exn -> string option;  (** The driver's faults. *)
  capability : capability option;
  release : int;  (** The C release list. *)
  lock : Mutex.t;  (** Guards everything mutable below. *)
  mutable budget : int;
  mutable used : int;  (** Own bytes in live buffers, code and cache. *)
  mutable cached : int;
  cache : (int, entry list) Hashtbl.t;  (** By [bytes * 4 + kind]. *)
  mutable retiring : entry list;  (** Waiting for other devices' uses. *)
  mutable pending : (int * pending) list;  (** Waiting for its own value. *)
  mutable pairs : int array;  (** By producer index: 0 unknown, 1 host. *)
  mutable pair_maps : region list;
  mutable afters : (int * (unit -> unit)) list;
}

(* Memory kinds: [Buffer.memory]'s, and the host's heap. *)
and entry = {
  owner : device;
  memory : int;
      (** 0 device, 1 pinned, 2 mapped, 3 heap, 4 kept by its maker. *)
  bytes : int;
  region : region option;
  io_region : io_region option;
  mutable stamps : int;  (** The C stamps, a hold's once held; 0 none. *)
  mutable own : int;  (** The memory's own stamps. *)
  mutable maps : mapping list;  (** Other devices' mappings of it. *)
  mutable held : bool;
}

and mapping = { on : device; map : region; at : int; by : nativeint }

(* A release that waits for a value of its device, which covers the work that
   may use it without naming it. *)
and pending = Free of entry | Unmap of region | Unload of image * int
and image = Image : { m : ('a, 'r, 'i) dm; h : 'a; i : 'i } -> image

type claim = {
  mutable count : int; [@atomic]  (** Readers, or [-1] when exclusive. *)
  mutable generation : int; [@atomic]
  mutable why : string;  (** The last consumption's reason. *)
}

type keep =
  | Nothing
  | Heap of
      (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
      * token
  | Bigarray : ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> keep

(* A memory: owned, or a borrow of [root]'s on [dev]. *)
type memory = {
  dev : device;
  bytes : int;
  host : int;  (** Host address, [-1] if the host does not address it. *)
  address : int;  (** As [dev]'s work addresses it, [-1] if none. *)
  handle : nativeint;
  claim : claim;
  mutable entry : entry;
  mutable token : token;
  keep : keep;
  root : memory;
}

type buffer = {
  mem : memory;
  offset : int;
  dtype : Scalar.t;
  length : int;
  generation : int;
}

type program = {
  pdev : device;
  image : image;
  code_bytes : int;
  ptoken : token;
}

type hold = { hstamps : int; members : buffer list; htoken : token }

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

(* What a release list holds. *)
type released =
  | Memory of entry
  | Program of image * int  (** The image and its code's bytes. *)
  | Release of { stamps : int; release : unit -> unit }
