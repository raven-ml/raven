(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Nx_array

(* The value

   GADT constructors (the operations of [Op.t]) require that type variables in
   the payload be deducible from the return type. A transparent alias of
   [Nx_array.t] would not be injective, so the value is a GADT of its own, whose
   parameters the return type determines.

   A value is one of three things. [Host] is an nx.cpu array, the value of the
   host placement, with no cell, so that the default path pays nothing for what
   compiled calls need of a storage. [Placed] is a value at any other placement,
   held in runtime buffers, one per device of its placement: nx knows its
   placement, dtype and view, and the buffers are its storage. [Traced] is a
   node of a trace: it has no bytes and never will, and the tracer that made it
   keeps its payload in [t_node].

   The views of one placed storage share one cell, which holds what belongs to
   the storage rather than to a view: whether it is live or was consumed by a
   compiled call, and how many reachable programs bind it. A storage holds
   bytes, and each view reads them as its own dtype, so a bitcast of a placed
   value is a view of its storage too. *)

type ('a, 'b) t =
  | Host : ('a, 'b) Nx_array.t -> ('a, 'b) t
  | Placed : ('a, 'b) resident -> ('a, 'b) t
  | Traced : ('a, 'b) traced -> ('a, 'b) t

and ('a, 'b) resident = {
  r_id : int; (* fresh per value; identity tables key by it *)
  r_placement : Placement.t; (* never the host placement *)
  r_dtype : ('a, 'b) Nx_dtype.t;
  r_view : View.t; (* per shard, the same on every shard *)
  r_cell : cell; (* one per storage, shared by all its views *)
}

and cell = {
  placement : Placement.t; (* where the storage lives, whichever views it *)
  bytes : int; (* bytes of the storage, per shard *)
  mutable state : state;
  bound : int Atomic.t; (* reachable program bindings to the storage *)
  lock : Mutex.t;
  mutable readers : int;
  mutable exclusive : bool;
}

and state =
  | Live of Nx_device.Buffer.t list
    (* one buffer per device of the cell's placement, in its order *)
  | Consumed of consumption

(* Where a compiled call consumed a storage: the consumed leaf's path in the
   call's arguments, whose first segment is the argument's position from 0. *)
and consumption = { path : string }

and ('a, 'b) traced = {
  t_id : int; (* fresh; identity tables key by it *)
  t_placement : Placement.t; (* where the value lives *)
  t_context : Placement.t; (* where the trace creates its values *)
  t_dtype : ('a, 'b) Nx_dtype.t;
  t_view : View.t; (* the layout of the value it stands for *)
  t_node : ('a, 'b) node; (* the tracer's payload *)
}

and ('a, 'b) node = ..

(* Where a creation makes its value. *)
type context = Placement.t

(* A host array of [dtype] and [shape], C-contiguous from its first element. *)
let alloc (type a b) (dtype : (a, b) Nx_dtype.t) shape : (a, b) Nx_array.t =
  let n = Array.fold_left ( * ) 1 shape in
  { dtype; view = View.create shape; buffer = Elements.create dtype n }

(* A host array of [dtype] and [shape] whose elements are [v]. *)
let filled dtype shape v =
  let a = alloc dtype shape in
  Elements.fill dtype a.buffer v;
  a

let id_counter = Atomic.make 0
let fresh_id () = Atomic.fetch_and_add id_counter 1 + 1

let outside_trace () =
  invalid_arg
    "a traced tensor has no bytes; it was used outside the trace that made it"

(* Why a value consumed at [path] is dead: the message of every later use. *)
let why_consumed { path } =
  Printf.sprintf
    "this value was consumed at %s in a compiled call's arguments; use the \
     value the call returned"
    path

let consumed k = invalid_arg (why_consumed k)

(* The lock only protects the cell's bookkeeping. Readers and consumers keep
   their claim while executing outside it; overlapping consumption never
   waits. *)
module Cell = struct
  let busy () =
    invalid_arg "Nx: storage is in use by another reader or consuming call"

  let state c =
    Mutex.lock c.lock;
    let state = c.state in
    Mutex.unlock c.lock;
    state

  let borrow c =
    Mutex.lock c.lock;
    let blocked = c.exclusive in
    if not blocked then c.readers <- c.readers + 1;
    Mutex.unlock c.lock;
    if blocked then busy ()

  let release c =
    Mutex.lock c.lock;
    let valid = (not c.exclusive) && c.readers > 0 in
    if valid then c.readers <- c.readers - 1;
    Mutex.unlock c.lock;
    if not valid then invalid_arg "Nx: unbalanced storage borrow"

  (* A borrow reads the storage, so it also claims the memory of its buffers for
     reading: the claim raises before [f] if a buffer is dead, or if a lost
     device can reach its memory ([Nx_device.Lost]), even for no elements. *)
  let with_borrow c f =
    borrow c;
    Fun.protect
      ~finally:(fun () -> release c)
      (fun () ->
        match c.state with
        | Live bufs ->
            Nx_device.Buffer.Claim.with_ ~read:bufs ~donate:[] (fun _ -> f ())
        | Consumed k -> consumed k)

  let upgrade c =
    Mutex.lock c.lock;
    let available = (not c.exclusive) && c.readers = 1 in
    if available then begin
      c.readers <- 0;
      c.exclusive <- true
    end;
    Mutex.unlock c.lock;
    if not available then busy ()

  let consume c why =
    let state = Consumed why in
    Mutex.lock c.lock;
    let valid = c.exclusive in
    if valid then c.state <- state;
    Mutex.unlock c.lock;
    if not valid then invalid_arg "Nx: consumption requires exclusive storage"

  let finish c =
    Mutex.lock c.lock;
    let retire =
      match c.state with
      | Consumed _ -> Atomic.get c.bound = 0
      | Live _ -> false
    in
    c.exclusive <- false;
    c.readers <- 1;
    Mutex.unlock c.lock;
    retire

  let pin c =
    Mutex.lock c.lock;
    ignore (Atomic.fetch_and_add c.bound 1);
    Mutex.unlock c.lock

  let unpin c =
    Mutex.lock c.lock;
    let previous = Atomic.fetch_and_add c.bound (-1) in
    let retire =
      previous = 1 && (not c.exclusive)
      && match c.state with Consumed _ -> true | Live _ -> false
    in
    Mutex.unlock c.lock;
    retire
end

(* [global p shape] is the shape of a value whose tiles at [p] have [shape]. *)
let global p shape =
  match Placement.cuts p with
  | [] -> shape
  | cuts ->
      let shape = Array.copy shape in
      List.iter (fun (a, n) -> shape.(a) <- shape.(a) * n) cuts;
      shape

(* A split value's view is each shard's, and its shape the whole's. *)
let whole_view r =
  match Placement.cuts r.r_placement with
  | [] -> r.r_view
  | _ ->
      let v = r.r_view in
      View.create ~offset:(View.offset v) ~strides:(View.strides v)
        (global r.r_placement (View.shape v))

(* Placed constructors *)

(* A cell over [storage], one runtime buffer of one size per device of
   [placement], which the runtime releases with the buffers. *)
let cell ~placement storage =
  {
    placement;
    bytes = Nx_device.Buffer.nbytes (List.hd storage);
    state = Live storage;
    bound = Atomic.make 0;
    lock = Mutex.create ();
    readers = 0;
    exclusive = false;
  }

(* [placed what p dtype view cell] is the value at [p] of [cell] under [view].
   Raises [Invalid_argument] naming [what] if [p] is the host's, or if [cell]'s
   devices do not hold [p]'s memories. *)
let placed what placement dtype view cell =
  if Placement.is_host placement then
    invalid_arg (what ^ ": a placed value is never on the host");
  let held = List.map Device.memory (Placement.devices cell.placement) in
  if
    not
      (List.for_all
         (fun d -> List.memq (Device.memory d) held)
         (Placement.devices placement))
  then
    invalid_arg
      (Format.asprintf "%s: a value on %a views a storage on %a" what
         Placement.pp placement Placement.pp cell.placement);
  Placed
    {
      r_id = fresh_id ();
      r_placement = placement;
      r_dtype = dtype;
      r_view = view;
      r_cell = cell;
    }

(* [read_as dtype b] is [b]'s bytes read as [dtype]'s elements: [b] itself when
   it holds them, a view of its memory otherwise. Raises [Invalid_argument] if
   [b]'s first byte is not aligned to one of them. *)
let read_as (type a b) (dtype : (a, b) Nx_dtype.t) b =
  let s = Nx_dtype.Scalar.of_dtype dtype in
  if Nx_dtype.Scalar.equal (Nx_device.Buffer.dtype b) s then b
  else
    Nx_device.Buffer.view b ~offset:0 s
      (Nx_device.Buffer.nbytes b * 8 / Nx_dtype.Scalar.bitsize s)

(* [aligned dtype b] is [true] iff [b]'s first byte is aligned to one of
   [dtype]'s elements, as reading [b] as them needs ({!read_as}). A file's bytes
   are read at any offset. *)
let aligned (type a b) (dtype : (a, b) Nx_dtype.t) b =
  let size = Int.max 1 (Nx_dtype.Scalar.(bitsize (of_dtype dtype)) / 8) in
  Nx_device.equal (Nx_device.Buffer.device b) Nx_device.disk
  || Nativeint.rem (Nx_device.Buffer.address b) (Nativeint.of_int size) = 0n

(* [capacity dtype c] is the elements of [dtype] that each shard of [c] holds:
   its buffers' length when they are of [dtype]'s format, which stops a packed
   format at its last element, and its bytes' worth otherwise. *)
let capacity (type a b) (dtype : (a, b) Nx_dtype.t) c =
  let s = Nx_dtype.Scalar.of_dtype dtype in
  match Cell.state c with
  | Live (b :: _) when Nx_dtype.Scalar.equal (Nx_device.Buffer.dtype b) s ->
      Nx_device.Buffer.length b
  | Live _ | Consumed _ -> c.bytes * 8 / Nx_dtype.Scalar.bitsize s

(* Raises unless [b] is of [dtype]'s format. *)
let check_format what dtype b =
  let s = Nx_device.Buffer.dtype b in
  if not (Nx_dtype.Scalar.equal s (Nx_dtype.Scalar.of_dtype dtype)) then
    invalid_arg
      (Printf.sprintf "%s: a %s buffer read as %s" what
         (Nx_dtype.Scalar.to_string s)
         (Nx_dtype.to_string dtype))

(* Raises unless [buffer] is a host buffer of [dtype]'s format, as nx.cpu reads
   it: through its host address, [dtype]'s elements at a time. *)
let check_host what dtype buffer =
  if not (Nx_device.equal (Nx_device.Buffer.device buffer) Nx_device.host) then
    invalid_arg
      (Printf.sprintf "%s: the buffer is on %s, not CPU" what
         (Nx_device.name (Nx_device.Buffer.device buffer)));
  check_format what dtype buffer

(* Traced constructor *)

let traced (type a b) ?view (ctx : context) (p : Placement.t)
    (dtype : (a, b) Nx_dtype.t) (shape : int array) (node : (a, b) node) :
    (a, b) t =
  let t_view =
    match view with
    | None -> View.create shape
    | Some v when Shape.equal (View.shape v) shape -> v
    | Some v ->
        invalid_arg
          (Printf.sprintf "Nx.Repr.Traced.v: a view of shape %s for shape %s"
             (Shape.to_string (View.shape v))
             (Shape.to_string shape))
  in
  Traced
    {
      t_id = fresh_id ();
      t_placement = p;
      t_context = ctx;
      t_dtype = dtype;
      t_view;
      t_node = node;
    }

type packed = P : ('a, 'b) t -> packed

(* Lenses. Metadata is the value's own: reading it runs no interpreter. *)

let view (type a b) (x : (a, b) t) : View.t =
  match x with
  | Host t -> t.view
  | Placed r -> whole_view r
  | Traced t -> t.t_view

let dtype : type a b. (a, b) t -> (a, b) Nx_dtype.t = function
  | Host t -> t.dtype
  | Placed r -> r.r_dtype
  | Traced t -> t.t_dtype

(* A value made beside a placed one is a full copy on each of its devices. The
   frontend asks for a context each time it builds a constant beside an operand,
   so the host's is one value. *)
let context : type a b. (a, b) t -> context = function
  | Host _ -> Placement.host
  | Placed r when Placement.on_disk r.r_placement -> Placement.host
  | Placed r -> Placement.replicated (Placement.devices r.r_placement)
  | Traced t -> t.t_context

(* Whether a creation at [p] makes a host tensor. The frontend's contexts on the
   host are [Placement.host] itself, so the first test decides the common
   case. *)
let on_host (p : context) = p == Placement.host || Placement.is_host p

let placement (type a b) (x : (a, b) t) : Placement.t =
  match x with
  | Host _ -> Placement.host
  | Placed r -> r.r_placement
  | Traced t -> t.t_placement

(* Values over runtime buffers *)

(* [shard_storage what p buffers] is the storage of [buffers], one per device of
   [p], in order, of one length, each in the memory of its device. *)
let shard_storage what p buffers =
  let ds = Placement.devices p in
  if List.compare_lengths ds buffers <> 0 then
    invalid_arg
      (Printf.sprintf "%s: %d buffers for %d devices" what (List.length buffers)
         (List.length ds));
  let bytes = Nx_device.Buffer.nbytes (List.hd buffers) in
  List.iter2
    (fun d b ->
      if Nx_device.Buffer.nbytes b <> bytes then
        invalid_arg (what ^ ": buffers of different sizes");
      if Nx_device.Buffer.device b != Device.memory d then
        invalid_arg
          (Printf.sprintf "%s: a buffer for %s is on %s" what (Device.name d)
             (Nx_device.name (Nx_device.Buffer.device b))))
    ds buffers;
  cell ~placement:p buffers

(* [host_value what dtype view b] is the host value of [b] under [view]. *)
let host_value what dtype view b =
  check_host what dtype b;
  if not (View.within view (Nx_device.Buffer.length b)) then
    invalid_arg (what ^ ": the view reaches outside the buffer");
  Host { dtype; view; buffer = b }

(* [placed_value what p dtype view c] is the value at [p] of [c] under [view].
   Kernels read [view]'s elements of [c]'s bytes as [dtype]'s, so the view lies
   within them and each buffer starts on a byte aligned to one; bool reads only
   bool storage. Consumed storage has no bytes to reach. *)
let placed_value (type a b) what p (dtype : (a, b) Nx_dtype.t) view c =
  if not (View.within view (capacity dtype c)) then
    invalid_arg (what ^ ": the view reaches outside the storage");
  (match (Cell.state c, dtype) with
  | Live bufs, Nx_dtype.Bool ->
      (* A bool's only bytes are 0 and 1, which only bool storage holds. *)
      List.iter (check_format what dtype) bufs
  | Live bufs, _ ->
      if not (List.for_all (aligned dtype) bufs) then
        invalid_arg
          (Printf.sprintf "%s: a buffer at a byte not aligned to %s" what
             (Nx_dtype.to_string dtype))
  | Consumed _, _ -> ());
  placed what p dtype view c

(* [of_shards what p dtype view buffers] is the value at [p] whose elements, on
   each device, are those [view] reaches in its buffer of [buffers]. *)
let of_shards (type a b) what p (dtype : (a, b) Nx_dtype.t) view buffers :
    (a, b) t =
  if Placement.is_host p then
    match buffers with
    | [ b ] -> host_value what dtype view b
    | _ ->
        invalid_arg
          (Printf.sprintf "%s: %d buffers for 1 device" what
             (List.length buffers))
  else begin
    List.iter (check_format what dtype) buffers;
    placed_value what p dtype view (shard_storage what p buffers)
  end

(* [of_buffer dtype shape b] is the value of [shape] over [b]'s elements in C
   order. [View.create] reads a negative dimension as [0] in a shape that has a
   [0], so [shape]'s own dimensions are checked. *)
let of_buffer (type a b) (dtype : (a, b) Nx_dtype.t) shape b : (a, b) t =
  let what = "Nx.of_buffer" in
  let n = Nx_device.Buffer.length b and view = View.create shape in
  if
    Array.exists (fun d -> d < 0) shape
    || not (View.within view n && View.numel view = n)
  then
    invalid_arg
      (Printf.sprintf "%s: shape %s for %d elements" what
         (Shape.to_string shape) n);
  let p = Placement.on (Device.make (Nx_device.Buffer.device b)) in
  of_shards what p dtype view [ b ]

(* The buffer, of [bufs], one per device of [c]'s placement, that holds [c]'s
   storage in [d]'s memory. A value's devices hold their memories' buffers of
   its storage, whichever device over that memory made it. *)
let buffer_on (c : cell) bufs d =
  match
    List.find_index
      (fun h -> Device.memory h == Device.memory d)
      (Placement.devices c.placement)
  with
  | Some i -> List.nth bufs i
  | None -> invalid_arg ("Nx: no storage on " ^ Device.name d)

let shards (type a b) (x : (a, b) t) =
  match x with
  | Host a -> ([ a.buffer ], a.view)
  | Placed r -> (
      match Cell.state r.r_cell with
      | Live buffers ->
          ( List.map
              (fun d -> read_as r.r_dtype (buffer_on r.r_cell buffers d))
              (Placement.devices r.r_placement),
            r.r_view )
      | Consumed k -> consumed k)
  | Traced _ -> outside_trace ()
