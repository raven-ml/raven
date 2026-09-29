(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Nx_array

(* Types

   OCaml extensible GADT constructors (the [E_add], [E_mul], ... below) require
   that type variables in the payload be deducible from the return type. A
   transparent alias of [Nx_cpu.t] would not be injective, so the tensor is
   a GADT of its own, whose parameters the return type determines.

   A tensor is one of three things. [Host] is an nx.cpu tensor, the value of the
   host placement: the host device with the host backend.
   [Placed] is a value at any other placement, held in the memory of its
   devices: nx knows its placement, dtype and view, and the devices' memory
   alone knows its storage. [Traced] is a node of a trace: it has no bytes and
   never will, and the tracer that made it keeps its payload in [t_node].

   The host device thus holds values in two ways. At the host placement a value
   is nx.cpu's own tensor, with no cell, so that the default path
   pays nothing for what compiled calls need of a storage; at another placement
   of the host device (another backend, or beside other devices) it is placed,
   in the runtime's host buffers.

   The views of one placed storage share one cell, which holds what belongs to
   the storage rather than to a view: whether it is live or was consumed by a
   compiled call, and how many reachable programs bind it.

   A placement holds its backend, and a backend's operations take tensors and
   placements, so the types and the backend's signature are one recursive
   definition. *)

(* Placements over a grid

   A placement is one device, or a grid: distinct devices in row-major order
   over the grid's extents, and for each cut tensor axis the grid axes that cut
   it, major first. A grid axis that no cut names holds copies. The
   representation is kept in normal form: extents are at least 2, adjacent grid
   axes merge when no cut names either or one cut names both in order, and a
   grid of one device is that device. It is abstract, over any device type, so
   that nothing outside this module matches on it: it is taken apart by the
   window each device holds. *)

module Grid : sig
  type 'd t

  val device : 'd -> 'd t
  val v : 'd list -> int list -> (int * int list) list -> 'd t
  (* [v devices extents cuts] is the grid over [devices], in row-major order
     over [extents], each [(axis, over)] of [cuts] cutting tensor [axis] over
     the grid axes [over], major first, in normal form. Raises
     [Invalid_argument] unless the extents multiply to the number of devices,
     and the cut axes and the grid axes they name are distinct and in range. *)

  val devices : 'd t -> 'd list
  val cuts : 'd t -> (int * int) list
  (* [cuts p] is each cut tensor axis of [p] with the number of tiles it is cut
     into, by increasing axis. *)

  val tile_index : 'd t -> int -> (int * int) list
  (* [tile_index p k] is, for each cut tensor axis of [p], the index of the tile
     that the device at position [k] holds. *)

  val map_axes : (int -> int) -> 'd t -> 'd t
  (* [map_axes f p] cuts tensor axis [f a] where [p] cuts [a]. *)

  val select : 'd t -> axis:int -> int -> 'd t
  (* [select p ~axis j] is the placement of the devices of [p] that hold tile
     [j] of the cut tensor [axis]: the grid axes of that cut go. *)

  val uncut : 'd t -> axis:int -> 'd t
  (* [uncut p ~axis] is [p] with tensor [axis] whole on every device: the grid
     axes that cut it hold copies. *)

  val equal : ('d -> 'd -> bool) -> 'd t -> 'd t -> bool
  val pp : (Format.formatter -> 'd -> unit) -> Format.formatter -> 'd t -> unit
end = struct
  type cut = { axis : int; over : int list }

  type 'd t =
    | One of 'd
    | Grid of { devices : 'd list; extents : int list; cuts : cut list }

  let device d = One d

  (* Grid axis [g] removed from the cuts, the axes after it renumbered. *)
  let without g cuts =
    List.map
      (fun c ->
        {
          c with
          over =
            List.filter_map
              (fun h ->
                if h = g then None else Some (if h > g then h - 1 else h))
              c.over;
        })
      cuts

  let rec follows g = function
    | a :: (b :: _ as rest) -> (a = g && b = g + 1) || follows g rest
    | _ -> false

  let rec normal devices extents cuts =
    let mentions g = List.exists (fun c -> List.mem g c.over) cuts in
    let mergeable g =
      (not (mentions g || mentions (g + 1)))
      || List.exists (fun c -> follows g c.over) cuts
    in
    let n = List.length extents in
    match List.find_index (( = ) 1) extents with
    | Some g ->
        normal devices
          (List.filteri (fun i _ -> i <> g) extents)
          (without g cuts)
    | None -> (
        match
          List.find_opt mergeable (List.init (Int.max 0 (n - 1)) Fun.id)
        with
        | Some g ->
            let extents =
              List.concat
                (List.mapi
                   (fun i e ->
                     if i = g then [ e * List.nth extents (g + 1) ]
                     else if i = g + 1 then []
                     else [ e ])
                   extents)
            in
            normal devices extents (without (g + 1) cuts)
        | None -> (
            match devices with
            | [ d ] -> One d
            | _ ->
                let cuts = List.filter (fun c -> c.over <> []) cuts in
                let cuts =
                  List.sort (fun a b -> Int.compare a.axis b.axis) cuts
                in
                Grid { devices; extents; cuts }))

  let v devices extents cuts =
    let fail fmt = Printf.ksprintf invalid_arg ("Nx_effect.Grid.v: " ^^ fmt) in
    let rank = List.length extents in
    if List.fold_left ( * ) 1 extents <> List.length devices then
      fail "the extents do not multiply to the number of devices";
    let axes = List.map fst cuts and over = List.concat_map snd cuts in
    let distinct l =
      List.length (List.sort_uniq Int.compare l) = List.length l
    in
    if not (distinct axes && distinct over) then fail "an axis is cut twice";
    if List.exists (fun a -> a < 0) axes then fail "a negative axis";
    if List.exists (fun g -> g < 0 || g >= rank) over then
      fail "a grid axis out of range";
    normal devices extents (List.map (fun (axis, over) -> { axis; over }) cuts)

  let devices = function One d -> [ d ] | Grid g -> g.devices

  let count extents c =
    List.fold_left (fun n g -> n * List.nth extents g) 1 c.over

  let cuts = function
    | One _ -> []
    | Grid { extents; cuts; _ } ->
        List.map (fun c -> (c.axis, count extents c)) cuts

  let tile_index p k =
    match p with
    | One _ -> []
    | Grid { extents; cuts; _ } ->
        let e = Array.of_list extents in
        let coord = Array.make (Array.length e) 0 and r = ref k in
        for g = Array.length e - 1 downto 0 do
          coord.(g) <- !r mod e.(g);
          r := !r / e.(g)
        done;
        List.map
          (fun c ->
            ( c.axis,
              List.fold_left (fun j g -> (j * e.(g)) + coord.(g)) 0 c.over ))
          cuts

  let map_axes f = function
    | One _ as p -> p
    | Grid g ->
        let cuts = List.map (fun c -> { c with axis = f c.axis }) g.cuts in
        Grid
          {
            g with
            cuts = List.sort (fun a b -> Int.compare a.axis b.axis) cuts;
          }

  let select p ~axis j =
    match p with
    | One _ -> p
    | Grid { devices; extents; cuts } ->
        let cut = List.find (fun c -> c.axis = axis) cuts in
        let keep =
          List.filteri
            (fun k _ -> List.assoc axis (tile_index p k) = j)
            (List.mapi (fun k d -> (k, d)) devices)
        in
        let gone = List.sort (fun a b -> Int.compare b a) cut.over in
        let extents = List.filteri (fun g _ -> not (List.mem g gone)) extents in
        let cuts =
          List.fold_left
            (fun cuts g -> without g cuts)
            (List.filter (fun c -> c.axis <> axis) cuts)
            gone
        in
        normal (List.map snd keep) extents cuts

  let uncut p ~axis =
    match p with
    | One _ -> p
    | Grid { devices; extents; cuts } ->
        normal devices extents (List.filter (fun c -> c.axis <> axis) cuts)

  (* Two placements are equal when every device holds the same window under
     both, whatever the shape: the same tile of the same number along every cut
     axis. *)
  let equal eq p q =
    let dq = devices q in
    let tiles p k =
      List.map2 (fun (a, n) (_, j) -> (a, n, j)) (cuts p) (tile_index p k)
    in
    List.compare_lengths (devices p) dq = 0
    && List.for_all
         (fun (k, d) ->
           match List.find_index (eq d) dq with
           | Some k' -> tiles p k = tiles q k'
           | None -> false)
         (List.mapi (fun k d -> (k, d)) (devices p))

  let pp pp_device ppf p =
    let list ppf ds =
      Format.pp_print_list
        ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
        pp_device ppf ds
    in
    match p with
    | One d -> pp_device ppf d
    | Grid { devices; extents = [ _ ]; cuts = [] } ->
        Format.fprintf ppf "replicated [%a]" list devices
    | Grid { devices; extents = [ _ ]; cuts = [ { axis; _ } ] } ->
        Format.fprintf ppf "sharded ~axis:%d [%a]" axis list devices
    | Grid { devices; extents; cuts } ->
        let ints =
          Format.pp_print_list
            ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "x")
            Format.pp_print_int
        in
        Format.fprintf ppf "grid %a [%a]" ints extents list devices;
        List.iter
          (fun c ->
            Format.fprintf ppf " ~axis:%d/%a" c.axis
              (Format.pp_print_list
                 ~pp_sep:(fun ppf () -> Format.pp_print_string ppf ",")
                 Format.pp_print_int)
              c.over)
          cuts
end

module rec Types : sig
  type ('a, 'b) t =
    | Host : ('a, 'b) Nx_array.t -> ('a, 'b) t
    | Placed : ('a, 'b) resident -> ('a, 'b) t
    | Traced : ('a, 'b) traced -> ('a, 'b) t

  and ('a, 'b) resident = {
    r_id : int; (* fresh per value; identity tables key by it *)
    r_placement : placement; (* never the host placement *)
    r_dtype : ('a, 'b) Nx_dtype.t;
    r_view : View.t; (* per shard, the same on every shard *)
    r_cell : cell; (* one per storage, shared by all its views *)
  }

  and cell = {
    placement : placement; (* where the storage lives, whichever views it *)
    length : int; (* elements of the storage, per shard *)
    mutable state : state;
    bound : int Atomic.t; (* reachable program bindings to the storage *)
    lock : Mutex.t;
    mutable readers : int;
    mutable exclusive : bool;
  }

  and state = Live of storage | Consumed of consumption

  (* Where a compiled call consumed a storage: the consumed leaf's path in the
     call's arguments, whose first segment is the argument's position from 0. *)
  and consumption = { path : string }

  and ('a, 'b) traced = {
    t_id : int; (* fresh; identity tables key by it *)
    t_context : placement; (* where the trace creates its values *)
    t_dtype : ('a, 'b) Nx_dtype.t;
    t_view : View.t; (* C-contiguous over the tensor's shape *)
    t_node : node; (* the tracer's payload *)
  }

  (* How a device holds bytes: the library that opens it reads and places the
     values in its memory. Computing on them is not the memory's. *)
  and memory = {
    read : 'a 'b. ('a, 'b) resident -> Nx_device.Buffer.t;
        (* the view's elements, in C order, in a host buffer the caller owns *)
    place : 'a 'b. placement -> ('a, 'b) t -> ('a, 'b) t;
        (* the value on a placement of devices of this memory; its source
           stays. Raises [Invalid_argument] if the placement cannot hold the
           dtype. *)
  }

  and device = { d_id : int; d_name : string; d_memory : memory }
  (* Devices, a layout and the one backend that computes on the values there. *)
  and placement = { grid : device Grid.t; backend : backend }

  and backend = (module Backend_sig.S)
  and storage = ..
  and node = ..
end =
  Types

and Backend_sig : sig
  module type S = sig
    val name : string
    val runs_on : Types.device -> bool
    val place : Types.placement -> ('a, 'b) Types.t -> ('a, 'b) Types.t

    include
      Backend_intf.S
        with type ('a, 'b) t := ('a, 'b) Types.t
         and type context := Types.placement
  end
end =
  Backend_sig

include Types

(* Where a creation makes its value. *)
type context = placement

(* The memory of [Nx_device] devices, the host among them: one buffer per device
   of the cell's placement, in its order. *)
type storage += Runtime of Nx_device.Buffer.t list

let id_counter = Atomic.make 0
let fresh_id () = Atomic.fetch_and_add id_counter 1 + 1

(* Ids are handed out in increasing order, so a tracer can tell the traced
   tensors made before a point of its trace from those made after it. *)
let next_traced_id () = Atomic.get id_counter + 1

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
   their claim while executing outside it; overlapping consumption never waits. *)
module Cell = struct
  let busy () = invalid_arg "Nx: storage is in use by another reader or consuming call"

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
    let valid = not c.exclusive && c.readers > 0 in
    if valid then c.readers <- c.readers - 1;
    Mutex.unlock c.lock;
    if not valid then invalid_arg "Nx: unbalanced storage borrow"

  let with_borrow c f =
    borrow c;
    Fun.protect ~finally:(fun () -> release c) (fun () ->
        match c.state with Live _ -> f () | Consumed k -> consumed k)

  let upgrade c =
    Mutex.lock c.lock;
    let available = not c.exclusive && c.readers = 1 in
    if available then begin c.readers <- 0; c.exclusive <- true end;
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
      match c.state with Consumed _ -> Atomic.get c.bound = 0 | Live _ -> false
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
    let retire = previous = 1 && not c.exclusive
      && match c.state with Consumed _ -> true | Live _ -> false in
    Mutex.unlock c.lock;
    retire
end

(* Reading placed values *)

(* The elements of a placed value's view. *)
let read_elements (type a b) (r : (a, b) resident) : Nx_device.Buffer.t =
  Cell.with_borrow r.r_cell (fun () ->
  match r.r_cell.state with
  | Consumed k -> consumed k
  | Live _ -> (List.hd (Grid.devices r.r_cell.placement.grid)).d_memory.read r)

(* [global p shape] is the shape of a value whose tiles at [p] have [shape]. *)
let global p shape =
  match Grid.cuts p.grid with
  | [] -> shape
  | cuts ->
      let shape = Array.copy shape in
      List.iter (fun (a, n) -> shape.(a) <- shape.(a) * n) cuts;
      shape

(* A split value's view is each shard's, and its shape the whole's. *)
let whole_view r =
  match Grid.cuts r.r_placement.grid with
  | [] -> r.r_view
  | _ ->
      let v = r.r_view in
      View.create ~offset:(View.offset v) ~strides:(View.strides v)
        (global r.r_placement (View.shape v))

(* Raises unless [buffer] is a host buffer of [dtype]'s format, as nx.cpu reads
   it: through its host address, [dtype]'s elements at a time. *)
let check_host fn dtype buffer =
  if not (Nx_device.equal (Nx_device.Buffer.device buffer) Nx_device.host) then
    invalid_arg
      (Printf.sprintf "Nx_effect.%s: the buffer is on %s, not CPU" fn
         (Nx_device.name (Nx_device.Buffer.device buffer)));
  if
    not
      (Nx_dtype.Scalar.equal
         (Nx_device.Buffer.dtype buffer)
         (Nx_dtype.Scalar.of_dtype dtype))
  then
    invalid_arg
      (Printf.sprintf "Nx_effect.%s: a %s buffer read as %s" fn
         (Nx_dtype.Scalar.to_string (Nx_device.Buffer.dtype buffer))
         (Nx_dtype.to_string dtype))

(* [file_run r] is the file bytes that hold [r]'s storage, when they lie on the
   disk at a byte aligned to an element, so that the host can read them in
   place. *)
let file_run (type a b) (r : (a, b) resident) =
  match Cell.state r.r_cell with
  | Live (Runtime [ b ])
    when Nx_device.equal (Nx_device.Buffer.device b) Nx_device.disk ->
      let size =
        Int.max 1 (Nx_dtype.Scalar.bitsize (Nx_device.Buffer.dtype b) / 8)
      in
      if Nativeint.to_int (Nx_device.Buffer.address b) mod size = 0 then Some b
      else None
  | _ -> None

(* A placed value is read by the backend that made its storage, the cell's,
   which a view at a placement of another backend shares. It is read once and
   checked: a buffer of another device or format would otherwise reach nx.cpu's
   kernels. *)
let read_copy (type a b) (r : (a, b) resident) : (a, b) Nx_cpu.t =
  let (module B : Backend_sig.S) = r.r_cell.placement.backend in
  let elements = B.to_host (Placed r) in
  check_host "read" r.r_dtype elements;
  Nx_cpu.reshape
    (Nx_cpu.from_host () r.r_dtype elements)
    (View.shape (whole_view r))

(* A value on the disk is read where it lies, in its file's pages, and keeps its
   view; it is copied when the system does not map the file. *)
let read_host (type a b) (r : (a, b) resident) : (a, b) Nx_cpu.t =
  match file_run r with
  | Some b -> (
      match
        Cell.with_borrow r.r_cell (fun () ->
            Nx_device.Buffer.borrow Nx_device.host b)
      with
      | Ok buffer -> { Nx_array.dtype = r.r_dtype; view = whole_view r; buffer }
      | Error _ -> read_copy r)
  | None -> read_copy r

(* [host_of x] is [x]'s value as a host tensor: [x] itself on the host, a copy
   of its view's elements when it is placed. *)
let host_of : type a b. (a, b) t -> (a, b) Nx_cpu.t = function
  | Host t -> t
  | Placed r -> read_host r
  | Traced _ -> outside_trace ()

(* Devices and placements

   The host device is the one of id 0; ids from [fresh_id] start at 1. It is
   made with the runtime's memory, so the [Device] module is defined below that
   memory. The host backend routes its operations through the code below, so it
   and the [Placement] module come after the routing; the functions here are
   what the code in between needs. The host backend is packed into
   [host_backend] where it is defined: until then no placement exists. *)

exception Out_of_memory of device * int

let () =
  Printexc.register_printer (function
    | Out_of_memory (d, n) ->
        Some (Printf.sprintf "Nx.Device.Out_of_memory(%s, %d bytes)" d.d_name n)
    | _ -> None)

let is_host_device d = d.d_id = 0
let host_backend : backend option ref = ref None
let is_host_backend b = match !host_backend with Some h -> b == h | None -> false

let is_host_placement p =
  is_host_backend p.backend
  && match Grid.devices p.grid with [ d ] -> is_host_device d | _ -> false

let pp_device ppf d = Format.pp_print_string ppf d.d_name
let pp_grid ppf g = Grid.pp pp_device ppf g

let pp_placement ppf p =
  pp_grid ppf p.grid;
  if not (is_host_backend p.backend) then
    let (module B : Backend_sig.S) = p.backend in
    Format.fprintf ppf " with %s" B.name

let devices_of p = Grid.devices p.grid
let memory_of p = (List.hd (devices_of p)).d_memory

(* Raises unless every cut of [p] divides its axis of [shape] evenly. *)
let check_shape what p shape =
  List.iter
    (fun (a, n) ->
      if a >= Array.length shape then
        invalid_arg
          (Printf.sprintf "%s: shape %s has no axis %d to split" what
             (Shape.to_string shape) a);
      if shape.(a) mod n <> 0 then
        invalid_arg
          (Printf.sprintf
             "%s: axis %d of shape %s does not split evenly over %d devices"
             what a (Shape.to_string shape) n))
    (Grid.cuts p.grid)

let window_of p shape d =
  match List.find_index (( == ) d) (devices_of p) with
  | None ->
      invalid_arg
        (Printf.sprintf "Nx.Placement.window: %s holds no window" d.d_name)
  | Some k ->
      check_shape "Nx.Placement.window" p shape;
      let w = Array.map (fun n -> (0, n)) shape in
      List.iter2
        (fun (a, n) (_, j) ->
          let size = shape.(a) / n in
          w.(a) <- (j * size, (j + 1) * size))
        (Grid.cuts p.grid) (Grid.tile_index p.grid k);
      w

(* Placed constructors, for device memories *)

(* A cell over [storage] of [length] elements per device of [placement], whose
   memory owns it. The memory attaches the finaliser that releases the
   storage. *)
let cell ~placement ~length storage =
  { placement; length; state = Live storage; bound = Atomic.make 0;
    lock = Mutex.create (); readers = 0; exclusive = false }

let placed placement dtype view cell =
  if is_host_placement placement then
    invalid_arg "Nx_effect.placed: a placed value is never on the host";
  let held = devices_of cell.placement in
  if
    not (List.for_all (fun d -> List.memq d held) (devices_of placement))
  then
    invalid_arg
      (Format.asprintf "Nx_effect.placed: a value on %a views a storage on %a"
         pp_placement placement pp_placement cell.placement);
  Placed
    {
      r_id = fresh_id ();
      r_placement = placement;
      r_dtype = dtype;
      r_view = view;
      r_cell = cell;
    }

(* Whether [r]'s view covers its storage: C order, offset 0, every element. *)
let covers r =
  View.is_c_contiguous r.r_view
  && View.offset r.r_view = 0
  && View.numel r.r_view = r.r_cell.length

(* [iter_rows box ~into ~at f] calls [f src_off dst_off] for each row of a box
   of extents [box], at [src_off] in the box's elements in C order and at
   [dst_off] in those of shape [into] in C order, the box's corner at [at]. *)
let iter_rows box ~into ~at f =
  let rank = Array.length box in
  let strides = Shape.c_contiguous_strides into in
  let run = box.(rank - 1) and idx = Array.make rank 0 in
  let n = Array.fold_left ( * ) 1 box in
  for row = 0 to (if run = 0 then 0 else n / run) - 1 do
    let base = ref at.(rank - 1) in
    for a = 0 to rank - 2 do
      base := !base + ((at.(a) + idx.(a)) * strides.(a))
    done;
    f (row * run) !base;
    let a = ref (rank - 2) in
    while !a >= 0 do
      idx.(!a) <- idx.(!a) + 1;
      if idx.(!a) < box.(!a) then a := -1
      else begin
        idx.(!a) <- 0;
        decr a
      end
    done
  done

(* [blit_box src box dst ~into ~at] copies [src], the elements of a box of
   extents [box] in C order, into [dst], the elements of shape [into] in C
   order, with the box's corner at [at]. Rows are copied whole, as integer words
   of the element's width: a float copied through an OCaml float would quiet a
   signalling NaN. 4-bit elements are copied as their bits, one at a time. *)
let blit_box src box dst ~into ~at =
  let box, into, at =
    if Array.length box = 0 then ([| 1 |], [| 1 |], [| 0 |]) else (box, into, at)
  in
  let words (type c d) (word : (c, d) Bigarray.kind) w =
    let scale a =
      let a = Array.copy a in
      let r = Array.length a - 1 in
      a.(r) <- a.(r) * w;
      a
    in
    let s = Nx_device.Buffer.bigarray word src
    and d = Nx_device.Buffer.bigarray word dst in
    let box = scale box in
    let run = box.(Array.length box - 1) in
    iter_rows box ~into:(scale into) ~at:(scale at) (fun src_off dst_off ->
        Bigarray.Array1.blit
          (Bigarray.Array1.sub s src_off run)
          (Bigarray.Array1.sub d dst_off run))
  in
  match Nx_dtype.Scalar.bitsize (Nx_device.Buffer.dtype src) with
  | 4 ->
      let bits b =
        Nx_device.Buffer.view b ~offset:0 Nx_dtype.Scalar.UInt4
          (Nx_device.Buffer.length b)
      in
      let get = Elements.get Nx_dtype.uint4 (bits src)
      and set = Elements.set Nx_dtype.uint4 (bits dst) in
      let run = box.(Array.length box - 1) in
      iter_rows box ~into ~at (fun src_off dst_off ->
          for i = 0 to run - 1 do
            set (dst_off + i) (get (src_off + i))
          done)
  | 8 -> words Bigarray.int8_unsigned 1
  | 16 -> words Bigarray.int16_unsigned 1
  | 32 -> words Bigarray.int32 1
  | bits -> words Bigarray.int64 (bits / 64)

(* The box two windows share, [None] when they share no element. *)
let intersect a b =
  let w =
    Array.map2 (fun (lo, hi) (lo', hi') -> (Int.max lo lo', Int.min hi hi')) a b
  in
  if Array.exists (fun (lo, hi) -> lo >= hi) w then None else Some w

(* [within outer w] is window [w] measured from [outer]'s corner. *)
let within outer w =
  Array.map2 (fun (o, _) (lo, hi) -> (lo - o, hi - o)) outer w

let extents w = Array.map (fun (lo, hi) -> hi - lo) w

(* [assemble r window read] is the elements of [window] of the value [r], in C
   order, from [read d v], the elements of the per-shard view [v] on device [d]
   in C order. Each tile meeting the window is read once, from the first device
   that holds it, and only where it meets the window. Memories read placed
   values, and gather the pieces of a move, this way. *)
let assemble (type a b) (r : (a, b) resident) window
    (read : device -> View.t -> Nx_device.Buffer.t) : Nx_device.Buffer.t =
  let p = r.r_placement in
  let shape = global p (View.shape r.r_view) in
  let pieces =
    List.fold_left
      (fun pieces d ->
        let t = window_of p shape d in
        if List.exists (fun (_, t', _) -> t' = t) pieces then pieces
        else
          match intersect window t with
          | Some i -> (d, t, i) :: pieces
          | None -> pieces)
      [] (devices_of p)
  in
  let piece (d, t, i) = read d (View.shrink r.r_view (within t i)) in
  match pieces with
  | [ ((_, _, i) as only) ] when i = window -> piece only
  | _ ->
      let into = extents window in
      let dst = Elements.create r.r_dtype (Array.fold_left ( * ) 1 into) in
      List.iter
        (fun ((_, _, i) as p) ->
          blit_box (piece p) (extents i) dst ~into
            ~at:(Array.map fst (within window i)))
        pieces;
      dst

(* Runtime memory *)

let runtime_lock = Mutex.create ()
let opened : (Nx_device.t * device) list ref = ref []

let runtime_of d =
  if is_host_device d then Nx_device.host
  else
    Mutex.protect runtime_lock (fun () ->
        match List.find_opt (fun (_, d') -> d' == d) !opened with
        | Some (rd, _) -> rd
        | None -> invalid_arg ("Nx: " ^ d.d_name ^ " is not a runtime device"))

let create_runtime d s n =
  try Nx_device.Buffer.create (runtime_of d) s n
  with Nx_device.Out_of_memory (_, bytes) ->
    raise (Out_of_memory (d, bytes))

(* The elements of view [v] of [b], of [dtype]. Int4 storage is read whole: its
   elements may not start on a byte. A strided view is gathered by nx.cpu. *)
let read_view dtype b v =
  let s = Nx_device.Buffer.dtype b in
  let n = View.numel v in
  if n = 0 then Nx_device.Buffer.create Nx_device.host s 0
  else
    let lo, hi =
      match s with
      | Nx_dtype.Scalar.Int4 | UInt4 -> (0, Nx_device.Buffer.length b)
      | _ -> View.extent v
    in
    let span = Nx_device.Buffer.create Nx_device.host s (hi - lo) in
    Nx_device.Buffer.copy
      ~src:
        (Nx_device.Buffer.view b
           ~offset:(lo * Nx_dtype.Scalar.bitsize s / 8)
           s (hi - lo))
      ~dst:span;
    let view =
      View.create
        ~offset:(View.offset v - lo)
        ~strides:(View.strides v) (View.shape v)
    in
    if View.is_c_contiguous view then Elements.contiguous span view
    else Nx_cpu.to_host (Nx_cpu.copy { Nx_array.dtype; view; buffer = span })

(* [run_in b v] is the elements of the view [v] of [b], in C order, as a view of
   [b], when they are a contiguous run of it that starts on a byte. *)
let run_in b v =
  let s = Nx_device.Buffer.dtype b in
  let bits = Nx_dtype.Scalar.bitsize s and n = View.numel v in
  let first = View.offset v * bits in
  if n > 0 && View.is_c_contiguous v && first mod 8 = 0 then
    Some (Nx_device.Buffer.view b ~offset:(first / 8) s n)
  else None

(* [file_windows v ds windows b] is the view each device of [ds] has of its
   window of the view [v] of the file bytes [b], and each device's storage: the
   bytes the window reaches, borrowed from the file's pages. It is [None] unless
   every device shares the host's memory, every window has an element and
   starts on a byte, the windows are one view of their storages, and every
   device borrows them. *)
let file_windows v ds windows b =
  let s = Nx_device.Buffer.dtype b in
  let bits = Nx_dtype.Scalar.bitsize s in
  let span w =
    let vw = View.shrink v w in
    if View.numel vw = 0 then None
    else
      let lo, hi = View.extent vw in
      if lo * bits mod 8 <> 0 then None
      else
        Some
          ( (View.offset vw - lo, View.strides vw, View.shape vw),
            Nx_device.Buffer.view b ~offset:(lo * bits / 8) s (hi - lo) )
  in
  let spans = List.map span windows in
  if
    List.for_all (fun d -> Nx_device.shares_host_memory (runtime_of d)) ds
    && List.for_all Option.is_some spans
  then
    let spans = List.map Option.get spans in
    let ((offset, strides, shape) as view) = fst (List.hd spans) in
    if List.for_all (fun (v', _) -> v' = view) spans then
      let borrows =
        List.map2
          (fun d (_, run) -> Nx_device.Buffer.borrow (runtime_of d) run)
          ds spans
      in
      if List.for_all Result.is_ok borrows then
        Some (View.create ~offset ~strides shape, List.map Result.get_ok borrows)
      else None
    else None
  else None

let runtime_memory =
  let read : type a b. (a, b) resident -> Nx_device.Buffer.t =
   fun r ->
    match Cell.state r.r_cell with
    | Live (Runtime bufs) ->
        let holders = devices_of r.r_cell.placement in
        let buffer_on d =
          List.nth bufs (Option.get (List.find_index (( == ) d) holders))
        in
        let shape = global r.r_placement (View.shape r.r_view) in
        assemble r
          (Array.map (fun n -> (0, n)) shape)
          (fun d v -> read_view r.r_dtype (buffer_on d) v)
    | _ -> assert false (* nx reads consumed values itself *)
  in
  (* A value on the disk is placed on devices that share the host's memory by
     borrowing its file's pages, and keeps its view ([file_windows]). Otherwise
     a window that is a contiguous run of a value's one runtime buffer is copied
     from it, device to device: a value on the disk is read into the device.
     Other windows are copied from the value read to the host. *)
  let place : type a b. placement -> (a, b) t -> (a, b) t =
   fun p x ->
    let host = lazy (match x with Placed r -> read_copy r | _ -> host_of x) in
    let dt, v, run =
      match x with
      | Placed r -> (
          match Cell.state r.r_cell with
          | Live (Runtime [ b ]) -> (r.r_dtype, r.r_view, run_in b)
          | _ -> (r.r_dtype, whole_view r, fun _ -> None))
      | _ ->
          let h = Lazy.force host in
          (h.dtype, h.view, fun _ -> None)
    in
    let shape = View.shape v in
    let s = Nx_dtype.Scalar.of_dtype dt in
    let ds = devices_of p in
    let windows = List.map (fun d -> window_of p shape d) ds in
    let local = extents (List.hd windows) in
    let n = Array.fold_left ( * ) 1 local in
    let borrowed =
      match x with
      | Placed r -> Option.bind (file_run r) (file_windows r.r_view ds windows)
      | _ -> None
    in
    match borrowed with
    | Some (view, bufs) ->
        placed p dt view
          (cell ~placement:p
             ~length:(Nx_device.Buffer.length (List.hd bufs))
             (Runtime bufs))
    | None ->
        let piece w =
          match run (View.shrink v w) with
          | Some b -> b
          | None ->
              let h = Lazy.force host in
              Elements.contiguous (Nx_cpu.to_host h) (View.shrink h.view w)
        in
        let bufs =
          List.map2
            (fun d w ->
              let b = create_runtime d s n in
              if n > 0 then Nx_device.Buffer.copy ~src:(piece w) ~dst:b;
              b)
            ds windows
        in
        placed p dt (View.create local)
          (cell ~placement:p ~length:n (Runtime bufs))
  in
  { read; place }

(* The disk's memory: files, whose values are read as every runtime device's
   are, and where nothing is placed. *)
let disk_memory =
  let place _ _ =
    invalid_arg
      "Nx.place: values on DISK are read from files, and none is placed there"
  in
  { runtime_memory with place }

let disk = { d_id = fresh_id (); d_name = "DISK"; d_memory = disk_memory }
let () = opened := (Nx_device.disk, disk) :: !opened
let on_disk p = match Grid.devices p.grid with [ d ] -> d == disk | _ -> false

(* Devices *)

module Device = struct
  type t = device

  exception Out_of_memory = Out_of_memory

  let host = { d_id = 0; d_name = "CPU"; d_memory = runtime_memory }

  let make name memory =
    { d_id = fresh_id (); d_name = name; d_memory = memory }

  let of_runtime rd =
    if Nx_device.equal rd Nx_device.host then host
    else
      Mutex.protect runtime_lock (fun () ->
          match
            List.find_opt (fun (rd', _) -> Nx_device.equal rd rd') !opened
          with
          | Some (_, d) -> d
          | None ->
              let d = make (Nx_device.name rd) runtime_memory in
              opened := (rd, d) :: !opened;
              d)

  let is_host = is_host_device
  let name d = d.d_name
  let memory d = d.d_memory
  let equal = ( == )
  let compare a b = Int.compare a.d_id b.d_id
  let pp = pp_device
end

(* A hash for identity tables. A placed or traced value hashes by its id, which
   never changes; a host tensor by its structure, which no table sees change,
   since tensors are values. *)
let identity_hash : type a b. (a, b) t -> int = function
  | Host _ as x -> Hashtbl.hash x
  | Placed r -> r.r_id
  | Traced t -> t.t_id

(* Traced constructor *)

let traced (type a b) (ctx : context) (dtype : (a, b) Nx_dtype.t)
    (shape : int array) (node : node) : (a, b) t =
  Traced
    {
      t_id = fresh_id ();
      t_context = ctx;
      t_dtype = dtype;
      t_view = View.create shape;
      t_node = node;
    }

type packed = P : ('a, 'b) t -> packed

(* Effects *)

type _ Effect.t +=
  | E_view : ('a, 'b) t -> View.t Effect.t
  | E_add : { a : ('a, 'b) t; b : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_sub : { a : ('a, 'b) t; b : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_mul : { a : ('a, 'b) t; b : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_idiv : { a : ('a, 'b) t; b : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_fdiv : { a : ('a, 'b) t; b : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_max : { a : ('a, 'b) t; b : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_min : { a : ('a, 'b) t; b : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_mod : { a : ('a, 'b) t; b : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_pow : { a : ('a, 'b) t; b : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_xor : { a : ('a, 'b) t; b : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_or : { a : ('a, 'b) t; b : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_and : { a : ('a, 'b) t; b : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_atan2 : { a : ('a, 'b) t; b : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_cmpeq : {
      a : ('a, 'b) t;
      b : ('a, 'b) t;
    }
      -> (bool, Nx_dtype.bool_elt) t Effect.t
  | E_cmpne : {
      a : ('a, 'b) t;
      b : ('a, 'b) t;
    }
      -> (bool, Nx_dtype.bool_elt) t Effect.t
  | E_cmplt : {
      a : ('a, 'b) t;
      b : ('a, 'b) t;
    }
      -> (bool, Nx_dtype.bool_elt) t Effect.t
  | E_cmple : {
      a : ('a, 'b) t;
      b : ('a, 'b) t;
    }
      -> (bool, Nx_dtype.bool_elt) t Effect.t
  | E_neg : { t_in : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_sin : { t_in : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_sqrt : { t_in : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_recip : { t_in : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_log : { t_in : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_exp : { t_in : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_cos : { t_in : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_abs : { t_in : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_sign : { t_in : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_tan : { t_in : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_asin : { t_in : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_acos : { t_in : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_atan : { t_in : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_sinh : { t_in : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_cosh : { t_in : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_tanh : { t_in : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_trunc : { t_in : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_ceil : { t_in : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_floor : { t_in : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_round : { t_in : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_erf : { t_in : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_where : {
      condition : (bool, Nx_dtype.bool_elt) t;
      if_true : ('a, 'b) t;
      if_false : ('a, 'b) t;
    }
      -> ('a, 'b) t Effect.t
  | E_reduce_sum : {
      t_in : ('a, 'b) t;
      axes : int array;
    }
      -> ('a, 'b) t Effect.t
  | E_reduce_max : {
      t_in : ('a, 'b) t;
      axes : int array;
    }
      -> ('a, 'b) t Effect.t
  | E_reduce_min : {
      t_in : ('a, 'b) t;
      axes : int array;
    }
      -> ('a, 'b) t Effect.t
  | E_reduce_prod : {
      t_in : ('a, 'b) t;
      axes : int array;
    }
      -> ('a, 'b) t Effect.t
  | E_argmax : {
      t_in : ('a, 'b) t;
      axis : int;
      keepdims : bool;
    }
      -> (int32, Nx_dtype.int32_elt) t Effect.t
  | E_argmin : {
      t_in : ('a, 'b) t;
      axis : int;
      keepdims : bool;
    }
      -> (int32, Nx_dtype.int32_elt) t Effect.t
  | E_sort : {
      t_in : ('a, 'b) t;
      axis : int;
      descending : bool;
    }
      -> ('a, 'b) t Effect.t
  | E_argsort : {
      t_in : ('a, 'b) t;
      axis : int;
      descending : bool;
    }
      -> (int32, Nx_dtype.int32_elt) t Effect.t
  | E_associative_scan : {
      t_in : ('a, 'b) t;
      axis : int;
      op : [ `Sum | `Prod | `Max | `Min ];
    }
      -> ('a, 'b) t Effect.t
  | E_permute : { t_in : ('a, 'b) t; axes : int array } -> ('a, 'b) t Effect.t
  | E_reshape : {
      t_in : ('a, 'b) t;
      new_shape : int array;
    }
      -> ('a, 'b) t Effect.t
  | E_expand : {
      t_in : ('a, 'b) t;
      new_target_shape : int array;
    }
      -> ('a, 'b) t Effect.t
  | E_pad : {
      t_in : ('a, 'b) t;
      padding_config : (int * int) array;
      fill_value : 'a;
    }
      -> ('a, 'b) t Effect.t
  | E_shrink : {
      t_in : ('a, 'b) t;
      limits : (int * int) array;
    }
      -> ('a, 'b) t Effect.t
  | E_flip : {
      t_in : ('a, 'b) t;
      dims_to_flip : bool array;
    }
      -> ('a, 'b) t Effect.t
  | E_sliding_window : {
      t_in : ('a, 'b) t;
      axis : int;
      window : int;
      step : int;
    }
      -> ('a, 'b) t Effect.t
  | E_cat : { t_list : ('a, 'b) t list; axis : int } -> ('a, 'b) t Effect.t
  | E_cast : {
      t_in : ('a, 'b) t;
      target_dtype : ('c, 'd) Nx_dtype.t;
    }
      -> ('c, 'd) t Effect.t
  | E_bitcast : {
      t_in : ('a, 'b) t;
      target_dtype : ('c, 'd) Nx_dtype.t;
    }
      -> ('c, 'd) t Effect.t
  | E_contiguous : { t_in : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_copy : { t_in : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_threefry : {
      key : (int32, Nx_dtype.int32_elt) t;
      ctr : (int32, Nx_dtype.int32_elt) t;
    }
      -> (int32, Nx_dtype.int32_elt) t Effect.t
  | E_gather : {
      data : ('a, 'b) t;
      indices : (int32, Nx_dtype.int32_elt) t;
      axis : int;
    }
      -> ('a, 'b) t Effect.t
  | E_scatter : {
      data_template : ('a, 'b) t;
      indices : (int32, Nx_dtype.int32_elt) t;
      updates : ('a, 'b) t;
      axis : int;
      mode : [ `Set | `Add ];
      unique_indices : bool;
    }
      -> ('a, 'b) t Effect.t
  | E_update : {
      t_in : ('a, 'b) t;
      starts : (int32, Nx_dtype.int32_elt) t;
      v : ('a, 'b) t;
    }
      -> ('a, 'b) t Effect.t
  | E_place : {
      placement : placement;
      t_in : ('a, 'b) t;
    }
      -> ('a, 'b) t Effect.t
  | E_placement : ('a, 'b) t -> placement Effect.t
  | E_unfold : {
      t_in : ('a, 'b) t;
      kernel_size : int array;
      stride : int array;
      dilation : int array;
      padding : (int * int) array;
    }
      -> ('a, 'b) t Effect.t
  | E_fold : {
      t_in : ('a, 'b) t;
      output_size : int array;
      kernel_size : int array;
      stride : int array;
      dilation : int array;
      padding : (int * int) array;
    }
      -> ('a, 'b) t Effect.t
  | E_matmul : { a : ('a, 'b) t; b : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_fft : {
      t : (Complex.t, 'b) t;
      axes : int array;
    }
      -> (Complex.t, 'b) t Effect.t
  | E_ifft : {
      t : (Complex.t, 'b) t;
      axes : int array;
    }
      -> (Complex.t, 'b) t Effect.t
  | E_rfft : {
      t : (float, 'b) t;
      dtype : (Complex.t, 'c) Nx_dtype.t;
      axes : int array;
    }
      -> (Complex.t, 'c) t Effect.t
  | E_irfft : {
      t : (Complex.t, 'b) t;
      dtype : (float, 'c) Nx_dtype.t;
      axes : int array;
      s : int array option;
    }
      -> (float, 'c) t Effect.t
  | E_cholesky : { t_in : ('a, 'b) t; upper : bool } -> ('a, 'b) t Effect.t
  | E_qr : {
      t_in : ('a, 'b) t;
      reduced : bool;
    }
      -> (('a, 'b) t * ('a, 'b) t) Effect.t
  | E_lu : {
      t_in : ('a, 'b) t;
    }
      -> (('a, 'b) t
         * (int32, Nx_dtype.int32_elt) t
         * (int32, Nx_dtype.int32_elt) t)
         Effect.t
  | E_svd : {
      t_in : ('a, 'b) t;
      full_matrices : bool;
    }
      -> (('a, 'b) t * (float, Nx_dtype.float64_elt) t * ('a, 'b) t) Effect.t
  | E_eigvals : {
      t_in : ('a, 'b) t;
    }
      -> (Complex.t, Nx_dtype.complex64_elt) t Effect.t
  | E_eig : {
      t_in : ('a, 'b) t;
    }
      -> ((Complex.t, Nx_dtype.complex64_elt) t
         * (Complex.t, Nx_dtype.complex64_elt) t)
         Effect.t
  | E_eigvalsh : {
      t_in : ('a, 'b) t;
    }
      -> (float, Nx_dtype.float64_elt) t Effect.t
  | E_eigh : {
      t_in : ('a, 'b) t;
    }
      -> ((float, Nx_dtype.float64_elt) t * ('a, 'b) t) Effect.t
  | E_solve_triangular : {
      a : ('a, 'b) t;
      b : ('a, 'b) t;
      upper : bool;
      transpose : bool;
      unit_diag : bool;
    }
      -> ('a, 'b) t Effect.t
  | E_to_host : ('a, 'b) t -> Nx_device.Buffer.t Effect.t

(* Lenses. The effect is performed first: a handler may present a transformed
   view (vmap shows batched tensors without their batch axis) or placement; only
   the unhandled fallback answers from the tensor. *)

let view (type a b) (x : (a, b) t) : View.t =
  try Effect.perform (E_view x)
  with Effect.Unhandled _ -> (
    match x with
    | Host t -> t.view
    | Placed r -> whole_view r
    | Traced t -> t.t_view)

let dtype : type a b. (a, b) t -> (a, b) Nx_dtype.t = function
  | Host t -> t.dtype
  | Placed r -> r.r_dtype
  | Traced t -> t.t_dtype

(* Routing

   Every fallback runs where its operands live. Operands all on the host run on
   nx.cpu. Placed operands must share their devices and backend, and host
   operands join them: on the host backend, the operation reads the placed
   operands' windows, runs nx.cpu and places its result. The route is decided
   before anything is read, so operands on two device lists raise before any
   work.

   The result takes the placement tolk's multi-device rewrite gives the same
   operation in a compiled program (schedule/multi.ml), so eager and compiled
   placements agree and stay as they are once devices compute: an elementwise
   operation keeps its operands' split, resharding to the last split axis among
   them; a reduction over a split axis holds a copy on every device; an
   operation along a split axis, or of a kind tolk has no rule for, raises. *)

type route = On_host | At of placement

(* How the axes of an operation's result derive from its operands', which
   decides where the result lives. *)
type rule =
  | Elementwise (* the operands' shape *)
  | Along of int list
    (* acts along these axes, the others as elementwise: sort, scan, pad,
       concatenation, fft, linear algebra *)
  | Gather of int
    (* reads its first operand along this axis at the positions its second
       holds, the other axes as elementwise; along a split axis, a reduction *)
  | Reduce of { axes : int array; keepdims : bool }
  | Contract
    (* a product over the last axis of the first and the next-to-last of the
       second *)
  | Into (* the first operand, with the others written into it *)

(* The disk holds values and computes on none: a value on it takes part in an
   operation as a host value, which the operation reads. *)
let placement_of : type a b. (a, b) t -> placement option = function
  | Host _ -> None
  | Placed r when on_disk r.r_placement -> None
  | Placed r -> Some r.r_placement
  | Traced _ -> outside_trace ()

let rank (P x) = Array.length (View.shape (view x))

(* The last two axes of [x], along which linear algebra acts. *)
let matrix_axes x =
  let r = rank (P x) in
  [ r - 2; r - 1 ]

(* Every axis of [x] but the first, along which windows are taken. *)
let spatial x = List.init (rank (P x) - 1) succ

(* The axes [padding] pads. *)
let padded padding =
  List.filter
    (fun a -> padding.(a) <> (0, 0))
    (List.init (Array.length padding) Fun.id)

let along op axis =
  invalid_arg
    (Printf.sprintf
       "Nx.%s: the operation runs along the split axis %d, whose shards are on \
        different devices; place the value replicated or on one device first"
       op axis)

(* [combine op gs] is the grid of an elementwise operation's result over
   operands on grids [gs]: that of the split ones, which must be alike; copies
   take it. *)
let combine op gs =
  match List.filter (fun g -> Grid.cuts g <> []) gs with
  | [] -> List.hd gs
  | g :: rest -> (
      match List.find_opt (fun h -> not (Grid.equal ( == ) g h)) rest with
      | None -> g
      | Some h ->
          invalid_arg
            (Format.asprintf
               "Nx.%s: operands at %a and %a are split differently; place them \
                alike first"
               op pp_grid g pp_grid h))

(* [result op rule operands] is where [op]'s result lives, over operands of
   these placements ([None] on the host) and ranks, the placed ones among them
   sharing their devices and backend, which the result keeps. *)
let result op rule operands =
  let backend =
    match List.find_map fst operands with
    | Some p -> p.backend
    | None -> invalid_arg "Nx_effect.result: no placed operand"
  in
  let operands = List.map (fun (p, n) -> (Option.map (fun p -> p.grid) p, n)) operands in
  let ps = List.filter_map fst operands in
  let cut p a = List.mem_assoc a (Grid.cuts p) in
  let grid =
    match rule with
    | Elementwise -> combine op ps
    | Along axes ->
        List.iter
          (fun p -> List.iter (fun a -> if cut p a then along op a) axes)
          ps;
        combine op ps
    | Gather axis -> (
        match operands with
        | (Some p, _) :: _ when cut p axis ->
            (* Each device selects among the rows it holds and the selections sum
               across the devices, as tolk lowers a gather: a copy on each. A 1-D
               grid's only cut is this one; a grid cut along other axes too would
               keep those cuts, renumbered as [Reduce] does. *)
            List.fold_left (fun p (a, _) -> Grid.uncut p ~axis:a) p (Grid.cuts p)
        | _ -> combine op ps)
    | Reduce { axes; keepdims } ->
        let reduce p =
          let p =
            Array.fold_left
              (fun p a -> if cut p a then Grid.uncut p ~axis:a else p)
              p axes
          in
          if keepdims then p
          else
            Grid.map_axes
              (fun a ->
                a - Array.fold_left (fun n r -> if r < a then n + 1 else n) 0 axes)
              p
        in
        combine op (List.map reduce ps)
    | Contract ->
        (* As [a @ b] is [a [..., m, 1, k] * b [..., 1, n, k]] summed over [k]. *)
        let r = List.fold_left (fun r (_, n) -> Int.max r n) 0 operands in
        let lift j (p, n) =
          let axis i =
            if (j = 0 && i = n - 1) || (j = 1 && i = n - 2) then r else i + r - n
          in
          Option.map (Grid.map_axes axis) p
        in
        let p =
          combine op
            (List.concat
               (List.mapi (fun j x -> Option.to_list (lift j x)) operands))
        in
        if cut p r then Grid.uncut p ~axis:r else p
    | Into -> (
        match operands with
        | (Some p, _) :: _ -> p
        | _ ->
            let p = combine op ps in
            List.fold_left (fun p (a, _) -> Grid.uncut p ~axis:a) p (Grid.cuts p))
  in
  { grid; backend }

let same_devices p q =
  let dp = devices_of p and dq = devices_of q in
  List.compare_lengths dp dq = 0 && List.for_all (fun d -> List.memq d dq) dp

(* Views of whole shards of one split storage, each on its own device. A
   compiled program copies such a view to every device of the storage's list
   (schedule/multi.ml, shrink_multi), so eager code joins them as copies on that
   list: [Nx.roll] of a value split in two by one shard succeeds and lands where
   it does compiled. [whole_shards xs] is that list. *)
let whole_shards xs =
  let views =
    List.filter_map
      (fun (P x) ->
        match x with
        | Placed r when not (on_disk r.r_placement) ->
            Some (r.r_cell, r.r_view, r.r_placement)
        | _ -> None)
      xs
  in
  (* A view of a whole shard reaches as many elements as the shard holds, and
     none twice through a broadcast axis. *)
  let whole c v =
    View.numel v = c.length
    && not
         (Array.exists2
            (fun n s -> n > 1 && s = 0)
            (View.shape v) (View.strides v))
  in
  match views with
  | (cell, _, _) :: _
    when List.for_all
           (fun (c, v, p) ->
             c == cell && whole c v
             && List.compare_length_with (devices_of p) 1 = 0)
           views ->
      Some (devices_of cell.placement)
  | _ -> None

let mixed op p q =
  invalid_arg
    (Format.asprintf "Nx.%s: operands on %a and %a; place one of them" op
       pp_placement p pp_placement q)

(* [route op rule xs] is where [op] runs over [xs]. Placed operands with
   different backends raise, and so do those on different device sets, but for
   [whole_shards]. *)
let route op rule xs =
  match List.filter_map (fun (P x) -> placement_of x) xs with
  | [] -> On_host
  | p :: rest -> (
      (match List.find_opt (fun q -> q.backend != p.backend) rest with
      | Some q -> mixed op p q
      | None -> ());
      match List.find_opt (fun q -> not (same_devices p q)) rest with
      | None ->
          At
            (result op rule
               (List.map (fun (P x as o) -> (placement_of x, rank o)) xs))
      | Some q -> (
          match whole_shards xs with
          | Some ds -> At { grid = Grid.v ds [ List.length ds ] []; backend = p.backend }
          | None -> mixed op p q))

(* [routing e] is the name, rule and operands by which the operation that
   performs [e] is routed, and [None] for an effect that no operation routes:
   views, creation, movement, placement and reads. Eager routing and a compiler
   checking where values live read the same rule from it. *)
let routing : type r. r Effect.t -> (string * rule * packed list) option =
 fun e ->
  let each name xs = Some (name, Elementwise, xs) in
  let reduced axes = Reduce { axes; keepdims = false } in
  match e with
  | E_add { a; b } -> each "add" [ P a; P b ]
  | E_sub { a; b } -> each "sub" [ P a; P b ]
  | E_mul { a; b } -> each "mul" [ P a; P b ]
  | E_idiv { a; b } -> each "div" [ P a; P b ]
  | E_fdiv { a; b } -> each "div" [ P a; P b ]
  | E_max { a; b } -> each "max" [ P a; P b ]
  | E_min { a; b } -> each "min" [ P a; P b ]
  | E_mod { a; b } -> each "mod" [ P a; P b ]
  | E_pow { a; b } -> each "pow" [ P a; P b ]
  | E_xor { a; b } -> each "xor" [ P a; P b ]
  | E_or { a; b } -> each "or" [ P a; P b ]
  | E_and { a; b } -> each "and" [ P a; P b ]
  | E_atan2 { a; b } -> each "atan2" [ P a; P b ]
  | E_cmpeq { a; b } -> each "equal" [ P a; P b ]
  | E_cmpne { a; b } -> each "not_equal" [ P a; P b ]
  | E_cmplt { a; b } -> each "less" [ P a; P b ]
  | E_cmple { a; b } -> each "less_equal" [ P a; P b ]
  | E_neg { t_in } -> each "neg" [ P t_in ]
  | E_sin { t_in } -> each "sin" [ P t_in ]
  | E_sqrt { t_in } -> each "sqrt" [ P t_in ]
  | E_recip { t_in } -> each "recip" [ P t_in ]
  | E_log { t_in } -> each "log" [ P t_in ]
  | E_exp { t_in } -> each "exp" [ P t_in ]
  | E_cos { t_in } -> each "cos" [ P t_in ]
  | E_abs { t_in } -> each "abs" [ P t_in ]
  | E_sign { t_in } -> each "sign" [ P t_in ]
  | E_tan { t_in } -> each "tan" [ P t_in ]
  | E_asin { t_in } -> each "asin" [ P t_in ]
  | E_acos { t_in } -> each "acos" [ P t_in ]
  | E_atan { t_in } -> each "atan" [ P t_in ]
  | E_sinh { t_in } -> each "sinh" [ P t_in ]
  | E_cosh { t_in } -> each "cosh" [ P t_in ]
  | E_tanh { t_in } -> each "tanh" [ P t_in ]
  | E_trunc { t_in } -> each "trunc" [ P t_in ]
  | E_ceil { t_in } -> each "ceil" [ P t_in ]
  | E_floor { t_in } -> each "floor" [ P t_in ]
  | E_round { t_in } -> each "round" [ P t_in ]
  | E_erf { t_in } -> each "erf" [ P t_in ]
  | E_contiguous { t_in } -> each "contiguous" [ P t_in ]
  | E_copy { t_in } -> each "copy" [ P t_in ]
  | E_cast { t_in; _ } -> each "cast" [ P t_in ]
  | E_bitcast { t_in; _ } -> each "bitcast" [ P t_in ]
  | E_where { condition; if_true; if_false } ->
      each "where" [ P condition; P if_true; P if_false ]
  | E_threefry { key; ctr } -> each "threefry" [ P key; P ctr ]
  | E_reduce_sum { t_in; axes } -> Some ("reduce", reduced axes, [ P t_in ])
  | E_reduce_prod { t_in; axes } -> Some ("reduce", reduced axes, [ P t_in ])
  | E_reduce_max { t_in; axes } -> Some ("reduce", reduced axes, [ P t_in ])
  | E_reduce_min { t_in; axes } -> Some ("reduce", reduced axes, [ P t_in ])
  | E_argmax { t_in; axis; keepdims } ->
      Some ("argmax", Reduce { axes = [| axis |]; keepdims }, [ P t_in ])
  | E_argmin { t_in; axis; keepdims } ->
      Some ("argmin", Reduce { axes = [| axis |]; keepdims }, [ P t_in ])
  | E_associative_scan { t_in; axis; _ } ->
      Some ("associative_scan", Along [ axis ], [ P t_in ])
  | E_sort { t_in; axis; _ } -> Some ("sort", Along [ axis ], [ P t_in ])
  | E_argsort { t_in; axis; _ } -> Some ("argsort", Along [ axis ], [ P t_in ])
  | E_pad { t_in; padding_config; _ } ->
      Some ("pad", Along (padded padding_config), [ P t_in ])
  | E_cat { t_list; axis } ->
      Some ("concatenate", Along [ axis ], List.map (fun x -> P x) t_list)
  | E_gather { data; indices; axis } ->
      Some ("take", Gather axis, [ P data; P indices ])
  | E_update { t_in; starts; v } -> Some ("set", Into, [ P t_in; P starts; P v ])
  | E_scatter { data_template; indices; updates; _ } ->
      Some ("scatter", Into, [ P data_template; P indices; P updates ])
  | E_unfold { t_in; _ } -> Some ("unfold", Along (spatial t_in), [ P t_in ])
  | E_fold { t_in; _ } -> Some ("fold", Along (spatial t_in), [ P t_in ])
  | E_matmul { a; b } -> Some ("matmul", Contract, [ P a; P b ])
  | E_fft { t; axes } -> Some ("fft", Along (Array.to_list axes), [ P t ])
  | E_ifft { t; axes } -> Some ("ifft", Along (Array.to_list axes), [ P t ])
  | E_rfft { t; axes; _ } -> Some ("rfft", Along (Array.to_list axes), [ P t ])
  | E_irfft { t; axes; _ } -> Some ("irfft", Along (Array.to_list axes), [ P t ])
  | E_cholesky { t_in; _ } ->
      Some ("cholesky", Along (matrix_axes t_in), [ P t_in ])
  | E_qr { t_in; _ } -> Some ("qr", Along (matrix_axes t_in), [ P t_in ])
  | E_lu { t_in } -> Some ("lu", Along (matrix_axes t_in), [ P t_in ])
  | E_svd { t_in; _ } -> Some ("svd", Along (matrix_axes t_in), [ P t_in ])
  | E_eigvals { t_in } -> Some ("eigvals", Along (matrix_axes t_in), [ P t_in ])
  | E_eig { t_in } -> Some ("eig", Along (matrix_axes t_in), [ P t_in ])
  | E_eigvalsh { t_in } ->
      Some ("eigvalsh", Along (matrix_axes t_in), [ P t_in ])
  | E_eigh { t_in } -> Some ("eigh", Along (matrix_axes t_in), [ P t_in ])
  | E_solve_triangular { a; b; _ } ->
      Some ("solve_triangular", Along (matrix_axes a), [ P a; P b ])
  | _ -> None

(* Where the operation that performs [e] runs. *)
let route_of e =
  match routing e with
  | Some (op, rule, xs) -> route op rule xs
  | None -> invalid_arg "Nx_effect.route_of: no operation routes this effect"

let settle : type a b. route -> (a, b) Nx_cpu.t -> (a, b) t =
 fun r h ->
  match r with
  | On_host -> Host h
  | At p -> (memory_of p).place p (Host h)

(* [routed e x f] runs [f] over [x], no host tensor, where the operation that
   performs [e] runs. *)
let routed e x f =
  let r = route_of e in
  settle r (f (host_of x))

(* Movements

   A movement of a placed value is view arithmetic over the same storage, the
   same on every device. A split value moves shard by shard, so the movement
   must leave every element on its device: the split axis may move, stay whole,
   or be reshaped with the whole axes before it, but it is never flipped,
   windowed or cut across shards. A cut inside one shard is a view of that
   shard, on its device alone. These are the rules tolk applies to a compiled
   program over split values (schedule/multi.ml), so a value moved eagerly has
   the placement the same movement has in a compiled program, except for a cut
   inside one shard: tolk copies a whole shard to every device and refuses a
   part of one. *)

type movement =
  | Reshape of int array
  | Expand of int array
  | Permute of int array
  | Shrink of (int * int) array
  | Flip of bool array
  | Sliding_window of { axis : int; window : int; step : int }

(* The movement [e] performs, and its operand; [None] for an effect that moves
   nothing. *)
let movement_of : type r. r Effect.t -> (packed * movement) option = function
  | E_reshape { t_in; new_shape } -> Some (P t_in, Reshape new_shape)
  | E_expand { t_in; new_target_shape } -> Some (P t_in, Expand new_target_shape)
  | E_permute { t_in; axes } -> Some (P t_in, Permute axes)
  | E_shrink { t_in; limits } -> Some (P t_in, Shrink limits)
  | E_flip { t_in; dims_to_flip } -> Some (P t_in, Flip dims_to_flip)
  | E_sliding_window { t_in; axis; window; step } ->
      Some (P t_in, Sliding_window { axis; window; step })
  | _ -> None

let move_view v = function
  | Reshape shape -> View.reshape v shape
  | Expand shape -> View.expand v shape
  | Permute order -> View.permute v order
  | Shrink limits -> View.shrink v limits
  | Flip dims -> View.flip v dims
  | Sliding_window { axis; window; step } ->
      View.sliding_window v ~axis ~window ~step

(* What a movement does to one cut axis: the axis stays cut, at an index, or the
   movement keeps a single tile along it. *)
type split_axis = Split of int | Shard of int

(* [split_axis ~axis ~n shape m] is what [m] does to the tensor [axis] of a
   value of shape [shape] cut in [n] tiles along it. Every cut is decided
   against the whole shapes, as tolk's rewrite does, so a value cut along
   several axes cannot land two cuts on one axis. Raises [Invalid_argument] if
   [m] would move elements between tiles. *)
let split_axis ~axis ~n shape m =
  let k = shape.(axis) / n in
  let across what =
    invalid_arg
      (Printf.sprintf
         "Nx: a %s of the split axis %d of shape %s would move elements \
          between devices; place the value replicated or on one device first"
         what axis (Shape.to_string shape))
  in
  match m with
  | Permute order ->
      let a = ref 0 in
      Array.iteri (fun i o -> if o = axis then a := i) order;
      Split !a
  | Expand _ ->
      (* The split axis spans at least two shards, so it is never broadcast. *)
      Split axis
  | Reshape target ->
      (* The split axis becomes the last axis whose leading extents multiply to
         those of the split axis; its extent must divide over the shards. *)
      let lead = ref 1 in
      for d = 0 to axis - 1 do
        lead := !lead * shape.(d)
      done;
      let a = ref (-1) and acc = ref 1 in
      Array.iteri
        (fun i d ->
          if !acc = !lead then a := i;
          acc := !acc * d)
        target;
      if !acc <> Array.fold_left ( * ) 1 shape then
        invalid_arg
          (Printf.sprintf "Nx.reshape: cannot reshape %s to %s"
             (Shape.to_string shape) (Shape.to_string target));
      if !a < 0 || target.(!a) mod n <> 0 then across "reshape";
      Split !a
  | Shrink limits ->
      let lo, hi = limits.(axis) in
      if lo = 0 && hi = shape.(axis) then Split axis
      else if lo / k = (hi - 1) / k then Shard (lo / k)
      else across "cut"
  | Flip dims -> if dims.(axis) then across "flip" else Split axis
  | Sliding_window { axis = a; _ } ->
      if a = axis then across "window" else Split axis

(* [fates p shape m] is what [m] does to each cut axis of a value of shape
   [shape] at [p]: the axis, its number of tiles and its fate. *)
let fates p shape m =
  List.map
    (fun (axis, n) -> (axis, n, split_axis ~axis ~n shape m))
    (Grid.cuts p.grid)

(* [localize shape m fates] is [m] as one tile of a value of shape [shape] sees
   it, [fates] giving each cut axis, its number of tiles and what [m] does to
   it. *)
let localize shape m fates =
  let each f =
    List.iter (fun (axis, n, fate) -> f axis (shape.(axis) / n) n fate) fates
  in
  match m with
  | Reshape target ->
      let local = Array.copy target in
      each (fun _ _ n fate ->
          match fate with
          | Split a -> local.(a) <- target.(a) / n
          | Shard _ -> ());
      Reshape local
  | Expand target ->
      let local = Array.copy target in
      each (fun axis k _ _ ->
          if axis < Array.length local && target.(axis) = shape.(axis) then
            local.(axis) <- k);
      Expand local
  | Shrink limits ->
      let local = Array.copy limits in
      each (fun axis k _ fate ->
          let lo, hi = limits.(axis) in
          local.(axis) <-
            (match fate with
            | Split _ -> (0, k)
            | Shard j -> (lo - (j * k), hi - (j * k))));
      Shrink local
  | Permute _ | Flip _ | Sliding_window _ -> m

(* The placement of a value at [p] moved as [fates] say: an axis that stays cut
   moves where the movement puts it, and a cut to one tile keeps the devices
   that hold that tile. *)
let placement_after p fates =
  let g =
    List.fold_left
      (fun g (axis, _, fate) ->
        match fate with Shard j -> Grid.select g ~axis j | Split _ -> g)
      p.grid fates
  in
  let grid =
    Grid.map_axes
      (fun a ->
        match List.find (fun (axis, _, _) -> axis = a) fates with
        | _, _, Split a' -> a'
        | _, _, Shard _ -> a)
      g
  in
  { p with grid }

(* [moved_placement p shape m] is the placement of a value of shape [shape] at
   [p] moved by [m]. Raises [Invalid_argument] as [split_axis] does. *)
let moved_placement p shape m = placement_after p (fates p shape m)

(* [split_view p v m] is the placement and per-shard view of a value at [p]
   whose per-shard view is [v], moved by [m]. *)
let split_view p v m =
  match Grid.cuts p.grid with
  | [] -> (p, move_view v m)
  | _ ->
      let shape = global p (View.shape v) in
      let fates = fates p shape m in
      (placement_after p fates, move_view v (localize shape m fates))

(* The host backend

   nx.cpu's kernels over nx's values. Host operands run in place. Otherwise the
   operation reads its operands' elements to the host through their backends,
   runs there, and [settle] places the result where [route] says, through the
   memory of the result's devices: today's routing, whatever the operands'
   backend, so a backend that includes this one keeps it. *)

(* The route is decided before anything is read. *)
let routed2 e a b f =
  let r = route_of e in
  settle r (f (host_of a) (host_of b))
let unary e op x = match x with Host t -> Host (op t) | _ -> routed e x op

let binary e op a b =
  match (a, b) with
  | Host a, Host b -> Host (op a b)
  | _ -> routed2 e a b op

let moved e op x arg =
  match (x, movement_of e) with
  | Host t, _ -> Host (op t arg)
  | Placed r, Some (_, m) ->
      let r_placement, r_view = split_view r.r_placement r.r_view m in
      Placed { r with r_id = fresh_id (); r_placement; r_view }
  | Placed _, None -> invalid_arg "Nx_effect.moved: not a movement"
  | Traced _, _ -> outside_trace ()

let create p make =
  if is_host_placement p then Host (make ()) else settle (At p) (make ())

let reduce_effect ~op ~axes t_in =
  match op with
  | `Sum -> E_reduce_sum { t_in; axes }
  | `Prod -> E_reduce_prod { t_in; axes }
  | `Max -> E_reduce_max { t_in; axes }
  | `Min -> E_reduce_min { t_in; axes }

module Host_backend = struct
  let name = "host"
  let runs_on _ = true

  (* A placed value at a placement that differs from [p] only in backend is a
     view of its storage, which that backend still reads. *)
  let place (type a b) p (x : (a, b) t) : (a, b) t =
    match x with
    | Placed r when Grid.equal ( == ) r.r_placement.grid p.grid ->
        Placed { r with r_id = fresh_id (); r_placement = p }
    | _ -> (memory_of p).place p x

  let to_host : type a b. (a, b) t -> Nx_device.Buffer.t = function
    | Host t -> Nx_cpu.to_host t
    | Placed r -> read_elements r
    | Traced _ -> outside_trace ()

  let buffer p dtype shape =
    create p (fun () -> Nx_cpu.buffer () dtype shape)

  let full p dtype shape value =
    create p (fun () -> Nx_cpu.full () dtype shape value)

  let from_host p dtype buffer =
    create p (fun () -> Nx_cpu.from_host () dtype buffer)

  let add a b = binary (E_add { a; b }) Nx_cpu.add a b
  let sub a b = binary (E_sub { a; b }) Nx_cpu.sub a b
  let mul a b = binary (E_mul { a; b }) Nx_cpu.mul a b
  let fdiv a b = binary (E_fdiv { a; b }) Nx_cpu.fdiv a b
  let idiv a b = binary (E_idiv { a; b }) Nx_cpu.idiv a b
  let mod_ a b = binary (E_mod { a; b }) Nx_cpu.mod_ a b
  let pow a b = binary (E_pow { a; b }) Nx_cpu.pow a b
  let atan2 a b = binary (E_atan2 { a; b }) Nx_cpu.atan2 a b
  let cmpeq a b = binary (E_cmpeq { a; b }) Nx_cpu.cmpeq a b
  let cmpne a b = binary (E_cmpne { a; b }) Nx_cpu.cmpne a b
  let cmplt a b = binary (E_cmplt { a; b }) Nx_cpu.cmplt a b
  let cmple a b = binary (E_cmple { a; b }) Nx_cpu.cmple a b
  let max a b = binary (E_max { a; b }) Nx_cpu.max a b
  let min a b = binary (E_min { a; b }) Nx_cpu.min a b
  let xor a b = binary (E_xor { a; b }) Nx_cpu.xor a b
  let or_ a b = binary (E_or { a; b }) Nx_cpu.or_ a b
  let and_ a b = binary (E_and { a; b }) Nx_cpu.and_ a b
  let neg t = unary (E_neg { t_in = t }) Nx_cpu.neg t
  let recip t = unary (E_recip { t_in = t }) Nx_cpu.recip t
  let abs t = unary (E_abs { t_in = t }) Nx_cpu.abs t
  let sqrt t = unary (E_sqrt { t_in = t }) Nx_cpu.sqrt t
  let sign t = unary (E_sign { t_in = t }) Nx_cpu.sign t
  let exp t = unary (E_exp { t_in = t }) Nx_cpu.exp t
  let log t = unary (E_log { t_in = t }) Nx_cpu.log t
  let sin t = unary (E_sin { t_in = t }) Nx_cpu.sin t
  let cos t = unary (E_cos { t_in = t }) Nx_cpu.cos t
  let tan t = unary (E_tan { t_in = t }) Nx_cpu.tan t
  let asin t = unary (E_asin { t_in = t }) Nx_cpu.asin t
  let acos t = unary (E_acos { t_in = t }) Nx_cpu.acos t
  let atan t = unary (E_atan { t_in = t }) Nx_cpu.atan t
  let sinh t = unary (E_sinh { t_in = t }) Nx_cpu.sinh t
  let cosh t = unary (E_cosh { t_in = t }) Nx_cpu.cosh t
  let tanh t = unary (E_tanh { t_in = t }) Nx_cpu.tanh t
  let trunc t = unary (E_trunc { t_in = t }) Nx_cpu.trunc t
  let ceil t = unary (E_ceil { t_in = t }) Nx_cpu.ceil t
  let floor t = unary (E_floor { t_in = t }) Nx_cpu.floor t
  let round t = unary (E_round { t_in = t }) Nx_cpu.round t
  let erf t = unary (E_erf { t_in = t }) Nx_cpu.erf t

  let where condition if_true if_false =
    match (condition, if_true, if_false) with
    | Host c, Host a, Host b -> Host (Nx_cpu.where c a b)
    | _ ->
        let r = route_of (E_where { condition; if_true; if_false }) in
        settle r
          (Nx_cpu.where (host_of condition) (host_of if_true)
             (host_of if_false))

  let reduce ~op ~axes t_in =
    unary (reduce_effect ~op ~axes t_in) (Nx_cpu.reduce ~op ~axes) t_in

  let argmax ~axis ~keepdims t_in =
    unary
      (E_argmax { t_in; axis; keepdims })
      (Nx_cpu.argmax ~axis ~keepdims)
      t_in

  let argmin ~axis ~keepdims t_in =
    unary
      (E_argmin { t_in; axis; keepdims })
      (Nx_cpu.argmin ~axis ~keepdims)
      t_in

  let associative_scan ~axis ~op t_in =
    unary
      (E_associative_scan { t_in; axis; op })
      (Nx_cpu.associative_scan ~axis ~op)
      t_in

  let sort ~axis ~descending t_in =
    unary
      (E_sort { t_in; axis; descending })
      (Nx_cpu.sort ~axis ~descending)
      t_in

  let argsort ~axis ~descending t_in =
    unary
      (E_argsort { t_in; axis; descending })
      (Nx_cpu.argsort ~axis ~descending)
      t_in

  let expand t_in new_target_shape =
    moved
      (E_expand { t_in; new_target_shape })
      Nx_cpu.expand t_in new_target_shape

  let reshape t_in new_shape =
    moved (E_reshape { t_in; new_shape }) Nx_cpu.reshape t_in new_shape

  let permute t_in axes =
    moved (E_permute { t_in; axes }) Nx_cpu.permute t_in axes

  let shrink t_in limits =
    moved (E_shrink { t_in; limits }) Nx_cpu.shrink t_in limits

  let flip t_in dims_to_flip =
    moved (E_flip { t_in; dims_to_flip }) Nx_cpu.flip t_in dims_to_flip

  let sliding_window t_in ~axis ~window ~step =
    moved
      (E_sliding_window { t_in; axis; window; step })
      (fun t () -> Nx_cpu.sliding_window t ~axis ~window ~step)
      t_in ()

  let pad t_in padding_config fill_value =
    unary
      (E_pad { t_in; padding_config; fill_value })
      (fun t -> Nx_cpu.pad t padding_config fill_value)
      t_in

  let cat t_list ~axis =
    if List.for_all (function Host _ -> true | _ -> false) t_list then
      Host (Nx_cpu.cat (List.map host_of t_list) ~axis)
    else
      let r = route_of (E_cat { t_list; axis }) in
      settle r (Nx_cpu.cat (List.map host_of t_list) ~axis)

  let cast (type a b c d) ~(dtype : (c, d) Nx_dtype.t) (t_in : (a, b) t) :
      (c, d) t =
    match t_in with
    | Host t -> Host (Nx_cpu.cast ~dtype t)
    | _ ->
        routed (E_cast { t_in; target_dtype = dtype }) t_in (Nx_cpu.cast ~dtype)

  let bitcast (type a b c d) ~(dtype : (c, d) Nx_dtype.t) (t_in : (a, b) t) :
      (c, d) t =
    match t_in with
    | Host t -> Host (Nx_cpu.bitcast ~dtype t)
    | _ ->
        routed
          (E_bitcast { t_in; target_dtype = dtype })
          t_in (Nx_cpu.bitcast ~dtype)

  (* A placed value whose view covers its storage is already contiguous. *)
  let contiguous t_in =
    match t_in with
    | Host t -> Host (Nx_cpu.contiguous t)
    | Placed r when covers r -> Placed { r with r_id = fresh_id () }
    | _ -> routed (E_contiguous { t_in }) t_in Nx_cpu.contiguous

  let copy t_in = unary (E_copy { t_in }) Nx_cpu.copy t_in
  let threefry key ctr =
    binary (E_threefry { key; ctr }) Nx_cpu.threefry key ctr

  let gather data indices ~axis =
    binary
      (E_gather { data; indices; axis })
      (fun d i -> Nx_cpu.gather d i ~axis)
      data indices

  let scatter ~mode ~unique_indices data_template ~indices ~updates ~axis =
    match (data_template, indices, updates) with
    | Host d, Host i, Host u ->
        Host
          (Nx_cpu.scatter ~mode ~unique_indices d ~indices:i ~updates:u
             ~axis)
    | _ ->
        let r =
          route_of
            (E_scatter
               { data_template; indices; updates; axis; mode; unique_indices })
        in
        settle r
          (Nx_cpu.scatter ~mode ~unique_indices (host_of data_template)
             ~indices:(host_of indices) ~updates:(host_of updates) ~axis)

  let update t_in ~starts v =
    match (t_in, starts, v) with
    | Host t, Host s, Host v -> Host (Nx_cpu.update t ~starts:s v)
    | _ ->
        let r = route_of (E_update { t_in; starts; v }) in
        settle r
          (Nx_cpu.update (host_of t_in) ~starts:(host_of starts) (host_of v))

  let unfold t_in ~kernel_size ~stride ~dilation ~padding =
    unary
      (E_unfold { t_in; kernel_size; stride; dilation; padding })
      (fun t -> Nx_cpu.unfold t ~kernel_size ~stride ~dilation ~padding)
      t_in

  let fold t_in ~output_size ~kernel_size ~stride ~dilation ~padding =
    unary
      (E_fold { t_in; output_size; kernel_size; stride; dilation; padding })
      (fun t ->
        Nx_cpu.fold t ~output_size ~kernel_size ~stride ~dilation ~padding)
      t_in

  let matmul a b = binary (E_matmul { a; b }) Nx_cpu.matmul a b
  let fft t ~axes = unary (E_fft { t; axes }) (Nx_cpu.fft ~axes) t
  let ifft t ~axes = unary (E_ifft { t; axes }) (Nx_cpu.ifft ~axes) t

  let rfft (type a c) (t : (float, a) t) ~(dtype : (Complex.t, c) Nx_dtype.t)
      ~axes : (Complex.t, c) t =
    match t with
    | Host h -> Host (Nx_cpu.rfft h ~dtype ~axes)
    | _ -> routed (E_rfft { t; dtype; axes }) t (Nx_cpu.rfft ~dtype ~axes)

  let irfft (type a c) ?s (t : (Complex.t, a) t)
      ~(dtype : (float, c) Nx_dtype.t) ~axes : (float, c) t =
    match t with
    | Host h -> Host (Nx_cpu.irfft ?s h ~dtype ~axes)
    | _ ->
        routed (E_irfft { t; dtype; axes; s }) t (Nx_cpu.irfft ?s ~dtype ~axes)

  let cholesky ~upper t_in =
    unary (E_cholesky { t_in; upper }) (Nx_cpu.cholesky ~upper) t_in

  let qr ~reduced t_in =
    let r = route_of (E_qr { t_in; reduced }) in
    let q, rr = Nx_cpu.qr ~reduced (host_of t_in) in
    (settle r q, settle r rr)

  let lu t_in =
    let r = route_of (E_lu { t_in }) in
    let lu, pivots, perm = Nx_cpu.lu (host_of t_in) in
    (settle r lu, settle r pivots, settle r perm)

  let svd ~full_matrices t_in =
    let r = route_of (E_svd { t_in; full_matrices }) in
    let u, s, vt = Nx_cpu.svd ~full_matrices (host_of t_in) in
    (settle r u, settle r s, settle r vt)

  let eigvals t_in = unary (E_eigvals { t_in }) Nx_cpu.eigvals t_in

  let eig t_in =
    let r = route_of (E_eig { t_in }) in
    let vals, vecs = Nx_cpu.eig (host_of t_in) in
    (settle r vals, settle r vecs)

  let eigvalsh t_in = unary (E_eigvalsh { t_in }) Nx_cpu.eigvalsh t_in

  let eigh t_in =
    let r = route_of (E_eigh { t_in }) in
    let vals, vecs = Nx_cpu.eigh (host_of t_in) in
    (settle r vals, settle r vecs)

  let solve_triangular ~upper ~transpose ~unit_diag a b =
    binary
      (E_solve_triangular { a; b; upper; transpose; unit_diag })
      (Nx_cpu.solve_triangular ~upper ~transpose ~unit_diag)
      a b
end

(* Backends *)

module Backend = struct
  module type S = Backend_sig.S

  type t = backend

  exception Refused of string

  let () =
    Printexc.register_printer (function
      | Refused reason -> Some (Printf.sprintf "Nx.Backend.Refused(%S)" reason)
      | _ -> None)

  let name (module B : S) = B.name
  let equal : t -> t -> bool = ( == )

  module Host : S = Host_backend

  let host : t = (module Host)
  let () = host_backend := Some host
end

(* Placements *)

module Placement = struct
  type t = placement

  let v backend grid = { grid; backend }
  let grid p = p.grid
  let backend p = p.backend
  let host = { grid = Grid.device Device.host; backend = Backend.host }
  let devices = devices_of
  let memory = memory_of
  let is_host = is_host_placement
  let cuts p = Grid.cuts p.grid
  let uncut p ~axis = { p with grid = Grid.uncut p.grid ~axis }
  let map_axes f p = { p with grid = Grid.map_axes f p.grid }

  let check what backend ds =
    let fail fmt =
      Printf.ksprintf invalid_arg ("Nx.Placement.%s: " ^^ fmt) what
    in
    let rec distinct = function
      | [] -> ()
      | d :: rest ->
          if List.memq d rest then fail "%s appears twice" d.d_name;
          distinct rest
    in
    let (module B : Backend.S) = backend in
    match ds with
    | [] -> fail "no device"
    | d :: rest ->
        distinct ds;
        List.iter
          (fun d' ->
            if d'.d_memory != d.d_memory then
              fail "%s and %s have different memories" d.d_name d'.d_name)
          rest;
        List.iter
          (fun d -> if not (B.runs_on d) then fail "%s does not run on %s" B.name d.d_name)
          ds

  let device ?(backend = Backend.host) d =
    check "device" backend [ d ];
    { grid = Grid.device d; backend }

  let replicated ?(backend = Backend.host) ds =
    check "replicated" backend ds;
    { grid = Grid.v ds [ List.length ds ] []; backend }

  let sharded ?(backend = Backend.host) ~axis ds =
    if axis < 0 then
      invalid_arg (Printf.sprintf "Nx.Placement.sharded: axis %d < 0" axis);
    check "sharded" backend ds;
    { grid = Grid.v ds [ List.length ds ] [ (axis, [ 0 ]) ]; backend }

  let check_shape = check_shape
  let window = window_of

  (* The placement of a value with a new leading axis, and of one without its
     leading axis, which no cut may name. *)
  let with_leading_axis p = map_axes succ p

  let without_leading_axis p =
    if List.mem_assoc 0 (cuts p) then None else Some (map_axes pred p)

  let equal p q = p.backend == q.backend && Grid.equal ( == ) p.grid q.grid
  let pp = pp_placement
end

(* Lenses, continued *)

(* A value made beside a placed one is a full copy on each of its devices, with
   its backend. The frontend asks for a context each time it builds a constant
   beside an operand, so the host's is one value. *)
let context : type a b. (a, b) t -> context = function
  | Host _ -> Placement.host
  | Placed r when on_disk r.r_placement -> Placement.host
  | Placed r ->
      let p = r.r_placement in
      Placement.replicated ~backend:p.backend (devices_of p)
  | Traced t -> t.t_context

(* Whether a creation at [p] makes a host tensor. The frontend's contexts on the
   host are [Placement.host] itself, so the first test decides the common
   case. *)
let on_host (p : context) = p == Placement.host || is_host_placement p

let placement (type a b) (x : (a, b) t) : placement =
  try Effect.perform (E_placement x)
  with Effect.Unhandled _ -> (
    match x with
    | Host _ -> Placement.host
    | Placed r -> r.r_placement
    | Traced _ -> outside_trace ())

(* nx.cpu's storage of a host value, and the view's elements of a placed one,
   read by the backend that made its storage: readers take [contiguous] first, so the view
   is the storage. *)
let to_host (type a b) (x : (a, b) t) : Nx_device.Buffer.t =
  try Effect.perform (E_to_host x)
  with Effect.Unhandled _ -> (
    match x with
    | Host t -> Nx_cpu.to_host t
    | Placed r ->
        let (module B : Backend.S) = r.r_cell.placement.backend in
        B.to_host x
    | Traced _ -> outside_trace ())

(* Moving. The target placement's backend makes the value there. *)

let move (type a b) p (x : (a, b) t) : (a, b) t =
  let move () =
    match x with
    | Traced _ -> outside_trace ()
    | Placed r when is_host_placement p -> Host (read_host r)
    | Host _ | Placed _ ->
        check_shape "Nx.place" p (View.shape (view x));
        let (module B : Backend.S) = p.backend in
        B.place p x
  in
  match x with
  | Placed r -> Cell.with_borrow r.r_cell move
  | Host _ | Traced _ -> move ()

(* A value already at [p] is returned without an effect: an effect's result is
   always a fresh value, which the transformations take for a new node. *)
let place (type a b) p (x : (a, b) t) : (a, b) t =
  if Placement.equal (placement x) p then
    match x with
    | Placed r -> Cell.with_borrow r.r_cell (fun () -> x)
    | Host _ | Traced _ -> x
  else
    try Effect.perform (E_place { placement = p; t_in = x })
    with Effect.Unhandled _ -> move p x

(* Dispatch

   Every operation first performs its effect. Unhandled, operands all on the
   host run nx.cpu directly, and any other operands run on the backend their
   placed ones share: operands with two backends raise. *)

let backend_among op xs =
  let p =
    List.fold_left
      (fun acc (P x) ->
        match (x, acc) with
        | Host _, _ -> acc
        | Traced _, _ -> outside_trace ()
        | Placed r, _ when on_disk r.r_placement -> acc
        | Placed r, None -> Some r.r_placement
        | Placed r, Some p ->
            if r.r_placement.backend == p.backend then acc
            else mixed op p r.r_placement)
      None xs
  in
  match p with Some p -> p.backend | None -> Backend.host

(* The backend that runs the operation performing [e]. *)
let backend_of e =
  match routing e with
  | Some (op, _, xs) -> backend_among op xs
  | None -> (
      match movement_of e with
      | Some (x, _) -> backend_among "move" [ x ]
      | None -> invalid_arg "Nx_effect.backend_of: no operation performs this")

let unary_op e host_op pick t_in =
  try Effect.perform e
  with Effect.Unhandled _ -> (
    match t_in with
    | Host t -> Host (host_op t)
    | _ -> pick (backend_of e) t_in)

let binary_op e host_op pick a b =
  try Effect.perform e
  with Effect.Unhandled _ -> (
    match (a, b) with
    | Host a, Host b -> Host (host_op a b)
    | _ -> pick (backend_of e) a b)

let movement_op e host_op pick t_in arg =
  try Effect.perform e
  with Effect.Unhandled _ -> (
    match t_in with
    | Host t -> Host (host_op t arg)
    | _ -> pick (backend_of e) t_in arg)

(* Binary operations *)

let add a b =
  binary_op (E_add { a; b }) Nx_cpu.add
    (fun (module B : Backend.S) -> B.add) a b

let sub a b =
  binary_op (E_sub { a; b }) Nx_cpu.sub
    (fun (module B : Backend.S) -> B.sub) a b

let mul a b =
  binary_op (E_mul { a; b }) Nx_cpu.mul
    (fun (module B : Backend.S) -> B.mul) a b

let max a b =
  binary_op (E_max { a; b }) Nx_cpu.max
    (fun (module B : Backend.S) -> B.max) a b

let min a b =
  binary_op (E_min { a; b }) Nx_cpu.min
    (fun (module B : Backend.S) -> B.min) a b

let mod_ a b =
  binary_op (E_mod { a; b }) Nx_cpu.mod_
    (fun (module B : Backend.S) -> B.mod_) a b

let pow a b =
  binary_op (E_pow { a; b }) Nx_cpu.pow
    (fun (module B : Backend.S) -> B.pow) a b

let xor a b =
  binary_op (E_xor { a; b }) Nx_cpu.xor
    (fun (module B : Backend.S) -> B.xor) a b

let or_ a b =
  binary_op (E_or { a; b }) Nx_cpu.or_
    (fun (module B : Backend.S) -> B.or_) a b

let and_ a b =
  binary_op (E_and { a; b }) Nx_cpu.and_
    (fun (module B : Backend.S) -> B.and_) a b

let atan2 a b =
  binary_op (E_atan2 { a; b }) Nx_cpu.atan2
    (fun (module B : Backend.S) -> B.atan2) a b

let fdiv a b =
  binary_op (E_fdiv { a; b }) Nx_cpu.fdiv
    (fun (module B : Backend.S) -> B.fdiv) a b

let idiv a b =
  binary_op (E_idiv { a; b }) Nx_cpu.idiv
    (fun (module B : Backend.S) -> B.idiv) a b

(* Comparison operations *)

let cmpeq a b =
  binary_op (E_cmpeq { a; b }) Nx_cpu.cmpeq
    (fun (module B : Backend.S) -> B.cmpeq) a b

let cmpne a b =
  binary_op (E_cmpne { a; b }) Nx_cpu.cmpne
    (fun (module B : Backend.S) -> B.cmpne) a b

let cmplt a b =
  binary_op (E_cmplt { a; b }) Nx_cpu.cmplt
    (fun (module B : Backend.S) -> B.cmplt) a b

let cmple a b =
  binary_op (E_cmple { a; b }) Nx_cpu.cmple
    (fun (module B : Backend.S) -> B.cmple) a b

(* Unary operations *)

let neg t =
  unary_op (E_neg { t_in = t }) Nx_cpu.neg
    (fun (module B : Backend.S) -> B.neg) t

let sin t =
  unary_op (E_sin { t_in = t }) Nx_cpu.sin
    (fun (module B : Backend.S) -> B.sin) t

let sqrt t =
  unary_op (E_sqrt { t_in = t }) Nx_cpu.sqrt
    (fun (module B : Backend.S) -> B.sqrt) t

let recip t =
  unary_op (E_recip { t_in = t }) Nx_cpu.recip
    (fun (module B : Backend.S) -> B.recip) t

let log t =
  unary_op (E_log { t_in = t }) Nx_cpu.log
    (fun (module B : Backend.S) -> B.log) t

let exp t =
  unary_op (E_exp { t_in = t }) Nx_cpu.exp
    (fun (module B : Backend.S) -> B.exp) t

let cos t =
  unary_op (E_cos { t_in = t }) Nx_cpu.cos
    (fun (module B : Backend.S) -> B.cos) t

let abs t =
  unary_op (E_abs { t_in = t }) Nx_cpu.abs
    (fun (module B : Backend.S) -> B.abs) t

let sign t =
  unary_op (E_sign { t_in = t }) Nx_cpu.sign
    (fun (module B : Backend.S) -> B.sign) t

let tan t =
  unary_op (E_tan { t_in = t }) Nx_cpu.tan
    (fun (module B : Backend.S) -> B.tan) t

let asin t =
  unary_op (E_asin { t_in = t }) Nx_cpu.asin
    (fun (module B : Backend.S) -> B.asin) t

let acos t =
  unary_op (E_acos { t_in = t }) Nx_cpu.acos
    (fun (module B : Backend.S) -> B.acos) t

let atan t =
  unary_op (E_atan { t_in = t }) Nx_cpu.atan
    (fun (module B : Backend.S) -> B.atan) t

let sinh t =
  unary_op (E_sinh { t_in = t }) Nx_cpu.sinh
    (fun (module B : Backend.S) -> B.sinh) t

let cosh t =
  unary_op (E_cosh { t_in = t }) Nx_cpu.cosh
    (fun (module B : Backend.S) -> B.cosh) t

let tanh t =
  unary_op (E_tanh { t_in = t }) Nx_cpu.tanh
    (fun (module B : Backend.S) -> B.tanh) t

let trunc t =
  unary_op (E_trunc { t_in = t }) Nx_cpu.trunc
    (fun (module B : Backend.S) -> B.trunc) t

let ceil t =
  unary_op (E_ceil { t_in = t }) Nx_cpu.ceil
    (fun (module B : Backend.S) -> B.ceil) t

let floor t =
  unary_op (E_floor { t_in = t }) Nx_cpu.floor
    (fun (module B : Backend.S) -> B.floor) t

let round t =
  unary_op (E_round { t_in = t }) Nx_cpu.round
    (fun (module B : Backend.S) -> B.round) t

let erf t =
  unary_op (E_erf { t_in = t }) Nx_cpu.erf
    (fun (module B : Backend.S) -> B.erf) t

(* Reduction operations. The host case of each operation below calls nx.cpu
   directly, allocating nothing beyond the effect and the result. *)

let reduce ~op ~axes t_in =
  unary_op
    (reduce_effect ~op ~axes t_in)
    (Nx_cpu.reduce ~op ~axes)
    (fun (module B : Backend.S) -> B.reduce ~op ~axes)
    t_in

let argmax ~axis ~keepdims t_in =
  unary_op
    (E_argmax { t_in; axis; keepdims })
    (Nx_cpu.argmax ~axis ~keepdims)
    (fun (module B : Backend.S) -> B.argmax ~axis ~keepdims)
    t_in

let argmin ~axis ~keepdims t_in =
  unary_op
    (E_argmin { t_in; axis; keepdims })
    (Nx_cpu.argmin ~axis ~keepdims)
    (fun (module B : Backend.S) -> B.argmin ~axis ~keepdims)
    t_in

let associative_scan ~axis ~op t_in =
  unary_op
    (E_associative_scan { t_in; axis; op })
    (Nx_cpu.associative_scan ~axis ~op)
    (fun (module B : Backend.S) -> B.associative_scan ~axis ~op)
    t_in

let sort ~axis ~descending t_in =
  unary_op
    (E_sort { t_in; axis; descending })
    (Nx_cpu.sort ~axis ~descending)
    (fun (module B : Backend.S) -> B.sort ~axis ~descending)
    t_in

let argsort ~axis ~descending t_in =
  unary_op
    (E_argsort { t_in; axis; descending })
    (Nx_cpu.argsort ~axis ~descending)
    (fun (module B : Backend.S) -> B.argsort ~axis ~descending)
    t_in

(* Movement operations *)

let reshape t_in new_shape =
  movement_op
    (E_reshape { t_in; new_shape })
    Nx_cpu.reshape
    (fun (module B : Backend.S) -> B.reshape)
    t_in new_shape

let expand t_in new_target_shape =
  movement_op
    (E_expand { t_in; new_target_shape })
    Nx_cpu.expand
    (fun (module B : Backend.S) -> B.expand)
    t_in new_target_shape

let permute t_in axes =
  movement_op
    (E_permute { t_in; axes })
    Nx_cpu.permute
    (fun (module B : Backend.S) -> B.permute)
    t_in axes

let shrink t_in limits =
  movement_op
    (E_shrink { t_in; limits })
    Nx_cpu.shrink
    (fun (module B : Backend.S) -> B.shrink)
    t_in limits

let flip t_in dims_to_flip =
  movement_op
    (E_flip { t_in; dims_to_flip })
    Nx_cpu.flip
    (fun (module B : Backend.S) -> B.flip)
    t_in dims_to_flip

let sliding_window t_in ~axis ~window ~step =
  movement_op
    (E_sliding_window { t_in; axis; window; step })
    (fun t () -> Nx_cpu.sliding_window t ~axis ~window ~step)
    (fun (module B : Backend.S) t () -> B.sliding_window t ~axis ~window ~step)
    t_in ()

let pad t_in padding_config fill_value =
  unary_op
    (E_pad { t_in; padding_config; fill_value })
    (fun t -> Nx_cpu.pad t padding_config fill_value)
    (fun (module B : Backend.S) t -> B.pad t padding_config fill_value)
    t_in

(* Copy operations *)

let contiguous t_in =
  unary_op (E_contiguous { t_in }) Nx_cpu.contiguous
    (fun (module B : Backend.S) -> B.contiguous) t_in

let copy t_in =
  unary_op (E_copy { t_in }) Nx_cpu.copy
    (fun (module B : Backend.S) -> B.copy) t_in

(* Creation. A constant is not an operation: a filled value is one element on
   the host, placed where it is made and expanded. One of more than one element
   is then copied into storage of its own, so that its view covers its storage
   and a compiled call can consume it. *)

let broadcast scalar shape_arr =
  if Array.length shape_arr = 0 then scalar
  else
    let ones = Array.map (fun _ -> 1) shape_arr in
    let x = reshape scalar ones in
    if Shape.equal ones shape_arr then x else expand x shape_arr

let full (ctx : context) dtype shape_arr value =
  let e = Host (Nx_cpu.full () dtype [||] value) in
  if on_host ctx then
    let x = broadcast e shape_arr in
    if Array.fold_left ( * ) 1 shape_arr <= 1 then x else copy x
  else
    let copies =
      List.fold_left
        (fun p (axis, _) -> Placement.uncut p ~axis)
        ctx (Placement.cuts ctx)
    in
    let x = broadcast (place copies e) shape_arr in
    if Array.fold_left ( * ) 1 shape_arr <= 1 then x
    else if copies == ctx then copy x
    else place ctx x

let from_host (ctx : context) dtype buffer =
  check_host "from_host" dtype buffer;
  let x = Host (Nx_cpu.from_host () dtype buffer) in
  if on_host ctx then x else place ctx x

(* The host buffer of exactly [x]'s elements in C order: its storage when it is
   contiguous on the host. A placed value's read is its view's elements. *)
let elements x =
  let x = match x with Placed _ -> x | Host _ | Traced _ -> contiguous x in
  let b = to_host x in
  Nx_device.Buffer.view b ~offset:0 (Nx_device.Buffer.dtype b)
    (View.numel (view x))

(* The buffer of exactly [x]'s elements in C order, without a copy, when there
   is one: [x] is on the host, or its storage is one runtime buffer, and its
   view is a contiguous run of its storage. *)
let run (type a b) (x : (a, b) t) =
  match x with
  | Host t -> run_in t.buffer t.view
  | Placed r -> (
      match Cell.state r.r_cell with
      | Live (Runtime [ b ]) -> run_in b r.r_view
      | _ -> None)
  | Traced _ -> outside_trace ()

(* [of_buffer dtype shape b] is the value of shape [shape] whose elements, of
   [dtype], are [b]'s in C order, without a copy: a host value for a buffer of
   the host, and on [b]'s device otherwise. *)
let of_buffer (type a b) (dtype : (a, b) Nx_dtype.t) shape b : (a, b) t =
  let n = Nx_device.Buffer.length b in
  if Array.fold_left ( * ) 1 shape <> n then
    invalid_arg
      (Printf.sprintf "Nx_effect.of_buffer: shape %s for %d elements"
         (Shape.to_string shape) n);
  let s = Nx_device.Buffer.dtype b in
  if not (Nx_dtype.Scalar.equal s (Nx_dtype.Scalar.of_dtype dtype)) then
    invalid_arg
      (Printf.sprintf "Nx_effect.of_buffer: a %s buffer read as %s"
         (Nx_dtype.Scalar.to_string s)
         (Nx_dtype.to_string dtype));
  let d = Nx_device.Buffer.device b in
  if Nx_device.equal d Nx_device.host then
    reshape (from_host Placement.host dtype b) shape
  else
    let p = Placement.device (Device.of_runtime d) in
    placed p dtype (View.create shape)
      (cell ~placement:p ~length:n (Runtime [ b ]))

(* Ternary operations *)

let where condition if_true if_false =
  let e = E_where { condition; if_true; if_false } in
  try Effect.perform e
  with Effect.Unhandled _ -> (
    match (condition, if_true, if_false) with
    | Host c, Host a, Host b -> Host (Nx_cpu.where c a b)
    | _ ->
        let (module B : Backend.S) = backend_of e in
        B.where condition if_true if_false)

(* Cat *)

let cat t_list ~axis =
  let e = E_cat { t_list; axis } in
  try Effect.perform e
  with Effect.Unhandled _ ->
    if List.for_all (function Host _ -> true | _ -> false) t_list then
      Host (Nx_cpu.cat (List.map host_of t_list) ~axis)
    else
      let (module B : Backend.S) = backend_of e in
      B.cat t_list ~axis

(* Cast *)

let cast (type a b c d) ~(dtype : (c, d) Nx_dtype.t) (t_in : (a, b) t) :
    (c, d) t =
  unary_op
    (E_cast { t_in; target_dtype = dtype })
    (Nx_cpu.cast ~dtype)
    (fun (module B : Backend.S) -> B.cast ~dtype)
    t_in

let bitcast (type a b c d) ~(dtype : (c, d) Nx_dtype.t) (t_in : (a, b) t) :
    (c, d) t =
  unary_op
    (E_bitcast { t_in; target_dtype = dtype })
    (Nx_cpu.bitcast ~dtype)
    (fun (module B : Backend.S) -> B.bitcast ~dtype)
    t_in

(* Indexed access *)

let gather data indices ~axis =
  binary_op
    (E_gather { data; indices; axis })
    (fun d i -> Nx_cpu.gather d i ~axis)
    (fun (module B : Backend.S) d i -> B.gather d i ~axis)
    data indices

let update t_in ~starts v =
  let e = E_update { t_in; starts; v } in
  try Effect.perform e
  with Effect.Unhandled _ -> (
    match (t_in, starts, v) with
    | Host t, Host s, Host v -> Host (Nx_cpu.update t ~starts:s v)
    | _ ->
        let (module B : Backend.S) = backend_of e in
        B.update t_in ~starts v)

let scatter ~mode ~unique_indices data_template ~indices ~updates ~axis =
  let e =
    E_scatter { data_template; indices; updates; axis; mode; unique_indices }
  in
  try Effect.perform e
  with Effect.Unhandled _ -> (
    match (data_template, indices, updates) with
    | Host d, Host i, Host u ->
        Host
          (Nx_cpu.scatter ~mode ~unique_indices d ~indices:i ~updates:u
             ~axis)
    | _ ->
        let (module B : Backend.S) = backend_of e in
        B.scatter ~mode ~unique_indices data_template ~indices ~updates ~axis)

(* Random *)

let threefry key ctr =
  binary_op (E_threefry { key; ctr }) Nx_cpu.threefry
    (fun (module B : Backend.S) -> B.threefry) key ctr

(* Window operations *)

let unfold t_in ~kernel_size ~stride ~dilation ~padding =
  unary_op
    (E_unfold { t_in; kernel_size; stride; dilation; padding })
    (fun t -> Nx_cpu.unfold t ~kernel_size ~stride ~dilation ~padding)
    (fun (module B : Backend.S) t ->
      B.unfold t ~kernel_size ~stride ~dilation ~padding)
    t_in

let fold t_in ~output_size ~kernel_size ~stride ~dilation ~padding =
  unary_op
    (E_fold { t_in; output_size; kernel_size; stride; dilation; padding })
    (fun t ->
      Nx_cpu.fold t ~output_size ~kernel_size ~stride ~dilation ~padding)
    (fun (module B : Backend.S) t ->
      B.fold t ~output_size ~kernel_size ~stride ~dilation ~padding)
    t_in

(* Matrix operations *)

let matmul a b =
  binary_op (E_matmul { a; b }) Nx_cpu.matmul
    (fun (module B : Backend.S) -> B.matmul) a b

(* FFT operations *)

let fft t ~axes =
  unary_op (E_fft { t; axes }) (Nx_cpu.fft ~axes)
    (fun (module B : Backend.S) -> B.fft ~axes) t

let ifft t ~axes =
  unary_op (E_ifft { t; axes }) (Nx_cpu.ifft ~axes)
    (fun (module B : Backend.S) -> B.ifft ~axes) t

let rfft (type a c) (t : (float, a) t) ~(dtype : (Complex.t, c) Nx_dtype.t)
    ~axes : (Complex.t, c) t =
  unary_op
    (E_rfft { t; dtype; axes })
    (Nx_cpu.rfft ~dtype ~axes)
    (fun (module B : Backend.S) -> B.rfft ~dtype ~axes)
    t

let irfft (type a c) ?s (t : (Complex.t, a) t) ~(dtype : (float, c) Nx_dtype.t)
    ~axes : (float, c) t =
  unary_op
    (E_irfft { t; dtype; axes; s })
    (Nx_cpu.irfft ?s ~dtype ~axes)
    (fun (module B : Backend.S) -> B.irfft ?s ~dtype ~axes)
    t

(* Linear algebra. The decompositions run on the backend even on the host:
   their host case settles its results as a placed call's does. *)

let cholesky ~upper t_in =
  unary_op
    (E_cholesky { t_in; upper })
    (Nx_cpu.cholesky ~upper)
    (fun (module B : Backend.S) -> B.cholesky ~upper)
    t_in

let qr ~reduced t_in =
  let e = E_qr { t_in; reduced } in
  try Effect.perform e
  with Effect.Unhandled _ ->
    let (module B : Backend.S) = backend_of e in
    B.qr ~reduced t_in

let lu t_in =
  let e = E_lu { t_in } in
  try Effect.perform e
  with Effect.Unhandled _ ->
    let (module B : Backend.S) = backend_of e in
    B.lu t_in

let svd ~full_matrices t_in =
  let e = E_svd { t_in; full_matrices } in
  try Effect.perform e
  with Effect.Unhandled _ ->
    let (module B : Backend.S) = backend_of e in
    B.svd ~full_matrices t_in

let eigvals t_in =
  unary_op (E_eigvals { t_in }) Nx_cpu.eigvals
    (fun (module B : Backend.S) -> B.eigvals) t_in

let eig t_in =
  let e = E_eig { t_in } in
  try Effect.perform e
  with Effect.Unhandled _ ->
    let (module B : Backend.S) = backend_of e in
    B.eig t_in

let eigvalsh t_in =
  unary_op (E_eigvalsh { t_in }) Nx_cpu.eigvalsh
    (fun (module B : Backend.S) -> B.eigvalsh) t_in

let eigh t_in =
  let e = E_eigh { t_in } in
  try Effect.perform e
  with Effect.Unhandled _ ->
    let (module B : Backend.S) = backend_of e in
    B.eigh t_in

let solve_triangular ~upper ~transpose ~unit_diag a b =
  binary_op
    (E_solve_triangular { a; b; upper; transpose; unit_diag })
    (Nx_cpu.solve_triangular ~upper ~transpose ~unit_diag)
    (fun (module B : Backend.S) -> B.solve_triangular ~upper ~transpose ~unit_diag)
    a b
