(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Nx_array

(* Types

   GADT constructors (the operations of [Op.t] below) require that type
   variables in the payload be deducible from the return type. A transparent
   alias of [Nx_cpu.t] would not be injective, so the tensor is a GADT of its
   own, whose parameters the return type determines.

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
  | E_placement : ('a, 'b) t -> placement Effect.t

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

(* Operations

   Every operation nx performs is a constructor of [Op.t]: the computing ones,
   which the placement's backend answers, and the movements, placing and
   reading, which nx answers itself. A kind names the function among the
   operations of one constructor. *)

module Op = struct
  type move =
    | Reshape of int array
    | Expand of int array
    | Permute of int array
    | Shrink of (int * int) array
    | Flip of bool array
    | Window of { axis : int; size : int; step : int }

  type int32_t = (int32, Nx_dtype.int32_elt) Types.t

  type _ t =
    | Unary : Nx_backend.unary * ('a, 'b) Types.t -> ('a, 'b) Types.t t
    | Binary :
        Nx_backend.binary * ('a, 'b) Types.t * ('a, 'b) Types.t
        -> ('a, 'b) Types.t t
    | Compare :
        Nx_backend.compare * ('a, 'b) Types.t * ('a, 'b) Types.t
        -> (bool, Nx_dtype.bool_elt) Types.t t
    | Where :
        (bool, Nx_dtype.bool_elt) Types.t * ('a, 'b) Types.t * ('a, 'b) Types.t
        -> ('a, 'b) Types.t t
    | Reduce :
        Nx_backend.reduce * int array * ('a, 'b) Types.t
        -> ('a, 'b) Types.t t
    | Scan : Nx_backend.reduce * int * ('a, 'b) Types.t -> ('a, 'b) Types.t t
    | Arg_reduce :
        Nx_backend.arg_reduce * int * ('a, 'b) Types.t
        -> int32_t t
    | Sort : {
        descending : bool;
        axis : int;
        x : ('a, 'b) Types.t;
      }
        -> ('a, 'b) Types.t t
    | Argsort : {
        descending : bool;
        axis : int;
        x : ('a, 'b) Types.t;
      }
        -> int32_t t
    | Pad : (int * int) array * 'a * ('a, 'b) Types.t -> ('a, 'b) Types.t t
    | Cat : int * ('a, 'b) Types.t list -> ('a, 'b) Types.t t
    | Convert :
        Nx_backend.conversion * ('c, 'd) Nx_dtype.t * ('a, 'b) Types.t
        -> ('c, 'd) Types.t t
    | Threefry : int32_t * int32_t -> int32_t t
    | Gather : int * int32_t * ('a, 'b) Types.t -> ('a, 'b) Types.t t
    | Scatter : {
        mode : [ `Set | `Add ];
        unique : bool;
        axis : int;
        indices : int32_t;
        updates : ('a, 'b) Types.t;
        into : ('a, 'b) Types.t;
      }
        -> ('a, 'b) Types.t t
    | Update :
        ('a, 'b) Types.t * int32_t * ('a, 'b) Types.t
        -> ('a, 'b) Types.t t
    | Unfold : {
        kernel_size : int array;
        stride : int array;
        dilation : int array;
        padding : (int * int) array;
        x : ('a, 'b) Types.t;
      }
        -> ('a, 'b) Types.t t
    | Fold : {
        output_size : int array;
        kernel_size : int array;
        stride : int array;
        dilation : int array;
        padding : (int * int) array;
        x : ('a, 'b) Types.t;
      }
        -> ('a, 'b) Types.t t
    | Matmul : ('a, 'b) Types.t * ('a, 'b) Types.t -> ('a, 'b) Types.t t
    | Fft : {
        inverse : bool;
        axes : int array;
        x : (Complex.t, 'b) Types.t;
      }
        -> (Complex.t, 'b) Types.t t
    | Rfft : {
        dtype : (Complex.t, 'c) Nx_dtype.t;
        axes : int array;
        x : (float, 'b) Types.t;
      }
        -> (Complex.t, 'c) Types.t t
    | Irfft : {
        dtype : (float, 'c) Nx_dtype.t;
        axes : int array;
        s : int array option;
        x : (Complex.t, 'b) Types.t;
      }
        -> (float, 'c) Types.t t
    | Contiguous : ('a, 'b) Types.t -> ('a, 'b) Types.t t
    | Cholesky : { upper : bool; x : ('a, 'b) Types.t } -> ('a, 'b) Types.t t
    | Qr : {
        reduced : bool;
        x : ('a, 'b) Types.t;
      }
        -> (('a, 'b) Types.t * ('a, 'b) Types.t) t
    | Lu : ('a, 'b) Types.t -> (('a, 'b) Types.t * int32_t * int32_t) t
    | Svd : {
        full_matrices : bool;
        x : ('a, 'b) Types.t;
      }
        -> (('a, 'b) Types.t
           * (float, Nx_dtype.float64_elt) Types.t
           * ('a, 'b) Types.t)
           t
    | Eig : {
        vectors : bool;
        x : ('a, 'b) Types.t;
      }
        -> ((Complex.t, Nx_dtype.complex64_elt) Types.t
           * (Complex.t, Nx_dtype.complex64_elt) Types.t option)
           t
    | Eigh : {
        vectors : bool;
        x : ('a, 'b) Types.t;
      }
        -> ((float, Nx_dtype.float64_elt) Types.t * ('a, 'b) Types.t option) t
    | Solve_triangular : {
        upper : bool;
        transpose : bool;
        unit_diag : bool;
        a : ('a, 'b) Types.t;
        b : ('a, 'b) Types.t;
      }
        -> ('a, 'b) Types.t t
    | Move : ('a, 'b) Types.t * move -> ('a, 'b) Types.t t
    | Place : placement * ('a, 'b) Types.t -> ('a, 'b) Types.t t
    | Read : ('a, 'b) Types.t -> Nx_device.Buffer.t t

  let name : type r. r t -> string =
   fun op ->
    match[@warning "@4@8"] op with
    | Unary (k, _) -> (
        match k with
        | Neg -> "neg"
        | Recip -> "recip"
        | Abs -> "abs"
        | Sqrt -> "sqrt"
        | Sign -> "sign"
        | Exp -> "exp"
        | Log -> "log"
        | Sin -> "sin"
        | Cos -> "cos"
        | Tan -> "tan"
        | Asin -> "asin"
        | Acos -> "acos"
        | Atan -> "atan"
        | Sinh -> "sinh"
        | Cosh -> "cosh"
        | Tanh -> "tanh"
        | Trunc -> "trunc"
        | Ceil -> "ceil"
        | Floor -> "floor"
        | Round -> "round"
        | Erf -> "erf")
    | Binary (k, _, _) -> (
        match k with
        | Add -> "add"
        | Sub -> "sub"
        | Mul -> "mul"
        | Fdiv | Idiv -> "div"
        | Mod -> "mod"
        | Pow -> "pow"
        | Atan2 -> "atan2"
        | Maximum -> "maximum"
        | Minimum -> "minimum"
        | And -> "bitwise_and"
        | Or -> "bitwise_or"
        | Xor -> "bitwise_xor")
    | Compare (k, _, _) -> (
        match k with
        | Equal -> "equal"
        | Not_equal -> "not_equal"
        | Less -> "less"
        | Less_equal -> "less_equal")
    | Where _ -> "where"
    | Reduce (k, _, _) -> (
        match k with
        | Sum -> "sum"
        | Prod -> "prod"
        | Max -> "max"
        | Min -> "min")
    | Scan (k, _, _) -> (
        match k with
        | Sum -> "cumsum"
        | Prod -> "cumprod"
        | Max -> "cummax"
        | Min -> "cummin")
    | Arg_reduce (k, _, _) -> (
        match k with Argmax -> "argmax" | Argmin -> "argmin")
    | Sort _ -> "sort"
    | Argsort _ -> "argsort"
    | Pad _ -> "pad"
    | Cat _ -> "concatenate"
    | Convert (k, _, _) -> ( match k with Cast -> "cast" | Bitcast -> "bitcast")
    | Threefry _ -> "threefry"
    | Gather _ -> "take_along_axis"
    | Scatter _ -> "scatter"
    | Update _ -> "set"
    | Unfold _ -> "unfold"
    | Fold _ -> "fold"
    | Matmul _ -> "matmul"
    | Fft { inverse; _ } -> if inverse then "ifft" else "fft"
    | Rfft _ -> "rfft"
    | Irfft _ -> "irfft"
    | Contiguous _ -> "contiguous"
    | Cholesky _ -> "cholesky"
    | Qr _ -> "qr"
    | Lu _ -> "lu"
    | Svd _ -> "svd"
    | Eig { vectors; _ } -> if vectors then "eig" else "eigvals"
    | Eigh { vectors; _ } -> if vectors then "eigh" else "eigvalsh"
    | Solve_triangular _ -> "solve_triangular"
    | Move (_, m) -> (
        match m with
        | Reshape _ -> "reshape"
        | Expand _ -> "expand"
        | Permute _ -> "permute"
        | Shrink _ -> "shrink"
        | Flip _ -> "flip"
        | Window _ -> "sliding_window")
    | Place _ -> "place"
    | Read _ -> "read"

  let operands : type r. r t -> packed list =
   fun op ->
    match[@warning "@4@8"] op with
    | Unary (_, x) -> [ P x ]
    | Binary (_, a, b) -> [ P a; P b ]
    | Compare (_, a, b) -> [ P a; P b ]
    | Where (c, a, b) -> [ P c; P a; P b ]
    | Reduce (_, _, x) -> [ P x ]
    | Scan (_, _, x) -> [ P x ]
    | Arg_reduce (_, _, x) -> [ P x ]
    | Sort { x; _ } -> [ P x ]
    | Argsort { x; _ } -> [ P x ]
    | Pad (_, _, x) -> [ P x ]
    | Cat (_, xs) -> List.map (fun x -> P x) xs
    | Convert (_, _, x) -> [ P x ]
    | Threefry (key, ctr) -> [ P key; P ctr ]
    | Gather (_, indices, x) -> [ P x; P indices ]
    | Scatter { indices; updates; into; _ } -> [ P into; P indices; P updates ]
    | Update (x, starts, v) -> [ P x; P starts; P v ]
    | Unfold { x; _ } -> [ P x ]
    | Fold { x; _ } -> [ P x ]
    | Matmul (a, b) -> [ P a; P b ]
    | Fft { x; _ } -> [ P x ]
    | Rfft { x; _ } -> [ P x ]
    | Irfft { x; _ } -> [ P x ]
    | Contiguous x -> [ P x ]
    | Cholesky { x; _ } -> [ P x ]
    | Qr { x; _ } -> [ P x ]
    | Lu x -> [ P x ]
    | Svd { x; _ } -> [ P x ]
    | Eig { x; _ } -> [ P x ]
    | Eigh { x; _ } -> [ P x ]
    | Solve_triangular { a; b; _ } -> [ P a; P b ]
    | Move (x, _) -> [ P x ]
    | Place (_, x) -> [ P x ]
    | Read x -> [ P x ]

  let pp ppf op =
    let operand ppf (P x) =
      Format.fprintf ppf "%s%s" (Nx_dtype.to_string (dtype x))
        (Shape.to_string (View.shape (view x)))
    in
    let space ppf () = Format.pp_print_char ppf ' ' in
    Format.fprintf ppf "%s %a" (name op)
      (Format.pp_print_list ~pp_sep:space operand)
      (operands op)
end

type move = Op.move =
  | Reshape of int array
  | Expand of int array
  | Permute of int array
  | Shrink of (int * int) array
  | Flip of bool array
  | Window of { axis : int; size : int; step : int }

type _ Effect.t += E_op : 'r Op.t -> 'r Effect.t

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

(* How an operation is routed: a computation by a rule over its operands, a
   movement of one operand, placing and reading. Eager routing and a compiler
   checking where values live read the same rule from it. *)
type routing =
  | Computes of rule * packed list
  | Moves of packed * move
  | Places of placement
  | Reads of packed

let routing : type r. r Op.t -> routing =
 fun op ->
  let computes rule = Computes (rule, Op.operands op) in
  let along_axes axes = computes (Along axes) in
  match[@warning "@4@8"] op with
  | Unary _ | Binary _ | Compare _ | Where _ | Convert _ | Threefry _
  | Contiguous _ ->
      computes Elementwise
  | Reduce (_, axes, _) -> computes (Reduce { axes; keepdims = false })
  | Arg_reduce (_, axis, _) ->
      computes (Reduce { axes = [| axis |]; keepdims = false })
  | Scan (_, axis, _) -> along_axes [ axis ]
  | Sort { axis; _ } -> along_axes [ axis ]
  | Argsort { axis; _ } -> along_axes [ axis ]
  | Pad (padding, _, _) -> along_axes (padded padding)
  | Cat (axis, _) -> along_axes [ axis ]
  | Gather (axis, _, _) -> computes (Gather axis)
  | Scatter _ | Update _ -> computes Into
  | Unfold { x; _ } -> along_axes (spatial x)
  | Fold { x; _ } -> along_axes (spatial x)
  | Matmul _ -> computes Contract
  | Fft { axes; _ } -> along_axes (Array.to_list axes)
  | Rfft { axes; _ } -> along_axes (Array.to_list axes)
  | Irfft { axes; _ } -> along_axes (Array.to_list axes)
  | Cholesky { x; _ } -> along_axes (matrix_axes x)
  | Qr { x; _ } -> along_axes (matrix_axes x)
  | Lu x -> along_axes (matrix_axes x)
  | Svd { x; _ } -> along_axes (matrix_axes x)
  | Eig { x; _ } -> along_axes (matrix_axes x)
  | Eigh { x; _ } -> along_axes (matrix_axes x)
  | Solve_triangular { a; _ } -> along_axes (matrix_axes a)
  | Move (x, m) -> Moves (P x, m)
  | Place (p, _) -> Places p
  | Read x -> Reads (P x)

(* Where a computing operation runs. *)
let route_of : type r. r Op.t -> route =
 fun op ->
  match routing op with
  | Computes (rule, xs) -> route (Op.name op) rule xs
  | Moves _ | Places _ | Reads _ ->
      invalid_arg "Nx_effect.route_of: the operation computes nothing"

let settle : type a b. route -> (a, b) Nx_cpu.t -> (a, b) t =
 fun r h ->
  match r with
  | On_host -> Host h
  | At p -> (memory_of p).place p (Host h)

(* [routed op x f] runs [f] over [x], no host tensor, where [op] runs. *)
let routed op x f =
  let r = route_of op in
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

let move_view v = function
  | Reshape shape -> View.reshape v shape
  | Expand shape -> View.expand v shape
  | Permute order -> View.permute v order
  | Shrink limits -> View.shrink v limits
  | Flip dims -> View.flip v dims
  | Window { axis; size; step } ->
      View.sliding_window v ~axis ~window:size ~step

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
  | Window { axis = a; _ } -> if a = axis then across "window" else Split axis

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
  | Permute _ | Flip _ | Window _ -> m

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
let routed2 op a b f =
  let r = route_of op in
  settle r (f (host_of a) (host_of b))
let unary op f x = match x with Host t -> Host (f t) | _ -> routed op x f

let binary op f a b =
  match (a, b) with
  | Host a, Host b -> Host (f a b)
  | _ -> routed2 op a b f

(* [moved x m] is [x] moved by [m]: view arithmetic over the same storage. *)
let moved (type a b) (x : (a, b) t) m : (a, b) t =
  match x with
  | Host t -> Host { t with view = move_view t.view m }
  | Placed r ->
      let r_placement, r_view = split_view r.r_placement r.r_view m in
      Placed { r with r_id = fresh_id (); r_placement; r_view }
  | Traced _ -> outside_trace ()

let create p make =
  if is_host_placement p then Host (make ()) else settle (At p) (make ())

let reduce_kind : [ `Sum | `Prod | `Max | `Min ] -> Nx_backend.reduce = function
  | `Sum -> Sum
  | `Prod -> Prod
  | `Max -> Max
  | `Min -> Min

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

  let add a b = binary (Op.Binary (Add, a, b)) Nx_cpu.add a b
  let sub a b = binary (Op.Binary (Sub, a, b)) Nx_cpu.sub a b
  let mul a b = binary (Op.Binary (Mul, a, b)) Nx_cpu.mul a b
  let fdiv a b = binary (Op.Binary (Fdiv, a, b)) Nx_cpu.fdiv a b
  let idiv a b = binary (Op.Binary (Idiv, a, b)) Nx_cpu.idiv a b
  let mod_ a b = binary (Op.Binary (Mod, a, b)) Nx_cpu.mod_ a b
  let pow a b = binary (Op.Binary (Pow, a, b)) Nx_cpu.pow a b
  let atan2 a b = binary (Op.Binary (Atan2, a, b)) Nx_cpu.atan2 a b
  let cmpeq a b = binary (Op.Compare (Equal, a, b)) Nx_cpu.cmpeq a b
  let cmpne a b = binary (Op.Compare (Not_equal, a, b)) Nx_cpu.cmpne a b
  let cmplt a b = binary (Op.Compare (Less, a, b)) Nx_cpu.cmplt a b
  let cmple a b = binary (Op.Compare (Less_equal, a, b)) Nx_cpu.cmple a b
  let max a b = binary (Op.Binary (Maximum, a, b)) Nx_cpu.max a b
  let min a b = binary (Op.Binary (Minimum, a, b)) Nx_cpu.min a b
  let xor a b = binary (Op.Binary (Xor, a, b)) Nx_cpu.xor a b
  let or_ a b = binary (Op.Binary (Or, a, b)) Nx_cpu.or_ a b
  let and_ a b = binary (Op.Binary (And, a, b)) Nx_cpu.and_ a b
  let neg t = unary (Op.Unary (Neg, t)) Nx_cpu.neg t
  let recip t = unary (Op.Unary (Recip, t)) Nx_cpu.recip t
  let abs t = unary (Op.Unary (Abs, t)) Nx_cpu.abs t
  let sqrt t = unary (Op.Unary (Sqrt, t)) Nx_cpu.sqrt t
  let sign t = unary (Op.Unary (Sign, t)) Nx_cpu.sign t
  let exp t = unary (Op.Unary (Exp, t)) Nx_cpu.exp t
  let log t = unary (Op.Unary (Log, t)) Nx_cpu.log t
  let sin t = unary (Op.Unary (Sin, t)) Nx_cpu.sin t
  let cos t = unary (Op.Unary (Cos, t)) Nx_cpu.cos t
  let tan t = unary (Op.Unary (Tan, t)) Nx_cpu.tan t
  let asin t = unary (Op.Unary (Asin, t)) Nx_cpu.asin t
  let acos t = unary (Op.Unary (Acos, t)) Nx_cpu.acos t
  let atan t = unary (Op.Unary (Atan, t)) Nx_cpu.atan t
  let sinh t = unary (Op.Unary (Sinh, t)) Nx_cpu.sinh t
  let cosh t = unary (Op.Unary (Cosh, t)) Nx_cpu.cosh t
  let tanh t = unary (Op.Unary (Tanh, t)) Nx_cpu.tanh t
  let trunc t = unary (Op.Unary (Trunc, t)) Nx_cpu.trunc t
  let ceil t = unary (Op.Unary (Ceil, t)) Nx_cpu.ceil t
  let floor t = unary (Op.Unary (Floor, t)) Nx_cpu.floor t
  let round t = unary (Op.Unary (Round, t)) Nx_cpu.round t
  let erf t = unary (Op.Unary (Erf, t)) Nx_cpu.erf t

  let where condition if_true if_false =
    match (condition, if_true, if_false) with
    | Host c, Host a, Host b -> Host (Nx_cpu.where c a b)
    | _ ->
        let r = route_of (Op.Where (condition, if_true, if_false)) in
        settle r
          (Nx_cpu.where (host_of condition) (host_of if_true)
             (host_of if_false))

  let reduce ~op ~axes t_in =
    unary
      (Op.Reduce (reduce_kind op, axes, t_in))
      (Nx_cpu.reduce ~op ~axes) t_in

  let argmax ~axis ~keepdims t_in =
    unary
      (Op.Arg_reduce (Argmax, axis, t_in))
      (Nx_cpu.argmax ~axis ~keepdims)
      t_in

  let argmin ~axis ~keepdims t_in =
    unary
      (Op.Arg_reduce (Argmin, axis, t_in))
      (Nx_cpu.argmin ~axis ~keepdims)
      t_in

  let associative_scan ~axis ~op t_in =
    unary
      (Op.Scan (reduce_kind op, axis, t_in))
      (Nx_cpu.associative_scan ~axis ~op)
      t_in

  let sort ~axis ~descending t_in =
    unary
      (Op.Sort { descending; axis; x = t_in })
      (Nx_cpu.sort ~axis ~descending)
      t_in

  let argsort ~axis ~descending t_in =
    unary
      (Op.Argsort { descending; axis; x = t_in })
      (Nx_cpu.argsort ~axis ~descending)
      t_in

  let expand t_in shape = moved t_in (Expand shape)
  let reshape t_in shape = moved t_in (Reshape shape)
  let permute t_in axes = moved t_in (Permute axes)
  let shrink t_in limits = moved t_in (Shrink limits)
  let flip t_in dims = moved t_in (Flip dims)

  let sliding_window t_in ~axis ~window ~step =
    moved t_in (Window { axis; size = window; step })

  let pad t_in padding_config fill_value =
    unary
      (Op.Pad (padding_config, fill_value, t_in))
      (fun t -> Nx_cpu.pad t padding_config fill_value)
      t_in

  let cat t_list ~axis =
    if List.for_all (function Host _ -> true | _ -> false) t_list then
      Host (Nx_cpu.cat (List.map host_of t_list) ~axis)
    else
      let r = route_of (Op.Cat (axis, t_list)) in
      settle r (Nx_cpu.cat (List.map host_of t_list) ~axis)

  let cast (type a b c d) ~(dtype : (c, d) Nx_dtype.t) (t_in : (a, b) t) :
      (c, d) t =
    match t_in with
    | Host t -> Host (Nx_cpu.cast ~dtype t)
    | _ ->
        routed (Op.Convert (Cast, dtype, t_in)) t_in (Nx_cpu.cast ~dtype)

  let bitcast (type a b c d) ~(dtype : (c, d) Nx_dtype.t) (t_in : (a, b) t) :
      (c, d) t =
    match t_in with
    | Host t -> Host (Nx_cpu.bitcast ~dtype t)
    | _ ->
        routed
          (Op.Convert (Bitcast, dtype, t_in))
          t_in (Nx_cpu.bitcast ~dtype)

  let contiguous t_in = unary (Op.Contiguous t_in) Nx_cpu.copy t_in
  let copy = contiguous
  let threefry key ctr =
    binary (Op.Threefry (key, ctr)) Nx_cpu.threefry key ctr

  let gather data indices ~axis =
    binary
      (Op.Gather (axis, indices, data))
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
            (Op.Scatter
               {
                 mode;
                 unique = unique_indices;
                 axis;
                 indices;
                 updates;
                 into = data_template;
               })
        in
        settle r
          (Nx_cpu.scatter ~mode ~unique_indices (host_of data_template)
             ~indices:(host_of indices) ~updates:(host_of updates) ~axis)

  let update t_in ~starts v =
    match (t_in, starts, v) with
    | Host t, Host s, Host v -> Host (Nx_cpu.update t ~starts:s v)
    | _ ->
        let r = route_of (Op.Update (t_in, starts, v)) in
        settle r
          (Nx_cpu.update (host_of t_in) ~starts:(host_of starts) (host_of v))

  let unfold t_in ~kernel_size ~stride ~dilation ~padding =
    unary
      (Op.Unfold { kernel_size; stride; dilation; padding; x = t_in })
      (fun t -> Nx_cpu.unfold t ~kernel_size ~stride ~dilation ~padding)
      t_in

  let fold t_in ~output_size ~kernel_size ~stride ~dilation ~padding =
    unary
      (Op.Fold
         { output_size; kernel_size; stride; dilation; padding; x = t_in })
      (fun t ->
        Nx_cpu.fold t ~output_size ~kernel_size ~stride ~dilation ~padding)
      t_in

  let matmul a b = binary (Op.Matmul (a, b)) Nx_cpu.matmul a b
  let fft t ~axes =
    unary (Op.Fft { inverse = false; axes; x = t }) (Nx_cpu.fft ~axes) t

  let ifft t ~axes =
    unary (Op.Fft { inverse = true; axes; x = t }) (Nx_cpu.ifft ~axes) t

  let rfft (type a c) (t : (float, a) t) ~(dtype : (Complex.t, c) Nx_dtype.t)
      ~axes : (Complex.t, c) t =
    match t with
    | Host h -> Host (Nx_cpu.rfft h ~dtype ~axes)
    | _ -> routed (Op.Rfft { dtype; axes; x = t }) t (Nx_cpu.rfft ~dtype ~axes)

  let irfft (type a c) ?s (t : (Complex.t, a) t)
      ~(dtype : (float, c) Nx_dtype.t) ~axes : (float, c) t =
    match t with
    | Host h -> Host (Nx_cpu.irfft ?s h ~dtype ~axes)
    | _ ->
        routed
          (Op.Irfft { dtype; axes; s; x = t })
          t (Nx_cpu.irfft ?s ~dtype ~axes)

  let cholesky ~upper t_in =
    unary (Op.Cholesky { upper; x = t_in }) (Nx_cpu.cholesky ~upper) t_in

  let qr ~reduced t_in =
    let r = route_of (Op.Qr { reduced; x = t_in }) in
    let q, rr = Nx_cpu.qr ~reduced (host_of t_in) in
    (settle r q, settle r rr)

  let lu t_in =
    let r = route_of (Op.Lu t_in) in
    let lu, pivots, perm = Nx_cpu.lu (host_of t_in) in
    (settle r lu, settle r pivots, settle r perm)

  let svd ~full_matrices t_in =
    let r = route_of (Op.Svd { full_matrices; x = t_in }) in
    let u, s, vt = Nx_cpu.svd ~full_matrices (host_of t_in) in
    (settle r u, settle r s, settle r vt)

  let eigvals t_in =
    unary (Op.Eig { vectors = false; x = t_in }) Nx_cpu.eigvals t_in

  let eig t_in =
    let r = route_of (Op.Eig { vectors = true; x = t_in }) in
    let vals, vecs = Nx_cpu.eig (host_of t_in) in
    (settle r vals, settle r vecs)

  let eigvalsh t_in =
    unary (Op.Eigh { vectors = false; x = t_in }) Nx_cpu.eigvalsh t_in

  let eigh t_in =
    let r = route_of (Op.Eigh { vectors = true; x = t_in }) in
    let vals, vecs = Nx_cpu.eigh (host_of t_in) in
    (settle r vals, settle r vecs)

  let solve_triangular ~upper ~transpose ~unit_diag a b =
    binary
      (Op.Solve_triangular { upper; transpose; unit_diag; a; b })
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

(* Dispatch

   With no interpretation, operands all on the host run nx.cpu directly, and any
   other operands run on the backend their placed ones share: operands with two
   backends raise. nx answers movements, placing and reading itself. *)

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

(* The backend that runs the computing operation [op]. *)
let backend_of : type r. r Op.t -> backend =
 fun op ->
  match routing op with
  | Computes (_, xs) -> backend_among (Op.name op) xs
  | Moves _ | Places _ | Reads _ ->
      invalid_arg "Nx_effect.backend_of: the operation computes nothing"

(* The elements of [x]'s view in C order, in a host buffer: nx.cpu's storage of
   a host value, whose readers make it contiguous first, and a copy of a placed
   one's, read by the backend that made its storage. *)
let read_elements_of (type a b) (x : (a, b) t) : Nx_device.Buffer.t =
  match x with
  | Host t -> Nx_cpu.to_host t
  | Placed r ->
      let (module B : Backend.S) = r.r_cell.placement.backend in
      B.to_host x
  | Traced _ -> outside_trace ()

(* [x] at [p]. The target placement's backend makes the value there. *)
let move_to (type a b) p (x : (a, b) t) : (a, b) t =
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

let reduce_op : Nx_backend.reduce -> [ `Sum | `Prod | `Max | `Min ] = function
  | Sum -> `Sum
  | Prod -> `Prod
  | Max -> `Max
  | Min -> `Min

(* nx.cpu's kernel and a backend's function for each kind. *)

let cpu_unary : type a b.
    Nx_backend.unary -> (a, b) Nx_cpu.t -> (a, b) Nx_cpu.t = function
  | Neg -> Nx_cpu.neg
  | Recip -> Nx_cpu.recip
  | Abs -> Nx_cpu.abs
  | Sqrt -> Nx_cpu.sqrt
  | Sign -> Nx_cpu.sign
  | Exp -> Nx_cpu.exp
  | Log -> Nx_cpu.log
  | Sin -> Nx_cpu.sin
  | Cos -> Nx_cpu.cos
  | Tan -> Nx_cpu.tan
  | Asin -> Nx_cpu.asin
  | Acos -> Nx_cpu.acos
  | Atan -> Nx_cpu.atan
  | Sinh -> Nx_cpu.sinh
  | Cosh -> Nx_cpu.cosh
  | Tanh -> Nx_cpu.tanh
  | Trunc -> Nx_cpu.trunc
  | Ceil -> Nx_cpu.ceil
  | Floor -> Nx_cpu.floor
  | Round -> Nx_cpu.round
  | Erf -> Nx_cpu.erf

let backend_unary : type a b.
    backend -> Nx_backend.unary -> (a, b) t -> (a, b) t =
 fun (module B : Backend.S) k ->
  match k with
  | Neg -> B.neg
  | Recip -> B.recip
  | Abs -> B.abs
  | Sqrt -> B.sqrt
  | Sign -> B.sign
  | Exp -> B.exp
  | Log -> B.log
  | Sin -> B.sin
  | Cos -> B.cos
  | Tan -> B.tan
  | Asin -> B.asin
  | Acos -> B.acos
  | Atan -> B.atan
  | Sinh -> B.sinh
  | Cosh -> B.cosh
  | Tanh -> B.tanh
  | Trunc -> B.trunc
  | Ceil -> B.ceil
  | Floor -> B.floor
  | Round -> B.round
  | Erf -> B.erf

let cpu_binary : type a b.
    Nx_backend.binary -> (a, b) Nx_cpu.t -> (a, b) Nx_cpu.t -> (a, b) Nx_cpu.t
    = function
  | Add -> Nx_cpu.add
  | Sub -> Nx_cpu.sub
  | Mul -> Nx_cpu.mul
  | Fdiv -> Nx_cpu.fdiv
  | Idiv -> Nx_cpu.idiv
  | Mod -> Nx_cpu.mod_
  | Pow -> Nx_cpu.pow
  | Atan2 -> Nx_cpu.atan2
  | Maximum -> Nx_cpu.max
  | Minimum -> Nx_cpu.min
  | And -> Nx_cpu.and_
  | Or -> Nx_cpu.or_
  | Xor -> Nx_cpu.xor

let backend_binary : type a b.
    backend -> Nx_backend.binary -> (a, b) t -> (a, b) t -> (a, b) t =
 fun (module B : Backend.S) k ->
  match k with
  | Add -> B.add
  | Sub -> B.sub
  | Mul -> B.mul
  | Fdiv -> B.fdiv
  | Idiv -> B.idiv
  | Mod -> B.mod_
  | Pow -> B.pow
  | Atan2 -> B.atan2
  | Maximum -> B.max
  | Minimum -> B.min
  | And -> B.and_
  | Or -> B.or_
  | Xor -> B.xor

let cpu_compare : type a b.
    Nx_backend.compare ->
    (a, b) Nx_cpu.t ->
    (a, b) Nx_cpu.t ->
    (bool, Nx_dtype.bool_elt) Nx_cpu.t = function
  | Equal -> Nx_cpu.cmpeq
  | Not_equal -> Nx_cpu.cmpne
  | Less -> Nx_cpu.cmplt
  | Less_equal -> Nx_cpu.cmple

let backend_compare : type a b.
    backend ->
    Nx_backend.compare ->
    (a, b) t ->
    (a, b) t ->
    (bool, Nx_dtype.bool_elt) t =
 fun (module B : Backend.S) k ->
  match k with
  | Equal -> B.cmpeq
  | Not_equal -> B.cmpne
  | Less -> B.cmplt
  | Less_equal -> B.cmple

let all_host xs = List.for_all (function Host _ -> true | _ -> false) xs

(* The operations that host code calls most, on nx.cpu with no closure when
   their operands are on the host. *)

let direct_unary op k x =
  match x with
  | Host t -> Host (cpu_unary k t)
  | Placed _ | Traced _ -> backend_unary (backend_of op) k x

let direct_binary op k x y =
  match (x, y) with
  | Host x, Host y -> Host (cpu_binary k x y)
  | _ -> backend_binary (backend_of op) k x y

let direct_compare op k x y =
  match (x, y) with
  | Host x, Host y -> Host (cpu_compare k x y)
  | _ -> backend_compare (backend_of op) k x y

let direct_where op c x y =
  match (c, x, y) with
  | Host c, Host x, Host y -> Host (Nx_cpu.where c x y)
  | _ ->
      let (module B : Backend.S) = backend_of op in
      B.where c x y

let direct_reduce op k axes x =
  match x with
  | Host t -> Host (Nx_cpu.reduce ~op:(reduce_op k) ~axes t)
  | Placed _ | Traced _ ->
      let (module B : Backend.S) = backend_of op in
      B.reduce ~op:(reduce_op k) ~axes x

let direct_matmul op x y =
  match (x, y) with
  | Host x, Host y -> Host (Nx_cpu.matmul x y)
  | _ ->
      let (module B : Backend.S) = backend_of op in
      B.matmul x y

let direct_copy op x =
  match x with
  | Host t -> Host (Nx_cpu.copy t)
  | Placed _ | Traced _ ->
      let (module B : Backend.S) = backend_of op in
      B.copy x

(* [on_host1 x f g] is [f] of [x]'s array when [x] is on the host, and [g ()]
   otherwise; [on_host2] and [on_host3] over two and three operands. *)
let on_host1 x f g =
  match x with Host t -> Host (f t) | Placed _ | Traced _ -> g ()

let on_host2 x y f g =
  match (x, y) with Host x, Host y -> Host (f x y) | _ -> g ()

let on_host3 x y z f g =
  match (x, y, z) with Host x, Host y, Host z -> Host (f x y z) | _ -> g ()

(* [direct op] answers [op] with no interpretation. The decompositions run on
   the backend even on the host: their host case settles its results as a
   placed call's does. *)
let direct : type r. r Op.t -> r =
 fun op ->
  match[@warning "@4@8"] op with
  | Unary (k, x) -> direct_unary op k x
  | Binary (k, x, y) -> direct_binary op k x y
  | Compare (k, x, y) -> direct_compare op k x y
  | Where (c, x, y) -> direct_where op c x y
  | Reduce (k, axes, x) -> direct_reduce op k axes x
  | Scan (k, axis, x) ->
      let kind = reduce_op k in
      on_host1 x (Nx_cpu.associative_scan ~axis ~op:kind) (fun () ->
          let (module B : Backend.S) = backend_of op in
          B.associative_scan ~axis ~op:kind x)
  | Arg_reduce (Argmax, axis, x) ->
      on_host1 x (Nx_cpu.argmax ~axis ~keepdims:false) (fun () ->
          let (module B : Backend.S) = backend_of op in
          B.argmax ~axis ~keepdims:false x)
  | Arg_reduce (Argmin, axis, x) ->
      on_host1 x (Nx_cpu.argmin ~axis ~keepdims:false) (fun () ->
          let (module B : Backend.S) = backend_of op in
          B.argmin ~axis ~keepdims:false x)
  | Sort { descending; axis; x } ->
      on_host1 x (Nx_cpu.sort ~axis ~descending) (fun () ->
          let (module B : Backend.S) = backend_of op in
          B.sort ~axis ~descending x)
  | Argsort { descending; axis; x } ->
      on_host1 x (Nx_cpu.argsort ~axis ~descending) (fun () ->
          let (module B : Backend.S) = backend_of op in
          B.argsort ~axis ~descending x)
  | Pad (padding, v, x) ->
      on_host1 x
        (fun t -> Nx_cpu.pad t padding v)
        (fun () ->
          let (module B : Backend.S) = backend_of op in
          B.pad x padding v)
  | Cat (axis, xs) ->
      if all_host xs then Host (Nx_cpu.cat (List.map host_of xs) ~axis)
      else
        let (module B : Backend.S) = backend_of op in
        B.cat xs ~axis
  | Convert (Cast, dtype, x) ->
      on_host1 x (Nx_cpu.cast ~dtype) (fun () ->
          let (module B : Backend.S) = backend_of op in
          B.cast ~dtype x)
  | Convert (Bitcast, dtype, x) ->
      on_host1 x (Nx_cpu.bitcast ~dtype) (fun () ->
          let (module B : Backend.S) = backend_of op in
          B.bitcast ~dtype x)
  | Threefry (key, ctr) ->
      on_host2 key ctr Nx_cpu.threefry (fun () ->
          let (module B : Backend.S) = backend_of op in
          B.threefry key ctr)
  | Gather (axis, indices, data) ->
      on_host2 data indices
        (fun d i -> Nx_cpu.gather d i ~axis)
        (fun () ->
          let (module B : Backend.S) = backend_of op in
          B.gather data indices ~axis)
  | Scatter { mode; unique; axis; indices; updates; into } ->
      on_host3 into indices updates
        (fun d i u ->
          Nx_cpu.scatter ~mode ~unique_indices:unique d ~indices:i ~updates:u
            ~axis)
        (fun () ->
          let (module B : Backend.S) = backend_of op in
          B.scatter ~mode ~unique_indices:unique into ~indices ~updates ~axis)
  | Update (x, starts, v) ->
      on_host3 x starts v
        (fun t s v -> Nx_cpu.update t ~starts:s v)
        (fun () ->
          let (module B : Backend.S) = backend_of op in
          B.update x ~starts v)
  | Unfold { kernel_size; stride; dilation; padding; x } ->
      on_host1 x
        (fun t -> Nx_cpu.unfold t ~kernel_size ~stride ~dilation ~padding)
        (fun () ->
          let (module B : Backend.S) = backend_of op in
          B.unfold x ~kernel_size ~stride ~dilation ~padding)
  | Fold { output_size; kernel_size; stride; dilation; padding; x } ->
      on_host1 x
        (fun t ->
          Nx_cpu.fold t ~output_size ~kernel_size ~stride ~dilation ~padding)
        (fun () ->
          let (module B : Backend.S) = backend_of op in
          B.fold x ~output_size ~kernel_size ~stride ~dilation ~padding)
  | Matmul (x, y) -> direct_matmul op x y
  | Fft { inverse; axes; x } ->
      on_host1 x
        (if inverse then Nx_cpu.ifft ~axes else Nx_cpu.fft ~axes)
        (fun () ->
          let (module B : Backend.S) = backend_of op in
          if inverse then B.ifft x ~axes else B.fft x ~axes)
  | Rfft { dtype; axes; x } ->
      on_host1 x (Nx_cpu.rfft ~dtype ~axes) (fun () ->
          let (module B : Backend.S) = backend_of op in
          B.rfft x ~dtype ~axes)
  | Irfft { dtype; axes; s; x } ->
      on_host1 x (Nx_cpu.irfft ?s ~dtype ~axes) (fun () ->
          let (module B : Backend.S) = backend_of op in
          B.irfft ?s x ~dtype ~axes)
  | Contiguous x -> direct_copy op x
  | Cholesky { upper; x } ->
      on_host1 x (Nx_cpu.cholesky ~upper) (fun () ->
          let (module B : Backend.S) = backend_of op in
          B.cholesky ~upper x)
  | Qr { reduced; x } ->
      let (module B : Backend.S) = backend_of op in
      B.qr ~reduced x
  | Lu x ->
      let (module B : Backend.S) = backend_of op in
      B.lu x
  | Svd { full_matrices; x } ->
      let (module B : Backend.S) = backend_of op in
      B.svd ~full_matrices x
  | Eig { vectors; x } ->
      let (module B : Backend.S) = backend_of op in
      if vectors then
        let values, vectors = B.eig x in
        (values, Some vectors)
      else (B.eigvals x, None)
  | Eigh { vectors; x } ->
      let (module B : Backend.S) = backend_of op in
      if vectors then
        let values, vectors = B.eigh x in
        (values, Some vectors)
      else (B.eigvalsh x, None)
  | Solve_triangular { upper; transpose; unit_diag; a; b = y } ->
      on_host2 a y
        (Nx_cpu.solve_triangular ~upper ~transpose ~unit_diag)
        (fun () ->
          let (module B : Backend.S) = backend_of op in
          B.solve_triangular ~upper ~transpose ~unit_diag a y)
  | Move (x, m) -> moved x m
  | Place (p, x) -> move_to p x
  | Read x -> read_elements_of x

(* Interpretation

   [eval op] performs [op] for the transformations around the caller, and
   answers it directly when there is none: only a perform of this very effect
   that no handler took falls back. *)
let eval : type r. r Op.t -> r =
 fun op ->
  let e = E_op op in
  match Effect.perform e with
  | v -> v
  | exception Effect.Unhandled e' when Obj.repr e' == Obj.repr e -> direct op

(* Entry functions, one per constructor. *)

let unary k x = eval (Unary (k, x))
let binary k x y = eval (Binary (k, x, y))
let cmp k x y = eval (Compare (k, x, y))
let where c x y = eval (Where (c, x, y))
let reduce k ~axes x = eval (Reduce (k, axes, x))
let scan k ~axis x = eval (Scan (k, axis, x))
let arg_reduce k ~axis x = eval (Arg_reduce (k, axis, x))
let sort ~descending ~axis x = eval (Sort { descending; axis; x })
let argsort ~descending ~axis x = eval (Argsort { descending; axis; x })
let pad padding v x = eval (Pad (padding, v, x))
let cat ~axis xs = eval (Cat (axis, xs))
let cast dtype x = eval (Convert (Cast, dtype, x))
let bitcast dtype x = eval (Convert (Bitcast, dtype, x))
let threefry key ctr = eval (Threefry (key, ctr))
let gather ~axis indices x = eval (Gather (axis, indices, x))

let scatter ~mode ~unique ~axis ~indices ~updates into =
  eval (Scatter { mode; unique; axis; indices; updates; into })

let update x ~starts v = eval (Update (x, starts, v))

let unfold ~kernel_size ~stride ~dilation ~padding x =
  eval (Unfold { kernel_size; stride; dilation; padding; x })

let fold ~output_size ~kernel_size ~stride ~dilation ~padding x =
  eval (Fold { output_size; kernel_size; stride; dilation; padding; x })

let matmul x y = eval (Matmul (x, y))
let fft ~inverse ~axes x = eval (Fft { inverse; axes; x })
let rfft dtype ~axes x = eval (Rfft { dtype; axes; x })
let irfft ?s dtype ~axes x = eval (Irfft { dtype; axes; s; x })
let cholesky ~upper x = eval (Cholesky { upper; x })
let qr ~reduced x = eval (Qr { reduced; x })
let lu x = eval (Lu x)
let svd ~full_matrices x = eval (Svd { full_matrices; x })
let eigvals x = fst (eval (Eig { vectors = false; x }))
let eigvalsh x = fst (eval (Eigh { vectors = false; x }))

let with_vectors op = function
  | values, Some vectors -> (values, vectors)
  | _, None -> invalid_arg ("Nx." ^ op ^ ": the eigenvectors are missing")

let eig x = with_vectors "eig" (eval (Eig { vectors = true; x }))
let eigh x = with_vectors "eigh" (eval (Eigh { vectors = true; x }))

let solve_triangular ~upper ~transpose ~unit_diag a b =
  eval (Solve_triangular { upper; transpose; unit_diag; a; b })

let move x m = eval (Move (x, m))
let reshape x shape = move x (Reshape shape)
let expand x shape = move x (Expand shape)
let permute x axes = move x (Permute axes)
let shrink x limits = move x (Shrink limits)
let flip x dims = move x (Flip dims)

let sliding_window x ~axis ~window ~step =
  move x (Window { axis; size = window; step })

(* The elements of [x]'s view in C order, in a host buffer. The storage of a
   host value that is contiguous from its first element is that buffer. *)
let read x = eval (Read x)

(* A value already at [p] is returned as it is. *)
let place (type a b) p (x : (a, b) t) : (a, b) t =
  if Placement.equal (placement x) p then
    match x with
    | Placed r -> Cell.with_borrow r.r_cell (fun () -> x)
    | Host _ | Traced _ -> x
  else eval (Place (p, x))

(* [copy x] is [x] in storage of its own, C-contiguous from its first element:
   it always copies. [contiguous x] is [x] itself when its bytes are already
   C-contiguous from its first element. A traced value has no bytes: the
   interpretation that made it answers its copy. *)
let copy x = eval (Contiguous x)

let contiguous x =
  match x with
  | Traced _ -> copy x
  | Host _ | Placed _ ->
      let v = view x in
      if View.is_c_contiguous v && View.offset v = 0 then x else copy x

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
  let b = read x in
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
