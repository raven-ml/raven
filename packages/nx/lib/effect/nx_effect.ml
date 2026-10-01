(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Nx_array

(* Types

   GADT constructors (the operations of [Op.t] below) require that type
   variables in the payload be deducible from the return type. A transparent
   alias of [Nx_array.t] would not be injective, so the tensor is a GADT of its
   own, whose parameters the return type determines.

   A tensor is one of three things. [Host] is an nx.cpu tensor, the value of the
   host placement: the host device with the host backend.
   [Placed] is a value at any other placement, held in runtime buffers, one per
   device of its placement: nx knows its placement, dtype and view, and the
   buffers are its storage. [Traced] is a node of a trace: it has no bytes and
   never will, and the tracer that made it keeps its payload in [t_node].

   The host device thus holds values in two ways. At the host placement a value
   is nx.cpu's own tensor, with no cell, so that the default path
   pays nothing for what compiled calls need of a storage; at another placement
   of the host device (another backend, or beside other devices) it is placed,
   in the runtime's host buffers.

   The views of one placed storage share one cell, which holds what belongs to
   the storage rather than to a view: whether it is live or was consumed by a
   compiled call, and how many reachable programs bind it.

   A placement holds its backend, whose kernels compute on arrays and know
   nothing of these types. *)

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

module Types = struct
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

  and state =
    | Live of Nx_device.Buffer.t list
        (* one buffer per device of the cell's placement, in its order *)
    | Consumed of consumption

  (* Where a compiled call consumed a storage: the consumed leaf's path in the
     call's arguments, whose first segment is the argument's position from 0. *)
  and consumption = { path : string }

  and ('a, 'b) traced = {
    t_id : int; (* fresh; identity tables key by it *)
    t_placement : placement; (* where the value lives *)
    t_context : placement; (* where the trace creates its values *)
    t_dtype : ('a, 'b) Nx_dtype.t;
    t_view : View.t; (* the layout of the value it stands for *)
    t_node : ('a, 'b) node; (* the tracer's payload *)
  }

  (* Devices, a layout and the one backend that computes on the values there. *)
  and placement = { grid : Nx_device.t Grid.t; backend : backend }

  and backend = Nx_backend.t
  and ('a, 'b) node = ..
end

include Types

(* Where a creation makes its value. *)
type context = placement

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

(* Devices and placements

   The [Placement] module is defined after the routing; the functions here are
   what the code in between needs. *)

let is_host_device d = Nx_device.equal d Nx_device.host
let is_host_backend b = Nx_backend.equal b Nx_cpu.backend

let is_host_placement p =
  is_host_backend p.backend
  && match Grid.devices p.grid with [ d ] -> is_host_device d | _ -> false

let pp_grid ppf g = Grid.pp Nx_device.pp ppf g

let pp_placement ppf p =
  pp_grid ppf p.grid;
  if not (is_host_backend p.backend) then
    Format.fprintf ppf " with %s" (Nx_backend.name p.backend)

let devices_of p = Grid.devices p.grid

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
        (Printf.sprintf "Nx.Placement.window: %s holds no window"
           (Nx_device.name d))
  | Some k ->
      check_shape "Nx.Placement.window" p shape;
      let w = Array.map (fun n -> (0, n)) shape in
      List.iter2
        (fun (a, n) (_, j) ->
          let size = shape.(a) / n in
          w.(a) <- (j * size, (j + 1) * size))
        (Grid.cuts p.grid) (Grid.tile_index p.grid k);
      w

(* Placed constructors *)

(* A cell over [storage], one runtime buffer of [length] elements per device of
   [placement], which the runtime releases with the buffers. *)
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
    (read : Nx_device.t -> View.t -> Nx_device.Buffer.t) : Nx_device.Buffer.t =
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

(* Runtime buffers *)

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
    else
      let dst = alloc dtype (View.shape view) in
      Nx_cpu.contiguous { dtype; view; buffer = span } ~dst;
      dst.buffer

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
    List.for_all Nx_device.shares_host_memory ds
    && List.for_all Option.is_some spans
  then
    let spans = List.map Option.get spans in
    let ((offset, strides, shape) as view) = fst (List.hd spans) in
    if List.for_all (fun (v', _) -> v' = view) spans then
      let borrows =
        List.map2
          (fun d (_, run) -> Nx_device.Buffer.borrow d run)
          ds spans
      in
      if List.for_all Result.is_ok borrows then
        Some (View.create ~offset ~strides shape, List.map Result.get_ok borrows)
      else None
    else None
  else None

(* Reading placed values *)

(* The elements of a placed value's view. *)
let read_elements (type a b) (r : (a, b) resident) : Nx_device.Buffer.t =
  Cell.with_borrow r.r_cell (fun () ->
      match r.r_cell.state with
      | Consumed k -> consumed k
      | Live bufs ->
          let holders = devices_of r.r_cell.placement in
          let buffer_on d =
            List.nth bufs
              (Option.get (List.find_index (Nx_device.equal d) holders))
          in
          let shape = global r.r_placement (View.shape r.r_view) in
          assemble r
            (Array.map (fun n -> (0, n)) shape)
            (fun d v -> read_view r.r_dtype (buffer_on d) v))

(* Raises unless [buffer] is a host buffer of [dtype]'s format, as nx.cpu reads
   it: through its host address, [dtype]'s elements at a time. *)
let check_host fn dtype buffer =
  if not (Nx_device.equal (Nx_device.Buffer.device buffer) Nx_device.host) then
    invalid_arg
      (Printf.sprintf "%s: the buffer is on %s, not CPU" fn
         (Nx_device.name (Nx_device.Buffer.device buffer)));
  if
    not
      (Nx_dtype.Scalar.equal
         (Nx_device.Buffer.dtype buffer)
         (Nx_dtype.Scalar.of_dtype dtype))
  then
    invalid_arg
      (Printf.sprintf "%s: a %s buffer read as %s" fn
         (Nx_dtype.Scalar.to_string (Nx_device.Buffer.dtype buffer))
         (Nx_dtype.to_string dtype))

(* [file_run r] is the file bytes that hold [r]'s storage, when they lie on the
   disk at a byte aligned to an element, so that the host can read them in
   place. *)
let file_run (type a b) (r : (a, b) resident) =
  match Cell.state r.r_cell with
  | Live [ b ]
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
let read_copy (type a b) (r : (a, b) resident) : (a, b) Nx_array.t =
  let buffer = read_elements r in
  check_host "Nx_effect.read" r.r_dtype buffer;
  { dtype = r.r_dtype; view = View.create (View.shape (whole_view r)); buffer }

(* A value on the disk is read where it lies, in its file's pages, and keeps its
   view; it is copied when the system does not map the file. *)
let read_host (type a b) (r : (a, b) resident) : (a, b) Nx_array.t =
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
let host_of : type a b. (a, b) t -> (a, b) Nx_array.t = function
  | Host t -> t
  | Placed r -> read_host r
  | Traced _ -> outside_trace ()

let on_disk p =
  match Grid.devices p.grid with
  | [ d ] -> Nx_device.equal d Nx_device.disk
  | _ -> false

(* A value on the disk is placed on devices that share the host's memory by
   borrowing its file's pages, and keeps its view ([file_windows]). Otherwise
   a window that is a contiguous run of a value's one runtime buffer is copied
   from it, device to device: a value on the disk is read into the device.
   Other windows are copied from the value read to the host. *)
let place_at : type a b. placement -> (a, b) t -> (a, b) t =
 fun p x ->
  if on_disk p then
    invalid_arg
      "Nx.place: values on DISK are read from files, and none is placed there";
  let host = lazy (match x with Placed r -> read_copy r | _ -> host_of x) in
  let dt, v, run =
    match x with
    | Placed r -> (
        match Cell.state r.r_cell with
        | Live [ b ] -> (r.r_dtype, r.r_view, run_in b)
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
           bufs)
  | None ->
      let piece w =
        match run (View.shrink v w) with
        | Some b -> b
        | None ->
            let h = Lazy.force host in
            Elements.contiguous h.buffer (View.shrink h.view w)
      in
      let bufs =
        List.map2
          (fun d w ->
            let b = Nx_device.Buffer.create d s n in
            if n > 0 then Nx_device.Buffer.copy ~src:(piece w) ~dst:b;
            b)
          ds windows
      in
      placed p dt (View.create local)
        (cell ~placement:p ~length:n bufs)

(* Traced constructor *)

let traced (type a b) ?view (ctx : context) (p : placement)
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

(* The gate: how many interceptions are live, on any domain. While none is,
   nothing performs an effect to find one. The count is global because a
   suspended fiber may resume on another domain. *)
let intercepts = Atomic.make 0
let intercepting () = Atomic.get intercepts > 0

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

(* Operations

   Every operation nx performs is a constructor of [Op.t]: the computing ones,
   which the placement's backend answers, and the movements, placing and
   reading, which nx answers itself. A kind names the function among the
   operations of one constructor. *)

(* The two dtype conversions: [Cast] converts values, and [Bitcast] reads the
   elements' bytes in row-major order as elements of another dtype, consuming a
   last axis of [k] when it is [k] times wider and adding one when it is [k]
   times narrower. *)
type conversion = Cast | Bitcast

module Op = struct
  type move =
    | Reshape of int array
    | Expand of int array
    | Permute of int array
    | Shrink of (int * int) array
    | Flip of bool array
    | Window of { axis : int; size : int; step : int }

  type int32_t = (int32, Nx_dtype.int32_elt) Types.t
  type int64_t = (int64, Nx_dtype.int64_elt) Types.t

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
    | Arg_reduce : Nx_backend.arg_reduce * int * ('a, 'b) Types.t -> int64_t t
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
        -> int64_t t
    | Pad : (int * int) array * 'a * ('a, 'b) Types.t -> ('a, 'b) Types.t t
    | Cat : int * ('a, 'b) Types.t list -> ('a, 'b) Types.t t
    | Convert :
        conversion * ('c, 'd) Nx_dtype.t * ('a, 'b) Types.t
        -> ('c, 'd) Types.t t
    | Threefry : int32_t * int32_t -> int32_t t
    | Gather : int * int64_t * ('a, 'b) Types.t -> ('a, 'b) Types.t t
    | Scatter : {
        mode : Nx_backend.scatter;
        unique : bool;
        axis : int;
        indices : int64_t;
        updates : ('a, 'b) Types.t;
        into : ('a, 'b) Types.t;
      }
        -> ('a, 'b) Types.t t
    | Update :
        ('a, 'b) Types.t * int64_t * ('a, 'b) Types.t
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
    | Lu : ('a, 'b) Types.t -> (('a, 'b) Types.t * int64_t * int64_t) t
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
    | Read : { by : string; x : ('a, 'b) Types.t } -> Nx_device.Buffer.t t

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
    | Read { x; _ } -> [ P x ]

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

(* Routing

   Every fallback runs where its operands live. Operands all on the host run on
   nx.cpu. Placed operands must share their devices and backend, and host
   operands join them: the backend runs on each device (see Computing). The
   route is decided before anything is read, so operands on two device lists
   raise before any work.

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

(* [route where op rule xs] is where [op] runs over [xs], [where] giving each
   operand's placement, [None] for one that joins any. Placed operands with
   different backends raise, and so do those on different device sets, but for
   [whole_shards]. *)
let route where op rule xs =
  match List.filter_map where xs with
  | [] -> On_host
  | p :: rest -> (
      (match List.find_opt (fun q -> q.backend != p.backend) rest with
      | Some q -> mixed op p q
      | None -> ());
      match List.find_opt (fun q -> not (same_devices p q)) rest with
      | None ->
          At
            (result op rule (List.map (fun o -> (where o, rank o)) xs))
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
  | Unary _ | Binary _ | Compare _ | Where _
  | Convert (Cast, _, _)
  | Threefry _ | Contiguous _ ->
      computes Elementwise
  | Convert (Bitcast, dt, x) ->
      (* A widening reads each run of the last axis as one element, so the axis
         must be whole on each device. *)
      if Nx_dtype.itemsize dt > Nx_dtype.itemsize (dtype x) then
        along_axes [ rank (P x) - 1 ]
      else computes Elementwise
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
  | Read { x; _ } -> Reads (P x)

(* Where a computing operation runs. *)
let route_of : type r. r Op.t -> route =
 fun op ->
  match routing op with
  | Computes (rule, xs) ->
      route (fun (P x) -> placement_of x) (Op.name op) rule xs
  | Moves _ | Places _ | Reads _ ->
      invalid_arg "Nx_effect.route_of: the operation computes nothing"

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

(* [moved x m] is [x] moved by [m]: view arithmetic over the same storage. *)
let moved (type a b) (x : (a, b) t) m : (a, b) t =
  match x with
  | Host t -> Host { t with view = move_view t.view m }
  | Placed r ->
      let r_placement, r_view = split_view r.r_placement r.r_view m in
      Placed { r with r_id = fresh_id (); r_placement; r_view }
  | Traced _ -> outside_trace ()

(* Placements *)

module Placement = struct
  type t = placement

  let v backend grid = { grid; backend }
  let grid p = p.grid
  let backend p = p.backend
  let host = { grid = Grid.device Nx_device.host; backend = Nx_cpu.backend }
  let devices = devices_of
  let is_host = is_host_placement
  let cuts p = Grid.cuts p.grid
  let uncut p ~axis = { p with grid = Grid.uncut p.grid ~axis }
  let map_axes f p = { p with grid = Grid.map_axes f p.grid }

  (* A backend that does not run on a device is legal: a compiled call needs
     only the devices, and the first eager operation there refuses. *)
  let check what ds =
    let fail fmt =
      Printf.ksprintf invalid_arg ("Nx.Placement.%s: " ^^ fmt) what
    in
    let rec distinct = function
      | [] -> ()
      | d :: rest ->
          if List.memq d rest then
            fail "%s appears twice" (Nx_device.name d);
          distinct rest
    in
    match ds with [] -> fail "no device" | _ -> distinct ds

  let device ?(backend = Nx_cpu.backend) d =
    check "device" [ d ];
    { grid = Grid.device d; backend }

  let replicated ?(backend = Nx_cpu.backend) ds =
    check "replicated" ds;
    { grid = Grid.v ds [ List.length ds ] []; backend }

  let sharded ?(backend = Nx_cpu.backend) ~axis ds =
    if axis < 0 then
      invalid_arg (Printf.sprintf "Nx.Placement.sharded: axis %d < 0" axis);
    check "sharded" ds;
    { grid = Grid.v ds [ List.length ds ] [ (axis, [ 0 ]) ]; backend }

  let check_shape = check_shape
  let window = window_of

  (* The placement of a value with a new leading axis, and of one without its
     leading axis: a grid axis that cut it then holds copies. *)
  let with_leading_axis p = map_axes succ p

  let without_leading_axis p =
    match cuts p with [] -> p | _ -> map_axes pred (uncut p ~axis:0)

  let equal p q =
    Nx_backend.equal p.backend q.backend && Grid.equal ( == ) p.grid q.grid

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
  match x with
  | Host _ -> Placement.host
  | Placed r -> r.r_placement
  | Traced t -> t.t_placement

(* Where [op]'s result lives: where evaluation puts it, a traced operand joining
   as its placement says. Raises [Invalid_argument] as evaluation does when the
   operands cannot meet. *)
let result_placement : type r. r Op.t -> placement =
 fun op ->
  let where (P x) =
    match x with
    | Traced t when is_host_placement t.t_placement -> None
    | Traced t -> Some t.t_placement
    | Host _ | Placed _ -> placement_of x
  in
  match routing op with
  | Computes (rule, xs) -> (
      match route where (Op.name op) rule xs with
      | On_host -> Placement.host
      | At p -> p)
  | Moves (P x, m) -> moved_placement (placement x) (View.shape (view x)) m
  | Places p -> p
  | Reads _ -> Placement.host

(* Results' metadata

   The shape and dtype of an operation's result, from its operands', without
   computing it: what nx allocates before a kernel writes it, and what
   [Op.shape] and [Op.dtype] answer. *)

let pad_shape padding s =
  Array.mapi
    (fun i d ->
      let before, after = padding.(i) in
      d + before + after)
    s

(* The shape of [shape]'s elements of [src] read as [dst]: a [k] times wider
   [dst] consumes the last axis, of [k], and a [k] times narrower one adds a
   last axis of [k]. *)
let bitcast_shape src dst shape =
  let w = Nx_dtype.itemsize src and w' = Nx_dtype.itemsize dst in
  if w' > w then Array.sub shape 0 (Array.length shape - 1)
  else if w' < w then Array.append shape [| w / w' |]
  else shape

let cat_shape axis = function
  | [] -> invalid_arg "Nx.concatenate: no value to concatenate"
  | s :: _ as shapes ->
      let total = List.fold_left (fun n s -> n + s.(axis)) 0 shapes in
      Array.mapi (fun i d -> if i = axis then total else d) s

(* The windows along each spatial axis of extents [spatial]: none where the
   dilated kernel is longer than the padded extent. *)
let window_counts kernel_size stride dilation padding spatial =
  Array.mapi
    (fun i n ->
      let before, after = padding.(i) in
      let extent = (dilation.(i) * (kernel_size.(i) - 1)) + 1 in
      let padded = n + before + after in
      if padded < extent then 0 else ((padded - extent) / stride.(i)) + 1)
    spatial

let unfold_shape kernel_size stride dilation padding s =
  let k = Array.length kernel_size in
  let lead = Array.length s - k in
  let windows =
    window_counts kernel_size stride dilation padding (Array.sub s lead k)
  in
  Array.append (Array.sub s 0 lead)
    [| Array.fold_left ( * ) 1 kernel_size; Array.fold_left ( * ) 1 windows |]

let fold_shape output_size s =
  Array.append (Array.sub s 0 (Array.length s - 2)) output_size

(* Leading batch axes broadcast; the product of [m, k] and [k, n] is [m, n]. *)
let matmul_shape sa sb =
  let na = Array.length sa and nb = Array.length sb in
  let r = Int.max na nb - 2 in
  let batch =
    Array.init r (fun i ->
        let da = if i - (r + 2 - na) >= 0 then sa.(i - (r + 2 - na)) else 1 in
        let db = if i - (r + 2 - nb) >= 0 then sb.(i - (r + 2 - nb)) else 1 in
        if da = 1 then db else da)
  in
  Array.append batch [| sa.(na - 2); sb.(nb - 1) |]

(* The last transformed axis of a real transform holds [n / 2 + 1] complex
   values, and its inverse [s]'s last size, or [2 (n - 1)]. *)
let rfft_shape axes s =
  let s = Array.copy s in
  let last = axes.(Array.length axes - 1) in
  s.(last) <- (s.(last) / 2) + 1;
  s

let irfft_shape axes sizes s =
  let s = Array.copy s in
  let n = Array.length axes - 1 in
  let last = axes.(n) in
  s.(last) <-
    (match sizes with Some sizes -> sizes.(n) | None -> (s.(last) - 1) * 2);
  s

let result_shape : type a b. (a, b) t Op.t -> int array =
 fun op ->
  let s x = View.shape (view x) in
  match op with
  | Unary (_, x) -> s x
  | Binary (_, a, _) -> s a
  | Compare (_, a, _) -> s a
  | Where (_, a, _) -> s a
  | Reduce (_, axes, x) -> Shape.reduce_output_shape (s x) axes false
  | Scan (_, _, x) -> s x
  | Arg_reduce (_, axis, x) -> Shape.reduce_output_shape (s x) [| axis |] false
  | Sort { x; _ } -> s x
  | Argsort { x; _ } -> s x
  | Pad (padding, _, x) -> pad_shape padding (s x)
  | Cat (axis, xs) -> cat_shape axis (List.map s xs)
  | Convert (Cast, _, x) -> s x
  | Convert (Bitcast, dt, x) -> bitcast_shape (dtype x) dt (s x)
  | Threefry (_, ctr) -> s ctr
  | Gather (_, indices, _) -> s indices
  | Scatter { into; _ } -> s into
  | Update (x, _, _) -> s x
  | Unfold { kernel_size; stride; dilation; padding; x } ->
      unfold_shape kernel_size stride dilation padding (s x)
  | Fold { output_size; x; _ } -> fold_shape output_size (s x)
  | Matmul (a, b) -> matmul_shape (s a) (s b)
  | Fft { x; _ } -> s x
  | Rfft { axes; x; _ } -> rfft_shape axes (s x)
  | Irfft { axes; s = sizes; x; _ } -> irfft_shape axes sizes (s x)
  | Contiguous x -> s x
  | Cholesky { x; _ } -> s x
  | Solve_triangular { b; _ } -> s b
  | Move (x, m) -> View.shape (move_view (view x) m)
  | Place (_, x) -> s x
  (* A read gives a buffer, whose abstract type the checker cannot tell from a
     value's. *)
  | Read _ -> assert false

let result_dtype : type a b. (a, b) t Op.t -> (a, b) Nx_dtype.t =
 fun op ->
  match op with
  | Unary (_, x) -> dtype x
  | Binary (_, a, _) -> dtype a
  | Compare _ -> Nx_dtype.Bool
  | Where (_, a, _) -> dtype a
  | Reduce (_, _, x) -> dtype x
  | Scan (_, _, x) -> dtype x
  | Arg_reduce _ -> Nx_dtype.Int64
  | Sort { x; _ } -> dtype x
  | Argsort _ -> Nx_dtype.Int64
  | Pad (_, _, x) -> dtype x
  | Cat (_, x :: _) -> dtype x
  | Cat (_, []) -> invalid_arg "Nx.concatenate: no value to concatenate"
  | Convert (_, dt, _) -> dt
  | Threefry _ -> Nx_dtype.Int32
  | Gather (_, _, data) -> dtype data
  | Scatter { into; _ } -> dtype into
  | Update (x, _, _) -> dtype x
  | Unfold { x; _ } -> dtype x
  | Fold { x; _ } -> dtype x
  | Matmul (a, _) -> dtype a
  | Fft { x; _ } -> dtype x
  | Rfft { dtype; _ } -> dtype
  | Irfft { dtype; _ } -> dtype
  | Contiguous x -> dtype x
  | Cholesky { x; _ } -> dtype x
  | Solve_triangular { b; _ } -> dtype b
  | Move (x, _) -> dtype x
  | Place (_, x) -> dtype x
  | Read _ -> assert false

(* Dispatch

   With no interpretation, operands all on the host run nx.cpu directly, and any
   other operands run on the backend of the placement their route gives: the
   one their placed operands share. nx answers movements, placing and reading
   itself. *)

(* Exactly the elements of [x]'s view in C order, in a host buffer: a host
   value's storage when they are one run of it, gathered by nx.cpu otherwise,
   and a copy of a placed one's, read from its runtime buffers. *)
let read_elements_of (type a b) (x : (a, b) t) : Nx_device.Buffer.t =
  match x with
  | Host t -> (
      match run_in t.buffer t.view with
      | Some b -> b
      | None ->
          let dst = alloc t.dtype (View.shape t.view) in
          Nx_cpu.contiguous t ~dst;
          dst.buffer)
  | Placed r -> read_elements r
  | Traced _ -> outside_trace ()

(* [x] at [p]. A placed value at a placement that differs from [p] only in
   backend is a view of its storage; otherwise it is placed there anew. *)
let move_to (type a b) p (x : (a, b) t) : (a, b) t =
  let move () =
    match x with
    | Traced _ -> outside_trace ()
    | Placed r when is_host_placement p -> Host (read_host r)
    | Placed r when Grid.equal ( == ) r.r_placement.grid p.grid ->
        Placed { r with r_id = fresh_id (); r_placement = p }
    | Host _ | Placed _ ->
        check_shape "Nx.place" p (View.shape (view x));
        place_at p x
  in
  match x with
  | Placed r -> Cell.with_borrow r.r_cell move
  | Host _ | Traced _ -> move ()

(* Computing

   nx allocates each result, C-contiguous from its first element, and a
   backend's kernel writes it. Operands all on the host run nx.cpu's kernels on
   their own arrays. Operands at any other placement run its backend once per
   device, on that device's arrays, after nx copies there the operands that are
   not: host values, values on the disk, and values that the device needs
   whole where they are split. *)

type kernels = (module Nx_backend.S)

(* Where kernels run: their backend's functions, how a result is allocated
   there, and each operand's array there. *)
type env = {
  kernels : kernels;
  alloc : 'a 'b. ('a, 'b) Nx_dtype.t -> int array -> ('a, 'b) Nx_array.t;
  arr : 'a 'b. ('a, 'b) t -> ('a, 'b) Nx_array.t;
}

let host_env = { kernels = (module Nx_cpu); alloc; arr = host_of }
let shape_of (a : ('a, 'b) Nx_array.t) = View.shape a.view

(* Kernels read their operands' memory under read claims, so that no compiled
   call lends it to a result meanwhile. Claims are released however the kernel
   ends, and none allocates. *)

let claim (a : ('a, 'b) Nx_array.t) = Nx_device.Buffer.Claim.read a.buffer
let release (a : ('a, 'b) Nx_array.t) = Nx_device.Buffer.Claim.release a.buffer

let release2 a b =
  release a;
  release b

let release3 a b c =
  release2 a b;
  release c

let claim2 a b =
  claim a;
  match claim b with
  | () -> ()
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release a;
      Printexc.raise_with_backtrace e bt

let claim3 a b c =
  claim2 a b;
  match claim c with
  | () -> ()
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release2 a b;
      Printexc.raise_with_backtrace e bt

let rec claim_all = function
  | [] -> ()
  | a :: rest -> (
      claim a;
      match claim_all rest with
      | () -> ()
      | exception e ->
          let bt = Printexc.get_raw_backtrace () in
          release a;
          Printexc.raise_with_backtrace e bt)

let release_all xs = List.iter release xs

(* Each operation's results, allocated, and written by kernels [k]. *)

let k_unary (e : env) k a =
  let (module K) = e.kernels in
  let dst = e.alloc a.dtype (shape_of a) in
  claim a;
  (match K.unary k a ~dst with
  | () -> release a
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release a;
      Printexc.raise_with_backtrace e bt);
  dst

let k_binary (e : env) k a b =
  let (module K) = e.kernels in
  let dst = e.alloc a.dtype (shape_of a) in
  claim2 a b;
  (match K.binary k a b ~dst with
  | () -> release2 a b
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release2 a b;
      Printexc.raise_with_backtrace e bt);
  dst

let k_compare (e : env) k a b =
  let (module K) = e.kernels in
  let dst = e.alloc Nx_dtype.Bool (shape_of a) in
  claim2 a b;
  (match K.compare k a b ~dst with
  | () -> release2 a b
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release2 a b;
      Printexc.raise_with_backtrace e bt);
  dst

let k_where (e : env) c a b =
  let (module K) = e.kernels in
  let dst = e.alloc a.dtype (shape_of a) in
  claim3 c a b;
  (match K.where c a b ~dst with
  | () -> release3 c a b
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release3 c a b;
      Printexc.raise_with_backtrace e bt);
  dst

let k_reduce (e : env) k axes a =
  let (module K) = e.kernels in
  let dst =
    e.alloc a.dtype (Shape.reduce_output_shape (shape_of a) axes false)
  in
  claim a;
  (match K.reduce k ~axes a ~dst with
  | () -> release a
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release a;
      Printexc.raise_with_backtrace e bt);
  dst

let k_scan (e : env) k axis a =
  let (module K) = e.kernels in
  let dst = e.alloc a.dtype (shape_of a) in
  claim a;
  (match K.scan k ~axis a ~dst with
  | () -> release a
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release a;
      Printexc.raise_with_backtrace e bt);
  dst

let k_arg_reduce (e : env) k axis a =
  let (module K) = e.kernels in
  let dst =
    e.alloc Nx_dtype.Int64
      (Shape.reduce_output_shape (shape_of a) [| axis |] false)
  in
  claim a;
  (match K.arg_reduce k ~axis a ~dst with
  | () -> release a
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release a;
      Printexc.raise_with_backtrace e bt);
  dst

let k_sort (e : env) descending axis a =
  let (module K) = e.kernels in
  let dst = e.alloc a.dtype (shape_of a) in
  claim a;
  (match K.sort ~descending ~axis a ~dst with
  | () -> release a
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release a;
      Printexc.raise_with_backtrace e bt);
  dst

let k_argsort (e : env) descending axis a =
  let (module K) = e.kernels in
  let dst = e.alloc Nx_dtype.Int64 (shape_of a) in
  claim a;
  (match K.argsort ~descending ~axis a ~dst with
  | () -> release a
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release a;
      Printexc.raise_with_backtrace e bt);
  dst

let k_pad (e : env) padding v a =
  let (module K) = e.kernels in
  let dst = e.alloc a.dtype (pad_shape padding (shape_of a)) in
  claim a;
  (match K.pad padding v a ~dst with
  | () -> release a
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release a;
      Printexc.raise_with_backtrace e bt);
  dst

let k_cat (e : env) axis xs =
  let (module K) = e.kernels in
  let shape = cat_shape axis (List.map shape_of xs) in
  let dst = e.alloc (List.hd xs).dtype shape in
  claim_all xs;
  (match K.cat ~axis xs ~dst with
  | () -> release_all xs
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release_all xs;
      Printexc.raise_with_backtrace e bt);
  dst

let k_cast (e : env) dtype a =
  let (module K) = e.kernels in
  let dst = e.alloc dtype (shape_of a) in
  claim a;
  (match K.cast a ~dst with
  | () -> release a
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release a;
      Printexc.raise_with_backtrace e bt);
  dst

let k_threefry (e : env) key ctr =
  let (module K) = e.kernels in
  let dst = e.alloc Nx_dtype.Int32 (shape_of ctr) in
  claim2 key ctr;
  (match K.threefry key ctr ~dst with
  | () -> release2 key ctr
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release2 key ctr;
      Printexc.raise_with_backtrace e bt);
  dst

let k_gather (e : env) axis indices data =
  let (module K) = e.kernels in
  let dst = e.alloc data.dtype (shape_of indices) in
  claim2 indices data;
  (match K.gather ~axis indices data ~dst with
  | () -> release2 indices data
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release2 indices data;
      Printexc.raise_with_backtrace e bt);
  dst

let k_scatter (e : env) mode unique axis indices updates into =
  let (module K) = e.kernels in
  let dst = e.alloc into.dtype (shape_of into) in
  claim3 indices updates into;
  (match K.scatter ~mode ~unique ~axis ~indices ~updates into ~dst with
  | () -> release3 indices updates into
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release3 indices updates into;
      Printexc.raise_with_backtrace e bt);
  dst

let k_update (e : env) a starts v =
  let (module K) = e.kernels in
  let dst = e.alloc a.dtype (shape_of a) in
  claim3 a starts v;
  (match K.update a ~starts v ~dst with
  | () -> release3 a starts v
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release3 a starts v;
      Printexc.raise_with_backtrace e bt);
  dst

let k_unfold (e : env) kernel_size stride dilation padding a =
  let (module K) = e.kernels in
  let dst =
    e.alloc a.dtype
      (unfold_shape kernel_size stride dilation padding (shape_of a))
  in
  claim a;
  (match K.unfold ~kernel_size ~stride ~dilation ~padding a ~dst with
  | () -> release a
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release a;
      Printexc.raise_with_backtrace e bt);
  dst

let k_fold (e : env) output_size kernel_size stride dilation
    padding a =
  let (module K) = e.kernels in
  let dst = e.alloc a.dtype (fold_shape output_size (shape_of a)) in
  claim a;
  (match K.fold ~output_size ~kernel_size ~stride ~dilation ~padding a ~dst with
  | () -> release a
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release a;
      Printexc.raise_with_backtrace e bt);
  dst

let k_matmul (e : env) a b =
  let (module K) = e.kernels in
  let dst = e.alloc a.dtype (matmul_shape (shape_of a) (shape_of b)) in
  claim2 a b;
  (match K.matmul a b ~dst with
  | () -> release2 a b
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release2 a b;
      Printexc.raise_with_backtrace e bt);
  dst

let k_fft (e : env) inverse axes a =
  let (module K) = e.kernels in
  let dst = e.alloc a.dtype (shape_of a) in
  claim a;
  (match K.fft ~inverse ~axes a ~dst with
  | () -> release a
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release a;
      Printexc.raise_with_backtrace e bt);
  dst

let k_rfft (e : env) dtype axes a =
  let (module K) = e.kernels in
  let dst = e.alloc dtype (rfft_shape axes (shape_of a)) in
  claim a;
  (match K.rfft ~axes a ~dst with
  | () -> release a
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release a;
      Printexc.raise_with_backtrace e bt);
  dst

let k_irfft (e : env) dtype axes s a =
  let (module K) = e.kernels in
  let dst = e.alloc dtype (irfft_shape axes s (shape_of a)) in
  claim a;
  (match K.irfft ~axes ~s a ~dst with
  | () -> release a
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release a;
      Printexc.raise_with_backtrace e bt);
  dst

let k_contiguous (e : env) a =
  let (module K) = e.kernels in
  let dst = e.alloc a.dtype (shape_of a) in
  claim a;
  (match K.contiguous a ~dst with
  | () -> release a
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release a;
      Printexc.raise_with_backtrace e bt);
  dst

(* [bitcast_array e dtype a] is [a]'s bytes read as elements of [dtype]: at
   [a]'s width, the same view; [k] times narrower, each element as [k] along a
   new last axis; [k] times wider, each run of [k] along the last axis, of [k],
   as one element: a view of [a]'s memory when [a] is C-contiguous from a first
   element aligned to [dtype]'s width, and of a C-contiguous copy of [a] that
   [e] makes otherwise. *)
let bitcast_array (type a b c d) (e : env) (dtype : (c, d) Nx_dtype.t)
    (a : (a, b) Nx_array.t) : (c, d) Nx_array.t =
  let w = Nx_dtype.itemsize a.dtype and w' = Nx_dtype.itemsize dtype in
  let over (a : (a, b) Nx_array.t) view : (c, d) Nx_array.t =
    let n = Nx_device.Buffer.nbytes a.buffer / w' in
    let scalar = Nx_dtype.Scalar.of_dtype dtype in
    { dtype; view; buffer = Nx_device.Buffer.view a.buffer ~offset:0 scalar n }
  in
  let v = a.view in
  if w' = w then over a v
  else if w' < w then
    let k = w / w' in
    over a
      (View.create
         ~offset:(View.offset v * k)
         ~strides:(Array.append (Array.map (( * ) k) (View.strides v)) [| 1 |])
         (Array.append (View.shape v) [| k |]))
  else
    let k = w' / w in
    let first = View.offset v * w in
    let aligned =
      Nativeint.(
        rem (add (Nx_device.Buffer.address a.buffer) (of_int first)) (of_int w'))
      = 0n
    in
    let a, first =
      if View.is_c_contiguous v && aligned then (a, first)
      else (k_contiguous e a, 0)
    in
    let s = View.shape a.view in
    {
      dtype;
      view = View.create (Array.sub s 0 (Array.length s - 1));
      buffer =
        Nx_device.Buffer.view a.buffer ~offset:first
          (Nx_dtype.Scalar.of_dtype dtype)
          (View.numel a.view / k);
    }

let k_cholesky (e : env) upper a =
  let (module K) = e.kernels in
  let dst = e.alloc a.dtype (shape_of a) in
  claim a;
  (match K.cholesky ~upper a ~dst with
  | () -> release a
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release a;
      Printexc.raise_with_backtrace e bt);
  dst

(* The batch axes of a matrix of shape [..., m, n], and [m] and [n]. *)
let matrix a =
  let s = shape_of a in
  let r = Array.length s in
  (Array.sub s 0 (r - 2), s.(r - 2), s.(r - 1))

let k_qr (e : env) reduced a =
  let (module K) = e.kernels in
  let batch, m, n = matrix a in
  let k = Int.min m n in
  let dims r c = Array.append batch [| r; c |] in
  let q = e.alloc a.dtype (if reduced then dims m k else dims m m) in
  let r = e.alloc a.dtype (if reduced then dims k n else dims m n) in
  claim a;
  (match K.qr ~reduced a ~q ~r with
  | () -> release a
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release a;
      Printexc.raise_with_backtrace e bt);
  (q, r)

let k_lu (e : env) a =
  let (module K) = e.kernels in
  let batch, m, n = matrix a in
  let lu = e.alloc a.dtype (shape_of a) in
  let pivots = e.alloc Nx_dtype.Int64 (Array.append batch [| Int.min m n |]) in
  let perm = e.alloc Nx_dtype.Int64 (Array.append batch [| m |]) in
  claim a;
  (match K.lu a ~lu ~pivots ~perm with
  | () -> release a
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release a;
      Printexc.raise_with_backtrace e bt);
  (lu, pivots, perm)

let k_svd (e : env) full_matrices a =
  let (module K) = e.kernels in
  let batch, m, n = matrix a in
  let k = Int.min m n in
  let dims r c = Array.append batch [| r; c |] in
  let u = e.alloc a.dtype (if full_matrices then dims m m else dims m k) in
  let s = e.alloc Nx_dtype.Float64 (Array.append batch [| k |]) in
  let vt = e.alloc a.dtype (if full_matrices then dims n n else dims k n) in
  claim a;
  (match K.svd a ~u ~s ~vt with
  | () -> release a
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release a;
      Printexc.raise_with_backtrace e bt);
  (u, s, vt)

let k_eig (e : env) vectors a =
  let (module K) = e.kernels in
  let batch, _, n = matrix a in
  let values = e.alloc Nx_dtype.Complex128 (Array.append batch [| n |]) in
  let vectors =
    if vectors then Some (e.alloc Nx_dtype.Complex128 (shape_of a)) else None
  in
  claim a;
  (match K.eig a ~values ~vectors with
  | () -> release a
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release a;
      Printexc.raise_with_backtrace e bt);
  (values, vectors)

let k_eigh (e : env) vectors a =
  let (module K) = e.kernels in
  let batch, _, n = matrix a in
  let values = e.alloc Nx_dtype.Float64 (Array.append batch [| n |]) in
  let vectors = if vectors then Some (e.alloc a.dtype (shape_of a)) else None in
  claim a;
  (match K.eigh a ~values ~vectors with
  | () -> release a
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release a;
      Printexc.raise_with_backtrace e bt);
  (values, vectors)

let k_solve_triangular (e : env) upper transpose unit_diag a b =
  let (module K) = e.kernels in
  let dst = e.alloc b.dtype (shape_of b) in
  claim2 a b;
  (match K.solve_triangular ~upper ~transpose ~unit_diag a b ~dst with
  | () -> release2 a b
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      release2 a b;
      Printexc.raise_with_backtrace e bt);
  dst

let all_host xs = List.for_all (function Host _ -> true | _ -> false) xs

(* Dispatch to devices *)

(* How each device's results become the operation's: [settle] makes a value of
   one result's arrays, one per device. *)
type settle = { settle : 'a 'b. ('a, 'b) Nx_array.t list -> ('a, 'b) t }

let host_settle =
  {
    settle =
      (function
      | [ a ] -> Host a | _ -> invalid_arg "Nx_effect: one host result");
  }

(* [compute envs s op] is [op] computed where each of [envs] says, its results
   made by [s]. *)
let compute : type r. env list -> settle -> r Op.t -> r =
 fun envs s op ->
  let each f = s.settle (List.map f envs) in
  match[@warning "@4@8"] op with
  | Unary (k, x) -> each (fun e -> k_unary e k (e.arr x))
  | Binary (k, a, b) -> each (fun e -> k_binary e k (e.arr a) (e.arr b))
  | Compare (k, a, b) -> each (fun e -> k_compare e k (e.arr a) (e.arr b))
  | Where (c, a, b) -> each (fun e -> k_where e (e.arr c) (e.arr a) (e.arr b))
  | Reduce (k, axes, x) -> each (fun e -> k_reduce e k axes (e.arr x))
  | Scan (k, axis, x) -> each (fun e -> k_scan e k axis (e.arr x))
  | Arg_reduce (k, axis, x) -> each (fun e -> k_arg_reduce e k axis (e.arr x))
  | Sort { descending; axis; x } ->
      each (fun e -> k_sort e descending axis (e.arr x))
  | Argsort { descending; axis; x } ->
      each (fun e -> k_argsort e descending axis (e.arr x))
  | Pad (padding, v, x) -> each (fun e -> k_pad e padding v (e.arr x))
  | Cat (axis, xs) -> each (fun e -> k_cat e axis (List.map e.arr xs))
  | Convert (Cast, dtype, x) -> each (fun e -> k_cast e dtype (e.arr x))
  | Convert (Bitcast, dtype, x) ->
      each (fun e -> bitcast_array e dtype (e.arr x))
  | Threefry (key, ctr) -> each (fun e -> k_threefry e (e.arr key) (e.arr ctr))
  | Gather (axis, indices, data) ->
      each (fun e -> k_gather e axis (e.arr indices) (e.arr data))
  | Scatter { mode; unique; axis; indices; updates; into } ->
      each (fun e ->
          k_scatter e mode unique axis (e.arr indices) (e.arr updates)
            (e.arr into))
  | Update (x, starts, v) ->
      each (fun e -> k_update e (e.arr x) (e.arr starts) (e.arr v))
  | Unfold { kernel_size; stride; dilation; padding; x } ->
      each (fun e -> k_unfold e kernel_size stride dilation padding (e.arr x))
  | Fold { output_size; kernel_size; stride; dilation; padding; x } ->
      each (fun e ->
          k_fold e output_size kernel_size stride dilation padding (e.arr x))
  | Matmul (a, b) -> each (fun e -> k_matmul e (e.arr a) (e.arr b))
  | Fft { inverse; axes; x } -> each (fun e -> k_fft e inverse axes (e.arr x))
  | Rfft { dtype; axes; x } -> each (fun e -> k_rfft e dtype axes (e.arr x))
  | Irfft { dtype; axes; s = sizes; x } ->
      each (fun e -> k_irfft e dtype axes sizes (e.arr x))
  | Contiguous x -> each (fun e -> k_contiguous e (e.arr x))
  | Cholesky { upper; x } -> each (fun e -> k_cholesky e upper (e.arr x))
  | Qr { reduced; x } ->
      let rs = List.map (fun e -> k_qr e reduced (e.arr x)) envs in
      (s.settle (List.map fst rs), s.settle (List.map snd rs))
  | Lu x ->
      let rs = List.map (fun e -> k_lu e (e.arr x)) envs in
      ( s.settle (List.map (fun (a, _, _) -> a) rs),
        s.settle (List.map (fun (_, a, _) -> a) rs),
        s.settle (List.map (fun (_, _, a) -> a) rs) )
  | Svd { full_matrices; x } ->
      let rs = List.map (fun e -> k_svd e full_matrices (e.arr x)) envs in
      ( s.settle (List.map (fun (a, _, _) -> a) rs),
        s.settle (List.map (fun (_, a, _) -> a) rs),
        s.settle (List.map (fun (_, _, a) -> a) rs) )
  | Eig { vectors; x } ->
      let rs = List.map (fun e -> k_eig e vectors (e.arr x)) envs in
      ( s.settle (List.map fst rs),
        if vectors then
          Some (s.settle (List.map (fun (_, v) -> Option.get v) rs))
        else None )
  | Eigh { vectors; x } ->
      let rs = List.map (fun e -> k_eigh e vectors (e.arr x)) envs in
      ( s.settle (List.map fst rs),
        if vectors then
          Some (s.settle (List.map (fun (_, v) -> Option.get v) rs))
        else None )
  | Solve_triangular { upper; transpose; unit_diag; a; b } ->
      each (fun e ->
          k_solve_triangular e upper transpose unit_diag (e.arr a) (e.arr b))
  | Move _ | Place _ | Read _ ->
      invalid_arg "Nx_effect.compute: the operation computes nothing"

type mapper = { f : 'a 'b. ('a, 'b) t -> ('a, 'b) t }

(* [with_operands o op] is [op] over [o.f] of each of its value operands. *)
let with_operands : type r. mapper -> r Op.t -> r Op.t =
 fun o op ->
  let f = o.f in
  match[@warning "@4@8"] op with
  | Unary (k, x) -> Unary (k, f x)
  | Binary (k, a, b) -> Binary (k, f a, f b)
  | Compare (k, a, b) -> Compare (k, f a, f b)
  | Where (c, a, b) -> Where (f c, f a, f b)
  | Reduce (k, axes, x) -> Reduce (k, axes, f x)
  | Scan (k, axis, x) -> Scan (k, axis, f x)
  | Arg_reduce (k, axis, x) -> Arg_reduce (k, axis, f x)
  | Sort s -> Sort { s with x = f s.x }
  | Argsort s -> Argsort { s with x = f s.x }
  | Pad (padding, v, x) -> Pad (padding, v, f x)
  | Cat (axis, xs) -> Cat (axis, List.map f xs)
  | Convert (c, dtype, x) -> Convert (c, dtype, f x)
  | Threefry (key, ctr) -> Threefry (f key, f ctr)
  | Gather (axis, indices, data) -> Gather (axis, f indices, f data)
  | Scatter s ->
      Scatter
        { s with indices = f s.indices; updates = f s.updates; into = f s.into }
  | Update (x, starts, v) -> Update (f x, f starts, f v)
  | Unfold u -> Unfold { u with x = f u.x }
  | Fold u -> Fold { u with x = f u.x }
  | Matmul (a, b) -> Matmul (f a, f b)
  | Fft t -> Fft { t with x = f t.x }
  | Rfft t -> Rfft { t with x = f t.x }
  | Irfft t -> Irfft { t with x = f t.x }
  | Contiguous x -> Contiguous (f x)
  | Cholesky c -> Cholesky { c with x = f c.x }
  | Qr q -> Qr { q with x = f q.x }
  | Lu x -> Lu (f x)
  | Svd d -> Svd { d with x = f d.x }
  | Eig d -> Eig { d with x = f d.x }
  | Eigh d -> Eigh { d with x = f d.x }
  | Solve_triangular t -> Solve_triangular { t with a = f t.a; b = f t.b }
  | Move (x, m) -> Move (f x, m)
  | Place (p, x) -> Place (p, f x)
  | Read r -> Read { r with x = f r.x }

(* [x]'s array on [d]: its storage there, through its view. *)
let local (type a b) d (x : (a, b) t) : (a, b) Nx_array.t =
  match x with
  | Placed r -> (
      match Cell.state r.r_cell with
      | Live bufs ->
          let holders = devices_of r.r_cell.placement in
          let i =
            match List.find_index (Nx_device.equal d) holders with
            | Some i -> i
            | None -> assert false (* moved to the target's devices *)
          in
          { dtype = r.r_dtype; view = r.r_view; buffer = List.nth bufs i }
      | Consumed k -> consumed k)
  | Host _ -> invalid_arg "Nx_effect.local: a host value has no device array"
  | Traced _ -> outside_trace ()

(* Computing on [d] with [backend]: results in [d]'s memory. *)
let env_on backend d =
  let alloc (type a b) (dtype : (a, b) Nx_dtype.t) shape : (a, b) Nx_array.t =
    let n = Array.fold_left ( * ) 1 shape in
    let s = Nx_dtype.Scalar.of_dtype dtype in
    { dtype; view = View.create shape; buffer = Nx_device.Buffer.create d s n }
  in
  { kernels = Nx_backend.kernels backend; alloc; arr = (fun x -> local d x) }

(* The results of every device at [q], each the whole result, or, when
   [windowed], each device's window of the whole result copied out on it. *)
let settle_on q ~windowed envs =
  let ds = devices_of q in
  let window e d a =
    check_shape "Nx" q (shape_of a);
    let w = window_of q (shape_of a) d in
    k_contiguous e { a with view = View.shrink a.view w }
  in
  {
    settle =
      (fun arrays ->
        let arrays =
          if windowed then
            List.map2 (fun (e, d) a -> window e d a) (List.combine envs ds)
              arrays
          else arrays
        in
        let a = List.hd arrays in
        placed q a.dtype
          (View.create (shape_of a))
          (cell ~placement:q
             ~length:(Nx_device.Buffer.length a.buffer)
             (List.map (fun (a : (_, _) Nx_array.t) -> a.buffer) arrays)));
  }

(* Whether each device can compute its tile of the result at [q] from its tiles
   of the operands, split alike: a gather reads its data whole along its
   axis. *)
let per_tile q = function
  | Elementwise | Along _ | Reduce _ -> true
  | Gather axis -> not (List.mem_assoc axis (Grid.cuts q.grid))
  | Contract | Into -> false

let refuse op backend d =
  raise
    (Nx_backend.Refused
       (Printf.sprintf
          "%s: %s does not compute on %s; place with a backend that runs on \
           %s, or compute under a compiled call"
          op (Nx_backend.name backend) (Nx_device.name d) (Nx_device.name d)))

let rec with_cells cells f =
  match cells with
  | [] -> f ()
  | c :: rest -> Cell.with_borrow c (fun () -> with_cells rest f)

let cells_of op =
  List.fold_left
    (fun cells (P x) ->
      match x with
      | Placed r when not (List.memq r.r_cell cells) -> r.r_cell :: cells
      | _ -> cells)
    [] (Op.operands op)

(* [on_devices op] is [op] where it runs: nx.cpu over host operands, and at a
   placement [q] its backend once per device of [q]. Each device holds its
   tiles of the operands when the result is split and the operation keeps tiles
   apart, and whole copies of them otherwise, which nx copies there first; a
   split result is then each device's window of the whole it computed. A
   contraction split along an outer axis of its left operand is computed whole
   on each device too, N times the work of a tile each: one rule serves every
   split contraction and scatter until a consumer needs the tiles. *)
let on_devices : type r. r Op.t -> r =
 fun op ->
  match routing op with
  | Computes (rule, xs) -> (
      match route (fun (P x) -> placement_of x) (Op.name op) rule xs with
      | On_host -> compute [ host_env ] host_settle op
      | At q ->
          let ds = devices_of q in
          List.iter
            (fun d ->
              if not (Nx_backend.runs_on q.backend d) then
                refuse (Op.name op) q.backend d)
            ds;
          let split =
            List.find_map
              (fun (P x) ->
                match placement_of x with
                | Some p when Grid.cuts p.grid <> [] -> Some p.grid
                | _ -> None)
              xs
          in
          let cut = Grid.cuts q.grid <> [] in
          let target, windowed =
            match split with
            | Some g when cut && per_tile q rule -> ({ q with grid = g }, false)
            | _ ->
                let whole =
                  List.fold_left
                    (fun g (a, _) -> Grid.uncut g ~axis:a)
                    q.grid (Grid.cuts q.grid)
                in
                ({ q with grid = whole }, cut)
          in
          let op = with_operands { f = (fun x -> move_to target x) } op in
          let envs = List.map (env_on q.backend) ds in
          with_cells (cells_of op) (fun () ->
              compute envs (settle_on q ~windowed envs) op))
  | Moves _ | Places _ | Reads _ ->
      invalid_arg "Nx_effect.on_devices: the operation computes nothing"

(* Each operation with no interpretation: nx.cpu on host operands, with no
   closure and no operation built, and [on_devices] otherwise. *)

let direct_unary k x =
  match x with
  | Host a -> Host (k_unary host_env k a)
  | _ -> on_devices (Unary (k, x))

let direct_binary k x y =
  match (x, y) with
  | Host a, Host b -> Host (k_binary host_env k a b)
  | _ -> on_devices (Binary (k, x, y))

let direct_compare k x y =
  match (x, y) with
  | Host a, Host b -> Host (k_compare host_env k a b)
  | _ -> on_devices (Compare (k, x, y))

let direct_where c x y =
  match (c, x, y) with
  | Host c', Host a, Host b -> Host (k_where host_env c' a b)
  | _ -> on_devices (Where (c, x, y))

let direct_reduce k axes x =
  match x with
  | Host a -> Host (k_reduce host_env k axes a)
  | _ -> on_devices (Reduce (k, axes, x))

let direct_matmul x y =
  match (x, y) with
  | Host a, Host b -> Host (k_matmul host_env a b)
  | _ -> on_devices (Matmul (x, y))

let direct_copy x =
  match x with
  | Host a -> Host (k_contiguous host_env a)
  | _ -> on_devices (Contiguous x)

let direct_scan k axis x =
  match x with
  | Host a -> Host (k_scan host_env k axis a)
  | _ -> on_devices (Scan (k, axis, x))

let direct_arg_reduce k axis x =
  match x with
  | Host a -> Host (k_arg_reduce host_env k axis a)
  | _ -> on_devices (Arg_reduce (k, axis, x))

let direct_sort descending axis x =
  match x with
  | Host a -> Host (k_sort host_env descending axis a)
  | _ -> on_devices (Sort { descending; axis; x })

let direct_argsort descending axis x =
  match x with
  | Host a -> Host (k_argsort host_env descending axis a)
  | _ -> on_devices (Argsort { descending; axis; x })

let direct_pad padding v x =
  match x with
  | Host a -> Host (k_pad host_env padding v a)
  | _ -> on_devices (Pad (padding, v, x))

let direct_cat axis xs =
  if all_host xs then Host (k_cat host_env axis (List.map host_of xs))
  else on_devices (Cat (axis, xs))

let direct_convert (type a b c d) (c : conversion)
    (dtype : (c, d) Nx_dtype.t) (x : (a, b) t) : (c, d) t =
  match (c, x) with
  | Cast, Host a -> Host (k_cast host_env dtype a)
  | Bitcast, Host a -> Host (bitcast_array host_env dtype a)
  | _ -> on_devices (Convert (c, dtype, x))

let direct_threefry key ctr =
  match (key, ctr) with
  | Host k, Host c -> Host (k_threefry host_env k c)
  | _ -> on_devices (Threefry (key, ctr))

let direct_gather axis indices data =
  match (data, indices) with
  | Host d, Host i -> Host (k_gather host_env axis i d)
  | _ -> on_devices (Gather (axis, indices, data))

let direct_scatter mode unique axis indices updates into =
  match (into, indices, updates) with
  | Host d, Host i, Host u -> Host (k_scatter host_env mode unique axis i u d)
  | _ -> on_devices (Scatter { mode; unique; axis; indices; updates; into })

let direct_update x starts v =
  match (x, starts, v) with
  | Host a, Host s, Host w -> Host (k_update host_env a s w)
  | _ -> on_devices (Update (x, starts, v))

let direct_unfold kernel_size stride dilation padding x =
  match x with
  | Host a -> Host (k_unfold host_env kernel_size stride dilation padding a)
  | _ -> on_devices (Unfold { kernel_size; stride; dilation; padding; x })

let direct_fold output_size kernel_size stride dilation padding x =
  match x with
  | Host a ->
      Host (k_fold host_env output_size kernel_size stride dilation padding a)
  | _ ->
      on_devices
        (Fold { output_size; kernel_size; stride; dilation; padding; x })

let direct_fft inverse axes x =
  match x with
  | Host a -> Host (k_fft host_env inverse axes a)
  | _ -> on_devices (Fft { inverse; axes; x })

let direct_rfft dtype axes x =
  match x with
  | Host a -> Host (k_rfft host_env dtype axes a)
  | _ -> on_devices (Rfft { dtype; axes; x })

let direct_irfft dtype axes s x =
  match x with
  | Host a -> Host (k_irfft host_env dtype axes s a)
  | _ -> on_devices (Irfft { dtype; axes; s; x })

let direct_cholesky upper x =
  match x with
  | Host a -> Host (k_cholesky host_env upper a)
  | _ -> on_devices (Cholesky { upper; x })

let direct_solve_triangular upper transpose unit_diag a b =
  match (a, b) with
  | Host x, Host y ->
      Host (k_solve_triangular host_env upper transpose unit_diag x y)
  | _ -> on_devices (Solve_triangular { upper; transpose; unit_diag; a; b })

(* [direct op] answers [op] with no interpretation. The decompositions run
   through [on_devices] even on the host, which settles each result. *)
let direct : type r. r Op.t -> r =
 fun op ->
  match[@warning "@4@8"] op with
  | Unary (k, x) -> direct_unary k x
  | Binary (k, x, y) -> direct_binary k x y
  | Compare (k, x, y) -> direct_compare k x y
  | Where (c, x, y) -> direct_where c x y
  | Reduce (k, axes, x) -> direct_reduce k axes x
  | Scan (k, axis, x) -> direct_scan k axis x
  | Arg_reduce (k, axis, x) -> direct_arg_reduce k axis x
  | Sort { descending; axis; x } -> direct_sort descending axis x
  | Argsort { descending; axis; x } -> direct_argsort descending axis x
  | Pad (padding, v, x) -> direct_pad padding v x
  | Cat (axis, xs) -> direct_cat axis xs
  | Convert (c, dtype, x) -> direct_convert c dtype x
  | Threefry (key, ctr) -> direct_threefry key ctr
  | Gather (axis, indices, data) -> direct_gather axis indices data
  | Scatter { mode; unique; axis; indices; updates; into } ->
      direct_scatter mode unique axis indices updates into
  | Update (x, starts, v) -> direct_update x starts v
  | Unfold { kernel_size; stride; dilation; padding; x } ->
      direct_unfold kernel_size stride dilation padding x
  | Fold { output_size; kernel_size; stride; dilation; padding; x } ->
      direct_fold output_size kernel_size stride dilation padding x
  | Matmul (x, y) -> direct_matmul x y
  | Fft { inverse; axes; x } -> direct_fft inverse axes x
  | Rfft { dtype; axes; x } -> direct_rfft dtype axes x
  | Irfft { dtype; axes; s; x } -> direct_irfft dtype axes s x
  | Contiguous x -> direct_copy x
  | Cholesky { upper; x } -> direct_cholesky upper x
  | Qr _ | Lu _ | Svd _ | Eig _ | Eigh _ -> on_devices op
  | Solve_triangular { upper; transpose; unit_diag; a; b } ->
      direct_solve_triangular upper transpose unit_diag a b
  | Move (x, m) -> moved x m
  | Place (p, x) -> move_to p x
  | Read { x; _ } -> read_elements_of x


(* Interception

   [intercept i f] runs [f] with every operation its fiber performs and [i]
   claims delivered to [i.run], which runs outside [f]'s handlers: the
   operations [i.run] issues reach the enclosing interpretation. An operation
   [i] does not claim reaches the enclosing interpretation as performed. The
   gate is raised for the extent of [f], however it ends. *)

type interpreter = { run : 'r. 'r Op.t -> 'r; claims : 'r. 'r Op.t -> bool }
type _ Effect.t += E_op : 'r Op.t -> 'r Effect.t | E_intercepted : bool Effect.t

let intercept i f =
  Atomic.incr intercepts;
  Fun.protect ~finally:(fun () -> Atomic.decr intercepts) @@ fun () ->
  let effc : type c a.
      c Effect.t -> ((c, a) Effect.Deep.continuation -> a) option = function
    | E_op op -> (
        match i.claims op with
        | false -> None
        | true ->
            Some
              (fun k ->
                match i.run op with
                | v -> Effect.Deep.continue k v
                | exception e ->
                    let bt = Printexc.get_raw_backtrace () in
                    Effect.Deep.discontinue_with_backtrace k e bt)
        | exception e ->
            let bt = Printexc.get_raw_backtrace () in
            Some (fun k -> Effect.Deep.discontinue_with_backtrace k e bt))
    | E_intercepted -> Some (fun k -> Effect.Deep.continue k true)
    | _ -> None
  in
  Effect.Deep.match_with f () { retc = Fun.id; exnc = raise; effc }

(* Whether the calling fiber is inside an interception, outside its [run]. *)
let intercepted () =
  intercepting ()
  &&
  let e = E_intercepted in
  match Effect.perform e with
  | b -> b
  | exception Effect.Unhandled e' when Obj.repr e' == Obj.repr e -> false

(* [perform op] delivers [op] to the interception around the caller, and
   answers it directly when there is none: only an unhandled perform of this
   very effect falls back. *)
let perform : type r. r Op.t -> r =
 fun op ->
  let e = E_op op in
  match Effect.perform e with
  | v -> v
  | exception Effect.Unhandled e' when Obj.repr e' == Obj.repr e -> direct op

(* [eval op] is [op] in the current interpretation. *)
let eval op = if intercepting () then perform op else direct op

(* Entry functions, one per constructor. While the gate is down, each answers
   directly and builds no operation. *)

let unary k x =
  if intercepting () then perform (Unary (k, x)) else direct_unary k x

let binary k x y =
  if intercepting () then perform (Binary (k, x, y)) else direct_binary k x y

let cmp k x y =
  if intercepting () then perform (Compare (k, x, y)) else direct_compare k x y

let where c x y =
  if intercepting () then perform (Where (c, x, y)) else direct_where c x y

let reduce k ~axes x =
  if intercepting () then perform (Reduce (k, axes, x))
  else direct_reduce k axes x

let scan k ~axis x =
  if intercepting () then perform (Scan (k, axis, x)) else direct_scan k axis x

let arg_reduce k ~axis x =
  if intercepting () then perform (Arg_reduce (k, axis, x))
  else direct_arg_reduce k axis x

let sort ~descending ~axis x =
  if intercepting () then perform (Sort { descending; axis; x })
  else direct_sort descending axis x

let argsort ~descending ~axis x =
  if intercepting () then perform (Argsort { descending; axis; x })
  else direct_argsort descending axis x

let pad padding v x =
  if intercepting () then perform (Pad (padding, v, x))
  else direct_pad padding v x

let cat ~axis xs =
  if intercepting () then perform (Cat (axis, xs)) else direct_cat axis xs

let cast dtype x =
  if intercepting () then perform (Convert (Cast, dtype, x))
  else direct_convert Cast dtype x

let bitcast dtype x =
  if intercepting () then perform (Convert (Bitcast, dtype, x))
  else direct_convert Bitcast dtype x

let threefry key ctr =
  if intercepting () then perform (Threefry (key, ctr))
  else direct_threefry key ctr

let gather ~axis indices x =
  if intercepting () then perform (Gather (axis, indices, x))
  else direct_gather axis indices x

let scatter ~mode ~unique ~axis ~indices ~updates into =
  if intercepting () then
    perform (Scatter { mode; unique; axis; indices; updates; into })
  else direct_scatter mode unique axis indices updates into

let update x ~starts v =
  if intercepting () then perform (Update (x, starts, v))
  else direct_update x starts v

let unfold ~kernel_size ~stride ~dilation ~padding x =
  if intercepting () then
    perform (Unfold { kernel_size; stride; dilation; padding; x })
  else direct_unfold kernel_size stride dilation padding x

let fold ~output_size ~kernel_size ~stride ~dilation ~padding x =
  if intercepting () then
    perform (Fold { output_size; kernel_size; stride; dilation; padding; x })
  else direct_fold output_size kernel_size stride dilation padding x

let matmul x y =
  if intercepting () then perform (Matmul (x, y)) else direct_matmul x y

let fft ~inverse ~axes x =
  if intercepting () then perform (Fft { inverse; axes; x })
  else direct_fft inverse axes x

let rfft dtype ~axes x =
  if intercepting () then perform (Rfft { dtype; axes; x })
  else direct_rfft dtype axes x

let irfft ?s dtype ~axes x =
  if intercepting () then perform (Irfft { dtype; axes; s; x })
  else direct_irfft dtype axes s x

let cholesky ~upper x =
  if intercepting () then perform (Cholesky { upper; x })
  else direct_cholesky upper x

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
  if intercepting () then
    perform (Solve_triangular { upper; transpose; unit_diag; a; b })
  else direct_solve_triangular upper transpose unit_diag a b

let move x m =
  if intercepting () then perform (Move (x, m)) else moved x m
let reshape x shape = move x (Reshape shape)
let expand x shape = move x (Expand shape)
let permute x axes = move x (Permute axes)
let shrink x limits = move x (Shrink limits)
let flip x dims = move x (Flip dims)

let sliding_window x ~axis ~window ~step =
  move x (Window { axis; size = window; step })

(* The elements of [x]'s view in C order, in a host buffer, read by the surface
   function [by]. The storage of a host value that is contiguous from its first
   element is that buffer. *)
let read ~by x =
  if intercepting () then perform (Read { by; x }) else read_elements_of x

(* A value already at [p] is returned as it is. *)
let place (type a b) p (x : (a, b) t) : (a, b) t =
  if Placement.equal (placement x) p then
    match x with
    | Placed r -> Cell.with_borrow r.r_cell (fun () -> x)
    | Host _ | Traced _ -> x
  else if intercepting () then perform (Place (p, x))
  else move_to p x

(* [copy x] is [x] in storage of its own, C-contiguous from its first element:
   it always copies. A traced value has no bytes: the interpretation that made
   it answers its copy. [contiguous x] is [x] itself when its view, a traced
   value's too, is already C-contiguous from its first element. *)
let copy x = if intercepting () then perform (Contiguous x) else direct_copy x

let contiguous x =
  let v = view x in
  if View.is_c_contiguous v && View.offset v = 0 then x else copy x

(* Creation. A constant is not an operation. Uninterpreted on the host, a
   filled value is nx.cpu's fill of storage of its own. Otherwise it is one
   element on the host, placed where it is made and expanded, so that an
   interpretation sees a constant (a compiled call folds it into its kernels);
   one of more than one element is then copied into storage of its own, so
   that its view covers its storage and a compiled call can consume it. *)

let broadcast scalar shape_arr =
  if Array.length shape_arr = 0 then scalar
  else
    let ones = Array.map (fun _ -> 1) shape_arr in
    let x = reshape scalar ones in
    if Shape.equal ones shape_arr then x else expand x shape_arr

let full (ctx : context) dtype shape_arr value =
  if on_host ctx && not (intercepting ()) then
    Host (filled dtype shape_arr value)
  else
    let e = Host (filled dtype [||] value) in
    let n = Array.fold_left ( * ) 1 shape_arr in
    if on_host ctx then
      let x = broadcast e shape_arr in
      if n <= 1 then x else copy x
    else
      let copies =
        List.fold_left
          (fun p (axis, _) -> Placement.uncut p ~axis)
          ctx (Placement.cuts ctx)
      in
      let x = broadcast (place copies e) shape_arr in
      if n <= 1 then x else if copies == ctx then copy x else place ctx x

let from_host (ctx : context) dtype buffer =
  check_host "Nx_effect.from_host" dtype buffer;
  let x =
    Host
      {
        dtype;
        view = View.create [| Nx_device.Buffer.length buffer |];
        buffer;
      }
  in
  if on_host ctx then x else place ctx x

(* The buffer of exactly [x]'s elements in C order, without a copy, when there
   is one: [x] is on the host, or its storage is one runtime buffer, and its
   view is a contiguous run of its storage. *)
let run (type a b) (x : (a, b) t) =
  match x with
  | Host t -> run_in t.buffer t.view
  | Placed r -> (
      match Cell.state r.r_cell with
      | Live [ b ] -> run_in b r.r_view
      | _ -> None)
  | Traced _ -> outside_trace ()

(* Values over runtime buffers *)

(* Whether the view [v] reaches only elements [0] to [n - 1]. *)
let fits v n =
  View.numel v = 0
  ||
  let lo, hi = View.extent v in
  lo >= 0 && hi <= n

(* [shard_storage what p buffers] is the storage of [buffers], one per device of
   [p], in order, of one length, each in the memory of its device. *)
let shard_storage what p buffers =
  let ds = devices_of p in
  if List.compare_lengths ds buffers <> 0 then
    invalid_arg
      (Printf.sprintf "%s: %d buffers for %d devices" what (List.length buffers)
         (List.length ds));
  let length = Nx_device.Buffer.length (List.hd buffers) in
  List.iter2
    (fun d b ->
      if Nx_device.Buffer.length b <> length then
        invalid_arg (what ^ ": buffers of different lengths");
      if Nx_device.Buffer.device b != d then
        invalid_arg
          (Printf.sprintf "%s: a buffer for %s is on %s" what (Nx_device.name d)
             (Nx_device.name (Nx_device.Buffer.device b))))
    ds buffers;
  cell ~placement:p ~length buffers

(* [host_value what dtype view b] is the host value of [b] under [view]. *)
let host_value what dtype view b =
  check_host what dtype b;
  if not (fits view (Nx_device.Buffer.length b)) then
    invalid_arg (what ^ ": the view reaches outside the buffer");
  Host { dtype; view; buffer = b }

(* [placed_value what p dtype view c] is the value at [p] of [c] under
   [view]. *)
let placed_value what p dtype view c =
  if not (fits view c.length) then
    invalid_arg (what ^ ": the view reaches outside the storage");
  placed p dtype view c

let check_format what dtype b =
  let s = Nx_device.Buffer.dtype b in
  if not (Nx_dtype.Scalar.equal s (Nx_dtype.Scalar.of_dtype dtype)) then
    invalid_arg
      (Printf.sprintf "%s: a %s buffer read as %s" what
         (Nx_dtype.Scalar.to_string s)
         (Nx_dtype.to_string dtype))

let of_shards (type a b) p (dtype : (a, b) Nx_dtype.t) view buffers : (a, b) t =
  let what = "Nx.of_shards" in
  if is_host_placement p then
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

let shards (type a b) (x : (a, b) t) =
  match x with
  | Host a -> ([ a.buffer ], a.view)
  | Placed r -> (
      match Cell.state r.r_cell with
      | Live buffers ->
          let holders = devices_of r.r_cell.placement in
          let on d =
            List.nth buffers (Option.get (List.find_index (( == ) d) holders))
          in
          (List.map on (devices_of r.r_placement), r.r_view)
      | Consumed k -> consumed k)
  | Traced _ -> outside_trace ()

let of_buffer (type a b) ?(backend = Nx_cpu.backend) (dtype : (a, b) Nx_dtype.t)
    shape b : (a, b) t =
  let what = "Nx.of_buffer" in
  let n = Nx_device.Buffer.length b in
  if Array.fold_left ( * ) 1 shape <> n then
    invalid_arg
      (Printf.sprintf "%s: shape %s for %d elements" what
         (Shape.to_string shape) n);
  check_format what dtype b;
  let p =
    Placement.device ~backend (Nx_device.Buffer.device b)
  in
  of_shards p dtype (View.create shape) [ b ]

(* An empty buffer of [x]'s dtype on its device. *)
let empty (type a b) (x : (a, b) t) =
  let s = Nx_dtype.Scalar.of_dtype (dtype x) in
  match x with
  | Placed r when not (on_disk r.r_placement) ->
      Nx_device.Buffer.create (List.hd (devices_of r.r_placement)) s 0
  | Host _ | Placed _ | Traced _ -> Nx_device.Buffer.create Nx_device.host s 0

(* A value on the disk, which computes nothing, is copied to the host, as is one
   in memory of its own, which has no runtime buffer. *)
let to_buffer (type a b) (x : (a, b) t) =
  match x with
  | Traced _ -> invalid_arg "Nx.to_buffer: a traced value has no buffer"
  | Placed r when List.compare_length_with (devices_of r.r_placement) 1 > 0 ->
      invalid_arg
        (Format.asprintf "Nx.to_buffer: a value at %a is on several devices"
           pp_placement r.r_placement)
  | Host _ | Placed _ -> (
      match run x with
      | Some b -> b
      | None when View.numel (view x) = 0 -> empty x
      | None -> (
          let y = copy x in
          match run y with Some b -> b | None -> read ~by:"Nx.to_buffer" y))
