(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Nx_core

(* Types

   OCaml extensible GADT constructors (the [E_add], [E_mul], ... below) require
   that type variables in the payload be deducible from the return type. A
   transparent alias of [Nx_backend.t] would not be injective, so the tensor is
   a GADT of its own, whose parameters the return type determines.

   A tensor is one of three things. [Host] is a tensor of the link-time engine,
   the host's. [Placed] is a value that an engine carried by a device holds on a
   device list other than the host's: nx knows its placement, dtype and view,
   and the engine alone knows its storage. [Traced] is a node of a trace: it has
   no bytes and never will, and the tracer that made it keeps its payload in
   [t_node].

   The views of one placed storage share one cell, which holds what belongs to
   the storage rather than to a view: whether it is live or was consumed by a
   compiled call, and how many reachable programs bind it.

   A context says where a creation effect makes its value. It is declared over
   any device type, ahead of the recursive definition, so that its [Host] stays
   apart from the tensor's. *)

type 'device context_of = Host of Nx_backend.context | On of 'device list

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

type ('a, 'b) t =
  | Host : ('a, 'b) Nx_backend.t -> ('a, 'b) t
  | Placed : ('a, 'b) resident -> ('a, 'b) t
  | Traced : ('a, 'b) traced -> ('a, 'b) t

and ('a, 'b) resident = {
  r_id : int; (* fresh per value; identity tables key by it *)
  r_placement : placement; (* never the host *)
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
  t_context : device context_of;
  t_dtype : ('a, 'b) Nx_dtype.t;
  t_view : View.t; (* C-contiguous over the tensor's shape *)
  t_node : node; (* the tracer's payload *)
}

and engine = {
  read : 'a 'b. ('a, 'b) resident -> ('a, 'b) Nx_buffer.t;
      (* the view's elements, in C order, in a buffer the caller owns *)
  place : 'a 'b. placement -> ('a, 'b) t -> ('a, 'b) t;
      (* the value on a placement of this engine; its source stays. Placing an
         empty value allocates nothing, and raises [Invalid_argument] if the
         placement cannot hold the dtype: nx checks held values this way. *)
}

and device = { d_id : int; d_name : string; d_engine : engine }
and placement = device Grid.t
and storage = ..
and node = ..

type context = device context_of

(* A value of one element that nx holds itself: a scalar created in a device
   context, or a one-element result. It allocates nothing on the device; an
   engine passes it to a program as it passes a host value. *)
type storage += Held : ('a, 'b) Nx_dtype.t * 'a -> storage

let id_counter = Atomic.make 0
let fresh_id () = Atomic.fetch_and_add id_counter 1 + 1

(* Ids are handed out in increasing order, so a tracer can tell the traced
   tensors made before a point of its trace from those made after it. *)
let next_traced_id () = Atomic.get id_counter + 1
let host_context = Nx_backend.create_context ()

let outside_trace () =
  invalid_arg
    "a traced tensor has no bytes; it was used outside the trace that made it"

let consumed { path } =
  invalid_arg
    (Printf.sprintf
       "this value was consumed at %s in a compiled call's arguments; use the \
        value the call returned"
       path)

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

(* The elements of a placed value's view. A held value's are its one element,
   broadcast. *)
let read_elements (type a b) (r : (a, b) resident) : (a, b) Nx_buffer.t =
  Cell.with_borrow r.r_cell (fun () ->
  match r.r_cell.state with
  | Consumed k -> consumed k
  | Live (Held (dt, v)) -> (
      let buf = Nx_buffer.create r.r_dtype (View.numel r.r_view) in
      match Nx_dtype.equal_witness dt r.r_dtype with
      | Some Type.Equal ->
          Nx_buffer.fill buf v;
          buf
      | None -> assert false)
  | Live _ -> (List.hd (Grid.devices r.r_cell.placement)).d_engine.read r)

(* [global p shape] is the shape of a value whose tiles at [p] have [shape]. *)
let global p shape =
  match Grid.cuts p with
  | [] -> shape
  | cuts ->
      let shape = Array.copy shape in
      List.iter (fun (a, n) -> shape.(a) <- shape.(a) * n) cuts;
      shape

(* A split value's view is each shard's, and its shape the whole's. *)
let whole_view r =
  match Grid.cuts r.r_placement with
  | [] -> r.r_view
  | _ ->
      let v = r.r_view in
      View.create ~offset:(View.offset v) ~strides:(View.strides v)
        (global r.r_placement (View.shape v))

let read_host (type a b) (r : (a, b) resident) : (a, b) Nx_backend.t =
  Nx_backend.reshape
    (Nx_backend.from_host host_context (read_elements r))
    (View.shape (whole_view r))

(* [host_of x] is [x]'s value as a host tensor: [x] itself on the host, a copy
   of its view's elements when it is placed. *)
let host_of : type a b. (a, b) t -> (a, b) Nx_backend.t = function
  | Host t -> t
  | Placed r -> read_host r
  | Traced _ -> outside_trace ()

(* Devices *)

module Device = struct
  type t = device

  exception Out_of_memory of t * int

  let make name engine =
    { d_id = fresh_id (); d_name = name; d_engine = engine }

  let name d = d.d_name
  let engine d = d.d_engine
  let equal = ( == )
  let compare a b = Int.compare a.d_id b.d_id
  let pp ppf d = Format.pp_print_string ppf d.d_name

  (* The host holds no placed value: a value moves to it by a read. *)
  let rec host_engine =
    {
      read = (fun _ -> invalid_arg "the host holds no placed value");
      place =
        (fun p x ->
          match Grid.devices p with
          | [ d ] when d.d_engine == host_engine -> Host (host_of x)
          | _ -> invalid_arg "the host engine places values on the host only");
    }

  let host = make "CPU" host_engine

  let () =
    Printexc.register_printer (function
      | Out_of_memory (d, n) ->
          Some
            (Printf.sprintf "Nx.Device.Out_of_memory(%s, %d bytes)" d.d_name n)
      | _ -> None)
end

(* Placements *)

module Placement = struct
  type t = placement

  let host = Grid.device Device.host
  let device d = Grid.device d
  let devices = Grid.devices
  let engine p = (List.hd (devices p)).d_engine
  let is_host p = match devices p with [ d ] -> d == Device.host | _ -> false

  let check what ds =
    let fail fmt =
      Printf.ksprintf invalid_arg ("Nx.Placement.%s: " ^^ fmt) what
    in
    let rec distinct = function
      | [] -> ()
      | d :: rest ->
          if List.memq d rest then fail "%s appears twice" d.d_name;
          distinct rest
    in
    match ds with
    | [] -> fail "no device"
    | d :: rest ->
        distinct ds;
        List.iter
          (fun d' ->
            if d'.d_engine != d.d_engine then
              fail "%s and %s belong to different engines" d.d_name d'.d_name)
          rest

  let replicated ds =
    check "replicated" ds;
    Grid.v ds [ List.length ds ] []

  let sharded ~axis ds =
    if axis < 0 then
      invalid_arg (Printf.sprintf "Nx.Placement.sharded: axis %d < 0" axis);
    check "sharded" ds;
    Grid.v ds [ List.length ds ] [ (axis, [ 0 ]) ]

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
      (Grid.cuts p)

  let window p shape d =
    match List.find_index (( == ) d) (devices p) with
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
          (Grid.cuts p) (Grid.tile_index p k);
        w

  (* The placement of a value with a new leading axis, and of one without its
     leading axis, which no cut may name. *)
  let with_leading_axis p = Grid.map_axes succ p

  let without_leading_axis p =
    if List.mem_assoc 0 (Grid.cuts p) then None else Some (Grid.map_axes pred p)

  let equal = Grid.equal ( == )
  let pp ppf p = Grid.pp Device.pp ppf p
end

(* Placed constructors, for engines *)

(* A cell over [storage] of [length] elements per device of [placement], whose
   engine owns it. The engine attaches the finaliser that releases the
   storage. *)
let cell ~placement ~length storage =
  { placement; length; state = Live storage; bound = Atomic.make 0;
    lock = Mutex.create (); readers = 0; exclusive = false }

let placed placement dtype view cell =
  if Placement.is_host placement then
    invalid_arg "Nx_effect.placed: a placed value is never on the host";
  let held = Placement.devices cell.placement in
  if
    not (List.for_all (fun d -> List.memq d held) (Placement.devices placement))
  then
    invalid_arg
      (Format.asprintf "Nx_effect.placed: a value on %a views a storage on %a"
         Placement.pp placement Placement.pp cell.placement);
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
   signalling NaN. 4-bit elements are copied as values. *)
let blit_box (type a b) (src : (a, b) Nx_buffer.t) box
    (dst : (a, b) Nx_buffer.t) ~into ~at =
  let box, into, at =
    if Array.length box = 0 then ([| 1 |], [| 1 |], [| 0 |]) else (box, into, at)
  in
  let words (type c d) (word : (c, d) Nx_dtype.t) w =
    let scale a =
      let a = Array.copy a in
      let r = Array.length a - 1 in
      a.(r) <- a.(r) * w;
      a
    in
    let s = Nx_buffer.to_bigarray1 (Nx_buffer.reinterpret word src)
    and d = Nx_buffer.to_bigarray1 (Nx_buffer.reinterpret word dst) in
    let box = scale box in
    let run = box.(Array.length box - 1) in
    iter_rows box ~into:(scale into) ~at:(scale at) (fun src_off dst_off ->
        Bigarray.Array1.blit
          (Bigarray.Array1.sub s src_off run)
          (Bigarray.Array1.sub d dst_off run))
  in
  match Nx_buffer.dtype src with
  | Nx_dtype.Int4 | Nx_dtype.UInt4 ->
      let run = box.(Array.length box - 1) in
      iter_rows box ~into ~at (fun src_off dst_off ->
          for i = 0 to run - 1 do
            Nx_buffer.unsafe_set dst (dst_off + i)
              (Nx_buffer.unsafe_get src (src_off + i))
          done)
  | kind -> (
      match Nx_dtype.itemsize kind with
      | 1 -> words Nx_dtype.Int8 1
      | 2 -> words Nx_dtype.Int16 1
      | 4 -> words Nx_dtype.Int32 1
      | 8 -> words Nx_dtype.Int64 1
      | n -> words Nx_dtype.Int64 (n / 8))

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
   that holds it, and only where it meets the window. Engines read placed
   values, and gather the pieces of a move, this way. *)
let assemble (type a b) (r : (a, b) resident) window
    (read : device -> View.t -> (a, b) Nx_buffer.t) : (a, b) Nx_buffer.t =
  let p = r.r_placement in
  let shape = global p (View.shape r.r_view) in
  let pieces =
    List.fold_left
      (fun pieces d ->
        let t = Placement.window p shape d in
        if List.exists (fun (_, t', _) -> t' = t) pieces then pieces
        else
          match intersect window t with
          | Some i -> (d, t, i) :: pieces
          | None -> pieces)
      [] (Placement.devices p)
  in
  let piece (d, t, i) = read d (View.shrink r.r_view (within t i)) in
  match pieces with
  | [ ((_, _, i) as only) ] when i = window -> piece only
  | _ ->
      let into = extents window in
      let dst = Nx_buffer.create r.r_dtype (Array.fold_left ( * ) 1 into) in
      List.iter
        (fun ((_, _, i) as p) ->
          blit_box (piece p) (extents i) dst ~into
            ~at:(Array.map fst (within window i)))
        pieces;
      dst

(* A held value of shape [shape] on [p]. The engine is asked to place an empty
   value of the dtype first, which allocates nothing and raises if [p] cannot
   hold the dtype. *)
let held p dtype value shape =
  ignore
    ((Placement.engine p).place p
       (Host (Nx_backend.buffer host_context dtype [| 0 |])));
  placed p dtype (View.create shape)
    (cell ~placement:p ~length:1 (Held (dtype, value)))

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

(* [Nx_buffer.t] is not injective in its parameters either (it abbreviates a
   bigarray), so a host buffer cannot be an effect's result directly; the same
   boxing trick restores deducibility for [E_to_host]. *)
type ('a, 'b) host_buffer =
  | Host_buffer : ('a, 'b) Nx_buffer.t -> ('a, 'b) host_buffer

type packed = P : ('a, 'b) t -> packed

(* Effects *)

type _ Effect.t +=
  | E_view : ('a, 'b) t -> View.t Effect.t
  | E_buffer : {
      context : context;
      dtype : ('a, 'b) Nx_dtype.t;
      size_in_elements : int;
    }
      -> ('a, 'b) t Effect.t
  | E_const_scalar : {
      context : context;
      value : 'a;
      dtype : ('a, 'b) Nx_dtype.t;
    }
      -> ('a, 'b) t Effect.t
  | E_from_host : {
      context : context;
      array : ('a, 'b) Nx_buffer.t;
    }
      -> ('a, 'b) t Effect.t
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
  | E_psum : { t_in : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_axis_index : (int32, Nx_dtype.int32_elt) t Effect.t
  | E_cholesky : { t_in : ('a, 'b) t; upper : bool } -> ('a, 'b) t Effect.t
  | E_qr : {
      t_in : ('a, 'b) t;
      reduced : bool;
    }
      -> (('a, 'b) t * ('a, 'b) t) Effect.t
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
  | E_to_host : ('a, 'b) t -> ('a, 'b) host_buffer Effect.t

(* Lenses. The effect is performed first: a handler may present a transformed
   view (vmap shows batched tensors without their batch axis) or placement; only
   the unhandled fallback answers from the tensor. *)

let create_context () : context = Host (Nx_backend.create_context ())

(* The context of host tensors, allocated once: the frontend asks for a context
   each time it builds a constant beside an operand. *)
let host_tensor_context : context = Host host_context

let context : type a b. (a, b) t -> context = function
  | Host t ->
      let c = Nx_backend.context t in
      if c == host_context then host_tensor_context else Host c
  | Placed r -> On (Placement.devices r.r_placement)
  | Traced t -> t.t_context

let view (type a b) (x : (a, b) t) : View.t =
  try Effect.perform (E_view x)
  with Effect.Unhandled _ -> (
    match x with
    | Host t -> Nx_backend.view t
    | Placed r -> whole_view r
    | Traced t -> t.t_view)

let dtype : type a b. (a, b) t -> (a, b) Nx_dtype.t = function
  | Host t -> Nx_backend.dtype t
  | Placed r -> r.r_dtype
  | Traced t -> t.t_dtype

let placement (type a b) (x : (a, b) t) : placement =
  try Effect.perform (E_placement x)
  with Effect.Unhandled _ -> (
    match x with
    | Host _ -> Placement.host
    | Placed r -> r.r_placement
    | Traced _ -> outside_trace ())

(* The host engine's storage of a host value, and the view's elements of a
   placed one: readers take [contiguous] first, so the view is the storage. *)
let to_host (type a b) (x : (a, b) t) : (a, b) Nx_buffer.t =
  try
    let (Host_buffer buf) = Effect.perform (E_to_host x) in
    buf
  with Effect.Unhandled _ -> (
    match x with
    | Host t -> Nx_backend.to_host t
    | Placed r -> read_elements r
    | Traced _ -> outside_trace ())

(* Moving *)

let move (type a b) p (x : (a, b) t) : (a, b) t =
  let move () = match x with
  | Traced _ -> outside_trace ()
  | Placed { r_placement; _ } when Placement.equal r_placement p -> x
  | Host _ when Placement.is_host p -> x
  | Placed r when Placement.is_host p -> Host (read_host r)
  | Host _ | Placed _ ->
      Placement.check_shape "Nx.place" p (View.shape (view x));
      (Placement.engine p).place p x in
  match x with
  | Placed r -> Cell.with_borrow r.r_cell move
  | Host _ | Traced _ -> move ()

let place p x =
  try Effect.perform (E_place { placement = p; t_in = x })
  with Effect.Unhandled _ -> move p x

(* Routing

   Every fallback runs where its operands live. Operands all on the host run on
   the link-time engine. Placed operands must share their devices, and host
   operands join them: until devices compute, the operation reads the placed
   operands' windows, runs on the host engine and places its result. A result of
   one element is held by nx instead, so reading it back moves nothing. The
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

let placement_of : type a b. (a, b) t -> placement option = function
  | Host _ -> None
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

(* [combine op ps] is the placement of an elementwise operation's result over
   operands at [ps]: that of the split ones, which must be alike; copies take
   it. *)
let combine op ps =
  match List.filter (fun p -> Grid.cuts p <> []) ps with
  | [] -> List.hd ps
  | p :: rest -> (
      match List.find_opt (fun q -> not (Placement.equal p q)) rest with
      | None -> p
      | Some q ->
          invalid_arg
            (Format.asprintf
               "Nx.%s: operands at %a and %a are split differently; place them \
                alike first"
               op Placement.pp p Placement.pp q))

(* [result op rule operands] is where [op]'s result lives, over operands of
   these placements ([None] on the host) and ranks, the placed ones among them
   sharing their devices. *)
let result op rule operands =
  let ps = List.filter_map fst operands in
  let cut p a = List.mem_assoc a (Grid.cuts p) in
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

let same_devices p q =
  let dp = Placement.devices p and dq = Placement.devices q in
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
        | Placed r -> Some (r.r_cell, r.r_view, r.r_placement)
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
             && List.compare_length_with (Placement.devices p) 1 = 0)
           views ->
      Some (Placement.devices cell.placement)
  | _ -> None

(* [route op rule xs] is where [op] runs over [xs]. Placed operands on different
   device sets raise, but for [whole_shards]. *)
let route op rule xs =
  match List.filter_map (fun (P x) -> placement_of x) xs with
  | [] -> On_host
  | p :: rest -> (
      match List.find_opt (fun q -> not (same_devices p q)) rest with
      | None ->
          At
            (result op rule
               (List.map (fun (P x as o) -> (placement_of x, rank o)) xs))
      | Some q -> (
          match whole_shards xs with
          | Some ds -> At (Placement.replicated ds)
          | None ->
              invalid_arg
                (Format.asprintf
                   "Nx.%s: operands on %a and %a; place one of them" op
                   Placement.pp p Placement.pp q)))

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

let settle : type a b. route -> (a, b) Nx_backend.t -> (a, b) t =
 fun r h ->
  match r with
  | On_host -> Host h
  | At p ->
      let shape = View.shape (Nx_backend.view h) in
      if Array.fold_left ( * ) 1 shape = 1 then
        let v =
          Nx_buffer.get (Nx_backend.to_host h) (View.offset (Nx_backend.view h))
        in
        held p (Nx_backend.dtype h) v shape
      else (Placement.engine p).place p (Host h)

(* [routed e x f] runs [f] over [x], no host tensor, where the operation that
   performs [e] runs. *)
let routed e x f =
  let r = route_of e in
  settle r (f (host_of x))

let unary_op e host_op t_in =
  try Effect.perform e
  with Effect.Unhandled _ -> (
    match t_in with Host t -> Host (host_op t) | _ -> routed e t_in host_op)

let binary_op e host_op a b =
  try Effect.perform e
  with Effect.Unhandled _ -> (
    match (a, b) with
    | Host a, Host b -> Host (host_op a b)
    | _ ->
        let r = route_of e in
        settle r (host_op (host_of a) (host_of b)))

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
    (Grid.cuts p)

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
  let p =
    List.fold_left
      (fun p (axis, _, fate) ->
        match fate with Shard j -> Grid.select p ~axis j | Split _ -> p)
      p fates
  in
  Grid.map_axes
    (fun a ->
      match List.find (fun (axis, _, _) -> axis = a) fates with
      | _, _, Split a' -> a'
      | _, _, Shard _ -> a)
    p

(* [moved_placement p shape m] is the placement of a value of shape [shape] at
   [p] moved by [m]. Raises [Invalid_argument] as [split_axis] does. *)
let moved_placement p shape m = placement_after p (fates p shape m)

(* [split_view p v m] is the placement and per-shard view of a value at [p]
   whose per-shard view is [v], moved by [m]. *)
let split_view p v m =
  match Grid.cuts p with
  | [] -> (p, move_view v m)
  | _ ->
      let shape = global p (View.shape v) in
      let fates = fates p shape m in
      (placement_after p fates, move_view v (localize shape m fates))

let movement_op e host_op t_in arg =
  try Effect.perform e
  with Effect.Unhandled _ -> (
    match (t_in, movement_of e) with
    | Host t, _ -> Host (host_op t arg)
    | Placed r, Some (_, m) ->
        let r_placement, r_view = split_view r.r_placement r.r_view m in
        Placed { r with r_id = fresh_id (); r_placement; r_view }
    | Placed _, None -> invalid_arg "Nx_effect.movement_op: not a movement"
    | Traced _, _ -> outside_trace ())

(* Binary operations *)

let add a b = binary_op (E_add { a; b }) Nx_backend.add a b
let sub a b = binary_op (E_sub { a; b }) Nx_backend.sub a b
let mul a b = binary_op (E_mul { a; b }) Nx_backend.mul a b
let max a b = binary_op (E_max { a; b }) Nx_backend.max a b
let min a b = binary_op (E_min { a; b }) Nx_backend.min a b
let mod_ a b = binary_op (E_mod { a; b }) Nx_backend.mod_ a b
let pow a b = binary_op (E_pow { a; b }) Nx_backend.pow a b
let xor a b = binary_op (E_xor { a; b }) Nx_backend.xor a b
let or_ a b = binary_op (E_or { a; b }) Nx_backend.or_ a b
let and_ a b = binary_op (E_and { a; b }) Nx_backend.and_ a b
let atan2 a b = binary_op (E_atan2 { a; b }) Nx_backend.atan2 a b
let fdiv a b = binary_op (E_fdiv { a; b }) Nx_backend.fdiv a b
let idiv a b = binary_op (E_idiv { a; b }) Nx_backend.idiv a b

(* Comparison operations *)

(* [routed2 e a b f] runs [f] over [a] and [b], one no host tensor, where the
   operation that performs [e] runs. *)
let routed2 e a b f =
  let r = route_of e in
  settle r (f (host_of a) (host_of b))

let cmpeq a b =
  let e = E_cmpeq { a; b } in
  try Effect.perform e
  with Effect.Unhandled _ -> (
    match (a, b) with
    | Host a, Host b -> Host (Nx_backend.cmpeq a b)
    | _ -> routed2 e a b Nx_backend.cmpeq)

let cmpne a b =
  let e = E_cmpne { a; b } in
  try Effect.perform e
  with Effect.Unhandled _ -> (
    match (a, b) with
    | Host a, Host b -> Host (Nx_backend.cmpne a b)
    | _ -> routed2 e a b Nx_backend.cmpne)

let cmplt a b =
  let e = E_cmplt { a; b } in
  try Effect.perform e
  with Effect.Unhandled _ -> (
    match (a, b) with
    | Host a, Host b -> Host (Nx_backend.cmplt a b)
    | _ -> routed2 e a b Nx_backend.cmplt)

let cmple a b =
  let e = E_cmple { a; b } in
  try Effect.perform e
  with Effect.Unhandled _ -> (
    match (a, b) with
    | Host a, Host b -> Host (Nx_backend.cmple a b)
    | _ -> routed2 e a b Nx_backend.cmple)

(* Unary operations *)

let neg t = unary_op (E_neg { t_in = t }) Nx_backend.neg t
let sin t = unary_op (E_sin { t_in = t }) Nx_backend.sin t
let sqrt t = unary_op (E_sqrt { t_in = t }) Nx_backend.sqrt t
let recip t = unary_op (E_recip { t_in = t }) Nx_backend.recip t
let log t = unary_op (E_log { t_in = t }) Nx_backend.log t
let exp t = unary_op (E_exp { t_in = t }) Nx_backend.exp t
let cos t = unary_op (E_cos { t_in = t }) Nx_backend.cos t
let abs t = unary_op (E_abs { t_in = t }) Nx_backend.abs t
let sign t = unary_op (E_sign { t_in = t }) Nx_backend.sign t
let tan t = unary_op (E_tan { t_in = t }) Nx_backend.tan t
let asin t = unary_op (E_asin { t_in = t }) Nx_backend.asin t
let acos t = unary_op (E_acos { t_in = t }) Nx_backend.acos t
let atan t = unary_op (E_atan { t_in = t }) Nx_backend.atan t
let sinh t = unary_op (E_sinh { t_in = t }) Nx_backend.sinh t
let cosh t = unary_op (E_cosh { t_in = t }) Nx_backend.cosh t
let tanh t = unary_op (E_tanh { t_in = t }) Nx_backend.tanh t
let trunc t = unary_op (E_trunc { t_in = t }) Nx_backend.trunc t
let ceil t = unary_op (E_ceil { t_in = t }) Nx_backend.ceil t
let floor t = unary_op (E_floor { t_in = t }) Nx_backend.floor t
let round t = unary_op (E_round { t_in = t }) Nx_backend.round t
let erf t = unary_op (E_erf { t_in = t }) Nx_backend.erf t

let op_psum t_in =
  try Effect.perform (E_psum { t_in })
  with Effect.Unhandled _ -> failwith "psum must be used under vmap"

(* Reduction operations. The host case of each operation below calls its backend
   directly, allocating nothing beyond the effect and the result. *)

let reduce ~op ~axes t_in =
  let eff =
    match op with
    | `Sum -> E_reduce_sum { t_in; axes }
    | `Prod -> E_reduce_prod { t_in; axes }
    | `Max -> E_reduce_max { t_in; axes }
    | `Min -> E_reduce_min { t_in; axes }
  in
  try Effect.perform eff
  with Effect.Unhandled _ -> (
    match t_in with
    | Host t -> Host (Nx_backend.reduce ~op ~axes t)
    | _ -> routed eff t_in (Nx_backend.reduce ~op ~axes))

let argmax ~axis ~keepdims t_in =
  let e = E_argmax { t_in; axis; keepdims } in
  try Effect.perform e
  with Effect.Unhandled _ -> (
    match t_in with
    | Host t -> Host (Nx_backend.argmax ~axis ~keepdims t)
    | _ -> routed e t_in (Nx_backend.argmax ~axis ~keepdims))

let argmin ~axis ~keepdims t_in =
  let e = E_argmin { t_in; axis; keepdims } in
  try Effect.perform e
  with Effect.Unhandled _ -> (
    match t_in with
    | Host t -> Host (Nx_backend.argmin ~axis ~keepdims t)
    | _ -> routed e t_in (Nx_backend.argmin ~axis ~keepdims))

let associative_scan ~axis ~op t_in =
  let e = E_associative_scan { t_in; axis; op } in
  try Effect.perform e
  with Effect.Unhandled _ -> (
    match t_in with
    | Host t -> Host (Nx_backend.associative_scan ~axis ~op t)
    | _ -> routed e t_in (Nx_backend.associative_scan ~axis ~op))

let sort ~axis ~descending t_in =
  let e = E_sort { t_in; axis; descending } in
  try Effect.perform e
  with Effect.Unhandled _ -> (
    match t_in with
    | Host t -> Host (Nx_backend.sort ~axis ~descending t)
    | _ -> routed e t_in (Nx_backend.sort ~axis ~descending))

let argsort ~axis ~descending t_in =
  let e = E_argsort { t_in; axis; descending } in
  try Effect.perform e
  with Effect.Unhandled _ -> (
    match t_in with
    | Host t -> Host (Nx_backend.argsort ~axis ~descending t)
    | _ -> routed e t_in (Nx_backend.argsort ~axis ~descending))

(* Movement operations *)

let reshape t_in new_shape =
  movement_op (E_reshape { t_in; new_shape }) Nx_backend.reshape t_in new_shape

let expand t_in new_target_shape =
  movement_op
    (E_expand { t_in; new_target_shape })
    Nx_backend.expand t_in new_target_shape

let permute t_in axes =
  movement_op (E_permute { t_in; axes }) Nx_backend.permute t_in axes

let shrink t_in limits =
  movement_op (E_shrink { t_in; limits }) Nx_backend.shrink t_in limits

let flip t_in dims_to_flip =
  movement_op (E_flip { t_in; dims_to_flip }) Nx_backend.flip t_in dims_to_flip

let sliding_window t_in ~axis ~window ~step =
  movement_op
    (E_sliding_window { t_in; axis; window; step })
    (fun t () -> Nx_backend.sliding_window t ~axis ~window ~step)
    t_in ()

let pad t_in padding_config fill_value =
  let e = E_pad { t_in; padding_config; fill_value } in
  try Effect.perform e
  with Effect.Unhandled _ -> (
    match t_in with
    | Host t -> Host (Nx_backend.pad t padding_config fill_value)
    | _ -> routed e t_in (fun t -> Nx_backend.pad t padding_config fill_value))

(* Creation operations. A value created in the context of devices lives there, a
   full copy on each, and a scalar there is held by nx and allocates nothing. A
   filled value of more than one element has storage of its own, so that its
   view covers its storage and a compiled call can consume it. *)

let at_devices ds = At (Placement.replicated ds)

let buffer (ctx : context) dtype shape_arr =
  let size_in_elements = Array.fold_left ( * ) 1 shape_arr in
  let flat =
    try Effect.perform (E_buffer { context = ctx; dtype; size_in_elements })
    with Effect.Unhandled _ -> (
      match ctx with
      | Host c -> Host (Nx_backend.buffer c dtype shape_arr)
      | On ds ->
          settle (at_devices ds)
            (Nx_backend.buffer host_context dtype shape_arr))
  in
  reshape flat shape_arr

let const_scalar (ctx : context) value dtype =
  try Effect.perform (E_const_scalar { context = ctx; value; dtype })
  with Effect.Unhandled _ -> (
    match ctx with
    | Host c -> Host (Nx_backend.full c dtype [||] value)
    | On ds -> held (Placement.replicated ds) dtype value [||])

let broadcast scalar shape_arr =
  if Array.length shape_arr = 0 then scalar
  else expand (reshape scalar (Array.map (fun _ -> 1) shape_arr)) shape_arr

let full (ctx : context) dtype shape_arr value =
  (* Under an effect handler (jit tracing) a filled tensor is a broadcast scalar
     constant: no bytes are materialized. Until devices compute (RFC 0005 stage
     3), a fill in a device's context runs on the host and is placed, which
     holds a host copy of the value until the next collection; stage 3 fills on
     the device. *)
  match Effect.perform (E_const_scalar { context = ctx; value; dtype }) with
  | scalar -> broadcast scalar shape_arr
  | exception Effect.Unhandled _ -> (
      match ctx with
      | Host c -> Host (Nx_backend.full c dtype shape_arr value)
      | On ds ->
          settle (at_devices ds)
            (Nx_backend.full host_context dtype shape_arr value))

(* [full_at p dtype shape value] is [full] at the placement [p]: a split one
   gives each device its window. *)
let full_at p dtype shape_arr value =
  let context = On (Placement.devices p) in
  match Effect.perform (E_const_scalar { context; value; dtype }) with
  | scalar -> broadcast scalar shape_arr
  | exception Effect.Unhandled _ ->
      settle (At p) (Nx_backend.full host_context dtype shape_arr value)

let from_host (ctx : context) array =
  try Effect.perform (E_from_host { context = ctx; array })
  with Effect.Unhandled _ -> (
    match ctx with
    | Host c -> Host (Nx_backend.from_host c array)
    | On ds -> settle (at_devices ds) (Nx_backend.from_host host_context array))

(* Copy operations. A placed value whose view covers its storage is already
   contiguous. *)

let contiguous t_in =
  let e = E_contiguous { t_in } in
  try Effect.perform e
  with Effect.Unhandled _ -> (
    match t_in with
    | Host t -> Host (Nx_backend.contiguous t)
    | Placed r when covers r -> t_in
    | _ -> routed e t_in Fun.id)

let copy t_in =
  let e = E_copy { t_in } in
  try Effect.perform e
  with Effect.Unhandled _ -> (
    match t_in with
    | Host t -> Host (Nx_backend.copy t)
    | _ -> routed e t_in Nx_backend.copy)

(* Ternary operations *)

let where condition if_true if_false =
  let e = E_where { condition; if_true; if_false } in
  try Effect.perform e
  with Effect.Unhandled _ -> (
    match (condition, if_true, if_false) with
    | Host c, Host a, Host b -> Host (Nx_backend.where c a b)
    | _ ->
        let r = route_of e in
        settle r
          (Nx_backend.where (host_of condition) (host_of if_true)
             (host_of if_false)))

(* Cat *)

let cat t_list ~axis =
  let e = E_cat { t_list; axis } in
  try Effect.perform e
  with Effect.Unhandled _ ->
    if List.for_all (function Host _ -> true | _ -> false) t_list then
      Host (Nx_backend.cat (List.map host_of t_list) ~axis)
    else
      let r = route_of e in
      settle r (Nx_backend.cat (List.map host_of t_list) ~axis)

(* Cast *)

let cast (type a b c d) ~(dtype : (c, d) Nx_dtype.t) (t_in : (a, b) t) :
    (c, d) t =
  let e = E_cast { t_in; target_dtype = dtype } in
  try Effect.perform e
  with Effect.Unhandled _ -> (
    match t_in with
    | Host t -> Host (Nx_backend.cast ~dtype t)
    | _ -> routed e t_in (Nx_backend.cast ~dtype))

let bitcast (type a b c d) ~(dtype : (c, d) Nx_dtype.t) (t_in : (a, b) t) :
    (c, d) t =
  let e = E_bitcast { t_in; target_dtype = dtype } in
  try Effect.perform e
  with Effect.Unhandled _ -> (
    match t_in with
    | Host t -> Host (Nx_backend.bitcast ~dtype t)
    | _ -> routed e t_in (Nx_backend.bitcast ~dtype))

(* Indexed access *)

let gather data indices ~axis =
  let e = E_gather { data; indices; axis } in
  try Effect.perform e
  with Effect.Unhandled _ -> (
    match (data, indices) with
    | Host d, Host i -> Host (Nx_backend.gather d i ~axis)
    | _ -> routed2 e data indices (fun d i -> Nx_backend.gather d i ~axis))

let update t_in ~starts v =
  let e = E_update { t_in; starts; v } in
  try Effect.perform e
  with Effect.Unhandled _ -> (
    match (t_in, starts, v) with
    | Host t, Host s, Host v -> Host (Nx_backend.update t ~starts:s v)
    | _ ->
        let r = route_of e in
        settle r
          (Nx_backend.update (host_of t_in) ~starts:(host_of starts) (host_of v)))

let scatter ~mode ~unique_indices data_template ~indices ~updates ~axis =
  let e =
    E_scatter { data_template; indices; updates; axis; mode; unique_indices }
  in
  try Effect.perform e
  with Effect.Unhandled _ -> (
    match (data_template, indices, updates) with
    | Host d, Host i, Host u ->
        Host
          (Nx_backend.scatter ~mode ~unique_indices d ~indices:i ~updates:u
             ~axis)
    | _ ->
        let r = route_of e in
        settle r
          (Nx_backend.scatter ~mode ~unique_indices (host_of data_template)
             ~indices:(host_of indices) ~updates:(host_of updates) ~axis))

(* Random *)

let threefry key ctr =
  let e = E_threefry { key; ctr } in
  try Effect.perform e
  with Effect.Unhandled _ -> (
    match (key, ctr) with
    | Host k, Host c -> Host (Nx_backend.threefry k c)
    | _ -> routed2 e key ctr Nx_backend.threefry)

(* The index of the current lane along the innermost mapped axis. The vmap
   handler answers with a per-lane (batched) index; with no handler there is a
   single lane, index 0. [Nx.Rng.fold_in_axis] folds it into a key to
   decorrelate lanes. *)
let axis_index ctx =
  try Effect.perform E_axis_index
  with Effect.Unhandled _ -> const_scalar ctx 0l Nx_dtype.int32

(* Window operations *)

let unfold t_in ~kernel_size ~stride ~dilation ~padding =
  let e = E_unfold { t_in; kernel_size; stride; dilation; padding } in
  try Effect.perform e
  with Effect.Unhandled _ -> (
    match t_in with
    | Host t ->
        Host (Nx_backend.unfold t ~kernel_size ~stride ~dilation ~padding)
    | _ ->
        routed e t_in (fun t ->
            Nx_backend.unfold t ~kernel_size ~stride ~dilation ~padding))

let fold t_in ~output_size ~kernel_size ~stride ~dilation ~padding =
  let e =
    E_fold { t_in; output_size; kernel_size; stride; dilation; padding }
  in
  try Effect.perform e
  with Effect.Unhandled _ -> (
    match t_in with
    | Host t ->
        Host
          (Nx_backend.fold t ~output_size ~kernel_size ~stride ~dilation
             ~padding)
    | _ ->
        routed e t_in (fun t ->
            Nx_backend.fold t ~output_size ~kernel_size ~stride ~dilation
              ~padding))

(* Matrix operations *)

let matmul a b =
  let e = E_matmul { a; b } in
  try Effect.perform e
  with Effect.Unhandled _ -> (
    match (a, b) with
    | Host a, Host b -> Host (Nx_backend.matmul a b)
    | _ -> routed2 e a b Nx_backend.matmul)

(* FFT operations *)

let fft t ~axes =
  let e = E_fft { t; axes } in
  try Effect.perform e
  with Effect.Unhandled _ -> (
    match t with
    | Host h -> Host (Nx_backend.fft h ~axes)
    | _ -> routed e t (Nx_backend.fft ~axes))

let ifft t ~axes =
  let e = E_ifft { t; axes } in
  try Effect.perform e
  with Effect.Unhandled _ -> (
    match t with
    | Host h -> Host (Nx_backend.ifft h ~axes)
    | _ -> routed e t (Nx_backend.ifft ~axes))

let rfft (type a c) (t : (float, a) t) ~(dtype : (Complex.t, c) Nx_dtype.t)
    ~axes : (Complex.t, c) t =
  let e = E_rfft { t; dtype; axes } in
  try Effect.perform e
  with Effect.Unhandled _ -> (
    match t with
    | Host h -> Host (Nx_backend.rfft h ~dtype ~axes)
    | _ -> routed e t (Nx_backend.rfft ~dtype ~axes))

let irfft (type a c) ?s (t : (Complex.t, a) t) ~(dtype : (float, c) Nx_dtype.t)
    ~axes : (float, c) t =
  let e = E_irfft { t; dtype; axes; s } in
  try Effect.perform e
  with Effect.Unhandled _ -> (
    match t with
    | Host h -> Host (Nx_backend.irfft ?s h ~dtype ~axes)
    | _ -> routed e t (Nx_backend.irfft ?s ~dtype ~axes))

(* Linear algebra *)

let cholesky ~upper t_in =
  let e = E_cholesky { t_in; upper } in
  try Effect.perform e
  with Effect.Unhandled _ -> (
    match t_in with
    | Host t -> Host (Nx_backend.cholesky ~upper t)
    | _ -> routed e t_in (Nx_backend.cholesky ~upper))

let qr ~reduced t_in =
  let e = E_qr { t_in; reduced } in
  try Effect.perform e
  with Effect.Unhandled _ ->
    let r = route_of e in
    let q, rr = Nx_backend.qr ~reduced (host_of t_in) in
    (settle r q, settle r rr)

let svd ~full_matrices t_in =
  let e = E_svd { t_in; full_matrices } in
  try Effect.perform e
  with Effect.Unhandled _ ->
    let r = route_of e in
    let u, s, vt = Nx_backend.svd ~full_matrices (host_of t_in) in
    (settle r u, settle r s, settle r vt)

let eigvals t_in =
  let e = E_eigvals { t_in } in
  try Effect.perform e
  with Effect.Unhandled _ -> (
    match t_in with
    | Host t -> Host (Nx_backend.eigvals t)
    | _ -> routed e t_in Nx_backend.eigvals)

let eig t_in =
  let e = E_eig { t_in } in
  try Effect.perform e
  with Effect.Unhandled _ ->
    let r = route_of e in
    let vals, vecs = Nx_backend.eig (host_of t_in) in
    (settle r vals, settle r vecs)

let eigvalsh t_in =
  let e = E_eigvalsh { t_in } in
  try Effect.perform e
  with Effect.Unhandled _ -> (
    match t_in with
    | Host t -> Host (Nx_backend.eigvalsh t)
    | _ -> routed e t_in Nx_backend.eigvalsh)

let eigh t_in =
  let e = E_eigh { t_in } in
  try Effect.perform e
  with Effect.Unhandled _ ->
    let r = route_of e in
    let vals, vecs = Nx_backend.eigh (host_of t_in) in
    (settle r vals, settle r vecs)

let solve_triangular ~upper ~transpose ~unit_diag a b =
  let e = E_solve_triangular { a; b; upper; transpose; unit_diag } in
  try Effect.perform e
  with Effect.Unhandled _ -> (
    match (a, b) with
    | Host a, Host b ->
        Host (Nx_backend.solve_triangular ~upper ~transpose ~unit_diag a b)
    | _ ->
        routed2 e a b (Nx_backend.solve_triangular ~upper ~transpose ~unit_diag))
