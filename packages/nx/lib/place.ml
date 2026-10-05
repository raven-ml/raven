(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Nx_array
open Value

(* Placing and reading

   nx answers placing and reading itself. Devices over one memory share its
   storage, so placing between them is a view; between memories it is a copy,
   each device reading only its window, and reading a placed value copies its
   elements to the host. *)

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
   signalling NaN. Elements narrower than a byte are packed: a row that starts
   and ends on bytes of both buffers is copied as its bytes, any other one
   element at a time. *)
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
  let packed (type c d) (dt : (c, d) Nx_dtype.t) bits =
    let as_dt b =
      Nx_device.Buffer.view b ~offset:0
        (Nx_dtype.Scalar.of_dtype dt)
        (Nx_device.Buffer.length b)
    in
    let get = Elements.get dt (as_dt src)
    and set = Elements.set dt (as_dt dst) in
    let s = Nx_device.Buffer.bigarray Bigarray.int8_unsigned src
    and d = Nx_device.Buffer.bigarray Bigarray.int8_unsigned dst in
    let run = box.(Array.length box - 1) in
    let on_bytes i = i * bits mod 8 = 0 in
    iter_rows box ~into ~at (fun src_off dst_off ->
        if on_bytes src_off && on_bytes dst_off && on_bytes run then
          Bigarray.Array1.blit
            (Bigarray.Array1.sub s (src_off * bits / 8) (run * bits / 8))
            (Bigarray.Array1.sub d (dst_off * bits / 8) (run * bits / 8))
        else
          for i = 0 to run - 1 do
            set (dst_off + i) (get (src_off + i))
          done)
  in
  match Nx_dtype.Scalar.bitsize (Nx_device.Buffer.dtype src) with
  | 1 -> packed Nx_dtype.bit 1
  | 4 -> packed Nx_dtype.uint4 4
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
    (read : Device.t -> View.t -> Nx_device.Buffer.t) : Nx_device.Buffer.t =
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
      let dst = Elements.create r.r_dtype (Array.fold_left ( * ) 1 into) in
      List.iter
        (fun ((_, _, i) as p) ->
          blit_box (piece p) (extents i) dst ~into
            ~at:(Array.map fst (within window i)))
        pieces;
      dst

(* Runtime buffers *)

module Cpu = (val Nx_backend.kernels Nx_cpu.backend)

(* The elements of view [v] of [b], of [dtype]. Storage of elements narrower
   than a byte is read whole: its elements may not start on a byte. A strided
   view is gathered by nx.cpu. *)
let read_view dtype b v =
  let s = Nx_device.Buffer.dtype b in
  let n = View.numel v in
  if n = 0 then Nx_device.Buffer.create Nx_device.host s 0
  else
    let lo, hi =
      if Nx_dtype.Scalar.bitsize s < 8 then (0, Nx_device.Buffer.length b)
      else View.extent v
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
      Cpu.contiguous { dtype; view; buffer = span } ~dst;
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
   every device shares the host's memory, every window has an element and starts
   on a byte, the windows are one view of their storages, and every device
   borrows them. *)
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
    List.for_all (fun d -> Nx_device.shares_host_memory (Device.memory d)) ds
    && List.for_all Option.is_some spans
  then
    let spans = List.map Option.get spans in
    let ((offset, strides, shape) as view) = fst (List.hd spans) in
    if List.for_all (fun (v', _) -> v' = view) spans then
      let borrows =
        List.map2
          (fun d (_, run) -> Nx_device.Buffer.borrow (Device.memory d) run)
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
          let shape = global r.r_placement (View.shape r.r_view) in
          assemble r
            (Array.map (fun n -> (0, n)) shape)
            (fun d v -> read_view r.r_dtype (buffer_on r.r_cell bufs d) v))

(* [file_run r] is the file bytes that hold [r]'s storage, when they lie on the
   disk at a byte aligned to an element, so that the host can read them in
   place. *)
let file_run (type a b) (r : (a, b) resident) =
  match Cell.state r.r_cell with
  | Live [ b ] when Nx_device.equal (Nx_device.Buffer.device b) Nx_device.disk
    ->
      let size =
        Int.max 1 (Nx_dtype.Scalar.bitsize (Nx_device.Buffer.dtype b) / 8)
      in
      if Nativeint.to_int (Nx_device.Buffer.address b) mod size = 0 then Some b
      else None
  | _ -> None

(* A placed value is read once and checked: a buffer of another device or format
   would otherwise reach nx.cpu's kernels. *)
let read_copy (type a b) (r : (a, b) resident) : (a, b) Nx_array.t =
  let buffer = read_elements r in
  check_host "Nx.read" r.r_dtype buffer;
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

(* A value on the disk is placed on devices that share the host's memory by
   borrowing its file's pages, and keeps its view ([file_windows]). Otherwise a
   window that is a contiguous run of a value's one runtime buffer is copied
   from it, device to device: a value on the disk is read into the device. Other
   windows are copied from the value read to the host. *)
let place_at : type a b. Placement.t -> (a, b) t -> (a, b) t =
 fun p x ->
  if Placement.on_disk p then
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
  let ds = Placement.devices p in
  let windows = List.map (fun d -> Placement.window p shape d) ds in
  let local = extents (List.hd windows) in
  let n = Array.fold_left ( * ) 1 local in
  let borrowed =
    match x with
    | Placed r -> Option.bind (file_run r) (file_windows r.r_view ds windows)
    | _ -> None
  in
  match borrowed with
  | Some (view, bufs) ->
      placed "Nx.place" p dt view
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
            let b = Nx_device.Buffer.create (Device.memory d) s n in
            if n > 0 then Nx_device.Buffer.copy ~src:(piece w) ~dst:b;
            b)
          ds windows
      in
      placed "Nx.place" p dt (View.create local)
        (cell ~placement:p ~length:n bufs)

(* Placing over one memory

   Devices over one memory share its storage, so placing a value between them is
   a view: of a host value from the device over the host's memory, of a placed
   value from devices over the memories that hold the same windows, and of a
   value whole on each of its memories from devices that each hold a window of
   it, when every window is one view of the storage, as for a broadcast
   constant. A placed value seen from the host's own device is a host value over
   its buffer. *)

let same_view v v' =
  View.offset v = View.offset v'
  && View.strides v = View.strides v'
  && View.shape v = View.shape v'

let view_at (type a b) p (x : (a, b) t) : (a, b) t option =
  match x with
  | Host t -> (
      match Placement.devices p with
      | [ d ] when Nx_device.equal (Device.memory d) Nx_device.host ->
          let length = Nx_device.Buffer.length t.buffer in
          Some
            (placed "Nx.place" p t.dtype t.view
               (cell ~placement:p ~length [ t.buffer ]))
      | _ -> None)
  | Placed r when Placement.on_disk r.r_placement -> None
  | Placed r ->
      let held = List.map Device.memory (Placement.devices r.r_placement) in
      let views () =
        let shape = View.shape r.r_view in
        List.map
          (fun d -> View.shrink r.r_view (Placement.window p shape d))
          (Placement.devices p)
      in
      if Placement.same_memories p r.r_placement then
        Some (Placed { r with r_id = fresh_id (); r_placement = p })
      else if
        Placement.cuts r.r_placement = []
        && List.for_all
             (fun d -> List.memq (Device.memory d) held)
             (Placement.devices p)
      then
        match views () with
        | v :: rest when List.for_all (same_view v) rest ->
            Some
              (Placed { r with r_id = fresh_id (); r_placement = p; r_view = v })
        | _ -> None
      else None
  | Traced _ -> outside_trace ()

(* [r]'s value from the host's device, as a host value over its buffer, when it
   lies whole in the host's memory. *)
let host_view (type a b) (r : (a, b) resident) : (a, b) Nx_array.t option =
  match (Placement.devices r.r_placement, Cell.state r.r_cell) with
  | [ d ], Live [ buffer ] when Nx_device.equal (Device.memory d) Nx_device.host
    ->
      Some { dtype = r.r_dtype; view = r.r_view; buffer }
  | _ -> None

(* [x] at [p]: [x] itself when it is there, and placed there anew otherwise. *)
let move_to (type a b) p (x : (a, b) t) : (a, b) t =
  let move () =
    match x with
    | Traced _ -> outside_trace ()
    | Placed r when Placement.is_host p -> (
        match host_view r with Some a -> Host a | None -> Host (read_host r))
    | Placed r when Placement.equal r.r_placement p -> x
    | Host _ | Placed _ -> (
        Placement.check_shape "Nx.place" p (View.shape (view x));
        match view_at p x with Some y -> y | None -> place_at p x)
  in
  match x with
  | Placed r -> Cell.with_borrow r.r_cell move
  | Host _ | Traced _ -> move ()
