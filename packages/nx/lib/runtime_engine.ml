(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Nx_core
module B = Nx_device.Buffer

(* One buffer per device of the cell's placement, in its order. *)
type Nx_effect.storage += Runtime of B.t list

(* Nx_buffer conversion

   Cells and engines still exchange elements as Nx_buffer; these two functions
   are where they meet device buffers. *)

let bigarray_of_bytes raw =
  let ba =
    Bigarray.Array1.create Bigarray.int8_unsigned Bigarray.c_layout
      (Bytes.length raw)
  in
  Bytes.iteri (fun i c -> ba.{i} <- Char.code c) raw;
  ba

let bytes_of (type a b) (nb : (a, b) Nx_buffer.t) =
  B.of_bigarray
    (Nx_buffer.to_bigarray1 (Nx_buffer.reinterpret Nx_dtype.uint8 nb))

(* Int4 elements have no byte view: they go through their packed bytes. *)
let copy_in (type a b) (nb : (a, b) Nx_buffer.t) dst =
  match Nx_buffer.dtype nb with
  | Nx_dtype.Int4 | Nx_dtype.UInt4 ->
      let raw = Bytes.create (B.nbytes dst) in
      Nx_buffer.blit_to_bytes nb raw;
      B.copy ~src:(B.of_bigarray (bigarray_of_bytes raw)) ~dst
  | _ -> B.copy ~src:(bytes_of nb) ~dst

let copy_out (type a b) src (nb : (a, b) Nx_buffer.t) =
  match Nx_buffer.dtype nb with
  | Nx_dtype.Int4 | Nx_dtype.UInt4 ->
      let ba =
        Bigarray.Array1.create Bigarray.int8_unsigned Bigarray.c_layout
          (B.nbytes src)
      in
      B.copy ~src ~dst:(B.of_bigarray ba);
      let raw = Bytes.init (B.nbytes src) (fun i -> Char.chr ba.{i}) in
      Nx_buffer.blit_from_bytes raw nb
  | _ -> B.copy ~src ~dst:(bytes_of nb)

(* Devices *)

let lock = Mutex.create ()
let opened : (Nx_device.t * Nx_effect.device) list ref = ref []

let runtime_of d =
  Mutex.protect lock (fun () ->
      match List.find_opt (fun (_, d') -> d' == d) !opened with
      | Some (rd, _) -> rd
      | None ->
          invalid_arg
            ("Nx: " ^ Nx_effect.Device.name d ^ " is not a runtime device"))

let create d s n =
  try B.create (runtime_of d) s n
  with Nx_device.Out_of_memory (_, bytes) ->
    raise (Nx_effect.Device.Out_of_memory (d, bytes))

(* Reading *)

(* The lowest and one past the highest storage element [v] reaches. *)
let extent v =
  let lo = ref (View.offset v) and hi = ref (View.offset v) in
  Array.iteri
    (fun a n ->
      let s = (View.strides v).(a) * (n - 1) in
      if s < 0 then lo := !lo + s else hi := !hi + s)
    (View.shape v);
  (!lo, !hi + 1)

(* [v]'s elements in C order, from [src], the storage elements from [base] on.
   Elements are copied as integer words of their width, [k] words to an element:
   a float read as an OCaml float would quiet a signalling NaN. 4-bit elements
   are copied as values. *)
let gather (type a b) (src : (a, b) Nx_buffer.t) ~base v =
  let dst = Nx_buffer.create (Nx_buffer.dtype src) (View.numel v) in
  let copy (type c d) (src : (c, d) Nx_buffer.t) (dst : (c, d) Nx_buffer.t) k =
    let shape = Array.append (View.shape v) [| k |] in
    let strides = Array.append (Array.map (( * ) k) (View.strides v)) [| 1 |] in
    let rank = Array.length shape in
    let idx = Array.make rank 0 in
    for w = 0 to Nx_buffer.length dst - 1 do
      let off = ref ((View.offset v - base) * k) in
      for a = 0 to rank - 1 do
        off := !off + (idx.(a) * strides.(a))
      done;
      Nx_buffer.unsafe_set dst w (Nx_buffer.unsafe_get src !off);
      let a = ref (rank - 1) in
      while !a >= 0 do
        idx.(!a) <- idx.(!a) + 1;
        if idx.(!a) < shape.(!a) then a := -1
        else begin
          idx.(!a) <- 0;
          decr a
        end
      done
    done
  in
  let words (type c d) (word : (c, d) Nx_dtype.t) k =
    copy (Nx_buffer.reinterpret word src) (Nx_buffer.reinterpret word dst) k
  in
  (match Nx_buffer.dtype src with
  | Nx_dtype.Int4 | Nx_dtype.UInt4 -> copy src dst 1
  | dt -> (
      match Nx_dtype.itemsize dt with
      | 1 -> words Nx_dtype.Int8 1
      | 2 -> words Nx_dtype.Int16 1
      | 4 -> words Nx_dtype.Int32 1
      | n -> words Nx_dtype.Int64 (n / 8)));
  dst

(* The elements of view [v] of [b], whose elements are of [dt]. Int4 storage is
   read whole: its elements may not start on a byte. *)
let read_view (type a b) (dt : (a, b) Nx_dtype.t) b v =
  let n = View.numel v in
  if n = 0 then Nx_buffer.create dt 0
  else
    let lo, hi =
      match dt with
      | Nx_dtype.Int4 | Nx_dtype.UInt4 -> (0, B.length b)
      | _ -> extent v
    in
    let span = Nx_buffer.create dt (hi - lo) in
    copy_out
      (B.view b ~offset:(lo * Nx_dtype.itemsize dt) (B.dtype b) (hi - lo))
      span;
    if View.is_c_contiguous v && hi - lo = n && View.offset v = lo then span
    else gather span ~base:lo v

let read : type a b. (a, b) Nx_effect.resident -> (a, b) Nx_buffer.t =
 fun r ->
  match Nx_effect.Cell.state r.r_cell with
  | Live (Runtime bufs) ->
      let holders = Nx_effect.Placement.devices r.r_cell.placement in
      let buffer_on d =
        List.nth bufs (Option.get (List.find_index (( == ) d) holders))
      in
      let shape = Nx_effect.global r.r_placement (View.shape r.r_view) in
      Nx_effect.assemble r
        (Array.map (fun n -> (0, n)) shape)
        (fun d v -> read_view r.r_dtype (buffer_on d) v)
  | _ -> assert false (* nx reads held and consumed values itself *)

(* Placing *)

(* The elements of [t] in C order, as a buffer of exactly them. *)
let elements t =
  let buf = Nx_backend.to_host t and v = Nx_backend.view t in
  if
    View.is_c_contiguous v
    && View.offset v = 0
    && Nx_buffer.length buf = View.numel v
  then buf
  else Nx_backend.to_host (Nx_backend.copy t)

let place : type a b.
    Nx_effect.placement -> (a, b) Nx_effect.t -> (a, b) Nx_effect.t =
 fun p x ->
  let h = Nx_effect.host_of x in
  let dt = Nx_backend.dtype h and shape = View.shape (Nx_backend.view h) in
  let s = Nx_dtype.Scalar.of_dtype dt in
  let ds = Nx_effect.Placement.devices p in
  let windows = List.map (fun d -> Nx_effect.Placement.window p shape d) ds in
  let local = Nx_effect.extents (List.hd windows) in
  let n = Array.fold_left ( * ) 1 local in
  let bufs =
    List.map2
      (fun d w ->
        let b = create d s n in
        if n > 0 then copy_in (elements (Nx_backend.shrink h w)) b;
        b)
      ds windows
  in
  Nx_effect.placed p dt (View.create local)
    (Nx_effect.cell ~placement:p ~length:n (Runtime bufs))

let engine = { Nx_effect.read; place }

let device rd =
  if Nx_device.equal rd Nx_device.host then Nx_effect.Device.host
  else
    Mutex.protect lock (fun () ->
        match
          List.find_opt (fun (rd', _) -> Nx_device.equal rd rd') !opened
        with
        | Some (_, d) -> d
        | None ->
            let d = Nx_effect.Device.make (Nx_device.name rd) engine in
            opened := (rd, d) :: !opened;
            d)
