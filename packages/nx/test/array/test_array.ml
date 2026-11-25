(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Nx_array_gen
module A = Nx_array
module D = Nx_array.Dtype
module B = Rig.Buffer
module S = Nx_array_support

let strf = Printf.sprintf
let f32 = D.Float32

(* Every index's element by [get], in C order. *)
let gets a =
  let idxs = indices (Iarray.to_array (L.shape (A.layout a))) in
  Array.of_list (List.map (fun i -> A.get a (Iarray.of_array i)) idxs)

(* [b] is [a]'s own buffer. *)
let own_buffer a b =
  satisfies ~claim:"the source's own buffer" pass (fun b -> b == A.buffer a) b

(* [c]'s buffer shares no byte with [a]'s. *)
let fresh a c =
  satisfies ~claim:"shares no byte with the source" pass
    (fun c -> not (B.overlaps (A.buffer c) (A.buffer a)))
    c

(* Arrays of every dtype *)

type case = Case : ('v, 's) A.t * M.t option -> case

let pp_case ppf (Case (a, m)) =
  Format.fprintf ppf "%a %a%a" D.pp (A.dtype a) L.pp (A.layout a)
    (fun ppf -> function
      | None -> () | Some m -> Format.fprintf ppf ", %a" pp_move m)
    m

(* An array of any dtype and shape, and a movement of it. *)
let case =
  let open Gen in
  let* (D.Any dt) = of_list ~pp:(fun ppf (D.Any dt) -> D.pp ppf dt) D.all in
  let* s = shape in
  let* a = filled dt s in
  let+ m = option (movement ~apart:false s) in
  Case (a, m)

let case = Gen.with_pp pp_case case

(* An array viewed by up to one movement, and a movement of the view: a reshape
   of a strided view is often one no view expresses. *)
let viewed =
  let open Gen in
  let* (D.Any dt) = of_list ~pp:(fun ppf (D.Any dt) -> D.pp ppf dt) D.all in
  let* s = shape in
  let* a = filled dt s in
  let* first = option (movement ~apart:false s) in
  let a = Option.value ~default:a (Option.bind first (fun m -> A.move m a)) in
  let s = Iarray.to_array (L.shape (A.layout a)) in
  let+ m = option (movement ~apart:false s) in
  Case (a, m)

let viewed = Gen.with_pp pp_case viewed

(* Construction *)

(* Layouts over a host buffer of [bytes] bytes, and whether they lie within
   them: sub-byte elements count bits, a negative stride its reach below the
   offset. *)
let within =
  [
    ("float32 [4] in 16 bytes", D.Any f32, L.contiguous [| 4 |], 16, true);
    ( "float32 [4] reversed from 3",
      D.Any f32,
      L.v ~offset:3 ~strides:[| -1 |] [| 4 |],
      16,
      true );
    ("float32 [5] in 16 bytes", D.Any f32, L.contiguous [| 5 |], 16, false);
    ( "float32 [4] from 1 in 16 bytes",
      D.Any f32,
      L.v ~offset:1 ~strides:[| 1 |] [| 4 |],
      16,
      false );
    ("int4 [32] in 16 bytes", D.Any D.Int4, L.contiguous [| 32 |], 16, true);
    ("bit [128] in 16 bytes", D.Any D.Bit, L.contiguous [| 128 |], 16, true);
    ("int4 [33] in 16 bytes", D.Any D.Int4, L.contiguous [| 33 |], 16, false);
    ("bit [129] in 16 bytes", D.Any D.Bit, L.contiguous [| 129 |], 16, false);
    ( "complex128 [2] in 16 bytes",
      D.Any D.Complex128,
      L.contiguous [| 2 |],
      16,
      false );
    ("float32 [0] in no byte", D.Any f32, L.contiguous [| 0 |], 0, true);
  ]

let test_within (_, D.Any dt, l, bytes, taken) =
  let b = B.create Rig.host bytes in
  if taken then equal layout l (A.layout (A.v dt l b))
  else raises_match (Exn.invalid_arg ~sub:"reaches past") (fun () -> A.v dt l b)

(* Layouts over 32 bytes from byte [first] of a host buffer, and whether their
   first element is aligned: a dtype's width, a complex one's component's, a
   byte below. *)
let alignments =
  [
    ("float32 at byte 1", D.Any f32, L.contiguous [| 1 |], 1, false);
    ("float32 at byte 2", D.Any f32, L.contiguous [| 1 |], 2, false);
    ("float64 at byte 4", D.Any D.Float64, L.contiguous [| 1 |], 4, false);
    ("int16 at byte 1", D.Any D.Int16, L.contiguous [| 1 |], 1, false);
    ("complex64 at byte 4", D.Any D.Complex64, L.contiguous [| 1 |], 4, true);
    ("uint8 at byte 1", D.Any D.Uint8, L.contiguous [| 1 |], 1, true);
    ("int4 at byte 1", D.Any D.Int4, L.contiguous [| 1 |], 1, true);
    ( "float32 from element 1 at byte 4",
      D.Any f32,
      L.v ~offset:1 ~strides:[| 1 |] [| 1 |],
      4,
      true );
    ( "float32 from element 1 at byte 2",
      D.Any f32,
      L.v ~offset:1 ~strides:[| 1 |] [| 1 |],
      2,
      false );
    ("float32 [0] at byte 1", D.Any f32, L.contiguous [| 0 |], 1, true);
  ]

let test_alignment (_, D.Any dt, l, first, aligned) =
  let b = B.view (B.create Rig.host 64) ~first ~length:32 in
  if aligned then equal layout l (A.layout (A.v dt l b))
  else
    raises_match (Exn.invalid_arg ~sub:"not on a multiple") (fun () ->
        A.v dt l b)

let test_dead () =
  let b = B.create Rig.host 16 in
  kill b;
  raises_match (Exn.invalid_arg ~sub:"dead") (fun () ->
      A.v f32 (L.contiguous [| 4 |]) b)

(* The alignment [v] asks of a dtype's first element, in bytes: its width, one
   component's for complex dtypes, none below a byte. *)
let alignment (D.Any dt) =
  match D.kind dt with D.Complex -> D.bits dt / 16 | _ -> max 1 (D.bits dt / 8)

(* Bytes [n] elements of [b] bits reach, rounded up. *)
let reach_bytes b n = ((n * b) + 7) / 8
let any_dtype = Gen.of_list ~pp:(fun ppf (D.Any dt) -> D.pp ppf dt) D.all

(* A dtype, a layout and a view of a host buffer at any byte. *)
let parts =
  let open Gen in
  with_pp
    (fun ppf (D.Any dt, l, first, length) ->
      Format.fprintf ppf "%a %a, bytes %d to %d" D.pp dt L.pp l first
        (first + length))
    (let* dt = any_dtype in
     let* l = any_layout in
     let* first = int_range 0 17 in
     let (D.Any d) = dt in
     let need = reach_bytes (D.bits d) (snd (L.span l)) - first in
     let+ length = int_range (max 0 (need - 2)) (max 0 need + 2) in
     (dt, l, first, length))

let law_v (D.Any dt, l, first, length) =
  let b = B.view (B.create Rig.host (first + length)) ~first ~length in
  let within = reach_bytes (D.bits dt) (snd (L.span l)) <= length in
  let aligned = B.aligned b ~at:0 (alignment (D.Any dt)) in
  let empty = L.numel l = 0 in
  cover "no element" empty;
  cover "past the bytes" ((not empty) && not within);
  cover "misaligned" ((not empty) && within && not aligned);
  cover "accepted" ((not empty) && within && aligned);
  if empty || (within && aligned) then equal layout l (A.layout (A.v dt l b))
  else raises_match Exn.invalid_arg (fun () -> A.v dt l b)

(* [create] is contiguous at offset 0 over the bytes its elements fill, and a
   sub-byte array's last byte is zero past its last element. *)
let law_create (D.Any dt, s) =
  let s = Iarray.of_array s in
  let a = A.create Rig.host dt s in
  let n = L.numel (L.contiguous s) in
  let b = A.buffer a in
  equal layout (L.contiguous s) (A.layout a);
  equal int (D.bytes dt n) (B.length b);
  equal device Rig.host (A.device a);
  let used = n * D.bits dt mod 8 in
  cover "a partial last byte" (used > 0);
  if used > 0 then begin
    let bytes = A.v D.Uint8 (L.contiguous [| B.length b |]) b in
    let last = A.get bytes [| B.length b - 1 |] in
    equal ~msg:"bits past the last element" int 0 (last lsr used)
  end

let dtype_and_shape =
  Gen.with_pp
    (fun ppf (D.Any dt, s) -> Format.fprintf ppf "%a %a" D.pp dt pp_ints s)
    Gen.(pair any_dtype (array ~size:(int_range 0 3) (int_range 0 9)))

let memory_kind =
  Testable.make
    ~pp:(fun ppf k ->
      Format.pp_print_string ppf
        (match k with
        | B.Device -> "Device"
        | B.Pinned -> "Pinned"
        | B.Mapped -> "Mapped"))
    ~equal:( = )

(* [create ~memory] allocates [d]'s memory of that kind, [Device] by default.
   Every kind holds the array [create] makes, on the host too, whose memory is
   of one kind, but [Pinned] on a device whose memory the host addresses,
   which has none and raises as {!Rig.Buffer.create} does. Such a device's
   [Mapped] memory is its [Device] memory, the kind its driver is asked for.
   Each array has its own size, so none reuses another's memory. *)
let test_create_memory () =
  let d, driver = late () in
  let allocations () =
    let ~count, ~last = S.Late.allocations driver in
    (count, last)
  in
  let made ~msg d memory rows =
    let s : int iarray = [| rows; 3 |] in
    let a = A.create ?memory d f32 s in
    equal ~msg:(msg ^ ": layout") layout (L.contiguous s) (A.layout a);
    equal ~msg:(msg ^ ": device") device d (A.device a);
    equal ~msg:(msg ^ ": bytes") int (12 * rows) (B.length (A.buffer a))
  in
  List.iteri
    (fun k (msg, memory, kind) ->
      let count, _ = allocations () in
      made ~msg d memory (k + 1);
      equal ~msg
        (pair int (option memory_kind))
        (count + 1, Some kind) (allocations ()))
    [
      ("default", None, B.Device);
      ("Device", Some B.Device, B.Device);
      ("Mapped", Some B.Mapped, B.Device);
    ];
  raises_match Exn.invalid_arg (fun () ->
      made ~msg:"Pinned" d (Some B.Pinned) 4);
  raises_match Exn.invalid_arg (fun () ->
      made ~msg:"host, Pinned" Rig.host (Some B.Pinned) 1);
  made ~msg:"host, Mapped" Rig.host (Some B.Mapped) 2

let test_create_refuses () =
  raises_match Exn.invalid_arg (fun () -> A.create Rig.host f32 [| 2; -1 |]);
  raises_match Exn.invalid_arg (fun () ->
      A.create Rig.host f32 (Iarray.init 33 (Fun.const 1)));
  raises_match Exn.invalid_arg (fun () ->
      A.create Rig.host f32 [| max_int; 2 |])

(* Every movement keeps the array inside its buffer: [v] takes its parts
   back. *)
let law_bounds (Case (a, m)) =
  let b = Option.bind m (fun m -> A.move m a) in
  cover "a view" (Option.is_some b);
  match b with
  | None -> ()
  | Some b -> ignore (A.v (A.dtype b) (A.layout b) (A.buffer b))

let law_move (Case (a, m)) =
  let moved = Option.map (fun m -> (A.move m a, L.move m (A.layout a))) m in
  let view, none =
    match moved with
    | Some (Some _, _) -> (true, false)
    | Some (None, None) -> (false, true)
    | _ -> (false, false)
  in
  let viewed by = view && match m with Some m -> by m | None -> false in
  cover "a strided array" (not (L.is_contiguous (A.layout a)));
  cover "a C-contiguous array" (L.is_contiguous (A.layout a));
  cover "a reshape" (viewed (function M.Reshape _ -> true | _ -> false));
  cover "a broadcast" (viewed (function M.Broadcast _ -> true | _ -> false));
  cover "an axis map" (viewed (function M.Permute _ -> true | _ -> false));
  cover "a slice" (viewed (function M.Slice _ -> true | _ -> false));
  cover "a window" (viewed (function M.Window _ -> true | _ -> false));
  cover "no view" none;
  match moved with
  | None | Some (None, None) -> ()
  | Some (Some b, Some l) ->
      equal layout l (A.layout b);
      own_buffer a (A.buffer b)
  | Some (Some _, None) -> fail "move answered a view Layout.move refuses"
  | Some (None, Some _) ->
      fail "move answered None where Layout.move has a view"

(* Bitcasts *)

let test_bitcast_bytes () =
  let a = floats32 [| 2 |] [| 1.; -2. |] in
  let u = Option.get (A.bitcast D.Uint8 a) in
  equal iints [| 2; 4 |] (L.shape (A.layout u));
  equal (array int) [| 0; 0; 0x80; 0x3f; 0; 0; 0; 0xc0 |] (A.to_array u);
  let back = Option.get (A.bitcast f32 u) in
  equal layout (A.layout a) (A.layout back);
  equal (values f32) [| 1.; -2. |] (A.to_array back)

let test_bitcast_sub_byte () =
  let a = A.of_array D.Uint8 [| 2 |] [| 0x21; 0xF7 |] in
  let n = Option.get (A.bitcast D.Int4 a) in
  equal iints [| 2; 2 |] (L.shape (A.layout n));
  equal (array int) [| 1; 2; 7; -1 |] (A.to_array n);
  let bits = Option.get (A.bitcast D.Bit a) in
  equal iints [| 2; 8 |] (L.shape (A.layout bits));
  equal (array bool)
    [| true; false; false; false; false; true; false; false |]
    (Array.sub (A.to_array bits) 0 8)

let test_bitcast_refuses () =
  let a = A.of_array D.Uint8 [| 2; 4 |] (Array.init 8 Fun.id) in
  let t = Option.get (A.move (M.Permute [| 1; 0 |]) a) in
  is_none (A.bitcast D.Uint16 t);
  let s =
    Option.get
      (A.move
         (M.Slice
            [|
              { start = 0; count = 2; step = 1 };
              { start = 1; count = 2; step = 1 };
            |])
         a)
  in
  is_none (A.bitcast D.Uint16 s);
  is_none (A.bitcast f32 (A.of_array D.Uint8 [| 2; 3 |] (Array.make 6 0)));
  let e = A.create Rig.host D.Uint8 [| 0; 4 |] in
  equal (option iints) (Some [| 0 |])
    (Option.map (fun a -> L.shape (A.layout a)) (A.bitcast f32 e));
  is_none (A.bitcast f32 (A.create Rig.host D.Uint8 [| 0; 3 |]));
  let b = B.view (B.create Rig.host 32) ~first:4 ~length:16 in
  let c = A.v D.Complex64 (L.contiguous [| 2 |]) b in
  is_none (A.bitcast D.Float64 c);
  let deep = A.create Rig.host D.Uint16 (Iarray.init 32 (Fun.const 1)) in
  raises_match Exn.invalid_arg (fun () -> A.bitcast D.Uint8 deep)

(* A narrowing bitcast and its widening are inverse, over any layout of a
   byte-wide array. *)
let law_bitcast_round_trip (Case (a, m)) =
  let a =
    match m with Some m -> Option.value ~default:a (A.move m a) | None -> a
  in
  let narrows = D.bits (A.dtype a) >= 8 && L.rank (A.layout a) < L.max_rank in
  cover "a strided array" (narrows && not (L.is_contiguous (A.layout a)));
  if narrows then
    match A.bitcast D.Uint8 a with
    | None -> fail "a narrowing bitcast answered None"
    | Some u -> (
        match A.bitcast (A.dtype a) u with
        | None -> fail "widening a narrowing answered None"
        | Some back -> equal layout (A.layout a) (A.layout back))

(* The layout [bitcast] gives a layout [l] of [src] read as [dst] over [b], by
   the rule its interface states; [None] where widening's conditions fail or the
   first element is misaligned for [dst]. *)
let bitcast_reference (D.Any src) (D.Any dst) l b =
  let bs = D.bits src and bd = D.bits dst in
  let s = L.shape l and st = L.strides l and o = L.offset l in
  let r = Iarray.length s in
  let result =
    if bs = bd then Some l
    else if bs > bd then
      let k = bs / bd in
      Some
        (L.v ~offset:(o * k)
           ~strides:(Iarray.append (Iarray.map (( * ) k) st) [| 1 |])
           (Iarray.append s [| k |]))
    else
      let k = bd / bs in
      let outer = Iarray.sub s ~pos:0 ~len:(max 0 (r - 1)) in
      if r = 0 || Iarray.get s (r - 1) <> k then None
      else if L.numel l = 0 then
        Some (L.v ~strides:(Iarray.init (r - 1) (Fun.const 0)) outer)
      else
        let st' = Iarray.sub st ~pos:0 ~len:(r - 1) in
        if
          Iarray.get st (r - 1) <> 1
          || o mod k <> 0
          || Iarray.exists (fun t -> t mod k <> 0) st'
        then None
        else
          Some
            (L.v ~offset:(o / k)
               ~strides:(Iarray.map (fun t -> t / k) st')
               outer)
  in
  Option.bind result (fun l' ->
      if L.numel l' = 0 || B.aligned b ~at:0 (alignment (D.Any dst)) then
        Some l'
      else None)

(* Two dtypes and a layout of the first over a host buffer at any byte where its
   elements are aligned: often the narrowing of a layout of the second, so that
   widening has its trailing axis. *)
let bitcast_case =
  let open Gen in
  with_pp
    (fun ppf (D.Any src, D.Any dst, l, first) ->
      Format.fprintf ppf "%a to %a, %a at byte %d" D.pp src D.pp dst L.pp l
        first)
    (let* (D.Any src as s) = any_dtype in
     let* (D.Any dst as d) = any_dtype in
     let* l0 = any_layout in
     let k = D.bits dst / D.bits src in
     let* narrowed = bool in
     let l =
       if narrowed && k >= 2 && L.rank l0 < L.max_rank then
         L.v
           ~offset:(L.offset l0 * k)
           ~strides:
             (Iarray.append (Iarray.map (( * ) k) (L.strides l0)) [| 1 |])
           (Iarray.append (L.shape l0) [| k |])
       else l0
     in
     let+ slot = int_range 0 4 in
     (s, d, l, slot * alignment s))

let law_bitcast (D.Any src, (D.Any dst as d), l, first) =
  let length = reach_bytes (D.bits src) (snd (L.span l)) in
  let b = B.view (B.create Rig.host (length + 64)) ~first ~length in
  let a = A.v src l b in
  let bs = D.bits src and bd = D.bits dst in
  cover "equal widths" (bs = bd);
  cover "narrowing" (bs > bd);
  (* A narrowing would append an axis past max_rank, which raises; its
     fixed case is test_bitcast_refuses's. *)
  let deep = bs > bd && L.rank l = L.max_rank in
  let reference = if deep then None else bitcast_reference (D.Any src) d l b in
  cover "widening" (bs < bd && Option.is_some reference);
  cover "widening refused" ((not deep) && Option.is_none reference);
  if deep then raises_match Exn.invalid_arg (fun () -> A.bitcast dst a)
  else
    match (reference, A.bitcast dst a) with
    | Some l', Some c ->
        equal layout l' (A.layout c);
        satisfies ~claim:"over the source's buffer" pass
          (fun c -> A.buffer c == b)
          c
    | None, None -> ()
    | Some l', None -> failf "None where %a was expected" L.pp l'
    | None, Some c -> failf "%a where None was expected" L.pp (A.layout c)

let test_expect () =
  let a = floats32 [| 1 |] [| 3. |] in
  equal (values f32) [| 3. |] (A.to_array (A.expect f32 (A.Any a)));
  match A.expect D.Float64 (A.Any a) with
  | _ -> fail "expect returned"
  | exception Invalid_argument m ->
      contains ~sub:"float32" m;
      contains ~sub:"float64" m

(* Elements *)

let law_set_get (Case (a, _)) =
  let check : type v s. (v, s) A.t -> unit =
   fun a ->
    let dt = A.dtype a in
    List.iteri
      (fun k idx ->
        let x = D.of_float dt (Float.of_int ((k * 37) - 100) /. 4.) in
        let idx = Iarray.of_array idx in
        A.set a idx x;
        equal (value dt) x (A.get a idx))
      (indices (Iarray.to_array (L.shape (A.layout a))))
  in
  check a

(* An [int] outside its dtype's range is one the dtype does not hold, refused
   by [set] and [of_array], and one inside stores as itself. *)
let law_range (type s) (dt : (int, s) D.t) =
  let lo = D.min_value dt and hi = D.max_value dt in
  let x =
    Gen.frequency
      [
        (4, Gen.int_range (lo - 3) (hi + 3));
        ( 2,
          Gen.of_list ~pp:Format.pp_print_int
            [ min_int; lo - 1; lo; hi; hi + 1; max_int ] );
      ]
  in
  prop ~count:400 (D.name dt)
    ~examples:[ min_int; lo - 1; lo; hi; hi + 1; max_int ]
    x
    (fun x ->
      let a = A.create Rig.host dt [| 1 |] in
      let outside = x < lo || x > hi in
      cover "out of range" outside;
      cover "a bound" (x = lo || x = hi);
      if outside then begin
        raises_match Exn.invalid_arg (fun () -> A.set a [| 0 |] x);
        raises_match Exn.invalid_arg (fun () -> A.of_array dt [| 1 |] [| x |])
      end
      else begin
        A.set a [| 0 |] x;
        equal int x (A.get a [| 0 |]);
        equal (array int) [| x |] (A.to_array (A.of_array dt [| 1 |] [| x |]))
      end)

let ranges =
  [
    law_range D.Int4;
    law_range D.Uint4;
    law_range D.Int8;
    law_range D.Uint8;
    law_range D.Int16;
    law_range D.Uint16;
  ]

let test_of_array_refuses () =
  raises_match Exn.invalid_arg (fun () -> A.of_array f32 [| 2; 2 |] [| 1. |]);
  raises_match Exn.invalid_arg (fun () -> A.of_array f32 [| 1 |] [| 1.; 2. |]);
  raises_match Exn.invalid_arg (fun () -> A.of_array f32 [| -1 |] [||]);
  let a = A.of_array f32 [| 0; 3 |] [||] in
  equal device Rig.host (A.device a);
  equal layout (L.contiguous [| 0; 3 |]) (A.layout a)

(* Indices of a [2; 3] array outside its shape: past an extent, below 0, of
   another number of entries. *)
let outside = [ [| 2; 0 |]; [| 0; 3 |]; [| 0; -1 |]; [| 0 |]; [| 0; 0; 0 |] ]

let test_index i =
  let a = floats32 [| 2; 3 |] (Array.make 6 0.) in
  let i = Iarray.of_array i in
  raises_match ~msg:"get" Exn.invalid_arg (fun () -> A.get a i);
  raises_match ~msg:"set" Exn.invalid_arg (fun () -> A.set a i 1.)

(* [a]'s dtype and layout over the host's borrow of mapped pages whose memory
   admits only reads. *)
let read_only a =
  let io = S.mapped_read_only a in
  A.v (A.dtype a) (A.layout a) (Option.get (B.borrow Rig.host (A.buffer io)))

let test_set_twice () =
  let a = floats32 [| 3 |] [| 1.; 2.; 3. |] in
  let b = Option.get (A.move (M.Broadcast [| 2; 3 |]) a) in
  raises_match (Exn.invalid_arg ~sub:"twice") (fun () -> A.set b [| 0; 0 |] 5.)

let test_set_read_only () =
  let a = read_only (floats32 [| 2 |] [| 1.; 2. |]) in
  raises_match
    (fun e ->
      Exn.invalid_arg ~sub:"Nx_array.set" e
      && Exn.invalid_arg ~sub:"read-only" e)
    (fun () -> A.set a [| 0 |] 5.);
  equal (values f32) [| 1.; 2. |] (A.to_array a)

let test_off_host_refuses () =
  let d = S.off_host () in
  let a = A.to_device d (floats32 [| 2 |] [| 1.; 2. |]) in
  let fails f = raises_match (Exn.invalid_arg ~sub:"host") f in
  fails (fun () -> A.get a [| 0 |]);
  fails (fun () -> A.set a [| 0 |] 0.);
  fails (fun () -> A.to_array a);
  raises_match (Exn.invalid_arg ~sub:"Nx_array.bigarray") (fun () ->
      A.bigarray a)

let test_dead_access () =
  let a = floats32 [| 2 |] [| 1.; 2. |] in
  kill (A.buffer a);
  let fails f =
    raises_match
      (fun e ->
        Exn.invalid_arg ~sub:"dead" e
        || Exn.invalid_arg ~sub:"consumed by the test" e)
      f
  in
  fails (fun () -> A.get a [| 0 |]);
  raises_match ~msg:"get names itself" (Exn.invalid_arg ~sub:"Nx_array.get")
    (fun () -> A.get a [| 0 |]);
  fails (fun () -> A.to_array a);
  fails (fun () -> A.bitcast D.Uint32 a);
  fails (fun () -> A.bitcast D.Uint8 a)

let test_dead_bigarray () =
  let a = A.to_device (S.off_host ()) (floats32 [| 2 |] [| 1.; 2. |]) in
  kill (A.buffer a);
  raises_match (Exn.invalid_arg ~sub:"dead") (fun () ->
      A.bigarray a)

let test_dead_empty () =
  let b = B.create Rig.host 16 in
  let a = A.v f32 (L.contiguous [| 0 |]) b in
  kill b;
  raises_match (Exn.invalid_arg ~sub:"dead") (fun () -> A.to_device Rig.host a);
  raises_match (Exn.invalid_arg ~sub:"consumed by the test") (fun () ->
      A.to_array a)

(* A copy of memory the host does not address is a fresh array on its device:
   a transposed view on such a device, copied there. *)
let test_copy_off_host () =
  let d = S.off_host () in
  let m = A.to_device d (floats32 [| 2; 3 |] [| 0.; 1.; 2.; 3.; 4.; 5. |]) in
  let t = Option.get (A.move (M.Permute [| 1; 0 |]) m) in
  let c = A.copy t in
  equal ~msg:"device" device d (A.device c);
  equal ~msg:"layout" layout (L.contiguous [| 3; 2 |]) (A.layout c);
  fresh m c;
  equal ~msg:"elements" (values f32) [| 0.; 3.; 1.; 4.; 2.; 5. |]
    (A.to_array (A.to_device Rig.host c))

(* Writes to the elements of one byte from two domains keep each other. *)
let int4s = abstract "a"
let element = Gen.int_range 0 3
let nibble = Gen.int_range (-8) 7

let int4_commands =
  [
    command "create"
      (Gen.unit @-> makes int4s)
      (fun () -> Array.make 4 0)
      (fun () -> A.of_array D.Int4 [| 4 |] (Array.make 4 0));
    command "set"
      (element @-> nibble @-> int4s ^-> returns unit)
      (fun i x m -> m.(i) <- x)
      (fun i x a -> A.set a [| i |] x);
    command "get"
      (element @-> int4s ^-> returns int)
      (fun i m -> m.(i))
      (fun i a -> A.get a [| i |]);
  ]

(* A gather into elements 1 and 2 of four int4s writes two bytes, each shared
   with a neighbour, element 0 or 3: stores to those from another domain are
   kept. The source starts on a byte or inside one. The gather stores no whole
   byte, which only elements of its own run would share: two gathers from two
   domains still race on elements 1 and 2, so only elements 0 and 3 are read. *)
let gather_into phase xs a =
  let all =
    A.of_array D.Int4 [| phase + 2 |] (Array.append (Array.make phase 0) xs)
  in
  let range start = M.Slice [| { M.start; count = 2; step = 1 } |] in
  let src = Option.get (A.move (range phase) all) in
  S.copy_into (Option.get (A.move (range 1) a)) src

let gather_commands =
  let nibbles = Gen.array ~size:(const 2) nibble in
  let neighbour = Gen.map (fun last -> if last then 3 else 0) Gen.bool in
  [
    command "create"
      (Gen.unit @-> makes int4s)
      (fun () -> Array.make 4 0)
      (fun () -> A.of_array D.Int4 [| 4 |] (Array.make 4 0));
    command "copy"
      (Gen.int_range 0 1 @-> nibbles @-> int4s ^-> returns answer)
      (fun _ xs m ->
        Array.blit xs 0 m 1 2;
        A.Done)
      gather_into;
    command "set"
      (neighbour @-> nibble @-> int4s ^-> returns unit)
      (fun i x m -> m.(i) <- x)
      (fun i x a -> A.set a [| i |] x);
    command "get"
      (neighbour @-> int4s ^-> returns int)
      (fun i m -> m.(i))
      (fun i a -> A.get a [| i |]);
  ]

(* to_device of [count] int4s from element 1, the first sharing a byte with
   element 0, while another domain stores element 0 and the element after the
   view: the copy holds the view's elements, and the stores are kept. Four
   elements reach three bytes and pack; five reach three and keep the layout,
   copying element 0's half byte. *)
let to_device_commands count =
  let n = count + 2 in
  let inner =
    Array.init count (fun i -> if i mod 2 = 0 then i + 1 else -i - 1)
  in
  let start = Array.concat [ [| 0 |]; inner; [| 0 |] ] in
  let neighbour = Gen.map (fun last -> if last then n - 1 else 0) Gen.bool in
  let placed a =
    let view = M.Slice [| { M.start = 1; count; step = 1 } |] in
    A.to_array (A.to_device Rig.host (Option.get (A.move view a)))
  in
  [
    command "create"
      (Gen.unit @-> makes int4s)
      (fun () -> Array.copy start)
      (fun () -> A.of_array D.Int4 [| n |] start);
    command "to_device"
      (int4s ^-> returns (array int))
      (fun m -> Array.sub m 1 count)
      placed;
    command "set"
      (neighbour @-> nibble @-> int4s ^-> returns unit)
      (fun i x m -> m.(i) <- x)
      (fun i x a -> A.set a [| i |] x);
    command "get"
      (neighbour @-> int4s ^-> returns int)
      (fun i m -> m.(i))
      (fun i a -> A.get a [| i |]);
  ]

(* Bulk access *)

let law_round_trip (Case (a, _)) =
  let check : type v s. (v, s) A.t -> unit =
   fun a ->
    let dt = A.dtype a in
    let xs = A.to_array a in
    equal (values dt) xs (A.to_array (A.of_array dt (L.shape (A.layout a)) xs))
  in
  check a

(* [to_array] reads any layout in C order of indices, as [get] does. *)
let law_to_array (Case (a, m)) =
  let check : type v s. (v, s) A.t -> unit =
   fun a -> equal (values (A.dtype a)) (gets a) (A.to_array a)
  in
  let b = Option.bind m (fun m -> A.move m a) in
  cover "a strided view"
    (match b with
    | Some b -> not (L.is_contiguous (A.layout b))
    | None -> false);
  match b with Some b -> check b | None -> check a

let test_copy () =
  let a = floats32 [| 2; 3 |] [| 0.; 1.; 2.; 3.; 4.; 5. |] in
  let t = Option.get (A.move (M.Permute [| 1; 0 |]) a) in
  let c = A.copy t in
  equal layout (L.contiguous [| 3; 2 |]) (A.layout c);
  equal (values f32) [| 0.; 3.; 1.; 4.; 2.; 5. |] (A.to_array c);
  fresh a c;
  A.set c [| 0; 0 |] 9.;
  equal (values f32) [| 0. |] [| A.get a [| 0; 0 |] |]

(* NaN payloads of both kinds and -0., as float32 bits. *)
let test_nan_payloads () =
  let bits = [| 0x7fc00001l; 0xffa00002l; 0x80000000l; 0x7f800001l |] in
  let f = Option.get (A.bitcast f32 (A.of_array D.Uint32 [| 4 |] bits)) in
  let back a = A.to_array (Option.get (A.bitcast D.Uint32 a)) in
  equal ~msg:"copy" (array int32) bits (back (A.copy f));
  equal ~msg:"to_device" (array int32) bits (back (A.to_device Rig.host f));
  let dst = A.create Rig.host f32 [| 4 |] in
  A.transfer ~src:f ~dst;
  equal ~msg:"transfer" (array int32) bits (back dst)

let law_copy (Case (a, m)) =
  let check : type v s. (v, s) A.t -> unit =
   fun a ->
    let c = A.copy a in
    equal layout (L.contiguous (L.shape (A.layout a))) (A.layout c);
    equal (values (A.dtype a)) (A.to_array a) (A.to_array c)
  in
  let b = Option.bind m (fun m -> A.move m a) in
  cover "a strided view"
    (match b with
    | Some b -> not (L.is_contiguous (A.layout b))
    | None -> false);
  match b with Some b -> check b | None -> check a

(* [transfer] on the host into the transpose of a fresh array, a destination
   whose layout is not C order for two axes or more. *)
let law_transfer_layouts (Case (a, m)) =
  let check : type v s. (v, s) A.t -> unit =
   fun a ->
    let s = L.shape (A.layout a) in
    let r = Iarray.length s in
    let rev x = Iarray.init r (fun i -> Iarray.get x (r - 1 - i)) in
    let base = A.create Rig.host (A.dtype a) (rev s) in
    let dst =
      Option.get (A.move (M.Permute (rev (Iarray.init r Fun.id))) base)
    in
    cover "a strided destination" (not (L.is_contiguous (A.layout dst)));
    A.transfer ~src:a ~dst;
    equal (values (A.dtype a)) (A.to_array a) (A.to_array dst)
  in
  let b = Option.bind m (fun m -> A.move m a) in
  cover "a strided source"
    (match b with
    | Some b -> not (L.is_contiguous (A.layout b))
    | None -> false);
  match b with Some b -> check b | None -> check a

(* [transfer] on the host writes its destination's elements alone: an int4
   window starting inside a byte keeps its neighbours, and an array stored
   into itself is unchanged. *)
let test_transfer_writes () =
  let base = A.of_array D.Int4 [| 6 |] [| 1; 2; 3; 4; 5; 6 |] in
  let mid =
    Option.get (A.move (M.Slice [| { start = 1; count = 3; step = 1 } |]) base)
  in
  A.transfer ~src:(A.of_array D.Int4 [| 3 |] [| -1; -2; -3 |]) ~dst:mid;
  equal ~msg:"a window" (values D.Int4) [| 1; -1; -2; -3; 5; 6 |]
    (A.to_array base);
  A.transfer ~src:base ~dst:base;
  equal ~msg:"itself" (values D.Int4) [| 1; -1; -2; -3; 5; 6 |]
    (A.to_array base)

(* Two windows of one uint8 array that share no byte, [(first, first', n)]:
   the source from [first] and the destination from [first'], [n] bytes each,
   in a 16-byte array, the destination up to 2 bytes after or before the
   source, so often inside one of rig's words with it. *)
let windows =
  let open Gen in
  let* n = int_range 1 5 in
  let* first = int_range 0 (16 - n) in
  let+ after = bool and+ gap = int_range 0 2 in
  let fits f = f >= 0 && f + n <= 16 in
  let later = first + n + gap and earlier = first - n - gap in
  let first' =
    if after && fits later then later
    else if fits earlier then earlier
    else first + n
  in
  (first, first', n)

(* Where an array's memory lies: on the host, on a device whose memory the
   host does not address, or in an io library's region, which the host reads
   and writes through the library's calls. *)
type lies = Host_memory | Off_host_memory | Io_memory

let lies =
  Gen.of_list
    ~pp:(fun ppf l ->
      Format.pp_print_string ppf
        (match l with
        | Host_memory -> "host memory"
        | Off_host_memory -> "off the host"
        | Io_memory -> "io memory"))
    [ Host_memory; Off_host_memory; Io_memory ]

(* An array of [a]'s elements and layout whose memory lies as [l] says. *)
let lying l a =
  match l with
  | Host_memory -> a
  | Off_host_memory -> A.to_device (S.off_host ()) a
  | Io_memory -> S.on_io a

(* [x], or its elements moved to the host if the host does not address it. *)
let on_host x =
  if Rig.reaches Rig.host (A.device x) then x else A.to_device Rig.host x

(* [transfer] between two windows of one array that share no byte stores the
   source's elements into the destination, and leaves every other element:
   windows that share a word of their memory and no byte included, wherever
   the array's memory lies. *)
let law_transfer_windows ((first, first', n), l) =
  let bytes = Array.init 16 (fun i -> i + 1) in
  let base = lying l (A.of_array D.Uint8 [| 16 |] bytes) in
  cover "in io memory" (l = Io_memory);
  cover "off the host" (l = Off_host_memory);
  let window first =
    Option.get
      (A.move (M.Slice [| { start = first; count = n; step = 1 } |]) base)
  in
  let w = Rig.word in
  cover "windows that share a word and no byte"
    ((first + n - 1) / w = first' / w || (first' + n - 1) / w = first / w);
  A.transfer ~src:(window first) ~dst:(window first');
  let expected = Array.copy bytes in
  Array.blit bytes first expected first' n;
  equal (values D.Uint8) expected (A.to_array (on_host base))

(* Arrays of the byte-wide dtypes [dts] over drawn bytes, NaN payloads, non-0/1
   bools and every other pattern included, and a movement of them. *)
let raw_of dts =
  let open Gen in
  let* (D.Any dt) = of_list ~pp:(fun ppf (D.Any dt) -> D.pp ppf dt) dts in
  let* s = shape in
  let w = D.bits dt / 8 in
  let n = Array.fold_left ( * ) 1 s * w in
  let* bytes = array ~size:(const n) (int_range 0 255) in
  let u = A.of_array D.Uint8 (Iarray.of_array (Array.append s [| w |])) bytes in
  let a = Option.get (A.bitcast dt u) in
  let+ m =
    option (movement ~apart:false (Iarray.to_array (L.shape (A.layout a))))
  in
  Case (a, m)

let raw_of dts = Gen.with_pp pp_case (raw_of dts)
let raw = raw_of (List.filter (fun (D.Any dt) -> D.bits dt >= 8) D.all)

(* The bytes of [a]'s elements in C order of indices. *)
let bytes_of a = A.to_array (Option.get (A.bitcast D.Uint8 a))

(* The bytes of each element of a host array and a movement of it, then of copy,
   to_device and transfer of the same view of its elements lying where [src]
   says, into memory lying where [dst] says, read back on the host. *)
let law_bits_kept (Case (a, m), (src, dst)) =
  let view a = Option.value ~default:a (Option.bind m (fun m -> A.move m a)) in
  let want = bytes_of (view a) in
  let a = view (lying src a) in
  cover "a strided view" (not (L.is_contiguous (A.layout a)));
  cover "from memory off the host" (src <> Host_memory);
  cover "into memory off the host" (dst <> Host_memory);
  let bytes x = bytes_of (on_host x) in
  (* An io device makes no memory, so a copy there raises as create does. *)
  if src <> Io_memory then equal ~msg:"copy" (array int) want (bytes (A.copy a));
  equal ~msg:"to_device to the host" (array int) want
    (bytes (A.to_device Rig.host a));
  equal ~msg:"to_device off the host" (array int) want
    (bytes (A.to_device (S.off_host ()) a));
  let into = lying dst (A.create Rig.host (A.dtype a) (L.shape (A.layout a))) in
  A.transfer ~src:a ~dst:into;
  equal ~msg:"transfer" (array int) want (bytes into)

(* A bool element is true iff its byte is not zero. *)
let law_bool_bytes (Case (a, m)) =
  let a = Option.value ~default:a (Option.bind m (fun m -> A.move m a)) in
  let a = A.expect D.Bool (A.Any a) in
  let bytes = bytes_of a in
  cover "a byte other than 0 and 1" (Array.exists (fun b -> b > 1) bytes);
  let want = Array.map (fun b -> b <> 0) bytes in
  equal ~msg:"to_array" (array bool) want (A.to_array a);
  equal ~msg:"get" (array bool) want (gets a)

(* Gathers of views about 256 bytes a side

   A gather copies the runs a layout coalesces into, each through the block
   copy: views of two or three axes whose extents straddle 256 bytes and four
   elements, over drawn bytes, checked against the bytes each index's position
   names. *)

(* The extent, in elements of [dt], of 256 bytes. *)
let side (D.Any dt) = 256 / max 1 (D.bits dt / 8)

type tiles = Tiles : ('v, 's) A.t * string -> tiles

let pp_tiles ppf (Tiles (a, _)) =
  Format.fprintf ppf "%a %a" D.pp (A.dtype a) L.pp (A.layout a)

(* [n] bytes from the seed [seed]. *)
let bytes_of_seed seed n =
  let x = ref (seed land 0xFFFFFFFFFFFF) in
  String.init n (fun _ ->
      x := ((!x * 0x5DEECE66D) + 11) land 0xFFFFFFFFFFFF;
      Char.chr ((!x lsr 24) land 0xFF))

(* A view of an array of [dt] over drawn bytes: transposed, sliced, a row
   broadcast, its last axis from the second element, or as it is, with
   extents about 256 bytes, and one in seventeen past 512. *)
let tiles =
  let open Gen in
  let* (D.Any dt as d) = any_dtype in
  let t = side d in
  let extent =
    frequency
      [
        (12, ints_of [ 1; 3; 4; 5; t - 1; t; t + 1 ]);
        (4, int_range 1 ((2 * t) + 5));
        (1, constant ((2 * t) + 1));
      ]
  in
  let* outer =
    frequency [ (2, constant []); (1, map (fun k -> [ k ]) (int_range 1 3)) ]
  in
  let* rows = extent in
  let* cols = extent in
  let s = Array.of_list (outer @ [ rows; cols ]) in
  let n = Array.fold_left ( * ) 1 s in
  let* seed = int in
  let src = bytes_of_seed seed (D.bytes dt n) in
  let a = A.v dt (L.contiguous (Iarray.of_array s)) (B.of_string src) in
  let r = Array.length s in
  let reverse = M.Permute (Iarray.init r (fun i -> r - 1 - i)) in
  let swap =
    M.Permute
      (Iarray.init r (fun i -> if i >= r - 2 then (2 * r) - 3 - i else i))
  in
  let slice =
    let axis d =
      frequency
        [
          (1, constant { M.start = 0; count = d; step = 1 });
          (1, constant { M.start = d - 1; count = d; step = -1 });
          (2, range d);
        ]
    in
    let rec all i =
      if i = r then constant []
      else
        let+ x = axis s.(i) and+ xs = all (i + 1) in
        x :: xs
    in
    let+ rs = all 0 in
    M.Slice (Iarray.of_list rs)
  in
  let row =
    M.Slice
      (Iarray.init r (fun i ->
           if i = r - 2 then { M.start = 0; count = 1; step = 1 }
           else { M.start = 0; count = s.(i); step = 1 }))
  in
  (* The last axis from its second element: a sub-byte view then starts
     inside a byte. *)
  let shift =
    M.Slice
      (Iarray.init r (fun i ->
           let d = s.(i) in
           if i = r - 1 && d > 1 then { M.start = 1; count = d - 1; step = 1 }
           else { M.start = 0; count = d; step = 1 }))
  in
  let+ moves =
    frequency
      [
        (1, constant []);
        (1, constant [ swap ]);
        (1, constant [ reverse ]);
        (1, map (fun m -> [ m; swap ]) slice);
        (1, constant [ row; M.Broadcast (Iarray.of_array s) ]);
        (1, constant [ row; M.Broadcast (Iarray.of_array s); swap ]);
        (2, constant [ shift ]);
      ]
  in
  let a =
    List.fold_left
      (fun a m -> match A.move m a with Some b -> b | None -> a)
      a moves
  in
  Tiles (a, src)

let tiles = Gen.with_pp pp_tiles tiles

(* The bytes of [a]'s elements in C order of indices, read from [src], the bytes
   of its buffer, at the positions its layout gives: a contiguous array's bytes,
   its last byte's bits past its last element zero. *)
let gathered src bits l =
  let s = Iarray.to_array (L.shape l) in
  let n = Array.fold_left ( * ) 1 s in
  let out = Bytes.make (((n * bits) + 7) / 8) '\000' in
  let r = Array.length s in
  let idx = Array.make r 0 in
  for k = 0 to n - 1 do
    let p = ref (L.offset l) in
    Array.iteri (fun i j -> p := !p + (j * L.stride l i)) idx;
    (if bits >= 8 then
       let w = bits / 8 in
       Bytes.blit_string src (!p * w) out (k * w) w
     else
       let at = !p * bits in
       let v =
         (Char.code src.[at / 8] lsr (at mod 8)) land ((1 lsl bits) - 1)
       in
       let to_ = k * bits in
       let byte = Char.code (Bytes.get out (to_ / 8)) in
       Bytes.set out (to_ / 8) (Char.chr (byte lor (v lsl (to_ mod 8)))));
    (* The next index in C order. *)
    let i = ref (r - 1) in
    while
      !i >= 0
      &&
      (idx.(!i) <- idx.(!i) + 1;
       idx.(!i) = s.(!i))
    do
      idx.(!i) <- 0;
      decr i
    done
  done;
  Bytes.unsafe_to_string out

let widths = [ 1; 2; 4; 8; 16 ]

(* What a store of the value [x] of [dt] holds: a float format's infinity
   saturates where the format has no store for it. *)
let stored : type v s. (v, s) D.t -> v -> v =
 fun dt x -> match D.kind dt with D.Float -> D.of_float dt x | _ -> x

let law_tiles (Tiles (a, src)) =
  let dt = A.dtype a in
  let l = A.layout a in
  let bits = D.bits dt in
  let t = side (D.Any dt) in
  let ds = L.shape l in
  let at_edge k = Iarray.exists (fun d -> d = k) ds in
  cover "an extent of 256 bytes less one element" (at_edge (t - 1));
  cover "an extent of 256 bytes" (at_edge t);
  cover "an extent of 256 bytes and one element" (at_edge (t + 1));
  cover "an extent past 512 bytes" (Iarray.exists (fun d -> d > 2 * t) ds);
  cover "an extent past 4 and no multiple of 4"
    (Iarray.exists (fun d -> d > 4 && d mod 4 <> 0) ds);
  cover "a transposed view"
    (L.rank l >= 2
    && abs (L.stride l (L.rank l - 2)) = 1
    && L.dim l (L.rank l - 2) > 1);
  cover "a broadcast axis"
    (Iarray.exists2 (fun d st -> d > 1 && st = 0) ds (L.strides l));
  cover "a reversed axis" (Iarray.exists (fun st -> st < 0) (L.strides l));
  cover "sub-byte elements" (bits < 8);
  cover "a sub-byte view starting inside a byte"
    (bits < 8 && L.offset l * bits mod 8 <> 0);
  cover "sub-byte rows ending inside a byte"
    (bits < 8 && L.dim l (L.rank l - 1) * bits mod 8 <> 0);
  List.iter (fun w -> cover (strf "%d-byte elements" w) (bits = 8 * w)) widths;
  let want = gathered src bits l in
  let c = A.copy a in
  let got = Bytes.create (B.length (A.buffer c)) in
  B.blit_to_bytes (A.buffer c) 0 got 0 (Bytes.length got);
  equal ~msg:"copy" string want (Bytes.to_string got);
  let r = A.v dt (L.contiguous ds) (B.of_string want) in
  let xs = A.to_array r in
  equal ~msg:"to_array" (values dt) xs (A.to_array a);
  equal ~msg:"of_array" (values dt)
    (Array.map (stored dt) xs)
    (A.to_array (A.of_array dt ds xs))

(* The elements of [l]: one per index, an axis of stride 0 counted once. *)
let count l =
  if L.numel l = 0 then 0
  else
    Iarray.fold_left ( * ) 1
      (Iarray.mapi (fun a d -> if L.stride l a = 0 then 1 else d) (L.shape l))

(* The layout [to_device] gives a layout [l] of [bits]-bit elements, by its
   .mli, and whether it keeps [l]: [l] shifted to the copy where the bytes of
   its span are no more than its elements', one per index with an axis of stride
   0 counted once; otherwise [l]'s stride order from position 0. *)
let moved bits l =
  let r = L.rank l in
  let axes = List.init r Fun.id in
  let n = count l in
  let lo, hi = L.span l in
  let first = lo * bits / 8 in
  if reach_bytes bits hi - first <= reach_bytes bits n then
    ( `Kept,
      L.v
        ~offset:(L.offset l - (8 * first / bits))
        ~strides:(L.strides l) (L.shape l) )
  else
    let order =
      List.stable_sort
        (fun a b -> compare (abs (L.stride l b)) (abs (L.stride l a)))
        (List.filter (fun a -> L.stride l a <> 0) axes)
    in
    let strides = Array.make r 0 and offset = ref 0 and next = ref 1 in
    List.iter
      (fun a ->
        let t = L.stride l a in
        strides.(a) <- (if t < 0 then - !next else !next);
        if t < 0 then offset := !offset + ((L.dim l a - 1) * !next);
        next := !next * L.dim l a)
      (List.rev order);
    let strides = Iarray.of_array strides in
    (`Packed, L.v ~offset:!offset ~strides (L.shape l))

let check_moved ?(msg = "") d a =
  let c = A.to_device d a in
  let bits = D.bits (A.dtype a) in
  let how, want = moved bits (A.layout a) in
  equal ~msg:(msg ^ "device") device d (A.device c);
  equal ~msg:(msg ^ "layout") layout want (A.layout c);
  let lo, hi = L.span want in
  let bytes =
    match how with
    | `Kept -> reach_bytes bits hi - (lo * bits / 8)
    | `Packed -> reach_bytes bits hi
  in
  equal ~msg:(msg ^ "bytes") int bytes (B.length (A.buffer c));
  at_most
    ~msg:(msg ^ "no larger than the elements")
    int
    ~than:(reach_bytes bits (count (A.layout a)))
    (B.length (A.buffer c));
  let on_host x =
    if Rig.reaches Rig.host (A.device x) then x else A.to_device Rig.host x
  in
  equal ~msg:(msg ^ "elements")
    (values (A.dtype a))
    (A.to_array (on_host a))
    (A.to_array (on_host c));
  (how, c)

(* An array and a movement of it, as often kept by to_device as packed. *)
let to_place =
  Gen.with_pp pp_case
    (Gen.map (fun (A.Any a, m) -> Case (a, m)) Nx_array_gen.to_device_case)

let law_to_device (Case (a, m)) =
  let check : type v s. (v, s) A.t -> unit =
   fun a ->
    let how, c = check_moved Rig.host a in
    let l = A.layout a and bits = D.bits (A.dtype a) in
    cover "a packed copy" (how = `Packed);
    cover "a packed sub-byte copy" (how = `Packed && bits < 8);
    cover "a kept strided view" (how = `Kept && not (L.is_contiguous l));
    cover "a sub-byte span sharing an end byte"
      (how = `Kept && bits < 8
      && L.numel l > 0
      &&
      let lo, hi = L.span l in
      lo * bits mod 8 <> 0 || hi * bits mod 8 <> 0);
    if B.length (A.buffer c) > 0 then fresh a c
  in
  let b = Option.bind m (fun m -> A.move m a) in
  cover "a view" (Option.is_some b);
  match b with Some b -> check b | None -> check a

let slice rs a =
  Option.get
    (A.move
       (M.Slice
          (Iarray.of_list
             (List.map
                (fun (start, count, step) -> { M.start; count; step })
                rs)))
       a)

let iota dt s =
  let n = Iarray.fold_left ( * ) 1 s in
  A.of_array dt s
    (Array.init n (fun i -> D.of_float dt (float_of_int (i mod 7))))

(* A column of a 1024 x 1024 matrix moves its 4 KiB alone, at stride 1. *)
let test_to_device_column () =
  let n = 1024 in
  let m = iota f32 [| n; n |] in
  let col = slice [ (0, n, 1); (3, 1, 1) ] m in
  let c = A.to_device Rig.host col in
  equal ~msg:"bytes" int (4 * n) (B.length (A.buffer c));
  equal layout (L.v ~strides:[| 1; 0 |] [| n; 1 |]) (A.layout c);
  equal (values f32) (A.to_array col) (A.to_array c)

let test_to_device_layouts () =
  let moves name dt want a =
    let c = A.to_device Rig.host a in
    equal ~msg:name layout want (A.layout c);
    equal ~msg:name (values dt) (A.to_array a) (A.to_array c)
  in
  let m = iota f32 [| 8; 8 |] in
  moves "overlap" f32
    (L.v ~strides:[| 2; 1 |] [| 3; 2 |])
    (A.v f32 (L.v ~strides:[| 2; 2 |] [| 3; 2 |]) (A.buffer m));
  moves "broadcast of a column" f32
    (L.v ~strides:[| 1; 0 |] [| 8; 4 |])
    (Option.get
       (A.move (M.Broadcast [| 8; 4 |]) (slice [ (0, 8, 1); (2, 1, 1) ] m)));
  moves "transposed slice" f32
    (L.v ~strides:[| 1; 3 |] [| 3; 4 |])
    (Option.get
       (A.move (M.Permute [| 1; 0 |]) (slice [ (0, 4, 2); (1, 3, 2) ] m)));
  moves "reversed column" f32
    (L.v ~offset:7 ~strides:[| -1; 0 |] [| 8; 1 |])
    (slice [ (7, 8, -1); (5, 1, 1) ] m);
  let q = iota D.Int4 [| 4; 4 |] in
  moves "int4 column" D.Int4
    (L.v ~strides:[| 1; 0 |] [| 4; 1 |])
    (slice [ (0, 4, 1); (1, 1, 1) ] q);
  moves "window with a gap, kept" f32
    (L.v ~strides:[| 4; 1; 1 |] [| 2; 2; 2 |])
    (A.v f32 (L.v ~offset:9 ~strides:[| 4; 1; 1 |] [| 2; 2; 2 |]) (A.buffer m))

(* transfer stores its source's elements, wherever its memory lies, into the
   elements at odd indices of an array whose memory lies anywhere, and no
   other element; a sub-byte destination off the host raises. *)
let law_transfer (Case (a, m), (src_lies, dst_lies)) =
  let check : type v s. (v, s) A.t -> unit =
   fun src ->
    let dt = A.dtype src and s = L.shape (A.layout src) in
    let wide = Iarray.map (fun d -> (2 * d) + 1) s in
    let n = Iarray.fold_left ( * ) 1 wide in
    let before =
      A.of_array dt wide
        (Array.init n (fun k -> D.of_float dt (float_of_int (k mod 5))))
    in
    let big = lying dst_lies (A.copy before) in
    let dst = slice (Iarray.to_list (Iarray.map (fun d -> (1, d, 2)) s)) big in
    cover "from off the host" (src_lies = Off_host_memory);
    cover "from io memory" (src_lies = Io_memory);
    cover "into off the host" (dst_lies = Off_host_memory);
    cover "into io memory" (dst_lies = Io_memory);
    if D.bits dt < 8 && dst_lies <> Host_memory then
      raises_match (Exn.invalid_arg ~sub:"host") (fun () ->
          A.transfer ~src ~dst)
    else begin
      A.transfer ~src ~dst;
      equal ~msg:"elements" (values dt)
        (A.to_array (on_host src))
        (A.to_array (on_host dst));
      let outside k =
        let r = ref k and out = ref false in
        for ax = Iarray.length wide - 1 downto 0 do
          if !r mod Iarray.get wide ax mod 2 = 0 then out := true;
          r := !r / Iarray.get wide ax
        done;
        !out
      in
      let others xs =
        Array.of_list (List.filteri (fun k _ -> outside k) (Array.to_list xs))
      in
      equal ~msg:"other elements" (values dt)
        (others (A.to_array before))
        (others (A.to_array (on_host big)))
    end
  in
  let base = lying src_lies a in
  match Option.bind m (fun m -> A.move m base) with
  | Some b -> check b
  | None -> check base

(* Held devices. A Polled device runs its queue only when the test runs it
   (Rig_support.Polled). The host addresses neither device's memory; one maps
   host memory and the other does not. Each one's queue waits for the other's
   work, as a GPU's for a peer's of its driver, and holds the parts of every
   copy a case submits: a full queue makes a submit wait for its work. *)

module Polled = Rig_support.Polled

let capacity = 1 lsl 16

(* The two devices and their queues, opened by the first test that uses them. *)
let held_device =
  lazy
    (Polled.open_ ~capacity ~host_visible:false ~waits_on:[ `Host ]
       "nx-array-held")

let flat_device =
  lazy
    (Polled.open_ ~capacity ~host_visible:false ~maps_host:false
       ~waits_on:[ `Host ] "nx-array-flat")

let held () = fst (Lazy.force held_device)
let flat () = fst (Lazy.force flat_device)

(* Runs every queued copy. *)
let drain () =
  List.iter
    (fun q ->
      while Polled.run q > 0 do
        ()
      done)
    [ snd (Lazy.force held_device); snd (Lazy.force flat_device) ]

(* The bytes of [a]'s buffer, read on the host. *)
let buffer_bytes a =
  let a =
    if Rig.reaches Rig.host (A.device a) then a else A.to_device Rig.host a
  in
  let b = A.buffer a in
  let s = Bytes.create (B.length b) in
  B.blit_to_bytes b 0 s 0 (B.length b);
  Bytes.to_string s

type side = On_host | Held | Flat

let device_of = function
  | On_host -> Rig.host
  | Held -> held ()
  | Flat -> flat ()

let pp_side ppf s =
  Format.pp_print_string ppf
    (match s with On_host -> "host" | Held -> "held" | Flat -> "flat")

(* Where a transfer's arrays lie: [Sides (s, d)] for a source on [s] and a
   destination on [d]; [Shared k] for two views of one allocation, on the
   host and on a held device:
   - [Interleaved], whose spans share bytes and whose elements do not;
   - [Itself], one view transferred onto itself;
   - [Turned], a square's transpose onto the square;
   - [Shifted], a window onto the window one row on. *)
type shared = Interleaved | Itself | Turned | Shifted

type route = Sides of side * side | Shared of shared

let pp_route ppf = function
  | Sides (s, d) -> Format.fprintf ppf "%a to %a" pp_side s pp_side d
  | Shared k ->
      Format.pp_print_string ppf
        (match k with
        | Interleaved -> "interleaved"
        | Itself -> "itself"
        | Turned -> "turned"
        | Shifted -> "shifted")

let route =
  let sides = [ On_host; Held; Flat ] in
  let pairs =
    List.concat_map
      (fun s ->
        List.filter_map
          (fun d ->
            if s = On_host && d = On_host then None else Some (Sides (s, d)))
          sides)
      sides
  in
  let shared = [ Interleaved; Itself; Turned; Shifted ] in
  Gen.with_pp pp_route
    (Gen.frequency
       (List.map (fun k -> (2, Gen.constant (Shared k))) shared
       @ List.map (fun r -> (1, Gen.constant r)) pairs))

(* The elements of [a], of shape [s], at odd indices along every axis, in
   order or, [flip], reversed along each. *)
let odd ~flip s a =
  let range d =
    if flip then { M.start = (2 * d) - 1; count = d; step = -2 }
    else { M.start = 1; count = d; step = 2 }
  in
  Option.get (A.move (M.Slice (Iarray.map range s)) a)

let pattern dt s =
  let n = Iarray.fold_left ( * ) 1 s in
  A.of_array dt s
    (Array.init n (fun k -> D.of_float dt (float_of_int (k mod 5))))

(* Element [k] along a last axis of extent 2 of [base], an array of shape [s]
   and that axis, as an array of shape [s], reversed along each axis where
   [flip]. *)
let half ~flip s base k =
  let r = Iarray.length s in
  let range i =
    if i < r then { M.start = 0; count = Iarray.get s i; step = 1 }
    else { M.start = k; count = 1; step = 1 }
  in
  let w = Option.get (A.move (M.Slice (Iarray.init (r + 1) range)) base) in
  let w = Option.get (A.move (M.Reshape s) w) in
  if not flip then w
  else
    let back d = { M.start = d - 1; count = d; step = -1 } in
    Option.get (A.move (M.Slice (Iarray.map back s)) w)

(* transfer through held devices stores what the host's transfer stores
   between host arrays of the same layouts: every byte of the destination's
   buffer, its other elements included, ends as the host's. A sub-byte
   destination the host does not address raises. *)
let law_transfer_held ((Case (a, m), flip), r) =
  let check : type v s. (v, s) A.t -> (v, s) A.t -> unit =
   fun a v ->
    let dt = A.dtype v and s = L.shape (A.layout v) in
    let bits = D.bits dt in
    let n = L.numel (A.layout v) in
    let sides p =
      match r with Sides (ss, ds) -> p ss ds | Shared _ -> false
    in
    let shared p =
      match r with Shared k -> bits >= 8 && p k | Sides _ -> false
    in
    cover "flipped destination" flip;
    cover "into a held device"
      (sides (fun ss ds -> ss = On_host && ds <> On_host));
    cover "from a held device"
      (sides (fun ss ds -> ss <> On_host && ds = On_host));
    cover "between held devices"
      (sides (fun ss ds -> ss <> On_host && ds <> On_host));
    cover "a sub-byte source off the host"
      (sides (fun ss _ -> bits < 8 && ss <> On_host));
    cover "spans that share bytes" (shared (fun k -> k = Interleaved && n > 1));
    cover "onto itself" (shared (fun k -> k = Itself));
    cover "a transpose onto itself" (shared (fun k -> k = Turned));
    cover "a window onto the next" (shared (fun k -> k = Shifted && n > 0));
    let refused =
      match r with
      | Sides (ss, ds) ->
          let wide = Iarray.map (fun d -> (2 * d) + 1) s in
          let before = pattern dt wide in
          let reference = A.copy before in
          A.transfer ~src:v ~dst:(odd ~flip s reference);
          let src =
            if ss = On_host then v
            else
              let on = A.to_device (device_of ss) a in
              match Option.bind m (fun m -> A.move m on) with
              | Some w -> w
              | None -> on
          in
          let big = A.to_device (device_of ds) before in
          let dst = odd ~flip s big in
          if bits < 8 && ds <> On_host then
            raises_match (Exn.invalid_arg ~sub:"host") (fun () ->
                A.transfer ~src ~dst)
          else begin
            A.transfer ~src ~dst;
            drain ();
            equal string ~msg:"bytes" (buffer_bytes reference)
              (buffer_bytes big)
          end;
          false
      | Shared k ->
          (* A held device stores no element narrower than a byte. *)
          assume (bits >= 8);
          let r = Iarray.length s in
          let base_shape, views =
            match k with
            | Interleaved | Itself ->
                let first = if k = Itself then 1 else 0 in
                ( Iarray.append s [| 2 |],
                  fun b -> (half ~flip s b first, half ~flip s b 1) )
            | Turned ->
                ( [| 3; 3 |],
                  fun b -> (Option.get (A.move (M.Permute [| 1; 0 |]) b), b) )
            | Shifted ->
                let range start i =
                  if i = 0 then { M.start; count = 2; step = 1 }
                  else { M.start = 0; count = Iarray.get s (i - 1); step = 1 }
                in
                let rows start = M.Slice (Iarray.init (r + 1) (range start)) in
                let from, onto = if flip then (1, 0) else (0, 1) in
                ( Iarray.append [| 3 |] s,
                  fun b ->
                    ( Option.get (A.move (rows from) b),
                      Option.get (A.move (rows onto) b) ) )
          in
          let before = pattern dt base_shape in
          let outcome base =
            let src, dst = views base in
            match A.transfer ~src ~dst with
            | () -> "stored"
            | exception Invalid_argument _ -> "refused"
          in
          let reference = A.copy before in
          let expected = outcome reference in
          let big = A.to_device (held ()) before in
          let actual = outcome big in
          equal string ~msg:"outcome" expected actual;
          drain ();
          equal string ~msg:"bytes" (buffer_bytes reference) (buffer_bytes big);
          expected = "refused"
    in
    cover "refused" refused
  in
  match Option.bind m (fun m -> A.move m a) with
  | Some b -> check a b
  | None -> check a a

(* A sub-byte array created on a held device leaves the fill of its last
   byte queued, and once it has run the bits past the last element read zero:
   in memory that a freed buffer filled with ones held, which rig may give the
   array. *)
let test_create_fill_queued () =
  let held = held () in
  let dirty = B.create held 3 in
  B.fill dirty '\xff';
  ignore (B.free dirty : bool);
  drain ();
  let a = A.create held D.Int4 [| 5 |] in
  less int ~msg:"the fill is queued" ~than:(Rig.submitted held)
    (Rig.signaled held);
  drain ();
  let s = buffer_bytes a in
  equal int ~msg:"bytes" 3 (String.length s);
  equal int ~msg:"the padding" 0 (Char.code s.[2] lsr 4)

(* Each refusal names transfer, on the host and off it. *)
let test_transfer_refusals () =
  let a = floats32 [| 2 |] [| 1.; 2. |] in
  let fails what substring f =
    raises_match ~msg:what
      (fun e ->
        Exn.invalid_arg ~sub:"Nx_array.transfer" e
        && Exn.invalid_arg ~sub:substring e)
      f
  in
  List.iter
    (fun (where, d) ->
      fails (where ^ ": another shape") "do not fit" (fun () ->
          A.transfer ~src:a ~dst:(A.create d f32 [| 3 |]));
      fails (where ^ ": a broadcast destination") "twice" (fun () ->
          let one = A.create d f32 [| 1 |] in
          A.transfer ~src:a ~dst:(Option.get (A.move (M.Broadcast [| 2 |]) one))))
    [ ("host", Rig.host); ("off the host", S.off_host ()) ];
  let dead = A.create Rig.host f32 [| 2 |] in
  kill (A.buffer dead);
  fails "a dead destination" "consumed by the test" (fun () ->
      A.transfer ~src:a ~dst:dead);
  let b = floats32 [| 4 |] [| 1.; 2.; 3.; 4. |] in
  let part first =
    Option.get (A.move (M.Slice [| { start = first; count = 3; step = 1 } |]) b)
  in
  fails "a destination overlapping the source" "shares bytes" (fun () ->
      A.transfer ~src:(part 0) ~dst:(part 1));
  fails "a Read destination" "read-only" (fun () ->
      A.transfer ~src:a ~dst:(read_only (floats32 [| 2 |] [| 0.; 0. |])));
  (* An empty destination on Read memory raises wherever the memory lies. *)
  let empty = A.create Rig.host f32 [| 0 |] in
  let read_only b = A.v f32 (L.contiguous [| 0 |]) b in
  let paged = S.mapped_read_only (A.create Rig.host f32 [| 4 |]) in
  fails "an empty Read destination off the host" "read-only" (fun () ->
      A.transfer ~src:empty ~dst:(read_only (S.read_only 0)));
  fails "an empty Read destination on the host" "read-only" (fun () ->
      A.transfer ~src:empty
        ~dst:(read_only (Option.get (B.borrow Rig.host (A.buffer paged)))))

(* A bitcast is a view on any device: it keeps the buffer and moves no byte. *)
let memory = lazy (Result.get_ok (Rig.software "nx-test-memory"))

(* A borrow shares its source's memory, dtype and layout, on any device that
   maps it, and back. *)
let law_borrow (Case (a, m)) =
  let check : type v s. (v, s) A.t -> unit =
   fun a ->
    let d = Lazy.force memory in
    let b = Option.get (A.borrow d a) in
    equal ~msg:"device" device d (A.device b);
    equal iints (L.shape (A.layout a)) (L.shape (A.layout b));
    equal iints (L.strides (A.layout a)) (L.strides (A.layout b));
    equal int (L.offset (A.layout a)) (L.offset (A.layout b));
    equal (values (A.dtype a)) (A.to_array a) (A.to_array b);
    if B.length (A.buffer a) > 0 then
      satisfies ~claim:"shares the source's memory" pass
        (fun b -> B.overlaps (A.buffer a) (A.buffer b))
        b;
    let back = Option.get (A.borrow Rig.host b) in
    equal ~msg:"back" (values (A.dtype a)) (A.to_array a) (A.to_array back)
  in
  let b = Option.bind m (fun m -> A.move m a) in
  cover "a view" (Option.is_some b);
  match b with Some b -> check b | None -> check a

let test_borrow_own () =
  let a = floats32 [| 2 |] [| 1.; 2. |] in
  satisfies ~claim:"the array itself" pass
    (fun b -> b == a)
    (Option.get (A.borrow Rig.host a))

let test_borrow_unmapped () =
  let a = A.to_device (S.off_host ()) (floats32 [| 2 |] [| 1.; 2. |]) in
  is_none (A.borrow Rig.host a)

let test_borrow_writes () =
  let a = floats32 [| 2 |] [| 1.; 2. |] in
  let b = Option.get (A.borrow (Lazy.force memory) a) in
  A.set b [| 1 |] 5.;
  equal float_exact 5. (A.get a [| 1 |])

let test_borrow_dead () =
  let dead = floats32 [| 2 |] [| 1.; 2. |] in
  kill (A.buffer dead);
  raises_match (Exn.invalid_arg ~sub:"dead") (fun () ->
      A.borrow (Lazy.force memory) dead)

(* A closed device's arrays, borrowed onto it and from the host. *)
let test_borrow_lost () =
  let a = floats32 [| 2 |] [| 1.; 2. |] in
  let closed = Result.get_ok (Rig.software "nx-test-borrow-closed") in
  let on_closed = A.to_device closed a in
  Rig.close closed;
  raises_match ~msg:"from the host" lost (fun () -> A.borrow closed a);
  raises_match ~msg:"on its own device" lost (fun () ->
      A.borrow closed on_closed)

let test_bitcast_devices () =
  let x = floats32 [| 2 |] [| 1.; -2. |] in
  let want = bytes_of x in
  let bitcast d =
    let a = A.to_device d x in
    let before = S.off_host_allocations () in
    let u = Option.get (A.bitcast D.Uint8 a) in
    equal ~msg:"allocations" int before (S.off_host_allocations ());
    own_buffer a (A.buffer u);
    equal ~msg:"device" device d (A.device u);
    u
  in
  equal ~msg:"software device" (array int) want
    (A.to_array (bitcast (Lazy.force memory)));
  equal ~msg:"off the host" (array int) want
    (A.to_array (A.to_device Rig.host (bitcast (S.off_host ()))))

(* A sub-byte view of an io device's array moves to the host. *)
let test_to_device_io () =
  let a = S.on_io (A.of_array D.Int4 [| 7 |] [| 1; -2; 3; -4; 5; -6; 7 |]) in
  let s =
    Option.get (A.move (M.Slice [| { start = 5; count = 3; step = -2 } |]) a)
  in
  let back = A.to_device Rig.host s in
  equal device Rig.host (A.device back);
  equal (array int) [| -6; -4; -2 |] (A.to_array back);
  equal layout (L.v ~offset:2 ~strides:[| -1 |] [| 3 |]) (A.layout back)

(* Views of an io device's array move through the host, to the host and to a
   device whose memory the host does not address, as does a sub-byte view off a
   software device. *)
let test_to_device_io_views () =
  let off = S.off_host () in
  let m = S.on_io (iota f32 [| 6; 5 |]) in
  let views =
    [
      ("column", slice [ (0, 6, 1); (2, 1, 1) ] m);
      ("reversed rows, strided", slice [ (5, 3, -2); (0, 3, 2) ] m);
      ( "transposed",
        Option.get
          (A.move (M.Permute [| 1; 0 |]) (slice [ (0, 3, 2); (0, 5, 1) ] m)) );
    ]
  in
  List.iter
    (fun (name, v) ->
      ignore (check_moved ~msg:(name ^ " to the host: ") Rig.host v);
      ignore (check_moved ~msg:(name ^ " off the host: ") off v))
    views;
  let q = S.on_io (iota D.Int4 [| 5; 3 |]) in
  ignore
    (check_moved ~msg:"int4 column: " Rig.host
       (slice [ (0, 5, 1); (1, 1, 1) ] q));
  let mem = A.to_device (Lazy.force memory) (iota D.Int4 [| 5; 3 |]) in
  ignore
    (check_moved ~msg:"int4 column off the host: " off
       (slice [ (0, 5, 1); (1, 1, 1) ] mem))

(* Bigarrays *)

let test_bigarray () =
  let a = floats32 [| 2; 3 |] [| 0.; 1.; 2.; 3.; 4.; 5. |] in
  let g = A.bigarray a in
  equal ints [| 2; 3 |] (Bigarray.Genarray.dims g);
  Bigarray.Genarray.set g [| 1; 2 |] 9.;
  equal (values f32) [| 9. |] [| A.get a [| 1; 2 |] |];
  let t = Option.get (A.move (M.Permute [| 1; 0 |]) a) in
  raises_match (Exn.invalid_arg ~sub:"not C-contiguous") (fun () ->
      A.bigarray t);
  let row =
    Option.get
      (A.move
         (M.Slice
            [|
              { start = 1; count = 1; step = 1 };
              { start = 0; count = 3; step = 1 };
            |])
         a)
  in
  let g = A.bigarray row in
  equal float_exact 9. (Bigarray.Genarray.get g [| 0; 2 |]);
  claimed ~msg:"never taken after bigarray" (A.buffer a)

(* Bigarray's kind for a dtype, where Bigarray has the format. *)
let kind : type v s. (v, s) D.t -> (v, s) Bigarray.kind option = function
  | D.Float64 -> Some Bigarray.float64
  | D.Float32 -> Some Bigarray.float32
  | D.Float16 -> Some Bigarray.float16
  | D.Int64 -> Some Bigarray.int64
  | D.Int32 -> Some Bigarray.int32
  | D.Int16 -> Some Bigarray.int16_signed
  | D.Uint16 -> Some Bigarray.int16_unsigned
  | D.Int8 -> Some Bigarray.int8_signed
  | D.Uint8 -> Some Bigarray.int8_unsigned
  | D.Complex128 -> Some Bigarray.complex64
  | D.Complex64 -> Some Bigarray.complex32
  | _ -> None

(* [bigarray] of a host array holds its elements if the array is C-contiguous,
   of rank at most 16 and of a dtype Bigarray has a kind for, and raises
   naming itself otherwise. *)
let law_bigarray (Case (a, m)) =
  let check : type v s. (v, s) A.t -> unit =
   fun a ->
    let l = A.layout a in
    let has_kind = Option.is_some (kind (A.dtype a)) in
    let fits = L.is_contiguous l && L.rank l <= 16 in
    cover "no kind" (not has_kind);
    cover "raises" (has_kind && not fits);
    cover "a bigarray" (has_kind && fits);
    match kind (A.dtype a) with
    | None ->
        raises_match (Exn.invalid_arg ~sub:"Nx_array.bigarray") (fun () ->
            A.bigarray a)
    | Some _ when not fits ->
        raises_match (Exn.invalid_arg ~sub:"Nx_array.bigarray") (fun () ->
            A.bigarray a)
    | Some _ ->
        let g = A.bigarray a in
        let s = Iarray.to_array (L.shape l) in
        equal ints s (Bigarray.Genarray.dims g);
        equal
          (values (A.dtype a))
          (A.to_array a)
          (Array.of_list (List.map (Bigarray.Genarray.get g) (indices s)))
  in
  match Option.bind m (fun m -> A.move m a) with
  | Some b -> check b
  | None -> check a

let test_of_bigarray_misaligned () =
  raises_match Exn.invalid_arg (fun () ->
      A.of_bigarray D.Int16 (S.int16_at 1 4));
  let a = A.of_bigarray D.Int16 (S.int16_at 2 4) in
  equal iints [| 4 |] (L.shape (A.layout a))

let test_bigarray_rank () =
  let a = A.create Rig.host D.Uint8 (Iarray.init 16 (Fun.const 1)) in
  equal ints (Array.make 16 1)
    (Bigarray.Genarray.dims (A.bigarray a));
  let a = A.create Rig.host D.Uint8 (Iarray.init 17 (Fun.const 1)) in
  raises_match (Exn.invalid_arg ~sub:"at most 16") (fun () ->
      A.bigarray a)

(* [bigarray] waits for the work on its array's memory: a Late device's store
   into a host array it borrowed, pending until a wait sleeps on the device. A
   host buffer of [paged_from] bytes starts on a page, which the Late device
   maps: 64 KiB, and 256 KiB on 64 KiB pages. *)
let test_wait_bigarray () =
  let d, _ = late () in
  let a = A.create Rig.host f32 [| Rig.Buffer.paged_from / 4 |] in
  pending (Option.get (A.borrow d a)) [| 3.; 4. |];
  satisfies ~claim:"pending: signaled below submitted" (pair int int)
    (fun (signaled, submitted) -> signaled < submitted)
    (Rig.signaled d, Rig.submitted d);
  let g = A.bigarray a in
  equal (values f32) [| 3.; 4. |]
    (Array.init 2 (fun i -> Bigarray.Genarray.get g [| i |]))

let test_of_bigarray () =
  let g =
    Bigarray.Genarray.create Bigarray.int16_signed Bigarray.c_layout [| 2; 2 |]
  in
  Bigarray.Genarray.fill g 3;
  let a = A.of_bigarray D.Int16 g in
  equal string "int16" (D.name (A.dtype a));
  Bigarray.Genarray.set g [| 1; 0 |] (-7);
  equal (array int) [| 3; 3; -7; 3 |] (A.to_array a);
  A.set a [| 0; 1 |] 11;
  equal int 11 (Bigarray.Genarray.get g [| 0; 1 |])

(* Scoped reads *)

let read_fails ?msg substring f =
  raises_match ?msg
    (fun e ->
      Exn.invalid_arg ~sub:"Nx_array.read" e && Exn.invalid_arg ~sub:substring e)
    f

(* [read] passes a bigarray of a C-contiguous array's shape holding its
   elements, on the host and on a software device, and refuses a strided one and
   a dtype Bigarray has no kind for. It leaves no claim. *)
let law_read (Case (a, m)) =
  let check : type v s. (v, s) A.t -> Rig.Buffer.t =
   fun a ->
    let l = A.layout a in
    let contiguous = L.is_contiguous l in
    cover "contiguous" contiguous;
    cover "strided" (not contiguous);
    cover "an offset" (contiguous && L.offset l > 0);
    cover "no element" (L.numel l = 0);
    cover "no kind" (Option.is_none (kind (A.dtype a)));
    (match kind (A.dtype a) with
    | None ->
        read_fails (if contiguous then "Bigarray kind" else "contiguous")
          (fun () -> A.read a ignore)
    | Some _ when contiguous ->
        let s = Iarray.to_array (L.shape l) in
        let elements g =
          Array.of_list (List.map (Bigarray.Genarray.get g) (indices s))
        in
        let dims, xs =
          A.read a (fun g -> (Bigarray.Genarray.dims g, elements g))
        in
        equal ints s dims;
        equal (values (A.dtype a)) (A.to_array a) xs
    | Some _ -> read_fails "contiguous" (fun () -> A.read a ignore));
    A.buffer (retaken a)
  in
  let a =
    match Option.bind m (fun m -> A.move m a) with Some b -> b | None -> a
  in
  (* Each check takes the memory, which [a] then reads through the buffer
     the take gave. *)
  let over b = A.v (A.dtype a) (A.layout a) b in
  let a = over (check a) in
  (* The elements after the first as a vector: C-contiguous at an offset. *)
  let l = A.layout a in
  let n = L.numel l in
  let a =
    if L.is_contiguous l && n >= 2 then
      over
        (check
           (A.v (A.dtype a)
              (L.v ~offset:(L.offset l + 1) ~strides:[| 1 |] [| n - 1 |])
              (A.buffer a)))
    else a
  in
  ignore (check (Option.get (A.borrow (Lazy.force memory) a)))

(* [write] passes a bigarray of a C-contiguous array's shape over its own
   bytes: it reads the array's elements, and what it stores is the array's
   elements afterwards. Here it stores them in reverse. It refuses a strided
   array and a dtype Bigarray has no kind for, and leaves no claim. *)
let law_write (Case (a, m)) =
  let check : type v s. (v, s) A.t -> unit =
   fun a ->
    let l = A.layout a in
    let contiguous = L.is_contiguous l in
    cover "contiguous" contiguous;
    cover "strided" (not contiguous);
    cover "no element" (L.numel l = 0);
    cover "no kind" (Option.is_none (kind (A.dtype a)));
    let fails sub =
      raises_match
        (fun e ->
          Exn.invalid_arg ~sub:"Nx_array.write" e && Exn.invalid_arg ~sub e)
        (fun () -> A.write a ignore)
    in
    (match kind (A.dtype a) with
    | None -> if contiguous then fails "Bigarray kind"
    | Some _ when contiguous ->
        let s = Iarray.to_array (L.shape l) in
        let idxs = Array.of_list (indices s) in
        let n = Array.length idxs in
        let before = A.to_array a in
        let seen =
          A.write a (fun g ->
              let xs = Array.map (Bigarray.Genarray.get g) idxs in
              Array.iteri
                (fun k i -> Bigarray.Genarray.set g i xs.(n - 1 - k))
                idxs;
              (Bigarray.Genarray.dims g, xs))
        in
        equal ints s (fst seen);
        equal ~msg:"read through the bigarray" (values (A.dtype a)) before
          (snd seen);
        equal ~msg:"stored through the bigarray" (values (A.dtype a))
          (Array.init n (fun k -> before.(n - 1 - k)))
          (A.to_array a)
    | Some _ -> fails "contiguous");
    released (A.buffer a)
  in
  check
    (match Option.bind m (fun m -> A.move m a) with Some b -> b | None -> a)

(* A write claims the memory while [f] runs and ends the claim when it returns
   or raises. *)
let test_write_claim () =
  let a = floats32 [| 2 |] [| 1.; 2. |] in
  A.write a (fun _ -> claimed ~msg:"claimed while f runs" (A.buffer a));
  let a = retaken a in
  raises Raised (fun () -> A.write a (fun _ -> raise Raised));
  ignore (retaken ~msg:"released after a raise" a)

(* [host] makes a C-contiguous array of the shape on the host, which [write]
   fills. *)
let test_host () =
  let a = A.host D.Int64 [| 2; 3 |] in
  equal ~msg:"on the host" device Rig.host (A.device a);
  equal ~msg:"its shape" iints
    (Iarray.of_array [| 2; 3 |])
    (L.shape (A.layout a));
  A.write a (fun g ->
      Bigarray.Genarray.fill g 0L;
      Bigarray.Genarray.set g [| 1; 2 |] 7L);
  equal (array int64) [| 0L; 0L; 0L; 0L; 0L; 7L |] (A.to_array a)

(* The claim holds while [f] runs and ends when it returns or raises, leaving
   the memory to later donations. *)
let test_read_claim () =
  let a = floats32 [| 2 |] [| 1.; 2. |] in
  A.read a (fun _ -> claimed ~msg:"claimed while f runs" (A.buffer a));
  let a = retaken a in
  raises Raised (fun () -> A.read a (fun _ -> raise Raised));
  ignore (retaken ~msg:"released after a raise" a)

let test_read_refuses () =
  let off = A.to_device (S.off_host ()) (floats32 [| 2 |] [| 1.; 2. |]) in
  read_fails ~msg:"off the host" "host" (fun () ->
      A.read off ignore);
  let dead = floats32 [| 2 |] [| 1.; 2. |] in
  kill (A.buffer dead);
  read_fails ~msg:"dead" "consumed by the test" (fun () ->
      A.read dead ignore);
  let a = A.create Rig.host D.Uint8 (Iarray.init 16 (Fun.const 1)) in
  equal ~msg:"16 axes" ints (Array.make 16 1)
    (A.read a Bigarray.Genarray.dims);
  let a = A.create Rig.host D.Uint8 (Iarray.init 17 (Fun.const 1)) in
  read_fails ~msg:"17 axes" "17" (fun () ->
      A.read a ignore)

(* An array with no element passes an empty bigarray, wherever its memory
   lies. *)
let test_read_empty () =
  let dims d = A.read (A.create d f32 [| 2; 0 |]) Bigarray.Genarray.dims in
  equal ~msg:"on the host" ints [| 2; 0 |] (dims Rig.host);
  equal ~msg:"off the host" ints [| 2; 0 |] (dims (S.off_host ()))

let test_write_empty () =
  let dims d = A.write (A.create d f32 [| 2; 0 |]) Bigarray.Genarray.dims in
  equal ~msg:"on the host" ints [| 2; 0 |] (dims Rig.host);
  equal ~msg:"off the host" ints [| 2; 0 |] (dims (S.off_host ()))

let test_wait_read () =
  let d, _ = late () in
  let a = on_late d [| 0.; 0. |] in
  pending a [| 3.; 4. |];
  equal (values f32) [| 3.; 4. |]
    (A.read a (fun g ->
         Array.init 2 (fun i -> Bigarray.Genarray.get g [| i |])));
  released (A.buffer a)

(* Calls over an array of a lost device: its work faulted, or the device
   closed. *)
let faulted () =
  let d, l = late () in
  let x = on_late d [| 1.; 2. |] in
  pending x [| 3.; 4. |];
  S.Late.fault l "gone";
  x

let closed () =
  let d, _ = late () in
  let y = on_late d [| 1.; 2. |] in
  Rig.close d;
  y

(* read and write over a lost device's array raise Lost at the claim, before
   their function runs. *)
let test_read_lost () =
  let read x = A.read x (fun _ -> fail "read ran its function") in
  raises_match ~msg:"faulted work" lost (fun () -> read (faulted ()));
  raises_match ~msg:"a closed device" lost (fun () -> read (closed ()))

let test_write_lost () =
  let write x = A.write x (fun _ -> fail "write ran its function") in
  raises_match ~msg:"faulted work" lost (fun () -> write (faulted ()));
  raises_match ~msg:"a closed device" lost (fun () -> write (closed ()))

(* A host array a Late device borrowed and has pending work on, which faults:
   bigarray's wait raises Lost. *)
let test_bigarray_lost () =
  let d, l = late () in
  let a = A.create Rig.host f32 [| Rig.Buffer.paged_from / 4 |] in
  pending (Option.get (A.borrow d a)) [| 3.; 4. |];
  S.Late.fault l "gone";
  raises_match lost (fun () -> A.bigarray a)

(* get and to_array read after the work on their array, and raise Lost where
   that work's device is lost. *)
let test_wait_get () =
  let d, _ = late () in
  let a = on_late d [| 0.; 0. |] in
  pending a [| 3.; 4. |];
  equal float_exact 4. (A.get a [| 1 |]);
  released (A.buffer a)

let test_wait_to_array () =
  let d, _ = late () in
  let a = on_late d [| 0.; 0. |] in
  pending a [| 3.; 4. |];
  equal ~msg:"float32" (values f32) [| 3.; 4. |] (A.to_array a);
  let b = A.create d D.Int32 [| 2 |] in
  pending b [| Int32.float_of_bits 5l; Int32.float_of_bits 6l |];
  equal ~msg:"int32" (values D.Int32) [| 5l; 6l |] (A.to_array b)

let test_get_lost () = raises_match lost (fun () -> A.get (faulted ()) [| 0 |])

let test_to_array_lost () =
  raises_match lost (fun () -> A.to_array (faulted ()))

let test_wait_copy () =
  let d, _ = late () in
  let a = on_late d [| 0.; 0. |] in
  pending a [| 3.; 4. |];
  equal (values f32) [| 3.; 4. |] (A.to_array (A.copy a))

(* Writes refused by memory that admits only reads. *)
let test_write_read_only () =
  let a = read_only (floats32 [| 2 |] [| 1.; 2. |]) in
  raises_match
    (fun e ->
      Exn.invalid_arg ~sub:"Nx_array.write" e
      && Exn.invalid_arg ~sub:"read-only" e)
    (fun () -> A.write a ignore)

(* A float array of to_array is a flat float array, where the compiler has
   them. *)
let test_flat_floats () =
  if Obj.tag (Obj.repr [| 0.; 1. |]) <> Obj.double_array_tag then
    skip ~reason:"the compiler has no flat float arrays" ();
  satisfies ~claim:"a flat float array" pass
    (fun xs -> Obj.tag (Obj.repr xs) = Obj.double_array_tag)
    (A.to_array (floats32 [| 3 |] [| 1.; 2.; 3. |]))

let test_bigarray_off_rig_host () =
  let a = A.to_device (Lazy.force memory) (floats32 [| 2 |] [| 1.; 2. |]) in
  raises_match (Exn.invalid_arg ~sub:"Nx_array.bigarray") (fun () ->
      A.bigarray a)

let tests =
  [
    group "v"
      [
        cases
          "takes a layout within its buffer's bytes and refuses one past them"
          ~id:(fun (n, _, _, _, _) -> n)
          within test_within;
        cases "refuses a misaligned first element"
          ~id:(fun (n, _, _, _, _) -> n)
          alignments test_alignment;
        test "refuses a dead buffer" test_dead;
        prop ~count:2000
          "takes a layout within its buffer's bytes on an aligned first \
           element, and only those"
          parts law_v;
        prop "every movement keeps an array in its buffer" case law_bounds;
      ];
    group "create"
      [
        prop ~count:400
          "is C-contiguous at offset 0 over its bytes, its tail bits zero"
          dtype_and_shape law_create;
        test "allocates the kind of memory it is given" test_create_memory;
        test "refuses what contiguous refuses" test_create_refuses;
        test "on a held device leaves its tail's fill queued"
          test_create_fill_queued;
      ];
    test "host makes a C-contiguous host array that write fills" test_host;
    test "expect recovers the dtype, or raises naming both" test_expect;
    (* One case in a hundred is a movement no view expresses. *)
    prop ~count:3000 "move is Layout.move over the same buffer" viewed law_move;
    group "bitcast"
      [
        test "a narrowing reads the bytes, its widening returns"
          test_bitcast_bytes;
        test "sub-byte elements lie LSB first" test_bitcast_sub_byte;
        test "a widening refuses strides it cannot divide" test_bitcast_refuses;
        prop ~count:400 "a narrowing then its widening is the identity" case
          law_bitcast_round_trip;
        prop ~count:1000 "follows its rule over any layout" bitcast_case
          law_bitcast;
        test "is a view on a software device and off the host"
          test_bitcast_devices;
      ];
    group "get"
      [
        prop "reads what set stores" case law_set_get;
        cases "refuses an index outside the shape"
          ~id:(fun i -> Format.asprintf "%a" pp_ints i)
          outside test_index;
        prop "a bool reads true iff its byte is not zero"
          (raw_of [ D.Any D.Bool ]) law_bool_bytes;
        test "reads after the work on its array" test_wait_get;
        test "raises Lost for a lost device's work" test_get_lost;
        test "refuses memory the host does not address" test_off_host_refuses;
        test "refuses a dead buffer" test_dead_access;
      ];
    group "set"
      [
        test "refuses an array that reaches an element twice" test_set_twice;
        test "refuses memory that admits only reads" test_set_read_only;
        group "refuses an int outside its dtype's range" ranges;
        (* Each case runs its program 50 times on two domains: 100 cases
           whatever --prop-count asks. *)
        stateful ~domains:2 ~count:100
          "writes to one byte from two domains keep each other" int4_commands;
      ];
    group "to_array"
      [
        prop ~count:300 "reads any layout in C order" case law_to_array;
        test "of floats is a flat float array" test_flat_floats;
        test "reads after the work on its array" test_wait_to_array;
        test "raises Lost for a lost device's work" test_to_array_lost;
        test "refuses a dead buffer under no element" test_dead_empty;
      ];
    group "of_array"
      [
        prop "then to_array is the identity" case law_round_trip;
        test "refuses a count other than the shape's" test_of_array_refuses;
      ];
    group "placement"
      [
        prop ~count:300
          "copy, to_device and transfer keep every byte of every element, on \
           any devices"
          (Gen.pair raw (Gen.pair lies lies))
          law_bits_kept;
        test "copy, to_device and transfer keep NaN payloads" test_nan_payloads;
        prop ~count:500
          "copy, to_array and of_array keep every element of views about 256 \
           bytes a side"
          tiles law_tiles;
      ];
    group "to_device"
      [
        prop ~count:400 "copies the elements, keeping a layout that fits them"
          to_place law_to_device;
        test "moves a column's elements alone" test_to_device_column;
        test "packs in stride order" test_to_device_layouts;
        test "moves a sub-byte view of an io device to the host"
          test_to_device_io;
        test "packs views of io and software devices on the host"
          test_to_device_io_views;
        stateful ~domains:2 ~count:100
          "packs a sub-byte view while another domain stores its neighbours"
          (to_device_commands 4);
        stateful ~domains:2 ~count:100
          "copies a sub-byte span while another domain stores its neighbours"
          (to_device_commands 5);
      ];
    group "transfer"
      [
        prop ~count:300 "stores any layout's elements into any layout" case
          law_transfer_layouts;
        test "writes its destination's elements alone" test_transfer_writes;
        prop "between windows of one array that share no byte stores the source"
          (Gen.pair windows lies) law_transfer_windows;
        prop "stores the elements alone, wherever they lie"
          (Gen.pair case (Gen.pair lies lies))
          law_transfer;
        prop ~count:600 "through held devices stores what the host's does"
          (Gen.pair (Gen.pair case Gen.bool) route)
          law_transfer_held;
        test "refuses what it cannot store, naming itself"
          test_transfer_refusals;
        stateful ~domains:2 ~count:100
          "a gather keeps the other elements of its end bytes, written from \
           another domain"
          gather_commands;
      ];
    group "copy"
      [
        test "gathers into a fresh buffer" test_copy;
        test "makes a fresh array off the host" test_copy_off_host;
        prop ~count:300 "is contiguous with the same elements" case law_copy;
        test "reads after the work on its array" test_wait_copy;
      ];
    group "borrow"
      [
        prop "shares memory and keeps the layout" case law_borrow;
        test "is the array itself on its own device" test_borrow_own;
        test "is None where a device cannot map the memory" test_borrow_unmapped;
        test "shows writes through either array" test_borrow_writes;
        test "refuses a dead buffer" test_borrow_dead;
        test "raises Lost for a lost device" test_borrow_lost;
      ];
    group "bigarray"
      [
        test "shares a contiguous host array's bytes, never taken after"
          test_bigarray;
        test "waits for the work on its array's memory" test_wait_bigarray;
        prop ~count:500 "views a C-contiguous host array and raises otherwise"
          case law_bigarray;
        test "refuses an array off Rig.host" test_bigarray_off_rig_host;
        test "takes up to 16 axes" test_bigarray_rank;
        test "refuses a dead buffer off the host" test_dead_bigarray;
        test "raises Lost for memory a lost device's work must precede"
          test_bigarray_lost;
      ];
    group "read"
      [
        prop ~count:300 "passes a C-contiguous host-addressed array's bytes"
          case law_read;
        test "claims the memory while its function runs" test_read_claim;
        test "refuses what it cannot pass" test_read_refuses;
        test "passes an array with no element, whatever its memory"
          test_read_empty;
        test "waits for the work on its array" test_wait_read;
        test "raises Lost before its function runs" test_read_lost;
      ];
    group "write"
      [
        prop ~count:300 "passes a C-contiguous host array's bytes to store into"
          case law_write;
        test "claims the memory while its function runs" test_write_claim;
        test "refuses memory that admits only reads" test_write_read_only;
        test "passes an array with no element, whatever its memory"
          test_write_empty;
        test "raises Lost before its function runs" test_write_lost;
      ];
    group "of_bigarray"
      [
        test "shares the bigarray's bytes" test_of_bigarray;
        test "refuses a bigarray misaligned for its elements"
          test_of_bigarray_misaligned;
      ];
  ]

let () = exit (run "nx_array" tests)
