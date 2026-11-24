(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout

type pad = { pad : Nx_kernel.pad; fill : string }

type operand = {
  dtype : D.any;
  layout : L.t;
  buffer : int;
  skew : int;
  residue : int;
  pad : pad option;
}

type t = {
  dsts : operand array;
  loads : operand array;
  carry : operand option;
  overlaps : bool array;
}

(* The address of [b]'s first byte modulo 32, a bit at a time: where it is [r]
   modulo [n], it is [r] modulo [2 n] if [b]'s byte [2 n - r] lies on [2 n], and
   [r + n] otherwise. *)
let rec residue_from b r n =
  if n = 32 then r
  else
    let on = Rig.Buffer.aligned b ~at:((2 * n) - r) (2 * n) in
    residue_from b (if on then r else r + n) (2 * n)

let residue b = residue_from b (if Rig.Buffer.aligned b ~at:0 2 then 0 else 1) 2

let operand (type v s) (x : (v, s) A.t) buffer pad =
  let b = A.buffer x in
  {
    dtype = D.Any (A.dtype x);
    layout = A.layout x;
    buffer;
    skew = Rig.Buffer.offset b mod 4;
    residue = residue b;
    pad;
  }

(* The value [v]'s arrays hold before each operand is set. *)
let placeholder =
  {
    dtype = D.Any D.Uint8;
    layout = L.v ~strides:(Iarray.of_list []) (Iarray.of_list []);
    buffer = 0;
    skew = 0;
    residue = 0;
    pad = None;
  }

(* The ints of an operand: its dtype's code, buffer, skew, residue, offset and
   rank, its extents and strides, then 0, or 1 and its pad's lows, highs and
   interiors, its count of windows and each one's axis, size, step and dilation,
   and its fill's length and bytes. *)
let pad_ints (p : pad) =
  let r = Iarray.length p.pad.lo and w = Iarray.length p.pad.windows in
  (3 * r) + 1 + (4 * w) + 1 + String.length p.fill

let ints o =
  let r = L.rank o.layout in
  6 + (2 * r) + 1 + match o.pad with None -> 0 | Some p -> pad_ints p

let put_pad k at (p : pad) =
  let at = ref at in
  let put v =
    k.(!at) <- v;
    incr at
  in
  Iarray.iter put p.pad.lo;
  Iarray.iter put p.pad.hi;
  Iarray.iter put p.pad.interior;
  put (Iarray.length p.pad.windows);
  Iarray.iter
    (fun (w : A.Move.window) ->
      put w.axis;
      put w.size;
      put w.step;
      put w.dilation)
    p.pad.windows;
  put (String.length p.fill);
  String.iter (fun c -> put (Char.code c)) p.fill;
  !at

let put k at o =
  let l = o.layout in
  let r = L.rank l in
  let (D.Any dt) = o.dtype in
  k.(at) <- D.code dt;
  k.(at + 1) <- o.buffer;
  k.(at + 2) <- o.skew;
  k.(at + 3) <- o.residue;
  k.(at + 4) <- L.offset l;
  k.(at + 5) <- r;
  for a = 0 to r - 1 do
    k.(at + 6 + (2 * a)) <- L.dim l a;
    k.(at + 7 + (2 * a)) <- L.stride l a
  done;
  let at = at + 6 + (2 * r) in
  match o.pad with
  | None ->
      k.(at) <- 0;
      at + 1
  | Some p ->
      k.(at) <- 1;
      put_pad k (at + 1) p

(* The key: the counts of destinations and loads and whether a carry follows,
   each operand's ints, then the overlaps as 0 or 1. *)
let key t =
  let { dsts; loads; carry; overlaps } = t in
  let n = ref (3 + Array.length overlaps) in
  for i = 0 to Array.length dsts - 1 do
    n := !n + ints dsts.(i)
  done;
  for j = 0 to Array.length loads - 1 do
    n := !n + ints loads.(j)
  done;
  (match carry with None -> () | Some o -> n := !n + ints o);
  let k = Array.make !n 0 in
  k.(0) <- Array.length dsts;
  k.(1) <- Array.length loads;
  k.(2) <- Bool.to_int (Option.is_some carry);
  let at = ref 3 in
  for i = 0 to Array.length dsts - 1 do
    at := put k !at dsts.(i)
  done;
  for j = 0 to Array.length loads - 1 do
    at := put k !at loads.(j)
  done;
  (match carry with None -> () | Some o -> at := put k !at o);
  for i = 0 to Array.length overlaps - 1 do
    k.(!at + i) <- Bool.to_int overlaps.(i)
  done;
  k

let hash k =
  let h = ref 0 in
  for i = 0 to Array.length k - 1 do
    h := ((!h * 31) + k.(i)) land max_int
  done;
  !h

(* The facts are made at every call, so these functions take the call's parts as
   arguments and build no closure. *)

let load_buffer (l : Nx_kernel.load) =
  match l with Plain x -> A.buffer x | Padded { x; _ } -> A.buffer x

(* Operand [i]'s buffer: destination [i], load [i - n] or the carry. *)
let buffer_at dsts loads carry i =
  let n = Iarray.length dsts and m = Iarray.length loads in
  if i < n then
    let (A.Any y) = Iarray.get dsts i in
    A.buffer y
  else if i < n + m then load_buffer (Iarray.get loads (i - n))
  else
    match carry with
    | Some (A.Any c) -> A.buffer c
    | None -> invalid_arg "Operands: no carry"

(* The first operand of [i]'s buffer. *)
let first dsts loads carry i =
  let b = buffer_at dsts loads carry i in
  let j = ref 0 in
  while buffer_at dsts loads carry !j != b do
    incr j
  done;
  !j

let dst_operand dsts loads carry i =
  let (A.Any y) = Iarray.get dsts i in
  operand y (first dsts loads carry i) None

let load_operand dsts loads carry j =
  let at = first dsts loads carry (Iarray.length dsts + j) in
  match Iarray.get loads j with
  | Nx_kernel.Plain x -> operand x at None
  | Padded { x; fill; pad } ->
      let fill = Nx_kernel.Prog.bits (A.dtype x) fill in
      operand x at (Some { pad; fill })

let v ?carry ~(dsts : A.any iarray) (loads : Nx_kernel.load iarray) =
  let n = Iarray.length dsts and m = Iarray.length loads in
  let ds = Array.make n placeholder and ls = Array.make m placeholder in
  for i = 0 to n - 1 do
    ds.(i) <- dst_operand dsts loads carry i
  done;
  for j = 0 to m - 1 do
    ls.(j) <- load_operand dsts loads carry j
  done;
  let carry =
    match carry with
    | None -> None
    | Some (A.Any c) -> Some (operand c (first dsts loads carry (n + m)) None)
  in
  let overlaps = Array.make (m * n) false in
  for j = 0 to m - 1 do
    let b = load_buffer (Iarray.get loads j) in
    for k = 0 to n - 1 do
      let (A.Any y) = Iarray.get dsts k in
      overlaps.((j * n) + k) <- Rig.Buffer.overlaps b (A.buffer y)
    done
  done;
  { dsts = ds; loads = ls; carry; overlaps }

let overlaps t ~load ~dst = t.overlaps.((load * Array.length t.dsts) + dst)

let one i o =
  let r = L.rank o.layout in
  let layout =
    L.v ~strides:(Iarray.init r (fun _ -> 0)) (Iarray.init r (fun _ -> 1))
  in
  { o with layout; buffer = i; skew = 0; residue = 0; pad = None }

let ones t =
  let n = Array.length t.dsts and m = Array.length t.loads in
  let dsts = Array.mapi one t.dsts in
  let loads = Array.mapi (fun j o -> one (n + j) o) t.loads in
  let carry = Option.map (one (n + m)) t.carry in
  { dsts; loads; carry; overlaps = Array.make (n * m) false }
