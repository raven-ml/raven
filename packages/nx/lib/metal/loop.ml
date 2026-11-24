(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A loop's launches of loop.metal: its walk, its operands' descriptors and its
   program lowered to a tape, written as each launch's table, and the launches
   that keep each within the time budget.

   program, loads --> walk (axes ordered, merged) --> descriptors program -->
   tape over planes --> table, held or a constant units --> launches within
   Plan.allowed --> Plan.t, one submission

   A walk orders the loop's axes as the kernel walks them: a reduction's kept
   axes then its reduced ones, a scan's other axes then its axis, each in its
   order, so that the terms keep their C order. Where no load is padded and no
   node reads a coordinate, it drops axes of one element and merges an axis into
   the one before where every operand steps along both as one.

   Each kind computes through its function in nx_kinds.h, which nx.cpu's kinds
   call too, so a kind gives nx.cpu's value at every dtype, but at float32 and
   bfloat16 where an operand is subnormal or a result below 2^-126
   (nx_metal.mli). float64 and complex128 values only move. *)

module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module P = Nx_kernel.Prog
module K = Kernels

(* Kinds *)

(* A kind nx.metal does not compute: arithmetic at float64 or complex128. *)
exception Declined

let unary_index : P.unary -> int = function
  | Neg -> 0
  | Recip -> 1
  | Abs -> 2
  | Sign -> 3
  | Sqrt -> 4
  | Exp -> 5
  | Exp2 -> 6
  | Log -> 7
  | Log2 -> 8
  | Log1p -> 9
  | Expm1 -> 10
  | Sin -> 11
  | Cos -> 12
  | Tan -> 13
  | Asin -> 14
  | Acos -> 15
  | Atan -> 16
  | Sinh -> 17
  | Cosh -> 18
  | Tanh -> 19
  | Asinh -> 20
  | Acosh -> 21
  | Atanh -> 22
  | Erf -> 23
  | Floor -> 24
  | Ceil -> 25
  | Round -> 26
  | Trunc -> 27

let binary_index : P.binary -> int = function
  | Add -> 0
  | Sub -> 1
  | Mul -> 2
  | Fdiv -> 3
  | Idiv -> 4
  | Mod -> 5
  | Pow -> 6
  | Atan2 -> 7
  | Maximum -> 8
  | Minimum -> 9
  | And -> 10
  | Or -> 11
  | Xor -> 12
  | Threefry -> 13

let compare_index : P.compare -> int = function
  | Equal -> 0
  | Not_equal -> 1
  | Less -> 2
  | Less_equal -> 3

(* The code of the kind of [node] as the tape's ops read it (kernels.h): an
   Op1's, NX_LOOP_BINARY plus an Op2's, NX_LOOP_TERNARY plus an Op3's, each
   nx_kernel.h's code, in Prog's order of its constructors. *)
let kind_code (node : P.node) =
  match node with
  | Op1 (Unary u, _, _) -> K.loop_unary + unary_index u
  | Op2 (Binary b, _, _) -> K.loop_binary + binary_index b
  | Op2 (Compare c, _, _) -> K.loop_binary + K.loop_compare + compare_index c
  | Op3 (Fma, _, _, _) -> K.loop_ternary + 1
  | Op3 (Where, _, _, _) -> K.loop_ternary
  | Op1 ((Copy | Cast | Bitcast), _, _) | In _ | Coord _ | Const _ -> -1

(* Dtypes *)

let code (D.Any d) = D.code d
let int64 = D.Any D.Int64
let float32 = D.Any D.Float32
let bits (D.Any d) = D.bits d

let words (D.Any d) =
  let b = D.bits d in
  if b <= 32 then 1 else b / 32

let narrow_float = function
  | D.Any
      ( D.Float16 | D.Bfloat16 | D.Float8_e4m3fn | D.Float8_e5m2
      | D.Float4_e2m1fn ) ->
      true
  | D.Any _ -> false

let narrow_int = function
  | D.Any (D.Int16 | D.Uint16 | D.Int8 | D.Uint8 | D.Int4 | D.Uint4) -> true
  | D.Any _ -> false

let boolean = function D.Any (D.Bool | D.Bit) -> true | D.Any _ -> false

(* Whether [dt]'s values only move here. *)
let moves_only = function
  | D.Any (D.Float64 | D.Complex128) -> true
  | D.Any _ -> false

(* The type a kind of [dt] computes in (Prog's compute types). *)
let compute = function
  | D.Any (D.Float64 | D.Complex128) -> raise Declined
  | D.Any D.Complex64 -> K.loop_c64
  | D.Any D.Int64 -> K.loop_i64
  | D.Any D.Uint64 -> K.loop_u64
  | D.Any (D.Int32 | D.Int16 | D.Int8 | D.Int4) -> K.loop_i32
  | D.Any (D.Uint32 | D.Uint16 | D.Uint8 | D.Uint4 | D.Bool | D.Bit) ->
      K.loop_u32
  | D.Any _ -> K.loop_f32

(* The held value of [dt] whose bits are the low ones of [r] (tape.h's of_raw):
   a signed integer of 16 bits or fewer sign-extended to 32. *)
let held dt r =
  let extend bits =
    let x = Int64.to_int r land ((1 lsl bits) - 1) in
    let x = if x land (1 lsl (bits - 1)) <> 0 then x - (1 lsl bits) else x in
    Int64.of_int (x land 0xffffffff)
  in
  match dt with
  | D.Any D.Int16 -> extend 16
  | D.Any D.Int8 -> extend 8
  | D.Any D.Int4 -> extend 4
  | D.Any _ -> r

(* Words *)

(* Words as they grow: [n] little-endian words of [b]. *)
type words = { mutable b : Bytes.t; mutable n : int }

let words_for n = { b = Bytes.create (8 * n); n = 0 }

(* Inlined, so that a word read and taken apart in one function is never
   boxed. *)
let[@inline] get ws i = Bytes.get_int64_le ws.b (8 * i)

(* Inlined, so that a word computed at the call is never boxed. *)
let[@inline] push ws x =
  if 8 * ws.n = Bytes.length ws.b then begin
    let b = Bytes.create (2 * Bytes.length ws.b) in
    Bytes.blit ws.b 0 b 0 (8 * ws.n);
    ws.b <- b
  end;
  Bytes.set_int64_le ws.b (8 * ws.n) x;
  ws.n <- ws.n + 1

let push_int ws x = push ws (Int64.of_int x)

(* Planes *)

(* The most planes an instruction names: 12 bits. *)
let max_planes = 4096

exception Too_wide

(* The planes in use, a byte each; those past [used]'s length are free. The map
   starts at [first_planes] and grows as a tape takes more, so that a plan of a
   few values does not allocate and clear [max_planes] bytes. *)
type planes = { mutable used : Bytes.t; mutable top : int }

let first_planes = 64
let planes () = { used = Bytes.make first_planes '\000'; top = 0 }
let taken pl p = p < Bytes.length pl.used && Bytes.get pl.used p <> '\000'

let rec free_run pl p n k =
  k = n || ((not (taken pl (p + k))) && free_run pl p n (k + 1))

(* [pl]'s map grown to hold planes below [n]. *)
let reach pl n =
  let have = Bytes.length pl.used in
  if n > have then begin
    let used = Bytes.make (Int.min max_planes (Int.max n (2 * have))) '\000' in
    Bytes.blit pl.used 0 used 0 have;
    pl.used <- used
  end

(* The first [n] free planes in a row, now taken. *)
let alloc pl n =
  let rec find p =
    if p + n > max_planes then raise Too_wide
    else if free_run pl p n 0 then p
    else find (p + 1)
  in
  let p = find 0 in
  reach pl (p + n);
  Bytes.fill pl.used p n '\001';
  pl.top <- Int.max pl.top (p + n);
  p

let release pl p n = Bytes.fill pl.used p n '\000'

(* Tapes *)

type tape = { pro : words; body : words; epi : words; pl : planes }

let tape () =
  let w () = words_for 8 in
  { pro = w (); body = w (); epi = w (); pl = planes () }

let ins ws op ~d ~a ~b ~c =
  let f x s = Int64.shift_left (Int64.of_int x) s in
  push ws
    Int64.(
      logor (of_int op)
        (logor (f d 16) (logor (f a 28) (logor (f b 40) (f c 52)))))

let kind ws ct k ~d ~a ~b ~c = ins ws ((ct lsl 7) lor k) ~d ~a ~b ~c

(* Plane [a] of [from] cast to [into] in plane [d]: between float32 and a narrow
   float, which rounds once from the float32 or widens exactly, the encode or
   the decode the runner runs in line; else the cast. A cast from or into a
   dtype whose values only move is declined, but into its own dtype. *)
let cast ws ~d ~a ~from ~into =
  if from <> into && (moves_only from || moves_only into) then raise Declined;
  if from = float32 && narrow_float into then
    ins ws K.loop_encode ~d ~a ~b:(code into) ~c:0
  else if narrow_float from && into = float32 then
    ins ws K.loop_decode ~d ~a ~b:(code from) ~c:0
  else ins ws K.loop_cast ~d ~a ~b:(code from) ~c:(code into)

(* A value of the tape: its first plane and its dtype. *)
type value = { plane : int; dt : D.any }

let operands_of : P.node -> int list = function
  | P.In _ | Coord _ | Const _ -> []
  | Op1 (_, _, a) -> [ a ]
  | Op2 (_, a, b) -> [ a; b ]
  | Op3 (_, a, b, c) -> [ a; b; c ]

(* Lowers the nodes of [p] that the outputs [roots] reach into [t]'s prologue,
   constants, and body: first a load of each operand [k] read where [eager k],
   then the nodes in order, each other operand loaded at its first reader or its
   first output, so that its planes are taken no sooner. [load k] is operand
   [k]'s load in the launch's table, [axis a] the kernel's axis of the loop's
   axis [a]. After node [i] computes, [out k v] is called for each output [k]
   among [roots] that is node [i], with its value. A value's planes are freed
   after the last node that reads it, a constant's never. Raises [Declined] at
   arithmetic on a dtype whose values only move. *)
let lower t (p : P.t) ~roots ~load ~axis ~out ~eager =
  let n = Iarray.length p.nodes in
  let node i = Iarray.get p.nodes i in
  (* Each node's dtype, in one pass: [P.dtype] follows a Binary kind or an Op3
     back to an operand's, so asking it of each node of a chain takes time
     quadratic in the chain's length. *)
  let types = Array.make n int64 in
  for i = 0 to n - 1 do
    types.(i) <-
      (match node i with
      | Op2 (Binary _, a, _) -> types.(a)
      | Op3 (_, _, b, _) -> types.(b)
      | In _ | Coord _ | Const _ | Op1 _ | Op2 (Compare _, _, _) -> P.dtype p i)
  done;
  let dtype i = types.(i) in
  let operands i = operands_of (node i) in
  (* Live nodes, from the outputs back. *)
  let live = Array.make n false in
  List.iter (fun k -> live.(Iarray.get p.outs k) <- true) roots;
  for i = n - 1 downto 0 do
    if live.(i) then List.iter (fun a -> live.(a) <- true) (operands i)
  done;
  (* The node whose value each is: an operand's for one that moves no bit, the
     first read of an operand for a later one. *)
  let root = Array.init n Fun.id in
  let first_in = Hashtbl.create 8 in
  for i = 0 to n - 1 do
    if live.(i) then
      match node i with
      | P.Op1 (Copy, _, a) -> root.(i) <- root.(a)
      | Op1 ((Cast | Bitcast), y, a) when dtype a = y -> root.(i) <- root.(a)
      | Op1 (Unary (Floor | Ceil | Round | Trunc), dt, a)
        when (not (moves_only dt)) && compute (dtype a) >= K.loop_i32 ->
          root.(i) <- root.(a)
      | In k -> (
          match Hashtbl.find_opt first_in k with
          | Some j -> root.(i) <- j
          | None -> Hashtbl.add first_in k i)
      | _ -> ()
  done;
  (* The last node reading each value; an output's own node reads it. *)
  let last = Array.make n (-1) in
  for i = 0 to n - 1 do
    if live.(i) then
      List.iter
        (fun a -> last.(root.(a)) <- Int.max last.(root.(a)) i)
        (operands i)
  done;
  List.iter
    (fun k ->
      let o = Iarray.get p.outs k in
      last.(root.(o)) <- Int.max last.(root.(o)) o)
    roots;
  let values = Array.make n { plane = -1; dt = int64 } in
  let loaded = Array.make n false in
  let fresh i =
    let dt = dtype i in
    let v = { plane = alloc t.pl (words dt); dt } in
    values.(i) <- v;
    v
  in
  (* Constants first, in the prologue: their planes, written once, are no other
     value's. Then the loads: a load a destination is identical to is read
     before any store. *)
  for i = 0 to n - 1 do
    match node i with
    | P.Const (dt, bits) when live.(i) ->
        let v = fresh i in
        let b = Bytes.make 16 '\000' in
        Bytes.blit_string bits 0 b 0 (String.length bits);
        ins t.pro K.loop_const ~d:v.plane ~a:0 ~b:(words dt) ~c:0;
        push t.pro (held dt (Bytes.get_int64_le b 0));
        push t.pro (Bytes.get_int64_le b 8)
    | _ -> ()
  done;
  let load_in i =
    match node i with
    | P.In k when not loaded.(i) ->
        loaded.(i) <- true;
        let v = fresh i in
        ins t.body K.loop_load ~d:v.plane ~a:(load k) ~b:0 ~c:0
    | _ -> ()
  in
  for i = 0 to n - 1 do
    match node i with
    | P.In k when live.(i) && root.(i) = i && eager k -> load_in i
    | _ -> ()
  done;
  let temp () = alloc t.pl 1 in
  let free_temp q = release t.pl q 1 in
  for i = 0 to n - 1 do
    if live.(i) && root.(i) = i then begin
      (* Operands not yet loaded, loaded and waited for. *)
      let unloaded r =
        match node r with P.In _ -> not loaded.(r) | _ -> false
      in
      let late =
        List.sort_uniq Int.compare
          (List.filter unloaded (List.map (fun a -> root.(a)) (operands i)))
      in
      List.iter load_in late;
      let arg a = values.(root.(a)).plane in
      match node i with
      | P.In _ | Const _ -> ()
      | Coord a ->
          let v = fresh i in
          ins t.body K.loop_coord ~d:v.plane ~a:(axis a) ~b:0 ~c:0
      | Op1 (Cast, y, a) ->
          let v = fresh i in
          cast t.body ~d:v.plane ~a:(arg a) ~from:(dtype a) ~into:y
      | Op1 (Bitcast, y, a) ->
          let v = fresh i in
          ins t.body K.loop_bitcast ~d:v.plane ~a:(arg a)
            ~b:(code (dtype a))
            ~c:(code y)
      | Op1 (Copy, _, _) -> ()
      | Op1 (Unary _, dt, a) ->
          let ct = compute dt in
          let v = fresh i and k = kind_code (node i) in
          if narrow_float dt then begin
            let c = code dt in
            ins t.body K.loop_decode ~d:v.plane ~a:(arg a) ~b:c ~c:0;
            kind t.body K.loop_f32 k ~d:v.plane ~a:v.plane ~b:0 ~c:0;
            ins t.body K.loop_encode ~d:v.plane ~a:v.plane ~b:c ~c:0
          end
          else begin
            kind t.body ct k ~d:v.plane ~a:(arg a) ~b:0 ~c:0;
            if narrow_int dt then
              ins t.body K.loop_wrap ~d:v.plane ~a:v.plane ~b:(code dt) ~c:0
          end
      | Op2 (op, a, b) ->
          let dt = dtype a in
          let ct = compute dt in
          let v = fresh i and k = kind_code (node i) in
          let binary =
            match op with P.Binary _ -> true | Compare _ -> false
          in
          if narrow_float dt then begin
            let c = code dt and q = temp () in
            ins t.body K.loop_decode ~d:v.plane ~a:(arg a) ~b:c ~c:0;
            ins t.body K.loop_decode ~d:q ~a:(arg b) ~b:c ~c:0;
            kind t.body K.loop_f32 k ~d:v.plane ~a:v.plane ~b:q ~c:0;
            free_temp q;
            if binary then
              ins t.body K.loop_encode ~d:v.plane ~a:v.plane ~b:c ~c:0
          end
          else if boolean dt then begin
            let q = temp () in
            ins t.body K.loop_truth ~d:v.plane ~a:(arg a) ~b:0 ~c:0;
            ins t.body K.loop_truth ~d:q ~a:(arg b) ~b:0 ~c:0;
            kind t.body K.loop_u32 k ~d:v.plane ~a:v.plane ~b:q ~c:0;
            free_temp q
          end
          else begin
            kind t.body ct k ~d:v.plane ~a:(arg a) ~b:(arg b) ~c:0;
            if binary && narrow_int dt then
              ins t.body K.loop_wrap ~d:v.plane ~a:v.plane ~b:(code dt) ~c:0
          end
      | Op3 (Where, c, a, b) ->
          let v = fresh i in
          let op =
            match words v.dt with
            | 1 -> K.loop_where1
            | 2 -> K.loop_where2
            | _ -> K.loop_where4
          in
          ins t.body op ~d:v.plane ~a:(arg c) ~b:(arg a) ~c:(arg b)
      | Op3 (Fma, a, b, c) ->
          let dt = dtype a in
          let ct = compute dt in
          let v = fresh i and k = kind_code (node i) in
          if narrow_float dt then begin
            let cd = code dt and q = temp () and r = temp () in
            ins t.body K.loop_decode ~d:v.plane ~a:(arg a) ~b:cd ~c:0;
            ins t.body K.loop_decode ~d:q ~a:(arg b) ~b:cd ~c:0;
            ins t.body K.loop_decode ~d:r ~a:(arg c) ~b:cd ~c:0;
            (* The exact value rounded to odd, which the encode rounds once. *)
            let fma =
              match dt with
              | D.Any D.Bfloat16 -> K.loop_fma_bf16
              | _ -> K.loop_fma_narrow
            in
            kind t.body K.loop_f32 fma ~d:v.plane ~a:v.plane ~b:q ~c:r;
            free_temp q;
            free_temp r;
            ins t.body K.loop_encode ~d:v.plane ~a:v.plane ~b:cd ~c:0
          end
          else begin
            kind t.body ct k ~d:v.plane ~a:(arg a) ~b:(arg b) ~c:(arg c);
            if narrow_int dt then
              ins t.body K.loop_wrap ~d:v.plane ~a:v.plane ~b:(code dt) ~c:0
          end
    end;
    if live.(i) then begin
      List.iter
        (fun k ->
          if Iarray.get p.outs k = i then begin
            load_in root.(i);
            out k values.(root.(i))
          end)
        roots;
      (* Values read last here go, a constant's planes but never. *)
      let dead r =
        if last.(r) = i then
          match node r with
          | P.Const _ -> ()
          | _ -> release t.pl values.(r).plane (words values.(r).dt)
      in
      List.iter (fun a -> dead root.(a)) (operands i);
      dead root.(i)
    end
  done

(* Walks *)

(* An operand as the walk reads it, over the loop's axes in the caller's order:
   a plain one by its strides; a padded one by, per axis of its array, the loop
   axes that give its padded coordinate and their coefficients. *)
type axis_read = {
  lo : int;
  interior : int;
  extent : int;
  stride : int;
  a1 : int;
  c1 : int;
  a2 : int; (* -1 for none *)
  c2 : int;
  reach : int; (* the largest magnitude of its padded coordinate *)
}

type access =
  | Strided of int array
  | Pad of { fill : string; reads : axis_read array; steps : int array }

(* Where an operand's elements lie: a call's operand's buffer, by the number of
   the first operand of it, its skew and its residue ({!Operands.operand}), a
   buffer of the call's temporaries, by its number ({!stand_ins}), its pass's
   piece of the workspace, or a constant of {!zero_bytes} zero bytes. *)
type buf =
  | Arr of { first : int; skew : int; residue : int }
  | Temp of int
  | Space
  | Zero

let same_buf a b =
  match (a, b) with
  | Arr x, Arr y -> x.first = y.first
  | Temp i, Temp j -> i = j
  | Space, Space | Zero, Zero -> true
  | (Arr _ | Temp _ | Space | Zero), _ -> false

(* The zero constant's bytes: an element of any dtype the loops take. *)
let zero_bytes = 16

(* An operand: its buffer, its dtype, the position of its element at index 0 and
   the positions its elements lie in, [lo] to [hi] - 1. *)
type operand = {
  buffer : buf;
  dtype : D.any;
  offset : int;
  lo : int;
  hi : int;
  access : access;
}

(* The bit of [o]'s element at its low bound, from the start of the 4-byte word
   that holds its buffer's first byte, for an operand narrower than a byte: the
   kernel reads and writes it in whole words, which start at multiples of 4 from
   its allocation's first byte (rig.mli's Words). An array may start inside a
   word; temporaries and the workspace start on one. *)
let low_bit o b =
  let skew =
    match o.buffer with Arr x -> x.skew | Temp _ | Space | Zero -> 0
  in
  ((8 * skew) + (o.lo * b), skew)

let arr (o : Operands.operand) =
  Arr { first = o.buffer; skew = o.skew; residue = o.residue }

(* Whether [o]'s element at index 0 lies on [n] bytes, a power of two at most
   16, for [o] of whole bytes: by its buffer's residue, and for temporaries and
   the workspace by the offset alone, since their pieces start on 256 bytes
   ({!Plan.take}) of a buffer of the device's. *)
let first_on (o : operand) n =
  let residue =
    match o.buffer with Arr x -> x.residue | Temp _ | Space | Zero -> 0
  in
  (residue + (o.offset * bits o.dtype / 8)) land (n - 1) = 0

let plain (o : Operands.operand) =
  let l = o.layout in
  let lo, hi = L.span l in
  {
    buffer = arr o;
    dtype = o.dtype;
    offset = L.offset l;
    lo;
    hi;
    access = Strided (Iarray.to_array (L.strides l));
  }

let padded (o : Operands.operand) ({ pad; fill } : Operands.pad) =
  let l = o.layout in
  let r = L.rank l in
  let windows = Iarray.to_array pad.windows in
  let window a =
    let rec find k =
      if k = Array.length windows then None
      else if windows.(k).Nx_array.Move.axis = a then Some (k, windows.(k))
      else find (k + 1)
    in
    find 0
  in
  let reads =
    Array.init r (fun a ->
        let lo = Iarray.get pad.lo a and interior = Iarray.get pad.interior a in
        let extent = L.dim l a and stride = L.stride l a in
        (* The padded coordinate runs from -lo to the padded extent's last index
           less lo; a window's taps stay within the padded extent. *)
        let last =
          extent + (Int.max 0 (extent - 1) * interior) + Iarray.get pad.hi a - 1
        in
        let reach = Int.max (abs lo) (abs last) in
        match window a with
        | None ->
            {
              lo;
              interior;
              extent;
              stride;
              a1 = a;
              c1 = 1;
              a2 = -1;
              c2 = 0;
              reach;
            }
        | Some (k, w) ->
            {
              lo;
              interior;
              extent;
              stride;
              a1 = a;
              c1 = w.step;
              a2 = r + k;
              c2 = w.dilation;
              reach;
            })
  in
  let lo, hi = L.span l in
  {
    buffer = arr o;
    dtype = o.dtype;
    offset = L.offset l;
    lo;
    hi;
    access = Pad { fill; reads; steps = [||] };
  }

let of_load (o : Operands.operand) =
  match o.pad with None -> plain o | Some p -> padded o p

(* A padded operand's step along each kernel axis of an unmerged walk of axes
   [perm], in positions: its reads' coefficients times their strides. An
   interior's step is the most an index moves. *)
let pad_steps (reads : axis_read array) perm =
  Array.map
    (fun a ->
      Array.fold_left
        (fun s (r : axis_read) ->
          s
          + (if r.a1 = a then r.c1 * r.stride else 0)
          + if r.a2 = a then r.c2 * r.stride else 0)
        0 reads)
    perm

(* A walk: the kernel's extents, and the loop's axis each kernel axis merged
   last; the kept axes' count. *)
type walk = { ext : int array; perm : int array; kept : int }

(* The walk of a loop of extents [shape] whose kernel order is [order], the
   first [kept] kept, over [ops], merging where [merge]. Each strided operand's
   strides come back in the kernel's order, and, where [steps], each padded
   one's steps ({!pad_steps}), which a map's bytes are charged by; a reduction's
   and a scan's padded loads are charged a line an element. *)
let walk ?(steps = false) shape order ~kept ~merge (ops : operand array) =
  let strides (o : operand) =
    match o.access with Strided s -> s | Pad _ -> [||]
  in
  let ext = ref [] and perm = ref [] and st = Array.map (fun _ -> ref []) ops in
  let n = ref 0 and nk = ref 0 and start = ref 0 in
  let fits a e =
    let ok = ref true in
    Array.iteri
      (fun k o ->
        match !(st.(k)) with
        | prev :: _ -> if prev <> (strides o).(a) * e then ok := false
        | [] -> ())
      ops;
    !ok
  in
  Array.iteri
    (fun j a ->
      if j = kept then start := !n;
      let e = shape.(a) in
      if merge && e = 1 then ()
      else if merge && !n > !start && fits a e then begin
        ext := (List.hd !ext * e) :: List.tl !ext;
        perm := a :: List.tl !perm;
        Array.iteri
          (fun k o -> st.(k) := (strides o).(a) :: List.tl !(st.(k)))
          ops
      end
      else begin
        ext := e :: !ext;
        perm := a :: !perm;
        incr n;
        Array.iteri
          (fun k o ->
            match o.access with
            | Strided s -> st.(k) := s.(a) :: !(st.(k))
            | Pad _ -> ())
          ops
      end;
      if j < kept then nk := !n)
    order;
  let w =
    {
      ext = Array.of_list (List.rev !ext);
      perm = Array.of_list (List.rev !perm);
      kept = !nk;
    }
  in
  let ops =
    Array.mapi
      (fun k (o : operand) ->
        match o.access with
        | Strided _ ->
            { o with access = Strided (Array.of_list (List.rev !(st.(k)))) }
        | Pad p when steps ->
            { o with access = Pad { p with steps = pad_steps p.reads w.perm } }
        | Pad _ -> o)
      ops
  in
  (w, ops)

(* The walk of one axis of extent 1, of a loop of [shape] whose merged walk kept
   none, over [ops]: each strided operand at stride 1 there, since its one
   element is at index 0 whatever its strides, so that a map takes quads
   ({!quads}). The axis is the loop's last, or 0 of a loop of no axis; no
   coordinate or padded load reads it, since a merged walk has none. *)
let one_axis shape (ops : operand array) =
  let last = Int.max 0 (Array.length shape - 1) in
  let w = { ext = [| 1 |]; perm = [| last |]; kept = 0 } in
  let one (o : operand) =
    match o.access with
    | Strided _ -> { o with access = Strided [| 1 |] }
    | Pad _ -> o
  in
  (w, Array.map one ops)

(* The kernel axis of the loop's axis [a], with no axis merged. *)
let kernel_axis w a =
  let rec find i = if w.perm.(i) = a then i else find (i + 1) in
  find 0

let product a i j =
  let r = ref 1 in
  for k = i to j - 1 do
    r := !r * a.(k)
  done;
  !r

(* Whether a node of [p] reads a coordinate. *)
let reads_coord (p : P.t) =
  Iarray.exists (function P.Coord _ -> true | _ -> false) p.nodes

let is_padded (o : operand) = match o.access with Pad _ -> true | _ -> false

(* Whether one of [os] from [k] is padded, allocating nothing, as Array.exists's
   closure would. *)
let rec any_padded (os : operand array) k =
  k < Array.length os && (is_padded os.(k) || any_padded os (k + 1))

(* Whether [os] from [k] are each of 32 bits. *)
let rec all_bits n (os : operand array) k =
  k >= Array.length os
  || (let (D.Any d) = os.(k).dtype in
      D.bits d = n)
     && all_bits n os (k + 1)

(* Launches

   A launch's buffer fields each hold what rig writes there: an operand's buffer
   at its lowest element, the workspace of its pass or the tickets. A pass is
   launches that share a piece of workspace, in order. *)

type target = Buf of buf | Tickets

(* A buffer field: its target, the byte in it, whether the launch writes it. *)
type field = { target : target; at : int; write : bool }

type launch = {
  kernel : K.t;
  groups : int;
  shared : int;
  resident : int;
  cost : Plan.cost;
  header : Bytes.t;  (** The launch's words, its fields but the buffers. *)
  table : string option;  (** The table, where the tail holds it. *)
  fields : field array;
  tickets : int;
}

(* A copy of a pass: an output that moves the bits of [src], a strided operand
   of a byte or more, into [dst] over a loop of extents [shape]. *)
type copy = { src : operand; dst : operand; shape : int array }

(* A pass: its launches, the workspace they take, then its copies. *)
type pass = { launches : launch list; work : int; copies : copy list }

let ceil_div a b = (a + b - 1) / b
let align16 n = (n + 15) land lnot 15
let threads = K.loop_threads
let wide_bound = 1 lsl 32

(* Costs

   Counts in Plan's units, each taken high: a weight is a kind's work on an
   element in float32 operations at its slowest. A cheap kind's few operations;
   a costly one's polynomial and reductions; a complex transcendental's several
   of those. An instruction's dispatch over a thread's elements adds [dispatch]
   each. *)

let dispatch = 16
let cheap = 8
let costly = 512
let costly_complex = 4096
let wide_division = 1024

(* An instruction's links in a chain: a load's dependent read, a kind's
   operations at one link each up to [costly]. *)
let chain_of_weight w = Int.min w costly

(* The work of folding a term into a state, or merging two, in float32
   operations, and its links in a chain: an Arg's compare and select, a
   compensated step, at their slowest. *)
let fold_ops = 32
let fold_links = 16

(* The weight of an instruction [t] of a loop of [rank] axes: its work on one
   element. *)
let weight ~rank t =
  let op = Int64.to_int (Int64.logand t 0xffffL) in
  let place = 16 + (16 * rank) in
  if op < K.loop_load then
    let ct = op lsr 7 and k = op land 127 in
    let cheap_kind =
      (k >= K.loop_unary && k < K.loop_unary + 4)
      || (k >= K.loop_unary + 24 && k < K.loop_unary + 28)
      || k = K.loop_binary
      || k = K.loop_binary + 1
      || k = K.loop_binary + 2
      || (k >= K.loop_binary + 8 && k < K.loop_binary + 13)
      || (k >= K.loop_binary + K.loop_compare && k < K.loop_ternary)
      || k = K.loop_ternary + 1
    in
    if ct = K.loop_c64 then if cheap_kind then 4 * cheap else costly_complex
    else if cheap_kind && ct <> K.loop_f32 && k = K.loop_binary + 2 then
      4 * cheap
    else if cheap_kind then cheap
    else if ct = K.loop_i64 || ct = K.loop_u64 then wide_division
    else costly
  else if op = K.loop_load || op = K.loop_store then place
  else if op = K.loop_coord then place
  else if op = K.loop_const then 0
  else if op = K.loop_cast || op = K.loop_bitcast then 128
  else 2 * cheap

(* The work and the chain of [n] words of [ws] from [from] on one element, each
   word's [dispatch] included. *)
let tape_cost ~dispatch ~rank ws ~from n =
  let ops = ref 0 and chain = ref 0 in
  for k = from to from + n - 1 do
    let t = get ws k in
    let op = Int64.to_int (Int64.logand t 0xffffL) in
    let w = weight ~rank t + dispatch in
    ops := !ops + w;
    chain :=
      !chain + if op = K.loop_load then Plan.read_links else chain_of_weight w
  done;
  (!ops, !chain)

(* The bytes [e] elements a tile moves of an operand of [width]-byte elements at
   strides [s] over the innermost [rank] axes of a walk of extents [ext], by its
   runs. A run starts at the innermost axis and takes each axis further out
   while the lines it touches are touched again soon, so that each is charged
   once: an axis of stride 0 while the run spans at most a line; one whose steps
   stride past the run while they leave a gap of at most a line; and one whose
   steps land within the run, as overlapping windows do, while the run spans at
   most a line. A line touched again further away may have left the cache, and
   the next run is charged it again: a column broadcast along rows of 2 is
   charged a line a row, a row broadcast along columns its lines a row, a
   transpose's read a line an element.

   A run of [len] elements, [d] of them distinct, over [span] bytes touches at
   most d lines, and at most ceil (span / line) + 1 at any base, each charged
   whole, so no plan depends on where an operand lies: a run of 2 float32s in a
   row of 64 moves one or two lines, not 8 bytes; 3 channels of 4, 2x2 blocks
   and windows a float apart are charged their union. A long run's last line is
   a share of its own: a contiguous run of 2^20 float32s is charged 0.02% past
   its lines, one of 1,024 3%, one of 32 2x. On runs of a line or less the bound
   errs high by up to 2x from the last line, and another 2x where the GPU moves
   64-byte sectors (the M1 Max does) of Plan.line's 128. *)
let run_bytes ~ext ~rank ~e ~width s =
  let len = ref 1 and distinct = ref 1 and span = ref 1 in
  let k = ref (rank - 1) and near = ref true in
  while !near && !k >= 0 do
    let n = ext.(!k) and st = abs s.(!k) in
    near :=
      n = 1
      ||
      if st = 0 || st < !span then !span * width <= Plan.line
      else (st - !span) * width <= Plan.line;
    if !near then begin
      len := !len * n;
      if st <> 0 then begin
        distinct := !distinct * n;
        span := !span + ((n - 1) * st)
      end;
      decr k
    end
  done;
  let lines = Int.min !distinct (ceil_div (!span * width) Plan.line + 1) in
  let run = lines * Plan.line in
  let q = run / !len and rem = run mod !len in
  (e * q) + ceil_div (e * rem) !len

(* The bytes [e] elements of [o] a tile moves, on a walk of extents [ext], as
   {!run_bytes} says: a padded one's by its steps, as though every element were
   read, or a line an element where it has none. *)
let traffic ~ext ~e (o : operand) =
  let width = Int.max 1 (bits o.dtype / 8) in
  match o.access with
  | Pad { steps = [||]; _ } -> e * Plan.line
  | Pad { steps; _ } ->
      run_bytes ~ext ~rank:(Array.length steps) ~e ~width steps
  | Strided s when Array.length s = 0 -> e * width
  | Strided s -> run_bytes ~ext ~rank:(Array.length s) ~e ~width s

(* [b]'s axes [lo] to [hi] as the extents of [n] consecutive elements of a walk
   of extents [ext]: each axis from [hi] down takes what the axes inside it
   leave, at most its extent. *)
let fill_box b ~ext ~lo ~hi n =
  let n = ref n in
  for k = hi downto lo do
    b.(k) <- Int.min ext.(k) !n;
    n := ceil_div !n ext.(k)
  done

(* The extents of a box of [outputs] consecutive outputs by [terms] consecutive
   terms of each, on a walk of extents [ext] whose first [kept] axes are its
   outputs', as {!fill_box} fills them. *)
let box ~ext ~kept ~outputs ~terms =
  let b = Array.make (Array.length ext) 1 in
  fill_box b ~ext ~lo:0 ~hi:(kept - 1) outputs;
  fill_box b ~ext ~lo:kept ~hi:(Array.length ext - 1) terms;
  b

(* The bytes a box [b] of [o] moves, by {!run_bytes} over the box's axes from
   the least stride out, so that its outputs side by side share their lines as
   its terms do. A padded load's or a scalar's are {!traffic}'s of the box's
   elements. Axes already in that order, as a row's are, are taken as they are,
   which allocates nothing. *)
let box_traffic ~ext b (o : operand) =
  let rank = Array.length b in
  let e = Array.fold_left ( * ) 1 b in
  match o.access with
  | Strided s when Array.length s = rank ->
      let width = Int.max 1 (bits o.dtype / 8) in
      let ordered = ref true in
      for k = 1 to rank - 1 do
        if abs s.(k) > abs s.(k - 1) then ordered := false
      done;
      if !ordered then run_bytes ~ext:b ~rank ~e ~width s
      else begin
        (* The axes by decreasing stride, by insertion. *)
        let by = Array.init rank Fun.id in
        for i = 1 to rank - 1 do
          let k = by.(i) and j = ref (i - 1) in
          while !j >= 0 && abs s.(by.(!j)) < abs s.(k) do
            by.(!j + 1) <- by.(!j);
            decr j
          done;
          by.(!j + 1) <- k
        done;
        run_bytes
          ~ext:(Array.map (Array.get b) by)
          ~rank ~e ~width
          (Array.map (Array.get s) by)
      end
  | Strided _ | Pad _ -> traffic ~ext ~e o

(* The bytes a box [b] of each of [os] moves, as {!box_traffic}. *)
let box_traffics ~ext b (os : operand array) =
  let t = ref 0 in
  for k = 0 to Array.length os - 1 do
    t := !t + box_traffic ~ext b os.(k)
  done;
  !t

(* The bytes [e] elements of each of [os] move, as {!traffic}. *)
let traffics ~ext ~e (os : operand array) =
  let t = ref 0 in
  for k = 0 to Array.length os - 1 do
    t := !t + traffic ~ext ~e os.(k)
  done;
  !t

let seconds ~groups ~resident cost =
  Plan.seconds ~threads ~groups:(groups, 1, 1) ~resident cost

exception Over_budget

(* The most groups, at most [total], a launch whose cost [cost n] for [n] groups
   is within {!Plan.allowed}; a multiple of [step] where it is below [total].
   Raises [Over_budget] where [step] groups exceed it. *)
let most_groups ~total ~step ~resident cost =
  let fits n = seconds ~groups:n ~resident (cost n) <= Plan.allowed () in
  if fits total then total
  else
    match Plan.most ~fits:(fun m -> fits (m * step)) (ceil_div total step) with
    | None -> raise Over_budget
    | Some m -> step * m

(* Tables and launches *)

(* The table of a loop over the walk [w], of loads [ops] whose fields are
   [ofield k], each in [staged] staged, and destinations [dsts]
   whose fields are [dfield k], and [t]'s tape; and the header words it
   sets. *)
(* Pushes onto [ws] the walk [w]'s extents, then their divisors, and
   sets the header [h]'s rank and indices. *)
let push_walk ws w h =
  h.(K.loop_rank - K.loop_mode) <- Array.length w.ext;
  Array.iter (push_int ws) w.ext;
  Array.iter (fun e -> push_int ws (Plan.divisor e)) w.ext;
  h.(K.loop_numel - K.loop_mode) <- Array.fold_left ( * ) 1 w.ext

(* Pushes onto [ws] the descriptor of [o] over the walk [w], its buffer that of
   field [fld]. *)
let push_descriptor ws w (o : operand) ~fld =
  let padded = match o.access with Pad _ -> 1 | Strided _ -> 0 in
  let b = bits o.dtype in
  let phase = if b < 8 then fst (low_bit o b) mod 32 else 0 in
  push_int ws
    (code o.dtype lor (padded lsl 8) lor (fld lsl 16) lor (phase lsl 24));
  push_int ws (o.offset - o.lo);
  match o.access with
  | Strided s -> Array.iter (push_int ws) s
  | Pad { fill; reads; _ } ->
      let b = Bytes.make 16 '\000' in
      Bytes.blit_string fill 0 b 0 (String.length fill);
      push ws (Bytes.get_int64_le b 0);
      push ws (Bytes.get_int64_le b 8);
      push_int ws (Array.length reads);
      Array.iter
        (fun (r : axis_read) ->
          let ax a = if a < 0 then 0xffff else kernel_axis w a in
          List.iter (push_int ws)
            [
              r.lo;
              r.interior;
              r.extent;
              r.stride;
              ax r.a1 lor (ax r.a2 lsl 16);
              r.c1;
              r.c2;
            ];
          push_int ws (Plan.divisor (r.interior + 1)))
        reads

(* A staged load's stage, [j] from 0, in its descriptor at word [d] of [ws]:
   from 1, in the descriptor's first word. *)
let mark_stage ws d j =
  Bytes.set_int64_le ws.b (8 * d)
    (Int64.logor
       (Bytes.get_int64_le ws.b (8 * d))
       (Int64.shift_left (Int64.of_int (j + 1)) 32))

let table ?(staged = [||]) w (ops : operand array) ~ofield
    (dsts : operand array) ~dfield (t : tape) (h : int array) =
  let ws = words_for 64 in
  push_walk ws w h;
  let tables field (os : operand array) fld =
    let at = ws.n in
    h.(field - K.loop_mode) <- at;
    Array.iter (fun _ -> push ws 0L) os;
    Array.iteri
      (fun k (o : operand) ->
        Bytes.set_int64_le ws.b (8 * (at + k)) (Int64.of_int ws.n);
        push_descriptor ws w o ~fld:(fld k))
      os
  in
  tables K.loop_loads ops ofield;
  tables K.loop_dsts dsts dfield;
  for j = 0 to Array.length staged - 1 do
    let at = h.(K.loop_loads - K.loop_mode) + staged.(j) in
    mark_stage ws (Int64.to_int (Bytes.get_int64_le ws.b (8 * at))) j
  done;
  h.(K.loop_tape - K.loop_mode) <- ws.n;
  let copy (s : words) =
    for k = 0 to s.n - 1 do
      push ws (get s k)
    done
  in
  h.(K.loop_prologue - K.loop_mode) <- t.pro.n;
  copy t.pro;
  h.(K.loop_body - K.loop_mode) <- t.body.n;
  copy t.body;
  h.(K.loop_epilogue - K.loop_mode) <- t.epi.n;
  copy t.epi;
  h.(K.loop_table - K.loop_mode) <- ws.n;
  Bytes.sub_string ws.b 0 (8 * ws.n)

(* The direct form [f]'s operand count and operand [j]'s load (kernels.h,
   NX_LOOP_DIRECT). *)
let form_count f = (f lsr K.direct_count) land 3
let form_load f j = (f lsr (K.direct_operands + (4 * j))) land 3

(* Slot [k]'s word in a direct map's table (kernels.h, NX_DIRECT_SLOTS). *)
let direct_slot k = K.direct_slots + (k * K.direct_stride)

(* The slot of the first of the form [f]'s [m] operands from [j] that reads load
   [a]. *)
let rec reader f m a j =
  if j = m then invalid_arg "Loop: a direct map's load no operand reads"
  else if form_load f j = a then 1 + j
  else reader f m a (j + 1)

(* The or of the direct form [f]'s hows (kernels.h, NX_LOOP_DIRECT): its
   operands' from [j] below [m], and its result's. *)
let rec hows f j m =
  if j >= m then (f lsr K.direct_result) land 3
  else (f lsr (K.direct_operands + (4 * j) + 2)) land 3 lor hows f (j + 1) m

(* Whether [f] takes each operand's element and gives its result as it is: every
   how is NX_DIRECT_AS_IS, which is 0. *)
let as_is f = hows f 0 ((f lsr K.direct_count) land 3) = K.direct_as_is

(* Slot [k]'s place (kernels.h, NX_DIRECT_PLACES) over the walk [w] of one axis:
   of stride 1 along it, flat, its quads on their size where its first element
   is, as each quad starts at an index of a multiple of 4; of stride 0, a load,
   at its first element. An axis of one element takes stride 1, as its only
   index is 0. *)
let place_in w k (o : operand) =
  match o.access with
  | Pad _ -> K.direct_placed
  | Strided s ->
      let st = if w.ext.(0) = 1 then 1 else s.(0) in
      if st = 0 && k > 0 then K.direct_first
      else if st <> 1 then K.direct_placed
      else if first_on o (4 * bits o.dtype / 8) then K.direct_flat_on
      else K.direct_flat

(* The form [f] of a direct map over [ops] into [dst], over the walk [w], with
   each slot's place where the map runs a flat kernel (loop_map_direct32_flat):
   its walk is one axis, it takes and gives each element as it is, each slot is
   of 32 bits and none is placed by its descriptor. Else [f]. Every kernel masks
   the form's fields, so one that runs a form with places, as a wide map's
   loop_map_direct does, reads them as nothing; and {!slot_operand} moves a flat
   slot's first position into its field, from which every kernel finds it. No
   test reaches the wide case: a wide map of 32-bit elements is 16 GiB. *)
let placed_form w (ops : operand array) (dst : operand) f =
  let m = form_count f in
  let slot k = if k = 0 then dst else ops.(form_load f (k - 1)) in
  let rec flat k =
    k > m
    ||
    let o = slot k in
    bits o.dtype = 32 && place_in w k o <> K.direct_placed && flat (k + 1)
  in
  if Array.length w.ext <> 1 || (not (as_is f)) || not (flat 0) then f
  else
    let placed = ref f in
    for k = 0 to m do
      let code = place_in w k (slot k) in
      placed := !placed lor (code lsl (K.direct_places + (2 * k)))
    done;
    !placed

(* The place of slot [k] in the form [f] (kernels.h, NX_DIRECT_PLACES). *)
let place_of f k = (f lsr (K.direct_places + (2 * k))) land 3

(* Slot [k]'s operand [o] as the direct kernels read it by the form [f]: a slot
   that reads no descriptor finds its first element at its field's offset
   ({!of_op}), so its low bound is its first position. *)
let slot_operand f k (o : operand) =
  if place_of f k = K.direct_placed then o else { o with lo = o.offset }

(* The table of a direct map of the form [f] over [ops] into [dst] (kernels.h,
   NX_DIRECT_SLOTS): the walk, the slots, then the loads' table, each load
   naming the slot of the first operand that reads it, and the destinations'. A
   staged load's stage is in each slot that reads it. The map's tape is not in
   it: the direct kernels run none. *)
let direct_table ?(staged = [||]) w (ops : operand array) (dst : operand) f
    (h : int array) =
  let ws = words_for 64 in
  push_walk ws w h;
  let m = form_count f in
  for k = 0 to m do
    while ws.n < direct_slot k do
      push ws 0L
    done;
    let o = if k = 0 then dst else ops.(form_load f (k - 1)) in
    push_descriptor ws w (slot_operand f k o) ~fld:k
  done;
  h.(K.loop_loads - K.loop_mode) <- ws.n;
  for a = 0 to Array.length ops - 1 do
    push_int ws (direct_slot (reader f m a 0))
  done;
  h.(K.loop_dsts - K.loop_mode) <- ws.n;
  push_int ws (direct_slot 0);
  for i = 0 to Array.length staged - 1 do
    for j = 0 to m - 1 do
      if form_load f j = staged.(i) then mark_stage ws (direct_slot (1 + j)) i
    done
  done;
  h.(K.loop_prologue - K.loop_mode) <- 0;
  h.(K.loop_body - K.loop_mode) <- 0;
  h.(K.loop_epilogue - K.loop_mode) <- 0;
  h.(K.loop_tape - K.loop_mode) <- ws.n;
  h.(K.loop_table - K.loop_mode) <- ws.n;
  assert (ws.n <= K.loop_holds);
  Bytes.sub_string ws.b 0 (8 * ws.n)

(* Whether a loop of [n] indices over [os] takes 64-bit indices: past 2^32
   indices, an operand's positions past 2^32 apart, or a padded coordinate of
   magnitude 2^31 or more, which a 32-bit int does not hold. *)
let wide n (os : operand array) =
  let far (r : axis_read) = r.reach >= wide_bound / 2 in
  n >= wide_bound
  || Array.exists
       (fun (o : operand) ->
         o.hi - o.lo >= wide_bound
         ||
         match o.access with
         | Pad { reads; _ } -> Array.exists far reads
         | Strided _ -> false)
       os

(* Lean loops (tape.h, kind): a word's kinds, at float32, int32 and uint32, and
   the ops a kernel of 4 elements a thread runs, over plain operands of whole
   bytes of at most 32 bits. *)

let[@inline] lean_kind t =
  let ct = Int64.to_int (Int64.logand t 0xffffL) lsr 7 in
  ct = K.loop_f32 || ct = K.loop_i32 || ct = K.loop_u32

let lean_word ws k =
  let t = get ws k in
  let op = Int64.to_int (Int64.logand t 0xffffL) in
  let b = Int64.to_int (Int64.logand (Int64.shift_right_logical t 40) 0xfffL) in
  let e = Int64.to_int (Int64.logand (Int64.shift_right_logical t 52) 0xfffL) in
  if op < K.loop_load then lean_kind t
  else if op = K.loop_decode || op = K.loop_encode then
    b = D.code D.Bfloat16 || b = D.code D.Float16
  else if op = K.loop_cast then
    b = e
    && match D.of_code b with Some (D.Any d) -> D.bits d <= 32 | None -> false
  else
    op = K.loop_load || op = K.loop_store || op = K.loop_const
    || op = K.loop_where1 || op = K.loop_truth || op = K.loop_wrap

let lean_words ws =
  let k = ref 0 in
  while !k < ws.n && lean_word ws !k do
    incr k
  done;
  !k >= ws.n

(* Whether a kernel of 4 elements a thread reads or writes [o]: whole bytes of
   at most 32 bits, plain, or padded where [padded]. *)
let lean_operand ?(padded = false) (o : operand) =
  let b = bits o.dtype in
  (b = 8 || b = 16 || b = 32)
  && match o.access with Strided _ -> true | Pad _ -> padded

(* Whether a kernel of 4 elements a thread stores a reduction's output of
   [carrier] into [o] (fold.h's put_dst): whole bytes of at most 32 bits, of the
   carrier or, from float32, bfloat16 or float16. *)
let lean_store ~carrier (o : operand) =
  lean_operand o
  && (o.dtype = carrier
     || carrier = float32
        && (o.dtype = D.Any D.Bfloat16 || o.dtype = D.Any D.Float16))

(* Whether a kernel of 4 stores each of [dsts], but an Arg's positions, its
   second, which take a store of their own. *)
let lean_stores ~carrier ~arg (dsts : operand array) =
  let lean = ref true in
  for i = 0 to Array.length dsts - 1 do
    if not ((arg && i = 1) || lean_store ~carrier dsts.(i)) then lean := false
  done;
  !lean

let rec lean_from ~padded (os : operand array) k =
  k = Array.length os
  || (lean_operand ~padded os.(k) && lean_from ~padded os (k + 1))

(* Whether a loop's thread may take 4 elements: its tape [t] and its operands
   [os] are lean, padded ones where [padded], and its indices are 32-bit. *)
let lean_loop ?(padded = false) ~wide (t : tape) os =
  (not wide) && lean_words t.body && lean_words t.epi && lean_from ~padded os 0

(* Elements and planes

   A thread takes 4 elements where its loop's tape and operands are lean and its
   indices 32-bit ({!lean_loop}), a map's padded loads lean too, its group's
   planes then take at most 16 KiB, at most 4 planes, so that a core holds four
   groups and 1024 threads, and one group's run of 4 elements a thread fits a
   launch; else 1, up to a group's 32 KiB, its [extra] bytes included. Decided
   here for every mode, so that a loop's least work is a run of one element a
   thread, a map's and a part's as a scan's. The kernels are compiled for 4 and
   for 1 alone (loop.metal). *)

let plane_bytes v = v * threads * 4

(* The most a mode adds to one group's run of 4 elements a thread past its tape,
   at its least: a reduction's 16 slots' stores and states, a network tile's two
   joins a term, each an exp at Logsumexp, its sweeps, and a scan chunk's first
   thread folding 16 threads' 4 terms. *)
let mode_cost =
  {
    Plan.bytes = Plan.line * threads;
    ops = 2 * threads * K.loop_most * costly;
    chain = 16 * K.loop_most * fold_links;
  }

(* The elements a thread of a loop over the operands [os] takes, the tape [t]
   lowered over a walk of extents [ext], its indices 64-bit where [wide]; 1
   where its destinations take no kernel of 4's stores ([stores]). A padded load
   is lean where [padded], a map's. *)
let elements ?(extra = 0) ?padded ~stores ~wide ~ext (t : tape) os =
  let planes = t.pl.top and rank = Array.length ext in
  let run_fits () =
    let e = K.loop_most * threads in
    let bytes = traffics ~ext ~e os in
    let body_ops, body_chain =
      tape_cost ~dispatch ~rank t.body ~from:0 t.body.n
    in
    let epi_ops, epi_chain = tape_cost ~dispatch ~rank t.epi ~from:0 t.epi.n in
    let run =
      {
        Plan.bytes = bytes + mode_cost.bytes;
        ops = (e * (body_ops + epi_ops)) + mode_cost.ops;
        chain = (K.loop_most * (body_chain + epi_chain)) + mode_cost.chain;
      }
    in
    seconds ~groups:1 ~resident:1 run <= Plan.allowed ()
  in
  if
    stores
    && lean_loop ?padded ~wide t os
    && (planes * plane_bytes K.loop_most) + extra <= 16 * 1024
    && run_fits ()
  then K.loop_most
  else if (planes * plane_bytes 1) + extra <= Plan.group_memory then 1
  else raise Too_wide

(* A header: its words from NX_LOOP_MODE, [v] and [wide] set. *)
let header ?(wide = false) v =
  let h = Array.make (K.loop_held - K.loop_mode) 0 in
  h.(0) <- (v lor if wide then K.loop_wide else 0);
  h.(K.loop_carry - K.loop_mode) <- -1;
  h

let seti h i x = h.(i - K.loop_mode) <- x

(* The fields of a launch over [ops] then [dsts], then, [shared], the workspace
   and the tickets. *)
let of_op write (o : operand) =
  let b = bits o.dtype in
  let at =
    if b >= 8 then o.lo * (b / 8)
    else
      let bit, skew = low_bit o b in
      (bit / 32 * 4) - skew
  in
  { target = Buf o.buffer; at; write }

let fields ?(shared = false) (ops : operand array) (dsts : operand array) =
  let fs =
    Array.concat
      [
        Array.map (of_op false) ops;
        Array.map (of_op true) dsts;
        (if shared then
           [|
             { target = Buf Space; at = 0; write = true };
             { target = Tickets; at = 0; write = true };
           |]
         else [||]);
      ]
  in
  if Array.length fs > K.loop_buffers then raise Too_wide;
  fs

(* The fields of a direct map of the form [f] over [ops] into [dst], one a slot
   (kernels.h, NX_DIRECT_SLOTS): the destination's, then each operand's
   load's. *)
let direct_fields (ops : operand array) (dst : operand) f =
  Array.init
    (1 + form_count f)
    (fun k ->
      if k = 0 then of_op true (slot_operand f 0 dst)
      else of_op false (slot_operand f k ops.(form_load f (k - 1))))

(* A launch's mode and its kernel, of the v and width its header [h] names. *)
type mode =
  | Map
  | Direct
  | Cast of K.t
  | Reduce
  | Reduce_direct
  | Reduce_centered
  | Reduce_row of K.t
  | Merge
  | Scan
  | Carries
  | Network

(* The narrow float whose codec the direct form [f] decodes each operand and
   encodes its result by, NX_DIRECT_BF16 or NX_DIRECT_F16; else
   NX_DIRECT_AS_IS. *)
let codec f =
  let m = (f lsr K.direct_count) land 3 in
  let c = (f lsr K.direct_result) land 3 in
  let rec each j =
    j >= m
    || ((f lsr (K.direct_operands + (4 * j) + 2)) land 3 = c && each (j + 1))
  in
  if (c = K.direct_bf16 || c = K.direct_f16) && each 0 then c
  else K.direct_as_is

(* Whether [op], a direct form's op, is a long kind, which a 32-bit direct map
   runs in loop_map_direct32_long (kernels.h, NX_DIRECT_LONG_REAL1). *)
let long_kind op =
  let ct = op lsr 7 and k = op land 127 in
  let has set bit = (set lsr bit) land 1 = 1 in
  let real = ct = K.loop_f32 in
  if ct <> K.loop_f32 && ct <> K.loop_i32 && ct <> K.loop_u32 then false
  else if k < K.loop_binary then
    has (if real then K.direct_long_real1 else K.direct_long_int1) k
  else if k < K.loop_ternary then
    has
      (if real then K.direct_long_real2 else K.direct_long_int2)
      (k - K.loop_binary)
  else false

(* Whether the direct form [f] is one an 8-bit kernel runs: each operand taken
   as it is or as a truth, the result given as it is or wrapped into a dtype of
   8 bits, whose wrap is the stored byte, and its op no long kind. *)
let bytes_form f =
  let m = (f lsr K.direct_count) land 3 in
  let rec each j =
    j >= m
    ||
    let how = (f lsr (K.direct_operands + (4 * j) + 2)) land 3 in
    (how = K.direct_as_is || how = K.direct_truth) && each (j + 1)
  in
  let r = (f lsr K.direct_result) land 3 in
  let byte_wrap () =
    match D.of_code ((f lsr K.direct_wrap_dt) land 0xff) with
    | Some (D.Any d) -> D.bits d = 8
    | None -> false
  in
  (r = K.direct_as_is || (r = K.direct_wrap && byte_wrap ()))
  && each 0
  && not (long_kind (f land 0xffff))

let kernel_of mode h =
  let v = h.(0) land 0xff and wide = h.(0) land K.loop_wide <> 0 in
  let tiled = h.(0) land K.loop_tiled <> 0 in
  let padded = h.(0) land K.loop_padded <> 0 in
  let b32 = h.(0) land K.loop_bits32 <> 0 in
  let b16 = h.(0) land K.loop_bits16 <> 0 in
  let b8 = h.(0) land K.loop_bits8 <> 0 in
  let pick ~v4 ~v1 ~wide_k =
    match (v, wide) with
    | 1, true -> wide_k
    | 4, false -> v4
    | 1, false -> v1
    | _ ->
        invalid_arg (Printf.sprintf "Loop: no kernel of v %d, wide %b" v wide)
  in
  (* Tiles and direct maps run 4 elements a thread of 32-bit indices, and read
     no padded load. *)
  let lean what k =
    if v = K.loop_most && (not wide) && not padded then k
    else
      invalid_arg
        (Printf.sprintf "Loop: no %s kernel of v %d, wide %b, padded %b" what v
           wide padded)
  in
  match mode with
  | Map | Cast _ | Reduce | Reduce_direct | Reduce_centered | Reduce_row _
  | Merge | Scan | Carries | Network
    when b32 || b16 || b8 ->
      invalid_arg
        "Loop: no kernel of 32-, 16- or 8-bit lanes but a direct map's"
  | Direct when (b32 && b16) || (b32 && b8) || (b16 && b8) ->
      invalid_arg "Loop: no kernel of lanes of two widths"
  | Direct when b32 ->
      (* 32-bit lanes run as they come, in quads. *)
      let quads = h.(0) land K.loop_quads <> 0 in
      let form = h.(K.loop_direct - K.loop_mode) in
      if not (quads && as_is form) then
        invalid_arg
          (Printf.sprintf "Loop: no 32-bit kernel of quads %b, form %#x" quads
             form);
      (* A flat map's slots are each placed flat or at their first element
         ({!placed_form}), and it takes no tiles: its walk is one axis. *)
      let flat = place_of form 0 <> K.direct_placed in
      let long = long_kind (form land 0xffff) in
      lean "32-bit"
        (match (long, tiled, flat) with
        | false, false, true -> K.Loop_map_direct32_flat
        | true, false, true -> K.Loop_map_direct32_long_flat
        | false, false, false -> K.Loop_map_direct32
        | false, true, _ -> K.Loop_map_direct32_tiled
        | true, false, false -> K.Loop_map_direct32_long
        | true, true, _ -> K.Loop_map_direct32_long_tiled)
  | Direct when b16 ->
      (* A narrow float's lanes run in quads, by its codec. *)
      let quads = h.(0) land K.loop_quads <> 0 in
      let form = h.(K.loop_direct - K.loop_mode) in
      let c = codec form in
      if not (quads && c <> K.direct_as_is) then
        invalid_arg
          (Printf.sprintf "Loop: no 16-bit kernel of quads %b, form %#x" quads
             form);
      let bf16 = c = K.direct_bf16 in
      lean "16-bit"
        (match (bf16, long_kind (form land 0xffff), tiled) with
        | true, false, false -> K.Loop_map_direct16_bf16
        | true, false, true -> K.Loop_map_direct16_bf16_tiled
        | true, true, false -> K.Loop_map_direct16_bf16_long
        | true, true, true -> K.Loop_map_direct16_bf16_long_tiled
        | false, false, false -> K.Loop_map_direct16_f16
        | false, false, true -> K.Loop_map_direct16_f16_tiled
        | false, true, false -> K.Loop_map_direct16_f16_long
        | false, true, true -> K.Loop_map_direct16_f16_long_tiled)
  | Direct when b8 ->
      (* 8-bit lanes run in quads, of no long kind, which keeps the generic
         body. *)
      let quads = h.(0) land K.loop_quads <> 0 in
      let form = h.(K.loop_direct - K.loop_mode) in
      if not (quads && bytes_form form) then
        invalid_arg
          (Printf.sprintf "Loop: no 8-bit kernel of quads %b, form %#x" quads
             form);
      lean "8-bit"
        (if tiled then K.Loop_map_direct8_tiled else K.Loop_map_direct8)
  | Map when tiled -> lean "tiled" K.Loop_map_tiled
  | Map ->
      let v4 = if padded then K.Loop_map_padded else K.Loop_map_v4 in
      pick ~v4 ~v1:K.Loop_map_v1 ~wide_k:K.Loop_map_wide
  | Direct when tiled -> lean "tiled" K.Loop_map_direct_tiled
  | Direct -> lean "direct" K.Loop_map_direct
  | Cast _ when tiled -> invalid_arg "Loop: no cast kernel of tiles"
  | Cast k -> lean "cast" k
  | Reduce ->
      pick ~v4:K.Loop_reduce_v4 ~v1:K.Loop_reduce_v1 ~wide_k:K.Loop_reduce_wide
  | Reduce_direct -> K.Loop_reduce_direct
  | Reduce_centered -> K.Loop_reduce_centered
  | Reduce_row k -> k
  | Scan -> pick ~v4:K.Loop_scan_v4 ~v1:K.Loop_scan_v1 ~wide_k:K.Loop_scan_wide
  | Network ->
      pick ~v4:K.Loop_network_v4 ~v1:K.Loop_network_v1
        ~wide_k:K.Loop_network_wide
  | Merge -> K.Loop_merge
  | Carries -> K.Loop_carries

(* The launches of [mode] over groups [first] to [total] - 1, [step] at a time
   at least and at most [most], each of [cost n] for its [n] groups and [tickets
   first n] tickets, from the header [h] and the table [table], each kernel's
   groups held as [holds] says. *)
let launches ?(first = 0) ?(most = max_int) ~holds mode ~total ~step ~shared
    ~fields ~tickets ~table ~h ~cost () =
  let kernel = kernel_of mode h in
  let resident = Plan.resident ~granted:holds kernel ~shared in
  let most =
    Int.min most (most_groups ~total:(total - first) ~step ~resident cost)
  in
  let rec go first acc =
    if first >= total then List.rev acc
    else
      let n = Int.min most (total - first) in
      let h = Array.copy h in
      seti h K.loop_first first;
      seti h K.loop_end (first + n);
      let words = Plan.words kernel in
      Array.iteri (fun i x -> Plan.set words (K.loop_mode + i) x) h;
      let held = String.length table <= 8 * K.loop_holds in
      if held then
        Bytes.blit_string table 0 words (8 * K.loop_held) (String.length table);
      let l =
        {
          kernel;
          groups = n;
          shared;
          resident;
          cost = cost n;
          header = words;
          table = (if held then None else Some table);
          fields;
          tickets = tickets first n;
        }
      in
      go (first + n) (l :: acc)
  in
  go first []

(* [build ~eager] with every load read first where its planes fit, else with
   only the loads [needed] read first, the others each before its first reader,
   which frees their planes sooner. *)
let fitted build ~needed =
  match build ~eager:(fun _ -> true) with
  | l -> l
  | exception Too_wide -> build ~eager:needed

let zero_const (D.Any d) = P.Const (D.Any d, String.make (D.bytes d 1) '\000')

(* [p] with its outputs [roots] over the operands of [ops] they read,
   renumbered, a node of no live reader that reads another operand made a
   constant. *)
let compact (p : P.t) ~roots (ops : operand array) =
  let n = Iarray.length p.nodes in
  let live = Array.make n false in
  List.iter (fun k -> live.(Iarray.get p.outs k) <- true) roots;
  for i = n - 1 downto 0 do
    if live.(i) then
      List.iter (fun a -> live.(a) <- true) (operands_of (Iarray.get p.nodes i))
  done;
  let used = Array.make (Array.length ops) false in
  Iarray.iteri
    (fun i nd ->
      match nd with P.In k when live.(i) -> used.(k) <- true | _ -> ())
    p.nodes;
  if Array.for_all Fun.id used then (p, ops)
  else
    let index = Array.make (Array.length ops) (-1) and m = ref 0 in
    Array.iteri
      (fun k u ->
        if u then (
          index.(k) <- !m;
          incr m))
      used;
    let kept =
      List.filter (fun k -> used.(k)) (List.init (Array.length ops) Fun.id)
    in
    let nodes =
      Iarray.map
        (function
          | P.In k when index.(k) >= 0 -> P.In index.(k)
          | P.In k -> zero_const (Iarray.get p.ins k)
          | nd -> nd)
        p.nodes
    in
    ( P.v
        ~ins:(Iarray.of_list (List.map (Iarray.get p.ins) kept))
        nodes ~outs:p.outs,
      Array.of_list (List.map (Array.get ops) kept) )

(* Temporaries

   A plan lays its temporaries out in buffers it numbers, each a piece of the
   call's workspace once the plan is whole: a pass tried, too wide, and planned
   again in parts takes none. *)

type stand_ins = { mutable sizes : int list; mutable count : int }

let stand_ins () = { sizes = []; count = 0 }

(* Temporaries in [buffer] of each dtype and extents of [xs], each an axis of
   the loop's or 1 where its node does not vary along it, read there at stride
   0: C-contiguous from 16-byte aligned bytes. Also the bytes they take. *)
let temporaries buffer xs =
  let at = ref 0 in
  let ts =
    List.map
      (fun (D.Any d, ext) ->
        let strides = Array.make (Array.length ext) 0 and n = ref 1 in
        for a = Array.length ext - 1 downto 0 do
          if ext.(a) > 1 then strides.(a) <- !n;
          n := !n * ext.(a)
        done;
        let byte = !at in
        at := align16 (byte + D.bytes d !n);
        let offset = byte * 8 / D.bits d in
        {
          buffer;
          dtype = D.Any d;
          offset;
          lo = offset;
          hi = offset + !n;
          access = Strided strides;
        })
      xs
  in
  (Array.of_list ts, !at)

(* Temporaries of [xs] in a buffer of their own, numbered in [s]. *)
let fresh (s : stand_ins) xs =
  let ts, bytes = temporaries (Temp s.count) xs in
  s.sizes <- bytes :: s.sizes;
  s.count <- s.count + 1;
  ts

let bigbytes s =
  let n = String.length s in
  let b = Bigarray.Array1.create Bigarray.int8_unsigned Bigarray.c_layout n in
  String.iteri (fun i c -> Bigarray.Array1.unsafe_set b i (Char.code c)) s;
  b

(* Whether [passes] name each of [s]'s temporaries. *)
let named (s : stand_ins) (passes : pass list) =
  let named = Array.make s.count false in
  let note f =
    match f.target with Buf (Temp i) -> named.(i) <- true | _ -> ()
  in
  let note_copy (k : copy) =
    List.iter
      (fun (o : operand) ->
        match o.buffer with
        | Temp i -> named.(i) <- true
        | Arr _ | Space | Zero -> ())
      [ k.src; k.dst ]
  in
  List.iter
    (fun c ->
      List.iter (fun l -> Array.iter note l.fields) c.launches;
      List.iter note_copy c.copies)
    passes;
  named

let sizes (s : stand_ins) = Array.of_list (List.rev s.sizes)

(* Adds [passes]' launches to [b], in order, temporary [i] at byte [base.(i)] of
   its workspace and each pass's workspace in a piece it gives back after the
   pass, which a later pass takes again. A table past the parameters is a
   constant of its bytes. *)
let emit_passes b ~base (passes : pass list) =
  (* A pass of no workspace, or a launch of no ticket, does not name the
     device's workspace or tickets: calls that take neither do not wait on each
     other. Their fields stay 0. *)
  let emit ~worked work l =
    let w = Bytes.copy l.header in
    let ticket = if l.tickets > 0 then Plan.tickets b l.tickets else 0 in
    let ref j f =
      let word = K.loop_buffer + j in
      let named slot at =
        Plan.set w word at;
        Some { Plan.word; slot; write = f.write }
      in
      match f.target with
      | Buf (Arr x) -> named (Plan.slot b x.first) f.at
      | Buf (Temp i) -> named (Plan.workspace b) (base.(i) + f.at)
      | Buf Space ->
          if worked then named (Plan.workspace b) (work + f.at) else None
      | Buf Zero ->
          let slot =
            Plan.constant b ~key:"loop zero" ~length:zero_bytes (fun () ->
                bigbytes (String.make zero_bytes '\000'))
          in
          named slot f.at
      | Tickets ->
          if l.tickets > 0 then named (Plan.tickets_slot b) (ticket + f.at)
          else None
    in
    let refs =
      List.filter_map Fun.id (Array.to_list (Array.mapi ref l.fields))
    in
    let tail =
      match l.table with
      | None -> []
      | Some t ->
          let slot =
            Plan.constant b ~key:("loop " ^ t) ~length:(String.length t)
              (fun () -> bigbytes t)
          in
          [ { Plan.word = K.loop_tail; slot; write = false } ]
    in
    Plan.launch b l.kernel ~groups:(l.groups, 1, 1) ~shared:l.shared
      ~resident:l.resident ~cost:l.cost w
      (Array.of_list (refs @ tail))
  in
  let site (o : operand) shape =
    let strides =
      match o.access with
      | Strided s -> Iarray.of_array s
      | Pad _ -> invalid_arg "Loop.plan: a copy of a padded operand"
    in
    let l = L.v ~offset:o.offset ~strides (Iarray.of_array shape) in
    match o.buffer with
    | Arr x ->
        Moves.site_at ~slot:(Plan.slot b x.first) ~base:0 ~residue:x.residue
          o.dtype l
    | Temp i ->
        Moves.site_at ~slot:(Plan.workspace b) ~base:base.(i) ~residue:0 o.dtype
          l
    | Space -> invalid_arg "Loop.plan: a copy of a pass's workspace"
    | Zero -> invalid_arg "Loop.plan: a copy of the zero constant"
  in
  let copy (k : copy) =
    let n = Array.fold_left ( * ) 1 k.shape in
    ignore
      (Moves.launch b ~n ~dst:(site k.dst k.shape) (site k.src k.shape) : float)
  in
  List.iter
    (fun c ->
      let m = Plan.mark b in
      let work = if c.work > 0 then Plan.take b c.work else 0 in
      List.iter (emit ~worked:(c.work > 0) work) c.launches;
      List.iter copy c.copies;
      Plan.release b m)
    passes

(* The plan of [passes], in order: the temporaries of [s] they name, then the
   passes ({!emit_passes}). *)
let plan (s : stand_ins) (passes : pass list) =
  let b = Plan.make () in
  let named = named s passes in
  let base =
    Array.mapi (fun i n -> if named.(i) then Plan.take b n else 0) (sizes s)
  in
  emit_passes b ~base passes;
  Plan.finish_some b

(* Maps *)

(* Whether the plain operand [o] is one of [outs]: identical to a destination,
   which a map stores only after it reads each index. *)
let identical outs (o : operand) =
  match o.access with
  | Pad _ -> false
  | Strided _ ->
      Array.exists
        (fun (y : operand) ->
          same_buf y.buffer o.buffer && y.dtype = o.dtype && y.offset = o.offset
          && y.access = o.access)
        outs

(* Padded loads' decode (tape.h, padded_place and Padded quads), in op-times an
   element of each padded load: where each element finds its coordinates,
   [pad_element] and [pad_axis] a kernel axis, or [pad_axis4] a kernel axis at 4
   elements a thread; under quads, which find them once a thread,
   [pad_quad_axis] a kernel axis; on a wide loop, [pad_wide] and [pad_divide]
   for each 64-bit divide and remainder its coordinates take.

   Fitted to an M1 Max's GPU time at its sustained clock, scaled to the M1 (32/7
   cores, 1.296/1.278 GHz), over ranks 1-5, 1-27 loads, interiors, a window, and
   8 to 64 bits. An axis is charged 1.48 times its measured cost (164, 173 and
   54 op-times; 457 a divide), so that no rank takes the estimate under 1.48
   times the time; each base is the least that keeps every case there. At 4
   elements a thread no base is needed and the least margin is 1.79 without
   quads, 2.42 with them: there the GPU overlaps the decode with the reads,
   while the estimate adds the terms. Under quads the bytes and chain alone
   bound the time. A row of rank 1 takes quads at any length, its last quad
   partial: at 2^22 - 1 and 2^22 - 3 elements its estimate and its time are
   those at 2^22, and at 2^20 - 1 the least margin is 2.25. *)
let pad_element = 493
let pad_axis = 243
let pad_axis4 = 257
let pad_quad_axis = 81
let pad_wide = 2179
let pad_divide = 677

(* The 64-bit divides a wide padded load's coordinates take on a walk [w]: each
   read axis's coordinate divides through the axes after it. *)
let wide_divides w (reads : axis_read array) =
  let rank = Array.length w.ext and n = ref 0 in
  for j = 0 to Array.length reads - 1 do
    let r = reads.(j) in
    n := !n + rank - kernel_axis w r.a1;
    if r.a2 >= 0 then n := !n + rank - kernel_axis w r.a2
  done;
  !n

(* The decode of a map's padded loads [os] an element, at a thread of [v]
   elements, under quads where [quads], on a walk [w], 64-bit where [wide]. *)
let pad_cost ~v ~quads ~wide w (os : operand array) =
  let rank = Array.length w.ext and c = ref 0 in
  for k = 0 to Array.length os - 1 do
    match os.(k).access with
    | Strided _ -> ()
    | Pad { reads; _ } ->
        c :=
          !c
          +
          if wide then pad_wide + (pad_divide * wide_divides w reads)
          else if v = K.loop_most && quads then pad_quad_axis * rank
          else if v = K.loop_most then pad_axis4 * rank
          else pad_element + (pad_axis * rank)
  done;
  !c

(* The bytes, work and chain of a tile of [v] elements a thread, over [ops] and
   [outs], of the tape [ws]'s [n] words from [from]. *)
let tile_cost ~dispatch ~ext ~v ops outs ws ~from n =
  let e = v * threads in
  let bytes = traffics ~ext ~e ops + traffics ~ext ~e outs in
  let ops, chain = tape_cost ~dispatch ~rank:(Array.length ext) ws ~from n in
  (bytes, e * ops, v * chain)

(* Whether a map of 4 elements a thread over the walk [w] takes quads (tape.h):
   a row of the last axis is a multiple of 4, or the walk is one row, whose last
   quad ends at its end; and each operand is of 8, 16 or 32 bits, or of 64 and a
   load of 1 or 4 where [cast], as a cast's kernel reads them (direct.metal),
   and plain, its last stride 1, or 0 for a load, or a padded load, whose quad's
   indices share every coordinate but the last. *)
let quads ~cast w ~loads ~dsts =
  let rank = Array.length w.ext in
  let fits ~broadcast (o : operand) =
    let b = bits o.dtype in
    (b = 8 || b = 16 || b = 32
    || (cast && (b = 64 || (broadcast && (b = 1 || b = 4)))))
    &&
    match o.access with
    | Strided s -> s.(rank - 1) = 1 || (broadcast && s.(rank - 1) = 0)
    | Pad _ -> broadcast
  in
  rank > 0
  && (rank = 1 || w.ext.(rank - 1) mod 4 = 0)
  && Array.for_all (fits ~broadcast:true) loads
  && Array.for_all (fits ~broadcast:false) dsts

(* Tiled maps (tape.h, Tiles)

   A load whose stride along the walk's last axis takes each of a quad's
   elements to a line of its own reads a line an element: t_add's transposed
   operand moves 32 times its bytes at float32. A map of 4 elements a thread and
   32-bit indices may take tiles of its walk's last two axes of extent past 1, a
   and b, a group's 32 along b by 32 along a, and stage such a load through its
   group's memory, read along b, where its lines are whole. A load is staged
   where it is plain, of 8, 16 or 32 bits, of stride 1 along b and at least a
   tile's along a, so that no two of a quad's elements share a line's read, and
   both extents take a tile. Each stage takes {!stage_bytes} past the planes,
   all within {!Plan.group_memory}, which holds at most 7, as NX_LOOP_STAGED
   packs them. A map takes tiles where its estimate with them is below its
   estimate without: a stage's memory can cut the groups a core holds, which a
   long chain pays for. *)

let tile = K.loop_tile
let stage_bytes = 4 * K.loop_stage_words

(* The group memory of a direct map that stages nothing: its argument for stages
   is bound, at a length of a multiple of 16. *)
let stub_bytes = 16

(* A barrier's links, as a network's level charges them. *)
let barrier_links = 8

(* A tiling: the axes a and b, the staged loads, and the 2^shift tiles along b a
   run of groups takes before the next along a: the tiles a line of the
   narrowest staged load holds along b. *)
type tiling = { a : int; b : int; staged : int array; shift : int }

let stageable ~a ~b (o : operand) =
  let w = bits o.dtype in
  (w = 8 || w = 16 || w = 32)
  &&
  match o.access with
  | Strided s -> s.(b) = 1 && abs s.(a) >= tile
  | Pad _ -> false

let rec stageables ~a ~b (loads : operand array) k n =
  if k = Array.length loads then n
  else
    stageables ~a ~b loads (k + 1)
      (if stageable ~a ~b loads.(k) then n + 1 else n)

(* The tiling of a map of [v] elements a thread over the walk [w] of loads
   [loads], its planes taking [planes] bytes, or [None]. A map of 4 elements a
   thread reads no coordinate, and takes no tiles where it reads a padded load,
   so its walk is merged, which keeps no axis of extent 1: a and b are its last
   two axes. *)
let tiling ~wide ~v ~planes (w : walk) (loads : operand array) =
  let rank = Array.length w.ext in
  if wide || v <> K.loop_most || rank < 2 || Array.exists is_padded loads then
    None
  else
    let a = rank - 1 and b = rank - 2 in
    if w.ext.(a) < tile || w.ext.(b) < tile then None
    else
      let k = stageables ~a ~b loads 0 0 in
      if k = 0 || planes + (k * stage_bytes) > Plan.group_memory then None
      else begin
        let staged = Array.make k 0 and j = ref 0 and width = ref 4 in
        for i = 0 to Array.length loads - 1 do
          if stageable ~a ~b loads.(i) then begin
            staged.(!j) <- i;
            width := Int.min !width (bits loads.(i).dtype / 8);
            incr j
          end
        done;
        let rec shift s =
          if (tile lsl s) * !width >= Plan.line then s else shift (s + 1)
        in
        Some { a; b; staged; shift = shift 0 }
      end

(* The bytes a tile moves of [o]: its runs over the tile's axes, along b first
   where [staged]. *)
let tile_traffic (tl : tiling) ~staged (o : operand) =
  let ext = [| tile; tile |] and e = tile * tile in
  match o.access with
  | Pad _ -> traffic ~ext ~e o
  | Strided s ->
      let s =
        if staged then [| s.(tl.a); s.(tl.b) |] else [| s.(tl.b); s.(tl.a) |]
      in
      traffic ~ext ~e { o with access = Strided s }

(* The bytes, work and chain of a tile of [tl] over [ops] and [outs], of the
   tape [ws]'s [n] words from [from]: a staged element's write and read of its
   stage, a stage's read and barrier. *)
let tiled_cost ~dispatch ~ext (tl : tiling) ops outs ws ~from n =
  let v = K.loop_most in
  let e = v * threads and k = Array.length tl.staged in
  let bytes = ref 0 in
  Array.iteri
    (fun j o ->
      bytes := !bytes + tile_traffic tl ~staged:(Array.mem j tl.staged) o)
    ops;
  Array.iter (fun o -> bytes := !bytes + tile_traffic tl ~staged:false o) outs;
  let ops, chain = tape_cost ~dispatch ~rank:(Array.length ext) ws ~from n in
  ( !bytes,
    e * (ops + (2 * k)),
    (v * chain) + (k * (Plan.read_links + barrier_links)) )

(* The header words of the tiling [tl] on extents [ext], its stages from word
   [stage] of the group's memory, and its groups. *)
let set_tiling h (tl : tiling) ~ext ~stage =
  h.(0) <- h.(0) lor K.loop_tiled;
  seti h K.loop_tile_a ext.(tl.a);
  seti h K.loop_tile_b ext.(tl.b);
  seti h K.loop_stage stage;
  let list = ref (Array.length tl.staged) in
  Array.iteri (fun j k -> list := !list lor (k lsl (4 + (8 * j)))) tl.staged;
  seti h K.loop_staged !list;
  (* A plane's groups: its tiles along b rounded up to whole runs, so that a
     group's tile comes from its number by two host divisors. A run takes at
     most the plane's tiles along b, so that fewer than half its groups have
     none. *)
  let ta = ceil_div ext.(tl.a) tile and tb = ceil_div ext.(tl.b) tile in
  let rec fits s = if s > 0 && 1 lsl s > tb then fits (s - 1) else s in
  let shift = fits tl.shift in
  seti h K.loop_tile_run shift;
  let run = 1 lsl shift in
  let plane = ta * (ceil_div tb run * run) in
  seti h K.loop_tiles_a (Plan.divisor ta);
  seti h K.loop_tiles (Plan.divisor plane);
  let n = Array.fold_left ( * ) 1 ext in
  n / (ext.(tl.a) * ext.(tl.b)) * plane

(* Direct maps

   A map takes loop_map_direct (direct.metal) where its tape is a direct form:
   no constant, at most three loads, each read as it is, as a truth or decoded
   from bfloat16 or float16; then one kind, a Where of one word, or neither; the
   result as it is, wrapped or encoded into bfloat16 or float16; one store, of
   destination 0. The kernel runs the tape's steps through their own functions,
   in registers, so the choice reads the tape's words alone, never an extent or
   a value, and changes no bit. Its loop is lean ({!lean_loop}) and of 32-bit
   indices, as a kernel of 4 elements a thread's is. *)

(* The most words a direct form takes: three loads, a decode or a truth of each,
   the op, its result's wrap or encode, the store. *)
let direct_words = 9

(* The direct form of a tape, read word by word. *)
module Form = struct
  (* A plane's value as a direct form goes, an int, so that the test allocates
     little on a call's path: load [k] taken by [how], or the op's result given
     to the store by [how], wrapped to [dt]; [-1] for none. *)
  let loaded k how = k lor (how lsl 2)
  let result_of how dt = 16 lor (how lsl 2) lor (dt lsl 5)
  let is_result v = v land 16 <> 0
  let how_of v = (v lsr 2) land 3

  (* A direct form as its words go: its planes' values, its op's word with its
     operands ([-1] before the op), and its store's result ([-1] before the
     store). *)
  type t = { values : int array; mutable op : int; mutable stored : int }

  exception Not_direct

  let value f p = if f.values.(p) < 0 then raise Not_direct else f.values.(p)

  (* Plane [p]'s value as an operand: a load's. *)
  let taken f p =
    let v = value f p in
    if is_result v then raise Not_direct else v

  (* The op [o] of the [n] operands [args], once. *)
  let compute f o n args =
    if f.op >= 0 then raise Not_direct;
    f.op <- o lor (n lsl K.direct_count) lor (args lsl K.direct_operands)

  (* Plane [p]'s value as the op's result: its own, or its load's as it is. *)
  let result f p =
    let v = value f p in
    if is_result v then v
    else (
      compute f K.loop_cast 1 v;
      result_of K.direct_as_is 0)

  let narrow b =
    if b = D.code D.Bfloat16 then K.direct_bf16
    else if b = D.code D.Float16 then K.direct_f16
    else raise Not_direct

  let[@inline] field w s =
    Int64.to_int (Int64.logand (Int64.shift_right_logical w s) 0xfffL)

  (* The operands [a], [b] and [c], loads' values, as the op's word takes
     them. *)
  let three f a b c = taken f a lor (taken f b lsl 4) lor (taken f c lsl 8)

  let form_word f ws k =
    let w = get ws k in
    let o = Int64.to_int (Int64.logand w 0xffffL) in
    let d = field w 16 and a = field w 28 in
    let b = field w 40 and c = field w 52 in
    if o < K.loop_load then begin
      if not (lean_kind w) then raise Not_direct;
      let k = o land 127 in
      if k < K.loop_binary then compute f o 1 (taken f a)
      else if k < K.loop_ternary then
        compute f o 2 (taken f a lor (taken f b lsl 4))
      else compute f o 3 (three f a b c);
      f.values.(d) <- result_of K.direct_as_is 0
    end
    else if o = K.loop_load then
      if a > 3 then raise Not_direct
      else f.values.(d) <- loaded a K.direct_as_is
    else if o = K.loop_decode || o = K.loop_truth then (
      let v = taken f a in
      if how_of v <> K.direct_as_is then raise Not_direct;
      let how = if o = K.loop_truth then K.direct_truth else narrow b in
      f.values.(d) <- loaded (v land 3) how)
    else if o = K.loop_where1 then begin
      compute f o 3 (three f a b c);
      f.values.(d) <- result_of K.direct_as_is 0
    end
    else if o = K.loop_encode || o = K.loop_wrap then begin
      if how_of (result f a) <> K.direct_as_is then raise Not_direct;
      f.values.(d) <-
        (if o = K.loop_wrap then result_of K.direct_wrap b
         else result_of (narrow b) 0)
    end
    else if o = K.loop_store && b = 0 && f.stored < 0 then
      f.stored <- result f a
    else raise Not_direct

  (* The word of the direct form of [t], a lean tape (kernels.h,
     NX_LOOP_DIRECT), or [None] where it is none. *)
  let of_tape (t : tape) =
    if t.pro.n > 0 || t.epi.n > 0 || t.body.n > direct_words then None
    else
      let f = { values = Array.make t.pl.top (-1); op = -1; stored = -1 } in
      match
        for k = 0 to t.body.n - 1 do
          form_word f t.body k
        done
      with
      | exception Not_direct -> None
      | () when f.op < 0 || f.stored < 0 -> None
      | () ->
          let how = how_of f.stored and dt = f.stored lsr 5 in
          Some (f.op lor (how lsl K.direct_result) lor (dt lsl K.direct_wrap_dt))

  (* The word of the direct form of [t], a lean reduction's tape whose term is
     plane [term] (kernels.h, NX_LOOP_DIRECT): its one load's element, as it is,
     as a truth or decoded from a narrow float, cast into the term plane; or
     [None]. *)
  let of_term (t : tape) ~term =
    if t.pro.n > 0 || t.epi.n > 0 || t.body.n <> 2 then None
    else
      let w0 = get t.body 0 and w1 = get t.body 1 in
      let op w = Int64.to_int (Int64.logand w 0xffffL) in
      let o1 = op w1 and b1 = field w1 40 and c1 = field w1 52 in
      let how =
        if op w0 <> K.loop_load || field w0 28 > 3 then None
        else if field w1 16 <> term || field w1 28 <> field w0 16 then None
        else if o1 = K.loop_truth then Some K.direct_truth
        else if o1 = K.loop_cast && b1 = c1 then Some K.direct_as_is
        else if o1 = K.loop_decode then
          match narrow b1 with h -> Some h | exception Not_direct -> None
        else None
      in
      Option.map
        (fun how ->
          (1 lsl K.direct_count)
          lor (loaded (field w0 28) how lsl K.direct_operands))
        how
end

(* Casts by name

   A map takes its cast's kernel (direct.metal, cast_map) where its tape loads
   its one operand, casts it, and stores the result: a cast, a decode or an
   encode between its load and its store, into a dtype one of the kernels
   writes. Both operands are plain and the loop's indices 32-bit. The kernel
   computes tape.h's cast with the source's dtype as a constant, or takes its
   bits from a table of every code, so it changes no bit. *)

let cast_kernel ~wide (t : tape) (os : operand array) =
  let plain (o : operand) =
    match o.access with Strided _ -> true | Pad _ -> false
  in
  if wide || t.pro.n > 0 || t.epi.n > 0 || t.body.n <> 3 then None
  else if Array.length os <> 2 || not (Array.for_all plain os) then None
  else
    let w0 = get t.body 0 and w1 = get t.body 1 and w2 = get t.body 2 in
    let op w = Int64.to_int (Int64.logand w 0xffffL) in
    let field = Form.field in
    let o = op w1 and b = field w1 40 and c = field w1 52 in
    if
      op w0 <> K.loop_load
      || field w0 28 <> 0
      || op w2 <> K.loop_store
      || field w2 40 <> 0
      || field w1 28 <> field w0 16
      || field w2 28 <> field w1 16
    then None
    else if o = K.loop_cast then K.cast b c
    else if o = K.loop_decode then K.cast b (D.code D.Float32)
    else if o = K.loop_encode then K.cast (D.code D.Float32) b
    else None

(* Whether a cast's quads over the walk [w] are flat (direct.metal, Casts by
   name): one axis of whole quads, which each operand of [os] walks at stride 1
   from an element of an array whose address is a multiple of its quad's bytes,
   or for 4-bit elements whose bit is a multiple of 16. A flat thread writes its
   whole quad, so a walk whose last quad is short is not flat. The facts hold an
   address modulo 32 ({!Operands.operand}), a quad of 64-bit elements' bytes. *)
let flat w (os : operand array) =
  let on (o : operand) =
    let b = bits o.dtype in
    match (o.buffer, o.access) with
    | Arr x, Strided [| 1 |] when b >= 8 ->
        (x.residue + (o.offset * (b / 8))) land ((b / 2) - 1) = 0
    | Arr _, Strided [| 1 |] when b = 4 ->
        (fst (low_bit o b) + (b * (o.offset - o.lo))) mod 16 = 0
    | _ -> false
  in
  Array.length w.ext = 1 && w.ext.(0) mod 4 = 0 && Array.for_all on os

(* Whether a map of a direct form or a cast takes its kernel: [false] in tests
   that check it against the tape. *)
let direct = Atomic.make true

let bound_direct b =
  Atomic.set direct b;
  Plan.renew ()

(* The direct form of tape [t] over the walk [w] and the operands [all], where
   the direct kernels take it. *)
let direct_form ~wide w all t =
  if
    Atomic.get direct
    && Array.length w.ext <= K.direct_rank
    && lean_loop ~wide t all
  then Form.of_tape t
  else None

(* A thread of quads keeps its coordinates in group memory for its padded loads,
   which read their axes from their descriptors, so that an array of them in
   registers would go to the stack: [coords w] bytes a group. *)
let coords w = threads * 4 * Array.length w.ext

(* Whether a map of [v] elements a thread whose group's planes take [planes]
   bytes takes quads ({!quads}): where it reads a padded load ([padded]), only
   where the coordinates fit the 16 KiB that keeps four groups a core
   ({!elements}); else each element finds its own. A cast's operands may be
   wider or narrower ([cast]). *)
let map_quads ~cast ~padded ~v ~planes w ~loads ~dsts =
  v = K.loop_most
  && ((not padded) || planes + coords w <= 16 * 1024)
  && quads ~cast w ~loads ~dsts

(* The cost of a tile of a map of [v] elements a thread ({!tile_cost}), its
   padded loads' decode included ({!pad_cost}). *)
let map_cost ~dispatch ~v ~planes ~wide ~padded w wops wouts t =
  let c =
    tile_cost ~dispatch ~ext:w.ext ~v wops wouts t.body ~from:0 t.body.n
  in
  if not padded then c
  else
    let bytes, ops, chain = c in
    let quads =
      map_quads ~cast:false ~padded ~v ~planes w ~loads:wops ~dsts:wouts
    in
    (bytes, ops + (v * threads * pad_cost ~v ~quads ~wide w wops), chain)

(* The tape kernel of a map of [v] elements a thread that takes no tiles, as
   {!kernel_of} reads a header of [v], [wide] and [padded]. *)
let map_kernel ~wide ~padded v =
  if wide then K.Loop_map_wide
  else if v <> K.loop_most then K.Loop_map_v1
  else if padded then K.Loop_map_padded
  else K.Loop_map_v4

(* The launches of a map of [p] over [ops] into [outs], on a loop of [shape]:
   tiles of the indices, as many a launch as its budget allows. *)
let map_launches ~holds (p : P.t) ~shape ops outs =
  let p, ops = compact p ~roots:(List.init (Iarray.length p.outs) Fun.id) ops in
  let nl = Array.length ops in
  let needed k = identical outs ops.(k) in
  let merge = not (reads_coord p || Array.exists is_padded ops) in
  let w, all =
    walk ~steps:true shape
      (Array.init (Array.length shape) Fun.id)
      ~kept:0 ~merge (Array.append ops outs)
  in
  let w, all =
    if merge && Array.length w.ext = 0 then one_axis shape all else (w, all)
  in
  let wops = Array.sub all 0 nl
  and wouts = Array.sub all nl (Array.length outs) in
  let n = Array.fold_left ( * ) 1 w.ext in
  let wide = wide n all in
  let build ~eager =
    let t = tape () in
    lower t p
      ~roots:(List.init (Iarray.length p.outs) Fun.id)
      ~load:Fun.id ~axis:(kernel_axis w) ~eager
      ~out:(fun k v -> ins t.body K.loop_store ~d:0 ~a:v.plane ~b:k ~c:0);
    let v = elements ~padded:true ~stores:true ~wide ~ext:w.ext t all in
    (t, v)
  in
  let padded = any_padded wops 0 in
  (* The estimate of a build [t] at [v] elements a thread, its group's planes
     [planes]. *)
  let estimate t ~v ~planes =
    let shared = planes * plane_bytes v in
    let groups = ceil_div n (v * threads) in
    let bytes, ops, chain =
      map_cost ~dispatch ~v ~planes:shared ~wide ~padded w wops wouts t
    in
    let kernel = map_kernel ~wide ~padded v in
    seconds ~groups
      ~resident:(Plan.resident ~granted:holds kernel ~shared)
      { Plan.bytes = groups * bytes; ops = groups * ops; chain }
  in
  (* Reading loads first hides their latency, 4 elements a thread spread the
     dispatch. A map takes the build of the lower estimate: its loads read
     first, or each before its first reader. The two take the same words, so the
     lazy one, which holds fewer planes, can gain only where the eager one's
     planes cut the groups a core holds or keep a lean map from 4 elements a
     thread; only there are both made. The eager one holds at least its loads'
     planes and the lazy one's: where those loads alone pass 4's planes on a
     lean map, the lazy build is made first, and the eager one only where its
     planes, at that least, fit a group and its estimate is the lower; then its
     own planes are estimated, and the lower build taken. *)
  let lean = (not wide) && lean_from ~padded:true all 0 in
  let gains t v =
    let kernel = map_kernel ~wide ~padded v in
    Plan.resident ~granted:holds kernel ~shared:(t.pl.top * plane_bytes v)
    < Plan.resident ~granted:holds kernel ~shared:0
    || (v = 1 && lean_loop ~padded:true ~wide t all)
  in
  (* A cast's kernel is named from the first build, before the choice, which
     would weigh the tape's kernel that a cast does not launch. A tape
     {!cast_kernel} takes loads its one operand and casts it, and {!lower}
     writes loads into the body alone, so its eager and lazy builds are the same
     tape. A longer tape it took would be asked of both. *)
  let cast_of t = if Atomic.get direct then cast_kernel ~wide t all else None in
  let t, v =
    if (not lean) || nl * plane_bytes K.loop_most <= 16 * 1024 then
      match build ~eager:(fun _ -> true) with
      | exception Too_wide -> build ~eager:needed
      | t, v ->
          if
            cast_of t <> None
            || direct_form ~wide w all t <> None
            || not (gains t v)
          then (t, v)
          else
            let t', v' = build ~eager:needed in
            if
              estimate t' ~v:v' ~planes:t'.pl.top
              < estimate t ~v ~planes:t.pl.top
            then (t', v')
            else (t, v)
    else
      let t4, v4 = build ~eager:needed in
      let least = Int.max nl t4.pl.top in
      if cast_of t4 <> None || least * plane_bytes 1 > Plan.group_memory then
        (t4, v4)
      else
        let lazy_estimate = estimate t4 ~v:v4 ~planes:t4.pl.top in
        if lazy_estimate < estimate t4 ~v:1 ~planes:least then (t4, v4)
        else
          let t, v = fitted build ~needed in
          if estimate t ~v ~planes:t.pl.top < lazy_estimate then (t, v)
          else (t4, v4)
  in
  let named = cast_of t in
  let form =
    match named with
    | Some _ -> None
    | None -> (
        match direct_form ~wide w all t with
        | Some f -> Some (placed_form w wops wouts.(0) f)
        | None -> None)
  in
  let mode =
    match (named, form) with
    | Some k, _ -> Cast k
    | None, Some _ -> Direct
    | None, None -> Map
  in
  let fields =
    match form with
    | Some f -> direct_fields wops wouts.(0) f
    | None -> fields wops wouts
  in
  let built = v in
  let map_planes = t.pl.top * plane_bytes built in
  let unstaged staged =
    Array.of_list
      (List.filteri (fun k _ -> not (Array.mem k staged)) (Array.to_list wops))
  in
  (* Sets in [mode]'s header [h], its tiles set, the bits by which its threads
     read [loads] of [shared] bytes of group memory: quads where they take them,
     flat for a cast, a padded map's coordinates and a direct map's form and
     lanes; and is the group memory the coordinates grow it to. The bits name
     the kernel ({!kernel_of}), so that an estimate reads the residency of the
     kernel it would launch. *)
  let finish mode h ~v ~shared ~loads =
    let cast = match mode with Cast _ -> true | _ -> false in
    let quads =
      map_quads ~cast ~padded ~v ~planes:shared w ~loads ~dsts:wouts
    in
    if quads then h.(0) <- h.(0) lor K.loop_quads;
    if quads && cast && flat w all then h.(0) <- h.(0) lor K.loop_flat;
    let shared =
      if quads && padded then begin
        h.(0) <- h.(0) lor K.loop_coords;
        seti h K.loop_coord_at (shared / 4);
        shared + coords w
      end
      else shared
    in
    (match form with
    | Some f ->
        seti h K.loop_direct f;
        if quads && as_is f && all_bits 32 wops 0 && all_bits 32 wouts 0 then
          h.(0) <- h.(0) lor K.loop_bits32
        else if
          quads
          && codec f <> K.direct_as_is
          && all_bits 16 wops 0 && all_bits 16 wouts 0
        then h.(0) <- h.(0) lor K.loop_bits16
        else if quads && bytes_form f && all_bits 8 wops 0 && all_bits 8 wouts 0
        then h.(0) <- h.(0) lor K.loop_bits8
    | None -> ());
    shared
  in
  (* [mode]'s elements a thread, finished header, group memory, groups, cost and
     staged loads, flat or in tiles where their estimate is the lower, and the
     estimate of the one it takes. The direct and cast kernels dispatch no word
     and hold no plane; a cast's binds no group memory and takes no tiles. *)
  let laid mode =
    let v = if mode = Map then built else K.loop_most in
    let planes = if mode = Map then map_planes else 0 in
    let dispatch = if mode = Map then dispatch else 0 in
    let h = header ~wide v in
    if padded then h.(0) <- h.(0) lor K.loop_padded;
    let flat_shared =
      match mode with Map -> planes | Direct -> stub_bytes | _ -> 0
    in
    let flat_groups = ceil_div n (v * threads) in
    let untiled = map_cost ~dispatch ~v ~planes ~wide ~padded w wops wouts t in
    let estimate h shared groups (bytes, ops, chain) =
      let resident = Plan.resident ~granted:holds (kernel_of mode h) ~shared in
      seconds ~groups ~resident
        { Plan.bytes = groups * bytes; ops = groups * ops; chain }
    in
    let tiling =
      match mode with Cast _ -> None | _ -> tiling ~wide ~v ~planes w wops
    in
    (* The tiles' header, copied before the flat one is finished; [h] itself
       where there are none, so that nothing is allocated. *)
    let h' = match tiling with None -> h | Some _ -> Array.copy h in
    let grown = finish mode h ~v ~shared:flat_shared ~loads:wops in
    let flat_estimate = estimate h flat_shared flat_groups untiled in
    let tiled =
      match tiling with
      | None -> None
      | Some tl ->
          let shared = planes + (Array.length tl.staged * stage_bytes) in
          let groups = set_tiling h' tl ~ext:w.ext ~stage:(planes / 4) in
          let grown = finish mode h' ~v ~shared ~loads:(unstaged tl.staged) in
          let c =
            tiled_cost ~dispatch ~ext:w.ext tl wops wouts t.body ~from:0
              t.body.n
          in
          let e = estimate h' shared groups c in
          if e < flat_estimate then Some (tl.staged, grown, groups, c, e)
          else None
    in
    match tiled with
    | Some (staged, grown, groups, c, e) ->
        (v, h', grown, groups, c, Some staged, e)
    | None -> (v, h, grown, flat_groups, untiled, None, flat_estimate)
  in
  (* A cast whose load a map would stage, such as a transposed one, takes the
     tape's tiles where their estimate is below its kernel's: the kernel reads a
     line an element there. A map that takes no tiles ({!tiling}) is not
     estimated, so that the cast waits for no map's kernel. *)
  let mode, (_, h, shared, total, (bytes, ops, chain), staged, _) =
    let l = laid mode in
    match mode with
    | Cast _ when tiling ~wide ~v:built ~planes:map_planes w wops <> None -> (
        let _, _, _, _, _, _, cast_best = l in
        match laid Map with
        | (_, _, _, _, _, Some _, map_best) as m when map_best < cast_best ->
            (Map, m)
        | _ -> (mode, l))
    | _ -> (mode, l)
  in
  let table =
    match form with
    | Some f -> direct_table ?staged w wops wouts.(0) f h
    | None ->
        table ?staged w wops ~ofield:Fun.id wouts ~dfield:(fun k -> nl + k) t h
  in
  let cost g = { Plan.bytes = g * bytes; ops = g * ops; chain } in
  launches ~holds mode ~total ~step:1 ~shared ~fields
    ~tickets:(fun _ _ -> 0)
    ~table ~h ~cost ()

(* [p] with its outputs [ks] alone, in that order. *)
let outputs (p : P.t) ks =
  P.v ~ins:p.ins p.nodes
    ~outs:(Iarray.of_list (List.map (Iarray.get p.outs) ks))

let pick a ks = Array.of_list (List.map (Array.get a) ks)

(* Whether [o] is C-contiguous over [shape]. *)
let contiguous (o : operand) shape =
  match o.access with
  | Strided s ->
      L.is_contiguous
        (L.v ~offset:o.offset ~strides:(Iarray.of_array s)
           (Iarray.of_array shape))
  | Pad _ -> false

(* [p]'s outputs [ks] into [outs] as copies, where each moves the bits of a
   strided operand of a byte or more into a C-contiguous destination, as
   Moves.map takes it. *)
let copies (p : P.t) ~shape ops outs ks =
  let copy k =
    match Moves.moved p k with
    | Some j when contiguous outs.(k) shape -> (
        let src = ops.(j) in
        match src.access with
        | Strided _ when bits src.dtype >= 8 ->
            Some { src; dst = outs.(k); shape }
        | Strided _ | Pad _ -> None)
    | Some _ | None -> None
  in
  let cs = List.filter_map copy ks in
  if List.length cs = List.length ks then Some cs else None

(* The passes of a map of [p]'s outputs [ks] into [outs], none of which is
   identical to a load: copies where each output moves bits, else one pass, else
   the outputs in halves. *)
let rec halves ~holds (p : P.t) ~shape ops outs ks =
  match copies p ~shape ops outs ks with
  | Some cs -> [ { launches = []; work = 0; copies = cs } ]
  | None -> (
      match map_launches ~holds (outputs p ks) ~shape ops (pick outs ks) with
      | ls -> [ { launches = ls; work = 0; copies = [] } ]
      | exception Too_wide when List.length ks > 1 ->
          let h = List.length ks / 2 in
          halves ~holds p ~shape ops outs (List.filteri (fun i _ -> i < h) ks)
          @ halves ~holds p ~shape ops outs
              (List.filteri (fun i _ -> i >= h) ks))

(* A map's passes: one, else its outputs apart, each computed with the whole's
   bits. A destination identical to a load is written only after every launch
   that reads the load: such outputs go last, in one pass, else into temporaries
   copied into them at the end. Each pass's launches read an index's operands
   before they write it, so its launches split by indices keep that. *)
let map_passes ~holds s (p : P.t) ~shape ops outs =
  let all = List.init (Iarray.length p.outs) Fun.id in
  match map_launches ~holds p ~shape ops outs with
  | ls -> [ { launches = ls; work = 0; copies = [] } ]
  | exception Too_wide when List.length all > 1 ->
      let read k = Array.exists (identical [| outs.(k) |]) ops in
      let aliased, apart = List.partition read all in
      let first =
        if apart = [] then [] else halves ~holds p ~shape ops outs apart
      in
      if aliased = [] then first
      else
        let last =
          match
            map_launches ~holds (outputs p aliased) ~shape ops
              (pick outs aliased)
          with
          | ls -> [ { launches = ls; work = 0; copies = [] } ]
          | exception Too_wide ->
              let dts =
                List.map (fun k -> P.dtype p (Iarray.get p.outs k)) aliased
              in
              let ts = fresh s (List.map (fun d -> (d, shape)) dts) in
              let m = Array.length ts in
              let copy =
                P.v ~ins:(Iarray.of_list dts)
                  (Iarray.init m (fun k -> P.In k))
                  ~outs:(Iarray.init m Fun.id)
              in
              let ks = List.init m Fun.id in
              halves ~holds (outputs p aliased) ~shape ops ts ks
              @ halves ~holds copy ~shape ts (pick outs aliased) ks
        in
        first @ last

(* Parts

   A program too wide for a launch, past a launch's buffers or a group's 32 KiB
   of threadgroup memory for its values held at once, runs in parts. A part maps
   the values that its nodes before a cut pass to later ones into temporaries in
   a buffer of its own, each over the loop's axes along which its node varies,
   and the rest of the program reads them as loads, at stride 0 along the other
   axes. A node's value is an element of its dtype, so a temporary holds it
   exactly and the parts give one launch's bits. *)

let computed : P.node -> bool = function
  | P.In _ | Coord _ | Const _ -> false
  | Op1 _ | Op2 _ | Op3 _ -> true

(* Whether each node of [p] is one its outputs read. *)
let live_nodes (p : P.t) =
  let n = Iarray.length p.nodes in
  let live = Array.make n false in
  Iarray.iter (fun o -> live.(o) <- true) p.outs;
  for i = n - 1 downto 0 do
    if live.(i) then
      List.iter (fun a -> live.(a) <- true) (operands_of (Iarray.get p.nodes i))
  done;
  live

(* The computed nodes before [c] that a live node from [c] on or an output
   reads, in order. *)
let crossing (p : P.t) ~live c =
  let read = Array.make c false in
  let mark i =
    if i < c && computed (Iarray.get p.nodes i) then read.(i) <- true
  in
  for j = c to Iarray.length p.nodes - 1 do
    if live.(j) then List.iter mark (operands_of (Iarray.get p.nodes j))
  done;
  Iarray.iter mark p.outs;
  List.filter (fun i -> read.(i)) (List.init c Fun.id)

(* The extents of each node of [p] over [ops] on a loop of [shape]: the loop's
   along the axes its leaves vary along, a load's of more than one element and a
   nonzero stride, a padded load's or a coordinate's, and 1 along the others. *)
let extents (p : P.t) ops shape =
  let ext = Array.make (Iarray.length p.nodes) [||] in
  let only f = Array.mapi (fun a e -> if f a then e else 1) shape in
  Iarray.iteri
    (fun i nd ->
      ext.(i) <-
        (match nd with
        | P.In k -> (
            match ops.(k).access with
            | Strided s -> only (fun a -> s.(a) <> 0)
            | Pad _ -> shape)
        | Coord c -> only (( = ) c)
        | Const _ -> only (fun _ -> false)
        | Op1 _ | Op2 _ | Op3 _ ->
            only (fun a ->
                List.exists (fun o -> ext.(o).(a) > 1) (operands_of nd))))
    p.nodes;
  ext

(* The first part of [p] over [ops] on a loop of [shape], its temporaries in
   stand-ins of [s], and the rest: the largest cut whose first part one launch
   maps, else the widest load copied into a temporary. [None] where neither fits
   a launch. *)
let part ~holds s (p : P.t) ~shape ops =
  let ext = extents p ops shape in
  let temp i = (P.dtype p i, ext.(i)) in
  let live = live_nodes p in
  let first c =
    let xs = crossing p ~live c in
    let p1 =
      P.v ~ins:p.ins
        (Iarray.sub p.nodes ~pos:0 ~len:c)
        ~outs:(Iarray.of_list xs)
    in
    (p1, xs)
  in
  let fits c =
    let p1, xs = first c in
    let ts, _ = temporaries (Temp (-1)) (List.map temp xs) in
    match map_launches ~holds p1 ~shape ops ts with
    | _ -> true
    | exception Too_wide -> false
  in
  (* A cut after each live computed node, each part cut there computing the live
     nodes before it, so that the rest computes fewer. *)
  let cuts =
    Array.of_list
      (List.filter_map
         (fun i ->
           if live.(i) && computed (Iarray.get p.nodes i) then Some (i + 1)
           else None)
         (List.init (Iarray.length p.nodes) Fun.id))
  in
  (* The last cut that fits, by bisection: the first fits, the last does not. *)
  let rec last lo hi =
    if hi - lo <= 1 then lo
    else
      let m = (lo + hi) / 2 in
      if fits cuts.(m) then last m hi else last lo m
  in
  if Array.length cuts > 0 && fits cuts.(0) then begin
    let c = cuts.(last 0 (Array.length cuts)) in
    let p1, xs = first c in
    let ts = fresh s (List.map temp xs) in
    let nins = Iarray.length p.ins in
    let index = Hashtbl.create 8 in
    List.iteri (fun j i -> Hashtbl.add index i (nins + j)) xs;
    let nodes =
      Iarray.mapi
        (fun i nd ->
          match Hashtbl.find_opt index i with Some k -> P.In k | None -> nd)
        p.nodes
    in
    let p2 =
      P.v
        ~ins:(Iarray.append p.ins (Iarray.of_list (List.map (P.dtype p) xs)))
        nodes ~outs:p.outs
    in
    let ls = map_launches ~holds p1 ~shape ops ts in
    Some ({ launches = ls; work = 0; copies = [] }, p2, Array.append ops ts)
  end
  else
    (* The widest padded operand, by its descriptor's words: a plain one copied
       reads as many. *)
    let size (o : operand) =
      match o.access with
      | Strided s -> 2 + Array.length s
      | Pad { reads; _ } -> 5 + (K.loop_pad * Array.length reads)
    in
    let widest = ref (-1) in
    Array.iteri
      (fun k (o : operand) ->
        let plain = size o <= 2 + Array.length shape in
        if (not plain) && (!widest < 0 || size o > size ops.(!widest)) then
          widest := k)
      ops;
    if !widest < 0 then None
    else
      let k = !widest in
      let dt = ops.(k).dtype in
      let ts = fresh s [ (dt, shape) ] in
      let copy =
        P.v ~ins:(Iarray.of_list [ dt ])
          (Iarray.of_list [ P.In 0 ])
          ~outs:(Iarray.of_list [ 0 ])
      in
      match map_launches ~holds copy ~shape [| ops.(k) |] ts with
      | ls ->
          let ops = Array.copy ops in
          ops.(k) <- ts.(0);
          Some ({ launches = ls; work = 0; copies = [] }, p, ops)
      | exception Too_wide -> None

(* The passes of [p] over [ops], [final]'s where they fit a launch, else in
   parts, then closed by [close] (as {!plan}), their temporaries numbered after
   [fixed]'s; [Error Declined] where a kind computes at a dtype whose values
   only move or a launch's least work exceeds the budget. Each part leaves fewer
   live computed nodes, or as many and one fewer padded load, so the parts
   end. *)
let parted ~holds ?(fixed = stand_ins ()) ~close (p : P.t) ~shape ops final =
  let cut () =
    let s = { sizes = fixed.sizes; count = fixed.count } in
    let rec go p ops parts =
      match final ~holds s p ops with
      | cs -> close s (List.rev_append parts cs)
      | exception Too_wide -> (
          match part ~holds s p ~shape ops with
          | None -> raise Declined
          | Some (c, p, ops) -> go p ops (c :: parts))
    in
    go p ops []
  in
  match cut () with
  | r -> Ok r
  | exception (Declined | Over_budget) -> Error A.Declined

(* A call of no element answers as [f ()], the call of one, would
   ({!Operands.ones}): [Ok None] where its kinds compute, [Error Declined] where
   they do not. Its launches are not the call's, so no trace sees them. *)
let empty f = match Plan.unseen f with Ok _ -> Ok None | Error _ as e -> e
let shape_of (o : Operands.operand) = Iarray.to_array (L.shape o.layout)

let rec map ~holds (p : P.t) (xs : Operands.t) =
  let shape = shape_of xs.dsts.(0) in
  if Array.exists (( = ) 0) shape then
    empty (fun () -> map ~holds p (Operands.ones xs))
  else
    let outs = Array.map plain xs.dsts in
    parted ~holds ~close:plan p ~shape (Array.map of_load xs.loads)
      (fun ~holds s p ops -> map_passes ~holds s p ~shape ops outs)

(* Reductions and scans *)

(* A reduction as loop.metal folds it. *)
let reduction_code : Nx_kernel.reduction -> int = function
  | Monoid Sum -> K.loop_sum
  | Monoid Prod -> K.loop_prod
  | Monoid Max -> K.loop_max
  | Monoid Min -> K.loop_min
  | Arg Max -> K.loop_arg_max
  | Arg Min -> K.loop_arg_min
  | Logsumexp -> K.loop_logsumexp
  | Moments -> K.loop_moments

(* A reduction's pass: its fold, what it does with each output's state, and
   whether its terms read a center. *)
type fold = { rk : int; fin : int; centered : bool }

let folds (r : Nx_kernel.reduction) =
  match r with
  | Logsumexp ->
      [
        { rk = K.loop_max; fin = K.loop_put_center; centered = false };
        { rk = K.loop_exps; fin = K.loop_put_logsumexp; centered = true };
      ]
  | Moments ->
      [
        { rk = K.loop_sum; fin = K.loop_put_mean; centered = false };
        { rk = K.loop_squares; fin = K.loop_put_moments; centered = true };
      ]
  | r -> [ { rk = reduction_code r; fin = K.loop_put; centered = false } ]

(* The carrier of output [k] of [p], which a reduction of [r] computes in:
   declined where its values only move, and for Logsumexp and Moments but in
   float32. *)
let carrier_of (p : P.t) k (r : Nx_kernel.reduction) =
  let c = Nx_kernel.carrier (P.dtype p (Iarray.get p.outs k)) in
  if moves_only c then raise Declined;
  (match r with
  | Logsumexp | Moments -> if c <> float32 then raise Declined
  | Monoid _ | Arg _ -> ());
  c

(* Fixed planes of a reduction or a scan: the term, the results, the center. *)
type fixed = { term : int; result : int; center : int }

let fixed t ~carrier ~center ~result =
  let term = alloc t.pl (words carrier) in
  let result = if result then alloc t.pl 6 else 0 in
  let center = if center then alloc t.pl 2 else 0 in
  { term; result; center }

(* Lowers output [k] of [p] into [t]'s body, its value cast into the term plane
   in [carrier], a boolean as true or false. *)
let lower_term t p k ~carrier ~axis ~eager ~(f : fixed) =
  lower t p ~roots:[ k ] ~load:Fun.id ~axis ~eager ~out:(fun _ v ->
      if boolean carrier then
        ins t.body K.loop_truth ~d:f.term ~a:v.plane ~b:0 ~c:0
      else cast t.body ~d:f.term ~a:v.plane ~from:v.dt ~into:carrier)

let reduction_word ~rk ~ct ~carrier ~fin =
  rk lor (ct lsl 8) lor (code carrier lsl 16) lor (fin lsl 24)

(* The bytes of a state in the workspace. *)
let state_bytes = 4 * K.loop_state

(* The words of a reduction's scratch, over its planes once its runs are done:
   its 16 slots' states and a flag. *)
let scratch_words = (16 * K.loop_state) + 1

(* The levels of a pairwise tree over [n] states. *)
let levels_of n =
  let rec go n l = if n <= 1 then l else go ((n + 1) / 2) (l + 1) in
  go n 0

(* The destinations of a reduction or a scan's outputs: C-contiguous, an
   output's element at its first position plus its number. *)
let outputs_at (o : operand) = { o with access = Strided [||] }

(* The center of each output, a float32 in the workspace from byte [byte]: one
   an output, C-contiguous over the walk [w]'s kept axes, the same along the
   reduced ones. *)
let center_load w ~byte =
  let n = Array.length w.ext in
  let s = Array.make n 0 in
  let acc = ref 1 in
  for i = w.kept - 1 downto 0 do
    s.(i) <- !acc;
    acc := !acc * w.ext.(i)
  done;
  let offset = byte / 4 in
  {
    buffer = Space;
    dtype = float32;
    offset;
    lo = offset;
    hi = offset + !acc;
    access = Strided s;
  }

(* The seconds of [total] groups of [steps] steps each, in batches of [batch]
   groups, each in pieces of [rn] steps, by [time n r] for [n] groups of [r]
   steps. *)
let batched ~time ~total ~steps ~batch ~rn =
  let pieces = ceil_div steps rn in
  let batch_time n =
    (float (pieces - 1) *. time n rn) +. time n (steps - ((pieces - 1) * rn))
  in
  let rest = total mod batch in
  (float (total / batch) *. batch_time batch)
  +. if rest > 0 then batch_time rest else 0.

(* The launches of [mode] over [total] groups, each of [steps] steps in order (a
   block's runs, a slice's chunk or block totals), [cost r n] for [n] groups of
   [r] steps. Where a group's steps fit a launch, they may be {!launches} of all
   its steps, the most groups that fit. Else, or where that takes longer, groups
   go in batches of the most that fit with one step each, each batch's steps in
   pieces of the most that fit, a piece's first step [first r] in [h]'s word
   [word], by default NX_LOOP_SEGMENT, and its steps in NX_LOOP_RUNS; a unit
   leaves its state in workspace for the batch's next piece, which is one
   launch. The faster of the two by the estimate: all steps a group can hold
   less than a wave, which splitting fills. Also whether a unit's steps took
   several pieces. Raises [Over_budget] where one step of one group exceeds a
   launch. *)
let stepped ?(word = K.loop_segment) ~holds mode ~total ~steps ~first ~shared
    ~fields ~table ~h ~cost () =
  let resident = Plan.resident ~granted:holds (kernel_of mode h) ~shared in
  let time n r = seconds ~groups:n ~resident (cost r n) in
  let fits n r = time n r <= Plan.allowed () in
  let whole () =
    seti h word (first 0);
    seti h K.loop_runs steps;
    ( launches ~holds mode ~total ~step:1 ~shared ~fields ~table ~h
        ~cost:(cost steps)
        ~tickets:(fun _ _ -> 0)
        (),
      false )
  in
  let all_steps =
    if fits 1 steps then
      Some (most_groups ~total ~step:1 ~resident (cost steps))
    else None
  in
  match all_steps with
  | Some n when n >= total -> whole ()
  | _ -> (
      let batch = most_groups ~total ~step:1 ~resident (cost 1) in
      (* The most steps that fit [batch] groups, by bisection: [lo] fits, [hi]
         does not. *)
      let rec go lo hi =
        if hi - lo <= 1 then lo
        else
          let m = (lo + hi) / 2 in
          if fits batch m then go m hi else go lo m
      in
      let rn = if fits batch steps then steps else go 1 steps in
      let batched = batched ~time ~total ~steps in
      match all_steps with
      | Some n when batched ~batch:n ~rn:steps <= batched ~batch ~rn -> whole ()
      | _ ->
          let pieces = ceil_div steps rn in
          let rec batches at acc =
            if at >= total then List.concat (List.rev acc)
            else
              let n = Int.min batch (total - at) in
              let piece c =
                let h = Array.copy h in
                let r = Int.min rn (steps - (c * rn)) in
                seti h word (first (c * rn));
                seti h K.loop_runs r;
                launches ~first:at ~holds mode ~total:(at + n) ~most:n ~step:n
                  ~shared ~fields ~table ~h ~cost:(cost r)
                  ~tickets:(fun _ _ -> 0)
                  ()
              in
              batches (at + n)
                (List.concat_map piece (List.init pieces Fun.id) :: acc)
          in
          (batches 0 [], pieces > 1))

(* loop_merge's launches, a group for each of [outputs], each merging its
   output's [m] states into [put_bytes] of results. *)
let merge_launches ~holds ~outputs ~m ~put_bytes ~fields ~table h =
  let cost n =
    {
      Plan.bytes = n * ((m * state_bytes) + put_bytes);
      ops = n * m * fold_ops;
      chain = levels_of m * ((2 * Plan.read_links) + fold_links);
    }
  in
  launches ~holds Merge ~total:outputs ~step:1 ~shared:0 ~fields ~table ~h ~cost
    ~tickets:(fun _ _ -> 0)
    ()

(* The folds of a direct reduction: built once, since a list of another module's
   values is built where it is written. *)
let direct_folds =
  [
    K.loop_sum;
    K.loop_max;
    K.loop_min;
    K.loop_arg_max;
    K.loop_arg_min;
    K.loop_exps;
    K.loop_squares;
  ]

(* The kernel of a row (kernels.h, NX_METAL_ROW_KERNELS) that a direct reduction
   [rk] at compute type [ct], its term's element taken by [how], over a walk of
   [axes] reduced axes takes: one where it computes in float32, takes a float32
   element as it is or decodes a bfloat16 or float16 one, and reduces one axis,
   along which each lane's terms step; else the generic direct kernel. A
   Logsumexp's or Moments' second pass takes its row's kernel where its element
   is a float32's as it is, else loop_reduce_centered. Each mode is a constant,
   so planning allocates none. *)
let row_kernel ~rk ~ct ~how ~axes =
  let pick ~f32 ~bf16 ~f16 =
    if how = K.direct_as_is then f32
    else if how = K.direct_bf16 then bf16
    else if how = K.direct_f16 then f16
    else Reduce_direct
  in
  let row = ct = K.loop_f32 && axes = 1 in
  if rk = K.loop_exps || rk = K.loop_squares then
    if not (row && how = K.direct_as_is) then Reduce_centered
    else if rk = K.loop_exps then Reduce_row K.Loop_row_logsumexp_f32
    else Reduce_row K.Loop_row_moments_f32
  else if not row then Reduce_direct
  else if rk = K.loop_sum then
    pick ~f32:(Reduce_row K.Loop_row_sum_f32)
      ~bf16:(Reduce_row K.Loop_row_sum_bf16)
      ~f16:(Reduce_row K.Loop_row_sum_f16)
  else if rk = K.loop_max then
    pick ~f32:(Reduce_row K.Loop_row_max_f32)
      ~bf16:(Reduce_row K.Loop_row_max_bf16)
      ~f16:(Reduce_row K.Loop_row_max_f16)
  else if rk = K.loop_min then
    pick ~f32:(Reduce_row K.Loop_row_min_f32)
      ~bf16:(Reduce_row K.Loop_row_min_bf16)
      ~f16:(Reduce_row K.Loop_row_min_f16)
  else if rk = K.loop_arg_max then
    pick ~f32:(Reduce_row K.Loop_row_argmax_f32)
      ~bf16:(Reduce_row K.Loop_row_argmax_bf16)
      ~f16:(Reduce_row K.Loop_row_argmax_f16)
  else if rk = K.loop_arg_min then
    pick ~f32:(Reduce_row K.Loop_row_argmin_f32)
      ~bf16:(Reduce_row K.Loop_row_argmin_bf16)
      ~f16:(Reduce_row K.Loop_row_argmin_f16)
  else Reduce_direct

(* A row's kernel's work on a term of a lane, at its slowest (loop.metal,
   row_terms), as its passes, its operations and its links: a read every 8
   terms, the fewest a lane reads at once, then its address one step on, its
   decode and a fold by a plain addition, or a comparison and a select; then,
   where a NaN or a zero could make the lanes' bits differ from the tape's, the
   term read again, one at a time, and folded by the tape's. A float16 term
   takes the tape's fold alone, its decode a few cheap kinds. A lane's first
   place, which the cost leaves out, is within the 4 terms a run it charges a
   lane of one term. On the M1 Max, launches over terms all NaN, the data that
   refolds every lane, took at most 0.80 of the estimate at its rates (340 GB/s,
   32 cores at 1.296 GHz): all 15 kernels over rows of 8192 terms, 3 of them
   over rows of 4, 2 over walks of 3 axes, and 7 at a line a term. After the GPU
   idled they took at most 1.79 of it, on rows of 4 terms: within the margin
   plan.ml keeps for a launch after idle. *)
let row_ahead = 8
let plain_links = 2
let row_passes ~how = if how = K.direct_f16 then 1 else 2

let decode_ops ~how =
  if how = K.direct_as_is then 0
  else if how = K.direct_bf16 then cheap
  else 4 * cheap

let row_ops ~how =
  let decode = decode_ops ~how in
  if how = K.direct_f16 then cheap + decode + fold_ops
  else cheap + decode + (2 * cheap) + (cheap + decode + fold_ops)

let row_links ~how =
  let read = Plan.read_links / row_ahead in
  if how = K.direct_f16 then read + fold_links
  else read + plain_links + Plan.read_links + fold_links

(* A direct Logsumexp's or Moments' second pass's work on a term, as a row's
   (loop.metal, centered_run): a read every 8 terms, a decode, the term's
   deviation and its exp, or its square, and plain adds; then the term read
   again and folded by the tape's. A float16 term's fold is the tape's alone. A
   term's exp or square depends on its read alone, not on the sums, so its links
   count once for the 8 terms a lane reads at once. The refold takes a term at a
   time, its read issued a term ahead: a term's links are its exp's or square's
   and the fold's, or the read's where that is longer. The exp is fold.h's [ex],
   whose clamps, reduction, polynomial and scale take about 24 operations.

   On the M1 Max, calls over 1024^2 and 8192^2 terms all NaN, or one NaN a lane,
   the data that refolds every lane, took at most 0.48 of the estimate at its
   rates (340 GB/s, 32 cores at 1.296 GHz), host times that hold the call's
   submission. *)
let centered_ahead = 8
let centered_passes = row_passes
let centered_exp = 4 * cheap

let centered_term ~rk =
  if rk = K.loop_exps then cheap + centered_exp else 3 * cheap

let centered_ops ~how ~rk =
  let decode = decode_ops ~how and term = centered_term ~rk in
  let kept = cheap + decode + term + fold_ops in
  if how = K.direct_f16 then kept
  else cheap + decode + term + (2 * cheap) + kept

let centered_links ~how ~rk =
  let read = Plan.read_links / centered_ahead in
  let term = chain_of_weight (centered_term ~rk) in
  let ahead = term / centered_ahead in
  if how = K.direct_f16 then read + ahead + fold_links
  else read + ahead + plain_links + Int.max Plan.read_links (term + fold_links)

(* The bytes a group's [per] consecutive outputs put, by {!run_bytes}'s rule:
   its share of the run of the [outputs] outputs into each of [dsts], which are
   C-contiguous, so that consecutive outputs lie side by side whatever axes the
   walk dropped or merged; or, for a scan's chunk totals and a Logsumexp's
   centers, which the workspace holds in order, a run of [per] states or
   float32s. *)
let put_cost ~fin ~per ~outputs (dsts : operand array) =
  let run n width = run_bytes ~ext:[| n |] ~rank:1 ~e:per ~width [| 1 |] in
  if fin = K.loop_put_total then run per state_bytes
  else if fin = K.loop_put_center then run per 4
  else
    Array.fold_left
      (fun t (o : operand) -> t + run outputs (Int.max 1 (bits o.dtype / 8)))
      0 dsts

(* A reduction's launches over the walk [w]: [rows] rows of [terms] terms, cut
   into segments of [seg], of output [k] of [p], by [fold], into [dsts]. Fields:
   the loads, the destinations, the workspace, the tickets; a centered fold
   reads its centers through [center], a load of the workspace. Its partial
   states start at byte [partials] of its pass's workspace; it also gives the
   byte their end.

   An output of several groups takes a ticket where its groups fit one launch;
   else its groups leave their states apart, and loop_merge's launches, a group
   an output, merge them. A group whose runs pass a launch takes them in pieces,
   its threads' states between them past the partial states.

   Where [once], [fold] is a Logsumexp's or Moments' second pass, which folds
   each output's center first in the same launch (kernels.h, NX_LOOP_ONCE): it
   raises [Not_once] unless the pass is direct, each output's groups are one and
   a group fits a launch. *)
exception Not_once

let reduce_launches ?(once = false) ~holds (p : P.t) w ops dsts ~k ~carrier
    ~fold ~rows ~terms ~seg ~partials ~centers ~center () =
  let ops =
    match center with
    | Some c when fold.centered && not once -> Array.append ops [| c |]
    | _ -> ops
  in
  let nl = Array.length ops in
  let segs = ceil_div terms seg in
  let outputs = rows * segs in
  let len = Int.min seg terms in
  let nb = ceil_div len K.loop_block in
  let scratch = align16 (4 * scratch_words) in
  let wide = wide (rows * terms) ops in
  let lean =
    lean_stores ~carrier
      ~arg:(fold.rk = K.loop_arg_max || fold.rk = K.loop_arg_min)
      dsts
  in
  (* The term's direct form is read before the center's load, which the tape
     alone runs. *)
  let build ~eager =
    let t = tape () in
    let f =
      fixed t ~carrier ~center:(fold.centered && not once) ~result:false
    in
    lower_term t p k ~carrier ~axis:(kernel_axis w) ~eager ~f;
    let term = Form.of_term t ~term:f.term in
    if fold.centered && not once then
      ins t.body K.loop_load ~d:f.center ~a:(nl - 1) ~b:0 ~c:0;
    let v = elements ~stores:lean ~wide ~ext:w.ext t ops in
    (t, f, v, term)
  in
  let t, f, v, term = fitted build ~needed:(fun _ -> false) in
  (* A Sum, Max, Min or Arg of one load at 4 elements a thread takes
     loop_reduce_direct where its tape is a direct form ({!Form.of_term}), which
     reads the tape's words alone, or its row's kernel where {!row_kernel} gives
     one; a Logsumexp's or Moments' second pass loop_reduce_centered or its
     row's. *)
  let form =
    if Atomic.get direct && v = K.loop_most && List.mem fold.rk direct_folds
    then term
    else None
  in
  if once && form = None then raise Not_once;
  let how_of f = Form.how_of (f lsr K.direct_operands) in
  let mode =
    match form with
    | None -> Reduce
    | Some f ->
        row_kernel ~rk:fold.rk ~ct:(compute carrier) ~how:(how_of f)
          ~axes:(Array.length w.ext - w.kept)
  in
  (* A unit is [g] blocks of an output, a power of two up to its blocks; an
     output of more than 16 blocks takes [m] groups. An output of at most half a
     slot's lanes of terms spans the least power of two lanes at or past them,
     and a slot holds [pack] outputs. *)
  let rec pow2 n x = if x >= n then x else pow2 n (2 * x) in
  let g = Int.min K.loop_lanes (pow2 nb 1) in
  let m = ceil_div nb g in
  let span = if len <= K.loop_lanes / 2 then pow2 len 1 else K.loop_lanes in
  let pack = K.loop_lanes / span in
  let per = K.loop_lanes / g * pack in
  let runs = ceil_div (Int.min len K.loop_block) (K.loop_lanes * v) in
  let fields = fields ~shared:true ops (Array.map outputs_at dsts) in
  let nf = Array.length fields in
  let h = header ~wide v in
  if once then h.(0) <- h.(0) lor K.loop_once;
  seti h K.loop_term f.term;
  seti h K.loop_center f.center;
  seti h K.loop_reduction
    (reduction_word ~rk:fold.rk ~ct:(compute carrier) ~carrier ~fin:fold.fin);
  seti h K.loop_terms terms;
  seti h K.loop_segment seg;
  seti h K.loop_segments segs;
  seti h K.loop_outputs outputs;
  seti h K.loop_group g;
  seti h K.loop_groups m;
  seti h K.loop_span span;
  seti h K.loop_runs runs;
  seti h K.loop_from 0;
  seti h K.loop_work (nf - 2);
  seti h K.loop_tickets (nf - 1);
  seti h K.loop_partials partials;
  seti h K.loop_totals 0;
  seti h K.loop_centers centers;
  (match form with Some f -> seti h K.loop_direct f | None -> ());
  let table =
    table w ops ~ofield:Fun.id
      (Array.map outputs_at dsts)
      ~dfield:(fun j -> nl + j)
      t h
  in
  (* The scratch takes the planes' memory once the runs are done; the direct
     kernel holds no plane and dispatches no word. *)
  let shared =
    if form = None then Int.max (t.pl.top * plane_bytes v) scratch else scratch
  in
  let dispatch = if form = None then dispatch else 0 in
  (* A group's run: [g] blocks of its [per] outputs side by side, each output's
     next 16 v terms or all it has. Its bytes are a share of its [runs] runs'
     over the box of those outputs by their blocks' terms, which follow one
     another (reduce_mode); its work and chain are 16 v terms of each slot's
     lanes, so that a lane of fewer terms pays for its first place. A group's
     chain is its runs', then its lanes' and blocks' trees and its finish. *)
  let tile_bytes =
    let b =
      box ~ext:w.ext ~kept:w.kept ~outputs:per
        ~terms:(Int.min len (g * K.loop_block))
    in
    ceil_div (box_traffics ~ext:w.ext b ops) runs
  in
  let tile_ops, tile_chain =
    let o, c =
      tape_cost ~dispatch ~rank:(Array.length w.ext) t.body ~from:0 t.body.n
    in
    (threads * v * o, v * c)
  in
  (* A run's bytes, operations and chain, a row's kernel's by its terms' work
     ({!row_ops}), a second pass's by {!centered_ops}, after its first's where
     [once]. *)
  let row = match mode with Reduce_row _ -> true | _ -> false in
  let how = match form with Some f -> how_of f | None -> K.direct_as_is in
  let centered =
    fold.centered && Option.is_some form
    && match mode with Reduce_row _ | Reduce_centered -> true | _ -> false
  in
  let plain_bytes = if row then row_passes ~how * tile_bytes else tile_bytes in
  let plain_ops =
    if row then threads * v * row_ops ~how
    else tile_ops + (threads * v * fold_ops)
  in
  let plain_chain =
    if row then v * row_links ~how else tile_chain + (v * fold_links)
  in
  (* A centered fold runs its first pass where [once], then its second, which
     reads the tile again but in a row's kernel. [how] holds only where [form]
     does, as [row] and [centered] do. *)
  let run_bytes =
    if not centered then plain_bytes
    else (if once then plain_bytes else 0) + (centered_passes ~how * tile_bytes)
  in
  let run_ops =
    if not centered then plain_ops
    else
      (if once then plain_ops else 0)
      + (if row then 0 else tile_ops)
      + (threads * v * centered_ops ~how ~rk:fold.rk)
  in
  let run_chain =
    if not centered then plain_chain
    else
      (if once then plain_chain else 0)
      + (if row then 0 else tile_chain)
      + (v * centered_links ~how ~rk:fold.rk)
  in
  (* A merge's group puts one output; a group of runs puts its [per] units'
     outputs. *)
  let put_bytes = Array.length dsts * Plan.line * pack in
  let puts = put_cost ~fin:fold.fin ~per ~outputs dsts in
  (* [n] groups of [rn] runs each; where [once], the center's trees too. *)
  let trees = if once then 2 else 1 in
  let group_cost ~rn ~merges n =
    {
      Plan.bytes = n * ((rn * run_bytes) + puts + (2 * state_bytes));
      ops = n * rn * run_ops;
      chain =
        ((rn * run_chain)
        + (trees * levels_of span * (fold_links + 8))
        + (trees * levels_of g * (fold_links + 8))
        + (2 * Plan.read_links)
        +
        if merges then levels_of m * ((2 * Plan.read_links) + fold_links) else 0
        );
    }
  in
  let total = ceil_div (outputs * m) per in
  let resident = Plan.resident ~granted:holds (kernel_of mode h) ~shared in
  let one_fits =
    seconds ~groups:1 ~resident (group_cost ~rn:runs ~merges:false 1)
    <= Plan.allowed ()
  in
  let ticketed =
    one_fits
    && (m = 1
       ||
       match
         most_groups ~total ~step:m ~resident (group_cost ~rn:runs ~merges:true)
       with
       | _ -> true
       | exception Over_budget -> false)
  in
  let partials_end =
    if m > 1 then partials + (outputs * m * state_bytes) else partials
  in
  if once && not (m = 1 && ticketed) then raise Not_once;
  if ticketed then
    ( launches ~holds mode ~total
        ~step:(if m > 1 then m else 1)
        ~shared ~fields ~table ~h
        ~cost:(group_cost ~rn:runs ~merges:(m > 1))
        ~tickets:(fun _ n -> if m > 1 then n / m else 0)
        (),
      partials_end )
  else if one_fits then begin
    (* Apart: the groups leave their states, then loop_merge's groups merge an
       output's [m] each. *)
    h.(0) <- h.(0) lor K.loop_apart;
    let parts =
      launches ~holds mode ~total ~step:1 ~shared ~fields ~table ~h
        ~cost:(group_cost ~rn:runs ~merges:false)
        ~tickets:(fun _ _ -> 0)
        ()
    in
    ( parts @ merge_launches ~holds ~outputs ~m ~put_bytes ~fields ~table h,
      partials_end )
  end
  else begin
    (* A group's runs past a launch: in pieces, each thread's state between them
       in the workspace past the parts, then apart where an output has several
       groups. *)
    h.(0) <- h.(0) lor K.loop_pieces;
    if m > 1 then h.(0) <- h.(0) lor K.loop_apart;
    let parts, _ =
      stepped ~word:K.loop_from ~holds mode ~total ~steps:runs ~first:Fun.id
        ~shared ~fields ~table ~h
        ~cost:(fun rn -> group_cost ~rn ~merges:false)
        ()
    in
    let held = List.fold_left (fun n l -> Int.max n l.groups) 0 parts in
    ( (if m > 1 then
         parts @ merge_launches ~holds ~outputs ~m ~put_bytes ~fields ~table h
       else parts),
      partials_end + (held * threads * state_bytes) )
  end

let order_of rank ~reduced =
  let kept =
    List.filter (fun a -> not (List.mem a reduced)) (List.init rank Fun.id)
  in
  (Array.of_list (kept @ reduced), List.length kept)

let destinations : Nx_kernel.reduction -> int = function
  | Arg _ | Moments -> 2
  | Monoid _ | Logsumexp -> 1

(* A Logsumexp's or Moments' second pass with its first in its launch
   (reduce_launches' [once]). *)
let once_fold : Nx_kernel.reduction -> fold option = function
  | Logsumexp ->
      Some { rk = K.loop_exps; fin = K.loop_put_logsumexp; centered = true }
  | Moments ->
      Some { rk = K.loop_squares; fin = K.loop_put_moments; centered = true }
  | Monoid _ | Arg _ -> None

(* One pass of the reductions [rs] of [p] over [ops] into [outs], in its turn
   each. A Logsumexp or Moments runs in one launch where it can, whose outputs'
   blocks fit a group; else its centers pass, then its second. Its workspace
   holds each output's center, a float32, where a reduction's second pass reads
   it, then the partial states. *)
let reduce_pass ~holds (p : P.t) ~shape ~axes rs ops outs =
  let rank = Array.length shape in
  let p, ops = compact p ~roots:(List.map snd (Array.to_list rs)) ops in
  let order, kept = order_of rank ~reduced:(Iarray.to_list axes) in
  let merge = not (reads_coord p || Array.exists is_padded ops) in
  let w, wops = walk shape order ~kept ~merge ops in
  let rows = product w.ext 0 w.kept in
  let terms = product w.ext w.kept (Array.length w.ext) in
  let firsts = Array.make (Array.length rs) 0 in
  for i = 1 to Array.length rs - 1 do
    firsts.(i) <- firsts.(i - 1) + destinations (fst rs.(i - 1))
  done;
  (* An output of at most a group's 16 slots of blocks (NX_LOOP_LANES counts
     both a slot's lanes and a group's slots). *)
  let few = ceil_div terms K.loop_block <= K.loop_lanes in
  (* Each reduction's launches in one launch where it takes them; a centered one
     that does not needs the centers' workspace. Loops, so that a plan's
     reductions allocate no closure. *)
  let n = Array.length rs in
  let onces = Array.make n None in
  let centered = ref false in
  for i = 0 to n - 1 do
    let r, k = rs.(i) in
    match once_fold r with
    | Some fold when few -> (
        let dsts = Array.sub outs firsts.(i) (destinations r) in
        match
          reduce_launches ~once:true ~holds p w wops dsts ~k
            ~carrier:(carrier_of p k r) ~fold ~rows ~terms ~seg:terms
            ~partials:0 ~centers:0 ~center:None ()
        with
        | ls, _ -> onces.(i) <- Some ls
        | exception Not_once -> centered := true)
    | Some _ -> centered := true
    | None -> ()
  done;
  let partials = if !centered then align16 (4 * rows) else 0 in
  let bytes = ref partials in
  let center = Some (center_load w ~byte:0) in
  (* The launches in reverse, each reduction's folds in turn. *)
  let rev = ref [] in
  for i = 0 to n - 1 do
    match onces.(i) with
    | Some ls -> rev := List.rev_append ls !rev
    | None ->
        let r, k = rs.(i) in
        let dsts = Array.sub outs firsts.(i) (destinations r) in
        let carrier = carrier_of p k r in
        let folds = ref (folds r) in
        while not (List.is_empty !folds) do
          let fold = List.hd !folds in
          folds := List.tl !folds;
          let ls, e =
            reduce_launches ~holds p w wops dsts ~k ~carrier ~fold ~rows ~terms
              ~seg:terms ~partials ~centers:0 ~center ()
          in
          bytes := Int.max !bytes e;
          rev := List.rev_append ls !rev
        done
  done;
  { launches = List.rev !rev; work = !bytes; copies = [] }

(* A reduction's passes: one, else its reductions split in halves. *)
let rec reduce_passes ~holds p ~shape ~axes rs ops outs =
  match reduce_pass ~holds p ~shape ~axes rs ops outs with
  | c -> [ c ]
  | exception Too_wide when Array.length rs > 1 ->
      let h = Array.length rs / 2 in
      let first = Array.sub rs 0 h
      and rest = Array.sub rs h (Array.length rs - h) in
      let d = Array.fold_left (fun n (r, _) -> n + destinations r) 0 first in
      reduce_passes ~holds p ~shape ~axes first ops (Array.sub outs 0 d)
      @ reduce_passes ~holds p ~shape ~axes rest ops
          (Array.sub outs d (Array.length outs - d))

(* One-term reductions and scans

   A Sum's, a Prod's, a Max's or a Min's output of one term is that term folded
   from its lane's start (fold.h): a float or complex Sum's lane starts at +0,
   so its output is Add (+0, term), which makes a -0 a +0 and flushes as Add
   does; the other lanes start at their first term, so their output is the term.
   A scan's result of one term is the term folded from the chain's start: so, or
   from its carry, the carry first. A reduction each of whose outputs has one
   term, its reductions all such, and a scan along an axis of one term, run as a
   map of those values over the loop of the loads. A +0 is a load of the zero
   constant at stride 0, so that the map stays direct where its term is a
   load. *)

let zero_sum (r : Nx_kernel.reduction) carrier =
  match (r, carrier) with
  | Monoid Sum, D.Any (D.Float32 | D.Complex64) -> true
  | _ -> false

(* The kind that folds a term into a state of [r] at [carrier], a boolean's as
   its lane's integers stored as truths do. *)
let combine (m : Nx_kernel.monoid) carrier : P.binary =
  match (m, boolean carrier) with
  | (Sum | Max), true -> Or
  | (Prod | Min), true -> And
  | Sum, false -> Add
  | Prod, false -> Mul
  | Max, false -> Maximum
  | Min, false -> Minimum

(* The map of [p]'s monoids [rs] each of one term into [dsts], from [carry]
   where given: its program, then its operands past [ops]. *)
let one_term_map (p : P.t) ~rank ?carry rs ops (dsts : operand array) =
  let nodes = ref (List.rev (Iarray.to_list p.nodes)) in
  let count = ref (Iarray.length p.nodes) in
  let add node =
    nodes := node :: !nodes;
    incr count;
    !count - 1
  in
  let cast x ~from ~into =
    if from = into then x else add (P.Op1 (Cast, into, x))
  in
  let inputs = ref [] in
  let input (o : operand) =
    inputs := !inputs @ [ o ];
    add (P.In (Iarray.length p.ins + List.length !inputs - 1))
  in
  let zeros = ref [] in
  let zero carrier =
    match List.assoc_opt carrier !zeros with
    | Some i -> i
    | None ->
        let i =
          input
            {
              buffer = Zero;
              dtype = carrier;
              offset = 0;
              lo = 0;
              hi = 1;
              access = Strided (Array.make rank 0);
            }
        in
        zeros := (carrier, i) :: !zeros;
        i
  in
  let carried = Option.map (fun c -> lazy (input c)) carry in
  let outs =
    Array.mapi
      (fun i ((r : Nx_kernel.reduction), k) ->
        let m =
          match r with
          | Monoid m -> m
          | Arg _ | Logsumexp | Moments -> invalid_arg "Loop.one_term_map"
        in
        let carrier = carrier_of p k r in
        let x = Iarray.get p.outs k in
        (* A boolean term is read as true or false, whatever its byte, as the
           lane reads it: through a byte of 0 or 1. *)
        let term =
          if boolean carrier then
            let byte = add (P.Op1 (Cast, D.Any D.Uint8, x)) in
            add (P.Op1 (Cast, carrier, byte))
          else cast x ~from:(P.dtype p x) ~into:carrier
        in
        let v =
          match carried with
          | Some c ->
              add (P.Op2 (Binary (combine m carrier), Lazy.force c, term))
          | None when zero_sum r carrier ->
              add (P.Op2 (Binary Add, zero carrier, term))
          | None -> term
        in
        cast v ~from:carrier ~into:dsts.(i).dtype)
      rs
  in
  let ins =
    Iarray.of_list
      (Iarray.to_list p.ins @ List.map (fun (o : operand) -> o.dtype) !inputs)
  in
  ( P.v ~ins (Iarray.of_list (List.rev !nodes)) ~outs:(Iarray.of_array outs),
    Array.append ops (Array.of_list !inputs) )

(* [o], a destination over the loop's kept axes, over the whole loop: its own
   strides along the kept axes, 0 along the reduced [axes]. *)
let seen ~rank ~axes (o : operand) =
  match o.access with
  | Pad _ -> invalid_arg "Loop.seen: a padded destination"
  | Strided s ->
      let s' = Array.make rank 0 and j = ref 0 in
      for a = 0 to rank - 1 do
        if not (Array.mem a axes) then begin
          s'.(a) <- s.(!j);
          incr j
        end
      done;
      { o with access = Strided s' }

let is_monoid ((r : Nx_kernel.reduction), _) =
  match r with Monoid _ -> true | Arg _ | Logsumexp | Moments -> false

(* Whether each of [axes] is of extent 1 in [shape]. *)
let of_one_term shape axes =
  let one = ref true in
  for i = 0 to Iarray.length axes - 1 do
    if shape.(Iarray.get axes i) <> 1 then one := false
  done;
  !one

let rec reduce ~holds (p : P.t) ~shape ~axes
    (rs : (Nx_kernel.reduction * int) iarray) (xs : Operands.t) =
  if Iarray.exists (( = ) 0) shape then
    empty (fun () ->
        reduce ~holds p
          ~shape:(Iarray.map (fun _ -> 1) shape)
          ~axes rs (Operands.ones xs))
  else
    let shape = Iarray.to_array shape and rs = Iarray.to_array rs in
    let outs = Array.map plain xs.dsts in
    if of_one_term shape axes && Array.for_all is_monoid rs then
      let rank = Array.length shape and axes = Iarray.to_array axes in
      let outs = Array.map (seen ~rank ~axes) outs in
      parted ~holds ~close:plan p ~shape (Array.map of_load xs.loads)
        (fun ~holds s p ops ->
          let m, ops = one_term_map p ~rank rs ops outs in
          map_passes ~holds s m ~shape ops outs)
    else
      let shapes = Array.map shape_of xs.dsts in
      let strided =
        List.filter
          (fun j -> not (contiguous outs.(j) shapes.(j)))
          (List.init (Array.length outs) Fun.id)
      in
      (* The reduction kernel puts an output's results from its first position
         on, one after another (outputs_at). A destination that is not
         C-contiguous takes them in a C-contiguous temporary of the stand-ins,
         and a copy pass after the reduction's, a loop of [In], writes the
         temporary into it. *)
      parted ~holds ~close:plan p ~shape (Array.map of_load xs.loads)
        (fun ~holds s p ops ->
          let into = Array.copy outs in
          let copies =
            List.concat_map
              (fun j ->
                let o = outs.(j) and shape = shapes.(j) in
                let t = fresh s [ (o.dtype, shape) ] in
                into.(j) <- t.(0);
                let copy = P.v ~ins:[| o.dtype |] [| P.In 0 |] ~outs:[| 0 |] in
                halves ~holds copy ~shape t [| o |] [ 0 ])
              strided
          in
          reduce_passes ~holds p ~shape ~axes rs ops into @ copies)

(* Scans *)

(* The threads that load a chunk of a scan whose results step by one along the
   slice, so that its loads are coalesced; else one, each thread a chunk, so
   that slices side by side load side by side. *)
let chunk_items = 16

(* The levels a network over [terms] terms holds: one per bit of the greatest
   term's position, and one above. *)
let network_levels terms =
  let rec bits n = if n = 0 then 0 else 1 + bits (n lsr 1) in
  bits terms + 1

(* A network state's bytes: three float32. *)
let net_bytes = 12

(* The slices a network's group takes side by side, a power of two: as many as
   its tiles hold a slice's [terms] in, at least 32 where the results do not
   step by one along the slice, so that neighbouring threads load neighbouring
   slices, and at most [slices]. *)
let network_side ~stepping ~terms slices =
  let rec up w = if w >= Int.min terms threads then w else up (2 * w) in
  let rec down w n = if 2 * w > n then w else down (2 * w) n in
  let fill = threads / up 1 in
  down 1 (Int.min slices (Int.max fill (if stepping then 1 else 32)))

(* A slice's bytes of a network's threadgroup memory past the planes: its
   block's tile network's [levels], two states each, and the last two tiles'
   ends. *)
let network_scratch levels = 4 * 3 * ((2 * levels) + 2)

(* A join's work and links: Logsumexp's takes an exp. *)
let join_ops (r : Nx_kernel.reduction) =
  match r with Logsumexp -> 512 | Moments | Monoid _ | Arg _ -> fold_ops

(* A scan's kernel over its units, of [s] threads each: its header, but its
   runs, its table, fields and threadgroup bytes, with [scratch] past its
   planes, the elements a thread takes, and the cost of a tile of its body and
   of its epilogue. *)
type scan_kernel = {
  h : int array;
  table : string;
  fields : field array;
  shared : int;
  v : int;
  tile_bytes : int;
  tape_ops : int;
  tape_chain : int;
  epi_ops : int;
  epi_chain : int;
}

let scan_kernel (p : P.t) w ops outs ~k ~carrier ~mode ~rk ~units ~terms ~chunks
    ~carry ~s ~scratch ~(r : Nx_kernel.reduction) =
  let network = mode = Network in
  let len = if network then terms else Int.min terms K.loop_chunk in
  let wide = wide (units * len) (Array.append ops outs) in
  let build ~eager =
    let t = tape () in
    let f = fixed t ~carrier ~center:false ~result:true in
    lower_term t p k ~carrier ~axis:(kernel_axis w) ~eager ~f;
    (* The body's values are dead once the results are folded: the epilogue's
       temporaries take any free planes. *)
    let store j ~plane ~from =
      let (o : operand) = outs.(j) in
      if code o.dtype = code from then
        ins t.epi K.loop_store ~d:0 ~a:plane ~b:j ~c:0
      else begin
        let q = alloc t.pl (words o.dtype) in
        cast t.epi ~d:q ~a:plane ~from ~into:o.dtype;
        ins t.epi K.loop_store ~d:0 ~a:q ~b:j ~c:0;
        release t.pl q (words o.dtype)
      end
    in
    store 0 ~plane:f.result ~from:(if network then float32 else carrier);
    (match r with
    | Arg _ -> store 1 ~plane:(f.result + 4) ~from:int64
    | Moments -> store 1 ~plane:(f.result + 4) ~from:float32
    | Monoid _ | Logsumexp -> ());
    ( t,
      f,
      elements ~extra:scratch ~stores:true ~wide ~ext:w.ext t
        (Array.append ops outs) )
  in
  let t, f, v = fitted build ~needed:(fun _ -> false) in
  let nl = Array.length ops in
  let fields = fields ~shared:true ops outs in
  let nf = Array.length fields in
  let h = header ~wide v in
  seti h K.loop_term f.term;
  seti h K.loop_result f.result;
  seti h K.loop_reduction
    (reduction_word ~rk ~ct:(compute carrier) ~carrier ~fin:K.loop_put);
  seti h K.loop_terms terms;
  seti h K.loop_segments chunks;
  seti h K.loop_outputs units;
  seti h K.loop_group s;
  seti h K.loop_work (nf - 2);
  seti h K.loop_tickets (nf - 1);
  seti h K.loop_partials 0;
  seti h K.loop_totals 0;
  seti h K.loop_carry carry;
  seti h K.loop_center (if network then t.pl.top else 0);
  let table = table w ops ~ofield:Fun.id outs ~dfield:(fun j -> nl + j) t h in
  let tile_bytes, tape_ops, tape_chain =
    tile_cost ~dispatch ~ext:w.ext ~v ops outs t.body ~from:0 t.body.n
  in
  let _, epi_ops, epi_chain =
    tile_cost ~dispatch ~ext:w.ext ~v [||] [||] t.epi ~from:0 t.epi.n
  in
  {
    h;
    table;
    fields;
    shared = (t.pl.top * plane_bytes v) + scratch;
    v;
    tile_bytes;
    tape_ops;
    tape_chain;
    epi_ops;
    epi_chain;
  }

(* The threads a chunk of [len] terms spreads them over: the least power of two
   from [base] whose share of a term's [links] of tape take no longer than the
   chunk's first thread folding it, at most a group's and as many as its terms
   need. So the tape's chain stays within the fold's, whatever the program. *)
let spread ~base ~len links =
  let rec up s =
    if s >= threads || s >= len || links <= s * fold_links then s else up (2 * s)
  in
  up base

(* A Monoid's or Arg's scan's launches over its units, chunks of a slice, of
   {!spread} threads each from [base]: the body's terms, then the epilogue's
   stores of the results into [outs]. Also whether a chunk's runs took several
   pieces. *)
let scan_launches ~holds (p : P.t) w ops outs ~k ~carrier ~rk ~units ~terms
    ~chunks ~carry ~base ~(r : Nx_kernel.reduction) =
  let sk =
    scan_kernel p w ops outs ~k ~carrier ~mode:Scan ~rk ~units ~terms ~chunks
      ~carry ~s:base ~scratch:0 ~r
  in
  let v = sk.v and len = Int.min terms K.loop_chunk in
  (* A chunk's first thread folds its chunk's terms in order: a run of [s]
     threads' terms folds [s v] of them. *)
  let run_ops = sk.tape_ops + sk.epi_ops + (threads * v * fold_ops) in
  let run_chain s = sk.tape_chain + sk.epi_chain + (s * v * fold_links) in
  (* The cost of [n] groups of [rn] runs each of [s] threads' terms. *)
  let cost_of s rn n =
    {
      Plan.bytes = n * rn * sk.tile_bytes;
      ops = n * rn * run_ops;
      chain = rn * run_chain s;
    }
  in
  let resident =
    Plan.resident ~granted:holds (kernel_of Scan sk.h) ~shared:sk.shared
  in
  let fits s = seconds ~groups:1 ~resident (cost_of s 1 1) <= Plan.allowed () in
  (* The spread, down to [base] where one run of it would take past a launch. *)
  let rec narrow s = if s <= base || fits s then s else narrow (s / 2) in
  let s = narrow (spread ~base ~len (sk.tape_chain / v)) in
  seti sk.h K.loop_group s;
  let runs = ceil_div len (s * v) in
  seti sk.h K.loop_runs runs;
  stepped ~holds Scan
    ~total:(ceil_div units (threads / s))
    ~steps:runs
    ~first:(fun r -> r * s * v)
    ~shared:sk.shared ~fields:sk.fields ~table:sk.table ~h:sk.h
    ~cost:(cost_of s) ()

(* The launches that turn each slice's chunk totals into the states its chunks
   start from, a thread a slice, its chunks in pieces where they take longer
   than a launch, its state between them in the workspace from byte
   [running]. *)
let carries_launches ~holds w ops outs ~carrier ~rk ~slices ~terms ~chunks
    ~carry ~running =
  let nl = Array.length ops in
  let fields = fields ~shared:true ops outs in
  let nf = Array.length fields in
  let h = header ~wide:(wide (slices * terms) ops) 1 in
  seti h K.loop_reduction
    (reduction_word ~rk ~ct:(compute carrier) ~carrier ~fin:K.loop_put);
  seti h K.loop_terms terms;
  seti h K.loop_segments chunks;
  seti h K.loop_outputs slices;
  seti h K.loop_work (nf - 2);
  seti h K.loop_tickets (nf - 1);
  seti h K.loop_totals 0;
  seti h K.loop_partials running;
  seti h K.loop_carry carry;
  let table =
    table w ops ~ofield:Fun.id outs ~dfield:(fun j -> nl + j) (tape ()) h
  in
  (* [n] groups of [cn] chunks each, and the running state's read and write. *)
  let cost cn n =
    {
      Plan.bytes = n * threads * ((2 * (cn + 1) * state_bytes) + Plan.line);
      ops = n * threads * cn * fold_ops;
      chain = (cn + 1) * ((2 * Plan.read_links) + fold_links);
    }
  in
  stepped ~holds Carries ~total:(ceil_div slices threads) ~steps:chunks
    ~first:Fun.id ~shared:0 ~fields ~table ~h ~cost ()

(* The launches that turn each slice's block totals into the network's results
   at its blocks' ends, a thread a slice, its blocks in pieces where they take
   longer than a launch; the totals at byte 0 of the workspace, the stream's
   levels from byte [levels]. *)
let chain_launches ~holds w ops outs ~(r : Nx_kernel.reduction) ~slices ~terms
    ~blocks ~levels =
  let nl = Array.length ops in
  let fields = fields ~shared:true ops outs in
  let nf = Array.length fields in
  let h = header ~wide:(wide (slices * terms) ops) 1 in
  seti h K.loop_reduction
    (reduction_word ~rk:(reduction_code r) ~ct:(compute float32)
       ~carrier:float32 ~fin:K.loop_put);
  seti h K.loop_segments blocks;
  seti h K.loop_outputs slices;
  seti h K.loop_work (nf - 2);
  seti h K.loop_tickets (nf - 1);
  seti h K.loop_totals 0;
  seti h K.loop_partials levels;
  let table =
    table w ops ~ofield:Fun.id outs ~dfield:(fun j -> nl + j) (tape ()) h
  in
  (* [n] groups of [cn] blocks each: a block's total read and its end written, a
     join on average for its set low bits and one for its end, and the levels
     they read and write. *)
  let cost cn n =
    {
      Plan.bytes = n * threads * ((6 * cn * net_bytes) + Plan.line);
      ops = n * threads * cn * 2 * join_ops r;
      chain =
        Plan.read_links + (cn * ((2 * (fold_links + 8)) + (2 * Plan.read_links)));
    }
  in
  stepped ~holds Carries ~total:(ceil_div slices threads) ~steps:blocks
    ~first:Fun.id ~shared:0 ~fields ~table ~h ~cost ()

(* The launches of Moments' centers: a map of [p]'s output [k] at each slice's
   term 0, cast to float32, over [loads] on a loop of [shape] with [axis] cut to
   its first index, into a word a slice from byte [at] of its pass's workspace,
   C-contiguous, the network's order of its slices. *)
let centers_launches ~holds (p : P.t) ~k ~shape ~axis loads ~at =
  let f32 = D.Any D.Float32 and n = Iarray.length p.nodes in
  let cast = P.Op1 (Cast, f32, Iarray.get p.outs k) in
  let p = P.v ~ins:p.ins (Iarray.append p.nodes [| cast |]) ~outs:[| n |] in
  let heads = Array.copy shape in
  heads.(axis) <- 1;
  let ts, _ = temporaries Space [ (f32, heads) ] in
  let c = ts.(0) and word = at / 4 in
  let dst =
    { c with offset = c.offset + word; lo = c.lo + word; hi = c.hi + word }
  in
  map_launches ~holds p ~shape:heads loads [| dst |]

(* A Logsumexp's or Moments' scan's launches over [slices] slices of [terms],
   [side] side by side a group. A slice is cut into blocks of [g] tiles, the
   most, a power of two, that keep a block within a chunk of 4096 terms and its
   group within a launch. A slice of one block is one down launch's; one of
   several takes an up pass, the blocks' chain, then a down pass, each block in
   parallel but the chain. Moments' centers come first, a map over [loads] on a
   loop of [shape]. *)
let network_launches ~holds (p : P.t) w ops outs ~k ~carrier ~slices ~terms
    ~side ~shape ~axis ~loads ~(r : Nx_kernel.reduction) =
  let s = threads / side in
  (* The threadgroup memory past the planes, for a block's tiles at one element
     a thread, the most. *)
  let scratch =
    let tiles = Int.min (K.loop_chunk / s) (ceil_div terms s) in
    align16 (side * network_scratch (network_levels tiles))
  in
  let sk =
    scan_kernel p w ops outs ~k ~carrier ~mode:Network ~rk:(reduction_code r)
      ~units:slices ~terms ~chunks:1 ~carry:(-1) ~s ~scratch ~r
  in
  let v = sk.v and tile = s * sk.v in
  let tiles = ceil_div terms tile in
  (* A down tile joins each term twice, up and down, a level a barrier, and its
     end through the block's levels, a join on average; an up tile once, and its
     end. A group reads its prefix, its last end and its center. *)
  let down_ops = sk.tape_ops + sk.epi_ops + (2 * threads * v * join_ops r) in
  let down_chain =
    sk.tape_chain + sk.epi_chain
    + (2 * levels_of tile * (fold_links + 8))
    + (2 * fold_links)
  in
  let up_ops = sk.tape_ops + (threads * v * join_ops r) in
  let up_chain =
    sk.tape_chain + (levels_of tile * (fold_links + 8)) + fold_links
  in
  let cost ~ops ~chain g n =
    {
      Plan.bytes = n * g * sk.tile_bytes;
      ops = n * g * ops;
      chain = (3 * Plan.read_links) + (g * chain);
    }
  in
  let resident =
    Plan.resident ~granted:holds (kernel_of Network sk.h) ~shared:sk.shared
  in
  let fits g =
    seconds ~groups:1 ~resident (cost ~ops:down_ops ~chain:down_chain g 1)
    <= Plan.allowed ()
  in
  let rec pick g = if g = 1 || fits g then g else pick (g / 2) in
  let g = pick (Int.max 1 (K.loop_chunk / tile)) in
  let runs = Int.min g tiles and blocks = ceil_div tiles g in
  (* The workspace: the blocks' totals and the chain's levels, where a slice has
     several blocks, then Moments' centers. *)
  let totals = if blocks > 1 then slices * blocks * net_bytes else 0 in
  let chained =
    if blocks = 1 then 0
    else totals + (slices * 2 * network_levels blocks * net_bytes)
  in
  let moments = r = Moments in
  let h = sk.h in
  seti h K.loop_segment (runs * tile);
  seti h K.loop_segments blocks;
  seti h K.loop_runs runs;
  seti h K.loop_centers chained;
  let pass h ~total ~cost =
    launches ~holds Network ~total ~step:1 ~shared:sk.shared ~fields:sk.fields
      ~table:sk.table ~h ~cost
      ~tickets:(fun _ _ -> 0)
      ()
  in
  let groups = ceil_div slices side * blocks in
  let heads =
    if moments then centers_launches ~holds p ~k ~shape ~axis loads ~at:chained
    else []
  in
  let down =
    pass h ~total:groups ~cost:(cost ~ops:down_ops ~chain:down_chain runs)
  in
  let work = if moments then chained + (4 * slices) else chained in
  if blocks = 1 then { launches = heads @ down; work; copies = [] }
  else begin
    let up = Array.copy h in
    up.(0) <- up.(0) lor K.loop_up;
    let ups =
      pass up ~total:groups ~cost:(cost ~ops:up_ops ~chain:up_chain runs)
    in
    let chain, _ =
      chain_launches ~holds w ops outs ~r ~slices ~terms ~blocks ~levels:totals
    in
    { launches = heads @ ups @ chain @ down; work; copies = [] }
  end

(* A scan's pass. Fields: the loads, a carry, the destinations, the workspace,
   the tickets. A monoid's or Arg's scan of several chunks a slice first reduces
   each chunk into the workspace, its total, then folds a slice's totals in
   order from its carry into the states its chunks start from. A network's group
   scans a block of each of its slices' tiles in order in threadgroup memory. *)
let scan_pass ?carry ~holds (p : P.t) ~shape ~axis
    ((r : Nx_kernel.reduction), k) ops outs =
  let p, ops = compact p ~roots:[ k ] ops in
  let order, kept = order_of (Array.length shape) ~reduced:[ axis ] in
  let ops =
    match carry with Some c -> Array.append ops [| c |] | None -> ops
  in
  let nl = Array.length ops and nd = Array.length outs in
  let merge = not (reads_coord p || Array.exists is_padded ops) in
  let w, all = walk shape order ~kept ~merge (Array.append ops outs) in
  let wops = Array.sub all 0 nl and wouts = Array.sub all nl nd in
  let slices = product w.ext 0 w.kept in
  let terms = product w.ext w.kept (Array.length w.ext) in
  let carrier = carrier_of p k r in
  let carry = if Option.is_some carry then nl - 1 else -1 in
  let rk = reduction_code r in
  match r with
  | Logsumexp | Moments ->
      let rank = Array.length w.ext in
      let stepping =
        match wouts.(0).access with
        | Strided [||] -> true
        | Strided st -> st.(rank - 1) = 1
        | Pad _ -> false
      in
      let side = network_side ~stepping ~terms slices in
      network_launches ~holds p w wops wouts ~k ~carrier ~slices ~terms ~side
        ~shape ~axis ~loads:ops ~r
  | Monoid _ | Arg _ ->
      let chunks = ceil_div terms K.loop_chunk in
      let units = slices * chunks in
      let totals = units * state_bytes in
      (* The totals, then the totals pass's partials, which the carries' running
         states take once it is done. *)
      let first, work =
        if chunks = 1 then ([], 0)
        else
          let ls, e =
            reduce_launches ~holds p w wops wouts ~k ~carrier
              ~fold:{ rk; fin = K.loop_put_total; centered = false }
              ~rows:slices ~terms ~seg:K.loop_chunk ~partials:(align16 totals)
              ~centers:0 ~center:None ()
          in
          let running = align16 totals in
          let cs, _ =
            carries_launches ~holds w wops wouts ~carrier ~rk ~slices ~terms
              ~chunks ~carry ~running
          in
          (ls @ cs, Int.max e (running + (slices * state_bytes)))
      in
      (* Coalesced where the results step by one along the slice. *)
      let rank = Array.length w.ext in
      let base =
        match wouts.(0).access with
        | Strided st when terms >= chunk_items && st.(rank - 1) = 1 ->
            chunk_items
        | Strided _ | Pad _ -> 1
      in
      (* A chunk whose runs take several launches leaves its state in its
         total's place between them. *)
      let last, split =
        scan_launches ~holds p w wops wouts ~k ~carrier ~rk ~units ~terms
          ~chunks ~carry ~base ~r
      in
      let work = if split then Int.max work totals else work in
      { launches = first @ last; work; copies = [] }

(* A carry's stride along [axis] is 0, so that every index of a slice reads
   it. *)
let rec scan ~holds (p : P.t) ~axis rk (xs : Operands.t) =
  let shape = shape_of xs.dsts.(0) in
  if Array.exists (( = ) 0) shape then
    empty (fun () -> scan ~holds p ~axis rk (Operands.ones xs))
  else
    let carry =
      Option.map
        (fun c ->
          let o = plain c in
          match o.access with
          | Strided s ->
              let s = Array.copy s in
              s.(axis) <- 0;
              { o with access = Strided s }
          | Pad _ -> o)
        xs.carry
    in
    let outs = Array.map plain xs.dsts in
    if shape.(axis) = 1 && is_monoid rk then
      let rank = Array.length shape in
      parted ~holds ~close:plan p ~shape (Array.map of_load xs.loads)
        (fun ~holds s p ops ->
          let m, ops = one_term_map p ~rank ?carry [| rk |] ops outs in
          map_passes ~holds s m ~shape ops outs)
    else
      parted ~holds ~close:plan p ~shape (Array.map of_load xs.loads)
        (fun ~holds _ p ops ->
          [ scan_pass ?carry ~holds p ~shape ~axis rk ops outs ])

(* Temporaries of nodes

   Another family's call computes nodes of a program by these maps into
   temporaries in its own builder's workspace, which its later launches read. *)

(* The maps of nodes into temporaries: their passes over stand-ins of [sizes]
   bytes, of which they name [named]; the score's stand-in, -1 for none, and the
   roots'; each root's byte in its stand-in and its strides. *)
type temps = {
  passes : pass list;
  sizes : int array;
  named : bool array;
  score_buffer : int;
  root_buffer : int;
  roots : (int * int array) array;
}

let temps ~holds (p : P.t) ~shape ~score loads roots =
  if Array.exists (fun e -> e <= 0) shape then
    invalid_arg "Loop.temps: a loop of no element";
  let fixed = stand_ins () in
  let loads = Array.map of_load loads in
  (* With a score, input 0 reads a temporary of the loop's extents; without, it
     reads nothing: each [In 0] is a zero, and the loads are inputs 0 on. *)
  let ins, nodes, ops, score_buffer =
    if score then
      let t = fresh fixed [ (float32, shape) ] in
      (p.ins, p.nodes, Array.append t loads, 0)
    else
      let nodes =
        Iarray.map
          (function
            | P.In 0 -> zero_const (Iarray.get p.ins 0)
            | P.In k -> P.In (k - 1)
            | nd -> nd)
          p.nodes
      in
      let ins = Iarray.sub p.ins ~pos:1 ~len:(Iarray.length p.ins - 1) in
      (ins, nodes, loads, -1)
  in
  let q = P.v ~ins nodes ~outs:(Iarray.of_list roots) in
  let ext = extents q ops shape in
  let root_buffer = fixed.count in
  let outs = fresh fixed (List.map (fun i -> (P.dtype q i, ext.(i))) roots) in
  let root (o : operand) =
    match o.access with
    | Strided s -> (o.offset * bits o.dtype / 8, s)
    | Pad _ -> invalid_arg "Loop.temps: a padded temporary"
  in
  let close s passes =
    {
      passes;
      sizes = sizes s;
      named = named s passes;
      score_buffer;
      root_buffer;
      roots = Array.map root outs;
    }
  in
  parted ~holds ~fixed ~close q ~shape ops (fun ~holds s q ops ->
      map_passes ~holds s q ~shape ops outs)

let score_at b t =
  if t.score_buffer < 0 then invalid_arg "Loop.score_at: maps of no score";
  Plan.take b t.sizes.(t.score_buffer)

let emit ?score b t =
  let base = Array.make (Array.length t.sizes) 0 in
  (match score with
  | Some at when t.score_buffer >= 0 -> base.(t.score_buffer) <- at
  | None when t.score_buffer < 0 -> ()
  | Some _ -> invalid_arg "Loop.emit: a score for maps of none"
  | None -> invalid_arg "Loop.emit: no score for maps of one");
  let roots = Plan.take b t.sizes.(t.root_buffer) in
  base.(t.root_buffer) <- roots;
  (* The roots' and the score's pieces stay taken for the caller's later
     launches; the parts' temporaries are given back. *)
  let m = Plan.mark b in
  Array.iteri
    (fun i n ->
      if t.named.(i) && i <> t.root_buffer && i <> t.score_buffer then
        base.(i) <- Plan.take b n)
    t.sizes;
  emit_passes b ~base t.passes;
  Plan.release b m;
  Array.map (fun (at, s) -> (roots + at, Array.copy s)) t.roots
