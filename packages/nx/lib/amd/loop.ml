(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A loop's launches of loop.hip: its walk, its operands' descriptors and
   its program lowered to a tape, written as each launch's words.

     program, loads --> walk (axes ordered, merged) --> descriptors
     program --> tape over planes --> words --> one launch per pass
     launches, temporaries --> Plan.t, one submission

   A walk orders the loop's axes as the kernel walks them: a reduction's
   kept axes then its reduced ones, a scan's other axes then its axis, each
   in its order, so that the terms keep their C order. Where no load is
   padded and no node reads a coordinate, it drops axes of one element and
   merges an axis into the one before where every operand steps along both
   as one.

   Each kind computes through its function in nx_kinds.h, which nx.cpu's
   kinds call too, so a kind gives nx.cpu's value at every dtype. *)

module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module P = Nx_kernel.Prog
module K = Kernels

external raw_kind : P.node -> int = "nx_amd_kind" [@@noalloc]

(* The code of the kind of [node] as the tape's ops read it (kernels.h): an
   Op1's, NX_LOOP_BINARY plus an Op2's, NX_LOOP_TERNARY plus an Op3's. *)
let kind_code (node : P.node) =
  match node with
  | Op1 _ -> raw_kind node
  | Op2 _ -> K.loop_binary + raw_kind node
  | Op3 _ -> K.loop_ternary + raw_kind node
  | In _ | Coord _ | Const _ -> -1

(* Dtypes *)

let code (D.Any d) = D.code d
let int64 = D.Any D.Int64

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

(* The type a kind of [dt] computes in (Prog's compute types). *)
let compute = function
  | D.Any D.Float64 -> K.loop_f64
  | D.Any D.Complex64 -> K.loop_c64
  | D.Any D.Complex128 -> K.loop_c128
  | D.Any D.Int64 -> K.loop_i64
  | D.Any D.Uint64 -> K.loop_u64
  | D.Any (D.Int32 | D.Int16 | D.Int8 | D.Int4) -> K.loop_i32
  | D.Any (D.Uint32 | D.Uint16 | D.Uint8 | D.Uint4 | D.Bool | D.Bit) ->
      K.loop_u32
  | D.Any _ -> K.loop_f32

(* The held value of [dt] whose bits are the low ones of [r] (loop.cu's
   of_raw): a signed integer of 16 bits or fewer sign-extended to 32. *)
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

(* A launch's parameters as they grow: [n] little-endian words of [b],
   whose setters box nothing. *)
type words = { mutable b : Bytes.t; mutable n : int }

let words_for n = { b = Bytes.create (8 * n); n = 0 }
let get ws i = Bytes.get_int64_le ws.b (8 * i)
let set ws i x = Bytes.set_int64_le ws.b (8 * i) x
let set_int ws i x = Bytes.set_int64_le ws.b (8 * i) (Int64.of_int x)

let room ws =
  if 8 * ws.n = Bytes.length ws.b then begin
    let b = Bytes.create (2 * Bytes.length ws.b) in
    Bytes.blit ws.b 0 b 0 (8 * ws.n);
    ws.b <- b
  end

let push ws x =
  room ws;
  set ws ws.n x;
  ws.n <- ws.n + 1

let push_int ws x =
  room ws;
  set_int ws ws.n x;
  ws.n <- ws.n + 1

(* Planes *)

(* The most planes an instruction names: 12 bits. *)
let max_planes = 4096

exception Too_wide

type planes = { used : Bytes.t; mutable top : int }

let planes () = { used = Bytes.make max_planes '\000'; top = 0 }

let rec free_run pl p n k =
  k = n || (Bytes.get pl.used (p + k) = '\000' && free_run pl p n (k + 1))

(* The first [n] free planes in a row, now taken. *)
let alloc pl n =
  let rec find p =
    if p + n > max_planes then raise Too_wide
    else if free_run pl p n 0 then p
    else find (p + 1)
  in
  let p = find 0 in
  Bytes.fill pl.used p n '\001';
  pl.top <- Int.max pl.top (p + n);
  p

let release pl p n = Bytes.fill pl.used p n '\000'

(* Tapes *)

type tape = { pro : words; body : words; epi : words; pl : planes }

let tape () =
  let w () = words_for 64 in
  { pro = w (); body = w (); epi = w (); pl = planes () }

let ins ws op ~d ~a ~b ~c =
  let f x s = Int64.shift_left (Int64.of_int x) s in
  push ws
    Int64.(
      logor (of_int op)
        (logor (f d 16) (logor (f a 28) (logor (f b 40) (f c 52)))))

let kind ws ct k ~d ~a ~b ~c = ins ws ((ct lsl 7) lor k) ~d ~a ~b ~c

(* Plane [a] of [from] cast to [into] in plane [d]: between float32 and a
   narrow float, which rounds once from the float32 or widens exactly,
   the encode or the decode the runner runs in line; else the cast. *)
let cast ws ~d ~a ~from ~into =
  let f32 = D.Any D.Float32 in
  if from = f32 && narrow_float into then
    ins ws K.loop_encode ~d ~a ~b:(code into) ~c:0
  else if narrow_float from && into = f32 then
    ins ws K.loop_decode ~d ~a ~b:(code from) ~c:0
  else ins ws K.loop_cast ~d ~a ~b:(code from) ~c:(code into)

(* A value of the tape: its first plane and its dtype. *)
type value = { plane : int; dt : D.any }

(* Lowers the nodes of [p] that the outputs [roots] reach into [t]'s
   prologue, constants, and body: first a load of each operand [k] read
   where [eager k], then the nodes in order, each other operand loaded
   before its first reader. [load k] is operand [k]'s load in the launch's
   table, [axis a] the kernel's axis of the loop's axis [a]. After node [i]
   computes, [out k v] is called for each output [k] among [roots] that is
   node [i], with its value. A value's planes are freed after the last
   node that reads it, a constant's never. An [In] or [Coord] node [n]
   with [given n = Some q] holds its value in planes from [q] that the
   kernel fills: no instruction computes it and its planes are never
   freed. *)
let lower ?(given = fun _ -> None) t (p : P.t) ~roots ~load ~axis ~out ~eager =
  let n = Iarray.length p.nodes in
  let node i = Iarray.get p.nodes i in
  let dtype i = P.dtype p i in
  let operands i =
    match node i with
    | P.In _ | Coord _ | Const _ -> []
    | Op1 (_, _, a) -> [ a ]
    | Op2 (_, a, b) -> [ a; b ]
    | Op3 (_, a, b, c) -> [ a; b; c ]
  in
  (* Live nodes, from the outputs back. *)
  let live = Array.make n false in
  List.iter (fun k -> live.(Iarray.get p.outs k) <- true) roots;
  for i = n - 1 downto 0 do
    if live.(i) then List.iter (fun a -> live.(a) <- true) (operands i)
  done;
  (* The node whose value each is: an operand's for one that moves no bit,
     the first read of an operand for a later one. *)
  let root = Array.init n Fun.id in
  let first_in = Hashtbl.create 8 in
  for i = 0 to n - 1 do
    if live.(i) then
      match node i with
      | P.Op1 (Copy, _, a) -> root.(i) <- root.(a)
      | Op1 ((Cast | Bitcast), y, a) when dtype a = y -> root.(i) <- root.(a)
      | Op1 (Unary (Floor | Ceil | Round | Trunc), _, a)
        when compute (dtype a) > K.loop_c128 ->
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
  let fixed =
    Array.init n (fun i ->
        match node i with P.In _ | Coord _ -> given (node i) | _ -> None)
  in
  let loaded = Array.make n false in
  Array.iteri
    (fun i q ->
      match q with
      | Some q when live.(i) ->
          values.(i) <- { plane = q; dt = dtype i };
          loaded.(i) <- true
      | _ -> ())
    fixed;
  let fresh i =
    let dt = dtype i in
    let v = { plane = alloc t.pl (words dt); dt } in
    values.(i) <- v;
    v
  in
  (* Constants first, in the prologue: their planes, written once, are no
     other value's. Then the loads: a load a destination is identical to is
     read before any store. *)
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
  let narrow_code d = code d in
  for i = 0 to n - 1 do
    if live.(i) && root.(i) = i then begin
      (* Operands not yet loaded, loaded and waited for. *)
      let unloaded r =
        match node r with P.In _ -> not loaded.(r) | _ -> false
      in
      let late =
        List.sort_uniq Int.compare
          (List.filter unloaded
             (i :: List.map (fun a -> root.(a)) (operands i)))
      in
      List.iter load_in late;
      let arg a = values.(root.(a)).plane in
      match node i with
      | P.In _ -> ()
      | Coord _ when Option.is_some fixed.(i) -> ()
      | Coord a ->
          let v = fresh i in
          ins t.body K.loop_coord ~d:v.plane ~a:(axis a) ~b:0 ~c:0
      | Const _ -> ()
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
          let v = fresh i and k = kind_code (node i) in
          if narrow_float dt then begin
            let c = narrow_code dt in
            ins t.body K.loop_decode ~d:v.plane ~a:(arg a) ~b:c ~c:0;
            kind t.body K.loop_f32 k ~d:v.plane ~a:v.plane ~b:0 ~c:0;
            ins t.body K.loop_encode ~d:v.plane ~a:v.plane ~b:c ~c:0
          end
          else begin
            kind t.body (compute dt) k ~d:v.plane ~a:(arg a) ~b:0 ~c:0;
            if narrow_int dt then
              ins t.body K.loop_wrap ~d:v.plane ~a:v.plane ~b:(code dt) ~c:0
          end
      | Op2 (op, a, b) ->
          let dt = dtype a and v = fresh i and k = kind_code (node i) in
          let binary =
            match op with P.Binary _ -> true | Compare _ -> false
          in
          if narrow_float dt then begin
            let c = narrow_code dt and q = temp () in
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
            kind t.body (compute dt) k ~d:v.plane ~a:(arg a) ~b:(arg b) ~c:0;
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
          let dt = dtype a and v = fresh i and k = kind_code (node i) in
          if narrow_float dt then begin
            let cd = narrow_code dt and q = temp () and r = temp () in
            ins t.body K.loop_decode ~d:v.plane ~a:(arg a) ~b:cd ~c:0;
            ins t.body K.loop_decode ~d:q ~a:(arg b) ~b:cd ~c:0;
            ins t.body K.loop_decode ~d:r ~a:(arg c) ~b:cd ~c:0;
            (* The exact value rounded to odd, which the encode rounds
               once. *)
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
            kind t.body (compute dt) k ~d:v.plane ~a:(arg a) ~b:(arg b)
              ~c:(arg c);
            if narrow_int dt then
              ins t.body K.loop_wrap ~d:v.plane ~a:v.plane ~b:(code dt) ~c:0
          end
    end;
    if live.(i) then begin
      List.iter
        (fun k -> if Iarray.get p.outs k = i then out k values.(root.(i)))
        roots;
      (* Values read last here go, a constant's planes but never. *)
      let dead r =
        if last.(r) = i then
          match node r with
          | P.Const _ -> ()
          | _ when Option.is_some fixed.(r) -> ()
          | _ -> release t.pl values.(r).plane (words values.(r).dt)
      in
      List.iter (fun a -> dead root.(a)) (operands i);
      dead root.(i)
    end
  done

(* Walks *)

(* An operand as the walk reads it, over the loop's axes in the caller's
   order: a plain one by its strides; a padded one by, per axis of its
   array, the loop axes that give its padded coordinate and their
   coefficients. *)
type axis_read = {
  lo : int;
  interior : int;
  extent : int;
  stride : int;
  a1 : int;
  c1 : int;
  a2 : int; (* -1 for none *)
  c2 : int;
}

type access =
  | Strided of int array
  | Pad of { fill : string; reads : axis_read array }

(* Where an operand's elements lie: an array's buffer, or a buffer of the
   call's temporaries, by its number ({!stand_ins}). *)
type buf = Arr of Rig.Buffer.t | Temp of int

let same_buf a b =
  match (a, b) with
  | Arr x, Arr y -> x == y
  | Temp i, Temp j -> i = j
  | Arr _, Temp _ | Temp _, Arr _ -> false

type operand = {
  buffer : buf;
  dtype : D.any;
  offset : int;
  access : access;
}

let shape_of a = Iarray.to_array (L.shape (A.layout a))

let plain (type v s) (x : (v, s) A.t) =
  let l = A.layout x in
  {
    buffer = Arr (A.buffer x);
    dtype = D.Any (A.dtype x);
    offset = L.offset l;
    access = Strided (Iarray.to_array (L.strides l));
  }

let padded (type v s) (x : (v, s) A.t) (fill : v) (pad : Nx_kernel.pad) =
  let l = A.layout x and dt = A.dtype x in
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
        match window a with
        | None ->
            { lo; interior; extent; stride; a1 = a; c1 = 1; a2 = -1; c2 = 0 }
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
            })
  in
  {
    buffer = Arr (A.buffer x);
    dtype = D.Any dt;
    offset = L.offset l;
    access = Pad { fill = P.bits dt fill; reads };
  }

let of_load : Nx_kernel.load -> operand = function
  | Plain x -> plain x
  | Padded { x; fill; pad } -> padded x fill pad

(* The array of [f] of each element of [a]. *)
let mapped f a = Array.init (Iarray.length a) (fun i -> f (Iarray.get a i))

(* A walk: the kernel's extents, and the loop's axis each kernel axis
   merged last; the kept axes' count. *)
type walk = { ext : int array; perm : int array; kept : int }

(* The walk of a loop of extents [shape] whose kernel order is [order],
   the first [kept] kept, over [ops], merging where [merge]. Each strided
   operand's strides come back in the kernel's order. *)
let walk shape order ~kept ~merge (ops : operand array) =
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
        | Pad _ -> o)
      ops
  in
  (w, ops)

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

(* Launches

   A launch's address words, from word NX_LOOP_HEADER, each hold what rig
   writes there: an operand's buffer, the workspace of its pass or the
   tickets. A pass is launches that share a piece of workspace, in
   order. *)

type target = Buf of buf | Work | Tickets
type launch = {
  kernel : K.t;
  groups : int;
  shared : int;
  ws : words;
  targets : (target * bool) array;
  tickets : int;
}

type pass = { launches : launch list; work : int }

(* A launch's groups loop over their shares of its units, as many as
   {!Plan.grid} gives for its units, so that the count changes no bit. A
   spread reduction aims to run at most {!Plan.stride_groups} at once. *)
let group_slots = Plan.stride_groups

(* A spread reduction's least terms a block, and the bytes of its lanes'
   states, one a work-item, past the planes. *)
let lanes_least = 128
let lane_states = 4 * K.loop_state * K.loop_threads
let ceil_div a b = (a + b - 1) / b
let align16 n = (n + 15) land lnot 15

(* The LDS a group takes for [planes] at [v] elements a work-item. *)
let shared_bytes planes v = planes * v * K.loop_threads * 4

(* The most LDS a launch asks for, the least that a loop kernel with
   planes leaves past its own ({!Plan.shared_most}), and the most it aims
   for, which leaves room for two groups a compute unit. *)
let shared_most =
  List.fold_left
    (fun m k -> Int.min m (Plan.shared_most k))
    max_int
    K.[ Loop_map; Loop_reduce; Loop_scan; Loop_network ]

let shared_aim = 24 * 1024

(* The targets of a launch over [ops] then [outs], then, [shared], the
   workspace and the tickets. *)
let targets ?(shared = false) (ops : operand array) (outs : operand array) =
  Array.concat
    [
      Array.map (fun (o : operand) -> (Buf o.buffer, false)) ops;
      Array.map (fun (o : operand) -> (Buf o.buffer, true)) outs;
      (if shared then [| (Work, true); (Tickets, true) |] else [||]);
    ]

let launch_of kernel (ws : words) ~targets ~groups ~planes ~tickets =
  set_int ws K.loop_grid groups;
  set_int ws K.loop_used ws.n;
  { kernel; groups; shared = planes; ws; targets; tickets }

(* The elements a work-item holds: the most of [most] (a power of two)
   and its halves whose planes and [extra] bytes fit the aim, else 1
   within the most. *)
let elements ?(most = 4) ?(extra = 0) planes =
  let rec fit v =
    if v = 1 then
      if shared_bytes planes 1 + extra <= shared_most then 1 else raise Too_wide
    else if shared_bytes planes v + extra <= shared_aim then v
    else fit (v / 2)
  in
  fit most

(* The words of a launch: [header], set by the caller, then the address
   words, the extents, the descriptors and [t]'s tape. *)
let params ~nslots w (ops : operand array) ~slots (dsts : operand array)
    ~dst_slots (t : tape) (h : int array) =
  let ws = words_for 255 in
  for _ = 0 to K.loop_header + nslots - 1 do
    push ws 0L
  done;
  let set i x = h.(i) <- x in
  set K.loop_rank (Array.length w.ext);
  set K.loop_shape ws.n;
  Array.iter
    (fun e ->
      let at = ws.n in
      push ws 0L;
      push ws 0L;
      Plan.axis ws.b at e)
    w.ext;
  set K.loop_numel (Array.fold_left ( * ) 1 w.ext);
  let table field (os : operand array) slot_of =
    let at = ws.n in
    set field at;
    Array.iter (fun _ -> push ws 0L) os;
    Array.iteri
      (fun k (o : operand) ->
        set_int ws (at + k) ws.n;
        let padded = match o.access with Pad _ -> 1 | Strided _ -> 0 in
        push_int ws (code o.dtype lor (padded lsl 8) lor (slot_of k lsl 16));
        push_int ws o.offset;
        match o.access with
        | Strided s -> Array.iter (push_int ws) s
        | Pad { fill; reads } ->
            let b = Bytes.make 16 '\000' in
            Bytes.blit_string fill 0 b 0 (String.length fill);
            push ws (Bytes.get_int64_le b 0);
            push ws (Bytes.get_int64_le b 8);
            push_int ws (Array.length reads);
            Array.iter
              (fun r ->
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
                  ])
              reads)
      os
  in
  table K.loop_loads ops slots;
  table K.loop_dsts dsts dst_slots;
  set K.loop_tape ws.n;
  let copy (s : words) =
    for k = 0 to s.n - 1 do
      push ws (get s k)
    done
  in
  set K.loop_prologue t.pro.n;
  copy t.pro;
  set K.loop_body t.body.n;
  copy t.body;
  set K.loop_epilogue t.epi.n;
  copy t.epi;
  for i = 0 to K.loop_header - 1 do
    set_int ws i h.(i)
  done;
  if ws.n > K.loop_words then raise Too_wide;
  ws

(* [build ~eager] with every load read first where its planes fit, else
   with only the loads [needed] read first, the others each before its
   first reader, which frees their planes sooner. *)
let fitted build ~needed =
  match build ~eager:(fun _ -> true) with
  | l -> l
  | exception Too_wide -> build ~eager:needed

(* The header of a launch at [v] elements a work-item. *)
let header v =
  let h = Array.make K.loop_header 0 in
  h.(K.loop_mode) <- v lsl 8;
  h.(K.loop_carry) <- -1;
  h

let seti h i x = h.(i) <- x
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
      match Iarray.get p.nodes i with
      | P.In _ | Coord _ | Const _ -> ()
      | Op1 (_, _, a) -> live.(a) <- true
      | Op2 (_, a, b) ->
          live.(a) <- true;
          live.(b) <- true
      | Op3 (_, a, b, c) ->
          live.(a) <- true;
          live.(b) <- true;
          live.(c) <- true
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
    let keep a =
      List.filter (fun k -> used.(k)) (List.init (Array.length a) Fun.id)
    in
    let kept = keep ops in
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

   A plan lays its temporaries out in buffers it numbers, each a piece of
   the call's workspace once the plan is whole: a pass tried, too wide,
   and planned again in parts takes none. *)

type stand_ins = { mutable sizes : int list; mutable count : int }

let stand_ins () = { sizes = []; count = 0 }

(* Temporaries in [buffer] of each dtype and extents of [xs], each an
   axis of the loop's or 1 where its node does not vary along it, read
   there at stride 0: C-contiguous from 16-byte aligned bytes. Also the
   bytes they take. *)
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
        {
          buffer;
          dtype = D.Any d;
          offset = byte * 8 / D.bits d;
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

(* The plan of [passes], in order: the temporaries of [s] they name, then
   each pass's workspace, a piece of the workspace each, which a later
   pass takes again. *)
let plan (s : stand_ins) (passes : pass list) =
  let b = Plan.make () in
  let named = Array.make s.count false in
  let note (t, _) = match t with Buf (Temp i) -> named.(i) <- true | _ -> () in
  List.iter
    (fun c -> List.iter (fun l -> Array.iter note l.targets) c.launches)
    passes;
  let sizes = Array.of_list (List.rev s.sizes) in
  let base = Array.mapi (fun i n -> if named.(i) then Plan.take b n else 0) sizes in
  (* A pass of no workspace, or a launch of no ticket, does not name the
     device's workspace or tickets: calls that take neither do not wait on
     each other. Their address words stay 0. *)
  let emit ~worked work l =
    let w = Plan.words l.kernel in
    Bytes.blit l.ws.b 0 w 0 (8 * l.ws.n);
    let ticket = if l.tickets > 0 then Plan.tickets b l.tickets else 0 in
    let ref j (t, write) =
      let word = K.loop_header + j in
      let named slot at =
        Plan.set w word at;
        Some { Plan.word; slot; write }
      in
      match t with
      | Buf (Arr x) -> named (Plan.slot b x) 0
      | Buf (Temp i) -> named (Plan.workspace b) base.(i)
      | Work -> if worked then named (Plan.workspace b) work else None
      | Tickets ->
          if l.tickets > 0 then named (Plan.tickets_slot b) ticket else None
    in
    let refs =
      List.filter_map Fun.id (Array.to_list (Array.mapi ref l.targets))
    in
    Plan.launch b l.kernel ~groups:(l.groups, 1, 1) ~shared:l.shared
      ~used:l.ws.n w (Array.of_list refs)
  in
  List.iter
    (fun c ->
      let m = Plan.mark b in
      let work = if c.work > 0 then Plan.take b c.work else 0 in
      List.iter (emit ~worked:(c.work > 0) work) c.launches;
      Plan.release b m)
    passes;
  Plan.finish_some b

(* Maps *)

(* Whether the plain operand [o] is one of [outs]: identical to a
   destination, which a map stores only after it reads each index. *)
let identical outs (o : operand) =
  match o.access with
  | Pad _ -> false
  | Strided _ ->
      Array.exists
        (fun (y : operand) ->
          same_buf y.buffer o.buffer && y.dtype = o.dtype
          && y.offset = o.offset && y.access = o.access)
        outs

(* One launch of a map of [p] over [ops] into [outs], on a loop of
   [shape]. *)
let map_launch (p : P.t) ~shape ops outs =
  let p, ops = compact p ~roots:(List.init (Iarray.length p.outs) Fun.id) ops in
  let nl = Array.length ops in
  let needed k = identical outs ops.(k) in
  let merge = not (reads_coord p || Array.exists is_padded ops) in
  let w, all =
    walk shape
      (Array.init (Array.length shape) Fun.id)
      ~kept:0 ~merge (Array.append ops outs)
  in
  let wops = Array.sub all 0 nl
  and wouts = Array.sub all nl (Array.length outs) in
  let n = Array.fold_left ( * ) 1 w.ext in
  let build ~eager =
    let t = tape () in
    lower t p
      ~roots:(List.init (Iarray.length p.outs) Fun.id)
      ~load:Fun.id ~axis:(kernel_axis w) ~eager
      ~out:(fun k v -> ins t.body K.loop_store ~d:0 ~a:v.plane ~b:k ~c:0);
    let v = elements ~most:K.loop_most t.pl.top in
    (t, v)
  in
  let t, v = fitted build ~needed in
  let nslots = nl + Array.length outs in
  let ws =
    params ~nslots w wops ~slots:Fun.id wouts
      ~dst_slots:(fun k -> nl + k)
      t (header v)
  in
  let l =
    launch_of K.Loop_map ws ~targets:(targets ops outs)
      ~groups:(Plan.grid ~per:(v * K.loop_threads) K.Loop_map n)
      ~planes:(shared_bytes t.pl.top v) ~tickets:0
  in
  { launches = [ l ]; work = 0 }

(* [p] with its outputs [ks] alone, in that order. *)
let outputs (p : P.t) ks =
  P.v ~ins:p.ins p.nodes
    ~outs:(Iarray.of_list (List.map (Iarray.get p.outs) ks))

let pick a ks = Array.of_list (List.map (Array.get a) ks)

(* The passes of a map of [p]'s outputs [ks] into [outs], none of which is
   identical to a load: one launch, else the outputs in halves. *)
let rec halves (p : P.t) ~shape ops outs ks =
  match map_launch (outputs p ks) ~shape ops (pick outs ks) with
  | c -> [ c ]
  | exception Too_wide when List.length ks > 1 ->
      let h = List.length ks / 2 in
      halves p ~shape ops outs (List.filteri (fun i _ -> i < h) ks)
      @ halves p ~shape ops outs (List.filteri (fun i _ -> i >= h) ks)

(* A map's passes: one launch, else its outputs apart, each computed with
   the whole's bits. A destination identical to a load is written only
   after every launch that reads the load: such outputs go last, in one
   launch, else into temporaries copied into them at the end. *)
let map_passes s (p : P.t) ~shape ops outs =
  let all = List.init (Iarray.length p.outs) Fun.id in
  match map_launch p ~shape ops outs with
  | c -> [ c ]
  | exception Too_wide when List.length all > 1 ->
      let read k = Array.exists (identical [| outs.(k) |]) ops in
      let aliased, apart = List.partition read all in
      let first = if apart = [] then [] else halves p ~shape ops outs apart in
      if aliased = [] then first
      else
        let last =
          match
            map_launch (outputs p aliased) ~shape ops (pick outs aliased)
          with
          | c -> [ c ]
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
              halves (outputs p aliased) ~shape ops ts ks
              @ halves copy ~shape ts (pick outs aliased) ks
        in
        first @ last

(* Parts

   A program too wide for a launch, past its 4 KiB of words (512, more
   than the part that copies out one padded load of 32 axes takes) or a
   group's 60 KiB of LDS for its values held at once, runs in parts. A part maps the values that its nodes before a
   cut pass to later ones into temporaries in a buffer of its own, each
   over the loop's axes along which its node varies, and the rest of the
   program reads them as loads, at stride 0 along the other axes;
   a load too wide to read beside a node's others is first copied into
   one. A node's value is an element of its dtype, so a temporary holds
   it exactly and the parts give one launch's bits. *)

let operands_of : P.node -> int list = function
  | P.In _ | Coord _ | Const _ -> []
  | Op1 (_, _, a) -> [ a ]
  | Op2 (_, a, b) -> [ a; b ]
  | Op3 (_, a, b, c) -> [ a; b; c ]

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

(* The extents of each node of [p] over [ops] on a loop of [shape]: the
   loop's along the axes its leaves vary along, a load's of more than one
   element and a nonzero stride, a padded load's or a coordinate's, and 1
   along the others. *)
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

(* The first part of [p] over [ops] on a loop of [shape], its temporaries
   in stand-ins of [s], and the rest: the largest cut whose first part one
   launch maps, else the widest load copied into a temporary. [None] where
   neither fits a launch. *)
let part s (p : P.t) ~shape ops =
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
    match map_launch p1 ~shape ops ts with
    | _ -> true
    | exception Too_wide -> false
  in
  (* A cut after each live computed node, each part cut there computing
     the live nodes before it, so that the rest computes fewer. *)
  let cuts =
    Array.of_list
      (List.filter_map
         (fun i ->
           if live.(i) && computed (Iarray.get p.nodes i) then Some (i + 1)
           else None)
         (List.init (Iarray.length p.nodes) Fun.id))
  in
  (* The last cut that fits, by bisection: the first fits, the last does
     not. *)
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
    Some (map_launch p1 ~shape ops ts, p2, Array.append ops ts)
  end
  else
    (* The widest padded operand, by its descriptor's words: a plain one
       copied reads as many. *)
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
      match map_launch copy ~shape [| ops.(k) |] ts with
      | c ->
          let ops = Array.copy ops in
          ops.(k) <- ts.(0);
          Some (c, p, ops)
      | exception Too_wide -> None

(* The passes of [p] over [ops], [final]'s where they fit a launch, else
   in parts, then planned. Each part leaves fewer live computed nodes, or
   as many and one fewer padded load, so the parts end.

   [part] never answers [None], so this never declines. A loop's rank is
   at most 32 ({!L.max_rank}), so an operand's descriptor takes at most
   35 words, a padded one's 230 (6, then 7 a read, a read an axis). With a
   padded load, its copy fits: 32 header words, 2 address words, 64
   extent words (an extent and its divisor each), 230 and 35, and a tape
   of a load and a store, under
   {!K.loop_words}' 512, in 2 values' planes. With every load plain and a
   live computed node, the first cut's part fits: one node over at most 3
   operands, 4 descriptors and a few instructions, in at most 18 planes,
   18 KiB at one element a work-item. With neither, every output is an
   [In], a [Coord] or a [Const], and [final], which splits outputs and
   reductions down to one and compacts each to the loads it reads,
   launches each over at most one plain load and a scan's carry: at most 4
   descriptors, in planes that depend on the dtypes alone, which the
   suites launch at every dtype. So [final] does not raise there. *)
let parted (p : P.t) ~shape ops final =
  let s = stand_ins () in
  let rec go p ops parts =
    match final s p ops with
    | cs -> Ok (plan s (List.rev_append parts cs))
    | exception Too_wide -> (
        match part s p ~shape ops with
        | None -> Error A.Declined
        | Some (c, p, ops) -> go p ops (c :: parts))
  in
  go p ops []

let map (p : P.t) ~(dsts : A.any iarray) (loads : Nx_kernel.load iarray) =
  let (A.Any y0) = Iarray.get dsts 0 in
  let shape = shape_of y0 in
  if Array.exists (( = ) 0) shape then Ok None
  else
    let outs = mapped (fun (A.Any y) -> plain y) dsts in
    parted p ~shape (mapped of_load loads) (fun s p ops ->
        map_passes s p ~shape ops outs)

(* Reductions and scans *)

(* A reduction as loop.hip folds it. *)
let reduction_code : Nx_kernel.reduction -> int = function
  | Monoid Sum -> K.loop_sum
  | Monoid Prod -> K.loop_prod
  | Monoid Max -> K.loop_max
  | Monoid Min -> K.loop_min
  | Arg Max -> K.loop_arg_max
  | Arg Min -> K.loop_arg_min
  | Logsumexp -> K.loop_logsumexp
  | Moments -> K.loop_moments

(* A reduction's pass: its fold, what it does with each output's state,
   and whether its terms read a center. *)
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

(* Fixed planes of a reduction or a scan: the term, the state where
   [state] (a spread reduction's lie past the planes), the results, the
   center. *)
type fixed = { term : int; state : int; result : int; center : int }

let fixed ?(state = true) t ~carrier ~center ~result =
  let term = alloc t.pl (words carrier) in
  let state = if state then alloc t.pl K.loop_state else 0 in
  let result = if result then alloc t.pl 6 else 0 in
  let center = if center then alloc t.pl 2 else 0 in
  { term; state; result; center }

(* Lowers output [k] of [p] into [t]'s body, its value cast into the term
   plane in [carrier], a boolean as true or false. *)
let lower_term t p k ~carrier ~axis ~eager ~(f : fixed) =
  lower t p ~roots:[ k ] ~load:Fun.id ~axis ~eager ~out:(fun _ v ->
      if boolean carrier then
        ins t.body K.loop_truth ~d:f.term ~a:v.plane ~b:0 ~c:0
      else cast t.body ~d:f.term ~a:v.plane ~from:v.dt ~into:carrier)

let reduction_word ~rk ~ct ~carrier ~fin =
  rk lor (ct lsl 8) lor (code carrier lsl 16) lor (fin lsl 24)

(* The bytes of a state in the workspace. *)
let state_bytes = 4 * K.loop_state

(* A reduction launch over the walk [w]: [rows] rows of [terms] terms, cut
   into segments of [seg], of output [k] of [p], by [fold]. Address words:
   the loads, the destinations, the workspace, the tickets ([nslots] in
   all); a centered fold reads its centers through [center], a load of the
   workspace. Its partial states start at byte [partials] of its pass's
   workspace; it also gives the byte their end. *)
let reduce_launch (p : P.t) w ops dsts ~dst_slots ~nslots ~targets ~k ~carrier
    ~fold ~rows ~terms ~seg ~partials ~centers ~center =
  let nl = Array.length ops in
  let work = nslots - 2 and tickets = nslots - 1 in
  let segs = ceil_div terms seg in
  let outputs = rows * segs in
  let len = Int.min seg (Int.min terms K.loop_block) in
  let nb = ceil_div (Int.min seg terms) K.loop_block in
  (* Spread: a block's lanes on 16 work-items or more, their states past
     the planes, for blocks of {!lanes_least} terms or more. *)
  let spread = len >= lanes_least in
  let build ~eager =
    let t = tape () in
    let f =
      fixed t ~state:(not spread) ~carrier ~center:fold.centered ~result:false
    in
    if fold.centered then ins t.body K.loop_load ~d:f.center ~a:nl ~b:0 ~c:0;
    lower_term t p k ~carrier ~axis:(kernel_axis w) ~eager ~f;
    let v =
      if spread then elements ~most:K.loop_most ~extra:lane_states t.pl.top
      else elements t.pl.top
    in
    (t, f, v)
  in
  let t, f, v = fitted build ~needed:(fun _ -> false) in
  let ops, slots =
    match center with
    | Some c when fold.centered ->
        (Array.append ops [| c |], fun i -> if i = nl then work else i)
    | _ -> (ops, Fun.id)
  in
  let rec pow2 n x = if x >= n then x else pow2 n (2 * x) in
  (* A layout of [tb] work-items a block: its blocks a group, an output's
     groups and the launch's groups. *)
  let layout tb =
    let lb = K.loop_threads / tb in
    let g = Int.min lb (pow2 nb 1) in
    let m = ceil_div nb g in
    (tb, g, m, ceil_div (outputs * m) (lb / g))
  in
  (* A block's work-items: 16 / v, each holding v lanes, fewer where its
     terms fill fewer lanes; spread, 16, then 32, 64, ... up to a group
     while the launch's groups still fit {!group_slots}, a run holds no
     more terms than a block and a group still holds as many of an
     output's blocks. *)
  let rec wider ((tb, _, m, _) as l) =
    let next = 2 * tb in
    if next > K.loop_threads || next * v > pow2 len 1 then l
    else
      let ((_, _, m', groups) as l') = layout next in
      if m' > m || groups > group_slots then l else wider l'
  in
  let tb, g, m, _ =
    if spread then wider (layout K.loop_lanes)
    else layout (Int.max 1 (Int.min K.loop_lanes (pow2 len 1) / v))
  in
  let h = header v in
  h.(K.loop_mode) <- h.(K.loop_mode) lor (tb lsl 16);
  seti h K.loop_term f.term;
  seti h K.loop_state_at (if spread then t.pl.top * v else f.state);
  seti h K.loop_center f.center;
  seti h K.loop_reduction
    (reduction_word ~rk:fold.rk ~ct:(compute carrier) ~carrier ~fin:fold.fin);
  seti h K.loop_terms terms;
  seti h K.loop_segment seg;
  seti h K.loop_segments segs;
  seti h K.loop_outputs outputs;
  seti h K.loop_group g;
  seti h K.loop_groups m;
  seti h K.loop_runs (ceil_div len (tb * v));
  seti h K.loop_work work;
  seti h K.loop_tickets tickets;
  seti h K.loop_partials partials;
  seti h K.loop_totals 0;
  seti h K.loop_centers centers;
  let ws = params ~nslots w ops ~slots dsts ~dst_slots t h in
  (* An output of several groups leaves their states in the workspace and
     takes a ticket. *)
  let work_end, ticket_words =
    if m > 1 then (partials + (outputs * m * state_bytes), outputs)
    else (partials, 0)
  in
  ( launch_of K.Loop_reduce ws ~targets
      ~groups:
        (Plan.grid ~per:(K.loop_threads / tb / g) K.Loop_reduce (outputs * m))
      ~planes:(shared_bytes t.pl.top v + if spread then lane_states else 0)
      ~tickets:ticket_words,
    work_end )

(* The center of each output, a load of the workspace from byte [byte]:
   [carrier]'s values, one an output, C-contiguous over the kept axes, the
   same along the reduced ones. *)
let center_load w ~buffer ~carrier ~byte =
  let n = Array.length w.ext in
  let s = Array.make n 0 in
  let acc = ref 1 in
  for i = w.kept - 1 downto 0 do
    s.(i) <- !acc;
    acc := !acc * w.ext.(i)
  done;
  {
    buffer;
    dtype = carrier;
    offset = byte / (words carrier * 4);
    access = Strided s;
  }

(* Plain reductions and scans *)

(* The plain fold of [r] at output [k] of [p] over [ops], loop.hip's code
   of its reduction and its carrier: where [r] is a Sum, Max or Min whose
   terms are the one plain load [ops.(0)] at its carrier, an integer or
   float of 32 or 64 bits, or, where [narrow], a bfloat16 or float16 into
   float32, over a walk [w] of at most one kept axis and one reduced
   one. *)
let plain_fold ?(narrow = false) (p : P.t) w (ops : operand array) ~k
    ~(r : Nx_kernel.reduction) =
  let carrier = Nx_kernel.carrier (P.dtype p (Iarray.get p.outs k)) in
  let rank = Array.length w.ext in
  let rk =
    match r with
    | Monoid Sum -> K.loop_sum
    | Monoid Max -> K.loop_max
    | Monoid Min -> K.loop_min
    | _ -> -1
  in
  let typed =
    match carrier with
    | D.Any (D.Float32 | D.Float64 | D.Int32 | D.Uint32 | D.Int64 | D.Uint64) ->
        true
    | D.Any _ -> false
  in
  match (Iarray.get p.nodes (Iarray.get p.outs k), ops) with
  | P.In 0, [| { access = Strided _; dtype; _ } |]
    when rk >= 0 && typed && rank > 0 && w.kept <= 1
         && rank = w.kept + 1
         && (dtype = carrier
            || (narrow && (dtype = D.Any D.Bfloat16 || dtype = D.Any D.Float16))
            ) ->
      Some (rk, carrier)
  | _ -> None

(* The walk [w]'s rows (its kept axis, else 1) and terms. *)
let rows_terms w =
  let rank = Array.length w.ext in
  ((if rank = 2 then w.ext.(0) else 1), w.ext.(rank - 1))

(* The strides of the plain operand [o] along [w]'s rows and terms. *)
let row_term_strides w (o : operand) =
  match o.access with
  | Strided s when Array.length w.ext = 2 -> (s.(0), s.(1))
  | Strided s -> (0, s.(0))
  | Pad _ -> invalid_arg "Loop.row_term_strides"

(* A plain reduction's launch (loop.hip's Plain reductions) of [rk] at
   [carrier] over [ops.(0)] into [dsts], its rows' segments of [seg]
   terms, [fin] their outputs' put: rows where its terms step by one,
   columns where its rows do and a segment holds at most
   {!K.loop_plain_tall} blocks; else none. Rows: a group takes up to 16
   blocks of an output, or one block where a reduction's blocks are
   fewer than a group's work-items, which spreads them over as many
   groups; a scan's chunk totals do not spread, so a scan's workspace
   stays its totals. An output of more blocks than its group takes
   several groups, whose values lie in the workspace from byte
   [partials], and a ticket.
   Any power-of-two blocks a group give the same pairwise tree.
   Columns: a group takes a segment of 32 rows, all its blocks. It also
   gives the byte the workspace's values end. *)
let plain_launch w ops dsts ~nslots ~targets ~rk ~carrier ~seg ~fin
    ~partials =
  let rows, terms = rows_terms w in
  let sr, st = row_term_strides w ops.(0) in
  let segs = ceil_div terms seg in
  let nb = ceil_div (Int.min seg terms) K.loop_block in
  let columns = st <> 1 in
  if columns && (sr <> 1 || nb > K.loop_plain_tall) then None
  else
    let outputs = rows * segs in
    let spread =
      (not columns) && fin = K.loop_put && outputs * nb < K.loop_threads
    in
    let cells = if spread then 1 else K.loop_threads / K.loop_lanes in
    let rec pow2 x = if x >= nb then x else pow2 (2 * x) in
    let g = if columns then 1 else Int.min cells (pow2 1) in
    let m = if columns then 1 else ceil_div nb g in
    let h = header 1 in
    seti h K.loop_reduction
      (reduction_word ~rk ~ct:(compute carrier) ~carrier ~fin);
    seti h K.loop_terms terms;
    seti h K.loop_segment seg;
    seti h K.loop_segments segs;
    seti h K.loop_outputs outputs;
    seti h K.loop_group g;
    seti h K.loop_groups m;
    seti h K.loop_cells cells;
    seti h K.loop_work (nslots - 2);
    seti h K.loop_tickets (nslots - 1);
    seti h K.loop_partials partials;
    seti h K.loop_totals 0;
    let ws =
      params ~nslots w ops ~slots:Fun.id dsts
        ~dst_slots:(fun j -> Array.length ops + j)
        (tape ()) h
    in
    let groups, values =
      if columns then
        (Plan.grid ~per:1 K.Loop_plain (ceil_div rows 32 * segs), 32 * (8 + nb))
      else
        ( Plan.grid ~per:(cells / g) K.Loop_plain (outputs * m),
          (K.loop_threads / K.loop_lanes)
          + if m > 1 then K.loop_plain_chunk else 0 )
    in
    let several = m > 1 in
    Some
      ( launch_of K.Loop_plain ws ~targets ~groups ~planes:(8 * values)
          ~tickets:(if several then outputs else 0),
        if several then partials + (8 * outputs * m) else partials )

(* A plain scan's launches of [rk] at [carrier] over [ops.(0)] into
   [outs.(0)] of the carrier's dtype, its carry, if any, the load [carry],
   and the workspace's bytes they take. A work-item scans a chunk
   (loop.hip's Plain scans), a group taking as many work-items as keep
   {!group_slots} groups busy, 32 to all; past one chunk a slice, the
   chunks' totals and their carries first, into the workspace from byte
   0. Where terms and results both step by one, the scan tiles: each of
   a group's waves takes {!K.loop_tile} bytes of LDS. *)
let plain_scan w ops outs ~nslots ~targets ~rk ~carrier ~carry =
  let rows, terms = rows_terms w in
  let chunks = ceil_div terms K.loop_chunk in
  let units = rows * chunks in
  let steps_one o = snd (row_term_strides w o) = 1 in
  let tiled = steps_one ops.(0) && steps_one outs.(0) in
  let launch kernel ~per ~units ~group ~planes =
    let h = header 1 in
    seti h K.loop_reduction
      (reduction_word ~rk ~ct:(compute carrier) ~carrier ~fin:K.loop_put);
    seti h K.loop_terms terms;
    seti h K.loop_segment K.loop_chunk;
    seti h K.loop_segments chunks;
    seti h K.loop_outputs rows;
    seti h K.loop_group group;
    seti h K.loop_work (nslots - 2);
    seti h K.loop_tickets (nslots - 1);
    seti h K.loop_totals 0;
    seti h K.loop_carry carry;
    seti h K.loop_tiled (Bool.to_int (planes > 0));
    let ws =
      params ~nslots w ops ~slots:Fun.id outs
        ~dst_slots:(fun j -> Array.length ops + j)
        (tape ()) h
    in
    (* Plan.grid's band step counts a group's 256 work-items, where a
       scan's group has [a] at work and the rest return at once: it steps
       neither kernel, both above 96 VGPRs. *)
    launch_of kernel ws ~targets ~groups:(Plan.grid ~per kernel units)
      ~planes ~tickets:0
  in
  let rec most a =
    if a < K.loop_threads && units / (2 * a) >= group_slots then most (2 * a)
    else a
  in
  let a = most 32 in
  let planes = if tiled then K.loop_tile * (a / K.loop_wave) else 0 in
  let last = launch K.Loop_plain_scan ~per:a ~units ~group:a ~planes in
  if chunks = 1 then Some { launches = [ last ]; work = 0 }
  else
    match
      plain_launch w ops outs ~nslots ~targets ~rk ~carrier ~seg:K.loop_chunk
        ~fin:K.loop_put_total ~partials:(align16 (8 * units))
    with
    | None -> None
    | Some (totals, work) ->
        let carries =
          launch K.Loop_plain_carries ~per:K.loop_threads ~units:rows ~group:1
            ~planes:0
        in
        Some { launches = [ totals; carries; last ]; work }

let order_of rank ~reduced =
  let kept =
    List.filter (fun a -> not (List.mem a reduced)) (List.init rank Fun.id)
  in
  (Array.of_list (kept @ reduced), List.length kept)

let destinations : Nx_kernel.reduction -> int = function
  | Arg _ | Moments -> 2
  | Monoid _ | Logsumexp -> 1

(* One pass of the reductions [rs] of [p] over [ops] into [outs], in its
   turn each. Its workspace holds each output's center, [carrier]'s value
   in 8 bytes, where a reduction takes one, then the partial states. *)
let reduce_pass (p : P.t) ~shape ~axes rs ops outs =
  let rank = Array.length shape in
  let p, ops = compact p ~roots:(List.map snd (Array.to_list rs)) ops in
  let order, kept = order_of rank ~reduced:(Iarray.to_list axes) in
  let merge = not (reads_coord p || Array.exists is_padded ops) in
  let w, wops = walk shape order ~kept ~merge ops in
  let rows = product w.ext 0 w.kept in
  let terms = product w.ext w.kept (Array.length w.ext) in
  let nl = Array.length ops and nd = Array.length outs in
  let nslots = nl + nd + 2 in
  let targets = targets ~shared:true ops outs in
  let centered =
    Array.exists
      (fun ((r : Nx_kernel.reduction), _) -> r = Logsumexp || r = Moments)
      rs
  in
  let plain =
    match rs with
    | [| (r, k) |] -> (
        match plain_fold ~narrow:true p w wops ~k ~r with
        | Some (rk, carrier) ->
            plain_launch w wops outs ~nslots ~targets ~rk ~carrier ~seg:terms
              ~fin:K.loop_put ~partials:0
        | None -> None)
    | _ -> None
  in
  match plain with
  | Some (l, work) -> { launches = [ l ]; work }
  | None ->
      let partials = if centered then align16 (8 * rows) else 0 in
      let bytes = ref partials and first = ref 0 in
      let launches =
        List.concat_map
          (fun (r, k) ->
            let carrier = Nx_kernel.carrier (P.dtype p (Iarray.get p.outs k)) in
            let d0 = !first and n = destinations r in
            first := d0 + n;
            let center =
              Some (center_load w ~buffer:outs.(0).buffer ~carrier ~byte:0)
            in
            List.map
              (fun fold ->
                let l, e =
                  reduce_launch p w wops (Array.sub outs d0 n)
                    ~dst_slots:(fun j -> nl + d0 + j)
                    ~nslots ~targets ~k ~carrier ~fold ~rows ~terms ~seg:terms
                    ~partials ~centers:0 ~center
                in
                bytes := Int.max !bytes e;
                l)
              (folds r))
          (Array.to_list rs)
      in
      { launches; work = !bytes }

(* A reduction's passes: one, else its reductions split in halves. *)
let rec reduce_passes p ~shape ~axes rs ops outs =
  match reduce_pass p ~shape ~axes rs ops outs with
  | c -> [ c ]
  | exception Too_wide when Array.length rs > 1 ->
      let h = Array.length rs / 2 in
      let first = Array.sub rs 0 h
      and rest = Array.sub rs h (Array.length rs - h) in
      let d = Array.fold_left (fun n (r, _) -> n + destinations r) 0 first in
      reduce_passes p ~shape ~axes first ops (Array.sub outs 0 d)
      @ reduce_passes p ~shape ~axes rest ops
          (Array.sub outs d (Array.length outs - d))

let reduce (p : P.t) ~shape ~axes (rs : (Nx_kernel.reduction * int) iarray)
    ~(dsts : A.any iarray) (loads : Nx_kernel.load iarray) =
  let shape = Iarray.to_array shape and rs = Iarray.to_array rs in
  if Array.exists (( = ) 0) shape then Ok None
  else
    let outs = mapped (fun (A.Any y) -> plain y) dsts in
    parted p ~shape (mapped of_load loads) (fun _ p ops ->
        reduce_passes p ~shape ~axes rs ops outs)

(* Scans *)

(* The work-items that load a chunk of a scan whose results step by one
   along the slice, so that its loads are coalesced; else one, each
   work-item a chunk, so that slices side by side load side by side. *)
let chunk_items = 16

(* The log₂ of a network's slot of a slice of [terms] at [v] elements a
   work-item: its terms rounded up to a power of two where they fit a
   group's work-items, a run's slots then holding several slices, else a
   run's elements, a slice then taking runs in order (loop.hip). *)
let network_slot ~terms v =
  let n = if terms <= K.loop_threads then terms else v * K.loop_threads in
  let rec log2_up b = if 1 lsl b >= n then b else log2_up (b + 1) in
  log2_up 0

(* A scan's launch of [kernel], over runs of its terms: the body's terms,
   then the epilogue's stores of the results into [outs]. A unit is a
   chunk of a slice of [s] work-items ([Loop_scan]), or a run's slots of
   slices ([Loop_network]). *)
let scan_launch (p : P.t) w ops outs ~dst_slots ~nslots ~targets ~k ~carrier
    ~kernel ~rk ~units ~terms ~chunks ~carry ~partials ~s
    ~(r : Nx_kernel.reduction) =
  let build ~eager =
    let t = tape () in
    let f = fixed t ~state:false ~carrier ~center:false ~result:true in
    lower_term t p k ~carrier ~axis:(kernel_axis w) ~eager ~f;
    ignore (elements t.pl.top);
    (t, f)
  in
  let t, f = fitted build ~needed:(fun _ -> false) in
  (* The body's values are dead once the results are folded: the
     epilogue's temporaries take any free planes. *)
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
  store 0 ~plane:f.result ~from:carrier;
  (match r with
  | Arg _ -> store 1 ~plane:(f.result + 4) ~from:int64
  | Moments -> store 1 ~plane:(f.result + 4) ~from:carrier
  | Monoid _ | Logsumexp -> ());
  let v = elements t.pl.top in
  let h = header v in
  seti h K.loop_term f.term;
  seti h K.loop_result f.result;
  seti h K.loop_reduction
    (reduction_word ~rk ~ct:(compute carrier) ~carrier ~fin:K.loop_put);
  seti h K.loop_terms terms;
  seti h K.loop_segments chunks;
  seti h K.loop_outputs units;
  seti h K.loop_group s;
  let network = kernel = K.Loop_network in
  let n = network_slot ~terms v in
  seti h K.loop_segment (if network then n else K.loop_chunk);
  seti h K.loop_runs
    (if network then ceil_div terms (1 lsl n)
     else ceil_div (Int.min terms K.loop_chunk) (s * v));
  seti h K.loop_work (nslots - 2);
  seti h K.loop_tickets (nslots - 1);
  seti h K.loop_partials partials;
  seti h K.loop_totals 0;
  seti h K.loop_carry carry;
  let ws = params ~nslots w ops ~slots:Fun.id outs ~dst_slots t h in
  let groups =
    if network then
      Plan.grid ~per:1 kernel (ceil_div units ((v * K.loop_threads) lsr n))
    else Plan.grid ~per:(K.loop_threads / s) kernel units
  in
  launch_of kernel ws ~targets ~groups ~planes:(shared_bytes t.pl.top v)
    ~tickets:0

(* The chunks of a slice from which a group, not a work-item, folds its
   totals: one work-item's chain of loads would wait on each, a group's
   loads them side by side into LDS first. *)
let group_chunks = 16

(* The launch that turns each slice's chunk totals into the states its
   chunks start from, a work-item or a group a slice. *)
let carries_launch w ops outs ~dst_slots ~nslots ~targets ~carrier ~rk ~slices
    ~terms ~chunks ~carry =
  let h = header 1 in
  let grouped = chunks >= group_chunks in
  seti h K.loop_group (if grouped then K.loop_threads else 1);
  seti h K.loop_reduction
    (reduction_word ~rk ~ct:(compute carrier) ~carrier ~fin:K.loop_put);
  seti h K.loop_terms terms;
  seti h K.loop_segments chunks;
  seti h K.loop_outputs slices;
  seti h K.loop_work (nslots - 2);
  seti h K.loop_tickets (nslots - 1);
  seti h K.loop_totals 0;
  seti h K.loop_carry carry;
  let ws = params ~nslots w ops ~slots:Fun.id outs ~dst_slots (tape ()) h in
  let per = if grouped then 1 else K.loop_threads in
  launch_of K.Loop_carries ws ~targets
    ~groups:(Plan.grid ~per K.Loop_carries slices)
    ~planes:0 ~tickets:0

(* A scan's pass. Address words: the loads, a carry, the destinations,
   the workspace, the tickets. A monoid's or Arg's scan of several chunks
   a slice first reduces each chunk into the workspace, its total, then
   folds a slice's totals in order from its carry into the states its
   chunks start from. A network's take no workspace. *)
let scan_pass ?carry (p : P.t) ~shape ~axis ((r : Nx_kernel.reduction), k) ops
    outs =
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
  let carrier = Nx_kernel.carrier (P.dtype p (Iarray.get p.outs k)) in
  let nslots = nl + nd + 2 in
  let targets = targets ~shared:true ops outs in
  let carry = if Option.is_some carry then nl - 1 else -1 in
  let dst_slots j = nl + j and rk = reduction_code r in
  let plain =
    match (r, wouts) with
    | Monoid _, [| { access = Strided _; dtype; _ } |] when dtype = carrier -> (
        let loads = Array.sub wops 0 (if carry >= 0 then nl - 1 else nl) in
        match plain_fold p w loads ~k ~r with
        | Some (rk, carrier) ->
            plain_scan w wops wouts ~nslots ~targets ~rk ~carrier ~carry
        | None -> None)
    | _ -> None
  in
  match (plain, r) with
  | Some pass, _ -> pass
  | None, (Logsumexp | Moments) ->
      let l =
        scan_launch p w wops wouts ~dst_slots ~nslots ~targets ~k ~carrier
          ~kernel:K.Loop_network ~rk ~units:slices ~terms ~chunks:1 ~carry
          ~partials:0 ~s:K.loop_threads ~r
      in
      { launches = [ l ]; work = 0 }
  | None, (Monoid _ | Arg _) ->
      let chunks = ceil_div terms K.loop_chunk in
      let units = slices * chunks in
      let totals = units * state_bytes in
      let first, work =
        if chunks = 1 then ([], 0)
        else
          (* The totals' partials, if any, follow the totals. *)
          let l, e =
            reduce_launch p w wops wouts ~dst_slots ~nslots ~targets ~k
              ~carrier
              ~fold:{ rk; fin = K.loop_put_total; centered = false }
              ~rows:slices ~terms ~seg:K.loop_chunk ~partials:(align16 totals)
              ~centers:0 ~center:None
          in
          ( [
              l;
              carries_launch w wops wouts ~dst_slots ~nslots ~targets ~carrier
                ~rk ~slices ~terms ~chunks ~carry;
            ],
            e )
      in
      (* Coalesced where the results step by one along the slice. *)
      let rank = Array.length w.ext in
      let s =
        match wouts.(0).access with
        | Strided st when terms >= chunk_items && st.(rank - 1) = 1 ->
            chunk_items
        | Strided _ | Pad _ -> 1
      in
      let last =
        scan_launch p w wops wouts ~dst_slots ~nslots ~targets ~k ~carrier
          ~kernel:K.Loop_scan ~rk ~units ~terms ~chunks ~carry ~partials:0 ~s
          ~r
      in
      { launches = first @ [ last ]; work }

(* A carry's stride along [axis] is 0, so that every index of a slice
   reads it. *)
let scan ?carry (p : P.t) ~axis rk ~(dsts : A.any iarray)
    (loads : Nx_kernel.load iarray) =
  let (A.Any y0) = Iarray.get dsts 0 in
  let shape = shape_of y0 in
  let carry =
    Option.map
      (fun (A.Any c) ->
        let o = plain c in
        match o.access with
        | Strided s ->
            let s = Array.copy s in
            s.(axis) <- 0;
            { o with access = Strided s }
        | Pad _ -> o)
      carry
  in
  if Array.exists (( = ) 0) shape then Ok None
  else
    let outs = mapped (fun (A.Any y) -> plain y) dsts in
    parted p ~shape (mapped of_load loads) (fun _ p ops ->
        [ scan_pass ?carry p ~shape ~axis rk ops outs ])
