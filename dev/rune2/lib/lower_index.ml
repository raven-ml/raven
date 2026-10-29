(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Tolk
module D = Nx_array.Dtype
module Arith = Lower_arith
module Bits = Lower_bits

let unlowered what =
  invalid_arg (Printf.sprintf "Rune.Lower.op: %s is not lowered" what)

(* Work

   A scatter meets each update at every position along its axis, [n·k] work
   against eager's [n + k], and an assembly selects each of its [p] pieces at
   every element, [p·N] against eager's [N + m] for pieces of [m] elements in
   all. Past [work_bound] times eager's work, a scatter combines in a loop of
   its own (below), and an assembly is refused: the jit then evaluates its
   program uncompiled. The bound is a cost model's, not yet timed: below it
   the extra work is a few comparisons and selects per unit on elements the
   kernel holds, which the model takes as about what moving each element
   through memory costs a host core, and as hidden on a GPU.
   Past it the work outgrows what compiling the rest of the program saves: an
   embedding's gradient, 512 updates over 50000 rows, would make 2.6e7
   comparisons per feature against eager's 5e4. *)
let work_bound = 16

let extents u = List.init (Shape.ndim u) (Bits.extent u)
let dims s = List.map (fun n -> Ops.Int n) s
let numel s = List.fold_left ( * ) 1 s
let ceil_div a b = (a + b - 1) / b
let int u n = Shape.const_like u (`Int (Bigint.of_int n))

let broadcast c s =
  Shape.expand (Shape.reshape c (List.map (fun _ -> Ops.Int 1) s)) (dims s)

let zeros t s = broadcast (Ops.const ~dtype:t (`Int Bigint.zero)) s

(* [laid s axis v] is the vector [v] laid along [axis] of the shape [s]. *)
let laid s axis v =
  let r = List.length s in
  let along = List.init r (fun i -> if i = axis then List.nth s i else 1) in
  Shape.expand (Shape.reshape v (dims along)) (dims s)

(* [joined axis a b] is [a] then [b] along [axis]. Floats join as their bits:
   tolk's Shape.cat of pieces of unequal extents along the axis adds them, each
   zero-padded to the whole, and adding the padding's [+0] turns a [-0] into
   [+0] and quiets a signalling NaN. A Shape.cat that selected each element
   from its piece would let floats join as they are. *)
let joined axis a b =
  match Ops.dtype a with
  | t when Dtype.is_float t ->
      let u = Bits.unsigned t in
      Ops.bitcast (Shape.cat ~axis (Ops.bitcast a u) [ Ops.bitcast b u ]) t
  | _ -> Shape.cat ~axis a [ b ]

(* [negative_zero u] is whether the float [u] is [-0]: its sign bit alone. *)
let negative_zero u =
  let b = Bits.bits u in
  let w = 8 * Dtype.itemsize (Ops.dtype u) in
  Ops.eq b (Shape.const_like b (`Int (Bigint.shift_left Bigint.one (w - 1))))

(* Gathers

   A gather is tolk's index of the operand's elements, flattened, at each
   element's offset: its own index's along the other axes, its position's along
   [axis]. A position outside the axis is read at [0], so that every read is in
   bounds, and selected as zero. *)

let take u axis p =
  let s = extents u and t = extents p in
  let n = List.nth s axis in
  if n = 0 || numel t = 0 then zeros (Ops.dtype u) t
  else
    let inside =
      Ops.lt (Ops.bitcast p Uint64)
        (Shape.const_like ~dtype:Uint64 p (`Int (Bigint.of_int n)))
    in
    let at = Ops.cast (Ops.where inside p (int p 0)) Weak_int in
    let stride d = numel (List.filteri (fun i _ -> i > d) s) in
    let term d =
      let v =
        if d = axis then at
        else laid t d (Shape.arange ~dtype:Weak_int (List.nth t d))
      in
      match stride d with 1 -> v | k -> Ops.mul v (Ops.int k)
    in
    let offset =
      List.fold_left
        (fun o d -> Ops.add o (term d))
        (term 0)
        (List.init (List.length s - 1) succ)
    in
    let read = Ops.index (Shape.reshape u [ Ops.Int (numel s) ]) [ offset ] in
    Ops.where inside read (int read 0)

let gather d ~axis idx x =
  Arith.of_parts d (List.map (fun u -> take u axis idx) (Arith.parts d x))

(* Scatters

   Each element of [into] meets each update along [axis] in a mask along a new
   last axis, true where the update's position is the element's: n x k work for
   n positions and k updates along [axis]. Updates that land on one element
   differ only along [axis], so C order of updates is their order along it. *)

(* [summed ca ~landed xs us] is, in the carrier [ca], the sum of each element of
   [xs] and the updates [us] that land on it, [landed] their mask along a last
   axis. It is [-0] where every term is, which a sum from tolk's [+0]
   accumulator would make [+0]. *)
let summed ca ~landed xs us =
  let r = Shape.ndim landed - 1 in
  let terms x u = joined r x (Ops.where landed u (int u 0)) in
  let sums =
    Lower_reduce.reduce (Monoid Sum) [ r ] ~src:ca ca
      (Arith.of_parts ca (List.map2 terms xs us))
  in
  let signed (x, u) y =
    if not (Dtype.is_float (Ops.dtype y)) then y
    else
      let other =
        Ops.bitwise_and landed (Ops.logical_not (negative_zero u))
      in
      let other = Shape.rop (Ops.cast other Uint8) Op.Max [ r ] in
      let x = Shape.reshape x (Shape.shape other) in
      let none = Ops.eq other (int other 0) in
      let all = Ops.bitwise_and (negative_zero x) none in
      Ops.where all (Shape.const_like y (`Float (-0.))) y
  in
  Arith.of_parts ca
    (List.map2 signed (List.combine xs us) (Arith.parts ca (List.hd sums)))

(* Looped scatters

   Past [work_bound], a scatter updates in place, in a kernel of its own,
   storage that holds [into]: its caller's [dst], such as a fresh buffer that a
   kernel stored [into] into, one element per thread, or a donated target's own
   storage, which needs no copy. [into] is [p·n·q] elements, an axis of [n]
   between [p] and [q], and the updates [p·k·q]. The axis is cut into runs of
   positions. For each index of the axes before the axis and each run, a thread
   walks the [k] updates in C order and combines each that lands in its run
   with the element it lands on: every element meets its updates in eager's
   order, and no two threads write one element. Work is [runs·k + n] at each
   index of the other axes. On the host, the axes after the axis are a serial
   loop inside the walk, so that an update writes a contiguous row, and
   positions an expansion repeats along them are read once a row. On a GPU they
   are threads too, adjacent ones at adjacent addresses.

   The host takes one run: at 1024 rows of 64 floats into 4096, one thread's
   walk takes 6 µs, and 16 runs split among its threads take 37 µs. A GPU takes
   up to [gpu_runs], which bounds the walks to [gpu_runs·k].

   A narrow float's Add sums in a [float32] copy and marks where updates land
   in a byte per element, so that elsewhere the result keeps [into]'s bits,
   which a trip through [float32] quiets. *)

let gpu_runs = 16

(* [spread u] is the axes along which the view [u] broadcasts its source: those
   an expansion adds, moved by permutations. *)
let rec spread u =
  match Ops.op u with
  | Op.Expand -> (
      match Shape.marg u with
      | Expand front ->
          let m = List.length front in
          List.init m Fun.id @ List.map (( + ) m) (spread (Ops.nth u 0))
      | _ -> [])
  | Op.Permute -> (
      match Shape.marg u with
      | Permute perm ->
          let inner = spread (Ops.nth u 0) in
          let moved i a = if List.mem a inner then [ i ] else [] in
          List.concat (List.mapi moved perm)
      | _ -> [])
  | _ -> []

(* [walk ~host ~rows ~w ~p ~n ~q ~k f args] is the kernel over [args] =
   [y; i; u] or [y; i; u; hit] that combines by [f] each update of [u], at its
   position in [i], into [y], [w] parts an element, and sets [hit] to [1] where
   one lands. Where [rows], [i] holds one position per row of [q] updates. *)
let walk ~host ~rows ~w ~p ~n ~q ~k f args =
  let len = if host then n else ceil_div n (Int.min n gpu_runs) in
  let runs = ceil_div n len in
  let range t e id = Ops.range ~axis_type:t (Ops.Int e) [ id ] in
  let a = range Weak p 0 and g = range Weak runs 1 in
  let j, b =
    if host then (range Reduce k 2, range Reduce q 3)
    else (range Reduce k 3, range Weak q 2)
  in
  let y, i, u, hit =
    match args with
    | [ y; i; u ] -> (y, i, u, None)
    | [ y; i; u; h ] -> (y, i, u, Some h)
    | _ -> assert false
  in
  let open Ops.O in
  let row = (a * int k) + j in
  let update = (row * int q) + b in
  let pos = Ops.load (Ops.index i [ (if rows then row else update) ]) [] in
  let u64 v = Ops.const ~dtype:Uint64 (`Int (Bigint.of_int v)) in
  let first = Ops.cast (g * int len) Int64 in
  (* Unsigned, a negative position lies past every bound, and [pos - first]
     wraps where it would overflow. *)
  let upos = Ops.bitcast pos Uint64 in
  let ufirst = Ops.cast (g * int len) Uint64 in
  let run = Ops.bitwise_and (upos < u64 n) (upos - ufirst < u64 len) in
  (* In the run, [pos] is [first + pos mod len], an address that names the
     run's range. An update outside the run reads and stores nothing: its index
     is invalid. *)
  let within = Ops.bitcast (upos % u64 len) Int64 in
  let at = Ops.cast (first + within) Weak_int in
  let e = (((a * int n) + at) * int q) + b in
  let part m o c =
    Ops.index m [ Ops.where run ((o * int w) + int c) Ops.invalid ]
  in
  let state = Ops.after y (if host then [ j; b ] else [ j ]) in
  let olds = List.init w (fun c -> Ops.load (part state e c) []) in
  let us = List.init w (fun c -> Ops.load (part u update c) []) in
  let stores = List.mapi (fun c v -> Ops.store (part y e c) v) (f olds us) in
  let marks =
    match hit with
    | None -> []
    | Some h ->
        let one = Ops.const ~dtype:Uint8 (`Int Bigint.one) in
        [ Ops.store (part h e 0) one ]
  in
  let body = Ops.group (stores @ marks) in
  let ends =
    if host then Ops.end_ (Ops.end_ (Ops.end_ body [ b ]) [ j ]) [ a; g ]
    else Ops.end_ (Ops.end_ body [ j ]) [ a; g; b ]
  in
  let kernel = Ops.kernel_info ~name:"scatter" ~opts_to_apply:[] () in
  Ops.sink ~kernel [ ends ]

(* [fresh device v] is a buffer of [device] once [v] is stored into it. *)
let fresh device v =
  let b = Ops.new_buffer device (Shape.max_numel v) (Ops.dtype v) in
  Ops.after b [ Ops.store b v ]

let looped ~dst ~host device (D.Any dt as d) (c : Nx_kernel.combine) ~axis
    ~idx ~updates into =
  let s = extents (List.hd (Arith.parts d into)) and t = extents idx in
  let r = List.length s in
  let p = numel (List.filteri (fun i _ -> i < axis) s) in
  let q = numel (List.filteri (fun i _ -> i > axis) s) in
  let n = List.nth s axis and k = List.nth t axis in
  if c = Add && D.is D.Boolean dt then unlowered "a boolean Add Scatter";
  let w = if D.is D.Complex dt then 2 else 1 in
  let narrow = c = Add && D.is D.Float dt && D.bits dt < 32 in
  let ca = if narrow then D.Any D.Float32 else d in
  let on_device v =
    if Option.is_none (Ops.device v) then Ops.copy_to_device v device else v
  in
  (* [v], a node of [d], as the flat parts of [ca]. *)
  let flat v =
    let v = if narrow then Arith.cast ~src:d ca v else v in
    Shape.reshape v [ Ops.Int (Shape.max_numel v) ]
  in
  let fresh = fresh device in
  (* A kernel's argument starts on 16 bytes, as its parameter assumes: a view
     at another phase, such as an argument's, is copied. *)
  let arg v =
    let v = Ops.contiguous (on_device v) in
    let align, phase = Call.storage_phase v in
    if align >= 16 && phase mod 16 = 0 then v else fresh v
  in
  let combined olds us =
    match c with
    | Set -> us
    | Add -> Arith.binary Add ca olds us
    | Max -> Arith.binary Maximum ca olds us
    | Min -> Arith.binary Minimum ca olds us
  in
  (* Positions an expansion repeats along every axis after [axis] are read once
     a row, from the view's first element along those axes. *)
  let rows =
    let spread = spread idx in
    List.for_all
      (fun i -> i <= axis || List.nth t i = 1 || List.mem i spread)
      (List.init r Fun.id)
  in
  let i =
    if not rows then Shape.reshape idx [ Ops.Int (numel t) ]
    else
      let first i = if i > axis then Some (Ops.Int 0, Ops.Int 1) else None in
      Shape.reshape (Shape.shrink idx (List.init r first)) [ Ops.Int (p * k) ]
  in
  (* The store that fills [y] computes [into]: a constant [into] is a fill. *)
  let y = if narrow then fresh (flat into) else dst (flat into) in
  let hit = if not narrow then [] else [ fresh (zeros Uint8 [ numel s ]) ] in
  let outs =
    Call.custom_kernel
      ([ y; arg i; arg (flat updates) ] @ hit)
      (walk ~host ~rows ~w ~p ~n ~q ~k combined)
  in
  let y = Shape.reshape (List.hd outs) (Arith.dims ca (Array.of_list s)) in
  if not narrow then y
  else
    let hit = Shape.reshape (List.nth outs 3) (dims s) in
    Ops.where (Ops.ne hit (int hit 0)) (Arith.cast ~src:ca d y) into

let scatter ?dst ~host device (D.Any dt as d) (c : Nx_kernel.combine) ~axis
    ~idx ~updates into =
  let s = extents (List.hd (Arith.parts d into)) and t = extents idx in
  let r = List.length s in
  let n = List.nth s axis and k = List.nth t axis in
  if n = 0 || numel t = 0 then into
  else if n * k > work_bound * (n + k) then
    let dst = Option.value dst ~default:(fresh device) in
    looped ~dst ~host device d c ~axis ~idx ~updates into
  else
    let wide = s @ [ k ] in
    (* A node of [idx]'s shape moved to [wide], its axis [axis] last. *)
    let spread u =
      let others = List.filter (( <> ) axis) (List.init r Fun.id) in
      let one = List.mapi (fun i e -> if i = axis then 1 else e) s @ [ k ] in
      Shape.expand
        (Shape.reshape (Shape.permute u (others @ [ axis ])) (dims one))
        (dims wide)
    in
    let own u = Shape.reshape u (dims (s @ [ 1 ])) in
    let mask =
      Ops.eq (spread idx) (laid wide axis (Shape.arange ~dtype:Int64 n))
    in
    let landed =
      let m = Shape.rop (Ops.cast mask Uint8) Op.Max [ r ] in
      Ops.ne m (int m 0)
    in
    let kept y =
      Arith.of_parts d
        (List.map2 (Ops.where landed) (Arith.parts d y) (Arith.parts d into))
    in
    let extreme m =
      (* [into]'s element, first, then each update, the element standing in
         for one that does not land. *)
      let terms x u =
        let x = own x in
        joined r x (Ops.where mask (spread u) (Shape.expand x (dims wide)))
      in
      let xs = List.map2 terms (Arith.parts d into) (Arith.parts d updates) in
      let y = Lower_reduce.reduce (Monoid m) [ r ] ~src:d d in
      kept (List.hd (y (Arith.of_parts d xs)))
    in
    match c with
    | Set ->
        (* The last update to land, of highest position along [axis], wins:
           its bits are picked among zeros by an unsigned maximum. *)
        let order = laid wide r (Shape.arange ~dtype:Int64 k) in
        let latest =
          Shape.rop (Ops.where mask order (int order (-1))) Op.Max [ r ]
        in
        let last = Ops.eq order (Shape.expand (own latest) (dims wide)) in
        let pick u =
          let b = Bits.bits (spread u) in
          let b = Shape.rop (Ops.where last b (int b 0)) Op.Max [ r ] in
          Ops.bitcast b (Ops.dtype u)
        in
        kept (Arith.of_parts d (List.map pick (Arith.parts d updates)))
    | Max -> extreme Max
    | Min -> extreme Min
    | Add ->
        if D.is D.Boolean dt then unlowered "a boolean Add Scatter";
        let ca = Nx_kernel.carrier d in
        let lift u = Arith.parts ca (Arith.cast ~src:d ca u) in
        let xs = List.map own (lift into) in
        let us = List.map spread (lift updates) in
        kept (Arith.cast ~src:ca d (summed ca ~landed:mask xs us))

(* Folds

   A fold undoes a padded load's movements in reverse, each by its adjoint: a
   window's sums the windows' elements back into place, the ends' pads what
   they cropped and crops what they padded, and the interior padding's keeps
   every [k + 1]th element. Each part sums in the carrier, an integer on an
   unsigned pattern of 32 or 64 bits as {!Lower_reduce} folds integers, and
   rounds once. The same movements over a mark per term, [2] for a term other
   than [-0] and [1] for [-0], folded by their maximum, find where every term
   is [-0], which the sum, from tolk's [+0] accumulator, makes [+0]. *)

(* [only r i b] is [b] at axis [i] of [r] axes, [None] elsewhere. *)
let only r i b = List.init r (fun j -> if j = i then Some b else None)

(* [rows u i k] is [u] with its axis [i] split into rows of [k]. *)
let rows u i k =
  let e = extents u in
  let split = List.mapi (fun j d -> if j = i then [ d / k; k ] else [ d ]) e in
  Shape.reshape u (dims (List.concat split))

let ints a b = (Ops.Int a, Ops.Int b)

(* [unwindow op u w s] is the adjoint of the window [w] over a value of shape
   [s]: [u]'s elements, of the windowed shape, folded by [op] into the place
   each reads. The axis is laid out as rows one element longer than
   [w.dilation] apart from a whole number of copies of it, so that the
   window's element [j] at window [l] lands [j·dilation + l·step] into a copy:
   the copies' sum is the fold. *)
let unwindow op u (w : Nx_array.Move.window) s =
  let a = w.axis and r = List.length s in
  let n = List.nth s a and l = Bits.extent u a in
  if n = 0 || l = 0 then zeros (Ops.dtype u) s
  else
    let others = List.filter (( <> ) a) (List.init r Fun.id) in
    let o = List.map (List.nth s) others in
    let scale = Int.max 1 (ceil_div ((l * w.step) - w.dilation) n) in
    let row = (n * scale) + w.dilation in
    let copies = ceil_div (w.size * row) n in
    let reshape x e = Shape.reshape x (dims (o @ e)) in
    let pad_to x e = Shape.pad_to x (List.map Option.some (dims (o @ e))) in
    let x = Shape.permute u (others @ [ r; a ]) in
    let x = reshape x [ w.size; l; 1 ] in
    let x = pad_to x [ w.size; l; w.step ] in
    let x = reshape x [ w.size; l * w.step ] in
    let x = pad_to x [ w.size; row ] in
    let x = reshape x [ w.size * row ] in
    let x = pad_to x [ copies * n ] in
    let x = Shape.rop (reshape x [ copies; n ]) op [ r - 1 ] in
    let back i = if i = a then r - 1 else if i < a then i else i - 1 in
    Shape.permute x (List.init r back)

(* [unends lo hi s u] is the adjoint of padding a value of shape [s] by [lo]
   and [hi]: [u] padded with zeros where the load cropped, then cropped where
   it padded. *)
let unends lo hi s u =
  let at f = List.init (List.length s) (fun i -> Some (f i)) in
  let cropped i a = Int.max 0 (-Iarray.get a i) in
  let u = Shape.pad u (at (fun i -> ints (cropped i lo) (cropped i hi))) in
  Shape.shrink u
    (at (fun i ->
         let b = Int.max 0 (Iarray.get lo i) in
         ints b (b + List.nth s i)))

(* [uninterior k s u] is the adjoint of spacing a value of shape [s] by [k]:
   every [k.(i) + 1]th element of [u] along each axis [i]. *)
let uninterior k s u =
  let r = List.length s in
  let every u i k =
    if k = 0 || List.nth s i = 0 then u
    else
      let u = rows (Shape.pad u (only r i (ints 0 k))) i (k + 1) in
      let u = Shape.shrink u (only (r + 1) (i + 1) (ints 0 1)) in
      Shape.flatten ~start:i ~stop:(i + 1) u
  in
  let u = ref u in
  Iarray.iteri (fun i k -> u := every !u i k) k;
  !u

let fold (D.Any dt as d) (pad : Nx_kernel.pad) shape x =
  if D.is D.Boolean dt then unlowered "a boolean Fold";
  let s0 = Iarray.to_list shape in
  let spaced i n =
    if n = 0 then 0 else n + ((n - 1) * Iarray.get pad.interior i)
  in
  let s1 = List.mapi spaced s0 in
  let s2 =
    List.mapi (fun i n -> n + Iarray.get pad.lo i + Iarray.get pad.hi i) s1
  in
  (* Each window with the shape it windows, the last first. *)
  let windows, _ =
    Iarray.fold_left
      (fun (ws, s) (w : Nx_array.Move.window) ->
        let m = Nx_array.Move.Window (Iarray.of_list [ w ]) in
        let s' = Iarray.to_list (Nx_array.Move.shape m (Iarray.of_list s)) in
        ((w, s) :: ws, s'))
      ([], s2) pad.windows
  in
  let back op u =
    let u = List.fold_left (fun u (w, s) -> unwindow op u w s) u windows in
    uninterior pad.interior s0 (unends pad.lo pad.hi s1 u)
  in
  let ca = Nx_kernel.carrier d in
  let part u =
    let t = Ops.dtype u in
    if not (Dtype.is_float t) then
      let wide : Dtype.t =
        if Dtype.itemsize t < 4 then Uint32 else Bits.unsigned t
      in
      let y = back Op.Add (Ops.cast u wide) in
      (* A cast, not a bitcast, back to a signed type: tolk renders a
         bitcast of a constant of 64 bits from a 32-bit literal. *)
      Ops.cast (Ops.cast y (Bits.unsigned t)) t
    else
      let y = back Op.Add u in
      let mark =
        Ops.where (negative_zero u) (int (Bits.bits u) 1) (int (Bits.bits u) 2)
      in
      let mark = back Op.Max (Ops.cast mark Uint8) in
      Ops.where (Ops.eq mark (int mark 1)) (Shape.const_like y (`Float (-0.))) y
  in
  let x = Arith.cast ~src:d ca x in
  Arith.cast ~src:ca d (Arith.of_parts ca (List.map part (Arith.parts ca x)))

(* Assemblies

   A piece is placed into the result's shape by the adjoint of its region's
   slice, which spaces its elements [step] apart and pads around them, and so
   is a mask of where it lies. Each piece in turn selects its elements where
   it lies over what came before. *)

(* [placed s ranges u] is [u] at the positions [ranges] keep of a shape [s],
   zeros elsewhere. *)
let placed s (ranges : Nx_array.Move.range iarray) u =
  let r = List.length s in
  let place u i ({ start; count; step } : Nx_array.Move.range) =
    let u, first, step =
      if step < 0 then (Shape.flip u [ i ], start + ((count - 1) * step), -step)
      else (u, start, step)
    in
    let span = ((count - 1) * step) + 1 in
    let u =
      if step = 1 then u
      else
        let gaps = only (r + 1) (i + 1) (ints 0 (step - 1)) in
        let u = Shape.pad (rows u i 1) gaps in
        let u = Shape.flatten ~start:i ~stop:(i + 1) u in
        Shape.shrink u (only r i (ints 0 span))
    in
    Shape.pad u (only r i (ints first (List.nth s i - first - span)))
  in
  let u = ref u in
  Iarray.iteri (fun i range -> u := place !u i range) ranges;
  !u

(* [words dt x] is the bits of each part of the element [x] of [dt], as
   unsigned integers in the host's order. *)
let words (type v s) (dt : (v, s) D.t) (x : v) =
  let b = Nx_kernel.Prog.bits dt x in
  let word b =
    let n = ref Bigint.zero in
    for i = String.length b - 1 downto 0 do
      n := Bigint.add (Bigint.shift_left !n 8) (Bigint.of_int (Char.code b.[i]))
    done;
    `Int !n
  in
  if not (D.is D.Complex dt) then [ word b ]
  else
    let h = String.length b / 2 in
    [ word (String.sub b 0 h); word (String.sub b h h) ]

(* A float assembles as its bits. Where a float constant fill meets a padded
   piece in an Ops.where, tolk quiets a signalling NaN constant before it
   renders the kernel: a float64 fill of bits 0x7ff0000000000001 beside a
   piece reaches the kernel's source as 0x7ff8000000000001. A [-0] fill stays
   [-0], so no addition is involved, and a fill alone keeps its bits. *)
let assemble (type v s) (dt : (v, s) D.t) (fill : v) shape pieces =
  let d = D.Any dt in
  let s = Iarray.to_list shape in
  let count (r : Nx_array.Move.range) = r.count in
  let size (ranges, _) = Iarray.fold_left (fun m r -> m * count r) 1 ranges in
  let pieces = List.filter (fun piece -> size piece > 0) pieces in
  let p = List.length pieces and total = numel s in
  let m = List.fold_left (fun m piece -> m + size piece) 0 pieces in
  if p * total > work_bound * (total + m) then
    unlowered
      (Printf.sprintf "an Assemble of %d pieces into %d elements" p total);
  let tdt = Arith.dtype d in
  let floats = Dtype.is_float tdt in
  let start =
    if floats then
      let tb = Bits.unsigned tdt in
      List.map (fun w -> broadcast (Ops.const ~dtype:tb w) s) (words dt fill)
    else
      let const c = broadcast (Ops.const ~dtype:tdt c) s in
      List.map const (Arith.fill dt fill)
  in
  let put acc (ranges, u) =
    let parts = Arith.parts d u in
    let parts = if floats then List.map Bits.bits parts else parts in
    let ones = Shape.const_like ~dtype:Bool (List.hd parts) (`Bool true) in
    let mask = placed s ranges ones in
    List.map2 (fun a p -> Ops.where mask (placed s ranges p) a) acc parts
  in
  let y = Arith.of_parts d (List.fold_left put start pieces) in
  if floats then Ops.bitcast y tdt else y

(* Sorts

   A sort orders keys, unsigned integers that order as the elements do, by a
   bitonic network of maxima and minima, which sorts integers exactly but is
   not stable. Each key runs packed above its position: packed integers are
   distinct, so the order the network gives them is the stable one, and the
   positions are their low bits. A key of more than 32 bits sorts in passes of
   32 bits, its least significant first: each pass sorts the next bits in the
   previous pass's order, and being stable keeps that order among equal bits.
   The values are the elements gathered at the positions, bits unchanged. *)

let bit_length n =
  let rec go n b = if n = 0 then b else go (n lsr 1) (b + 1) in
  go n 0

let halves u axis =
  match Shape.split ~axis u [ 1; 1 ] with
  | [ a; b ] -> (a, b)
  | _ -> assert false

(* [bitonic x axis] is the integers [x] sorted increasingly along [axis]. The
   axis, padded to a power of two with the greatest integer, is split into axes
   of two elements, one per stage of the network. *)
let bitonic x axis =
  let n = Bits.extent x axis in
  if n <= 1 then x
  else
    let stages = bit_length (n - 1) and shape = Shape.shape x in
    let x =
      Shape.pad_to
        ~value:(Dtype.max (Ops.dtype x) :> Dtype.const)
        x
        (List.init (Shape.ndim x) (fun d ->
             if d = axis then Some (Ops.Int (1 lsl stages)) else None))
    in
    let x = Shape.unflatten x axis (List.init stages (fun _ -> Ops.Int 2)) in
    let r = Shape.ndim x in
    (* A stage's crossover reverses the second half of each of its boxes, so
       that comparisons all in one direction merge them; it is undone after. *)
    let crossover stage x =
      let c = axis + stages - stage - 1 in
      let blue, green = halves x c in
      let flipped =
        List.init (stage + List.length shape - axis) (fun i -> r - 1 - i)
      in
      Shape.cat ~axis:c blue [ Shape.flip green flipped ]
    in
    let compare x sub =
      let p = axis + stages - sub - 1 in
      let top, bottom = halves x p in
      Ops.contiguous
        (Shape.cat ~axis:p (Ops.minimum top bottom) [ Ops.maximum top bottom ])
    in
    let stage x s =
      let x = if s < stages then Ops.contiguous (crossover s x) else x in
      let x = List.fold_left compare x (List.init s (fun i -> s - 1 - i)) in
      if s < stages then crossover s x else x
    in
    let x = List.fold_left stage x (List.init stages (fun i -> i + 1)) in
    Shape.shrink_to
      (Shape.flatten ~start:axis ~stop:(axis + stages - 1) x)
      (List.map Option.some shape)

(* [ranked axis h] is the int64 positions that sort the int64 keys [h], below
   [2^32], stably along [axis]. *)
let ranked axis h =
  let s = extents h in
  let n = List.nth s axis in
  let low = bit_length (n - 1) in
  let ranks = laid s axis (Shape.arange ~dtype:Int64 n) in
  let packed = Ops.bitwise_or (Ops.shl h (int h low)) ranks in
  let sorted = bitonic (Ops.contiguous packed) axis in
  Ops.bitwise_and sorted (int h ((1 lsl low) - 1))

(* [key u] is the unsigned integer of [u]'s width that orders as its elements:
   a signed integer's or a float's order key with its sign bit flipped, a
   float's order key being the signed integer of its bits with a negative
   float's magnitude bits flipped, so that [-0] lies just below [+0]. A boolean
   is its truth. *)
let most k = Shape.const_like k (Dtype.max (Ops.dtype k) :> Dtype.const)
let least k = Shape.const_like k (Dtype.min (Ops.dtype k) :> Dtype.const)

let key u =
  let t = Ops.dtype u in
  let biased k = Ops.bitcast (Ops.bitwise_xor k (least k)) (Bits.unsigned t) in
  if Dtype.is_float t then
    let k = Ops.bitcast u (Bits.signed t) in
    biased (Ops.where (Ops.lt k (int k 0)) (Ops.bitwise_xor k (most k)) k)
  else if Dtype.is_unsigned t then u
  else biased u

(* [keys d ~descending u] is the keys of the node [u] of a value of [d], their
   most significant 32 bits first, each an int64 below [2^32]: complemented
   where [descending], every NaN, and every complex number with a NaN part,
   the greatest, and a complex number its real part's then its imaginary
   part's. *)
let keys (D.Any dt as d) ~descending u =
  let parts =
    if D.is D.Boolean dt then [ Ops.cast (Ops.ne u (int u 0)) Uint8 ]
    else if D.is D.Float dt && D.bits dt < 16 then
      (* A float of a code orders as its float32 value. *)
      [ Arith.cast ~src:d (D.Any D.Float32) u ]
    else Arith.parts d u
  in
  let ks = List.map key parts in
  let ks =
    if not (Dtype.is_float (Ops.dtype (List.hd parts))) then ks
    else
      let nan =
        List.fold_left
          (fun a p -> Ops.bitwise_or a (Bits.is_nan p))
          (Bits.is_nan (List.hd parts))
          (List.tl parts)
      in
      List.map (fun k -> Ops.where nan (most k) k) ks
  in
  let ks = if descending then List.map Ops.bitwise_not ks else ks in
  let split k =
    if Dtype.itemsize (Ops.dtype k) < 8 then [ Ops.cast k Int64 ]
    else
      [
        Ops.cast (Ops.shr k (int k 32)) Int64;
        Ops.cast (Ops.bitwise_and k (int k 0xffff_ffff)) Int64;
      ]
  in
  List.concat_map split ks

let sort d ~axis ~descending ~k x =
  let s = extents (List.hd (Arith.parts d x)) in
  let n = List.nth s axis in
  if n > 1 lsl 31 then unlowered "a Sort of more than 2^31 elements";
  let positions =
    if n <= 1 || numel s = 0 then zeros Int64 s
    else
      match List.rev (keys d ~descending x) with
      | [] -> assert false
      | least :: rest ->
          List.fold_left
            (fun p h -> take p axis (ranked axis (take h axis p)))
            (ranked axis least) rest
  in
  let values =
    if n <= 1 || numel s = 0 then x
    else
      let gathered u = take u axis positions in
      Arith.of_parts d (List.map gathered (Arith.parts d x))
  in
  match k with
  | None -> (values, positions)
  | Some k ->
      let first u =
        Shape.shrink u
          (List.init (Shape.ndim u) (fun i ->
               if i = axis then Some (Ops.Int 0, Ops.Int k) else None))
      in
      (first values, first positions)
