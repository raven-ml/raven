(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Gathers and assemblies, planned as index.metal's moves.

   A move is the strides of its destination, its source and its positions along
   its axes, in elements, which [unitize] turns into units: an element's bytes
   are one more axis, adjacent axes that every array steps alike merge, and the
   unit is the widest that every address lies on and that divides the innermost
   run's bytes. *)

module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module M = Nx_array.Move
module K = Kernels

(* Arrays *)

let shape_of x = Iarray.to_array (L.shape (A.layout x))
let strides_of x = Iarray.to_array (L.strides (A.layout x))
let numel s = Array.fold_left ( * ) 1 s
let bits_of x = D.bits (A.dtype x)

(* The index of the first of [ops] from [k] on [buffer]. *)
let rec first_on (ops : A.any array) buffer k =
  let (A.Any y) = ops.(k) in
  if A.buffer y == buffer then k else first_on ops buffer (k + 1)

(* The slot of [x]'s buffer in [b]: that of the first of [ops], the call's
   operands as the door numbers them ({!Plan.slot}), on it. *)
let slot b ops x = Plan.slot b (first_on ops (A.buffer x) 0)

(* A move's end: its slot, the byte of its address, the address's residue modulo
   32, and for elements of 1 or 4 bits the bit of its word at which its position
   0 starts. Its positions count from position 0. *)
type site = { slot : int; at : int; residue : int; bit : int }

let site b ops (A.Any x) =
  let s = slot b ops x in
  let bits = bits_of x in
  if bits >= 8 then
    {
      slot = s;
      at = L.offset (A.layout x) * (bits / 8);
      residue = Operands.residue (A.buffer x);
      bit = 0;
    }
  else
    (* Moves.place gives the word of the lowest element; position 0 lies [pos]
       elements past it. *)
    let at, pos, phase = Moves.place (A.Any x) in
    let bit = phase + (bits * pos) in
    { slot = s; at = at + (4 * (bit / 32)); residue = 0; bit = bit mod 32 }

(* Axes *)

(* An axis of a move: its extent and the strides along it of the destination,
   the source and the positions. *)
type axis = { e : int; sd : int; ss : int; si : int }

let ax e ~sd ~ss ~si = { e; sd; ss; si }

(* [axes] with the axes of one element dropped and each axis merged into the one
   outside it where all three strides step alike: the same places in the same C
   order. *)
let merge axes =
  let out =
    Array.fold_left
      (fun out a ->
        if a.e = 1 then out
        else
          match out with
          | o :: rest
            when o.sd = a.sd * a.e && o.ss = a.ss * a.e && o.si = a.si * a.e ->
              { a with e = o.e * a.e } :: rest
          | _ -> a :: out)
      [] axes
  in
  Array.of_list (List.rev out)

(* A move in units: its axes, the source's stride along the moved axis, the
   unit's log2 of bits (index.metal's [how]) and its bytes, 0 for an element of
   1 or 4 bits. *)
type units = { axes : axis array; sa : int; log : int; bytes : int }

let rec log2 n = if n <= 1 then 0 else 1 + log2 (n / 2)

(* Whether [a], the innermost of [axes], is a run: both ends step one unit along
   it and the positions none. *)
let is_run a = a.sd = 1 && a.ss = 1 && a.si = 0

(* The move of [bits]-bit elements along [axes], of the source's stride [sa]
   along the moved axis, between [dst] and [src], in units: for elements of a
   byte or more, the widest of 16, 8, 4, 2 and 1 bytes that divides each end's
   address, every stride of each end in bytes, [sa]'s and the innermost run's
   bytes, a byte where no run is innermost. *)
let unitize ~bits ~dst ~src ~sa axes =
  if bits < 8 then { axes = merge axes; sa; log = log2 bits; bytes = 0 }
  else
    let w = bits / 8 in
    let bytes =
      Array.append
        (Array.map (fun a -> { a with sd = a.sd * w; ss = a.ss * w }) axes)
        [| ax w ~sd:1 ~ss:1 ~si:0 |]
    in
    let axes = merge bytes and sa = sa * w in
    let r = Array.length axes in
    (* A unit past a byte lies within a run: without one innermost, as when an
       element of a byte steps by more than one, the unit is a byte. *)
    let inner_run = r > 0 && is_run axes.(r - 1) in
    let fits u =
      (u = 1 || inner_run)
      && (dst.residue + dst.at) mod u = 0
      && (src.residue + src.at) mod u = 0
      && sa mod u = 0
      &&
      let ok = ref true in
      Array.iteri
        (fun j a ->
          if j = r - 1 && is_run a then ok := !ok && a.e mod u = 0
          else ok := !ok && a.sd mod u = 0 && a.ss mod u = 0)
        axes;
      !ok
    in
    let u = List.find fits [ 16; 8; 4; 2; 1 ] in
    let scaled =
      Array.mapi
        (fun j a ->
          if j = r - 1 && is_run a then { a with e = a.e / u }
          else { a with sd = a.sd / u; ss = a.ss / u })
        axes
    in
    { axes = merge scaled; sa = sa / u; log = log2 (8 * u); bytes = u }

(* The units of a move, by the number of its axes. *)
let count_of axes = Array.fold_left (fun n a -> n * a.e) 1 axes

(* Whether a launch's units reach 2^32, past which the kernel divides by 64-bit
   divisors. *)
let is_wide n = n > 1 lsl 32

(* The divisor of [e] for a launch of [wide] units. *)
let divisor ~wide e =
  if wide then Fft.divisor e else Int64.of_int (Plan.divisor e)

(* Writes [axes] as the move kernel reads them into [s] from word [at]. *)
let put_axes s ~at ~wide axes =
  Array.iteri
    (fun j a ->
      let word i = 8 * (at + (K.move_axis_words * j) + i) in
      Bytes.set_int64_le s (word K.move_axis_extent) (Int64.of_int a.e);
      Bytes.set_int64_le s (word K.move_axis_divisor) (divisor ~wide a.e);
      Bytes.set_int64_le s (word K.move_axis_sd) (Int64.of_int a.sd);
      Bytes.set_int64_le s (word K.move_axis_ss) (Int64.of_int a.ss);
      Bytes.set_int64_le s (word K.move_axis_si) (Int64.of_int a.si))
    axes

(* Time *)

(* A move's thread's work in float32 operations: a copy thread's 128 before its
   axes (Moves); for each axis but the outermost, a division by the host's
   divisor ({!Plan.divide_ops}, 64-bit past 2^32 units) and its three ends'
   products and sums, 48, an estimate until measured; and for an element of 1 or
   4 bits, two atomics on its word. *)
let unit_ops = 128
let division ~wide = if wide then Plan.wide_divide_ops else Plan.divide_ops
let axis_ops ~wide = division ~wide + 48
let atomic_ops = 512

(* The most units of [count] whose launch of groups of [threads] threads and
   [per] units, of [resident] a core, [cost m groups] for [m] units, fits
   {!Plan.allowed}; 0 where a unit's does not. *)
let most ~threads ~per ~resident ~count cost =
  let seconds m =
    let groups = (m + per - 1) / per in
    Plan.seconds ~threads ~groups:(groups, 1, 1) ~resident (cost m groups)
  in
  let allowed = Plan.allowed () in
  if seconds count <= allowed then count
  else if seconds 1 > allowed then 0
  else
    (* [lo] fits and [hi] does not. *)
    let rec search lo hi =
      if hi - lo <= 1 then lo
      else
        let mid = lo + ((hi - lo) / 2) in
        if seconds mid <= allowed then search mid hi else search lo mid
    in
    search 1 count

exception Over_budget

(* [f ()], or [Error Declined] where a unit's launch passes the budget. *)
let fitting f = try f () with Over_budget -> Error A.Declined

(* Adds the launches of [k] over the units [first] (defaults to 0) to [count] -
   1, [per] a group (a thread each by default), in pieces that each fit
   {!Plan.allowed}: [words] with each piece's first unit and end at the words
   [at], and [refs]. Raises [Over_budget] where a unit's launch passes the
   budget. *)
let launches b k ?per ?(shared = 0) ?(first = 0) ~at:(wf, wc) ~resident ~cost
    ~count words refs =
  let threads = K.threads k in
  let per = Option.value per ~default:threads in
  let piece = most ~threads ~per ~resident ~count:(count - first) cost in
  if piece = 0 then raise Over_budget;
  let at = ref first in
  while !at < count do
    let m = Int.min piece (count - !at) in
    let w = Bytes.copy words in
    Plan.set w wf !at;
    Plan.set w wc (!at + m);
    let groups = (m + per - 1) / per in
    Plan.launch b k ~groups:(groups, 1, 1) ~shared ~resident
      ~cost:(cost m groups) w refs;
    at := !at + m
  done

(* Moves *)

(* A move's mode: a copy; a gather of [d] positions along its axis by the
   positions at [idx], a slot and the byte of position 0, of 32 bits where
   [narrow] and 64 otherwise; or a fill of the 16 bytes [lo], [hi]. *)
type mode =
  | Copy
  | Gather of { idx : int * int; d : int; narrow : bool }
  | Fill of { lo : int64; hi : int64 }

(* The bytes of a gather's position. *)
let position_bytes ~narrow = if narrow then 4 else 8

(* The bytes a unit's read of the source takes: its own where the innermost
   axis's units lie at most one apart, a line otherwise, and a line for a
   gather's read whose position changes along it. *)
let read_bytes ~mode ~unit inner =
  match mode with
  | Fill _ -> 0
  | Gather _ when inner.ss = 0 && inner.si <> 0 -> Plan.line
  | Copy | Gather _ ->
      if abs inner.ss <= 1 then unit else Int.min Plan.line (abs inner.ss * unit)

(* The cost of [m] units in [groups] groups of a move of [u]'s units: each
   unit's bytes written and read, a gather's positions, a line at each end of
   each run of the innermost axis, and each thread's work; a gather's thread
   reads its position, then its unit. *)
let move_cost ~mode (u : units) m groups =
  let r = Array.length u.axes in
  let inner = if r = 0 then ax 1 ~sd:0 ~ss:0 ~si:0 else u.axes.(r - 1) in
  let unit = if u.bytes = 0 then 4 else u.bytes in
  let runs = (m / inner.e) + 2 in
  let idx =
    match mode with
    | Copy | Fill _ -> 0
    | Gather { narrow; _ } ->
        if inner.si = 0 then runs * Plan.line
        else m * Int.min Plan.line (position_bytes ~narrow * abs inner.si)
  in
  let wide = is_wide (count_of u.axes) in
  let ops =
    unit_ops
    + (Int.max 0 (r - 1) * axis_ops ~wide)
    + if u.bytes = 0 then atomic_ops else 0
  in
  {
    Plan.bytes =
      (m * (unit + read_bytes ~mode ~unit inner)) + idx + (2 * Plan.line * runs);
    ops = K.threads K.Move * groups * ops;
    chain =
      (match mode with Fill _ -> 0 | Copy -> 1 | Gather _ -> 2)
      * Plan.read_links;
  }

let how ~mode ~wide (u : units) ~dst ~src =
  let m =
    match mode with
    | Copy -> K.move_copy
    | Gather { narrow; _ } ->
        K.move_gather lor (Bool.to_int narrow lsl K.how_narrow)
    | Fill _ -> K.move_fill
  in
  m
  lor (Bool.to_int wide lsl K.how_wide)
  lor (u.log lsl K.how_unit)
  lor (dst.bit lsl K.how_dst_bit)
  lor (src.bit lsl K.how_src_bit)
  lor (Array.length u.axes lsl K.how_rank)

(* The ref of a move's axes past those its words hold: a constant's, kept for
   the calls of the same axes. *)
let axes_ref b ~wide axes =
  let r = Array.length axes in
  if r <= K.move_holds then []
  else
    let s = Bytes.create (8 * K.move_axis_words * r) in
    put_axes s ~at:0 ~wide axes;
    let t = Bytes.unsafe_to_string s in
    let c =
      Plan.constant b ~key:("move " ^ t) ~length:(String.length t) (fun () ->
          Moves.bigbytes t)
    in
    [ { Plan.word = K.move_axes; slot = c; write = false } ]

(* A move planned: its mode, units, count of units, words and refs. *)
type planned = {
  mode : mode;
  u : units;
  count : int;
  words : Bytes.t;
  refs : Plan.ref array;
}

(* The move of [src]'s elements of [bits] bits into [dst]'s by [mode], along
   [axes] in elements, [sa] the source's stride along the moved axis. *)
let plan_move b ~mode ~bits ~dst ~src ?(sa = 0) axes =
  let u = unitize ~bits ~dst ~src ~sa axes in
  let count = count_of u.axes in
  let wide = is_wide count in
  let w = Plan.words K.Move in
  Plan.set w K.move_dst dst.at;
  Plan.set w K.move_how (how ~mode ~wide u ~dst ~src);
  let dst_ref = { Plan.word = K.move_dst; slot = dst.slot; write = true } in
  let read () =
    Plan.set w K.move_src src.at;
    Plan.set w K.move_sa u.sa;
    { Plan.word = K.move_src; slot = src.slot; write = false }
  in
  let refs =
    match mode with
    | Copy -> [ dst_ref; read () ]
    | Gather { idx = slot, at; d; _ } ->
        Plan.set w K.move_idx at;
        Plan.set w K.move_d d;
        [ { Plan.word = K.move_idx; slot; write = false }; dst_ref; read () ]
    | Fill { lo; hi } ->
        Bytes.set_int64_le w (8 * K.move_d) lo;
        Bytes.set_int64_le w (8 * K.move_sa) hi;
        [ dst_ref ]
  in
  if Array.length u.axes <= K.move_holds then
    put_axes w ~at:K.move_held ~wide u.axes;
  let refs = Array.of_list (refs @ axes_ref b ~wide u.axes) in
  { mode; u; count; words = w; refs }

(* Adds the launches of [m]'s units [first] to [upto] - 1, which read their
   positions from the one of [from]. *)
let launch_move b m ~from ~first ~upto =
  Plan.set m.words K.move_from from;
  let k =
    match m.mode with Fill _ -> K.Move_fill | Copy | Gather _ -> K.Move
  in
  let resident = Plan.holds ~threads:(K.threads k) 0 in
  launches b k ~first
    ~at:(K.move_first, K.move_count)
    ~resident
    ~cost:(move_cost ~mode:m.mode m.u)
    ~count:upto m.words m.refs

(* Adds the launches of a move ({!plan_move}). *)
let move b ~mode ~bits ~dst ~src ?sa axes =
  let m = plan_move b ~mode ~bits ~dst ~src ?sa axes in
  if m.count > 0 then launch_move b m ~from:0 ~first:0 ~upto:m.count

(* Gathers *)

let gather ~ops ~axis ~dst idx x =
  let ys = shape_of dst in
  if numel ys = 0 then Ok None
  else
    fitting @@ fun () ->
    let b = Plan.make () in
    let ds = L.strides (L.contiguous (Iarray.of_array ys)) in
    let xs = strides_of x and is = strides_of idx in
    let axes =
      Array.mapi
        (fun a e ->
          ax e ~sd:(Iarray.get ds a)
            ~ss:(if a = axis then 0 else xs.(a))
            ~si:is.(a))
        ys
    in
    let ix = slot b ops idx in
    let mode =
      Gather
        {
          idx = (ix, 8 * L.offset (A.layout idx));
          d = (shape_of x).(axis);
          narrow = false;
        }
    in
    move b ~mode ~bits:(bits_of dst) ~dst:(site b ops (A.Any dst))
      ~src:(site b ops (A.Any x)) ~sa:xs.(axis) axes;
    Ok (Some (Plan.finish ~most:2 b))

(* Range gathers *)

(* The entries of a runs group's chunk, {!K.runs_chunk} but in tests. *)
let chunk_bound = Atomic.make K.runs_chunk

(* The most bytes of positions a range gather holds at once,
   {!Plan.kept_workspace} but in tests. *)
let region_bound = Atomic.make Plan.kept_workspace

(* The log2 of the extent of a source axis from which a range gather's positions
   take 64 bits: they hold -1 to its extent less 1. 31 but in tests. *)
let narrow_bound = Atomic.make 31

(* The log2 of the least position, tap number or step from which a fold computes
   in 64 bits. 31 but in tests. *)
let taps_bound = Atomic.make 31

let bound ?(runs = K.runs_chunk) ?(region = Plan.kept_workspace) ?(narrow = 31)
    ?(taps = 31) () =
  if runs < 1 || runs > K.runs_chunk || runs land (runs - 1) <> 0 then
    invalid_arg "Nx_metal.bound_index: runs";
  if region < 8 || region > Plan.kept_workspace then
    invalid_arg "Nx_metal.bound_index: region";
  if narrow < 1 || narrow > 31 then invalid_arg "Nx_metal.bound_index: narrow";
  if taps < 1 || taps > 31 then invalid_arg "Nx_metal.bound_index: taps";
  Atomic.set chunk_bound runs;
  Atomic.set region_bound region;
  Atomic.set narrow_bound narrow;
  Atomic.set taps_bound taps;
  Plan.renew ()

(* A runs group's work in float32 operations, an estimate: 16 entries' maxima
   and stores, 5 shuffles and the SIMD-groups' maxima. *)
let runs_ops = 512

(* The cost of [groups] groups of runs over [n] entries a row from a source
   [stride] entries apart: each entry's read and write, a line at each end of a
   group's, and each thread's work; a thread reads its entries, then steps 16
   maxima, 5 shuffles, a barrier and 8 maxima. *)
let runs_cost ~stride ~n m groups =
  let entries = Int.min (m * Atomic.get chunk_bound) (m * n) in
  {
    Plan.bytes =
      (entries * (8 + Int.min Plan.line (8 * abs stride)))
      + (4 * Plan.line * groups);
    ops = K.threads K.Runs * groups * runs_ops;
    chain = Plan.read_links + 64;
  }

(* The cost of [m] entries of runs_carry: each entry read and written and its
   chunk's carry. *)
let carry_cost m groups =
  {
    Plan.bytes = (m * 16) + (2 * Plan.line * groups);
    ops = K.threads K.Runs_carry * groups * unit_ops;
    chain = 2 * Plan.read_links;
  }

(* The words of a launch of [k], runs or runs_carry, over the rows [axes]
   describe, of the runs' level where [bottom]. *)
let runs_words ?(wide = false) k ~dst ~src ~top ~bottom ~axes ~n ~chunks ~stride
    =
  let w = Plan.words k in
  Plan.set w K.runs_dst dst;
  Plan.set w K.runs_src src;
  Plan.set w K.runs_top top;
  Plan.set w K.runs_how
    ((Array.length axes lsl K.how_rank)
    lor (Bool.to_int bottom lsl K.how_unit)
    lor (log2 (Atomic.get chunk_bound) lsl K.how_chunk)
    lor (Bool.to_int wide lsl K.how_wide));
  Plan.set w K.runs_n n;
  Plan.set w K.runs_chunks chunks;
  Plan.set w K.runs_stride stride;
  if Array.length axes <= K.move_holds then
    put_axes w ~at:K.runs_held ~wide:false axes;
  w

(* Adds the launches that write each row's running maxima of [ends] (runs, then
   levels of their chunks' maxima, then their carries) into the workspace from
   byte [at]: a row is two words, its starts' and ends' places, then its [k]
   maxima. [lead] axes of [rows] rows. *)
let runs b ~ops ~at ~rows ~k starts ends =
  let ws = Plan.workspace b in
  let sh = shape_of starts in
  let lead = Array.length sh - 1 in
  let ss = strides_of starts and es = strides_of ends in
  let cs = L.strides (L.contiguous (Iarray.of_array (Array.sub sh 0 lead))) in
  (* A row's places: its block's, its ends', its starts'. *)
  let axes =
    merge
      (Array.init lead (fun a ->
           ax sh.(a) ~sd:(Iarray.get cs a * (k + 2)) ~ss:es.(a) ~si:ss.(a)))
  in
  let tail =
    if Array.length axes <= K.move_holds then []
    else
      let s = Bytes.create (8 * K.move_axis_words * Array.length axes) in
      put_axes s ~at:0 ~wide:false axes;
      let t = Bytes.unsafe_to_string s in
      let c =
        Plan.constant b ~key:("runs " ^ t) ~length:(String.length t) (fun () ->
            Moves.bigbytes t)
      in
      [ { Plan.word = K.runs_axes; slot = c; write = false } ]
  in
  let resident = Plan.holds ~threads:(K.threads K.Runs) K.runs_shared in
  (* Each level's entries a row, chunks a row, and the byte of its chunks'
     maxima, from the runs' level. *)
  let chunk = Atomic.get chunk_bound in
  let rec levels n top acc =
    let c = (n + chunk - 1) / chunk in
    let acc = (n, c, top) :: acc in
    if c = 1 then List.rev acc else levels c (top + (8 * rows * c)) acc
  in
  let tops = levels k (at + (8 * rows * (k + 2))) [] in
  let scan ~bottom ~dst ~src ~stride ~axes ~refs (n, chunks, top) =
    let w = runs_words K.Runs ~dst ~src ~top ~bottom ~axes ~n ~chunks ~stride in
    Plan.set w K.runs_divisor (Plan.divisor chunks);
    let refs =
      Array.of_list
        ({ Plan.word = K.runs_dst; slot = ws; write = true }
        :: { Plan.word = K.runs_top; slot = ws; write = true }
        :: refs)
    in
    launches b K.Runs ~per:1 ~shared:K.runs_shared
      ~at:(K.runs_first, K.runs_count)
      ~resident ~cost:(runs_cost ~stride ~n) ~count:(rows * chunks) w refs
  in
  let carry ~bottom ~dst ~stride (n, chunks, top) =
    let count = rows * n in
    let wide = is_wide count in
    let w =
      runs_words ~wide K.Runs_carry ~dst ~src:0 ~top ~bottom ~axes:[||] ~n
        ~chunks ~stride
    in
    Bytes.set_int64_le w (8 * K.runs_divisor) (divisor ~wide n);
    let refs =
      [|
        { Plan.word = K.runs_dst; slot = ws; write = true };
        { Plan.word = K.runs_top; slot = ws; write = false };
      |]
    in
    launches b K.Runs_carry
      ~at:(K.runs_first, K.runs_count)
      ~resident:(Plan.holds ~threads:(K.threads K.Runs_carry) 0)
      ~cost:carry_cost ~count w refs
  in
  let ends_ref =
    { Plan.word = K.runs_src; slot = slot b ops ends; write = false }
  in
  (* The rows' blocks, then the chunks' maxima of each level in place: a level's
     row of [n] entries is [n] apart. *)
  let below l =
    if l = 0 then at
    else
      let _, _, top = List.nth tops (l - 1) in
      top
  in
  List.iteri
    (fun l ((n, _, _) as level) ->
      if l = 0 then
        scan ~bottom:true ~dst:at
          ~src:(8 * L.offset (A.layout ends))
          ~stride:es.(lead) ~axes ~refs:(ends_ref :: tail) level
      else
        let under = below l in
        scan ~bottom:false ~dst:under ~src:under ~stride:1
          ~axes:[| ax rows ~sd:n ~ss:n ~si:0 |]
          ~refs:[ { Plan.word = K.runs_src; slot = ws; write = false } ]
          level)
    tops;
  (* Each level but the last takes the running maxima of its chunks before it,
     from the level above, from the top down. *)
  let n_levels = List.length tops in
  for l = n_levels - 2 downto 0 do
    let n, chunks, top = List.nth tops l in
    let stride = if l = 0 then k + 2 else n in
    carry ~bottom:(l = 0) ~dst:(below l) ~stride (n, chunks, top)
  done

(* The cost of [m] positions of [width] bytes in [groups] groups of a range
   gather of [k] runs a row: each position's write, the start and end it finds,
   the rows' maxima once, a halving's step of work each, and a chain of a read a
   step. *)
let positions_cost ~width ~k ~rows m groups =
  let steps = log2 (2 * k) + 1 in
  {
    Plan.bytes = (m * (width + (2 * Plan.line))) + Int.min (m * 8) (rows * k * 8);
    ops = K.threads K.Runs_positions * groups * (unit_ops + (32 * steps));
    chain = (steps + 2) * Plan.read_links;
  }

let gather_ranges ~ops ~dst ~starts ~ends x =
  let ys = shape_of dst in
  if numel ys = 0 then Ok None
  else
    fitting @@ fun () ->
    let b = Plan.make () in
    let ws = Plan.workspace b in
    let sh = shape_of starts in
    let lead = Array.length sh - 1 in
    let k = sh.(lead) and size = ys.(lead) in
    let rows = numel (Array.sub sh 0 lead) in
    let chunk = Atomic.get chunk_bound in
    let chunks n = (n + chunk - 1) / chunk in
    (* The rows' blocks, then each level's chunks' maxima. *)
    let rec bytes n =
      let c = chunks n in
      (8 * rows * c) + if c = 1 then 0 else bytes c
    in
    (* Without runs, every position reads past every end. *)
    let maxima =
      if k = 0 then 0
      else
        let at = Plan.take b ((8 * rows * (k + 2)) + bytes k) in
        runs b ~ops ~at ~rows ~k starts ends;
        at
    in
    (* The gather at each destination position's source position, which a region
       of the workspace holds for a chunk of positions at a time, -1 past every
       end and outside the source's axis: 32 bits where they fit. The positions'
       launches of a chunk run after the gather of the one before. *)
    let d = (shape_of x).(lead) in
    let narrow = d < 1 lsl Atomic.get narrow_bound in
    let width = position_bytes ~narrow in
    let count = rows * size in
    let chunk = Int.min count (Atomic.get region_bound / width) in
    let region = Plan.take b (width * chunk) in
    let ps =
      L.strides (L.contiguous (Iarray.of_array (Array.sub ys 0 (lead + 1))))
    in
    let ds = L.strides (L.contiguous (Iarray.of_array ys)) in
    let xs = strides_of x in
    let axes =
      Array.mapi
        (fun a e ->
          ax e ~sd:(Iarray.get ds a)
            ~ss:(if a = lead then 0 else xs.(a))
            ~si:(if a <= lead then Iarray.get ps a else 0))
        ys
    in
    let m =
      plan_move b
        ~mode:(Gather { idx = (ws, region); d; narrow })
        ~bits:(bits_of dst) ~dst:(site b ops (A.Any dst))
        ~src:(site b ops (A.Any x)) ~sa:xs.(lead) axes
    in
    (* A position's cells are [per] units, consecutive in C order. *)
    let per = m.count / count in
    let wide = is_wide count in
    let w = Plan.words K.Runs_positions in
    Plan.set w K.positions_dst region;
    Plan.set w K.positions_maxima maxima;
    Plan.set w K.positions_how
      ((Bool.to_int wide lsl K.how_wide)
      lor (Bool.to_int narrow lsl K.how_narrow));
    Plan.set w K.positions_k k;
    Plan.set w K.positions_size size;
    Bytes.set_int64_le w (8 * K.positions_divisor) (divisor ~wide size);
    Plan.set w K.positions_ss (strides_of starts).(lead);
    Plan.set w K.positions_es (strides_of ends).(lead);
    Plan.set w K.positions_d d;
    let operand word a =
      Plan.set w word (8 * L.offset (A.layout a));
      { Plan.word; slot = slot b ops a; write = false }
    in
    let refs =
      Array.of_list
        ({ Plan.word = K.positions_dst; slot = ws; write = true }
        :: { Plan.word = K.positions_maxima; slot = ws; write = false }
        ::
        (if k = 0 then []
         else
           [ operand K.positions_starts starts; operand K.positions_ends ends ])
        )
    in
    let from = ref 0 in
    while !from < count do
      let upto = Int.min count (!from + chunk) in
      Plan.set w K.positions_from !from;
      launches b K.Runs_positions ~first:!from
        ~at:(K.positions_first, K.positions_count)
        ~resident:(Plan.holds ~threads:(K.threads K.Runs_positions) 0)
        ~cost:(positions_cost ~width ~k ~rows)
        ~count:upto w refs;
      launch_move b m ~from:!from ~first:(!from * per) ~upto:(upto * per);
      from := upto
    done;
    Ok (Some (Plan.finish ~most:8 b))

(* Assemblies *)

(* Whether [r] is the whole of an axis of extent [n]. *)
let whole n (r : M.range) = r.start = 0 && r.step = 1 && r.count = n

(* Whether [x], an assembly's first piece of region [rs], is [dst]'s elements in
   place: of its buffer and layout, of the whole of it. Its elements are already
   there. *)
let in_place ~dst (rs, x) =
  A.buffer x == A.buffer dst
  && L.equal (A.layout x) (A.layout dst)
  && Iarray.for_all2 whole (L.shape (A.layout dst)) rs

(* The bits of [v], an element of [dt], repeated over 16 bytes, as two
   little-endian words. *)
let pattern (type v s) (dt : (v, s) D.t) (v : v) =
  let s = Nx_kernel.Prog.bits dt v in
  let bits = D.bits dt in
  let b = Bytes.make 16 '\000' in
  if bits >= 8 then
    for i = 0 to (16 / String.length s) - 1 do
      Bytes.blit_string s 0 b (i * String.length s) (String.length s)
    done
  else Bytes.set_uint8 b 0 (Char.code s.[0]);
  (Bytes.get_int64_le b 0, Bytes.get_int64_le b 8)

(* A region's box, the least and greatest index along each axis, if each of its
   ranges steps by one in either direction. *)
let box (rs : M.range iarray) =
  if Iarray.exists (fun (r : M.range) -> r.count > 1 && abs r.step <> 1) rs then
    None
  else
    Some
      (Array.map
         (fun (r : M.range) ->
           let last = r.start + ((r.count - 1) * r.step) in
           (Int.min r.start last, Int.max r.start last))
         (Iarray.to_array rs))

(* [d], a site, moved [n] elements of [bits] bits on, [n] of either sign. *)
let shift d ~bits n =
  if bits >= 8 then { d with at = d.at + (n * bits / 8) }
  else
    let bit = d.bit + (n * bits) in
    let word = if bit >= 0 then bit / 32 else -((31 - bit) / 32) in
    { d with at = d.at + (4 * word); bit = bit - (32 * word) }

(* The boxes, of an index range each along each axis of [ys], that hold every
   element of [ys] outside the box [b], none of them twice: along each axis [j],
   the ranges before and after [b]'s, within [b]'s ranges along the axes before
   [j]. *)
let complement ys b =
  let r = Array.length ys in
  List.concat_map
    (fun j ->
      let slab (lo, hi) =
        Array.init r (fun i ->
            if i < j then b.(i) else if i = j then (lo, hi) else (0, ys.(i) - 1))
      in
      let lo, hi = b.(j) in
      List.filter_map
        (fun (l, h) -> if l > h then None else Some (slab (l, h)))
        [ (0, lo - 1); (hi + 1, ys.(j) - 1) ])
    (List.init r Fun.id)

(* Adds the launches that write [v] into each element of [dst], of shape [ys] at
   [d], outside the box [skip]: a move of the fill's pattern into each box of
   {!complement}, or of the whole of [dst] without [skip]. *)
let fill b ~d ~ys ~skip (type v s) (dst : (v, s) A.t) (v : v) =
  let dt = A.dtype dst in
  let bits = D.bits dt in
  let lo, hi = pattern dt v in
  let cs = L.strides (L.contiguous (Iarray.of_array ys)) in
  let boxes =
    match skip with
    | None -> [ Array.map (fun e -> (0, e - 1)) ys ]
    | Some box -> complement ys box
  in
  List.iter
    (fun box ->
      let base = ref 0 in
      Array.iteri (fun j (l, _) -> base := !base + (l * Iarray.get cs j)) box;
      let d = shift d ~bits !base in
      let axes =
        Array.mapi
          (fun j (l, h) ->
            let s = Iarray.get cs j in
            ax (h - l + 1) ~sd:s ~ss:s ~si:0)
          box
      in
      move b ~mode:(Fill { lo; hi }) ~bits ~dst:d ~src:d axes)
    boxes

let assemble ~ops ~fill:v ~dst pieces =
  let ys = shape_of dst in
  let n = numel ys in
  if n = 0 then Ok None
  else
    fitting @@ fun () ->
    let b = Plan.make () in
    let pieces = Iarray.to_list pieces in
    (* The first piece in place leaves its elements, every one of dst's. *)
    let pieces, covered =
      match pieces with
      | p :: rest when in_place ~dst p -> (rest, true)
      | _ -> (pieces, false)
    in
    let pieces = List.filter (fun (_, x) -> numel (shape_of x) > 0) pieces in
    let regions = Array.of_list (List.map fst pieces) in
    let covered = covered || M.covers (Iarray.of_array ys) regions in
    let d = site b ops (A.Any dst) in
    (* The fill skips the largest box of a piece, which the piece's move writes
       whole after it. *)
    if not covered then begin
      let volume = Array.fold_left (fun n (l, h) -> n * (h - l + 1)) 1 in
      let skip =
        List.fold_left
          (fun best (rs, _) ->
            match (box rs, best) with
            | Some x, Some y when volume x <= volume y -> best
            | Some x, _ -> Some x
            | None, _ -> best)
          None pieces
      in
      fill b ~d ~ys ~skip dst v
    end;
    let cs = L.strides (L.contiguous (Iarray.of_array ys)) in
    List.iter
      (fun (rs, x) ->
        let base = ref 0 in
        Iarray.iteri
          (fun a (r : M.range) -> base := !base + (r.start * Iarray.get cs a))
          rs;
        let xs = strides_of x in
        let axes =
          Array.mapi
            (fun a e ->
              let r : M.range = Iarray.get rs a in
              ax e ~sd:(r.step * Iarray.get cs a) ~ss:xs.(a) ~si:0)
            (shape_of x)
        in
        (* The piece's place in dst: its region's first element's. *)
        let bits = bits_of dst in
        let dst = shift d ~bits !base in
        move b ~mode:Copy ~bits ~dst ~src:(site b ops (A.Any x)) axes)
      pieces;
    Ok (Plan.finish_some ~most:8 b)

(* Folds *)

(* An axis of a fold, as kernels.h's Folds records it: [dst]'s extent along it;
   the padded position of its first index and the spacing of the rest; and its
   window's taps, dilation and step, its windows, and [x]'s strides along the
   windows and the taps. An axis of no window is one of one tap and step 1 whose
   windows are its padded positions. *)
type fold_axis = {
  extent : int;
  lo : int;
  space : int;
  taps : int;
  dil : int;
  step : int;
  nwin : int;
  sw : int;
  st : int;
}

let rec gcd x y = if y = 0 then x else gcd y (x mod y)

(* The taps of window [a] that read one padded position lie this many apart, and
   their windows [drop a]. *)
let apart a = a.step / gcd a.step a.dil
let drop a = a.dil / gcd a.step a.dil

(* Whether [a]'s first tap that reads a position comes at once, by a division by
   [apart a]: where its dilation divides its step. *)
let direct a = a.step mod a.dil = 0

(* [a]'s record at word [at] of [s], [span] its tap's weight. *)
let put_fold s ~at ~wide ~span a =
  let word i v = Bytes.set_int64_le s (8 * (at + i)) (Int64.of_int v) in
  word K.fold_axis_extent a.extent;
  Bytes.set_int64_le s (8 * (at + K.fold_axis_divisor)) (divisor ~wide a.extent);
  word K.fold_axis_lo a.lo;
  word K.fold_axis_space a.space;
  word K.fold_axis_taps a.taps;
  word K.fold_axis_dil a.dil;
  word K.fold_axis_step a.step;
  Bytes.set_int64_le s
    (8 * (at + K.fold_axis_step_divisor))
    (divisor ~wide a.step);
  Bytes.set_int64_le s
    (8 * (at + K.fold_axis_dil_divisor))
    (divisor ~wide a.dil);
  word K.fold_axis_apart (apart a);
  Bytes.set_int64_le s
    (8 * (at + K.fold_axis_apart_divisor))
    (divisor ~wide (apart a));
  word K.fold_axis_direct (Bool.to_int (direct a));
  word K.fold_axis_drop (drop a);
  word K.fold_axis_nwin a.nwin;
  word K.fold_axis_sw a.sw;
  word K.fold_axis_st a.st;
  word K.fold_axis_span span

(* The most taps of window [a] that read one padded position: they lie [apart a]
   apart, each in a window of its own. *)
let reads_of a = Int.min a.nwin ((a.taps + apart a - 1) / apart a)

(* The taps a scan of window [a] steps at most to find one that reads: one where
   it comes at once. *)
let scan_of a = if direct a then 1 else Int.min a.taps (apart a)

(* The divisions a scan of window [a] takes: two by its dilation, for its least
   and most taps; one by its step for each tap it steps, and where the first
   comes at once, one by [apart a] before it. The kernel skips a division by
   1. *)
let divisions a =
  (if a.dil > 1 then 2 else 0)
  + (if a.step > 1 then scan_of a else 0)
  + if direct a && apart a > 1 then 1 else 0

(* A fold's addition in float32 operations: an addition, a narrow float's two
   conversions and a rounding; and in links, a narrow float's conversions and
   rounding, which the next addition waits for. *)
let add_ops = 64
let add_links ~narrow = if narrow then 8 else 0

(* A window's step of the odometer, its share of a read's address and its carry,
   in operations and links, by kernel: in registers in fold_0 to fold_3; in
   thread arrays, which fold_n and fold_w index at run time; and in 64-bit
   values in fold_w. Fit on the M1 Max to each kernel's launches of long tap
   chains, at most 0.80 of the estimate: fold_3's 27 taps resumed in launches of
   1 ms took 0.80 of it at 8 links a step; fold_w's 9 taps over 32-bit positions
   held in 64 took 0.74. *)
let step_ops k = match k with K.Fold_w -> 176 | K.Fold_n -> 160 | _ -> 64
let step_links k = match k with K.Fold_w -> 20 | K.Fold_n -> 6 | _ -> 10

(* The cost of [m] elements of [width] bytes, [narrow] floats or not, in
   [groups] groups of kernel [k]'s fold of [rank] axes over [l] tap numbers,
   whose windows [wins] read at most [reads] taps an element: each element's
   write, its reads of x and, where [resume], of its sum; a division an axis and
   each division of a scan of a window's taps for its first tap and, where
   [resume], two more scans for its seek; a step of each scan and of each window
   a read; and a chain of a read, its addition and its windows' steps a tap. *)
let fold_cost k ~wide ~rank ~width ~narrow ~wins ~reads ~resume ~l m groups =
  let reads = Int.min l reads in
  let w = List.length wins in
  let scans = if resume then 3 else 1 in
  let sum f = List.fold_left (fun n a -> n + f a) 0 wins in
  let divs = scans * sum divisions in
  let steps = (scans * sum (fun a -> 2 + scan_of a)) + (w * reads) in
  let ops =
    unit_ops
    + (rank * axis_ops ~wide)
    + (divs * division ~wide)
    + (steps * step_ops k)
    + (reads * add_ops)
  in
  {
    Plan.bytes =
      (m * width * (1 + (2 * reads) + Bool.to_int resume))
      + (2 * Plan.line * groups);
    ops = K.threads K.Fold_n * groups * ops;
    chain =
      ((reads + 1) * Plan.read_links)
      + (reads * add_links ~narrow)
      + (steps * step_links k);
  }

(* Folds of windows apart *)

(* Whether no two taps of a fold over [axes] read one padded position: without
   interior padding, each window's taps reach no further than its step. Each
   element of [dst] is then one element of [x], or [+0]. *)
let apart_windows axes =
  Array.for_all
    (fun a -> a.space = 1 && ((a.taps - 1) * a.dil) + 1 <= a.step)
    axes

(* The quotient of [v] by [d] > 0, rounded down and up. *)
let floor_div v d = if v >= 0 then v / d else -((d - 1 - v) / d)
let ceil_div v d = -(floor_div (-v) d)

(* A box of a fold axis's (window, tap) pairs: windows [q], [q] + [nq] - 1 and
   their taps [j], [j] + [nj] - 1. *)
type pairs = { q : int; nq : int; j : int; nj : int }

(* The pairs of axis [a] whose taps land in [dst], window [q] tap [j] at index
   [q step + j dil - lo]. Since [j dil < step], they follow each other in order
   of window then tap, from window [q0]'s tap [j0] to window [q1]'s tap [j1]:
   the first window's from [j0] on, whole windows, then the last window's up to
   [j1], at most three boxes, none where no tap lands. *)
let landing a =
  let last = a.extent - 1 + a.lo in
  let q0 = Int.max 0 (ceil_div (a.lo - ((a.taps - 1) * a.dil)) a.step) in
  let q1 = Int.min (a.nwin - 1) (floor_div last a.step) in
  let j0 = Int.max 0 (ceil_div (a.lo - (q0 * a.step)) a.dil) in
  let j1 = Int.min (a.taps - 1) (floor_div (last - (q1 * a.step)) a.dil) in
  let box q q' j j' = { q; nq = q' - q + 1; j; nj = j' - j + 1 } in
  if q0 > q1 || (q0 = q1 && j0 > j1) then []
  else if q0 = q1 then [ box q0 q0 j0 j1 ]
  else
    let cut0 = j0 > 0 and cut1 = j1 < a.taps - 1 in
    let w0 = if cut0 then q0 + 1 else q0 and w1 = if cut1 then q1 - 1 else q1 in
    List.concat
      [
        (if cut0 then [ box q0 q0 j0 (a.taps - 1) ] else []);
        (if w0 <= w1 then [ box w0 w1 0 (a.taps - 1) ] else []);
        (if cut1 then [ box q1 q1 0 j1 ] else []);
      ]

(* The index of [dst] at which pair ([q], [j]) of axis [a] lands. *)
let index_of a ~q ~j = (q * a.step) + (j * a.dil) - a.lo

(* Adds the launches of a fold over [axes] whose windows are apart
   ({!apart_windows}), of [x] at [src ()] into [dst] at [d]: [+0] where no tap
   lands, as {!fill} writes around the box the taps cover where their indices
   along each axis follow each other, and everywhere otherwise; then a move of
   each product of an axis's boxes ({!landing}), at strides [step] and [dil]
   along [dst]. Raises [Over_budget] as they do. *)
let fold_moves b ~d ~src axes (type v s) (dst : (v, s) A.t) =
  let ys = shape_of dst in
  let bits = bits_of dst in
  let zero = D.zero (A.dtype dst) in
  let cs = L.strides (L.contiguous (Iarray.of_array ys)) in
  let boxes = Array.map landing axes in
  (* Along an axis whose taps land on consecutive indices, they cover those from
     its first pair's to its last's. *)
  let follows a = a.dil = 1 && a.step = a.taps in
  let span a l =
    let first = List.hd l and final = List.nth l (List.length l - 1) in
    ( index_of a ~q:first.q ~j:first.j,
      index_of a ~q:(final.q + final.nq - 1) ~j:(final.j + final.nj - 1) )
  in
  if Array.exists (( = ) []) boxes || not (Array.for_all follows axes) then
    fill b ~d ~ys ~skip:None dst zero
  else begin
    let box = Array.map2 span axes boxes in
    let whole (l, h) e = l = 0 && h = e - 1 in
    if not (Array.for_all2 whole box ys) then
      fill b ~d ~ys ~skip:(Some box) dst zero
  end;
  (* A product of an axis's box each: its first element's place in [dst] and in
     [x], and along each axis two of the move's, windows then taps. *)
  let move_of ps =
    let at = ref 0 and from = ref 0 in
    let axes =
      List.mapi
        (fun i p ->
          let a = axes.(i) and c = Iarray.get cs i in
          at := !at + (index_of a ~q:p.q ~j:p.j * c);
          from := !from + (p.q * a.sw) + (p.j * a.st);
          [
            ax p.nq ~sd:(a.step * c) ~ss:a.sw ~si:0;
            ax p.nj ~sd:(a.dil * c) ~ss:a.st ~si:0;
          ])
        ps
    in
    let dst = shift d ~bits !at and src = shift (src ()) ~bits !from in
    move b ~mode:Copy ~bits ~dst ~src (Array.of_list (List.concat axes))
  in
  let rec products i =
    if i = Array.length boxes then [ [] ]
    else
      let rest = products (i + 1) in
      List.concat_map (fun p -> List.map (List.cons p) rest) boxes.(i)
  in
  List.iter move_of (products 0)

(* The launches of a fold over [axes] apart: a product of one box an axis
   each. *)
let moves_of axes =
  Array.fold_left (fun m a -> m * List.length (landing a)) 1 axes

let fold ~ops (pad : Nx_kernel.pad) ~dst x =
  let ys = shape_of dst in
  let n = numel ys in
  if n = 0 then Ok None
  else
    fitting @@ fun () ->
    let b = Plan.make () in
    let r = Array.length ys in
    let xshape = shape_of x and xs = strides_of x in
    let empty = numel xshape = 0 in
    let window a =
      let found = ref None in
      Iarray.iteri
        (fun v (w : M.window) -> if w.axis = a then found := Some (v, w))
        pad.windows;
      !found
    in
    (* An operand of no element has no window along some axis, since every
       window has a tap: every axis is one of one tap, and that axis's [nwin] of
       0 reads no element. *)
    let axes =
      Array.init r (fun a ->
          let lo = Iarray.get pad.lo a
          and space = Iarray.get pad.interior a + 1 in
          match window a with
          | Some (v, w) when w.size > 1 && not empty ->
              {
                extent = ys.(a);
                lo;
                space;
                taps = w.size;
                dil = w.dilation;
                step = w.step;
                nwin = xshape.(a);
                sw = xs.(a);
                st = xs.(r + v);
              }
          | Some (_, w) ->
              {
                extent = ys.(a);
                lo;
                space;
                taps = 1;
                dil = 1;
                step = w.step;
                nwin = xshape.(a);
                sw = xs.(a);
                st = 0;
              }
          | None ->
              {
                extent = ys.(a);
                lo;
                space;
                taps = 1;
                dil = 1;
                step = 1;
                nwin = xshape.(a);
                sw = xs.(a);
                st = 0;
              })
    in
    let wins = List.filter (fun a -> a.taps > 1) (Array.to_list axes) in
    let nw = List.length wins in
    let total = List.fold_left (fun t a -> t * a.taps) 1 wins in
    let reads = List.fold_left (fun t a -> t * reads_of a) 1 wins in
    (* 31 bits hold every padded position, step and reach below 2^32
       elements. *)
    let top = 1 lsl Atomic.get taps_bound in
    let wide =
      is_wide n
      || Array.exists
           (fun a ->
             let last = a.lo + ((a.extent - 1) * a.space) in
             List.exists
               (fun v -> abs v >= top)
               [ a.lo; last; a.step; a.taps * a.dil; a.nwin ])
           axes
    in
    (* Each window's tap weight: the product of the later windows' taps. *)
    let spans =
      snd
        (List.fold_left
           (fun (span, acc) a -> (span * a.taps, span :: acc))
           (1, []) (List.rev wins))
    in
    let d = site b ops (A.Any dst) in
    (* Windows apart move their elements, unless their moves' dispatches alone
       would take a launch's time: a fold of many axes each cut at both ends,
       which the fold kernel computes in one launch. *)
    if
      apart_windows axes
      && float (moves_of axes) *. Plan.dispatch <= Plan.allowed ()
    then begin
      fold_moves b ~d ~src:(fun () -> site b ops (A.Any x)) axes dst;
      Ok (Plan.finish_some ~most:8 b)
    end
    else
      let src = site b ops (A.Any x) in
      let dt = A.dtype dst in
      (* 32 bits hold each element index, padded position, position in x, tap
         and tap number where each lies below 2^30, so that a sum of two lies
         below 2^31. *)
      let reach =
        Array.fold_left ( + ) 0
          (Array.mapi (fun a e -> Int.max 0 (e - 1) * abs xs.(a)) xshape)
      in
      let small v = abs v < top / 2 in
      let narrow =
        List.for_all small [ n; reach; total ]
        && Array.for_all
             (fun a ->
               List.for_all small
                 [
                   abs a.lo + ((a.extent - 1) * a.space);
                   a.taps * a.dil;
                   a.nwin * a.step;
                 ])
             axes
      in
      let k =
        if wide || not narrow then K.Fold_w
        else
          match nw with
          | 0 -> K.Fold_0
          | 1 -> K.Fold_1
          | 2 -> K.Fold_2
          | 3 -> K.Fold_3
          | _ -> K.Fold_n
      in
      let w = Plan.words k in
      Plan.set w K.fold_dst d.at;
      Plan.set w K.fold_x src.at;
      Plan.set w K.fold_how
        ((Bool.to_int wide lsl K.how_wide)
        lor (d.bit lsl K.how_dst_bit)
        lor (src.bit lsl K.how_src_bit)
        lor (D.code dt lsl K.how_dtype)
        lor (r lsl K.how_rank));
      Plan.set w K.fold_windows nw;
      let width = Int.max 1 (D.bits dt / 8) in
      let threads = K.threads k in
      let resident = Plan.holds ~threads 0 in
      let narrow = D.is D.Float dt && D.bits dt < 32 in
      let cost ~resume ~l =
        fold_cost k ~wide ~rank:r ~width ~narrow ~wins ~reads ~resume ~l
      in
      (* Tap numbers in ranges as long as an element's launch allows. *)
      let fits l =
        most ~threads ~per:threads ~resident ~count:1 (cost ~resume:true ~l) > 0
      in
      let rec range l = if l <= 1 || fits l then l else range ((l + 1) / 2) in
      let l = range total in
      let ranges = (total + l - 1) / l in
      (* The constant: the axes' records, the windows', then each later range's
         first tap number's digits, its tap along each window. *)
      let records = Array.append axes (Array.of_list wins) in
      let seek = K.fold_axis_words * Array.length records in
      let s = Bytes.make (8 * Int.max 1 (seek + ((ranges - 1) * nw))) '\000' in
      Array.iteri
        (fun i a ->
          let span = if i < r then 0 else List.nth spans (i - r) in
          put_fold s ~at:(K.fold_axis_words * i) ~wide ~span a)
        records;
      for i = 1 to ranges - 1 do
        List.iteri
          (fun u (a, span) ->
            let digit = i * l / span mod a.taps in
            Bytes.set_int64_le s
              (8 * (seek + ((i - 1) * nw) + u))
              (Int64.of_int digit))
          (List.combine wins spans)
      done;
      let t = Bytes.unsafe_to_string s in
      let c =
        Plan.constant b ~key:("fold " ^ t) ~length:(String.length t) (fun () ->
            Moves.bigbytes t)
      in
      let refs =
        [|
          { Plan.word = K.fold_dst; slot = d.slot; write = true };
          { Plan.word = K.fold_x; slot = src.slot; write = false };
          { Plan.word = K.fold_axes; slot = c; write = false };
        |]
      in
      for i = 0 to ranges - 1 do
        let t0 = i * l in
        let t1 = Int.min total (t0 + l) in
        let w = Bytes.copy w in
        Plan.set w K.fold_t0 t0;
        Plan.set w K.fold_t1 t1;
        Plan.set w K.fold_seek (seek + (Int.max 0 (i - 1) * nw));
        launches b k
          ~at:(K.fold_first, K.fold_count)
          ~resident
          ~cost:(cost ~resume:(i > 0) ~l:(t1 - t0))
          ~count:n w refs
      done;
      Ok (Plan.finish_some ~most:2 b)
