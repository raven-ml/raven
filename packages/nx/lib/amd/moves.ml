(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Moves and their adjoints: each call planned as launches of moves.hip, their
   words written at kernels.h's indices, as one call ({!Plan}).

   A launch's grid follows its work alone ({!Plan.grid}): every kernel strides
   over its work, so its bits do not depend on the grid. A launch writes the
   words its kernel stages, no more ([used]).

   A float Add sums in float32 or float64. A complex one runs as the float Add
   of its parts: the arrays read as floats with a last axis of the two parts,
   along which idx steps 0, so that the row tiles sum both parts of a target
   together, each part by its own position tree. *)

module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module K = Kernels

(* Launches *)

let read word slot = { Plan.word; slot; write = false }
let write word slot = { Plan.word; slot; write = true }
let set = Plan.set

(* The words a launch of [k] with words [w] reads, as moves.hip stages them: the
   axes' words only where its kernel reads them, by the rank [w] holds. *)
let used (k : K.t) w =
  let get i = Int64.to_int (Bytes.get_int64_le w (8 * i)) in
  match k with
  | Move -> K.move_ax + (get K.move_rank * K.move_axis_words)
  | Runs_chunk -> K.runs_ax + (get K.runs_lead * K.runs_axis_words)
  | Runs_carry | Runs_merge -> K.runs_ax
  | Assemble ->
      let r = get K.assemble_rank in
      K.assemble_dn + (K.assemble_dn_axis * r)
      + (get K.assemble_n * (K.assemble_head + (r * K.assemble_axis)))
  | Fold -> K.fold_ax + (get K.fold_rank * K.fold_axis_words)
  | Scatter | Scatter_tile_f32 | Scatter_tile_f64 | Scatter_rows_f32
  | Scatter_rows_f64 | Scatter_tally | Scatter_pack_n | Scatter_pack_w
  | Scatter_count_n | Scatter_count_w | Scatter_move_n | Scatter_move_w
  | Scatter_sum_f32_n | Scatter_sum_f32_w | Scatter_sum_f64_n
  | Scatter_sum_f64_w ->
      K.scatter_ax + (get K.scatter_rank * K.scatter_axis_words)
  | k -> K.words k (* no move launches it *)

(* Adds to [b] a launch of [k] whose groups stride over [work] units, one a
   work-item, their count in word [groups] of [w]. *)
let launch b k ~groups ~work ?shared w refs =
  let g = Plan.grid k work in
  set w groups g;
  Plan.launch b k ~groups:(g, 1, 1) ?shared ~used:(used k w) w
    (Array.of_list refs)

(* Arrays *)

let shape_of a = Iarray.to_array (L.shape (A.layout a))
let strides_of a = Iarray.to_array (L.strides (A.layout a))
let offset_of a = L.offset (A.layout a)
let numel s = Array.fold_left ( * ) 1 s
let code a = D.code (A.dtype a)
let slot b a = Plan.slot b (A.buffer a)

(* An element's units, of its dtype's alignment, and their count: two for a
   complex number, a part each. *)
let unit_count a =
  let dt = A.dtype a in
  if D.is D.Complex dt then (D.bits dt / 2, 2) else (D.bits dt, 1)

(* C-order strides of [s]. *)
let c_strides s =
  let r = Array.length s in
  let st = Array.make r 1 in
  for i = r - 2 downto 0 do
    st.(i) <- st.(i + 1) * s.(i + 1)
  done;
  st

let small n = Bool.to_int (Plan.small n)

let rec ceil_log2 ?(b = 0) n =
  if 1 lsl b >= n then b else ceil_log2 ~b:(b + 1) n

(* A sub-byte destination's work: its 32-bit words. *)
let work_of ~unit n = if unit < 8 then (n * unit / 32) + 2 else n
let up8 x = (x + 7) land lnot 7

(* Moves *)

(* The axes of [shape], each with its strides [axes a] of the source and the
   positions, as fewer axes that reach the same elements in the same C order:
   an axis of one element dropped, two adjacent axes merged where both strides
   lay them out as one, but for axis [keep]. Gives the shape, the strides and
   [keep]'s new index, -1 for none. *)
let merge ~keep shape axes =
  let out = ref [] and kept = ref (-1) in
  Array.iteri
    (fun a e ->
      let ss, si = axes a in
      if a = keep then begin
        kept := List.length !out;
        out := (e, ss, si) :: !out
      end
      else if e <> 1 then
        match !out with
        | (e', ss', si') :: rest
          when List.length rest <> !kept && ss' = ss * e && si' = si * e ->
            out := (e' * e, ss, si) :: rest
        | _ -> out := (e, ss, si) :: !out)
    shape;
  let out = Array.of_list (List.rev !out) in
  ( Array.map (fun (e, _, _) -> e) out,
    Array.map (fun (_, ss, si) -> (ss, si)) out,
    !kept )

(* A wave's work-items. *)
let wave = 32

(* The chunks of K.move_chunk bytes of a run of [bytes]. *)
let run_chunks bytes = (bytes + K.move_chunk - 1) / K.move_chunk

(* Whether a row of [n] elements is a run: of K.move_run bytes or more, its
   elements of a byte or more. *)
let is_run ~unit ~count n = unit >= 8 && n * unit / 8 * count >= K.move_run

(* The chunks a run of the last axis spans; 0 where the last axis is no run. *)
let chunks ~unit ~count shape axes =
  let r = Array.length shape in
  if r = 0 || axes.(r - 1) <> (1, 0) then 0
  else if not (is_run ~unit ~count shape.(r - 1)) then 0
  else run_chunks (shape.(r - 1) * unit / 8 * count)

(* [dst], C-contiguous over [shape] from [dst_off] in slot [dst], gets [mode]'s
   elements: [axes a] is axis [a]'s strides of the source and of the positions.
   [runs] are the workspace's bytes of a range gather's starts, ends and
   maxima. *)
let move b ~dst ~dst_off ~unit:(unit, count) ~mode ~shape ~axes ?(axis = -1)
    ?src ?(src_off = 0) ?idx ?(idx_off = 0) ?runs ?(d = 0) ?(sa = 0) ?(k = 0) ()
    =
  let n = numel shape in
  if n > 0 then begin
    let w = Plan.words K.Move in
    let refs = ref [ write K.move_dst dst ] in
    Option.iter (fun s -> refs := read K.move_src s :: !refs) src;
    Option.iter (fun s -> refs := read K.move_idx s :: !refs) idx;
    Option.iter
      (fun (s, e, m) ->
        let ws = Plan.workspace b in
        set w K.move_starts s;
        set w K.move_ends e;
        set w K.move_maxima m;
        refs :=
          read K.move_starts ws :: read K.move_ends ws :: read K.move_maxima ws
          :: !refs)
      runs;
    set w K.move_numel n;
    set w K.move_dst_off dst_off;
    set w K.move_src_off src_off;
    set w K.move_idx_off idx_off;
    set w K.move_d d;
    set w K.move_sa sa;
    set w K.move_k k;
    (* A range gather reads its position along [axis]: that axis stays. *)
    let keep = if mode = K.move_ranges then axis else -1 in
    let shape, axes, axis = merge ~keep shape axes in
    let r = Array.length shape in
    set w K.move_rank r;
    set w K.move_axis axis;
    set w K.move_mode mode;
    set w K.move_small (small n);
    set w K.move_unit unit;
    set w K.move_count count;
    let cs = c_strides shape in
    for a = 0 to r - 1 do
      let at = K.move_ax + (a * K.move_axis_words) in
      let ss, si = axes.(a) in
      Plan.axis w at shape.(a);
      set w (at + K.move_axis_sd) cs.(a);
      set w (at + K.move_axis_ss) ss;
      set w (at + K.move_axis_si) si
    done;
    let c = chunks ~unit ~count shape axes in
    Plan.axis w K.move_chunks c;
    (* A run's chunk a wave. *)
    let work =
      if c = 0 then work_of ~unit n else n / shape.(r - 1) * c * wave
    in
    launch b K.Move ~groups:K.move_groups ~work w (List.rev !refs)
  end

(* Copies [src] into [dst], C-contiguous of its shape. *)
let copy b ~dst src =
  let ss = strides_of src in
  move b ~dst:(slot b dst) ~dst_off:(offset_of dst) ~unit:(unit_count dst)
    ~mode:K.move_copy ~shape:(shape_of src)
    ~axes:(fun a -> (ss.(a), 0))
    ~src:(slot b src) ~src_off:(offset_of src) ()

let gather ~axis ~dst idx x =
  let b = Plan.make () in
  let xs = strides_of x and is = strides_of idx in
  move b ~dst:(slot b dst) ~dst_off:(offset_of dst) ~unit:(unit_count dst)
    ~mode:K.move_gather ~shape:(shape_of dst) ~axis
    ~axes:(fun a -> ((if a = axis then 0 else xs.(a)), is.(a)))
    ~src:(slot b x) ~src_off:(offset_of x) ~idx:(slot b idx)
    ~idx_off:(offset_of idx)
    ~d:(shape_of x).(axis)
    ~sa:xs.(axis) ();
  Ok (Plan.finish_some b)

(* Range gathers *)

(* The runs' launches: each row's starts, ends and their running maximum into
   the workspace from byte [at], [rows] rows of [k], each chunk's maximum
   after. *)
let runs b ~at ~rows ~k ~chunks starts ends =
  let lead = L.rank (A.layout starts) - 1 in
  let ss = strides_of starts and es = strides_of ends in
  let sh = shape_of starts in
  let row_bytes = 8 * rows * k in
  let w = Plan.words K.Runs_chunk in
  let ws = Plan.workspace b in
  set w K.runs_starts at;
  set w K.runs_ends (at + row_bytes);
  set w K.runs_maxima (at + (2 * row_bytes));
  set w K.runs_carry (at + (3 * row_bytes));
  set w K.runs_rows rows;
  set w K.runs_k k;
  set w K.runs_chunks chunks;
  set w K.runs_s_k ss.(lead);
  set w K.runs_e_k es.(lead);
  set w K.runs_s_off (offset_of starts);
  set w K.runs_e_off (offset_of ends);
  set w K.runs_lead lead;
  set w K.runs_small (small rows);
  for a = 0 to lead - 1 do
    let ax = K.runs_ax + (a * K.runs_axis_words) in
    Plan.axis w ax sh.(a);
    set w (ax + K.runs_axis_ss) ss.(a);
    set w (ax + K.runs_axis_se) es.(a)
  done;
  let refs =
    [
      read K.runs_in_s (slot b starts);
      read K.runs_in_e (slot b ends);
      write K.runs_starts ws;
      write K.runs_ends ws;
      write K.runs_maxima ws;
      write K.runs_carry ws;
    ]
  in
  let step k ~work =
    launch b k ~groups:K.runs_groups ~work (Bytes.copy w) refs
  in
  step K.Runs_chunk ~work:(rows * chunks * K.move_threads);
  if chunks > 1 then begin
    step K.Runs_carry ~work:(rows * K.move_threads);
    step K.Runs_merge ~work:(rows * k)
  end

let gather_ranges ~dst ~starts ~ends x =
  (* An empty [dst] has no element to write, so the call plans nothing, its runs
     included. *)
  if numel (shape_of dst) = 0 then Ok None
  else
    let b = Plan.make () in
    let sh = shape_of starts in
    let lead = Array.length sh - 1 in
    let k = sh.(lead) in
    let rows = numel (Array.sub sh 0 lead) in
    let chunks = (k + K.runs_chunk - 1) / K.runs_chunk in
    let row_bytes = 8 * rows * k in
    let at = Plan.take b ((3 * row_bytes) + (8 * rows * chunks)) in
    if rows * k > 0 then runs b ~at ~rows ~k ~chunks starts ends;
    (* Position j of a row's runs is word (row k + j) of each table. *)
    let row_st = c_strides (Array.append (Array.sub sh 0 lead) [| k |]) in
    let xs = strides_of x in
    move b ~dst:(slot b dst) ~dst_off:(offset_of dst) ~unit:(unit_count dst)
      ~mode:K.move_ranges ~shape:(shape_of dst) ~axis:lead
      ~axes:(fun a ->
        ((if a = lead then 0 else xs.(a)), if a < lead then row_st.(a) else 0))
      ~src:(slot b x) ~src_off:(offset_of x)
      ~runs:(at, at + row_bytes, at + (2 * row_bytes))
      ~d:(shape_of x).(lead)
      ~sa:xs.(lead) ~k ();
    Ok (Plan.finish_some b)

(* Assemblies *)

(* The bits of [v], an element of [dt], as two little-endian words. *)
let fill_bits (type v s) (dt : (v, s) D.t) (v : v) =
  let s = Nx_kernel.Prog.bits dt v in
  let b = Bytes.make 16 '\000' in
  Bytes.blit_string s 0 b 0 (String.length s);
  (Bytes.get_int64_le b 0, Bytes.get_int64_le b 8)

(* The least and greatest element of [r], and its step's size. *)
let span (r : A.Move.range) =
  let last = r.start + ((r.count - 1) * r.step) in
  (Int.min r.start last, Int.max r.start last, abs r.step)

(* Whether two ranges may share an element: their spans overlap and, where their
   steps are equal, they are of one residue. *)
let may_meet r r' =
  let lo, hi, s = span r and lo', hi', s' = span r' in
  lo <= hi' && lo' <= hi && (s <> s' || (r.start - r'.start) mod s = 0)

let regions_meet a b =
  let ok = ref true in
  Iarray.iteri
    (fun i r -> if not (may_meet r (Iarray.get b i)) then ok := false)
    a;
  !ok

let whole ys (rs : A.Move.range iarray) =
  let ok = ref true in
  Iarray.iteri
    (fun i (r : A.Move.range) ->
      if r.count <> ys.(i) || abs r.step <> 1 then ok := false)
    rs;
  !ok

let assemble ~fill ~dst pieces =
  let b = Plan.make () in
  let ys = shape_of dst in
  let r = Array.length ys and n = numel ys in
  let pieces =
    Array.of_list
      (List.filter
         (fun (_, x) -> numel (shape_of x) > 0)
         (Iarray.to_list pieces))
  in
  let np = Array.length pieces in
  let rec_words = K.assemble_head + (r * K.assemble_axis) in
  let per = Int.min K.assemble_pieces (K.assemble_table / rec_words) in
  let cs = c_strides ys in
  let unit, count = unit_count dst in
  let disjoint = ref true and total = ref 0 in
  for i = 0 to np - 1 do
    total := !total + numel (shape_of (snd pieces.(i)));
    for j = i + 1 to np - 1 do
      if regions_meet (fst pieces.(i)) (fst pieces.(j)) then disjoint := false
    done
  done;
  let covered =
    Array.exists (fun (rs, _) -> whole ys rs) pieces || (!disjoint && !total = n)
  in
  let d = slot b dst in
  let es = unit / 8 * count in
  let last = r - 1 in
  let step_last (rs, _) = (Iarray.get rs last : A.Move.range).step in
  (* Whether the launch of [npieces] pieces from [first] copies by runs: each
     piece's rows runs, laid out by one element along the last axis in the
     piece and dst, meeting no later piece. *)
  let copies_runs first npieces =
    r > 0
    && List.for_all
         (fun i ->
           let ((rs, x) as p) = pieces.(first + i) in
           step_last p = 1
           && (strides_of x).(last) = 1
           && is_run ~unit ~count (shape_of x).(last)
           && List.for_all
                (fun j -> not (regions_meet rs (fst pieces.(first + j))))
                (List.init (npieces - i - 1) (fun j -> i + 1 + j)))
         (List.init npieces Fun.id)
  in
  (* A piece's chunks, by runs. *)
  let piece_chunks x =
    let xs = shape_of x in
    numel xs / xs.(last) * run_chunks (xs.(last) * es)
  in
  (* The words and refs of [npieces] pieces from [first], their prefixes in
     chunks where [runs]. *)
  let table ?(runs = false) mode first npieces =
    let w = Plan.words K.Assemble in
    set w K.assemble_dst_off (offset_of dst);
    Array.iteri
      (fun a e -> Plan.axis w (K.assemble_dn + (K.assemble_dn_axis * a)) e)
      ys;
    if r > 0 then Plan.axis w K.assemble_chunks (run_chunks (ys.(last) * es));
    let f0, f1 = fill_bits (A.dtype dst) fill in
    (* A complex64 fill is two units of 32 bits, a word each. *)
    let f0, f1 =
      if unit = 32 && count = 2 then
        (Int64.logand f0 0xFFFFFFFFL, Int64.shift_right_logical f0 32)
      else (f0, f1)
    in
    Bytes.set_int64_le w (8 * K.assemble_fill_lo) f0;
    Bytes.set_int64_le w (8 * K.assemble_fill_hi) f1;
    set w K.assemble_rank r;
    set w K.assemble_n npieces;
    set w K.assemble_small (small (Int.max n !total));
    set w K.assemble_unit unit;
    set w K.assemble_count count;
    set w K.assemble_mode mode;
    set w K.assemble_runs (Bool.to_int runs);
    let refs = ref [ write K.assemble_dst d ] in
    let prefix = ref 0 in
    for i = 0 to npieces - 1 do
      let rs, x = pieces.(first + i) in
      (* The records follow dst's extents. *)
      let at =
        K.assemble_dn + (K.assemble_dn_axis * r) + (i * rec_words)
      in
      refs := read (at + K.assemble_rec_src) (slot b x) :: !refs;
      let base = ref 0 in
      Iarray.iteri
        (fun a (g : A.Move.range) -> base := !base + (g.start * cs.(a)))
        rs;
      let later = ref 0 in
      for j = i + 1 to npieces - 1 do
        if regions_meet rs (fst pieces.(first + j)) then
          later := !later lor (1 lsl j)
      done;
      set w (at + K.assemble_rec_off) (offset_of x);
      set w (at + K.assemble_rec_base) !base;
      set w (at + K.assemble_rec_prefix) !prefix;
      set w (at + K.assemble_rec_later) !later;
      let xs = shape_of x and xst = strides_of x in
      if r > 0 then
        Plan.axis w (at + K.assemble_rec_chunks)
          (run_chunks (xs.(last) * es));
      Iarray.iteri
        (fun a (g : A.Move.range) ->
          let ax = at + K.assemble_head + (a * K.assemble_axis) in
          Plan.axis w ax xs.(a);
          set w (ax + K.assemble_axis_sx) xst.(a);
          set w (ax + K.assemble_axis_start) g.start;
          set w (ax + K.assemble_axis_step) g.step)
        rs;
      prefix := !prefix + if runs then piece_chunks x else numel xs
    done;
    set w K.assemble_total !prefix;
    (w, List.rev !refs, !prefix)
  in
  if n > 0 && not covered then begin
    let check = np <= per in
    (* By runs where dst's rows are runs and each piece the fill skips steps by
       1 or -1 along the last axis. *)
    let runs =
      r > 0
      && is_run ~unit ~count ys.(last)
      && ((not check) || Array.for_all (fun p -> abs (step_last p) = 1) pieces)
    in
    let w, refs, _ = table ~runs K.assemble_fill 0 (Int.min np per) in
    set w K.assemble_check (Bool.to_int check);
    let work =
      if runs then n / ys.(last) * run_chunks (ys.(last) * es) * wave
      else work_of ~unit n
    in
    launch b K.Assemble ~groups:K.assemble_groups ~work w refs
  end;
  let first = ref 0 in
  while !first < np do
    let c = Int.min per (np - !first) in
    let runs = copies_runs !first c in
    let w, refs, total = table ~runs K.assemble_copy !first c in
    let work = if runs then total * wave else total in
    launch b K.Assemble ~groups:K.assemble_groups ~work w refs;
    first := !first + c
  done;
  Ok (Plan.finish_some b)

(* Folds *)

let fold (pad : Nx_kernel.pad) ~dst x =
  let b = Plan.make () in
  let ys = shape_of dst and xs = strides_of x in
  let r = Array.length ys and n = numel ys in
  let unit, count = unit_count dst in
  let w = Plan.words K.Fold in
  set w K.fold_numel n;
  set w K.fold_dst_off (offset_of dst);
  set w K.fold_x_off (offset_of x);
  set w K.fold_rank r;
  set w K.fold_dtype (code dst);
  set w K.fold_small (small n);
  set w K.fold_unit unit;
  set w K.fold_count count;
  let xshape = shape_of x in
  (* Whether 31 bits hold every index, padded position and tap's reach. *)
  let narrow = ref (small n = 1) in
  let fits v = if abs v >= 1 lsl 30 then narrow := false in
  let divisor i d = Bytes.set_int64_le w (8 * i) (Plan.divisor d) in
  for a = 0 to r - 1 do
    let at = K.fold_ax + (a * K.fold_axis_words) in
    let lo = Iarray.get pad.lo a and hi = Iarray.get pad.hi a in
    let inter = Iarray.get pad.interior a in
    let pext =
      lo + hi + ys.(a) + if ys.(a) > 0 then (ys.(a) - 1) * inter else 0
    in
    List.iter fits [ ys.(a); lo; hi; inter; pext; lo + (ys.(a) * (inter + 1)) ];
    Plan.axis w at ys.(a);
    set w (at + K.fold_axis_lo) lo;
    set w (at + K.fold_axis_inter) inter;
    set w (at + K.fold_axis_pext) pext;
    set w (at + K.fold_axis_sx) xs.(a);
    set w (at + K.fold_axis_win) (-1)
  done;
  let windows = Iarray.length pad.windows in
  Iarray.iteri
    (fun v (win : A.Move.window) ->
      let at = K.fold_ax + (win.axis * K.fold_axis_words) in
      set w (at + K.fold_axis_taps) win.size;
      set w (at + K.fold_axis_step) win.step;
      set w (at + K.fold_axis_dil) win.dilation;
      set w (at + K.fold_axis_nwin) xshape.(win.axis);
      set w (at + K.fold_axis_sw) xs.(win.axis);
      set w (at + K.fold_axis_st) xs.(r + v);
      set w (at + K.fold_axis_win) v;
      List.iter fits
        [
          win.size;
          win.step;
          win.dilation;
          win.size * win.dilation;
          xshape.(win.axis);
        ];
      if win.step >= 1 && win.step < 1 lsl 30 then
        divisor (at + K.fold_axis_ms) win.step
      else narrow := false)
    pad.windows;
  set w K.fold_windows windows;
  set w K.fold_narrow (Bool.to_int !narrow);
  (* A narrow launch of K.fold_few windows or fewer keeps them in
     registers. *)
  let lds = if !narrow && windows <= K.fold_few then 0 else 16 * windows in
  if n > 0 then
    launch b K.Fold ~groups:K.fold_groups ~work:n
      ~shared:(lds * K.fold_threads)
      w
      [ write K.fold_dst (slot b dst); read K.fold_x (slot b x) ];
  Ok (Plan.finish_some b)

(* Scatters *)

(* A scatter's geometry: the updates' shape and strides, idx's strides, dst's
   shape, each array's first element, and the dtype code, unit and count of an
   element. *)
type geometry = {
  us : int array;
  ust : int array;
  ist : int array;
  ys : int array;
  dst_off : int;
  idx_off : int;
  upd_off : int;
  dtype : int;
  unit : int;
  count : int;
}

let geometry ~dst idx u =
  let unit, count = unit_count dst in
  {
    us = shape_of u;
    ust = strides_of u;
    ist = strides_of idx;
    ys = shape_of dst;
    dst_off = offset_of dst;
    idx_off = offset_of idx;
    upd_off = offset_of u;
    dtype = code dst;
    unit;
    count;
  }

(* [g], of complex numbers, as floats: a last axis of their two parts, along
   which idx steps 0. *)
let as_parts g =
  let last a x = Array.append a [| x |] in
  {
    us = last g.us 2;
    ust = last (Array.map (fun s -> 2 * s) g.ust) 1;
    ist = last g.ist 0;
    ys = last g.ys 2;
    dst_off = 2 * g.dst_off;
    idx_off = g.idx_off;
    upd_off = 2 * g.upd_off;
    dtype = (if g.unit = 64 then D.code D.Float64 else D.code D.Float32);
    unit = g.unit;
    count = 1;
  }

(* A scatter kernel's words, every one [g] gives, and the refs of its arrays. *)
let scatter_words b ~axis ~dst idx u g =
  let w = Plan.words K.Scatter in
  let cs = c_strides g.ys in
  let nu = numel g.us and nd = numel g.ys in
  set w K.scatter_numel_u nu;
  set w K.scatter_numel_d nd;
  set w K.scatter_dst_off g.dst_off;
  set w K.scatter_idx_off g.idx_off;
  set w K.scatter_upd_off g.upd_off;
  set w K.scatter_d g.ys.(axis);
  Plan.axis w K.scatter_m g.us.(axis);
  set w K.scatter_sda cs.(axis);
  set w K.scatter_rank (Array.length g.us);
  set w K.scatter_axis axis;
  set w K.scatter_dtype g.dtype;
  set w K.scatter_small (small (Int.max nu nd));
  set w K.scatter_unit g.unit;
  set w K.scatter_count g.count;
  Array.iteri
    (fun a e ->
      let at = K.scatter_ax + (a * K.scatter_axis_words) in
      Plan.axis w at e;
      set w (at + K.scatter_axis_sd) cs.(a);
      set w (at + K.scatter_axis_si) g.ist.(a);
      set w (at + K.scatter_axis_su) g.ust.(a))
    g.us;
  ( w,
    [
      write K.scatter_dst (slot b dst);
      read K.scatter_idx (slot b idx);
      read K.scatter_upd (slot b u);
    ] )

(* Bounds for tests: the most parts a sort pass cuts its items into, the most
   groups of its launches, the longest run summed whole, and whether items are
   wide whatever their bits. *)
type bounds = { parts : int; groups : int; whole : int; wide : bool }

let bounds =
  Atomic.make
    {
      parts = K.scatter_parts;
      groups = Plan.stride_groups;
      whole = K.scatter_short;
      wide = false;
    }

let bound_scatters ~parts ~groups ~whole ~wide =
  if parts < 1 || parts > K.scatter_parts then
    invalid_arg "Nx_amd.bound_scatters: parts";
  if groups < 1 || groups > Plan.stride_groups then
    invalid_arg "Nx_amd.bound_scatters: groups";
  if whole < 0 || whole > K.scatter_short then
    invalid_arg "Nx_amd.bound_scatters: whole";
  Atomic.set bounds { parts; groups; whole; wide }

(* A float Add by the position tree (moves.hip): an axis of one tile in one
   launch; else a sort of the entries by target, the updates or, where a slice
   has at most a tile's targets, the tiles' entries, then the runs' sums, a
   launch each level of the tree. Tiles are row tiles where idx steps 0 along
   the last axis off the scattered one, and an update's entry is then its
   row. *)
let tree_add b ~axis ~dst idx u g =
  let us = g.us and r = Array.length g.us in
  (* A value of the sum's format: float64, or float32 for the narrow floats
     too. *)
  let vb = if g.unit = 64 then 8 else 4 in
  let f64 = vb = 8 in
  let m = us.(axis) and nu = numel us and nd = numel g.ys in
  let d = g.ys.(axis) in
  let rows = axis < r - 1 && g.ist.(r - 1) = 0 && us.(r - 1) > 1 in
  let tile = if rows then K.scatter_rows else K.scatter_tile in
  let last = if rows then us.(r - 1) else 1 in
  let tcols =
    if rows then
      ceil_log2
        (Int.min last
           (Int.min 32 (K.scatter_row_bytes / (K.scatter_rows * vb))))
    else 0
  in
  let nc = (last + (1 lsl tcols) - 1) lsr tcols in
  let run = Int.min tile (1 lsl ceil_log2 m) in
  let group = tile / run in
  let nb = (m + run - 1) / run in
  let slices = nu / m / last in
  let nsg = (slices + group - 1) / group in
  let tiles = nsg * nb * nc in
  let w, refs = scatter_words b ~axis ~dst idx u g in
  Plan.axis w K.scatter_m m;
  set w K.scatter_slices slices;
  set w K.scatter_run run;
  set w K.scatter_group group;
  set w K.scatter_nb nb;
  set w K.scatter_nsg nsg;
  set w K.scatter_tiles tiles;
  Plan.axis w K.scatter_last last;
  Plan.axis w K.scatter_nc nc;
  set w K.scatter_lcols tcols;
  let tiler =
    match (rows, f64) with
    | true, false -> K.Scatter_rows_f32
    | true, true -> K.Scatter_rows_f64
    | false, false -> K.Scatter_tile_f32
    | false, true -> K.Scatter_tile_f64
  in
  let step k ~work w refs =
    launch b k ~groups:K.scatter_groups ~work (Bytes.copy w) refs
  in
  if nb = 1 then step tiler ~work:(tiles * K.move_threads) w refs
  else begin
    (* The entries. Past one tile, a tile holds [tile] positions of one slice,
       so at most [d] entries: where [d <= tile], the tiles' entries, in slots
       of [d] a tile, else the updates, an entry a row of [last] columns with
       row tiles. A key is an entry's target, its row's, or its row tile's.
       Each entry holds an update of the slices' [m] positions, so at most
       [most] are sorted. *)
    let from_tiles = d <= tile in
    let slots = tiles * d in
    let entries = if from_tiles then slots else nu / last in
    let most = if from_tiles then Int.min slots (slices * m * nc) else entries in
    let keys = if from_tiles && rows then nd / last * nc else nd / last in
    let lcols = if from_tiles then tcols else ceil_log2 last in
    (* An item: narrow where its key's and index's bits fit 64. *)
    let kb = ceil_log2 keys and pb = ceil_log2 entries in
    let bound = Atomic.get bounds in
    let narrow = (not bound.wide) && kb + pb <= 64 in
    let pack_k, count_k, move_k, per, ib =
      if narrow then
        ( K.Scatter_pack_n,
          K.Scatter_count_n,
          K.Scatter_move_n,
          K.scatter_items_n,
          8 )
      else
        ( K.Scatter_pack_w,
          K.Scatter_count_w,
          K.Scatter_move_w,
          K.scatter_items_w,
          16 )
    in
    let passes = Int.max 1 ((kb + 7) / 8) in
    let tile_items = K.move_threads * per in
    let parts =
      Int.min bound.parts ((entries + tile_items - 1) / tile_items)
    in
    let pbits = ceil_log2 (if from_tiles then nb else m) in
    let lb = K.scatter_level_bits in
    let levels = Int.max 1 ((pbits + lb - 1) / lb) in
    (* The workspace's regions, each at a multiple of 8 bytes. Once sorted,
       the items the last pass read hold the ends of the runs' blocks, one
       array of them where items are narrow, two where wide. *)
    let regions =
      (if from_tiles then
         [
           (K.scatter_ent_g, 8 * slots);
           (K.scatter_ent_v, up8 (vb * (slots lsl tcols)));
           (K.scatter_ent_n, up8 (4 * tiles));
         ]
       else [])
      @ [
          (K.scatter_items0, ib * most);
          (K.scatter_items1, ib * most);
          (K.scatter_counts, 8 * 256 * parts);
          (K.scatter_ctl, 8);
          (K.scatter_part, up8 (vb * (most lsl lcols)));
          (K.scatter_mark, up8 most);
        ]
      @ if narrow then [ (K.scatter_plen0, 8 * most) ] else []
    in
    let size = List.fold_left (fun o (_, bytes) -> o + bytes) 0 regions in
    let at = Plan.take b size and ws = Plan.workspace b in
    let place = Hashtbl.create 16 in
    let refs =
      snd
        (List.fold_left
           (fun (o, refs) (word, bytes) ->
             set w word (at + o);
             Hashtbl.replace place word (at + o);
             (o + bytes, write word ws :: refs))
           (0, refs) regions)
    in
    let refs =
      write K.scatter_plen1 ws
      :: (if narrow then refs else write K.scatter_plen0 ws :: refs)
    in
    set w K.scatter_small (small (Int.max (Int.max nu nd) entries));
    set w K.scatter_est d;
    set w K.scatter_from
      (if from_tiles then K.scatter_from_tiles else K.scatter_from_updates);
    set w K.scatter_entries entries;
    set w K.scatter_pb pb;
    Plan.axis w K.scatter_span (d * nsg * nc);
    set w K.scatter_pbits pbits;
    set w K.scatter_whole
      (if pbits <= K.scatter_levels then bound.whole else 0);
    if from_tiles then step tiler ~work:(tiles * K.move_threads) w refs;
    set w K.scatter_lcols lcols;
    (* A launch of [k] in [parts] parts, a group a part; with [~ticket],
       the part last done takes the ticket. *)
    let parted ?(ticket = false) ?(pass = 0) k ~parts =
      let w = Bytes.copy w in
      set w K.scatter_nparts parts;
      set w K.scatter_pass pass;
      let g = Int.min bound.groups (Plan.grid ~per:1 k parts) in
      set w K.scatter_groups g;
      let refs =
        if not ticket then refs
        else begin
          set w K.scatter_ticket (Plan.tickets b 1);
          write K.scatter_ticket (Plan.tickets_slot b) :: refs
        end
      in
      Plan.launch b k ~groups:(g, 1, 1) ~used:(used k w) w (Array.of_list refs)
    in
    (* The tiles' entries, packed as items in parts of at least MT tiles. *)
    if from_tiles then begin
      let packs =
        Int.min bound.parts ((tiles + K.move_threads - 1) / K.move_threads)
      in
      parted ~ticket:true K.Scatter_tally ~parts:packs;
      parted pack_k ~parts:packs
    end;
    for q = 0 to passes - 1 do
      parted ~ticket:true ~pass:q count_k ~parts;
      parted ~pass:q move_k ~parts
    done;
    (* Pass q wrote items[q mod 2]: the sums read the last pass's, ITEMS0, and
       keep their ends in the other. *)
    let sorted, free =
      if (passes - 1) land 1 = 0 then (K.scatter_items0, K.scatter_items1)
      else (K.scatter_items1, K.scatter_items0)
    in
    let free = Hashtbl.find place free in
    set w K.scatter_items0 (Hashtbl.find place sorted);
    set w K.scatter_plen1 free;
    if not narrow then set w K.scatter_plen0 (free + (8 * most));
    let sum =
      match (f64, narrow) with
      | false, true -> K.Scatter_sum_f32_n
      | false, false -> K.Scatter_sum_f32_w
      | true, true -> K.Scatter_sum_f64_n
      | true, false -> K.Scatter_sum_f64_w
    in
    for l = 1 to levels do
      set w K.scatter_level l;
      step sum ~work:(most lsl lcols) w refs
    done
  end

let scatter (c : Nx_kernel.combine) ~unique ~axis ~dst ~into idx u =
  let b = Plan.make () in
  let identical =
    A.buffer into == A.buffer dst && L.equal (A.layout into) (A.layout dst)
  in
  if not identical then copy b ~dst into;
  let g = geometry ~dst idx u in
  let nu = numel g.us and nd = numel g.ys in
  let float = D.is D.Float (A.dtype dst) || D.is D.Complex (A.dtype dst) in
  let groups = K.scatter_groups in
  (* A pass of [op] over the updates, or with [~targets] over dst's elements,
     [ws] the workspace's piece it keys by. *)
  let pass ?ws ?(wide = false) ?(greater = false) ?(targets = false) op =
    let w, refs = scatter_words b ~axis ~dst idx u g in
    set w K.scatter_op op;
    set w K.scatter_wide (Bool.to_int wide);
    set w K.scatter_greater (Bool.to_int greater);
    let refs =
      match ws with
      | None -> refs
      | Some at ->
          set w K.scatter_owner at;
          write K.scatter_owner (Plan.workspace b) :: refs
    in
    launch b K.Scatter ~groups ~work:(if targets then nd else nu) w refs
  in
  (* A piece of the workspace of [words] zero 32-bit words, for each target's
     owner or keys. *)
  let zeros words =
    let at = Plan.take b (4 * words) in
    Plan.zero b ~slot:(Plan.workspace b) ~at (4 * words);
    at
  in
  if nu > 0 && nd > 0 then begin
    match c with
    | Set when unique -> pass K.scatter_set
    | Set when g.count = 1 && List.mem g.unit [ 8; 16; 32 ]
               && g.us.(axis) <= 0xFFFFFFFF ->
        (* A 64-bit word a target, its last update's position plus one above
           its bits: one random pass finds both, then a pass over the
           targets stores them. *)
        let ws = zeros (2 * nd) in
        pass ~ws K.scatter_own_packed;
        pass ~ws ~targets:true K.scatter_packed_store
    | Set ->
        (* An owner is a position plus one, 0 for none: 32 bits hold it up to
           2^32 - 1 positions. *)
        let wide = g.us.(axis) > 0xFFFFFFFF in
        let ws = zeros (if wide then 2 * nd else nd) in
        pass ~ws ~wide K.scatter_own;
        pass ~ws ~wide K.scatter_owned
    | (Max | Min) when g.count = 2 ->
        (* complex128: two 64-bit keys a target, the real parts' then the
           imaginary parts'; complex64: one, of both parts. *)
        let pair = g.unit = 32 and greater = c = Max in
        let ws = zeros (if pair then 2 * nd else 4 * nd) in
        pass ~ws ~greater K.scatter_top_re;
        if not pair then pass ~ws ~greater K.scatter_top_im;
        pass ~ws ~greater ~targets:true K.scatter_top_store
    | Max -> pass K.scatter_max
    | Min -> pass K.scatter_min
    | Add when not float -> pass K.scatter_add_int
    | Add when unique -> pass K.scatter_add_unique
    | Add -> tree_add b ~axis ~dst idx u (if g.count = 2 then as_parts g else g)
  end;
  Ok (Plan.finish_some b)

