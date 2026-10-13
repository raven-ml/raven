(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A plan declines a call, and nx's expansion computes it, when an operand, init
   or y is complex or narrower than a byte; a float operand or init is wider
   than a float accumulator, which would round it before the sum; init is of the
   other kind than the accumulator; an integer meets a float accumulator or the
   reverse; the batch is above 65,535 (the grid's z); m, n or k is above 2^31 -
   1; a pack's elements outnumber the work-items a grid holds; or the grid's
   workgroups along x pass 2^31 - 1. A call whose axes do not group answers
   what Contract_view.v answers. *)

module K = Kernels
module V = Nx_kernel.Contract_view
module D = Nx_array.Dtype
module Run = Rig.Submission.Run
module Sub = Rig.Submission

(* Dtypes *)

let f64 = D.Any D.Float64
let f32 = D.Any D.Float32
let f16 = D.Any D.Float16
let bf16 = D.Any D.Bfloat16
let i64 = D.Any D.Int64
let i32 = D.Any D.Int32
let i8 = D.Any D.Int8
let same (D.Any a) (D.Any b) = D.equal a b
let bytes (D.Any d) = D.bits d / 8
let code (D.Any d) = D.code d
let dtype_of x = D.Any (Nx_array.dtype x)

let is_f8 : D.any -> bool = function
  | Any Float8_e4m3fn -> true
  | Any Float8_e5m2 -> true
  | Any _ -> false

let is_int : D.any -> bool = function
  | Any Int64
  | Any Uint64
  | Any Int32
  | Any Uint32
  | Any Int16
  | Any Uint16
  | Any Int8
  | Any Uint8
  | Any Bool ->
      true
  | Any _ -> false

let unsigned_64 (D.Any d) = D.equal d D.Uint64
let unsigned_32 (D.Any d) = D.equal d D.Uint32

(* Whether [d] reads as a value the kernels sum: no complex, no sub-byte
   element. *)
let summable (D.Any d) = (not (D.is D.Complex d)) && D.bits d >= 8

(* Instances *)

let rec index_from k i =
  if snd K.kernels.(k) = i then k else index_from (k + 1) i

let count = Array.length K.kernels
let pack_kernel = index_from 0 K.Pack
let zero_kernel = index_from 0 K.Zero

(* The index of an instance, or -1. Loops with their state as arguments: a local
   loop closing over them would allocate its closure on every call. *)
let rec find_wmma_from k kind a b t =
  if k = count then -1
  else
    match snd K.kernels.(k) with
    | Wmma (kind', a', b', t')
      when kind' == kind && a' == a && b' == b && t' == t ->
        k
    | _ -> find_wmma_from (k + 1) kind a b t

let find_wmma kind a b t = find_wmma_from 0 kind a b t

let rec find_simt_from k sum rows columns =
  if k = count then -1
  else
    match snd K.kernels.(k) with
    | Simt (sum', rows', columns')
      when sum' == sum && rows' = rows && columns' = columns ->
        k
    | _ -> find_simt_from (k + 1) sum rows columns

let find_simt sum rows columns = find_simt_from 0 sum rows columns

let rec find_skinny_from k sum form =
  if k = count then -1
  else
    match snd K.kernels.(k) with
    | Skinny (sum', form') when sum' == sum && form' == form -> k
    | _ -> find_skinny_from (k + 1) sum form

let find_skinny sum form = find_skinny_from 0 sum form

let rec find_lane_from k sum reads form =
  if k = count then -1
  else
    match snd K.kernels.(k) with
    | Lane (sum', reads', form')
      when sum' == sum && reads' = reads && form' == form ->
        k
    | _ -> find_lane_from (k + 1) sum reads form

let find_lane sum reads form = find_lane_from 0 sum reads form
let tile_threads (s : K.shape) = s.bm / s.wm * (s.bn / s.wn) * 32

(* Plans *)

let max_int32 = 0x7fff_ffff
let max_uint32 = 0xffff_ffff
let ceil_div a b = (a + b - 1) / b

(* An operand a or b as it lies, which a pack copies into the workspace: its
   dtype, the dtype its pack writes, its first element (bytes into its slot)
   and strides; its rows, and the elements a packed row. *)
type pack = {
  own : D.any;
  into : D.any;
  src : int;
  ps : int array;
  rows : int;
  lead : int;
}

(* An operand a or b as the contraction kernel reads it. [slot] is the call's
   operand it is, 0 or 1: a lane plan makes the operand of the short extent a.
   Its buffer; its dtype, as the kernel's parameters name it; its first
   element, bytes into its slot or into the workspace once packed; whether
   that element lies on a 16-byte boundary as the device addresses it, the
   workspace being 256-byte aligned; its strides along batch, its row or
   column, and k; and its pack, if packed. *)
type operand = {
  slot : int;
  buffer : Rig.Buffer.t;
  dtype : D.any;
  first : int;
  vector_first : bool;
  s : int array;
  pack : pack option;
}

(* init as the kernel reads it: its buffer, dtype, first element (bytes into
   its slot) and strides along batch, m and n. *)
type init = {
  i_buffer : Rig.Buffer.t;
  i_dtype : D.any;
  i_first : int;
  si : int array;
}

(* A call as the rules read it, a and b as they lie. *)
type call = {
  a : operand;
  b : operand;
  y : Rig.Buffer.t;
  y_dtype : D.any;
  y_first : int;
  y_vector_first : bool;
  sy : int array;
  init : init option;
  acc : D.any;
  order : Nx_kernel.order;
  batch : int;
  m : int;
  n : int;
  k : int;
}

(* A call's launches: the call, its a and b traded where a lane plan swaps
   them; a and b as the kernel reads them; the contraction kernel's [blocks] x
   [splits] x batch workgroups of [threads] work-items; the workspace bytes
   taken; and the split sum's
   pieces: its partials and tickets in the workspace, and its output tiles, a
   ticket each. *)
type launch = {
  call : call;
  a : operand;
  b : operand;
  kernel : int;
  splits : int;
  blocks : int;
  threads : int;
  aligned : int;
  used : int;
  partials : int;
  tickets : int;
  tiles : int;
}

type verdict = Nothing | Launch of launch

let declined = Error Nx_array.Declined

(* Whether rows along [contiguous] (stride 1) load as 16-byte vectors: every
   other stride a whole number of vectors, and the first element on a vector. *)
let vectors vector_first contiguous lead batch bytes =
  let per = 16 / bytes in
  contiguous = 1 && lead mod per = 0 && batch mod per = 0 && vector_first

(* Whether an operand of the strides [s] (batch, row, k) has its rows contiguous
   rather than its k. *)
let free_contiguous s = s.(1) = 1 && s.(2) <> 1

(* Whether [o]'s rows load as 16-byte vectors along its contiguous axis. *)
let rows_vectors o =
  let s = o.s in
  if free_contiguous s then
    vectors o.vector_first s.(1) s.(2) s.(0) (bytes o.dtype)
  else vectors o.vector_first s.(2) s.(1) s.(0) (bytes o.dtype)

(* Each workspace piece starts on a 256-byte boundary. *)
let align used = (used + 255) land lnot 255

(* [o], of [rows] rows, packed into the workspace after its [used] bytes as
   elements of [into] with k contiguous and rows of whole vectors, a work-item
   an element: the kernel reads the copy. [None] if its elements outnumber the
   work-items a grid holds. With no k, the kernel reads no element, and [o]
   is not packed. *)
let pack (c : call) o ~rows ~into ~used =
  let es = bytes into in
  let per = 16 / es in
  let lead = ceil_div c.k per * per in
  if c.k = 0 then Some o
  else if c.batch * rows * lead > max_uint32 * K.threads then None
  else
    let at = align used in
    Some
      {
        o with
        first = at;
        vector_first = at mod 16 = 0;
        s = [| rows * lead; lead; 1 |];
        pack =
          Some { own = o.dtype; into; src = o.first; ps = o.s; rows; lead };
      }

(* The workspace bytes taken up to the end of [o]'s copy, [used] if it has
   none. *)
let used_after (c : call) used o =
  match o.pack with
  | None -> used
  | Some p -> o.first + (c.batch * o.s.(0) * bytes p.into)

(* a and b of [c], each packed into [into] where [pack_a] and [pack_b] say,
   a's copy first, and the workspace bytes the copies take; [None] if a pack
   fails. *)
let packed_as (c : call) ~into ~pack_a ~pack_b =
  let a = if pack_a then pack c c.a ~rows:c.m ~into ~used:0 else Some c.a in
  match a with
  | None -> None
  | Some a -> (
      let used = used_after c 0 a in
      let b = if pack_b then pack c c.b ~rows:c.n ~into ~used else Some c.b in
      match b with None -> None | Some b -> Some (a, b, used_after c used b))

(* The split count of a grid of [grid] workgroups: doubled while the grid has
   fewer than [target] workgroups and each range keeps at least [k_min] of k, at
   most 16. The targets are constants of the processor, never a GPU's own count.
   The 16 x 64 tile's target of 128 is measured on the R9700; the other rules'
   targets and k_min (WMMA m > 16: 64 and 1024, SIMT: 64 and 128, skinny: 256
   and 1024) carry over from Ada's, as the R9700's runs at those shapes did not
   favour other values. *)
let split_count grid target k k_min =
  let s = ref 1 in
  while !s < 16 && grid * !s < target && k / (2 * !s) >= k_min do
    s := 2 * !s
  done;
  !s

(* Outputs stored 16 bytes at once: no init, y's j contiguous, its rows on
   16-byte boundaries, and its dtype one the kernels store as they are, or
   bfloat16 or float16 from float32 (store8). *)
let whole (c : call) ~wmma_s8 (sum : K.acc) =
  let yd = c.y_dtype in
  let yb = bytes yd in
  let natural =
    if wmma_s8 then same yd i32 || unsigned_32 yd
    else
      match sum with
      | F32 -> same yd f32 || same yd bf16 || same yd f16
      | F64 -> same yd f64
      | I64 ->
          (same yd i64 || unsigned_64 yd)
          && (same c.acc i64 || unsigned_64 c.acc)
  in
  Option.is_none c.init && natural
  && c.sy.(2) = 1
  && c.sy.(1) * yb mod 16 = 0
  && c.sy.(0) * yb mod 16 = 0
  && c.y_vector_first

(* [c]'s launch of [kernel] over a and b as [packed_as] gives them, [None]
   past the grid's workgroups along x. With [splits] above 1, its partials,
   [values] of [sum_bytes] a work-item, then its tickets follow the copies in
   the workspace: combine.h's last workgroup leaves its tile's ticket counted,
   so a launch zeroes the tickets before each contraction, and no buffer
   stays zero across calls. [whole]'s stores add y_whole to [aligned]. *)
let launch (c : call) (a, b, used) ~kernel ~blocks ~splits ~threads ~values
    ~sum_bytes ~aligned ~wmma_s8 sum =
  if blocks > max_int32 then None
  else
    let split = splits > 1 in
    let tiles = if split then c.batch * blocks else 0 in
    let partials = if split then align used else 0 in
    let sums = tiles * splits * values * threads * sum_bytes in
    let tickets = if split then align (partials + sums) else 0 in
    let whole = whole c ~wmma_s8 sum in
    Some
      {
        call = c;
        a;
        b;
        kernel;
        splits;
        blocks;
        threads;
        aligned = (if whole then aligned lor K.y_whole else aligned);
        used = (if split then tickets + (tiles * 4) else used);
        partials;
        tickets;
        tiles;
      }

(* Workgroups run in waves of [wave] on gfx1201: the 128 x 128 tile's time steps
   every 96 workgroups, three to each of the R9700's 32 work-group processors. A
   GPU of another size runs the same tiles, with the same bits. *)
let wave = 96

(* The cost of [bm] x [bn] tiles of [efficiency], waves times outputs per tile
   over its efficiency. Inlined, so that its float is never boxed. *)
let[@inline] cost (c : call) bm bn efficiency =
  let groups = c.batch * ceil_div c.m bm * ceil_div c.n bn in
  Float.of_int (ceil_div groups wave)
  *. Float.of_int bm *. Float.of_int bn /. Float.of_int efficiency

(* The WMMA tiles' outputs per unit of time against the 128 x 128 tile's, in
   percent, on the R9700: the 64 x 64 tile makes 76-83% at 2048 and 4096 cubed,
   bfloat16 and float16. The 16 x 64 tile's 33 is Ada's: m <= 16 takes that tile
   alone, so its rate is never weighed. *)
let efficiency : K.tile -> int = function
  | T128x128 -> 100
  | T64x64 -> 80
  | T16x64 -> 33

(* The index in [K.tiles] of the WMMA tile of a product among those [kind] has
   an instance of with k contiguous: the one of least cost, by its shape alone;
   m <= 16 takes the 16-row tile, and only it. -1 if [kind] has none for the
   shape, int8 at 16 rows or fewer: the skinny kernels sum it. *)
let wmma_tile (c : call) kind =
  let best = ref (-1) and least = ref 0. in
  for i = 0 to Array.length K.tiles - 1 do
    let t = K.tiles.(i) in
    if find_wmma kind K.K K.K t >= 0 && c.m <= 16 = (t == K.T16x64) then begin
      let s = K.shape t in
      let cost = cost c s.bm s.bn (efficiency t) in
      if !best < 0 || cost < !least then begin
        least := cost;
        best := i
      end
    end
  done;
  !best

(* Whether the WMMA kernels read [o], of [rows] rows, as it lies: vectors of
   [per] elements along its contiguous axis, its rows whole vectors where they
   are contiguous, k whole vectors ([whole]), and no float8. *)
let readable o rows per whole =
  rows_vectors o && (not (is_f8 o.dtype)) && whole
  && ((not (free_contiguous o.s)) || rows mod per = 0)

(* The WMMA kernels, of [kind] on the tile [t]. They read whole vectors along
   an operand's contiguous axis and never past its extent: an operand whose k,
   or whose rows where they are contiguous, end inside a vector, or in a layout
   [t] has no instance of, is packed into the workspace with k contiguous, its
   rows padded with zeros. The parameters keep the operands' own dtypes: a pack
   changes none. *)
let plan_wmma (c : call) kind t ~wmma_s8 sum =
  let s = K.shape t in
  let width = if kind == K.S8 then 1 else 2 in
  let per = 16 / width in
  let whole = c.k mod per = 0 in
  let la = if free_contiguous c.a.s then K.M else K.K in
  let lb = if free_contiguous c.b.s then K.N else K.K in
  let pack_a =
    (not (readable c.a c.m per whole))
    || (la != K.K && find_wmma kind la K.K t < 0)
  in
  let la = if pack_a then K.K else la in
  let pack_b =
    (not (readable c.b c.n per whole))
    || (lb != K.K && find_wmma kind la lb t < 0)
  in
  let lb = if pack_b then K.K else lb in
  let into = match kind with S8 -> i8 | F16 -> f16 | Bf16 -> bf16 in
  match packed_as c ~into ~pack_a ~pack_b with
  | None -> None
  | Some read ->
      let blocks = ceil_div c.m s.bm * ceil_div c.n s.bn in
      let threads = tile_threads s in
      launch c read ~kernel:(find_wmma kind la lb t) ~blocks
        (* A split sum stores and reloads its partials: worth it to fill a GPU
           short of workgroups, or to stream a long k for a few rows. *)
        ~splits:
          (if c.m <= 16 then
             split_count (blocks * c.batch) 128 c.k (4 * s.bkb / width)
           else split_count (blocks * c.batch) 64 c.k 1024)
        ~threads ~values:(s.bm * s.bn / threads) ~sum_bytes:4 ~aligned:0
        ~wmma_s8 sum

(* Whether [d] is the dtype the SIMT and skinny kernels of [sum] read. *)
let reads_own (sum : K.acc) d =
  match sum with
  | F32 -> same d f32
  | F64 -> same d f64
  | I64 -> same d i64 || unsigned_64 d

let own_dtype : K.acc -> D.any = function F32 -> f32 | F64 -> f64 | I64 -> i64

(* SIMT and skinny kernels read their accumulator's own dtype: an operand of
   another is packed into it, exactly, with k contiguous, and read as it. *)
let packed_own (c : call) sum =
  let into = own_dtype sum in
  match
    packed_as c ~into
      ~pack_a:(not (reads_own sum c.a.dtype))
      ~pack_b:(not (reads_own sum c.b.dtype))
  with
  | None -> None
  | Some (a, b, used) ->
      let read o =
        match o.pack with None -> o | Some _ -> { o with dtype = into }
      in
      Some (read a, read b, used)

(* The skinny kernel of an integer [sum], m <= 16, in its column form, or
   across when b's n axis is contiguous. *)
let plan_skinny (c : call) ((a, b, _) as read) ~wmma_s8 sum =
  let across = free_contiguous b.s in
  launch c read
    ~kernel:(find_skinny sum (if across then Across else Column))
    ~blocks:(if across then ceil_div c.n 32 else ceil_div c.n 8)
    ~splits:(split_count (ceil_div c.n 8 * c.batch) 256 c.k 1024)
    ~threads:K.threads ~values:2 ~sum_bytes:8
    ~aligned:
      ((if (not across) && rows_vectors b then K.b_vectors else 0)
      lor
      if (not (free_contiguous a.s)) && rows_vectors a then K.a_vectors else 0)
    ~wmma_s8 sum

(* Whether a float sum of [c] adds in lane order, as Nx_kernel's orders
   section puts it: where it is given that order. *)
let in_lane_order (c : call) (sum : K.acc) =
  sum != K.I64 && c.order = Nx_kernel.Lane

(* A lane kernel's workgroup, as contract.hip's lane_contract shapes it: the
   blocks of 1024 of a column it sums at once, G, k's between 1 and 8, and its
   columns, 32, or across 32 (8 / G). *)
let lane_blocks k = Int.max 1 (Int.min 8 (ceil_div k 1024))
let lane_columns k ~across = if across then 32 * (8 / lane_blocks k) else 32

(* [c] with a the operand of the short extent: a's rows and b's columns trade
   places, as y's and init's do. *)
let swap (c : call) =
  let flip s = [| s.(0); s.(2); s.(1) |] in
  {
    c with
    a = c.b;
    b = c.a;
    m = c.n;
    n = c.m;
    sy = flip c.sy;
    init = Option.map (fun i -> { i with si = flip i.si }) c.init;
  }

(* The lane kernel of [sum], a the operand of the short extent when its rows
   pass a workgroup's: bfloat16 operands read as they are, others packed into
   [sum]'s dtype; across when b's n axis is contiguous. *)
let plan_lane (c : call) ~wmma_s8 sum =
  let c = if c.m > K.lane_rows && c.n < c.m then swap c else c in
  let bf16s = sum == K.F32 && same c.a.dtype bf16 && same c.b.dtype bf16 in
  let read = if bf16s then Some (c.a, c.b, 0) else packed_own c sum in
  match read with
  | None -> None
  | Some ((a, b, _) as read) ->
      let across = (not bf16s) && free_contiguous b.s in
      let reads = if bf16s then Some K.Bf16 else None in
      let k_vectors o = (not (free_contiguous o.s)) && rows_vectors o in
      launch c read
        ~kernel:(find_lane sum reads (if across then Across else Column))
        ~blocks:
          (ceil_div c.m K.lane_rows * ceil_div c.n (lane_columns c.k ~across))
        ~splits:1 ~threads:(lane_blocks c.k * lane_columns c.k ~across)
        ~values:0
        ~sum_bytes:(if sum == K.F32 then 4 else 8)
        ~aligned:
          ((if k_vectors b then K.b_vectors else 0)
          lor if k_vectors a then K.a_vectors else 0)
        ~wmma_s8 sum

(* The SIMT tile of [sum]: where m is 16 or fewer, its short tile, 16 x 128
   for float32 and 8 x 128 for float64, whose float64 sums are compute bound;
   otherwise the square tile of least cost among its instances: on the R9700
   the 64-wide tile computes half as fast as the 128-wide one (39-56% at 2048
   and 4096 cubed, float32). *)
let plan_simt (c : call) ((a, b, _) as read) ~wmma_s8 sum =
  let cheaper = cost c 64 64 50 < cost c 128 128 100 in
  let bm, bn =
    if c.m <= 16 then ((if sum == K.F64 then 8 else 16), 128)
    else if
      find_simt sum 128 128 >= 0 && (find_simt sum 64 64 < 0 || not cheaper)
    then (128, 128)
    else (64, 64)
  in
  let blocks = ceil_div c.m bm * ceil_div c.n bn in
  launch c read ~kernel:(find_simt sum bm bn) ~blocks
    ~splits:(split_count (blocks * c.batch) 64 c.k 128)
    ~threads:K.threads ~values:(bm * bn / K.threads)
    ~sum_bytes:(if sum == K.F32 then 4 else 8)
    ~aligned:
      ((if rows_vectors a then K.a_vectors else 0)
      lor if rows_vectors b then K.b_vectors else 0)
    ~wmma_s8 sum

(* The WMMA kind of operands of [a] and [b] into [acc], if one sums them. float8
   operands decode exactly to bfloat16 and sum on its matrix unit, whose sums
   the suite checks against the error bound: the float8 unit's are unchecked. *)
let wmma_kind a b acc : K.kind option =
  let bf16_like x = same x bf16 || is_f8 x in
  if not (same acc f32 || same acc i32) then None
  else if bf16_like a && bf16_like b then
    if same acc f32 then Some Bf16 else None
  else if same a f16 && same b f16 then if same acc f32 then Some F16 else None
  else if same a i8 && same b i8 && same acc i32 then Some S8
  else None

let declines ~a ~b ~y ~i ~init ~acc =
  let float_acc = same acc f32 || same acc f64 in
  (not (summable a))
  || (not (summable b))
  || (not (summable y))
  || (init && ((not (summable i)) || is_int i = float_acc))
  || float_acc
     && (bytes a > bytes acc
        || bytes b > bytes acc
        || (init && bytes i > bytes acc))

(* The accumulator family of [acc] over [a] and [b], or [None] to decline. *)
let sum_of ~acc a b : K.acc option =
  if same acc f32 || same acc f64 then
    if is_int a || is_int b then None
    else if same acc f32 then Some F32
    else Some F64
  else if same acc i32 || unsigned_32 acc || same acc i64 || unsigned_64 acc
  then if is_int a && is_int b then Some I64 else None
  else None

(* The rules of nx.amd's contraction, reading nothing but [c]. *)
let rules (c : call) =
  let batch = c.batch and m = c.m and n = c.n and k = c.k in
  if batch > 65535 || m > max_int32 || n > max_int32 || k > max_int32 then
    declined
  else if batch * m * n = 0 then Ok Nothing
  else
    let a = c.a.dtype and b = c.b.dtype and acc = c.acc in
    let init = Option.is_some c.init in
    let i = match c.init with Some i -> i.i_dtype | None -> c.y_dtype in
    if declines ~a ~b ~y:c.y_dtype ~i ~init ~acc then declined
    else
      let kind = wmma_kind a b acc in
      let t = match kind with None -> -1 | Some kd -> wmma_tile c kd in
      let wmma_s8 = match kind with Some S8 -> t >= 0 | _ -> false in
      match sum_of ~acc a b with
      | None -> declined
      | Some sum -> (
          let planned =
            match kind with
            | _ when in_lane_order c sum -> plan_lane c ~wmma_s8 sum
            | Some kd when t >= 0 -> plan_wmma c kd K.tiles.(t) ~wmma_s8 sum
            | _ -> (
                match packed_own c sum with
                | None -> None
                | Some read ->
                    (* An integer sum is exact in its wrapping, so every
                       order gives it the same bits: few rows take the skinny
                       kernels. *)
                    if m <= 16 && sum == K.I64 then
                      plan_skinny c read ~wmma_s8 sum
                    else plan_simt c read ~wmma_s8 sum)
          in
          match planned with None -> declined | Some l -> Ok (Launch l))

(* [x] as the operand [slot] of [v], [which], over its axis [free] and k. *)
let operand v slot which (free : V.axis) x =
  let dtype = dtype_of x in
  let first = V.offset v which * bytes dtype in
  let buffer = Nx_array.buffer x in
  {
    slot;
    buffer;
    dtype;
    first;
    vector_first = Rig.Buffer.aligned buffer ~at:first 16;
    s =
      [|
        V.stride v which Batch;
        V.stride v which free;
        V.stride v which Contracted;
      |];
    pack = None;
  }

let strides v which : int array =
  [| V.stride v which Batch; V.stride v which Row; V.stride v which Column |]

let choose ~order ~batch ~contracting ~acc ~dst ~init a b =
  match V.v ~batch ~contracting ~dst ~init a b with
  | Error answer -> Error answer
  | Ok v ->
      let y_dtype = dtype_of dst in
      let y_first = V.offset v Dst * bytes y_dtype in
      let y = Nx_array.buffer dst in
      let init =
        match init with
        | None -> None
        | Some i ->
            let i_dtype = dtype_of i in
            Some
              {
                i_buffer = Nx_array.buffer i;
                i_dtype;
                i_first = V.offset v Init * bytes i_dtype;
                si = strides v Init;
              }
      in
      rules
        {
          a = operand v 0 A Row a;
          b = operand v 1 B Column b;
          y;
          y_dtype;
          y_first;
          y_vector_first = Rig.Buffer.aligned y ~at:y_first 16;
          sy = strides v Dst;
          init;
          acc;
          order;
          batch = V.extent v Batch;
          m = V.extent v Row;
          n = V.extent v Column;
          k = V.extent v Contracted;
        }

(* Sequences *)

(* The submission's key: its contraction kernel, then its packs, split, init and
   swap, dense over [count] kernels, so no count collides. *)
let flags = 32
let sequences = count * flags
let pack_a_flag = 1
let pack_b_flag = 2
let split_flag = 4
let init_flag = 8
let swap_flag = 16
let packed o = Option.is_some o.pack
let has_init l = Option.is_some l.call.init
let flag f b = if b then f else 0

let sequence l =
  l.kernel
  + count
    * (flag pack_a_flag (packed l.a)
      lor flag pack_b_flag (packed l.b)
      lor flag split_flag (l.splits > 1)
      lor flag init_flag (has_init l)
      lor flag swap_flag (l.a.slot = 1))

(* The workspace is a slot of the sequence whenever a launch addresses it,
   whatever its bytes: a pack of no element takes none. *)
let uses_workspace l = packed l.a || packed l.b || l.splits > 1
let workspace l = if uses_workspace l then Int.max 1 l.used else 0

(* The slots: the call's [a] 0, [b] 1 and [y] 2, then [init] and the
   workspace where the sequence has them. *)
let y_slot = 2
let init_slot = 3
let ws_slot l = init_slot + Bool.to_int (has_init l)

let buffers l ~workspace =
  let c = l.call in
  let ws = uses_workspace l in
  let bs = Array.make (3 + Bool.to_int (has_init l) + Bool.to_int ws) c.y in
  bs.(l.a.slot) <- l.a.buffer;
  bs.(l.b.slot) <- l.b.buffer;
  (match c.init with Some i -> bs.(init_slot) <- i.i_buffer | None -> ());
  if ws then bs.(ws_slot l) <- workspace;
  bs

(* The launches of the sequence [key] in order: each kernel, its parameter
   bytes and its refs into the slots [buffers] fills, each with its slot's
   access. a reads the call's operand 1 where the sequence swaps. *)
let launches key =
  if key < 0 || key >= sequences then
    invalid_arg "Plan.parts: a key past sequences";
  let kernel = key mod count and flags = key / count in
  let has flag = flags land flag <> 0 in
  let pack_a = has pack_a_flag and pack_b = has pack_b_flag in
  let split = has split_flag and init = has init_flag in
  let a_slot = Bool.to_int (has swap_flag) in
  let b_slot = 1 - a_slot in
  let ws = init_slot + Bool.to_int init in
  let acc =
    Array.concat
      [
        [| Rig.Read; Rig.Read; Rig.Read_write |];
        (if init then [| Rig.Read |] else [||]);
        (if pack_a || pack_b || split then [| Rig.Read_write |] else [||]);
      ]
  in
  let slot_ref at slot =
    match acc.(slot) with
    | Rig.Read -> Sub.read slot ~at
    | Read_write -> Sub.write slot ~at
  in
  let pack slot =
    ( pack_kernel,
      K.Pack_params.size,
      [| slot_ref K.Pack_params.src slot; slot_ref K.Pack_params.dst ws |] )
  in
  let module P = K.Contract_params in
  let refs =
    List.concat
      [
        [ slot_ref P.a (if pack_a then ws else a_slot) ];
        [ slot_ref P.b (if pack_b then ws else b_slot) ];
        (if init then [ slot_ref P.init init_slot ] else []);
        [ slot_ref P.y y_slot ];
        (if split then [ slot_ref P.partials ws; slot_ref P.tickets ws ]
         else []);
      ]
  in
  List.concat
    [
      (if pack_a then [ pack a_slot ] else []);
      (if pack_b then [ pack b_slot ] else []);
      (if split then
         [
           (zero_kernel, K.Zero_params.size, [| slot_ref K.Zero_params.p ws |]);
         ]
       else []);
      [ (kernel, P.size, Array.of_list refs) ];
    ]

let parts key image =
  let part (kernel, params, refs) =
    let kernel = fst K.kernels.(kernel) in
    Sub.launch ~refs image kernel ~params
  in
  Array.of_list (List.map part (launches key))

(* Writing a run *)

let write_pack run at (c : call) o (p : pack) =
  let module P = K.Pack_params in
  let n = c.batch * p.rows * p.lead in
  Run.groups run at (ceil_div n K.threads) 1 1;
  Run.threads run at K.threads 1 1;
  Run.shared run at 0;
  Run.int64 run at P.src p.src;
  Run.int64 run at P.dst o.first;
  Run.int64 run at P.s p.ps.(0);
  Run.int64 run at (P.s + 8) p.ps.(1);
  Run.int64 run at (P.s + 16) p.ps.(2);
  Run.int64 run at P.lead p.lead;
  Run.int32 run at P.batch c.batch;
  Run.int32 run at P.rows p.rows;
  Run.int32 run at P.k c.k;
  Run.int32 run at P.dtype (code p.own);
  Run.int32 run at P.out (code p.into);
  Run.int32 run at P.bytes (bytes p.into)

let write_strides run at field s =
  Run.int64 run at field s.(0);
  Run.int64 run at (field + 8) s.(1);
  Run.int64 run at (field + 16) s.(2)

(* Stores every field of each launch's struct, and test_amd_kernels.ml shows
   that kernels.ml's fields tile each struct, [zero] included: every parameter
   byte is stored before each submit, so bytes another submission left in the
   run never reach a kernel. *)
let write l run sub =
  (* Parts in [launches]' order: a's pack, b's pack, the tickets' zeroing, the
     contraction. *)
  let c = l.call in
  (match l.a.pack with
  | Some p -> write_pack run (Rig.Submission.block sub 0) c l.a p
  | None -> ());
  let a = Bool.to_int (packed l.a) in
  (match l.b.pack with
  | Some p -> write_pack run (Rig.Submission.block sub a) c l.b p
  | None -> ());
  let b = Bool.to_int (packed l.b) in
  let split = l.splits > 1 in
  if split then begin
    let at = Rig.Submission.block sub (a + b) in
    Run.groups run at (ceil_div l.tiles K.threads) 1 1;
    Run.threads run at K.threads 1 1;
    Run.shared run at 0;
    Run.int64 run at K.Zero_params.p l.tickets;
    Run.int64 run at K.Zero_params.n l.tiles
  end;
  (* Without init, the kernel reads y's dtype and strides in its place. *)
  let i_first, si, i_dtype =
    match c.init with
    | Some i -> (i.i_first, i.si, i.i_dtype)
    | None -> (0, c.sy, c.y_dtype)
  in
  let module P = K.Contract_params in
  let at = Rig.Submission.block sub (a + b + Bool.to_int split) in
  Run.groups run at l.blocks l.splits c.batch;
  Run.threads run at l.threads 1 1;
  Run.shared run at 0;
  Run.int64 run at P.a l.a.first;
  Run.int64 run at P.b l.b.first;
  Run.int64 run at P.init i_first;
  Run.int64 run at P.y c.y_first;
  Run.int64 run at P.partials (if split then l.partials else 0);
  Run.int64 run at P.tickets (if split then l.tickets else 0);
  write_strides run at P.sa l.a.s;
  write_strides run at P.sb l.b.s;
  write_strides run at P.si si;
  write_strides run at P.sy c.sy;
  Run.int32 run at P.batch c.batch;
  Run.int32 run at P.m c.m;
  Run.int32 run at P.n c.n;
  Run.int32 run at P.k c.k;
  Run.int32 run at P.splits l.splits;
  Run.int32 run at P.a_dtype (code l.a.dtype);
  Run.int32 run at P.b_dtype (code l.b.dtype);
  Run.int32 run at P.init_dtype (code i_dtype);
  Run.int32 run at P.y_dtype (code c.y_dtype);
  Run.int32 run at P.acc_dtype (code c.acc);
  Run.int32 run at P.aligned l.aligned;
  Run.int32 run at P.zero 0
