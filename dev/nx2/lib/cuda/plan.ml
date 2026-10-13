(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A plan declines a call, and nx's expansion computes it, when an operand, init
   or y is complex or narrower than a byte; a float operand or init is wider
   than a float accumulator, which would round it before the sum; init is of the
   other kind than the accumulator; an integer meets a float accumulator or the
   reverse; the batch is above 65,535 (the grid's z); m, n or k is above 2^31 -
   1; or a batch element has more than 2^31 - 1 output tiles (the grid's x). A
   call whose axes do not group answers what Contract_view.v answers. *)

module K = Kernels
module V = Nx_kernel.Contract_view
module D = Nx_array.Dtype
module Run = Rig.Submission.Run
module Sub = Rig.Submission

(* Dtypes *)

(* A dtype's code: only the kernels' parameters hold one. *)
let code (D.Any d) = D.code d
let dtype_of x = D.Any (Nx_array.dtype x)
let bits (D.Any d) = D.bits d
let width dt = bits dt / 8
let signed (D.Any d) = D.is Signed d
let same (D.Any d) (D.Any d') = D.equal d d'
let is_f8 = function D.Any (Float8_e4m3fn | Float8_e5m2) -> true | _ -> false

let is_int (D.Any d) =
  match D.kind d with
  | Signed | Unsigned | Boolean -> true
  | Float | Complex -> false

(* Whether [dt] reads as a value the kernels sum: no complex, no sub-byte
   element. *)
let summable (D.Any d) = (not (D.is Complex d)) && D.bits d >= 8

(* The integer dtype the skinny kernel reads integer operands [x] and [y] in:
   the narrowest that holds both exactly, or 64 bits, whose products wrap as the
   accumulator's sum does. *)
let common_int x y : D.any =
  let sx = signed x and sy = signed y in
  let widest = Int.max (bits x) (bits y) in
  (* A signed type holds an unsigned one's values with a bit to spare. *)
  let unsigned = if sx then bits y else bits x in
  let b = if sx <> sy then Int.max widest (2 * unsigned) else widest in
  if b > 32 then D.Any Int64
  else if sx || sy then
    if b <= 8 then D.Any Int8 else if b <= 16 then D.Any Int16 else D.Any Int32
  else if b <= 8 then D.Any Uint8
  else if b <= 16 then D.Any Uint16
  else D.Any Uint32

(* Whether a SIMT or skinny kernel reads [dt] as [into] with no pack: the same
   bytes, bool as uint8, and either 64-bit integer as the other (a float64 never
   meets an integer [into]: the plan declines that call). *)
let reads_as dt into =
  same dt into
  || (match (dt, into) with D.Any Bool, D.Any Uint8 -> true | _ -> false)
  || (width dt = 8 && width into = 8)

(* Instances *)

let count = Array.length K.kernels

let index i =
  let rec go k = if snd K.kernels.(k) = i then k else go (k + 1) in
  go 0

let pack_kernel = index K.Pack

(* The first mma instance that sums [kind] with a's and b's contiguous axes [a]
   and [b] on the tile [t], or -1. An instance of kind [Any] sums every kind.
   These finds are loops with their state as arguments: a local loop closing
   over them would allocate its closure on every call. *)
let rec find_mma_from k kind a b t =
  if k = count then -1
  else
    match snd K.kernels.(k) with
    | Mma (kind', a', b', t')
      when (kind' == kind || kind' == K.Any) && a' == a && b' == b && t' == t ->
        k
    | _ -> find_mma_from (k + 1) kind a b t

let find_mma kind a b t = find_mma_from 0 kind a b t

(* A SIMT instance's block of outputs, its threads and its efficiency, its
   outputs per unit of time against the 128 x 256 block's, in percent, on
   sm_89. *)
let simt_bm (s : K.simt) = s.warps_m * 4 * s.tm
let simt_bn (s : K.simt) = s.warps_n * 8 * s.tn
let simt_threads (s : K.simt) = s.warps_m * s.warps_n * 32
let simt_efficiency s = if simt_bm s * simt_bn s >= 128 * 256 then 100 else 60

let rec find_skinny_from k sum =
  if k = count then -1
  else
    match snd K.kernels.(k) with
    | Skinny sum' when sum' == sum -> k
    | _ -> find_skinny_from (k + 1) sum

let find_skinny sum = find_skinny_from 0 sum
let threads (s : K.shape) = s.bm / s.wm * (s.bn / s.wn) * 32
let shared (s : K.shape) = s.stages * (s.bm + s.bn) * s.bkb

(* Plans *)

let max_int32 = 0x7fff_ffff

(* The largest row, column and contracted extent the kernels take. They form
   coordinates in int32: an extent plus a tile or a run, a split's first
   k-tile, a row times an element's two bytes. Each stays below 2^31 from an
   extent of 2^30. *)
let max_extent = 1 lsl 30
let ceil_div a b = (a + b - 1) / b


(* An operand a or b as it lies, which a pack copies into the workspace: its
   dtype, its first element (bytes into its slot) and its strides; its rows,
   the elements of a packed row, and the pack's blocks. *)
type pack = {
  own : D.any;
  src : int;
  ps : int array;
  rows : int;
  lead : int;
  grid : int;
}

(* An operand a or b as the contraction kernel reads it: its buffer; its
   dtype; its first element, bytes into its slot or into the workspace once
   packed; whether that element lies on a 16-byte boundary as the device
   addresses it, the workspace being 256-byte aligned; its strides along
   batch, its row or column, and k; and its pack, if packed. *)
type operand = {
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

(* A call as the rules read it: a and b as they lie, y as a's init. *)
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
  batch : int;
  m : int;
  n : int;
  k : int;
}

(* A call's launches: a and b as the kernel reads them; the contraction
   kernel's [blocks] x [splits] x batch blocks of [threads] threads and
   [shared] dynamic shared bytes; the workspace bytes taken; and the split
   sum's partials, at [partials] in the workspace, and its output tiles, a
   ticket each. *)
type launch = {
  call : call;
  a : operand;
  b : operand;
  kernel : int;
  splits : int;
  blocks : int;
  threads : int;
  shared : int;
  aligned : int;
  used : int;
  partials : int;
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
    vectors o.vector_first s.(1) s.(2) s.(0) (width o.dtype)
  else vectors o.vector_first s.(2) s.(1) s.(0) (width o.dtype)

(* Each workspace piece starts on a 256-byte boundary. *)
let align used = (used + 255) land lnot 255

(* [o], of [rows] rows, packed into the workspace after its [used] bytes as
   elements of [into], with k contiguous and rows of whole vectors: the kernel
   reads the copy. *)
let pack (c : call) o ~rows ~into ~used =
  let es = width into in
  let per = 16 / es in
  let lead = ceil_div c.k per * per in
  let at = align used in
  (* A thread a 16-byte vector. *)
  let grid = ceil_div (c.batch * rows * lead / per) 256 in
  let grid = if grid = 0 then 1 else Int.min grid 65535 in
  {
    o with
    dtype = into;
    first = at;
    vector_first = at mod 16 = 0;
    s = [| rows * lead; lead; 1 |];
    pack = Some { own = o.dtype; src = o.first; ps = o.s; rows; lead; grid };
  }

(* The workspace bytes taken up to the end of [o]'s copy, [used] if it has
   none. *)
let used_after (c : call) used o =
  match o.pack with
  | None -> used
  | Some _ -> o.first + (c.batch * o.s.(0) * width o.dtype)

(* a and b read as [into], each packed where [pack_a] and [pack_b] say, a's
   copy first, and the workspace bytes the copies take. *)
let read_as (c : call) ~into ~pack_a ~pack_b =
  let a =
    if pack_a then pack c c.a ~rows:c.m ~into ~used:0
    else { c.a with dtype = into }
  in
  let used = used_after c 0 a in
  let b =
    if pack_b then pack c c.b ~rows:c.n ~into ~used
    else { c.b with dtype = into }
  in
  (a, b, used_after c used b)

(* The split count of a grid of [grid] blocks: doubled while the grid has fewer
   than [target] blocks and each range keeps at least [k_min] of k, at most 16.
   The targets are constants measured on an architecture's GPUs, never a GPU's
   own count. *)
let split_count grid target k k_min =
  let s = ref 1 in
  while !s < 16 && grid * !s < target && k / (2 * !s) >= k_min do
    s := 2 * !s
  done;
  !s

(* [c]'s launch of [kernel] over a and b as [read_as] gives them; with
   [splits] above 1, its partials, [sums] of [sum_bytes] a block, follow the
   copies in the workspace. [whole] adds y_whole to [aligned]. *)
let launch (c : call) (a, b, used) ~kernel ~blocks ~splits ~threads ~shared
    ~sums ~sum_bytes ~aligned ~whole =
  let split = splits > 1 in
  let tiles = if split then c.batch * blocks else 0 in
  let partials = if split then align used else 0 in
  {
    call = c;
    a;
    b;
    kernel;
    splits;
    blocks;
    threads;
    shared;
    aligned = (if whole then aligned lor K.y_whole else aligned);
    used =
      (if split then partials + (tiles * splits * sums * sum_bytes) else used);
    partials;
    tiles;
  }

(* Blocks run in waves of [wave], the 100 SMs of kimchi's RTX 5000 Ada, and a
   wave lasts its tile's outputs over the tile's efficiency, its outputs per
   unit of time against its family's fastest tile, in percent. A GPU of another
   size runs the same tiles, with the same bits. *)
let wave = 100

(* The cost of [bm] x [bn] tiles of [efficiency]. Inlined, so that its float
   is never boxed. *)
let[@inline] cost (c : call) bm bn efficiency =
  let blocks = c.batch * ceil_div c.m bm * ceil_div c.n bn in
  Float.of_int (ceil_div blocks wave)
  *. Float.of_int bm *. Float.of_int bn /. Float.of_int efficiency

(* The mma tiles' efficiency, against the 128 x 256 tile's, on sm_89. *)
let efficiency : K.tile -> int = function
  | T128x128 -> 90
  | T128x256 -> 100
  | T64x64 -> 60
  | T16x64 -> 30

(* The index in [K.tiles] of the mma tile of a product among those [kind] has an
   instance of with k contiguous: the one of least cost, by its shape alone; m
   <= 16 takes the 16-row tile, which larger m take where it costs least. -1 if
   [kind] has none for the shape: the SIMT or skinny kernels sum it. *)
let mma_tile (c : call) kind =
  let best = ref (-1) and least = ref 0. in
  for i = 0 to Array.length K.tiles - 1 do
    let t = K.tiles.(i) in
    if find_mma kind K.K K.K t >= 0 && (c.m > 16 || t == K.T16x64) then begin
      let s = K.shape t in
      let cost = cost c s.bm s.bn (efficiency t) in
      if !best < 0 || cost < !least then begin
        least := cost;
        best := i
      end
    end
  done;
  !best

(* The mma kernels, of [kind] on the tile [t]. An operand whose rows are not
   vectors is packed into the workspace with k contiguous, as is a float8 one
   and one in a layout the tile has no instance of. *)
let plan_mma (c : call) kind t ~whole =
  let s = K.shape t in
  let la = if free_contiguous c.a.s then K.M else K.K in
  let lb = if free_contiguous c.b.s then K.N else K.K in
  let pack_a =
    (not (rows_vectors c.a))
    || is_f8 c.a.dtype
    || (la != K.K && find_mma kind la K.K t < 0)
  in
  let la = if pack_a then K.K else la in
  let pack_b =
    (not (rows_vectors c.b))
    || is_f8 c.b.dtype
    || (lb != K.K && find_mma kind la lb t < 0)
  in
  let lb = if pack_b then K.K else lb in
  let kernel = find_mma kind la lb t in
  let blocks = ceil_div c.m s.bm * ceil_div c.n s.bn in
  if kernel < 0 || blocks > max_int32 then None
  else
    let into =
      match kind with
      | S8 -> D.Any Int8
      | F16 -> D.Any Float16
      | Bf16 | K.Any -> D.Any Bfloat16
    in
    (* A split sum stores and reloads its partials: worth it to fill a GPU short
       of blocks, or to stream a long k for a few rows. *)
    let splits =
      if c.m <= 16 then
        split_count (blocks * c.batch) 256 c.k (4 * s.bkb / width into)
      else split_count (blocks * c.batch) 64 c.k 1024
    in
    Some
      (launch c
         (read_as c ~into ~pack_a ~pack_b)
         ~kernel ~blocks ~splits ~threads:(threads s) ~shared:(shared s)
         ~sums:(s.bm * s.bn) ~sum_bytes:4 ~aligned:0 ~whole)

(* The dtype of the accumulator [sum]: what the SIMT kernels read, and the float
   skinny kernels. *)
let own_dtype : K.acc -> D.any = function
  | F32 -> D.Any Float32
  | F64 -> D.Any Float64
  | I64 -> D.Any Int64

(* a and b read as [into], each the SIMT or skinny kernel cannot read as
   [into] packed into it, exactly, with k contiguous. *)
let read_into (c : call) into =
  read_as c ~into
    ~pack_a:(not (reads_as c.a.dtype into))
    ~pack_b:(not (reads_as c.b.dtype into))

(* The SIMT kernels of the accumulator [sum], m > 16: the instance of least
   cost among its. They read their accumulator's own dtype. *)
let plan_simt (c : call) sum ~whole =
  let best = ref (-1) and least = ref 0. in
  for k = 0 to count - 1 do
    match snd K.kernels.(k) with
    | Simt (sum', s) when sum' == sum ->
        let cost = cost c (simt_bm s) (simt_bn s) (simt_efficiency s) in
        if !best < 0 || cost < !least then begin
          least := cost;
          best := k
        end
    | _ -> ()
  done;
  match if !best < 0 then None else Some (snd K.kernels.(!best)) with
  | Some (Simt (_, s)) ->
      let bm = simt_bm s and bn = simt_bn s in
      let blocks = ceil_div c.m bm * ceil_div c.n bn in
      if blocks > max_int32 then None
      else
        let ((a, b, _) as read) = read_into c (own_dtype sum) in
        (* Each stage holds a's and b's k-tiles, k-major, a k's row padded by
           16 bytes. *)
        let bytes = width (own_dtype sum) in
        Some
          (launch c read ~kernel:!best ~blocks
             (* A SIMT block's k-tiles run one after another, each waiting on
                its loads: split while the grid has fewer than 256 blocks, down
                to 64 of k a range. *)
             ~splits:(split_count (blocks * c.batch) 256 c.k 64)
             ~threads:(simt_threads s)
             ~shared:(K.simt_stages * K.simt_bk * (((bm + bn) * bytes) + 32))
             ~sums:(bm * bn) ~sum_bytes:bytes
             ~aligned:
               ((if rows_vectors a then K.a_vectors else 0)
               lor if rows_vectors b then K.b_vectors else 0)
             ~whole)
  | _ -> None

(* The skinny kernel of the accumulator [sum], m <= 16. It reads a float
   accumulator's own dtype, or one integer dtype both operands hold. *)
let plan_skinny (c : call) sum ~whole =
  let kernel = find_skinny sum in
  let blocks = ceil_div c.m K.skinny_rows * ceil_div c.n 32 in
  if kernel < 0 || blocks > max_int32 then None
  else
    let into =
      if sum == K.I64 then common_int c.a.dtype c.b.dtype else own_dtype sum
    in
    let ((a, b, _) as read) = read_into c into in
    (* A 32-bit integer accumulator sums in 32 bits here. *)
    let narrow = match c.acc with D.Any (Int32 | Uint32) -> true | _ -> false in
    let across = free_contiguous b.s in
    Some
      (launch c read ~kernel ~blocks
         (* By the columns alone: a row's sums are the same bits whatever rows
            come with it. *)
         ~splits:(split_count (ceil_div c.n 32 * c.batch) 64 c.k 1024)
         ~threads:256 ~shared:0 ~sums:256
         ~sum_bytes:(if sum == K.F32 || narrow then 4 else 8)
         ~aligned:
           ((if across then K.b_across
             else if rows_vectors b then K.b_vectors
             else 0)
           lor
           if (not (free_contiguous a.s)) && rows_vectors a then K.a_vectors
           else 0)
         ~whole)

(* The mma kind of operands of [a] and [b] into [acc], if one sums them. float8
   operands decode exactly to bfloat16 and sum on its matrix unit: Ada's float8
   unit keeps 13 bits of its sums, where the error bound needs each addition to
   err by at most 2u. *)
let mma_kind a b acc : K.kind option =
  match (acc, a, b) with
  | ( D.Any Float32,
      D.Any (Bfloat16 | Float8_e4m3fn | Float8_e5m2),
      D.Any (Bfloat16 | Float8_e4m3fn | Float8_e5m2) ) ->
      Some Bf16
  | D.Any Float32, D.Any Float16, D.Any Float16 -> Some F16
  | D.Any Int32, D.Any Int8, D.Any Int8 -> Some S8
  | _ -> None

let is_float_acc = function D.Any (Float32 | Float64) -> true | _ -> false

let declines ~ad ~bd ~yd ~id ~init ~acc =
  let float_acc = is_float_acc acc in
  (not (summable ad))
  || (not (summable bd))
  || (not (summable yd))
  || (init && ((not (summable id)) || is_int id = float_acc))
  || float_acc
     && (width ad > width acc
        || width bd > width acc
        || (init && width id > width acc))

(* The accumulator family of [acc] over [a] and [b], or [None] to decline. *)
let sum_of ~acc a b : K.acc option =
  match acc with
  | D.Any Float32 -> if is_int a || is_int b then None else Some F32
  | D.Any Float64 -> if is_int a || is_int b then None else Some F64
  | D.Any (Int32 | Uint32 | Int64 | Uint64) ->
      if is_int a && is_int b then Some I64 else None
  | _ -> None

(* Outputs stored 16 bytes at once: no init, y's j contiguous, its rows on
   16-byte boundaries, and its dtype one the kernels store as they are, or
   bfloat16 or float16 from float32 (store8, store_pair). *)
let whole (c : call) ~mma_s8 (sum : K.acc) =
  let yd = c.y_dtype in
  let yb = width yd in
  let natural =
    if mma_s8 then match yd with D.Any (Int32 | Uint32) -> true | _ -> false
    else
      match (sum, yd, c.acc) with
      | F32, D.Any (Float32 | Bfloat16 | Float16), _ -> true
      | F64, D.Any Float64, _ -> true
      | I64, D.Any (Int64 | Uint64), D.Any (Int64 | Uint64) -> true
      | _ -> false
  in
  Option.is_none c.init && natural
  && c.sy.(2) = 1
  && c.sy.(1) * yb mod 16 = 0
  && c.sy.(0) * yb mod 16 = 0
  && c.y_vector_first

(* The rules of nx.cuda's contraction, reading nothing but [c]. *)
let rules (c : call) =
  let batch = c.batch and m = c.m and n = c.n and k = c.k in
  if batch > 65535 || m > max_extent || n > max_extent || k > max_extent then
    declined
  else if batch * m * n = 0 then Ok Nothing
  else
    let ad = c.a.dtype and bd = c.b.dtype and acc = c.acc in
    let init = Option.is_some c.init in
    let id = match c.init with Some i -> i.i_dtype | None -> c.y_dtype in
    if declines ~ad ~bd ~yd:c.y_dtype ~id ~init ~acc then declined
    else
      match sum_of ~acc ad bd with
      | None -> declined
      | Some sum -> (
          let kind = mma_kind ad bd acc in
          let t = match kind with None -> -1 | Some kd -> mma_tile c kd in
          let mma_s8 = match kind with Some S8 -> t >= 0 | _ -> false in
          let whole = whole c ~mma_s8 sum in
          let planned =
            match kind with
            | Some kd when t >= 0 -> plan_mma c kd K.tiles.(t) ~whole
            | _ ->
                if m <= 16 then plan_skinny c sum ~whole
                else plan_simt c sum ~whole
          in
          match planned with None -> declined | Some l -> Ok (Launch l))

(* [x] as the operand [which] of [v], over its axis [free] and k. *)
let operand v which (free : V.axis) x =
  let dtype = dtype_of x in
  let first = V.offset v which * width dtype in
  let buffer = Nx_array.buffer x in
  {
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

let choose ~batch ~contracting ~acc ~dst ~init a b =
  match V.v ~batch ~contracting ~dst ~init a b with
  | Error answer -> Error answer
  | Ok v ->
      let y_dtype = dtype_of dst in
      let y_first = V.offset v Dst * width y_dtype in
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
                i_first = V.offset v Init * width i_dtype;
                si = strides v Init;
              }
      in
      rules
        {
          a = operand v A Row a;
          b = operand v B Column b;
          y;
          y_dtype;
          y_first;
          y_vector_first = Rig.Buffer.aligned y ~at:y_first 16;
          sy = strides v Dst;
          init;
          acc;
          batch = V.extent v Batch;
          m = V.extent v Row;
          n = V.extent v Column;
          k = V.extent v Contracted;
        }

(* Sequences *)

(* The submission's key: its contraction kernel, then its packs, split and init,
   dense over [count] kernels, so no count collides. *)
let flags = 16
let sequences = count * flags
let pack_a_flag = 1
let pack_b_flag = 2
let split_flag = 4
let init_flag = 8
let packed o = Option.is_some o.pack
let has_init l = Option.is_some l.call.init
let flag f b = if b then f else 0

let sequence l =
  l.kernel
  + count
    * (flag pack_a_flag (packed l.a)
      lor flag pack_b_flag (packed l.b)
      lor flag split_flag (l.splits > 1)
      lor flag init_flag (has_init l))

(* The workspace is a slot of the sequence whenever a launch addresses it,
   whatever its bytes: a pack of no element takes none. *)
let uses_workspace l = packed l.a || packed l.b || l.splits > 1
let tickets l = if l.splits > 1 then 4 * l.tiles else 0
let workspace l = if uses_workspace l then Int.max 1 l.used else 0

(* The slots: [a] 0, [b] 1 and [y] 2, then [init], the workspace and the tickets
   where the sequence has them. A split uses the workspace, so the tickets
   follow it. *)
let y_slot = 2
let init_slot = 3
let ws_slot l = init_slot + Bool.to_int (has_init l)

let buffers l ~workspace ~tickets =
  let c = l.call in
  let ws = uses_workspace l and split = l.splits > 1 in
  let count =
    3 + Bool.to_int (has_init l) + Bool.to_int ws + Bool.to_int split
  in
  let bs = Array.make count c.y in
  bs.(0) <- c.a.buffer;
  bs.(1) <- c.b.buffer;
  (match c.init with Some i -> bs.(init_slot) <- i.i_buffer | None -> ());
  if ws then bs.(ws_slot l) <- workspace;
  if split then bs.(ws_slot l + 1) <- tickets;
  bs

(* The launches of the sequence [key] in order: each kernel, its parameter
   bytes and its refs into the slots [buffers] fills, each with its slot's
   access. *)
let launches key =
  if key < 0 || key >= sequences then
    invalid_arg "Plan.parts: a key past sequences";
  let kernel = key mod count and flags = key / count in
  let has flag = flags land flag <> 0 in
  let pack_a = has pack_a_flag and pack_b = has pack_b_flag in
  let split = has split_flag and init = has init_flag in
  let ws = init_slot + Bool.to_int init in
  let acc =
    Array.concat
      [
        [| Rig.Read; Rig.Read; Rig.Read_write |];
        (if init then [| Rig.Read |] else [||]);
        (if pack_a || pack_b || split then [| Rig.Read_write |] else [||]);
        (if split then [| Rig.Read_write |] else [||]);
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
        [ slot_ref P.a (if pack_a then ws else 0) ];
        [ slot_ref P.b (if pack_b then ws else 1) ];
        (if init then [ slot_ref P.init init_slot ] else []);
        [ slot_ref P.y y_slot ];
        (if split then [ slot_ref P.partials ws; slot_ref P.tickets (ws + 1) ]
         else []);
      ]
  in
  List.concat
    [
      (if pack_a then [ pack 0 ] else []);
      (if pack_b then [ pack 1 ] else []);
      [ (kernel, P.size, Array.of_list refs) ];
    ]

let parts key image =
  let part (kernel, params, refs) =
    let kernel = fst K.kernels.(kernel) in
    Sub.launch ~refs image kernel ~params
  in
  Array.of_list (List.map part (launches key))

(* Writing a run *)

let write_pack run at c o (p : pack) =
  let module P = K.Pack_params in
  Run.groups run at p.grid 1 1;
  Run.threads run at 256 1 1;
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
  Run.int32 run at P.out (code o.dtype);
  Run.int32 run at P.bytes (width o.dtype)

let write_strides run at field s =
  Run.int64 run at field s.(0);
  Run.int64 run at (field + 8) s.(1);
  Run.int64 run at (field + 16) s.(2)

let write l run sub =
  (* Parts in [parts]' order: a's pack, b's pack, the contraction. *)
  let c = l.call in
  (match l.a.pack with
  | Some p -> write_pack run (Rig.Submission.block sub 0) c l.a p
  | None -> ());
  let a = Bool.to_int (packed l.a) in
  (match l.b.pack with
  | Some p -> write_pack run (Rig.Submission.block sub a) c l.b p
  | None -> ());
  let b = Bool.to_int (packed l.b) in
  (* Without init, the kernel reads y's dtype and strides in its place. *)
  let i_first, si, i_dtype =
    match c.init with
    | Some i -> (i.i_first, i.si, i.i_dtype)
    | None -> (0, c.sy, c.y_dtype)
  in
  let split = l.splits > 1 in
  let module P = K.Contract_params in
  let at = Rig.Submission.block sub (a + b) in
  Run.groups run at l.blocks l.splits c.batch;
  Run.threads run at l.threads 1 1;
  Run.shared run at l.shared;
  Run.int64 run at P.a l.a.first;
  Run.int64 run at P.b l.b.first;
  Run.int64 run at P.init i_first;
  Run.int64 run at P.y c.y_first;
  Run.int64 run at P.partials (if split then l.partials else 0);
  Run.int64 run at P.tickets 0;
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
  Run.int32 run at P.aligned l.aligned
