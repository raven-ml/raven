(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A plan declines a call, and nx's expansion computes it, when an operand, init
   or y is complex or narrower than a byte; a float operand or init is wider
   than a float accumulator, which would round it before the sum; init is of the
   other kind than the accumulator; an integer meets a float accumulator or the
   reverse; the batch is above 65,535 (the grid's z); or m, n or k is above 2^31
   - 1. A call whose axes do not group never reaches the plan:
   Contract_view.fill answers false first. *)

module K = Kernels
module V = Nx_kernel.Spec.Contract_view
module S = Nx_kernel.Spec
module D = Nx_array.Dtype
module Run = Rig.Submission.Run

(* Dtypes, by code *)

let codes = List.length D.all
let bits = Array.make codes 0
let signed = Array.make codes false
let unsigned = Array.make codes false
let complex = Array.make codes false
let boolean = Array.make codes false

let () =
  List.iter
    (fun (D.Any d) ->
      let c = D.code d in
      bits.(c) <- D.bits d;
      signed.(c) <- D.is D.Signed d;
      unsigned.(c) <- D.is D.Unsigned d;
      complex.(c) <- D.is D.Complex d;
      boolean.(c) <- D.is D.Boolean d)
    D.all

let f64 = D.code D.Float64
let f32 = D.code D.Float32
let f16 = D.code D.Float16
let bf16 = D.code D.Bfloat16
let e4m3 = D.code D.Float8_e4m3fn
let e5m2 = D.code D.Float8_e5m2
let i64 = D.code D.Int64
let u64 = D.code D.Uint64
let i32 = D.code D.Int32
let u32 = D.code D.Uint32
let i16 = D.code D.Int16
let u16 = D.code D.Uint16
let i8 = D.code D.Int8
let u8 = D.code D.Uint8
let bool = D.code D.Bool
let code (D.Any d) = D.code d
let code_of (Nx_array.Any x) = D.code (Nx_array.dtype x)
let width c = bits.(c) / 8
let is_f8 c = c = e4m3 || c = e5m2
let is_int c = signed.(c) || unsigned.(c) || boolean.(c)

(* Whether [c] reads as a value the kernels sum: no complex, no sub-byte
   element. *)
let summable c = (not complex.(c)) && bits.(c) >= 8

(* The integer dtype the skinny kernel reads integer operands [x] and [y] in:
   the narrowest that holds both exactly, or 64 bits, whose products wrap as the
   accumulator's sum does. *)
let common_int x y =
  let sx = signed.(x) and sy = signed.(y) in
  let widest = Int.max bits.(x) bits.(y) in
  (* A signed type holds an unsigned one's values with a bit to spare. *)
  let unsigned = if sx then bits.(y) else bits.(x) in
  let b = if sx <> sy then Int.max widest (2 * unsigned) else widest in
  if b > 32 then i64
  else if sx || sy then if b <= 8 then i8 else if b <= 16 then i16 else i32
  else if b <= 8 then u8
  else if b <= 16 then u16
  else u32

(* Whether a SIMT or skinny kernel reads [dt] as [into] with no pack: the same
   bytes, bool as uint8, and either 64-bit integer as the other (a float64 never
   meets an integer [into]: the plan declines that call). *)
let reads_as dt into =
  dt = into || (dt = bool && into = u8) || (width dt = 8 && width into = 8)

(* Instances *)

let index i =
  let rec go k = if K.instances.(k) = i then k else go (k + 1) in
  go 0

let pack_kernel = index K.Pack

(* The first mma instance that sums [kind] with a's and b's contiguous axes [a]
   and [b] on the tile [t], or -1. An instance of kind [Any] sums every kind. *)
(* Loops with their state as arguments: a local loop closing over them would
   allocate its closure on every call. *)
let rec find_mma_from k kind a b t =
  if k = K.count then -1
  else
    match K.instances.(k) with
    | Mma (kind', a', b', t')
      when (kind' == kind || kind' == K.Any) && a' == a && b' == b && t' == t ->
        k
    | _ -> find_mma_from (k + 1) kind a b t

let find_mma kind a b t = find_mma_from 0 kind a b t

let rec find_simt_from k sum side =
  if k = K.count then -1
  else
    match K.instances.(k) with
    | Simt (sum', side') when sum' == sum && side' = side -> k
    | _ -> find_simt_from (k + 1) sum side

let find_simt sum side = find_simt_from 0 sum side

let rec find_skinny_from k sum =
  if k = K.count then -1
  else
    match K.instances.(k) with
    | Skinny sum' when sum' == sum -> k
    | _ -> find_skinny_from (k + 1) sum

let find_skinny sum = find_skinny_from 0 sum
let threads (s : K.shape) = s.bm / s.wm * (s.bn / s.wn) * 32
let shared (s : K.shape) = s.stages * (s.bm + s.bn) * s.bkb

(* Plans *)

let max_int32 = 0x7fff_ffff
let ceil_div a b = (a + b - 1) / b

(* An operand a or b as the contraction kernel reads it, and, when packed, as it
   is: the pack copies it into the workspace. *)
type operand = {
  mutable dtype : int;  (** As the kernel reads it. *)
  mutable first : int;
      (** Its first element: bytes into its slot, or into the workspace once
          packed. *)
  mutable address : int;
      (** [first] as the device addresses it, for alignment: the workspace
          offset once packed, the workspace being 256-byte aligned. *)
  s : int array;  (** Strides along batch, its row or column, and k. *)
  mutable packed : bool;
  mutable own : int;  (** Its own dtype. *)
  mutable src : int;  (** Its own first element, bytes into its slot. *)
  ps : int array;  (** Its own strides. *)
  mutable rows : int;
  mutable lead : int;  (** Elements a packed row. *)
  mutable grid : int;  (** The pack's blocks. *)
}

let operand () =
  {
    dtype = 0;
    first = 0;
    address = 0;
    s = Array.make 3 0;
    packed = false;
    own = 0;
    src = 0;
    ps = Array.make 3 0;
    rows = 0;
    lead = 0;
    grid = 0;
  }

type t = {
  a : operand;
  b : operand;
  sy : int array;
  si : int array;
  mutable y_first : int;
  mutable y_address : int;
  mutable y_dtype : int;
  mutable i_first : int;
  mutable i_dtype : int;
  mutable init : bool;
  mutable acc : int;
  mutable batch : int;
  mutable m : int;
  mutable n : int;
  mutable k : int;
  (* The contraction kernel's launch: its [blocks] x [splits] x batch blocks of
     [threads] threads and [shared] dynamic shared bytes, and the [sums] partial
     sums a block holds when it splits, [sum_bytes] each. *)
  mutable kernel : int;
  mutable splits : int;
  mutable blocks : int;
  mutable threads : int;
  mutable shared : int;
  mutable sums : int;
  mutable sum_bytes : int;
  mutable aligned : int;
  (* The workspace: its bytes taken, and the split sum's pieces. *)
  mutable used : int;
  mutable partials : int;
  mutable count : int;  (** Output tiles of a split sum: its tickets. *)
  costs : Float.Array.t;
      (** Two tiles' costs while choosing one: floats unboxed. *)
}

let make () =
  {
    a = operand ();
    b = operand ();
    sy = Array.make 3 0;
    si = Array.make 3 0;
    y_first = 0;
    y_address = 0;
    y_dtype = 0;
    i_first = 0;
    i_dtype = 0;
    init = false;
    acc = 0;
    batch = 0;
    m = 0;
    n = 0;
    k = 0;
    kernel = 0;
    splits = 1;
    blocks = 0;
    threads = 0;
    shared = 0;
    sums = 0;
    sum_bytes = 0;
    aligned = 0;
    used = 0;
    partials = 0;
    count = 0;
    costs = Float.Array.make 2 0.;
  }

type verdict = Declined | Nothing | Launches

(* Whether rows along [contiguous] (stride 1) load as 16-byte vectors: every
   other stride a whole number of vectors, and the first element on a vector. *)
let vectors address contiguous lead batch bytes =
  let per = 16 / bytes in
  contiguous = 1 && lead mod per = 0 && batch mod per = 0 && address mod 16 = 0

(* Whether an operand of the strides [s] (batch, row, k) has its rows contiguous
   rather than its k. *)
let free_contiguous s = s.(1) = 1 && s.(2) <> 1

(* Whether [o]'s rows load as 16-byte vectors along its contiguous axis. *)
let rows_vectors o =
  let s = o.s in
  if free_contiguous s then vectors o.address s.(1) s.(2) s.(0) (width o.dtype)
  else vectors o.address s.(2) s.(1) s.(0) (width o.dtype)

(* [bytes] of the workspace after those taken: their offset, each piece on a
   256-byte boundary. *)
let take c bytes =
  let at = (c.used + 255) land lnot 255 in
  c.used <- at + bytes;
  at

(* Packs [o], of [rows] rows, into the workspace as elements of [into] with k
   contiguous and rows of whole vectors, and makes the kernel read the copy. *)
let pack c o ~rows ~into =
  let es = width into in
  let per = 16 / es in
  let lead = ceil_div c.k per * per in
  let at = take c (c.batch * rows * lead * es) in
  o.packed <- true;
  o.own <- o.dtype;
  o.src <- o.first;
  Array.blit o.s 0 o.ps 0 3;
  o.rows <- rows;
  o.lead <- lead;
  (* A thread a 16-byte vector. *)
  let grid = ceil_div (c.batch * rows * lead / per) 256 in
  o.grid <- (if grid = 0 then 1 else Int.min grid 65535);
  o.dtype <- into;
  o.first <- at;
  o.address <- at;
  o.s.(0) <- rows * lead;
  o.s.(1) <- lead;
  o.s.(2) <- 1

(* Packs each operand the SIMT or skinny kernel cannot read as [into] into
   [into], exactly, with k contiguous: the kernel then reads both as [into]. *)
let pack_into c into =
  if not (reads_as c.a.dtype into) then pack c c.a ~rows:c.m ~into;
  if not (reads_as c.b.dtype into) then pack c c.b ~rows:c.n ~into;
  c.a.dtype <- into;
  c.b.dtype <- into

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

(* Blocks run in waves of [wave], the 100 SMs of kimchi's RTX 5000 Ada, and a
   wave lasts its tile's outputs over the tile's efficiency, its outputs per
   unit of time against its family's fastest tile, in percent. A GPU of another
   size runs the same tiles, with the same bits. *)
let wave = 100

(* Stores the cost of [bm] x [bn] tiles of [efficiency] into [c.costs]'s [slot]:
   an array of floats, so that no float is boxed. *)
let cost c slot bm bn efficiency =
  let blocks = c.batch * ceil_div c.m bm * ceil_div c.n bn in
  Float.Array.unsafe_set c.costs slot
    (Float.of_int (ceil_div blocks wave)
    *. Float.of_int bm *. Float.of_int bn /. Float.of_int efficiency)

let cheaper c =
  Float.Array.unsafe_get c.costs 1 < Float.Array.unsafe_get c.costs 0

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
let mma_tile c kind =
  let best = ref (-1) in
  for i = 0 to Array.length K.tiles - 1 do
    let t = K.tiles.(i) in
    if find_mma kind K.K K.K t >= 0 && not (c.m <= 16 && t != K.T16x64) then begin
      let s = K.shape t in
      cost c 1 s.bm s.bn (efficiency t);
      if !best < 0 || cheaper c then begin
        Float.Array.unsafe_set c.costs 0 (Float.Array.unsafe_get c.costs 1);
        best := i
      end
    end
  done;
  !best

(* The mma kernels, of [kind] on the tile [t]. An operand whose rows are not
   vectors is packed into the workspace with k contiguous, as is a float8 one
   and one in a layout the tile has no instance of. *)
let plan_mma c kind t =
  let s = K.shape t in
  let a = c.a and b = c.b in
  let la = if free_contiguous a.s then K.M else K.K in
  let lb = if free_contiguous b.s then K.N else K.K in
  let pack_a =
    (not (rows_vectors a))
    || is_f8 a.dtype
    || (la != K.K && find_mma kind la K.K t < 0)
  in
  let la = if pack_a then K.K else la in
  let pack_b =
    (not (rows_vectors b))
    || is_f8 b.dtype
    || (lb != K.K && find_mma kind la lb t < 0)
  in
  let lb = if pack_b then K.K else lb in
  c.kernel <- find_mma kind la lb t;
  c.blocks <- ceil_div c.m s.bm * ceil_div c.n s.bn;
  if c.kernel < 0 || c.blocks > max_int32 then false
  else begin
    let into = match kind with S8 -> i8 | F16 -> f16 | Bf16 | Any -> bf16 in
    if pack_a then pack c a ~rows:c.m ~into;
    if pack_b then pack c b ~rows:c.n ~into;
    a.dtype <- into;
    b.dtype <- into;
    (* A split sum stores and reloads its partials: worth it to fill a GPU short
       of blocks, or to stream a long k for a few rows. *)
    let mes = if kind == K.S8 then 1 else 2 in
    c.splits <-
      (if c.m <= 16 then
         split_count (c.blocks * c.batch) 256 c.k (4 * s.bkb / mes)
       else split_count (c.blocks * c.batch) 64 c.k 1024);
    c.threads <- threads s;
    c.shared <- shared s;
    c.sums <- s.bm * s.bn;
    c.sum_bytes <- 4;
    true
  end

(* The dtype of the accumulator [sum]: what the SIMT kernels read, and the float
   skinny kernels. *)
let own_dtype : K.acc -> int = function F32 -> f32 | F64 -> f64 | I64 -> i64

(* The SIMT kernels of the accumulator [sum], m > 16: the tile of least cost
   among its instances; on sm_89, the 64-wide tile computes 70% as fast as the
   128-wide one. They read their accumulator's own dtype. *)
let plan_simt c sum =
  cost c 0 128 128 100;
  cost c 1 64 64 70;
  let side =
    if find_simt sum 128 >= 0 && (find_simt sum 64 < 0 || not (cheaper c)) then
      128
    else 64
  in
  c.kernel <- find_simt sum side;
  c.blocks <- ceil_div c.m side * ceil_div c.n side;
  if c.kernel < 0 || c.blocks > max_int32 then false
  else begin
    pack_into c (own_dtype sum);
    (* A SIMT block's k-tiles run one after another, each waiting on its loads:
       split while the grid has fewer than 256 blocks, down to 64 of k a
       range. *)
    c.splits <- split_count (c.blocks * c.batch) 256 c.k 64;
    c.threads <- 256;
    c.shared <- 0;
    c.sums <- side * side;
    c.sum_bytes <- (if sum == K.F32 then 4 else 8);
    c.aligned <-
      ((if rows_vectors c.a then K.a_vectors else 0)
      lor if rows_vectors c.b then K.b_vectors else 0);
    true
  end

(* The skinny kernel of the accumulator [sum], m <= 16. It reads a float
   accumulator's own dtype, or one integer dtype both operands hold. *)
let plan_skinny c sum =
  c.kernel <- find_skinny sum;
  c.blocks <- ceil_div c.m K.skinny_rows * ceil_div c.n 32;
  if c.kernel < 0 || c.blocks > max_int32 then false
  else begin
    let into =
      if sum == K.I64 then common_int c.a.dtype c.b.dtype else own_dtype sum
    in
    pack_into c into;
    (* By the columns alone: a row's sums are the same bits whatever rows come
       with it. *)
    c.splits <- split_count (ceil_div c.n 32 * c.batch) 64 c.k 1024;
    (* A 32-bit integer accumulator sums in 32 bits here. *)
    let narrow = c.acc = i32 || c.acc = u32 in
    c.threads <- 256;
    c.shared <- 0;
    c.sums <- 256;
    c.sum_bytes <- (if sum == K.F32 || narrow then 4 else 8);
    let across = free_contiguous c.b.s in
    c.aligned <-
      ((if across then K.b_across
        else if rows_vectors c.b then K.b_vectors
        else 0)
      lor
      if (not (free_contiguous c.a.s)) && rows_vectors c.a then K.a_vectors
      else 0);
    true
  end

(* [o] as the operand [which] of [v], read from the array [x]. *)
let read o v which x =
  let dt = code_of x in
  let (Nx_array.Any arr) = x in
  o.dtype <- dt;
  o.first <- V.offset v which * width dt;
  o.address <- Rig.Buffer.address (Nx_array.buffer arr) + o.first;
  o.packed <- false

let strides into v which (row : V.axis) (col : V.axis) =
  into.(0) <- V.stride v which Batch;
  into.(1) <- V.stride v which row;
  into.(2) <- V.stride v which col

(* The mma kind of operands of [a] and [b] into [acc], if one sums them. float8
   operands decode exactly to bfloat16 and sum on its matrix unit: Ada's float8
   unit keeps 13 bits of its sums, where the error bound needs each addition to
   err by at most 2u. *)
let mma_kind a b acc : K.kind option =
  let bf16_like x = x = bf16 || is_f8 x in
  if acc = f32 && bf16_like a && bf16_like b then Some Bf16
  else if acc = f32 && a = f16 && b = f16 then Some F16
  else if acc = i32 && a = i8 && b = i8 then Some S8
  else None

let declines ~ad ~bd ~yd ~id ~init ~acc =
  let float_acc = acc = f32 || acc = f64 in
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
  if acc = f32 || acc = f64 then
    if is_int a || is_int b then None
    else if acc = f32 then Some F32
    else Some F64
  else if acc = i32 || acc = u32 || acc = i64 || acc = u64 then
    if is_int a && is_int b then Some I64 else None
  else None

(* Outputs stored 16 bytes at once: no init, y's j contiguous, its rows on
   16-byte boundaries, and its dtype one the kernels store as they are, or
   bfloat16 or float16 from float32 (store8, store_pair). *)
let whole c ~mma_s8 (sum : K.acc) ~y_address =
  let yd = c.y_dtype in
  let yb = width yd in
  let natural =
    if mma_s8 then yd = i32 || yd = u32
    else
      match sum with
      | F32 -> yd = f32 || yd = bf16 || yd = f16
      | F64 -> yd = f64
      | I64 -> (yd = i64 || yd = u64) && (c.acc = i64 || c.acc = u64)
  in
  (not c.init) && natural
  && c.sy.(2) = 1
  && c.sy.(1) * yb mod 16 = 0
  && c.sy.(0) * yb mod 16 = 0
  && y_address mod 16 = 0

(* The plan of the call whose fields [choose] read: the rules of nx.cuda's
   contraction, reading nothing but [c]. *)
let rules c =
  let batch = c.batch and m = c.m and n = c.n and k = c.k in
  if batch > 65535 || m > max_int32 || n > max_int32 || k > max_int32 then
    Declined
  else if batch * m * n = 0 then Nothing
  else
    let ad = c.a.dtype and bd = c.b.dtype and acc = c.acc in
    if declines ~ad ~bd ~yd:c.y_dtype ~id:c.i_dtype ~init:c.init ~acc then
      Declined
    else
      match sum_of ~acc ad bd with
      | None -> Declined
      | Some sum ->
          c.used <- 0;
          c.aligned <- 0;
          let kind = mma_kind ad bd acc in
          let t = match kind with None -> -1 | Some kd -> mma_tile c kd in
          let planned =
            match kind with
            | Some kd when t >= 0 -> plan_mma c kd K.tiles.(t)
            | _ -> if m <= 16 then plan_skinny c sum else plan_simt c sum
          in
          if not planned then Declined
          else begin
            let mma_s8 = match kind with Some S8 -> t >= 0 | _ -> false in
            if whole c ~mma_s8 sum ~y_address:c.y_address then
              c.aligned <- c.aligned lor K.y_whole;
            (* The split sum's partials, and a ticket a tile. *)
            if c.splits > 1 then begin
              c.count <- batch * c.blocks;
              c.partials <- take c (c.count * c.splits * c.sums * c.sum_bytes)
            end;
            Launches
          end

let choose c v s ~dst ops =
  c.batch <- V.extent v Batch;
  c.m <- V.extent v Row;
  c.n <- V.extent v Column;
  c.k <- V.extent v Contracted;
  c.init <- S.init s;
  c.acc <- code (S.acc s);
  read c.a v A ops.(0);
  strides c.a.s v A Row Contracted;
  read c.b v B ops.(1);
  strides c.b.s v B Column Contracted;
  let (Nx_array.Any y) = dst in
  c.y_dtype <- code_of dst;
  c.y_first <- V.offset v Dst * width c.y_dtype;
  c.y_address <- Rig.Buffer.address (Nx_array.buffer y) + c.y_first;
  strides c.sy v Dst Row Column;
  if c.init then begin
    c.i_dtype <- code_of ops.(2);
    c.i_first <- V.offset v Init * width c.i_dtype;
    strides c.si v Init Row Column
  end
  else begin
    c.i_dtype <- c.y_dtype;
    Array.blit c.sy 0 c.si 0 3
  end;
  rules c

(* Sequences *)

(* The submission's key: its contraction kernel, then its packs, split and init,
   dense over [K.count] kernels, so no count collides. *)
let flags = 16
let sequences = K.count * flags

let sequence c =
  c.kernel
  + K.count
    * (Bool.to_int c.a.packed
      lor (Bool.to_int c.b.packed lsl 1)
      lor (Bool.to_int (c.splits > 1) lsl 2)
      lor (Bool.to_int c.init lsl 3))

let reads c = if c.init then 3 else 2

(* The workspace is a slot of the sequence whenever a launch addresses it,
   whatever its bytes: a pack of no element takes none. *)
let uses_workspace c = c.a.packed || c.b.packed || c.splits > 1
let writes c =
  1 + Bool.to_int (uses_workspace c) + Bool.to_int (c.splits > 1)

let tickets c = if c.splits > 1 then 4 * c.count else 0
let workspace c = if uses_workspace c then Int.max 1 c.used else 0

(* [c]'s launches in order: each kernel, its parameter bytes and its refs into
   the slots [reads] and [writes] count. *)
let launches c =
  let y = reads c in
  let ws = y + 1 in
  let ref at slot = { Rig.Submission.at; slot } in
  let pack slot =
    ( pack_kernel,
      K.Pack_params.size,
      [| ref K.Pack_params.src slot; ref K.Pack_params.dst ws |] )
  in
  let module P = K.Contract_params in
  let split = c.splits > 1 in
  let refs =
    List.concat
      [
        [ ref P.a (if c.a.packed then ws else 0) ];
        [ ref P.b (if c.b.packed then ws else 1) ];
        (if c.init then [ ref P.init 2 ] else []);
        [ ref P.y y ];
        (if split then [ ref P.partials ws; ref P.tickets (ws + 1) ] else []);
      ]
  in
  List.concat
    [
      (if c.a.packed then [ pack 0 ] else []);
      (if c.b.packed then [ pack 1 ] else []);
      [ (c.kernel, P.size, Array.of_list refs) ];
    ]

let parts c image ~queue =
  let part (kernel, params, refs) =
    let kernel = K.names.(kernel) in
    {
      Rig.Submission.queue;
      after = [||];
      work = Launch { image; kernel; params; refs };
    }
  in
  Array.of_list (List.map part (launches c))

(* Writing a run *)

let write_pack run at c o =
  let module P = K.Pack_params in
  Run.groups run at o.grid 1 1;
  Run.threads run at 256 1 1;
  Run.shared run at 0;
  Run.int64 run at P.src o.src;
  Run.int64 run at P.dst o.first;
  Run.int64 run at P.s o.ps.(0);
  Run.int64 run at (P.s + 8) o.ps.(1);
  Run.int64 run at (P.s + 16) o.ps.(2);
  Run.int64 run at P.lead o.lead;
  Run.int32 run at P.batch c.batch;
  Run.int32 run at P.rows o.rows;
  Run.int32 run at P.k c.k;
  Run.int32 run at P.dtype o.own;
  Run.int32 run at P.out o.dtype;
  Run.int32 run at P.bytes (width o.dtype)

let write_strides run at field s =
  Run.int64 run at field s.(0);
  Run.int64 run at (field + 8) s.(1);
  Run.int64 run at (field + 16) s.(2)

let write run sub c =
  (* Parts in [parts]' order; a counter, so no closure is allocated. *)
  let i = ref 0 in
  if c.a.packed then begin
    write_pack run (Rig.Submission.block sub !i) c c.a;
    incr i
  end;
  if c.b.packed then begin
    write_pack run (Rig.Submission.block sub !i) c c.b;
    incr i
  end;
  let split = c.splits > 1 in
  let module P = K.Contract_params in
  let at = Rig.Submission.block sub !i in
  Run.groups run at c.blocks c.splits c.batch;
  Run.threads run at c.threads 1 1;
  Run.shared run at c.shared;
  Run.int64 run at P.a c.a.first;
  Run.int64 run at P.b c.b.first;
  Run.int64 run at P.init (if c.init then c.i_first else 0);
  Run.int64 run at P.y c.y_first;
  Run.int64 run at P.partials (if split then c.partials else 0);
  Run.int64 run at P.tickets 0;
  write_strides run at P.sa c.a.s;
  write_strides run at P.sb c.b.s;
  write_strides run at P.si c.si;
  write_strides run at P.sy c.sy;
  Run.int32 run at P.batch c.batch;
  Run.int32 run at P.m c.m;
  Run.int32 run at P.n c.n;
  Run.int32 run at P.k c.k;
  Run.int32 run at P.splits c.splits;
  Run.int32 run at P.a_dtype c.a.dtype;
  Run.int32 run at P.b_dtype c.b.dtype;
  Run.int32 run at P.init_dtype c.i_dtype;
  Run.int32 run at P.y_dtype c.y_dtype;
  Run.int32 run at P.acc_dtype c.acc;
  Run.int32 run at P.aligned c.aligned;
  Run.int32 run at P.unused 0
