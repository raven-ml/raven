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
   workgroups along x pass 2^31 - 1. A call whose axes do not group never
   reaches the plan: Contract_view.fill answers false first. *)

module K = Kernels
module V = Nx_kernel.Spec.Contract_view
module S = Nx_kernel.Spec
module D = Nx_array.Dtype
module Run = Rig.Submission.Run

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
let dtype_of (Nx_array.Any x) = D.Any (Nx_array.dtype x)

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
let rec find_wmma_from k kind t =
  if k = count then -1
  else
    match snd K.kernels.(k) with
    | Wmma (kind', t') when kind' == kind && t' == t -> k
    | _ -> find_wmma_from (k + 1) kind t

let find_wmma kind t = find_wmma_from 0 kind t

let rec find_simt_from k sum side =
  if k = count then -1
  else
    match snd K.kernels.(k) with
    | Simt (sum', side') when sum' == sum && side' = side -> k
    | _ -> find_simt_from (k + 1) sum side

let find_simt sum side = find_simt_from 0 sum side

let rec find_skinny_from k sum form =
  if k = count then -1
  else
    match snd K.kernels.(k) with
    | Skinny (sum', form') when sum' == sum && form' == form -> k
    | _ -> find_skinny_from (k + 1) sum form

let find_skinny sum form = find_skinny_from 0 sum form
let tile_threads (s : K.shape) = s.bm / s.wm * (s.bn / s.wn) * 32

(* Plans *)

let max_int32 = 0x7fff_ffff
let max_uint32 = 0xffff_ffff
let ceil_div a b = (a + b - 1) / b

(* An operand a or b as the contraction kernel reads it, and, when packed, as it
   is: the pack copies it into the workspace. *)
type operand = {
  (* As the kernel reads it: its dtype, as the kernel's parameters name it; its
     first element, bytes into its slot or into the workspace once packed;
     [first] as the device addresses it, for alignment, the workspace offset
     once packed, the workspace being 256-byte aligned; and its strides along
     batch, its row or column, and k. *)
  mutable dtype : D.any;
  mutable first : int;
  mutable address : int;
  s : int array;
  (* As it is, once packed: its dtype, the dtype its pack writes, its first
     element (bytes into its slot) and strides; its rows, and the elements a
     packed row. *)
  mutable packed : bool;
  mutable own : D.any;
  mutable into : D.any;
  mutable src : int;
  ps : int array;
  mutable rows : int;
  mutable lead : int;
}

let operand () =
  {
    dtype = f32;
    first = 0;
    address = 0;
    s = Array.make 3 0;
    packed = false;
    own = f32;
    into = f32;
    src = 0;
    ps = Array.make 3 0;
    rows = 0;
    lead = 0;
  }

type t = {
  a : operand;
  b : operand;
  sy : int array;
  si : int array;
  mutable y_first : int;
  mutable y_address : int;
  mutable y_dtype : D.any;
  mutable i_first : int;
  mutable i_dtype : D.any;
  mutable init : bool;
  mutable acc : D.any;
  mutable batch : int;
  mutable m : int;
  mutable n : int;
  mutable k : int;
  (* The contraction kernel's launch: its [blocks] x [splits] x batch workgroups
     of [threads] work-items, and the [values] partial sums each work-item holds
     when it splits, [sum_bytes] each. *)
  mutable kernel : int;
  mutable splits : int;
  mutable blocks : int;
  mutable threads : int;
  mutable values : int;
  mutable sum_bytes : int;
  mutable aligned : int;
  (* The workspace: its bytes taken, and the split sum's pieces: its partials,
     its tickets, and its output tiles, a ticket each. *)
  mutable used : int;
  mutable partials : int;
  mutable tickets : int;
  mutable tiles : int;
  (* Two tiles' costs while choosing one: floats unboxed. *)
  costs : Float.Array.t;
}

let make () =
  {
    a = operand ();
    b = operand ();
    sy = Array.make 3 0;
    si = Array.make 3 0;
    y_first = 0;
    y_address = 0;
    y_dtype = f32;
    i_first = 0;
    i_dtype = f32;
    init = false;
    acc = f32;
    batch = 0;
    m = 0;
    n = 0;
    k = 0;
    kernel = 0;
    splits = 1;
    blocks = 0;
    threads = 0;
    values = 0;
    sum_bytes = 0;
    aligned = 0;
    used = 0;
    partials = 0;
    tickets = 0;
    tiles = 0;
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
  if free_contiguous s then vectors o.address s.(1) s.(2) s.(0) (bytes o.dtype)
  else vectors o.address s.(2) s.(1) s.(0) (bytes o.dtype)

(* [bytes] of the workspace after those taken: their offset, each piece on a
   256-byte boundary. *)
let take c bytes =
  let at = (c.used + 255) land lnot 255 in
  c.used <- at + bytes;
  at

(* Packs [o], of [rows] rows, into the workspace as elements of [into] with k
   contiguous and rows of whole vectors, a work-item an element, and makes the
   kernel read the copy; [false] if its elements outnumber the work-items a grid
   holds. *)
let pack c o ~rows ~into =
  let es = bytes into in
  let per = 16 / es in
  let lead = ceil_div c.k per * per in
  if c.batch * rows * lead > max_uint32 * K.threads then false
  else begin
    let at = take c (c.batch * rows * lead * es) in
    o.packed <- true;
    o.own <- o.dtype;
    o.into <- into;
    o.src <- o.first;
    Array.blit o.s 0 o.ps 0 3;
    o.rows <- rows;
    o.lead <- lead;
    o.first <- at;
    o.address <- at;
    o.s.(0) <- rows * lead;
    o.s.(1) <- lead;
    o.s.(2) <- 1;
    true
  end

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

(* Workgroups run in waves of [wave] on gfx1201: the 128 x 128 tile's time steps
   every 96 workgroups, three to each of the R9700's 32 work-group processors. A
   GPU of another size runs the same tiles, with the same bits. *)
let wave = 96

(* Stores the cost of [bm] x [bn] tiles of [efficiency] into [c.costs]'s [slot],
   waves times outputs per tile over its efficiency: an array of floats, so that
   no float is boxed. *)
let cost c slot bm bn efficiency =
  let groups = c.batch * ceil_div c.m bm * ceil_div c.n bn in
  Float.Array.unsafe_set c.costs slot
    (Float.of_int (ceil_div groups wave)
    *. Float.of_int bm *. Float.of_int bn /. Float.of_int efficiency)

let cheaper c =
  Float.Array.unsafe_get c.costs 1 < Float.Array.unsafe_get c.costs 0

(* The WMMA tiles' outputs per unit of time against the 128 x 128 tile's, in
   percent, on the R9700: the 64 x 64 tile makes 76-83% at 2048 and 4096 cubed,
   bfloat16 and float16. The 16 x 64 tile's 33 is Ada's: m <= 16 takes that tile
   alone, so its rate is never weighed. *)
let efficiency : K.tile -> int = function
  | T128x128 -> 100
  | T64x64 -> 80
  | T16x64 -> 33

(* The index in [K.tiles] of the WMMA tile of a product among those [kind] has
   an instance of: the one of least cost, by its shape alone; m <= 16 takes the
   16-row tile, and only it. -1 if [kind] has none for the shape: the SIMT or
   skinny kernels sum it. *)
let wmma_tile c kind =
  let best = ref (-1) in
  for i = 0 to Array.length K.tiles - 1 do
    let t = K.tiles.(i) in
    if find_wmma kind t >= 0 && c.m <= 16 = (t == K.T16x64) then begin
      let s = K.shape t in
      cost c 1 s.bm s.bn (efficiency t);
      if !best < 0 || cheaper c then begin
        Float.Array.unsafe_set c.costs 0 (Float.Array.unsafe_get c.costs 1);
        best := i
      end
    end
  done;
  !best

(* The WMMA kernels, of [kind] on the tile [t]. They read rows of k as whole
   vectors and never past a row's last: another operand, or one whose k ends
   inside a vector, is packed into the workspace with k contiguous, its rows
   padded with zeros. The parameters keep the operands' own dtypes: a pack
   changes none. *)
let plan_wmma c kind t =
  let s = K.shape t in
  let a = c.a and b = c.b in
  let width = if kind == K.S8 then 1 else 2 in
  let whole = c.k mod (16 / width) = 0 in
  let pack_a =
    free_contiguous a.s || (not (rows_vectors a)) || is_f8 a.dtype || not whole
  in
  let pack_b =
    free_contiguous b.s || (not (rows_vectors b)) || is_f8 b.dtype || not whole
  in
  let into = match kind with S8 -> i8 | F16 -> f16 | Bf16 -> bf16 in
  let packs_a = (not pack_a) || pack c a ~rows:c.m ~into in
  let packs_b = packs_a && ((not pack_b) || pack c b ~rows:c.n ~into) in
  if not packs_b then false
  else begin
    c.kernel <- find_wmma kind t;
    c.blocks <- ceil_div c.m s.bm * ceil_div c.n s.bn;
    (* A split sum stores and reloads its partials: worth it to fill a GPU short
       of workgroups, or to stream a long k for a few rows. *)
    c.splits <-
      (if c.m <= 16 then
         split_count (c.blocks * c.batch) 128 c.k (4 * s.bkb / width)
       else split_count (c.blocks * c.batch) 64 c.k 1024);
    c.threads <- tile_threads s;
    c.values <- s.bm * s.bn / c.threads;
    c.sum_bytes <- 4;
    true
  end

(* Whether [d] is the dtype the SIMT and skinny kernels of [sum] read. *)
let reads_own (sum : K.acc) d =
  match sum with
  | F32 -> same d f32
  | F64 -> same d f64
  | I64 -> same d i64 || unsigned_64 d

let own_dtype : K.acc -> D.any = function F32 -> f32 | F64 -> f64 | I64 -> i64

(* SIMT and skinny kernels read their accumulator's own dtype: an operand of
   another is packed into it, exactly, with k contiguous. *)
let pack_own c sum =
  let into = own_dtype sum in
  let packs_a = reads_own sum c.a.dtype || pack c c.a ~rows:c.m ~into in
  let packs_b =
    packs_a && (reads_own sum c.b.dtype || pack c c.b ~rows:c.n ~into)
  in
  if packs_a && c.a.packed then c.a.dtype <- into;
  if packs_b && c.b.packed then c.b.dtype <- into;
  packs_b

(* The skinny kernel of [sum], m <= 16, in its column form, or across when b's n
   axis is contiguous. Both forms split by the column form's grid, so the
   association is the shape's. *)
let plan_skinny c sum =
  let across = free_contiguous c.b.s in
  c.kernel <- find_skinny sum (if across then Across else Column);
  c.blocks <- (if across then ceil_div c.n 32 else ceil_div c.n 8);
  c.splits <- split_count (ceil_div c.n 8 * c.batch) 256 c.k 1024;
  c.values <- 2;
  c.threads <- K.threads;
  c.sum_bytes <- (if sum == K.F32 then 4 else 8);
  c.aligned <-
    ((if (not across) && rows_vectors c.b then K.b_vectors else 0)
    lor
    if (not (free_contiguous c.a.s)) && rows_vectors c.a then K.a_vectors else 0
    )

(* The SIMT tile of least cost among [sum]'s instances, m > 16: on the R9700 the
   64-wide tile computes half as fast as the 128-wide one (39-56% at 2048 and
   4096 cubed, float32). *)
let plan_simt c sum =
  cost c 0 128 128 100;
  cost c 1 64 64 50;
  let side =
    if find_simt sum 128 >= 0 && (find_simt sum 64 < 0 || not (cheaper c)) then
      128
    else 64
  in
  c.kernel <- find_simt sum side;
  c.blocks <- ceil_div c.m side * ceil_div c.n side;
  c.splits <- split_count (c.blocks * c.batch) 64 c.k 128;
  c.values <- side * side / K.threads;
  c.threads <- K.threads;
  c.sum_bytes <- (if sum == K.F32 then 4 else 8);
  c.aligned <-
    ((if rows_vectors c.a then K.a_vectors else 0)
    lor if rows_vectors c.b then K.b_vectors else 0)

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

(* Outputs stored 16 bytes at once: no init, y's j contiguous, its rows on
   16-byte boundaries, and its dtype one the kernels store as they are, or
   bfloat16 or float16 from float32 (store8). *)
let whole c ~wmma_s8 (sum : K.acc) =
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
  (not c.init) && natural
  && c.sy.(2) = 1
  && c.sy.(1) * yb mod 16 = 0
  && c.sy.(0) * yb mod 16 = 0
  && c.y_address mod 16 = 0

(* The plan of the call whose fields [choose] read: the rules of nx.amd's
   contraction, reading nothing but [c]. *)
let rules c =
  let batch = c.batch and m = c.m and n = c.n and k = c.k in
  if batch > 65535 || m > max_int32 || n > max_int32 || k > max_int32 then
    Declined
  else if batch * m * n = 0 then Nothing
  else
    let a = c.a.dtype and b = c.b.dtype and acc = c.acc in
    if declines ~a ~b ~y:c.y_dtype ~i:c.i_dtype ~init:c.init ~acc then Declined
    else
      let kind = wmma_kind a b acc in
      let t = match kind with None -> -1 | Some kd -> wmma_tile c kd in
      match sum_of ~acc a b with
      | None -> Declined
      | Some sum ->
          c.used <- 0;
          c.aligned <- 0;
          let planned =
            match kind with
            | Some kd when t >= 0 -> plan_wmma c kd K.tiles.(t)
            | _ ->
                pack_own c sum
                && begin
                  if m <= 16 then plan_skinny c sum else plan_simt c sum;
                  true
                end
          in
          if (not planned) || c.blocks > max_int32 then Declined
          else begin
            let wmma_s8 = match kind with Some S8 -> t >= 0 | _ -> false in
            if whole c ~wmma_s8 sum then c.aligned <- c.aligned lor K.y_whole;
            (* The split sum's partials and tickets, the tickets zeroed
               first. *)
            if c.splits > 1 then begin
              c.tiles <- batch * c.blocks;
              c.partials <-
                take c (c.tiles * c.splits * c.values * c.threads * c.sum_bytes);
              c.tickets <- take c (c.tiles * 4)
            end;
            Launches
          end

(* [o] as the operand [which] of [v], read from the array [x]. *)
let read o v which x =
  let (Nx_array.Any arr) = x in
  o.dtype <- dtype_of x;
  o.first <- V.offset v which * bytes o.dtype;
  o.address <- Rig.Buffer.address (Nx_array.buffer arr) + o.first;
  o.packed <- false

let strides into v which (row : V.axis) (col : V.axis) =
  into.(0) <- V.stride v which Batch;
  into.(1) <- V.stride v which row;
  into.(2) <- V.stride v which col

let choose c v s ~dst ops =
  c.batch <- V.extent v Batch;
  c.m <- V.extent v Row;
  c.n <- V.extent v Column;
  c.k <- V.extent v Contracted;
  c.init <- S.init s;
  c.acc <- S.acc s;
  read c.a v A ops.(0);
  strides c.a.s v A Row Contracted;
  read c.b v B ops.(1);
  strides c.b.s v B Column Contracted;
  let (Nx_array.Any y) = dst in
  c.y_dtype <- dtype_of dst;
  c.y_first <- V.offset v Dst * bytes c.y_dtype;
  c.y_address <- Rig.Buffer.address (Nx_array.buffer y) + c.y_first;
  strides c.sy v Dst Row Column;
  if c.init then begin
    c.i_dtype <- dtype_of ops.(2);
    c.i_first <- V.offset v Init * bytes c.i_dtype;
    strides c.si v Init Row Column
  end
  else begin
    c.i_dtype <- c.y_dtype;
    Array.blit c.sy 0 c.si 0 3
  end;
  rules c

(* Sequences *)

(* The submission's key: its contraction kernel, then its packs, split and init,
   dense over [count] kernels, so no count collides. *)
let flags = 16
let sequences = count * flags

let sequence c =
  c.kernel
  + count
    * (Bool.to_int c.a.packed
      lor (Bool.to_int c.b.packed lsl 1)
      lor (Bool.to_int (c.splits > 1) lsl 2)
      lor (Bool.to_int c.init lsl 3))

let reads c = if c.init then 3 else 2

(* The workspace is a slot of the sequence whenever a launch addresses it,
   whatever its bytes: a pack of no element takes none. *)
let uses_workspace c = c.a.packed || c.b.packed || c.splits > 1
let writes c = if uses_workspace c then 2 else 1
let workspace c = if uses_workspace c then Int.max 1 c.used else 0

(* [c]'s launches in order: each kernel, its parameter bytes and its refs into
   the slots [reads] and [writes] count. *)
let launches c =
  let y = reads c in
  let ws = y + 1 in
  let slot_ref at slot = { Rig.Submission.at; slot } in
  let pack slot =
    ( pack_kernel,
      K.Pack_params.size,
      [| slot_ref K.Pack_params.src slot; slot_ref K.Pack_params.dst ws |] )
  in
  let module P = K.Contract_params in
  let split = c.splits > 1 in
  let refs =
    List.concat
      [
        [ slot_ref P.a (if c.a.packed then ws else 0) ];
        [ slot_ref P.b (if c.b.packed then ws else 1) ];
        (if c.init then [ slot_ref P.init 2 ] else []);
        [ slot_ref P.y y ];
        (if split then [ slot_ref P.partials ws; slot_ref P.tickets ws ]
         else []);
      ]
  in
  List.concat
    [
      (if c.a.packed then [ pack 0 ] else []);
      (if c.b.packed then [ pack 1 ] else []);
      (if split then
         [
           (zero_kernel, K.Zero_params.size, [| slot_ref K.Zero_params.p ws |]);
         ]
       else []);
      [ (c.kernel, P.size, Array.of_list refs) ];
    ]

let parts c image ~queue =
  let part (kernel, params, refs) =
    let kernel = fst K.kernels.(kernel) in
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
  let n = c.batch * o.rows * o.lead in
  Run.groups run at (ceil_div n K.threads) 1 1;
  Run.threads run at K.threads 1 1;
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
  Run.int32 run at P.dtype (code o.own);
  Run.int32 run at P.out (code o.into);
  Run.int32 run at P.bytes (bytes o.into)

let write_strides run at field s =
  Run.int64 run at field s.(0);
  Run.int64 run at (field + 8) s.(1);
  Run.int64 run at (field + 16) s.(2)

(* Stores every field of each launch's struct, and test_amd_kernels.ml shows
   that kernels.ml's fields tile each struct, [zero] included: every parameter
   byte is stored before each submit, so bytes another submission left in the
   run never reach a kernel. *)
let write run sub c =
  (* Parts in [launches]' order: a's pack, b's pack, the tickets' zeroing, the
     contraction. *)
  let a = Bool.to_int c.a.packed and b = Bool.to_int c.b.packed in
  let split = c.splits > 1 in
  if c.a.packed then write_pack run (Rig.Submission.block sub 0) c c.a;
  if c.b.packed then write_pack run (Rig.Submission.block sub a) c c.b;
  if split then begin
    let at = Rig.Submission.block sub (a + b) in
    Run.groups run at (ceil_div c.tiles K.threads) 1 1;
    Run.threads run at K.threads 1 1;
    Run.shared run at 0;
    Run.int64 run at K.Zero_params.p c.tickets;
    Run.int64 run at K.Zero_params.n c.tiles
  end;
  let module P = K.Contract_params in
  let at = Rig.Submission.block sub (a + b + Bool.to_int split) in
  Run.groups run at c.blocks c.splits c.batch;
  Run.threads run at c.threads 1 1;
  Run.shared run at 0;
  Run.int64 run at P.a c.a.first;
  Run.int64 run at P.b c.b.first;
  Run.int64 run at P.init (if c.init then c.i_first else 0);
  Run.int64 run at P.y c.y_first;
  Run.int64 run at P.partials (if split then c.partials else 0);
  Run.int64 run at P.tickets (if split then c.tickets else 0);
  write_strides run at P.sa c.a.s;
  write_strides run at P.sb c.b.s;
  write_strides run at P.si c.si;
  write_strides run at P.sy c.sy;
  Run.int32 run at P.batch c.batch;
  Run.int32 run at P.m c.m;
  Run.int32 run at P.n c.n;
  Run.int32 run at P.k c.k;
  Run.int32 run at P.splits c.splits;
  Run.int32 run at P.a_dtype (code c.a.dtype);
  Run.int32 run at P.b_dtype (code c.b.dtype);
  Run.int32 run at P.init_dtype (code c.i_dtype);
  Run.int32 run at P.y_dtype (code c.y_dtype);
  Run.int32 run at P.acc_dtype (code c.acc);
  Run.int32 run at P.aligned c.aligned;
  Run.int32 run at P.zero 0
