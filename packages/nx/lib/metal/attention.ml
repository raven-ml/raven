(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module P = Nx_kernel.Prog
module K = Kernels

let ceil_div a b = (a + b - 1) / b

(* The program

   Output 0 is no mask where it is the constant [true], a bool load read in
   place, or a temporary that Loop maps, from the stored scores where it reads
   the score. Output 1 is the score; the score added to, subtracted from or
   multiplied by a node that does not read it, a float32 load or a temporary;
   or, for any other program, a temporary Loop maps from the scores a first
   launch stores. Each temporary holds its node's values exactly, so the bits
   are the program's. *)

type source = Load of int | Temp
type mask = No_mask | Mask of source * int
type score = Plain | Op of int * source * int | Stored

let reads_score (p : P.t) root =
  let seen = Array.make (Iarray.length p.nodes) false in
  let rec go i =
    if seen.(i) then false
    else begin
      seen.(i) <- true;
      match Iarray.get p.nodes i with
      | P.In 0 -> true
      | P.In _ | Coord _ | Const _ -> false
      | Op1 (_, _, a) -> go a
      | Op2 (_, a, b) -> go a || go b
      | Op3 (_, a, b, c) -> go a || go b || go c
    end
  in
  go root

let source (p : P.t) i =
  match Iarray.get p.nodes i with P.In j -> Load (j - 1) | _ -> Temp

let mask_of (p : P.t) =
  let o0 = Iarray.get p.outs 0 in
  match Iarray.get p.nodes o0 with
  | P.Const (_, bits) when bits <> "\000" -> No_mask
  | _ -> Mask (source p o0, o0)

let keeps_all (p : P.t) =
  match Iarray.get p.nodes (Iarray.get p.outs 2) with
  | P.Const (_, bits) -> bits <> "\000"
  | _ -> false

let is_score (p : P.t) i =
  match Iarray.get p.nodes i with P.In 0 -> true | _ -> false

let score_of (p : P.t) =
  let o1 = Iarray.get p.outs 1 in
  let with_x flag x = Op (flag, source p x, x) in
  match Iarray.get p.nodes o1 with
  | P.In 0 -> Plain
  | Op2 (Binary op, a, b) when is_score p a && not (reads_score p b) -> (
      match op with
      | Add -> with_x K.att_add b
      | Sub -> with_x K.att_sub b
      | Mul -> with_x K.att_mul b
      | _ -> Stored)
  | Op2 (Binary op, a, b) when is_score p b && not (reads_score p a) -> (
      match op with
      | Add -> with_x K.att_add a
      | Sub -> with_x K.att_rsub a
      | Mul -> with_x K.att_mul a
      | _ -> Stored)
  | _ -> Stored

(* Launches *)

(* The dims a kernel holds: the least of 64 and 128 that holds [dk], or none
   past 128, where it stages them in chunks of NX_ATT_CHUNK. *)
type dims = D64 | D128 | Wide

let dims_of dk = if dk <= 64 then D64 else if dk <= 128 then D128 else Wide

let kernel (D.Any dt) dims c =
  match (dt, dims, c) with
  | D.Float32, D64, 32 -> K.Attention_tile_f32_64_32
  | D.Float32, D64, 64 -> K.Attention_tile_f32_64_64
  | D.Float32, D64, _ -> K.Attention_tile_f32_64_128
  | D.Float32, D128, 32 -> K.Attention_tile_f32_128_32
  | D.Float32, D128, 64 -> K.Attention_tile_f32_128_64
  | D.Float32, D128, _ -> K.Attention_tile_f32_128_128
  | D.Float32, Wide, 32 -> K.Attention_tile_f32_w_32
  | D.Float32, Wide, 64 -> K.Attention_tile_f32_w_64
  | D.Float32, Wide, _ -> K.Attention_tile_f32_w_128
  | D.Float16, D64, 32 -> K.Attention_tile_f16_64_32
  | D.Float16, D64, 64 -> K.Attention_tile_f16_64_64
  | D.Float16, D64, _ -> K.Attention_tile_f16_64_128
  | D.Float16, D128, 32 -> K.Attention_tile_f16_128_32
  | D.Float16, D128, 64 -> K.Attention_tile_f16_128_64
  | D.Float16, D128, _ -> K.Attention_tile_f16_128_128
  | D.Float16, Wide, 32 -> K.Attention_tile_f16_w_32
  | D.Float16, Wide, 64 -> K.Attention_tile_f16_w_64
  | D.Float16, Wide, _ -> K.Attention_tile_f16_w_128
  | _, D64, 32 -> K.Attention_tile_bf16_64_32
  | _, D64, 64 -> K.Attention_tile_bf16_64_64
  | _, D64, _ -> K.Attention_tile_bf16_64_128
  | _, D128, 32 -> K.Attention_tile_bf16_128_32
  | _, D128, 64 -> K.Attention_tile_bf16_128_64
  | _, D128, _ -> K.Attention_tile_bf16_128_128
  | _, Wide, 32 -> K.Attention_tile_bf16_w_32
  | _, Wide, 64 -> K.Attention_tile_bf16_w_64
  | _, Wide, _ -> K.Attention_tile_bf16_w_128

(* A kernel's threadgroup bytes (kernels.h's NX_ATT_SHARED_OF), by its elements'
   bytes and the dims it stages. *)
let shared ~bytes dims =
  match (bytes, dims) with
  | 4, D64 -> K.att_shared_4_64
  | 4, _ -> K.att_shared_4_128
  | _, D64 -> K.att_shared_2_64
  | _, _ -> K.att_shared_2_128

(* The columns of a piece: the least of 32, 64 and 128 that holds [dv]. *)
let piece_width dv =
  if dv <= 32 then 32 else if dv <= 64 then 64 else K.att_piece

(* A unit's estimated counts, which err high: each of its padded rows' pairs
   takes [2 (dk + c)] operations on the fragments and 48 more for its weight,
   mask and score; its keys and values are read once, its mask and x for each
   row. *)
let unit_cost ~bytes ~tk ~dk ~c ~mask ~x =
  let blocks = ceil_div (Int.max tk 1) K.att_block in
  let pairs = blocks * K.att_block * K.att_rows in
  let dkp = 8 * ceil_div dk 8 in
  let ops = pairs * ((2 * (dkp + c)) + 48) in
  let traffic =
    (tk * (dk + c) * bytes)
    + (K.att_rows * (dk + c) * bytes)
    + (if mask then K.att_rows * tk else 0)
    + if x then 4 * K.att_rows * tk else 0
  in
  let chain = blocks * (dkp + 64 + c + 96 + (4 * Plan.read_links)) in
  { Plan.bytes = traffic + (2 * Plan.line); ops; chain }

let shape_of a = Iarray.to_array (L.shape a)

(* Where an array lies: its slot, its first element's byte, and its element
   steps along the lead axes and the axes past them. *)
type site = { slot : int; at : int; lead : int array; rest : int array }

let site_of b ~leads (o : Operands.operand) =
  let l = o.layout in
  let (D.Any dt) = o.dtype in
  let n = L.rank l in
  let s i = L.stride l i in
  {
    slot = Plan.slot b o.buffer;
    at = D.bytes dt (L.offset l);
    lead = Array.init leads s;
    rest = Array.init (n - leads) (fun i -> s (leads + i));
  }

let temp_site b ~leads (at, strides) =
  {
    slot = Plan.workspace b;
    at;
    lead = Array.sub strides 0 leads;
    rest = Array.sub strides leads (Array.length strides - leads);
  }

(* Whether [o]'s rows of [n] elements at [s] load 4 elements a vector: each row
   contiguous, [n] a multiple of 4, and every row's first element on a vector's
   bytes, [bytes] an element's. *)
let vector ~bytes (o : Operands.operand) (s : site) n =
  let r = Array.length s.rest in
  let steps = Array.append s.lead (Array.sub s.rest 0 (r - 1)) in
  n mod 4 = 0
  && s.rest.(r - 1) = 1
  && (o.residue + s.at) land ((4 * bytes) - 1) = 0
  && Array.for_all (fun x -> x mod 4 = 0) steps

(* The elements a kernel reaches past an index's over [e] indices of [s]'s axis
   [i] past the lead ones. *)
let reach (s : site) i e = Int.max 0 (e - 1) * abs s.rest.(i)

(* The span below which a block's keys and a row's elements index in 32 bits:
   2^31, or a test's bound. *)
let offsets_bound = Atomic.make (1 lsl 31)

let bound_offsets n =
  if n < 0 || n > 1 lsl 31 then invalid_arg "Nx_metal.bound_offsets";
  Atomic.set offsets_bound n;
  Plan.renew ()

let no_site ~leads =
  { slot = -1; at = 0; lead = Array.make leads 0; rest = [| 0; 0; 0; 0 |] }

(* The lead table: the lead extents, their divisors, then each array's steps
   along them, in kernels.h's order. *)
let lead_table extents (sites : site array) =
  let steps =
    Array.concat (Array.to_list (Array.map (fun s -> s.lead) sites))
  in
  let n = Array.length extents in
  let b = Bytes.make (8 * ((2 * n) + Array.length steps)) '\000' in
  let put i x = Bytes.set_int64_le b (8 * i) (Int64.of_int x) in
  Array.iteri
    (fun i e ->
      put i e;
      if e >= 1 then Bytes.set_int64_le b (8 * (n + i)) (Fft.divisor e))
    extents;
  Array.iteri (fun i x -> put ((2 * n) + i) x) steps;
  Bytes.unsafe_to_string b

let bigbytes s =
  let n = String.length s in
  let b = Bigarray.Array1.create Bigarray.int8_unsigned Bigarray.c_layout n in
  String.iteri (fun i c -> Bigarray.Array1.unsafe_set b i (Char.code c)) s;
  b

(* A unit holds [c] of a key head's [g] query heads and [r] rows of each, at
   most NX_ATT_ROWS rows. *)
let unit_shape ~g ~tq =
  let c = Int.min g (Int.max 1 (K.att_rows / Int.min tq K.att_rows)) in
  (c, Int.min tq (K.att_rows / c))

exception Too_long

(* The tile counts a unit's blocks in 32 bits: 2^32 blocks. *)
let block_range = 0x1_0000_0000

(* The launches of [units] units by [k] over [pieces] pieces of columns, each
   holding the most units whose estimate fits a launch, with words [w] but for
   the first unit. Raises [Too_long] where one unit's does not. *)
let launch_units b k ~shared ~holds ~units ~pieces ~blocks ~(cost : Plan.cost) w
    refs =
  let threads = K.threads k in
  let resident = Plan.resident ~granted:holds k ~shared in
  let cost_of m =
    {
      Plan.bytes = m * pieces * cost.bytes;
      ops = m * pieces * cost.ops;
      chain = cost.chain;
    }
  in
  let seconds m =
    Plan.seconds ~threads ~groups:(m, pieces, 1) ~resident (cost_of m)
  in
  let fits m = seconds m <= Plan.allowed () in
  (* A unit's estimate charges each of its [blocks] more than budget / 2^32, so
     a unit of 2^32 blocks passes a launch's budget and raises [Too_long] before
     the tile's counter could wrap. *)
  assert (seconds 1 *. float block_range > float blocks *. Plan.budget);
  let most =
    match Plan.most ~fits (Int.min units Plan.most_groups) with
    | Some m -> m
    | None -> raise Too_long
  in
  let first = ref 0 in
  while !first < units do
    let m = Int.min most (units - !first) in
    let w = Bytes.copy w in
    Plan.set w K.att_first_unit !first;
    Plan.launch b k ~groups:(m, pieces, 1) ~shared ~resident ~cost:(cost_of m) w
      refs;
    first := !first + m
  done

let plan ~holds (p : P.t) ~scale ~paged ~lengths (xs : Operands.t) =
  let q = xs.loads.(0) and k = xs.loads.(1) and v = xs.loads.(2) in
  let qs = shape_of q.layout in
  let leads = Array.length qs - 3 in
  let lead = Array.sub qs 0 leads in
  let nb = Array.fold_left ( * ) 1 lead in
  let hq = qs.(leads) and tq = qs.(leads + 1) and dk = qs.(leads + 2) in
  let ks = shape_of k.layout and vs = shape_of v.layout in
  let lengths_at = if paged then 4 else 3 in
  let first_load = lengths_at + if lengths then 1 else 0 in
  let hk, tk, page, pages, dv =
    if paged then
      let ts = shape_of xs.loads.(3).layout in
      (ks.(1), ts.(leads) * ks.(2), ks.(2), ks.(0), vs.(3))
    else (ks.(leads), ks.(leads + 1), 0, 0, vs.(leads + 2))
  in
  if nb = 0 || hq = 0 || tq = 0 then Ok None
  else
    let g = hq / hk in
    let c, r = unit_shape ~g ~tq in
    let units = nb * hk * ceil_div g c * ceil_div tq r in
    let width = piece_width dv in
    let pieces = Int.max 1 (ceil_div dv width) in
    let dims = dims_of dk in
    let kern = kernel q.dtype dims width in
    let mask = if tk = 0 then No_mask else mask_of p in
    let score = if tk = 0 then Plain else score_of p in
    let stored =
      score = Stored
      ||
      match mask with
      | Mask (Temp, o0) -> reads_score p o0
      | Mask (Load _, _) | No_mask -> false
    in
    let roots =
      (match mask with
        | Mask (Temp, o0) -> [ o0 ]
        | Mask (Load _, _) | No_mask -> [])
      @
      match score with
      | Op (_, Temp, x) -> [ x ]
      | Stored -> [ Iarray.get p.outs 1 ]
      | Op (_, Load _, _) | Plain -> []
    in
    let shape = Array.append lead [| hq; tq; tk |] in
    let loads =
      Array.sub xs.loads first_load (Array.length xs.loads - first_load)
    in
    let temps =
      match roots with
      | [] -> Ok None
      | _ ->
          Result.map Option.some
            (Loop.temps ~holds p ~shape ~score:stored loads roots)
    in
    match temps with
    | Error e -> Error e
    | Ok temps -> (
        let b = Plan.make () in
        let site o = site_of b ~leads o in
        let pool o =
          let s = site_of b ~leads:0 o in
          { s with lead = Array.make leads 0 }
        in
        let qsite = site q
        and osite = site xs.dsts.(0)
        and lsite = site xs.dsts.(1) in
        let ksite = if paged then pool k else site k in
        let vsite = if paged then pool v else site v in
        let tsite = if paged then site xs.loads.(3) else no_site ~leads in
        let nsite =
          if lengths then site xs.loads.(lengths_at) else no_site ~leads
        in
        let (D.Any qdt) = q.dtype in
        let bytes = D.bytes qdt 1 in
        let vec flag o s n = if vector ~bytes o s n then flag else 0 in
        (* Whether a row's [d] elements lie within [below] of its first, and a
           dense block's keys within [below] of its first key. *)
        let below = Atomic.get offsets_bound in
        let fits (s : site) d =
          let r = Array.length s.rest in
          reach s (r - 1) d < below
          && (paged || reach s (r - 2) (Int.min tk K.att_block) < below)
        in
        let flags0 =
          (if paged then K.att_paged else 0)
          lor (if lengths then K.att_lengths else 0)
          lor (if xs.dsts.(0).dtype = D.Any D.Float32 then K.att_o32 else 0)
          lor vec K.att_qvec q qsite dk lor vec K.att_kvec k ksite dk
          lor vec K.att_vvec v vsite dv
          lor if fits ksite dk && fits vsite dv then K.att_narrow else 0
        in
        let scale_bits =
          Int32.to_int (Int32.bits_of_float scale) land 0xffff_ffff
        in
        (* The words and refs of a launch over [sites] of the mask and x,
           storing the scores at [scores] where given. *)
        let words ~flags ~msite ~xsite ~scores =
          let w = Plan.words kern in
          let refs = ref [] in
          let bind word (s : site) ~write =
            if s.slot >= 0 then begin
              Plan.set w word s.at;
              refs := { Plan.word; slot = s.slot; write } :: !refs
            end
          in
          bind K.att_q qsite ~write:false;
          bind K.att_k ksite ~write:false;
          bind K.att_v vsite ~write:false;
          bind K.att_o osite ~write:true;
          bind K.att_lse lsite ~write:true;
          bind K.att_lengths_buf nsite ~write:false;
          bind K.att_table tsite ~write:false;
          bind K.att_mask_buf msite ~write:false;
          bind K.att_x_buf xsite ~write:false;
          (match scores with
          | Some at ->
              bind K.att_scores_buf
                { slot = Plan.workspace b; at; lead = [||]; rest = [||] }
                ~write:true
          | None -> ());
          if leads > 0 then begin
            let t =
              lead_table lead
                [|
                  qsite; ksite; vsite; osite; lsite; nsite; tsite; msite; xsite;
                |]
            in
            let slot =
              Plan.constant b ~key:("attention " ^ t) ~length:(String.length t)
                (fun () -> bigbytes t)
            in
            refs := { Plan.word = K.att_lead; slot; write = false } :: !refs
          end;
          let set = Plan.set w in
          set K.att_flags flags;
          set K.att_scale scale_bits;
          set K.att_dk dk;
          set K.att_dv dv;
          set K.att_tq tq;
          set K.att_tk tk;
          set K.att_hq hq;
          set K.att_hk hk;
          set K.att_unit_heads c;
          set K.att_unit_rows r;
          set K.att_page page;
          set K.att_pages pages;
          set K.att_leads leads;
          set K.att_nb nb;
          set K.att_g g;
          set K.att_tiles_n (ceil_div tq r);
          set K.att_chunks_n (ceil_div g c);
          (* nx_divide's divisors; dense keys' page of 0 divides nothing. *)
          let set_div i d =
            if d >= 1 then Bytes.set_int64_le w (8 * i) (Fft.divisor d)
          in
          set_div K.att_tiles_div (ceil_div tq r);
          set_div K.att_chunks_div (ceil_div g c);
          set_div K.att_hk_div hk;
          set K.att_rows_div (Plan.divisor r);
          set_div K.att_page_div page;
          let rest (s : site) i =
            if i < Array.length s.rest then s.rest.(i) else 0
          in
          set K.att_q_h (rest qsite 0);
          set K.att_q_t (rest qsite 1);
          set K.att_q_d (rest qsite 2);
          let kv (s : site) ~h ~t ~d ~p =
            if paged then begin
              set p (rest s 0);
              set h (rest s 1);
              set t (rest s 2);
              set d (rest s 3)
            end
            else begin
              set p 0;
              set h (rest s 0);
              set t (rest s 1);
              set d (rest s 2)
            end
          in
          kv ksite ~h:K.att_k_h ~t:K.att_k_t ~d:K.att_k_d ~p:K.att_k_p;
          kv vsite ~h:K.att_v_h ~t:K.att_v_t ~d:K.att_v_d ~p:K.att_v_p;
          set K.att_o_h (rest osite 0);
          set K.att_o_t (rest osite 1);
          set K.att_o_d (rest osite 2);
          set K.att_lse_h (rest lsite 0);
          set K.att_lse_t (rest lsite 1);
          set K.att_tab_p (rest tsite 0);
          set K.att_mask_h (rest msite 0);
          set K.att_mask_t (rest msite 1);
          set K.att_mask_k (rest msite 2);
          set K.att_x_h (rest xsite 0);
          set K.att_x_t (rest xsite 1);
          set K.att_x_k (rest xsite 2);
          (w, Array.of_list !refs)
        in
        let cost ~mask ~x = unit_cost ~bytes ~tk ~dk ~c:width ~mask ~x in
        let blocks = ceil_div (Int.max tk 1) K.att_block in
        let shared = shared ~bytes dims in
        let load j = site xs.loads.(first_load + j) in
        try
          let score_at =
            match temps with
            | Some t when stored -> Some (Loop.score_at b t)
            | Some _ | None -> None
          in
          (match score_at with
          | Some at ->
              let w, refs =
                words ~flags:(flags0 lor K.att_scores) ~msite:(no_site ~leads)
                  ~xsite:(no_site ~leads) ~scores:(Some at)
              in
              launch_units b kern ~shared ~holds ~units ~pieces:1 ~blocks
                ~cost:(cost ~mask:false ~x:false)
                w refs
          | None -> ());
          let placed =
            match temps with
            | Some t -> Loop.emit ?score:score_at b t
            | None -> [||]
          in
          let next = ref 0 in
          let temp () =
            let s = temp_site b ~leads placed.(!next) in
            incr next;
            s
          in
          let msite, mflag =
            match mask with
            | No_mask -> (no_site ~leads, 0)
            | Mask (Load j, _) -> (load j, K.att_mask)
            | Mask (Temp, _) -> (temp (), K.att_mask)
          in
          let xsite, xflag =
            match score with
            | Plain -> (no_site ~leads, 0)
            | Op (flag, Load j, _) -> (load j, flag)
            | Op (flag, Temp, _) -> (temp (), flag)
            | Stored -> (temp (), K.att_x)
          in
          let w, refs =
            words ~flags:(flags0 lor mflag lor xflag) ~msite ~xsite ~scores:None
          in
          launch_units b kern ~shared ~holds ~units ~pieces ~blocks
            ~cost:(cost ~mask:(mflag <> 0) ~x:(xflag <> 0))
            w refs;
          Ok (Plan.finish_some b)
        with Too_long -> Error A.Declined)
