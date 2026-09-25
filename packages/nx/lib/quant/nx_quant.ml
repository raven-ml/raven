(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t =
  | Mxfp4 of {
      codes : (int, Nx.uint8_elt) Nx.t;
      scales : (int, Nx.uint8_elt) Nx.t;
    }

let strf = Printf.sprintf

let pp_shape s =
  "[" ^ String.concat "; " (Array.to_list (Array.map string_of_int s)) ^ "]"

(* Construction. Checks read shapes only, never bytes. *)

let check_mxfp4 fn codes scales =
  let c = Nx.shape codes in
  let r = Array.length c in
  if r < 2 || c.(r - 1) mod 16 <> 0 then
    invalid_arg
      (strf
         "%s: codes must have shape [...; n; k / 2] with k a multiple of 32, \
          got %s"
         fn (pp_shape c));
  let expected = Array.copy c in
  expected.(r - 1) <- c.(r - 1) / 16;
  if Nx.shape scales <> expected then
    invalid_arg
      (strf "%s: scales must have shape %s, one per 32 values, got %s" fn
         (pp_shape expected)
         (pp_shape (Nx.shape scales)))

let mxfp4 ~scales codes =
  check_mxfp4 "Nx_quant.mxfp4" codes scales;
  Mxfp4 { codes; scales }

let place p (Mxfp4 { codes; scales }) =
  (* A group is 16 bytes of codes: every window must start and stop at one. *)
  let c = Nx.shape codes in
  let r = Array.length c in
  List.iter
    (fun d ->
      let lo, hi = (Nx.Placement.window p c d).(r - 1) in
      if lo mod 16 <> 0 || hi mod 16 <> 0 then
        invalid_arg
          (strf
             "Nx_quant.place: splitting codes and scales along axis %d in %d \
              cuts a 32-value group (%d groups)"
             (r - 1)
             (c.(r - 1) / (hi - lo))
             (Nx.shape scales).(r - 1)))
    (Nx.Placement.devices p);
  Mxfp4 { codes = Nx.place p codes; scales = Nx.place p scales }

let shape (Mxfp4 { codes; _ }) =
  let s = Array.copy (Nx.shape codes) in
  let r = Array.length s in
  s.(r - 1) <- 2 * s.(r - 1);
  s

(* Structure *)

let walk c (Mxfp4 { codes; scales }) =
  let open Nx.Ptree.Walk in
  case c "mxfp4";
  let codes = field c "codes" tensor codes in
  let scales = field c "scales" tensor scales in
  check_mxfp4 "Nx_quant.walk" codes scales;
  Mxfp4 { codes; scales }

type weight = t

module Structure = struct
  type _ t = weight

  let walk = walk
end

let ptree = Nx.Ptree.instantiate (module Structure)

(* Decoding. A byte holds two e2m1 codes, the low nibble first; a code is a sign
   bit over the magnitudes 0, 0.5, 1, 1.5, 2, 3, 4 and 6. Every value, scaled,
   is exact at float32 barring overflow, so the table of a byte's two values and
   the table of the 256 scales, whose products are rounded once, give the
   format's values. nx's exp2 is not exact at integer arguments. *)

let byte_values =
  let e2m1 = [| 0.; 0.5; 1.; 1.5; 2.; 3.; 4.; 6. |] in
  let value c = if c land 8 = 0 then e2m1.(c land 7) else -.e2m1.(c land 7) in
  Nx.create Nx.float32 [| 256; 2 |]
    (Array.init 512 (fun i ->
         let byte = i lsr 1 in
         value (if i land 1 = 0 then byte land 15 else byte lsr 4)))

let e8m0 =
  Nx.create Nx.float32 [| 256 |]
    (Array.init 256 (fun s ->
         if s = 255 then Float.nan else Float.ldexp 1.0 (s - 127)))

(* [decode codes scales] is rows [[| r; k / 2 |]] and [[| r; k / 32 |]] at
   float32, [[| r; k |]]. *)
let decode codes scales =
  let r = Nx.dim 0 codes and k = 2 * Nx.dim 1 codes in
  let indices t = Nx.cast Nx.int32 (Nx.reshape [| -1 |] (Nx.contiguous t)) in
  let values = Nx.take ~axis:0 ~indices:(indices codes) byte_values in
  let scale = Nx.take ~indices:(indices scales) e8m0 in
  Nx.reshape [| r; k |]
    (Nx.mul
       (Nx.reshape [| r; k / 32; 32 |] values)
       (Nx.reshape [| r; k / 32; 1 |] scale))

(* The eager loop decodes at most [chunk] values at a time, and at least one
   row. A chunk's dispatch costs well under one percent of its decode on the C
   backend: about 10 us against 8 ms on an M1 Max. *)
let chunk = 1 lsl 22
let rows_per_chunk k = max 1 (chunk / k)

let buffer_kind : type b. (float, b) Nx.dtype -> (float, b) Nx_buffer.kind =
  function
  | Nx.Float16 -> Nx_buffer.float16
  | Nx.Float32 -> Nx_buffer.float32
  | Nx.Float64 -> Nx_buffer.float64
  | Nx.BFloat16 -> Nx_buffer.bfloat16
  | Nx.Float8_e4m3 -> Nx_buffer.float8_e4m3
  | Nx.Float8_e5m2 -> Nx_buffer.float8_e5m2

let bytes buf =
  Nx_buffer.to_bigarray1 (Nx_buffer.reinterpret Nx_buffer.uint8 buf)

(* Matrices and chunks. [matrix lead t j] is the matrix [j] of the part [t]
   whose leading axes are [lead], a view. [chunks n k f] calls [f r0 r] on the
   chunks of rows of an [n]-row matrix. *)

let unravel dims p =
  let idx = Array.make (Array.length dims) 0 in
  let p = ref p in
  for a = Array.length dims - 1 downto 0 do
    idx.(a) <- !p mod dims.(a);
    p := !p / dims.(a)
  done;
  idx

let matrix lead t j =
  Nx.slice (Array.to_list (Array.map (fun i -> Nx.I i) (unravel lead j))) t

let chunks n k f =
  let per = rows_per_chunk k in
  let r0 = ref 0 in
  while !r0 < n do
    let r = min per (n - !r0) in
    f !r0 r;
    r0 := !r0 + r
  done

let range r0 r t = Nx.slice [ R (r0, r0 + r) ] t

(* Eager [dequant]: each matrix is decoded a chunk of rows at a time into one
   host buffer, wrapped once full. *)
let decode_all (type b) (dt : (float, b) Nx.dtype) codes scales :
    (float, b) Nx.t =
  let s = Array.copy (Nx.shape codes) in
  let r = Array.length s in
  let lead = Array.sub s 0 (r - 2) and n = s.(r - 2) and k = 2 * s.(r - 1) in
  s.(r - 1) <- k;
  let count = Array.fold_left ( * ) 1 lead in
  if count * n * k = 0 then Nx.zeros dt s
  else begin
    let kind = buffer_kind dt in
    let out = Nx_buffer.create kind (count * n * k) in
    let dst = bytes out in
    let item = Nx_buffer.kind_size_in_bytes kind in
    for j = 0 to count - 1 do
      let codes = matrix lead codes j and scales = matrix lead scales j in
      chunks n k (fun r0 r ->
          let values = decode (range r0 r codes) (range r0 r scales) in
          let src = bytes (Nx.to_buffer (Nx.cast dt values)) in
          Bigarray.Array1.blit src
            (Bigarray.Array1.sub dst
               (((j * n) + r0) * k * item)
               (Bigarray.Array1.dim src)))
    done;
    Nx.of_buffer out ~shape:s
  end

(* Batch axes, aligned on the right and broadcast as Nx.matmul's. *)

let broadcast fn a b =
  let la = Array.length a and lb = Array.length b in
  let l = max la lb in
  Array.init l (fun i ->
      let da = if i < l - la then 1 else a.(i - l + la) in
      let db = if i < l - lb then 1 else b.(i - l + lb) in
      if da = db || db = 1 then da
      else if da = 1 then db
      else
        invalid_arg
          (strf "%s: batch axes %s and %s do not broadcast" fn (pp_shape a)
             (pp_shape b)))

(* [locate dims idx] is the row-major position, in batch axes [dims], of the
   multi-index [idx] of a broadcast of [dims] on the right: an index along a
   unit axis of [dims] is 0. *)
let locate dims idx =
  let l = Array.length idx and ld = Array.length dims in
  let p = ref 0 in
  for a = 0 to ld - 1 do
    let d = dims.(a) in
    p := (!p * d) + if d = 1 then 0 else idx.(l - ld + a)
  done;
  !p

let next dims idx =
  let a = ref (Array.length dims - 1) in
  while
    !a >= 0
    &&
    (idx.(!a) <- idx.(!a) + 1;
     idx.(!a) = dims.(!a))
  do
    idx.(!a) <- 0;
    decr a
  done

let is_range a =
  let ok = ref true in
  Array.iteri (fun i v -> if v <> i then ok := false) a;
  !ok

let rows_at indices t =
  if is_range indices && Array.length indices = Nx.dim 0 t then t
  else
    Nx.take ~axis:0
      ~indices:
        (Nx.create Nx.int32
           [| Array.length indices |]
           (Array.map Int32.of_int indices))
      t

(* The shapes of a product: [w']'s batch axes and the result's shape. A
   transposed product multiplies by the weight rather than by its transpose. *)
let product_shape ~transpose ?ids ws xs =
  let fn = "Nx_quant.apply" in
  let wr = Array.length ws in
  let n = ws.(wr - 2) and k = 2 * ws.(wr - 1) in
  let inputs, outputs = if transpose then (n, k) else (k, n) in
  let xr = Array.length xs in
  if xr = 0 then invalid_arg (strf "%s: x must have at least one axis" fn);
  if xs.(xr - 1) <> inputs then
    invalid_arg
      (strf "%s: x's last axis is %d, the weight's %s are %d" fn
         xs.(xr - 1)
         (if transpose then "outputs" else "inputs")
         inputs);
  let xb = if xr = 1 then [||] else Array.sub xs 0 (xr - 2) in
  let wb =
    match ids with
    | None -> Array.sub ws 0 (wr - 2)
    | Some is ->
        let p = wr - 3 in
        if p < 0 then
          invalid_arg
            (strf "%s: ids need a weight with an expert axis, got shape %s" fn
               (pp_shape (Array.append (Array.sub ws 0 (wr - 1)) [| k |])));
        if Array.length is < p then
          invalid_arg
            (strf "%s: ids of shape %s lack the weight's %d leading axes" fn
               (pp_shape is) p);
        Array.append
          (broadcast fn (Array.sub ws 0 p) (Array.sub is 0 p))
          (Array.sub is p (Array.length is - p))
  in
  let rb = broadcast fn xb wb in
  ( wb,
    Array.concat
      [ rb; (if xr = 1 then [||] else [| xs.(xr - 2) |]); [| outputs |] ] )

(* Eager [apply]. A result's matrix instances are grouped by the matrix of the
   weight they meet. Each group is one product, computed a chunk of the matrix's
   rows at a time into its slots of one float32 buffer, and one gather puts the
   slots in instance order, reading a zero slot for an instance that meets no
   matrix. A transposed product sums its chunks' products. *)
let product_all (type b) ~transpose ?ids codes scales (x : (float, b) Nx.t) :
    (float, b) Nx.t =
  let dt = Nx.dtype x in
  let ws = Nx.shape codes and xs = Nx.shape x in
  let wb, out = product_shape ~transpose ?ids:(Option.map Nx.shape ids) ws xs in
  let wr = Array.length ws and xr = Array.length xs in
  let n = ws.(wr - 2) and k = 2 * ws.(wr - 1) in
  let inputs, outputs = if transpose then (n, k) else (k, n) in
  let xb = if xr = 1 then [||] else Array.sub xs 0 (xr - 2) in
  let m = if xr = 1 then 1 else xs.(xr - 2) in
  (* The weight's matrices as seen from [wb]: [matrix_at] is the index of the
     one at [idx] in [wb], -1 for none, among [count]. *)
  let count, matrix_at =
    match ids with
    | None -> (Array.fold_left ( * ) 1 wb, fun idx -> locate wb idx)
    | Some ids ->
        let p = wr - 3 and is = Nx.shape ids in
        let lanes = Array.sub ws 0 p and e = ws.(p) in
        let values = lazy (Nx.to_array ids) in
        let matrix_at idx =
          let id = Int32.to_int (Lazy.force values).(locate is idx) in
          if id < 0 || id >= e then -1
          else (locate lanes (Array.sub idx 0 p) * e) + id
        in
        (Array.fold_left ( * ) 1 lanes * e, matrix_at)
  in
  let rb = Array.sub out 0 (Array.length out - if xr = 1 then 1 else 2) in
  let instances = Array.fold_left ( * ) 1 rb in
  if Array.exists (( = ) 0) out || inputs = 0 then Nx.zeros dt out
  else begin
    (* Each instance's matrix and row of [x]'s batch. *)
    let members = Array.make count [] in
    let x_row = Array.make instances 0 in
    let idx = Array.make (Array.length rb) 0 in
    let lw = Array.length wb and lr = Array.length rb in
    let widx = Array.make lw 0 in
    for i = 0 to instances - 1 do
      x_row.(i) <- locate xb idx;
      for a = 0 to lw - 1 do
        widx.(a) <- (if wb.(a) = 1 then 0 else idx.(lr - lw + a))
      done;
      let j = matrix_at widx in
      if j >= 0 then members.(j) <- i :: members.(j);
      next rb idx
    done;
    (* Slots: each instance's group position, [filled] for none. *)
    let slot = Array.make instances (-1) and filled = ref 0 in
    Array.iter
      (fun group ->
        List.iteri (fun g i -> slot.(i) <- !filled + g) (List.rev group);
        filled := !filled + List.length group)
      members;
    let slots = !filled + if Array.mem (-1) slot then 1 else 0 in
    let y = Nx_buffer.create Nx_buffer.float32 (slots * m * outputs) in
    let dst = Nx_buffer.to_bigarray1 y in
    if slots > !filled then
      Bigarray.Array1.fill
        (Bigarray.Array1.sub dst (!filled * m * outputs) (m * outputs))
        0.0;
    let x =
      Nx.reshape
        [| Array.fold_left ( * ) 1 xb; m; inputs |]
        (Nx.contiguous (Nx.cast Nx.float32 x))
    in
    let lead = Array.sub ws 0 (wr - 2) in
    let base = ref 0 in
    Array.iteri
      (fun j group ->
        if group <> [] then begin
          let group = Array.of_list (List.rev group) in
          let g = Array.length group in
          let rows = rows_at (Array.map (fun i -> x_row.(i)) group) x in
          let codes = matrix lead codes j and scales = matrix lead scales j in
          let decoded r0 r = decode (range r0 r codes) (range r0 r scales) in
          if transpose then begin
            let sum = ref None in
            chunks n k (fun r0 r ->
                let p =
                  Nx.matmul
                    (Nx.slice [ A; A; R (r0, r0 + r) ] rows)
                    (decoded r0 r)
                in
                sum := Some (match !sum with None -> p | Some s -> Nx.add s p));
            let src = Nx_buffer.to_bigarray1 (Nx.to_buffer (Option.get !sum)) in
            Bigarray.Array1.blit src
              (Bigarray.Array1.sub dst (!base * m * k) (g * m * k))
          end
          else
            chunks n k (fun r0 r ->
                let p = Nx.matmul rows (Nx.matrix_transpose (decoded r0 r)) in
                let src = Nx_buffer.to_bigarray1 (Nx.to_buffer p) in
                for q = 0 to (g * m) - 1 do
                  Bigarray.Array1.blit
                    (Bigarray.Array1.sub src (q * r) r)
                    (Bigarray.Array1.sub dst ((((!base * m) + q) * n) + r0) r)
                done);
          base := !base + g
        end)
      members;
    let slot = Array.map (fun s -> if s < 0 then !filled else s) slot in
    let y = Nx.of_buffer y ~shape:[| slots; m; outputs |] in
    Nx.cast dt (Nx.reshape out (rows_at slot y))
  end

(* The effect *)

module Effect = struct
  type (_, _) op =
    | Apply : {
        ids : (int32, Nx.int32_elt) Nx.t option;
        x : (float, 'b) Nx.t;
        transpose : bool;
      }
        -> (float, 'b) op
    | Dequant : (float, 'b) Nx.dtype -> (float, 'b) op

  type _ Stdlib.Effect.t +=
    | E_quant : { w : t; op : ('a, 'b) op } -> ('a, 'b) Nx.t Stdlib.Effect.t

  let perform : type a b. t -> (a, b) op -> (a, b) Nx.t =
   fun (Mxfp4 { codes; scales } as w) op ->
    match op with
    | Apply { ids; x; transpose } -> (
        ignore
          (product_shape ~transpose ?ids:(Option.map Nx.shape ids)
             (Nx.shape codes) (Nx.shape x));
        try Stdlib.Effect.perform (E_quant { w; op })
        with Stdlib.Effect.Unhandled _ ->
          product_all ~transpose ?ids codes scales x)
    | Dequant dt -> (
        try Stdlib.Effect.perform (E_quant { w; op })
        with Stdlib.Effect.Unhandled _ -> decode_all dt codes scales)
end

(* Products *)

let dequant dt w = Effect.perform w (Effect.Dequant dt)

let apply ?ids w x =
  Effect.perform w (Effect.Apply { ids; x; transpose = false })
