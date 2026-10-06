(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module B = Nx_device.Buffer
module A = Bigarray.Array1

type bytes = (int, Nx.uint8_elt) Nx_ragged.t

(* Every helper takes its bytes at this type, so that the compiler reads each
   byte inline rather than through the runtime's generic accessor. *)
type buf = (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) A.t
type int64s = (int64, Bigarray.int64_elt, Bigarray.c_layout) A.t

(* [reading ~by x f] is [f b] for [b] a host buffer of [x]'s elements in C
   order, under a read claim so that no compiled call lends its memory while [f]
   reads it. *)
let reading ~by x f =
  let b = Nx.Op.eval (Read { by; x }) in
  B.Claim.read b;
  Fun.protect ~finally:(fun () -> B.Claim.release b) (fun () -> f b)

(* Host bytes *)

(* [host ~by ?mask r f] is [f o v m] for [o] the offsets of [r], [v] its bytes
   and [m] the bytes of the bit mask [mask], read on the host from its first
   element: row [i]'s bit is bit [i mod 8] of byte [i / 8]. *)
let host ~by ?mask r f =
  let read x f = reading ~by x f in
  read (Nx_ragged.offsets r) @@ fun o ->
  read (Nx_ragged.values r) @@ fun v ->
  let o : int64s = B.bigarray Bigarray.int64 o
  and v : buf = B.bigarray Bigarray.int8_unsigned v in
  match mask with
  | None -> f o v None
  | Some mask ->
      read mask @@ fun m -> f o v (Some (B.bigarray Bigarray.int8_unsigned m))

(* [rows ~by ?mask r f] calls [f i v first stop] on each row [i] of [r] that
   [mask] holds, [v] the bytes and [\[first, stop)] the row's. *)
let rows ~by ?mask r f =
  host ~by ?mask r @@ fun o v m ->
  let row i =
    f i v
      (Int64.to_int (A.unsafe_get o i))
      (Int64.to_int (A.unsafe_get o (i + 1)))
  in
  match m with
  | None ->
      for i = 0 to Nx_ragged.length r - 1 do
        row i
      done
  | Some (m : buf) ->
      for i = 0 to Nx_ragged.length r - 1 do
        if (A.unsafe_get m (i lsr 3) lsr (i land 7)) land 1 <> 0 then row i
      done

(* UTF-8, in C ([talon_strings.c]): validation runs over every byte of a column,
   as Parquet pages and parsed text are built. *)

(* [utf_8_invalid v i stop] is the first byte of [i, stop) that starts no valid
   sequence, or [-1]. *)
external utf_8_invalid :
  buf -> (int[@untagged]) -> (int[@untagged]) -> (int[@untagged])
  = "talon_utf_8_invalid_byte" "talon_utf_8_invalid"
[@@noalloc]

(* [utf_8_row v o m n] is the first of the [n] rows of [v], cut by [o], that is
   not valid UTF-8, or [-1]. A row whose bit in [m] is clear is not read. *)
external utf_8_row :
  buf -> int64s -> buf option -> (int[@untagged]) -> (int[@untagged])
  = "talon_utf_8_row_byte" "talon_utf_8_row"
[@@noalloc]

let utf_8 ~by ?mask r =
  host ~by ?mask r @@ fun o v m ->
  match utf_8_row v o m (Nx_ragged.length r) with
  | -1 -> None
  | i ->
      let first = Int64.to_int (A.unsafe_get o i)
      and stop = Int64.to_int (A.unsafe_get o (i + 1)) in
      let j = utf_8_invalid v first stop in
      Some (i, Printf.sprintf "invalid UTF-8 at byte %d" (j - first))

(* Scalar values *)

let tensor a = Nx.of_bigarray (Bigarray.genarray_of_array1 a)

(* [blit src lo dst at len] copies the bytes [lo, lo + len) of [src] to [dst]
   from [at]. A loop: [A.blit] of [A.sub]s allocates two proxies per row. *)
let blit (src : buf) lo (dst : buf) at len =
  for k = 0 to len - 1 do
    A.unsafe_set dst (at + k) (A.unsafe_get src (lo + k))
  done

let starts_scalar (v : buf) j = A.unsafe_get v j land 0xc0 <> 0x80

(* [count v i stop] is the number of scalar values in the bytes [i, stop). *)
let count (v : buf) i stop =
  let n = ref 0 in
  for j = i to stop - 1 do
    if starts_scalar v j then incr n
  done;
  !n

(* [skip v i stop k] is the byte at which scalar value [k] of [i, stop) starts,
   or [stop] past the last. *)
let rec skip (v : buf) i stop k =
  if i >= stop || (k = 0 && starts_scalar v i) then i
  else skip v (i + 1) stop (if starts_scalar v i then k - 1 else k)

let length ~by ?mask r =
  let ns = A.create Bigarray.int64 Bigarray.c_layout (Nx_ragged.length r) in
  A.fill ns 0L;
  rows ~by ?mask r (fun i v first stop ->
      A.unsafe_set ns i (Int64.of_int (count v first stop)));
  tensor ns

let slice ~by ?mask ~offset ~length r =
  let n = Nx_ragged.length r in
  let lo = Array.make n 0 and hi = Array.make n 0 in
  rows ~by ?mask r (fun i v first stop ->
      let p = if offset >= 0 then offset else count v first stop + offset in
      let p1 = if p > max_int - length then max_int else p + length in
      let p0 = Int.max p 0 and p1 = Int.max p1 0 in
      lo.(i) <- skip v first stop p0;
      hi.(i) <- skip v lo.(i) stop (Int.max 0 (p1 - p0)));
  let offsets = A.create Bigarray.int64 Bigarray.c_layout (n + 1) in
  A.unsafe_set offsets 0 0L;
  for i = 0 to n - 1 do
    A.unsafe_set offsets (i + 1)
      (Int64.add (A.unsafe_get offsets i) (Int64.of_int (hi.(i) - lo.(i))))
  done;
  let bytes =
    A.create Bigarray.int8_unsigned Bigarray.c_layout
      (Int64.to_int (A.unsafe_get offsets n))
  in
  reading ~by (Nx_ragged.values r) (fun v ->
      let v = B.bigarray Bigarray.int8_unsigned v in
      for i = 0 to n - 1 do
        let at = Int64.to_int (A.unsafe_get offsets i) in
        blit v lo.(i) bytes at (hi.(i) - lo.(i))
      done);
  Nx_ragged.v ~offsets:(tensor offsets) (tensor bytes)

type pattern =
  | Literal of string
  | Prefix of string
  | Suffix of string
  | Pieces of string list

(* [at v j s] is [true] iff the bytes of [s] start at byte [j]. *)
let at (v : buf) j s =
  let n = String.length s in
  let rec loop k =
    k = n || (A.unsafe_get v (j + k) = Char.code s.[k] && loop (k + 1))
  in
  loop 0

(* [find v i stop s] is the first byte of [i, stop) at which [s] lies whole, or
   [-1]. *)
external find :
  buf -> (int[@untagged]) -> (int[@untagged]) -> string -> (int[@untagged])
  = "talon_find_byte" "talon_find"
[@@noalloc]

let matches ~by ?mask p r =
  let n = Nx_ragged.length r in
  let hits = A.create Bigarray.int8_unsigned Bigarray.c_layout ((n + 7) / 8) in
  A.fill hits 0;
  let rec pieces v i stop = function
    | [] -> true
    | s :: ss ->
        let j = find v i stop s in
        j >= 0 && pieces v (j + String.length s) stop ss
  in
  let matches v first stop =
    match p with
    | Literal s -> find v first stop s >= 0
    | Prefix s -> stop - first >= String.length s && at v first s
    | Suffix s ->
        let j = stop - String.length s in
        j >= first && at v j s
    | Pieces ss -> pieces v first stop ss
  in
  let hit i =
    A.unsafe_set hits (i lsr 3)
      (A.unsafe_get hits (i lsr 3) lor (1 lsl (i land 7)))
  in
  rows ~by ?mask r (fun i v first stop -> if matches v first stop then hit i);
  Nx.slice
    [ Nx.R (0, n) ]
    (Nx.reshape [| -1 |] (Nx.bitcast Nx.bit (tensor hits)))

(* Literals

   A literal of valid UTF-8 found in valid UTF-8 starts and ends on scalar
   boundaries, so a byte search finds exactly its scalar matches. Matches are
   left to right, without overlap. *)

(* [occurrences v i stop s] is the number of matches of [s] in [i, stop). *)
let occurrences (v : buf) i stop s =
  let rec go i k =
    match find v i stop s with -1 -> k | j -> go (j + String.length s) (k + 1)
  in
  go i 0

(* [cumulative n f] is the [n + 1] offsets of rows of sizes [f 0] to [f (n -
   1)]. *)
let cumulative n f =
  let o = A.create Bigarray.int64 Bigarray.c_layout (n + 1) in
  A.unsafe_set o 0 0L;
  for i = 0 to n - 1 do
    A.unsafe_set o (i + 1) (Int64.add (A.unsafe_get o i) (Int64.of_int (f i)))
  done;
  o

let last (o : int64s) = Int64.to_int (A.unsafe_get o (A.dim o - 1))

let split ~by ?mask sep r =
  let n = Nx_ragged.length r and m = String.length sep in
  let matches = Array.make n (-1) and kept = Array.make n 0 in
  rows ~by ?mask r (fun i v first stop ->
      let k = occurrences v first stop sep in
      matches.(i) <- k;
      kept.(i) <- stop - first - (k * m));
  let lists = cumulative n (fun i -> matches.(i) + 1) in
  let pieces = A.create Bigarray.int64 Bigarray.c_layout (last lists + 1) in
  let bytes =
    A.create Bigarray.int8_unsigned Bigarray.c_layout
      (Array.fold_left ( + ) 0 kept)
  in
  A.unsafe_set pieces 0 0L;
  let piece = ref 0 and at = ref 0 in
  let emit v lo hi =
    blit v lo bytes !at (hi - lo);
    at := !at + hi - lo;
    incr piece;
    A.unsafe_set pieces !piece (Int64.of_int !at)
  in
  rows ~by ?mask r (fun _ v first stop ->
      let rec go i =
        match find v i stop sep with
        | -1 -> emit v i stop
        | j ->
            emit v i j;
            go (j + m)
      in
      go first);
  (tensor lists, Nx_ragged.v ~offsets:(tensor pieces) (tensor bytes))

let replace ~by ?mask ~sub ~into r =
  let n = Nx_ragged.length r
  and grow = String.length into - String.length sub in
  let sizes = Array.make n 0 in
  rows ~by ?mask r (fun i v first stop ->
      sizes.(i) <- stop - first + (grow * occurrences v first stop sub));
  let offsets = cumulative n (Array.get sizes) in
  let bytes =
    A.create Bigarray.int8_unsigned Bigarray.c_layout (last offsets)
  in
  let put = ref 0 in
  let copy v lo hi =
    blit v lo bytes !put (hi - lo);
    put := !put + hi - lo
  in
  rows ~by ?mask r (fun i v first stop ->
      put := Int64.to_int (A.unsafe_get offsets i);
      let rec go j =
        match find v j stop sub with
        | -1 -> copy v j stop
        | h ->
            copy v j h;
            String.iteri
              (fun k c -> A.unsafe_set bytes (!put + k) (Char.code c))
              into;
            put := !put + String.length into;
            go (h + String.length sub)
      in
      go first);
  Nx_ragged.v ~offsets:(tensor offsets) (tensor bytes)

(* [compare_rows v o n s signs] writes to [signs] -1, 0 or 1 where each of the
   [n] rows of [v], cut by [o], orders before, as or after [s]. *)
external compare_rows :
  buf ->
  int64s ->
  (int[@untagged]) ->
  string ->
  (int, Bigarray.int8_signed_elt, Bigarray.c_layout) A.t ->
  unit = "talon_compare_byte" "talon_compare"
[@@noalloc]

let compare ~by r one =
  let s = ref "" in
  rows ~by one (fun _ v first stop ->
      s :=
        String.init (stop - first) (fun k ->
            Char.unsafe_chr (A.unsafe_get v (first + k))));
  let n = Nx_ragged.length r in
  let signs = A.create Bigarray.int8_signed Bigarray.c_layout n in
  host ~by r (fun o v _ -> compare_rows v o n !s signs);
  tensor signs
