(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module A = Bigarray.Array1

type bytes = (int, Nx.int8_unsigned_elt, Nx.host) Nx_ragged.t

(* Every helper takes its bytes at this type, so that the compiler reads each
   byte inline rather than through the runtime's generic accessor. *)
type buf = (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) A.t
type int64s = (int64, Bigarray.int64_elt, Bigarray.c_layout) A.t

(* [array_bytes a k f] is [f b], [b] the bytes of the host array [a]'s elements
   in C order as elements of [k]: a view of its buffer where its elements lie
   there in order from a byte that [k] aligns, a copy's otherwise. *)
let array_bytes a k f =
  let module L = Nx_array.Layout in
  let dt = Nx_array.dtype a and l = Nx_array.layout a in
  let size = Bigarray.kind_size_in_bytes k in
  let n = Nx_array.Dtype.bytes dt (L.numel l) / size in
  let first = L.offset l * Nx_array.Dtype.bits dt in
  if L.is_contiguous l && first mod (8 * size) = 0 then
    Rig.Buffer.read (Nx_array.buffer a) k (fun b ->
        f (A.sub b (first / (8 * size)) n))
  else
    Rig.Buffer.read
      (Nx_array.buffer (Nx_array.copy a))
      k
      (fun b -> f (A.sub b 0 n))

(* A value nx cannot read, a traced one, fails naming [by]; [f]'s own failures
   pass as they are. *)
let reading ~by x k f =
  let entered = ref false in
  try
    Nx.Repr.read x (fun a ->
        entered := true;
        array_bytes a k f)
  with Invalid_argument m when not !entered -> invalid_arg (by ^ ": " ^ m)

(* Host arrays

   The arrays a kernel makes are host arrays of nx.array ({!Nx_array.host}):
   rig's host memory, which paces the collector by the program's memory and is
   reused once collected. A kernel fills them under write claims and gives
   them to the values it returns, with no copy. *)

let fresh dt n = Nx_array.host dt [| n |]
let fill a f = Nx_array.write a (fun g -> f (Bigarray.array1_of_genarray g))
let value a = Nx.Repr.unsafe_of_array Nx.Host.v a

(* [peek a f] is [f b], [b] the elements of the host vector [a], under a
   read claim. *)
let peek a f = Nx_array.read a (fun g -> f (Bigarray.array1_of_genarray g))

(* Host bytes *)

(* [host ~by ?mask r f] is [f o v m] for [o] the offsets of [r], [v] its bytes
   and [m] the bytes of the bit mask [mask], read on the host from its first
   element: row [i]'s bit is bit [i mod 8] of byte [i / 8]. *)
let host ~by ?mask r f =
  reading ~by (Nx_ragged.offsets r) Bigarray.int64 @@ fun (o : int64s) ->
  reading ~by (Nx_ragged.values r) Bigarray.int8_unsigned @@ fun (v : buf) ->
  match mask with
  | None -> f o v None
  | Some mask ->
      reading ~by mask Bigarray.int8_unsigned @@ fun (m : buf) -> f o v (Some m)

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

let last (o : int64s) = Int64.to_int (A.unsafe_get o (A.dim o - 1))

(* [cumulative n f] is the host array of the [n + 1] offsets of rows of sizes
   [f 0] to [f (n - 1)], and the last. *)
let cumulative n f =
  let a = fresh Nx.int64 (n + 1) in
  let total =
    fill a @@ fun (o : int64s) ->
    A.unsafe_set o 0 0L;
    for i = 0 to n - 1 do
      A.unsafe_set o (i + 1) (Int64.add (A.unsafe_get o i) (Int64.of_int (f i)))
    done;
    last o
  in
  (a, total)

(* [ranges ~by r lo hi] is the bytes [\[lo.(i), hi.(i))] of each row [i] of
   [r]. *)
let ranges ~by r lo hi =
  let n = Nx_ragged.length r in
  let offsets, size = cumulative n (fun i -> hi.(i) - lo.(i)) in
  let bytes = fresh Nx.uint8 size in
  (fill bytes @@ fun (b : buf) ->
   peek offsets @@ fun (o : int64s) ->
   reading ~by (Nx_ragged.values r) Bigarray.int8_unsigned @@ fun (v : buf) ->
   for i = 0 to n - 1 do
     let at = Int64.to_int (A.unsafe_get o i) in
     blit v lo.(i) b at (hi.(i) - lo.(i))
   done);
  Nx_ragged.v ~offsets:(value offsets) (value bytes)

let length ~by ?mask r =
  let a = fresh Nx.int64 (Nx_ragged.length r) in
  (fill a @@ fun (ns : int64s) ->
   A.fill ns 0L;
   rows ~by ?mask r (fun i v first stop ->
       A.unsafe_set ns i (Int64.of_int (count v first stop))));
  value a

let slice ~by ?mask ~offset ~length r =
  let n = Nx_ragged.length r in
  let lo = Array.make n 0 and hi = Array.make n 0 in
  rows ~by ?mask r (fun i v first stop ->
      let p = if offset >= 0 then offset else count v first stop + offset in
      let p1 = if p > max_int - length then max_int else p + length in
      let p0 = Int.max p 0 and p1 = Int.max p1 0 in
      lo.(i) <- skip v first stop p0;
      hi.(i) <- skip v lo.(i) stop (Int.max 0 (p1 - p0)));
  ranges ~by r lo hi

(* White space *)

(* [is_white_space u] is [true] iff the scalar value [u] has Unicode's
   White_Space property: one test per line of PropList.txt, Unicode 18.0.0, 25
   scalar values in all. Each is one byte of UTF-8 (U+0009 to U+000D and
   U+0020), two (U+0085 and U+00A0) or three. *)
let is_white_space u =
  (u >= 0x0009 && u <= 0x000D)
  || u = 0x0020 || u = 0x0085 || u = 0x00A0 || u = 0x1680
  || (u >= 0x2000 && u <= 0x200A)
  || u = 0x2028 || u = 0x2029 || u = 0x202F || u = 0x205F || u = 0x3000

(* [white_at v j stop] is the length of the UTF-8 sequence of a white space
   scalar value that starts at byte [j] and ends by [stop], or [0]. A byte that
   starts no sequence of at most three bytes reads as no white space. *)
let white_at (v : buf) j stop =
  let b0 = A.unsafe_get v j in
  if b0 < 0x80 then if is_white_space b0 then 1 else 0
  else if b0 land 0xe0 = 0xc0 && j + 1 < stop then
    let b1 = A.unsafe_get v (j + 1) land 0x3f in
    if is_white_space (((b0 land 0x1f) lsl 6) lor b1) then 2 else 0
  else if b0 land 0xf0 = 0xe0 && j + 2 < stop then
    let b1 = A.unsafe_get v (j + 1) land 0x3f in
    let b2 = A.unsafe_get v (j + 2) land 0x3f in
    if is_white_space (((b0 land 0x0f) lsl 12) lor (b1 lsl 6) lor b2) then 3
    else 0
  else 0

(* [skip_white v j stop] is the first byte from [j] that starts no white space
   before [stop]. *)
let rec skip_white v j stop =
  let w = if j < stop then white_at v j stop else 0 in
  if w = 0 then j else skip_white v (j + w) stop

(* [back_white v lo j] is the byte after the last scalar value of [\[lo, j)]
   that is not white space, or [lo]. *)
let rec back_white v lo j =
  if j <= lo then lo
  else
    let k = ref (j - 1) in
    while !k > lo && not (starts_scalar v !k) do
      decr k
    done;
    let w = white_at v !k j in
    if w > 0 && !k + w = j then back_white v lo !k else j

let trim ~by ?mask r =
  let n = Nx_ragged.length r in
  let lo = Array.make n 0 and hi = Array.make n 0 in
  rows ~by ?mask r (fun i v first stop ->
      let l = skip_white v first stop in
      lo.(i) <- l;
      hi.(i) <- back_white v l stop);
  ranges ~by r lo hi

(* Joining *)

(* [hosts ~by rs f] is [f os vs], [os] and [vs] the offsets and bytes of each of
   [rs], in order, read on the host. *)
let rec hosts ~by rs f =
  match rs with
  | [] -> f [] []
  | r :: rs ->
      host ~by r (fun o v _ ->
          hosts ~by rs (fun os vs -> f (o :: os) (v :: vs)))

(* [masked ~by mask f] is [f m], [m] the bytes of the bit mask [mask] if there
   is one. *)
let masked ~by mask f =
  match mask with
  | None -> f None
  | Some mask -> reading ~by mask Bigarray.int8_unsigned (fun m -> f (Some m))

let concat ~by ?mask ~length:n sep parts =
  let k = List.length parts and m = String.length sep in
  (* A part of one row is broadcast: its row 0 stands for each row. *)
  let broadcast =
    Array.of_list (List.map (fun r -> Nx_ragged.length r = 1) parts)
  in
  hosts ~by parts @@ fun os vs ->
  masked ~by mask @@ fun mask ->
  let os = Array.of_list os and vs = Array.of_list vs in
  let held i =
    match (mask : buf option) with
    | None -> true
    | Some b -> (A.unsafe_get b (i lsr 3) lsr (i land 7)) land 1 <> 0
  in
  let first j i =
    Int64.to_int (A.unsafe_get os.(j) (if broadcast.(j) then 0 else i))
  in
  let stop j i =
    Int64.to_int (A.unsafe_get os.(j) (if broadcast.(j) then 1 else i + 1))
  in
  let size i =
    if not (held i) then 0
    else begin
      let s = ref ((k - 1) * m) in
      for j = 0 to k - 1 do
        s := !s + stop j i - first j i
      done;
      !s
    end
  in
  let offsets, total = cumulative n size in
  let bytes = fresh Nx.uint8 total in
  (fill bytes @@ fun (b : buf) ->
   peek offsets @@ fun (o : int64s) ->
   for i = 0 to n - 1 do
     if held i then begin
       let at = ref (Int64.to_int (A.unsafe_get o i)) in
       for j = 0 to k - 1 do
         if j > 0 then begin
           for q = 0 to m - 1 do
             A.unsafe_set b (!at + q) (Char.code (String.unsafe_get sep q))
           done;
           at := !at + m
         end;
         let lo = first j i in
         let len = stop j i - lo in
         blit vs.(j) lo b !at len;
         at := !at + len
       done
     end
   done);
  Nx_ragged.v ~offsets:(value offsets) (value bytes)

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
  let a = fresh Nx.uint8 ((n + 7) / 8) in
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
  (fill a @@ fun (hits : buf) ->
   A.fill hits 0;
   let hit i =
     A.unsafe_set hits (i lsr 3)
       (A.unsafe_get hits (i lsr 3) lor (1 lsl (i land 7)))
   in
   rows ~by ?mask r (fun i v first stop -> if matches v first stop then hit i));
  Nx.slice
    [ Nx.R (0, n) ]
    (Nx.reshape [| -1 |] (Nx.bitcast Nx.bit (value a)))

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

let split ~by ?mask sep r =
  let n = Nx_ragged.length r and m = String.length sep in
  let matches = Array.make n (-1) and kept = Array.make n 0 in
  rows ~by ?mask r (fun i v first stop ->
      let k = occurrences v first stop sep in
      matches.(i) <- k;
      kept.(i) <- stop - first - (k * m));
  let lists, total = cumulative n (fun i -> matches.(i) + 1) in
  let pieces = fresh Nx.int64 (total + 1) in
  let bytes = fresh Nx.uint8 (Array.fold_left ( + ) 0 kept) in
  (fill pieces @@ fun (p : int64s) ->
   fill bytes @@ fun (b : buf) ->
   A.unsafe_set p 0 0L;
   let piece = ref 0 and at = ref 0 in
   let emit v lo hi =
     blit v lo b !at (hi - lo);
     at := !at + hi - lo;
     incr piece;
     A.unsafe_set p !piece (Int64.of_int !at)
   in
   rows ~by ?mask r (fun _ v first stop ->
       let rec go i =
         match find v i stop sep with
         | -1 -> emit v i stop
         | j ->
             emit v i j;
             go (j + m)
       in
       go first));
  (value lists, Nx_ragged.v ~offsets:(value pieces) (value bytes))

let replace ~by ?mask ~sub ~into r =
  let n = Nx_ragged.length r
  and grow = String.length into - String.length sub in
  let sizes = Array.make n 0 in
  rows ~by ?mask r (fun i v first stop ->
      sizes.(i) <- stop - first + (grow * occurrences v first stop sub));
  let offsets, total = cumulative n (Array.get sizes) in
  let bytes = fresh Nx.uint8 total in
  (fill bytes @@ fun (b : buf) ->
   peek offsets @@ fun (o : int64s) ->
   let put = ref 0 in
   let copy v lo hi =
     blit v lo b !put (hi - lo);
     put := !put + hi - lo
   in
   rows ~by ?mask r (fun i v first stop ->
       put := Int64.to_int (A.unsafe_get o i);
       let rec go j =
         match find v j stop sub with
         | -1 -> copy v j stop
         | h ->
             copy v j h;
             String.iteri
               (fun k c -> A.unsafe_set b (!put + k) (Char.code c))
               into;
             put := !put + String.length into;
             go (h + String.length sub)
       in
       go first));
  Nx_ragged.v ~offsets:(value offsets) (value bytes)

(* [compare_rows v o n s accept bits] sets bit [r] of [bits] iff the order of
   row [r] of the [n] rows of [v], cut by [o], against [s] is in [accept]: bit
   [1 + c] for order [c]. *)
external compare_rows :
  buf -> int64s -> (int[@untagged]) -> string -> (int[@untagged]) -> buf -> unit
  = "talon_compare_byte" "talon_compare"
[@@noalloc]

(* [member_rows v o n texts bits] sets bit [r] of [bits] iff row [r] of the [n]
   rows of [v], cut by [o], is one of [texts]. *)
external member_rows :
  buf -> int64s -> (int[@untagged]) -> string array -> buf -> unit
  = "talon_member_byte" "talon_member"
[@@noalloc]

let accept = function
  | `Lt -> 0b001
  | `Eq -> 0b010
  | `Le -> 0b011
  | `Gt -> 0b100
  | `Ne -> 0b101
  | `Ge -> 0b110

(* [words ~by r] is the rows of [r] as strings. *)
let words ~by r =
  let words = Array.make (Nx_ragged.length r) "" in
  rows ~by r (fun i v first stop ->
      words.(i) <-
        String.init (stop - first) (fun k ->
            Char.unsafe_chr (A.unsafe_get v (first + k))));
  words

(* [bits ~by r f] is the bits [f] sets, one a row of [r], given the offsets
   and bytes of [r]. *)
let bits ~by r f =
  let n = Nx_ragged.length r in
  let a = fresh Nx.uint8 ((n + 7) / 8) in
  (fill a @@ fun bits -> host ~by r (fun o v _ -> f v o n bits));
  Nx.slice [ Nx.R (0, n) ] (Nx.reshape [| -1 |] (Nx.bitcast Nx.bit (value a)))

let compare ~by op r one =
  let s = (words ~by one).(0) in
  bits ~by r (fun v o n b -> compare_rows v o n s (accept op) b)

let member ~by r texts =
  let texts = words ~by texts in
  bits ~by r (fun v o n b -> member_rows v o n texts b)
