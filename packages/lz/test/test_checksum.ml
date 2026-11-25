open Windtrap
open Bigarray

(* References, written from the specifications' definitions, one byte or
   one lane at a time: RFC 1952 section 8's bitwise CRC-32, RFC 1950
   section 2.2's Adler-32 sums, the xxHash specification's steps. *)

let m32 = 0xFFFFFFFF

let ref_crc32 crc s =
  let r = ref (Int32.to_int crc land m32 lxor m32) in
  String.iter
    (fun c ->
      r := !r lxor Char.code c;
      for _ = 1 to 8 do
        r := if !r land 1 = 1 then (!r lsr 1) lxor 0xEDB88320 else !r lsr 1
      done)
    s;
  Int32.of_int (!r lxor m32)

let ref_adler32 adler s =
  let a = Int32.to_int adler land m32 in
  let s1 = ref (a land 0xFFFF) and s2 = ref (a lsr 16) in
  String.iter
    (fun c ->
      s1 := (!s1 + Char.code c) mod 65521;
      s2 := (!s2 + !s1) mod 65521)
    s;
  Int32.of_int ((!s2 lsl 16) lor !s1)

let rotl32 x r = ((x lsl r) lor (x lsr (32 - r))) land m32
let mul32 a b = a * b land m32
let le32 s i = Int32.to_int (String.get_int32_le s i) land m32

let ref_xxh32 seed s =
  let p1 = 0x9E3779B1 and p2 = 0x85EBCA77 and p3 = 0xC2B2AE3D in
  let p4 = 0x27D4EB2F and p5 = 0x165667B1 in
  let seed = Int32.to_int seed land m32 and n = String.length s in
  let round acc lane = mul32 (rotl32 ((acc + mul32 lane p2) land m32) 13) p1 in
  let i = ref 0 and h = ref ((seed + p5) land m32) in
  if n >= 16 then begin
    let a =
      [|
        (seed + p1 + p2) land m32;
        (seed + p2) land m32;
        seed;
        (seed - p1) land m32;
      |]
    in
    while n - !i >= 16 do
      for k = 0 to 3 do
        a.(k) <- round a.(k) (le32 s (!i + (4 * k)))
      done;
      i := !i + 16
    done;
    h :=
      (rotl32 a.(0) 1 + rotl32 a.(1) 7 + rotl32 a.(2) 12 + rotl32 a.(3) 18)
      land m32
  end;
  h := (!h + n) land m32;
  while n - !i >= 4 do
    h := mul32 (rotl32 ((!h + mul32 (le32 s !i) p3) land m32) 17) p4;
    i := !i + 4
  done;
  while n - !i >= 1 do
    h := mul32 (rotl32 ((!h + mul32 (Char.code s.[!i]) p5) land m32) 11) p1;
    incr i
  done;
  let h = !h in
  let h = mul32 (h lxor (h lsr 15)) p2 in
  let h = mul32 (h lxor (h lsr 13)) p3 in
  Int32.of_int (h lxor (h lsr 16))

let ref_xxh64 seed s =
  let open Int64 in
  let p1 = 0x9E3779B185EBCA87L and p2 = 0xC2B2AE3D27D4EB4FL in
  let p3 = 0x165667B19E3779F9L and p4 = 0x85EBCA77C2B2AE63L in
  let p5 = 0x27D4EB2F165667C5L in
  let rotl x r = logor (shift_left x r) (shift_right_logical x (64 - r)) in
  let round acc lane = mul (rotl (add acc (mul lane p2)) 31) p1 in
  let merge h acc = add (mul (logxor h (round 0L acc)) p1) p4 in
  let n = String.length s in
  let i = ref 0 and h = ref (add seed p5) in
  if n >= 32 then begin
    let a = [| add seed (add p1 p2); add seed p2; seed; sub seed p1 |] in
    while n - !i >= 32 do
      for k = 0 to 3 do
        a.(k) <- round a.(k) (String.get_int64_le s (!i + (8 * k)))
      done;
      i := !i + 32
    done;
    h :=
      add (add (rotl a.(0) 1) (rotl a.(1) 7)) (add (rotl a.(2) 12) (rotl a.(3) 18));
    Array.iter (fun acc -> h := merge !h acc) a
  end;
  h := add !h (of_int n);
  while n - !i >= 8 do
    let lane = String.get_int64_le s !i in
    h := add (mul (rotl (logxor !h (round 0L lane)) 27) p1) p4;
    i := !i + 8
  done;
  if n - !i >= 4 then begin
    let lane = of_int (le32 s !i) in
    h := add (mul (rotl (logxor !h (mul lane p1)) 23) p2) p3;
    i := !i + 4
  end;
  while n - !i >= 1 do
    h := mul (rotl (logxor !h (mul (of_int (Char.code s.[!i])) p5)) 11) p1;
    incr i
  done;
  let h = !h in
  let h = mul (logxor h (shift_right_logical h 33)) p2 in
  let h = mul (logxor h (shift_right_logical h 29)) p3 in
  logxor h (shift_right_logical h 32)

(* The four checksums, each with its running value or seed as an int64, so that
   one law runs over all of them. *)

type checksum = {
  name : string;
  bigbytes : int64 -> Lz.bigbytes -> int64;
  string : int64 -> ?first:int -> ?length:int -> string -> int64;
  reference : int64 -> string -> int64;
  value : int64 -> int64;
      (* A running value or seed of the checksum drawn from any int64. *)
  paths : (string * (int32 -> Lz.bigbytes -> int32)) list;
      (* The checksum's paths this host can run, from Lz.Paths, which the module
         alone cannot reach: it takes only the fastest. *)
}

let u32 x = Int64.logand (Int64.of_int32 x) 0xFFFFFFFFL
let i32 = Int64.to_int32
let width32 c = Int64.logand c 0xFFFFFFFFL

(* An Adler-32 value's halves are each below 65521. *)
let adler_valid c =
  let half x = Int64.rem (Int64.logand x 0xFFFFL) 65521L in
  Int64.logor
    (Int64.shift_left (half (Int64.shift_right_logical c 16)) 16)
    (half c)

let crc32 =
  {
    name = "CRC-32";
    bigbytes =
      (fun c b -> Int64.of_int (Lz.Crc32.bigbytes ~crc:(Int64.to_int c) b));
    string =
      (fun c ?first ?length s ->
        Int64.of_int (Lz.Crc32.string ~crc:(Int64.to_int c) ?first ?length s));
    reference = (fun c s -> u32 (ref_crc32 (i32 c) s));
    value = width32;
    paths = Lz.Paths.crc32;
  }

let adler32 =
  {
    name = "Adler-32";
    bigbytes = (fun c b -> u32 (Lz.Adler32.bigbytes ~adler:(i32 c) b));
    string =
      (fun c ?first ?length s ->
        u32 (Lz.Adler32.string ~adler:(i32 c) ?first ?length s));
    reference = (fun c s -> u32 (ref_adler32 (i32 c) s));
    value = adler_valid;
    paths = Lz.Paths.adler32;
  }

let xxh32 =
  {
    name = "XXH32";
    bigbytes = (fun c b -> u32 (Lz.Xxh32.bigbytes ~seed:(i32 c) b));
    string =
      (fun c ?first ?length s ->
        u32 (Lz.Xxh32.string ~seed:(i32 c) ?first ?length s));
    reference = (fun c s -> u32 (ref_xxh32 (i32 c) s));
    value = width32;
    paths = [];
  }

let xxh64 =
  {
    name = "XXH64";
    bigbytes = (fun c b -> Lz.Xxh64.bigbytes ~seed:c b);
    string =
      (fun c ?first ?length s -> Lz.Xxh64.string ~seed:c ?first ?length s);
    reference = ref_xxh64;
    value = Fun.id;
    paths = [];
  }

let checksums = [ crc32; adler32; xxh32; xxh64 ]

(* Every way to sum a buffer with [c]: its module, then each of its paths. *)
let ways c =
  (c.name, c.bigbytes)
  :: List.map
       (fun (path, run) ->
         (c.name ^ " " ^ path, fun v b -> u32 (run (i32 v) b)))
       c.paths

let hex =
  Testable.make
    ~pp:(fun ppf x -> Format.fprintf ppf "0x%Lx" x)
    ~equal:Int64.equal

(* Every way to sum [b], which holds [s], from [v] gives the reference's
   value. *)
let agrees c v s b =
  let expected = c.reference v s in
  List.iter (fun (name, run) -> equal ~msg:name hex expected (run v b)) (ways c)

(* Check values *)

(* CRC-32's check value is the CRC catalogue's ("CRC-32/ISO-HDLC", check =
   0xCBF43926); Adler-32's is the worked example of "Wikipedia". The xxHash
   specification (0.2.0) lists no test vectors: the XXH32 and XXH64 rows are the
   digests of xxHash 0.8.3, the reference library, through its Python binding.
   The inputs cross a stripe, and leave tails of every kind the specification's
   step 5 reads. *)
let checks =
  [
    (crc32, 0L, "123456789", 0xCBF43926L);
    (crc32, 0L, "", 0L);
    (adler32, 1L, "Wikipedia", 0x11E60398L);
    (adler32, 1L, "", 1L);
    (xxh32, 0L, "", 0x02CC5D05L);
    (xxh32, 0L, "a", 0x550D7456L);
    (xxh32, 0L, "abc", 0x32D153FFL);
    (xxh32, 0L, "123456789", 0x937BAD67L);
    (xxh32, 0L, "Nobody inspects the spammish repetition", 0xE2293B2FL);
    (xxh32, 0L, "0123456789abcdef0123456789abcdef0123456789", 0x386F91D2L);
    (xxh64, 0L, "", 0xEF46DB3751D8E999L);
    (xxh64, 0L, "a", 0xD24EC4F1A98C6E5BL);
    (xxh64, 0L, "abc", 0x44BC2CF5AD770999L);
    (xxh64, 0L, "123456789", 0x8CB841DB40E6AE83L);
    (xxh64, 0L, "Nobody inspects the spammish repetition", 0xFBCEA83C8A378BF1L);
    ( xxh64,
      0L,
      "0123456789abcdef0123456789abcdef0123456789",
      0xA76190C3ACF08A1CL );
  ]

let check_name (c, v, s, _) = Printf.sprintf "%s 0x%Lx %S" c.name v s

(* Drawn inputs *)

let bytes_of_seed seed n =
  let st = Random.State.make [| seed |] in
  String.init n (fun _ -> Char.chr (Random.State.int st 256))

(* Every way agrees with the reference at every length to 300 and every
   alignment to 15: the lengths cross each block and each tail of the paths,
   whose widest step is 256 bytes. A test sweeps them all over fixed bytes, and
   a property draws the bytes, a length and an alignment. *)
let short_lengths = 300
let alignments = 16

(* The views, one for each length at each alignment, are made once and take each
   case's bytes. *)
let buffers =
  lazy
    (Array.init alignments (fun offset ->
         let v = Lz_support.room ~offset short_lengths in
         (v, Array.init (short_lengths + 1) (fun n -> Array1.sub v 0 n))))

let fill v s = String.iteri (fun i b -> Array1.unsafe_set v i (Char.code b)) s

let sweep () =
  let s = bytes_of_seed 0 short_lengths in
  let views =
    Array.map
      (fun (v, prefixes) ->
        fill v s;
        prefixes)
      (Lazy.force buffers)
  in
  List.iter
    (fun c ->
      let v = c.value 0x0123_4567_89ab_cdefL in
      let expected =
        Array.init (short_lengths + 1) (fun n ->
            c.reference v (String.sub s 0 n))
      in
      List.iter
        (fun (name, run) ->
          for n = 0 to short_lengths do
            Array.iteri
              (fun a prefixes ->
                let got = run v prefixes.(n) in
                if not (Int64.equal got expected.(n)) then
                  failf "%s: length %d at alignment %d: 0x%Lx, expected 0x%Lx"
                    name n a got expected.(n))
              views
          done)
        (ways c))
    checksums

let drawn_length =
  Gen.(
    quad int int64 (int_range 0 short_lengths) (int_range 0 (alignments - 1)))

let at_length (seed, v, n, a) =
  let b, prefixes = (Lazy.force buffers).(a) in
  let s = bytes_of_seed seed n in
  fill b s;
  List.iter (fun c -> agrees c (c.value v) s prefixes.(n)) checksums

(* Lengths to 70000 at any alignment: across Adler-32's block of 5552 bytes and
   many turns of every loop. Each band is drawn half the time, so that every run
   of 100 cases covers both. *)
let long_input =
  Gen.(
    quad
      (one_of [ int_range (short_lengths + 1) 5552; int_range 5553 70000 ])
      (int_range 0 (alignments - 1))
      int int64)

let long_agrees (n, offset, seed, v) =
  cover "over 5552 bytes" (n > 5552);
  cover "from 301 to 5552 bytes" (n > short_lengths && n <= 5552);
  let s = bytes_of_seed seed n in
  let b = Lz_support.placed ~offset s in
  List.iter (fun c -> agrees c (c.value v) s b) checksums

(* Bytes of 255 from the largest running value, where Adler-32's sums grow
   fastest: at the edges of its 5552-byte reduction and of the vector paths'
   runs of 5504 bytes, and over many of them. *)
let saturated = [ 5503; 5504; 5505; 5551; 5552; 5553; 11008; 11009; 70001 ]

let saturating n =
  let s = String.make n '\xff' in
  let b = Lz_support.of_string s in
  List.iter (fun c -> agrees c (c.value 0xFFF0FFF0L) s b) checksums

(* 1 MiB, the size from which each module's [bigbytes] releases the domain lock.
   The paths release nothing, and the drawn lengths cover their loops. *)
let large () =
  let s = Lz_support.corpus Random (1 lsl 20) in
  let b = Lz_support.of_string s in
  List.iter
    (fun c ->
      let v = c.value 0x0123456789ABCDEFL in
      equal ~msg:c.name hex (c.reference v s) (c.bigbytes v b))
    checksums

(* CRC-32 and Adler-32 continue a running value: the checksum of [a ^ b] is the
   checksum of [b] from the checksum of [a]. *)
let continued =
  let s = Gen.(string_of ~size:(int_range 0 600) char) in
  Gen.triple s s Gen.int64

let continues (a, b, v) =
  List.iter
    (fun c ->
      let v = c.value v in
      let whole = c.string v (a ^ b) in
      equal ~msg:c.name hex whole (c.string (c.string v a) b);
      let ba = Lz_support.of_string a and bb = Lz_support.of_string b in
      equal ~msg:c.name hex whole (c.bigbytes (c.bigbytes v ba) bb))
    [ crc32; adler32 ]

(* [string ~first ~length] is [bigbytes] of the substring, and [string ~first]
   of the bytes from [first] on. *)
let range_input =
  Gen.(
    let* s = string_of ~size:(int_range 0 400) char in
    let n = String.length s in
    let* first = int_range 0 n in
    let* length = int_range 0 (n - first) in
    let+ v = int64 in
    (s, first, length, v))

let substring (s, first, length, v) =
  let sum c v t = c.bigbytes v (Lz_support.of_string t) in
  List.iter
    (fun c ->
      let v = c.value v in
      equal ~msg:c.name hex
        (sum c v (String.sub s first length))
        (c.string v ~first ~length s);
      equal ~msg:(c.name ^ ", no length") hex
        (sum c v (String.sub s first (String.length s - first)))
        (c.string v ~first s))
    checksums

(* Ranges that name no bytes of "abc". *)
let ranges =
  [
    (-1, None);
    (0, Some 4);
    (4, None);
    (4, Some 0);
    (2, Some (-1));
    (1, Some max_int);
    (max_int, None);
    (min_int, Some 1);
  ]

let name (first, length) =
  Printf.sprintf "first %d, length %s" first
    (match length with Some l -> string_of_int l | None -> "none")

let () =
  exit
    (run "Checksum"
       [
         group "Crc32"
           [
             test "~crc defaults to 0" (fun () ->
                 equal Windtrap.int 0xCBF43926 (Lz.Crc32.string "123456789");
                 equal Windtrap.int 0xCBF43926
                   (Lz.Crc32.bigbytes (Lz_support.of_string "123456789")));
             test "a ~crc of 2^32 - 1 continues over no bytes to itself"
               (fun () ->
                 equal Windtrap.int 0xFFFF_FFFF
                   (Lz.Crc32.string ~crc:0xFFFF_FFFF "");
                 equal Windtrap.int 0xFFFF_FFFF
                   (Lz.Crc32.bigbytes ~crc:0xFFFF_FFFF (Lz_support.of_string "")));
             cases "a ~crc outside [0, 2^32) raises Invalid_argument"
               ~id:string_of_int
               [ -1; 1 lsl 32; min_int; max_int ]
               (fun crc ->
                 let bad f = raises_match Exn.invalid_arg f in
                 bad (fun () -> Lz.Crc32.string ~crc "abc");
                 bad (fun () ->
                     Lz.Crc32.bigbytes ~crc (Lz_support.of_string "abc")));
           ];
         group "checksums"
           [
             cases "a sum of a check input is its check value" ~id:check_name
               checks (fun (c, v, s, expected) ->
                 equal ~msg:"string" hex expected (c.string v s);
                 equal ~msg:"bigbytes" hex expected
                   (c.bigbytes v (Lz_support.of_string s)));
             cases "the references give the check values" ~id:check_name checks
               (fun (c, v, s, expected) -> equal hex expected (c.reference v s));
             test
               "every length to 300 at every alignment agrees with the \
                reference"
               sweep;
             prop "drawn bytes, lengths and alignments agree with the reference"
               drawn_length at_length;
             prop "long inputs agree with the reference" long_input long_agrees;
             cases
               "bytes of 255 from the largest running value agree with the \
                reference"
               ~id:(Printf.sprintf "%d bytes of 255")
               saturated saturating;
             test "1 MiB agrees with the reference" large;
             prop "CRC-32 and Adler-32 continue a running value" continued
               continues;
             (* [first] at the string's end names its empty tail. *)
             prop "string ~first ~length is bigbytes of the substring"
               ~examples:[ ("abc", 3, 0, 0L) ]
               range_input substring;
             cases "a range outside the string raises Invalid_argument" ~id:name
               ranges (fun (first, length) ->
                 let bad f = raises_match Exn.invalid_arg f in
                 bad (fun () -> Lz.Crc32.string ~first ?length "abc");
                 bad (fun () -> Lz.Adler32.string ~first ?length "abc");
                 bad (fun () -> Lz.Xxh32.string ~first ?length "abc");
                 bad (fun () -> Lz.Xxh64.string ~first ?length "abc"));
           ];
         group "Paths"
           [
             test "the generic paths come first" (fun () ->
                 equal string "generic" (fst (List.hd Lz.Paths.crc32));
                 equal string "generic" (fst (List.hd Lz.Paths.adler32)));
           ];
       ])
