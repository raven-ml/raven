open Bigarray

(* Buffers *)

let create n =
  let b = Array1.create int8_unsigned c_layout n in
  Array1.fill b 0xa5;
  b

let blit s b = String.iteri (fun i c -> Array1.unsafe_set b i (Char.code c)) s

let of_string s =
  let b = Array1.create int8_unsigned c_layout (String.length s) in
  blit s b;
  b

let to_string ?length b =
  let n = match length with Some n -> n | None -> Array1.dim b in
  String.init n (fun i -> Char.unsafe_chr (Array1.get b i))

let room ~offset n = Array1.sub (create (offset + n)) offset n

let placed ~offset s =
  let b = room ~offset (String.length s) in
  blit s b;
  b

let stepped ?cut d data =
  let src = of_string data and n = String.length data in
  let out = Buffer.create (2 * n) in
  let add run = Buffer.add_string out (to_string run) in
  let rec go pos k idle =
    if idle > 1 then Windtrap.fail "two steps in a row made no progress";
    let left = n - pos in
    let len = match cut with Some c -> min (max 0 (c k)) left | None -> left in
    let last = len = left in
    let before = Buffer.length out in
    match Lz.Decoder.step d ~last (Array1.sub src pos len) add with
    | Error e -> Error e
    | Ok (Lz.Done j) -> Ok (Buffer.contents out, pos + j)
    | Ok (More j) ->
        let still = j = 0 && Buffer.length out = before in
        go (pos + j) (k + 1) (if still then idle + 1 else 0)
  in
  go 0 0 0

(* Corpora *)

type kind = Text | Columns | Random

let kind_of_string = function
  | "text" -> Text
  | "columns" -> Columns
  | "random" -> Random
  | s -> invalid_arg ("Lz_support.kind_of_string: " ^ s)

(* A xorshift64 generator from [seed]. *)
let xorshift seed =
  let x = ref seed in
  fun () ->
    let v = !x in
    let v = Int64.logxor v (Int64.shift_left v 13) in
    let v = Int64.logxor v (Int64.shift_right_logical v 7) in
    let v = Int64.logxor v (Int64.shift_left v 17) in
    x := v;
    v

let urem v k = Int64.to_int (Int64.unsigned_rem v (Int64.of_int k))

let syllables =
  [| "ka"; "lo"; "mi"; "ne"; "su"; "ta"; "ri"; "po"; "an"; "el"; "or"; "is";
     "um"; "ve"; "da"; "go" |]

let text n =
  let next = xorshift 0x51F15EEDL in
  let words =
    Array.init 512 (fun _ ->
        let k = urem (next ()) 3 + 1 in
        String.concat "" (List.init k (fun _ -> syllables.(urem (next ()) 16))))
  in
  let b = Buffer.create (n + 16) in
  while Buffer.length b < n do
    Buffer.add_string b words.(urem (next ()) 512);
    Buffer.add_char b (if urem (next ()) 12 = 0 then '\n' else ' ')
  done;
  Buffer.sub b 0 n

let columns n =
  let next = xorshift 0x9E3779B97F4A7C15L in
  let b = Buffer.create (n + 16) in
  let key = ref 0L in
  while Buffer.length b < n do
    key := Int64.add !key (Int64.of_int (urem (next ()) 16 + 1));
    let price = float_of_int (urem (next ()) 100000) /. 100. in
    Buffer.add_int64_le b !key;
    Buffer.add_int64_le b (Int64.bits_of_float price)
  done;
  Buffer.sub b 0 n

let random n =
  let next = xorshift 0x2545F4914F6CDD1DL in
  let b = Buffer.create (n + 8) in
  while Buffer.length b < n do
    Buffer.add_int64_le b (next ())
  done;
  Buffer.sub b 0 n

(* Each kind is built once, at the largest size asked so far: the streams
   are prefixes of each other. *)
let cache = Hashtbl.create 3

let corpus kind n =
  match Hashtbl.find_opt cache kind with
  | Some s when String.length s >= n -> String.sub s 0 n
  | _ ->
      let s =
        match kind with Text -> text n | Columns -> columns n | Random -> random n
      in
      Hashtbl.replace cache kind s;
      s

(* Drawn inputs *)

type recipe =
  | Empty
  | Byte of char
  | Run of char * int
  | Noise of int * int
  | Corpus of kind * int
  | Match of { distance : int; length : int; seed : int }
  | Concat of recipe list

let noise seed n =
  let st = Random.State.make [| seed |] in
  String.init n (fun _ -> Char.chr (Random.State.int st 256))

let rec build = function
  | Empty -> ""
  | Byte c -> String.make 1 c
  | Run (c, n) -> String.make n c
  | Noise (seed, n) -> noise seed n
  | Corpus (k, n) -> corpus k n
  | Match { distance; length; seed } ->
      let head = noise seed distance in
      let b = Bytes.create (distance + length + 8) in
      Bytes.blit_string head 0 b 0 distance;
      for i = distance to distance + length - 1 do
        Bytes.set b i (Bytes.get b (i - distance))
      done;
      Bytes.blit_string (noise (seed + 1) 8) 0 b (distance + length) 8;
      Bytes.unsafe_to_string b
  | Concat rs -> String.concat "" (List.map build rs)

let kind_name = function
  | Text -> "Text"
  | Columns -> "Columns"
  | Random -> "Random"

let rec pp_recipe ppf = function
  | Empty -> Format.pp_print_string ppf "Empty"
  | Byte c -> Format.fprintf ppf "Byte %C" c
  | Run (c, n) -> Format.fprintf ppf "Run (%C, %d)" c n
  | Noise (s, n) -> Format.fprintf ppf "Noise (%d, %d)" s n
  | Corpus (k, n) -> Format.fprintf ppf "Corpus (%s, %d)" (kind_name k) n
  | Match { distance; length; seed } ->
      Format.fprintf ppf "Match { distance = %d; length = %d; seed = %d }"
        distance length seed
  | Concat rs ->
      Format.fprintf ppf "Concat [%a]"
        (Format.pp_print_list
           ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
           pp_recipe)
        rs

let kib = 1024
let mib = 1024 * kib

let distances =
  [ 1; 2; 3; 4; 7; 8; 15; 16; 31; 32; 33; 64; 65; 256; 257; 1024;
    (32 * kib) - 1; 32 * kib; (32 * kib) + 1; 64 * kib; mib ]

let lengths = [ 3; 4; 5; 15; 16; 19; 258; 259; 273; 65535; 65536 ]

let recipe =
  let open Windtrap.Gen in
  let kind = of_list [ Text; Columns; Random ] in
  let size = one_of [ int_range 1 64; int_range 65 4096; int_range 4097 200_000 ] in
  let leaf =
    frequency
      [
        (1, constant Empty);
        (1, map (fun c -> Byte c) char);
        (1, map (fun (c, n) -> Run (c, n)) (pair char (int_range 2 70_000)));
        (1, map (fun (s, n) -> Noise (s, n)) (pair small_int size));
        (1, map (fun (k, n) -> Corpus (k, n)) (pair kind size));
        ( 1,
          map
            (fun ((distance, length), seed) -> Match { distance; length; seed })
            (pair (pair (of_list distances) (of_list lengths)) small_int) );
      ]
  in
  with_pp pp_recipe
    (frequency
       [ (6, leaf); (1, map (fun rs -> Concat rs) (list ~size:(int_range 2 4) leaf)) ])

let large =
  [
    Corpus (Text, (128 * kib) + 1);
    Corpus (Columns, (4 * mib) + 1);
    Concat
      [ Corpus (Text, 5 * mib); Match { distance = mib; length = 65536; seed = 1 } ];
  ]

(* Bounds *)

let has_bound bound n =
  match bound n with _ -> true | exception Invalid_argument _ -> false

let largest_bound bound =
  Windtrap.test "a bound past max_int raises" (fun () ->
      let open Windtrap in
      let past n =
        raises_match ~msg:(string_of_int n)
          (Exn.invalid_arg ~substring:"max_int")
          (fun () -> bound n)
      in
      past max_int;
      (* [lo] has a bound, [hi] has none. *)
      let rec find lo hi =
        if hi - lo = 1 then lo
        else
          let mid = lo + ((hi - lo) / 2) in
          if has_bound bound mid then find mid hi else find lo mid
      in
      let t = find 0 max_int in
      let b = bound t in
      at_least int ~msg:"at least the size" ~than:t b;
      greater int ~msg:"above max_int - 8" ~than:(max_int - 8) b;
      at_most int ~msg:"the size before's" ~than:b (bound (t - 1));
      past (t + 1);
      past (t + 2))

(* The runtime lock *)

type lock = Released | Held | Unproven

(* [watch long ~small] runs [long ()] on another domain and, once it has
   started, [small ()] 100 times here, which gives [long] time to enter its
   kernel, then one minor collection, which waits for every domain that holds
   its runtime lock. [finished] read before and after the collection tells
   the three outcomes apart. *)
let watch long ~small =
  let started = Atomic.make false and finished = Atomic.make false in
  let d =
    Domain.spawn (fun () ->
        Atomic.set started true;
        let r = long () in
        Atomic.set finished true;
        r)
  in
  while not (Atomic.get started) do
    Domain.cpu_relax ()
  done;
  for _ = 1 to 100 do
    small ()
  done;
  let before = Atomic.get finished in
  Gc.minor ();
  let after = Atomic.get finished in
  let r = Domain.join d in
  let lock = if before then Unproven else if after then Held else Released in
  (lock, r)

let lock_released name long ~small ~check =
  Windtrap.test name (fun () ->
      let run k =
        let lock, r = watch (long k) ~small in
        check r;
        lock
      in
      match run 1 with
      | Released -> ()
      | Held | Unproven -> (
          match run 2 with
          | Released -> ()
          | Held ->
              Windtrap.fail
                "a minor collection begun while the long call ran returned \
                 only after it ended"
          | Unproven ->
              Windtrap.skip
                ~reason:"the long call ended before the collection began" ()))

(* Short destinations *)

type short = Short of int | Part of int

let pp_short ppf = function
  | Short d -> Format.fprintf ppf "Short %d" d
  | Part p -> Format.fprintf ppf "Part %d" p

let short =
  let open Windtrap.Gen in
  with_pp pp_short
    (frequency
       [
         (1, constant (Short 1));
         (1, constant (Part 0));
         (2, map (fun d -> Short d) (int_range 2 16));
         (2, map (fun p -> Part p) (int_range 1 999));
       ])

let capacity k = function Short d -> max 0 (k - d) | Part p -> k * p / 1000

(* Checksum paths *)

type path = { name : string; run : int32 -> Lz.bigbytes -> int32 }

let paths = List.map (fun (name, run) -> { name; run })
let crc32_paths = paths Lz.Paths.crc32
let adler32_paths = paths Lz.Paths.adler32

(* Zstandard paths *)

type zstd_path = {
  label : string;
  decode : Lz.bigbytes -> Lz.bigbytes -> (Lz.progress, Lz.error) result;
}

let zstd_paths =
  List.map (fun (label, decode) -> { label; decode }) Lz.Paths.zstd_decode

type zstd_enc_path = {
  label : string;
  encode : level:int -> Lz.bigbytes -> Lz.bigbytes -> int option;
}

let zstd_enc_paths =
  List.map
    (fun (label, encode) : zstd_enc_path -> { label; encode })
    Lz.Paths.zstd_encode

(* [limited_steps e ~step src dst] steps [e] over [src], at most [step]
   bytes a step, and writes the runs to [dst]: the stream's length, or
   [None] when it does not fit. *)
let limited_steps e ~step src dst =
  let n = Array1.dim src in
  let out = ref 0 in
  let add run =
    let k = Array1.dim run in
    if !out + k <= Array1.dim dst then
      Array1.blit run (Array1.sub dst !out k);
    out := !out + k
  in
  let rec go at =
    let len = min step (n - at) in
    let last = at + len = n in
    match Lz.Encoder.step e ~last (Array1.sub src at len) add with
    | Lz.Done _ -> ()
    | More taken -> go (at + taken)
  in
  go 0;
  if !out <= Array1.dim dst then Some !out else None

let zstd_encode_limit ~limit ?step ~level src dst =
  match step with
  | None -> Lz.zstd_encode_limit ~limit ~level src dst
  | Some s when s > 0 ->
      limited_steps (Lz.zstd_encoder_limit ~limit ~level) ~step:s src dst
  | Some _ -> invalid_arg "Lz_support.zstd_encode_limit: step"

(* Deflate paths *)

type deflate_enc_path = {
  label : string;
  encode : level:int -> Lz.bigbytes -> Lz.bigbytes -> int option;
}

let deflate_enc_paths =
  List.map
    (fun (label, encode) : deflate_enc_path -> { label; encode })
    Lz.Paths.deflate_encode

(* Testables *)

let pp_why ppf = function
  | Lz.Truncated -> Format.pp_print_string ppf "Truncated"
  | Malformed s -> Format.fprintf ppf "Malformed %S" s
  | Too_long -> Format.pp_print_string ppf "Too_long"
  | Unsupported s -> Format.fprintf ppf "Unsupported %S" s

let why = Windtrap.Testable.make ~pp:pp_why ~equal:( = )

let pp_error ppf { Lz.at; why } =
  Format.fprintf ppf "{ at = %d; why = %a }" at pp_why why

let error = Windtrap.Testable.make ~pp:pp_error ~equal:( = )

let progress =
  Windtrap.Testable.make
    ~pp:(fun ppf { Lz.consumed; produced } ->
      Format.fprintf ppf "{ consumed = %d; produced = %d }" consumed produced)
    ~equal:( = )

let step =
  Windtrap.Testable.make
    ~pp:(fun ppf -> function
      | Lz.Done k -> Format.fprintf ppf "Done %d" k
      | More k -> Format.fprintf ppf "More %d" k)
    ~equal:( = )

(* A string longer than this prints as its length and digest. *)
let shown = 64

let pp_bytes ppf s =
  if String.length s <= shown then Format.fprintf ppf "%S" s
  else
    Format.fprintf ppf "<%d bytes, md5 %s>" (String.length s)
      (Digest.to_hex (Digest.string s))

let decoded =
  Windtrap.Testable.make
    ~pp:(fun ppf -> function
      | Ok s -> Format.fprintf ppf "Ok %a" pp_bytes s
      | Error e -> Format.fprintf ppf "Error %a" pp_error e)
    ~equal:( = )

(* Goldens *)

type golden = {
  file : string;
  codec : string;
  vendor : string;
  members : int;
  raw : string Lazy.t;
  data : string Lazy.t;
}

let read path = In_channel.with_open_bin path In_channel.input_all

(* A raw spec is [kind:n] parts joined by [+]. *)
let raw_of_spec spec =
  String.split_on_char '+' spec
  |> List.map (fun part ->
         match String.split_on_char ':' part with
         | [ k; n ] -> corpus (kind_of_string k) (int_of_string n)
         | _ -> invalid_arg ("Lz_support: raw spec " ^ spec))
  |> String.concat ""

let goldens () =
  read (Filename.concat "golden" "manifest.txt")
  |> String.split_on_char '\n'
  |> List.filter_map (fun line ->
         match String.split_on_char ' ' (String.trim line) with
         | [ file; codec; vendor; members; spec ] ->
             Some
               {
                 file;
                 codec;
                 vendor;
                 members = int_of_string members;
                 raw = lazy (raw_of_spec spec);
                 data = lazy (read (Filename.concat "golden" file));
               }
         | [ "" ] -> None
         | _ -> invalid_arg ("Lz_support: manifest line " ^ line))

let goldens_of codec = List.filter (fun g -> g.codec = codec) (goldens ())

let decoder = function
  | "deflate" -> Lz.Deflate.decode
  | "zlib" -> Lz.Zlib.decode
  | "gzip" -> Lz.Gzip.decode
  | "zstd" -> Lz.Zstd.decode
  | "lz4b" -> Lz.Lz4.Block.decode
  | "lz4f" -> Lz.Lz4.Frame.decode
  | "snappy" -> Lz.Snappy.decode
  | c -> invalid_arg ("Lz_support.decoder: " ^ c)

(* Suites over the goldens *)

let decode_string codec data n =
  let dst = create n in
  match decoder codec (of_string data) dst with
  | Ok { Lz.consumed; produced } ->
      Windtrap.equal Windtrap.int ~msg:"consumed" (String.length data) consumed;
      Ok (to_string ~length:produced dst)
  | Error e -> Error e

let agreement codec =
  Windtrap.cases ~name:(fun g -> g.file) "vendor goldens" (goldens_of codec)
    (fun g ->
      let raw = Lazy.force g.raw in
      Windtrap.equal decoded (Ok raw)
        (decode_string codec (Lazy.force g.data) (String.length raw)))

let truncations codec =
  if codec = "lz4b" then invalid_arg "Lz_support.truncations: lz4b";
  let files =
    List.filter
      (fun g -> g.members = 1 && String.length (Lazy.force g.data) <= 1024)
      (goldens_of codec)
  in
  Windtrap.cases ~name:(fun g -> g.file) "every prefix is Truncated" files
    (fun g ->
      let data = Lazy.force g.data in
      let n = String.length (Lazy.force g.raw) in
      for cut = 0 to String.length data - 1 do
        let r = decoder codec (of_string (String.sub data 0 cut)) (create n) in
        Windtrap.equal (Windtrap.result progress error)
          ~msg:(Printf.sprintf "cut at %d" cut)
          (Error { Lz.at = cut; why = Truncated })
          r
      done)

(* A decode of mutated bytes may succeed or fail; it must stay inside its
   buffers, which the sanitizers check, and report offsets inside them. *)
let within data r =
  let len = String.length data in
  match r with
  | Ok { Lz.consumed; produced = _ } ->
      Windtrap.at_most Windtrap.int ~than:len consumed
  | Error { Lz.at; why = _ } ->
      Windtrap.at_least Windtrap.int ~than:0 at;
      Windtrap.at_most Windtrap.int ~than:len at

let mutations codec =
  let files limit =
    List.filter (fun g -> String.length (Lazy.force g.data) <= limit)
      (goldens_of codec)
  in
  let decode g data =
    decoder codec (of_string data) (create (String.length (Lazy.force g.raw)))
  in
  let flips =
    Windtrap.cases ~name:(fun g -> g.file) "every bit flipped" (files 512)
      (fun g ->
        let data = Lazy.force g.data in
        for bit = 0 to (8 * String.length data) - 1 do
          let b = Bytes.of_string data in
          let i = bit / 8 in
          Bytes.set b i
            (Char.chr (Char.code data.[i] lxor (1 lsl (bit mod 8))));
          let m = Bytes.to_string b in
          within m (decode g m)
        done)
  in
  let pool = Array.of_list (files 65536) in
  let replaced =
    Windtrap.prop "any byte replaced"
      Windtrap.Gen.(triple (int_range 0 (max 0 (Array.length pool - 1))) nat char)
      (fun (k, at, c) ->
        if Array.length pool = 0 then Windtrap.skip ();
        let g = pool.(k) in
        let data = Lazy.force g.data in
        Windtrap.assume (String.length data > 0);
        let b = Bytes.of_string data in
        Bytes.set b (at mod String.length data) c;
        let m = Bytes.to_string b in
        within m (decode g m))
  in
  Windtrap.group "mutations" [ flips; replaced ]
