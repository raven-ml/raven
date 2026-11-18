(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* WAV. Files built here byte by byte from RIFF's specification,
   WAVEFORMATEXTENSIBLE and EBU Tech 3306 load as the samples they encode, each
   in the top bits of its dtype, for every container and coding a load reads;
   G.711 codes and files libsndfile wrote load as libsndfile reads them; every
   other coding, a size past the file and a set bit below a sample's valid bits
   raise [Failure] naming the file, and nothing else. *)

open Windtrap
open Io_support
module Audio = Nx_io.Audio

(* Files *)

let le n v = String.init n (fun i -> Char.chr ((v lsr (8 * i)) land 0xff))

(* A chunk, with the zero byte after an odd size. *)
let chunk id body =
  let n = String.length body in
  id ^ le 4 n ^ body ^ if n land 1 = 1 then "\000" else ""

let riff ?(id = "RIFF") chunks =
  let body = "WAVE" ^ String.concat "" chunks in
  id ^ le 4 (String.length body) ^ body

(* The SubFormat GUID of a format code. *)
let guid code =
  le 2 code ^ "\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"

let common ~channels ~rate ~container =
  let align = channels * container in
  le 2 channels ^ le 4 rate ^ le 4 (rate * align) ^ le 2 align

let plain ?(rate = 8000) ~code ~channels ~container bits =
  le 2 code ^ common ~channels ~rate ~container ^ le 2 bits

let extensible ?(rate = 8000) ?(mask = 0) ~code ~channels ~container valid =
  le 2 0xfffe
  ^ common ~channels ~rate ~container
  ^ le 2 (8 * container)
  ^ le 2 22 ^ le 2 valid ^ le 4 mask ^ guid code

let wav ?id fmt data = riff ?id [ chunk "fmt " fmt; chunk "data" data ]
let load_one path = Audio.samples (Nx_io.load_audio path)

(* Values *)

(* The value of [dt] whose elements' bytes, little-endian, are [s]. *)
let of_le (type v s) (dt : (v, s) D.t) shape s =
  let w = D.bytes dt 1 in
  let host =
    if not Sys.big_endian then s
    else
      String.init (String.length s) (fun i ->
          s.[(i / w * w) + (w - 1 - (i mod w))])
  in
  Nx.P
    (Nx.Repr.unsafe_of_array Nx.Host.v
       (A.v dt (L.contiguous shape) (Rig.Buffer.of_string host)))

(* The dtype of a PCM container of [k] bytes. *)
let pcm_dtype k =
  match k with
  | 1 -> D.Any D.Uint8
  | 2 -> D.Any D.Int16
  | 3 | 4 -> D.Any D.Int32
  | _ -> D.Any D.Int64

(* The value a load gives for the containers [s] of [k] bytes, in rows of [c]:
   each container's bytes, the top of an element of its dtype. *)
let expected ~k ~c s =
  let (D.Any dt) = pcm_dtype k in
  let w = D.bytes dt 1 in
  let n = String.length s / k in
  let elements =
    String.concat ""
      (List.init n (fun i ->
           String.make (w - k) '\000' ^ String.sub s (i * k) k))
  in
  of_le dt [| n / c; c |] elements

let facts a =
  ( (Audio.rate a, Audio.length a, Audio.channels a),
    (Audio.bits a, Audio.channel_mask a) )

let facts_t = pair (triple int int int) (pair int int)

(* Containers *)

(* A case: a container of [k] bytes, its [valid] bits, [c] channels, [rows]
   rows, WAVEFORMATEXTENSIBLE or plain, and the containers' bytes, zero below
   the valid bits. *)
type pcm = {
  k : int;
  valid : int;
  c : int;
  rows : int;
  ext : bool;
  data : string;
}

let pcm_gen =
  let open Gen in
  let* k = int_range 1 8 in
  let* valid = int_range 1 (8 * k) in
  let* ext = if (valid + 7) / 8 = k then bool else constant true in
  let* c = frequency [ (3, int_range 1 2); (1, int_range 3 9) ] in
  let* rows =
    frequency [ (1, constant 0); (1, constant 1); (4, int_range 2 9) ]
  in
  let+ data = string_of ~size:(constant (k * c * rows)) char in
  (* Each container's bits below its top [valid] are cleared. *)
  let low = (8 * k) - valid in
  let data =
    String.mapi
      (fun i ch ->
        let bit = 8 * (i mod k) in
        let keep =
          if bit + 8 <= low then 0
          else if bit >= low then 0xff
          else 0xff lxor ((1 lsl (low - bit)) - 1)
        in
        Char.chr (Char.code ch land keep))
      data
  in
  { k; valid; c; rows; ext; data }

let pcm_gen =
  Gen.with_pp
    (fun ppf p ->
      Format.fprintf ppf "%d valid bits in %d bytes, %d channels, %d rows, %s"
        p.valid p.k p.c p.rows
        (if p.ext then "extensible" else "plain"))
    pcm_gen

let pcm_file p =
  let fmt =
    if p.ext then
      extensible ~mask:0x5 ~code:1 ~channels:p.c ~container:p.k p.valid
    else plain ~code:1 ~channels:p.c ~container:p.k p.valid
  in
  file ~suffix:".wav" (wav fmt p.data)

let containers =
  prop "PCM loads as its containers in the top bits of its dtype" pcm_gen
    (fun p ->
      cover "1 byte" (p.k = 1);
      cover "3 bytes" (p.k = 3);
      cover "5 to 7 bytes" (p.k >= 5 && p.k <= 7);
      cover "8 bytes" (p.k = 8);
      cover "fewer valid bits than the container's" (p.valid < 8 * p.k);
      cover "plain" (not p.ext);
      cover "more than 2 channels" (p.c > 2);
      cover "no row" (p.rows = 0);
      let a = Nx_io.load_audio (pcm_file p) in
      let mask =
        if p.ext then 0x5 else match p.c with 1 -> 4 | 2 -> 3 | _ -> 0
      in
      equal facts_t ((8000, p.rows, p.c), (p.valid, mask)) (facts a);
      equal packed (expected ~k:p.k ~c:p.c p.data) (Audio.samples a))

let floats =
  prop "IEEE floats load as stored, every bit kept"
    Gen.(
      pair (int_range 0 1)
        (pair (int_range 0 5) (string_of ~size:(constant 80) char)))
    (fun (double, (rows, bits)) ->
      let k = if double = 1 then 8 else 4 in
      let c = 2 in
      let data = String.sub bits 0 (rows * c * k) in
      let dt = if k = 8 then D.Any D.Float64 else D.Any D.Float32 in
      let (D.Any dt) = dt in
      List.iter
        (fun fmt ->
          let a = Nx_io.load_audio (file (wav fmt data)) in
          equal facts_t ((8000, rows, c), (8 * k, 3)) (facts a);
          equal packed (of_le dt [| rows; c |] data) (Audio.samples a))
        [
          plain ~code:3 ~channels:c ~container:k (8 * k);
          extensible ~mask:3 ~code:3 ~channels:c ~container:k (8 * k);
        ])

(* A signalling NaN, a NaN with a payload, -0. and the infinities, at both
   widths. *)
let float_specials =
  test "float specials load bit for bit" (fun () ->
      let f32 = [ 0x7f800001l; 0xffc00123l; 0x80000000l; 0x7f800000l ] in
      let data =
        String.concat ""
          (List.map (fun v -> le 4 (Int32.to_int v land 0xffffffff)) f32)
      in
      equal packed
        (of_le D.Float32 [| 2; 2 |] data)
        (load_one (file (wav (plain ~code:3 ~channels:2 ~container:4 32) data)));
      let f64 =
        [
          0x7ff0000000000001L;
          0xfff8000000000abcL;
          Int64.min_int;
          0x7ff0000000000000L;
        ]
      in
      let data =
        String.concat ""
          (List.map
             (fun v ->
               String.init 8 (fun i ->
                   Char.chr
                     (Int64.to_int (Int64.shift_right_logical v (8 * i))
                     land 0xff)))
             f64)
      in
      equal packed
        (of_le D.Float64 [| 2; 2 |] data)
        (load_one (file (wav (plain ~code:3 ~channels:2 ~container:8 64) data))))

(* G.711 *)

let g711 =
  cases "every G.711 code loads as libsndfile reads it" ~id:fst
    [ ("A-law", (6, "wav_alaw", 13)); ("mu-law", (7, "wav_mulaw", 14)) ]
    (fun (_, (code, name, bits)) ->
      let data = String.init 256 Char.chr in
      List.iter
        (fun fmt ->
          let a = Nx_io.load_audio (file (wav fmt data)) in
          equal facts_t ((8000, 256, 1), (bits, 4)) (facts a);
          equal packed
            (Nx_io.load_npy (fixture (name ^ ".npy")))
            (Audio.samples a))
        [
          plain ~code ~channels:1 ~container:1 8;
          extensible ~mask:4 ~code ~channels:1 ~container:1 8;
        ];
      equal packed
        (Nx_io.load_npy (fixture (name ^ ".npy")))
        (load_one (fixture (name ^ ".wav"))))

(* libsndfile *)

let libsndfile =
  cases "files libsndfile wrote load as libsndfile reads them" ~id:Fun.id
    [
      "wav_sf_pcm_u8";
      "wav_sf_pcm_16";
      "wav_sf_pcm_24";
      "wav_sf_pcm_32";
      "wav_sf_float";
      "wav_sf_double";
      "wavex_sf_pcm24";
    ] (fun name ->
      let a = Nx_io.load_audio (fixture (name ^ ".wav")) in
      equal int 44100 (Audio.rate a);
      equal packed (Nx_io.load_npy (fixture (name ^ ".npy"))) (Audio.samples a))

(* Layout *)

(* [junk n] is a chunk of [n] bytes a load skips. *)
let junk n = chunk "JUNK" (String.make n 'j')

let layout =
  group "layout"
    [
      test "float32 at a multiple of 4 bytes loads over the file's pages"
        (fun () ->
          let data = String.make 8 '\001' in
          let fmt = chunk "fmt " (plain ~code:3 ~channels:2 ~container:4 32) in
          let s = riff [ fmt; chunk "data" data ] in
          equal int 0 ((String.length s - 8) mod 4);
          equal memory Over_file (memory_of (load_one (file s))));
      test "float32 at 2 bytes past a multiple of 4 loads into its own memory"
        (fun () ->
          let data = String.make 8 '\001' in
          let fmt = chunk "fmt " (plain ~code:3 ~channels:2 ~container:4 32) in
          let s = riff [ fmt; junk 2; chunk "data" data ] in
          equal int 2 ((String.length s - 8) mod 4);
          let x = load_one (file s) in
          equal memory Own_memory (memory_of x);
          equal packed (of_le D.Float32 [| 1; 2 |] data) x);
      test "a chunk of an odd size is followed by its pad byte" (fun () ->
          let data = "\001\002" in
          let s =
            riff
              [
                junk 1;
                chunk "fmt " (plain ~code:1 ~channels:1 ~container:2 16);
                junk 5;
                chunk "data" data;
              ]
          in
          equal packed (of_le D.Int16 [| 1; 1 |] data) (load_one (file s)));
      test "a 24-bit file loads into memory of its own" (fun () ->
          let path =
            file
              (wav (plain ~code:1 ~channels:1 ~container:3 24) "\001\002\003")
          in
          equal memory Own_memory (memory_of (load_one path)));
      test "nothing after data is read" (fun () ->
          let s =
            wav (plain ~code:1 ~channels:1 ~container:2 16) "\001\002"
            ^ "LIST\255\255\255\255"
          in
          let a = Nx_io.load_audio (file s) in
          equal int 1 (Audio.length a));
      test "a data size of 0xFFFFFFFF runs to the end of the file" (fun () ->
          let fmt = chunk "fmt " (plain ~code:1 ~channels:2 ~container:2 16) in
          let data = "\001\002\003\004\005\006\007\008" in
          let s = riff [ fmt ] ^ "data" ^ le 4 0xffffffff ^ data in
          let a = Nx_io.load_audio (file s) in
          equal int 2 (Audio.length a);
          equal packed (of_le D.Int16 [| 2; 2 |] data) (Audio.samples a));
      test "a data size of 0 is no row" (fun () ->
          let s =
            riff
              [
                chunk "fmt " (plain ~code:1 ~channels:1 ~container:2 16);
                chunk "data" "";
              ]
            ^ "\001\002"
          in
          equal int 0 (Audio.length (Nx_io.load_audio (file s))));
    ]

(* RF64 *)

(* An RF64 file whose ds64 chunk states the data's size, with the data chunk's
   size 0xFFFFFFFF, and a JUNK chunk whose size the table states. *)
let rf64 data =
  let fmt = chunk "fmt " (plain ~code:1 ~channels:2 ~container:2 16) in
  let n = String.length data in
  let junk_body = "jjjjjj" in
  let ds64 =
    chunk "ds64"
      (le 8 0 ^ le 8 n
      ^ le 8 (n / 4)
      ^ le 4 1 ^ "JUNK"
      ^ le 8 (String.length junk_body))
  in
  "RF64" ^ le 4 0xffffffff ^ "WAVE" ^ ds64 ^ fmt ^ "JUNK" ^ le 4 0xffffffff
  ^ junk_body ^ "data" ^ le 4 0xffffffff ^ data ^ "trailing"

let rf64_cases =
  group "RF64"
    [
      test "the ds64 chunk states the sizes of 0xFFFFFFFF" (fun () ->
          let data = "\001\002\003\004\005\006\007\008" in
          let a = Nx_io.load_audio (file (rf64 data)) in
          equal int 2 (Audio.length a);
          equal packed (of_le D.Int16 [| 2; 2 |] data) (Audio.samples a));
      test "an RF64 file without a ds64 chunk first" (fun () ->
          let s =
            wav ~id:"RF64"
              (plain ~code:1 ~channels:1 ~container:2 16)
              "\001\002"
          in
          let path = file s in
          fails ~naming:"ds64" ~path (fun () -> Nx_io.load_audio path));
    ]

(* Refusals *)

let pcm16 = plain ~code:1 ~channels:1 ~container:2 16

let refused =
  cases "a file a load does not read raises naming it" ~id:fst
    [
      ("a RIFX form", (wav ~id:"RIFX" pcm16 "\000\001", "RIFX"));
      ( "an Ogg stream",
        ("OggS\000\002" ^ String.make 30 '\000', "neither a WAV nor a FLAC") );
      ("no data chunk", (riff [ chunk "fmt " pcm16 ], "no data chunk"));
      ( "fmt after data",
        (riff [ chunk "data" "\000\001"; chunk "fmt " pcm16 ], "before the fmt")
      );
      ( "two fmt chunks",
        ( riff [ chunk "fmt " pcm16; chunk "fmt " pcm16; chunk "data" "" ],
          "two fmt" ) );
      ( "a short fmt chunk",
        ( riff [ chunk "fmt " (String.sub pcm16 0 14); chunk "data" "" ],
          "fewer than 16" ) );
      ( "no channel",
        (wav (plain ~code:1 ~channels:0 ~container:2 16) "", "no channel") );
      ( "a block that is not whole containers",
        ( wav (le 2 1 ^ le 2 2 ^ le 4 8000 ^ le 4 24000 ^ le 2 3 ^ le 2 8) "",
          "blocks of 3" ) );
      ( "containers of 9 bytes",
        (wav (plain ~code:1 ~channels:1 ~container:9 72) "", "blocks of 9") );
      ( "24 bits in 4 bytes, plain, as Cool Edit writes floats",
        (wav (plain ~code:1 ~channels:1 ~container:4 24) "", "24 bits in 4") );
      ( "17 bits in 2 bytes",
        (wav (plain ~code:1 ~channels:1 ~container:2 17) "", "17 bits in 2") );
      ( "0 valid bits",
        (wav (plain ~code:1 ~channels:1 ~container:2 0) "", "0 valid") );
      ( "extensible, 0 valid bits",
        (wav (extensible ~code:1 ~channels:1 ~container:2 0) "", "0 valid") );
      ( "extensible, valid bits past the container",
        (wav (extensible ~code:1 ~channels:1 ~container:2 17) "", "17 bits in 2")
      );
      ( "extensible, a container of other bits",
        ( wav
            (String.sub (extensible ~code:1 ~channels:1 ~container:2 16) 0 14
            ^ le 2 24
            ^ String.sub (extensible ~code:1 ~channels:1 ~container:2 16) 16 24
            )
            "",
          "24 bits in containers of 2" ) );
      ( "extensible in fewer than 40 bytes",
        ( wav
            (String.sub (extensible ~code:1 ~channels:1 ~container:2 16) 0 38)
            "",
          "fewer than 40" ) );
      ( "extensible, an unknown SubFormat",
        ( wav
            (String.sub (extensible ~code:1 ~channels:1 ~container:2 16) 0 39
            ^ "\x72")
            "",
          "SubFormat" ) );
      ("ADPCM", (wav (plain ~code:2 ~channels:1 ~container:2 4) "", "format 2"));
      ( "float of 16 bits",
        (wav (plain ~code:3 ~channels:1 ~container:2 16) "", "format 3") );
      ( "float of 3 bytes",
        (wav (plain ~code:3 ~channels:1 ~container:3 24) "", "format 3") );
      ( "A-law of 2 bytes",
        (wav (plain ~code:6 ~channels:1 ~container:2 16) "", "format 6") );
      ( "data holding part of a row",
        ( wav (plain ~code:1 ~channels:2 ~container:2 16) "\001\002\003",
          "part of a row" ) );
      ( "a data size of 0x80000000, as arecord writes to a pipe",
        ( riff [ chunk "fmt " pcm16 ] ^ "data" ^ le 4 0x80000000 ^ "\001\002",
          "states 2147483648 bytes" ) );
      ( "a data size of 0x7ffff000, as sox once wrote",
        ( riff [ chunk "fmt " pcm16 ] ^ "data" ^ le 4 0x7ffff000 ^ "\001\002",
          "states 2147479552 bytes" ) );
      ( "a chunk past the end",
        ( riff [ chunk "fmt " pcm16 ] ^ "LIST" ^ le 4 100 ^ "ab",
          "ends past the file" ) );
      ( "an RF64 size the ds64 chunk does not state",
        ( (let s = rf64 "\001\002\003\004" in
           (* The table's chunk is renamed. *)
           let i = 12 + 8 + 28 in
           String.sub s 0 i ^ "KNUJ"
           ^ String.sub s (i + 4) (String.length s - i - 4)),
          "no size for \"JUNK\"" ) );
    ]
    (fun (_, (s, naming)) ->
      let path = file ~suffix:".wav" s in
      fails ~naming ~path (fun () -> Nx_io.load_audio path))

(* Valid bits *)

(* 20 valid bits in containers of 3 bytes, and 24 in 4: a container whose bits
   below them are set raises naming its row. *)
let low_bits =
  cases "a set bit below the valid bits raises naming the row" ~id:fst
    [
      ("20 bits in 3 bytes", (3, 20));
      ("24 bits in 4 bytes", (4, 24));
      ("12 bits in 2 bytes", (2, 12));
    ]
    (fun (_, (k, valid)) ->
      let clean = String.make (k * 2 * 3) '\000' in
      let fmt = extensible ~code:1 ~channels:2 ~container:k valid in
      let a = Nx_io.load_audio (file (wav fmt clean)) in
      equal int 3 (Audio.length a);
      (* Row 2's second sample has its lowest bit set. *)
      let dirty = Bytes.of_string clean in
      Bytes.set dirty (k * 5) '\001';
      let path = file (wav fmt (Bytes.to_string dirty)) in
      let a = Nx_io.load_audio path in
      fails ~naming:"row 2" ~path (fun () -> Audio.samples a))

(* Windows and blocks *)

let pieces =
  List.map
    (fun name -> fixture (name ^ ".wav"))
    [
      "wav_sf_pcm_u8";
      "wav_sf_pcm_24";
      "wav_sf_double";
      "wavex_sf_pcm24";
      "wav_alaw";
    ]

let whole = List.map (fun p -> (p, load_one p)) pieces
let slice (Nx.P x) ~row ~rows = Nx.P (Nx.slice [ Nx.R (row, row + rows) ] x)

let piece =
  let open Gen in
  let* path, (Nx.P x as p) =
    of_list ~pp:(fun ppf (p, _) -> Format.pp_print_string ppf p) whole
  in
  let n = (Nx.shape x).(0) in
  let* row = int_range 0 n in
  let+ rows = int_range 0 (n - row) and+ k = int_range 1 (n + 1) in
  (path, p, row, rows, k)

let windows =
  prop "a window is the slice of the samples it covers" piece
    (fun (path, x, row, rows, _) ->
      cover "no row" (rows = 0);
      cover "from a row past 0" (row > 0 && rows > 0);
      let w = Audio.window (Nx_io.load_audio path) ~row ~rows in
      equal memory Own_memory (memory_of w);
      equal packed (slice x ~row ~rows) w)

let joined =
  prop "blocks joined are the samples from their row" piece
    (fun (path, (Nx.P x as p), row, _, rows) ->
      let n = (Nx.shape x).(0) in
      cover "several blocks" (n - row > rows);
      let bs = List.of_seq (Audio.blocks ~row (Nx_io.load_audio path) ~rows) in
      equal int ((n - row + rows - 1) / rows) (List.length bs);
      match bs with
      | [] -> equal int n row
      | bs ->
          let dt = Nx.dtype x in
          equal packed
            (slice p ~row ~rows:(n - row))
            (Nx.P (Nx.concatenate ~axis:0 (List.map (Nx.unpack dt) bs))))

let read_rows =
  test "a window reads its rows, and a set bit below the valid ones raises"
    (fun () ->
      (* 20 valid bits in 3 bytes, row 5 of 8 with its lowest bit set. *)
      let data = Bytes.make (3 * 8) '\000' in
      Bytes.set data (3 * 5) '\001';
      let path =
        file
          (wav
             (extensible ~code:1 ~channels:1 ~container:3 20)
             (Bytes.to_string data))
      in
      let a = Nx_io.load_audio path in
      ignore (Audio.window a ~row:0 ~rows:5);
      ignore (Audio.window a ~row:6 ~rows:2);
      fails ~naming:"row 5" ~path (fun () -> Audio.window a ~row:4 ~rows:2);
      let bs = Audio.blocks a ~rows:3 in
      ignore (Seq.take 1 bs |> List.of_seq);
      fails ~naming:"row 5" ~path (fun () -> List.of_seq bs);
      invalid ~by:"Nx_io.Audio.window" (fun () -> Audio.window a ~row:7 ~rows:2);
      invalid ~by:"Nx_io.Audio.blocks" (fun () -> Audio.blocks a ~rows:0))

(* Saving *)

(* A case to save: a dtype, rows and channels, the significant bits of an
   integer sample, a channel mask if stated, and a view of the samples. *)
type save = { case : case; bits : int option; mask : int option; rate : int }

(* [clear dt low v] is [v] with its [low] lowest bits cleared. *)
let clear (type v s) (dt : (v, s) D.t) low (v : v) : v =
  match dt with
  | D.Int32 -> Int32.(shift_left (shift_right v low) low)
  | D.Int64 -> Int64.(shift_left (shift_right v low) low)
  | D.Int16 -> (v asr low) lsl low
  | D.Uint8 -> (v asr low) lsl low
  | _ -> v

let save_dtypes =
  D.[ Any Uint8; Any Int16; Any Int32; Any Int64; Any Float32; Any Float64 ]

let saves =
  let open Gen in
  let* (D.Any dt) =
    of_list ~pp:(fun ppf (D.Any dt) -> D.pp ppf dt) save_dtypes
  in
  let* n = frequency [ (1, constant 0); (1, constant 1); (4, int_range 2 9) ] in
  let* c = frequency [ (3, int_range 1 2); (1, int_range 3 9) ] in
  let w = D.bytes dt 1 in
  let* bits =
    if D.is D.Float dt then constant None
    else
      frequency
        [ (1, constant None); (2, map Option.some (int_range 1 (8 * w))) ]
  in
  let* mask =
    frequency [ (2, constant None); (1, map Option.some (int_range 0 0x3ffff)) ]
  in
  let* rate = int_range 1 192000 in
  let* vs = array ~size:(constant (n * c)) (element dt) in
  let+ view = of_list [ Plain; Transposed; Broadcast ] in
  (* An integer's bits below its top [bits] are cleared. *)
  let vs =
    match bits with
    | Some b -> Array.map (clear dt ((8 * w) - b)) vs
    | None -> vs
  in
  let case =
    match view with
    | Transposed -> Case (dt, [| c; n |], vs, Transposed)
    | Broadcast when n > 0 -> Case (dt, [| 1; c |], Array.sub vs 0 c, Broadcast)
    | _ -> Case (dt, [| n; c |], vs, Plain)
  in
  { case; bits; mask; rate }

let saves =
  Gen.with_pp
    (fun ppf s ->
      let (Case (dt, sh, _, v)) = s.case in
      Format.fprintf ppf "%a %a %a, bits %s, mask %s" D.pp dt pp_shape sh
        pp_view v
        (Option.fold ~none:"-" ~some:string_of_int s.bits)
        (Option.fold ~none:"-" ~some:string_of_int s.mask))
    saves

(* The samples of a case, of shape [[|n; c|]]. *)
let samples_of s =
  let (Case (_, sh, _, view)) = s.case in
  let (Nx.P x) = value s.case in
  match view with
  | Broadcast -> Nx.P (Nx.reshape [| 2 * sh.(0); sh.(1) |] x)
  | _ -> Nx.P x

let round_trips =
  prop "a save loads back as its samples, rate, bits and mask" saves (fun s ->
      let (Nx.P x) = samples_of s in
      let n, c = match Nx.shape x with [| n; c |] -> (n, c) | _ -> (0, 0) in
      let w = D.bytes (Nx.dtype x) 1 in
      cover "no row" (n = 0);
      cover "more than 2 channels" (c > 2);
      cover "fewer bits than the dtype's"
        (Option.exists (fun b -> b < 8 * w) s.bits);
      cover "a stated mask" (s.mask <> None);
      cover "strided"
        (match s.case with
        | Case (_, _, _, Transposed) -> n > 1 && c > 1
        | _ -> false);
      cover "floats" (D.is D.Float (Nx.dtype x));
      let path = temp_file ~suffix:".wav" () in
      Nx_io.save_wav ?bits:s.bits ?channel_mask:s.mask path ~rate:s.rate x;
      let a = Nx_io.load_audio path in
      let mask =
        Option.value s.mask ~default:(match c with 1 -> 4 | 2 -> 3 | _ -> 0)
      in
      equal facts_t
        ((s.rate, n, c), (Option.value s.bits ~default:(8 * w), mask))
        (facts a);
      equal packed (Nx.P x) (Audio.samples a))

let saved path = Nx_io.load_audio path

let save_cases =
  group "saving"
    [
      test "a save of 4- and 8-byte samples loads over the file's pages"
        (fun () ->
          List.iter
            (fun (Nx.P x) ->
              List.iter
                (fun c ->
                  let x = Nx.reshape [| 6 / c; c |] x in
                  let path = temp_file ~suffix:".wav" () in
                  Nx_io.save_wav path ~rate:8000 x;
                  equal memory Over_file
                    (memory_of (Audio.samples (saved path))))
                [ 1; 2; 3 ])
            [
              Nx.P (Nx.create Nx.int16 [| 6 |] [| 1; 2; 3; 4; 5; 6 |]);
              Nx.P (Nx.create Nx.int32 [| 6 |] [| 1l; 2l; 3l; 4l; 5l; 6l |]);
              Nx.P (Nx.create Nx.float32 [| 6 |] [| 1.; 2.; 3.; 4.; 5.; 6. |]);
              Nx.P (Nx.create Nx.float64 [| 6 |] [| 1.; 2.; 3.; 4.; 5.; 6. |]);
              Nx.P (Nx.create Nx.int64 [| 6 |] [| 1L; 2L; 3L; 4L; 5L; 6L |]);
            ]);
      test "24 bits of int32 take 3 bytes a sample" (fun () ->
          let x =
            Nx.create Nx.int32 [| 2; 2 |]
              [| 256l; -256l; 0x7fffff00l; Int32.min_int |]
          in
          let path = temp_file ~suffix:".wav" () in
          Nx_io.save_wav ~bits:24 path ~rate:48000 x;
          let s = read path in
          equal int 12 (String.length s - String.index_from s 36 'd' - 8);
          equal packed (Nx.P x) (Audio.samples (saved path)));
      test "a set bit below the bits raises naming the row and leaves the file"
        (fun () ->
          let path = file ~suffix:".wav" "before" in
          let x = Nx.create Nx.int32 [| 3; 1 |] [| 256l; 512l; 513l |] in
          raises_match (Exn.invalid_arg ~sub:"Nx_io.save_wav: row 2") (fun () ->
              Nx_io.save_wav ~bits:24 path ~rate:8000 x);
          equal string "before" (read path));
      test "a FLAC file saves as the WAV file that loads as it" (fun () ->
          List.iter
            (fun name ->
              let a = Nx_io.load_audio name in
              let (Nx.P x) = Audio.samples a in
              let path = temp_file ~suffix:".wav" () in
              Nx_io.save_wav ~bits:(Audio.bits a) path ~rate:(Audio.rate a) x;
              let b = saved path in
              equal facts_t (facts a) (facts b);
              equal packed (Nx.P x) (Audio.samples b))
            [
              fixture "flac_depth_12.flac";
              fixture "flac_depth_20.flac";
              fixture "flac_depth_24.flac";
              fixture "flac_depth_32.flac";
              Filename.concat "cut" "ietf_subset_37.flac";
            ]);
      cases "a value WAV does not store raises" ~id:fst
        [
          ( "a vector",
            (Nx.P (Nx.zeros Nx.int16 [| 4 |]), None, None, 8000, "1 axes") );
          ( "no channel",
            (Nx.P (Nx.zeros Nx.int16 [| 4; 0 |]), None, None, 8000, "no channel")
          );
          ( "int8",
            (Nx.P (Nx.zeros Nx.int8 [| 4; 1 |]), None, None, 8000, "unsigned")
          );
          ( "bool",
            (Nx.P (Nx.zeros Nx.bool [| 4; 1 |]), None, None, 8000, "no bool") );
          ( "float16",
            ( Nx.P (Nx.zeros Nx.float16 [| 4; 1 |]),
              None,
              None,
              8000,
              "no float16" ) );
          ( "a row past 65535 bytes",
            ( Nx.P (Nx.zeros Nx.int16 [| 1; 32768 |]),
              None,
              None,
              1,
              "65536 bytes" ) );
          ( "a rate of 0",
            (Nx.P (Nx.zeros Nx.int16 [| 4; 1 |]), None, None, 0, "rate of 0") );
          ( "a rate of 2^32",
            ( Nx.P (Nx.zeros Nx.uint8 [| 4; 1 |]),
              None,
              None,
              1 lsl 32,
              "rate of" ) );
          ( "bytes a second of 2^32",
            ( Nx.P (Nx.zeros Nx.int16 [| 4; 2 |]),
              None,
              None,
              1 lsl 30,
              "rate of" ) );
          ( "0 bits",
            (Nx.P (Nx.zeros Nx.int16 [| 4; 1 |]), Some 0, None, 8000, "0 bits")
          );
          ( "17 bits of int16",
            (Nx.P (Nx.zeros Nx.int16 [| 4; 1 |]), Some 17, None, 8000, "17 bits")
          );
          ( "bits of floats",
            ( Nx.P (Nx.zeros Nx.float32 [| 4; 1 |]),
              Some 24,
              None,
              8000,
              "bits is given" ) );
          ( "a negative mask",
            (Nx.P (Nx.zeros Nx.int16 [| 4; 1 |]), None, Some (-1), 8000, "mask")
          );
          ( "a mask of 33 bits",
            ( Nx.P (Nx.zeros Nx.int16 [| 4; 1 |]),
              None,
              Some (1 lsl 32),
              8000,
              "mask" ) );
        ]
        (fun (_, (Nx.P x, bits, channel_mask, rate, sub)) ->
          let path = temp_file ~suffix:".wav" () in
          raises_match (Exn.invalid_arg ~sub:"Nx_io.save_wav: ") (fun () ->
              Nx_io.save_wav ?bits ?channel_mask path ~rate x);
          raises_match (Exn.invalid_arg ~sub) (fun () ->
              Nx_io.save_wav ?bits ?channel_mask path ~rate x));
    ]

(* Corruption *)

let small =
  riff
    [
      chunk "fmt " (extensible ~code:1 ~channels:2 ~container:3 20);
      junk 3;
      chunk "data" (String.make 12 '\000');
    ]

let corrupted =
  group "corrupted files"
    [
      test "every proper prefix of a file raises" (fun () ->
          prefixes_fail Nx_io.load_audio small);
      test "a flipped byte loads or raises Failure" (fun () ->
          flips_load_or_fail (fun p -> Audio.samples (Nx_io.load_audio p)) small);
    ]

let () =
  exit
    (run "nx.io WAV"
       [
         containers;
         floats;
         float_specials;
         g711;
         libsndfile;
         layout;
         rf64_cases;
         refused;
         low_bits;
         windows;
         joined;
         read_rows;
         round_trips;
         save_cases;
         corrupted;
       ])
