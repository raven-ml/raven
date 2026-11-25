(* Every codec against what its goldens become when cut or mutated, and every
   whole-buffer call against a source and destination that overlap. The
   format-specific refusals (sizes that lie, distances before the output,
   reserved bits) are stated in each codec's own suite. *)

open Windtrap

(* The codecs whose decoders exist, as Lz_support.golden.codec names them. *)
let codecs = [ "deflate"; "zlib"; "gzip"; "zstd"; "lz4b"; "lz4f"; "snappy" ]
let file g = g.Lz_support.file

(* [decode g s] decodes [s] as [g]'s codec into room for [g]'s output. *)
let decode g s =
  let n = String.length (Lazy.force g.Lz_support.raw) in
  Lz_support.decoder g.codec (Lz_support.of_string s) (Lz_support.create n)

(* Cut goldens *)

(* An LZ4 block has no end mark, so a prefix of one may be a whole block. *)
let truncations =
  cases "every proper prefix of a one-member golden is Truncated at its length"
    ~id:file
    (List.filter
       (fun g ->
         g.Lz_support.members = 1 && g.size <= 1024 && g.codec <> "lz4b")
       (Lz_support.goldens ()))
    (fun g ->
      let data = Lazy.force g.data in
      for cut = 0 to String.length data - 1 do
        equal
          (result Lz_support.progress Lz_support.error)
          ~msg:(Printf.sprintf "cut at %d" cut)
          (Error { Lz.at = cut; why = Truncated })
          (decode g (String.sub data 0 cut))
      done)

(* Mutated goldens *)

(* What lz.mli promises of a decode of [g]'s codec over any input [s]: it writes
   inside its destination and consumes all of [s], or for a raw deflate or zlib
   stream up to the byte that holds its last bit; or it refuses [s], Truncated
   at its length or any other error at a byte inside it. The sanitizers check
   that it reads and writes inside its buffers. *)
let promised ~msg g s =
  let len = String.length s in
  let room = String.length (Lazy.force g.Lz_support.raw) in
  match decode g s with
  | Ok { Lz.consumed; produced } -> (
      at_most int ~msg:(msg ^ ", produced") ~than:room produced;
      match g.codec with
      | "deflate" | "zlib" ->
          at_least int ~msg:(msg ^ ", consumed") ~than:1 consumed;
          at_most int ~msg:(msg ^ ", consumed") ~than:len consumed
      | _ -> equal int ~msg:(msg ^ ", consumed") len consumed)
  | Error { at; why = Truncated } ->
      equal int ~msg:(msg ^ ", Truncated at") len at
  | Error { at; why = Malformed _ | Too_long | Unsupported _ } ->
      at_least int ~msg:(msg ^ ", error at") ~than:0 at;
      less int ~msg:(msg ^ ", error at") ~than:len at

let flips =
  cases
    "a golden with a bit flipped decodes, or is refused inside it or Truncated \
     at its end"
    ~id:file
    (List.filter (fun g -> g.Lz_support.size <= 512) (Lz_support.goldens ()))
    (fun g ->
      let data = Lazy.force g.data in
      for bit = 0 to (8 * String.length data) - 1 do
        let b = Bytes.of_string data in
        let i = bit / 8 in
        Bytes.set b i (Char.chr (Char.code data.[i] lxor (1 lsl (bit mod 8))));
        promised ~msg:(Printf.sprintf "bit %d" bit) g (Bytes.to_string b)
      done)

(* A golden of up to 64 KiB, its codec drawn first, so that each codec is drawn
   a seventh of the time. *)
let golden =
  let pp ppf g = Format.pp_print_string ppf (file g) in
  Gen.bind (Gen.of_list codecs) (fun codec ->
      Gen.of_list ~pp
        (List.filter
           (fun g -> g.Lz_support.size <= 65536)
           (Lz_support.goldens_of codec)))

(* The first and last bytes of the first golden of each codec, replaced. *)
let ends =
  List.concat_map
    (fun codec ->
      let g = List.hd (Lz_support.goldens_of codec) in
      [ (g, 0, '\x07'); (g, -1, '\x07') ])
    codecs

(* Each codec is drawn with probability 1/7, so a cover of one misses one run in
   about 10^47 at 700 cases. *)
let replaced =
  prop
    "a golden with a byte replaced decodes, or is refused inside it or \
     Truncated at its end"
    ~count:700 ~examples:ends
    Gen.(triple golden Lz_support.anywhere char)
    (fun (g, i, c) ->
      List.iter (fun codec -> cover codec (g.codec = codec)) codecs;
      let data = Lazy.force g.data in
      let i = Lz_support.position i (String.length data) in
      let b = Bytes.of_string data in
      Bytes.set b i c;
      promised ~msg:(Printf.sprintf "byte %d" i) g (Bytes.to_string b))

(* Overlapping buffers *)

let decoders =
  [
    ("deflate", Lz.Deflate.decode);
    ("zlib", Lz.Zlib.decode);
    ("gzip", Lz.Gzip.decode);
    ("zstd", Lz.Zstd.decode);
    ("lz4 block", Lz.Lz4.Block.decode);
    ("lz4 frame", Lz.Lz4.Frame.decode);
    ("snappy", Lz.Snappy.decode);
    ("lzw", Lz.Lzw.decode);
    ("lzf", Lz.Lzf.decode);
  ]

let encoders =
  [
    ("deflate", fun s d -> Lz.Deflate.encode s d);
    ("zlib", fun s d -> Lz.Zlib.encode s d);
    ("gzip", fun s d -> Lz.Gzip.encode s d);
    ("zstd", fun s d -> Lz.Zstd.encode s d);
    ("lz4 block", Lz.Lz4.Block.encode);
    ("lz4 frame", Lz.Lz4.Frame.encode);
    ("snappy", Lz.Snappy.encode);
  ]

(* Views of one buffer of 4 KiB, as (name, src, dst), that share from all their
   bytes to one. *)
let overlapping () =
  let b = Lz_support.create (4 * 1024) in
  let sub first n = Bigarray.Array1.sub b first n in
  let same = sub 512 1024 in
  [
    ("the same view", same, same);
    ("identical views", sub 512 1024, sub 512 1024);
    ("src before dst", sub 0 2048, sub 1024 2048);
    ("dst before src", sub 1024 2048, sub 0 2048);
    ("dst inside src", sub 0 4096, sub 1024 16);
    ("src inside dst", sub 1024 16, sub 0 4096);
    ("one shared byte", sub 0 2049, sub 2048 2048);
  ]

(* Views of one buffer of 4 KiB that share no byte. *)
let disjoint () =
  let b = Lz_support.create (4 * 1024) in
  let sub first n = Bigarray.Array1.sub b first n in
  [
    ("side by side", sub 0 2048, sub 2048 2048);
    ("an empty src inside dst", sub 1024 0, sub 0 2048);
    ("an empty dst inside src", sub 0 2048, sub 1024 0);
  ]

(* The message tells the refusal of an overlap from an access out of bounds,
   which raises Invalid_argument too. *)
let refuses ~msg f = raises_match ~msg (Exn.invalid_arg ~sub:"overlap") f

let decode_overlap =
  cases "raises Invalid_argument if src and dst overlap" ~id:fst decoders
    (fun (_, decode) ->
      List.iter
        (fun (name, src, dst) -> refuses ~msg:name (fun () -> decode src dst))
        (overlapping ()))

let decode_apart =
  cases "takes a src and dst that share no byte" ~id:fst decoders
    (fun (_, decode) ->
      List.iter
        (fun (_, src, dst) ->
          ignore (decode src dst : (Lz.progress, Lz.error) result))
        (disjoint ()))

let encode_overlap =
  cases "raises Invalid_argument if src and dst overlap" ~id:fst encoders
    (fun (_, encode) ->
      List.iter
        (fun (name, src, dst) -> refuses ~msg:name (fun () -> encode src dst))
        (overlapping ()))

let encode_apart =
  cases "takes a src and dst that share no byte" ~id:fst encoders
    (fun (_, encode) ->
      List.iter
        (fun (_, src, dst) -> ignore (encode src dst : int option))
        (disjoint ()))

let () =
  exit
    (run "Hostile"
       [
         group "Truncated" [ truncations ];
         group "error" [ flips; replaced ];
         group "decode" [ decode_overlap; decode_apart ];
         group "encode" [ encode_overlap; encode_apart ];
       ])
