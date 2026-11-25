(* Every codec against what its goldens become when cut or mutated, and
   every whole-buffer call against a source and destination that overlap.
   The format-specific refusals (sizes that lie, distances before the
   output, reserved bits) are stated in each codec's own suite. *)

open Windtrap

(* The codecs whose decoders exist, as Lz_support.golden.codec names them. *)
let codecs = [ "deflate"; "zlib"; "gzip"; "zstd"; "lz4b"; "lz4f"; "snappy" ]

(* [decode codec g s] decodes [s] as [codec] into room for [g]'s output. *)
let decode codec g s =
  let n = String.length (Lazy.force g.Lz_support.raw) in
  Lz_support.decoder codec (Lz_support.of_string s) (Lz_support.create n)

(* Cut goldens *)

let truncations codec =
  cases "every proper prefix is Truncated at its length"
    ~id:(fun g -> g.Lz_support.file)
    (List.filter
       (fun g -> g.Lz_support.members = 1 && g.size <= 1024)
       (Lz_support.goldens_of codec))
    (fun g ->
      let data = Lazy.force g.data in
      for cut = 0 to String.length data - 1 do
        equal
          (result Lz_support.progress Lz_support.error)
          ~msg:(Printf.sprintf "cut at %d" cut)
          (Error { Lz.at = cut; why = Truncated })
          (decode codec g (String.sub data 0 cut))
      done)

(* Mutated goldens *)

(* A decode of mutated bytes may succeed or fail; it must stay inside its
   buffers, which the sanitizers check, and report offsets inside them. *)
let within data r =
  let len = String.length data in
  match r with
  | Ok { Lz.consumed; produced = _ } -> at_most int ~than:len consumed
  | Error { Lz.at; why = _ } ->
      at_least int ~than:0 at;
      at_most int ~than:len at

let up_to limit codec =
  List.filter
    (fun g -> g.Lz_support.size <= limit)
    (Lz_support.goldens_of codec)

let flips codec =
  cases "a bit flipped decodes or fails inside the input"
    ~id:(fun g -> g.Lz_support.file)
    (up_to 512 codec)
    (fun g ->
      let data = Lazy.force g.data in
      for bit = 0 to (8 * String.length data) - 1 do
        let b = Bytes.of_string data in
        let i = bit / 8 in
        Bytes.set b i (Char.chr (Char.code data.[i] lxor (1 lsl (bit mod 8))));
        let m = Bytes.to_string b in
        within m (decode codec g m)
      done)

let replaced codec =
  let pool = Array.of_list (up_to 65536 codec) in
  let edges =
    match pool with [||] -> [] | _ -> [ (0, 0, '\x07'); (0, -1, '\x07') ]
  in
  prop "a byte replaced decodes or fails inside the input" ~examples:edges
    Gen.(
      triple
        (int_range 0 (max 0 (Array.length pool - 1)))
        Lz_support.anywhere char)
    (fun (k, at, c) ->
      if Array.length pool = 0 then skip ();
      let g = pool.(k) in
      let data = Lazy.force g.data in
      assume (String.length data > 0);
      let b = Bytes.of_string data in
      Bytes.set b (Lz_support.position at (String.length data)) c;
      let m = Bytes.to_string b in
      within m (decode codec g m))

(* A raw LZ4 block has no end to miss, so ["lz4b"] has no truncations. *)
let suite codec =
  let truncations = if codec = "lz4b" then [] else [ truncations codec ] in
  group codec
    (truncations @ [ group "mutations" [ flips codec; replaced codec ] ])

let manifest =
  test "every golden names a decoder" (fun () ->
      List.iter
        (fun g -> ignore (Lz_support.decoder g.Lz_support.codec : _ -> _ -> _))
        (Lz_support.goldens ()))

(* Overlapping buffers *)

type whole = {
  name : string;
  decode : Lz.bigbytes -> Lz.bigbytes -> unit;
  encode : Lz.bigbytes -> Lz.bigbytes -> unit;
}

let whole name decode encode =
  {
    name;
    decode = (fun s d -> ignore (decode s d : (Lz.progress, Lz.error) result));
    encode = (fun s d -> ignore (encode s d : int option));
  }

let wholes =
  [
    whole "deflate" Lz.Deflate.decode (fun s d -> Lz.Deflate.encode s d);
    whole "zlib" Lz.Zlib.decode (fun s d -> Lz.Zlib.encode s d);
    whole "gzip" Lz.Gzip.decode (fun s d -> Lz.Gzip.encode s d);
    whole "zstd" Lz.Zstd.decode (fun s d -> Lz.Zstd.encode s d);
    whole "lz4 block" Lz.Lz4.Block.decode Lz.Lz4.Block.encode;
    whole "lz4 frame" Lz.Lz4.Frame.decode Lz.Lz4.Frame.encode;
    whole "snappy" Lz.Snappy.decode Lz.Snappy.encode;
  ]

(* Views of one buffer of 4 KiB, as (name, src, dst): sharing from all their
   bytes to one, then sharing none, side by side or empty. *)
let views () =
  let b = Lz_support.create (4 * 1024) in
  let sub first n = Bigarray.Array1.sub b first n in
  let same = sub 512 1024 in
  ( [
      ("the same view", same, same);
      ("identical views", sub 512 1024, sub 512 1024);
      ("src before dst", sub 0 2048, sub 1024 2048);
      ("dst before src", sub 1024 2048, sub 0 2048);
      ("dst inside src", sub 0 4096, sub 1024 16);
      ("one shared byte", sub 0 2049, sub 2048 2048);
    ],
    [
      ("side by side", sub 0 2048, sub 2048 2048);
      ("an empty src inside dst", sub 1024 0, sub 0 2048);
    ] )

let overlap =
  cases "decode and encode raise on overlapping buffers"
    ~id:(fun w -> w.name)
    wholes
    (fun w ->
      let shared, _ = views () in
      List.iter
        (fun (name, src, dst) ->
          let raise_on call f =
            raises_match ~msg:(name ^ ", " ^ call)
              (Exn.invalid_arg ~sub:"overlap")
              (fun () -> f src dst)
          in
          raise_on "decode" w.decode;
          raise_on "encode" w.encode)
        shared)

let apart =
  cases "decode and encode take buffers side by side or empty"
    ~id:(fun w -> w.name)
    wholes
    (fun w ->
      let _, apart = views () in
      List.iter
        (fun (_, src, dst) ->
          w.decode src dst;
          w.encode src dst)
        apart)

let () =
  exit
    (run "Hostile" ((manifest :: List.map suite codecs) @ [ overlap; apart ]))
