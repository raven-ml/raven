(* Every codec against what its goldens become when cut or mutated, and
   every whole-buffer call against a source and destination that overlap.
   The format-specific refusals (sizes that lie, distances before the
   output, reserved bits) are stated in each codec's own suite. *)

open Windtrap

(* The codecs whose decoders exist, as Lz_support.golden.codec names them. *)
let codecs = [ "deflate"; "zlib"; "gzip"; "zstd"; "lz4b"; "lz4f"; "snappy" ]

let suite codec =
  let truncations =
    if codec = "lz4b" then [] else [ Lz_support.truncations codec ]
  in
  group codec (truncations @ [ Lz_support.mutations codec ])

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
  cases "decode and encode refuse overlapping buffers"
    ~id:(fun w -> w.name)
    wholes
    (fun w ->
      let shared, apart = views () in
      List.iter
        (fun (name, src, dst) ->
          let raise_on call f =
            raises_match ~msg:(name ^ ", " ^ call)
              (Exn.invalid_arg ~sub:"overlap")
              (fun () -> f src dst)
          in
          raise_on "decode" w.decode;
          raise_on "encode" w.encode)
        shared;
      List.iter
        (fun (_, src, dst) ->
          w.decode src dst;
          w.encode src dst)
        apart)

let () =
  exit (run "Hostile" ((manifest :: List.map suite codecs) @ [ overlap ]))
