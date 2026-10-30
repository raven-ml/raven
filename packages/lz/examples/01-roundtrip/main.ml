(* Encoding and decoding whole buffers.

   A whole-buffer call is given all of its input. This program encodes
   a table of prices with each format, into a destination of the format's
   [bound], and decodes the stream back into a destination of the table's
   length, which a real program learns from the container that holds the
   stream, such as a file header. It then compares deflate's levels, and
   shows where a deflate stream ends inside a longer input. *)

open Bigarray

let bytes n = Array1.create int8_unsigned c_layout n

let of_string s =
  Array1.init int8_unsigned c_layout (String.length s) (fun i ->
      Char.code s.[i])

let to_string b = String.init (Array1.dim b) (fun i -> Char.chr b.{i})

(* A CSV table of 2,000 rows, as a program might write to disk. *)
let table =
  let b = Buffer.create 65536 in
  Buffer.add_string b "id,item,price\n";
  for i = 0 to 1999 do
    Printf.bprintf b "%d,item-%d,%d.%02d\n" i (i mod 37) (i * 7 mod 500)
      (i mod 100)
  done;
  Buffer.contents b

(* One format's whole-buffer calls. *)
type format = {
  name : string;
  bound : int -> int;
  encode : Lz.bigbytes -> Lz.bigbytes -> int option;
  decode : Lz.bigbytes -> Lz.bigbytes -> (Lz.progress, Lz.error) result;
}

let formats =
  let open Lz in
  [
    {
      name = "deflate";
      bound = Deflate.bound;
      encode = (fun s d -> Deflate.encode s d);
      decode = Deflate.decode;
    };
    {
      name = "zlib";
      bound = Zlib.bound;
      encode = (fun s d -> Zlib.encode s d);
      decode = Zlib.decode;
    };
    {
      name = "gzip";
      bound = Gzip.bound;
      encode = (fun s d -> Gzip.encode s d);
      decode = Gzip.decode;
    };
    {
      name = "zstd";
      bound = Zstd.bound;
      encode = (fun s d -> Zstd.encode s d);
      decode = Zstd.decode;
    };
    {
      name = "lz4 block";
      bound = Lz4.Block.bound;
      encode = Lz4.Block.encode;
      decode = Lz4.Block.decode;
    };
    {
      name = "lz4 frame";
      bound = Lz4.Frame.bound;
      encode = Lz4.Frame.encode;
      decode = Lz4.Frame.decode;
    };
    {
      name = "snappy";
      bound = Snappy.bound;
      encode = Snappy.encode;
      decode = Snappy.decode;
    };
  ]

let roundtrip src f =
  let n = Array1.dim src in
  (* A destination of [bound n] bytes holds the stream of any [n] bytes, so
     [encode] returns [Some]. *)
  let dst = bytes (f.bound n) in
  let len = Option.get (f.encode src dst) in
  let stream = Array1.sub dst 0 len in
  let out = bytes n in
  match f.decode stream out with
  | Ok { consumed; produced } ->
      Printf.printf "%-9s %5d bytes (bound %5d), consumed %5d, same: %b\n"
        f.name len (f.bound n) consumed
        (produced = n && to_string out = table)
  | Error e -> Format.printf "%-9s %a@." f.name Lz.pp_error e

let () =
  let src = of_string table in
  Printf.printf "The table: %d bytes\n\n" (Array1.dim src);
  List.iter (roundtrip src) formats;

  (* A higher deflate level searches longer for matches. That usually
     writes fewer bytes, and on this table level 9 writes a few more than
     level 6. *)
  print_newline ();
  List.iter
    (fun level ->
      Printf.printf "deflate level %d: %5d bytes\n" level
        (String.length (Lz.Deflate.encode_string ~level table)))
    [ 1; 6; 9 ];

  (* A deflate stream ends at an exact byte. Bytes after it, here a trailer
     of the program's own, are left for the program, and [consumed] says
     where they begin. *)
  let stream = Lz.Deflate.encode_string table in
  let input = of_string (stream ^ "TRAILER") in
  let out = bytes (String.length table) in
  match Lz.Deflate.decode input out with
  | Ok { consumed; _ } ->
      let rest = Array1.sub input consumed (Array1.dim input - consumed) in
      Printf.printf "\ndeflate stream %d bytes, consumed %d, then %S\n"
        (String.length stream) consumed (to_string rest)
  | Error e -> Format.printf "%a@." Lz.pp_error e
