(* Decoders.

   Decoders convert token strings back to natural text by reversing
   encoding-specific transformations: prefix/suffix removal, space insertion,
   byte-level decoding, and marker replacement. *)

open Brot

let show name decoder tokens =
  let result = Decoder.decode decoder tokens in
  Printf.printf "  %-22s [%s] -> %S\n" name
    (String.concat "; " (List.map (fun s -> Printf.sprintf "%S" s) tokens))
    result

let () =
  Printf.printf "=== Per-token Decoders ===\n\n";

  show "wordpiece" (Decoder.wordpiece ()) [ "play"; "##ing"; "un"; "##happy" ];

  show "bpe (suffix=</w>)"
    (Decoder.bpe ~suffix:"</w>" ())
    [ "hel"; "lo</w>"; "wor"; "ld</w>" ];

  show "metaspace" (Decoder.metaspace ())
    [ "\xe2\x96\x81Hello"; "\xe2\x96\x81world" ];

  show "byte_fallback" (Decoder.byte_fallback ()) [ "hello"; "<0x21>" ];

  Printf.printf "\n=== Collapsing Decoders ===\n\n";

  show "fuse" (Decoder.fuse ()) [ "h"; "e"; "l"; "l"; "o" ];

  show "replace ('_' -> ' ')"
    (Decoder.replace ~pattern:"_" ~by:" " ())
    [ "hello_world" ];

  Printf.printf "\n=== Composed Decoder ===\n\n";

  let composed =
    Decoder.sequence
      [ Decoder.wordpiece (); Decoder.replace ~pattern:"  " ~by:" " () ]
  in
  show "wordpiece + replace" composed [ "play"; "##ing"; "is"; "great" ];

  Printf.printf "\n=== Integrated with Tokenizer ===\n\n";

  let vocab =
    [
      ("[UNK]", 0);
      ("[CLS]", 1);
      ("[SEP]", 2);
      ("play", 3);
      ("##ing", 4);
      ("##ed", 5);
      ("great", 6);
    ]
  in
  let tokenizer =
    wordpiece ~vocab ~unk_token:"[UNK]"
      ~specials:[ special "[CLS]"; special "[SEP]" ]
      ~post:(Post_processor.bert ~cls:("[CLS]", 1) ~sep:("[SEP]", 2) ())
      ~decoder:(Decoder.wordpiece ()) ()
  in

  let text = "playing" in
  let encoding = encode tokenizer text in
  let ids = Encoding.ids encoding in
  Printf.printf "  Text:    %S\n" text;
  Printf.printf "  Tokens:  [%s]\n"
    (String.concat "; "
       (List.map
          (fun s -> Printf.sprintf "%S" s)
          (Array.to_list (Encoding.tokens encoding))));
  Printf.printf "  IDs:     [%s]\n"
    (String.concat "; " (Array.to_list (Array.map string_of_int ids)));
  Printf.printf "  Decoded: %S\n" (decode tokenizer ids);
  Printf.printf "  Decoded (skip specials): %S\n"
    (decode tokenizer ~skip_special_tokens:true ids)
