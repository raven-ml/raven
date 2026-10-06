(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Packs the code objects into the archive nx.amd embeds:

     pack.exe ARCHIVE TARGET/KEY.co ...

   The archive is the 64-bit little-endian count of its members, then for
   each, in increasing order of key, the length of its key, the key, the
   offset of its bytes from the archive's start and their length, each length
   and offset 64 bits, then the members' bytes. A member's key is its path
   without [.co], ["gfx12-generic/cast.float32"]. *)

let () =
  match Array.to_list Sys.argv with
  | _ :: archive :: files ->
      let members =
        List.sort compare
          (List.map
             (fun f ->
               ( Filename.concat (Filename.basename (Filename.dirname f))
                   (Filename.remove_extension (Filename.basename f)),
                 In_channel.with_open_bin f In_channel.input_all ))
             files)
      in
      let index = Buffer.create 4096 and add64 b n = Buffer.add_int64_le b (Int64.of_int n) in
      add64 index (List.length members);
      let header =
        List.fold_left (fun n (k, _) -> n + 24 + String.length k) 8 members
      in
      ignore
        (List.fold_left
           (fun at (k, data) ->
             add64 index (String.length k);
             Buffer.add_string index k;
             add64 index at;
             add64 index (String.length data);
             at + String.length data)
           header members);
      Out_channel.with_open_bin archive (fun oc ->
          Buffer.output_buffer oc index;
          List.iter (fun (_, data) -> output_string oc data) members)
  | _ -> prerr_endline "usage: pack.exe ARCHIVE FILE..."; exit 2
