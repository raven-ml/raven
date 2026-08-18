(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = unit

let create () = ()

let encode_into () ids text ~pos ~len =
  for i = pos to pos + len - 1 do
    Ints.add ids (Char.code (String.unsafe_get text i))
  done

let table = Array.init 256 (fun b -> String.make 1 (Char.unsafe_chr b))
let lengths = Array.make 256 1
let token_table () = table
let len_table () = lengths

let token_to_id () token =
  if String.length token = 1 then Some (Char.code token.[0]) else None

let id_to_token () id =
  if id >= 0 && id <= 255 then Some (Array.unsafe_get table id) else None

let get_vocab () = []
let get_vocab_size () = 256
let save () ~folder:_ () = []
