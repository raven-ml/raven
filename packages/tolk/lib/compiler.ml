(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

type t = {
  name : string;
  cachekey : string option;
  compile : string -> bytes;
}

exception Compile_error of string

let ccache = Helpers.getenv "CCACHE" 1

let make ~name ?cachekey ~compile () =
  let cachekey = if ccache <> 0 then cachekey else None in
  { name; cachekey; compile }

let name t = t.name

let compile t src = t.compile src

let compile_cached t src =
  match t.cachekey with
  | None -> t.compile src
  | Some table ->
      match Diskcache.get ~table ~key:src with
      | Some lib -> lib
      | None ->
          let lib = t.compile src in
          Diskcache.put ~table ~key:src lib;
          lib
