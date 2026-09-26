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

let make ~name ?cachekey ~compile () =
  let cachekey = if Helpers.Context_var.get Helpers.ccache = 0 then None else cachekey in
  { name; cachekey; compile }

let name t = t.name

let cachekey t = t.cachekey

let compile t src = t.compile src

let compile_cached t src =
  let cached = Option.bind t.cachekey (fun table -> Diskcache.get ~table ~key:src) in
  match cached with
  | Some lib -> lib
  | None ->
      if Helpers.getenv "ASSERT_COMPILE" 0 <> 0 then
        raise (Compile_error ("compilation disabled by ASSERT_COMPILE\n" ^ src));
      let lib = t.compile src in
      Option.iter (fun table -> Diskcache.put ~table ~key:src lib) t.cachekey;
      lib
