(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module H = Header
module V = Value
module I = Image

module Fits = struct
  module Value = struct
    type 'a t = 'a Value.t

    let bool = Value.bool
    let int = Value.int
    let float = Value.float
    let string = Value.string
    let text = Value.text
    let map = Value.map
  end

  module Header = struct
    type t = Header.t

    let empty = Header.empty
    let of_string = Header.of_string
    let to_string = Header.to_string
    let records = Header.records
    let equal = Header.equal
    let pp = Header.pp
    let get = Header.get
    let find = Header.find
    let set = Header.set
    let remove = Header.remove
    let commentary = Header.commentary
    let add_commentary = Header.add_commentary
  end

  type hdu = Hdu.t

  let read = Hdu.read
  let of_bytes = Hdu.of_bytes
  let get = Hdu.get
  let header = Hdu.header
  let data = Hdu.data
  let name = Hdu.name
  let digest = Hdu.digest
  let v = Files.v
  let with_header = Files.with_header
  let verify = Files.verify
  let write = Files.write

  module Image = struct
    type t = Image.t

    let of_hdu = Image.of_hdu
    let shape = Image.shape
    let element = Image.element
    let scaled = Image.scaled
    let pp = Image.pp
    let raw = Image.raw
    let values = Image.values
    let validity = Image.validity
    let hdu = Image.hdu
  end

  let pp ppf hdu =
    let h = header hdu in
    let kind =
      match H.find_struct V.string "XTENSION" h with
      | Ok (Some x) -> x
      | _ -> "PRIMARY"
    in
    let name =
      match Hdu.extname hdu with
      | Some n -> Printf.sprintf " %s %d" n (Hdu.extver hdu)
      | None -> ""
    in
    Format.fprintf ppf "%s%s: " kind name;
    match I.of_hdu hdu with
    | Ok i -> I.pp ppf i
    | Error _ -> Format.fprintf ppf "%d bytes" (Hdu.store hdu).size
end
