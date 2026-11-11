(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf

(* Error messages *)

let err_date (y, m, d) = strf "Stac.search: %04d-%02d-%02d is not a day" y m d
let err_order = "Stac.search: the last day precedes the first"
let err_bbox = "Stac.search: bbox not finite, or south above north"
let err_page url e = strf "%s: %s" url e
let err_loop url = strf "%s: the next link repeats a request" url
let err_grid href e = strf "%s: %s" href e

let err_crs href (g : Jord.Grid.t) crs =
  Format.asprintf "%s: the raster is in %a, the points in %a" href Jord.Crs.pp
    g.crs Jord.Crs.pp crs

(* Search *)

type date = int * int * int

let search_url = "https://planetarycomputer.microsoft.com/api/stac/v1/search"

(* Items per page: the size of an answer, not a bound on the result. *)
let page_size = 100

let leap y = (y mod 4 = 0 && y mod 100 <> 0) || y mod 400 = 0

let month_days y m =
  match m with
  | 2 -> if leap y then 29 else 28
  | 4 | 6 | 9 | 11 -> 30
  | _ -> 31

let check_date ((y, m, d) as date) =
  if m < 1 || m > 12 || d < 1 || d > month_days y m || y < 0 || y > 9999 then
    invalid_arg (err_date date)

(* A day's first and last instants: MPC's catalogue keeps times to the
   microsecond. *)
let interval (y, m, d) (y', m', d') =
  strf "%04d-%02d-%02dT00:00:00Z/%04d-%02d-%02dT23:59:59.999999Z" y m d y' m'
    d'

let mems = function Jsont.Object (ms, _) -> ms | _ -> []
let mem name j = Option.map snd (Jsont.Json.find_mem name (mems j))

let string name j =
  match mem name j with Some (Jsont.String (s, _)) -> Some s | _ -> None

type request = Get of string | Post of string * Jsont.json

(* [b]'s members, then those of [base] that [b] does not name. *)
let merged base b =
  let named = List.map (fun ((n, _), _) -> n) (mems b) in
  let kept ((n, _), _) = not (List.mem n named) in
  Jsont.Json.object' (mems b @ List.filter kept (mems base))

(* The request a page's [next] link states, if any. A POST's body replaces
   the request's, or with [merge] is merged into it. *)
let next url request page =
  let links =
    match mem "links" page with Some (Jsont.Array (l, _)) -> l | _ -> []
  in
  match List.find_opt (fun l -> string "rel" l = Some "next") links with
  | None -> None
  | Some l -> (
      let href =
        match string "href" l with
        | Some h -> h
        | None -> failwith (err_page url "a next link without href")
      in
      let base =
        match request with Post (_, b) -> b | Get _ -> Jsont.Json.object' []
      in
      match (string "method" l, mem "body" l, mem "merge" l) with
      | Some "POST", None, _ -> Some (Post (href, base))
      | Some "POST", Some b, Some (Jsont.Bool (true, _)) ->
          Some (Post (href, merged base b))
      | Some "POST", Some b, _ -> Some (Post (href, b))
      | _ -> Some (Get href))

let same a b =
  match (a, b) with
  | Get u, Get u' -> u = u'
  | Post (u, b), Post (u', b') -> u = u' && Jsont.Json.equal b b'
  | Get _, Post _ | Post _, Get _ -> false

let search ~collection ~bbox:((w, s, e, n) as bbox) (first, last) =
  check_date first;
  check_date last;
  if compare first last > 0 then invalid_arg err_order;
  let finite (w, s, e, n) = List.for_all Float.is_finite [ w; s; e; n ] in
  if not (finite bbox) || s > n then invalid_arg err_bbox;
  let body =
    let module J = Jsont.Json in
    let field n v = J.mem (J.name n) v in
    J.object'
      [
        field "collections" (J.list [ J.string collection ]);
        field "bbox" (J.list (List.map J.number [ w; s; e; n ]));
        field "datetime" (J.string (interval first last));
        field "limit" (J.int page_size);
      ]
  in
  let rec pages acc seen request =
    let url, page =
      match request with
      | Get url -> (url, Fetch.get_json url)
      | Post (url, body) -> (url, Fetch.post_json url body)
    in
    let items =
      match Jord_stac.items page with
      | Ok items -> items
      | Error e -> failwith (err_page url e)
    in
    let acc = List.rev_append items acc in
    let seen = request :: seen in
    match next url request page with
    | None -> List.rev acc
    | Some r when List.exists (same r) seen -> failwith (err_loop url)
    | Some r -> pages acc seen r
  in
  pages [] [] (Post (search_url, body))

(* Windows *)

type session = Fetch.session

let session = Fetch.session
let token = Fetch.token

exception Changed = Fetch.Changed

type window = {
  row : int;
  col : int;
  samples : Nx.host Nx.packed;
  points : Nx.host Nx.float64_t;
}

(* The first IFD, and the partial file it was decoded from. *)
let rec first f p =
  match Nx_io.decode_tiff p with
  | Ok ds -> (List.hd ds, p)
  | Error rs -> first f (Fetch.fill f p rs)

(* Each round adds the ranges the last one lacked, and monotony keeps what
   was decoded, so the loop ends. *)
let rec block f p ~row ~col ~rows ~cols =
  let d, p = first f p in
  match Nx_io.Tiff.window d ~row ~col ~rows ~cols with
  | Ok x -> x
  | Error rs -> block f (Fetch.fill f p rs) ~row ~col ~rows ~cols

let window s rule (asset : Jord_stac.asset) crs c =
  let f, p = Fetch.open_file s asset.href in
  let d, p = first f p in
  let src =
    match Jord_geotiff.grid (Nx_io.Tiff.tags d) with
    | Ok src -> src
    | Error e -> failwith (err_grid asset.href e)
  in
  if not (Jord.Crs.equal src.crs crs) then
    failwith (err_crs asset.href src crs);
  let p' = Jord.Warp.pixel src c in
  let row, col, rows, cols = Jord.Warp.window rule src p' in
  let samples = block f p ~row ~col ~rows ~cols in
  let corner = Float.[| of_int row; of_int col |] in
  let points = Nx.sub p' (Nx.create Nx.float64 [| 2 |] corner) in
  { row; col; samples; points }
