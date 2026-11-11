(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Jord
open Tessera

(* Requests go to http/curl.exe, put on the PATH as curl. It answers from the
   directory [served] makes: recorded or written answers in order, and files
   served by byte ranges. *)

let rec rm path =
  if Sys.is_directory path then (
    Array.iter (fun e -> rm (Filename.concat path e)) (Sys.readdir path);
    Sys.rmdir path)
  else Sys.remove path

let () =
  let bin = Filename.temp_dir "tessera" "bin" in
  at_exit (fun () -> rm bin);
  let exe = Filename.concat (Sys.getcwd ()) "http/curl.exe" in
  Unix.symlink exe (Filename.concat bin "curl");
  Unix.putenv "PATH" (bin ^ ":" ^ Sys.getenv "PATH")

let read file = In_channel.with_open_bin file In_channel.input_all
let write file s =
  Out_channel.with_open_bin file (fun oc -> output_string oc s)
let recorded name = read (Filename.concat "http" name)

let rec mkdir_p d =
  if not (Sys.file_exists d) then (
    mkdir_p (Filename.dirname d);
    Sys.mkdir d 0o755)

let host_path url =
  let n = String.length "https://" in
  String.sub url n (String.length url - n)

(* [served ~script ~files f] is [f dir] while curl answers from [dir]: the
   answers [script] in order, and each [(url, bytes, etags)] of [files]. *)
let served ?(script = []) ?(files = []) f =
  let dir = Filename.temp_dir "tessera" "http" in
  Fun.protect
    ~finally:(fun () -> rm dir)
    (fun () ->
      Unix.putenv "FAKE_CURL_DIR" dir;
      let scripts = Filename.concat dir "script" in
      Sys.mkdir scripts 0o755;
      let numbered i = Filename.concat scripts (Printf.sprintf "%03d" i) in
      List.iteri (fun i s -> write (numbered i) s) script;
      let files_dir = Filename.concat dir "files" in
      List.iter
        (fun (url, bytes, etags) ->
          let file = Filename.concat files_dir (host_path url) in
          mkdir_p (Filename.dirname file);
          write file bytes;
          if etags <> [] then
            write (file ^ ".etag") (String.concat "\n" etags))
        files;
      f dir)

type request = { url : string; headers : string list; body : string option }

(* The requests curl was asked, in order. *)
let log dir =
  let file = Filename.concat dir "log" in
  if not (Sys.file_exists file) then []
  else
    read file |> String.split_on_char '\n'
    |> List.filter (( <> ) "")
    |> List.map (fun l ->
        match String.split_on_char '\t' l with
        | first :: rest ->
            let meth, url =
              match String.index_opt first ' ' with
              | Some i ->
                  ( String.sub first 0 i,
                    String.sub first (i + 1) (String.length first - i - 1) )
              | None -> failf "log line %S" l
            in
            let headers, body =
              if meth = "POST" then
                match List.rev rest with
                | b :: hs -> (List.rev hs, Some b)
                | [] -> failf "log line %S" l
              else (rest, None)
            in
            { url; headers; body }
        | [] -> failf "log line %S" l)

let answer ?(headers = []) code body =
  let hs = List.map (fun (k, v) -> Printf.sprintf "%s: %s\r\n" k v) headers in
  Printf.sprintf "HTTP/1.1 %d Status\r\n%sContent-Length: %d\r\n\r\n%s" code
    (String.concat "" hs) (String.length body) body

let retry code = answer ~headers:[ ("Retry-After", "0") ] code ""
let json_answer s =
  answer ~headers:[ ("Content-Type", "application/json") ] 200 s

let decode s =
  match Jsont_bytesrw.decode_string Jsont.json s with
  | Ok j -> j
  | Error e -> failwith e

let json = Testable.make ~pp:Jsont.pp_json ~equal:Jsont.Json.equal

(* Search *)

let search_url = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
let berlin = (13.3, 52.4, 13.5, 52.6)
let day = (2024, 6, 26)
let s1 ?(bbox = berlin) days =
  Stac.search ~collection:"sentinel-1-rtc" ~bbox days

let ids = List.map (fun (i : Jord_stac.item) -> i.id)
let acquisition = "S1A_IW_GRDH_1SDV_20240626T0525"
let first_id = acquisition ^ "52_20240626T052617_054490_06A192_rtc"
let second_id = acquisition ^ "27_20240626T052552_054490_06A192_rtc"

(* A page of no item whose links are [links]. *)
let page links =
  json_answer
    (Printf.sprintf {|{"type":"FeatureCollection","features":[],"links":[%s]}|}
       links)

let body r = decode (Option.get r.body)

let search =
  group "search"
    [
      test "follows next links to the last page" (fun () ->
          served
            ~script:[ recorded "s1_page1.http"; recorded "s1_page2.http" ]
            (fun dir ->
              equal (list string) [ first_id; second_id ]
                (ids (s1 (day, day)));
              match log dir with
              | [ a; b ] ->
                  equal string search_url a.url;
                  mem string "Content-Type: application/json" a.headers;
                  equal json
                    (decode
                       {|{"collections": ["sentinel-1-rtc"],
                          "bbox": [13.3, 52.4, 13.5, 52.6],
                          "datetime": "2024-06-26T00:00:00Z/2024-06-26T23:59:59.999999Z",
                          "limit": 100}|})
                    (body a);
                  equal string
                    (search_url ^ "?collections=sentinel-1-rtc")
                    b.url;
                  equal json
                    (decode
                       {|{"collections": ["sentinel-1-rtc"],
                          "bbox": [13.3, 52.4, 13.5, 52.6],
                          "datetime": "2024-06-26T00:00:00Z/2024-06-26T23:59:59Z",
                          "limit": 1,
                          "token": "next:sentinel-1-rtc:S1A_IW_GRDH_1SDV_20240626T052552_20240626T052617_054490_06A192_rtc"}|})
                    (body b)
              | l -> failf "%d requests" (List.length l)));
      test "merges a next body that says merge into the request's" (fun () ->
          let next =
            {|{"rel": "next", "method": "POST", "merge": true,
               "href": "https://example.org/search",
               "body": {"token": "t", "limit": 7}}|}
          in
          served
            ~script:[ page next; recorded "s1_page2.http" ]
            (fun dir ->
              equal (list string) [ second_id ] (ids (s1 (day, day)));
              match log dir with
              | [ _; b ] ->
                  equal string "https://example.org/search" b.url;
                  equal json
                    (decode
                       {|{"collections": ["sentinel-1-rtc"],
                          "bbox": [13.3, 52.4, 13.5, 52.6],
                          "datetime": "2024-06-26T00:00:00Z/2024-06-26T23:59:59.999999Z",
                          "limit": 7, "token": "t"}|})
                    (body b)
              | l -> failf "%d requests" (List.length l)));
      test "follows a next link of no method by GET" (fun () ->
          let next = {|{"rel":"next","href":"https://example.org/page2"}|} in
          served
            ~script:[ page next; recorded "s1_page2.http" ]
            (fun dir ->
              equal (list string) [ second_id ] (ids (s1 (day, day)));
              match log dir with
              | [ _; b ] ->
                  equal string "https://example.org/page2" b.url;
                  is_none b.body
              | l -> failf "%d requests" (List.length l)));
      test "refuses a next link that repeats a request of the search"
        (fun () ->
          let link u = Printf.sprintf {|{"rel": "next", "href": %S}|} u in
          let a = "https://example.org/a" and b = "https://example.org/b" in
          served
            ~script:[ page (link a); page (link b); page (link a) ]
            (fun dir ->
              raises_match (Exn.failure ~sub:b) (fun () -> s1 (day, day));
              equal int 3 (List.length (log dir))));
      test "sends again after 429, 5xx and a reset connection" (fun () ->
          served
            ~script:
              [ "curl-exit 56"; retry 429; retry 503; recorded "s1_page2.http" ]
            (fun dir ->
              equal (list string) [ second_id ] (ids (s1 (day, day)));
              equal int 4 (List.length (log dir))));
      test "gives up after five attempts, naming the status" (fun () ->
          served
            ~script:(List.init 6 (fun _ -> retry 503))
            (fun dir ->
              raises_match (Exn.failure ~sub:"HTTP status 503") (fun () ->
                  s1 (day, day));
              equal int 5 (List.length (log dir))));
      test "gives up at once on a Retry-After past a minute" (fun () ->
          let wait = answer ~headers:[ ("Retry-After", "120") ] 429 "" in
          served
            ~script:[ wait; recorded "s1_page2.http" ]
            (fun dir ->
              raises_match
                (Exn.failure ~sub:"HTTP status 429 (Retry-After 120)")
                (fun () -> s1 (day, day));
              equal int 1 (List.length (log dir))));
      test "raises on a client error without sending again" (fun () ->
          served
            ~script:[ answer 400 "bad bbox"; recorded "s1_page2.http" ]
            (fun dir ->
              raises_match
                (Exn.failure ~sub:(search_url ^ ": HTTP status 400"))
                (fun () -> s1 (day, day));
              equal int 1 (List.length (log dir))));
      test "raises on an answer that is not JSON" (fun () ->
          served
            ~script:[ answer 200 "<html>" ]
            (fun _ ->
              raises_match (Exn.failure ~sub:search_url) (fun () ->
                  s1 (day, day))));
      test "refuses a day not of the calendar, and days out of order"
        (fun () ->
          raises_match (Exn.invalid_arg ~sub:"2023-02-29") (fun () ->
              s1 ((2023, 2, 29), (2023, 3, 1)));
          raises_match (Exn.invalid_arg ~sub:"precedes") (fun () ->
              s1 ((2024, 3, 2), (2024, 3, 1)));
          raises_match (Exn.invalid_arg ~sub:"bbox") (fun () ->
              s1 ~bbox:(0., Float.nan, 1., 1.) (day, day));
          raises_match (Exn.invalid_arg ~sub:"bbox") (fun () ->
              s1 ~bbox:(0., 2., 1., 1.) (day, day)));
    ]

(* Windows *)

let utm33 = Result.get_ok (Crs.of_epsg 32633)
let utm32 = Result.get_ok (Crs.of_epsg 32632)

(* A raster of 200 × 300 cells at 10 m, in 16² Deflate tiles of numbers
   that do not compress: 120 KB, past the first 64 KiB read. *)
let src =
  Grid.v utm33 ~rows:200 ~cols:300 ~origin:(400000., 5800000.)
    ~row_step:(0., -10.) ~col_step:(10., 0.)

let raster ~rows ~cols =
  Nx.init Nx.uint16 [| rows; cols; 1 |] (fun i ->
      Hashtbl.hash (i.(0), i.(1)) land 0xffff)

let cog ?(tags = Jord_geotiff.tags src) ~rows ~cols () =
  let file = Filename.temp_file "tessera" ".tif" in
  Fun.protect
    ~finally:(fun () -> Sys.remove file)
    (fun () ->
      Nx_io.save_tiff ~tile:(16, 16) ~compression:Deflate ~tags file
        (raster ~rows ~cols);
      read file)

let big = cog ~rows:200 ~cols:300 ()
let token_url = "https://planetarycomputer.microsoft.com/api/sas/v1/token/"

(* The token endpoint's answer of a token that expires at [expiry]. *)
let token ?(expiry = "2999-01-01T00:00:00Z") () =
  json_answer
    (Printf.sprintf {|{"msft:expiry": %S, "token": "st=x&sig=SECRET"}|}
       expiry)

(* RFC 3339's form of [s] seconds from now. *)
let from_now s =
  let t = Unix.gmtime (Unix.gettimeofday () +. s) in
  Printf.sprintf "%04d-%02d-%02dT%02d:%02d:%02dZ" (t.tm_year + 1900)
    (t.tm_mon + 1) t.tm_mday t.tm_hour t.tm_min t.tm_sec

let s2_blob = "https://sentinel2l2a01.blob.core.windows.net/sentinel2-l2/"
let s1_blob =
  "https://sentinel1euwestrtc.blob.core.windows.net/sentinel1-grd-rtc/"

let asset href =
  {
    Jord_stac.href;
    media_type = None;
    roles = [ "data" ];
    grid = None;
    bands = [];
  }

(* [href]'s window for [g], in a session of its own unless [s] is given. *)
let read_window ?s rule href (g : Grid.t) =
  let s = match s with Some s -> s | None -> Stac.session () in
  Stac.window s rule (asset href) g.crs (Warp.centres ~at:Nx.Host.on g.crs g)

let rule =
  Gen.of_list
    ~pp:(fun ppf r ->
      Format.pp_print_string ppf
        (match r with Warp.Nearest -> "nearest" | Bilinear -> "bilinear"))
    [ Warp.Nearest; Bilinear ]

(* A grid over, across or beside the raster, rotated or not. *)
let dst =
  let open Gen in
  let+ rows = int_range 0 12
  and+ cols = int_range 0 12
  and+ d = float_range 3. 40.
  and+ x = float_range (-400.) 3400.
  and+ y = float_range (-400.) 2400.
  and+ turn = frequency [ (2, constant 0.); (1, float_range 0. 6.3) ] in
  let c = cos turn and s = sin turn in
  Grid.v utm33 ~rows ~cols
    ~origin:(400000. +. x, 5800000. -. y)
    ~row_step:(-.d *. s, -.d *. c)
    ~col_step:(d *. c, -.d *. s)

let range_of r =
  List.find_map
    (fun h -> Scanf.sscanf_opt h "Range: bytes=%d-%d%!" (fun a b -> (a, b)))
    r.headers

let is_token r = String.starts_with ~prefix:token_url r.url
let tokens dir = List.filter is_token (log dir)
let blobs dir = List.filter (fun r -> not (is_token r)) (log dir)

let sample rule p x =
  match rule with
  | Warp.Nearest -> Warp.nearest p x
  | Bilinear -> Warp.bilinear p x

let ints x = Nx.to_array (Nx.cast Nx.int32 x)

let agrees =
  prop "agrees with the warp on the whole raster" (Gen.pair rule dst)
    (fun (rule, g) ->
      let href = s2_blob ^ "x.tif" in
      served ~script:[ token () ] ~files:[ (href, big, []) ] (fun dir ->
          let w = read_window rule href g in
          let whole = raster ~rows:200 ~cols:300 in
          let p = Warp.pixel src (Warp.centres ~at:Nx.Host.on utm33 g) in
          let row, col, rows, cols = Warp.window rule src p in
          let x = Nx.unpack Nx.uint16 w.samples in
          equal (quad int int int int) (row, col, rows, cols)
            (w.row, w.col, (Nx.shape x).(0), (Nx.shape x).(1));
          let block = Nx.slice [ R (row, row + rows); R (col, col + cols) ] in
          equal (array int32) (ints (block whole)) (ints x);
          let v, ok = sample rule w.points x in
          let v', ok' = sample rule p whole in
          equal (array int32) (ints v') (ints v);
          equal (array bool) (Nx.to_array ok') (Nx.to_array ok);
          let rs = blobs dir in
          cover "no cell of the raster" (rows = 0);
          cover "a window past the first read" (List.length rs > 1);
          equal (list string) [ token_url ^ "sentinel2l2a01/sentinel2-l2" ]
            (List.map (fun r -> r.url) (tokens dir));
          let signed = href ^ "?st=x&sig=SECRET" in
          List.iter (fun r -> equal string signed r.url) rs;
          match rs with
          | [] -> fail "no request"
          | first :: rest ->
              equal (list string) [ "Range: bytes=0-65535" ] first.headers;
              let tagged r = mem string "If-Match: \"0x1\"" r.headers in
              List.iter tagged rest;
              let ranges = List.sort compare (List.filter_map range_of rest) in
              let rec apart = function
                | (_, b) :: ((a, _) :: _ as l) ->
                    at_least int ~than:16384 (a - b - 1);
                    apart l
                | _ -> ()
              in
              apart ranges))

(* A grid of the raster's last rows, whose tiles lie past the first read. *)
let bottom = Grid.window ~row:190 ~col:100 ~rows:8 ~cols:8 src
let example = "https://example.org/x.tif"

let windows =
  group "window"
    [
      agrees;
      test "fetches tiles less than 16 KiB apart as one range" (fun () ->
          (* Tiles of 16² uint16 that do not compress, about 520 bytes each,
             in rows of [cols / 16]. A window of one column of tiles reads
             tiles a row of tiles apart: with 2 to a row the tile between
             them is joined to them, with 256 to a row (128 KiB) it is not. *)
          let requests ~rows ~cols ~row ~col =
            let grid =
              Grid.v utm33 ~rows ~cols ~origin:(400000., 5800000.)
                ~row_step:(0., -10.) ~col_step:(10., 0.)
            in
            let file = cog ~tags:(Jord_geotiff.tags grid) ~rows ~cols () in
            let g = Grid.window ~row ~col ~rows:8 ~cols:8 grid in
            served ~files:[ (example, file, []) ] (fun dir ->
                ignore (read_window Nearest example g);
                List.length (log dir))
          in
          equal ~msg:"tall" int 2
            (requests ~rows:4096 ~cols:32 ~row:3004 ~col:20);
          equal ~msg:"wide" int 3
            (requests ~rows:48 ~cols:4096 ~row:12 ~col:3000));
      test "refuses points in another system than the raster's" (fun () ->
          served ~files:[ (example, big, []) ] (fun _ ->
              let c = Warp.centres ~at:Nx.Host.on utm32 bottom in
              let s = Stac.session () in
              raises_match (Exn.failure ~sub:example) (fun () ->
                  Stac.window s Nearest (asset example) utm32 c)));
      test "raises Changed when the file changes while it is read" (fun () ->
          let href = s2_blob ^ "changed.tif" in
          served ~script:[ token () ]
            ~files:[ (href, big, [ "\"0x1\""; "\"0x2\"" ]) ]
            (fun _ ->
              raises (Stac.Changed href) (fun () ->
                  read_window Bilinear href bottom)));
      cases ~id:Fun.id "refuses an answer that is not the range asked for"
        [ "whole"; "short"; "span"; "length"; "past" ]
        (fun lie ->
          served ~files:[ (example, big, []) ] (fun dir ->
              write
                (Filename.concat dir ("files/" ^ host_path example ^ ".lie"))
                lie;
              raises_match (Exn.failure ~sub:example) (fun () ->
                  read_window Nearest example bottom)));
      test "refuses a first answer whose range lies past its length"
        (fun () ->
          let cr = ("Content-Range", "bytes 0-9/5") in
          served ~script:[ answer ~headers:[ cr ] 206 "0123456789" ] (fun _ ->
              raises_match (Exn.failure ~sub:example) (fun () ->
                  read_window Nearest example bottom)));
      test "reads a whole file answered to the first range" (fun () ->
          let small = cog ~rows:20 ~cols:20 () in
          served ~script:[ answer 200 small ] (fun dir ->
              let g = Grid.window ~row:2 ~col:3 ~rows:10 ~cols:9 src in
              let w = read_window Bilinear example g in
              let x = Nx.unpack Nx.uint16 w.samples in
              equal (array int) [| 11; 10; 1 |] (Nx.shape x);
              equal int 1 (List.length (log dir))));
      test "reads an empty file as one that names no IFD" (fun () ->
          let cr = ("Content-Range", "bytes */0") in
          served ~script:[ answer ~headers:[ cr ] 416 "" ] (fun _ ->
              raises_match (Exn.failure ~sub:example) (fun () ->
                  read_window Nearest example bottom)));
      cases ~id:Fun.id "refuses an href that is not https://, unrequested"
        [ "file:///etc/passwd"; "-K/etc/passwd"; "http://example.org/x" ]
        (fun href ->
          served ~files:[ (example, big, []) ] (fun dir ->
              raises_match (Exn.failure ~sub:"not an https:// URL") (fun () ->
                  read_window Nearest href bottom);
              equal int 0 (List.length (log dir))));
      cases ~id:Fun.id "reads any host but MPC's blob hosts unsigned"
        (* MPC's hosts in another authority, a lookalike host, a fragment,
           and a blob outside a container. *)
        [
          "https://sentinel2l2a01.evil.example?.blob.core.windows.net/sentinel2-l2/x.tif";
          "https://sentinel2l2a01.blob.core.windows.net.evil.example/sentinel2-l2/x.tif";
          "https://evil.sentinel2l2a01.blob.core.windows.net/sentinel2-l2/x.tif";
          "https://other.blob.core.windows.net/sentinel2-l2/x.tif";
          "https://user@sentinel2l2a01.blob.core.windows.net/sentinel2-l2/x.tif";
          "https://sentinel2l2a01.blob.core.windows.net/sentinel2-l2/x.tif#f";
          "https://sentinel2l2a01.blob.core.windows.net/x.tif";
        ]
        (fun href ->
          served ~script:[ answer 404 "" ] (fun dir ->
              raises_match (Exn.failure ~sub:"HTTP status 404") (fun () ->
                  read_window Nearest href bottom);
              let urls = List.map (fun r -> r.url) (log dir) in
              equal (list string) [ href ] urls));
      test "keeps a token until a minute before it expires" (fun () ->
          (* Each read asks two ranges, each signed. *)
          let asked expiry =
            let href = s1_blob ^ "x.tif" in
            let script = List.init 4 (fun _ -> token ~expiry ()) in
            served ~script ~files:[ (href, big, []) ] (fun dir ->
                let s = Stac.session () in
                ignore (read_window ~s Nearest href bottom);
                ignore (read_window ~s Nearest href bottom);
                equal int 4 (List.length (blobs dir));
                List.length (tokens dir))
          in
          equal ~msg:"in 2 minutes" int 1 (asked (from_now 120.));
          equal ~msg:"in 30 seconds" int 4 (asked (from_now 30.));
          equal ~msg:"in 2020" int 4 (asked "2020-01-01T00:00:00Z"));
      test "asks for a new token after a 403" (fun () ->
          let href = s2_blob ^ "denied.tif" in
          served ~script:[ token (); token () ] ~files:[ (href, big, []) ]
            (fun dir ->
              write
                (Filename.concat dir ("files/" ^ host_path href ^ ".deny"))
                "1";
              ignore (read_window Nearest href bottom);
              equal int 2 (List.length (tokens dir));
              equal int 3 (List.length (blobs dir))));
      test "signs with the token MPC's endpoint answers" (fun () ->
          let href = s1_blob ^ "x.tif" in
          served ~script:[ recorded "token.http" ] ~files:[ (href, big, []) ]
            (fun dir ->
              ignore (read_window Nearest href bottom);
              let query = "?st=2026-11-10T21%3A40%3A24Z&se=" in
              List.iter
                (fun r ->
                  starts_with ~prefix:(href ^ query) r.url;
                  ends_with ~suffix:"&skv=2025-07-05&sig=REDACTED" r.url)
                (blobs dir)));
      test "refuses a file whose tags state no grid" (fun () ->
          let plain = cog ~tags:[] ~rows:20 ~cols:20 () in
          served ~files:[ (example, plain, []) ] (fun _ ->
              raises_match (Exn.failure ~sub:example) (fun () ->
                  read_window Nearest example bottom)));
      test "names the href of a failed answer" (fun () ->
          served ~script:[ answer 404 "BlobNotFound" ] (fun _ ->
              raises_match (Exn.failure ~sub:(example ^ ": HTTP status 404"))
                (fun () -> read_window Nearest example bottom)));
    ]

(* Sessions *)

module Smap = Map.Make (String)

(* The model: each key's token and expiry; a token is kept while it expires
   more than a minute after the call's time. *)
let model_token key now (t, expiry) m =
  match Smap.find_opt key !m with
  | Some (t', e) when now < e -. 60. -> t'
  | _ ->
      m := Smap.add key (t, expiry) !m;
      t

let tokens_stateful =
  let session = abstract "s" in
  let key = Gen.of_list ~pp:Format.pp_print_string [ "a/x"; "b/y" ] in
  let now = Gen.map Float.of_int (Gen.int_range 0 200) in
  let now = Gen.with_pp (fun ppf t -> Format.fprintf ppf "%g" t) now in
  let fetched =
    let open Gen in
    let+ t = Gen.of_list ~pp:Format.pp_print_string [ "t1"; "t2"; "t3" ]
    and+ e = Gen.int_range 0 300 in
    (t, Float.of_int e)
  in
  let fetched =
    Gen.with_pp (fun ppf (t, e) -> Format.fprintf ppf "%s,%g" t e) fetched
  in
  stateful ~domains:2 "a session's tokens behave as a map kept by expiry"
    [
      command "session"
        (Gen.unit @-> makes session)
        (fun () -> ref Smap.empty)
        (fun () -> Stac.session ());
      command "token"
        (key @-> now @-> fetched @-> session ^-> returns string)
        model_token
        (fun key now (t, e) s -> Stac.token s ~now key (fun () -> (t, e)));
    ]

let () =
  exit
    (run "tessera.stac"
       [ search; windows; group "session" [ tokens_stateful ] ])
