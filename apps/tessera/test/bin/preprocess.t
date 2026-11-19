preprocess records its request and the items it finds in the tile's
directory, and a run in that directory reads them again: here curl is a
stand-in that answers each STAC search with no item, and no other request.

  $ ./make_labelled.exe tile
  $ mkdir bin fake fake/script
  $ ln -s "$(pwd)/../http/curl.exe" bin/curl
  $ export PATH="$(pwd)/bin:$PATH" FAKE_CURL_DIR="$(pwd)/fake"
  $ none='{"type":"FeatureCollection","features":[]}'
  $ for i in 0 1; do printf 'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n%s' ${#none} "$none" > fake/script/00$i; done
  $ pre () { ../../bin/main.exe preprocess --landmask tile/landmask.tiff --out d "$@"; }
  $ pre --days 2025-07-01,2025-07-03
  0 Sentinel-2 items (0 distinct), 0 Sentinel-1 items
  d: 0 Sentinel-2 frames, 0 ascending, 0 descending
  $ cut -c1-60 fake/log
  POST https://planetarycomputer.microsoft.com/api/stac/v1/sea
  POST https://planetarycomputer.microsoft.com/api/stac/v1/sea

The same request again reads the recorded items, and asks nothing.

  $ pre --days 2025-07-01,2025-07-03
  0 Sentinel-2 items (0 distinct), 0 Sentinel-1 items
  d: 0 Sentinel-2 frames, 0 ascending, 0 descending
  $ wc -l < fake/log
  2

Another span is refused, naming both requests, and so is another region of
the same grid (its landmask's last cell cleared), and a directory of other
files.

  $ pre --days 2025-07-01,2025-07-31 2>&1 | grep -o 'holds the tile of another request\|"days":\[[^]]*\]'
  holds the tile of another request
  "days":["2025-07-01","2025-07-03"]
  "days":["2025-07-01","2025-07-31"]
  $ mkdir region && cp tile/landmask.tiff region/
  $ printf '\000' | dd of=region/landmask.tiff bs=1 seek=$(( $(wc -c < region/landmask.tiff) - 1 )) conv=notrunc 2> /dev/null
  $ ../../bin/main.exe preprocess --landmask region/landmask.tiff --days 2025-07-01,2025-07-03 --out d 2>&1 | grep -o 'holds the tile of another request\|"cells":[0-9]*'
  holds the tile of another request
  "cells":36
  "cells":35
  $ ./make_labelled.exe other > /dev/null
  $ ../../bin/main.exe preprocess --landmask tile/landmask.tiff --days 2025-07-01,2025-07-03 --out other 2>&1
  tessera: other holds files and no request.json: give a new directory
  [2]
