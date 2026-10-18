# munin

Local run tracking: metrics, artifacts and run comparison on the local
filesystem, as plain JSON readable with `jq`, with a terminal dashboard.
No server and no accounts.

- `munin.sys` samples system metrics through C stubs per operating system,
  one sample per tick with no averaging, every 15 s by default.
