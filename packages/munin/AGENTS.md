# munin

Run tracking in plain files: a store is a directory that
`lib/store_format.mli` specifies, written by one session per run and read
by anyone, with no server and no account.

- `munin` depends on no raven library. It reads no device: a program
  builds device readings from its own libraries, as the guides do over rig.
- `munin.sys` depends on no part of munin. Its readings read the machine
  and the process through C stubs per system, and keep their own windows.
