munin metrics writes each value as the fewest digits from 15 that read back
to it, and each timestamp to the microsecond, the same text on every system:
the C library's %g writes 1e+06, and 1e+006 under mingw-w64, and keeps six
digits.

  $ id=$(./log_run.exe store)

  $ munin metrics --root store $id
  key	latest_value	latest_step	count
  loss	nan	4	5

  $ munin metrics --root store $id --key loss
  step	timestamp	value
  0	1709385958.677423	1000000
  1	2.000000	4.58e-72
  2	3.000000	0.1
  3	4.000000	inf
  4	5.000000	nan

  $ munin metrics --root store $id --key loss --format csv
  step,timestamp,value
  0,1709385958.677423,1000000
  1,2.000000,4.58e-72
  2,3.000000,0.1
  3,4.000000,inf
  4,5.000000,nan

JSON has no number for NaN or the infinities: they are the strings "NaN",
"Infinity" and "-Infinity".

  $ munin metrics --root store $id --key loss --format json
  [{"step":0,"timestamp":1709385958.677423,"value":1000000},{"step":1,"timestamp":2.000000,"value":4.58e-72},{"step":2,"timestamp":3.000000,"value":0.1},{"step":3,"timestamp":4.000000,"value":"Infinity"},{"step":4,"timestamp":5.000000,"value":"NaN"}]
