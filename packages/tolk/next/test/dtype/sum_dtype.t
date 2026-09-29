The defaults come from the environment alone:

  $ unset SUM_DTYPE DEFAULT_FLOAT DEFAULT_INT

Sums of floats accumulate in at least SUM_DTYPE, float32 when it is unset:

  $ ./sum_acc.exe half double
  half: dtypes.float
  double: dtypes.double
  half: dtypes.float
  double: dtypes.double

SUM_DTYPE is read once, at its first use: setting it later changes nothing.

  $ SUM_DTYPE=double ./sum_acc.exe half float
  half: dtypes.double
  float: dtypes.double
  half: dtypes.double
  float: dtypes.double

  $ SUM_DTYPE=bfloat16 ./sum_acc.exe half
  half: dtypes.float
  half: dtypes.float

  $ SUM_DTYPE=uchar ./sum_acc.exe half
  half: dtypes.half
  half: dtypes.half

  $ SUM_DTYPE=default_float DEFAULT_FLOAT=half ./sum_acc.exe half
  half: dtypes.half
  half: dtypes.half

Integers ignore it:

  $ SUM_DTYPE=double ./sum_acc.exe char ushort
  char: dtypes.int
  ushort: dtypes.uint
  char: dtypes.int
  ushort: dtypes.uint

A SUM_DTYPE that names no data type is rejected:

  $ SUM_DTYPE=f32 ./sum_acc.exe half
  rejected
  [1]
  $ SUM_DTYPE=bf16 ./sum_acc.exe half
  rejected
  [1]
  $ SUM_DTYPE=' half ' ./sum_acc.exe half
  rejected
  [1]
  $ SUM_DTYPE=uint128 ./sum_acc.exe half
  rejected
  [1]
