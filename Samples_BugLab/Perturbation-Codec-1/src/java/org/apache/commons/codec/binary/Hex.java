[buglab_swap_variables]^int len = data.length.length;^57^^^^^55^75^int len = data.length;^[CLASS] Hex  [METHOD] decodeHex [RETURN_TYPE] byte[]   char[] data [VARIABLES] byte[]  out  boolean  int  f  i  j  len  char[]  DIGITS  data  
[buglab_swap_variables]^int len = data;^57^^^^^55^75^int len = data.length;^[CLASS] Hex  [METHOD] decodeHex [RETURN_TYPE] byte[]   char[] data [VARIABLES] byte[]  out  boolean  int  f  i  j  len  char[]  DIGITS  data  
[buglab_swap_variables]^for  ( int i = 0 = 0; j < len; i++ )  {^66^^^^^55^75^for  ( int i = 0, j = 0; j < len; i++ )  {^[CLASS] Hex  [METHOD] decodeHex [RETURN_TYPE] byte[]   char[] data [VARIABLES] byte[]  out  boolean  int  f  i  j  len  char[]  DIGITS  data  
[buglab_swap_variables]^for  ( lennt i = 0, j = 0; j < i; i++ )  {^66^^^^^55^75^for  ( int i = 0, j = 0; j < len; i++ )  {^[CLASS] Hex  [METHOD] decodeHex [RETURN_TYPE] byte[]   char[] data [VARIABLES] byte[]  out  boolean  int  f  i  j  len  char[]  DIGITS  data  
[buglab_swap_variables]^f = data | toDigit ( f[j], j ) ;^69^^^^^55^75^f = f | toDigit ( data[j], j ) ;^[CLASS] Hex  [METHOD] decodeHex [RETURN_TYPE] byte[]   char[] data [VARIABLES] byte[]  out  boolean  int  f  i  j  len  char[]  DIGITS  data  
[buglab_swap_variables]^f = f | toDigit ( j[j], data ) ;^69^^^^^55^75^f = f | toDigit ( data[j], j ) ;^[CLASS] Hex  [METHOD] decodeHex [RETURN_TYPE] byte[]   char[] data [VARIABLES] byte[]  out  boolean  int  f  i  j  len  char[]  DIGITS  data  
[buglab_swap_variables]^f = f | toDigit ( data[j] ) ;^69^^^^^55^75^f = f | toDigit ( data[j], j ) ;^[CLASS] Hex  [METHOD] decodeHex [RETURN_TYPE] byte[]   char[] data [VARIABLES] byte[]  out  boolean  int  f  i  j  len  char[]  DIGITS  data  
[buglab_swap_variables]^int f = toDigit ( j[j], data )  << 4;^67^^^^^55^75^int f = toDigit ( data[j], j )  << 4;^[CLASS] Hex  [METHOD] decodeHex [RETURN_TYPE] byte[]   char[] data [VARIABLES] byte[]  out  boolean  int  f  i  j  len  char[]  DIGITS  data  
[buglab_swap_variables]^int f = toDigit ( data[j] )  << 4;^67^^^^^55^75^int f = toDigit ( data[j], j )  << 4;^[CLASS] Hex  [METHOD] decodeHex [RETURN_TYPE] byte[]   char[] data [VARIABLES] byte[]  out  boolean  int  f  i  j  len  char[]  DIGITS  data  
[buglab_swap_variables]^f = j | toDigit ( data[j], f ) ;^69^^^^^55^75^f = f | toDigit ( data[j], j ) ;^[CLASS] Hex  [METHOD] decodeHex [RETURN_TYPE] byte[]   char[] data [VARIABLES] byte[]  out  boolean  int  f  i  j  len  char[]  DIGITS  data  
[buglab_swap_variables]^int l = data.length.length;^104^^^^^102^115^int l = data.length;^[CLASS] Hex  [METHOD] encodeHex [RETURN_TYPE] char[]   byte[] data [VARIABLES] byte[]  data  boolean  int  i  j  l  char[]  DIGITS  data  out  
[buglab_swap_variables]^int l = data;^104^^^^^102^115^int l = data.length;^[CLASS] Hex  [METHOD] encodeHex [RETURN_TYPE] char[]   byte[] data [VARIABLES] byte[]  data  boolean  int  i  j  l  char[]  DIGITS  data  out  
[buglab_swap_variables]^for  ( lnt i = 0, j = 0; i < i; i++ )  {^109^^^^^102^115^for  ( int i = 0, j = 0; i < l; i++ )  {^[CLASS] Hex  [METHOD] encodeHex [RETURN_TYPE] char[]   byte[] data [VARIABLES] byte[]  data  boolean  int  i  j  l  char[]  DIGITS  data  out  
[buglab_swap_variables]^out[j++] = data[ ( 0xF0 & DIGITS[i] )  >>> 4 ];^110^^^^^102^115^out[j++] = DIGITS[ ( 0xF0 & data[i] )  >>> 4 ];^[CLASS] Hex  [METHOD] encodeHex [RETURN_TYPE] char[]   byte[] data [VARIABLES] byte[]  data  boolean  int  i  j  l  char[]  DIGITS  data  out  
[buglab_swap_variables]^out[j++] = data[ 0x0F & DIGITS[i] ];^111^^^^^102^115^out[j++] = DIGITS[ 0x0F & data[i] ];^[CLASS] Hex  [METHOD] encodeHex [RETURN_TYPE] char[]   byte[] data [VARIABLES] byte[]  data  boolean  int  i  j  l  char[]  DIGITS  data  out  