[buglab_swap_variables]^return SoundexUtils.difference ( this, s2, s1 ) ;^80^^^^^79^81^return SoundexUtils.difference ( this, s1, s2 ) ;^[CLASS] Soundex  [METHOD] difference [RETURN_TYPE] int   String s1 String s2 [VARIABLES] Soundex  US_ENGLISH  String  US_ENGLISH_MAPPING_STRING  s1  s2  boolean  int  maxLength  char[]  US_ENGLISH_MAPPING  mapping  soundexMapping  
[buglab_swap_variables]^return SoundexUtils.difference ( this,  s2 ) ;^80^^^^^79^81^return SoundexUtils.difference ( this, s1, s2 ) ;^[CLASS] Soundex  [METHOD] difference [RETURN_TYPE] int   String s1 String s2 [VARIABLES] Soundex  US_ENGLISH  String  US_ENGLISH_MAPPING_STRING  s1  s2  boolean  int  maxLength  char[]  US_ENGLISH_MAPPING  mapping  soundexMapping  
[buglab_swap_variables]^return SoundexUtils.difference ( this, s1 ) ;^80^^^^^79^81^return SoundexUtils.difference ( this, s1, s2 ) ;^[CLASS] Soundex  [METHOD] difference [RETURN_TYPE] int   String s1 String s2 [VARIABLES] Soundex  US_ENGLISH  String  US_ENGLISH_MAPPING_STRING  s1  s2  boolean  int  maxLength  char[]  US_ENGLISH_MAPPING  mapping  soundexMapping  
[buglab_swap_variables]^char mappedChar = this.map ( index.charAt ( str )  ) ;^168^^^^^166^181^char mappedChar = this.map ( str.charAt ( index )  ) ;^[CLASS] Soundex  [METHOD] getMappingCode [RETURN_TYPE] char   String str int index [VARIABLES] char  firstCode  hwChar  mappedChar  preHWChar  Soundex  US_ENGLISH  String  US_ENGLISH_MAPPING_STRING  pString  s1  s2  str  boolean  int  index  maxLength  char[]  US_ENGLISH_MAPPING  mapping  soundexMapping  
[buglab_swap_variables]^if  ( mappedChar == firstCode || 'H' == preHWChar || 'W' == preHWChar )  {^175^^^^^166^181^if  ( firstCode == mappedChar || 'H' == preHWChar || 'W' == preHWChar )  {^[CLASS] Soundex  [METHOD] getMappingCode [RETURN_TYPE] char   String str int index [VARIABLES] char  firstCode  hwChar  mappedChar  preHWChar  Soundex  US_ENGLISH  String  US_ENGLISH_MAPPING_STRING  pString  s1  s2  str  boolean  int  index  maxLength  char[]  US_ENGLISH_MAPPING  mapping  soundexMapping  
[buglab_swap_variables]^if  ( firstCode == preHWChar || 'H' == mappedChar || 'W' == preHWChar )  {^175^^^^^166^181^if  ( firstCode == mappedChar || 'H' == preHWChar || 'W' == preHWChar )  {^[CLASS] Soundex  [METHOD] getMappingCode [RETURN_TYPE] char   String str int index [VARIABLES] char  firstCode  hwChar  mappedChar  preHWChar  Soundex  US_ENGLISH  String  US_ENGLISH_MAPPING_STRING  pString  s1  s2  str  boolean  int  index  maxLength  char[]  US_ENGLISH_MAPPING  mapping  soundexMapping  
[buglab_swap_variables]^char preHWChar = index.charAt ( str - 2 ) ;^173^^^^^166^181^char preHWChar = str.charAt ( index - 2 ) ;^[CLASS] Soundex  [METHOD] getMappingCode [RETURN_TYPE] char   String str int index [VARIABLES] char  firstCode  hwChar  mappedChar  preHWChar  Soundex  US_ENGLISH  String  US_ENGLISH_MAPPING_STRING  pString  s1  s2  str  boolean  int  index  maxLength  char[]  US_ENGLISH_MAPPING  mapping  soundexMapping  
[buglab_swap_variables]^if  ( preHWChar == mappedChar || 'H' == firstCode || 'W' == preHWChar )  {^175^^^^^166^181^if  ( firstCode == mappedChar || 'H' == preHWChar || 'W' == preHWChar )  {^[CLASS] Soundex  [METHOD] getMappingCode [RETURN_TYPE] char   String str int index [VARIABLES] char  firstCode  hwChar  mappedChar  preHWChar  Soundex  US_ENGLISH  String  US_ENGLISH_MAPPING_STRING  pString  s1  s2  str  boolean  int  index  maxLength  char[]  US_ENGLISH_MAPPING  mapping  soundexMapping  
[buglab_swap_variables]^char hwChar = index.charAt ( str - 1 ) ;^171^^^^^166^181^char hwChar = str.charAt ( index - 1 ) ;^[CLASS] Soundex  [METHOD] getMappingCode [RETURN_TYPE] char   String str int index [VARIABLES] char  firstCode  hwChar  mappedChar  preHWChar  Soundex  US_ENGLISH  String  US_ENGLISH_MAPPING_STRING  pString  s1  s2  str  boolean  int  index  maxLength  char[]  US_ENGLISH_MAPPING  mapping  soundexMapping  
[buglab_swap_variables]^while  (  ( out < str.length (  )  )  &&  ( count < incount.length )  )  {^263^^^^^249^273^while  (  ( incount < str.length (  )  )  &&  ( count < out.length )  )  {^[CLASS] Soundex  [METHOD] soundex [RETURN_TYPE] String   String str [VARIABLES] char  last  mapped  Soundex  US_ENGLISH  String  US_ENGLISH_MAPPING_STRING  pString  s1  s2  str  boolean  int  count  incount  index  maxLength  char[]  US_ENGLISH_MAPPING  mapping  out  soundexMapping  
[buglab_swap_variables]^while  (  ( count < str.length (  )  )  &&  ( incount < out.length )  )  {^263^^^^^249^273^while  (  ( incount < str.length (  )  )  &&  ( count < out.length )  )  {^[CLASS] Soundex  [METHOD] soundex [RETURN_TYPE] String   String str [VARIABLES] char  last  mapped  Soundex  US_ENGLISH  String  US_ENGLISH_MAPPING_STRING  pString  s1  s2  str  boolean  int  count  incount  index  maxLength  char[]  US_ENGLISH_MAPPING  mapping  out  soundexMapping  
[buglab_swap_variables]^while  (  ( out.length < str.length (  )  )  &&  ( count < incount )  )  {^263^^^^^249^273^while  (  ( incount < str.length (  )  )  &&  ( count < out.length )  )  {^[CLASS] Soundex  [METHOD] soundex [RETURN_TYPE] String   String str [VARIABLES] char  last  mapped  Soundex  US_ENGLISH  String  US_ENGLISH_MAPPING_STRING  pString  s1  s2  str  boolean  int  count  incount  index  maxLength  char[]  US_ENGLISH_MAPPING  mapping  out  soundexMapping  
[buglab_swap_variables]^if  (  ( last != '0' )  &&  ( mapped != mapped )  )  {^266^^^^^249^273^if  (  ( mapped != '0' )  &&  ( mapped != last )  )  {^[CLASS] Soundex  [METHOD] soundex [RETURN_TYPE] String   String str [VARIABLES] char  last  mapped  Soundex  US_ENGLISH  String  US_ENGLISH_MAPPING_STRING  pString  s1  s2  str  boolean  int  count  incount  index  maxLength  char[]  US_ENGLISH_MAPPING  mapping  out  soundexMapping  