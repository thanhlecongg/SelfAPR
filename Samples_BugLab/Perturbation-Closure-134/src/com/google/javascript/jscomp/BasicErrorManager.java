[buglab_swap_variables]^if  ( messages.add ( Pair.of ( level, error )  )  )  {^48^^^^^47^55^if  ( messages.add ( Pair.of ( error, level )  )  )  {^[CLASS] BasicErrorManager LeveledJSErrorComparator  [METHOD] report [RETURN_TYPE] void   CheckLevel level JSError error [VARIABLES] boolean  SortedSet  messages  double  typedPercent  int  P1_GT_P2  P1_LT_P2  errorCount  warningCount  CheckLevel  level  JSError  error  
[buglab_swap_variables]^if  ( messages.add ( Pair.of (  level )  )  )  {^48^^^^^47^55^if  ( messages.add ( Pair.of ( error, level )  )  )  {^[CLASS] BasicErrorManager LeveledJSErrorComparator  [METHOD] report [RETURN_TYPE] void   CheckLevel level JSError error [VARIABLES] boolean  SortedSet  messages  double  typedPercent  int  P1_GT_P2  P1_LT_P2  errorCount  warningCount  CheckLevel  level  JSError  error  
[buglab_swap_variables]^if  ( messages.add ( Pair.of ( error )  )  )  {^48^^^^^47^55^if  ( messages.add ( Pair.of ( error, level )  )  )  {^[CLASS] BasicErrorManager LeveledJSErrorComparator  [METHOD] report [RETURN_TYPE] void   CheckLevel level JSError error [VARIABLES] boolean  SortedSet  messages  double  typedPercent  int  P1_GT_P2  P1_LT_P2  errorCount  warningCount  CheckLevel  level  JSError  error  
[buglab_swap_variables]^if  ( level.add ( Pair.of ( error, messages )  )  )  {^48^^^^^47^55^if  ( messages.add ( Pair.of ( error, level )  )  )  {^[CLASS] BasicErrorManager LeveledJSErrorComparator  [METHOD] report [RETURN_TYPE] void   CheckLevel level JSError error [VARIABLES] boolean  SortedSet  messages  double  typedPercent  int  P1_GT_P2  P1_LT_P2  errorCount  warningCount  CheckLevel  level  JSError  error  
[buglab_swap_variables]^if  ( error.add ( Pair.of ( messages, level )  )  )  {^48^^^^^47^55^if  ( messages.add ( Pair.of ( error, level )  )  )  {^[CLASS] BasicErrorManager LeveledJSErrorComparator  [METHOD] report [RETURN_TYPE] void   CheckLevel level JSError error [VARIABLES] boolean  SortedSet  messages  double  typedPercent  int  P1_GT_P2  P1_LT_P2  errorCount  warningCount  CheckLevel  level  JSError  error  
[buglab_swap_variables]^if  ( source2 != null && source1 != null )  {^142^^^^^127^157^if  ( source1 != null && source2 != null )  {^[CLASS] BasicErrorManager LeveledJSErrorComparator  [METHOD] compare [RETURN_TYPE] int   CheckLevel> p1 CheckLevel> p2 [VARIABLES] String  source1  source2  boolean  SortedSet  messages  double  typedPercent  int  P1_GT_P2  P1_LT_P2  charno1  charno2  errorCount  lineno1  lineno2  sourceCompare  warningCount  Pair  p1  p2  
[buglab_swap_variables]^} else if  ( source2 != null && source1 == null )  {^149^^^^^134^164^} else if  ( source1 != null && source2 == null )  {^[CLASS] BasicErrorManager LeveledJSErrorComparator  [METHOD] compare [RETURN_TYPE] int   CheckLevel> p1 CheckLevel> p2 [VARIABLES] String  source1  source2  boolean  SortedSet  messages  double  typedPercent  int  P1_GT_P2  P1_LT_P2  charno1  charno2  errorCount  lineno1  lineno2  sourceCompare  warningCount  Pair  p1  p2  
[buglab_swap_variables]^int sourceCompare = source2.compareTo ( source1 ) ;^143^^^^^128^158^int sourceCompare = source1.compareTo ( source2 ) ;^[CLASS] BasicErrorManager LeveledJSErrorComparator  [METHOD] compare [RETURN_TYPE] int   CheckLevel> p1 CheckLevel> p2 [VARIABLES] String  source1  source2  boolean  SortedSet  messages  double  typedPercent  int  P1_GT_P2  P1_LT_P2  charno1  charno2  errorCount  lineno1  lineno2  sourceCompare  warningCount  Pair  p1  p2  
[buglab_swap_variables]^} else if  ( source2 == null && source1 != null )  {^147^^^^^132^162^} else if  ( source1 == null && source2 != null )  {^[CLASS] BasicErrorManager LeveledJSErrorComparator  [METHOD] compare [RETURN_TYPE] int   CheckLevel> p1 CheckLevel> p2 [VARIABLES] String  source1  source2  boolean  SortedSet  messages  double  typedPercent  int  P1_GT_P2  P1_LT_P2  charno1  charno2  errorCount  lineno1  lineno2  sourceCompare  warningCount  Pair  p1  p2  
[buglab_swap_variables]^if  ( lineno2 != lineno1 )  {^155^^^^^140^170^if  ( lineno1 != lineno2 )  {^[CLASS] BasicErrorManager LeveledJSErrorComparator  [METHOD] compare [RETURN_TYPE] int   CheckLevel> p1 CheckLevel> p2 [VARIABLES] String  source1  source2  boolean  SortedSet  messages  double  typedPercent  int  P1_GT_P2  P1_LT_P2  charno1  charno2  errorCount  lineno1  lineno2  sourceCompare  warningCount  Pair  p1  p2  
[buglab_swap_variables]^} else if  ( 0 <= lineno2 && lineno1 < 0 )  {^159^^^^^144^174^} else if  ( 0 <= lineno1 && lineno2 < 0 )  {^[CLASS] BasicErrorManager LeveledJSErrorComparator  [METHOD] compare [RETURN_TYPE] int   CheckLevel> p1 CheckLevel> p2 [VARIABLES] String  source1  source2  boolean  SortedSet  messages  double  typedPercent  int  P1_GT_P2  P1_LT_P2  charno1  charno2  errorCount  lineno1  lineno2  sourceCompare  warningCount  Pair  p1  p2  
[buglab_swap_variables]^return lineno2 - lineno1;^156^^^^^141^171^return lineno1 - lineno2;^[CLASS] BasicErrorManager LeveledJSErrorComparator  [METHOD] compare [RETURN_TYPE] int   CheckLevel> p1 CheckLevel> p2 [VARIABLES] String  source1  source2  boolean  SortedSet  messages  double  typedPercent  int  P1_GT_P2  P1_LT_P2  charno1  charno2  errorCount  lineno1  lineno2  sourceCompare  warningCount  Pair  p1  p2  
[buglab_swap_variables]^} else if  ( lineno2 < 0 && 0 <= lineno1 )  {^157^^^^^142^172^} else if  ( lineno1 < 0 && 0 <= lineno2 )  {^[CLASS] BasicErrorManager LeveledJSErrorComparator  [METHOD] compare [RETURN_TYPE] int   CheckLevel> p1 CheckLevel> p2 [VARIABLES] String  source1  source2  boolean  SortedSet  messages  double  typedPercent  int  P1_GT_P2  P1_LT_P2  charno1  charno2  errorCount  lineno1  lineno2  sourceCompare  warningCount  Pair  p1  p2  
[buglab_swap_variables]^} else if  ( 0 <= charno2 && charno1 < 0 )  {^169^^^^^154^184^} else if  ( 0 <= charno1 && charno2 < 0 )  {^[CLASS] BasicErrorManager LeveledJSErrorComparator  [METHOD] compare [RETURN_TYPE] int   CheckLevel> p1 CheckLevel> p2 [VARIABLES] String  source1  source2  boolean  SortedSet  messages  double  typedPercent  int  P1_GT_P2  P1_LT_P2  charno1  charno2  errorCount  lineno1  lineno2  sourceCompare  warningCount  Pair  p1  p2  
[buglab_swap_variables]^return charno2 - charno1;^166^^^^^151^181^return charno1 - charno2;^[CLASS] BasicErrorManager LeveledJSErrorComparator  [METHOD] compare [RETURN_TYPE] int   CheckLevel> p1 CheckLevel> p2 [VARIABLES] String  source1  source2  boolean  SortedSet  messages  double  typedPercent  int  P1_GT_P2  P1_LT_P2  charno1  charno2  errorCount  lineno1  lineno2  sourceCompare  warningCount  Pair  p1  p2  
[buglab_swap_variables]^} else if  ( source2 != null && source1 == null )  {^149^^^^^134^164^} else if  ( source1 != null && source2 == null )  {^[CLASS] LeveledJSErrorComparator  [METHOD] compare [RETURN_TYPE] int   CheckLevel> p1 CheckLevel> p2 [VARIABLES] String  source1  source2  boolean  int  P1_GT_P2  P1_LT_P2  charno1  charno2  lineno1  lineno2  sourceCompare  Pair  p1  p2  
[buglab_swap_variables]^int sourceCompare = source2.compareTo ( source1 ) ;^143^^^^^128^158^int sourceCompare = source1.compareTo ( source2 ) ;^[CLASS] LeveledJSErrorComparator  [METHOD] compare [RETURN_TYPE] int   CheckLevel> p1 CheckLevel> p2 [VARIABLES] String  source1  source2  boolean  int  P1_GT_P2  P1_LT_P2  charno1  charno2  lineno1  lineno2  sourceCompare  Pair  p1  p2  
[buglab_swap_variables]^} else if  ( source2 == null && source1 != null )  {^147^^^^^132^162^} else if  ( source1 == null && source2 != null )  {^[CLASS] LeveledJSErrorComparator  [METHOD] compare [RETURN_TYPE] int   CheckLevel> p1 CheckLevel> p2 [VARIABLES] String  source1  source2  boolean  int  P1_GT_P2  P1_LT_P2  charno1  charno2  lineno1  lineno2  sourceCompare  Pair  p1  p2  
[buglab_swap_variables]^if  ( lineno2 != lineno1 )  {^155^^^^^140^170^if  ( lineno1 != lineno2 )  {^[CLASS] LeveledJSErrorComparator  [METHOD] compare [RETURN_TYPE] int   CheckLevel> p1 CheckLevel> p2 [VARIABLES] String  source1  source2  boolean  int  P1_GT_P2  P1_LT_P2  charno1  charno2  lineno1  lineno2  sourceCompare  Pair  p1  p2  
[buglab_swap_variables]^} else if  ( lineno2 < 0 && 0 <= lineno1 )  {^157^^^^^142^172^} else if  ( lineno1 < 0 && 0 <= lineno2 )  {^[CLASS] LeveledJSErrorComparator  [METHOD] compare [RETURN_TYPE] int   CheckLevel> p1 CheckLevel> p2 [VARIABLES] String  source1  source2  boolean  int  P1_GT_P2  P1_LT_P2  charno1  charno2  lineno1  lineno2  sourceCompare  Pair  p1  p2  
[buglab_swap_variables]^} else if  ( 0 <= lineno2 && lineno1 < 0 )  {^159^^^^^144^174^} else if  ( 0 <= lineno1 && lineno2 < 0 )  {^[CLASS] LeveledJSErrorComparator  [METHOD] compare [RETURN_TYPE] int   CheckLevel> p1 CheckLevel> p2 [VARIABLES] String  source1  source2  boolean  int  P1_GT_P2  P1_LT_P2  charno1  charno2  lineno1  lineno2  sourceCompare  Pair  p1  p2  
[buglab_swap_variables]^return lineno2 - lineno1;^156^^^^^141^171^return lineno1 - lineno2;^[CLASS] LeveledJSErrorComparator  [METHOD] compare [RETURN_TYPE] int   CheckLevel> p1 CheckLevel> p2 [VARIABLES] String  source1  source2  boolean  int  P1_GT_P2  P1_LT_P2  charno1  charno2  lineno1  lineno2  sourceCompare  Pair  p1  p2  
[buglab_swap_variables]^} else if  ( 0 <= charno2 && charno1 < 0 )  {^169^^^^^154^184^} else if  ( 0 <= charno1 && charno2 < 0 )  {^[CLASS] LeveledJSErrorComparator  [METHOD] compare [RETURN_TYPE] int   CheckLevel> p1 CheckLevel> p2 [VARIABLES] String  source1  source2  boolean  int  P1_GT_P2  P1_LT_P2  charno1  charno2  lineno1  lineno2  sourceCompare  Pair  p1  p2  
[buglab_swap_variables]^return charno2 - charno1;^166^^^^^151^181^return charno1 - charno2;^[CLASS] LeveledJSErrorComparator  [METHOD] compare [RETURN_TYPE] int   CheckLevel> p1 CheckLevel> p2 [VARIABLES] String  source1  source2  boolean  int  P1_GT_P2  P1_LT_P2  charno1  charno2  lineno1  lineno2  sourceCompare  Pair  p1  p2  