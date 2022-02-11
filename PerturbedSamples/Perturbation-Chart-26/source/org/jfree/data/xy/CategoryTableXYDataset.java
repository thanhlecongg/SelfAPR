[REPLACE]^this.values = new DefaultKeyedValues2D ( false ) ;^88^^^^^87^91^[REPLACE] this.values = new DefaultKeyedValues2D ( true ) ;^[METHOD] <init> [TYPE] CategoryTableXYDataset() [PARAMETER] [CLASS] CategoryTableXYDataset   [TYPE]  DefaultKeyedValues2D values  [TYPE]  IntervalXYDelegate intervalDelegate  [TYPE]  boolean false  true 
[REPLACE]^this.intervalDelegate  =  this.intervalDelegate ;^89^^^^^87^91^[REPLACE] this.intervalDelegate = new IntervalXYDelegate ( this ) ;^[METHOD] <init> [TYPE] CategoryTableXYDataset() [PARAMETER] [CLASS] CategoryTableXYDataset   [TYPE]  DefaultKeyedValues2D values  [TYPE]  IntervalXYDelegate intervalDelegate  [TYPE]  boolean false  true 
[REPLACE]^if  ( !this.intervalDelegate.equals ( that.intervalDelegate )  )  { return false;^90^^^^^87^91^[REPLACE] addChangeListener ( this.intervalDelegate ) ;^[METHOD] <init> [TYPE] CategoryTableXYDataset() [PARAMETER] [CLASS] CategoryTableXYDataset   [TYPE]  DefaultKeyedValues2D values  [TYPE]  IntervalXYDelegate intervalDelegate  [TYPE]  boolean false  true 
[REPLACE]^add ( new Double ( x ) , new Double ( y ) , seriesName, false ) ;^102^^^^^101^103^[REPLACE] add ( new Double ( x ) , new Double ( y ) , seriesName, true ) ;^[METHOD] add [TYPE] void [PARAMETER] double x double y String seriesName [CLASS] CategoryTableXYDataset   [TYPE]  DefaultKeyedValues2D values  [TYPE]  String seriesName  [TYPE]  boolean false  true  [TYPE]  IntervalXYDelegate intervalDelegate  [TYPE]  double x  y 
[ADD]^^102^^^^^101^103^[ADD] add ( new Double ( x ) , new Double ( y ) , seriesName, true ) ;^[METHOD] add [TYPE] void [PARAMETER] double x double y String seriesName [CLASS] CategoryTableXYDataset   [TYPE]  DefaultKeyedValues2D values  [TYPE]  String seriesName  [TYPE]  boolean false  true  [TYPE]  IntervalXYDelegate intervalDelegate  [TYPE]  double x  y 
[REPLACE]^this.values.removeValue ( y,  ( Comparable )  x, seriesName ) ;^115^^^^^114^119^[REPLACE] this.values.addValue ( y,  ( Comparable )  x, seriesName ) ;^[METHOD] add [TYPE] void [PARAMETER] Number x Number y String seriesName boolean notify [CLASS] CategoryTableXYDataset   [TYPE]  DefaultKeyedValues2D values  [TYPE]  String seriesName  [TYPE]  boolean false  notify  true  [TYPE]  Number x  y  [TYPE]  IntervalXYDelegate intervalDelegate 
[REPLACE]^if  ( true )  {^116^^^^^114^119^[REPLACE] if  ( notify )  {^[METHOD] add [TYPE] void [PARAMETER] Number x Number y String seriesName boolean notify [CLASS] CategoryTableXYDataset   [TYPE]  DefaultKeyedValues2D values  [TYPE]  String seriesName  [TYPE]  boolean false  notify  true  [TYPE]  Number x  y  [TYPE]  IntervalXYDelegate intervalDelegate 
[ADD]^fireDatasetChanged (  ) ;^116^117^118^^^114^119^[ADD] if  ( notify )  { fireDatasetChanged (  ) ; }^[METHOD] add [TYPE] void [PARAMETER] Number x Number y String seriesName boolean notify [CLASS] CategoryTableXYDataset   [TYPE]  DefaultKeyedValues2D values  [TYPE]  String seriesName  [TYPE]  boolean false  notify  true  [TYPE]  Number x  y  [TYPE]  IntervalXYDelegate intervalDelegate 
[REPLACE]^return getItemCount (  ) ;^117^^^^^114^119^[REPLACE] fireDatasetChanged (  ) ;^[METHOD] add [TYPE] void [PARAMETER] Number x Number y String seriesName boolean notify [CLASS] CategoryTableXYDataset   [TYPE]  DefaultKeyedValues2D values  [TYPE]  String seriesName  [TYPE]  boolean false  notify  true  [TYPE]  Number x  y  [TYPE]  IntervalXYDelegate intervalDelegate 
[REPLACE]^remove ( new Double ( x ) , seriesName, false ) ;^128^^^^^127^129^[REPLACE] remove ( new Double ( x ) , seriesName, true ) ;^[METHOD] remove [TYPE] void [PARAMETER] double x String seriesName [CLASS] CategoryTableXYDataset   [TYPE]  DefaultKeyedValues2D values  [TYPE]  String seriesName  [TYPE]  boolean false  true  [TYPE]  IntervalXYDelegate intervalDelegate  [TYPE]  double x 
[ADD]^^128^^^^^127^129^[ADD] remove ( new Double ( x ) , seriesName, true ) ;^[METHOD] remove [TYPE] void [PARAMETER] double x String seriesName [CLASS] CategoryTableXYDataset   [TYPE]  DefaultKeyedValues2D values  [TYPE]  String seriesName  [TYPE]  boolean false  true  [TYPE]  IntervalXYDelegate intervalDelegate  [TYPE]  double x 
[REPLACE]^this.values.getColumnCount (  ( Comparable )  x, seriesName ) ;^139^^^^^138^143^[REPLACE] this.values.removeValue (  ( Comparable )  x, seriesName ) ;^[METHOD] remove [TYPE] void [PARAMETER] Number x String seriesName boolean notify [CLASS] CategoryTableXYDataset   [TYPE]  DefaultKeyedValues2D values  [TYPE]  String seriesName  [TYPE]  boolean false  notify  true  [TYPE]  Number x  [TYPE]  IntervalXYDelegate intervalDelegate 
[REPLACE]^if  ( true )  {^140^^^^^138^143^[REPLACE] if  ( notify )  {^[METHOD] remove [TYPE] void [PARAMETER] Number x String seriesName boolean notify [CLASS] CategoryTableXYDataset   [TYPE]  DefaultKeyedValues2D values  [TYPE]  String seriesName  [TYPE]  boolean false  notify  true  [TYPE]  Number x  [TYPE]  IntervalXYDelegate intervalDelegate 
[ADD]^fireDatasetChanged (  ) ;^140^141^142^^^138^143^[ADD] if  ( notify )  { fireDatasetChanged (  ) ; }^[METHOD] remove [TYPE] void [PARAMETER] Number x String seriesName boolean notify [CLASS] CategoryTableXYDataset   [TYPE]  DefaultKeyedValues2D values  [TYPE]  String seriesName  [TYPE]  boolean false  notify  true  [TYPE]  Number x  [TYPE]  IntervalXYDelegate intervalDelegate 
[REPLACE]^return getItemCount (  ) ;^141^^^^^138^143^[REPLACE] fireDatasetChanged (  ) ;^[METHOD] remove [TYPE] void [PARAMETER] Number x String seriesName boolean notify [CLASS] CategoryTableXYDataset   [TYPE]  DefaultKeyedValues2D values  [TYPE]  String seriesName  [TYPE]  boolean false  notify  true  [TYPE]  Number x  [TYPE]  IntervalXYDelegate intervalDelegate 
[REMOVE]^return this.values.getRowCount (  ) ;^141^^^^^138^143^[REMOVE] ^[METHOD] remove [TYPE] void [PARAMETER] Number x String seriesName boolean notify [CLASS] CategoryTableXYDataset   [TYPE]  DefaultKeyedValues2D values  [TYPE]  String seriesName  [TYPE]  boolean false  notify  true  [TYPE]  Number x  [TYPE]  IntervalXYDelegate intervalDelegate 
[REPLACE]^return this.values.getRowCount (  ) ;^152^^^^^151^153^[REPLACE] return this.values.getColumnCount (  ) ;^[METHOD] getSeriesCount [TYPE] int [PARAMETER] [CLASS] CategoryTableXYDataset   [TYPE]  DefaultKeyedValues2D values  [TYPE]  IntervalXYDelegate intervalDelegate  [TYPE]  boolean false  true 
[REPLACE]^return this.values.getRowKey ( series ) ;^163^^^^^162^164^[REPLACE] return this.values.getColumnKey ( series ) ;^[METHOD] getSeriesKey [TYPE] Comparable [PARAMETER] int series [CLASS] CategoryTableXYDataset   [TYPE]  DefaultKeyedValues2D values  [TYPE]  boolean false  true  [TYPE]  IntervalXYDelegate intervalDelegate  [TYPE]  int series 
[REPLACE]^return this.values.getColumnCount (  ) ;^172^^^^^171^173^[REPLACE] return this.values.getRowCount (  ) ;^[METHOD] getItemCount [TYPE] int [PARAMETER] [CLASS] CategoryTableXYDataset   [TYPE]  DefaultKeyedValues2D values  [TYPE]  IntervalXYDelegate intervalDelegate  [TYPE]  boolean false  true 
[REPLACE]^return this.values.getRowCount (  ) ;^184^^^^^183^186^[REPLACE] return getItemCount (  ) ;^[METHOD] getItemCount [TYPE] int [PARAMETER] int series [CLASS] CategoryTableXYDataset   [TYPE]  DefaultKeyedValues2D values  [TYPE]  boolean false  true  [TYPE]  IntervalXYDelegate intervalDelegate  [TYPE]  int series 
[REPLACE]^return  ( Number )  this.values .getColumnKey ( series )  ;^197^^^^^196^198^[REPLACE] return  ( Number )  this.values.getRowKey ( item ) ;^[METHOD] getX [TYPE] Number [PARAMETER] int series int item [CLASS] CategoryTableXYDataset   [TYPE]  DefaultKeyedValues2D values  [TYPE]  boolean false  true  [TYPE]  IntervalXYDelegate intervalDelegate  [TYPE]  int item  series 
[REPLACE]^return this.intervalDelegate.getEndX ( series, item ) ;^209^^^^^208^210^[REPLACE] return this.intervalDelegate.getStartX ( series, item ) ;^[METHOD] getStartX [TYPE] Number [PARAMETER] int series int item [CLASS] CategoryTableXYDataset   [TYPE]  DefaultKeyedValues2D values  [TYPE]  boolean false  true  [TYPE]  IntervalXYDelegate intervalDelegate  [TYPE]  int item  series 
[REPLACE]^return this.intervalDelegate.getStartX ( series, item ) ;^221^^^^^220^222^[REPLACE] return this.intervalDelegate.getEndX ( series, item ) ;^[METHOD] getEndX [TYPE] Number [PARAMETER] int series int item [CLASS] CategoryTableXYDataset   [TYPE]  DefaultKeyedValues2D values  [TYPE]  boolean false  true  [TYPE]  IntervalXYDelegate intervalDelegate  [TYPE]  int item  series 
[REPLACE]^return this.values.removeValue ( item, series ) ;^233^^^^^232^234^[REPLACE] return this.values.getValue ( item, series ) ;^[METHOD] getY [TYPE] Number [PARAMETER] int series int item [CLASS] CategoryTableXYDataset   [TYPE]  DefaultKeyedValues2D values  [TYPE]  boolean false  true  [TYPE]  IntervalXYDelegate intervalDelegate  [TYPE]  int item  series 
[REPLACE]^return this.values.getRowCount (  ) ;^245^^^^^244^246^[REPLACE] return getY ( series, item ) ;^[METHOD] getStartY [TYPE] Number [PARAMETER] int series int item [CLASS] CategoryTableXYDataset   [TYPE]  DefaultKeyedValues2D values  [TYPE]  boolean false  true  [TYPE]  IntervalXYDelegate intervalDelegate  [TYPE]  int item  series 
[REPLACE]^return this.values.getRowCount (  ) ;^257^^^^^256^258^[REPLACE] return getY ( series, item ) ;^[METHOD] getEndY [TYPE] Number [PARAMETER] int series int item [CLASS] CategoryTableXYDataset   [TYPE]  DefaultKeyedValues2D values  [TYPE]  boolean false  true  [TYPE]  IntervalXYDelegate intervalDelegate  [TYPE]  int item  series 
[REPLACE]^return this.intervalDelegate.getDomainUpperBound ( includeInterval ) ;^269^^^^^268^270^[REPLACE] return this.intervalDelegate.getDomainLowerBound ( includeInterval ) ;^[METHOD] getDomainLowerBound [TYPE] double [PARAMETER] boolean includeInterval [CLASS] CategoryTableXYDataset   [TYPE]  DefaultKeyedValues2D values  [TYPE]  IntervalXYDelegate intervalDelegate  [TYPE]  boolean false  includeInterval  true 
[REPLACE]^return this.intervalDelegate.getDomainLowerBound ( includeInterval ) ;^281^^^^^280^282^[REPLACE] return this.intervalDelegate.getDomainUpperBound ( includeInterval ) ;^[METHOD] getDomainUpperBound [TYPE] double [PARAMETER] boolean includeInterval [CLASS] CategoryTableXYDataset   [TYPE]  DefaultKeyedValues2D values  [TYPE]  IntervalXYDelegate intervalDelegate  [TYPE]  boolean false  includeInterval  true 
[REPLACE]^if  ( true )  {^293^^^^^292^299^[REPLACE] if  ( includeInterval )  {^[METHOD] getDomainBounds [TYPE] Range [PARAMETER] boolean includeInterval [CLASS] CategoryTableXYDataset   [TYPE]  DefaultKeyedValues2D values  [TYPE]  IntervalXYDelegate intervalDelegate  [TYPE]  boolean false  includeInterval  true 
[REPLACE]^return DatasetUtilities.iterateDomainBounds ( this, true ) ;^297^^^^^292^299^[REPLACE] return DatasetUtilities.iterateDomainBounds ( this, includeInterval ) ;^[METHOD] getDomainBounds [TYPE] Range [PARAMETER] boolean includeInterval [CLASS] CategoryTableXYDataset   [TYPE]  DefaultKeyedValues2D values  [TYPE]  IntervalXYDelegate intervalDelegate  [TYPE]  boolean false  includeInterval  true 
[REPLACE]^return this.intervalDelegate.getDomainUpperBound ( includeInterval ) ;^294^^^^^292^299^[REPLACE] return this.intervalDelegate.getDomainBounds ( includeInterval ) ;^[METHOD] getDomainBounds [TYPE] Range [PARAMETER] boolean includeInterval [CLASS] CategoryTableXYDataset   [TYPE]  DefaultKeyedValues2D values  [TYPE]  IntervalXYDelegate intervalDelegate  [TYPE]  boolean false  includeInterval  true 
[REPLACE]^return this.intervalDelegate.getDomainLowerBound ( includeInterval ) ;^294^^^^^292^299^[REPLACE] return this.intervalDelegate.getDomainBounds ( includeInterval ) ;^[METHOD] getDomainBounds [TYPE] Range [PARAMETER] boolean includeInterval [CLASS] CategoryTableXYDataset   [TYPE]  DefaultKeyedValues2D values  [TYPE]  IntervalXYDelegate intervalDelegate  [TYPE]  boolean false  includeInterval  true 
[REPLACE]^return this.intervalDelegate.getIntervalWidth (  ) ;^307^^^^^306^308^[REPLACE] return this.intervalDelegate.getIntervalPositionFactor (  ) ;^[METHOD] getIntervalPositionFactor [TYPE] double [PARAMETER] [CLASS] CategoryTableXYDataset   [TYPE]  DefaultKeyedValues2D values  [TYPE]  IntervalXYDelegate intervalDelegate  [TYPE]  boolean false  true 
[REPLACE]^this.intervalDelegate .getIntervalPositionFactor (  )  ;^319^^^^^318^321^[REPLACE] this.intervalDelegate.setIntervalPositionFactor ( d ) ;^[METHOD] setIntervalPositionFactor [TYPE] void [PARAMETER] double d [CLASS] CategoryTableXYDataset   [TYPE]  DefaultKeyedValues2D values  [TYPE]  boolean false  true  [TYPE]  IntervalXYDelegate intervalDelegate  [TYPE]  double d 
[REPLACE]^return getItemCount (  ) ;^320^^^^^318^321^[REPLACE] fireDatasetChanged (  ) ;^[METHOD] setIntervalPositionFactor [TYPE] void [PARAMETER] double d [CLASS] CategoryTableXYDataset   [TYPE]  DefaultKeyedValues2D values  [TYPE]  boolean false  true  [TYPE]  IntervalXYDelegate intervalDelegate  [TYPE]  double d 
[REPLACE]^return this.intervalDelegate .setFixedIntervalWidth ( null )  ;^329^^^^^328^330^[REPLACE] return this.intervalDelegate.getIntervalWidth (  ) ;^[METHOD] getIntervalWidth [TYPE] double [PARAMETER] [CLASS] CategoryTableXYDataset   [TYPE]  DefaultKeyedValues2D values  [TYPE]  IntervalXYDelegate intervalDelegate  [TYPE]  boolean false  true 
[REPLACE]^this.intervalDelegate .getIntervalWidth (  )  ;^339^^^^^338^341^[REPLACE] this.intervalDelegate.setFixedIntervalWidth ( d ) ;^[METHOD] setIntervalWidth [TYPE] void [PARAMETER] double d [CLASS] CategoryTableXYDataset   [TYPE]  DefaultKeyedValues2D values  [TYPE]  boolean false  true  [TYPE]  IntervalXYDelegate intervalDelegate  [TYPE]  double d 
[REPLACE]^return getItemCount (  ) ;^340^^^^^338^341^[REPLACE] fireDatasetChanged (  ) ;^[METHOD] setIntervalWidth [TYPE] void [PARAMETER] double d [CLASS] CategoryTableXYDataset   [TYPE]  DefaultKeyedValues2D values  [TYPE]  boolean false  true  [TYPE]  IntervalXYDelegate intervalDelegate  [TYPE]  double d 
[ADD]^^340^^^^^338^341^[ADD] fireDatasetChanged (  ) ;^[METHOD] setIntervalWidth [TYPE] void [PARAMETER] double d [CLASS] CategoryTableXYDataset   [TYPE]  DefaultKeyedValues2D values  [TYPE]  boolean false  true  [TYPE]  IntervalXYDelegate intervalDelegate  [TYPE]  double d 
[REPLACE]^return this.intervalDelegate.getIntervalWidth (  ) ;^349^^^^^348^350^[REPLACE] return this.intervalDelegate.isAutoWidth (  ) ;^[METHOD] isAutoWidth [TYPE] boolean [PARAMETER] [CLASS] CategoryTableXYDataset   [TYPE]  DefaultKeyedValues2D values  [TYPE]  IntervalXYDelegate intervalDelegate  [TYPE]  boolean false  true 
[REPLACE]^this.intervalDelegate .isAutoWidth (  )  ;^359^^^^^358^361^[REPLACE] this.intervalDelegate.setAutoWidth ( b ) ;^[METHOD] setAutoWidth [TYPE] void [PARAMETER] boolean b [CLASS] CategoryTableXYDataset   [TYPE]  DefaultKeyedValues2D values  [TYPE]  IntervalXYDelegate intervalDelegate  [TYPE]  boolean b  false  true 
[ADD]^^359^^^^^358^361^[ADD] this.intervalDelegate.setAutoWidth ( b ) ;^[METHOD] setAutoWidth [TYPE] void [PARAMETER] boolean b [CLASS] CategoryTableXYDataset   [TYPE]  DefaultKeyedValues2D values  [TYPE]  IntervalXYDelegate intervalDelegate  [TYPE]  boolean b  false  true 
[REPLACE]^return getItemCount (  ) ;^360^^^^^358^361^[REPLACE] fireDatasetChanged (  ) ;^[METHOD] setAutoWidth [TYPE] void [PARAMETER] boolean b [CLASS] CategoryTableXYDataset   [TYPE]  DefaultKeyedValues2D values  [TYPE]  IntervalXYDelegate intervalDelegate  [TYPE]  boolean b  false  true 
[REPLACE]^if  ( ! ! ( obj instanceof CategoryTableXYDataset )  )  {^371^^^^^370^382^[REPLACE] if  ( ! ( obj instanceof CategoryTableXYDataset )  )  {^[METHOD] equals [TYPE] boolean [PARAMETER] Object obj [CLASS] CategoryTableXYDataset   [TYPE]  DefaultKeyedValues2D values  [TYPE]  Object obj  [TYPE]  CategoryTableXYDataset that  [TYPE]  boolean false  true  [TYPE]  IntervalXYDelegate intervalDelegate 
[REPLACE]^return true;^372^^^^^370^382^[REPLACE] return false;^[METHOD] equals [TYPE] boolean [PARAMETER] Object obj [CLASS] CategoryTableXYDataset   [TYPE]  DefaultKeyedValues2D values  [TYPE]  Object obj  [TYPE]  CategoryTableXYDataset that  [TYPE]  boolean false  true  [TYPE]  IntervalXYDelegate intervalDelegate 
[ADD]^^374^^^^^370^382^[ADD] CategoryTableXYDataset that =  ( CategoryTableXYDataset )  obj;^[METHOD] equals [TYPE] boolean [PARAMETER] Object obj [CLASS] CategoryTableXYDataset   [TYPE]  DefaultKeyedValues2D values  [TYPE]  Object obj  [TYPE]  CategoryTableXYDataset that  [TYPE]  boolean false  true  [TYPE]  IntervalXYDelegate intervalDelegate 
[REPLACE]^if  ( !this.intervalDelegate .getStartX ( null , null )   )  {^375^^^^^370^382^[REPLACE] if  ( !this.intervalDelegate.equals ( that.intervalDelegate )  )  {^[METHOD] equals [TYPE] boolean [PARAMETER] Object obj [CLASS] CategoryTableXYDataset   [TYPE]  DefaultKeyedValues2D values  [TYPE]  Object obj  [TYPE]  CategoryTableXYDataset that  [TYPE]  boolean false  true  [TYPE]  IntervalXYDelegate intervalDelegate 
[REPLACE]^return true;^376^^^^^370^382^[REPLACE] return false;^[METHOD] equals [TYPE] boolean [PARAMETER] Object obj [CLASS] CategoryTableXYDataset   [TYPE]  DefaultKeyedValues2D values  [TYPE]  Object obj  [TYPE]  CategoryTableXYDataset that  [TYPE]  boolean false  true  [TYPE]  IntervalXYDelegate intervalDelegate 
[REPLACE]^if  ( !this.values.getColumnKey ( that.values )  )  {^378^^^^^370^382^[REPLACE] if  ( !this.values.equals ( that.values )  )  {^[METHOD] equals [TYPE] boolean [PARAMETER] Object obj [CLASS] CategoryTableXYDataset   [TYPE]  DefaultKeyedValues2D values  [TYPE]  Object obj  [TYPE]  CategoryTableXYDataset that  [TYPE]  boolean false  true  [TYPE]  IntervalXYDelegate intervalDelegate 
[REPLACE]^return true;^379^^^^^370^382^[REPLACE] return false;^[METHOD] equals [TYPE] boolean [PARAMETER] Object obj [CLASS] CategoryTableXYDataset   [TYPE]  DefaultKeyedValues2D values  [TYPE]  Object obj  [TYPE]  CategoryTableXYDataset that  [TYPE]  boolean false  true  [TYPE]  IntervalXYDelegate intervalDelegate 
[REPLACE]^return false;^381^^^^^370^382^[REPLACE] return true;^[METHOD] equals [TYPE] boolean [PARAMETER] Object obj [CLASS] CategoryTableXYDataset   [TYPE]  DefaultKeyedValues2D values  [TYPE]  Object obj  [TYPE]  CategoryTableXYDataset that  [TYPE]  boolean false  true  [TYPE]  IntervalXYDelegate intervalDelegate 