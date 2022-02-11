[REPLACE]^private static final long serialVersionUID ;^64^^^^^^^[REPLACE] private static final long serialVersionUID = -3853824986520333437L;^ [CLASS] IntervalCategoryToolTipGenerator  
[REPLACE]^public static final String DEFAULT_TOOL_TIP_FORMAT_STRING ;^67^68^^^^67^68^[REPLACE] public static final String DEFAULT_TOOL_TIP_FORMAT_STRING = " ( {0}, {1} )  = {3} - {4}";^ [CLASS] IntervalCategoryToolTipGenerator  
[REPLACE]^super ( labelFormat, formatter ) ;^74^^^^^73^75^[REPLACE] super ( DEFAULT_TOOL_TIP_FORMAT_STRING, NumberFormat.getInstance (  )  ) ;^[METHOD] <init> [TYPE] IntervalCategoryToolTipGenerator() [PARAMETER] [CLASS] IntervalCategoryToolTipGenerator   [TYPE]  long serialVersionUID  [TYPE]  String DEFAULT_TOOL_TIP_FORMAT_STRING  [TYPE]  boolean false  true 
[ADD]^^74^^^^^73^75^[ADD] super ( DEFAULT_TOOL_TIP_FORMAT_STRING, NumberFormat.getInstance (  )  ) ;^[METHOD] <init> [TYPE] IntervalCategoryToolTipGenerator() [PARAMETER] [CLASS] IntervalCategoryToolTipGenerator   [TYPE]  long serialVersionUID  [TYPE]  String DEFAULT_TOOL_TIP_FORMAT_STRING  [TYPE]  boolean false  true 
[REPLACE]^super ( DEFAULT_TOOL_TIP_FORMAT_STRING, formatter ) ;^86^^^^^84^87^[REPLACE] super ( labelFormat, formatter ) ;^[METHOD] <init> [TYPE] NumberFormat) [PARAMETER] String labelFormat NumberFormat formatter [CLASS] IntervalCategoryToolTipGenerator   [TYPE]  String DEFAULT_TOOL_TIP_FORMAT_STRING  labelFormat  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  NumberFormat formatter 
[REPLACE]^super ( DEFAULT_TOOL_TIP_FORMAT_STRING, formatter ) ;^98^^^^^96^99^[REPLACE] super ( labelFormat, formatter ) ;^[METHOD] <init> [TYPE] DateFormat) [PARAMETER] String labelFormat DateFormat formatter [CLASS] IntervalCategoryToolTipGenerator   [TYPE]  String DEFAULT_TOOL_TIP_FORMAT_STRING  labelFormat  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  DateFormat formatter 
[REPLACE]^Object[] result = new Object[3];^113^^^^^112^138^[REPLACE] Object[] result = new Object[5];^[METHOD] createItemArray [TYPE] Object[] [PARAMETER] CategoryDataset dataset int row int column [CLASS] IntervalCategoryToolTipGenerator   [TYPE]  IntervalCategoryDataset icd  [TYPE]  boolean false  true  [TYPE]  Number end  start  value  [TYPE]  CategoryDataset dataset  [TYPE]  String DEFAULT_TOOL_TIP_FORMAT_STRING  labelFormat  [TYPE]  long serialVersionUID  [TYPE]  int column  row  [TYPE]  Object[] result 
[REPLACE]^result[2] = dataset.getRowKey ( row ) .toString (  ) ;^114^^^^^112^138^[REPLACE] result[0] = dataset.getRowKey ( row ) .toString (  ) ;^[METHOD] createItemArray [TYPE] Object[] [PARAMETER] CategoryDataset dataset int row int column [CLASS] IntervalCategoryToolTipGenerator   [TYPE]  IntervalCategoryDataset icd  [TYPE]  boolean false  true  [TYPE]  Number end  start  value  [TYPE]  CategoryDataset dataset  [TYPE]  String DEFAULT_TOOL_TIP_FORMAT_STRING  labelFormat  [TYPE]  long serialVersionUID  [TYPE]  int column  row  [TYPE]  Object[] result 
[REPLACE]^result[0] = dataset.getRowKey ( row ) .toString (  ) ; ;^115^^^^^112^138^[REPLACE] result[1] = dataset.getColumnKey ( column ) .toString (  ) ;^[METHOD] createItemArray [TYPE] Object[] [PARAMETER] CategoryDataset dataset int row int column [CLASS] IntervalCategoryToolTipGenerator   [TYPE]  IntervalCategoryDataset icd  [TYPE]  boolean false  true  [TYPE]  Number end  start  value  [TYPE]  CategoryDataset dataset  [TYPE]  String DEFAULT_TOOL_TIP_FORMAT_STRING  labelFormat  [TYPE]  long serialVersionUID  [TYPE]  int column  row  [TYPE]  Object[] result 
[REPLACE]^Number value = dataset .getColumnKey ( column )  ;^116^^^^^112^138^[REPLACE] Number value = dataset.getValue ( row, column ) ;^[METHOD] createItemArray [TYPE] Object[] [PARAMETER] CategoryDataset dataset int row int column [CLASS] IntervalCategoryToolTipGenerator   [TYPE]  IntervalCategoryDataset icd  [TYPE]  boolean false  true  [TYPE]  Number end  start  value  [TYPE]  CategoryDataset dataset  [TYPE]  String DEFAULT_TOOL_TIP_FORMAT_STRING  labelFormat  [TYPE]  long serialVersionUID  [TYPE]  int column  row  [TYPE]  Object[] result 
[REPLACE]^if  ( getNumberFormat (  )  == null )  {^117^^^^^112^138^[REPLACE] if  ( getNumberFormat (  )  != null )  {^[METHOD] createItemArray [TYPE] Object[] [PARAMETER] CategoryDataset dataset int row int column [CLASS] IntervalCategoryToolTipGenerator   [TYPE]  IntervalCategoryDataset icd  [TYPE]  boolean false  true  [TYPE]  Number end  start  value  [TYPE]  CategoryDataset dataset  [TYPE]  String DEFAULT_TOOL_TIP_FORMAT_STRING  labelFormat  [TYPE]  long serialVersionUID  [TYPE]  int column  row  [TYPE]  Object[] result 
[REPLACE]^else {^120^^^^^112^138^[REPLACE] else if  ( getDateFormat (  )  != null )  {^[METHOD] createItemArray [TYPE] Object[] [PARAMETER] CategoryDataset dataset int row int column [CLASS] IntervalCategoryToolTipGenerator   [TYPE]  IntervalCategoryDataset icd  [TYPE]  boolean false  true  [TYPE]  Number end  start  value  [TYPE]  CategoryDataset dataset  [TYPE]  String DEFAULT_TOOL_TIP_FORMAT_STRING  labelFormat  [TYPE]  long serialVersionUID  [TYPE]  int column  row  [TYPE]  Object[] result 
[REPLACE]^result[0] = getDateFormat (  ) .format ( value ) ;^121^^^^^112^138^[REPLACE] result[2] = getDateFormat (  ) .format ( value ) ;^[METHOD] createItemArray [TYPE] Object[] [PARAMETER] CategoryDataset dataset int row int column [CLASS] IntervalCategoryToolTipGenerator   [TYPE]  IntervalCategoryDataset icd  [TYPE]  boolean false  true  [TYPE]  Number end  start  value  [TYPE]  CategoryDataset dataset  [TYPE]  String DEFAULT_TOOL_TIP_FORMAT_STRING  labelFormat  [TYPE]  long serialVersionUID  [TYPE]  int column  row  [TYPE]  Object[] result 
[REPLACE]^result[4] = getDateFormat (  ) .format ( value ) ;^121^^^^^112^138^[REPLACE] result[2] = getDateFormat (  ) .format ( value ) ;^[METHOD] createItemArray [TYPE] Object[] [PARAMETER] CategoryDataset dataset int row int column [CLASS] IntervalCategoryToolTipGenerator   [TYPE]  IntervalCategoryDataset icd  [TYPE]  boolean false  true  [TYPE]  Number end  start  value  [TYPE]  CategoryDataset dataset  [TYPE]  String DEFAULT_TOOL_TIP_FORMAT_STRING  labelFormat  [TYPE]  long serialVersionUID  [TYPE]  int column  row  [TYPE]  Object[] result 
[ADD]^^121^122^^^^112^138^[ADD] result[2] = getDateFormat (  ) .format ( value ) ; }^[METHOD] createItemArray [TYPE] Object[] [PARAMETER] CategoryDataset dataset int row int column [CLASS] IntervalCategoryToolTipGenerator   [TYPE]  IntervalCategoryDataset icd  [TYPE]  boolean false  true  [TYPE]  Number end  start  value  [TYPE]  CategoryDataset dataset  [TYPE]  String DEFAULT_TOOL_TIP_FORMAT_STRING  labelFormat  [TYPE]  long serialVersionUID  [TYPE]  int column  row  [TYPE]  Object[] result 
[REPLACE]^result[3] = getDateFormat (  ) .format ( start ) ; ;^118^^^^^112^138^[REPLACE] result[2] = getNumberFormat (  ) .format ( value ) ;^[METHOD] createItemArray [TYPE] Object[] [PARAMETER] CategoryDataset dataset int row int column [CLASS] IntervalCategoryToolTipGenerator   [TYPE]  IntervalCategoryDataset icd  [TYPE]  boolean false  true  [TYPE]  Number end  start  value  [TYPE]  CategoryDataset dataset  [TYPE]  String DEFAULT_TOOL_TIP_FORMAT_STRING  labelFormat  [TYPE]  long serialVersionUID  [TYPE]  int column  row  [TYPE]  Object[] result 
[ADD]^^118^119^^^^112^138^[ADD] result[2] = getNumberFormat (  ) .format ( value ) ; }^[METHOD] createItemArray [TYPE] Object[] [PARAMETER] CategoryDataset dataset int row int column [CLASS] IntervalCategoryToolTipGenerator   [TYPE]  IntervalCategoryDataset icd  [TYPE]  boolean false  true  [TYPE]  Number end  start  value  [TYPE]  CategoryDataset dataset  [TYPE]  String DEFAULT_TOOL_TIP_FORMAT_STRING  labelFormat  [TYPE]  long serialVersionUID  [TYPE]  int column  row  [TYPE]  Object[] result 
[REPLACE]^result[1] = getDateFormat (  ) .format ( value ) ;^121^^^^^112^138^[REPLACE] result[2] = getDateFormat (  ) .format ( value ) ;^[METHOD] createItemArray [TYPE] Object[] [PARAMETER] CategoryDataset dataset int row int column [CLASS] IntervalCategoryToolTipGenerator   [TYPE]  IntervalCategoryDataset icd  [TYPE]  boolean false  true  [TYPE]  Number end  start  value  [TYPE]  CategoryDataset dataset  [TYPE]  String DEFAULT_TOOL_TIP_FORMAT_STRING  labelFormat  [TYPE]  long serialVersionUID  [TYPE]  int column  row  [TYPE]  Object[] result 
[REPLACE]^result[2 << 0] = getDateFormat (  ) .format ( value ) ;^121^^^^^112^138^[REPLACE] result[2] = getDateFormat (  ) .format ( value ) ;^[METHOD] createItemArray [TYPE] Object[] [PARAMETER] CategoryDataset dataset int row int column [CLASS] IntervalCategoryToolTipGenerator   [TYPE]  IntervalCategoryDataset icd  [TYPE]  boolean false  true  [TYPE]  Number end  start  value  [TYPE]  CategoryDataset dataset  [TYPE]  String DEFAULT_TOOL_TIP_FORMAT_STRING  labelFormat  [TYPE]  long serialVersionUID  [TYPE]  int column  row  [TYPE]  Object[] result 
[REPLACE]^if  ( ! dataset instanceof IntervalCategoryDataset )  {^124^^^^^112^138^[REPLACE] if  ( dataset instanceof IntervalCategoryDataset )  {^[METHOD] createItemArray [TYPE] Object[] [PARAMETER] CategoryDataset dataset int row int column [CLASS] IntervalCategoryToolTipGenerator   [TYPE]  IntervalCategoryDataset icd  [TYPE]  boolean false  true  [TYPE]  Number end  start  value  [TYPE]  CategoryDataset dataset  [TYPE]  String DEFAULT_TOOL_TIP_FORMAT_STRING  labelFormat  [TYPE]  long serialVersionUID  [TYPE]  int column  row  [TYPE]  Object[] result 
[REPLACE]^if  ( getNumberFormat (  )  == true )  {^128^^^^^112^138^[REPLACE] if  ( getNumberFormat (  )  != null )  {^[METHOD] createItemArray [TYPE] Object[] [PARAMETER] CategoryDataset dataset int row int column [CLASS] IntervalCategoryToolTipGenerator   [TYPE]  IntervalCategoryDataset icd  [TYPE]  boolean false  true  [TYPE]  Number end  start  value  [TYPE]  CategoryDataset dataset  [TYPE]  String DEFAULT_TOOL_TIP_FORMAT_STRING  labelFormat  [TYPE]  long serialVersionUID  [TYPE]  int column  row  [TYPE]  Object[] result 
[ADD]^result[3] = getNumberFormat (  ) .format ( start ) ;^128^129^130^131^^112^138^[ADD] if  ( getNumberFormat (  )  != null )  { result[3] = getNumberFormat (  ) .format ( start ) ; result[4] = getNumberFormat (  ) .format ( end ) ; }^[METHOD] createItemArray [TYPE] Object[] [PARAMETER] CategoryDataset dataset int row int column [CLASS] IntervalCategoryToolTipGenerator   [TYPE]  IntervalCategoryDataset icd  [TYPE]  boolean false  true  [TYPE]  Number end  start  value  [TYPE]  CategoryDataset dataset  [TYPE]  String DEFAULT_TOOL_TIP_FORMAT_STRING  labelFormat  [TYPE]  long serialVersionUID  [TYPE]  int column  row  [TYPE]  Object[] result 
[REPLACE]^else {^132^^^^^112^138^[REPLACE] else if  ( getDateFormat (  )  != null )  {^[METHOD] createItemArray [TYPE] Object[] [PARAMETER] CategoryDataset dataset int row int column [CLASS] IntervalCategoryToolTipGenerator   [TYPE]  IntervalCategoryDataset icd  [TYPE]  boolean false  true  [TYPE]  Number end  start  value  [TYPE]  CategoryDataset dataset  [TYPE]  String DEFAULT_TOOL_TIP_FORMAT_STRING  labelFormat  [TYPE]  long serialVersionUID  [TYPE]  int column  row  [TYPE]  Object[] result 
[REPLACE]^result[2] = getNumberFormat (  ) .format ( value ) ; ;^133^^^^^112^138^[REPLACE] result[3] = getDateFormat (  ) .format ( start ) ;^[METHOD] createItemArray [TYPE] Object[] [PARAMETER] CategoryDataset dataset int row int column [CLASS] IntervalCategoryToolTipGenerator   [TYPE]  IntervalCategoryDataset icd  [TYPE]  boolean false  true  [TYPE]  Number end  start  value  [TYPE]  CategoryDataset dataset  [TYPE]  String DEFAULT_TOOL_TIP_FORMAT_STRING  labelFormat  [TYPE]  long serialVersionUID  [TYPE]  int column  row  [TYPE]  Object[] result 
[REPLACE]^result[2] = getDateFormat (  ) .format ( end ) ;^134^^^^^112^138^[REPLACE] result[4] = getDateFormat (  ) .format ( end ) ;^[METHOD] createItemArray [TYPE] Object[] [PARAMETER] CategoryDataset dataset int row int column [CLASS] IntervalCategoryToolTipGenerator   [TYPE]  IntervalCategoryDataset icd  [TYPE]  boolean false  true  [TYPE]  Number end  start  value  [TYPE]  CategoryDataset dataset  [TYPE]  String DEFAULT_TOOL_TIP_FORMAT_STRING  labelFormat  [TYPE]  long serialVersionUID  [TYPE]  int column  row  [TYPE]  Object[] result 
[REPLACE]^result[5] = getDateFormat (  ) .format ( start ) ;^133^^^^^112^138^[REPLACE] result[3] = getDateFormat (  ) .format ( start ) ;^[METHOD] createItemArray [TYPE] Object[] [PARAMETER] CategoryDataset dataset int row int column [CLASS] IntervalCategoryToolTipGenerator   [TYPE]  IntervalCategoryDataset icd  [TYPE]  boolean false  true  [TYPE]  Number end  start  value  [TYPE]  CategoryDataset dataset  [TYPE]  String DEFAULT_TOOL_TIP_FORMAT_STRING  labelFormat  [TYPE]  long serialVersionUID  [TYPE]  int column  row  [TYPE]  Object[] result 
[REPLACE]^result[0] = getDateFormat (  ) .format ( end ) ;^134^^^^^112^138^[REPLACE] result[4] = getDateFormat (  ) .format ( end ) ;^[METHOD] createItemArray [TYPE] Object[] [PARAMETER] CategoryDataset dataset int row int column [CLASS] IntervalCategoryToolTipGenerator   [TYPE]  IntervalCategoryDataset icd  [TYPE]  boolean false  true  [TYPE]  Number end  start  value  [TYPE]  CategoryDataset dataset  [TYPE]  String DEFAULT_TOOL_TIP_FORMAT_STRING  labelFormat  [TYPE]  long serialVersionUID  [TYPE]  int column  row  [TYPE]  Object[] result 
[REPLACE]^result[3] = getDateFormat (  ) .format ( start ) ; ;^129^^^^^112^138^[REPLACE] result[3] = getNumberFormat (  ) .format ( start ) ;^[METHOD] createItemArray [TYPE] Object[] [PARAMETER] CategoryDataset dataset int row int column [CLASS] IntervalCategoryToolTipGenerator   [TYPE]  IntervalCategoryDataset icd  [TYPE]  boolean false  true  [TYPE]  Number end  start  value  [TYPE]  CategoryDataset dataset  [TYPE]  String DEFAULT_TOOL_TIP_FORMAT_STRING  labelFormat  [TYPE]  long serialVersionUID  [TYPE]  int column  row  [TYPE]  Object[] result 
[ADD]^^129^130^131^^^112^138^[ADD] result[3] = getNumberFormat (  ) .format ( start ) ; result[4] = getNumberFormat (  ) .format ( end ) ; }^[METHOD] createItemArray [TYPE] Object[] [PARAMETER] CategoryDataset dataset int row int column [CLASS] IntervalCategoryToolTipGenerator   [TYPE]  IntervalCategoryDataset icd  [TYPE]  boolean false  true  [TYPE]  Number end  start  value  [TYPE]  CategoryDataset dataset  [TYPE]  String DEFAULT_TOOL_TIP_FORMAT_STRING  labelFormat  [TYPE]  long serialVersionUID  [TYPE]  int column  row  [TYPE]  Object[] result 
[REPLACE]^result[4 * 4] = getNumberFormat (  ) .format ( end ) ;^130^^^^^112^138^[REPLACE] result[4] = getNumberFormat (  ) .format ( end ) ;^[METHOD] createItemArray [TYPE] Object[] [PARAMETER] CategoryDataset dataset int row int column [CLASS] IntervalCategoryToolTipGenerator   [TYPE]  IntervalCategoryDataset icd  [TYPE]  boolean false  true  [TYPE]  Number end  start  value  [TYPE]  CategoryDataset dataset  [TYPE]  String DEFAULT_TOOL_TIP_FORMAT_STRING  labelFormat  [TYPE]  long serialVersionUID  [TYPE]  int column  row  [TYPE]  Object[] result 
[REPLACE]^if  ( getDateFormat (  )  == this )  {^132^^^^^112^138^[REPLACE] else if  ( getDateFormat (  )  != null )  {^[METHOD] createItemArray [TYPE] Object[] [PARAMETER] CategoryDataset dataset int row int column [CLASS] IntervalCategoryToolTipGenerator   [TYPE]  IntervalCategoryDataset icd  [TYPE]  boolean false  true  [TYPE]  Number end  start  value  [TYPE]  CategoryDataset dataset  [TYPE]  String DEFAULT_TOOL_TIP_FORMAT_STRING  labelFormat  [TYPE]  long serialVersionUID  [TYPE]  int column  row  [TYPE]  Object[] result 
[ADD]^^133^134^135^^^112^138^[ADD] result[3] = getDateFormat (  ) .format ( start ) ; result[4] = getDateFormat (  ) .format ( end ) ; }^[METHOD] createItemArray [TYPE] Object[] [PARAMETER] CategoryDataset dataset int row int column [CLASS] IntervalCategoryToolTipGenerator   [TYPE]  IntervalCategoryDataset icd  [TYPE]  boolean false  true  [TYPE]  Number end  start  value  [TYPE]  CategoryDataset dataset  [TYPE]  String DEFAULT_TOOL_TIP_FORMAT_STRING  labelFormat  [TYPE]  long serialVersionUID  [TYPE]  int column  row  [TYPE]  Object[] result 
[REPLACE]^result[3] = getDateFormat (  ) .format ( start ) ; ;^134^^^^^112^138^[REPLACE] result[4] = getDateFormat (  ) .format ( end ) ;^[METHOD] createItemArray [TYPE] Object[] [PARAMETER] CategoryDataset dataset int row int column [CLASS] IntervalCategoryToolTipGenerator   [TYPE]  IntervalCategoryDataset icd  [TYPE]  boolean false  true  [TYPE]  Number end  start  value  [TYPE]  CategoryDataset dataset  [TYPE]  String DEFAULT_TOOL_TIP_FORMAT_STRING  labelFormat  [TYPE]  long serialVersionUID  [TYPE]  int column  row  [TYPE]  Object[] result 
[REPLACE]^result[1] = getNumberFormat (  ) .format ( start ) ;^129^^^^^112^138^[REPLACE] result[3] = getNumberFormat (  ) .format ( start ) ;^[METHOD] createItemArray [TYPE] Object[] [PARAMETER] CategoryDataset dataset int row int column [CLASS] IntervalCategoryToolTipGenerator   [TYPE]  IntervalCategoryDataset icd  [TYPE]  boolean false  true  [TYPE]  Number end  start  value  [TYPE]  CategoryDataset dataset  [TYPE]  String DEFAULT_TOOL_TIP_FORMAT_STRING  labelFormat  [TYPE]  long serialVersionUID  [TYPE]  int column  row  [TYPE]  Object[] result 
[REPLACE]^result[1] = getNumberFormat (  ) .format ( end ) ;^130^^^^^112^138^[REPLACE] result[4] = getNumberFormat (  ) .format ( end ) ;^[METHOD] createItemArray [TYPE] Object[] [PARAMETER] CategoryDataset dataset int row int column [CLASS] IntervalCategoryToolTipGenerator   [TYPE]  IntervalCategoryDataset icd  [TYPE]  boolean false  true  [TYPE]  Number end  start  value  [TYPE]  CategoryDataset dataset  [TYPE]  String DEFAULT_TOOL_TIP_FORMAT_STRING  labelFormat  [TYPE]  long serialVersionUID  [TYPE]  int column  row  [TYPE]  Object[] result 
[ADD]^^130^131^^^^112^138^[ADD] result[4] = getNumberFormat (  ) .format ( end ) ; }^[METHOD] createItemArray [TYPE] Object[] [PARAMETER] CategoryDataset dataset int row int column [CLASS] IntervalCategoryToolTipGenerator   [TYPE]  IntervalCategoryDataset icd  [TYPE]  boolean false  true  [TYPE]  Number end  start  value  [TYPE]  CategoryDataset dataset  [TYPE]  String DEFAULT_TOOL_TIP_FORMAT_STRING  labelFormat  [TYPE]  long serialVersionUID  [TYPE]  int column  row  [TYPE]  Object[] result 
[REPLACE]^result[3L] = getDateFormat (  ) .format ( start ) ;^133^^^^^112^138^[REPLACE] result[3] = getDateFormat (  ) .format ( start ) ;^[METHOD] createItemArray [TYPE] Object[] [PARAMETER] CategoryDataset dataset int row int column [CLASS] IntervalCategoryToolTipGenerator   [TYPE]  IntervalCategoryDataset icd  [TYPE]  boolean false  true  [TYPE]  Number end  start  value  [TYPE]  CategoryDataset dataset  [TYPE]  String DEFAULT_TOOL_TIP_FORMAT_STRING  labelFormat  [TYPE]  long serialVersionUID  [TYPE]  int column  row  [TYPE]  Object[] result 
[REPLACE]^Number value = dataset.getValue ( row, column ) ;^125^^^^^112^138^[REPLACE] IntervalCategoryDataset icd =  ( IntervalCategoryDataset )  dataset;^[METHOD] createItemArray [TYPE] Object[] [PARAMETER] CategoryDataset dataset int row int column [CLASS] IntervalCategoryToolTipGenerator   [TYPE]  IntervalCategoryDataset icd  [TYPE]  boolean false  true  [TYPE]  Number end  start  value  [TYPE]  CategoryDataset dataset  [TYPE]  String DEFAULT_TOOL_TIP_FORMAT_STRING  labelFormat  [TYPE]  long serialVersionUID  [TYPE]  int column  row  [TYPE]  Object[] result 
[REPLACE]^Number start = icd .getEndValue ( row , column )  ;^126^^^^^112^138^[REPLACE] Number start = icd.getStartValue ( row, column ) ;^[METHOD] createItemArray [TYPE] Object[] [PARAMETER] CategoryDataset dataset int row int column [CLASS] IntervalCategoryToolTipGenerator   [TYPE]  IntervalCategoryDataset icd  [TYPE]  boolean false  true  [TYPE]  Number end  start  value  [TYPE]  CategoryDataset dataset  [TYPE]  String DEFAULT_TOOL_TIP_FORMAT_STRING  labelFormat  [TYPE]  long serialVersionUID  [TYPE]  int column  row  [TYPE]  Object[] result 
[REPLACE]^Number end = icd .getStartValue ( column , row )  ;^127^^^^^112^138^[REPLACE] Number end = icd.getEndValue ( row, column ) ;^[METHOD] createItemArray [TYPE] Object[] [PARAMETER] CategoryDataset dataset int row int column [CLASS] IntervalCategoryToolTipGenerator   [TYPE]  IntervalCategoryDataset icd  [TYPE]  boolean false  true  [TYPE]  Number end  start  value  [TYPE]  CategoryDataset dataset  [TYPE]  String DEFAULT_TOOL_TIP_FORMAT_STRING  labelFormat  [TYPE]  long serialVersionUID  [TYPE]  int column  row  [TYPE]  Object[] result 
[ADD]^^127^^^^^112^138^[ADD] Number end = icd.getEndValue ( row, column ) ;^[METHOD] createItemArray [TYPE] Object[] [PARAMETER] CategoryDataset dataset int row int column [CLASS] IntervalCategoryToolTipGenerator   [TYPE]  IntervalCategoryDataset icd  [TYPE]  boolean false  true  [TYPE]  Number end  start  value  [TYPE]  CategoryDataset dataset  [TYPE]  String DEFAULT_TOOL_TIP_FORMAT_STRING  labelFormat  [TYPE]  long serialVersionUID  [TYPE]  int column  row  [TYPE]  Object[] result 
[REPLACE]^if  ( getNumberFormat (  )  == null )  {^128^^^^^112^138^[REPLACE] if  ( getNumberFormat (  )  != null )  {^[METHOD] createItemArray [TYPE] Object[] [PARAMETER] CategoryDataset dataset int row int column [CLASS] IntervalCategoryToolTipGenerator   [TYPE]  IntervalCategoryDataset icd  [TYPE]  boolean false  true  [TYPE]  Number end  start  value  [TYPE]  CategoryDataset dataset  [TYPE]  String DEFAULT_TOOL_TIP_FORMAT_STRING  labelFormat  [TYPE]  long serialVersionUID  [TYPE]  int column  row  [TYPE]  Object[] result 
[REPLACE]^else if  ( getDateFormat (  )  == null )  {^132^^^^^112^138^[REPLACE] else if  ( getDateFormat (  )  != null )  {^[METHOD] createItemArray [TYPE] Object[] [PARAMETER] CategoryDataset dataset int row int column [CLASS] IntervalCategoryToolTipGenerator   [TYPE]  IntervalCategoryDataset icd  [TYPE]  boolean false  true  [TYPE]  Number end  start  value  [TYPE]  CategoryDataset dataset  [TYPE]  String DEFAULT_TOOL_TIP_FORMAT_STRING  labelFormat  [TYPE]  long serialVersionUID  [TYPE]  int column  row  [TYPE]  Object[] result 
[REPLACE]^result[4 >>> 0] = getDateFormat (  ) .format ( end ) ;^134^^^^^112^138^[REPLACE] result[4] = getDateFormat (  ) .format ( end ) ;^[METHOD] createItemArray [TYPE] Object[] [PARAMETER] CategoryDataset dataset int row int column [CLASS] IntervalCategoryToolTipGenerator   [TYPE]  IntervalCategoryDataset icd  [TYPE]  boolean false  true  [TYPE]  Number end  start  value  [TYPE]  CategoryDataset dataset  [TYPE]  String DEFAULT_TOOL_TIP_FORMAT_STRING  labelFormat  [TYPE]  long serialVersionUID  [TYPE]  int column  row  [TYPE]  Object[] result 
[REPLACE]^result[3 + 2] = getDateFormat (  ) .format ( start ) ;^133^^^^^112^138^[REPLACE] result[3] = getDateFormat (  ) .format ( start ) ;^[METHOD] createItemArray [TYPE] Object[] [PARAMETER] CategoryDataset dataset int row int column [CLASS] IntervalCategoryToolTipGenerator   [TYPE]  IntervalCategoryDataset icd  [TYPE]  boolean false  true  [TYPE]  Number end  start  value  [TYPE]  CategoryDataset dataset  [TYPE]  String DEFAULT_TOOL_TIP_FORMAT_STRING  labelFormat  [TYPE]  long serialVersionUID  [TYPE]  int column  row  [TYPE]  Object[] result 
[REPLACE]^result[3] = getDateFormat (  ) .format ( start ) ; ;^130^^^^^112^138^[REPLACE] result[4] = getNumberFormat (  ) .format ( end ) ;^[METHOD] createItemArray [TYPE] Object[] [PARAMETER] CategoryDataset dataset int row int column [CLASS] IntervalCategoryToolTipGenerator   [TYPE]  IntervalCategoryDataset icd  [TYPE]  boolean false  true  [TYPE]  Number end  start  value  [TYPE]  CategoryDataset dataset  [TYPE]  String DEFAULT_TOOL_TIP_FORMAT_STRING  labelFormat  [TYPE]  long serialVersionUID  [TYPE]  int column  row  [TYPE]  Object[] result 
[REPLACE]^result[2] = getNumberFormat (  ) .format ( end ) ;^130^^^^^112^138^[REPLACE] result[4] = getNumberFormat (  ) .format ( end ) ;^[METHOD] createItemArray [TYPE] Object[] [PARAMETER] CategoryDataset dataset int row int column [CLASS] IntervalCategoryToolTipGenerator   [TYPE]  IntervalCategoryDataset icd  [TYPE]  boolean false  true  [TYPE]  Number end  start  value  [TYPE]  CategoryDataset dataset  [TYPE]  String DEFAULT_TOOL_TIP_FORMAT_STRING  labelFormat  [TYPE]  long serialVersionUID  [TYPE]  int column  row  [TYPE]  Object[] result 
[REPLACE]^result[2] = getDateFormat (  ) .format ( start ) ;^133^^^^^112^138^[REPLACE] result[3] = getDateFormat (  ) .format ( start ) ;^[METHOD] createItemArray [TYPE] Object[] [PARAMETER] CategoryDataset dataset int row int column [CLASS] IntervalCategoryToolTipGenerator   [TYPE]  IntervalCategoryDataset icd  [TYPE]  boolean false  true  [TYPE]  Number end  start  value  [TYPE]  CategoryDataset dataset  [TYPE]  String DEFAULT_TOOL_TIP_FORMAT_STRING  labelFormat  [TYPE]  long serialVersionUID  [TYPE]  int column  row  [TYPE]  Object[] result 
[REPLACE]^result[4L] = getDateFormat (  ) .format ( end ) ;^134^^^^^112^138^[REPLACE] result[4] = getDateFormat (  ) .format ( end ) ;^[METHOD] createItemArray [TYPE] Object[] [PARAMETER] CategoryDataset dataset int row int column [CLASS] IntervalCategoryToolTipGenerator   [TYPE]  IntervalCategoryDataset icd  [TYPE]  boolean false  true  [TYPE]  Number end  start  value  [TYPE]  CategoryDataset dataset  [TYPE]  String DEFAULT_TOOL_TIP_FORMAT_STRING  labelFormat  [TYPE]  long serialVersionUID  [TYPE]  int column  row  [TYPE]  Object[] result 
[ADD]^^125^126^^^^112^138^[ADD] IntervalCategoryDataset icd =  ( IntervalCategoryDataset )  dataset; Number start = icd.getStartValue ( row, column ) ;^[METHOD] createItemArray [TYPE] Object[] [PARAMETER] CategoryDataset dataset int row int column [CLASS] IntervalCategoryToolTipGenerator   [TYPE]  IntervalCategoryDataset icd  [TYPE]  boolean false  true  [TYPE]  Number end  start  value  [TYPE]  CategoryDataset dataset  [TYPE]  String DEFAULT_TOOL_TIP_FORMAT_STRING  labelFormat  [TYPE]  long serialVersionUID  [TYPE]  int column  row  [TYPE]  Object[] result 
[REPLACE]^Number start = icd.getEndValue ( row, column ) ;^126^^^^^112^138^[REPLACE] Number start = icd.getStartValue ( row, column ) ;^[METHOD] createItemArray [TYPE] Object[] [PARAMETER] CategoryDataset dataset int row int column [CLASS] IntervalCategoryToolTipGenerator   [TYPE]  IntervalCategoryDataset icd  [TYPE]  boolean false  true  [TYPE]  Number end  start  value  [TYPE]  CategoryDataset dataset  [TYPE]  String DEFAULT_TOOL_TIP_FORMAT_STRING  labelFormat  [TYPE]  long serialVersionUID  [TYPE]  int column  row  [TYPE]  Object[] result 
[REPLACE]^Number end = icd.getStartValue ( row, column ) ;^127^^^^^112^138^[REPLACE] Number end = icd.getEndValue ( row, column ) ;^[METHOD] createItemArray [TYPE] Object[] [PARAMETER] CategoryDataset dataset int row int column [CLASS] IntervalCategoryToolTipGenerator   [TYPE]  IntervalCategoryDataset icd  [TYPE]  boolean false  true  [TYPE]  Number end  start  value  [TYPE]  CategoryDataset dataset  [TYPE]  String DEFAULT_TOOL_TIP_FORMAT_STRING  labelFormat  [TYPE]  long serialVersionUID  [TYPE]  int column  row  [TYPE]  Object[] result 