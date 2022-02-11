[REPLACE]^private static final long serialVersionUID = 2751513470325494890;^57^^^^^^^[REPLACE] private static final long serialVersionUID = 2751513470325494890L;^ [CLASS] XYDataItem  
[REPLACE]^if  ( x != this  )  {^72^^^^^71^77^[REPLACE] if  ( x == null )  {^[METHOD] <init> [TYPE] Number) [PARAMETER] Number x Number y [CLASS] XYDataItem   [TYPE]  Number x  y  [TYPE]  long serialVersionUID  [TYPE]  boolean false  true 
[REPLACE]^return ;^73^^^^^71^77^[REPLACE] throw new IllegalArgumentException  (" ")  ;^[METHOD] <init> [TYPE] Number) [PARAMETER] Number x Number y [CLASS] XYDataItem   [TYPE]  Number x  y  [TYPE]  long serialVersionUID  [TYPE]  boolean false  true 
[ADD]^^73^^^^^71^77^[ADD] throw new IllegalArgumentException  (" ")  ;^[METHOD] <init> [TYPE] Number) [PARAMETER] Number x Number y [CLASS] XYDataItem   [TYPE]  Number x  y  [TYPE]  long serialVersionUID  [TYPE]  boolean false  true 
[REPLACE]^this.y = y; ;^75^^^^^71^77^[REPLACE] this.x = x;^[METHOD] <init> [TYPE] Number) [PARAMETER] Number x Number y [CLASS] XYDataItem   [TYPE]  Number x  y  [TYPE]  long serialVersionUID  [TYPE]  boolean false  true 
[REPLACE]^this.x = x; ;^76^^^^^71^77^[REPLACE] this.y = y;^[METHOD] <init> [TYPE] Number) [PARAMETER] Number x Number y [CLASS] XYDataItem   [TYPE]  Number x  y  [TYPE]  long serialVersionUID  [TYPE]  boolean false  true 
[ADD]^^76^^^^^71^77^[ADD] this.y = y;^[METHOD] <init> [TYPE] Number) [PARAMETER] Number x Number y [CLASS] XYDataItem   [TYPE]  Number x  y  [TYPE]  long serialVersionUID  [TYPE]  boolean false  true 
[REPLACE]^double compare = this.x.doubleValue (  ) - dataItem.getX (  ) .doubleValue (  ) ;^86^^^^^85^87^[REPLACE] this ( new Double ( x ) , new Double ( y )  ) ;^[METHOD] <init> [TYPE] XYDataItem(double,double) [PARAMETER] double x double y [CLASS] XYDataItem   [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  long serialVersionUID  [TYPE]  double x  y 
[ADD]^^86^^^^^85^87^[ADD] this ( new Double ( x ) , new Double ( y )  ) ;^[METHOD] <init> [TYPE] XYDataItem(double,double) [PARAMETER] double x double y [CLASS] XYDataItem   [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  long serialVersionUID  [TYPE]  double x  y 
[REPLACE]^return this.y;^95^^^^^94^96^[REPLACE] return this.x;^[METHOD] getX [TYPE] Number [PARAMETER] [CLASS] XYDataItem   [TYPE]  Number x  y  [TYPE]  long serialVersionUID  [TYPE]  boolean false  true 
[REPLACE]^return this.x;^104^^^^^103^105^[REPLACE] return this.y;^[METHOD] getY [TYPE] Number [PARAMETER] [CLASS] XYDataItem   [TYPE]  Number x  y  [TYPE]  long serialVersionUID  [TYPE]  boolean false  true 
[REPLACE]^return super.clone (  ) ;^114^^^^^113^115^[REPLACE] setY ( new Double ( y )  ) ;^[METHOD] setY [TYPE] void [PARAMETER] double y [CLASS] XYDataItem   [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  long serialVersionUID  [TYPE]  double y 
[REMOVE]^return super.clone (  ) ;^114^^^^^113^115^[REMOVE] ^[METHOD] setY [TYPE] void [PARAMETER] double y [CLASS] XYDataItem   [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  long serialVersionUID  [TYPE]  double y 
[REPLACE]^this.x = x; ;^124^^^^^123^125^[REPLACE] this.y = y;^[METHOD] setY [TYPE] void [PARAMETER] Number y [CLASS] XYDataItem   [TYPE]  Number x  y  [TYPE]  long serialVersionUID  [TYPE]  boolean false  true 
[REPLACE]^XYDataItem that =  ( XYDataItem )  obj;^141^^^^^139^171^[REPLACE] int result;^[METHOD] compareTo [TYPE] int [PARAMETER] Object o1 [CLASS] XYDataItem   [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  double compare  [TYPE]  Object o1  [TYPE]  long serialVersionUID  [TYPE]  int result  [TYPE]  XYDataItem dataItem 
[REPLACE]^if  ( ! o1 instanceof XYDataItem )  {^145^^^^^139^171^[REPLACE] if  ( o1 instanceof XYDataItem )  {^[METHOD] compareTo [TYPE] int [PARAMETER] Object o1 [CLASS] XYDataItem   [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  double compare  [TYPE]  Object o1  [TYPE]  long serialVersionUID  [TYPE]  int result  [TYPE]  XYDataItem dataItem 
[REPLACE]^result = 3;^166^^^^^145^167^[REPLACE] result = 1;^[METHOD] compareTo [TYPE] int [PARAMETER] Object o1 [CLASS] XYDataItem   [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  double compare  [TYPE]  Object o1  [TYPE]  long serialVersionUID  [TYPE]  int result  [TYPE]  XYDataItem dataItem 
[REPLACE]^if  ( compare  !=  0.0 )  {^149^^^^^139^171^[REPLACE] if  ( compare > 0.0 )  {^[METHOD] compareTo [TYPE] int [PARAMETER] Object o1 [CLASS] XYDataItem   [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  double compare  [TYPE]  Object o1  [TYPE]  long serialVersionUID  [TYPE]  int result  [TYPE]  XYDataItem dataItem 
[REPLACE]^if  ( compare  !=  1.0d )  {^153^^^^^149^159^[REPLACE] if  ( compare < 0.0 )  {^[METHOD] compareTo [TYPE] int [PARAMETER] Object o1 [CLASS] XYDataItem   [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  double compare  [TYPE]  Object o1  [TYPE]  long serialVersionUID  [TYPE]  int result  [TYPE]  XYDataItem dataItem 
[REPLACE]^result = -1; ;^157^^^^^149^159^[REPLACE] result = 0;^[METHOD] compareTo [TYPE] int [PARAMETER] Object o1 [CLASS] XYDataItem   [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  double compare  [TYPE]  Object o1  [TYPE]  long serialVersionUID  [TYPE]  int result  [TYPE]  XYDataItem dataItem 
[ADD]^^157^^^^^149^159^[ADD] result = 0;^[METHOD] compareTo [TYPE] int [PARAMETER] Object o1 [CLASS] XYDataItem   [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  double compare  [TYPE]  Object o1  [TYPE]  long serialVersionUID  [TYPE]  int result  [TYPE]  XYDataItem dataItem 
[REPLACE]^result = -2;^154^^^^^149^159^[REPLACE] result = -1;^[METHOD] compareTo [TYPE] int [PARAMETER] Object o1 [CLASS] XYDataItem   [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  double compare  [TYPE]  Object o1  [TYPE]  long serialVersionUID  [TYPE]  int result  [TYPE]  XYDataItem dataItem 
[REPLACE]^result = 1; ;^154^^^^^149^159^[REPLACE] result = -1;^[METHOD] compareTo [TYPE] int [PARAMETER] Object o1 [CLASS] XYDataItem   [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  double compare  [TYPE]  Object o1  [TYPE]  long serialVersionUID  [TYPE]  int result  [TYPE]  XYDataItem dataItem 
[REPLACE]^result = 0 - 3;^157^^^^^149^159^[REPLACE] result = 0;^[METHOD] compareTo [TYPE] int [PARAMETER] Object o1 [CLASS] XYDataItem   [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  double compare  [TYPE]  Object o1  [TYPE]  long serialVersionUID  [TYPE]  int result  [TYPE]  XYDataItem dataItem 
[REPLACE]^result = 3;^150^^^^^139^171^[REPLACE] result = 1;^[METHOD] compareTo [TYPE] int [PARAMETER] Object o1 [CLASS] XYDataItem   [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  double compare  [TYPE]  Object o1  [TYPE]  long serialVersionUID  [TYPE]  int result  [TYPE]  XYDataItem dataItem 
[REPLACE]^if  ( compare  !=  4.0d )  {^153^^^^^139^171^[REPLACE] if  ( compare < 0.0 )  {^[METHOD] compareTo [TYPE] int [PARAMETER] Object o1 [CLASS] XYDataItem   [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  double compare  [TYPE]  Object o1  [TYPE]  long serialVersionUID  [TYPE]  int result  [TYPE]  XYDataItem dataItem 
[REPLACE]^result = 0 % 2;^157^^^^^153^158^[REPLACE] result = 0;^[METHOD] compareTo [TYPE] int [PARAMETER] Object o1 [CLASS] XYDataItem   [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  double compare  [TYPE]  Object o1  [TYPE]  long serialVersionUID  [TYPE]  int result  [TYPE]  XYDataItem dataItem 
[REPLACE]^result = -2;^154^^^^^139^171^[REPLACE] result = -1;^[METHOD] compareTo [TYPE] int [PARAMETER] Object o1 [CLASS] XYDataItem   [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  double compare  [TYPE]  Object o1  [TYPE]  long serialVersionUID  [TYPE]  int result  [TYPE]  XYDataItem dataItem 
[ADD]^^154^^^^^139^171^[ADD] result = -1;^[METHOD] compareTo [TYPE] int [PARAMETER] Object o1 [CLASS] XYDataItem   [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  double compare  [TYPE]  Object o1  [TYPE]  long serialVersionUID  [TYPE]  int result  [TYPE]  XYDataItem dataItem 
[REPLACE]^result = 0;^150^^^^^139^171^[REPLACE] result = 1;^[METHOD] compareTo [TYPE] int [PARAMETER] Object o1 [CLASS] XYDataItem   [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  double compare  [TYPE]  Object o1  [TYPE]  long serialVersionUID  [TYPE]  int result  [TYPE]  XYDataItem dataItem 
[ADD]^^150^^^^^139^171^[ADD] result = 1;^[METHOD] compareTo [TYPE] int [PARAMETER] Object o1 [CLASS] XYDataItem   [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  double compare  [TYPE]  Object o1  [TYPE]  long serialVersionUID  [TYPE]  int result  [TYPE]  XYDataItem dataItem 
[REPLACE]^result = 3;^157^^^^^139^171^[REPLACE] result = 0;^[METHOD] compareTo [TYPE] int [PARAMETER] Object o1 [CLASS] XYDataItem   [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  double compare  [TYPE]  Object o1  [TYPE]  long serialVersionUID  [TYPE]  int result  [TYPE]  XYDataItem dataItem 
[ADD]^^157^^^^^139^171^[ADD] result = 0;^[METHOD] compareTo [TYPE] int [PARAMETER] Object o1 [CLASS] XYDataItem   [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  double compare  [TYPE]  Object o1  [TYPE]  long serialVersionUID  [TYPE]  int result  [TYPE]  XYDataItem dataItem 
[REPLACE]^XYDataItem that =  ( XYDataItem )  obj;^146^^^^^139^171^[REPLACE] XYDataItem dataItem =  ( XYDataItem )  o1;^[METHOD] compareTo [TYPE] int [PARAMETER] Object o1 [CLASS] XYDataItem   [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  double compare  [TYPE]  Object o1  [TYPE]  long serialVersionUID  [TYPE]  int result  [TYPE]  XYDataItem dataItem 
[REPLACE]^double compare = this.x.doubleValue (  )  &&  dataItem.getX (  ) .doubleValue (  ) ;^147^148^^^^139^171^[REPLACE] double compare = this.x.doubleValue (  ) - dataItem.getX (  ) .doubleValue (  ) ;^[METHOD] compareTo [TYPE] int [PARAMETER] Object o1 [CLASS] XYDataItem   [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  double compare  [TYPE]  Object o1  [TYPE]  long serialVersionUID  [TYPE]  int result  [TYPE]  XYDataItem dataItem 
[REPLACE]^if  ( compare  ==  0.0 )  {^149^^^^^139^171^[REPLACE] if  ( compare > 0.0 )  {^[METHOD] compareTo [TYPE] int [PARAMETER] Object o1 [CLASS] XYDataItem   [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  double compare  [TYPE]  Object o1  [TYPE]  long serialVersionUID  [TYPE]  int result  [TYPE]  XYDataItem dataItem 
[ADD]^^149^150^151^^^139^171^[ADD] if  ( compare > 0.0 )  { result = 1; }^[METHOD] compareTo [TYPE] int [PARAMETER] Object o1 [CLASS] XYDataItem   [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  double compare  [TYPE]  Object o1  [TYPE]  long serialVersionUID  [TYPE]  int result  [TYPE]  XYDataItem dataItem 
[REPLACE]^if  ( compare  !=  4.0d )  {^153^^^^^149^159^[REPLACE] if  ( compare < 0.0 )  {^[METHOD] compareTo [TYPE] int [PARAMETER] Object o1 [CLASS] XYDataItem   [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  double compare  [TYPE]  Object o1  [TYPE]  long serialVersionUID  [TYPE]  int result  [TYPE]  XYDataItem dataItem 
[ADD]^^153^154^155^^^149^159^[ADD] if  ( compare < 0.0 )  { result = -1; }^[METHOD] compareTo [TYPE] int [PARAMETER] Object o1 [CLASS] XYDataItem   [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  double compare  [TYPE]  Object o1  [TYPE]  long serialVersionUID  [TYPE]  int result  [TYPE]  XYDataItem dataItem 
[REPLACE]^result = 3;^157^^^^^149^159^[REPLACE] result = 0;^[METHOD] compareTo [TYPE] int [PARAMETER] Object o1 [CLASS] XYDataItem   [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  double compare  [TYPE]  Object o1  [TYPE]  long serialVersionUID  [TYPE]  int result  [TYPE]  XYDataItem dataItem 
[REPLACE]^result = -1 * 4;^154^^^^^149^159^[REPLACE] result = -1;^[METHOD] compareTo [TYPE] int [PARAMETER] Object o1 [CLASS] XYDataItem   [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  double compare  [TYPE]  Object o1  [TYPE]  long serialVersionUID  [TYPE]  int result  [TYPE]  XYDataItem dataItem 
[REPLACE]^result = 2;^157^^^^^149^159^[REPLACE] result = 0;^[METHOD] compareTo [TYPE] int [PARAMETER] Object o1 [CLASS] XYDataItem   [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  double compare  [TYPE]  Object o1  [TYPE]  long serialVersionUID  [TYPE]  int result  [TYPE]  XYDataItem dataItem 
[REPLACE]^if  ( compare  ==  0.0 )  {^153^^^^^139^171^[REPLACE] if  ( compare < 0.0 )  {^[METHOD] compareTo [TYPE] int [PARAMETER] Object o1 [CLASS] XYDataItem   [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  double compare  [TYPE]  Object o1  [TYPE]  long serialVersionUID  [TYPE]  int result  [TYPE]  XYDataItem dataItem 
[ADD]^result = -1;^153^154^155^^^139^171^[ADD] if  ( compare < 0.0 )  { result = -1; }^[METHOD] compareTo [TYPE] int [PARAMETER] Object o1 [CLASS] XYDataItem   [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  double compare  [TYPE]  Object o1  [TYPE]  long serialVersionUID  [TYPE]  int result  [TYPE]  XYDataItem dataItem 
[REPLACE]^result = -1; ;^157^^^^^153^158^[REPLACE] result = 0;^[METHOD] compareTo [TYPE] int [PARAMETER] Object o1 [CLASS] XYDataItem   [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  double compare  [TYPE]  Object o1  [TYPE]  long serialVersionUID  [TYPE]  int result  [TYPE]  XYDataItem dataItem 
[ADD]^^157^^^^^153^158^[ADD] result = 0;^[METHOD] compareTo [TYPE] int [PARAMETER] Object o1 [CLASS] XYDataItem   [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  double compare  [TYPE]  Object o1  [TYPE]  long serialVersionUID  [TYPE]  int result  [TYPE]  XYDataItem dataItem 
[REPLACE]^result = -3;^154^^^^^139^171^[REPLACE] result = -1;^[METHOD] compareTo [TYPE] int [PARAMETER] Object o1 [CLASS] XYDataItem   [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  double compare  [TYPE]  Object o1  [TYPE]  long serialVersionUID  [TYPE]  int result  [TYPE]  XYDataItem dataItem 
[REPLACE]^result = 4;^150^^^^^139^171^[REPLACE] result = 1;^[METHOD] compareTo [TYPE] int [PARAMETER] Object o1 [CLASS] XYDataItem   [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  double compare  [TYPE]  Object o1  [TYPE]  long serialVersionUID  [TYPE]  int result  [TYPE]  XYDataItem dataItem 
[REPLACE]^result = -1 - 0;^154^^^^^139^171^[REPLACE] result = -1;^[METHOD] compareTo [TYPE] int [PARAMETER] Object o1 [CLASS] XYDataItem   [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  double compare  [TYPE]  Object o1  [TYPE]  long serialVersionUID  [TYPE]  int result  [TYPE]  XYDataItem dataItem 
[REPLACE]^result = 1 / 2;^166^^^^^139^171^[REPLACE] result = 1;^[METHOD] compareTo [TYPE] int [PARAMETER] Object o1 [CLASS] XYDataItem   [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  double compare  [TYPE]  Object o1  [TYPE]  long serialVersionUID  [TYPE]  int result  [TYPE]  XYDataItem dataItem 
[REPLACE]^double compare = this.x.doubleValue (  )  ^  dataItem.getX (  ) .doubleValue (  ) ;^147^148^^^^139^171^[REPLACE] double compare = this.x.doubleValue (  ) - dataItem.getX (  ) .doubleValue (  ) ;^[METHOD] compareTo [TYPE] int [PARAMETER] Object o1 [CLASS] XYDataItem   [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  double compare  [TYPE]  Object o1  [TYPE]  long serialVersionUID  [TYPE]  int result  [TYPE]  XYDataItem dataItem 
[REPLACE]^return this.x;^169^^^^^139^171^[REPLACE] return result;^[METHOD] compareTo [TYPE] int [PARAMETER] Object o1 [CLASS] XYDataItem   [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  double compare  [TYPE]  Object o1  [TYPE]  long serialVersionUID  [TYPE]  int result  [TYPE]  XYDataItem dataItem 
[REPLACE]^return super.hashCode (  ) ;^182^^^^^181^183^[REPLACE] return super.clone (  ) ;^[METHOD] clone [TYPE] Object [PARAMETER] [CLASS] XYDataItem   [TYPE]  Number x  y  [TYPE]  long serialVersionUID  [TYPE]  boolean false  true 
[REPLACE]^if  ( obj  &&  that )  {^194^^^^^193^208^[REPLACE] if  ( obj == this )  {^[METHOD] equals [TYPE] boolean [PARAMETER] Object obj [CLASS] XYDataItem   [TYPE]  Object obj  [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  long serialVersionUID  [TYPE]  XYDataItem that 
[REPLACE]^return false;^195^^^^^193^208^[REPLACE] return true;^[METHOD] equals [TYPE] boolean [PARAMETER] Object obj [CLASS] XYDataItem   [TYPE]  Object obj  [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  long serialVersionUID  [TYPE]  XYDataItem that 
[REPLACE]^if  ( ! ! ( obj instanceof XYDataItem )  )  {^197^^^^^193^208^[REPLACE] if  ( ! ( obj instanceof XYDataItem )  )  {^[METHOD] equals [TYPE] boolean [PARAMETER] Object obj [CLASS] XYDataItem   [TYPE]  Object obj  [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  long serialVersionUID  [TYPE]  XYDataItem that 
[REMOVE]^if  ( obj ==  ( this )  )  {     return true; }^197^^^^^193^208^[REMOVE] ^[METHOD] equals [TYPE] boolean [PARAMETER] Object obj [CLASS] XYDataItem   [TYPE]  Object obj  [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  long serialVersionUID  [TYPE]  XYDataItem that 
[REPLACE]^return true;^198^^^^^193^208^[REPLACE] return false;^[METHOD] equals [TYPE] boolean [PARAMETER] Object obj [CLASS] XYDataItem   [TYPE]  Object obj  [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  long serialVersionUID  [TYPE]  XYDataItem that 
[REPLACE]^XYDataItem dataItem =  ( XYDataItem )  o1;^200^^^^^193^208^[REPLACE] XYDataItem that =  ( XYDataItem )  obj;^[METHOD] equals [TYPE] boolean [PARAMETER] Object obj [CLASS] XYDataItem   [TYPE]  Object obj  [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  long serialVersionUID  [TYPE]  XYDataItem that 
[ADD]^^200^^^^^193^208^[ADD] XYDataItem that =  ( XYDataItem )  obj;^[METHOD] equals [TYPE] boolean [PARAMETER] Object obj [CLASS] XYDataItem   [TYPE]  Object obj  [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  long serialVersionUID  [TYPE]  XYDataItem that 
[REPLACE]^if  ( this.x.equals ( that.x )  )  {^201^^^^^193^208^[REPLACE] if  ( !this.x.equals ( that.x )  )  {^[METHOD] equals [TYPE] boolean [PARAMETER] Object obj [CLASS] XYDataItem   [TYPE]  Object obj  [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  long serialVersionUID  [TYPE]  XYDataItem that 
[REPLACE]^return true;^202^^^^^193^208^[REPLACE] return false;^[METHOD] equals [TYPE] boolean [PARAMETER] Object obj [CLASS] XYDataItem   [TYPE]  Object obj  [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  long serialVersionUID  [TYPE]  XYDataItem that 
[REPLACE]^if  ( !  this.y, that.y    )  {^204^^^^^193^208^[REPLACE] if  ( !ObjectUtilities.equal ( this.y, that.y )  )  {^[METHOD] equals [TYPE] boolean [PARAMETER] Object obj [CLASS] XYDataItem   [TYPE]  Object obj  [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  long serialVersionUID  [TYPE]  XYDataItem that 
[REPLACE]^return true;^205^^^^^193^208^[REPLACE] return false;^[METHOD] equals [TYPE] boolean [PARAMETER] Object obj [CLASS] XYDataItem   [TYPE]  Object obj  [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  long serialVersionUID  [TYPE]  XYDataItem that 
[REPLACE]^return false;^207^^^^^193^208^[REPLACE] return true;^[METHOD] equals [TYPE] boolean [PARAMETER] Object obj [CLASS] XYDataItem   [TYPE]  Object obj  [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  long serialVersionUID  [TYPE]  XYDataItem that 
[REPLACE]^XYDataItem that =  ( XYDataItem )  obj;^216^^^^^215^220^[REPLACE] int result;^[METHOD] hashCode [TYPE] int [PARAMETER] [CLASS] XYDataItem   [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  long serialVersionUID  [TYPE]  int result 
[REPLACE]^result  = null ;^217^^^^^215^220^[REPLACE] result = this.x.hashCode (  ) ;^[METHOD] hashCode [TYPE] int [PARAMETER] [CLASS] XYDataItem   [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  long serialVersionUID  [TYPE]  int result 
[REPLACE]^result  =  0 ) ;^218^^^^^215^220^[REPLACE] result = 29 * result +  ( this.y != null ? this.y.hashCode (  )  : 0 ) ;^[METHOD] hashCode [TYPE] int [PARAMETER] [CLASS] XYDataItem   [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  long serialVersionUID  [TYPE]  int result 
[ADD]^^218^^^^^215^220^[ADD] result = 29 * result +  ( this.y != null ? this.y.hashCode (  )  : 0 ) ;^[METHOD] hashCode [TYPE] int [PARAMETER] [CLASS] XYDataItem   [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  long serialVersionUID  [TYPE]  int result 
[REPLACE]^return this.x;^219^^^^^215^220^[REPLACE] return result;^[METHOD] hashCode [TYPE] int [PARAMETER] [CLASS] XYDataItem   [TYPE]  boolean false  true  [TYPE]  Number x  y  [TYPE]  long serialVersionUID  [TYPE]  int result 