[REPLACE]^private  final long serialVersionUID = 2751513470325494890;^59^^^^^^^[REPLACE] private static final long serialVersionUID = 2751513470325494890L;^ [CLASS] ComparableObjectItem  
[REPLACE]^if  ( x != this )  {^74^^^^^73^79^[REPLACE] if  ( x == null )  {^[METHOD] <init> [TYPE] Object) [PARAMETER] Comparable x Object y [CLASS] ComparableObjectItem   [TYPE]  Comparable x  [TYPE]  Object obj  y  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 
[REPLACE]^return ;^75^^^^^73^79^[REPLACE] throw new IllegalArgumentException  (" ")  ;^[METHOD] <init> [TYPE] Object) [PARAMETER] Comparable x Object y [CLASS] ComparableObjectItem   [TYPE]  Comparable x  [TYPE]  Object obj  y  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 
[REPLACE]^this.x =  null;^77^^^^^73^79^[REPLACE] this.x = x;^[METHOD] <init> [TYPE] Object) [PARAMETER] Comparable x Object y [CLASS] ComparableObjectItem   [TYPE]  Comparable x  [TYPE]  Object obj  y  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 
[REPLACE]^this.obj =  null;^78^^^^^73^79^[REPLACE] this.obj = y;^[METHOD] <init> [TYPE] Object) [PARAMETER] Comparable x Object y [CLASS] ComparableObjectItem   [TYPE]  Comparable x  [TYPE]  Object obj  y  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 
[REPLACE]^return this.obj;^87^^^^^86^88^[REPLACE] return this.x;^[METHOD] getComparable [TYPE] Comparable [PARAMETER] [CLASS] ComparableObjectItem   [TYPE]  Comparable x  [TYPE]  Object obj  y  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 
[REPLACE]^return this.x;^96^^^^^95^97^[REPLACE] return this.obj;^[METHOD] getObject [TYPE] Object [PARAMETER] [CLASS] ComparableObjectItem   [TYPE]  Comparable x  [TYPE]  Object obj  y  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 
[REPLACE]^this.obj =  null;^106^^^^^105^107^[REPLACE] this.obj = y;^[METHOD] setObject [TYPE] void [PARAMETER] Object y [CLASS] ComparableObjectItem   [TYPE]  Comparable x  [TYPE]  Object obj  y  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 
[REPLACE]^ComparableObjectItem that =  ( ComparableObjectItem )  o1;^123^^^^^121^141^[REPLACE] int result;^[METHOD] compareTo [TYPE] int [PARAMETER] Object o1 [CLASS] ComparableObjectItem   [TYPE]  Comparable x  [TYPE]  Object o1  obj  y  [TYPE]  ComparableObjectItem that  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  int result 
[REPLACE]^if  ( ! o3 instanceof ComparableObjectItem )  {^127^^^^^121^141^[REPLACE] if  ( o1 instanceof ComparableObjectItem )  {^[METHOD] compareTo [TYPE] int [PARAMETER] Object o1 [CLASS] ComparableObjectItem   [TYPE]  Comparable x  [TYPE]  Object o1  obj  y  [TYPE]  ComparableObjectItem that  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  int result 
[REPLACE]^result = 4;^136^^^^^121^141^[REPLACE] result = 1;^[METHOD] compareTo [TYPE] int [PARAMETER] Object o1 [CLASS] ComparableObjectItem   [TYPE]  Comparable x  [TYPE]  Object o1  obj  y  [TYPE]  ComparableObjectItem that  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  int result 
[REPLACE]^return this.x .ComparableObjectItem ( x , y )  ;^129^^^^^121^141^[REPLACE] return this.x.compareTo ( that.x ) ;^[METHOD] compareTo [TYPE] int [PARAMETER] Object o1 [CLASS] ComparableObjectItem   [TYPE]  Comparable x  [TYPE]  Object o1  obj  y  [TYPE]  ComparableObjectItem that  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  int result 
[REPLACE]^ComparableObjectItem that =  ( ComparableObjectItem )  obj;^128^^^^^121^141^[REPLACE] ComparableObjectItem that =  ( ComparableObjectItem )  o1;^[METHOD] compareTo [TYPE] int [PARAMETER] Object o1 [CLASS] ComparableObjectItem   [TYPE]  Comparable x  [TYPE]  Object o1  obj  y  [TYPE]  ComparableObjectItem that  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  int result 
[REPLACE]^result = 3;^136^^^^^121^141^[REPLACE] result = 1;^[METHOD] compareTo [TYPE] int [PARAMETER] Object o1 [CLASS] ComparableObjectItem   [TYPE]  Comparable x  [TYPE]  Object o1  obj  y  [TYPE]  ComparableObjectItem that  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  int result 
[REPLACE]^return this.x;^129^^^^^121^141^[REPLACE] return this.x.compareTo ( that.x ) ;^[METHOD] compareTo [TYPE] int [PARAMETER] Object o1 [CLASS] ComparableObjectItem   [TYPE]  Comparable x  [TYPE]  Object o1  obj  y  [TYPE]  ComparableObjectItem that  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  int result 
[REPLACE]^return this.x;^139^^^^^121^141^[REPLACE] return result;^[METHOD] compareTo [TYPE] int [PARAMETER] Object o1 [CLASS] ComparableObjectItem   [TYPE]  Comparable x  [TYPE]  Object o1  obj  y  [TYPE]  ComparableObjectItem that  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  int result 
[REPLACE]^return super.hashCode (  ) ;^152^^^^^151^153^[REPLACE] return super.clone (  ) ;^[METHOD] clone [TYPE] Object [PARAMETER] [CLASS] ComparableObjectItem   [TYPE]  Comparable x  [TYPE]  Object o1  obj  y  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 
[REPLACE]^if  ( o1  ||  that )  {^164^^^^^163^178^[REPLACE] if  ( obj == this )  {^[METHOD] equals [TYPE] boolean [PARAMETER] Object obj [CLASS] ComparableObjectItem   [TYPE]  Comparable x  [TYPE]  Object o1  obj  y  [TYPE]  ComparableObjectItem that  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 
[REPLACE]^return false;^165^^^^^163^178^[REPLACE] return true;^[METHOD] equals [TYPE] boolean [PARAMETER] Object obj [CLASS] ComparableObjectItem   [TYPE]  Comparable x  [TYPE]  Object o1  obj  y  [TYPE]  ComparableObjectItem that  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 
[REPLACE]^if  ( ! ! ( o1 instanceof ComparableObjectItem )  )  {^167^^^^^163^178^[REPLACE] if  ( ! ( obj instanceof ComparableObjectItem )  )  {^[METHOD] equals [TYPE] boolean [PARAMETER] Object obj [CLASS] ComparableObjectItem   [TYPE]  Comparable x  [TYPE]  Object o1  obj  y  [TYPE]  ComparableObjectItem that  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 
[REPLACE]^return true;^168^^^^^163^178^[REPLACE] return false;^[METHOD] equals [TYPE] boolean [PARAMETER] Object obj [CLASS] ComparableObjectItem   [TYPE]  Comparable x  [TYPE]  Object o1  obj  y  [TYPE]  ComparableObjectItem that  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 
[REPLACE]^ComparableObjectItem that =  ( ComparableObjectItem )  o1;^170^^^^^163^178^[REPLACE] ComparableObjectItem that =  ( ComparableObjectItem )  obj;^[METHOD] equals [TYPE] boolean [PARAMETER] Object obj [CLASS] ComparableObjectItem   [TYPE]  Comparable x  [TYPE]  Object o1  obj  y  [TYPE]  ComparableObjectItem that  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 
[REPLACE]^if  ( !this.x .setObject ( obj )   )  {^171^^^^^163^178^[REPLACE] if  ( !this.x.equals ( that.x )  )  {^[METHOD] equals [TYPE] boolean [PARAMETER] Object obj [CLASS] ComparableObjectItem   [TYPE]  Comparable x  [TYPE]  Object o1  obj  y  [TYPE]  ComparableObjectItem that  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 
[REMOVE]^if  ( ! ( obj instanceof ComparableObjectItem )  )  {     return false; }^171^^^^^163^178^[REMOVE] ^[METHOD] equals [TYPE] boolean [PARAMETER] Object obj [CLASS] ComparableObjectItem   [TYPE]  Comparable x  [TYPE]  Object o1  obj  y  [TYPE]  ComparableObjectItem that  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 
[REPLACE]^return true;^172^^^^^163^178^[REPLACE] return false;^[METHOD] equals [TYPE] boolean [PARAMETER] Object obj [CLASS] ComparableObjectItem   [TYPE]  Comparable x  [TYPE]  Object o1  obj  y  [TYPE]  ComparableObjectItem that  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 
[REPLACE]^if  ( !  this.obj, that.obj    )  {^174^^^^^163^178^[REPLACE] if  ( !ObjectUtilities.equal ( this.obj, that.obj )  )  {^[METHOD] equals [TYPE] boolean [PARAMETER] Object obj [CLASS] ComparableObjectItem   [TYPE]  Comparable x  [TYPE]  Object o1  obj  y  [TYPE]  ComparableObjectItem that  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 
[REPLACE]^return true;^175^^^^^163^178^[REPLACE] return false;^[METHOD] equals [TYPE] boolean [PARAMETER] Object obj [CLASS] ComparableObjectItem   [TYPE]  Comparable x  [TYPE]  Object o1  obj  y  [TYPE]  ComparableObjectItem that  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 
[REPLACE]^return false;^177^^^^^163^178^[REPLACE] return true;^[METHOD] equals [TYPE] boolean [PARAMETER] Object obj [CLASS] ComparableObjectItem   [TYPE]  Comparable x  [TYPE]  Object o1  obj  y  [TYPE]  ComparableObjectItem that  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 
[REPLACE]^ComparableObjectItem that =  ( ComparableObjectItem )  o1;^186^^^^^185^190^[REPLACE] int result;^[METHOD] hashCode [TYPE] int [PARAMETER] [CLASS] ComparableObjectItem   [TYPE]  Comparable x  [TYPE]  Object o1  obj  y  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  int result 
[REPLACE]^result  =  result ;^187^^^^^185^190^[REPLACE] result = this.x.hashCode (  ) ;^[METHOD] hashCode [TYPE] int [PARAMETER] [CLASS] ComparableObjectItem   [TYPE]  Comparable x  [TYPE]  Object o1  obj  y  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  int result 
[REPLACE]^result  =  this.obj.hashCode (  )  ;^188^^^^^185^190^[REPLACE] result = 29 * result +  ( this.obj != null ? this.obj.hashCode (  )  : 0 ) ;^[METHOD] hashCode [TYPE] int [PARAMETER] [CLASS] ComparableObjectItem   [TYPE]  Comparable x  [TYPE]  Object o1  obj  y  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  int result 
[ADD]^^188^^^^^185^190^[ADD] result = 29 * result +  ( this.obj != null ? this.obj.hashCode (  )  : 0 ) ;^[METHOD] hashCode [TYPE] int [PARAMETER] [CLASS] ComparableObjectItem   [TYPE]  Comparable x  [TYPE]  Object o1  obj  y  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  int result 
[REPLACE]^return this.x;^189^^^^^185^190^[REPLACE] return result;^[METHOD] hashCode [TYPE] int [PARAMETER] [CLASS] ComparableObjectItem   [TYPE]  Comparable x  [TYPE]  Object o1  obj  y  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  int result 