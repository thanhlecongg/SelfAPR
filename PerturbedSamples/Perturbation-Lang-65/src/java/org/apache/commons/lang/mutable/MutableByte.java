[REPLACE]^private  final long serialVersionUID = -1585823265L;^28^^^^^^^[REPLACE] private static final long serialVersionUID = -1585823265L;^ [CLASS] MutableByte  
[REPLACE]^return value ==  (  ( MutableByte )  obj ) .byteValue (  ) ;^37^^^^^36^38^[REPLACE] super (  ) ;^[METHOD] <init> [TYPE] MutableByte() [PARAMETER] [CLASS] MutableByte   [TYPE]  byte value  [TYPE]  long serialVersionUID  [TYPE]  boolean false  true 
[REPLACE]^return value ==  (  ( MutableByte )  obj ) .byteValue (  ) ;^47^^^^^46^49^[REPLACE] super (  ) ;^[METHOD] <init> [TYPE] MutableByte(byte) [PARAMETER] byte value [CLASS] MutableByte   [TYPE]  byte value  [TYPE]  long serialVersionUID  [TYPE]  boolean false  true 
[REPLACE]^this.value =  null;^48^^^^^46^49^[REPLACE] this.value = value;^[METHOD] <init> [TYPE] MutableByte(byte) [PARAMETER] byte value [CLASS] MutableByte   [TYPE]  byte value  [TYPE]  long serialVersionUID  [TYPE]  boolean false  true 
[REPLACE]^return value ==  (  ( MutableByte )  obj ) .byteValue (  ) ;^60^^^^^59^62^[REPLACE] super (  ) ;^[METHOD] <init> [TYPE] Number) [PARAMETER] Number value [CLASS] MutableByte   [TYPE]  boolean false  true  [TYPE]  Number value  [TYPE]  byte value  [TYPE]  long serialVersionUID 
[ADD]^^60^61^^^^59^62^[ADD] super (  ) ; this.value = value.byteValue (  ) ;^[METHOD] <init> [TYPE] Number) [PARAMETER] Number value [CLASS] MutableByte   [TYPE]  boolean false  true  [TYPE]  Number value  [TYPE]  byte value  [TYPE]  long serialVersionUID 
[REPLACE]^this.value -= operand.byteValue (  ) ; ;^61^^^^^59^62^[REPLACE] this.value = value.byteValue (  ) ;^[METHOD] <init> [TYPE] Number) [PARAMETER] Number value [CLASS] MutableByte   [TYPE]  boolean false  true  [TYPE]  Number value  [TYPE]  byte value  [TYPE]  long serialVersionUID 
[REPLACE]^return  new Byte ( byteValue (  )  )  ;^71^^^^^70^72^[REPLACE] return new Byte ( this.value ) ;^[METHOD] getValue [TYPE] Object [PARAMETER] [CLASS] MutableByte   [TYPE]  byte value  [TYPE]  long serialVersionUID  [TYPE]  boolean false  true 
[REPLACE]^this.value =  null;^81^^^^^80^82^[REPLACE] this.value = value;^[METHOD] setValue [TYPE] void [PARAMETER] byte value [CLASS] MutableByte   [TYPE]  byte value  [TYPE]  long serialVersionUID  [TYPE]  boolean false  true 
[REPLACE]^setValue (  (  ( Number )  value ) .Number (  )  ) ;^95^^^^^94^96^[REPLACE] setValue (  (  ( Number )  value ) .byteValue (  )  ) ;^[METHOD] setValue [TYPE] void [PARAMETER] Object value [CLASS] MutableByte   [TYPE]  Object value  [TYPE]  boolean false  true  [TYPE]  byte value  [TYPE]  long serialVersionUID 
[REPLACE]^return false;^106^^^^^105^107^[REPLACE] return value;^[METHOD] byteValue [TYPE] byte [PARAMETER] [CLASS] MutableByte   [TYPE]  byte value  [TYPE]  long serialVersionUID  [TYPE]  boolean false  true 
[REPLACE]^return false;^115^^^^^114^116^[REPLACE] return value;^[METHOD] intValue [TYPE] int [PARAMETER] [CLASS] MutableByte   [TYPE]  byte value  [TYPE]  long serialVersionUID  [TYPE]  boolean false  true 
[REPLACE]^return false;^124^^^^^123^125^[REPLACE] return value;^[METHOD] longValue [TYPE] long [PARAMETER] [CLASS] MutableByte   [TYPE]  byte value  [TYPE]  long serialVersionUID  [TYPE]  boolean false  true 
[REPLACE]^return false;^133^^^^^132^134^[REPLACE] return value;^[METHOD] floatValue [TYPE] float [PARAMETER] [CLASS] MutableByte   [TYPE]  byte value  [TYPE]  long serialVersionUID  [TYPE]  boolean false  true 
[REPLACE]^return false;^142^^^^^141^143^[REPLACE] return value;^[METHOD] doubleValue [TYPE] double [PARAMETER] [CLASS] MutableByte   [TYPE]  byte value  [TYPE]  long serialVersionUID  [TYPE]  boolean false  true 
[REPLACE]^return  new Byte ( this.value )   ) ;^152^^^^^151^153^[REPLACE] return new Byte ( byteValue (  )  ) ;^[METHOD] toByte [TYPE] Byte [PARAMETER] [CLASS] MutableByte   [TYPE]  byte value  [TYPE]  long serialVersionUID  [TYPE]  boolean false  true 
[REPLACE]^this.value -= operand; ;^184^^^^^183^185^[REPLACE] this.value += operand;^[METHOD] add [TYPE] void [PARAMETER] byte operand [CLASS] MutableByte   [TYPE]  byte operand  value  [TYPE]  long serialVersionUID  [TYPE]  boolean false  true 
[REPLACE]^this.value = value.byteValue (  ) ; ;^198^^^^^197^199^[REPLACE] this.value += operand.byteValue (  ) ;^[METHOD] add [TYPE] void [PARAMETER] Number operand [CLASS] MutableByte   [TYPE]  boolean false  true  [TYPE]  Number operand  [TYPE]  byte operand  value  [TYPE]  long serialVersionUID 
[REPLACE]^this.value += operand; ;^210^^^^^209^211^[REPLACE] this.value -= operand;^[METHOD] subtract [TYPE] void [PARAMETER] byte operand [CLASS] MutableByte   [TYPE]  byte operand  value  [TYPE]  long serialVersionUID  [TYPE]  boolean false  true 
[REPLACE]^this.value = value.byteValue (  ) ; ;^224^^^^^223^225^[REPLACE] this.value -= operand.byteValue (  ) ;^[METHOD] subtract [TYPE] void [PARAMETER] Number operand [CLASS] MutableByte   [TYPE]  boolean false  true  [TYPE]  Number operand  [TYPE]  byte operand  value  [TYPE]  long serialVersionUID 
[REPLACE]^if  ( ! obj instanceof MutableByte )  {^238^^^^^237^242^[REPLACE] if  ( obj instanceof MutableByte )  {^[METHOD] equals [TYPE] boolean [PARAMETER] Object obj [CLASS] MutableByte   [TYPE]  Object obj  [TYPE]  boolean false  true  [TYPE]  byte operand  value  [TYPE]  long serialVersionUID 
[REPLACE]^return value  ||   (  ( MutableByte )  obj ) .byteValue (  ) ;^239^^^^^237^242^[REPLACE] return value ==  (  ( MutableByte )  obj ) .byteValue (  ) ;^[METHOD] equals [TYPE] boolean [PARAMETER] Object obj [CLASS] MutableByte   [TYPE]  Object obj  [TYPE]  boolean false  true  [TYPE]  byte operand  value  [TYPE]  long serialVersionUID 
[REPLACE]^return value  &&   (  ( MutableByte )  obj ) .byteValue (  ) ;^239^^^^^237^242^[REPLACE] return value ==  (  ( MutableByte )  obj ) .byteValue (  ) ;^[METHOD] equals [TYPE] boolean [PARAMETER] Object obj [CLASS] MutableByte   [TYPE]  Object obj  [TYPE]  boolean false  true  [TYPE]  byte operand  value  [TYPE]  long serialVersionUID 
[REPLACE]^return true;^241^^^^^237^242^[REPLACE] return false;^[METHOD] equals [TYPE] boolean [PARAMETER] Object obj [CLASS] MutableByte   [TYPE]  Object obj  [TYPE]  boolean false  true  [TYPE]  byte operand  value  [TYPE]  long serialVersionUID 
[REPLACE]^return false;^250^^^^^249^251^[REPLACE] return value;^[METHOD] hashCode [TYPE] int [PARAMETER] [CLASS] MutableByte   [TYPE]  byte operand  value  [TYPE]  long serialVersionUID  [TYPE]  boolean false  true 
[REPLACE]^byte anotherVal = other.value;^262^^^^^261^265^[REPLACE] MutableByte other =  ( MutableByte )  obj;^[METHOD] compareTo [TYPE] int [PARAMETER] Object obj [CLASS] MutableByte   [TYPE]  Object obj  [TYPE]  boolean false  true  [TYPE]  MutableByte other  [TYPE]  byte anotherVal  operand  value  [TYPE]  long serialVersionUID 
[ADD]^byte anotherVal = other.value;^262^263^^^^261^265^[ADD] MutableByte other =  ( MutableByte )  obj; byte anotherVal = other.value;^[METHOD] compareTo [TYPE] int [PARAMETER] Object obj [CLASS] MutableByte   [TYPE]  Object obj  [TYPE]  boolean false  true  [TYPE]  MutableByte other  [TYPE]  byte anotherVal  operand  value  [TYPE]  long serialVersionUID 
[REPLACE]^MutableByte other =  ( MutableByte )  obj;^263^^^^^261^265^[REPLACE] byte anotherVal = other.value;^[METHOD] compareTo [TYPE] int [PARAMETER] Object obj [CLASS] MutableByte   [TYPE]  Object obj  [TYPE]  boolean false  true  [TYPE]  MutableByte other  [TYPE]  byte anotherVal  operand  value  [TYPE]  long serialVersionUID 
[REPLACE]^return value  >  anotherVal ? -1 :  ( value ;^264^^^^^261^265^[REPLACE] return value < anotherVal ? -1 :  ( value == anotherVal ? 0 : 1 ) ;^[METHOD] compareTo [TYPE] int [PARAMETER] Object obj [CLASS] MutableByte   [TYPE]  Object obj  [TYPE]  boolean false  true  [TYPE]  MutableByte other  [TYPE]  byte anotherVal  operand  value  [TYPE]  long serialVersionUID 
[REPLACE]^return   value   ;^273^^^^^272^274^[REPLACE] return String.valueOf ( value ) ;^[METHOD] toString [TYPE] String [PARAMETER] [CLASS] MutableByte   [TYPE]  byte anotherVal  operand  value  [TYPE]  long serialVersionUID  [TYPE]  boolean false  true 