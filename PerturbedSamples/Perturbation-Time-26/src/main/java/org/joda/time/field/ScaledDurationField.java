[REPLACE]^private static final  short  serialVersionUID = -3205227092378684157L;^34^^^^^^^[REPLACE] private static final long serialVersionUID = -3205227092378684157L;^ [CLASS] ScaledDurationField  
[REPLACE]^private final  short  iScalar;^36^^^^^^^[REPLACE] private final int iScalar;^ [CLASS] ScaledDurationField  
[REPLACE]^return getWrappedField (  ) .getValueAsLong ( duration )  / iScalar;^47^^^^^46^52^[REPLACE] super ( field, type ) ;^[METHOD] <init> [TYPE] DurationFieldType,int) [PARAMETER] DurationField field DurationFieldType type int scalar [CLASS] ScaledDurationField   [TYPE]  boolean false  true  [TYPE]  DurationField field  [TYPE]  long serialVersionUID  [TYPE]  int iScalar  scalar  [TYPE]  DurationFieldType type 
[REPLACE]^if  ( scalar == 0 && scalar == 1 )  {^48^^^^^46^52^[REPLACE] if  ( scalar == 0 || scalar == 1 )  {^[METHOD] <init> [TYPE] DurationFieldType,int) [PARAMETER] DurationField field DurationFieldType type int scalar [CLASS] ScaledDurationField   [TYPE]  boolean false  true  [TYPE]  DurationField field  [TYPE]  long serialVersionUID  [TYPE]  int iScalar  scalar  [TYPE]  DurationFieldType type 
[REPLACE]^return ;^49^^^^^46^52^[REPLACE] throw new IllegalArgumentException  (" ")  ;^[METHOD] <init> [TYPE] DurationFieldType,int) [PARAMETER] DurationField field DurationFieldType type int scalar [CLASS] ScaledDurationField   [TYPE]  boolean false  true  [TYPE]  DurationField field  [TYPE]  long serialVersionUID  [TYPE]  int iScalar  scalar  [TYPE]  DurationFieldType type 
[REPLACE]^iScalar =  null;^51^^^^^46^52^[REPLACE] iScalar = scalar;^[METHOD] <init> [TYPE] DurationFieldType,int) [PARAMETER] DurationField field DurationFieldType type int scalar [CLASS] ScaledDurationField   [TYPE]  boolean false  true  [TYPE]  DurationField field  [TYPE]  long serialVersionUID  [TYPE]  int iScalar  scalar  [TYPE]  DurationFieldType type 
[REPLACE]^return getWrappedField (  ) .getValue ( serialVersionUID )   ;^55^^^^^54^56^[REPLACE] return getWrappedField (  ) .getValue ( duration )  / iScalar;^[METHOD] getValue [TYPE] int [PARAMETER] long duration [CLASS] ScaledDurationField   [TYPE]  long duration  serialVersionUID  [TYPE]  int iScalar  scalar  [TYPE]  boolean false  true 
[REPLACE]^return getWrappedField (  ) .getValueAsLong ( duration )   ;^59^^^^^58^60^[REPLACE] return getWrappedField (  ) .getValueAsLong ( duration )  / iScalar;^[METHOD] getValueAsLong [TYPE] long [PARAMETER] long duration [CLASS] ScaledDurationField   [TYPE]  long duration  serialVersionUID  [TYPE]  int iScalar  scalar  [TYPE]  boolean false  true 
[REPLACE]^return getWrappedField (  ) .getValue ( duration, instant )   ;^63^^^^^62^64^[REPLACE] return getWrappedField (  ) .getValue ( duration, instant )  / iScalar;^[METHOD] getValue [TYPE] int [PARAMETER] long duration long instant [CLASS] ScaledDurationField   [TYPE]  long duration  instant  serialVersionUID  [TYPE]  int iScalar  scalar  [TYPE]  boolean false  true 
[REPLACE]^return getWrappedField (  ) .getValueAsLong ( duration, instant )   ;^67^^^^^66^68^[REPLACE] return getWrappedField (  ) .getValueAsLong ( duration, instant )  / iScalar;^[METHOD] getValueAsLong [TYPE] long [PARAMETER] long duration long instant [CLASS] ScaledDurationField   [TYPE]  long duration  instant  serialVersionUID  [TYPE]  int iScalar  scalar  [TYPE]  boolean false  true 
[REPLACE]^long scaled =  (  ( long )  value )   ;^71^^^^^70^73^[REPLACE] long scaled =  (  ( long )  value )  *  (  ( long )  iScalar ) ;^[METHOD] getMillis [TYPE] long [PARAMETER] int value [CLASS] ScaledDurationField   [TYPE]  long duration  instant  scaled  serialVersionUID  [TYPE]  int iScalar  scalar  value  [TYPE]  boolean false  true 
[REPLACE]^return getWrappedField (  )  .getMillis ( scalar , serialVersionUID )  ;^72^^^^^70^73^[REPLACE] return getWrappedField (  ) .getMillis ( scaled ) ;^[METHOD] getMillis [TYPE] long [PARAMETER] int value [CLASS] ScaledDurationField   [TYPE]  long duration  instant  scaled  serialVersionUID  [TYPE]  int iScalar  scalar  value  [TYPE]  boolean false  true 
[REPLACE]^long scaled = FieldUtils.safeMultiply ( serialVersionUID, iScalar ) ;^76^^^^^75^78^[REPLACE] long scaled = FieldUtils.safeMultiply ( value, iScalar ) ;^[METHOD] getMillis [TYPE] long [PARAMETER] long value [CLASS] ScaledDurationField   [TYPE]  long duration  instant  scaled  serialVersionUID  value  [TYPE]  int iScalar  scalar  value  [TYPE]  boolean false  true 
[REPLACE]^return getWrappedField (  ) .getValue ( scaled ) ;^77^^^^^75^78^[REPLACE] return getWrappedField (  ) .getMillis ( scaled ) ;^[METHOD] getMillis [TYPE] long [PARAMETER] long value [CLASS] ScaledDurationField   [TYPE]  long duration  instant  scaled  serialVersionUID  value  [TYPE]  int iScalar  scalar  value  [TYPE]  boolean false  true 
[REPLACE]^long scaled =  (  ( long )  value )   ;^81^^^^^80^83^[REPLACE] long scaled =  (  ( long )  value )  *  (  ( long )  iScalar ) ;^[METHOD] getMillis [TYPE] long [PARAMETER] int value long instant [CLASS] ScaledDurationField   [TYPE]  long duration  instant  scaled  serialVersionUID  value  [TYPE]  int iScalar  scalar  value  [TYPE]  boolean false  true 
[ADD]^^81^^^^^80^83^[ADD] long scaled =  (  ( long )  value )  *  (  ( long )  iScalar ) ;^[METHOD] getMillis [TYPE] long [PARAMETER] int value long instant [CLASS] ScaledDurationField   [TYPE]  long duration  instant  scaled  serialVersionUID  value  [TYPE]  int iScalar  scalar  value  [TYPE]  boolean false  true 
[REPLACE]^return getWrappedField (  ) .getMillis ( scaled ) ;^82^^^^^80^83^[REPLACE] return getWrappedField (  ) .getMillis ( scaled, instant ) ;^[METHOD] getMillis [TYPE] long [PARAMETER] int value long instant [CLASS] ScaledDurationField   [TYPE]  long duration  instant  scaled  serialVersionUID  value  [TYPE]  int iScalar  scalar  value  [TYPE]  boolean false  true 
[REPLACE]^long scaled = FieldUtils.safeMultiply ( serialVersionUID, iScalar ) ;^86^^^^^85^88^[REPLACE] long scaled = FieldUtils.safeMultiply ( value, iScalar ) ;^[METHOD] getMillis [TYPE] long [PARAMETER] long value long instant [CLASS] ScaledDurationField   [TYPE]  long duration  instant  scaled  serialVersionUID  value  [TYPE]  int iScalar  scalar  value  [TYPE]  boolean false  true 
[ADD]^^86^^^^^85^88^[ADD] long scaled = FieldUtils.safeMultiply ( value, iScalar ) ;^[METHOD] getMillis [TYPE] long [PARAMETER] long value long instant [CLASS] ScaledDurationField   [TYPE]  long duration  instant  scaled  serialVersionUID  value  [TYPE]  int iScalar  scalar  value  [TYPE]  boolean false  true 
[REPLACE]^return getWrappedField (  ) .getMillis ( value, instant ) ;^87^^^^^85^88^[REPLACE] return getWrappedField (  ) .getMillis ( scaled, instant ) ;^[METHOD] getMillis [TYPE] long [PARAMETER] long value long instant [CLASS] ScaledDurationField   [TYPE]  long duration  instant  scaled  serialVersionUID  value  [TYPE]  int iScalar  scalar  value  [TYPE]  boolean false  true 
[REPLACE]^long scaled =  (  ( long )  value )   ;^91^^^^^90^93^[REPLACE] long scaled =  (  ( long )  value )  *  (  ( long )  iScalar ) ;^[METHOD] add [TYPE] long [PARAMETER] long instant int value [CLASS] ScaledDurationField   [TYPE]  long duration  instant  scaled  serialVersionUID  value  [TYPE]  int iScalar  scalar  value  [TYPE]  boolean false  true 
[ADD]^^91^^^^^90^93^[ADD] long scaled =  (  ( long )  value )  *  (  ( long )  iScalar ) ;^[METHOD] add [TYPE] long [PARAMETER] long instant int value [CLASS] ScaledDurationField   [TYPE]  long duration  instant  scaled  serialVersionUID  value  [TYPE]  int iScalar  scalar  value  [TYPE]  boolean false  true 
[REPLACE]^return getWrappedField (  )  .add ( instant , scalar )  ;^92^^^^^90^93^[REPLACE] return getWrappedField (  ) .add ( instant, scaled ) ;^[METHOD] add [TYPE] long [PARAMETER] long instant int value [CLASS] ScaledDurationField   [TYPE]  long duration  instant  scaled  serialVersionUID  value  [TYPE]  int iScalar  scalar  value  [TYPE]  boolean false  true 
[REPLACE]^long scaled = FieldUtils.safeMultiply ( serialVersionUID, iScalar ) ;^96^^^^^95^98^[REPLACE] long scaled = FieldUtils.safeMultiply ( value, iScalar ) ;^[METHOD] add [TYPE] long [PARAMETER] long instant long value [CLASS] ScaledDurationField   [TYPE]  long duration  instant  scaled  serialVersionUID  value  [TYPE]  int iScalar  scalar  value  [TYPE]  boolean false  true 
[REPLACE]^return getWrappedField (  ) .add ( instant, value ) ;^97^^^^^95^98^[REPLACE] return getWrappedField (  ) .add ( instant, scaled ) ;^[METHOD] add [TYPE] long [PARAMETER] long instant long value [CLASS] ScaledDurationField   [TYPE]  long duration  instant  scaled  serialVersionUID  value  [TYPE]  int iScalar  scalar  value  [TYPE]  boolean false  true 
[REPLACE]^return getWrappedField (  )  .getDifferenceAsLong ( value , serialVersionUID )   / iScalar;^101^^^^^100^102^[REPLACE] return getWrappedField (  ) .getDifference ( minuendInstant, subtrahendInstant )  / iScalar;^[METHOD] getDifference [TYPE] int [PARAMETER] long minuendInstant long subtrahendInstant [CLASS] ScaledDurationField   [TYPE]  long duration  instant  minuendInstant  scaled  serialVersionUID  subtrahendInstant  value  [TYPE]  int iScalar  scalar  value  [TYPE]  boolean false  true 
[REPLACE]^return getWrappedField (  ) .getDifferenceAsLong ( minuendInstant, subtrahendInstant )   ;^105^^^^^104^106^[REPLACE] return getWrappedField (  ) .getDifferenceAsLong ( minuendInstant, subtrahendInstant )  / iScalar;^[METHOD] getDifferenceAsLong [TYPE] long [PARAMETER] long minuendInstant long subtrahendInstant [CLASS] ScaledDurationField   [TYPE]  long duration  instant  minuendInstant  scaled  serialVersionUID  subtrahendInstant  value  [TYPE]  int iScalar  scalar  value  [TYPE]  boolean false  true 
[REPLACE]^return getWrappedField (  ) .getUnitMillis (  )   ;^109^^^^^108^110^[REPLACE] return getWrappedField (  ) .getUnitMillis (  )  * iScalar;^[METHOD] getUnitMillis [TYPE] long [PARAMETER] [CLASS] ScaledDurationField   [TYPE]  long duration  instant  minuendInstant  scaled  serialVersionUID  subtrahendInstant  value  [TYPE]  int iScalar  scalar  value  [TYPE]  boolean false  true 
[REPLACE]^return hash;^119^^^^^118^120^[REPLACE] return iScalar;^[METHOD] getScalar [TYPE] int [PARAMETER] [CLASS] ScaledDurationField   [TYPE]  long duration  instant  minuendInstant  scaled  serialVersionUID  subtrahendInstant  value  [TYPE]  int iScalar  scalar  value  [TYPE]  boolean false  true 
[REPLACE]^if  ( this  ||  obj )  {^130^^^^^129^139^[REPLACE] if  ( this == obj )  {^[METHOD] equals [TYPE] boolean [PARAMETER] Object obj [CLASS] ScaledDurationField   [TYPE]  Object obj  [TYPE]  ScaledDurationField other  [TYPE]  boolean false  true  [TYPE]  long duration  instant  minuendInstant  scaled  serialVersionUID  subtrahendInstant  value  [TYPE]  int iScalar  scalar  value 
[REPLACE]^}  else {^132^^^^^129^139^[REPLACE] } else if  ( obj instanceof ScaledDurationField )  {^[METHOD] equals [TYPE] boolean [PARAMETER] Object obj [CLASS] ScaledDurationField   [TYPE]  Object obj  [TYPE]  ScaledDurationField other  [TYPE]  boolean false  true  [TYPE]  long duration  instant  minuendInstant  scaled  serialVersionUID  subtrahendInstant  value  [TYPE]  int iScalar  scalar  value 
[REPLACE]^return  ( getWrappedField (  ) .getValue ( other.getWrappedField (  )  )  )  ;^134^135^136^^^129^139^[REPLACE] return  ( getWrappedField (  ) .equals ( other.getWrappedField (  )  )  )  && ( getType (  )  == other.getType (  )  )  && ( iScalar == other.iScalar ) ;^[METHOD] equals [TYPE] boolean [PARAMETER] Object obj [CLASS] ScaledDurationField   [TYPE]  Object obj  [TYPE]  ScaledDurationField other  [TYPE]  boolean false  true  [TYPE]  long duration  instant  minuendInstant  scaled  serialVersionUID  subtrahendInstant  value  [TYPE]  int iScalar  scalar  value 
[REPLACE]^int hash =  ( int )   ( scalar ^  ( scalar >>> 32 )  ) ;^133^^^^^129^139^[REPLACE] ScaledDurationField other =  ( ScaledDurationField )  obj;^[METHOD] equals [TYPE] boolean [PARAMETER] Object obj [CLASS] ScaledDurationField   [TYPE]  Object obj  [TYPE]  ScaledDurationField other  [TYPE]  boolean false  true  [TYPE]  long duration  instant  minuendInstant  scaled  serialVersionUID  subtrahendInstant  value  [TYPE]  int iScalar  scalar  value 
[REPLACE]^return true ;^134^135^136^^^129^139^[REPLACE] return  ( getWrappedField (  ) .equals ( other.getWrappedField (  )  )  )  && ( getType (  )  == other.getType (  )  )  && ( iScalar == other.iScalar ) ;^[METHOD] equals [TYPE] boolean [PARAMETER] Object obj [CLASS] ScaledDurationField   [TYPE]  Object obj  [TYPE]  ScaledDurationField other  [TYPE]  boolean false  true  [TYPE]  long duration  instant  minuendInstant  scaled  serialVersionUID  subtrahendInstant  value  [TYPE]  int iScalar  scalar  value 
[REPLACE]^return false;^131^^^^^129^139^[REPLACE] return true;^[METHOD] equals [TYPE] boolean [PARAMETER] Object obj [CLASS] ScaledDurationField   [TYPE]  Object obj  [TYPE]  ScaledDurationField other  [TYPE]  boolean false  true  [TYPE]  long duration  instant  minuendInstant  scaled  serialVersionUID  subtrahendInstant  value  [TYPE]  int iScalar  scalar  value 
[REPLACE]^}  if  ( ! obj instanceof ScaledDurationField )  {^132^^^^^129^139^[REPLACE] } else if  ( obj instanceof ScaledDurationField )  {^[METHOD] equals [TYPE] boolean [PARAMETER] Object obj [CLASS] ScaledDurationField   [TYPE]  Object obj  [TYPE]  ScaledDurationField other  [TYPE]  boolean false  true  [TYPE]  long duration  instant  minuendInstant  scaled  serialVersionUID  subtrahendInstant  value  [TYPE]  int iScalar  scalar  value 
[REPLACE]^return  ( getWrappedField (  ) .equals ( other .getValue ( instant )   )  )  && ( getType (  )  == other.getType (  )  )  && ( iScalar == other.iScalar ) ;^134^135^136^^^129^139^[REPLACE] return  ( getWrappedField (  ) .equals ( other.getWrappedField (  )  )  )  && ( getType (  )  == other.getType (  )  )  && ( iScalar == other.iScalar ) ;^[METHOD] equals [TYPE] boolean [PARAMETER] Object obj [CLASS] ScaledDurationField   [TYPE]  Object obj  [TYPE]  ScaledDurationField other  [TYPE]  boolean false  true  [TYPE]  long duration  instant  minuendInstant  scaled  serialVersionUID  subtrahendInstant  value  [TYPE]  int iScalar  scalar  value 
[REPLACE]^return false ;^134^135^136^^^129^139^[REPLACE] return  ( getWrappedField (  ) .equals ( other.getWrappedField (  )  )  )  && ( getType (  )  == other.getType (  )  )  && ( iScalar == other.iScalar ) ;^[METHOD] equals [TYPE] boolean [PARAMETER] Object obj [CLASS] ScaledDurationField   [TYPE]  Object obj  [TYPE]  ScaledDurationField other  [TYPE]  boolean false  true  [TYPE]  long duration  instant  minuendInstant  scaled  serialVersionUID  subtrahendInstant  value  [TYPE]  int iScalar  scalar  value 
[REPLACE]^return true;^138^^^^^129^139^[REPLACE] return false;^[METHOD] equals [TYPE] boolean [PARAMETER] Object obj [CLASS] ScaledDurationField   [TYPE]  Object obj  [TYPE]  ScaledDurationField other  [TYPE]  boolean false  true  [TYPE]  long duration  instant  minuendInstant  scaled  serialVersionUID  subtrahendInstant  value  [TYPE]  int iScalar  scalar  value 
[REPLACE]^int hash =  ( int )   ( scalar ^  ( scalar >>> 32 )  ) ;^147^^^^^146^152^[REPLACE] long scalar = iScalar;^[METHOD] hashCode [TYPE] int [PARAMETER] [CLASS] ScaledDurationField   [TYPE]  long duration  instant  minuendInstant  scalar  scaled  serialVersionUID  subtrahendInstant  value  [TYPE]  int hash  iScalar  scalar  value  [TYPE]  boolean false  true 
[ADD]^^147^148^^^^146^152^[ADD] long scalar = iScalar; int hash =  ( int )   ( scalar ^  ( scalar >>> 32 )  ) ;^[METHOD] hashCode [TYPE] int [PARAMETER] [CLASS] ScaledDurationField   [TYPE]  long duration  instant  minuendInstant  scalar  scaled  serialVersionUID  subtrahendInstant  value  [TYPE]  int hash  iScalar  scalar  value  [TYPE]  boolean false  true 
[REPLACE]^int hash =  ( int )   (serialVersionUID ^  ( value >>> 0 )  ) ;^148^^^^^146^152^[REPLACE] int hash =  ( int )   ( scalar ^  ( scalar >>> 32 )  ) ;^[METHOD] hashCode [TYPE] int [PARAMETER] [CLASS] ScaledDurationField   [TYPE]  long duration  instant  minuendInstant  scalar  scaled  serialVersionUID  subtrahendInstant  value  [TYPE]  int hash  iScalar  scalar  value  [TYPE]  boolean false  true 
[REPLACE]^hash += getWrappedField (  ) .hashCode (  ) ; ;^149^^^^^146^152^[REPLACE] hash += getType (  ) .hashCode (  ) ;^[METHOD] hashCode [TYPE] int [PARAMETER] [CLASS] ScaledDurationField   [TYPE]  long duration  instant  minuendInstant  scalar  scaled  serialVersionUID  subtrahendInstant  value  [TYPE]  int hash  iScalar  scalar  value  [TYPE]  boolean false  true 
[REPLACE]^hash += getType (  ) .hashCode (  ) ; ;^150^^^^^146^152^[REPLACE] hash += getWrappedField (  ) .hashCode (  ) ;^[METHOD] hashCode [TYPE] int [PARAMETER] [CLASS] ScaledDurationField   [TYPE]  long duration  instant  minuendInstant  scalar  scaled  serialVersionUID  subtrahendInstant  value  [TYPE]  int hash  iScalar  scalar  value  [TYPE]  boolean false  true 
[REPLACE]^return value;^151^^^^^146^152^[REPLACE] return hash;^[METHOD] hashCode [TYPE] int [PARAMETER] [CLASS] ScaledDurationField   [TYPE]  long duration  instant  minuendInstant  scalar  scaled  serialVersionUID  subtrahendInstant  value  [TYPE]  int hash  iScalar  scalar  value  [TYPE]  boolean false  true 