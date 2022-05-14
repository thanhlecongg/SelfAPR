[buglab_swap_variables]^int sum = val2 + val1;^64^^^^^63^71^int sum = val1 + val2;^[CLASS] FieldUtils  [METHOD] safeAdd [RETURN_TYPE] int   int val1 int val2 [VARIABLES] boolean  int  sum  val1  val2  
[buglab_swap_variables]^if  (  ( sum ^ val1 )  < 0 &&  ( val1 ^ val2 )  >= 0 )  {^66^^^^^63^71^if  (  ( val1 ^ sum )  < 0 &&  ( val1 ^ val2 )  >= 0 )  {^[CLASS] FieldUtils  [METHOD] safeAdd [RETURN_TYPE] int   int val1 int val2 [VARIABLES] boolean  int  sum  val1  val2  
[buglab_swap_variables]^if  (  ( val1 ^ val2 )  < 0 &&  ( val1 ^ sum )  >= 0 )  {^66^^^^^63^71^if  (  ( val1 ^ sum )  < 0 &&  ( val1 ^ val2 )  >= 0 )  {^[CLASS] FieldUtils  [METHOD] safeAdd [RETURN_TYPE] int   int val1 int val2 [VARIABLES] boolean  int  sum  val1  val2  
[buglab_swap_variables]^throw new ArithmeticException ( "The calculation caused an overflow: " + val2 + " + " + val1 ) ;^67^68^^^^63^71^throw new ArithmeticException ( "The calculation caused an overflow: " + val1 + " + " + val2 ) ;^[CLASS] FieldUtils  [METHOD] safeAdd [RETURN_TYPE] int   int val1 int val2 [VARIABLES] boolean  int  sum  val1  val2  
[buglab_swap_variables]^long sum = val2 + val1;^82^^^^^81^89^long sum = val1 + val2;^[CLASS] FieldUtils  [METHOD] safeAdd [RETURN_TYPE] long   long val1 long val2 [VARIABLES] boolean  long  sum  val1  val2  
[buglab_swap_variables]^if  (  ( val2 ^ sum )  < 0 &&  ( val1 ^ val1 )  >= 0 )  {^84^^^^^81^89^if  (  ( val1 ^ sum )  < 0 &&  ( val1 ^ val2 )  >= 0 )  {^[CLASS] FieldUtils  [METHOD] safeAdd [RETURN_TYPE] long   long val1 long val2 [VARIABLES] boolean  long  sum  val1  val2  
[buglab_swap_variables]^if  (  ( sum ^ val1 )  < 0 &&  ( val1 ^ val2 )  >= 0 )  {^84^^^^^81^89^if  (  ( val1 ^ sum )  < 0 &&  ( val1 ^ val2 )  >= 0 )  {^[CLASS] FieldUtils  [METHOD] safeAdd [RETURN_TYPE] long   long val1 long val2 [VARIABLES] boolean  long  sum  val1  val2  
[buglab_swap_variables]^if  (  ( val1 ^ val2 )  < 0 &&  ( val1 ^ sum )  >= 0 )  {^84^^^^^81^89^if  (  ( val1 ^ sum )  < 0 &&  ( val1 ^ val2 )  >= 0 )  {^[CLASS] FieldUtils  [METHOD] safeAdd [RETURN_TYPE] long   long val1 long val2 [VARIABLES] boolean  long  sum  val1  val2  
[buglab_swap_variables]^throw new ArithmeticException ( "The calculation caused an overflow: " + val2 + " + " + val1 ) ;^85^86^^^^81^89^throw new ArithmeticException ( "The calculation caused an overflow: " + val1 + " + " + val2 ) ;^[CLASS] FieldUtils  [METHOD] safeAdd [RETURN_TYPE] long   long val1 long val2 [VARIABLES] boolean  long  sum  val1  val2  
[buglab_swap_variables]^long diff = val2 - val1;^100^^^^^99^107^long diff = val1 - val2;^[CLASS] FieldUtils  [METHOD] safeSubtract [RETURN_TYPE] long   long val1 long val2 [VARIABLES] boolean  long  diff  val1  val2  
[buglab_swap_variables]^if  (  ( diff ^ val1 )  < 0 &&  ( val1 ^ val2 )  < 0 )  {^102^^^^^99^107^if  (  ( val1 ^ diff )  < 0 &&  ( val1 ^ val2 )  < 0 )  {^[CLASS] FieldUtils  [METHOD] safeSubtract [RETURN_TYPE] long   long val1 long val2 [VARIABLES] boolean  long  diff  val1  val2  
[buglab_swap_variables]^if  (  ( val1 ^ val2 )  < 0 &&  ( val1 ^ diff )  < 0 )  {^102^^^^^99^107^if  (  ( val1 ^ diff )  < 0 &&  ( val1 ^ val2 )  < 0 )  {^[CLASS] FieldUtils  [METHOD] safeSubtract [RETURN_TYPE] long   long val1 long val2 [VARIABLES] boolean  long  diff  val1  val2  
[buglab_swap_variables]^throw new ArithmeticException ( "The calculation caused an overflow: " + val2 + " - " + val1 ) ;^103^104^^^^99^107^throw new ArithmeticException ( "The calculation caused an overflow: " + val1 + " - " + val2 ) ;^[CLASS] FieldUtils  [METHOD] safeSubtract [RETURN_TYPE] long   long val1 long val2 [VARIABLES] boolean  long  diff  val1  val2  
[buglab_swap_variables]^throw new ArithmeticException ( "The calculation caused an overflow: " + val2 + " * " + val1 ) ;^121^122^^^^118^125^throw new ArithmeticException ( "The calculation caused an overflow: " + val1 + " * " + val2 ) ;^[CLASS] FieldUtils  [METHOD] safeMultiply [RETURN_TYPE] int   int val1 int val2 [VARIABLES] boolean  int  val1  val2  long  total  
[buglab_swap_variables]^long total = scalar * val1;^145^^^^^136^151^long total = val1 * scalar;^[CLASS] FieldUtils  [METHOD] safeMultiply [RETURN_TYPE] long   long val1 int scalar [VARIABLES] boolean  long  total  val1  int  scalar  
[buglab_swap_variables]^if  ( val1 / scalar != total )  {^146^^^^^136^151^if  ( total / scalar != val1 )  {^[CLASS] FieldUtils  [METHOD] safeMultiply [RETURN_TYPE] long   long val1 int scalar [VARIABLES] boolean  long  total  val1  int  scalar  
[buglab_swap_variables]^if  ( scalar / total != val1 )  {^146^^^^^136^151^if  ( total / scalar != val1 )  {^[CLASS] FieldUtils  [METHOD] safeMultiply [RETURN_TYPE] long   long val1 int scalar [VARIABLES] boolean  long  total  val1  int  scalar  
[buglab_swap_variables]^if  ( total / val1 != scalar )  {^146^^^^^136^151^if  ( total / scalar != val1 )  {^[CLASS] FieldUtils  [METHOD] safeMultiply [RETURN_TYPE] long   long val1 int scalar [VARIABLES] boolean  long  total  val1  int  scalar  
[buglab_swap_variables]^throw new ArithmeticException ( "The calculation caused an overflow: " + scalar + " * " + val1 ) ;^147^148^^^^136^151^throw new ArithmeticException ( "The calculation caused an overflow: " + val1 + " * " + scalar ) ;^[CLASS] FieldUtils  [METHOD] safeMultiply [RETURN_TYPE] long   long val1 int scalar [VARIABLES] boolean  long  total  val1  int  scalar  
[buglab_swap_variables]^long total = val2 * val1;^168^^^^^161^174^long total = val1 * val2;^[CLASS] FieldUtils  [METHOD] safeMultiply [RETURN_TYPE] long   long val1 long val2 [VARIABLES] boolean  long  total  val1  val2  
[buglab_swap_variables]^if  ( val2 / total != val1 )  {^169^^^^^161^174^if  ( total / val2 != val1 )  {^[CLASS] FieldUtils  [METHOD] safeMultiply [RETURN_TYPE] long   long val1 long val2 [VARIABLES] boolean  long  total  val1  val2  
[buglab_swap_variables]^if  ( total / val1 != val2 )  {^169^^^^^161^174^if  ( total / val2 != val1 )  {^[CLASS] FieldUtils  [METHOD] safeMultiply [RETURN_TYPE] long   long val1 long val2 [VARIABLES] boolean  long  total  val1  val2  
[buglab_swap_variables]^throw new ArithmeticException ( "The calculation caused an overflow: " + val2 + " * " + val1 ) ;^170^171^^^^161^174^throw new ArithmeticException ( "The calculation caused an overflow: " + val1 + " * " + val2 ) ;^[CLASS] FieldUtils  [METHOD] safeMultiply [RETURN_TYPE] long   long val1 long val2 [VARIABLES] boolean  long  total  val1  val2  
[buglab_swap_variables]^long val = FieldUtils.safeMultiply ( val2, val1 ) ;^199^^^^^198^201^long val = FieldUtils.safeMultiply ( val1, val2 ) ;^[CLASS] FieldUtils  [METHOD] safeMultiplyToInt [RETURN_TYPE] int   long val1 long val2 [VARIABLES] boolean  long  val  val1  val2  
[buglab_swap_variables]^long val = FieldUtils.safeMultiply (  val2 ) ;^199^^^^^198^201^long val = FieldUtils.safeMultiply ( val1, val2 ) ;^[CLASS] FieldUtils  [METHOD] safeMultiplyToInt [RETURN_TYPE] int   long val1 long val2 [VARIABLES] boolean  long  val  val1  val2  
[buglab_swap_variables]^long val = FieldUtils.safeMultiply ( val1 ) ;^199^^^^^198^201^long val = FieldUtils.safeMultiply ( val1, val2 ) ;^[CLASS] FieldUtils  [METHOD] safeMultiplyToInt [RETURN_TYPE] int   long val1 long val2 [VARIABLES] boolean  long  val  val1  val2  
[buglab_swap_variables]^if  (  ( upperBound < lowerBound )  ||  ( value > value )  )  {^214^^^^^212^219^if  (  ( value < lowerBound )  ||  ( value > upperBound )  )  {^[CLASS] FieldUtils  [METHOD] verifyValueBounds [RETURN_TYPE] void   DateTimeField field int value int lowerBound int upperBound [VARIABLES] boolean  DateTimeField  field  int  lowerBound  upperBound  value  
[buglab_swap_variables]^if  (  ( value < upperBound )  ||  ( value > lowerBound )  )  {^214^^^^^212^219^if  (  ( value < lowerBound )  ||  ( value > upperBound )  )  {^[CLASS] FieldUtils  [METHOD] verifyValueBounds [RETURN_TYPE] void   DateTimeField field int value int lowerBound int upperBound [VARIABLES] boolean  DateTimeField  field  int  lowerBound  upperBound  value  
[buglab_swap_variables]^throw new IllegalFieldValueException ( lowerBound.getType (  ) , new Integer ( value ) , new Integer ( field ) , new Integer ( upperBound )  ) ;^215^216^217^^^212^219^throw new IllegalFieldValueException ( field.getType (  ) , new Integer ( value ) , new Integer ( lowerBound ) , new Integer ( upperBound )  ) ;^[CLASS] FieldUtils  [METHOD] verifyValueBounds [RETURN_TYPE] void   DateTimeField field int value int lowerBound int upperBound [VARIABLES] boolean  DateTimeField  field  int  lowerBound  upperBound  value  
[buglab_swap_variables]^throw new IllegalFieldValueException ( field.getType (  ) , new Integer ( lowerBound ) , new Integer ( value ) , new Integer ( upperBound )  ) ;^215^216^217^^^212^219^throw new IllegalFieldValueException ( field.getType (  ) , new Integer ( value ) , new Integer ( lowerBound ) , new Integer ( upperBound )  ) ;^[CLASS] FieldUtils  [METHOD] verifyValueBounds [RETURN_TYPE] void   DateTimeField field int value int lowerBound int upperBound [VARIABLES] boolean  DateTimeField  field  int  lowerBound  upperBound  value  
[buglab_swap_variables]^throw new IllegalFieldValueException ( field.getType (  ) , new Integer ( value ) , new Integer ( upperBound ) , new Integer ( lowerBound )  ) ;^215^216^217^^^212^219^throw new IllegalFieldValueException ( field.getType (  ) , new Integer ( value ) , new Integer ( lowerBound ) , new Integer ( upperBound )  ) ;^[CLASS] FieldUtils  [METHOD] verifyValueBounds [RETURN_TYPE] void   DateTimeField field int value int lowerBound int upperBound [VARIABLES] boolean  DateTimeField  field  int  lowerBound  upperBound  value  
[buglab_swap_variables]^throw new IllegalFieldValueException ( upperBound.getType (  ) , new Integer ( value ) , new Integer ( lowerBound ) , new Integer ( field )  ) ;^215^216^217^^^212^219^throw new IllegalFieldValueException ( field.getType (  ) , new Integer ( value ) , new Integer ( lowerBound ) , new Integer ( upperBound )  ) ;^[CLASS] FieldUtils  [METHOD] verifyValueBounds [RETURN_TYPE] void   DateTimeField field int value int lowerBound int upperBound [VARIABLES] boolean  DateTimeField  field  int  lowerBound  upperBound  value  
[buglab_swap_variables]^throw new IllegalFieldValueException ( field.getType (  ) , new Integer ( upperBound ) , new Integer ( lowerBound ) , new Integer ( value )  ) ;^215^216^217^^^212^219^throw new IllegalFieldValueException ( field.getType (  ) , new Integer ( value ) , new Integer ( lowerBound ) , new Integer ( upperBound )  ) ;^[CLASS] FieldUtils  [METHOD] verifyValueBounds [RETURN_TYPE] void   DateTimeField field int value int lowerBound int upperBound [VARIABLES] boolean  DateTimeField  field  int  lowerBound  upperBound  value  
[buglab_swap_variables]^if  (  ( lowerBound < value )  ||  ( value > upperBound )  )  {^232^^^^^230^237^if  (  ( value < lowerBound )  ||  ( value > upperBound )  )  {^[CLASS] FieldUtils  [METHOD] verifyValueBounds [RETURN_TYPE] void   DateTimeFieldType fieldType int value int lowerBound int upperBound [VARIABLES] boolean  DateTimeFieldType  fieldType  int  lowerBound  upperBound  value  
[buglab_swap_variables]^throw new IllegalFieldValueException ( lowerBound, new Integer ( value ) , new Integer ( fieldType ) , new Integer ( upperBound )  ) ;^233^234^235^^^230^237^throw new IllegalFieldValueException ( fieldType, new Integer ( value ) , new Integer ( lowerBound ) , new Integer ( upperBound )  ) ;^[CLASS] FieldUtils  [METHOD] verifyValueBounds [RETURN_TYPE] void   DateTimeFieldType fieldType int value int lowerBound int upperBound [VARIABLES] boolean  DateTimeFieldType  fieldType  int  lowerBound  upperBound  value  
[buglab_swap_variables]^throw new IllegalFieldValueException (  new Integer ( value ) , new Integer ( lowerBound ) , new Integer ( upperBound )  ) ;^233^234^235^^^230^237^throw new IllegalFieldValueException ( fieldType, new Integer ( value ) , new Integer ( lowerBound ) , new Integer ( upperBound )  ) ;^[CLASS] FieldUtils  [METHOD] verifyValueBounds [RETURN_TYPE] void   DateTimeFieldType fieldType int value int lowerBound int upperBound [VARIABLES] boolean  DateTimeFieldType  fieldType  int  lowerBound  upperBound  value  
[buglab_swap_variables]^throw new IllegalFieldValueException ( fieldType, new Integer ( lowerBound ) , new Integer ( value ) , new Integer ( upperBound )  ) ;^233^234^235^^^230^237^throw new IllegalFieldValueException ( fieldType, new Integer ( value ) , new Integer ( lowerBound ) , new Integer ( upperBound )  ) ;^[CLASS] FieldUtils  [METHOD] verifyValueBounds [RETURN_TYPE] void   DateTimeFieldType fieldType int value int lowerBound int upperBound [VARIABLES] boolean  DateTimeFieldType  fieldType  int  lowerBound  upperBound  value  
[buglab_swap_variables]^throw new IllegalFieldValueException ( fieldType, new Integer ( value ) , new Integer ( upperBound ) , new Integer ( lowerBound )  ) ;^233^234^235^^^230^237^throw new IllegalFieldValueException ( fieldType, new Integer ( value ) , new Integer ( lowerBound ) , new Integer ( upperBound )  ) ;^[CLASS] FieldUtils  [METHOD] verifyValueBounds [RETURN_TYPE] void   DateTimeFieldType fieldType int value int lowerBound int upperBound [VARIABLES] boolean  DateTimeFieldType  fieldType  int  lowerBound  upperBound  value  
[buglab_swap_variables]^throw new IllegalFieldValueException ( upperBound, new Integer ( value ) , new Integer ( lowerBound ) , new Integer ( fieldType )  ) ;^233^234^235^^^230^237^throw new IllegalFieldValueException ( fieldType, new Integer ( value ) , new Integer ( lowerBound ) , new Integer ( upperBound )  ) ;^[CLASS] FieldUtils  [METHOD] verifyValueBounds [RETURN_TYPE] void   DateTimeFieldType fieldType int value int lowerBound int upperBound [VARIABLES] boolean  DateTimeFieldType  fieldType  int  lowerBound  upperBound  value  
[buglab_swap_variables]^throw new IllegalFieldValueException ( fieldType, new Integer ( upperBound ) , new Integer ( lowerBound ) , new Integer ( value )  ) ;^233^234^235^^^230^237^throw new IllegalFieldValueException ( fieldType, new Integer ( value ) , new Integer ( lowerBound ) , new Integer ( upperBound )  ) ;^[CLASS] FieldUtils  [METHOD] verifyValueBounds [RETURN_TYPE] void   DateTimeFieldType fieldType int value int lowerBound int upperBound [VARIABLES] boolean  DateTimeFieldType  fieldType  int  lowerBound  upperBound  value  
[buglab_swap_variables]^if  (  ( upperBound < lowerBound )  ||  ( value > value )  )  {^249^^^^^247^254^if  (  ( value < lowerBound )  ||  ( value > upperBound )  )  {^[CLASS] FieldUtils  [METHOD] verifyValueBounds [RETURN_TYPE] void   String fieldName int value int lowerBound int upperBound [VARIABLES] boolean  String  fieldName  int  lowerBound  upperBound  value  
[buglab_swap_variables]^if  (  ( lowerBound < value )  ||  ( value > upperBound )  )  {^249^^^^^247^254^if  (  ( value < lowerBound )  ||  ( value > upperBound )  )  {^[CLASS] FieldUtils  [METHOD] verifyValueBounds [RETURN_TYPE] void   String fieldName int value int lowerBound int upperBound [VARIABLES] boolean  String  fieldName  int  lowerBound  upperBound  value  
[buglab_swap_variables]^throw new IllegalFieldValueException ( value, new Integer ( fieldName ) , new Integer ( lowerBound ) , new Integer ( upperBound )  ) ;^250^251^252^^^247^254^throw new IllegalFieldValueException ( fieldName, new Integer ( value ) , new Integer ( lowerBound ) , new Integer ( upperBound )  ) ;^[CLASS] FieldUtils  [METHOD] verifyValueBounds [RETURN_TYPE] void   String fieldName int value int lowerBound int upperBound [VARIABLES] boolean  String  fieldName  int  lowerBound  upperBound  value  
[buglab_swap_variables]^throw new IllegalFieldValueException (  new Integer ( value ) , new Integer ( lowerBound ) , new Integer ( upperBound )  ) ;^250^251^252^^^247^254^throw new IllegalFieldValueException ( fieldName, new Integer ( value ) , new Integer ( lowerBound ) , new Integer ( upperBound )  ) ;^[CLASS] FieldUtils  [METHOD] verifyValueBounds [RETURN_TYPE] void   String fieldName int value int lowerBound int upperBound [VARIABLES] boolean  String  fieldName  int  lowerBound  upperBound  value  
[buglab_swap_variables]^throw new IllegalFieldValueException ( fieldName, new Integer ( lowerBound ) , new Integer ( value ) , new Integer ( upperBound )  ) ;^250^251^252^^^247^254^throw new IllegalFieldValueException ( fieldName, new Integer ( value ) , new Integer ( lowerBound ) , new Integer ( upperBound )  ) ;^[CLASS] FieldUtils  [METHOD] verifyValueBounds [RETURN_TYPE] void   String fieldName int value int lowerBound int upperBound [VARIABLES] boolean  String  fieldName  int  lowerBound  upperBound  value  
[buglab_swap_variables]^throw new IllegalFieldValueException ( fieldName, new Integer ( upperBound ) , new Integer ( lowerBound ) , new Integer ( value )  ) ;^250^251^252^^^247^254^throw new IllegalFieldValueException ( fieldName, new Integer ( value ) , new Integer ( lowerBound ) , new Integer ( upperBound )  ) ;^[CLASS] FieldUtils  [METHOD] verifyValueBounds [RETURN_TYPE] void   String fieldName int value int lowerBound int upperBound [VARIABLES] boolean  String  fieldName  int  lowerBound  upperBound  value  
[buglab_swap_variables]^throw new IllegalFieldValueException ( lowerBound, new Integer ( value ) , new Integer ( fieldName ) , new Integer ( upperBound )  ) ;^250^251^252^^^247^254^throw new IllegalFieldValueException ( fieldName, new Integer ( value ) , new Integer ( lowerBound ) , new Integer ( upperBound )  ) ;^[CLASS] FieldUtils  [METHOD] verifyValueBounds [RETURN_TYPE] void   String fieldName int value int lowerBound int upperBound [VARIABLES] boolean  String  fieldName  int  lowerBound  upperBound  value  
[buglab_swap_variables]^throw new IllegalFieldValueException ( fieldName, new Integer ( value ) , new Integer ( upperBound ) , new Integer ( lowerBound )  ) ;^250^251^252^^^247^254^throw new IllegalFieldValueException ( fieldName, new Integer ( value ) , new Integer ( lowerBound ) , new Integer ( upperBound )  ) ;^[CLASS] FieldUtils  [METHOD] verifyValueBounds [RETURN_TYPE] void   String fieldName int value int lowerBound int upperBound [VARIABLES] boolean  String  fieldName  int  lowerBound  upperBound  value  
[buglab_swap_variables]^return getWrappedValue ( maxValue + wrapValue, minValue, currentValue ) ;^273^^^^^271^274^return getWrappedValue ( currentValue + wrapValue, minValue, maxValue ) ;^[CLASS] FieldUtils  [METHOD] getWrappedValue [RETURN_TYPE] int   int currentValue int wrapValue int minValue int maxValue [VARIABLES] boolean  int  currentValue  maxValue  minValue  wrapValue  
[buglab_swap_variables]^return getWrappedValue ( currentValue + minValue, wrapValue, maxValue ) ;^273^^^^^271^274^return getWrappedValue ( currentValue + wrapValue, minValue, maxValue ) ;^[CLASS] FieldUtils  [METHOD] getWrappedValue [RETURN_TYPE] int   int currentValue int wrapValue int minValue int maxValue [VARIABLES] boolean  int  currentValue  maxValue  minValue  wrapValue  
[buglab_swap_variables]^return getWrappedValue ( currentValue +  minValue, maxValue ) ;^273^^^^^271^274^return getWrappedValue ( currentValue + wrapValue, minValue, maxValue ) ;^[CLASS] FieldUtils  [METHOD] getWrappedValue [RETURN_TYPE] int   int currentValue int wrapValue int minValue int maxValue [VARIABLES] boolean  int  currentValue  maxValue  minValue  wrapValue  
[buglab_swap_variables]^return getWrappedValue ( minValue + wrapValue, currentValue, maxValue ) ;^273^^^^^271^274^return getWrappedValue ( currentValue + wrapValue, minValue, maxValue ) ;^[CLASS] FieldUtils  [METHOD] getWrappedValue [RETURN_TYPE] int   int currentValue int wrapValue int minValue int maxValue [VARIABLES] boolean  int  currentValue  maxValue  minValue  wrapValue  
[buglab_swap_variables]^return getWrappedValue ( currentValue + wrapValue,  maxValue ) ;^273^^^^^271^274^return getWrappedValue ( currentValue + wrapValue, minValue, maxValue ) ;^[CLASS] FieldUtils  [METHOD] getWrappedValue [RETURN_TYPE] int   int currentValue int wrapValue int minValue int maxValue [VARIABLES] boolean  int  currentValue  maxValue  minValue  wrapValue  
[buglab_swap_variables]^return getWrappedValue ( currentValue + wrapValue, minValue ) ;^273^^^^^271^274^return getWrappedValue ( currentValue + wrapValue, minValue, maxValue ) ;^[CLASS] FieldUtils  [METHOD] getWrappedValue [RETURN_TYPE] int   int currentValue int wrapValue int minValue int maxValue [VARIABLES] boolean  int  currentValue  maxValue  minValue  wrapValue  
[buglab_swap_variables]^return getWrappedValue ( wrapValue + currentValue, minValue, maxValue ) ;^273^^^^^271^274^return getWrappedValue ( currentValue + wrapValue, minValue, maxValue ) ;^[CLASS] FieldUtils  [METHOD] getWrappedValue [RETURN_TYPE] int   int currentValue int wrapValue int minValue int maxValue [VARIABLES] boolean  int  currentValue  maxValue  minValue  wrapValue  
[buglab_swap_variables]^return getWrappedValue ( currentValue + wrapValue, maxValue, minValue ) ;^273^^^^^271^274^return getWrappedValue ( currentValue + wrapValue, minValue, maxValue ) ;^[CLASS] FieldUtils  [METHOD] getWrappedValue [RETURN_TYPE] int   int currentValue int wrapValue int minValue int maxValue [VARIABLES] boolean  int  currentValue  maxValue  minValue  wrapValue  
[buglab_swap_variables]^if  ( maxValue >= minValue )  {^289^^^^^288^306^if  ( minValue >= maxValue )  {^[CLASS] FieldUtils  [METHOD] getWrappedValue [RETURN_TYPE] int   int value int minValue int maxValue [VARIABLES] boolean  int  maxValue  minValue  remByRange  value  wrapRange  
[buglab_swap_variables]^int wrapRange = minValue - maxValue + 1;^293^^^^^288^306^int wrapRange = maxValue - minValue + 1;^[CLASS] FieldUtils  [METHOD] getWrappedValue [RETURN_TYPE] int   int value int minValue int maxValue [VARIABLES] boolean  int  maxValue  minValue  remByRange  value  wrapRange  
[buglab_swap_variables]^return  ( minValue % wrapRange )  + value;^297^^^^^288^306^return  ( value % wrapRange )  + minValue;^[CLASS] FieldUtils  [METHOD] getWrappedValue [RETURN_TYPE] int   int value int minValue int maxValue [VARIABLES] boolean  int  maxValue  minValue  remByRange  value  wrapRange  
[buglab_swap_variables]^return  ( value % minValue )  + wrapRange;^297^^^^^288^306^return  ( value % wrapRange )  + minValue;^[CLASS] FieldUtils  [METHOD] getWrappedValue [RETURN_TYPE] int   int value int minValue int maxValue [VARIABLES] boolean  int  maxValue  minValue  remByRange  value  wrapRange  
[buglab_swap_variables]^return  ( wrapRange % value )  + minValue;^297^^^^^288^306^return  ( value % wrapRange )  + minValue;^[CLASS] FieldUtils  [METHOD] getWrappedValue [RETURN_TYPE] int   int value int minValue int maxValue [VARIABLES] boolean  int  maxValue  minValue  remByRange  value  wrapRange  
[buglab_swap_variables]^return  ( remByRange - wrapRange )  + minValue;^305^^^^^288^306^return  ( wrapRange - remByRange )  + minValue;^[CLASS] FieldUtils  [METHOD] getWrappedValue [RETURN_TYPE] int   int value int minValue int maxValue [VARIABLES] boolean  int  maxValue  minValue  remByRange  value  wrapRange  
[buglab_swap_variables]^return  ( wrapRange - minValue )  + remByRange;^305^^^^^288^306^return  ( wrapRange - remByRange )  + minValue;^[CLASS] FieldUtils  [METHOD] getWrappedValue [RETURN_TYPE] int   int value int minValue int maxValue [VARIABLES] boolean  int  maxValue  minValue  remByRange  value  wrapRange  
[buglab_swap_variables]^return  ( minValue - remByRange )  + wrapRange;^305^^^^^288^306^return  ( wrapRange - remByRange )  + minValue;^[CLASS] FieldUtils  [METHOD] getWrappedValue [RETURN_TYPE] int   int value int minValue int maxValue [VARIABLES] boolean  int  maxValue  minValue  remByRange  value  wrapRange  
[buglab_swap_variables]^if  ( object2 == object1 )  {^318^^^^^317^325^if  ( object1 == object2 )  {^[CLASS] FieldUtils  [METHOD] equals [RETURN_TYPE] boolean   Object object1 Object object2 [VARIABLES] boolean  Object  object1  object2  
[buglab_swap_variables]^if  ( object2 == null || object1 == null )  {^321^^^^^317^325^if  ( object1 == null || object2 == null )  {^[CLASS] FieldUtils  [METHOD] equals [RETURN_TYPE] boolean   Object object1 Object object2 [VARIABLES] boolean  Object  object1  object2  
[buglab_swap_variables]^return object2.equals ( object1 ) ;^324^^^^^317^325^return object1.equals ( object2 ) ;^[CLASS] FieldUtils  [METHOD] equals [RETURN_TYPE] boolean   Object object1 Object object2 [VARIABLES] boolean  Object  object1  object2  