[REPLACE]^private static final long serialVersionUID ;^16^^^^^^^[REPLACE] private static final long serialVersionUID = 1L;^ [CLASS] InvalidFormatException  
[REPLACE]^protected  Class<?> _targetType;^28^^^^^^^[REPLACE] protected final Class<?> _targetType;^ [CLASS] InvalidFormatException  
[REPLACE]^super ( msg, loc ) ;^39^^^^^36^42^[REPLACE] super ( msg ) ;^[METHOD] <init> [TYPE] Class) [PARAMETER] String msg Object value Class<?> targetType [CLASS] InvalidFormatException   [TYPE]  Object _value  value  [TYPE]  Class _targetType  targetType  [TYPE]  String msg  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 
[REPLACE]^_value =  null;^40^^^^^36^42^[REPLACE] _value = value;^[METHOD] <init> [TYPE] Class) [PARAMETER] String msg Object value Class<?> targetType [CLASS] InvalidFormatException   [TYPE]  Object _value  value  [TYPE]  Class _targetType  targetType  [TYPE]  String msg  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 
[REPLACE]^_targetType =  0;^41^^^^^36^42^[REPLACE] _targetType = targetType;^[METHOD] <init> [TYPE] Class) [PARAMETER] String msg Object value Class<?> targetType [CLASS] InvalidFormatException   [TYPE]  Object _value  value  [TYPE]  Class _targetType  targetType  [TYPE]  String msg  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 
[REPLACE]^super ( msg ) ;^47^^^^^44^50^[REPLACE] super ( msg, loc ) ;^[METHOD] <init> [TYPE] Class) [PARAMETER] String msg JsonLocation loc Object value Class<?> targetType [CLASS] InvalidFormatException   [TYPE]  Object _value  value  [TYPE]  Class _targetType  targetType  [TYPE]  JsonLocation loc  [TYPE]  String msg  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 
[REPLACE]^_value =  null;^48^^^^^44^50^[REPLACE] _value = value;^[METHOD] <init> [TYPE] Class) [PARAMETER] String msg JsonLocation loc Object value Class<?> targetType [CLASS] InvalidFormatException   [TYPE]  Object _value  value  [TYPE]  Class _targetType  targetType  [TYPE]  JsonLocation loc  [TYPE]  String msg  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 
[REPLACE]^_targetType =  false;^49^^^^^44^50^[REPLACE] _targetType = targetType;^[METHOD] <init> [TYPE] Class) [PARAMETER] String msg JsonLocation loc Object value Class<?> targetType [CLASS] InvalidFormatException   [TYPE]  Object _value  value  [TYPE]  Class _targetType  targetType  [TYPE]  JsonLocation loc  [TYPE]  String msg  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 
[REPLACE]^return new InvalidFormatException ( msg, jp.getTokenLocation (  ) , value, this ) ;^55^56^^^^52^57^[REPLACE] return new InvalidFormatException ( msg, jp.getTokenLocation (  ) , value, targetType ) ;^[METHOD] from [TYPE] InvalidFormatException [PARAMETER] JsonParser jp String msg Object value Class<?> targetType [CLASS] InvalidFormatException   [TYPE]  Object _value  value  [TYPE]  Class _targetType  targetType  [TYPE]  String msg  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  JsonParser jp 
[REPLACE]^return _targetType;^72^^^^^71^73^[REPLACE] return _value;^[METHOD] getValue [TYPE] Object [PARAMETER] [CLASS] InvalidFormatException   [TYPE]  Object _value  value  [TYPE]  Class _targetType  targetType  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 
[REPLACE]^return _value;^82^^^^^81^83^[REPLACE] return _targetType;^[METHOD] getTargetType [TYPE] Class [PARAMETER] [CLASS] InvalidFormatException   [TYPE]  Object _value  value  [TYPE]  Class _targetType  targetType  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 