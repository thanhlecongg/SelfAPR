[buglab_swap_variables]^return new AnnotatedField ( ann, _field ) ;^47^^^^^46^48^return new AnnotatedField ( _field, ann ) ;^[CLASS] AnnotatedField Serialization  [METHOD] withAnnotations [RETURN_TYPE] AnnotatedField   AnnotationMap ann [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  boolean  AnnotationMap  ann  Class  clazz  String  name  long  serialVersionUID  
[buglab_swap_variables]^return new AnnotatedField ( _field ) ;^47^^^^^46^48^return new AnnotatedField ( _field, ann ) ;^[CLASS] AnnotatedField Serialization  [METHOD] withAnnotations [RETURN_TYPE] AnnotatedField   AnnotationMap ann [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  boolean  AnnotationMap  ann  Class  clazz  String  name  long  serialVersionUID  
[buglab_swap_variables]^return new AnnotatedField (  ann ) ;^47^^^^^46^48^return new AnnotatedField ( _field, ann ) ;^[CLASS] AnnotatedField Serialization  [METHOD] withAnnotations [RETURN_TYPE] AnnotatedField   AnnotationMap ann [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  boolean  AnnotationMap  ann  Class  clazz  String  name  long  serialVersionUID  
[buglab_swap_variables]^return  ( acls == null )  ? null : _annotations.get ( _annotations ) ;^78^^^^^76^79^return  ( _annotations == null )  ? null : _annotations.get ( acls ) ;^[CLASS] AnnotatedField Serialization  [METHOD] getAnnotation [RETURN_TYPE] <A   Class<A> acls [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  Class  acls  clazz  String  name  boolean  long  serialVersionUID  
[buglab_swap_variables]^_field.set ( value, pojo ) ;^107^^^^^104^112^_field.set ( pojo, value ) ;^[CLASS] AnnotatedField Serialization  [METHOD] setValue [RETURN_TYPE] void   Object pojo Object value [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  boolean  Class  acls  clazz  Object  pojo  value  String  name  long  serialVersionUID  IllegalAccessException  e  
[buglab_swap_variables]^_field.set (  value ) ;^107^^^^^104^112^_field.set ( pojo, value ) ;^[CLASS] AnnotatedField Serialization  [METHOD] setValue [RETURN_TYPE] void   Object pojo Object value [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  boolean  Class  acls  clazz  Object  pojo  value  String  name  long  serialVersionUID  IllegalAccessException  e  
[buglab_swap_variables]^_field.set ( pojo ) ;^107^^^^^104^112^_field.set ( pojo, value ) ;^[CLASS] AnnotatedField Serialization  [METHOD] setValue [RETURN_TYPE] void   Object pojo Object value [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  boolean  Class  acls  clazz  Object  pojo  value  String  name  long  serialVersionUID  IllegalAccessException  e  
[buglab_swap_variables]^return pojo.get ( _field ) ;^118^^^^^115^123^return _field.get ( pojo ) ;^[CLASS] AnnotatedField Serialization  [METHOD] getValue [RETURN_TYPE] Object   Object pojo [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  boolean  Class  acls  clazz  Object  pojo  String  name  long  serialVersionUID  IllegalAccessException  e  
[buglab_swap_variables]^Class<?> clazz = _serialization;^154^^^^^153^166^Class<?> clazz = _serialization.clazz;^[CLASS] AnnotatedField Serialization  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  boolean  Class  acls  clazz  String  name  long  serialVersionUID  Exception  e  
[buglab_swap_variables]^Class<?> clazz = _serialization.clazz.clazz;^154^^^^^153^166^Class<?> clazz = _serialization.clazz;^[CLASS] AnnotatedField Serialization  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  boolean  Class  acls  clazz  String  name  long  serialVersionUID  Exception  e  
[buglab_swap_variables]^Field f = _serialization.getDeclaredField ( clazz.name ) ;^156^^^^^153^166^Field f = clazz.getDeclaredField ( _serialization.name ) ;^[CLASS] AnnotatedField Serialization  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  boolean  Class  acls  clazz  String  name  long  serialVersionUID  Exception  e  
[buglab_swap_variables]^Field f = _serialization.name.getDeclaredField ( clazz ) ;^156^^^^^153^166^Field f = clazz.getDeclaredField ( _serialization.name ) ;^[CLASS] AnnotatedField Serialization  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  boolean  Class  acls  clazz  String  name  long  serialVersionUID  Exception  e  
[buglab_swap_variables]^Field f = clazz.getDeclaredField ( _serialization.name.name ) ;^156^^^^^153^166^Field f = clazz.getDeclaredField ( _serialization.name ) ;^[CLASS] AnnotatedField Serialization  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  boolean  Class  acls  clazz  String  name  long  serialVersionUID  Exception  e  
[buglab_swap_variables]^Field f = clazz.getDeclaredField ( _serialization ) ;^156^^^^^153^166^Field f = clazz.getDeclaredField ( _serialization.name ) ;^[CLASS] AnnotatedField Serialization  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  boolean  Class  acls  clazz  String  name  long  serialVersionUID  Exception  e  