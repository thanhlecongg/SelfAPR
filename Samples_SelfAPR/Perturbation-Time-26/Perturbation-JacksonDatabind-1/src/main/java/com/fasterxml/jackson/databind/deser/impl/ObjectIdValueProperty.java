[P1_Replace_Type]^private static final  short  serialVersionUID = 1L;^21^^^^^16^26^private static final long serialVersionUID = 1L;^[CLASS] ObjectIdValueProperty   [VARIABLES] 
[P8_Replace_Mix]^private static final  int  serialVersionUID = 1;^21^^^^^16^26^private static final long serialVersionUID = 1L;^[CLASS] ObjectIdValueProperty   [VARIABLES] 
[P8_Replace_Mix]^private final ObjectIdReader _objectIdReader;^23^^^^^18^28^protected final ObjectIdReader _objectIdReader;^[CLASS] ObjectIdValueProperty   [VARIABLES] 
[P3_Replace_Literal]^this ( objectIdReader, false ) ;^27^^^^^26^28^this ( objectIdReader, true ) ;^[CLASS] ObjectIdValueProperty  [METHOD] <init> [RETURN_TYPE] ObjectIdReader)   ObjectIdReader objectIdReader [VARIABLES] long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  boolean  
[P5_Replace_Variable]^this ( _objectIdReader, true ) ;^27^^^^^26^28^this ( objectIdReader, true ) ;^[CLASS] ObjectIdValueProperty  [METHOD] <init> [RETURN_TYPE] ObjectIdReader)   ObjectIdReader objectIdReader [VARIABLES] long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  boolean  
[P14_Delete_Statement]^^27^^^^^26^28^this ( objectIdReader, true ) ;^[CLASS] ObjectIdValueProperty  [METHOD] <init> [RETURN_TYPE] ObjectIdReader)   ObjectIdReader objectIdReader [VARIABLES] long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  boolean  
[P5_Replace_Variable]^super ( objectIdReader.propertyName.idType, null, null, null, isRequired ) ;^33^34^^^^30^37^super ( objectIdReader.propertyName, objectIdReader.idType, null, null, null, isRequired ) ;^[CLASS] ObjectIdValueProperty  [METHOD] <init> [RETURN_TYPE] ObjectIdReader,boolean)   ObjectIdReader objectIdReader boolean isRequired [VARIABLES] long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  boolean  isRequired  
[P5_Replace_Variable]^super ( objectIdReader.propertyName, objectIdReader.idType, null, null, null ) ;^33^34^^^^30^37^super ( objectIdReader.propertyName, objectIdReader.idType, null, null, null, isRequired ) ;^[CLASS] ObjectIdValueProperty  [METHOD] <init> [RETURN_TYPE] ObjectIdReader,boolean)   ObjectIdReader objectIdReader boolean isRequired [VARIABLES] long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  boolean  isRequired  
[P5_Replace_Variable]^super (  objectIdReader.idType, null, null, null, isRequired ) ;^33^34^^^^30^37^super ( objectIdReader.propertyName, objectIdReader.idType, null, null, null, isRequired ) ;^[CLASS] ObjectIdValueProperty  [METHOD] <init> [RETURN_TYPE] ObjectIdReader,boolean)   ObjectIdReader objectIdReader boolean isRequired [VARIABLES] long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  boolean  isRequired  
[P5_Replace_Variable]^super ( objectIdReader.propertyName,  null, null, null, isRequired ) ;^33^34^^^^30^37^super ( objectIdReader.propertyName, objectIdReader.idType, null, null, null, isRequired ) ;^[CLASS] ObjectIdValueProperty  [METHOD] <init> [RETURN_TYPE] ObjectIdReader,boolean)   ObjectIdReader objectIdReader boolean isRequired [VARIABLES] long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  boolean  isRequired  
[P14_Delete_Statement]^^33^34^^^^30^37^super ( objectIdReader.propertyName, objectIdReader.idType, null, null, null, isRequired ) ;^[CLASS] ObjectIdValueProperty  [METHOD] <init> [RETURN_TYPE] ObjectIdReader,boolean)   ObjectIdReader objectIdReader boolean isRequired [VARIABLES] long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  boolean  isRequired  
[P5_Replace_Variable]^_objectIdReader = _objectIdReader;^35^^^^^30^37^_objectIdReader = objectIdReader;^[CLASS] ObjectIdValueProperty  [METHOD] <init> [RETURN_TYPE] ObjectIdReader,boolean)   ObjectIdReader objectIdReader boolean isRequired [VARIABLES] long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  boolean  isRequired  
[P8_Replace_Mix]^_objectIdReader =  null;^35^^^^^30^37^_objectIdReader = objectIdReader;^[CLASS] ObjectIdValueProperty  [METHOD] <init> [RETURN_TYPE] ObjectIdReader,boolean)   ObjectIdReader objectIdReader boolean isRequired [VARIABLES] long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  boolean  isRequired  
[P11_Insert_Donor_Statement]^_objectIdReader = src._objectIdReader;_objectIdReader = objectIdReader;^35^^^^^30^37^_objectIdReader = objectIdReader;^[CLASS] ObjectIdValueProperty  [METHOD] <init> [RETURN_TYPE] ObjectIdReader,boolean)   ObjectIdReader objectIdReader boolean isRequired [VARIABLES] long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  boolean  isRequired  
[P8_Replace_Mix]^_valueDeserializer =  _objectIdReader.deserializer;^36^^^^^30^37^_valueDeserializer = objectIdReader.deserializer;^[CLASS] ObjectIdValueProperty  [METHOD] <init> [RETURN_TYPE] ObjectIdReader,boolean)   ObjectIdReader objectIdReader boolean isRequired [VARIABLES] long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  boolean  isRequired  
[P5_Replace_Variable]^super (  deser ) ;^41^^^^^39^43^super ( src, deser ) ;^[CLASS] ObjectIdValueProperty  [METHOD] <init> [RETURN_TYPE] JsonDeserializer)   ObjectIdValueProperty src JsonDeserializer<?> deser [VARIABLES] ObjectIdValueProperty  src  boolean  JsonDeserializer  deser  long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  
[P5_Replace_Variable]^super ( src ) ;^41^^^^^39^43^super ( src, deser ) ;^[CLASS] ObjectIdValueProperty  [METHOD] <init> [RETURN_TYPE] JsonDeserializer)   ObjectIdValueProperty src JsonDeserializer<?> deser [VARIABLES] ObjectIdValueProperty  src  boolean  JsonDeserializer  deser  long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  
[P5_Replace_Variable]^super ( deser, src ) ;^41^^^^^39^43^super ( src, deser ) ;^[CLASS] ObjectIdValueProperty  [METHOD] <init> [RETURN_TYPE] JsonDeserializer)   ObjectIdValueProperty src JsonDeserializer<?> deser [VARIABLES] ObjectIdValueProperty  src  boolean  JsonDeserializer  deser  long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  
[P14_Delete_Statement]^^41^42^^^^39^43^super ( src, deser ) ; _objectIdReader = src._objectIdReader;^[CLASS] ObjectIdValueProperty  [METHOD] <init> [RETURN_TYPE] JsonDeserializer)   ObjectIdValueProperty src JsonDeserializer<?> deser [VARIABLES] ObjectIdValueProperty  src  boolean  JsonDeserializer  deser  long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  
[P11_Insert_Donor_Statement]^super ( src, newName ) ;super ( src, deser ) ;^41^^^^^39^43^super ( src, deser ) ;^[CLASS] ObjectIdValueProperty  [METHOD] <init> [RETURN_TYPE] JsonDeserializer)   ObjectIdValueProperty src JsonDeserializer<?> deser [VARIABLES] ObjectIdValueProperty  src  boolean  JsonDeserializer  deser  long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  
[P5_Replace_Variable]^_objectIdReader = objectIdReader;^42^^^^^39^43^_objectIdReader = src._objectIdReader;^[CLASS] ObjectIdValueProperty  [METHOD] <init> [RETURN_TYPE] JsonDeserializer)   ObjectIdValueProperty src JsonDeserializer<?> deser [VARIABLES] ObjectIdValueProperty  src  boolean  JsonDeserializer  deser  long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  
[P5_Replace_Variable]^_objectIdReader = src._objectIdReader._objectIdReader;^42^^^^^39^43^_objectIdReader = src._objectIdReader;^[CLASS] ObjectIdValueProperty  [METHOD] <init> [RETURN_TYPE] JsonDeserializer)   ObjectIdValueProperty src JsonDeserializer<?> deser [VARIABLES] ObjectIdValueProperty  src  boolean  JsonDeserializer  deser  long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  
[P5_Replace_Variable]^_objectIdReader = src;^42^^^^^39^43^_objectIdReader = src._objectIdReader;^[CLASS] ObjectIdValueProperty  [METHOD] <init> [RETURN_TYPE] JsonDeserializer)   ObjectIdValueProperty src JsonDeserializer<?> deser [VARIABLES] ObjectIdValueProperty  src  boolean  JsonDeserializer  deser  long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  
[P8_Replace_Mix]^_objectIdReader =  null._objectIdReader;^42^^^^^39^43^_objectIdReader = src._objectIdReader;^[CLASS] ObjectIdValueProperty  [METHOD] <init> [RETURN_TYPE] JsonDeserializer)   ObjectIdValueProperty src JsonDeserializer<?> deser [VARIABLES] ObjectIdValueProperty  src  boolean  JsonDeserializer  deser  long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  
[P11_Insert_Donor_Statement]^_objectIdReader = objectIdReader;_objectIdReader = src._objectIdReader;^42^^^^^39^43^_objectIdReader = src._objectIdReader;^[CLASS] ObjectIdValueProperty  [METHOD] <init> [RETURN_TYPE] JsonDeserializer)   ObjectIdValueProperty src JsonDeserializer<?> deser [VARIABLES] ObjectIdValueProperty  src  boolean  JsonDeserializer  deser  long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  
[P5_Replace_Variable]^super (  newName ) ;^46^^^^^45^48^super ( src, newName ) ;^[CLASS] ObjectIdValueProperty  [METHOD] <init> [RETURN_TYPE] String)   ObjectIdValueProperty src String newName [VARIABLES] ObjectIdValueProperty  src  String  newName  boolean  long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  
[P5_Replace_Variable]^super ( src ) ;^46^^^^^45^48^super ( src, newName ) ;^[CLASS] ObjectIdValueProperty  [METHOD] <init> [RETURN_TYPE] String)   ObjectIdValueProperty src String newName [VARIABLES] ObjectIdValueProperty  src  String  newName  boolean  long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  
[P5_Replace_Variable]^super ( newName, src ) ;^46^^^^^45^48^super ( src, newName ) ;^[CLASS] ObjectIdValueProperty  [METHOD] <init> [RETURN_TYPE] String)   ObjectIdValueProperty src String newName [VARIABLES] ObjectIdValueProperty  src  String  newName  boolean  long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  
[P14_Delete_Statement]^^46^^^^^45^48^super ( src, newName ) ;^[CLASS] ObjectIdValueProperty  [METHOD] <init> [RETURN_TYPE] String)   ObjectIdValueProperty src String newName [VARIABLES] ObjectIdValueProperty  src  String  newName  boolean  long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  
[P11_Insert_Donor_Statement]^super ( src, deser ) ;super ( src, newName ) ;^46^^^^^45^48^super ( src, newName ) ;^[CLASS] ObjectIdValueProperty  [METHOD] <init> [RETURN_TYPE] String)   ObjectIdValueProperty src String newName [VARIABLES] ObjectIdValueProperty  src  String  newName  boolean  long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  
[P5_Replace_Variable]^_objectIdReader = objectIdReader;^47^^^^^45^48^_objectIdReader = src._objectIdReader;^[CLASS] ObjectIdValueProperty  [METHOD] <init> [RETURN_TYPE] String)   ObjectIdValueProperty src String newName [VARIABLES] ObjectIdValueProperty  src  String  newName  boolean  long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  
[P5_Replace_Variable]^_objectIdReader = src._objectIdReader._objectIdReader;^47^^^^^45^48^_objectIdReader = src._objectIdReader;^[CLASS] ObjectIdValueProperty  [METHOD] <init> [RETURN_TYPE] String)   ObjectIdValueProperty src String newName [VARIABLES] ObjectIdValueProperty  src  String  newName  boolean  long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  
[P5_Replace_Variable]^_objectIdReader = src;^47^^^^^45^48^_objectIdReader = src._objectIdReader;^[CLASS] ObjectIdValueProperty  [METHOD] <init> [RETURN_TYPE] String)   ObjectIdValueProperty src String newName [VARIABLES] ObjectIdValueProperty  src  String  newName  boolean  long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  
[P8_Replace_Mix]^_objectIdReader =  null._objectIdReader;^47^^^^^45^48^_objectIdReader = src._objectIdReader;^[CLASS] ObjectIdValueProperty  [METHOD] <init> [RETURN_TYPE] String)   ObjectIdValueProperty src String newName [VARIABLES] ObjectIdValueProperty  src  String  newName  boolean  long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  
[P11_Insert_Donor_Statement]^_objectIdReader = objectIdReader;_objectIdReader = src._objectIdReader;^47^^^^^45^48^_objectIdReader = src._objectIdReader;^[CLASS] ObjectIdValueProperty  [METHOD] <init> [RETURN_TYPE] String)   ObjectIdValueProperty src String newName [VARIABLES] ObjectIdValueProperty  src  String  newName  boolean  long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  
[P4_Replace_Constructor]^return return  new ObjectIdValueProperty ( this, deser )  ;^52^^^^^51^53^return new ObjectIdValueProperty ( this, newName ) ;^[CLASS] ObjectIdValueProperty  [METHOD] withName [RETURN_TYPE] ObjectIdValueProperty   String newName [VARIABLES] String  newName  boolean  long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  
[P8_Replace_Mix]^return  new ObjectIdValueProperty ( this, deser )  ;^52^^^^^51^53^return new ObjectIdValueProperty ( this, newName ) ;^[CLASS] ObjectIdValueProperty  [METHOD] withName [RETURN_TYPE] ObjectIdValueProperty   String newName [VARIABLES] String  newName  boolean  long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  
[P4_Replace_Constructor]^return return  new ObjectIdValueProperty ( this, newName )  ;^57^^^^^56^58^return new ObjectIdValueProperty ( this, deser ) ;^[CLASS] ObjectIdValueProperty  [METHOD] withValueDeserializer [RETURN_TYPE] ObjectIdValueProperty   JsonDeserializer<?> deser [VARIABLES] boolean  JsonDeserializer  deser  long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  
[P8_Replace_Mix]^return  new ObjectIdValueProperty ( this, newName )  ;^57^^^^^56^58^return new ObjectIdValueProperty ( this, deser ) ;^[CLASS] ObjectIdValueProperty  [METHOD] withValueDeserializer [RETURN_TYPE] ObjectIdValueProperty   JsonDeserializer<?> deser [VARIABLES] boolean  JsonDeserializer  deser  long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  
[P8_Replace_Mix]^return false;^64^^^^^63^65^return null;^[CLASS] ObjectIdValueProperty  [METHOD] getAnnotation [RETURN_TYPE] <A   Class<A> acls [VARIABLES] Class  acls  boolean  long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  
[P8_Replace_Mix]^@Override public AnnotatedMember getMember (  )  {  return true; }^67^^^^^62^72^@Override public AnnotatedMember getMember (  )  {  return null; }^[CLASS] ObjectIdValueProperty  [METHOD] getMember [RETURN_TYPE] AnnotatedMember   [VARIABLES] long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  boolean  
[P5_Replace_Variable]^deserializeSetAndReturn (  ctxt, instance ) ;^80^^^^^76^81^deserializeSetAndReturn ( jp, ctxt, instance ) ;^[CLASS] ObjectIdValueProperty  [METHOD] deserializeAndSet [RETURN_TYPE] void   JsonParser jp DeserializationContext ctxt Object instance [VARIABLES] DeserializationContext  ctxt  Object  instance  boolean  long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  JsonParser  jp  
[P5_Replace_Variable]^deserializeSetAndReturn ( jp,  instance ) ;^80^^^^^76^81^deserializeSetAndReturn ( jp, ctxt, instance ) ;^[CLASS] ObjectIdValueProperty  [METHOD] deserializeAndSet [RETURN_TYPE] void   JsonParser jp DeserializationContext ctxt Object instance [VARIABLES] DeserializationContext  ctxt  Object  instance  boolean  long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  JsonParser  jp  
[P5_Replace_Variable]^deserializeSetAndReturn ( jp, ctxt ) ;^80^^^^^76^81^deserializeSetAndReturn ( jp, ctxt, instance ) ;^[CLASS] ObjectIdValueProperty  [METHOD] deserializeAndSet [RETURN_TYPE] void   JsonParser jp DeserializationContext ctxt Object instance [VARIABLES] DeserializationContext  ctxt  Object  instance  boolean  long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  JsonParser  jp  
[P5_Replace_Variable]^deserializeSetAndReturn ( instance, ctxt, jp ) ;^80^^^^^76^81^deserializeSetAndReturn ( jp, ctxt, instance ) ;^[CLASS] ObjectIdValueProperty  [METHOD] deserializeAndSet [RETURN_TYPE] void   JsonParser jp DeserializationContext ctxt Object instance [VARIABLES] DeserializationContext  ctxt  Object  instance  boolean  long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  JsonParser  jp  
[P5_Replace_Variable]^deserializeSetAndReturn ( jp, instance, ctxt ) ;^80^^^^^76^81^deserializeSetAndReturn ( jp, ctxt, instance ) ;^[CLASS] ObjectIdValueProperty  [METHOD] deserializeAndSet [RETURN_TYPE] void   JsonParser jp DeserializationContext ctxt Object instance [VARIABLES] DeserializationContext  ctxt  Object  instance  boolean  long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  JsonParser  jp  
[P7_Replace_Invocation]^deserializeAndSet ( jp, ctxt, instance ) ;^80^^^^^76^81^deserializeSetAndReturn ( jp, ctxt, instance ) ;^[CLASS] ObjectIdValueProperty  [METHOD] deserializeAndSet [RETURN_TYPE] void   JsonParser jp DeserializationContext ctxt Object instance [VARIABLES] DeserializationContext  ctxt  Object  instance  boolean  long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  JsonParser  jp  
[P14_Delete_Statement]^^80^^^^^76^81^deserializeSetAndReturn ( jp, ctxt, instance ) ;^[CLASS] ObjectIdValueProperty  [METHOD] deserializeAndSet [RETURN_TYPE] void   JsonParser jp DeserializationContext ctxt Object instance [VARIABLES] DeserializationContext  ctxt  Object  instance  boolean  long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  JsonParser  jp  
[P5_Replace_Variable]^Object id = _valueDeserializer.deserialize (  ctxt ) ;^89^^^^^84^98^Object id = _valueDeserializer.deserialize ( jp, ctxt ) ;^[CLASS] ObjectIdValueProperty  [METHOD] deserializeSetAndReturn [RETURN_TYPE] Object   JsonParser jp DeserializationContext ctxt Object instance [VARIABLES] ReadableObjectId  roid  boolean  SettableBeanProperty  idProp  ObjectIdReader  _objectIdReader  objectIdReader  DeserializationContext  ctxt  Object  id  instance  long  serialVersionUID  JsonParser  jp  
[P5_Replace_Variable]^Object id = _valueDeserializer.deserialize ( jp ) ;^89^^^^^84^98^Object id = _valueDeserializer.deserialize ( jp, ctxt ) ;^[CLASS] ObjectIdValueProperty  [METHOD] deserializeSetAndReturn [RETURN_TYPE] Object   JsonParser jp DeserializationContext ctxt Object instance [VARIABLES] ReadableObjectId  roid  boolean  SettableBeanProperty  idProp  ObjectIdReader  _objectIdReader  objectIdReader  DeserializationContext  ctxt  Object  id  instance  long  serialVersionUID  JsonParser  jp  
[P5_Replace_Variable]^Object id = jp.deserialize ( _valueDeserializer, ctxt ) ;^89^^^^^84^98^Object id = _valueDeserializer.deserialize ( jp, ctxt ) ;^[CLASS] ObjectIdValueProperty  [METHOD] deserializeSetAndReturn [RETURN_TYPE] Object   JsonParser jp DeserializationContext ctxt Object instance [VARIABLES] ReadableObjectId  roid  boolean  SettableBeanProperty  idProp  ObjectIdReader  _objectIdReader  objectIdReader  DeserializationContext  ctxt  Object  id  instance  long  serialVersionUID  JsonParser  jp  
[P5_Replace_Variable]^Object id = _valueDeserializer.deserialize ( ctxt, jp ) ;^89^^^^^84^98^Object id = _valueDeserializer.deserialize ( jp, ctxt ) ;^[CLASS] ObjectIdValueProperty  [METHOD] deserializeSetAndReturn [RETURN_TYPE] Object   JsonParser jp DeserializationContext ctxt Object instance [VARIABLES] ReadableObjectId  roid  boolean  SettableBeanProperty  idProp  ObjectIdReader  _objectIdReader  objectIdReader  DeserializationContext  ctxt  Object  id  instance  long  serialVersionUID  JsonParser  jp  
[P8_Replace_Mix]^Object id = null.deserialize ( jp, ctxt ) ;^89^^^^^84^98^Object id = _valueDeserializer.deserialize ( jp, ctxt ) ;^[CLASS] ObjectIdValueProperty  [METHOD] deserializeSetAndReturn [RETURN_TYPE] Object   JsonParser jp DeserializationContext ctxt Object instance [VARIABLES] ReadableObjectId  roid  boolean  SettableBeanProperty  idProp  ObjectIdReader  _objectIdReader  objectIdReader  DeserializationContext  ctxt  Object  id  instance  long  serialVersionUID  JsonParser  jp  
[P5_Replace_Variable]^Object id = ctxt.deserialize ( jp, _valueDeserializer ) ;^89^^^^^84^98^Object id = _valueDeserializer.deserialize ( jp, ctxt ) ;^[CLASS] ObjectIdValueProperty  [METHOD] deserializeSetAndReturn [RETURN_TYPE] Object   JsonParser jp DeserializationContext ctxt Object instance [VARIABLES] ReadableObjectId  roid  boolean  SettableBeanProperty  idProp  ObjectIdReader  _objectIdReader  objectIdReader  DeserializationContext  ctxt  Object  id  instance  long  serialVersionUID  JsonParser  jp  
[P14_Delete_Statement]^^89^^^^^84^98^Object id = _valueDeserializer.deserialize ( jp, ctxt ) ;^[CLASS] ObjectIdValueProperty  [METHOD] deserializeSetAndReturn [RETURN_TYPE] Object   JsonParser jp DeserializationContext ctxt Object instance [VARIABLES] ReadableObjectId  roid  boolean  SettableBeanProperty  idProp  ObjectIdReader  _objectIdReader  objectIdReader  DeserializationContext  ctxt  Object  id  instance  long  serialVersionUID  JsonParser  jp  
[P5_Replace_Variable]^ReadableObjectId roid = ctxt.findObjectId (  _objectIdReader.generator ) ;^90^^^^^84^98^ReadableObjectId roid = ctxt.findObjectId ( id, _objectIdReader.generator ) ;^[CLASS] ObjectIdValueProperty  [METHOD] deserializeSetAndReturn [RETURN_TYPE] Object   JsonParser jp DeserializationContext ctxt Object instance [VARIABLES] ReadableObjectId  roid  boolean  SettableBeanProperty  idProp  ObjectIdReader  _objectIdReader  objectIdReader  DeserializationContext  ctxt  Object  id  instance  long  serialVersionUID  JsonParser  jp  
[P5_Replace_Variable]^ReadableObjectId roid = ctxt.findObjectId ( id ) ;^90^^^^^84^98^ReadableObjectId roid = ctxt.findObjectId ( id, _objectIdReader.generator ) ;^[CLASS] ObjectIdValueProperty  [METHOD] deserializeSetAndReturn [RETURN_TYPE] Object   JsonParser jp DeserializationContext ctxt Object instance [VARIABLES] ReadableObjectId  roid  boolean  SettableBeanProperty  idProp  ObjectIdReader  _objectIdReader  objectIdReader  DeserializationContext  ctxt  Object  id  instance  long  serialVersionUID  JsonParser  jp  
[P5_Replace_Variable]^ReadableObjectId roid = ctxt.findObjectId ( id.generator ) ;^90^^^^^84^98^ReadableObjectId roid = ctxt.findObjectId ( id, _objectIdReader.generator ) ;^[CLASS] ObjectIdValueProperty  [METHOD] deserializeSetAndReturn [RETURN_TYPE] Object   JsonParser jp DeserializationContext ctxt Object instance [VARIABLES] ReadableObjectId  roid  boolean  SettableBeanProperty  idProp  ObjectIdReader  _objectIdReader  objectIdReader  DeserializationContext  ctxt  Object  id  instance  long  serialVersionUID  JsonParser  jp  
[P14_Delete_Statement]^^90^91^^^^84^98^ReadableObjectId roid = ctxt.findObjectId ( id, _objectIdReader.generator ) ; roid.bindItem ( instance ) ;^[CLASS] ObjectIdValueProperty  [METHOD] deserializeSetAndReturn [RETURN_TYPE] Object   JsonParser jp DeserializationContext ctxt Object instance [VARIABLES] ReadableObjectId  roid  boolean  SettableBeanProperty  idProp  ObjectIdReader  _objectIdReader  objectIdReader  DeserializationContext  ctxt  Object  id  instance  long  serialVersionUID  JsonParser  jp  
[P5_Replace_Variable]^roid.bindItem ( id ) ;^91^^^^^84^98^roid.bindItem ( instance ) ;^[CLASS] ObjectIdValueProperty  [METHOD] deserializeSetAndReturn [RETURN_TYPE] Object   JsonParser jp DeserializationContext ctxt Object instance [VARIABLES] ReadableObjectId  roid  boolean  SettableBeanProperty  idProp  ObjectIdReader  _objectIdReader  objectIdReader  DeserializationContext  ctxt  Object  id  instance  long  serialVersionUID  JsonParser  jp  
[P14_Delete_Statement]^^91^^^^^84^98^roid.bindItem ( instance ) ;^[CLASS] ObjectIdValueProperty  [METHOD] deserializeSetAndReturn [RETURN_TYPE] Object   JsonParser jp DeserializationContext ctxt Object instance [VARIABLES] ReadableObjectId  roid  boolean  SettableBeanProperty  idProp  ObjectIdReader  _objectIdReader  objectIdReader  DeserializationContext  ctxt  Object  id  instance  long  serialVersionUID  JsonParser  jp  
[P2_Replace_Operator]^if  ( idProp == null )  {^94^^^^^84^98^if  ( idProp != null )  {^[CLASS] ObjectIdValueProperty  [METHOD] deserializeSetAndReturn [RETURN_TYPE] Object   JsonParser jp DeserializationContext ctxt Object instance [VARIABLES] ReadableObjectId  roid  boolean  SettableBeanProperty  idProp  ObjectIdReader  _objectIdReader  objectIdReader  DeserializationContext  ctxt  Object  id  instance  long  serialVersionUID  JsonParser  jp  
[P5_Replace_Variable]^if  ( id != null )  {^94^^^^^84^98^if  ( idProp != null )  {^[CLASS] ObjectIdValueProperty  [METHOD] deserializeSetAndReturn [RETURN_TYPE] Object   JsonParser jp DeserializationContext ctxt Object instance [VARIABLES] ReadableObjectId  roid  boolean  SettableBeanProperty  idProp  ObjectIdReader  _objectIdReader  objectIdReader  DeserializationContext  ctxt  Object  id  instance  long  serialVersionUID  JsonParser  jp  
[P8_Replace_Mix]^if  ( idProp != false )  {^94^^^^^84^98^if  ( idProp != null )  {^[CLASS] ObjectIdValueProperty  [METHOD] deserializeSetAndReturn [RETURN_TYPE] Object   JsonParser jp DeserializationContext ctxt Object instance [VARIABLES] ReadableObjectId  roid  boolean  SettableBeanProperty  idProp  ObjectIdReader  _objectIdReader  objectIdReader  DeserializationContext  ctxt  Object  id  instance  long  serialVersionUID  JsonParser  jp  
[P15_Unwrap_Block]^return idProp.setAndReturn(instance, id);^94^95^96^^^84^98^if  ( idProp != null )  { return idProp.setAndReturn ( instance, id ) ; }^[CLASS] ObjectIdValueProperty  [METHOD] deserializeSetAndReturn [RETURN_TYPE] Object   JsonParser jp DeserializationContext ctxt Object instance [VARIABLES] ReadableObjectId  roid  boolean  SettableBeanProperty  idProp  ObjectIdReader  _objectIdReader  objectIdReader  DeserializationContext  ctxt  Object  id  instance  long  serialVersionUID  JsonParser  jp  
[P16_Remove_Block]^^94^95^96^^^84^98^if  ( idProp != null )  { return idProp.setAndReturn ( instance, id ) ; }^[CLASS] ObjectIdValueProperty  [METHOD] deserializeSetAndReturn [RETURN_TYPE] Object   JsonParser jp DeserializationContext ctxt Object instance [VARIABLES] ReadableObjectId  roid  boolean  SettableBeanProperty  idProp  ObjectIdReader  _objectIdReader  objectIdReader  DeserializationContext  ctxt  Object  id  instance  long  serialVersionUID  JsonParser  jp  
[P5_Replace_Variable]^return idProp.setAndReturn (  id ) ;^95^^^^^84^98^return idProp.setAndReturn ( instance, id ) ;^[CLASS] ObjectIdValueProperty  [METHOD] deserializeSetAndReturn [RETURN_TYPE] Object   JsonParser jp DeserializationContext ctxt Object instance [VARIABLES] ReadableObjectId  roid  boolean  SettableBeanProperty  idProp  ObjectIdReader  _objectIdReader  objectIdReader  DeserializationContext  ctxt  Object  id  instance  long  serialVersionUID  JsonParser  jp  
[P5_Replace_Variable]^return idProp.setAndReturn ( instance ) ;^95^^^^^84^98^return idProp.setAndReturn ( instance, id ) ;^[CLASS] ObjectIdValueProperty  [METHOD] deserializeSetAndReturn [RETURN_TYPE] Object   JsonParser jp DeserializationContext ctxt Object instance [VARIABLES] ReadableObjectId  roid  boolean  SettableBeanProperty  idProp  ObjectIdReader  _objectIdReader  objectIdReader  DeserializationContext  ctxt  Object  id  instance  long  serialVersionUID  JsonParser  jp  
[P5_Replace_Variable]^return instance.setAndReturn ( idProp, id ) ;^95^^^^^84^98^return idProp.setAndReturn ( instance, id ) ;^[CLASS] ObjectIdValueProperty  [METHOD] deserializeSetAndReturn [RETURN_TYPE] Object   JsonParser jp DeserializationContext ctxt Object instance [VARIABLES] ReadableObjectId  roid  boolean  SettableBeanProperty  idProp  ObjectIdReader  _objectIdReader  objectIdReader  DeserializationContext  ctxt  Object  id  instance  long  serialVersionUID  JsonParser  jp  
[P5_Replace_Variable]^return instanceProp.setAndReturn ( id, id ) ;^95^^^^^84^98^return idProp.setAndReturn ( instance, id ) ;^[CLASS] ObjectIdValueProperty  [METHOD] deserializeSetAndReturn [RETURN_TYPE] Object   JsonParser jp DeserializationContext ctxt Object instance [VARIABLES] ReadableObjectId  roid  boolean  SettableBeanProperty  idProp  ObjectIdReader  _objectIdReader  objectIdReader  DeserializationContext  ctxt  Object  id  instance  long  serialVersionUID  JsonParser  jp  
[P13_Insert_Block]^if  ( idProp != null )  {     return idProp.setAndReturn ( instance, id ) ; }^95^^^^^84^98^[Delete]^[CLASS] ObjectIdValueProperty  [METHOD] deserializeSetAndReturn [RETURN_TYPE] Object   JsonParser jp DeserializationContext ctxt Object instance [VARIABLES] ReadableObjectId  roid  boolean  SettableBeanProperty  idProp  ObjectIdReader  _objectIdReader  objectIdReader  DeserializationContext  ctxt  Object  id  instance  long  serialVersionUID  JsonParser  jp  
[P5_Replace_Variable]^return id.setAndReturn ( instance, idProp ) ;^95^^^^^84^98^return idProp.setAndReturn ( instance, id ) ;^[CLASS] ObjectIdValueProperty  [METHOD] deserializeSetAndReturn [RETURN_TYPE] Object   JsonParser jp DeserializationContext ctxt Object instance [VARIABLES] ReadableObjectId  roid  boolean  SettableBeanProperty  idProp  ObjectIdReader  _objectIdReader  objectIdReader  DeserializationContext  ctxt  Object  id  instance  long  serialVersionUID  JsonParser  jp  
[P14_Delete_Statement]^^95^^^^^84^98^return idProp.setAndReturn ( instance, id ) ;^[CLASS] ObjectIdValueProperty  [METHOD] deserializeSetAndReturn [RETURN_TYPE] Object   JsonParser jp DeserializationContext ctxt Object instance [VARIABLES] ReadableObjectId  roid  boolean  SettableBeanProperty  idProp  ObjectIdReader  _objectIdReader  objectIdReader  DeserializationContext  ctxt  Object  id  instance  long  serialVersionUID  JsonParser  jp  
[P5_Replace_Variable]^return idProp.setAndReturn ( id, id ) ;^95^^^^^84^98^return idProp.setAndReturn ( instance, id ) ;^[CLASS] ObjectIdValueProperty  [METHOD] deserializeSetAndReturn [RETURN_TYPE] Object   JsonParser jp DeserializationContext ctxt Object instance [VARIABLES] ReadableObjectId  roid  boolean  SettableBeanProperty  idProp  ObjectIdReader  _objectIdReader  objectIdReader  DeserializationContext  ctxt  Object  id  instance  long  serialVersionUID  JsonParser  jp  
[P5_Replace_Variable]^return idPropProp.setAndReturn ( instance, id ) ;^95^^^^^84^98^return idProp.setAndReturn ( instance, id ) ;^[CLASS] ObjectIdValueProperty  [METHOD] deserializeSetAndReturn [RETURN_TYPE] Object   JsonParser jp DeserializationContext ctxt Object instance [VARIABLES] ReadableObjectId  roid  boolean  SettableBeanProperty  idProp  ObjectIdReader  _objectIdReader  objectIdReader  DeserializationContext  ctxt  Object  id  instance  long  serialVersionUID  JsonParser  jp  
[P5_Replace_Variable]^return id;^97^^^^^84^98^return instance;^[CLASS] ObjectIdValueProperty  [METHOD] deserializeSetAndReturn [RETURN_TYPE] Object   JsonParser jp DeserializationContext ctxt Object instance [VARIABLES] ReadableObjectId  roid  boolean  SettableBeanProperty  idProp  ObjectIdReader  _objectIdReader  objectIdReader  DeserializationContext  ctxt  Object  id  instance  long  serialVersionUID  JsonParser  jp  
[P5_Replace_Variable]^setAndReturn (  value ) ;^103^^^^^102^104^setAndReturn ( instance, value ) ;^[CLASS] ObjectIdValueProperty  [METHOD] set [RETURN_TYPE] void   Object instance Object value [VARIABLES] Object  instance  value  boolean  long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  
[P5_Replace_Variable]^setAndReturn ( instance ) ;^103^^^^^102^104^setAndReturn ( instance, value ) ;^[CLASS] ObjectIdValueProperty  [METHOD] set [RETURN_TYPE] void   Object instance Object value [VARIABLES] Object  instance  value  boolean  long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  
[P5_Replace_Variable]^setAndReturn ( value, instance ) ;^103^^^^^102^104^setAndReturn ( instance, value ) ;^[CLASS] ObjectIdValueProperty  [METHOD] set [RETURN_TYPE] void   Object instance Object value [VARIABLES] Object  instance  value  boolean  long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  
[P7_Replace_Invocation]^set ( instance, value ) ;^103^^^^^102^104^setAndReturn ( instance, value ) ;^[CLASS] ObjectIdValueProperty  [METHOD] set [RETURN_TYPE] void   Object instance Object value [VARIABLES] Object  instance  value  boolean  long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  
[P14_Delete_Statement]^^103^^^^^102^104^setAndReturn ( instance, value ) ;^[CLASS] ObjectIdValueProperty  [METHOD] set [RETURN_TYPE] void   Object instance Object value [VARIABLES] Object  instance  value  boolean  long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  
[P11_Insert_Donor_Statement]^return idProp.setAndReturn ( instance, id ) ;setAndReturn ( instance, value ) ;^103^^^^^102^104^setAndReturn ( instance, value ) ;^[CLASS] ObjectIdValueProperty  [METHOD] set [RETURN_TYPE] void   Object instance Object value [VARIABLES] Object  instance  value  boolean  long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  
[P11_Insert_Donor_Statement]^return idProp.setAndReturn ( instance, value ) ;setAndReturn ( instance, value ) ;^103^^^^^102^104^setAndReturn ( instance, value ) ;^[CLASS] ObjectIdValueProperty  [METHOD] set [RETURN_TYPE] void   Object instance Object value [VARIABLES] Object  instance  value  boolean  long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  
[P2_Replace_Operator]^if  ( idProp != null )  {^111^^^^^107^116^if  ( idProp == null )  {^[CLASS] ObjectIdValueProperty  [METHOD] setAndReturn [RETURN_TYPE] Object   Object instance Object value [VARIABLES] Object  instance  value  boolean  SettableBeanProperty  idProp  long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  
[P8_Replace_Mix]^if  ( idProp == false )  {^111^^^^^107^116^if  ( idProp == null )  {^[CLASS] ObjectIdValueProperty  [METHOD] setAndReturn [RETURN_TYPE] Object   Object instance Object value [VARIABLES] Object  instance  value  boolean  SettableBeanProperty  idProp  long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  
[P15_Unwrap_Block]^throw new java.lang.UnsupportedOperationException("Should not call set() on ObjectIdProperty that has no SettableBeanProperty");^111^112^113^114^^107^116^if  ( idProp == null )  { throw new UnsupportedOperationException ( "Should not call set (  )  on ObjectIdProperty that has no SettableBeanProperty" ) ; }^[CLASS] ObjectIdValueProperty  [METHOD] setAndReturn [RETURN_TYPE] Object   Object instance Object value [VARIABLES] Object  instance  value  boolean  SettableBeanProperty  idProp  long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  
[P16_Remove_Block]^^111^112^113^114^^107^116^if  ( idProp == null )  { throw new UnsupportedOperationException ( "Should not call set (  )  on ObjectIdProperty that has no SettableBeanProperty" ) ; }^[CLASS] ObjectIdValueProperty  [METHOD] setAndReturn [RETURN_TYPE] Object   Object instance Object value [VARIABLES] Object  instance  value  boolean  SettableBeanProperty  idProp  long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  
[P13_Insert_Block]^if  ( idProp == null )  {     throw new UnsupportedOperationException ( "Should not call set (  )  on ObjectIdProperty that has no SettableBeanProperty" ) ; }^112^^^^^107^116^[Delete]^[CLASS] ObjectIdValueProperty  [METHOD] setAndReturn [RETURN_TYPE] Object   Object instance Object value [VARIABLES] Object  instance  value  boolean  SettableBeanProperty  idProp  long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  
[P5_Replace_Variable]^return idProp.setAndReturn ( instance, instance ) ;^115^^^^^107^116^return idProp.setAndReturn ( instance, value ) ;^[CLASS] ObjectIdValueProperty  [METHOD] setAndReturn [RETURN_TYPE] Object   Object instance Object value [VARIABLES] Object  instance  value  boolean  SettableBeanProperty  idProp  long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  
[P5_Replace_Variable]^return idProp.setAndReturn (  value ) ;^115^^^^^107^116^return idProp.setAndReturn ( instance, value ) ;^[CLASS] ObjectIdValueProperty  [METHOD] setAndReturn [RETURN_TYPE] Object   Object instance Object value [VARIABLES] Object  instance  value  boolean  SettableBeanProperty  idProp  long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  
[P5_Replace_Variable]^return idProp.setAndReturn ( instance ) ;^115^^^^^107^116^return idProp.setAndReturn ( instance, value ) ;^[CLASS] ObjectIdValueProperty  [METHOD] setAndReturn [RETURN_TYPE] Object   Object instance Object value [VARIABLES] Object  instance  value  boolean  SettableBeanProperty  idProp  long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  
[P5_Replace_Variable]^return value.setAndReturn ( instance, idProp ) ;^115^^^^^107^116^return idProp.setAndReturn ( instance, value ) ;^[CLASS] ObjectIdValueProperty  [METHOD] setAndReturn [RETURN_TYPE] Object   Object instance Object value [VARIABLES] Object  instance  value  boolean  SettableBeanProperty  idProp  long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  
[P5_Replace_Variable]^return idProp.setAndReturn ( value, instance ) ;^115^^^^^107^116^return idProp.setAndReturn ( instance, value ) ;^[CLASS] ObjectIdValueProperty  [METHOD] setAndReturn [RETURN_TYPE] Object   Object instance Object value [VARIABLES] Object  instance  value  boolean  SettableBeanProperty  idProp  long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  
[P5_Replace_Variable]^return instance.setAndReturn ( idProp, value ) ;^115^^^^^107^116^return idProp.setAndReturn ( instance, value ) ;^[CLASS] ObjectIdValueProperty  [METHOD] setAndReturn [RETURN_TYPE] Object   Object instance Object value [VARIABLES] Object  instance  value  boolean  SettableBeanProperty  idProp  long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  
[P14_Delete_Statement]^^115^^^^^107^116^return idProp.setAndReturn ( instance, value ) ;^[CLASS] ObjectIdValueProperty  [METHOD] setAndReturn [RETURN_TYPE] Object   Object instance Object value [VARIABLES] Object  instance  value  boolean  SettableBeanProperty  idProp  long  serialVersionUID  ObjectIdReader  _objectIdReader  objectIdReader  