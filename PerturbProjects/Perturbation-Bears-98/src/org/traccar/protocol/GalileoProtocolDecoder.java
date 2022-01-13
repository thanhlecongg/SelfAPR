[REPLACE]^private static final  short  TAG_IMEI = 0x03;^42^^^^^^^[REPLACE] private static final int TAG_IMEI = 0x03;^ [CLASS] GalileoProtocolDecoder  
[REPLACE]^private static final int TAG_COORDINATES  = null ;^44^^^^^^^[REPLACE] private static final int TAG_COORDINATES = 0x30;^ [CLASS] GalileoProtocolDecoder  
[REPLACE]^private static   long  TAG_SPEED_COURSE = 0x33;^45^^^^^^^[REPLACE] private static final int TAG_SPEED_COURSE = 0x33;^ [CLASS] GalileoProtocolDecoder  
[REPLACE]^private static final  short  TAG_ALTITUDE = 0x34;^46^^^^^^^[REPLACE] private static final int TAG_ALTITUDE = 0x34;^ [CLASS] GalileoProtocolDecoder  
[REPLACE]^private static final int TAG_STATUS ;^47^^^^^^^[REPLACE] private static final int TAG_STATUS = 0x40;^ [CLASS] GalileoProtocolDecoder  
[REPLACE]^private static  int TAG_POWER = 0x41;^48^^^^^^^[REPLACE] private static final int TAG_POWER = 0x41;^ [CLASS] GalileoProtocolDecoder  
[REPLACE]^private static final  short  TAG_BATTERY = 0x42;^49^^^^^^^[REPLACE] private static final int TAG_BATTERY = 0x42;^ [CLASS] GalileoProtocolDecoder  
[REPLACE]^private static final  short  TAG_ODOMETER = 0xd4;^50^^^^^^^[REPLACE] private static final int TAG_ODOMETER = 0xd4;^ [CLASS] GalileoProtocolDecoder  
[REPLACE]^private static final int TAG_REFRIGERATOR ;^51^^^^^^^[REPLACE] private static final int TAG_REFRIGERATOR = 0x5b;^ [CLASS] GalileoProtocolDecoder  
[REPLACE]^private static  int TAG_PRESSURE = 0x5c;^52^^^^^^^[REPLACE] private static final int TAG_PRESSURE = 0x5c;^ [CLASS] GalileoProtocolDecoder  
[REPLACE]^private static final int TAG_CAN  = null ;^53^^^^^^^[REPLACE] private static final int TAG_CAN = 0xc1;^ [CLASS] GalileoProtocolDecoder  
[REPLACE]^private static final int TAG_ADC0  = null ;^54^^^^^^^[REPLACE] private static final int TAG_ADC0 = 0x50;^ [CLASS] GalileoProtocolDecoder  
[REPLACE]^private static  int TAG_ADC1 = 0x51;^55^^^^^^^[REPLACE] private static final int TAG_ADC1 = 0x51;^ [CLASS] GalileoProtocolDecoder  
[REPLACE]^private static final int TAG_ADC3 ;^57^^^^^^^[REPLACE] private static final int TAG_ADC3 = 0x53;^ [CLASS] GalileoProtocolDecoder  
[REPLACE]^private static final Map<Integer, Integer> TAG_LENGTH_MAP ;^59^^^^^^^[REPLACE] private static final Map<Integer, Integer> TAG_LENGTH_MAP = new HashMap<> (  ) ;^ [CLASS] GalileoProtocolDecoder  
[ADD]^^121^^^^^119^236^[ADD] ChannelBuffer buf =  ( ChannelBuffer )  msg;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] GalileoProtocolDecoder   [TYPE]  Set tags  [TYPE]  boolean false  hasLocation  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position p  position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  List positions  [TYPE]  SocketAddress remoteAddress  [TYPE]  Map TAG_LENGTH_MAP  [TYPE]  int TAG_ADC0  TAG_ADC1  TAG_ADC2  TAG_ADC3  TAG_ALTITUDE  TAG_BATTERY  TAG_CAN  TAG_COORDINATES  TAG_DATE  TAG_IMEI  TAG_ODOMETER  TAG_POWER  TAG_PRESSURE  TAG_REFRIGERATOR  TAG_SPEED_COURSE  TAG_STATUS  checksum  length  tag  [TYPE]  ChannelBuffer buf 
[REPLACE]^( buf.readUnsignedByte (  )  )  ;^123^^^^^119^236^[REPLACE] buf.readUnsignedByte (  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] GalileoProtocolDecoder   [TYPE]  Set tags  [TYPE]  boolean false  hasLocation  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position p  position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  List positions  [TYPE]  SocketAddress remoteAddress  [TYPE]  Map TAG_LENGTH_MAP  [TYPE]  int TAG_ADC0  TAG_ADC1  TAG_ADC2  TAG_ADC3  TAG_ALTITUDE  TAG_BATTERY  TAG_CAN  TAG_COORDINATES  TAG_DATE  TAG_IMEI  TAG_ODOMETER  TAG_POWER  TAG_PRESSURE  TAG_REFRIGERATOR  TAG_SPEED_COURSE  TAG_STATUS  checksum  length  tag  [TYPE]  ChannelBuffer buf 
[REPLACE]^int length - 5 =  ( buf.readBytes (  )  & 0x7fff )  + 3;^124^^^^^119^236^[REPLACE] int length =  ( buf.readUnsignedShort (  )  & 0x7fff )  + 3;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] GalileoProtocolDecoder   [TYPE]  Set tags  [TYPE]  boolean false  hasLocation  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position p  position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  List positions  [TYPE]  SocketAddress remoteAddress  [TYPE]  Map TAG_LENGTH_MAP  [TYPE]  int TAG_ADC0  TAG_ADC1  TAG_ADC2  TAG_ADC3  TAG_ALTITUDE  TAG_BATTERY  TAG_CAN  TAG_COORDINATES  TAG_DATE  TAG_IMEI  TAG_ODOMETER  TAG_POWER  TAG_PRESSURE  TAG_REFRIGERATOR  TAG_SPEED_COURSE  TAG_STATUS  checksum  length  tag  [TYPE]  ChannelBuffer buf 
[ADD]^^127^^^^^119^236^[ADD] Set<Integer> tags = new HashSet<> (  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] GalileoProtocolDecoder   [TYPE]  Set tags  [TYPE]  boolean false  hasLocation  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position p  position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  List positions  [TYPE]  SocketAddress remoteAddress  [TYPE]  Map TAG_LENGTH_MAP  [TYPE]  int TAG_ADC0  TAG_ADC1  TAG_ADC2  TAG_ADC3  TAG_ALTITUDE  TAG_BATTERY  TAG_CAN  TAG_COORDINATES  TAG_DATE  TAG_IMEI  TAG_ODOMETER  TAG_POWER  TAG_PRESSURE  TAG_REFRIGERATOR  TAG_SPEED_COURSE  TAG_STATUS  checksum  length  tag  [TYPE]  ChannelBuffer buf 
[REPLACE]^boolean hasLocation = true;^128^^^^^119^236^[REPLACE] boolean hasLocation = false;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] GalileoProtocolDecoder   [TYPE]  Set tags  [TYPE]  boolean false  hasLocation  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position p  position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  List positions  [TYPE]  SocketAddress remoteAddress  [TYPE]  Map TAG_LENGTH_MAP  [TYPE]  int TAG_ADC0  TAG_ADC1  TAG_ADC2  TAG_ADC3  TAG_ALTITUDE  TAG_BATTERY  TAG_CAN  TAG_COORDINATES  TAG_DATE  TAG_IMEI  TAG_ODOMETER  TAG_POWER  TAG_PRESSURE  TAG_REFRIGERATOR  TAG_SPEED_COURSE  TAG_STATUS  checksum  length  tag  [TYPE]  ChannelBuffer buf 
[ADD]^^130^^^^^119^236^[ADD] Position position = new Position (  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] GalileoProtocolDecoder   [TYPE]  Set tags  [TYPE]  boolean false  hasLocation  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position p  position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  List positions  [TYPE]  SocketAddress remoteAddress  [TYPE]  Map TAG_LENGTH_MAP  [TYPE]  int TAG_ADC0  TAG_ADC1  TAG_ADC2  TAG_ADC3  TAG_ALTITUDE  TAG_BATTERY  TAG_CAN  TAG_COORDINATES  TAG_DATE  TAG_IMEI  TAG_ODOMETER  TAG_POWER  TAG_PRESSURE  TAG_REFRIGERATOR  TAG_SPEED_COURSE  TAG_STATUS  checksum  length  tag  [TYPE]  ChannelBuffer buf 
[ADD]^^136^137^138^139^^119^236^[ADD] if  ( tags.contains ( tag )  )  { if  ( hasLocation && position.getFixTime (  )  != null )  { positions.add ( position ) ; }^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] GalileoProtocolDecoder   [TYPE]  Set tags  [TYPE]  boolean false  hasLocation  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position p  position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  List positions  [TYPE]  SocketAddress remoteAddress  [TYPE]  Map TAG_LENGTH_MAP  [TYPE]  int TAG_ADC0  TAG_ADC1  TAG_ADC2  TAG_ADC3  TAG_ALTITUDE  TAG_BATTERY  TAG_CAN  TAG_COORDINATES  TAG_DATE  TAG_IMEI  TAG_ODOMETER  TAG_POWER  TAG_PRESSURE  TAG_REFRIGERATOR  TAG_SPEED_COURSE  TAG_STATUS  checksum  length  tag  [TYPE]  ChannelBuffer buf 
[REPLACE]^if  ( hasLocation && position.getFixTime (  )  == null )  {^137^^^^^119^236^[REPLACE] if  ( hasLocation && position.getFixTime (  )  != null )  {^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] GalileoProtocolDecoder   [TYPE]  Set tags  [TYPE]  boolean false  hasLocation  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position p  position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  List positions  [TYPE]  SocketAddress remoteAddress  [TYPE]  Map TAG_LENGTH_MAP  [TYPE]  int TAG_ADC0  TAG_ADC1  TAG_ADC2  TAG_ADC3  TAG_ALTITUDE  TAG_BATTERY  TAG_CAN  TAG_COORDINATES  TAG_DATE  TAG_IMEI  TAG_ODOMETER  TAG_POWER  TAG_PRESSURE  TAG_REFRIGERATOR  TAG_SPEED_COURSE  TAG_STATUS  checksum  length  tag  [TYPE]  ChannelBuffer buf 
[REPLACE]^positions.add ( p ) ;^138^^^^^119^236^[REPLACE] positions.add ( position ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] GalileoProtocolDecoder   [TYPE]  Set tags  [TYPE]  boolean false  hasLocation  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position p  position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  List positions  [TYPE]  SocketAddress remoteAddress  [TYPE]  Map TAG_LENGTH_MAP  [TYPE]  int TAG_ADC0  TAG_ADC1  TAG_ADC2  TAG_ADC3  TAG_ALTITUDE  TAG_BATTERY  TAG_CAN  TAG_COORDINATES  TAG_DATE  TAG_IMEI  TAG_ODOMETER  TAG_POWER  TAG_PRESSURE  TAG_REFRIGERATOR  TAG_SPEED_COURSE  TAG_STATUS  checksum  length  tag  [TYPE]  ChannelBuffer buf 
[REPLACE]^hasLocation = true;^141^^^^^119^236^[REPLACE] hasLocation = false;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] GalileoProtocolDecoder   [TYPE]  Set tags  [TYPE]  boolean false  hasLocation  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position p  position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  List positions  [TYPE]  SocketAddress remoteAddress  [TYPE]  Map TAG_LENGTH_MAP  [TYPE]  int TAG_ADC0  TAG_ADC1  TAG_ADC2  TAG_ADC3  TAG_ALTITUDE  TAG_BATTERY  TAG_CAN  TAG_COORDINATES  TAG_DATE  TAG_IMEI  TAG_ODOMETER  TAG_POWER  TAG_PRESSURE  TAG_REFRIGERATOR  TAG_SPEED_COURSE  TAG_STATUS  checksum  length  tag  [TYPE]  ChannelBuffer buf 
[REPLACE]^position = new Object (  ) ;^142^^^^^119^236^[REPLACE] position = new Position (  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] GalileoProtocolDecoder   [TYPE]  Set tags  [TYPE]  boolean false  hasLocation  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position p  position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  List positions  [TYPE]  SocketAddress remoteAddress  [TYPE]  Map TAG_LENGTH_MAP  [TYPE]  int TAG_ADC0  TAG_ADC1  TAG_ADC2  TAG_ADC3  TAG_ALTITUDE  TAG_BATTERY  TAG_CAN  TAG_COORDINATES  TAG_DATE  TAG_IMEI  TAG_ODOMETER  TAG_POWER  TAG_PRESSURE  TAG_REFRIGERATOR  TAG_SPEED_COURSE  TAG_STATUS  checksum  length  tag  [TYPE]  ChannelBuffer buf 
[REPLACE]^if  (  position.getFixTime (  )  == true )  {^137^^^^^119^236^[REPLACE] if  ( hasLocation && position.getFixTime (  )  != null )  {^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] GalileoProtocolDecoder   [TYPE]  Set tags  [TYPE]  boolean false  hasLocation  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position p  position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  List positions  [TYPE]  SocketAddress remoteAddress  [TYPE]  Map TAG_LENGTH_MAP  [TYPE]  int TAG_ADC0  TAG_ADC1  TAG_ADC2  TAG_ADC3  TAG_ALTITUDE  TAG_BATTERY  TAG_CAN  TAG_COORDINATES  TAG_DATE  TAG_IMEI  TAG_ODOMETER  TAG_POWER  TAG_PRESSURE  TAG_REFRIGERATOR  TAG_SPEED_COURSE  TAG_STATUS  checksum  length  tag  [TYPE]  ChannelBuffer buf 
[ADD]^^138^^^^^119^236^[ADD] positions.add ( position ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] GalileoProtocolDecoder   [TYPE]  Set tags  [TYPE]  boolean false  hasLocation  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position p  position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  List positions  [TYPE]  SocketAddress remoteAddress  [TYPE]  Map TAG_LENGTH_MAP  [TYPE]  int TAG_ADC0  TAG_ADC1  TAG_ADC2  TAG_ADC3  TAG_ALTITUDE  TAG_BATTERY  TAG_CAN  TAG_COORDINATES  TAG_DATE  TAG_IMEI  TAG_ODOMETER  TAG_POWER  TAG_PRESSURE  TAG_REFRIGERATOR  TAG_SPEED_COURSE  TAG_STATUS  checksum  length  tag  [TYPE]  ChannelBuffer buf 
[REPLACE]^hasLocation = true  ;^141^^^^^119^236^[REPLACE] hasLocation = false;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] GalileoProtocolDecoder   [TYPE]  Set tags  [TYPE]  boolean false  hasLocation  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position p  position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  List positions  [TYPE]  SocketAddress remoteAddress  [TYPE]  Map TAG_LENGTH_MAP  [TYPE]  int TAG_ADC0  TAG_ADC1  TAG_ADC2  TAG_ADC3  TAG_ALTITUDE  TAG_BATTERY  TAG_CAN  TAG_COORDINATES  TAG_DATE  TAG_IMEI  TAG_ODOMETER  TAG_POWER  TAG_PRESSURE  TAG_REFRIGERATOR  TAG_SPEED_COURSE  TAG_STATUS  checksum  length  tag  [TYPE]  ChannelBuffer buf 
[REPLACE]^hasLocation = false;^157^^^^^119^236^[REPLACE] hasLocation = true;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] GalileoProtocolDecoder   [TYPE]  Set tags  [TYPE]  boolean false  hasLocation  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position p  position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  List positions  [TYPE]  SocketAddress remoteAddress  [TYPE]  Map TAG_LENGTH_MAP  [TYPE]  int TAG_ADC0  TAG_ADC1  TAG_ADC2  TAG_ADC3  TAG_ALTITUDE  TAG_BATTERY  TAG_CAN  TAG_COORDINATES  TAG_DATE  TAG_IMEI  TAG_ODOMETER  TAG_POWER  TAG_PRESSURE  TAG_REFRIGERATOR  TAG_SPEED_COURSE  TAG_STATUS  checksum  length  tag  [TYPE]  ChannelBuffer buf 
[REPLACE]^int tag = buf .readUnsignedInt (  )  ;^135^^^^^119^236^[REPLACE] int tag = buf.readUnsignedByte (  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] GalileoProtocolDecoder   [TYPE]  Set tags  [TYPE]  boolean false  hasLocation  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position p  position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  List positions  [TYPE]  SocketAddress remoteAddress  [TYPE]  Map TAG_LENGTH_MAP  [TYPE]  int TAG_ADC0  TAG_ADC1  TAG_ADC2  TAG_ADC3  TAG_ALTITUDE  TAG_BATTERY  TAG_CAN  TAG_COORDINATES  TAG_DATE  TAG_IMEI  TAG_ODOMETER  TAG_POWER  TAG_PRESSURE  TAG_REFRIGERATOR  TAG_SPEED_COURSE  TAG_STATUS  checksum  length  tag  [TYPE]  ChannelBuffer buf 
[REPLACE]^if  (  position.getFixTime (  )  == true )  {^216^^^^^119^236^[REPLACE] if  ( hasLocation && position.getFixTime (  )  != null )  {^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] GalileoProtocolDecoder   [TYPE]  Set tags  [TYPE]  boolean false  hasLocation  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position p  position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  List positions  [TYPE]  SocketAddress remoteAddress  [TYPE]  Map TAG_LENGTH_MAP  [TYPE]  int TAG_ADC0  TAG_ADC1  TAG_ADC2  TAG_ADC3  TAG_ALTITUDE  TAG_BATTERY  TAG_CAN  TAG_COORDINATES  TAG_DATE  TAG_IMEI  TAG_ODOMETER  TAG_POWER  TAG_PRESSURE  TAG_REFRIGERATOR  TAG_SPEED_COURSE  TAG_STATUS  checksum  length  tag  [TYPE]  ChannelBuffer buf 
[ADD]^^216^217^218^^^119^236^[ADD] if  ( hasLocation && position.getFixTime (  )  != null )  { positions.add ( position ) ; }^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] GalileoProtocolDecoder   [TYPE]  Set tags  [TYPE]  boolean false  hasLocation  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position p  position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  List positions  [TYPE]  SocketAddress remoteAddress  [TYPE]  Map TAG_LENGTH_MAP  [TYPE]  int TAG_ADC0  TAG_ADC1  TAG_ADC2  TAG_ADC3  TAG_ALTITUDE  TAG_BATTERY  TAG_CAN  TAG_COORDINATES  TAG_DATE  TAG_IMEI  TAG_ODOMETER  TAG_POWER  TAG_PRESSURE  TAG_REFRIGERATOR  TAG_SPEED_COURSE  TAG_STATUS  checksum  length  tag  [TYPE]  ChannelBuffer buf 
[REPLACE]^positions.add ( p ) ;^217^^^^^119^236^[REPLACE] positions.add ( position ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] GalileoProtocolDecoder   [TYPE]  Set tags  [TYPE]  boolean false  hasLocation  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position p  position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  List positions  [TYPE]  SocketAddress remoteAddress  [TYPE]  Map TAG_LENGTH_MAP  [TYPE]  int TAG_ADC0  TAG_ADC1  TAG_ADC2  TAG_ADC3  TAG_ALTITUDE  TAG_BATTERY  TAG_CAN  TAG_COORDINATES  TAG_DATE  TAG_IMEI  TAG_ODOMETER  TAG_POWER  TAG_PRESSURE  TAG_REFRIGERATOR  TAG_SPEED_COURSE  TAG_STATUS  checksum  length  tag  [TYPE]  ChannelBuffer buf 
[REPLACE]^if  ( deviceSession != null )  {^221^^^^^119^236^[REPLACE] if  ( deviceSession == null )  {^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] GalileoProtocolDecoder   [TYPE]  Set tags  [TYPE]  boolean false  hasLocation  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position p  position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  List positions  [TYPE]  SocketAddress remoteAddress  [TYPE]  Map TAG_LENGTH_MAP  [TYPE]  int TAG_ADC0  TAG_ADC1  TAG_ADC2  TAG_ADC3  TAG_ALTITUDE  TAG_BATTERY  TAG_CAN  TAG_COORDINATES  TAG_DATE  TAG_IMEI  TAG_ODOMETER  TAG_POWER  TAG_PRESSURE  TAG_REFRIGERATOR  TAG_SPEED_COURSE  TAG_STATUS  checksum  length  tag  [TYPE]  ChannelBuffer buf 
[ADD]^return null;^221^222^223^^^119^236^[ADD] if  ( deviceSession == null )  { return null; }^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] GalileoProtocolDecoder   [TYPE]  Set tags  [TYPE]  boolean false  hasLocation  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position p  position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  List positions  [TYPE]  SocketAddress remoteAddress  [TYPE]  Map TAG_LENGTH_MAP  [TYPE]  int TAG_ADC0  TAG_ADC1  TAG_ADC2  TAG_ADC3  TAG_ALTITUDE  TAG_BATTERY  TAG_CAN  TAG_COORDINATES  TAG_DATE  TAG_IMEI  TAG_ODOMETER  TAG_POWER  TAG_PRESSURE  TAG_REFRIGERATOR  TAG_SPEED_COURSE  TAG_STATUS  checksum  length  tag  [TYPE]  ChannelBuffer buf 
[REPLACE]^return this;^222^^^^^119^236^[REPLACE] return null;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] GalileoProtocolDecoder   [TYPE]  Set tags  [TYPE]  boolean false  hasLocation  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position p  position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  List positions  [TYPE]  SocketAddress remoteAddress  [TYPE]  Map TAG_LENGTH_MAP  [TYPE]  int TAG_ADC0  TAG_ADC1  TAG_ADC2  TAG_ADC3  TAG_ALTITUDE  TAG_BATTERY  TAG_CAN  TAG_COORDINATES  TAG_DATE  TAG_IMEI  TAG_ODOMETER  TAG_POWER  TAG_PRESSURE  TAG_REFRIGERATOR  TAG_SPEED_COURSE  TAG_STATUS  checksum  length  tag  [TYPE]  ChannelBuffer buf 
[REPLACE]^sendReply ( channel, buf.readUnsignedInt (  )  ) ;^225^^^^^119^236^[REPLACE] sendReply ( channel, buf.readUnsignedShort (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] GalileoProtocolDecoder   [TYPE]  Set tags  [TYPE]  boolean false  hasLocation  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position p  position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  List positions  [TYPE]  SocketAddress remoteAddress  [TYPE]  Map TAG_LENGTH_MAP  [TYPE]  int TAG_ADC0  TAG_ADC1  TAG_ADC2  TAG_ADC3  TAG_ALTITUDE  TAG_BATTERY  TAG_CAN  TAG_COORDINATES  TAG_DATE  TAG_IMEI  TAG_ODOMETER  TAG_POWER  TAG_PRESSURE  TAG_REFRIGERATOR  TAG_SPEED_COURSE  TAG_STATUS  checksum  length  tag  [TYPE]  ChannelBuffer buf 
[REPLACE]^( buf.readUnsignedShort (  )  )  ;^225^^^^^119^236^[REPLACE] sendReply ( channel, buf.readUnsignedShort (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] GalileoProtocolDecoder   [TYPE]  Set tags  [TYPE]  boolean false  hasLocation  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position p  position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  List positions  [TYPE]  SocketAddress remoteAddress  [TYPE]  Map TAG_LENGTH_MAP  [TYPE]  int TAG_ADC0  TAG_ADC1  TAG_ADC2  TAG_ADC3  TAG_ALTITUDE  TAG_BATTERY  TAG_CAN  TAG_COORDINATES  TAG_DATE  TAG_IMEI  TAG_ODOMETER  TAG_POWER  TAG_PRESSURE  TAG_REFRIGERATOR  TAG_SPEED_COURSE  TAG_STATUS  checksum  length  tag  [TYPE]  ChannelBuffer buf 
[ADD]^^225^^^^^119^236^[ADD] sendReply ( channel, buf.readUnsignedShort (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] GalileoProtocolDecoder   [TYPE]  Set tags  [TYPE]  boolean false  hasLocation  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position p  position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  List positions  [TYPE]  SocketAddress remoteAddress  [TYPE]  Map TAG_LENGTH_MAP  [TYPE]  int TAG_ADC0  TAG_ADC1  TAG_ADC2  TAG_ADC3  TAG_ALTITUDE  TAG_BATTERY  TAG_CAN  TAG_COORDINATES  TAG_DATE  TAG_IMEI  TAG_ODOMETER  TAG_POWER  TAG_PRESSURE  TAG_REFRIGERATOR  TAG_SPEED_COURSE  TAG_STATUS  checksum  length  tag  [TYPE]  ChannelBuffer buf 
[ADD]^^227^228^229^230^^119^236^[ADD] for  ( Position p : positions )  { p.setProtocol ( getProtocolName (  )  ) ; p.setDeviceId ( deviceSession.getDeviceId (  )  ) ; }^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] GalileoProtocolDecoder   [TYPE]  Set tags  [TYPE]  boolean false  hasLocation  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position p  position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  List positions  [TYPE]  SocketAddress remoteAddress  [TYPE]  Map TAG_LENGTH_MAP  [TYPE]  int TAG_ADC0  TAG_ADC1  TAG_ADC2  TAG_ADC3  TAG_ALTITUDE  TAG_BATTERY  TAG_CAN  TAG_COORDINATES  TAG_DATE  TAG_IMEI  TAG_ODOMETER  TAG_POWER  TAG_PRESSURE  TAG_REFRIGERATOR  TAG_SPEED_COURSE  TAG_STATUS  checksum  length  tag  [TYPE]  ChannelBuffer buf 
[REPLACE]^p .setTime ( null )  ;^228^^^^^119^236^[REPLACE] p.setProtocol ( getProtocolName (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] GalileoProtocolDecoder   [TYPE]  Set tags  [TYPE]  boolean false  hasLocation  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position p  position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  List positions  [TYPE]  SocketAddress remoteAddress  [TYPE]  Map TAG_LENGTH_MAP  [TYPE]  int TAG_ADC0  TAG_ADC1  TAG_ADC2  TAG_ADC3  TAG_ALTITUDE  TAG_BATTERY  TAG_CAN  TAG_COORDINATES  TAG_DATE  TAG_IMEI  TAG_ODOMETER  TAG_POWER  TAG_PRESSURE  TAG_REFRIGERATOR  TAG_SPEED_COURSE  TAG_STATUS  checksum  length  tag  [TYPE]  ChannelBuffer buf 
[ADD]^^228^^^^^119^236^[ADD] p.setProtocol ( getProtocolName (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] GalileoProtocolDecoder   [TYPE]  Set tags  [TYPE]  boolean false  hasLocation  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position p  position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  List positions  [TYPE]  SocketAddress remoteAddress  [TYPE]  Map TAG_LENGTH_MAP  [TYPE]  int TAG_ADC0  TAG_ADC1  TAG_ADC2  TAG_ADC3  TAG_ALTITUDE  TAG_BATTERY  TAG_CAN  TAG_COORDINATES  TAG_DATE  TAG_IMEI  TAG_ODOMETER  TAG_POWER  TAG_PRESSURE  TAG_REFRIGERATOR  TAG_SPEED_COURSE  TAG_STATUS  checksum  length  tag  [TYPE]  ChannelBuffer buf 
[REPLACE]^deviceSession.getDeviceId (  )  ;^229^^^^^119^236^[REPLACE] p.setDeviceId ( deviceSession.getDeviceId (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] GalileoProtocolDecoder   [TYPE]  Set tags  [TYPE]  boolean false  hasLocation  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position p  position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  List positions  [TYPE]  SocketAddress remoteAddress  [TYPE]  Map TAG_LENGTH_MAP  [TYPE]  int TAG_ADC0  TAG_ADC1  TAG_ADC2  TAG_ADC3  TAG_ALTITUDE  TAG_BATTERY  TAG_CAN  TAG_COORDINATES  TAG_DATE  TAG_IMEI  TAG_ODOMETER  TAG_POWER  TAG_PRESSURE  TAG_REFRIGERATOR  TAG_SPEED_COURSE  TAG_STATUS  checksum  length  tag  [TYPE]  ChannelBuffer buf 
[REPLACE]^p.setDeviceId ( deviceSession.getDeviceId (  )  )  ;^229^^^^^119^236^[REPLACE] p.setDeviceId ( deviceSession.getDeviceId (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] GalileoProtocolDecoder   [TYPE]  Set tags  [TYPE]  boolean false  hasLocation  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position p  position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  List positions  [TYPE]  SocketAddress remoteAddress  [TYPE]  Map TAG_LENGTH_MAP  [TYPE]  int TAG_ADC0  TAG_ADC1  TAG_ADC2  TAG_ADC3  TAG_ALTITUDE  TAG_BATTERY  TAG_CAN  TAG_COORDINATES  TAG_DATE  TAG_IMEI  TAG_ODOMETER  TAG_POWER  TAG_PRESSURE  TAG_REFRIGERATOR  TAG_SPEED_COURSE  TAG_STATUS  checksum  length  tag  [TYPE]  ChannelBuffer buf 
[ADD]^^229^^^^^119^236^[ADD] p.setDeviceId ( deviceSession.getDeviceId (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] GalileoProtocolDecoder   [TYPE]  Set tags  [TYPE]  boolean false  hasLocation  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position p  position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  List positions  [TYPE]  SocketAddress remoteAddress  [TYPE]  Map TAG_LENGTH_MAP  [TYPE]  int TAG_ADC0  TAG_ADC1  TAG_ADC2  TAG_ADC3  TAG_ALTITUDE  TAG_BATTERY  TAG_CAN  TAG_COORDINATES  TAG_DATE  TAG_IMEI  TAG_ODOMETER  TAG_POWER  TAG_PRESSURE  TAG_REFRIGERATOR  TAG_SPEED_COURSE  TAG_STATUS  checksum  length  tag  [TYPE]  ChannelBuffer buf 
[REPLACE]^if  ( false.isEmpty (   )  && tags.contains ( tag )  )  )  {^232^^^^^119^236^[REPLACE] if  ( positions.isEmpty (  )  )  {^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] GalileoProtocolDecoder   [TYPE]  Set tags  [TYPE]  boolean false  hasLocation  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position p  position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  List positions  [TYPE]  SocketAddress remoteAddress  [TYPE]  Map TAG_LENGTH_MAP  [TYPE]  int TAG_ADC0  TAG_ADC1  TAG_ADC2  TAG_ADC3  TAG_ALTITUDE  TAG_BATTERY  TAG_CAN  TAG_COORDINATES  TAG_DATE  TAG_IMEI  TAG_ODOMETER  TAG_POWER  TAG_PRESSURE  TAG_REFRIGERATOR  TAG_SPEED_COURSE  TAG_STATUS  checksum  length  tag  [TYPE]  ChannelBuffer buf 
[REPLACE]^return true;^233^^^^^119^236^[REPLACE] return null;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] GalileoProtocolDecoder   [TYPE]  Set tags  [TYPE]  boolean false  hasLocation  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position p  position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  List positions  [TYPE]  SocketAddress remoteAddress  [TYPE]  Map TAG_LENGTH_MAP  [TYPE]  int TAG_ADC0  TAG_ADC1  TAG_ADC2  TAG_ADC3  TAG_ALTITUDE  TAG_BATTERY  TAG_CAN  TAG_COORDINATES  TAG_DATE  TAG_IMEI  TAG_ODOMETER  TAG_POWER  TAG_PRESSURE  TAG_REFRIGERATOR  TAG_SPEED_COURSE  TAG_STATUS  checksum  length  tag  [TYPE]  ChannelBuffer buf 
[REPLACE]^return false;^233^^^^^119^236^[REPLACE] return null;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] GalileoProtocolDecoder   [TYPE]  Set tags  [TYPE]  boolean false  hasLocation  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position p  position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  List positions  [TYPE]  SocketAddress remoteAddress  [TYPE]  Map TAG_LENGTH_MAP  [TYPE]  int TAG_ADC0  TAG_ADC1  TAG_ADC2  TAG_ADC3  TAG_ALTITUDE  TAG_BATTERY  TAG_CAN  TAG_COORDINATES  TAG_DATE  TAG_IMEI  TAG_ODOMETER  TAG_POWER  TAG_PRESSURE  TAG_REFRIGERATOR  TAG_SPEED_COURSE  TAG_STATUS  checksum  length  tag  [TYPE]  ChannelBuffer buf 
[REPLACE]^return 0;^235^^^^^119^236^[REPLACE] return positions;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] GalileoProtocolDecoder   [TYPE]  Set tags  [TYPE]  boolean false  hasLocation  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position p  position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  List positions  [TYPE]  SocketAddress remoteAddress  [TYPE]  Map TAG_LENGTH_MAP  [TYPE]  int TAG_ADC0  TAG_ADC1  TAG_ADC2  TAG_ADC3  TAG_ALTITUDE  TAG_BATTERY  TAG_CAN  TAG_COORDINATES  TAG_DATE  TAG_IMEI  TAG_ODOMETER  TAG_POWER  TAG_PRESSURE  TAG_REFRIGERATOR  TAG_SPEED_COURSE  TAG_STATUS  checksum  length  tag  [TYPE]  ChannelBuffer buf 
