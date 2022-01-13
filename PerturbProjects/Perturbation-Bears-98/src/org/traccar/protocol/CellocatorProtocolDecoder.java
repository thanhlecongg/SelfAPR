[REPLACE]^static  int MSG_CLIENT_STATUS = 0;^36^^^^^^^[REPLACE] static final int MSG_CLIENT_STATUS = 0;^ [CLASS] CellocatorProtocolDecoder  
[REPLACE]^static final int MSG_CLIENT_PROGRAMMING ;^37^^^^^^^[REPLACE] static final int MSG_CLIENT_PROGRAMMING = 3;^ [CLASS] CellocatorProtocolDecoder  
[REPLACE]^final int MSG_CLIENT_SERIAL_LOG = 5;^38^^^^^^^[REPLACE] static final int MSG_CLIENT_SERIAL_LOG = 7;^ [CLASS] CellocatorProtocolDecoder  
[REPLACE]^static final int MSG_CLIENT_SERIAL = 8 * 2;^39^^^^^^^[REPLACE] static final int MSG_CLIENT_SERIAL = 8;^ [CLASS] CellocatorProtocolDecoder  
[REPLACE]^static final int MSG_CLIENT_MODULAR  = null ;^40^^^^^^^[REPLACE] static final int MSG_CLIENT_MODULAR = 9;^ [CLASS] CellocatorProtocolDecoder  
[REPLACE]^public static final int MSG_SERVER_ACKNOWLEDGE  = null ;^42^^^^^^^[REPLACE] public static final int MSG_SERVER_ACKNOWLEDGE = 4;^ [CLASS] CellocatorProtocolDecoder  
[REPLACE]^private  short  commandCount;^44^^^^^^^[REPLACE] private byte commandCount;^ [CLASS] CellocatorProtocolDecoder  
[REPLACE]^ChannelBuffer reply = ChannelBuffers.directBuffer ( ByteOrder.LITTLE_ENDIAN, 28 + 3 ) ;^47^^^^^46^69^[REPLACE] ChannelBuffer reply = ChannelBuffers.directBuffer ( ByteOrder.LITTLE_ENDIAN, 28 ) ;^[METHOD] sendReply [TYPE] void [PARAMETER] Channel channel long deviceId byte packetNumber [CLASS] CellocatorProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  byte checksum  commandCount  packetNumber  [TYPE]  int MSG_CLIENT_MODULAR  MSG_CLIENT_PROGRAMMING  MSG_CLIENT_SERIAL  MSG_CLIENT_SERIAL_LOG  MSG_CLIENT_STATUS  MSG_SERVER_ACKNOWLEDGE  i  [TYPE]  long deviceId  [TYPE]  Channel channel  [TYPE]  ChannelBuffer reply 
[REPLACE]^reply.writeByte ( 'C' )  ;^48^^^^^46^69^[REPLACE] reply.writeByte ( 'M' ) ;^[METHOD] sendReply [TYPE] void [PARAMETER] Channel channel long deviceId byte packetNumber [CLASS] CellocatorProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  byte checksum  commandCount  packetNumber  [TYPE]  int MSG_CLIENT_MODULAR  MSG_CLIENT_PROGRAMMING  MSG_CLIENT_SERIAL  MSG_CLIENT_SERIAL_LOG  MSG_CLIENT_STATUS  MSG_SERVER_ACKNOWLEDGE  i  [TYPE]  long deviceId  [TYPE]  Channel channel  [TYPE]  ChannelBuffer reply 
[REPLACE]^reply.writeByte ( 'M' )  ;^49^^^^^46^69^[REPLACE] reply.writeByte ( 'C' ) ;^[METHOD] sendReply [TYPE] void [PARAMETER] Channel channel long deviceId byte packetNumber [CLASS] CellocatorProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  byte checksum  commandCount  packetNumber  [TYPE]  int MSG_CLIENT_MODULAR  MSG_CLIENT_PROGRAMMING  MSG_CLIENT_SERIAL  MSG_CLIENT_SERIAL_LOG  MSG_CLIENT_STATUS  MSG_SERVER_ACKNOWLEDGE  i  [TYPE]  long deviceId  [TYPE]  Channel channel  [TYPE]  ChannelBuffer reply 
[REMOVE]^reply.writeByte ( 'M' )  ;^49^^^^^46^69^[REMOVE] ^[METHOD] sendReply [TYPE] void [PARAMETER] Channel channel long deviceId byte packetNumber [CLASS] CellocatorProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  byte checksum  commandCount  packetNumber  [TYPE]  int MSG_CLIENT_MODULAR  MSG_CLIENT_PROGRAMMING  MSG_CLIENT_SERIAL  MSG_CLIENT_SERIAL_LOG  MSG_CLIENT_STATUS  MSG_SERVER_ACKNOWLEDGE  i  [TYPE]  long deviceId  [TYPE]  Channel channel  [TYPE]  ChannelBuffer reply 
[REPLACE]^reply .writeInt ( MSG_CLIENT_STATUS )  ;^50^^^^^46^69^[REPLACE] reply.writeByte ( 'G' ) ;^[METHOD] sendReply [TYPE] void [PARAMETER] Channel channel long deviceId byte packetNumber [CLASS] CellocatorProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  byte checksum  commandCount  packetNumber  [TYPE]  int MSG_CLIENT_MODULAR  MSG_CLIENT_PROGRAMMING  MSG_CLIENT_SERIAL  MSG_CLIENT_SERIAL_LOG  MSG_CLIENT_STATUS  MSG_SERVER_ACKNOWLEDGE  i  [TYPE]  long deviceId  [TYPE]  Channel channel  [TYPE]  ChannelBuffer reply 
[REPLACE]^reply .writeInt ( MSG_CLIENT_SERIAL )  ;^51^^^^^46^69^[REPLACE] reply.writeByte ( 'P' ) ;^[METHOD] sendReply [TYPE] void [PARAMETER] Channel channel long deviceId byte packetNumber [CLASS] CellocatorProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  byte checksum  commandCount  packetNumber  [TYPE]  int MSG_CLIENT_MODULAR  MSG_CLIENT_PROGRAMMING  MSG_CLIENT_SERIAL  MSG_CLIENT_SERIAL_LOG  MSG_CLIENT_STATUS  MSG_SERVER_ACKNOWLEDGE  i  [TYPE]  long deviceId  [TYPE]  Channel channel  [TYPE]  ChannelBuffer reply 
[REPLACE]^reply .writeInt ( MSG_CLIENT_PROGRAMMING )  ;^52^^^^^46^69^[REPLACE] reply.writeByte ( MSG_SERVER_ACKNOWLEDGE ) ;^[METHOD] sendReply [TYPE] void [PARAMETER] Channel channel long deviceId byte packetNumber [CLASS] CellocatorProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  byte checksum  commandCount  packetNumber  [TYPE]  int MSG_CLIENT_MODULAR  MSG_CLIENT_PROGRAMMING  MSG_CLIENT_SERIAL  MSG_CLIENT_SERIAL_LOG  MSG_CLIENT_STATUS  MSG_SERVER_ACKNOWLEDGE  i  [TYPE]  long deviceId  [TYPE]  Channel channel  [TYPE]  ChannelBuffer reply 
[REPLACE]^reply .writeByte ( true )  ;^53^^^^^46^69^[REPLACE] reply.writeInt (  ( int )  deviceId ) ;^[METHOD] sendReply [TYPE] void [PARAMETER] Channel channel long deviceId byte packetNumber [CLASS] CellocatorProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  byte checksum  commandCount  packetNumber  [TYPE]  int MSG_CLIENT_MODULAR  MSG_CLIENT_PROGRAMMING  MSG_CLIENT_SERIAL  MSG_CLIENT_SERIAL_LOG  MSG_CLIENT_STATUS  MSG_SERVER_ACKNOWLEDGE  i  [TYPE]  long deviceId  [TYPE]  Channel channel  [TYPE]  ChannelBuffer reply 
[REMOVE]^reply.writeByte (  (  ( commandCount ) ++ )  )  ;^53^^^^^46^69^[REMOVE] ^[METHOD] sendReply [TYPE] void [PARAMETER] Channel channel long deviceId byte packetNumber [CLASS] CellocatorProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  byte checksum  commandCount  packetNumber  [TYPE]  int MSG_CLIENT_MODULAR  MSG_CLIENT_PROGRAMMING  MSG_CLIENT_SERIAL  MSG_CLIENT_SERIAL_LOG  MSG_CLIENT_STATUS  MSG_SERVER_ACKNOWLEDGE  i  [TYPE]  long deviceId  [TYPE]  Channel channel  [TYPE]  ChannelBuffer reply 
[REPLACE]^reply .writeByte ( null )  ;^54^^^^^46^69^[REPLACE] reply.writeByte ( commandCount++ ) ;^[METHOD] sendReply [TYPE] void [PARAMETER] Channel channel long deviceId byte packetNumber [CLASS] CellocatorProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  byte checksum  commandCount  packetNumber  [TYPE]  int MSG_CLIENT_MODULAR  MSG_CLIENT_PROGRAMMING  MSG_CLIENT_SERIAL  MSG_CLIENT_SERIAL_LOG  MSG_CLIENT_STATUS  MSG_SERVER_ACKNOWLEDGE  i  [TYPE]  long deviceId  [TYPE]  Channel channel  [TYPE]  ChannelBuffer reply 
[REPLACE]^reply.writeByte ( 0 )  ;^55^^^^^46^69^[REPLACE] reply.writeInt ( 0 ) ;^[METHOD] sendReply [TYPE] void [PARAMETER] Channel channel long deviceId byte packetNumber [CLASS] CellocatorProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  byte checksum  commandCount  packetNumber  [TYPE]  int MSG_CLIENT_MODULAR  MSG_CLIENT_PROGRAMMING  MSG_CLIENT_SERIAL  MSG_CLIENT_SERIAL_LOG  MSG_CLIENT_STATUS  MSG_SERVER_ACKNOWLEDGE  i  [TYPE]  long deviceId  [TYPE]  Channel channel  [TYPE]  ChannelBuffer reply 
[REPLACE]^reply.writeByte ( 0 << 0 ) ;^56^^^^^46^69^[REPLACE] reply.writeByte ( 0 ) ;^[METHOD] sendReply [TYPE] void [PARAMETER] Channel channel long deviceId byte packetNumber [CLASS] CellocatorProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  byte checksum  commandCount  packetNumber  [TYPE]  int MSG_CLIENT_MODULAR  MSG_CLIENT_PROGRAMMING  MSG_CLIENT_SERIAL  MSG_CLIENT_SERIAL_LOG  MSG_CLIENT_STATUS  MSG_SERVER_ACKNOWLEDGE  i  [TYPE]  long deviceId  [TYPE]  Channel channel  [TYPE]  ChannelBuffer reply 
[REPLACE]^reply.writeByte ( checksum )  ;^57^^^^^46^69^[REPLACE] reply.writeByte ( packetNumber ) ;^[METHOD] sendReply [TYPE] void [PARAMETER] Channel channel long deviceId byte packetNumber [CLASS] CellocatorProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  byte checksum  commandCount  packetNumber  [TYPE]  int MSG_CLIENT_MODULAR  MSG_CLIENT_PROGRAMMING  MSG_CLIENT_SERIAL  MSG_CLIENT_SERIAL_LOG  MSG_CLIENT_STATUS  MSG_SERVER_ACKNOWLEDGE  i  [TYPE]  long deviceId  [TYPE]  Channel channel  [TYPE]  ChannelBuffer reply 
[REPLACE]^reply.writeZero ( 12 ) ;^58^^^^^46^69^[REPLACE] reply.writeZero ( 11 ) ;^[METHOD] sendReply [TYPE] void [PARAMETER] Channel channel long deviceId byte packetNumber [CLASS] CellocatorProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  byte checksum  commandCount  packetNumber  [TYPE]  int MSG_CLIENT_MODULAR  MSG_CLIENT_PROGRAMMING  MSG_CLIENT_SERIAL  MSG_CLIENT_SERIAL_LOG  MSG_CLIENT_STATUS  MSG_SERVER_ACKNOWLEDGE  i  [TYPE]  long deviceId  [TYPE]  Channel channel  [TYPE]  ChannelBuffer reply 
[REPLACE]^byte checksum = 2;^60^^^^^46^69^[REPLACE] byte checksum = 0;^[METHOD] sendReply [TYPE] void [PARAMETER] Channel channel long deviceId byte packetNumber [CLASS] CellocatorProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  byte checksum  commandCount  packetNumber  [TYPE]  int MSG_CLIENT_MODULAR  MSG_CLIENT_PROGRAMMING  MSG_CLIENT_SERIAL  MSG_CLIENT_SERIAL_LOG  MSG_CLIENT_STATUS  MSG_SERVER_ACKNOWLEDGE  i  [TYPE]  long deviceId  [TYPE]  Channel channel  [TYPE]  ChannelBuffer reply 
[REPLACE]^checksum +=  null.getByte ( i ) ;^62^^^^^46^69^[REPLACE] checksum += reply.getByte ( i ) ;^[METHOD] sendReply [TYPE] void [PARAMETER] Channel channel long deviceId byte packetNumber [CLASS] CellocatorProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  byte checksum  commandCount  packetNumber  [TYPE]  int MSG_CLIENT_MODULAR  MSG_CLIENT_PROGRAMMING  MSG_CLIENT_SERIAL  MSG_CLIENT_SERIAL_LOG  MSG_CLIENT_STATUS  MSG_SERVER_ACKNOWLEDGE  i  [TYPE]  long deviceId  [TYPE]  Channel channel  [TYPE]  ChannelBuffer reply 
[ADD]^^61^62^63^^^46^69^[ADD] for  ( int i = 4; i < 27; i++ )  { checksum += reply.getByte ( i ) ; }^[METHOD] sendReply [TYPE] void [PARAMETER] Channel channel long deviceId byte packetNumber [CLASS] CellocatorProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  byte checksum  commandCount  packetNumber  [TYPE]  int MSG_CLIENT_MODULAR  MSG_CLIENT_PROGRAMMING  MSG_CLIENT_SERIAL  MSG_CLIENT_SERIAL_LOG  MSG_CLIENT_STATUS  MSG_SERVER_ACKNOWLEDGE  i  [TYPE]  long deviceId  [TYPE]  Channel channel  [TYPE]  ChannelBuffer reply 
[REPLACE]^reply.writeByte ( 'C' )  ;^64^^^^^46^69^[REPLACE] reply.writeByte ( checksum ) ;^[METHOD] sendReply [TYPE] void [PARAMETER] Channel channel long deviceId byte packetNumber [CLASS] CellocatorProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  byte checksum  commandCount  packetNumber  [TYPE]  int MSG_CLIENT_MODULAR  MSG_CLIENT_PROGRAMMING  MSG_CLIENT_SERIAL  MSG_CLIENT_SERIAL_LOG  MSG_CLIENT_STATUS  MSG_SERVER_ACKNOWLEDGE  i  [TYPE]  long deviceId  [TYPE]  Channel channel  [TYPE]  ChannelBuffer reply 
[REPLACE]^if  ( channel == true )  {^66^^^^^46^69^[REPLACE] if  ( channel != null )  {^[METHOD] sendReply [TYPE] void [PARAMETER] Channel channel long deviceId byte packetNumber [CLASS] CellocatorProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  byte checksum  commandCount  packetNumber  [TYPE]  int MSG_CLIENT_MODULAR  MSG_CLIENT_PROGRAMMING  MSG_CLIENT_SERIAL  MSG_CLIENT_SERIAL_LOG  MSG_CLIENT_STATUS  MSG_SERVER_ACKNOWLEDGE  i  [TYPE]  long deviceId  [TYPE]  Channel channel  [TYPE]  ChannelBuffer reply 
[ADD]^^66^67^68^^^46^69^[ADD] if  ( channel != null )  { channel.write ( reply ) ; }^[METHOD] sendReply [TYPE] void [PARAMETER] Channel channel long deviceId byte packetNumber [CLASS] CellocatorProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  byte checksum  commandCount  packetNumber  [TYPE]  int MSG_CLIENT_MODULAR  MSG_CLIENT_PROGRAMMING  MSG_CLIENT_SERIAL  MSG_CLIENT_SERIAL_LOG  MSG_CLIENT_STATUS  MSG_SERVER_ACKNOWLEDGE  i  [TYPE]  long deviceId  [TYPE]  Channel channel  [TYPE]  ChannelBuffer reply 
[ADD]^^75^^^^^73^143^[ADD] ChannelBuffer buf =  ( ChannelBuffer )  msg;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CellocatorProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  byte checksum  commandCount  packetNumber  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_CLIENT_MODULAR  MSG_CLIENT_PROGRAMMING  MSG_CLIENT_SERIAL  MSG_CLIENT_SERIAL_LOG  MSG_CLIENT_STATUS  MSG_SERVER_ACKNOWLEDGE  i  operator  type  [TYPE]  long deviceUniqueId  [TYPE]  ChannelBuffer buf 
[REPLACE]^buf.skipBytes ( 6 )  ;^77^^^^^73^143^[REPLACE] buf.skipBytes ( 4 ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CellocatorProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  byte checksum  commandCount  packetNumber  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_CLIENT_MODULAR  MSG_CLIENT_PROGRAMMING  MSG_CLIENT_SERIAL  MSG_CLIENT_SERIAL_LOG  MSG_CLIENT_STATUS  MSG_SERVER_ACKNOWLEDGE  i  operator  type  [TYPE]  long deviceUniqueId  [TYPE]  ChannelBuffer buf 
[ADD]^^78^^^^^73^143^[ADD] int type = buf.readUnsignedByte (  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CellocatorProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  byte checksum  commandCount  packetNumber  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_CLIENT_MODULAR  MSG_CLIENT_PROGRAMMING  MSG_CLIENT_SERIAL  MSG_CLIENT_SERIAL_LOG  MSG_CLIENT_STATUS  MSG_SERVER_ACKNOWLEDGE  i  operator  type  [TYPE]  long deviceUniqueId  [TYPE]  ChannelBuffer buf 
[REPLACE]^long deviceUniqueId = buf.readUnsignedByte (  ) ;^79^^^^^73^143^[REPLACE] long deviceUniqueId = buf.readUnsignedInt (  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CellocatorProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  byte checksum  commandCount  packetNumber  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_CLIENT_MODULAR  MSG_CLIENT_PROGRAMMING  MSG_CLIENT_SERIAL  MSG_CLIENT_SERIAL_LOG  MSG_CLIENT_STATUS  MSG_SERVER_ACKNOWLEDGE  i  operator  type  [TYPE]  long deviceUniqueId  [TYPE]  ChannelBuffer buf 
[REPLACE]^if  ( type  ==  MSG_CLIENT_SERIAL  || type ==  ( MSG_CLIENT_STATUS )  )  {^81^^^^^73^143^[REPLACE] if  ( type != MSG_CLIENT_SERIAL )  {^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CellocatorProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  byte checksum  commandCount  packetNumber  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_CLIENT_MODULAR  MSG_CLIENT_PROGRAMMING  MSG_CLIENT_SERIAL  MSG_CLIENT_SERIAL_LOG  MSG_CLIENT_STATUS  MSG_SERVER_ACKNOWLEDGE  i  operator  type  [TYPE]  long deviceUniqueId  [TYPE]  ChannelBuffer buf 
[REPLACE]^buf.readUnsignedInt (  ) ;^82^^^^^73^143^[REPLACE] buf.readUnsignedShort (  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CellocatorProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  byte checksum  commandCount  packetNumber  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_CLIENT_MODULAR  MSG_CLIENT_PROGRAMMING  MSG_CLIENT_SERIAL  MSG_CLIENT_SERIAL_LOG  MSG_CLIENT_STATUS  MSG_SERVER_ACKNOWLEDGE  i  operator  type  [TYPE]  long deviceUniqueId  [TYPE]  ChannelBuffer buf 
[REPLACE]^byte packetNumber = buf.readInt (  ) ;^84^^^^^73^143^[REPLACE] byte packetNumber = buf.readByte (  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CellocatorProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  byte checksum  commandCount  packetNumber  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_CLIENT_MODULAR  MSG_CLIENT_PROGRAMMING  MSG_CLIENT_SERIAL  MSG_CLIENT_SERIAL_LOG  MSG_CLIENT_STATUS  MSG_SERVER_ACKNOWLEDGE  i  operator  type  [TYPE]  long deviceUniqueId  [TYPE]  ChannelBuffer buf 
[REPLACE]^sendReply ( channel, deviceUniqueId, checksum ) ;^86^^^^^73^143^[REPLACE] sendReply ( channel, deviceUniqueId, packetNumber ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CellocatorProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  byte checksum  commandCount  packetNumber  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_CLIENT_MODULAR  MSG_CLIENT_PROGRAMMING  MSG_CLIENT_SERIAL  MSG_CLIENT_SERIAL_LOG  MSG_CLIENT_STATUS  MSG_SERVER_ACKNOWLEDGE  i  operator  type  [TYPE]  long deviceUniqueId  [TYPE]  ChannelBuffer buf 
[ADD]^^86^^^^^73^143^[ADD] sendReply ( channel, deviceUniqueId, packetNumber ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CellocatorProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  byte checksum  commandCount  packetNumber  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_CLIENT_MODULAR  MSG_CLIENT_PROGRAMMING  MSG_CLIENT_SERIAL  MSG_CLIENT_SERIAL_LOG  MSG_CLIENT_STATUS  MSG_SERVER_ACKNOWLEDGE  i  operator  type  [TYPE]  long deviceUniqueId  [TYPE]  ChannelBuffer buf 
[REPLACE]^if  ( type  && MSG_SERVER_ACKNOWLEDGE )  {^88^^^^^73^143^[REPLACE] if  ( type == MSG_CLIENT_STATUS )  {^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CellocatorProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  byte checksum  commandCount  packetNumber  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_CLIENT_MODULAR  MSG_CLIENT_PROGRAMMING  MSG_CLIENT_SERIAL  MSG_CLIENT_SERIAL_LOG  MSG_CLIENT_STATUS  MSG_SERVER_ACKNOWLEDGE  i  operator  type  [TYPE]  long deviceUniqueId  [TYPE]  ChannelBuffer buf 
[REPLACE]^if  ( deviceSession != this )  {^94^^^^^73^143^[REPLACE] if  ( deviceSession == null )  {^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CellocatorProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  byte checksum  commandCount  packetNumber  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_CLIENT_MODULAR  MSG_CLIENT_PROGRAMMING  MSG_CLIENT_SERIAL  MSG_CLIENT_SERIAL_LOG  MSG_CLIENT_STATUS  MSG_SERVER_ACKNOWLEDGE  i  operator  type  [TYPE]  long deviceUniqueId  [TYPE]  ChannelBuffer buf 
[REPLACE]^return this;^95^^^^^73^143^[REPLACE] return null;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CellocatorProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  byte checksum  commandCount  packetNumber  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_CLIENT_MODULAR  MSG_CLIENT_PROGRAMMING  MSG_CLIENT_SERIAL  MSG_CLIENT_SERIAL_LOG  MSG_CLIENT_STATUS  MSG_SERVER_ACKNOWLEDGE  i  operator  type  [TYPE]  long deviceUniqueId  [TYPE]  ChannelBuffer buf 
[REPLACE]^operator +=  null.readUnsignedByte (  ) ;^106^^^^^73^143^[REPLACE] operator += buf.readUnsignedByte (  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CellocatorProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  byte checksum  commandCount  packetNumber  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_CLIENT_MODULAR  MSG_CLIENT_PROGRAMMING  MSG_CLIENT_SERIAL  MSG_CLIENT_SERIAL_LOG  MSG_CLIENT_STATUS  MSG_SERVER_ACKNOWLEDGE  i  operator  type  [TYPE]  long deviceUniqueId  [TYPE]  ChannelBuffer buf 
[REPLACE]^operator <<= 0;^113^^^^^73^143^[REPLACE] operator <<= 8;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CellocatorProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  byte checksum  commandCount  packetNumber  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_CLIENT_MODULAR  MSG_CLIENT_PROGRAMMING  MSG_CLIENT_SERIAL  MSG_CLIENT_SERIAL_LOG  MSG_CLIENT_STATUS  MSG_SERVER_ACKNOWLEDGE  i  operator  type  [TYPE]  long deviceUniqueId  [TYPE]  ChannelBuffer buf 
[REPLACE]^operator +=  null.readUnsignedByte (  ) ;^114^^^^^73^143^[REPLACE] operator += buf.readUnsignedByte (  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CellocatorProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  byte checksum  commandCount  packetNumber  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_CLIENT_MODULAR  MSG_CLIENT_PROGRAMMING  MSG_CLIENT_SERIAL  MSG_CLIENT_SERIAL_LOG  MSG_CLIENT_STATUS  MSG_SERVER_ACKNOWLEDGE  i  operator  type  [TYPE]  long deviceUniqueId  [TYPE]  ChannelBuffer buf 
[REPLACE]^int operator =  ( buf.readUnsignedByte (  )  & 0xf0 )   <=  4;^105^^^^^73^143^[REPLACE] int operator =  ( buf.readUnsignedByte (  )  & 0xf0 )  << 4;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CellocatorProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  byte checksum  commandCount  packetNumber  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_CLIENT_MODULAR  MSG_CLIENT_PROGRAMMING  MSG_CLIENT_SERIAL  MSG_CLIENT_SERIAL_LOG  MSG_CLIENT_STATUS  MSG_SERVER_ACKNOWLEDGE  i  operator  type  [TYPE]  long deviceUniqueId  [TYPE]  ChannelBuffer buf 
[REPLACE]^DateBuilder dateBuilder = new DateBuilder (  ) .setTimeReverse ( buf.readUnsignedInt (  ) , buf.readUnsignedByte (  ) , buf.readUnsignedByte (  )  ) .setDateReverse ( buf.readUnsignedByte (  ) , buf.readUnsignedByte (  ) , buf.readUnsignedShort (  )  ) ;^134^135^136^^^73^143^[REPLACE] DateBuilder dateBuilder = new DateBuilder (  ) .setTimeReverse ( buf.readUnsignedByte (  ) , buf.readUnsignedByte (  ) , buf.readUnsignedByte (  )  ) .setDateReverse ( buf.readUnsignedByte (  ) , buf.readUnsignedByte (  ) , buf.readUnsignedShort (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CellocatorProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  byte checksum  commandCount  packetNumber  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_CLIENT_MODULAR  MSG_CLIENT_PROGRAMMING  MSG_CLIENT_SERIAL  MSG_CLIENT_SERIAL_LOG  MSG_CLIENT_STATUS  MSG_SERVER_ACKNOWLEDGE  i  operator  type  [TYPE]  long deviceUniqueId  [TYPE]  ChannelBuffer buf 
[REPLACE]^if  ( deviceSession != null )  {^94^^^^^73^143^[REPLACE] if  ( deviceSession == null )  {^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CellocatorProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  byte checksum  commandCount  packetNumber  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_CLIENT_MODULAR  MSG_CLIENT_PROGRAMMING  MSG_CLIENT_SERIAL  MSG_CLIENT_SERIAL_LOG  MSG_CLIENT_STATUS  MSG_SERVER_ACKNOWLEDGE  i  operator  type  [TYPE]  long deviceUniqueId  [TYPE]  ChannelBuffer buf 
[REPLACE]^return true;^95^^^^^73^143^[REPLACE] return null;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CellocatorProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  byte checksum  commandCount  packetNumber  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_CLIENT_MODULAR  MSG_CLIENT_PROGRAMMING  MSG_CLIENT_SERIAL  MSG_CLIENT_SERIAL_LOG  MSG_CLIENT_STATUS  MSG_SERVER_ACKNOWLEDGE  i  operator  type  [TYPE]  long deviceUniqueId  [TYPE]  ChannelBuffer buf 
[ADD]^^113^^^^^73^143^[ADD] operator <<= 8;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CellocatorProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  byte checksum  commandCount  packetNumber  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_CLIENT_MODULAR  MSG_CLIENT_PROGRAMMING  MSG_CLIENT_SERIAL  MSG_CLIENT_SERIAL_LOG  MSG_CLIENT_STATUS  MSG_SERVER_ACKNOWLEDGE  i  operator  type  [TYPE]  long deviceUniqueId  [TYPE]  ChannelBuffer buf 
[REPLACE]^int operator =  ( buf.readUnsignedByte (  )  & 0xf0 )   >  4;^105^^^^^73^143^[REPLACE] int operator =  ( buf.readUnsignedByte (  )  & 0xf0 )  << 4;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CellocatorProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  byte checksum  commandCount  packetNumber  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_CLIENT_MODULAR  MSG_CLIENT_PROGRAMMING  MSG_CLIENT_SERIAL  MSG_CLIENT_SERIAL_LOG  MSG_CLIENT_STATUS  MSG_SERVER_ACKNOWLEDGE  i  operator  type  [TYPE]  long deviceUniqueId  [TYPE]  ChannelBuffer buf 
[ADD]^^134^135^136^^^73^143^[ADD] DateBuilder dateBuilder = new DateBuilder (  ) .setTimeReverse ( buf.readUnsignedByte (  ) , buf.readUnsignedByte (  ) , buf.readUnsignedByte (  )  ) .setDateReverse ( buf.readUnsignedByte (  ) , buf.readUnsignedByte (  ) , buf.readUnsignedShort (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CellocatorProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  byte checksum  commandCount  packetNumber  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_CLIENT_MODULAR  MSG_CLIENT_PROGRAMMING  MSG_CLIENT_SERIAL  MSG_CLIENT_SERIAL_LOG  MSG_CLIENT_STATUS  MSG_SERVER_ACKNOWLEDGE  i  operator  type  [TYPE]  long deviceUniqueId  [TYPE]  ChannelBuffer buf 
[REPLACE]^return false;^142^^^^^73^143^[REPLACE] return null;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CellocatorProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  byte checksum  commandCount  packetNumber  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_CLIENT_MODULAR  MSG_CLIENT_PROGRAMMING  MSG_CLIENT_SERIAL  MSG_CLIENT_SERIAL_LOG  MSG_CLIENT_STATUS  MSG_SERVER_ACKNOWLEDGE  i  operator  type  [TYPE]  long deviceUniqueId  [TYPE]  ChannelBuffer buf 
