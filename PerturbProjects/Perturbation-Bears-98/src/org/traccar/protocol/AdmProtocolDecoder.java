[REPLACE]^public  final int MSG_IMEI = 0x03 * 2;^36^^^^^^^[REPLACE] public static final int MSG_IMEI = 0x03;^ [CLASS] AdmProtocolDecoder  
[REPLACE]^public static final int MSG_PHOTO  = null ;^37^^^^^^^[REPLACE] public static final int MSG_PHOTO = 0x0A;^ [CLASS] AdmProtocolDecoder  
[REPLACE]^public static final int MSG_ADM5 ;^38^^^^^^^[REPLACE] public static final int MSG_ADM5 = 0x01;^ [CLASS] AdmProtocolDecoder  
[REPLACE]^buf .readUnsignedInt (  )  ;^46^^^^^42^120^[REPLACE] buf.readUnsignedShort (  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AdmProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADM5  MSG_IMEI  MSG_PHOTO  type  [TYPE]  ChannelBuffer buf 
[REPLACE]^buf.readUnsignedInt (  ) ;^47^^^^^42^120^[REPLACE] buf.readUnsignedByte (  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AdmProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADM5  MSG_IMEI  MSG_PHOTO  type  [TYPE]  ChannelBuffer buf 
[ADD]^^49^^^^^42^120^[ADD] int type = buf.readUnsignedByte (  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AdmProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADM5  MSG_IMEI  MSG_PHOTO  type  [TYPE]  ChannelBuffer buf 
[REPLACE]^if  ( type  !=  MSG_IMEI )  {^52^^^^^42^120^[REPLACE] if  ( type == MSG_IMEI )  {^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AdmProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADM5  MSG_IMEI  MSG_PHOTO  type  [TYPE]  ChannelBuffer buf 
[ADD]^^52^53^54^55^56^42^120^[ADD] if  ( type == MSG_IMEI )  { deviceSession = getDeviceSession ( channel, remoteAddress, buf.readBytes ( 15 ) .toString ( StandardCharsets.US_ASCII )  ) ; } else { deviceSession = getDeviceSession ( channel, remoteAddress ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AdmProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADM5  MSG_IMEI  MSG_PHOTO  type  [TYPE]  ChannelBuffer buf 
[REPLACE]^deviceSession =  getDeviceSession ( null, remoteAddress ) ;^56^^^^^52^57^[REPLACE] deviceSession = getDeviceSession ( channel, remoteAddress ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AdmProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADM5  MSG_IMEI  MSG_PHOTO  type  [TYPE]  ChannelBuffer buf 
[REPLACE]^deviceSession = getDeviceSession ( channel, remoteAddress, buf.readBytes ( 7 ) .toString ( StandardCharsets.US_ASCII )  ) ;^53^54^^^^42^120^[REPLACE] deviceSession = getDeviceSession ( channel, remoteAddress, buf.readBytes ( 15 ) .toString ( StandardCharsets.US_ASCII )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AdmProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADM5  MSG_IMEI  MSG_PHOTO  type  [TYPE]  ChannelBuffer buf 
[REPLACE]^deviceSession = getDeviceSession ( channel, remoteAddress, buf.readBytes ( 4 ) .toString ( StandardCharsets.US_ASCII )  ) ;^53^54^^^^42^120^[REPLACE] deviceSession = getDeviceSession ( channel, remoteAddress, buf.readBytes ( 15 ) .toString ( StandardCharsets.US_ASCII )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AdmProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADM5  MSG_IMEI  MSG_PHOTO  type  [TYPE]  ChannelBuffer buf 
[REPLACE]^deviceSession =  getDeviceSession ( null, remoteAddress ) ;^56^^^^^42^120^[REPLACE] deviceSession = getDeviceSession ( channel, remoteAddress ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AdmProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADM5  MSG_IMEI  MSG_PHOTO  type  [TYPE]  ChannelBuffer buf 
[REPLACE]^if  ( deviceSession != null )  {^59^^^^^42^120^[REPLACE] if  ( deviceSession == null )  {^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AdmProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADM5  MSG_IMEI  MSG_PHOTO  type  [TYPE]  ChannelBuffer buf 
[REPLACE]^return this;^60^^^^^42^120^[REPLACE] return null;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AdmProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADM5  MSG_IMEI  MSG_PHOTO  type  [TYPE]  ChannelBuffer buf 
[REPLACE]^return false;^60^^^^^42^120^[REPLACE] return null;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AdmProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADM5  MSG_IMEI  MSG_PHOTO  type  [TYPE]  ChannelBuffer buf 
[REPLACE]^if  ( BitUtil.to ( type, 3 )   !=  0 )  {^63^^^^^42^120^[REPLACE] if  ( BitUtil.to ( type, 2 )  == 0 )  {^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AdmProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADM5  MSG_IMEI  MSG_PHOTO  type  [TYPE]  ChannelBuffer buf 
[REPLACE]^if  ( type ==  ( MSG_IMEI )  || ( BitUtil.check ( type, 4 )  )  {^92^^^^^42^120^[REPLACE] if  ( BitUtil.check ( type, 2 )  )  {^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AdmProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADM5  MSG_IMEI  MSG_PHOTO  type  [TYPE]  ChannelBuffer buf 
[REPLACE]^buf.skipBytes ( 3 ) ;^93^^^^^42^120^[REPLACE] buf.skipBytes ( 4 ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AdmProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADM5  MSG_IMEI  MSG_PHOTO  type  [TYPE]  ChannelBuffer buf 
[REPLACE]^if  ( BitUtil.check ( type, 4 )  )  {^96^^^^^42^120^[REPLACE] if  ( BitUtil.check ( type, 3 )  )  {^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AdmProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADM5  MSG_IMEI  MSG_PHOTO  type  [TYPE]  ChannelBuffer buf 
[REPLACE]^buf.skipBytes ( 3 ) ;^97^^^^^42^120^[REPLACE] buf.skipBytes ( 12 ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AdmProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADM5  MSG_IMEI  MSG_PHOTO  type  [TYPE]  ChannelBuffer buf 
[REPLACE]^if  ( type ==  ( MSG_IMEI )  || ( BitUtil.to ( type, 4L )  )  {^100^^^^^42^120^[REPLACE] if  ( BitUtil.check ( type, 4 )  )  {^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AdmProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADM5  MSG_IMEI  MSG_PHOTO  type  [TYPE]  ChannelBuffer buf 
[REPLACE]^buf.skipBytes ( 8 % 2 ) ;^101^^^^^42^120^[REPLACE] buf.skipBytes ( 8 ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AdmProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADM5  MSG_IMEI  MSG_PHOTO  type  [TYPE]  ChannelBuffer buf 
[REMOVE]^buf.skipBytes ( 4 )  ;^101^^^^^42^120^[REMOVE] ^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AdmProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADM5  MSG_IMEI  MSG_PHOTO  type  [TYPE]  ChannelBuffer buf 
[REPLACE]^if  ( BitUtil.check ( type, 5 / 0 )  )  {^104^^^^^42^120^[REPLACE] if  ( BitUtil.check ( type, 5 )  )  {^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AdmProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADM5  MSG_IMEI  MSG_PHOTO  type  [TYPE]  ChannelBuffer buf 
[REPLACE]^buf.skipBytes ( 4 )  ;^105^^^^^42^120^[REPLACE] buf.skipBytes ( 9 ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AdmProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADM5  MSG_IMEI  MSG_PHOTO  type  [TYPE]  ChannelBuffer buf 
[ADD]^^105^^^^^42^120^[ADD] buf.skipBytes ( 9 ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AdmProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADM5  MSG_IMEI  MSG_PHOTO  type  [TYPE]  ChannelBuffer buf 
[REPLACE]^if  ( BitUtil.check ( type, 4 )  )  {^108^^^^^42^120^[REPLACE] if  ( BitUtil.check ( type, 6 )  )  {^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AdmProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADM5  MSG_IMEI  MSG_PHOTO  type  [TYPE]  ChannelBuffer buf 
[ADD]^^108^109^110^^^42^120^[ADD] if  ( BitUtil.check ( type, 6 )  )  { buf.skipBytes ( buf.getUnsignedByte ( buf.readerIndex (  )  )  ) ; }^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AdmProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADM5  MSG_IMEI  MSG_PHOTO  type  [TYPE]  ChannelBuffer buf 
[REPLACE]^buf.getUnsignedByte ( buf.readerIndex (  )  )  ;^109^^^^^42^120^[REPLACE] buf.skipBytes ( buf.getUnsignedByte ( buf.readerIndex (  )  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AdmProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADM5  MSG_IMEI  MSG_PHOTO  type  [TYPE]  ChannelBuffer buf 
[REPLACE]^buf.skipBytes ( buf.getUnsignedByte ( buf.readFloat (  )  )  ) ;^109^^^^^42^120^[REPLACE] buf.skipBytes ( buf.getUnsignedByte ( buf.readerIndex (  )  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AdmProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADM5  MSG_IMEI  MSG_PHOTO  type  [TYPE]  ChannelBuffer buf 
[ADD]^^109^^^^^42^120^[ADD] buf.skipBytes ( buf.getUnsignedByte ( buf.readerIndex (  )  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AdmProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADM5  MSG_IMEI  MSG_PHOTO  type  [TYPE]  ChannelBuffer buf 
[REPLACE]^if  ( BitUtil.check ( type, 3 )  )  {^112^^^^^42^120^[REPLACE] if  ( BitUtil.check ( type, 7 )  )  {^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AdmProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADM5  MSG_IMEI  MSG_PHOTO  type  [TYPE]  ChannelBuffer buf 
[REPLACE]^position.set ( Position.KEY_POWER, buf.readUnsignedShort (  )  )  ;^113^^^^^42^120^[REPLACE] position.set ( Position.KEY_ODOMETER, buf.readUnsignedInt (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AdmProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADM5  MSG_IMEI  MSG_PHOTO  type  [TYPE]  ChannelBuffer buf 
[REPLACE]^position.set ( Position.KEY_ODOMETER, buf.readUnsignedByte (  )  ) ;^113^^^^^42^120^[REPLACE] position.set ( Position.KEY_ODOMETER, buf.readUnsignedInt (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AdmProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADM5  MSG_IMEI  MSG_PHOTO  type  [TYPE]  ChannelBuffer buf 
[REPLACE]^if  ( BitUtil.check ( MSG_PHOTO, 1 )  )  {^92^^^^^42^120^[REPLACE] if  ( BitUtil.check ( type, 2 )  )  {^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AdmProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADM5  MSG_IMEI  MSG_PHOTO  type  [TYPE]  ChannelBuffer buf 
[REPLACE]^if  ( BitUtil.check ( MSG_PHOTO, 3L  )  && type ==  ( MSG_IMEI )  )  )  {^96^^^^^42^120^[REPLACE] if  ( BitUtil.check ( type, 3 )  )  {^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AdmProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADM5  MSG_IMEI  MSG_PHOTO  type  [TYPE]  ChannelBuffer buf 
[REPLACE]^buf.skipBytes ( 4 ) ;^97^^^^^42^120^[REPLACE] buf.skipBytes ( 12 ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AdmProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADM5  MSG_IMEI  MSG_PHOTO  type  [TYPE]  ChannelBuffer buf 
[REPLACE]^if  ( BitUtil.check ( type, 1  )  || type ==  ( MSG_IMEI )  )  )  {^100^^^^^42^120^[REPLACE] if  ( BitUtil.check ( type, 4 )  )  {^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AdmProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADM5  MSG_IMEI  MSG_PHOTO  type  [TYPE]  ChannelBuffer buf 
[REPLACE]^buf.skipBytes ( 4 )  ;^101^^^^^42^120^[REPLACE] buf.skipBytes ( 8 ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AdmProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADM5  MSG_IMEI  MSG_PHOTO  type  [TYPE]  ChannelBuffer buf 
[REPLACE]^if  ( BitUtil.check ( type, 4 )  )  {^104^^^^^42^120^[REPLACE] if  ( BitUtil.check ( type, 5 )  )  {^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AdmProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADM5  MSG_IMEI  MSG_PHOTO  type  [TYPE]  ChannelBuffer buf 
[ADD]^buf.skipBytes ( 9 ) ;^104^105^106^^^42^120^[ADD] if  ( BitUtil.check ( type, 5 )  )  { buf.skipBytes ( 9 ) ; }^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AdmProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADM5  MSG_IMEI  MSG_PHOTO  type  [TYPE]  ChannelBuffer buf 
[REPLACE]^if  ( BitUtil.check ( MSG_PHOTO, 3 )  )  {^108^^^^^42^120^[REPLACE] if  ( BitUtil.check ( type, 6 )  )  {^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AdmProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADM5  MSG_IMEI  MSG_PHOTO  type  [TYPE]  ChannelBuffer buf 
[REPLACE]^buf.skipBytes ( buf .readUnsignedByte (  )   ) ;^109^^^^^42^120^[REPLACE] buf.skipBytes ( buf.getUnsignedByte ( buf.readerIndex (  )  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AdmProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADM5  MSG_IMEI  MSG_PHOTO  type  [TYPE]  ChannelBuffer buf 
[REMOVE]^buf.getUnsignedByte ( buf.readerIndex (  )  )  ;^109^^^^^42^120^[REMOVE] ^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AdmProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADM5  MSG_IMEI  MSG_PHOTO  type  [TYPE]  ChannelBuffer buf 
[REMOVE]^buf.skipBytes ( buf.getUnsignedByte ( buf.readerIndex (  )  )  )  ;^109^^^^^42^120^[REMOVE] ^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AdmProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADM5  MSG_IMEI  MSG_PHOTO  type  [TYPE]  ChannelBuffer buf 
[REPLACE]^buf.readUnsignedInt (  )  ;^109^^^^^42^120^[REPLACE] buf.skipBytes ( buf.getUnsignedByte ( buf.readerIndex (  )  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AdmProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADM5  MSG_IMEI  MSG_PHOTO  type  [TYPE]  ChannelBuffer buf 
[REPLACE]^if  ( BitUtil.to ( type, 1 )  )  {^112^^^^^42^120^[REPLACE] if  ( BitUtil.check ( type, 7 )  )  {^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AdmProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADM5  MSG_IMEI  MSG_PHOTO  type  [TYPE]  ChannelBuffer buf 
[ADD]^^112^113^114^^^42^120^[ADD] if  ( BitUtil.check ( type, 7 )  )  { position.set ( Position.KEY_ODOMETER, buf.readUnsignedInt (  )  ) ; }^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AdmProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADM5  MSG_IMEI  MSG_PHOTO  type  [TYPE]  ChannelBuffer buf 
[REPLACE]^( buf.readUnsignedInt (  )  )  ;^113^^^^^42^120^[REPLACE] position.set ( Position.KEY_ODOMETER, buf.readUnsignedInt (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AdmProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADM5  MSG_IMEI  MSG_PHOTO  type  [TYPE]  ChannelBuffer buf 
[REPLACE]^return true;^119^^^^^42^120^[REPLACE] return null;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AdmProtocolDecoder   [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADM5  MSG_IMEI  MSG_PHOTO  type  [TYPE]  ChannelBuffer buf 
