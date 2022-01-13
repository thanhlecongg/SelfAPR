[REPLACE]^private  final Logger LOGGER = LogManager.getLogger (  ) ;^11^^^^^^^[REPLACE] private static final Logger LOGGER = LogManager.getLogger (  ) ;^ [CLASS] Main  
[REPLACE]^private static final int CYCLE_PERIOD  = null ;^12^^^^^^^[REPLACE] private static final int CYCLE_PERIOD = 40;^ [CLASS] Main  
[REPLACE]^provide (  ) .getBoolean ( "general.debug" )  ;^22^^^^^19^47^[REPLACE] LOGGER.info ( ConfigProvider.provide (  ) .getBoolean ( "general.debug" )  ) ;^[METHOD] main [TYPE] void [PARAMETER] String[] args [CLASS] Main   [TYPE]  boolean false  true  [TYPE]  AutomatedCar car  [TYPE]  Gui gui  [TYPE]  String[] args  [TYPE]  Logger LOGGER  [TYPE]  World w  [TYPE]  InterruptedException e  [TYPE]  int CYCLE_PERIOD 
[REPLACE]^info ( provide (  ) .getBoolean ( "general.debug" )  )  ;^22^^^^^19^47^[REPLACE] LOGGER.info ( ConfigProvider.provide (  ) .getBoolean ( "general.debug" )  ) ;^[METHOD] main [TYPE] void [PARAMETER] String[] args [CLASS] Main   [TYPE]  boolean false  true  [TYPE]  AutomatedCar car  [TYPE]  Gui gui  [TYPE]  String[] args  [TYPE]  Logger LOGGER  [TYPE]  World w  [TYPE]  InterruptedException e  [TYPE]  int CYCLE_PERIOD 
[ADD]^^22^^^^^19^47^[ADD] LOGGER.info ( ConfigProvider.provide (  ) .getBoolean ( "general.debug" )  ) ;^[METHOD] main [TYPE] void [PARAMETER] String[] args [CLASS] Main   [TYPE]  boolean false  true  [TYPE]  AutomatedCar car  [TYPE]  Gui gui  [TYPE]  String[] args  [TYPE]  Logger LOGGER  [TYPE]  World w  [TYPE]  InterruptedException e  [TYPE]  int CYCLE_PERIOD 
[REPLACE]^World w = new World ( 2, 600 ) ;^25^^^^^19^47^[REPLACE] World w = new World ( 800, 600 ) ;^[METHOD] main [TYPE] void [PARAMETER] String[] args [CLASS] Main   [TYPE]  boolean false  true  [TYPE]  AutomatedCar car  [TYPE]  Gui gui  [TYPE]  String[] args  [TYPE]  Logger LOGGER  [TYPE]  World w  [TYPE]  InterruptedException e  [TYPE]  int CYCLE_PERIOD 
[REPLACE]^AutomatedCar car = new AutomatedCar ( 4, 4, "car_2_white.png" ) ;^27^^^^^19^47^[REPLACE] AutomatedCar car = new AutomatedCar ( 20, 20, "car_2_white.png" ) ;^[METHOD] main [TYPE] void [PARAMETER] String[] args [CLASS] Main   [TYPE]  boolean false  true  [TYPE]  AutomatedCar car  [TYPE]  Gui gui  [TYPE]  String[] args  [TYPE]  Logger LOGGER  [TYPE]  World w  [TYPE]  InterruptedException e  [TYPE]  int CYCLE_PERIOD 
[REPLACE]^w .World ( CYCLE_PERIOD , CYCLE_PERIOD )  ;^29^^^^^19^47^[REPLACE] w.addObjectToWorld ( car ) ;^[METHOD] main [TYPE] void [PARAMETER] String[] args [CLASS] Main   [TYPE]  boolean false  true  [TYPE]  AutomatedCar car  [TYPE]  Gui gui  [TYPE]  String[] args  [TYPE]  Logger LOGGER  [TYPE]  World w  [TYPE]  InterruptedException e  [TYPE]  int CYCLE_PERIOD 
[REPLACE]^gui.getCourseDisplay (  )  ;^35^^^^^19^47^[REPLACE] gui.getCourseDisplay (  ) .drawWorld ( w ) ;^[METHOD] main [TYPE] void [PARAMETER] String[] args [CLASS] Main   [TYPE]  boolean false  true  [TYPE]  AutomatedCar car  [TYPE]  Gui gui  [TYPE]  String[] args  [TYPE]  Logger LOGGER  [TYPE]  World w  [TYPE]  InterruptedException e  [TYPE]  int CYCLE_PERIOD 
[REPLACE]^gui .Gui (  )  .drawWorld ( w ) ;^35^^^^^19^47^[REPLACE] gui.getCourseDisplay (  ) .drawWorld ( w ) ;^[METHOD] main [TYPE] void [PARAMETER] String[] args [CLASS] Main   [TYPE]  boolean false  true  [TYPE]  AutomatedCar car  [TYPE]  Gui gui  [TYPE]  String[] args  [TYPE]  Logger LOGGER  [TYPE]  World w  [TYPE]  InterruptedException e  [TYPE]  int CYCLE_PERIOD 
[REMOVE]^gui.getCourseDisplay (  ) .drawWorld ( w )  ;^35^^^^^19^47^[REMOVE] ^[METHOD] main [TYPE] void [PARAMETER] String[] args [CLASS] Main   [TYPE]  boolean false  true  [TYPE]  AutomatedCar car  [TYPE]  Gui gui  [TYPE]  String[] args  [TYPE]  Logger LOGGER  [TYPE]  World w  [TYPE]  InterruptedException e  [TYPE]  int CYCLE_PERIOD 
[ADD]^car.drive (  ) ;gui.getCourseDisplay (  ) .drawWorld ( w ) ;Thread.sleep ( CYCLE_PERIOD ) ;^38^39^40^41^42^19^47^[ADD] try { car.drive (  ) ; gui.getCourseDisplay (  ) .drawWorld ( w ) ; Thread.sleep ( CYCLE_PERIOD ) ; } catch  ( InterruptedException e )  {^[METHOD] main [TYPE] void [PARAMETER] String[] args [CLASS] Main   [TYPE]  boolean false  true  [TYPE]  AutomatedCar car  [TYPE]  Gui gui  [TYPE]  String[] args  [TYPE]  Logger LOGGER  [TYPE]  World w  [TYPE]  InterruptedException e  [TYPE]  int CYCLE_PERIOD 
[REPLACE]^car .AutomatedCar ( CYCLE_PERIOD , CYCLE_PERIOD , this )  ;^39^^^^^19^47^[REPLACE] car.drive (  ) ;^[METHOD] main [TYPE] void [PARAMETER] String[] args [CLASS] Main   [TYPE]  boolean false  true  [TYPE]  AutomatedCar car  [TYPE]  Gui gui  [TYPE]  String[] args  [TYPE]  Logger LOGGER  [TYPE]  World w  [TYPE]  InterruptedException e  [TYPE]  int CYCLE_PERIOD 
[REPLACE]^gui .Gui (  )  .drawWorld ( w ) ;^40^^^^^19^47^[REPLACE] gui.getCourseDisplay (  ) .drawWorld ( w ) ;^[METHOD] main [TYPE] void [PARAMETER] String[] args [CLASS] Main   [TYPE]  boolean false  true  [TYPE]  AutomatedCar car  [TYPE]  Gui gui  [TYPE]  String[] args  [TYPE]  Logger LOGGER  [TYPE]  World w  [TYPE]  InterruptedException e  [TYPE]  int CYCLE_PERIOD 
[ADD]^^40^41^^^^19^47^[ADD] gui.getCourseDisplay (  ) .drawWorld ( w ) ; Thread.sleep ( CYCLE_PERIOD ) ;^[METHOD] main [TYPE] void [PARAMETER] String[] args [CLASS] Main   [TYPE]  boolean false  true  [TYPE]  AutomatedCar car  [TYPE]  Gui gui  [TYPE]  String[] args  [TYPE]  Logger LOGGER  [TYPE]  World w  [TYPE]  InterruptedException e  [TYPE]  int CYCLE_PERIOD 
[REPLACE]^gui.Gui (  ) .drawWorld ( w ) ;^40^^^^^19^47^[REPLACE] gui.getCourseDisplay (  ) .drawWorld ( w ) ;^[METHOD] main [TYPE] void [PARAMETER] String[] args [CLASS] Main   [TYPE]  boolean false  true  [TYPE]  AutomatedCar car  [TYPE]  Gui gui  [TYPE]  String[] args  [TYPE]  Logger LOGGER  [TYPE]  World w  [TYPE]  InterruptedException e  [TYPE]  int CYCLE_PERIOD 
