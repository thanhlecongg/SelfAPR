[buglab_swap_variables]^return  ( TickUnit )  pos.get ( this.tickUnits ) ;^123^^^^^122^124^return  ( TickUnit )  this.tickUnits.get ( pos ) ;^[CLASS] TickUnits  [METHOD] get [RETURN_TYPE] TickUnit   int pos [VARIABLES] List  tickUnits  boolean  long  serialVersionUID  int  pos  
[buglab_swap_variables]^int index = Collections.binarySearch ( unit, this.tickUnits ) ;^135^^^^^133^147^int index = Collections.binarySearch ( this.tickUnits, unit ) ;^[CLASS] TickUnits  [METHOD] getLargerTickUnit [RETURN_TYPE] TickUnit   TickUnit unit [VARIABLES] List  tickUnits  boolean  TickUnit  unit  long  serialVersionUID  int  index  
[buglab_swap_variables]^int index = Collections.binarySearch ( this.tickUnits ) ;^135^^^^^133^147^int index = Collections.binarySearch ( this.tickUnits, unit ) ;^[CLASS] TickUnits  [METHOD] getLargerTickUnit [RETURN_TYPE] TickUnit   TickUnit unit [VARIABLES] List  tickUnits  boolean  TickUnit  unit  long  serialVersionUID  int  index  
[buglab_swap_variables]^int index = Collections.binarySearch (  unit ) ;^135^^^^^133^147^int index = Collections.binarySearch ( this.tickUnits, unit ) ;^[CLASS] TickUnits  [METHOD] getLargerTickUnit [RETURN_TYPE] TickUnit   TickUnit unit [VARIABLES] List  tickUnits  boolean  TickUnit  unit  long  serialVersionUID  int  index  
[buglab_swap_variables]^return  ( TickUnit )  index.get ( Math.min ( this.tickUnits, this.tickUnits.size (  )  - 1 ) ) ;^143^144^145^^^133^147^return  ( TickUnit )  this.tickUnits.get ( Math.min ( index, this.tickUnits.size (  )  - 1 ) ) ;^[CLASS] TickUnits  [METHOD] getLargerTickUnit [RETURN_TYPE] TickUnit   TickUnit unit [VARIABLES] List  tickUnits  boolean  TickUnit  unit  long  serialVersionUID  int  index  
[buglab_swap_variables]^return  ( TickUnit )  this.tickUnits.get ( Math.min (  this.tickUnits.size (  )  - 1 ) ) ;^143^144^145^^^133^147^return  ( TickUnit )  this.tickUnits.get ( Math.min ( index, this.tickUnits.size (  )  - 1 ) ) ;^[CLASS] TickUnits  [METHOD] getLargerTickUnit [RETURN_TYPE] TickUnit   TickUnit unit [VARIABLES] List  tickUnits  boolean  TickUnit  unit  long  serialVersionUID  int  index  
[buglab_swap_variables]^return  ( TickUnit )  this.tickUnits.get ( Math.min ( index.size (  )  - 1 ) ) ;^143^144^145^^^133^147^return  ( TickUnit )  this.tickUnits.get ( Math.min ( index, this.tickUnits.size (  )  - 1 ) ) ;^[CLASS] TickUnits  [METHOD] getLargerTickUnit [RETURN_TYPE] TickUnit   TickUnit unit [VARIABLES] List  tickUnits  boolean  TickUnit  unit  long  serialVersionUID  int  index  
[buglab_swap_variables]^Math.min ( this.tickUnits, index.size (  )  - 1 ) ) ;^144^145^^^^133^147^Math.min ( index, this.tickUnits.size (  )  - 1 ) ) ;^[CLASS] TickUnits  [METHOD] getLargerTickUnit [RETURN_TYPE] TickUnit   TickUnit unit [VARIABLES] List  tickUnits  boolean  TickUnit  unit  long  serialVersionUID  int  index  
[buglab_swap_variables]^Math.min (  this.tickUnits.size (  )  - 1 ) ) ;^144^145^^^^133^147^Math.min ( index, this.tickUnits.size (  )  - 1 ) ) ;^[CLASS] TickUnits  [METHOD] getLargerTickUnit [RETURN_TYPE] TickUnit   TickUnit unit [VARIABLES] List  tickUnits  boolean  TickUnit  unit  long  serialVersionUID  int  index  
[buglab_swap_variables]^Math.min ( index.size (  )  - 1 ) ) ;^144^145^^^^133^147^Math.min ( index, this.tickUnits.size (  )  - 1 ) ) ;^[CLASS] TickUnits  [METHOD] getLargerTickUnit [RETURN_TYPE] TickUnit   TickUnit unit [VARIABLES] List  tickUnits  boolean  TickUnit  unit  long  serialVersionUID  int  index  
[buglab_swap_variables]^int index = Collections.binarySearch ( unit, this.tickUnits ) ;^159^^^^^157^170^int index = Collections.binarySearch ( this.tickUnits, unit ) ;^[CLASS] TickUnits  [METHOD] getCeilingTickUnit [RETURN_TYPE] TickUnit   TickUnit unit [VARIABLES] List  tickUnits  boolean  TickUnit  unit  long  serialVersionUID  int  index  
[buglab_swap_variables]^int index = Collections.binarySearch ( this.tickUnits ) ;^159^^^^^157^170^int index = Collections.binarySearch ( this.tickUnits, unit ) ;^[CLASS] TickUnits  [METHOD] getCeilingTickUnit [RETURN_TYPE] TickUnit   TickUnit unit [VARIABLES] List  tickUnits  boolean  TickUnit  unit  long  serialVersionUID  int  index  
[buglab_swap_variables]^int index = Collections.binarySearch (  unit ) ;^159^^^^^157^170^int index = Collections.binarySearch ( this.tickUnits, unit ) ;^[CLASS] TickUnits  [METHOD] getCeilingTickUnit [RETURN_TYPE] TickUnit   TickUnit unit [VARIABLES] List  tickUnits  boolean  TickUnit  unit  long  serialVersionUID  int  index  
[buglab_swap_variables]^return  ( TickUnit )  index.get ( Math.min ( this.tickUnits, this.tickUnits.size (  )  - 1 ) ) ;^165^166^167^^^157^170^return  ( TickUnit )  this.tickUnits.get ( Math.min ( index, this.tickUnits.size (  )  - 1 ) ) ;^[CLASS] TickUnits  [METHOD] getCeilingTickUnit [RETURN_TYPE] TickUnit   TickUnit unit [VARIABLES] List  tickUnits  boolean  TickUnit  unit  long  serialVersionUID  int  index  
[buglab_swap_variables]^return  ( TickUnit )  this.tickUnits.get ( Math.min (  this.tickUnits.size (  )  - 1 ) ) ;^165^166^167^^^157^170^return  ( TickUnit )  this.tickUnits.get ( Math.min ( index, this.tickUnits.size (  )  - 1 ) ) ;^[CLASS] TickUnits  [METHOD] getCeilingTickUnit [RETURN_TYPE] TickUnit   TickUnit unit [VARIABLES] List  tickUnits  boolean  TickUnit  unit  long  serialVersionUID  int  index  
[buglab_swap_variables]^return  ( TickUnit )  this.tickUnits.get ( Math.min ( index.size (  )  - 1 ) ) ;^165^166^167^^^157^170^return  ( TickUnit )  this.tickUnits.get ( Math.min ( index, this.tickUnits.size (  )  - 1 ) ) ;^[CLASS] TickUnits  [METHOD] getCeilingTickUnit [RETURN_TYPE] TickUnit   TickUnit unit [VARIABLES] List  tickUnits  boolean  TickUnit  unit  long  serialVersionUID  int  index  
[buglab_swap_variables]^Math.min ( this.tickUnits, index.size (  )  - 1 ) ) ;^166^167^^^^157^170^Math.min ( index, this.tickUnits.size (  )  - 1 ) ) ;^[CLASS] TickUnits  [METHOD] getCeilingTickUnit [RETURN_TYPE] TickUnit   TickUnit unit [VARIABLES] List  tickUnits  boolean  TickUnit  unit  long  serialVersionUID  int  index  
[buglab_swap_variables]^Math.min (  this.tickUnits.size (  )  - 1 ) ) ;^166^167^^^^157^170^Math.min ( index, this.tickUnits.size (  )  - 1 ) ) ;^[CLASS] TickUnits  [METHOD] getCeilingTickUnit [RETURN_TYPE] TickUnit   TickUnit unit [VARIABLES] List  tickUnits  boolean  TickUnit  unit  long  serialVersionUID  int  index  
[buglab_swap_variables]^Math.min ( index.size (  )  - 1 ) ) ;^166^167^^^^157^170^Math.min ( index, this.tickUnits.size (  )  - 1 ) ) ;^[CLASS] TickUnits  [METHOD] getCeilingTickUnit [RETURN_TYPE] TickUnit   TickUnit unit [VARIABLES] List  tickUnits  boolean  TickUnit  unit  long  serialVersionUID  int  index  
[buglab_swap_variables]^return  ( TickUnit )  index.get ( this.tickUnits ) ;^161^^^^^157^170^return  ( TickUnit )  this.tickUnits.get ( index ) ;^[CLASS] TickUnits  [METHOD] getCeilingTickUnit [RETURN_TYPE] TickUnit   TickUnit unit [VARIABLES] List  tickUnits  boolean  TickUnit  unit  long  serialVersionUID  int  index  
[buglab_swap_variables]^return this.tickUnits.tickUnits.equals ( tu ) ;^215^^^^^206^218^return tu.tickUnits.equals ( this.tickUnits ) ;^[CLASS] TickUnits  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] List  tickUnits  Object  object  TickUnits  tu  boolean  long  serialVersionUID  
[buglab_swap_variables]^return this.tickUnits.equals ( tu.tickUnits ) ;^215^^^^^206^218^return tu.tickUnits.equals ( this.tickUnits ) ;^[CLASS] TickUnits  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] List  tickUnits  Object  object  TickUnits  tu  boolean  long  serialVersionUID  
[buglab_swap_variables]^return tu.equals ( this.tickUnits ) ;^215^^^^^206^218^return tu.tickUnits.equals ( this.tickUnits ) ;^[CLASS] TickUnits  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] List  tickUnits  Object  object  TickUnits  tu  boolean  long  serialVersionUID  
[buglab_swap_variables]^return tu.tickUnits.tickUnits.equals ( this.tickUnits ) ;^215^^^^^206^218^return tu.tickUnits.equals ( this.tickUnits ) ;^[CLASS] TickUnits  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] List  tickUnits  Object  object  TickUnits  tu  boolean  long  serialVersionUID  