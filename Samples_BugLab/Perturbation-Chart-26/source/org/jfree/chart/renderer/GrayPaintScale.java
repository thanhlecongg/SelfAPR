[buglab_swap_variables]^if  ( upperBound >= lowerBound )  {^81^^^^^80^87^if  ( lowerBound >= upperBound )  {^[CLASS] GrayPaintScale  [METHOD] <init> [RETURN_TYPE] GrayPaintScale(double,double)   double lowerBound double upperBound [VARIABLES] double  lowerBound  upperBound  boolean  
[buglab_swap_variables]^double v = Math.max ( this.lowerBound, value ) ;^115^^^^^114^120^double v = Math.max ( value, this.lowerBound ) ;^[CLASS] GrayPaintScale  [METHOD] getPaint [RETURN_TYPE] Paint   double value [VARIABLES] double  lowerBound  upperBound  v  value  int  g  boolean  
[buglab_swap_variables]^double v = Math.max (  this.lowerBound ) ;^115^^^^^114^120^double v = Math.max ( value, this.lowerBound ) ;^[CLASS] GrayPaintScale  [METHOD] getPaint [RETURN_TYPE] Paint   double value [VARIABLES] double  lowerBound  upperBound  v  value  int  g  boolean  
[buglab_swap_variables]^double v = Math.max ( value ) ;^115^^^^^114^120^double v = Math.max ( value, this.lowerBound ) ;^[CLASS] GrayPaintScale  [METHOD] getPaint [RETURN_TYPE] Paint   double value [VARIABLES] double  lowerBound  upperBound  v  value  int  g  boolean  
[buglab_swap_variables]^v = Math.min ( this.upperBound, v ) ;^116^^^^^114^120^v = Math.min ( v, this.upperBound ) ;^[CLASS] GrayPaintScale  [METHOD] getPaint [RETURN_TYPE] Paint   double value [VARIABLES] double  lowerBound  upperBound  v  value  int  g  boolean  
[buglab_swap_variables]^v = Math.min (  this.upperBound ) ;^116^^^^^114^120^v = Math.min ( v, this.upperBound ) ;^[CLASS] GrayPaintScale  [METHOD] getPaint [RETURN_TYPE] Paint   double value [VARIABLES] double  lowerBound  upperBound  v  value  int  g  boolean  
[buglab_swap_variables]^v = Math.min ( v ) ;^116^^^^^114^120^v = Math.min ( v, this.upperBound ) ;^[CLASS] GrayPaintScale  [METHOD] getPaint [RETURN_TYPE] Paint   double value [VARIABLES] double  lowerBound  upperBound  v  value  int  g  boolean  
[buglab_swap_variables]^int g =  ( int )   (  ( this.lowerBound - value )  /  ( this.upperBound - this.lowerBound )  * 255.0 ) ;^117^118^^^^114^120^int g =  ( int )   (  ( value - this.lowerBound )  /  ( this.upperBound - this.lowerBound )  * 255.0 ) ;^[CLASS] GrayPaintScale  [METHOD] getPaint [RETURN_TYPE] Paint   double value [VARIABLES] double  lowerBound  upperBound  v  value  int  g  boolean  
[buglab_swap_variables]^int g =  ( int )   (  ( value - this.upperBound )  /  ( this.lowerBound - this.lowerBound )  * 255.0 ) ;^117^118^^^^114^120^int g =  ( int )   (  ( value - this.lowerBound )  /  ( this.upperBound - this.lowerBound )  * 255.0 ) ;^[CLASS] GrayPaintScale  [METHOD] getPaint [RETURN_TYPE] Paint   double value [VARIABLES] double  lowerBound  upperBound  v  value  int  g  boolean  
[buglab_swap_variables]^if  ( this.lowerBound != that.lowerBound.lowerBound )  {^143^^^^^135^150^if  ( this.lowerBound != that.lowerBound )  {^[CLASS] GrayPaintScale  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  boolean  double  lowerBound  upperBound  v  value  GrayPaintScale  that  
[buglab_swap_variables]^if  ( that.lowerBound != this.lowerBound )  {^143^^^^^135^150^if  ( this.lowerBound != that.lowerBound )  {^[CLASS] GrayPaintScale  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  boolean  double  lowerBound  upperBound  v  value  GrayPaintScale  that  
[buglab_swap_variables]^if  ( this.lowerBound != that )  {^143^^^^^135^150^if  ( this.lowerBound != that.lowerBound )  {^[CLASS] GrayPaintScale  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  boolean  double  lowerBound  upperBound  v  value  GrayPaintScale  that  
[buglab_swap_variables]^if  ( this.upperBound != that.upperBound.upperBound )  {^146^^^^^135^150^if  ( this.upperBound != that.upperBound )  {^[CLASS] GrayPaintScale  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  boolean  double  lowerBound  upperBound  v  value  GrayPaintScale  that  
[buglab_swap_variables]^if  ( that.upperBound != this.upperBound )  {^146^^^^^135^150^if  ( this.upperBound != that.upperBound )  {^[CLASS] GrayPaintScale  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  boolean  double  lowerBound  upperBound  v  value  GrayPaintScale  that  
[buglab_swap_variables]^if  ( this.upperBound != that )  {^146^^^^^135^150^if  ( this.upperBound != that.upperBound )  {^[CLASS] GrayPaintScale  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  boolean  double  lowerBound  upperBound  v  value  GrayPaintScale  that  