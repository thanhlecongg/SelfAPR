[buglab_swap_variables]^return solve ( max, min, UnivariateRealSolverUtils.midpoint ( min, max )  ) ;^61^^^^^59^62^return solve ( min, max, UnivariateRealSolverUtils.midpoint ( min, max )  ) ;^[CLASS] NewtonSolver  [METHOD] solve [RETURN_TYPE] double   double min double max [VARIABLES] UnivariateRealFunction  derivative  boolean  long  serialVersionUID  double  max  min  
[buglab_swap_variables]^return solve (  max, UnivariateRealSolverUtils.midpoint ( min, max )  ) ;^61^^^^^59^62^return solve ( min, max, UnivariateRealSolverUtils.midpoint ( min, max )  ) ;^[CLASS] NewtonSolver  [METHOD] solve [RETURN_TYPE] double   double min double max [VARIABLES] UnivariateRealFunction  derivative  boolean  long  serialVersionUID  double  max  min  
[buglab_swap_variables]^return solve ( min,  UnivariateRealSolverUtils.midpoint ( min, max )  ) ;^61^^^^^59^62^return solve ( min, max, UnivariateRealSolverUtils.midpoint ( min, max )  ) ;^[CLASS] NewtonSolver  [METHOD] solve [RETURN_TYPE] double   double min double max [VARIABLES] UnivariateRealFunction  derivative  boolean  long  serialVersionUID  double  max  min  
[buglab_swap_variables]^verifySequence ( max, startValue, min ) ;^80^^^^^76^100^verifySequence ( min, startValue, max ) ;^[CLASS] NewtonSolver  [METHOD] solve [RETURN_TYPE] double   double min double max double startValue [VARIABLES] UnivariateRealFunction  derivative  boolean  long  serialVersionUID  double  max  min  startValue  x0  x1  int  i  
[buglab_swap_variables]^verifySequence (  startValue, max ) ;^80^^^^^76^100^verifySequence ( min, startValue, max ) ;^[CLASS] NewtonSolver  [METHOD] solve [RETURN_TYPE] double   double min double max double startValue [VARIABLES] UnivariateRealFunction  derivative  boolean  long  serialVersionUID  double  max  min  startValue  x0  x1  int  i  
[buglab_swap_variables]^verifySequence ( startValue, min, max ) ;^80^^^^^76^100^verifySequence ( min, startValue, max ) ;^[CLASS] NewtonSolver  [METHOD] solve [RETURN_TYPE] double   double min double max double startValue [VARIABLES] UnivariateRealFunction  derivative  boolean  long  serialVersionUID  double  max  min  startValue  x0  x1  int  i  
[buglab_swap_variables]^verifySequence ( min,  max ) ;^80^^^^^76^100^verifySequence ( min, startValue, max ) ;^[CLASS] NewtonSolver  [METHOD] solve [RETURN_TYPE] double   double min double max double startValue [VARIABLES] UnivariateRealFunction  derivative  boolean  long  serialVersionUID  double  max  min  startValue  x0  x1  int  i  
[buglab_swap_variables]^verifySequence ( min, startValue ) ;^80^^^^^76^100^verifySequence ( min, startValue, max ) ;^[CLASS] NewtonSolver  [METHOD] solve [RETURN_TYPE] double   double min double max double startValue [VARIABLES] UnivariateRealFunction  derivative  boolean  long  serialVersionUID  double  max  min  startValue  x0  x1  int  i  
[buglab_swap_variables]^if  ( Math.abs ( x0 - x1 )  <= absoluteAccuracy )  {^88^^^^^76^100^if  ( Math.abs ( x1 - x0 )  <= absoluteAccuracy )  {^[CLASS] NewtonSolver  [METHOD] solve [RETURN_TYPE] double   double min double max double startValue [VARIABLES] UnivariateRealFunction  derivative  boolean  long  serialVersionUID  double  max  min  startValue  x0  x1  int  i  
[buglab_swap_variables]^if  ( Math.abs ( absoluteAccuracy - x0 )  <= x1 )  {^88^^^^^76^100^if  ( Math.abs ( x1 - x0 )  <= absoluteAccuracy )  {^[CLASS] NewtonSolver  [METHOD] solve [RETURN_TYPE] double   double min double max double startValue [VARIABLES] UnivariateRealFunction  derivative  boolean  long  serialVersionUID  double  max  min  startValue  x0  x1  int  i  
[buglab_swap_variables]^setResult ( i, x1 ) ;^90^^^^^76^100^setResult ( x1, i ) ;^[CLASS] NewtonSolver  [METHOD] solve [RETURN_TYPE] double   double min double max double startValue [VARIABLES] UnivariateRealFunction  derivative  boolean  long  serialVersionUID  double  max  min  startValue  x0  x1  int  i  
[buglab_swap_variables]^setResult (  i ) ;^90^^^^^76^100^setResult ( x1, i ) ;^[CLASS] NewtonSolver  [METHOD] solve [RETURN_TYPE] double   double min double max double startValue [VARIABLES] UnivariateRealFunction  derivative  boolean  long  serialVersionUID  double  max  min  startValue  x0  x1  int  i  
[buglab_swap_variables]^setResult ( x1 ) ;^90^^^^^76^100^setResult ( x1, i ) ;^[CLASS] NewtonSolver  [METHOD] solve [RETURN_TYPE] double   double min double max double startValue [VARIABLES] UnivariateRealFunction  derivative  boolean  long  serialVersionUID  double  max  min  startValue  x0  x1  int  i  
[buglab_swap_variables]^x1 = f -  ( x0.value ( x0 )  / derivative.value ( x0 )  ) ;^87^^^^^76^100^x1 = x0 -  ( f.value ( x0 )  / derivative.value ( x0 )  ) ;^[CLASS] NewtonSolver  [METHOD] solve [RETURN_TYPE] double   double min double max double startValue [VARIABLES] UnivariateRealFunction  derivative  boolean  long  serialVersionUID  double  max  min  startValue  x0  x1  int  i  
[buglab_swap_variables]^x1 = x0 -  ( derivative.value ( x0 )  / f.value ( x0 )  ) ;^87^^^^^76^100^x1 = x0 -  ( f.value ( x0 )  / derivative.value ( x0 )  ) ;^[CLASS] NewtonSolver  [METHOD] solve [RETURN_TYPE] double   double min double max double startValue [VARIABLES] UnivariateRealFunction  derivative  boolean  long  serialVersionUID  double  max  min  startValue  x0  x1  int  i  
[buglab_swap_variables]^x1 = derivative -  ( f.value ( x0 )  / x0.value ( x0 )  ) ;^87^^^^^76^100^x1 = x0 -  ( f.value ( x0 )  / derivative.value ( x0 )  ) ;^[CLASS] NewtonSolver  [METHOD] solve [RETURN_TYPE] double   double min double max double startValue [VARIABLES] UnivariateRealFunction  derivative  boolean  long  serialVersionUID  double  max  min  startValue  x0  x1  int  i  