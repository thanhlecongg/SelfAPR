[buglab_swap_variables]^scale.setTickLabelFont ( new Font ( "Dialog", Font. 14 )  ) ;^78^^^^^63^93^scale.setTickLabelFont ( new Font ( "Dialog", Font.PLAIN, 14 )  ) ;^[CLASS] DialDemo4  [METHOD] <init> [RETURN_TYPE] String)   String title [VARIABLES] DefaultValueDataset  dataset  StandardDialFrame  dialFrame  JPanel  content  DialBackground  sdb  ChartPanel  cp1  GradientPaint  gp  String  title  DialPlot  plot  DialPointer  needle  JFreeChart  chart1  StandardDialScale  scale  boolean  JSlider  slider  
[buglab_swap_variables]^plot.addScale ( 0 ) ;^80^^^^^65^95^plot.addScale ( 0, scale ) ;^[CLASS] DialDemo4  [METHOD] <init> [RETURN_TYPE] String)   String title [VARIABLES] DefaultValueDataset  dataset  StandardDialFrame  dialFrame  JPanel  content  DialBackground  sdb  ChartPanel  cp1  GradientPaint  gp  String  title  DialPlot  plot  DialPointer  needle  JFreeChart  chart1  StandardDialScale  scale  boolean  JSlider  slider  
[buglab_swap_variables]^content.add (  BorderLayout.SOUTH ) ;^96^^^^^81^111^content.add ( this.slider, BorderLayout.SOUTH ) ;^[CLASS] DialDemo4  [METHOD] <init> [RETURN_TYPE] String)   String title [VARIABLES] DefaultValueDataset  dataset  StandardDialFrame  dialFrame  JPanel  content  DialBackground  sdb  ChartPanel  cp1  GradientPaint  gp  String  title  DialPlot  plot  DialPointer  needle  JFreeChart  chart1  StandardDialScale  scale  boolean  JSlider  slider  