[REPLACE]^setContentPane ( chartPanel ) ;^34^^^^^33^38^[REPLACE] super ( title ) ;^[METHOD] <init> [TYPE] String) [PARAMETER] String title [CLASS] LogAxisDemo1   [TYPE]  JPanel chartPanel  [TYPE]  boolean false  true  [TYPE]  String title 
[REPLACE]^JFreeChart chart = createChart ( createDataset (  )  ) ;^35^^^^^33^38^[REPLACE] JPanel chartPanel = createDemoPanel (  ) ;^[METHOD] <init> [TYPE] String) [PARAMETER] String title [CLASS] LogAxisDemo1   [TYPE]  JPanel chartPanel  [TYPE]  boolean false  true  [TYPE]  String title 
[REPLACE]^chartPanel.setPreferredSize ( new java.awt.Dimension ( 500L, 270 )  ) ;^36^^^^^33^38^[REPLACE] chartPanel.setPreferredSize ( new java.awt.Dimension ( 500, 270 )  ) ;^[METHOD] <init> [TYPE] String) [PARAMETER] String title [CLASS] LogAxisDemo1   [TYPE]  JPanel chartPanel  [TYPE]  boolean false  true  [TYPE]  String title 
[REPLACE]^series.add ( 1.0, 500.2 ) ;^37^^^^^33^38^[REPLACE] setContentPane ( chartPanel ) ;^[METHOD] <init> [TYPE] String) [PARAMETER] String title [CLASS] LogAxisDemo1   [TYPE]  JPanel chartPanel  [TYPE]  boolean false  true  [TYPE]  String title 
[ADD]^^41^42^43^^^40^57^[ADD] JFreeChart chart = ChartFactory.createScatterPlot ( "Log Axis Demo 1", "X",^[METHOD] createChart [TYPE] JFreeChart [PARAMETER] XYDataset dataset [CLASS] LogAxisDemo1   [TYPE]  boolean false  true  [TYPE]  XYPlot plot  [TYPE]  XYDataset dataset  [TYPE]  LogAxis xAxis  yAxis  [TYPE]  JFreeChart chart 
[REPLACE]^JPanel chartPanel = createDemoPanel (  ) ;^51^^^^^40^57^[REPLACE] XYPlot plot =  ( XYPlot )  chart.getPlot (  ) ;^[METHOD] createChart [TYPE] JFreeChart [PARAMETER] XYDataset dataset [CLASS] LogAxisDemo1   [TYPE]  boolean false  true  [TYPE]  XYPlot plot  [TYPE]  XYDataset dataset  [TYPE]  LogAxis xAxis  yAxis  [TYPE]  JFreeChart chart 
[REPLACE]^LogAxis yAxis = new LogAxis ( "Y" ) ;^52^^^^^40^57^[REPLACE] LogAxis xAxis = new LogAxis ( "X" ) ;^[METHOD] createChart [TYPE] JFreeChart [PARAMETER] XYDataset dataset [CLASS] LogAxisDemo1   [TYPE]  boolean false  true  [TYPE]  XYPlot plot  [TYPE]  XYDataset dataset  [TYPE]  LogAxis xAxis  yAxis  [TYPE]  JFreeChart chart 
[REPLACE]^LogAxis xAxis = new LogAxis ( "X" ) ;^53^^^^^40^57^[REPLACE] LogAxis yAxis = new LogAxis ( "Y" ) ;^[METHOD] createChart [TYPE] JFreeChart [PARAMETER] XYDataset dataset [CLASS] LogAxisDemo1   [TYPE]  boolean false  true  [TYPE]  XYPlot plot  [TYPE]  XYDataset dataset  [TYPE]  LogAxis xAxis  yAxis  [TYPE]  JFreeChart chart 
[REPLACE]^plot.setRangeAxis ( xAxis ) ;^54^^^^^40^57^[REPLACE] plot.setDomainAxis ( xAxis ) ;^[METHOD] createChart [TYPE] JFreeChart [PARAMETER] XYDataset dataset [CLASS] LogAxisDemo1   [TYPE]  boolean false  true  [TYPE]  XYPlot plot  [TYPE]  XYDataset dataset  [TYPE]  LogAxis xAxis  yAxis  [TYPE]  JFreeChart chart 
[REPLACE]^plot.setDomainAxis ( yAxis ) ;^55^^^^^40^57^[REPLACE] plot.setRangeAxis ( yAxis ) ;^[METHOD] createChart [TYPE] JFreeChart [PARAMETER] XYDataset dataset [CLASS] LogAxisDemo1   [TYPE]  boolean false  true  [TYPE]  XYPlot plot  [TYPE]  XYDataset dataset  [TYPE]  LogAxis xAxis  yAxis  [TYPE]  JFreeChart chart 
[REPLACE]^return new ChartPanel ( chart ) ;^56^^^^^40^57^[REPLACE] return chart;^[METHOD] createChart [TYPE] JFreeChart [PARAMETER] XYDataset dataset [CLASS] LogAxisDemo1   [TYPE]  boolean false  true  [TYPE]  XYPlot plot  [TYPE]  XYDataset dataset  [TYPE]  LogAxis xAxis  yAxis  [TYPE]  JFreeChart chart 
[REPLACE]^LogAxis yAxis = new LogAxis ( "Y" ) ;^65^^^^^64^76^[REPLACE] XYSeries series = new XYSeries ( "Random Data" ) ;^[METHOD] createDataset [TYPE] XYDataset [PARAMETER] [CLASS] LogAxisDemo1   [TYPE]  XYSeries series  [TYPE]  boolean false  true 
[ADD]^^65^^^^^64^76^[ADD] XYSeries series = new XYSeries ( "Random Data" ) ;^[METHOD] createDataset [TYPE] XYDataset [PARAMETER] [CLASS] LogAxisDemo1   [TYPE]  XYSeries series  [TYPE]  boolean false  true 
[REPLACE]^series.af ( 3.0d, 500.2 ) ;^66^^^^^64^76^[REPLACE] series.add ( 1.0, 500.2 ) ;^[METHOD] createDataset [TYPE] XYDataset [PARAMETER] [CLASS] LogAxisDemo1   [TYPE]  XYSeries series  [TYPE]  boolean false  true 
[REPLACE]^series.af ( 3.0d, 694.1 ) ;^67^^^^^64^76^[REPLACE] series.add ( 5.0, 694.1 ) ;^[METHOD] createDataset [TYPE] XYDataset [PARAMETER] [CLASS] LogAxisDemo1   [TYPE]  XYSeries series  [TYPE]  boolean false  true 
[ADD]^^67^^^^^64^76^[ADD] series.add ( 5.0, 694.1 ) ;^[METHOD] createDataset [TYPE] XYDataset [PARAMETER] [CLASS] LogAxisDemo1   [TYPE]  XYSeries series  [TYPE]  boolean false  true 
[REPLACE]^series.af ( 1.0d, 100.0 ) ;^68^^^^^64^76^[REPLACE] series.add ( 4.0, 100.0 ) ;^[METHOD] createDataset [TYPE] XYDataset [PARAMETER] [CLASS] LogAxisDemo1   [TYPE]  XYSeries series  [TYPE]  boolean false  true 
[REPLACE]^series.af ( 2.0d, 734.4 ) ;^69^^^^^64^76^[REPLACE] series.add ( 12.5, 734.4 ) ;^[METHOD] createDataset [TYPE] XYDataset [PARAMETER] [CLASS] LogAxisDemo1   [TYPE]  XYSeries series  [TYPE]  boolean false  true 
[REPLACE]^series.af ( 4.0d, 453.2 ) ;^70^^^^^64^76^[REPLACE] series.add ( 17.3, 453.2 ) ;^[METHOD] createDataset [TYPE] XYDataset [PARAMETER] [CLASS] LogAxisDemo1   [TYPE]  XYSeries series  [TYPE]  boolean false  true 
[REMOVE]^series.add ( 1.0, 500.2 ) ;^70^^^^^64^76^[REMOVE] ^[METHOD] createDataset [TYPE] XYDataset [PARAMETER] [CLASS] LogAxisDemo1   [TYPE]  XYSeries series  [TYPE]  boolean false  true 
[REPLACE]^series.af ( 0.0d, 500.2 ) ;^71^^^^^64^76^[REPLACE] series.add ( 21.2, 500.2 ) ;^[METHOD] createDataset [TYPE] XYDataset [PARAMETER] [CLASS] LogAxisDemo1   [TYPE]  XYSeries series  [TYPE]  boolean false  true 
[REMOVE]^series.add ( 1.0, 500.2 ) ;^71^^^^^64^76^[REMOVE] ^[METHOD] createDataset [TYPE] XYDataset [PARAMETER] [CLASS] LogAxisDemo1   [TYPE]  XYSeries series  [TYPE]  boolean false  true 
[REPLACE]^series.af ( 21.9D, 9005.5 ) ;^72^^^^^64^76^[REPLACE] series.add ( 21.9, 9005.5 ) ;^[METHOD] createDataset [TYPE] XYDataset [PARAMETER] [CLASS] LogAxisDemo1   [TYPE]  XYSeries series  [TYPE]  boolean false  true 
[ADD]^^72^73^^^^64^76^[ADD] series.add ( 21.9, 9005.5 ) ; series.add ( 25.6, 734.4 ) ;^[METHOD] createDataset [TYPE] XYDataset [PARAMETER] [CLASS] LogAxisDemo1   [TYPE]  XYSeries series  [TYPE]  boolean false  true 
[REPLACE]^series.af ( 4.0d, 734.4 ) ;^73^^^^^64^76^[REPLACE] series.add ( 25.6, 734.4 ) ;^[METHOD] createDataset [TYPE] XYDataset [PARAMETER] [CLASS] LogAxisDemo1   [TYPE]  XYSeries series  [TYPE]  boolean false  true 
[REPLACE]^series.af ( 3000.0D, 453.2 ) ;^74^^^^^64^76^[REPLACE] series.add ( 3000.0, 453.2 ) ;^[METHOD] createDataset [TYPE] XYDataset [PARAMETER] [CLASS] LogAxisDemo1   [TYPE]  XYSeries series  [TYPE]  boolean false  true 
[REPLACE]^return new ChartPanel ( chart ) ;^75^^^^^64^76^[REPLACE] return new XYSeriesCollection ( series ) ;^[METHOD] createDataset [TYPE] XYDataset [PARAMETER] [CLASS] LogAxisDemo1   [TYPE]  XYSeries series  [TYPE]  boolean false  true 
[REPLACE]^JPanel chartPanel = createDemoPanel (  ) ;^84^^^^^83^86^[REPLACE] JFreeChart chart = createChart ( createDataset (  )  ) ;^[METHOD] createDemoPanel [TYPE] JPanel [PARAMETER] [CLASS] LogAxisDemo1   [TYPE]  boolean false  true  [TYPE]  JFreeChart chart 
[REPLACE]^return chart;^85^^^^^83^86^[REPLACE] return new ChartPanel ( chart ) ;^[METHOD] createDemoPanel [TYPE] JPanel [PARAMETER] [CLASS] LogAxisDemo1   [TYPE]  boolean false  true  [TYPE]  JFreeChart chart 
[REPLACE]^XYSeries series = new XYSeries ( "Random Data" ) ;^95^^^^^93^100^[REPLACE] LogAxisDemo1 demo = new LogAxisDemo1 ( "Log Axis Demo 1" ) ;^[METHOD] main [TYPE] void [PARAMETER] String[] args [CLASS] LogAxisDemo1   [TYPE]  LogAxisDemo1 demo  [TYPE]  boolean false  true  [TYPE]  String[] args 
[ADD]^^95^^^^^93^100^[ADD] LogAxisDemo1 demo = new LogAxisDemo1 ( "Log Axis Demo 1" ) ;^[METHOD] main [TYPE] void [PARAMETER] String[] args [CLASS] LogAxisDemo1   [TYPE]  LogAxisDemo1 demo  [TYPE]  boolean false  true  [TYPE]  String[] args 
[REPLACE]^setContentPane ( chartPanel ) ;^96^^^^^93^100^[REPLACE] demo.pack (  ) ;^[METHOD] main [TYPE] void [PARAMETER] String[] args [CLASS] LogAxisDemo1   [TYPE]  LogAxisDemo1 demo  [TYPE]  boolean false  true  [TYPE]  String[] args 
[REPLACE]^demo   ;^97^^^^^93^100^[REPLACE] RefineryUtilities.centerFrameOnScreen ( demo ) ;^[METHOD] main [TYPE] void [PARAMETER] String[] args [CLASS] LogAxisDemo1   [TYPE]  LogAxisDemo1 demo  [TYPE]  boolean false  true  [TYPE]  String[] args 
[REPLACE]^demo.setVisible ( false ) ;^98^^^^^93^100^[REPLACE] demo.setVisible ( true ) ;^[METHOD] main [TYPE] void [PARAMETER] String[] args [CLASS] LogAxisDemo1   [TYPE]  LogAxisDemo1 demo  [TYPE]  boolean false  true  [TYPE]  String[] args 