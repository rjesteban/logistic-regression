/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package util;

import com.xeiam.xchart.Chart;
import com.xeiam.xchart.SeriesLineStyle;
import com.xeiam.xchart.SwingWrapper;

/**
 *
 * @author rjesteban
 */
public class CostPlot extends Plot{
    double[] J;
    
    public CostPlot(double[] _J){
        J = _J;
    }
    
    @Override
    public void run() {
        Chart c = new Chart(500,500);
        c.addSeries("cost", null, J).setLineStyle(SeriesLineStyle.NONE);
        c.setChartTitle("Cost x iterations plot");
        c.setXAxisTitle("Cost");
        c.setYAxisTitle("Plot");
        new SwingWrapper(c).displayChart("Cost Plot");
    }
    
}
