/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package util;

import Jama.Matrix;
import com.xeiam.xchart.Chart;
import com.xeiam.xchart.SeriesLineStyle;
import com.xeiam.xchart.SwingWrapper;
import java.util.ArrayList;

/**
 *
 * @author rjesteban
 */
public class DataPlot extends Thread{
    private Matrix X;
    private Matrix y;
    Chart chart;
    String x_label;
    String y_label;
    public DataPlot(Matrix _X, Matrix _y, String x_label, String y_label){
        X = _X;
        y = _y;
        this.x_label = x_label;
        this.y_label = y_label;
        chart = new Chart(500,500);
    }
    
    @Override
    public void run(){
        chart.setXAxisTitle(x_label);
        chart.setYAxisTitle(y_label);
        
        ArrayList<Double> yesX = new ArrayList<Double>();
        ArrayList<Double> yesY = new ArrayList<Double>();
        ArrayList<Double> noX = new ArrayList<Double>();
        ArrayList<Double> noY = new ArrayList<Double>();
        
        
            for(int r=0; r<X.getRowDimension(); r++){
                if(y.get(r, 0)==1){
                    yesX.add(X.get(r, 0));
                    yesY.add(X.get(r, 1));
                }
                else{
                    noX.add(X.get(r, 0));
                    noY.add(X.get(r, 1));
                }
            }
        
        chart.addSeries("positive", yesX, yesY).setLineStyle(SeriesLineStyle.NONE);
        chart.addSeries("negative", noX, noY).setLineStyle(SeriesLineStyle.NONE);
        new SwingWrapper(chart).displayChart("Data Plot");
        
    }
    
    public void addHypothesis(double[] x, double[] y){
        chart.addSeries("hypothesis", x, y);
    }
}
