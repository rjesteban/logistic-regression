/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package MultivariateLinearRegression;

import Jama.Matrix;

/**
 *
 * @author rjesteban
 */
public class GradientDescentValues {

    Matrix theta;
    Matrix costHistory;
    
    public GradientDescentValues(Matrix _theta, Matrix _costHistory){
        theta = _theta;
        costHistory = _costHistory;
    }

    public GradientDescentValues() {
    }
    
    public Matrix getTheta() {
        return theta;
    }

    public void setTheta(Matrix theta) {
        this.theta = theta;
    }

    public Matrix getCostHistory() {
        return costHistory;
    }

    public void setCostHistory(Matrix costHistory) {
        this.costHistory = costHistory;
    }
}
