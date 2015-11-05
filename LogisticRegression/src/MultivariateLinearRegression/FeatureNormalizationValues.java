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
public class FeatureNormalizationValues {

    Matrix X;
    Matrix mu;
    Matrix sigma;
    
    public FeatureNormalizationValues(Matrix _X, Matrix _mu, Matrix _sigma){
        X = _X;
        mu = _mu;
        sigma = _sigma;
    }

    public Matrix getX() {
        return X;
    }

    public void setX(Matrix X) {
        this.X = X;
    }

    public Matrix getMu() {
        return mu;
    }

    public void setMu(Matrix mu) {
        this.mu = mu;
    }

    public Matrix getSigma() {
        return sigma;
    }

    public void setSigma(Matrix sigma) {
        this.sigma = sigma;
    }
}